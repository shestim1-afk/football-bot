import os
import sys
import time
import logging
import httpx
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# --- Config from env ---
missing = []
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    missing.append("TELEGRAM_BOT_TOKEN")

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_CHAT_ID:
    missing.append("TELEGRAM_CHAT_ID")

_raw_keys = os.environ.get("RAPIDAPI_KEY", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not API_KEYS:
    missing.append("RAPIDAPI_KEY")

if missing:
    log.error(f"MISSING ENV VARIABLES: {', '.join(missing)}")
    log.error("Please add them in Railway > Variables tab")
    sys.exit(1)

API_BASE = "https://v3.football.api-sports.io"
TELEGRAM_API = "https://api.telegram.org"

LEAGUE_IDS = {
    39: "Premier League", 140: "La Liga", 78: "Bundesliga", 79: "2. Bundesliga",
    135: "Serie A", 61: "Ligue 1", 2: "Champions League", 3: "Europa League",
    848: "Conference League", 357: "First League (Bulgaria)", 94: "Primeira Liga",
    88: "Eredivisie", 203: "Super Lig", 169: "Austrian Bundesliga",
    283: "SuperLiga (Serbia)", 210: "HNL (Croatia)", 345: "Czech First League",
    119: "Danish Superliga", 137: "Veikkausliiga (Finland)", 191: "NB I (Hungary)",
}

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}

# Active monitoring window: 14:00-23:00 Bulgaria local time
# Uses actual Sofia timezone (handles EET/EEST DST automatically)
BULGARIA_TZ = ZoneInfo("Europe/Sofia")
ACTIVE_HOUR_START = 14  # 14:00 local
ACTIVE_HOUR_END = 23    # 23:00 local

# v9.5: 20'-80' window (strict — no late tracking beyond 80')
MINUTE_MIN = 20
MINUTE_MAX = 80

# Fast SOT polling window: once SOT reaches 2+, we poll faster for a
# limited time.  After the window expires, polling reverts to base rate.
FAST_SOT_WINDOW = 5 * 60  # 300 seconds

# Max fixture IDs per batched request (API-Football limit for /fixtures?ids=...)
BATCH_SIZE_LIMIT = 20

# --- State ---
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []
rate_limited_until: float = 0.0

# API-Football reports the daily quota at the subscription level.
# Do NOT assume that multiple API keys multiply the daily allowance.
quota_remaining: int | None = None
quota_limit: int | None = None
minute_remaining: int | None = None
minute_limit: int | None = None

# --- v9.5: Round-robin API key management ---
# Each key has health state: rate-limited until timestamp, or auth-failed.
key_health: list[dict] = []
rr_index: int = 0  # round-robin counter

# --- v9.5.4: Per-team signal limit tracking ---
# Key: (fixture_id, team_id) -> {"count": N, "goals_at_last_signal": G}
# 1st signal: always sent (SOT >= 3, the gold signal)
# 2nd signal: sent if +1 SOT (guaranteed by classify_signal dedup)
# 3rd+ signal: only if +2 SOT jump AND 0 goals scored since LAST signal
signaled_teams: dict[tuple[int, int], dict] = {}
# Keep fixture-level set for backward compat in logs/cleanup
signaled_fixtures: set[int] = set()

# --- Adaptive polling state ---
last_discovery_time: float = 0.0
last_stats_check: dict[int, float] = {}   # fixture_id -> timestamp of last stats fetch
fast_monitored: set[int] = set()         # fixture IDs currently monitored
fast_priority: dict[int, int] = {}       # fixture_id -> rank score (discovery-time)
cached_fixtures: list[dict] = []        # last discovery result (reused for filtering only)

# --- Fast SOT window state ---
fast_sot_until: dict[int, float] = {}


# ============================================================
# API KEY ROUND-ROBIN (v9.5)
# ============================================================

def init_key_health():
    global key_health
    key_health = [
        {"rate_limited_until": 0.0, "auth_failed": False}
        for _ in API_KEYS
    ]


def pick_key() -> str | None:
    """Round-robin through healthy keys. Returns None if all unhealthy."""
    global rr_index
    if not key_health:
        init_key_health()

    tried = 0
    while tried < len(API_KEYS):
        idx = rr_index % len(API_KEYS)
        health = key_health[idx]
        rr_index += 1

        if health["auth_failed"]:
            tried += 1
            continue
        if time.time() < health["rate_limited_until"]:
            tried += 1
            continue

        return API_KEYS[idx]

    return None


def mark_key_auth_failed(key: str):
    global key_health
    if key in API_KEYS:
        idx = API_KEYS.index(key)
        if idx < len(key_health):
            key_health[idx]["auth_failed"] = True
            log.warning(f"  Key {key[:8]}... marked as AUTH FAILED")


def mark_key_rate_limited(key: str, duration: float = 120.0):
    global key_health
    if key in API_KEYS:
        idx = API_KEYS.index(key)
        if idx < len(key_health):
            key_health[idx]["rate_limited_until"] = time.time() + duration
            log.warning(f"  Key {key[:8]}... rate-limited for {int(duration)}s")


def healthy_key_count() -> int:
    if not key_health:
        return len(API_KEYS)
    now = time.time()
    count = 0
    for h in key_health:
        if not h["auth_failed"] and now >= h["rate_limited_until"]:
            count += 1
    return count


# ============================================================
# QUOTA TRACKING
# ============================================================

def update_quota(resp: httpx.Response):
    global quota_remaining, quota_limit
    global minute_remaining, minute_limit

    try:
        value = resp.headers.get("x-ratelimit-requests-remaining")
        if value:
            quota_remaining = int(value)

        value = resp.headers.get("x-ratelimit-requests-limit")
        if value:
            quota_limit = int(value)

        value = resp.headers.get("x-ratelimit-remaining")
        if value:
            minute_remaining = int(value)

        value = resp.headers.get("x-ratelimit-limit")
        if value:
            minute_limit = int(value)

    except (ValueError, TypeError):
        pass


# ============================================================
# API HELPERS
# ============================================================

def api_get(client: httpx.Client, endpoint: str, params: dict = None) -> dict:
    global request_count, rate_limited_until

    if quota_remaining is not None and quota_remaining <= 0:
        raise Exception("Daily API quota exhausted")

    # Round-robin through healthy keys
    attempts = 0
    max_attempts = len(API_KEYS) * 2

    while attempts < max_attempts:
        key = pick_key()
        if key is None:
            raise Exception("All API keys are unhealthy (auth failed or rate limited)")

        request_count += 1

        log.info(
            f"  API GET {endpoint} "
            f"(key={key[:8]}..., req #{request_count}, "
            f"remaining={quota_remaining})"
        )

        resp = client.get(
            f"{API_BASE}{endpoint}",
            params=params,
            headers={"x-apisports-key": key},
        )

        update_quota(resp)

        if resp.status_code == 429:
            # Rate limit: mark this key and try another immediately
            mark_key_rate_limited(key, 120.0)
            attempts += 1
            # If NO healthy keys remain, set global backoff
            if healthy_key_count() == 0:
                rate_limited_until = time.time() + 120
                raise Exception("Rate limited (429) on all keys, backing off 120s")
            continue

        if resp.status_code in (401, 403):
            mark_key_auth_failed(key)
            attempts += 1
            if healthy_key_count() == 0:
                raise Exception(
                    f"All {len(API_KEYS)} key(s) failed with auth errors"
                )
            continue

        resp.raise_for_status()
        return resp.json()

    raise Exception("All API keys failed after max attempts")


def send_telegram(client: httpx.Client, text: str) -> bool:
    try:
        resp = client.post(
            f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


# ============================================================
# CANDIDATE RANKING (v9.5.1 — pressure-first priority)
# ============================================================
#
# 1. SOT / attacking pressure     (dominant factor)
# 2. xG                          (important secondary)
# 3. Game minute                 (prefer 45-70')
# 4. Scoreline / game state      (small 0-0 bonus, no penalty for goals)
# 5. Diversification             (modest unsignaled bonus)
#
# Key principle: rank by actual pressure, not scoreline.
# A 1-0 match with SOT=4 should outrank a 0-0 match with SOT=0.


def rank_candidate(fixture: dict, team_id: int) -> int:
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    fid = fixture["fixture"]["id"]

    # --- 1. SOT / attacking pressure (DOMINANT) ---
    prev = team_state.get((fid, team_id))
    prev_sot = prev.get("last_sot", 0) if prev else 0

    # Scaled heavily so it always dominates other factors.
    # SOT=0 unknown=10, SOT=1=100, SOT=2=300, SOT=3=600,
    # SOT=4=900, SOT=5=1200, SOT=6+=1500
    if prev_sot >= 6:
        score += 1500
    elif prev_sot >= 5:
        score += 1200
    elif prev_sot >= 4:
        score += 900
    elif prev_sot >= 3:
        score += 600
    elif prev_sot >= 2:
        score += 300
    elif prev_sot >= 1:
        score += 100
    else:
        score += 10  # unknown — worth a first check

    # --- 2. xG (important secondary — up to ~150 points) ---
    if prev and prev.get("last_xg"):
        try:
            xg = float(prev["last_xg"])
            score += int(xg * 100)
        except (ValueError, TypeError):
            pass

    # --- 3. Game minute (prefer 45-70') ---
    if 45 <= minute <= 70:
        score += 30
    elif 30 <= minute <= 80:
        score += 15
    elif 20 <= minute <= 29:
        score += 5

    # --- 4. Scoreline / game state (small bonus only) ---
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if tg == 0 and og == 0:
        score += 40
    elif abs(tg - og) <= 1:
        score += 20
    # No penalty for any scoreline — all scorelines eligible

    # --- 5. Diversification (modest unsignaled bonus) ---
    if fid not in signaled_fixtures:
        score += 50

    return score


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    if quota_remaining is None:
        return "UNKNOWN"
    if quota_remaining <= 0:
        return "STOP"
    if quota_remaining <= 10:
        return "EMERGENCY"
    if quota_remaining <= 25:
        return "STRICT"
    if quota_remaining <= 50:
        return "CAREFUL"
    return "NORMAL"


def get_max_fast_monitored() -> int:
    """v9.5: Dynamic max monitored fixtures.

    With batched API requests (up to 20 fixtures per request),
    monitoring more fixtures costs the same as monitoring few.
    The real cost driver is polling frequency, not fixture count.

    Reserve ~10 credits for discovery + safety.
    """
    if quota_remaining is None:
        return 10

    available = quota_remaining - 10
    if available <= 0:
        return 0

    # Conservative: ~5 batched requests per fixture lifecycle.
    # With batching, this is very conservative since one request
    # covers up to 20 fixtures simultaneously.
    estimated_per_fixture = 4
    max_by_quota = available // estimated_per_fixture

    # Floor: at least 1 if we have any budget.  Cap: batch size limit.
    return min(BATCH_SIZE_LIMIT, max(1, max_by_quota))


def get_discovery_interval(
    budget_mode: str,
    has_tracked_live: bool,
    has_candidates: bool,
) -> int:
    """Seconds between live-fixture discovery calls."""

    if not has_tracked_live:
        return 1800

    if not has_candidates:
        return {
            "NORMAL": 900,
            "CAREFUL": 1200,
            "STRICT": 1800,
            "EMERGENCY": 2400,
            "UNKNOWN": 1800,
        }.get(budget_mode, 1800)

    return {
        "NORMAL": 1800,
        "CAREFUL": 2400,
        "STRICT": 3000,
        "EMERGENCY": 3600,
        "UNKNOWN": 1800,
    }.get(budget_mode, 1800)


def get_stats_interval(budget_mode: str) -> int:
    """Base interval for statistics batched requests.

    v9.5: With 10+ fixtures monitored, each batched request checks
    multiple fixtures simultaneously. The base interval is the minimum
    time between ANY stats request.
    """
    if budget_mode == "NORMAL":
        return 240
    if budget_mode == "CAREFUL":
        return 300
    if budget_mode == "STRICT":
        return 420
    if budget_mode == "EMERGENCY":
        return 600
    return 600


# ============================================================
# FAST SOT WINDOW
# ============================================================

def get_fixture_best_sot(fid: int) -> int:
    best = 0
    for (f, t), state in team_state.items():
        if f == fid:
            sot = state.get("last_sot", 0)
            if sot > best:
                best = sot
    return best


def activate_fast_sot(fid: int):
    fast_sot_until[fid] = time.time() + FAST_SOT_WINDOW
    log.info(
        f"  Fast SOT window ACTIVATED for fixture {fid} "
        f"(expires in {FAST_SOT_WINDOW}s)"
    )


def is_fast_sot_active(fid: int) -> bool:
    until = fast_sot_until.get(fid)
    if until is None:
        return False
    if time.time() >= until:
        del fast_sot_until[fid]
        return False
    return True


def expire_fast_sot(fid: int):
    fast_sot_until.pop(fid, None)


# ============================================================
# SIGNAL CLASSIFICATION — SOT >= 3 is mandatory
# ============================================================

TIER_ORDER = ["PRESSURE", "STRONG", "VERY STRONG"]


def bump_tier(tier: str) -> str:
    idx = TIER_ORDER.index(tier)
    if idx < len(TIER_ORDER) - 1:
        return TIER_ORDER[idx + 1]
    return tier


def classify_signal(sot: int, state: dict | None, current_minute: int) -> tuple[str | None, str, float]:
    # SOT < 3: NO signal (mandatory criterion)
    if sot < 3:
        return None, "", 0.0

    # Deduplication: only send when SOT increases
    last_sot = state["last_sot"] if state else 0
    if sot <= last_sot:
        return None, "", 0.0

    if sot >= 5:
        tier = "VERY STRONG"
    elif sot >= 4:
        tier = "STRONG"
    else:
        tier = "PRESSURE"

    trend = ""
    sot_rate = 0.0
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    if sot_rate >= 0.3:
        tier = bump_tier(tier)

    return tier, trend, sot_rate


def tier_emoji(tier: str) -> str:
    if tier == "VERY STRONG": return "\U0001f534"
    if tier == "STRONG": return "\U0001f7e0"
    return "\U0001f7e1"


# ============================================================
# LOCAL FILTERING (zero API cost)
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[int, int, int]]:
    """Find all candidate teams from live fixtures in tracked leagues.

    v9.5: No scoreline filtering — all fixtures in the 20'-80' window
    are candidates. Scoreline affects RANKING, not inclusion.
    """
    candidates = []
    for fixture in fixtures:
        lid = fixture["league"]["id"]
        if lid not in LEAGUE_IDS:
            continue
        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue
        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

        # v9.5: strict 20'-80' window, no late tracking
        if minute < MINUTE_MIN or minute > MINUTE_MAX:
            continue

        fid = fixture["fixture"]["id"]
        home_tid = fixture["teams"]["home"]["id"]
        away_tid = fixture["teams"]["away"]["id"]

        # Both teams are candidates — ranking decides priority
        candidates.append((fid, home_tid, rank_candidate(fixture, home_tid)))
        candidates.append((fid, away_tid, rank_candidate(fixture, away_tid)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def cleanup_state(live_fixture_ids: set[int]):
    to_delete = [k for k in team_state if k[0] not in live_fixture_ids]
    for k in to_delete:
        del team_state[k]
    if to_delete:
        log.info(f"  Cleaned up state for {len(to_delete)} ended fixture(s)")
    for fid in list(signaled_fixtures):
        if fid not in live_fixture_ids:
            signaled_fixtures.discard(fid)
    # v9.5.4: Clean per-team signaled dict
    for key in list(signaled_teams):
        if key[0] not in live_fixture_ids:
            signaled_teams.pop(key, None)
    for fid in list(last_stats_check):
        if fid not in live_fixture_ids:
            del last_stats_check[fid]
    for fid in list(fast_priority):
        if fid not in live_fixture_ids:
            del fast_priority[fid]
    for fid in list(fast_sot_until):
        if fid not in live_fixture_ids:
            del fast_sot_until[fid]


def find_cached_fixture(fid: int):
    for f in cached_fixtures:
        if f["fixture"]["id"] == fid:
            return f
    return None


def is_fixture_monitorable(fixture: dict) -> bool:
    """v9.5: strict 80' max — no late tracking beyond 80'."""
    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        return False
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    if minute < MINUTE_MIN or minute > MINUTE_MAX:
        return False
    return True


# ============================================================
# SOT-SMART PRIORITY (updated for v9.5 diversification)
# ============================================================

def get_fixture_sot_priority(fid: int) -> int:
    """Dynamic priority for stats polling order.
    Higher = more urgent.
    """
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    is_signaled = fid in signaled_fixtures

    if not has_state:
        base = 55
    elif best_sot >= 3:
        base = 95
    elif best_sot == 2:
        base = 90
    elif best_sot == 1:
        base = 40
    else:
        base = 10

    # v9.5.3: Penalize only if BOTH teams have signaled
    both_signaled = all(
        (fid, tid) in signaled_teams
        for (f, tid) in team_state if f == fid
    )
    if both_signaled and is_signaled:
        base = int(base * 0.5)

    return base


def get_sot_based_interval(fid: int, base_interval: int) -> int:
    """v9.5: SOT-aware polling interval with diversification.

    Already-signaled fixtures get 2x interval to prioritize new matches.
    Fast window only activates for unsignaled fixtures.
    """
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    # v9.5.3: Per-team signaled check
    team_signaled_count = sum(1 for (f, t) in signaled_teams if f == fid)
    both_teams_signaled = team_signaled_count >= 2

    if not has_state:
        interval = base_interval
    elif best_sot >= 2 and not both_teams_signaled and is_fast_sot_active(fid):
        # Fast window: poll at shorter interval (but not 60s to save quota)
        # v9.5.3: active as long as at least one team is unsignaled
        interval = 120
    elif best_sot >= 2:
        interval = 180
    elif best_sot == 1:
        interval = int(base_interval * 1.2)
    else:
        interval = int(base_interval * 1.5)

    # v9.5.3: Only apply 2x if BOTH teams in this fixture have signaled
    if both_teams_signaled:
        interval = int(interval * 2.0)

    return interval


# ============================================================
# DISCOVERY — fetch live fixtures, update candidates & monitored set
# ============================================================

def do_discovery(client: httpx.Client) -> bool:
    """Run one discovery cycle. Updates global state.
    Returns True if discovery succeeded.
    """
    global last_discovery_time, cached_fixtures, fast_monitored

    data = api_get(client, "/fixtures", {"live": "all"})
    cached_fixtures = data.get("response", [])
    last_discovery_time = time.time()

    tracked = [f for f in cached_fixtures if f["league"]["id"] in LEAGUE_IDS]
    budget = get_budget_mode()

    log.info(
        f"Discovery: Quota: {quota_remaining}/{quota_limit} | Mode: {budget} | "
        f"Live: {len(cached_fixtures)} | Tracked: {len(tracked)} | Reqs: {request_count}"
    )

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(
                f"  -> {m['league']['name']}: "
                f"{m['teams']['home']['name']} vs {m['teams']['away']['name']} "
                f"({m['fixture']['status']['short']} {minute}')"
            )

    # Cleanup ended fixtures from all state
    live_ids = {f["fixture"]["id"] for f in cached_fixtures}
    cleanup_state(live_ids)

    # Local pre-filter
    candidates = find_candidates(cached_fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN}-{MINUTE_MAX}')")

    # Build fixture -> best rank mapping
    max_fast = get_max_fast_monitored()
    fixture_best_rank: dict[int, int] = {}
    for fid, tid, rank in candidates:
        if fid not in fixture_best_rank or rank > fixture_best_rank[fid]:
            fixture_best_rank[fid] = rank

    # --- Preserve existing monitored fixtures that are still valid ---
    retained = set()
    for fid in list(fast_monitored):
        fixture = find_cached_fixture(fid)
        if fixture and is_fixture_monitorable(fixture):
            retained.add(fid)

    # Merge: keep existing + add new candidates
    merged = retained | set(fixture_best_rank.keys())

    # If over limit, rank all and keep top N
    if len(merged) > max_fast:
        all_ranked = []
        for fid in merged:
            rank = max(
                fixture_best_rank.get(fid, 0),
                fast_priority.get(fid, 0),
            )
            all_ranked.append((fid, rank))
        all_ranked.sort(key=lambda x: x[1], reverse=True)
        merged = set(fid for fid, _ in all_ranked[:max_fast])

    # Update global state
    fast_monitored.clear()
    fast_monitored |= merged
    for fid in merged:
        if fid in fixture_best_rank:
            fast_priority[fid] = max(
                fast_priority.get(fid, 0), fixture_best_rank[fid]
            )

    log.info(
        f"  -> Monitored: {len(fast_monitored)} fixture(s) "
        f"({len(retained)} retained, {len(merged) - len(retained)} new)"
        f" | Signaled: {len(signaled_fixtures)} | Max: {max_fast}"
    )
    return True


# ============================================================
# BATCHED STATS — one API call for all monitored fixtures
# ============================================================

def parse_xg(tstats: dict) -> str:
    """Extract xG from team statistics. Returns string or 'N/A'."""
    for key in ("Expected Goals", "expectedGoals", "Expected goals"):
        val = tstats.get(key)
        if val is not None:
            try:
                return str(float(val))
            except (ValueError, TypeError):
                pass
    return "N/A"


def get_red_card_string(teams_data: dict, home_name: str, away_name: str) -> str:
    """Build red card string for the signal message."""
    parts = []
    for tname, tstats in teams_data.items():
        try:
            rc = int(tstats.get("Red Cards", "0"))
        except (ValueError, TypeError):
            rc = 0
        if rc > 0:
            parts.append(f"{tname} - {rc}")

    if parts:
        return " | ".join(parts)
    return "None"


def process_fixture_stats(client: httpx.Client, fixture: dict) -> None:
    """Process one fixture from batched /fixtures?ids=... response.

    v9.5: New signal format with xG, top SOT player, red cards.
    Strict 80' cutoff. Tracks signaled fixtures for diversification.
    """
    fid = fixture["fixture"]["id"]

    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        return

    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    # v9.5: strict 80' max — remove from monitoring immediately
    if minute > MINUTE_MAX:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        log.info(f"  Fixture {fid} past {MINUTE_MAX}', removed from monitoring")
        return

    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    home_tid = home["id"]
    away_tid = away["id"]

    statistics = fixture.get("statistics") or []
    if not statistics:
        return

    # Parse all team stats into a dict keyed by team name
    teams_data = {}
    for team_entry in statistics:
        team = team_entry.get("team", {})
        tname = team.get("name")
        if not tname:
            continue
        tmap = {}
        for stat in team_entry.get("statistics", []):
            stat_type = stat.get("type")
            value = stat.get("value")
            if stat_type:
                tmap[stat_type] = "0" if value is None else str(value).strip()
        teams_data[tname] = tmap

    if not teams_data:
        return

    # --- Fast SOT window management (fixture-level) ---
    # Check BOTH teams' SOT. Activate if either >= 2.
    # v9.5: Only activate for unsignaled fixtures.
    best_current_sot = 0
    for tname, tstats in teams_data.items():
        try:
            current_sot = int(tstats.get("Shots on Goal", "0"))
        except (ValueError, TypeError):
            current_sot = 0
        best_current_sot = max(best_current_sot, current_sot)

    # v9.5.3: Activate fast window unless BOTH teams have signaled
    if best_current_sot >= 2:
        team_sig_count = sum(1 for (f, t) in signaled_teams if f == fid)
        if team_sig_count < 2 and not is_fast_sot_active(fid):
            activate_fast_sot(fid)
    elif best_current_sot < 2:
        expire_fast_sot(fid)

    # --- Pre-compute red cards and xG for both teams ---
    red_card_str = get_red_card_string(teams_data, home["name"], away["name"])

    # Parse xG for both teams
    team_xg = {}
    for tname, tstats in teams_data.items():
        team_xg[tname] = parse_xg(tstats)

    # --- SOT SIGNAL CHECK (per team) ---
    for tid, tname in ((home_tid, home["name"]), (away_tid, away["name"])):
        tstats = teams_data.get(tname)
        if not tstats:
            continue

        try:
            sot = int(tstats.get("Shots on Goal", "0"))
        except (ValueError, TypeError):
            continue

        # Get xG for this team
        xg_str = team_xg.get(tname, "N/A")

        # Get opponent SOT and xG
        opponent_sot = "0"
        opponent_xg = "N/A"
        for oname, ostats in teams_data.items():
            if oname != tname:
                try:
                    opponent_sot = str(int(ostats.get("Shots on Goal", "0")))
                except (ValueError, TypeError):
                    opponent_sot = "0"
                opponent_xg = team_xg.get(oname, "N/A")
                break

        state = team_state.get((fid, tid))
        tier, trend, sot_rate = classify_signal(sot, state, minute)

        # Store state AFTER classification (for next comparison)
        team_state[(fid, tid)] = {
            "last_sot": sot,
            "last_minute": minute,
            "last_xg": xg_str if xg_str != "N/A" else None,
        }

        if not tier:
            continue

        # --- v9.5.4: Signal limit rules ---
        # 1st signal: always send (the gold signal, SOT >= 3)
        # 2nd signal: sent if +1 SOT (already guaranteed by classify_signal)
        # 3rd+ signal: only if +2 SOT jump AND 0 goals since LAST signal
        team_sig = signaled_teams.get((fid, tid))
        sig_count = team_sig["count"] if team_sig else 0

        if sig_count >= 2:
            # 3rd+ signal: need +2 SOT jump
            prev_sot = state["last_sot"] if state else 0
            sot_jump = sot - prev_sot
            if sot_jump < 2:
                log.info(
                    f"  BLOCKED {tier}: {tname} - "
                    f"{sot} SOT (+{sot_jump} only, need +2) "
                    f"(sig #{sig_count + 1}, fixture {fid})"
                )
                continue

            # Check if team scored since LAST signal
            is_home_check = (tid == home_tid)
            current_goals = (
                fixture["goals"]["home"] if is_home_check
                else fixture["goals"]["away"]
            ) or 0
            goals_at_last = team_sig.get("goals_at_last_signal", current_goals)
            if current_goals > goals_at_last:
                log.info(
                    f"  BLOCKED {tier}: {tname} - "
                    f"{sot} SOT (+{sot_jump}) but scored "
                    f"{current_goals - goals_at_last} goal(s) since last signal "
                    f"(sig #{sig_count + 1}, fixture {fid})"
                )
                continue

        # --- Signal passes all checks, send it ---
        # Always update goals_at_last_signal to current goals
        is_new_team = sig_count == 0
        is_home_sg = (tid == home_tid)
        goals_now = (
            fixture["goals"]["home"] if is_home_sg
            else fixture["goals"]["away"]
        ) or 0
        if is_new_team:
            signaled_teams[(fid, tid)] = {
                "count": 1, "goals_at_last_signal": goals_now
            }
        else:
            signaled_teams[(fid, tid)]["count"] = sig_count + 1
            signaled_teams[(fid, tid)]["goals_at_last_signal"] = goals_now
        signaled_fixtures.add(fid)

        league = LEAGUE_IDS.get(
            fixture["league"]["id"], fixture["league"]["name"]
        )
        sh = fixture["goals"]["home"] or 0
        sa = fixture["goals"]["away"] or 0

        # Build the signal message
        sig_num = sig_count + 1
        sig_label = f"{sig_num}{'st' if sig_num == 1 else 'nd' if sig_num == 2 else 'rd' if sig_num == 3 else 'th'}"
        msg = (
            f"{tier_emoji(tier)} {tier} GOAL PRESSURE ({sig_label})\n\n"
            f"{home['name']}  {sh} - {sa}  {away['name']}\n"
            f"{league} | {minute}'\n\n"
            f"{tname}\n"
            f"SOT: {sot}\n"
            f"xG: {xg_str}\n"
            f"Opponent SOT: {opponent_sot}\n"
            f"Opponent xG: {opponent_xg}\n\n"
            f"Top SOT Player: N/A\n\n"
            f"Red Cards: {red_card_str}\n"
        )
        if trend:
            msg += f"Trend: {trend}"

        if send_telegram(client, msg):
            log.info(
                f"  SIGNAL {tier}: {tname} - "
                f"{sot} SOT, xG={xg_str} (fixture {fid}, "
                f"{sig_label} signal)"
            )

        signals_sent.append({
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "fixture": fid, "team": tname, "league": league,
            "minute": minute, "sot": sot, "xg": xg_str,
            "red_cards": red_card_str, "tier": tier,
            "trend": trend, "is_new": is_new_team,
        })


def check_monitored_stats(
    client: httpx.Client,
    fixture_ids: list[int],
) -> bool:
    """Fetch statistics for monitored fixtures.

    v9.5.3:
    1. First batch-refresh score/minute via /fixtures?ids=X-Y-Z (1 call)
    2. Then fetch individual /fixtures/statistics?fixture=X for SOT data
    This ensures signals show live scores and correct match minutes.

    Also always updates last_stats_check to prevent infinite re-polling.
    """
    global last_stats_check, cached_fixtures

    valid_ids = []
    for fid in fixture_ids:
        fixture = find_cached_fixture(fid)
        if not fixture:
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            continue
        if not is_fixture_monitorable(fixture):
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            continue
        valid_ids.append(fid)

    if not valid_ids:
        return False

    # --- v9.5.3: Refresh cached fixture data (score/minute) ---
    # Use /fixtures?ids=X-Y-Z to get fresh scores and minutes.
    # This endpoint returns fixture metadata but NOT statistics.
    if valid_ids:
        ids_str = "-".join(str(fid) for fid in valid_ids)
        try:
            refresh_data = api_get(
                client, "/fixtures", {"ids": ids_str}
            )
            refresh_response = refresh_data.get("response", [])
            if refresh_response:
                # Update cached fixtures with fresh data
                refreshed_ids = set()
                for rf in refresh_response:
                    rf_id = rf["fixture"]["id"]
                    # Find and update the cached entry, or add new
                    for i, cf in enumerate(cached_fixtures):
                        if cf["fixture"]["id"] == rf_id:
                            cached_fixtures[i] = rf
                            refreshed_ids.add(rf_id)
                            break
                    else:
                        cached_fixtures.append(rf)
                        refreshed_ids.add(rf_id)
                if refreshed_ids:
                    log.info(
                        f"  Refreshed score/minute for "
                        f"{len(refreshed_ids)} fixture(s)"
                    )
        except Exception as e:
            log.warning(f"  Score/minute refresh failed: {e}")

    any_success = False
    checked_ids = []

    for fid in valid_ids:
        try:
            stats_data = api_get(
                client, "/fixtures/statistics", {"fixture": fid}
            )
            checked_ids.append(fid)

            stats_response = stats_data.get("response", [])
            if not stats_response:
                log.debug(f"  Fixture {fid}: empty statistics response")
                continue

            any_success = True

            # Build a merged fixture dict from FRESH cached data + fresh stats
            cached = find_cached_fixture(fid)
            if not cached:
                continue

            # Create a copy with the statistics injected
            merged = {
                "fixture": cached["fixture"],
                "teams": cached["teams"],
                "goals": cached["goals"],
                "league": cached["league"],
                "statistics": stats_response,
            }
            process_fixture_stats(client, merged)

        except Exception as e:
            log.warning(f"  Stats failed for fixture {fid}: {e}")
            checked_ids.append(fid)

    # v9.5.2 FIX: ALWAYS update last_stats_check to prevent 10s loop.
    # Even if the response was empty or failed, we still waited and tried.
    now = time.time()
    for fid in checked_ids:
        last_stats_check[fid] = now

    return any_success


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v9.5.4 — Signal Limits: 1st always, 2nd +1 SOT, 3rd+ +2 SOT & 0 goals")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (round-robin with health tracking)")
    log.info("")
    log.info("RANKING (pressure-first):")
    log.info("  1. SOT / attacking pressure  (dominant — 10-1500 pts)")
    log.info("  2. xG                       (important — up to ~150 pts)")
    log.info("  3. Game minute              (prefer 45-70')")
    log.info("  4. Scoreline                (small 0-0 bonus, no penalty)")
    log.info("  5. Diversification          (modest unsignaled bonus)")
    log.info("")
    log.info("SIGNAL: SOT >= 3 mandatory, only on SOT increase")
    log.info(f"  Window: 20'-80' | Max monitored: up to {BATCH_SIZE_LIMIT}")
    log.info("  Key rotation: round-robin with health tracking")
    log.info("  Format: SOT, xG, opponent SOT/xG, red cards, trend")
    log.info("")
    log.info("Architecture: discovery + batched SOT-smart stats")
    log.info(f"  Active: {ACTIVE_HOUR_START}:00-{ACTIVE_HOUR_END}:00 Bulgaria local (Sofia TZ, DST-auto)")
    log.info(f"  Fast SOT window: {FAST_SOT_WINDOW}s (unsignaled only)")
    log.info("")
    log.info("Quota: subscription-level from API headers (not additive)")
    log.info("  >=51: NORMAL  (240s base stats)")
    log.info("  26-50: CAREFUL (300s base stats)")
    log.info("  11-25: STRICT  (420s base stats)")
    log.info("  1-10: EMERGENCY (600s base stats)")
    log.info("  0: STOP")
    log.info("")
    log.info("SOT-smart intervals (v9.5):")
    log.info("  SOT>=2 + fast window: 120s (unsignaled only)")
    log.info("  SOT>=2 + no window:  180s")
    log.info("  SOT==1:               1.2x base")
    log.info("  SOT==0:               1.5x base")
    log.info("  Signaled fixtures: 2x all intervals (both teams signaled)")
    log.info("  v9.5.4: Signal limits — 1st always, 2nd +1 SOT, 3rd+ +2 SOT & 0 goals since last")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        while True:
            now = time.time()
            utc_hour = datetime.now(timezone.utc).hour

            # --- Dead hours (zero API cost) ---
            # Check using actual Bulgaria local hour (handles DST)
            local_hour = datetime.now(BULGARIA_TZ).hour
            if not (ACTIVE_HOUR_START <= local_hour < ACTIVE_HOUR_END):
                log.info(
                    f"Dead hours (local {local_hour}:00, "
                    f"active {ACTIVE_HOUR_START}:00-{ACTIVE_HOUR_END}:00 Bulgaria), "
                    f"sleeping 30 min..."
                )
                time.sleep(1800)
                continue

            # --- Rate limit backoff ---
            if now < rate_limited_until:
                sleep_remaining = int(rate_limited_until - now)
                log.warning(f"Rate limited, backing off {sleep_remaining}s")
                time.sleep(min(sleep_remaining, 60))
                continue

            budget = get_budget_mode()

            if budget == "STOP":
                log.warning(
                    f"Quota exhausted ({quota_remaining}/{quota_limit}), "
                    f"sleeping 30 min..."
                )
                time.sleep(1800)
                continue

            # --- Determine current state from cached data ---
            has_tracked_live = bool(
                [f for f in cached_fixtures if f["league"]["id"] in LEAGUE_IDS]
            ) if cached_fixtures else False
            has_candidates = bool(fast_monitored)

            # --- DISCOVERY (timed independently) ---
            discovery_interval = get_discovery_interval(
                budget, has_tracked_live, has_candidates
            )
            need_discovery = (now - last_discovery_time) >= discovery_interval

            if need_discovery:
                try:
                    do_discovery(client)
                    has_candidates = bool(fast_monitored)
                    has_tracked_live = bool(
                        [f for f in cached_fixtures
                         if f["league"]["id"] in LEAGUE_IDS]
                    )
                    discovery_interval = get_discovery_interval(
                        budget, has_tracked_live, has_candidates
                    )
                except Exception as e:
                    log.error(f"Discovery failed: {e}")
                    time.sleep(60)
                    continue

            # ------------------------------------------------------------
            # STATS CHECK — batched, SOT-smart, diversification-aware
            # ------------------------------------------------------------

            if fast_monitored and budget != "STOP":
                ordered = sorted(
                    fast_monitored,
                    key=get_fixture_sot_priority,
                    reverse=True,
                )

                eligible = []
                for fid in ordered:
                    fixture_interval = get_sot_based_interval(
                        fid, get_stats_interval(budget)
                    )
                    time_since = now - last_stats_check.get(fid, 0)
                    if time_since >= fixture_interval:
                        eligible.append(fid)

                if eligible:
                    if (quota_remaining is not None
                            and quota_remaining <= 2):
                        log.warning(
                            "Quota nearly exhausted; skipping stats."
                        )
                    else:
                        log.info(
                            f"  Stats -> {len(eligible)} fixture(s): "
                            f"{eligible}"
                        )
                        check_monitored_stats(client, eligible)

            # --- CALCULATE SLEEP ---
            now = time.time()
            next_disc_in = max(
                0, discovery_interval - (now - last_discovery_time)
            )

            next_stats_in = -1
            if fast_monitored:
                stats_int = get_stats_interval(get_budget_mode())
                for fid in fast_monitored:
                    fixture_interval = get_sot_based_interval(fid, stats_int)
                    fid_next = max(
                        0,
                        fixture_interval - (now - last_stats_check.get(fid, 0))
                    )
                    if next_stats_in < 0 or fid_next < next_stats_in:
                        next_stats_in = fid_next

            if next_stats_in < 0:
                sleep_time = next_disc_in
            else:
                sleep_time = min(next_disc_in, next_stats_in)

            sleep_time = max(sleep_time, 10)
            sleep_time = min(sleep_time, 60)

            tracked_count = len(
                [f for f in cached_fixtures
                 if f["league"]["id"] in LEAGUE_IDS]
            ) if cached_fixtures else 0

            # Status summary
            fast_window_info = ""
            if fast_sot_until:
                parts = []
                for fid, until in fast_sot_until.items():
                    remaining = max(0, int(until - now))
                    sot = get_fixture_best_sot(fid)
                    sig_flag = f"{sum(1 for (f,t) in signaled_teams if f==fid)}/2"
                    parts.append(f"{fid}(SOT={sot},sig={sig_flag},{remaining}s)")
                fast_window_info = f" | FastWin: {', '.join(parts)}"

            total_signals = sum(v["count"] for v in signaled_teams.values())
            stats_str = (
                f"{int(next_stats_in)}s" if next_stats_in >= 0 else "N/A"
            )
            log.info(
                f"Quota: {quota_remaining}/{quota_limit} | "
                f"Mode: {get_budget_mode()} | "
                f"Tracked: {tracked_count} | Mon: {len(fast_monitored)} | "
                f"Sig: {len(signaled_teams)}teams/{len(signaled_fixtures)}fix/{total_signals}sent | "
                f"Keys: {healthy_key_count()}/{len(API_KEYS)} | "
                f"Next disc: {int(next_disc_in)}s | Next stats: {stats_str} | "
                f"Sleep: {int(sleep_time)}s | Reqs: {request_count} | "
                f"Signals: {len(signals_sent)}{fast_window_info}"
            )

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
