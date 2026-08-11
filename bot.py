import os
import sys
import time
import logging
import httpx
from datetime import datetime, timezone
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

# Dead hours UTC — no European league action worth polling
# 21:00 UTC = 23:00 EET (Bulgaria) — no evening matches worth tracking
# 08:00 UTC = 10:00 EET (Bulgaria) — earliest kickoffs ~18:00 local
DEAD_HOUR_START = 21  # 21:00 UTC (23:00 Bulgaria)
DEAD_HOUR_END = 8     # 08:00 UTC (10:00 Bulgaria)

MINUTE_MIN = 25
MINUTE_MAX = 80
MINUTE_LATE_MAX = 90  # only for already-tracked teams

# Fast SOT polling window: once SOT reaches 2+, we poll at 60s for at most
# this many seconds. After the window expires, SOT=2 drops to 120s.
# This prevents quota burn if a team hovers at SOT=2 for a long time.
FAST_SOT_WINDOW = 5 * 60  # 300 seconds

# Max fixture IDs per batched request (API-Football limit for /fixtures?ids=...)
BATCH_SIZE_LIMIT = 20

# --- State ---
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []
red_card_info: dict[int, str] = {}   # fixture_id -> context string
rate_limited_until: float = 0.0    # timestamp — back off until this time

# API-Football reports the daily quota at the subscription level.
# Do NOT assume that multiple API keys multiply the daily allowance.
quota_remaining: int | None = None
quota_limit: int | None = None
minute_remaining: int | None = None
minute_limit: int | None = None

# Keep one key active. Rotate only after an actual 401/403 failure.
active_key_index = 0

# --- Adaptive polling state ---
last_discovery_time: float = 0.0
last_stats_check: dict[int, float] = {}   # fixture_id -> timestamp of last stats fetch
fast_monitored: set[int] = set()         # fixture IDs currently fast-polled
fast_priority: dict[int, int] = {}       # fixture_id -> rank score (discovery-time)
cached_fixtures: list[dict] = []        # last discovery result (reused for filtering only)

# --- Fast SOT window state ---
# Maps fixture_id -> timestamp until which 60s polling is active.
# Activated when a team's SOT reaches >= 2. After FAST_SOT_WINDOW seconds,
# the window expires and polling for SOT=2 reverts to 120s.
fast_sot_until: dict[int, float] = {}


# ============================================================
# API KEY & QUOTA
# ============================================================

def pick_key() -> str:
    return API_KEYS[active_key_index % len(API_KEYS)]


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


def rotate_key(reason: str):
    """Move to the next API key after an authentication failure."""
    global active_key_index
    old = active_key_index
    active_key_index = (active_key_index + 1) % len(API_KEYS)
    log.warning(
        f"Key rotated: index {old} -> {active_key_index} "
        f"({reason})"
    )


# ============================================================
# API HELPERS
# ============================================================

def api_get(client: httpx.Client, endpoint: str, params: dict = None) -> dict:
    global request_count, rate_limited_until, active_key_index

    if quota_remaining is not None and quota_remaining <= 0:
        raise Exception("Daily API quota exhausted")

    # Try up to all available keys on auth failure
    keys_tried = 0
    max_attempts = len(API_KEYS)

    while keys_tried < max_attempts:
        key = pick_key()
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
            rate_limited_until = time.time() + 120
            raise Exception("Rate limited (429), backing off 120s")

        if resp.status_code in (401, 403):
            keys_tried += 1
            if keys_tried < max_attempts:
                rotate_key(f"HTTP {resp.status_code}")
                continue
            raise Exception(
                f"All {max_attempts} key(s) failed with auth errors"
            )

        resp.raise_for_status()
        return resp.json()

    # Should not reach here, but safety net
    raise Exception("No API keys available")


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
# CANDIDATE RANKING (free data only — no API cost)
# ============================================================

def is_interesting_scoreline(fixture: dict, team_id: int) -> bool:
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0
    if tg - og >= 3:
        return False
    if og - tg >= 4:
        return False
    return True


def get_team_goals(fixture: dict, team_id: int) -> int:
    is_home = fixture["teams"]["home"]["id"] == team_id
    return (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0


def rank_candidate(fixture: dict, team_id: int) -> int:
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    fid = fixture["fixture"]["id"]

    # --- SOT-related factors (DOMINANT weight — SOT is king) ---
    prev = team_state.get((fid, team_id))
    if prev:
        prev_sot = prev.get("last_sot", 0)
        if prev_sot >= 3:
            score += 20    # already at signal level — top priority
        elif prev_sot >= 2:
            score += 15    # one shot away from signal
        elif prev_sot >= 1:
            score += 5     # some activity

    # --- Minute window (secondary factor) ---
    if 65 <= minute <= MINUTE_MAX:
        score += 3
    elif 55 <= minute <= 64:
        score += 2
    elif MINUTE_MIN <= minute <= 54:
        score += 1

    # --- Scoreline context (tertiary factor) ---
    tg = get_team_goals(fixture, team_id)
    is_home = fixture["teams"]["home"]["id"] == team_id
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if og > tg:
        if og - tg == 1:
            score += 3
        else:
            score += 2
    elif tg == 0 and og == 0:
        score += 1
    elif tg > og:
        if tg - og == 1:
            score += 1

    return score


def get_score_context(fixture: dict, team_id: int) -> str:
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0
    diff = og - tg

    if diff >= 3: return f"down {tg}-{og}"
    if diff == 2: return f"down 2 goals {tg}-{og}"
    if diff == 1: return f"trailing {tg}-{og}"
    if diff == -1: return f"leading {tg}-{og}"
    if diff <= -2: return f"up {tg}-{og}"
    return f"level {tg}-{og}"


def get_score_emoji(fixture: dict, team_id: int) -> str:
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if og - tg == 1: return "\U0001f525\U0001f525"
    if tg == og: return "\U0001f525"
    if tg - og == 1: return "\U0001f525"
    if og - tg == 2: return "\U0001f525"
    if tg - og >= 2: return "\u26a0\ufe0f"
    if og - tg >= 3: return "\u26a0\ufe0f"
    return ""


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    """
    Budget is based on the real subscription-level remaining quota.

    Keep ~10 requests in reserve so the bot doesn't die completely because
    of a burst or an unexpected retry.
    """
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
    """
    One fixture is preferred on a 100/day plan.
    Two fixtures are allowed only when there is substantial quota left.
    """
    if quota_remaining is None:
        return 1
    if quota_remaining >= 60:
        return 2
    if quota_remaining >= 10:
        return 1
    return 0


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
            "NORMAL": 900,       # 15 min
            "CAREFUL": 1200,     # 20 min
            "STRICT": 1800,      # 30 min
            "EMERGENCY": 2400,   # 40 min
            "UNKNOWN": 1800,
        }.get(budget_mode, 1800)

    # Once a candidate is being monitored, spend quota on statistics,
    # not repeatedly downloading the complete live fixture list.
    return {
        "NORMAL": 1800,          # 30 min
        "CAREFUL": 2400,         # 40 min
        "STRICT": 3000,          # 50 min
        "EMERGENCY": 3600,       # 60 min
        "UNKNOWN": 1800,
    }.get(budget_mode, 1800)


def get_stats_interval(budget_mode: str) -> int:
    """
    Base interval for statistics.

    On the Free 100/day plan, 90s is far too aggressive if a match remains
    monitored for hours. Start at 4 minutes and dynamically slow down when
    SOT is low.
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
# FAST SOT WINDOW — 60s polling when SOT>=2 (time-limited)
# ============================================================

def get_fixture_best_sot(fid: int) -> int:
    """Return the highest known SOT for any team in a fixture."""
    best = 0
    for (f, t), state in team_state.items():
        if f == fid:
            sot = state.get("last_sot", 0)
            if sot > best:
                best = sot
    return best


def activate_fast_sot(fid: int):
    """Activate the 60s polling window for a fixture.
    Called when any team in the fixture reaches SOT >= 2.
    Resets the window if already active (fresh 5 minutes)."""
    fast_sot_until[fid] = time.time() + FAST_SOT_WINDOW
    log.info(
        f"  Fast SOT window ACTIVATED for fixture {fid} "
        f"(expires in {FAST_SOT_WINDOW}s)"
    )


def is_fast_sot_active(fid: int) -> bool:
    """Check if the 60s polling window is currently active for a fixture."""
    until = fast_sot_until.get(fid)
    if until is None:
        return False
    if time.time() >= until:
        # Window expired, clean up
        del fast_sot_until[fid]
        return False
    return True


def expire_fast_sot(fid: int):
    """Force-expire the fast SOT window for a fixture."""
    fast_sot_until.pop(fid, None)


# ============================================================
# SIGNAL CLASSIFICATION — SOT is king (UNCHANGED)
# ============================================================

TIER_ORDER = ["PRESSURE", "STRONG", "VERY STRONG"]


def bump_tier(tier: str) -> str:
    idx = TIER_ORDER.index(tier)
    if idx < len(TIER_ORDER) - 1:
        return TIER_ORDER[idx + 1]
    return tier


def classify_signal(sot: int, state: dict | None, current_minute: int) -> tuple[str | None, str, float]:
    if sot < 3:
        return None, "", 0.0

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


def tier_label(tier: str) -> str:
    if tier == "VERY STRONG": return "[RED]"
    if tier == "STRONG": return "[ORANGE]"
    return "[YELLOW]"


# ============================================================
# LOCAL FILTERING (zero API cost)
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[int, int, int]]:
    candidates = []
    for fixture in fixtures:
        lid = fixture["league"]["id"]
        if lid not in LEAGUE_IDS:
            continue
        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue
        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
        fid = fixture["fixture"]["id"]

        home_tid = fixture["teams"]["home"]["id"]
        away_tid = fixture["teams"]["away"]["id"]

        in_window = False
        if MINUTE_MIN <= minute <= MINUTE_MAX:
            in_window = True
        elif MINUTE_MAX < minute <= MINUTE_LATE_MAX:
            if (fid, home_tid) in team_state or (fid, away_tid) in team_state:
                in_window = True

        if not in_window:
            continue

        if is_interesting_scoreline(fixture, home_tid):
            candidates.append((fid, home_tid, rank_candidate(fixture, home_tid)))
        if is_interesting_scoreline(fixture, away_tid):
            candidates.append((fid, away_tid, rank_candidate(fixture, away_tid)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def cleanup_state(live_fixture_ids: set[int]):
    to_delete = [k for k in team_state if k[0] not in live_fixture_ids]
    for k in to_delete:
        del team_state[k]
    if to_delete:
        log.info(f"  Cleaned up state for {len(to_delete)} ended fixture(s)")
    for fid in list(red_card_info):
        if fid not in live_fixture_ids:
            del red_card_info[fid]


def find_cached_fixture(fid: int):
    """Find fixture in cached discovery data."""
    for f in cached_fixtures:
        if f["fixture"]["id"] == fid:
            return f
    return None


def is_fixture_monitorable(fixture: dict) -> bool:
    """Check if a fixture is still worth monitoring."""
    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        return False
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    if minute < MINUTE_MIN:
        return False
    if minute > MINUTE_LATE_MAX:
        return False
    if minute > MINUTE_MAX:
        fid = fixture["fixture"]["id"]
        home_tid = fixture["teams"]["home"]["id"]
        away_tid = fixture["teams"]["away"]["id"]
        if (fid, home_tid) not in team_state and (fid, away_tid) not in team_state:
            return False
    return True


# ============================================================
# SOT-SMART PRIORITY — dynamic, updated after every stats check
# ============================================================

def get_fixture_sot_priority(fid: int) -> int:
    """Dynamic priority based on latest known SOT for the best team in fixture.
    Higher score = more urgent stats check needed.

    SOT>=3: 95 (at signal level — next increase triggers another signal)
    SOT==2: 90 (one shot from threshold — highest urgency for FIRST signal)
    SOT==1: 40 (building, but far from threshold)
    SOT==0: 10 (no SOT activity)
    Unknown (no state): 55 (first check needed — establishes baseline)"""
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    if not has_state:
        return 55
    if best_sot >= 3:
        return 95
    if best_sot == 2:
        return 90
    if best_sot == 1:
        return 40
    return 10


def get_sot_based_interval(fid: int, base_interval: int) -> int:
    """SOT-aware polling interval. The closer a team is to the signal
    threshold (SOT>=3), the more frequently we check.

    SOT>=2 + fast window active: 60s (poll at API's natural ~1min update cadence)
    SOT>=2 + no fast window:    120s (still near threshold, but window expired)
    SOT==1:                     1.75x base
    SOT==0:                     3x base
    Unknown:                    1.5x base (first check important but not urgent)

    The fast window activates when SOT first reaches >=2 and lasts
    FAST_SOT_WINDOW (5 min). After that, SOT=2 polling slows to 120s
    to prevent quota burn if a team hovers at SOT=2 indefinitely."""
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)

    if not has_state:
        return int(base_interval * 1.5)

    if best_sot >= 2:
        # At or near the signal threshold — this is the critical zone.
        if is_fast_sot_active(fid):
            # Fast window active: poll at the API's natural ~1-minute
            # statistics update cadence. Every request is likely to show
            # new data because 60s > API's ~60s update cycle.
            return 60
        # Window expired but SOT still >=2: slower but still attentive.
        # The team has been at SOT>=2 for >5 min without hitting 3.
        return 120

    if best_sot == 1:
        return int(base_interval * 1.75)
    return int(base_interval * 3.0)


# ============================================================
# DISCOVERY — fetch live fixtures, update candidates & fast_monitored
# ============================================================

def do_discovery(client: httpx.Client) -> bool:
    """Run one discovery cycle. Updates global state.
    Returns True if discovery succeeded."""
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

    # Clean polling state for ended fixtures
    for fid in list(last_stats_check):
        if fid not in live_ids:
            del last_stats_check[fid]
    for fid in list(fast_priority):
        if fid not in live_ids:
            del fast_priority[fid]
    for fid in list(fast_sot_until):
        if fid not in live_ids:
            del fast_sot_until[fid]

    # Local pre-filter
    candidates = find_candidates(cached_fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN}-{MINUTE_MAX}')")

    # Build fixture -> best rank mapping from new candidates
    max_fast = get_max_fast_monitored()
    fixture_best_rank: dict[int, int] = {}
    for fid, tid, rank in candidates:
        if fid not in fixture_best_rank or rank > fixture_best_rank[fid]:
            fixture_best_rank[fid] = rank

    # --- Preserve existing monitored fixtures ---
    retained = set()
    for fid in list(fast_monitored):
        fixture = find_cached_fixture(fid)
        if fixture and is_fixture_monitorable(fixture):
            retained.add(fid)

    # Merge: keep existing + add new candidates
    merged = retained | set(fixture_best_rank.keys())

    # If over limit, rank all and keep top N (weakest removed first)
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
        f"  -> Fast-monitored: {len(fast_monitored)} fixture(s)"
        f" ({len(retained)} retained, {len(merged) - len(retained)} new)"
    )
    return True


# ============================================================
# BATCHED STATS — one API call for all monitored fixtures
# ============================================================

def process_fixture_stats(client: httpx.Client, fixture: dict) -> None:
    """
    Process one fixture returned by /fixtures?ids=...

    IMPORTANT: Uses the fresh fixture data from the API response,
    NOT the stale cached_fixtures. This fixes the problem where
    scores/minutes were several minutes old during signal generation.
    """
    fid = fixture["fixture"]["id"]

    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        return

    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    if minute > MINUTE_LATE_MAX:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        return

    # Use fresh data from the batched response
    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    home_tid = home["id"]
    away_tid = away["id"]

    # The /fixtures?ids=... response includes statistics for live fixtures
    statistics = fixture.get("statistics") or []
    if not statistics:
        return

    # Parse all team stats
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

    # --- Red card context (NO separate alert, just context for SOT messages) ---
    rc_parts = []
    for tname, tstats in teams_data.items():
        try:
            rc = int(tstats.get("Red Cards", "0"))
        except (ValueError, TypeError):
            rc = 0
        if rc > 0:
            rc_parts.append(f"{tname}: {rc}")
    if rc_parts:
        red_card_info[fid] = ", ".join(rc_parts)
    else:
        red_card_info.pop(fid, None)

    # --- Fast SOT window management (fixture-level) ---
    # The window belongs to the fixture, not a single team.
    # Check BOTH teams' SOT before deciding — if either team has SOT >= 2,
    # activate the fast window. Only expire when both are below 2.
    best_current_sot = 0
    for tname, tstats in teams_data.items():
        try:
            current_sot = int(tstats.get("Shots on Goal", "0"))
        except (ValueError, TypeError):
            current_sot = 0
        best_current_sot = max(best_current_sot, current_sot)

    if best_current_sot >= 2:
        if not is_fast_sot_active(fid):
            activate_fast_sot(fid)
    else:
        expire_fast_sot(fid)

    # --- SOT SIGNAL CHECK (per team in this fixture) ---
    for tid, tname in ((home_tid, home["name"]), (away_tid, away["name"])):
        tstats = teams_data.get(tname)
        if not tstats:
            continue

        try:
            sot = int(tstats.get("Shots on Goal", "0"))
            total_shots = int(tstats.get("Total Shots", "0"))
            corners = int(tstats.get("Corner Kicks", "0"))
        except (ValueError, TypeError):
            continue

        possession = tstats.get("Ball Possession", "50%")

        state = team_state.get((fid, tid))
        tier, trend, sot_rate = classify_signal(sot, state, minute)

        # Use fresh score from this API response, not cached data
        current_goals = get_team_goals(fixture, tid)
        prev_goals = (
            state.get("last_goals", current_goals)
            if state
            else current_goals
        )
        scored_since_last = current_goals > prev_goals

        # Store state AFTER classification
        team_state[(fid, tid)] = {
            "last_sot": sot,
            "last_minute": minute,
            "last_goals": current_goals,
        }

        if not tier:
            continue

        ctx = get_score_context(fixture, tid)
        score_emoji = get_score_emoji(fixture, tid)

        opponent_sot = "?"
        for oname, ostats in teams_data.items():
            if oname != tname:
                opponent_sot = ostats.get("Shots on Goal", "?")
                break

        league = LEAGUE_IDS.get(
            fixture["league"]["id"], fixture["league"]["name"]
        )
        sh = fixture["goals"]["home"]
        sa = fixture["goals"]["away"]

        msg = (
            f"{tier_emoji(tier)} {tier_label(tier)} "
            f"{tier} GOAL PRESSURE {score_emoji}\n\n"
            f"{home['name']}  {sh} - {sa}  {away['name']}\n"
            f"{league}  {minute}'\n\n"
            f"{tname} ({ctx})\n"
            f"  Shots on target: {sot}\n"
            f"  Total shots: {total_shots}\n"
            f"  Opponent SOT: {opponent_sot}\n"
            f"  Possession: {possession}\n"
            f"  Corners: {corners}\n"
        )
        if trend:
            msg += f"  Trend: {trend}\n"
        if fid in red_card_info:
            msg += f"  \U0001f7e5 Red cards: {red_card_info[fid]}\n"
        if scored_since_last:
            msg += (
                "  \u26bd Scored since last signal — "
                "pressure continues\n"
            )

        if send_telegram(client, msg):
            log.info(
                f"  SIGNAL {tier}: {tname} ({ctx}) - "
                f"{sot} SOT, {total_shots} shots (fixture {fid})"
            )

        signals_sent.append({
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "fixture": fid, "team": tname, "league": league,
            "minute": minute, "sot": sot, "total_shots": total_shots,
            "corners": corners, "context": ctx, "tier": tier,
            "trend": trend, "scored_since_last": scored_since_last,
        })


def check_monitored_stats(
    client: httpx.Client,
    fixture_ids: list[int],
) -> bool:
    """
    Fetch statistics for all currently monitored fixtures in ONE API call.

    API-Football supports multiple fixture IDs through /fixtures?ids=...
    The response includes full fixture data + statistics for live fixtures.

    Batch size is limited to BATCH_SIZE_LIMIT (20) IDs per request.
    If more fixtures need checking, they are split into multiple batches.
    """
    global last_stats_check

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

    # Split into batches of at most BATCH_SIZE_LIMIT IDs each.
    # This is a hard API constraint — /fixtures?ids=... supports max 20 IDs.
    batches = []
    for i in range(0, len(valid_ids), BATCH_SIZE_LIMIT):
        batches.append(valid_ids[i:i + BATCH_SIZE_LIMIT])

    any_success = False
    for batch in batches:
        ids = "-".join(str(fid) for fid in batch)

        try:
            data = api_get(client, "/fixtures", {"ids": ids})
        except Exception as e:
            log.warning(f"  Batched stats failed (batch {len(batch)} IDs): {e}")
            continue

        responses = data.get("response", [])
        if responses:
            any_success = True
            for fixture_data in responses:
                process_fixture_stats(client, fixture_data)

    # Update last_stats_check for all valid fixtures (even if some batches failed,
    # we don't want to immediately retry the same batch)
    if any_success:
        now = time.time()
        for fid in valid_ids:
            last_stats_check[fid] = now

    return any_success


# ============================================================
# MAIN LOOP — SOT-smart adaptive polling with batched stats
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v9.4 — Fast SOT Window + Real Key Rotation")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (rotate on 401/403, 429->120s backoff)")
    log.info("")
    log.info("NOTE: All state is in-memory. On Railway restart, team_state resets.")
    log.info("  This means a duplicate signal is possible if SOT>=3 at restart.")
    log.info("  This is accepted — detecting pressure late is better than missing it.")
    log.info("")
    log.info("Architecture: discovery + batched SOT-smart stats")
    log.info("  No tracked live:       discovery 30 min")
    log.info("  Tracked, no cand:      discovery 15-40 min (by budget)")
    log.info("  Active monitoring:     discovery 30-60 min, stats batched")
    log.info(f"  Dead hours:            {DEAD_HOUR_START}:00-{DEAD_HOUR_END}:00 UTC ({DEAD_HOUR_START+2}:00-{DEAD_HOUR_END+2}:00 Bulgaria)")
    log.info("")
    log.info("Quota: REAL subscription-level from API headers (not additive)")
    log.info("  >=51: NORMAL  (240s base stats)")
    log.info("  26-50: CAREFUL (300s base stats)")
    log.info("  11-25: STRICT  (420s base stats)")
    log.info("  1-10: EMERGENCY (600s base stats)")
    log.info("  0: STOP")
    log.info("")
    log.info("SOT-smart intervals (v9.4 — fast window):")
    log.info("  SOT>=2 + fast window:  60s  (API natural cadence)")
    log.info("  SOT>=2 + no window:    120s (window expired, still near)")
    log.info("  SOT==1:                1.75x base (building)")
    log.info("  SOT==0:                3x base (no activity)")
    log.info("  Unknown:               1.5x base (first check)")
    log.info(f"  Fast SOT window:       {FAST_SOT_WINDOW}s (5 min)")
    log.info("")
    log.info("Key optimizations:")
    log.info(f"  /fixtures?ids=... batches N fixtures in 1 request (max {BATCH_SIZE_LIMIT})")
    log.info("  Fast window activates at SOT>=2, expires after 5 min")
    log.info("  Key rotation: only on 401/403, tries all keys before failing")
    log.info("")
    log.info("Signal logic:")
    log.info("  SOT>=3 AND SOT increased -> PRESSURE/STRONG/VERY STRONG")
    log.info("  Red cards: context only (included in SOT message, no separate alert)")
    log.info("  Priority: SOT state > minute window > scoreline")
    log.info(f"  Window: {MINUTE_MIN}'-{MINUTE_MAX}' ({MINUTE_MIN}'-{MINUTE_LATE_MAX}' tracked)")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        while True:
            now = time.time()
            utc_hour = datetime.now(timezone.utc).hour

            # --- Dead hours (zero API cost) ---
            if DEAD_HOUR_START <= utc_hour < DEAD_HOUR_END:
                log.info(
                    f"Dead hours ({DEAD_HOUR_START}:00-{DEAD_HOUR_END}:00 UTC), "
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
            # STATS CHECK — batched, SOT-smart, fast-window-aware
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

                # One batched API request for all eligible fixtures.
                # Split into chunks of BATCH_SIZE_LIMIT (20) if needed.
                if eligible:
                    if (quota_remaining is not None
                            and quota_remaining <= 2):
                        log.warning(
                            "Quota nearly exhausted; skipping stats."
                        )
                    else:
                        log.info(
                            f"  Stats batch -> {len(eligible)} fixture(s): "
                            f"{eligible}"
                        )
                        check_monitored_stats(client, eligible)

            # --- CALCULATE SLEEP ---
            now = time.time()
            next_disc_in = max(
                0, discovery_interval - (now - last_discovery_time)
            )

            # Find the soonest stats check across all monitored fixtures
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

            sleep_time = max(sleep_time, 10)   # min 10s
            sleep_time = min(sleep_time, 60)   # max 60s — re-evaluate often

            tracked_count = len(
                [f for f in cached_fixtures
                 if f["league"]["id"] in LEAGUE_IDS]
            ) if cached_fixtures else 0

            # Build a status summary of fast SOT windows
            fast_window_info = ""
            if fast_sot_until:
                parts = []
                for fid, until in fast_sot_until.items():
                    remaining = max(0, int(until - now))
                    sot = get_fixture_best_sot(fid)
                    parts.append(f"{fid}(SOT={sot},{remaining}s)")
                fast_window_info = f" | FastWin: {', '.join(parts)}"

            stats_str = (
                f"{int(next_stats_in)}s" if next_stats_in >= 0 else "N/A"
            )
            log.info(
                f"Quota: {quota_remaining}/{quota_limit} | "
                f"Mode: {get_budget_mode()} | "
                f"Tracked: {tracked_count} | Fast: {len(fast_monitored)} | "
                f"Next disc: {int(next_disc_in)}s | Next stats: {stats_str} | "
                f"Sleep: {int(sleep_time)}s | Reqs: {request_count} | "
                f"Signals: {len(signals_sent)}{fast_window_info}"
            )

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
