import os
import sys
import time
import logging
import httpx
from datetime import datetime, timedelta, timezone
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

# Active monitoring window: dynamically computed from daily schedule
# Falls back to 14:00-23:00 if schedule fetch fails
BULGARIA_TZ = ZoneInfo("Europe/Sofia")
ACTIVE_HOUR_START_FALLBACK = 14  # fallback
ACTIVE_HOUR_END_FALLBACK = 23    # fallback

# v9.5: 20'-80' window (strict — no late tracking beyond 80')
MINUTE_MIN = 20
MINUTE_MAX = 80

# v9.7.1: Night hours — no European tracked leagues play
# Skip ALL schedule rechecks during this window (saves 2 credits per skipped recheck)
NIGHT_HOUR_START = 1   # 01:00 Bulgaria — all European leagues finished
NIGHT_HOUR_END = 10    # 10:00 Bulgaria — earliest possible kickoff (~11:00 Scandinavia)

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
# Key: (fixture_id, team_id) -> {"count": N, "goals_at_last_signal": G, "sot_at_last_signal": S}
# 1st signal: always sent (SOT >= 3, the gold signal)
# 2nd signal: sent if +1 SOT (guaranteed by classify_signal dedup)
# 3rd+ signal: only if +2 SOT jump from LAST SIGNAL (not last poll) AND 0 goals since LAST signal
# v9.7: sot_at_last_signal fixes the bug where jump was measured from last poll's SOT
signaled_teams: dict[tuple[int, int], dict] = {}
# Keep fixture-level set for backward compat in logs/cleanup
signaled_fixtures: set[int] = set()

# --- v9.8: Signal outcome tracking (backtesting) ---
# Records every signal sent, then checks if a goal followed within 15 game minutes.
# This turns threshold tuning from guessing into data-driven optimization.
# Each entry: {fixture_id, team_id, team_name, league, signal_time, game_minute,
#   sot, tier, goals_at_signal, opponent_goals_at_signal, is_home,
#   outcome: None/HIT/MISS, goal_minute, resolved: bool}
signal_outcomes: list[dict] = []
OUTCOME_WINDOW_MINUTES = 15  # game minutes to wait for a goal before calling MISS

# --- Adaptive polling state ---
last_discovery_time: float = 0.0
last_stats_check: dict[int, float] = {}   # fixture_id -> timestamp of last stats fetch
fast_monitored: set[int] = set()         # fixture IDs currently monitored
fast_priority: dict[int, int] = {}       # fixture_id -> rank score (discovery-time)
cached_fixtures: list[dict] = []        # last discovery result (reused for filtering only)

# --- v9.5.8: Daily match count for adaptive mode ---
total_matches_today: int = 0
FIRST_SIGNAL_ONLY_THRESHOLD = 20  # >20 games: first signal only, move on
FULL_TRACKING_THRESHOLD = 15   # <15 games: full tracking (multiple signals)

# --- Fast SOT window state ---
fast_sot_until: dict[int, float] = {}

# --- v9.7: SOT acceleration tracking ---
# Fixtures where the best SOT increased in the last poll.
# These get priority polling (60-90s) to catch the next SOT increase.
accelerating_fixtures: set[int] = set()

# --- v9.7: Dead fixture tracking (zero-pressure matches) ---
# Fixtures where BOTH teams had SOT=0 on last stats check.
# These are removed from monitoring to save credits.
# They get REVIVED if discovery detects a score change (momentum shift).
# Key: fixture_id -> (home_goals, away_goals) at time of death.
dead_fixtures: dict[int, tuple[int, int]] = {}

# --- v9.7: /fixtures?ids= statistics detection ---
# Lazily tested on first stats call. If True, one /fixtures?ids= call
# returns EVERYTHING (score, minute, SOT, xG, red cards) — no individual
# /fixtures/statistics calls needed. Massive credit saving.
_ids_endpoint_has_stats: bool | None = None  # None=untested, True/False=cached

# --- v9.6.1: Scheduled 20' entry times (UTC timestamps) ---
# Populated from the known schedule at startup.
# The bot wakes up ~60s before each entry instead of polling blindly.
scheduled_window_entries: list[float] = []  # sorted UTC timestamps when kickoffs+20min occur

# --- Dynamic active hours (v9.5.4) ---
# Computed from /fixtures?date=today, re-checked every 3h for late additions
dynamic_active_start: int = ACTIVE_HOUR_START_FALLBACK
dynamic_active_end: int = ACTIVE_HOUR_END_FALLBACK
schedule_date: str = ""  # YYYY-MM-DD of cached schedule
schedule_no_matches: bool = False
schedule_last_fetch: float = 0.0  # timestamp of last schedule fetch
SCHEDULE_RECHECK_INTERVAL = 6 * 3600  # v9.7: reduced from 3h to 6h (saves ~4 credits/day)

# --- v9.7.2: Tomorrow's kickoff cache ---
# When we fetch today+tomorrow schedule, we also cache tomorrow's kickoff hours.
# At midnight, the new "today" matches yesterday's "tomorrow" — so we set the
# active window from cache (0 credits) instead of fetching again.
# A real API fetch still happens ~30min before window start to get fixture IDs.
cached_tomorrow_date: str = ""  # YYYY-MM-DD of cached tomorrow data
cached_tomorrow_kickoffs: list[float] = []  # kickoff hours (local) for that date
schedule_fixture_ids_loaded: bool = False  # True after real fetch got fixture IDs

# --- Friendly matches (v9.6.0) ---
# Track international friendlies where >=1 team plays in our 20 tracked leagues
# (via pre-seeded + auto-filled team ID cache).
# Skipped on busy days (>= FRIENDLY_BUSY_THRESHOLD main-league matches)
# to avoid spreading credits too thin.
FRIENDLY_BUSY_THRESHOLD = 10  # skip friendlies when 10+ league matches
active_friendly_fixtures: set[int] = set()  # fixture IDs of approved international friendlies
FRIENDLY_NAME_KEYWORDS = ("friendly", "friendlies")

# --- v9.5.6: Pre-seeded major club team IDs ---
# Solves the pre-season chicken-and-egg problem: during summer, big clubs
# only play friendlies, so their IDs never enter the cache from league fixtures.
# These IDs are stable in API-Football (api-sports.io) across seasons.
# The cache still auto-fills from actual league matches once seasons start.
PRESEEDED_TEAM_IDS = {
    # --- Premier League (39) ---
    42, 50, 40, 33, 49, 47, 34, 66, 51, 48, 65, 52, 36, 55,
    # Arsenal, Man City, Liverpool, Man Utd, Chelsea, Tottenham,
    # Newcastle, Aston Villa, Brighton, West Ham, Nott'm Forest, Crystal Palace, Fulham, Brentford

    # --- La Liga (140) ---
    541, 529, 530, 536, 548, 533, 531,
    # Real Madrid, Barcelona, Atletico, Sevilla, Real Sociedad, Villarreal, Real Betis

    # --- Bundesliga (78) ---
    157, 165, 168, 173, 161, 169, 172, 163,
    # Bayern, Dortmund, Leverkusen, Leipzig, Wolfsburg, Frankfurt, Stuttgart, M'gladbach

    # --- 2. Bundesliga (79) ---
    174, 192, 190, 189,
    # Hamburg, Schalke 04, Hertha, Hannover 96

    # --- Serie A (135) ---
    505, 489, 496, 492, 497, 487, 499, 502,
    # Inter, AC Milan, Juventus, Napoli, Roma, Lazio, Atalanta, Fiorentina

    # --- Ligue 1 (61) ---
    85, 81, 80, 91, 79, 84,
    # PSG, Marseille, Lyon, Monaco, Lille, Nice

    # --- Primeira Liga (94) ---
    211, 212, 228, 213,
    # Benfica, Porto, Sporting CP, Braga

    # --- Eredivisie (88) ---
    194, 197, 215,
    # Ajax, PSV, Feyenoord

    # --- Super Lig (203) ---
    263, 264, 262, 261,
    # Galatasaray, Fenerbahce, Besiktas, Trabzonspor

    # --- Austrian Bundesliga (169) ---
    556, 559, 558, 560,
    # Salzburg, Rapid Wien, Austria Wien, Sturm Graz

    # --- SuperLiga Serbia (283) ---
    2634, 2635,
    # Red Star, Partizan

    # --- HNL Croatia (210) ---
    1992, 1994,
    # Dinamo Zagreb, Hajduk Split

    # --- Czech First League (345) ---
    1021, 1023, 620,
    # Sparta Prague, Slavia Prague, Viktoria Plzen

    # --- Danish Superliga (119) ---
    281, 283, 282, 300,
    # Copenhagen, Brondby, Midtjylland, AGF

    # --- Veikkausliiga Finland (137) ---
    677, 679,
    # HJK Helsinki, KuPS

    # --- NB I Hungary (191) ---
    688, 689, 821,
    # Ferencvaros, Paks, Puskas Akademia

    # --- First League Bulgaria (357) ---
    # (populated from actual league fixtures — always in season during tracked months)
}
known_league_team_ids: set[int] = set(PRESEEDED_TEAM_IDS)


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
    # v9.5.8: Boosted range for better xG differentiation
    if prev and prev.get("last_xg"):
        try:
            xg = float(prev["last_xg"])
            score += int(xg * 100)
            # Bonus: high xG without SOT=3 yet means pressure is building
            if prev_sot < 3 and xg >= 0.8:
                score += 80  # likely about to hit SOT=3
            elif prev_sot < 3 and xg >= 0.5:
                score += 40
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
    """v9.6.2: Dynamic max monitored fixtures (batch-aware).

    With batched API requests (up to 20 fixtures per request),
    monitoring more fixtures costs the SAME as monitoring few —
    1 credit per batch, regardless of how many fixtures are in it.
    The cost driver is number of batch polls, not fixture count.

    Reserve only 3 credits (1 discovery + 2 safety). The old reserve
    of 10 was from pre-batching era and caused Max=0 when quota≤10,
    meaning the bot found candidates but refused to monitor any.
    """
    if quota_remaining is None:
        return BATCH_SIZE_LIMIT

    reserve = 3  # 1 discovery + 2 safety
    available = quota_remaining - reserve
    if available <= 0:
        return 0

    # With batching, cost is per-batch not per-fixture.
    # 1 credit = stats for ALL monitored fixtures (up to 20).
    # In EMERGENCY (600s interval), a 60-min window needs ~6 batch checks.
    # In NORMAL (240s interval), ~15 batch checks.
    # We need at least 2 batch checks to be useful (catch existing SOT≥3).
    if available < 2:
        return 0

    # Always return full batch limit — more fixtures monitored
    # doesn't cost more credits with batching.
    return BATCH_SIZE_LIMIT


def get_schedule_based_discovery_interval(budget_mode: str) -> int | None:
    """v9.6.1: Use the KNOWN schedule to predict when the next wave of
    matches enters the 20' window, and wake up just in time.

    This replaces blind polling with targeted discovery — the bot already
    knows all kickoff times from the schedule fetch, so it can calculate
    exactly when each wave hits 20' and sleep until then.

    Returns None if no future window entries exist (all waves processed).
    """
    global scheduled_window_entries
    now = time.time()
    buffer = 60  # wake up 60s before the 20' mark

    # Prune past entries
    scheduled_window_entries = [
        t for t in scheduled_window_entries if t > now - 120
    ]

    if not scheduled_window_entries:
        return None

    # Find the next future entry
    next_entry = min(t for t in scheduled_window_entries if t > now - buffer)
    gap = next_entry - now - buffer

    if gap <= 0:
        # v9.6.2: CONSUME this entry so it doesn't keep firing.
        # Old bug: returned 10s repeatedly, causing rapid-fire
        # discovery loops that burned credits (e.g. 6 disc in 51s).
        if next_entry in scheduled_window_entries:
            scheduled_window_entries.remove(next_entry)
        return 60

    # Cap at the mode's max interval (don't sleep longer than 30 min)
    max_interval = {
        "NORMAL": 1800,
        "CAREFUL": 2400,
        "STRICT": 3000,
        "EMERGENCY": 3600,
        "UNKNOWN": 1800,
    }.get(budget_mode, 1800)

    return min(max(60, int(gap)), max_interval)


def get_discovery_interval(
    budget_mode: str,
    has_tracked_live: bool,
    has_candidates: bool,
) -> int:
    """Seconds between live-fixture discovery calls.

    v9.6.2: Uses the pre-computed schedule to wake up right when
    kickoff waves enter the 20' window, instead of blind polling.
    Falls back to live-fixture-minute-based estimation if schedule is empty.

    v9.6.2: If we can't monitor anything (Max=0), don't rapid-fire
    discovery — it just burns credits finding candidates we can't check.
    """
    # v9.6.2 guard: if we can't monitor anything, back off massively.
    # No point discovering candidates if we can't check their stats.
    if get_max_fast_monitored() == 0:
        return 3600  # check once per hour, just in case quota resets

    # Priority 1: Use the schedule (known kickoff times)
    if not has_candidates:
        sched_interval = get_schedule_based_discovery_interval(budget_mode)
        if sched_interval is not None:
            return sched_interval

    # Priority 2: When candidates exist, still check schedule for new waves
    # but also allow a longer interval since stats are already running
    if has_candidates:
        sched_interval = get_schedule_based_discovery_interval(budget_mode)
        if sched_interval is not None and sched_interval < 300:
            # A new wave is about to hit 20' — discover soon
            return sched_interval
        # Otherwise use the standard interval
        return {
            "NORMAL": 1800,
            "CAREFUL": 2400,
            "STRICT": 3000,
            "EMERGENCY": 3600,
            "UNKNOWN": 1800,
        }.get(budget_mode, 1800)

    # Fallback: no schedule data, no tracked live
    return 1800


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
        if not is_tracked_match(fixture):
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
    # v9.7: Clean acceleration tracking
    for fid in list(accelerating_fixtures):
        if fid not in live_fixture_ids:
            accelerating_fixtures.discard(fid)
    # v9.7: Clean dead fixtures
    for fid in list(dead_fixtures):
        if fid not in live_fixture_ids:
            dead_fixtures.pop(fid, None)


def find_cached_fixture(fid: int):
    for f in cached_fixtures:
        if f["fixture"]["id"] == fid:
            return f
    return None


def is_first_signal_only_mode() -> bool:
    """v9.5.8: On busy days (>20 games), only send first signal per team
    then move on to cover more matches. On light days (<15), full tracking."""
    return total_matches_today > FIRST_SIGNAL_ONLY_THRESHOLD


def is_fixture_done_first_signal(fid: int) -> bool:
    """Check if this fixture has given at least one signal.
    In first-signal-only mode, such fixtures can be dropped from monitoring."""
    if not is_first_signal_only_mode():
        return False
    return fid in signaled_fixtures


def is_fixture_monitorable(fixture: dict) -> bool:
    """v9.5: strict 80' max — no late tracking beyond 80'.
    v9.5.8: In first-signal-only mode, skip already-signaled fixtures."""
    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        return False
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    if minute < MINUTE_MIN or minute > MINUTE_MAX:
        return False
    # v9.5.8: In first-signal-only mode, drop fixtures that already signaled
    fid = fixture["fixture"]["id"]
    if is_fixture_done_first_signal(fid):
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
    """v9.7: SOT-aware polling interval with acceleration tiers.

    v9.7 adds acceleration detection: if a fixture's best SOT increased
    in the last poll, it gets top priority (60s) to catch the next increase.
    This is the key improvement for catching the 0→1→2→3 transition.

    Tiers (per fixture, before both-teams-signaled 2x multiplier):
      Accelerating (SOT increased last poll): 60s
      SOT >= 3:                          90s
      SOT == 2 + fast window active:      120s
      SOT == 2:                          180s
      SOT == 1:                          base * 1.0 (faster than before)
      Unknown / 0 SOT:                   base * 1.5 (NORMAL) or 2.5 (CAREFUL+)
    """
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    team_signaled_count = sum(1 for (f, t) in signaled_teams if f == fid)
    both_teams_signaled = team_signaled_count >= 2

    # v9.7: Acceleration tier — highest priority
    if fid in accelerating_fixtures and not both_teams_signaled:
        interval = 60
    elif not has_state:
        interval = base_interval
    elif best_sot >= 3 and not both_teams_signaled:
        # v9.7: SOT>=3 now gets dedicated 90s tier (was mixed with fast window)
        interval = 90
    elif best_sot >= 1 and not both_teams_signaled and is_fast_sot_active(fid):
        interval = 120
    elif best_sot >= 2:
        interval = 180
    elif best_sot == 1:
        # v9.7: SOT=1 gets base interval directly (was 1.2x)
        interval = base_interval
    else:
        budget = get_budget_mode()
        if budget in ("CAREFUL", "STRICT", "EMERGENCY"):
            interval = int(base_interval * 2.5)
        else:
            interval = int(base_interval * 1.5)

    # Only apply 2x if BOTH teams in this fixture have signaled
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

    # v9.8: Check signal outcomes for all tracked fixtures in discovery
    # This catches goals from fixtures we stopped monitoring (first-signal-only)
    # and fixtures that ended between our last stats check.
    for f in cached_fixtures:
        if is_tracked_match(f):
            check_signal_outcomes(f)

    tracked = [f for f in cached_fixtures if is_tracked_match(f)]
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

    # v9.7: Check dead fixtures for score changes (momentum shift revival)
    revived = 0
    for fid, (dh, da) in list(dead_fixtures.items()):
        f = find_cached_fixture(fid)
        if not f:
            dead_fixtures.pop(fid, None)
            continue
        ch = f["goals"]["home"] or 0
        ca = f["goals"]["away"] or 0
        if ch != dh or ca != da:
            dead_fixtures.pop(fid, None)
            revived += 1
            log.info(
                f"  REVIVED: fixture {fid} "
                f"({f['teams']['home']['name']} vs {f['teams']['away']['name']}) "
                f"score {dh}-{da} -> {ch}-{ca} (momentum shift, re-monitoring)"
            )
    if revived:
        log.info(f"  -> {revived} dead fixture(s) revived")

    # Local pre-filter
    candidates = find_candidates(cached_fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN}-{MINUTE_MAX}')")

    # Build fixture -> best rank mapping
    max_fast = get_max_fast_monitored()
    fixture_best_rank: dict[int, int] = {}
    for fid, tid, rank in candidates:
        if fid not in fixture_best_rank or rank > fixture_best_rank[fid]:
            fixture_best_rank[fid] = rank

    # v9.7: Exclude dead fixtures (SOT=0 for both teams) from monitoring.
    # They waste credits polling matches with zero attacking pressure.
    if dead_fixtures:
        before = len(fixture_best_rank)
        fixture_best_rank = {
            fid: rank for fid, rank in fixture_best_rank.items()
            if fid not in dead_fixtures
        }
        skipped = before - len(fixture_best_rank)
        if skipped:
            log.info(f"  -> Skipped {skipped} dead fixture(s) (both teams SOT=0)")

    # v9.6.2: In FIRST_SIGNAL_ONLY mode, exclude already-signaled
    # fixtures from candidates. Old bug: signaled fixtures kept being
    # re-added ("0 retained, 1 new") then immediately removed by
    # is_fixture_monitorable(), creating an infinite discover→add→
    # stats→remove loop that burned 2 credits every 10 seconds.
    if is_first_signal_only_mode() and signaled_fixtures:
        before = len(fixture_best_rank)
        fixture_best_rank = {
            fid: rank for fid, rank in fixture_best_rank.items()
            if fid not in signaled_fixtures
        }
        skipped = before - len(fixture_best_rank)
        if skipped:
            log.info(f"  -> Skipped {skipped} already-signaled fixture(s)")

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


def check_signal_outcomes(fixture: dict) -> None:
    """v9.8: Check if any pending signals for this fixture resolved.

    A signal is HIT if the team scored within OUTCOME_WINDOW_MINUTES game minutes.
    A signal is MISS if the fixture ended or window expired without a goal.
    """
    fid = fixture["fixture"]["id"]
    status = fixture["fixture"]["status"]["short"]
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0

    for entry in signal_outcomes:
        if entry["fixture_id"] != fid or entry["resolved"]:
            continue

        # Current goals for the signaled team
        current_team_goals = home_goals if entry["is_home"] else away_goals
        goals_since_signal = current_team_goals - entry["goals_at_signal"]

        if goals_since_signal > 0:
            # Goal scored! HIT
            entry["outcome"] = "HIT"
            entry["goal_minute"] = minute
            entry["resolved"] = True
            mins_to_goal = minute - entry["game_minute"]
            log.info(
                f"  OUTCOME HIT: {entry['team_name']} scored at {minute}' "
                f"(+{mins_to_goal}' after {entry['tier']} signal at {entry['game_minute']}') "
                f"[{entry['league']}]"
            )
        elif status not in LIVE_STATUSES:
            # Fixture ended (FT, AET, etc.) with no goal — MISS
            entry["outcome"] = "MISS"
            entry["resolved"] = True
            log.info(
                f"  OUTCOME MISS: {entry['team_name']} no goal "
                f"(signal at {entry['game_minute']}', match ended {minute}') "
                f"[{entry['league']}]"
            )
        elif (minute - entry["game_minute"]) >= OUTCOME_WINDOW_MINUTES:
            # Window expired, still live, no goal — MISS
            entry["outcome"] = "MISS"
            entry["resolved"] = True
            log.info(
                f"  OUTCOME MISS: {entry['team_name']} no goal in {OUTCOME_WINDOW_MINUTES}' "
                f"(signal at {entry['game_minute']}', now {minute}') "
                f"[{entry['league']}]"
            )


def log_outcome_summary() -> None:
    """v9.8: Log hit-rate summary for all resolved signals today."""
    resolved = [e for e in signal_outcomes if e["resolved"]]
    if not resolved:
        return

    hits = [e for e in resolved if e["outcome"] == "HIT"]
    misses = [e for e in resolved if e["outcome"] == "MISS"]
    total = len(resolved)
    hit_rate = len(hits) / total * 100 if total else 0

    log.info("")
    log.info(f"=== SIGNAL OUTCOME SUMMARY: {total} signals, "
             f"{len(hits)} HIT, {len(misses)} MISS ({hit_rate:.0f}%) ===")

    # By tier
    for tier in ("VERY STRONG", "STRONG", "PRESSURE"):
        tier_signals = [e for e in resolved if e["tier"] == tier]
        if not tier_signals:
            continue
        t_hits = sum(1 for e in tier_signals if e["outcome"] == "HIT")
        t_total = len(tier_signals)
        avg_min_to_goal = None
        hit_entries = [e for e in tier_signals if e["outcome"] == "HIT" and e["goal_minute"]]
        if hit_entries:
            avg_min_to_goal = sum(
                e["goal_minute"] - e["game_minute"] for e in hit_entries
            ) / len(hit_entries)
        goal_info = f", avg +{avg_min_to_goal:.0f}' to goal" if avg_min_to_goal else ""
        log.info(
            f"  {tier}: {t_hits}/{t_total} ({t_hits/t_total*100:.0f}%){goal_info}"
        )

    # By signal number (1st vs 2nd vs 3rd+)
    log.info("")
    log.info("")


def process_fixture_stats(client: httpx.Client, fixture: dict) -> None:
    """Process one fixture from batched /fixtures?ids=... response.

    v9.5: New signal format with xG, top SOT player, red cards.
    Strict 80' cutoff. Tracks signaled fixtures for diversification.
    """
    fid = fixture["fixture"]["id"]

    # v9.8: Check outcome of pending signals FIRST (before early returns)
    # This catches: goals scored since last poll, fixtures that just ended
    check_signal_outcomes(fixture)

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

    # v9.5.7: Activate fast window at SOT>=1 to catch 1->2->3 transition
    if best_current_sot >= 1:
        team_sig_count = sum(1 for (f, t) in signaled_teams if f == fid)
        if team_sig_count < 2 and not is_fast_sot_active(fid):
            activate_fast_sot(fid)
    else:
        expire_fast_sot(fid)

    # --- v9.7: SOT acceleration tracking (fixture-level) ---
    # If the best SOT increased since last poll, mark as accelerating.
    # This gives 60s polling to catch the NEXT SOT increase.
    prev_best_sot = get_fixture_best_sot(fid)
    # (best_current_sot was computed above from this poll's data)
    # We'll update accelerating_fixtures AFTER the per-team loop
    # (since team_state gets updated there, which affects get_fixture_best_sot)
    _sot_increased = best_current_sot > prev_best_sot

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

        # --- v9.5.8: First-signal-only on busy days ---
        # If >20 games today, only send the FIRST signal per team, then move on.
        # This saves credits by not re-polling signaled fixtures.
        if is_first_signal_only_mode() and sig_count >= 1:
            log.info(
                f"  SKIP {tier}: {tname} - "
                f"{sot} SOT (first-signal-only mode, {total_matches_today} games today) "
                f"(fixture {fid})"
            )
            continue

        if sig_count >= 2:
            # v9.7 FIX: 3rd+ signal: need +2 SOT jump from LAST SIGNAL (not last poll)
            # BUG in v9.6.2: used state["last_sot"] which is the PREVIOUS POLL's SOT,
            # not the SOT at the last signal. Between polls, SOT can change without
            # triggering a signal, making the jump calculation wrong.
            # Example: signal #2 at SOT=4, poll at SOT=5 (no signal), poll at SOT=6.
            #   Old: jump = 6-5 = 1 (BLOCKED, wrong)
            #   New: jump = 6-4 = 2 (sent, correct)
            last_signal_sot = team_sig.get("sot_at_last_signal", 0)
            sot_jump = sot - last_signal_sot
            if sot_jump < 2:
                log.info(
                    f"  BLOCKED {tier}: {tname} - "
                    f"{sot} SOT (+{sot_jump} from last signal, need +2) "
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
                "count": 1, "goals_at_last_signal": goals_now,
                "sot_at_last_signal": sot,
            }
        else:
            signaled_teams[(fid, tid)]["count"] = sig_count + 1
            signaled_teams[(fid, tid)]["goals_at_last_signal"] = goals_now
            signaled_teams[(fid, tid)]["sot_at_last_signal"] = sot
        signaled_fixtures.add(fid)

        league = LEAGUE_IDS.get(
            fixture["league"]["id"], fixture["league"].get("name", "?")
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

        # v9.8: Record signal for outcome tracking
        opp_goals = (sa if is_home_sg else sh)
        signal_outcomes.append({
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": league,
            "signal_time": time.time(),
            "game_minute": minute,
            "sot": sot,
            "tier": tier,
            "goals_at_signal": goals_now,
            "opponent_goals_at_signal": opp_goals,
            "is_home": is_home_sg,
            "outcome": None,  # None=pending, "HIT", "MISS"
            "goal_minute": None,
            "resolved": False,
        })

        # v9.5.8: In first-signal-only mode, remove fixture from monitoring
        # immediately after signaling so we move on to other matches.
        if is_first_signal_only_mode():
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            log.info(
                f"  FIRST-SIGNAL-DONE: fixture {fid} removed from monitoring "
                f"(moving on, {len(fast_monitored)} still monitored)"
            )

    # --- v9.7: Update acceleration flag AFTER all teams processed ---
    # (must be after team_state updates so get_fixture_best_sot is accurate)
    if _sot_increased:
        accelerating_fixtures.add(fid)
        log.debug(f"  Accelerating: fixture {fid} (SOT {prev_best_sot}->{best_current_sot})")
    else:
        accelerating_fixtures.discard(fid)

    # --- v9.7: Dead fixture detection (both teams SOT=0) ---
    # Stop wasting credits polling matches with zero attacking pressure.
    # Will be revived by discovery if score changes (momentum shift).
    if best_current_sot == 0 and prev_best_sot == 0:
        # Both polls show 0 SOT for both teams — this match is dead.
        hg = fixture["goals"]["home"] or 0
        ag = fixture["goals"]["away"] or 0
        dead_fixtures[fid] = (hg, ag)
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        accelerating_fixtures.discard(fid)
        log.info(
            f"  DEAD: fixture {fid} ({home['name']} vs {away['name']}) "
            f"both teams SOT=0 at {minute}' — stopped polling "
            f"(will revive on score change)"
        )


def _update_cached_fixtures(refreshed: list[dict]) -> set[int]:
    """Update cached_fixtures list with fresh data from /fixtures?ids= response.
    Returns set of updated fixture IDs."""
    global cached_fixtures
    refreshed_ids = set()
    for rf in refreshed:
        rf_id = rf["fixture"]["id"]
        for i, cf in enumerate(cached_fixtures):
            if cf["fixture"]["id"] == rf_id:
                cached_fixtures[i] = rf
                refreshed_ids.add(rf_id)
                break
        else:
            cached_fixtures.append(rf)
            refreshed_ids.add(rf_id)
    return refreshed_ids


def check_monitored_stats(
    client: httpx.Client,
    fixture_ids: list[int],
) -> bool:
    """v9.7: Unified stats fetch — ONE /fixtures?ids= call replaces everything.

    v9.7 Architecture:
    ====================
    1. Single /fixtures?ids=X-Y-Z call (1 credit) — gets fresh fixture data.
    2. Check if the response includes statistics (API-plan dependent).
    3. If YES: process all fixtures directly from the one response.
       Credit cost: 1 credit for ALL fixtures (was 1+ N before).
    4. If NO: fall back to individual /fixtures/statistics?fixture=X calls.
       Credit cost: 1 + N (same as v9.6.2 in NORMAL mode, but saves 1
       credit in CAREFUL+ modes where v9.6.2 skipped the refresh).

    Credit comparison for 10 monitored fixtures:
    - v9.6.2 NORMAL:  1 (refresh) + 10 (stats) = 11 credits/cycle
    - v9.6.2 CAREFUL: 0 (skip refresh) + 10 (stats) = 10 credits/cycle
    - v9.7 with stats:  1 credit/cycle (11x saving!)
    - v9.7 without stats: 1 + 10 = 11 credits/cycle (same as NORMAL)

    The key improvement: we test ONCE whether the API returns stats
    in /fixtures?ids= responses, then use the optimal path forever.

    Always updates last_stats_check to prevent infinite re-polling.
    """
    global last_stats_check, _ids_endpoint_has_stats

    # --- Pre-filter: remove invalid/unmonitorable fixtures ---
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

    now = time.time()
    any_success = False
    valid_set = set(valid_ids)

    # ================================================================
    # STEP 1: Single /fixtures?ids= call (always — gets fresh data)
    # ================================================================
    ids_str = "-".join(str(fid) for fid in valid_ids)
    refreshed_fixtures = {}  # fid -> fixture dict from this call

    try:
        batch_data = api_get(client, "/fixtures", {"ids": ids_str})
        batch_response = batch_data.get("response", [])

        # Update cached fixtures with fresh score/minute/data
        refreshed_ids = _update_cached_fixtures(batch_response)
        log.info(
            f"  Batch refresh: {len(refreshed_ids)} fixture(s) updated"
            f"{f' (stats embedded!)' if _ids_endpoint_has_stats is True else ''}"
            f"{f' (testing for stats...)' if _ids_endpoint_has_stats is None else ''}"
        )

        # Index fixtures by ID for easy lookup
        for rf in batch_response:
            rf_id = rf["fixture"]["id"]
            if rf_id in valid_set:
                refreshed_fixtures[rf_id] = rf

    except Exception as e:
        log.warning(f"  Batch /fixtures?ids= failed: {e}")
        # Mark all as checked to prevent retry storm
        for fid in valid_ids:
            last_stats_check[fid] = now
        return False

    # ================================================================
    # STEP 2: Check if statistics are embedded in the response
    # ================================================================
    if _ids_endpoint_has_stats is None and batch_response:
        # First time: test whether any fixture has statistics
        has_any_stats = False
        for rf in batch_response:
            stats = rf.get("statistics")
            if stats and isinstance(stats, list) and len(stats) > 0:
                # Verify it looks like real statistics data
                # (should have team objects with statistics arrays)
                first_entry = stats[0] if stats else {}
                if "team" in first_entry or "statistics" in first_entry:
                    has_any_stats = True
                    break

        _ids_endpoint_has_stats = has_any_stats
        if has_any_stats:
            log.info(
                "  *** v9.7 DETECTION: /fixtures?ids= INCLUDES statistics! ***"
                " *** Single-call mode ACTIVE (massive credit saving) ***"
            )
        else:
            log.info(
                "  v9.7 DETECTION: /fixtures?ids= does NOT include statistics."
                " Using individual /fixtures/statistics calls as fallback."
            )

    # ================================================================
    # STEP 3A: If stats are embedded — process directly (1 credit total!)
    # ================================================================
    if _ids_endpoint_has_stats is True:
        for fid in valid_ids:
            rf = refreshed_fixtures.get(fid)
            if not rf:
                last_stats_check[fid] = now
                continue

            stats = rf.get("statistics")
            if not stats:
                log.debug(f"  Fixture {fid}: no stats in batch response")
                last_stats_check[fid] = now
                continue

            try:
                process_fixture_stats(client, rf)
                any_success = True
            except Exception as e:
                log.warning(f"  Stats processing failed for fixture {fid}: {e}")

            last_stats_check[fid] = now

        log.info(
            f"  v9.7 single-call: processed {len(valid_ids)} fixture(s) "
            f"for 1 credit (stats embedded in /fixtures?ids= response)"
        )
        return any_success

    # ================================================================
    # STEP 3B: Fallback — individual /fixtures/statistics calls
    # ================================================================
    # This path costs 1 (batch above) + N (individual) credits.
    # In v9.6.2 NORMAL mode it was also 1+ N, so same cost.
    # In v9.6.2 CAREFUL+ mode it was 0+ N (skipped refresh), so this
    # costs 1 extra credit but gives fresher score/minute data.
    log.info(
        f"  v9.7 fallback: {len(valid_ids)} fixture(s), "
        f"fetching individual statistics ({1 + len(valid_ids)} credits this cycle)"
    )

    for fid in valid_ids:
        try:
            stats_data = api_get(
                client, "/fixtures/statistics", {"fixture": fid}
            )

            stats_response = stats_data.get("response", [])
            if not stats_response:
                log.debug(f"  Fixture {fid}: empty statistics response")
                last_stats_check[fid] = now
                continue

            any_success = True

            # Use the FRESH fixture data from the batch call (not stale cache)
            rf = refreshed_fixtures.get(fid)
            if not rf:
                rf = find_cached_fixture(fid)
            if not rf:
                last_stats_check[fid] = now
                continue

            # Create merged fixture with statistics injected
            merged = {
                "fixture": rf["fixture"],
                "teams": rf["teams"],
                "goals": rf["goals"],
                "league": rf["league"],
                "statistics": stats_response,
            }
            process_fixture_stats(client, merged)

        except Exception as e:
            log.warning(f"  Stats failed for fixture {fid}: {e}")

        last_stats_check[fid] = now

    return any_success


# ============================================================
# LEAGUE FILTERING (v9.5.4)
# ============================================================

def is_tracked_match(fixture: dict) -> bool:
    """Check if a fixture should be tracked (main 20 leagues OR approved international friendly)."""
    lid = fixture["league"]["id"]
    fid = fixture["fixture"]["id"]
    return lid in LEAGUE_IDS or fid in active_friendly_fixtures


# ============================================================
# DYNAMIC ACTIVE HOURS (v9.5.4)
# ============================================================

def fetch_daily_active_hours(client: httpx.Client) -> bool:
    """Fetch today's fixtures and compute the active monitoring window.
    
    Calls /fixtures?date=YYYY-MM-DD twice per fetch (today + tomorrow).
    Re-checks every 3 hours to catch late-added fixtures and widen window.
    Sets dynamic_active_start/end based on actual kickoff times.
    Returns True if there are matches to monitor today.
    """
    global dynamic_active_start, dynamic_active_end
    global schedule_date, schedule_no_matches, schedule_last_fetch
    global total_matches_today
    global scheduled_window_entries
    global schedule_fixture_ids_loaded
    global cached_tomorrow_date, cached_tomorrow_kickoffs
    
    today_bulgaria = datetime.now(BULGARIA_TZ)
    today_str = today_bulgaria.strftime("%Y-%m-%d")
    tomorrow_str = (today_bulgaria + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # v9.7.2: Night gate — never burn credits checking schedule at night
    # European tracked leagues don't play between 01:00-10:00 Bulgaria time.
    # Just return cached result. If new day started at 3 AM, return False
    # (no matches at 3 AM) — will properly fetch when morning comes.
    local_hour = today_bulgaria.hour
    if NIGHT_HOUR_START <= local_hour < NIGHT_HOUR_END:
        if schedule_date == today_str:
            # Same day, already fetched during evening — return cached
            return not schedule_no_matches
        # New day but still night — check if we have cached tomorrow data
        if cached_tomorrow_date == today_str and cached_tomorrow_kickoffs is not None:
            # We cached this day's kickoffs from yesterday's fetch!
            # Set window from cache (0 credits) — real fetch happens before window
            if cached_tomorrow_kickoffs:
                earliest = min(cached_tomorrow_kickoffs)
                latest = max(cached_tomorrow_kickoffs)
                window_start = max(0, int(earliest - (30 / 60)))
                window_end = min(24, int(latest + (100 / 60)) + (1 if (latest + (100 / 60)) % 1 > 0 else 0))
                # Only set window if it makes sense (not stale data pointing to wrong day)
                dynamic_active_start = window_start
                dynamic_active_end = window_end
                total_matches_today = len(cached_tomorrow_kickoffs)
                schedule_no_matches = False
                schedule_date = today_str  # mark as "handled" so we don't re-trigger
                schedule_fixture_ids_loaded = False  # but fixture IDs NOT loaded yet
                log.info(
                    f"v9.7.2 CACHE HIT: {today_str} window from yesterday's fetch — "
                    f"{len(cached_tomorrow_kickoffs)} matches, "
                    f"kickoffs {int(earliest):02d}:{int((earliest % 1) * 60):02d}-"
                    f"{int(latest):02d}:{int((latest % 1) * 60):02d}, "
                    f"active {window_start:02d}:00-{window_end:02d} Bulgaria (0 credits)"
                )
                return True
            else:
                # Cached as "no matches" — truly no games today
                schedule_date = today_str
                schedule_no_matches = True
                schedule_fixture_ids_loaded = False
                log.info(
                    f"v9.7.2 CACHE HIT: {today_str} no matches (from yesterday's fetch, 0 credits)"
                )
                return False
        # No cache available (first run or cache miss) — return False, will fetch at NIGHT_HOUR_END
        return False
    
    # Already fetched today — re-check every 6h to catch late-added fixtures
    # v9.7.2: BUT always fetch if fixture IDs not loaded (cache-set window needs real data)
    if (schedule_date == today_str
            and schedule_fixture_ids_loaded
            and (time.time() - schedule_last_fetch) < SCHEDULE_RECHECK_INTERVAL):
        return not schedule_no_matches
    
    if schedule_date == today_str and schedule_fixture_ids_loaded:
        is_recheck = True
        log.info(f"Re-checking schedule for {today_str} (last fetch >6h ago, looking for late additions)...")
    elif schedule_date == today_str and not schedule_fixture_ids_loaded:
        is_recheck = False
        log.info(f"v9.7.2: Loading fixture IDs for cached window {today_str} (needed for monitoring)...")
    else:
        is_recheck = False
    
    log.info(f"Fetching daily schedule for {today_str} + {tomorrow_str} (2 API calls)...")
    
    try:
        # Query BOTH today and tomorrow (Bulgaria time) to handle timezone edge cases.
        # A match at 22:00 UTC on Aug 11 is 01:00 Aug 12 Bulgaria — API has it
        # under Aug 11, so querying only Aug 12 would miss it.
        all_fixtures = []
        for query_date in [today_str, tomorrow_str]:
            try:
                data = api_get(client, "/fixtures", {"date": query_date})
                all_fixtures.extend(data.get("response", []))
            except Exception as e:
                log.warning(f"  Schedule fetch for {query_date} failed: {e}")
        
        # DEBUG: Log all unseen league IDs (for adding qualifier rounds etc.)
        unseen_leagues = {}
        for f in all_fixtures:
            lid = f["league"]["id"]
            lname = f["league"].get("name", "?")
            if lid not in LEAGUE_IDS and lid not in unseen_leagues:
                unseen_leagues[lid] = lname
        if unseen_leagues:
            log.info(f"  DEBUG: Unseen league IDs in today's fetch ({len(unseen_leagues)}):")
            for lid, lname in sorted(unseen_leagues.items()):
                # Count how many fixtures from this league
                count = sum(1 for f in all_fixtures if f["league"]["id"] == lid)
                log.info(f"    league_id={lid} -> \"{lname}\" ({count} fixture(s))")

        # Two-pass approach:
        # Pass 1: scan ALL fixtures (any date) to collect team IDs from tracked
        #         leagues into persistent cache — this is how we learn which teams
        #         are "big" for friendly filtering.
        # Pass 2: filter by today's Bulgaria date for kickoff window.
        for f in all_fixtures:
            lid = f["league"]["id"]
            if lid in LEAGUE_IDS:
                known_league_team_ids.add(f["teams"]["home"]["id"])
                known_league_team_ids.add(f["teams"]["away"]["id"])

        # Pass 2: build kickoff list and friendly candidates for today (Bulgaria date)
        kickoff_hours = []
        friendly_candidates = []  # (fixture, hour, league_name)

        for f in all_fixtures:
            lid = f["league"]["id"]
            league_name = f["league"].get("name", "").lower()

            date_str = f["fixture"]["date"]
            try:
                kickoff_utc = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                if kickoff_local.strftime("%Y-%m-%d") != today_str:
                    continue
                hour = kickoff_local.hour + kickoff_local.minute / 60.0
            except Exception:
                continue

            if lid in LEAGUE_IDS:
                kickoff_hours.append(hour)
            elif any(kw in league_name for kw in FRIENDLY_NAME_KEYWORDS):
                friendly_candidates.append((f, hour, f["league"].get("name", "?")))

        # v9.7.2: Cache tomorrow's kickoff hours from this fetch
        # We already have tomorrow's fixtures in all_fixtures, just extract them.
        # Tomorrow at midnight, we'll use this cache to set the window for 0 credits.
        tomorrow_ko = []
        for f in all_fixtures:
            lid = f["league"]["id"]
            if lid not in LEAGUE_IDS:
                continue
            date_str = f["fixture"]["date"]
            try:
                kickoff_utc = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                if kickoff_local.strftime("%Y-%m-%d") != tomorrow_str:
                    continue
                hour = kickoff_local.hour + kickoff_local.minute / 60.0
                tomorrow_ko.append(hour)
            except Exception:
                continue
        if tomorrow_ko:
            cached_tomorrow_date = tomorrow_str
            cached_tomorrow_kickoffs = sorted(tomorrow_ko)
            log.info(
                f"  Cached tomorrow ({tomorrow_str}): {len(tomorrow_ko)} matches, "
                f"kickoffs {int(tomorrow_ko[0]):02d}:{int((tomorrow_ko[0] % 1) * 60):02d}-"
                f"{int(tomorrow_ko[-1]):02d}:{int((tomorrow_ko[-1] % 1) * 60):02d} Bulgaria"
            )
        else:
            cached_tomorrow_date = tomorrow_str
            cached_tomorrow_kickoffs = []
            log.info(f"  Tomorrow ({tomorrow_str}): no tracked league matches")

        schedule_date = today_str
        schedule_last_fetch = time.time()
        schedule_fixture_ids_loaded = True  # v9.7.2: real fetch got fixture IDs

        # Include international club friendlies where at least one team
        # is in our 20 tracked leagues, UNLESS it's a busy day.
        global active_friendly_fixtures
        if len(kickoff_hours) < FRIENDLY_BUSY_THRESHOLD and friendly_candidates:
            added = 0
            skipped = 0
            for f, hour, name in friendly_candidates:
                if (
                    f["teams"]["home"]["id"] in known_league_team_ids
                    or f["teams"]["away"]["id"] in known_league_team_ids
                ):
                    active_friendly_fixtures.add(f["fixture"]["id"])
                    kickoff_hours.append(hour)
                    added += 1
                    log.info(
                        f"    Friendly OK: {f['teams']['home']['name']} vs "
                        f"{f['teams']['away']['name']} ({name})"
                    )
                else:
                    skipped += 1
            if added:
                log.info(
                    f"  Friendly filter: {added} friendly(ies) added, "
                    f"{skipped} lower-division skipped "
                    f"({len(kickoff_hours) - added} league + {added} friendly, "
                    f"team cache: {len(known_league_team_ids)} IDs)"
                )
        else:
            if active_friendly_fixtures:
                reason = "busy day" if len(kickoff_hours) >= FRIENDLY_BUSY_THRESHOLD else "no friendly candidates"
                log.info(f"  Clearing {len(active_friendly_fixtures)} friendly fixture IDs ({reason})")
            active_friendly_fixtures = set()
        
        # v9.6.1: Compute scheduled 20' entry times (UTC timestamps)
        # Group kickoffs by minute to find waves, then compute when each
        # wave enters the 20' monitoring window.
        from collections import Counter
        kickoff_wave_minutes = []
        for f in all_fixtures:
            lid = f["league"]["id"]
            fid = f["fixture"]["id"]
            is_tracked = lid in LEAGUE_IDS or fid in active_friendly_fixtures
            if not is_tracked:
                continue
            date_str = f["fixture"]["date"]
            try:
                kickoff_utc = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                if kickoff_local.strftime("%Y-%m-%d") != today_str:
                    continue
                # Round to nearest minute to group wave kickoffs
                wave_minute = int(kickoff_utc.timestamp() // 60)
                kickoff_wave_minutes.append(wave_minute)
            except Exception:
                continue

        # Deduplicate waves and compute 20' entry times
        scheduled_window_entries = sorted(set(
            (wave * 60) + (MINUTE_MIN * 60)  # kickoff_ts + 20 minutes
            for wave in set(kickoff_wave_minutes)
        ))
        if scheduled_window_entries:
            entries_bg = [
                datetime.fromtimestamp(t, BULGARIA_TZ).strftime("%H:%M")
                for t in scheduled_window_entries
            ]
            log.info(
                f"  20' entries (schedule): {entries_bg}"
                f" ({len(scheduled_window_entries)} waves)"
            )
        
        if not kickoff_hours:
            schedule_no_matches = True
            log.info(
                f"  No tracked league matches today ({today_str}) — "
                f"will re-check in {SCHEDULE_RECHECK_INTERVAL // 3600}h (2 credits used)"
            )
            return False
        
        schedule_no_matches = False
        earliest = min(kickoff_hours)
        latest = max(kickoff_hours)
        
        # Buffer: 30 min before earliest kickoff (be ready at 20'),
        # 100 min after latest kickoff (cover full 90' match + buffer)
        window_start = earliest - (30 / 60)
        window_end = latest + (100 / 60)
        
        # Convert to integer hours, floor start / ceil end
        new_start = max(0, int(window_start))
        new_end = min(24, int(window_end) + (1 if window_end % 1 > 0 else 0))
        
        if is_recheck and not schedule_no_matches:
            # Re-check WITH existing matches: only WIDEN (never shrink)
            if new_start < dynamic_active_start:
                log.info(f"  Window widened start: {dynamic_active_start:02d}:00 -> {new_start:02d}:00")
                dynamic_active_start = new_start
            if new_end > dynamic_active_end:
                log.info(f"  Window widened end: {dynamic_active_end:02d}:00 -> {new_end:02d}:00 (late-added fixture detected)")
                dynamic_active_end = new_end
        else:
            # First fetch of the day, OR first time finding matches today:
            # set window directly from actual kickoffs.
            old_start, old_end = dynamic_active_start, dynamic_active_end
            dynamic_active_start = new_start
            dynamic_active_end = new_end
            if is_recheck:
                log.info(
                    f"  Window corrected: {old_start:02d}:00-{old_end:02d}:00 -> "
                    f"{new_start:02d}:00-{new_end:02d}:00 (matches found after no-match period)"
                )
        
        match_count = len(kickoff_hours)
        total_matches_today = match_count
        log.info(
            f"  {match_count} matches: kickoffs "
            f"{int(earliest):02d}:{int((earliest % 1) * 60):02d} - "
            f"{int(latest):02d}:{int((latest % 1) * 60):02d} Bulgaria"
            f"{f' -> FIRST SIGNAL ONLY mode' if match_count > FIRST_SIGNAL_ONLY_THRESHOLD else f' -> FULL TRACKING mode' if match_count < FULL_TRACKING_THRESHOLD else ''}"
        )
        log.info(
            f"  Active window: "
            f"{dynamic_active_start:02d}:00 - {dynamic_active_end:02d}:00 Bulgaria"
            f"{'' if is_recheck else f' (saves {max(0, dynamic_active_start - ACTIVE_HOUR_START_FALLBACK) + max(0, ACTIVE_HOUR_END_FALLBACK - dynamic_active_end)}h of idle polling)'}"
        )
        return True
        
    except Exception as e:
        log.warning(
            f"Schedule fetch failed: {e}, "
            f"using fallback {ACTIVE_HOUR_START_FALLBACK}:00-{ACTIVE_HOUR_END_FALLBACK}:00"
        )
        dynamic_active_start = ACTIVE_HOUR_START_FALLBACK
        dynamic_active_end = ACTIVE_HOUR_END_FALLBACK
        schedule_date = today_str
        schedule_last_fetch = time.time()
        schedule_no_matches = False
        schedule_fixture_ids_loaded = False  # v9.7.2: fallback didn't get real data
        return True


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v9.8 — Signal outcome tracking (backtesting)")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (round-robin for rate-limit resilience, NOT quota expansion)")
    log.info("")
    log.info("v9.7 CHANGES:")
    log.info("  P0 #1: Unified /fixtures?ids= call (auto-detects embedded statistics)")
    log.info("          If API returns stats: 1 credit/cycle for ALL fixtures (was 1+N)")
    log.info("          If not: falls back to individual stats calls (same as before)")
    log.info("  P0 #2: Fixed sot_at_last_signal bug (3rd+ signals used wrong SOT reference)")
    log.info("  P1:    SOT-acceleration polling (60s for fixtures where SOT just increased)")
    log.info("  P2:    Schedule recheck 3h -> 6h (saves ~4 credits/day)")
    log.info("")
    log.info("v9.7.1 CHANGES:")
    log.info(f"  Night gate: {NIGHT_HOUR_START:02d}:00-{NIGHT_HOUR_END:02d}:00 Bulgaria = ZERO schedule checks (saves 2-4 credits/night)")
    log.info("  Smart sleep: dead hours sleep until next event (not 30min wake-ups)")
    log.info("")
    log.info("v9.7.2 CHANGES:")
    log.info("  Tomorrow cache: evening fetch caches next day's kickoffs (0 credits at midnight)")
    log.info("  Window set from cache at midnight, real fetch only when window approaches")
    log.info("")
    log.info("v9.8 CHANGES:")
    log.info(f"  Signal outcome tracking: every signal tagged HIT/MISS within {OUTCOME_WINDOW_MINUTES}'")
    log.info("  Hit rate by tier in status line + daily summary when matches end")
    log.info("  Discovery checks outcomes for ALL tracked fixtures (catches removed-from-monitoring)")
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
    log.info("Architecture: discovery + unified batch stats")
    log.info(f"  Active: DYNAMIC from daily schedule (fallback {ACTIVE_HOUR_START_FALLBACK}:00-{ACTIVE_HOUR_END_FALLBACK}:00)")
    log.info(f"  Team cache: {len(known_league_team_ids)} IDs ({len(PRESEEDED_TEAM_IDS)} pre-seeded + auto-fill from fixtures)")
    log.info(f"  Friendly tracking: ON when <{FRIENDLY_BUSY_THRESHOLD} league matches (20-league teams only, skip on busy days)")
    log.info(f"  Schedule recheck: every {SCHEDULE_RECHECK_INTERVAL // 3600}h (was 3h)")
    log.info("")
    log.info("Quota: subscription-level from API headers (not additive)")
    log.info("  >=51: NORMAL  (240s base stats)")
    log.info("  26-50: CAREFUL (300s base stats)")
    log.info("  11-25: STRICT  (420s base stats)")
    log.info("  1-10: EMERGENCY (600s base stats)")
    log.info("  0: STOP")
    log.info("")
    log.info(f"v9.5.8 Adaptive mode:")
    log.info(f"  >{FIRST_SIGNAL_ONLY_THRESHOLD} games: FIRST SIGNAL ONLY (signal -> move on, cover more games)")
    log.info(f"  <{FULL_TRACKING_THRESHOLD} games: FULL TRACKING (multiple signals as before)")
    log.info(f"  {FULL_TRACKING_THRESHOLD}-{FIRST_SIGNAL_ONLY_THRESHOLD} games: NORMAL (signal limits apply)")
    log.info(f"  xG boost: high xG + low SOT = priority (catching pressure build-up)")
    log.info("")
    log.info("v9.7 Polling tiers (per fixture, before 2x both-signaled multiplier):")
    log.info("  Accelerating (SOT up last poll): 60s")
    log.info("  SOT >= 3:                       90s")
    log.info("  SOT == 2 + fast window:          120s")
    log.info("  SOT == 2:                       180s")
    log.info("  SOT == 1:                       base interval (240-600s)")
    log.info("  SOT == 0 / unknown:             1.5x base (NORMAL) or 2.5x (CAREFUL+)")
    log.info("  Both teams signaled:            2x all intervals")
    log.info("")
    log.info("Signal limits — 1st always, 2nd +1 SOT, 3rd+ +2 from LAST SIGNAL & 0 goals")
    log.info("v9.7 fix: 3rd+ jump measured from sot_at_last_signal, not last poll")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        while True:
            now = time.time()

            # --- Fetch daily schedule (1 call/day, re-fetches on date change) ---
            has_matches = fetch_daily_active_hours(client)
            
            if not has_matches:
                # v9.7.1: Smart sleep — calculate how long until next meaningful wake
                now_bg = datetime.now(BULGARIA_TZ)
                h = now_bg.hour
                # If night: sleep until NIGHT_HOUR_END
                if NIGHT_HOUR_START <= h < NIGHT_HOUR_END:
                    wake_at = now_bg.replace(hour=NIGHT_HOUR_END, minute=0, second=0, microsecond=0)
                    sleep_s = int((wake_at - now_bg).total_seconds())
                else:
                    sleep_s = 1800  # 30 min during daytime no-matches
                log.info(
                    f"No tracked matches today, sleeping {sleep_s // 60}m..."
                )
                time.sleep(sleep_s)
                continue
            
            # --- Dead hours (zero API cost) ---
            # Uses dynamic window from today's schedule
            local_hour = datetime.now(BULGARIA_TZ).hour
            if not (dynamic_active_start <= local_hour < dynamic_active_end):
                # v9.7.1: Smart sleep — sleep until window start (or night end if overnight)
                now_bg = datetime.now(BULGARIA_TZ)
                if NIGHT_HOUR_START <= now_bg.hour < NIGHT_HOUR_END:
                    # Deep night — sleep until morning
                    wake_at = now_bg.replace(hour=NIGHT_HOUR_END, minute=0, second=0, microsecond=0)
                else:
                    # Pre-window or post-window — sleep until window starts
                    # (if window already passed today, it means tomorrow's schedule isn't loaded yet)
                    wake_at = now_bg.replace(hour=dynamic_active_start, minute=0, second=0, microsecond=0)
                    # If window start already passed today, next check is 30 min (schedule might update)
                    if wake_at <= now_bg:
                        wake_at = now_bg + timedelta(minutes=30)
                sleep_s = int((wake_at - now_bg).total_seconds())
                sleep_s = max(sleep_s, 60)  # minimum 1 min
                log.info(
                    f"Dead hours (local {local_hour}:00, "
                    f"active {dynamic_active_start:02d}:00-{dynamic_active_end:02d}:00 Bulgaria), "
                    f"sleeping {sleep_s // 60}m {sleep_s % 60}s..."
                )
                time.sleep(sleep_s)
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
                [f for f in cached_fixtures if is_tracked_match(f)]
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
                         if is_tracked_match(f)]
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
                 if is_tracked_match(f)]
            ) if cached_fixtures else 0

            # Status summary
            mode_label = get_budget_mode()
            if is_first_signal_only_mode():
                mode_label += " +FIRST_ONLY"
            elif total_matches_today < FULL_TRACKING_THRESHOLD:
                mode_label += " +FULL"
            # v9.7: Show stats mode and acceleration in status line
            extra_info = ""
            if _ids_endpoint_has_stats is True:
                extra_info += " | Stats:1call"
            elif _ids_endpoint_has_stats is False:
                extra_info += " | Stats:N+1"
            if accelerating_fixtures:
                accel_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in accelerating_fixtures]
                extra_info += f" | Accel: {', '.join(accel_parts[:3])}{'...' if len(accel_parts) > 3 else ''}"
            if dead_fixtures:
                extra_info += f" | Dead: {len(dead_fixtures)}"
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
            # v9.8: Show outcome tracking in status line
            resolved_outcomes = [e for e in signal_outcomes if e["resolved"]]
            pending_outcomes = [e for e in signal_outcomes if not e["resolved"]]
            hits_today = sum(1 for e in resolved_outcomes if e["outcome"] == "HIT")
            misses_today = sum(1 for e in resolved_outcomes if e["outcome"] == "MISS")
            outcome_str = ""
            if resolved_outcomes:
                hr = hits_today / len(resolved_outcomes) * 100
                outcome_str = f" | Outcomes: {hits_today}H/{misses_today}M ({hr:.0f}%)"
            if pending_outcomes:
                outcome_str += f" | Pending: {len(pending_outcomes)}"

            log.info(
                f"Quota: {quota_remaining}/{quota_limit} | "
                f"Mode: {mode_label} | "
                f"Tracked: {tracked_count} | Mon: {len(fast_monitored)} | "
                f"Sig: {len(signaled_teams)}teams/{len(signaled_fixtures)}fix/{total_signals}sent | "
                f"Keys: {healthy_key_count()}/{len(API_KEYS)} | "
                f"Next disc: {int(next_disc_in)}s | Next stats: {stats_str} | "
                f"Sleep: {int(sleep_time)}s | Reqs: {request_count} | "
                f"Signals: {len(signals_sent)}{fast_window_info}{extra_info}{outcome_str}"
            )

            # v9.8: Log outcome summary when all matches done
            if (resolved_outcomes and not pending_outcomes
                    and not has_tracked_live and not fast_monitored):
                log_outcome_summary()
                # Clear old outcomes after summary (keep memory clean)
                signal_outcomes.clear()

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
