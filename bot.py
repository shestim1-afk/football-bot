import json
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

# v10.19: Suppress httpx INFO logs — Telegram getUpdates fires every 2min
# during dead hours, flooding logs with useless "HTTP/1.1 200 OK" lines.
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    88: "Eredivisie", 203: "Super Lig",
    283: "Liga I (Romania)", 210: "HNL (Croatia)", 345: "Czech First League",
    119: "Danish Superliga", 137: "Veikkausliiga (Finland)", 191: "NB I (Hungary)",
}

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}

# Active monitoring window: dynamically computed from daily schedule
# Falls back to 14:00-23:00 if schedule fetch fails
BULGARIA_TZ = ZoneInfo("Europe/Sofia")
ACTIVE_HOUR_START_FALLBACK = 14  # fallback
ACTIVE_HOUR_END_FALLBACK = 23    # fallback

# v10.10: 21'-61' window — data shows 68% WR here, 0-20' is noisy, 62'+ drops to ~35%
MINUTE_MIN = 21
MINUTE_MAX = 61

# v10.15: Pre-window monitoring — start polling from minute 1.
# v10.20: Raised from 12→20 so min_allowed = 21-20 = 1'.
# Fixes chicken-and-egg: bot needs to poll stats to discover high early SOT,
# but the old 9' floor prevented polling until SOT was already known.
# Credit cost is ~0: stats are batched (1 credit per up to 20 fixtures),
# so adding early fixtures to an existing batch costs nothing extra.
# Signal gates (PRE-WINDOW block, EARLY GATE GPS≥75) still prevent noise.
PRE_WINDOW_MINUTES = 20  # discover & poll from 1', signal from 21'

# v10.13.2: Extended window for SOT-accelerating fixtures
# If a fixture shows SOT acceleration (SOT increased last poll) OR
# best_sot >= 2, the window extends:
#   - BEFORE 21': signals fire from minute 9+ (early pressure, v10.19: GPS≥75 required)
#   - AFTER 61': signals fire until minute 85 (late pressure buildup, v10.19: raised from 75)
# Note: polling starts from 1' (PRE_WINDOW_MINUTES=20), but signals
# before 9' are always blocked — 9' is the earliest possible signal minute.
EXTENDED_MIN = 9    # minimum minute for early-pressure signals
EXTENDED_MAX = 85   # maximum minute for late-pressure signals

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
# Key: (fixture_id, team_id) -> {"count": N, "goals_at_last_signal": G, "sot_at_last_signal": S, "last_signal_time": T}
# 1st signal: always sent (SOT >= 3 increased -> CRITICAL, or GPS-based EARLY WARNING)
# 2nd signal: no explicit SOT-jump gate; classify_signal dedup handles it (+1 SOT suffices)
# 3rd+ signal: SOT-jump gate from LAST SIGNAL (not last poll):
#   - +1 SOT in <3 min with GPS >= 60 (fast-jump discount: floor = 75-15)
#   - +2 SOT with any GPS (passes jump requirement regardless)
#   - +1 SOT slower than 3 min requires GPS >= 75
#   - blocked if team scored since last signal
# v9.7: sot_at_last_signal fixes the bug where jump was measured from last poll's SOT
# v10.19.3: fast-jump GPS discount for rapid +1 SOT increments
signaled_teams: dict[tuple[int, int], dict] = {}
# Keep fixture-level set for backward compat in logs/cleanup
signaled_fixtures: set[int] = set()

# --- v10.28: Signal outcome tracking (backtesting) ---
# Records every signal sent with enriched recency data for empirical analysis.
#   PRIMARY KPI:   outcome_full (did they score at any point after signal before FT?)
#   SECONDARY KPI: outcome_5min/10min/15min (how quickly did they score?)
# v10.28: Added recency fields: SOT/shots/xG/IB/corners at previous poll,
#   5min ago, 10min ago, deltas, GPS change, seconds since last signal, recency_ratio.
# Persisted to JSONL file on persistent volume so data survives restarts.
signal_outcomes: list[dict] = []
OUTCOME_WINDOW_MINUTES = 15  # game minutes for "imminent" window
# Railway volume mount point — survives redeployments
_VOLUME_DIR = os.environ.get("VOLUME_DIR", "/data")
os.makedirs(_VOLUME_DIR, exist_ok=True)
OUTCOMES_FILE = os.path.join(_VOLUME_DIR, "signal_outcomes.jsonl")
EOD_SENT_FILE = os.path.join(_VOLUME_DIR, "eod_sent.txt")

# --- v10: Goal Pressure Score (GPS) ---
# Composite 0-100 score calculated on EVERY stats poll.
# Uses all available API stats (zero extra cost — data already in response).
# v10.26 Components (max ~111, capped at 100):
#   SOT: 0-28 | Inside Box: 0-22 | Shot Vol: 0-14 | xG: 0-18 | Corners: 0-4 | Accel: 0-22
#   REMOVED: possession (always 0 in API), dangerous_attacks (not in API-Football v3)
# The acceleration component is the key differentiator from v9.9.
GPS_EARLY_WARNING = 55   # 55-74: EARLY WARNING signal
GPS_CRITICAL = 75         # 75+: CRITICAL signal
GPS_BUILDING = 40        # 40-54: logged as BUILDING (no Telegram)

# --- v10: Pressure acceleration tracking ---
# Fixtures where GPS is rising AND multiple stat deltas are positive.
# These get 60s polling to catch the SOT transition in real time,
# even BEFORE SOT reaches 2. This is the key v10 improvement.
pressure_accelerating: set[int] = set()  # fixture IDs

# --- v10.17: SOT burst tracking ---
# Fixtures where best SOT jumped by 2+ in a single poll interval.
# Gets highest priority (97) polling and GPS bonus.
sot_burst_fixtures: set[int] = set()

# --- v10: Poll-level data collection for backtesting ---
# Records ALL indicators from every stats poll (not just signals).
# This builds the dataset to empirically validate which combinations
# predict goals within 5/10/15 minutes.
# v10.19.3: Moved to /data volume so poll data survives redeployments
POLL_DATA_FILE = os.path.join(_VOLUME_DIR, "pressure_polls.jsonl")

# --- v10.11: Daily summary ---
daily_summary_date: str = ""
todays_tracked_fixtures: list[dict] = []  # populated during schedule fetch

# --- v10.12: End-of-day Telegram summary ---
eod_summary_sent_date: str = ""  # track if today's EOD summary was sent via Telegram


def _load_eod_sent_date() -> str:
    """v10.16: Persist EOD sent date to disk so restarts don't re-send."""
    try:
        if os.path.exists(EOD_SENT_FILE):
            with open(EOD_SENT_FILE, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


def _save_eod_sent_date(date_str: str) -> None:
    """v10.16: Persist EOD sent date to disk."""
    try:
        with open(EOD_SENT_FILE, "w") as f:
            f.write(date_str)
    except Exception:
        pass


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
# v10.21: Track SOT level at last fast-window activation.
# Prevents infinite re-activation when SOT is stuck (e.g. SOT=1 for 10+ min).
fast_sot_activated_at_sot: dict[int, int] = {}

# --- v9.7: SOT acceleration tracking ---
# Fixtures where the best SOT increased in the last poll.
# These get priority polling (60-90s) to catch the next SOT increase.
accelerating_fixtures: set[int] = set()

# --- v10.1: Per-team GPS history (last 5 polls for 5-min windowed rates) ---
# Key: (fixture_id, team_id) -> list of dicts with GPS + raw stats + timestamp per poll
# v10.1: Expanded from 3 to 5 to enable 5-game-minute window calculations.
team_gps_history: dict[tuple[int, int], list[dict]] = {}
GPS_HISTORY_MAX = 5  # v10.1: keep last 5 polls (covers ~5 min at 60s polling)
GPS_WINDOW_MINUTES = 5  # v10.1: window for rate-of-change calculations

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
SCHEDULE_RECHECK_INTERVAL = 8 * 3600  # v9.7: 6h→8h (saves ~1 credit/day, late additions rare after 8h)

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
    matches enters the pre-window, and wake up just in time.

    v10.15: buffer=-60 means we wake 60s AFTER the pre-window mark,
    accounting for API-Football's ~60s minute reporting delay. Entries
    are NOT consumed on gap<=0 — instead they age out after 180s.
    This retries every 30s if the API hasn't caught up yet.

    Returns None if no future window entries exist (all waves processed).
    """
    global scheduled_window_entries
    now = time.time()
    buffer = -60  # v10.15: wake 60s AFTER the pre-window mark (accounts for API minute lag)

    # v10.15: Prune entries >3 min old (gives API time to catch up)
    scheduled_window_entries = [
        t for t in scheduled_window_entries if t > now - 180
    ]

    if not scheduled_window_entries:
        return None

    # Find the next future entry (within 3 min window)
    future_entries = [t for t in scheduled_window_entries if t > now - buffer]
    if not future_entries:
        # v10.19: All entries are within the buffer window or past it.
        # Return short interval — either a wave is imminent or entries are stale.
        return 30

    next_entry = min(future_entries)
    gap = next_entry - now - buffer

    if gap <= 0:
        # v10.15: DON'T consume — the API minute might not have caught up yet.
        # Retry in 30s. Entry ages out naturally after 180s (3 retries max).
        # This replaces the old v9.6.2 consume-on-sight behavior which could
        # miss matches when API minute lag exceeded the 60s buffer.
        return 30

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

    v10.13: Reduced from 240→90s NORMAL, 300→150s CAREFUL, etc.
    With batched API calls (1 credit for up to 20 fixtures), the cost
    of polling 3x more often is minimal (~+2 credits per cycle with
    4 fixtures) but coverage improves dramatically — fixtures now
    get 3-4 polls in the 40-min window instead of 1.
    """
    if budget_mode == "NORMAL":
        return 90
    if budget_mode == "CAREFUL":
        return 150
    if budget_mode == "STRICT":
        return 210
    if budget_mode == "EMERGENCY":
        return 300
    return 300


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


def activate_fast_sot(fid: int, current_best_sot: int = 0):
    fast_sot_until[fid] = time.time() + FAST_SOT_WINDOW
    fast_sot_activated_at_sot[fid] = current_best_sot
    log.info(
        f"  Fast SOT window ACTIVATED for fixture {fid} "
        f"(expires in {FAST_SOT_WINDOW}s, SOT={current_best_sot})"
    )


def get_fixture_max_gps(fid: int) -> float:
    """v10.22: Get the highest GPS of any team in this fixture."""
    best = 0.0
    for (f, _), history in team_gps_history.items():
        if f == fid and history:
            best = max(best, history[-1].get("gps", 0))
    return best


def is_in_edge_zone(fid: int) -> bool:
    """v10.22: Edge zone = before 21' or after 60'.

    In these zones, signals require GPS>=75 (EARLY/LATE GATE),
    so fast polling is only worthwhile for genuinely high pressure.
    Low-pressure games here burn credits with near-zero signal chance.
    """
    for (f, _), state in team_state.items():
        if f == fid:
            minute = state.get("last_minute", 0)
            if minute > 0 and (minute < MINUTE_MIN or minute > MINUTE_MAX):
                return True
    return False


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
    fast_sot_activated_at_sot.pop(fid, None)


# ============================================================
# SIGNAL CLASSIFICATION — v10: GPS-based 3-stage system
# ============================================================
# v10 stages:
#   GPS < 40:            no signal
#   GPS 40-54:           BUILDING (log only, no Telegram)
#   GPS 55-74 + SOT>=1: EARLY WARNING (Telegram)
#   GPS 75+ OR SOT>=3:   CRITICAL (Telegram)
#
# SAFETY: SOT>=3 ALWAYS triggers CRITICAL regardless of GPS.
# This ensures the proven v9.x trigger still works while we
# collect data to validate the GPS thresholds.

TIER_ORDER = ["EARLY WARNING", "CRITICAL"]


def classify_signal(
    sot: int, state: dict | None, current_minute: int,
    gps: float = 0.0, accel_count: int = 0,
    inside_box_ratio: float = 0.0, sustained_count: int = 0,
) -> tuple[str | None, str, float]:
    """v10.1: Classify signal based on GPS + SOT hybrid system.

    The GPS allows earlier detection (SOT=1 with high acceleration),
    while SOT>=3 remains a guaranteed CRITICAL trigger (safety net).

    v10.1 quality gates:
    - EARLY WARNING requires inside_box_ratio >= 30% (shot quality)
    - EARLY WARNING requires sustained_count >= 1 OR gps >= CRITICAL
      (single-poll spikes don't trigger — need sustained acceleration)

    v10.24 quality gates:
    - IB ratio < 50% blocks ALL signals (shots not dangerous)
    - GPS < 60 blocks EARLY WARNING (SOT>=3 CRITICAL exempt)
    - LOSING teams: EW blocked, CRITICAL tagged with warning
    """
    trend = ""
    sot_rate = 0.0
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    # v10.29: SOT≥3 SAFETY NET — must fire BEFORE IB gate.
    # Bug: v10.24 placed IB<50% before this check, so a team with 8 SOT
    # but IB=42.9% (Nordsjaelland, 77') was blocked. SOT≥3 is the
    # gold standard — if 3+ shots are on target, pressure is real
    # regardless of where other (off-target) shots came from.
    if sot >= 3:
        last_sot = state["last_sot"] if state else 0
        if sot > last_sot:  # dedup: only on SOT increase
            return "CRITICAL", trend, sot_rate
        return None, "", 0.0

    # v10.24: IB RATIO HARD FLOOR — <50% means most shots are outside the box.
    # Data: 0% short-term HIT (0/5), only 1/5 full HIT was 92' consolation while losing 1-3.
    # Teams chipping long-range shots generate high shot volume but no danger.
    # Applies ONLY to GPS-based signals (SOT < 3). SOT≥3 bypasses above.
    if inside_box_ratio < 0.50:
        return None, "", 0.0

    # v10.24: GPS FLOOR for EARLY WARNING — GPS<60 means composite pressure
    # is not meaningfully elevated. Data: 0/3 HIT across all windows.
    # SOT>=3 CRITICAL signals are exempt (SOT itself is strong evidence).
    if gps < 60:
        return None, "", 0.0

    # v10.1 GPS-based triggers for SOT < 3:
    # QUALITY GATE: shot quality matters.
    # 3 harmless long-range shots should NOT equal 3 dangerous inside-box shots.
    # Require at least 30% of shots inside the box for EARLY WARNING.
    if sot >= 1 and gps >= GPS_EARLY_WARNING:
        if inside_box_ratio < 0.30 and gps < GPS_CRITICAL:
            # Low shot quality + not at critical GPS — skip
            return None, "", 0.0

        # SUSTAINED GATE: single-poll acceleration spikes need confirmation.
        # Either: sustained over 2+ polls, OR GPS already at CRITICAL level.
        if sustained_count < 1 and gps < GPS_CRITICAL:
            return None, "", 0.0

        last_sot = state["last_sot"] if state else 0
        # For SOT=1: allow if GPS is high (acceleration-driven detection)
        # For SOT=2: allow if GPS is high (replaces v9.9 pressure_building gate)
        if sot > last_sot or gps >= GPS_CRITICAL:
            return "EARLY WARNING", trend, sot_rate

    return None, "", 0.0


def detect_pressure_building(tstats: dict, state: dict | None, minute: int) -> tuple[bool, str]:
    """v10.10: Detect if a team is building dangerous pressure despite low SOT.

    v10.10: Replaced DA with shots_off_target (API-Football v3 has no DA).
    Checks: shots off target rate, total shot volume.
    Returns (is_building, description_string).
    """
    if not state or not state.get("last_minute", 0) > 0:
        return False, ""
    sotr = safe_int(get_stat(tstats, "shots_off_target"))
    ts = safe_int(get_stat(tstats, "total_shots"))

    prev_sotr = state.get("last_shots_off_target", 0)
    prev_min = state["last_minute"]
    mins_passed = max(minute - prev_min, 1)
    sotr_delta = sotr - prev_sotr
    sotr_rate = sotr_delta / mins_passed

    # Pressure building conditions:
    #   1. Off-target shots increasing (at least 1 every 2 minutes)
    #   2. Lots of total shots
    #   3. Minimum absolute off-target to filter early-game noise
    is_building = (
        sotr_rate >= 0.2
        and ts >= 5
        and sotr >= 3
    )

    desc = (f"SOffT:{sotr}(+{sotr_delta}), TS:{ts}, SOffT-rate:{sotr_rate:.1f}/min")
    return is_building, desc


def tier_emoji(tier: str) -> str:
    if tier == "CRITICAL": return "\U0001f534"
    if tier == "EARLY WARNING": return "\U0001f525"
    if tier == "STRONG": return "\U0001f7e0"
    if tier == "PRESSURE": return "\U0001f7e1"
    return "\u26aa"  # BUILDING (white circle)


# ============================================================
# v10.1: FIELD NAME NORMALIZATION
# ============================================================
# API-Football field names can vary by API plan or version.
# A silent 0 from a wrong field name would systematically understate GPS.
# Instead of assuming one exact spelling, try multiple candidates.

STAT_ALIASES: dict[str, list[str]] = {
    "sot":                ["Shots on Goal", "Shots on Target", "Shot on Goal", "Shots on goal"],
    "total_shots":        ["Total Shots", "Total shots"],
    "shots_inside_box":   ["Shots insidebox", "Shots Inside Box", "Shots inside Box"],
    "shots_off_target":   ["Shots off Goal", "Shots off Target"],
    # v10.10: REMOVED dangerous_attacks — API-Football v3 does NOT provide this stat.
    # It was silently returning 0, costing us 8 GPS points + breaking acceleration.
    # "dangerous_attacks":  ["Dangerous Attacks", "Dangerous attacks"],
    "corner_kicks":       ["Corner Kicks", "Corners", "corner kicks"],
    "red_cards":          ["Red Cards", "Red cards"],
    "possession":         ["Ball Possession", "Ball possession", "Possession %", "Possession"],
    "fouls":              ["Fouls", "fouls"],
    "total_passes":       ["Total passes", "Total Passes"],
    "passes_accurate":    ["Passes accurate", "Passes Accurate"],
    # v10.10: xG — API-Football v3 returns stat type "expected_goals" (underscore format)
    # Previously tried "Expected Goals" which never matched. Now via STAT_ALIASES.
    "expected_goals":     ["expected_goals", "Expected Goals", "expectedGoals", "Expected goals"],
}


def get_stat(tstats: dict, stat_key: str, default: str = "0") -> str:
    """Look up a stat using normalized field name aliases.

    Tries each candidate in order, returns first non-None value.
    Falls back to default if all candidates miss or value is None.
    """
    candidates = STAT_ALIASES.get(stat_key, [stat_key])
    for candidate in candidates:
        val = tstats.get(candidate)
        if val is not None:
            return str(val).strip()
    return default


def get_top_sot_player(client: httpx.Client, fixture_id: int, team_name: str) -> str:
    """v10.18: Fetch top SOT player for a team in a fixture.

    Calls /fixtures/statistics?fixture=X (1 credit) to get player-level data.
    Returns string like 'PlayerName (SOT: 3)' or empty string on failure.
    Only called when a signal fires (rare), so credit cost is negligible.
    """
    try:
        data = api_get(client, "/fixtures/statistics", {"fixture": fixture_id})
        response = data.get("response", [])
        if not response:
            return ""

        for team_entry in response:
            entry_team_name = team_entry.get("team", {}).get("name", "")
            if entry_team_name != team_name:
                continue

            players = team_entry.get("players", [])
            if not players:
                return ""

            best_player = ""
            best_sot = 0
            for p in players:
                pname = p.get("player", {}).get("name", "")
                if not pname:
                    continue
                pstats = p.get("statistics", [])
                if not pstats:
                    continue
                pstat0 = pstats[0] if isinstance(pstats, list) else pstats
                shots = pstat0.get("shots", {})
                if isinstance(shots, dict):
                    p_sot = shots.get("on", 0) or 0
                else:
                    p_sot = 0
                if p_sot > best_sot:
                    best_sot = p_sot
                    best_player = pname

            if best_player and best_sot >= 1:
                return f"{best_player} (SOT: {best_sot})"
            return ""
    except Exception:
        return ""


# ============================================================
# v10: GOAL PRESSURE SCORE (composite, all available API stats)
# ============================================================

def safe_int(val, default=0) -> int:
    """Safely parse a stat value to int."""
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def safe_float(val, default=None) -> float | None:
    """Safely parse a stat value to float. Returns None if unavailable."""
    if val is None or str(val).strip() in ("", "N/A"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def calculate_goal_pressure_score(
    sot: int,
    total_shots: int,
    shots_inside_box: int,
    shots_off_target: int,  # v10.10: replaces dangerous_attacks (API-Football v3 has no DA)
    xg_value: float | None,
    corners: int,
    minute: int,
    prev_state: dict | None,
    gps_history: list[dict],
    possession: int = 0,  # v10.26: kept for API compat, NOT used in score (always 0 in API)
) -> tuple[float, str, dict]:
    """v10.26: Calculate 0-100 Goal Pressure Score.

    v10.26 changes from v10.10:
      - REMOVED possession component (5 pts) — API-Football v3 NEVER returns possession
        data across any of the 12 tracked leagues (confirmed across 2,168 polls).
      - REDISTRIBUTED 5 possession points: +3 to xG (max 15→18), +2 to acceleration cap (max 20→22)
      - xG is the strongest single-shot quality predictor; acceleration catches momentum.

    v10.10 changes from v10.1:
      - REMOVED dangerous_attacks (not in API-Football v3 — was always 0)
      - REDISTRIBUTED 8 DA points: +4 to inside-box ratio (max 22), +2 to shot volume (max 14),
        +2 to possession (max 5, now also removed)
      - Added shots_off_target as volume indicator for acceleration
      - xG now actually works (field name fix in parse_xg)

    Returns: (score_0_to_100, description, component_breakdown)
    """
    components = {}
    score = 0.0

    # === 1. SOT component (0-28 points) ===
    # Absolute SOT matters, but with diminishing returns.
    # SOT=0: 0, SOT=1: 4, SOT=2: 10, SOT=3: 20, SOT=4: 25, SOT=5+: 28
    sot_table = {0: 0, 1: 4, 2: 10, 3: 20, 4: 25}
    sot_pts = sot_table.get(min(sot, 5), 28)
    score += sot_pts
    components["sot"] = sot_pts

    # === 2. Shots inside box ratio (0-22 points) ===  # v10.10: boosted from 18 → 22 (absorbed 4 DA points)
    # High inside-box ratio = quality shot selection = more dangerous.
    ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
    ib_pts = min(ib_ratio * 30, 22)  # 73%+ = full 22 pts
    score += ib_pts
    components["inside_box"] = round(ib_pts, 1)

    # === 3. Shot volume relative to minute (0-14 points) ===
    # v10.10: boosted from 12 → 14 (absorbed 2 DA points)
    # More shots per minute = sustained attacking intent.
    if minute > 0:
        shots_per_min = total_shots / minute
        sv_pts = min(shots_per_min * 140, 14)  # ~0.1/min = 14 pts
    else:
        sv_pts = 0.0
        shots_per_min = 0.0
    score += sv_pts
    components["shot_vol"] = round(sv_pts, 1)

    # === 4. xG (0-18 points) ===
    # v10.26: boosted from 0-15 → 0-18 (absorbed 3 possession points)
    # Quality of chances created. 1.0 xG = 18 pts (was 15).
    if xg_value is not None and xg_value >= 0:
        xg_pts = min(xg_value * 18, 18)
    else:
        xg_pts = 0.0
    score += xg_pts
    components["xg"] = round(xg_pts, 1)

    # === 5. Possession — REMOVED in v10.26 ===
    # API-Football v3 NEVER returns possession data (0 in all 2,168 polls across
    # 12 leagues). The 5 points were redistributed: +3 xG, +2 acceleration cap.
    # Kept as 0 for backward compat in components dict and JSONL records.
    components["possession"] = 0.0

    # v10.10: shots_off_target stored for acceleration calc
    sot_rate_delta = 0.0

    # === 6. Corner kicks (0-4 points) ===
    # Set pieces = additional scoring opportunities.
    corner_pts = min(corners * 1.0, 4)
    score += corner_pts
    components["corners"] = corner_pts

    # === 7. ACCELERATION component (0-15 points) — THE KEY DIFFERENTIATOR ===
    # v10.10: Uses WINDOWED per-minute rates over GPS_WINDOW_MINUTES game minutes.
    # v10.10: Replaced DA delta with shots_off_target delta (DA doesn't exist in API).
    #    Shots off target rising = team is getting shots away even if not on target.
    accel_pts = 0.0
    accel_details = ""
    sustained_count = 0
    sot_burst_pts = 0  # v10.17
    goalless_pts = 0   # v10.17

    if prev_state and prev_state.get("last_minute", 0) > 0:
        mins_passed = max(minute - prev_state["last_minute"], 1)
        sot_delta = sot - prev_state.get("last_sot", 0)
        ts_delta = total_shots - prev_state.get("last_total_shots", 0)
        sot_delta_var = shots_off_target - prev_state.get("last_shots_off_target", 0)
        prev_xg = prev_state.get("last_xg")
        xg_delta = 0.0
        if xg_value is not None and prev_xg is not None:
            try:
                xg_delta = xg_value - float(prev_xg)
            except (ValueError, TypeError):
                pass

        # v10.10: Per-minute rates (interval-independent thresholds)
        sot_rate = sot_delta / mins_passed
        ts_rate = ts_delta / mins_passed
        sotr_rate = sot_delta_var / mins_passed  # shots off target rate

        # v10.10: Count accelerating indicators using RATE thresholds.
        accel_count = 0
        if sot_rate >= 0.15:       # 1 SOT per ~7 min or faster
            accel_count += 1
        if ts_rate >= 0.3:        # 1 shot per ~3 min or faster
            accel_count += 1
        if sotr_rate >= 0.2:      # 1 off-target shot per 5 min (replaces DA)
            accel_count += 1
        if xg_delta >= 0.1 and (xg_delta / mins_passed) >= 0.02:
            accel_count += 1

        # Windowed rates over last GPS_WINDOW_MINUTES game minutes
        window_sot_rate = sot_rate
        window_ts_rate = ts_rate
        window_sotr_rate = sotr_rate
        if len(gps_history) >= 2:
            window_start_min = minute - GPS_WINDOW_MINUTES
            window_entry = None
            for entry in gps_history:
                if entry.get("minute", 0) >= window_start_min:
                    window_entry = entry
                    break
            if window_entry:
                window_mins = max(minute - window_entry["minute"], 1)
                window_sot_rate = (sot - window_entry.get("sot", 0)) / window_mins
                window_ts_rate = (total_shots - window_entry.get("total_shots", 0)) / window_mins
                window_sotr_rate = (shots_off_target - window_entry.get("shots_off_target", 0)) / window_mins

                # Use windowed rates if they show MORE acceleration
                window_accel = 0
                if window_sot_rate >= 0.15:
                    window_accel += 1
                if window_ts_rate >= 0.3:
                    window_accel += 1
                if window_sotr_rate >= 0.2:
                    window_accel += 1
                if xg_value is not None and window_entry.get("xg") is not None:
                    try:
                        wxg_delta = xg_value - float(window_entry["xg"])
                        if wxg_delta >= 0.1 and wxg_delta / window_mins >= 0.02:
                            window_accel += 1
                    except (ValueError, TypeError):
                        pass
                if window_accel > accel_count:
                    accel_count = window_accel

        if accel_count >= 3:
            accel_pts = 15
        elif accel_count >= 2:
            accel_pts = 10
        elif accel_count == 1:
            accel_pts = 4

        # SUSTAINED pressure bonus
        for h in gps_history:
            if h.get("accel_count", 0) >= 2:
                sustained_count += 1
        if accel_count >= 2 and sustained_count >= 1:
            accel_pts = min(accel_pts + 4, 15)

        # v10.17: SOT BURST bonus — SOT jumped 2+ in one poll interval
        # A team going from 0→2 or 1→3 SOT in 90-150s is an extreme
        # pressure spike. Much stronger than gradual SOT=1 acceleration.
        sot_burst_pts = 0
        sot_burst_detail = ""
        if sot_delta >= 2:
            burst_base = 6 if sot_delta >= 3 else 3
            # Combo: if GPS is also rising vs last poll, extra bonus
            gps_rising = len(gps_history) >= 1 and score > gps_history[-1].get("gps", 0)
            combo_extra = 3 if gps_rising else 0
            sot_burst_pts = min(burst_base + combo_extra, 8)
            sot_burst_detail = (
                f" BURST+{sot_delta}SOT({burst_base}"
                f"{'+3combo' if combo_extra else ''})"
            )
        accel_pts = min(accel_pts + sot_burst_pts, 22)  # v10.17: cap 15→20, v10.26: 20→22

        # v10.17: GOALLESS PRESSURE bonus — sustained high GPS, no conversion
        # If a team has GPS >= 50 for 3+ recent polls, they're dominating
        # but not scoring — odds of a breakthrough goal keep rising.
        # If a goal WAS scored, pressure typically drops, resetting the count.
        goalless_count = sum(1 for h in gps_history if h.get("gps", 0) >= 50)
        goalless_pts = 0
        if goalless_count >= 4:
            goalless_pts = 5
        elif goalless_count >= 3:
            goalless_pts = 3
        elif goalless_count >= 2:
            goalless_pts = 1
        accel_pts = min(accel_pts + goalless_pts, 22)  # v10.26: cap raised 20→22 (absorbed 2 poss pts)

        accel_details = (
            f"SOT {sot_rate:.2f}/m(w:{window_sot_rate:.2f}), "
            f"TS {ts_rate:.2f}/m(w:{window_ts_rate:.2f}), "
            f"SOffT {sotr_rate:.1f}/m(w:{window_sotr_rate:.1f})"
            + (f", xG+{xg_delta:.2f}" if xg_delta > 0 else "")
            + f" [{accel_count} accel{' sustained' if sustained_count > 0 and accel_count >= 2 else ''}"
            + (f"{sot_burst_detail}" if sot_burst_detail else "")
            + (f" goalless:{goalless_count}" if goalless_count >= 2 else "")
            + "]"
        )

    score += accel_pts
    components["acceleration"] = round(accel_pts, 1)
    components["sustained"] = sustained_count  # v10.1: for backtesting
    components["sot_burst"] = sot_burst_pts
    components["goalless_pressure"] = goalless_pts

    score = min(score, 100.0)

    desc = (
        f"GPS:{score:.0f} SOT:{sot}({components['sot']}pts) "
        f"IB:{ib_ratio:.0%}({components['inside_box']}pts) "
        f"TS/min:{shots_per_min:.2f}({components['shot_vol']}pts) "
        f"xG:{xg_value or 0:.2f}({components['xg']}pts) "
        f"Accel:{accel_pts:.0f}pts"
    )
    if accel_details:
        desc += f" | {accel_details}"

    return score, desc, components


def _build_recency_fields(
    fid: int, tid: int, minute: int,
    sot: int, total_shots: int, shots_inside_box: int,
    shots_off_target: int, xg_value: float | None, corners: int,
    gps: float, accel_count: int,
) -> dict:
    """v10.28: Compute enriched recency fields from GPS history.

    For every signal (and poll), compute historical snapshots at
    previous poll, ~5 game minutes ago, and ~10 game minutes ago.
    Also compute deltas, GPS change, seconds since last signal, and
    a recency_ratio that measures how "fresh" the pressure is.

    recency_ratio = (SOT_last_10m_delta) / max(SOT_total, 1)
    High ratio = most SOT arrived recently (fresh pressure).
    Low ratio = SOT accumulated over a long period (stale pressure).
    """
    history = team_gps_history.get((fid, tid), [])
    fields = {
        # Previous poll values
        "prev_poll_sot": None, "prev_poll_shots": None,
        "prev_poll_ib": None, "prev_poll_xg": None,
        "prev_poll_corners": None, "prev_poll_gps": None,
        # 5 game minutes ago
        "sot_5m_ago": None, "shots_5m_ago": None,
        "ib_5m_ago": None, "xg_5m_ago": None,
        "corners_5m_ago": None, "gps_5m_ago": None,
        # 10 game minutes ago
        "sot_10m_ago": None, "shots_10m_ago": None,
        "ib_10m_ago": None, "xg_10m_ago": None,
        "corners_10m_ago": None, "gps_10m_ago": None,
        # Deltas from previous poll
        "delta_sot": None, "delta_shots": None,
        "delta_ib": None, "delta_xg": None,
        "delta_corners": None, "delta_gps": None,
        # 5min and 10min window deltas
        "sot_delta_5m": None, "shots_delta_5m": None,
        "xg_delta_5m": None, "ib_delta_5m": None,
        "corners_delta_5m": None,
        "sot_delta_10m": None, "shots_delta_10m": None,
        "xg_delta_10m": None, "ib_delta_10m": None,
        "corners_delta_10m": None,
        # Derived
        "seconds_since_prev_signal": None,
        "gps_change": None,
        "recency_ratio": None,
    }

    if not history:
        return fields

    # Previous poll (most recent entry before this one)
    prev = history[-1] if history else None
    if prev:
        fields["prev_poll_sot"] = prev.get("sot")
        fields["prev_poll_shots"] = prev.get("total_shots")
        fields["prev_poll_ib"] = prev.get("shots_inside_box")
        fields["prev_poll_xg"] = prev.get("xg")
        fields["prev_poll_corners"] = prev.get("corners")
        fields["prev_poll_gps"] = round(prev.get("gps", 0), 1)

        # Deltas from previous poll
        fields["delta_sot"] = sot - (prev.get("sot") or 0)
        fields["delta_shots"] = total_shots - (prev.get("total_shots") or 0)
        fields["delta_ib"] = shots_inside_box - (prev.get("shots_inside_box") or 0)
        prev_xg = prev.get("xg")
        if xg_value is not None and prev_xg is not None:
            try:
                fields["delta_xg"] = round(xg_value - float(prev_xg), 3)
            except (ValueError, TypeError):
                pass
        fields["delta_corners"] = corners - (prev.get("corners") or 0)
        fields["delta_gps"] = round(gps - prev.get("gps", 0), 1)

    # Find entries at ~5 and ~10 game minutes ago
    entry_5m = None
    entry_10m = None
    for entry in reversed(history):
        e_min = entry.get("minute", 0)
        if entry_5m is None and e_min <= minute - 5:
            entry_5m = entry
        if entry_10m is None and e_min <= minute - 10:
            entry_10m = entry
        if entry_10m is not None:
            break

    if entry_5m:
        fields["sot_5m_ago"] = entry_5m.get("sot")
        fields["shots_5m_ago"] = entry_5m.get("total_shots")
        fields["ib_5m_ago"] = entry_5m.get("shots_inside_box")
        fields["xg_5m_ago"] = entry_5m.get("xg")
        fields["corners_5m_ago"] = entry_5m.get("corners")
        fields["gps_5m_ago"] = round(entry_5m.get("gps", 0), 1)

        fields["sot_delta_5m"] = sot - (entry_5m.get("sot") or 0)
        fields["shots_delta_5m"] = total_shots - (entry_5m.get("total_shots") or 0)
        e5m_xg = entry_5m.get("xg")
        if xg_value is not None and e5m_xg is not None:
            try:
                fields["xg_delta_5m"] = round(xg_value - float(e5m_xg), 3)
            except (ValueError, TypeError):
                pass
        fields["ib_delta_5m"] = shots_inside_box - (entry_5m.get("shots_inside_box") or 0)
        fields["corners_delta_5m"] = corners - (entry_5m.get("corners") or 0)

    if entry_10m:
        fields["sot_10m_ago"] = entry_10m.get("sot")
        fields["shots_10m_ago"] = entry_10m.get("total_shots")
        fields["ib_10m_ago"] = entry_10m.get("shots_inside_box")
        fields["xg_10m_ago"] = entry_10m.get("xg")
        fields["corners_10m_ago"] = entry_10m.get("corners")
        fields["gps_10m_ago"] = round(entry_10m.get("gps", 0), 1)

        fields["sot_delta_10m"] = sot - (entry_10m.get("sot") or 0)
        fields["shots_delta_10m"] = total_shots - (entry_10m.get("total_shots") or 0)
        e10m_xg = entry_10m.get("xg")
        if xg_value is not None and e10m_xg is not None:
            try:
                fields["xg_delta_10m"] = round(xg_value - float(e10m_xg), 3)
            except (ValueError, TypeError):
                pass
        fields["ib_delta_10m"] = shots_inside_box - (entry_10m.get("shots_inside_box") or 0)
        fields["corners_delta_10m"] = corners - (entry_10m.get("corners") or 0)

    # Seconds since previous signal for this team
    team_sig = signaled_teams.get((fid, tid))
    if team_sig and team_sig.get("last_signal_time", 0) > 0:
        fields["seconds_since_prev_signal"] = round(time.time() - team_sig["last_signal_time"])

    # GPS change (current vs earliest history entry)
    if len(history) >= 1:
        fields["gps_change"] = round(gps - history[0].get("gps", 0), 1)

    # Recency ratio: what fraction of SOT arrived in the last 10 game minutes?
    # High = fresh pressure, Low = accumulated pressure
    sot_10m_ago = fields.get("sot_10m_ago")
    if sot_10m_ago is not None and sot > 0:
        fields["recency_ratio"] = round(
            (sot - sot_10m_ago) / sot, 3
        )
    elif entry_5m is not None and sot > 0:
        # Fallback: use 5m window if no 10m data
        sot_5m_ago = fields.get("sot_5m_ago")
        if sot_5m_ago is not None:
            fields["recency_ratio"] = round(
                (sot - sot_5m_ago) / sot, 3
            )

    return fields


def record_pressure_poll(
    fid: int, tid: int, tname: str, league: str,
    minute: int, sot: int, total_shots: int,
    shots_inside_box: int, shots_off_target: int,  # v10.10: replaces dangerous_attacks
    xg_value: float | None, corners: int,
    gps: float, components: dict,
    accel_count: int, is_home: bool,
    score_home: int, score_away: int,
    possession: int = 0,
    stats_sot_raw: int = -1,  # v10.31: raw statistics SOT (before events supplement)
    events_sot: int = 0,      # v10.31: event-based SOT (from /fixtures/events)
) -> None:
    """v10.1: Write poll-level data to JSONL for backtesting.

    This records EVERY stats poll (not just signals), building the dataset
    to answer: "which combination of stats predicts a goal within 5/10/15 min?"

    v10.26: Added real data_quality field (fraction of non-null/non-zero fields)
    replacing the previous hardcoded 0.5 that never varied.

    v10.28: Added recency fields (prev poll, 5m/10m ago, deltas, recency_ratio).
    """
    try:
        # v10.26: Compute real data quality score (0.0-1.0)
        # Fraction of key fields that are populated (non-null, non-zero where applicable)
        key_fields = {
            "sot": sot > 0,
            "total_shots": total_shots > 0,
            "shots_inside_box": shots_inside_box > 0,
            "shots_off_target": shots_off_target > 0,
            "xg": xg_value is not None and xg_value >= 0,
            "corners": corners >= 0,  # always present but check anyway
            "possession": possession > 0,  # always False (dead field)
        }
        data_quality = sum(1 for v in key_fields.values() if v) / len(key_fields)

        # v10.28: Compute recency fields from GPS history
        recency = _build_recency_fields(
            fid, tid, minute, sot, total_shots,
            shots_inside_box, shots_off_target, xg_value, corners,
            gps, accel_count,
        )

        entry = {
            "ts": time.time(),
            "fixture_id": fid, "team_id": tid, "team_name": tname,
            "league": league, "minute": minute, "is_home": is_home,
            "sot": sot, "total_shots": total_shots,
            "shots_inside_box": shots_inside_box,
            "shots_off_target": shots_off_target,  # v10.10
            "xg": round(xg_value, 3) if xg_value is not None else None,
            "corners": corners,
            "possession": possession,  # v10.26: kept for compat, always 0
            "data_quality": round(data_quality, 2),  # v10.26: REAL quality score
            "gps": round(gps, 1),
            "gps_sot": components.get("sot", 0),
            "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0),
            "gps_xg": components.get("xg", 0),
            "gps_poss": components.get("possession", 0),  # v10.26: always 0 (deprecated)
            "gps_accel": components.get("acceleration", 0),
            "gps_sot_burst": components.get("sot_burst", 0),  # v10.17
            "gps_goalless": components.get("goalless_pressure", 0),  # v10.17
            "sustained": components.get("sustained", 0),
            "accel_count": accel_count,
            "score_home": score_home, "score_away": score_away,
            "stats_sot_raw": stats_sot_raw,  # v10.31: audit trail
            "events_sot": events_sot,        # v10.31: audit trail
        }
        # v10.28: Merge recency fields into poll entry
        entry.update(recency)
        with open(POLL_DATA_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.warning(f"  Failed to save poll data: {e}")


def get_accel_count_from_state(
    sot: int, total_shots: int, shots_off_target: int,  # v10.10: replaces dangerous_attacks
    xg_value: float | None, prev_state: dict | None, minute: int,
) -> int:
    """v10.10: Count accelerating indicators using RATE-based thresholds.

    v10.10: DA replaced with shots_off_target (API-Football v3 has no DA).
    Must match the rate thresholds in calculate_goal_pressure_score.
    Used by pressure-accelerating fixture detection for 60s polling.
    Returns count of accelerating indicators (0-4).
    """
    if not prev_state or prev_state.get("last_minute", 0) <= 0:
        return 0

    mins_passed = max(minute - prev_state.get("last_minute", 0), 1)
    accel_count = 0
    sot_delta = sot - prev_state.get("last_sot", 0)
    ts_delta = total_shots - prev_state.get("last_total_shots", 0)
    sotr_delta = shots_off_target - prev_state.get("last_shots_off_target", 0)

    # v10.10: Same rate thresholds as GPS calculation
    if (sot_delta / mins_passed) >= 0.15:
        accel_count += 1
    if (ts_delta / mins_passed) >= 0.3:
        accel_count += 1
    if (sotr_delta / mins_passed) >= 0.2:  # replaces DA threshold
        accel_count += 1
    if xg_value is not None:
        prev_xg = prev_state.get("last_xg")
        if prev_xg is not None:
            try:
                xg_delta = xg_value - float(prev_xg)
                if xg_delta >= 0.1 and (xg_delta / mins_passed) >= 0.02:
                    accel_count += 1
            except (ValueError, TypeError):
                pass
    return accel_count


def is_fixture_pressure_accelerating(fid: int) -> bool:
    """v10: Check if ANY team in this fixture is pressure-accelerating.

    Used by polling interval logic to give 60s polling to fixtures
    where pressure is building BEFORE SOT reaches 2-3.
    """
    for (f, _), history in team_gps_history.items():
        if f == fid and len(history) >= 2:
            curr = history[-1]
            prev = history[-2]
            # GPS rising AND 2+ indicators accelerating
            if (curr.get("gps", 0) > prev.get("gps", 0)
                    and curr.get("accel_count", 0) >= 2):
                return True
    return False


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

        # v10.15: pre-window — allow from (MINUTE_MIN - PRE_WINDOW_MINUTES)
        # Signal sending still requires minute >= MINUTE_MIN
        # v10.13.2: Extended window for SOT-accelerating fixtures
        fid = fixture["fixture"]["id"]
        best_sot = get_fixture_best_sot(fid)
        sot_accel = fid in accelerating_fixtures or fid in pressure_accelerating
        is_extended = sot_accel or best_sot >= 2

        # v10.19.2: Post-restart safety — if team_state has no record for this
        # fixture (bot crashed and restarted), we don't know its SOT history.
        # Default to extended window so we don't miss SOT>=3 safety net triggers.
        if not is_extended and minute >= MINUTE_MIN:
            has_state = any(f == fid for f, _ in team_state)
            if not has_state:
                is_extended = True

        min_allowed = (MINUTE_MIN - PRE_WINDOW_MINUTES)
        # v10.20: Always use extended max for candidate inclusion.
        # Same chicken-and-egg as early window: can't know late SOT buildup
        # if we stop polling at 61'. LATE GATE protects signal quality.
        # v10.31: Event-extended fixtures get 90' ceiling (event SOT > stats SOT)
        max_allowed = EXTENDED_MAX
        if fid in event_extended_fixtures:
            max_allowed = 90
        if minute < min_allowed or minute > max_allowed:
            continue

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
    # v10: Clean pressure acceleration tracking
    for fid in list(pressure_accelerating):
        if fid not in live_fixture_ids:
            pressure_accelerating.discard(fid)
    # v10.17: Clean SOT burst tracking
    for fid in list(sot_burst_fixtures):
        if fid not in live_fixture_ids:
            sot_burst_fixtures.discard(fid)
    # v10: Clean GPS history
    to_delete_gps = [k for k in team_gps_history if k[0] not in live_fixture_ids]
    for k in to_delete_gps:
        del team_gps_history[k]
    # v10.31: Clean event SOT cache and extension set
    for fid in list(event_extended_fixtures):
        if fid not in live_fixture_ids:
            event_extended_fixtures.discard(fid)
    for fid in list(_event_sot_cache):
        if fid not in live_fixture_ids:
            del _event_sot_cache[fid]


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
    # v10.15: pre-window — allow monitoring from (MINUTE_MIN - PRE_WINDOW_MINUTES)
    # v10.13.2: Extended window for SOT-accelerating fixtures
    fid = fixture["fixture"]["id"]
    best_sot = get_fixture_best_sot(fid)
    sot_accel = fid in accelerating_fixtures or fid in pressure_accelerating
    is_extended = sot_accel or best_sot >= 2

    # v10.19.2: Post-restart safety — same logic as find_candidates()
    if not is_extended and minute >= MINUTE_MIN:
        has_state = any(f == fid for f, _ in team_state)
        if not has_state:
            is_extended = True

    min_allowed = (MINUTE_MIN - PRE_WINDOW_MINUTES)
    # v10.20: Always use extended max (same fix as find_candidates)
    # v10.31: Event-extended fixtures get 90' ceiling
    max_allowed = EXTENDED_MAX
    if fid in event_extended_fixtures:
        max_allowed = 90
    if minute < min_allowed or minute > max_allowed:
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
    """v10: Dynamic priority for stats polling order.

    v10 addition: pressure-accelerating fixtures (GPS rising + 2+ indicators)
    get boosted priority even at SOT=0-1, because they may be about
    to hit SOT=2-3.
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

    # v10: Boost for pressure-accelerating fixtures (pre-SOT acceleration)
    if fid in pressure_accelerating and best_sot < 2:
        base = max(base, 70)  # boost to SOT=2-level priority

    # v10.17: SOT burst — highest priority (2+ SOT in one poll)
    if fid in sot_burst_fixtures:
        base = max(base, 97)

    # v10.17: Sustained goalless pressure — team dominating but not scoring
    # Check if ANY team in this fixture has 3+ recent polls with GPS >= 50
    if best_sot >= 1:
        for (f, _), history in team_gps_history.items():
            if f == fid and len(history) >= 3:
                high_gps_streak = sum(1 for h in history if h.get("gps", 0) >= 50)
                if high_gps_streak >= 3:
                    base = max(base, 93)
                    break

    # v9.5.3: Penalize only if BOTH teams have signaled
    both_signaled = all(
        (fid, tid) in signaled_teams
        for (f, tid) in team_state if f == fid
    )
    if both_signaled and is_signaled:
        base = int(base * 0.5)

    return base


SECOND_CHANCE_BOOST = True  # v10.13: fast-repoll fixtures with SOT 1-2


def get_sot_based_interval(fid: int, base_interval: int) -> int:
    """v10.22: Pressure-aware polling interval with credit optimizations.

    v10.22 changes vs v10.21:
      - Edge zone gate: before 21' or after 60', 60s/90s fast polling only
        if best_sot>=2 OR max_gps>=50. Low-pressure games in edge zones
        can't signal (need GPS>=75) so fast polling wastes credits.
      - SOT=0 with state: 1.5x in ALL modes (was 1.0x NORMAL)
      - Pre-window SOT=0 (<21'): 2x base (can't signal, only building baseline)
      - Both-teams-signaled: 3x multiplier (was 2x, diminishing returns)
    """
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    team_signaled_count = sum(1 for (f, t) in signaled_teams if f == fid)
    both_teams_signaled = team_signaled_count >= 2

    # Get current minute for second-chance logic
    current_minute = 0
    for (f, _), state in team_state.items():
        if f == fid:
            current_minute = max(current_minute, state.get("last_minute", 0))

    # v10.22: Edge zone gate — before 21' or after 60', fast polling
    # only if genuinely high pressure (SOT>=2 OR GPS>=50).
    # These zones require GPS>=75 to signal anyway, so burning 60s
    # polls on SOT=0/1 GPS=30 fixtures is pure waste.
    edge_zone = is_in_edge_zone(fid)
    edge_allows_fast = (not edge_zone
                        or best_sot >= 2
                        or get_fixture_max_gps(fid) >= 50)

    # v10: Pressure acceleration — catches pre-SOT pressure build
    # This is THE key improvement: 60s polling starts when shots, DA,
    # and xG are ALL rising, even if SOT is still 0 or 1.
    if (fid in pressure_accelerating
            and not both_teams_signaled
            and best_sot < 3
            and edge_allows_fast):
        interval = 60
    # v10.17: SOT burst — 2+ SOT jump = extreme urgency, 60s polling
    elif fid in sot_burst_fixtures and not both_teams_signaled and edge_allows_fast:
        interval = 60
    # v9.7: SOT acceleration tier — still top priority
    elif fid in accelerating_fixtures and not both_teams_signaled and edge_allows_fast:
        interval = 60
    # v10.21: Second-chance boost — fast poll when SOT=2 and clock
    # is ticking (past 40'). v10.13 used SOT 1-2 but SOT=1 with GPS<30
    # is noise (one lucky shot). SOT=2 shows sustained pressure.
    elif (SECOND_CHANCE_BOOST
          and best_sot == 2
          and current_minute >= 40
          and not both_teams_signaled
          and edge_allows_fast):
        interval = 60
    elif not has_state:
        interval = base_interval
    elif best_sot >= 3 and not both_teams_signaled:
        interval = 90
    # v10.22: Fast SOT window (FastWin) also gated in edge zones
    elif best_sot >= 1 and not both_teams_signaled and is_fast_sot_active(fid) and edge_allows_fast:
        interval = 90
    elif best_sot >= 2:
        interval = base_interval  # v10.13: was 180, now 90s NORMAL
    elif best_sot == 1:
        interval = max(60, int(base_interval * 0.75))  # v10.13: was base, now 67s NORMAL
    else:
        # v10.21: SOT=0 with state — 1.5x in ALL modes (was 1.0x NORMAL)
        # If both teams have SOT=0, no signal is possible — slower polling is safe.
        interval = int(base_interval * 1.5)

    # v10.21: Pre-window SOT=0 — can't signal until 21', poll at 2x
    # Only building baseline state, no urgency.
    if best_sot == 0 and current_minute > 0 and current_minute < MINUTE_MIN:
        interval = max(interval, int(base_interval * 2.0))

    # v10.21: 3x if BOTH teams in this fixture have signaled (was 2x)
    # Both teams already signaled — diminishing returns on further polling.
    if both_teams_signaled:
        interval = int(interval * 3.0)

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
            check_signal_outcomes(f, client)

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
    # v10.10: Don't revive fixtures past our window (saves credits)
    for fid, (dh, da) in list(dead_fixtures.items()):
        f = find_cached_fixture(fid)
        if not f:
            dead_fixtures.pop(fid, None)
            continue
        # v10.10: Skip revival if match is past our window
        # v10.13.2: Allow revival if SOT-accelerating
        fixture_minute = safe_int(str(f["fixture"]["status"].get("elapsed", 0) or 0))
        best_sot = get_fixture_best_sot(fid)
        sot_accel = fid in accelerating_fixtures or fid in pressure_accelerating
        is_extended = sot_accel or best_sot >= 2
        # v10.20: Always use extended max for revival check
        # v10.31: Event-extended fixtures get 90' ceiling
        hard_max = EXTENDED_MAX
        if fid in event_extended_fixtures:
            hard_max = 90
        if fixture_minute > hard_max:
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
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN - PRE_WINDOW_MINUTES}-{EXTENDED_MAX}', signal from {MINUTE_MIN}+)")

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
    """Extract xG from team statistics. Returns string or 'N/A'.

    v10.10: Uses STAT_ALIASES for lookup (handles "expected_goals" key).
    """
    val = get_stat(tstats, "expected_goals", default=None)
    if val is not None and val != "0" and val != "N/A":
        try:
            return str(float(val))
        except (ValueError, TypeError):
            pass
    return "N/A"


def get_red_card_string(teams_data: dict, home_name: str, away_name: str) -> str:
    """Build red card string for the signal message.

    v10.1: Uses normalized alias lookup.
    """
    parts = []
    for tname, tstats in teams_data.items():
        rc = safe_int(get_stat(tstats, "red_cards"))
        if rc > 0:
            parts.append(f"{tname} - {rc}")

    if parts:
        return " | ".join(parts)
    return "None"


def save_outcome(entry: dict) -> None:
    """Append a signal outcome to the JSONL file."""
    try:
        with open(OUTCOMES_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        log.info(f"  Saved outcome to file: {entry.get('team_name')} F{entry.get('fixture_id')} (pending)")
    except Exception as e:
        log.warning(f"  Failed to save outcome to file: {e}")


def rewrite_outcomes_file() -> None:
    """v10.11: Rewrite the entire JSONL file with current in-memory state.

    Called after resolving outcomes to replace pending entries with resolved ones.
    This prevents duplicate entries (pending + resolved) in the file.
    """
    try:
        with open(OUTCOMES_FILE, "w") as f:
            for entry in signal_outcomes:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.warning(f"  Failed to rewrite outcomes file: {e}")


def load_all_outcomes() -> list[dict]:
    """v10.19.3: Load ALL signal outcomes from JSONL (resolved + pending).

    Previous version only loaded unresolved entries, which meant resolved
    outcomes were lost from memory after restart. When rewrite_outcomes_file()
    then ran, it would overwrite the file with only the pending entries,
    permanently deleting all resolved history.
    """
    all_entries = []
    if not os.path.exists(OUTCOMES_FILE):
        return all_entries
    try:
        with open(OUTCOMES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                all_entries.append(entry)
    except Exception as e:
        log.warning(f"  Failed to load outcomes file: {e}")
    return all_entries


# Keep old name as alias for any code that references it
load_pending_outcomes = load_all_outcomes


# v10.14: Auto-resolution via goal events API
_goal_events_cache: dict[int, list[dict]] = {}  # fixture_id -> list of goal event dicts

# v10.31: Event-based SOT supplement for late-window lag mitigation
# When /fixtures/statistics lags 2-3 min behind real-time (common 75'-90'),
# we fetch /fixtures/events (near-real-time) and count Shot/OnTarget + Goal
# events per team. effective_sot = max(stats_sot, events_sot).
# Only activated for fixtures with SOT>=2 at 75'+ (meaningful pressure).
# Cache TTL prevents excessive credit burn (1 credit per fetch).
event_extended_fixtures: set[int] = set()  # fixtures approved for 90' monitoring
_event_sot_cache: dict[int, dict] = {}     # {fid: {tid: sot, "ts": float, "credit_cost": int}}
EVENT_SOT_CACHE_TTL = 90              # seconds between re-fetches
EVENT_SOT_MINUTE = 75                 # minute threshold to activate event SOT
EVENT_SOT_MIN_PRESSURE = 2            # minimum best_sot to activate


def fetch_goal_events(client: httpx.Client, fixture_id: int) -> list[dict]:
    """v10.14: Fetch goal events for a finished fixture.

    Uses /fixtures/events?fixture=ID endpoint.
    Returns list of goal events with team_id, minute, and type info.
    Cached per fixture_id to avoid duplicate API calls.
    Each call costs 1 API credit.
    """
    if fixture_id in _goal_events_cache:
        return _goal_events_cache[fixture_id]

    try:
        data = api_get(client, "/fixtures/events", {"fixture": fixture_id})
        events = data.get("response", [])
        # Filter only goal events (exclude subs, cards, VAR, etc.)
        goal_events = []
        for ev in events:
            if ev.get("type") == "Goal":
                goal_events.append({
                    "team_id": ev.get("team", {}).get("id"),
                    "minute": safe_int(str(ev.get("time", {}).get("elapsed", 0) or 0)),
                    "minute_extra": safe_int(str(ev.get("time", {}).get("extra", 0) or 0)),
                    "detail": ev.get("detail", ""),  # "Normal Goal", "Penalty", "Own Goal", etc.
                    "player": ev.get("player", {}).get("name", ""),
                })
        _goal_events_cache[fixture_id] = goal_events
        log.info(f"  GOAL EVENTS: fixture {fixture_id} has {len(goal_events)} goal(s)")
        return goal_events
    except Exception as e:
        log.warning(f"  Failed to fetch goal events for fixture {fixture_id}: {e}")
        return []


def fetch_live_sot_from_events(
    client: httpx.Client, fixture_id: int,
) -> dict[int, int] | None:
    """v10.31: Count SOT from /fixtures/events (near-real-time, lag << statistics).

    API-Football's events endpoint updates faster than statistics because it's
    event-driven, not aggregated. A shot recorded at 82' appears in events
    within seconds, but may not reach /fixtures/statistics for 2-3 minutes.

    Counts per team:
      - type=="Shot" AND detail=="On target"  (saved, blocked, woodwork)
      - type=="Goal" AND detail!="Own Goal"   (goals are on target by definition)

    Returns {team_id: sot_count} or None on failure.
    Each call costs 1 API credit.
    """
    # Check cache
    cached = _event_sot_cache.get(fixture_id)
    if cached and time.time() - cached.get("ts", 0) < EVENT_SOT_CACHE_TTL:
        return {k: v for k, v in cached.items() if isinstance(k, int)}

    try:
        data = api_get(client, "/fixtures/events", {"fixture": fixture_id})
        events = data.get("response", [])

        sot_by_team: dict[int, int] = {}
        shot_count = 0
        goal_count = 0
        for ev in events:
            etype = ev.get("type", "")
            detail = ev.get("detail", "")
            tid_ev = ev.get("team", {}).get("id")
            if not tid_ev:
                continue

            if etype == "Shot" and detail == "On target":
                sot_by_team[tid_ev] = sot_by_team.get(tid_ev, 0) + 1
                shot_count += 1
            elif etype == "Goal" and detail != "Own Goal":
                # Goals are shots on target — include in SOT count
                sot_by_team[tid_ev] = sot_by_team.get(tid_ev, 0) + 1
                goal_count += 1

        # Cache with timestamp (include non-int keys for TTL)
        cache_entry = {"ts": time.time(), "credit_cost": 1}
        cache_entry.update(sot_by_team)
        _event_sot_cache[fixture_id] = cache_entry

        log.info(
            f"  EVENT SOT: F{fixture_id} — "
            f"{shot_count} shots on target + {goal_count} goals = "
            f"{dict((k, v) for k, v in sot_by_team.items() if isinstance(k, int))}"
        )
        return sot_by_team

    except Exception as e:
        log.warning(f"  EVENT SOT failed for F{fixture_id}: {e}")
        return None

def resolve_with_goal_events(
    entry: dict, goal_events: list[dict], home_id: int, away_id: int
) -> bool:
    """v10.14: Resolve a signal using precise goal event times.

    This replaces the old approach of comparing final score to goals_at_signal.
    With goal events, we know the EXACT minute each goal was scored,
    so we can precisely determine if a goal fell within 5/10/15 min windows.

    Args:
        entry: signal outcome dict (modified in place)
        goal_events: list of goal event dicts from fetch_goal_events
        home_id: home team API ID
        away_id: away team API ID

    Returns:
        True if any field was updated (needs file rewrite)
    """
    updated = False
    sig_minute = entry["game_minute"]
    team_id = entry["team_id"]
    is_home = entry["is_home"]

    # Find the correct team ID from events
    # (entry stores team_id, events use team.id from API)
    event_team_id = home_id if is_home else away_id

    # Filter goals for this team that happened AFTER the signal minute
    team_goals_after = [
        g for g in goal_events
        if g["team_id"] == event_team_id and g["minute"] > sig_minute
        and g["detail"] != "Own Goal"  # own goals don't count for pressure team
    ]

    # Sort by minute
    team_goals_after.sort(key=lambda g: g["minute"])

    first_goal = team_goals_after[0] if team_goals_after else None

    if first_goal:
        first_goal_minute = first_goal["minute"] + first_goal["minute_extra"]
        mins_to_goal = first_goal_minute - sig_minute

        # Resolve each time window
        for window, outcome_key, minute_key in [
            (5, "outcome_5min", "goal_minute_5"),
            (10, "outcome_10min", "goal_minute_10"),
            (15, "outcome_15min", "goal_minute_15"),
        ]:
            if entry.get(outcome_key) is None:
                if mins_to_goal <= window:
                    entry[outcome_key] = "HIT"
                    entry[minute_key] = first_goal_minute
                    updated = True
                else:
                    entry[outcome_key] = "MISS"
                    updated = True

        # Full match outcome
        if entry.get("outcome_full") is None:
            entry["outcome_full"] = "HIT"
            entry["goal_minute_full"] = first_goal_minute
            log.info(
                f"  GOAL-EVENT HIT: {entry['team_name']} scored at {first_goal_minute}' "
                f"(signal at {sig_minute}', +{mins_to_goal}') GPS={entry.get('gps', '?')} "
                f"[{entry.get('league', '?')}]"
            )
            updated = True
    else:
        # No goals after signal — all windows MISS
        for outcome_key in ("outcome_5min", "outcome_10min", "outcome_15min"):
            if entry.get(outcome_key) is None:
                entry[outcome_key] = "MISS"
                updated = True

        if entry.get("outcome_full") is None:
            entry["outcome_full"] = "MISS"
            log.info(
                f"  GOAL-EVENT MISS: {entry['team_name']} no goal after {sig_minute}' "
                f"GPS={entry.get('gps', '?')} [{entry.get('league', '?')}]"
            )
            updated = True

    # Mark as resolved
    if not entry.get("resolved"):
        entry["resolved"] = True
        entry["resolved_via"] = "goal_events"  # v10.14: track resolution method
        updated = True

    return updated


def check_signal_outcomes(fixture: dict, client: httpx.Client = None) -> None:
    """v10: Check pending signals for this fixture.

    Four-track system:
      outcome_5min:  HIT if goal within 5 game min
      outcome_10min: HIT if goal within 10 game min
      outcome_15min: HIT if goal within 15 game min
      outcome_full:  HIT if goal at any point before match ends
    Entry is 'resolved' only when ALL are decided (match must end).

    v10.14: When fixture is FT (finished), fetches /fixtures/events to get
    precise goal minutes for accurate window resolution. This replaces the
    old approach of comparing final score (which couldn't determine WHEN
    the goal was scored within the window).
    """
    global signal_outcomes
    fid = fixture["fixture"]["id"]
    status = fixture["fixture"]["status"]["short"]
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]

    # v10.14: For FT fixtures, use goal events API for precise resolution
    is_finished = status not in LIVE_STATUSES
    goal_events = None
    if is_finished and client is not None:
        goal_events = fetch_goal_events(client, fid)

    any_updated = False
    for entry in signal_outcomes:
        if entry["fixture_id"] != fid or entry["resolved"]:
            continue

        # v10.14: Use goal events for precise resolution when available
        if goal_events is not None:
            if resolve_with_goal_events(entry, goal_events, home_id, away_id):
                any_updated = True
            continue

        # --- Fallback: live tracking (no goal events, match still in progress) ---
        current_team_goals = home_goals if entry["is_home"] else away_goals
        goals_since_signal = current_team_goals - entry["goals_at_signal"]
        mins_since = minute - entry["game_minute"]

        if goals_since_signal > 0:
            mins_to_goal = minute - entry["game_minute"]

            # Check each time window
            for window, outcome_key, minute_key in [
                (5, "outcome_5min", "goal_minute_5"),
                (10, "outcome_10min", "goal_minute_10"),
                (15, "outcome_15min", "goal_minute_15"),
            ]:
                if entry.get(outcome_key) is None:
                    if mins_to_goal <= window:
                        entry[outcome_key] = "HIT"
                        entry[minute_key] = minute
                    elif mins_since >= window:
                        entry[outcome_key] = "MISS"

            if entry["outcome_full"] is None:
                entry["outcome_full"] = "HIT"
                entry["goal_minute_full"] = minute
                log.info(
                    f"  OUTCOME HIT: {entry['team_name']} scored at {minute}' "
                    f"(+{mins_to_goal}') GPS={entry.get('gps', '?')} [{entry['league']}]"
                )

            if status not in LIVE_STATUSES:
                entry["resolved"] = True
                entry["resolved_via"] = "live_score"  # v10.14: track resolution method
                any_updated = True

        else:
            # No goal yet — check windows
            for window, outcome_key in [(5, "outcome_5min"), (10, "outcome_10min"),
                                         (15, "outcome_15min")]:
                if entry.get(outcome_key) is None and mins_since >= window:
                    entry[outcome_key] = "MISS"

            if status not in LIVE_STATUSES:
                if entry["outcome_full"] is None:
                    entry["outcome_full"] = "MISS"
                    log.info(
                        f"  OUTCOME MISS: {entry['team_name']} no goal "
                        f"(signal at {entry['game_minute']}', GPS={entry.get('gps', '?')}, "
                        f"match ended {minute}') [{entry['league']}]"
                    )
                # Fill any remaining windows as MISS
                for outcome_key in ("outcome_5min", "outcome_10min", "outcome_15min"):
                    if entry.get(outcome_key) is None:
                        entry[outcome_key] = "MISS"
                entry["resolved"] = True
                entry["resolved_via"] = "live_score"
                any_updated = True

    # v10.14: Rewrite file if any entries were updated
    if any_updated:
        rewrite_outcomes_file()


def _log_outcome_block(label: str, resolved: list[dict]) -> None:
    """v10.19: Log a summary block for a given set of resolved entries."""
    if not resolved:
        return

    total = len(resolved)
    h5 = sum(1 for e in resolved if e.get("outcome_5min") == "HIT")
    h10 = sum(1 for e in resolved if e.get("outcome_10min") == "HIT")
    h15 = sum(1 for e in resolved if e.get("outcome_15min") == "HIT")
    hf = sum(1 for e in resolved if e.get("outcome_full") == "HIT")

    log.info(f"=== {label}: {total} signals ===")
    log.info(f"  5-min:   {h5}/{total} ({h5/total*100:.0f}%)")
    log.info(f"  10-min:  {h10}/{total} ({h10/total*100:.0f}%)")
    log.info(f"  15-min:  {h15}/{total} ({h15/total*100:.0f}%)")
    log.info(f"  Full:    {hf}/{total} ({hf/total*100:.0f}%)")

    # By tier
    for tier in ("CRITICAL", "EARLY WARNING"):
        tier_r = [e for e in resolved if e["tier"] == tier]
        if not tier_r:
            continue
        t_h5 = sum(1 for e in tier_r if e.get("outcome_5min") == "HIT")
        t_h15 = sum(1 for e in tier_r if e.get("outcome_15min") == "HIT")
        t_hf = sum(1 for e in tier_r if e.get("outcome_full") == "HIT")
        t_total = len(tier_r)
        avg_gps = sum(e.get("gps", 0) for e in tier_r) / t_total
        hits_15 = [e for e in tier_r if e.get("outcome_15min") == "HIT" and e.get("goal_minute_15")]
        avg_to_goal = None
        if hits_15:
            avg_to_goal = sum(e["goal_minute_15"] - e["game_minute"] for e in hits_15) / len(hits_15)
        goal_info = f", avg +{avg_to_goal:.0f}' to goal" if avg_to_goal else ""
        log.info(
            f"  {tier}: 5m {t_h5}/{t_total} 15m {t_h15}/{t_total} ({t_h15/t_total*100:.0f}%){goal_info} | "
            f"full {t_hf}/{t_total} ({t_hf/t_total*100:.0f}%) | avg GPS: {avg_gps:.0f}"
        )

    # GPS-triggered vs SOT-triggered
    gps_sigs = [e for e in resolved if e.get("gps_triggered")]
    sot_sigs = [e for e in resolved if not e.get("gps_triggered")]
    if gps_sigs:
        g_h5 = sum(1 for e in gps_sigs if e.get("outcome_5min") == "HIT")
        g_h15 = sum(1 for e in gps_sigs if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in gps_sigs if e.get("outcome_full") == "HIT")
        g_t = len(gps_sigs)
        avg_gps = sum(e.get("gps", 0) for e in gps_sigs) / g_t
        avg_sot = sum(e.get("sot", 0) for e in gps_sigs) / g_t
        log.info(
            f"  GPS-TRIGGERED (EARLY WARNING): 5m {g_h5}/{g_t} 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | "
            f"full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | avg GPS: {avg_gps:.0f}, avg SOT: {avg_sot:.1f}"
        )
    if sot_sigs:
        s_h15 = sum(1 for e in sot_sigs if e.get("outcome_15min") == "HIT")
        s_hf = sum(1 for e in sot_sigs if e.get("outcome_full") == "HIT")
        s_t = len(sot_sigs)
        log.info(
            f"  SOT-TRIGGERED (CRITICAL): 15min {s_h15}/{s_t} ({s_h15/s_t*100:.0f}%) | "
            f"full {s_hf}/{s_t} ({s_hf/s_t*100:.0f}%)"
        )

    # v10.27: By window tag (CORE / EARLY_OVERRIDE / LATE_OVERRIDE)
    for wt in ("CORE", "EARLY_OVERRIDE", "LATE_OVERRIDE"):
        group = [e for e in resolved if e.get("window_tag") == wt]
        if not group:
            continue
        w_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        w_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        w_t = len(group)
        avg_gps = sum(e.get("gps", 0) for e in group) / w_t
        avg_min = sum(e.get("game_minute", 0) for e in group) / w_t
        log.info(
            f"  {wt}: 15min {w_h15}/{w_t} ({w_h15/w_t*100:.0f}%) | "
            f"full {w_hf}/{w_t} ({w_hf/w_t*100:.0f}%) | "
            f"avg GPS: {avg_gps:.0f}, avg min: {avg_min:.0f}'"
        )

    # By GPS score range
    for range_label, range_filter in [
        ("GPS 55-64", lambda e: 55 <= e.get("gps", 0) < 65),
        ("GPS 65-74", lambda e: 65 <= e.get("gps", 0) < 75),
        ("GPS 75-84", lambda e: 75 <= e.get("gps", 0) < 85),
        ("GPS 85+", lambda e: e.get("gps", 0) >= 85),
    ]:
        group = [e for e in resolved if range_filter(e)]
        if not group:
            continue
        g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        g_t = len(group)
        avg_accel = sum(e.get("accel_count", 0) for e in group) / g_t
        log.info(
            f"  {range_label}: 15min {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | "
            f"full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | avg accel: {avg_accel:.1f}"
        )


def log_outcome_summary() -> None:
    """v10: Log hit-rate summary for all resolved signals today.

    v10.19: Shows BOTH first-signal-only and all-signals blocks,
    matching the Telegram /stats output format.
    """
    resolved_first = [e for e in signal_outcomes if e["resolved"] and e.get("sig_num", 1) == 1]
    resolved_all = [e for e in signal_outcomes if e["resolved"]]

    if not resolved_first and not resolved_all:
        return

    log.info("")

    if resolved_first:
        _log_outcome_block("SIGNAL OUTCOME SUMMARY (1st signal only)", resolved_first)

    # If there are 2nd/3rd signals, show all-signals block too
    if resolved_all and len(resolved_all) != len(resolved_first):
        log.info("")
        _log_outcome_block("SIGNAL OUTCOME SUMMARY (all signals)", resolved_all)


def resolve_stale_outcomes(client: httpx.Client) -> int:
    """v10.11: Resolve pending outcomes from finished fixtures.

    When the bot sleeps before matches end, pending outcomes stay unresolved.
    This function fetches finished fixtures by ID and resolves them.
    Called on startup and periodically during idle.
    Returns number of newly resolved outcomes.
    """
    global signal_outcomes
    pending = [e for e in signal_outcomes if not e["resolved"]]
    if not pending:
        return 0

    # Collect unique fixture IDs (batch up to 20 per API call)
    fixture_ids = list(set(e["fixture_id"] for e in pending))
    resolved_count = 0

    for i in range(0, len(fixture_ids), BATCH_SIZE_LIMIT):
        batch = fixture_ids[i:i + BATCH_SIZE_LIMIT]
        ids_param = "-".join(str(fid) for fid in batch)
        try:
            data = api_get(client, "/fixtures", {"ids": ids_param})
            fixtures = data.get("response", [])
            for f in fixtures:
                check_signal_outcomes(f, client)
        except Exception as e:
            log.warning(f"  Stale outcome resolution failed for batch {batch}: {e}")
            continue

    # Check how many got resolved
    still_pending = [e for e in signal_outcomes if not e["resolved"]]
    resolved_count = len(pending) - len(still_pending)
    if resolved_count > 0:
        log.info(f"Resolved {resolved_count} stale outcome(s), {len(still_pending)} still pending")
    elif still_pending:
        log.info(f"{len(still_pending)} outcome(s) still pending (fixtures may not be finished yet)")

    return resolved_count


def load_all_outcomes() -> list[dict]:
    """v10.11: Load ALL signal outcomes from JSONL (resolved + unresolved).

    Used for /stats command to show historical win rate.
    """
    all_entries = []
    if not os.path.exists(OUTCOMES_FILE):
        return all_entries
    try:
        with open(OUTCOMES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                all_entries.append(entry)
    except Exception as e:
        log.warning(f"  Failed to load outcomes file: {e}")
    return all_entries


def _format_stats_block(entries: list[dict], label: str) -> list[str]:
    """v10.18: Format a stats block for a given set of entries."""
    resolved = [e for e in entries if e.get("resolved")]
    pending = [e for e in entries if not e.get("resolved")]
    lines = []
    lines.append(f"📊 {label} ({len(entries)} total)")
    lines.append("")

    if resolved:
        total = len(resolved)
        h5 = sum(1 for e in resolved if e.get("outcome_5min") == "HIT")
        h10 = sum(1 for e in resolved if e.get("outcome_10min") == "HIT")
        h15 = sum(1 for e in resolved if e.get("outcome_15min") == "HIT")
        hf = sum(1 for e in resolved if e.get("outcome_full") == "HIT")

        lines.append(f"✅ Resolved: {total}")
        # v10.28: Full WR is PRIMARY KPI — show first and prominently
        lines.append(f"  ⭐ Full WR: {hf}/{total} ({hf/total*100:.0f}%)")
        lines.append(f"  Timing: 5m {h5}/{total} | 10m {h10}/{total} | 15m {h15}/{total}")
        lines.append("")

        # By tier
        for tier in ("CRITICAL", "EARLY WARNING"):
            tier_r = [e for e in resolved if e.get("tier") == tier]
            if not tier_r:
                continue
            t = len(tier_r)
            t_h15 = sum(1 for e in tier_r if e.get("outcome_15min") == "HIT")
            t_hf = sum(1 for e in tier_r if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in tier_r) / t
            lines.append(f"  {tier}: full {t_hf}/{t} ({t_hf/t*100:.0f}%) | 15m {t_h15}/{t} ({t_h15/t*100:.0f}%) | avg GPS {avg_gps:.0f}")

        lines.append("")

        # By trigger type — v10.28: full WR first
        gps_sigs = [e for e in resolved if e.get("gps_triggered")]
        sot_sigs = [e for e in resolved if not e.get("gps_triggered")]
        if gps_sigs:
            g_t = len(gps_sigs)
            g_h15 = sum(1 for e in gps_sigs if e.get("outcome_15min") == "HIT")
            g_hf = sum(1 for e in gps_sigs if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in gps_sigs) / g_t
            avg_sot = sum(e.get("sot", 0) for e in gps_sigs) / g_t
            lines.append(f"  GPS-triggered (EW): full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | avg GPS {avg_gps:.0f} SOT {avg_sot:.1f}")
        if sot_sigs:
            s_t = len(sot_sigs)
            s_h15 = sum(1 for e in sot_sigs if e.get("outcome_15min") == "HIT")
            s_hf = sum(1 for e in sot_sigs if e.get("outcome_full") == "HIT")
            lines.append(f"  SOT-triggered (CRITICAL): full {s_hf}/{s_t} ({s_hf/s_t*100:.0f}%) | 15m {s_h15}/{s_t} ({s_h15/s_t*100:.0f}%)")

        lines.append("")

        # v10.27/28: By window tag — full WR first
        for wt_label, wt_key in [("CORE (21-60')", "CORE"), ("EARLY OVERRIDE (<21')", "EARLY_OVERRIDE"), ("LATE OVERRIDE (61'+)", "LATE_OVERRIDE")]:
            group = [e for e in resolved if e.get("window_tag") == wt_key]
            if not group:
                continue
            w_t = len(group)
            w_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
            w_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in group) / w_t
            lines.append(f"  {wt_label}: full {w_hf}/{w_t} ({w_hf/w_t*100:.0f}%) | 15m {w_h15}/{w_t} ({w_h15/w_t*100:.0f}%) | avg GPS {avg_gps:.0f}")

        lines.append("")

        # By GPS range
        for range_label, range_filter in [
            ("GPS 55-64", lambda e: 55 <= e.get("gps", 0) < 65),
            ("GPS 65-74", lambda e: 65 <= e.get("gps", 0) < 75),
            ("GPS 75-84", lambda e: 75 <= e.get("gps", 0) < 85),
            ("GPS 85+", lambda e: e.get("gps", 0) >= 85),
        ]:
            group = [e for e in resolved if range_filter(e)]
            if not group:
                continue
            g_t = len(group)
            g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
            g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
            lines.append(f"  {range_label}: full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%)")

        lines.append("")

        # By minute range
        for min_label, min_filter in [
            ("21-35'", lambda e: 21 <= e.get("game_minute", 0) <= 35),
            ("36-45'", lambda e: 36 <= e.get("game_minute", 0) <= 45),
            ("46-55'", lambda e: 46 <= e.get("game_minute", 0) <= 55),
            ("56-61'", lambda e: 56 <= e.get("game_minute", 0) <= 61),
        ]:
            group = [e for e in resolved if min_filter(e)]
            if not group:
                continue
            g_t = len(group)
            g_h5 = sum(1 for e in group if e.get("outcome_5min") == "HIT")
            g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
            g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in group) / g_t
            lines.append(f"  {min_label}: full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | 5m {g_h5}/{g_t} | avg GPS {avg_gps:.0f}")

        lines.append("")

        # v10.19: GPS x minute cross-tab (first signals only)
        # Rows = GPS ranges, Columns = minute buckets
        minute_buckets = [
            ("9-35'", lambda e: 9 <= e.get("game_minute", 0) <= 35),
            ("36-45'", lambda e: 36 <= e.get("game_minute", 0) <= 45),
            ("46-61'", lambda e: 46 <= e.get("game_minute", 0) <= 61),
            ("62-85'", lambda e: 62 <= e.get("game_minute", 0) <= 85),
        ]
        gps_rows = [
            ("GPS 55-64", lambda e: 55 <= e.get("gps", 0) < 65),
            ("GPS 65-74", lambda e: 65 <= e.get("gps", 0) < 75),
            ("GPS 75-84", lambda e: 75 <= e.get("gps", 0) < 85),
            ("GPS 85+", lambda e: e.get("gps", 0) >= 85),
        ]
        # Check if any cell has data
        has_cross_data = False
        for _, gfilter in gps_rows:
            for _, mfilter in minute_buckets:
                if [e for e in resolved if gfilter(e) and mfilter(e)]:
                    has_cross_data = True
                    break
            if has_cross_data:
                break
        if has_cross_data:
            lines.append("GPS x Minute (full WR):")
            # Header
            header = "             " + "  ".join(f"{ml:>8}" for ml, _ in minute_buckets)
            lines.append(header)
            for glabel, gfilter in gps_rows:
                row_parts = [f"{glabel:<12}"]
                for _, mfilter in minute_buckets:
                    cell = [e for e in resolved if gfilter(e) and mfilter(e)]
                    if cell:
                        ch = sum(1 for e in cell if e.get("outcome_full") == "HIT")
                        ct = len(cell)
                        row_parts.append(f"{ch}/{ct}({ch/ct*100:.0f}%)")
                    else:
                        row_parts.append(f"{'---':>8}")
                lines.append("  ".join(row_parts))
            lines.append("")

        # Last 10 resolved signals detail — v10.28: full outcome first, show recency_ratio
        recent = resolved[-10:]
        lines.append("📋 Last signals:")
        for e in recent:
            of_ = e.get("outcome_full", "?")
            o15 = e.get("outcome_15min", "?")
            icon = "✅" if of_ == "HIT" else "❌"
            gm = e.get("goal_minute_full") or e.get("goal_minute_15", "")
            gm_str = f" goal@{gm}'" if gm else ""
            trigger = "GPS" if e.get("gps_triggered") else "SOT"
            rr = e.get("recency_ratio")
            rr_str = f" RR:{rr:.2f}" if rr is not None else ""
            lines.append(f"  {icon} {e.get('team_name','?')} | {e.get('league','?')} | GPS {e.get('gps','?')} | SOT {e.get('sot','?')} | {trigger} | full:{of_} 15m:{o15}{gm_str}{rr_str}")

        # v10.28: Recency ratio analysis (if data available)
        with_rr = [e for e in resolved if e.get("recency_ratio") is not None]
        if len(with_rr) >= 5:
            lines.append("")
            lines.append("🔄 Recency Analysis (fresh vs accumulated pressure):")
            # Bin by recency_ratio: low (accumulated) vs high (fresh)
            mid = 0.3
            low_rr = [e for e in with_rr if e.get("recency_ratio", 0) < mid]
            high_rr = [e for e in with_rr if e.get("recency_ratio", 0) >= mid]
            for label, group in [(f"RR <{mid} (accumulated)", low_rr), (f"RR ≥{mid} (fresh pressure)", high_rr)]:
                if len(group) < 2:
                    continue
                gt = len(group)
                ghf = sum(1 for e in group if e.get("outcome_full") == "HIT")
                gh15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
                avg_rr = sum(e.get("recency_ratio", 0) for e in group) / gt
                avg_gps = sum(e.get("gps", 0) for e in group) / gt
                avg_accel = sum(e.get("accel_count", 0) for e in group) / gt
                lines.append(f"  {label}: full {ghf}/{gt} ({ghf/gt*100:.0f}%) | 15m {gh15}/{gt} ({gh15/gt*100:.0f}%) | avg RR {avg_rr:.2f} GPS {avg_gps:.0f} accel {avg_accel:.1f}")
    else:
        lines.append("No resolved signals yet.")

    if pending:
        lines.append("")
        lines.append(f"⏳ Pending: {len(pending)} (awaiting match end or resolution)")

    return lines


def format_outcome_stats(entries: list[dict]) -> str:
    """v10.18: Format outcome statistics showing BOTH counting methods.

    Shows two blocks:
    1. 1st SIGNAL ONLY (per team per fixture) — the fair judge
    2. ALL SIGNALS (every signal fired) — for comparison
    """
    if not entries:
        return "No signal data yet."

    first_sig = [e for e in entries if e.get("sig_num", 1) == 1]
    all_sig = entries

    # If both sets are identical (no duplicates), show single block
    if len(first_sig) == len(all_sig):
        lines = _format_stats_block(first_sig, "SIGNAL PERFORMANCE")
        return "\n".join(lines)

    # Show both blocks for comparison
    lines = _format_stats_block(first_sig, "SIGNAL PERFORMANCE — 1st signal only")
    lines.append("")
    lines.append("─" * 20)
    lines.append("")
    lines_all = _format_stats_block(all_sig, "SIGNAL PERFORMANCE — all signals")
    lines.extend(lines_all)

    return "\n".join(lines)


def send_end_of_day_summary(client: httpx.Client, days: int = 1) -> None:
    """v10.13: Send end-of-day win rate summary via Telegram.

    v10.13 fix: Loads from JSONL FILE (not in-memory) because signal_outcomes
    gets cleared before this runs. This was the root cause of missing EOD stats.

    Args:
        days: How many days back to include (1=today, 3=past 3 days)
    Called once when the bot exits active hours (all matches ended).
    Reuses format_outcome_stats() so the output matches /stats.
    """
    global eod_summary_sent_date

    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
    # For auto EOD (days=1), guard against duplicate sends
    if days == 1 and eod_summary_sent_date == today_str:
        return  # Already sent today
    # v10.16: Also check persisted marker (survives restarts)
    if days == 1 and _load_eod_sent_date() == today_str:
        log.info(f"v10.16: EOD already sent for {today_str} (persisted marker), skipping")
        eod_summary_sent_date = today_str  # sync in-memory
        return

    # v10.13: ALWAYS load from file — in-memory may be cleared
    all_entries = load_all_outcomes()
    if not all_entries:
        log.info("v10.13: No outcome data in file, skipping EOD summary")
        return

    # Filter to requested day range
    # v10.16: Use days=2 window to catch signals from the date that just ended.
    # EOD typically fires around midnight when signals carry yesterday's date.
    effective_days = max(days, 2)
    cutoff_date = (datetime.now(BULGARIA_TZ) - timedelta(days=effective_days - 1)).strftime("%Y-%m-%d")
    filtered = []
    for e in all_entries:
        sig_date = e.get("signal_clock", "")[:10]
        if sig_date >= cutoff_date:
            filtered.append(e)

    if not filtered:
        log.info(f"v10.16: No signals in last {effective_days} day(s), skipping EOD summary")
        return

    resolved = [e for e in filtered if e.get("resolved")]
    if not resolved:
        log.info(f"v10.16: {len(filtered)} signals but none resolved yet")
        return

    pending = [e for e in filtered if not e.get("resolved")]
    day_label = "TODAY" if days == 1 else f"PAST {effective_days} DAYS"
    log.info(f"v10.13: Sending EOD summary {day_label} ({len(resolved)} resolved, {len(pending)} pending)")

    stats_text = format_outcome_stats(filtered)
    sent = send_telegram(client, stats_text)
    if sent and days == 1:
        eod_summary_sent_date = today_str
        _save_eod_sent_date(today_str)
        log.info("v10.16: End-of-day summary sent (persisted)")
    elif not sent:
        log.warning("v10.13: End-of-day Telegram send FAILED")


def check_telegram_commands(client: httpx.Client) -> None:
    """v10.18: Check for Telegram commands.

    Polls getUpdates with long polling disabled (quick check).
    /stats    = all-time stats (1st signal vs all signals)
    /stats3d  = past 3 days
    /stats7d  = past 7 days
    /recap    = yesterday's stats (was auto-sent, now command-only)
    /today    = match list (no form/scorers, only if <10 games)
    /matches  = alias for /today
    Runs once per main loop iteration (minimal overhead).
    """
    try:
        resp = client.post(
            f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            json={"timeout": 0, "allowed_updates": ["message"]},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return
        results = resp.json().get("result", [])
        for update in results:
            msg = update.get("message", {})
            text = (msg.get("text") or "").strip()
            chat_id = msg.get("chat", {}).get("id")
            update_id = update.get("update_id")

            # Only respond to our configured chat
            if str(chat_id) != str(TELEGRAM_CHAT_ID):
                # Acknowledge to clear from queue
                if update_id:
                    client.post(
                        f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                        json={"offset": update_id + 1, "timeout": 0},
                        timeout=5.0,
                    )
                continue

            if text == "/stats":
                # First resolve any stale outcomes
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                # Load ALL outcomes from file (not just in-memory)
                all_entries = load_all_outcomes()
                stats_text = format_outcome_stats(all_entries)
                send_telegram(client, stats_text)

            elif text == "/stats3d":
                # Past 3 days stats
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                cutoff = (datetime.now(BULGARIA_TZ) - timedelta(days=3)).strftime("%Y-%m-%d")
                filtered = [e for e in all_entries if e.get("signal_clock", "")[:10] >= cutoff]
                if not filtered:
                    send_telegram(client, f"No signals in the past 3 days (since {cutoff}).")
                else:
                    header = f"STATS: PAST 3 DAYS ({cutoff} to today)"
                    stats_text = header + "\n" + format_outcome_stats(filtered)
                    send_telegram(client, stats_text)

            elif text == "/stats7d":
                # Past 7 days stats
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                cutoff = (datetime.now(BULGARIA_TZ) - timedelta(days=7)).strftime("%Y-%m-%d")
                filtered = [e for e in all_entries if e.get("signal_clock", "")[:10] >= cutoff]
                if not filtered:
                    send_telegram(client, f"No signals in the past 7 days (since {cutoff}).")
                else:
                    header = f"STATS: PAST 7 DAYS ({cutoff} to today)"
                    stats_text = header + "\n" + format_outcome_stats(filtered)
                    send_telegram(client, stats_text)

            elif text == "/matches" or text == "/today":
                # v10.18: On-demand match list ONLY (no form/scorers), only if <10 games
                _send_today_command(client)

            elif text == "/recap":
                # v10.18: Yesterday's recap (was auto-sent, now command-only)
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                yesterday_str = (datetime.now(BULGARIA_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
                yesterday_signals = [e for e in all_entries
                                    if e.get("signal_clock", "")[:10] == yesterday_str]
                if not yesterday_signals:
                    send_telegram(client, f"No signals yesterday ({yesterday_str}).")
                else:
                    header = f"MORNING RECAP: {yesterday_str}"
                    stats_text = header + "\n" + format_outcome_stats(yesterday_signals)
                    send_telegram(client, stats_text)

            # Acknowledge update to clear from queue
            if update_id:
                client.post(
                    f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                    json={"offset": update_id + 1, "timeout": 0},
                    timeout=5.0,
                )
    except Exception:
        pass  # Non-critical, don't spam logs


def send_daily_summary(client: httpx.Client) -> None:
    """v10.17: Send daily match list to Telegram (0 extra credits).

    v10.17: Removed auto-sent team form + top scorers (was burning 40-80 credits/day).
    Match list only. Team form + scorers available via /matches command.
    """
    global daily_summary_date

    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
    if daily_summary_date == today_str:
        return  # Already sent today
    if not todays_tracked_fixtures:
        log.info("v10.12: No tracked fixtures, skipping daily summary")
        return

    log.info(f"v10.17: Building daily summary for {len(todays_tracked_fixtures)} match(es)...")

    lines = []
    lines.append(f"\u26bd TODAY'S MATCHES ({today_str})")
    lines.append("")

    for fix in todays_tracked_fixtures:
        home = fix["teams"]["home"]
        away = fix["teams"]["away"]
        league_name = fix["league"].get("name", "?")
        try:
            kickoff_utc = datetime.fromisoformat(
                fix["fixture"]["date"].replace("Z", "+00:00")
            )
            kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
            ko_str = kickoff_local.strftime("%H:%M")
        except Exception:
            ko_str = "??:??"
        lines.append(f"\U0001f3c6 {league_name} — {ko_str} Bulgaria")
        lines.append(f"{home['name']} vs {away['name']}")
        lines.append("")

    msg = "\n".join(lines)
    sent = send_telegram(client, msg)

    if sent:
        daily_summary_date = today_str
        log.info(f"v10.17: Daily summary sent ({len(todays_tracked_fixtures)} matches, 0 extra credits)")
    else:
        log.warning("v10.17: Daily summary Telegram send FAILED — will retry next wake-up")


def _send_today_command(client: httpx.Client) -> None:
    """v10.18: /today or /matches command — match list ONLY (no form, no scorers).

    Only works when <10 games. 0 extra API credits.
    Team form and scorers removed to save credits.
    """
    if not todays_tracked_fixtures:
        send_telegram(client, "No tracked matches today.")
        return

    if len(todays_tracked_fixtures) >= 10:
        send_telegram(
            client,
            f"Too many matches today ({len(todays_tracked_fixtures)}). "
            f"/today only available when <10 games to avoid spam."
        )
        return

    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
    lines = []
    lines.append(f"\u26bd TODAY'S MATCHES ({today_str})")
    lines.append("")

    for fix in todays_tracked_fixtures:
        home = fix["teams"]["home"]
        away = fix["teams"]["away"]
        league_name = fix["league"].get("name", "?")
        try:
            kickoff_utc = datetime.fromisoformat(
                fix["fixture"]["date"].replace("Z", "+00:00")
            )
            kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
            ko_str = kickoff_local.strftime("%H:%M")
        except Exception:
            ko_str = "??:??"
        lines.append(f"\U0001f3c6 {league_name} — {ko_str} Bulgaria")
        lines.append(f"{home['name']} vs {away['name']}")
        lines.append("")

    msg = "\n".join(lines)
    send_telegram(client, msg)
    log.info(f"v10.18: /today sent ({len(todays_tracked_fixtures)} matches, 0 extra credits)")


def process_fixture_stats(client: httpx.Client, fixture: dict) -> None:
    """Process one fixture from batched /fixtures?ids=... response.

    v9.5: New signal format with xG, top SOT player, red cards.
    Strict 80' cutoff. Tracks signaled fixtures for diversification.
    """
    fid = fixture["fixture"]["id"]

    # v9.8: Check outcome of pending signals FIRST (before early returns)
    # This catches: goals scored since last poll, fixtures that just ended
    check_signal_outcomes(fixture, client)

    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        return

    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    home_tid = home["id"]
    away_tid = away["id"]

    statistics = fixture.get("statistics") or []
    if not statistics:
        return

    # Parse all team stats into a dict keyed by team name AND team ID
    teams_data = {}
    teams_data_by_id = {}  # v10.31: ID-based fallback when names don't match
    for team_entry in statistics:
        team = team_entry.get("team", {})
        tname = team.get("name")
        tid_stat = team.get("id")  # v10.31: also key by ID
        if not tname:
            continue
        tmap = {}
        for stat in team_entry.get("statistics", []):
            stat_type = stat.get("type")
            value = stat.get("value")
            if stat_type:
                # v10.1: store BOTH original key and stripped version
                # so alias lookup always has something to find
                # v10.19: Treat None AND literal "N/A" as missing data (→ "0")
                # This prevents GPS from being calculated on incomplete stats.
                if value is None or str(value).strip() in ("", "N/A"):
                    tmap[stat_type] = "0"
                else:
                    tmap[stat_type] = str(value).strip()
                if stat_type != stat_type.strip():
                    tmap[stat_type.strip()] = tmap[stat_type]
        teams_data[tname] = tmap
        if tid_stat:
            teams_data_by_id[tid_stat] = tmap

    if not teams_data:
        return

    # --- Fast SOT window management (fixture-level) ---
    # Check BOTH teams' SOT. Activate if either >= 2.
    # v9.5: Only activate for unsignaled fixtures.
    # v10.1: Use get_stat() for all field access (normalized aliases)
    best_current_sot = 0
    for tname, tstats in teams_data.items():
        current_sot = safe_int(get_stat(tstats, "sot"))
        best_current_sot = max(best_current_sot, current_sot)

    # v10.21: Activate fast window only if SOT increased since last activation.
    # Prevents infinite re-activation loop when SOT is stuck (e.g. SOT=1 for 10+ min).
    if best_current_sot >= 1:
        team_sig_count = sum(1 for (f, t) in signaled_teams if f == fid)
        if team_sig_count < 2 and not is_fast_sot_active(fid):
            last_activated_sot = fast_sot_activated_at_sot.get(fid, 0)
            if best_current_sot > last_activated_sot:
                activate_fast_sot(fid, best_current_sot)
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

    # v10.31: EVENT-BASED SOT SUPPLEMENT — close the 2-3 min statistics lag.
    # After 75' with meaningful pressure, fetch /fixtures/events (near-real-time)
    # to detect SOT jumps that statistics hasn't caught up to yet.
    # effective_sot = max(stats_sot, events_sot) — audit trail preserved.
    event_sot_by_team: dict[int, int] = {}
    if minute >= EVENT_SOT_MINUTE and best_current_sot >= EVENT_SOT_MIN_PRESSURE:
        ev_sot = fetch_live_sot_from_events(client, fid)
        if ev_sot:
            event_sot_by_team = ev_sot
            # Check if events reveal higher SOT than statistics for ANY team
            for tid_check, ev_sot_val in ev_sot.items():
                if ev_sot_val > best_current_sot:
                    event_extended_fixtures.add(fid)
                    log.info(
                        f"  EVENT-EXTEND: F{fid} {minute}' — events SOT={ev_sot_val} "
                        f"> stats best={best_current_sot}, monitoring extended to 90'"
                    )
                    break

    # v10.31: Deferred hard_max check — now after stats parsing + events.
    # Event-extended fixtures (events SOT > stats SOT) get 90' ceiling.
    hard_max = EXTENDED_MAX
    if fid in event_extended_fixtures:
        hard_max = 90
    if minute > hard_max:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        # Clean up event extension when fixture exits
        event_extended_fixtures.discard(fid)
        _event_sot_cache.pop(fid, None)
        log.info(f"  Fixture {fid} past {hard_max}', removed from monitoring")
        return

    # --- Pre-compute red cards and xG for both teams ---
    red_card_str = get_red_card_string(teams_data, home["name"], away["name"])

    # Parse xG for both teams
    team_xg = {}
    for tname, tstats in teams_data.items():
        team_xg[tname] = parse_xg(tstats)

    # --- SOT SIGNAL CHECK (per team) ---
    # v10: Calculate GPS for every team on every poll (zero extra API cost).
    # This builds the dataset and drives acceleration-aware polling.
    league = LEAGUE_IDS.get(
        fixture["league"]["id"], fixture["league"].get("name", "?")
    )
    sh = fixture["goals"]["home"] or 0
    sa = fixture["goals"]["away"] or 0

    # v10.22: Safety net — initialize gps before per-team loop.
    # Prevents rare 'gps not defined' NameError when both teams
    # hit DATA_INCOMPLETE skip, causing fixture to crash on every poll,
    # burn 1 credit each time, and never save state (infinite retry loop).
    gps = 0.0
    # v10.31: Same safety net for 'state' — Python 3.13+ raises
    # 'cannot access local variable' when both teams skip via
    # 'if not tstats: continue' before the assignment inside the loop.
    state = None

    for tid, tname in ((home_tid, home["name"]), (away_tid, away["name"])):
        tstats = teams_data.get(tname)
        # v10.31: Fallback to ID-based lookup when team names don't match
        # (e.g. /fixtures?ids= returns different name format than statistics)
        if not tstats:
            tstats = teams_data_by_id.get(tid)
        if not tstats:
            log.debug(f"  No stats found for {tname} (tid={tid}) F{fid} — name/ID mismatch")
            continue

        # v10.1: All stat access via get_stat() (normalized aliases)
        stats_sot = safe_int(get_stat(tstats, "sot"))

        # v10.31: Event-based SOT supplement — use higher of stats vs events.
        # Events endpoint updates in seconds; statistics lags 2-3 min.
        # This closes the blind spot for 75'-90' SOT jumps.
        events_sot = event_sot_by_team.get(tid, 0)
        sot = max(stats_sot, events_sot)
        if events_sot > stats_sot:
            log.info(
                f"  F{fid} {minute}' {tname}: "
                f"stats_sot={stats_sot} | events_sot={events_sot} | effective_sot={sot}"
            )

        # v10.10: Parse ALL available stats for GPS calculation
        total_shots = safe_int(get_stat(tstats, "total_shots"))

        # v10.13.1: Data sanity check — reject clearly corrupted API responses
        # v10.31: Use stats_sot (not effective_sot) — these check the
        # statistics API response quality, not the events supplement.
        _data_corrupt = False
        if stats_sot > total_shots and total_shots > 0:
            _data_corrupt = True
            log.warning(
                f"  DATA CORRUPT: {tname} F{fid} {minute}' — "
                f"SOT({stats_sot}) > Total Shots({total_shots}), skipping signal"
            )
        if stats_sot >= 3 and total_shots == 0:
            _data_corrupt = True
            log.warning(
                f"  DATA CORRUPT: {tname} F{fid} {minute}' — "
                f"SOT({stats_sot}) but Total Shots=0, skipping signal"
            )
        # v10.19: Detect completely missing stat response from API
        # If total_shots=0 AND sot=0 AND shots_inside_box will be 0,
        # the API likely returned N/A for everything — GPS would be meaningless.
        if total_shots == 0 and stats_sot == 0:
            _data_incomplete = True
        else:
            _data_incomplete = False
        shots_off_target = safe_int(get_stat(tstats, "shots_off_target"))  # v10.10: replaces DA
        shots_inside_box = safe_int(get_stat(tstats, "shots_inside_box"))
        corners = safe_int(get_stat(tstats, "corner_kicks"))
        xg_str = team_xg.get(tname, "N/A")
        xg_value = safe_float(xg_str)
        possession = safe_int(get_stat(tstats, "possession"))

        # Get opponent SOT and xG (v10.1: normalized)
        opponent_sot = "0"
        opponent_xg = "N/A"
        for oname, ostats in teams_data.items():
            if oname != tname:
                opponent_sot = str(safe_int(get_stat(ostats, "sot")))
                opponent_xg = team_xg.get(oname, "N/A")
                break

        state = team_state.get((fid, tid))
        history = team_gps_history.get((fid, tid), [])

        # v10.19: Skip GPS calculation when API returned no real stat data.
        # GPS from all-zeros is meaningless and could produce false signals.
        if _data_incomplete:
            log.info(f"  DATA INCOMPLETE: {tname} F{fid} {minute}' — shots=0, SOT=0, skipping GPS")
            # Still update state to avoid crashes on next poll
            team_state[(fid, tid)] = {
                "last_sot": 0, "last_minute": minute, "last_xg": None,
                "last_shots_off_target": 0, "last_total_shots": 0,
                "last_shots_inside_box": 0, "last_corners": 0,
                "last_possession": 0,
            }
            continue

        # === v10.26: Calculate Goal Pressure Score (poss removed, xG boosted) ===
        gps, gps_desc, components = calculate_goal_pressure_score(
            sot=sot, total_shots=total_shots,
            shots_inside_box=shots_inside_box,
            shots_off_target=shots_off_target,
            xg_value=xg_value, corners=corners,
            minute=minute, prev_state=state,
            gps_history=history,
            possession=possession,
        )

        # Count accelerating indicators (for polling + logging)
        accel_count = get_accel_count_from_state(
            sot, total_shots, shots_off_target, xg_value, state, minute
        )

        # === v10: Log GPS on every poll ===
        if gps >= GPS_BUILDING:
            stage_label = "BUILDING" if gps < GPS_EARLY_WARNING else "EARLY" if gps < GPS_CRITICAL else "CRITICAL"
            log.info(
                f"  GPS {stage_label}: {tname} | {gps_desc}"
            )
        else:
            # v10.13.1: Always log per-fixture SOT+GPS for visibility
            log.info(
                f"  F{fid} {minute}' {tname}: SOT={sot} Shots={total_shots} GPS={gps:.0f}"
            )

        # === v10.1: Record poll data for backtesting (includes possession) ===
        is_home_team = (tid == home_tid)
        record_pressure_poll(
            fid=fid, tid=tid, tname=tname, league=league,
            minute=minute, sot=sot, total_shots=total_shots,
            shots_inside_box=shots_inside_box,
            shots_off_target=shots_off_target,
            xg_value=xg_value, corners=corners,
            gps=gps, components=components,
            accel_count=accel_count, is_home=is_home_team,
            score_home=sh, score_away=sa,
            possession=possession,
            stats_sot_raw=stats_sot, events_sot=events_sot,  # v10.31
        )

        # === v10.1: Update GPS history (includes timestamp + xg for window calc) ===
        new_history_entry = {
            "ts": time.time(),
            "gps": round(gps, 1),
            "sot": sot, "total_shots": total_shots,
            "shots_off_target": shots_off_target,  # v10.10
            "accel_count": accel_count, "minute": minute,
            "xg": xg_value,
        }
        if (fid, tid) not in team_gps_history:
            team_gps_history[(fid, tid)] = []
        team_gps_history[(fid, tid)].append(new_history_entry)
        if len(team_gps_history[(fid, tid)]) > GPS_HISTORY_MAX:
            team_gps_history[(fid, tid)] = team_gps_history[(fid, tid)][-GPS_HISTORY_MAX:]

        # === v10.1: Classify signal with quality gates ===
        ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
        sustained = components.get("sustained", 0)
        tier, trend, sot_rate = classify_signal(
            sot, state, minute, gps=gps, accel_count=accel_count,
            inside_box_ratio=ib_ratio, sustained_count=sustained,
        )

        # v10.10: Log WHY GPS was high but signal was blocked (for threshold tuning)
        # Fixed: was producing empty reason strings when no gate condition matched
        if not tier and gps >= GPS_EARLY_WARNING:
            gate_reason = []
            if ib_ratio < 0.50:
                gate_reason.append(f"IB={ib_ratio:.0%}<50%")
            elif ib_ratio < 0.30:
                gate_reason.append(f"IB={ib_ratio:.0%}<30%")
            if gps < 60:
                gate_reason.append(f"GPS={gps:.0f}<60")
            if sustained < 1 and gps < GPS_CRITICAL:
                gate_reason.append("not sustained")
            # v10.10: Always provide a reason when GPS is high enough to be noticed
            if not gate_reason:
                gate_reason.append(f"GPS {gps:.0f} below tier threshold (need SOT incr or higher GPS)")
            log.info(
                f"  GATE SKIP: {tname} GPS={gps:.0f} SOT={sot} — {', '.join(gate_reason)}"
            )

        # Store state AFTER classification (for next comparison)
        team_state[(fid, tid)] = {
            "last_sot": sot,
            "last_minute": minute,
            "last_xg": xg_str if xg_str != "N/A" else None,
            "last_shots_off_target": shots_off_target,  # v10.10
            "last_total_shots": total_shots,
            "last_shots_inside_box": shots_inside_box,
            "last_corners": corners,
            "last_possession": possession,
        }

        # v10.15: Pre-window gate — build baseline but don't signal yet
        # v10.13.1: Data corrupt gate — block signals from bad API data
        if not tier or _data_corrupt:
            if _data_corrupt and tier:
                log.warning(
                    f"  BLOCKED {tier}: {tname} F{fid} — data corrupt flag set, signal suppressed"
                )
            continue

        # v10.24: LOSING TEAM FILTER
        # Data: losing teams hit 28.6% full vs 76.9% when winning/drawing.
        # Losing teams generate pressure stats out of desperation, not dominance.
        # Block EARLY WARNING entirely for losing teams.
        # Allow CRITICAL but tag with warning so user can decide.
        team_goals = (sh if is_home_team else sa) or 0
        opp_goals = (sa if is_home_team else sh) or 0
        is_losing = team_goals < opp_goals
        losing_tag = ""
        if is_losing:
            if tier == "EARLY WARNING":
                log.info(
                    f"  LOSING BLOCK: {tname} {tier} at {minute}' — "
                    f"losing {team_goals}-{opp_goals}, EW blocked for losing teams"
                )
                continue
            else:
                # CRITICAL allowed but tagged
                losing_tag = "\n\u26a0\ufe0f LOSING {}-{} — pressure may be desperation, not dominance".format(
                    team_goals, opp_goals
                )

        # v10.13.2: Extended window — allow early signals (16'-20') if SOT-accelerating
        # v10.19: BUT require CRITICAL GPS (≥75) for EARLY WARNING in early window.
        # A GPS 60 at 18' is noise — same principle as the late-window gate.
        if minute < MINUTE_MIN:
            fid_best_sot = get_fixture_best_sot(fid)
            fid_accel = fid in accelerating_fixtures or fid in pressure_accelerating
            is_extended_early = minute >= EXTENDED_MIN and (fid_accel or fid_best_sot >= 2)

            if not is_extended_early:
                log.info(
                    f"  PRE-WINDOW: {tname} {tier} at {minute}' (need {MINUTE_MIN}'+) — "
                    f"building baseline, SOT={sot} GPS={gps:.0f}"
                )
                continue

            # Early-window GPS gate: require CRITICAL for EARLY WARNING signals
            if tier == "EARLY WARNING" and gps < GPS_CRITICAL:
                log.info(
                    f"  EARLY GATE: {tname} GPS={gps:.0f} at {minute}' — "
                    f"early window requires GPS ≥ {GPS_CRITICAL} (got {gps:.0f}), skipping"
                )
                continue

            log.info(
                f"  EXTENDED EARLY: {tname} {tier} at {minute}' (accel/sot>={2}) — "
                f"SOT={sot} GPS={gps:.0f}, allowing early signal"
            )
            # Don't continue — fall through to signal sending

        # v10.19: Late-window GPS gate — after 60', require CRITICAL GPS (≥75)
        # or SOT≥3. A GPS 63 signal at 67' is fundamentally different
        # from GPS 80 at 67' — the former is noise, the latter is real pressure.
        if minute > 60 and tier == "EARLY WARNING" and gps < GPS_CRITICAL:
            log.info(
                f"  LATE GATE: {tname} GPS={gps:.0f} at {minute}' — "
                f"late window requires GPS ≥ {GPS_CRITICAL} (got {gps:.0f}), skipping"
            )
            continue

        # --- v9.5.4: Signal limit rules ---
        team_sig = signaled_teams.get((fid, tid))
        sig_count = team_sig["count"] if team_sig else 0

        # v10.24: STALE STATS SUPPRESSION — don't re-signal if nothing changed.
        # Rizespor fired 3 identical signals (42', 45', 45') with same
        # SOT=3, IB=0.429, GPS=57.3 because stats didn't change between polls.
        if sig_count >= 1 and team_sig:
            last_sot_sig = team_sig.get("sot_at_last_signal", -1)
            last_time_sig = team_sig.get("last_signal_time", 0)
            seconds_since_last = time.time() - last_time_sig
            if (sot == last_sot_sig
                    and round(ib_ratio, 1) == round(
                        team_sig.get("last_ib_ratio", -1), 1)
                    and round(gps) == round(
                        team_sig.get("last_gps", -1))
                    and seconds_since_last < 300):
                log.info(
                    f"  STALE SUPPRESS: {tname} {tier} at {minute}' — "
                    f"stats unchanged from last signal "
                    f"(SOT={sot} IB={ib_ratio:.1f} GPS={gps:.0f}, "
                    f"{int(seconds_since_last)}s ago)"
                )
                continue

        # --- v9.5.8: First-signal-only on busy days ---
        # v10.19.3: Exception override — if pressure is EXPLODING after the 1st signal,
        # allow a 2nd signal even on busy days. Keeps polling for data collection.
        # Exceptional = SOT burst (2+ in one poll) OR GPS>=85 with pressure acceleration.
        if is_first_signal_only_mode() and sig_count >= 1:
            is_exceptional = (
                (fid in sot_burst_fixtures)
                or (gps >= 85 and (fid in pressure_accelerating or fid in accelerating_fixtures))
            )
            if not is_exceptional:
                log.info(
                    f"  SKIP {tier}: {tname} - "
                    f"{sot} SOT GPS={gps:.0f} (first-signal-only mode, {total_matches_today} games) "
                    f"(fixture {fid})"
                )
                continue
            log.info(
                f"  EXCEPTIONAL OVERRIDE: {tname} - "
                f"{sot} SOT GPS={gps:.0f} {'SOT-BURST' if fid in sot_burst_fixtures else 'GPS+ACCEL'} "
                f"(allowing 2nd signal in first-signal-only mode, fixture {fid})"
            )

        if sig_count >= 2:
            last_signal_sot = team_sig.get("sot_at_last_signal", 0)
            sot_jump = sot - last_signal_sot
            seconds_since = time.time() - team_sig.get("last_signal_time", time.time())
            # v10.19.3: Speed bonus — fast SOT jumps earn a GPS discount.
            # A +1 SOT jump in <3 min is strong pressure acceleration even if
            # GPS hasn't crossed 75 yet. Slower +1 jumps still need GPS>=75.
            fast_jump = sot_jump == 1 and seconds_since < 180
            gps_floor = GPS_CRITICAL - 15 if fast_jump else GPS_CRITICAL
            if sot_jump < 1 or (sot_jump < 2 and gps < gps_floor):
                if sot_jump < 1:
                    reason = f"no SOT increase (still {sot})"
                elif fast_jump:
                    reason = f"+{sot_jump} in {int(seconds_since)}s but GPS {gps:.0f} < {int(gps_floor)}"
                else:
                    reason = f"+{sot_jump} too slow, GPS {gps:.0f} < {int(gps_floor)} (need +2 or GPS>={int(gps_floor)})"
                log.info(
                    f"  BLOCKED {tier}: {tname} - "
                    f"{sot} SOT ({reason}) "
                    f"(sig #{sig_count + 1}, fixture {fid})"
                )
                continue

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
        is_new_team = sig_count == 0
        is_home_sg = (tid == home_tid)
        goals_now = (
            fixture["goals"]["home"] if is_home_sg
            else fixture["goals"]["away"]
        ) or 0
        if is_new_team:
            signaled_teams[(fid, tid)] = {
                "count": 1, "goals_at_last_signal": goals_now,
                "sot_at_last_signal": sot, "last_signal_time": time.time(),
                "last_ib_ratio": ib_ratio, "last_gps": gps,
            }
        else:
            signaled_teams[(fid, tid)]["count"] = sig_count + 1
            signaled_teams[(fid, tid)]["goals_at_last_signal"] = goals_now
            signaled_teams[(fid, tid)]["sot_at_last_signal"] = sot
            signaled_teams[(fid, tid)]["last_signal_time"] = time.time()
            signaled_teams[(fid, tid)]["last_ib_ratio"] = ib_ratio
            signaled_teams[(fid, tid)]["last_gps"] = gps
        signaled_fixtures.add(fid)

        # Build the signal message (v10: includes GPS)
        sig_num = sig_count + 1
        sig_label = f"{sig_num}{'st' if sig_num == 1 else 'nd' if sig_num == 2 else 'rd' if sig_num == 3 else 'th'}"
        # v10.10: off-target from API directly (more accurate than total - SOT)
        ib_pct = f"{shots_inside_box / total_shots * 100:.0f}%" if total_shots > 0 else "N/A"

        # Determine trigger type for message
        if tier == "EARLY WARNING":
            trigger = f"GPS-TRIGGERED (accel{' sustained' if sustained >= 1 else ''}, IB={ib_ratio:.0%})"
        else:
            trigger = "SOT>=3 CONFIRMED"

        # v10.21: Top SOT player removed — /fixtures/statistics returns team stats
        # only (no player array). Was silently returning empty + costing 1 credit/signal.
        top_sot_str = ""

        # v10.27: Window tag for signal classification analysis
        # CORE = primary 21-60' window, EARLY_OVERRIDE/LATE_OVERRIDE = edge zones
        if minute < MINUTE_MIN:
            window_tag = "EARLY_OVERRIDE"
            window_label = "⚡ EARLY OVERRIDE"
        elif minute > MINUTE_MAX - 1:  # MINUTE_MAX=61, so >60'
            window_tag = "LATE_OVERRIDE"
            window_label = "⏰ LATE OVERRIDE"
        else:
            window_tag = "CORE"
            window_label = ""

        msg = (
            f"{tier_emoji(tier)} {tier} GOAL PRESSURE ({sig_label})"
            f"{f' {window_label}' if window_label else ''}\n\n"
            f"{home['name']}  {sh} - {sa}  {away['name']}\n"
            f"{league} | {minute}'\n\n"
            f"{tname}\n"
            f"SOT: {sot} | Total Shots: {total_shots} (In-box: {ib_pct})\n"
            f"xG: {xg_str} | Off-target: {shots_off_target} | Corners: {corners}\n"
            f"Opponent SOT: {opponent_sot} | Opponent xG: {opponent_xg}\n\n"
            f"GPS: {gps:.0f}/100 | Trigger: {trigger}\n"
            f"Red Cards: {red_card_str}"
            f"{losing_tag}"
        )
        if top_sot_str:
            msg += f"\n\n\U0001f3af Top SOT: {top_sot_str}"
        if trend:
            msg += f"\nTrend: {trend}"

        # v10.28: Show recency info in signal message
        _recency_preview = _build_recency_fields(
            fid, tid, minute, sot, total_shots,
            shots_inside_box, shots_off_target, xg_value, corners,
            gps, accel_count,
        )
        _rr = _recency_preview.get("recency_ratio")
        _sot_d5 = _recency_preview.get("sot_delta_5m")
        _sot_d10 = _recency_preview.get("sot_delta_10m")
        _gps_chg = _recency_preview.get("gps_change")
        if _rr is not None or _sot_d5 or _gps_chg:
            msg += "\n\n"
            if _rr is not None:
                msg += f"\U0001f504 Recency: {_rr:.0%}"
            if _sot_d5 is not None:
                msg += f" | SOT +{_sot_d5} (5m)"
            if _sot_d10 is not None:
                msg += f" +{_sot_d10} (10m)"
            if _gps_chg is not None:
                msg += f" | GPS {_gps_chg:+.0f}"

        if send_telegram(client, msg):
            log.info(
                f"  SIGNAL {tier}: {tname} - "
                f"{sot} SOT, GPS={gps:.0f}, xG={xg_str} (fixture {fid}, "
                f"{sig_label} signal, {window_tag}, {trigger})"
            )

        signals_sent.append({
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "fixture": fid, "team": tname, "league": league,
            "minute": minute, "sot": sot, "xg": xg_str,
            "gps": round(gps, 1),
            "red_cards": red_card_str, "tier": tier,
            "window_tag": window_tag,  # v10.27
            "trend": trend, "is_new": is_new_team,
        })

        # v10.1: Enriched outcome record with all raw indicators
        opp_goals = (sa if is_home_sg else sh)
        signal_outcomes.append({
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": league,
            "signal_time": time.time(),
            "signal_clock": time.strftime("%Y-%m-%d %H:%M"),
            "game_minute": minute,
            "sot": sot,
            "stats_sot_raw": stats_sot,  # v10.31: audit trail
            "events_sot": events_sot,      # v10.31: audit trail
            "total_shots": total_shots,
            "shots_inside_box": shots_inside_box,
            "ib_ratio": round(ib_ratio, 3),
            "shots_off_target": shots_off_target,  # v10.10
            "xg": round(xg_value, 3) if xg_value is not None else None,
            "corners": corners,
            "possession": possession,
            "gps": round(gps, 1),
            "gps_sot": components.get("sot", 0),
            "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0),
            "gps_xg": components.get("xg", 0),
            "gps_poss": components.get("possession", 0),  # v10.10: was gps_da
            "gps_accel": components.get("acceleration", 0),
            "sustained": sustained,
            "accel_count": accel_count,
            "tier": tier,
            "window_tag": window_tag,  # v10.27: CORE / EARLY_OVERRIDE / LATE_OVERRIDE
            "goals_at_signal": goals_now,
            "opponent_goals_at_signal": opp_goals,
            "is_home": is_home_sg,
            "minutes_remaining": 90 - minute,  # v10.28: natural time ceiling for late signals
            "outcome_5min": None,   # v10: expanded windows
            "outcome_10min": None,  # v10: expanded windows
            "outcome_15min": None,
            "outcome_full": None,
            "goal_minute_5": None,
            "goal_minute_10": None,
            "goal_minute_15": None,
            "goal_minute_full": None,
            "sig_num": sig_num,
            "gps_triggered": tier == "EARLY WARNING",
            "resolved": False,
        })
        # v10.28: Merge enriched recency fields into signal outcome
        recency = _build_recency_fields(
            fid, tid, minute, sot, total_shots,
            shots_inside_box, shots_off_target, xg_value, corners,
            gps, accel_count,
        )
        signal_outcomes[-1].update(recency)
        # v10.11: Save pending outcome immediately (survives restarts)
        save_outcome(signal_outcomes[-1])

        # v10.19.3: In first-signal-only mode, keep polling for data collection
        # and exceptional 2nd signal detection. Don't remove from monitoring.
        if is_first_signal_only_mode():
            log.info(
                f"  FIRST-SIGNAL-DONE: fixture {fid} stays monitored for data + exceptional override "
                f"({len(fast_monitored)} still monitored)"
            )

    # --- v9.7: Update SOT acceleration flag ---
    if _sot_increased:
        accelerating_fixtures.add(fid)
        log.debug(f"  SOT-Accelerating: fixture {fid} (SOT {prev_best_sot}->{best_current_sot})")
    else:
        accelerating_fixtures.discard(fid)

    # --- v10.17: Update SOT burst flag (2+ SOT jump in one poll) ---
    _sot_burst = (best_current_sot - prev_best_sot) >= 2
    if _sot_burst:
        sot_burst_fixtures.add(fid)
        log.info(
            f"  SOT-BURST: fixture {fid} (SOT {prev_best_sot}->{best_current_sot}, "
            f"+{best_current_sot - prev_best_sot} in one poll)"
        )
    else:
        sot_burst_fixtures.discard(fid)

    # --- v10: Update pressure acceleration flag ---
    # Check if ANY team in this fixture has rising GPS + 2+ accelerating indicators
    if is_fixture_pressure_accelerating(fid):
        pressure_accelerating.add(fid)
        log.info(f"  PRESSURE-ACCEL: fixture {fid} (GPS rising + multi-indicator acceleration)")
    else:
        pressure_accelerating.discard(fid)

    # --- v9.7: Dead fixture detection (both teams SOT=0) ---
    if best_current_sot == 0 and prev_best_sot == 0:
        hg = fixture["goals"]["home"] or 0
        ag = fixture["goals"]["away"] or 0
        dead_fixtures[fid] = (hg, ag)
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        accelerating_fixtures.discard(fid)
        pressure_accelerating.discard(fid)
        sot_burst_fixtures.discard(fid)
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
    global active_friendly_fixtures, todays_tracked_fixtures
    
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
        
        # DEBUG: Log unseen league IDs (collapsed to 1 line to avoid Railway log rate limit)
        unseen_leagues = {}
        for f in all_fixtures:
            lid = f["league"]["id"]
            lname = f["league"].get("name", "?")
            if lid not in LEAGUE_IDS and lid not in unseen_leagues:
                unseen_leagues[lid] = lname
        if unseen_leagues:
            # Show top 5 by fixture count, rest as summary
            league_counts = []
            for lid, lname in sorted(unseen_leagues.items()):
                count = sum(1 for f in all_fixtures if f["league"]["id"] == lid)
                league_counts.append((count, lid, lname))
            league_counts.sort(reverse=True)
            top = league_counts[:5]
            top_str = ", ".join(f"{lid}-{lname}({c})" for c, lid, lname in top)
            log.info(f"  DEBUG: {len(unseen_leagues)} unseen leagues. Top: {top_str}")

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

        # v10.11: Store today's tracked fixtures for daily summary
        global todays_tracked_fixtures
        todays_tracked_fixtures = []
        for f in all_fixtures:
            lid = f["league"]["id"]
            fid = f["fixture"]["id"]
            if lid in LEAGUE_IDS or fid in active_friendly_fixtures:
                date_str = f["fixture"]["date"]
                try:
                    kickoff_utc = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    )
                    kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                    if kickoff_local.strftime("%Y-%m-%d") == today_str:
                        todays_tracked_fixtures.append(f)
                except Exception:
                    pass

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

        # Deduplicate waves and compute 1' entry times (kickoff + pre-window offset)
        scheduled_window_entries = sorted(set(
            (wave * 60) + ((MINUTE_MIN - PRE_WINDOW_MINUTES) * 60)  # kickoff_ts + 1' (pre-window)
            for wave in set(kickoff_wave_minutes)
        ))
        if scheduled_window_entries:
            entries_bg = [
                datetime.fromtimestamp(t, BULGARIA_TZ).strftime("%H:%M")
                for t in scheduled_window_entries
            ]
            log.info(
                f"  {MINUTE_MIN - PRE_WINDOW_MINUTES}' entries (schedule, signal from {MINUTE_MIN}'): {entries_bg}"
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
        
        # Buffer: 30 min before earliest kickoff (covers pre-window from 1'),
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
        scheduled_window_entries = []  # v10.19.1: clear stale entries on fetch failure
        return True


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v10.31 — Fix 'state' UnboundLocalError + team name mismatch fallback")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (round-robin for rate-limit resilience, NOT quota expansion)")
    log.info("")
    log.info("v10.1 CHANGES (robustness + quality gates):")
    log.info("  FIELD NORMALIZATION: STAT_ALIASES with fallbacks prevent silent zeros")
    log.info("    e.g. 'Shots insidebox' -> 'Shots Inside Box' -> 'Shots inside Box'")
    log.info("  WINDOWED RATES: acceleration uses per-minute rates over 5-game-min window")
    log.info("    (interval-independent: 60s poll = 240s poll if per-min rates match)")
    log.info("  SUSTAINED PRESSURE GATE: EARLY WARNING requires 2+ consecutive accel polls")
    log.info("    + inside-box ratio >= 30% (3 long-range shots != 3 dangerous shots)")
    log.info("  POSSESSION TRACKING: recorded in poll data, 0-3 GPS pts (low weight)")
    log.info("  GPS HISTORY: expanded to 5 polls (was 3) for windowed calculations")
    log.info("  GATE SKIP LOGS: blocked signals log WHY (IB% too low, not sustained)")
    log.info("")
    log.info("v10.14 CHANGES (auto goal-event resolution):")
    log.info("  When fixture finishes (FT), bot fetches /fixtures/events to get")
    log.info("  EXACT goal minutes. Resolves 5/10/15 min windows precisely instead")
    log.info("  of comparing final score (which can't determine WHEN goal scored).")
    log.info("  Own goals excluded. 1 credit per finished fixture with signals.")
    log.info("  Caches events per fixture to avoid duplicate calls.")
    log.info("")
    log.info("v10.13 CHANGES:")
    log.info("  /stats, /stats3d, /stats7d commands, EOD + morning recap from file")
    log.info("")
    log.info("v10.11 CHANGES (win rate tracking):")
    log.info("  STALE OUTCOME RESOLUTION: on startup, fetches finished fixtures by ID")
    log.info("    to resolve signals that were pending when bot slept")
    log.info("  /stats TELEGRAM COMMAND: type /stats in your bot chat anytime")
    log.info("    to get full win rate breakdown sent to Telegram")
    log.info("  HISTORICAL DATA: /stats reads ALL outcomes from JSONL file")
    log.info("")
    log.info("v10.17 CHANGES (SOT burst + goalless pressure):")
    log.info("  SOT BURST: 2+ SOT jump in one poll → +3/6 GPS pts + 3 combo if GPS rising")
    log.info("  GOALLESS PRESSURE: 3+ polls GPS>=50 w/o goal → +3/5 GPS pts (rising odds)")
    log.info("  PRIORITY: SOT burst → 97, sustained goalless → 93, both get 60s polling")
    log.info("  ACCEL CAP raised 15→20 to accommodate burst + goalless bonuses")
    log.info("")
    log.info("v10.19 CHANGES (window gates + data protection + stats):")
    log.info("  EXTENDED_MIN 10→9, EXTENDED_MAX 75→85 (wider both edges)")
    log.info("  PRE_WINDOW 12→20 min (polling from 1' to build baseline)")
    log.info("  EARLY GATE: <21' requires GPS≥75 (CRITICAL) for EARLY WARNING signals")
    log.info("  LATE GATE: >60' requires GPS≥75 (CRITICAL) for EARLY WARNING signals")
    log.info("  N/A PROTECTION: literal 'N/A' stat values → '0' + DATA INCOMPLETE skip")
    log.info("  GPS×MINUTE CROSS-TAB in /stats and /recap output")
    log.info("  LOG SUMMARY: shows both 1st-signal-only AND all-signals blocks")
    log.info("  HTTPX LOG SUPPRESSION: no more getUpdates spam in dead hours")
    log.info("")
    log.info("v10.18 CHANGES (daily summary reform + top SOT player):")
    log.info("  REMOVED auto-send daily summary at startup (was spamming match list)")
    log.info("  REMOVED team form + top scorers from /matches (was burning 40-80 credits/day)")
    log.info("  /today and /matches = simple match list, 0 extra credits, only if <10 games")
    log.info("  TOP SOT PLAYER: signal now shows who has the most SOT (1 credit/signal)")
    log.info("")
    log.info("v10 GPS (preserved):")
    log.info("  v10.26 Components: SOT(28) + InBox(22) + ShotVol(14) + xG(18) + Corners(4) + ACCEL(22)")
    log.info("  v10.26 REMOVED: possession (0 in 2,168 polls), dangerous_attacks (not in API)")
    log.info("  v10.26 REDISTRIBUTED: 5 poss pts → xG +3 (max 18), accel cap +2 (max 22)")
    log.info("  v10.26 data_quality: real computation (non-null field fraction) in pressure_polls.jsonl")
    log.info(f"  Signal window: {MINUTE_MIN}-{MINUTE_MAX}' | Polling: {MINUTE_MIN - PRE_WINDOW_MINUTES}-{EXTENDED_MAX}' (full range, gates protect quality) | Pre-window: {MINUTE_MIN - PRE_WINDOW_MINUTES}-{MINUTE_MIN - 1}' (baseline only)")
    log.info(f"  Thresholds: BUILDING>={GPS_BUILDING} EARLY_WARNING>={GPS_EARLY_WARNING} CRITICAL>={GPS_CRITICAL}")
    log.info("")
    log.info("PRESERVED FROM v9.x:")
    log.info("  SOT>=3 ALWAYS triggers CRITICAL (safety net — proven v9.x trigger)")
    log.info("  Credit optimization: batched requests, night gate, tomorrow cache")
    log.info("  Dead fixture removal, first-signal-only on busy days")
    log.info("  Unified /fixtures?ids= call (1 credit for all fixtures if stats embedded)")
    log.info("")
    log.info("v10.26 CHANGES (GPS formula fix — dead components removed):")
    log.info("  POSSESSION component REMOVED (5 pts) — always 0 across 2,168 polls / 12 leagues")
    log.info("  DANGEROUS_ATTACKS already removed in v10.10 (not in API-Football v3)")
    log.info("  REDISTRIBUTED 5 pts: xG max 15→18 (+3), acceleration cap 20→22 (+2)")
    log.info("  DATA_QUALITY now computed from non-null field fraction (was hardcoded 0.5)")
    log.info("  Signal messages no longer show 'Possession: 0%' (was always dead)")
    log.info("")
    log.info("v10.25 CHANGES (interval-gated stats polling):")
    log.info("  Stats API call now GATED on per-fixture intervals.")
    log.info("  Previously: stats called every cycle (~60s) regardless of intervals.")
    log.info("  The 60s sleep cap made all tiers (60s/90s/180s) poll at 60s.")
    log.info("  Now: 1 credit/batch only when at least 1 fixture is due.")
    log.info("  Hot fixtures (SOT>=2, accelerating): 60s polling = UNCHANGED speed.")
    log.info("  Cold fixtures (SOT=0/1): skip cycles until due = REAL credit savings.")
    log.info("  Signal delay impact: +0s hot, +30s max SOT=2, negligible else.")
    log.info("")
    log.info("v10.24 CHANGES (signal quality filters):")
    log.info("  4 quality gates in classify_signal() and process_fixture_stats():")
    log.info("  1. IB<50% blocks ALL signals (shots not dangerous enough)")
    log.info("  2. GPS<60 blocks EARLY WARNING (SOT>=3 CRITICAL exempt)")
    log.info("  3. LOSING teams: EW blocked, CRITICAL tagged with warning")
    log.info("  4. Stale suppression: blocks re-signal if SOT/IB/GPS unchanged in 5min")
    log.info("")
    log.info("v10.23 CHANGES (unified batch polling):")
    log.info("  ALL monitored fixtures polled together in ONE batch every cycle.")
    log.info("  Previously, fast-polling fixtures split into separate 1-fixture API calls,")
    log.info("  burning 1 extra credit per cycle (e.g. 'Stats -> 1 fixture(s): [1584380]').")
    log.info("  Now: always 1 credit/cycle regardless of how many fixtures have different intervals.")
    log.info("  Fastest fixture sets the cycle pace; slower fixtures polled 'for free'.")
    log.info("")
    log.info("v10.22 CHANGES (edge-zone fast-polling gate):")
    log.info("  Before 21' or after 60': 60s/90s fast polling ONLY if SOT>=2 OR GPS>=50")
    log.info("  These zones require GPS>=75 to signal — fast polling SOT=0/1 GPS=30 is waste")
    log.info("  Saves credits on low-pressure edge-zone games (e.g. Çorum FK 51'-58' SOT=0/1)")
    log.info("")
    log.info("POLLING TIERS (per fixture, before 2x both-signaled multiplier):")
    log.info("  v10.17:  SOT-burst (2+ SOT jump in one poll):         60s")
    log.info("  v10.1:  Pressure-accelerating (rate-based 2+ deltas):  60s")
    log.info("  v9.7:    SOT-accelerating (SOT increased last poll):   60s")
    log.info("  SOT >= 3:                                          90s")
    log.info("  SOT == 2 + fast window:                           120s")
    log.info("  SOT == 2:                                         180s")
    log.info("  SOT == 1:                                         base interval")
    log.info("  *** v10.22: Above 60s/90s tiers BLOCKED in edge zones (<21'/'>60')")
    log.info("      unless SOT>=2 or GPS>=50 (high-pressure threshold)")
    log.info(f"  Active: DYNAMIC from daily schedule (fallback {ACTIVE_HOUR_START_FALLBACK}:00-{ACTIVE_HOUR_END_FALLBACK}:00)")
    log.info(f"  Team cache: {len(known_league_team_ids)} IDs ({len(PRESEEDED_TEAM_IDS)} pre-seeded)")
    log.info("=" * 60)

    # v10.19.3: Load ALL outcomes (resolved + pending) from JSONL
    global signal_outcomes
    all_loaded = load_all_outcomes()
    if all_loaded:
        signal_outcomes = all_loaded
        resolved_count = sum(1 for e in all_loaded if e.get("resolved"))
        pending_count = sum(1 for e in all_loaded if not e.get("resolved"))
        log.info(f"Loaded {len(all_loaded)} outcome(s) from {OUTCOMES_FILE} ({resolved_count} resolved, {pending_count} pending)")

    with httpx.Client(timeout=30.0) as client:
        # v10.11: Resolve any stale outcomes from previous sessions
        pending_count = sum(1 for e in signal_outcomes if not e.get("resolved"))
        if pending_count:
            log.info("v10.11: Resolving stale outcomes from finished fixtures...")
            resolve_stale_outcomes(client)
            # Log summary after resolution
            resolved_count = sum(1 for e in signal_outcomes if e.get("resolved"))
            still_pending = sum(1 for e in signal_outcomes if not e.get("resolved"))
            log.info(f"v10.11: After resolution: {resolved_count} resolved, {still_pending} pending")
            if resolved_count > 0:
                log_outcome_summary()

        # v10.18: Morning recap removed from auto-send. Use /recap or /stats in Telegram.
        log.info("v10.18: Morning recap skipped (command-only now). Use /recap for yesterday's stats.")

        while True:
            now = time.time()

            # --- Fetch daily schedule (1 call/day, re-fetches on date change) ---
            has_matches = fetch_daily_active_hours(client)
            
            # v10.18: Daily summary removed from auto-send. Use /today in Telegram.
            # Auto-send was burning credits on team form; now command-only, <10 games.
            
            if not has_matches:
                # v9.7.1: Smart sleep — calculate how long until next meaningful wake
                now_bg = datetime.now(BULGARIA_TZ)
                h = now_bg.hour
                # If night: sleep until NIGHT_HOUR_END
                if NIGHT_HOUR_START <= h < NIGHT_HOUR_END:
                    wake_at = now_bg.replace(hour=NIGHT_HOUR_END, minute=0, second=0, microsecond=0)
                    sleep_s = int((wake_at - now_bg).total_seconds())
                    sleep_s = max(sleep_s, 60)  # min 60s to avoid infinite loop
                else:
                    sleep_s = 1800  # 30 min during daytime no-matches
                log.info(
                    f"No tracked matches today, sleeping until {sleep_s // 60}m (waking every 2m for commands)..."
                )
                # v10.18: EOD summary removed from auto-send. Use /recap or /stats in Telegram.
                # v10.13: Short-interval sleep loop — checks Telegram every 2 min
                # Telegram API is free (0 football credits), so this costs nothing
                while sleep_s > 0:
                    check_telegram_commands(client)
                    chunk = min(sleep_s, 120)  # wake every 2 min
                    time.sleep(chunk)
                    sleep_s -= chunk
                continue
            
            # --- Dead hours (zero API cost) ---
            # Uses dynamic window from today's schedule
            local_hour = datetime.now(BULGARIA_TZ).hour
            if not (dynamic_active_start <= local_hour < dynamic_active_end):
                # v10.18: EOD summary removed from auto-send. Use /recap or /stats in Telegram.

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
                    f"sleeping {sleep_s // 60}m {sleep_s % 60}s (waking every 2m for commands)..."
                )
                # v10.13: Short-interval sleep loop — checks Telegram every 2 min
                # Telegram API is free (0 football credits), so this costs nothing
                while sleep_s > 0:
                    check_telegram_commands(client)
                    chunk = min(sleep_s, 120)  # wake every 2 min
                    time.sleep(chunk)
                    sleep_s -= chunk
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
            # STATS CHECK — batched, interval-gated (v10.25)
            # ------------------------------------------------------------
            # v10.25: Gate stats API call on actual fixture intervals.
            # Previously, stats were called every cycle regardless of
            # per-fixture intervals (the 60s sleep cap forced ~60s cycles).
            # Now, 1 credit/batch is only spent when at least one fixture
            # is due. Hot fixtures (60s interval) still poll every 60s.
            # Cold fixtures (SOT=0/1, 135-180s) skip cycles = real savings.
            # Signal speed: UNCHANGED for hot fixtures (60s),
            # +30s max for SOT=2 (90s interval), negligible for others.

            any_stats_due = False
            stats_int = get_stats_interval(budget)
            if fast_monitored and budget != "STOP":
                for fid in fast_monitored:
                    fixture_interval = get_sot_based_interval(fid, stats_int)
                    elapsed = now - last_stats_check.get(fid, 0)
                    if elapsed >= fixture_interval:
                        any_stats_due = True
                        break

            if any_stats_due and fast_monitored and budget != "STOP":
                ordered = sorted(
                    fast_monitored,
                    key=get_fixture_sot_priority,
                    reverse=True,
                )

                # v10.23: ALL monitored fixtures polled together (1 credit/batch).
                # v10.25: But only when at least one is due (credit gate).
                if (quota_remaining is not None
                        and quota_remaining <= 2):
                    log.warning(
                        "Quota nearly exhausted; skipping stats."
                    )
                else:
                    # Count how many are actually due vs riding free
                    due_count = sum(
                        1 for fid in fast_monitored
                        if (now - last_stats_check.get(fid, 0))
                           >= get_sot_based_interval(fid, stats_int)
                    )
                    log.info(
                        f"  Stats -> {len(ordered)} fixture(s) (batched, "
                        f"{due_count} due): {ordered}"
                    )
                    check_monitored_stats(client, ordered)
            elif fast_monitored and not any_stats_due and budget != "STOP":
                # v10.25: Log when stats are skipped (credit saving)
                soonest_due = min(
                    max(0, get_sot_based_interval(fid, stats_int)
                         - (now - last_stats_check.get(fid, 0)))
                    for fid in fast_monitored
                )
                log.info(
                    f"  Stats skipped (no fixture due, next in {soonest_due}s)"
                )

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
                extra_info += f" | SOT-Accel: {', '.join(accel_parts[:3])}{'...' if len(accel_parts) > 3 else ''}"
            if pressure_accelerating:
                paccel_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in pressure_accelerating]
                extra_info += f" | GPS-Accel: {', '.join(paccel_parts[:3])}{'...' if len(paccel_parts) > 3 else ''}"
            if sot_burst_fixtures:
                burst_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in sot_burst_fixtures]
                extra_info += f" | SOT-Burst: {', '.join(burst_parts[:3])}{'...' if len(burst_parts) > 3 else ''}"
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
            # v9.8: Show outcome tracking in status line (dual: 15-min + full-match)
            resolved_outcomes = [e for e in signal_outcomes if e["resolved"]]
            pending_outcomes = [e for e in signal_outcomes if not e["resolved"]]
            h15 = sum(1 for e in resolved_outcomes if e.get("outcome_15min") == "HIT")
            m15 = sum(1 for e in resolved_outcomes if e.get("outcome_15min") == "MISS")
            hf = sum(1 for e in resolved_outcomes if e.get("outcome_full") == "HIT")
            mf = sum(1 for e in resolved_outcomes if e.get("outcome_full") == "MISS")
            outcome_str = ""
            if resolved_outcomes:
                r15 = h15 / len(resolved_outcomes) * 100
                rf = hf / len(resolved_outcomes) * 100
                outcome_str = f" | 15m:{h15}H/{m15}M({r15:.0f}%) full:{hf}H/{mf}M({rf:.0f}%)"
            if pending_outcomes:
                outcome_str += f" | Pend:{len(pending_outcomes)}"

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
            # v10.15: Also try resolving stale outcomes first — fixtures
            # that went FT disappear from /fixtures?live=all, so pending
            # signals for them never get resolved during normal discovery.
            if (resolved_outcomes or pending_outcomes) and not has_tracked_live and not fast_monitored:
                # Try to resolve any remaining pending (FT fixtures not in live feed)
                if pending_outcomes:
                    resolve_stale_outcomes(client)
                    # Recompute after resolution attempt
                    resolved_outcomes = [e for e in signal_outcomes if e["resolved"]]
                    pending_outcomes = [e for e in signal_outcomes if not e["resolved"]]
                if resolved_outcomes:
                    log_outcome_summary()
                    # v10.18: EOD summary removed from auto-send. Use /recap or /stats.
                    # v10.19.3: Rewrite file before clearing — preserves all resolved data
                    rewrite_outcomes_file()
                    signal_outcomes.clear()

            # v10.11: Check Telegram /stats command
            check_telegram_commands(client)

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
