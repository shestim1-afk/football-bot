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
    # --- Top 5 leagues + 2. Bundesliga ---
    39: "Premier League", 140: "La Liga", 78: "Bundesliga", 79: "2. Bundesliga",
    135: "Serie A", 61: "Ligue 1",
    # --- European 2nd divisions (v10.2) ---
    40: "Championship", 141: "Segunda División", 136: "Serie B", 62: "Ligue 2",
    # --- European top leagues (v10.2 additions) ---
    144: "Belgian Pro League", 340: "Scottish Premiership",
    332: "Allsvenskan (Sweden)",  # summer season, active now
    # --- European cups ---
    2: "Champions League", 3: "Europa League", 848: "Conference League",
    # --- Existing top leagues ---
    94: "Primeira Liga", 88: "Eredivisie", 203: "Super Lig",
    310: "Austrian Bundesliga",  # 169 was Chinese Super League!
    # --- Smaller European leagues ---
    357: "First League (Bulgaria)",
    # 283 REMOVED — API-Football maps it to Liga I (Romania), NOT SuperLiga Serbia
    210: "HNL (Croatia)", 345: "Czech First League",
    119: "Danish Superliga",
    # 137 REMOVED — API-Football maps it to Coppa Italia, NOT Veikkausliiga
    # 283 REMOVED — API-Football maps it to Liga I (Romania), NOT SuperLiga Serbia
    191: "NB I (Hungary)",
}

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}

# Active monitoring window: dynamically computed from daily schedule
# Falls back to 14:00-23:00 if schedule fetch fails
BULGARIA_TZ = ZoneInfo("Europe/Sofia")
ACTIVE_HOUR_START_FALLBACK = 14  # fallback
ACTIVE_HOUR_END_FALLBACK = 23    # fallback

# v10.3.2: 5'-85' window (wider catch — early pressure + late drama)
MINUTE_MIN = 5
MINUTE_MAX = 85

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
# Records every signal sent, tracks FOUR outcomes:
#   outcome_5min:  HIT if team scored within 5 game minutes
#   outcome_10min: HIT if team scored within 10 game minutes
#   outcome_15min: HIT if team scored within 15 game minutes
#   outcome_full:  HIT if team scored at all before match ended
# Persisted to JSONL file so data survives restarts.
signal_outcomes: list[dict] = []
OUTCOME_WINDOW_MINUTES = 15  # game minutes for "imminent" window

# --- v10.4: Data directory for persistent files ---
# On Railway, set DATA_DIR=/data (volume mount). Locally, uses script directory.
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
OUTCOMES_FILE = os.path.join(DATA_DIR, "signal_outcomes.jsonl")
POLL_DATA_FILE = os.path.join(DATA_DIR, "pressure_polls.jsonl")
SIGNAL_STATE_FILE = os.path.join(DATA_DIR, "signal_state.json")

# ============================================================
# v10.5: TEAM FORM & H2H CONTEXT SYSTEM
# ============================================================
# Fetches last 5 fixtures for form + last 5 H2H meetings.
# Cached per-match (fetched once when first signal fires).
# Modifies GPS: FINAL_GPS = raw_gps + form_adjustment + h2h_modifier
# API cost: 2 credits per match (1 form per team, 1 H2H).
# ============================================================

FORM_FILE = os.path.join(DATA_DIR, "form_h2h_cache.json")

# In-memory cache: fixture_id -> {
#   home_team_id: { form_score, form_string, avg_gf, avg_ga, last5 },
#   away_team_id: { form_score, form_string, avg_gf, avg_ga, last5 },
#   h2h: { games, teamA_wins, draws, teamB_wins, teamA_gf, teamB_gf, modifier, avg_total }
# }
form_h2h_cache: dict[int, dict] = {}

# Form fetch cooldown: don't re-fetch for same fixture within 6 hours
FORM_FETCH_COOLDOWN = 6 * 3600
form_fetch_timestamps: dict[int, float] = {}  # fixture_id -> last fetch time


def fetch_team_form_sync(client: httpx.Client, team_id: int) -> dict:
    """Fetch last 5 completed fixtures for a team to calculate form.

    Uses /fixtures?team={id}&last=5 endpoint.
    Returns dict with form_score, form_string, avg_gf, avg_ga, last5 list.
    """
    try:
        data = api_get(client, "/fixtures", {"team": team_id, "last": "5"})
        fixtures = data.get("response", [])

        if not fixtures:
            return {"form_score": 50, "form_string": "N/A", "avg_gf": 0, "avg_ga": 0, "last5": []}

        wins, draws, losses = 0, 0, 0
        total_gf, total_ga = 0, 0
        last5 = []  # list of "W"/"D"/"L"

        for f in fixtures:
            status = f["fixture"]["status"]["short"]
            if status not in ("FT", "AET", "PEN"):
                continue  # skip live/postponed

            is_home = f["teams"]["home"]["id"] == team_id
            gf = (f["goals"]["home"] if is_home else f["goals"]["away"]) or 0
            ga = (f["goals"]["away"] if is_home else f["goals"]["home"]) or 0

            total_gf += gf
            total_ga += ga

            if gf > ga:
                wins += 1
                last5.append("W")
            elif gf == ga:
                draws += 1
                last5.append("D")
            else:
                losses += 1
                last5.append("L")

        played = wins + draws + losses
        avg_gf = total_gf / played if played > 0 else 0
        avg_ga = total_ga / played if played > 0 else 0

        # Form Score (0-100)
        # Based on: wins (up to 60), draws (up to 21), scoring rate (up to 12), defensive bonus (up to 7)
        form_score = (
            (wins * 12) +        # max 60 (5W)
            (draws * 7) +        # max 21 (5D — but can't have 5D if any W/L)
            min(avg_gf * 6, 12) +  # max 12 (2+ goals/game avg)
            min(max(0, 10 - avg_ga * 5), 7)  # max 7 (concede <1/g)
        )
        form_score = min(max(int(form_score), 0), 100)

        form_string = "-".join(last5) if last5 else "N/A"

        log.info(
            f"  FORM: Team {team_id} -> {form_string} "
            f"({avg_gf:.1f} GF, {avg_ga:.1f} GA, score={form_score})"
        )

        return {
            "form_score": form_score,
            "form_string": form_string,
            "avg_gf": round(avg_gf, 2),
            "avg_ga": round(avg_ga, 2),
            "last5": last5,
        }
    except Exception as e:
        log.warning(f"  Form fetch failed for team {team_id}: {e}")
        return {"form_score": 50, "form_string": "N/A", "avg_gf": 0, "avg_ga": 0, "last5": []}


def fetch_h2h(client: httpx.Client, team_a_id: int, team_b_id: int) -> dict:
    """Fetch head-to-head record between two teams (last 5 meetings).

    Uses /fixtures/h2h?teamA={id}&teamB={id}&last=5 endpoint.
    Returns dict with games count, wins/draws/losses for teamA, goals, modifier.
    """
    try:
        data = api_get(
            client, "/fixtures/h2h",
            {"teamA": team_a_id, "teamB": team_b_id, "last": "5"},
        )
        fixtures = data.get("response", [])

        if len(fixtures) < 2:
            # Not enough H2H data — return neutral
            return {
                "games": len(fixtures), "teamA_wins": 0, "draws": 0,
                "teamB_wins": 0, "teamA_gf": 0, "teamB_gf": 0,
                "modifier": 0, "avg_total": 0,
            }

        teamA_wins, draws, teamB_wins = 0, 0, 0
        teamA_gf, teamB_gf = 0, 0

        for f in fixtures:
            status = f["fixture"]["status"]["short"]
            if status not in ("FT", "AET", "PEN"):
                continue

            a_home = f["teams"]["home"]["id"] == team_a_id
            a_goals = (f["goals"]["home"] if a_home else f["goals"]["away"]) or 0
            b_goals = (f["goals"]["away"] if a_home else f["goals"]["home"]) or 0

            teamA_gf += a_goals
            teamB_gf += b_goals

            if a_goals > b_goals:
                teamA_wins += 1
            elif a_goals == b_goals:
                draws += 1
            else:
                teamB_wins += 1

        played = teamA_wins + draws + teamB_wins
        if played == 0:
            return {
                "games": 0, "teamA_wins": 0, "draws": 0,
                "teamB_wins": 0, "teamA_gf": 0, "teamB_gf": 0,
                "modifier": 0, "avg_total": 0,
            }

        teamA_avg = teamA_gf / played
        teamB_avg = teamB_gf / played
        avg_total = (teamA_gf + teamB_gf) / played
        win_pct = teamA_wins / played

        # H2H Modifier (-15 to +15)
        # Positive = teamA has good H2H record vs teamB
        modifier = (
            (win_pct * 10) - 5 +                    # -5 to +5 (centered on 50%)
            (teamA_avg - 1.0) * 5 +               # -5 to +5 (centered on 1.0 goal/game)
            (avg_total - 2.5) * 2                   # -3 to +3 (high-scoring H2Hs boost both)
        )
        modifier = max(min(round(modifier, 1), 15), -15)

        h2h_str = f"{teamA_wins}W {draws}D {teamB_wins}L"
        log.info(
            f"  H2H: Team {team_a_id} vs {team_b_id} -> {h2h_str} "
            f"(avg {teamA_avg:.1f}-{teamB_avg:.1f}, total {avg_total:.1f}/g, mod={modifier})"
        )

        return {
            "games": played,
            "teamA_wins": teamA_wins,
            "draws": draws,
            "teamB_wins": teamB_wins,
            "teamA_gf": teamA_gf,
            "teamB_gf": teamB_gf,
            "modifier": modifier,
            "avg_total": round(avg_total, 2),
        }
    except Exception as e:
        log.warning(f"  H2H fetch failed for {team_a_id} vs {team_b_id}: {e}")
        return {
            "games": 0, "teamA_wins": 0, "draws": 0,
            "teamB_wins": 0, "teamA_gf": 0, "teamB_gf": 0,
            "modifier": 0, "avg_total": 0,
        }


def fetch_form_h2h_for_fixture(client: httpx.Client, fixture: dict) -> dict:
    """Fetch form + H2H for both teams in a fixture.

    Caches result per fixture_id. Costs 3 credits per match (2 form + 1 H2H).
    Skips if already fetched within FORM_FETCH_COOLDOWN.
    """
    fid = fixture["fixture"]["id"]
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]

    # Check cache
    if fid in form_h2h_cache:
        return form_h2h_cache[fid]

    # Check cooldown
    now = time.time()
    if fid in form_fetch_timestamps and (now - form_fetch_timestamps[fid]) < FORM_FETCH_COOLDOWN:
        return None  # don't fetch yet

    log.info(f"  FORM/H2H: Fetching for fixture {fid} ({home_id} vs {away_id})...")

    # Fetch form for both teams
    home_form = fetch_team_form_sync(client, home_id)
    away_form = fetch_team_form_sync(client, away_id)

    # Fetch H2H (teamA=home, teamB=away in the cache)
    h2h_home = fetch_h2h(client, home_id, away_id)

    # For away team, we need H2H from their perspective (swap wins/losses)
    h2h_away = {
        "games": h2h_home["games"],
        "teamA_wins": h2h_home["teamB_wins"],  # away's wins = home's losses
        "draws": h2h_home["draws"],
        "teamB_wins": h2h_home["teamA_wins"],  # away's "opponent wins" = home's wins
        "teamA_gf": h2h_home["teamB_gf"],
        "teamB_gf": h2h_home["teamA_gf"],
        "modifier": -h2h_home["modifier"],  # negate for opposite perspective
        "avg_total": h2h_home["avg_total"],
    }

    result = {
        home_id: home_form,
        away_id: away_form,
        "h2h": h2h_home,         # from home perspective
        "h2h_away": h2h_away,    # from away perspective
    }

    form_h2h_cache[fid] = result
    form_fetch_timestamps[fid] = now

    log.info(
        f"  FORM/H2H: Fixture {fid} cached. "
        f"Home form={home_form['form_string']} ({home_form['form_score']}), "
        f"Away form={away_form['form_string']} ({away_form['form_score']}), "
        f"H2H modifier={h2h_home['modifier']}"
    )

    return result


def get_form_h2h_modifier(fid: int, team_id: int) -> tuple[float, str, dict | None]:
    """Get the combined form + H2H modifier for a team in a fixture.

    Returns (modifier_value, description_string, form_data_or_None).
    Modifier range: roughly -25 to +25.
    """
    cached = form_h2h_cache.get(fid)
    if not cached:
        return 0.0, "", None

    team_form = cached.get(team_id)
    if not team_form:
        return 0.0, "", None

    form_score = team_form.get("form_score", 50)
    form_string = team_form.get("form_string", "N/A")
    avg_gf = team_form.get("avg_gf", 0)

    # Form adjustment: (form_score - 50) * 0.3
    # Range: -15 (form_score=0) to +15 (form_score=100)
    form_adj = (form_score - 50) * 0.3

    # Determine which team's H2H perspective to use
    home_id = None
    for k in cached:
        if isinstance(k, int) and k != "h2h" and k != "h2h_away":
            home_id = k
            break

    if team_id == home_id:
        h2h_data = cached.get("h2h", {})
    else:
        h2h_data = cached.get("h2h_away", {})

    h2h_mod = h2h_data.get("modifier", 0)
    h2h_games = h2h_data.get("games", 0)

    # If fewer than 2 H2H games, neutralize the modifier
    if h2h_games < 2:
        h2h_mod = 0

    total_modifier = round(form_adj + h2h_mod, 1)

    # Build description for signal message
    parts = []
    parts.append(f"Form: {form_string} ({form_score}/100)")
    if h2h_games >= 2:
        h2h_wins = h2h_data.get("teamA_wins", 0)
        h2h_draws = h2h_data.get("draws", 0)
        h2h_losses = h2h_data.get("teamB_wins", 0)
        h2h_avg = h2h_data.get("avg_total", 0)
        parts.append(
            f"H2H: {h2h_wins}W {h2h_draws}D {h2h_losses}L "
            f"(avg {h2h_avg:.1f} goals/game)"
        )
    else:
        parts.append("H2H: N/A (no history)")

    desc = " | ".join(parts)

    return total_modifier, desc, team_form


def save_form_cache():
    """Persist form/H2H cache to disk."""
    try:
        with open(FORM_FILE, "w") as f:
            json.dump(form_h2h_cache, f, default=str)
    except Exception as e:
        log.warning(f"  Failed to save form cache: {e}")


def load_form_cache():
    """Load form/H2H cache from disk."""
    global form_h2h_cache
    if not os.path.exists(FORM_FILE):
        return
    try:
        with open(FORM_FILE, "r") as f:
            form_h2h_cache = json.load(f)
        log.info(f"  Loaded form/H2H cache: {len(form_h2h_cache)} fixture(s)")
    except Exception as e:
        log.warning(f"  Failed to load form cache: {e}")


# --- v10: Goal Pressure Score (GPS) ---
# Composite 0-100 score calculated on EVERY stats poll.
# Uses all available API stats (zero extra cost — data already in response).
# Components: SOT, shots inside box, DA rate, xG, shot volume, ACCELERATION.
# The acceleration component is the key differentiator from v9.9.
GPS_EARLY_WARNING = 55   # 55-74: EARLY WARNING signal
GPS_CRITICAL = 75         # 75+: CRITICAL signal
GPS_BUILDING = 40        # 40-54: logged as BUILDING (no Telegram)

# --- v10.5: Scoreline-aware signal filtering ---
# Don't signal teams that are already comfortably leading (they protect, not attack)
# Don't signal in last 10 minutes (too little time for a goal)
SCORELINE_SKIP_LEAD = 2       # Skip if team leads by 2+ goals
MINUTE_FORM_MIN = 5        # Don't fetch form before 5' (waste of credits)
FORM_GPS_ADJUSTMENT_WEIGHT = 0.3  # How strongly form adjusts GPS


# --- v10: Pressure acceleration tracking ---
# Fixtures where GPS is rising AND multiple stat deltas are positive.
# These get 60s polling to catch the SOT transition in real time,
# even BEFORE SOT reaches 2. This is the key v10 improvement.
pressure_accelerating: set[int] = set()  # fixture IDs

# --- v10: Poll-level data collection for backtesting ---
# Records ALL indicators from every stats poll (not just signals).
# This builds the dataset to empirically validate which combinations
# predict goals within 5/10/15 minutes.
# File path set above via DATA_DIR.

# --- Adaptive polling state ---
last_discovery_time: float = 0.0
last_stats_check: dict[int, float] = {}   # fixture_id -> timestamp of last stats fetch
fast_monitored: set[int] = set()         # fixture IDs currently monitored
fast_priority: dict[int, int] = {}       # fixture_id -> rank score (discovery-time)
cached_fixtures: list[dict] = []        # last discovery result (reused for filtering only)

# --- v10.4: Daily summary dedup ---
# Track which date we last sent the Telegram summary, so we don't spam it
# every loop iteration when no matches are live.
_daily_summary_sent_date: str = ""

# --- v9.5.8: Daily match count for adaptive mode ---
total_matches_today: int = 0
FIRST_SIGNAL_ONLY_THRESHOLD = 50  # v10.2: premium 7500/day, was 20
FULL_TRACKING_THRESHOLD = 30   # v10.2: premium 7500/day, was 15

# --- Fast SOT window state ---
fast_sot_until: dict[int, float] = {}

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
SCHEDULE_RECHECK_INTERVAL = 3 * 3600  # v10.1.2: premium (7500 req/day), back to 3h

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
# Premium (7500 req/day): always track friendlies, no threshold needed.
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

    # --- Championship (40) ---
    35, 34, 40, 49, 42, 33,
    # Leeds, Sunderland, Burnley, Norwich, Leicester, Man Utd (if relegated)
    66, 65, 51, 47,
    # Aston Villa (if relegated), Nott'm Forest (if relegated), West Ham, Chelsea
    # Note: pre-seeded IDs include PL teams that may be in Championship;
    # the cache auto-fills from actual fixtures anyway.

    # --- Belgian Pro League (144) ---
    569, 573, 572,
    # Club Brugge, Union St. Gilloise, Anderlecht

    # --- Scottish Premiership (340) ---
    247, 252, 248, 257,
    # Celtic, Rangers, Aberdeen, Hearts

    # --- Allsvenskan Sweden (332) ---
    691, 695, 692,
    # Malmo FF, AIK, Djurgardens IF

    # --- Segunda División (141) ---
    546, 537, 535,
    # Racing Santander, Zaragoza, Levante

    # --- Serie B (136) ---
    501, 503, 508,
    # Bari, Palermo, Brescia

    # --- Ligue 2 (62) ---
    # Pre-seeds not added — cache auto-fills from Ligue 1 tracked fixtures
    # and any Ligue 2 friendlies during season.

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

    # --- Austrian Bundesliga (310, v10.2 corrected) ---
    556, 559, 558, 560,
    # Salzburg, Rapid Wien, Austria Wien, Sturm Graz

    # --- SuperLiga Serbia ---
    # ID 283 removed (was Liga I Romania, not Serbian SuperLiga)

    # --- HNL Croatia (210) ---
    1992, 1994,
    # Dinamo Zagreb, Hajduk Split

    # --- Czech First League (345) ---
    1021, 1023, 620,
    # Sparta Prague, Slavia Prague, Viktoria Plzen

    # --- Danish Superliga (119) ---
    281, 283, 282, 300,
    # Copenhagen, Brondby, Midtjylland, AGF

    # --- Veikkausliiga Finland ---
    # ID 137 removed (was Coppa Italia, not Veikkausliiga)
    # Add correct ID when found via /leagues endpoint

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


def send_telegram_document(client: httpx.Client, filepath: str, caption: str) -> bool:
    """v10.4: Upload a file as a Telegram document.

    Used to send signal_outcomes.jsonl at end of day for analysis.
    """
    if not os.path.exists(filepath):
        log.warning(f"  Cannot send document — file not found: {filepath}")
        return False
    try:
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            resp = client.post(
                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                },
                files={"document": (filename, f)},
            )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"  Telegram document send failed: {e}")
        return False


def get_top_goalless_sot_player(client: httpx.Client, fixture_id: int) -> tuple[str | None, int, str | None]:
    """v10.3: Find the player with most SOT who has 0 goals across BOTH teams.

    Uses /fixtures/events to get goal scorers (1 credit), then /players
    to get SOT per player (1 credit). Total: 2 credits per signal.

    Returns (player_name, sot_count, team_name) for the best goalless SOT player.
    Returns (None, 0, None) on any failure or if no goalless SOT player exists.
    """
    try:
        # Step 1: Get all goal scorers for this fixture
        data = api_get(client, "/fixtures/events", {"fixture": fixture_id})
        scorer_pids: set[int] = set()
        for event in data.get("response", []):
            if event.get("type") == "Goal":
                pid = event.get("player", {}).get("id")
                if pid:
                    scorer_pids.add(pid)

        # Step 2: Get all players and find top SOT among non-scorers
        pdata = api_get(client, "/players", {"fixture": fixture_id})
        top_name = None
        top_sot = 0
        top_team = None
        for entry in pdata.get("response", []):
            pid = entry.get("player", {}).get("id")
            if pid in scorer_pids:
                continue  # skip goal scorers
            stats_list = entry.get("statistics", [])
            if not stats_list:
                continue
            stat = stats_list[0]
            shots = stat.get("shots", {})
            sot = safe_int(shots.get("on"))
            if sot > top_sot:
                top_sot = sot
                top_name = entry.get("player", {}).get("name", "?")
                top_team = stat.get("team", {}).get("name")
        return (top_name, top_sot, top_team)
    except Exception as e:
        log.warning(f"  Goalless SOT player fetch failed for fixture {fixture_id}: {e}")
        return (None, 0, None)


# get_goal_scorers removed in v10.3 — merged into get_top_goalless_sot_player



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
    # v10.2: Premium thresholds (7500 req/day) — relaxed further
    if quota_remaining <= 0:
        return "STOP"
    if quota_remaining <= 20:
        return "EMERGENCY"      # was 50
    if quota_remaining <= 100:
        return "STRICT"       # was 200
    if quota_remaining <= 300:
        return "CAREFUL"      # was 500
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
        # v10.2: Tightened for premium (was 1800/2400/3000/3600)
        "NORMAL": 900,
        "CAREFUL": 1200,
        "STRICT": 1800,
        "EMERGENCY": 2400,
        "UNKNOWN": 900,
    }.get(budget_mode, 900)

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
        # v10.2: Tightened for premium (was 1800/2400/3000/3600)
        return {
            "NORMAL": 900,
            "CAREFUL": 1200,
            "STRICT": 1800,
            "EMERGENCY": 2400,
            "UNKNOWN": 900,
        }.get(budget_mode, 900)

    # Fallback: no schedule data, no tracked live
    return 900  # v10.2: was 1800


def get_stats_interval(budget_mode: str) -> int:
    """Base interval for statistics batched requests.

    v9.5: With 10+ fixtures monitored, each batched request checks
    multiple fixtures simultaneously. The base interval is the minimum
    time between ANY stats request.
    """
    # v10.2: Tightened for premium (7500 req/day) — was 240/300/420/600
    if budget_mode == "NORMAL":
        return 150
    if budget_mode == "CAREFUL":
        return 200
    if budget_mode == "STRICT":
        return 300
    if budget_mode == "EMERGENCY":
        return 420
    return 420


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
    """
    trend = ""
    sot_rate = 0.0
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    # SAFETY NET: SOT >= 3 ALWAYS triggers CRITICAL (proven v9.x trigger)
    # No quality gate needed — SOT>=3 is the gold standard.
    if sot >= 3:
        last_sot = state["last_sot"] if state else 0
        if sot > last_sot:  # dedup: only on SOT increase
            return "CRITICAL", trend, sot_rate
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
    """v9.9: Detect if a team is building dangerous pressure despite low SOT.

    v10.1: Uses normalized stat access.
    Checks: dangerous attacks rate, shots off target volume.
    Returns (is_building, description_string).
    """
    if not state or not state.get("last_minute", 0) > 0:
        return False, ""
    da = safe_int(get_stat(tstats, "dangerous_attacks"))
    ts = safe_int(get_stat(tstats, "total_shots"))

    prev_da = state.get("last_dangerous_attacks", 0)
    prev_min = state["last_minute"]
    mins_passed = max(minute - prev_min, 1)
    da_delta = da - prev_da
    da_rate = da_delta / mins_passed  # dangerous attacks per minute

    # Need current SOT to compute off-target shots
    sot = safe_int(get_stat(tstats, "sot"))
    shots_off = ts - sot

    # Pressure building conditions:
    #   1. Dangerous attacks increasing (at least 1 every 2 minutes)
    #   2. Lots of shots NOT on target (blocked/wide/saved = keeper busy)
    #   3. Minimum absolute DA to filter early-game noise
    is_building = (
        da_rate >= 0.5
        and shots_off >= 3
        and da >= 10
    )

    desc = (f"DA:{da}(+{da_delta}), TS:{ts}, SOT:{sot}, Off-target:{shots_off}, "
            f"DA-rate:{da_rate:.1f}/min")
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
    "dangerous_attacks":  ["Dangerous Attacks", "Dangerous attacks"],
    "corner_kicks":       ["Corner Kicks", "Corners", "corner kicks"],
    "red_cards":          ["Red Cards", "Red cards"],
    "possession":         ["Ball Possession", "Ball possession", "Possession %", "Possession"],
    "fouls":              ["Fouls", "fouls"],
    "total_passes":       ["Total passes", "Total Passes"],
    "passes_accurate":    ["Passes accurate", "Passes Accurate"],
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
    dangerous_attacks: int,
    xg_value: float | None,
    corners: int,
    minute: int,
    prev_state: dict | None,
    gps_history: list[dict],
    possession: int = 0,  # v10.1: tracked (low weight)
) -> tuple[float, str, dict]:
    """v10.1: Calculate 0-100 Goal Pressure Score.

    Uses ALL available API stats (zero extra API cost — data already in response).
    v10.1 improvements:
      - Windowed per-minute rates (interval-independent acceleration)
      - Sustained pressure bonus (2+ consecutive polls)
      - Possession component (0-3 pts, low weight)
      - Shot quality data in every output for backtesting

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

    # === 2. Shots inside box ratio (0-18 points) ===
    # High inside-box ratio = quality shot selection = more dangerous.
    # Team A (8/11 = 73%) vs Team B (1/3 = 33%) is the key differentiator.
    ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
    ib_pts = min(ib_ratio * 25, 18)  # 72%+ = full 18 pts
    score += ib_pts
    components["inside_box"] = round(ib_pts, 1)

    # === 3. Shot volume relative to minute (0-12 points) ===
    # More shots per minute = sustained attacking intent.
    if minute > 0:
        shots_per_min = total_shots / minute
        sv_pts = min(shots_per_min * 120, 12)  # ~0.1/min = 12 pts
    else:
        sv_pts = 0.0
        shots_per_min = 0.0
    score += sv_pts
    components["shot_vol"] = round(sv_pts, 1)

    # === 4. xG (0-15 points) ===
    # Quality of chances created. 1.0 xG = 15 pts.
    if xg_value is not None and xg_value >= 0:
        xg_pts = min(xg_value * 15, 15)
    else:
        xg_pts = 0.0
    score += xg_pts
    components["xg"] = round(xg_pts, 1)

    # === 5. Dangerous attacks rate (0-8 points) ===
    # Sustained territorial pressure.
    if minute > 0:
        da_per_min = dangerous_attacks / minute
        da_pts = min(da_per_min * 20, 8)
    else:
        da_pts = 0.0
        da_per_min = 0.0
    score += da_pts
    components["da_rate"] = round(da_pts, 1)

    # === 6. Corner kicks (0-4 points) ===
    # Set pieces = additional scoring opportunities.
    corner_pts = min(corners * 1.0, 4)
    score += corner_pts
    components["corners"] = corner_pts

    # === 7. ACCELERATION component (0-15 points) — THE KEY DIFFERENTIATOR ===
    # v10.1: Uses WINDOWED per-minute rates over GPS_WINDOW_MINUTES game minutes.
    # This is interval-independent: a 2-shot delta over 60s scores the same
    # as a 2-shot delta over 240s if the per-minute rates match.
    # It also checks SUSTAINED acceleration over 2+ consecutive polls.
    accel_pts = 0.0
    accel_details = ""
    sustained_count = 0  # v10.1: how many recent polls show 2+ accel indicators

    if prev_state and prev_state.get("last_minute", 0) > 0:
        mins_passed = max(minute - prev_state["last_minute"], 1)
        sot_delta = sot - prev_state.get("last_sot", 0)
        ts_delta = total_shots - prev_state.get("last_total_shots", 0)
        da_delta = dangerous_attacks - prev_state.get("last_dangerous_attacks", 0)
        prev_xg = prev_state.get("last_xg")
        xg_delta = 0.0
        if xg_value is not None and prev_xg is not None:
            try:
                xg_delta = xg_value - float(prev_xg)
            except (ValueError, TypeError):
                pass

        # v10.1: Per-minute rates (interval-independent thresholds)
        sot_rate = sot_delta / mins_passed
        ts_rate = ts_delta / mins_passed
        da_rate_delta = da_delta / mins_passed

        # v10.1: Count accelerating indicators using RATE thresholds.
        # A shot every 2 min (0.5/m), DA every 30s (2.0/m), SOT in 5 min (0.2/m)
        accel_count = 0
        if sot_rate >= 0.15:       # 1 SOT per ~7 min or faster
            accel_count += 1
        if ts_rate >= 0.3:        # 1 shot per ~3 min or faster
            accel_count += 1
        if da_rate_delta >= 1.0:   # 1 DA per minute or faster
            accel_count += 1
        if xg_delta >= 0.1 and (xg_delta / mins_passed) >= 0.02:
            accel_count += 1

        # v10.1: Windowed rates over last GPS_WINDOW_MINUTES game minutes
        # from GPS history (if we have enough data)
        window_sot_rate = sot_rate  # fallback to current poll rate
        window_ts_rate = ts_rate
        window_da_rate = da_rate_delta
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
                window_da_rate = (dangerous_attacks - window_entry.get("dangerous_attacks", 0)) / window_mins

                # Use windowed rates if they show MORE acceleration
                window_accel = 0
                if window_sot_rate >= 0.15:
                    window_accel += 1
                if window_ts_rate >= 0.3:
                    window_accel += 1
                if window_da_rate >= 1.0:
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
            accel_pts = 15  # strong multi-indicator acceleration
        elif accel_count >= 2:
            accel_pts = 10  # moderate acceleration
        elif accel_count == 1:
            accel_pts = 4   # single indicator rising

        # v10.1: SUSTAINED pressure bonus (STRONGER than v10)
        # Count how many of the recent polls show 2+ accelerating indicators.
        for h in gps_history:
            if h.get("accel_count", 0) >= 2:
                sustained_count += 1
        if accel_count >= 2 and sustained_count >= 1:
            accel_pts = min(accel_pts + 4, 15)  # 2+ consecutive polls = sustained

        accel_details = (
            f"SOT {sot_rate:.2f}/m(w:{window_sot_rate:.2f}), "
            f"TS {ts_rate:.2f}/m(w:{window_ts_rate:.2f}), "
            f"DA {da_rate_delta:.1f}/m(w:{window_da_rate:.1f})"
            + (f", xG+{xg_delta:.2f}" if xg_delta > 0 else "")
            + f" [{accel_count} accel{' sustained' if sustained_count > 0 and accel_count >= 2 else ''}]"
        )

    score += accel_pts
    components["acceleration"] = round(accel_pts, 1)
    components["sustained"] = sustained_count  # v10.1: for backtesting

    score = min(score, 100.0)

    desc = (
        f"GPS:{score:.0f} SOT:{sot}({components['sot']}pts) "
        f"IB:{ib_ratio:.0%}({components['inside_box']}pts) "
        f"TS/min:{shots_per_min:.2f}({components['shot_vol']}pts) "
        f"xG:{xg_value or 0:.2f}({components['xg']}pts) "
        f"DA/min:{da_per_min:.2f}({components['da_rate']}pts) "
        f"Accel:{accel_pts:.0f}pts"
    )
    if accel_details:
        desc += f" | {accel_details}"

    return score, desc, components


def record_pressure_poll(
    fid: int, tid: int, tname: str, league: str,
    minute: int, sot: int, total_shots: int,
    shots_inside_box: int, dangerous_attacks: int,
    xg_value: float | None, corners: int,
    gps: float, components: dict,
    accel_count: int, is_home: bool,
    score_home: int, score_away: int,
    possession: int = 0,  # v10.1: tracked for backtesting
) -> None:
    """v10.1: Write poll-level data to JSONL for backtesting.

    This records EVERY stats poll (not just signals), building the dataset
    to answer: "which combination of stats predicts a goal within 5/10/15 min?"
    """
    try:
        entry = {
            "ts": time.time(),
            "fixture_id": fid, "team_id": tid, "team_name": tname,
            "league": league, "minute": minute, "is_home": is_home,
            "sot": sot, "total_shots": total_shots,
            "shots_inside_box": shots_inside_box,
            "dangerous_attacks": dangerous_attacks,
            "xg": round(xg_value, 3) if xg_value is not None else None,
            "corners": corners,
            "possession": possession,
            "gps": round(gps, 1),
            "gps_sot": components.get("sot", 0),
            "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0),
            "gps_xg": components.get("xg", 0),
            "gps_da": components.get("da_rate", 0),
            "gps_accel": components.get("acceleration", 0),
            "gps_poss": components.get("possession", 0),
            "sustained": components.get("sustained", 0),
            "accel_count": accel_count,
            "score_home": score_home, "score_away": score_away,
        }
        with open(POLL_DATA_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.warning(f"  Failed to save poll data: {e}")


def get_accel_count_from_state(
    sot: int, total_shots: int, dangerous_attacks: int,
    xg_value: float | None, prev_state: dict | None, minute: int,
) -> int:
    """v10.1: Count accelerating indicators using RATE-based thresholds.

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
    da_delta = dangerous_attacks - prev_state.get("last_dangerous_attacks", 0)

    # v10.1: Same rate thresholds as GPS calculation
    if (sot_delta / mins_passed) >= 0.15:
        accel_count += 1
    if (ts_delta / mins_passed) >= 0.3:
        accel_count += 1
    if (da_delta / mins_passed) >= 1.0:
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
    # v10: Clean pressure acceleration tracking
    for fid in list(pressure_accelerating):
        if fid not in live_fixture_ids:
            pressure_accelerating.discard(fid)
    # v10: Clean GPS history
    to_delete_gps = [k for k in team_gps_history if k[0] not in live_fixture_ids]
    for k in to_delete_gps:
        del team_gps_history[k]


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

    # v9.5.3: Penalize only if BOTH teams have signaled
    both_signaled = all(
        (fid, tid) in signaled_teams
        for (f, tid) in team_state if f == fid
    )
    if both_signaled and is_signaled:
        base = int(base * 0.5)

    return base


def get_sot_based_interval(fid: int, base_interval: int) -> int:
    """v10: Pressure-aware polling interval with acceleration tiers.

    v10 adds PRESSURE ACCELERATION as a top-priority tier:
      v10 NEW: Pressure-accelerating (GPS rising, 2+ deltas): 60s
      v9.7:   SOT-accelerating (SOT increased last poll):    60s
      v9.7:   SOT >= 3:                                     90s
              SOT == 2 + fast window:                        120s
              SOT == 2:                                     180s
              SOT == 1:                                     base interval
              Unknown / 0 SOT:                              1.5x or 2.5x base
    """
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    team_signaled_count = sum(1 for (f, t) in signaled_teams if f == fid)
    both_teams_signaled = team_signaled_count >= 2

    # v10: Pressure acceleration — catches pre-SOT pressure build
    # This is THE key improvement: 60s polling starts when shots, DA,
    # and xG are ALL rising, even if SOT is still 0 or 1.
    if (fid in pressure_accelerating
            and not both_teams_signaled
            and best_sot < 3):
        interval = 60
    # v9.7: SOT acceleration tier — still top priority
    elif fid in accelerating_fixtures and not both_teams_signaled:
        interval = 60
    elif not has_state:
        interval = base_interval
    elif best_sot >= 3 and not both_teams_signaled:
        interval = 90
    elif best_sot >= 1 and not both_teams_signaled and is_fast_sot_active(fid):
        interval = 120
    elif best_sot >= 2:
        interval = 180
    elif best_sot == 1:
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

def is_scoreline_blocked(is_home: bool, sh: int, sa: int, minute: int) -> tuple[bool, str]:
    """v10.5: Check if a signal should be blocked due to scoreline context.

    Rules:
    - Team leads by 2+ goals -> likely protecting lead, not attacking
    - Match is past 78' -> too little time remaining
    - Team already has 3+ goals -> diminishing returns

    Returns (blocked: bool, reason: str).
    """
    tg = sh if is_home else sa
    og = sa if is_home else sh
    lead = tg - og

    if lead >= SCORELINE_SKIP_LEAD:
        return True, f"leading {tg}-{og} by {lead} (protecting)"
    if minute >= 78:
        return True, f"too late ({minute}', <12 min remaining)"
    if tg >= 3 and lead >= 2:
        return True, f"already {tg} goals (sitting back)"

    return False, ""


def parse_xg(tstats: dict) -> str:
    """Extract xG from team statistics. Returns string or 'N/A'.

    v10.1: Uses normalized alias lookup.
    """
    for key in ("Expected Goals", "expectedGoals", "Expected goals"):
        val = get_stat(tstats, key, default=None)
        if val is not None and val != "0":
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


def _append_jsonl(filepath: str, entry: dict) -> None:
    """Append one JSON line to a file (used by all persistence functions)."""
    try:
        with open(filepath, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.warning(f"  Failed to write to {os.path.basename(filepath)}: {e}")


def save_signal_entry(entry: dict) -> None:
    """v10.4: Write a NEW signal entry to JSONL immediately when sent.

    This is the critical fix: previously, signal entries only existed in memory.
    If the bot restarted before the match ended, the signal was lost forever.
    Now every signal is persisted to disk the instant it's sent.
    """
    _append_jsonl(OUTCOMES_FILE, entry)


def save_signal_state() -> None:
    """v10.4: Persist signaled_teams and signaled_fixtures to disk.

    Prevents duplicate signals when Railway redeploys mid-match.
    Called after every new signal and on clean shutdown.
    """
    try:
        # Convert tuple keys to string for JSON serialization
        state = {
            "signaled_teams": {
                f"{fid},{tid}": v
                for (fid, tid), v in signaled_teams.items()
            },
            "signaled_fixtures": list(signaled_fixtures),
            "saved_at": time.time(),
        }
        with open(SIGNAL_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.warning(f"  Failed to save signal state: {e}")


def load_signal_state() -> None:
    """v10.4: Load signaled_teams and signaled_fixtures from disk.

    Restores signal dedup state after a restart, preventing duplicate
    signals for the same team in the same match.
    """
    global signaled_teams, signaled_fixtures
    if not os.path.exists(SIGNAL_STATE_FILE):
        return
    try:
        with open(SIGNAL_STATE_FILE, "r") as f:
            state = json.load(f)

        restored_teams = 0
        raw_teams = state.get("signaled_teams", {})
        for key, val in raw_teams.items():
            try:
                fid, tid = key.split(",")
                signaled_teams[(int(fid), int(tid))] = val
                restored_teams += 1
            except (ValueError, AttributeError):
                pass

        raw_fixtures = state.get("signaled_fixtures", [])
        signaled_fixtures.update(int(x) for x in raw_fixtures if isinstance(x, (int, str, float)))

        if restored_teams > 0:
            log.info(
                f"  Restored signal state: {restored_teams} team(s), "
                f"{len(signaled_fixtures)} fixture(s) from {SIGNAL_STATE_FILE}"
            )

        # Clean up stale entries (fixtures that ended >6h ago are safe to drop)
        cutoff = time.time() - 6 * 3600
        if state.get("saved_at", 0) < cutoff:
            log.info("  Signal state file is stale (>6h), clearing to start fresh")
            signaled_teams.clear()
            signaled_fixtures.clear()
            try:
                os.remove(SIGNAL_STATE_FILE)
            except OSError:
                pass
    except Exception as e:
        log.warning(f"  Failed to load signal state: {e}")


def save_outcome(entry: dict) -> None:
    """Append the resolved signal entry to JSONL.

    v10.4: Append-only design. The original signal entry was already written
    when the signal was sent. When resolved, we write the full updated entry
    again. On load, we deduplicate by keeping the LAST entry per signal
    (identified by fixture_id + team_id + signal_time).
    This avoids rewriting the entire file on every resolution.
    """
    _append_jsonl(OUTCOMES_FILE, entry)


def load_all_outcomes() -> list[dict]:
    """v10.4: Load ALL signal entries from JSONL (both resolved and pending).

    Append-only file may have multiple entries for the same signal
    (original + resolution update). We keep the LAST occurrence of each
    unique signal (identified by fixture_id + team_id + signal_time).

    Previous version only loaded pending (unresolved) entries, which meant:
    - Resolved outcomes were lost from memory after summary/clear
    - Win rate could never be recalculated after restart
    Now we load everything for full backtesting capability.
    """
    seen: dict[tuple, dict] = {}  # (fid, tid, signal_time) -> latest entry
    if not os.path.exists(OUTCOMES_FILE):
        return []
    try:
        with open(OUTCOMES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (
                    entry.get("fixture_id"),
                    entry.get("team_id"),
                    entry.get("signal_time"),
                )
                seen[key] = entry  # last write wins
    except Exception as e:
        log.warning(f"  Failed to load outcomes file: {e}")
    return list(seen.values())


# Keep old name as alias for any remaining references
load_pending_outcomes = load_all_outcomes


def check_signal_outcomes(fixture: dict) -> None:
    """v10: Check pending signals for this fixture.

    Four-track system:
      outcome_5min:  HIT if goal within 5 game min
      outcome_10min: HIT if goal within 10 game min
      outcome_15min: HIT if goal within 15 game min
      outcome_full:  HIT if goal at any point before match ends
    Entry is 'resolved' only when ALL are decided (match must end).
    """
    fid = fixture["fixture"]["id"]
    status = fixture["fixture"]["status"]["short"]
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0

    for entry in signal_outcomes:
        if entry["fixture_id"] != fid or entry["resolved"]:
            continue

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
                save_outcome(entry)

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
                save_outcome(entry)


def log_outcome_summary() -> None:
    """v10: Log hit-rate summary for all resolved signals today.

    Shows 5/10/15-min and full-match hit rates,
    broken down by: tier, signal number, GPS-triggered vs SOT-triggered,
    and GPS score ranges (the key v10 addition for backtesting).
    """
    resolved = [e for e in signal_outcomes if e["resolved"]]
    if not resolved:
        return

    total = len(resolved)
    h5 = sum(1 for e in resolved if e.get("outcome_5min") == "HIT")
    h10 = sum(1 for e in resolved if e.get("outcome_10min") == "HIT")
    h15 = sum(1 for e in resolved if e.get("outcome_15min") == "HIT")
    hf = sum(1 for e in resolved if e.get("outcome_full") == "HIT")

    log.info("")
    log.info(f"=== SIGNAL OUTCOME SUMMARY: {total} signals ===")
    log.info(f"  5-min:   {h5}/{total} ({h5/total*100:.0f}%)")
    log.info(f"  10-min:  {h10}/{total} ({h10/total*100:.0f}%)")
    log.info(f"  15-min:  {h15}/{total} ({h15/total*100:.0f}%)")
    log.info(f"  Full:    {hf}/{total} ({hf/total*100:.0f}%)")

    # By tier (v10: EARLY WARNING vs CRITICAL)
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

    # By signal number (1st vs 2nd vs 3rd+)
    for sig_label, sig_filter in [("1st signal", lambda e: e.get("sig_num") == 1),
                                   ("2nd signal", lambda e: e.get("sig_num") == 2),
                                   ("3rd+ signal", lambda e: e.get("sig_num", 0) >= 3)]:
        group = [e for e in resolved if sig_filter(e)]
        if not group:
            continue
        g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        g_total = len(group)
        avg_gps = sum(e.get("gps", 0) for e in group) / g_total
        log.info(
            f"  {sig_label}: 15min {g_h15}/{g_total} ({g_h15/g_total*100:.0f}%) | "
            f"full {g_hf}/{g_total} ({g_hf/g_total*100:.0f}%) | avg GPS: {avg_gps:.0f}"
        )

    # v10: GPS-triggered vs SOT-triggered
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

    # v10: By GPS score range (key for validating thresholds)
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


def send_daily_summary_telegram(client: httpx.Client) -> None:
    """v10.4: Send end-of-day summary + data file to Telegram chat.

    Called right after log_outcome_summary() when all matches are done.
    Sends:
    1. A human-readable summary message with hit rates
    2. signal_outcomes.jsonl as a document (for deeper analysis)
    3. pressure_polls.jsonl as a document (for GPS calibration)
    """
    # Reload from disk to get the complete picture (memory may have been cleared)
    all_entries = load_all_outcomes()
    resolved = [e for e in all_entries if e.get("resolved")]
    if not resolved:
        return

    total = len(resolved)
    today = time.strftime("%Y-%m-%d")
    h5 = sum(1 for e in resolved if e.get("outcome_5min") == "HIT")
    h10 = sum(1 for e in resolved if e.get("outcome_10min") == "HIT")
    h15 = sum(1 for e in resolved if e.get("outcome_15min") == "HIT")
    hf = sum(1 for e in resolved if e.get("outcome_full") == "HIT")

    def pct(n, d):
        return f"{n}/{d} ({n/d*100:.0f}%)" if d > 0 else "N/A"

    lines = [
        f"📊 DAILY SIGNAL REPORT — {today}",
        f"{total} signal(s) resolved",
        "",
        f"⏱ 5-min:  {pct(h5, total)}",
        f"⏱ 10-min: {pct(h10, total)}",
        f"⏱ 15-min: {pct(h15, total)}",
        f"⏱ Full:    {pct(hf, total)}",
        "",
        "── By Signal Number ──",
    ]

    for sig_label, sig_filter in [
        ("1st signal", lambda e: e.get("sig_num") == 1),
        ("2nd signal", lambda e: e.get("sig_num") == 2),
        ("3rd+ signal", lambda e: e.get("sig_num", 0) >= 3),
    ]:
        group = [e for e in resolved if sig_filter(e)]
        if not group:
            continue
        g_t = len(group)
        g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        avg_gps = sum(e.get("gps", 0) for e in group) / g_t
        lines.append(f"  {sig_label}: 15m {pct(g_h15, g_t)} | full {pct(g_hf, g_t)} | avg GPS {avg_gps:.0f}")

    lines.append("")
    lines.append("── By Tier ──")
    for tier in ("CRITICAL", "EARLY WARNING"):
        tier_r = [e for e in resolved if e.get("tier") == tier]
        if not tier_r:
            continue
        t_t = len(tier_r)
        t_h5 = sum(1 for e in tier_r if e.get("outcome_5min") == "HIT")
        t_h15 = sum(1 for e in tier_r if e.get("outcome_15min") == "HIT")
        t_hf = sum(1 for e in tier_r if e.get("outcome_full") == "HIT")
        avg_gps = sum(e.get("gps", 0) for e in tier_r) / t_t
        hits_15 = [e for e in tier_r if e.get("outcome_15min") == "HIT" and e.get("goal_minute_15")]
        avg_to_goal = ""
        if hits_15:
            avg = sum(e["goal_minute_15"] - e["game_minute"] for e in hits_15) / len(hits_15)
            avg_to_goal = f" | avg +{avg:.0f}' to goal"
        lines.append(f"  {tier}: 5m {pct(t_h5, t_t)} | 15m {pct(t_h15, t_t)} | full {pct(t_hf, t_t)} | GPS {avg_gps:.0f}{avg_to_goal}")

    lines.append("")
    lines.append("── By GPS Range ──")
    for range_label, range_filter in [
        ("55-64", lambda e: 55 <= e.get("gps", 0) < 65),
        ("65-74", lambda e: 65 <= e.get("gps", 0) < 75),
        ("75-84", lambda e: 75 <= e.get("gps", 0) < 85),
        ("85+", lambda e: e.get("gps", 0) >= 85),
    ]:
        group = [e for e in resolved if range_filter(e)]
        if not group:
            continue
        g_t = len(group)
        g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        lines.append(f"  GPS {range_label}: 15m {pct(g_h15, g_t)} | full {pct(g_hf, g_t)}")

    lines.append("")
    lines.append("── Individual Signals ──")
    for e in resolved:
        status_15 = e.get("outcome_15min", "?")
        status_full = e.get("outcome_full", "?")
        icon_15 = "✅" if status_15 == "HIT" else "❌"
        icon_full = "✅" if status_full == "HIT" else "❌"
        sig_n = e.get("sig_num", "?")
        lines.append(
            f"  {icon_15} {icon_full} {e.get('team_name', '?')} [{e.get('league', '?')}] "
            f"{e.get('game_minute', '?')}' SOT={e.get('sot', '?')} GPS={e.get('gps', '?')} "
            f"({sig_n}{'st' if sig_n == 1 else 'nd' if sig_n == 2 else 'rd' if sig_n == 3 else 'th'})"
        )

    msg = "\n".join(lines)
    send_telegram(client, msg)
    log.info("  Daily summary sent to Telegram")

    # Send the JSONL data file
    if os.path.exists(OUTCOMES_FILE):
        send_telegram_document(
            client, OUTCOMES_FILE,
            f"signal_outcomes.jsonl — {today} ({total} signals)"
        )

    # Also send polls file if it exists and isn't too large (<5MB Telegram limit)
    try:
        if os.path.exists(POLL_DATA_FILE):
            polls_size = os.path.getsize(POLL_DATA_FILE)
            if polls_size < 5 * 1024 * 1024:
                send_telegram_document(
                    client, POLL_DATA_FILE,
                    f"pressure_polls.jsonl — {today} (GPS calibration data)"
                )
            else:
                log.info(f"  Polls file too large ({polls_size / 1024 / 1024:.1f}MB), skipping")
    except Exception as e:
        log.warning(f"  Polls file send skipped: {e}")


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
    _logged_stat_keys = False  # v10.3.2: log available stat keys once per fixture
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
                # v10.1: store BOTH original key and stripped version
                # so alias lookup always has something to find
                tmap[stat_type] = "0" if value is None else str(value).strip()
                if stat_type != stat_type.strip():
                    tmap[stat_type.strip()] = tmap[stat_type]
        # v10.3.2: Log all stat keys for first team (diagnose missing DA/xG/Possession)
        if not _logged_stat_keys and tmap:
            _logged_stat_keys = True
            log.info(f"  Fixture {fid} stat keys: {list(tmap.keys())}")
        teams_data[tname] = tmap

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
    # v10: Calculate GPS for every team on every poll (zero extra API cost).
    # This builds the dataset and drives acceleration-aware polling.
    league = LEAGUE_IDS.get(
        fixture["league"]["id"], fixture["league"].get("name", "?")
    )
    sh = fixture["goals"]["home"] or 0
    sa = fixture["goals"]["away"] or 0

    for tid, tname in ((home_tid, home["name"]), (away_tid, away["name"])):
        tstats = teams_data.get(tname)
        if not tstats:
            continue

        # v10.1: All stat access via get_stat() (normalized aliases)
        sot = safe_int(get_stat(tstats, "sot"))

        # v10: Parse ALL available stats for GPS calculation
        total_shots = safe_int(get_stat(tstats, "total_shots"))
        dangerous_attacks = safe_int(get_stat(tstats, "dangerous_attacks"))
        shots_inside_box = safe_int(get_stat(tstats, "shots_inside_box"))
        corners = safe_int(get_stat(tstats, "corner_kicks"))
        xg_str = team_xg.get(tname, "N/A")
        xg_value = safe_float(xg_str)
        possession = safe_int(get_stat(tstats, "possession"))  # v10.1: tracked

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

        # === v10.1: Calculate Goal Pressure Score (includes possession) ===
        gps, gps_desc, components = calculate_goal_pressure_score(
            sot=sot, total_shots=total_shots,
            shots_inside_box=shots_inside_box,
            dangerous_attacks=dangerous_attacks,
            xg_value=xg_value, corners=corners,
            minute=minute, prev_state=state,
            gps_history=history,
            possession=possession,
        )

        # Count accelerating indicators (for polling + logging)
        accel_count = get_accel_count_from_state(
            sot, total_shots, dangerous_attacks, xg_value, state, minute
        )

        # === v10: Log GPS on every poll ===
        if gps >= GPS_BUILDING:
            stage_label = "BUILDING" if gps < GPS_EARLY_WARNING else "EARLY" if gps < GPS_CRITICAL else "CRITICAL"
            log.info(
                f"  GPS {stage_label}: {tname} | {gps_desc}"
            )

        # === v10.1: Record poll data for backtesting (includes possession) ===
        is_home_team = (tid == home_tid)
        record_pressure_poll(
            fid=fid, tid=tid, tname=tname, league=league,
            minute=minute, sot=sot, total_shots=total_shots,
            shots_inside_box=shots_inside_box,
            dangerous_attacks=dangerous_attacks,
            xg_value=xg_value, corners=corners,
            gps=gps, components=components,
            accel_count=accel_count, is_home=is_home_team,
            score_home=sh, score_away=sa,
            possession=possession,
        )

        # === v10.1: Update GPS history (includes timestamp + xg for window calc) ===
        new_history_entry = {
            "ts": time.time(),
            "gps": round(gps, 1),
            "sot": sot, "total_shots": total_shots,
            "dangerous_attacks": dangerous_attacks,
            "accel_count": accel_count, "minute": minute,
            "xg": xg_value,  # v10.1: for windowed xG delta
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

        # v10.1: Log WHY GPS was high but signal was blocked (for threshold tuning)
        if not tier and gps >= GPS_EARLY_WARNING:
            gate_reason = []
            if ib_ratio < 0.30:
                gate_reason.append(f"IB={ib_ratio:.0%}<30%")
            if sustained < 1 and gps < GPS_CRITICAL:
                gate_reason.append("not sustained")
            log.info(
                f"  GATE SKIP: {tname} GPS={gps:.0f} SOT={sot} — {', '.join(gate_reason)}"
            )

        # Store state AFTER classification (for next comparison)
        team_state[(fid, tid)] = {
            "last_sot": sot,
            "last_minute": minute,
            "last_xg": xg_str if xg_str != "N/A" else None,
            "last_dangerous_attacks": dangerous_attacks,
            "last_total_shots": total_shots,
            "last_shots_inside_box": shots_inside_box,
            "last_corners": corners,
            "last_possession": possession,  # v10.1: tracked
        }

        if not tier:
            continue

        # === v10.5: Scoreline-aware filtering ===
        is_home_sg = (tid == home_tid)
        blocked, block_reason = is_scoreline_blocked(is_home_sg, sh, sa, minute)
        if blocked:
            log.info(
                f"  SCORELINE BLOCK: {tname} - "
                f"{sot} SOT GPS={gps:.0f} ({block_reason}) "
                f"(fixture {fid})"
            )
            continue

        # === v10.5: Fetch form + H2H (once per fixture, 3 credits) ===
        form_modifier = 0.0
        form_desc = ""
        form_data = None
        adjusted_gps = gps

        if minute >= MINUTE_FORM_MIN and fid not in form_h2h_cache:
            try:
                fetch_form_h2h_for_fixture(client, fixture)
            except Exception as e:
                log.warning(f"  Form/H2H fetch skipped for fixture {fid}: {e}")

        if fid in form_h2h_cache:
            form_modifier, form_desc, form_data = get_form_h2h_modifier(fid, tid)
            adjusted_gps = gps + form_modifier
            adjusted_gps = max(0, min(adjusted_gps, 100))

            log.info(
                f"  FORM ADJUST: {tname} GPS {gps:.0f} -> {adjusted_gps:.0f} "
                f"(modifier={form_modifier:+.1f})"
            )

        # --- v9.5.4: Signal limit rules ---
        team_sig = signaled_teams.get((fid, tid))
        sig_count = team_sig["count"] if team_sig else 0

        # --- v9.5.8: First-signal-only on busy days ---
        if is_first_signal_only_mode() and sig_count >= 1:
            log.info(
                f"  SKIP {tier}: {tname} - "
                f"{sot} SOT GPS={gps:.0f} (first-signal-only mode, {total_matches_today} games) "
                f"(fixture {fid})"
            )
            continue

        if sig_count >= 2:
            last_signal_sot = team_sig.get("sot_at_last_signal", 0)
            sot_jump = sot - last_signal_sot
            if sot_jump < 2:
                log.info(
                    f"  BLOCKED {tier}: {tname} - "
                    f"{sot} SOT (+{sot_jump} from last signal, need +2) "
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
                "sot_at_last_signal": sot,
            }
        else:
            signaled_teams[(fid, tid)]["count"] = sig_count + 1
            signaled_teams[(fid, tid)]["goals_at_last_signal"] = goals_now
            signaled_teams[(fid, tid)]["sot_at_last_signal"] = sot
        signaled_fixtures.add(fid)

        # Build the signal message
        sig_num = sig_count + 1
        sig_label = f"{sig_num}{'st' if sig_num == 1 else 'nd' if sig_num == 2 else 'rd' if sig_num == 3 else 'th'}"
        shots_off = total_shots - sot
        ib_pct = f"{shots_inside_box / total_shots * 100:.0f}%" if total_shots > 0 else "N/A"

        # Determine trigger type for message
        if tier == "EARLY WARNING":
            trigger = f"GPS-TRIGGERED (accel{' sustained' if sustained >= 1 else ''}, IB={ib_ratio:.0%})"
        else:
            trigger = "SOT>=3 CONFIRMED"

        # --- v10.3: Fetch top goalless SOT player (both teams) ---
        # Costs 2 extra credits per signal (events + players).
        # Graceful: if fetch fails, signal still sends without player info.
        player_line = ""
        try:
            top_player, top_player_sot, player_team = get_top_goalless_sot_player(client, fid)
            if top_player and top_player_sot > 0:
                player_line = f"\n\nshootout: {top_player} ({player_team}) - {top_player_sot} SOT, 0 goals"
            else:
                log.info(f"  No goalless SOT player found (API has no player data for this match)")
        except Exception as e:
            log.warning(f"  Player lookup skipped: {e}")

        # --- v10.5: Build GPS line with form adjustment ---
        if abs(form_modifier) > 0.05:
            gps_line = (
                f"GPS: {gps:.0f} -> {adjusted_gps:.0f}/100 | Trigger: {trigger}\n"
                f"  Form+H2H: {form_desc} ({form_modifier:+.1f})"
            )
        else:
            gps_line = f"GPS: {gps:.0f}/100 | Trigger: {trigger}"

        msg = (
            f"{tier_emoji(tier)} {tier} GOAL PRESSURE ({sig_label})\n\n"
            f"{home['name']}  {sh} - {sa}  {away['name']}\n"
            f"{league} | {minute}'\n\n"
            f"{tname}\n"
            f"SOT: {sot} | Total Shots: {total_shots} (In-box: {ib_pct})\n"
            f"xG: {xg_str} | DA: {dangerous_attacks} | Corners: {corners}\n"
            f"Off-target: {shots_off} | Possession: {possession}%\n"
            f"Opponent SOT: {opponent_sot} | Opponent xG: {opponent_xg}\n\n"
            f"{gps_line}\n"
            f"Red Cards: {red_card_str}"
            f"{player_line}"
        )
        if trend:
            msg += f"\nTrend: {trend}"

        if send_telegram(client, msg):
            log.info(
                f"  SIGNAL {tier}: {tname} - "
                f"{sot} SOT, GPS={gps:.0f}, xG={xg_str} (fixture {fid}, "
                f"{sig_label} signal, {trigger})"
            )

        signals_sent.append({
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "fixture": fid, "team": tname, "league": league,
            "minute": minute, "sot": sot, "xg": xg_str,
            "gps": round(gps, 1),
            "red_cards": red_card_str, "tier": tier,
            "trend": trend, "is_new": is_new_team,
        })

        # v10.1: Enriched outcome record with all raw indicators
        opp_goals = (sa if is_home_sg else sh)
        outcome_entry = {
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": league,
            "signal_time": time.time(),
            "signal_clock": time.strftime("%Y-%m-%d %H:%M"),
            "game_minute": minute,
            "sot": sot,
            "total_shots": total_shots,
            "shots_inside_box": shots_inside_box,
            "ib_ratio": round(ib_ratio, 3),
            "dangerous_attacks": dangerous_attacks,
            "xg": round(xg_value, 3) if xg_value is not None else None,
            "corners": corners,
            "possession": possession,
            "gps": round(gps, 1),
            "gps_sot": components.get("sot", 0),
            "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0),
            "gps_xg": components.get("xg", 0),
            "gps_da": components.get("da_rate", 0),
            "gps_accel": components.get("acceleration", 0),
            "gps_poss": components.get("possession", 0),
            "sustained": sustained,
            "accel_count": accel_count,
            "tier": tier,
            "goals_at_signal": goals_now,
            "opponent_goals_at_signal": opp_goals,
            "is_home": is_home_sg,
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
            # v10.5: Form & H2H context
            "form_score": form_data.get("form_score") if form_data else None,
            "form_string": form_data.get("form_string") if form_data else None,
            "form_avg_gf": form_data.get("avg_gf") if form_data else None,
            "form_avg_ga": form_data.get("avg_ga") if form_data else None,
            "h2h_modifier": form_modifier,
            "gps_raw": round(gps, 1),
            "gps_adjusted": round(adjusted_gps, 1),
            "scoreline_blocked": False,  # reached here = not blocked
        }
        signal_outcomes.append(outcome_entry)

        # v10.4: Persist signal to disk IMMEDIATELY + update dedup state
        save_signal_entry(outcome_entry)
        save_signal_state()

        # v9.5.8: In first-signal-only mode, remove fixture from monitoring
        if is_first_signal_only_mode():
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            log.info(
                f"  FIRST-SIGNAL-DONE: fixture {fid} removed from monitoring "
                f"(moving on, {len(fast_monitored)} still monitored)"
            )

    # --- v9.7: Update SOT acceleration flag ---
    if _sot_increased:
        accelerating_fixtures.add(fid)
        log.debug(f"  SOT-Accelerating: fixture {fid} (SOT {prev_best_sot}->{best_current_sot})")
    else:
        accelerating_fixtures.discard(fid)

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
    # STEP 1: Single /fixtures?ids= call (skip if we know it's useless)
    # ================================================================
    # v10.1.1: If batch endpoint returns empty, stop wasting 1 credit/cycle
    if _ids_endpoint_has_stats is False:
        batch_response = []
        refreshed_fixtures = {}
    else:
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
    # STEP 2: Detect whether batch returns stats (or is empty/useless)
    # ================================================================
    if _ids_endpoint_has_stats is None and not batch_response:
        # Batch returned empty — stop trying, go straight to individual calls
        _ids_endpoint_has_stats = False
        log.info(
            "  v10.1.1: /fixtures?ids= returns empty response. "
            "Skipping batch call from now on (saves 1 credit/cycle)."
        )
    elif _ids_endpoint_has_stats is None and batch_response:
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
                # v10.3.1: INFO log + mark dead after 2 empty polls (saves credits)
                empty_count = dead_fixtures.get(f"_empty_stats_{fid}", 0)
                empty_count += 1
                dead_fixtures[f"_empty_stats_{fid}"] = empty_count
                if empty_count == 1:
                    log.info(f"  Fixture {fid}: empty statistics (will retry once)")
                else:
                    log.info(f"  Fixture {fid}: empty statistics 2x — removing from monitoring (no live stats available)")
                    fast_monitored.discard(fid)
                    expire_fast_sot(fid)
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
    if fid in active_friendly_fixtures:
        return True
    if lid in LEAGUE_IDS:
        # v10.3.2: Runtime league name verification — catch ID mismatches
        # like 137 (expected Veikkausliiga, actually Coppa Italia)
        expected = LEAGUE_IDS[lid]
        actual = fixture["league"].get("name", "")
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        if len(expected_words & actual_words) < 1:
            log.warning(
                f"  LEAGUE MISMATCH at runtime: ID {lid} expected '{expected}' "
                f"but fixture says '{actual}' — SKIPPING (remove ID from LEAGUE_IDS)"
            )
            return False
        return True
    return False


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
        # is in our 20 tracked leagues.
        global active_friendly_fixtures
        # v10.1.2: Always track friendlies (premium: 7500 req/day, no threshold)
        added = 0
        skipped = 0
        if friendly_candidates:
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
        if not friendly_candidates and active_friendly_fixtures:
            log.info(f"  Clearing {len(active_friendly_fixtures)} friendly fixture IDs (no candidates)")
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
    log.info("Football Bot v10.5.0 — Form, H2H + Scoreline-Aware Signals")
    log.info("=" * 60)
    log.info(f"Data dir: {DATA_DIR}")
    log.info(f"Outcomes file: {OUTCOMES_FILE}")
    log.info(f"Polls file: {POLL_DATA_FILE}")
    log.info(f"Signal state file: {SIGNAL_STATE_FILE}")
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (round-robin for rate-limit resilience, NOT quota expansion)")
    log.info("")
    log.info("v10.5 CHANGES (form + H2H + scoreline awareness):")
    log.info("  FORM: Fetches last 5 fixtures per team, calculates 0-100 form score")
    log.info("    — W=12pts, D=7pts, avg GF (up to 12), defensive bonus (up to 7)")
    log.info("  H2H: Last 5 meetings between teams, calculates -15 to +15 modifier")
    log.info("    — Win%, goals/game, total goals/game all factor in")
    log.info("  GPS ADJUSTMENT: FINAL_GPS = raw_gps + (form-50)*0.3 + h2h_modifier")
    log.info("    — displayed as 'GPS: 62 -> 78/100' in signal message")
    log.info("  SCORELINE FILTER: Skip signals for teams leading by 2+ goals")
    log.info("  LATE MATCH FILTER: Skip signals after 78' (too little time)")
    log.info("  API COST: 3 credits per match (2 form + 1 H2H), cached for 6h")
    log.info("  BACKTESTING: form_score, h2h_modifier, gps_raw, gps_adjusted in outcomes")
    log.info("")
    log.info("v10.4 CHANGES (persistent data pipeline + daily report):")
    log.info("  PERSISTENT OUTCOMES: signal_outcomes.jsonl written on EVERY signal (not just resolve)")
    log.info("    — survives restarts, enables offline win-rate analysis")
    log.info("  SIGNAL DEDUP: signaled_teams/signaled_fixtures saved to signal_state.json")
    log.info("    — prevents duplicate signals when Railway redeploys mid-match")
    log.info("  FULL RELOAD: all outcomes (resolved + pending) loaded on startup")
    log.info("  DAILY TELEGRAM REPORT: summary message + JSONL files sent to chat at EOD")
    log.info("    — hit rates by signal #, tier, GPS range, per-signal results")
    log.info("    — signal_outcomes.jsonl + pressure_polls.jsonl as documents")
    log.info("  DATA_DIR env var: set to /data on Railway for volume persistence")
    log.info("")
    log.info("v10.3 CHANGES:")
    log.info("  SHOOTOUT LINE: searches BOTH teams for top SOT player with 0 goals")
    log.info("    (was: only checked signaling team, missed goalless players on other team)")
    log.info("  PLAYER LOOKUP: /fixtures/events + /players (2 credits per signal)")
    log.info("")
    log.info("v10.2 CHANGES (premium expansion):")
    log.info("  LEAGUES: 20 -> 27 (added Championship, Serie B, Ligue 2, Segunda,")
    log.info("           Belgian Pro League, Scottish Prem, Allsvenskan Sweden)")
    log.info("  THRESHOLDS: FIRST_SIGNAL_ONLY 20->50, FULL_TRACKING 15->30")
    log.info("  POLLING: base 240s->150s, discovery 1800s->900s (NORMAL mode)")
    log.info("  CREDIT TIERS: NORMAL>300 (was >500), skip guard <=2 (was <=10)")
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
    log.info("v10.1.2 NEW: Startup league ID verification (1 credit) — auto-removes wrong IDs")
    log.info("")
    log.info("v10 GPS (preserved):")
    log.info("  Components: SOT(28) + InBox(18) + ShotVol(12) + xG(15) + DA(8) + Corners(4) + Poss(3) + ACCEL(15)")
    log.info(f"  Thresholds: BUILDING>={GPS_BUILDING} EARLY_WARNING>={GPS_EARLY_WARNING} CRITICAL>={GPS_CRITICAL}")
    log.info("")
    log.info("PRESERVED FROM v9.x:")
    log.info("  SOT>=3 ALWAYS triggers CRITICAL (safety net — proven v9.x trigger)")
    log.info("  Credit optimization: batched requests, night gate, tomorrow cache")
    log.info("  Dead fixture removal, first-signal-only on busy days")
    log.info("  Unified /fixtures?ids= call (1 credit for all fixtures if stats embedded)")
    log.info("")
    log.info("POLLING TIERS (per fixture, before 2x both-signaled multiplier):")
    log.info("  v10.1:  Pressure-accelerating (rate-based 2+ deltas):  60s")
    log.info("  v9.7:    SOT-accelerating (SOT increased last poll):   60s")
    log.info("  SOT >= 3:                                          90s")
    log.info("  SOT == 2 + fast window:                           120s")
    log.info("  SOT == 2:                                         180s")
    log.info("  SOT == 1:                                         base interval")
    log.info(f"  Active: DYNAMIC from daily schedule (fallback {ACTIVE_HOUR_START_FALLBACK}:00-{ACTIVE_HOUR_END_FALLBACK}:00)")
    log.info(f"  Team cache: {len(known_league_team_ids)} IDs ({len(PRESEEDED_TEAM_IDS)} pre-seeded)")
    log.info("=" * 60)

    # v10.4: Load ALL outcomes from JSONL (not just pending)
    global signal_outcomes
    all_loaded = load_all_outcomes()
    if all_loaded:
        signal_outcomes = all_loaded
        resolved_count = sum(1 for e in all_loaded if e.get("resolved"))
        pending_count = sum(1 for e in all_loaded if not e.get("resolved"))
        log.info(
            f"Loaded {len(all_loaded)} outcome(s) from {OUTCOMES_FILE} "
            f"({resolved_count} resolved, {pending_count} pending)"
        )

    # v10.5: Load form/H2H cache
    load_form_cache()

    # v10.4: Load signal dedup state (prevents duplicate signals on redeploy)
    load_signal_state()

    with httpx.Client(timeout=30.0) as client:
        # --- v10.1.2→v10.2.1: Verify small-league IDs against API (1 credit) ---
        # Prevents another ID-169-style disaster. Only checks IDs not in the
        # BIG_LEAGUE_IDS set which are universally correct.
        # v10.2.1 FIX: Batch in groups of 5 (API limit) and only REMOVE on
        # NAME MISMATCH (not on "not found" — API may just not return it).
        BIG_LEAGUE_IDS = {39, 140, 78, 79, 135, 61, 2, 3, 94, 88, 203, 848,
                               40, 144, 340}  # v10.2: added Championship, Belgium, Scotland
        verify_ids = [lid for lid in LEAGUE_IDS if lid not in BIG_LEAGUE_IDS]
        if verify_ids:
            try:
                all_api_names = {}
                # Batch in groups of 5 — API may not return all IDs in one call
                BATCH_VERIFY_SIZE = 5
                for i in range(0, len(verify_ids), BATCH_VERIFY_SIZE):
                    batch = verify_ids[i:i + BATCH_VERIFY_SIZE]
                    vresp = client.get(
                        f"{API_BASE}/leagues",
                        params={"ids": ",".join(str(x) for x in batch),
                                "current": "true"},
                        headers={"x-apisports-key": API_KEYS[0]},
                    )
                    update_quota(vresp)
                    if vresp.status_code == 200:
                        for lg in vresp.json().get("response", []):
                            all_api_names[lg["league"]["id"]] = lg["league"]["name"]

                mismatches = []
                not_found = []
                for lid in sorted(verify_ids):
                    expected = LEAGUE_IDS[lid]
                    actual = all_api_names.get(lid)
                    if actual is None:
                        # v10.2.1: Don't remove! API may just not return it.
                        # Only the ID-169 case (WRONG league) is dangerous.
                        not_found.append(lid)
                        log.info(f"  League ID {lid} '{expected}': not in /leagues response (kept — may still work in /fixtures)")
                    else:
                        expected_words = set(expected.lower().split())
                        actual_words = set(actual.lower().split())
                        overlap = expected_words & actual_words
                        if len(overlap) < min(2, len(expected_words)):
                            mismatches.append(f"  ID {lid}: expected '{expected}' but API says '{actual}' — REMOVED")
                        else:
                            log.info(f"  League ID OK: {lid} = {actual}")

                if mismatches:
                    log.warning("LEAGUE ID VERIFICATION — removing WRONG IDs:")
                    for m in mismatches:
                        log.warning(m)
                        try:
                            bad_id = int(m.split("ID ")[1].split(":")[0])
                            del LEAGUE_IDS[bad_id]
                        except (IndexError, ValueError):
                            pass
                    log.warning(f"  Remaining tracked leagues: {len(LEAGUE_IDS)}")
                if not_found:
                    log.info(f"  {len(not_found)} ID(s) not found in /leagues (kept): {not_found}")
                if not mismatches and not not_found:
                    log.info(f"  All {len(verify_ids)} small-league IDs verified OK")
            except Exception as e:
                log.warning(f"  League verification failed: {e}, skipping")
        else:
            log.info("  No small-league IDs to verify")
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
                            and quota_remaining <= 2):  # v10.2: was 10 (premium 7500/day)
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
                extra_info += f" | SOT-Accel: {', '.join(accel_parts[:3])}{'...' if len(accel_parts) > 3 else ''}"
            if pressure_accelerating:
                paccel_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in pressure_accelerating]
                extra_info += f" | GPS-Accel: {', '.join(paccel_parts[:3])}{'...' if len(paccel_parts) > 3 else ''}"
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
            if (resolved_outcomes and not pending_outcomes
                    and not has_tracked_live and not fast_monitored):
                log_outcome_summary()
                # v10.4: Send summary + data files to Telegram (once per day)
                today_str = time.strftime("%Y-%m-%d")
                if _daily_summary_sent_date != today_str:
                    send_daily_summary_telegram(client)
                    _daily_summary_sent_date = today_str
                # v10.4: Don't clear — reload from file next time we need summary.
                # Outcomes persist in JSONL. Clear memory only to free RAM,
                # but the data is never lost.
                signal_outcomes.clear()
                # v10.4: Clean up stale signal state file (all matches done)
                if signaled_teams:
                    save_signal_state()

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
