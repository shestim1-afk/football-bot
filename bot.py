import os
import sys
import time
import logging
import httpx
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
if not TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_CHAT_ID: missing.append("TELEGRAM_CHAT_ID")

_raw_keys = os.environ.get("RAPIDAPI_KEY", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not API_KEYS: missing.append("RAPIDAPI_KEY")

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

# Minute windows
# Normal monitoring: 25-80' (pressure builds over time, late pressure matters)
# Extended monitoring: 80-90' — only for fixtures already being tracked (known candidates)
MINUTE_MIN = 25
MINUTE_MAX = 80
MINUTE_LATE_MAX = 90  # only for fixtures already in team_state

# --- Architecture v6 ---
# TWO-LOOP DESIGN:
#
#   DISCOVERY LOOP (slow, cheap)
#     /fixtures?live=all  (1 request)
#           |
#           v
#     Tracked league? 25-80'? Interesting scoreline?
#           |
#           v
#     Update active_fixtures + candidates list
#     Sleep 5-10 min (quota-dependent)
#
#   MONITORING LOOP (fast, expensive)
#     For each active candidate fixture:
#       /fixtures/statistics  (1 request per fixture)
#           |
#           v
#       3+ SOT? SOT increased?
#           |
#           v
#       Calculate pressure score -> classify tier
#           |
#           v
#       Telegram signal
#     Sleep 2-5 min (based on strongest candidate)
#
# The key insight: monitoring cycles do NOT call /fixtures?live=all.
# Discovery happens independently. Monitoring uses cached fixture data
# and only requests statistics.

# --- State ---
# Key: (fixture_id, team_id) -> {
#   last_sot, last_minute, last_shots, last_goals,
#   fixture: <cached fixture dict for this team>
# }
# Tracks SOT history so we only signal on INCREASES.
# NOT reset when a team scores. Goals tracked to detect inter-signal scoring.
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []

# Cached fixtures from last discovery (used by monitoring loop)
# Key: fixture_id -> fixture dict
cached_fixtures: dict[int, dict] = {}

# Active candidates from last discovery
# List of (fixture_id, team_id, rank_score)
active_candidates: list[tuple[int, int, int]] = []

# --- Per-key quota tracking ---
# Initialize to None; set from first API response header.
# Do NOT assume each key = 100 requests independently.
quota_by_key: dict[str, int | None] = {k: None for k in API_KEYS}


def pick_key() -> str:
    """Pick the key with the most remaining quota.
    Unknown keys get priority (to probe their actual quota)."""
    def sort_val(k):
        v = quota_by_key.get(k)
        return v if v is not None else 999
    return max(quota_by_key, key=sort_val)


def total_quota() -> int:
    """Sum of known remaining quotas. Unknown keys NOT counted
    (they're available for probing but we don't assume a value)."""
    return sum(v for v in quota_by_key.values() if v is not None)


def unknown_key_count() -> int:
    return sum(1 for v in quota_by_key.values() if v is None)


def effective_quota() -> int:
    """Quota for budget decisions. Includes unknown keys as a
    conservative estimate (+10 each) so we don't under-utilize them.
    Capped at a safety limit to prevent runaway spending."""
    known = total_quota()
    unknown_bonus = unknown_key_count() * 10
    return min(known + unknown_bonus, 150)  # hard safety cap


def update_key_quota(resp: httpx.Response, key: str):
    """Read actual quota from API response header."""
    try:
        val = resp.headers.get("x-ratelimit-requests-remaining", "")
        if val:
            new_val = int(val)
            old_val = quota_by_key.get(key)
            quota_by_key[key] = new_val
            if old_val is not None and new_val != old_val:
                log.info(f"  Key {key[:8]}... quota: {old_val} -> {new_val}")
            elif old_val is None:
                log.info(f"  Key {key[:8]}... quota initialized: {new_val}")
    except (ValueError, TypeError):
        pass


# ============================================================
# API HELPERS
# ============================================================

def api_get(client: httpx.Client, endpoint: str, params: dict = None) -> dict:
    global request_count
    key = pick_key()
    request_count += 1
    log.info(f"  API GET {endpoint} (key={key[:8]}..., req #{request_count})")
    resp = client.get(
        f"{API_BASE}{endpoint}",
        params=params,
        headers={"x-apisports-key": key},
    )
    resp.raise_for_status()
    update_key_quota(resp, key)
    return resp.json()


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
    """Check if the scoreline makes this team worth monitoring.
    NOT a 0-goals filter. Only filters out blowouts."""
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    # Up by 3+ = likely coasting, skip
    if tg - og >= 3:
        return False
    # Down by 4+ = probably collapsed, skip
    if og - tg >= 4:
        return False

    return True


def get_team_goals(fixture: dict, team_id: int) -> int:
    is_home = fixture["teams"]["home"]["id"] == team_id
    return (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0


def rank_candidate(fixture: dict, team_id: int) -> int:
    """Score a candidate 0-10 using free data only.
    Higher = more likely to be generating real attacking pressure."""
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    # Time pressure: later = more urgency
    if 65 <= minute <= MINUTE_MAX: score += 3
    elif 55 <= minute <= 64: score += 2
    elif MINUTE_MIN <= minute <= 54: score += 1

    tg = get_team_goals(fixture, team_id)
    og_home = fixture["goals"]["home"] or 0
    og_away = fixture["goals"]["away"] or 0
    is_home = fixture["teams"]["home"]["id"] == team_id
    og = og_away if is_home else og_home

    # Scoreline context
    if og > tg:
        if og - tg == 1: score += 4  # trailing by 1 = pushing hard
        else: score += 3  # trailing by more = still urgent
    elif tg == 0 and og == 0:
        score += 2  # 0-0: both pushing for opener
    elif tg > og:
        if tg - og == 1: score += 2  # narrow lead, pressing for insurance
        else: score += 1  # comfortable, less urgency
    else:
        score += 2  # drawing (1-1, 2-2)

    # Prior pressure: had SOT before = likely still generating
    fid = fixture["fixture"]["id"]
    prev = team_state.get((fid, team_id))
    if prev and prev.get("last_sot", 0) >= 2:
        score += 3

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

    if og - tg == 1: return "🔥🔥"  # trailing by 1 = desperate urgency
    if tg == og: return "🔥"       # open match
    if tg - og == 1: return "🔥"   # pressing for insurance
    if og - tg == 2: return "🔥"   # urgent comeback attempt
    if tg - og >= 2: return "⚠️"   # comfortable, may coast
    if og - tg >= 3: return "⚠️"   # probably overwhelmed
    return ""


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return "PROBE"  # haven't probed any key yet
    if eq >= 70: return "NORMAL"
    elif eq >= 40: return "CAREFUL"
    elif eq >= 20: return "STRICT"
    elif eq >= 5: return "EMERGENCY"
    else: return "STOP"


def get_max_candidates() -> int:
    """Max fixtures to request statistics for per monitoring cycle."""
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return 2  # probing mode: conservative
    if eq >= 70: return 5
    elif eq >= 40: return 3
    elif eq >= 20: return 2
    elif eq >= 5: return 1
    else: return 0


def get_discovery_interval() -> int:
    """Seconds between /fixtures?live=all discovery scans."""
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return 300  # 5 min while probing
    if eq >= 70: return 300   # 5 min
    elif eq >= 40: return 600  # 10 min
    elif eq >= 20: return 900  # 15 min
    else: return 1800  # 30 min


def get_monitoring_interval(best_score: int) -> int:
    """Seconds between monitoring cycles (stats-only, no discovery).
    Only called when there ARE active candidates."""
    if best_score >= 8: return 120  # 2 min
    if best_score >= 6: return 180  # 3 min
    if best_score >= 4: return 300  # 5 min
    return 420  # 7 min


# ============================================================
# SIGNAL CLASSIFICATION — Pressure Scoring System
# ============================================================
# Score-based, not rigid conditions. Each stat adds points.
#
#   +3   5+ SOT
#   +2   4 SOT
#   +2   10+ total shots
#   +1   8+ total shots
#   +2   SOT increased since last check
#   +2   rapid SOT growth (>= 0.2 per minute)
#   +1   55%+ possession
#   +1   3+ corners
#
# Tiers:
#   0-4   PRESSURE    (yellow)
#   5-7   STRONG      (orange)
#   8+    VERY STRONG (red)
# ============================================================

def calc_pressure_score(sot: int, total_shots: int, corners: int,
                          possession_pct: float, state: dict | None,
                          current_minute: int) -> tuple[int, float, str]:
    """Calculate pressure score and return (score, sot_rate, trend_string).
    Does NOT check SOT >= 3 or SOT increase here — caller handles that."""
    score = 0
    sot_rate = 0.0
    trend = ""

    # SOT volume
    if sot >= 5: score += 3
    elif sot >= 4: score += 2

    # Total shots
    if total_shots >= 10: score += 2
    elif total_shots >= 8: score += 1

    # SOT increase (already verified by caller, but add points for it)
    if state and state.get("last_sot", 0) > 0:
        score += 2  # confirmed increase

    # SOT growth rate
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        if sot_rate >= 0.2:
            score += 2  # rapid growth
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    # Possession
    if possession_pct >= 55:
        score += 1

    # Corners
    if corners >= 3:
        score += 1

    return score, sot_rate, trend


def classify_signal(sot: int, total_shots: int, corners: int,
                     possession_raw: str, state: dict | None,
                     current_minute: int) -> tuple[str | None, str]:
    """Return (tier, trend_string) or (None, "").
    Requirements:
      1. SOT >= 3 (hard floor)
      2. SOT must have INCREASED since last check (dedup)"""
    if sot < 3:
        return None, ""

    last_sot = state["last_sot"] if state else 0
    if sot <= last_sot:
        return None, ""  # no increase = no new signal

    # Parse possession
    try:
        possession_pct = float(possession_raw.replace("%", ""))
    except (ValueError, TypeError):
        possession_pct = 50.0

    score, sot_rate, trend = calc_pressure_score(
        sot, total_shots, corners, possession_pct, state, current_minute
    )

    if score >= 8:
        return "VERY STRONG", trend
    elif score >= 5:
        return "STRONG", trend
    else:
        return "PRESSURE", trend


def tier_emoji(tier: str) -> str:
    if tier == "VERY STRONG": return "🔴"
    if tier == "STRONG": return "🟠"
    return "🟡"


def tier_label(tier: str) -> str:
    if tier == "VERY STRONG": return "[RED]"
    if tier == "STRONG": return "[ORANGE]"
    return "[YELLOW]"


# ============================================================
# LOCAL FILTERING (zero API cost)
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[int, int, int]]:
    """Return ranked list of (fixture_id, team_id, rank_score).
    Scoreline is context, not a filter."""
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

        # Minute window: 25-80 normally, 80-90 only if already tracked
        in_window = False
        if MINUTE_MIN <= minute <= MINUTE_MAX:
            in_window = True
        elif MINUTE_MAX < minute <= MINUTE_LATE_MAX:
            # Late window: only if this team was already being monitored
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

    # Also clean cached fixtures
    for fid in list(cached_fixtures.keys()):
        if fid not in live_fixture_ids:
            del cached_fixtures[fid]


# ============================================================
# DISCOVERY LOOP (slow, cheap — 1 request)
# ============================================================

def run_discovery(client: httpx.Client) -> bool:
    """Fetch all live fixtures, filter, rank, update active candidates.
    Returns True if there are candidates to monitor."""
    global active_candidates

    try:
        data = api_get(client, "/fixtures", {"live": "all"})
    except Exception as e:
        log.error(f"Discovery request failed: {e}")
        return False

    fixtures = data.get("response", [])
    tracked = [f for f in fixtures if f["league"]["id"] in LEAGUE_IDS]

    # Update cached fixtures
    cached_fixtures.clear()
    for f in fixtures:
        cached_fixtures[f["fixture"]["id"]] = f

    eq = effective_quota()
    budget = get_budget_mode()
    log.info(f"--- DISCOVERY ---")
    log.info(f"Quota: known={total_quota()} effective={eq} [{budget}] | "
             f"Live: {len(fixtures)} | Tracked: {len(tracked)} | Requests: {request_count}")
    for k, v in quota_by_key.items():
        label = f"{v}" if v is not None else "unknown"
        log.info(f"  Key {k[:8]}...: {label} remaining")

    if budget == "STOP":
        log.warning("Quota exhausted.")
        active_candidates = []
        return False

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(f"  -> {m['league']['name']}: {m['teams']['home']['name']} vs "
                     f"{m['teams']['away']['name']} ({m['fixture']['status']['short']} {minute}')")

    # Cleanup state for ended fixtures
    live_ids = {f["fixture"]["id"] for f in fixtures}
    cleanup_state(live_ids)

    # Find and rank candidates (local filter, zero API cost)
    active_candidates = find_candidates(fixtures)
    log.info(f"  -> {len(active_candidates)} team-candidate(s) ({MINUTE_MIN}-{MINUTE_MAX}')")

    return len(active_candidates) > 0


# ============================================================
# MONITORING LOOP (fast, expensive — 1 request per fixture)
# Does NOT call /fixtures?live=all. Uses cached fixtures.
# ============================================================

def run_monitoring(client: httpx.Client) -> int:
    """Check statistics for active candidates only.
    Returns the best candidate rank score."""
    if not active_candidates:
        return 0

    max_cand = get_max_candidates()
    eq = effective_quota()

    # Deduplicate by fixture
    fixture_team_map: dict[int, list[tuple[int, int]]] = {}
    for fid, tid, rank in active_candidates:
        fixture_team_map.setdefault(fid, []).append((tid, rank))

    # Sort fixtures by best team rank
    fixture_ranks = []
    for fid, teams in fixture_team_map.items():
        best_rank = max(r for _, r in teams)
        fixture_ranks.append((fid, best_rank, teams))
    fixture_ranks.sort(key=lambda x: x[1], reverse=True)

    selected = fixture_ranks[:max_cand]
    log.info(f"--- MONITORING ---")
    log.info(f"Quota: effective={eq} | Checking {len(selected)} fixture(s) "
             f"from {len(fixture_team_map)} eligible | Requests: {request_count}")

    best_score = selected[0][1] if selected else 0

    for fid, _, team_entries in selected:
        if effective_quota() <= 3:
            log.warning("Quota nearly gone, stopping stats fetches.")
            break

        try:
            stats_data = api_get(client, "/fixtures/statistics", {"fixture": fid})
        except Exception as e:
            log.warning(f"Stats failed for fixture {fid}: {e}")
            continue

        stats = stats_data.get("response", [])
        if not stats:
            continue

        # Get fixture from cache (no API call)
        fixture = cached_fixtures.get(fid)
        if not fixture:
            continue

        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

        # Parse team stats
        teams_data = {}
        for team_entry in stats:
            tname = team_entry["team"]["name"]
            tmap = {}
            for s in team_entry.get("statistics", []):
                val = s.get("value", "0")
                if val is None:
                    val = "0"
                tmap[s["type"]] = str(val).strip()
            teams_data[tname] = tmap

        for tid, _ in team_entries:
            tname = None
            if fixture["teams"]["home"]["id"] == tid:
                tname = fixture["teams"]["home"]["name"]
            elif fixture["teams"]["away"]["id"] == tid:
                tname = fixture["teams"]["away"]["name"]
            if not tname or tname not in teams_data:
                continue

            tstats = teams_data[tname]
            try:
                sot = int(tstats.get("Shots on Goal", "0"))
                total_shots = int(tstats.get("Total Shots", "0"))
                corners = int(tstats.get("Corner Kicks", "0"))
                possession = tstats.get("Ball Possession", "50%")
            except (ValueError, TypeError):
                continue

            # --- SIGNAL LOGIC ---
            state = team_state.get((fid, tid))
            tier, trend = classify_signal(sot, total_shots, corners, possession, state, minute)

            # Detect if team scored since last signal
            current_goals = get_team_goals(fixture, tid)
            prev_goals = state.get("last_goals", current_goals) if state else current_goals
            scored_since_last = current_goals > prev_goals

            if tier:
                ctx = get_score_context(fixture, tid)
                score_emoji = get_score_emoji(fixture, tid)
                opp_sot = "?"
                for oname, ostats in teams_data.items():
                    if oname != tname:
                        opp_sot = ostats.get("Shots on Goal", "?")
                        break

                league = LEAGUE_IDS.get(fixture["league"]["id"], fixture["league"]["name"])
                home = fixture["teams"]["home"]["name"]
                away = fixture["teams"]["away"]["name"]
                sh = fixture["goals"]["home"]
                sa = fixture["goals"]["away"]

                msg = (
                    f"{tier_emoji(tier)} {tier_label(tier)} {tier} GOAL PRESSURE {score_emoji}\n\n"
                    f"{home}  {sh} - {sa}  {away}\n"
                    f"{league}  {minute}'\n\n"
                    f"{tname} ({ctx})\n"
                    f"  Shots on target: {sot}\n"
                    f"  Total shots: {total_shots}\n"
                    f"  Opponent SOT: {opp_sot}\n"
                    f"  Possession: {possession}\n"
                    f"  Corners: {corners}\n"
                )
                if trend:
                    msg += f"  Trend: {trend}\n"
                if scored_since_last:
                    msg += f"  ⚽ Scored since last signal — pressure continues\n"

                if send_telegram(client, msg):
                    log.info(f"SIGNAL {tier}: {tname} ({ctx}) - "
                             f"{sot} SOT, {total_shots} shots, {corners} corners (fixture {fid})")

                # ALWAYS update state
                team_state[(fid, tid)] = {
                    "last_sot": sot,
                    "last_minute": minute,
                    "last_shots": total_shots,
                    "last_goals": current_goals,
                }

                signals_sent.append({
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "fixture": fid, "team": tname, "league": league,
                    "minute": minute, "sot": sot, "total_shots": total_shots,
                    "corners": corners, "context": ctx, "tier": tier,
                    "trend": trend, "scored_since_last": scored_since_last,
                })
            else:
                # No signal, but update state for SOT tracking + goal tracking
                team_state[(fid, tid)] = {
                    "last_sot": sot,
                    "last_minute": minute,
                    "last_shots": total_shots,
                    "last_goals": current_goals,
                }

    return best_score


# ============================================================
# MAIN LOOP — Two-loop architecture
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v6 — Two-Loop Pressure Detection")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (quota from API headers, not assumed)")
    log.info("")
    log.info("Architecture:")
    log.info("  DISCOVERY  /fixtures?live=all  -> every 5-30 min (1 req)")
    log.info("  MONITORING /fixtures/statistics -> 2-7 min (1 req per fixture)")
    log.info("  Key: monitoring does NOT re-fetch live fixtures")
    log.info("")
    log.info("Signal logic:")
    log.info("  Trigger: 3+ SOT (hard floor)")
    log.info("  Dedup: SOT must INCREASE since last check")
    log.info("  Score: CONTEXT, not filter")
    log.info("  Post-goal: monitoring CONTINUES, goal noted in signal")
    log.info("  Minute window: 25-80' (80-90' for already-tracked teams)")
    log.info("")
    log.info("Pressure scoring (flexible, not rigid):")
    log.info("  +3  5+ SOT  |  +2  4 SOT  |  +2  10+ shots  |  +1  8+ shots")
    log.info("  +2  SOT increased  |  +2  rapid SOT growth  |  +1  55%+ poss")
    log.info("  +1  3+ corners")
    log.info("  0-4=PRESSURE  5-7=STRONG  8+=VERY STRONG")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        has_candidates = False
        last_discovery = 0.0

        while True:
            now = time.time()
            discovery_interval = get_discovery_interval()
            monitoring_interval = get_monitoring_interval(
                active_candidates[0][2] if active_candidates else 0
            ) if has_candidates else 0

            # --- Decide: discover or monitor? ---
            should_discover = (
                (now - last_discovery) >= discovery_interval
                or not has_candidates
            )

            if should_discover:
                try:
                    has_candidates = run_discovery(client)
                except Exception as e:
                    log.error(f"Discovery cycle error: {e}")
                    has_candidates = False
                last_discovery = time.time()
                sleep_time = monitoring_interval if has_candidates else discovery_interval
            else:
                # --- Monitoring cycle (no discovery request) ---
                try:
                    best_score = run_monitoring(client)
                    if best_score > 0:
                        sleep_time = get_monitoring_interval(best_score)
                    else:
                        # Candidates went cold — force next cycle to rediscover
                        has_candidates = False
                        sleep_time = 60  # quick re-discover
                except Exception as e:
                    log.error(f"Monitoring cycle error: {e}")
                    sleep_time = discovery_interval

            log.info(f"Next: {'DISCOVER' if not has_candidates else 'MONITOR'} "
                     f"in {sleep_time}s ({sleep_time // 60}m) | "
                     f"Quota: {total_quota()} | Reqs: {request_count} | "
                     f"Signals: {len(signals_sent)}")
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
