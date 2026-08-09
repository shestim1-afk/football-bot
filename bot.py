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

# Minute window for monitoring (pressure builds over time)
MINUTE_MIN = 25
MINUTE_MAX = 75

# --- Architecture ---
# SLOW DISCOVERY (cheap):  /fixtures?live=all every DISCOVERY_INTERVAL
# FAST MONITORING (expensive): /fixtures/statistics only for ranked candidates
#
# Pipeline:
#   /fixtures?live=all (1 request)
#        |
#        v
#   Tracked league?  --->  NO  ---> skip
#        |
#        YES
#        v
#   25-75 minutes?  --->  NO  ---> skip
#        |
#        YES
#        v
#   Interesting scoreline?  --->  NO  ---> skip
#        |
#        YES
#        v
#   Candidate (rank by free data)
#        |
#        v
#   Top N by rank (quota-aware)
#        |
#        v
#   GET /fixtures/statistics (1 request per fixture)
#        |
#        v
#   3+ SOT?  --->  NO  ---> update state, no signal
#        |
#        YES
#        v
#   SOT increased since last check?  --->  NO  ---> no duplicate signal
#        |
#        YES
#        v
#   CLASSIFY TIER --> Telegram signal
#        |
#        v
#   Continue monitoring after goals

# --- State ---
# Key: (fixture_id, team_id) -> {last_sot, last_minute, last_shots}
# Tracks SOT history so we only signal on INCREASES.
# NOT reset when a team scores.
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []

# --- Per-key quota tracking ---
# Initialize to unknown; will be set from first API response header.
# Do NOT assume each key = 100 requests independently.
quota_by_key: dict[str, int | None] = {k: None for k in API_KEYS}


def pick_key() -> str:
    """Pick the key with the most remaining quota (or unknown = optimistic)."""
    def sort_val(k):
        v = quota_by_key.get(k)
        return v if v is not None else 999  # unknown keys get priority to probe
    return max(quota_by_key, key=sort_val)


def total_quota() -> int:
    """Sum of known remaining quotas. Unknown keys counted as 0."""
    return sum(v for v in quota_by_key.values() if v is not None)


def has_unknown_quota() -> bool:
    return any(v is None for v in quota_by_key.values())


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
    This is NOT a 0-goals filter. It filters out blowouts where
    attacking pressure is irrelevant."""
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    # Filter out blowouts: if a team is up by 3+, skip
    if tg - og >= 3:
        return False
    # A team down by 4+ is unlikely to generate meaningful pressure
    if og - tg >= 4:
        return False

    return True


def rank_candidate(fixture: dict, team_id: int) -> int:
    """Score a candidate 0-10 using free data only.
    Higher = more likely to be generating real attacking pressure."""
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    # --- Time pressure: later in the game = more urgency ---
    if 65 <= minute <= 75: score += 3
    elif 55 <= minute <= 64: score += 2
    elif MINUTE_MIN <= minute <= 54: score += 1

    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    # --- Scoreline context ---
    if og > tg:
        # Losing team = maximum attacking urgency (especially by 1 goal)
        if og - tg == 1: score += 4  # trailing by 1, pushing hard
        else: score += 3  # trailing by more, still urgent
    elif tg == 0 and og == 0:
        score += 2  # 0-0: both sides may be pushing for opener
    elif tg > og:
        # Winning team: may be content or pressing for more
        if tg - og == 1: score += 2  # narrow lead, might press for insurance
        else: score += 1  # comfortable lead, less urgency
    else:
        score += 2  # drawing (1-1, 2-2, etc.) — both teams pushing

    # --- Prior pressure: if this team had SOT before, likely still generating ---
    fid = fixture["fixture"]["id"]
    prev = team_state.get((fid, team_id))
    if prev and prev.get("last_sot", 0) >= 2:
        score += 3  # already showed pressure = high probability of continued pressure

    return score


def get_score_context(fixture: dict, team_id: int) -> str:
    """Human-readable score context for the team."""
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
    """Emoji modifier based on scoreline context for signal strength."""
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    # Trailing by 1 with pressure = strongest signal (desperate, urgent)
    if og - tg == 1: return "🔥🔥"
    # Level game = strong signal (open match)
    if tg == og: return "🔥"
    # Leading by 1 = still pressing (good)
    if tg - og == 1: return "🔥"
    # Down 2 = urgent but might be overwhelmed
    if og - tg == 2: return "🔥"
    # Leading comfortably = lower priority (may be coasting)
    if tg - og >= 2: return "⚠️"
    # Down 3+ = cautious
    if og - tg >= 3: return "⚠️"
    return ""


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    q = total_quota()
    if has_unknown_quota():
        # First run — be conservative until we know actual quota
        return "PROBE"
    if q >= 70: return "NORMAL"
    elif q >= 40: return "CAREFUL"
    elif q >= 20: return "STRICT"
    elif q >= 5: return "EMERGENCY"
    else: return "STOP"


def get_max_candidates() -> int:
    """Max number of fixtures to request statistics for this cycle."""
    q = total_quota()
    if has_unknown_quota(): return 2  # conservative on first run
    if q >= 70: return 5
    elif q >= 40: return 3
    elif q >= 20: return 2
    elif q >= 5: return 1
    else: return 0


def get_discovery_interval() -> int:
    """Seconds between /fixtures?live=all scans.
    This is the CHEAP request — but still shouldn't be every 60s."""
    q = total_quota()
    if has_unknown_quota(): return 300  # 5 min while probing
    if q >= 70: return 300   # 5 min
    elif q >= 40: return 600  # 10 min
    elif q >= 20: return 900  # 15 min
    else: return 1800  # 30 min — almost out


def get_stats_check_interval(best_candidate_score: int) -> int:
    """After finding candidates, how long before next stats check?
    Faster for stronger candidates, but NEVER below 120s."""
    if best_candidate_score >= 8: return 120  # 2 min for very promising
    if best_candidate_score >= 6: return 180  # 3 min
    if best_candidate_score >= 4: return 300  # 5 min
    return 600  # 10 min for weak candidates


# ============================================================
# SIGNAL CLASSIFICATION
# Philosophy: detect sustained attacking PRESSURE.
# Score is CONTEXT, not a filter.
# SOT >= 3 is the hard floor. SOT must INCREASE for a new signal.
# ============================================================

def classify_signal(sot: int, total_shots: int, corners: int,
                     state: dict | None, current_minute: int) -> tuple[str | None, str]:
    """Return (tier, trend_string) or (None, "").
    Requirements:
      1. SOT >= 3 (hard floor — no noise from 1-2 SOT)
      2. SOT must have INCREASED since last check (no duplicate signals)
    """
    if sot < 3:
        return None, ""

    # SOT must be INCREASING (core concept: sustained/growing pressure)
    last_sot = state["last_sot"] if state else 0
    if sot <= last_sot:
        return None, ""  # No increase = no new signal

    # Calculate SOT growth rate for trend description
    trend = ""
    sot_rate = 0.0
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    # Classify tier
    if sot >= 5 and total_shots >= 10 and corners >= 3:
        return "VERY STRONG", trend
    if sot >= 4 and total_shots >= 8:
        if sot_rate >= 0.2:
            return "VERY STRONG", trend  # rapid SOT growth
        return "STRONG", trend
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
# Pipeline: tracked league -> 25-75' -> interesting scoreline -> rank
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[dict, int, int]]:
    """Return ranked list of (fixture, team_id, rank_score).
    Both teams per match are evaluated. Scoreline is context, not a filter."""
    candidates = []
    for fixture in fixtures:
        lid = fixture["league"]["id"]
        if lid not in LEAGUE_IDS:
            continue
        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue
        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
        if minute < MINUTE_MIN or minute > MINUTE_MAX:
            continue

        home_tid = fixture["teams"]["home"]["id"]
        away_tid = fixture["teams"]["away"]["id"]

        # Check scoreline interest for each team
        if is_interesting_scoreline(fixture, home_tid):
            candidates.append((fixture, home_tid, rank_candidate(fixture, home_tid)))
        if is_interesting_scoreline(fixture, away_tid):
            candidates.append((fixture, away_tid, rank_candidate(fixture, away_tid)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def cleanup_state(live_fixture_ids: set[int]):
    """Remove state for fixtures that are no longer live."""
    to_delete = [k for k in team_state if k[0] not in live_fixture_ids]
    for k in to_delete:
        del team_state[k]
    if to_delete:
        log.info(f"  Cleaned up state for {len(to_delete)} ended fixture(s)")


# ============================================================
# MAIN CHECK CYCLE
# ============================================================

def check_cycle(client: httpx.Client) -> int:
    """Run one discovery + selective monitoring cycle.
    Returns the best candidate score (for dynamic interval)."""
    budget = get_budget_mode()

    # --- STEP 1: Discovery (1 request) ---
    try:
        data = api_get(client, "/fixtures", {"live": "all"})
    except Exception as e:
        log.error(f"Discovery request failed: {e}")
        return 0

    fixtures = data.get("response", [])
    tracked = [f for f in fixtures if f["league"]["id"] in LEAGUE_IDS]
    q = total_quota()
    max_cand = get_max_candidates()

    log.info(f"Quota: {q} [{budget}] | Max stats checks: {max_cand} | "
             f"Live: {len(fixtures)} | Tracked: {len(tracked)} | Requests: {request_count}")
    for k, v in quota_by_key.items():
        label = f"{v}" if v is not None else "unknown"
        log.info(f"  Key {k[:8]}...: {label} remaining")

    if budget == "STOP":
        log.warning("Quota exhausted. Skipping stats checks.")
        return 0

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(f"  -> {m['league']['name']}: {m['teams']['home']['name']} vs "
                     f"{m['teams']['away']['name']} ({m['fixture']['status']['short']} {minute}')")

    # Cleanup state for ended fixtures
    live_ids = {f["fixture"]["id"] for f in fixtures}
    cleanup_state(live_ids)

    # --- STEP 2: Local pre-filter (free) ---
    candidates = find_candidates(fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) after local filter ({MINUTE_MIN}-{MINUTE_MAX}')")

    if not candidates:
        return 0

    # --- STEP 3: Quota-aware selection ---
    # Deduplicate by fixture (one stats request per fixture covers both teams)
    fixture_team_map: dict[int, list[tuple[int, int]]] = {}  # fid -> [(team_id, rank)]
    for fixture, tid, rank in candidates:
        fid = fixture["fixture"]["id"]
        fixture_team_map.setdefault(fid, []).append((tid, rank))

    # Sort fixtures by their best team rank
    fixture_ranks = []
    for fid, teams in fixture_team_map.items():
        best_rank = max(r for _, r in teams)
        fixture_ranks.append((fid, best_rank, teams))
    fixture_ranks.sort(key=lambda x: x[1], reverse=True)

    # Select top N fixtures
    selected_fixtures = fixture_ranks[:max_cand]
    log.info(f"  -> Selected {len(selected_fixtures)} fixture(s) for stats check "
             f"(from {len(fixture_team_map)} eligible)")

    best_score = selected_fixtures[0][1] if selected_fixtures else 0

    # --- STEP 4: Get statistics ONLY for selected fixtures ---
    for fid, _, team_entries in selected_fixtures:
        if total_quota() <= 2:
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

        # Find the fixture object for context
        fixture = None
        for f in fixtures:
            if f["fixture"]["id"] == fid:
                fixture = f
                break
        if not fixture:
            continue

        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

        # Parse all team stats from response
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

        # Check each team that was a candidate for this fixture
        for tid, _ in team_entries:
            # Find team name
            tname = None
            if fixture["teams"]["home"]["id"] == tid:
                tname = fixture["teams"]["home"]["name"]
            elif fixture["teams"]["away"]["id"] == tid:
                tname = fixture["teams"]["away"]["name"]
            if not tname or tname not in teams_data:
                continue

            tstats = teams_data[tname]
            sot_raw = tstats.get("Shots on Goal", "0")
            shots_raw = tstats.get("Total Shots", "0")
            corners_raw = tstats.get("Corner Kicks", "0")
            try:
                sot = int(sot_raw)
                total_shots = int(shots_raw)
                corners = int(corners_raw)
            except (ValueError, TypeError):
                continue

            # --- SIGNAL LOGIC ---
            state = team_state.get((fid, tid))
            tier, trend = classify_signal(sot, total_shots, corners, state, minute)

            if tier:
                # Build signal message
                ctx = get_score_context(fixture, tid)
                score_emoji = get_score_emoji(fixture, tid)
                possession = tstats.get("Ball Possession", "N/A")
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

                if send_telegram(client, msg):
                    log.info(f"SIGNAL {tier}: {tname} ({ctx}) - "
                             f"{sot} SOT, {total_shots} shots, {corners} corners (fixture {fid})")

                # ALWAYS update state after a signal (whether send succeeds or not)
                team_state[(fid, tid)] = {
                    "last_sot": sot,
                    "last_minute": minute,
                    "last_shots": total_shots,
                }

                signals_sent.append({
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "fixture": fid, "team": tname, "league": league,
                    "minute": minute, "sot": sot, "total_shots": total_shots,
                    "corners": corners, "context": ctx, "tier": tier,
                    "trend": trend,
                })
            else:
                # No signal, but update state for SOT tracking
                # (so next cycle can detect an increase)
                team_state[(fid, tid)] = {
                    "last_sot": sot,
                    "last_minute": minute,
                    "last_shots": total_shots,
                }

    return best_score


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v5 — Sustained Pressure Detection")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (quota tracked from API headers)")
    log.info("")
    log.info("Architecture: slow discovery + selective statistics")
    log.info("  /fixtures?live=all  -> every 5-30 min (quota-dependent)")
    log.info("  /fixtures/statistics -> only for top-ranked candidates")
    log.info("")
    log.info("Signal logic:")
    log.info("  Trigger: 3+ SOT (hard floor)")
    log.info("  Dedup: SOT must INCREASE since last check")
    log.info("  Score: CONTEXT, not filter (0-0, 0-1, 1-0, 1-1, 2-1 all valid)")
    log.info("  Post-goal: monitoring CONTINUES (no reset)")
    log.info("  Blowout filter: skip if team is up 3+ or down 4+")
    log.info("")
    log.info("Tiers:")
    log.info("  PRESSURE    = 3+ SOT")
    log.info("  STRONG      = 4+ SOT, 8+ shots")
    log.info("  VERY STRONG = 5+ SOT, 10+ shots, 3+ corners (or rapid SOT growth)")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        while True:
            try:
                best_score = check_cycle(client)
            except Exception as e:
                log.error(f"Cycle error: {e}")
                best_score = 0

            # Determine next interval
            if best_score > 0:
                # We have active candidates — check stats sooner
                interval = get_stats_check_interval(best_score)
            else:
                # No interesting matches — slow discovery
                interval = get_discovery_interval()

            log.info(f"Next check in {interval}s ({interval // 60}m) | "
                     f"Best score: {best_score} | Quota: {total_quota()} | "
                     f"Total requests: {request_count} | Signals sent: {len(signals_sent)}")
            time.sleep(interval)


if __name__ == "__main__":
    main()
