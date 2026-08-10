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

MINUTE_MIN = 25
MINUTE_MAX = 80
MINUTE_LATE_MAX = 90

# --- State ---
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []

# --- Per-key quota tracking ---
quota_by_key: dict[str, int | None] = {k: None for k in API_KEYS}


def pick_key() -> str:
    def sort_val(k):
        v = quota_by_key.get(k)
        return v if v is not None else 999
    return max(quota_by_key, key=sort_val)


def total_quota() -> int:
    return sum(v for v in quota_by_key.values() if v is not None)


def unknown_key_count() -> int:
    return sum(1 for v in quota_by_key.values() if v is None)


def effective_quota() -> int:
    known = total_quota()
    unknown_bonus = unknown_key_count() * 10
    return min(known + unknown_bonus, 150)


def update_key_quota(resp: httpx.Response, key: str):
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
# CANDIDATE RANKING (free data only)
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

    if 65 <= minute <= MINUTE_MAX: score += 3
    elif 55 <= minute <= 64: score += 2
    elif MINUTE_MIN <= minute <= 54: score += 1

    tg = get_team_goals(fixture, team_id)
    is_home = fixture["teams"]["home"]["id"] == team_id
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if og > tg:
        if og - tg == 1: score += 4
        else: score += 3
    elif tg == 0 and og == 0:
        score += 2
    elif tg > og:
        if tg - og == 1: score += 2
        else: score += 1
    else:
        score += 2

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
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return "PROBE"
    if eq >= 70: return "NORMAL"
    elif eq >= 40: return "CAREFUL"
    elif eq >= 20: return "STRICT"
    elif eq >= 5: return "EMERGENCY"
    else: return "STOP"


def get_max_candidates() -> int:
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return 2
    if eq >= 70: return 5
    elif eq >= 40: return 3
    elif eq >= 20: return 2
    elif eq >= 5: return 1
    else: return 0


def get_cycle_interval(has_candidates: bool, has_live: bool) -> int:
    eq = effective_quota()
    if not has_live:
        return 1800
    if not has_candidates:
        if eq >= 40: return 300
        else: return 600
    if eq >= 40: return 180
    elif eq >= 20: return 300
    else: return 600


# ============================================================
# SIGNAL CLASSIFICATION — SOT is king
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


# ============================================================
# MAIN CYCLE
# ============================================================

def check_cycle(client: httpx.Client) -> tuple[bool, bool, int]:
    try:
        data = api_get(client, "/fixtures", {"live": "all"})
    except Exception as e:
        log.error(f"Discovery request failed: {e}")
        return False, False, 0

    fixtures = data.get("response", [])
    tracked = [f for f in fixtures if f["league"]["id"] in LEAGUE_IDS]
    has_any_live = len(fixtures) > 0
    eq = effective_quota()
    budget = get_budget_mode()

    log.info(f"Quota: {total_quota()} [{budget}] | "
             f"Live: {len(fixtures)} | Tracked: {len(tracked)} | Requests: {request_count}")

    if budget == "STOP":
        log.warning("Quota exhausted.")
        return False, has_any_live, 0

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(f"  -> {m['league']['name']}: {m['teams']['home']['name']} vs "
                     f"{m['teams']['away']['name']} ({m['fixture']['status']['short']} {minute}')")

    live_ids = {f["fixture"]["id"] for f in fixtures}
    cleanup_state(live_ids)

    candidates = find_candidates(fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN}-{MINUTE_MAX}')")

    if not candidates:
        return False, has_any_live, 0

    max_cand = get_max_candidates()

    fixture_team_map: dict[int, list[tuple[int, int]]] = {}
    for fid, tid, rank in candidates:
        fixture_team_map.setdefault(fid, []).append((tid, rank))

    fixture_ranks = []
    for fid, teams in fixture_team_map.items():
        best_rank = max(r for _, r in teams)
        fixture_ranks.append((fid, best_rank, teams))
    fixture_ranks.sort(key=lambda x: x[1], reverse=True)

    selected = fixture_ranks[:max_cand]
    log.info(f"  -> Checking {len(selected)} fixture(s) for stats")
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

        fixture = None
        for f in fixtures:
            if f["fixture"]["id"] == fid:
                fixture = f
                break
        if not fixture:
            continue

        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

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

            state = team_state.get((fid, tid))
            tier, trend, sot_rate = classify_signal(sot, state, minute)

            current_goals = get_team_goals(fixture, tid)
            prev_goals = state.get("last_goals", current_goals) if state else current_goals
            scored_since_last = current_goals > prev_goals

            team_state[(fid, tid)] = {
                "last_sot": sot,
                "last_minute": minute,
                "last_goals": current_goals,
            }

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
                    msg += f"  \u26bd Scored since last signal — pressure continues\n"

                if send_telegram(client, msg):
                    log.info(f"SIGNAL {tier}: {tname} ({ctx}) - "
                             f"{sot} SOT, {total_shots} shots (fixture {fid})")

                signals_sent.append({
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "fixture": fid, "team": tname, "league": league,
                    "minute": minute, "sot": sot, "total_shots": total_shots,
                    "corners": corners, "context": ctx, "tier": tier,
                    "trend": trend, "scored_since_last": scored_since_last,
                })

    return len(candidates) > 0, has_any_live, best_score


def main():
    log.info("=" * 60)
    log.info("Football Bot v7 — SOT is King")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (quota from API headers)")
    log.info("")
    log.info("Architecture: unified cycle, always fresh data")
    log.info("  Every cycle: /fixtures?live=all (1 req) + selective stats")
    log.info("  0 live: 30 min | No candidates: 5 min | Has candidates: 3 min")
    log.info("")
    log.info("Signal logic:")
    log.info("  Trigger: 3+ SOT AND SOT increased since last check")
    log.info("  No goal restriction. Score is context only.")
    log.info("  Post-goal: monitoring CONTINUES")
    log.info("")
    log.info("Tier = SOT count:")
    log.info("  3 SOT    = PRESSURE (yellow)")
    log.info("  4 SOT    = STRONG (orange)")
    log.info("  5+ SOT   = VERY STRONG (red)")
    log.info("  Rapid SOT growth (>= 0.3/min) bumps tier up by 1")
    log.info("")
    log.info("Context stats (in message, NOT tier drivers):")
    log.info("  Possession, corners, total shots, opponent SOT, scoreline")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        while True:
            try:
                has_candidates, has_live, best_score = check_cycle(client)
            except Exception as e:
                log.error(f"Cycle error: {e}")
                has_candidates, has_live, best_score = False, False, 0

            interval = get_cycle_interval(has_candidates, has_live)
            log.info(f"Next in {interval}s ({interval // 60}m) | "
                     f"Quota: {total_quota()} | Reqs: {request_count} | "
                     f"Signals: {len(signals_sent)}")
            time.sleep(interval)


if __name__ == "__main__":
    main()