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
MINUTE_MAX = 75

# --- State ---
notified: dict[tuple[int, int], int] = {}
request_count = 0
signals_sent: list[dict] = []

# --- Per-key quota tracking ---
quota_by_key: dict[str, int] = {k: 100 for k in API_KEYS}
rate_limit_remaining: int = 10


# ============================================================
# API HELPERS
# ============================================================

def pick_key() -> str:
    return max(quota_by_key, key=quota_by_key.get)


def total_quota() -> int:
    return sum(quota_by_key.values())


def update_key_quota(resp: httpx.Response, key: str):
    global rate_limit_remaining
    try:
        val = resp.headers.get("x-ratelimit-requests-remaining", "")
        if val:
            quota_by_key[key] = int(val)
    except (ValueError, TypeError):
        pass
    try:
        val = resp.headers.get("X-RateLimit-Remaining", "")
        if val:
            rate_limit_remaining = int(val)
    except (ValueError, TypeError):
        pass


def api_get(client: httpx.Client, endpoint: str, params: dict = None) -> dict:
    global request_count
    if total_quota() <= 0:
        raise Exception("Daily quota exhausted")
    key = pick_key()
    request_count += 1
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
# CANDIDATE RANKING (uses only FREE data)
# ============================================================

def rank_candidate(fixture: dict, team_id: int) -> int:
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    if 60 <= minute <= 75:
        score += 3
    elif 45 <= minute <= 59:
        score += 2
    elif MINUTE_MIN <= minute <= 44:
        score += 1

    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0
    is_home = fixture["teams"]["home"]["id"] == team_id
    opp_goals = away_goals if is_home else home_goals

    if opp_goals >= 1:
        score += 3
    elif home_goals == 0 and away_goals == 0:
        score += 2

    fixture_id = fixture["fixture"]["id"]
    if (fixture_id, team_id) in notified:
        score += 3

    return score


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    q = total_quota()
    if q >= 70: return "NORMAL"
    elif q >= 40: return "CAREFUL"
    elif q >= 20: return "STRICT"
    elif q >= 5: return "EMERGENCY"
    else: return "STOP"


def get_max_candidates() -> int:
    q = total_quota()
    if q >= 70: return 10
    elif q >= 40: return 5
    elif q >= 20: return 3
    elif q >= 5: return 1
    else: return 0


# ============================================================
# DYNAMIC POLLING
# ============================================================

def get_poll_interval(best_score: int, has_candidates: bool) -> int:
    if not has_candidates:
        return 1800
    if best_score >= 7: return 120
    if best_score >= 5: return 180
    if best_score >= 3: return 300
    return 600


# ============================================================
# SIGNAL TIERS
# ============================================================

def classify_signal(sot: int, total_shots: int, goals: int) -> str | None:
    if goals > 0 or sot < 3:
        return None
    if sot >= 5 and total_shots >= 10:
        return "VERY STRONG"
    if sot >= 4 and total_shots >= 8:
        return "STRONG"
    return "WATCH"


def tier_emoji(tier: str) -> str:
    if tier == "VERY STRONG": return "[RED]"
    if tier == "STRONG": return "[ORANGE]"
    return "[YELLOW]"


# ============================================================
# LOCAL FILTERING (zero API cost)
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[dict, int, int]]:
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

        home_goals = fixture["goals"]["home"] or 0
        away_goals = fixture["goals"]["away"] or 0

        if home_goals == 0:
            tid = fixture["teams"]["home"]["id"]
            rank = rank_candidate(fixture, tid)
            candidates.append((fixture, tid, rank))
        if away_goals == 0:
            tid = fixture["teams"]["away"]["id"]
            rank = rank_candidate(fixture, tid)
            candidates.append((fixture, tid, rank))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


# ============================================================
# MAIN CHECK CYCLE
# ============================================================

def check_cycle(client: httpx.Client) -> int:
    data = api_get(client, "/fixtures", {"live": "all"})
    fixtures = data.get("response", [])
    tracked = [f for f in fixtures if f["league"]["id"] in LEAGUE_IDS]
    budget = get_budget_mode()
    max_cand = get_max_candidates()
    q = total_quota()

    log.info(
        f"Quota: {q} [{budget}] | Max candidates: {max_cand} | "
        f"Live: {len(fixtures)} | Tracked: {len(tracked)} | Requests: {request_count}"
    )

    for k, v in quota_by_key.items():
        log.info(f"  Key {k[:8]}...: {v} remaining")

    if budget == "STOP":
        log.warning("Quota exhausted. Pausing API calls.")
        return 0

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(f"  -> {m['league']['name']}: {m['teams']['home']['name']} vs {m['teams']['away']['name']} ({m['fixture']['status']['short']} {minute}')")

    candidates = find_candidates(fixtures)
    log.info(f"  -> {len(candidates)} candidate(s) after local filter (0-goal teams, {MINUTE_MIN}-{MINUTE_MAX}')")

    if not candidates:
        return 0

    selected = candidates[:max_cand]

    fixture_team_map: dict[int, list[int]] = {}
    for fixture, tid, rank in selected:
        fid = fixture["fixture"]["id"]
        if fid not in fixture_team_map:
            fixture_team_map[fid] = []
        fixture_team_map[fid].append(tid)

    log.info(f"  -> Checking {len(fixture_team_map)} fixture(es) for stats (quota allows {max_cand})")

    best_score = selected[0][2] if selected else 0

    for fid, team_ids in fixture_team_map.items():
        if total_quota() <= 1:
            log.warning("Quota nearly gone, stopping stats fetches.")
            break

        try:
            stats_data = api_get(client, "/fixtures/statistics", {"fixture": fid})
        except Exception as e:
            log.warning(f"Stats failed for {fid}: {e}")
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

        teams_data = {}
        for team_entry in stats:
            tname = team_entry["team"]["name"]
            tmap = {}
            for s in team_entry.get("statistics", []):
                val = s.get("value", "0")
                if val is None: val = "0"
                tmap[s["type"]] = str(val).strip()
            teams_data[tname] = tmap

        for tname, tstats in teams_data.items():
            sot_raw = tstats.get("Shots on Goal", "0")
            shots_raw = tstats.get("Total Shots", "0")
            try:
                sot = int(sot_raw)
                total_shots = int(shots_raw)
            except (ValueError, TypeError):
                continue

            if fixture["teams"]["home"]["name"] == tname:
                tid = fixture["teams"]["home"]["id"]
                goals = fixture["goals"]["home"] or 0
            else:
                tid = fixture["teams"]["away"]["id"]
                goals = fixture["goals"]["away"] or 0

            tier = classify_signal(sot, total_shots, goals)

            if tier:
                key = (fid, tid)
                last_sot = notified.get(key, 0)

                if sot > last_sot:
                    possession = tstats.get("Ball Possession", "N/A")
                    corners = tstats.get("Corner Kicks", "N/A")
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
                    minute = fixture["fixture"]["status"]["elapsed"]

                    msg = (
                        f"{tier_emoji(tier)} {tier} SIGNAL\n\n"
                        f"{home}  {sh} - {sa}  {away}\n"
                        f"{league}  {minute}'\n\n"
                        f"{tname}\n"
                        f"  Shots on target: {sot}\n"
                        f"  Total shots: {total_shots}\n"
                        f"  Opponent SOT: {opp_sot}\n"
                        f"  Goals: {goals}\n"
                        f"  Possession: {possession}\n"
                        f"  Corners: {corners}"
                    )

                    if send_telegram(client, msg):
                        log.info(f"{tier}: {tname} - {sot} SOT, {total_shots} shots, 0 goals (fixture {fid})")
                        notified[key] = sot

                        signals_sent.append({
                            "time": time.strftime("%Y-%m-%d %H:%M"),
                            "fixture": fid,
                            "team": tname,
                            "league": league,
                            "minute": minute,
                            "sot": sot,
                            "total_shots": total_shots,
                            "opp_sot": opp_sot,
                            "possession": possession,
                            "tier": tier,
                        })

            else:
                key = (fid, tid)
                if key in notified:
                    del notified[key]

    return best_score


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log.info("Football Bot v3 starting...")
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"Keys: {len(API_KEYS)} (tracked individually, not assumed additive)")
    log.info("Smart budget: rank candidates -> quota limit -> dynamic polling")
    log.info("Signal tiers: WATCH (3 SOT) / STRONG (4+ SOT, 8+ shots) / VERY STRONG (5+ SOT, 10+ shots)")

    with httpx.Client(timeout=30.0) as client:
        while True:
            try:
                best_score = check_cycle(client)
            except Exception as e:
                log.error(f"Cycle error: {e}")
                best_score = 0

            interval = get_poll_interval(best_score, best_score > 0)
            log.info(f"Next check in {interval}s | Best candidate score: {best_score} | Quota: {total_quota()}")
            time.sleep(interval)


if __name__ == "__main__":
    main()