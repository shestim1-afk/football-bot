"""
Football Live Tracker Telegram Bot v2 — Smart Budget Edition
===========================================================
Architecture:
  1. /fixtures?live=all  →  free scan (score, minute, status)
  2. Local pre-filter    →  20 leagues + 0-goal teams + minute window
  3. Candidate ranking   →  only spend stats requests on best candidates
  4. Dynamic budget      →  reads x-ratelimit-requests-remaining from API
  5. Tiered strictness   →  narrows minute window as quota drops

Signal criteria:
  - Team has 3+ shots on target AND 0 goals
  - Only re-notifies if shots on target increased since last alert
  - Resets when team scores
"""

import os
import sys
import time
import logging
import itertools
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

log.info(f"Loaded {len(API_KEYS)} API key(s) = {len(API_KEYS) * 100} requests/day")

API_BASE = "https://v3.football.api-sports.io"
TELEGRAM_API = "https://api.telegram.org"

# Round-robin key rotation
_key_cycle = itertools.cycle(API_KEYS)

# --- Adaptive polling ---
IDLE_INTERVAL = 1800
ACTIVE_INTERVAL = 600
_current_interval = IDLE_INTERVAL

# --- Dynamic quota tracking ---
remaining_quota = 100


def get_headers() -> dict:
    key = next(_key_cycle)
    return {"x-apisports-key": key}


def update_quota(resp: httpx.Response):
    global remaining_quota
    try:
        val = resp.headers.get("x-ratelimit-requests-remaining", "")
        if val:
            remaining_quota = int(val)
    except (ValueError, TypeError):
        pass


LEAGUE_IDS = {
    39:   "Premier League",
    140:  "La Liga",
    78:   "Bundesliga",
    79:   "2. Bundesliga",
    135:  "Serie A",
    61:   "Ligue 1",
    2:    "Champions League",
    3:    "Europa League",
    848:  "Conference League",
    357:  "First League (Bulgaria)",
    94:   "Primeira Liga",
    88:   "Eredivisie",
    203:  "Super Lig",
    169:  "Austrian Bundesliga",
    283:  "SuperLiga (Serbia)",
    210:  "HNL (Croatia)",
    345:  "Czech First League",
    119:  "Danish Superliga",
    137:  "Veikkausliiga (Finland)",
    191:  "NB I (Hungary)",
}

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}
notified: dict[tuple[int, int], int] = {}
request_count = 0


def get_minute_window() -> tuple[int, int]:
    if remaining_quota > 50:
        return (25, 75)
    elif remaining_quota > 30:
        return (30, 70)
    elif remaining_quota > 15:
        return (35, 65)
    else:
        return (40, 60)


def get_budget_mode() -> str:
    if remaining_quota > 50:
        return "NORMAL"
    elif remaining_quota > 30:
        return "CAREFUL"
    elif remaining_quota > 15:
        return "STRICT"
    elif remaining_quota > 0:
        return "EMERGENCY"
    else:
        return "EXHAUSTED"


def get_live_fixtures(client: httpx.Client) -> list[dict]:
    global request_count
    request_count += 1
    resp = client.get(f"{API_BASE}/fixtures", params={"live": "all"}, headers=get_headers())
    resp.raise_for_status()
    update_quota(resp)
    data = resp.json()
    return data.get("response", [])


def get_fixture_stats(client: httpx.Client, fixture_id: int) -> list[dict] | None:
    global request_count
    if remaining_quota <= 1:
        log.warning(f"Quota exhausted ({remaining_quota}), skipping stats for fixture {fixture_id}")
        return None
    request_count += 1
    resp = client.get(
        f"{API_BASE}/fixtures/statistics",
        params={"fixture": fixture_id},
        headers=get_headers(),
    )
    resp.raise_for_status()
    update_quota(resp)
    data = resp.json()
    return data.get("response", [])


def send_telegram_message(client: httpx.Client, text: str) -> bool:
    try:
        resp = client.post(
            f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Failed to send Telegram message: {e}")
        return False


def build_signal_message(fixture: dict, team_name: str, team_stats: dict) -> str:
    league_name = LEAGUE_IDS.get(fixture["league"]["id"], fixture["league"]["name"])
    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]
    score_home = fixture["goals"]["home"]
    score_away = fixture["goals"]["away"]
    minute = fixture["fixture"]["status"]["elapsed"]

    possession = team_stats.get("possession", "N/A")
    corners = team_stats.get("corners", "N/A")
    shots_on_target = team_stats.get("shots_on_target", "?")
    goals = team_stats.get("goals", "0")

    msg = (
        f"SHOTS ON TARGET BUT NO GOAL\n\n"
        f"{home}  {score_home} - {score_away}  {away}\n"
        f"{league_name}  {minute}'\n\n"
        f"{team_name}\n"
        f"  Shots on target: {shots_on_target}\n"
        f"  Goals scored: {goals}\n"
        f"  Possession: {possession}\n"
        f"  Corners: {corners}"
    )
    return msg


def find_candidates(fixtures: list[dict]) -> list[tuple[dict, int, int, int]]:
    min_min, max_min = get_minute_window()
    candidates = []

    for fixture in fixtures:
        league_id = fixture["league"]["id"]
        if league_id not in LEAGUE_IDS:
            continue

        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue

        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

        if minute < min_min or minute > max_min:
            continue

        home_goals = fixture["goals"]["home"] or 0
        away_goals = fixture["goals"]["away"] or 0

        if home_goals == 0:
            candidates.append((fixture, fixture["teams"]["home"]["id"], 0, minute))
        if away_goals == 0:
            candidates.append((fixture, fixture["teams"]["away"]["id"], 0, minute))

    return candidates


def check_fixtures(client: httpx.Client) -> bool:
    global _current_interval

    fixtures = get_live_fixtures(client)
    tracked_matches = [f for f in fixtures if f["league"]["id"] in LEAGUE_IDS]

    budget_mode = get_budget_mode()
    min_min, max_min = get_minute_window()
    log.info(
        f"Quota: {remaining_quota} [{budget_mode}] | "
        f"Window: {min_min}-{max_min}' | "
        f"Total live: {len(fixtures)} | Tracked: {len(tracked_matches)} | "
        f"Requests used: {request_count}"
    )

    if budget_mode == "EXHAUSTED":
        log.warning("Quota exhausted. Waiting for reset. Next check in 30 min.")
        _current_interval = IDLE_INTERVAL
        return False

    if tracked_matches:
        for m in tracked_matches:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(f"  -> {m['league']['name']}: {m['teams']['home']['name']} vs {m['teams']['away']['name']} ({m['fixture']['status']['short']} {minute}')")

    _current_interval = ACTIVE_INTERVAL if tracked_matches else IDLE_INTERVAL

    candidates = find_candidates(fixtures)
    log.info(f"  -> {len(candidates)} candidate(s) after local filter (0-goal teams in {min_min}-{max_min}')")

    checked_fixtures = set()

    for fixture, team_id, goals, minute in candidates:
        fixture_id = fixture["fixture"]["id"]

        if fixture_id in checked_fixtures:
            continue
        checked_fixtures.add(fixture_id)

        try:
            stats = get_fixture_stats(client, fixture_id)
        except (httpx.HTTPError, Exception) as e:
            log.warning(f"Could not fetch stats for fixture {fixture_id}: {e}")
            continue

        if not stats:
            continue

        teams_data = {}
        for team_entry in stats:
            tname = team_entry["team"]["name"]
            team_stats_map = {}
            for s in team_entry.get("statistics", []):
                stype = s["type"]
                svalue = s.get("value", "0")
                if svalue is None:
                    svalue = "0"
                team_stats_map[stype] = str(svalue).strip()
            teams_data[tname] = team_stats_map

        for team_name, tstats in teams_data.items():
            shots_raw = tstats.get("Shots on Goal", "0")
            try:
                shots_on_target = int(shots_raw)
            except (ValueError, TypeError):
                continue

            if fixture["teams"]["home"]["name"] == team_name:
                team_goals = fixture["goals"]["home"] or 0
                tid = fixture["teams"]["home"]["id"]
            else:
                team_goals = fixture["goals"]["away"] or 0
                tid = fixture["teams"]["away"]["id"]

            if shots_on_target >= 3 and team_goals == 0:
                key = (fixture_id, tid)
                last_notified_shots = notified.get(key, 0)

                if shots_on_target > last_notified_shots:
                    possession = tstats.get("Ball Possession", "N/A")
                    corners = tstats.get("Corner Kicks", "N/A")

                    team_stats_summary = {
                        "shots_on_target": str(shots_on_target),
                        "goals": str(team_goals),
                        "possession": possession,
                        "corners": corners,
                    }

                    msg = build_signal_message(fixture, team_name, team_stats_summary)
                    if send_telegram_message(client, msg):
                        log.info(f"SIGNAL: {team_name} — {shots_on_target} SOT, 0 goals (fixture {fixture_id})")
                        notified[key] = shots_on_target

            else:
                key = (fixture_id, tid)
                if key in notified:
                    del notified[key]

    return len(tracked_matches) > 0


def main():
    log.info("Football Live Tracker Bot v2 starting...")
    log.info(f"Idle poll: {IDLE_INTERVAL}s | Active poll: {ACTIVE_INTERVAL}s")
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info("Smart budget: local pre-filter + dynamic quota tracking")

    with httpx.Client(timeout=30.0) as client:
        while True:
            try:
                check_fixtures(client)
            except httpx.HTTPError as e:
                log.error(f"API request failed: {e}")
            except Exception as e:
                log.error(f"Unexpected error: {e}")

            log.info(f"Next check in {_current_interval}s (quota left: {remaining_quota})")
            time.sleep(_current_interval)


if __name__ == "__main__":
    main()