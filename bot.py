"""
Football Live Tracker Telegram Bot
=================================
Monitors live football matches across major leagues (including Bulgarian Parva Liga).
Sends a Telegram notification when a team has >0 shots on target but 0 goals scored,
along with possession %, corners, and goals info.

Uses api-football.com (RapidAPI) for live data.
"""

import os
import sys
import time
import logging
from telegram import Bot
from telegram.error import TelegramError
import httpx
from dotenv import load_dotenv

load_dotenv()

# --- Logging (set up FIRST so we can see errors) ---
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

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    missing.append("RAPIDAPI_KEY")

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL", "60"))

if missing:
    log.error(f"MISSING ENV VARIABLES: {', '.join(missing)}")
    log.error("Please add them in Railway > Variables tab")
    sys.exit(1)

API_BASE = "https://api-football-v1.p.rapidapi.com/v3"
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
}

# Major league IDs (api-football.com)
LEAGUE_IDS = {
    39:   "Premier League",
    140:  "La Liga",
    78:   "Bundesliga",
    135:  "Serie A",
    61:   "Ligue 1",
    2:    "Champions League",
    3:    "Europa League",
    848:  "Conference League",
    211:  "Parva Liga (Bulgaria)",
    94:   "Primeira Liga",
    88:   "Eredivisie",
    203:  "Super Lig",
    144:  "Liga Profesional",
    71:   "Serie A (Brazil)",
    340:  "Liga MX",
}

# Live fixture statuses we care about
LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}

# Track notified (fixture_id, team_id) -> last known shots_on_target
notified: dict[tuple[int, int], int] = {}


def get_live_fixtures(client: httpx.Client) -> list[dict]:
    resp = client.get(f"{API_BASE}/fixtures", params={"live": "all"}, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])


def get_fixture_stats(client: httpx.Client, fixture_id: int) -> list[dict]:
    resp = client.get(
        f"{API_BASE}/fixtures/statistics",
        params={"fixture": fixture_id},
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])


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


def check_fixtures(client: httpx.Client, bot: Bot):
    fixtures = get_live_fixtures(client)
    log.info(f"Found {len(fixtures)} live fixtures")

    for fixture in fixtures:
        league_id = fixture["league"]["id"]
        if league_id not in LEAGUE_IDS:
            continue

        fixture_id = fixture["fixture"]["id"]
        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue

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
                goals = fixture["goals"]["home"] or 0
                team_id = fixture["teams"]["home"]["id"]
            else:
                goals = fixture["goals"]["away"] or 0
                team_id = fixture["teams"]["away"]["id"]

            if shots_on_target > 0 and goals == 0:
                key = (fixture_id, team_id)
                last_notified_shots = notified.get(key, 0)

                if shots_on_target > last_notified_shots:
                    possession = tstats.get("Ball Possession", "N/A")
                    corners = tstats.get("Corner Kicks", "N/A")

                    team_stats_summary = {
                        "shots_on_target": str(shots_on_target),
                        "goals": str(goals),
                        "possession": possession,
                        "corners": corners,
                    }

                    msg = build_signal_message(fixture, team_name, team_stats_summary)
                    try:
                        bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=msg,
                        )
                        log.info(
                            f"Signal sent: {team_name} has {shots_on_target} shots on target, 0 goals "
                            f"(fixture {fixture_id})"
                        )
                        notified[key] = shots_on_target
                    except TelegramError as e:
                        log.error(f"Failed to send Telegram message: {e}")

            else:
                key = (fixture_id, team_id)
                if key in notified:
                    del notified[key]


def main():
    log.info("Football Live Tracker Bot starting...")
    log.info(f"Polling every {POLL_INTERVAL_SECONDS}s")
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    with httpx.Client(timeout=30.0) as client:
        while True:
            try:
                check_fixtures(client, bot)
            except httpx.HTTPError as e:
                log.error(f"API request failed: {e}")
            except Exception as e:
                log.error(f"Unexpected error: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
