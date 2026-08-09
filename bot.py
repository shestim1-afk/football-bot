"""
Football Live Tracker Telegram Bot
=================================
Monitors live football matches across major leagues (including Bulgarian Parva Liga).
Sends a Telegram notification when a team has >0 shots on target but 0 goals scored,
along with possession %, corners, and goals info.

Uses api-football.com (RapidAPI) for live data.
"""

import os
import time
import logging
from datetime import datetime, timezone
from telegram import Bot
from telegram.error import TelegramError
import httpx
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# --- Config from env ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
RAPIDAPI_KEY = os.environ["RAPIDAPI_KEY"]
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL", "60"))

API_BASE = "https://api-football-v1.p.rapidapi.com/v3"
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
}

# Major league IDs (api-football.com)
# Full list: https://www.api-football.com/documentation-v3
LEAGUE_IDS = {
    39:   "🏴 Premier League",
    140:  "🇪🇸 La Liga",
    78:   "🇩🇪 Bundesliga",
    135:  "🇮🇹 Serie A",
    61:   "🇫🇷 Ligue 1",
    2:    "🏆 Champions League",
    3:    "🏆 Europa League",
    848:  "🏆 Conference League",
    211:  "🇧🇬 Parva Liga (Bulgaria)",
    94:   "🇵🇹 Primeira Liga",
    88:   "🇳🇱 Eredivisie",
    203:  "🇹🇷 Süper Lig",
    144:  "🇦🇷 Liga Profesional",
    71:   "🇧🇷 Serie A (Brazil)",
    340:  "🇲🇽 Liga MX",
    5:    "🇫🇷 Ligue 1",
}

# Live fixture statuses we care about
LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}

# Track notified (fixture_id, team_id) -> last known shots_on_target
# Re-notify when shots_on_target increases (still 0 goals)
notified: dict[tuple[int, int], int] = {}


def get_live_fixtures(client: httpx.Client) -> list[dict]:
    """Fetch all currently live fixtures."""
    resp = client.get(f"{API_BASE}/fixtures", params={"live": "all"}, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])


def get_fixture_stats(client: httpx.Client, fixture_id: int) -> list[dict]:
    """Fetch statistics for a specific fixture."""
    resp = client.get(
        f"{API_BASE}/fixtures/statistics",
        params={"fixture": fixture_id},
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", [])


def extract_team_stat(stats: list[dict], team_name: str, stat_type: str) -> str | None:
    """Extract a specific stat value for a team from the stats array.
    stat_type examples: 'Shots on Goal', 'Ball Possession', 'Corner Kicks'
    """
    for entry in stats:
        if entry.get("type") == stat_type:
            value = entry.get(team_name)
            if value is not None:
                return str(value).strip()
    return None


def build_signal_message(fixture: dict, team_name: str, team_stats: dict) -> str:
    """Build the Telegram message for a signal."""
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
        f"🚨 <b>SHOTS ON TARGET BUT NO GOAL</b>\n\n"
        f"⚽ <b>{home}  {score_home} - {score_away}  {away}</b>\n"
        f"📡 {league_name}  ⏱ {minute}'\n\n"
        f"🔴 <b>{team_name}</b>\n"
        f"   🎯 Shots on target: <b>{shots_on_target}</b>\n"
        f"   ⚽ Goals scored: <b>{goals}</b>\n"
        f"   📊 Possession: <b>{possession}</b>\n"
        f"   📐 Corners: <b>{corners}</b>"
    )
    return msg


def check_fixtures(client: httpx.Client, bot: Bot):
    """Main logic: fetch live fixtures, check stats, send signals."""
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

        # Build a dict of team stats for both home and away
        teams_data = {}
        for team_entry in stats:
            tname = team_entry["team"]["name"]
            team_stats_map = {}
            for s in team_entry.get("statistics", []):
                stype = s["type"]
                svalue = s.get("value", "0")
                # Normalize empty/none values
                if svalue is None:
                    svalue = "0"
                team_stats_map[stype] = str(svalue).strip()
            teams_data[tname] = team_stats_map

        # Check each team
        for team_name, tstats in teams_data.items():
            shots_raw = tstats.get("Shots on Goal", "0")
            try:
                shots_on_target = int(shots_raw)
            except (ValueError, TypeError):
                continue

            # Determine this team's goals from the fixture
            if fixture["teams"]["home"]["name"] == team_name:
                goals = fixture["goals"]["home"] or 0
                team_id = fixture["teams"]["home"]["id"]
            else:
                goals = fixture["goals"]["away"] or 0
                team_id = fixture["teams"]["away"]["id"]

            # Condition: >0 shots on target AND 0 goals
            if shots_on_target > 0 and goals == 0:
                key = (fixture_id, team_id)
                last_notified_shots = notified.get(key, 0)

                # Only notify if shots_on_target increased since last notification
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
                            parse_mode="HTML",
                        )
                        log.info(
                            f"Signal sent: {team_name} has {shots_on_target} shots on target, 0 goals "
                            f"(fixture {fixture_id})"
                        )
                        notified[key] = shots_on_target
                    except TelegramError as e:
                        log.error(f"Failed to send Telegram message: {e}")

            else:
                # Team scored — clear their notification state for this fixture
                key = (fixture_id, team_id)
                if key in notified:
                    del notified[key]


def main():
    log.info("Football Live Tracker Bot starting...")
    log.info(f"Polling every {POLL_INTERVAL_SECONDS}s")
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues")
    log.info(f"League IDs: {list(LEAGUE_IDS.keys())}")

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


if __name__ == "main__":
    main()
