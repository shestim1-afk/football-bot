#!/usr/bin/env python3
"""
Investigate why Juventus and/or FC København had no signals on a given date.
Run this ON the Railway container (or locally if you have /data/pressure_polls.jsonl).

Usage:
  python investigate_teams.py                  # default: 2026-08-29, Juventus + København
  python investigate_teams.py --date 2026-08-29 --teams "Juventus,Kjøbenhavn"
  python investigate_teams.py --date 2026-08-29 --teams "Juventus"
  python investigate_teams.py --date 2026-08-29 --teams "Kjøbenhavn" --api-key YOUR_KEY
"""

import json
import os
import sys
import re
import argparse
from datetime import datetime, timezone
from collections import defaultdict

VOLUME_DIR = os.environ.get("VOLUME_DIR", "/data")
POLL_FILE = os.path.join(VOLUME_DIR, "pressure_polls.jsonl")
OUTCOMES_FILE = os.path.join(VOLUME_DIR, "signal_outcomes.jsonl")
API_BASE = "https://v3.football.api-sports.io"

# Try to load API key from bot env or .env
API_KEY = os.environ.get("API_FOOTBALL_KEY", "")


def load_jsonl(path):
    """Load all entries from a JSONL file."""
    entries = []
    if not os.path.exists(path):
        print(f"  [WARN] File not found: {path}")
        return entries
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def find_team_polls(polls, team_patterns, target_date):
    """Find all poll entries matching team name patterns on target date."""
    matched = []
    for entry in polls:
        ts = entry.get("ts", 0)
        try:
            entry_date = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            continue
        if entry_date != target_date:
            continue
        name = entry.get("team_name", "")
        for pattern in team_patterns:
            if pattern.lower() in name.lower():
                entry["_match_pattern"] = pattern
                matched.append(entry)
                break
    return matched


def find_team_signals(signals, team_patterns, target_date):
    """Find signal outcome entries matching team patterns."""
    matched = []
    for entry in signals:
        name = entry.get("team_name", "")
        for pattern in team_patterns:
            if pattern.lower() in name.lower():
                entry["_match_pattern"] = pattern
                matched.append(entry)
                break
    return matched


def print_gps_timeline(polls, team_name):
    """Print a minute-by-minute GPS timeline for a team."""
    if not polls:
        return

    # Sort by minute
    polls_sorted = sorted(polls, key=lambda x: (x.get("minute", 0), x.get("ts", 0)))

    print(f"\n{'='*90}")
    print(f"  GPS TIMELINE: {team_name}  ({len(polls_sorted)} polls)")
    print(f"{'='*90}")
    print(f"{'Min':>4} {'GPS':>6} {'SOT':>4} {'Shots':>5} {'SIB':>4} {'xG':>6} {'Corners':>7} {'Sust':>4} {'Accel':>5} {'Score':>7}  Gate")
    print(f"{'-'*4} {'-'*6} {'-'*4} {'-'*5} {'-'*4} {'-'*6} {'-'*7} {'-'*4} {'-'*5} {'-'*7}  {'-'*30}")

    prev_gps = None
    gps_55_count = 0
    max_gps = 0
    gps_above_55_minutes = set()

    for p in polls_sorted:
        minute = p.get("minute", 0)
        gps = p.get("gps", 0)
        sot = p.get("sot", 0)
        shots = p.get("total_shots", 0)
        sib = p.get("shots_inside_box", 0)
        xg = p.get("xg")
        corners = p.get("corners", 0)
        sust = p.get("sustained", 0)
        accel = p.get("accel_count", 0)
        sh = p.get("score_home", 0)
        sa = p.get("score_away", 0)
        is_home = p.get("is_home", True)
        score_str = f"{sh}-{sa}" + (" (H)" if is_home else " (A)")

        # Determine gate status from GPS
        gate = ""
        if gps >= 65:
            gate = "CRITICAL"
        elif gps >= 55:
            gate = "EARLY"
        elif gps >= 45:
            gate = "elevated"
        else:
            gate = "low"

        xg_str = f"{xg:.2f}" if xg is not None else "  N/A"

        # Highlight rows above threshold
        marker = " >>>" if gps >= 55 else ""

        print(f"{minute:>4}' {gps:>6.1f} {sot:>4} {shots:>5} {sib:>4} {xg_str:>6} {corners:>7} {sust:>4} {accel:>5} {score_str:>7}  {gate}{marker}")

        if gps >= 55:
            gps_55_count += 1
            gps_above_55_minutes.add(minute)
        if gps > max_gps:
            max_gps = gps
        prev_gps = gps

    # Summary
    print(f"\n  Summary for {team_name}:")
    print(f"    Max GPS: {max_gps:.1f}")
    print(f"    Polls with GPS >= 55: {gps_55_count}/{len(polls_sorted)}")
    print(f"    Minutes with GPS >= 55: {sorted(gps_above_55_minutes)}")

    if max_gps < 55:
        print(f"    >>> GPS NEVER reached signal threshold (55). No signal possible.")
        # Check what was the limiting factor
        if polls_sorted:
            last = polls_sorted[-1]
            print(f"    >>> Last poll at {last.get('minute', 0)}': SOT={last.get('sot', 0)}, "
                  f"Shots={last.get('total_shots', 0)}, SIB={last.get('shots_inside_box', 0)}, "
                  f"xG={last.get('xg', 'N/A')}, Corners={last.get('corners', 0)}")
    elif gps_55_count < 3:
        print(f"    >>> GPS crossed 55 briefly ({gps_55_count} polls) but likely not sustained enough.")


def check_why_blocked(polls, signals, team_name):
    """Analyze why a team that reached GPS>=55 still didn't get a signal."""
    above = [p for p in polls if p.get("gps", 0) >= 55]
    if not above:
        return

    print(f"\n  --- Gate Analysis for {team_name} (GPS >= 55 polls) ---")

    for p in above:
        minute = p.get("minute", 0)
        gps = p.get("gps", 0)
        sot = p.get("sot", 0)
        sh = p.get("score_home", 0)
        sa = p.get("score_away", 0)
        is_home = p.get("is_home", True)
        goal_diff = (sh - sa) if is_home else (sa - sh)
        sust = p.get("sustained", 0)
        accel = p.get("accel_count", 0)

        blocks = []

        # Check SOT gate (need >= 1 for EARLY, >= 2 for non-ACCEL)
        if sot < 1:
            blocks.append(f"SOT={sot} (need >=1)")
        elif sot < 2 and accel == 0:
            blocks.append(f"SOT={sot} + no accel (need SOT>=2 or accel)")

        # Check losing gate
        if goal_diff < 0:
            blocks.append(f"LOSING by {abs(goal_diff)}")

        # Check score dampener (winning by 2+)
        if goal_diff >= 2 and accel == 0:
            blocks.append(f"SCORE DAMPENER (winning by {goal_diff}, no accel)")

        # Check freshness (86'+)
        if minute >= 86:
            blocks.append(f"FRESHNESS 86'+ block")

        # Check sustained (need 1 for EARLY to pass)
        if sust == 0 and gps < 65:
            blocks.append(f"Not sustained (0 consecutive polls above threshold)")

        if blocks:
            print(f"    {minute}': GPS={gps:.1f} SOT={sot} score={'H' if is_home else 'A'} {sh}-{sa} => BLOCKED: {', '.join(blocks)}")
        else:
            print(f"    {minute}': GPS={gps:.1f} SOT={sot} score={'H' if is_home else 'A'} {sh}-{sa} => SHOULD HAVE SIGNAL (check FIRST_ONLY/cooldown)")


def search_stdout_logs(team_patterns, target_date):
    """Search Railway stdout logs if poll data is missing."""
    # Check common log locations on Railway
    log_locations = [
        "/app/bot.log",
        "/data/bot.log",
        "/tmp/bot.log",
    ]
    
    print(f"\n  Searching stdout logs for team mentions...")
    found_any = False

    for log_path in log_locations:
        if not os.path.exists(log_path):
            continue
        print(f"    Checking {log_path}...")
        with open(log_path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                for pattern in team_patterns:
                    if pattern.lower() in line.lower() and target_date in line:
                        print(f"      Line {i}: {line.strip()[:200]}")
                        found_any = True

    if not found_any:
        print(f"    No mentions found in log files.")
        print(f"    NOTE: Railway stdout logs are not accessible from inside the container.")
        print(f"    Check the Railway dashboard > Deployments > Logs for the full output.")


def query_api_football(team_patterns, target_date, api_key):
    """Query API-Football for fixture info if no poll data exists."""
    if not api_key:
        print(f"\n  [SKIP] API-Football query — no API_FOOTBALL_KEY env var set.")
        print(f"    Set it with: export API_FOOTBALL_KEY=your_key_here")
        return

    import httpx

    print(f"\n  Querying API-Football for fixtures on {target_date}...")
    headers = {"x-apisports-key": api_key}

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{API_BASE}/fixtures", headers=headers, params={"date": target_date})
            data = resp.json()

            if data.get("errors"):
                print(f"    API Error: {data['errors']}")
                return

            fixtures = data.get("response", [])
            print(f"    Found {len(fixtures)} fixtures on {target_date}")

            for pattern in team_patterns:
                for fx in fixtures:
                    home = fx.get("teams", {}).get("home", {}).get("name", "")
                    away = fx.get("teams", {}).get("away", {}).get("name", "")
                    league = fx.get("league", {}).get("name", "")
                    fx_id = fx.get("fixture", {}).get("id")
                    status = fx.get("fixture", {}).get("status", {}).get("short", "")
                    goals_h = fx.get("goals", {}).get("home", 0)
                    goals_a = fx.get("goals", {}).get("away", 0)

                    if pattern.lower() in home.lower() or pattern.lower() in away.lower():
                        print(f"\n    {'='*60}")
                        print(f"    MATCH: {home} vs {away}")
                        print(f"    League: {league}  |  Fixture ID: {fx_id}  |  Status: {status}")
                        print(f"    Final score: {goals_h} - {goals_a}")
                        print(f"    {'='*60}")

                        # Try to get fixture statistics
                        print(f"\n    Fetching statistics...")
                        try:
                            resp2 = client.get(f"{API_BASE}/fixtures/statistics", headers=headers, params={"fixture": fx_id})
                            stats_data = resp2.json()
                            stats = stats_data.get("response", [])

                            if not stats:
                                print(f"      No statistics available (game may not have been covered).")
                                continue

                            for team_stats in stats:
                                tname = team_stats.get("team", {}).get("name", "")
                                print(f"\n      --- {tname} ---")
                                for stat in team_stats.get("statistics", []):
                                    label = stat.get("type", "")
                                    home_val = stat.get("value", "N/A") if stat.get("value") is not None else "0"
                                    away_val = stat.get("value", "N/A")
                                    if isinstance(home_val, str) and home_val.isdigit():
                                        home_val = int(home_val)
                                    print(f"        {label:>22}: {home_val}")
                        except Exception as e:
                            print(f"      Failed to fetch stats: {e}")

    except Exception as e:
        print(f"    API query failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Investigate why teams had no signals")
    parser.add_argument("--date", default="2026-08-29", help="Date to investigate (YYYY-MM-DD)")
    parser.add_argument("--teams", default="Juventus,Kjøbenhavn,Kobenhavn,Copenhagen",
                       help="Comma-separated team name patterns to search")
    parser.add_argument("--api-key", default=None, help="API-Football key (or set API_FOOTBALL_KEY env)")
    parser.add_argument("--api-only", action="store_true", help="Skip local files, query API-Football only")
    args = parser.parse_args()

    team_patterns = [t.strip() for t in args.teams.split(",") if t.strip()]
    target_date = args.date
    api_key = args.api_key or API_KEY

    print(f"\n{'#'*60}")
    print(f"# INVESTIGATE: {', '.join(team_patterns)}")
    print(f"# Date: {target_date}")
    print(f"# Poll file: {POLL_FILE}")
    print(f"# Outcomes file: {OUTCOMES_FILE}")
    print(f"{'#'*60}")

    if not args.api_only:
        # Load poll data
        print(f"\n[1] Loading poll data from {POLL_FILE}...")
        polls = load_jsonl(POLL_FILE)
        print(f"    Total poll entries: {len(polls)}")

        # Filter by date to count
        date_polls = 0
        for p in polls:
            ts = p.get("ts", 0)
            try:
                d = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                if d == target_date:
                    date_polls += 1
            except:
                pass
        print(f"    Polls on {target_date}: {date_polls}")

        # Search for team polls
        print(f"\n[2] Searching for team polls...")
        for pattern in team_patterns:
            team_polls = find_team_polls(polls, [pattern], target_date)
            if team_polls:
                print(f"    FOUND {len(team_polls)} polls for '{pattern}'")
                print_gps_timeline(team_polls, pattern)
                check_why_blocked(team_polls, None, pattern)
            else:
                print(f"    NO polls found for '{pattern}' on {target_date}")

        # Check signals
        print(f"\n[3] Checking signal outcomes...")
        signals = load_jsonl(OUTCOMES_FILE)
        for pattern in team_patterns:
            team_signals = find_team_signals(signals, [pattern], target_date)
            if team_signals:
                print(f"    FOUND {len(team_signals)} signals for '{pattern}':")
                for s in team_signals:
                    print(f"      {s.get('minute', '?')}' | GPS={s.get('gps', '?')} | SOT={s.get('sot', '?')} | "
                          f"result={s.get('result', '?')} | {s.get('team_name', '?')}")
            else:
                print(f"    NO signals for '{pattern}' (confirmed)")

        # Search stdout logs
        print(f"\n[4] Searching stdout logs...")
        search_stdout_logs(team_patterns, target_date)

        # Check what fixtures WERE tracked
        print(f"\n[5] All teams tracked on {target_date} (from polls):")
        teams_seen = defaultdict(int)
        for p in polls:
            ts = p.get("ts", 0)
            try:
                d = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                if d == target_date:
                    teams_seen[p.get("team_name", "unknown")] += 1
            except:
                pass
        if teams_seen:
            for tname, count in sorted(teams_seen.items()):
                in_search = any(pat.lower() in tname.lower() for pat in team_patterns)
                marker = " <<<" if in_search else ""
                print(f"    {tname}: {count} polls{marker}")
        else:
            print(f"    No polls found for {target_date} at all.")

    # API-Football fallback
    print(f"\n[6] API-Football fixture lookup...")
    query_api_football(team_patterns, target_date, api_key)

    print(f"\n{'#'*60}")
    print(f"# INVESTIGATION COMPLETE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
