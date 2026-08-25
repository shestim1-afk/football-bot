#!/usr/bin/env python3
"""Find Champions League qualifying league ID from API-Football.
Run on Railway: python3 /app/scripts/find_cl_league_id.py
"""
import httpx, os, json

key = os.environ.get("RAPIDAPI_KEY", "")
if not key:
    print("No RAPIDAPI_KEY env var")
    exit(1)

headers = {"x-apisports-key": key}

# Search for leagues with 'Champions' in name
resp = httpx.get(
    "https://v3.football.api-sports.io/leagues",
    params={"search": "Champions League", "current": "true"},
    headers=headers,
    timeout=15,
)
data = resp.json()

print(f"Found {data.get('results', 0)} leagues matching 'Champions League' (current season):")
print()

for league in data.get("response", []):
    lid = league["league"]["id"]
    name = league["league"]["name"]
    country = league["country"]["name"] if isinstance(league.get("country"), dict) else league.get("country")
    season = league["seasons"]
    curr = [s for s in season if s.get("current")]
    curr_str = f"season {curr[0]['year']}" if curr else "no current"
    print(f"  ID {lid:>4}: {name:<50s} | {country:<20s} | {curr_str}")

print()
# Also search for 'Champions League Qualifying' specifically
resp2 = httpx.get(
    "https://v3.football.api-sports.io/leagues",
    params={"search": "Champions League Qualifying", "current": "true"},
    headers=headers,
    timeout=15,
)
data2 = resp2.json()
if data2.get("response"):
    print(f"Found {data2.get('results', 0)} leagues matching 'Champions League Qualifying':")
    for league in data2["response"]:
        lid = league["league"]["id"]
        name = league["league"]["name"]
        print(f"  ID {lid:>4}: {name}")
else:
    print("No leagues found for 'Champions League Qualifying' (may be under same ID as main CL)")
