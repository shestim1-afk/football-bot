#!/usr/bin/env python3
"""
EOD (End-of-Day) Automated Data Collection Script
==============================================
Reads signal_outcomes.jsonl + pressure_polls.jsonl from the bot's /data volume,
generates a comprehensive daily report, and optionally sends it to Telegram.

Usage:
  python3 eod_report.py                     # Today's report (console only)
  python3 eod_report.py --date 2026-08-23    # Specific date
  python3 eod_report.py --send               # Send to Telegram
  python3 eod_report.py --days 3             # Last N days
  python3 eod_report.py --polls-only         # Only poll-level analysis
  python3 eod_report.py --out /path/to/file  # Write report to file

Cron example (Railway cron or system cron — run at 00:30 Bulgaria):
  30 0 * * * python3 /app/scripts/eod_report.py --send --days 1

Environment variables (same as bot):
  TELEGRAM_BOT_TOKEN  — for --send mode
  TELEGRAM_CHAT_ID    — for --send mode
  VOLUME_DIR          — defaults to /data
  RAPIDAPI_KEY        — NOT needed (reads local files only)
"""

import argparse
import json
import os
import sys
import httpx
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BULGARIA_TZ = ZoneInfo("Europe/Sofia")
VOLUME_DIR = os.environ.get("VOLUME_DIR", "/data")
OUTCOMES_FILE = os.path.join(VOLUME_DIR, "signal_outcomes.jsonl")
POLL_DATA_FILE = os.path.join(VOLUME_DIR, "pressure_polls.jsonl")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org"


def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_date_key(entry: dict) -> str:
    """Extract YYYY-MM-DD from a signal_outcomes entry (Bulgaria timezone)."""
    clock = entry.get("signal_clock", "")
    if clock:
        return clock[:10]
    # Fallback: convert timestamp
    ts = entry.get("signal_time", 0)
    if ts:
        return datetime.fromtimestamp(ts, tz=BULGARIA_TZ).strftime("%Y-%m-%d")
    return ""


def get_poll_date_key(entry: dict) -> str:
    """Extract YYYY-MM-DD from a pressure_polls entry."""
    ts = entry.get("ts", 0)
    if ts:
        return datetime.fromtimestamp(ts, tz=BULGARIA_TZ).strftime("%Y-%m-%d")
    return ""


def filter_by_date(entries: list[dict], target_date: str, date_fn) -> list[dict]:
    """Filter entries to a specific date string YYYY-MM-DD."""
    return [e for e in entries if date_fn(e) == target_date]


def filter_by_date_range(entries: list[dict], days: int, date_fn) -> list[dict]:
    """Filter entries to the last N days (Bulgaria time).
    
    --days 1 means "today + yesterday" (the last 1 full day including yesterday),
    because an EOD report at 00:30 on Aug 24 should cover Aug 23's signals,
    not Aug 24 (which hasn't happened yet).
    Fixed: was using range(days) which only included today for --days 1.
    """
    now_bg = datetime.now(BULGARIA_TZ)
    cutoffs = set()
    for d in range(days + 1):  # +1: --days 1 → today + yesterday
        day = (now_bg - timedelta(days=d)).strftime("%Y-%m-%d")
        cutoffs.add(day)
    return [e for e in entries if date_fn(e) in cutoffs]


def wr(numerator: int, denominator: int) -> str:
    """Format win rate as fraction + percentage."""
    if denominator == 0:
        return "0/0"
    pct = numerator / denominator * 100
    return f"{numerator}/{denominator} ({pct:.0f}%)"


def hit_count(entries: list[dict], field: str) -> int:
    return sum(1 for e in entries if e.get(field) == "HIT")


def avg(entries: list[dict], field: str) -> float:
    vals = [e.get(field, 0) or 0 for e in entries]
    return sum(vals) / len(vals) if vals else 0


# ============================================================
# SIGNAL OUTCOMES ANALYSIS
# ============================================================

def analyze_signals(signals: list[dict], all_signals: list[dict] | None = None) -> list[str]:
    """Generate full signal analysis report. Returns list of text lines."""
    lines = []
    resolved = [s for s in signals if s.get("resolved")]
    pending = [s for s in signals if not s.get("resolved")]
    first_only = [s for s in signals if s.get("sig_num") == 1]
    first_resolved = [s for s in first_only if s.get("resolved")]

    # --- Header ---
    total = len(signals)
    r_total = len(resolved)
    lines.append(f"SIGNALS: {total} total | {r_total} resolved | {len(pending)} pending")
    if not resolved:
        lines.append("No resolved signals.")
        return lines

    # --- Overall win rates (1st signal only) ---
    fr = first_resolved
    if fr:
        lines.append("")
        lines.append(f"=== 1ST SIGNAL ONLY ({len(fr)} resolved) ===")
        fh5 = hit_count(fr, "outcome_5min")
        fh10 = hit_count(fr, "outcome_10min")
        fh15 = hit_count(fr, "outcome_15min")
        fhf = hit_count(fr, "outcome_full")
        lines.append(f"  Full:    {wr(fhf, len(fr))}")
        lines.append(f"  15-min:  {wr(fh15, len(fr))}")
        lines.append(f"  10-min:  {wr(fh10, len(fr))}")
        lines.append(f"  5-min:   {wr(fh5, len(fr))}")

        # Avg signal-to-goal time for full hits
        goal_times = [e.get("goal_minute_full") for e in fr
                      if e.get("outcome_full") == "HIT" and e.get("goal_minute_full")]
        if goal_times:
            sig_mins = [e.get("game_minute", 0) for e in fr
                        if e.get("outcome_full") == "HIT" and e.get("goal_minute_full")]
            deltas = [g - s for g, s in zip(goal_times, sig_mins) if g > s]
            if deltas:
                lines.append(f"  Avg time to goal (full HIT): +{sum(deltas)/len(deltas):.0f}' from signal")

    # --- All signals win rates ---
    if all_signals and len(all_signals) > total:
        all_resolved = [s for s in all_signals if s.get("resolved")]
        if all_resolved:
            lines.append("")
            lines.append(f"=== ALL SIGNALS ({len(all_resolved)} resolved) ===")
            ah5 = hit_count(all_resolved, "outcome_5min")
            ah10 = hit_count(all_resolved, "outcome_10min")
            ah15 = hit_count(all_resolved, "outcome_15min")
            ahf = hit_count(all_resolved, "outcome_full")
            lines.append(f"  Full:    {wr(ahf, len(all_resolved))}")
            lines.append(f"  15-min:  {wr(ah15, len(all_resolved))}")
            lines.append(f"  10-min:  {wr(ah10, len(all_resolved))}")
            lines.append(f"  5-min:   {wr(ah5, len(all_resolved))}")

    # --- By tier ---
    lines.append("")
    lines.append("=== BY TIER ===")
    for tier in ("CRITICAL", "EARLY WARNING"):
        group = [e for e in resolved if e.get("tier") == tier]
        if not group:
            continue
        t = len(group)
        th15 = hit_count(group, "outcome_15min")
        thf = hit_count(group, "outcome_full")
        th5 = hit_count(group, "outcome_5min")
        agps = avg(group, "gps")
        lines.append(f"  {tier}: full {wr(thf, t)} | 15m {wr(th15, t)} | 5m {wr(th5, t)} | avg GPS {agps:.0f}")

    # --- By trigger type ---
    lines.append("")
    lines.append("=== BY TRIGGER ===")
    gps_trig = [e for e in resolved if e.get("gps_triggered")]
    sot_trig = [e for e in resolved if not e.get("gps_triggered")]
    if gps_trig:
        g = len(gps_trig)
        lines.append(f"  GPS-triggered (EW): full {wr(hit_count(gps_trig, 'outcome_full'), g)} | "
                      f"15m {wr(hit_count(gps_trig, 'outcome_15min'), g)} | "
                      f"avg GPS {avg(gps_trig, 'gps'):.0f} SOT {avg(gps_trig, 'sot'):.1f}")
    if sot_trig:
        g = len(sot_trig)
        lines.append(f"  SOT-triggered (CRIT): full {wr(hit_count(sot_trig, 'outcome_full'), g)} | "
                      f"15m {wr(hit_count(sot_trig, 'outcome_15min'), g)}")

    # --- By window tag ---
    lines.append("")
    lines.append("=== BY WINDOW ===")
    for label, key in [("CORE (21-60')", "CORE"),
                       ("EARLY (<21')", "EARLY_OVERRIDE"),
                       ("LATE (61'+)", "LATE_OVERRIDE")]:
        group = [e for e in resolved if e.get("window_tag") == key]
        if not group:
            continue
        t = len(group)
        lines.append(f"  {label}: full {wr(hit_count(group, 'outcome_full'), t)} | "
                      f"15m {wr(hit_count(group, 'outcome_15min'), t)} | "
                      f"avg GPS {avg(group, 'gps'):.0f} avg min {avg(group, 'game_minute'):.0f}'")

    # --- By GPS range ---
    lines.append("")
    lines.append("=== BY GPS RANGE ===")
    for label, lo, hi in [("55-64", 55, 65), ("65-74", 65, 75),
                           ("75-84", 75, 85), ("85+", 85, 999)]:
        group = [e for e in resolved if lo <= e.get("gps", 0) < hi]
        if not group:
            continue
        t = len(group)
        lines.append(f"  GPS {label}: full {wr(hit_count(group, 'outcome_full'), t)} | "
                      f"15m {wr(hit_count(group, 'outcome_15min'), t)} | "
                      f"avg accel {avg(group, 'accel_count'):.1f}")

    # --- By minute range ---
    lines.append("")
    lines.append("=== BY MINUTE ===")
    for label, lo, hi in [("21-35'", 21, 36), ("36-45'", 36, 46),
                           ("46-55'", 46, 56), ("56-61'", 56, 62),
                           ("62-75'", 62, 76), ("76-85'", 76, 86)]:
        group = [e for e in resolved if lo <= e.get("game_minute", 0) < hi]
        if not group:
            continue
        t = len(group)
        lines.append(f"  {label}: full {wr(hit_count(group, 'outcome_full'), t)} | "
                      f"15m {wr(hit_count(group, 'outcome_15min'), t)} | "
                      f"5m {wr(hit_count(group, 'outcome_5min'), t)} | "
                      f"avg GPS {avg(group, 'gps'):.0f}")

    # --- GPS x Minute cross-tab ---
    minute_buckets = [
        ("9-35'", lambda e: 9 <= e.get("game_minute", 0) <= 35),
        ("36-45'", lambda e: 36 <= e.get("game_minute", 0) <= 45),
        ("46-61'", lambda e: 46 <= e.get("game_minute", 0) <= 61),
        ("62-85'", lambda e: 62 <= e.get("game_minute", 0) <= 85),
    ]
    gps_rows = [
        ("GPS 55-64", lambda e: 55 <= e.get("gps", 0) < 65),
        ("GPS 65-74", lambda e: 65 <= e.get("gps", 0) < 75),
        ("GPS 75-84", lambda e: 75 <= e.get("gps", 0) < 85),
        ("GPS 85+", lambda e: e.get("gps", 0) >= 85),
    ]
    has_data = any(
        [e for e in resolved if gf(e) and mf(e)]
        for _, gf in gps_rows for _, mf in minute_buckets
    )
    if has_data:
        lines.append("")
        lines.append("=== GPS x MINUTE (full WR) ===")
        header = "             " + "  ".join(f"{ml:>10}" for ml, _ in minute_buckets)
        lines.append(header)
        for glabel, gfilter in gps_rows:
            row = [f"{glabel:<13}"]
            for _, mfilter in minute_buckets:
                cell = [e for e in resolved if gfilter(e) and mfilter(e)]
                if cell:
                    ch = hit_count(cell, "outcome_full")
                    ct = len(cell)
                    row.append(f"{ch}/{ct}({ch/ct*100:.0f}%)  ")
                else:
                    row.append(f"{'---':>10}")
            lines.append("  ".join(row))

    # --- Recency ratio analysis ---
    with_rr = [e for e in resolved if e.get("recency_ratio") is not None]
    if len(with_rr) >= 3:
        lines.append("")
        lines.append("=== RECENCY ANALYSIS ===")
        mid = 0.3
        for label, fn in [(f"RR <{mid} (accumulated)", lambda e: e.get("recency_ratio", 0) < mid),
                          (f"RR >={mid} (fresh)", lambda e: e.get("recency_ratio", 0) >= mid)]:
            group = [e for e in with_rr if fn(e)]
            if len(group) < 2:
                continue
            t = len(group)
            lines.append(f"  {label}: full {wr(hit_count(group, 'outcome_full'), t)} | "
                          f"15m {wr(hit_count(group, 'outcome_15min'), t)} | "
                          f"avg RR {avg(group, 'recency_ratio'):.2f} "
                          f"GPS {avg(group, 'gps'):.0f}")

    # --- Individual signal details ---
    lines.append("")
    lines.append("=== SIGNAL DETAILS ===")
    for e in resolved:
        icon = "HIT" if e.get("outcome_full") == "HIT" else "MISS"
        gm = e.get("goal_minute_full") or ""
        gm_str = f" goal@{gm}'" if gm else ""
        trigger = "GPS" if e.get("gps_triggered") else "SOT"
        rr = e.get("recency_ratio")
        rr_str = f" RR:{rr:.2f}" if rr is not None else ""
        score = f" [{e.get('goals_at_signal', '?')}-{e.get('opponent_goals_at_signal', '?')}]"
        ha = "H" if e.get("is_home") else "A"
        lines.append(
            f"  {icon:4s} | {e.get('team_name', '?'):<20s} | {e.get('league', '?'):<25s} | "
            f"{e.get('game_minute', '?'):>2}' {ha} | GPS {e.get('gps', 0):.0f} | SOT {e.get('sot', 0)} | "
            f"{trigger} | {e.get('tier', '?'):<13s} | full:{e.get('outcome_full', '?')} "
            f"15m:{e.get('outcome_15min', '?')}{gm_str}{rr_str}{score}"
        )

    if pending:
        lines.append("")
        lines.append(f"=== PENDING ({len(pending)}) ===")
        for e in pending:
            lines.append(
                f"  ... | {e.get('team_name', '?'):<20s} | {e.get('league', '?'):<25s} | "
                f"{e.get('game_minute', '?'):>2}' | GPS {e.get('gps', 0):.0f} | SOT {e.get('sot', 0)}"
            )

    return lines


# ============================================================
# PRESSURE POLLS ANALYSIS
# ============================================================

def analyze_polls(polls: list[dict]) -> list[str]:
    """Generate poll-level analysis report."""
    lines = []
    if not polls:
        lines.append("No poll data for this period.")
        return lines

    # Basic stats
    fixtures = set(p.get("fixture_id") for p in polls)
    teams = set((p.get("fixture_id"), p.get("team_id")) for p in polls)
    leagues = set(p.get("league") for p in polls if p.get("league"))
    minutes_covered = [p.get("minute", 0) for p in polls]

    lines.append(f"POLLS: {len(polls)} total | {len(fixtures)} fixtures | {len(teams)} team-sides | {len(leagues)} leagues")
    lines.append(f"  Minute range: {min(minutes_covered)}' - {max(minutes_covered)}'")
    lines.append(f"  Avg polls/team-side: {len(polls)/max(len(teams),1):.1f}")

    # GPS distribution
    gps_vals = [p.get("gps", 0) for p in polls]
    lines.append("")
    lines.append("=== GPS DISTRIBUTION ===")
    for label, lo, hi in [("0-39 (none)", 0, 40), ("40-54 (BUILDING)", 40, 55),
                           ("55-74 (EARLY WARN)", 55, 75), ("75+ (CRITICAL)", 75, 200)]:
        count = sum(1 for g in gps_vals if lo <= g < hi)
        pct = count / len(gps_vals) * 100 if gps_vals else 0
        avg_g = sum(g for g in gps_vals if lo <= g < hi) / count if count else 0
        lines.append(f"  {label}: {count} ({pct:.0f}%) avg {avg_g:.0f}")

    # Peak GPS per fixture-team
    lines.append("")
    lines.append("=== PEAK GPS PER FIXTURE-TEAM ===")
    peak_by_team = defaultdict(lambda: {"gps": 0, "minute": 0, "sot": 0, "league": ""})
    for p in polls:
        key = (p.get("fixture_id"), p.get("team_id"), p.get("team_name", "?"))
        if p.get("gps", 0) > peak_by_team[key]["gps"]:
            peak_by_team[key] = {
                "gps": p.get("gps", 0),
                "minute": p.get("minute", 0),
                "sot": p.get("sot", 0),
                "league": p.get("league", ""),
                "xg": p.get("xg"),
                "ib": p.get("shots_inside_box", 0),
                "accel": p.get("accel_count", 0),
            }
    # Sort by GPS descending
    sorted_peaks = sorted(peak_by_team.items(), key=lambda x: x[1]["gps"], reverse=True)
    for (fid, tid, tname), peak in sorted_peaks[:20]:
        xg_str = f" xG:{peak['xg']}" if peak.get("xg") is not None else ""
        accel_str = f" accel:{peak['accel']}" if peak.get("accel", 0) > 0 else ""
        lines.append(
            f"  GPS {peak['gps']:5.1f} | {tname:<20s} | {peak['league']:<25s} | "
            f"{peak['minute']:>2}' | SOT {peak['sot']} | IB {peak['ib']}{xg_str}{accel_str}"
        )

    # Acceleration events (polls where accel_count > 0)
    accel_polls = [p for p in polls if p.get("accel_count", 0) > 0]
    if accel_polls:
        lines.append("")
        lines.append(f"=== ACCELERATION EVENTS: {len(accel_polls)} polls ===")
        # Group by fixture-team, count acceleration polls
        accel_by_team = defaultdict(int)
        for p in accel_polls:
            key = (p.get("fixture_id"), p.get("team_id"), p.get("team_name", "?"))
            accel_by_team[key] += 1
        sorted_accel = sorted(accel_by_team.items(), key=lambda x: x[1], reverse=True)
        for (fid, tid, tname), count in sorted_accel[:10]:
            lines.append(f"  {tname:<20s}: {count} accelerating polls")

    # GPS component analysis (avg values at different GPS levels)
    lines.append("")
    lines.append("=== COMPONENT BREAKDOWN AT SIGNAL LEVEL (GPS>=55) ===")
    signal_polls = [p for p in polls if p.get("gps", 0) >= 55]
    if signal_polls:
        for label, lo, hi in [("55-64", 55, 65), ("65-74", 65, 75), ("75-84", 75, 85), ("85+", 85, 200)]:
            group = [p for p in signal_polls if lo <= p.get("gps", 0) < hi]
            if not group:
                continue
            t = len(group)
            lines.append(f"  GPS {label} ({t} polls):")
            lines.append(f"    SOT: {avg(group, 'sot'):.1f} | IB: {avg(group, 'shots_inside_box'):.1f} | "
                          f"Shots: {avg(group, 'total_shots'):.1f} | xG: {avg(group, 'xg'):.2f} | "
                          f"Corners: {avg(group, 'corners'):.1f}")
            lines.append(f"    Components: sot={avg(group, 'gps_sot'):.1f} ib={avg(group, 'gps_ib'):.1f} "
                          f"sv={avg(group, 'gps_sv'):.1f} xg={avg(group, 'gps_xg'):.1f} "
                          f"accel={avg(group, 'gps_accel'):.1f}")

    # SOT progression over time (aggregated)
    lines.append("")
    lines.append("=== SOT PROGRESSION (avg by minute bucket) ===")
    minute_buckets_polls = [
        ("1-20'", 1, 21), ("21-35'", 21, 36), ("36-45'", 36, 46),
        ("46-55'", 46, 56), ("56-65'", 56, 66), ("66-75'", 66, 76),
        ("76-85'", 76, 86), ("86+", 86, 200),
    ]
    for label, lo, hi in minute_buckets_polls:
        group = [p for p in polls if lo <= p.get("minute", 0) < hi]
        if not group:
            continue
        t = len(group)
        lines.append(f"  {label:>5s}: SOT {avg(group, 'sot'):.1f} | GPS {avg(group, 'gps'):.0f} | "
                      f"IB {avg(group, 'shots_inside_box'):.1f} | xG {avg(group, 'xg'):.2f} | "
                      f"({t} polls)")

    # Data quality
    dq_vals = [p.get("data_quality", 0) for p in polls if p.get("data_quality") is not None]
    if dq_vals:
        lines.append("")
        lines.append(f"=== DATA QUALITY ===")
        lines.append(f"  Avg data quality: {sum(dq_vals)/len(dq_vals):.2f}")
        low_dq = sum(1 for d in dq_vals if d < 0.5)
        lines.append(f"  Low quality (<0.5): {low_dq}/{len(dq_vals)}")

    return lines


# ============================================================
# POLL-TO-GOAL CORRELATION (advanced backtesting)
# ============================================================

def analyze_poll_to_goal(polls: list[dict], signals: list[dict]) -> list[str]:
    """Cross-reference high-GPS polls with goals from signal outcomes.
    Answers: did any team reach GPS>=55 in polls but never got a signal,
    and then scored? These would be missed opportunities. Also shows the
    poll-to-goal timeline for teams that did score."""
    lines = []
    if not polls or not signals:
        return lines

    # Build goal timeline from all outcomes (HIT or MISS — we want ALL goals)
    goals_by_team: dict[tuple, list[int]] = defaultdict(list)
    for s in signals:
        if not s.get("resolved"):
            continue
        fid = s.get("fixture_id")
        tid = s.get("team_id")
        gm = s.get("goal_minute_full")
        if gm:
            # Only count if this team scored (HIT means they scored after signal)
            if s.get("outcome_full") == "HIT":
                goals_by_team[(fid, tid)].append(gm)

    if not goals_by_team:
        return lines

    # Build poll timeline per fixture-team
    poll_timeline: dict[tuple, list[dict]] = defaultdict(list)
    for p in polls:
        key = (p.get("fixture_id"), p.get("team_id"))
        poll_timeline[key].append(p)

    # For each team that scored, find their peak GPS before the goal
    lines.append("")
    lines.append("=== POLL-TO-GOAL TIMELINE ===")
    lines.append("  (peak GPS before each goal, from poll data)")

    entries = []
    for (fid, tid), goal_mins in sorted(goals_by_team.items()):
        team_polls = sorted(poll_timeline.get((fid, tid), []), key=lambda p: p.get("minute", 0))
        if not team_polls:
            continue
        team_name = team_polls[0].get("team_name", "?")
        league = team_polls[0].get("league", "?")
        for gm in sorted(goal_mins):
            # Find peak GPS in polls before the goal
            before_goal = [p for p in team_polls if p.get("minute", 0) <= gm]
            if before_goal:
                peak = max(before_goal, key=lambda p: p.get("gps", 0))
                entries.append({
                    "team": team_name, "league": league,
                    "goal_min": gm, "peak_gps": peak.get("gps", 0),
                    "peak_min": peak.get("minute", 0),
                    "sot_at_peak": peak.get("sot", 0),
                    "ib_at_peak": peak.get("shots_inside_box", 0),
                    "xg_at_peak": peak.get("xg"),
                    "accel_at_peak": peak.get("accel_count", 0),
                    "gap": gm - peak.get("minute", 0),
                })

    if not entries:
        lines.append("  (no poll data matching scored fixtures)")
        return lines

    for e in sorted(entries, key=lambda x: x["goal_min"]):
        xg_str = f" xG:{e['xg_at_peak']}" if e.get('xg_at_peak') is not None else ""
        accel_str = f" accel:{e['accel_at_peak']}" if e.get('accel_at_peak', 0) > 0 else ""
        lines.append(
            f"  Goal@{e['goal_min']:>2}' | {e['team']:<20s} | {e['league']:<25s} | "
            f"peak GPS {e['peak_gps']:5.1f}@{e['peak_min']:>2}' | "
            f"SOT {e['sot_at_peak']} | IB {e['ib_at_peak']}{xg_str}{accel_str} | "
            f"gap: {e['gap']}'"
        )

    return lines


# ============================================================
# DAILY SUMMARY FILE
# ============================================================

def write_daily_json_report(signals: list[dict], polls: list[dict],
                            target_date: str, output_dir: str) -> str:
    """Write a structured JSON daily report for programmatic consumption."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"eod_{target_date}.json")

    resolved = [s for s in signals if s.get("resolved")]
    first_resolved = [s for s in signals if s.get("sig_num", 1) == 1 and s.get("resolved")]

    report = {
        "date": target_date,
        "generated_at": datetime.now(BULGARIA_TZ).isoformat(),
        "signals": {
            "total": len(signals),
            "resolved": len(resolved),
            "pending": len(signals) - len(resolved),
            "first_signal_only": {
                "count": len(first_resolved),
                "full_wr": wr(hit_count(first_resolved, "outcome_full"), len(first_resolved)) if first_resolved else "N/A",
                "15m_wr": wr(hit_count(first_resolved, "outcome_15min"), len(first_resolved)) if first_resolved else "N/A",
                "5m_wr": wr(hit_count(first_resolved, "outcome_5min"), len(first_resolved)) if first_resolved else "N/A",
            },
            "all_resolved": {
                "count": len(resolved),
                "full_wr": wr(hit_count(resolved, "outcome_full"), len(resolved)) if resolved else "N/A",
                "15m_wr": wr(hit_count(resolved, "outcome_15min"), len(resolved)) if resolved else "N/A",
            },
            "by_tier": {},
            "by_gps_range": {},
            "by_minute": {},
            "details": resolved,
        },
        "polls": {
            "total": len(polls),
            "fixtures": len(set(p.get("fixture_id") for p in polls)),
            "team_sides": len(set((p.get("fixture_id"), p.get("team_id")) for p in polls)),
            "avg_gps": round(avg(polls, "gps"), 1) if polls else 0,
            "max_gps": round(max((p.get("gps", 0) for p in polls), default=0), 1),
            "avg_data_quality": round(avg(polls, "data_quality"), 2) if polls else 0,
        },
    }

    # By tier
    for tier in ("CRITICAL", "EARLY WARNING"):
        group = [e for e in resolved if e.get("tier") == tier]
        if group:
            t = len(group)
            report["signals"]["by_tier"][tier] = {
                "count": t,
                "full_wr": wr(hit_count(group, "outcome_full"), t),
                "15m_wr": wr(hit_count(group, "outcome_15min"), t),
                "avg_gps": round(avg(group, "gps"), 1),
            }

    # By GPS range
    for label, lo, hi in [("55-64", 55, 65), ("65-74", 65, 75),
                           ("75-84", 75, 85), ("85+", 85, 999)]:
        group = [e for e in resolved if lo <= e.get("gps", 0) < hi]
        if group:
            t = len(group)
            report["signals"]["by_gps_range"][label] = {
                "count": t,
                "full_wr": wr(hit_count(group, "outcome_full"), t),
                "15m_wr": wr(hit_count(group, "outcome_15min"), t),
                "avg_accel": round(avg(group, "accel_count"), 1),
            }

    # By minute
    for label, lo, hi in [("21-35'", 21, 36), ("36-45'", 36, 46),
                           ("46-55'", 46, 56), ("56-61'", 56, 62),
                           ("62-75'", 62, 76), ("76-85'", 76, 86)]:
        group = [e for e in resolved if lo <= e.get("game_minute", 0) < hi]
        if group:
            t = len(group)
            report["signals"]["by_minute"][label] = {
                "count": t,
                "full_wr": wr(hit_count(group, "outcome_full"), t),
                "15m_wr": wr(hit_count(group, "outcome_15min"), t),
                "avg_gps": round(avg(group, "gps"), 1),
            }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return filepath


# ============================================================
# TELEGRAM SENDING
# ============================================================

def send_to_telegram(text: str) -> bool:
    """Send text to Telegram. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.", file=sys.stderr)
        return False

    # Telegram message limit is 4096 chars; split if needed
    max_len = 4000
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]

    for i, chunk in enumerate(chunks):
        url = f"{TELEGRAM_API}/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=30)
            data = resp.json()
            if not data.get("ok"):
                print(f"Telegram error: {data}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Telegram send failed: {e}", file=sys.stderr)
            return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EOD data collection for Football Bot")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today Bulgaria time)")
    parser.add_argument("--days", type=int, default=None,
                        help="Last N days (overrides --date)")
    parser.add_argument("--send", action="store_true",
                        help="Send report to Telegram")
    parser.add_argument("--polls-only", action="store_true",
                        help="Only show poll-level analysis")
    parser.add_argument("--signals-only", action="store_true",
                        help="Only show signal analysis")
    parser.add_argument("--out", type=str, default=None,
                        help="Write text report to file")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write JSON report to directory")
    parser.add_argument("--data-dir", type=str, default=VOLUME_DIR,
                        help=f"Data directory (default: {VOLUME_DIR})")
    parser.add_argument("--quiet", action="store_true",
                        help="Don't print to stdout (useful with --send)")
    args = parser.parse_args()

    # Determine target date(s)
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")

    # Load data
    all_outcomes = load_jsonl(os.path.join(args.data_dir, "signal_outcomes.jsonl"))
    all_polls = load_jsonl(os.path.join(args.data_dir, "pressure_polls.jsonl"))

    # Filter
    if args.days:
        day_signals = filter_by_date_range(all_outcomes, args.days, get_date_key)
        day_polls = filter_by_date_range(all_polls, args.days, get_poll_date_key)
        date_label = f"last {args.days} days"
    else:
        day_signals = filter_by_date(all_outcomes, target_date, get_date_key)
        day_polls = filter_by_date(all_polls, target_date, get_poll_date_key)
        date_label = target_date

    # Generate report
    report_lines = []
    now_bg = datetime.now(BULGARIA_TZ)
    report_lines.append(f"EOD REPORT: {date_label}")
    report_lines.append(f"Generated: {now_bg.strftime('%Y-%m-%d %H:%M')} Bulgaria")
    report_lines.append(f"Data: {args.data_dir} ({len(all_outcomes)} total outcomes, {len(all_polls)} total polls)")
    report_lines.append("=" * 60)

    if not args.polls_only:
        report_lines.append("")
        report_lines.append(analyze_signals(day_signals, all_outcomes))

    if not args.signals_only:
        report_lines.append("")
        report_lines.append(analyze_polls(day_polls))

    # Cross-reference polls with goals (only when both data sources exist)
    if day_polls and day_signals:
        report_lines.append("")
        report_lines.append(analyze_poll_to_goal(day_polls, day_signals))

    # Flatten
    flat_lines = []
    for item in report_lines:
        if isinstance(item, list):
            flat_lines.extend(item)
        else:
            flat_lines.append(item)

    report_text = "\n".join(flat_lines)

    # Output
    if not args.quiet:
        print(report_text)

    # Write to file
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report_text + "\n")
        print(f"\nReport written to: {args.out}", file=sys.stderr)

    # Write JSON report
    if args.json_out:
        if args.days:
            # Generate one JSON per day
            for d in range(args.days - 1, -1, -1):
                day = (now_bg - timedelta(days=d)).strftime("%Y-%m-%d")
                d_signals = filter_by_date(all_outcomes, day, get_date_key)
                d_polls = filter_by_date(all_polls, day, get_poll_date_key)
                path = write_daily_json_report(d_signals, d_polls, day, args.json_out)
                print(f"JSON report: {path}", file=sys.stderr)
        else:
            path = write_daily_json_report(day_signals, day_polls, target_date, args.json_out)
            print(f"JSON report: {path}", file=sys.stderr)

    # Send to Telegram
    if args.send:
        if send_to_telegram(report_text):
            print(f"\nSent to Telegram ({len(report_text)} chars).", file=sys.stderr)
        else:
            print("\nFailed to send to Telegram.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
