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


def get_blocked_date_key(entry: dict) -> str:
    """v10.49: Extract YYYY-MM-DD from a blocked_outcomes entry."""
    clock = entry.get("blocked_clock", "")
    if clock:
        return clock[:10]
    ts = entry.get("blocked_time", 0)
    if ts:
        return datetime.fromtimestamp(ts, tz=BULGARIA_TZ).strftime("%Y-%m-%d")
    return ""


def get_shadow_date_key(entry: dict) -> str:
    """v10.50: Extract YYYY-MM-DD from a fastlane_shadow entry."""
    clock = entry.get("shadow_clock", "")
    if clock:
        return clock[:10]
    ts = entry.get("shadow_time", 0)
    if ts:
        return datetime.fromtimestamp(ts, tz=BULGARIA_TZ).strftime("%Y-%m-%d")
    return ""


def get_ratio_trial_date_key(entry: dict) -> str:
    """v10.117: Extract YYYY-MM-DD from a ratio_trial entry."""
    clock = entry.get("trial_clock", "")
    if clock:
        return clock[:10]
    ts = entry.get("trial_time", 0)
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


def compute_correlated_groups(signals: list[dict]) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """Group signals into correlated clusters.

    Grouping key: (fixture_id, team_id, goal_minute_full)
    - Signals pointing to the same goal event are in the same group.
    - Signals with goal_minute_full=None (no goal) are grouped by
      (fixture_id, team_id, None) — so multiple MISS signals on the same
      team-side are one group.
    - Signals pointing to different goal minutes are in different groups
      (e.g., Chelsea goal@42' vs goal@50' are separate events).

    Returns:
        groups:       group_id -> list of signal dicts
        group_info:   group_id -> {"team", "league", "n_signals", "goal_min", "n_distinct_goals"}
    """
    cluster_key_to_group: dict[tuple, int] = {}
    groups: dict[int, list[dict]] = {}
    group_info: dict[int, dict] = {}
    next_id = 1

    for s in signals:
        fid = s.get("fixture_id")
        tid = s.get("team_id")
        goal_min = s.get("goal_minute_full")  # None if no goal
        key = (fid, tid, goal_min)

        if key not in cluster_key_to_group:
            gid = next_id
            next_id += 1
            cluster_key_to_group[key] = gid
            groups[gid] = []
            group_info[gid] = {
                "team": s.get("team_name", "?"),
                "league": s.get("league", "?"),
                "fixture_id": fid,
                "team_id": tid,
                "n_signals": 0,
                "goal_min": goal_min,
            }

        gid = cluster_key_to_group[key]
        s["_corr_group_id"] = gid  # annotate in-place
        groups[gid].append(s)
        group_info[gid]["n_signals"] += 1

    # Compute distinct goals per fixture-team across all groups
    team_goals: dict[tuple, set] = defaultdict(set)
    for gid, info in group_info.items():
        fk = (info["fixture_id"], info["team_id"])
        if info["goal_min"] is not None:
            team_goals[fk].add(info["goal_min"])
    for gid, info in group_info.items():
        fk = (info["fixture_id"], info["team_id"])
        info["n_distinct_goals"] = len(team_goals[fk])

    return groups, group_info


def deduplicate_signals(signals: list[dict]) -> list[dict]:
    """Remove duplicate signal records.

    Duplicates can occur from:
    - Bot restarts losing in-memory stale-suppress state (pre-v10.24)
    - rewrite_outcomes_file() edge cases

    Dedup key: (fixture_id, team_id, game_minute) — keeps first occurrence.
    """
    seen = set()
    deduped = []
    for s in signals:
        key = (s.get("fixture_id"), s.get("team_id"), s.get("game_minute"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    removed = len(signals) - len(deduped)
    if removed > 0:
        print(f"  [dedup] Removed {removed} duplicate signal(s) ({len(signals)} -> {len(deduped)})",
              file=sys.stderr)
    return deduped


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
    """Generate full signal analysis report. Returns list of text lines.

    NOTE on "resolved": the bot's `resolved` flag is only set True once ALL
    four outcome windows (5/10/15min + full) are finalized. But outcome_full
    can be known (HIT, because a goal was detected) well before the shorter
    windows finish their own bookkeeping — e.g. a signal at 20' with a goal
    at 76' has outcome_full=HIT immediately, but resolved may still read
    False for a beat. Gating every stat in this report on the blanket
    `resolved` flag silently undercounts real, known outcomes (confirmed:
    3 Barcelona signals on 2026-08-31 had outcome_full="HIT" recorded but
    resolved=False, and were excluded from every total until this fix).
    We now treat a signal as "decided" for the full-match stat as soon as
    outcome_full is HIT or MISS, independent of the blanket flag.
    """
    lines = []
    resolved = [s for s in signals if s.get("outcome_full") in ("HIT", "MISS")]
    pending = [s for s in signals if s.get("outcome_full") not in ("HIT", "MISS")]
    first_only = [s for s in signals if s.get("sig_num") == 1]
    first_resolved = [s for s in first_only if s.get("outcome_full") in ("HIT", "MISS")]

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

    # --- Correlated signal analysis ---
    corr_groups, corr_info = compute_correlated_groups(resolved)
    multi_groups = {gid: gs for gid, gs in corr_groups.items() if len(gs) > 1}
    n_in_cluster = sum(len(gs) for gs in multi_groups.values())
    n_groups = len(corr_groups)
    n_multi = len(multi_groups)
    if n_multi > 0:
        lines.append("")
        lines.append(f"=== CORRELATED SIGNALS ===")
        lines.append(f"  {n_groups} unique goal events | {n_multi} clusters with 2+ signals | "
                      f"{n_in_cluster}/{len(resolved)} signals in clusters ({100*n_in_cluster/len(resolved):.0f}%)")
        # Deduplicated rates: one representative per group (first signal in each group)
        dedup_representatives = [gs[0] for gs in corr_groups.values()]
        dh15 = hit_count(dedup_representatives, "outcome_15min")
        dhf = hit_count(dedup_representatives, "outcome_full")
        lines.append(f"  Raw:          full {wr(hit_count(resolved, 'outcome_full'), len(resolved))} | "
                      f"15m {wr(hit_count(resolved, 'outcome_15min'), len(resolved))}")
        lines.append(f"  Deduplicated: full {wr(dhf, n_groups)} | "
                      f"15m {wr(dh15, n_groups)}")
        # Show clusters
        lines.append(f"  Clusters ({n_multi}):")
        for gid in sorted(multi_groups.keys()):
            gs = multi_groups[gid]
            info = corr_info[gid]
            goal_str = f"goal@{info['goal_min']}" if info['goal_min'] else "no goal"
            mins = ", ".join(str(s.get("game_minute", "?")) + "'" for s in gs)
            lines.append(f"    G{gid}: {info['team']:<20s} | {len(gs)} sigs @ {mins} | {goal_str}")

    # --- All-time signals win rates (NOT scoped to this report's date range —
    # kept separate and clearly labeled so it can't be mistaken for a same-day figure) ---
    if all_signals and len(all_signals) > total:
        all_resolved = [s for s in all_signals if s.get("outcome_full") in ("HIT", "MISS")]
        if all_resolved:
            lines.append("")
            lines.append(f"=== ALL-TIME HISTORY, ALL DAYS COMBINED ({len(all_resolved)} resolved) ===")
            lines.append(f"  (NOT scoped to this report's date range — full historical file, for long-run context only)")
            ah5 = hit_count(all_resolved, "outcome_5min")
            ah10 = hit_count(all_resolved, "outcome_10min")
            ah15 = hit_count(all_resolved, "outcome_15min")
            ahf = hit_count(all_resolved, "outcome_full")
            lines.append(f"  Full:    {wr(ahf, len(all_resolved))}")
            lines.append(f"  15-min:  {wr(ah15, len(all_resolved))}")
            lines.append(f"  10-min:  {wr(ah10, len(all_resolved))}")
            lines.append(f"  5-min:   {wr(ah5, len(all_resolved))}")

    # --- Per-team breakdown (most conservative read: collapses every repeat
    # signal for a team into one win/loss, so one hot team on one big night
    # can't quietly carry the whole day's headline number) ---
    by_team: dict[str, list[dict]] = defaultdict(list)
    for s in resolved:
        by_team[s.get("team_name", "?")].append(s)
    if by_team:
        lines.append("")
        lines.append(f"=== PER-TEAM (one result per team, {len(by_team)} teams) ===")
        team_wins = sum(1 for v in by_team.values() if any(x.get("outcome_full") == "HIT" for x in v))
        lines.append(f"  Team-level win rate: {wr(team_wins, len(by_team))}")
        # Flag concentration: any team contributing >25% of the day's total wins
        total_wins = hit_count(resolved, "outcome_full")
        if total_wins:
            for team, sigs in sorted(by_team.items(), key=lambda kv: -len(kv[1])):
                team_hits = hit_count(sigs, "outcome_full")
                if team_hits and team_hits / total_wins >= 0.25:
                    lines.append(
                        f"  ⚠ {team}: {len(sigs)} signal(s), {team_hits} win(s) "
                        f"= {100*team_hits/total_wins:.0f}% of today's total wins — "
                        f"headline rate is sensitive to this one team"
                    )

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

    # --- v10.116: BOX-EDGE SHADOW (research only — signals are never gated) ---
    # Backtest basis (Sep 12-15, 162 resolved signals): losing the box-shot
    # battle hit 18% full-window vs 69% for box-dominant signals. This section
    # regrades that counterfactual on LIVE stamps every night; the live veto
    # is a separate future change (n >= 50 stamped per bucket, >= 20pp full-WR
    # gap held across >= 2 match-weeks).
    stamped = [e for e in resolved if e.get("boxedge_would_veto") is not None]
    if stamped:
        def _wr2(g: list[dict]) -> str:
            if not g:
                return "n/a yet"
            return (f"full {wr(hit_count(g, 'outcome_full'), len(g))} | "
                    f"15m {wr(hit_count(g, 'outcome_15min'), len(g))}")
        veto = [e for e in stamped if e.get("boxedge_would_veto")]
        kept = [e for e in stamped if not e.get("boxedge_would_veto")]
        boost = [e for e in stamped if e.get("boxedge_boost")]
        offspike = [e for e in stamped if (e.get("offside_delta_10") or 0) >= 2]
        opplead = [e for e in stamped if e.get("opp_press_while_leading")]
        lines.append("")
        lines.append("=== BOX-EDGE SHADOW (v10.116 — research only, never gated) ===")
        lines.append(f"  Stamped signals: {len(stamped)} (stamps began with v10.116)")
        lines.append(f"  would-VETO (dom<=-1 or share<40%): {_wr2(veto)}")
        lines.append(f"  kept (rest):                    {_wr2(kept)}")
        lines.append(f"  BOOST (dom>=+5 or share>=85%):  {_wr2(boost)}")
        lines.append(f"  offside spike >=2 in 10':       {_wr2(offspike)}")
        lines.append(f"  opp pressing, we lead/level:   {_wr2(opplead)}")
        lines.append("  LIVE-VETO RULE: promote only after >=50 veto-stamped signals")
        lines.append("  AND a >=20pp full-WR gap (veto vs kept) held across 2+ weeks")
    else:
        lines.append("")
        lines.append("=== BOX-EDGE SHADOW: no stamped signals yet (v10.116 field) ===")

    # --- v10.36: Market / EV analysis (v10.48: chat-safe wording) ---
    # v10.77: P&L-GRADE filter — grade ONLY genuine live prices.
    # prematch_fallback stale prices (the Sep-4/Sep-6 garbage class that
    # inflated paper P&L) stay in the record for research but are EXCLUDED
    # here. Old records without odds_pnl_grade derive it from
    # odds_source/odds_suspect (backward compatible).
    def _pnl_grade(e):
        if "odds_pnl_grade" in e:
            return bool(e.get("odds_pnl_grade"))
        # v10.111: feed-2 live prices + manual /price receipts are P&L-grade
        return (e.get("odds_source") in ("live", "oddsapi_live", "manual")
                and not e.get("odds_suspect"))

    with_odds_all = [e for e in resolved if e.get("odds_over_odds") is not None]
    with_odds = [e for e in with_odds_all if _pnl_grade(e)]
    stale_n = len(with_odds_all) - len(with_odds)
    if with_odds_all and len(with_odds) < 3 and stale_n:
        lines.append("")
        lines.append(
            f"=== MARKET / EV ANALYSIS — SKIPPED ({stale_n} stale pre-match "
            f"price(s) excluded, only {len(with_odds)} P&L-grade live "
            f"price(s), 3 needed) ===")
        lines.append("  v10.111: paper P&L needs live odds (odds_source")
        lines.append("  live/oddsapi_live) or a manual /price receipt;")
        lines.append("  pre-match fallback prices are not P&L-grade.")
    if len(with_odds) >= 3:
        lines.append("")
        lines.append(f"=== MARKET / EV ANALYSIS ({len(with_odds)} P&L-grade live prices"
                     + (f"; {stale_n} stale pre-match excluded" if stale_n else "")
                     + ") ===")
        avg_over = avg(with_odds, "odds_over_odds")
        avg_impl = avg(with_odds, "odds_over_implied")
        hits_odds = [e for e in with_odds if e.get("outcome_full") == "HIT"]
        empirical_wr = len(hits_odds) / len(with_odds) if with_odds else 0
        lines.append(f" Avg Over price: {avg_over:.2f} | Avg implied: {avg_impl:.1%}")
        lines.append(f"  Empirical full WR: {empirical_wr:.1%}")
        edge = empirical_wr - avg_impl
        lines.append(f"  Edge vs market: {edge:+.1%} ({'+EV' if edge > 0 else '-EV'})")
        # ROI calculation (flat 1 unit stake)
        profit = sum(1 if e.get("outcome_full") == "HIT" else -1 for e in with_odds)
        lines.append(f"  Flat ROI: {profit / len(with_odds) * 100:+.1f}% ({profit:+d} units)")

        # By GPS range with odds
        for label, lo, hi in [("55-64", 55, 65), ("65-74", 65, 75),
                               ("75-84", 75, 85), ("85+", 85, 999)]:
            group = [e for e in with_odds if lo <= e.get("gps", 0) < hi]
            if len(group) < 2:
                continue
            g = len(group)
            gh = hit_count(group, "outcome_full")
            go = avg(group, "odds_over_odds")
            gi = avg(group, "odds_over_implied")
            ge = gh / g - gi if g else 0
            gp = (sum(1 if e.get("outcome_full") == "HIT" else -1 for e in group)) / g * 100
            lines.append(f"  GPS {label}: {gh}/{g} ({gh/g*100:.0f}%) @ {go:.2f} impl {gi:.1%} "
                          f"edge {ge:+.1%} ROI {gp:+.1f}%")

    # --- v10.114: SHADOW LEAGUES — trial leagues (logged, never sent) ---
    # Austria / Switzerland / Norway / Sweden signals carry shadow_league=True
    # in the ledger; this section grades them nightly so promotion is a data
    # decision (n>=15 & WR>=65% at live prices), never a guess.
    shadow = [e for e in resolved if e.get("shadow_league")]
    if shadow:
        lines.append("")
        lines.append("=== SHADOW LEAGUES (trial — signals logged, NOT sent) ===")
        by_lg: dict[str, list[dict]] = {}
        for e in shadow:
            by_lg.setdefault(e.get("league", "?"), []).append(e)
        for lg, es in sorted(by_lg.items(), key=lambda kv: -len(kv[1])):
            h = hit_count(es, "outcome_full")
            n = len(es)
            live_n = sum(
                1 for e in es
                if e.get("odds_source") in ("live", "oddsapi_live")
            )
            lines.append(
                f"  {lg}: {h}/{n} ({100 * h / n:.0f}%) full-match | "
                f"{live_n}/{n} live-priced | promote at n>=15 & WR>=65%"
            )
        lines.append("  (paper signals accumulate conversion stats here;")
        lines.append("   flip the league out of SHADOW_LEAGUES to start sending)")

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
        odds_str = ""
        if e.get("odds_over_odds"):
            odds_str = f" O{e.get('odds_over_line')}@{e['odds_over_odds']}"
        # Correlated group annotation
        gid = e.get("_corr_group_id")
        group_size = len(corr_groups.get(gid, [])) if gid else 1
        cluster_tag = f" G{gid}" if group_size > 1 else ""
        lines.append(
            f"  {icon:4s} | {e.get('team_name', '?'):<20s} | {e.get('league', '?'):<25s} | "
            f"{e.get('game_minute', '?'):>2}' {ha} | GPS {e.get('gps', 0):.0f} | SOT {e.get('sot', 0)} | "
            f"{trigger} | {e.get('tier', '?'):<13s} | full:{e.get('outcome_full', '?')} "
            f"15m:{e.get('outcome_15min', '?')}{gm_str}{rr_str}{odds_str}{cluster_tag}{score}"
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
    goals_by_team: dict[tuple, set[int]] = defaultdict(set)
    for s in signals:
        if s.get("outcome_full") not in ("HIT", "MISS"):
            continue
        fid = s.get("fixture_id")
        tid = s.get("team_id")
        gm = s.get("goal_minute_full")
        if gm:
            # Only count if this team scored (HIT means they scored after signal)
            if s.get("outcome_full") == "HIT":
                goals_by_team[(fid, tid)].add(gm)

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
# v10.50: FAST-LANE SHADOW ANALYSIS (virtual signals — never sent)
# ============================================================

def analyze_fastlane(shadows: list[dict], day_signals: list[dict]) -> list[str]:
    """v10.50: What would firing signals from the events feed have produced?

    Shadow records = virtual signals the fast lane detected from the events
    feed (2-3 min ahead of stats) but NEVER sent. This section compares them
    with real outcomes AND with the real signals that actually fired:
      - WR (5/10/15/full) of the virtual signals
      - speed gain: seconds between shadow fire-time and the real signal for
        the same team (when both fired) — the measurable speed win
      - pure-speed wins: shadows with NO real signal within 5 min after
        (the stats path never fired — live fast-lane firing would have been
        the ONLY warning)
      - duplicates: real signal arrived within 5 min anyway
    Promotion criteria (Phase 2, per FASTLANE_PROPOSAL.md): 100+ resolved
    shadows, 15' WR >= stats-path WR - 5pts, median speed gain >= 30s.
    """
    lines = ["", "=== FAST-LANE SHADOW ANALYSIS (virtual signals, never sent) ==="]
    if not shadows:
        lines.append("  (no shadow records today — fast lane saw no qualifying bursts)")
        return lines

    lines.append(f"  Shadow signals recorded: {len(shadows)}")
    resolved = [s for s in shadows if s.get("outcome_full") in ("HIT", "MISS")]
    if not resolved:
        lines.append("  none resolved yet — outcomes appear after matches finish")
        return lines

    total = len(resolved)
    h5 = hit_count(resolved, "outcome_5min")
    h10 = hit_count(resolved, "outcome_10min")
    h15 = hit_count(resolved, "outcome_15min")
    hf = hit_count(resolved, "outcome_full")
    lines.append(f"  Resolved: {total} | 5': {wr(h5, total)} | 10': {wr(h10, total)} | 15': {wr(h15, total)} | full: {wr(hf, total)}")

    # By trigger type
    by_trig: dict[str, list[dict]] = defaultdict(list)
    for s in resolved:
        by_trig[s.get("trigger", "?")].append(s)
    if len(by_trig) > 1:
        lines.append("  By trigger:")
        for trig, entries in sorted(by_trig.items()):
            n = len(entries)
            t15 = hit_count(entries, "outcome_15min")
            tf = hit_count(entries, "outcome_full")
            lines.append(f"    {trig}: {n} | 15' {wr(t15, n)} | full {wr(tf, n)}")

    # Events-ahead share (data quality: events feed led stats at fire time)
    _ahead = [s for s in resolved if (s.get("lag_sot") or 0) > 0]
    if _ahead:
        lines.append(f"  Events ahead of stats at fire: {len(_ahead)}/{total} ({100 * len(_ahead) // total}%)")

    # --- Join with REAL signals: speed gain + duplicates ---
    sig_by_team: dict[tuple, list[float]] = defaultdict(list)
    for s in day_signals:
        st = s.get("signal_time") or 0
        if st:
            sig_by_team[(s.get("fixture_id"), s.get("team_id"))].append(float(st))

    duplicates = 0
    pure_speed = 0
    gains: list[float] = []
    for sh in shadows:
        sh_t = float(sh.get("shadow_time") or 0)
        if not sh_t:
            continue
        later_sigs = [t for t in sig_by_team.get((sh.get("fixture_id"), sh.get("team_id")), [])
                      if sh_t < t <= sh_t + 300]
        if later_sigs:
            duplicates += 1
            gains.append(min(later_sigs) - sh_t)
        else:
            pure_speed += 1

    lines.append("")
    lines.append("  vs REAL signals (same team, real signal within 5 min after shadow):")
    lines.append(f"    duplicates (real signal fired anyway): {duplicates}")
    lines.append(f"    pure-speed (stats path NEVER fired): {pure_speed}")
    if gains:
        gains.sort()
        med = gains[len(gains) // 2]
        lines.append(f"    speed gain when duplicated: median {med:.0f}s | max {max(gains):.0f}s (shadow led real signal)")

    # --- Promotion criteria scoreboard (Phase 2 decision) ---
    lines.append("")
    lines.append("  PHASE-2 PROMOTION CRITERIA (need ALL before live firing):")
    real_res = [s for s in day_signals if s.get("outcome_15min") in ("HIT", "MISS")]
    if real_res:
        stats_wr15 = 100.0 * hit_count(real_res, "outcome_15min") / len(real_res)
        shadow_wr15 = 100.0 * h15 / total
        target = stats_wr15 - 5
        ok = shadow_wr15 >= target
        lines.append(
            f"    15' WR: shadow {shadow_wr15:.0f}% vs stats-path {stats_wr15:.0f}% "
            f"(target >= {target:.0f}%): {'PASS' if ok else 'not yet'}"
        )
    n_target = 100
    lines.append(f"    volume: {total}/{n_target} resolved shadows: {'PASS' if total >= n_target else 'accumulating'}")
    if gains:
        med = sorted(gains)[len(gains) // 2]
        lines.append(f"    speed: median gain {med:.0f}s (target >= 30s): {'PASS' if med >= 30 else 'not yet'}")
    lines.append("  NOTE: shadow signals are never sent; this is measurement only.")
    return lines


# ============================================================
# v10.117: RATIO-TRIAL ANALYSIS (tagged trial alerts — never gate signals)
# ============================================================

def _rt117_series(day_polls: list[dict]) -> dict:
    """v10.117: (fid, tid) -> poll list sorted by minute."""
    series: dict = defaultdict(list)
    for p in day_polls:
        fid, tid, m = p.get("fixture_id"), p.get("team_id"), p.get("minute")
        if fid is None or tid is None or m is None:
            continue
        series[(fid, tid)].append(p)
    for k in series:
        series[k].sort(key=lambda p: p.get("minute", 0))
    return series


def _rt117_value_at(seq: list[dict], field: str, minute: int) -> int | None:
    """v10.117: max-smoothed counter value at the latest poll <= minute.

    field='own_goals' resolves the team's own scoreline (goal outcomes).
    """
    best = None
    for p in seq:
        if p.get("minute", 0) > minute:
            break
        if field == "own_goals":
            v = (p.get("score_home") if p.get("is_home") else p.get("score_away")) or 0
        else:
            v = p.get(field)
        if v is not None:
            best = v if best is None else max(best, v)
    return best


def _rt117_added(seq: list[dict], field: str, m0: int, m1: int):
    """v10.117: max-smoothed increase of a counter in (m0, m1].

    Returns (added, covered) — covered=False when the ledger never observes
    the window tail (bot drops fixtures ~85'), i.e. the outcome is censored.
    """
    at0 = _rt117_value_at(seq, field, m0)
    mx = None
    for p in seq:
        pm = p.get("minute", 0)
        if pm <= m0:
            continue
        if pm > m1:
            break
        v = ((p.get("score_home") if p.get("is_home") else p.get("score_away")) or 0) \
            if field == "own_goals" else p.get(field)
        if v is not None:
            mx = v if mx is None else max(mx, v)
    last_m = seq[-1].get("minute", 0) if seq else 0
    covered = last_m >= m1
    if at0 is None or mx is None:
        return (None, covered)
    return (max(0, mx - at0), covered)


def analyze_ratio_trials(trials: list[dict], day_signals: list[dict],
                         day_polls: list[dict]) -> list[str]:
    """v10.117: Grade the RATIO-TRIAL alerts + the signal-time ratio stamps.

    Trials are tagged live alerts (offside-pressure / card-radar /
    corner-cluster) derived from the Sep 12-15 ratio lab. The bot resolves
    most records live; this pass closes anything still pending from the
    polls ledger (counter deltas, censored windows reported honestly),
    then grades per trigger and regrades the signal-stamp buckets nightly
    against the lab numbers. Promotion to standard signalling needs
    >= 50 records per trigger AND a held edge across 2+ match-weeks.
    """
    lines = ["", "=== v10.117 RATIO-TRIAL (offside-pressure / card-radar / corner-cluster) ==="]

    # --- resolve still-pending records from the polls ledger ---
    series = _rt117_series(day_polls) if day_polls else {}
    n_censored = 0
    for e in trials:
        if e.get("outcome_15min") is not None:
            continue
        seq = series.get((e.get("fixture_id"), e.get("team_id")))
        if not seq:
            n_censored += 1
            continue
        m0, m1 = e.get("game_minute") or 0, (e.get("game_minute") or 0) + 15
        ttype = e.get("trial_type")
        field = {
            "offside_pressure": "own_goals",
            "card_debt": "yellow_cards",
            "card_lead_protect": "yellow_cards",
            "corner_cluster": "corners",
        }.get(ttype, "own_goals")
        added, covered = _rt117_added(seq, field, m0, m1)
        if not covered or added is None:
            n_censored += 1
            continue
        e["outcome_15min"] = "HIT" if added > 0 else "MISS"
        e["resolved"] = True
        # research extras from the ledger close
        e["goals_added_15"] = _rt117_added(seq, "own_goals", m0, m1)[0] or 0
        e["yc_added_15"] = _rt117_added(seq, "yellow_cards", m0, m1)[0] or 0
        e["corners_added_15"] = _rt117_added(seq, "corners", m0, m1)[0] or 0
        e["sot_added_15"] = _rt117_added(seq, "sot", m0, m1)[0] or 0

    # --- per-trigger grades ---
    if not trials:
        lines.append("  no trial records yet (v10.117 field)")
    else:
        graded = [e for e in trials if e.get("outcome_15min") in ("HIT", "MISS")]
        sent = [e for e in trials if e.get("sent_live")]
        lines.append(
            f"  records: {len(trials)} ({len(sent)} sent live, "
            f"{len(graded)} graded, {n_censored} censored-window)"
        )
        for ttype, label in [
            ("offside_pressure", "OFFSIDE-PRESSURE (2+ offsides/10')"),
            ("card_debt", "CARD-RADAR debt (5+ fouls since card)"),
            ("card_lead_protect", "CARD-RADAR lead-protect (1-2 & 60'+)"),
            ("corner_cluster", "CORNER-CLUSTER (3+ corners/10')"),
        ]:
            grp = [e for e in graded if e.get("trial_type") == ttype]
            if not grp:
                lines.append(f"  {label}: none yet")
                continue
            h15 = hit_count(grp, "outcome_15min")
            extra = ""
            if ttype == "corner_cluster":
                c2 = sum(1 for e in grp if (e.get("corners_added_15") or 0) >= 2)
                s1 = sum(1 for e in grp if (e.get("sot_added_15") or 0) >= 1)
                extra = (f" | 2+ corners {c2}/{len(grp)} ({c2/len(grp)*100:.0f}%)"
                         f" | next SOT {s1}/{len(grp)} ({s1/len(grp)*100:.0f}%)")
            if ttype == "offside_pressure":
                hf = hit_count(grp, "outcome_full")
                extra = f" | full {wr(hf, len(grp))}"
            lines.append(
                f"  {label}: 15' {wr(h15, len(grp))} (n={len(grp)}){extra}"
            )
        lines.append("  LAB BASELINES: offside 57% / card debt 37% / protect 36% / corners 34%+59% SOT")
        lines.append("  PROMOTION RULE: >= 50 records per trigger AND a held edge across 2+ match-weeks")

    # --- signal-time ratio stamps (v10.117 fields on signal records) ---
    resolved = [s for s in day_signals if s.get("outcome_full") in ("HIT", "MISS")]
    stamped = [s for s in resolved if s.get("corner_vel_10") is not None
               or s.get("fouls_since_card") is not None]
    if stamped:
        def _bucket(fn, label) -> None:
            grp = [s for s in stamped if fn(s)]
            rest = [s for s in stamped if not fn(s)]
            if not grp or not rest:
                return
            lines.append(
                f"  {label}: 15' {wr(hit_count(grp, 'outcome_15min'), len(grp))} "
                f"| full {wr(hit_count(grp, 'outcome_full'), len(grp))} "
                f"(n={len(grp)}) vs rest "
                f"{wr(hit_count(rest, 'outcome_15min'), len(rest))} / "
                f"{wr(hit_count(rest, 'outcome_full'), len(rest))} (n={len(rest)})"
            )
        lines.append("  --- signal stamps (lab: offside>=2 71% full; convdebt>=4.5 +8pp; savestorm -19pp) ---")
        _bucket(lambda s: (s.get("offside_delta_10") or 0) >= 2, "offside push >=2/10'")
        _bucket(lambda s: (s.get("corner_vel_10") or 0) >= 2, "corner vel >=2/10'")
        _bucket(lambda s: (s.get("fouls_since_card") or 0) >= 5, "foul debt >=5")
        _bucket(lambda s: (s.get("conversion_debt") or -9) >= 4.5, "conversion debt >=4.5")
        _bucket(lambda s: (s.get("save_storm_10") or 0) >= 2, "save storm >=2 (veto?)")
        _bucket(lambda s: bool(s.get("lead_protect_60")), "lead-protect 60'+")
    else:
        lines.append("  no v10.117 signal stamps yet (stamps began with v10.117)")
    return lines


# ============================================================
# v10.49: FALSE-NEGATIVE (BLOCKED-SIGNAL) ANALYSIS
# ============================================================

def analyze_blocked(blocked: list[dict]) -> list[str]:
    """v10.49: What did the gates cost us?

    Blocked candidates = moments where a QUALIFYING signal (tier assigned)
    was suppressed by a policy gate. A HIT here means the team scored
    AFTER the block — the gate potentially cost a winning signal (false
    negative). Bot-side dedupe: max 1 record per (fixture, team, reason)
    per 10 game-minutes, cap 600/day.
    """
    lines = ["", "=== FALSE-NEGATIVE ANALYSIS (blocked-signal candidates) ==="]
    if not blocked:
        lines.append("  (no blocked-candidate records — file missing or no blocks today)")
        return lines

    resolved = [b for b in blocked if b.get("outcome_full") in ("HIT", "MISS")]
    if not resolved:
        lines.append(f"  {len(blocked)} candidate(s) recorded, none resolved yet")
        return lines

    total = len(resolved)
    h15 = hit_count(resolved, "outcome_15min")
    hf = hit_count(resolved, "outcome_full")
    lines.append(f"  Resolved candidates: {total}")
    lines.append(f"  Blocked-then-scored 15': {wr(h15, total)}")
    lines.append(f"  Blocked-then-scored full: {wr(hf, total)}")
    lines.append("")

    # By gate reason — sorted by full HIT count descending (worst cost first)
    by_reason = defaultdict(list)
    for b in resolved:
        by_reason[b.get("block_reason", "?")].append(b)

    ranked = sorted(by_reason.items(), key=lambda kv: -hit_count(kv[1], "outcome_full"))
    lines.append("  By gate (n | 15' HIT | full HIT | avg GPS) — worst cost first:")
    for reason, entries in ranked:
        n = len(entries)
        r_h15 = hit_count(entries, "outcome_15min")
        r_hf = hit_count(entries, "outcome_full")
        avg_gps = avg(entries, "gps")
        lines.append(f"    {reason}: {n} | {r_h15} | {r_hf} | {avg_gps:.0f}")

    lines.append("")
    lines.append("  NOTE: a HIT means the team scored after the block. Gates with high HIT")
    lines.append("  rates are tuning candidates — but volume gates (cooldown, stale, first-only)")
    lines.append("  block repeats by design; compare signal-level WR before changing any gate.")
    return lines


# ============================================================
# v10.49: POISSON CALIBRATION ANALYSIS
# ============================================================

def analyze_calibration(day_signals: list[dict], data_dir: str) -> list[str]:
    """v10.49: Predicted vs actual total goals — is GOALS_PER_SOT=0.31 right?

    Prefers the bot-maintained poisson_calibration.json (cumulative, all days).
    Falls back to computing from resolved signal outcomes when the file is
    absent. Bias = actual/predicted (>1 = model under-predicts totals in
    that league). For 'est' xg-source rows, implied calibrated constant =
    0.31 * bias (the candidate GOALS_PER_SOT once n >= 30 per league).
    """
    lines = ["", "=== POISSON CALIBRATION (predicted vs actual totals) ==="]
    calib_path = os.path.join(data_dir, "poisson_calibration.json")

    rows: list[dict] = []
    src_note = ""
    if os.path.exists(calib_path):
        try:
            with open(calib_path) as f:
                calib = json.load(f)
            if isinstance(calib, dict) and calib:
                rows = list(calib.values())
                src_note = "cumulative file"
        except Exception:
            rows = []
    if not rows:
        # Fallback: compute from resolved outcomes (today / selected range)
        by_key = defaultdict(list)
        for s in day_signals:
            pred = s.get("pred_expected_total")
            actual = s.get("pred_actual_total_goals")
            if pred is None or actual is None:
                continue
            key = f"{s.get('league') or 'UNKNOWN'}|{s.get('pred_xg_source') or 'unknown'}"
            by_key[key].append((float(pred), float(actual)))
        for key, vals in by_key.items():
            lg, src = key.split("|", 1)
            rows.append({
                "league": lg, "xg_source": src, "n": len(vals),
                "pred_sum": sum(v[0] for v in vals),
                "actual_sum": sum(v[1] for v in vals),
            })
        src_note = "today's outcomes" if rows else ""

    if not rows:
        lines.append("  (no calibration data yet — needs resolved signals with predictions)")
        return lines

    lines.append(f"  Source: {src_note} | GOALS_PER_SOT global constant: 0.31")
    lines.append("  league | xg src | n | avg pred | avg actual | bias(act/pred)")
    for r in sorted(rows, key=lambda r: -(r.get("n") or 0)):
        n = r.get("n") or 0
        if n < 1:
            continue
        pred_sum = r.get("pred_sum") or 0.0
        act_sum = r.get("actual_sum") or 0.0
        bias = (act_sum / pred_sum) if pred_sum > 0 else None
        bias_str = f"{bias:.2f}" if bias is not None else "n/a"
        implied = ""
        if bias is not None and r.get("xg_source") == "est":
            implied = f" | implied GPS/SOT: {0.31 * bias:.2f}"
        lines.append(
            f"    {r.get('league')} | {r.get('xg_source')} | {n} | "
            f"{pred_sum / n:.1f} | {act_sum / n:.1f} | {bias_str}{implied}"
        )
    lines.append("  NOTE: bias > 1 = model under-predicts totals (league scores more than")
    lines.append("  predicted). 'est' rows derive xG from SOT*0.31 — their implied constant")
    lines.append("  is the calibrated GOALS_PER_SOT candidate once n >= 30 per league.")
    lines.append("  Logging only — no auto-change to predictions.")
    return lines


# ============================================================
# DAILY SUMMARY FILE
# ============================================================

def write_daily_json_report(signals: list[dict], polls: list[dict],
                            target_date: str, output_dir: str) -> str:
    """Write a structured JSON daily report for programmatic consumption."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"eod_{target_date}.json")

    resolved = [s for s in signals if s.get("outcome_full") in ("HIT", "MISS")]
    first_resolved = [s for s in signals if s.get("sig_num", 1) == 1 and s.get("outcome_full") in ("HIT", "MISS")]

    # Correlated signal deduplication
    corr_groups, corr_info = compute_correlated_groups(resolved)
    dedup_reps = [gs[0] for gs in corr_groups.values()]
    multi_groups = {gid: gs for gid, gs in corr_groups.items() if len(gs) > 1}
    n_in_cluster = sum(len(gs) for gs in multi_groups.values())

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
            "deduplicated": {
                "n_unique_events": len(corr_groups),
                "n_clusters_multi": len(multi_groups),
                "signals_in_clusters": n_in_cluster,
                "pct_in_clusters": round(100 * n_in_cluster / len(resolved), 1) if resolved else 0,
                "full_wr": wr(hit_count(dedup_reps, "outcome_full"), len(dedup_reps)) if dedup_reps else "N/A",
                "15m_wr": wr(hit_count(dedup_reps, "outcome_15min"), len(dedup_reps)) if dedup_reps else "N/A",
                "clusters": [
                    {
                        "group_id": gid,
                        "team": corr_info[gid]["team"],
                        "n_signals": len(gs),
                        "goal_min": corr_info[gid]["goal_min"],
                        "signal_minutes": [s.get("game_minute") for s in gs],
                    }
                    for gid, gs in multi_groups.items()
                ],
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
    parser.add_argument("--all", action="store_true",
                        help="Show ALL data (YTD) with per-day breakdown")
    args = parser.parse_args()

    # Determine target date(s)
    if args.all:
        target_date = None
        date_label = "all data (YTD)"
    elif args.date:
        target_date = args.date
    else:
        target_date = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")

    # Load data
    all_outcomes = load_jsonl(os.path.join(args.data_dir, "signal_outcomes.jsonl"))
    all_polls = load_jsonl(os.path.join(args.data_dir, "pressure_polls.jsonl"))
    all_blocked = load_jsonl(os.path.join(args.data_dir, "blocked_outcomes.jsonl"))  # v10.49
    all_shadow = load_jsonl(os.path.join(args.data_dir, "fastlane_shadow.jsonl"))  # v10.50
    all_ratio = load_jsonl(os.path.join(args.data_dir, "ratio_trial.jsonl"))  # v10.117

    # Apply dedup to outcomes globally
    all_outcomes = deduplicate_signals(all_outcomes)

    # Filter
    if args.all:
        day_signals = all_outcomes
        day_polls = all_polls
        day_blocked = all_blocked  # v10.49
        day_shadow = all_shadow  # v10.50
        day_ratio = all_ratio  # v10.117
    elif args.days:
        day_signals = filter_by_date_range(all_outcomes, args.days, get_date_key)
        day_polls = filter_by_date_range(all_polls, args.days, get_poll_date_key)
        day_blocked = filter_by_date_range(all_blocked, args.days, get_blocked_date_key)  # v10.49
        day_shadow = filter_by_date_range(all_shadow, args.days, get_shadow_date_key)  # v10.50
        day_ratio = filter_by_date_range(all_ratio, args.days, get_ratio_trial_date_key)  # v10.117
        date_label = f"last {args.days} days"
    else:
        day_signals = filter_by_date(all_outcomes, target_date, get_date_key)
        day_polls = filter_by_date(all_polls, target_date, get_poll_date_key)
        day_blocked = filter_by_date(all_blocked, target_date, get_blocked_date_key)  # v10.49
        day_shadow = filter_by_date(all_shadow, target_date, get_shadow_date_key)  # v10.50
        day_ratio = filter_by_date(all_ratio, target_date, get_ratio_trial_date_key)  # v10.117
        date_label = target_date

    # Generate report
    report_lines = []
    now_bg = datetime.now(BULGARIA_TZ)
    report_lines.append(f"EOD REPORT: {date_label}")
    report_lines.append(f"Generated: {now_bg.strftime('%Y-%m-%d %H:%M')} Bulgaria")
    report_lines.append(f"Data: {args.data_dir} ({len(all_outcomes)} total outcomes, {len(all_polls)} total polls, {len(all_blocked)} blocked candidates, {len(all_shadow)} fast-lane shadows, {len(all_ratio)} ratio trials)")
    report_lines.append("=" * 60)

    # --- Per-day breakdown (only in --all mode) ---
    if args.all and day_signals:
        # Group signals by date
        by_date = defaultdict(list)
        for s in day_signals:
            dk = get_date_key(s)
            if dk:
                by_date[dk].append(s)

        sorted_dates = sorted(by_date.keys())
        report_lines.append("")
        report_lines.append(f"=== DAILY SUMMARY ({len(sorted_dates)} days) ===")
        for d in sorted_dates:
            d_sigs = by_date[d]
            d_resolved = [s for s in d_sigs if s.get("outcome_full") in ("HIT", "MISS")]
            d_first = [s for s in d_sigs if s.get("sig_num") == 1 and s.get("outcome_full") in ("HIT", "MISS")]
            d_fh = hit_count(d_first, "outcome_full") if d_first else 0
            d_f15 = hit_count(d_first, "outcome_15min") if d_first else 0
            d_ah = hit_count(d_resolved, "outcome_full")
            d_a15 = hit_count(d_resolved, "outcome_15min")
            # Count polls for this day
            d_polls = [p for p in all_polls if get_poll_date_key(p) == d]
            report_lines.append(
                f"  {d}: {len(d_sigs)} sigs ({len(d_resolved)} res) | "
                f"1st: {wr(d_fh, len(d_first))} | "
                f"all: {wr(d_ah, len(d_resolved))} | "
                f"{len(d_polls)} polls"
            )
        report_lines.append("")

    if not args.polls_only:
        report_lines.append("")
        report_lines.append(analyze_signals(day_signals, all_outcomes))
        # v10.49: false-negative section — what did the gates cost us?
        report_lines.append("")
        report_lines.append(analyze_blocked(day_blocked))
        # v10.50: fast-lane shadow section — what would events-feed firing have done?
        report_lines.append("")
        report_lines.append(analyze_fastlane(day_shadow, day_signals))
        # v10.117: ratio-trial section — offside-pressure / card-radar /
        # corner-cluster trials + the signal-time ratio stamps
        report_lines.append("")
        report_lines.append(analyze_ratio_trials(day_ratio, day_signals, day_polls))

    if not args.signals_only:
        report_lines.append("")
        report_lines.append(analyze_polls(day_polls))

    # Cross-reference polls with goals (only when both data sources exist)
    if day_polls and day_signals:
        report_lines.append("")
        report_lines.append(analyze_poll_to_goal(day_polls, day_signals))

    # v10.49: Poisson calibration section — is GOALS_PER_SOT=0.31 right per league?
    report_lines.append("")
    report_lines.append(analyze_calibration(day_signals, args.data_dir))

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
