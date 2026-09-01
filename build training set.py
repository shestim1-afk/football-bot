#!/usr/bin/env python3
"""
build_training_set.py
======================
Builds a labeled ML training set from pressure_polls.jsonl for goal-prediction
model training. This is poll-level data (every stats poll, not just fired
signals), which gives a much larger sample (tens of thousands of rows/day)
than signal_outcomes.jsonl (dozens/day).

WHY THIS EXISTS (context from prior analysis):
- The bot's hand-tuned GPS formula has never been validated against a proper
  ML baseline.
- signal_outcomes.jsonl is too small (~30-60 signals/day) to train on directly.
- pressure_polls.jsonl has every poll, but does NOT include goal event
  timestamps or the enriched recency fields (sot_5m_ago, recency_ratio, etc.)
  that signal_outcomes.jsonl computes at signal time — this script derives
  both from the raw poll timeline itself.

METHODOLOGY:
1. Detect goal events per (fixture_id, team_id) by finding score increases
   between consecutive polls, using the poll's own `minute` as the goal
   time (best available approximation from this data source).
2. For every poll, label:
     - label_15min: did this team's score increase within the next 15
       real game-minutes of this poll?
     - label_full:  did this team's score increase at any point after
       this poll, before the match ended?
3. Compute recency features (sot/xg/shots deltas over the last ~5 and ~10
   real minutes) directly from each team's own poll history, since the raw
   file doesn't pre-compute these.
4. CRITICAL — match-level train/test split, not row-level. Polls within one
   match are highly correlated (same team, same spell of pressure sampled
   every 10-60s) — a random row split would let near-duplicate rows leak
   between train and test and produce a misleadingly good-looking model.
   This script assigns whole fixtures to train or test, never splits a
   fixture across both.

USAGE:
  python3 build_training_set.py --polls pressure_polls.jsonl --out training_set.jsonl
  python3 build_training_set.py --polls pressure_polls.jsonl --out training_set.jsonl --test-frac 0.2 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


def load_polls(path: str) -> list[dict]:
    polls = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                polls.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return polls


def safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_team_timelines(polls: list[dict]) -> dict[tuple, list[dict]]:
    """Group polls by (fixture_id, team_id), sorted chronologically by ts."""
    timelines: dict[tuple, list[dict]] = defaultdict(list)
    for p in polls:
        key = (p.get("fixture_id"), p.get("team_id"))
        timelines[key].append(p)
    for key in timelines:
        timelines[key].sort(key=lambda p: p.get("ts", 0))
    return timelines


def detect_goal_events(timeline: list[dict]) -> list[tuple[int, float]]:
    """Detect goal events for one team from score increases between polls.

    Returns list of (minute, timestamp) for each detected goal.
    Uses the poll's own `is_home` flag to read the correct side's score.
    """
    events = []
    prev_score = None
    for p in timeline:
        is_home = p.get("is_home")
        score = p.get("score_home") if is_home else p.get("score_away")
        if score is None:
            continue
        if prev_score is not None and score > prev_score:
            events.append((p.get("minute", 0), p.get("ts", 0)))
        prev_score = score
    return events


def find_prior_entry(timeline: list[dict], idx: int, minutes_back: int) -> dict | None:
    """Find the most recent poll at least `minutes_back` real minutes before
    timeline[idx], by wall-clock ts (falls back to minute field if ts missing).
    """
    cur = timeline[idx]
    cur_ts = cur.get("ts", 0)
    cur_min = cur.get("minute", 0)
    target_ts = cur_ts - minutes_back * 60
    target_min = cur_min - minutes_back
    best = None
    for j in range(idx - 1, -1, -1):
        p = timeline[j]
        p_ts = p.get("ts", 0)
        if p_ts and cur_ts:
            if p_ts <= target_ts:
                best = p
                break
        else:
            if p.get("minute", 0) <= target_min:
                best = p
                break
    return best


def build_row(poll: dict, timeline: list[dict], idx: int, goal_events: list[tuple[int, float]]) -> dict:
    minute = poll.get("minute", 0)
    ts = poll.get("ts", 0)
    sot = poll.get("sot", 0) or 0
    total_shots = poll.get("total_shots", 0) or 0
    shots_inside_box = poll.get("shots_inside_box", 0) or 0
    xg = safe_float(poll.get("xg"), 0.0)
    corners = poll.get("corners", 0) or 0
    gps = safe_float(poll.get("gps"), 0.0)
    accel_count = poll.get("accel_count", 0) or 0
    is_home = poll.get("is_home", False)
    score_home = poll.get("score_home", 0) or 0
    score_away = poll.get("score_away", 0) or 0
    team_score = score_home if is_home else score_away
    opp_score = score_away if is_home else score_home
    data_quality = poll.get("data_quality")

    # --- Opponent features (v10.44k+ bot versions only — older poll exports
    # won't have these; they'll come through as 0/None until a fresh /polls
    # pull is taken from a bot running this code) ---
    opp_sot = poll.get("opp_sot", 0) or 0
    opp_total_shots = poll.get("opp_total_shots", 0) or 0
    opp_shots_inside_box = poll.get("opp_shots_inside_box", 0) or 0
    opp_xg = safe_float(poll.get("opp_xg"), 0.0)
    opp_corners = poll.get("opp_corners", 0) or 0
    opp_gps = safe_float(poll.get("opp_gps"), 0.0)

    # --- Labels: does a goal event happen after this poll? ---
    label_15min = 0
    label_full = 0
    minutes_to_next_goal = None
    for g_min, g_ts in goal_events:
        # goal must be strictly after this poll (by minute; ts as tiebreak)
        after = (g_min > minute) or (g_min == minute and g_ts > ts)
        if not after:
            continue
        label_full = 1
        gap = g_min - minute
        if minutes_to_next_goal is None or gap < minutes_to_next_goal:
            minutes_to_next_goal = gap
        if gap <= 15:
            label_15min = 1

    # --- Recency features, computed from this team's own timeline ---
    prior_5 = find_prior_entry(timeline, idx, 5)
    prior_10 = find_prior_entry(timeline, idx, 10)

    def delta(field, prior):
        if prior is None:
            return None
        cur_val = poll.get(field)
        prior_val = prior.get(field)
        if cur_val is None or prior_val is None:
            return None
        return safe_float(cur_val) - safe_float(prior_val)

    sot_delta_5m = delta("sot", prior_5)
    sot_delta_10m = delta("sot", prior_10)
    xg_delta_5m = delta("xg", prior_5)
    shots_delta_5m = delta("total_shots", prior_5)

    ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
    recency_ratio = None
    if prior_10 is not None and sot > 0:
        sot_10m_ago = prior_10.get("sot")
        if sot_10m_ago is not None:
            recency_ratio = (sot - sot_10m_ago) / sot

    # --- last_goal_minute: NOT a real poll field (checked — it isn't written
    # to pressure_polls.jsonl). Derived here from this team's own goal_events
    # timeline instead: most recent goal at or before this poll's minute. ---
    last_goal_minute = None
    for g_min, _g_ts in goal_events:
        if g_min <= minute:
            if last_goal_minute is None or g_min > last_goal_minute:
                last_goal_minute = g_min
    minutes_since_last_goal = (minute - last_goal_minute) if last_goal_minute is not None else None

    return {
        # identifiers (kept for grouping/splitting/audit — drop before training)
        "fixture_id": poll.get("fixture_id"),
        "team_id": poll.get("team_id"),
        "team_name": poll.get("team_name"),
        "league": poll.get("league"),
        "minute": minute,
        "ts": ts,
        # features
        "sot": sot,
        "total_shots": total_shots,
        "shots_inside_box": shots_inside_box,
        "ib_ratio": round(ib_ratio, 3),
        "xg": xg,
        "corners": corners,
        "gps": gps,
        "gps_sot": poll.get("gps_sot", 0),
        "gps_ib": poll.get("gps_ib", 0),
        "gps_sv": poll.get("gps_sv", 0),
        "gps_xg": poll.get("gps_xg", 0),
        "gps_accel": poll.get("gps_accel", 0),
        "accel_count": accel_count,
        "is_home": bool(is_home),
        "score_diff": team_score - opp_score,
        "team_score": team_score,
        "opp_score": opp_score,
        "opp_sot": opp_sot,
        "opp_total_shots": opp_total_shots,
        "opp_shots_inside_box": opp_shots_inside_box,
        "opp_xg": opp_xg,
        "opp_corners": opp_corners,
        "opp_gps": opp_gps,
        "data_quality": data_quality,
        "sot_delta_5m": sot_delta_5m,
        "sot_delta_10m": sot_delta_10m,
        "xg_delta_5m": xg_delta_5m,
        "shots_delta_5m": shots_delta_5m,
        "recency_ratio": round(recency_ratio, 3) if recency_ratio is not None else None,
        "last_goal_minute": last_goal_minute,
        "minutes_since_last_goal": minutes_since_last_goal,
        # labels
        "label_15min": label_15min,
        "label_full": label_full,
        "minutes_to_next_goal": minutes_to_next_goal,
    }


def build_training_set(polls: list[dict], minute_min: int | None = None,
                        minute_max: int | None = None,
                        min_data_quality: float | None = None) -> list[dict]:
    timelines = build_team_timelines(polls)
    rows = []
    for key, timeline in timelines.items():
        goal_events = detect_goal_events(timeline)
        for idx, poll in enumerate(timeline):
            minute = poll.get("minute", 0)
            if minute_min is not None and minute < minute_min:
                continue
            if minute_max is not None and minute > minute_max:
                continue
            dq = poll.get("data_quality")
            if min_data_quality is not None and dq is not None and dq < min_data_quality:
                continue
            rows.append(build_row(poll, timeline, idx, goal_events))
    return rows


def match_level_split(rows: list[dict], test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by fixture_id, never by row — prevents leakage between
    highly-correlated polls from the same match."""
    fixture_ids = sorted(set(r["fixture_id"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(fixture_ids)
    n_test = max(1, int(len(fixture_ids) * test_frac))
    test_ids = set(fixture_ids[:n_test])
    train = [r for r in rows if r["fixture_id"] not in test_ids]
    test = [r for r in rows if r["fixture_id"] in test_ids]
    return train, test


def main():
    ap = argparse.ArgumentParser(description="Build labeled training set from pressure_polls.jsonl")
    ap.add_argument("--polls", required=True, help="Path to pressure_polls.jsonl")
    ap.add_argument("--out", required=True, help="Output path for full labeled JSONL")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Fraction of fixtures held out for test")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the match-level split")
    ap.add_argument("--train-out", default=None, help="Optional separate train JSONL path")
    ap.add_argument("--test-out", default=None, help="Optional separate test JSONL path")
    ap.add_argument("--minute-min", type=int, default=9, help="Drop polls before this minute (bot's own window floor)")
    ap.add_argument("--minute-max", type=int, default=85, help="Drop polls after this minute (bot's own window ceiling)")
    ap.add_argument("--min-data-quality", type=float, default=0.5, help="Drop polls below this data_quality score")
    args = ap.parse_args()

    polls = load_polls(args.polls)
    print(f"Loaded {len(polls)} raw polls")

    rows = build_training_set(polls, minute_min=args.minute_min, minute_max=args.minute_max,
                               min_data_quality=args.min_data_quality)
    print(f"Built {len(rows)} labeled rows (minute {args.minute_min}-{args.minute_max}, "
          f"data_quality>={args.min_data_quality}) across "
          f"{len(set(r['fixture_id'] for r in rows))} fixtures, "
          f"{len(set((r['fixture_id'], r['team_id']) for r in rows))} team-sides")

    pos_15 = sum(r["label_15min"] for r in rows)
    pos_full = sum(r["label_full"] for r in rows)
    print(f"Positive rate: label_15min={pos_15}/{len(rows)} ({100*pos_15/len(rows):.1f}%) | "
          f"label_full={pos_full}/{len(rows)} ({100*pos_full/len(rows):.1f}%)")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote full set to {args.out}")

    if args.train_out and args.test_out:
        train, test = match_level_split(rows, args.test_frac, args.seed)
        print(f"Match-level split: {len(train)} train rows "
              f"({len(set(r['fixture_id'] for r in train))} fixtures) | "
              f"{len(test)} test rows ({len(set(r['fixture_id'] for r in test))} fixtures)")
        with open(args.train_out, "w") as f:
            for r in train:
                f.write(json.dumps(r, default=str) + "\n")
        with open(args.test_out, "w") as f:
            for r in test:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"Wrote {args.train_out} / {args.test_out}")


if __name__ == "__main__":
    main()
