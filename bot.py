import json
import math
import os
import subprocess
import sys
import time
import gzip
import io
import logging
import httpx
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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

# v10.19: Suppress httpx INFO logs — Telegram getUpdates fires every 2min
# during dead hours, flooding logs with useless "HTTP/1.1 200 OK" lines.
logging.getLogger("httpx").setLevel(logging.WARNING)

# v10.45: ML shadow-scoring — loads a trained model (if present) to compute
# a second, independent pressure score alongside GPS on every poll.
# IMPORTANT: this ONLY logs a comparison score. It never gates, blocks, or
# changes which signals get sent — that stays 100% GPS-controlled for now.
# Wrapped so any failure (missing package, missing/corrupt model file) just
# disables ML scoring silently — the bot keeps running exactly as before.
# MUST come after `log` is defined above — an earlier version of this block
# sat before the logging setup and crashed the whole bot on any load failure
# (NameError: name 'log' is not defined), which defeated the entire point
# of wrapping it in try/except. Fixed here.
ML_MODEL = None
ML_FEATURES: list[str] = []
try:
    import lightgbm as _lgb
    import numpy as _np
    _ML_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_predictor_v1.txt")
    _ML_FEATURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_predictor_v1_features.json")
    if os.path.exists(_ML_MODEL_PATH) and os.path.exists(_ML_FEATURES_PATH):
        ML_MODEL = _lgb.Booster(model_file=_ML_MODEL_PATH)
        with open(_ML_FEATURES_PATH) as _f:
            ML_FEATURES = json.load(_f)
        log.info(f"v10.45: ML shadow model loaded ({len(ML_FEATURES)} features) — logging only, not gating signals")
    else:
        log.info("v10.45: ML model files not found — ML shadow scoring disabled, GPS-only (no change in behavior)")
except Exception as _ml_load_err:
    log.warning(f"v10.45: ML shadow model failed to load ({_ml_load_err}) — GPS-only (no change in behavior)")


def calculate_ml_score(feature_values: dict) -> float | None:
    """v10.45: Score one poll with the trained model. Returns 0-100 (same
    scale as GPS) or None if the model isn't loaded / a value is missing.
    Never raises — any error just means no ML score for this poll.
    """
    if ML_MODEL is None or not ML_FEATURES:
        return None
    try:
        row = [[
            float("nan") if feature_values.get(f) is None else float(feature_values.get(f))
            for f in ML_FEATURES
        ]]
        x = _np.array(row)
        proba = ML_MODEL.predict(x)[0]
        return round(float(proba) * 100, 1)
    except Exception:
        return None



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

API_BASE = "https://v3.football.api-sports.io"
TELEGRAM_API = "https://api.telegram.org"

LEAGUE_IDS = {
    39: "Premier League", 140: "La Liga", 78: "Bundesliga", 79: "2. Bundesliga",
    135: "Serie A", 61: "Ligue 1", 2: "Champions League", 3: "Europa League",
    848: "Conference League", 357: "First League (Bulgaria)", 172: "First League (Bulgaria)",
    656: "Super Cup (Bulgaria)",  # v10.81: Levski-CSKA derby Sep 9 was UNTRACKED (league 656)
    94: "Primeira Liga",
    88: "Eredivisie", 203: "Super Lig",
    283: "Liga I (Romania)", 210: "HNL (Croatia)", 345: "Czech First League",
    119: "Danish Superliga", 137: "Veikkausliiga (Finland)", 191: "NB I (Hungary)",
    # v10.44d-fix: League of Ireland (API may return this ID for Irish Prem Div)
    543: "League of Ireland",
}

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "LIVE", "IN_PLAY"}

# Active monitoring window: dynamically computed from daily schedule
# Falls back to 14:00-23:00 if schedule fetch fails
BULGARIA_TZ = ZoneInfo("Europe/Sofia")
ACTIVE_HOUR_START_FALLBACK = 14  # fallback
ACTIVE_HOUR_END_FALLBACK = 23    # fallback

# v10.10: 21'-61' window — data shows 68% WR here, 0-20' is noisy, 62'+ drops to ~35%
MINUTE_MIN = 21
MINUTE_MAX = 61

# v10.15: Pre-window monitoring — start polling from minute 1.
# v10.20: Raised from 12→20 so min_allowed = 21-20 = 1'.
# Fixes chicken-and-egg: bot needs to poll stats to discover high early SOT,
# but the old 9' floor prevented polling until SOT was already known.
# Credit cost is ~0: stats are batched (1 credit per up to 20 fixtures),
# so adding early fixtures to an existing batch costs nothing extra.
# Signal gates (PRE-WINDOW block, EARLY GATE GPS≥75) still prevent noise.
PRE_WINDOW_MINUTES = 20  # discover & poll from 1', signal from 21'

# v10.13.2: Extended window for SOT-accelerating fixtures
# If a fixture shows SOT acceleration (SOT increased last poll) OR
# best_sot >= 2, the window extends:
#   - BEFORE 21': signals fire from minute 9+ (early pressure, v10.19: GPS≥75 required)
#   - AFTER 61': signals fire until minute 85 (late pressure buildup, v10.19: raised from 75)
# Note: polling starts from 1' (PRE_WINDOW_MINUTES=20), but signals
# before 9' are always blocked — 9' is the earliest possible signal minute.
EXTENDED_MIN = 9    # minimum minute for early-pressure signals
EXTENDED_MAX = 85   # maximum minute for late-pressure signals

# v10.34: Freshness thresholds for late-window signals (61'+)
# A team with GPS 90 but no recent activity is stale domination, not danger.
# Evidence: GPS 100 signals went 0/4 (Juventus 45', AC Milan 75', Venezia 81', Göztepe 69')
# while fresh late signals like Hajduk 82' (SOT 5, GPS 82, scored 83') still hit.
FRESHNESS_MINUTE = 61           # minute threshold for freshness requirement
FRESHNESS_HARD_STOP = 80        # v10.40: lowered from 86. 80'+ blocked unless event-confirmed SOT burst
FRESHNESS_SOT_DELTA = 1        # SOT increased in last ~5 game minutes
FRESHNESS_GPS_RISING = 0.5     # GPS rose by 0.5+ from 5 min ago
FRESHNESS_ACCEL_MIN = 1        # at least 1 acceleration indicator
FRESHNESS_IB_FLOOR = 0.60      # 60% IB required for 76'+ (stricter quality gate)
# v10.84: ATTEMPT-BURST + RESPONSE WINDOW + window-delta repair
ATTEMPT_BURST_MIN = 3          # attempts (any shot) in the last 10 game minutes
ATTEMPT_BURST_IB_DELTA = 1     # at least 1 box shot inside the burst (quality)
RESPONSE_WINDOW_MIN = 10       # minutes after conceding that re-arm the watch
RESPONSE_GPS_FLOOR = 40        # reduced GPS floor inside the response window
MINUTE_ARCHIVE_LEN = 18        # 1-per-game-minute snapshots (>= 10m lookback)
# v10.85: SIMPLIFIED SIGNAL — plain-language betting lines (display-only;
# ledger schema and every gate untouched)
# v10.86: EMPIRICAL LAMBDA CALIBRATION — minute-banded deflate of the
# remaining-goals lambdas (prediction-only; fair prices honest, advice
# threshold stricter; 61+ signals stay full alerts per user decision)
# v10.87: GOAL-RACE GUARD — feed-ahead-of-score mute: the pre-send events
# knowledge (valid goals) vs the scoreline the signal was gated on; the
# PSV-Shakhtar 45' class (signal arrived WITH the goal — unbettable)
# v10.89: RED-AWARE LOSING RELAXATION — trailing by <= 1 while the
# opponent is down a NET man passes the losing filter at STANDARD tier
# bars (Slavia 1-0 Lens class), tagged in message + ledger (red_relax)
# v10.88: POST-GOAL HONESTY — stale 5-20m post-goal CRITICALs are BLOCKED
# (the goal's own shot completed their GPS trigger — Man Utd 33'); every
# sent 5-20m post-goal signal says it watches the NEXT goal (Como 28');
# header ordinal becomes (next) once the signaling team has scored
# v10.34: Stricter losing-team threshold for CRITICAL signals
# Data: losing teams 22.2% WR vs winning 64.5%. Desperation shots inflate stats.
LOSING_GPS_MIN = 80            # losing teams need GPS >= 80
LOSING_IB_MIN = 0.65           # losing teams need IB >= 65%
LOSING_SOT_MIN = 4             # losing teams need SOT >= 4

# v10.36: Post-goal cooldown — suppress signals after team scores.
# Stats are inflated by the goal itself (the shot that scored counts as SOT).
# Without cooldown, bot signals "look at this pressure!" after the goal.
POST_GOAL_COOLDOWN = 5        # hard suppress for 5 min after scoring (unless new SOT)
POST_GOAL_RELEVANCE = 20     # 5-20 min: require fresh pressure; 20+: normal logic

# v10.44: Score-state dampener — winning teams generate phantom pressure signals.
# Evidence: AEK up 4-0 SOT=7 GPS=71 (MISS x2), Viking up 3-1 SOT=5 GPS=73 (MISS x3).
# A team winning comfortably takes low-urgency shots that inflate SOT/GPS
# without genuine scoring threat.
SCORE_DIFF_SUPPRESS = 3    # +3 or more: suppress unless fresh acceleration
# SCORE_DIFF_GPS_OVERRIDE removed — GPS magnitude alone doesn't prove fresh intent

# v9.7.1: Night hours — no European tracked leagues play
# Skip ALL schedule rechecks during this window (saves 2 credits per skipped recheck)
NIGHT_HOUR_START = 1   # 01:00 Bulgaria — all European leagues finished
NIGHT_HOUR_END = 10    # 10:00 Bulgaria — earliest possible kickoff (~11:00 Scandinavia)

# Fast SOT polling window: once SOT reaches 2+, we poll faster for a
# limited time.  After the window expires, polling reverts to base rate.
FAST_SOT_WINDOW = 5 * 60  # 300 seconds

# v10.44f: Signal cooldown — after a team signals, require GPS to drop
# below signal threshold for 2+ consecutive polls before re-signaling.
# This prevents re-firing every poll while GPS sustains above threshold.
SIGNAL_COOLDOWN_POLLS = 2   # consecutive polls below threshold to re-qualify
SIGNAL_COOLDOWN_GPS_FLOOR = 55  # GPS_EARLY_WARNING value — below this = pressure broken

# v10.46: Minimum wall-clock gap between signals for the SAME team via the
# pressure-buildup override. Data (Benfica 46'->47', Athletic Club): when the
# stats API lags then catches up, SOT/xG can jump +3/+0.50 in ONE poll — the
# buildup override read that as "genuinely new danger" and re-signaled within
# a minute. Real pressure buildups take 4+ game minutes to add +3 SOT; API
# catch-ups land within 1-2 poll cycles (<120s). 180s blocks the fake jumps
# without delaying genuine escalations. GOAL RESET and COOLDOWN RE-QUALIFY
# paths are exempt — they already represent real state changes / elapsed time.
SIGNAL_MIN_GAP_SECONDS = 180

# Max fixture IDs per batched request (API-Football limit for /fixtures?ids=...)
BATCH_SIZE_LIMIT = 20

# --- State ---
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []
rate_limited_until: float = 0.0

# API-Football reports the daily quota at the subscription level.
# Do NOT assume that multiple API keys multiply the daily allowance.
quota_remaining: int | None = None
quota_limit: int | None = None
minute_remaining: int | None = None
minute_limit: int | None = None
_last_stale_resolve: float = 0.0  # v10.44d-fix: periodic resolution timer

# --- v10.71: FEED-GUARD — silent API feed-death detection ---
# 2026-09-05 incident: /fixtures?live=all returned an EMPTY live set while
# 7 tracked matches were at 18'-78' (all HTTP 200, quota counter frozen at
# 7499 for 60+ min). The bot trusted the feed, dropped every fixture, wiped
# all monitoring state and idled silently for the rest of the evening. The
# API's own status page never recorded the incident (its monitor only pings
# endpoint availability, not data integrity), so the bot must detect this
# class of failure itself.
FEED_GUARD_MIN = 85            # last-seen minute below which a vanished fixture is suspicious
FEED_GUARD_HOLD_MAX = 90 * 60  # hard cap: stop holding a feed-death incident after 90 min
FEED_GUARD_ALARM_EVERY = 15 * 60  # Telegram re-alarm cadence while an incident is active
FEED_GUARD_FROZEN_CALLS = 45   # identical remaining readings -> frozen-quota warning
QUOTA_FROZEN_WARN_EVERY = 12 * 3600  # frozen-quota warning cadence (once per 12h)
# Incident state: active / since / last_alarm / last_cycle / held_fids /
# alarms_sent / gave_up. Reset only when the incident closes.
feed_guard: dict = {
    "active": False, "since": 0.0, "last_alarm": 0.0, "last_cycle": 0.0,
    "held_fids": set(), "alarms_sent": 0, "gave_up": False,
}
# Quota-freeze watchdog state: value = last remaining reading, count = how
# many consecutive calls reported the SAME value. Warning is sent from the
# main loop (needs the httpx client), detection happens in update_quota().
quota_freeze: dict = {"value": None, "count": 0, "last_warn": 0.0}
# v10.71: latest scheduled tracked kickoff today (UTC ts) — gates the EOD
# block so mid-day monitoring gaps do not fire the EOD report/clear.
# None = schedule unknown (permissive, legacy behavior). 0.0 = no matches
# today (gate open — legitimate wrap-up of yesterday's stragglers).
todays_kickoff_latest_ts: float | None = None
# v10.71: EOD anti-hammer — full EOD attempts (resolve+retry+report+backup)
# at most once per 30 min; deferral notes at most once per 10 min.
_last_eod_attempt: float = 0.0
_last_eod_defer_note: float = 0.0

# --- v10.72: SHADOW GATES + STOP PROBE + MIDNIGHT GUARD ---
# (a)+(b): two would-suppress classes identified from the Sep 4-6 outcome
# data. SHADOW mode = log + outcome-tag only, signals go out UNCHANGED.
# The hard flags are the one-line flip: set True after the ~1-week shadow
# sample (~300 combined tagged signals) says the would-blocked set is
# mostly misses. Until then every rule below is observation-only.
#   DAMP   — team winning by 2+ with GPS < 85: 23% full WR vs 31% baseline
#            (the "won 2-0 and keeps shooting" dead zone; +3+ is already
#            hard-suppressed by the v10.44 SCORE DAMPENER)
#   LATE75 — signal at 75'+: only 18% convert, too little time left to score
DAMPENER_HARD_GATE = False        # flip to True to make DAMP a real gate
LATE_HARD_GATE = False            # flip to True to make LATE75 a real gate
DAMPENER_SHADOW_LEAD = 2          # winning by this many goals ...
DAMPENER_SHADOW_GPS_MAX = 85.0    # ... while GPS below this -> DAMP class
LATE_SHADOW_MINUTE = 75           # signal minute at/after this -> LATE75 class

# --- v10.75: BOX-BURST SHADOW (logging-only, never sent) -----------------
# The low-SOT box-volume class from the Sep 2-8 polls backtest (team-sides
# monitored to 80'+, first crossing, one per side, ~3.5 fires/day):
#   SOT 1-2 + ib>=8 at 21-61' plain     64.3% scored later / 35.7% <=15m (n=14)
#   same + ib rising (last 2 polls)     69.2% scored later / 38.5% <=15m (n=13)
#   broad relaxations are NEGATIVE: ib>=6 = 48.4% / 48.1% — do NOT widen.
#   late window kills it: same cell at 62-74' = 41.7%.
# Base-rate comparison: any side at the 45' checkpoint scores later 48% of
# the time — the cell is +16pp with thin n. SHADOW FIRST: every crossing
# records a virtual signal resolved exactly like a real one; NEVER sent,
# never gates anything, zero extra credits (runs on the stats poll the bot
# already fetched). Go-live rule after ~2 weeks (~40 samples): >=60% later
# AND >=40% <=15m -> flip BOXBURST_LIVE (compact alert via its own client);
# otherwise discard the class. Rising is RECORDED as a flag, not required
# (+5pp on n=13 is too thin to gate on; note: the poll recency fields
# ib_5m_ago/ib_10m_ago are NEVER populated in the data, so this evaluator
# keeps its own 2-poll ib history — surge-watch pattern).
BOXBURST_LIVE = False             # True -> shadow record + compact Telegram alert
BOXBURST_SOT_MIN = 1              # cell = SOT between min and max (low-SOT class)
BOXBURST_SOT_MAX = 2
BOXBURST_IB_MIN = 8               # shots inside box threshold
BOXBURST_MIN_MINUTE = 21          # before 21' the shot counts are too small
BOXBURST_MAX_MINUTE = 61          # 62'+ = the late dead zone (41.7%)
BOXBURST_DAILY_CAP = 60           # bookkeeping safety cap (~3.5/day expected)
# ---- v10.76: GOAL-BURST (banked goals) — the totals path -------------------
# The Sep 9 post-mortem (Lille 2-3 Betis: 5 goals by 53', ZERO signals):
# Betis' 3 SOT *were* their 3 goals — GOAL-SHOT NET strips them -> effective
# pressure 0 — and neither GPS hit 70 (52/65). The system only certifies
# sustained NON-goal pressure, so goal-burst games are structurally invisible.
# The Sep 2-8 backtest (63,489 polls, 124 fixtures) adds: 69% of FIRST goals
# land while the scorer's SOT is still <=2 (43% on their very 1st SOT) — the
# SOT>=3 net is late by design. This class reacts to the goals themselves
# (banked goals = totals evidence), ONE virtual match-level record per
# (fixture, class) at the first poll inside the window:
#   G1  first goal by 25'  -> game reaches 3+ 62.5%  (n=32)  [react O2.5]
#   G2  two goals by 40'   -> next goal 84.1%, <=15m 47.7%  (n=44)  [O2.5]
#        GPS-blind subset 78.8%, no-signal-before 87.5% — the exact blind spot
#   G3  three goals by 55' -> next goal 75.7%, <=15m 32.4%  (n=37)  [O3.5]
# G2 already clears the box-burst go-live bar (>=60% later AND >=40% <=15m),
# so its compact alert ships LIVE; G3/G1 stay shadow behind their own flags
# until their files say otherwise. Zero gate changes, zero extra credits.
GOALBURST_LIVE = True             # G2 alert ON (84% / 48% clears the go-live bar)
GOALBURST_LIVE_G3 = False         # flip only at >=75% later AND >=40% <=15m
GOALBURST_LIVE_G1 = False         # flip only if reach-3+ >= 62% holds at n>=60
GOALBURST_G1_MAX_MINUTE = 25
GOALBURST_G2_MAX_MINUTE = 40
GOALBURST_G3_MAX_MINUTE = 55
GOALBURST_DAILY_CAP = 60          # bookkeeping safety cap (~17 crossings/day)
_GB_META: dict[str, tuple] = {    # class -> (trigger, bet_target, bet_desc, hist)
    "G1": ("1 goal by 25'", 3, "react O2.5 after early goal 1", "62% reach 3+ (n=32)"),
    "G2": ("2 goals by 40'", 3, "O2.5 after 2 banked goals", "84% next goal, 48% <=15m (n=44)"),
    "G3": ("3 goals by 55'", 4, "O3.5 after 3 banked goals", "76% next goal, 32% <=15m (n=37)"),
}
# v10.78: empirical continuation rate per class (the Sep 2-8 backtest
# cells) — powers the break-even line in the live goal-burst alert:
# break-even = 1 / rate (G2: 1/0.841 = 1.19).
_GB_CONT_RATE: dict[str, float] = {"G1": 0.625, "G2": 0.841, "G3": 0.757}
_shadow_tags: dict[str, int] = {}  # session-scoped would-suppress counters (heartbeat)
# (d): STOP-mode renewal probe — api_get() raises BEFORE any call once
# quota_remaining <= 0, which made STOP a dead-end: the bot could never see
# the 00:00 UTC / 03:00 Sofia renewal and looped 30-min sleeps forever
# (Sep 6 audit). One direct probe call per STOP wake re-reads the live
# header so exhaustion self-heals.
STOP_PROBE_EVERY = 30 * 60        # probe cadence while in STOP
stop_probe: dict = {"last": 0.0}  # last probe wall-time (0.0 = not yet armed)
# (f): midnight live-game guard — never abandon a live monitored match at
# the active-window rollover (Sep 6 23:59->00:00 Sofia: GIL Vicente live at
# 71' with a FastWin countdown 41s from firing; the bot slept 17h through
# the final 19 minutes and a frozen "FastWin 0s" zombie line stayed in
# every heartbeat until afternoon).
MIDNIGHT_GUARD_MAX = 2 * 60 * 60       # cap: keep watching at most 2h past window end
MIDNIGHT_GUARD_POLL_FRESH = 15 * 60    # guard only fixtures polled this recently
# v10.74: BOOT-PATH midnight guard — a restart at ~00:00 local previously
# entered dead-hours sleep with pending outcomes sitting on LIVE fixtures
# (Sep 8 00:01 boot: last 22:30-kickoff game ~90' live, pending unresolved
# until the morning command wake). The resolver's live-fixture knowledge
# ("pendings on LIVE fixtures") now feeds the same hold logic. Freshness:
# the periodic resolver re-verifies every 10m while pendings exist, so a
# 25m window survives one missed pass and releases on the next.
PENDING_GUARD_FRESH = 25 * 60
MIDNIGHT_GUARD_NOTE_EVERY = 5 * 60     # holding-note log cadence
midnight_guard: dict = {"since": 0.0, "last_note": 0.0, "gave_up": False}
# FastWin zombie grace: expired countdown entries whose fixture is no
# longer being polled are purged from the heartbeat after this long.
FASTWIN_ZOMBIE_GRACE = 15 * 60

# --- v9.5: Round-robin API key management ---
# Each key has health state: rate-limited until timestamp, or auth-failed.
key_health: list[dict] = []
rr_index: int = 0  # round-robin counter

# --- v9.5.4: Per-team signal limit tracking ---
# Key: (fixture_id, team_id) -> {"count": N, "goals_at_last_signal": G, "sot_at_last_signal": S, "xg_at_last_signal": X, "last_signal_time": T}
# 1st signal: always sent (SOT >= 3 increased -> CRITICAL, or GPS-based EARLY WARNING)
# 2nd signal: no explicit SOT-jump gate; classify_signal dedup handles it (+1 SOT suffices)
# 3rd+ signal: SOT-jump gate from LAST SIGNAL (not last poll):
#   - +1 SOT in <3 min with GPS >= 60 (fast-jump discount: floor = 75-15)
#   - +2 SOT with any GPS (passes jump requirement regardless)
#   - +1 SOT slower than 3 min requires GPS >= 75
#   - blocked if team scored since last signal
# v9.7: sot_at_last_signal fixes the bug where jump was measured from last poll's SOT
# v10.19.3: fast-jump GPS discount for rapid +1 SOT increments
signaled_teams: dict[tuple[int, int], dict] = {}
# v10.56: GOAL-SOT LEDGER — "the goal shot never triggers" (user spec
# 2026-09-03, fourth round: apply the v10.55 surge-watch goal semantics to
# the MAIN signal pipeline, specifically the repeat-signal logic sig_num >= 2).
# A goal IS a shot on target: it lands in the SOT counter (stats lag 1-3 min
# behind the score). Left unadjusted, that +1 reads as "fresh pressure" /
# "SOT jump" / "buildup" and re-signals about a goal that ALREADY happened —
# zero advance-warning value by definition.
#   _pending_goal_sot (fid,tid) -> goal shots presumed not yet in the SOT counter
#     (registered at goal detection, consumed as SOT rises)
#   _goal_sot_landed (fid,tid)  -> cumulative goal shots consumed into the counter
#   genuine SOT jump since last signal = (sot - sot_at_last_signal)
#                                     - (landed_now - landed_at_last_signal)
_pending_goal_sot: dict[tuple[int, int], int] = {}
_goal_sot_landed: dict[tuple[int, int], int] = {}
# v10.44f: Per-team signal cooldown tracking.
# After a team signals, don't re-signal unless GPS dropped below threshold
# for 2+ consecutive polls (pressure died and rebuilt) OR a goal reset the state.
# Key: (fid, tid) -> count of consecutive polls with GPS < signal threshold
# since last signal. When count >= 2, the team is "re-qualified" for a new signal.
team_cooldown_polls: dict[tuple[int, int], int] = {}
# Keep fixture-level set for backward compat in logs/cleanup
signaled_fixtures: set[int] = set()

# --- v10.28: Signal outcome tracking (backtesting) ---
# Records every signal sent with enriched recency data for empirical analysis.
#   PRIMARY KPI:   outcome_full (did they score at any point after signal before FT?)
#   SECONDARY KPI: outcome_5min/10min/15min (how quickly did they score?)
# v10.28: Added recency fields: SOT/shots/xG/IB/corners at previous poll,
#   5min ago, 10min ago, deltas, GPS change, seconds since last signal, recency_ratio.
# Persisted to JSONL file on persistent volume so data survives restarts.
signal_outcomes: list[dict] = []
OUTCOME_WINDOW_MINUTES = 15  # game minutes for "imminent" window
# Railway volume mount point — survives redeployments
_VOLUME_DIR = os.environ.get("VOLUME_DIR", "/data")
os.makedirs(_VOLUME_DIR, exist_ok=True)
OUTCOMES_FILE = os.path.join(_VOLUME_DIR, "signal_outcomes.jsonl")
EOD_SENT_FILE = os.path.join(_VOLUME_DIR, "eod_sent.txt")
ML_BACKUP_SENT_FILE = os.path.join(_VOLUME_DIR, "ml_backup_sent.txt")

# --- v10.49: False-negative tracking + Poisson calibration (LOGGING ONLY) ---
# blocked_outcomes: moments where a QUALIFYING signal (tier assigned) was
# suppressed by a policy gate. Resolved like signal_outcomes so the EOD
# report can show which gates cost us wins. Never affects signal logic.
BLOCKED_FILE = os.path.join(_VOLUME_DIR, "blocked_outcomes.jsonl")
CALIBRATION_FILE = os.path.join(_VOLUME_DIR, "poisson_calibration.json")
blocked_outcomes: list[dict] = []        # v10.49: suppressed-signal candidates
_blocked_dedupe: dict[tuple, int] = {}   # v10.49: (fid, tid, reason) -> minute bucket
_blocked_count_today: int = 0           # v10.49: daily volume cap counter
_blocked_count_date: str | None = None  # v10.49: date of the counter
_poisson_calibration: dict[str, dict] = {}  # v10.49: league|source -> accumulator

# --- v10.60: FIELD-AVAILABILITY CENSUS (LOGGING ONLY) ---
# Empirical ground truth on which statistics fields the API actually
# delivers per league — learned from live responses, never assumed from
# docs. Motivated by the big_chances post-mortem: shipped as available in
# v10.44d, actually 0 across ALL 81 signals + 3965 polls (never delivered
# on this plan). The census tells us which KPIs are REAL training material
# for brain v2 — 'no' fields are recorded as null, never fake zeros.
FIELD_CENSUS_FILE = os.path.join(_VOLUME_DIR, "field_census.json")
_field_census: dict[int, dict[str, bool]] = {}   # league_id -> {field: arrived}

# --- v10.50: FAST-LANE SHADOW MODE (Phase 1 — LOGGING ONLY, never sent) ---
# Virtual signals detected from the events feed (2-3 min ahead of stats).
# NOTHING here sends Telegram messages or changes signal decisions. Each
# virtual signal is recorded + resolved like a real one so the EOD report
# can measure: WR vs the stats path, median speed gain, pure-speed wins vs
# duplicates. Promotion to live firing (Phase 2) is a SEPARATE future change
# gated on the criteria in FASTLANE_PROPOSAL.md.
FASTLANE_SHADOW_FILE = os.path.join(_VOLUME_DIR, "fastlane_shadow.jsonl")
GOAL_FLASH_FILE = os.path.join(_VOLUME_DIR, "goal_flash.jsonl")  # v10.53: goal-flash alert log
SURGE_WATCH_FILE = os.path.join(_VOLUME_DIR, "surge_watch.jsonl")  # v10.54: pre-goal surge alert log
fastlane_shadow: list[dict] = []        # v10.50: virtual-signal records
_fl_shadow_dedupe: dict[tuple, float] = {}  # v10.50: (fid, tid) -> last shadow wall-time
_fl_shadow_count_today: int = 0         # v10.50: daily shadow cap counter (400)
_fl_shadow_count_date: str | None = None
BOXBURST_SHADOW_FILE = os.path.join(_VOLUME_DIR, "boxburst_shadow.jsonl")  # v10.75
boxburst_shadow: list[dict] = []        # v10.75: box-burst virtual-signal records
_boxburst_fired: dict[tuple, float] = {}   # v10.75: (fid, tid) -> wall-time (dedupe)
_boxburst_ib_hist: dict[tuple, list] = {}  # v10.75: (fid, tid) -> [(minute, ib)] short history
_boxburst_count_today: int = 0            # v10.75: daily cap counter
_boxburst_count_date: str | None = None
GOALBURST_SHADOW_FILE = os.path.join(_VOLUME_DIR, "goalburst_shadow.jsonl")  # v10.76
goalburst_shadow: list[dict] = []        # v10.76: goal-burst virtual-signal records
_goalburst_fired: dict[tuple, float] = {}   # v10.76: (fid, class) -> wall-time (dedupe)
_goalburst_count_today: int = 0            # v10.76: daily cap counter
_goalburst_count_date: str | None = None

# --- v10: Bot Version (module-level so all functions can access it) ---
# v10.79 — SCORER STAMPING: the FT resolver now keeps the goal-scorer names
# already present in the /fixtures/events payloads it fetches anyway
# (zero extra credits). Signal records get post_signal_scorer / _minute /
# _scorers / _scorer_is_named (grades the Top-SOT 'scores next' hint
# against ground truth); goal-burst records get post_crossing_scorer.
# Display is unchanged — names live in the ledger only, never Telegram
# (the never-show-a-scorer rule is FINAL). This is the labeled dataset the
# next-scorer model needs: you cannot get better at predicting the scorer
# until you record who actually scored.
# v10.80 — CARDS & CORNERS MARKET BLOCK: every Telegram signal gains a
# prediction block for the total-cards and total-corners markets (live
# counts from the same batch statistics, book O/U line + prices from the
# SAME odds fetch — zero extra credits, PRE ref labeling like v10.78; a
# transparent v1 heuristic projection -> P(over) -> fair price ->
# OVER/UNDER lean). The block's counts auto-update via editMessageText
# as cards/corners land. FT labels: card events (who/when, zero credits
# — same events call) + FT corner totals (one /fixtures/statistics call
# per fixture with signals, quota-guarded) + referee name (free).
# mkt_*_ft_result rows grade every lean once resolved — the calibration
# dataset. Parser hardening: the goals-O/U branch is now guarded to the
# GOALS market only (Cards/Corners/1st-half O/U names could contaminate
# the goals line).
# v10.81 — BULGARIAN SUPER CUP (656) TRACKED: the Sep 9 Levski vs CSKA
# derby was the Super Cup (league 656), not the First League (172/357) —
# discovery listed it as UNTRACKED, so 4-5 SOT and a goal produced ZERO
# polls and zero signals. One line fixes the coverage hole.
# v10.82 — CARDS & CORNERS BLOCK COMPACT (user request, Sep 9: "too
# complex"): 8 lines -> one line per market (count vs line + lean + the
# break-even of the LEANED side; UNDER now prints 1/(1-p) instead of the
# v10.80 fair-O). Ledger mkt_* fields, record schema and FT grading
# unchanged; same single render path for send + live edits.
# v10.83 — BOOT VERSION BANNER: the bot never printed BOT_VERSION at
# startup (banners were hardcoded per-feature strings), so a fresh
# deploy could not be verified from logs until the first signal carried
# its [v10.xx] tag. One f-string line fixes it for every future bump.
BOT_VERSION = "v10.89"

# --- v10: Goal Pressure Score (GPS) ---
# Composite 0-100 score calculated on EVERY stats poll.
# Uses all available API stats (zero extra cost — data already in response).
# v10.26 Components (max ~111, capped at 100):
#   Domestic (xG available): SOT 0-28 | IB 0-22 | SV 0-14 | xG 0-18 | Corners 0-4 | Accel 0-22
#   European (no xG):       SOT 0-35 | IB 0-27 | SV 0-14 | xG 0     | Corners 0-4 | Accel 0-28
#   REMOVED: possession (always 0 in API), dangerous_attacks (not in API-Football v3)
# The acceleration component is the key differentiator from v9.9.
GPS_EARLY_WARNING = 55   # 55-74: EARLY WARNING signal
GPS_CRITICAL = 75         # 75+: CRITICAL signal
GPS_BUILDING = 40        # 40-54: logged as BUILDING (no Telegram)

# v10.61: BC-missing redistribution. big_chances is NEVER delivered on this
# API plan (0 across all 81 signals + 3,965 polls, every league), so the 6-pt
# BC component is always absent — and when xG IS present there is no
# compensation path at all (corner_cap stays 2). gps_restored redistributes
# that weight proportionally (same intent as the xG-missing cap raises) and
# is logged into every poll/signal/blocked record. The live gates stay on the
# historical scale: every threshold (EW floor 60, CRITICAL 75, cooldown floor
# 55) was tuned on it with real outcome data, and all 74 CRITICALs ever
# recorded came from the SOT>=3 safety net. See CHANGES_v10.61.md.
GPS_BC_MAX = 6.0
GPS_BC_REDISTRIBUTE = False  # shadow by default; flip ONLY with a threshold re-tune

# v10.44i: League-tier GPS adjustment for minor leagues.
# Lower-quality leagues produce fewer shots/lower IB ratios.
# Instead of changing GPS formula (frozen until 100+ signals),
# we lower the SIGNAL THRESHOLD for classification.
# Tier 2 = -10 GPS floor (Bulgaria, Romania, Croatia, Finland, Hungary, Ireland)
LEAGUE_TIER2_IDS = {357, 283, 210, 137, 191, 543}
LEAGUE_GPS_FLOOR_ADJUSTMENT = -10  # Tier 2 leagues: GPS 50+ can trigger EW (vs 60+)

# --- v10: Pressure acceleration tracking ---
# Fixtures where GPS is rising AND multiple stat deltas are positive.
# These get 60s polling to catch the SOT transition in real time,
# even BEFORE SOT reaches 2. This is the key v10 improvement.
pressure_accelerating: set[int] = set()  # fixture IDs

# --- v10.17: SOT burst tracking ---
# Fixtures where best SOT jumped by 2+ in a single poll interval.
# Gets highest priority (97) polling and GPS bonus.
sot_burst_fixtures: set[int] = set()
genuine_burst_fixtures: set[int] = set()   # v10.56: 2+ GENUINE (non-goal) SOT jump in one poll

# --- v10: Poll-level data collection for backtesting ---
# Records ALL indicators from every stats poll (not just signals).
# This builds the dataset to empirically validate which combinations
# predict goals within 5/10/15 minutes.
# v10.19.3: Moved to /data volume so poll data survives redeployments
POLL_DATA_FILE = os.path.join(_VOLUME_DIR, "pressure_polls.jsonl")

# --- v10.11: Daily summary ---
daily_summary_date: str = ""
todays_tracked_fixtures: list[dict] = []  # populated during schedule fetch

# --- v10.12: End-of-day Telegram summary ---
eod_summary_sent_date: str = ""  # track if today's EOD summary was sent via Telegram
eod_report_sent_date: str = ""  # v10.32: track EOD report subprocess (once per day)

# v10.35: Separate persistence file for subprocess EOD (vs old Telegram EOD summary)
EOD_REPORT_SENT_FILE = os.path.join(_VOLUME_DIR, "eod_report_sent.txt")


def _load_eod_report_sent_date() -> str:
    """v10.35: Persist EOD report subprocess sent date to disk."""
    try:
        if os.path.exists(EOD_REPORT_SENT_FILE):
            with open(EOD_REPORT_SENT_FILE, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


def _save_eod_report_sent_date(date_str: str) -> None:
    """v10.35: Persist EOD report subprocess sent date to disk."""
    try:
        with open(EOD_REPORT_SENT_FILE, "w") as f:
            f.write(date_str)
    except Exception:
        pass


def _load_eod_sent_date() -> str:
    """v10.16: Persist EOD sent date to disk so restarts don't re-send."""
    try:
        if os.path.exists(EOD_SENT_FILE):
            with open(EOD_SENT_FILE, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


def _save_eod_sent_date(date_str: str) -> None:
    """v10.16: Persist EOD sent date to disk."""
    try:
        with open(EOD_SENT_FILE, "w") as f:
            f.write(date_str)
    except Exception:
        pass


last_discovery_time: float = 0.0
last_stats_check: dict[int, float] = {}   # fixture_id -> timestamp of last stats fetch
fast_monitored: set[int] = set()         # fixture IDs currently monitored
fast_priority: dict[int, int] = {}       # fixture_id -> rank score (discovery-time)
cached_fixtures: list[dict] = []        # last discovery result (reused for filtering only)

# --- v9.5.8: Daily match count for adaptive mode ---
total_matches_today: int = 0
FIRST_SIGNAL_ONLY_THRESHOLD = 20  # >20 games: first signal only, move on
FULL_TRACKING_THRESHOLD = 15   # <15 games: full tracking (multiple signals)

# --- Fast SOT window state ---
fast_sot_until: dict[int, float] = {}
# v10.21: Track SOT level at last fast-window activation.
# Prevents infinite re-activation when SOT is stuck (e.g. SOT=1 for 10+ min).
fast_sot_activated_at_sot: dict[int, int] = {}

# --- v9.7: SOT acceleration tracking ---
# Fixtures where the best SOT increased in the last poll.
# These get priority polling (60-90s) to catch the next SOT increase.
accelerating_fixtures: set[int] = set()

# --- v10.1: Per-team GPS history (last 5 polls for 5-min windowed rates) ---
# Key: (fixture_id, team_id) -> list of dicts with GPS + raw stats + timestamp per poll
# v10.1: Expanded from 3 to 5 to enable 5-game-minute window calculations.
team_gps_history: dict[tuple[int, int], list[dict]] = {}
GPS_HISTORY_MAX = 5  # v10.1: keep last 5 polls (covers ~5 min at 60s polling)
# v10.84: minute-resolution archive — the 5-entry live history spans only
# ~75s at the 15s cadence, so it can NEVER reach 5/10 game minutes back and
# the 5m/10m window deltas were structurally null (99.8% of Sep-9 polls).
# This archive keeps the LATEST poll per game minute and repairs them.
_minute_history: dict[tuple[int, int], list[dict]] = {}
_goal_shot_minutes: dict[tuple[int, int], list[int]] = {}
_fixture_goal_log: dict[int, list[tuple[int, int]]] = {}
GPS_WINDOW_MINUTES = 5  # v10.1: window for rate-of-change calculations

# --- v9.7: Dead fixture tracking (zero-pressure matches) ---
# Fixtures where BOTH teams had SOT=0 on last stats check.
# These are removed from monitoring to save credits.
# They get REVIVED if discovery detects a score change (momentum shift),
# or every 5min if either team has SOT>=3 or big acceleration (v10.43).
# Key: fixture_id -> (home_goals, away_goals, death_timestamp).
dead_fixtures: dict[int, tuple[int, int, float]] = {}

# --- v10.44b: Data-dead fixture tracking (API has no stats for this match) ---
# Fixtures where BOTH teams return shots=0 AND SOT=0 at minute >= 10.
# This means the API data provider doesn't cover this match (not corruption
# — the data simply doesn't exist). Removed from monitoring to save credits.
# Revived every 5min if stats appear in discovery-cached data.
# Key: fixture_id -> death_timestamp.
DATA_DEAD_MINUTE = 10  # minimum minute before declaring data-dead
data_dead_fixtures: dict[int, float] = {}

# --- Irish team name detection (API-Football ID collision: league 357 = both Bulgaria & Ireland) ---
IRISH_TEAM_NAMES = {
    "Bohemians", "Bohemian FC", "Shelbourne", "St Patrick's Athletic",
    "Waterford", "Waterford FC", "Drogheda United", "Dundalk",
    "Galway United", "Derry City", "Sligo Rovers", "Cork City",
    "Athlone Town", "Longford Town", "Finn Harps", "Treaty United",
    "Cobh Ramblers", "Wexford", "Bray Wanderers",
}


def _fix_league_name(league: str, home_name: str, away_name: str) -> str:
    """Override league name for known API-Football ID collisions.
    
    API-Football returns league ID 357 for both Bulgarian First League and
    Irish Premier Division. The API name is 'First League' for both.
    Detect Irish teams by name and return correct league name.
    """
    if home_name in IRISH_TEAM_NAMES or away_name in IRISH_TEAM_NAMES:
        return "League of Ireland"
    return league


# --- v9.7: /fixtures?ids= statistics detection ---
# Lazily tested on first stats call. If True, one /fixtures?ids= call
# returns EVERYTHING (score, minute, SOT, xG, red cards) — no individual
# /fixtures/statistics calls needed. Massive credit saving.
_ids_endpoint_has_stats: bool | None = None  # None=untested, True/False=cached

# --- v9.6.1: Scheduled 20' entry times (UTC timestamps) ---
# Populated from the known schedule at startup.
# The bot wakes up ~60s before each entry instead of polling blindly.
scheduled_window_entries: list[float] = []  # sorted UTC timestamps when kickoffs+20min occur

# --- Dynamic active hours (v9.5.4) ---
# Computed from /fixtures?date=today, re-checked every 3h for late additions
dynamic_active_start: int = ACTIVE_HOUR_START_FALLBACK
dynamic_active_end: int = ACTIVE_HOUR_END_FALLBACK
schedule_date: str = ""  # YYYY-MM-DD of cached schedule
schedule_no_matches: bool = False
schedule_last_fetch: float = 0.0  # timestamp of last schedule fetch
SCHEDULE_RECHECK_INTERVAL = 8 * 3600  # v9.7: 6h→8h (saves ~1 credit/day, late additions rare after 8h)

# --- v9.7.2: Tomorrow's kickoff cache ---
# When we fetch today+tomorrow schedule, we also cache tomorrow's kickoff hours.
# At midnight, the new "today" matches yesterday's "tomorrow" — so we set the
# active window from cache (0 credits) instead of fetching again.
# A real API fetch still happens ~30min before window start to get fixture IDs.
cached_tomorrow_date: str = ""  # YYYY-MM-DD of cached tomorrow data
cached_tomorrow_kickoffs: list[float] = []  # kickoff hours (local) for that date
schedule_fixture_ids_loaded: bool = False  # True after real fetch got fixture IDs

# --- Friendly matches (v9.6.0) ---
# Track international friendlies where >=1 team plays in our 20 tracked leagues
# (via pre-seeded + auto-filled team ID cache).
# Skipped on busy days (>= FRIENDLY_BUSY_THRESHOLD main-league matches)
# to avoid spreading credits too thin.
FRIENDLY_BUSY_THRESHOLD = 10  # skip friendlies when 10+ league matches
active_friendly_fixtures: set[int] = set()  # fixture IDs of approved international friendlies
FRIENDLY_NAME_KEYWORDS = ("friendly", "friendlies")

# --- v9.5.6: Pre-seeded major club team IDs ---
# Solves the pre-season chicken-and-egg problem: during summer, big clubs
# only play friendlies, so their IDs never enter the cache from league fixtures.
# These IDs are stable in API-Football (api-sports.io) across seasons.
# The cache still auto-fills from actual league matches once seasons start.
PRESEEDED_TEAM_IDS = {
    # --- Premier League (39) ---
    42, 50, 40, 33, 49, 47, 34, 66, 51, 48, 65, 52, 36, 55,
    # Arsenal, Man City, Liverpool, Man Utd, Chelsea, Tottenham,
    # Newcastle, Aston Villa, Brighton, West Ham, Nott'm Forest, Crystal Palace, Fulham, Brentford

    # --- La Liga (140) ---
    541, 529, 530, 536, 548, 533, 531,
    # Real Madrid, Barcelona, Atletico, Sevilla, Real Sociedad, Villarreal, Real Betis

    # --- Bundesliga (78) ---
    157, 165, 168, 173, 161, 169, 172, 163,
    # Bayern, Dortmund, Leverkusen, Leipzig, Wolfsburg, Frankfurt, Stuttgart, M'gladbach

    # --- 2. Bundesliga (79) ---
    174, 192, 190, 189,
    # Hamburg, Schalke 04, Hertha, Hannover 96

    # --- Serie A (135) ---
    505, 489, 496, 492, 497, 487, 499, 502,
    # Inter, AC Milan, Juventus, Napoli, Roma, Lazio, Atalanta, Fiorentina

    # --- Ligue 1 (61) ---
    85, 81, 80, 91, 79, 84,
    # PSG, Marseille, Lyon, Monaco, Lille, Nice

    # --- Primeira Liga (94) ---
    211, 212, 228, 213,
    # Benfica, Porto, Sporting CP, Braga

    # --- Eredivisie (88) ---
    194, 197, 215,
    # Ajax, PSV, Feyenoord

    # --- Super Lig (203) ---
    263, 264, 262, 261,
    # Galatasaray, Fenerbahce, Besiktas, Trabzonspor

    # --- Austrian Bundesliga (169) ---
    556, 559, 558, 560,
    # Salzburg, Rapid Wien, Austria Wien, Sturm Graz

    # --- SuperLiga Serbia (283) ---
    2634, 2635,
    # Red Star, Partizan

    # --- HNL Croatia (210) ---
    1992, 1994,
    # Dinamo Zagreb, Hajduk Split

    # --- Czech First League (345) ---
    1021, 1023, 620,
    # Sparta Prague, Slavia Prague, Viktoria Plzen

    # --- Danish Superliga (119) ---
    281, 283, 282, 300,
    # Copenhagen, Brondby, Midtjylland, AGF

    # --- Veikkausliiga Finland (137) ---
    677, 679,
    # HJK Helsinki, KuPS

    # --- NB I Hungary (191) ---
    688, 689, 821,
    # Ferencvaros, Paks, Puskas Akademia

    # --- First League Bulgaria (357) ---
    # (populated from actual league fixtures — always in season during tracked months)
}
known_league_team_ids: set[int] = set(PRESEEDED_TEAM_IDS)


# ============================================================
# API KEY ROUND-ROBIN (v9.5)
# ============================================================

def init_key_health():
    global key_health
    key_health = [
        {"rate_limited_until": 0.0, "auth_failed": False}
        for _ in API_KEYS
    ]


def pick_key() -> str | None:
    """Round-robin through healthy keys. Returns None if all unhealthy."""
    global rr_index
    if not key_health:
        init_key_health()

    tried = 0
    while tried < len(API_KEYS):
        idx = rr_index % len(API_KEYS)
        health = key_health[idx]
        rr_index += 1

        if health["auth_failed"]:
            tried += 1
            continue
        if time.time() < health["rate_limited_until"]:
            tried += 1
            continue

        return API_KEYS[idx]

    return None


def mark_key_auth_failed(key: str):
    global key_health
    if key in API_KEYS:
        idx = API_KEYS.index(key)
        if idx < len(key_health):
            key_health[idx]["auth_failed"] = True
            log.warning(f"  Key {key[:8]}... marked as AUTH FAILED")


def mark_key_rate_limited(key: str, duration: float = 120.0):
    global key_health
    if key in API_KEYS:
        idx = API_KEYS.index(key)
        if idx < len(key_health):
            key_health[idx]["rate_limited_until"] = time.time() + duration
            log.warning(f"  Key {key[:8]}... rate-limited for {int(duration)}s")


def healthy_key_count() -> int:
    if not key_health:
        return len(API_KEYS)
    now = time.time()
    count = 0
    for h in key_health:
        if not h["auth_failed"] and now >= h["rate_limited_until"]:
            count += 1
    return count


# ============================================================
# QUOTA TRACKING
# ============================================================

def update_quota(resp: httpx.Response):
    global quota_remaining, quota_limit
    global minute_remaining, minute_limit

    try:
        value = resp.headers.get("x-ratelimit-requests-remaining")
        if value:
            new_remaining = int(value)
            # v10.71: frozen-quota watchdog — count consecutive calls that
            # report the IDENTICAL daily-remaining value. Healthy metering
            # decrements every call (verified: 1501->1495 across ~6 calls in
            # 50s on Sep 5 17:16 UTC); during the Sep 5 feed death the value
            # stayed 7499 across 60+ calls in 60 min. Advisory only — the
            # warning itself is sent from the main loop.
            if quota_freeze["value"] == new_remaining:
                quota_freeze["count"] += 1
            else:
                quota_freeze["value"] = new_remaining
                quota_freeze["count"] = 1
            quota_remaining = new_remaining

        value = resp.headers.get("x-ratelimit-requests-limit")
        if value:
            quota_limit = int(value)

        value = resp.headers.get("x-ratelimit-remaining")
        if value:
            minute_remaining = int(value)

        value = resp.headers.get("x-ratelimit-limit")
        if value:
            minute_limit = int(value)

    except (ValueError, TypeError):
        pass


# ============================================================
# API HELPERS
# ============================================================

def api_get(client: httpx.Client, endpoint: str, params: dict = None) -> dict:
    global request_count, rate_limited_until

    if quota_remaining is not None and quota_remaining <= 0:
        raise Exception("Daily API quota exhausted")

    # Round-robin through healthy keys
    attempts = 0
    max_attempts = len(API_KEYS) * 2

    while attempts < max_attempts:
        key = pick_key()
        if key is None:
            raise Exception("All API keys are unhealthy (auth failed or rate limited)")

        request_count += 1

        log.info(
            f"  API GET {endpoint} "
            f"(key={key[:8]}..., req #{request_count}, "
            f"remaining={quota_remaining})"
        )

        resp = client.get(
            f"{API_BASE}{endpoint}",
            params=params,
            headers={"x-apisports-key": key},
        )

        update_quota(resp)

        if resp.status_code == 429:
            # Rate limit: mark this key and try another immediately
            mark_key_rate_limited(key, 120.0)
            attempts += 1
            # If NO healthy keys remain, set global backoff
            if healthy_key_count() == 0:
                rate_limited_until = time.time() + 120
                raise Exception("Rate limited (429) on all keys, backing off 120s")
            continue

        if resp.status_code in (401, 403):
            mark_key_auth_failed(key)
            attempts += 1
            if healthy_key_count() == 0:
                raise Exception(
                    f"All {len(API_KEYS)} key(s) failed with auth errors"
                )
            continue

        resp.raise_for_status()
        return resp.json()

    raise Exception("All API keys failed after max attempts")


def _v10_72_stop_renewal_probe(client: httpx.Client) -> bool:
    """v10.72: STOP-mode renewal probe — ONE direct request to re-read the
    daily quota header.

    api_get() refuses every call once quota_remaining <= 0 (its pre-flight
    raise), so in STOP mode the bot is blind to the 00:00 UTC / 03:00 Sofia
    renewal: it loops "sleeping 30 min" forever and only a manual restart
    re-reads reality (documented dead-end, Sep 6 audit). This probe
    intentionally BYPASSES that raise: one GET /status with a healthy key,
    response fed to update_quota() so the live header re-populates
    quota_remaining. Returns True when the quota has renewed (> 0).

    Cost: at most 1 credit per probe (~2/hour while stopped) — the /status
    endpoint is the API's own quota/account check and is not billed against
    the daily allowance; even if a plan bills it, one call per 30 min is
    negligible against a fresh 7,500.
    """
    global request_count
    key = pick_key()
    if key is None:
        log.warning("v10.72: STOP probe skipped — no healthy API key")
        return False
    try:
        request_count += 1
        log.info(
            f"  v10.72 STOP PROBE: GET /status (key={key[:8]}..., req #{request_count})"
        )
        resp = client.get(f"{API_BASE}/status", headers={"x-apisports-key": key})
        update_quota(resp)  # header is authoritative even on error statuses
        if resp.status_code == 429:
            mark_key_rate_limited(key, 120.0)
            log.warning("v10.72: STOP probe rate-limited (429) — retry next wake")
            return False
        if resp.status_code in (401, 403):
            mark_key_auth_failed(key)
            log.warning(f"v10.72: STOP probe auth failed ({resp.status_code}) — retry next wake")
            return False
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"v10.72: STOP renewal probe failed: {e}")
        return False
    if quota_remaining is not None and quota_remaining > 0:
        log.info(
            f"v10.72: STOP probe — quota RENEWED ({quota_remaining}/{quota_limit}), "
            f"resuming normal operation"
        )
        return True
    log.info(
        f"v10.72: STOP probe — still exhausted ({quota_remaining}/{quota_limit})"
    )
    return False


def send_telegram(client: httpx.Client, text: str) -> bool:
    """Send message to Telegram, auto-splitting at 4000 chars (Telegram limit: 4096).

    v10.38: Signal messages with losing_tag + stale_tag + recency info
    can exceed 4096 chars, causing silent truncation. Now splits at 4000
    char boundaries (96 char safety margin for any Telegram overhead).
    Splits on newline boundaries when possible to avoid mid-word breaks.
    """
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        try:
            resp = client.post(
                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            )
            resp.raise_for_status()
            # v10.80: return the message_id (int, truthy) so the market
            # block can register for editMessageText live updates. None on
            # failure keeps the old falsy contract for every caller.
            try:
                return int(resp.json().get("result", {}).get("message_id"))
            except Exception:
                return 1  # sent OK, id unreadable — edits will no-op safely
        except Exception as e:
            log.error(f"Telegram send failed: {e}")
            return None

    # Split long messages into chunks at newline boundaries
    chunks = []
    remaining = text
    while len(remaining) > MAX_LEN:
        # Find last newline within the limit
        split_at = remaining.rfind("\n", 0, MAX_LEN)
        if split_at <= 0:
            split_at = MAX_LEN  # fallback: hard split
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    chunks.append(remaining)

    # v10.80: return the LAST chunk's message_id (the market block always
    # sits at the very end, so it lives in the last chunk). None if any
    # chunk failed (preserves the old all_ok semantics).
    any_fail = False
    last_id = None
    for i, chunk in enumerate(chunks):
        try:
            resp = client.post(
                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
            )
            resp.raise_for_status()
            try:
                last_id = int(resp.json().get("result", {}).get("message_id"))
            except Exception:
                last_id = 1
        except Exception as e:
            log.error(f"Telegram send failed (chunk {i+1}/{len(chunks)}): {e}")
            any_fail = True
    return None if any_fail else last_id


# ============================================================
# CANDIDATE RANKING (v9.5.1 — pressure-first priority)
# ============================================================
#
# 1. SOT / attacking pressure     (dominant factor)
# 2. xG                          (important secondary)
# 3. Game minute                 (prefer 45-70')
# 4. Scoreline / game state      (small 0-0 bonus, no penalty for goals)
# 5. Diversification             (modest unsignaled bonus)
#
# Key principle: rank by actual pressure, not scoreline.
# A 1-0 match with SOT=4 should outrank a 0-0 match with SOT=0.


def rank_candidate(fixture: dict, team_id: int) -> int:
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    fid = fixture["fixture"]["id"]

    # --- 1. SOT / attacking pressure (DOMINANT) ---
    prev = team_state.get((fid, team_id))
    prev_sot = prev.get("last_sot", 0) if prev else 0

    # Scaled heavily so it always dominates other factors.
    # SOT=0 unknown=10, SOT=1=100, SOT=2=300, SOT=3=600,
    # SOT=4=900, SOT=5=1200, SOT=6+=1500
    if prev_sot >= 6:
        score += 1500
    elif prev_sot >= 5:
        score += 1200
    elif prev_sot >= 4:
        score += 900
    elif prev_sot >= 3:
        score += 600
    elif prev_sot >= 2:
        score += 300
    elif prev_sot >= 1:
        score += 100
    else:
        score += 10  # unknown — worth a first check

    # --- 2. xG (important secondary — up to ~150 points) ---
    # v9.5.8: Boosted range for better xG differentiation
    if prev and prev.get("last_xg"):
        try:
            xg = float(prev["last_xg"])
            score += int(xg * 100)
            # Bonus: high xG without SOT=3 yet means pressure is building
            if prev_sot < 3 and xg >= 0.8:
                score += 80  # likely about to hit SOT=3
            elif prev_sot < 3 and xg >= 0.5:
                score += 40
        except (ValueError, TypeError):
            pass

    # --- 3. Game minute (prefer 45-70') ---
    if 45 <= minute <= 70:
        score += 30
    elif 30 <= minute <= 80:
        score += 15
    elif 20 <= minute <= 29:
        score += 5

    # --- 4. Scoreline / game state (small bonus only) ---
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if tg == 0 and og == 0:
        score += 40
    elif abs(tg - og) <= 1:
        score += 20
    # No penalty for any scoreline — all scorelines eligible

    # --- 5. Diversification (modest unsignaled bonus) ---
    if fid not in signaled_fixtures:
        score += 50

    return score


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    if quota_remaining is None:
        return "UNKNOWN"
    if quota_remaining <= 0:
        return "STOP"
    if quota_remaining <= 10:
        return "EMERGENCY"
    if quota_remaining <= 25:
        return "STRICT"
    if quota_remaining <= 50:
        return "CAREFUL"
    return "NORMAL"


def get_max_fast_monitored() -> int:
    """v9.6.2: Dynamic max monitored fixtures (batch-aware).

    With batched API requests (up to 20 fixtures per request),
    monitoring more fixtures costs the SAME as monitoring few —
    1 credit per batch, regardless of how many fixtures are in it.
    The cost driver is number of batch polls, not fixture count.

    Reserve only 3 credits (1 discovery + 2 safety). The old reserve
    of 10 was from pre-batching era and caused Max=0 when quota≤10,
    meaning the bot found candidates but refused to monitor any.
    """
    if quota_remaining is None:
        return BATCH_SIZE_LIMIT

    reserve = 3  # 1 discovery + 2 safety
    available = quota_remaining - reserve
    if available <= 0:
        return 0

    # With batching, cost is per-batch not per-fixture.
    # 1 credit = stats for ALL monitored fixtures (up to 20).
    # In EMERGENCY (600s interval), a 60-min window needs ~6 batch checks.
    # In NORMAL (240s interval), ~15 batch checks.
    # We need at least 2 batch checks to be useful (catch existing SOT≥3).
    if available < 2:
        return 0

    # Always return full batch limit — more fixtures monitored
    # doesn't cost more credits with batching.
    return BATCH_SIZE_LIMIT


def get_schedule_based_discovery_interval(budget_mode: str) -> int | None:
    """v9.6.1: Use the KNOWN schedule to predict when the next wave of
    matches enters the pre-window, and wake up just in time.

    v10.15: buffer=-60 means we wake 60s AFTER the pre-window mark,
    accounting for API-Football's ~60s minute reporting delay. Entries
    are NOT consumed on gap<=0 — instead they age out after 180s.
    This retries every 30s if the API hasn't caught up yet.

    Returns None if no future window entries exist (all waves processed).
    """
    global scheduled_window_entries
    now = time.time()
    buffer = -60  # v10.15: wake 60s AFTER the pre-window mark (accounts for API minute lag)

    # v10.15: Prune entries >3 min old (gives API time to catch up)
    scheduled_window_entries = [
        t for t in scheduled_window_entries if t > now - 180
    ]

    if not scheduled_window_entries:
        return None

    # Find the next future entry (within 3 min window)
    future_entries = [t for t in scheduled_window_entries if t > now - buffer]
    if not future_entries:
        # v10.19: All entries are within the buffer window or past it.
        # Return short interval — either a wave is imminent or entries are stale.
        return 30

    next_entry = min(future_entries)
    gap = next_entry - now - buffer

    if gap <= 0:
        # v10.15: DON'T consume — the API minute might not have caught up yet.
        # Retry in 30s. Entry ages out naturally after 180s (3 retries max).
        # This replaces the old v9.6.2 consume-on-sight behavior which could
        # miss matches when API minute lag exceeded the 60s buffer.
        return 30

    # Cap at the mode's max interval (don't sleep longer than 30 min)
    max_interval = {
        "NORMAL": 1800,
        "CAREFUL": 2400,
        "STRICT": 3000,
        "EMERGENCY": 3600,
        "UNKNOWN": 1800,
    }.get(budget_mode, 1800)

    return min(max(60, int(gap)), max_interval)


def get_discovery_interval(
    budget_mode: str,
    has_tracked_live: bool,
    has_candidates: bool,
) -> int:
    """Seconds between live-fixture discovery calls.

    v9.6.2: Uses the pre-computed schedule to wake up right when
    kickoff waves enter the 20' window, instead of blind polling.
    Falls back to live-fixture-minute-based estimation if schedule is empty.

    v9.6.2: If we can't monitor anything (Max=0), don't rapid-fire
    discovery — it just burns credits finding candidates we can't check.
    """
    # v9.6.2 guard: if we can't monitor anything, back off massively.
    # No point discovering candidates if we can't check their stats.
    if get_max_fast_monitored() == 0:
        return 3600  # check once per hour, just in case quota resets

    # Priority 1: Use the schedule (known kickoff times)
    if not has_candidates:
        sched_interval = get_schedule_based_discovery_interval(budget_mode)
        if sched_interval is not None:
            return sched_interval

    # Priority 2: When candidates exist, still check schedule for new waves
    # but also allow a longer interval since stats are already running
    # v10.44s: If last discovery found untracked live fixtures, retry sooner
    # for the first few cycles. Catches league ID changes (e.g. 357→172)
    # or newly added leagues. After 5 retries, give up (league IDs won't
    # change mid-session). Cost: ~5 extra credits.
    if _last_untracked_live_count > 0 and _untracked_retry_count <= 5:
        return 120  # check every 2 min for first 5 retries

    if has_candidates:
        sched_interval = get_schedule_based_discovery_interval(budget_mode)
        if sched_interval is not None and sched_interval < 300:
            # A new wave is about to hit 20' — discover soon
            return sched_interval
        # Otherwise use the standard interval
        return {
            "NORMAL": 1800,
            "CAREFUL": 2400,
            "STRICT": 3000,
            "EMERGENCY": 3600,
            "UNKNOWN": 1800,
        }.get(budget_mode, 1800)

    # Fallback: no schedule data, no tracked live
    return 1800


def get_stats_interval(budget_mode: str) -> int:
    """Base interval for statistics batched requests.

    v10.13: Reduced from 240→90s NORMAL, 300→150s CAREFUL, etc.
    With batched API calls (1 credit for up to 20 fixtures), the cost
    of polling 3x more often is minimal (~+2 credits per cycle with
    4 fixtures) but coverage improves dramatically — fixtures now
    get 3-4 polls in the 40-min window instead of 1.
    """
    # v10.39: NORMAL 90s -> 45s (7500 credits, <4% usage at 45s)
    # Budget protection modes UNCHANGED.
    if budget_mode == "NORMAL":
        return 45
    if budget_mode == "CAREFUL":
        return 150
    if budget_mode == "STRICT":
        return 210
    if budget_mode == "EMERGENCY":
        return 300
    return 300


# ============================================================
# FAST SOT WINDOW
# ============================================================

def get_fixture_best_sot(fid: int) -> int:
    best = 0
    for (f, t), state in team_state.items():
        if f == fid:
            sot = state.get("last_sot", 0)
            if sot > best:
                best = sot
    return best


def activate_fast_sot(fid: int, current_best_sot: int = 0):
    fast_sot_until[fid] = time.time() + FAST_SOT_WINDOW
    fast_sot_activated_at_sot[fid] = current_best_sot
    log.info(
        f"  Fast SOT window ACTIVATED for fixture {fid} "
        f"(expires in {FAST_SOT_WINDOW}s, SOT={current_best_sot})"
    )


def get_fixture_max_gps(fid: int) -> float:
    """v10.22: Get the highest GPS of any team in this fixture."""
    best = 0.0
    for (f, _), history in team_gps_history.items():
        if f == fid and history:
            best = max(best, history[-1].get("gps", 0))
    return best


def is_in_edge_zone(fid: int) -> bool:
    """v10.22: Edge zone = before 21' or after 60'.

    In these zones, signals require GPS>=75 (EARLY/LATE GATE),
    so fast polling is only worthwhile for genuinely high pressure.
    Low-pressure games here burn credits with near-zero signal chance.
    """
    for (f, _), state in team_state.items():
        if f == fid:
            minute = state.get("last_minute", 0)
            if minute > 0 and (minute < MINUTE_MIN or minute > MINUTE_MAX):
                return True
    return False


def is_fast_sot_active(fid: int) -> bool:
    until = fast_sot_until.get(fid)
    if until is None:
        return False
    if time.time() >= until:
        del fast_sot_until[fid]
        return False
    return True


def expire_fast_sot(fid: int):
    fast_sot_until.pop(fid, None)
    fast_sot_activated_at_sot.pop(fid, None)


# ============================================================
# SIGNAL CLASSIFICATION — v10: GPS-based 3-stage system
# ============================================================
# v10 stages:
#   GPS < 40:            no signal
#   GPS 40-54:           BUILDING (log only, no Telegram)
#   GPS 55-74 + SOT>=1: EARLY WARNING (Telegram)
#   GPS 75+ OR SOT>=3:   CRITICAL (Telegram)
#
# SAFETY: SOT>=3 ALWAYS triggers CRITICAL regardless of GPS.
# This ensures the proven v9.x trigger still works while we
# collect data to validate the GPS thresholds.

TIER_ORDER = ["EARLY WARNING", "CRITICAL"]


def classify_signal(
    sot: int, state: dict | None, current_minute: int,
    gps: float = 0.0, accel_count: int = 0,
    inside_box_ratio: float = 0.0, sustained_count: int = 0,
    league_id: int = 0,
) -> tuple[str | None, str, float]:
    """v10.1: Classify signal based on GPS + SOT hybrid system.

    The GPS allows earlier detection (SOT=1 with high acceleration),
    while SOT>=3 remains a guaranteed CRITICAL trigger (safety net).

    v10.1 quality gates:
    - EARLY WARNING requires inside_box_ratio >= 30% (shot quality)
    - EARLY WARNING requires sustained_count >= 1 OR gps >= CRITICAL
      (single-poll spikes don't trigger — need sustained acceleration)

    v10.24 quality gates:
    - IB ratio < 50% blocks ALL signals (shots not dangerous)
    - GPS < 60 blocks EARLY WARNING (SOT>=3 CRITICAL exempt)
    - LOSING teams: EW blocked, CRITICAL tagged with warning

    v10.44i: league_id adjusts GPS floor for minor leagues (-10).
    """
    trend = ""
    sot_rate = 0.0
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    # v10.29: SOT≥3 SAFETY NET — must fire BEFORE IB gate.
    # Bug: v10.24 placed IB<50% before this check, so a team with 8 SOT
    # but IB=42.9% (Nordsjaelland, 77') was blocked. SOT≥3 is the
    # gold standard — if 3+ shots are on target, pressure is real
    # regardless of where other (off-target) shots came from.
    # v10.43: Removed sot > last_sot dedup. It blocked signals when SOT was already
    # high at first poll (e.g. SOT=5 at 45') and never increased. Downstream
    # signaled_teams + stale suppression already prevents duplicate signals.
    if sot >= 3:
        return "CRITICAL", trend, sot_rate

    # v10.24: IB RATIO HARD FLOOR — <50% means most shots are outside the box.
    # Data: 0% short-term HIT (0/5), only 1/5 full HIT was 92' consolation while losing 1-3.
    # Teams chipping long-range shots generate high shot volume but no danger.
    # Applies ONLY to GPS-based signals (SOT < 3). SOT≥3 bypasses above.
    if inside_box_ratio < 0.50:
        return None, "", 0.0

    # v10.24: GPS FLOOR for EARLY WARNING — GPS<60 means composite pressure
    # is not meaningfully elevated. Data: 0/3 HIT across all windows.
    # SOT>=3 CRITICAL signals are exempt (SOT itself is strong evidence).
    # v10.44i: Tier 2 leagues get -10 adjustment (GPS 50+ can trigger).
    _gps_floor = 60 + (LEAGUE_GPS_FLOOR_ADJUSTMENT if league_id in LEAGUE_TIER2_IDS else 0)
    if gps < _gps_floor:
        return None, "", 0.0

    # v10.1 GPS-based triggers for SOT < 3:
    # QUALITY GATE: shot quality matters.
    # 3 harmless long-range shots should NOT equal 3 dangerous inside-box shots.
    # Require at least 30% of shots inside the box for EARLY WARNING.
    if sot >= 1 and gps >= GPS_EARLY_WARNING:
        if inside_box_ratio < 0.30 and gps < GPS_CRITICAL:
            # Low shot quality + not at critical GPS — skip
            return None, "", 0.0

        # SUSTAINED GATE: single-poll acceleration spikes need confirmation.
        # Either: sustained over 2+ polls, OR GPS already at CRITICAL level.
        if sustained_count < 1 and gps < GPS_CRITICAL:
            return None, "", 0.0

        # v10.48: ACCELERATION GATE for EARLY WARNING — EW tier fired at
        # only 29% full WR (2/7). A static profile (SOT/shots/xG/off-target
        # rates all flat this poll, accel_count=0) means cumulative drift,
        # not rising danger — that is exactly the junk EW profile.
        # Require at least ONE accelerating indicator unless GPS is already
        # at CRITICAL level (strong enough on its own). Note: a SOT rise of
        # 1+ within ~6 minutes yields accel_count >= 1 (rate >= 0.15/min),
        # so genuinely rising teams pass; flat-cumulative teams are blocked.
        if accel_count < 1 and gps < GPS_CRITICAL:
            return None, "", 0.0
        last_sot = state["last_sot"] if state else 0
        # For SOT=1: allow if GPS is high (acceleration-driven detection)
        # For SOT=2: allow if GPS is high (replaces v9.9 pressure_building gate)
        if sot > last_sot or gps >= GPS_CRITICAL:
            return "EARLY WARNING", trend, sot_rate

    return None, "", 0.0


def detect_pressure_building(tstats: dict, state: dict | None, minute: int) -> tuple[bool, str]:
    """v10.10: Detect if a team is building dangerous pressure despite low SOT.

    v10.10: Replaced DA with shots_off_target (API-Football v3 has no DA).
    Checks: shots off target rate, total shot volume.
    Returns (is_building, description_string).
    """
    if not state or not state.get("last_minute", 0) > 0:
        return False, ""
    sotr = safe_int(get_stat(tstats, "shots_off_target"))
    ts = safe_int(get_stat(tstats, "total_shots"))

    prev_sotr = state.get("last_shots_off_target", 0)
    prev_min = state["last_minute"]
    mins_passed = max(minute - prev_min, 1)
    sotr_delta = sotr - prev_sotr
    sotr_rate = sotr_delta / mins_passed

    # Pressure building conditions:
    #   1. Off-target shots increasing (at least 1 every 2 minutes)
    #   2. Lots of total shots
    #   3. Minimum absolute off-target to filter early-game noise
    is_building = (
        sotr_rate >= 0.2
        and ts >= 5
        and sotr >= 3
    )

    desc = (f"SOffT:{sotr}(+{sotr_delta}), TS:{ts}, SOffT-rate:{sotr_rate:.1f}/min")
    return is_building, desc


def tier_emoji(tier: str) -> str:
    if tier == "CRITICAL": return "\U0001f534"
    if tier == "EARLY WARNING": return "\U0001f525"
    if tier == "STRONG": return "\U0001f7e0"
    if tier == "PRESSURE": return "\U0001f7e1"
    return "\u26aa"  # BUILDING (white circle)


# ============================================================
# v10.44g: GOAL PREDICTION (Poisson-based over/under)
# ============================================================
# Projects total match goals from current xG/SOT/minute using
# Poisson distribution. No extra API calls. Predictions saved in
# outcomes JSONL for future ML model training.

# Historical calibration: average goals per SOT across top-5 leagues.
# Source: fbref.com 2024-25 season aggregates.
GOALS_PER_SOT = 0.31  # ~31% of SOT become goals

# v10.69: PROJECTION SANITY — remaining-share projection + late-game blend.
# The old math multiplied the observed per-minute rate by a 90/minute
# time_factor and used the FULL-MATCH projection as the REMAINING-goals
# lambda: the last 20' were priced as if they were 90' at the observed
# rate (Elversberg 70' GPS-100 frenzy -> "9.0 goals", actual 7). Now the
# observed rate is projected over the REMAINING minutes only, blended
# toward the league average after 70', lifted by a GPS hotness factor for
# the signaling team, and capped.
PROJECTION_AVG_TOTAL = 2.7        # league-average full-match goals (both teams)
PROJECTION_LATE_MINUTE = 70       # blend the observed rate toward the average after this minute
PROJECTION_LATE_W = 0.55          # weight of the team's own observed rate in the late blend
# v10.74: GPS-hotness lift shrunk 0.8 -> 0.2. The Sep 1-7 backtest (n=170
# undecided O2.5) showed the lift was pure bias: model avg 74.5% vs 54.1%
# landed, and a naive no-pressure Poisson scored a BETTER Brier (0.207 vs
# 0.301). High GPS does not raise the remaining-goals rate by +80%; the
# signal's value is the team-goal timing, not extra total goals.
PROJECTION_HOTGPS_W = 0.2         # signaling-team hotness add: (gps-50)/50 * W (GPS 100 -> +20%)
PROJECTION_FUTURE_CAP = 2.0       # per-team cap on the remaining-goals lambda
PROJECTION_OPP_RATE_FLOOR = 0.25  # opponent rate floor as a share of average (0-SOT teams still score)

# v10.86: EMPIRICAL CALIBRATION DEFLATE — minute-banded multipliers applied
# to the FINAL remaining-goals lambdas (after every other adjustment), so
# every over/BTTS/FT probability, the expected total, and the ledger's proj
# fields measure the shipped math. Measured on the settled ledger Sep 2-9
# (186 lambda-graded signals): opponent lambda ~1.5x hot in BOTH gate
# regimes (factor 0.645 Sep 7-9 / 0.665 Sep 2-9 — the stable bias);
# signal-team lambda hot at the peak-pressure selection (0.86 actual/proj
# on the current-gate regime, 0.34 all-time); the 61'+ band collapses in
# every window (0.1-0.2x). Factors are the regime-weighted compromise,
# tilted conservative: an over-deflated fair price costs one skipped bet,
# an under-deflated one costs money. PREDICTION ONLY — never a gate; the
# 61'+ signals still fire as full alerts (user decision stands).
PROJECTION_CAL_DEFLATE_SIG = ((45, 0.85), (60, 0.80), (999, 0.45))  # signal team
PROJECTION_CAL_DEFLATE_OPP = ((45, 0.70), (60, 0.60), (999, 0.30))  # opponent

# v10.74: RED-CARD LAMBDA — 10v11 shifts scoring rates more than any other
# in-play state. A short-handed team's remaining lambda is scaled DOWN, the
# full-strength opponent's UP. Values from the live-football literature
# (a red card is worth roughly 0.3-0.45 goals net over a full match);
# applied per NET red-card difference, only when events-based counts exist
# (honest missingness: no events coverage -> no adjustment, never fake 0).
# Display + prediction input ONLY — never a signal gate, never GPS.
REDCARD_LAMBDA_DOWN = 0.72
REDCARD_LAMBDA_UP = 1.08

# v10.74: EMPIRICAL GAME-STATE TABLE — P(>=k MORE goals after signal) by
# (minute band, current total), measured from the 229 resolved signals of
# Sep 1-7. This is what the market calls game-state selection: a 1-1 at 56'
# is a cagy game by selection and lands ~1 goal more far less often than
# the league average says. Cells with n>=10 only; thinner cells fall back
# to the model value (honest, no fake precision). Bands: 0=21-45', 1=46-55',
# 2=56-90'. Used ONLY to calibrate the displayed over lines (50/50 blend
# with the model), never as a gate.
_EMPIRICAL_MORE_GOALS = {  # (band, current_total) -> (P>=1, P>=2, P>=3)
    (0, 0): (0.80, 0.48, 0.40),
    (0, 1): (0.771, 0.514, 0.314),
    (0, 2): (0.80, 0.511, 0.311),
    (0, 3): (1.00, 0.643, 0.286),
    (1, 1): (0.643, 0.50, 0.214),
    (1, 2): (0.75, 0.417, 0.167),
    (2, 1): (0.20, 0.20, 0.10),
    (2, 2): (0.40, 0.10, 0.10),
    (2, 3): (0.50, 0.214, 0.0),
    (2, 4): (0.696, 0.565, 0.348),
}


def _empirical_band(minute: int) -> int:
    """v10.74: game-state band for the empirical table (0=21-45',
    1=46-55', 2=56-90')."""
    if minute <= 45:
        return 0
    if minute <= 55:
        return 1
    return 2


def _empirical_p_more(current_total: int, minute: int, goals_needed: int) -> float | None:
    """v10.74: Empirical P(>=goals_needed more goals) at this game state,
    or None when the cell is too thin (caller falls back to the model)."""
    band = _empirical_band(minute)
    key = (band, min(current_total, 4))
    cell = _EMPIRICAL_MORE_GOALS.get(key)
    if cell is None:
        return None
    k = max(1, min(3, goals_needed))
    return cell[k - 1]

# xG projection cap: don't extrapolate beyond 90' even if minute < 20.
# Below 20', cap at 4.5x (20' is 4.5x of 90-minute game).
MINUTE_FLOOR = 20


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(X=k) = (lam^k * e^-lam) / k!"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _poisson_over(prob_a: float, prob_b: float, threshold: float) -> float:
    """P(total goals > threshold) for two independent Poisson variables.
    
    Uses the complement: P(over X.5) = 1 - P(A + B <= X).
    Sums the joint PMF over all (i,j) where i+j <= floor(X.5).
    """
    max_goals = int(threshold) + 3  # enough range for accuracy
    p_under = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j <= int(threshold):
                p_under += _poisson_pmf(i, prob_a) * _poisson_pmf(j, prob_b)
    return max(0.0, min(1.0, 1.0 - p_under))


def _poisson_over_with_scoreline(prob_a: float, prob_b: float, threshold: float, current_goals: int) -> float:
    """P(final total > threshold) accounting for goals already scored.
    
    The Poisson lambdas model FUTURE goals only.
    So P(final > X.5) = P(current + future > X.5) = P(future > X.5 - current).
    If current_goals >= threshold + 1, we're already over → return 1.0.
    """
    remaining_needed = threshold - current_goals
    if remaining_needed < 0:
        return 1.0  # already over
    return _poisson_over(prob_a, prob_b, remaining_needed)


def compute_goal_predictions(
    minute: int,
    signal_team_xg: float | None,
    signal_team_sot: int,
    opponent_xg: float | None,
    opponent_sot: int,
    score_home: int,
    score_away: int,
    is_home_signal: bool,
    gps: float | None = None,
    team_reds: int | None = None,
    opp_reds: int | None = None,
) -> dict:
    """Compute over/under goal probabilities for the full match.
    
    Uses a Poisson model based on projected xG.
    - If xG available: project to 90' from current minute
    - If xG N/A (European comps): estimate xG from SOT * GOALS_PER_SOT
    
    v10.69: REMAINING-SHARE projection. The lambdas model FUTURE goals:
    the observed per-minute rate is projected over the REMAINING minutes
    (not a 90/minute full-match projection misused as the future lambda).
    After PROJECTION_LATE_MINUTE the rate is blended toward the league
    average (frenzies cool), the signaling team gets a GPS hotness lift,
    and each team's lambda is capped at PROJECTION_FUTURE_CAP.
    v10.86: the final lambdas are then deflated by minute-banded empirical
    factors (PROJECTION_CAL_DEFLATE_SIG/OPP) measured on the settled ledger
    — the raw chain over-projects remaining goals, especially late.
    Returns dict with projected xG, over/under probs, and input features
    for future ML training.
    """
    # --- Estimate xG for each team ---
    if signal_team_xg is not None and signal_team_xg > 0:
        est_xg_signal = signal_team_xg
    else:
        est_xg_signal = signal_team_sot * GOALS_PER_SOT
    
    if opponent_xg is not None and opponent_xg > 0:
        est_xg_opponent = opponent_xg
    else:
        est_xg_opponent = opponent_sot * GOALS_PER_SOT
    
    # --- v10.69: project the observed rate over the REMAINING minutes ---
    # Use effective minute (floor at MINUTE_FLOOR to avoid crazy extrapolation)
    eff_minute = max(minute, MINUTE_FLOOR)
    remaining = max(90.0 - eff_minute, 5.0)
    
    rate_sig = est_xg_signal / eff_minute
    rate_opp = est_xg_opponent / eff_minute
    # v10.69: opponent floor — a 0-SOT opponent is not a 0% scorer forever
    _avg_rate = (PROJECTION_AVG_TOTAL / 2.0) / 90.0
    rate_opp = max(rate_opp, PROJECTION_OPP_RATE_FLOOR * _avg_rate)
    # v10.69: late-game blend toward the league average (frenzies cool)
    if minute >= PROJECTION_LATE_MINUTE:
        rate_sig = PROJECTION_LATE_W * rate_sig + (1.0 - PROJECTION_LATE_W) * _avg_rate
        rate_opp = PROJECTION_LATE_W * rate_opp + (1.0 - PROJECTION_LATE_W) * _avg_rate
    
    proj_xg_signal = rate_sig * remaining
    proj_xg_opponent = rate_opp * remaining
    
    # v10.69: GPS hotness lift for the SIGNALING team (live pressure lifts
    # its own remaining rate; capped, never applied to the opponent)
    if gps is not None and gps > 50:
        _hot = min((gps - 50.0) / 50.0 * PROJECTION_HOTGPS_W, PROJECTION_HOTGPS_W)
        proj_xg_signal *= (1.0 + _hot)
    
    # v10.69: sanity caps (the old math could emit 4+ future goals)
    proj_xg_signal = min(proj_xg_signal, PROJECTION_FUTURE_CAP)
    proj_xg_opponent = min(proj_xg_opponent, PROJECTION_FUTURE_CAP)
    
    # Scoreline adjustment: teams that are trailing tend to push harder,
    # teams leading tend to sit back. Small adjustments based on evidence.
    signal_goals = score_home if is_home_signal else score_away
    opp_goals = score_away if is_home_signal else score_home
    goal_diff = signal_goals - opp_goals
    
    if goal_diff >= 2:
        # Leading comfortably: slight suppression on further scoring
        proj_xg_signal *= 0.90
        proj_xg_opponent *= 1.05
    elif goal_diff == 1:
        # Leading by 1: mild effect
        proj_xg_signal *= 0.95
        proj_xg_opponent *= 1.03
    elif goal_diff == -1:
        # Trailing by 1: push harder
        proj_xg_signal *= 1.05
        proj_xg_opponent *= 0.95
    elif goal_diff <= -2:
        # Trailing heavily: may push hard OR give up
        proj_xg_signal *= 1.03
        proj_xg_opponent *= 0.92
    
    # v10.74: RED-CARD LAMBDA — scale each team's remaining-goals rate by
    # the NET man-power difference (per net red card). Only when
    # events-based counts exist (None = no events coverage -> no
    # adjustment, never fake zeros). Prediction input ONLY, never a gate.
    _rc_applied = False
    if team_reds is not None and opp_reds is not None and team_reds != opp_reds:
        _net = team_reds - opp_reds
        if _net > 0:
            proj_xg_signal *= REDCARD_LAMBDA_DOWN ** _net
            proj_xg_opponent *= REDCARD_LAMBDA_UP ** _net
        else:
            proj_xg_opponent *= REDCARD_LAMBDA_DOWN ** (-_net)
            proj_xg_signal *= REDCARD_LAMBDA_UP ** (-_net)
        _rc_applied = True
    
    # v10.86: EMPIRICAL CALIBRATION DEFLATE (see PROJECTION_CAL_DEFLATE_*).
    # Prediction ONLY — never a gate, never GPS input. Applied last so the
    # whole chain (rate -> remaining -> hotness -> scoreline -> reds ->
    # caps) is corrected exactly as shipped. Undecided lines only in
    # spirit: decided lines return 1.0 regardless of the multiplier.
    for _bound, _fac in PROJECTION_CAL_DEFLATE_SIG:
        if minute <= _bound:
            proj_xg_signal *= _fac
            break
    for _bound, _fac in PROJECTION_CAL_DEFLATE_OPP:
        if minute <= _bound:
            proj_xg_opponent *= _fac
            break
    
    # --- Compute over/under probabilities ---
    # v10.44h: Account for goals already scored!
    # Poisson lambdas model FUTURE goals only, so shift threshold by current scoreline.
    current_total_goals = score_home + score_away
    p_over_25 = _poisson_over_with_scoreline(proj_xg_signal, proj_xg_opponent, 2.5, current_total_goals)
    p_over_35 = _poisson_over_with_scoreline(proj_xg_signal, proj_xg_opponent, 3.5, current_total_goals)
    p_over_45 = _poisson_over_with_scoreline(proj_xg_signal, proj_xg_opponent, 4.5, current_total_goals)
    # v10.69: ADAPTIVE LINES — the next three UNDECIDED over lines.
    # A 2-2 game shows O4.5/O5.5/O6.5, not the already-decided O2.5/O3.5.
    # The FIRST entry is always the next-goal line (current total + 0.5) —
    # the exact market the bot's odds capture prices.
    _base_line = float(int(current_total_goals)) + 0.5
    over_lines = [
        (_base_line, _poisson_over_with_scoreline(proj_xg_signal, proj_xg_opponent, _base_line, current_total_goals)),
        (_base_line + 1.0, _poisson_over_with_scoreline(proj_xg_signal, proj_xg_opponent, _base_line + 1.0, current_total_goals)),
        (_base_line + 2.0, _poisson_over_with_scoreline(proj_xg_signal, proj_xg_opponent, _base_line + 2.0, current_total_goals)),
    ]
    
    # v10.74: CALIBRATED over probabilities — 50/50 blend of the model
    # with the empirical game-state table P(>=k more goals | total x
    # minute band). The Sep 1-7 backtest showed the raw model is ~20pp
    # overconfident; the empirical table is model-independent (pure
    # outcome rates) and carries the game-state selection the Poisson
    # blend misses. Decided lines (already over) skip the blend.
    def _cal(p_model: float, line: float, cur_total: int) -> tuple[float, bool]:
        if p_model >= 0.999:
            return p_model, False
        needed = int(math.ceil(line - cur_total))
        p_emp = _empirical_p_more(cur_total, minute, needed)
        if p_emp is None:
            return p_model, False
        return 0.5 * p_model + 0.5 * p_emp, True

    _cal_lines = []
    _any_blend = False
    for _l, _p in over_lines:
        _cp, _bl = _cal(_p, _l, current_total_goals)
        _cal_lines.append((_l, round(_cp, 2)))
        _any_blend = _any_blend or _bl
    _p25c, _b25 = _cal(p_over_25, 2.5, current_total_goals)
    _p35c, _b35 = _cal(p_over_35, 3.5, current_total_goals)
    _p45c, _b45 = _cal(p_over_45, 4.5, current_total_goals)
    _any_blend = _any_blend or _b25 or _b35 or _b45

    # --- Expected total goals: projected future + already scored ---
    expected_total = proj_xg_signal + proj_xg_opponent + current_total_goals
    
    # --- BTTS probability ---
    # v10.44h: Account for goals already scored.
    # BTTS = both teams score in the FULL match.
    # If a team already scored, they only need the other to score (at least 1 future goal).
    # If both already scored → BTTS already happened → 100%.
    if signal_goals > 0 and opp_goals > 0:
        p_btts = 1.0  # BTTS already achieved
    elif signal_goals > 0:
        # Signal team scored, need opponent to score at least 1 in future
        p_btts = 1.0 - _poisson_pmf(0, proj_xg_opponent)
    elif opp_goals > 0:
        # Opponent scored, need signal team to score at least 1 in future
        p_btts = 1.0 - _poisson_pmf(0, proj_xg_signal)
    else:
        # Neither scored yet: both must score in future
        p_signal_zero = _poisson_pmf(0, proj_xg_signal)
        p_opponent_zero = _poisson_pmf(0, proj_xg_opponent)
        p_btts = 1.0 - p_signal_zero - p_opponent_zero + (p_signal_zero * p_opponent_zero)
    
    return {
        "proj_xg_signal": round(proj_xg_signal, 2),
        "proj_xg_opponent": round(proj_xg_opponent, 2),
        "expected_total_goals": round(expected_total, 2),
        "p_over_25": round(p_over_25, 2),
        "p_over_35": round(p_over_35, 2),
        "p_over_45": round(p_over_45, 2),
        "over_lines": [(round(l, 1), round(p, 2)) for l, p in over_lines],
        # v10.74: calibrated display values + honesty tag + red-card echo
        "over_lines_cal": [(round(l, 1), round(p, 2)) for l, p in _cal_lines],
        "p_over_25_cal": round(_p25c, 2),
        "p_over_35_cal": round(_p35c, 2),
        "p_over_45_cal": round(_p45c, 2),
        "cal_mode": "blend" if _any_blend else "model",
        "team_reds": team_reds,
        "opp_reds": opp_reds,
        "rc_applied": _rc_applied,
        "p_btts": round(p_btts, 2),
        "xg_source": "api" if (signal_team_xg is not None and signal_team_xg > 0) else "sot_estimate",
        "remaining_minutes": round(remaining, 0),
        "goal_diff_at_signal": goal_diff,
        "current_total_goals": current_total_goals,
    }


def compute_ft_prediction(goal_pred: dict, sig_goals: int, opp_goals: int) -> dict:
    """v10.74: Full-time 1X2 prediction (win/draw/loss for the SIGNALED
    team) from the live state at signal time.

    Takes the already-computed goal prediction dict (whose remaining-goals
    lambdas carry the scoreline push/sit effects AND the v10.74 red-card
    lambda adjustment) plus the current scoreline, and integrates the
    independent-Poisson grid over future goals: P(final_i, final_j) summed
    into win/draw/loss buckets. This is an INFORMATIONAL prediction for
    the signal message (and pred_ft_* fields for later calibration against
    the resolved FT result) — never a gate, never GPS input, zero extra
    credits. The odds markets are NOT mixed in: the recorded 1X2 odds are
    pre-match/stale (Sep 4-7 audit) and would poison the prediction.
    """
    lam_sig = max(float(goal_pred.get("proj_xg_signal") or 0.0), 0.0)
    lam_opp = max(float(goal_pred.get("proj_xg_opponent") or 0.0), 0.0)
    p_w = p_d = p_l = 0.0
    for i in range(8):
        pi = _poisson_pmf(i, lam_sig)
        for j in range(8):
            pj = _poisson_pmf(j, lam_opp)
            fs = sig_goals + i
            fo = opp_goals + j
            if fs > fo:
                p_w += pi * pj
            elif fs == fo:
                p_d += pi * pj
            else:
                p_l += pi * pj
    s = p_w + p_d + p_l
    if s <= 0.0:
        return {"p_team": None, "p_draw": None, "p_opp": None}
    return {
        "p_team": round(p_w / s, 3),
        "p_draw": round(p_d / s, 3),
        "p_opp": round(p_l / s, 3),
    }


def _v10_74_fill_ft_actual(entry: dict, home_goals: int, away_goals: int) -> None:
    """v10.74: stamp the factual FT result (win/draw/loss for the SIGNALED
    team) onto a resolving outcome record, so pred_ft_* predictions can be
    calibrated against reality once enough samples accumulate."""
    try:
        _sig_final = home_goals if entry.get("is_home") else away_goals
        _opp_final = away_goals if entry.get("is_home") else home_goals
        entry["pred_ft_actual"] = (
            "win" if _sig_final > _opp_final
            else "draw" if _sig_final == _opp_final
            else "loss"
        )
    except Exception:
        pass


# ============================================================
# v10.1: FIELD NAME NORMALIZATION
# ============================================================
# API-Football field names can vary by API plan or version.
# A silent 0 from a wrong field name would systematically understate GPS.
# Instead of assuming one exact spelling, try multiple candidates.

STAT_ALIASES: dict[str, list[str]] = {
    "sot":                ["Shots on Goal", "Shots on Target", "Shot on Goal", "Shots on goal"],
    "total_shots":        ["Total Shots", "Total shots"],
    "shots_inside_box":   ["Shots insidebox", "Shots Inside Box", "Shots inside Box"],
    "shots_off_target":   ["Shots off Goal", "Shots off Target"],
    # v10.10: REMOVED dangerous_attacks — API-Football v3 does NOT provide this stat.
    # It was silently returning 0, costing us 8 GPS points + breaking acceleration.
    # "dangerous_attacks":  ["Dangerous Attacks", "Dangerous attacks"],
    "corner_kicks":       ["Corner Kicks", "Corners", "corner kicks"],
    "red_cards":          ["Red Cards", "Red cards"],
    "possession":         ["Ball Possession", "Ball possession", "Possession %", "Possession"],
    "fouls":              ["Fouls", "fouls"],
    "total_passes":       ["Total passes", "Total Passes"],
    "passes_accurate":    ["Passes accurate", "Passes Accurate"],
    # v10.10: xG — API-Football v3 returns stat type "expected_goals" (underscore format)
    # Previously tried "Expected Goals" which never matched. Now via STAT_ALIASES.
    "expected_goals":     ["expected_goals", "Expected Goals", "expectedGoals", "Expected goals"],
    # v10.44d: Big Chances — quality chances (SOT where GK had to make a save/attempt)
    # Available for domestic leagues (PL, La Liga, Serie A, etc.) like xG.
    # European comps may not return this — treated same as xG (0 pts when missing).
    "big_chances":        ["Big Chances", "big_chances", "Big chances"],
    # v10.60: free-tier KPI expansion — parsed from the SAME statistics
    # response, LOGGING ONLY (never in GPS, gates or tiers). Aliases cover
    # spellings seen across API-Football v3 plans; arrival is verified live
    # by the field census instead of assumed.
    "gk_saves":           ["Goalkeeper Saves", "Goalkeeper saves", "goalkeeper_saves"],
    "yellow_cards":       ["Yellow Cards", "Yellow cards", "yellow_cards"],
    "offsides":           ["Offsides", "Offside", "offsides"],
}


def get_stat(tstats: dict, stat_key: str, default: str = "0") -> str:
    """Look up a stat using normalized field name aliases.

    Tries each candidate in order, returns first non-None value.
    Falls back to default if all candidates miss or value is None.
    """
    candidates = STAT_ALIASES.get(stat_key, [stat_key])
    for candidate in candidates:
        val = tstats.get(candidate)
        if val is not None:
            return str(val).strip()
    return default


def get_stat_present(tstats: dict, stat_key: str) -> tuple[int | None, bool]:
    """v10.60: get_stat, but tells 'field arrived' apart from 'field missing'.

    Returns (value, arrived):
      arrived=True  -> some alias exists in the response (0 is a REAL zero)
      arrived=False -> no alias matched -> caller records None, never 0
    Rationale: safe_int(get_stat(...)) cannot distinguish '0 saves so far'
    from 'the API never sent the field'. The big_chances post-mortem (0 in
    every recorded poll = never actually delivered) proved this matters:
    fake zeros poison ML training data with silent missingness.
    """
    candidates = STAT_ALIASES.get(stat_key, [stat_key])
    for candidate in candidates:
        val = tstats.get(candidate)
        if val is not None:
            return safe_int(val), True
    return None, False


# ============================================================
# v10: GOAL PRESSURE SCORE (composite, all available API stats)
# ============================================================

def safe_int(val, default=0) -> int:
    """Safely parse a stat value to int. Returns default for None/N/A/empty."""
    if val is None or str(val).strip() in ("", "N/A", "None", "null"):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=None) -> float | None:
    """Safely parse a stat value to float. Returns None if unavailable."""
    if val is None or str(val).strip() in ("", "N/A"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def bc_restore_gps(score: float, big_chances: int | None) -> float:
    """v10.61: proportional redistribution of the dead big_chances weight.

    Mirrors the xG-missing pattern (missing weight -> other components
    compensate) in its simplest mechanical form: rescale by the missing
    share, 100/(100-GPS_BC_MAX). BC is dead on this API plan, so in
    production this is always min(score*100/94, 100). SHADOW value unless
    GPS_BC_REDISTRIBUTE is True.
    """
    if big_chances is not None and big_chances > 0:
        return min(float(score), 100.0)
    return min(round(float(score) * 100.0 / (100.0 - GPS_BC_MAX), 1), 100.0)


def calculate_goal_pressure_score(
    sot: int,
    total_shots: int,
    shots_inside_box: int,
    shots_off_target: int,  # v10.10: replaces dangerous_attacks (API-Football v3 has no DA)
    xg_value: float | None,
    corners: int,
    big_chances: int = 0,  # v10.44d: quality chances from API
    minute: int = 0,
    prev_state: dict | None = None,
    gps_history: list[dict] | None = None,
    possession: int = 0,  # v10.26: kept for API compat, NOT used in score (always 0 in API)
) -> tuple[float, str, dict]:
    """v10.44d: Calculate 0-100 Goal Pressure Score.

    v10.44d changes from v10.42:
      - ADDED Big Chances component (0-3 pts): min(BC * 1.5, 3)
        Big Chances = SOT where GK had to make a save/attempt.
        More predictive than corners. Missing → 0 pts (same as xG).
        When BC available: corners 4→2 pts (BC takes 2, corners keeps 2)
        When BC missing: corners 4 pts (unchanged)

    v10.58 changes:
      - Big Chances weight nearly DOUBLED: min(BC * 2.5, 6) (was 1.5/3).
        Big chances are the strongest single pre-goal stat the API offers
        (a team creating them without scoring converts soon); the old 3-pt
        cap made them near-noise in a 100-pt score.

    v10.61 changes:
      - BC-MISSING REDISTRIBUTION (shadow): big_chances is never delivered
        on this API plan, so the 6-pt BC component is always absent — and
        uncompensated when xG is present (the "94-scale" issue). gps_restored
        = bc_restore_gps(score, big_chances) is computed here and logged in
        components; the returned score is UNCHANGED unless the module flag
        GPS_BC_REDISTRIBUTE is True (audit says gates stay on the historical
        scale — see CHANGES_v10.61.md).

    v10.42 changes:
      - Adaptive GPS: xG kept for domestic leagues (18pts), redistributed
        for European comps (SOT 35, IB 27, accel 28 instead of 28/22/22).

    v10.26 changes from v10.10:
      - REMOVED possession component (5 pts)
      - REDISTRIBUTED 18 xG points: +7 SOT (28→35), +5 IB (22→27), +6 accel (22→28)

    Returns: (score_0_to_100, description, component_breakdown)
    """
    if gps_history is None:
        gps_history = []
    components = {}
    score = 0.0

    # v10.42: xG is available for domestic leagues but NOT European competitions.
    # When xG is available: use original caps (SOT 28, IB 22, accel 22) + xG (18).
    # When xG is missing: redistribute 18pts to SOT/IB/accel (35/27/28).
    has_xg = xg_value is not None and xg_value >= 0.01
    has_bc = big_chances > 0  # v10.44d: big chances available
    accel_cap = 22 if has_xg else 28

    # === 1. SOT component ===
    # Absolute SOT matters, but with diminishing returns.
    if has_xg:
        # Domestic leagues: SOT 0-28 (original)
        sot_table = {0: 0, 1: 4, 2: 10, 3: 20, 4: 25}
        sot_pts = sot_table.get(min(sot, 5), 28)
    else:
        # European comps: SOT 0-35 (+7 from xG redistribution)
        sot_table = {0: 0, 1: 5, 2: 12, 3: 22, 4: 30}
        sot_pts = sot_table.get(min(sot, 5), 35)
    score += sot_pts
    components["sot"] = sot_pts

    # === 2. Shots inside box ratio ===
    # High inside-box ratio = quality shot selection = more dangerous.
    ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
    if has_xg:
        ib_pts = min(ib_ratio * 30, 22)  # Domestic: 73%+ = full 22 pts (original)
    else:
        ib_pts = min(ib_ratio * 36, 27)  # European: 75%+ = full 27 pts (+5 from xG)
    score += ib_pts
    components["inside_box"] = round(ib_pts, 1)

    # === 3. Shot volume relative to minute (0-14 points) ===
    # More shots per minute = sustained attacking intent.
    if minute > 0:
        shots_per_min = total_shots / minute
        sv_pts = min(shots_per_min * 140, 14)  # ~0.1/min = 14 pts
    else:
        sv_pts = 0.0
        shots_per_min = 0.0
    score += sv_pts
    components["shot_vol"] = round(sv_pts, 1)

    # === 4. xG (0-18 points) — domestic leagues only ===
    # v10.42: European competitions (CL/EL/CL) never return xG.
    # Domestic leagues (Serie A, PL, La Liga, etc.) do return xG.
    # When available, use original 18pt xG component.
    # When missing, 18pts redistributed to SOT/IB/accel above.
    if has_xg:
        xg_pts = min(xg_value * 18, 18)
    else:
        xg_pts = 0.0
    score += xg_pts
    components["xg"] = round(xg_pts, 1)

    # === 5. Possession — REMOVED in v10.26 ===
    # API-Football v3 NEVER returns possession data (0 in all 2,168 polls across
    # 12 leagues). The 5 points were redistributed: +3 xG, +2 acceleration cap.
    # Kept as 0 for backward compat in components dict and JSONL records.
    components["possession"] = 0.0

    # v10.10: shots_off_target stored for acceleration calc
    sot_rate_delta = 0.0

    # === 6. Corner kicks (0-2 or 0-4 points) ===
    # Set pieces = additional scoring opportunities.
    # v10.44d: reduced to 2 pts when Big Chances available (BC more predictive)
    corner_cap = 2 if (has_bc or has_xg) else 4
    corner_pts = min(corners * 1.0, corner_cap)
    score += corner_pts
    components["corners"] = corner_pts

    # === 6b. Big Chances (0-6 points) — v10.44d, doubled in v10.58 ===
    # Quality chances where GK had to make a save/attempt.
    # More predictive than corners: directly measures chance quality.
    # v10.58: the strongest single "goal is coming" stat the API offers, so
    # its voice is nearly doubled: 2.5 pts per BC (was 1.5), cap 6 (was 3).
    # When missing (European comps): 0 pts, corners keeps full 4 pts.
    bc_pts = 0.0
    if has_bc:
        bc_pts = min(big_chances * 2.5, 6)
    score += bc_pts
    components["big_chances"] = bc_pts

    # === 7. ACCELERATION component (0-15 points) — THE KEY DIFFERENTIATOR ===
    # v10.10: Uses WINDOWED per-minute rates over GPS_WINDOW_MINUTES game minutes.
    # v10.10: Replaced DA delta with shots_off_target delta (DA doesn't exist in API).
    #    Shots off target rising = team is getting shots away even if not on target.
    accel_pts = 0.0
    accel_details = ""
    sustained_count = 0
    sot_burst_pts = 0  # v10.17
    goalless_pts = 0   # v10.17

    if prev_state and prev_state.get("last_minute", 0) > 0:
        mins_passed = max(minute - prev_state["last_minute"], 1)
        sot_delta = sot - prev_state.get("last_sot", 0)
        ts_delta = total_shots - prev_state.get("last_total_shots", 0)
        sot_delta_var = shots_off_target - prev_state.get("last_shots_off_target", 0)
        prev_xg = prev_state.get("last_xg")
        xg_delta = 0.0
        if xg_value is not None and prev_xg is not None:
            try:
                xg_delta = xg_value - float(prev_xg)
            except (ValueError, TypeError):
                pass

        # v10.10: Per-minute rates (interval-independent thresholds)
        sot_rate = sot_delta / mins_passed
        ts_rate = ts_delta / mins_passed
        sotr_rate = sot_delta_var / mins_passed  # shots off target rate

        # v10.10: Count accelerating indicators using RATE thresholds.
        accel_count = 0
        if sot_rate >= 0.15:       # 1 SOT per ~7 min or faster
            accel_count += 1
        if ts_rate >= 0.3:        # 1 shot per ~3 min or faster
            accel_count += 1
        if sotr_rate >= 0.2:      # 1 off-target shot per 5 min (replaces DA)
            accel_count += 1
        if xg_delta >= 0.1 and (xg_delta / mins_passed) >= 0.02:
            accel_count += 1

        # Windowed rates over last GPS_WINDOW_MINUTES game minutes
        window_sot_rate = sot_rate
        window_ts_rate = ts_rate
        window_sotr_rate = sotr_rate
        if len(gps_history) >= 2:
            window_start_min = minute - GPS_WINDOW_MINUTES
            window_entry = None
            for entry in gps_history:
                if entry.get("minute", 0) >= window_start_min:
                    window_entry = entry
                    break
            if window_entry:
                window_mins = max(minute - window_entry["minute"], 1)
                window_sot_rate = (sot - window_entry.get("sot", 0)) / window_mins
                window_ts_rate = (total_shots - window_entry.get("total_shots", 0)) / window_mins
                window_sotr_rate = (shots_off_target - window_entry.get("shots_off_target", 0)) / window_mins

                # Use windowed rates if they show MORE acceleration
                window_accel = 0
                if window_sot_rate >= 0.15:
                    window_accel += 1
                if window_ts_rate >= 0.3:
                    window_accel += 1
                if window_sotr_rate >= 0.2:
                    window_accel += 1
                if xg_value is not None and window_entry.get("xg") is not None:
                    try:
                        wxg_delta = xg_value - float(window_entry["xg"])
                        if wxg_delta >= 0.1 and wxg_delta / window_mins >= 0.02:
                            window_accel += 1
                    except (ValueError, TypeError):
                        pass
                if window_accel > accel_count:
                    accel_count = window_accel

        if accel_count >= 3:
            accel_pts = 15
        elif accel_count >= 2:
            accel_pts = 10
        elif accel_count == 1:
            accel_pts = 4

        # SUSTAINED pressure bonus
        for h in gps_history:
            if h.get("accel_count", 0) >= 2:
                sustained_count += 1
        if accel_count >= 2 and sustained_count >= 1:
            accel_pts = min(accel_pts + 4, 15)

        # v10.17: SOT BURST bonus — SOT jumped 2+ in one poll interval
        # A team going from 0→2 or 1→3 SOT in 90-150s is an extreme
        # pressure spike. Much stronger than gradual SOT=1 acceleration.
        sot_burst_pts = 0
        sot_burst_detail = ""
        if sot_delta >= 2:
            burst_base = 6 if sot_delta >= 3 else 3
            # Combo: if GPS is also rising vs last poll, extra bonus
            gps_rising = len(gps_history) >= 1 and score > gps_history[-1].get("gps", 0)
            combo_extra = 3 if gps_rising else 0
            sot_burst_pts = min(burst_base + combo_extra, 8)
            sot_burst_detail = (
                f" BURST+{sot_delta}SOT({burst_base}"
                f"{'+3combo' if combo_extra else ''})"
            )
        accel_pts = min(accel_pts + sot_burst_pts, accel_cap)  # v10.42: cap varies by xG availability

        # v10.17: GOALLESS PRESSURE bonus — sustained high GPS, no conversion
        # If a team has GPS >= 50 for 3+ recent polls, they're dominating
        # but not scoring — odds of a breakthrough goal keep rising.
        # If a goal WAS scored, pressure typically drops, resetting the count.
        goalless_count = sum(1 for h in gps_history if h.get("gps", 0) >= 50)
        goalless_pts = 0
        if goalless_count >= 4:
            goalless_pts = 5
        elif goalless_count >= 3:
            goalless_pts = 3
        elif goalless_count >= 2:
            goalless_pts = 1
        accel_pts = min(accel_pts + goalless_pts, accel_cap)  # v10.42: cap varies by xG availability

        accel_details = (
            f"SOT {sot_rate:.2f}/m(w:{window_sot_rate:.2f}), "
            f"TS {ts_rate:.2f}/m(w:{window_ts_rate:.2f}), "
            f"SOffT {sotr_rate:.1f}/m(w:{window_sotr_rate:.1f})"
            + (f", xG+{xg_delta:.2f}" if xg_delta > 0 else "")
            + f" [{accel_count} accel{' sustained' if sustained_count > 0 and accel_count >= 2 else ''}"
            + (f"{sot_burst_detail}" if sot_burst_detail else "")
            + (f" goalless:{goalless_count}" if goalless_count >= 2 else "")
            + "]"
        )

    score += accel_pts
    components["acceleration"] = round(accel_pts, 1)
    components["sustained"] = sustained_count  # v10.1: for backtesting
    components["sot_burst"] = sot_burst_pts
    components["goalless_pressure"] = goalless_pts

    score = min(score, 100.0)

    # === v10.61: BC-MISSING REDISTRIBUTION (shadow by default) ===
    gps_restored = bc_restore_gps(score, big_chances)
    components["gps_restored"] = round(gps_restored, 1)
    if GPS_BC_REDISTRIBUTE:
        score = gps_restored

    desc = (
        f"GPS:{score:.0f} SOT:{sot}({components['sot']}pts) "
        f"IB:{ib_ratio:.0%}({components['inside_box']}pts) "
        f"TS/min:{shots_per_min:.2f}({components['shot_vol']}pts) "
        f"xG:{xg_value or 0:.2f}({components['xg']}pts) "
        f"BC:{big_chances}({bc_pts:.0f}pts) "
        f"Accel:{accel_pts:.0f}pts"
    )
    if accel_details:
        desc += f" | {accel_details}"

    return score, desc, components



def _goal_shots_in_window(fid: int, tid: int, minute: int, window: int = 10) -> int:
    """v10.84: goal shots registered inside the trailing window.

    Netted out of attempt-burst evidence — the shot that scored is the thing
    itself, not advance warning (v10.56 principle applied to shot volume).
    """
    return sum(
        1 for m in (_goal_shot_minutes.get((fid, tid)) or [])
        if m >= minute - window
    )


def _attempt_burst_evaluate(
    fid: int, tid: int, minute: int, sot: int,
    shots_delta_10m, ib_delta_10m, shots_inside_box: int,
):
    """v10.84: ATTEMPT-BURST — shot-volume bursts the SOT channel cannot see.

    Sep 9 backtest (5,398 team-polls, base P(goal in 10m) 11.4%):
    attempts d10>=3 -> 21.8% (1.9x), d10>=4 -> 37.1% (3.3x); episode level
    26 alert-episodes, 38% 10m hit (current stack: 17%). Ledger-wide, the
    signals that carried att d10>=3 hit 75-80% FULL (n=8) vs 53% baseline.
    The SOT tier freezes exactly when a team peppers the goal wide or saved
    (Twente 65', Arsenal 75', Barcelona 78', Lille-Betis 54' class) —
    attempts do not. Goal shots are netted out first.

    Returns ("EARLY WARNING", tag) when it fires, else (None, "").
    """
    if shots_delta_10m is None:
        return None, ""
    if not (21 <= minute <= 79):
        return None, ""
    att_eff = shots_delta_10m - _goal_shots_in_window(fid, tid, minute, 10)
    if att_eff < ATTEMPT_BURST_MIN:
        return None, ""
    if sot < 1:
        return None, ""
    _ib_ok = (ib_delta_10m or 0) >= ATTEMPT_BURST_IB_DELTA or shots_inside_box >= 3
    if not _ib_ok:
        return None, ""
    _strong = "STRONG " if att_eff >= ATTEMPT_BURST_MIN + 1 else ""
    return "EARLY WARNING", (
        f"ATTEMPT-BURST {_strong}d10 +{att_eff} (SOT {sot}, net of goal shots)"
    )


def _response_window_evaluate(
    fid: int, tid: int, minute: int, team_goals: int, opp_goals: int,
    sot: int, gps: float,
):
    """v10.84: RESPONSE WINDOW — the 10 minutes after conceding.

    Sep 7-9 evidence: 3/21 Sep-9 goals plus the Sep-8 Lille 36' / Dortmund 72'
    class were concede->answer within 2-8 minutes with ZERO stat signature —
    only the timing pattern sees them. Level-score teams promote to EARLY
    WARNING at a reduced floor; trailing-by-1 stays SHADOW because the v10.34
    losing filter (22.2% full WR for losing EW) outranks one day of evidence.

    Returns (tier, tag, is_shadow).
    """
    if not (21 <= minute <= 79):
        return None, "", False
    conceded = False
    for gm, gtid in (_fixture_goal_log.get(fid) or []):
        if gtid != tid and 0 <= minute - gm <= RESPONSE_WINDOW_MIN:
            conceded = True
            break
    if not conceded:
        return None, "", False
    if sot < 1 or gps < RESPONSE_GPS_FLOOR:
        return None, "", False
    diff = team_goals - opp_goals
    tag = (
        f"RESPONSE WINDOW (conceded <= {RESPONSE_WINDOW_MIN}m ago, "
        f"score {team_goals}-{opp_goals})"
    )
    if diff == 0:
        return "EARLY WARNING", tag, False
    if diff == -1:
        return None, tag, True   # shadow: losing filter outranks (v10.34 data)
    return None, "", False


def _build_recency_fields(
    fid: int, tid: int, minute: int,
    sot: int, total_shots: int, shots_inside_box: int,
    shots_off_target: int, xg_value: float | None, corners: int,
    gps: float, accel_count: int,
) -> dict:
    """v10.28: Compute enriched recency fields from GPS history.

    For every signal (and poll), compute historical snapshots at
    previous poll, ~5 game minutes ago, and ~10 game minutes ago.
    Also compute deltas, GPS change, seconds since last signal, and
    a recency_ratio that measures how "fresh" the pressure is.

    recency_ratio = (SOT_last_10m_delta) / max(SOT_total, 1)
    High ratio = most SOT arrived recently (fresh pressure).
    Low ratio = SOT accumulated over a long period (stale pressure).
    """
    history = team_gps_history.get((fid, tid), [])
    fields = {
        # Previous poll values
        "prev_poll_sot": None, "prev_poll_shots": None,
        "prev_poll_ib": None, "prev_poll_xg": None,
        "prev_poll_corners": None, "prev_poll_gps": None,
        # 5 game minutes ago
        "sot_5m_ago": None, "shots_5m_ago": None,
        "ib_5m_ago": None, "xg_5m_ago": None,
        "corners_5m_ago": None, "gps_5m_ago": None,
        # 10 game minutes ago
        "sot_10m_ago": None, "shots_10m_ago": None,
        "ib_10m_ago": None, "xg_10m_ago": None,
        "corners_10m_ago": None, "gps_10m_ago": None,
        # Deltas from previous poll
        "delta_sot": None, "delta_shots": None,
        "delta_ib": None, "delta_xg": None,
        "delta_corners": None, "delta_gps": None,
        # 5min and 10min window deltas
        "sot_delta_5m": None, "shots_delta_5m": None,
        "xg_delta_5m": None, "ib_delta_5m": None,
        "corners_delta_5m": None,
        "sot_delta_10m": None, "shots_delta_10m": None,
        "xg_delta_10m": None, "ib_delta_10m": None,
        "corners_delta_10m": None,
        # Derived
        "seconds_since_prev_signal": None,
        "gps_change": None,
        "recency_ratio": None,
    }

    # v10.84: the minute archive can carry the window lookback even when
    # the live history is empty — only bail when BOTH are empty.
    if not history and not _minute_history.get((fid, tid)):
        return fields

    # Previous poll (most recent entry before this one)
    prev = history[-1] if history else None
    if prev:
        fields["prev_poll_sot"] = prev.get("sot")
        fields["prev_poll_shots"] = prev.get("total_shots")
        fields["prev_poll_ib"] = prev.get("shots_inside_box")
        fields["prev_poll_xg"] = prev.get("xg")
        fields["prev_poll_corners"] = prev.get("corners")
        fields["prev_poll_gps"] = round(prev.get("gps", 0), 1)

        # Deltas from previous poll
        fields["delta_sot"] = sot - (prev.get("sot") or 0)
        fields["delta_shots"] = total_shots - (prev.get("total_shots") or 0)
        fields["delta_ib"] = shots_inside_box - (prev.get("shots_inside_box") or 0)
        prev_xg = prev.get("xg")
        if xg_value is not None and prev_xg is not None:
            try:
                fields["delta_xg"] = round(xg_value - float(prev_xg), 3)
            except (ValueError, TypeError):
                pass
        fields["delta_corners"] = corners - (prev.get("corners") or 0)
        fields["delta_gps"] = round(gps - prev.get("gps", 0), 1)

    # v10.84: search the minute-resolution archive first — the live history
    # (GPS_HISTORY_MAX=5) spans ~75s at the 15s cadence and can never reach
    # 5/10 game minutes back, so these lookups were structurally returning
    # None (window deltas null on 99.8% of Sep-9 polls). Fall back to the
    # live history only when no archive exists (cold start).
    arc = _minute_history.get((fid, tid), [])
    _src = arc if arc else history
    entry_5m = None
    entry_10m = None
    for entry in reversed(_src):
        e_min = entry.get("minute", 0)
        if entry_5m is None and e_min <= minute - 5:
            entry_5m = entry
        if entry_10m is None and e_min <= minute - 10:
            entry_10m = entry
        if entry_10m is not None:
            break

    if entry_5m:
        fields["sot_5m_ago"] = entry_5m.get("sot")
        fields["shots_5m_ago"] = entry_5m.get("total_shots")
        fields["ib_5m_ago"] = entry_5m.get("shots_inside_box")
        fields["xg_5m_ago"] = entry_5m.get("xg")
        fields["corners_5m_ago"] = entry_5m.get("corners")
        fields["gps_5m_ago"] = round(entry_5m.get("gps", 0), 1)

        fields["sot_delta_5m"] = sot - (entry_5m.get("sot") or 0)
        fields["shots_delta_5m"] = total_shots - (entry_5m.get("total_shots") or 0)
        e5m_xg = entry_5m.get("xg")
        if xg_value is not None and e5m_xg is not None:
            try:
                fields["xg_delta_5m"] = round(xg_value - float(e5m_xg), 3)
            except (ValueError, TypeError):
                pass
        fields["ib_delta_5m"] = shots_inside_box - (entry_5m.get("shots_inside_box") or 0)
        fields["corners_delta_5m"] = corners - (entry_5m.get("corners") or 0)

    if entry_10m:
        fields["sot_10m_ago"] = entry_10m.get("sot")
        fields["shots_10m_ago"] = entry_10m.get("total_shots")
        fields["ib_10m_ago"] = entry_10m.get("shots_inside_box")
        fields["xg_10m_ago"] = entry_10m.get("xg")
        fields["corners_10m_ago"] = entry_10m.get("corners")
        fields["gps_10m_ago"] = round(entry_10m.get("gps", 0), 1)

        fields["sot_delta_10m"] = sot - (entry_10m.get("sot") or 0)
        fields["shots_delta_10m"] = total_shots - (entry_10m.get("total_shots") or 0)
        e10m_xg = entry_10m.get("xg")
        if xg_value is not None and e10m_xg is not None:
            try:
                fields["xg_delta_10m"] = round(xg_value - float(e10m_xg), 3)
            except (ValueError, TypeError):
                pass
        fields["ib_delta_10m"] = shots_inside_box - (entry_10m.get("shots_inside_box") or 0)
        fields["corners_delta_10m"] = corners - (entry_10m.get("corners") or 0)

    # Seconds since previous signal for this team
    team_sig = signaled_teams.get((fid, tid))
    if team_sig and team_sig.get("last_signal_time", 0) > 0:
        fields["seconds_since_prev_signal"] = round(time.time() - team_sig["last_signal_time"])

    # GPS change (current vs earliest history entry)
    if len(history) >= 1:
        fields["gps_change"] = round(gps - history[0].get("gps", 0), 1)

    # Recency ratio: what fraction of SOT arrived in the last 10 game minutes?
    # High = fresh pressure, Low = accumulated pressure
    sot_10m_ago = fields.get("sot_10m_ago")
    if sot_10m_ago is not None and sot > 0:
        fields["recency_ratio"] = round(
            (sot - sot_10m_ago) / sot, 3
        )
    elif entry_5m is not None and sot > 0:
        # Fallback: use 5m window if no 10m data
        sot_5m_ago = fields.get("sot_5m_ago")
        if sot_5m_ago is not None:
            fields["recency_ratio"] = round(
                (sot - sot_5m_ago) / sot, 3
            )

    return fields


def record_pressure_poll(
    fid: int, tid: int, tname: str, league: str,
    minute: int, sot: int, total_shots: int,
    shots_inside_box: int, shots_off_target: int,
    xg_value: float | None, corners: int,
    gps: float, components: dict,
    accel_count: int, is_home: bool,
    score_home: int, score_away: int,
    big_chances: int = 0,  # v10.44d
    possession: int = 0,
    stats_sot_raw: int = -1,  # v10.31
    events_sot: int = 0,      # v10.31
    # v10.44k: Opponent stats for ML (contextual pressure balance)
    opp_sot: int = 0,
    opp_total_shots: int = 0,
    opp_shots_inside_box: int = 0,
    opp_xg: float | None = None,
    opp_corners: int = 0,
    opp_big_chances: int = 0,
    opp_gps: float = 0.0,
    league_id: int = 0,  # v10.44k: integer league ID for ML grouping
    ml_score: float | None = None,  # v10.59: ML shadow opinion for this poll
    # v10.60: free-tier KPI expansion (LOGGING ONLY, null-safe)
    gk_saves: int | None = None,
    opp_gk_saves: int | None = None,
    fouls: int | None = None,
    opp_fouls: int | None = None,
    offsides: int | None = None,
    opp_offsides: int | None = None,
    yellow_cards: int | None = None,
    opp_yellow_cards: int | None = None,
    total_passes: int | None = None,
    pass_accuracy: float | None = None,
    blocked_shots: int | None = None,
    subst_count: int | None = None,
    subst_latest_minute: int | None = None,
    card_latest_minute: int | None = None,
    # v10.73: events-based red-card counts (logging only, null-safe)
    red_cards: int | None = None,
    opp_red_cards: int | None = None,
) -> None:
    """v10.1: Write poll-level data to JSONL for backtesting.

    This records EVERY stats poll (not just signals), building the dataset
    to answer: "which combination of stats predicts a goal within 5/10/15 min?"

    v10.26: Added real data_quality field (fraction of non-null/non-zero fields)
    replacing the previous hardcoded 0.5 that never varied.

    v10.28: Added recency fields (prev poll, 5m/10m ago, deltas, recency_ratio).

    v10.44k: Added opponent stats (SOT, shots, IB, xG, corners, big_chances, GPS)
    so ML can learn whether opponent pressure reduces goal probability.
    Also added league_id (int) for league-tier grouping.

    v10.59: Added 'ml' — the frozen model's shadow opinion on this poll, so
    offline analysis can ask "who was high EARLIER, GPS or ML?".

    v10.60: Added the free-tier KPI expansion fields (gk_saves, fouls,
    offsides, yellow cards, pass volume/accuracy + events-derived blocked
    shots / subs / cards). LOGGING ONLY — None means the API did not deliver
    the field; never fake zeros (big_chances lesson).
    """
    try:
        # v10.26: Compute real data quality score (0.0-1.0)
        # Fraction of key fields that are populated (non-null, non-zero where applicable)
        key_fields = {
            "sot": sot > 0,
            "total_shots": total_shots > 0,
            "shots_inside_box": shots_inside_box > 0,
            "shots_off_target": shots_off_target > 0,
            "xg": xg_value is not None and xg_value >= 0,
            "corners": corners >= 0,  # always present but check anyway
            "possession": possession > 0,  # always False (dead field)
        }
        data_quality = sum(1 for v in key_fields.values() if v) / len(key_fields)

        # v10.28: Compute recency fields from GPS history
        recency = _build_recency_fields(
            fid, tid, minute, sot, total_shots,
            shots_inside_box, shots_off_target, xg_value, corners,
            gps, accel_count,
        )

        entry = {
            "ts": time.time(),
            "fixture_id": fid, "team_id": tid, "team_name": tname,
            "league": league, "minute": minute, "is_home": is_home,
            "sot": sot, "total_shots": total_shots,
            "shots_inside_box": shots_inside_box,
            "shots_off_target": shots_off_target,  # v10.10
            "xg": round(xg_value, 3) if xg_value is not None else None,
            "big_chances": big_chances,  # v10.44d
            "corners": corners,
            "possession": possession,  # v10.26: kept for compat, always 0
            "data_quality": round(data_quality, 2),  # v10.26: REAL quality score
            "gps": round(gps, 1),
            "gps_restored": components.get("gps_restored"),  # v10.61: BC-redistributed shadow value
            "ml": round(ml_score, 1) if ml_score is not None else None,  # v10.59: shadow opinion
            "gps_sot": components.get("sot", 0),
            "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0),
            "gps_xg": components.get("xg", 0),
            "gps_bc": components.get("big_chances", 0),  # v10.44d
            "gps_corners": components.get("corners", 0),  # v10.44d-patch: was missing
            "gps_poss": components.get("possession", 0),  # v10.26: always 0 (deprecated)
            "gps_accel": components.get("acceleration", 0),
            "gps_sot_burst": components.get("sot_burst", 0),  # v10.17
            "gps_goalless": components.get("goalless_pressure", 0),  # v10.17
            "sustained": components.get("sustained", 0),
            "accel_count": accel_count,
            "score_home": score_home, "score_away": score_away,
            "stats_sot_raw": stats_sot_raw,  # v10.31: audit trail
            "events_sot": events_sot,        # v10.31: audit trail
            # v10.44k: Opponent stats for ML
            "opp_sot": opp_sot,
            "opp_total_shots": opp_total_shots,
            "opp_shots_inside_box": opp_shots_inside_box,
            "opp_xg": round(opp_xg, 3) if opp_xg is not None else None,
            "opp_corners": opp_corners,
            "opp_big_chances": opp_big_chances,
            "opp_gps": round(opp_gps, 1),
            "league_id": league_id,  # v10.44k: integer for ML grouping
            # v10.44r: Goal priority context in poll data
            "goal_in_priority_window": fid in goal_priority_until,
            # v10.60: free-tier KPI expansion — LOGGING ONLY, never in GPS.
            # None = field not delivered by the API (never fake zeros).
            "gk_saves": gk_saves, "opp_gk_saves": opp_gk_saves,
            "fouls": fouls, "opp_fouls": opp_fouls,
            "offsides": offsides, "opp_offsides": opp_offsides,
            "yellow_cards": yellow_cards, "opp_yellow_cards": opp_yellow_cards,
            # v10.73: events-based red cards (None = no events coverage)
            "red_cards": red_cards, "opp_red_cards": opp_red_cards,
            "total_passes": total_passes, "pass_accuracy": pass_accuracy,
            # events-derived (only when an events response covered the fixture)
            "blocked_shots": blocked_shots,
            "subst_count": subst_count,
            "subst_latest_minute": subst_latest_minute,
            "card_latest_minute": card_latest_minute,
        }
        # v10.28: Merge recency fields into poll entry
        entry.update(recency)
        with open(POLL_DATA_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.warning(f"  Failed to save poll data: {e}")

    # v10.75: BOX-BURST shadow evaluation — logging-only, never sent,
    # zero credits (runs on the stats this poll already fetched). Kept
    # OUTSIDE the recording try so a shadow bug can never lose a poll.
    try:
        evaluate_boxburst_shadow(
            fid, tid, tname, league, minute, sot, shots_inside_box, total_shots,
            gps, is_home, score_home, score_away,
            red_cards=red_cards, opp_red_cards=opp_red_cards,
        )
    except Exception as _e:
        log.debug(f"  v10.75 box-burst eval error: {_e}")

    # v10.76: GOAL-BURST shadow evaluation — the banked-goals/totals path
    # (the Lille-Betis class: goals as evidence, not just pre-goal pressure).
    # Same discipline: OUTSIDE the recording try, zero extra credits.
    try:
        evaluate_goalburst_shadow(
            fid, tname, league, minute, score_home, score_away,
            sot, opp_sot, gps, opp_gps, is_home,
            red_cards=red_cards, opp_red_cards=opp_red_cards,
        )
    except Exception as _e:
        log.debug(f"  v10.76 goal-burst eval error: {_e}")


def get_accel_count_from_state(
    sot: int, total_shots: int, shots_off_target: int,  # v10.10: replaces dangerous_attacks
    xg_value: float | None, prev_state: dict | None, minute: int,
) -> int:
    """v10.10: Count accelerating indicators using RATE-based thresholds.

    v10.10: DA replaced with shots_off_target (API-Football v3 has no DA).
    Must match the rate thresholds in calculate_goal_pressure_score.
    Used by pressure-accelerating fixture detection for 60s polling.
    Returns count of accelerating indicators (0-4).
    """
    if not prev_state or prev_state.get("last_minute", 0) <= 0:
        return 0

    mins_passed = max(minute - prev_state.get("last_minute", 0), 1)
    accel_count = 0
    sot_delta = sot - prev_state.get("last_sot", 0)
    ts_delta = total_shots - prev_state.get("last_total_shots", 0)
    sotr_delta = shots_off_target - prev_state.get("last_shots_off_target", 0)

    # v10.10: Same rate thresholds as GPS calculation
    if (sot_delta / mins_passed) >= 0.15:
        accel_count += 1
    if (ts_delta / mins_passed) >= 0.3:
        accel_count += 1
    if (sotr_delta / mins_passed) >= 0.2:  # replaces DA threshold
        accel_count += 1
    if xg_value is not None:
        prev_xg = prev_state.get("last_xg")
        if prev_xg is not None:
            try:
                xg_delta = xg_value - float(prev_xg)
                if xg_delta >= 0.1 and (xg_delta / mins_passed) >= 0.02:
                    accel_count += 1
            except (ValueError, TypeError):
                pass
    return accel_count


def is_fixture_pressure_accelerating(fid: int) -> bool:
    """v10: Check if ANY team in this fixture is pressure-accelerating.

    Used by polling interval logic to give 60s polling to fixtures
    where pressure is building BEFORE SOT reaches 2-3.
    """
    for (f, _), history in team_gps_history.items():
        if f == fid and len(history) >= 2:
            curr = history[-1]
            prev = history[-2]
            # GPS rising AND 2+ indicators accelerating
            if (curr.get("gps", 0) > prev.get("gps", 0)
                    and curr.get("accel_count", 0) >= 2):
                return True
    return False


# ============================================================
# LOCAL FILTERING (zero API cost)
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[int, int, int]]:
    """Find all candidate teams from live fixtures in tracked leagues.

    v9.5: No scoreline filtering — all fixtures in the 20'-80' window
    are candidates. Scoreline affects RANKING, not inclusion.
    """
    candidates = []
    for fixture in fixtures:
        lid = fixture["league"]["id"]
        if not is_tracked_match(fixture):
            continue
        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue
        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

        # v10.15: pre-window — allow from (MINUTE_MIN - PRE_WINDOW_MINUTES)
        # Signal sending still requires minute >= MINUTE_MIN
        # v10.13.2: Extended window for SOT-accelerating fixtures
        fid = fixture["fixture"]["id"]
        best_sot = get_fixture_best_sot(fid)
        sot_accel = fid in accelerating_fixtures or fid in pressure_accelerating
        is_extended = sot_accel or best_sot >= 2

        # v10.19.2: Post-restart safety — if team_state has no record for this
        # fixture (bot crashed and restarted), we don't know its SOT history.
        # Default to extended window so we don't miss SOT>=3 safety net triggers.
        if not is_extended and minute >= MINUTE_MIN:
            has_state = any(f == fid for f, _ in team_state)
            if not has_state:
                is_extended = True

        min_allowed = (MINUTE_MIN - PRE_WINDOW_MINUTES)
        # v10.20: Always use extended max for candidate inclusion.
        # Same chicken-and-egg as early window: can't know late SOT buildup
        # if we stop polling at 61'. LATE GATE protects signal quality.
        # v10.31: Event-extended fixtures get 90' ceiling (event SOT > stats SOT)
        max_allowed = EXTENDED_MAX
        if fid in event_extended_fixtures:
            max_allowed = 90
        if minute < min_allowed or minute > max_allowed:
            continue

        home_tid = fixture["teams"]["home"]["id"]
        away_tid = fixture["teams"]["away"]["id"]

        # Both teams are candidates — ranking decides priority
        candidates.append((fid, home_tid, rank_candidate(fixture, home_tid)))
        candidates.append((fid, away_tid, rank_candidate(fixture, away_tid)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def cleanup_state(live_fixture_ids: set[int]):
    to_delete = [k for k in team_state if k[0] not in live_fixture_ids]
    for k in to_delete:
        del team_state[k]
    if to_delete:
        log.info(f"  Cleaned up state for {len(to_delete)} ended fixture(s)")
    for fid in list(signaled_fixtures):
        if fid not in live_fixture_ids:
            signaled_fixtures.discard(fid)
    # v9.5.4: signaled_teams cleanup moved to EOD clear.
    # v10.44s: Do NOT delete signaled_teams/team_cooldown_polls here.
    # A fixture can momentarily drop from /fixtures?live=all (API glitch,
    # HT status transition, brief timeout). Deleting these entries causes
    # sig_num to reset to 1 on the next signal, producing ghost first-signals.
    # These dicts are small; stale entries are cleared at EOD.
    for fid in list(last_stats_check):
        if fid not in live_fixture_ids:
            del last_stats_check[fid]
    for fid in list(fast_priority):
        if fid not in live_fixture_ids:
            del fast_priority[fid]
    for fid in list(fast_sot_until):
        if fid not in live_fixture_ids:
            del fast_sot_until[fid]
    # v9.7: Clean acceleration tracking
    for fid in list(accelerating_fixtures):
        if fid not in live_fixture_ids:
            accelerating_fixtures.discard(fid)
    # v9.7: Clean dead fixtures
    for fid in list(dead_fixtures):
        if fid not in live_fixture_ids:
            dead_fixtures.pop(fid, None)
    # v10.44b: Clean data-dead fixtures
    for fid in list(data_dead_fixtures):
        if fid not in live_fixture_ids:
            data_dead_fixtures.pop(fid, None)
    # v10: Clean pressure acceleration tracking
    for fid in list(pressure_accelerating):
        if fid not in live_fixture_ids:
            pressure_accelerating.discard(fid)
    # v10.17: Clean SOT burst tracking
    for fid in list(sot_burst_fixtures):
        if fid not in live_fixture_ids:
            sot_burst_fixtures.discard(fid)
    for fid in list(genuine_burst_fixtures):
        if fid not in live_fixture_ids:
            genuine_burst_fixtures.discard(fid)
    # v10.56: goal-SOT ledger cleanup for finished fixtures
    for _k in list(_pending_goal_sot):
        if _k[0] not in live_fixture_ids:
            del _pending_goal_sot[_k]
    for _k in list(_goal_sot_landed):
        if _k[0] not in live_fixture_ids:
            del _goal_sot_landed[_k]
    # v10: Clean GPS history
    to_delete_gps = [k for k in team_gps_history if k[0] not in live_fixture_ids]
    for k in to_delete_gps:
        del team_gps_history[k]
    # v10.84: clean the minute archive, goal-shot minutes, fixture goal log
    for _k in list(_minute_history):
        if _k[0] not in live_fixture_ids:
            del _minute_history[_k]
    for _k in list(_goal_shot_minutes):
        if _k[0] not in live_fixture_ids:
            del _goal_shot_minutes[_k]
    for _fid in list(_fixture_goal_log):
        if _fid not in live_fixture_ids:
            del _fixture_goal_log[_fid]
    # v10.31: Clean event SOT cache and extension set
    for fid in list(event_extended_fixtures):
        if fid not in live_fixture_ids:
            event_extended_fixtures.discard(fid)
    for fid in list(_event_sot_cache):
        if fid not in live_fixture_ids:
            del _event_sot_cache[fid]
    # v10.60: event-extras cache follows the same lifecycle
    for fid in list(_event_extras_cache):
        if fid not in live_fixture_ids:
            del _event_extras_cache[fid]
    # v10.57: Top-SOT player cache + goal counters for finished fixtures
    for fid in list(_player_sot_cache):
        if fid not in live_fixture_ids:
            del _player_sot_cache[fid]
    for fid in list(_player_sot_cache_goals):
        if fid not in live_fixture_ids:
            del _player_sot_cache_goals[fid]
    for fid in list(_fixture_valid_goals):
        if fid not in live_fixture_ids:
            del _fixture_valid_goals[fid]
    # v10.58: SOT-growth snapshots for the Top-SOT player cache
    for fid in list(_player_sot_cache_built_sot):
        if fid not in live_fixture_ids:
            del _player_sot_cache_built_sot[fid]
    # v10.57 hygiene: latest-SOT-event minute map existed since v10.44p but
    # was never cleaned — clear it with the other per-fixture maps too.
    for fid in list(_latest_sot_event_minute):
        if fid not in live_fixture_ids:
            del _latest_sot_event_minute[fid]
    # v10.44r: Clean goal priority tracking
    for fid in list(goal_priority_until):
        if fid not in live_fixture_ids:
            del goal_priority_until[fid]
    for fid in list(_goal_detect_ts):
        if fid not in live_fixture_ids:
            del _goal_detect_ts[fid]
    for fid in list(_goal_game_minute):
        if fid not in live_fixture_ids:
            del _goal_game_minute[fid]
    for fid in list(_goal_stats_ts):
        if fid not in live_fixture_ids:
            del _goal_stats_ts[fid]
    _goal_stats_recorded.difference_update(live_fixture_ids)  # remove ended fids
    # v10.44p/v10.50: Clean event fast lane fixtures that finished
    global _event_fast_lane_fids
    _fl_before = _event_fast_lane_fids
    _event_fast_lane_fids = [f for f in _event_fast_lane_fids if f in live_fixture_ids]
    if len(_event_fast_lane_fids) != len(_fl_before):
        log.info("  v10.44p: Event fast lane cleared (finished fixture removed)")
    # v10.53: Clean goal-watch state for finished fixtures
    global _goal_watch_fids
    _goal_watch_fids = [f for f in _goal_watch_fids if f in live_fixture_ids]
    for fid in list(_goal_watch_seen):
        if fid not in live_fixture_ids:
            del _goal_watch_seen[fid]
    for _k in list(_goal_watch_flash_keys):
        if _k[0] not in live_fixture_ids:
            _goal_watch_flash_keys.discard(_k)
    _coldstart_warmed.difference_update(
        {f for f in _coldstart_warmed if f not in live_fixture_ids}
    )
    # v10.67: Clean late-retained fixtures that finished (belt and
    # suspenders — poll_goal_watch self-cleans on non-live status too)
    global _late_retain_fids
    for fid in list(_late_retain_fids):
        if fid not in live_fixture_ids:
            del _late_retain_fids[fid]
    # v10.54: Clean surge-watch state for finished fixtures
    # (alert records persist in surge_watch.jsonl; only in-flight state resets)
    for _k in list(_surge_seen_sot):
        if _k[0] not in live_fixture_ids:
            del _surge_seen_sot[_k]
    for _k in list(_surge_seen_shots):
        if _k[0] not in live_fixture_ids:
            del _surge_seen_shots[_k]
    for _k in list(_surge_wake_minute):
        if _k[0] not in live_fixture_ids:
            del _surge_wake_minute[_k]
    for _k in list(_surge_burst_last_minute):
        if _k[0] not in live_fixture_ids:
            del _surge_burst_last_minute[_k]
    for _k in list(_surge_burst_count):
        if _k[0] not in live_fixture_ids:
            del _surge_burst_count[_k]
    for _k in list(_surge_last_alert_ts):
        if _k[0] not in live_fixture_ids:
            del _surge_last_alert_ts[_k]
    for _k in list(_surge_flood_minute):
        if _k[0] not in live_fixture_ids:
            del _surge_flood_minute[_k]
    for fid in list(_surge_fixture_alerts):
        if fid not in live_fixture_ids:
            del _surge_fixture_alerts[fid]
    # v10.50: purge fast-lane per-event history for finished fixtures
    # (shadow records themselves persist in fastlane_shadow.jsonl for resolution)
    for _k in list(_fl_sot_events):
        if _k[0] not in live_fixture_ids:
            del _fl_sot_events[_k]
    for _k in list(_fl_seen_sot_count):
        if _k[0] not in live_fixture_ids:
            del _fl_seen_sot_count[_k]
    for _k in list(_fl_goal_minutes):
        if _k not in live_fixture_ids:
            del _fl_goal_minutes[_k]
    for _k in list(_fl_seen_goal_count):
        if _k not in live_fixture_ids:
            del _fl_seen_goal_count[_k]
    for _k in list(_fl_shadow_dedupe):
        if _k[0] not in live_fixture_ids:
            del _fl_shadow_dedupe[_k]
    # v10.75: purge box-burst runtime state for finished fixtures
    # (shadow records themselves persist in boxburst_shadow.jsonl)
    for _k in list(_boxburst_fired):
        if _k[0] not in live_fixture_ids:
            del _boxburst_fired[_k]
    for _k in list(_boxburst_ib_hist):
        if _k[0] not in live_fixture_ids:
            del _boxburst_ib_hist[_k]
    # v10.76: purge goal-burst runtime state for finished fixtures
    # (shadow records themselves persist in goalburst_shadow.jsonl)
    for _k in list(_goalburst_fired):
        if _k[0] not in live_fixture_ids:
            del _goalburst_fired[_k]


def find_cached_fixture(fid: int):
    for f in cached_fixtures:
        if f["fixture"]["id"] == fid:
            return f
    return None


def is_first_signal_only_mode() -> bool:
    """v9.5.8: On busy days (>20 games), only send first signal per team
    then move on to cover more matches. On light days (<15), full tracking."""
    return total_matches_today > FIRST_SIGNAL_ONLY_THRESHOLD


def _fixture_score_changed_since_signal(fid: int) -> bool:
    """v10.44c: Check if ANY team's goals changed since last signal.
    Used to revive signaled fixtures when game state evolved."""
    f = find_cached_fixture(fid)
    if not f:
        return False
    cur_home = f["goals"]["home"] or 0
    cur_away = f["goals"]["away"] or 0
    for (sfid, stid), sinfo in signaled_teams.items():
        if sfid != fid:
            continue
        sig_goals = sinfo.get("goals_at_last_signal", 0)
        is_home = (stid == f["teams"]["home"]["id"])
        cur_team = cur_home if is_home else cur_away
        if cur_team != sig_goals:
            return True
    return False


def is_fixture_done_first_signal(fid: int) -> bool:
    """Check if ALL teams in this fixture have sent at least one signal.
    In first-signal-only mode, such fixtures can be dropped from monitoring.
    v10.44c: Exception — if score changed since signal, not done (revive).
    v10.44e: Per-team — fixture only 'done' when BOTH teams have signaled,
    so the other team's pressure buildup isn't suppressed."""
    if not is_first_signal_only_mode():
        return False
    # Count distinct teams that have signaled in this fixture
    signaled_team_count = sum(1 for (sfid, _) in signaled_teams if sfid == fid)
    if signaled_team_count < 2:
        return False  # At least one team hasn't had their first signal yet
    # v10.44c: Score changed = game evolved, revive for re-evaluation
    if _fixture_score_changed_since_signal(fid):
        return False
    return True


def is_fixture_monitorable(fixture: dict) -> bool:
    """v9.5: strict 80' max — no late tracking beyond 80'.
    v9.5.8: In first-signal-only mode, skip already-signaled fixtures."""
    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        return False
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    # v10.15: pre-window — allow monitoring from (MINUTE_MIN - PRE_WINDOW_MINUTES)
    # v10.13.2: Extended window for SOT-accelerating fixtures
    fid = fixture["fixture"]["id"]
    best_sot = get_fixture_best_sot(fid)
    sot_accel = fid in accelerating_fixtures or fid in pressure_accelerating
    is_extended = sot_accel or best_sot >= 2

    # v10.19.2: Post-restart safety — same logic as find_candidates()
    if not is_extended and minute >= MINUTE_MIN:
        has_state = any(f == fid for f, _ in team_state)
        if not has_state:
            is_extended = True

    min_allowed = (MINUTE_MIN - PRE_WINDOW_MINUTES)
    # v10.20: Always use extended max (same fix as find_candidates)
    # v10.31: Event-extended fixtures get 90' ceiling
    max_allowed = EXTENDED_MAX
    if fid in event_extended_fixtures:
        max_allowed = 90
    if minute < min_allowed or minute > max_allowed:
        return False
    # v9.5.8: In first-signal-only mode, drop fixtures that already signaled
    fid = fixture["fixture"]["id"]
    if is_fixture_done_first_signal(fid):
        return False
    return True


# ============================================================
# SOT-SMART PRIORITY (updated for v9.5 diversification)
# ============================================================

def get_fixture_sot_priority(fid: int) -> int:
    """v10: Dynamic priority for stats polling order.

    v10 addition: pressure-accelerating fixtures (GPS rising + 2+ indicators)
    get boosted priority even at SOT=0-1, because they may be about
    to hit SOT=2-3.
    Higher = more urgent.
    """
    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    is_signaled = fid in signaled_fixtures

    if not has_state:
        base = 55
    elif best_sot >= 3:
        base = 95
    elif best_sot == 2:
        base = 90
    elif best_sot == 1:
        base = 40
    else:
        base = 10

    # v10: Boost for pressure-accelerating fixtures (pre-SOT acceleration)
    if fid in pressure_accelerating and best_sot < 2:
        base = max(base, 70)  # boost to SOT=2-level priority

    # v10.17: SOT burst — highest priority (2+ SOT in one poll)
    if fid in sot_burst_fixtures:
        base = max(base, 97)

    # v10.17: Sustained goalless pressure — team dominating but not scoring
    # Check if ANY team in this fixture has 3+ recent polls with GPS >= 50
    if best_sot >= 1:
        for (f, _), history in team_gps_history.items():
            if f == fid and len(history) >= 3:
                high_gps_streak = sum(1 for h in history if h.get("gps", 0) >= 50)
                if high_gps_streak >= 3:
                    base = max(base, 93)
                    break

    # v9.5.3: Penalize only if BOTH teams have signaled
    both_signaled = all(
        (fid, tid) in signaled_teams
        for (f, tid) in team_state if f == fid
    )
    if both_signaled and is_signaled:
        base = int(base * 0.5)

    return base


SECOND_CHANCE_BOOST = True  # v10.13: fast-repoll fixtures with SOT 1-2


def get_sot_based_interval(fid: int, base_interval: int) -> int:
    """v10.22: Pressure-aware polling interval with credit optimizations.

    v10.22 changes vs v10.21:
      - Edge zone gate: before 21' or after 60', 60s/90s fast polling only
        if best_sot>=2 OR max_gps>=50. Low-pressure games in edge zones
        can't signal (need GPS>=75) so fast polling wastes credits.
      - SOT=0 with state: 1.5x in ALL modes (was 1.0x NORMAL)
      - Pre-window SOT=0 (<21'): 2x base (can't signal, only building baseline)
      - Both-teams-signaled: 3x multiplier (was 2x, diminishing returns)

    v10.44r: Goal-triggered priority override — if a goal was just detected
    (by discovery or stats processing), poll at 15s regardless of SOT/GPS.
    This ensures post-goal pressure is caught with minimum latency.
    """
    # v10.44r: Goal priority check FIRST — overrides all other tiers.
    # A goal just happened; we need fresh stats ASAP regardless of SOT level.
    # Cost: ~4 extra polls per goal (15s x 60s window), batched with other
    # fixtures = typically 0-1 extra credits.
    _now = time.time()
    if fid in goal_priority_until:
        if _now < goal_priority_until[fid]:
            _remaining = int(goal_priority_until[fid] - _now)
            return 15  # ULTRA-FAST: same tier as pre-signal hot fixtures
        else:
            # Priority expired, clean up
            del goal_priority_until[fid]

    best_sot = get_fixture_best_sot(fid)
    has_state = best_sot > 0 or any(f == fid for f, _ in team_state)
    team_signaled_count = sum(1 for (f, t) in signaled_teams if f == fid)
    both_teams_signaled = team_signaled_count >= 2

    # Get current minute for second-chance logic
    current_minute = 0
    for (f, _), state in team_state.items():
        if f == fid:
            current_minute = max(current_minute, state.get("last_minute", 0))

    # v10.22: Edge zone gate — before 21' or after 60', fast polling
    # only if genuinely high pressure (SOT>=2 OR GPS>=50).
    # These zones require GPS>=75 to signal anyway, so burning 60s
    # polls on SOT=0/1 GPS=30 fixtures is pure waste.
    edge_zone = is_in_edge_zone(fid)
    edge_allows_fast = (not edge_zone
                        or best_sot >= 2
                        or get_fixture_max_gps(fid) >= 50)

    # v10.40: Tightened late-game gates (stale CRITICAL→block, hard stop 80', GPS>=80 floor) (60s->30s, 90s->45s).
    # With 7500 credits and batching, cost is ~4% of budget.
    # v10.44e: 15s ULTRA-FAST tier for fixtures about to signal.
    # Targets GPS>=55 + SOT>=2 (one SOT/GPS bump away from threshold).
    # Credit cost: ~+30 credits/day (batched), well within budget.
    _max_gps = get_fixture_max_gps(fid)
    _ultra_fast = (
        not both_teams_signaled
        and edge_allows_fast
        and best_sot >= 2
        and _max_gps >= 55
        and (fid in pressure_accelerating
             or fid in sot_burst_fixtures
             or fid in accelerating_fixtures
             or _max_gps >= 65)
    )
    if _ultra_fast:
        interval = 15
    # v10: Pressure acceleration — catches pre-SOT pressure build
    elif (fid in pressure_accelerating
            and not both_teams_signaled
            and best_sot < 3
            and edge_allows_fast):
        interval = 30
    # v10.17: SOT burst — 2+ SOT jump = extreme urgency
    elif fid in sot_burst_fixtures and not both_teams_signaled and edge_allows_fast:
        interval = 30
    # v9.7: SOT acceleration tier — still top priority
    elif fid in accelerating_fixtures and not both_teams_signaled and edge_allows_fast:
        interval = 30
    # v10.21: Second-chance boost — fast poll when SOT=2, clock ticking
    elif (SECOND_CHANCE_BOOST
          and best_sot == 2
          and current_minute >= 40
          and not both_teams_signaled
          and edge_allows_fast):
        interval = 30
    elif not has_state:
        interval = base_interval
    elif best_sot >= 3 and not both_teams_signaled:
        interval = 30  # v10.50: 45->30 — SOT>=3 fixtures are the most dangerous, go-max speed
    # v10.22: Fast SOT window (FastWin) also gated in edge zones
    elif best_sot >= 1 and not both_teams_signaled and is_fast_sot_active(fid) and edge_allows_fast:
        interval = 30  # v10.50: 45->30 — fast-sot window means a signal was recently near
    elif best_sot >= 2:
        # v10.50: SOT>=2 baseline 45->30s in NORMAL budget (credit guard:
        # slower ECO/EMERGENCY bases are preserved — no speedup when quota low)
        interval = min(base_interval, 30) if base_interval <= 45 else base_interval
    elif best_sot == 1:
        interval = max(30, int(base_interval * 0.75))  # v10.39: ~34s NORMAL
    else:
        # v10.21: SOT=0 with state — 1.5x base (no signal possible)
        interval = int(base_interval * 1.5)

    # v10.21: Pre-window SOT=0 — can't signal until 21', poll at 2x
    # Only building baseline state, no urgency.
    if best_sot == 0 and current_minute > 0 and current_minute < MINUTE_MIN:
        interval = max(interval, int(base_interval * 2.0))

    # v10.21: 3x if BOTH teams in this fixture have signaled (was 2x)
    # Both teams already signaled — diminishing returns on further polling.
    # v10.44p: BUT skip slowdown if one team still has extreme pressure
    # (GPS>=75 + SOT>=3 + fresh acceleration). The game is still dangerous.
    if both_teams_signaled:
        _skip_slowdown = False
        for (sfid, stid), sinfo in signaled_teams.items():
            if sfid == fid:
                # v10.50-BUGFIX: signaled_teams stores last_gps / sot_at_last_signal
                # (the old gps/sot/accel_count reads always returned 0, so the
                # v10.44p skip-slowdown for still-hot fixtures never triggered)
                st_gps = sinfo.get("last_gps", 0) or 0
                st_sot = sinfo.get("sot_at_last_signal", 0) or 0
                _gh = team_gps_history.get((sfid, stid)) or []
                st_accel = int((_gh[-1].get("accel_count", 0) or 0)) if _gh else 0
                st_minute = (team_state.get((sfid, stid), {}) or {}).get("last_minute", 0) or 0
                if st_gps >= 75 and st_sot >= 3 and st_accel >= 1 and st_minute < 75:
                    _skip_slowdown = True
                    log.info(f"  v10.44p: F{fid} both signaled but GPS={st_gps} SOT={st_sot} accel={st_accel} — skipping 3x slowdown")
                    break
        if not _skip_slowdown:
            interval = int(interval * 3.0)

    return interval


def _check_dead_revival_stats(fid: int, fixture: dict) -> str | None:
    """v10.43: Check if a dead fixture should revive based on fresh discovery data.
    Returns a reason string if revival is warranted, None otherwise.
    Reads SOT directly from the fixture's embedded stats (discovery-cached),
    not from team_state (which is stale for dead fixtures)."""
    stats_list = fixture.get("statistics", [])
    if not stats_list:
        return None

    teams_data = {}
    for s in stats_list:
        team_name = s.get("team", {}).get("name", "")
        teams_data[team_name] = s

    best_sot = 0
    for tname, tstats in teams_data.items():
        sot_val = safe_int(get_stat(tstats, "sot"))
        if sot_val > best_sot:
            best_sot = sot_val

    # Revive if either team reached SOT>=3
    if best_sot >= 3:
        return f"SOT>=3 (best={best_sot})"

    # Revive if big acceleration: check if team_state has prior data showing a surge
    # For truly dead fixtures with no team_state, skip acceleration check
    has_prior = any(f == fid for f, _ in team_state)
    if has_prior:
        for (f, tid), state in team_state.items():
            if f == fid:
                prev_sot = state.get("last_sot", 0)
                # If SOT went from 0 to 1-2 with other rising indicators, that's acceleration
                if prev_sot == 0 and best_sot >= 1:
                    prev_ts = state.get("last_shots", 0)
                    ib = state.get("last_ib_ratio", 0)
                    # Big acceleration = SOT appeared + decent inside-box ratio
                    if ib >= 0.4:
                        return f"acceleration (SOT 0->{best_sot}, IB={ib:.0%})"
    
    return None


# ============================================================
# v10.62: REDEPLOY GUARD — verify no data lost + games still tracked
# ============================================================
# The user's ask: "make sure after each redeployment no data is lost and
# all ongoing games are still being tracked." The persistence machinery
# already existed (volume files, EOD Telegram backup, cold-start warm-up,
# rebuild_signaled_teams_from_file) — what was missing was PROOF: nothing
# told the user the redeploy was clean. Two checks now do:
#   1. redeploy_data_check()  — at startup, before anything is loaded:
#      every JSONL data file is parsed line-by-line; a corrupt tail (mid-
#      append crash) is rotated to .corrupt-<ts> and the file is rewritten
#      clean (backup first, verify, atomic replace). Counts are logged.
#   2. redeploy_tracking_check(client) — once, right after the FIRST
#      discovery cycle: every live in-progress fixture is classified
#      (monitored / pickup pending / past 85' ceiling / untracked league)
#      and ONE Telegram message confirms data counts + live coverage.
# Zero extra API credits (discovery already ran); one Telegram message.

_redeploy_file_counts: dict[str, int] = {}   # basename -> ok line count
_redeploy_check_done: bool = False           # one-shot guard (set in main loop)
# v10.63: STARTUP EOD RACE FIX. Before the first discovery cycle,
# cached_fixtures is empty, so the old EOD condition ("outcomes exist AND
# no live games AND nothing monitored") was TRUE on the first loop
# iteration of every redeploy — a fresh start was mistaken for "day over"
# (Sep 4 post-mortem: 90s stall, duplicate ML backup, in-memory outcomes
# cleared mid-games, pre-restart pending outcomes orphaned until the NEXT
# redeploy). EOD may now fire only after this session has actually LOOKED
# at live state: first discovery completed, or the schedule says there
# are no tracked matches today.
_eod_lookup_ok: bool = False
_eod_defer_note_done: bool = False           # one-shot "EOD deferred" log line
# v10.63: fixture IDs the last resolve_stale_outcomes() pass saw as still
# LIVE — lets the startup retry skip the pointless 30s wait when every
# pending outcome sits on an in-progress game.
_resolve_live_fids: set = set()
# v10.74: timestamp of the resolver pass that built _resolve_live_fids —
# the boot-path midnight guard's freshness anchor (see
# _v10_74_pending_live_fids).
_resolve_live_ts: float = 0.0
_REDEPLOY_JSONL_FILES: list[str] = [
    "signal_outcomes.jsonl", "pressure_polls.jsonl", "blocked_outcomes.jsonl",
    "fastlane_shadow.jsonl", "goal_flash.jsonl", "surge_watch.jsonl",
    "boxburst_shadow.jsonl", "goalburst_shadow.jsonl",
]
_REDEPLOY_JSON_FILES: list[str] = [
    "poisson_calibration.json", "field_census.json", "sot_feed_census.json",
    "players_feed_census.json",
]


def _redeploy_check_jsonl(path: str) -> tuple[int, int, bool]:
    """Validate one JSONL file. Returns (ok_lines, bad_lines, repaired).

    If bad lines exist the file is rewritten clean: original copied to
    <path>.corrupt-<ts>, good lines written to a tmp file, tmp verified
    line-by-line, then atomic os.replace. Any failure leaves the original
    untouched (loaders skip bad lines anyway)."""
    if not os.path.exists(path):
        return -1, 0, False          # -1 = file absent (fresh volume: fine)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        log.warning(f"  REDEPLOY GUARD: cannot read {os.path.basename(path)}: {e}")
        return 0, 0, False
    lines = raw.splitlines()
    good: list[str] = []
    bad = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            json.loads(s)
            good.append(s)
        except json.JSONDecodeError:
            bad += 1
    if bad == 0:
        return len(good), 0, False
    # Repair path (rare: only after a mid-append crash)
    base = os.path.basename(path)
    ts = time.strftime("%Y%m%d%H%M%S")
    backup = f"{path}.corrupt-{ts}"
    tmp = f"{path}.tmp-{ts}"
    try:
        with open(backup, "w", encoding="utf-8") as f:
            f.write(raw)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(good) + ("\n" if good else ""))
        # verify the rewrite is fully parseable before replacing
        with open(tmp, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    json.loads(s)
        os.replace(tmp, path)
        log.warning(
            f"  REDEPLOY GUARD: {base} had {bad} corrupt line(s) — rotated to "
            f"{os.path.basename(backup)}, file rewritten clean ({len(good)} records)"
        )
        return len(good), bad, True
    except Exception as e:
        log.error(
            f"  REDEPLOY GUARD: repair of {base} failed ({e}); original left "
            f"untouched — loaders will skip the bad line(s)"
        )
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return len(good), bad, False


def redeploy_data_check(base_dir: str | None = None) -> dict[str, int]:
    """v10.62: Startup data-integrity check over all volume files.

    base_dir=None uses the module-level paths (production). A custom dir is
    used by smoke tests. Fills the _redeploy_file_counts global used by the
    tracking-check Telegram message. Returns the counts dict."""
    global _redeploy_file_counts
    _base = base_dir or _VOLUME_DIR
    counts: dict[str, int] = {}
    parts: list[str] = []
    for name in _REDEPLOY_JSONL_FILES:
        path = os.path.join(_base, name)
        ok, bad, _rep = _redeploy_check_jsonl(path)
        counts[name] = ok
        if ok == -1:
            parts.append(f"{name} absent")
        else:
            parts.append(f"{name} {ok}")
            if bad:
                parts[-1] += f" (+{bad} bad)"
    for name in _REDEPLOY_JSON_FILES:
        path = os.path.join(_base, name)
        if not os.path.exists(path):
            counts[name] = -1
            parts.append(f"{name} absent")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            counts[name] = 1
            parts.append(f"{name} ok")
        except Exception:
            # Rotate the corrupt json out of the way — loaders treat a
            # missing file as a fresh start, a corrupt one as garbage.
            ts = time.strftime("%Y%m%d%H%M%S")
            try:
                os.replace(path, f"{path}.corrupt-{ts}")
                log.warning(
                    f"  REDEPLOY GUARD: {name} corrupt — rotated to "
                    f"{name}.corrupt-{ts}; bot will recreate it fresh"
                )
                counts[name] = 0
                parts.append(f"{name} CORRUPT(rotated)")
            except OSError:
                counts[name] = 0
                parts.append(f"{name} CORRUPT(unreadable)")
    log.info(f"REDEPLOY GUARD [data]: {' | '.join(parts)}")
    if base_dir is None:
        _redeploy_file_counts = counts
    return counts


def redeploy_tracking_check(client: httpx.Client) -> None:
    """v10.62: One-shot post-first-discovery coverage check.

    Classifies every live in-progress fixture (1H/2H/HT):
      MONITORED       — in fast_monitored (stats lane actively polling)
      PICKUP-PENDING  — tracked league, joins on the next stats scan
      PAST-CEILING    — tracked but past the 85' monitoring ceiling
      UNTRACKED       — league not in the followed set (listed by name)
    Then sends ONE Telegram message: data counts + live coverage + warm-up.
    Never raises; failures are logged only."""
    try:
        live = [
            f for f in cached_fixtures
            if f.get("fixture", {}).get("status", {}).get("short", "") in ("1H", "2H", "HT")
        ]
        monitored_n = pickup_n = past_n = untracked_n = 0
        untracked_names: list[str] = []
        for f in live:
            fid = f["fixture"]["id"]
            if not is_tracked_match(f):
                untracked_n += 1
                if len(untracked_names) < 4:
                    untracked_names.append(
                        f"{f['league'].get('name', '?')} ({f['teams']['home']['name']}-"
                        f"{f['teams']['away']['name']})"
                    )
                continue
            minute = safe_int(str(f["fixture"].get("status", {}).get("elapsed", 0) or 0))
            if minute > EXTENDED_MAX:
                past_n += 1
            elif fid in fast_monitored:
                monitored_n += 1
            else:
                pickup_n += 1

        _fc = _redeploy_file_counts
        _sig_n = _fc.get("signal_outcomes.jsonl", 0)
        _poll_n = _fc.get("pressure_polls.jsonl", 0)
        # v10.63: count pending from the FILE, not memory — the startup EOD
        # race (now fixed) used to clear in-memory outcomes before this
        # message went out, reporting "0 pending" while the disk held 8.
        _pend_n = 0
        try:
            if os.path.exists(OUTCOMES_FILE):
                with open(OUTCOMES_FILE, "r") as _pf:
                    for _ln in _pf:
                        _ln = _ln.strip()
                        if not _ln:
                            continue
                        try:
                            if not json.loads(_ln).get("resolved"):
                                _pend_n += 1
                        except Exception:
                            _pend_n += 1  # unreadable line = pending (conservative)
        except Exception:
            _pend_n = sum(1 for e in signal_outcomes if not e.get("resolved"))
        _blocked_n = _fc.get("blocked_outcomes.jsonl", 0)
        _shadow_n = _fc.get("fastlane_shadow.jsonl", 0)
        _box_n = _fc.get("boxburst_shadow.jsonl", 0)
        _gb_n = _fc.get("goalburst_shadow.jsonl", 0)  # v10.76
        _warm_n = len(_coldstart_warmed)

        data_line = (
            f"Data: {_sig_n} signals ({_pend_n} pending) | {_poll_n} polls | "
            f"{_blocked_n} blocked | {_shadow_n} fl-shadow | {_box_n} box-shadow | "
            f"{_gb_n} gb-shadow — files verified"
        )
        if live:
            live_line = (
                f"Live: {len(live)} games — {monitored_n} monitored, "
                f"{pickup_n} pickup pending, {past_n} past 85', {untracked_n} untracked"
            )
            if untracked_names:
                live_line += " [" + "; ".join(untracked_names) + "]"
        else:
            live_line = "Live: 0 games right now — next kickoff picked up automatically"
        warm_line = (
            f"Warm-up: {_warm_n} fixture(s) backfilled"
            + ("" if _warm_n else " (queued for first stats cycle)")
        )
        msg = (
            f"\U0001f6e1 v10.62 REDEPLOY CHECK\n"
            f"{data_line}\n{live_line}\n{warm_line}\n"
            f"No data lost."
        )
        log.info(
            f"REDEPLOY GUARD [tracking]: monitored={monitored_n} "
            f"pickup={pickup_n} past85={past_n} untracked={untracked_n} "
            f"(live={len(live)})"
        )
        try:
            send_telegram(client, msg)
        except Exception as e:
            log.warning(f"  REDEPLOY GUARD: summary message failed: {e}")
    except Exception as e:
        log.warning(f"  REDEPLOY GUARD: tracking check failed: {e}")


# ============================================================
# v10.71: FEED-GUARD — silent API feed-death detection & hold
# ============================================================

def _v10_71_no_future_kickoff() -> bool:
    """v10.71: True when every scheduled tracked kickoff today has already
    started (or the schedule is unknown). Gates the EOD block so mid-day
    monitoring gaps — block transitions, feed-guard holds — never fire the
    EOD report (once-per-day marker) or clear the outcome memory.
    """
    ts = todays_kickoff_latest_ts
    if ts is None:
        return True  # schedule unknown -> permissive (legacy behavior)
    return time.time() > ts + 45 * 60  # last kickoff started >= 45 min ago


def _v10_72_live_monitored_fids(now: float) -> list[int]:
    """v10.72: monitored fixtures that are LIVE and were polled recently.

    The midnight guard's hold set: a fixture counts only when (a) it is
    still in fast_monitored, (b) its last stats poll is fresh
    (< MIDNIGHT_GUARD_POLL_FRESH — proves ACTIVE watching, never a stale
    pre-sleep cache), and (c) its cached object still reports a live
    status. The freshness window is what keeps the guard safe outside the
    rollover case: a stale yesterday-object can never trick the bot into
    an all-night vigil, and a STOP-budget lapse (no polls -> stale) ends
    the hold automatically.
    """
    out: list[int] = []
    for fid in fast_monitored:
        if now - last_stats_check.get(fid, 0.0) > MIDNIGHT_GUARD_POLL_FRESH:
            continue
        cf = find_cached_fixture(fid)
        if cf and cf["fixture"]["status"]["short"] in LIVE_STATUSES:
            out.append(fid)
    return out


def _v10_74_pending_live_fids(now: float) -> list[int]:
    """v10.74: BOOT-PATH midnight guard — fixtures with PENDING outcomes
    that the resolver RECENTLY verified as still LIVE.

    Why this exists: the v10.72 guard requires fixtures in fast_monitored
    with a fresh stats poll — true while the bot is RUNNING across
    midnight, but IMPOSSIBLE at a fresh boot (fast_monitored is empty
    before the first discovery). The Sep 8 00:01 v10.73 deploy then slept
    838 minutes while the last 22:30-kickoff game was still live at ~90'
    with 1 pending outcome — the startup code even logged "periodic
    resolver will finish them" and then entered a sleep that suspends the
    periodic resolver (the GIL Vicente class, boot variant).

    Fix: the startup resolver already knows the answer ("pending outcome(s)
    all on LIVE fixtures") — feed that knowledge into the same hold. While
    held, the loop keeps running, the periodic resolver re-verifies every
    10m and resolves the outcome within ~10m of FT, then the hold
    releases. Freshness (PENDING_GUARD_FRESH) ends the hold if the
    resolver stops confirming (feed gap / STOP lapse) — a stale live-set
    can never fake an all-night vigil. The v10.72 2h cap applies unchanged.
    """
    if now - _resolve_live_ts > PENDING_GUARD_FRESH:
        return []
    pend = {
        e.get("fixture_id")
        for e in signal_outcomes
        if not e.get("resolved")
    }
    return [f for f in _resolve_live_fids if f in pend]


def _v10_72_fixture_label(fid: int) -> str:
    """v10.72: human-readable label for guard logs/Telegram messages."""
    cf = find_cached_fixture(fid)
    if not cf:
        return f"F{fid}"
    st = cf["fixture"]["status"]
    return (
        f"{cf['teams']['home']['name']} {cf['goals']['home']}-{cf['goals']['away']} "
        f"{cf['teams']['away']['name']} ({st['short']} {st.get('elapsed', '?')})"
    )


def _v10_71_feed_guard_close(reason: str) -> None:
    """v10.71: close an active feed-guard incident (log only — the caller
    sends the Telegram message when appropriate)."""
    global feed_guard
    if feed_guard["active"]:
        _dur = int((time.time() - feed_guard["since"]) // 60)
        log.info(
            f"v10.71 FEED-GUARD: incident closed after {_dur}m — {reason}"
        )
    feed_guard.update({
        "active": False, "since": 0.0, "last_alarm": 0.0, "last_cycle": 0.0,
        "held_fids": set(), "alarms_sent": 0, "gave_up": False,
    })


def _v10_71_feed_guard(client: httpx.Client, pre_feed_by_id: dict[int, dict]) -> None:
    """v10.71: run ONE feed-death verification pass inside do_discovery.

    Called AFTER the live-feed response replaces cached_fixtures but BEFORE
    the outcome checks, cleanup_state and the monitored-set prune — so a
    verified-still-live fixture can be re-injected into cached_fixtures and
    everything downstream (state cleanup, prune, heartbeat, EOD gates)
    treats it as live. Detection signature: ALL monitored fixtures vanish
    from the live feed in ONE cycle while at least one was last seen in
    live play below the 85' ceiling (natural block ends stagger — games
    finish one by one at 85'+; a simultaneous all-vanish of sub-85' games
    is an API failure, not football).

    Cost: one batched /fixtures?ids= call (1 credit) per triggered cycle.
    Holds are capped at FEED_GUARD_HOLD_MAX; alarms re-sent at
    FEED_GUARD_ALARM_EVERY.
    """
    global feed_guard
    now = time.time()
    monitored = set(fast_monitored)
    if not monitored:
        if feed_guard["active"]:
            _v10_71_feed_guard_close("no monitored fixtures remain")
        return

    new_live_ids = {f["fixture"]["id"] for f in cached_fixtures}
    vanished = [fid for fid in monitored if fid not in new_live_ids]
    if not vanished:
        # Feed healthy for every monitored fixture (or recovered).
        if feed_guard["active"]:
            _dur = int((now - feed_guard["since"]) // 60)
            _v10_71_feed_guard_close(
                f"live feed recovered after {_dur}m — monitoring resumes seamlessly"
            )
            try:
                send_telegram(
                    client,
                    f"✅ v10.71 FEED-GUARD: live feed RECOVERED after {_dur}m. "
                    f"Monitoring resumed automatically — no restart needed.",
                )
            except Exception as e:
                log.warning(f"v10.71 FEED-GUARD: recovery message failed: {e}")
        return

    all_vanished = len(vanished) == len(monitored)
    if not all_vanished:
        # Partial drop: routine API glitch / HT transition (v10.44s class).
        # The normal prune + next discovery re-add handles it — do nothing.
        return

    # Suspicious only if at least one vanished fixture was last seen in
    # live play below the tracking ceiling (natural ends stagger 85'+).
    def _last_live_young(fid: int) -> bool:
        f = pre_feed_by_id.get(fid)
        if not f:
            return False
        st = f["fixture"]["status"]["short"]
        mn = safe_int(str(f["fixture"]["status"].get("elapsed", 0) or 0))
        return st in LIVE_STATUSES and mn < FEED_GUARD_MIN

    if not any(_last_live_young(fid) for fid in vanished):
        # Every vanished fixture was 85'+ or already non-live: natural end.
        if feed_guard["active"]:
            _v10_71_feed_guard_close("all vanished fixtures were 85'+ — natural end")
        return

    # --- Suspicious all-vanish: verify directly by fixture ID ---
    if feed_guard["active"] and now - feed_guard["since"] > FEED_GUARD_HOLD_MAX:
        # Hard cap: stop holding (state will be pruned naturally this cycle).
        _held_n = len(feed_guard["held_fids"])
        _v10_71_feed_guard_close(
            f"hold cap reached ({FEED_GUARD_HOLD_MAX // 60}m) — releasing {_held_n} fixture(s)"
        )
        try:
            send_telegram(
                client,
                f"⚠️ v10.71 FEED-GUARD: gave up holding after {FEED_GUARD_HOLD_MAX // 60}m "
                f"(feed still not recovered). Fixtures released — pending outcomes stay "
                f"in memory and resolve when the API catches up. Manual check recommended.",
            )
        except Exception as e:
            log.warning(f"v10.71 FEED-GUARD: give-up message failed: {e}")
        return

    vmap: dict[int, dict] = {}
    try:
        ids_param = "-".join(str(fid) for fid in vanished[:BATCH_SIZE_LIMIT])
        vdata = api_get(client, "/fixtures", {"ids": ids_param})
        vmap = {f["fixture"]["id"]: f for f in vdata.get("response", [])}
    except Exception as e:
        log.warning(
            f"v10.71 FEED-GUARD: verification call failed ({e}) — treating as feed death"
        )
        vmap = {}

    held: list[dict] = []
    verified_ft: list[tuple[int, str]] = []
    missing = 0
    for fid in vanished:
        vf = vmap.get(fid)
        if vf is None:
            # Not in the ids= response either: keep monitoring on the
            # last-known object (a finished match always resolves by ID).
            pf = pre_feed_by_id.get(fid)
            if pf is not None:
                held.append(pf)
                missing += 1
        elif vf["fixture"]["status"]["short"] in LIVE_STATUSES:
            held.append(vf)  # fresh, verified still-in-progress object
        else:
            verified_ft.append((fid, str(vf["fixture"]["status"]["short"])))
            # Keep the finished object in the cache so this cycle's outcome
            # checks + prune see the true FT status.
            cached_fixtures.append(vf)
            # v10.72: the FastWin countdown is a live-polling concept —
            # kill it the moment the guard itself proves the fixture FT
            # (GIL Vicente zombie: frozen "FastWin 0s" entry showed in
            # every heartbeat for 17h because nothing polled the fixture
            # again to trigger the natural expiry path).
            expire_fast_sot(fid)

    if held:
        cached_fixtures.extend(held)
        held_fids = {f["fixture"]["id"] for f in held}
        if not feed_guard["active"]:
            feed_guard["active"] = True
            feed_guard["since"] = now
            feed_guard["last_alarm"] = 0.0  # force first alarm below
            feed_guard["alarms_sent"] = 0
        feed_guard["held_fids"] = held_fids
        feed_guard["last_cycle"] = now

        if now - feed_guard["last_alarm"] >= FEED_GUARD_ALARM_EVERY:
            feed_guard["last_alarm"] = now
            feed_guard["alarms_sent"] += 1
            _names = "; ".join(
                f"{f['teams']['home']['name']} {f['goals']['home']}-{f['goals']['away']} "
                f"{f['teams']['away']['name']} ({f['fixture']['status']['short']} "
                f"{f['fixture']['status'].get('elapsed', '?')}')"
                for f in held[:7]
            )
            _mins = int((now - feed_guard["since"]) // 60)
            _frozen = (
                f"quota counter frozen at {quota_freeze['value']} across "
                f"{quota_freeze['count']} calls"
                if quota_freeze["count"] >= FEED_GUARD_FROZEN_CALLS
                else f"freeze-watch {quota_freeze['count']}/{FEED_GUARD_FROZEN_CALLS} identical"
            )
            log.info(
                f"v10.71 FEED-GUARD: FEED DEATH — live feed empty but {len(held)} tracked "
                f"match(es) verified in progress ({missing} missing from ids= response); "
                f"{len(verified_ft)} verified finished; incident {_mins}m"
            )
            try:
                send_telegram(
                    client,
                    f"🚨 v10.71 FEED-GUARD: the live feed returned NO games, but "
                    f"{len(held)} tracked match(es) are STILL IN PROGRESS (verified by "
                    f"fixture-ID check).\nHeld: {_names}\nIncident: {_mins}m | "
                    f"re-verifying every discovery cycle | {_frozen}\n"
                    f"Monitoring state PRESERVED — auto-resume when the feed recovers. "
                    f"Check api-sports.io; restart only if this persists past "
                    f"{FEED_GUARD_HOLD_MAX // 60}m.",
                )
            except Exception as e:
                log.warning(f"v10.71 FEED-GUARD: alarm message failed: {e}")
        else:
            log.info(
                f"v10.71 FEED-GUARD: holding {len(held)} fixture(s) "
                f"(incident {int((now - feed_guard['since']) // 60)}m, alarm throttled)"
            )
    else:
        # Everything verified genuinely finished — a staggered natural end
        # that looked suspicious from the live feed alone.
        if feed_guard["active"]:
            _v10_71_feed_guard_close(
                f"all {len(verified_ft)} vanished fixture(s) verified finished"
            )
        else:
            log.info(
                f"v10.71 FEED-GUARD: all {len(verified_ft)} vanished monitored "
                f"fixture(s) verified FINISHED by ID — natural end-of-block"
            )


# ============================================================
# DISCOVERY — fetch live fixtures, update candidates & monitored set
# ============================================================

def do_discovery(client: httpx.Client) -> bool:
    """Run one discovery cycle. Updates global state.
    Returns True if discovery succeeded.
    """
    global last_discovery_time, cached_fixtures, fast_monitored
    global goal_priority_until, _goal_detect_ts, _goal_game_minute

    # v10.44r: Snapshot old scores BEFORE replacing cache, so we can
    # detect score changes from discovery (which runs independently of stats).
    _old_scores: dict[int, tuple[int, int]] = {}
    for f in cached_fixtures:
        fid = f["fixture"]["id"]
        _old_scores[fid] = (f["goals"]["home"] or 0, f["goals"]["away"] or 0)

    # v10.71: snapshot the pre-discovery fixture objects by ID — the
    # feed-guard needs each monitored fixture's last-known status/minute
    # to judge an all-vanish (feed death) vs a staggered natural end.
    _pre_feed_by_id: dict[int, dict] = {
        f["fixture"]["id"]: f for f in cached_fixtures
    }

    data = api_get(client, "/fixtures", {"live": "all"})
    cached_fixtures = data.get("response", [])
    last_discovery_time = time.time()
    _now = time.time()

    # v10.71: FEED-GUARD — verify suspicious all-vanish patterns and hold
    # verified-still-live fixtures BEFORE any state cleanup / prune runs.
    # Re-injects held fixtures into cached_fixtures so the rest of this
    # cycle (outcome checks, cleanup_state, prune) treats them as live.
    try:
        _v10_71_feed_guard(client, _pre_feed_by_id)
    except Exception as _fg_e:
        log.warning(f"v10.71 FEED-GUARD: pass failed: {_fg_e}")

    # v10.44r: Detect score changes from discovery and trigger goal priority.
    # Discovery runs on /fixtures?live=all which updates scores within ~15s.
    # Without this, the bot knows the score changed but waits for the normal
    # stats polling cycle to act on it — wasting precious seconds.
    for f in cached_fixtures:
        fid = f["fixture"]["id"]
        _new_h = f["goals"]["home"] or 0
        _new_a = f["goals"]["away"] or 0
        if fid in _old_scores:
            _old_h, _old_a = _old_scores[fid]
            if _new_h != _old_h or _new_a != _old_a:
                # Score changed! Check if this is a tracked + monitored fixture
                if fid in fast_monitored or is_tracked_match(f):
                    _fname = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"
                    _changes = []
                    if _new_h != _old_h:
                        _changes.append(f"Home {_old_h}->{_new_h}")
                    if _new_a != _old_a:
                        _changes.append(f"Away {_old_a}->{_new_a}")
                    log.info(
                        f"  DISCOVERY GOAL: F{fid} {_fname} — {', '.join(_changes)} "
                        f"-> goal priority for {GOAL_PRIORITY_WINDOW}s"
                    )
                    # Elevate to 15s polling for GOAL_PRIORITY_WINDOW seconds
                    goal_priority_until[fid] = _now + GOAL_PRIORITY_WINDOW
                    # Record detection latency baseline
                    if fid not in _goal_detect_ts:
                        _goal_detect_ts[fid] = _now
                        _game_min = safe_int(str(f["fixture"].get("elapsed", 0) or 0))
                        _goal_game_minute[fid] = _game_min

    # v9.8: Check signal outcomes for all tracked fixtures in discovery
    # This catches goals from fixtures we stopped monitoring (first-signal-only)
    # and fixtures that ended between our last stats check.
    for f in cached_fixtures:
        if is_tracked_match(f):
            check_signal_outcomes(f, client)
            check_blocked_outcomes(f, client)  # v10.49: false-negative resolution
            check_fastlane_shadow(f, client)  # v10.50: fast-lane shadow resolution

    tracked = [f for f in cached_fixtures if is_tracked_match(f)]
    budget = get_budget_mode()

    # v10.71: surface feed-guard holds in the discovery line
    _fg_note = ""
    if feed_guard["active"]:
        _held_n = len(feed_guard["held_fids"])
        _fg_mins = int((time.time() - feed_guard["since"]) // 60)
        _fg_note = f" (FEED-GUARD: holding {_held_n}, incident {_fg_mins}m)"

    log.info(
        f"Discovery: Quota: {quota_remaining}/{quota_limit} | Mode: {budget} | "
        f"Live: {len(cached_fixtures)}{_fg_note} | Tracked: {len(tracked)} | Reqs: {request_count}"
    )

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(
                f"  -> {m['league']['name']}: "
                f"{m['teams']['home']['name']} vs {m['teams']['away']['name']} "
                f"({m['fixture']['status']['short']} {minute}')"
            )

    # v10.44l: Log UNTRACKED live fixtures — catches league ID changes
    # (e.g. API returns different ID for new season of a tracked league).
    _untracked_live = [f for f in cached_fixtures if not is_tracked_match(f)
                       and f["fixture"]["status"]["short"] in ("1H", "2H", "HT")]
    global _last_untracked_live_count, _untracked_retry_count
    _last_untracked_live_count = len(_untracked_live)
    if _last_untracked_live_count > 0:
        _untracked_retry_count += 1
    else:
        _untracked_retry_count = 0
    if _untracked_live:
        _untracked_leagues = {}
        for _uf in _untracked_live:
            _ulid = _uf["league"]["id"]
            if _ulid not in _untracked_leagues:
                _untracked_leagues[_ulid] = _uf["league"].get("name", "?")
        _ut_str = ", ".join(f"{lid}:{name}" for lid, name in sorted(_untracked_leagues.items()))
        log.info(f"  UNTRACKED live: {len(_untracked_live)} fixture(s) in {len(_untracked_leagues)} league(s): {_ut_str}")
        for _uf in _untracked_live:
            _umin = _uf["fixture"]["status"].get("elapsed", "?")
            log.info(f"    UNTRACKED: {_uf['teams']['home']['name']} vs {_uf['teams']['away']['name']} | league {_uf['league']['id']}:{_uf['league'].get('name','?')} | {_uf['fixture']['status']['short']} {_umin}'")

    # v10.44b: Log STATUS-DEAD fixtures — API reports BT/NS but other matches
    # are live, suggesting a status data failure. These are tracked but never
    # reach the signal engine. Logged for experiment coverage analysis.
    NOT_STARTED = {"BT", "NS"}
    has_live_play = any(
        m["fixture"]["status"]["short"] not in NOT_STARTED
        for m in tracked
    )
    if has_live_play:
        for m in tracked:
            st = m["fixture"]["status"]["short"]
            if st in NOT_STARTED:
                log.info(
                    f"  -> STATUS-DEAD: {m['teams']['home']['name']} vs "
                    f"{m['teams']['away']['name']} | API status={st} | "
                    f"API minute={m['fixture']['status'].get('elapsed', 0) or 0}"
                )

    # Cleanup ended fixtures from all state
    live_ids = {f["fixture"]["id"] for f in cached_fixtures}
    cleanup_state(live_ids)

    # v9.7: Check dead fixtures for score changes (momentum shift revival)
    # v10.43: Also revive every 5min if SOT>=3 or big acceleration
    revived = 0
    now = time.time()
    for fid, (dh, da, death_ts) in list(dead_fixtures.items()):
        f = find_cached_fixture(fid)
        if not f:
            dead_fixtures.pop(fid, None)
            continue
        # v10.10: Skip revival if match is past our window
        # v10.13.2: Allow revival if SOT-accelerating
        fixture_minute = safe_int(str(f["fixture"]["status"].get("elapsed", 0) or 0))
        best_sot = get_fixture_best_sot(fid)
        sot_accel = fid in accelerating_fixtures or fid in pressure_accelerating
        is_extended = sot_accel or best_sot >= 2
        # v10.20: Always use extended max for revival check
        # v10.31: Event-extended fixtures get 90' ceiling
        hard_max = EXTENDED_MAX
        if fid in event_extended_fixtures:
            hard_max = 90
        if fixture_minute > hard_max:
            dead_fixtures.pop(fid, None)
            continue
        ch = f["goals"]["home"] or 0
        ca = f["goals"]["away"] or 0
        revive_reason = None

        # v10.43: Score-change revival (original)
        if ch != dh or ca != da:
            revive_reason = f"score {dh}-{da} -> {ch}-{ca} (momentum shift)"
        # v10.43: Time-based revival every 5min — check SOT from fresh discovery data
        elif now - death_ts >= 300:
            revive_reason = _check_dead_revival_stats(fid, f)
            if revive_reason:
                # Reset timer so next check is 5min from now
                dead_fixtures[fid] = (dh, da, now)

        if revive_reason:
            dead_fixtures.pop(fid, None)
            revived += 1
            log.info(
                f"  REVIVED: fixture {fid} "
                f"({f['teams']['home']['name']} vs {f['teams']['away']['name']}) "
                f"{revive_reason}, re-monitoring"
            )
    if revived:
        log.info(f"  -> {revived} dead fixture(s) revived")

    # v10.44b: Data-dead revival — every 5min, check if API now has stats
    # Uses the discovery-cached fixture data (stats embedded in /fixtures?ids=
    # response, or from _check_dead_revival_stats which reads the same).
    data_revived = 0
    now_dd = time.time()
    for fid, death_ts in list(data_dead_fixtures.items()):
        f = find_cached_fixture(fid)
        if not f:
            data_dead_fixtures.pop(fid, None)
            continue
        fixture_minute = safe_int(str(f["fixture"]["status"].get("elapsed", 0) or 0))
        if fixture_minute > EXTENDED_MAX:
            data_dead_fixtures.pop(fid, None)
            continue
        # Check every 5 minutes
        if now_dd - death_ts >= 300:
            # Check if the discovery-cached data now has stats
            has_stats = False
            stats_list = f.get("statistics", [])
            if stats_list:
                for s in stats_list:
                    team_data = s
                    for stat in team_data.get("statistics", []):
                        if (stat.get("type", "") in ("Shots on Goal", "Total Shots")
                                and safe_int(stat.get("value", "0")) > 0):
                            has_stats = True
                            break
                    if has_stats:
                        break
            if has_stats:
                data_dead_fixtures.pop(fid, None)
                data_revived += 1
                log.info(
                    f"  DATA-DEAD REVIVED: fixture {fid} "
                    f"({f['teams']['home']['name']} vs {f['teams']['away']['name']}) "
                    f"stats appeared after {int(now_dd - death_ts)}s, re-monitoring"
                )
            else:
                # Reset timer so next check is 5min from now
                data_dead_fixtures[fid] = now_dd
    if data_revived:
        log.info(f"  -> {data_revived} data-dead fixture(s) revived")

    # Local pre-filter
    candidates = find_candidates(cached_fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN - PRE_WINDOW_MINUTES}-{EXTENDED_MAX}', signal from {MINUTE_MIN}+)")

    # Build fixture -> best rank mapping
    max_fast = get_max_fast_monitored()
    fixture_best_rank: dict[int, int] = {}
    for fid, tid, rank in candidates:
        if fid not in fixture_best_rank or rank > fixture_best_rank[fid]:
            fixture_best_rank[fid] = rank

    # v9.7: Exclude dead fixtures (SOT=0 for both teams) from monitoring.
    # They waste credits polling matches with zero attacking pressure.
    if dead_fixtures:
        before = len(fixture_best_rank)
        dead_skipped_fids = [fid for fid in fixture_best_rank if fid in dead_fixtures]
        fixture_best_rank = {
            fid: rank for fid, rank in fixture_best_rank.items()
            if fid not in dead_fixtures
        }
        skipped = before - len(fixture_best_rank)
        if skipped:
            log.info(f"  -> Skipped {skipped} dead fixture(s) (both teams SOT=0)")
            for fid in dead_skipped_fids:
                f = find_cached_fixture(fid)
                if f:
                    log.info(f"     DEAD: {f['teams']['home']['name']} vs {f['teams']['away']['name']}")

    # v10.44b: Exclude data-dead fixtures (API has no stats coverage)
    if data_dead_fixtures:
        before = len(fixture_best_rank)
        dd_skipped_fids = [fid for fid in fixture_best_rank if fid in data_dead_fixtures]
        fixture_best_rank = {
            fid: rank for fid, rank in fixture_best_rank.items()
            if fid not in data_dead_fixtures
        }
        skipped = before - len(fixture_best_rank)
        if skipped:
            log.info(f"  -> Skipped {skipped} data-dead fixture(s) (no API stats coverage)")
            for fid in dd_skipped_fids:
                f = find_cached_fixture(fid)
                if f:
                    log.info(f"     DATA-DEAD: {f['teams']['home']['name']} vs {f['teams']['away']['name']}")

    # v9.6.2: In FIRST_SIGNAL_ONLY mode, exclude already-signaled
    # fixtures from candidates. Old bug: signaled fixtures kept being
    # re-added ("0 retained, 1 new") then immediately removed by
    # is_fixture_monitorable(), creating an infinite discover→add→
    # stats→remove loop that burned 2 credits every 10 seconds.
    # v10.44c: Score-change revival — is_fixture_done_first_signal() now
    # returns False if the score changed since the signal, so signaled
    # fixtures with goal events are kept as candidates.
    if is_first_signal_only_mode() and signaled_fixtures:
        before = len(fixture_best_rank)
        sig_skipped_fids = []
        sig_partial_fids = []  # 1/2 teams signaled, other eligible
        sig_revived_fids = []  # 2/2 teams but score changed (v10.44c revive)
        _new_rank = {}
        for fid, rank in fixture_best_rank.items():
            if not is_fixture_done_first_signal(fid):
                _new_rank[fid] = rank
                if fid in signaled_fixtures:
                    _stc = sum(1 for (sf, _) in signaled_teams if sf == fid)
                    if _stc < 2:
                        sig_partial_fids.append(fid)
                    else:
                        sig_revived_fids.append(fid)
                continue
            sig_skipped_fids.append(fid)
        fixture_best_rank = _new_rank
        skipped = len(sig_skipped_fids)
        if skipped:
            log.info(f"  -> Skipped {skipped} fully-signaled fixture(s)")
            for fid in sig_skipped_fids:
                f = find_cached_fixture(fid)
                if f:
                    log.info(f"     SIG-DONE (2/2 teams): {f['teams']['home']['name']} vs {f['teams']['away']['name']}")
        if sig_partial_fids:
            log.info(f"  -> Kept {len(sig_partial_fids)} partial-signal fixture(s) (other team still eligible)")
            for fid in sig_partial_fids:
                f = find_cached_fixture(fid)
                if f:
                    log.info(f"     SIG-PARTIAL (1/2 teams): {f['teams']['home']['name']} vs {f['teams']['away']['name']}")
        if sig_revived_fids:
            log.info(f"  -> Revived {len(sig_revived_fids)} fixture(s) (score changed since signal)")
            for fid in sig_revived_fids:
                f = find_cached_fixture(fid)
                if f:
                    log.info(f"     SIG-REVIVE (2/2, score changed): {f['teams']['home']['name']} vs {f['teams']['away']['name']}")

    # --- Preserve existing monitored fixtures that are still valid ---
    # v10.51: log WHY each monitored fixture is dropped (past-85' ceiling,
    # fully signaled, ended, gone from live feed). Before, fixtures just
    # silently vanished from "Monitored:" — the classic "why isn't the
    # bot tracking game X" mystery (seen live with Slavia Sofia vs Levski
    # Sofia, 2H 89' > 85' ceiling: dropped correctly but invisibly).
    retained = set()
    dropped_notes: list[str] = []
    for fid in list(fast_monitored):
        fixture = find_cached_fixture(fid)
        if fixture and is_fixture_monitorable(fixture):
            retained.add(fid)
            continue
        if not fixture:
            dropped_notes.append(f"F{fid} (no longer in live feed)")
            continue
        # v10.67: a fixture dropped here while STILL LIVE and 86-90' (any
        # prune reason: ceiling / 2-2-signaled / out-of-window) keeps its
        # events-watch eligibility to the final whistle — the helper's own
        # gates make this a no-op for every other drop.
        _retain_late_fixture(fid, fixture, "discovery prune")
        _names = (f"{fixture['teams']['home']['name']} vs "
                  f"{fixture['teams']['away']['name']}")
        _st = fixture["fixture"]["status"]["short"]
        _min = safe_int(str(fixture["fixture"]["status"].get("elapsed", 0) or 0))
        if _st not in LIVE_STATUSES:
            dropped_notes.append(f"{_names} (status {_st})")
        elif is_fixture_done_first_signal(fid):
            dropped_notes.append(f"{_names} (2/2 teams signaled, {_st} {_min}')")
        else:
            _ceiling = 90 if fid in event_extended_fixtures else EXTENDED_MAX
            if _min > _ceiling:
                dropped_notes.append(f"{_names} ({_min}' > {_ceiling}' tracking ceiling)")
            else:
                dropped_notes.append(f"{_names} (minute {_min}' out of window)")
    if dropped_notes:
        log.info(f"  -> Dropped from monitoring: {'; '.join(dropped_notes)}")

    # Merge: keep existing + add new candidates
    merged = retained | set(fixture_best_rank.keys())

    # If over limit, rank all and keep top N
    if len(merged) > max_fast:
        all_ranked = []
        for fid in merged:
            rank = max(
                fixture_best_rank.get(fid, 0),
                fast_priority.get(fid, 0),
            )
            all_ranked.append((fid, rank))
        all_ranked.sort(key=lambda x: x[1], reverse=True)
        merged = set(fid for fid, _ in all_ranked[:max_fast])

    # Update global state
    fast_monitored.clear()
    fast_monitored |= merged
    for fid in merged:
        if fid in fixture_best_rank:
            fast_priority[fid] = max(
                fast_priority.get(fid, 0), fixture_best_rank[fid]
            )

    log.info(
        f"  -> Monitored: {len(fast_monitored)} fixture(s) "
        f"({len(retained)} retained, {len(merged) - len(retained)} new)"
        f" | Signaled: {len(signaled_fixtures)} | Max: {max_fast}"
    )
    return True


# ============================================================
# BATCHED STATS — one API call for all monitored fixtures
# ============================================================

def parse_xg(tstats: dict) -> str:
    """Extract xG from team statistics. Returns string or 'N/A'.

    v10.10: Uses STAT_ALIASES for lookup (handles "expected_goals" key).
    """
    val = get_stat(tstats, "expected_goals", default=None)
    if val is not None and val != "0" and val != "N/A":
        try:
            return str(float(val))
        except (ValueError, TypeError):
            pass
    return "N/A"


def get_red_card_string(teams_data: dict, home_name: str, away_name: str) -> str:
    """Build red card string for the signal message.

    v10.1: Uses normalized alias lookup.
    """
    parts = []
    for tname, tstats in teams_data.items():
        rc = safe_int(get_stat(tstats, "red_cards"))
        if rc > 0:
            parts.append(f"{tname} - {rc}")

    if parts:
        return " | ".join(parts)
    return "None"


def save_outcome(entry: dict) -> None:
    """Append a signal outcome to the JSONL file."""
    try:
        with open(OUTCOMES_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        log.info(f"  Saved outcome to file: {entry.get('team_name')} F{entry.get('fixture_id')} (pending)")
    except Exception as e:
        log.warning(f"  Failed to save outcome to file: {e}")


# v10.44d-fix: Record non-signal fixtures for ML negative examples
def record_non_signal_fixture(fixture: dict) -> None:
    """Record a monitored fixture that never triggered a signal (negative example for ML)."""
    fid = fixture["fixture"]["id"]
    if fid in signaled_fixtures:
        return
    sh = fixture["goals"]["home"] or 0
    sa = fixture["goals"]["away"] or 0
    _home_name = fixture["teams"]["home"]["name"]
    _away_name = fixture["teams"]["away"]["name"]
    _raw_league = fixture["league"].get("name", "?")
    _league = _fix_league_name(_raw_league, _home_name, _away_name)
    for is_h, tinfo in [(True, fixture["teams"]["home"]), (False, fixture["teams"]["away"])]:
        entry = {
            "fixture_id": fid, "team_id": tinfo["id"], "team_name": tinfo["name"],
            "league": _league,
            "signal_time": None, "signal_clock": None,
            "game_minute": fixture["fixture"]["status"].get("elapsed", 90) or 90,
            "sot": 0, "stats_sot_raw": 0, "events_sot": 0,
            "total_shots": 0, "shots_inside_box": 0, "ib_ratio": 0,
            "shots_off_target": 0, "xg": None, "big_chances": 0,
            "corners": 0, "possession": 0, "gps": 0, "gps_restored": 0,  # v10.61
            "gps_sot": 0, "gps_ib": 0, "gps_sv": 0,
            "gps_xg": 0, "gps_bc": 0, "gps_corners": 0, "gps_poss": 0, "gps_accel": 0,
            "sustained": 0, "accel_count": 0,
            "tier": "NON-SIGNAL", "window_tag": "NON-SIGNAL",
            "goals_at_signal": sh if is_h else sa,
            "opponent_goals_at_signal": sa if is_h else sh,
            "is_home": is_h,
            "scoreline": "winning" if (sh if is_h else sa) > (sa if is_h else sh) else "drawing" if (sh if is_h else sa) == (sa if is_h else sh) else "losing",
            "is_losing": (sh if is_h else sa) < (sa if is_h else sh),
            "is_stale_critical": False, "post_goal_minutes_since": None,
            "post_goal_tag": "NON-SIGNAL", "goal_detected_this_poll": False,
            "minutes_remaining": 0, "version": BOT_VERSION,
            "outcome_5min": "N/A", "outcome_10min": "N/A",
            "outcome_15min": "N/A", "outcome_full": "N/A",
            "goal_minute_5": None, "goal_minute_10": None,
            "goal_minute_15": None, "goal_minute_full": None,
            "sig_num": 0, "gps_triggered": False,
            "resolved": True, "resolved_via": "non_signal",
            # v10.44q: Enriched fields (null for non-signals)
            "poll_interval": None, "last_goal_minute": None,
            "time_since_prev_signal": None,
            # v10.44r: Latency fields (null for non-signals)
            "goal_detect_lag": None, "stats_update_lag": None,
            "signal_lag": None, "goal_in_priority_window": None,
        }
        signal_outcomes.append(entry)
        save_outcome(entry)
    log.info(f"  NON-SIGNAL: Recorded fixture {fid} ({tinfo['name']} & opp) for ML")


def rewrite_outcomes_file() -> None:
    """v10.11: Rewrite the JSONL file, merging disk entries with in-memory state.

    v10.44m fix: Previously overwrote with only in-memory data, losing all
    previous days' signals. Now reads existing file, merges with memory
    (memory takes priority for same key), and writes everything back.
    This preserves historical signals across daily EOD rewrites.
    """
    try:
        # 1. Load existing entries from disk
        disk_entries = {}
        if os.path.exists(OUTCOMES_FILE):
            with open(OUTCOMES_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("fixture_id"), obj.get("team_id"), obj.get("signal_time"))
                        if None not in key:
                            disk_entries[key] = obj
                    except Exception:
                        continue

        # 2. Merge: in-memory entries override disk (they have updated resolution status)
        for entry in signal_outcomes:
            key = (entry.get("fixture_id"), entry.get("team_id"), entry.get("signal_time"))
            if None not in key:
                disk_entries[key] = entry

        # 3. Write merged result
        with open(OUTCOMES_FILE, "w") as f:
            for entry in disk_entries.values():
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        log.warning(f"  Failed to rewrite outcomes file: {e}")


# ============================================================
# v10.49: FALSE-NEGATIVE TRACKING + POISSON CALIBRATION
# (LOGGING ONLY — never gates, never sends, never changes thresholds)
# ============================================================
# Blocked candidates = moments where a QUALIFYING signal (tier assigned)
# was suppressed by a policy gate. Currently those moments vanish — false
# negatives are invisible. This records them (deduped: max 1 record per
# (fixture, team, reason) per 10 game-minutes, cap 600/day) and resolves
# them exactly like signal_outcomes, so the EOD report can show which
# gates cost us wins.


def _track_blocked_candidate(
    fid: int, tid: int, tname: str, league: str, minute: int,
    tier: str, reason: str, gps: float, sot: int, ib_ratio: float,
    sh, sa, is_home_team: bool,
    ml_score: float | None = None,  # v10.59: ML shadow opinion at block time
) -> None:
    """v10.49: Record a moment where a qualifying signal was suppressed by a gate.

    LOGGING ONLY — never affects signal decisions. A later HIT resolution
    means the team scored AFTER the block: the gate potentially cost us a
    winning signal (false negative). Gives gate-tuning data instead of
    silently discarding blocked pressure.

    v10.59: records the ML shadow opinion too, so /mlstats can ask
    "would the ML brain have warned on the goals the gates blocked?".
    """
    global _blocked_count_today, _blocked_count_date
    try:
        _today = time.strftime("%Y-%m-%d")
        if _blocked_count_date != _today:
            _blocked_count_date = _today
            _blocked_count_today = 0
        if _blocked_count_today >= 600:
            return

        # Dedupe: one record per (fixture, team, reason) per 10 game-minutes
        _bucket = int(minute) // 10
        _key = (fid, tid, reason)
        if _blocked_dedupe.get(_key) == _bucket:
            return
        _blocked_dedupe[_key] = _bucket
        _blocked_count_today += 1

        _team_goals = (sh if is_home_team else sa) or 0
        _opp_goals = (sa if is_home_team else sh) or 0
        entry = {
            "blocked_time": time.time(),
            "blocked_clock": time.strftime("%Y-%m-%d %H:%M"),
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": league,
            "game_minute": minute,
            "tier": tier,
            "block_reason": reason,
            "sot": sot,
            "gps": round(float(gps), 1),
            "gps_restored": round(bc_restore_gps(gps, None), 1),  # v10.61: shadow (BC dead in prod -> None correct)
            "ml_score": round(ml_score, 1) if ml_score is not None else None,  # v10.59: shadow opinion
            "ib_ratio": round(float(ib_ratio), 3),
            "goals_at_block": _team_goals,
            "opponent_goals_at_block": _opp_goals,
            "is_home": bool(is_home_team),
            "scoreline": (
                "winning" if _team_goals > _opp_goals
                else "drawing" if _team_goals == _opp_goals
                else "losing"
            ),
            "outcome_5min": None,
            "outcome_10min": None,
            "outcome_15min": None,
            "outcome_full": None,
            "goal_minute_5": None,
            "goal_minute_10": None,
            "goal_minute_15": None,
            "goal_minute_full": None,
            "resolved": False,
            "version": BOT_VERSION,
        }
        blocked_outcomes.append(entry)
        with open(BLOCKED_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.debug(f"  v10.49 blocked-track error: {e}")


def _load_blocked_outcomes() -> list[dict]:
    """v10.49: Load blocked-candidate records from JSONL (survives restarts)."""
    entries: list[dict] = []
    if not os.path.exists(BLOCKED_FILE):
        return entries
    try:
        with open(BLOCKED_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"v10.49: Failed to load blocked outcomes file: {e}")
    return entries


def rewrite_blocked_file() -> None:
    """v10.49: Rewrite blocked JSONL after in-place resolution updates.

    Same merge discipline as rewrite_outcomes_file(): disk entries are the
    base, in-memory entries override (they carry updated resolution status).
    """
    try:
        disk_entries: dict[tuple, dict] = {}
        if os.path.exists(BLOCKED_FILE):
            with open(BLOCKED_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("fixture_id"), obj.get("team_id"),
                               obj.get("block_reason"), obj.get("blocked_time"))
                        if None not in key:
                            disk_entries[key] = obj
                    except Exception:
                        continue
        for entry in blocked_outcomes:
            key = (entry.get("fixture_id"), entry.get("team_id"),
                   entry.get("block_reason"), entry.get("blocked_time"))
            if None not in key:
                disk_entries[key] = entry
        tmp = BLOCKED_FILE + ".tmp"
        with open(tmp, "w") as f:
            for entry in disk_entries.values():
                f.write(json.dumps(entry, default=str) + "\n")
        os.replace(tmp, BLOCKED_FILE)
    except Exception as e:
        log.warning(f"v10.49: Failed to rewrite blocked outcomes file: {e}")


def check_blocked_outcomes(fixture: dict, client: httpx.Client = None) -> None:
    """v10.49: Resolve pending blocked-candidate entries for this fixture.

    Mirrors check_signal_outcomes (goal-events resolution when FT, live
    score fallback otherwise). A HIT means the team scored AFTER the block
    minute — logged as FN-COST (false-negative cost) for gate tuning.
    LOGGING ONLY.
    """
    if not blocked_outcomes:
        return
    fid = fixture["fixture"]["id"]
    status = fixture["fixture"]["status"]["short"]
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    is_finished = status not in LIVE_STATUSES

    goal_events = None
    if is_finished and client is not None:
        goal_events = fetch_goal_events(client, fid)

    any_updated = False
    for entry in blocked_outcomes:
        if entry["fixture_id"] != fid or entry.get("resolved"):
            continue

        # v10.14 path: precise goal-event resolution when fixture is finished
        if goal_events is not None:
            if resolve_with_goal_events(
                entry, goal_events, home_id, away_id,
                home_goals=home_goals, away_goals=away_goals,
                entry_kind="blocked",
            ):
                any_updated = True
            if entry.get("outcome_full") == "HIT":
                log.info(
                    f"  FN-COST: [{entry.get('block_reason')}] {entry['team_name']} "
                    f"blocked at {entry['game_minute']}' then scored (FT resolution) "
                    f"GPS={entry.get('gps', '?')} [{entry.get('league', '?')}]"
                )
            continue

        # --- Fallback: live tracking (match still in progress) ---
        current_team_goals = home_goals if entry["is_home"] else away_goals
        goals_since_block = current_team_goals - entry["goals_at_block"]
        mins_since = minute - entry["game_minute"]

        if goals_since_block > 0:
            mins_to_goal = minute - entry["game_minute"]
            for window, outcome_key in [
                (5, "outcome_5min"), (10, "outcome_10min"), (15, "outcome_15min"),
            ]:
                if entry.get(outcome_key) is None:
                    if mins_to_goal <= window:
                        entry[outcome_key] = "HIT"
                    elif mins_since >= window:
                        entry[outcome_key] = "MISS"
            if entry.get("outcome_full") is None:
                entry["outcome_full"] = "HIT"
                entry["goal_minute_full"] = minute
                log.info(
                    f"  FN-COST: [{entry.get('block_reason')}] {entry['team_name']} "
                    f"blocked at {entry['game_minute']}' then scored at {minute}' "
                    f"(+{mins_to_goal}') GPS={entry.get('gps', '?')} [{entry.get('league', '?')}]"
                )
            if is_finished:
                entry["resolved"] = True
                any_updated = True
        else:
            for window, outcome_key in [
                (5, "outcome_5min"), (10, "outcome_10min"), (15, "outcome_15min"),
            ]:
                if entry.get(outcome_key) is None and mins_since >= window:
                    entry[outcome_key] = "MISS"
            if is_finished:
                if entry.get("outcome_full") is None:
                    entry["outcome_full"] = "MISS"
                for outcome_key in ("outcome_5min", "outcome_10min", "outcome_15min"):
                    if entry.get(outcome_key) is None:
                        entry[outcome_key] = "MISS"
                entry["resolved"] = True
                any_updated = True

    if any_updated:
        rewrite_blocked_file()


def _evaluate_fl_shadow(fid: int, tid: int, m: int, tname: str, hist: list[tuple[int, float]]) -> None:
    """v10.50: Evaluate ONE new events-feed SOT event for a SHADOW (virtual) signal.

    Candidate rules (FASTLANE_PROPOSAL.md 3.1):
      - SOT burst: team's 2nd SOT event within 3 game-minutes, OR
      - goal-proximity: SOT event within 2 min after any Goal event, AND
      - minute window 21-80' (80' hard stop, same as stats path),
      - no REAL signal for this team in the last 180s,
      - no shadow for this team in the last 180s,
      - daily cap 400 shadow records.
    NEVER sends Telegram messages, never gates anything. Recording only.
    """
    global _fl_shadow_count_today, _fl_shadow_count_date
    try:
        # Minute window
        if not (21 <= m <= 80):
            return

        # Daily cap + date rollover
        _today = time.strftime("%Y-%m-%d")
        if _fl_shadow_count_date != _today:
            _fl_shadow_count_date = _today
            _fl_shadow_count_today = 0
        if _fl_shadow_count_today >= 400:
            return

        # Trigger 1 — SOT burst: previous SOT event within 3 game-minutes
        _trigger = None
        for pm, _pts in hist:
            if m - 3 <= pm <= m - 1:
                _trigger = "sot_burst"
                break

        # Trigger 2 — goal proximity: any goal event within 2 min before this shot
        if _trigger is None:
            for gmin, _gtid in _fl_goal_minutes.get(fid, []):
                if m - 2 <= gmin <= m:
                    _trigger = "goal_proximity"
                    break
        if _trigger is None:
            return

        # Gate: no real signal for this team in the last 180s
        _team_sig = signaled_teams.get((fid, tid))
        if _team_sig and (time.time() - (_team_sig.get("last_signal_time", 0) or 0)) < 180:
            return

        # Gate: shadow dedupe — 1 per (fixture, team) per 180s
        if (time.time() - _fl_shadow_dedupe.get((fid, tid), 0)) < 180:
            return

        # Context: stats-side state + fixture info
        # v10.50-BUGFIX: team_state stores last_sot (not sot); current GPS
        # lives in team_gps_history[(fid,tid)][-1]
        _stats_sot = (team_state.get((fid, tid), {}) or {}).get("last_sot")
        _gh = team_gps_history.get((fid, tid)) or []
        _gps = _gh[-1].get("gps") if _gh else None
        _ev_sot_total = len(hist) + 1  # history + this event

        _league = "?"
        _is_home = False
        _team_goals = 0
        _opp_goals = 0
        _fx = find_cached_fixture(fid)
        if _fx:
            try:
                _league = _fx.get("league", {}).get("name") or "?"
                _home_id = _fx.get("teams", {}).get("home", {}).get("id")
                _is_home = (_home_id == tid)
                _sh = _fx.get("goals", {}).get("home") or 0
                _sa = _fx.get("goals", {}).get("away") or 0
                _team_goals = (_sh if _is_home else _sa) or 0
                _opp_goals = (_sa if _is_home else _sh) or 0
            except Exception:
                pass

        _fl_shadow_dedupe[(fid, tid)] = time.time()
        _fl_shadow_count_today += 1

        entry = {
            "shadow_time": time.time(),
            "shadow_clock": time.strftime("%Y-%m-%d %H:%M"),
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": _league,
            "game_minute": m,
            "tier": "FAST_SHADOW",
            "trigger": _trigger,
            "ev_sot": _ev_sot_total,
            "stats_sot": _stats_sot,
            "lag_sot": (_ev_sot_total - _stats_sot) if _stats_sot is not None else None,
            "gps": round(float(_gps), 1) if _gps is not None else None,
            "goals_at_shadow": _team_goals,
            "opponent_goals_at_shadow": _opp_goals,
            "is_home": bool(_is_home),
            "scoreline": (
                "winning" if _team_goals > _opp_goals
                else "drawing" if _team_goals == _opp_goals
                else "losing"
            ),
            "outcome_5min": None,
            "outcome_10min": None,
            "outcome_15min": None,
            "outcome_full": None,
            "goal_minute_5": None,
            "goal_minute_10": None,
            "goal_minute_15": None,
            "goal_minute_full": None,
            "resolved": False,
            "version": BOT_VERSION,
        }
        fastlane_shadow.append(entry)
        with open(FASTLANE_SHADOW_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        _lag_txt = f"+{entry['lag_sot']} ahead" if entry["lag_sot"] is not None else "n/a"
        log.info(
            f"  v10.50 SHADOW: {tname} F{fid} {m}' [{_trigger}] "
            f"evSOT={_ev_sot_total} statsSOT={_stats_sot if _stats_sot is not None else '?'} ({_lag_txt}) "
            f"— virtual, NOT sent"
        )
    except Exception as e:
        log.debug(f"  v10.50 shadow-eval error: {e}")


def _load_fastlane_shadow() -> list[dict]:
    """v10.50: Load fast-lane shadow records from JSONL (survives restarts)."""
    entries: list[dict] = []
    if not os.path.exists(FASTLANE_SHADOW_FILE):
        return entries
    try:
        with open(FASTLANE_SHADOW_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"v10.50: Failed to load fast-lane shadow file: {e}")
    return entries


def rewrite_fastlane_file() -> None:
    """v10.50: Rewrite shadow JSONL after in-place resolution updates.

    Same merge discipline as rewrite_blocked_file(): disk entries are the
    base, in-memory entries override (they carry updated resolution status).
    """
    try:
        disk_entries: dict[tuple, dict] = {}
        if os.path.exists(FASTLANE_SHADOW_FILE):
            with open(FASTLANE_SHADOW_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("fixture_id"), obj.get("team_id"),
                               obj.get("shadow_time"))
                        if None not in key:
                            disk_entries[key] = obj
                    except Exception:
                        continue
        for entry in fastlane_shadow:
            key = (entry.get("fixture_id"), entry.get("team_id"),
                   entry.get("shadow_time"))
            if None not in key:
                disk_entries[key] = entry
        tmp = FASTLANE_SHADOW_FILE + ".tmp"
        with open(tmp, "w") as f:
            for entry in disk_entries.values():
                f.write(json.dumps(entry, default=str) + "\n")
        os.replace(tmp, FASTLANE_SHADOW_FILE)
    except Exception as e:
        log.warning(f"v10.50: Failed to rewrite fast-lane shadow file: {e}")


def _load_boxburst_shadow() -> list[dict]:
    """v10.75: Load box-burst shadow records from JSONL (survives restarts)."""
    entries: list[dict] = []
    if not os.path.exists(BOXBURST_SHADOW_FILE):
        return entries
    try:
        with open(BOXBURST_SHADOW_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"v10.75: Failed to load box-burst shadow file: {e}")
    return entries


def rewrite_boxburst_file() -> None:
    """v10.75: Rewrite box-burst shadow JSONL after in-place resolution updates.

    Same merge discipline as rewrite_fastlane_file(): disk entries are the
    base, in-memory entries override (they carry updated resolution status).
    """
    try:
        disk_entries: dict[tuple, dict] = {}
        if os.path.exists(BOXBURST_SHADOW_FILE):
            with open(BOXBURST_SHADOW_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("fixture_id"), obj.get("team_id"),
                               obj.get("shadow_time"))
                        if None not in key:
                            disk_entries[key] = obj
                    except Exception:
                        continue
        for entry in boxburst_shadow:
            key = (entry.get("fixture_id"), entry.get("team_id"),
                   entry.get("shadow_time"))
            if None not in key:
                disk_entries[key] = entry
        tmp = BOXBURST_SHADOW_FILE + ".tmp"
        with open(tmp, "w") as f:
            for entry in disk_entries.values():
                f.write(json.dumps(entry, default=str) + "\n")
        os.replace(tmp, BOXBURST_SHADOW_FILE)
    except Exception as e:
        log.warning(f"v10.75: Failed to rewrite box-burst shadow file: {e}")


def evaluate_boxburst_shadow(
    fid: int, tid: int, tname: str, league: str,
    minute: int, sot: int, shots_inside_box: int, total_shots: int,
    gps: float, is_home: bool, score_home: int, score_away: int,
    red_cards: int | None = None, opp_red_cards: int | None = None,
) -> None:
    """v10.75: BOX-BURST shadow — SOT 1-2 but heavy box volume.

    Records a VIRTUAL signal (never sent, zero credits) the FIRST time a
    team-side crosses the box-burst cell (SOT 1-2 + ib>=8 within 21-61'),
    then resolves it exactly like a real signal via check_fastlane_shadow
    (which now walks BOTH shadow stores). The whole class stays shadow
    until the go-live rule in the constants block is met — flip
    BOXBURST_LIVE to also send a compact Telegram alert.

    Rising is a RECORDED flag (ib grew by >=1 over the last <=2 polls of
    the shadow's own history): the backtest difference (+5pp on n=13) is
    too thin to require. The poll recency fields (ib_5m_ago/ib_10m_ago)
    are never populated in the data, so this keeps its own short history
    (surge-watch pattern). LOGGING ONLY — never gates, never blocks,
    never sends while BOXBURST_LIVE is False.
    """
    global _boxburst_count_today, _boxburst_count_date
    try:
        if (fid, tid) in _boxburst_fired:
            return  # one shadow per team-side per match (first crossing)

        # own short ib history — appended on EVERY poll before any cell
        # check, so the rising flag can compare against the previous <=2
        # polls even when this poll doesn't cross the cell yet
        _hist = _boxburst_ib_hist.setdefault((fid, tid), [])
        _recent = [ib for m_, ib in _hist[-2:] if minute - m_ <= 10]
        _hist.append((minute, shots_inside_box))
        if len(_hist) > 4:
            del _hist[: len(_hist) - 4]

        # daily cap + date rollover
        _today = time.strftime("%Y-%m-%d")
        if _boxburst_count_date != _today:
            _boxburst_count_date = _today
            _boxburst_count_today = 0
        if _boxburst_count_today >= BOXBURST_DAILY_CAP:
            return

        # cell conditions
        if not (BOXBURST_MIN_MINUTE <= minute <= BOXBURST_MAX_MINUTE):
            return
        if not (BOXBURST_SOT_MIN <= sot <= BOXBURST_SOT_MAX):
            return
        if shots_inside_box < BOXBURST_IB_MIN:
            return

        # rising flag: >=1 new ib vs the max of the previous <=2 polls,
        # only entries within 10 game minutes (a polling gap must not
        # fake recency)
        ib_rising = bool(_recent and shots_inside_box - max(_recent) >= 1)

        team_goals = score_home if is_home else score_away
        opp_goals = score_away if is_home else score_home

        _boxburst_fired[(fid, tid)] = time.time()
        _boxburst_count_today += 1
        _shadow_tags["BOX"] = _shadow_tags.get("BOX", 0) + 1  # heartbeat

        entry = {
            "shadow_time": time.time(),
            "shadow_clock": time.strftime("%Y-%m-%d %H:%M"),
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": league,
            "game_minute": minute,
            "tier": "BOXBURST_SHADOW",
            "trigger": f"ib>={BOXBURST_IB_MIN} sot{BOXBURST_SOT_MIN}-{BOXBURST_SOT_MAX}",
            "sot": sot,
            "shots_inside_box": shots_inside_box,
            "total_shots": total_shots,
            "ib_ratio": round(shots_inside_box / total_shots, 2) if total_shots else None,
            "ib_rising": ib_rising,
            "gps": round(float(gps), 1) if gps is not None else None,
            "real_signal_before": (fid, tid) in signaled_teams,
            "goals_at_shadow": team_goals,
            "opponent_goals_at_shadow": opp_goals,
            "is_home": bool(is_home),
            "scoreline": (
                "winning" if team_goals > opp_goals
                else "drawing" if team_goals == opp_goals
                else "losing"
            ),
            "red_cards": red_cards,
            "opp_red_cards": opp_red_cards,
            "outcome_5min": None,
            "outcome_10min": None,
            "outcome_15min": None,
            "outcome_full": None,
            "goal_minute_5": None,
            "goal_minute_10": None,
            "goal_minute_15": None,
            "goal_minute_full": None,
            "resolved": False,
            "version": BOT_VERSION,
        }
        boxburst_shadow.append(entry)
        with open(BOXBURST_SHADOW_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        _ib_pct = (
            f"{shots_inside_box / total_shots * 100:.0f}%" if total_shots > 0 else "N/A"
        )
        log.info(
            f"  v10.75 BOX-BURST SHADOW: {tname} F{fid} {minute}' "
            f"SOT={sot} IB={shots_inside_box}/{total_shots} ({_ib_pct}) "
            f"GPS={gps:.0f} rising={ib_rising} — virtual, NOT sent"
        )

        # live flip (only after the go-live rule is met; see constants block)
        if BOXBURST_LIVE:
            try:
                with httpx.Client(timeout=10.0) as _c:
                    send_telegram(_c, (
                        f"\U0001f4e6 BOX-BURST — {tname} ({minute}')\n"
                        f"{score_home}-{score_away} ({league})\n"
                        f"SOT {sot} · {shots_inside_box} box shots of "
                        f"{total_shots} total ({_ib_pct}) · GPS {gps:.0f}\n"
                        f"Box volume high while SOT low — box-burst class"
                        f"{' rising' if ib_rising else ''}\n"
                        f"[{BOT_VERSION}]"
                    ))
            except Exception as _e:
                log.warning(f"  v10.75 box-burst live send failed: {_e}")
    except Exception as e:
        log.debug(f"  v10.75 box-burst eval error: {e}")


def _load_goalburst_shadow() -> list[dict]:
    """v10.76: Load goal-burst shadow records from JSONL (survives restarts)."""
    entries: list[dict] = []
    if not os.path.exists(GOALBURST_SHADOW_FILE):
        return entries
    try:
        with open(GOALBURST_SHADOW_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"v10.76: Failed to load goal-burst shadow file: {e}")
    return entries


def rewrite_goalburst_file() -> None:
    """v10.76: Rewrite goal-burst shadow JSONL after in-place resolution updates.

    Same merge discipline as the other shadow files: disk entries are the
    base, in-memory entries override (they carry updated resolution status).
    """
    try:
        disk_entries: dict[tuple, dict] = {}
        if os.path.exists(GOALBURST_SHADOW_FILE):
            with open(GOALBURST_SHADOW_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("fixture_id"), obj.get("gb_class"),
                               obj.get("shadow_time"))
                        if None not in key:
                            disk_entries[key] = obj
                    except Exception:
                        continue
        for entry in goalburst_shadow:
            key = (entry.get("fixture_id"), entry.get("gb_class"),
                   entry.get("shadow_time"))
            if None not in key:
                disk_entries[key] = entry
        tmp = GOALBURST_SHADOW_FILE + ".tmp"
        with open(tmp, "w") as f:
            for entry in disk_entries.values():
                f.write(json.dumps(entry, default=str) + "\n")
        os.replace(tmp, GOALBURST_SHADOW_FILE)
    except Exception as e:
        log.warning(f"v10.76: Failed to rewrite goal-burst shadow file: {e}")


def _goalburst_send_alert(entry: dict, label: str) -> None:
    """v10.76: compact Telegram alert for a LIVE goal-burst class.

    Own short-lived client (the box-burst live-flip pattern) — never on the
    API path. Informational: reports the empirical continuation so the user
    can compare it against the live Over price BEFORE deciding anything.

    v10.78: ODDS IN THE ALERT — one fast quota-guarded capture
    (for_message mode: single pass, no retry sleep) prices the exact
    'one more goal' market (Over total+0.5) and the alert carries the
    MARKET PRICE (source-labeled LIVE vs PRE ref, suspect flagged) plus
    the class BREAK-EVEN (1 / empirical continuation: G2 84% -> 1.19).
    ~2 credits per alert (~6-8/day), skipped at <=5 quota remaining;
    a failed fetch degrades to 'no market price captured' and the alert
    still goes out.
    """
    # v10.78: market price for the next-goal line (total + 0.5)
    _odds_line = ""
    try:
        _fid = entry.get("fixture_id")
        _tot = entry.get("total_goals_at")
        if _fid and _tot is not None:
            with httpx.Client(timeout=10.0) as _oc:
                _od = fetch_signal_odds(
                    _oc, _fid, int(_tot),
                    game_minute=entry.get("game_minute"), for_message=True,
                )
            if _od and _od.get("over_odds"):
                _live = (
                    _od.get("odds_source") == "live"
                    and not _od.get("suspect")
                )
                _sus = (
                    " \u26a0\ufe0f impossible price — ignore"
                    if _od.get("suspect") else ""
                )
                _odds_line = (
                    f"market O{float(_od['over_line']):.1f} "
                    f"@{float(_od['over_odds']):.2f} "
                    f"({_od.get('bookmaker') or '?'} \u00b7 "
                    f"{'LIVE' if _live else 'PRE ref'}{_sus})"
                )
                # record the captured price on the shadow entry (ledger)
                entry["odds_over_line"] = _od.get("over_line")
                entry["odds_over_odds"] = _od.get("over_odds")
                entry["odds_source"] = _od.get("odds_source")
                entry["odds_suspect"] = _od.get("suspect")
    except Exception as _e:
        log.debug(f"  v10.78 goal-burst odds fetch failed: {_e}")
    if not _odds_line:
        _odds_line = "no market price captured"

    # v10.78: class break-even from the backtest continuation rate
    _rate = _GB_CONT_RATE.get(entry.get("gb_class") or "", 0.0)
    _be = f"{1.0 / _rate:.2f}" if _rate > 0 else "n/a"

    try:
        with httpx.Client(timeout=10.0) as _c:
            send_telegram(_c, (
                f"\u26a1 GOAL-BURST {entry['gb_class']} — {label}\n"
                f"{entry['score_home']}-{entry['score_away']} "
                f"({entry['league']}) {entry['game_minute']}'\n"
                f"{entry['trigger']} — historical: {entry['gb_hist']}\n"
                f"maxGPS {entry['max_gps']:.0f} — "
                f"{'BLIND class (no pressure signal fired)' if entry['gps_blind'] else 'pressure already lit'}\n"
                f"bet frame: {entry['bet_desc']} — informational, compare the live O-line price\n"
                f"\U0001f4b0 {_odds_line} | break-even {_be} — bet only above\n"
                f"[{BOT_VERSION}]"
            ))
    except Exception as _e:
        log.warning(f"  v10.76 goal-burst live send failed: {_e}")


def evaluate_goalburst_shadow(
    fid: int, tname: str, league: str, minute: int,
    score_home: int, score_away: int,
    sot: int, opp_sot: int, gps: float, opp_gps: float,
    is_home: bool, red_cards: int | None = None, opp_red_cards: int | None = None,
) -> None:
    """v10.76: GOAL-BURST (banked-goals) shadow — the totals path.

    The Lille 2-3 Betis post-mortem (Sep 9: 5 goals by 53', ZERO signals —
    Betis' 3 SOT *were* the 3 goals, GOAL-SHOT NET -> effective 0, GPS
    52/65 never lit) exposed the structural blind spot: the system only
    certifies sustained NON-goal pressure, so goal-burst games are
    invisible. This class treats banked goals as evidence: ONE virtual
    match-level record per (fixture, class) at the FIRST poll inside the
    window, resolved like signals but with ANY-goal semantics (own goals
    count — that is how Over lines settle). Backtest cells in the
    constants block. LOGGING ONLY — never gates, never blocks. G2 sends
    a compact alert while GOALBURST_LIVE (the cell already clears the
    box-burst go-live bar); G1/G3 alert only behind their own flags.
    Zero extra credits (runs on the stats the poll already fetched; the
    alert uses its own short-lived Telegram client).
    """
    global _goalburst_count_today, _goalburst_count_date
    try:
        total = (score_home or 0) + (score_away or 0)
        if total < 1:
            return

        # daily cap + date rollover
        _today = time.strftime("%Y-%m-%d")
        if _goalburst_count_date != _today:
            _goalburst_count_date = _today
            _goalburst_count_today = 0
        if _goalburst_count_today >= GOALBURST_DAILY_CAP:
            return

        fire: list[str] = []
        if (total >= 1 and minute <= GOALBURST_G1_MAX_MINUTE
                and (fid, "G1") not in _goalburst_fired):
            fire.append("G1")
        if (total >= 2 and minute <= GOALBURST_G2_MAX_MINUTE
                and (fid, "G2") not in _goalburst_fired):
            fire.append("G2")
        if (total >= 3 and minute <= GOALBURST_G3_MAX_MINUTE
                and (fid, "G3") not in _goalburst_fired):
            fire.append("G3")
        if not fire:
            return

        c_sot = (sot or 0) + (opp_sot or 0)
        max_gps = max(gps or 0.0, opp_gps or 0.0)
        sig_before = any(
            _e.get("fixture_id") == fid and (_e.get("game_minute") or 0) <= minute
            for _e in signal_outcomes
        )

        # friendly label: cached fixture names when available (zero credits)
        label = f"{tname} game F{fid}"
        try:
            _fx = find_cached_fixture(fid)
            if _fx:
                label = (
                    f"{_fx['teams']['home']['name']}-"
                    f"{_fx['teams']['away']['name']}"
                )
        except Exception:
            pass

        for cls in fire:
            _trig, _target, _desc, _hist = _GB_META[cls]
            _goalburst_fired[(fid, cls)] = time.time()
            _goalburst_count_today += 1
            _shadow_tags["GB"] = _shadow_tags.get("GB", 0) + 1  # heartbeat

            entry = {
                "shadow_time": time.time(),
                "shadow_clock": time.strftime("%Y-%m-%d %H:%M"),
                "fixture_id": fid,
                "team_id": None,           # match-level: no owning team
                "match_level": True,
                "gb_class": cls,
                "team_name": tname,        # polling side (context only)
                "is_home": bool(is_home),
                "league": league,
                "game_minute": minute,
                "tier": "GOALBURST_SHADOW",
                "trigger": _trig,
                "bet_target": _target,
                "bet_desc": _desc,
                "gb_hist": _hist,
                "total_goals_at": total,
                "score_home": score_home,
                "score_away": score_away,
                "scoreline": (
                    "level" if (score_home or 0) == (score_away or 0)
                    else f"{abs((score_home or 0) - (score_away or 0))}-diff"
                ),
                "combined_sot": c_sot,
                "clinical": bool(c_sot <= total + 1),
                "max_gps": round(float(max_gps), 1),
                "gps_blind": bool(max_gps < 70),
                "real_signal_before": sig_before,
                "red_cards": red_cards,
                "opp_red_cards": opp_red_cards,
                "outcome_5min": None,
                "outcome_10min": None,
                "outcome_15min": None,
                "outcome_full": None,
                "goal_minute_5": None,
                "goal_minute_10": None,
                "goal_minute_15": None,
                "goal_minute_full": None,
                "goals_after_crossing": None,
                "ft_total": None,
                "bet_hit": None,
                "resolved": False,
                "version": BOT_VERSION,
            }
            # v10.78: the LIVE alert (which also captures odds and stamps
            # them onto this entry) runs BEFORE the file append so the
            # captured market price lands in the persisted shadow record.
            # _goalburst_send_alert is fully exception-guarded — the
            # record can never be lost to a Telegram/odds failure.
            _live_on = (
                (cls == "G2" and GOALBURST_LIVE)
                or (cls == "G3" and GOALBURST_LIVE_G3)
                or (cls == "G1" and GOALBURST_LIVE_G1)
            )
            if _live_on:
                _goalburst_send_alert(entry, label)

            goalburst_shadow.append(entry)
            with open(GOALBURST_SHADOW_FILE, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            log.info(
                f"  v10.76 GOAL-BURST {cls} SHADOW: {label} "
                f"{score_home}-{score_away} {minute}' {_trig} — cSOT={c_sot} "
                f"maxGPS={max_gps:.0f} "
                f"{'GPS-BLIND' if entry['gps_blind'] else 'gps-lit'} "
                f"sig_before={sig_before} — virtual, NOT sent"
            )
    except Exception as e:
        log.debug(f"  v10.76 goal-burst eval error: {e}")


def resolve_goalburst_with_goal_events(
    entry: dict, goal_events: list[dict],
    home_goals: int | None = None, away_goals: int | None = None,
) -> bool:
    """v10.76: FT resolution for MATCH-LEVEL goal-burst records.

    Different semantics from the team-level resolver: ANY goal after the
    crossing minute counts — either team, own goals included (that is how
    Over lines settle). Uses the true event minutes (v10.66 discipline):
    live-stamped windows are recomputed against them, a live HIT with no
    event AND no final-score confirmation flips to MISS (phantom), and an
    events coverage hole with final-score confirmation is HELD (never
    corrupt a true HIT). Also stamps ft_total + bet_hit so the totals bet
    (reach bet_target) is graded honestly.
    Returns True if any field was updated (needs file rewrite).
    """
    updated = False
    sig_minute = entry["game_minute"]
    tot_at = entry.get("total_goals_at") or 0

    goals_after: list[tuple[int, dict]] = []
    for g in goal_events or []:
        try:
            _gm = (g.get("minute") or 0) + (g.get("minute_extra") or 0)
        except Exception:
            _gm = g.get("minute") or 0
        if _gm > sig_minute:
            goals_after.append((_gm, g))
    goals_after.sort(key=lambda t: t[0])

    final_total = None
    if home_goals is not None and away_goals is not None:
        final_total = (home_goals or 0) + (away_goals or 0)

    if goals_after:
        first_min = goals_after[0][0]
        mins_to_goal = first_min - sig_minute
        # v10.79: stamp the first ANY-goal scorer after the crossing (this
        # is how the Over bet settles — either team, own goals included).
        # Idempotent, zero credits, ledger-only.
        _gb79 = ((goals_after[0][1].get("player") or "")).strip() or "?"
        if entry.get("post_crossing_scorer") != _gb79:
            entry["post_crossing_scorer"] = _gb79
            entry["post_crossing_scorer_minute"] = first_min
            updated = True
        for _w, _ok, _mk in [
            (5, "outcome_5min", "goal_minute_5"),
            (10, "outcome_10min", "goal_minute_10"),
            (15, "outcome_15min", "goal_minute_15"),
        ]:
            _true = "HIT" if mins_to_goal <= _w else "MISS"
            if entry.get(_ok) != _true:
                entry[_ok] = _true
                updated = True
            if _true == "HIT":
                if entry.get(_mk) != first_min:
                    entry[_mk] = first_min
                    updated = True
            elif entry.get(_mk) is not None:
                entry.pop(_mk, None)
                updated = True
        if entry.get("outcome_full") != "HIT":
            entry["outcome_full"] = "HIT"
            updated = True
        if entry.get("goal_minute_full") != first_min:
            entry["goal_minute_full"] = first_min
            updated = True
        entry["goals_after_crossing"] = len(goals_after)
        if final_total is not None:
            entry["ft_total"] = final_total
            entry["bet_hit"] = final_total >= (entry.get("bet_target") or 0)
        entry["resolved"] = True
        return True

    # no event goal after the crossing minute
    if final_total is not None and final_total > tot_at:
        # events coverage hole: the final score proves a goal the events
        # feed missed — HOLD the live verdict, never corrupt a true HIT
        if entry.get("outcome_full") == "HIT":
            entry["events_missing_goal"] = True
        entry["ft_total"] = final_total
        entry["bet_hit"] = final_total >= (entry.get("bet_target") or 0)
        entry["resolved"] = True
        return True

    for _ok in ("outcome_5min", "outcome_10min", "outcome_15min", "outcome_full"):
        if entry.get(_ok) != "MISS":
            entry[_ok] = "MISS"
            updated = True
    for _mk in ("goal_minute_5", "goal_minute_10", "goal_minute_15",
                "goal_minute_full"):
        if entry.get(_mk) is not None:
            entry.pop(_mk, None)
            updated = True
    if final_total is not None:
        entry["ft_total"] = final_total
        entry["bet_hit"] = final_total >= (entry.get("bet_target") or 0)
    entry["resolved"] = True
    return updated or True


def check_fastlane_shadow(fixture: dict, client: httpx.Client = None) -> None:
    """v10.50: Resolve pending fast-lane shadow records for this fixture.

    Mirrors check_blocked_outcomes (goal-events resolution when FT, live
    score fallback otherwise). A HIT means the team scored AFTER the shadow
    minute — logged as SHADOW-HIT. EOD joins these with real signals to
    compute speed gain + duplicates. LOGGING ONLY.

    v10.75: now resolves BOTH shadow stores — the v10.50 fast-lane records
    AND the v10.75 box-burst records — with identical semantics, so every
    existing call site (discovery, periodic resolver, poll path) resolves
    both classes for free.

    v10.76: also walks the goal-burst store; MATCH-LEVEL entries (the
    banked-goals class) resolve on ANY goal after the crossing minute —
    either team, own goals included (Over-lines settlement semantics).
    """
    if not fastlane_shadow and not boxburst_shadow and not goalburst_shadow:
        return
    fid = fixture["fixture"]["id"]
    status = fixture["fixture"]["status"]["short"]
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    is_finished = status not in LIVE_STATUSES

    goal_events = None
    if is_finished and client is not None:
        goal_events = fetch_goal_events(client, fid)

    any_updated = False
    # v10.75: walk BOTH shadow stores (fast-lane v10.50 + box-burst v10.75)
    # with identical semantics — every existing call site resolves both.
    # v10.76: + the goal-burst store (match-level banked-goals records).
    for _shadow_store, _shadow_kind in (
        (fastlane_shadow, "fast-lane"), (boxburst_shadow, "box-burst"),
        (goalburst_shadow, "goal-burst"),
    ):
        if not _shadow_store:
            continue
        for entry in _shadow_store:
            if entry["fixture_id"] != fid or entry.get("resolved"):
                continue

            # v10.76: MATCH-LEVEL goal-burst records resolve on ANY goal
            # (either team, own goals included — Over-lines semantics).
            if entry.get("match_level"):
                if goal_events is not None:
                    if resolve_goalburst_with_goal_events(
                        entry, goal_events,
                        home_goals=home_goals, away_goals=away_goals,
                    ):
                        any_updated = True
                    if entry.get("outcome_full") == "HIT":
                        log.info(
                            f"  SHADOW-HIT: {entry.get('gb_class', '?')} goal-burst "
                            f"shadow at {entry['game_minute']}' then a goal came "
                            f"(FT resolution) [{entry.get('trigger')}] "
                            f"[{entry.get('league', '?')}]"
                        )
                    continue
                _gb_now_total = (home_goals or 0) + (away_goals or 0)
                _gb_since = _gb_now_total - (entry.get("total_goals_at") or 0)
                _gb_mins_since = minute - entry["game_minute"]
                if _gb_since > 0:
                    _gb_mins_to_goal = minute - entry["game_minute"]
                    for _w, _ok in [
                        (5, "outcome_5min"), (10, "outcome_10min"), (15, "outcome_15min"),
                    ]:
                        if entry.get(_ok) is None:
                            if _gb_mins_to_goal <= _w:
                                entry[_ok] = "HIT"
                            elif _gb_mins_since >= _w:
                                entry[_ok] = "MISS"
                    if entry.get("outcome_full") is None:
                        entry["outcome_full"] = "HIT"
                        entry["goal_minute_full"] = minute
                        log.info(
                            f"  SHADOW-HIT: {entry.get('gb_class', '?')} goal-burst "
                            f"shadow at {entry['game_minute']}' then a goal at {minute}' "
                            f"(+{_gb_mins_to_goal}') [{entry.get('trigger')}] "
                            f"[{entry.get('league', '?')}]"
                        )
                    if is_finished:
                        entry["ft_total"] = _gb_now_total
                        entry["bet_hit"] = (
                            _gb_now_total >= (entry.get("bet_target") or 0)
                        )
                        entry["resolved"] = True
                        any_updated = True
                else:
                    for _w, _ok in [
                        (5, "outcome_5min"), (10, "outcome_10min"), (15, "outcome_15min"),
                    ]:
                        if entry.get(_ok) is None and _gb_mins_since >= _w:
                            entry[_ok] = "MISS"
                    if is_finished:
                        if entry.get("outcome_full") is None:
                            entry["outcome_full"] = "MISS"
                        for _ok in ("outcome_5min", "outcome_10min", "outcome_15min"):
                            if entry.get(_ok) is None:
                                entry[_ok] = "MISS"
                        entry["ft_total"] = _gb_now_total
                        entry["bet_hit"] = (
                            _gb_now_total >= (entry.get("bet_target") or 0)
                        )
                        entry["resolved"] = True
                        any_updated = True
                continue

            if goal_events is not None:
                if resolve_with_goal_events(
                    entry, goal_events, home_id, away_id,
                    home_goals=home_goals, away_goals=away_goals,
                    entry_kind="shadow",
                ):
                    any_updated = True
                if entry.get("outcome_full") == "HIT":
                    log.info(
                        f"  SHADOW-HIT: {entry['team_name']} {_shadow_kind} shadow at "
                        f"{entry['game_minute']}' then scored (FT resolution) "
                        f"[{entry.get('trigger')}] [{entry.get('league', '?')}]"
                    )
                continue

            # --- Fallback: live tracking (match still in progress) ---
            current_team_goals = home_goals if entry["is_home"] else away_goals
            goals_since_shadow = current_team_goals - entry["goals_at_shadow"]
            mins_since = minute - entry["game_minute"]

            if goals_since_shadow > 0:
                mins_to_goal = minute - entry["game_minute"]
                for window, outcome_key in [
                    (5, "outcome_5min"), (10, "outcome_10min"), (15, "outcome_15min"),
                ]:
                    if entry.get(outcome_key) is None:
                        if mins_to_goal <= window:
                            entry[outcome_key] = "HIT"
                        elif mins_since >= window:
                            entry[outcome_key] = "MISS"
                if entry.get("outcome_full") is None:
                    entry["outcome_full"] = "HIT"
                    entry["goal_minute_full"] = minute
                    log.info(
                        f"  SHADOW-HIT: {entry['team_name']} {_shadow_kind} shadow at "
                        f"{entry['game_minute']}' then scored at {minute}' (+{mins_to_goal}') "
                        f"[{entry.get('trigger')}] [{entry.get('league', '?')}]"
                    )
                if is_finished:
                    entry["resolved"] = True
                    any_updated = True
            else:
                for window, outcome_key in [
                    (5, "outcome_5min"), (10, "outcome_10min"), (15, "outcome_15min"),
                ]:
                    if entry.get(outcome_key) is None and mins_since >= window:
                        entry[outcome_key] = "MISS"
                if is_finished:
                    if entry.get("outcome_full") is None:
                        entry["outcome_full"] = "MISS"
                    for outcome_key in ("outcome_5min", "outcome_10min", "outcome_15min"):
                        if entry.get(outcome_key) is None:
                            entry[outcome_key] = "MISS"
                    entry["resolved"] = True
                    any_updated = True

    if any_updated:
        rewrite_fastlane_file()
        rewrite_boxburst_file()
        rewrite_goalburst_file()


def _update_poisson_calibration(entry: dict) -> None:
    """v10.49: Accumulate predicted-vs-actual total goals per league + xg source.

    GOALS_PER_SOT (0.31) is a global constant that was never calibrated per
    league. This accumulates the data needed to calibrate it (n, pred_sum,
    actual_sum, bias per league|xg_source). LOGGING ONLY — never changes
    predictions, signals, or thresholds.
    """
    try:
        _pred = entry.get("pred_expected_total")
        _actual = entry.get("pred_actual_total_goals")
        if _pred is None or _actual is None:
            return
        _lg = entry.get("league") or "UNKNOWN"
        _src = entry.get("pred_xg_source") or "unknown"
        _key = f"{_lg}|{_src}"
        rec = _poisson_calibration.setdefault(
            _key, {"league": _lg, "xg_source": _src, "n": 0, "pred_sum": 0.0, "actual_sum": 0.0}
        )
        rec["n"] += 1
        rec["pred_sum"] += float(_pred)
        rec["actual_sum"] += float(_actual)
        if rec["pred_sum"] > 0:
            rec["bias"] = round(rec["actual_sum"] / rec["pred_sum"], 3)
        tmp = CALIBRATION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_poisson_calibration, f, indent=2, default=str)
        os.replace(tmp, CALIBRATION_FILE)
    except Exception as e:
        log.debug(f"v10.49 calibration error: {e}")


def _load_poisson_calibration() -> None:
    """v10.49: Load persisted calibration accumulator (survives restarts)."""
    if not os.path.exists(CALIBRATION_FILE):
        return
    try:
        with open(CALIBRATION_FILE, "r") as f:
            _loaded = json.load(f)
        if isinstance(_loaded, dict):
            _poisson_calibration.clear()
            _poisson_calibration.update(_loaded)
            log.info(f"v10.49: Poisson calibration loaded ({len(_loaded)} league/source combos)")
    except Exception as e:
        log.warning(f"v10.49: Failed to load calibration file: {e}")


def _backup_ml_data(client: httpx.Client) -> None:
    """v10.44l: Auto-backup ML data files to Telegram at end of day.

    Sends signal_outcomes.jsonl and pressure_polls.jsonl as date-stamped
    documents. After successful backup, truncates the polls file to prevent
    unbounded growth (signals file is rewritten daily by existing logic).

    Tracks backup date in ML_BACKUP_SENT_FILE to avoid double-sends on restart.
    """
    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")

    # Check if already backed up today
    try:
        if os.path.exists(ML_BACKUP_SENT_FILE):
            with open(ML_BACKUP_SENT_FILE, "r") as f:
                if f.read().strip() == today_str:
                    return  # Already backed up today
    except Exception:
        pass

    _sent_files = []
    _polls_backup_ok = False
    _MAX_BYTES = 50 * 1024 * 1024  # Telegram 50MB limit

    # --- Backup signal_outcomes.jsonl ---
    if os.path.exists(OUTCOMES_FILE) and os.path.getsize(OUTCOMES_FILE) > 0:
        fsize = os.path.getsize(OUTCOMES_FILE)
        if fsize <= _MAX_BYTES:
            try:
                fname = f"ml_signals_{today_str}.jsonl"
                with open(OUTCOMES_FILE, "rb") as f:
                    client.post(
                        f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                        data={"chat_id": TELEGRAM_CHAT_ID},
                        files={"document": (fname, f, "application/jsonl")},
                        timeout=60.0,
                    )
                _lines = 0
                with open(OUTCOMES_FILE, "r") as f:
                    for _ in f:
                        _lines += 1
                _sent_files.append(f"{fname} ({_lines} signals, {fsize / 1024:.1f} KB)")
            except Exception as e:
                log.warning(f"ML backup: failed to send signals file: {e}")
        else:
            log.warning(f"ML backup: signals file too large ({fsize / 1024 / 1024:.1f} MB)")

    # --- Backup pressure_polls.jsonl ---
    # v10.44n: Gzip compress if > 10 MB to stay under Telegram's 50 MB limit
    if os.path.exists(POLL_DATA_FILE) and os.path.getsize(POLL_DATA_FILE) > 0:
        fsize = os.path.getsize(POLL_DATA_FILE)
        fname = f"ml_polls_{today_str}.jsonl"
        _gzip_used = fsize > 10 * 1024 * 1024
        try:
            if _gzip_used:
                # Compress in memory and send as .jsonl.gz
                buf = io.BytesIO()
                with open(POLL_DATA_FILE, "rb") as raw:
                    with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
                        gz.write(raw.read())
                buf.seek(0)
                gz_fname = f"ml_polls_{today_str}.jsonl.gz"
                gz_size = buf.getbuffer().nbytes
                client.post(
                    f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    files={"document": (gz_fname, buf, "application/gzip")},
                    timeout=60.0,
                )
                _lines = 0
                with open(POLL_DATA_FILE, "r") as f:
                    for _ in f:
                        _lines += 1
                _sent_files.append(f"{gz_fname} ({_lines} polls, {fsize / 1024 / 1024:.1f} MB -> {gz_size / 1024 / 1024:.1f} MB gz)")
            elif fsize <= _MAX_BYTES:
                with open(POLL_DATA_FILE, "rb") as f:
                    client.post(
                        f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                        data={"chat_id": TELEGRAM_CHAT_ID},
                        files={"document": (fname, f, "application/jsonl")},
                        timeout=60.0,
                    )
                _lines = 0
                with open(POLL_DATA_FILE, "r") as f:
                    for _ in f:
                        _lines += 1
                _sent_files.append(f"{fname} ({_lines} polls, {fsize / 1024:.1f} KB)")
            else:
                log.warning(f"ML backup: polls file too large even for gzip ({fsize / 1024 / 1024:.1f} MB)")
            _polls_backup_ok = True
        except Exception as e:
            log.warning(f"ML backup: failed to send polls file: {e}")

    # --- Truncate polls file only if its backup succeeded ---
    if _polls_backup_ok and os.path.exists(POLL_DATA_FILE):
        try:
            with open(POLL_DATA_FILE, "w") as f:
                pass  # Truncate to empty
            log.info("ML backup: truncated polls file after backup")
        except Exception as e:
            log.warning(f"ML backup: failed to truncate polls file: {e}")

    # --- Mark backup as sent & notify ---
    if _sent_files:
        try:
            with open(ML_BACKUP_SENT_FILE, "w") as f:
                f.write(today_str)
        except Exception:
            pass
        _msg = "📦 ML DATA BACKUP\n\n" + "\n".join(f"  ✅ {f}" for f in _sent_files)
        _msg += "\n\nSaved to Telegram. Polls file truncated for next day."
        send_telegram(client, _msg)
        log.info(f"v10.44l: ML data backup sent: {', '.join(_sent_files)}")


def load_all_outcomes() -> list[dict]:
    """v10.19.3: Load ALL signal outcomes from JSONL (resolved + pending).

    Previous version only loaded unresolved entries, which meant resolved
    outcomes were lost from memory after restart. When rewrite_outcomes_file()
    then ran, it would overwrite the file with only the pending entries,
    permanently deleting all resolved history.
    """
    all_entries = []
    if not os.path.exists(OUTCOMES_FILE):
        return all_entries
    try:
        with open(OUTCOMES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                all_entries.append(entry)
    except Exception as e:
        log.warning(f"  Failed to load outcomes file: {e}")
    return all_entries


# Keep old name as alias for any code that references it
load_pending_outcomes = load_all_outcomes


def rebuild_signaled_teams_from_file() -> None:
    """v10.44m: Rebuild signaled_teams and team_cooldown_polls from
    the outcomes file on startup, so cooldown/stale-suppress work
    immediately after a redeploy (no ghost signals).

    For each (fixture_id, team_id), keep the entry with the highest sig_num
    (or latest signal_time if sig_num missing/ambiguous due to redeploy resets).
    """
    global signaled_teams, team_cooldown_polls, signaled_fixtures

    if not os.path.exists(OUTCOMES_FILE):
        return

    try:
        with open(OUTCOMES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue

                fid = e.get("fixture_id")
                tid = e.get("team_id")
                if fid is None or tid is None:
                    continue

                key = (fid, tid)
                sig_time = e.get("signal_time", 0) or 0

                if key not in signaled_teams:
                    # First entry for this team — always populate
                    signaled_teams[key] = {
                        "count": 1,
                        "goals_at_last_signal": e.get("goals_at_signal", 0),
                        "sot_at_last_signal": e.get("sot", 0),
                        "xg_at_last_signal": e.get("xg"),
                        "last_signal_time": sig_time,
                        "last_gps": e.get("gps", 0),
                        "last_ib_ratio": e.get("ib_ratio", 0),
                    }
                    signaled_fixtures.add(fid)
                else:
                    # Keep the latest signal's data
                    existing = signaled_teams[key]
                    if sig_time > 0 and sig_time > existing.get("last_signal_time", 0):
                        existing["count"] = existing.get("count", 1) + 1
                        existing["goals_at_last_signal"] = e.get("goals_at_signal", 0)
                        existing["sot_at_last_signal"] = e.get("sot", 0)
                        existing["xg_at_last_signal"] = e.get("xg")
                        existing["last_signal_time"] = sig_time
                        existing["last_gps"] = e.get("gps", 0)
                        existing["last_ib_ratio"] = e.get("ib_ratio", 0)

        # NOTE: Do NOT initialize team_cooldown_polls here.
        # Setting it to 0 puts the key in the dict, and the next poll
        # with GPS >= SIGNAL_COOLDOWN_GPS_FLOOR immediately deletes it
        # (line 5492), making the team instantly re-qualified.
        # Instead, leave the key OUT of team_cooldown_polls so the
        # natural "not yet in cooldown tracking" path (line 5466)
        # starts tracking only when GPS actually drops below 55.

        if signaled_teams:
            log.info(f"v10.44s: Rebuilt signaled_teams for {len(signaled_teams)} team(s) from file (cooldown deferred)")
    except Exception as e:
        log.warning(f"v10.44m: Failed to rebuild signaled_teams: {e}")


# v10.14: Auto-resolution via goal events API
_goal_events_cache: dict[int, list[dict]] = {}  # fixture_id -> list of goal event dicts
_card_events_cache: dict[int, list[dict]] = {}  # v10.80: Card events (yellow/red, who/when) from the SAME /fixtures/events call — zero extra credits

# v10.31: Event-based SOT supplement for late-window lag mitigation
# When /fixtures/statistics lags 2-3 min behind real-time (common 75'-90'),
# we fetch /fixtures/events (near-real-time) and count Shot/OnTarget + Goal
# events per team. effective_sot = max(stats_sot, events_sot).
# Only activated for fixtures with SOT>=2 at 75'+ (meaningful pressure).
# Cache TTL prevents excessive credit burn (1 credit per fetch).
event_extended_fixtures: set[int] = set()  # fixtures approved for 90' monitoring
_event_sot_cache: dict[int, dict] = {}     # {fid: {"ts": float, "sot": {tid: n}, "goals": {tid: n}, "credit_cost": int}}  # v10.56 format
EVENT_SOT_CACHE_TTL = 90              # seconds between re-fetches
EVENT_SOT_MINUTE = 75                 # minute threshold to activate event SOT
EVENT_SOT_MIN_PRESSURE = 2            # minimum best_sot to activate

# --- v10.60: FIELD-AVAILABILITY CENSUS + EVENTS-EXTRAS (LOGGING ONLY) ---
# CENSUS: which stats fields each league's /fixtures/statistics response
# actually contains. Updated from live polls; grows monotonically; one log
# line per new field per league; persisted to FIELD_CENSUS_FILE so restarts
# keep the knowledge. /fields renders it on Telegram.
CENSUS_FIELDS = [
    "gk_saves", "fouls", "offsides", "yellow_cards",
    "total_passes", "big_chances", "expected_goals",
]


def load_field_census() -> None:
    """v10.60: Load the census from /data at startup (missing file = empty)."""
    global _field_census
    try:
        if os.path.exists(FIELD_CENSUS_FILE):
            with open(FIELD_CENSUS_FILE, "r") as f:
                _field_census = {int(k): dict(v) for k, v in json.load(f).items()}
            log.info(f"v10.60: Field census loaded ({len(_field_census)} league(s))")
        else:
            log.info("v10.60: Field census empty — learning which KPIs the API delivers from tonight's polls")
    except Exception as e:
        log.warning(f"v10.60: Field census load failed ({e}) — starting empty")


def _save_field_census() -> None:
    try:
        tmp = FIELD_CENSUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_field_census, f)
        os.replace(tmp, FIELD_CENSUS_FILE)
    except Exception as e:
        log.warning(f"v10.60: Field census save failed: {e}")


def update_field_census(league_id: int, league_name: str, arrivals: dict[str, bool]) -> None:
    """v10.60: Merge per-field arrival flags for one league.

    LOGGING ONLY: reads nothing, changes nothing outside the census file.
    Saves + logs 'NEW-FIELDS' exactly when a league's KNOWN set grows
    (once per new field per league, not once per poll).
    """
    known = _field_census.setdefault(int(league_id), {})
    grew = False
    for field, arrived in arrivals.items():
        if arrived and not known.get(field):
            known[field] = True
            grew = True
    if grew:
        _save_field_census()
        status = " ".join(
            f"{f}={'yes' if known.get(f) else 'no'}" for f in CENSUS_FIELDS if f in known
        )
        log.info(f"  NEW-FIELDS: {status} [{league_name}] (id {league_id}) — census saved")


def format_field_census() -> str:
    """v10.60: Render the census for /fields (Telegram)."""
    if not _field_census:
        return (
            "\U0001f4ca FIELD CENSUS\n\n"
            "No live statistics parsed yet.\n"
            "The census learns which KPI fields the API actually\n"
            "delivers, per league, from live polls after this deploy."
        )
    lines = ["\U0001f4ca FIELD-AVAILABILITY CENSUS (live-learned)", ""]
    for lid in sorted(_field_census):
        fields = _field_census[lid]
        name = LEAGUE_IDS.get(lid, f"league {lid}")
        parts = [f"{f}={'yes' if fields.get(f) else 'no'}" for f in CENSUS_FIELDS if f in fields]
        if not parts:
            continue
        lines.append(f"{name} (id {lid}):")
        for i in range(0, len(parts), 3):
            lines.append("  " + "  ".join(parts[i:i + 3]))
    lines.append("")
    lines.append("'no' = never seen in this league's responses (recorded as null, not 0).")
    return "\n".join(lines)


# EVENTS-EXTRAS: blocked shots / substitutions / cards extracted from event
# responses ALREADY fetched by the fast lane (top-3 fixtures, ~10s) and the
# main loop's 75'+ SOT supplement. ZERO extra credits. Recorded per poll /
# per signal; None whenever no fresh events response covers the fixture —
# honest missingness, never fake zeros.
_event_extras_cache: dict[int, dict] = {}


def _update_event_extras_from_events(fixture_id: int, events: list) -> None:
    """v10.60: Count blocked shots / subs / cards from an events response
    that was already fetched for other purposes. LOGGING ONLY.

    v10.73: RED CARDS — the same response now also yields the red-card
    list (player + minute + kind) per fixture. Straight reds arrive as
    detail 'Red Card', second-yellow sendoffs as 'Second Yellow card'.
    Display + record context only, NEVER a gate. The latest response
    wins (events responses are cumulative full-match lists)."""
    blocked: dict[int, int] = {}
    subst: dict[int, int] = {}
    subst_latest: dict[int, int] = {}
    cards: dict[int, int] = {}
    card_latest: dict[int, int] = {}
    red_cards: list[dict] = []
    injury_subst: dict[int, int] = {}  # v10.74: Subst events labeled 'Injury'
    for ev in events:
        etype = ev.get("type", "")
        detail = ev.get("detail", "")
        tid_ev = (ev.get("team") or {}).get("id")
        if tid_ev is None:
            continue
        ev_min = safe_int(str((ev.get("time") or {}).get("elapsed", 0) or 0))
        if etype == "Shot" and detail == "Blocked":
            blocked[tid_ev] = blocked.get(tid_ev, 0) + 1
        elif etype == "Subst":
            subst[tid_ev] = subst.get(tid_ev, 0) + 1
            if ev_min > subst_latest.get(tid_ev, 0):
                subst_latest[tid_ev] = ev_min
            # v10.74: forced injury substitution — providers that label the
            # reason set detail='Injury' on the Subst event. Display + record
            # only (brain v2 feature); a sub alone never moves the lambda.
            if detail == "Injury":
                injury_subst[tid_ev] = injury_subst.get(tid_ev, 0) + 1
        elif etype == "Card":
            cards[tid_ev] = cards.get(tid_ev, 0) + 1
            if ev_min > card_latest.get(tid_ev, 0):
                card_latest[tid_ev] = ev_min
            # v10.73: red cards with player identity (events already fetched)
            if detail in ("Red Card", "Second Yellow card"):
                red_cards.append({
                    "team_id": tid_ev,
                    "player": ((ev.get("player") or {}).get("name") or "?"),
                    "minute": ev_min,
                    "kind": "red" if detail == "Red Card" else "yellowred",
                })
    _event_extras_cache[fixture_id] = {
        "ts": time.time(),
        "blocked": blocked, "subst": subst, "subst_latest": subst_latest,
        "cards": cards, "card_latest": card_latest,
        "red_cards": red_cards,
        "injury_subst": injury_subst,  # v10.74
    }


def get_event_extras(fixture_id: int) -> dict | None:
    """v10.60: Latest per-team blocked/subs/cards for a fixture, or None
    when no reasonably fresh events response exists for it."""
    x = _event_extras_cache.get(fixture_id)
    if not x or time.time() - x.get("ts", 0) >= EVENT_SOT_CACHE_TTL + 30:
        return None
    return x


# v10.73: how recent a red card must be (in GAME minutes) to carry the
# NEW RED CARD warning in the signal — same spirit as the NEW BIG CHANCE
# freshness line, but event-based and minute-compared (no state needed).
RED_CARD_FRESH_MINUTES = 10


def get_red_card_events(fixture_id: int) -> list[dict] | None:
    """v10.73: Red cards for a fixture (player + minute + kind) from the
    latest fresh events response, or None when NO fresh events response
    covers the fixture (honest missingness — the signal then falls back
    to the team-statistics red-card count)."""
    x = get_event_extras(fixture_id)
    if x is None:
        return None
    return x.get("red_cards") or []


def _build_red_card_block(
    team_id: int, minute: int, rc_events: list[dict] | None,
    stats_fallback_str: str, home, away,
) -> tuple[str, int | None, int | None, list[dict] | None]:
    """v10.73: Build the red-card segment for the signal message.

    Sources: rc_events (fresh events response: player + minute + kind)
    primary; stats_fallback_str (team-statistics counts) when no events
    response covers the fixture. Returns (block_str, team_reds, opp_reds,
    rc_events) — counts are None exactly when events coverage is missing
    (the outcome record keeps honest missingness).

    Display + record context ONLY: never a gate, never GPS input.
    """
    try:
        if rc_events is None:
            # No events coverage for this fixture — stats-based count only
            # (no player names). v10.85: silent when there is nothing to
            # report — a "Red Cards: None" line was pure display noise.
            if stats_fallback_str in ("None", "", "N/A"):
                return "", None, None, None
            return f"\n\U0001f7e5 Red Cards: {stats_fallback_str}", None, None, None
        if not rc_events:
            # events coverage confirms: zero reds — nothing to say
            return "", 0, 0, []
        _id2name = {home[0]: home[1], away[0]: away[1]}
        _parts = []
        _team_reds = 0
        _opp_reds = 0
        _latest_min = 0
        _latest_name = ""
        _latest_player = ""
        _earliest_opp_min = None
        for rc in rc_events:
            _tnm = _id2name.get(rc.get("team_id"), f"team {rc.get('team_id')}")
            _kind = "R" if rc.get("kind") == "red" else "2Y"
            _parts.append(f"{_tnm} - {rc.get('player', '?')} ({_kind} {rc.get('minute', 0)}')")
            _m = rc.get("minute") or 0
            if rc.get("team_id") == team_id:
                _team_reds += 1
            else:
                _opp_reds += 1
                if _earliest_opp_min is None or _m < _earliest_opp_min:
                    _earliest_opp_min = _m
            if _m >= _latest_min:
                _latest_min = _m
                _latest_name = _tnm
                _latest_player = rc.get("player", "?")
        block = "\n\U0001f7e5 Red Cards: " + " | ".join(_parts)
        # NEW RED CARD freshness (<= RED_CARD_FRESH_MINUTES game minutes old)
        if _latest_min > 0 and 0 <= (minute - _latest_min) <= RED_CARD_FRESH_MINUTES:
            block += (
                f"\n\U0001f6a8 NEW RED CARD ({minute - _latest_min}' ago) — "
                f"{_latest_player} off, {_latest_name} down to 10"
            )
        # Man-power context: 10v11 / 11v10 shifts goal probability strongly
        if _opp_reds > _team_reds and _earliest_opp_min is not None:
            block += f"\n\U0001f4aa Man-up: opponent down to 10 (since {_earliest_opp_min}')"
        elif _team_reds > _opp_reds:
            block += "\n\u26a0\ufe0f Man-down: pressing with 10 men"
        return block, _team_reds, _opp_reds, rc_events
    except Exception:
        return "", None, None, None

# v10.44p: Event fast lane — 10s event polling for the single hottest fixture
EVENT_FAST_LANE_INTERVAL = 10         # seconds between event polls
EVENT_FAST_LANE_MIN_GPS = 65          # minimum GPS to activate
EVENT_FAST_LANE_MIN_SOT = 2           # minimum SOT to activate
FAST_LANE_MAX_FIXTURES = 3            # v10.50: poll events for up to 3 hottest fixtures (was 1)

# v10.53: GOAL WATCH — instant goal flash alerts for close late games.
# Rationale: goals cluster. The Levski derby double strike (74' + 75') sent
# no prediction signal because goal 1 had zero buildup (1 SOT all half) —
# no stats model honestly predicts that. But goal 1 IS the earliest reliable
# warning for goal 2. Goal watch reports the FACT of a goal within seconds
# (fast lane ~10s, goal-watch lane ~30s) so the user can act on the cluster
# while it is live. Flashes are ALERTS, not signals: they never touch
# signal_outcomes, win-rate stats, or signal gates. Logged to goal_flash.jsonl.
GOAL_WATCH_INTERVAL = 30              # seconds between goal-watch event polls
GOAL_WATCH_MINUTE = 60                # watch from 60' (late-game cluster zone)
GOAL_WATCH_MAX_FIXTURES = 5           # concurrent goal-watch fixtures (credit control)
GOAL_WATCH_MAX_DIFF = 2               # only close games (|home-away| <= 2)
GOAL_WATCH_CREDIT_CAP = 1500          # daily credit budget for goal-watch polls
GOAL_WATCH_CLUSTER_MIN = 5            # game minutes: 2 goals within this = "cluster"

# v10.54: SURGE WATCH — pre-goal pressure alarms (user request 2026-09-03:
# "I don't want to be notified when a goal is scored — I want to be notified
# about the pressure/SOT buildup BEFORE the goal. Levski had 1 SOT in 73'
# and then erupted: catch the signs BEFORE it happens.").
# Rides the SAME events polls as goal watch (fast lane ~10s for the 3 hottest
# fixtures, watch lane ~30s for up to 5 close games 60'+) -> ZERO extra
# credits. Four layers, earliest first:
#   SHOT-FLOOD — >= SURGE_SHOT_FLOOD shots of ANY kind (off target/blocked
#                included) within SURGE_SHOT_WINDOW after >= 12' with no
#                shots at all (shots usually precede SOT — earliest layer)
#   WAKE-UP    — team's first shot ON TARGET after >= SURGE_SOT_QUIET_MIN
#                game minutes of team-SOT silence ("Levski: 1 SOT in 73'")
#   ESCALATION — 2nd SOT within SURGE_SOT_BURST_WINDOW of a wake-up — the
#                strongest pre-goal sign ("goals come in bursts")
#   SUSTAIN    — v10.55: every FURTHER SOT within SURGE_SOT_BURST_WINDOW of
#                the previous episode SOT also alerts — the episode no
#                longer closes after the 2nd SOT, so "team keeps generating
#                continuous pressure" is fully covered, not just the first
#                two shots (capped by SURGE_MAX_PER_FIXTURE)
# v10.55 goal semantics (user spec 2026-09-03, second round):
#   * a goal-scoring shot is COUNTED as the team's last known shot — future
#     silence is measured from the goal's minute — but it can NEVER open,
#     complete, or trigger an alert: the goal already happened, there is
#     nothing left to warn about;
#   * a goal CLOSES any open episode, so a shot right after it can never
#     piggyback on the pre-goal burst as a false "escalation";
#   * consequence: post-goal pressure needs a FRESH quiet spell (15' SOT
#     silence / 12' shot silence measured from the goal) before it warns
#     again — which is exactly the second-goal early warning wanted.
# Alerts are WARNINGS, never betting signals: they do not touch
# signal_outcomes, win-rate stats, or signal gates. Logged to
# surge_watch.jsonl for later threshold tuning. /surgewatch on|off.
SURGE_MIN_MINUTE = 50            # alerts only from 50' (watch lane covers 60'+)
SURGE_SOT_QUIET_MIN = 15         # game minutes of team-SOT silence -> wake-up
SURGE_SOT_BURST_WINDOW = 10      # 2nd SOT within this of the wake-up -> escalate
SURGE_SHOT_QUIET_MIN = 12        # minutes with ZERO shots before a shot-flood
SURGE_SHOT_WINDOW = 5            # game-minute window for the shot cluster
SURGE_SHOT_FLOOD = 2             # shots inside the window to trigger the flood
SURGE_TEAM_COOLDOWN = 600        # wall seconds — wake-up/flood rate limit per team
SURGE_SOT_GUARD = 60             # wall seconds — flood silenced right after a SOT alert
SURGE_MAX_PER_FIXTURE = 5        # max alerts per fixture per day
SURGE_MAX_PER_DAY = 40           # global daily alert cap (spam safety)
SURGE_STALENESS_TOL = 5          # event minute this far behind fixture minute = stale
SURGE_LATE_BONUS = 2             # v10.67: extra per-fixture alert budget for 86'+ retained fixtures
                                 # (they re-enter the lane only at the end of the game; the base
                                 # 5 cap is usually already burned by then — Vratsa pattern)
# v10.44n set (mirrors fetch_goal_events): VAR-disallowed goals must never flash
_GOAL_DISALLOWED_DETAILS = {
    "Missed Penalty", "Goal Disallowed", "Penalty not awarded",
    "Cancelled Goal", "Disallowed Goal",
}

_event_fast_lane_fids: list[int] = []     # v10.50: fixture IDs currently in fast lane
_event_fast_lane_last: float = 0          # timestamp of last event fast lane poll
_event_fast_lane_credits_today: int = 0  # credit counter (safety cap)
# --- v10.50: fast-lane per-event runtime state (shadow detection) ---
_fl_sot_events: dict[tuple, list[tuple[int, float]]] = {}  # (fid,tid) -> [(minute, wall_ts)]
_fl_seen_sot_count: dict[tuple, int] = {}                  # (fid,tid) -> events counted so far
_fl_goal_minutes: dict[int, list[tuple[int, int]]] = {}    # fid -> [(minute, team_id)]
_fl_seen_goal_count: dict[int, int] = {}                   # fid -> goal events counted so far
# v10.53: GOAL WATCH runtime state (goal flash alerts — facts, not predictions)
_goal_watch_enabled: bool = os.environ.get("GOAL_WATCH_ENABLED", "false").lower() == "true"  # v10.54: default OFF — user wants PRE-goal warnings, not goal reports
_goal_watch_fids: list[int] = []           # fixtures currently in the 30s goal-watch lane
_goal_watch_last: float = 0.0              # last goal-watch poll (wall time)
_goal_watch_credits_today: int = 0         # daily credit counter (cap GOAL_WATCH_CREDIT_CAP)
_goal_watch_credits_date: str | None = None
_goal_watch_seen: dict[int, int] = {}      # fid -> valid goal events counted (shared: fast lane + goal watch)
_goal_watch_flash_keys: set = set()        # (fid, minute, team_id) already flashed
_gw_flashes_today: int = 0                 # daily flash counter
_gw_flashes_date: str | None = None
_coldstart_warmed: set = set()             # v10.53: fids that attempted the events backfill
# v10.54: SURGE WATCH runtime state (pre-goal pressure alarms)
_surge_watch_enabled: bool = os.environ.get("SURGE_WATCH_ENABLED", "true").lower() == "true"
_surge_seen_sot: dict[tuple, int] = {}     # (fid,tid) -> SOT events seen so far (goals incl.)
_surge_seen_shots: dict[tuple, int] = {}   # (fid,tid) -> ALL shot events seen (goals incl.)
_surge_wake_minute: dict[tuple, int] = {}  # (fid,tid) -> wake-up minute of an open episode
_surge_burst_last_minute: dict[tuple, int] = {}  # v10.55: (fid,tid) -> last SOT minute of the open episode
_surge_burst_count: dict[tuple, int] = {}        # v10.55: (fid,tid) -> SOTs in the open episode (wake=1)
_surge_last_alert_ts: dict[tuple, float] = {}  # (fid,tid) -> wall ts of last SENT alert
_surge_flood_minute: dict[tuple, int] = {}     # (fid,tid) -> last shot-flood alert minute
_surge_fixture_alerts: dict[int, int] = {}     # fid -> alerts sent today
_surge_alerts_today: int = 0                    # daily alert counter (cap SURGE_MAX_PER_DAY)
_surge_alerts_date: str | None = None

# v10.67: LATE SURGE — close games stay in the events watch lane to the final whistle.
# Post-mortem (Botev Vratsa 86'/88' vs Septemvri Sofia + Sep 2-4 live data):
# 13% of tracked full-window HIT goals land at 86'+, where the bot is
# structurally BLIND — fixtures leave fast_monitored at the 85' ceiling and
# BOTH event lanes (10s fast lane + 30s watch lane) select from fast_monitored,
# so surge alarms / goal flashes die with monitoring at 86' even though goal
# production itself does not fade (~4-5%/min flat through 85'). Fix: fixtures
# dropped ONLY because of the minute ceiling are retained in _late_retain_fids
# and stay eligible for the 30s watch lane (minute 60-90, |score diff| <=
# GOAL_WATCH_MAX_DIFF re-checked on every poll, top GOAL_WATCH_MAX_FIXTURES
# with late-retained taking pick priority — minutes left, not half-hours)
# until FT. The stats lane is NOT re-entered: no stats polls, no signal
# gates, signal_outcomes untouched — surge stays a warning-only lane.
# Cost: ~10-16 events polls per retained fixture (1 credit each) inside the
# existing GOAL_WATCH_CREDIT_CAP budget. Blows out or finishes? The per-poll
# filters / FT cleanup release the fixture; retention itself costs nothing.
_late_retain_fids: dict[int, float] = {}  # v10.67: fid -> wall ts when retained
LATE_RETAIN_MAX = 20                      # safety cap on the retain set (oldest evicted)


# v10.44r: Goal-triggered priority polling + latency measurement
# When a goal is detected (by discovery OR stats processing), the fixture
# gets 60s of 15s polling regardless of SOT/GPS. This bridges the gap between
# score update (fast, from /fixtures?live=all) and stats update (slower).
# Key insight: goal detection and stats refresh are decoupled. Without this,
# the bot detects the goal but waits for the normal polling cycle to act on it.
GOAL_PRIORITY_WINDOW = 60  # seconds of 15s polling after goal detection
goal_priority_until: dict[int, float] = {}  # fid -> timestamp when priority expires

# v10.44r: Goal detection latency tracking (0 extra credits)
# Measures the pipeline: actual goal -> discovery detects -> stats poll -> signal sent.
# goal_detect_ts: wall-clock time when discovery first saw the score change
# goal_game_minute: game minute at detection time (from fixture.elapsed)
# goal_stats_ts: wall-clock time of first stats poll after goal was detected
_goal_detect_ts: dict[int, float] = {}     # fid -> discovery timestamp
_goal_game_minute: dict[int, int] = {}     # fid -> game minute at detection
_goal_stats_ts: dict[int, float] = {}      # fid -> first stats poll timestamp after goal
_goal_stats_recorded: set[int] = set()    # fids where stats_ts has been recorded
_last_untracked_live_count: int = 0  # v10.44s: untracked live fixtures from last discovery
_untracked_retry_count: int = 0      # v10.44s: consecutive retries with untracked > 0


def fetch_goal_events(client: httpx.Client, fixture_id: int) -> list[dict]:
    """v10.14: Fetch goal events for a finished fixture.

    Uses /fixtures/events?fixture=ID endpoint.
    Returns list of goal events with team_id, minute, and type info.
    Cached per fixture_id to avoid duplicate API calls.
    Each call costs 1 API credit.
    """
    if fixture_id in _goal_events_cache:
        return _goal_events_cache[fixture_id]

    try:
        data = api_get(client, "/fixtures/events", {"fixture": fixture_id})
        events = data.get("response", [])
        # Filter only VALID goal events (exclude subs, cards, VAR, etc.)
        # v10.44n: Also exclude disallowed goals ("Missed Penalty", "Goal Disallowed",
        # "Penalty not awarded" etc.) — these appear as type=="Goal" in API-Football
        # but were overturned by VAR. They inflate win rate if counted.
        _disallowed_details = {
            "Missed Penalty", "Goal Disallowed", "Penalty not awarded",
            "Cancelled Goal", "Disallowed Goal",
        }
        goal_events = []
        card_events = []  # v10.80: bookings (yellow=1, red=1) — how card O/U lines settle
        for ev in events:
            # v10.80: Card events captured from the same payload (who/when)
            if ev.get("type") == "Card":
                card_events.append({
                    "team_id": ev.get("team", {}).get("id"),
                    "minute": safe_int(str(ev.get("time", {}).get("elapsed", 0) or 0)),
                    "minute_extra": safe_int(str(ev.get("time", {}).get("extra", 0) or 0)),
                    "detail": ev.get("detail", ""),
                    "player": ev.get("player", {}).get("name", ""),
                })
                continue
            if ev.get("type") == "Goal":
                detail = ev.get("detail", "")
                if detail in _disallowed_details:
                    log.info(
                        f"  GOAL EVENTS: skipping disallowed goal at "
                        f"{ev.get('time', {}).get('elapsed', '?')}' by "
                        f"{ev.get('player', {}).get('name', '?')} (detail: {detail})"
                    )
                    continue
                goal_events.append({
                    "team_id": ev.get("team", {}).get("id"),
                    "minute": safe_int(str(ev.get("time", {}).get("elapsed", 0) or 0)),
                    "minute_extra": safe_int(str(ev.get("time", {}).get("extra", 0) or 0)),
                    "detail": detail,
                    "player": ev.get("player", {}).get("name", ""),
                })
        _goal_events_cache[fixture_id] = goal_events
        _card_events_cache[fixture_id] = card_events  # v10.80
        log.info(f"  GOAL EVENTS: fixture {fixture_id} has {len(goal_events)} valid goal(s)")
        return goal_events
    except Exception as e:
        log.warning(f"  Failed to fetch goal events for fixture {fixture_id}: {e}")
        return []


# v10.44n: Cache for per-player SOT data keyed by fixture_id.
# v10.57/v10.58: value maps team_id -> [(name, sot, total_shots, scored), ...]
_player_sot_cache: dict[int, dict[int, list[tuple[str, int, int, bool]]]] = {}
_latest_sot_event_minute: dict[int, int] = {}  # v10.44p: fixture_id -> latest SOT event game minute
# v10.57: Top-SOT goal refresh — the player who scored leaves the line.
# _player_sot_cache_goals: valid-goal count in the events feed WHEN the cache
#   was built. _fixture_valid_goals: latest count seen by the event lanes.
#   Cache is stale (must rebuild) when latest > build-time count.
_player_sot_cache_goals: dict[int, int] = {}
_fixture_valid_goals: dict[int, int] = {}
# v10.58: stats-side SOT snapshot when the cache was built — {fid: {tid: sot}}.
# When the team's SOT has grown past this, new shooters may exist: the cache
# is rebuilt on the next signal (1 credit) so fresh names reach the Top SOT
# line. Only stats-source values are stored (never events sums) so the
# growth comparison is always same-source.
_player_sot_cache_built_sot: dict[int, dict[int, int]] = {}
# v10.58: seconds to wait before the ONE events-feed lag retry in
# fetch_top_sot_players (runs AFTER the signal is already in Telegram).
PLAYER_FEED_RETRY_DELAY = 3
# v10.63: deferred Top-SOT line recovery. The events feed can lag the
# statistics endpoint by minutes, and the 3-second retry above cannot
# bridge that: on feed-lag signals the feed often listed ONLY the goal
# scorer, so the never-show-a-scorer rule (v10.58) produced an empty line
# and the "scores next" hint vanished (Sparta/PEC post-mortem: stats knew
# SOT=3, the feed listed 1 event). A deferred retry re-fetches AFTER the
# feed catches up and sends the line late (display-only, never gates,
# never blocks the signal loop).
TOP_SOT_RETRY_DELAY = 60          # seconds after the failed line before retry #1
TOP_SOT_RETRY_MAX_ATTEMPTS = 3    # per (fixture, team) — v10.65: 2->3 (the Sep 4
                                  # GAVE-UPs were 2-attempt losses; feeds that
                                  # catch up in 1-3 min now get their line)
TOP_SOT_RETRY_CREDIT_CAP = 60     # recovery fetches per day (1 credit each)
_top_sot_retry_queue: dict[tuple[int, int], dict] = {}  # (fid,tid) -> pending recovery
_top_sot_retry_credits_today: int = 0
_top_sot_retry_date: str | None = None
# v10.65: per-league SHOT-EVENT FEED CENSUS. The Sep 4 post-mortem
# (Sparta/PEC, Porto 2/6-listed, Real Betis 0/6-listed) proved the events
# feed's per-player Shot coverage is a per-league reality, not just lag.
# The census counts what each signal-time fetch and each recovery actually
# found, per league, so "no player data" becomes a LEARNED verdict instead
# of a recurring surprise — and leagues that never deliver shot events stop
# burning recovery credits. Persisted to sot_feed_census.json; /sotfeed.
SOT_FEED_CENSUS_FILE = os.path.join(_VOLUME_DIR, "sot_feed_census.json")
_sot_feed_census: dict[int, dict] = {}   # league_id -> counters (see _update_sot_feed_census)
# v10.65: outcome of the LAST fetch_top_sot_players() call — read by the
# signal path to embed the right Top-SOT note in the signal and to feed
# the census. {"outcome": ok|no_shooters|all_scored|error, "listed_sot": n}
_last_top_sot_info: dict = {}
# v10.73: PLAYERS-STATS FALLBACK for the Top-SOT line. The events feed
# itemizes per-player Shot events for only a minority of leagues at
# signal time (the Sep 1-5 outcomes audit: Serie A 0/16, Bundesliga 0/14,
# Eredivisie 0/9, Premier League 1/10, Ligue 1 1/13 signals carried the
# player line — the empty-line problem is coverage, not just lag).
# /fixtures/players delivers per-player shot/goal STATISTICS for those
# same leagues from the stats pipeline (same latency as team stats), so
# when the events line is empty and the feed is behind stats, ONE call
# resolves the 'scores next' names. 1 credit, daily-capped, per-league
# censused (never burn credits on a league that never delivers).
PLAYERS_SOT_CREDIT_CAP = 60      # players-API fetches per day (1 credit each)
PLAYERS_CENSUS_MIN_FETCHES = 6   # fetches before a league can be learned NO-DATA
PLAYERS_FEED_CENSUS_FILE = os.path.join(_VOLUME_DIR, "players_feed_census.json")
_players_feed_census: dict[int, dict] = {}   # league_id -> counters
_players_sot_credits_today: int = 0
_players_sot_date: str | None = None


def _note_fixture_goals(fid: int, n_goals: int) -> None:
    """v10.57: Note the latest valid-goal count for a fixture (event lanes).

    A goal changes who should headline the Top SOT line: the player who
    scored already scored — the line is a "scores NEXT" hint, so he must
    demote. When the count grows past what the player cache was built with,
    drop that cache; the next fetch_top_sot_players() call (after the next
    signal) rebuilds from fresh events with up-to-date scorer flags.
    Zero API cost — the rebuild only happens when a signal is sent anyway.
    """
    _cur = _fixture_valid_goals.get(fid, -1)
    if n_goals > _cur:
        _fixture_valid_goals[fid] = n_goals
        if fid in _player_sot_cache and n_goals > _player_sot_cache_goals.get(fid, 0):
            del _player_sot_cache[fid]
            _player_sot_cache_built_sot.pop(fid, None)  # v10.58
            log.info(
                f"  v10.57 TOP-SOT REFRESH: F{fid} — valid goals now {n_goals}; "
                f"player cache dropped (scorer demotes from the Top SOT line)"
            )


def goal_race_mute(feed_goals, sh, sa) -> bool:
    """v10.87: True when the events feed knows more valid goals than the
    scoreline the signal was built on — the feed-ahead-of-score race.

    PSV-Shakhtar Sep 10 (45'): the signal was gated on a 0-0 stats batch
    while the events feed already knew the 45' goal; the message (and its
    prices) shipped fiction the user could not bet. The pre-send Top-SOT
    fetch counts the feed's valid goals (own goals in, disallowed out);
    when that count exceeds the message scoreline, the goal is already in
    the books. The REVERSE direction (score ahead, feed behind — the
    Fenerbahce 49' class) never trips this: feed_goals <= scoreline, and
    the stats-lane post-goal gates already own that case.
    """
    if feed_goals is None:
        return False
    try:
        return int(feed_goals) > ((sh or 0) + (sa or 0))
    except (TypeError, ValueError):
        return False


def _filter_non_scorers(entries: list[tuple[str, int, int, bool]]) -> list[tuple[str, int, int]]:
    """v10.57/v10.58: The Top SOT line shows players who have NOT scored yet.

    Input entries are (name, sot, total_shots, scored), sorted non-scorers
    first (SOT desc), then scorers. Returns (name, sot, total_shots):

      1. non-scorers with SOT > 0 — the "scores next" candidates (primary);
      2. v10.58 fallback: when every SOT taker already scored, non-scorers
         by TOTAL shots — players shooting without converting are the
         next-best "scores next" hint (display: "(n shots)");
      3. when every shooter has scored: [] — a scorer is NEVER shown
         (user rule: the line must not point at someone who already scored).
    """
    _non_sot = [(n, s, t) for n, s, t, _scored in entries if not _scored and s > 0]
    if _non_sot:
        return _non_sot
    _non_shots = [(n, s, t) for n, s, t, _scored in entries if not _scored and t > 0]
    _non_shots.sort(key=lambda x: (-x[2], x[0]))
    return _non_shots


def _log_top_sot_skip(
    fixture_id: int, team_id: int, where: str,
    entries: list[tuple[str, int, int, bool]] | None, team_sot_now: int | None,
) -> None:
    """v10.62: The v10.58 silence rule (never show a scorer) can make the
    Top SOT line disappear with zero trace in the logs. Log WHY it was
    skipped: feed empty, or every listed shooter already scored. This is
    observability only — no behavior change."""
    try:
        _n = len(entries) if entries else 0
        if _n == 0:
            _reason = "events feed lists no shooters for this team yet"
        else:
            _scored = sum(1 for e in entries if e[3])
            _reason = f"every listed shooter already scored ({_scored}/{_n})"
        log.info(
            f"  v10.62 TOP-SOT LINE SKIPPED: F{fixture_id} T{team_id} "
            f"({where}) — {_reason}"
            + (f" | stats SOT={team_sot_now}" if team_sot_now is not None else "")
        )
    except Exception:
        pass


# ============================================================
# v10.65: SHOT-EVENT FEED CENSUS (per league, live-learned)
# ============================================================

def load_sot_feed_census() -> None:
    """v10.65: Load the per-league shot-event feed census at startup."""
    global _sot_feed_census
    try:
        if os.path.exists(SOT_FEED_CENSUS_FILE):
            with open(SOT_FEED_CENSUS_FILE, "r") as f:
                _sot_feed_census = {int(k): dict(v) for k, v in json.load(f).items()}
            log.info(f"v10.65: Shot-event feed census loaded ({len(_sot_feed_census)} league(s))")
    except Exception as e:
        log.warning(f"v10.65: sot_feed_census.json unreadable ({e}) — starting fresh")
        _sot_feed_census = {}


def _save_sot_feed_census() -> None:
    try:
        tmp = SOT_FEED_CENSUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_sot_feed_census, f, indent=1)
        os.replace(tmp, SOT_FEED_CENSUS_FILE)
    except Exception as e:
        log.warning(f"v10.65: failed to save sot_feed_census.json: {e}")


def _update_sot_feed_census(
    league_id: int | None, league_name: str,
    outcome: str, listed_sot: int = 0, stats_sot: int = 0,
) -> None:
    """v10.65: Count one Top-SOT fetch outcome for a league.

    outcome: "ok" (line delivered) | "no_shooters" (feed listed nobody)
             | "all_scored" (everyone listed already scored)
             | "recovered" / "gave_up" (deferred recovery result)
    listed_sot / stats_sot: events-side vs stats-side SOT at fetch time
    (the ratio is the coverage measure; mid-game lag lowers it, so the
    verdict leans on lines/recovered/gave_up, not the ratio alone).
    """
    if league_id is None:
        return
    try:
        c = _sot_feed_census.setdefault(int(league_id), {})
        c["name"] = league_name
        if outcome in ("ok", "no_shooters", "all_scored"):
            c["fetches"] = c.get("fetches", 0) + 1
            if outcome == "ok":
                c["lines"] = c.get("lines", 0) + 1
            else:
                c[outcome] = c.get(outcome, 0) + 1
        else:
            c[outcome] = c.get(outcome, 0) + 1
        if listed_sot:
            c["listed_sot"] = c.get("listed_sot", 0) + int(listed_sot)
        if stats_sot:
            c["stats_sot"] = c.get("stats_sot", 0) + int(stats_sot)
        _save_sot_feed_census()
    except Exception:
        pass


def _sot_census_says_no_data(league_id: int | None) -> bool:
    """v10.65: True when a league is LEARNED to (almost) never deliver
    per-player Shot events: enough signal-time fetches, zero lines ever,
    zero recoveries ever. Recovery queueing is skipped for such leagues
    (the follow-up would burn credits on a feed that never catches up)."""
    if league_id is None:
        return False
    c = _sot_feed_census.get(int(league_id))
    if not c:
        return False
    return (
        c.get("fetches", 0) >= 8
        and c.get("lines", 0) == 0
        and c.get("recovered", 0) == 0
    )


def format_sot_feed_census() -> str:
    """v10.65: Render the shot-event feed census for /sotfeed (Telegram)."""
    if not _sot_feed_census:
        return (
            "\U0001f3af SHOT-EVENT FEED CENSUS\n\n"
            "No Top-SOT fetches recorded yet.\n"
            "The census learns, per league, whether the events feed\n"
            "delivers per-player Shot events at signal time."
        )
    lines = ["\U0001f3af SHOT-EVENT FEED CENSUS (live-learned)", ""]
    rows = sorted(
        _sot_feed_census.items(),
        key=lambda kv: (-kv[1].get("fetches", 0), kv[1].get("name", "?")),
    )
    for lid, c in rows:
        f_n = c.get("fetches", 0)
        l_n = c.get("lines", 0)
        rec = c.get("recovered", 0)
        gave = c.get("gave_up", 0)
        if f_n >= 8 and l_n == 0 and rec == 0:
            verdict = "\u274c NO DATA (never delivers)"
        elif gave > rec and gave >= 2:
            verdict = "\u23f3 LAGGING (recoveries often give up)"
        elif l_n == 0 and rec > 0:
            verdict = "\u23f1 SLOW (only recovers late)"
        else:
            verdict = "\u2705 OK"
        ls = c.get("listed_sot", 0)
        ss = c.get("stats_sot", 0)
        cov = f"{ls}/{ss}" if ss else "-"
        lines.append(
            f"{c.get('name', '?')} (id {lid})\n"
            f"  fetches {f_n} | line {l_n} | rec {rec} | gave-up {gave} | "
            f"listed SOT {cov} — {verdict}"
        )
    lines.append("")
    lines.append(
        "NO DATA leagues: signal note says 'player data unavailable'\n"
        "and recovery retries are skipped (no credits burned)."
    )
    # v10.73: players-stats fallback coverage (same command, second feed)
    if _players_feed_census:
        lines.append("")
        lines.append("PLAYERS-STATS FEED (Top-SOT fallback, v10.73):")
        for lid, c in sorted(
            _players_feed_census.items(),
            key=lambda kv: (-kv[1].get("fetches", 0), kv[1].get("name", "?")),
        )[:15]:
            _f = c.get("fetches", 0)
            _l = c.get("lines", 0)
            if _f >= PLAYERS_CENSUS_MIN_FETCHES and _l == 0:
                _verdict = "\u274c NO DATA (fallback off for this league)"
            elif _l > 0:
                _verdict = "\u2705 DELIVERS the names"
            else:
                _verdict = "\u2026 learning"
            lines.append(
                f"  {c.get('name', '?')} (id {lid}): fetches {_f} | "
                f"lines {_l} — {_verdict}"
            )
    return "\n".join(lines[:60])


def load_players_feed_census() -> None:
    """v10.73: Load the per-league players-stats feed census at startup."""
    global _players_feed_census
    try:
        if os.path.exists(PLAYERS_FEED_CENSUS_FILE):
            with open(PLAYERS_FEED_CENSUS_FILE, "r") as f:
                _players_feed_census = {int(k): dict(v) for k, v in json.load(f).items()}
            log.info(f"v10.73: Players-feed census loaded ({len(_players_feed_census)} league(s))")
    except Exception as e:
        log.warning(f"v10.73: players_feed_census.json unreadable ({e}) — starting fresh")
        _players_feed_census = {}


def _save_players_feed_census() -> None:
    try:
        tmp = PLAYERS_FEED_CENSUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_players_feed_census, f, indent=1)
        os.replace(tmp, PLAYERS_FEED_CENSUS_FILE)
    except Exception as e:
        log.warning(f"v10.73: failed to save players_feed_census.json: {e}")


def _players_census_says_no_data(league_id: int | None) -> bool:
    """v10.73: True when a league is LEARNED to never deliver per-player
    statistics from /fixtures/players (enough fetches, zero lines ever)."""
    if league_id is None:
        return False
    c = _players_feed_census.get(int(league_id))
    if not c:
        return False
    return (
        c.get("fetches", 0) >= PLAYERS_CENSUS_MIN_FETCHES
        and c.get("lines", 0) == 0
    )


def _update_players_feed_census(
    league_id: int | None, league_name: str,
    delivered_players: int, listed_sot: int = 0,
) -> None:
    """v10.73: Count one players-API fallback outcome for a league."""
    if league_id is None:
        return
    try:
        c = _players_feed_census.setdefault(int(league_id), {})
        c["name"] = league_name
        c["fetches"] = c.get("fetches", 0) + 1
        if delivered_players > 0:
            c["lines"] = c.get("lines", 0) + 1
        if listed_sot:
            c["listed_sot"] = c.get("listed_sot", 0) + int(listed_sot)
        _save_players_feed_census()
    except Exception:
        pass


def _players_sot_budget_ok() -> bool:
    """v10.73: Daily budget guard for the players-API fallback (with its
    own date rollover, mirroring the Top-SOT retry credit guard)."""
    global _players_sot_credits_today, _players_sot_date
    _today = time.strftime("%Y-%m-%d")
    if _players_sot_date != _today:
        _players_sot_date = _today
        _players_sot_credits_today = 0
    return _players_sot_credits_today < PLAYERS_SOT_CREDIT_CAP


def fetch_top_sot_players_from_players_api(
    client: httpx.Client, fixture_id: int, team_id: int,
    max_players: int = 3, team_sot_now: int | None = None,
    league_id: int | None = None, league_name: str = "?",
) -> list[tuple[str, int, int]]:
    """v10.73: Top-SOT line from /fixtures/players (the STATS pipeline).

    Why: the events feed's per-player Shot itemization is a per-league
    coverage property, and for most big leagues (Serie A, Bundesliga,
    Premier League, Eredivisie, Ligue 1 — see the Sep 1-5 audit in the
    constants block) it is EMPTY at signal time. The players-statistics
    endpoint delivers shots.on / shots.total / goals.total per player
    from the same pipeline as the team stats the signal already trusts.

    Behavior:
      - ONE api_get call, 1 credit, guarded by the daily
        PLAYERS_SOT_CREDIT_CAP and the per-league census
        (_players_census_says_no_data leagues are never called again).
      - Parses BOTH teams; entries are the same 4-tuples the events
        pipeline uses (name, sot, total_shots, scored) so the existing
        never-a-scorer ladder (_filter_non_scorers) applies unchanged.
      - Merges into _player_sot_cache (keeps events entries for teams
        where the events feed DID itemize) + sets the growth snapshots
        from the players pipeline (stats-side source, comparable with
        the stats lane) + lifts the cache goal-count guard so the
        score-change invalidation still works.
      - Returns the team's (name, sot, shots) list, possibly empty.

    Display-only (the 'scores next' hint); never a gate.
    """
    global _players_sot_credits_today
    # Guards: census-blocked leagues and the daily budget never call.
    if league_id is not None and _players_census_says_no_data(league_id):
        return []
    if not _players_sot_budget_ok():
        return []
    try:
        data = api_get(client, "/fixtures/players", {"fixture": fixture_id})
        _players_sot_credits_today += 1
        response = data.get("response", [])
        if not isinstance(response, list) or not response:
            _update_players_feed_census(league_id, league_name, 0)
            return []
        _pdata: dict[int, list[tuple[str, int, int, bool]]] = {}
        _psot_sum: dict[int, int] = {}
        _pgoals_total = 0
        for blk in response:
            t = blk.get("team") or {}
            tid_blk = t.get("id")
            if tid_blk is None:
                continue
            entries = []
            for pl in blk.get("players") or []:
                nm = (pl.get("player") or {}).get("name") or ""
                if not nm:
                    continue
                st_arr = pl.get("statistics") or []
                st = st_arr[0] if st_arr else {}
                _shots = st.get("shots") or {}
                _goals = st.get("goals") or {}
                _sot_on = safe_int(str(_shots.get("on", 0) or 0))
                _shots_tot = safe_int(str(_shots.get("total", 0) or 0))
                _goals_tot = safe_int(str(_goals.get("total", 0) or 0))
                if _shots_tot <= 0 and _sot_on <= 0:
                    continue  # no shot activity — not a 'scores next' candidate
                entries.append((nm, _sot_on, _shots_tot, _goals_tot > 0))
                _pgoals_total += _goals_tot
            entries.sort(key=lambda x: (x[3], -x[1], -x[2], x[0]))
            _pdata[tid_blk] = entries
            _psot_sum[tid_blk] = sum(e[1] for e in entries)
        _team_entries = _pdata.get(team_id, [])
        _update_players_feed_census(
            league_id, league_name, len(_team_entries), _psot_sum.get(team_id, 0)
        )
        if not _pdata:
            return []
        # Merge into the per-fixture player cache: keep events entries for
        # teams where the events feed itemized shots (events ordering is
        # equivalent), take the players entries for the teams it did not.
        _merged: dict[int, list[tuple[str, int, int, bool]]] = {}
        for t2, entries in _pdata.items():
            _merged[t2] = entries
        for t2, ev_entries in (_player_sot_cache.get(fixture_id) or {}).items():
            if ev_entries and t2 not in _merged:
                _merged[t2] = ev_entries
        _player_sot_cache[fixture_id] = _merged
        # Goal-count guard: the cache must know at least as many goals as
        # the event lanes have seen, or the score-change invalidation fires
        # spuriously. Take the max of both pipelines' counts.
        _player_sot_cache_goals[fixture_id] = max(
            _player_sot_cache_goals.get(fixture_id, 0), _pgoals_total
        )
        # Growth snapshots from the players pipeline for BOTH teams
        # (stats-side source — comparable with the stats lane counts).
        for t2, _s_sum in _psot_sum.items():
            _prev = _player_sot_cache_built_sot.get(fixture_id, {}).get(t2) or 0
            _player_sot_cache_built_sot.setdefault(fixture_id, {})[t2] = max(_s_sum, _prev)
        if team_sot_now is not None:
            _prev = _player_sot_cache_built_sot.get(fixture_id, {}).get(team_id) or 0
            _player_sot_cache_built_sot.setdefault(fixture_id, {})[team_id] = max(
                team_sot_now, _prev
            )
        return _filter_non_scorers(_team_entries)[:max_players]
    except Exception as e:
        # A failed fetch does NOT count against the census (transport
        # errors are not coverage evidence) but the credit was spent.
        log.warning(f"  v10.73 players-API fallback failed for F{fixture_id}: {e}")
        return []


def _build_top_sot_segment(
    tname: str, players: list[tuple[str, int, int]] | None,
    league_id: int | None, fid: int | None = None, tid: int | None = None,
) -> str:
    """v10.65: Build the Top-SOT line EMBEDDED in the signal message.

    User rule (v10.65): the "scores next" hint must be visible INSIDE the
    signal itself, and an unavailable line must SAY so instead of going
    silent. Reads _last_top_sot_info (set by the fetch that just ran):

      - players found  -> "🎯 Top SOT — team: Player A (2), Player B (1)"
      - recovery queued (feed BEHIND stats — unlisted shooters exist)
                      -> "🎯 Top SOT: no player data yet (feed lag —
                           recovery pending)" — the scorer-only listing
                           that produced the empty line is a lag artifact,
                           so "every shooter scored" would mislead; or,
                           when the census has learned the league never
                           delivers shot events,
                           "🎯 Top SOT: player data unavailable for this league"
      - all scored (no queue, feed current)
                      -> "🎯 Top SOT: every listed shooter already scored"
      - unknown/error  -> "" (fetch failed; the deferred recovery follow-up
                           is the remaining path and says it when it lands)
    """
    try:
        if players:
            _parts = [
                f"{_n} ({_c})" if _c > 0 else f"{_n} ({_t} shots)"
                for _n, _c, _t in players
            ]
            _any = any(_c > 0 for _, _c, _ in players)
            _head = (
                "\U0001f3af Top SOT" if _any else "\U0001f3af Top shooters (no SOT yet)"
            )
            return f"\n{_head} — {tname}: {', '.join(_parts)}"
        # v10.65: a queued recovery means the feed is BEHIND the stats
        # (unlisted shooters exist) — the user-facing reason is lag.
        if fid is not None and (fid, tid) in _top_sot_retry_queue:
            if _sot_census_says_no_data(league_id):
                return "\n\U0001f3af Top SOT: player data unavailable for this league"
            return (
                "\n\U0001f3af Top SOT: no player data yet "
                "(feed lag — recovery pending)"
            )
        _o = _last_top_sot_info.get("outcome")
        if _o == "no_shooters":
            if _sot_census_says_no_data(league_id):
                return "\n\U0001f3af Top SOT: player data unavailable for this league"
            return (
                "\n\U0001f3af Top SOT: no player data yet "
                "(feed lag — recovery pending)"
            )
        if _o == "all_scored":
            # v10.70 (user rule, FINAL): the Top-SOT line only ever names
            # players who have NOT scored yet — never a scorer, no
            # "(scored)" tags. When every listed shooter already scored
            # the line says exactly that (v10.65 wording).
            return "\n\U0001f3af Top SOT: every listed shooter already scored"
        return ""
    except Exception:
        return ""


def fetch_top_sot_players(
    client: httpx.Client, fixture_id: int, team_id: int, max_players: int = 3,
    team_sot_now: int | None = None, pre_signal: bool = False,
    league_id: int | None = None,
) -> list[tuple[str, int, int]]:
    """v10.44n: Get top SOT players for a team from /fixtures/events.

    v10.46 ordering: non-goal-scorers by SOT desc FIRST, then goal-scorers
    by SOT desc. A player who already scored is less likely to be the NEXT
    scorer — unconverted SOT is the better "scores next" predictor.
    Goals still count toward a player's SOT total (a goal IS a shot on target).

    v10.57: the RETURNED list contains ONLY players who have not scored
    (up to max_players) — the top shown player is always a non-scorer when
    one exists. The per-fixture cache is dropped the moment a goal is seen
    (event lanes ~10-30s) or detected via score change (stats lane), so a
    player who just scored immediately leaves the line: the next signal
    headlines the second-best SOT player who has NOT scored.

    v10.58 (user spec: "must be the second top SOT player which has NOT
    scored a goal"):
      - NEVER show a scorer. When every SOT taker has already scored, the
        line falls back to the top NON-SCORERS by total shots ("(n shots)");
        when every shooter has scored, NO line is sent at all.
      - Feed-lag retry: the events feed can lag behind the statistics
        endpoint. When stats know SOT the events side hasn't listed yet AND
        the line would be empty, wait PLAYER_FEED_RETRY_DELAY seconds and
        re-fetch ONCE (+1 credit, only on signal sends, always AFTER the
        signal is already in Telegram — zero latency cost).
      - SOT-growth refresh: when the team's SOT grew since the cache was
        built (team_sot_now vs the snapshot), the cache is rebuilt so fresh
        shooters appear on the next line.

    v10.62 (phantom-goal post-mortem, Viborg 42' 1-1 -> 0-1 revert):
      - A disallowed/missed-penalty "Goal" event (the API reports these as
        type==Goal with detail Missed Penalty / Goal Disallowed / etc.) no
        longer flags its taker as a scorer and no longer counts as a SOT.
        Before, a phantom goal HID its taker from the line — exactly the
        player most likely to score next (a missed-penalty taker), and the
        events-vs-stats SOT mismatch (4 vs 3) churned the cache.
      - Whenever the line is silently skipped, a log line says WHY (feed
        empty / every listed shooter scored) — silence was invisible.

    v10.63 (Sparta/PEC post-mortem — the missing "scores next" line):
      - CACHE-LAG FALLTHROUGH: when the cache would serve an EMPTY line but
        stats know more SOT than the cached feed did, the cache is
        refreshed from a fresh fetch instead of returning silence built
        from stale data.
      - Cross-team SOT snapshots: the growth guard now covers BOTH teams
        (snapshots from team_gps_history), not just the signaling team —
        the second team's line was previously served from the other
        team's fetch-time data with no staleness check at all.
      - DEFERRED RECOVERY: when the line is still empty after the 3s retry
        WITH feed-lag evidence (stats SOT > events SOT), or the fetch
        itself failed, a queued retry runs ~45s later on the main loop,
        re-fetches (max 2 attempts, credit-capped) and sends the line
        late. True silence (every shooter scored, feed not lagging)
        never retries.

    v10.65 (Porto/Betis post-mortem — the line must be IN the signal):
      - pre_signal=True: called BEFORE send_telegram so the line is
        embedded in the signal itself; the inline 3s feed-lag retry is
        SKIPPED (the signal leaves now; the deferred recovery owns late
        feeds and sends the follow-up).
      - league_id: feeds the per-league shot-event census; leagues
        LEARNED to never deliver Shot events stop queueing recoveries.
      - _last_top_sot_info records WHY the line is empty (no_shooters /
        all_scored / error) so the signal can say it instead of going
        silent.

    Returns list of (player_name, sot_count, total_shots), up to max_players.
    Cached per fixture+team to avoid duplicate API calls; the cache is
    re-fetched after every goal (ordering depends on who scored) and after
    SOT growth (new shooters exist).
    Each rebuild costs 1 API credit — only on signal sends.
    """
    # v10.65: record WHY the line is empty for the signal path + census.
    global _last_top_sot_info
    _last_top_sot_info = {"outcome": "error", "listed_sot": 0}
    # Check cache — v10.57: serve it ONLY if no goal landed since it was
    # built (a goal changes the scorer flags and the ordering).
    # v10.58: AND only if the team's SOT has not grown since the build
    # (growth = new shooters the cache has never heard of).
    _cached = _player_sot_cache.get(fixture_id, {})
    _goals_ok = (
        _fixture_valid_goals.get(fixture_id, 0) <= _player_sot_cache_goals.get(fixture_id, 0)
    )
    _growth_ok = True
    if team_sot_now is not None:
        _built_sot = _player_sot_cache_built_sot.get(fixture_id, {}).get(team_id)
        if _built_sot is not None and team_sot_now > _built_sot:
            _growth_ok = False
    if team_id in _cached and _goals_ok and _growth_ok:
        _r = _filter_non_scorers(_cached[team_id])[:max_players]
        _ev_sot_cached = sum(e[1] for e in _cached[team_id])
        if _r:
            # v10.63: a later signal already delivered the line — cancel
            # any pending deferred recovery for this team.
            _top_sot_retry_queue.pop((fixture_id, team_id), None)
            _last_top_sot_info = {
                "outcome": "ok", "listed_sot": _ev_sot_cached,
                "feed_goals": _fixture_valid_goals.get(fixture_id, 0),  # v10.87
            }
            return _r
        if team_sot_now is None or team_sot_now <= _ev_sot_cached:
            # v10.58 true silence: every listed shooter already scored and
            # the feed is NOT lagging the stats (no unlisted new SOT).
            _log_top_sot_skip(fixture_id, team_id, "cache: every listed shooter already scored",
                              _cached.get(team_id, []), team_sot_now)
            _last_top_sot_info = {
                "outcome": "all_scored" if _cached.get(team_id) else "no_shooters",
                "listed_sot": _ev_sot_cached,
                "feed_goals": _fixture_valid_goals.get(fixture_id, 0),  # v10.87
            }
            return _r
        # v10.63 CACHE-LAG FALLTHROUGH: the cached feed knew fewer SOT than
        # stats. Before, this returned an empty line from stale data with
        # zero recovery (the fixture's second team had no growth snapshot —
        # PEC's 37' line was served Sparta-time entries). Fall through to a
        # fresh fetch instead.
        log.info(
            f"  v10.63 TOP-SOT CACHE-LAG: F{fixture_id} T{team_id} — cache "
            f"lists {_ev_sot_cached} events-SOT vs stats SOT={team_sot_now}; "
            f"refreshing from a fresh feed"
        )
    # (v10.63: fall through to the fresh fetch below)

    def _parse_feed() -> tuple[dict[int, list[tuple[str, int, int, bool]]], int, int]:
        """One events fetch -> (per-team player entries, goal count, latest SOT minute)."""
        data = api_get(client, "/fixtures/events", {"fixture": fixture_id})
        events = data.get("response", [])
        # v10.73: this fresh events response is also the (free) source for
        # red-card context — same parse as the fast lane, zero extra credits.
        # Keeps red cards fresh at signal time even for fixtures outside the
        # fast lane and below 75'.
        _update_event_extras_from_events(fixture_id, events)
        # Per (team, player): SOT count, total shots, goals (v10.46 + v10.58)
        _sot_by_team_player: dict[int, dict[str, int]] = {}
        _shots_by_team_player: dict[int, dict[str, int]] = {}
        _goals_by_team_player: dict[int, dict[str, int]] = {}
        for ev in events:
            etype = ev.get("type", "")
            detail = ev.get("detail", "")
            ev_tid = ev.get("team", {}).get("id")
            pname = ev.get("player", {}).get("name", "")
            if not ev_tid or not pname:
                continue
            # v10.62: disallowed/missed-pen "Goal" events are phantoms — they
            # must not flag a scorer, must not count as SOT (the stats lane
            # never counted them: Viborg events-SOT=4 vs stats-SOT=3), and
            # must not feed the latency tracker. They DO count as shot
            # attempts for the "(n shots)" fallback (display-only).
            _phantom = etype == "Goal" and detail in _GOAL_DISALLOWED_DETAILS
            is_sot = (
                (etype == "Shot" and detail == "On target")
                or (etype == "Goal" and detail != "Own Goal" and not _phantom)
            )
            # v10.58: total shots = every Shot event + Goals — for the
            # "(n shots)" fallback (players shooting without converting)
            if etype == "Shot" or (etype == "Goal" and detail != "Own Goal"):
                _shots_by_team_player.setdefault(ev_tid, {})
                _shots_by_team_player[ev_tid][pname] = _shots_by_team_player[ev_tid].get(pname, 0) + 1
            if is_sot:
                _sot_by_team_player.setdefault(ev_tid, {})
                _sot_by_team_player[ev_tid][pname] = _sot_by_team_player[ev_tid].get(pname, 0) + 1
            # v10.62: THE phantom-scorer fix — a real goal (not own goal,
            # not disallowed/missed-pen) is the only thing that flags a scorer.
            if etype == "Goal" and detail != "Own Goal" and not _phantom:
                _goals_by_team_player.setdefault(ev_tid, {})
                _goals_by_team_player[ev_tid][pname] = _goals_by_team_player[ev_tid].get(pname, 0) + 1

        # v10.46: non-scorers by SOT desc first, then scorers by SOT desc.
        # v10.57: entries carry the scorer flag. v10.58: entries are
        # (name, sot, total_shots, scored) so the filter can rank shooters.
        _fixture_data: dict[int, list[tuple[str, int, int, bool]]] = {}
        for tid, players in _shots_by_team_player.items():
            _goals = _goals_by_team_player.get(tid, {})
            _sots = _sot_by_team_player.get(tid, {})
            _entries = [
                (p, _sots.get(p, 0), c, _goals.get(p, 0) > 0)
                for p, c in players.items()
            ]
            _entries.sort(key=lambda x: (x[3], -x[1], -x[2], x[0]))
            _fixture_data[tid] = _entries

        # v10.57: valid-goal count (own goals count — they change the score
        # too; VAR-disallowed don't).
        _gc = sum(
            1 for ev in events
            if ev.get("type", "") == "Goal" and ev.get("detail", "") not in _GOAL_DISALLOWED_DETAILS
        )
        # v10.44p: Track latest SOT event minute for latency measurement
        # v10.62: phantom goals excluded (same rule as SOT counting)
        _latest_min = 0
        for ev in events:
            etype = ev.get("type", "")
            detail = ev.get("detail", "")
            _ph = etype == "Goal" and detail in _GOAL_DISALLOWED_DETAILS
            if (etype == "Shot" and detail == "On target") or (
                etype == "Goal" and detail != "Own Goal" and not _ph
            ):
                ev_min = safe_int(str(ev.get("time", {}).get("elapsed", 0) or 0))
                if ev_min > _latest_min:
                    _latest_min = ev_min
        return _fixture_data, _gc, _latest_min

    try:
        _fixture_data, _gc, _latest_min = _parse_feed()
        result = _filter_non_scorers(_fixture_data.get(team_id, []))[:max_players]

        # v10.58: feed-lag retry — stats know SOT the events feed hasn't
        # listed yet AND the line would be empty (every listed shooter
        # scored). Wait once, re-fetch once. This runs AFTER the signal is
        # already in Telegram, so it costs zero signal latency.
        # v10.65: pre_signal=True SKIPS the inline sleep — the fetch runs
        # BEFORE the send now, so a 3s stall would delay the signal; the
        # deferred recovery queue owns late-catching feeds instead.
        _team_ev_sot = sum(e[1] for e in _fixture_data.get(team_id, []))
        if not result and team_sot_now is not None and team_sot_now > _team_ev_sot:
            if pre_signal:
                log.info(
                    f"  v10.65 TOP-SOT PRE-SIGNAL: F{fixture_id} — feed lag "
                    f"(events {_team_ev_sot} vs stats {team_sot_now}); no inline "
                    f"retry, deferred recovery will send the follow-up"
                )
            else:
                log.info(
                    f"  v10.58 TOP-SOT FEED-LAG: F{fixture_id} — events list "
                    f"{_team_ev_sot} SOT vs stats SOT={team_sot_now}; one retry "
                    f"after {PLAYER_FEED_RETRY_DELAY}s"
                )
                time.sleep(PLAYER_FEED_RETRY_DELAY)
                _fixture_data, _gc, _latest_min = _parse_feed()
                result = _filter_non_scorers(_fixture_data.get(team_id, []))[:max_players]

        # v10.62: say WHY the line is empty instead of going silent
        if not result:
            _log_top_sot_skip(fixture_id, team_id, "fresh fetch",
                              _fixture_data.get(team_id, []), team_sot_now)
            # v10.63: DEFERRED RECOVERY — a 3-second retry cannot bridge a
            # minutes-long feed lag. Queue a late re-fetch that runs on the
            # next main-loop tick and sends the line once the feed
            # has caught up. Only on feed-lag EVIDENCE (stats know SOT the
            # feed has not listed); the true-silence case never retries.
            _team_ev_sot2 = sum(e[1] for e in _fixture_data.get(team_id, []))
            if team_sot_now is not None and team_sot_now > _team_ev_sot2:
                _queue_top_sot_retry(fixture_id, team_id, "lag",
                                     ev_sot=_team_ev_sot2, stats_sot=team_sot_now,
                                     league_id=league_id)
        else:
            # v10.63: line delivered — cancel any pending recovery
            _top_sot_retry_queue.pop((fixture_id, team_id), None)

        # Cache all teams' data for this fixture
        _player_sot_cache[fixture_id] = _fixture_data
        # v10.57: remember the valid-goal count this cache was built from
        _player_sot_cache_goals[fixture_id] = _gc
        _note_fixture_goals(fixture_id, _gc)  # seeds latest-known (never invalidates here)
        # v10.58: stats-side SOT snapshot for the growth guard (same-source
        # comparisons only — never events sums).
        if team_sot_now is not None:
            _player_sot_cache_built_sot.setdefault(fixture_id, {})[team_id] = team_sot_now
        # v10.63: snapshot the OTHER team(s) too, from their latest stats
        # poll in team_gps_history — before, only the signaling team got a
        # growth snapshot, so the second team's next line was served from
        # stale entries with no staleness check at all.
        for _other_tid in _fixture_data:
            if _other_tid not in _player_sot_cache_built_sot.setdefault(fixture_id, {}):
                _oh = team_gps_history.get((fixture_id, _other_tid), [])
                if _oh:
                    _player_sot_cache_built_sot[fixture_id][_other_tid] = int(
                        _oh[-1].get("sot", 0) or 0
                    )
        # v10.73: PLAYERS-STATS FALLBACK — when the events line is empty and
        # the events feed is BEHIND the stats lane (or listed nobody), ONE
        # /fixtures/players call (1 credit, per-league censused, daily-
        # capped) resolves the 'scores next' names from the STATS pipeline
        # (Serie A / Bundesliga / PL / Eredivisie / Ligue 1 class). Runs
        # pre-signal so the line is IN the signal (v10.65 user rule); repeat
        # signals serve from the merged cache at zero cost.
        _players_fallback_used = False
        if not result:
            _entries_ev = _fixture_data.get(team_id, [])
            _ev_sum = sum(e[1] for e in _entries_ev)
            _feed_behind = (
                (team_sot_now is not None and team_sot_now > _ev_sum)
                or not _entries_ev
            )
            if (
                _feed_behind
                and league_id is not None
                and not _players_census_says_no_data(league_id)
                and _players_sot_budget_ok()
            ):
                _pl_result = fetch_top_sot_players_from_players_api(
                    client, fixture_id, team_id, max_players=max_players,
                    team_sot_now=team_sot_now, league_id=league_id,
                    league_name=LEAGUE_IDS.get(league_id, f"league {league_id}"),
                )
                if _pl_result:
                    result = _pl_result
                    _players_fallback_used = True
                    # line delivered — cancel any pending deferred recovery
                    _top_sot_retry_queue.pop((fixture_id, team_id), None)
                    log.info(
                        f"  v10.73 TOP-SOT PLAYERS-FALLBACK: F{fixture_id} — events "
                        f"side empty/behind (stats SOT={team_sot_now}); "
                        f"/fixtures/players delivered {len(_pl_result)} name(s)"
                    )
        # v10.65: record the outcome for the signal path + census
        _entries_final = _fixture_data.get(team_id, [])
        _last_top_sot_info = {
            "outcome": "ok" if result else (
                "all_scored" if _entries_final else "no_shooters"
            ),
            "listed_sot": sum(e[1] for e in _entries_final),
            "feed_goals": _gc,  # v10.87: events valid-goal count for the race guard
        }
        if _players_fallback_used:
            # v10.73: mark the source so the events-feed census counts this
            # as a players-rescue, not as an events line (census honesty).
            _last_top_sot_info["source"] = "players"
        if _latest_min > 0:
            _latest_sot_event_minute[fixture_id] = _latest_min

        return result
    except Exception as e:
        log.warning(f"  Failed to fetch player SOT for fixture {fixture_id}: {e}")
        # v10.63: a failed fetch is indistinguishable from total feed lag —
        # queue one deferred recovery instead of losing the line outright.
        _queue_top_sot_retry(fixture_id, team_id, "error", league_id=league_id)
        _last_top_sot_info = {"outcome": "error", "listed_sot": 0}
        return []


def _queue_top_sot_retry(fid: int, tid: int, reason: str,
                         ev_sot: int | None = None, stats_sot: int | None = None,
                         league_id: int | None = None) -> None:
    """v10.63: Queue a deferred Top-SOT line recovery (feed lag / fetch error).

    The immediate 3-second retry (v10.58) cannot bridge an events-feed lag
    of minutes. This queues one late re-fetch (TOP_SOT_RETRY_DELAY, max
    TOP_SOT_RETRY_MAX_ATTEMPTS, daily credit cap) that sends the line
    AFTER the feed caught up. Display-only: never gates, never blocks the
    signal loop (processing happens on the next main-loop tick).
    v10.65: leagues the census has LEARNED never deliver Shot events are
    not queued at all — no credits on a hopeless follow-up."""
    global _top_sot_retry_credits_today, _top_sot_retry_date
    try:
        if league_id is not None and _sot_census_says_no_data(league_id):
            log.info(
                f"  v10.65 TOP-SOT: F{fid} T{tid} — league {league_id} census "
                f"says the feed never delivers shot events; recovery NOT queued"
            )
            return
        _today = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
        if _top_sot_retry_date != _today:
            _top_sot_retry_date = _today
            _top_sot_retry_credits_today = 0
        if _top_sot_retry_credits_today >= TOP_SOT_RETRY_CREDIT_CAP:
            return
        _k = (fid, tid)
        if _k in _top_sot_retry_queue:
            return  # already queued — first signal wins
        _top_sot_retry_queue[_k] = {
            "attempts": 0,
            "retry_at": time.time() + TOP_SOT_RETRY_DELAY,
            "reason": reason, "ev_sot": ev_sot, "stats_sot": stats_sot,
        }
        log.info(
            f"  v10.63 TOP-SOT DEFERRED: F{fid} T{tid} — line empty ({reason}; "
            f"events SOT={ev_sot} vs stats SOT={stats_sot}); recovery retry "
            f"in {TOP_SOT_RETRY_DELAY}s"
        )
    except Exception:
        pass


def process_top_sot_retries(client: httpx.Client) -> None:
    """v10.63: Run due deferred Top-SOT line recoveries (one per tick).

    Called every main-loop iteration (zero cost while the queue is empty).
    For each due entry: drop the fixture's player cache (it holds the
    lagged feed that produced the empty line), re-fetch fresh (1 credit),
    and send the recovered line with the CURRENT score/minute. Gives up
    after TOP_SOT_RETRY_MAX_ATTEMPTS. Never raises; never gates; skips
    fixtures that left the live set."""
    global _top_sot_retry_credits_today, _top_sot_retry_date
    if not _top_sot_retry_queue:
        return
    try:
        _now = time.time()
        _today = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
        if _top_sot_retry_date != _today:
            _top_sot_retry_date = _today
            _top_sot_retry_credits_today = 0
        for (fid, tid) in list(_top_sot_retry_queue.keys()):
            entry = _top_sot_retry_queue.get((fid, tid))
            if entry is None:
                continue
            if _now < entry.get("retry_at", 0):
                continue
            f = find_cached_fixture(fid)
            if not f:
                _top_sot_retry_queue.pop((fid, tid), None)
                continue
            _st = f.get("fixture", {}).get("status", {}).get("short", "")
            if _st not in LIVE_STATUSES:
                _top_sot_retry_queue.pop((fid, tid), None)
                continue
            if entry.get("attempts", 0) >= TOP_SOT_RETRY_MAX_ATTEMPTS:
                log.info(
                    f"  v10.63 TOP-SOT RECOVERY GAVE UP: F{fid} T{tid} — feed "
                    f"still empty after {TOP_SOT_RETRY_MAX_ATTEMPTS} retry(ies)"
                )
                # v10.65: census counts the give-up (per league) — enough
                # give-ups with zero lines/recoveries flips the league to
                # NO DATA and future queueing stops.
                _update_sot_feed_census(
                    f.get("league", {}).get("id"),
                    f.get("league", {}).get("name", "?"),
                    "gave_up",
                )
                _top_sot_retry_queue.pop((fid, tid), None)
                continue
            if _top_sot_retry_credits_today >= TOP_SOT_RETRY_CREDIT_CAP:
                log.info(
                    "  v10.63 TOP-SOT RECOVERY: daily credit cap reached — queue drained"
                )
                _top_sot_retry_queue.clear()
                return
            entry["attempts"] = entry.get("attempts", 0) + 1
            _top_sot_retry_credits_today += 1
            # Force a FRESH parse: the cache holds the lagged feed that
            # produced the empty line (goal-count bookkeeping is rebuilt
            # by the fetch itself).
            _player_sot_cache.pop(fid, None)
            _player_sot_cache_built_sot.pop(fid, None)
            _h = team_gps_history.get((fid, tid), [])
            _sot_now = int(_h[-1].get("sot", 0) or 0) if _h else None
            players = fetch_top_sot_players(
                client, fid, tid, 3, team_sot_now=_sot_now,
                league_id=f.get("league", {}).get("id"),
            )
            if players:
                _top_sot_retry_queue.pop((fid, tid), None)
                # v10.65: census counts the recovery (per league)
                _update_sot_feed_census(
                    f.get("league", {}).get("id"),
                    f.get("league", {}).get("name", "?"),
                    "recovered",
                    listed_sot=_last_top_sot_info.get("listed_sot", 0),
                    stats_sot=_sot_now or 0,
                )
                home = f["teams"]["home"]["name"]
                away = f["teams"]["away"]["name"]
                sh = f["goals"]["home"] or 0
                sa = f["goals"]["away"] or 0
                minute = safe_int(str(f["fixture"].get("status", {}).get("elapsed", 0) or 0))
                tname = home if f["teams"]["home"].get("id") == tid else away
                _parts = [
                    f"{_n} ({_s})" if _s > 0 else f"{_n} ({_t} shots)"
                    for _n, _s, _t in players
                ]
                _any_sot = any(_s > 0 for _, _s, _ in players)
                _head = "\U0001f3af Top SOT" if _any_sot else "\U0001f3af Top shooters (no SOT yet)"
                try:
                    send_telegram(
                        client,
                        f"{_head} — {tname}: {', '.join(_parts)}\n"
                        f"({home} {sh} - {sa} {away}, {minute}')\n"
                        f"\u23f1 feed-lag recovery (line unavailable at signal time)",
                    )
                    log.info(
                        f"  v10.63 TOP-SOT RECOVERED: F{fid} T{tid} — line sent "
                        f"after retry #{entry['attempts']}"
                    )
                except Exception as e:
                    log.warning(f"  v10.63 TOP-SOT RECOVERED but send failed: {e}")
            elif (_last_top_sot_info.get("outcome") == "all_scored"
                    and (_sot_now is None
                         or _sot_now <= _last_top_sot_info.get("listed_sot", 0))):
                # v10.70 (user rule, FINAL): the Top-SOT line only ever
                # names players who have NOT scored. The feed is CURRENT
                # (stats SOT fully itemized) and every listed shooter has
                # already scored, so no retry can produce a non-scorer
                # line — stop the retry chain QUIETLY (saves the remaining
                # retry credits; there is no eligible name to send).
                _top_sot_retry_queue.pop((fid, tid), None)
                log.info(
                    f"  v10.70 TOP-SOT RECOVERY (all scored): F{fid} T{tid} — feed "
                    f"current, every listed shooter scored; no non-scorer line "
                    f"possible, retries stopped (credit saved)"
                )
                _update_sot_feed_census(
                    f.get("league", {}).get("id"),
                    f.get("league", {}).get("name", "?"),
                    "all_scored",
                    listed_sot=_last_top_sot_info.get("listed_sot", 0),
                    stats_sot=_sot_now or 0,
                )
                # NOTE: when the feed still LAGS stats (_sot_now > listed)
                # the elif above does not fire — the else branch schedules
                # the next retry, because the unlisted SOT may belong to a
                # non-scorer (exactly the name the user wants).
            else:
                # still empty — schedule the next (and final) attempt
                entry["retry_at"] = _now + TOP_SOT_RETRY_DELAY
    except Exception as e:
        log.warning(f"  v10.63 TOP-SOT retry processing failed: {e}")


def update_event_fast_lane() -> None:
    """v10.50: Track the TOP N hottest fixtures for 10s event polling (was: 1).

    v10.50-BUGFIX: reads live values from team_gps_history (each entry has
    gps/sot/accel_count/minute). The v10.44p/v10.48 version read these keys
    from team_state — which only stores last_sot/last_minute — so every read
    returned 0 and the fast lane could never actually activate. Now it can.

    Criteria per fixture: GPS>=65 AND SOT>=2 AND minute>=21.
    v10.48 GPS 85+ LOCK kept: CORE-window teams at GPS 85+ (the strongest
    winrate bucket) take the fast lane unconditionally — up to 2 lock
    fixtures, then score-ranked picks until N=3 total.
    Widening to 3 (from 1) costs at most ~1080 credits/h worst case,
    still capped by the 2500/day fast-lane budget; typical days far less.
    """
    global _event_fast_lane_fids
    # Per-fixture best values from the LATEST poll of each team
    # (a fixture may have 2 pressuring teams)
    _fx_gps: dict[int, float] = {}
    _fx_sot: dict[int, int] = {}
    _fx_accel: dict[int, int] = {}
    _fx_minute: dict[int, int] = {}
    for (fid, _tid), history in team_gps_history.items():
        if not history:
            continue
        cur = history[-1]
        _g = float(cur.get("gps", 0) or 0)
        _s = int(cur.get("sot", 0) or 0)
        _a = int(cur.get("accel_count", 0) or 0)
        _m = int(cur.get("minute", 0) or 0)
        _fx_gps[fid] = max(_fx_gps.get(fid, 0.0), _g)
        _fx_sot[fid] = max(_fx_sot.get(fid, 0), _s)
        _fx_accel[fid] = max(_fx_accel.get(fid, 0), _a)
        _fx_minute[fid] = max(_fx_minute.get(fid, 0), _m)

    # v10.48: GPS 85+ lock candidates (CORE window 21-61')
    _gps85: list[tuple[int, float]] = []
    for (fid, _tid), history in team_gps_history.items():
        if not history:
            continue
        if fid not in fast_monitored:
            # v10.51 GHOST-PICK GUARD: fixture already dropped from monitoring
            # (past 85' ceiling / fully signaled). Its last GPS entry is stale
            # but still passes GPS>=65+SOT>=2, so it kept re-entering the pick
            # list while poll_event_fast_lane silently filtered it — spamming
            # "Event fast lane -> [...]" every cycle with frozen credits
            # (seen live: F1551080 at 89', credits stuck at 15 for minutes).
            continue
        cur = history[-1]
        _g = float(cur.get("gps", 0) or 0)
        if _g >= 85 and 21 <= int(cur.get("minute", 0) or 0) <= 61:
            _gps85.append((fid, _g))

    _picks: list[int] = []
    _lock_picked: list[int] = []
    for fid, gps in sorted(_gps85, key=lambda kv: -kv[1])[:2]:
        if fid not in _picks:
            _picks.append(fid)
            _lock_picked.append(fid)

    # Score-ranked candidates (best team per fixture)
    _scored: list[tuple[float, int]] = []
    for fid, _gps in _fx_gps.items():
        if fid not in fast_monitored:
            continue  # v10.51 GHOST-PICK GUARD (see GPS85 loop comment)
        if (_gps >= EVENT_FAST_LANE_MIN_GPS
                and _fx_sot.get(fid, 0) >= EVENT_FAST_LANE_MIN_SOT
                and _fx_minute.get(fid, 0) >= 21):
            _score = _gps + (_fx_accel.get(fid, 0) * 5)
            if fid in pressure_accelerating:
                _score += 10
            if fid in sot_burst_fixtures:
                _score += 15
            _scored.append((_score, fid))
    for _score, fid in sorted(_scored, key=lambda kv: -kv[0]):
        if len(_picks) >= FAST_LANE_MAX_FIXTURES:
            break
        if fid not in _picks:
            _picks.append(fid)

    if _picks != _event_fast_lane_fids:
        if _picks:
            _lock_note = ""
            if _lock_picked:
                _lock_note = f" (GPS85 lock: {', '.join('F' + str(f) for f in _lock_picked)})"
            log.info(
                f"  v10.50: Event fast lane -> {[f'F{f}' for f in _picks]}"
                f"{_lock_note} ({_event_fast_lane_credits_today} credits today)"
            )
        elif _event_fast_lane_fids:
            log.info("  v10.44p: Event fast lane cleared (no qualifying fixture)")
        _event_fast_lane_fids = _picks


def poll_event_fast_lane(client: httpx.Client) -> None:
    """v10.50: Poll events for up to FAST_LANE_MAX_FIXTURES fixtures every 10s.

    v10.44p behavior kept: detect SOT jumps the stats feed hasn't caught up
    to (logged only).

    v10.50 SHADOW MODE (Phase 1, FASTLANE_PROPOSAL.md): additionally detects
    VIRTUAL signals from event bursts — "+2 team SOT within 3 min" or
    "+1 SOT right after a goal" — and records them to fastlane_shadow.jsonl.
    Virtual signals are NEVER sent to Telegram and never touch signal logic;
    they resolve like real signals so the EOD report can measure what live
    fast-lane firing WOULD have won (WR, speed gain, duplicates).
    Daily credit cap: 2500 (safety limit; ~2.5h of 3-fixture 10s polling).
    """
    # v10.52 CRASH FIX: _event_fast_lane_fids MUST be declared global — the
    # pre-poll filter below ASSIGNS it; without the declaration Python treats
    # it as local, and the first read (if not _event_fast_lane_fids) raises
    # UnboundLocalError. Introduced in v10.50 when the fast lane was widened
    # to a list (global stmt still named the old singular variable); crashed
    # the deployed bot the moment any fixture became monitored; hotfixed
    # out-of-band but the hotfix was not in the v10.51 build base. Fixed at
    # the source now + covered by smoke_v1052.py (which CALLS this function)
    # and audit_global_vars.py (full-file scoping audit).
    global _event_fast_lane_last, _event_fast_lane_credits_today, _event_fast_lane_fids
    now = time.time()
    if not _event_fast_lane_fids:
        return
    if now - _event_fast_lane_last < EVENT_FAST_LANE_INTERVAL:
        return
    if _event_fast_lane_credits_today >= 2500:  # v10.48: raised 1500->2500 (GPS85 lock needs headroom; daily usage ~230/7500)
        return
    _event_fast_lane_fids = [f for f in _event_fast_lane_fids if f in fast_monitored]
    if not _event_fast_lane_fids:
        return

    for fid in list(_event_fast_lane_fids):
        if _event_fast_lane_credits_today >= 2500:
            break
        try:
            data = api_get(client, "/fixtures/events", {"fixture": fid})
            events = data.get("response", [])
            _event_fast_lane_credits_today += 1

            # v10.60: blocked shots / subs / cards from the SAME response
            # (zero extra credits, logging only)
            _update_event_extras_from_events(fid, events)

            # v10.50: ordered per-team SOT events + goal events (the events
            # list is cumulative for the whole match, chronological)
            sot_events: list[tuple[int, int]] = []   # (team_id, minute)
            goal_events: list[tuple[int, int]] = []  # (team_id, minute) — own goals etc.
            _team_names: dict[int, str] = {}
            ev_sot: dict[int, int] = {}
            latest_min = 0
            for ev in events:
                etype = ev.get("type", "")
                detail = ev.get("detail", "")
                _tev = ev.get("team", {})
                tid_ev = _tev.get("id")
                if tid_ev is None:
                    continue
                if tid_ev not in _team_names:
                    _team_names[tid_ev] = _tev.get("name") or f"team{tid_ev}"
                ev_min = safe_int(str(ev.get("time", {}).get("elapsed", 0) or 0))
                is_sot = (etype == "Shot" and detail == "On target") or (etype == "Goal" and detail != "Own Goal")
                # v10.52 fix: register ALL goal events for the shadow
                # goal-proximity rule (normal goals previously fell into the
                # SOT branch and the elif below only caught own goals, so
                # _fl_goal_minutes stayed empty ~always — logging-only path).
                if etype == "Goal":
                    goal_events.append((tid_ev, ev_min))
                if is_sot:
                    ev_sot[tid_ev] = ev_sot.get(tid_ev, 0) + 1
                    sot_events.append((tid_ev, ev_min))
                    if ev_min > latest_min:
                        latest_min = ev_min

            # Update latency tracking
            if latest_min > 0:
                _latest_sot_event_minute[fid] = latest_min

            # v10.50: register new goal events (any team) for proximity rule
            _g_seen = _fl_seen_goal_count.get(fid, 0)
            if len(goal_events) < _g_seen:
                _g_seen = 0
            for _gtid, _gmin in goal_events[_g_seen:]:
                _fl_goal_minutes.setdefault(fid, []).append((_gmin, _gtid))
            _fl_seen_goal_count[fid] = len(goal_events)

            # v10.53/v10.54: goal flashes + pre-goal SURGE alarms from the
            # SAME events fetch (zero extra credits). Shared parsing; each
            # processor applies its own enabled check and dedupe state.
            if _goal_watch_enabled or _surge_watch_enabled:
                _gw_goals, _gw_sot_mins, _gw_shot_mins = _parse_goal_watch_events(events)
                # v10.57: a new goal may demote the Top SOT headline player
                _note_fixture_goals(fid, len(_gw_goals))
                if _goal_watch_enabled:
                    _process_goal_watch_feed(client, fid, _gw_goals, _gw_sot_mins, source="fastlane")
                _process_surge_watch(client, fid, _gw_goals, _gw_sot_mins, _gw_shot_mins, source="fastlane")

            # v10.50: per-team new SOT events -> burst/proximity evaluation
            for tid_ev in sorted(set(t for t, _m in sot_events)):
                _key = (fid, tid_ev)
                _seen = _fl_seen_sot_count.get(_key, 0)
                _team_ev_minutes = [m for t, m in sot_events if t == tid_ev]
                if len(_team_ev_minutes) < _seen:
                    _seen = 0  # anomaly (feed re-order) — reset, no false bursts
                _new_mins = _team_ev_minutes[_seen:]
                _fl_seen_sot_count[_key] = len(_team_ev_minutes)
                if not _new_mins:
                    continue
                _hist = _fl_sot_events.setdefault(_key, [])
                for m in _new_mins:
                    _evaluate_fl_shadow(
                        fid, tid_ev, m,
                        _team_names.get(tid_ev, f"team{tid_ev}"),
                        _hist,
                    )
                    _hist.append((m, time.time()))

            # Compare with stats-based SOT to detect lag (v10.44p behavior)
            stats_best = get_fixture_best_sot(fid)
            ev_best = max(ev_sot.values()) if ev_sot else 0
            if ev_best > stats_best:
                lag_sot = ev_best - stats_best
                log.info(
                    f"  v10.44p FAST LANE: F{fid} events SOT={ev_best} > "
                    f"stats SOT={stats_best} (+{lag_sot} shot(s) ahead)"
                )
            else:
                log.debug(f"  v10.44p FAST LANE: F{fid} events SOT={ev_best} stats SOT={stats_best} (in sync)")
        except Exception as e:
            log.warning(f"  v10.44p FAST LANE error for F{fid}: {e}")
    _event_fast_lane_last = now


# v10.53: GOAL WATCH — instant goal flash alerts for close late games.
# (Design notes at the GOAL_WATCH_* constants, ~L3783.)


def _parse_goal_watch_events(events: list[dict]) -> tuple[list[dict], dict[int, list[int]], dict[int, list[int]]]:
    """v10.53: Parse /fixtures/events into (valid_goals, per-team SOT minutes).

    Valid goals exclude VAR-disallowed details (same set as fetch_goal_events).
    SOT minutes include goals (a goal is a shot on target) + "On target" shots.

    v10.54: also returns per-team ALL-shot minutes (every Shot event of any
    detail — On target, Off target, Blocked — plus goals). Used by SURGE
    WATCH: off-target shots usually precede SOT, so they are the earliest
    "pressure is building" sign.
    """
    goals: list[dict] = []
    sot_minutes: dict[int, list[int]] = {}
    shot_minutes: dict[int, list[int]] = {}
    for ev in events or []:
        etype = ev.get("type", "")
        detail = ev.get("detail", "")
        tid_ev = (ev.get("team") or {}).get("id")
        if tid_ev is None:
            continue
        ev_min = safe_int(str((ev.get("time") or {}).get("elapsed", 0) or 0))
        if etype == "Goal" and detail not in _GOAL_DISALLOWED_DETAILS:
            goals.append({
                "team_id": tid_ev,
                "minute": ev_min,
                "detail": detail,
                "player": (ev.get("player") or {}).get("name") or "",
            })
            sot_minutes.setdefault(tid_ev, []).append(ev_min)
            shot_minutes.setdefault(tid_ev, []).append(ev_min)
        elif etype == "Shot":
            shot_minutes.setdefault(tid_ev, []).append(ev_min)
            if detail == "On target":
                sot_minutes.setdefault(tid_ev, []).append(ev_min)
    return goals, sot_minutes, shot_minutes


def _process_goal_watch_feed(
    client: httpx.Client, fid: int,
    goals: list[dict], sot_minutes: dict[int, list[int]],
    source: str,
) -> None:
    """v10.53: Diff the valid-goal list vs what we've seen and flash new goals.

    Shared by the event fast lane (10s) and the goal-watch poll (30s) so a
    fixture in both lanes never double-flashes: ONE seen-counter per fixture
    plus a (fid, minute, team_id) flash key. First sight SEEDS (no flash) —
    a goal that happened while the bot was down is stale and must not
    trigger a live-betting alert.
    """
    global _goal_watch_seen
    seen = _goal_watch_seen.get(fid)
    if seen is None:
        _goal_watch_seen[fid] = len(goals)
        return
    if len(goals) < seen:
        seen = 0  # feed anomaly (re-order/truncation) — reset, never false-flash
    _new = goals[seen:]
    _goal_watch_seen[fid] = len(goals)
    for _g in _new:
        try:
            _handle_goal_flash(client, fid, _g, goals, sot_minutes, source)
        except Exception as _e:
            log.warning(f"  v10.53 GOAL FLASH error F{fid}: {_e}")


def _handle_goal_flash(
    client: httpx.Client, fid: int, goal: dict,
    valid_goals: list[dict], sot_minutes: dict[int, list[int]],
    source: str,
) -> None:
    """v10.53: Send ONE goal flash Telegram alert + log it to goal_flash.jsonl."""
    global _gw_flashes_today, _gw_flashes_date
    if not _goal_watch_enabled:
        return
    minute = goal.get("minute", 0) or 0
    if not (GOAL_WATCH_MINUTE <= minute <= 90):
        return
    # Dedupe across lanes: one flash per (fixture, minute, scoring event team)
    _key = (fid, minute, goal.get("team_id"))
    if _key in _goal_watch_flash_keys:
        return
    _goal_watch_flash_keys.add(_key)

    # Team names / ids / league from the cached fixture (stats feed)
    _home_name, _away_name, _home_id, _away_id, _league = "?", "?", None, None, "?"
    _fx = find_cached_fixture(fid)
    if _fx:
        try:
            _teams = _fx.get("teams", {}) or {}
            _home_name = _teams.get("home", {}).get("name") or "?"
            _away_name = _teams.get("away", {}).get("name") or "?"
            _home_id = _teams.get("home", {}).get("id")
            _away_id = _teams.get("away", {}).get("id")
            _league = (_fx.get("league", {}) or {}).get("name") or "?"
        except Exception:
            pass

    # Own goal: the event team shot it into their OWN net — beneficiary flips
    _is_og = (goal.get("detail") == "Own Goal")
    _scoring_tid = goal.get("team_id")
    if _is_og:
        _scoring_tid = _away_id if _scoring_tid == _home_id else _home_id
    if _scoring_tid == _home_id and _home_id is not None:
        _scoring_name = _home_name
    elif _scoring_tid == _away_id and _away_id is not None:
        _scoring_name = _away_name
    else:
        _scoring_name = f"team{_scoring_tid}"

    # Reconstruct the running score from valid goal events (own goals flip)
    _hg = 0
    _ag = 0
    for _g in valid_goals:
        _benef = _g.get("team_id")
        if _g.get("detail") == "Own Goal":
            _benef = _away_id if _benef == _home_id else _home_id
        if _benef == _home_id and _home_id is not None:
            _hg += 1
        elif _benef == _away_id and _away_id is not None:
            _ag += 1

    # Pressure context: scoring team's SOT in the last 10 game minutes
    _sot10 = 0
    if _scoring_tid is not None:
        _sot10 = sum(
            1 for m in sot_minutes.get(_scoring_tid, [])
            if minute - 10 <= m <= minute
        )

    # Cluster: any earlier valid goal (either team) within GOAL_WATCH_CLUSTER_MIN
    _cluster_gap = None
    for _g in valid_goals:
        _gm = _g.get("minute", 0) or 0
        if _gm < minute and 0 < (minute - _gm) <= GOAL_WATCH_CLUSTER_MIN:
            _cluster_gap = minute - _gm

    # Daily counter rollover
    _today = time.strftime("%Y-%m-%d")
    if _gw_flashes_date != _today:
        _gw_flashes_date = _today
        _gw_flashes_today = 0
    _gw_flashes_today += 1

    _scorer = goal.get("player") or ""
    _detail = goal.get("detail") or ""
    _lines = [f"\u26bd GOAL {minute}' \u2014 {_scoring_name}"]
    _lines.append(f"{_home_name} {_hg}-{_ag} {_away_name} ({_league})")
    if _scorer:
        _tag = " (OG)" if _is_og else (
            f" ({_detail})" if _detail and _detail != "Normal Goal" else ""
        )
        _lines.append(f"{_scorer}{_tag}")
    _lines.append(f"Shots on target last 10': {_sot10}")
    if _cluster_gap is not None:
        _lines.append(
            f"\U0001f525 CLUSTER \u2014 2nd goal in {_cluster_gap} min. "
            "Goals come in bursts \u2014 next-goal risk is HIGH right now."
        )
    else:
        _lines.append("\u26a1 Cluster watch: a 2nd goal often follows quickly.")
    _lines.append(f"[{source} | {BOT_VERSION}]")
    send_telegram(client, "\n".join(_lines))

    _append_goal_flash({
        "ts": round(time.time(), 3),
        "date": _today,
        "fixture_id": fid,
        "league": _league,
        "minute": minute,
        "team_id": _scoring_tid,
        "team_name": _scoring_name,
        "player": _scorer,
        "detail": _detail,
        "own_goal": _is_og,
        "score_home": _hg,
        "score_away": _ag,
        "sot_10m": _sot10,
        "cluster_gap_min": _cluster_gap,
        "source": source,
        "version": BOT_VERSION,
    })
    log.info(
        f"  v10.53 GOAL FLASH: F{fid} {minute}' {_scoring_name} "
        f"({_hg}-{_ag}) SOT10={_sot10} cluster={_cluster_gap} [{source}]"
    )


def _append_goal_flash(entry: dict) -> None:
    """v10.53: Append one goal-flash record to goal_flash.jsonl (best effort)."""
    try:
        with open(GOAL_FLASH_FILE, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        log.warning(f"  v10.53: goal_flash.jsonl append failed: {_e}")


# v10.54: SURGE WATCH — pre-goal pressure buildup alarms.
# (Design notes at the SURGE_* constants, ~L3828.)


def _surge_fixture_ctx(fid: int, valid_goals: list[dict]) -> tuple:
    """v10.54: Resolve (home_name, away_name, home_id, away_id, league, hg, ag).

    Score is reconstructed from the valid goal events with own-goal
    beneficiary flip (same rules as _handle_goal_flash, duplicated
    deliberately to keep the v10.53 flash path untouched).
    """
    _home_name, _away_name, _home_id, _away_id, _league = "?", "?", None, None, "?"
    _fx = find_cached_fixture(fid)
    if _fx:
        try:
            _teams = _fx.get("teams", {}) or {}
            _home_name = _teams.get("home", {}).get("name") or "?"
            _away_name = _teams.get("away", {}).get("name") or "?"
            _home_id = _teams.get("home", {}).get("id")
            _away_id = _teams.get("away", {}).get("id")
            _league = (_fx.get("league", {}) or {}).get("name") or "?"
        except Exception:
            pass
    _hg, _ag = 0, 0
    for _g in valid_goals or []:
        _benef = _g.get("team_id")
        if _g.get("detail") == "Own Goal":
            _benef = _away_id if _benef == _home_id else _home_id
        if _benef == _home_id and _home_id is not None:
            _hg += 1
        elif _benef == _away_id and _away_id is not None:
            _ag += 1
    return _home_name, _away_name, _home_id, _away_id, _league, _hg, _ag


def _surge_ordinal(n: int) -> str:
    """v10.55: 1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th' (sustain text)."""
    if 10 <= n % 100 <= 20:
        _sfx = "th"
    else:
        _sfx = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{_sfx}"


def _send_surge_alert(
    client, fid: int, tid: int, kind: str, minute: int,
    detail_lines: list[str], ctx: tuple, source: str,
) -> None:
    """v10.54/v10.55: Send ONE surge alert to Telegram + log it to surge_watch.jsonl."""
    global _surge_alerts_today, _surge_fixture_alerts
    _home_name, _away_name, _home_id, _away_id, _league, _hg, _ag = ctx
    if tid == _home_id and _home_id is not None:
        _team_name = _home_name
    elif tid == _away_id and _away_id is not None:
        _team_name = _away_name
    else:
        _team_name = f"team{tid}"

    _titles = {
        "wake": f"\U0001f50e SURGE WATCH \u2014 {_team_name} waking up ({minute}')",
        "burst": f"\U0001f525 SURGE \u2014 {_team_name} ({minute}')",
        "sustain": f"\U0001f525 SURGE CONTINUES \u2014 {_team_name} ({minute}')",
        "flood": f"\U0001f4c8 PRESSURE BUILDING \u2014 {_team_name} ({minute}')",
    }
    _lines = [_titles.get(kind, f"\u26a0\ufe0f SURGE \u2014 {_team_name} ({minute}')")]
    _lines.append(f"{_home_name} {_hg}-{_ag} {_away_name} ({_league})")
    _lines.extend(detail_lines)
    _lines.append(f"[{source} | {BOT_VERSION}]")
    send_telegram(client, "\n".join(_lines))

    _today = time.strftime("%Y-%m-%d")
    _surge_alerts_today += 1
    _surge_fixture_alerts[fid] = _surge_fixture_alerts.get(fid, 0) + 1
    try:
        with open(SURGE_WATCH_FILE, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "ts": round(time.time(), 3),
                "date": _today,
                "fixture_id": fid,
                "league": _league,
                "kind": kind,
                "minute": minute,
                "team_id": tid,
                "team_name": _team_name,
                "score_home": _hg,
                "score_away": _ag,
                "source": source,
                "version": BOT_VERSION,
            }, ensure_ascii=False) + "\n")
    except Exception as _e:
        log.warning(f"  v10.54: surge_watch.jsonl append failed: {_e}")
    log.info(
        f"  v10.54 SURGE ({kind}): F{fid} {minute}' {_team_name} "
        f"({_hg}-{_ag}) [{source}]"
    )


def _process_surge_watch(
    client, fid: int,
    goals: list[dict], sot_minutes: dict[int, list[int]],
    shot_minutes: dict[int, list[int]],
    source: str,
) -> None:
    """v10.54/v10.55: Raise pre-goal SURGE alarms when a quiet team suddenly wakes up.

    Called from BOTH event lanes (fast lane ~10s, watch lane ~30s) with the
    SAME parsed feed — shared seen-counters per (fid, team) mean a fixture in
    both lanes never double-alerts. Rules mirror goal-flash hygiene:
      - first sight SEEDS (events that happened while the fixture was not
        watched are history, not a surge)
      - feed re-order/truncation resets silently
      - staleness guard: an event whose minute lags the fixture's current
        minute by > SURGE_STALENESS_TOL is treated as missed-while-away
        (counted, never alerted)
    Layers: SHOT-FLOOD (any shots, earliest) / WAKE-UP (first SOT after a
    long silence) / BURST (2nd SOT in the burst window) / SUSTAIN (every
    further SOT in the burst window — episode stays open).

    v10.55 goal semantics: the goal-scoring shot is the team's last known
    shot (silence after it is measured from the goal's minute) and closes
    any open episode, but never opens, completes, or sends an alert —
    there is nothing left to warn about for a goal that already happened.
    Alerts never touch signal_outcomes or signal gates.
    """
    global _surge_alerts_today, _surge_alerts_date
    if not _surge_watch_enabled:
        return
    _today = time.strftime("%Y-%m-%d")
    if _surge_alerts_date != _today:
        _surge_alerts_date = _today
        _surge_alerts_today = 0
        _surge_fixture_alerts.clear()
    if _surge_alerts_today >= SURGE_MAX_PER_DAY:
        return
    _now = time.time()

    # fixture context: names, score, current minute (staleness guard)
    _ctx = _surge_fixture_ctx(fid, goals)
    _home_id, _away_id = _ctx[2], _ctx[3]
    _cur_min = None
    _fx = find_cached_fixture(fid)
    if _fx:
        try:
            _cur_min = safe_int(str(
                (((_fx.get("fixture", {}) or {}).get("status", {}) or {}).get("elapsed", 0) or 0)
            ))
        except Exception:
            _cur_min = None
    # SOT events that are goals (message suppressed when goal flash covers it)
    _goal_keys = {(g.get("team_id"), g.get("minute", 0) or 0) for g in (goals or [])}

    def _fresh(m: int) -> bool:
        # event minute must be at most SURGE_STALENESS_TOL behind the game
        return _cur_min is not None and (_cur_min - m) <= SURGE_STALENESS_TOL

    def _caps_ok() -> bool:
        # v10.67: retained 86'+ fixtures get a small alert-budget bonus —
        # the base 5-per-fixture cap is usually already burned by 85' and
        # the remaining game lifetime is ~5 minutes (no spam risk).
        _cap = SURGE_MAX_PER_FIXTURE + (SURGE_LATE_BONUS if fid in _late_retain_fids else 0)
        return (_surge_alerts_today < SURGE_MAX_PER_DAY
                and _surge_fixture_alerts.get(fid, 0) < _cap)

    # both teams are always iterated (even with zero events so far) so a
    # fixture that enters the lane quiet gets seeded NOW — its first-ever
    # SOT later counts as NEW and correctly raises the wake-up alarm
    _team_ids = ({_home_id, _away_id} | set(sot_minutes) | set(shot_minutes))
    _team_ids.discard(None)
    for _tid in sorted(_team_ids):
        _key = (fid, _tid)
        _sot_list = sorted(sot_minutes.get(_tid, []))
        _shot_list = sorted(shot_minutes.get(_tid, []))

        # ------------- SOT layer: wake-up + burst + sustain -------------
        # v10.55 goal semantics: a goal-scoring shot is COUNTED as the
        # team's last known shot (future silence is measured from its
        # minute) and CLOSES any open episode — but it can never open,
        # complete, or trigger an alert: the goal already happened.
        _seen_sot = _surge_seen_sot.get(_key)
        if _seen_sot is None:
            _surge_seen_sot[_key] = len(_sot_list)  # seed: history is not a surge
        else:
            if len(_sot_list) < _seen_sot:
                _seen_sot = 0  # feed anomaly — reset, never false-alert
            for _m in _sot_list[_seen_sot:]:
                if not _fresh(_m):
                    continue
                # ---- goal shot: close the episode, never alert ----
                if (_tid, _m) in _goal_keys:
                    _surge_wake_minute.pop(_key, None)
                    _surge_burst_last_minute.pop(_key, None)
                    _surge_burst_count.pop(_key, None)
                    # mark the moment so the shot-flood tier never
                    # double-reports a goal-covered burst (the surge
                    # message itself stays silent — goals never alert)
                    _surge_last_alert_ts[_key] = _now
                    continue
                # ---- burst (2nd SOT) / sustain (3rd+): episode SOTs ----
                # v10.55: the episode no longer closes after the 2nd SOT —
                # every additional SOT within SURGE_SOT_BURST_WINDOW of the
                # previous one keeps alerting (3rd, 4th, ... capped by
                # SURGE_MAX_PER_FIXTURE) so continuous pressure is fully
                # covered, not just the first two shots of it.
                _w = _surge_wake_minute.get(_key)
                if _w is not None:
                    _last = _surge_burst_last_minute.get(_key)
                    if _last is None:
                        _last = _w
                    if 0 <= (_m - _last) <= SURGE_SOT_BURST_WINDOW:
                        _n = _surge_burst_count.get(_key, 1) + 1
                        if _caps_ok():
                            if _n == 2:
                                _send_surge_alert(
                                    client, fid, _tid, "burst", _m,
                                    [f"2nd shot on target in {_m - _last}' "
                                     f"(wake-up was {_w}').",
                                     "Quiet team bursting \u2014 this is the pattern "
                                     "that precedes goals."],
                                    _ctx, source,
                                )
                            else:
                                _send_surge_alert(
                                    client, fid, _tid, "sustain", _m,
                                    [f"{_surge_ordinal(_n)} shot on target "
                                     f"\u2014 burst now {_m - _w}' long "
                                     f"(wake-up was {_w}').",
                                     "Sustained pressure \u2014 teams hammering "
                                     "like this usually convert."],
                                    _ctx, source,
                                )
                        # mark the moment even when the caps suppress the
                        # message so the flood tier never double-reports
                        _surge_burst_last_minute[_key] = _m
                        _surge_burst_count[_key] = _n
                        _surge_last_alert_ts[_key] = _now
                        continue
                    # episode expired: > SURGE_SOT_BURST_WINDOW since its
                    # last SOT — close it, then re-evaluate as a wake-up
                    _surge_wake_minute.pop(_key, None)
                    _surge_burst_last_minute.pop(_key, None)
                    _surge_burst_count.pop(_key, None)
                # wake-up: first SOT after a long team-SOT silence
                # (a goal counts as the last shot: silence after a goal is
                # measured from the goal's own minute — second-goal watch)
                _prev = max((_x for _x in _sot_list if _x < _m), default=0)
                _silence = _m - _prev
                if _m >= SURGE_MIN_MINUTE and _silence >= SURGE_SOT_QUIET_MIN:
                    _surge_wake_minute[_key] = _m  # open episode (state first)
                    _surge_burst_last_minute[_key] = _m
                    _surge_burst_count[_key] = 1
                    _cooldown_ok = (_now - _surge_last_alert_ts.get(_key, 0.0)
                                    >= SURGE_TEAM_COOLDOWN)
                    if _caps_ok() and _cooldown_ok:
                        if _prev > 0 and (_tid, _prev) in _goal_keys:
                            _since = f"since the {_prev}' goal"
                        elif _prev > 0:
                            _since = f"since {_prev}'"
                        else:
                            _since = "of the whole match"
                        _send_surge_alert(
                            client, fid, _tid, "wake", _m,
                            [f"First shot on target in {_silence}' ({_since}).",
                             "Quiet teams that wake up late often score \u2014 "
                             "watch this one."],
                            _ctx, source,
                        )
                    # v10.54: mark the moment even when suppressed (cooldown)
                    # so the flood tier never double-reports
                    _surge_last_alert_ts[_key] = _now
            _surge_seen_sot[_key] = len(_sot_list)

        # ---------------- shot layer: pressure building ------------------
        _seen_sh = _surge_seen_shots.get(_key)
        if _seen_sh is None:
            _surge_seen_shots[_key] = len(_shot_list)  # seed
        else:
            if len(_shot_list) < _seen_sh:
                _seen_sh = 0  # feed anomaly — reset
            for _m in _shot_list[_seen_sh:]:
                if not _fresh(_m):
                    continue
                # v10.55: goal shots never count toward the flood threshold
                # (they are the reference for quiet time, not triggers)
                _window = [_x for _x in _shot_list
                           if _m - SURGE_SHOT_WINDOW <= _x <= _m
                           and (_tid, _x) not in _goal_keys]
                if len(_window) < SURGE_SHOT_FLOOD:
                    continue
                _win_start = _m - SURGE_SHOT_WINDOW
                _prev_shot = max((_x for _x in _shot_list if _x < _win_start), default=0)
                # v10.55: a goal inside the window is the team's most recent
                # shot — quiet is measured from it (the team scored moments
                # ago; that is momentum, not a fresh surge)
                _goal_in_win = max((_x for _x in _shot_list
                                    if _win_start <= _x <= _m
                                    and (_tid, _x) in _goal_keys), default=0)
                if _goal_in_win > _prev_shot:
                    _prev_shot = _goal_in_win
                _quiet = _win_start - _prev_shot
                if (_m >= SURGE_MIN_MINUTE and _quiet >= SURGE_SHOT_QUIET_MIN
                        and _caps_ok()
                        and _m - _surge_flood_minute.get(_key, -99) > SURGE_SHOT_WINDOW
                        # don't double right after a SOT-tier alert for the same team
                        and _now - _surge_last_alert_ts.get(_key, 0.0) >= SURGE_SOT_GUARD):
                    _send_surge_alert(
                        client, fid, _tid, "flood", _m,
                        [f"{len(_window)} shots in {SURGE_SHOT_WINDOW}' "
                         f"after {_quiet}' without one.",
                         "Shots come before goals \u2014 earliest warning layer."],
                        _ctx, source,
                    )
                    _surge_flood_minute[_key] = _m
                    _surge_last_alert_ts[_key] = _now
            _surge_seen_shots[_key] = len(_shot_list)


def _append_surge_alert(entry: dict) -> None:  # kept for symmetry / future use
    """v10.54: Append one surge record to surge_watch.jsonl (best effort)."""
    try:
        with open(SURGE_WATCH_FILE, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        log.warning(f"  v10.54: surge_watch.jsonl append failed: {_e}")


def _retain_late_fixture(fid: int, fixture: dict, reason: str) -> None:
    """v10.67: keep a ceiling-dropped LIVE fixture eligible for the events watch lane.

    Called at every site that removes a fixture from fast_monitored. The
    helper self-gates — retention happens ONLY when the fixture is still
    live AND its minute is past the 85' ceiling AND not yet over 90' — so
    drops for other reasons (finished, 2/2-signaled mid-game, data-dead,
    minute-before-window) are all no-ops. The watch lane re-checks
    status/minute/|score diff| on every poll, so a retained blowout simply
    never gets picked: retention itself costs zero credits.
    """
    try:
        status = fixture["fixture"]["status"]["short"]
        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    except Exception:
        return
    if status not in LIVE_STATUSES:
        return
    if minute <= EXTENDED_MAX or minute > 90:
        return
    if fid in _late_retain_fids:
        return
    while len(_late_retain_fids) >= LATE_RETAIN_MAX:
        _oldest = min(_late_retain_fids, key=lambda f: _late_retain_fids[f])
        _late_retain_fids.pop(_oldest, None)
    _late_retain_fids[fid] = time.time()
    try:
        _names = (f"{fixture['teams']['home']['name']} vs "
                  f"{fixture['teams']['away']['name']}")
    except Exception:
        _names = f"F{fid}"
    log.info(
        f"  v10.67 LATE-RETAIN: {_names} {minute}' ({reason}) — "
        f"events watch lane extended to final whistle"
    )


def poll_goal_watch(client: httpx.Client) -> None:
    """v10.53/v10.54/v10.67: Poll events for close late-game fixtures not in the fast lane.

    Selection: monitored fixtures, minute GOAL_WATCH_MINUTE..90, |score diff|
    <= GOAL_WATCH_MAX_DIFF, not already fast-lane (those react at ~10s).
    Top GOAL_WATCH_MAX_FIXTURES by best SOT. Interval 30s; separate daily
    credit cap GOAL_WATCH_CREDIT_CAP (goal-watch + fast lane share the
    account budget; typical combined usage stays far under 7500).

    v10.54: this lane now ALSO feeds SURGE WATCH (pre-goal pressure alarms)
    whenever either feature is enabled — one events fetch, two detectors,
    zero extra credits.

    v10.67: LATE SURGE — fixtures retained past the 85' ceiling
    (_late_retain_fids: close games the stats lane dropped only because of
    the minute ceiling, e.g. Botev Vratsa 86'/88') stay eligible here until
    FT. Late-retained fixtures take PICK PRIORITY over normal candidates
    (they have minutes left; normal ones have half-hours) and self-clean:
    finished / over-90' fixtures are released from the retain set right
    here. The per-poll |score diff| filter also releases nothing — a blowout
    simply stops being picked while it stays retained (a 4-1 that becomes
    4-3 late still counts as close again).
    """
    global _goal_watch_last, _goal_watch_credits_today, _goal_watch_credits_date, _goal_watch_fids
    if not (_goal_watch_enabled or _surge_watch_enabled):  # v10.54: lane serves both features
        return
    now = time.time()
    if now - _goal_watch_last < GOAL_WATCH_INTERVAL:
        return
    _today = time.strftime("%Y-%m-%d")
    if _goal_watch_credits_date != _today:
        _goal_watch_credits_date = _today
        _goal_watch_credits_today = 0
    if _goal_watch_credits_today >= GOAL_WATCH_CREDIT_CAP:
        if _goal_watch_fids:
            log.info(
                f"  v10.53: Goal watch daily credit cap hit "
                f"({GOAL_WATCH_CREDIT_CAP}) \u2014 paused until tomorrow"
            )
            _goal_watch_fids = []
        return

    # v10.67: late-retained fixtures join the candidate pool (dict.fromkeys:
    # a fixture can be in both sets for one tick while the stats lane drops
    # it and discovery has not pruned yet)
    picks: list[tuple[int, int]] = []
    late_picks: list[tuple[int, int]] = []  # v10.67: retained 86'+ candidates
    for fid in dict.fromkeys(list(fast_monitored) + list(_late_retain_fids)):
        if fid in _event_fast_lane_fids:
            continue  # fast lane already flashes these at ~10s
        f = find_cached_fixture(fid)
        if not f:
            if fid in _late_retain_fids:  # v10.67: cache lost -> release
                _late_retain_fids.pop(fid, None)
            continue
        if fid in _late_retain_fids:  # v10.67: self-clean on non-live status
            _st = ((f.get("fixture", {}) or {}).get("status", {}) or {}).get("short")
            if _st not in LIVE_STATUSES:
                _late_retain_fids.pop(fid, None)
                log.info(f"  v10.67: LATE-RETAIN F{fid} ended ({_st}) — released")
                continue
        _fxs = f.get("fixture", {}) or {}
        _minute = safe_int(str(((_fxs.get("status", {}) or {}).get("elapsed", 0) or 0)))
        if _minute < GOAL_WATCH_MINUTE or _minute > 90:
            if fid in _late_retain_fids and _minute > 90:  # v10.67: past 90' -> release
                _late_retain_fids.pop(fid, None)
            continue
        _gh = f.get("goals", {}) or {}
        _hg = _gh.get("home") or 0
        _ag = _gh.get("away") or 0
        if abs(_hg - _ag) > GOAL_WATCH_MAX_DIFF:
            continue
        if fid in _late_retain_fids:
            late_picks.append((get_fixture_best_sot(fid), fid))
        else:
            picks.append((get_fixture_best_sot(fid), fid))
    late_picks.sort(key=lambda kv: (-kv[0], kv[1]))
    picks.sort(key=lambda kv: (-kv[0], kv[1]))
    # v10.67: late-retained first (minutes left), then normal candidates
    _goal_watch_fids = [fid for _, fid in (late_picks + picks)[:GOAL_WATCH_MAX_FIXTURES]]
    if len(late_picks) + len(picks) > GOAL_WATCH_MAX_FIXTURES:
        _late_note = (f", incl. {len(late_picks)} late-retained 86'+ (priority)"
                      if late_picks else "")
        log.info(
            f"  v10.53: Goal watch -> top {GOAL_WATCH_MAX_FIXTURES} of "
            f"{len(late_picks) + len(picks)} close games{_late_note}"
        )

    for fid in list(_goal_watch_fids):
        if _goal_watch_credits_today >= GOAL_WATCH_CREDIT_CAP:
            break
        try:
            data = api_get(client, "/fixtures/events", {"fixture": fid})
            _goal_watch_credits_today += 1
            events = data.get("response", [])
            _goals, _sot_mins, _shot_mins = _parse_goal_watch_events(events)
            # v10.57: a new goal may demote the Top SOT headline player
            _note_fixture_goals(fid, len(_goals))
            _process_goal_watch_feed(client, fid, _goals, _sot_mins, source="goalwatch")
            # v10.54: pre-goal surge alarms ride the same fetch (0 extra credits)
            _process_surge_watch(client, fid, _goals, _sot_mins, _shot_mins, source="goalwatch")
        except Exception as e:
            log.warning(f"  v10.53 GOAL WATCH error for F{fid}: {e}")
    _goal_watch_last = now


def _backfill_history_from_events(
    client: httpx.Client, fid: int, minute: int,
    home_tid: int, away_tid: int,
) -> None:
    """v10.53: Cold-start warm-up — backfill GPS history from the events feed.

    Problem: after a restart/redeploy the bot picks fixtures up mid-game with
    an empty GPS history. The 61'+ freshness gate needs SOT deltas over the
    last 5/10 GAME minutes, but polls build history in wall-time — a fixture
    picked up at 79' can never show sot_delta_5m for ~5+ minutes, so every
    late signal is FRESHNESS_61-blocked purely for lack of history (Levski
    derby: bot woke at 79', all polls blocked, zero visibility).

    Fix: on the first poll of a cold-started fixture at 61'+, fetch events
    once (1 credit) and PREPEND synthetic history entries at minute-5 /
    minute-10 whose SOT is back-computed from real event minutes. All other
    synthetic fields copy the current poll (deltas = 0, neutral) and gps is
    0.0 (neutral for goalless/accel scoring). The next poll's freshness
    gate then sees REAL shot activity instead of "no data".
    """
    try:
        data = api_get(client, "/fixtures/events", {"fixture": fid})
    except Exception as e:
        log.warning(f"  v10.53 WARM-UP: events fetch failed for F{fid}: {e}")
        return
    events = data.get("response", []) or []

    sot_mins: dict[int, list[int]] = {}
    for ev in events:
        etype = ev.get("type", "")
        detail = ev.get("detail", "")
        tid_ev = (ev.get("team") or {}).get("id")
        if tid_ev is None:
            continue
        ev_min = safe_int(str((ev.get("time") or {}).get("elapsed", 0) or 0))
        if ((etype == "Shot" and detail == "On target")
                or (etype == "Goal" and detail not in _GOAL_DISALLOWED_DETAILS)):
            sot_mins.setdefault(tid_ev, []).append(ev_min)

    _ts = time.time()
    for tid in (home_tid, away_tid):
        hist = team_gps_history.get((fid, tid)) or []
        if not hist:
            continue
        cur = hist[-1]
        sot_now = cur.get("sot", 0) or 0
        _s5 = [m for m in sot_mins.get(tid, []) if m > minute - 5]
        _s10 = [m for m in sot_mins.get(tid, []) if m > minute - 10]
        # CHRONOLOGICAL order (oldest first): the recency builder scans
        # reversed(history) and the windowed-rate calc scans forward — both
        # assume ascending minutes. 10m entry first, then 5m, then the real
        # polls. (smoke_v1053 catches this: entry_5m must NOT find the 10m
        # snapshot.)
        syn: list[dict] = []
        if minute >= 71:
            syn.append({
                "ts": _ts - 600,
                "gps": 0.0,
                "sot": max(sot_now - len(_s10), 0),
                "total_shots": cur.get("total_shots", 0),
                "shots_off_target": cur.get("shots_off_target", 0),
                "accel_count": 0,
                "minute": max(minute - 10, 1),
                "xg": cur.get("xg"),
                "synthetic": True,
            })
        syn.append({
            "ts": _ts - 300,
            "gps": 0.0,
            "sot": max(sot_now - len(_s5), 0),
            "total_shots": cur.get("total_shots", 0),
            "shots_off_target": cur.get("shots_off_target", 0),
            "accel_count": 0,
            "minute": max(minute - 5, 1),
            "xg": cur.get("xg"),
            "synthetic": True,
        })
        team_gps_history[(fid, tid)] = syn + hist
        log.info(
            f"  v10.53 WARM-UP: F{fid} team {tid} at {minute}' \u2014 backfilled "
            f"{len(syn)} synthetic history entries (SOT 5m={len(_s5)}, 10m={len(_s10)})"
        )


def _register_goal_sot_pending(fid: int, tid: int, goals_now: int, goals_prev: int) -> int:
    """v10.56: At goal detection — add the goal shot(s) to the pending ledger.

    Each goal is a shot on target that WILL land in the stats SOT counter
    (score feed updates first, stats 1-3 min later). Returns new pending.

    v10.57: also drops the Top-SOT player cache for this fixture — the
    player who just scored must leave the "scores next" line immediately;
    the next signal fetches fresh events and headlines a non-scorer.
    """
    _g = max(0, (goals_now or 0) - (goals_prev or 0))
    if _g:
        _pending_goal_sot[(fid, tid)] = _pending_goal_sot.get((fid, tid), 0) + _g
        if fid in _player_sot_cache:
            del _player_sot_cache[fid]
            _player_sot_cache_built_sot.pop(fid, None)  # v10.58
            log.info(
                f"  v10.57 TOP-SOT REFRESH: F{fid} — goal detected in stats poll; "
                f"player cache dropped (scorer demotes from the Top SOT line)"
            )
    return _pending_goal_sot.get((fid, tid), 0)


def _consume_pending_goal_sot(fid: int, tid: int, sot: int, prev_sot) -> int:
    """v10.56: SOT rose — assume goal shots land first; consume the ledger.

    Returns consumed count. With a settled stats feed the ledger ends at 0
    and every later SOT increase is genuine (non-goal) pressure.
    """
    if prev_sot is None or sot is None or sot <= prev_sot:
        return 0
    _pend = _pending_goal_sot.get((fid, tid), 0)
    if _pend <= 0:
        return 0
    _consumed = min(_pend, sot - prev_sot)
    _pending_goal_sot[(fid, tid)] = _pend - _consumed
    _goal_sot_landed[(fid, tid)] = _goal_sot_landed.get((fid, tid), 0) + _consumed
    return _consumed


def _genuine_sot_jump(team_sig: dict | None, sot: int, fid: int, tid: int) -> int:
    """v10.56: SOT increase since the last signal, EXCLUDING goal shots.

    The repeat-signal gates (SOT-jump rule, goal-pressure-continues, cooldown
    buildup, first-signal-only exception) must never count the shot that
    scored a goal as trigger evidence — it warns about what already happened.
    """
    if not team_sig:
        return sot
    _sot_at = team_sig.get("sot_at_last_signal", 0)
    _landed_at = team_sig.get("goal_sot_landed_at_last_signal", 0)
    _landed_now = _goal_sot_landed.get((fid, tid), 0)
    return (sot - _sot_at) - max(0, _landed_now - _landed_at)


def fetch_live_sot_from_events(
    client: httpx.Client, fixture_id: int,
) -> tuple[dict[int, int], dict[int, int]] | None:
    """v10.31/v10.56: Count SOT + goals from /fixtures/events (near-real-time).

    API-Football's events endpoint updates faster than statistics because it's
    event-driven, not aggregated. A shot recorded at 82' appears in events
    within seconds, but may not reach /fixtures/statistics for 2-3 minutes.

    Counts per team:
      - type=="Shot" AND detail=="On target"  (saved, blocked, woodwork)
      - type=="Goal" AND detail!="Own Goal"   (goals are on target by definition)

    v10.56: also returns per-team GOAL counts so callers can exclude goal
    shots from trigger evidence ("the goal shot never triggers").

    Returns ({team_id: sot_count}, {team_id: goal_count}) or None on failure.
    Each call costs 1 API credit.
    """
    # Check cache
    cached = _event_sot_cache.get(fixture_id)
    if cached and time.time() - cached.get("ts", 0) < EVENT_SOT_CACHE_TTL:
        _sot_c = {k: v for k, v in cached.get("sot", {}).items() if isinstance(k, int)}
        _gol_c = {k: v for k, v in cached.get("goals", {}).items() if isinstance(k, int)}
        return _sot_c, _gol_c

    try:
        data = api_get(client, "/fixtures/events", {"fixture": fixture_id})
        events = data.get("response", [])

        # v10.60: blocked shots / subs / cards from the SAME response
        # (zero extra credits, logging only)
        _update_event_extras_from_events(fixture_id, events)

        sot_by_team: dict[int, int] = {}
        goals_by_team: dict[int, int] = {}   # v10.56: goal shots (never trigger)
        shot_count = 0
        goal_count = 0
        for ev in events:
            etype = ev.get("type", "")
            detail = ev.get("detail", "")
            tid_ev = ev.get("team", {}).get("id")
            if not tid_ev:
                continue

            if etype == "Shot" and detail == "On target":
                sot_by_team[tid_ev] = sot_by_team.get(tid_ev, 0) + 1
                shot_count += 1
            elif etype == "Goal" and detail not in _GOAL_DISALLOWED_DETAILS and detail != "Own Goal":
                # Goals are shots on target — include in SOT count.
                # v10.56: VAR-disallowed goals are excluded (same set as
                # fetch_goal_events / goal-flash parsing — a disallowed
                # "goal" must never inflate the events SOT count).
                sot_by_team[tid_ev] = sot_by_team.get(tid_ev, 0) + 1
                goals_by_team[tid_ev] = goals_by_team.get(tid_ev, 0) + 1
                goal_count += 1

        # Cache with timestamp (include non-int keys for TTL)
        cache_entry = {"ts": time.time(), "credit_cost": 1,
                       "sot": sot_by_team, "goals": goals_by_team}
        _event_sot_cache[fixture_id] = cache_entry

        log.info(
            f"  EVENT SOT: F{fixture_id} — "
            f"{shot_count} shots on target + {goal_count} goals = "
            f"{dict((k, v) for k, v in sot_by_team.items() if isinstance(k, int))}"
            f" (goal shots: {goals_by_team})"
        )
        return sot_by_team, goals_by_team

    except Exception as e:
        log.warning(f"  EVENT SOT failed for F{fixture_id}: {e}")
        return None


# v10.36: Odds capture — passive metadata for EV analysis.
# Odds NEVER influence signal generation. They are recorded AFTER
# the signal decision is made, purely for post-hoc profitability analysis.
ODDS_CAPTURE_ENABLED = os.environ.get("ODDS_CAPTURE_ENABLED", "true").lower() == "true"
PREFERRED_BOOKMAKER = "Bet365"  # most liquid, widely available
ODDS_RETRY_DELAY = 3            # v10.68: seconds before the single odds retry
ODDS_SUSPECT_IMPLIED = 0.12     # v10.68: over price implying <12% = stale/garbage class (real live next-goal prices: 17-90%)
ODDS_SUSPECT_MINUTE_MAX = 80    # v10.68: signals at/after 80' exempt — dying-minute prices legitimately run high


def _parse_signal_odds(data: dict, total_goals: int) -> dict | None:
    """v10.68: Parse ONE /odds/live (or /odds) response into the capture dict.

    Pure parse — no fetching, no logging (the wrapper owns both), so the
    same parser serves the initial fetch, the failure retry and the
    suspect-price re-fetch. Returns None when the response carries no
    usable bookmaker/markets.
    """
    response = data.get("response", []) if isinstance(data, dict) else []
    if not response:
        return None
    bookmakers = response[0].get("bookmakers", []) or []
    if not bookmakers:
        return None

    # Prefer Bet365, fall back to first bookmaker with data
    chosen = None
    for bm in bookmakers:
        if bm.get("name") == PREFERRED_BOOKMAKER:
            chosen = bm
            break
    if not chosen:
        chosen = bookmakers[0]

    bets = chosen.get("bets", []) or []
    result = {
        "bookmaker": chosen.get("name", "?"),
        "total_goals_at_signal": total_goals,
        "over_line": None,
        "over_odds": None,
        "over_implied": None,
        "btts_yes_odds": None,
        "btts_implied": None,
        "match_home_odds": None,
        "match_away_odds": None,
        "match_draw_odds": None,
        "fetched_at": time.time(),
        "markets_available": [b.get("name", "") for b in bets],
        # v10.80: full O/U ladders for the card/corner markets
        "cards_lines": [],
        "corners_lines": [],
    }

    # Target: Over (current total + 0.5) goals
    target_line = total_goals + 0.5

    for bet in bets:
        bet_name = bet.get("name", "")
        values = bet.get("values", [])

        # Goals Over/Under
        # v10.80 GUARD: Cards/Corners/first-half bet names also contain
        # "Over"+"Under" — without this guard a Cards "Over 2.5" price
        # could be recorded as the goals next-goal price on a 1-1 game,
        # and "Goals Over/Under - Second Half" lines could overwrite the
        # main totals line (both are the wrong-market contamination class).
        if (
            ("Over/Under" in bet_name or ("Over" in bet_name and "Under" in bet_name))
            and not any(
                _x80 in bet_name
                for _x80 in ("Card", "Corner", "First Half", "Second Half",
                             "1st Half", "2nd Half", "Halves", "Team Total")
            )
        ):
            for v in values:
                val_str = str(v.get("value", ""))
                try:
                    # Parse "Over 2.5" -> 2.5
                    line_str = val_str.split("Over")[-1].strip()
                    line = float(line_str)
                    if abs(line - target_line) < 0.01:
                        odd = safe_float(str(v.get("odd", "")))
                        if odd and odd > 1.01:
                            result["over_line"] = target_line
                            result["over_odds"] = round(odd, 2)
                            result["over_implied"] = round(1.0 / odd, 3)
                except (ValueError, IndexError):
                    pass

        # v10.80: CARDS O/U (match total bookings: yellow=1, red=1)
        if bet_name in ("Cards Over/Under", "Cards Over Under", "Cards over/under",
                        "Total Cards Over/Under"):
            result["cards_lines"].extend(_collect_ou_lines(values))

        # v10.80: CORNERS O/U (match total)
        if bet_name in ("Corners Over/Under", "Corners Over Under", "Corners over/under",
                        "Total Corners Over/Under"):
            result["corners_lines"].extend(_collect_ou_lines(values))

        # Both Teams To Score
        if "Both Teams" in bet_name or "BTTS" in bet_name:
            for v in values:
                if "Yes" in str(v.get("value", "")):
                    odd = safe_float(str(v.get("odd", "")))
                    if odd and odd > 1.01:
                        result["btts_yes_odds"] = round(odd, 2)
                        result["btts_implied"] = round(1.0 / odd, 3)

        # Match Winner (1X2)
        if "Match Winner" in bet_name or "1X2" in bet_name or "Result" in bet_name:
            for v in values:
                label = str(v.get("value", ""))
                odd = safe_float(str(v.get("odd", "")))
                if odd and odd > 1.01:
                    if "Home" in label:
                        result["match_home_odds"] = round(odd, 2)
                    elif "Away" in label:
                        result["match_away_odds"] = round(odd, 2)
                    elif "Draw" in label:
                        result["match_draw_odds"] = round(odd, 2)

    has_odds = result["over_odds"] or result["btts_yes_odds"]
    if not has_odds:
        return None
    return result


def fetch_signal_odds(client: httpx.Client, fixture_id: int,
                      total_goals: int, game_minute: int | None = None,
                      for_message: bool = False) -> dict | None:
    """v10.36: Fetch odds at signal time for EV/ROI analysis.

    Captures odds PASSIVELY — the signal decision is already final.
    Costs 1 API credit per call.

    Captures:
    - Over (total_goals + 0.5) odds — "will another goal happen?"
    - BTTS Yes odds — "will both teams score?"
    - Match Winner odds — market view of match outcome

    v10.68: HARDENED CAPTURE. The Sep 4 audit found two data-quality
    holes: (a) 15/26 signals recorded NO odds at all (transient fetch
    failures, never retried); (b) 6/72 recorded prices were impossible
    for a live next-goal market (over 4.5 @ 23.00, over 6.5 @ 21.00 —
    real prices there are 4-6; all six 'won', inflating the paper P&L
    by ~1,400 EUR of phantom profit). Now:
      - ONE retry (ODDS_RETRY_DELAY s) when the first fetch returns
        nothing (quota-guarded: skipped at <=5 credits remaining);
      - SUSPECT-PRICE re-fetch: an over line implying <
        ODDS_SUSPECT_IMPLIED before ODDS_SUSPECT_MINUTE_MAX triggers
        one fresh /odds/live call — a sane fresh price REPLACES the
        suspect one; a confirmed suspect price is KEPT but flagged
        (odds_suspect=True) so P&L analysis can filter it;
      - source tagging: 'live' vs 'prematch_fallback' (a live-empty
        response falls back to pre-match /odds — pre-match totals
        prices are exactly the stale-high class seen Sep 4);
      - fetch rounds counted (odds_attempts).
    Extra cost: ~1 credit per failed-or-suspect signal (~20-25/day =
    0.3% of the 7,500 quota). Zero signal-logic changes.

    v10.78: for_message=True — the PRE-SEND fast capture whose result is
    embedded in the Telegram signal (the betting-decision block): ONE
    pass (live -> pre-match fallback), NO retry sleep, NO suspect
    re-fetch (the suspect price is kept + flagged, exactly the v10.77
    ledger discipline). Bounded ~1s so the signal leaves essentially as
    fast as before. The normal-path credit count is UNCHANGED vs v10.77
    (the same two calls: /odds/live then /odds); only a failed fast pass
    additionally triggers the full hardened fetch post-send for the
    ledger. The signal decision is ALWAYS already final when this runs —
    odds never influence it (v10.36 passive-capture principle).

    Returns dict with odds + implied probabilities, or None on failure.
    """
    if not ODDS_CAPTURE_ENABLED:
        return None
    if quota_remaining is not None and quota_remaining <= 5:
        log.info(" MKT SKIP: quota low, preserving credits")
        return None

    # v10.78: message mode = single pass, no sleeps (bounded latency)
    _max_attempts = 1 if for_message else 2
    attempts = 0
    result = None
    source = None
    fail_reason = "unknown"

    while attempts < _max_attempts:
        attempts += 1
        try:
            # v10.44s: /odds/live for in-play odds (better proxy for
            # signal-time market). Falls back to /odds (pre-match).
            data = api_get(client, "/odds/live", {"fixture": fixture_id})
            source = "live"
            result = _parse_signal_odds(data, total_goals)
            if result is None:
                data = api_get(client, "/odds", {"fixture": fixture_id})
                source = "prematch_fallback"
                result = _parse_signal_odds(data, total_goals)
                fail_reason = "no-markets" if result is None else None
            else:
                fail_reason = None
        except Exception as e:
            fail_reason = f"error: {e}"
            result = None
            source = None
        if result is not None:
            break
        # v10.68: single retry — only if the quota still allows it
        if quota_remaining is not None and quota_remaining <= 5:
            break
        if attempts < _max_attempts:
            log.info(
                f" MKT RETRY: F{fixture_id} — no odds on attempt 1 "
                f"({fail_reason}), one retry in {ODDS_RETRY_DELAY}s"
            )
            time.sleep(ODDS_RETRY_DELAY)

    if result is None:
        log.warning(
            f" MKT FETCH FAILED: fixture {fixture_id} after "
            f"{attempts} attempt(s) ({fail_reason})"
        )
        return None

    # v10.68: SUSPECT PRICE check on the over line (the P&L market).
    # A next-goal line priced implied < 12% before 80' is the stale /
    # pre-match-leftover class — refetch once for a fresh live price.
    suspect = False
    if (
        result.get("over_implied") is not None
        and result["over_implied"] < ODDS_SUSPECT_IMPLIED
        and (game_minute is None or game_minute < ODDS_SUSPECT_MINUTE_MAX)
    ):
        suspect = True
        log.warning(
            f" MKT SUSPECT: F{fixture_id} — O{result['over_line']} "
            f"@{result['over_odds']} (impl {result['over_implied']}) at "
            f"{game_minute}' — impossible for a live next-goal market, "
            f"refetching"
        )
        # v10.78: the re-fetch (3s sleep + 1-2 calls) is LEDGER-ONLY —
        # message mode skips it so the Telegram send is never delayed;
        # the suspect price is still flagged below either way.
        if (quota_remaining is None or quota_remaining > 5) and not for_message:
            time.sleep(ODDS_RETRY_DELAY)
            try:
                attempts += 1
                fresh_data = api_get(client, "/odds/live", {"fixture": fixture_id})
                fresh = _parse_signal_odds(fresh_data, total_goals)
                if fresh is not None and fresh.get("over_odds"):
                    if (
                        fresh.get("over_implied") is None
                        or fresh["over_implied"] >= ODDS_SUSPECT_IMPLIED
                    ):
                        log.info(
                            f" MKT SUSPECT REPLACED: F{fixture_id} — fresh "
                            f"O{fresh['over_line']} @{fresh['over_odds']} "
                            f"(impl {fresh['over_implied']}) replaces the "
                            f"suspect price"
                        )
                        result = fresh
                        source = "live"
                        suspect = False
                    else:
                        log.warning(
                            f" MKT SUSPECT KEPT: F{fixture_id} — fresh "
                            f"price also suspect (@{fresh['over_odds']}, "
                            f"impl {fresh['over_implied']}), keeping + "
                            f"flagging"
                        )
                # fresh parse without an over price: keep the original
                # suspect price + flag
            except Exception as e:
                log.warning(f" MKT SUSPECT REFETCH FAILED: F{fixture_id}: {e}")

    result["odds_source"] = source
    result["suspect"] = suspect
    result["attempts"] = attempts

    # v10.48: chosen source name NOT logged (chat-safe logs);
    # still stored in the outcome record as odds_bookmaker.
    log.info(
        f"  MKT: F{fixture_id} — "
        f"O{result['over_line']} @{result['over_odds']} "
        f"(impl {result['over_implied']}) "
        f"BTTS @{result['btts_yes_odds']} "
        f"[{', '.join(result['markets_available'][:5])}] "
        f"[{source}{' SUSPECT' if suspect else ''}, {attempts} try]"
    )
    return result


def _team_scores_empirical(minute: int) -> float:
    """v10.78: landed 'signaled team scored after signal' rate per minute
    band, measured from the user's OWN ledger (Sep 6-8: 98/164 = 59.8% in
    the 21-55 window; sub-bands 61.4% / 65.3% / 45.7%; 35.1% at 56'+).
    The signal team is empirically nearly the SOLE source of future goals
    (P(team)/P(any) ~= 0.98) — a lambda-split understates it. Bands are
    thin (n~40-60): RECALIBRATE after ~2 weeks of records.
    """
    if minute <= 35:
        return 0.614
    if minute <= 45:
        return 0.653
    if minute <= 55:
        return 0.457
    return 0.351


_TEAM_SCORES_FALLBACK = 0.575  # pooled 21-55 rate (98/164)

# ============================================================
# v10.80: CARDS & CORNERS MARKET BLOCK
# ============================================================
# The user's request: show an OVER/UNDER prediction for total cards and
# total corners INSIDE the Telegram signal, with the odds, and let the
# counts update as bookings/corners land. Zero extra API credits: the
# lines come from the SAME odds fetch (Bet365 prices both markets), the
# counts from the SAME batch statistics. v1 heuristic model, transparent
# multipliers, printed fair price + break-even rule — every lean is
# recorded (mkt_*) and graded at FT (mkt_*_ft_result) so ~2 weeks of
# labels tells us whether this market is actually beatable.

MKT_FULL_MATCH_MINUTES = 94.0   # 90 + ~4 stoppage on average
MKT_BASE_PACE_CARDS = 4.6 / 90.0      # avg total bookings/min (league blend, v1)
MKT_BASE_PACE_CORNERS = 10.5 / 90.0   # avg total corners/min (league blend, v1)
MKT_PRESS_BOOST_CORNERS = 1.08        # pressure-wave uplift on remaining corner pace
MKT_FOUL_BASE_PACE = 0.25             # fouls per minute baseline (both teams)
MKT_LEAN_OVER_P = 0.55                # P(over) needed to print LEAN: OVER
MKT_LEAN_UNDER_P = 0.45               # P(over) below this prints LEAN: UNDER
MKT_EDIT_MIN_INTERVAL = 45            # seconds between edits of one message

# (fid, tid) -> market-block live state for editMessageText updates
_market_block_live: dict[tuple[int, int], dict] = {}
_ft_market_stats_done: set[int] = set()   # fixtures whose FT corners/cards were stamped


def _poisson_p_at_least(lam: float, k: int) -> float:
    """v10.80: P(X >= k) for X ~ Poisson(lam). k<=0 -> 1.0."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    p_le = 0.0
    term = 1.0
    for i in range(k):
        if i > 0:
            term *= lam / float(i)
        p_le += term
    p_le *= math.exp(-lam)
    return max(0.0, min(1.0, 1.0 - p_le))


def _mkt_project_total(current: int | None, game_minute: int,
                       base_pace: float, boost: float) -> float:
    """v10.80: projected FT total = current + remaining * blended pace * boost.

    Pace blends the observed in-match rate with the league base (shrinkage
    toward the mean — a 0-corner half is low-pace, not proof of 0 forever).
    """
    remaining = max(0.0, MKT_FULL_MATCH_MINUTES - float(game_minute))
    cur = float(current or 0)
    if game_minute and game_minute > 0:
        observed = cur / float(game_minute)
        # 0.45/0.55 blend: the count AT a signal sits at wave-peak pace;
        # weighting the league base more tempers the projection (v1).
        pace = 0.45 * observed + 0.55 * base_pace
    else:
        pace = base_pace
    return cur + pace * remaining * boost


def _collect_ou_lines(values: list) -> list[tuple[float, float | None, float | None]]:
    """v10.80: collect Over/Under value pairs -> [(line, over_odd, under_odd)]."""
    pairs: dict[float, dict] = {}
    for v in values or []:
        val = str(v.get("value", "")).strip()
        odd = safe_float(str(v.get("odd", "")))
        if not odd or odd <= 1.01:
            continue
        for side in ("Over", "Under"):
            if val.lower().startswith(side.lower()):
                try:
                    line = float(val[len(side):].strip())
                except ValueError:
                    continue
                pairs.setdefault(line, {})[side] = round(odd, 2)
    return [(ln, d.get("Over"), d.get("Under")) for ln, d in sorted(pairs.items())]


def _pick_main_ou_line(lines: list) -> tuple:
    """v10.80: the main (most balanced) O/U line; falls back to any Over."""
    best = None
    for ln, ov, un in lines or []:
        if ov and un:
            bal = abs(ov - un)
            if best is None or bal < best[0]:
                best = (bal, ln, ov, un)
    if best:
        return best[1], best[2], best[3]
    for ln, ov, un in lines or []:
        if ov:
            return ln, ov, None
    return None, None, None


def _mkt_lean(p_over: float | None) -> str:
    if p_over is None:
        return "NEUTRAL"
    if p_over >= MKT_LEAN_OVER_P:
        return "OVER"
    if p_over <= MKT_LEAN_UNDER_P:
        return "UNDER"
    return "NEUTRAL"


def _build_market_block(
    game_minute: int,
    cards_now: int | None, fouls_now: int | None,
    corners_now: int, red_now: int | None,
    sig_losing: bool,
    cards_line: float | None, cards_ov: float | None, cards_un: float | None,
    corners_line: float | None, corners_ov: float | None, corners_un: float | None,
    team_name: str, book_name: str | None = None,
) -> tuple[str, dict]:
    """v10.82: CARDS & CORNERS block — COMPACT (user request, Sep 9).

    One line per market: live count vs line, the lean, and the single
    number the bet decision needs — the break-even price for the LEANED
    side (OVER -> 1/p, UNDER -> 1/(1-p); v10.80 printed the O-fair even
    when leaning UNDER, so the actionable number was missing).

    TEXT-ONLY change vs v10.80: the extras dict (mkt_* ledger fields),
    the record schema and the FT grading (mkt_*_ft_result) are
    byte-identical — projection / P(over) / fair-O still live in the
    ledger. Same single render path for the initial send and the live
    editMessageText updates -> the message stays byte-stable between
    edits (a redeploy resets _market_block_live anyway).
    Fully defensive: any missing input degrades that market's line,
    never the signal send. Also fixes a latent v10.80 crash (p_over==0
    with a line made fair=None hit an f-string -> block silently "").
    """
    extras: dict = {
        "mkt_cards_now": cards_now, "mkt_fouls_now": fouls_now,
        "mkt_corners_now": corners_now, "mkt_red_now": red_now,
        "mkt_cards_line": cards_line,
        "mkt_cards_over_odds": cards_ov, "mkt_cards_under_odds": cards_un,
        "mkt_corners_line": corners_line,
        "mkt_corners_over_odds": corners_ov, "mkt_corners_under_odds": corners_un,
        "mkt_cards_proj": None, "mkt_cards_p_over": None,
        "mkt_cards_fair_over": None, "mkt_cards_lean": "NEUTRAL",
        "mkt_corners_proj": None, "mkt_corners_p_over": None,
        "mkt_corners_fair_over": None, "mkt_corners_lean": "NEUTRAL",
    }

    def _compact(emoji: str, label: str, now: int | None,
                 line: float | None, p_over: float | None) -> str:
        if now is None:
            return f"{emoji} {label}: count n/a on this feed"
        if line is None:
            return f"{emoji} {label}: {now} \u00b7 no book line"
        if p_over is None:
            return f"{emoji} {label}: {now} vs line {line:g} \u00b7 no lean"
        lean = _mkt_lean(p_over)
        # v10.85: plain language — "need N more" = what the Over still
        # requires at FT; "max N more" = what the Under can absorb. The
        # break-even odds stay the actionable number they always were.
        _need = int(line) + 1 - (now or 0)
        _max = int(line) - (now or 0)
        _need_s = "already over" if _need <= 0 else f"need {_need} more"
        _max_s = "already under" if _max < 0 else f"max {_max} more"
        if lean == "OVER" and p_over > 1e-9:
            return (f"{emoji} {label}: {now} vs line {line:g} ({_need_s}) "
                    f"\u2014 bet Over only, odds \u2265 {1.0 / p_over:.2f}")
        if lean == "UNDER" and p_over < 1.0 - 1e-9:
            return (f"{emoji} {label}: {now} vs line {line:g} ({_max_s}) "
                    f"\u2014 bet Under only, odds \u2265 {1.0 / (1.0 - p_over):.2f}")
        return f"{emoji} {label}: {now} vs line {line:g} \u00b7 no lean, no bet"

    try:
        # v10.85: header says what "line" means — the book's pre-match total
        _bk = book_name or "book"
        parts = [
            "",
            f"\U0001f7e8\U0001f6a9 CARDS & CORNERS \u2014 {_bk} pre-match line, counts auto-update",
        ]

        # --- CARDS: v1 heuristic maths, ledger-identical to v10.80 ---
        lam_c = None
        p_c = None
        if cards_now is not None:
            if fouls_now is not None:
                heat = 1.0
                if game_minute and game_minute > 0:
                    fr = (fouls_now / float(game_minute)) / MKT_FOUL_BASE_PACE
                    heat *= 1.0 + 0.25 * max(0.0, min(1.4, fr - 1.0))
                if sig_losing:
                    heat *= 1.10
                if red_now:
                    heat *= 1.10
                heat = min(heat, 1.30)   # v1: cap the total card-heat
                lam_c = _mkt_project_total(cards_now, game_minute, MKT_BASE_PACE_CARDS, heat)
                if cards_line is not None:
                    p_c = _poisson_p_at_least(lam_c, int(cards_line) + 1)
            extras["mkt_cards_proj"] = round(lam_c, 1) if lam_c is not None else None
            extras["mkt_cards_p_over"] = round(p_c, 3) if p_c is not None else None
            if p_c is not None and p_c > 1e-9:
                extras["mkt_cards_fair_over"] = round(1.0 / p_c, 2)
            extras["mkt_cards_lean"] = _mkt_lean(p_c)
        parts.append(_compact("\U0001f7e8", "Cards", cards_now, cards_line, p_c))

        # --- CORNERS ---
        lam_n = _mkt_project_total(corners_now, game_minute, MKT_BASE_PACE_CORNERS,
                                   MKT_PRESS_BOOST_CORNERS)
        p_n = None
        if corners_line is not None:
            p_n = _poisson_p_at_least(lam_n, int(corners_line) + 1)
        extras["mkt_corners_proj"] = round(lam_n, 1)
        extras["mkt_corners_p_over"] = round(p_n, 3) if p_n is not None else None
        if p_n is not None and p_n > 1e-9:
            extras["mkt_corners_fair_over"] = round(1.0 / p_n, 2)
        extras["mkt_corners_lean"] = _mkt_lean(p_n)
        parts.append(_compact("\U0001f6a9", "Corners", corners_now, corners_line, p_n))

        return "\n".join(parts), extras
    except Exception:
        return "", {}


def edit_telegram(client: httpx.Client, msg_id: int, text: str) -> bool:
    """v10.80: edit our own message (market-block count refresh)."""
    try:
        resp = client.post(
            f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/editMessageText",
            json={"chat_id": TELEGRAM_CHAT_ID, "message_id": int(msg_id), "text": text},
        )
        try:
            body = resp.json()
        except Exception:
            body = {}
        if body.get("ok"):
            return True
        desc = str(body.get("description", ""))
        if "message is not modified" in desc.lower():
            return True
        log.debug(f"  v10.80 editMessageText rejected: {desc[:120]}")
        return False
    except Exception:
        return False


def _mkt_last_chunk_prefix(full_text: str, block: str) -> str:
    """v10.80: replicate send_telegram's split; return the last chunk's
    text minus the market block (the editable prefix)."""
    remaining = full_text
    while len(remaining) > 4000:
        split_at = remaining.rfind("\n", 0, 4000)
        if split_at <= 0:
            split_at = 4000
        remaining = remaining[split_at:]
    if block and remaining.endswith(block):
        return remaining[: len(remaining) - len(block)]
    return remaining


def _update_market_blocks(
    client: httpx.Client, fid: int, home_tid: int, away_tid: int,
    teams_data_by_id: dict, minute: int, home_goals: int, away_goals: int,
) -> None:
    """v10.80: live market-block edits. Runs every poll; touches Telegram
    ONLY when a fixture with a registered market block had a card/corner
    count change AND the 45s rate guard allows it."""
    now = time.time()
    keys = [k for k in _market_block_live if k[0] == fid]
    if not keys:
        return
    th = teams_data_by_id.get(home_tid) or {}
    ta = teams_data_by_id.get(away_tid) or {}
    if not th and not ta:
        return
    y_h, y_h_ok = get_stat_present(th, "yellow_cards")
    y_a, y_a_ok = get_stat_present(ta, "yellow_cards")
    c_h = safe_int(get_stat(th, "corner_kicks"))
    c_a = safe_int(get_stat(ta, "corner_kicks"))
    f_h, _ = get_stat_present(th, "fouls")
    f_a, _ = get_stat_present(ta, "fouls")
    r_h = safe_int(get_stat(th, "red_cards"))
    r_a = safe_int(get_stat(ta, "red_cards"))
    for key in keys:
        st = _market_block_live.get(key) or {}
        if now - (st.get("last_edit") or 0) < MKT_EDIT_MIN_INTERVAL:
            continue
        tid = st.get("tid")
        sig_is_home = tid == home_tid
        sig_goals = home_goals if sig_is_home else away_goals
        opp_goals = away_goals if sig_is_home else home_goals
        last = st.get("last_counts") or (None, None)
        cards_now = (y_h + y_a) if (y_h_ok and y_a_ok) else last[0]
        fouls_now = (f_h + f_a) if (f_h is not None and f_a is not None) else None
        corners_now = c_h + c_a
        red_now = r_h + r_a
        if (cards_now, corners_now) == (last[0], last[1]):
            continue  # no booking/corner change worth an edit
        try:
            block, _ = _build_market_block(
                game_minute=minute,
                cards_now=cards_now, fouls_now=fouls_now,
                corners_now=corners_now, red_now=red_now,
                sig_losing=(sig_goals < opp_goals),
                cards_line=st.get("cards_line"), cards_ov=st.get("cards_ov"),
                cards_un=st.get("cards_un"),
                corners_line=st.get("corners_line"), corners_ov=st.get("corners_ov"),
                corners_un=st.get("corners_un"),
                team_name=st.get("team_name", "?"), book_name=st.get("book"),
            )
            if not block:
                continue
            if edit_telegram(client, st["msg_id"], st["prefix"] + block):
                st["last_counts"] = (cards_now, corners_now)
                st["last_edit"] = now
            else:
                _market_block_live.pop(key, None)  # message gone — stop editing
        except Exception as _em80:
            log.debug(f"  v10.80 market edit failed: {_em80}")


def _stamp_ft_market_labels(client: httpx.Client, fixture: dict, fid: int) -> bool:
    """v10.80: FT labels for the cards/corners leans, once per fixture.

    Cards: from the events payload the resolver already fetched (zero
    credits) — bookings count yellow=1, red=1, exactly how the O/U line
    settles. Corners: one /fixtures/statistics call (quota-guarded).
    Every entry with a lean gets mkt_*_ft_result = HIT/MISS; all get the
    raw FT totals for calibration.
    """
    if fid in _ft_market_stats_done:
        return False
    entries = [e for e in signal_outcomes if e.get("fixture_id") == fid]
    if not entries:
        return False

    # --- cards from events (free — same call the resolver used) ---
    # None = events data unavailable (fetch failed) — NEVER stamped as 0;
    # an empty list = a real clean game with 0 bookings.
    cards = _card_events_cache.get(fid)
    if cards is None:
        try:
            fetch_goal_events(client, fid)
            cards = _card_events_cache.get(fid)
        except Exception:
            cards = None
    changed = False
    ft_cards_total = None
    if cards is not None:
        ft_cards_total = len(cards)
        for entry in entries:
            if entry.get("ft_cards_total") != ft_cards_total:
                entry["ft_cards_total"] = ft_cards_total
                changed = True
            if "ft_cards_list" not in entry:
                _lst = []
                for c in (cards or [])[:12]:
                    _is_red = ("Red" in (c.get("detail") or "")) or ("Second" in (c.get("detail") or ""))
                    _mm = c["minute"] + (c.get("minute_extra") or 0)
                    _lst.append(f"{'R' if _is_red else 'Y'}{_mm}' {c.get('player') or '?'}")
                entry["ft_cards_list"] = _lst
                changed = True
            if "cards_after_signal" not in entry:
                _after = [
                    c for c in (cards or [])
                    if c["minute"] + (c.get("minute_extra") or 0) > entry.get("game_minute", 0)
                ]
                entry["cards_after_signal"] = len(_after)
                changed = True

    # --- corners + yellow cross-check via one statistics call ---
    ft_c_h = ft_c_a = None
    if quota_remaining is None or quota_remaining > 5:
        try:
            data = api_get(client, "/fixtures/statistics", {"fixture": fid})
            resp = data.get("response", []) or []
            for blk in resp:
                tb = blk.get("team") or {}
                tmap = {}
                for st in blk.get("statistics") or []:
                    t = st.get("type")
                    if t:
                        tmap[str(t).strip()] = str(st.get("value") or 0)
                if tb.get("id") == fixture["teams"]["home"]["id"]:
                    ft_c_h = safe_int(get_stat(tmap, "corner_kicks"))
                elif tb.get("id") == fixture["teams"]["away"]["id"]:
                    ft_c_a = safe_int(get_stat(tmap, "corner_kicks"))
            _ft_market_stats_done.add(fid)
        except Exception as _e80:
            log.debug(f"  v10.80 FT corner stats failed for F{fid}: {_e80}")
    else:
        log.info(f"  v10.80 FT corners skipped for F{fid} (quota low) — cards still stamped")

    if ft_c_h is not None or ft_c_a is not None:
        ft_corners_total = (ft_c_h or 0) + (ft_c_a or 0)
        for entry in entries:
            if entry.get("ft_corners_total") != ft_corners_total:
                entry["ft_corners_total"] = ft_corners_total
                entry["ft_corners_home"] = ft_c_h
                entry["ft_corners_away"] = ft_c_a
                changed = True
            at_sig = entry.get("mkt_corners_now")
            if at_sig is not None and "corners_after_signal" not in entry:
                entry["corners_after_signal"] = max(0, ft_corners_total - at_sig)
                changed = True

    # --- grade the leans (line settles on FT TOTAL, the book's semantics) ---
    for entry in entries:
        line = entry.get("mkt_cards_line")
        lean = entry.get("mkt_cards_lean")
        if ft_cards_total is not None and line is not None and lean in ("OVER", "UNDER") \
                and "mkt_cards_ft_result" not in entry:
            hit = (ft_cards_total > line) if lean == "OVER" else (ft_cards_total < line)
            entry["mkt_cards_ft_result"] = "HIT" if hit else "MISS"
            changed = True
        line = entry.get("mkt_corners_line")
        lean = entry.get("mkt_corners_lean")
        if "ft_corners_total" in entry and line is not None and lean in ("OVER", "UNDER") \
                and "mkt_corners_ft_result" not in entry:
            ft_t = entry["ft_corners_total"]
            hit = (ft_t > line) if lean == "OVER" else (ft_t < line)
            entry["mkt_corners_ft_result"] = "HIT" if hit else "MISS"
            changed = True

    if changed:
        log.info(
            f"  v10.80 FT MARKET LABELS: F{fid} — cards {ft_cards_total if ft_cards_total is not None else 'n/a'}"
            f" / corners {(ft_c_h or 0) + (ft_c_a or 0) if (ft_c_h is not None or ft_c_a is not None) else 'n/a'}"
            f" stamped on {len(entries)} record(s)"
        )
    return changed


def _build_odds_value_block(
    odds_data: dict | None, goal_pred: dict | None, team_name: str,
    minute: int | None = None,
) -> tuple[str, dict]:
    """v10.78: ODDS & FAIR PRICE — the betting-decision block in every signal.

    The user's spec: 'I'd like to have the odds for each signal in the
    Telegram message so I can decide whether to bet on it or not.' The
    block shows the MARKET price captured at signal time (source-labeled
    LIVE vs pre-match reference, suspect prices flagged — the v10.77
    ledger honesty, now in the chat) beside the bot's own CALIBRATED fair
    prices / break-evens, so the decision happens in seconds: open the
    book, compare the live price against the printed break-even, done.

    Fair prices:
      - Over (total + 0.5): the calibrated any-goal line (v10.74's 50/50
        model/empirical blend) — the exact market the odds capture prices.
      - Team-to-score: the Poisson marginal 1 - exp(-lam_sig) blended 50/50
        with the minute-banded EMPIRICAL landed rate from the user's own
        ledger (the v10.74 calibration pattern applied to the team bet),
        capped at 0.98 x the any-goal line (team events are a subset of
        any-goal events — never display an impossible crossing).

    EV verdict ONLY against P&L-grade live prices (source='live' AND not
    suspect) — pre-match refs are display-only reference, exactly the
    v10.77 discipline (pre-match prices applied to in-play lines are the
    Sep-4/Sep-6 garbage class).

    Returns (message_block, extras): extras feeds the outcome record
    (pred_team_scores, pred_team_scores_cal, odds_ev_pct) so next week's
    calibration backtest can grade them. Defensive by design: ANY
    internal error returns ("", {}) — the signal send is sacred and must
    never fail because of this block.
    """
    try:
        if not goal_pred:
            return "", {}
        lam_sig = max(float(goal_pred.get("proj_xg_signal") or 0.0), 0.0)
        lam_opp = max(float(goal_pred.get("proj_xg_opponent") or 0.0), 0.0)
        cal_lines = goal_pred.get("over_lines_cal") or []
        if not cal_lines:
            return "", {}
        any_line = float(cal_lines[0][0])
        p_any = float(cal_lines[0][1])
        if p_any <= 0.0 or p_any >= 1.0:
            # degenerate (next-goal line decided / broken) — no fair price
            # makes sense; do not show a misleading 1.00/inf row
            return "", {}

        # --- team-to-score: model marginal blended with the user's own
        # empirical band rate, capped at the any-goal line ---
        p_team_model = 1.0 - math.exp(-lam_sig) if lam_sig > 0 else 0.0
        _p_emp = (
            _team_scores_empirical(int(minute))
            if minute is not None else _TEAM_SCORES_FALLBACK
        )
        p_team = min(0.5 * p_team_model + 0.5 * _p_emp, p_any * 0.98)
        be_any = 1.0 / p_any
        be_team = (1.0 / p_team) if p_team > 1e-9 else None

        # --- market side (source-honest labels) ---
        ev_pct = None
        _mkt_live = bool(
            odds_data
            and odds_data.get("over_odds")
            and odds_data.get("odds_source") == "live"
            and not odds_data.get("suspect")
        )
        # v10.85: plain language — the header names the BET, the fair
        # line carries the rule. Same numbers, same extras, fewer words.
        lines = [f"\n\U0001f4b0 BET \u2014 Over {any_line:.1f} goals"]
        if odds_data and odds_data.get("over_odds"):
            _src_tag = "LIVE" if _mkt_live else "pre-match ref"
            _sus = (
                " \u26a0\ufe0f impossible price — ignore"
                if odds_data.get("suspect") else ""
            )
            _mkt = (
                f"Book: O{odds_data['over_line']:.1f} "
                f"@{odds_data['over_odds']:.2f} "
                f"({odds_data.get('bookmaker') or '?'} \u00b7 {_src_tag}{_sus})"
            )
            if _mkt_live:
                ev_pct = (float(odds_data["over_odds"]) * p_any - 1.0) * 100.0
                if ev_pct >= 3.0:
                    _mkt += f" \u2192 VALUE +{ev_pct:.0f}%"
                elif ev_pct <= -3.0:
                    _mkt += " \u2192 no edge"
                else:
                    _mkt += " \u2192 thin edge"
            lines.append(_mkt)
        else:
            lines.append("Book: no price captured (quota/coverage)")

        # --- fair side (fair = break-even) — the rule rides the number,
        # one place, always: bet only at LIVE odds >= fair price ---
        _fair = (
            f"Fair: {be_any:.2f} \u2014 bet only at LIVE odds \u2265 {be_any:.2f} "
            f"({p_any:.0%} chance)"
        )
        if be_team is not None:
            _fair += (
                f"\n{team_name} to score (FT): fair {be_team:.2f} "
                f"({p_team:.0%})"
            )
        lines.append(_fair)

        # BTTS only while undecided AND captured
        _btts_odds = odds_data.get("btts_yes_odds") if odds_data else None
        _p_btts = goal_pred.get("p_btts")
        if (
            _btts_odds
            and _p_btts is not None
            and 0.0 < float(_p_btts) < 0.999
        ):
            lines.append(
                f"BTTS Yes: book @{float(_btts_odds):.2f} \u00b7 "
                f"fair {1.0 / float(_p_btts):.2f}"
            )

        # v10.85: the standalone rule footer is gone — the rule now rides
        # the fair line ("bet only at LIVE odds >= X") above.

        extras = {
            "pred_team_scores": round(p_team_model, 3),
            "pred_team_scores_cal": round(p_team, 3),
            "odds_ev_pct": round(ev_pct, 1) if ev_pct is not None else None,
        }
        return "\n".join(lines), extras
    except Exception:
        return "", {}


def resolve_with_goal_events(
    entry: dict, goal_events: list[dict], home_id: int, away_id: int,
    home_goals: int = None, away_goals: int = None, entry_kind: str = "signal",
) -> bool:
    """v10.14: Resolve a signal using precise goal event times.

    This replaces the old approach of comparing final score to goals_at_signal.
    With goal events, we know the EXACT minute each goal was scored,
    so we can precisely determine if a goal fell within 5/10/15 min windows.

    v10.66: EVENT-MINUTE CORRECTION. The live-tracking fallback resolves
    goals at the DETECTION minute (the poll's status.elapsed — biased late
    by feed lag + poll cadence: Botev Vratsa Sep 4 goals at 86'/88' were
    booked as "scored at 90' (+49')"). Fields already stamped by the live
    path are now RE-VERIFIED against the true event minutes in this FT pass:
    goal minutes corrected, 5/10/15m windows recomputed (a true in-window
    goal detected past its boundary flips MISS->HIT), a live HIT with no
    event AND no final-score confirmation flips to MISS (phantom/disallowed
    goal), and a live HIT the final score confirms but events lack is HELD
    with the live minute (coverage hole — never corrupt a true HIT).
    Zero extra credits: the /fixtures/events fetch already runs here.

    Args:
        entry: signal outcome dict (modified in place)
        goal_events: list of goal event dicts from fetch_goal_events
        home_id: home team API ID
        away_id: away team API ID
        home_goals: final home score (optional, enables phantom check)
        away_goals: final away score (optional, enables phantom check)
        entry_kind: log label — "signal" / "blocked" / "shadow"

    Returns:
        True if any field was updated (needs file rewrite)
    """
    updated = False
    sig_minute = entry["game_minute"]
    team_id = entry["team_id"]
    is_home = entry["is_home"]

    # Find the correct team ID from events
    # (entry stores team_id, events use team.id from API)
    event_team_id = home_id if is_home else away_id

    # Filter goals for this team that happened AFTER the signal minute
    team_goals_after = [
        g for g in goal_events
        if g["team_id"] == event_team_id and g["minute"] > sig_minute
        and g["detail"] != "Own Goal"  # own goals don't count for pressure team
    ]

    # Sort by minute
    team_goals_after.sort(key=lambda g: g["minute"])

    first_goal = team_goals_after[0] if team_goals_after else None

    # v10.79: SCORER STAMPING — keep the scorer identity that v10.78 threw
    # away. The Goal events in goal_events already carry player.name; this
    # runs inside the existing FT resolution pass (zero credits). Own goals
    # never reach here (team_goals_after filters them), so the stamp is a
    # genuine team scorer — exactly the ground truth the Top-SOT
    # 'scores next' hint is graded against. Idempotent: guarded by
    # inequality so re-resolution (v10.66 correction pass) never duplicates.
    if first_goal is not None:
        _sc79 = (first_goal.get("player") or "").strip()
        if entry.get("post_signal_scorer") != _sc79:
            entry["post_signal_scorer"] = _sc79
            entry["post_signal_scorer_minute"] = (
                first_goal["minute"] + (first_goal.get("minute_extra") or 0)
            )
            _all_scorers79 = []
            for _g79 in team_goals_after[:5]:
                _nm79 = (_g79.get("player") or "?").strip() or "?"
                _mn79 = _g79["minute"] + (_g79.get("minute_extra") or 0)
                _all_scorers79.append(f"{_nm79} {_mn79}'")
            entry["post_signal_scorers"] = _all_scorers79
            _named79 = {
                str(_p79.get("name", "")).strip()
                for _p79 in (entry.get("top_sot_players") or [])
            }
            entry["post_signal_scorer_is_named"] = (
                bool(_sc79) and _sc79 in _named79
            )
            updated = True
            if _sc79:
                _hint79 = (
                    " — WAS on the Top-SOT line"
                    if entry["post_signal_scorer_is_named"]
                    else " — not on the Top-SOT line"
                )
                log.info(
                    f"  SCORER STAMP: [{entry_kind}] {entry['team_name']} "
                    f"signal {sig_minute}' -> {_sc79} at "
                    f"{entry['post_signal_scorer_minute']}'{_hint79} "
                    f"[{entry.get('league', '?')}]"
                )

    if first_goal:
        first_goal_minute = first_goal["minute"] + first_goal["minute_extra"]
        mins_to_goal = first_goal_minute - sig_minute

        # v10.66: fields already stamped by the LIVE fallback? -> correct them
        # against the true event minutes. Otherwise keep the original pending
        # fill semantics (byte-identical to v10.65 for never-stamped entries).
        _live_fields_set = (
            entry.get("goal_minute_full") is not None
            or any(
                entry.get(k) is not None
                for k in ("outcome_5min", "outcome_10min", "outcome_15min")
            )
        )
        if _live_fields_set:
            _flips: list[str] = []
            _old_full_min = entry.get("goal_minute_full")
            for window, outcome_key, minute_key in [
                (5, "outcome_5min", "goal_minute_5"),
                (10, "outcome_10min", "goal_minute_10"),
                (15, "outcome_15min", "goal_minute_15"),
            ]:
                _true = "HIT" if mins_to_goal <= window else "MISS"
                _old = entry.get(outcome_key)
                if _old != _true:
                    _flips.append(f"{window}m {_old or 'pending'}->{_true}")
                    entry[outcome_key] = _true
                if _true == "HIT":
                    if (
                        entry.get(minute_key) is not None
                        and entry.get(minute_key) != first_goal_minute
                    ):
                        _flips.append(
                            f"goal_min_{window} {entry.get(minute_key)}'->"
                            f"{first_goal_minute}'"
                        )
                    entry[minute_key] = first_goal_minute
                elif entry.get(minute_key) is not None:
                    entry.pop(minute_key, None)
            if entry.get("outcome_full") != "HIT":
                _flips.append(f"full {entry.get('outcome_full') or 'pending'}->HIT")
                entry["outcome_full"] = "HIT"
            _minute_fixed = (
                _old_full_min is not None and _old_full_min != first_goal_minute
            )
            if _minute_fixed:
                entry["live_goal_minute_full"] = _old_full_min
            entry["goal_minute_full"] = first_goal_minute
            if _flips or _minute_fixed:
                entry["corrected_from_live"] = True
                updated = True
                _live_part = (
                    f"live detected {_old_full_min}' "
                    f"(+{_old_full_min - sig_minute}'), "
                    if _old_full_min is not None
                    else ""
                )
                log.info(
                    f"  OUTCOME CORRECTED: [{entry_kind}] {entry['team_name']} at "
                    f"{sig_minute}' — {_live_part}events say {first_goal_minute}' "
                    f"(+{mins_to_goal}')"
                    + (f" — flips: {', '.join(_flips)}" if _flips else "")
                    + f" [{entry.get('league', '?')}]"
                )
        else:
            # Pending entry (never live-stamped) — original fill semantics
            for window, outcome_key, minute_key in [
                (5, "outcome_5min", "goal_minute_5"),
                (10, "outcome_10min", "goal_minute_10"),
                (15, "outcome_15min", "goal_minute_15"),
            ]:
                if entry.get(outcome_key) is None:
                    if mins_to_goal <= window:
                        entry[outcome_key] = "HIT"
                        entry[minute_key] = first_goal_minute
                        updated = True
                    else:
                        entry[outcome_key] = "MISS"
                        updated = True

            # Full match outcome
            if entry.get("outcome_full") is None:
                entry["outcome_full"] = "HIT"
                entry["goal_minute_full"] = first_goal_minute
                log.info(
                    f"  GOAL-EVENT HIT: {entry['team_name']} scored at {first_goal_minute}' "
                    f"(signal at {sig_minute}', +{mins_to_goal}') GPS={entry.get('gps', '?')} "
                    f"[{entry.get('league', '?')}]"
                )
                updated = True
    else:
        # No goals after signal in the events feed.
        _live_hit = (
            entry.get("outcome_full") == "HIT"
            or any(
                entry.get(k) == "HIT"
                for k in ("outcome_5min", "outcome_10min", "outcome_15min")
            )
        )
        if _live_hit:
            # v10.66: live path saw a score rise the events feed does not
            # confirm. Verify against the FINAL score before touching fields.
            _final_team_goals = (
                (home_goals if entry.get("is_home") else away_goals)
                if (home_goals is not None or away_goals is not None)
                else None
            )
            _goals_at = entry.get("goals_at_signal")
            if _goals_at is None:
                _goals_at = entry.get("goals_at_shadow")
            if _goals_at is None:
                _goals_at = entry.get("goals_at_block")
            _old_live_min = entry.get("goal_minute_full")
            if (
                _final_team_goals is not None
                and _goals_at is not None
                and _final_team_goals <= _goals_at
            ):
                # Final score shows NO team goal beyond the baseline — the
                # live rise was a phantom (disallowed/reverted goal). Flip
                # HITs to MISS so win rates stay honest.
                for _k in (
                    "outcome_5min", "outcome_10min", "outcome_15min",
                    "outcome_full",
                ):
                    if entry.get(_k) == "HIT":
                        entry[_k] = "MISS"
                for _mk in (
                    "goal_minute_5", "goal_minute_10", "goal_minute_15",
                    "goal_minute_full",
                ):
                    entry.pop(_mk, None)
                entry["corrected_from_live"] = True
                entry["phantom_correction"] = True
                updated = True
                log.info(
                    f"  OUTCOME CORRECTED (phantom): [{entry_kind}] "
                    f"{entry['team_name']} at {sig_minute}' — live saw a goal at "
                    f"{_old_live_min}' but final score {home_goals}-{away_goals} "
                    f"shows none — MISS [{entry.get('league', '?')}]"
                )
            else:
                # Final score confirms a goal the events feed lacks (coverage
                # hole / slow FT feed), or the score is unavailable — keep the
                # live values, never corrupt a true HIT. Pending windows fill MISS.
                log.info(
                    f"  OUTCOME HELD: [{entry_kind}] {entry['team_name']} at "
                    f"{sig_minute}' — events feed lacks the goal the live path "
                    f"saw"
                    + (
                        f" (final score {home_goals}-{away_goals} confirms it; "
                        f"live minute {_old_live_min}' kept)"
                        if _final_team_goals is not None and _goals_at is not None
                        else " (final score unavailable — live minute kept)"
                    )
                    + f" [{entry.get('league', '?')}]"
                )

        # No goals after signal — fill any still-pending windows MISS
        for outcome_key in ("outcome_5min", "outcome_10min", "outcome_15min"):
            if entry.get(outcome_key) is None:
                entry[outcome_key] = "MISS"
                updated = True

        if entry.get("outcome_full") is None:
            entry["outcome_full"] = "MISS"
            log.info(
                f"  GOAL-EVENT MISS: {entry['team_name']} no goal after {sig_minute}' "
                f"GPS={entry.get('gps', '?')} [{entry.get('league', '?')}]"
            )
            updated = True

    # Mark as resolved
    if not entry.get("resolved"):
        entry["resolved"] = True
        entry["resolved_via"] = "goal_events"  # v10.14: track resolution method
        updated = True

    return updated


def check_signal_outcomes(fixture: dict, client: httpx.Client = None) -> None:
    """v10: Check pending signals for this fixture.

    Four-track system:
      outcome_5min:  HIT if goal within 5 game min
      outcome_10min: HIT if goal within 10 game min
      outcome_15min: HIT if goal within 15 game min
      outcome_full:  HIT if goal at any point before match ends
    Entry is 'resolved' only when ALL are decided (match must end).

    v10.14: When fixture is FT (finished), fetches /fixtures/events to get
    precise goal minutes for accurate window resolution. This replaces the
    old approach of comparing final score (which couldn't determine WHEN
    the goal was scored within the window).
    """
    global signal_outcomes
    fid = fixture["fixture"]["id"]
    status = fixture["fixture"]["status"]["short"]
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    home_goals = fixture["goals"]["home"] or 0
    away_goals = fixture["goals"]["away"] or 0
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]

    # v10.14: For FT fixtures, use goal events API for precise resolution
    is_finished = status not in LIVE_STATUSES
    goal_events = None
    if is_finished and client is not None:
        goal_events = fetch_goal_events(client, fid)

    any_updated = False
    for entry in signal_outcomes:
        if entry["fixture_id"] != fid or entry["resolved"]:
            continue

        # v10.14: Use goal events for precise resolution when available
        if goal_events is not None:
            if resolve_with_goal_events(
                entry, goal_events, home_id, away_id,
                home_goals=home_goals, away_goals=away_goals,
                entry_kind="signal",
            ):
                # v10.44g: Record actual total goals for prediction evaluation
                entry["pred_actual_total_goals"] = home_goals + away_goals
                _v10_74_fill_ft_actual(entry, home_goals, away_goals)  # v10.74
                # v10.49: Per-league Poisson calibration accumulation (logging-only)
                _update_poisson_calibration(entry)
                any_updated = True
            continue

        # --- Fallback: live tracking (no goal events, match still in progress) ---
        current_team_goals = home_goals if entry["is_home"] else away_goals
        goals_since_signal = current_team_goals - entry["goals_at_signal"]
        mins_since = minute - entry["game_minute"]

        if goals_since_signal > 0:
            mins_to_goal = minute - entry["game_minute"]

            # Check each time window
            for window, outcome_key, minute_key in [
                (5, "outcome_5min", "goal_minute_5"),
                (10, "outcome_10min", "goal_minute_10"),
                (15, "outcome_15min", "goal_minute_15"),
            ]:
                if entry.get(outcome_key) is None:
                    if mins_to_goal <= window:
                        entry[outcome_key] = "HIT"
                        entry[minute_key] = minute
                    elif mins_since >= window:
                        entry[outcome_key] = "MISS"

            if entry["outcome_full"] is None:
                entry["outcome_full"] = "HIT"
                entry["goal_minute_full"] = minute
                log.info(
                    f"  OUTCOME HIT: {entry['team_name']} scored at {minute}' "
                    f"(+{mins_to_goal}') GPS={entry.get('gps', '?')} [{entry['league']}]"
                )

            if status not in LIVE_STATUSES:
                entry["resolved"] = True
                entry["resolved_via"] = "live_score"  # v10.14: track resolution method
                # v10.44g: Record actual total goals for prediction evaluation
                entry["pred_actual_total_goals"] = home_goals + away_goals
                _v10_74_fill_ft_actual(entry, home_goals, away_goals)  # v10.74
                # v10.49: Per-league Poisson calibration accumulation (logging-only)
                _update_poisson_calibration(entry)
                any_updated = True

        else:
            # No goal yet — check windows
            for window, outcome_key in [(5, "outcome_5min"), (10, "outcome_10min"),
                                         (15, "outcome_15min")]:
                if entry.get(outcome_key) is None and mins_since >= window:
                    entry[outcome_key] = "MISS"

            if status not in LIVE_STATUSES:
                if entry["outcome_full"] is None:
                    entry["outcome_full"] = "MISS"
                    log.info(
                        f"  OUTCOME MISS: {entry['team_name']} no goal "
                        f"(signal at {entry['game_minute']}', GPS={entry.get('gps', '?')}, "
                        f"match ended {minute}') [{entry['league']}]"
                    )
                # Fill any remaining windows as MISS
                for outcome_key in ("outcome_5min", "outcome_10min", "outcome_15min"):
                    if entry.get(outcome_key) is None:
                        entry[outcome_key] = "MISS"
                entry["resolved"] = True
                entry["resolved_via"] = "live_score"
                # v10.44g: Record actual total goals for prediction evaluation
                entry["pred_actual_total_goals"] = home_goals + away_goals
                _v10_74_fill_ft_actual(entry, home_goals, away_goals)  # v10.74
                # v10.49: Per-league Poisson calibration accumulation (logging-only)
                _update_poisson_calibration(entry)
                any_updated = True

    # v10.80: fixture finished — stop editing its market blocks, and stamp
    # the FT market labels once (cards from the events the resolver already
    # fetched — free; corners via one quota-guarded statistics call).
    if is_finished and client is not None:
        for _k80 in [k for k in _market_block_live if k[0] == fid]:
            _market_block_live.pop(_k80, None)
        if _stamp_ft_market_labels(client, fixture, fid):
            any_updated = True

    # v10.14: Rewrite file if any entries were updated
    if any_updated:
        rewrite_outcomes_file()


def _log_outcome_block(label: str, resolved: list[dict]) -> None:
    """v10.19: Log a summary block for a given set of resolved entries."""
    if not resolved:
        return

    total = len(resolved)
    h5 = sum(1 for e in resolved if e.get("outcome_5min") == "HIT")
    h10 = sum(1 for e in resolved if e.get("outcome_10min") == "HIT")
    h15 = sum(1 for e in resolved if e.get("outcome_15min") == "HIT")
    hf = sum(1 for e in resolved if e.get("outcome_full") == "HIT")

    log.info(f"=== {label}: {total} signals ===")
    log.info(f"  5-min:   {h5}/{total} ({h5/total*100:.0f}%)")
    log.info(f"  10-min:  {h10}/{total} ({h10/total*100:.0f}%)")
    log.info(f"  15-min:  {h15}/{total} ({h15/total*100:.0f}%)")
    log.info(f"  Full:    {hf}/{total} ({hf/total*100:.0f}%)")

    # By tier
    for tier in ("CRITICAL", "EARLY WARNING"):
        tier_r = [e for e in resolved if e["tier"] == tier]
        if not tier_r:
            continue
        t_h5 = sum(1 for e in tier_r if e.get("outcome_5min") == "HIT")
        t_h15 = sum(1 for e in tier_r if e.get("outcome_15min") == "HIT")
        t_hf = sum(1 for e in tier_r if e.get("outcome_full") == "HIT")
        t_total = len(tier_r)
        avg_gps = sum(e.get("gps", 0) for e in tier_r) / t_total
        hits_15 = [e for e in tier_r if e.get("outcome_15min") == "HIT" and e.get("goal_minute_15")]
        avg_to_goal = None
        if hits_15:
            avg_to_goal = sum(e["goal_minute_15"] - e["game_minute"] for e in hits_15) / len(hits_15)
        goal_info = f", avg +{avg_to_goal:.0f}' to goal" if avg_to_goal else ""
        log.info(
            f"  {tier}: 5m {t_h5}/{t_total} 15m {t_h15}/{t_total} ({t_h15/t_total*100:.0f}%){goal_info} | "
            f"full {t_hf}/{t_total} ({t_hf/t_total*100:.0f}%) | avg GPS: {avg_gps:.0f}"
        )

    # GPS-triggered vs SOT-triggered
    gps_sigs = [e for e in resolved if e.get("gps_triggered")]
    sot_sigs = [e for e in resolved if not e.get("gps_triggered")]
    if gps_sigs:
        g_h5 = sum(1 for e in gps_sigs if e.get("outcome_5min") == "HIT")
        g_h15 = sum(1 for e in gps_sigs if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in gps_sigs if e.get("outcome_full") == "HIT")
        g_t = len(gps_sigs)
        avg_gps = sum(e.get("gps", 0) for e in gps_sigs) / g_t
        avg_sot = sum(e.get("sot", 0) for e in gps_sigs) / g_t
        log.info(
            f"  GPS-TRIGGERED (EARLY WARNING): 5m {g_h5}/{g_t} 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | "
            f"full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | avg GPS: {avg_gps:.0f}, avg SOT: {avg_sot:.1f}"
        )
    if sot_sigs:
        s_h15 = sum(1 for e in sot_sigs if e.get("outcome_15min") == "HIT")
        s_hf = sum(1 for e in sot_sigs if e.get("outcome_full") == "HIT")
        s_t = len(sot_sigs)
        log.info(
            f"  SOT-TRIGGERED (CRITICAL): 15min {s_h15}/{s_t} ({s_h15/s_t*100:.0f}%) | "
            f"full {s_hf}/{s_t} ({s_hf/s_t*100:.0f}%)"
        )

    # v10.27: By window tag (CORE / EARLY_OVERRIDE / LATE_OVERRIDE)
    for wt in ("CORE", "EARLY_OVERRIDE", "LATE_OVERRIDE"):
        group = [e for e in resolved if e.get("window_tag") == wt]
        if not group:
            continue
        w_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        w_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        w_t = len(group)
        avg_gps = sum(e.get("gps", 0) for e in group) / w_t
        avg_min = sum(e.get("game_minute", 0) for e in group) / w_t
        log.info(
            f"  {wt}: 15min {w_h15}/{w_t} ({w_h15/w_t*100:.0f}%) | "
            f"full {w_hf}/{w_t} ({w_hf/w_t*100:.0f}%) | "
            f"avg GPS: {avg_gps:.0f}, avg min: {avg_min:.0f}'"
        )

    # By GPS score range
    for range_label, range_filter in [
        ("GPS 55-64", lambda e: 55 <= e.get("gps", 0) < 65),
        ("GPS 65-74", lambda e: 65 <= e.get("gps", 0) < 75),
        ("GPS 75-84", lambda e: 75 <= e.get("gps", 0) < 85),
        ("GPS 85+", lambda e: e.get("gps", 0) >= 85),
    ]:
        group = [e for e in resolved if range_filter(e)]
        if not group:
            continue
        g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
        g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
        g_t = len(group)
        avg_accel = sum(e.get("accel_count", 0) for e in group) / g_t
        log.info(
            f"  {range_label}: 15min {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | "
            f"full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | avg accel: {avg_accel:.1f}"
        )


def log_outcome_summary() -> None:
    """v10: Log hit-rate summary for all resolved signals today.

    v10.19: Shows BOTH first-signal-only and all-signals blocks,
    matching the Telegram /stats output format.
    """
    # v10.44m: Dedup 1st signal by (fixture_id, team_id), not sig_num
    _seen = set()
    resolved_first = []
    for e in signal_outcomes:
        if not e.get("resolved"):
            continue
        key = (e.get("fixture_id"), e.get("team_id"))
        if key not in _seen:
            _seen.add(key)
            resolved_first.append(e)
    resolved_all = [e for e in signal_outcomes if e.get("resolved")]

    if not resolved_first and not resolved_all:
        return

    log.info("")

    if resolved_first:
        _log_outcome_block("SIGNAL OUTCOME SUMMARY (1st signal only)", resolved_first)

    # If there are 2nd/3rd signals, show all-signals block too
    if resolved_all and len(resolved_all) != len(resolved_first):
        log.info("")
        _log_outcome_block("SIGNAL OUTCOME SUMMARY (all signals)", resolved_all)


def resolve_stale_outcomes(client: httpx.Client) -> int:
    """v10.11: Resolve pending outcomes from finished fixtures.

    When the bot sleeps before matches end, pending outcomes stay unresolved.
    This function fetches finished fixtures by ID and resolves them.
    Called on startup and periodically during idle.
    Returns number of newly resolved outcomes.
    """
    global signal_outcomes, _resolve_live_fids, _resolve_live_ts
    pending = [e for e in signal_outcomes if not e.get("resolved")]
    if not pending:
        return 0

    # v10.44d-fix: Log resolution attempt
    _fids = list(set(e["fixture_id"] for e in pending))
    log.info(f"  Resolving {len(pending)} pending outcome(s) across {len(_fids)} fixture(s)...")
    # v10.63: remember which fixtures this pass saw as still LIVE — the
    # startup retry uses it to skip the pointless 30s wait when every
    # pending outcome is on an in-progress game.
    # v10.74: ...and the BOOT-PATH midnight guard uses it (with the
    # timestamp below as its freshness anchor) to hold the first sleep
    # until those fixtures verify FT.
    _resolve_live_fids = set()
    _resolve_live_ts = time.time()

    # Collect unique fixture IDs (batch up to 20 per API call)
    fixture_ids = list(set(e["fixture_id"] for e in pending))
    resolved_count = 0

    # v10.44d-fix: Track which fixture IDs had signals
    _signaled_fids = set(e["fixture_id"] for e in signal_outcomes if e.get("tier") != "NON-SIGNAL")
    for i in range(0, len(fixture_ids), BATCH_SIZE_LIMIT):
        batch = fixture_ids[i:i + BATCH_SIZE_LIMIT]
        ids_param = "-".join(str(fid) for fid in batch)
        try:
            data = api_get(client, "/fixtures", {"ids": ids_param})
            fixtures = data.get("response", [])
            for f in fixtures:
                if f["fixture"]["status"]["short"] in LIVE_STATUSES:
                    _resolve_live_fids.add(f["fixture"]["id"])
                check_signal_outcomes(f, client)
                check_blocked_outcomes(f, client)  # v10.49: false-negative resolution
                check_fastlane_shadow(f, client)  # v10.50: fast-lane shadow resolution
                # v10.44d-fix: Record non-signal fixtures for ML dataset
                _fid = f["fixture"]["id"]
                _fstatus = f["fixture"]["status"]["short"]
                if _fstatus not in LIVE_STATUSES and _fid not in _signaled_fids and _fid not in signaled_fixtures:
                    record_non_signal_fixture(f)
        except Exception as e:
            log.warning(f"  Stale outcome resolution failed for batch {batch}: {e}")
            continue

    # Check how many got resolved
    still_pending = [e for e in signal_outcomes if not e.get("resolved")]
    resolved_count = len(pending) - len(still_pending)
    if resolved_count > 0:
        log.info(f"Resolved {resolved_count} stale outcome(s), {len(still_pending)} still pending")
    elif still_pending:
        log.info(f"{len(still_pending)} outcome(s) still pending (fixtures may not be finished yet)")

    return resolved_count


# v10.47: duplicate load_all_outcomes() definition removed (was byte-identical
# to the v10.19.3 definition near module top — Python silently kept whichever
# came last, making the other dead code and confusing maintenance).


def _format_stats_block(entries: list[dict], label: str) -> list[str]:
    """v10.18: Format a stats block for a given set of entries."""
    resolved = [e for e in entries if e.get("resolved")]
    pending = [e for e in entries if not e.get("resolved")]
    lines = []
    lines.append(f"📊 {label} ({len(entries)} total)")
    lines.append("")

    if resolved:
        total = len(resolved)
        h5 = sum(1 for e in resolved if e.get("outcome_5min") == "HIT")
        h10 = sum(1 for e in resolved if e.get("outcome_10min") == "HIT")
        h15 = sum(1 for e in resolved if e.get("outcome_15min") == "HIT")
        hf = sum(1 for e in resolved if e.get("outcome_full") == "HIT")

        lines.append(f"✅ Resolved: {total}")
        # v10.28: Full WR is PRIMARY KPI — show first and prominently
        lines.append(f"  ⭐ Full WR: {hf}/{total} ({hf/total*100:.0f}%)")
        lines.append(f"  Timing: 5m {h5}/{total} | 10m {h10}/{total} | 15m {h15}/{total}")
        lines.append("")

        # By tier
        for tier in ("CRITICAL", "EARLY WARNING"):
            tier_r = [e for e in resolved if e.get("tier") == tier]
            if not tier_r:
                continue
            t = len(tier_r)
            t_h15 = sum(1 for e in tier_r if e.get("outcome_15min") == "HIT")
            t_hf = sum(1 for e in tier_r if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in tier_r) / t
            lines.append(f"  {tier}: full {t_hf}/{t} ({t_hf/t*100:.0f}%) | 15m {t_h15}/{t} ({t_h15/t*100:.0f}%) | avg GPS {avg_gps:.0f}")

        lines.append("")

        # By trigger type — v10.28: full WR first
        gps_sigs = [e for e in resolved if e.get("gps_triggered")]
        sot_sigs = [e for e in resolved if not e.get("gps_triggered")]
        if gps_sigs:
            g_t = len(gps_sigs)
            g_h15 = sum(1 for e in gps_sigs if e.get("outcome_15min") == "HIT")
            g_hf = sum(1 for e in gps_sigs if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in gps_sigs) / g_t
            avg_sot = sum(e.get("sot", 0) for e in gps_sigs) / g_t
            lines.append(f"  GPS-triggered (EW): full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | avg GPS {avg_gps:.0f} SOT {avg_sot:.1f}")
        if sot_sigs:
            s_t = len(sot_sigs)
            s_h15 = sum(1 for e in sot_sigs if e.get("outcome_15min") == "HIT")
            s_hf = sum(1 for e in sot_sigs if e.get("outcome_full") == "HIT")
            lines.append(f"  SOT-triggered (CRITICAL): full {s_hf}/{s_t} ({s_hf/s_t*100:.0f}%) | 15m {s_h15}/{s_t} ({s_h15/s_t*100:.0f}%)")

        lines.append("")

        # v10.27/28: By window tag — full WR first
        for wt_label, wt_key in [("CORE (21-60')", "CORE"), ("EARLY OVERRIDE (<21')", "EARLY_OVERRIDE"), ("LATE OVERRIDE (61'+)", "LATE_OVERRIDE")]:
            group = [e for e in resolved if e.get("window_tag") == wt_key]
            if not group:
                continue
            w_t = len(group)
            w_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
            w_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in group) / w_t
            lines.append(f"  {wt_label}: full {w_hf}/{w_t} ({w_hf/w_t*100:.0f}%) | 15m {w_h15}/{w_t} ({w_h15/w_t*100:.0f}%) | avg GPS {avg_gps:.0f}")

        lines.append("")

        # By GPS range
        for range_label, range_filter in [
            ("GPS 55-64", lambda e: 55 <= e.get("gps", 0) < 65),
            ("GPS 65-74", lambda e: 65 <= e.get("gps", 0) < 75),
            ("GPS 75-84", lambda e: 75 <= e.get("gps", 0) < 85),
            ("GPS 85+", lambda e: e.get("gps", 0) >= 85),
        ]:
            group = [e for e in resolved if range_filter(e)]
            if not group:
                continue
            g_t = len(group)
            g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
            g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
            lines.append(f"  {range_label}: full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%)")

        lines.append("")

        # By minute range
        for min_label, min_filter in [
            ("21-35'", lambda e: 21 <= e.get("game_minute", 0) <= 35),
            ("36-45'", lambda e: 36 <= e.get("game_minute", 0) <= 45),
            ("46-55'", lambda e: 46 <= e.get("game_minute", 0) <= 55),
            ("56-61'", lambda e: 56 <= e.get("game_minute", 0) <= 61),
        ]:
            group = [e for e in resolved if min_filter(e)]
            if not group:
                continue
            g_t = len(group)
            g_h5 = sum(1 for e in group if e.get("outcome_5min") == "HIT")
            g_h15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
            g_hf = sum(1 for e in group if e.get("outcome_full") == "HIT")
            avg_gps = sum(e.get("gps", 0) for e in group) / g_t
            lines.append(f"  {min_label}: full {g_hf}/{g_t} ({g_hf/g_t*100:.0f}%) | 15m {g_h15}/{g_t} ({g_h15/g_t*100:.0f}%) | 5m {g_h5}/{g_t} | avg GPS {avg_gps:.0f}")

        lines.append("")

        # v10.19: GPS x minute cross-tab (first signals only)
        # Rows = GPS ranges, Columns = minute buckets
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
        # Check if any cell has data
        has_cross_data = False
        for _, gfilter in gps_rows:
            for _, mfilter in minute_buckets:
                if [e for e in resolved if gfilter(e) and mfilter(e)]:
                    has_cross_data = True
                    break
            if has_cross_data:
                break
        if has_cross_data:
            lines.append("GPS x Minute (full WR):")
            # Header
            header = "             " + "  ".join(f"{ml:>8}" for ml, _ in minute_buckets)
            lines.append(header)
            for glabel, gfilter in gps_rows:
                row_parts = [f"{glabel:<12}"]
                for _, mfilter in minute_buckets:
                    cell = [e for e in resolved if gfilter(e) and mfilter(e)]
                    if cell:
                        ch = sum(1 for e in cell if e.get("outcome_full") == "HIT")
                        ct = len(cell)
                        row_parts.append(f"{ch}/{ct}({ch/ct*100:.0f}%)")
                    else:
                        row_parts.append(f"{'---':>8}")
                lines.append("  ".join(row_parts))
            lines.append("")

        # Last 10 resolved signals detail — v10.28: full outcome first, show recency_ratio
        recent = resolved[-10:]
        lines.append("📋 Last signals:")
        for e in recent:
            of_ = e.get("outcome_full", "?")
            o15 = e.get("outcome_15min", "?")
            icon = "✅" if of_ == "HIT" else "❌"
            gm = e.get("goal_minute_full") or e.get("goal_minute_15", "")
            gm_str = f" goal@{gm}'" if gm else ""
            trigger = "GPS" if e.get("gps_triggered") else "SOT"
            rr = e.get("recency_ratio")
            rr_str = f" RR:{rr:.2f}" if rr is not None else ""
            lines.append(f"  {icon} {e.get('team_name','?')} | {e.get('league','?')} | GPS {e.get('gps','?')} | SOT {e.get('sot','?')} | {trigger} | full:{of_} 15m:{o15}{gm_str}{rr_str}")

        # v10.28: Recency ratio analysis (if data available)
        with_rr = [e for e in resolved if e.get("recency_ratio") is not None]
        if len(with_rr) >= 5:
            lines.append("")
            lines.append("🔄 Recency Analysis (fresh vs accumulated pressure):")
            # Bin by recency_ratio: low (accumulated) vs high (fresh)
            mid = 0.3
            low_rr = [e for e in with_rr if e.get("recency_ratio", 0) < mid]
            high_rr = [e for e in with_rr if e.get("recency_ratio", 0) >= mid]
            for label, group in [(f"RR <{mid} (accumulated)", low_rr), (f"RR ≥{mid} (fresh pressure)", high_rr)]:
                if len(group) < 2:
                    continue
                gt = len(group)
                ghf = sum(1 for e in group if e.get("outcome_full") == "HIT")
                gh15 = sum(1 for e in group if e.get("outcome_15min") == "HIT")
                avg_rr = sum(e.get("recency_ratio", 0) for e in group) / gt
                avg_gps = sum(e.get("gps", 0) for e in group) / gt
                avg_accel = sum(e.get("accel_count", 0) for e in group) / gt
                lines.append(f"  {label}: full {ghf}/{gt} ({ghf/gt*100:.0f}%) | 15m {gh15}/{gt} ({gh15/gt*100:.0f}%) | avg RR {avg_rr:.2f} GPS {avg_gps:.0f} accel {avg_accel:.1f}")
    else:
        lines.append("No resolved signals yet.")

    if pending:
        lines.append("")
        lines.append(f"⏳ Pending: {len(pending)} (awaiting match end or resolution)")

    return lines


def format_outcome_stats(entries: list[dict]) -> str:
    """v10.18: Format outcome statistics showing BOTH counting methods.

    Shows two blocks:
    1. 1st SIGNAL ONLY (per team per fixture) — the fair judge
    2. ALL SIGNALS (every signal fired) — for comparison

    v10.44m fix: 1st signal dedup uses (fixture_id, team_id, signal_time) key
    instead of sig_num field, which resets to 1 after redeploy causing
    false duplicates.
    """
    if not entries:
        return "No signal data yet."

    # v10.44m: Dedup by actual first signal per (fixture_id, team_id)
    # using signal_time as tiebreaker (earliest = first)
    _seen_pairs: dict[tuple, dict] = {}
    for e in entries:
        key = (e.get("fixture_id"), e.get("team_id"))
        if key not in _seen_pairs:
            _seen_pairs[key] = e
        else:
            # Keep the one with earlier signal_time
            existing_st = _seen_pairs[key].get("signal_time", 0) or 0
            current_st = e.get("signal_time", 0) or 0
            if current_st > 0 and (existing_st == 0 or current_st < existing_st):
                _seen_pairs[key] = e
    first_sig = list(_seen_pairs.values())
    all_sig = entries

    # If both sets are identical (no duplicates), show single block
    if len(first_sig) == len(all_sig):
        lines = _format_stats_block(first_sig, "SIGNAL PERFORMANCE")
        return "\n".join(lines)

    # Show both blocks for comparison
    lines = _format_stats_block(first_sig, "SIGNAL PERFORMANCE — 1st signal only")
    lines.append("")
    lines.append("─" * 20)
    lines.append("")
    lines_all = _format_stats_block(all_sig, "SIGNAL PERFORMANCE — all signals")
    lines.extend(lines_all)

    return "\n".join(lines)


def send_end_of_day_summary(client: httpx.Client, days: int = 1) -> None:
    """v10.13: Send end-of-day win rate summary via Telegram.

    v10.13 fix: Loads from JSONL FILE (not in-memory) because signal_outcomes
    gets cleared before this runs. This was the root cause of missing EOD stats.

    Args:
        days: How many days back to include (1=today, 3=past 3 days)
    Called once when the bot exits active hours (all matches ended).
    Reuses format_outcome_stats() so the output matches /stats.
    """
    global eod_summary_sent_date

    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
    # For auto EOD (days=1), guard against duplicate sends
    if days == 1 and eod_summary_sent_date == today_str:
        return  # Already sent today
    # v10.16: Also check persisted marker (survives restarts)
    if days == 1 and _load_eod_sent_date() == today_str:
        log.info(f"v10.16: EOD already sent for {today_str} (persisted marker), skipping")
        eod_summary_sent_date = today_str  # sync in-memory
        return

    # v10.13: ALWAYS load from file — in-memory may be cleared
    all_entries = load_all_outcomes()
    if not all_entries:
        log.info("v10.13: No outcome data in file, skipping EOD summary")
        return

    # Filter to requested day range
    # v10.16: Use days=2 window to catch signals from the date that just ended.
    # EOD typically fires around midnight when signals carry yesterday's date.
    effective_days = max(days, 2)
    cutoff_date = (datetime.now(BULGARIA_TZ) - timedelta(days=effective_days - 1)).strftime("%Y-%m-%d")
    filtered = []
    for e in all_entries:
        sig_date = e.get("signal_clock", "")[:10]
        if sig_date >= cutoff_date:
            filtered.append(e)

    if not filtered:
        log.info(f"v10.16: No signals in last {effective_days} day(s), skipping EOD summary")
        return

    resolved = [e for e in filtered if e.get("resolved")]
    if not resolved:
        log.info(f"v10.16: {len(filtered)} signals but none resolved yet")
        return

    pending = [e for e in filtered if not e.get("resolved")]
    day_label = "TODAY" if days == 1 else f"PAST {effective_days} DAYS"
    log.info(f"v10.13: Sending EOD summary {day_label} ({len(resolved)} resolved, {len(pending)} pending)")

    stats_text = format_outcome_stats(filtered)
    sent = send_telegram(client, stats_text)
    if sent and days == 1:
        eod_summary_sent_date = today_str
        _save_eod_sent_date(today_str)
        log.info("v10.16: End-of-day summary sent (persisted)")
    elif not sent:
        log.warning("v10.13: End-of-day Telegram send FAILED")


def _restore_from_telegram_file(client: httpx.Client, file_id: str, filename: str) -> int | None:
    """v10.44m: Download a .jsonl file from Telegram and restore into local data.

    Auto-detects whether it's signals or polls data based on filename and content.
    Deduplicates by entry ID (fixture_id+team_id+signal_time for signals, ts+fid+tid for polls).
    Returns number of new entries restored, 0 if all duplicates, None if error.
    """
    global signal_outcomes

    # 1. Download file from Telegram
    try:
        file_resp = client.post(
            f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getFile",
            json={"file_id": file_id},
            timeout=10.0,
        )
        if file_resp.status_code != 200 or not file_resp.json().get("ok"):
            log.warning(f"/restore: getFile failed for {filename}")
            return None
        file_path = file_resp.json().get("result", {}).get("file_path", "")
        if not file_path:
            log.warning(f"/restore: no file_path for {filename}")
            return None

        dl_resp = client.get(
            f"{TELEGRAM_API}/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}",
            timeout=30.0,
        )
        if dl_resp.status_code != 200:
            log.warning(f"/restore: download failed for {filename}: {dl_resp.status_code}")
            return None
        raw_bytes = dl_resp.content

        # v10.44n: Auto-decompress .jsonl.gz files
        if filename.endswith(".gz"):
            try:
                raw_bytes = gzip.decompress(raw_bytes)
                log.info(f"/restore: decompressed {filename} ({len(dl_resp.content)} -> {len(raw_bytes)} bytes)")
            except Exception as e:
                log.warning(f"/restore: failed to decompress {filename}: {e}")
                return None

        content = raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        log.warning(f"/restore: error downloading {filename}: {e}")
        return None

    # 2. Parse lines
    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    if not lines:
        log.info(f"/restore: {filename} is empty")
        return 0

    # 3. Detect file type from filename, then from content
    # v10.44n: Strip .gz suffix for type detection
    _base_name = filename.lower()
    if _base_name.endswith(".gz"):
        _base_name = _base_name[:-3]
    is_signals = "signals" in _base_name or "outcome" in _base_name
    is_polls = "polls" in _base_name

    if not is_signals and not is_polls:
        # Auto-detect from first valid line
        for line in lines[:3]:
            try:
                obj = json.loads(line)
                if "signal_time" in obj or "outcome_full" in obj or "tier" in obj:
                    is_signals = True
                    break
                elif "ts" in obj and "gps" in obj and "fixture_id" in obj:
                    is_polls = True
                    break
            except Exception:
                continue

    if not is_signals and not is_polls:
        log.warning(f"/restore: cannot detect type of {filename}")
        return None

    # 4. Load existing IDs for dedup
    existing_ids = set()
    target_file = OUTCOMES_FILE if is_signals else POLL_DATA_FILE

    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if is_signals:
                        # Dedup key: fixture_id + team_id + signal_time
                        _key = (obj.get("fixture_id"), obj.get("team_id"), obj.get("signal_time"))
                    else:
                        # Dedup key: ts + fixture_id + team_id
                        _key = (obj.get("ts"), obj.get("fixture_id"), obj.get("team_id"))
                    existing_ids.add(_key)
                except Exception:
                    continue

    # 5. Parse and dedup new entries
    new_entries = []
    for line in lines:
        try:
            obj = json.loads(line)
            if is_signals:
                _key = (obj.get("fixture_id"), obj.get("team_id"), obj.get("signal_time"))
            else:
                _key = (obj.get("ts"), obj.get("fixture_id"), obj.get("team_id"))
            if _key not in existing_ids and None not in _key:
                new_entries.append(line)
                existing_ids.add(_key)
        except Exception:
            continue

    if not new_entries:
        log.info(f"/restore: {filename} — 0 new entries (all duplicates)")
        return 0

    # 6. Append to file
    try:
        with open(target_file, "a") as f:
            for entry in new_entries:
                f.write(entry + "\n")
    except Exception as e:
        log.warning(f"/restore: failed to write to {target_file}: {e}")
        return None

    # 7. If signals, also load into memory
    if is_signals:
        _existing_mem_ids = set()
        for _e in signal_outcomes:
            _existing_mem_ids.add((_e.get("fixture_id"), _e.get("team_id"), _e.get("signal_time")))
        for line in new_entries:
            try:
                obj = json.loads(line)
                _key = (obj.get("fixture_id"), obj.get("team_id"), obj.get("signal_time"))
                if _key not in _existing_mem_ids:
                    signal_outcomes.append(obj)
                    _existing_mem_ids.add(_key)
            except Exception:
                continue

    _type_label = "signals" if is_signals else "polls"
    log.info(f"/restore: {filename} -> {len(new_entries)} new {_type_label} appended to {target_file}")
    return len(new_entries)


def _format_ml_scoreboard(sig_entries: list[dict], blocked_entries: list[dict], days: int = 7) -> str:
    """v10.59: GPS vs ML scoreboard — who reads incoming goals better?

    READ-ONLY: computed from saved records (signal_outcomes + blocked
    candidates). Records only carry 'ml_score' from v10.59 on, so older
    data is excluded automatically — collection starts at the v10.59
    deploy. The ML model itself stays frozen; this NEVER changes signal
    decisions, it only measures.

    Three questions it answers (all 0 extra credits):
      1. On signals we SENT: did GPS or ML separate goal-follows from
         goal-doesn't-follow better? (winner/loser average gap)
      2. On signals the gates BLOCKED where a goal came anyway (false
         negatives): would ML (score >= 50) have warned?
      3. How often does ML simply agree with sent signals? (sanity)
    """
    ML_AGREE = 50  # ML >= 50 counts as "ML would have warned"
    cutoff = (datetime.now(BULGARIA_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")

    def _in_window(e: dict, key: str) -> bool:
        c = e.get(key) or ""
        return c[:10] >= cutoff

    sig_all = [e for e in sig_entries if e.get("ml_score") is not None]
    sig_w = [e for e in sig_all if _in_window(e, "signal_clock")]
    blk_all = [e for e in blocked_entries if e.get("ml_score") is not None]
    blk_w = [e for e in blk_all if _in_window(e, "blocked_clock")]

    if not sig_all and not blk_all:
        return (
            "\U0001f9e0 ML vs GPS SCOREBOARD\n\n"
            "No ML scores recorded yet.\n"
            "v10.59 started saving them — ask again after a few match days.\n"
            "(If this stays empty for days, the model file is missing.)"
        )

    lines: list[str] = []
    lines.append(f"\U0001f9e0 ML vs GPS SCOREBOARD (last {days} days)")
    lines.append("")
    lines.append("\U0001f4ca Coverage")
    lines.append(f"  Sent signals with ML opinion: {len(sig_w)} ({days}d) / {len(sig_all)} all-time")
    lines.append(f"  Blocked moments with ML opinion: {len(blk_w)} ({days}d) / {len(blk_all)} all-time")

    # --- 1. Sent signals: who separated winners from losers? ---
    resolved_w = [e for e in sig_w if e.get("resolved")]
    n_res = len(resolved_w)
    if n_res:
        lines.append("")
        lines.append(f"\u2705 On signals we SENT ({n_res} resolved)")
        hits = [e for e in resolved_w if e.get("outcome_15min") == "HIT"]
        misses = [e for e in resolved_w if e.get("outcome_15min") != "HIT"]
        if hits and misses:
            gps_h = sum(e.get("gps", 0) for e in hits) / len(hits)
            gps_m = sum(e.get("gps", 0) for e in misses) / len(misses)
            ml_h = sum(e.get("ml_score", 0) for e in hits) / len(hits)
            ml_m = sum(e.get("ml_score", 0) for e in misses) / len(misses)
            gps_gap = gps_h - gps_m
            ml_gap = ml_h - ml_m
            lines.append(f"  Goal followed (15m): {len(hits)} — GPS avg {gps_h:.0f} | ML avg {ml_h:.0f}")
            lines.append(f"  No goal (15m): {len(misses)} — GPS avg {gps_m:.0f} | ML avg {ml_m:.0f}")
            lines.append(f"  Winner/loser gap: GPS {gps_gap:.0f} pts | ML {ml_gap:.0f} pts")
            if n_res < 20:
                verdict = "Collecting — too few for a verdict (need ~20 resolved)."
            elif ml_gap > gps_gap + 2:
                verdict = "ML separates winners from losers BETTER than GPS so far."
            elif gps_gap > ml_gap + 2:
                verdict = "GPS still reads the difference better than ML."
            else:
                verdict = "Too close to call — both read it about the same."
            lines.append(f"  \u2192 {verdict}")
        else:
            lines.append(f"  {len(hits)} hits / {len(misses)} misses — need both for a comparison.")
        agree = sum(1 for e in resolved_w if e.get("ml_score", 0) >= ML_AGREE)
        lines.append(f"  \U0001f91d ML agreed (score \u2265{ML_AGREE}): {agree}/{n_res} ({agree / n_res * 100:.0f}%)")

    # --- 2. Blocked moments that led to goals (GPS misses) ---
    blk_res_w = [e for e in blk_w if e.get("resolved")]
    fn = [e for e in blk_res_w if e.get("outcome_15min") == "HIT"]
    if blk_res_w:
        lines.append("")
        lines.append(f"\U0001f6a8 Goals the gates blocked ({len(fn)} of {len(blk_res_w)} led to a goal)")
        if fn:
            caught = sum(1 for e in fn if e.get("ml_score", 0) >= ML_AGREE)
            lines.append(
                f"  ML was \u2265{ML_AGREE} in {caught}/{len(fn)} ({caught / len(fn) * 100:.0f}%) — "
                "goals ML would have warned about"
            )
        else:
            lines.append("  None in this window — gates are not costing goals right now.")

    lines.append("")
    lines.append("\u2139\ufe0f The ML brain is frozen (v1) and never changes signals — this only measures.")
    return "\n".join(lines)


def check_telegram_commands(client: httpx.Client) -> None:
    """v10.18: Check for Telegram commands.

    Polls getUpdates with long polling disabled (quick check).
    /stats    = all-time stats (1st signal vs all signals)
    /stats3d  = past 3 days
    /stats7d  = past 7 days
    /recap    = yesterday's stats (was auto-sent, now command-only)
    /eod      = full EOD report with poll data (yesterday)
    /eod3     = full EOD report (past 3 days)
    /eod7     = full EOD report (past 7 days)
    /eodall   = full EOD report (all data YTD)
    /outcomes = download signal_outcomes.jsonl file
    /polls    = download pressure_polls.jsonl (all poll data for ML)
    /polls_today = today's poll data only
    /count    = signal/poll counts + ML readiness progress
    /restore  = upload .jsonl backup file to restore ML data
    /mlstatus = full ML data status with dates
    /mlstats = GPS vs ML scoreboard (who reads incoming goals better)
    /today    = match list (no form/scorers, only if <10 games)
    /matches  = alias for /today
    Also handles .jsonl document uploads for /restore (auto-detect type, dedup).
    Runs once per main loop iteration (minimal overhead).
    """
    global _goal_watch_enabled, _surge_watch_enabled  # v10.53/v10.54: toggles
    try:
        resp = client.post(
            f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            json={"timeout": 0, "allowed_updates": ["message"]},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return
        results = resp.json().get("result", [])
        for update in results:
            msg = update.get("message", {})
            text = (msg.get("text") or "").strip()
            chat_id = msg.get("chat", {}).get("id")
            update_id = update.get("update_id")

            # Only respond to our configured chat
            if str(chat_id) != str(TELEGRAM_CHAT_ID):
                # Acknowledge to clear from queue
                if update_id:
                    client.post(
                        f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                        json={"offset": update_id + 1, "timeout": 0},
                        timeout=5.0,
                    )
                continue

            if text == "/help":
                send_telegram(client,
                    "\U0001f4cb AVAILABLE COMMANDS\n\n"
                    "\U0001f4ca SIGNALS & STATS\n"
                    "/stats \u2014 all-time signal win rate stats\n"
                    "/stats_today \u2014 today's signal win rate\n"
                    "/stats3d \u2014 stats for past 3 days\n"
                    "/stats7d \u2014 stats for past 7 days\n"
                    "/recap \u2014 yesterday's signal results\n\n"
                    "\U0001f4c4 REPORTS\n"
                    "/eod \u2014 full EOD report (yesterday)\n"
                    "/eod3 \u2014 EOD report (past 3 days)\n"
                    "/eod7 \u2014 EOD report (past 7 days)\n"
                    "/eodall \u2014 EOD report (all data YTD)\n\n"
                    "\U0001f4c1 ML DATA\n"
                    "/count \u2014 signal/poll counts + ML readiness\n"
                    "/mlstatus \u2014 full ML status with dates covered\n"
                    "/mlstats \u2014 GPS vs ML scoreboard (who reads goals better)\n"
                    "/fields \u2014 which KPI fields the API delivers per league\n"
                    "/sotfeed \u2014 which leagues deliver Top-SOT player data\n"
                    "/outcomes \u2014 download signal_outcomes.jsonl\n"
                    "/polls \u2014 download pressure_polls.jsonl (all)\n"
                    "/polls_today \u2014 download today's polls only\n"
                    "/restore \u2014 upload .jsonl file to restore data\n"
                    "  \u2192 Send any ml_signals_*.jsonl or ml_polls_*.jsonl\n"
                    "  \u2192 Deduplicates automatically, safe to re-upload\n\n"
                    "\u26bd MATCHES\n"
                    "/today or /matches \u2014 today's tracked matches\n\n"
                    "\u26a1 SURGE WATCH & GOAL FLASHES\n"
                    "/surgewatch \u2014 PRE-GOAL pressure alerts on/off (default ON)\n"
                    "  \u2192 Warns when a quiet team's shots suddenly start\n"
                    "  \u2192 coming \u2014 the buildup BEFORE the goal; sustained\n"
                    "  \u2192 bursts keep alerting, goals themselves never do\n"
                    "/goalwatch \u2014 instant goal alerts on/off (default OFF)\n"
                    "  \u2192 Fires ~10-30s AFTER a goal; goals come in bursts\n\n"
                    "\u2139\ufe0f AUTO BACKUP\n"
                    "Bot sends ml_signals_YYYY-MM-DD.jsonl +\n"
                    "ml_polls_YYYY-MM-DD.jsonl to this chat at EOD.\n"
                    "After a redeploy, upload those files back with /restore."
                )

            elif text == "/stats":
                # First resolve any stale outcomes
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                # Load ALL outcomes from file (not just in-memory)
                all_entries = load_all_outcomes()
                stats_text = format_outcome_stats(all_entries)
                send_telegram(client, stats_text)

            elif text == "/stats3d":
                # Past 3 days stats
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                cutoff = (datetime.now(BULGARIA_TZ) - timedelta(days=3)).strftime("%Y-%m-%d")
                filtered = [e for e in all_entries if e.get("signal_clock", "")[:10] >= cutoff]
                if not filtered:
                    send_telegram(client, f"No signals in the past 3 days (since {cutoff}).")
                else:
                    header = f"STATS: PAST 3 DAYS ({cutoff} to today)"
                    stats_text = header + "\n" + format_outcome_stats(filtered)
                    send_telegram(client, stats_text)

            elif text == "/stats_today":
                # Today's stats only
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
                filtered = [e for e in all_entries if e.get("signal_clock", "")[:10] == today_str]
                if not filtered:
                    send_telegram(client, f"No signals today ({today_str}).")
                else:
                    header = f"STATS: TODAY ({today_str})"
                    stats_text = header + "\n" + format_outcome_stats(filtered)
                    send_telegram(client, stats_text)

            elif text == "/stats7d":
                # Past 7 days stats
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                cutoff = (datetime.now(BULGARIA_TZ) - timedelta(days=7)).strftime("%Y-%m-%d")
                filtered = [e for e in all_entries if e.get("signal_clock", "")[:10] >= cutoff]
                if not filtered:
                    send_telegram(client, f"No signals in the past 7 days (since {cutoff}).")
                else:
                    header = f"STATS: PAST 7 DAYS ({cutoff} to today)"
                    stats_text = header + "\n" + format_outcome_stats(filtered)
                    send_telegram(client, stats_text)

            elif text == "/matches" or text == "/today":
                # v10.18: On-demand match list ONLY (no form/scorers), only if <10 games
                _send_today_command(client)

            elif text == "/recap":
                # v10.18: Yesterday's recap (was auto-sent, now command-only)
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    resolve_stale_outcomes(client)
                all_entries = load_all_outcomes()
                yesterday_str = (datetime.now(BULGARIA_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
                yesterday_signals = [e for e in all_entries
                                    if e.get("signal_clock", "")[:10] == yesterday_str]
                if not yesterday_signals:
                    send_telegram(client, f"No signals yesterday ({yesterday_str}).")
                else:
                    header = f"MORNING RECAP: {yesterday_str}"
                    stats_text = header + "\n" + format_outcome_stats(yesterday_signals)
                    send_telegram(client, stats_text)

            elif text in ("/eod", "/eod1", "/eod3", "/eod7", "/eodall"):
                # v10.35: On-demand full EOD report (with poll data)
                # /eod = yesterday only, /eod3 = 3 days, /eod7 = 7 days, /eodall = all data
                days = None
                specific_date = None
                if text == "/eod" or text == "/eod1":
                    specific_date = (datetime.now(BULGARIA_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
                elif text == "/eod3":
                    days = 3
                elif text == "/eod7":
                    days = 7
                # /eodall = no date filter, show everything
                # Resolve any pending first
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    try:
                        resolve_stale_outcomes(client)
                    except Exception:
                        pass
                if specific_date:
                    send_telegram(client, f"Generating EOD report for {specific_date}...")
                    cmd = ["python3", "eod_report.py", "--send", "--date", specific_date, "--quiet"]
                elif days:
                    send_telegram(client, f"Generating EOD report (last {days} days)...")
                    cmd = ["python3", "eod_report.py", "--send", "--days", str(days), "--quiet"]
                else:
                    send_telegram(client, "Generating EOD report (all data YTD)...")
                    cmd = ["python3", "eod_report.py", "--send", "--all", "--quiet"]
                try:
                    result = subprocess.run(cmd, cwd="/app", timeout=60)
                    if result.returncode != 0:
                        send_telegram(client, f"EOD report failed (exit code {result.returncode}).")
                except Exception as e:
                    send_telegram(client, f"EOD report error: {e}")

            elif text == "/outcomes":
                # v10.44d-patch: Send signal_outcomes.jsonl as a Telegram document
                pending_in_mem = [e for e in signal_outcomes if not e.get("resolved")]
                if pending_in_mem:
                    try:
                        resolve_stale_outcomes(client)
                    except Exception:
                        pass
                if not os.path.exists(OUTCOMES_FILE):
                    send_telegram(client, "No signal_outcomes.jsonl file found yet.")
                else:
                    file_size = os.path.getsize(OUTCOMES_FILE)
                    line_count = 0
                    with open(OUTCOMES_FILE, "r") as f:
                        for _ in f:
                            line_count += 1
                    try:
                        with open(OUTCOMES_FILE, "rb") as f:
                            client.post(
                                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                                data={"chat_id": TELEGRAM_CHAT_ID},
                                files={"document": ("signal_outcomes.jsonl", f, "application/jsonl")},
                                timeout=30.0,
                            )
                        send_telegram(client, f"sent signal_outcomes.jsonl ({line_count} entries, {file_size / 1024:.1f} KB)")
                        log.info(f"/outcomes: sent JSONL ({line_count} entries, {file_size / 1024:.1f} KB)")
                    except Exception as e:
                        send_telegram(client, f"Failed to send file: {e}")
                        log.error(f"/outcomes send failed: {e}")

            elif text == "/polls":
                # v10.44o: Send pressure_polls.jsonl as a Telegram document (gzip if >10MB)
                if not os.path.exists(POLL_DATA_FILE):
                    send_telegram(client, "No pressure_polls.jsonl file found yet.")
                else:
                    file_size = os.path.getsize(POLL_DATA_FILE)
                    line_count = 0
                    with open(POLL_DATA_FILE, "r") as f:
                        for _ in f:
                            line_count += 1
                    try:
                        if file_size > 10 * 1024 * 1024:
                            # Gzip compress for files > 10MB
                            buf = io.BytesIO()
                            with open(POLL_DATA_FILE, "rb") as raw:
                                with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
                                    gz.write(raw.read())
                            buf.seek(0)
                            gz_size = buf.getbuffer().nbytes
                            gz_fname = "pressure_polls.jsonl.gz"
                            client.post(
                                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                                data={"chat_id": TELEGRAM_CHAT_ID},
                                files={"document": (gz_fname, buf, "application/gzip")},
                                timeout=30.0,
                            )
                            send_telegram(client, f"sent {gz_fname} ({line_count} polls, {file_size / 1024 / 1024:.1f} MB -> {gz_size / 1024 / 1024:.1f} MB gzipped)")
                            log.info(f"/polls: sent gzipped ({line_count} polls, {file_size / 1024 / 1024:.1f} MB -> {gz_size / 1024 / 1024:.1f} MB)")
                        else:
                            with open(POLL_DATA_FILE, "rb") as f:
                                client.post(
                                    f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                                    data={"chat_id": TELEGRAM_CHAT_ID},
                                    files={"document": ("pressure_polls.jsonl", f, "application/jsonl")},
                                    timeout=30.0,
                                )
                            send_telegram(client, f"sent pressure_polls.jsonl ({line_count} polls, {file_size / 1024:.1f} KB)")
                            log.info(f"/polls: sent JSONL ({line_count} polls, {file_size / 1024:.1f} KB)")
                    except Exception as e:
                        send_telegram(client, f"Failed to send file: {e}")
                        log.error(f"/polls send failed: {e}")

            elif text == "/polls_today":
                # v10.44k: Send today's poll data only (smaller file)
                if not os.path.exists(POLL_DATA_FILE):
                    send_telegram(client, "No pressure_polls.jsonl file found yet.")
                else:
                    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
                    today_lines = []
                    with open(POLL_DATA_FILE, "r") as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                # Match by timestamp converted to Bulgaria timezone date
                                ts = entry.get("ts", 0)
                                if ts:
                                    entry_date = datetime.fromtimestamp(ts, BULGARIA_TZ).strftime("%Y-%m-%d")
                                    if entry_date == today_str:
                                        today_lines.append(line.strip())
                            except (json.JSONDecodeError, OSError):
                                continue
                    if not today_lines:
                        send_telegram(client, f"No poll data for today ({today_str}).")
                    else:
                        content = "\n".join(today_lines)
                        size_bytes = len(content.encode("utf-8"))
                        try:
                            buf = io.BytesIO(content.encode("utf-8"))
                            client.post(
                                f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                                data={"chat_id": TELEGRAM_CHAT_ID},
                                files={"document": (f"pressure_polls_{today_str}.jsonl", buf, "application/jsonl")},
                                timeout=30.0,
                            )
                            send_telegram(client, f"sent pressure_polls_{today_str}.jsonl ({len(today_lines)} polls, {size_bytes / 1024:.1f} KB)")
                            log.info(f"/polls_today: sent {len(today_lines)} polls ({size_bytes / 1024:.1f} KB)")
                        except Exception as e:
                            send_telegram(client, f"Failed to send file: {e}")
                            log.error(f"/polls_today send failed: {e}")

            elif text == "/count":
                # v10.44k: Show signal + poll counts for ML readiness
                _total_signals = 0
                _resolved_signals = 0
                _total_polls = 0
                _first_signal_ts = None
                _last_signal_ts = None
                if os.path.exists(OUTCOMES_FILE):
                    with open(OUTCOMES_FILE, "r") as _f:
                        for _line in _f:
                            _total_signals += 1
                            if '"resolved": true' in _line or '"resolved":True' in _line:
                                _resolved_signals += 1
                            # Extract timestamp for date range
                            try:
                                _obj = json.loads(_line)
                                _ts = _obj.get("signal_time")
                                if _ts and isinstance(_ts, (int, float)) and _ts > 0:
                                    _ts_dt = datetime.fromtimestamp(_ts, BULGARIA_TZ)
                                    _ts_str = _ts_dt.strftime("%Y-%m-%d %H:%M")
                                    if _first_signal_ts is None:
                                        _first_signal_ts = _ts_str
                                    _last_signal_ts = _ts_str
                            except Exception:
                                pass
                if os.path.exists(POLL_DATA_FILE):
                    with open(POLL_DATA_FILE, "r") as _f:
                        for _ in _f:
                            _total_polls += 1
                _ml_target = 200
                _pct = min(_resolved_signals / _ml_target * 100, 100) if _ml_target > 0 else 0
                _bar_len = 10
                _filled = int(_pct / 100 * _bar_len)
                _bar = "#" * _filled + "-" * (_bar_len - _filled)
                _date_range = ""
                if _first_signal_ts:
                    _date_range = f"\nData from: {_first_signal_ts}"
                    if _last_signal_ts and _last_signal_ts != _first_signal_ts:
                        _date_range += f" to {_last_signal_ts}"
                msg = (
                    f"📊 ML DATA COUNT\n\n"
                    f"Signals: {_total_signals} total, {_resolved_signals} resolved"
                    f"{_date_range}\n"
                    f"Polls: {_total_polls} total\n\n"
                    f"ML readiness ({_ml_target} resolved signals):\n"
                    f"[{_bar}] {_pct:.0f}%\n"
                    f"\nNeed {max(_ml_target - _resolved_signals, 0)} more resolved signals\n\n"
                    f"📦 Daily auto-backup to Telegram at EOD\n"
                    f"(combines across redeploys via daily files)"
                )
                send_telegram(client, msg)

            elif text == "/restore":
                send_telegram(client,
                    "\U0001f4c2 RESTORE ML DATA\n\n"
                    "Upload a .jsonl backup file as a document:\n"
                    "  \u2022 ml_signals_YYYY-MM-DD.jsonl \u2192 restores signals\n"
                    "  \u2022 ml_polls_YYYY-MM-DD.jsonl \u2192 restores polls\n"
                    "  \u2022 Any other .jsonl \u2192 auto-detected by content\n\n"
                    "Duplicate entries are skipped automatically.\n"
                    "After restore, /count and /stats will include restored data.\n\n"
                    "You can re-upload the same file safely — no doubles."
                )

            elif text == "/mlstatus":
                # v10.44m: Full ML data status
                _total_signals = 0
                _resolved_signals = 0
                _total_polls = 0
                _signal_dates = set()
                _poll_dates = set()
                if os.path.exists(OUTCOMES_FILE):
                    with open(OUTCOMES_FILE, "r") as _f:
                        for _line in _f:
                            _total_signals += 1
                            if '\"resolved\": true' in _line or '\"resolved\":True' in _line:
                                _resolved_signals += 1
                            try:
                                _obj = json.loads(_line)
                                _ts = _obj.get("signal_time")
                                if _ts and isinstance(_ts, (int, float)) and _ts > 0:
                                    _signal_dates.add(datetime.fromtimestamp(_ts, BULGARIA_TZ).strftime("%Y-%m-%d"))
                            except Exception:
                                pass
                if os.path.exists(POLL_DATA_FILE):
                    with open(POLL_DATA_FILE, "r") as _f:
                        for _line in _f:
                            _total_polls += 1
                            try:
                                _obj = json.loads(_line)
                                _ts = _obj.get("ts")
                                if _ts and isinstance(_ts, (int, float)) and _ts > 0:
                                    _poll_dates.add(datetime.fromtimestamp(_ts, BULGARIA_TZ).strftime("%Y-%m-%d"))
                            except Exception:
                                pass
                _ml_target = 200
                _pct = min(_resolved_signals / _ml_target * 100, 100) if _ml_target > 0 else 0
                _bar_len = 10
                _filled = int(_pct / 100 * _bar_len)
                _bar = "#" * _filled + "-" * (_bar_len - _filled)
                _sig_dates_str = ", ".join(sorted(_signal_dates)) if _signal_dates else "none"
                _poll_dates_str = ", ".join(sorted(_poll_dates)) if _poll_dates else "none"
                _backup_date = ""
                try:
                    if os.path.exists(ML_BACKUP_SENT_FILE):
                        with open(ML_BACKUP_SENT_FILE, "r") as _f:
                            _backup_date = _f.read().strip()
                except Exception:
                    pass
                _msg = (
                    f"\U0001f4ca ML DATA STATUS\n\n"
                    f"Signals: {_total_signals} total, {_resolved_signals} resolved\n"
                    f"Polls: {_total_polls} total\n\n"
                    f"ML readiness ({_ml_target} resolved):\n"
                    f"[{_bar}] {_pct:.0f}%\n"
                    f"Need {max(_ml_target - _resolved_signals, 0)} more\n\n"
                    f"\U0001f4c5 Signal dates:\n  {_sig_dates_str}\n"
                    f"\U0001f4c5 Poll dates:\n  {_poll_dates_str}\n"
                )
                if _backup_date:
                    _msg += f"\n\U0001f4e5 Last EOD backup: {_backup_date}"
                else:
                    _msg += "\n\U0001f4e5 Last EOD backup: none yet"
                # v10.44p/v10.50: Show event fast lane status
                _fl_msg = "idle"
                if _event_fast_lane_fids:
                    _fl_list = ", ".join(f"F{f}" for f in _event_fast_lane_fids)
                    _fl_msg = f"{_fl_list} ({_event_fast_lane_credits_today} credits today, {len(fastlane_shadow)} shadow records)"
                _msg += f"\n\U0001f535 Event fast lane: {_fl_msg}"
                # v10.53: Goal watch status
                _gw_msg = "OFF" if not _goal_watch_enabled else (
                    f"ON ({len(_goal_watch_fids)} watching, "
                    f"{_gw_flashes_today} flashes, {_goal_watch_credits_today} cr)"
                )
                _msg += f"\n\u26a1 Goal watch: {_gw_msg}"
                # v10.54: surge watch status
                _sw_msg = "OFF" if not _surge_watch_enabled else (
                    f"ON ({_surge_alerts_today} alerts today)"
                )
                _msg += f"\n\U0001f50e Surge watch: {_sw_msg}"
                send_telegram(client, _msg)

            elif text == "/mlstats":
                # v10.59: GPS vs ML scoreboard — who reads incoming goals better
                _pending_ml = [e for e in signal_outcomes if not e.get("resolved")]
                if _pending_ml:
                    resolve_stale_outcomes(client)
                _sig_all = load_all_outcomes()
                _blk_all = _load_blocked_outcomes()
                send_telegram(client, _format_ml_scoreboard(_sig_all, _blk_all, days=7))

            elif text == "/fields":
                # v10.60: field-availability census (live-learned, read-only)
                send_telegram(client, format_field_census())

            elif text == "/sotfeed":
                # v10.65: shot-event feed census (live-learned, read-only)
                send_telegram(client, format_sot_feed_census())

            elif text.startswith("/goalwatch"):
                # v10.53: goal flash alerts toggle/status
                _parts = text.split()
                if len(_parts) > 1 and _parts[1].lower() in ("on", "off"):
                    _goal_watch_enabled = (_parts[1].lower() == "on")
                _gw_watch_list = ", ".join(f"F{f}" for f in _goal_watch_fids) or "none"
                send_telegram(client, (
                    "\u26a1 GOAL WATCH\n"
                    f"State: {'ON' if _goal_watch_enabled else 'OFF'}\n"
                    f"Watching now: {_gw_watch_list}\n"
                    f"Flashes today: {_gw_flashes_today}\n"
                    f"Credits today: {_goal_watch_credits_today}/{GOAL_WATCH_CREDIT_CAP}\n\n"
                    "What it does: instant GOAL alerts (~10-30s) for close games "
                    f"from {GOAL_WATCH_MINUTE}'+. Goals come in bursts \u2014 the flash "
                    "tells you a game just opened so you can judge next-goal risk. "
                    "OFF by default since v10.54 \u2014 /surgewatch warns about the "
                    "buildup BEFORE goals instead."
                ))

            elif text.startswith("/surgewatch"):
                # v10.54: pre-goal pressure (surge) alerts toggle/status
                _parts = text.split()
                if len(_parts) > 1 and _parts[1].lower() in ("on", "off"):
                    _surge_watch_enabled = (_parts[1].lower() == "on")
                send_telegram(client, (
                    "\U0001f50e SURGE WATCH\n"
                    f"State: {'ON' if _surge_watch_enabled else 'OFF'}\n"
                    f"Alerts today: {_surge_alerts_today}/{SURGE_MAX_PER_DAY}\n\n"
                    "What it does: warns you the moment a QUIET team suddenly "
                    "starts shooting \u2014 pressure buildup BEFORE the goal:\n"
                    "\U0001f4c8 2+ shots in 5' after a long quiet spell\n"
                    "\U0001f50e first shot ON TARGET after 15'+ of silence\n"
                    "\U0001f525 2nd shot on target within 10' (strongest sign)\n"
                    "\U0001f525 every FURTHER shot keeps alerting (sustained)\n\n"
                    "Goals themselves never alert \u2014 a goal counts as the "
                    "team's last shot, so after a goal a fresh quiet spell "
                    "(15') is required before the next warning = your "
                    "second-goal early watch.\n"
                    "Close games from 60'+, checked every ~30s. Zero extra API "
                    "credits \u2014 rides the same event polls as goal watch.\n"
                    "v10.67: close games now watched TO THE FINAL WHISTLE \u2014 "
                    "86-90' pressure included (Botev Vratsa 86'/88' class).\n"
                    "These are WATCH alerts, not betting signals."
                ))

            else:
                # v10.44m: Check if user sent a document (file upload for /restore)
                _doc = msg.get("document")
                if _doc:
                    _filename = _doc.get("file_name", "")
                    _file_id = _doc.get("file_id", "")
                    if (_filename.endswith(".jsonl") or _filename.endswith(".jsonl.gz")) and _file_id:
                        _restore_ok = _restore_from_telegram_file(client, _file_id, _filename)
                        if _restore_ok is not None:
                            if _restore_ok:
                                send_telegram(client, f"\u2705 Restored {_restore_ok} entry/entries from {_filename}")
                            else:
                                send_telegram(client, f"\u26a0\ufe0f File {_filename} contained no new entries (all duplicates or invalid).")

            # Acknowledge update to clear from queue
            if update_id:
                client.post(
                    f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                    json={"offset": update_id + 1, "timeout": 0},
                    timeout=5.0,
                )
    except Exception:
        pass  # Non-critical, don't spam logs


def send_daily_summary(client: httpx.Client) -> None:
    """v10.17: Send daily match list to Telegram (0 extra credits).

    v10.17: Removed auto-sent team form + top scorers (was burning 40-80 credits/day).
    Match list only. Team form + scorers available via /matches command.
    """
    global daily_summary_date

    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
    if daily_summary_date == today_str:
        return  # Already sent today
    if not todays_tracked_fixtures:
        log.info("v10.12: No tracked fixtures, skipping daily summary")
        return

    log.info(f"v10.17: Building daily summary for {len(todays_tracked_fixtures)} match(es)...")

    lines = []
    lines.append(f"\u26bd TODAY'S MATCHES ({today_str})")
    lines.append("")

    for fix in todays_tracked_fixtures:
        home = fix["teams"]["home"]
        away = fix["teams"]["away"]
        league_name = fix["league"].get("name", "?")
        try:
            kickoff_utc = datetime.fromisoformat(
                fix["fixture"]["date"].replace("Z", "+00:00")
            )
            kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
            ko_str = kickoff_local.strftime("%H:%M")
        except Exception:
            ko_str = "??:??"
        lines.append(f"\U0001f3c6 {league_name} — {ko_str} Bulgaria")
        lines.append(f"{home['name']} vs {away['name']}")
        lines.append("")

    msg = "\n".join(lines)
    sent = send_telegram(client, msg)

    if sent:
        daily_summary_date = today_str
        log.info(f"v10.17: Daily summary sent ({len(todays_tracked_fixtures)} matches, 0 extra credits)")
    else:
        log.warning("v10.17: Daily summary Telegram send FAILED — will retry next wake-up")


def _send_today_command(client: httpx.Client) -> None:
    """v10.18: /today or /matches command — match list ONLY (no form, no scorers).

    Only works when <10 games. 0 extra API credits.
    Team form and scorers removed to save credits.
    """
    if not todays_tracked_fixtures:
        send_telegram(client, "No tracked matches today.")
        return

    if len(todays_tracked_fixtures) >= 10:
        send_telegram(
            client,
            f"Too many matches today ({len(todays_tracked_fixtures)}). "
            f"/today only available when <10 games to avoid spam."
        )
        return

    today_str = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
    lines = []
    lines.append(f"\u26bd TODAY'S MATCHES ({today_str})")
    lines.append("")

    for fix in todays_tracked_fixtures:
        home = fix["teams"]["home"]
        away = fix["teams"]["away"]
        league_name = fix["league"].get("name", "?")
        try:
            kickoff_utc = datetime.fromisoformat(
                fix["fixture"]["date"].replace("Z", "+00:00")
            )
            kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
            ko_str = kickoff_local.strftime("%H:%M")
        except Exception:
            ko_str = "??:??"
        lines.append(f"\U0001f3c6 {league_name} — {ko_str} Bulgaria")
        lines.append(f"{home['name']} vs {away['name']}")
        lines.append("")

    msg = "\n".join(lines)
    send_telegram(client, msg)
    log.info(f"v10.18: /today sent ({len(todays_tracked_fixtures)} matches, 0 extra credits)")


def process_fixture_stats(client: httpx.Client, fixture: dict) -> None:
    """Process one fixture from batched /fixtures?ids=... response.

    v9.5: New signal format with xG, top SOT player, red cards.
    Strict 80' cutoff. Tracks signaled fixtures for diversification.
    """
    fid = fixture["fixture"]["id"]

    # v9.8: Check outcome of pending signals FIRST (before early returns)
    # This catches: goals scored since last poll, fixtures that just ended
    check_signal_outcomes(fixture, client)
    check_blocked_outcomes(fixture, client)  # v10.49: false-negative resolution
    check_fastlane_shadow(fixture, client)  # v10.50: fast-lane shadow resolution

    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        return

    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    home = fixture["teams"]["home"]
    away = fixture["teams"]["away"]
    home_tid = home["id"]
    away_tid = away["id"]

    statistics = fixture.get("statistics") or []
    if not statistics:
        return

    # Parse all team stats into a dict keyed by team name AND team ID
    teams_data = {}
    teams_data_by_id = {}  # v10.31: ID-based fallback when names don't match
    for team_entry in statistics:
        team = team_entry.get("team", {})
        tname = team.get("name")
        tid_stat = team.get("id")  # v10.31: also key by ID
        if not tname:
            continue
        tmap = {}
        for stat in team_entry.get("statistics", []):
            stat_type = stat.get("type")
            value = stat.get("value")
            if stat_type:
                # v10.1: store BOTH original key and stripped version
                # so alias lookup always has something to find
                # v10.19: Treat None AND literal "N/A" as missing data (→ "0")
                # This prevents GPS from being calculated on incomplete stats.
                if value is None or str(value).strip() in ("", "N/A"):
                    tmap[stat_type] = "0"
                else:
                    tmap[stat_type] = str(value).strip()
                if stat_type != stat_type.strip():
                    tmap[stat_type.strip()] = tmap[stat_type]
        teams_data[tname] = tmap
        if tid_stat:
            teams_data_by_id[tid_stat] = tmap

    if not teams_data:
        return

    # v10.80: live market-block edits — for every signal of this fixture
    # carrying a cards/corners block, refresh the counts (editMessageText)
    # when they changed. Rate-guarded (45s); cleanup at FT happens in
    # check_signal_outcomes.
    try:
        _update_market_blocks(
            client, fid, home_tid, away_tid, teams_data_by_id,
            minute, fixture["goals"]["home"] or 0, fixture["goals"]["away"] or 0,
        )
    except Exception as _em80:
        log.debug(f"  v10.80 market edit hook skipped: {_em80}")

    # --- Fast SOT window management (fixture-level) ---
    # Check BOTH teams' SOT. Activate if either >= 2.
    # v9.5: Only activate for unsignaled fixtures.
    # v10.1: Use get_stat() for all field access (normalized aliases)
    best_current_sot = 0
    for tname, tstats in teams_data.items():
        current_sot = safe_int(get_stat(tstats, "sot"))
        best_current_sot = max(best_current_sot, current_sot)

    # v10.21: Activate fast window only if SOT increased since last activation.
    # Prevents infinite re-activation loop when SOT is stuck (e.g. SOT=1 for 10+ min).
    if best_current_sot >= 1:
        team_sig_count = sum(1 for (f, t) in signaled_teams if f == fid)
        if team_sig_count < 2 and not is_fast_sot_active(fid):
            last_activated_sot = fast_sot_activated_at_sot.get(fid, 0)
            if best_current_sot > last_activated_sot:
                activate_fast_sot(fid, best_current_sot)
    else:
        expire_fast_sot(fid)

    # --- v9.7: SOT acceleration tracking (fixture-level) ---
    # If the best SOT increased since last poll, mark as accelerating.
    # This gives 60s polling to catch the NEXT SOT increase.
    prev_best_sot = get_fixture_best_sot(fid)
    # (best_current_sot was computed above from this poll's data)
    # We'll update accelerating_fixtures AFTER the per-team loop
    # (since team_state gets updated there, which affects get_fixture_best_sot)
    _sot_increased = best_current_sot > prev_best_sot

    # v10.31: EVENT-BASED SOT SUPPLEMENT — close the 2-3 min statistics lag.
    # After 75' with meaningful pressure, fetch /fixtures/events (near-real-time)
    # to detect SOT jumps that statistics hasn't caught up to yet.
    # effective_sot = max(stats_sot, events_sot) — audit trail preserved.
    event_sot_by_team: dict[int, int] = {}
    event_goal_by_team: dict[int, int] = {}   # v10.56: goal shots per team
    if minute >= EVENT_SOT_MINUTE and best_current_sot >= EVENT_SOT_MIN_PRESSURE:
        ev_result = fetch_live_sot_from_events(client, fid)
        if ev_result:
            event_sot_by_team, event_goal_by_team = ev_result
            # Check if events reveal higher SOT than statistics for ANY team
            for tid_check, ev_sot_val in event_sot_by_team.items():
                if ev_sot_val > best_current_sot:
                    event_extended_fixtures.add(fid)
                    log.info(
                        f"  EVENT-EXTEND: F{fid} {minute}' — events SOT={ev_sot_val} "
                        f"> stats best={best_current_sot}, monitoring extended to 90'"
                    )
                    break

    # v10.60: per-fixture blocked-shots / subs / cards from the last events
    # response (fast lane or the 75'+ supplement above). None when absent.
    _ev60_extras = get_event_extras(fid)

    # v10.31: Deferred hard_max check — now after stats parsing + events.
    # Event-extended fixtures (events SOT > stats SOT) get 90' ceiling.
    hard_max = EXTENDED_MAX
    if fid in event_extended_fixtures:
        hard_max = 90
    if minute > hard_max:
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        _retain_late_fixture(fid, fixture, "stats-lane ceiling")  # v10.67: events watch to FT
        # Clean up event extension when fixture exits
        event_extended_fixtures.discard(fid)
        _event_sot_cache.pop(fid, None)
        _event_extras_cache.pop(fid, None)   # v10.60: same lifecycle
        log.info(f"  Fixture {fid} past {hard_max}', removed from monitoring")
        return

    # --- Pre-compute red cards and xG for both teams ---
    red_card_str = get_red_card_string(teams_data, home["name"], away["name"])

    # Parse xG for both teams
    team_xg = {}
    for tname, tstats in teams_data.items():
        team_xg[tname] = parse_xg(tstats)

    # --- SOT SIGNAL CHECK (per team) ---
    # v10: Calculate GPS for every team on every poll (zero extra API cost).
    # This builds the dataset and drives acceleration-aware polling.
    # v10.44d-fix: Use API league name for recording (avoids ID collisions).
    # LEAGUE_IDS is used ONLY for filtering (lid in LEAGUE_IDS), not naming.
    # Bug: API returns league ID 357 for both Bulgarian & Irish leagues,
    # causing Irish teams to be labeled "First League (Bulgaria)".
    # v10.44d-patch: Also apply Irish team name override (API name is same for both).
    _raw_league = fixture["league"].get("name", LEAGUE_IDS.get(fixture["league"]["id"], "?"))
    league = _fix_league_name(_raw_league, home["name"], away["name"])
    sh = fixture["goals"]["home"] or 0
    sa = fixture["goals"]["away"] or 0

    # v10.22: Safety net — initialize gps before per-team loop.
    # Prevents rare 'gps not defined' NameError when both teams
    # hit DATA_INCOMPLETE skip, causing fixture to crash on every poll,
    # burn 1 credit each time, and never save state (infinite retry loop).
    gps = 0.0
    # v10.31: Same safety net for 'state' — Python 3.13+ raises
    # 'cannot access local variable' when both teams skip via
    # 'if not tstats: continue' before the assignment inside the loop.
    state = None
    # v10.56: per-team GENUINE (non-goal) SOT rise this poll — collected for
    # the fixture-level genuine-burst flag (goal shots never make a fixture
    # "bursting" for signal purposes)
    genuine_poll_rise_by_team: dict[int, int] = {}

    for tid, tname in ((home_tid, home["name"]), (away_tid, away["name"])):
        tstats = teams_data.get(tname)
        # v10.31: Fallback to ID-based lookup when team names don't match
        # (e.g. /fixtures?ids= returns different name format than statistics)
        if not tstats:
            tstats = teams_data_by_id.get(tid)
        if not tstats:
            log.debug(f"  No stats found for {tname} (tid={tid}) F{fid} — name/ID mismatch")
            continue

        # v10.1: All stat access via get_stat() (normalized aliases)
        stats_sot = safe_int(get_stat(tstats, "sot"))

        # v10.31: Event-based SOT supplement — use higher of stats vs events.
        # Events endpoint updates in seconds; statistics lags 2-3 min.
        # This closes the blind spot for 75'-90' SOT jumps.
        events_sot = event_sot_by_team.get(tid, 0)
        sot = max(stats_sot, events_sot)
        if events_sot > stats_sot:
            log.info(
                f"  F{fid} {minute}' {tname}: "
                f"stats_sot={stats_sot} | events_sot={events_sot} | effective_sot={sot}"
            )

        # v10.10: Parse ALL available stats for GPS calculation
        total_shots = safe_int(get_stat(tstats, "total_shots"))

        # v10.38: Compute is_home_team BEFORE any usage (was used at line 3549
        # before being defined at line 3583, causing NameError or wrong value)
        is_home_team = (tid == home_tid)

        # v10.44b: Infer missing total_shots from SOT + off-target.
        # Small leagues (Conference League qual, etc.) often return SOT=3
        # but total_shots=0 — the field is missing, not corrupt.
        shots_off_target_early = safe_int(get_stat(tstats, "shots_off_target"))
        if total_shots == 0 and stats_sot > 0:
            total_shots = stats_sot + shots_off_target_early

        # v10.44d-patch: Infer missing shots_inside_box from SOT.
        # Conference League and smaller leagues sometimes return IB=0
        # despite the team having SOT. On-target shots are almost always
        # inside the box (a shot from outside that's on target is rare).
        # Without this, IB component gives 0 pts and GPS is deflated ~15-20 pts.
        shots_inside_box_early = safe_int(get_stat(tstats, "shots_inside_box"))
        if shots_inside_box_early == 0 and sot > 0:
            shots_inside_box_early = sot  # SOT as IB floor
            log.info(
                f"  IB INFERRED: {tname} F{fid} {minute}' — IB=0 but SOT={sot}, using IB={sot}"
            )

        # v10.13.1: Data sanity check — reject clearly corrupted API responses
        # v10.31: Use stats_sot (not effective_sot) — these check the
        # statistics API response quality, not the events supplement.
        _data_corrupt = False
        if stats_sot > total_shots and total_shots > 0:
            _data_corrupt = True
            log.warning(
                f"  DATA CORRUPT: {tname} F{fid} {minute}' — "
                f"SOT({stats_sot}) > Total Shots({total_shots}), skipping signal"
            )
        # v10.44b: Removed "SOT>=3 but Total Shots=0" corrupt check.
        # Missing total_shots is now inferred above (SOT + off-target).
        # v10.19: Detect completely missing stat response from API
        # If total_shots=0 AND sot=0 AND shots_inside_box will be 0,
        # the API likely returned N/A for everything — GPS would be meaningless.
        if total_shots == 0 and stats_sot == 0:
            _data_incomplete = True
        else:
            _data_incomplete = False
        shots_off_target = safe_int(get_stat(tstats, "shots_off_target"))  # v10.10: replaces DA
        shots_inside_box = shots_inside_box_early  # v10.44d-patch: uses inferred value if IB was 0
        big_chances = safe_int(get_stat(tstats, "big_chances"))  # v10.44d: quality chances
        corners = safe_int(get_stat(tstats, "corner_kicks"))
        xg_str = team_xg.get(tname, "N/A")
        xg_value = safe_float(xg_str)
        possession = safe_int(get_stat(tstats, "possession"))

        # v10.60: FREE-TIER KPI EXPANSION — parsed from the SAME statistics
        # response already fetched for GPS. LOGGING ONLY: never feeds GPS,
        # gates, tiers or polling. Fields record None when the API did not
        # deliver them (get_stat_present — dead-field proof: big_chances).
        gk_saves, _gk_arrived = get_stat_present(tstats, "gk_saves")
        fouls, _fouls_arrived = get_stat_present(tstats, "fouls")
        offsides, _offs_arrived = get_stat_present(tstats, "offsides")
        yellow_cards, _yc_arrived = get_stat_present(tstats, "yellow_cards")
        total_passes, _tp_arrived = get_stat_present(tstats, "total_passes")
        passes_accurate, _pa_arrived = get_stat_present(tstats, "passes_accurate")
        pass_accuracy = (
            round(passes_accurate / total_passes * 100, 1)
            if (total_passes is not None and total_passes > 0 and passes_accurate is not None)
            else None
        )
        # v10.60: events-derived KPIs — only when an events response covers
        # this fixture (fast lane / 75'+ supplement). None otherwise.
        blocked_shots = _ev60_extras["blocked"].get(tid, 0) if _ev60_extras else None
        subst_count = _ev60_extras["subst"].get(tid, 0) if _ev60_extras else None
        subst_latest_minute = _ev60_extras["subst_latest"].get(tid) if _ev60_extras else None
        card_latest_minute = _ev60_extras["card_latest"].get(tid) if _ev60_extras else None
        # v10.73: red-card counts for this team / the opponent (events-
        # based; None when no fresh events response covers the fixture —
        # honest missingness, never fake zeros).
        if _ev60_extras is not None:
            _rc_list60 = _ev60_extras.get("red_cards") or []
            red_cards_team_n = sum(1 for r in _rc_list60 if r.get("team_id") == tid)
            red_cards_opp_n = len(_rc_list60) - red_cards_team_n
        else:
            red_cards_team_n = None
            red_cards_opp_n = None
        # v10.60: census — which fields actually arrived (per league, once)
        _bc60_val, _bc_arrived = get_stat_present(tstats, "big_chances")
        update_field_census(
            fixture["league"]["id"], league,
            {
                "gk_saves": _gk_arrived, "fouls": _fouls_arrived,
                "offsides": _offs_arrived, "yellow_cards": _yc_arrived,
                "total_passes": _tp_arrived,
                "big_chances": _bc_arrived,
                "expected_goals": xg_value is not None,
            },
        )

        # Get opponent SOT and xG (v10.1: normalized)
        # v10.44k: Extract full opponent stats for ML (poll data)
        opponent_sot = "0"
        opponent_xg = "N/A"
        _opp_total_shots = 0
        _opp_shots_inside_box = 0
        _opp_corners = 0
        _opp_big_chances = 0
        _opp_shots_off_target = 0
        # v10.60: opponent KPI expansion (null-safe, logging only)
        _opp_gk_saves = None
        _opp_fouls = None
        _opp_offsides = None
        _opp_yellow_cards = None
        for oname, ostats in teams_data.items():
            if oname != tname:
                opponent_sot = str(safe_int(get_stat(ostats, "sot")))
                opponent_xg = team_xg.get(oname, "N/A")
                _opp_sot_int = safe_int(opponent_sot)
                _opp_total_shots = safe_int(get_stat(ostats, "total_shots"))
                _opp_shots_inside_box = safe_int(get_stat(ostats, "shots_inside_box"))
                _opp_shots_off_target = safe_int(get_stat(ostats, "shots_off_target"))
                _opp_corners = safe_int(get_stat(ostats, "corner_kicks"))
                _opp_big_chances = safe_int(get_stat(ostats, "big_chances"))
                _opp_gk_saves, _ = get_stat_present(ostats, "gk_saves")
                _opp_fouls, _ = get_stat_present(ostats, "fouls")
                _opp_offsides, _ = get_stat_present(ostats, "offsides")
                _opp_yellow_cards, _ = get_stat_present(ostats, "yellow_cards")
                # Infer opponent total_shots if missing (same logic as main team)
                if _opp_total_shots == 0 and (_opp_sot_int > 0 or _opp_shots_off_target > 0):
                    _opp_total_shots = _opp_sot_int + _opp_shots_off_target
                break

        state = team_state.get((fid, tid))
        history = team_gps_history.get((fid, tid), [])

        # v10.19: Skip GPS calculation when API returned no real stat data.
        # GPS from all-zeros is meaningless and could produce false signals.
        if _data_incomplete:
            log.info(f"  DATA INCOMPLETE: {tname} F{fid} {minute}' — shots=0, SOT=0, skipping GPS")
            # Still update state to avoid crashes on next poll
            team_state[(fid, tid)] = {
                "last_sot": 0, "last_minute": minute, "last_xg": None,
                "last_shots_off_target": 0, "last_total_shots": 0,
                "last_shots_inside_box": 0, "last_corners": 0,
                "last_big_chances": 0,  # v10.44d
                "last_possession": 0,
                "last_goals": (sh if is_home_team else sa) or 0,  # v10.36
                "last_goal_minute": 0,  # v10.36
                "sot_at_last_goal": 0,  # v10.44h: SOT snapshot when goal scored
            }
            continue

        # === v10.26: Calculate Goal Pressure Score (poss removed, xG boosted) ===
        gps, gps_desc, components = calculate_goal_pressure_score(
            sot=sot, total_shots=total_shots,
            shots_inside_box=shots_inside_box,
            shots_off_target=shots_off_target,
            xg_value=xg_value, corners=corners,
            big_chances=big_chances,  # v10.44d
            minute=minute, prev_state=state,
            gps_history=history,
            possession=possession,
        )

        # Count accelerating indicators (for polling + logging)
        accel_count = get_accel_count_from_state(
            sot, total_shots, shots_off_target, xg_value, state, minute
        )

        # === v10: Log GPS on every poll ===
        if gps >= GPS_BUILDING:
            stage_label = "BUILDING" if gps < GPS_EARLY_WARNING else "EARLY" if gps < GPS_CRITICAL else "CRITICAL"
            log.info(
                f"  GPS {stage_label}: {tname} | {gps_desc}"
            )
        else:
            # v10.13.1: Always log per-fixture SOT+GPS for visibility
            log.info(
                f"  F{fid} {minute}' {tname}: SOT={sot} Shots={total_shots} GPS={gps:.0f}"
            )

        # === v10.1: Record poll data for backtesting (includes possession) ===
        # v10.38: is_home_team now defined earlier (before _data_incomplete check)
        team_goals = (sh if is_home_team else sa) or 0  # v10.36: moved up for post-goal tracking
        # v10.44k: Compute opponent GPS for ML poll data
        _opp_xg_val = safe_float(opponent_xg)
        _opp_gps, _, _ = calculate_goal_pressure_score(
            sot=_opp_sot_int, total_shots=_opp_total_shots,
            shots_inside_box=_opp_shots_inside_box,
            shots_off_target=_opp_shots_off_target,
            xg_value=_opp_xg_val, corners=_opp_corners,
            big_chances=_opp_big_chances,
            minute=minute, prev_state=None, gps_history=[], possession=0,
        )

        # v10.36/v10.59: POST-GOAL DETECTION STATE — fetch previous goals and
        # goal minute BEFORE the gates (moved up in v10.59; nothing modifies
        # team_state between here and the goal-detection logic below).
        _prev_state = team_state.get((fid, tid))
        _prev_goals = _prev_state.get("last_goals", None) if _prev_state else None
        last_goal_minute = _prev_state.get("last_goal_minute", 0) if _prev_state else 0

        # v10.72: COLD-START LEDGER SEEDING (Benfica 16' class, Sep 5).
        # The v10.56 goal-shot ledger only ever fills from LIVE goal
        # detection — after a mid-match restart, goals scored BEFORE the
        # restart are invisible, so the v10.69 goal-shot net computed
        # eff SOT = raw SOT and the goal's own shot could trigger a false
        # first signal (Benfica: goal 7', restart, SOT=3 at 16' -> false
        # CRITICAL; honest eff = 2 -> GPS 64 < 75 -> no signal).
        # Seed the LANDED ledger from the current scoreline on this team's
        # first-ever poll: every pre-restart goal counts as an already-landed
        # goal shot, never as trigger evidence. Upward-only, zero credits,
        # semantics identical to the live score-change path (same score
        # feed, same per-(fid,tid) landing bucket).
        if _prev_state is None and team_goals > 0 and (fid, tid) not in _goal_sot_landed:
            _goal_sot_landed[(fid, tid)] = team_goals
            log.info(
                f"  v10.72 COLD-START LEDGER: {tname} — seeding {team_goals} "
                f"pre-restart goal shot(s) as landed (never trigger evidence)"
            )

        # v10.45/v10.59: ML SHADOW SCORE — the trained model's independent
        # opinion on this same poll. v10.59 moved this computation BEFORE the
        # gates so sent signals AND blocked candidates can carry it. Still
        # purely informational: NEVER affects which signals get sent — that
        # stays 100% GPS-driven. Feature values are unchanged by the move:
        # the 5m/10m history lookups never read the current poll's entry
        # (they search e_min <= minute-5 / -10), and this poll's own history
        # entry is appended further below.
        _ml_ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
        _ml_recency = _build_recency_fields(
            fid, tid, minute, sot, total_shots, shots_inside_box,
            shots_off_target, xg_value, corners, gps, accel_count,
        )
        _ml_minutes_since_goal = (minute - last_goal_minute) if last_goal_minute > 0 else None
        ml_score = calculate_ml_score({
            "sot": sot, "total_shots": total_shots, "shots_inside_box": shots_inside_box,
            "ib_ratio": _ml_ib_ratio, "xg": xg_value, "corners": corners,
            "gps": gps, "gps_sot": components.get("sot", 0), "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0), "gps_xg": components.get("xg", 0),
            "gps_accel": components.get("acceleration", 0), "accel_count": accel_count,
            "is_home": is_home_team, "score_diff": team_goals - (sa if is_home_team else sh),
            "team_score": team_goals, "opp_score": (sa if is_home_team else sh),
            "sot_delta_5m": _ml_recency.get("sot_delta_5m"), "sot_delta_10m": _ml_recency.get("sot_delta_10m"),
            "xg_delta_5m": _ml_recency.get("xg_delta_5m"), "shots_delta_5m": _ml_recency.get("shots_delta_5m"),
            "recency_ratio": _ml_recency.get("recency_ratio"),
            "opp_sot": _opp_sot_int, "opp_total_shots": _opp_total_shots,
            "opp_shots_inside_box": _opp_shots_inside_box, "opp_xg": _opp_xg_val,
            "opp_corners": _opp_corners, "opp_gps": _opp_gps,
            "last_goal_minute": last_goal_minute if last_goal_minute > 0 else None,
            "minutes_since_last_goal": _ml_minutes_since_goal, "minute": minute,
        })
        if ml_score is not None:
            log.info(f"  ML SHADOW: {tname} | GPS={gps:.0f} vs ML={ml_score:.0f} (comparison only, not gating)")
        record_pressure_poll(
            fid=fid, tid=tid, tname=tname, league=league,
            minute=minute, sot=sot, total_shots=total_shots,
            shots_inside_box=shots_inside_box,
            shots_off_target=shots_off_target,
            xg_value=xg_value, corners=corners,
            big_chances=big_chances,  # v10.44d
            gps=gps, components=components,
            accel_count=accel_count, is_home=is_home_team,
            score_home=sh, score_away=sa,
            possession=possession,
            stats_sot_raw=stats_sot, events_sot=events_sot,  # v10.31
            # v10.44k: Opponent stats for ML
            opp_sot=_opp_sot_int,
            opp_total_shots=_opp_total_shots,
            opp_shots_inside_box=_opp_shots_inside_box,
            opp_xg=_opp_xg_val,
            opp_corners=_opp_corners,
            opp_big_chances=_opp_big_chances,
            opp_gps=_opp_gps,
            league_id=fixture["league"]["id"],
            ml_score=ml_score,  # v10.59: ML shadow opinion saved into poll data
            # v10.60: free-tier KPI expansion (logging only, null-safe)
            gk_saves=gk_saves,
            opp_gk_saves=_opp_gk_saves,
            fouls=fouls,
            opp_fouls=_opp_fouls,
            offsides=offsides,
            opp_offsides=_opp_offsides,
            yellow_cards=yellow_cards,
            opp_yellow_cards=_opp_yellow_cards,
            total_passes=total_passes,
            pass_accuracy=pass_accuracy,
            blocked_shots=blocked_shots,
            subst_count=subst_count,
            subst_latest_minute=subst_latest_minute,
            card_latest_minute=card_latest_minute,
            # v10.73: events-based red-card counts (logging only, null-safe)
            red_cards=red_cards_team_n,
            opp_red_cards=red_cards_opp_n,
        )

        # === v10.1: Update GPS history (includes timestamp + xg for window calc) ===
        new_history_entry = {
            "ts": time.time(),
            "gps": round(gps, 1),
            "sot": sot, "total_shots": total_shots,
            "shots_off_target": shots_off_target,  # v10.10
            "accel_count": accel_count, "minute": minute,
            "xg": xg_value,
            "big_chances": big_chances,  # v10.58: BC-delta freshness warning
        }
        if (fid, tid) not in team_gps_history:
            team_gps_history[(fid, tid)] = []
        team_gps_history[(fid, tid)].append(new_history_entry)
        if len(team_gps_history[(fid, tid)]) > GPS_HISTORY_MAX:
            team_gps_history[(fid, tid)] = team_gps_history[(fid, tid)][-GPS_HISTORY_MAX:]
        # v10.84: minute-resolution archive (latest poll per game minute)
        _arc_key = (fid, tid)
        _arc = _minute_history.get(_arc_key)
        if _arc and _arc[-1].get("minute") == minute:
            _arc[-1] = new_history_entry
        else:
            if _arc is None:
                _arc = []
                _minute_history[_arc_key] = _arc
            _arc.append(new_history_entry)
        if len(_arc) > MINUTE_ARCHIVE_LEN:
            del _arc[:len(_arc) - MINUTE_ARCHIVE_LEN]

        # === v10.1: Classify signal with quality gates ===
        ib_ratio = shots_inside_box / total_shots if total_shots > 0 else 0.0
        sustained = components.get("sustained", 0)

        # v10.38: SOT DATA RELIABILITY GUARD
        # API-Football can report inflated SOT (Celtic CL: 3 SOT vs FotMob's 1).
        # When SOT>=3, check if the SOT/total_shots ratio is realistic.
        # Real matches rarely exceed 50% SOT rate; >70% is almost certainly bad data.
        # Also flag SOT>=3 with <4 total shots (3 on-target from 3 total = 100%).
        _sot_suspicious = False
        if sot >= 3 and total_shots > 0:
            _sot_ratio = sot / total_shots
            if _sot_ratio > 0.70:
                _sot_suspicious = True
                log.warning(
                    f"  SOT DATA QUALITY: {tname} F{fid} {minute}' — "
                    f"SOT={sot}/{total_shots} ({_sot_ratio:.0%}) is suspiciously high "
                    f"(API-Football may be inflating SOT vs real data)"
                )
            elif total_shots < 4 and sot >= 3:
                _sot_suspicious = True
                log.warning(
                    f"  SOT DATA QUALITY: {tname} F{fid} {minute}' — "
                    f"SOT={sot} but only {total_shots} total shots (unrealistic ratio)"
                )

        tier, trend, sot_rate = classify_signal(
            sot, state, minute, gps=gps, accel_count=accel_count,
            inside_box_ratio=ib_ratio, sustained_count=sustained,
            league_id=fixture["league"]["id"],
        )

        # v10.69: GOAL-SHOT-FREE SAFETY NET (CSKA Sofia post-mortem, Sep 5:
        # signal arrived ~just after the goal because the shot that SCORED
        # was the 3rd SOT). The SOT>=3 safety net now counts only NON-goal
        # shots: effective = sot - (pending + landed goal shots). When the
        # goal shot(s) are what crossed the threshold and GPS is not itself
        # CRITICAL-grade, the candidate is re-classified on the effective
        # count (GPS path can still fire — real pressure passes it).
        _goal_shots_total = (
            min(_pending_goal_sot.get((fid, tid), 0), 2)
            + _goal_sot_landed.get((fid, tid), 0)
        )
        if _goal_shots_total > 0 and sot >= 3 and tier == "CRITICAL":
            _eff_net = sot - _goal_shots_total
            if _eff_net < 3 and gps < GPS_CRITICAL:
                tier, trend, sot_rate = classify_signal(
                    _eff_net, state, minute, gps=gps, accel_count=accel_count,
                    inside_box_ratio=ib_ratio, sustained_count=sustained,
                    league_id=fixture["league"]["id"],
                )
                log.info(
                    f"  v10.69 GOAL-SHOT NET: {tname} {minute}' — SOT={sot} includes "
                    f"{_goal_shots_total} goal shot(s); effective {_eff_net} < 3, "
                    f"safety net silent, GPS path decides ({tier or 'no signal'})"
                )
                if tier != "CRITICAL":
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, "CRITICAL", "GOAL_SHOT_NET",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    if tier is None:
                        continue

        # v10.38: Downgrade suspicious SOT>=3 CRITICAL to GPS-based evaluation.
        # If SOT>=3 triggered CRITICAL but the SOT data looks inflated,
        # require GPS>=70 as additional confirmation (same as EARLY WARNING floor
        # but slightly higher). This prevents sole reliance on bad SOT data.
        # The Celtic case: SOT=3/3=100% would be caught, and GPS=60 < 70
        # would block the false CRITICAL.
        # v10.44d-fix: Also catch low-IB + high-SOT (Galway: SOT=4 IB=14%)
        _ib_suspicious = sot >= 3 and ib_ratio < 0.25
        # v10.44e: SOT >> shots_inside_box (API data corruption).
        # Original: SOT > IB was too aggressive (blocked Sevilla SOT=4 IB=3).
        # A shot on target from outside the box IS possible (long-range effort).
        # But >2 on-target from outside is almost certainly API inflation.
        # v10.44j: Two-tier approach instead of pure HARD BLOCK:
        #   gap 3-4: SOFT CORRECT — use min(SOT, IB+2) as effective SOT, re-classify
        #   gap >= 5: HARD BLOCK — data is truly unreliable
        _excess_sot = sot - shots_inside_box
        _impossible_sot = _excess_sot > 2 and shots_inside_box > 0
        if _impossible_sot:
            if _excess_sot >= 5:
                # HARD BLOCK: gap too large, data is garbage
                log.warning(
                    f"  SOT GUARD HARD BLOCK: {tname} {minute}' — "
                    f"SOT={sot} vs IB={shots_inside_box} (gap={_excess_sot}) is too large, "
                    f"blocking regardless of GPS={gps:.0f}"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "SOT_GUARD_HARD",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                tier = None
                trend = ""
                sot_rate = 0.0
            else:
                # SOFT CORRECT: gap 3-4, use IB+2 as effective SOT cap
                _corrected_sot = min(sot, shots_inside_box + 2)
                log.warning(
                    f"  SOT GUARD SOFT CORRECT: {tname} {minute}' — "
                    f"SOT={sot} vs IB={shots_inside_box} (gap={_excess_sot}), "
                    f"using effective SOT={_corrected_sot} for classification"
                )
                # Re-classify with corrected SOT
                tier, trend, sot_rate = classify_signal(
                    _corrected_sot, state, minute, gps=gps, accel_count=accel_count,
                    inside_box_ratio=ib_ratio, sustained_count=sustained,
                    league_id=fixture["league"]["id"],
                )
                if tier:
                    log.info(
                        f"  SOT GUARD SOFT CORRECT PASS: {tname} {minute}' — "
                        f"corrected SOT={_corrected_sot} still qualifies as {tier}"
                    )
                else:
                    log.info(
                        f"  SOT GUARD SOFT CORRECT BLOCK: {tname} {minute}' — "
                        f"corrected SOT={_corrected_sot} does not qualify for signal"
                    )
        elif tier == "CRITICAL" and (_sot_suspicious or _ib_suspicious):
            _reasons = []
            if _sot_suspicious:
                _reasons.append("SOT ratio suspicious")
            if _ib_suspicious:
                _reasons.append(f"IB={ib_ratio:.0%}<25% with SOT={sot}")
            if gps < 70:
                log.warning(
                    f"  SOT GUARD BLOCK: {tname} CRITICAL at {minute}' — "
                    f"{', '.join(_reasons)}, GPS={gps:.0f} < 70, "
                    f"downgrading from CRITICAL to blocked"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "SOT_GUARD",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                tier = None
                trend = ""
                sot_rate = 0.0
            else:
                log.warning(
                    f"  SOT GUARD PASS: {tname} CRITICAL at {minute}' — "
                    f"{', '.join(_reasons)} but GPS={gps:.0f} >= 70 confirms real pressure"
                )

        # v10.10: Log WHY GPS was high but signal was blocked (for threshold tuning)
        # Fixed: was producing empty reason strings when no gate condition matched
        if not tier and gps >= GPS_EARLY_WARNING:
            gate_reason = []
            if ib_ratio < 0.50:
                gate_reason.append(f"IB={ib_ratio:.0%}<50%")
            elif ib_ratio < 0.30:
                gate_reason.append(f"IB={ib_ratio:.0%}<30%")
            _lid = fixture["league"]["id"]
            _eff_floor = 60 + (LEAGUE_GPS_FLOOR_ADJUSTMENT if _lid in LEAGUE_TIER2_IDS else 0)
            if gps < _eff_floor:
                _floor_note = f" (tier2 adj: {_eff_floor})" if _lid in LEAGUE_TIER2_IDS else ""
                gate_reason.append(f"GPS={gps:.0f}<{_eff_floor}{_floor_note}")
            if sustained < 1 and gps < GPS_CRITICAL:
                gate_reason.append("not sustained")
            # v10.10: Always provide a reason when GPS is high enough to be noticed
            if not gate_reason:
                gate_reason.append(f"GPS {gps:.0f} below tier threshold (need SOT incr or higher GPS)")
            log.info(
                f"  GATE SKIP: {tname} GPS={gps:.0f} SOT={sot} — {', '.join(gate_reason)}"
            )

        # v10.59: The _prev_state fetch + ML SHADOW SCORE computation moved
        # UP (before record_pressure_poll / the gates) so blocked candidates
        # and poll records can carry the ML opinion. The goal-detection logic
        # below uses the same _prev_goals / last_goal_minute variables.

        if _prev_goals is not None and team_goals > _prev_goals:
            last_goal_minute = minute
            log.info(
                f"  GOAL DETECTED: {tname} {_prev_goals}->{team_goals} at ~{minute}' "
                f"(post-goal cooldown active for {POST_GOAL_COOLDOWN} min)"
            )
            # v10.44r: Elevate this fixture to 15s polling for 60s.
            # Stats processing detected the goal — but the NEXT stats poll
            # needs to happen fast to pick up any post-goal pressure changes.
            goal_priority_until[fid] = time.time() + GOAL_PRIORITY_WINDOW
            # v10.44r: Record latency measurement timestamps.
            # If discovery already detected this goal, _goal_detect_ts exists.
            # If not (goal happened between discovery cycles), use now as fallback.
            _stats_now = time.time()
            if fid not in _goal_detect_ts:
                _goal_detect_ts[fid] = _stats_now  # fallback: stats was first to see it
                _goal_game_minute[fid] = minute
            if fid not in _goal_stats_recorded:
                _goal_stats_ts[fid] = _stats_now
                _goal_stats_recorded.add(fid)
            # v10.44f: Goal resets signal cooldown — new game state,
            # next pressure buildup is genuinely new.
            if (fid, tid) in team_cooldown_polls:
                del team_cooldown_polls[(fid, tid)]
                log.info(
                    f"  COOLDOWN RESET: {tname} — goal detected, "
                    f"clearing signal cooldown for (fid={fid}, tid={tid})"
                )
            # v10.56: register the goal shot(s) in the pending ledger —
            # the shot that scored must NEVER count as trigger evidence
            # for the next signal (zero advance-warning value).
            _gs_new = _register_goal_sot_pending(fid, tid, team_goals, _prev_goals)
            # v10.84: goal-shot minutes (netted from attempt-burst evidence)
            # and fixture-level goal log (response-window input)
            if _gs_new:
                _gs_list = _goal_shot_minutes.setdefault((fid, tid), [])
                _gs_list.extend([minute] * _gs_new)
                if len(_gs_list) > 12:
                    del _gs_list[:len(_gs_list) - 12]
            _flog = _fixture_goal_log.setdefault(fid, [])
            _flog.append((minute, tid))
            if len(_flog) > 12:
                del _flog[:len(_flog) - 12]

        # v10.56: consume pending goal shots from this poll's SOT rise.
        # The score feed updates first; the SOT counter catches up 1-3 min
        # later. When it rises, goal shots are assumed to land FIRST —
        # only the remainder is genuine (non-goal) pressure.
        _prev_sot_for_ledger = _prev_state.get("last_sot") if _prev_state else None
        _ledger_consumed = _consume_pending_goal_sot(
            fid, tid, sot, _prev_sot_for_ledger)
        genuine_poll_rise_by_team[tid] = (
            (sot - _prev_sot_for_ledger) - _ledger_consumed
            if _prev_sot_for_ledger is not None else 0)
        if _ledger_consumed:
            log.info(
                f"  v10.56 GOAL-SOT LANDED: {tname} — SOT {_prev_sot_for_ledger}->{sot} "
                f"includes {_ledger_consumed} goal shot(s) (excluded from triggers)"
            )

        # v10.44h: Read SOT at the time of last goal from previous state.
        # If no previous goal recorded, sot_at_last_goal stays 0.
        sot_at_last_goal = _prev_state.get("sot_at_last_goal", 0) if _prev_state else 0
        # Update sot_at_last_goal when a new goal is detected THIS poll.
        if _prev_goals is not None and team_goals > _prev_goals:
            # v10.56: goal-INCLUSIVE baseline. The goal shot(s) (+ any earlier
            # goal shots still not in the SOT counter) must never read as
            # "new SOT since goal" — otherwise the goal warns about itself.
            sot_at_last_goal = sot + _pending_goal_sot.get((fid, tid), 0)
            log.info(
                f"  v10.56 SOT-AT-GOAL BASELINE: {tname} — {sot} + "
                f"{_pending_goal_sot.get((fid, tid), 0)} pending goal shot(s) "
                f"= {sot_at_last_goal} (goal shots never count as fresh pressure)"
            )

        # Store state AFTER classification (for next comparison)
        team_state[(fid, tid)] = {
            "last_sot": sot,
            "last_minute": minute,
            "last_xg": xg_str if xg_str != "N/A" else None,
            "last_shots_off_target": shots_off_target,  # v10.10
            "last_total_shots": total_shots,
            "last_shots_inside_box": shots_inside_box,
            "last_corners": corners,
            "last_big_chances": big_chances,  # v10.44d
            "last_possession": possession,
            "last_goals": team_goals,          # v10.36: for post-goal cooldown
            "last_goal_minute": last_goal_minute,  # v10.36: minute of most recent goal
            "sot_at_last_goal": sot_at_last_goal,  # v10.44h: SOT count when goal was scored
        }

        # v10.44f: Update signal cooldown counter.
        # If this team has signaled before AND GPS is now below the signal
        # threshold, increment the "below-threshold poll" counter.
        # If GPS is back above threshold, reset the counter.
        # This tracks whether pressure genuinely died and rebuilt.
        _key = (fid, tid)
        if _key in signaled_teams and _key not in team_cooldown_polls:
            # Team has signaled, not yet in cooldown tracking.
            # Start tracking if GPS drops below signal threshold.
            if gps < SIGNAL_COOLDOWN_GPS_FLOOR:
                team_cooldown_polls[_key] = 1
                log.info(
                    f"  COOLDOWN TRACK: {tname} GPS={gps:.0f} < {SIGNAL_COOLDOWN_GPS_FLOOR} "
                    f"after signal — below-threshold poll 1/{SIGNAL_COOLDOWN_POLLS}"
                )
        elif _key in team_cooldown_polls:
            if gps < SIGNAL_COOLDOWN_GPS_FLOOR:
                team_cooldown_polls[_key] += 1
                _cd_count = team_cooldown_polls[_key]
                if _cd_count >= SIGNAL_COOLDOWN_POLLS:
                    log.info(
                        f"  COOLDOWN READY: {tname} GPS={gps:.0f} below threshold "
                        f"for {_cd_count} polls — RE-QUALIFIED for new signal"
                    )
                else:
                    log.info(
                        f"  COOLDOWN TRACK: {tname} GPS={gps:.0f} < {SIGNAL_COOLDOWN_GPS_FLOOR} "
                        f"— below-threshold poll {_cd_count}/{SIGNAL_COOLDOWN_POLLS}"
                    )
            else:
                # GPS back above threshold before reaching cooldown count —
                # pressure never fully died, reset counter.
                del team_cooldown_polls[_key]

        # v10.84: BELOW-TIER CHANNELS — ATTEMPT-BURST + RESPONSE WINDOW.
        # Sep 9-10 post-mortem: 18/20 goals had no <=15m warning; at 10 the SOT
        # tier was already met but freshness/dampener/goal-shot gates held it;
        # at 3 more the ONLY observable was attempt volume (shots wide/saved
        # never move SOT); 3 more were concede->answer responses. Both
        # channels promote a below-tier team to EARLY WARNING and then run the
        # FULL existing gate gauntlet (losing filter, pre-window, post-goal,
        # freshness, dampener, FastWin, cooldown, min-gap) — no exemptions.
        if not tier and not _data_corrupt and 21 <= minute <= 79:
            _opp_goals_v84 = (sa if is_home_team else sh) or 0
            _rb = _build_recency_fields(
                fid, tid, minute, sot, total_shots,
                shots_inside_box, shots_off_target, xg_value, corners,
                gps, accel_count,
            )
            _btier, _btag = _attempt_burst_evaluate(
                fid, tid, minute, sot,
                _rb.get("shots_delta_10m"), _rb.get("ib_delta_10m"),
                shots_inside_box,
            )
            if _btier:
                tier = _btier
                trend = (trend + " | " if trend else "") + _btag
                log.info(
                    f"  v10.84 ATTEMPT-BURST PROMOTION: {tname} {minute}' — "
                    f"{_btag} (GPS={gps:.0f})"
                )
            else:
                _rtier, _rtag, _rshadow = _response_window_evaluate(
                    fid, tid, minute, team_goals, _opp_goals_v84, sot, gps,
                )
                if _rtier:
                    tier = _rtier
                    trend = (trend + " | " if trend else "") + _rtag
                    log.info(
                        f"  v10.84 RESPONSE PROMOTION: {tname} {minute}' — "
                        f"{_rtag} (SOT={sot} GPS={gps:.0f})"
                    )
                elif _rshadow:
                    log.info(
                        f"  v10.84 RESPONSE SHADOW: {tname} {minute}' — {_rtag} — "
                        f"trailing by 1, the v10.34 losing filter outranks the promotion"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, "EARLY WARNING",
                        "RESPONSE_LOSING", gps, sot, ib_ratio, sh, sa,
                        is_home_team, ml_score=ml_score,
                    )

        # v10.15: Pre-window gate — build baseline but don't signal yet
        # v10.13.1: Data corrupt gate — block signals from bad API data
        if not tier or _data_corrupt:
            if _data_corrupt and tier:
                log.warning(
                    f"  BLOCKED {tier}: {tname} F{fid} — data corrupt flag set, signal suppressed"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "DATA_CORRUPT",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
            continue

        # v10.34: LOSING TEAM FILTER (upgraded from v10.24)
        # Data: losing teams 22.2% full WR vs 64.5% winning, 33.3% drawing.
        # v10.24 blocked EW only. v10.34 also raises the bar for CRITICAL:
        #   GPS >= 80, IB >= 65%, SOT >= 4 -- desperation shots inflate stats
        #   without real danger. Still allowed but tagged.
        team_goals = (sh if is_home_team else sa) or 0
        opp_goals = (sa if is_home_team else sh) or 0
        is_losing = team_goals < opp_goals
        losing_tag = ""
        stale_tag = ""  # v10.35: context-aware CRITICAL staleness tag
        # v10.89: RED-AWARE LOSING RELAXATION — a team trailing by <= 1
        # while the opponent is down a NET man is NOT the 22.2% desperation
        # class the v10.34 gate was built on: an 11v10 chase behaves
        # closer to level play (Slavia 1-0 Lens 64' — Lens 3 SOT chasing
        # 10-man Slavia, LOSING_CRIT-blocked, user call). Both tiers pass
        # at their STANDARD bars, always tagged in the message and the
        # ledger (red_relax) so the class grades itself before tuning.
        # Fail-closed: no events data (None) = strict gate, as before.
        _red_relax = False
        _net_opp_reds = 0
        if is_losing:
            _net_opp_reds = (red_cards_opp_n or 0) - (red_cards_team_n or 0)
            if _net_opp_reds >= 1 and (opp_goals - team_goals) <= 1:
                _red_relax = True
        if _red_relax:
            log.info(
                f"  v10.89 RED-RELAX: {tname} {tier} at {minute}' — "
                f"losing {team_goals}-{opp_goals} but opponent down "
                f"{_net_opp_reds} net man, standard tier bars apply, tagged"
            )
            losing_tag = (
                f"\n\u26a0\ufe0f LOSING {team_goals}-{opp_goals} — "
                f"OPPONENT DOWN {_net_opp_reds} MAN "
                f"(11v{max(11 - _net_opp_reds, 7)}): "
                f"man-advantage chase, v10.89 relaxed bars"
            )
        elif is_losing:
            if tier == "EARLY WARNING":
                log.info(
                    f"  LOSING BLOCK: {tname} {tier} at {minute}' — "
                    f"losing {team_goals}-{opp_goals}, EW blocked for losing teams"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "LOSING_EW",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue
            else:
                # v10.34: Stricter CRITICAL for losing teams
                # GPS >= 90: very high pressure, relax IB (Liverpool GPS=98 IB=50% scored)
                # GPS >= 80: need quality evidence (IB>=65%, SOT>=4)
                if gps >= 90:
                    _losing_pass = sot >= 3
                elif gps >= LOSING_GPS_MIN:
                    _losing_pass = ib_ratio >= LOSING_IB_MIN and sot >= LOSING_SOT_MIN
                else:
                    _losing_pass = False
                if not _losing_pass:
                    log.info(
                        f"  LOSING GATE: {tname} CRITICAL at {minute}' — "
                        f"losing {team_goals}-{opp_goals}, needs "
                        f"GPS>={LOSING_GPS_MIN}({gps:.0f}) IB>={int(LOSING_IB_MIN*100)}%({ib_ratio:.0%}) "
                        f"SOT>={LOSING_SOT_MIN}({sot}), skipping"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "LOSING_CRIT",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue
                # Passed the higher bar -- tag but allow
                losing_tag = (
                    f"\n\u26a0\ufe0f LOSING {team_goals}-{opp_goals} — "
                    f"high bar passed (GPS={gps:.0f} IB={ib_ratio:.0%} SOT={sot})"
                )

        # v10.44: SCORE-STATE DAMPENER — winning teams generate phantom pressure.
        # Evidence: AEK up 4-0 SOT=7 GPS=71 (MISS), Viking up 3-1 SOT=5 GPS=73 (MISS).
        # A team winning comfortably takes low-urgency shots that inflate SOT/GPS
        # without genuine scoring threat. Soft approach: suppress +3+ unless
        # genuinely fresh acceleration. Tag +2 for monitoring.
        # v10.44b-fix: Stale accel fix — require SOT rising in last 5 game minutes
        # instead of any accel_count>0. Ajax 65' (+4, SOT 11->11, accel=3 from
        # 30min-ago acceleration) would now be blocked. Fresh acceleration at
        # 30' (SOT 2->5, rising in 5m window) would still pass.
        goal_diff = team_goals - opp_goals
        if goal_diff >= SCORE_DIFF_SUPPRESS:
            # +3 or more: require FRESH acceleration (SOT rising in last 5 min).
            # This catches blowout stat-padding while preserving teams that are
            # genuinely accelerating NOW (not 30 min ago).
            _sd_state = team_state.get((fid, tid))
            _sd_prev_sot = _sd_state["last_sot"] if _sd_state else 0
            # Build recency to check 5m SOT delta (same as 61'+ freshness engine)
            _sd_recency = _build_recency_fields(
                fid, tid, minute, sot, total_shots,
                shots_inside_box, shots_off_target, xg_value, corners,
                gps, accel_count,
            )
            _sd_sot_d5 = _sd_recency.get("sot_delta_5m") or 0
            # v10.44h: xG escape hatch — research shows xG is the #1 predictor
            # of incoming goals. If xG has risen since last signal, the team is
            # creating BETTER chances, even if SOT count is stable.
            _sd_team_sig = signaled_teams.get((fid, tid))
            _sd_xg_at_sig = _sd_team_sig.get("xg_at_last_signal", None) if _sd_team_sig else None
            _sd_xg_rising = False
            _xg_rise_str = ""
            if _sd_xg_at_sig is not None and xg_value is not None:
                _xg_val = safe_float(xg_value) if isinstance(xg_value, str) else xg_value
                if _xg_val is not None and _xg_val > _sd_xg_at_sig + 0.10:
                    _sd_xg_rising = True
                    _xg_rise_str = f"xG {_sd_xg_at_sig:.2f}->{_xg_val:.2f}"

            # Fresh = SOT actively rising in last 5 min OR since last poll
            # OR xG rising since last signal (quality pressure, research-backed)
            _sd_fresh = (
                _sd_sot_d5 >= 1
                or (sot > _sd_prev_sot and _sd_prev_sot > 0)
                or _sd_xg_rising
            )
            if not _sd_fresh:
                _xg_info = f"xG={xg_value}" if xg_value else "xG=N/A"
                log.info(
                    f"  SCORE DAMPENER: {tname} {tier} at {minute}' — "
                    f"winning {team_goals}-{opp_goals} (+{goal_diff}), no fresh pressure, suppressing"
                    f" (SOT_5m=+{_sd_sot_d5}, SOT {sot}->{sot}, {_xg_info})"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "SCORE_DIFF",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue
            # Fresh pressure present — allow but tag
            _pass_reason = f"SOT_5m=+{_sd_sot_d5}" if _sd_sot_d5 >= 1 else (
                f"SOT {_sd_prev_sot}->{sot}" if sot > _sd_prev_sot and _sd_prev_sot > 0 else _xg_rise_str)
            stale_tag += (
                "\n⚠️ WINNING +" + str(goal_diff) + f" ({team_goals}-{opp_goals}) — "
                f"fresh pressure pass ({_pass_reason})"
            )
            log.info(
                f"  SCORE DAMPENER PASS: {tname} {tier} at {minute}' — "
                f"winning {team_goals}-{opp_goals} (+{goal_diff}) but fresh pressure ({_pass_reason})"
            )
        elif goal_diff == 2:
            # +2: allow but tag for monitoring. Future data may justify stricter treatment.
            stale_tag += "\n⚠️ WINNING +2 (" + f"{team_goals}-{opp_goals}) — monitoring"
            log.info(
                f"  SCORE DAMPENER TAG: {tname} {tier} at {minute}' — "
                f"winning {team_goals}-{opp_goals} (+2), allowing with tag"
            )

        # v10.36: POST-GOAL COOLDOWN
        # Suppress signals for 5 min after team scores (stats inflated by the goal).
        # 5-20 min after: require fresh pressure to re-signal.
        # 20+ min after: normal signal logic (game state has evolved).
        # Context: GIL Vicente 2-0 GPS 98 at 46' (just scored), Fulham 2-3 GPS 100 at 56'.
        if last_goal_minute > 0:
            _min_since_goal = minute - last_goal_minute
            # v10.44h: SOT-since-goal check + pressure check.
            # SOT alone isn't enough — one lucky toe-poke after a goal isn't pressure.
            # Require: SOT increased since goal AND (GPS still meaningful OR building).
            _new_sot_since_goal = sot > sot_at_last_goal
            _post_goal_pressure = gps >= 60 or accel_count >= 1
            _post_goal_genuine = _new_sot_since_goal and _post_goal_pressure

            # v10.84: 0 = the goal and the trigger landed in the SAME stats
            # batch (Liverpool 51' / Benfica 45' / Villarreal 72' class) —
            # the goal-inclusive SOT baseline makes genuine-freshness
            # impossible on that poll, so it now suppresses like any other
            # post-goal poll instead of bypassing the window.
            if 0 <= _min_since_goal <= POST_GOAL_COOLDOWN:
                if _post_goal_genuine:
                    # Genuine post-goal pressure — SOT rose AND pressure is real.
                    _pressure_reason = "GPS "+str(int(gps)) if gps >= 60 else f"accel={accel_count}"
                    # v10.87: wording — say WHAT this is: the goal already
                    # landed, the watch is for the NEXT one (Fenerbahce 53'
                    # post-mortem: a genuine second-goal watch read as a
                    # late warning of the 49' goal because the tag never
                    # said which goal it was about).
                    stale_tag = (
                        f"\n\u26a0\ufe0f POST-GOAL — scored ~{last_goal_minute}' "
                        f"({_min_since_goal}m ago); this watches the NEXT goal: "
                        f"SOT {sot_at_last_goal}->{sot} fresh ({_pressure_reason})"
                    )
                    log.info(
                        f"  POST-GOAL PASS: {tname} {tier} at {minute}' — "
                        f"scored ~{last_goal_minute}' ({_min_since_goal}m ago), "
                        f"SOT {sot_at_last_goal}->{sot} + {_pressure_reason} — genuine pressure"
                    )
                else:
                    # No new SOT since goal, or SOT rose but no real pressure.
                    _reason = ""
                    if not _new_sot_since_goal:
                        _reason = f"SOT still {sot} (was {sot_at_last_goal} at goal)"
                    else:
                        _reason = f"SOT {sot_at_last_goal}->{sot} but GPS={gps:.0f} accel={accel_count} — no real pressure"
                    log.info(
                        f"  POST-GOAL SUPPRESS: {tname} {tier} at {minute}' — "
                        f"scored ~{last_goal_minute}' ({_min_since_goal}m ago), {_reason}"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "POST_GOAL_5M",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue
            elif _min_since_goal <= POST_GOAL_RELEVANCE:
                # After 5 min: require fresh pressure (SOT rising since last poll OR accel).
                _prev_sot = (_prev_state.get("last_sot", 0) or 0) if _prev_state else 0
                # v10.69: the raw `sot > _prev_sot` term counted the goal's
                # OWN shot when it landed late in the SOT counter (1-6 min
                # after the score updates) — CSKA Sofia Sep 5: goal ~20',
                # SOT counter 2->3 at 27' -> "fresh pressure" -> signal
                # 7 min after the goal. The ledger-adjusted genuine rise is
                # the only honest freshness evidence.
                _genuine_rise_20m = genuine_poll_rise_by_team.get(tid, 0)
                _post_goal_fresh = (
                    _new_sot_since_goal       # SOT increased since the goal itself
                    or _genuine_rise_20m > 0  # v10.69: was raw sot > _prev_sot
                    or accel_count >= 1       # any acceleration indicator
                )
                if not _post_goal_fresh:
                    if tier == "CRITICAL":
                        # v10.88: the tag-and-send exception is dead.
                        # "SOT=3 proves danger" was v10.36 wording from
                        # before the goal-shot ledger existed — with it,
                        # SOT=3 vs at-goal=3 (Man Utd Sep 10 33': goal
                        # ~27', prev=3, accel=0) proves ZERO fresh shots
                        # since the goal, and the GPS>=75 CONFIRMED
                        # trigger was completed by the goal's OWN shot.
                        # The v10.56 principle — a goal's own +1 can only
                        # ever CLOSE pressure windows, never open them —
                        # now covers the last loophole: block like every
                        # other tier, record POST_GOAL_STALE, let the next
                        # poll re-decide (genuine fresh pressure
                        # re-qualifies in minutes — Fernandes 43' class).
                        log.info(
                            f"  POST-GOAL BLOCK: {tname} CRITICAL at {minute}' — "
                            f"stale post-goal, no fresh pressure since the ~{last_goal_minute}' "
                            f"goal (SOT={sot} vs at-goal={sot_at_last_goal}, prev={_prev_sot}, "
                            f"accel={accel_count}) — goal-inclusive GPS no longer sends"
                        )
                        _track_blocked_candidate(
                            fid, tid, tname, league, minute, tier, "POST_GOAL_STALE",
                            gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                        )
                        continue
                    else:
                        log.info(
                            f"  POST-GOAL BLOCK: {tname} {tier} at {minute}' — "
                            f"scored ~{last_goal_minute}' ({_min_since_goal}m ago), "
                            f"no fresh pressure (SOT={sot} vs at-goal={sot_at_last_goal}, prev={_prev_sot})"
                        )
                        _track_blocked_candidate(
                            fid, tid, tname, league, minute, tier, "POST_GOAL_20M",
                            gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                        )
                        continue
                else:
                    # v10.88: this PASS was log-only — Como Sep 10 28'
                    # (goal 15', attempt-burst d10+3, SOT flat at 2)
                    # shipped with NO post-goal context, so a genuine
                    # next-goal watch (next goal +10m) read as a late
                    # warning of the 15' goal the user had already seen.
                    # The v10.87 wording pattern now covers this branch
                    # too: say which goal this watches and why it passed.
                    _fresh_why = (
                        f"SOT {sot_at_last_goal}->{sot} fresh"
                        if _new_sot_since_goal
                        else f"GPS accel, attempts rising (SOT flat {sot})"
                        if accel_count >= 1
                        else "genuine SOT rise since last poll"
                    )
                    stale_tag = (
                        f"\n\u26a0\ufe0f POST-GOAL — scored ~{last_goal_minute}' "
                        f"({_min_since_goal}m ago); this watches the NEXT goal: {_fresh_why}"
                    )
                    log.info(
                        f"  POST-GOAL PASS: {tname} {tier} at {minute}' — "
                        f"scored ~{last_goal_minute}' ({_min_since_goal}m ago) but fresh pressure "
                        f"(SOT {sot_at_last_goal}->{sot}, prev={_prev_sot}, accel={accel_count})"
                    )

        # v10.13.2: Extended window — allow early signals (16'-20') if SOT-accelerating
        # v10.19: BUT require CRITICAL GPS (≥75) for EARLY WARNING in early window.
        # A GPS 60 at 18' is noise — same principle as the late-window gate.
        if minute < MINUTE_MIN:
            fid_best_sot = get_fixture_best_sot(fid)
            fid_accel = fid in accelerating_fixtures or fid in pressure_accelerating
            is_extended_early = minute >= EXTENDED_MIN and (fid_accel or fid_best_sot >= 2)

            if not is_extended_early:
                log.info(
                    f"  PRE-WINDOW: {tname} {tier} at {minute}' (need {MINUTE_MIN}'+) — "
                    f"building baseline, SOT={sot} GPS={gps:.0f}"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "PRE_WINDOW",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue

            # Early-window GPS gate: require CRITICAL for EARLY WARNING signals
            if tier == "EARLY WARNING" and gps < GPS_CRITICAL:
                log.info(
                    f"  EARLY GATE: {tname} GPS={gps:.0f} at {minute}' — "
                    f"early window requires GPS ≥ {GPS_CRITICAL} (got {gps:.0f}), skipping"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "EARLY_EW_GATE",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue

            log.info(
                f"  EXTENDED EARLY: {tname} {tier} at {minute}' (accel/sot>={2}) — "
                f"SOT={sot} GPS={gps:.0f}, allowing early signal"
            )
            # Don't continue — fall through to signal sending

        # v10.34: FRESHNESS GATE — late-window signals need evidence of
        # ACTIVE pressure, not just cumulative stats.
        # Evidence: GPS 100 signals went 0/4 (stale domination).
        # Fresh late signals (Hajduk 82' → 83' goal) still hit.
        #
        # 61-75': require fresh pressure (SOT rising, GPS rising, or accel)
        #   OR very strong current indicators (GPS>=85 + IB>=60%)
        # 76-85': same but stricter — also require IB>=60%
        # 86'+:   hard stop unless event-confirmed SOT burst
        if minute >= FRESHNESS_HARD_STOP:
            # 86'+: only allow with event-confirmed SOT burst (v10.31 events_sot > stats_sot)
            # v10.56: goal shots NEVER count toward the burst — a goal event
            # pushing events ahead of stats is not pressure to warn about,
            # it is the thing itself (zero advance-warning value).
            _ev_goal_sot = event_goal_by_team.get(tid, 0)
            _ev_sot_nongoal = max(0, events_sot - _ev_goal_sot)
            is_event_burst = (_ev_sot_nongoal > stats_sot and _ev_sot_nongoal >= 3)
            if not is_event_burst:
                log.info(
                    f"  FRESHNESS HARD STOP: {tname} {tier} at {minute}' — "
                    f"{FRESHNESS_HARD_STOP}'+ blocked (SOT={sot} events_sot={events_sot} "
                    f"non-goal={_ev_sot_nongoal})"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "FRESH_HARD_STOP",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue
            log.info(
                f"  FRESHNESS {FRESHNESS_HARD_STOP}'+ EXCEPTION: {tname} {tier} at {minute}' — "
                f"event-confirmed GENUINE SOT burst (stats={stats_sot} events={events_sot} "
                f"non-goal={_ev_sot_nongoal})"
            )

        elif minute >= FRESHNESS_MINUTE:
            # 61-85': require evidence of fresh pressure
            # v10.48: LATE EW FLOOR — EARLY WARNING at 61'+ fired at only
            # 29% full WR (2/7). GPS 55-74 pressure is too weak to trust
            # that late in the game. Late window now requires CRITICAL-level
            # GPS for the EW tier. Tier quality gate — applies regardless of
            # whether recency history exists (covers the NODATA path too).
            if tier == "EARLY WARNING" and gps < GPS_CRITICAL:
                log.info(
                    f"  LATE EW FLOOR: {tname} EARLY WARNING GPS={gps:.0f} at {minute}' — "
                    f"late window requires GPS >= {GPS_CRITICAL} for EW tier, blocking"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "LATE_EW_FLOOR",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue
            # v10.34: If we don't have enough poll history to compute
            # recency (first/second poll for this team), skip the gate.
            # The gate should only block when history EXISTS and shows staleness.
            _recency_check = _build_recency_fields(
                fid, tid, minute, sot, total_shots,
                shots_inside_box, shots_off_target, xg_value, corners,
                gps, accel_count,
            )
            _sot_d5 = _recency_check.get("sot_delta_5m")
            _gps_chg = _recency_check.get("gps_change")
            _has_recency_data = (_sot_d5 is not None or _gps_chg is not None)
            _sot_d5 = _sot_d5 or 0
            _ib_76 = minute >= 76  # stricter IB gate for 76'+

            if _has_recency_data:
                # We HAVE history — enforce freshness
                                # v10.48: LATE_OVERRIDE TRIAGE — late signals hit at only
                # 27% full WR (6/22). The old OR-chain let ONE weak indicator
                # (GPS drifting +0.5, or a single accel tick while SOT sat
                # flat for 10 minutes) qualify a late signal. Late misses
                # cluster exactly there: avg GPS 84 but SOT not moving.
                # v10.48 requires REAL current danger at 61'+:
                #   - SOT actually rose in the last ~5 game minutes, OR
                #   - overwhelming current dominance (GPS>=85 + IB>=60%)
                # GPS-drift-only and accel-tick-only qualifiers are gone.
                # v10.84: ATTEMPT-FLOW FRESHNESS — SOT freezes exactly when a
                # team peppers the goal wide or saved (Sep 9: Barcelona 58',
                # Arsenal 75' class); genuine attempt flow is honest
                # freshness evidence, net of goal shots.
                _att_d5 = _recency_check.get("shots_delta_5m") or 0
                _att_d10 = _recency_check.get("shots_delta_10m") or 0
                _att_d5_eff = _att_d5 - _goal_shots_in_window(fid, tid, minute, 5)
                _att_d10_eff = _att_d10 - _goal_shots_in_window(fid, tid, minute, 10)
                _has_fresh_pressure = (
                    _sot_d5 >= FRESHNESS_SOT_DELTA           # SOT rising in last ~5 min (required)
                    or (gps >= 85 and ib_ratio >= 0.60)         # very strong current pressure
                    or _att_d5_eff >= 2                        # v10.84: 2+ genuine attempts in 5m
                    or _att_d10_eff >= ATTEMPT_BURST_MIN       # v10.84: burst-level attempt flow
                )

                _has_quality = (
                    ib_ratio >= FRESHNESS_IB_FLOOR if _ib_76 else True
                )

                if not _has_fresh_pressure:
                    # v10.41: BLOCK ALL stale signals at 61'+ (no SOT exception).
                    # Data: high SOT + stale pressure = miss (Bodo 69' SOT9 RR0.11, etc.).
                    # Freshness required regardless of SOT count.
                    log.info(
                        f"  FRESHNESS BLOCK: {tname} {tier} at {minute}' — "
                        f"no fresh pressure (SOT_5m=+{_sot_d5} att_d5={_att_d5_eff} "
                        f"att_d10={_att_d10_eff} GPS_chg={_gps_chg} accel={accel_count} GPS={gps:.0f} SOT={sot})"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "FRESHNESS_61",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue

                if not _has_quality:
                    # v10.41: BLOCK ALL low-IB signals at 76'+ (no SOT exception).
                    # Accumulated SOT with poor shot placement = not dangerous.
                    log.info(
                        f"  FRESHNESS IB BLOCK: {tname} {tier} at {minute}' — "
                        f"IB={ib_ratio:.0%} < {int(FRESHNESS_IB_FLOOR*100)}%, SOT={sot}"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "FRESH_IB_76",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue

                # v10.40: Late-window GPS floor for CRITICAL signals.
                # Data: LATE_OVERRIDE avg GPS=80, but signals at GPS 64-75 dragged
                # winrate down. Require GPS>=80 for 61-79' CRITICAL signals.
                # SOT>=5 exception: extreme danger overrides GPS floor.
                if tier == "CRITICAL" and minute >= FRESHNESS_MINUTE and gps < 80 and sot < 5:
                    log.info(
                        f"  LATE GPS FLOOR: {tname} CRITICAL GPS={gps:.0f} at {minute}' — "
                        f"late window requires GPS>=80 (got {gps:.0f}), SOT={sot}<5, blocking"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "LATE_GPS_FLOOR",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue

                log.info(
                    f"  FRESHNESS PASS: {tname} {tier} at {minute}' — "
                    f"SOT_5m=+{_sot_d5} GPS_chg={_gps_chg} accel={accel_count}"
                )
            else:
                # No recency data — allow but log for monitoring
                log.info(
                    f"  FRESHNESS NODATA: {tname} {tier} at {minute}' — "
                    f"no 5m history, allowing (GPS={gps:.0f} accel={accel_count})"
                )

        # --- v9.5.4: Signal limit rules ---
        team_sig = signaled_teams.get((fid, tid))
        sig_count = team_sig["count"] if team_sig else 0

        # v10.24: STALE STATS SUPPRESSION — don't re-signal if nothing changed.
        # Rizespor fired 3 identical signals (42', 45', 45') with same
        # SOT=3, IB=0.429, GPS=57.3 because stats didn't change between polls.
        if sig_count >= 1 and team_sig:
            last_sot_sig = team_sig.get("sot_at_last_signal", -1)
            last_time_sig = team_sig.get("last_signal_time", 0)
            seconds_since_last = time.time() - last_time_sig
            if (sot == last_sot_sig
                    and round(ib_ratio, 1) == round(
                        team_sig.get("last_ib_ratio", -1), 1)
                    and round(gps) == round(
                        team_sig.get("last_gps", -1))
                    and seconds_since_last < 300):
                log.info(
                    f"  STALE SUPPRESS: {tname} {tier} at {minute}' — "
                    f"stats unchanged from last signal "
                    f"(SOT={sot} IB={ib_ratio:.1f} GPS={gps:.0f}, "
                    f"{int(seconds_since_last)}s ago)"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "STALE_STATS",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue

        # v10.44f: SIGNAL COOLDOWN — prevent re-signaling the same team
        # in the same fixture unless pressure died and rebuilt.
        # Data: STALE signals (same team, GPS never dropped) = 25% WR vs
        # FRESH signals = 50% WR. Zero information gained from repeat signals
        # when GPS sustains above threshold.
        # Re-qualify via: (a) GPS dropped below 55 for 2+ consecutive polls,
        # or (b) team scored a goal (game state reset),
        # or (c) v10.44i: PRESSURE BUILDUP — SOT rose ≥3 or xG rose ≥0.50
        #     since last signal (genuinely new danger, not same pressure).
        if sig_count >= 1:
            # v10.44l: GOAL RESET — if team scored since last signal,
            # game state changed completely. Cooldown is irrelevant.
            _cd_goals_at_sig = team_sig.get("goals_at_last_signal", 0) if team_sig else 0
            _cd_goals_since = team_goals - _cd_goals_at_sig
            if _cd_goals_since > 0:
                log.info(
                    f"  COOLDOWN GOAL RESET: {tname} {tier} at {minute}' — "
                    f"scored {_cd_goals_since} goal(s) since signal ({_cd_goals_at_sig}->{team_goals}), new game state, bypassing cooldown"
                )
                # Clear cooldown tracking — fresh start
                if (fid, tid) in team_cooldown_polls:
                    del team_cooldown_polls[(fid, tid)]
            else:
                _cd_polls = team_cooldown_polls.get((fid, tid), 0)
                _cd_qualified = _cd_polls >= SIGNAL_COOLDOWN_POLLS
                # v10.44i: Pressure buildup override
                _cd_sot_at_sig = team_sig.get("sot_at_last_signal", 0) if team_sig else 0
                _cd_xg_at_sig = team_sig.get("xg_at_last_signal", None) if team_sig else None
                # v10.56: goal shots excluded — the rise must be GENUINE
                # pressure. (Pre-signal goals whose shots land late in the
                # SOT counter are subtracted via the landed ledger.)
                _cd_sot_rise_genuine = _genuine_sot_jump(team_sig, sot, fid, tid)
                _cd_xg_val = safe_float(xg_value) if isinstance(xg_value, str) else xg_value
                _cd_xg_rise = 0.0
                if _cd_xg_at_sig is not None and _cd_xg_val is not None:
                    _cd_xg_rise = _cd_xg_val - _cd_xg_at_sig
                # v10.46: Minimum time gap — a SOT/xG jump arriving <180s after
                # the last signal is almost always a stats-API catch-up
                # (lagged stats land in one poll), not a real pressure spell.
                # Benfica 46'->47' re-signal was exactly this failure mode.
                _cd_seconds_since = (time.time() - (team_sig.get("last_signal_time", 0) or 0)) if team_sig else 999999
                _cd_gap_ok = _cd_seconds_since >= SIGNAL_MIN_GAP_SECONDS
                if (_cd_sot_rise_genuine >= 3 or _cd_xg_rise >= 0.50) and not _cd_gap_ok:
                    log.info(
                        f"  COOLDOWN BUILDUP GAP BLOCK: {tname} {tier} at {minute}' — "
                        f"jump (SOT +{_cd_sot_rise_genuine} genuine, xG +{_cd_xg_rise:.2f}) arrived "
                        f"{int(_cd_seconds_since)}s after last signal (<{SIGNAL_MIN_GAP_SECONDS}s) "
                        f"— stats catch-up, not new pressure"
                    )
                _cd_buildup = (_cd_sot_rise_genuine >= 3 or _cd_xg_rise >= 0.50) and _cd_gap_ok
                if _cd_buildup:
                    _buildup_reason = []
                    if _cd_sot_rise_genuine >= 3:
                        _buildup_reason.append(f"SOT {_cd_sot_at_sig}->{sot} (+{_cd_sot_rise_genuine} genuine)")
                    if _cd_xg_rise >= 0.50:
                        _buildup_reason.append(f"xG {_cd_xg_at_sig:.2f}->{_cd_xg_val:.2f}")
                    log.info(
                        f"  COOLDOWN BUILDUP PASS: {tname} {tier} at {minute}' — "
                        f"genuine pressure buildup ({', '.join(_buildup_reason)}) overrides cooldown"
                    )
                    # Don't clear cooldown polls — still track, but allow THIS signal
                elif not _cd_qualified:
                    log.info(
                        f"  COOLDOWN BLOCK: {tname} {tier} at {minute}' — "
                        f"GPS={gps:.0f} SOT={sot} sig#{sig_count}, "
                        f"pressure never dropped below {SIGNAL_COOLDOWN_GPS_FLOOR} "
                        f"for {SIGNAL_COOLDOWN_POLLS}+ polls (cd={_cd_polls}) "
                        f"(fixture {fid})"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "COOLDOWN",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue
                else:
                    log.info(
                        f"  COOLDOWN PASS: {tname} {tier} at {minute}' — "
                        f"RE-QUALIFIED after {SIGNAL_COOLDOWN_POLLS}+ polls below threshold "
                        f"(GPS dropped and rebuilt to {gps:.0f})"
                    )
                    # Clear cooldown — team is now re-qualified, next repeat
                    # will need another drop-rebuild cycle.
                    del team_cooldown_polls[(fid, tid)]

        # --- v9.5.8: First-signal-only on busy days ---
        # v10.19.3: Exception override — if pressure is EXPLODING after the 1st signal,
        # allow a 2nd signal even on busy days. Keeps polling for data collection.
        # Exceptional = SOT burst (2+ in one poll) OR GPS>=85 with pressure acceleration
        # OR v10.44i: pressure buildup (SOT ≥3 or xG ≥0.50 since last signal).
        if is_first_signal_only_mode() and sig_count >= 1:
            _fso_xg_at_sig = team_sig.get("xg_at_last_signal", None) if team_sig else None
            # v10.56: genuine (goal-shot-free) rise — the goal itself is
            # never "exceptional pressure" to warn about
            _fso_sot_rise_genuine = _genuine_sot_jump(team_sig, sot, fid, tid)
            _fso_xg_val = safe_float(xg_value) if isinstance(xg_value, str) else xg_value
            _fso_xg_rise = 0.0
            if _fso_xg_at_sig is not None and _fso_xg_val is not None:
                _fso_xg_rise = _fso_xg_val - _fso_xg_at_sig
            _fso_buildup = _fso_sot_rise_genuine >= 3 or _fso_xg_rise >= 0.50
            is_exceptional = (
                (fid in genuine_burst_fixtures)
                or (gps >= 85 and (fid in pressure_accelerating or fid in accelerating_fixtures))
                or _fso_buildup
            )
            if not is_exceptional:
                log.info(
                    f"  SKIP {tier}: {tname} - "
                    f"{sot} SOT GPS={gps:.0f} (first-signal-only mode, {total_matches_today} games) "
                    f"(fixture {fid})"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "FIRST_ONLY",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue
            _override_reason = (
                'SOT-BURST' if fid in genuine_burst_fixtures
                else ('BUILDUP' if _fso_buildup else 'GPS+ACCEL')
            )
            log.info(
                f"  EXCEPTIONAL OVERRIDE: {tname} - "
                f"{sot} SOT GPS={gps:.0f} {_override_reason} "
                f"(allowing sig#{sig_count+1} in first-signal-only mode, fixture {fid})"
            )

        if sig_count >= 2:
            last_signal_sot = team_sig.get("sot_at_last_signal", 0)
            sot_jump = sot - last_signal_sot
            # v10.56: GENUINE jump — goal shots that landed since the last
            # signal are excluded. A jump made purely of goal shot(s) is a
            # warning about a goal that ALREADY happened: blocked, always.
            _goal_shots_in_jump = sot_jump - _genuine_sot_jump(team_sig, sot, fid, tid)
            sot_jump_genuine = sot_jump - _goal_shots_in_jump
            seconds_since = time.time() - team_sig.get("last_signal_time", time.time())
            # v10.19.3: Speed bonus — fast SOT jumps earn a GPS discount.
            # A +1 SOT jump in <3 min is strong pressure acceleration even if
            # GPS hasn't crossed 75 yet. Slower +1 jumps still need GPS>=75.
            fast_jump = sot_jump_genuine == 1 and seconds_since < 180
            gps_floor = GPS_CRITICAL - 15 if fast_jump else GPS_CRITICAL
            if sot_jump_genuine < 1 or (sot_jump_genuine < 2 and gps < gps_floor):
                if sot_jump_genuine < 1 and _goal_shots_in_jump > 0:
                    reason = (f"SOT +{sot_jump} but {_goal_shots_in_jump} was the goal "
                              f"shot itself — no new shots to warn about")
                elif sot_jump_genuine < 1:
                    reason = f"no SOT increase (still {sot})"
                elif fast_jump:
                    reason = f"+{sot_jump_genuine} in {int(seconds_since)}s but GPS {gps:.0f} < {int(gps_floor)}"
                else:
                    reason = f"+{sot_jump_genuine} too slow, GPS {gps:.0f} < {int(gps_floor)} (need +2 or GPS>={int(gps_floor)})"
                log.info(
                    f"  BLOCKED {tier}: {tname} - "
                    f"{sot} SOT ({reason}) "
                    f"(sig #{sig_count + 1}, fixture {fid})"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier, "NO_SOT_JUMP",
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue

            is_home_check = (tid == home_tid)
            current_goals = (
                fixture["goals"]["home"] if is_home_check
                else fixture["goals"]["away"]
            ) or 0
            goals_at_last = team_sig.get("goals_at_last_signal", current_goals)
            if current_goals > goals_at_last:
                if sot_jump_genuine >= 1:
                    # v10.33: Team scored AND SOT still rising —
                    # "they scored and they're STILL coming."
                    # v10.56: the jump must be GENUINE (post-goal shots);
                    # the goal shot itself no longer counts as escalation.
                    # Pressure is escalating, not resolved. Let signal through.
                    log.info(
                        f"  GOAL-PRESSURE CONTINUES: {tname} — scored "
                        f"{current_goals - goals_at_last} goal(s) and SOT +{sot_jump_genuine} "
                        f"genuine (goal shot excluded), pressure escalating "
                        f"(sig #{sig_count + 1}, fixture {fid})"
                    )
                else:
                    # Scored and the only SOT increase was the goal shot —
                    # pressure likely resolved, block.
                    signaled_teams[(fid, tid)]["goals_at_last_signal"] = current_goals
                    log.info(
                        f"  BLOCKED {tier}: {tname} - "
                        f"{sot} SOT (only +{sot_jump} = the goal shot, nothing new) "
                        f"after scoring {current_goals - goals_at_last} goal(s) "
                        f"— pressure likely resolved (sig #{sig_count + 1}, fixture {fid})"
                    )
                    _track_blocked_candidate(
                        fid, tid, tname, league, minute, tier, "SCORED_STALL",
                        gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                    )
                    continue

        # --- v10.72: SHADOW GATES (log-only unless the hard flags are flipped) ---
        # Two would-suppress classes from the Sep 4-6 outcome data:
        #   DAMP   — team winning by 2+ with GPS<85 (23% full WR vs 31% others)
        #   LATE75 — signal at 75'+ (18% full WR, too little time to convert)
        # Shadow mode: the signal still goes out UNCHANGED, but the log line
        # and the outcome record carry the tag, so a later re-analysis can
        # compute exactly what a hard gate WOULD have blocked. Flip
        # DAMPENER_HARD_GATE / LATE_HARD_GATE to True to make them real
        # gates — no other code changes needed.
        _shadow_blocks: list[str] = []
        if goal_diff >= DAMPENER_SHADOW_LEAD and gps < DAMPENER_SHADOW_GPS_MAX:
            _shadow_blocks.append("DAMP")
        if minute >= LATE_SHADOW_MINUTE:
            _shadow_blocks.append("LATE75")
        if _shadow_blocks:
            _hard_damp = DAMPENER_HARD_GATE and "DAMP" in _shadow_blocks
            _hard_late = LATE_HARD_GATE and "LATE75" in _shadow_blocks
            if _hard_damp or _hard_late:
                _hard_names = [n for n, on in (("DAMP", _hard_damp), ("LATE75", _hard_late)) if on]
                log.info(
                    f"  v10.72 SHADOW HARD GATE: {tname} {tier} at {minute}' — "
                    f"{'+'.join(_hard_names)} rule ACTIVE, blocking "
                    f"(winning {team_goals}-{opp_goals} +{goal_diff}, GPS={gps:.0f}, SOT={sot})"
                )
                _track_blocked_candidate(
                    fid, tid, tname, league, minute, tier,
                    "SHADOW_" + "_".join(_hard_names),
                    gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
                )
                continue
            for _sb in _shadow_blocks:
                _shadow_tags[_sb] = _shadow_tags.get(_sb, 0) + 1
            log.info(
                f"  v10.72 SHADOW: {tname} {tier} at {minute}' — "
                f"{'+'.join(_shadow_blocks)} would suppress "
                f"(winning {team_goals}-{opp_goals} +{goal_diff}, GPS={gps:.0f}, "
                f"minute {minute}') — SENT ANYWAY (shadow mode)"
            )

        # --- Signal passes all checks, send it ---
        is_new_team = sig_count == 0
        is_home_sg = (tid == home_tid)
        goals_now = (
            fixture["goals"]["home"] if is_home_sg
            else fixture["goals"]["away"]
        ) or 0
        if is_new_team:
            _xg_float = safe_float(xg_value) if isinstance(xg_value, str) else xg_value
            signaled_teams[(fid, tid)] = {
                "count": 1, "goals_at_last_signal": goals_now,
                "sot_at_last_signal": sot, "xg_at_last_signal": _xg_float,
                "goal_sot_landed_at_last_signal": _goal_sot_landed.get((fid, tid), 0),  # v10.56
                "last_signal_time": time.time(),
                "last_ib_ratio": ib_ratio, "last_gps": gps,
            }
        else:
            signaled_teams[(fid, tid)]["count"] = sig_count + 1
            signaled_teams[(fid, tid)]["goals_at_last_signal"] = goals_now
            signaled_teams[(fid, tid)]["sot_at_last_signal"] = sot
            signaled_teams[(fid, tid)]["goal_sot_landed_at_last_signal"] = _goal_sot_landed.get((fid, tid), 0)  # v10.56
            _xg_float = safe_float(xg_value) if isinstance(xg_value, str) else xg_value
            signaled_teams[(fid, tid)]["xg_at_last_signal"] = _xg_float
            signaled_teams[(fid, tid)]["last_signal_time"] = time.time()
            signaled_teams[(fid, tid)]["last_ib_ratio"] = ib_ratio
            signaled_teams[(fid, tid)]["last_gps"] = gps
        signaled_fixtures.add(fid)

        # Build the signal message (v10: includes GPS)
        sig_num = sig_count + 1
        # v10.88: "(1st)" is this team's 1st SIGNAL, but next to a 1-0
        # scoreline every reader parses it as a warning for the 1st goal —
        # Man Utd 33' / Como 28' (Sep 10) both READ as first-goal warnings
        # minutes after the first goal landed. Once the signaling team has
        # scored, the header says (next); the POST-GOAL line spells out
        # the detail. Pre-goal signals keep the ordinal (signal count).
        if team_goals > 0:
            sig_label = "next"
        else:
            sig_label = f"{sig_num}{'st' if sig_num == 1 else 'nd' if sig_num == 2 else 'rd' if sig_num == 3 else 'th'}"
        # v10.10: off-target from API directly (more accurate than total - SOT)
        ib_pct = f"{shots_inside_box / total_shots * 100:.0f}%" if total_shots > 0 else "N/A"

        # Determine trigger type for message
        if tier == "EARLY WARNING":
            trigger = f"GPS-TRIGGERED (accel{' sustained' if sustained >= 1 else ''}, IB={ib_ratio:.0%})"
        else:
            # v10.69: label the trigger honestly — "SOT>=3 CONFIRMED" only
            # when 3+ NON-goal shots are on target (goal-shot ledger
            # subtracted); a GPS-justified CRITICAL says so.
            _gs_total = (
                min(_pending_goal_sot.get((fid, tid), 0), 2)
                + _goal_sot_landed.get((fid, tid), 0)
            )
            if sot - _gs_total >= 3:
                trigger = "SOT>=3 CONFIRMED"
            else:
                trigger = f"GPS>={GPS_CRITICAL:.0f} CONFIRMED"

        # v10.49 SPEED FIX: fetch_top_sot_players() moved AFTER send_telegram().
        # It is a live /fixtures/events API round-trip (0.5-2s on cold cache)
        # and used to sit BEFORE the send — pure latency on every single signal.
        # Player data now ships as a fast follow-up message seconds later,
        # and still feeds the outcome record for ML/analysis.
        # v10.65 REVERSAL (user preference — Porto/Betis post-mortem): the
        # player line belongs INSIDE the signal. The fetch runs BEFORE the
        # send again (first signal per fixture: +0.5-2s, 1 credit; repeat
        # signals: cached, free); the inline 3s feed-lag retry is skipped
        # (pre_signal=True) so the worst case stays one round-trip; feeds
        # that catch up late get the separate recovery follow-up message.

        # v10.27: Window tag for signal classification analysis
        # CORE = primary 21-60' window, EARLY_OVERRIDE/LATE_OVERRIDE = edge zones
        if minute < MINUTE_MIN:
            window_tag = "EARLY_OVERRIDE"
            window_label = "⚡ EARLY OVERRIDE"
        elif minute > MINUTE_MAX - 1:  # MINUTE_MAX=61, so >60'
            window_tag = "LATE_OVERRIDE"
            window_label = "⏰ LATE OVERRIDE"
        else:
            window_tag = "CORE"
            window_label = ""

        # v10.58: FRESH BIG CHANCE warning — a big chance created since the
        # previous poll is the strongest single pre-goal sign on the stats
        # side. Only fires when the previous history entry carried a BC
        # value (old-format entries -> no warning -> no false positives).
        _bc_fresh = ""
        _hist_now = team_gps_history.get((fid, tid), [])
        if len(_hist_now) >= 2:
            _bc_prev = _hist_now[-2].get("big_chances")
            if _bc_prev is not None and (big_chances - int(_bc_prev)) >= 1:
                _bc_fresh = (
                    f"\n\u26a0\ufe0f NEW BIG CHANCE just now "
                    f"(+{big_chances - int(_bc_prev)} since last check) "
                    f"— strongest single goal-warning sign"
                )

        # v10.73: RED-CARD VOICE — events-based (player + minute + kind)
        # with the team-stats count fallback. Zero extra credits: the events
        # response was already fetched (fast lane / 75'+ supplement / the
        # Top-SOT pre-signal fetch a few lines below runs _parse_feed which
        # refreshes the extras cache too). A NEW RED CARD within 10 game
        # minutes gets a warning line, and the man-up / man-down context is
        # spelled out — 10v11 shifts goal probability more than almost any
        # other in-play event. Display + record only, NEVER a gate.
        _rc_events = get_red_card_events(fid)
        _rc_block, _rc_team_count, _rc_opp_count, _rc_events_recorded = _build_red_card_block(
            tid, minute, _rc_events, red_card_str,
            (home.get("id"), home["name"]), (away.get("id"), away["name"]),
        )

        # v10.85: SIMPLIFIED SIGNAL — one line per side. Dead display
        # channels dropped (xG when N/A, Big Chances 0 — never delivered on
        # this API plan, corners already in the market block); every value
        # still lands in the ledger outcome record below.
        _xg_bit = f" | xG {xg_str}" if xg_value is not None else ""
        _bc_bit = f" | Big Chances {big_chances}" if (big_chances or 0) > 0 else ""
        _oxg_v = safe_float(opponent_xg) if isinstance(opponent_xg, str) else opponent_xg
        _oxg_bit = f" | xG {_oxg_v:.2f}" if _oxg_v is not None else ""
        msg = (
            f"{tier_emoji(tier)} {tier} GOAL PRESSURE ({sig_label})"
            f"{f' {window_label}' if window_label else ''}\n\n"
            f"{home['name']}  {sh} - {sa}  {away['name']}\n"
            f"{league} | {minute}'\n\n"
            f"{tname}: SOT {sot} | shots {total_shots} (box {ib_pct})"
            f" | off-target {shots_off_target}{_xg_bit}{_bc_bit}\n"
            f"Opp: SOT {opponent_sot}{_oxg_bit}{_bc_fresh}\n\n"
            f"GPS: {gps:.0f}/100 | Trigger: {trigger}"
            f"{_rc_block}"
            f"{losing_tag}"
            f"{stale_tag}"
        )
        if trend:
            msg += f"\nTrend: {trend}"

        # v10.65: Top SOT line INSIDE the signal. Fetch first (pre-signal,
        # no inline 3s retry), feed the per-league census, then embed the
        # line — or the explicit note saying WHY there is no line.
        top_sot_players = fetch_top_sot_players(
            client, fid, tid, max_players=3, team_sot_now=sot,
            pre_signal=True, league_id=fixture["league"]["id"],
        )
        # v10.73: events-census honesty — a line rescued by the players-API
        # fallback is counted as a players-rescue, never as an events line.
        _ts_census_outcome = _last_top_sot_info.get("outcome", "error")
        if _ts_census_outcome == "ok" and _last_top_sot_info.get("source") == "players":
            _ts_census_outcome = "players"
        _update_sot_feed_census(
            fixture["league"]["id"], league,
            _ts_census_outcome,
            listed_sot=_last_top_sot_info.get("listed_sot", 0),
            stats_sot=sot,
        )
        msg += _build_top_sot_segment(
            tname, top_sot_players, fixture["league"]["id"], fid=fid, tid=tid
        )

        # v10.87: GOAL-RACE GUARD (PSV-Shakhtar 45' post-mortem, Sep 10 —
        # "signal at 45, goal at 45, no time to bet"). The stats batch this
        # signal was gated on can be seconds behind reality: the pre-send
        # Top-SOT fetch just parsed the events feed and counted its valid
        # goals (own goals in, disallowed out). If the feed knows MORE goals
        # than the scoreline above, the goal is already in the books and
        # every price in this message is fiction — mute the send, record the
        # candidate, and let the next poll's post-goal gates decide with
        # real data. Shakhtar proof: the Top-SOT line itself said "every
        # listed shooter already scored" at 0-0 — the feed knew the 45'
        # goal before the score did. Reverse direction (score ahead, feed
        # behind — Fenerbahce 49') never trips: feed_goals <= scoreline.
        # Runs BEFORE the odds fetch, so a race-mute saves that credit too.
        _feed_goals_v87 = _last_top_sot_info.get("feed_goals")
        if _feed_goals_v87 is None:
            _feed_goals_v87 = _fixture_valid_goals.get(fid, 0)
        if goal_race_mute(_feed_goals_v87, sh, sa):
            log.info(
                f"  v10.87 GOAL-RACE MUTE: {tname} {tier} at {minute}' — events "
                f"feed knows {_feed_goals_v87} valid goal(s), scoreline says "
                f"{sh}-{sa}; the goal is already in the books, signal not sent"
            )
            _top_sot_retry_queue.pop((fid, tid), None)
            _track_blocked_candidate(
                fid, tid, tname, league, minute, tier, "GOAL_RACE_FEED_AHEAD",
                gps, sot, ib_ratio, sh, sa, is_home_team, ml_score=ml_score,
            )
            continue

        # v10.28: Show recency info in signal message
        _recency_preview = _build_recency_fields(
            fid, tid, minute, sot, total_shots,
            shots_inside_box, shots_off_target, xg_value, corners,
            gps, accel_count,
        )
        _rr = _recency_preview.get("recency_ratio")
        _sot_d5 = _recency_preview.get("sot_delta_5m")
        _sot_d10 = _recency_preview.get("sot_delta_10m")
        _gps_chg = _recency_preview.get("gps_change")
        # v10.85: parts join with " | " — no more orphan " | GPS +0"
        # fragments (a +0.4 change rendered as "+0" on its own line).
        _rec_bits = []
        if _rr is not None:
            _rec_bits.append(f"\U0001f504 Recency: {_rr:.0%}")
        if _sot_d5 is not None:
            _rec_bits.append(f"SOT +{_sot_d5} (5m)")
        if _sot_d10 is not None:
            _rec_bits.append(f"SOT +{_sot_d10} (10m)")
        if _gps_chg is not None and abs(_gps_chg) >= 0.5:
            _rec_bits.append(f"GPS {_gps_chg:+.0f}")
        if _rec_bits:
            msg += "\n\n" + " | ".join(_rec_bits)

        # v10.44g: Goal predictions (Poisson-based)
        _opp_sot_int = safe_int(opponent_sot) if isinstance(opponent_sot, str) else (opponent_sot or 0)
        _opp_xg_val = safe_float(opponent_xg) if isinstance(opponent_xg, str) else None
        _goal_pred = compute_goal_predictions(
            minute=minute,
            signal_team_xg=xg_value,
            signal_team_sot=sot,
            opponent_xg=_opp_xg_val,
            opponent_sot=_opp_sot_int,
            score_home=sh or 0,
            score_away=sa or 0,
            is_home_signal=is_home_sg,
            gps=gps,  # v10.69: hotness lift (v10.74: shrunk 0.8->0.2) + late-game blend
            team_reds=_rc_team_count,  # v10.74: red-card lambda adjustment (None-safe)
            opp_reds=_rc_opp_count,
        )
        _current_goals = (sh or 0) + (sa or 0)
        _score_note = f" ({_current_goals} scored)" if _current_goals > 0 else ""
        # v10.69: ADAPTIVE LINES — only UNDECIDED over lines are shown. A
        # 2-2 game displays O4.5/O5.5/O6.5; the decided O2.5/O3.5 "100%"
        # rows carried zero information. BTTS is hidden once decided.
        # v10.74: the DISPLAYED percentages are the CALIBRATED ones (model
        # blended 50/50 with the empirical game-state table) — the raw
        # model was ~20pp overconfident in the Sep 1-7 backtest (74.5%
        # predicted vs 54.1% landed on undecided O2.5, n=170).
        _lines_str = " | ".join(
            f"O{_l:.1f}: {_p:.0%}" for _l, _p in _goal_pred["over_lines_cal"]
        )
        _btts_str = (
            f" | BTTS: {_goal_pred['p_btts']:.0%}" if _goal_pred["p_btts"] < 1.0 else ""
        )
        _cal_note = " (calibrated)" if _goal_pred.get("cal_mode") == "blend" else ""
        # v10.85: one line instead of three — the xG provenance note lives
        # in the ledger (xg_source); the numbers are what the bet needs.
        msg += (
            f"\n\n\U0001f4c8 Projection{_cal_note}: "
            f"{_goal_pred['expected_total_goals']:.1f} goals total{_score_note}"
            f" | {_lines_str}{_btts_str}"
        )

        # v10.74: FT 1X2 PREDICTION — win/draw/loss for the SIGNALED team
        # from the live state: current scoreline + the remaining-goals
        # lambdas (which carry the scoreline push/sit effects AND the
        # red-card lambda adjustment). Informational ONLY: never a gate,
        # never GPS input, zero extra credits. Recorded as pred_ft_* fields
        # and later compared with the resolved FT result for calibration.
        _sig_goals_ft = (sh or 0) if is_home_sg else (sa or 0)
        _opp_goals_ft = (sa or 0) if is_home_sg else (sh or 0)
        _ft = compute_ft_prediction(_goal_pred, _sig_goals_ft, _opp_goals_ft)
        if _ft.get("p_team") is not None:
            _opp_name_ft = away["name"] if is_home_sg else home["name"]
            _rc_adj_note = " | λ adj: RC" if _goal_pred.get("rc_applied") else ""
            msg += (
                f"\n\U0001f3c6 FT: {tname} {_ft['p_team']:.0%} | "
                f"Draw {_ft['p_draw']:.0%} | {_opp_name_ft} {_ft['p_opp']:.0%}"
                f"{_rc_adj_note}"
            )

        # v10.74: PITCH STATE — men on pitch (11 - reds), subs used and
        # injury-labeled subs from the events the bot already polls. The
        # whole line is omitted when no events coverage exists (honesty —
        # never fake zeros). Context for the FT prediction + brain v2.
        _pitch_parts = []
        _ev_extras_ft = get_event_extras(fid)
        _sub_sig = _sub_opp = None
        _inj_sig = _inj_opp = None
        if _rc_team_count is not None:
            _men_sig = max(11 - (_rc_team_count or 0), 7)
            _men_opp = max(11 - (_rc_opp_count or 0), 7)
            if _men_sig != _men_opp:
                _pitch_parts.append(f"On pitch: {_men_sig}v{_men_opp}")
        if _ev_extras_ft is not None:
            _opp_tid_ft = (away.get("id") if is_home_sg else home.get("id"))
            _sub_sig = (_ev_extras_ft.get("subst") or {}).get(tid)
            _sub_opp = (_ev_extras_ft.get("subst") or {}).get(_opp_tid_ft)
            if _sub_sig is not None or _sub_opp is not None:
                _pitch_parts.append(f"Subs used: {_sub_sig or 0}+{_sub_opp or 0}")
            _inj_sig = (_ev_extras_ft.get("injury_subst") or {}).get(tid, 0)
            _inj_opp = (_ev_extras_ft.get("injury_subst") or {}).get(_opp_tid_ft, 0)
            if _inj_sig or _inj_opp:
                _pitch_parts.append(f"\U0001f915 Injury subs: {_inj_sig}+{_inj_opp}")
        if _pitch_parts:
            msg += "\n\U0001f3a4 " + " | ".join(_pitch_parts)

        # v10.78: ODDS IN THE SIGNAL — the betting-decision block. ONE fast
        # quota-guarded capture (for_message=True: single pass, no retry
        # sleep, no suspect re-fetch) runs HERE, after every gate has
        # already passed and after the full message was composed — the
        # odds never influence the signal (v10.36 passive-capture
        # principle preserved; only the message content gains the market
        # + fair prices). ~1s later to Telegram, SAME normal-path credit
        # count as v10.77. The builder is fully defensive: worst case the
        # block is empty and the signal goes out exactly as before.
        _odds_msg = None
        _odds_extras: dict = {}
        try:
            _odds_msg = fetch_signal_odds(
                client, fid, _current_goals, game_minute=minute, for_message=True
            )
            _odds_block, _odds_extras = _build_odds_value_block(
                _odds_msg, _goal_pred, tname, minute=minute
            )
            if _odds_block:
                msg += _odds_block
        except Exception as _oe:
            log.debug(f"  v10.78 odds block failed (signal sent without it): {_oe}")

        # v10.80: CARDS & CORNERS market block — the OVER/UNDER prediction
        # line with odds, built from the SAME odds fetch (zero extra
        # credits) and the SAME batch statistics (both teams, already
        # parsed above). Counts live-update via editMessageText.
        _mkt_block, _mkt_extras = "", {}
        _msg_prefix_80 = msg
        _c80_line = _c80_ov = _c80_un = None
        _n80_line = _n80_ov = _n80_un = None
        _cards_now_80 = (
            (yellow_cards + _opp_yellow_cards)
            if (yellow_cards is not None and _opp_yellow_cards is not None) else None
        )
        _fouls_now_80 = (
            (fouls + _opp_fouls)
            if (fouls is not None and _opp_fouls is not None) else None
        )
        _corners_now_80 = corners + _opp_corners
        try:
            _c80_line, _c80_ov, _c80_un = _pick_main_ou_line(
                (_odds_msg or {}).get("cards_lines") or []
            )
            _n80_line, _n80_ov, _n80_un = _pick_main_ou_line(
                (_odds_msg or {}).get("corners_lines") or []
            )
            _opp_tid_80 = (away_tid if tid == home_tid else home_tid)
            _red80 = (
                safe_int(get_stat(teams_data_by_id.get(tid) or {}, "red_cards"))
                + safe_int(get_stat(teams_data_by_id.get(_opp_tid_80) or {}, "red_cards"))
            )
            _mkt_block, _mkt_extras = _build_market_block(
                game_minute=minute,
                cards_now=_cards_now_80, fouls_now=_fouls_now_80,
                corners_now=_corners_now_80, red_now=_red80,
                sig_losing=bool(is_losing),
                cards_line=_c80_line, cards_ov=_c80_ov, cards_un=_c80_un,
                corners_line=_n80_line, corners_ov=_n80_ov, corners_un=_n80_un,
                team_name=tname, book_name=(_odds_msg or {}).get("bookmaker"),
            )
            if _mkt_block:
                msg += _mkt_block
        except Exception as _me80:
            log.debug(f"  v10.80 market block failed (signal sent without it): {_me80}")
            _mkt_extras = {}

        # v10.49: signal leaves FIRST — full speed to Telegram.
        # v10.80: send_telegram now returns the message_id (int) — the
        # market block registers itself for live count edits here.
        _send_ok = send_telegram(client, msg)
        if isinstance(_send_ok, int) and _mkt_block:
            _prefix_80 = _msg_prefix_80
            if len(msg) > 4000:
                _prefix_80 = _mkt_last_chunk_prefix(msg, _mkt_block)
            _market_block_live[(fid, tid)] = {
                "msg_id": _send_ok, "prefix": _prefix_80, "tid": tid,
                "team_name": tname, "book": (_odds_msg or {}).get("bookmaker"),
                "cards_line": _c80_line, "cards_ov": _c80_ov, "cards_un": _c80_un,
                "corners_line": _n80_line, "corners_ov": _n80_ov, "corners_un": _n80_un,
                "last_counts": (_cards_now_80, _corners_now_80),
                "last_edit": time.time(),
            }
            log.info(
                f"  v10.80 MARKET BLOCK: {tname} F{fid} — cards line "
                f"{_c80_line} / corners line {_n80_line} — live edits on msg "
                f"{_send_ok}"
            )
        if _send_ok:
            log.info(
                f"  SIGNAL {tier}: {tname} - "
                f"{sot} SOT, GPS={gps:.0f}, xG={xg_str} (fixture {fid}, "
                f"{sig_label} signal, {window_tag}, {trigger})"
            )
        # (v10.49 post-send player fetch + separate follow-up message:
        #  REMOVED in v10.65 — the line is embedded in the signal above;
        #  late-catching feeds still get the recovery follow-up message.)

        signals_sent.append({
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "fixture": fid, "team": tname, "league": league,
            "minute": minute, "sot": sot, "xg": xg_str,
            "gps": round(gps, 1),
            "red_cards": red_card_str, "tier": tier,
            "window_tag": window_tag,  # v10.27
            "trend": trend, "is_new": is_new_team,
        })

        # v10.36/v10.68: odds for the LEDGER. v10.78: the pre-send fast
        # capture already produced data on the normal path (same credits
        # as v10.77); a FAILED fast pass falls back to the FULL hardened
        # fetch (3s retry, suspect re-fetch) here — record only, the
        # message already went out honestly saying no price was captured.
        # Signal decision is already final — odds never influence it.
        _total_goals_now = (sh or 0) + (sa or 0)
        if _odds_msg is not None:
            _odds_data = _odds_msg
        else:
            _odds_data = fetch_signal_odds(client, fid, _total_goals_now, game_minute=minute)

        # v10.1: Enriched outcome record with all raw indicators
        opp_goals = (sa if is_home_sg else sh)
        # v10.44q: Compute previous signal time for time_since_prev_signal field
        _prev_sig_time = team_sig.get("last_signal_time", 0) if team_sig else 0
        signal_outcomes.append({
            "fixture_id": fid,
            "team_id": tid,
            "team_name": tname,
            "league": league,
            "signal_time": time.time(),
            "signal_clock": time.strftime("%Y-%m-%d %H:%M"),
            "game_minute": minute,
            "sot": sot,
            "stats_sot_raw": stats_sot,  # v10.31: audit trail
            "events_sot": events_sot,      # v10.31: audit trail
            "total_shots": total_shots,
            "shots_inside_box": shots_inside_box,
            "ib_ratio": round(ib_ratio, 3),
            "shots_off_target": shots_off_target,  # v10.10
            "xg": round(xg_value, 3) if xg_value is not None else None,
            "big_chances": big_chances,  # v10.44d
            "corners": corners,
            "possession": possession,
            "gps": round(gps, 1),
            "gps_restored": components.get("gps_restored"),  # v10.61: BC-redistributed shadow value
            "ml_score": round(ml_score, 1) if ml_score is not None else None,  # v10.59: shadow opinion at signal time
            # v10.60: free-tier KPI expansion at signal time (logging only,
            # null-safe — None = API did not deliver the field)
            "gk_saves": gk_saves, "opp_gk_saves": _opp_gk_saves,
            "fouls": fouls, "opp_fouls": _opp_fouls,
            "offsides": offsides, "opp_offsides": _opp_offsides,
            "yellow_cards": yellow_cards, "opp_yellow_cards": _opp_yellow_cards,
            "total_passes": total_passes, "pass_accuracy": pass_accuracy,
            "blocked_shots": blocked_shots,
            "subst_count": subst_count,
            "subst_latest_minute": subst_latest_minute,
            "card_latest_minute": card_latest_minute,
            # v10.73: events-based red-card context (display + brain v2)
            "red_cards_team": _rc_team_count,
            "red_cards_opp": _rc_opp_count,
            "red_card_events": _rc_events_recorded,
            "red_relax": _red_relax,  # v10.89: man-advantage losing relaxation fired
            "gps_sot": components.get("sot", 0),
            "gps_ib": components.get("inside_box", 0),
            "gps_sv": components.get("shot_vol", 0),
            "gps_xg": components.get("xg", 0),
            "gps_bc": components.get("big_chances", 0),  # v10.44d
            "gps_corners": components.get("corners", 0),  # v10.44d-patch: was missing
            "gps_poss": components.get("possession", 0),  # v10.10: was gps_da
            "gps_accel": components.get("acceleration", 0),
            "sustained": sustained,
            "accel_count": accel_count,
            "tier": tier,
            "window_tag": window_tag,  # v10.27: CORE / EARLY_OVERRIDE / LATE_OVERRIDE
            "goals_at_signal": goals_now,
            "opponent_goals_at_signal": opp_goals,
            "is_home": is_home_sg,
            "scoreline": "winning" if goals_now > opp_goals else "drawing" if goals_now == opp_goals else "losing",  # v10.34: scoreline context
            "is_losing": is_losing,  # v10.34: for WR analysis
            "is_stale_critical": bool(stale_tag),  # v10.35: track stale CRITICAL outcomes
            "post_goal_minutes_since": minute - last_goal_minute if last_goal_minute > 0 else None,  # v10.36
            # v10.44e: PRE/POST-GOAL classification for ML accuracy
            "post_goal_tag": (
                "POST_GOAL" if last_goal_minute > 0 and (minute - last_goal_minute) <= 2
                else "STALE_POST_GOAL" if last_goal_minute > 0 and (minute - last_goal_minute) <= POST_GOAL_RELEVANCE
                else "PRE_GOAL"
            ),
            "goal_detected_this_poll": bool(_prev_goals is not None and team_goals > _prev_goals),
            "minutes_remaining": 90 - minute,  # v10.28: natural time ceiling for late signals
            # v10.36: Odds data (passive, never influences signal logic)
            "odds_bookmaker": _odds_data.get("bookmaker") if _odds_data else None,
            "odds_over_line": _odds_data.get("over_line") if _odds_data else None,
            "odds_over_odds": _odds_data.get("over_odds") if _odds_data else None,
            "odds_over_implied": _odds_data.get("over_implied") if _odds_data else None,
            "odds_btts_yes": _odds_data.get("btts_yes_odds") if _odds_data else None,
            "odds_btts_implied": _odds_data.get("btts_implied") if _odds_data else None,
            "odds_match_home": _odds_data.get("match_home_odds") if _odds_data else None,
            "odds_match_away": _odds_data.get("match_away_odds") if _odds_data else None,
            "odds_match_draw": _odds_data.get("match_draw_odds") if _odds_data else None,
            "odds_fetched_at": _odds_data.get("fetched_at") if _odds_data else None,
            "odds_source": _odds_data.get("odds_source") if _odds_data else None,  # v10.68: live vs prematch_fallback
            "odds_suspect": _odds_data.get("suspect") if _odds_data else None,  # v10.68: True = impossible price kept + flagged
            "odds_attempts": _odds_data.get("attempts") if _odds_data else None,  # v10.68: fetch rounds used
            # v10.77: P&L-GRADE flag — True ONLY for a genuine live price (source='live',
            # not suspect). prematch_fallback prices are the Sep-4/Sep-6 stale class
            # (Marseille O4.5 @ 26.0 with 4 goals already in) that inflated paper P&L
            # by ~1,900 EUR on Sep 6 alone; they stay recorded for research but are
            # excluded from every P&L / EV / ROI computation.
            "odds_pnl_grade": bool(_odds_data and _odds_data.get("odds_source") == "live" and not _odds_data.get("suspect")) if _odds_data else None,
            "odds_markets": _odds_data.get("markets_available") if _odds_data else [],
            # v10.78: ODDS-IN-MESSAGE calibration fields — the fair prices
            # shown in the Telegram block, recorded so the next calibration
            # pass grades them against the resolved outcomes (the v10.74
            # raw-vs-cal pattern extended to the team-to-score bet).
            "pred_team_scores": _odds_extras.get("pred_team_scores"),
            "pred_team_scores_cal": _odds_extras.get("pred_team_scores_cal"),
            "odds_ev_pct": _odds_extras.get("odds_ev_pct"),
            "odds_in_msg": bool(_odds_msg is not None),
            # v10.80: CARDS & CORNERS market fields — the v1 heuristic shown
            # in the Telegram block, recorded so FT labels grade every lean
            # (mkt_*_ft_result) and the calibration backtest has full state.
            "mkt_cards_now": _mkt_extras.get("mkt_cards_now"),
            "mkt_fouls_now": _mkt_extras.get("mkt_fouls_now"),
            "mkt_corners_now": _mkt_extras.get("mkt_corners_now"),
            "mkt_red_now": _mkt_extras.get("mkt_red_now"),
            "mkt_cards_line": _mkt_extras.get("mkt_cards_line"),
            "mkt_cards_over_odds": _mkt_extras.get("mkt_cards_over_odds"),
            "mkt_cards_under_odds": _mkt_extras.get("mkt_cards_under_odds"),
            "mkt_cards_proj": _mkt_extras.get("mkt_cards_proj"),
            "mkt_cards_p_over": _mkt_extras.get("mkt_cards_p_over"),
            "mkt_cards_fair_over": _mkt_extras.get("mkt_cards_fair_over"),
            "mkt_cards_lean": _mkt_extras.get("mkt_cards_lean"),
            "mkt_corners_line": _mkt_extras.get("mkt_corners_line"),
            "mkt_corners_over_odds": _mkt_extras.get("mkt_corners_over_odds"),
            "mkt_corners_under_odds": _mkt_extras.get("mkt_corners_under_odds"),
            "mkt_corners_proj": _mkt_extras.get("mkt_corners_proj"),
            "mkt_corners_p_over": _mkt_extras.get("mkt_corners_p_over"),
            "mkt_corners_fair_over": _mkt_extras.get("mkt_corners_fair_over"),
            "mkt_corners_lean": _mkt_extras.get("mkt_corners_lean"),
            "referee": (fixture.get("fixture") or {}).get("referee"),
            "outcome_5min": None,   # v10: expanded windows
            "outcome_10min": None,  # v10: expanded windows
            "outcome_15min": None,
            "outcome_full": None,
            "goal_minute_5": None,
            "goal_minute_10": None,
            "goal_minute_15": None,
            "goal_minute_full": None,
            "sig_num": sig_num,
            # v10.72: shadow-gate tags (None = no would-suppress rule matched;
            # "DAMP" / "LATE75" / "DAMP+LATE75" = the hard-gate candidates)
            "shadow_would_block": "+".join(_shadow_blocks) if _shadow_blocks else None,
            "gps_triggered": tier == "EARLY WARNING",
            "version": BOT_VERSION,  # v10.44d-fix: track version
            "resolved": False,
            "cooldown_requalified": sig_count >= 1,  # v10.44f: True if re-qualified after GPS drop
            # v10.44n: Top SOT players for signal context (v10.58: + shots)
            "top_sot_players": [
                {"name": n, "sot": c, "shots": t} for n, c, t in top_sot_players
            ] if top_sot_players else [],
            # v10.44p: Latency measurement (0 extra credits — reuses top SOT events fetch)
            # detection_game_lag = signal_minute - latest_shot_event_minute
            # Measures how far behind statistics the signal was vs real-time events.
            "latest_shot_event_minute": _latest_sot_event_minute.get(fid),
            "detection_game_lag": (minute - _latest_sot_event_minute[fid]) if fid in _latest_sot_event_minute else None,
            # v10.44q: Enriched signal metadata for ML (0 extra credits)
            # poll_interval: what polling interval was active when signal fired
            # (tells ML whether this was caught fast or slow)
            "poll_interval": get_sot_based_interval(fid, get_stats_interval(get_budget_mode())),
            # last_goal_minute: when this team last scored (0 = never)
            # Separates PRE-GOAL buildup from POST-GOAL inflation signals.
            "last_goal_minute": last_goal_minute if last_goal_minute > 0 else None,
            # time_since_previous_signal: seconds since this team's last signal
            # (None for first signal). Measures whether this is a quick re-signal
            # or a fresh buildup after pressure died.
            "time_since_prev_signal": round(time.time() - _prev_sig_time) if _prev_sig_time > 0 else None,
            # v10.44r: Goal detection latency measurement (0 extra credits)
            # goal_detect_lag: seconds from discovery score-change to signal
            #   (measures how fast the pipeline reacted to the goal)
            # stats_update_lag: seconds from discovery to first stats poll after goal
            #   (measures stats polling responsiveness)
            # goal_in_priority_window: was the fixture in goal-priority 15s mode?
            "goal_detect_lag": round(time.time() - _goal_detect_ts[fid]) if fid in _goal_detect_ts else None,
            "stats_update_lag": round(_goal_stats_ts[fid] - _goal_detect_ts[fid]) if (fid in _goal_stats_ts and fid in _goal_detect_ts) else None,
            "signal_lag": round(time.time() - _goal_stats_ts[fid]) if fid in _goal_stats_ts else None,
            "goal_in_priority_window": fid in goal_priority_until,
            # v10.44g: Goal predictions (Poisson) — saved for ML training
            "pred_proj_xg_signal": _goal_pred["proj_xg_signal"],
            "pred_proj_xg_opponent": _goal_pred["proj_xg_opponent"],
            "pred_expected_total": _goal_pred["expected_total_goals"],
            "pred_over_25": _goal_pred["p_over_25"],
            "pred_over_35": _goal_pred["p_over_35"],
            "pred_over_45": _goal_pred["p_over_45"],
            "pred_btts": _goal_pred["p_btts"],
            "pred_xg_source": _goal_pred["xg_source"],
            "pred_current_total_goals": _goal_pred["current_total_goals"],
            "pred_actual_total_goals": None,  # filled on resolution
            # v10.74: CALIBRATED over predictions (model blended 50/50 with
            # the empirical game-state table) — backtest next week compares
            # these against the raw model fields and the actual landings.
            "pred_over_25_cal": _goal_pred.get("p_over_25_cal"),
            "pred_over_35_cal": _goal_pred.get("p_over_35_cal"),
            "pred_over_45_cal": _goal_pred.get("p_over_45_cal"),
            "pred_cal_mode": _goal_pred.get("cal_mode"),
            # v10.74: FT 1X2 prediction (win/draw/loss for the signaled
            # team; None-safe) + its red-card lambda flag + the factual FT
            # result filled at resolution (pred_ft_actual).
            "pred_ft_sig": _ft.get("p_team"),
            "pred_ft_draw": _ft.get("p_draw"),
            "pred_ft_opp": _ft.get("p_opp"),
            "pred_ft_actual": None,  # filled on resolution
            "pred_ft_rc_applied": _goal_pred.get("rc_applied"),
            # v10.74: pitch-state context (events-based; None = no coverage)
            "opp_subst_count": _sub_opp,
            "injury_subs_team": _inj_sig,
            "injury_subs_opp": _inj_opp,
        })
        # v10.28: Merge enriched recency fields into signal outcome
        recency = _build_recency_fields(
            fid, tid, minute, sot, total_shots,
            shots_inside_box, shots_off_target, xg_value, corners,
            gps, accel_count,
        )
        signal_outcomes[-1].update(recency)
        # v10.11: Save pending outcome immediately (survives restarts)
        save_outcome(signal_outcomes[-1])

        # v10.19.3: In first-signal-only mode, keep polling for data collection
        # and exceptional 2nd signal detection. Don't remove from monitoring.
        if is_first_signal_only_mode():
            log.info(
                f"  FIRST-SIGNAL-DONE: fixture {fid} stays monitored for data + exceptional override "
                f"({len(fast_monitored)} still monitored)"
            )

    # --- v10.53: COLD-START WARM-UP (events backfill) ---
    # First poll of a fixture at 61'+ with no meaningful history = the bot
    # (re)started mid-game. Backfill synthetic history from the events feed
    # so the 61'+ freshness gates see REAL recent shot activity instead of
    # being blind for the first ~5 minutes (Levski derby failure mode).
    if (minute >= FRESHNESS_MINUTE and minute <= EXTENDED_MAX
            and fid not in _coldstart_warmed and best_current_sot >= 1):
        _coldstart_warmed.add(fid)
        _cw_max_hist = max(
            (len(team_gps_history.get((fid, t), [])) for t in (home_tid, away_tid)),
            default=0,
        )
        if _cw_max_hist <= 1:
            _backfill_history_from_events(client, fid, minute, home_tid, away_tid)

    # --- v9.7: Update SOT acceleration flag ---
    if _sot_increased:
        accelerating_fixtures.add(fid)
        log.debug(f"  SOT-Accelerating: fixture {fid} (SOT {prev_best_sot}->{best_current_sot})")
    else:
        accelerating_fixtures.discard(fid)

    # --- v10.17: Update SOT burst flag (2+ SOT jump in one poll) ---
    _sot_burst = (best_current_sot - prev_best_sot) >= 2
    if _sot_burst:
        sot_burst_fixtures.add(fid)
        log.info(
            f"  SOT-BURST: fixture {fid} (SOT {prev_best_sot}->{best_current_sot}, "
            f"+{best_current_sot - prev_best_sot} in one poll)"
        )
    else:
        sot_burst_fixtures.discard(fid)

    # --- v10.56: Update GENUINE burst flag (2+ non-goal SOT jump in one poll) ---
    # Same trigger discipline as the surge/signal gates: goal shots landing
    # in a catch-up poll must not make a fixture look "bursting" — only
    # genuine (non-goal) shots count. Used by the first-signal-only exception.
    _genuine_burst = any(_r >= 2 for _r in genuine_poll_rise_by_team.values())
    if _genuine_burst:
        genuine_burst_fixtures.add(fid)
        log.info(
            f"  GENUINE SOT-BURST: fixture {fid} — non-goal SOT +2 in one poll "
            f"{dict(genuine_poll_rise_by_team)}"
        )
    else:
        genuine_burst_fixtures.discard(fid)

    # --- v10: Update pressure acceleration flag ---
    # Check if ANY team in this fixture has rising GPS + 2+ accelerating indicators
    if is_fixture_pressure_accelerating(fid):
        pressure_accelerating.add(fid)
        log.info(f"  PRESSURE-ACCEL: fixture {fid} (GPS rising + multi-indicator acceleration)")
    else:
        pressure_accelerating.discard(fid)

    # --- v9.7: Dead fixture detection (both teams SOT=0) ---
    # v10.43: Grace period — SOT=0 is normal before 15', don't kill fixtures early
    if best_current_sot == 0 and prev_best_sot == 0 and minute >= 15:
        hg = fixture["goals"]["home"] or 0
        ag = fixture["goals"]["away"] or 0
        dead_fixtures[fid] = (hg, ag, time.time())
        fast_monitored.discard(fid)
        expire_fast_sot(fid)
        accelerating_fixtures.discard(fid)
        pressure_accelerating.discard(fid)
        sot_burst_fixtures.discard(fid)
        genuine_burst_fixtures.discard(fid)
        log.info(
            f"  DEAD: fixture {fid} ({home['name']} vs {away['name']}) "
            f"both teams SOT=0 at {minute}' — stopped polling "
            f"(will revive on score change)"
        )

    # --- v10.44b: Data-dead detection (API has no stats for this match) ---
    # If BOTH teams have shots=0 AND SOT=0 at minute >= DATA_DEAD_MINUTE,
    # the API data provider doesn't cover this match. Mark as data-dead
    # and stop polling. Revived every 5min if stats appear.
    # Different from pressure-dead: these fixtures may have real action,
    # we just can't see it. The 5min revival catches late data appearance.
    if minute >= DATA_DEAD_MINUTE and fid not in data_dead_fixtures:
        all_data_empty = True
        for tname, tstats in teams_data.items():
            ts = safe_int(get_stat(tstats, "total_shots"))
            ss = safe_int(get_stat(tstats, "sot"))
            if ts > 0 or ss > 0:
                all_data_empty = False
                break
        if all_data_empty:
            data_dead_fixtures[fid] = time.time()
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            accelerating_fixtures.discard(fid)
            pressure_accelerating.discard(fid)
            sot_burst_fixtures.discard(fid)
            log.info(
                f"  DATA-DEAD: fixture {fid} ({home['name']} vs {away['name']}) "
                f"both teams shots=0 SOT=0 at {minute}' — API has no stats coverage, "
                f"stopped polling (will re-check every 5min)"
            )


def _update_cached_fixtures(refreshed: list[dict]) -> set[int]:
    """Update cached_fixtures list with fresh data from /fixtures?ids= response.
    Returns set of updated fixture IDs."""
    global cached_fixtures
    refreshed_ids = set()
    for rf in refreshed:
        rf_id = rf["fixture"]["id"]
        for i, cf in enumerate(cached_fixtures):
            if cf["fixture"]["id"] == rf_id:
                cached_fixtures[i] = rf
                refreshed_ids.add(rf_id)
                break
        else:
            cached_fixtures.append(rf)
            refreshed_ids.add(rf_id)
    return refreshed_ids


def check_monitored_stats(
    client: httpx.Client,
    fixture_ids: list[int],
    dd_probes: list[int] | None = None,
) -> bool:
    """v9.7: Unified stats fetch — ONE /fixtures?ids= call replaces everything.

    v10.44c: dd_probes = data-dead fixture IDs to probe for stats revival.
    These are included in the /fixtures?ids= batch (zero extra cost) but
    are NOT processed for signals. If stats appeared, they get revived.

    v9.7 Architecture:
    ====================
    1. Single /fixtures?ids=X-Y-Z call (1 credit) — gets fresh fixture data.
    2. Check if the response includes statistics (API-plan dependent).
    3. If YES: process all fixtures directly from the one response.
       Credit cost: 1 credit for ALL fixtures (was 1+ N before).
    4. If NO: fall back to individual /fixtures/statistics?fixture=X calls.
       Credit cost: 1 + N (same as v9.6.2 in NORMAL mode, but saves 1
       credit in CAREFUL+ modes where v9.6.2 skipped the refresh).

    Credit comparison for 10 monitored fixtures:
    - v9.6.2 NORMAL:  1 (refresh) + 10 (stats) = 11 credits/cycle
    - v9.6.2 CAREFUL: 0 (skip refresh) + 10 (stats) = 10 credits/cycle
    - v9.7 with stats:  1 credit/cycle (11x saving!)
    - v9.7 without stats: 1 + 10 = 11 credits/cycle (same as NORMAL)

    The key improvement: we test ONCE whether the API returns stats
    in /fixtures?ids= responses, then use the optimal path forever.

    Always updates last_stats_check to prevent infinite re-polling.
    """
    global last_stats_check, _ids_endpoint_has_stats

    # v10.44c: Separate data-dead probes from normal monitored fixtures.
    # Probes are included in the API call but not processed for signals.
    probe_set = set(dd_probes or [])
    monitor_ids = [fid for fid in fixture_ids if fid not in probe_set]

    # --- Pre-filter: remove invalid/unmonitorable fixtures ---
    valid_ids = []
    for fid in monitor_ids:
        fixture = find_cached_fixture(fid)
        if not fixture:
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            continue
        if not is_fixture_monitorable(fixture):
            fast_monitored.discard(fid)
            expire_fast_sot(fid)
            _retain_late_fixture(fid, fixture, "monitorable pre-filter")  # v10.67
            continue
        valid_ids.append(fid)

    # Add probes to the API call (they bypass monitorable filter)
    all_api_ids = valid_ids + [fid for fid in probe_set if fid not in valid_ids]

    if not all_api_ids:
        return False

    now = time.time()
    any_success = False
    valid_set = set(valid_ids)
    probe_api_set = set(all_api_ids) - valid_set  # probes not already in valid_ids

    # ================================================================
    # STEP 1: Single /fixtures?ids= call (always — gets fresh data)
    # ================================================================
    ids_str = "-".join(str(fid) for fid in all_api_ids)
    refreshed_fixtures = {}  # fid -> fixture dict from this call

    try:
        batch_data = api_get(client, "/fixtures", {"ids": ids_str})
        batch_response = batch_data.get("response", [])

        # Update cached fixtures with fresh score/minute/data
        refreshed_ids = _update_cached_fixtures(batch_response)
        log.info(
            f"  Batch refresh: {len(refreshed_ids)} fixture(s) updated"
            f"{f' (stats embedded!)' if _ids_endpoint_has_stats is True else ''}"
            f"{f' (testing for stats...)' if _ids_endpoint_has_stats is None else ''}"
        )

        # Index fixtures by ID for easy lookup
        for rf in batch_response:
            rf_id = rf["fixture"]["id"]
            if rf_id in valid_set or rf_id in probe_api_set:
                refreshed_fixtures[rf_id] = rf

    except Exception as e:
        log.warning(f"  Batch /fixtures?ids= failed: {e}")
        # Mark all as checked to prevent retry storm
        for fid in all_api_ids:
            last_stats_check[fid] = now
        return False

    # ================================================================
    # v10.44c: Process data-dead probe results
    # ================================================================
    dd_revived_count = 0
    for fid in probe_api_set:
        rf = refreshed_fixtures.get(fid)
        if not rf:
            continue
        stats_list = rf.get("statistics", [])
        has_stats = False
        if stats_list:
            for s in stats_list:
                for stat in s.get("statistics", []):
                    if (stat.get("type", "") in ("Shots on Goal", "Total Shots")
                            and safe_int(stat.get("value", "0")) > 0):
                        has_stats = True
                        break
                if has_stats:
                    break
        if has_stats:
            data_dead_fixtures.pop(fid, None)
            dd_revived_count += 1
            f = find_cached_fixture(fid)
            fname = (f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}" if f else str(fid))
            log.info(f"  DATA-DEAD REVIVED: {fname} — stats found via probe")
        else:
            # Reset 5-min timer for next probe
            data_dead_fixtures[fid] = now
    if dd_revived_count:
        log.info(f"  -> {dd_revived_count} data-dead fixture(s) revived via probe")

    # ================================================================
    # v10.44g: Process pressure-dead probe results
    # ================================================================
    pd_revived_count = 0
    for fid in probe_api_set:
        if fid not in dead_fixtures:
            continue  # only process pressure-dead fixtures here
        rf = refreshed_fixtures.get(fid)
        if not rf:
            continue
        stats_list = rf.get("statistics", [])
        best_probe_sot = 0
        if stats_list:
            for s in stats_list:
                for stat in s.get("statistics", []):
                    if stat.get("type", "") == "Shots on Goal":
                        val = safe_int(stat.get("value", "0"))
                        if val > best_probe_sot:
                            best_probe_sot = val
        # Revive if SOT has risen from 0 (the death condition)
        if best_probe_sot >= 1:
            dead_fixtures.pop(fid, None)
            pd_revived_count += 1
            f = find_cached_fixture(fid)
            fname = (f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}" if f else str(fid))
            log.info(
                f"  PRESSURE-DEAD REVIVED: {fname} — SOT now {best_probe_sot} "
                f"via stats probe (was 0 at death)"
            )
        else:
            # Reset 5-min timer for next probe
            if fid in dead_fixtures:
                dh, da, _ = dead_fixtures[fid]
                dead_fixtures[fid] = (dh, da, now)
    if pd_revived_count:
        log.info(f"  -> {pd_revived_count} pressure-dead fixture(s) revived via probe")

    # Mark probes as checked (prevents discovery from re-probing)
    for fid in probe_api_set:
        last_stats_check[fid] = now

    # ================================================================
    # STEP 2: Check if statistics are embedded in the response
    # ================================================================
    if _ids_endpoint_has_stats is None and batch_response:
        # First time: test whether any fixture has statistics
        has_any_stats = False
        for rf in batch_response:
            stats = rf.get("statistics")
            if stats and isinstance(stats, list) and len(stats) > 0:
                # Verify it looks like real statistics data
                # (should have team objects with statistics arrays)
                first_entry = stats[0] if stats else {}
                if "team" in first_entry or "statistics" in first_entry:
                    has_any_stats = True
                    break

        _ids_endpoint_has_stats = has_any_stats
        if has_any_stats:
            log.info(
                "  *** v9.7 DETECTION: /fixtures?ids= INCLUDES statistics! ***"
                " *** Single-call mode ACTIVE (massive credit saving) ***"
            )
        else:
            log.info(
                "  v9.7 DETECTION: /fixtures?ids= does NOT include statistics."
                " Using individual /fixtures/statistics calls as fallback."
            )

    # ================================================================
    # STEP 3A: If stats are embedded — process directly (1 credit total!)
    # ================================================================
    if _ids_endpoint_has_stats is True:
        for fid in valid_ids:
            rf = refreshed_fixtures.get(fid)
            if not rf:
                last_stats_check[fid] = now
                continue

            stats = rf.get("statistics")
            if not stats:
                log.debug(f"  Fixture {fid}: no stats in batch response")
                last_stats_check[fid] = now
                continue

            try:
                process_fixture_stats(client, rf)
                any_success = True
            except Exception as e:
                log.warning(f"  Stats processing failed for fixture {fid}: {e}")

            last_stats_check[fid] = now

        log.info(
            f"  v9.7 single-call: processed {len(valid_ids)} fixture(s) "
            f"for 1 credit (stats embedded in /fixtures?ids= response)"
        )
        return any_success

    # ================================================================
    # STEP 3B: Fallback — individual /fixtures/statistics calls
    # ================================================================
    # This path costs 1 (batch above) + N (individual) credits.
    # In v9.6.2 NORMAL mode it was also 1+ N, so same cost.
    # In v9.6.2 CAREFUL+ mode it was 0+ N (skipped refresh), so this
    # costs 1 extra credit but gives fresher score/minute data.
    log.info(
        f"  v9.7 fallback: {len(valid_ids)} fixture(s), "
        f"fetching individual statistics ({1 + len(valid_ids)} credits this cycle)"
    )

    for fid in valid_ids:
        try:
            stats_data = api_get(
                client, "/fixtures/statistics", {"fixture": fid}
            )

            stats_response = stats_data.get("response", [])
            if not stats_response:
                log.debug(f"  Fixture {fid}: empty statistics response")
                last_stats_check[fid] = now
                continue

            any_success = True

            # Use the FRESH fixture data from the batch call (not stale cache)
            rf = refreshed_fixtures.get(fid)
            if not rf:
                rf = find_cached_fixture(fid)
            if not rf:
                last_stats_check[fid] = now
                continue

            # Create merged fixture with statistics injected
            merged = {
                "fixture": rf["fixture"],
                "teams": rf["teams"],
                "goals": rf["goals"],
                "league": rf["league"],
                "statistics": stats_response,
            }
            process_fixture_stats(client, merged)

        except Exception as e:
            log.warning(f"  Stats failed for fixture {fid}: {e}")

        last_stats_check[fid] = now

    return any_success


# ============================================================
# LEAGUE FILTERING (v9.5.4)
# ============================================================

def is_tracked_match(fixture: dict) -> bool:
    """Check if a fixture should be tracked (main 20 leagues OR approved international friendly)."""
    lid = fixture["league"]["id"]
    fid = fixture["fixture"]["id"]
    return lid in LEAGUE_IDS or fid in active_friendly_fixtures


# ============================================================
# DYNAMIC ACTIVE HOURS (v9.5.4)
# ============================================================

def fetch_daily_active_hours(client: httpx.Client) -> bool:
    """Fetch today's fixtures and compute the active monitoring window.
    
    Calls /fixtures?date=YYYY-MM-DD twice per fetch (today + tomorrow).
    Re-checks every 3 hours to catch late-added fixtures and widen window.
    Sets dynamic_active_start/end based on actual kickoff times.
    Returns True if there are matches to monitor today.
    """
    global dynamic_active_start, dynamic_active_end
    global schedule_date, schedule_no_matches, schedule_last_fetch
    global total_matches_today
    global scheduled_window_entries
    global schedule_fixture_ids_loaded
    global cached_tomorrow_date, cached_tomorrow_kickoffs
    global active_friendly_fixtures, todays_tracked_fixtures
    global todays_kickoff_latest_ts  # v10.71: EOD schedule gate
    
    today_bulgaria = datetime.now(BULGARIA_TZ)
    today_str = today_bulgaria.strftime("%Y-%m-%d")
    tomorrow_str = (today_bulgaria + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # v9.7.2: Night gate — never burn credits checking schedule at night
    # European tracked leagues don't play between 01:00-10:00 Bulgaria time.
    # Just return cached result. If new day started at 3 AM, return False
    # (no matches at 3 AM) — will properly fetch when morning comes.
    local_hour = today_bulgaria.hour
    if NIGHT_HOUR_START <= local_hour < NIGHT_HOUR_END:
        if schedule_date == today_str:
            # Same day, already fetched during evening — return cached
            return not schedule_no_matches
        # New day but still night — check if we have cached tomorrow data
        if cached_tomorrow_date == today_str and cached_tomorrow_kickoffs is not None:
            # We cached this day's kickoffs from yesterday's fetch!
            # Set window from cache (0 credits) — real fetch happens before window
            if cached_tomorrow_kickoffs:
                earliest = min(cached_tomorrow_kickoffs)
                latest = max(cached_tomorrow_kickoffs)
                window_start = max(0, int(earliest - (30 / 60)))
                window_end = min(24, int(latest + (100 / 60)) + (1 if (latest + (100 / 60)) % 1 > 0 else 0))
                # Only set window if it makes sense (not stale data pointing to wrong day)
                dynamic_active_start = window_start
                dynamic_active_end = window_end
                total_matches_today = len(cached_tomorrow_kickoffs)
                schedule_no_matches = False
                schedule_date = today_str  # mark as "handled" so we don't re-trigger
                schedule_fixture_ids_loaded = False  # but fixture IDs NOT loaded yet
                log.info(
                    f"v9.7.2 CACHE HIT: {today_str} window from yesterday's fetch — "
                    f"{len(cached_tomorrow_kickoffs)} matches, "
                    f"kickoffs {int(earliest):02d}:{int((earliest % 1) * 60):02d}-"
                    f"{int(latest):02d}:{int((latest % 1) * 60):02d}, "
                    f"active {window_start:02d}:00-{window_end:02d} Bulgaria (0 credits)"
                )
                return True
            else:
                # Cached as "no matches" — truly no games today
                schedule_date = today_str
                schedule_no_matches = True
                schedule_fixture_ids_loaded = False
                log.info(
                    f"v9.7.2 CACHE HIT: {today_str} no matches (from yesterday's fetch, 0 credits)"
                )
                return False
        # No cache available (first run or cache miss) — return False, will fetch at NIGHT_HOUR_END
        return False
    
    # Already fetched today — re-check every 6h to catch late-added fixtures
    # v9.7.2: BUT always fetch if fixture IDs not loaded (cache-set window needs real data)
    if (schedule_date == today_str
            and schedule_fixture_ids_loaded
            and (time.time() - schedule_last_fetch) < SCHEDULE_RECHECK_INTERVAL):
        return not schedule_no_matches
    
    if schedule_date == today_str and schedule_fixture_ids_loaded:
        is_recheck = True
        log.info(f"Re-checking schedule for {today_str} (last fetch >6h ago, looking for late additions)...")
    elif schedule_date == today_str and not schedule_fixture_ids_loaded:
        is_recheck = False
        log.info(f"v9.7.2: Loading fixture IDs for cached window {today_str} (needed for monitoring)...")
    else:
        is_recheck = False
    
    log.info(f"Fetching daily schedule for {today_str} + {tomorrow_str} (2 API calls)...")
    
    try:
        # Query BOTH today and tomorrow (Bulgaria time) to handle timezone edge cases.
        # A match at 22:00 UTC on Aug 11 is 01:00 Aug 12 Bulgaria — API has it
        # under Aug 11, so querying only Aug 12 would miss it.
        all_fixtures = []
        for query_date in [today_str, tomorrow_str]:
            try:
                data = api_get(client, "/fixtures", {"date": query_date})
                all_fixtures.extend(data.get("response", []))
            except Exception as e:
                log.warning(f"  Schedule fetch for {query_date} failed: {e}")
        
        # DEBUG: Log unseen league IDs (collapsed to 1 line to avoid Railway log rate limit)
        unseen_leagues = {}
        for f in all_fixtures:
            lid = f["league"]["id"]
            lname = f["league"].get("name", "?")
            if lid not in LEAGUE_IDS and lid not in unseen_leagues:
                unseen_leagues[lid] = lname
        if unseen_leagues:
            # Show top 5 by fixture count, rest as summary
            league_counts = []
            for lid, lname in sorted(unseen_leagues.items()):
                count = sum(1 for f in all_fixtures if f["league"]["id"] == lid)
                league_counts.append((count, lid, lname))
            league_counts.sort(reverse=True)
            top = league_counts[:5]
            top_str = ", ".join(f"{lid}-{lname}({c})" for c, lid, lname in top)
            log.info(f"  DEBUG: {len(unseen_leagues)} unseen leagues. Top: {top_str}")

        # Two-pass approach:
        # Pass 1: scan ALL fixtures (any date) to collect team IDs from tracked
        #         leagues into persistent cache — this is how we learn which teams
        #         are "big" for friendly filtering.
        # Pass 2: filter by today's Bulgaria date for kickoff window.
        for f in all_fixtures:
            lid = f["league"]["id"]
            if lid in LEAGUE_IDS:
                known_league_team_ids.add(f["teams"]["home"]["id"])
                known_league_team_ids.add(f["teams"]["away"]["id"])

        # Pass 2: build kickoff list and friendly candidates for today (Bulgaria date)
        kickoff_hours = []
        friendly_candidates = []  # (fixture, hour, league_name)

        for f in all_fixtures:
            lid = f["league"]["id"]
            league_name = f["league"].get("name", "").lower()

            date_str = f["fixture"]["date"]
            try:
                kickoff_utc = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                if kickoff_local.strftime("%Y-%m-%d") != today_str:
                    continue
                hour = kickoff_local.hour + kickoff_local.minute / 60.0
            except Exception:
                continue

            if lid in LEAGUE_IDS:
                kickoff_hours.append(hour)
            elif any(kw in league_name for kw in FRIENDLY_NAME_KEYWORDS):
                friendly_candidates.append((f, hour, f["league"].get("name", "?")))

        # v10.11: Store today's tracked fixtures for daily summary
        global todays_tracked_fixtures
        todays_tracked_fixtures = []
        for f in all_fixtures:
            lid = f["league"]["id"]
            fid = f["fixture"]["id"]
            if lid in LEAGUE_IDS or fid in active_friendly_fixtures:
                date_str = f["fixture"]["date"]
                try:
                    kickoff_utc = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    )
                    kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                    if kickoff_local.strftime("%Y-%m-%d") == today_str:
                        todays_tracked_fixtures.append(f)
                except Exception:
                    pass

        # v9.7.2: Cache tomorrow's kickoff hours from this fetch
        # We already have tomorrow's fixtures in all_fixtures, just extract them.
        # Tomorrow at midnight, we'll use this cache to set the window for 0 credits.
        tomorrow_ko = []
        for f in all_fixtures:
            lid = f["league"]["id"]
            if lid not in LEAGUE_IDS:
                continue
            date_str = f["fixture"]["date"]
            try:
                kickoff_utc = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                if kickoff_local.strftime("%Y-%m-%d") != tomorrow_str:
                    continue
                hour = kickoff_local.hour + kickoff_local.minute / 60.0
                tomorrow_ko.append(hour)
            except Exception:
                continue
        if tomorrow_ko:
            cached_tomorrow_date = tomorrow_str
            cached_tomorrow_kickoffs = sorted(tomorrow_ko)
            log.info(
                f"  Cached tomorrow ({tomorrow_str}): {len(tomorrow_ko)} matches, "
                f"kickoffs {int(tomorrow_ko[0]):02d}:{int((tomorrow_ko[0] % 1) * 60):02d}-"
                f"{int(tomorrow_ko[-1]):02d}:{int((tomorrow_ko[-1] % 1) * 60):02d} Bulgaria"
            )
        else:
            cached_tomorrow_date = tomorrow_str
            cached_tomorrow_kickoffs = []
            log.info(f"  Tomorrow ({tomorrow_str}): no tracked league matches")

        schedule_date = today_str
        schedule_last_fetch = time.time()
        schedule_fixture_ids_loaded = True  # v9.7.2: real fetch got fixture IDs

        # Include international club friendlies where at least one team
        # is in our 20 tracked leagues, UNLESS it's a busy day.
        if len(kickoff_hours) < FRIENDLY_BUSY_THRESHOLD and friendly_candidates:
            added = 0
            skipped = 0
            for f, hour, name in friendly_candidates:
                if (
                    f["teams"]["home"]["id"] in known_league_team_ids
                    or f["teams"]["away"]["id"] in known_league_team_ids
                ):
                    active_friendly_fixtures.add(f["fixture"]["id"])
                    kickoff_hours.append(hour)
                    added += 1
                    log.info(
                        f"    Friendly OK: {f['teams']['home']['name']} vs "
                        f"{f['teams']['away']['name']} ({name})"
                    )
                else:
                    skipped += 1
            if added:
                log.info(
                    f"  Friendly filter: {added} friendly(ies) added, "
                    f"{skipped} lower-division skipped "
                    f"({len(kickoff_hours) - added} league + {added} friendly, "
                    f"team cache: {len(known_league_team_ids)} IDs)"
                )
        else:
            if active_friendly_fixtures:
                reason = "busy day" if len(kickoff_hours) >= FRIENDLY_BUSY_THRESHOLD else "no friendly candidates"
                log.info(f"  Clearing {len(active_friendly_fixtures)} friendly fixture IDs ({reason})")
            active_friendly_fixtures = set()
        
        # v9.6.1: Compute scheduled 20' entry times (UTC timestamps)
        # Group kickoffs by minute to find waves, then compute when each
        # wave enters the 20' monitoring window.
        from collections import Counter
        kickoff_wave_minutes = []
        for f in all_fixtures:
            lid = f["league"]["id"]
            fid = f["fixture"]["id"]
            is_tracked = lid in LEAGUE_IDS or fid in active_friendly_fixtures
            if not is_tracked:
                continue
            date_str = f["fixture"]["date"]
            try:
                kickoff_utc = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                kickoff_local = kickoff_utc.astimezone(BULGARIA_TZ)
                if kickoff_local.strftime("%Y-%m-%d") != today_str:
                    continue
                # Round to nearest minute to group wave kickoffs
                wave_minute = int(kickoff_utc.timestamp() // 60)
                kickoff_wave_minutes.append(wave_minute)
            except Exception:
                continue

        # Deduplicate waves and compute 1' entry times (kickoff + pre-window offset)
        scheduled_window_entries = sorted(set(
            (wave * 60) + ((MINUTE_MIN - PRE_WINDOW_MINUTES) * 60)  # kickoff_ts + 1' (pre-window)
            for wave in set(kickoff_wave_minutes)
        ))
        if scheduled_window_entries:
            entries_bg = [
                datetime.fromtimestamp(t, BULGARIA_TZ).strftime("%H:%M")
                for t in scheduled_window_entries
            ]
            log.info(
                f"  {MINUTE_MIN - PRE_WINDOW_MINUTES}' entries (schedule, signal from {MINUTE_MIN}'): {entries_bg}"
                f" ({len(scheduled_window_entries)} waves)"
            )
        
        if not kickoff_hours:
            schedule_no_matches = True
            todays_kickoff_latest_ts = 0.0  # v10.71: no matches today -> EOD gate open
            log.info(
                f"  No tracked league matches today ({today_str}) — "
                f"will re-check in {SCHEDULE_RECHECK_INTERVAL // 3600}h (2 credits used)"
            )
            return False
        
        schedule_no_matches = False
        earliest = min(kickoff_hours)
        latest = max(kickoff_hours)
        # v10.71: record the last kickoff time (UTC ts) for the EOD gate —
        # latest is in local (Bulgaria) float hours on today's date.
        try:
            _day_start_ts = today_bulgaria.replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
            todays_kickoff_latest_ts = _day_start_ts + latest * 3600.0
        except Exception:
            todays_kickoff_latest_ts = None  # unknown -> permissive gate
        
        # Buffer: 30 min before earliest kickoff (covers pre-window from 1'),
        # 100 min after latest kickoff (cover full 90' match + buffer)
        window_start = earliest - (30 / 60)
        window_end = latest + (100 / 60)
        
        # Convert to integer hours, floor start / ceil end
        new_start = max(0, int(window_start))
        new_end = min(24, int(window_end) + (1 if window_end % 1 > 0 else 0))
        
        if is_recheck and not schedule_no_matches:
            # Re-check WITH existing matches: only WIDEN (never shrink)
            if new_start < dynamic_active_start:
                log.info(f"  Window widened start: {dynamic_active_start:02d}:00 -> {new_start:02d}:00")
                dynamic_active_start = new_start
            if new_end > dynamic_active_end:
                log.info(f"  Window widened end: {dynamic_active_end:02d}:00 -> {new_end:02d}:00 (late-added fixture detected)")
                dynamic_active_end = new_end
        else:
            # First fetch of the day, OR first time finding matches today:
            # set window directly from actual kickoffs.
            old_start, old_end = dynamic_active_start, dynamic_active_end
            dynamic_active_start = new_start
            dynamic_active_end = new_end
            if is_recheck:
                log.info(
                    f"  Window corrected: {old_start:02d}:00-{old_end:02d}:00 -> "
                    f"{new_start:02d}:00-{new_end:02d}:00 (matches found after no-match period)"
                )
        
        match_count = len(kickoff_hours)
        total_matches_today = match_count
        log.info(
            f"  {match_count} matches: kickoffs "
            f"{int(earliest):02d}:{int((earliest % 1) * 60):02d} - "
            f"{int(latest):02d}:{int((latest % 1) * 60):02d} Bulgaria"
            f"{f' -> FIRST SIGNAL ONLY mode' if match_count > FIRST_SIGNAL_ONLY_THRESHOLD else f' -> FULL TRACKING mode' if match_count < FULL_TRACKING_THRESHOLD else ''}"
        )
        log.info(
            f"  Active window: "
            f"{dynamic_active_start:02d}:00 - {dynamic_active_end:02d}:00 Bulgaria"
            f"{'' if is_recheck else f' (saves {max(0, dynamic_active_start - ACTIVE_HOUR_START_FALLBACK) + max(0, ACTIVE_HOUR_END_FALLBACK - dynamic_active_end)}h of idle polling)'}"
        )
        return True
        
    except Exception as e:
        log.warning(
            f"Schedule fetch failed: {e}, "
            f"using fallback {ACTIVE_HOUR_START_FALLBACK}:00-{ACTIVE_HOUR_END_FALLBACK}:00"
        )
        dynamic_active_start = ACTIVE_HOUR_START_FALLBACK
        dynamic_active_end = ACTIVE_HOUR_END_FALLBACK
        schedule_date = today_str
        schedule_last_fetch = time.time()
        schedule_no_matches = False
        todays_kickoff_latest_ts = None  # v10.71: unknown -> permissive EOD gate
        schedule_fixture_ids_loaded = False  # v9.7.2: fallback didn't get real data
        scheduled_window_entries = []  # v10.19.1: clear stale entries on fetch failure
        return True


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global signal_outcomes, eod_report_sent_date, _last_stale_resolve, blocked_outcomes
    # v10.63: EOD race gate + Top-SOT deferred recovery (assigned below)
    global _eod_lookup_ok, _eod_defer_note_done, _top_sot_retry_queue, _top_sot_retry_credits_today
    # v10.71: EOD schedule gate + anti-hammer timers (assigned in the loop)
    global _last_eod_attempt, _last_eod_defer_note
    # v10.50: fast-lane + shadow globals — the daily reset block assigns these
    # names, so they MUST be declared global here (previously the fast-lane
    # credit reset was a dead local assignment and never actually reset).
    global fastlane_shadow, _event_fast_lane_fids, _event_fast_lane_last, _event_fast_lane_credits_today
    global _fl_shadow_count_today, _fl_sot_events, _fl_seen_sot_count, _fl_goal_minutes, _fl_seen_goal_count, _fl_shadow_dedupe
    # v10.75: box-burst shadow state — the daily reset block and the startup
    # loader assign these names, so they MUST be declared global here
    # (same bug class as the v10.50 UnboundLocalError crash).
    global boxburst_shadow, _boxburst_fired, _boxburst_ib_hist, _boxburst_count_today, _boxburst_count_date
    global goalburst_shadow, _goalburst_fired, _goalburst_count_today, _goalburst_count_date  # v10.76
    # v10.62: redeploy guard — one-shot tracking check flag (main loop)
    global _redeploy_check_done
    # v10.53: goal-watch state — the daily reset block below assigns these
    # names, so they MUST be declared global here (same bug class as the
    # v10.50 UnboundLocalError crash).
    global _goal_watch_fids, _goal_watch_last, _goal_watch_credits_today, _goal_watch_credits_date
    global _goal_watch_seen, _goal_watch_flash_keys, _gw_flashes_today, _gw_flashes_date, _coldstart_warmed
    # v10.54: surge-watch state — the daily reset block below assigns these
    # names, so they MUST be declared global here (same bug class as the
    # v10.50 UnboundLocalError crash).
    global _surge_seen_sot, _surge_seen_shots, _surge_wake_minute, _surge_last_alert_ts
    global _surge_flood_minute, _surge_fixture_alerts, _surge_alerts_today
    global _surge_burst_last_minute, _surge_burst_count
    # v10.56: goal-SOT ledger + genuine burst flag (daily reset clears them)
    global _pending_goal_sot, _goal_sot_landed, genuine_burst_fixtures
    # v10.57: Top-SOT player cache + goal counters (daily reset clears them)
    global _player_sot_cache, _player_sot_cache_goals, _fixture_valid_goals, _player_sot_cache_built_sot
    global _latest_sot_event_minute
    # v10.73: players-stats fallback state (census + daily budget)
    global _players_feed_census, _players_sot_credits_today
    # v10.35: Load persisted EOD report date — prevents re-send on restart
    eod_report_sent_date = _load_eod_report_sent_date()
    log.info("=" * 60)
    # BOT_VERSION is now module-level (moved in v10.44d-patch)
    log.info(f"Football Bot {BOT_VERSION} — Goal predictions (Poisson, scoreline-aware) + dead probe revival + signal cooldown + 15s polling + PRE/POST-GOAL tagging + xG escape hatch + SOT-since-goal tracking + pressure buildup override + SOT guard soft correct + opponent stats in poll data + untracked live debug + daily ML backup to Telegram (gzip) + /restore file upload + EOD retry resolution (90s wait) + disallowed goal filter + top SOT player in signals + /polls gzip + detection latency measurement + event fast lane (10s) + conditional both-signal slowdown + enriched signal data (poll_interval, last_goal_minute, time_since_prev_signal) + goal-triggered priority polling (discovery + stats) + goal detection latency (detect_lag, stats_lag) + signal_lag tracking + untracked-retry fast discovery (120s) + Bulgarian league 172 fix + live market data + source discovery + ML shadow scoring (logging-only) + top SOT non-scorer priority + signal min-gap guard (180s) + late-window SOT-rise requirement + late EW GPS floor + EW acceleration gate + GPS85 fast-lane lock + v10.49: signal-send decoupled from player-SOT fetch (every signal 0.5-2s faster) + blocked-signal false-negative tracking (logging-only) + Poisson per-league calibration tracking (logging-only) + v10.50: event fast lane widened to top-3 fixtures + FAST-LANE SHADOW MODE (virtual signals, logging-only, never sent) + SOT>=2 stats polling 45s->30s (go-max) + fast-lane daily credit reset bugfix + v10.51: fast-lane ghost-pick guard (stale GPS history from fixtures past the 85' ceiling no longer re-enters the pick list) + monitoring-drop reasons logged in discovery (why a game left 'Monitored:') + v10.52: fast-lane poll crash fix (_event_fast_lane_fids missing from global decl — UnboundLocalError killed the bot when any fixture became monitored) + full-file global-scoping audit clean + poll_event_fast_lane now covered by smoke tests + v10.53: GOAL WATCH instant goal flashes (~10-30s latency, close games 60'+, /goalwatch toggle, goal_flash.jsonl log) + cold-start warm-up (events backfill so freshness gates work immediately after mid-game restarts) + v10.54: SURGE WATCH pre-goal pressure alarms (quiet-team SOT wake-up + burst escalation + shot-flood layer; rides the same event polls at ZERO extra credits; /surgewatch toggle, default ON; surge_watch.jsonl log) + goal flashes default OFF (user preference: warn BEFORE the goal, not after) + v10.55: goal-aware surge semantics (a goal counts as the team's LAST KNOWN SHOT for silence measurement and closes open episodes, but NEVER triggers an alert; post-goal pressure needs a fresh quiet spell first = second-goal early watch) + SUSTAIN tier (3rd+ SOT in a burst keeps alerting up to the 5-per-game cap — continuous pressure fully covered, not just the first two shots) + v10.56: GOAL-SHOT EXCLUSION in the main signal pipeline (the v10.55 surge-watch goal semantics applied to repeat signals: pending/landed goal-SOT ledger so the shot that scored never counts as SOT-jump / buildup / burst / post-goal fresh-pressure evidence — sig_num>=2 SOT-jump gate, goal-pressure-continues, cooldown buildup override, first-signal-only exception, 80'+ event-burst exception and the SOT-at-goal baseline are all goal-shot-free; a goal's own +1 can only ever CLOSE pressure windows, never open them) + v10.57: TOP-SOT PLAYER GOAL REFRESH (a goal instantly drops the per-fixture Top-SOT player cache — the player who scored leaves the 'scores next' line and the next signal headlines the top SOT player who has NOT scored; cache rebuilds from fresh events after every goal, event lanes track valid-goal counts, the stats lane drops the cache on score change, and finished-fixture/daily cleanup now also clears the player cache) + v10.58: TOP-SOT NEVER-A-SCORER + BIG-CHANCE VOICE (the Top SOT line can never headline a player who already scored: when every SOT taker has scored it falls back to the top shooters who have not — shown as '(n shots)' — and when every shooter has scored no line is sent at all; an events-feed lag retry and a SOT-growth cache refresh keep the names fresh as new shooters appear; Big Chances GPS weight nearly doubled (2.5 pts/BC, cap 6) because they are the strongest single pre-goal stat the API offers, and signals now carry a NEW BIG CHANCE freshness warning when a big chance was created since the last poll) + v10.59: GPS-vs-ML SCOREBOARD (the ML shadow opinion is now computed once per poll BEFORE the gates and saved into every signal outcome, every blocked candidate and every pressure-poll record; new /mlstats command shows who reads incoming goals better — sent-signal winner/loser gaps, goals-the-gates-blocked capture rate, agreement — the model itself stays frozen, logging-only, zero extra credits) + v10.60: FIELD EXPANSION + AVAILABILITY CENSUS (GK saves, fouls, offsides, yellow cards, pass volume and accuracy, blocked shots, substitutions and card events are now parsed from responses the bot ALREADY fetches and recorded into every poll and signal as null-safe fields for brain v2 — never used in GPS or gates, zero extra credits, zero behavior change; a live-learned per-league census (/fields command, field_census.json) now tells us which fields the API actually delivers, after the big_chances post-mortem proved fields must be verified from real responses, never assumed) + v10.61: BC-MISSING REDISTRIBUTION SHADOW (big_chances is never delivered on this API plan, so the GPS always runs without its 6-pt BC component and without compensation when xG is present; every poll and signal now also logs gps_restored, the BC weight proportionally redistributed exactly like the xG-missing pattern, while the live gates stay on the historical scale — thresholds were tuned on it with real outcome data, all 74 CRITICALs ever were SOT>=3 safety-net fires, and the marginal sub-threshold bands convert no better; GPS_BC_REDISTRIBUTE=False until a threshold re-tune says otherwise) + v10.62: PHANTOM-GOAL PROTECTION + REDEPLOY GUARD (disallowed/missed-penalty 'Goal' events no longer hide their taker from the Top SOT 'scores next' line and no longer inflate events-side SOT — Viborg 42' phantom-goal post-mortem; every silently-skipped Top SOT line now logs WHY; and after every redeploy the bot PROVES nothing was lost: all volume files line-verified at startup with corrupt-tail auto-repair, then ONE Telegram message after first discovery reports data counts and every live game classified monitored / pickup pending / past 85' / untracked) + v10.63: STARTUP EOD RACE FIX + TOP-SOT FEED-LAG RECOVERY (a fresh startup is never again mistaken for end-of-day — EOD fires only after the session has actually LOOKED at live state: first discovery or a no-matches schedule; the 90s stall / duplicate ML backup / mid-game outcome-clear / orphaned pending outcomes race from the Sep 4 redeploy is dead; the 30s startup retry skips when every pending fixture is still live; the resolver timer is seeded so the startup /fixtures burst is not duplicated; and the redeploy message counts pending from the file, not possibly-cleared memory — PLUS the Top SOT 'scores next' line now survives events-feed lag: a deferred recovery retry (45s, max 2 attempts, credit-capped) re-fetches after the feed catches up and sends the line late with a feed-lag note, the cache path refreshes instead of serving stale silence when stats know more SOT than the cached feed, and the growth snapshot now covers BOTH teams of a fixture — Sparta/PEC post-mortem: stats knew SOT=3, the feed listed only the scorer, the never-show-a-scorer rule silenced the line and the 3s retry could not bridge a minutes-long lag) + v10.64: STARTUP CRASH HOTFIX (v10.63's new startup branch wrapped an INT pending count in len() — TypeError crash-loop at every restart where all pending outcomes sat on live fixtures, e.g. the Sep 4 19:19 UTC evening-slate restart with 7 pendings; fixed, and the main() startup wiring is now smoke-tested by DIRECT EXECUTION of the exact block, not just the functions it calls) + v10.65: TOP-SOT IN THE SIGNAL + SHOT-EVENT FEED CENSUS (the 'scores next' player line is now EMBEDDED in the signal itself — fetched before the send, one round-trip, no inline 3s retry — and an unavailable line SAYS so: no player data yet / every listed shooter already scored / player data unavailable for this league, learned from a live per-league census (sot_feed_census.json, /sotfeed) of which feeds ever deliver per-player Shot events; recovery retries widened to 3×60s and SKIPPED for leagues the census has learned never deliver, so no credits burn on hopeless follow-ups — Porto/Betis post-mortem: their feeds listed 2/6 and 0/6 SOT at signal time and both recoveries gave up + v10.66: EVENT-MINUTE CORRECTION (outcome records resolved live carry the DETECTION minute, not the true event minute — feed lag + poll cadence bias them late: Botev Vratsa's 86'/88' goals were booked as 90' (+49') and a true in-window goal detected past its window boundary records a false MISS; the FT resolution pass now re-verifies every live-stamped field against the true goal-event minutes it already fetched — goal minutes corrected, 5/10/15m windows recomputed with MISS->HIT flips where the event truth says HIT, phantom live goals flipped back to MISS via the final-score check, events-missing goals HELD with the live minute — zero extra credits, zero gate changes) + v10.67: LATE SURGE TO THE FINAL WHISTLE (close games crossing the 85' ceiling — Botev Vratsa's 86'/88' vs Septemvri Sofia — are retained in the events watch lane until FT (LATE-RETAIN), so every surge tier stays live 86-90'; late fixtures take watch-lane pick priority + a +2 alert-budget bonus; the stats lane, signal gates and signal_outcomes are untouched — warning-only, ~10-16 events polls per retained game inside the existing 1500/day lane cap + v10.68: ODDS CAPTURE HARDENING (signal-time market data made P&L-grade: one 3s retry when the odds fetch fails — 15/26 Sep-4 signals recorded NO odds — plus a suspect-price re-fetch and flag for impossible live prices (over-line implied < 12% before 80') and a live vs pre-match-fallback source tag — the stale pre-match totals prices were the Sep-4 garbage class (over 4.5 @ 23.00 etc.) that inflated the paper P&L by ~1,400 EUR; ~20 extra credits/day, retries quota-guarded, zero signal-logic changes + v10.69: PROJECTION SANITY + ADAPTIVE LINES + GOAL-SHOT-FREE FIRST SIGNALS (remaining-goals lambda = observed rate x REMAINING minutes, late blend to league average after 70' + GPS hotness lift and caps — the Elversberg 70' GPS-100 'Exp. total 9.0' projected 4 future goals where 2 landed; the projection block now shows only UNDECIDED over lines — at 2-2 the O2.5/O3.5 '100%' rows were pure noise — and BTTS hides once decided; and the CSKA Sofia post-mortem — the goal's own shot can no longer be the 3rd SOT that triggers a first CRITICAL: the SOT>=3 safety net and the 5-20' post-goal freshness check now run on ledger-genuine goal-shot-free counts, so a signal arriving minutes after the goal it announced is gone) + v10.70: TOP-SOT NON-SCORER ONLY (the 'scores next' line ONLY ever names players who have NOT scored yet — never a scorer, no '(scored)' tags, exactly the user's spec; when every listed shooter has already scored the line says exactly that; the deferred recovery stops quietly once the feed is current and all listed shooters scored — no credits on a line that can never exist — while a still-lagging feed keeps retrying because the unlisted SOT may belong to a non-scorer + v10.71: FEED-GUARD (silent API feed-death detection — the Sep 5 18:34 UTC incident: /fixtures?live=all returned an empty live set while 7 tracked matches were at 18'-78' and the quota counter froze at 7499 for 60+ min, all HTTP 200; now an all-monitored-vanish below 85' triggers a batched fixture-ID verification, verified-still-live games are HELD in monitoring with a Telegram alarm every 15m and auto-resume when the feed recovers (90m hold cap); a frozen-quota watchdog warns when the daily counter stops decrementing across 45+ calls; AND the pending-outcome orphan fix — signal_outcomes is never cleared while outcomes are pending (the Sep 5 18:42 gap-clear orphaned 9 pendings to disk), the EOD report/backup/clear now fire only at the true end of day (schedule-gated, anti-hammered), and mid-day gaps hold outcomes in memory for the 10-min periodic resolver + v10.72: SHADOW GATES + SELF-HEALING SLEEP (two would-suppress classes from the Sep 4-6 outcome data — DAMP: winning by 2+ with GPS<85, 23% full WR vs 31% baseline, and LATE75: signals at 75'+, 18% full WR — are now TAGGED in the log and every outcome record but sent UNCHANGED; the hard-gate flip is the one-line DAMPENER_HARD_GATE / LATE_HARD_GATE constant, decision after ~1 week of shadow data, ~300 tagged signals; heartbeat shows 'Shadow: DAMP:n+LATE75:n'; COLD-START LEDGER SEEDING: pre-restart goals seed the landed goal-shot ledger on a team's first poll, so the goal's own shot can never trigger a false first signal after a mid-match restart — Benfica 16' class, zero credits; STOP-MODE RENEWAL PROBE: one direct /status call per 30-min STOP wake re-reads the live quota header, so quota exhaustion self-heals at the 03:00 Sofia renewal instead of looping forever — api_get's pre-flight raise used to hide the renewal completely; MIDNIGHT LIVE-GAME GUARD: a live monitored match at the active-window rollover keeps being watched to FT (2h cap, freshness-gated so a stale cache can never fake it) — the GIL Vicente 71'-abandoned-at-00:00 class — and verified-FT guard releases + a heartbeat purge kill the frozen 'FastWin 0s' zombie + v10.73: RED-CARD VOICE + PLAYERS-STATS TOP-SOT FALLBACK (red cards now speak in every signal: the events feed the bot already polls yields player + minute + kind — Dunav-Slavia class: two reds, 10’ and 76’, previously invisible — with a NEW RED CARD warning inside 10 game minutes, man-up / man-down context, and red_cards_team / red_cards_opp / red_card_events recorded into every signal outcome and every poll for brain v2, zero extra credits, zero gate changes; AND when the events feed has no per-player shots at signal time — Serie A 0/16, Bundesliga 0/14, Eredivisie 0/9, Premier League 1/10, Ligue 1 1/13 in the Sep 1-5 audit — one /fixtures/players call delivers the Top SOT names from the stats pipeline: 1 credit, daily-capped (60), per-league censused (players_feed_census.json, visible in /sotfeed), same never-a-scorer ladder, merged into the player cache so repeat signals are free + v10.74: BOOT-PATH MIDNIGHT GUARD + CALIBRATED OVER PROJECTION + FT 1X2 PREDICTION (a restart at ~00:00 local no longer sleeps through live pending-outcome fixtures — the resolver's 'pendings on LIVE fixtures' knowledge feeds the v10.72 hold with a 25m freshness anchor and the same 2h cap, so the Sep 8 00:01-boot class is dead; the goal projection's GPS hotness lift is shrunk 0.8->0.2 after the Sep 1-7 backtest proved it pure bias — 74.5% predicted vs 54.1% landed on undecided O2.5 (n=170) and a naive no-pressure Poisson scored the better Brier 0.207 vs 0.301 — and the DISPLAYED over lines are now calibrated 50/50 against an empirical game-state table P(>=k more goals | current total x minute band) measured from the 229 resolved signals, with pred_over_*_cal + cal_mode recorded beside the raw model values for the next-week comparison; red cards now move the remaining-goals lambdas inside the same Poisson engine (0.72x down / 1.08x up per net red, events-based None-safe) so overs and FT odds both see 10v11; every signal carries a full-time 1X2 PREDICTION line — win/draw/loss for the signaled team from the live scoreline + adjusted lambdas, informational only, zero extra credits — plus a pitch-state line (men on pitch 11-Reds, subs used, injury-labeled subs from Subst detail='Injury') and pred_ft_sig/draw/opp outcome fields, with pred_ft_actual filled from the true FT result at resolution for later calibration; zero signal-gate changes, zero extra credits + v10.75: BOX-BURST SHADOW (the low-SOT box-volume class — SOT 1-2 while shots-inside-box>=8 within 21-61' — now records a VIRTUAL signal at the first crossing, never sent, never gating, zero extra credits; the Sep 2-8 backtest says 64.3% scored later / 35.7% within 15m plain (n=14) and 69.2%/38.5% with ib rising (n=13) vs a 48% base — the go-live rule after ~2 weeks of boxburst_shadow.jsonl records is >=60% later AND >=40% <=15m, then the one-line BOXBURST_LIVE flip adds a compact Telegram alert; records carry sot/ib/total_shots/ib_rising/gps/red-cards/real_signal_before and resolve EXACTLY like real signals through check_fastlane_shadow, which now walks BOTH shadow stores; own 2-poll ib history kept by the evaluator because the poll recency fields ib_5m_ago/ib_10m_ago are never populated; dedupe survives restarts (today's fired keys reloaded at boot), runtime state resets daily and purges with finished fixtures; heartbeat shows Shadow: ...+BOX:n; the redeploy guard verifies the new file and reports its count + v10.76: GOAL-BURST — THE TOTALS PATH (the Lille 2-3 Betis post-mortem: 5 goals by 53', ZERO signals — Betis' 3 SOT WERE the 3 goals so GOAL-SHOT NET left effective pressure 0, and GPS 52/65 never lit; the system only certifies sustained NON-goal pressure, so goal-burst games are structurally invisible; the Sep 2-8 backtest on 63,489 polls adds that 69% of FIRST goals land while the scorer's SOT is still <=2 — the SOT>=3 net is late by design; three banked-goals cells now record virtual MATCH-LEVEL signals at first crossing — G1 first-goal-by-25' (62% reach 3+, n=32), G2 two-goals-by-40' (84% next goal, 48% <=15m, n=44; GPS-blind subset 79%, no-signal-before 87% = the exact blind spot), G3 three-goals-by-55' (76%, n=37) — resolved with ANY-goal semantics (own goals count, Over-lines settlement) through check_fastlane_shadow which now walks THREE shadow stores; zero gate changes, zero extra credits; G2 already clears the box-burst go-live bar so its compact alert is ON by default (GOALBURST_LIVE) while G3/G1 stay shadow behind their own flags; ft_total + bet_hit stamped at resolution so the totals bets grade honestly; heartbeat shows Shadow: ...+GB:n; the redeploy guard verifies goalburst_shadow.jsonl and reports its count + v10.77: P&L-GRADE ODDS FILTER (the honest ledger: every signal outcome now carries odds_pnl_grade, True ONLY when odds_source='live' AND not odds_suspect — the Sep-6 post-mortem found 100% of recorded prices were prematch_fallback stale pre-match totals (Marseille O4.5 @ 26.0 with 4 goals already in at 54', AC Horsens O3.5 @ 11.0 with 3 in at 45', 22/58 prices >= 3.00) and the paper P&L of +1,900 EUR / ROI +164% on Sep 6 was the same garbage class as Sep 4's +1,400 EUR — at realistic live prices the day was break-even; the EOD report's MARKET / EV / Flat-ROI block now grades ONLY P&L-grade prices and SAYS how many stale prices were excluded, with backward-compatible derivation for old records (no field -> source/suspect fallback), while all prices remain in the file for research; zero gate changes, zero extra credits + v10.78: ODDS IN THE SIGNAL (the user's betting-decision block: every Telegram signal and every live G2 goal-burst alert now carries the MARKET PRICE captured at signal time — source-labeled LIVE vs 'PRE ref' with suspect prices flagged, the v10.77 ledger honesty moved into the chat — beside the CALIBRATED fair prices and break-evens for the next-goal over line and the team-to-score bet, so the 'do I bet?' call is: open your book, compare the live price against the printed break-even, done; one fast single-pass odds fetch (for_message mode: no retry sleep, no suspect re-fetch, quota-guarded) runs AFTER all gates pass and BEFORE the send (~1s later, same normal-path credit count, odds never influence the signal), the same data feeds the outcome record, a failed fast pass falls back to the full v10.68 hardened capture post-send; team-to-score fair = Poisson marginal blended 50/50 with the minute-banded landed rate from the user's own Sep 6-8 ledger (61%/65%/46%/35% bands), capped at the any-goal line; G2 alerts price the exact one-more-goal market and print the class break-even (84% -> 1.19); pred_team_scores/_cal + odds_ev_pct + odds_in_msg recorded for the next calibration pass; zero gate changes + v10.85: SIMPLIFIED SIGNAL (display-only, message cut ~45%: one-line stats per side with dead xG/BigChances/corners display channels dropped, one-line calibrated projection + FT prediction, plain-language bet block where the fair line says bet-only-at-LIVE-odds>=X, cards & corners lines say need-N-more and bet-Over-only-odds>=X, the Red Cards None line is gone, the orphan GPS +0 fragment fixed; the LEDGER, every recorded field, every gate and threshold UNCHANGED + v10.86: EMPIRICAL LAMBDA CALIBRATION — minute-banded deflate of the final remaining-goals lambdas (signal team 0.85/0.80/0.45, opponent 0.70/0.60/0.30 for <=45/<=60/61+; measured on the settled ledger Sep 2-9: opponent lambda 1.5x hot in both regimes, late-game collapse) — prediction-only, never a gate; fair prices + projections honest, bet-only-at-LIVE-odds>=X threshold stricter; 61+ signals stay full alerts (user decision) + v10.87: GOAL-RACE GUARD (feed-ahead-of-score mute — the PSV-Shakhtar 45' class: signal composed on a 0-0 stats batch while the events feed already knew the goal; the pre-send Top-SOT fetch's valid-goal count vs the message scoreline, feed-ahead -> mute + blocked-candidate record GOAL_RACE_FEED_AHEAD, the post-goal gates re-decide on the next poll; the POST-GOAL tag now says it watches the NEXT goal — the Fenerbahce 53' class was a genuine second-goal watch, only the wording hid it; v10.88: POST-GOAL HONESTY (the Man Utd 33' class — a stale 5-20m post-goal CRITICAL whose GPS>=75 trigger was completed by the goal's own shot — is now BLOCKED like every other tier, with POST_GOAL_STALE blocked records; the Como 28' class — a genuine attempt-burst next-goal watch that passed the 5-20m freshness gate in silence — now carries the same this-watches-the-NEXT-goal annotation; and the header ordinal becomes (next) once the signaling team has scored, so no post-goal signal can read as a first-goal warning + v10.89: RED-AWARE LOSING RELAXATION (a losing team with the opponent down a NET man and deficit <= 1 — the Slavia 1-0 Lens class, Lens blocked at 64' while chasing 10 men — now passes at STANDARD tier bars with a man-advantage message tag and a red_relax ledger field so the class grades itself; 11v11 losing, deficit >= 2, and missing-events cases keep the strict v10.34 gate — fail-closed)")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (round-robin for rate-limit resilience, NOT quota expansion)")
    log.info(f"Signal window: {MINUTE_MIN}-{MINUTE_MAX}' | Polling: {MINUTE_MIN - PRE_WINDOW_MINUTES}-{EXTENDED_MAX}'")
    log.info(f"Thresholds: BUILDING>={GPS_BUILDING} EARLY_WARNING>={GPS_EARLY_WARNING} CRITICAL>={GPS_CRITICAL}")
    log.info(f"v10.61: GPS_BC_REDISTRIBUTE={GPS_BC_REDISTRIBUTE} (gps_restored shadow-logged in every poll/signal/blocked record)")
    log.info("v10.62: REDEPLOY GUARD active — data files verified at startup; live-game coverage check + Telegram summary after first discovery")
    log.info("v10.63: STARTUP EOD RACE FIXED (EOD gated on first live-state lookup) + TOP-SOT feed-lag recovery (45s deferred retry, max 2, credit-capped)")
    log.info("v10.64: startup crash hotfix — v10.63 crash-looped (len() of an int) when every pending outcome sat on a live fixture at restart; fixed, all v10.63 behavior now reachable")
    log.info("v10.65: TOP-SOT line embedded in signals + never-silent notes + per-league shot-event feed census (/sotfeed); recovery 3x60s, skipped for NO DATA leagues")
    log.info("v10.66: event-minute correction — live-resolved outcomes re-verified against true FT goal-event minutes (OUTCOME CORRECTED / OUTCOME HELD lines; zero extra credits)")
    log.info("v10.67: LATE SURGE — close games retained in the events watch lane to the final whistle (LATE-RETAIN lines; surge alarms live 86-90'; zero gate changes)")
    log.info(
        "v10.74: BOOT-PATH MIDNIGHT GUARD (pendings on resolver-verified LIVE fixtures hold the "
        "first sleep to FT, 25m freshness, 2h cap) + CALIBRATED OVER LINES (GPS lift 0.8->0.2; "
        "50/50 blend with the empirical game-state table) + RED-CARD LAMBDA (0.72x/1.08x per net "
        "red) + FT 1X2 PREDICTION + pitch state (men/subs/injury subs) in every signal — "
        "informational only, zero gate changes, zero extra credits"
    )
    log.info(
        "v10.77: P&L-GRADE ODDS FILTER (every outcome record now carries odds_pnl_grade: "
        "True ONLY for genuine live prices; prematch_fallback stale prices — the Sep-4/Sep-6 "
        "class that inflated paper P&L by ~1,900 EUR — are excluded from the EOD MARKET/EV/ROI "
        "analysis but stay recorded for research; zero extra credits, zero gate changes)"
    )
    log.info(
        "v10.78: ODDS IN THE SIGNAL (market price + calibrated fair price + break-even inside "
        "every signal and G2 goal-burst alert — LIVE vs PRE-ref labeled, suspect flagged, EV "
        "verdict only vs live prices; fast pre-send capture ~1s, same normal-path credits, "
        "zero gate changes)"
    )
    log.info(
        "v10.81: BULGARIAN SUPER CUP (league 656) tracked — the Sep 9 Levski-CSKA derby was "
        "UNTRACKED (Super Cup is not First League 172/357): 4-5 SOT + a goal, zero polls, "
        "zero signals. Coverage hole closed with one LEAGUE_IDS line; zero gate changes"
    )
    log.info(
        "v10.84: ATTEMPT-BURST (shot-volume trigger, net of goal shots) + "
        "RESPONSE WINDOW (10m post-concede, shadow for trailing-by-1) + "
        "SAME-POLL goal mute + MINUTE-ARCHIVE repair of the 5m/10m window "
        "deltas — the Sep 9-10 post-mortem: SOT freezes when shots go wide "
        "or saved, attempts don't (3.3x lift at d10>=4, 38% episode hit)"
    )
    # v10.83: unconditional version line — the deploy check. Every boot
    # answers "which version is running?" in the first seconds of log,
    # no need to wait for the first signal's [v10.xx] tag.
    log.info(f"BOOT {BOT_VERSION} \u2014 football goals pressure bot up")
    log.info(f"GPS: SOT+IB+ShotVol+xG+BC+Corners+Accel | Adaptive xG for domestic/European")
    log.info(f"Active: DYNAMIC from schedule (fallback {ACTIVE_HOUR_START_FALLBACK}:00-{ACTIVE_HOUR_END_FALLBACK}:00)")
    log.info(f"Team cache: {len(known_league_team_ids)} IDs")
    log.info("=" * 60)

    # v10.62: REDEPLOY GUARD [data] — verify every volume file parses cleanly
    # BEFORE anything is loaded; a corrupt mid-append tail is rotated out
    # and repaired here so the loads below see a clean file.
    try:
        redeploy_data_check()
    except Exception as _e:
        log.warning(f"v10.62: redeploy data check failed: {_e}")

    # v10.19.3: Load ALL outcomes (resolved + pending) from JSONL
    global signal_outcomes
    all_loaded = load_all_outcomes()
    if all_loaded:
        signal_outcomes = all_loaded
        resolved_count = sum(1 for e in all_loaded if e.get("resolved"))
        pending_count = sum(1 for e in all_loaded if not e.get("resolved"))
        log.info(f"Loaded {len(all_loaded)} outcome(s) from {OUTCOMES_FILE} ({resolved_count} resolved, {pending_count} pending)")

    # v10.49: Load blocked-candidate records (false-negative tracking) + Poisson calibration
    _blocked_loaded = _load_blocked_outcomes()
    if _blocked_loaded:
        blocked_outcomes = _blocked_loaded
        _bl_resolved = sum(1 for e in _blocked_loaded if e.get("resolved"))
        log.info(
            f"v10.49: Loaded {len(_blocked_loaded)} blocked-candidate record(s) "
            f"from {BLOCKED_FILE} ({_bl_resolved} resolved)"
        )
    _load_poisson_calibration()

    # v10.60: Load field-availability census (which KPIs the API delivers)
    load_field_census()

    # v10.65: Load the shot-event feed census (which leagues deliver
    # per-player Shot events for the Top SOT line)
    load_sot_feed_census()
    load_players_feed_census()   # v10.73: players-stats fallback census

    # v10.50: Load fast-lane shadow records (virtual signals, logging-only)
    _fl_loaded = _load_fastlane_shadow()
    if _fl_loaded:
        fastlane_shadow = _fl_loaded
        _fl_resolved = sum(1 for e in _fl_loaded if e.get("resolved"))
        log.info(
            f"v10.50: Loaded {len(_fl_loaded)} fast-lane shadow record(s) "
            f"from {FASTLANE_SHADOW_FILE} ({_fl_resolved} resolved)"
        )

    # v10.75: Load box-burst shadow records + rebuild the dedupe set so a
    # mid-match restart can never double-shadow a team-side that already
    # crossed the cell earlier today (same discipline as _fl_shadow_dedupe).
    _bb_loaded = _load_boxburst_shadow()
    if _bb_loaded:
        boxburst_shadow = _bb_loaded
        _today_key = time.strftime("%Y-%m-%d")
        for _bb_e in _bb_loaded:
            if str(_bb_e.get("shadow_clock", ""))[:10] == _today_key:
                _boxburst_fired[(_bb_e.get("fixture_id"), _bb_e.get("team_id"))] = (
                    _bb_e.get("shadow_time") or 0.0
                )
        _bb_resolved = sum(1 for e in _bb_loaded if e.get("resolved"))
        log.info(
            f"v10.75: Loaded {len(_bb_loaded)} box-burst shadow record(s) "
            f"from {BOXBURST_SHADOW_FILE} ({_bb_resolved} resolved, "
            f"{len(_boxburst_fired)} today-dedupe key(s))"
        )

    # v10.76: Load goal-burst shadow records + rebuild the dedupe set so a
    # mid-match restart can never double-shadow a crossing that already
    # fired earlier today (same discipline as box-burst / fast-lane).
    _gb_loaded = _load_goalburst_shadow()
    if _gb_loaded:
        goalburst_shadow = _gb_loaded
        _today_key = time.strftime("%Y-%m-%d")
        for _gb_e in _gb_loaded:
            if str(_gb_e.get("shadow_clock", ""))[:10] == _today_key:
                _goalburst_fired[(_gb_e.get("fixture_id"), _gb_e.get("gb_class"))] = (
                    _gb_e.get("shadow_time") or 0.0
                )
        _gb_resolved = sum(1 for e in _gb_loaded if e.get("resolved"))
        log.info(
            f"v10.76: Loaded {len(_gb_loaded)} goal-burst shadow record(s) "
            f"from {GOALBURST_SHADOW_FILE} ({_gb_resolved} resolved, "
            f"{len(_goalburst_fired)} today-dedupe key(s))"
        )

    # v10.44m: Rebuild signaled_teams from file so cooldown works after redeploy
    rebuild_signaled_teams_from_file()

    with httpx.Client(timeout=30.0) as client:
        # v10.11: Resolve any stale outcomes from previous sessions
        # v10.44n: Uses .get("resolved") for safety; retries once if still pending
        pending_count = sum(1 for e in signal_outcomes if not e.get("resolved"))
        if pending_count:
            log.info(f"v10.11: Resolving {pending_count} stale outcome(s) from finished fixtures...")
            resolve_stale_outcomes(client)
            still_pending = sum(1 for e in signal_outcomes if not e.get("resolved"))
            # v10.44n: If still pending after first attempt, wait 30s and retry
            # (API may need a moment to mark late-finishing matches as FT)
            if still_pending:
                # v10.63: when EVERY pending outcome sits on a still-LIVE
                # fixture the 30s retry cannot achieve anything (games do
                # not finish in 30 seconds) — skip the stall and let the
                # periodic resolver pick them up as they end.
                _pend_fids = {e.get("fixture_id") for e in signal_outcomes
                              if not e.get("resolved")}
                if _pend_fids and _pend_fids.issubset(_resolve_live_fids):
                    # v10.64 hotfix: still_pending is an INT count here (sum),
                    # not the list of the same name inside
                    # resolve_stale_outcomes — v10.63 wrapped it in len()
                    # and crash-looped the bot at startup whenever every
                    # pending outcome sat on a live fixture (Sep 4 19:19 UTC,
                    # 7 pendings, TypeError: object of type 'int' has no len).
                    log.info(
                        f"v10.63: {still_pending} pending outcome(s) all on LIVE "
                        f"fixtures — skipping the 30s startup retry (periodic "
                        f"resolver will finish them)"
                    )
                else:
                    log.info(f"v10.44n: {still_pending} still pending, waiting 30s and retrying...")
                    time.sleep(30)
                    resolve_stale_outcomes(client)
            # Log summary after resolution
            resolved_count = sum(1 for e in signal_outcomes if e.get("resolved"))
            still_pending = sum(1 for e in signal_outcomes if not e.get("resolved"))
            log.info(f"v10.11: After resolution: {resolved_count} resolved, {still_pending} pending")
            if resolved_count > 0:
                log_outcome_summary()

        # v10.63: seed the periodic resolver timer so it does NOT re-fire on
        # the first loop iteration — the startup resolve above already ran,
        # and the old `or _last_stale_resolve == 0` clause duplicated the
        # /fixtures call within the same second (3 redundant credits on the
        # Sep 4 redeploy: reqs #2/#4/#5 all in one second).
        _last_stale_resolve = time.time()

        # v10.18: Morning recap removed from auto-send. Use /recap or /stats in Telegram.
        log.info("v10.18: Morning recap skipped (command-only now). Use /recap for yesterday's stats.")

                # v10.44s: Discover available market-data sources (1 credit, once per startup)
        if ODDS_CAPTURE_ENABLED:
            try:
                _bm_data = api_get(client, "/odds/bookmakers")
                _bm_list = _bm_data.get("response", [])
                if _bm_list:
                    # v10.44s-fix: some source entries have name=null (key present,
                    # value None), not just a missing key — .get("name","?") only
                    # covers the missing-key case, so a null value passed straight
                    # through as None and crashed the next line's .lower() call.
                    # v10.48: source names are NOT logged (chat-safe logs) — count
                    # and preferred-source availability only. Names still stored
                    # in outcome data files for offline analysis.
                    _bm_names = [b.get("name") or "?" for b in _bm_list]
                    log.info(f"MKT: {len(_bm_names)} data source(s) available")
                    _bg_books = [n for n in _bm_names if n and any(kw in n.lower() for kw in ["efbet", "winbet", "palms", "betbulldog", "sesame", "bwin", "eurobet"])]
                    if _bg_books:
                        log.info(f"MKT: {len(_bg_books)} preferred-region source(s) found")
                    if PREFERRED_BOOKMAKER not in _bm_names:
                        log.warning("MKT: preferred source not in API list, will use first available")
                else:
                    log.info("MKT: source list unavailable (endpoint may require higher tier)")
            except Exception as _bm_e:
                log.warning(f"MKT: source discovery failed: {_bm_e}")

        while True:
            now = time.time()

            # v10.44d-fix: Periodic stale outcome resolution (every 10 min)
            # Fixes bug where outcomes stay PENDING because EOD clear
            # never triggers during continuous live match windows.
            _pending_now = sum(1 for e in signal_outcomes if not e.get("resolved"))
            if _pending_now > 0 and (now - _last_stale_resolve > 600 or _last_stale_resolve == 0):
                _last_stale_resolve = now
                try:
                    _rc = resolve_stale_outcomes(client)
                    if _rc > 0:
                        log.info(f"Periodic resolution: {_rc} outcome(s) resolved")
                        rewrite_outcomes_file()
                except Exception as _e:
                    log.warning(f"Periodic resolution failed: {_e}")

            # v10.33: EOD report — MUST be before sleep blocks, not after them.
            # Bug: old position was after stats/sleep, but both sleep paths
            # (no-matches + dead-hours) do `continue`, making EOD unreachable.
            # Fix: check at top of loop where nothing can skip it.
            _eod_resolved = [e for e in signal_outcomes if e.get("resolved")]
            _eod_pending = [e for e in signal_outcomes if not e.get("resolved")]
            _eod_has_live = bool(
                [f for f in cached_fixtures if is_tracked_match(f)]
            ) if cached_fixtures else False
            # v10.63: STARTUP EOD RACE FIX — a fresh session has not looked
            # at live state yet (cached_fixtures is empty until the first
            # discovery), which the old condition read as "day over".
            if (_eod_resolved or _eod_pending) and not _eod_lookup_ok and not _eod_defer_note_done:
                _eod_defer_note_done = True
                log.info(
                    "v10.63: EOD deferred — no live-state lookup yet this session "
                    "(first discovery/schedule pending); startup EOD race fixed"
                )
            if (_eod_resolved or _eod_pending) and not _eod_has_live and not fast_monitored and _eod_lookup_ok:
                # v10.71: EOD fires only at the TRUE end of the football day:
                # every scheduled kickoff already started (schedule gate), or
                # the early-morning window (covers late finishes after
                # midnight, when the schedule has already rolled over to the
                # new day). Mid-day monitoring gaps — block transitions,
                # feed-guard holds — must NOT fire the EOD report (its
                # once-per-day marker) or clear the outcome memory.
                _eod_true_end = (
                    _v10_71_no_future_kickoff()
                    or datetime.now(BULGARIA_TZ).hour < 6
                )
                if not _eod_true_end:
                    # Mid-day gap: hold everything; the 10-min periodic
                    # resolver at loop top retries pending outcomes, and the
                    # clear waits for the real end of day.
                    if now - _last_eod_defer_note > 600:
                        _last_eod_defer_note = now
                        log.info(
                            "v10.71: EOD deferred — future kickoffs remain today "
                            "(mid-day gap); outcomes stay in memory"
                        )
                elif now - _last_eod_attempt > 1800:
                    # v10.71 anti-hammer: at most ONE full EOD attempt
                    # (resolve + 90s retry + summary + report + backup) per
                    # 30 min, so stubborn pendings cannot spam the loop.
                    _last_eod_attempt = now
                    # Try resolving any pending outcomes first
                    if _eod_pending:
                        try:
                            resolve_stale_outcomes(client)
                        except Exception:
                            pass
                        _eod_resolved = [e for e in signal_outcomes if e.get("resolved")]
                        _eod_pending = [e for e in signal_outcomes if not e.get("resolved")]

                        # v10.44n: If still pending, wait 90s and retry once more.
                        # The API may not have marked the fixture FT yet even though
                        # the match ended (e.g. Athletic Club 0-2 at 21:00, API not FT at 21:15).
                        if _eod_pending:
                            _pending_fids = set(e["fixture_id"] for e in _eod_pending)
                            log.info(
                                f"v10.44n: {len(_eod_pending)} outcome(s) still pending for "
                                f"fixture(s) {_pending_fids}. Waiting 90s for API to update, then retrying..."
                            )
                            time.sleep(90)
                            try:
                                resolve_stale_outcomes(client)
                            except Exception:
                                pass
                            _eod_resolved = [e for e in signal_outcomes if e.get("resolved")]
                            _eod_pending = [e for e in signal_outcomes if not e.get("resolved")]
                            if _eod_pending:
                                log.warning(
                                    f"v10.44n: {len(_eod_pending)} outcome(s) STILL pending after retry. "
                                    f"Will resolve on next startup."
                                )

                    if _eod_resolved:
                        log_outcome_summary()
                        today_bg = datetime.now(BULGARIA_TZ).strftime("%Y-%m-%d")
                        if eod_report_sent_date != today_bg:
                            try:
                                subprocess.run(
                                    ["python3", "eod_report.py", "--send", "--days", "1", "--quiet"],
                                    cwd="/app", timeout=60,
                                )
                                eod_report_sent_date = today_bg
                                _save_eod_report_sent_date(today_bg)
                                log.info("v10.35: EOD report sent via subprocess")
                            except Exception as e:
                                log.warning(f"v10.33: EOD report subprocess failed: {e}")
                        # v10.44l: Auto-backup ML data to Telegram BEFORE rewrite/clear
                        # v10.44n: Now runs after retry, so fewer unresolved entries in backup
                        try:
                            _backup_ml_data(client)
                        except Exception as _e:
                            log.warning(f"ML data backup failed: {_e}")
                        if _eod_pending:
                            # v10.71 ORPHAN FIX: NEVER clear signal_outcomes while
                            # outcomes are still pending — the old code cleared
                            # here (and in the end-of-loop block) whenever any
                            # resolved entry existed, orphaning pendings in the
                            # file until the next restart (2026-09-05 18:42 UTC:
                            # 9 pendings). Pendings stay in memory; the periodic
                            # resolver finishes them and a later EOD pass clears.
                            log.info(
                                f"v10.71: EOD report/backup done, clear DEFERRED — "
                                f"{len(_eod_pending)} pending outcome(s) remain in memory "
                                f"(periodic resolver will finish them; next EOD pass clears)"
                            )
                        else:
                            # v10.19.3: Rewrite file before clearing
                            rewrite_outcomes_file()
                            signal_outcomes.clear()
                        # v10.44p: Reset daily event fast lane credit counter
                        # v10.50-fix: these names are now declared global in main()
                        # (previously dead local assignments — the credit counter
                        # never actually reset at midnight, only on redeploys)
                        _event_fast_lane_credits_today = 0
                        _event_fast_lane_fids = []
                        _event_fast_lane_last = 0
                        # v10.50: reset fast-lane shadow runtime state
                        # (shadow RECORDS persist in fastlane_shadow.jsonl; only
                        # the in-flight event history resets)
                        _fl_sot_events.clear()
                        _fl_seen_sot_count.clear()
                        _fl_goal_minutes.clear()
                        _fl_seen_goal_count.clear()
                        _fl_shadow_dedupe.clear()
                        _fl_shadow_count_today = 0
                        # v10.75: reset box-burst shadow runtime state
                        # (shadow RECORDS persist in boxburst_shadow.jsonl;
                        # only the in-flight dedupe/history resets — a new
                        # day means new matches, so per-side first-crossings
                        # start fresh)
                        _boxburst_fired.clear()
                        _boxburst_ib_hist.clear()
                        _boxburst_count_today = 0
                        # v10.76: reset goal-burst runtime state
                        # (shadow RECORDS persist in goalburst_shadow.jsonl;
                        # only the in-flight dedupe resets — a new day means
                        # new matches, so crossings start fresh)
                        _goalburst_fired.clear()
                        _goalburst_count_today = 0
                        # v10.53: reset goal-watch runtime state (records persist
                        # in goal_flash.jsonl; only in-flight tracking resets)
                        _goal_watch_fids = []
                        _goal_watch_last = 0.0
                        _goal_watch_credits_today = 0
                        _goal_watch_seen.clear()
                        _goal_watch_flash_keys.clear()
                        _gw_flashes_today = 0
                        _coldstart_warmed.clear()
                        # v10.54: reset surge-watch runtime state (records persist
                        # in surge_watch.jsonl; only in-flight tracking resets)
                        _surge_seen_sot.clear()
                        _surge_seen_shots.clear()
                        _surge_wake_minute.clear()
                        _surge_burst_last_minute.clear()
                        _surge_burst_count.clear()
                        _surge_last_alert_ts.clear()
                        _surge_flood_minute.clear()
                        _surge_fixture_alerts.clear()
                        _surge_alerts_today = 0
                        # v10.56: goal-SOT ledger reset (matches signaled_teams
                        # reset — a new day has no signal references)
                        _pending_goal_sot.clear()
                        _goal_sot_landed.clear()
                        genuine_burst_fixtures.clear()
                        # v10.57: Top-SOT player cache + goal counters
                        _player_sot_cache.clear()
                        _player_sot_cache_goals.clear()
                        _fixture_valid_goals.clear()
                        _latest_sot_event_minute.clear()
                        # v10.58: SOT-growth snapshots
                        _player_sot_cache_built_sot.clear()
                        # v10.63: deferred Top-SOT recovery queue + daily credits
                        _top_sot_retry_queue.clear()
                        _top_sot_retry_credits_today = 0
                        # v10.73: players-API fallback daily budget
                        _players_sot_credits_today = 0
                        # v10.44r: Reset daily goal latency tracking
                        _goal_detect_ts.clear()
                        _goal_game_minute.clear()
                        _goal_stats_ts.clear()
                        _goal_stats_recorded.clear()
                        goal_priority_until.clear()

            # --- Fetch daily schedule (1 call/day, re-fetches on date change) ---
            has_matches = fetch_daily_active_hours(client)
            
            # v10.18: Daily summary removed from auto-send. Use /today in Telegram.
            # Auto-send was burning credits on team form; now command-only, <10 games.
            
            # v10.72: MIDNIGHT LIVE-GAME GUARD — evaluation + bookkeeping.
            # Never abandon a live monitored match at the active-window
            # rollover (Sep 6 23:59->00:00 Sofia: GIL Vicente live at 71',
            # FastWin countdown 41s from firing; the bot slept 17h through
            # the final 19 minutes). While at least one monitored fixture is
            # LIVE and freshly polled (proves active watching, never a stale
            # cache), _mg_hold keeps the normal polling loop running past
            # BOTH sleep branches below — capped at 2h, then sleep wins.
            # v10.74: BOOT-PATH extension — the hold set now ALSO includes
            # fixtures with pending outcomes that the resolver recently
            # verified LIVE (a fresh boot has an empty fast_monitored, so
            # the v10.72 freshness gate could never fire there; the Sep 8
            # 00:01 deploy slept 838m with a ~90' game live). Same 2h cap,
            # same freshness honesty (PENDING_GUARD_FRESH).
            _mg_live = list(dict.fromkeys(
                _v10_72_live_monitored_fids(now) + _v10_74_pending_live_fids(now)
            ))
            _mg_hold = False
            if _mg_live:
                if midnight_guard["since"] == 0.0 and not midnight_guard["gave_up"]:
                    midnight_guard["since"] = now
                    midnight_guard["last_note"] = now
                    _mg_names = "; ".join(_v10_72_fixture_label(f) for f in _mg_live)
                    log.info(
                        f"v10.72/74 MIDNIGHT GUARD: {len(_mg_live)} live fixture(s) "
                        f"(monitored or boot-pending) at window rollover — holding "
                        f"watch until FT (max {MIDNIGHT_GUARD_MAX // 60}m): {_mg_names}"
                    )
                    try:
                        send_telegram(
                            client,
                            f"\U0001f319 v10.72 midnight guard: the active window closed, but "
                            f"{len(_mg_live)} monitored match(es) are still LIVE — keeping "
                            f"watch until full time (max {MIDNIGHT_GUARD_MAX // 3600}h):\n"
                            f"{_mg_names}",
                        )
                    except Exception as _mg_e:
                        log.warning(f"v10.72: midnight guard message failed: {_mg_e}")
                if midnight_guard["since"] > 0.0:
                    if (now - midnight_guard["since"]) <= MIDNIGHT_GUARD_MAX:
                        _mg_hold = True
                        if now - midnight_guard["last_note"] >= MIDNIGHT_GUARD_NOTE_EVERY:
                            midnight_guard["last_note"] = now
                            _mg_names = "; ".join(_v10_72_fixture_label(f) for f in _mg_live)
                            log.info(
                                f"v10.72 MIDNIGHT GUARD: still holding — "
                                f"{len(_mg_live)} live: {_mg_names}"
                            )
                    elif not midnight_guard["gave_up"]:
                        # 2h cap reached — give up holding (notify once).
                        midnight_guard["gave_up"] = True
                        log.warning(
                            f"v10.72 MIDNIGHT GUARD: {MIDNIGHT_GUARD_MAX // 3600}h cap "
                            f"reached — {len(_mg_live)} fixture(s) still live, "
                            f"entering dead hours anyway"
                        )
                        try:
                            send_telegram(
                                client,
                                f"\u26a0\ufe0f v10.72 midnight guard: gave up after "
                                f"{MIDNIGHT_GUARD_MAX // 3600}h — match(es) still live, "
                                f"entering dead-hours sleep.",
                            )
                        except Exception as _mg_e:
                            log.warning(f"v10.72: guard give-up message failed: {_mg_e}")
            elif midnight_guard["since"] > 0.0 or midnight_guard["gave_up"]:
                # No live monitored fixtures remain — clean reset for the
                # next rollover.
                log.info(
                    "v10.72 MIDNIGHT GUARD: no live monitored matches remain — guard released"
                )
                midnight_guard["since"] = 0.0
                midnight_guard["last_note"] = 0.0
                midnight_guard["gave_up"] = False
            
            if not has_matches and not _mg_hold:
                # v10.63: the schedule fetch looked — no tracked matches
                # today. This arms the EOD gate (legitimate wrap-up of
                # yesterday's stragglers, e.g. night restarts).
                if not _eod_lookup_ok:
                    _eod_lookup_ok = True
                    log.info(
                        "v10.63: schedule shows no tracked matches today — EOD gate armed"
                    )
                # v9.7.1: Smart sleep — calculate how long until next meaningful wake
                now_bg = datetime.now(BULGARIA_TZ)
                h = now_bg.hour
                # If night: sleep until NIGHT_HOUR_END
                if NIGHT_HOUR_START <= h < NIGHT_HOUR_END:
                    wake_at = now_bg.replace(hour=NIGHT_HOUR_END, minute=0, second=0, microsecond=0)
                    sleep_s = int((wake_at - now_bg).total_seconds())
                    sleep_s = max(sleep_s, 60)  # min 60s to avoid infinite loop
                else:
                    sleep_s = 1800  # 30 min during daytime no-matches
                log.info(
                    f"No tracked matches today, sleeping until {sleep_s // 60}m (waking every 2m for commands)..."
                )
                # v10.18: EOD summary removed from auto-send. Use /recap or /stats in Telegram.
                # v10.13: Short-interval sleep loop — checks Telegram every 2 min
                # Telegram API is free (0 football credits), so this costs nothing
                while sleep_s > 0:
                    check_telegram_commands(client)
                    chunk = min(sleep_s, 120)  # wake every 2 min
                    time.sleep(chunk)
                    sleep_s -= chunk
                continue
            
            # --- Dead hours (zero API cost) ---
            # Uses dynamic window from today's schedule
            # v10.72: while the midnight guard holds a live match, dead hours
            # are deferred — the loop falls through to normal polling.
            local_hour = datetime.now(BULGARIA_TZ).hour
            if not (dynamic_active_start <= local_hour < dynamic_active_end) and not _mg_hold:
                # v10.18: EOD summary removed from auto-send. Use /recap or /stats in Telegram.

                # v9.7.1: Smart sleep — sleep until window start (or night end if overnight)
                now_bg = datetime.now(BULGARIA_TZ)
                if NIGHT_HOUR_START <= now_bg.hour < NIGHT_HOUR_END:
                    # Deep night — sleep until morning
                    wake_at = now_bg.replace(hour=NIGHT_HOUR_END, minute=0, second=0, microsecond=0)
                else:
                    # Pre-window or post-window — sleep until window starts
                    # (if window already passed today, it means tomorrow's schedule isn't loaded yet)
                    wake_at = now_bg.replace(hour=dynamic_active_start, minute=0, second=0, microsecond=0)
                    # If window start already passed today, next check is 30 min (schedule might update)
                    if wake_at <= now_bg:
                        wake_at = now_bg + timedelta(minutes=30)
                sleep_s = int((wake_at - now_bg).total_seconds())
                sleep_s = max(sleep_s, 60)  # minimum 1 min
                log.info(
                    f"Dead hours (local {local_hour}:00, "
                    f"active {dynamic_active_start:02d}:00-{dynamic_active_end:02d}:00 Bulgaria), "
                    f"sleeping {sleep_s // 60}m {sleep_s % 60}s (waking every 2m for commands)..."
                )
                # v10.13: Short-interval sleep loop — checks Telegram every 2 min
                # Telegram API is free (0 football credits), so this costs nothing
                while sleep_s > 0:
                    check_telegram_commands(client)
                    chunk = min(sleep_s, 120)  # wake every 2 min
                    time.sleep(chunk)
                    sleep_s -= chunk
                continue

            # --- Rate limit backoff ---
            if now < rate_limited_until:
                sleep_remaining = int(rate_limited_until - now)
                log.warning(f"Rate limited, backing off {sleep_remaining}s")
                time.sleep(min(sleep_remaining, 60))
                continue

            budget = get_budget_mode()

            # v10.72: re-arm the STOP probe timer whenever we are NOT in STOP,
            # so a future exhaustion episode probes on its own cadence.
            if budget != "STOP" and stop_probe["last"] != 0.0:
                stop_probe["last"] = 0.0

            if budget == "STOP":
                # v10.72: RENEWAL PROBE — one direct /status call per 30-min
                # wake (bypasses api_get's pre-flight raise ON PURPOSE) so
                # STOP can never again be a dead-end: the 00:00 UTC / 03:00
                # Sofia renewal is observed and the bot wakes itself.
                _sp_now = time.time()
                if stop_probe["last"] == 0.0:
                    # First STOP cycle: arm the timer, probe at the next wake
                    # (quota just hit 0 — renewal is hours away, no point
                    # spending a call to confirm what we already know).
                    stop_probe["last"] = _sp_now
                    log.info(
                        "v10.72: STOP probe armed — first renewal check at the next 30-min wake"
                    )
                elif _sp_now - stop_probe["last"] >= STOP_PROBE_EVERY:
                    stop_probe["last"] = _sp_now
                    try:
                        if _v10_72_stop_renewal_probe(client):
                            try:
                                send_telegram(
                                    client,
                                    f"\u2705 v10.72: API quota RENEWED "
                                    f"({quota_remaining}/{quota_limit}) \u2014 bot waking "
                                    f"itself from STOP mode automatically (no restart needed).",
                                )
                            except Exception as _sp_e:
                                log.warning(f"v10.72: renewal message failed: {_sp_e}")
                            continue  # quota > 0 now — normal loop resumes
                    except Exception as _sp_e:
                        log.warning(f"v10.72: STOP probe error: {_sp_e}")
                log.warning(
                    f"Quota exhausted ({quota_remaining}/{quota_limit}), "
                    f"sleeping 30 min... (v10.72 renewal probe every {STOP_PROBE_EVERY // 60}m)"
                )
                time.sleep(1800)
                continue

            # --- Determine current state from cached data ---
            has_tracked_live = bool(
                [f for f in cached_fixtures if is_tracked_match(f)]
            ) if cached_fixtures else False
            has_candidates = bool(fast_monitored)

            # --- DISCOVERY (timed independently) ---
            discovery_interval = get_discovery_interval(
                budget, has_tracked_live, has_candidates
            )
            need_discovery = (now - last_discovery_time) >= discovery_interval

            if need_discovery:
                try:
                    do_discovery(client)
                    # v10.63: first successful discovery = this session has
                    # LOOKED at live state — EOD gate armed.
                    if not _eod_lookup_ok:
                        _eod_lookup_ok = True
                        log.info("v10.63: first discovery complete — EOD gate armed")
                    has_candidates = bool(fast_monitored)
                    has_tracked_live = bool(
                        [f for f in cached_fixtures
                         if is_tracked_match(f)]
                    )
                    discovery_interval = get_discovery_interval(
                        budget, has_tracked_live, has_candidates
                    )
                    # v10.62: REDEPLOY GUARD [tracking] — once, after the
                    # FIRST discovery: classify every live game + send the
                    # one summary message proving data + coverage survived.
                    if not _redeploy_check_done:
                        _redeploy_check_done = True
                        try:
                            redeploy_tracking_check(client)
                        except Exception as _e:
                            log.warning(f"v10.62: redeploy tracking check failed: {_e}")
                except Exception as e:
                    log.error(f"Discovery failed: {e}")
                    time.sleep(60)
                    continue

            # ------------------------------------------------------------
            # STATS CHECK — batched, interval-gated (v10.25)
            # ------------------------------------------------------------
            # v10.25: Gate stats API call on actual fixture intervals.
            # Previously, stats were called every cycle regardless of
            # per-fixture intervals (the 60s sleep cap forced ~60s cycles).
            # Now, 1 credit/batch is only spent when at least one fixture
            # is due. Hot fixtures (60s interval) still poll every 60s.
            # Cold fixtures (SOT=0/1, 135-180s) skip cycles = real savings.
            # Signal speed: UNCHANGED for hot fixtures (60s),
            # +30s max for SOT=2 (90s interval), negligible for others.

            any_stats_due = False
            stats_int = get_stats_interval(budget)
            if fast_monitored and budget != "STOP":
                for fid in fast_monitored:
                    fixture_interval = get_sot_based_interval(fid, stats_int)
                    elapsed = now - last_stats_check.get(fid, 0)
                    if elapsed >= fixture_interval:
                        any_stats_due = True
                        break

            # v10.44c: Also trigger stats call if data-dead probes are due
            dd_probe_due = False
            dd_probes = []
            if data_dead_fixtures and budget != "STOP":
                now_dd = now
                for fid, ts in list(data_dead_fixtures.items()):
                    if now_dd - ts < 300:
                        continue
                    f = find_cached_fixture(fid)
                    if not f:
                        data_dead_fixtures.pop(fid, None)
                        continue
                    fixture_minute = safe_int(str(f["fixture"].get("elapsed", 0) or 0))
                    if fixture_minute > EXTENDED_MAX:
                        data_dead_fixtures.pop(fid, None)
                        continue
                    dd_probes.append(fid)
                    dd_probe_due = True
                    if len(dd_probes) >= 3:
                        break

            # v10.44g: Also probe pressure-dead fixtures (both teams SOT=0 at 15'+)
            # Same piggyback as data-dead: inject into stats batch to check
            # if SOT has risen since death. The discovery-cached revival check
            # reads from /fixtures?live=all which has NO stats — so revival
            # via SOT>=3 never worked. This fixes the Juventus 15'->82' gap.
            dead_probes = []
            if dead_fixtures and budget != "STOP":
                for fid, (dh, da, death_ts) in list(dead_fixtures.items()):
                    if now - death_ts < 300:
                        continue
                    f = find_cached_fixture(fid)
                    if not f:
                        dead_fixtures.pop(fid, None)
                        continue
                    fixture_minute = safe_int(str(f["fixture"].get("elapsed", 0) or 0))
                    if fixture_minute > EXTENDED_MAX:
                        dead_fixtures.pop(fid, None)
                        continue
                    dead_probes.append(fid)
                    if len(dead_probes) >= 3:
                        break
            if dead_probes:
                dd_probe_due = True

            if any_stats_due and fast_monitored and budget != "STOP":
                ordered = sorted(
                    fast_monitored,
                    key=get_fixture_sot_priority,
                    reverse=True,
                )

                # v10.44c: Inject data-dead + pressure-dead probes into stats batch.
                # The /fixtures?ids= call includes statistics (unlike /fixtures?live=all
                # used by discovery). So we piggyback dead fixtures here to check
                # if SOT has risen since death. Zero extra credits — the batch
                # costs 1 credit regardless of how many IDs (up to 20).
                # This fixes the broken revival that checked cached fixtures
                # (from /fixtures?live=all) which never include statistics.
                all_probes = dd_probes + dead_probes
                if all_probes and len(ordered) + len(all_probes) <= BATCH_SIZE_LIMIT:
                    ordered = ordered + all_probes

                # v10.23: ALL monitored fixtures polled together (1 credit/batch).
                # v10.25: But only when at least one is due (credit gate).
                if (quota_remaining is not None
                        and quota_remaining <= 2):
                    log.warning(
                        "Quota nearly exhausted; skipping stats."
                    )
                else:
                    # Count how many are actually due vs riding free
                    due_count = sum(
                        1 for fid in fast_monitored
                        if (now - last_stats_check.get(fid, 0))
                           >= get_sot_based_interval(fid, stats_int)
                    )
                    log.info(
                        f"  Stats -> {len(ordered)} fixture(s) (batched, "
                        f"{due_count} due): {ordered}"
                    )
                    check_monitored_stats(client, ordered, dd_probes=dd_probes)
            elif dd_probe_due and budget != "STOP":
                # v10.44c/g: No monitored fixtures but dead probes due.
                # Probe them standalone (1 credit for the batch).
                if (quota_remaining is not None
                        and quota_remaining <= 2):
                    log.warning(
                        "Quota nearly exhausted; skipping dead probe."
                    )
                else:
                    all_probes = dd_probes + dead_probes
                    log.info(
                        f"  Stats -> {len(all_probes)} dead probe(s) (data-dead={len(dd_probes)}, pressure-dead={len(dead_probes)}): {all_probes}"
                    )
                    check_monitored_stats(client, all_probes, dd_probes=all_probes)
            elif fast_monitored and not any_stats_due and budget != "STOP":
                # v10.25: Log when stats are skipped (credit saving)
                soonest_due = min(
                    max(0, get_sot_based_interval(fid, stats_int)
                         - (now - last_stats_check.get(fid, 0)))
                    for fid in fast_monitored
                )
                log.info(
                    f"  Stats skipped (no fixture due, next in {soonest_due}s)"
                )

            # v10.63: deferred Top-SOT line recoveries (feed-lag retries) —
            # zero cost while the queue is empty; runs on every tick.
            process_top_sot_retries(client)

            # --- CALCULATE SLEEP ---
            now = time.time()

            # v10.44p: Event fast lane — update target and poll if due
            # v10.67: gate widened — late-retained fixtures keep the event
            # lanes alive after fast_monitored empties at the 85' ceiling
            # (end-of-slate scenario: all games 86'+, watch lane must run)
            if (fast_monitored or _late_retain_fids) and budget not in ("STOP", "EMERGENCY"):
                update_event_fast_lane()
                poll_event_fast_lane(client)
                poll_goal_watch(client)  # v10.53: goal flash alerts (~30s lane)

            next_disc_in = max(
                0, discovery_interval - (now - last_discovery_time)
            )

            next_stats_in = -1
            if fast_monitored:
                stats_int = get_stats_interval(get_budget_mode())
                for fid in fast_monitored:
                    fixture_interval = get_sot_based_interval(fid, stats_int)
                    fid_next = max(
                        0,
                        fixture_interval - (now - last_stats_check.get(fid, 0))
                    )
                    if next_stats_in < 0 or fid_next < next_stats_in:
                        next_stats_in = fid_next

            if next_stats_in < 0:
                sleep_time = next_disc_in
            else:
                sleep_time = min(next_disc_in, next_stats_in)

            sleep_time = max(sleep_time, 10)
            sleep_time = min(sleep_time, 60)
            # v10.53/v10.54: keep the loop tight while the late-game watch
            # lane is active (goal flashes + surge alarms, 30s poll interval)
            if _goal_watch_fids:
                sleep_time = min(sleep_time, GOAL_WATCH_INTERVAL)

            tracked_count = len(
                [f for f in cached_fixtures
                 if is_tracked_match(f)]
            ) if cached_fixtures else 0

            # Status summary
            mode_label = get_budget_mode()
            if is_first_signal_only_mode():
                mode_label += " +FIRST_ONLY"
            elif total_matches_today < FULL_TRACKING_THRESHOLD:
                mode_label += " +FULL"
            # v9.7: Show stats mode and acceleration in status line
            extra_info = ""
            if _ids_endpoint_has_stats is True:
                extra_info += " | Stats:1call"
            elif _ids_endpoint_has_stats is False:
                extra_info += " | Stats:N+1"
            if accelerating_fixtures:
                accel_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in accelerating_fixtures]
                extra_info += f" | SOT-Accel: {', '.join(accel_parts[:3])}{'...' if len(accel_parts) > 3 else ''}"
            if pressure_accelerating:
                paccel_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in pressure_accelerating]
                extra_info += f" | GPS-Accel: {', '.join(paccel_parts[:3])}{'...' if len(paccel_parts) > 3 else ''}"
            if sot_burst_fixtures:
                burst_parts = [f"{fid}(SOT={get_fixture_best_sot(fid)})" for fid in sot_burst_fixtures]
                extra_info += f" | SOT-Burst: {', '.join(burst_parts[:3])}{'...' if len(burst_parts) > 3 else ''}"
            if dead_fixtures:
                extra_info += f" | Dead: {len(dead_fixtures)}"
            if data_dead_fixtures:
                extra_info += f" | DataDead: {len(data_dead_fixtures)}"
            # v10.71: FEED-GUARD incident + frozen-quota watchdog in heartbeat
            if feed_guard["active"]:
                _fg_m = int((now - feed_guard["since"]) // 60)
                extra_info += (
                    f" | FeedGuard: HOLD {len(feed_guard['held_fids'])}fix "
                    f"{_fg_m}m (alarm #{feed_guard['alarms_sent']})"
                )
            # v10.72: shadow-gate would-suppress counters (session)
            if _shadow_tags:
                _sh_str = ", ".join(
                    f"{k}:{v}" for k, v in sorted(_shadow_tags.items())
                )
                extra_info += f" | Shadow: {_sh_str}"
            # v10.72: midnight guard state in heartbeat while holding
            if midnight_guard["since"] > 0.0:
                _mg_m = int((now - midnight_guard["since"]) // 60)
                _mg_state = "GAVE UP" if midnight_guard["gave_up"] else "HOLD"
                extra_info += f" | MidnightGuard: {_mg_state} {_mg_m}m"
            if quota_freeze["count"] >= FEED_GUARD_FROZEN_CALLS:
                extra_info += (
                    f" | QuotaFrozen: {quota_freeze['value']} x{quota_freeze['count']}"
                )
                # v10.71: advisory Telegram warning, max once per 12h
                if now - quota_freeze["last_warn"] > QUOTA_FROZEN_WARN_EVERY:
                    quota_freeze["last_warn"] = now
                    log.warning(
                        f"v10.71: quota counter FROZEN at {quota_freeze['value']} across "
                        f"{quota_freeze['count']} consecutive calls — possible silent "
                        f"API degradation (advisory)"
                    )
                    try:
                        send_telegram(
                            client,
                            f"⚠️ v10.71 freeze-watch: the API quota counter has not "
                            f"decremented across {quota_freeze['count']} calls "
                            f"(stuck at {quota_freeze['value']}/{quota_limit}). "
                            f"Possible silent API degradation — watch for missing "
                            f"signals; api-sports.io status page may still show green.",
                        )
                    except Exception as _qf_e:
                        log.warning(f"v10.71: freeze-watch message failed: {_qf_e}")
            # v10.44r: Show active goal priority windows
            if goal_priority_until:
                _gp_parts = []
                for fid, until in goal_priority_until.items():
                    _remaining = max(0, int(until - now))
                    _gp_parts.append(f"{fid}({_remaining}s)")
                extra_info += f" | GoalPrio: {', '.join(_gp_parts)}"
            fast_window_info = ""
            if fast_sot_until:
                # v10.72: FastWin zombie purge — an expired countdown whose
                # fixture is no longer being polled (FT / dropped / dead)
                # used to linger at "0s" forever in this heartbeat line
                # because is_fast_sot_active() only expires entries for
                # fixtures that keep being polled. Purge stale entries so
                # the line only ever shows live windows.
                _fw_zombies = [
                    _f for _f, _u in fast_sot_until.items()
                    if _u < now - FASTWIN_ZOMBIE_GRACE
                ]
                for _fz in _fw_zombies:
                    fast_sot_until.pop(_fz, None)
                    fast_sot_activated_at_sot.pop(_fz, None)
                if _fw_zombies:
                    log.info(
                        f"  v10.72: purged {len(_fw_zombies)} zombie FastWin "
                        f"entr{'y' if len(_fw_zombies) == 1 else 'ies'} "
                        f"({', '.join(str(f) for f in _fw_zombies)})"
                    )
                parts = []
                for fid, until in fast_sot_until.items():
                    remaining = max(0, int(until - now))
                    sot = get_fixture_best_sot(fid)
                    sig_flag = f"{sum(1 for (f,t) in signaled_teams if f==fid)}/2"
                    parts.append(f"{fid}(SOT={sot},sig={sig_flag},{remaining}s)")
                fast_window_info = f" | FastWin: {', '.join(parts)}"

            total_signals = sum(v["count"] for v in signaled_teams.values())
            stats_str = (
                f"{int(next_stats_in)}s" if next_stats_in >= 0 else "N/A"
            )
            # v9.8: Show outcome tracking in status line (dual: 15-min + full-match)
            resolved_outcomes = [e for e in signal_outcomes if e.get("resolved")]
            pending_outcomes = [e for e in signal_outcomes if not e.get("resolved")]
            h15 = sum(1 for e in resolved_outcomes if e.get("outcome_15min") == "HIT")
            m15 = sum(1 for e in resolved_outcomes if e.get("outcome_15min") == "MISS")
            hf = sum(1 for e in resolved_outcomes if e.get("outcome_full") == "HIT")
            mf = sum(1 for e in resolved_outcomes if e.get("outcome_full") == "MISS")
            outcome_str = ""
            if resolved_outcomes:
                r15 = h15 / len(resolved_outcomes) * 100
                rf = hf / len(resolved_outcomes) * 100
                outcome_str = f" | 15m:{h15}H/{m15}M({r15:.0f}%) full:{hf}H/{mf}M({rf:.0f}%)"
            if pending_outcomes:
                outcome_str += f" | Pend:{len(pending_outcomes)}"

            log.info(
                f"Quota: {quota_remaining}/{quota_limit} | "
                f"Mode: {mode_label} | "
                f"Tracked: {tracked_count} | Mon: {len(fast_monitored)} | "
                f"Sig: {len(signaled_teams)}teams/{len(signaled_fixtures)}fix/{total_signals}sent | "
                f"Keys: {healthy_key_count()}/{len(API_KEYS)} | "
                f"Next disc: {int(next_disc_in)}s | Next stats: {stats_str} | "
                f"Sleep: {int(sleep_time)}s | Reqs: {request_count} | "
                f"Signals: {len(signals_sent)}{fast_window_info}{extra_info}{outcome_str}"
            )

            # v9.8: Log outcome summary when all matches done
            # v10.33: EOD trigger moved to top of loop (before sleep blocks) —
            # that earlier block (~line 7128) already prints the summary and
            # handles the Telegram send. This block now only does file cleanup
            # (stale resolution + rewrite + clear); it no longer re-prints the
            # summary, since doing so here duplicated the exact same log block
            # right after the one above on the same pass (confirmed in
            # production logs, 2026-09-01 08:34:20 — identical summary twice).
            # v10.71: ORPHAN FIX — this block used to rewrite + CLEAR
            # signal_outcomes at every monitoring gap, which (a) orphaned
            # pending outcomes in the file until the next restart (2026-09-05
            # 18:42 UTC: 9 pendings cleared while unresolved — the resolve
            # retry had just failed, yet 155 resolved entries made the truthy
            # `if resolved_outcomes:` fire anyway), and (b) always won the
            # race against the top-of-loop EOD block, so the EOD
            # report/backup only ever fired in rare all-pending edge cases.
            # Now this block LOGS the gap state only: the top-of-loop EOD
            # block owns resolve/report/backup/clear, gated on the true end
            # of day (schedule) and on zero pendings; the 10-min periodic
            # resolver retries pending outcomes in between.
            if (resolved_outcomes or pending_outcomes) and not has_tracked_live and not fast_monitored and _eod_lookup_ok:
                if pending_outcomes:
                    log.info(
                        f"v10.71: monitoring gap — {len(pending_outcomes)} pending "
                        f"outcome(s) held in memory (periodic resolver active; "
                        f"clear deferred until resolved + true EOD)"
                    )
                else:
                    log.info(
                        f"v10.71: monitoring gap — {len(resolved_outcomes)} resolved "
                        f"outcome(s) in memory; EOD actions handled at loop top"
                    )

            # v10.11: Check Telegram /stats command
            check_telegram_commands(client)

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
