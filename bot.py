import os
import sys
import time
import logging
import httpx
from datetime import datetime, timezone
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

# Dead hours UTC — no European league action worth polling
DEAD_HOUR_START = 3   # 03:00 UTC
DEAD_HOUR_END = 11     # 11:00 UTC

MINUTE_MIN = 25
MINUTE_MAX = 80
MINUTE_LATE_MAX = 90  # only for already-tracked teams

# --- State ---
team_state: dict[tuple[int, int], dict] = {}
request_count = 0
signals_sent: list[dict] = []
sent_red_cards: set[int] = set()  # fixture IDs that already triggered red card signal
rate_limited_until: float = 0.0  # timestamp — back off until this time

# --- Per-key quota tracking ---
quota_by_key: dict[str, int | None] = {k: None for k in API_KEYS}

# --- Adaptive polling state ---
last_discovery_time: float = 0.0
last_stats_check: dict[int, float] = {}   # fixture_id -> timestamp of last stats fetch
fast_monitored: set[int] = set()         # fixture IDs currently fast-polled
fast_priority: dict[int, int] = {}       # fixture_id -> rank score (for ordering)
cached_fixtures: list[dict] = []        # last discovery result (reused by stats checks)


def pick_key() -> str:
    def sort_val(k):
        v = quota_by_key.get(k)
        return v if v is not None else 999
    return max(quota_by_key, key=sort_val)


def total_quota() -> int:
    return sum(v for v in quota_by_key.values() if v is not None)


def unknown_key_count() -> int:
    return sum(1 for v in quota_by_key.values() if v is None)


def effective_quota() -> int:
    known = total_quota()
    unknown_bonus = unknown_key_count() * 10
    return min(known + unknown_bonus, 150)


def update_key_quota(resp: httpx.Response, key: str):
    try:
        val = resp.headers.get("x-ratelimit-requests-remaining", "")
        if val:
            new_val = int(val)
            old_val = quota_by_key.get(key)
            quota_by_key[key] = new_val
            if old_val is not None and new_val != old_val:
                log.info(f"  Key {key[:8]}... quota: {old_val} -> {new_val}")
            elif old_val is None:
                log.info(f"  Key {key[:8]}... quota initialized: {new_val}")
    except (ValueError, TypeError):
        pass


# ============================================================
# API HELPERS
# ============================================================

def api_get(client: httpx.Client, endpoint: str, params: dict = None) -> dict:
    global request_count
    key = pick_key()
    request_count += 1
    log.info(f"  API GET {endpoint} (key={key[:8]}..., req #{request_count})")
    resp = client.get(
        f"{API_BASE}{endpoint}",
        params=params,
        headers={"x-apisports-key": key},
    )
    update_key_quota(resp, key)
    if resp.status_code == 429:
        rate_limited_until = time.time() + 300
        raise Exception("Rate limited (429), backing off 5 min")
    resp.raise_for_status()
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
# CANDIDATE RANKING (free data only — no API cost)
# ============================================================

def is_interesting_scoreline(fixture: dict, team_id: int) -> bool:
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0
    if tg - og >= 3:
        return False
    if og - tg >= 4:
        return False
    return True


def get_team_goals(fixture: dict, team_id: int) -> int:
    is_home = fixture["teams"]["home"]["id"] == team_id
    return (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0


def rank_candidate(fixture: dict, team_id: int) -> int:
    score = 0
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0

    if 65 <= minute <= MINUTE_MAX: score += 3
    elif 55 <= minute <= 64: score += 2
    elif MINUTE_MIN <= minute <= 54: score += 1

    tg = get_team_goals(fixture, team_id)
    is_home = fixture["teams"]["home"]["id"] == team_id
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if og > tg:
        if og - tg == 1: score += 4
        else: score += 3
    elif tg == 0 and og == 0:
        score += 2
    elif tg > og:
        if tg - og == 1: score += 2
        else: score += 1
    else:
        score += 2

    fid = fixture["fixture"]["id"]
    prev = team_state.get((fid, team_id))
    if prev and prev.get("last_sot", 0) >= 2:
        score += 3

    return score


def get_score_context(fixture: dict, team_id: int) -> str:
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0
    diff = og - tg

    if diff >= 3: return f"down {tg}-{og}"
    if diff == 2: return f"down 2 goals {tg}-{og}"
    if diff == 1: return f"trailing {tg}-{og}"
    if diff == -1: return f"leading {tg}-{og}"
    if diff <= -2: return f"up {tg}-{og}"
    return f"level {tg}-{og}"


def get_score_emoji(fixture: dict, team_id: int) -> str:
    is_home = fixture["teams"]["home"]["id"] == team_id
    tg = (fixture["goals"]["home"] if is_home else fixture["goals"]["away"]) or 0
    og = (fixture["goals"]["away"] if is_home else fixture["goals"]["home"]) or 0

    if og - tg == 1: return "\U0001f525\U0001f525"
    if tg == og: return "\U0001f525"
    if tg - og == 1: return "\U0001f525"
    if og - tg == 2: return "\U0001f525"
    if tg - og >= 2: return "\u26a0\ufe0f"
    if og - tg >= 3: return "\u26a0\ufe0f"
    return ""


# ============================================================
# QUOTA BUDGET
# ============================================================

def get_budget_mode() -> str:
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return "PROBE"
    if eq >= 60: return "NORMAL"
    elif eq >= 30: return "CAREFUL"
    elif eq >= 10: return "STRICT"
    elif eq >= 1: return "EMERGENCY"
    else: return "STOP"


def get_max_fast_monitored() -> int:
    """Max fixtures to fast-poll simultaneously."""
    eq = effective_quota()
    if unknown_key_count() > 0 and total_quota() == 0:
        return 2
    if eq >= 60: return 5
    elif eq >= 30: return 3
    elif eq >= 10: return 2
    elif eq >= 1: return 1
    else: return 0


def get_discovery_interval(budget_mode: str, has_tracked_live: bool, has_candidates: bool) -> int:
    """Seconds between /fixtures?live=all calls."""
    # No tracked leagues live at all — very slow
    if not has_tracked_live:
        return 1800  # 30 min

    # Tracked live but no active candidates
    if not has_candidates:
        if budget_mode in ("NORMAL", "CAREFUL"):
            return 300   # 5 min
        elif budget_mode == "STRICT":
            return 600   # 10 min
        else:  # EMERGENCY
            return 1200  # 20 min

    # Fast-monitoring active — slower discovery to save budget for stats
    if budget_mode == "NORMAL":
        return 600   # 10 min
    elif budget_mode == "CAREFUL":
        return 600
    elif budget_mode == "STRICT":
        return 900   # 15 min
    else:  # EMERGENCY
        return 1200  # 20 min


def get_stats_interval(budget_mode: str) -> int:
    """Seconds between stats checks for a single fast-monitored fixture."""
    if budget_mode == "NORMAL":
        return 90    # 1.5 min — fast SOT detection
    elif budget_mode == "CAREFUL":
        return 150   # 2.5 min
    elif budget_mode == "STRICT":
        return 300   # 5 min
    else:  # EMERGENCY
        return 600   # 10 min


# ============================================================
# SIGNAL CLASSIFICATION — SOT is king (UNCHANGED)
# ============================================================

TIER_ORDER = ["PRESSURE", "STRONG", "VERY STRONG"]


def bump_tier(tier: str) -> str:
    idx = TIER_ORDER.index(tier)
    if idx < len(TIER_ORDER) - 1:
        return TIER_ORDER[idx + 1]
    return tier


def classify_signal(sot: int, state: dict | None, current_minute: int) -> tuple[str | None, str, float]:
    if sot < 3:
        return None, "", 0.0

    last_sot = state["last_sot"] if state else 0
    if sot <= last_sot:
        return None, "", 0.0

    if sot >= 5:
        tier = "VERY STRONG"
    elif sot >= 4:
        tier = "STRONG"
    else:
        tier = "PRESSURE"

    trend = ""
    sot_rate = 0.0
    if state and state.get("last_minute", 0) > 0:
        prev_min = state["last_minute"]
        prev_sot = state["last_sot"]
        mins_passed = max(current_minute - prev_min, 1)
        sot_rate = (sot - prev_sot) / mins_passed
        trend = f"{prev_sot} -> {sot} SOT in {mins_passed}'"

    if sot_rate >= 0.3:
        tier = bump_tier(tier)

    return tier, trend, sot_rate


def tier_emoji(tier: str) -> str:
    if tier == "VERY STRONG": return "\U0001f534"
    if tier == "STRONG": return "\U0001f7e0"
    return "\U0001f7e1"


def tier_label(tier: str) -> str:
    if tier == "VERY STRONG": return "[RED]"
    if tier == "STRONG": return "[ORANGE]"
    return "[YELLOW]"


# ============================================================
# LOCAL FILTERING (zero API cost — UNCHANGED)
# ============================================================

def find_candidates(fixtures: list[dict]) -> list[tuple[int, int, int]]:
    candidates = []
    for fixture in fixtures:
        lid = fixture["league"]["id"]
        if lid not in LEAGUE_IDS:
            continue
        status = fixture["fixture"]["status"]["short"]
        if status not in LIVE_STATUSES:
            continue
        minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
        fid = fixture["fixture"]["id"]

        home_tid = fixture["teams"]["home"]["id"]
        away_tid = fixture["teams"]["away"]["id"]

        in_window = False
        if MINUTE_MIN <= minute <= MINUTE_MAX:
            in_window = True
        elif MINUTE_MAX < minute <= MINUTE_LATE_MAX:
            if (fid, home_tid) in team_state or (fid, away_tid) in team_state:
                in_window = True

        if not in_window:
            continue

        if is_interesting_scoreline(fixture, home_tid):
            candidates.append((fid, home_tid, rank_candidate(fixture, home_tid)))
        if is_interesting_scoreline(fixture, away_tid):
            candidates.append((fid, away_tid, rank_candidate(fixture, away_tid)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def cleanup_state(live_fixture_ids: set[int]):
    to_delete = [k for k in team_state if k[0] not in live_fixture_ids]
    for k in to_delete:
        del team_state[k]
    if to_delete:
        log.info(f"  Cleaned up state for {len(to_delete)} ended fixture(s)")
    # Also clean red card tracking for ended fixtures
    expired_rc = [fid for fid in sent_red_cards if fid not in live_fixture_ids]
    for fid in expired_rc:
        sent_red_cards.discard(fid)


# ============================================================
# DISCOVERY — fetch live fixtures, update candidates & fast_monitored
# ============================================================

def do_discovery(client: httpx.Client) -> bool:
    """Run one discovery cycle. Updates global state.
    Returns True if discovery succeeded."""
    global last_discovery_time, cached_fixtures

    data = api_get(client, "/fixtures", {"live": "all"})
    cached_fixtures = data.get("response", [])
    last_discovery_time = time.time()

    tracked = [f for f in cached_fixtures if f["league"]["id"] in LEAGUE_IDS]
    budget = get_budget_mode()

    log.info(f"Discovery: Quota: {total_quota()} | Mode: {budget} | "
             f"Live: {len(cached_fixtures)} | Tracked: {len(tracked)} | Requests: {request_count}")

    if tracked:
        for m in tracked:
            minute = m["fixture"]["status"].get("elapsed", "?")
            log.info(f"  -> {m['league']['name']}: {m['teams']['home']['name']} vs "
                     f"{m['teams']['away']['name']} ({m['fixture']['status']['short']} {minute}')")

    # Cleanup ended fixtures from all state
    live_ids = {f["fixture"]["id"] for f in cached_fixtures}
    cleanup_state(live_ids)
    fast_monitored &= live_ids
    fast_priority = {fid: fast_priority[fid] for fid in fast_monitored if fid in live_ids}
    for fid in list(last_stats_check):
        if fid not in live_ids:
            del last_stats_check[fid]

    # Local pre-filter
    candidates = find_candidates(cached_fixtures)
    log.info(f"  -> {len(candidates)} team-candidate(s) ({MINUTE_MIN}-{MINUTE_MAX}')")

    # Update fast_monitored: rank fixtures, keep top N
    max_fast = get_max_fast_monitored()
    fixture_best_rank: dict[int, int] = {}
    for fid, tid, rank in candidates:
        if fid not in fixture_best_rank or rank > fixture_best_rank[fid]:
            fixture_best_rank[fid] = rank

    sorted_fids = sorted(fixture_best_rank.keys(),
                         key=lambda f: fixture_best_rank[f], reverse=True)

    fast_monitored.clear()
    for fid in sorted_fids[:max_fast]:
        fast_monitored.add(fid)
        fast_priority[fid] = fixture_best_rank[fid]

    log.info(f"  -> Fast-monitored: {len(fast_monitored)} fixture(s)")
    return True


# ============================================================
# STATS CHECK — fetch + process stats for one fixture
# ============================================================

def check_fixture_stats(client: httpx.Client, fid: int) -> bool:
    """Fetch stats for one fixture and process signals.
    Returns True if the request was made (even if no signal)."""
    # Find fixture in cache
    fixture = None
    for f in cached_fixtures:
        if f["fixture"]["id"] == fid:
            fixture = f
            break
    if not fixture:
        fast_monitored.discard(fid)
        return False

    # Pre-check: still live and in window?
    status = fixture["fixture"]["status"]["short"]
    if status not in LIVE_STATUSES:
        fast_monitored.discard(fid)
        return False
    minute = fixture["fixture"]["status"].get("elapsed", 0) or 0
    if minute > MINUTE_LATE_MAX:
        fast_monitored.discard(fid)
        log.info(f"  -> Fixture {fid} past {MINUTE_LATE_MAX}', removed from fast monitoring")
        return False

    # Fetch stats
    try:
        stats_data = api_get(client, "/fixtures/statistics", {"fixture": fid})
    except Exception as e:
        log.warning(f"  Stats failed for fixture {fid}: {e}")
        return False

    last_stats_check[fid] = time.time()

    stats = stats_data.get("response", [])
    if not stats:
        return True

    # Parse all team stats into a dict
    teams_data = {}
    for team_entry in stats:
        tname = team_entry["team"]["name"]
        tmap = {}
        for s in team_entry.get("statistics", []):
            val = s.get("value", "0")
            if val is None:
                val = "0"
            tmap[s["type"]] = str(val).strip()
        teams_data[tname] = tmap

    # --- RED CARD CHECK (per fixture, once) ---
    if fid not in sent_red_cards:
        total_reds = 0
        red_details = []
        for tname, tstats in teams_data.items():
            rc = 0
            rc_raw = tstats.get("Red Cards", "0")
            try:
                rc = int(rc_raw)
            except (ValueError, TypeError):
                pass
            if rc > 0:
                total_reds += rc
                red_details.append(f"{tname}: {rc}")

        if total_reds >= 1:
            sent_red_cards.add(fid)
            league = LEAGUE_IDS.get(fixture["league"]["id"], fixture["league"]["name"])
            home = fixture["teams"]["home"]["name"]
            away = fixture["teams"]["away"]["name"]
            sh = fixture["goals"]["home"]
            sa = fixture["goals"]["away"]

            rc_msg = (
                f"\U0001f7e5 <b>RED CARD ALERT</b>\n\n"
                f"{home}  {sh} - {sa}  {away}\n"
                f"{league}  {minute}'\n\n"
                f"\U0001f7e5 Red Cards: {', '.join(red_details)}\n"
            )
            # Add SOT context if available
            sot_parts = []
            for tname, tstats in teams_data.items():
                sot_val = tstats.get("Shots on Goal", "?")
                sot_parts.append(f"{tname}: {sot_val} SOT")
            if sot_parts:
                rc_msg += f"\U0001f3af {', '.join(sot_parts)}\n"

            if send_telegram(client, rc_msg):
                log.info(f"  RED CARD: {home} vs {away} - {', '.join(red_details)} (fixture {fid})")

    # --- SOT SIGNAL CHECK (per team in this fixture) ---
    home_tid = fixture["teams"]["home"]["id"]
    away_tid = fixture["teams"]["away"]["id"]
    for tid in [home_tid, away_tid]:
        tname = None
        if fixture["teams"]["home"]["id"] == tid:
            tname = fixture["teams"]["home"]["name"]
        elif fixture["teams"]["away"]["id"] == tid:
            tname = fixture["teams"]["away"]["name"]
        if not tname or tname not in teams_data:
            continue

        tstats = teams_data[tname]
        try:
            sot = int(tstats.get("Shots on Goal", "0"))
            total_shots = int(tstats.get("Total Shots", "0"))
            corners = int(tstats.get("Corner Kicks", "0"))
            possession = tstats.get("Ball Possession", "50%")
        except (ValueError, TypeError):
            continue

        state = team_state.get((fid, tid))
        tier, trend, sot_rate = classify_signal(sot, state, minute)

        current_goals = get_team_goals(fixture, tid)
        prev_goals = state.get("last_goals", current_goals) if state else current_goals
        scored_since_last = current_goals > prev_goals

        team_state[(fid, tid)] = {
            "last_sot": sot,
            "last_minute": minute,
            "last_goals": current_goals,
        }

        if tier:
            ctx = get_score_context(fixture, tid)
            score_emoji = get_score_emoji(fixture, tid)
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

            msg = (
                f"{tier_emoji(tier)} {tier_label(tier)} {tier} GOAL PRESSURE {score_emoji}\n\n"
                f"{home}  {sh} - {sa}  {away}\n"
                f"{league}  {minute}'\n\n"
                f"{tname} ({ctx})\n"
                f"  Shots on target: {sot}\n"
                f"  Total shots: {total_shots}\n"
                f"  Opponent SOT: {opp_sot}\n"
                f"  Possession: {possession}\n"
                f"  Corners: {corners}\n"
            )
            if trend:
                msg += f"  Trend: {trend}\n"
            if scored_since_last:
                msg += f"  \u26bd Scored since last signal — pressure continues\n"

            if send_telegram(client, msg):
                log.info(f"  SIGNAL {tier}: {tname} ({ctx}) - "
                         f"{sot} SOT, {total_shots} shots (fixture {fid})")

            signals_sent.append({
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "fixture": fid, "team": tname, "league": league,
                "minute": minute, "sot": sot, "total_shots": total_shots,
                "corners": corners, "context": ctx, "tier": tier,
                "trend": trend, "scored_since_last": scored_since_last,
            })

    return True


# ============================================================
# MAIN LOOP — adaptive polling with independent timers
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Football Bot v9 — Adaptive Polling")
    log.info("=" * 60)
    log.info(f"Tracking {len(LEAGUE_IDS)} leagues: {list(LEAGUE_IDS.keys())}")
    log.info(f"API keys: {len(API_KEYS)} (quota from API headers)")
    log.info("")
    log.info("Architecture: independent discovery + stats timers")
    log.info("  No tracked live:     discovery every 30 min")
    log.info("  Tracked, no cand:    discovery every 5 min")
    log.info("  Fast monitoring:     stats every 90s, discovery every 10 min")
    log.info("  Dead hours:          03:00-11:00 UTC (0 requests)")
    log.info("")
    log.info("Budget (treat ~90/100 as usable):")
    log.info("  >=60: NORMAL   | 30-59: CAREFUL | 10-29: STRICT")
    log.info("  1-9: EMERGENCY | 0: STOP")
    log.info("")
    log.info("Signal logic (UNCHANGED):")
    log.info("  SOT Trigger: 3+ SOT AND SOT increased since last check")
    log.info("  Red Card Trigger: >=1 red card (once per fixture)")
    log.info("  3 SOT=PRESSURE  4 SOT=STRONG  5+ SOT=VERY STRONG")
    log.info("  Rapid SOT growth (>=0.3/min) bumps tier")
    log.info("")
    log.info(f"Minute window: {MINUTE_MIN}'-{MINUTE_MAX}' ({MINUTE_MIN}'-{MINUTE_LATE_MAX}' tracked)")
    log.info("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        while True:
            now = time.time()
            utc_hour = datetime.now(timezone.utc).hour

            # --- Dead hours (zero API cost) ---
            if DEAD_HOUR_START <= utc_hour < DEAD_HOUR_END:
                log.info(f"Dead hours ({DEAD_HOUR_START}:00-{DEAD_HOUR_END}:00 UTC), sleeping 30 min...")
                time.sleep(1800)
                continue

            # --- Rate limit backoff ---
            if now < rate_limited_until:
                sleep_remaining = int(rate_limited_until - now)
                log.warning(f"Rate limited, backing off {sleep_remaining}s")
                time.sleep(min(sleep_remaining, 60))
                continue

            budget = get_budget_mode()

            if budget == "STOP":
                log.warning(f"Quota exhausted ({total_quota()}), sleeping 30 min...")
                time.sleep(1800)
                continue

            # --- Determine current state from cached data ---
            has_tracked_live = bool(
                [f for f in cached_fixtures if f["league"]["id"] in LEAGUE_IDS]
            ) if cached_fixtures else False
            has_candidates = bool(fast_monitored)

            # --- DISCOVERY (timed independently) ---
            discovery_interval = get_discovery_interval(budget, has_tracked_live, has_candidates)
            need_discovery = (now - last_discovery_time) >= discovery_interval

            if need_discovery:
                try:
                    do_discovery(client)
                    # Refresh state after discovery
                    has_candidates = bool(fast_monitored)
                    has_tracked_live = bool(
                        [f for f in cached_fixtures if f["league"]["id"] in LEAGUE_IDS]
                    )
                    # Recalculate discovery interval with fresh state
                    discovery_interval = get_discovery_interval(budget, has_tracked_live, has_candidates)
                except Exception as e:
                    log.error(f"Discovery failed: {e}")
                    time.sleep(60)
                    continue

            # --- STATS CHECKS (one per loop iteration, highest priority first) ---
            stats_interval = get_stats_interval(budget)
            stats_checked = False

            if fast_monitored and budget != "STOP":
                # Sort by priority (highest first)
                ordered = sorted(fast_monitored,
                                key=lambda f: fast_priority.get(f, 0), reverse=True)
                for fid in ordered:
                    time_since_check = now - last_stats_check.get(fid, 0)
                    if time_since_check < stats_interval:
                        continue
                    if effective_quota() <= 3:
                        log.warning("  Quota nearly gone, skipping stats.")
                        break

                    check_fixture_stats(client, fid)
                    stats_checked = True
                    break  # One stats request per loop iteration

            # --- CALCULATE SLEEP ---
            now = time.time()
            next_disc_in = max(0, discovery_interval - (now - last_discovery_time))

            next_stats_in = 9999
            if fast_monitored:
                for fid in fast_monitored:
                    fid_next = max(0, stats_interval - (now - last_stats_check.get(fid, 0)))
                    next_stats_in = min(next_stats_in, fid_next)
            if not fast_monitored:
                next_stats_in = -1  # N/A

            sleep_time = min(next_disc_in, next_stats_in) if next_stats_in >= 0 else next_disc_in
            sleep_time = max(sleep_time, 10)   # min 10s
            sleep_time = min(sleep_time, 60)   # max 60s — re-evaluate often

            tracked_count = len([f for f in cached_fixtures if f["league"]["id"] in LEAGUE_IDS]) if cached_fixtures else 0
            stats_str = f"{int(next_stats_in)}s" if next_stats_in >= 0 else "N/A"
            log.info(
                f"Quota: {total_quota()} | Mode: {budget} | "
                f"Tracked: {tracked_count} | Fast: {len(fast_monitored)} | "
                f"Next disc: {int(next_disc_in)}s | Next stats: {stats_str} | "
                f"Sleep: {int(sleep_time)}s | Reqs: {request_count} | Signals: {len(signals_sent)}"
            )

            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
