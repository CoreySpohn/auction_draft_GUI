import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------- Raw fetchers --------------------


def fetch_league_raw(league_id: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.sleeper.app/v1/league/{league_id}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def fetch_draft_raw(draft_id: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.sleeper.app/v1/draft/{draft_id}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def fetch_draft_picks(draft_id: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def fetch_league_users(league_id: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.sleeper.app/v1/league/{league_id}/users"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def fetch_league_rosters(league_id: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.sleeper.app/v1/league/{league_id}/rosters"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


# -------------------- Player cache --------------------


def get_players(
    cache_path: Path = Path(".cache/sleeper_players_nfl.json"),
    max_age_hours: float = 168.0,
) -> Optional[Dict[str, Any]]:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if cache_path.exists():
        try:
            mtime = cache_path.stat().st_mtime
            if (time.time() - mtime) / 3600.0 <= max_age_hours:
                with cache_path.open("r") as f:
                    return json.load(f)
        except Exception:
            pass

    url = "https://api.sleeper.app/v1/players/nfl"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        try:
            with cache_path.open("w") as f:
                json.dump(data, f)
        except Exception:
            pass
        return data
    except requests.exceptions.RequestException:
        if cache_path.exists():
            try:
                with cache_path.open("r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None


# -------------------- Parsing/derivation --------------------


def derive_scoring_type(scoring_settings: Dict[str, Any]) -> str:
    ppr = float(scoring_settings.get("rec", 0) or 0)
    if ppr >= 1.0:
        return "ppr"
    if 0 < ppr < 1.0:
        return "half_ppr"
    return "standard"


def parse_roster_counts(roster_positions: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(roster_positions, list):
        return counts
    for pos in roster_positions:
        counts[pos] = counts.get(pos, 0) + 1
    return {
        "QB": counts.get("QB", 0),
        "RB": counts.get("RB", 0),
        "WR": counts.get("WR", 0),
        "TE": counts.get("TE", 0),
        "FLEX": counts.get("FLEX", 0),
        "K": counts.get("K", 0),
        "DEF": counts.get("DEF", 0),
        "DL": counts.get("DL", 0),
        "LB": counts.get("LB", 0),
        "DB": counts.get("DB", 0),
        "IDP": counts.get("IDP", 0) + counts.get("IDP_FLEX", 0),
        "B": counts.get("BN", 0),
    }


def is_idp_enabled(roster_positions: Any) -> bool:
    if not isinstance(roster_positions, list):
        return False
    for pos in roster_positions:
        if isinstance(pos, str) and (
            pos.startswith("IDP") or pos in ("DL", "LB", "DB")
        ):
            return True
    return False


def map_offense_scoring(scoring_settings: Dict[str, Any]) -> Dict[str, float]:
    return {
        "PASSING_ATT": 0.0,
        "PASSING_CMP": float(scoring_settings.get("pass_cmp", 0) or 0),
        "PASSING_INC": 0.0,
        "PASSING_YDS": float(scoring_settings.get("pass_yd", 0) or 0),
        "PASSING_TDS": float(scoring_settings.get("pass_td", 0) or 0),
        "PASSING_INTS": float(scoring_settings.get("pass_int", 0) or 0),
        "RUSHING_ATT": 0.0,
        "RUSHING_YDS": float(scoring_settings.get("rush_yd", 0) or 0),
        "RUSHING_TDS": float(scoring_settings.get("rush_td", 0) or 0),
        "MISC_FL": float(scoring_settings.get("fum_lost", 0) or 0),
        "RECEIVING_REC": float(scoring_settings.get("rec", 0) or 0),
        "RECEIVING_YDS": float(scoring_settings.get("rec_yd", 0) or 0),
        "RECEIVING_TDS": float(scoring_settings.get("rec_td", 0) or 0),
    }


def map_idp_scoring(scoring_settings: Dict[str, Any]) -> Dict[str, float]:
    tkl = scoring_settings.get("idp_tkl")
    if tkl is None:
        tkl = scoring_settings.get("idp_tkl_solo", 0)
    return {
        "Tackle": float(tkl or 0),
        "Assist": float(scoring_settings.get("idp_tkl_ast", 0) or 0),
        "Sack": float(scoring_settings.get("idp_sack", 0) or 0),
        "PD": float(scoring_settings.get("idp_pass_def", 0) or 0),
        "INT": float(scoring_settings.get("idp_int", 0) or 0),
        "FF": float(scoring_settings.get("idp_ff", 0) or 0),
        "FR": float(scoring_settings.get("idp_fum_rec", 0) or 0),
    }


# -------------------- Config builders --------------------


def build_league_overview(league_id: str) -> Optional[Dict[str, Any]]:
    league = fetch_league_raw(league_id)
    if not league:
        return None
    roster_positions = league.get("roster_positions", [])
    scoring_settings = league.get("scoring_settings", {}) or {}
    settings = league.get("settings", {}) or {}

    roster_counts = parse_roster_counts(roster_positions)
    idp = is_idp_enabled(roster_positions)
    scoring_type = derive_scoring_type(scoring_settings)
    te_premium_bonus = float(scoring_settings.get("bonus_rec_te", 0) or 0)
    te_premium = te_premium_bonus != 0
    teams = (
        int(league.get("total_rosters"))
        if league.get("total_rosters") is not None
        else int(settings.get("num_teams", 0) or 0)
    )

    return {
        "name": league.get("name"),
        "season": league.get("season"),
        "teams": teams,
        "roster_positions": roster_positions,
        "roster_counts": roster_counts,
        "idp_enabled": idp,
        "scoring_type": scoring_type,
        "scoring_settings_raw": scoring_settings,
        "scoring_coeffs": map_offense_scoring(scoring_settings),
        "def_scoring_coeffs": map_idp_scoring(scoring_settings) if idp else None,
        "te_premium": te_premium,
        "te_premium_bonus": te_premium_bonus,
    }


def build_draft_overview(draft_id: str) -> Optional[Dict[str, Any]]:
    draft = fetch_draft_raw(draft_id)
    if not draft:
        return None
    settings = draft.get("settings", {}) or {}
    return {
        "budget": int(settings.get("budget", 0) or 0),
        "teams": int(settings.get("teams", 0) or 0),
        "type": draft.get("type"),
        "rounds": int(settings.get("rounds", 0) or 0),
        "pick_timer": int(settings.get("pick_timer", 0) or 0),
        "nomination_timer": int(settings.get("nomination_timer", 0) or 0),
        "settings_raw": settings,
    }


def baseline_values_request(
    scoring_type: str, idp_enabled: bool, season: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    if not season:
        season = "2025"
    base_url = (
        f"https://api.sleeper.com/players/nfl/values/regular/{season}/{scoring_type}"
    )
    params = {"idp": str(idp_enabled).lower(), "is_dynasty": "false"}
    return base_url, params


def build_combined_config(league_id: str, draft_id: str) -> Optional[Dict[str, Any]]:
    league = build_league_overview(league_id)
    draft = build_draft_overview(draft_id)
    if not league or not draft:
        return None
    idp = bool(league.get("idp_enabled"))
    scoring_type = league.get("scoring_type", "standard")
    url, params = baseline_values_request(scoring_type, idp, league.get("season"))
    combined = {**league, **{"draft": draft}}
    combined.update({"baseline_values_url": url, "baseline_values_params": params})
    return combined


# -------------------- Display helpers --------------------


def summarize_pick(pick: Dict[str, Any], players: Optional[Dict[str, Any]]) -> str:
    meta = pick.get("metadata", {}) or {}
    player_id = str(pick.get("player_id") or meta.get("player_id") or "")
    first = meta.get("first_name")
    last = meta.get("last_name")
    position = meta.get("position")
    team = meta.get("team")
    if (not first or not last) and players and player_id in players:
        pdata = players[player_id] or {}
        first = first or pdata.get("first_name")
        last = last or pdata.get("last_name")
        position = position or pdata.get("position")
        team = team or pdata.get("team")
    amount = meta.get("amount")
    amount_text = f" for ${amount}" if amount else ""
    name = f"{first or ''} {last or ''}".strip() or f"ID:{player_id}"
    roster = pick.get("roster_id")
    pick_no = pick.get("pick_no")
    rnd = pick.get("round")
    slot = pick.get("draft_slot")
    parts = [
        f"Pick {pick_no} (R{rnd}, Slot {slot})",
        f"Roster {roster}",
        f"{name} ({position or 'NA'}/{team or 'FA'}){amount_text}",
    ]
    return " - ".join(parts)


# -------------------- Auction values (simple, per test_auction_values.py) --------------------


def get_sleeper_auction_value(
    base_value: Optional[float], league_budget: int, standard_budget: int = 200
) -> int:
    """
    Mirrors test_auction_values.get_sleeper_auction_value:
      - scale by (league_budget / standard_budget)
      - round to nearest integer
      - clamp minimum to 1
    """
    if base_value is None or base_value <= 0:
        return 1
    inflated_value = base_value * (league_budget / standard_budget)
    rounded_value = round(inflated_value)
    final_value = max(int(rounded_value), 1)
    return final_value


def get_league_auction_values_with_ranks(
    cfg: Dict[str, Any],
) -> Optional[Dict[int, Dict[str, int]]]:
    """
    Fetch Sleeper baseline values (raw) and convert to league-scaled auction values with ranks.

    Assumptions (as used in tests):
      - Baseline endpoint returns a JSON object mapping player_id (string) -> numeric base value
      - We scale each base value to the league budget using get_sleeper_auction_value
      - Ranks are computed by sorting scaled values descending (1 is best)
    """
    url = cfg.get("baseline_values_url")
    params = cfg.get("baseline_values_params", {})
    draft = cfg.get("draft", {}) or {}
    try:
        budget = int(draft.get("budget", 200))
    except Exception:
        budget = 200
    if not url:
        return None
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        baseline = resp.json()  # expected: {"<player_id>": <float>, ...}
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None

    # Scale and collect
    scaled: Dict[int, int] = {}
    for pid_str, base_val in baseline.items() if isinstance(baseline, dict) else []:
        try:
            pid_int = int(pid_str)
        except Exception:
            continue
        try:
            base_val_f = float(base_val)
        except Exception:
            base_val_f = None
        scaled[pid_int] = get_sleeper_auction_value(base_val_f, budget)

    if not scaled:
        return {}

    # Determine position for each player from Sleeper players dump
    players = get_players() or {}
    pid_to_pos: Dict[int, str] = {}
    for pid_str, pdata in players.items():
        try:
            pid_int = int(pid_str)
        except Exception:
            continue
        if not isinstance(pdata, dict):
            continue
        pos = pdata.get("position")
        if isinstance(pos, str) and pos:
            pid_to_pos[pid_int] = pos

    # Group by position and compute ranks (descending value) within each group
    out: Dict[int, Dict[str, int]] = {}
    pos_to_ids: Dict[str, list] = {}
    for pid, val in scaled.items():
        pos = pid_to_pos.get(pid, "UNK")
        pos_to_ids.setdefault(pos, []).append(pid)

    for pos, ids in pos_to_ids.items():
        ids_sorted = sorted(ids, key=lambda k: scaled[k], reverse=True)
        for rank_idx, pid in enumerate(ids_sorted, start=1):
            out[pid] = {"value": int(scaled[pid]), "rank": int(rank_idx)}

    # Any players without position info will be ranked in the UNK bucket above
    return out
