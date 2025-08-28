import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
from io import StringIO


FANTASYPROS_URL = "https://www.fantasypros.com/nfl/projections/{pos}.php?max-yes=true&min-yes=true&week=draft"


def _parse_triplet_number(cell: str) -> Tuple[float, float, float]:
    """
    Parse a FantasyPros table cell that encodes avg/high/low as a single string.

    The site renders three numbers in one cell; pandas.read_html preserves it as
    a single string. This parser mirrors the logic previously used in the UI
    layer to split those values based on decimal transitions.
    """
    if isinstance(cell, (int, float)):
        v = float(cell)
        return v, v, v
    s = str(cell)
    # Defensive: sometimes values like '-' appear
    if s.strip() in {"-", ""}:
        return 0.0, 0.0, 0.0
    parts = ["", "", ""]
    idx = 0
    transition = False
    for ch in s:
        if ch == ",":
            continue
        parts[idx] += ch
        if transition:
            idx = min(idx + 1, 2)
            transition = False
        if ch == ".":
            transition = True
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        # Fallback: attempt whitespace split
        try:
            tokens = s.replace(",", "").split()
            if len(tokens) >= 3:
                return float(tokens[0]), float(tokens[1]), float(tokens[2])
        except Exception:
            pass
    return 0.0, 0.0, 0.0


def _fetch_offense_tables() -> Dict[str, pd.DataFrame]:
    positions = ["QB", "RB", "WR", "TE"]
    avgdfs = []
    highdfs = []
    lowdfs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    for position in positions:
        url = FANTASYPROS_URL.format(pos=position.lower())
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        # First table holds projections
        table = pd.read_html(StringIO(resp.text))[0]
        # Build avg/high/low dicts
        avg_stats: Dict[str, Dict[str, float]] = {}
        high_stats: Dict[str, Dict[str, float]] = {}
        low_stats: Dict[str, Dict[str, float]] = {}
        relevant_categories = {"PASSING", "RUSHING", "RECEIVING", "MISC"}
        irrelevant_attributes = {"FPTS"}

        for _, row in table.iterrows():
            # Player name is everything but the last token (team/pos marker)
            # Use iloc for positional indexing to avoid deprecation warnings
            player_string = row.iloc[0]
            parts = str(player_string).split(" ")
            player_name = " ".join(parts[:-1]).strip()
            if not player_name:
                continue
            if player_name not in avg_stats:
                avg_stats[player_name] = {"POSITION": position}
                high_stats[player_name] = {"POSITION": position}
                low_stats[player_name] = {"POSITION": position}
            # MultiIndex columns → flat tuples
            for category, attribute in row.index.to_flat_index():
                stat_name = f"{category}_{attribute}"
                if (category in relevant_categories) and (
                    attribute not in irrelevant_attributes
                ):
                    cell = row[category][attribute]
                    avg_val, high_val, low_val = _parse_triplet_number(cell)
                    avg_stats[player_name][stat_name] = avg_val
                    high_stats[player_name][stat_name] = high_val
                    low_stats[player_name][stat_name] = low_val

        avgdf = _coerce_numeric_and_fill(pd.DataFrame(avg_stats).T)
        highdf = _coerce_numeric_and_fill(pd.DataFrame(high_stats).T)
        lowdf = _coerce_numeric_and_fill(pd.DataFrame(low_stats).T)
        if position == "QB":
            for df in (avgdf, highdf, lowdf):
                if {"PASSING_ATT", "PASSING_CMP"}.issubset(df.columns):
                    df["PASSING_INC"] = df["PASSING_ATT"] - df["PASSING_CMP"]
        avgdfs.append(avgdf)
        highdfs.append(highdf)
        lowdfs.append(lowdf)

    _avg = _coerce_numeric_and_fill(pd.concat(avgdfs))
    _avg.insert(0, "NAME", _avg.index)
    _avg = _avg.reset_index(drop=True)

    _high = _coerce_numeric_and_fill(pd.concat(highdfs))
    _high.insert(0, "NAME", _high.index)
    _high = _high.reset_index(drop=True)

    _low = _coerce_numeric_and_fill(pd.concat(lowdfs))
    _low.insert(0, "NAME", _low.index)
    _low = _low.reset_index(drop=True)

    return {"average": _avg, "high": _high, "low": _low}


def _fetch_idp_tables() -> pd.DataFrame:
    # Dictionary mapping defensive positions to their respective IDs on fftoday
    def_pos_ids = {"DL": 50, "LB": 60, "DB": 70}
    url_template = (
        "https://www.fftoday.com/rankings/playerproj.php?PosID={pos_id}&LeagueID="
    )
    avgdfs = []
    for position, pos_id in def_pos_ids.items():
        url = url_template.format(pos_id=pos_id)
        try:
            tables = pd.read_html(url)
        except ValueError:
            continue
        if len(tables) < 2:
            continue
        # Second table, then clean header and rows as originally done
        df = tables[7]
        position_data = df.iloc[1:, 1:].reset_index(drop=True)
        column_names = position_data.iloc[0].values
        column_names[0] = "Player"
        position_data.columns = column_names
        position_data = position_data.drop(position_data.index[0]).reset_index(
            drop=True
        )
        avg_stats: Dict[str, Dict[str, float]] = {}
        for _, row in position_data.iterrows():
            player_name = row["Player"]
            if player_name == "Josh Allen":
                player_name = "Josh Allen DE"
            if player_name not in avg_stats:
                avg_stats[player_name] = {"POSITION": position}
            for col in position_data.columns:
                if col in {"Player", "Bye", "Tm", "FPts"}:
                    continue
                val = float(row[col]) if pd.notna(row[col]) else 0
                avg_stats[player_name][col] = val
        avgdf = _coerce_numeric_and_fill(pd.DataFrame(avg_stats).T)
        avgdfs.append(avgdf)
    if not avgdfs:
        return pd.DataFrame(columns=["NAME", "POSITION"])  # empty
    _avg = _coerce_numeric_and_fill(pd.concat(avgdfs))
    _avg.insert(0, "NAME", _avg.index)
    _avg = _avg.reset_index(drop=True)
    return _avg


def load_fantasypros_nfl(
    cache_path: Path,
    scoring_coeffs: Dict[str, float],
    te_premium: bool,
    include_idp: bool,
    def_scoring_coeffs: Dict[str, float] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch NFL projections (offense from FantasyPros avg/high/low; optional IDP from FFToday),
    compute fantasy points and PPW for avg/high/low, and return a dict.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data = _fetch_offense_tables()
    if include_idp:
        idp_avg = _fetch_idp_tables()
        if not idp_avg.empty:
            data["average"] = (
                pd.concat([data["average"], idp_avg]).fillna(0).reset_index(drop=True)
            )

    # Compute fantasy points and PPW
    for key, statdf in data.items():
        # Only offense tables have these names; IDP brought in under "average"
        statdf["FANTASY_POINTS"] = np.zeros(statdf.shape[0])
        for name, coeff in scoring_coeffs.items():
            if name in statdf.columns:
                statdf["FANTASY_POINTS"] += statdf[name] * coeff
        if include_idp and key == "average" and def_scoring_coeffs is not None:
            for name, coeff in def_scoring_coeffs.items():
                if name in statdf.columns:
                    statdf["FANTASY_POINTS"] += statdf[name] * coeff
        if (
            te_premium
            and "POSITION" in statdf.columns
            and "RECEIVING_REC" in statdf.columns
        ):
            te_mask = statdf["POSITION"] == "TE"
            statdf.loc[te_mask, "FANTASY_POINTS"] += (
                0.5 * statdf.loc[te_mask, "RECEIVING_REC"]
            )
        statdf["PPW"] = statdf["FANTASY_POINTS"] / 17
        if "POSITION" in statdf.columns:
            statdf["Rank"] = statdf.groupby("POSITION")["PPW"].rank(ascending=False)

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data


def _coerce_numeric_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric-like columns to numeric dtype explicitly, then fill NaN in
    numeric columns only. Avoids pandas FutureWarning about downcasting on fillna.
    """
    out = df.copy()
    non_numeric_cols = [c for c in out.columns if c in ("POSITION", "NAME")]
    candidate_cols = [c for c in out.columns if c not in non_numeric_cols]
    for c in candidate_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    num_cols = out.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].fillna(0)
    return out
