from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

# Import the C++ nanobind module
from cpp_core import PlayerTable, Player


def _normalize_name(s: str) -> str:
    # Remove punctuation
    s = s.replace(".", "").replace(",", "").replace("'", "")
    return " ".join(str(s).lower().strip().split())


def build_player_table_from_csv(
    csv_path: Path | str,
) -> Tuple[PlayerTable, Dict[str, int]]:
    """
    Build a PlayerTable from the consolidated ID CSV and return a mapping
    from normalized merge_name -> sleeper_id for later joins.

    Args:
        csv_path: Path to .cache/db_playerids.csv

    Returns:
        (player_table, merge_name_to_sleeper_id)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=["name", "merge_name", "position", "team", "sleeper_id"],
        # Read all as strings first for robust cleaning, then coerce types
        dtype={
            "name": "string",
            "merge_name": "string",
            "position": "string",
            "team": "string",
            "sleeper_id": "string",
        },
        keep_default_na=True,
        na_values=["", "None", "none", "NA", "NaN"],
    )
    # Coerce sleeper_id to nullable integer dtype
    df["sleeper_id"] = pd.to_numeric(df["sleeper_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["sleeper_id"]).copy()
    # Normalize for robust joins
    df["merge_key"] = df["merge_name"].map(_normalize_name)

    table = PlayerTable()
    for _, r in df.iterrows():
        name = r.get("name") or ""
        pos = r.get("position") or ""
        team = r.get("team") or ""
        # r["sleeper_id"] is pandas Int64; convert to Python int
        sid = int(r["sleeper_id"])  # safe after dropna above
        table.add_player(Player(name, pos, team, sid))

    # Ensure mapping values are plain Python ints
    merge_map: Dict[str, int] = dict(zip(df["merge_key"], df["sleeper_id"].astype(int)))
    return table, merge_map
