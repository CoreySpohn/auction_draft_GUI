from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .cache import draft_states_dir, list_csv_files


def save_draft_state(df: pd.DataFrame) -> Path:
    base = draft_states_dir()
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = base / f"{ts}.csv"
    df.to_csv(path, index=False)
    return path


def load_most_recent_draft_state() -> Optional[pd.DataFrame]:
    base = draft_states_dir()
    files = list_csv_files(base)
    if not files:
        return None
    dates = [datetime.datetime.strptime(p.stem, "%Y_%m_%d_%H_%M_%S") for p in files]
    most_recent = files[int(np.argmax(dates))]
    df = pd.read_csv(most_recent)
    return df

