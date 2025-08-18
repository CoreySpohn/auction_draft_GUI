from pathlib import Path
from typing import List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cache_dir() -> Path:
    p = Path(".cache")
    ensure_dir(p)
    return p


def draft_states_dir() -> Path:
    p = cache_dir() / "draft_states"
    ensure_dir(p)
    return p


def list_csv_files(directory: Path) -> List[Path]:
    return [x for x in directory.glob("*.csv") if x.is_file()]

