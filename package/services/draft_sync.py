from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .sleeper import fetch_draft_picks


@dataclass(frozen=True)
class DraftPickUpdate:
    """
    Lightweight representation of a Sleeper draft pick relevant for syncing.

    Attributes:
        sleeper_id:
            Sleeper player ID (as an integer when possible). If missing, -1.
        amount:
            Winning bid amount for the pick. If missing/unknown, 0.
        picked_by:
            Sleeper user_id who made the pick; stable across seasons.
        roster_id:
            Sleeper roster_id (int) for the team, when provided.
        key:
            A deduplication key for this pick (pick_no, player_id) when available.
    """

    sleeper_id: int
    amount: int
    picked_by: str
    roster_id: Optional[int]
    key: Tuple[int, int]


class SleeperDraftSync:
    """
    Incremental poller for Sleeper draft picks, with simple de-duplication.

    This class holds a set of previously seen picks and only emits newly seen
    picks on each poll. It is intentionally stateless w.r.t. the caller's
    application state (dataframe/UI), keeping responsibilities separated.
    """

    def __init__(self, draft_id: str):
        self._draft_id = draft_id
        self._seen: Set[Tuple[int, int]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def poll_updates(self) -> List[DraftPickUpdate]:
        """
        Fetch the current Sleeper picks and return only new ones since last poll.

        Returns:
            A list of DraftPickUpdate objects for newly observed picks.
        """
        raw = fetch_draft_picks(self._draft_id)
        if not raw:
            return []

        new_updates: List[DraftPickUpdate] = []
        for pick in raw:
            try:
                meta: Dict[str, Any] = pick.get("metadata", {}) or {}
                player_id_any = pick.get("player_id", meta.get("player_id"))
                if player_id_any is None:
                    continue
                player_id = int(player_id_any)
                pick_no = int(pick.get("pick_no") or 0)
                amount = int(meta.get("amount") or 0)
                picked_by = str(pick.get("picked_by") or "")
                roster_any = pick.get("roster_id")
                roster_id = int(roster_any) if roster_any is not None else None
                key = (pick_no, player_id)
            except Exception:
                continue

            if key in self._seen:
                continue

            self._seen.add(key)
            new_updates.append(
                DraftPickUpdate(
                    sleeper_id=player_id,
                    amount=amount,
                    picked_by=picked_by,
                    roster_id=roster_id,
                    key=key,
                )
            )

        return new_updates
