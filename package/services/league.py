from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .draft_sync import DraftPickUpdate
from .sleeper import fetch_league_rosters, fetch_league_users


@dataclass
class RosterMeta:
    roster_id: int
    user_id: str
    display_name: str
    team_name: Optional[str]


class SleeperLeagueState:
    """
    Tracks Sleeper league metadata and each roster's acquired players.

    This class is deliberately UI-agnostic; it stores a minimal state that can be
    consumed by views. Players are tracked by Sleeper player ID and purchase
    amount, keyed by `roster_id`.
    """

    def __init__(self, league_id: str, draft_id: str):
        self._league_id = league_id
        self._draft_id = draft_id
        self.rosters: Dict[int, RosterMeta] = {}
        # roster_id -> list of (sleeper_id, amount)
        self.team_players: Dict[int, List[Tuple[int, int]]] = {}
        # user_id -> roster_id mapping for quick lookup via picked_by
        self.user_to_roster: Dict[str, int] = {}

    def initialize(self) -> None:
        users = fetch_league_users(self._league_id) or []
        rosters = fetch_league_rosters(self._league_id) or []
        user_by_id: Dict[str, Any] = {str(u.get("user_id")): u for u in users}
        for r in rosters:
            try:
                roster_id = int(r.get("roster_id"))
                uid = str(r.get("owner_id"))
            except Exception:
                continue
            u = user_by_id.get(uid, {})
            display = u.get("display_name") or u.get("username") or uid
            team_name = None
            metadata = (
                (u.get("metadata") or {}) if isinstance(u.get("metadata"), dict) else {}
            )
            team_name = metadata.get("team_name")
            self.rosters[roster_id] = RosterMeta(
                roster_id=roster_id,
                user_id=uid,
                display_name=display,
                team_name=team_name,
            )
            self.team_players.setdefault(roster_id, [])
            self.user_to_roster[uid] = roster_id

    def apply_updates(self, updates: List[DraftPickUpdate]) -> None:
        for u in updates:
            # Prefer mapping via picked_by -> user_id -> roster_id
            roster_id: Optional[int] = None
            if u.picked_by:
                roster_id = self.user_to_roster.get(str(u.picked_by))
            if roster_id is None:
                roster_id = u.roster_id if isinstance(u.roster_id, int) else None
            if roster_id is None:
                continue
            roster_list = self.team_players.setdefault(roster_id, [])
            # avoid duplicates by sleeper_id
            if any(pid == u.sleeper_id for pid, _ in roster_list):
                continue
            roster_list.append((u.sleeper_id, u.amount))

    def get_roster_players(self, roster_id: int) -> List[Tuple[int, int]]:
        return list(self.team_players.get(roster_id, []))

    def all_rosters(self) -> Dict[int, RosterMeta]:
        return dict(self.rosters)

    def export_team_players(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Compact structure for nanobind transfer to C++: {roster_id: [(sid, amt), ...]}
        """
        return {rid: list(players) for rid, players in self.team_players.items()}
