import copy
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def calc_player_oc(
    player_name: str,
    test_price: float,
    draft_df: pd.DataFrame,
    league_roster: Dict[str, int],
    auction_budget: int,
    starter_percent: float,
    team_cols: Dict[str, Any],
    sport: str,
) -> Optional[float]:
    """
    Compute the opportunity cost of drafting or not drafting a player at a test price.

    This mirrors previous behavior, delegating the team building to the CP-SAT optimizer.
    """
    from package.optimization.team_optimizer import find_optimal_team_sat

    player_ind = draft_df.loc[draft_df["Name"] == player_name].index

    test_df = copy.deepcopy(draft_df)
    # Take scenario
    test_df.loc[player_ind, "Drafted"] = 2
    test_df.loc[player_ind, "Draft$"] = test_price
    opt_take_team = find_optimal_team_sat(
        test_df, league_roster, auction_budget, starter_percent, team_cols, sport
    )
    if opt_take_team is None:
        return 100
    opt_take_ppw = opt_take_team["PPW"].sum()

    # Leave scenario
    test_df.loc[player_ind, "Drafted"] = 1
    test_df.loc[player_ind, "Draft$"] = test_price
    opt_leave_team = find_optimal_team_sat(
        test_df, league_roster, auction_budget, starter_percent, team_cols, sport
    )
    if opt_leave_team is None:
        return 100
    opt_leave_ppw = opt_leave_team["PPW"].sum()
    opportunity_cost = opt_leave_ppw - opt_take_ppw
    return float(opportunity_cost)

