import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cpp_core import ProjectionTable, Triangular, FantasyPointsProjection, AgentConfig


# Ensure project package import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package.services.forecast_adapter import (
    default_position_map,
    build_draft_state,
    compute_vorp_prices,
    run_simulation,
)


def _make_projection_table(ids):
    tbl = ProjectionTable()
    for sid in ids:
        # Deterministic tiny distribution
        dist = Triangular(10.0, 10.0, 10.0)
        proj = FantasyPointsProjection(dist, 2025, 0)
        tbl.set(int(sid), proj)
    return tbl


def test_compute_vorp_prices_basic():
    # 3 players, one drafted at $20, two undrafted with VORP 5 and 15
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "C"],
            "Position": ["RB", "RB", "RB"],
            "PPW": [10.0, 9.0, 8.0],
            "VORP": [0.0, 5.0, 15.0],
            "sleeper_id": [1, 2, 3],
            "Drafted": [1, 0, 0],
            "Draft$": [20, 0, 0],
        }
    )
    teams, roster_size = 2, 2
    my_budget, league_budget = 100, 100
    prices = compute_vorp_prices(df, teams, roster_size, my_budget, league_budget)
    # Drafted keeps realized price
    assert prices[0] >= 20 - 1e-6
    # Undrafted prices are proportional to VORP
    # total remaining excess dollars = 200 - n_left (3 players, 2*2=4 spots -> n_left=?)
    players_rostered = 1
    n_left = teams * roster_size - players_rostered
    remaining_excess = my_budget + league_budget - n_left
    remaining_value = df.loc[(df.Drafted == 0) & (df.VORP > 0), "VORP"].sum()
    edpv = remaining_excess / remaining_value
    assert np.isclose(prices[1], max(1.0, 5.0 * edpv))
    assert np.isclose(prices[2], max(1.0, 15.0 * edpv))


def test_take_conditioning_assigns_focal_player():
    pos_map = default_position_map()
    # 1 player, 2 teams
    df = pd.DataFrame(
        {
            "Name": ["A"],
            "Position": ["RB"],
            "PPW": [10.0],
            "VORP": [5.0],
            "sleeper_id": [11],
            "Drafted": [-1],
            "Draft$": [0],
        }
    )
    # Budgets and needs: 2 teams, 1 RB slot each
    team_budgets = [100, 100]
    needs = [[0] * (max(pos_map.values()) + 1) for _ in range(2)]
    needs[0][pos_map["RB"]] = 1
    needs[1][pos_map["RB"]] = 1
    ds = build_draft_state(df, pos_map, team_budgets, needs)

    player_ids = df["sleeper_id"].astype(int).tolist()
    pos_idx = np.array([pos_map["RB"]], dtype=np.int32)
    site = np.array([10.0], dtype=float)
    hist = np.array([10.0], dtype=float)
    vorp = np.array([10.0], dtype=float)
    tbl = _make_projection_table(player_ids)

    my_id = 0
    price = 7
    out = run_simulation(
        tbl,
        player_ids,
        pos_idx,
        site,
        hist,
        vorp,
        ds,
        n_sims=1,
        seed=0,
        my_roster_id=my_id,
        take=True,
        focal_sid=player_ids[0],
        focal_price=price,
    )
    # Final owner and price match conditioning
    assert int(out.final_owner[0]) == my_id
    assert int(out.final_price[0]) == price


def test_auction_winner_and_price_monotonic():
    # 1 player, 2 teams; set agent 0 cap low, agent 1 cap high
    pos_map = default_position_map()
    df = pd.DataFrame(
        {
            "Name": ["A"],
            "Position": ["RB"],
            "PPW": [10.0],
            "VORP": [5.0],
            "sleeper_id": [21],
            "Drafted": [-1],
            "Draft$": [0],
        }
    )
    team_budgets = [100, 100]
    needs = [[0] * (max(pos_map.values()) + 1) for _ in range(2)]
    needs[0][pos_map["RB"]] = 1
    needs[1][pos_map["RB"]] = 1
    ds = build_draft_state(df, pos_map, team_budgets, needs)

    player_ids = df["sleeper_id"].astype(int).tolist()
    pos_idx = np.array([pos_map["RB"]], dtype=np.int32)
    # Use site prices to control caps; agent 0 aggressiveness=1, agent 1=2
    site = np.array([12.0], dtype=float)
    hist = np.array([0.0], dtype=float)
    vorp = np.array([0.0], dtype=float)
    tbl = _make_projection_table(player_ids)

    cfg0 = AgentConfig()
    cfg0.w_site = 1.0
    cfg0.w_hist = 0.0
    cfg0.w_vorp = 0.0
    cfg0.aggressiveness = 1.0
    cfg0.noise_sigma = 0.0

    cfg1 = AgentConfig()
    cfg1.w_site = 1.0
    cfg1.w_hist = 0.0
    cfg1.w_vorp = 0.0
    cfg1.aggressiveness = 2.0
    cfg1.noise_sigma = 0.0

    out = run_simulation(
        tbl,
        player_ids,
        pos_idx,
        site,
        hist,
        vorp,
        ds,
        n_sims=1,
        seed=0,
        agent_cfgs=[cfg0, cfg1],
    )

    # Winner is the higher-cap agent (index 1)
    assert int(out.final_owner[0]) == 1
    # Price equals lower cap. Scarcity scales values by need/avg_need (clamped to [0.7,1.5]).
    # With needs only at RB=1 and 10 total positions, avg_need=0.1 -> multiplier=1.5.
    expected_lower_cap = int(site[0] * 1.5 * cfg0.aggressiveness)
    assert int(out.final_price[0]) == expected_lower_cap
