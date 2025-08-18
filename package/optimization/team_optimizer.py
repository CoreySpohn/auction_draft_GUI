import copy
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


def create_team_df(
    player_names: List[str],
    league_roster: Dict[str, int],
    draft_df: pd.DataFrame,
    team_cols: Dict[str, Any],
    sport: str,
) -> pd.DataFrame:
    """
    Create a team DataFrame from a list of player names according to the league roster.

    This function mirrors the legacy behavior from the monolithic UI file while being
    independent from any GUI state.

    Args:
        player_names: List of player names to include on the team in priority order.
        league_roster: Mapping from position to required counts.
        draft_df: Full draft state DataFrame containing player rows.
        team_cols: Column schema for the returned team table.
        sport: Either "nfl" or "nba".

    Returns:
        Team DataFrame with rows for each roster slot and player information filled in.
    """
    if sport == "nfl":
        relevant_positions = ["QB", "RB", "WR", "TE", "FLEX", "B"]
        relevant_roster_size = sum([league_roster[pos] for pos in relevant_positions])
        league_positions: List[str] = []
        league_position_numbers: List[int] = []
        for pos, n_required in league_roster.items():
            if pos not in relevant_positions:
                continue
            for npos in range(1, n_required + 1):
                league_positions.append(pos)
                league_position_numbers.append(npos)
        team_dict: Dict[str, Any] = {}
        for col, col_type in team_cols.items():
            if col == "Position":
                team_dict[col] = league_positions
            elif col == "Position rank":
                team_dict[col] = league_position_numbers
            elif col_type is str:
                team_dict[col] = np.repeat("", relevant_roster_size)
            elif col_type is int:
                team_dict[col] = np.repeat(0, relevant_roster_size)
            elif col_type is float:
                team_dict[col] = np.repeat(0.0, relevant_roster_size)

        team_df = pd.DataFrame.from_dict(team_dict)
        player_df = copy.deepcopy(draft_df.loc[draft_df["Name"].isin(player_names)])
        player_df["Position rank"] = player_df.groupby("Position")["PPW"].rank(
            ascending=False
        )
        # Determine flex and bench
        player_df["FLEX_eligible"] = np.zeros(player_df.shape[0], dtype=bool)
        flex_positions = ["RB", "WR", "TE"]
        for pos in flex_positions:
            flex_mask = (player_df["Position"] == pos) & (
                player_df["Position rank"] > league_roster[pos]
            )
            pos_flex = player_df.loc[flex_mask]

            if not pos_flex.empty:
                player_df.loc[flex_mask, "FLEX_eligible"] = np.repeat(
                    True, pos_flex.shape[0]
                )
        player_df.loc[player_df["FLEX_eligible"], "FLEX rank"] = player_df.loc[
            player_df["FLEX_eligible"], "PPW"
        ].rank(ascending=False)
        used_players: List[str] = []
        for pos, rank in zip(league_positions, league_position_numbers):
            if pos == "FLEX":
                player = player_df.loc[(player_df["FLEX rank"] == rank)]
            else:
                player = player_df.loc[
                    (player_df.Position == pos)
                    & (player_df["Position rank"] == rank)
                ]
            if not player.empty:
                for col in team_cols.keys():
                    if col in ["Position", "Position rank"]:
                        continue
                    team_df.loc[
                        (team_df.Position == pos) & (team_df["Position rank"] == rank),
                        col,
                    ] = player[col].values[0]
                used_players.append(player["Name"].values[0])
        # Add bench players
        remaining_players = [name for name in player_names if name not in used_players]
        if len(remaining_players) > 0:
            for i, player_name in enumerate(remaining_players):
                bench_row = len(used_players) + i
                player = player_df.loc[(player_df.Name == player_name)]
                for col in team_cols.keys():
                    if col in ["Position", "Position rank"]:
                        continue
                    team_df.loc[bench_row, col] = player[col].values[0]
    elif sport == "nba":
        relevant_positions = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "B"]
        relevant_roster_size = sum([league_roster[pos] for pos in relevant_positions])
        league_positions: List[str] = []
        league_position_numbers: List[int] = []
        for pos, n_required in league_roster.items():
            if pos not in relevant_positions:
                continue
            for npos in range(1, n_required + 1):
                league_positions.append(pos)
                league_position_numbers.append(npos)
        team_dict = {}
        for col, col_type in team_cols.items():
            if col == "Position":
                team_dict[col] = league_positions
            elif col == "Position rank":
                team_dict[col] = league_position_numbers
            elif col_type is str:
                team_dict[col] = np.repeat("", relevant_roster_size)
            elif col_type is int:
                team_dict[col] = np.repeat(0, relevant_roster_size)
            elif col_type is float:
                team_dict[col] = np.repeat(0.0, relevant_roster_size)

        team_df = pd.DataFrame.from_dict(team_dict)
        team_df["Eligible"] = np.repeat(0.0, relevant_roster_size)
        player_df = copy.deepcopy(draft_df.loc[draft_df["Name"].isin(player_names)])
        player_df["PPW rank"] = player_df["PPW"].rank(ascending=False)
        # Rank players into their spots
        player_df["n_positions"] = np.zeros(player_df.shape[0])
        for pos in relevant_positions:
            player_df[f"{pos}_eligible"] = np.zeros(player_df.shape[0])
        for i, player in player_df.iterrows():
            pos = player["Position"]
            PG = True if "PG" in pos else False
            SG = True if "SG" in pos else False
            SF = True if "SF" in pos else False
            PF = True if "PF" in pos else False
            C = True if "C" in pos else False
            player_df.at[i, "n_positions"] = PG + SG + SF + PF + C
            player_df.at[i, "PG_eligible"] = PG
            player_df.at[i, "SG_eligible"] = SG
            player_df.at[i, "SF_eligible"] = SF
            player_df.at[i, "PF_eligible"] = PF
            player_df.at[i, "C_eligible"] = C
            player_df.at[i, "G_eligible"] = PG or SG
            player_df.at[i, "F_eligible"] = SF or PF
            player_df.at[i, "UTIL_eligible"] = 1
            player_df.at[i, "B_eligible"] = 1
        used_players = []
        for pos, rank in zip(league_positions, league_position_numbers):
            remaining_players = [name for name in player_names if name not in used_players]
            remaining_player_df = player_df[player_df["Name"].isin(remaining_players)]
            # Check if there are any players who have eligibility for only the
            # current position
            if pos in ["PG", "SG", "SF", "PF", "C"]:
                eligible_players = remaining_player_df.loc[
                    (remaining_player_df.n_positions == 1)
                    & (remaining_player_df[f"{pos}_eligible"] == 1)
                ]
                if len(eligible_players) == 0:
                    eligible_players = remaining_player_df.loc[
                        (remaining_player_df.n_positions >= 1)
                        & (remaining_player_df[f"{pos}_eligible"] == 1)
                    ]
            elif pos in ["G", "F"]:
                eligible_players = remaining_player_df.loc[
                    (remaining_player_df.n_positions == 2)
                    & (remaining_player_df[f"{pos}_eligible"] == 1)
                ]
                if len(eligible_players) == 0:
                    eligible_players = remaining_player_df.loc[
                        (remaining_player_df.n_positions >= 2)
                        & (remaining_player_df[f"{pos}_eligible"] == 1)
                    ]
            else:
                eligible_players = remaining_player_df

            if len(eligible_players) > 0:
                player = eligible_players.iloc[0]
                for col in team_cols.keys():
                    if col in ["Position", "Position rank"]:
                        continue
                    elif col == "Eligible":
                        team_df.loc[
                            (team_df.Position == pos)
                            & (team_df["Position rank"] == rank),
                            col,
                        ] = player["Position"]
                    else:
                        team_df.loc[
                            (team_df.Position == pos)
                            & (team_df["Position rank"] == rank),
                            col,
                        ] = player[col]
                used_players.append(player["Name"])
    return team_df


def find_optimal_team_sat(
    draft_state_df: pd.DataFrame,
    league_roster: Dict[str, int],
    auction_budget: int,
    starter_percent: float,
    team_cols: Dict[str, Any],
    sport: str,
) -> Optional[pd.DataFrame]:
    """
    Build and solve a CP-SAT model to select an optimal team under constraints.

    This is extracted from the legacy UI file without behavior changes.
    """
    starter_budget = int(auction_budget * starter_percent)

    model = cp_model.CpModel()

    # Replace draft board with only undrafted or my players
    draft_state_df = draft_state_df.loc[draft_state_df["Drafted"] != 1].reset_index(
        drop=True
    )

    if sport == "nfl":
        # Set up starter/bench booleans
        starter_bools = []
        bench_bools = []
        for i, player in draft_state_df.iterrows():
            starter_bool = model.NewBoolVar(f"{player.Name} is starter")
            starter_bools.append(starter_bool)

            bench_bool = model.NewBoolVar(f"{player.Name} is on bench")
            bench_bools.append(bench_bool)

            model.Add(starter_bool + bench_bool <= 1)

        n_team = (
            league_roster["QB"]
            + league_roster["RB"]
            + league_roster["WR"]
            + league_roster["TE"]
            + league_roster["FLEX"]
            + league_roster["B"]
        )
        model.Add(sum(starter_bools) + sum(bench_bools) == n_team)
        # Positional constraints
        nf = league_roster["FLEX"]
        n_flex = (
            league_roster["RB"]
            + league_roster["WR"]
            + league_roster["TE"]
            + league_roster["FLEX"]
        )

        qb_target_terms = []
        rb_target_terms = []
        wr_target_terms = []
        te_target_terms = []
        flex_target_terms = []
        bench_target_terms = []
        qb_terms = []
        rb_terms = []
        wr_terms = []
        te_terms = []
        flex_terms = []
        bench_terms = []
        for i, player in draft_state_df.iterrows():
            pos = player["Position"]
            if (player["T/A"] == "T") & (player["Drafted"] != 2):
                qb_target_terms.append(starter_bools[i] * int(pos == "QB"))
                rb_target_terms.append(starter_bools[i] * int(pos == "RB"))
                wr_target_terms.append(starter_bools[i] * int(pos == "WR"))
                te_target_terms.append(starter_bools[i] * int(pos == "TE"))
                flex_target_terms.append(
                    starter_bools[i] * int(pos in ["RB", "WR", "TE"])
                )
                bench_target_terms.append(bench_bools[i] * int(pos in ["RB", "WR"]))
            qb_terms.append(starter_bools[i] * int(pos == "QB"))
            rb_terms.append(starter_bools[i] * int(pos == "RB"))
            wr_terms.append(starter_bools[i] * int(pos == "WR"))
            te_terms.append(starter_bools[i] * int(pos == "TE"))
            flex_terms.append(starter_bools[i] * int(pos in ["RB", "WR", "TE"]))
            bench_terms.append(bench_bools[i] * int(pos in ["RB", "WR"]))

        model.Add(sum(bench_terms) == league_roster["B"])
        model.Add(sum(qb_terms) == league_roster["QB"])

        model.Add(sum(rb_terms) >= league_roster["RB"])
        model.Add(sum(rb_terms) <= league_roster["RB"] + nf)

        model.Add(sum(wr_terms) >= league_roster["WR"])
        model.Add(sum(wr_terms) <= league_roster["WR"] + nf)

        model.Add(sum(te_terms) >= league_roster["TE"])
        model.Add(sum(te_terms) <= league_roster["TE"] + nf)

        model.Add(sum(flex_terms) == n_flex)

        model.Add(sum(bench_terms) == league_roster["B"])

        # On team and price constraints
        starter_price_terms = []
        bench_price_terms = []
        for i, player in draft_state_df.iterrows():
            if player["Drafted"] == 2:
                starter_price_terms.append(
                    starter_bools[i] * int(round(player["Draft$"], 0))
                )
                bench_price_terms.append(
                    bench_bools[i] * int(round(player["Draft$"], 0))
                )
                model.Add(starter_bools[i] + bench_bools[i] == 1)
            elif player["T/A"] == "A":
                model.Add(starter_bools[i] + bench_bools[i] == 0)
            else:
                starter_price_terms.append(
                    starter_bools[i] * int(round(player["Proj$"], 0))
                )
                bench_price_terms.append(
                    bench_bools[i] * int(round(player["Proj$"], 0))
                )
        model.Add(sum(starter_price_terms) <= starter_budget)
        model.Add(sum(starter_price_terms) + sum(bench_price_terms) <= auction_budget)

        starter_values = []
        bench_values = []
        for i, player in draft_state_df.iterrows():
            starter_values.append(starter_bools[i] * int(1000 * player.PPW))
            bench_values.append(bench_bools[i] * int(1000 * player.PPW))
        model.Maximize(sum(starter_values) + 0.01 * sum(bench_values))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            starters: List[str] = []
            bench: List[str] = []
            for i, player in draft_state_df.iterrows():
                if solver.Value(starter_bools[i]):
                    starters.append(player["Name"])
                elif solver.Value(bench_bools[i]):
                    bench.append(player["Name"])
            starters.extend(bench)
            opt_team = create_team_df(
                starters, league_roster, draft_state_df, team_cols, sport
            )
        else:
            opt_team = None
    elif sport == "nba":
        n_team = (
            league_roster["PG"]
            + league_roster["SG"]
            + league_roster["SF"]
            + league_roster["PF"]
            + league_roster["C"]
            + league_roster["G"]
            + league_roster["F"]
            + league_roster["UTIL"]
            + league_roster["B"]
        )
        player_vars = []
        for i, player in draft_state_df.iterrows():
            player_var = model.NewBoolVar(f"{player.Name}")
            player_vars.append(player_var)

        model.Add(sum(player_vars) == n_team)

        # Positional constraints
        pg_target_terms = []
        sg_target_terms = []
        sf_target_terms = []
        pf_target_terms = []
        c_target_terms = []
        pg_terms = []
        sg_terms = []
        sf_terms = []
        pf_terms = []
        c_terms = []
        g_terms = []
        f_terms = []
        for i, player in draft_state_df.iterrows():
            pos = player["Position"]
            PG = True if "PG" in pos else False
            SG = True if "SG" in pos else False
            SF = True if "SF" in pos else False
            PF = True if "PF" in pos else False
            C = True if "C" in pos else False
            G = True if (PG or SG) else False
            F = True if (SF or PF) else False
            if (player["T/A"] == "T") & (player["Drafted"] != 2):
                pg_target_terms.append(player_vars[i] * PG)
                sg_target_terms.append(player_vars[i] * SG)
                sf_target_terms.append(player_vars[i] * SF)
                pf_target_terms.append(player_vars[i] * PF)
                c_target_terms.append(player_vars[i] * C)
            pg_terms.append(player_vars[i] * PG)
            sg_terms.append(player_vars[i] * SG)
            sf_terms.append(player_vars[i] * SF)
            pf_terms.append(player_vars[i] * PF)
            c_terms.append(player_vars[i] * C)
            g_terms.append(player_vars[i] * G)
            f_terms.append(player_vars[i] * F)

        model.Add(sum(pg_terms) >= league_roster["PG"])
        model.Add(sum(sg_terms) >= league_roster["SG"])
        model.Add(sum(sf_terms) >= league_roster["SF"])
        model.Add(sum(pf_terms) >= league_roster["PF"])
        model.Add(sum(c_terms) >= league_roster["C"] + 2)
        model.Add(
            sum(g_terms)
            >= league_roster["G"] + league_roster["PG"] + league_roster["SG"]
        )
        model.Add(
            sum(f_terms)
            >= league_roster["F"] + league_roster["PF"] + league_roster["SF"]
        )

        # On team and price constraints
        player_price_terms = []
        for i, player in draft_state_df.iterrows():
            if player["Drafted"] == 2:
                player_price_terms.append(
                    player_vars[i] * int(round(player["Draft$"], 0))
                )
                model.Add(player_vars[i] == 1)
            elif player["T/A"] == "A":
                model.Add(player_vars[i] == 0)
                player_price_terms.append(player_vars[i] * (auction_budget + 1))
            else:
                player_price_terms.append(
                    player_vars[i] * max(int(round(player["Proj$"], 0)), 1)
                )
        model.Add(sum(player_price_terms) <= auction_budget)

        player_values = []
        for i, player in draft_state_df.iterrows():
            player_values.append(player_vars[i] * int(1000 * player.PPW))
        model.Maximize(sum(player_values))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            players: List[str] = []
            for i, player in draft_state_df.iterrows():
                if solver.Value(player_vars[i]):
                    players.append(player["Name"])
            opt_team = create_team_df(
                players, league_roster, draft_state_df, team_cols, sport
            )
        else:
            opt_team = None

    return opt_team

