import copy
import datetime
import pickle
import time
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from moepy import lowess
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtWidgets as qtw
from scipy.optimize import root_scalar
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.tree import DecisionTreeRegressor

from package.ui.mainwindow_ui import Ui_Form


class QCustomTableWidgetItem(qtw.QTableWidgetItem):
    def __init__(self, value):
        super(QCustomTableWidgetItem, self).__init__(str(value))

    def __lt__(self, other):
        if isinstance(other, QCustomTableWidgetItem):
            selfDataValue = float(self.data(qtc.Qt.EditRole))
            otherDataValue = float(other.data(qtc.Qt.EditRole))
            return selfDataValue < otherDataValue
        else:
            return qtw.QTableWidgetItem.__lt__(self, other)


class ReadOnlyDelegate(qtw.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return


class MainWindow(qtw.QWidget):  # Would be something else if you didn't use widget above
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Set up the plot
        self.oc_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.oc_ax = self.oc_canvas.figure.subplots()
        self.ui.gridLayout.addWidget(self.oc_canvas, 1, 0, 2, 1)

        self.p_canvas = FigureCanvas(Figure(figsize=(5, 5)))

        # self.ui.set.addWidget(self.oc_canvas, 1, 0, 1, 1)

        self.sport = "nba"
        if self.sport == "nfl":
            # Set up position cost plots
            self.pax = self.p_canvas.figure.subplots(ncols=2, nrows=2)
            self.ui.gridLayout.addWidget(self.p_canvas, 4, 0, 2, 1)
            self.teams = 10
            self.auction_budget = 300
            self.starter_percent = 0.97
            self.bench_percent = 1 - self.starter_percent

            setup = "1QB"
            self.te_premium = True
            self.scoring_coeffs = {
                "PASSING_ATT": 0,
                "PASSING_CMP": 0,
                "PASSING_INC": 0,
                "PASSING_YDS": 0.04,
                "PASSING_TDS": 4,
                "PASSING_INTS": -2,
                "RUSHING_ATT": 0,
                "RUSHING_YDS": 0.1,
                "RUSHING_TDS": 6,
                "MISC_FL": -2,
                "RECEIVING_REC": 1,
                "RECEIVING_YDS": 0.1,
                "RECEIVING_TDS": 6,
            }
            if setup == "1QB":
                self.league_roster = {
                    "QB": 1,
                    "RB": 2,
                    "WR": 3,
                    "TE": 1,
                    "FLEX": 2,
                    "K": 1,
                    "DEF": 1,
                    "B": 6,
                }
                self.prev_year = 2021
                self.site_file = ".cache/sleeper_1QB.csv"
            elif setup == "2QB":
                self.league_roster = {
                    "QB": 2,
                    "RB": 2,
                    "WR": 3,
                    "TE": 1,
                    "FLEX": 1,
                    "K": 1,
                    "DEF": 1,
                    "B": 6,
                }
                self.prev_year = 2022
                self.site_file = ".cache/sleeper_2QB.csv"
            self.relevant_starters = ["QB", "RB", "WR", "TE", "FLEX"]
            self.flex_positions = ["RB", "WR", "TE"]
            # "QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2}

            # Taken from elboberto's spreadsheet
            # self.league_baselines = {
            #     "QB": [10, 20],
            #     "RB": [30, 47],
            #     "WR": [40, 64],
            #     "TE": [10, 17],
            # }
            self.league_baselines = {"QB": [], "RB": [], "WR": [], "TE": []}
            self.single_eligible_positions = ["QB", "RB", "WR", "TE"]
            self.league_teams = 10
            self.flex_positions = ["RB", "WR", "TE"]
            # pos_bench_depth =
            self.relevant_positions = ["QB", "RB", "WR", "TE", "FLEX", "B"]
        elif self.sport == "nba":
            self.pax = self.p_canvas.figure.subplots(ncols=1, nrows=1)
            self.ui.gridLayout.addWidget(self.p_canvas, 4, 0, 2, 1)
            self.teams = 10
            self.auction_budget = 200
            self.league_roster = {
                "PG": 1,
                "SG": 1,
                "SF": 1,
                "PF": 1,
                "C": 1,
                "G": 1,
                "F": 1,
                "UTIL": 3,
                "B": 3,
            }
            self.relevant_positions = [
                "PG",
                "SG",
                "SF",
                "PF",
                "C",
                "G",
                "F",
                "UTIL",
                "B",
            ]
            self.single_eligible_positions = ["PG", "SG", "SF", "PF", "C"]
            self.relevant_starters = [
                "PG",
                "SG",
                "SF",
                "PF",
                "C",
                "G",
                "F",
                "UTIL",
                "B",
            ]
            self.prev_year = 2022
            self.site_file = Path(".cache/espn_from_hbball.csv")
            self.n_players = (
                self.league_roster["PG"]
                + self.league_roster["SG"]
                + self.league_roster["SF"]
                + self.league_roster["PF"]
                + self.league_roster["C"]
                + self.league_roster["G"]
                + self.league_roster["F"]
                + self.league_roster["UTIL"]
                + self.league_roster["B"]
            )
        self.league_starters = {
            pos: self.league_roster[pos] for pos in self.relevant_starters
        }

        # Set up the QB1, QB2, RB1, ...
        self.league_positions = []
        self.league_position_numbers = []
        for pos, n_required in self.league_roster.items():
            if pos not in self.relevant_positions:
                continue
            for npos in range(1, n_required + 1):
                self.league_positions.append(pos)
                self.league_position_numbers.append(npos)
        self.draft_board_display_cols = {
            "Name": {"is_editable": False, "dtype": str},
            "Position": {"is_editable": False, "dtype": str},
            "PPG": {"is_editable": False, "dtype": float},
            "T/A": {"is_editable": True, "dtype": str},
            "Rank": {"is_editable": False, "dtype": int},
            "Site": {"is_editable": False, "dtype": float},
            "Proj$": {"is_editable": False, "dtype": float},
            "OC": {"is_editable": False, "dtype": float},
            "Draft$": {"is_editable": True, "dtype": int},
            "Drafted": {"is_editable": True, "dtype": int},
        }
        if self.sport == "nfl":
            self.team_cols = {
                "Position": str,
                "Position rank": int,
                "Name": str,
                "T/A": str,
                "Rank": int,
                "PPW": float,
                # "AV": float,
                "Proj$": float,
                "Draft$": int,
            }
            self.team_table_cols = {
                "Position": str,
                # "Position rank": int,
                "Name": str,
                "T/A": str,
                "Rank": int,
                "PPW": float,
                # "AV": float,
                "Proj$": float,
                "Draft$": int,
            }
        elif self.sport == "nba":
            self.team_cols = {
                "Position": str,
                "Position rank": int,
                "Eligible": str,
                "Name": str,
                "T/A": str,
                "Rank": int,
                "PPG": float,
                "PPW": float,
                # "AV": float,
                "Proj$": float,
                "Draft$": int,
            }
            self.team_table_cols = {
                "Position": str,
                "Eligible": str,
                "Name": str,
                "T/A": str,
                "Rank": int,
                "PPG": float,
                "PPW": float,
                # "AV": float,
                "Proj$": float,
                "Draft$": int,
            }

        if self.sport == "nfl":
            self.available_budget = self.teams * (
                self.auction_budget
                - self.league_roster["K"]
                - self.league_roster["DEF"]
            )
            self.n_league_starters = 11
            self.bench_spots = 6
            self.roster_size = 17
        elif self.sport == "nba":
            self.available_budget = self.teams * self.auction_budget
            self.n_league_starters = 10
            self.bench_spots = 3
            self.roster_size = 13

        self.relevant_roster_size = sum(
            [self.league_roster[pos] for pos in self.relevant_positions]
        )
        self.n_relevant_starters = sum(
            [self.league_roster[pos] for pos in self.relevant_starters]
        )

        self.my_budget = copy.deepcopy(self.auction_budget)
        if self.sport == "nfl":
            self.my_starter_budget = self.starter_percent * (
                self.auction_budget
                - self.league_roster["K"]
                - self.league_roster["DEF"]
            )
        elif self.sport == "nba":
            # Everyone starts in NBA at some point in a matchup
            self.starter_percent = 1
            self.bench_percent = 0
            self.my_starter_budget = self.auction_budget
        # League is everyone but me
        self.league_budget = (self.teams - 1) * self.auction_budget

        self.my_excess_budget = self.my_budget - self.roster_size
        self.league_excess_budget = (
            self.league_budget - (self.teams - 1) * self.roster_size
        )

        self.my_rostered_players = 0
        self.league_rostered_players = 0

        self.create_colors()
        if self.sport == "nfl":
            self.get_nfldata()
            self.calc_fantasy_points()
            self.calc_player_vorp()
            self.add_extra_cols()
            self.get_historic_price_data()
            self.add_site_prices()
            self.calc_player_price()
        elif self.sport == "nba":
            self.get_nba_ppg()
            self.calc_player_vorp()
            self.add_extra_cols()
            self.get_historic_price_data()
            self.add_site_prices()
            self.calc_player_price()
        self.init_draft_board()
        # self.update_draft_board()
        self.init_team_table(self.ui.myTeamTable)
        self.init_team_table(self.ui.optTeamTable)
        # self.update_my_team()
        self.update_opt_team()
        self.update_top_ocs()

        self.ui.draftBoard.setSortingEnabled(True)
        self.ui.draftBoard.selectionModel().selectionChanged.connect(
            self.on_selectionChanged
        )
        self.ui.draftBoard.cellChanged.connect(self.on_draftBoardChanged)
        self.ui.exportButton.clicked.connect(self.export_draft)
        self.ui.importButton.clicked.connect(self.import_draft)

        # Find the optimal team after the draft
        # self.import_draft()
        # newdf = copy.deepcopy(self.draft_df)
        # newdf["Proj$"] = newdf["Draft$"]
        # newdf["Drafted"] = np.zeros(newdf.shape[0])
        # newdf["Draft$"] = np.zeros(newdf.shape[0])
        # opt_team = find_optimal_team_sat(
        #     newdf,
        #     self.league_roster,
        #     self.auction_budget,
        #     self.starter_percent,
        #     self.team_cols,
        # )
        # print(opt_team)
        # print(opt_team.loc[opt_team["Position"] != "B", "Proj$"].sum())
        # print(opt_team.loc[opt_team["Position"] != "B", "PPW"].sum())

    def export_draft(self):
        self.draft_df.to_csv(
            f".cache/draft_states/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
        )

    def import_draft(self):
        base_path = Path(".cache/draft_states")
        files = [x for x in base_path.glob("*.csv") if x.is_file()]
        dates = []
        for file in files:
            date = datetime.datetime.strptime(file.stem, "%Y_%m_%d_%H_%M_%S")
            dates.append(date)
        most_recent_draft_ind = np.argmax(dates)
        self.draft_df = pd.read_csv(files[most_recent_draft_ind])
        self.draft_df = self.draft_df.fillna({"T/A": ""})
        self.calc_player_price()

        self.ui.draftBoard.cellChanged.disconnect()
        for col in self.draft_board_display_cols.keys():
            if col in ["Draft$", "Drafted"]:
                continue
            # self.ui.draftBoard.cellChanged.disconnect()
            self.update_draft_board_column(col)
            # self.ui.draftBoard.cellChanged.connect(self.on_draftBoardChanged)
        self.update_my_team()
        self.update_opt_team()
        self.ui.draftBoard.cellChanged.connect(self.on_draftBoardChanged)

    def update_top_ocs(self):
        # Get top undrafted players at their positions
        if self.sport == "nfl":
            for pos in self.single_eligible_positions:
                pos_inds = self.draft_df.loc[
                    (self.draft_df["Position"] == pos) & (self.draft_df["Drafted"] == 0)
                ].index
                top_pos_inds = pos_inds[:5]
                names = self.draft_df.loc[top_pos_inds, "Name"]
                prices = self.draft_df.loc[top_pos_inds, "Proj$"]
                func_args = zip(
                    names,
                    prices,
                    repeat(self.draft_df),
                    repeat(self.league_roster),
                    repeat(self.auction_budget),
                    repeat(self.starter_percent),
                    repeat(self.team_cols),
                    repeat(self.sport),
                )
                with Pool(processes=10) as pool:
                    _opp_costs = pool.starmap(calc_player_oc, func_args)

                ocs = []
                for oc in _opp_costs:
                    if oc is None:
                        ocs.append(0)
                    else:
                        ocs.append(oc)
                self.draft_df.loc[top_pos_inds, "OC"] = ocs
        elif self.sport == "nba":
            remaining_inds = self.draft_df.loc[(self.draft_df["Drafted"] == 0)].index
            top_inds = remaining_inds[:10]
            names = self.draft_df.loc[top_inds, "Name"]
            prices = self.draft_df.loc[top_inds, "Proj$"]
            func_args = zip(
                names,
                prices,
                repeat(self.draft_df),
                repeat(self.league_roster),
                repeat(self.auction_budget),
                repeat(self.starter_percent),
                repeat(self.team_cols),
                repeat(self.sport),
            )
            with Pool(processes=10) as pool:
                _opp_costs = pool.starmap(calc_player_oc, func_args)

            ocs = []
            for oc in _opp_costs:
                if oc is None:
                    ocs.append(0)
                else:
                    ocs.append(oc)
            self.draft_df.loc[top_inds, "OC"] = ocs
        self.update_draft_board_column("OC")

    def create_team_df(self, player_names):
        team_dict = {}
        for col, col_type in self.team_cols.items():
            if col == "Position":
                team_dict[col] = self.league_positions
            elif col == "Position rank":
                team_dict[col] = self.league_position_numbers
            elif col_type == str:
                team_dict[col] = np.repeat("", self.relevant_roster_size)
            elif col_type == int:
                team_dict[col] = np.repeat(0, self.relevant_roster_size)
            elif col_type == float:
                team_dict[col] = np.repeat(0.0, self.relevant_roster_size)

        team_df = pd.DataFrame.from_dict(team_dict)
        player_df = copy.deepcopy(
            self.draft_df.loc[self.draft_df["Name"].isin(player_names)]
        )
        player_df["Position rank"] = player_df.groupby("Position")["PPW"].rank(
            ascending=False
        )
        # Determine flex and bench

        if self.sport == "nfl":
            player_df["FLEX_eligible"] = np.zeros(player_df.shape[0], dtype=bool)
            for pos in self.flex_positions:
                flex_mask = (player_df["Position"] == pos) & (
                    player_df["Position rank"] > self.league_roster[pos]
                )
                pos_flex = player_df.loc[flex_mask]

                if not pos_flex.empty:
                    player_df.loc[flex_mask, "FLEX_eligible"] = np.repeat(
                        True, pos_flex.shape[0]
                    )
        elif self.sport == "nba":
            player_df["FLEX_eligible"] = np.ones(player_df.shape[0])

        player_df.loc[player_df["FLEX_eligible"], "FLEX rank"] = player_df.loc[
            player_df["FLEX_eligible"], "PPW"
        ].rank()
        used_players = []
        for pos, rank in zip(self.league_positions, self.league_position_numbers):
            if pos == "FLEX":
                player = player_df.loc[(player_df["FLEX rank"] == rank)]
            else:
                player = player_df.loc[
                    (player_df.Position == pos) & (player_df["Position rank"] == rank)
                ]
            if not player.empty:
                for col in self.team_cols.keys():
                    if col in ["Position", "Position rank"]:
                        continue
                    team_df.loc[
                        (team_df.Position == pos) & (team_df["Position rank"] == rank),
                        col,
                    ] = player[col].values[0]
                used_players.append(player["Name"].values[0])

        return team_df

    def get_my_team(self):
        my_players = self.draft_df.loc[self.draft_df["Drafted"] == 2]["Name"].values
        my_team = create_team_df(
            my_players, self.league_roster, self.draft_df, self.team_cols, self.sport
        )
        return my_team

    def update_my_team(self):
        my_team = self.get_my_team()
        self.populate_team_table(self.ui.myTeamTable, my_team)
        current_ppw = my_team["PPW"].sum()
        my_money = self.auction_budget - my_team["Draft$"].sum()
        self.ui.myTeamLabel.setText(
            f"My Team: {current_ppw:.1f}, ${my_money} remaining"
        )

    def create_colors(self):
        cmap = plt.get_cmap("viridis")
        if self.sport == "nfl":
            positions = ["QB", "RB", "WR", "TE"]
        elif self.sport == "nba":
            positions = ["PG", "SG", "SF", "PF", "C"]
        self.position_colors = {}
        for position, color_val in zip(positions, np.linspace(0, 0.99, len(positions))):
            color = cmap(color_val)
            # color = np.array(color) * 256
            # color[3] = 1
            self.position_colors[position] = qtg.QColor.fromRgbF(
                color[0], color[1], color[2], 0.25
            )

        # Create target or avoid column colors
        self.ta_colors = {"T": "lime", "A": "lightcoral"}
        self.ta_colors_qt = {}
        for key, val in self.ta_colors.items():
            color = colors.to_rgba(val)
            self.ta_colors_qt[key] = qtg.QColor.fromRgbF(
                color[0], color[1], color[2], 0.25
            )
        self.empty_color = "lightgrey"
        empty = colors.to_rgba(self.empty_color)
        self.empty_color_qt = qtg.QColor.fromRgbF(empty[0], empty[1], empty[2], 0.25)
        self.on_team_color = "royalblue"
        on_team = colors.to_rgba(self.on_team_color)
        self.on_team_color_qt = qtg.QColor.fromRgbF(on_team[0], on_team[1], on_team[2], 0.25)
        self.on_other_team_color = "salmon"
        on_other_team = colors.to_rgba(self.on_other_team_color)
        self.on_other_team_color_qt = qtg.QColor.fromRgbF(on_other_team[0], on_other_team[1], on_other_team[2], 0.25)
        self.wanted_color = "lightgreen"
        wanted = colors.to_rgba(self.wanted_color)
        self.wanted_color_qt = qtg.QColor.fromRgbF(wanted[0], wanted[1], wanted[2], 0.25)
        self.player_status_colors = {"OnOtherTeam":self.on_other_team_color_qt,
                                     "OnTeam": self.on_team_color_qt,
                                    "Wanted":self.wanted_color_qt,
                                     "None":self.empty_color_qt}

    def update_opt_team(self):
        if self.draft_df.loc[self.draft_df["Drafted"] == 2].shape[0] < self.n_players:
            opt_team = find_optimal_team_sat(
                self.draft_df,
                self.league_roster,
                self.auction_budget,
                self.starter_percent,
                self.team_cols,
                self.sport,
            )

            self.populate_team_table(self.ui.optTeamTable, opt_team)
            opt_money = 0
            self.draft_df["OnOptTeam"] = np.zeros(self.draft_df.shape[0])
            if self.sport == "nfl":
                starter_ppw = opt_team.loc[opt_team["Position"] != "B", "PPW"].sum()
            elif self.sport == "nba":
                starter_ppw = opt_team["PPW"].sum()
            for i, player in opt_team.iterrows():
                if not player.Name:
                    continue
                player_ind = np.argwhere(self.draft_df.Name == player.Name)[0]
                player_drafted = self.draft_df.at[int(player_ind), "Drafted"]
                if player_drafted == 2:
                    opt_money += player["Draft$"]
                else:
                    opt_money += int(player["Proj$"])
                    self.highlight_player(player.Name, "lime")
                self.draft_df.loc[player_ind, "OnOptTeam"] = 1
            self.p_canvas.draw()
            self.ui.optTeamLabel.setText(
                f"Optimal team: {starter_ppw:.1f} for ${opt_money:.0f}"
            )
            cols_to_update = ["Name", "Position", "Rank", "Site", "Proj$", "OC"]
            for col in cols_to_update:
                self.update_draft_board_column(col)

    def init_team_table(self, table):
        # Column name and their read-only properties
        table.setRowCount(self.relevant_roster_size)
        table.setColumnCount(len(self.team_table_cols.keys()))
        table.setHorizontalHeaderLabels(self.team_table_cols.keys())
        delegate = ReadOnlyDelegate()
        for i, column in enumerate(self.team_table_cols.values()):
            table.setItemDelegateForColumn(i, delegate)

        empty_team = self.create_team_df([])
        self.populate_team_table(table, empty_team)

    def populate_team_table(self, table, team):
        for i, player in team.iterrows():
            player_position = player["Position"]
            player_draft_price = player["Draft$"]
            if player_draft_price > 0:
                draft_status = "OnTeam"
            elif player["PPW"] > 0:
                draft_status = "Wanted"
            else:
                draft_status = "None"
            for j, col in enumerate(self.team_table_cols.keys()):
                if self.team_table_cols[col] is str:
                    table.setItem(i, j, qtw.QTableWidgetItem(str(player[col])))
                else:
                    table.setItem(i, j, QCustomTableWidgetItem(f"{player[col]:.1f}"))
                # if player_position in self.position_colors.keys():
                if col == "T/A":
                    color = self.ta_colors_qt.get(player["T/A"])
                    if color:
                        table.item(i, j).setBackground(qtg.QColor(color))
                    else:
                        table.item(i, j).setBackground(
                            qtg.QColor(self.empty_color_qt)
                        )
                else:
                    # table.item(i, j).setBackground(
                    #     qtg.QColor(self.position_colors[player_position])
                    # )
                    table.item(i, j).setBackground(
                        qtg.QColor(self.player_status_colors[draft_status])
                    )
                    # table.item(i, j).setBackground(
                    #     qtg.QColor(self.team_position_colors[player_position])
                    # )
        table.resizeColumnsToContents()

    def init_draft_board(self):
        # Column name and their read-only properties
        self.ui.draftBoard.setRowCount(self.draft_df.shape[0])
        self.ui.draftBoard.setColumnCount(len(self.draft_board_display_cols))
        self.ui.draftBoard.setHorizontalHeaderLabels(self.draft_board_display_cols)
        # Making read-only columns
        delegate = ReadOnlyDelegate()
        for i, column in enumerate(self.draft_board_display_cols.values()):
            if not column["is_editable"]:
                self.ui.draftBoard.setItemDelegateForColumn(i, delegate)

        sorted_players = self.draft_df.sort_values("PPW", ascending=False)

        for i, player in sorted_players.iterrows():
            player_position = player["Position"]
            player_draft_status = player["Drafted"]
            if player_draft_status == 2:
                draft_status = "OnTeam"
            elif player_draft_status == 1:
                draft_status = "OnOtherTeam"
            elif player_draft_status == 0 and player["T/A"] == "T":
                draft_status = "Wanted"
            else:
                draft_status = "None"
            if "," in player_position:
                # For NBA players with dual position eligibility
                player_position = player_position.split(",")[0]
            for j, col in enumerate(self.draft_board_display_cols.keys()):
                # if col in str_cols:
                if self.draft_board_display_cols[col]["dtype"] is str:
                    self.ui.draftBoard.setItem(
                        i, j, qtw.QTableWidgetItem(str(player[col]))
                    )
                else:
                    self.ui.draftBoard.setItem(
                        i, j, QCustomTableWidgetItem(f"{player[col]:.1f}")
                    )
                if col == "T/A":
                    color = self.ta_colors_qt.get(player["T/A"])
                    if color:
                        self.ui.draftBoard.item(i, j).setBackground(qtg.QColor(color))
                    else:
                        self.ui.draftBoard.item(i, j).setBackground(
                            qtg.QColor(self.empty_color_qt)
                        )
                else:
                    self.ui.draftBoard.item(i, j).setBackground(
                        qtg.QColor(self.player_status_colors[draft_status])
                    )
        if self.sport == "nfl":
            self.ui.draftBoard.sortByColumn(3, qtc.Qt.DescendingOrder)
        elif self.sport == "nba":
            self.ui.draftBoard.sortByColumn(4, qtc.Qt.DescendingOrder)
        self.ui.draftBoard.resizeColumnsToContents()

    def update_draft_board_column(self, column_to_change):
        player_names = []
        for i in range(self.draft_df.shape[0]):
            player_name = self.ui.draftBoard.item(i, 0).text()
            player_names.append(player_name)

        self.ui.draftBoard.setSortingEnabled(False)
        for i, player_name in enumerate(player_names):
            player_row = self.draft_df.loc[self.draft_df["Name"] == player_name]
            if player_row.empty:
                continue
            player = player_row.squeeze(axis=0)
            player_position = player["Position"]
            player_draft_status = player["Drafted"]
            player_on_opt_team = player["OnOptTeam"]
            if player_draft_status == 2:
                draft_status = "OnTeam"
            elif player_on_opt_team:
                draft_status = "Wanted"
            elif player_draft_status == 1:
                draft_status = "OnOtherTeam"
            else:
                draft_status = "None"

            if "," in player_position:
                # For NBA players with dual position eligibility
                player_position = player_position.split(",")[0]
            for j, col in enumerate(self.draft_board_display_cols.keys()):
                if col != column_to_change:
                    continue
                # if col in str_cols:
                if self.draft_board_display_cols[col]["dtype"] is str:
                    self.ui.draftBoard.setItem(
                        i, j, qtw.QTableWidgetItem(str(player[col]))
                    )
                else:
                    self.ui.draftBoard.setItem(
                        i, j, QCustomTableWidgetItem(f"{player[col]:.1f}")
                    )
                if col == "T/A":
                    color = self.ta_colors_qt.get(player["T/A"])
                    if color:
                        self.ui.draftBoard.item(i, j).setBackground(qtg.QColor(color))
                    else:
                        self.ui.draftBoard.item(i, j).setBackground(
                            qtg.QColor(self.empty_color_qt)
                        )
                else:
                    self.ui.draftBoard.item(i, j).setBackground(
                        qtg.QColor(self.player_status_colors[draft_status])
                    )
        self.ui.draftBoard.setSortingEnabled(True)

    def on_selectionChanged(self, selected, deselected):
        # If name in players fill the selected player thing
        if len(selected.indexes()) == 1:
            ix = selected.indexes()[0]
            if ix.column() == 0:
                player_name = ix.data()
                self.ui.selectedPlayerLabel.setText(
                    f"Currently Up for Auction - {player_name}"
                )
                owned_players = self.draft_df.loc[
                    self.draft_df["Drafted"] > 0, "Name"
                ].values
                if player_name not in owned_players:
                    # self.calc_player_opportunity_cost(player_name)
                    self.plot_player_opportunity_cost(player_name)
                    self.select_player(player_name)

    def on_draftBoardChanged(self, row, column):
        player_name = self.ui.draftBoard.item(row, 0).text()
        changed_text = self.ui.draftBoard.item(row, column).text()
        column_name = list(self.draft_board_display_cols.keys())[column]
        # if column_name in ["Drafted", "Draft$"]:
        if column_name == "Draft$":
            changed_number = float(changed_text)
            player_df_row = np.argwhere(self.draft_df["Name"] == player_name)[0]
            column_name = self.ui.draftBoard.horizontalHeaderItem(column).text()
            self.draft_df.loc[player_df_row, column_name] = changed_number
        elif column_name == "Drafted":
            changed_number = float(changed_text)
            player_df_row = np.argwhere(self.draft_df["Name"] == player_name)[0]
            column_name = self.ui.draftBoard.horizontalHeaderItem(column).text()
            self.draft_df.loc[player_df_row, column_name] = changed_number
            self.calc_player_price()
            self.update_draft_board_column("Proj$")
            self.update_my_team()
            self.update_opt_team()
            self.export_draft()
            # self.update_top_ocs()
        elif column_name == "T/A":
            player_df_row = np.argwhere(self.draft_df["Name"] == player_name)[0]
            column_name = self.ui.draftBoard.horizontalHeaderItem(column).text()
            self.draft_df.loc[player_df_row, column_name] = changed_text
            color = self.ta_colors_qt.get(changed_text)
            if color:
                self.ui.draftBoard.item(row, column).setBackground(qtg.QColor(color))
            else:
                self.ui.draftBoard.item(row, column).setBackground(
                    qtg.QColor(self.empty_color_qt)
                )
            self.calc_player_price()
            self.update_my_team()
            self.update_opt_team()

    def get_nfldata(self):
        loc = Path(f".cache/raw_{datetime.date.today().strftime('%Y-%m-%d')}.csv")
        loc.parent.mkdir(exist_ok=True)
        if loc.exists():
            with open(loc, "rb") as f:
                self.player_stats = pickle.load(f)
        else:
            driver = webdriver.Firefox()
            positions = ["QB", "RB", "WR", "TE"]
            avgdfs = []
            highdfs = []
            lowdfs = []
            url_start = "https://www.fantasypros.com/nfl/projections/"
            url_end = ".php?max-yes=true&min-yes=true&week=draft"
            for position in positions:
                url = f"{url_start}{position.lower()}{url_end}"
                driver.get(url)
                position_data = pd.read_html(driver.page_source)[0]
                relevant_categories = ["PASSING", "RUSHING", "RECEIVING", "MISC"]
                irrelevant_attributes = ["FPTS"]
                avg_stats = {}
                high_stats = {}
                low_stats = {}
                for i, row in position_data.iterrows():
                    player_string = row[0]
                    player_name = ""
                    for i, _string_part in enumerate(player_string.split(" ")[:-2]):
                        if i > 0:
                            player_name += " "
                        player_name += _string_part
                    if player_name not in avg_stats.keys():
                        avg_stats[player_name] = {"POSITION": position}
                        high_stats[player_name] = {"POSITION": position}
                        low_stats[player_name] = {"POSITION": position}
                    for category, attribute in row.index.to_flat_index():
                        stat_name = f"{category}_{attribute}"
                        if (
                            category in relevant_categories
                            and attribute not in irrelevant_attributes
                        ):
                            row_string = row[category][attribute]
                            # The average, high, and low values for the player
                            # The table does not get easily read with pandas
                            _row_strings = ["", "", ""]
                            current_string_ind = 0
                            transition = False
                            for symbol in row_string:
                                if symbol == ",":
                                    continue
                                _row_strings[current_string_ind] += symbol
                                if transition:
                                    current_string_ind += 1
                                    transition = False
                                if symbol == ".":
                                    transition = True
                            avg_stats[player_name][stat_name] = float(_row_strings[0])
                            high_stats[player_name][stat_name] = float(_row_strings[1])
                            low_stats[player_name][stat_name] = float(_row_strings[2])
                avgdf = pd.DataFrame(avg_stats).T
                highdf = pd.DataFrame(high_stats).T
                lowdf = pd.DataFrame(low_stats).T
                if position == "QB":
                    avgdf["PASSING_INC"] = avgdf["PASSING_ATT"] - avgdf["PASSING_CMP"]
                    highdf["PASSING_INC"] = (
                        highdf["PASSING_ATT"] - highdf["PASSING_CMP"]
                    )
                    lowdf["PASSING_INC"] = lowdf["PASSING_ATT"] - lowdf["PASSING_CMP"]
                avgdfs.append(avgdf)
                highdfs.append(highdf)
                lowdfs.append(lowdf)
            _avg = pd.concat(avgdfs).fillna(value=0)
            _avg.insert(0, "NAME", _avg.index)
            _avg = _avg.reset_index(drop=True)

            _high = pd.concat(highdfs).fillna(value=0)
            _high.insert(0, "NAME", _high.index)
            _high = _high.reset_index(drop=True)

            _low = pd.concat(lowdfs).fillna(value=0)
            _low.insert(0, "NAME", _low.index)
            _low = _low.reset_index(drop=True)
            self.player_stats = {"average": _avg, "high": _high, "low": _low}
            driver.close()
            with open(loc, "wb") as f:
                pickle.dump(self.player_stats, f)

    def get_nba_ppg(self):
        loc = Path(f".cache/raw_nba_{datetime.date.today().strftime('%Y-%m-%d')}.csv")
        if loc.exists():
            data = pd.read_csv(loc)
        else:
            driver = webdriver.Firefox()
            driver.get(
                "https://hashtagbasketball.com/fantasy-basketball-points-league-rankings"
            )
            hbb_form = {
                "PTS": ("ContentPlaceHolder1_TextBox1", 1),
                "OREB": ("ContentPlaceHolder1_TextBox2", 0),
                "DREB": ("ContentPlaceHolder1_TextBox3", 0),
                "TREB": ("ContentPlaceHolder1_TextBox4", 1),
                "AST": ("ContentPlaceHolder1_TextBox5", 2),
                "STL": ("ContentPlaceHolder1_TextBox6", 4),
                "BLK": ("ContentPlaceHolder1_TextBox7", 4),
                "3PM": ("ContentPlaceHolder1_TextBox8", 1),
                "3PA": ("ContentPlaceHolder1_TextBox9", 0),
                "MPG": ("ContentPlaceHolder1_TextBox18", 0),
                "FGM": ("ContentPlaceHolder1_TextBox10", 2),
                "FGA": ("ContentPlaceHolder1_TextBox11", -1),
                "FG MISS": ("ContentPlaceHolder1_TextBox19", 0),
                "FTM": ("ContentPlaceHolder1_TextBox12", 1),
                "FTA": ("ContentPlaceHolder1_TextBox13", -1),
                "FT MISS": ("ContentPlaceHolder1_TextBox20", 0),
                "TO": ("ContentPlaceHolder1_TextBox14", -2),
                "DD": ("ContentPlaceHolder1_TextBox15", 0),
                "TD": ("ContentPlaceHolder1_TextBox16", 0),
                "PF": ("ContentPlaceHolder1_TextBox17", 0),
            }
            # Set positions to pull from ESPN
            pos_site = driver.find_element(By.ID, "ContentPlaceHolder1_DDPOSFROM")
            Select(pos_site).select_by_value("3")

            # Fill in all scoring rules
            for (box_id, value) in hbb_form.values():
                element = driver.find_element(By.ID, box_id)
                element.clear()
                element.send_keys(value)
                time.sleep(0.15)

            # Push update button
            button = driver.find_element(By.ID, "ContentPlaceHolder1_Button1")
            button.click()
            time.sleep(2)
            data = pd.read_html(driver.page_source)[2]
            data.to_csv(loc)
        # COULD USE THIS INSTEAD OF ASSUMING 3.5 GAMES FOR EACH PLAYER
        # data["GPW"] = data["GP"].values / 22
        # data["PPW"] = data["TOTAL"] * data["GPW"]
        data["PPW"] = data["TOTAL"] * 3.5
        data["GP"] = data["GP"]
        data = data.rename(columns={"NAME": "Name", "POS": "Position"})
        self.player_stats = data[["Name", "Position", "PPW", "GP"]]

    def calc_fantasy_points(self):
        for statdf in self.player_stats.values():
            statdf["FANTASY_POINTS"] = np.zeros(statdf.shape[0])
            for name, coeff in self.scoring_coeffs.items():
                statdf["FANTASY_POINTS"] += statdf[name] * coeff
            # TE PREMIUM
            if self.te_premium:
                te_mask = statdf["POSITION"] == "TE"
                statdf.loc[te_mask, "FANTASY_POINTS"] += (
                    0.5 * statdf.loc[te_mask, "RECEIVING_REC"]
                )
            # Points per week
            statdf["PPW"] = statdf["FANTASY_POINTS"] / 17
            statdf["PlayerStatus"] = np.repeat("DND", statdf.shape[0])
            statdf["Rank"] = statdf.groupby("POSITION")["PPW"].rank(ascending=False)

    def calc_player_vorp(self):
        if self.sport == "nfl":
            # Taken from elboberto's spreadsheet
            _df = self.player_stats["average"]
            vorp_bench_diffs = 0
            fdf = _df.loc[_df.POSITION.isin(self.flex_positions)].sort_values(
                "PPW", ascending=False
            )
            # Cut TE after number 20
            fdf = fdf.drop(
                fdf.loc[(fdf["POSITION"] == "TE") & (fdf["Rank"] > 17)].index
            )
            # QB baselines
            starting_QBs = self.league_roster["QB"] * self.league_teams
            self.league_baselines["QB"].append(starting_QBs)
            bench_QBs = min(starting_QBs + 1 * self.league_teams, 32) - starting_QBs
            self.league_baselines["QB"].append(starting_QBs + bench_QBs)

            fdf["FLEX_RANK"] = fdf["PPW"].rank(ascending=False)

            # Flex baselines
            starting_RBs = self.league_roster["RB"] * self.league_teams
            starting_WRs = self.league_roster["WR"] * self.league_teams
            starting_TEs = self.league_roster["TE"] * self.league_teams
            flex_start_point = starting_RBs + starting_WRs + starting_TEs
            flex_end_point = (
                flex_start_point + self.league_roster["FLEX"] * self.league_teams
            )
            flex_starter_df = fdf.loc[
                (fdf["FLEX_RANK"] > flex_start_point)
                & (fdf["FLEX_RANK"] <= flex_end_point)
            ]
            flex_start_counts = flex_starter_df.POSITION.value_counts()
            starting_RBs += flex_start_counts["RB"]
            starting_WRs += flex_start_counts["WR"]
            starting_TEs += flex_start_counts["TE"]
            bench_end_ind = (
                self.relevant_roster_size * self.league_teams - starting_QBs - bench_QBs
            )
            flex_bench_df = fdf.loc[
                (fdf["FLEX_RANK"] > flex_end_point)
                & (fdf["FLEX_RANK"] <= bench_end_ind)
            ]
            flex_bench_counts = flex_bench_df.POSITION.value_counts()
            bench_RBs = flex_bench_counts["RB"]
            bench_WRs = flex_bench_counts["WR"]
            bench_TEs = flex_bench_counts["TE"]
            self.league_baselines["RB"].append(starting_RBs)
            self.league_baselines["RB"].append(starting_RBs + bench_RBs)
            self.league_baselines["WR"].append(starting_WRs)
            self.league_baselines["WR"].append(starting_WRs + bench_WRs)
            self.league_baselines["TE"].append(starting_TEs)
            self.league_baselines["TE"].append(starting_TEs + bench_TEs)

            for pos, (
                starter_baseline,
                bench_baseline,
            ) in self.league_baselines.items():
                starter_baseline_ind = starter_baseline - 1
                bench_baseline_ind = bench_baseline - 1
                pos_mask = _df.POSITION == pos
                pos_df = _df.loc[pos_mask]
                starter_pos_replacement = pos_df.iloc[starter_baseline_ind]
                bench_pos_replacement = pos_df.iloc[bench_baseline_ind]
                starter_inds = (
                    _df.loc[(_df.POSITION == pos)].iloc[:starter_baseline_ind].index
                )
                _df.loc[starter_inds, "PlayerStatus"] = "Starter"
                bench_inds = (
                    _df.loc[(_df.POSITION == pos)]
                    .iloc[starter_baseline_ind:bench_baseline_ind]
                    .index
                )
                _df.loc[bench_inds, "PlayerStatus"] = "Bench"
                # starter_vorp = pos_df["PPW"].values - starter_pos_replacement["PPW"]
                # bench_vorp = pos_df["PPW"].values - bench_pos_replacement["PPW"]
                starter_vorp = (
                    pos_df["FANTASY_POINTS"].values
                    - starter_pos_replacement["FANTASY_POINTS"]
                )
                bench_vorp = (
                    pos_df["FANTASY_POINTS"].values
                    - bench_pos_replacement["FANTASY_POINTS"]
                )
                _df.loc[pos_mask, "starter_VORP"] = starter_vorp
                _df.loc[pos_mask, "bench_VORP"] = bench_vorp

                # Not totally sure why this is used?
                vorp_bench_diffs += (bench_vorp[0] - starter_vorp[0]) * starter_baseline
                drop_inds = _df.loc[pos_mask & (_df["bench_VORP"] < -5)].index
                _df = _df.drop(drop_inds)

            starter_mask = (_df["PlayerStatus"] == "Starter") & (
                _df["starter_VORP"] > 0
            )
            summed_starter_value = _df.loc[starter_mask, "starter_VORP"].sum()
            bench_mask = (_df["PlayerStatus"] == "Bench") & (_df["bench_VORP"] > 0)
            summed_bench_value = _df.loc[bench_mask, "bench_VORP"].sum()

            self.bench_budget = self.available_budget * self.bench_percent
            bench_pf = self.bench_budget / summed_bench_value
            self.starter_budget = self.available_budget - vorp_bench_diffs * bench_pf
            starter_pf = self.starter_budget / summed_starter_value
            _df["VORP"] = (_df["bench_VORP"] - _df["starter_VORP"]) * bench_pf + _df[
                "starter_VORP"
            ] * starter_pf
            _df = _df.reset_index(drop=True)

            # Setting up the draft dataframe
            # self.draft_df = copy.deepcopy(_df[["NAME", "POSITION", "PPW", "VORP"]])
            self.draft_df = copy.deepcopy(
                _df[["NAME", "POSITION", "PPW", "Rank", "VORP"]]
            )
            # self.draft_df[['NAME', 'POSITION', 'VORP']]
            self.draft_df = self.draft_df.rename(
                columns={"NAME": "Name", "POSITION": "Position"}
            )
        elif self.sport == "nba":
            _df = self.player_stats
            replacement_player = _df.iloc[self.teams * self.roster_size]

            vorp = _df["PPW"].values - replacement_player["PPW"]
            _df["VORP"] = vorp
            drop_inds = _df.loc[_df["VORP"] < -15].index
            _df = _df.drop(drop_inds)
            _df = _df.reset_index(drop=True)

            # Setting up the draft dataframe
            self.draft_df = copy.deepcopy(
                _df[["Name", "Position", "PPW", "VORP", "GP"]]
            )
            self.draft_df['PPG'] = self.draft_df['PPW'] / 3.5
            self.draft_df["Rank"] = self.draft_df["PPW"].rank(ascending=False)
            self.draft_df["Auction value"] = np.zeros(self.draft_df.shape[0])
            self.draft_df["Drafted"] = np.zeros(self.draft_df.shape[0])
            self.draft_df["Draft price"] = np.zeros(self.draft_df.shape[0])

    def add_extra_cols(self):
        self.draft_df["T/A"] = [""] * self.draft_df.shape[0]
        self.draft_df["AV"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["Draft$"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["Drafted"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["OC"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["OnOptTeam"] = np.zeros(self.draft_df.shape[0])

    def get_historic_price_data(self):
        if self.sport == "nfl":
            poss = ["QB", "RB", "WR", "TE"]
            self.draft_df["Proj$"] = np.ones(self.draft_df.shape[0])
            self.league_position_data = {}
            for pos in poss:
                df = pd.read_csv(f".cache/herndon_{self.prev_year}/{pos}s.csv").fillna(
                    1
                )
                x = df["PTS"].values / 17
                y = df["Paid"].values
                total_spend = df.Paid.sum()
                self.league_position_data[pos] = {
                    "hist_spend": total_spend,
                    "hist_ppw": x,
                    "hist_price": y,
                    "hist_rank": df["PTS"].rank(ascending=False).values,
                    "curr_spend": 0,
                    "curr_ppw": None,
                    "curr_price": None,
                }
        elif self.sport == "nba":
            self.draft_df["Projected price"] = np.ones(self.draft_df.shape[0])
            df = pd.read_csv(f".cache/bball/{self.prev_year}.csv").fillna(1)
            x = df["TOTAL"].values * 3.5
            y = df["Paid"].values
            total_spend = df.Paid.sum()
            self.league_hist_data = {
                "hist_spend": total_spend,
                "hist_ppw": x,
                "hist_price": y,
                "hist_rank": df["TOTAL"].rank(ascending=False).values,
                "hist_proj_gp": df["GP"],
                "hist_site": df["SITE"],
                "hist_site_rank": df["SITE"].rank(ascending=False).values,
                "curr_spend": 0,
                "curr_ppw": None,
                "curr_price": None,
            }

    def add_site_prices(self):
        if self.sport == "nfl":
            price_adjustment = self.auction_budget / 300
        elif self.sport == "nba":
            self.get_espn_avg_prices()
            price_adjustment = self.auction_budget / 200
        pdf = pd.read_csv(self.site_file)

        site_prices = []
        for _, row in self.draft_df.iterrows():
            name = row["Name"]
            site_price = pdf.loc[pdf["Name"] == name]["Site"]
            if len(site_price.values) == 0:
                site_prices.append(1)
            else:
                site_prices.append(site_price.values[0] * price_adjustment)
        self.draft_df["Site"] = site_prices
        self.draft_df["Site_rank"] = self.draft_df["Site"].rank(ascending=False)

    def get_espn_avg_prices(self):
        # Get the average prices from ESPN, taken from HashTag Basketball
        if not self.site_file.exists():
            url = "https://hashtagbasketball.com/fantasy-basketball-auction-values"
            driver = webdriver.Firefox()
            driver.get(url)
            data = pd.read_html(driver.page_source)[2]
            data = data.rename(columns={"PLAYER": "Name"})
            data["Site"] = np.array(
                data["ESPN AVG"].str.replace("$", "").values, dtype=float
            )
            data.to_csv(self.site_file)
        # return data
        # self.draft_df['Site'] = np.zeros(self.draft_df.shape[0])
        # for i, player in self.draft_df.iterrows():
        #     # Add the site price from the data dataframe to the
        #     # self.draft_df dataframe based on the "Name" column matching in both
        #     player_site_price = data.loc[data['Name'] == player.Name, 'Site']
        #     if len(player_site_price) == 0:
        #         self.draft_df.at[i, 'Site'] = 0
        #     else:
        #         self.draft_df.at[i, 'Site'] = player_site_price.values[0]
        #     breakpoint()

    def calc_player_price(self):
        if self.sport == "nfl":
            # Calculate the remaining excess dollars
            players_rostered = self.my_rostered_players + self.league_rostered_players
            n_left = self.teams * self.roster_size - players_rostered
            remaining_excess_dollars = self.my_budget + self.league_budget - n_left

            # Get the remaining value in the draft
            value_mask = self.draft_df["Drafted"] == 0
            positive_vorp_mask = self.draft_df["VORP"] > 0
            remaining_value = self.draft_df.loc[
                value_mask & positive_vorp_mask, "VORP"
            ].sum()

            # Excess dollars per VORP
            edpv = remaining_excess_dollars / remaining_value
            av = self.draft_df.loc[value_mask, "VORP"].values * edpv
            av[av <= 0] = 1
            self.draft_df.loc[value_mask, "AV"] = np.round(av)

            poss = ["QB", "RB", "WR", "TE"]
            cmap = plt.get_cmap("viridis")
            position_colors = [cmap(val) for val in np.linspace(0, 0.99, 4)]
            current_draft_excess_price = 0
            for pos, ax, color in zip(poss, self.pax.flatten(), position_colors):
                ax.clear()
                pos_data = self.league_position_data[pos]
                hist_price = pos_data["hist_price"]
                hist_rank = pos_data["hist_rank"]
                pos_inds = self.draft_df.loc[self.draft_df.Position == pos].index
                undrafted = self.draft_df.loc[pos_inds].loc[self.draft_df.Drafted == 0]
                drafted = self.draft_df.loc[pos_inds].loc[self.draft_df.Drafted != 0]

                all_curr_rank = self.draft_df.loc[pos_inds, "Rank"].values
                all_curr_site = self.draft_df.loc[pos_inds, "Site"].values

                # Get site information
                site_price = self.draft_df.loc[
                    self.draft_df["Position"] == pos, "Site"
                ].values
                site_rank = (
                    self.draft_df.loc[self.draft_df["Position"] == pos, "Site"]
                    .rank(method="first", ascending=False)
                    .values
                )
                hist_model = lowess.Lowess()
                curr_model = lowess.Lowess()
                use_hist_point = np.ones(len(hist_rank), dtype=bool)
                use_site_point = np.ones(len(site_rank), dtype=bool)
                hist_chi2 = []
                site_chi2 = []
                if drafted.empty:
                    hist_model.fit(
                        hist_rank,
                        hist_price,
                        frac=0.5,
                    )
                else:
                    # remove values near currently drafted players
                    curr_rank = drafted["Rank"]
                    curr_price = drafted["Draft$"].values
                    cuml_hist_price = []
                    cuml_hist_rank = []
                    cuml_site_price = []
                    cuml_site_rank = []
                    # Add historical values
                    for i, (rank, price) in enumerate(zip(hist_rank, hist_price)):
                        if np.any(np.abs(curr_rank - rank) < 2):
                            use_hist_point[i] = False
                        else:
                            cuml_hist_rank.append(rank)
                            cuml_hist_price.append(price)
                        if np.any(np.abs(curr_rank - rank) == 0):
                            hist_chi2.append(
                                (
                                    curr_price[
                                        np.argwhere(np.abs(curr_rank - rank) == 0)
                                    ]
                                    - price
                                )[0][0]
                                ** 2
                                / price
                            )
                    # Add site values
                    for i, (rank, price) in enumerate(zip(site_rank, site_price)):
                        if np.any(np.abs(curr_rank - rank) < 2):
                            use_site_point[i] = False
                        else:
                            cuml_site_rank.append(rank)
                            cuml_site_price.append(price)
                        if np.any(np.abs(curr_rank - rank) == 0):
                            site_chi2.append(
                                (
                                    curr_price[
                                        np.argwhere(np.abs(curr_rank - rank) == 0)
                                    ]
                                    - price
                                )[0][0]
                                ** 2
                                / price
                            )
                    for rank, price in zip(curr_rank, curr_price):
                        cuml_hist_rank.append(rank)
                        cuml_hist_price.append(price)
                    hist_model.fit(
                        np.array(cuml_hist_rank), np.array(cuml_hist_price), frac=0.5
                    )
                ax.scatter(
                    hist_rank[use_hist_point],
                    hist_price[use_hist_point],
                    color=color,
                    s=7,
                    alpha=0.25,
                    edgecolor="k",
                )
                ax.scatter(
                    hist_rank[~use_hist_point],
                    hist_price[~use_hist_point],
                    color=color,
                    s=3,
                    alpha=0.25,
                    edgecolor="k",
                )
                ax.scatter(
                    all_curr_rank[use_site_point],
                    site_price[use_site_point],
                    color=color,
                    s=7,
                    alpha=0.25,
                    edgecolor="r",
                )
                ax.scatter(
                    all_curr_rank[~use_site_point],
                    site_price[~use_site_point],
                    color=color,
                    s=3,
                    alpha=0.25,
                    edgecolor="r",
                )
                ax.set_ylim([-5, 105])
                ax.set_yticks(np.arange(0, 101, 10))
                if ax.get_subplotspec().is_first_col():
                    ax.set_ylabel("Price")
                if ax.get_subplotspec().is_last_row():
                    ax.set_xlabel("Rank")

                projected_hist_price = np.clip(
                    hist_model.predict(all_curr_rank), 1, None
                )
                projected_site_price = np.clip(all_curr_site, 1, None)
                if len(hist_chi2) > 0:
                    # Weighting based on the inverted chi^2 value of the historical
                    # and site data
                    hist_chi2_inv_sum = 1 / max(sum(hist_chi2), 1e-5)
                    site_chi2_inv_sum = 1 / max(sum(site_chi2), 1e-5)
                    chi2_inv_sum = hist_chi2_inv_sum + site_chi2_inv_sum
                    hist_weight = hist_chi2_inv_sum / chi2_inv_sum
                    site_weight = site_chi2_inv_sum / chi2_inv_sum
                else:
                    hist_weight = 0.5
                    site_weight = 0.5

                external_predictions = (
                    hist_weight * projected_hist_price
                    + site_weight * projected_site_price
                )
                # Go through and use the current prices when appropriate
                use_lowess = drafted.shape[0] >= 7
                use_polyfit = (drafted.shape[0] > 0) & (~use_lowess)
                # Lowess struggles when there are fewer than 7 data points
                if use_lowess:
                    sort_mask = np.argsort(curr_rank.values)
                    curr_model.fit(
                        np.array(curr_rank.values[sort_mask]),
                        np.array(curr_price[sort_mask]),
                        frac=0.5,
                    )
                    sort_mask = np.argsort(all_curr_rank)
                    curr_predictions = np.clip(
                        curr_model.predict(all_curr_rank[sort_mask]), 1, None
                    )
                    # ax.scatter(curr_rank, curr_price, zorder=5)
                    # ax.scatter(all_curr_rank[sort_mask], curr_predictions, zorder=5)
                elif use_polyfit:
                    curr_predictions = np.ones(len(all_curr_rank))
                undrafted_ranks = undrafted["Rank"].values
                closest_rank_range = 3

                projected_price = np.zeros(len(all_curr_rank))
                pos_lb = 1
                pos_ub = max(all_curr_rank)
                for i, rank in enumerate(all_curr_rank):
                    # Check how many drafted players are within the range of the rank
                    lb = rank - closest_rank_range
                    ub = rank + closest_rank_range
                    # Number of drafted players within the range of the undrafted
                    # player's rank
                    if drafted.shape[0] > 0:
                        near_neighbors = (curr_rank >= lb) & (curr_rank <= ub)
                        n_near_neighbors = sum(near_neighbors)
                        max_n = min(pos_ub, ub) - max(pos_lb, lb)
                        neighbor_coverage = n_near_neighbors / max_n
                        if use_lowess:
                            if neighbor_coverage == 1:
                                extern_weight = 0.05
                                curr_weight = 0.95
                            elif neighbor_coverage == 0:
                                extern_weight = 1
                                curr_weight = 0
                            else:
                                extern_weight = 1 - neighbor_coverage
                                curr_weight = neighbor_coverage
                            # projected_price[i] = (
                            #     curr_weight * curr_predictions[i]
                            #     + extern_weight * external_predictions[i]
                            # )
                        elif use_polyfit:
                            near_neighbor_ranks = curr_rank[near_neighbors]
                            near_neighbor_prices = curr_price[near_neighbors]
                            if (
                                np.any(near_neighbor_ranks < rank)
                                and np.any(near_neighbor_ranks > rank)
                                and neighbor_coverage > 0.5
                            ):
                                # if neighbor_coverage > 0.5:
                                # If there are points on either side then use the polynomial fit
                                z = np.polyfit(
                                    near_neighbor_ranks, near_neighbor_prices, 1
                                )
                                p = np.poly1d(z)
                                curr_predictions[i] = p(rank)
                                extern_weight = 1 - neighbor_coverage
                                curr_weight = neighbor_coverage
                            else:
                                extern_weight = 1
                                curr_weight = 0
                        projected_price[i] = (
                            curr_weight * curr_predictions[i]
                            + extern_weight * external_predictions[i]
                        )
                    else:
                        projected_price[i] = external_predictions[i]

                ax.set_title(f"{pos}")
                current_draft_excess_price += sum(projected_price) - len(
                    projected_price
                )
                self.draft_df.loc[pos_inds, "Proj$"] = projected_price
                # ud_projected_hist_price = np.clip(
                #     hist_model.predict(undrafted["Rank"].values), 1, None
                # )
                # ud_projected_site_price = np.clip(undrafted["Site"].values, 1, None)
                # ud_projected_prices = (
                #     hist_weight * ud_projected_hist_price
                #     + site_weight * ud_projected_site_price
                # )
                ud_inds = np.array(undrafted.index)
                ud_inds = ud_inds - min(ud_inds)
                ud_projected_prices = projected_price[ud_inds]
                # if pos == "TE":
                #     breakpoint()
                # drafted_colors = drafted[]
                if not drafted.empty:
                    other_drafted = drafted.loc[drafted["Drafted"] == 1]
                    my_drafted = drafted.loc[drafted["Drafted"] == 2]
                    other_dcolors = [
                        self.ta_colors.get(ta) for ta in other_drafted["T/A"].values
                    ]
                    other_dcolors = [
                        c if c is not None else self.empty_color for c in other_dcolors
                    ]
                    ax.scatter(
                        other_drafted["Rank"],
                        other_drafted["Draft$"].values,
                        c=other_dcolors,
                        marker="X",
                        edgecolor="k",
                        alpha=0.5,
                    )
                    my_drafted = drafted.loc[drafted["Drafted"] == 2]
                    my_dcolors = [
                        self.ta_colors.get(ta) for ta in my_drafted["T/A"].values
                    ]
                    my_dcolors = [
                        c if c is not None else self.empty_color for c in my_dcolors
                    ]
                    ax.scatter(
                        my_drafted["Rank"],
                        my_drafted["Draft$"].values,
                        c=my_dcolors,
                        marker="*",
                        s=40,
                        edgecolor=my_dcolors[0],
                        # alpha=0.5,
                    )
                    # for rank, price, color in zip(
                    #     my_drafted["Rank"].values,
                    #     my_drafted["Draft$"].values,
                    #     my_dcolors,
                    # ):
                    #     ax.axvline(
                    #         rank,
                    #         c=my_dcolors[0],
                    #         zorder=0
                    #         # price,
                    #         # marker="s",
                    #         # s=25,
                    #         # edgecolor="k",
                    #         # alpha=0.5,
                    #     )
                udcolors = [self.ta_colors.get(ta) for ta in undrafted["T/A"].values]
                udcolors = [c if c is not None else self.empty_color for c in udcolors]
                ax.scatter(
                    undrafted["Rank"],
                    ud_projected_prices,
                    s=15,
                    facecolors=udcolors,
                    edgecolors="k",
                )
            total_spent_budget = self.draft_df["Draft$"].sum()
            figtitle = (
                f"\${total_spent_budget:.0f} of \${self.available_budget:.0f} spent"
            )
            # if current_draft_excess_price > total_excess_budget:
            #     figtitle += "Expect discounts at end"
            # else:
            #     figtitle += "Players going for discounts now"
            self.p_canvas.figure.suptitle(figtitle)
            # self.p_canvas.figure.tight_layout()
            self.p_canvas.draw()
        elif self.sport == "nba":
            ax = self.pax
            #     # Calculate the remaining excess dollars
            #     players_rostered = self.my_rostered_players + self.league_rostered_players
            #     n_left = self.teams * self.roster_size - players_rostered
            #     remaining_excess_dollars = self.my_budget + self.league_budget - n_left

            #     # Get the remaining value in the draft
            #     value_mask = self.draft_df["Drafted"] == 0
            #     positive_vorp_mask = self.draft_df["VORP"] > 0
            #     remaining_value = self.draft_df.loc[
            #         value_mask & positive_vorp_mask, "VORP"
            #     ].sum()

            #     # Excess dollars per VORP
            #     edpv = remaining_excess_dollars / remaining_value
            #     av = self.draft_df.loc[value_mask, "VORP"].values * edpv
            #     av[av <= 0] = 1
            #     self.draft_df.loc[value_mask, "AV"] = np.round(av)

            #     current_draft_excess_price = 0
            ax.clear()
            ax.set_ylim([-5, 90])
            ax.set_xlim([-5, 150])
            #     cmap = plt.get_cmap("viridis")
            #     color = cmap(0.5)
            #     pos_data = self.league_hist_data
            #     hist_price = pos_data["hist_price"]
            #     hist_rank = pos_data["hist_rank"]
            undrafted = self.draft_df.loc[self.draft_df.Drafted == 0]
            drafted = self.draft_df.loc[self.draft_df.Drafted != 0]

            #     all_curr_rank = self.draft_df["Rank"].values
            #     all_curr_site = self.draft_df["Site"].values

            #     # Get site information
            #     site_price = self.draft_df["Site"].values
            #     site_rank = (
            #         self.draft_df["Site"].rank(method="first", ascending=False).values
            #     )
            #     hist_model = lowess.Lowess()
            #     curr_model = lowess.Lowess()
            #     use_hist_point = np.ones(len(hist_rank), dtype=bool)
            #     use_site_point = np.ones(len(site_rank), dtype=bool)
            #     hist_chi2 = []
            #     site_chi2 = []

            #     if drafted.empty:
            #         hist_model.fit(
            #             hist_rank,
            #             hist_price,
            #             frac=1,
            #         )
            #     else:
            #         # remove values near currently drafted players
            #         curr_rank = drafted["Rank"]
            #         curr_price = drafted["Draft$"].values
            #         cuml_hist_price = []
            #         cuml_hist_rank = []
            #         cuml_site_price = []
            #         cuml_site_rank = []
            #         # Add historical values
            #         for i, (rank, price) in enumerate(zip(hist_rank, hist_price)):
            #             if np.any(np.abs(curr_rank - rank) < 2):
            #                 use_hist_point[i] = False
            #             else:
            #                 cuml_hist_rank.append(rank)
            #                 cuml_hist_price.append(price)
            #             if np.any(np.abs(curr_rank - rank) == 0):
            #                 hist_chi2.append(
            #                     (
            #                         curr_price[np.argwhere(np.abs(curr_rank - rank) == 0)]
            #                         - price
            #                     )[0][0]
            #                     ** 2
            #                     / price
            #                 )
            #         # Add site values
            #         for i, (rank, price) in enumerate(zip(site_rank, site_price)):
            #             if np.any(np.abs(curr_rank - rank) < 2):
            #                 use_site_point[i] = False
            #             else:
            #                 cuml_site_rank.append(rank)
            #                 cuml_site_price.append(price)
            #             if np.any(np.abs(curr_rank - rank) == 0):
            #                 site_chi2.append(
            #                     (
            #                         curr_price[np.argwhere(np.abs(curr_rank - rank) == 0)]
            #                         - price
            #                     )[0][0]
            #                     ** 2
            #                     / price
            #                 )
            #         for rank, price in zip(curr_rank, curr_price):
            #             cuml_hist_rank.append(rank)
            #             cuml_hist_price.append(price)
            #         hist_model.fit(
            #             np.array(cuml_hist_rank), np.array(cuml_hist_price), frac=1
            #         )
            #     ax.scatter(
            #         hist_rank[use_hist_point],
            #         hist_price[use_hist_point],
            #         color=color,
            #         s=7,
            #         alpha=0.25,
            #         edgecolor="k",
            #     )
            #     ax.scatter(
            #         hist_rank[~use_hist_point],
            #         hist_price[~use_hist_point],
            #         color=color,
            #         s=3,
            #         alpha=0.25,
            #         edgecolor="k",
            #     )
            #     ax.scatter(
            #         all_curr_rank[use_site_point],
            #         site_price[use_site_point],
            #         color=color,
            #         s=7,
            #         alpha=0.25,
            #         edgecolor="r",
            #     )
            #     ax.scatter(
            #         all_curr_rank[~use_site_point],
            #         site_price[~use_site_point],
            #         color=color,
            #         s=3,
            #         alpha=0.25,
            #         edgecolor="r",
            #     )
            #     ax.set_ylim([-5, 105])
            #     ax.set_yticks(np.arange(0, 101, 10))
            #     if ax.get_subplotspec().is_first_col():
            #         ax.set_ylabel("Price")
            #     if ax.get_subplotspec().is_last_row():
            #         ax.set_xlabel("Rank")

            #     projected_hist_price = np.clip(hist_model.predict(all_curr_rank), 1, None)
            #     projected_site_price = np.clip(all_curr_site, 1, None)
            #     if len(hist_chi2) > 0:
            #         # Weighting based on the inverted chi^2 value of the historical
            #         # and site data
            #         hist_chi2_inv_sum = 1 / max(sum(hist_chi2), 1e-5)
            #         site_chi2_inv_sum = 1 / max(sum(site_chi2), 1e-5)
            #         chi2_inv_sum = hist_chi2_inv_sum + site_chi2_inv_sum
            #         hist_weight = hist_chi2_inv_sum / chi2_inv_sum
            #         site_weight = site_chi2_inv_sum / chi2_inv_sum
            #     else:
            #         hist_weight = 0.5
            #         site_weight = 0.5

            #     external_predictions = (
            #         hist_weight * projected_hist_price + site_weight * projected_site_price
            #     )
            #     # Go through and use the current prices when appropriate
            #     use_lowess = drafted.shape[0] >= 7
            #     use_polyfit = (drafted.shape[0] > 0) & (~use_lowess)
            #     # Lowess struggles when there are fewer than 7 data points
            #     if use_lowess:
            #         sort_mask = np.argsort(curr_rank.values)
            #         curr_model.fit(
            #             np.array(curr_rank.values[sort_mask]),
            #             np.array(curr_price[sort_mask]),
            #             frac=0.5,
            #         )
            #         sort_mask = np.argsort(all_curr_rank)
            #         curr_predictions = np.clip(
            #             curr_model.predict(all_curr_rank[sort_mask]), 1, None
            #         )
            #         # ax.scatter(curr_rank, curr_price, zorder=5)
            #         # ax.scatter(all_curr_rank[sort_mask], curr_predictions, zorder=5)
            #     elif use_polyfit:
            #         curr_predictions = np.ones(len(all_curr_rank))
            #     undrafted_ranks = undrafted["Rank"].values
            #     closest_rank_range = 3

            #     projected_price = np.zeros(len(all_curr_rank))
            #     pos_lb = 1
            #     pos_ub = max(all_curr_rank)
            #     for i, rank in enumerate(all_curr_rank):
            #         # Check how many drafted players are within the range of the rank
            #         lb = rank - closest_rank_range
            #         ub = rank + closest_rank_range
            #         # Number of drafted players within the range of the undrafted
            #         # player's rank
            #         if drafted.shape[0] > 0:
            #             near_neighbors = (curr_rank >= lb) & (curr_rank <= ub)
            #             n_near_neighbors = sum(near_neighbors)
            #             max_n = min(pos_ub, ub) - max(pos_lb, lb)
            #             neighbor_coverage = n_near_neighbors / max_n
            #             if use_lowess:
            #                 if neighbor_coverage == 1:
            #                     extern_weight = 0.05
            #                     curr_weight = 0.95
            #                 elif neighbor_coverage == 0:
            #                     extern_weight = 1
            #                     curr_weight = 0
            #                 else:
            #                     extern_weight = 1 - neighbor_coverage
            #                     curr_weight = neighbor_coverage
            #                 # projected_price[i] = (
            #                 #     curr_weight * curr_predictions[i]
            #                 #     + extern_weight * external_predictions[i]
            #                 # )
            #             elif use_polyfit:
            #                 near_neighbor_ranks = curr_rank[near_neighbors]
            #                 near_neighbor_prices = curr_price[near_neighbors]
            #                 if (
            #                     np.any(near_neighbor_ranks < rank)
            #                     and np.any(near_neighbor_ranks > rank)
            #                     and neighbor_coverage > 0.5
            #                 ):
            #                     # if neighbor_coverage > 0.5:
            #                     # If there are points on either side then use the polynomial fit
            #                     z = np.polyfit(near_neighbor_ranks, near_neighbor_prices, 1)
            #                     p = np.poly1d(z)
            #                     curr_predictions[i] = p(rank)
            #                     extern_weight = 1 - neighbor_coverage
            #                     curr_weight = neighbor_coverage
            #                 else:
            #                     extern_weight = 1
            #                     curr_weight = 0
            #             projected_price[i] = (
            #                 curr_weight * curr_predictions[i]
            #                 + extern_weight * external_predictions[i]
            #             )
            #         else:
            #             projected_price[i] = external_predictions[i]

            #     # ax.set_title(f"{pos}")
            #     current_draft_excess_price += sum(projected_price) - len(projected_price)
            #     self.draft_df["Proj$"] = projected_price
            #     # ud_projected_hist_price = np.clip(
            #     #     hist_model.predict(undrafted["Rank"].values), 1, None
            #     # )
            #     # ud_projected_site_price = np.clip(undrafted["Site"].values, 1, None)
            #     # ud_projected_prices = (
            #     #     hist_weight * ud_projected_hist_price
            #     #     + site_weight * ud_projected_site_price
            #     # )
            #     ud_inds = np.array(undrafted.index)
            #     # ud_inds = ud_inds - min(ud_inds)
            #     ud_projected_prices = projected_price[ud_inds]
            #     # if pos == "TE":
            #     #     breakpoint()
            #     # drafted_colors = drafted[]

            # TESTING NEW MODEL
            hist_ppw = self.league_hist_data["hist_ppw"]
            hist_ppg_over_avg = (hist_ppw - hist_ppw[130]) / 3.5
            hist_rank = self.league_hist_data["hist_rank"]
            hist_proj_gp = self.league_hist_data["hist_proj_gp"]
            hist_site = self.league_hist_data["hist_site"]
            hist_site_rank = self.league_hist_data["hist_site_rank"]
            X = np.vstack(
                [hist_ppg_over_avg, hist_rank, hist_proj_gp, hist_site, hist_site_rank]
            ).T
            y = self.league_hist_data["hist_price"]
            if self.draft_df["Drafted"].sum() > 0:
                drafted_inds = drafted.index
                # drafted_ppw = self.draft_df.loc[drafted_inds, "PPW"].values
                drafted_ppg_over_avg = (
                    self.draft_df.loc[drafted_inds, "PPW"].values
                    - self.draft_df["PPW"].values[130]
                ) / 3.5
                # drafted_ppg = (drafted_ppw-drafted_ppw[130]) / 3.5
                drafted_rank = self.draft_df.loc[drafted_inds, "Rank"].values
                drafted_proj_gp = self.draft_df.loc[drafted_inds, "GP"].values
                drafted_site = self.draft_df.loc[drafted_inds, "Site"].values
                drafted_site_rank = self.draft_df.loc[drafted_inds, "Site_rank"]
                drafted_X = np.vstack(
                    [
                        drafted_ppg_over_avg,
                        drafted_rank,
                        drafted_proj_gp,
                        drafted_site,
                        drafted_site_rank,
                    ]
                ).T
                X = np.vstack([X, drafted_X])
                drafted_prices = self.draft_df.loc[drafted_inds, "Draft$"].values
                y = np.concatenate([y, drafted_prices])
                # y.extend(drafted_prices)
                # y = np.stack([y, drafted_prices])
            # est = GradientBoostingRegressor(
            #     loss="huber",
            #     n_estimators=100,
            #     learning_rate=0.1,
            #     max_depth=3,
            # )
            est = LassoLarsIC(criterion="bic", max_iter=1000)
            est.fit(X, y)
            undrafted_inds = undrafted.index
            curr_ppw = self.draft_df.loc[undrafted_inds, "PPW"].values
            curr_ppg_over_avg = (curr_ppw - self.draft_df["PPW"].values[130]) / 3.5
            curr_rank = self.draft_df.loc[undrafted_inds, "Rank"].values
            curr_proj_gp = self.draft_df.loc[undrafted_inds, "GP"].values
            curr_site = self.draft_df.loc[undrafted_inds, "Site"].values
            curr_site_rank = self.draft_df.loc[undrafted_inds, "Site_rank"]
            curr_X = np.vstack(
                [curr_ppg_over_avg, curr_rank, curr_proj_gp, curr_site, curr_site_rank]
            ).T
            new_predicted_price = est.predict(curr_X)
            self.draft_df.loc[undrafted_inds, "Proj$"] = new_predicted_price
            # params = est.coef_

            # new_model.fit(hist_rank.reshape(-1, 1), hist_price)
            # Data values: Site price, site rank, expected games played
            # expected ppw

            all_points = []
            labels = {}
            if not drafted.empty:
                other_drafted = drafted.loc[drafted["Drafted"] == 1]
                my_drafted = drafted.loc[drafted["Drafted"] == 2]
                other_dcolors = [
                    self.ta_colors.get(ta) for ta in other_drafted["T/A"].values
                ]
                other_dcolors = [
                    c if c is not None else self.empty_color for c in other_dcolors
                ]
                otherdpoints = ax.scatter(
                    other_drafted["Rank"],
                    other_drafted["Draft$"].values,
                    c=other_dcolors,
                    marker="X",
                    edgecolor="k",
                    alpha=0.25,
                )
                otherdnames = other_drafted["Name"].values
                all_points.append(otherdpoints)
                # all_names.append(otherdnames)
                labels[otherdpoints] = otherdnames
                my_drafted = drafted.loc[drafted["Drafted"] == 2]
                my_dcolors = [self.ta_colors.get(ta) for ta in my_drafted["T/A"].values]
                my_dcolors = [
                    c if c is not None else self.on_team_color for c in my_dcolors
                ]
                mydpoints = ax.scatter(
                    my_drafted["Rank"],
                    my_drafted["Draft$"].values,
                    c=my_dcolors,
                    marker="*",
                    s=40,
                    # edgecolor=my_dcolors[0],
                    # alpha=1,
                )
                mydnames = my_drafted["Name"].values
                all_points.append(mydpoints)
                labels[mydpoints] = mydnames
                # all_names.append(mydnames)
                # for rank, price, color in zip(
                #     my_drafted["Rank"].values, my_drafted["Draft$"].values, my_dcolors
                # ):
                #     ax.axvline(
                #         rank,
                #         c=my_dcolors[0],
                #         zorder=0
                #         # price,
                #         # marker="s",
                #         # s=25,
                #         # edgecolor="k",
                #         # alpha=0.5,
                #     )
            udcolors = [self.ta_colors.get(ta) for ta in undrafted["T/A"].values]
            udcolors = [c if c is not None else self.empty_color for c in udcolors]
            udpoints = ax.scatter(
                self.draft_df.loc[undrafted_inds, "Rank"],
                self.draft_df.loc[undrafted_inds, "Proj$"],
                # ud_projected_prices,
                s=15,
                facecolors=udcolors,
                edgecolors="k",
            )
            udnames = self.draft_df.loc[undrafted_inds, "Name"].values
            # all_names = self.draft_df["Name"].values
            # all_points.append(udpoints)
            # all_names.append(udnames)
            labels[udpoints] = udnames
            cu_hover = mplcursors.cursor(
                ax, hover=2, annotation_kwargs={"backgroundcolor": "k", "color": "w"}
            )
            cu_hover.connect(
                "add",
                lambda sel: sel.annotation.set_text(labels[sel.artist][sel.index]),
            )
            cu_click = mplcursors.cursor(udpoints)
            cu_click.visible = False
            # cu_click.connect(
            #     "add", lambda sel: self.plot_player_opportunity_cost(udnames[sel.index])
            # )
            cu_click.connect(
                "add", lambda sel: self.plot_oc_from_proj_plot(udnames[sel.index])
            )
            total_spent_budget = self.draft_df["Draft$"].sum()
            figtitle = (
                f"\${total_spent_budget:.0f} of \${self.available_budget:.0f} spent"
            )
            # if current_draft_excess_price > total_excess_budget:
            #     figtitle += "Expect discounts at end"
            # else:
            #     figtitle += "Players going for discounts now"
            self.p_canvas.figure.suptitle(figtitle)
            # self.p_canvas.figure.tight_layout()
            self.p_canvas.draw()

    def calc_oc_plot_points(self, player_name):
        player_ind = self.draft_df.loc[self.draft_df["Name"] == player_name].index
        base_price = self.draft_df.loc[player_ind, "Proj$"].values[0]
        min_price = max(1, base_price - 25)
        max_price = base_price + 25
        initial_prices = np.linspace(min_price, max_price, 7, dtype=int)
        initial_opp_costs = self.calc_opp_costs_per_prices(player_name, initial_prices)
        # func_args = zip(
        #     repeat(player_name),
        #     test_prices,
        #     repeat(self.draft_df),
        #     repeat(self.league_roster),
        #     repeat(self.auction_budget),
        #     repeat(self.starter_percent),
        #     repeat(self.team_cols),
        #     repeat(self.sport),
        # )
        # with Pool(processes=7) as pool:
        #     opp_costs = pool.starmap(calc_player_oc, func_args)

        # Check if opp_costs has two different signs
        # if initial_opp_costs is None:

        pos_oc = initial_opp_costs[initial_opp_costs > 0]
        neg_oc = initial_opp_costs[initial_opp_costs < 0]
        # neg_oc = np.any(np.array(initial_opp_costs) < 0)
        if len(pos_oc) != 0 and len(neg_oc) != 0:
            # Test all prices between the two 
            last_neg_price = initial_prices[initial_opp_costs < 0][-1]
            first_pos_price = initial_prices[initial_opp_costs > 0][0]
            new_prices = np.arange(last_neg_price, first_pos_price, 1)
            new_opp_costs = self.calc_opp_costs_per_prices(player_name, new_prices)
            prices = np.concatenate([new_prices, initial_prices])
            opp_costs = np.concatenate([new_opp_costs, initial_opp_costs])
            sorted_inds = np.argsort(prices)
            prices = prices[sorted_inds]
            opp_costs = opp_costs[sorted_inds]
        else:
            prices = initial_prices
            opp_costs = initial_opp_costs

        return opp_costs, prices

    def calc_opp_costs_per_prices(self, player_name, prices):
        func_args = zip(
            repeat(player_name),
            prices,
            repeat(self.draft_df),
            repeat(self.league_roster),
            repeat(self.auction_budget),
            repeat(self.starter_percent),
            repeat(self.team_cols),
            repeat(self.sport),
        )
        with Pool(processes=10) as pool:
            opp_costs = pool.starmap(calc_player_oc, func_args)
        return np.array(opp_costs)

    def select_player(self, player_name):
        ax = self.pax
        if len(ax.patches):
            ax.patches[0].remove()
        player_ind = self.draft_df.loc[
            self.draft_df["Name"] == player_name
        ].index
        player_rank = self.draft_df.loc[player_ind, "Rank"].values[0]
        player_price = self.draft_df.loc[player_ind, "Proj$"].values[0]
        ax.add_patch(
            patches.Rectangle(
                xy=(player_rank - 1.2, player_price - 1.1),
                width=2.4,
                height=2.2,
                facecolor="none",
                edgecolor="r",
                mouseover=True,
            )
        )
        self.p_canvas.draw()

    def highlight_player(self, player_name, color):
        ax = self.pax
        if len(ax.patches):
            ax.patches[0].remove()
        player_ind = self.draft_df.loc[
            self.draft_df["Name"] == player_name
        ].index
        player_rank = self.draft_df.loc[player_ind, "Rank"].values[0]
        player_price = self.draft_df.loc[player_ind, "Proj$"].values[0]
        ax.scatter(player_rank, player_price, marker='o', s=20, edgecolor=color, facecolor='none')
        # ax.add_patch(
        #     patches.Rectangle(
        #         xy=(player_rank - 1.5, player_price - 1.5),
        #         width=3,
        #         height=3,
        #         facecolor="none",
        #         edgecolor="r",
        #         mouseover=True,
        #     )
        # )

    def plot_oc_from_proj_plot(self, player_name):
        ax = self.pax
        if len(ax.patches):
            ax.patches[0].remove()
        self.plot_player_opportunity_cost(player_name)
        self.select_player(player_name)

    def plot_player_opportunity_cost(self, player_name):
        cmap = plt.get_cmap("RdYlGn_r")
        if self.sport == "nfl":
            norm = mpl.colors.Normalize(vmin=-5, vmax=5)
        elif self.sport == "nba":
            norm = mpl.colors.Normalize(vmin=-50, vmax=50)
        _opp_costs, _test_prices = self.calc_oc_plot_points(player_name)

        # Replot to prices
        # self.calc_player_price()

        opp_costs = []
        test_prices = []
        too_expensive = []
        for oc, tp in zip(_opp_costs, _test_prices):
            if oc is not None:
                opp_costs.append(oc)
                test_prices.append(tp)
            else:
                too_expensive.append(tp)
        self.oc_ax.clear()
        if not len(opp_costs):
            self.oc_ax.set_title("YOU HAVE SPENT TOO MUCH")
            self.oc_canvas.draw()
            return 0
        opp_costs = np.array(opp_costs)
        test_prices = np.array(test_prices)
        self.oc_ax.scatter(
            test_prices,
            opp_costs,
            c=opp_costs,
            cmap=cmap,
            norm=norm,
            edgecolor="k",
            zorder=3
        )
        self.oc_ax.set_xlabel("Paid price")
        self.oc_ax.set_ylabel("Opportunity cost")
        self.oc_ax.xaxis.set_major_locator(MultipleLocator(10))
        self.oc_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.oc_ax.grid(which='major', color='#CCCCCC', linestyle='--', zorder=0)
        self.oc_ax.grid(which='minor', color='#CCCCCC', linestyle=':', zorder=0)
        self.oc_ax.axhline(0, color="k", ls="-", zorder=1)
        # self.oc_canvas.draw()
        # Set up linear fit to find the root
        z = np.polyfit(test_prices, opp_costs, 1)
        p = np.poly1d(z)
        # z2 = np.polyfit(test_prices, np.array(opp_costs) + 2, 1)
        # p2 = np.poly1d(z2)
        title = player_name
        # if np.sign(p(min_price)) != np.sign(p(max_price)):
        #     root = root_scalar(p, method="bisect", bracket=[min_price, max_price])
        #     self.ax.axvline(root.root, color="k", ls="--")
        #     title += f" - Draft below ${root.root:.2f}"
        # elif np.sign(p(min_price)) > 0 and np.sign(p(max_price)) > 0:
        #     title += " - Never draft"
        # elif np.sign(p(min_price)) < 0 and np.sign(p(max_price)) < 0:
        #     title += " - Draft for any price"
        # else:
        #     title += " - How is this possible"
        min_price = min(test_prices)
        max_price = max(test_prices)
        player_ind = self.draft_df.loc[self.draft_df["Name"] == player_name].index
        base_price = self.draft_df.loc[player_ind, "Proj$"].values[0]
        pos_oc = opp_costs[opp_costs > 0]
        neg_oc = opp_costs[opp_costs < 0]
        if len(pos_oc) != 0 and len(neg_oc) != 0:
            final_neg_ind = np.where(np.array(opp_costs) < 0)[0][-1]
            title += f" - Draft at or below ${test_prices[final_neg_ind]:.2f}"
        elif len(pos_oc) == len(opp_costs):# np.sign(p(min_price)) > 0 and np.sign(p(max_price)) > 0:
            title += " - Probably not worth it"
        elif len(neg_oc) == len(opp_costs):#np.sign(p(min_price)) < 0 and np.sign(p(max_price)) < 0:
            title += " - Draft for any price"
        else:
            title += " - How is this possible"
        # Plot the projected price
        ibp = int(round(base_price, 0))
        self.oc_ax.scatter(ibp, p(ibp), edgecolor='b', c=p(ibp), zorder=5)
        if len(too_expensive):
            self.oc_ax.axvline(min(too_expensive), color="r", ls="--")
        self.oc_ax.set_title(title)
        if self.sport == "nfl":
            self.oc_ax.set_ylim([-10, 10])
        elif self.sport == "nba":
            self.oc_ax.set_ylim([-100, 100])
        cont_prices = np.linspace(min_price, max_price, 100)
        cont_oc = p(cont_prices)
        self.oc_ax.scatter(
            cont_prices,
            cont_oc,
            c=cont_oc,
            cmap=cmap,
            norm=norm,
            ls="--",
            alpha=0.5,
            zorder=2,
        )
        self.oc_canvas.draw()


def calc_player_oc(
    player_name,
    test_price,
    draft_df,
    league_roster,
    auction_budget,
    starter_percent,
    team_cols,
    sport,
):
    player_ind = draft_df.loc[draft_df["Name"] == player_name].index

    spent_dollars = draft_df.loc[draft_df["Drafted"] == 2, "Draft$"].sum()

    # Get the number of starters remaining at the position
    player_position = draft_df.loc[player_ind, "Position"].values[0]

    test_df = copy.deepcopy(draft_df)
    # Get the optimal team if the player is drafted at the test price
    test_df.loc[player_ind, "Drafted"] = 2
    test_df.loc[player_ind, "Draft$"] = test_price
    opt_take_team = find_optimal_team_sat(
        test_df, league_roster, auction_budget, starter_percent, team_cols, sport
    )
    if opt_take_team is None:
        return 100
    opt_take_ppw = opt_take_team["PPW"].sum()

    test_df.loc[player_ind, "Drafted"] = 1
    test_df.loc[player_ind, "Draft$"] = test_price
    opt_leave_team = find_optimal_team_sat(
        test_df, league_roster, auction_budget, starter_percent, team_cols, sport
    )
    if opt_leave_team is None:
        return 100
    opt_leave_ppw = opt_leave_team["PPW"].sum()
    opportunity_cost = opt_leave_ppw - opt_take_ppw
    return opportunity_cost


# def find_optimal_team(draft_state_df, league_roster, my_starter_budget, team_cols):
#     solver = pywraplp.Solver("Draft", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

#     # Build variables, aka the players we'll be valuing
#     variables = []
#     for i, player in draft_state_df.iterrows():
#         if player["Drafted"] == 1:
#             variables.append(solver.IntVar(0, 0, player.Name))
#         else:
#             variables.append(solver.BoolVar(player.Name))

#     # Build objective
#     objective = solver.Objective()
#     objective.SetMaximization()
#     for i, player in draft_state_df.iterrows():
#         objective.SetCoefficient(variables[i], player.PPW)

#     # Build constraints
#     # Constraint for salary cap
#     salary_cap = solver.Constraint(0, my_starter_budget)
#     for i, player in draft_state_df.iterrows():
#         salary_cap.SetCoefficient(variables[i], player["Proj$"])

#     # Roster size
#     # roster_size = self.n_league_starters - 2
#     # size_cap = solver.Constraint(roster_size, roster_size)
#     # for variable in variables:
#     #     # roster size
#     #     size_cap.SetCoefficient(variable, 1)

#     # Positional constraints
#     nf = league_roster["FLEX"]
#     qb_cap = solver.Constraint(league_roster["QB"], league_roster["QB"])
#     rb_cap = solver.Constraint(league_roster["RB"], league_roster["RB"] + nf)
#     wr_cap = solver.Constraint(league_roster["WR"], league_roster["WR"] + nf)
#     te_cap = solver.Constraint(league_roster["TE"], league_roster["TE"] + nf)
#     n_flex = (
#         league_roster["RB"]
#         + league_roster["WR"]
#         + league_roster["TE"]
#         + league_roster["FLEX"]
#     )
#     flex_cap = solver.Constraint(n_flex, n_flex)
#     for i, player in draft_state_df.iterrows():
#         pos = player["Position"]
#         qb_cap.SetCoefficient(variables[i], int(pos == "QB"))
#         rb_cap.SetCoefficient(variables[i], int(pos == "RB"))
#         wr_cap.SetCoefficient(variables[i], int(pos == "WR"))
#         te_cap.SetCoefficient(variables[i], int(pos == "TE"))
#         flex_cap.SetCoefficient(variables[i], int(pos in ["RB", "WR", "TE"]))

#     # On team constraint
#     players_on_team = len(draft_state_df.loc[draft_state_df["Drafted"] == 2])
#     if players_on_team > 0:
#         players_on_team_constraint = solver.Constraint(
#             int(players_on_team), int(players_on_team)
#         )
#         for i, player in draft_state_df.iterrows():
#             if player["Drafted"] == 2:
#                 players_on_team_constraint.SetCoefficient(variables[i], 1)
#                 salary_cap.SetCoefficient(variables[i], player["Draft$"])
#             else:
#                 players_on_team_constraint.SetCoefficient(variables[i], 0)

#     sol = solver.Solve()
#     if sol == pywraplp.Solver.OPTIMAL:
#         sol_roster = [var.name() for var in variables if var.solution_value()]
#         opt_team = create_team_df(sol_roster, league_roster, draft_state_df, team_cols)
#     else:
#         opt_team = None
#     return opt_team


def find_optimal_team_sat(
    draft_state_df, league_roster, auction_budget, starter_percent, team_cols, sport
):
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
        n_drafted_qb = draft_state_df.loc[
            (draft_state_df["Drafted"] == 2) & (draft_state_df["Position"] == "QB")
        ].shape[0]
        n_drafted_rb = draft_state_df.loc[
            (draft_state_df["Drafted"] == 2) & (draft_state_df["Position"] == "RB")
        ].shape[0]
        n_drafted_wr = draft_state_df.loc[
            (draft_state_df["Drafted"] == 2) & (draft_state_df["Position"] == "WR")
        ].shape[0]
        n_drafted_te = draft_state_df.loc[
            (draft_state_df["Drafted"] == 2) & (draft_state_df["Position"] == "TE")
        ].shape[0]
        n_drafted_flex = draft_state_df.loc[
            (draft_state_df["Drafted"] == 2)
            & (draft_state_df["Position"].isin(["RB", "WR", "TE"]))
        ].shape[0]

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

        # already_on_team = []
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
        # model.Add(sum(already_on_team) == len(already_on_team))

        starter_values = []
        bench_values = []
        for i, player in draft_state_df.iterrows():
            starter_values.append(starter_bools[i] * int(1000 * player.PPW))
            bench_values.append(bench_bools[i] * int(1000 * player.PPW))
        model.Maximize(sum(starter_values) + 0.01 * sum(bench_values))
        solver = cp_model.CpSolver()
        # solver.parameters.num_search_workers = 12
        # solver.parameters.log_search_progress = True
        # solver.parameters.max_time_in_seconds = self.obs_scheme[
        #     "max_time_in_seconds"
        # ]
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            starters = []
            bench = []
            for i, player in draft_state_df.iterrows():
                if solver.Value(starter_bools[i]):
                    starters.append(player["Name"])
                elif solver.Value(bench_bools[i]):
                    bench.append(player["Name"])
            starters.extend(bench)
            opt_team = create_team_df(
                starters, league_roster, draft_state_df, team_cols
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

        # Number of players on team must equal the roster_size
        # minus one (for a streamer player)
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
            # Position booleans
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

        # Add streamer constraint
        # streamer_terms = []
        # for i, player in draft_state_df.iterrows():
        #     streamer_terms.append(player_vars[i] * (player["Proj$"] <= 1))
        # model.Add(sum(streamer_terms) == 1)

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
        # Less than the total budget, leaving 1 dollar for the streamer
        model.Add(sum(player_price_terms) <= auction_budget)

        player_values = []
        for i, player in draft_state_df.iterrows():
            player_values.append(player_vars[i] * int(1000 * player.PPW))
        model.Maximize(sum(player_values))

        solver = cp_model.CpSolver()
        # solver.parameters.num_search_workers = 12
        # solver.parameters.log_search_progress = True
        # solver.parameters.max_time_in_seconds = self.obs_scheme[
        #     "max_time_in_seconds"
        # ]
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            players = []
            for i, player in draft_state_df.iterrows():
                if solver.Value(player_vars[i]):
                    players.append(player["Name"])
            opt_team = create_team_df(
                players, league_roster, draft_state_df, team_cols, sport
            )
        else:
            opt_team = None

    return opt_team


def create_team_df(player_names, league_roster, draft_df, team_cols, sport):
    if sport == "nfl":
        relevant_positions = ["QB", "RB", "WR", "TE", "FLEX", "B"]
        relevant_roster_size = sum([league_roster[pos] for pos in relevant_positions])
        relevant_positions = ["QB", "RB", "WR", "TE", "FLEX", "B"]
        league_positions = []
        league_position_numbers = []
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
            elif col_type == str:
                team_dict[col] = np.repeat("", relevant_roster_size)
            elif col_type == int:
                team_dict[col] = np.repeat(0, relevant_roster_size)
            elif col_type == float:
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
        used_players = []
        for pos, rank in zip(league_positions, league_position_numbers):
            if pos == "FLEX":
                player = player_df.loc[(player_df["FLEX rank"] == rank)]
            else:
                player = player_df.loc[
                    (player_df.Position == pos) & (player_df["Position rank"] == rank)
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
        league_positions = []
        league_position_numbers = []
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
            elif col_type == str:
                team_dict[col] = np.repeat("", relevant_roster_size)
            elif col_type == int:
                team_dict[col] = np.repeat(0, relevant_roster_size)
            elif col_type == float:
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
            # Position booleans
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
        # In the team_df fill in the onesie positions first
        used_players = []
        for pos, rank in zip(league_positions, league_position_numbers):
            remaining_players = [
                name for name in player_names if name not in used_players
            ]
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

            # Highest scoring player with the necessary eligibility
            if len(eligible_players) > 0:
                # if not player.empty:
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
