import copy
import datetime
import pickle
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import statsmodels.api as sm
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from moepy import lowess
from ortools.linear_solver import pywraplp
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtWidgets as qtw
from scipy.optimize import root_scalar
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

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
        self.ui.gridLayout.addWidget(self.oc_canvas, 1, 0, 1, 1)

        # Set up position cost plots
        self.p_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.pax = self.p_canvas.figure.subplots(ncols=2, nrows=2)
        self.ui.gridLayout.addWidget(self.p_canvas, 3, 0, 1, 1)

        # self.ui.set.addWidget(self.oc_canvas, 1, 0, 1, 1)

        self.sport = "nba"
        if self.sport == "nfl":
            self.teams = 10
            self.auction_budget = 300
            self.starter_percent = 0.9
            self.bench_percent = 1 - self.starter_percent
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
            self.relevant_positions = ["QB", "RB", "WR", "TE", "FLEX", "B"]
            self.relevant_starters = ["QB", "RB", "WR", "TE", "FLEX"]
            self.flex_positions = ["RB", "WR", "TE"]
        elif self.sport == "nba":
            self.teams = 8
            self.auction_budget = 200
            # DO I NEED A STARTER BUDGET FOR NBA?
            # THE BENCH PLAYERS ALSO START SO...
            self.starter_percent = 0.9
            self.bench_percent = 1 - self.starter_percent
            self.league_roster = {
                "PG": 1,
                "SG": 1,
                "SF": 1,
                "PF": 1,
                "C": 1,
                "G": 1,
                "F": 1,
                "UTIL": 3,
                "B": 4,
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
            self.relevant_starters = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        # Set up the QB1, QB2, RB1, ...
        self.league_positions = []
        self.league_position_numbers = []
        for pos, n_required in self.league_roster.items():
            if pos not in self.relevant_positions:
                continue
            for npos in range(1, n_required + 1):
                self.league_positions.append(pos)
                self.league_position_numbers.append(npos)
        self.draft_board_cols = {
            "Name": {"is_editable": False, "dtype": str},
            # "Position": {"is_editable": False, "dtype": str},
            "PPW": {"is_editable": False, "dtype": float},
            "Projected price": {"is_editable": False, "dtype": float},
            "Auction value": {"is_editable": False, "dtype": float},
            "Draft price": {"is_editable": True, "dtype": int},
            "Drafted": {"is_editable": True, "dtype": int},
        }
        self.team_cols = {
            "Position": str,
            "Position rank": int,
            "Name": str,
            "PPW": float,
            "Projected price": float,
            "Auction value": float,
            "Draft price": int,
        }
        self.team_table_cols = {
            "Position": str,
            # "Position rank": int,
            "Name": str,
            "PPW": float,
            "Projected price": float,
            "Auction value": float,
            "Draft price": int,
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
        self.relevant_starters = sum(
            [self.league_roster[pos] for pos in self.relevant_starters]
        )

        self.my_budget = copy.deepcopy(self.auction_budget)
        if self.sport == "nfl":
            self.my_starter_budget = self.starter_percent * (
                self.auction_budget
                - self.league_roster["K"]
                - self.league_roster["DEF"]
            )
        # League is everyone but me
        self.league_budget = (self.teams - 1) * self.auction_budget

        self.my_excess_budget = self.my_budget - self.roster_size
        self.league_excess_budget = (
            self.league_budget - (self.teams - 1) * self.roster_size
        )

        self.my_rostered_players = 0
        self.league_rostered_players = 0

        if self.sport == "nfl":
            self.get_nfldata()
            self.calc_fantasy_points()
            self.calc_player_vorp()
            self.get_historic_price_data()
            self.calc_player_price()
            self.populate_draft_board()
            self.init_team_table(self.ui.myTeamTable)
            self.init_team_table(self.ui.optTeamTable)
        elif self.sport == "nba":
            self.get_nba_ppg()
            self.calc_player_vorp_nba()
            self.get_historic_price_data_nba()
            self.calc_player_price_nba()
            breakpoint()

        self.ui.draftBoard.setSortingEnabled(True)
        self.ui.draftBoard.selectionModel().selectionChanged.connect(
            self.on_selectionChanged
        )
        self.ui.draftBoard.cellChanged.connect(self.on_draftBoardChanged)

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
        # Add flex and bench players
        # remaining_players = [
        #     name for name in player_df["Name"].values if name not in used_players
        # ]
        # if len(remaining_players) > 0:
        #     remaining_df = player_df.loc[
        #         player_df["Name"].isin(remaining_players)
        #     ].sort_values(by="PPW", ascending=False)
        # for i, player in remaining_df.iterrows():

        return team_df

    def get_my_team(self):
        my_players = self.draft_df.loc[self.draft_df["Drafted"] == 2]["Name"].values
        my_team = self.create_team_df(my_players)
        return my_team

    def update_my_team(self):
        my_team = self.get_my_team()
        self.populate_team_table(self.ui.myTeamTable, my_team)
        current_ppw = my_team["PPW"].sum()
        my_money = self.auction_budget - 2 - my_team["Draft price"].sum()
        self.ui.myTeamLabel.setText(
            f"My Team: {current_ppw:.1f}, ${my_money} remaining"
        )

    def update_opt_team(self):
        opt_team = self.find_optimal_team(self.draft_df)
        self.populate_team_table(self.ui.optTeamTable, opt_team)
        current_ppw = opt_team["PPW"].sum()
        opt_money = 0
        print(opt_team)
        for i, player in opt_team.iterrows():
            if not player.Name:
                continue
            player_ind = np.argwhere(self.draft_df.Name == player.Name)[0]
            # player_ind = self.draft_df.loc[self.draft_df.Name == player.Name].index
            # print(player)
            player_drafted = self.draft_df.at[int(player_ind), "Drafted"]
            print(f"{player.Name} {player_drafted}")
            if player_drafted == 2:
                opt_money += player["Draft price"]
            else:
                opt_money += int(player["Projected price"])
        self.ui.optTeamLabel.setText(
            f"Optimal team: {current_ppw:.1f} for ${opt_money:.0f}"
        )

    def init_team_table(self, table):
        # Column name and their read-only properties
        table.setRowCount(self.relevant_roster_size)
        table.setColumnCount(len(self.team_table_cols.keys()))
        table.setHorizontalHeaderLabels(self.team_table_cols.keys())
        delegate = ReadOnlyDelegate()
        for i, column in enumerate(self.team_table_cols.values()):
            table.setItemDelegateForColumn(i, delegate)

        cmap = plt.get_cmap("viridis")
        positions = ["QB", "RB", "WR", "TE"]
        self.team_position_colors = {}
        for position, color_val in zip(positions, np.linspace(0, 0.99, 4)):
            color = cmap(color_val)
            self.team_position_colors[position] = qtg.QColor.fromRgbF(
                color[0], color[1], color[2], 0.25
            )
        empty_team = self.create_team_df([])
        self.populate_team_table(table, empty_team)

    def populate_team_table(self, table, team):
        str_cols = ["Name", "Position"]
        for i, player in team.iterrows():
            player_position = player["Position"]
            for j, col in enumerate(self.team_table_cols.keys()):
                if col in str_cols:
                    table.setItem(i, j, qtw.QTableWidgetItem(str(player[col])))
                else:
                    table.setItem(i, j, QCustomTableWidgetItem(f"{player[col]:.1f}"))
                if player_position in self.team_position_colors.keys():
                    table.item(i, j).setBackground(
                        qtg.QColor(self.team_position_colors[player_position])
                    )

    def populate_draft_board(self):
        # Column name and their read-only properties
        self.ui.draftBoard.setRowCount(self.draft_df.shape[0])
        # self.ui.draftBoard.setColumnCount(self.player_stats["average"].shape[1])
        self.ui.draftBoard.setColumnCount(len(self.draft_board_cols))
        self.ui.draftBoard.setHorizontalHeaderLabels(self.draft_board_cols)
        # sorted_players = self.player_stats["average"].sort_values(
        #     "Auction value", ascending=False
        # )
        # Making read-only columns
        delegate = ReadOnlyDelegate()
        for i, column in enumerate(self.draft_board_cols.values()):
            if not column["is_editable"]:
                self.ui.draftBoard.setItemDelegateForColumn(i, delegate)

        # sorted_players = self.draft_df.sort_values("Auction value", ascending=False)
        sorted_players = self.draft_df.sort_values("PPW", ascending=False)
        str_cols = ["Name", "Position"]

        cmap = plt.get_cmap("viridis")
        positions = ["QB", "RB", "WR", "TE"]
        position_colors = {}
        for position, color_val in zip(positions, np.linspace(0, 0.99, 4)):
            color = cmap(color_val)
            # color = np.array(color) * 256
            # color[3] = 1
            position_colors[position] = qtg.QColor.fromRgbF(
                color[0], color[1], color[2], 0.25
            )
        # self.ui.draftBoard.itemAt(i, j).setBackground(qtg.QColor('red'))
        for i, player in sorted_players.iterrows():
            player_position = player["Position"]
            for j, col in enumerate(self.draft_board_cols.keys()):
                if col in str_cols:
                    self.ui.draftBoard.setItem(
                        i, j, qtw.QTableWidgetItem(str(player[col]))
                    )
                else:
                    self.ui.draftBoard.setItem(
                        i, j, QCustomTableWidgetItem(f"{player[col]:.1f}")
                    )
                self.ui.draftBoard.item(i, j).setBackground(
                    qtg.QColor(position_colors[player_position])
                )
        self.ui.draftBoard.sortByColumn(1, qtc.Qt.DescendingOrder)

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
                    self.calc_player_opportunity_cost(player_name)

    def on_draftBoardChanged(self, row, column):
        player_name = self.ui.draftBoard.item(row, 0).text()
        changed_text = self.ui.draftBoard.item(row, column).text()
        changed_number = float(changed_text)
        player_df_row = np.argwhere(self.draft_df["Name"] == player_name)[0]
        column_name = self.ui.draftBoard.horizontalHeaderItem(column).text()
        self.draft_df.loc[player_df_row, column_name] = changed_number
        self.calc_player_price()
        # self.ui.draftBoard.cellChanged.disconnect()
        # self.populate_draft_board()
        # self.ui.draftBoard.cellChanged.connect(self.on_draftBoardChanged)
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
                avgdfs.append(pd.DataFrame(avg_stats).T)
                highdfs.append(pd.DataFrame(high_stats).T)
                lowdfs.append(pd.DataFrame(low_stats).T)
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
        data = data.rename(columns={"NAME": "Name", "POS": "Position"})
        self.player_stats = data[["Name", "Position", "PPW"]]

    def calc_fantasy_points(self):
        self.coeffs = {
            "PASSING_ATT": 0,
            "PASSING_CMP": 0,
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
        for statdf in self.player_stats.values():
            statdf["FANTASY_POINTS"] = np.zeros(statdf.shape[0])
            for name, coeff in self.coeffs.items():
                statdf["FANTASY_POINTS"] += statdf[name] * coeff
            # TE PREMIUM
            te_mask = statdf["POSITION"] == "TE"
            statdf.loc[te_mask, "FANTASY_POINTS"] += (
                0.5 * statdf.loc[te_mask, "RECEIVING_REC"]
            )
            # Points per week
            statdf["PPW"] = statdf["FANTASY_POINTS"] / 17
            statdf["PlayerStatus"] = np.repeat("DND", statdf.shape[0])

    def calc_player_vorp_nfl(self):
        # Taken from elboberto's spreadsheet
        self.league_baselines = {
            "QB": [20, 32],
            "RB": [25, 42],
            "WR": [35, 59],
            "TE": [10, 17],
        }
        _df = self.player_stats["average"]
        vorp_bench_diffs = 0
        for pos, (starter_baseline, bench_baseline) in self.league_baselines.items():
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
            drop_inds = _df.loc[pos_mask & (_df["bench_VORP"] < -3)].index
            _df = _df.drop(drop_inds)

        starter_mask = (_df["PlayerStatus"] == "Starter") & (_df["starter_VORP"] > 0)
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
        self.draft_df = copy.deepcopy(_df[["NAME", "POSITION", "PPW", "VORP"]])
        # self.draft_df[['NAME', 'POSITION', 'VORP']]
        self.draft_df = self.draft_df.rename(
            columns={"NAME": "Name", "POSITION": "Position"}
        )
        self.draft_df["Auction value"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["Drafted"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["Draft price"] = np.zeros(self.draft_df.shape[0])

    def calc_player_vorp_nba(self):
        # Make adjustable
        _df = self.player_stats
        # vorp_bench_diffs = 0
        # for pos, (starter_baseline, bench_baseline) in self.league_baselines.items():
        starter_baseline = self.teams * self.n_league_starters
        bench_baseline = starter_baseline + self.teams * self.bench_spots
        starter_baseline_ind = starter_baseline - 1
        bench_baseline_ind = bench_baseline - 1
        starter_inds = _df.iloc[:starter_baseline_ind].index
        _df.loc[starter_inds, "PlayerStatus"] = "Starter"
        bench_inds = _df.iloc[starter_baseline_ind:bench_baseline_ind].index
        _df.loc[bench_inds, "PlayerStatus"] = "Bench"

        starter_baseline_ind = starter_baseline - 1
        bench_baseline_ind = bench_baseline - 1
        starter_replacement = _df.iloc[starter_baseline_ind]
        bench_replacement = _df.iloc[bench_baseline_ind]
        # starter_vorp = pos_df["PPW"].values - starter_pos_replacement["PPW"]
        # bench_vorp = pos_df["PPW"].values - bench_pos_replacement["PPW"]
        starter_vorp = _df["PPW"].values - starter_replacement["PPW"]
        bench_vorp = _df["PPW"].values - bench_replacement["PPW"]
        _df["starter_VORP"] = starter_vorp
        _df["bench_VORP"] = bench_vorp

        # Not totally sure why this is used?
        vorp_bench_diffs = (bench_vorp[0] - starter_vorp[0]) * starter_baseline
        drop_inds = _df.loc[(_df["bench_VORP"] < -10)].index
        _df = _df.drop(drop_inds)

        starter_mask = (_df["PlayerStatus"] == "Starter") & (_df["starter_VORP"] > 0)
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
        self.draft_df = copy.deepcopy(_df[["Name", "Position", "PPW", "VORP"]])
        self.draft_df["Auction value"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["Drafted"] = np.zeros(self.draft_df.shape[0])
        self.draft_df["Draft price"] = np.zeros(self.draft_df.shape[0])

    def get_historic_price_data_nfl(self):
        poss = ["QB", "RB", "WR", "TE"]
        self.draft_df["Projected price"] = np.ones(self.draft_df.shape[0])
        self.league_position_data = {}
        for pos in poss:
            df = pd.read_csv(f".cache/herndon_2022/{pos}s.csv").fillna(1)
            # p_df = df.loc[df["Paid"] > 1]
            # excess_dollar_players = (p_df.Paid > 1).index
            # x = p_df.loc[excess_dollar_players]["PTS"] / 16
            # y = p_df.loc[excess_dollar_players]["Paid"]
            x = df["PTS"].values / 17
            y = df["Paid"].values
            # model = lowess.Lowess()
            # model.fit(x.values, y.values, frac=0.5)
            total_spend = df.Paid.sum()
            # percent_spend = total_spend / self.available_budget
            self.league_position_data[pos] = {
                "hist_spend": total_spend,
                "hist_ppw": x,
                "hist_price": y,
                "curr_spend": 0,
                "curr_ppw": None,
                "curr_price": None,
            }

    def get_historic_price_data_nba(self):
        self.draft_df["Projected price"] = np.ones(self.draft_df.shape[0])
        df = pd.read_csv(".cache/bball/2022.csv").fillna(1)
        x = df["TOTAL"].values * 3.5
        y = df["Paid"].values
        total_spend = df.Paid.sum()
        self.league_hist_data = {
            "hist_spend": total_spend,
            "hist_ppw": x,
            "hist_price": y,
            "curr_spend": 0,
            "curr_ppw": None,
            "curr_price": None,
        }

    def calc_player_price(self):
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
        self.draft_df.loc[value_mask, "Auction value"] = np.round(av)

        poss = ["QB", "RB", "WR", "TE"]
        cmap = plt.get_cmap("viridis")
        position_colors = [cmap(val) for val in np.linspace(0, 0.99, 4)]
        self.draft_df["Projected price"] = np.ones(self.draft_df.shape[0])
        for pos, ax, color in zip(poss, self.pax.flatten(), position_colors):
            ax.clear()
            pos_data = self.league_position_data[pos]
            hist_ppw = pos_data["hist_ppw"]
            hist_price = pos_data["hist_price"]
            # draft_vals = self.draft_df.loc[self.draft_df.Position == pos]["PPW"].values
            pos_inds = self.draft_df.loc[self.draft_df.Position == pos].index
            undrafted = self.draft_df.loc[pos_inds].loc[self.draft_df.Drafted == 0]
            undrafted_inds = undrafted.index
            drafted = self.draft_df.loc[pos_inds].loc[self.draft_df.Drafted != 0]
            model = lowess.Lowess()
            use_hist_point = np.ones(len(hist_ppw), dtype=bool)
            if drafted.empty:
                model.fit(hist_ppw, hist_price, frac=0.5)
            else:
                # remove historic values near currently
                curr_ppw = drafted["PPW"].values
                curr_price = drafted["Draft price"].values
                cuml_ppw = []
                cuml_price = []
                for i, (ppw, price) in enumerate(zip(hist_ppw, hist_price)):
                    if np.any(np.abs(curr_ppw - ppw) < 1.5):
                        print(curr_ppw)
                        print(ppw)
                        use_hist_point[i] = False
                    else:
                        cuml_ppw.append(ppw)
                        cuml_price.append(price)
                # cuml_ppw.extend(curr_ppw.tolist())
                # cuml_price.extend(curr_price.tolist())
                for ppw, price in zip(curr_ppw, curr_price):
                    cuml_ppw.append(ppw)
                    cuml_price.append(price)
                # x = np.concatenate([hist_ppw, drafted["PPW"].values])
                # y = np.concatenate([hist_price, drafted["Draft price"].values])
                model.fit(np.array(cuml_ppw), np.array(cuml_price), frac=0.5)
            ax.scatter(
                hist_ppw[use_hist_point],
                hist_price[use_hist_point],
                color=color,
                s=10,
                alpha=0.5,
                edgecolor="k",
            )
            ax.scatter(
                hist_ppw[~use_hist_point],
                hist_price[~use_hist_point],
                color=color,
                s=3,
                alpha=0.5,
                edgecolor="k",
            )
            if not drafted.empty:
                ax.scatter(
                    drafted["PPW"].values,
                    drafted["Draft price"].values,
                    color=color,
                    marker="x",
                )
            # total_spend = p_df.Paid.sum()
            # percent_spend = total_spend / self.available_budget
            vbds = np.linspace(5, 25, 100)
            ax.plot(
                vbds,
                np.clip(model.predict(vbds), 1, None),
                ls="--",
                zorder=0,
                color="k",
            )
            # ax.plot(vbds, np.clip(p(vbds), 1, None), ls="--", zorder=0, color="k")
            ax.set_title(f"{pos}")  # - {percent_spend:.2f}")
            ax.set_ylim([-5, 105])
            ax.set_yticks(np.arange(0, 101, 10))
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Price")
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Projected points")

            # projected_price = np.clip(p(draft_vals), 1, None)
            projected_price = np.clip(model.predict(undrafted["PPW"].values), 1, None)
            self.draft_df.loc[undrafted_inds, "Projected price"] = projected_price
            ax.scatter(
                undrafted["PPW"],
                projected_price,
                s=15,
                facecolors=color,
                edgecolors="k",
            )
        self.p_canvas.figure.tight_layout()
        self.p_canvas.draw()

    def calc_player_price_nba(self):
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
        self.draft_df.loc[value_mask, "Auction value"] = np.round(av)

        # poss = ["PG", "SG", "SF", "PF", "C"]
        cmap = plt.get_cmap("viridis")
        # position_colors = [cmap(val) for val in np.linspace(0, 0.99, 5)]
        self.draft_df["Projected price"] = np.ones(self.draft_df.shape[0])

        # USED TO BE A POSITION LOOP
        # for pos, ax, color in zip(poss, self.pax.flatten(), position_colors):
        ax = self.pax
        color = cmap(0.5)
        ax.clear()
        hist_data = self.league_hist_data
        hist_ppw = hist_data["hist_ppw"]
        hist_price = hist_data["hist_price"]
        # draft_vals = self.draft_df.loc[self.draft_df.Position == pos]["PPW"].values
        undrafted = self.draft_df.loc[self.draft_df.Drafted == 0]
        undrafted_inds = undrafted.index
        drafted = self.draft_df.loc[self.draft_df.Drafted != 0]
        model = lowess.Lowess()
        use_hist_point = np.ones(len(hist_ppw), dtype=bool)
        if drafted.empty:
            model.fit(hist_ppw, hist_price, frac=0.5)
        else:
            # remove historic values near currently
            curr_ppw = drafted["PPW"].values
            curr_price = drafted["Draft price"].values
            cuml_ppw = []
            cuml_price = []
            for i, (ppw, price) in enumerate(zip(hist_ppw, hist_price)):
                if np.any(np.abs(curr_ppw - ppw) < 1.5):
                    use_hist_point[i] = False
                else:
                    cuml_ppw.append(ppw)
                    cuml_price.append(price)
            # cuml_ppw.extend(curr_ppw.tolist())
            # cuml_price.extend(curr_price.tolist())
            for ppw, price in zip(curr_ppw, curr_price):
                cuml_ppw.append(ppw)
                cuml_price.append(price)
            # x = np.concatenate([hist_ppw, drafted["PPW"].values])
            # y = np.concatenate([hist_price, drafted["Draft price"].values])
            model.fit(np.array(cuml_ppw), np.array(cuml_price), frac=0.5)
        ax.scatter(
            hist_ppw[use_hist_point],
            hist_price[use_hist_point],
            color=color,
            s=10,
            alpha=0.5,
            edgecolor="k",
        )
        ax.scatter(
            hist_ppw[~use_hist_point],
            hist_price[~use_hist_point],
            color=color,
            s=3,
            alpha=0.5,
            edgecolor="k",
        )
        if not drafted.empty:
            ax.scatter(
                drafted["PPW"].values,
                drafted["Draft price"].values,
                color=color,
                marker="x",
            )
        # total_spend = p_df.Paid.sum()
        # percent_spend = total_spend / self.available_budget
        vbds = np.linspace(5, 25, 100)
        ax.plot(
            vbds,
            np.clip(model.predict(vbds), 1, None),
            ls="--",
            zorder=0,
            color="k",
        )
        # ax.plot(vbds, np.clip(p(vbds), 1, None), ls="--", zorder=0, color="k")
        # ax.set_title(f"{pos}")  # - {percent_spend:.2f}")
        ax.set_ylim([-5, 130])
        ax.set_yticks(np.arange(0, 101, 10))
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Price")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Projected points")

        # projected_price = np.clip(p(draft_vals), 1, None)
        projected_price = np.clip(model.predict(undrafted["PPW"].values), 1, None)
        self.draft_df.loc[undrafted_inds, "Projected price"] = projected_price
        ax.scatter(
            undrafted["PPW"],
            projected_price,
            s=15,
            facecolors=color,
            edgecolors="k",
        )
        self.p_canvas.figure.tight_layout()
        self.p_canvas.draw()

    def find_optimal_team(self, draft_state_df):
        solver = pywraplp.Solver("Draft", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Build variables, aka the players we'll be valuing
        variables = []
        for i, player in draft_state_df.iterrows():
            if player["Drafted"] == 1:
                variables.append(solver.IntVar(0, 0, player.Name))
            else:
                variables.append(solver.BoolVar(player.Name))

        # Build objective
        objective = solver.Objective()
        objective.SetMaximization()
        for i, player in draft_state_df.iterrows():
            objective.SetCoefficient(variables[i], player.PPW)

        # Build constraints
        # Constraint for salary cap
        salary_cap = solver.Constraint(0, self.my_starter_budget)
        for i, player in draft_state_df.iterrows():
            # salary_cap.SetCoefficient(variables[i], player["Auction value"])
            salary_cap.SetCoefficient(variables[i], player["Projected price"])

        # Roster size
        # roster_size = self.n_league_starters - 2
        # size_cap = solver.Constraint(roster_size, roster_size)
        # for variable in variables:
        #     # roster size
        #     size_cap.SetCoefficient(variable, 1)

        # Positional constraints
        nf = self.league_roster["FLEX"]
        qb_cap = solver.Constraint(self.league_roster["QB"], self.league_roster["QB"])
        rb_cap = solver.Constraint(
            self.league_roster["RB"], self.league_roster["RB"] + nf
        )
        wr_cap = solver.Constraint(
            self.league_roster["WR"], self.league_roster["WR"] + nf
        )
        te_cap = solver.Constraint(
            self.league_roster["TE"], self.league_roster["TE"] + nf
        )
        n_flex = (
            self.league_roster["RB"]
            + self.league_roster["WR"]
            + self.league_roster["TE"]
            + self.league_roster["FLEX"]
        )
        flex_cap = solver.Constraint(n_flex, n_flex)
        for i, player in draft_state_df.iterrows():
            pos = player["Position"]
            qb_cap.SetCoefficient(variables[i], int(pos == "QB"))
            rb_cap.SetCoefficient(variables[i], int(pos == "RB"))
            wr_cap.SetCoefficient(variables[i], int(pos == "WR"))
            te_cap.SetCoefficient(variables[i], int(pos == "TE"))
            flex_cap.SetCoefficient(variables[i], int(pos in ["RB", "WR", "TE"]))

        # On team constraint
        players_on_team = len(draft_state_df.loc[draft_state_df["Drafted"] == 2])
        if players_on_team > 0:
            players_on_team_constraint = solver.Constraint(
                int(players_on_team), int(players_on_team)
            )
            for i, player in draft_state_df.iterrows():
                if player["Drafted"] == 2:
                    players_on_team_constraint.SetCoefficient(variables[i], 1)
                    salary_cap.SetCoefficient(variables[i], player["Draft price"])
                else:
                    players_on_team_constraint.SetCoefficient(variables[i], 0)

        sol = solver.Solve()
        sol_roster = [var.name() for var in variables if var.solution_value()]
        # sol_points = sum(
        #     [
        #         player["PPW"]
        #         for (i, player), var in zip(draft_state_df.iterrows(), variables)
        #         if var.solution_value()
        #     ]
        # )
        # sol_price = 0
        # for (i, player), var in zip(draft_state_df.iterrows(), variables):
        #     if var.solution_value():
        #         if player["Drafted"] == 2:
        #             sol_price += player["Draft price"]
        #         else:
        #             sol_price += player["Auction value"]
        opt_team = self.create_team_df(sol_roster)
        return opt_team

    def find_optimal_team_nba(self, draft_state_df):
        solver = pywraplp.Solver("Draft", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        PG, SG, SF, PF, C, G, F = [], [], [], [], [], [], []
        for i, player in draft_state_df.iterrows():
            PG.append(True if "PG" in player.Position else False)
            SG.append(True if "SG" in player.Position else False)
            SF.append(True if "SF" in player.Position else False)
            PF.append(True if "PF" in player.Position else False)
            C.append(True if "C" in player.Position else False)
            G.append(True if (PG[-1] or SG[-1]) else False)
            F.append(True if (SF[-1] or PF[-1]) else False)
        # Build variables, aka the players we'll be valuing
        variables = []
        for i, player in draft_state_df.iterrows():
            if player["Drafted"] == 1:
                variables.append(solver.IntVar(0, 0, player.Name))
            else:
                variables.append(solver.BoolVar(player.Name))

        # Build objective
        objective = solver.Objective()
        objective.SetMaximization()
        for i, player in draft_state_df.iterrows():
            objective.SetCoefficient(variables[i], player.PPW)

        # Build constraints
        # Constraint for salary cap
        salary_cap = solver.Constraint(0, self.my_starter_budget)
        for i, player in draft_state_df.iterrows():
            salary_cap.SetCoefficient(variables[i], player["Projected price"])

        # Roster size
        roster_size = self.roster_size
        size_cap = solver.Constraint(roster_size, roster_size)
        for variable in variables:
            # roster size
            size_cap.SetCoefficient(variable, 1)

        # Positional constraints
        # nf = self.league_roster["UTIL"]
        # qb_cap = solver.Constraint(self.league_roster["QB"], self.league_roster["QB"])
        # rb_cap = solver.Constraint(
        #     self.league_roster["RB"], self.league_roster["RB"] + nf
        # )
        # wr_cap = solver.Constraint(
        #     self.league_roster["WR"], self.league_roster["WR"] + nf
        # )
        # te_cap = solver.Constraint(
        #     self.league_roster["TE"], self.league_roster["TE"] + nf
        # )
        # n_flex = (
        #     self.league_roster["RB"]
        #     + self.league_roster["WR"]
        #     + self.league_roster["TE"]
        #     + self.league_roster["FLEX"]
        # )
        # flex_cap = solver.Constraint(n_flex, n_flex)
        # for i, player in draft_state_df.iterrows():
        #     pos = player["Position"]
        #     qb_cap.SetCoefficient(variables[i], int(pos == "QB"))
        #     rb_cap.SetCoefficient(variables[i], int(pos == "RB"))
        #     wr_cap.SetCoefficient(variables[i], int(pos == "WR"))
        #     te_cap.SetCoefficient(variables[i], int(pos == "TE"))
        #     flex_cap.SetCoefficient(variables[i], int(pos in ["RB", "WR", "TE"]))

        pg_cap = solver.Constraint(2, self.roster_size)
        sg_cap = solver.Constraint(2, self.roster_size)
        sf_cap = solver.Constraint(2, self.roster_size)
        pf_cap = solver.Constraint(2, self.roster_size)
        c_cap = solver.Constraint(2, self.roster_size)
        g_cap = solver.Constraint(4, self.roster_size)
        f_cap = solver.Constraint(4, self.roster_size)
        for i, player in draft_state_df.iterrows():
            pg_cap.SetCoefficient(variables[i], PG[i])
            sg_cap.SetCoefficient(variables[i], SG[i])
            sf_cap.SetCoefficient(variables[i], SF[i])
            pf_cap.SetCoefficient(variables[i], PF[i])
            c_cap.SetCoefficient(variables[i], C[i])
            g_cap.SetCoefficient(variables[i], G[i])
            f_cap.SetCoefficient(variables[i], F[i])

        # On team constraint
        players_on_team = len(draft_state_df.loc[draft_state_df["Drafted"] == 2])
        if players_on_team > 0:
            players_on_team_constraint = solver.Constraint(
                int(players_on_team), int(players_on_team)
            )
            for i, player in draft_state_df.iterrows():
                if player["Drafted"] == 2:
                    players_on_team_constraint.SetCoefficient(variables[i], 1)
                    salary_cap.SetCoefficient(variables[i], player["Draft price"])
                else:
                    players_on_team_constraint.SetCoefficient(variables[i], 0)

        sol = solver.Solve()
        sol_roster = [var.name() for var in variables if var.solution_value()]
        # sol_points = sum(
        #     [
        #         player["PPW"]
        #         for (i, player), var in zip(draft_state_df.iterrows(), variables)
        #         if var.solution_value()
        #     ]
        # )
        # sol_price = 0
        # for (i, player), var in zip(draft_state_df.iterrows(), variables):
        #     if var.solution_value():
        #         if player["Drafted"] == 2:
        #             sol_price += player["Draft price"]
        #         else:
        #             sol_price += player["Auction value"]
        opt_team = self.create_team_df(sol_roster)
        return opt_team

    def calc_player_opportunity_cost(self, player_name):
        player_ind = self.draft_df.loc[self.draft_df["Name"] == player_name].index
        base_price = self.draft_df.loc[player_ind, "Projected price"].values[0]
        # base_price = self.draft_df.loc[player_ind, "Projected price"].values[0]
        min_price = max(1, base_price - 25)
        max_price = base_price + 25
        test_prices = np.linspace(min_price, max_price, 7)
        test_df = copy.deepcopy(self.draft_df)
        opp_costs = []
        for i, test_price in enumerate(test_prices):
            # Get the optimal team if the player is drafted at the test price
            test_df.loc[player_ind, "Drafted"] = 2
            test_df.loc[player_ind, "Draft price"] = test_price
            opt_take_team = self.find_optimal_team(test_df)
            opt_take_ppw = opt_take_team["PPW"].sum()
            # print(f"Take {player_name} at {test_price}: {opt_take_ppw}")

            test_df.loc[player_ind, "Drafted"] = 1
            test_df.loc[player_ind, "Draft price"] = test_price
            opt_leave_team = self.find_optimal_team(test_df)
            opt_leave_ppw = opt_leave_team["PPW"].sum()
            # print(f"Leave {player_name} at {test_price}: {opt_leave_ppw}")
            opportunity_cost = opt_leave_ppw - opt_take_ppw
            # print(f"Opportunity Cost = {opportunity_cost}\n")
            opp_costs.append(opportunity_cost)

        cmap = plt.get_cmap("RdYlGn_r")
        norm = mpl.colors.Normalize(vmin=-5, vmax=5)
        self.oc_ax.clear()
        self.oc_ax.scatter(
            test_prices[: i + 1],
            opp_costs,
            c=opp_costs,
            cmap=cmap,
            norm=norm,
            edgecolor="k",
        )
        self.oc_ax.set_xlabel("Paid price")
        self.oc_ax.set_ylabel("Opportunity cost")
        self.oc_ax.axhline(0, color="k", ls="--")
        self.oc_canvas.draw()
        # Set up linear fit to find the root
        z = np.polyfit(test_prices, opp_costs, 1)
        p = np.poly1d(z)
        z2 = np.polyfit(test_prices, np.array(opp_costs) + 2, 1)
        p2 = np.poly1d(z2)
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
        if np.sign(p2(min_price)) != np.sign(p2(max_price)):
            root = root_scalar(p2, method="bisect", bracket=[min_price, max_price])
            self.oc_ax.axvline(root.root, color="k", ls="--")
            title += f" - Draft below ${root.root:.2f}"
        elif np.sign(p2(min_price)) > 0 and np.sign(p2(max_price)) > 0:
            title += " - Probably not worth it"
        elif np.sign(p2(min_price)) < 0 and np.sign(p2(max_price)) < 0:
            title += " - Draft for any price"
        else:
            title += " - How is this possible"
        self.oc_ax.set_title(title)
        self.oc_ax.set_ylim([-10, 10])
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
            zorder=0,
        )
        self.oc_canvas.draw()

        # breakpoint()
