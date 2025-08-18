from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PyQt5 import QtGui as qtg


def position_qcolors(sport: str) -> Dict[str, qtg.QColor]:
    cmap = plt.get_cmap("viridis")
    if sport == "nfl":
        positions = ["QB", "RB", "WR", "TE"]
    else:
        positions = ["PG", "SG", "SF", "PF", "C"]
    result: Dict[str, qtg.QColor] = {}
    for position, color_val in zip(positions, np.linspace(0, 0.99, len(positions))):
        color = cmap(color_val)
        result[position] = qtg.QColor.fromRgbF(color[0], color[1], color[2], 0.25)
    return result


def qt_color_from_name(name: str, alpha: float = 0.25) -> qtg.QColor:
    r, g, b, _ = colors.to_rgba(name)
    return qtg.QColor.fromRgbF(r, g, b, alpha)

