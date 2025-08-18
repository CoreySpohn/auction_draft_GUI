from typing import Tuple

import numpy as np


def compute_linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Return slope and intercept of a 1D linear least squares fit.
    """
    z = np.polyfit(x, y, 1)
    return float(z[0]), float(z[1])

