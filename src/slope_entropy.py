"""Slope Entropy (Cuesta-Frau 2019) in pure NumPy."""

from __future__ import annotations

import numpy as np


def _symbolise(diff: np.ndarray, delta: float, gamma: float) -> np.ndarray:
    sym = np.zeros_like(diff, dtype=np.int8)
    sym[diff > gamma] = 2
    sym[(diff > delta) & (diff <= gamma)] = 1
    sym[(diff < -delta) & (diff >= -gamma)] = -1
    sym[diff < -gamma] = -2
    return sym


def slope_entropy(
    x: np.ndarray,
    m: int = 4,
    delta: float = 1e-3,
    gamma: float = 1e-1,
    adaptive: bool = True,
    normalise: bool = True,
) -> float:
    if m < 2:
        raise ValueError("m must be >= 2")
    if delta >= gamma:
        raise ValueError("delta must be < gamma")

    if adaptive:
        sigma = float(np.std(x))
        if sigma == 0.0:
            return 0.0
        delta_eff = delta * sigma
        gamma_eff = gamma * sigma
    else:
        delta_eff = delta
        gamma_eff = gamma

    diff = np.diff(np.asarray(x, dtype=np.float64))
    sym = _symbolise(diff, delta_eff, gamma_eff)

    if len(sym) < m - 1:
        return 0.0

    patterns = np.lib.stride_tricks.sliding_window_view(sym, m - 1)
    keys = np.zeros(len(patterns), dtype=np.int64)
    for k in range(m - 1):
        keys = keys * 5 + (patterns[:, k] + 2)

    _, counts = np.unique(keys, return_counts=True)
    p = counts / counts.sum()
    H = float(-np.sum(p * np.log(p)))
    if normalise:
        H /= float(np.log(5 ** (m - 1)))
    return H
