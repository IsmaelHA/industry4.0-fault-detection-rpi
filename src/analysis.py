"""Turning-point detection (3-sigma control limit) and the RMS indicator."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def control_limit_change_point(
    H: np.ndarray,
    healthy_fraction: float = 0.25,
    n_sigma: float = 3.0,
    min_consecutive: int = 3,
) -> Tuple[Optional[int], dict]:
    """Detect the turning point with a Shewhart-style control limit.

    The healthy baseline (mean mu0, std sigma0) is estimated from the first
    ``healthy_fraction`` of the series. A control band ``mu0 +/- n_sigma*sigma0``
    is formed. The turning point is the start of the first run of
    ``min_consecutive`` consecutive points that fall outside the band — a
    *sustained* excursion, which rejects isolated one-off spikes.

    Returns ``(change_point_index_or_None, band)`` where ``band`` holds
    ``mu0``, ``sigma0``, ``lower`` and ``upper`` for plotting.
    """
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    n_healthy = max(2, int(n * healthy_fraction))
    mu0 = float(H[:n_healthy].mean())
    sigma0 = float(H[:n_healthy].std() + 1e-12)
    lower = mu0 - n_sigma * sigma0
    upper = mu0 + n_sigma * sigma0
    band = {"mu0": mu0, "sigma0": sigma0, "lower": lower, "upper": upper}

    if n < 2:
        return None, band

    outside = (H < lower) | (H > upper)
    run = 0
    for i in range(n):
        run = run + 1 if outside[i] else 0
        if run >= min_consecutive:
            return i - min_consecutive + 1, band
    return None, band


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2)))
