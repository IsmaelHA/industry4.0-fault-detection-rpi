"""Windowing for IMS signals: fixed-size windows with optional overlap."""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


def n_windows(length: int, win: int = 2048, stride: Optional[int] = None) -> int:
    stride = win if stride is None else stride
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")
    return max(0, (length - win) // stride + 1)


def windows(
    signal: np.ndarray,
    win: int = 2048,
    stride: Optional[int] = None,
) -> Iterator[np.ndarray]:
    stride = win if stride is None else stride
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")
    n = n_windows(len(signal), win=win, stride=stride)
    for i in range(n):
        start = i * stride
        yield signal[start : start + win]
