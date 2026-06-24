"""Three-stage split (healthy / transition / faulty) and per-stage statistics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def assign_stage(index: int, i_star: int, margin: int = 10) -> str:
    if index < i_star - margin:
        return "healthy"
    if index < i_star + margin:
        return "transition"
    return "faulty"


def summary_stats(values: Iterable[float]) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }
