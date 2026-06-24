"""Per-file entropy aggregation and CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np

from src.segmentation import windows
from src.slope_entropy import slope_entropy


def file_entropy_stats(
    signal: np.ndarray,
    win: int,
    stride: int,
    m: int,
    delta: float,
    gamma: float,
    adaptive: bool,
) -> dict:
    vals: List[float] = []
    for w in windows(signal, win=win, stride=stride):
        vals.append(slope_entropy(w, m=m, delta=delta, gamma=gamma, adaptive=adaptive))
    if not vals:
        return {"mean": float("nan"), "q25": float("nan"), "q75": float("nan"),
                "std": float("nan"), "n_windows": 0, "values": []}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "std": float(a.std()),
        "n_windows": int(a.size),
        "values": vals,
    }


class TimelineWriter:
    HEADER = ["idx", "file_name", "mean_H", "q25_H", "q75_H", "std_H",
              "n_windows", "rms"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.fh = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fh)
        if new:
            self.writer.writerow(self.HEADER)
            self.fh.flush()

    def write(self, row: dict) -> None:
        self.writer.writerow([row.get(k, "") for k in self.HEADER])
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
