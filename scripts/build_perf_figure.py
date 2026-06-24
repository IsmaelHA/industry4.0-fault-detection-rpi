"""Phase 14: build F6 from perf_log_*.csv files."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.plots import plot_perf
from src.runtime import load_config


VARIANTS = [
    ("baseline", "Baseline\n(all cols, float64)"),
    ("usecols",  "Read only\nbearing column"),
    ("float32",  "+ float32"),
]


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    results_dir = Path(cfg["results_dir"])
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label in VARIANTS:
        path = results_dir / f"perf_log_{key}.csv"
        if not path.exists():
            print(f"[F6] Skipping {key}: {path} not found")
            continue
        dts = []
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                dts.append(float(row["dt_s"]))
        if dts:
            arr = np.asarray(dts)
            rows.append((label, float(arr.mean()), float(arr.std())))
            print(f"[F6] {key}: mean={arr.mean()*1000:.1f} ms +/- {arr.std()*1000:.1f} ms (n={arr.size})")

    if not rows:
        print("[F6] No perf logs found.")
        return 1

    out = figures_dir / "F6_execution_time.png"
    plot_perf(rows, out, title="F6 — Per-file processing time on Raspberry Pi 5")
    print(f"[F6] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
