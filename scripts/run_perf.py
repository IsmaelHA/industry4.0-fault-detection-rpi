"""Phase 9+: time the per-file pipeline under three easy optimisation variants.

  baseline : np.loadtxt parses all 4 channels (float64), then we slice the column
  usecols  : np.loadtxt parses only the bearing column   (float64)
  float32  : np.loadtxt parses only the bearing column, cast to float32

The signal numbers are identical across variants (float32 differs only in the
last few insignificant digits), so the entropy is unchanged; only the load cost
moves. Results feed Figure F6.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io_ims import list_files  # noqa: E402
from src.perf import Stopwatch  # noqa: E402
from src.runtime import load_config  # noqa: E402
from src.segmentation import windows  # noqa: E402
from src.slope_entropy import slope_entropy  # noqa: E402


def _load_baseline(path, channel):
    arr = np.loadtxt(path)
    return arr if arr.ndim == 1 else arr[:, channel]


def _load_usecols(path, channel):
    return np.loadtxt(path, usecols=channel)


def _load_float32(path, channel):
    return np.loadtxt(path, usecols=channel).astype(np.float32, copy=False)


def _time_variant(variant, paths, loader, channel, win, stride, m, delta, gamma, adaptive, out_csv):
    if out_csv.exists():
        out_csv.unlink()
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "file_name", "dt_s", "peak_rss_kb", "variant"])
        for idx, p in enumerate(paths):
            with Stopwatch() as sw:
                sig = loader(p, channel)
                for win_slice in windows(sig, win=win, stride=stride):
                    slope_entropy(win_slice, m=m, delta=delta, gamma=gamma, adaptive=adaptive)
            w.writerow([idx, p.name, f"{sw.dt:.6f}", sw.peak_rss, variant])
            if idx % 25 == 0:
                print(f"  [{variant}] {idx} dt={sw.dt*1000:.1f} ms")
    print(f"[Perf] {variant} -> {out_csv}")


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    win = int(cfg["segmentation"]["window"])
    stride = int(cfg["segmentation"]["stride"])
    m = int(cfg["entropy"]["m"])
    delta = float(cfg["entropy"]["delta"])
    gamma = float(cfg["entropy"]["gamma"])
    adaptive = bool(cfg["entropy"]["adaptive_thresholds"])
    sampling_stride = int(cfg["sweep"]["sampling_stride"])
    channel = int(cfg["channel"])

    files = list_files(Path(cfg["data_root"]))[::sampling_stride]
    print(f"[Perf] {len(files)} files (stride={sampling_stride})")

    _time_variant("baseline", files, _load_baseline, channel, win, stride, m, delta, gamma,
                  adaptive, results_dir / "perf_log_baseline.csv")
    _time_variant("usecols", files, _load_usecols, channel, win, stride, m, delta, gamma,
                  adaptive, results_dir / "perf_log_usecols.csv")
    _time_variant("float32", files, _load_float32, channel, win, stride, m, delta, gamma,
                  adaptive, results_dir / "perf_log_float32.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
