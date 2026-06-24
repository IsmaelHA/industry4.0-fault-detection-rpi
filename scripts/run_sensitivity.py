"""Phase 8: parameter sensitivity (F5). Sweeps embedding m and threshold gamma.

Produces:
  results/figures/F5_sensitivity_m.png        timeline at m in {3,4,5}
  results/figures/F5_sensitivity_gamma.png    timeline at gamma in {0.05,0.1,0.2}
  results/sensitivity_change_points.csv       turning point for every setting
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import control_limit_change_point, rms  # noqa: E402
from src.io_ims import display_name, iter_signals  # noqa: E402
from src.plots import plot_sensitivity  # noqa: E402
from src.runtime import load_config, set_seed  # noqa: E402
from src.timeline import TimelineWriter, file_entropy_stats  # noqa: E402


def _timeline(items, results_dir, tag, win, stride, m, delta, gamma, adaptive):
    out_csv = results_dir / f"entropy_timeline_{tag}.csv"
    if out_csv.exists():
        out_csv.unlink()
    H = []
    with TimelineWriter(out_csv) as tw:
        for idx, p, sig in items:
            st = file_entropy_stats(sig, win, stride, m, delta, gamma, adaptive)
            tw.write({"idx": idx, "file_name": display_name(p),
                      "mean_H": st["mean"], "q25_H": st["q25"], "q75_H": st["q75"],
                      "std_H": st["std"], "n_windows": st["n_windows"], "rms": rms(sig)})
            H.append(st["mean"])
    return out_csv, np.asarray(H)


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    set_seed(cfg["seed"])
    results_dir = Path(cfg["results_dir"])
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    win = int(cfg["segmentation"]["window"])
    stride = int(cfg["segmentation"]["stride"])
    m0 = int(cfg["entropy"]["m"])
    delta = float(cfg["entropy"]["delta"])
    gamma0 = float(cfg["entropy"]["gamma"])
    adaptive = bool(cfg["entropy"]["adaptive_thresholds"])
    sampling_stride = int(cfg["sweep"]["sampling_stride"])
    channel = int(cfg["channel"])
    hf = float(cfg["control_limit"]["healthy_fraction"])
    nsig = float(cfg["control_limit"]["n_sigma"])
    minc = int(cfg["control_limit"]["min_consecutive"])
    m_values = list(cfg["sensitivity"]["m_values"])
    gamma_values = list(cfg["sensitivity"]["gamma_values"])

    print(f"[Sens] Loading sparse subset (stride={sampling_stride})...")
    items = [(idx, p, sig) for idx, p, sig in iter_signals(cfg, channel=channel, stride=sampling_stride)]
    idxs = [it[0] for it in items]
    print(f"[Sens] {len(items)} files")

    rows = []

    # --- embedding-dimension sweep (gamma fixed at config default) ---
    csvs, labels, cps = [], [], []
    for m in m_values:
        csvp, H = _timeline(items, results_dir, f"m{m}", win, stride, m, delta, gamma0, adaptive)
        cp, _ = control_limit_change_point(H, hf, nsig, minc)
        csvs.append(csvp); labels.append(f"H (m={m})"); cps.append(cp)
        fi = idxs[cp] if cp is not None else None
        rows.append(["m", m, gamma0, cp if cp is not None else "", fi if fi is not None else ""])
        print(f"[Sens] m={m}: turning point file_idx={fi}")
    plot_sensitivity(csvs, labels, cps, figures_dir / "F5_sensitivity_m.png",
                     title=f"F5 — Sensitivity to embedding dimension m (gamma={gamma0})")
    print(f"[Sens] Saved: {figures_dir / 'F5_sensitivity_m.png'}")

    # --- threshold (gamma) sweep (m fixed at config default) ---
    csvs, labels, cps = [], [], []
    for g in gamma_values:
        csvp, H = _timeline(items, results_dir, f"g{g}", win, stride, m0, delta, g, adaptive)
        cp, _ = control_limit_change_point(H, hf, nsig, minc)
        csvs.append(csvp); labels.append(f"H (γ={g})"); cps.append(cp)
        fi = idxs[cp] if cp is not None else None
        rows.append(["gamma", m0, g, cp if cp is not None else "", fi if fi is not None else ""])
        print(f"[Sens] gamma={g}: turning point file_idx={fi}")
    plot_sensitivity(csvs, labels, cps, figures_dir / "F5_sensitivity_gamma.png",
                     title=f"F5 — Sensitivity to threshold gamma (m={m0})")
    print(f"[Sens] Saved: {figures_dir / 'F5_sensitivity_gamma.png'}")

    with (results_dir / "sensitivity_change_points.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["swept_param", "m", "gamma", "sweep_index", "file_index"])
        w.writerows(rows)
    print(f"[Sens] Wrote {results_dir / 'sensitivity_change_points.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
