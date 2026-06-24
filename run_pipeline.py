"""End-to-end pipeline (Phases 1-6). Reads every parameter from config.yaml.

Outputs:
  results/entropy_timeline.csv          per-file mean/IQR entropy + RMS
  results/control_limit_sweep.csv       turning point at n_sigma in {2,3,4}
  results/indicator_change_points.csv   entropy vs RMS turning points
  results/comparison_summary.csv        3-stage summary table
  results/perf_log_baseline.csv         per-file timing (default settings)
  results/figures/F1_raw_healthy.png
  results/figures/F2_raw_degraded.png
  results/figures/F3_entropy_timeline.png
  results/figures/F4_turning_point.png
  results/figures/Fextra_3stage_raw.png
  results/figures/Fextra_3stage_entropy.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.analysis import control_limit_change_point, rms
from src.comparison import assign_stage
from src.io_ims import (
    SAMPLING_RATE_HZ,
    basic_statistics,
    display_name,
    iter_signals,
    load_signal,
)
from src.perf import Stopwatch
from src.plots import (
    plot_raw_signal,
    plot_three_stage_entropy,
    plot_three_stage_raw,
    plot_timeline,
    plot_turning_point,
)
from src.runtime import load_config, set_seed
from src.segmentation import n_windows
from src.slope_entropy import slope_entropy
from src.timeline import TimelineWriter, file_entropy_stats


CANDIDATE_WINDOWS = (512, 1024, 2048, 4096, 8192)


def main() -> None:
    cfg = load_config("config.yaml")
    set_seed(cfg["seed"])

    results_dir = Path(cfg["results_dir"])
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    win = int(cfg["segmentation"]["window"])
    stride = int(cfg["segmentation"]["stride"])
    m = int(cfg["entropy"]["m"])
    delta = float(cfg["entropy"]["delta"])
    gamma = float(cfg["entropy"]["gamma"])
    adaptive = bool(cfg["entropy"]["adaptive_thresholds"])
    sampling_stride = int(cfg["sweep"]["sampling_stride"])
    channel = int(cfg["channel"])
    duration_ms = float(cfg["figure"]["duration_ms"])
    dpi = int(cfg["figure"]["dpi"])

    # =========================================================================
    # Phase 1 — Data loading + F1 (PDF 7.1)
    # =========================================================================
    first_idx, first_path, first_signal = next(iter_signals(cfg, channel=channel))
    stats = basic_statistics(first_signal)
    print(f"[Phase 1] First file: {display_name(first_path)} (idx {first_idx})")
    print(f"[Phase 1] length={stats['n_samples']} ({stats['duration_s']:.2f} s @ {SAMPLING_RATE_HZ} Hz)")
    print(f"[Phase 1] mean={stats['mean']:.4f} std={stats['std']:.4f} rms={stats['rms']:.4f} g")

    plot_raw_signal(first_signal, fs=SAMPLING_RATE_HZ,
                    out_path=figures_dir / "F1_raw_healthy.png",
                    title=f"F1 — Raw vibration (healthy stage, {display_name(first_path)})",
                    duration_ms=duration_ms, dpi=dpi)
    print(f"[Phase 1] Saved: {figures_dir / 'F1_raw_healthy.png'}")

    # =========================================================================
    # Phase 2 — Segmentation diagnostics (PDF 7.2)
    # =========================================================================
    print(f"[Phase 2] cfg: win={win} stride={stride} -> {n_windows(len(first_signal), win, stride)} windows/file")
    print("[Phase 2] Window-size sweep (no overlap):")
    for w in CANDIDATE_WINDOWS:
        print(f"  win={w:>5}  {w / SAMPLING_RATE_HZ * 1e3:>7.2f} ms  {n_windows(len(first_signal), w):>4} windows")

    # =========================================================================
    # Phase 3 — Slope Entropy smoke test (PDF 7.3)
    # =========================================================================
    h_smoke = slope_entropy(first_signal[:win], m=m, delta=delta, gamma=gamma, adaptive=adaptive)
    print(f"[Phase 3] slope_entropy(first window, m={m}, adaptive={adaptive}) = {h_smoke:.4f}")

    # =========================================================================
    # Phase 4 — Per-file timeline + F3 + perf log (PDF 7.4)
    # =========================================================================
    timeline_csv = results_dir / "entropy_timeline.csv"
    perf_csv = results_dir / "perf_log_baseline.csv"
    if timeline_csv.exists():
        timeline_csv.unlink()
    if perf_csv.exists():
        perf_csv.unlink()

    perf_fh = perf_csv.open("w", newline="", encoding="utf-8")
    perf_writer = csv.writer(perf_fh)
    perf_writer.writerow(["idx", "file_name", "dt_s", "peak_rss_kb", "variant"])

    per_file = []  # (file_idx, path, name, mean_H, values, rms)

    print(f"[Phase 4] Sweeping every {sampling_stride}th file...")
    with TimelineWriter(timeline_csv) as tw:
        for sw_pos, (file_idx, p, sig) in enumerate(
            iter_signals(cfg, channel=channel, stride=sampling_stride)
        ):
            name = display_name(p)
            with Stopwatch() as sw:
                stats_w = file_entropy_stats(sig, win, stride, m, delta, gamma, adaptive)
                r_val = rms(sig)
            tw.write({
                "idx": file_idx, "file_name": name,
                "mean_H": stats_w["mean"], "q25_H": stats_w["q25"], "q75_H": stats_w["q75"],
                "std_H": stats_w["std"], "n_windows": stats_w["n_windows"], "rms": r_val,
            })
            perf_writer.writerow([file_idx, name, f"{sw.dt:.6f}", sw.peak_rss, "baseline"])
            per_file.append((file_idx, p, name, stats_w["mean"], stats_w["values"], r_val))
            if sw_pos % 25 == 0:
                print(f"  [{sw_pos:>4}] file_idx={file_idx} dt={sw.dt*1000:.1f} ms H={stats_w['mean']:.3f}")
    perf_fh.close()
    print(f"[Phase 4] Wrote {timeline_csv} ({len(per_file)} rows)")
    print(f"[Phase 4] Wrote {perf_csv}")

    f3_path = figures_dir / "F3_entropy_timeline.png"
    plot_timeline(timeline_csv, f3_path,
                  title="F3 — Slope Entropy evolution over time", dpi=dpi)
    print(f"[Phase 4] Saved: {f3_path}")

    # =========================================================================
    # Phase 5 — Turning-point (3-sigma control limit) + F4 + sweep (PDF 7.5)
    # =========================================================================
    H = np.array([row[3] for row in per_file], dtype=np.float64)
    RMS = np.array([row[5] for row in per_file], dtype=np.float64)
    file_idx_arr = np.array([row[0] for row in per_file])
    healthy_frac = float(cfg["control_limit"]["healthy_fraction"])
    n_sigma = float(cfg["control_limit"]["n_sigma"])
    min_consec = int(cfg["control_limit"]["min_consecutive"])
    n_sigma_sweep = tuple(float(x) for x in cfg["control_limit"].get("n_sigma_sweep", [2.0, 3.0, 4.0]))

    # Uncertainty sweep on the entropy timeline.
    sweep_csv = results_dir / "control_limit_sweep.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n_sigma", "min_consecutive", "sweep_index", "file_index"])
        for ns in n_sigma_sweep:
            cp_s, _ = control_limit_change_point(H, healthy_frac, ns, min_consec)
            f_idx = per_file[cp_s][0] if cp_s is not None else ""
            w.writerow([ns, min_consec, cp_s if cp_s is not None else "", f_idx])
            print(f"[Phase 5] entropy control-limit n_sigma={ns}: sweep_idx={cp_s}, file_idx={f_idx or 'none'}")

    # Does entropy lead the classic amplitude indicator? Same detector on RMS.
    indic_csv = results_dir / "indicator_change_points.csv"
    with indic_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["indicator", "n_sigma", "min_consecutive", "sweep_index", "file_index"])
        for name, series in (("slope_entropy", H), ("rms", RMS)):
            cp_i, _ = control_limit_change_point(series, healthy_frac, n_sigma, min_consec)
            f_idx = int(file_idx_arr[cp_i]) if cp_i is not None else ""
            w.writerow([name, n_sigma, min_consec, cp_i if cp_i is not None else "", f_idx])
            print(f"[Phase 5] {name:13} control-limit (n_sigma={n_sigma}): "
                  f"sweep_idx={cp_i}, file_idx={f_idx or 'none'}")

    cp, band = control_limit_change_point(H, healthy_frac, n_sigma, min_consec)
    f4_path = figures_dir / "F4_turning_point.png"
    plot_turning_point(timeline_csv, cp, band, f4_path,
                       title=f"F4 — Turning-point detection ({n_sigma:.0f}σ control limit)",
                       dpi=dpi)
    print(f"[Phase 5] Saved: {f4_path}")

    if cp is None:
        print("[Phase 6] No sustained excursion detected — skipping 3-stage analysis.")
        print("          Try lowering n_sigma/min_consecutive, tuning thresholds, or another bearing.")
        return

    # =========================================================================
    # Phase 6 — Three-stage comparison + F2 + extras (PDF 7.6)
    # =========================================================================
    margin = int(cfg.get("comparison", {}).get("margin", 10))
    stages = {"healthy": [], "transition": [], "faulty": []}
    for sw_pos, row in enumerate(per_file):
        stages[assign_stage(sw_pos, cp, margin)].append(row)
    print("[Phase 6] Stage sizes: " + ", ".join(f"{s}={len(v)}" for s, v in stages.items()))

    rep = {}
    for label, rows in stages.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r[3])  # by mean_H
        rep[label] = rows_sorted[len(rows_sorted) // 2]

    rep_signals = {label: load_signal(info[1], channel=channel)
                   for label, info in rep.items()}
    # Healthy reference = file 0 (its stats are the ones reported in the paper).
    rep_signals["healthy"] = first_signal
    if "healthy" in rep:
        rep["healthy"] = (first_idx, first_path, display_name(first_path)) + rep["healthy"][3:]
    print("[Phase 6] Representatives: " + ", ".join(f"{k}={v[2]}" for k, v in rep.items()))

    y_max = max(float(np.max(np.abs(s))) for s in rep_signals.values())
    ylim = (-y_max * 1.05, y_max * 1.05)

    plot_raw_signal(first_signal, fs=SAMPLING_RATE_HZ,
                    out_path=figures_dir / "F1_raw_healthy.png",
                    title=f"F1 — Raw vibration (healthy stage, {display_name(first_path)})",
                    duration_ms=duration_ms, ylim=ylim, dpi=dpi)
    print("[Phase 6] Re-saved F1 (file 0) with shared y-axis range")

    if "faulty" in rep_signals:
        f2_path = figures_dir / "F2_raw_degraded.png"
        plot_raw_signal(rep_signals["faulty"], fs=SAMPLING_RATE_HZ, out_path=f2_path,
                        title=f"F2 — Raw vibration (faulty stage, {rep['faulty'][2]})",
                        duration_ms=duration_ms, ylim=ylim, dpi=dpi)
        print(f"[Phase 6] Saved: {f2_path}")

    panel_labels, panel_signals = [], []
    for label in ("healthy", "transition", "faulty"):
        if label in rep_signals:
            panel_labels.append(f"{label}\n({rep[label][2]})")
            panel_signals.append(rep_signals[label])
    fextra_raw = figures_dir / "Fextra_3stage_raw.png"
    plot_three_stage_raw(panel_signals, panel_labels, fs=SAMPLING_RATE_HZ,
                         out_path=fextra_raw, ylim=ylim, duration_ms=duration_ms,
                         title="Representative raw signals per stage", dpi=dpi)
    print(f"[Phase 6] Saved: {fextra_raw}")

    dist_labels, distributions = [], []
    for label in ("healthy", "transition", "faulty"):
        vals = []
        for row in stages[label]:
            vals.extend(row[4])  # window-level entropies
        if vals:
            dist_labels.append(f"{label}\n(n={len(vals)})")
            distributions.append(np.asarray(vals, dtype=np.float64))
    fextra_ent = figures_dir / "Fextra_3stage_entropy.png"
    plot_three_stage_entropy(distributions, dist_labels, fextra_ent,
                              title="Slope Entropy distribution per stage", dpi=dpi)
    print(f"[Phase 6] Saved: {fextra_ent}")

    summary_csv = results_dir / "comparison_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "n_files", "n_windows", "mean_H", "std_H", "median_H", "mean_rms"])
        for label in ("healthy", "transition", "faulty"):
            rows = stages[label]
            if not rows:
                w.writerow([label, 0, 0, "nan", "nan", "nan", "nan"])
                continue
            all_H = []
            for r_ in rows:
                all_H.extend(r_[4])
            arr = np.asarray(all_H, dtype=np.float64)
            rms_arr = np.asarray([r_[5] for r_ in rows], dtype=np.float64)
            w.writerow([label, len(rows), int(arr.size),
                        f"{arr.mean():.6f}", f"{arr.std():.6f}",
                        f"{float(np.median(arr)):.6f}", f"{rms_arr.mean():.6f}"])
    print(f"[Phase 6] Wrote {summary_csv}")
    print("[Done]")


if __name__ == "__main__":
    main()
