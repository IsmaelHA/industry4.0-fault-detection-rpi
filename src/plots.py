"""Matplotlib helpers. Agg backend so the same code works headless on the Pi."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import csv  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence, Tuple, Union  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_DPI = 300


def plot_raw_signal(
    signal: np.ndarray,
    fs: float,
    out_path: Union[str, Path],
    title: str = "Raw vibration signal",
    duration_ms: float = 50.0,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    n = int(duration_ms * 1e-3 * fs)
    n = max(1, min(n, len(signal)))
    t_ms = np.arange(n) / fs * 1e3
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(t_ms, signal[:n], linewidth=0.7, color="#1f4e79")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (g)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def _read_timeline_csv(csv_path: Path):
    idx, mean_H, q25_H, q75_H, rms_v = [], [], [], [], []
    with Path(csv_path).open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            idx.append(int(row["idx"]))
            mean_H.append(float(row["mean_H"]))
            q25_H.append(float(row["q25_H"]))
            q75_H.append(float(row["q75_H"]))
            rms_v.append(float(row.get("rms", "nan")))
    return (np.asarray(idx), np.asarray(mean_H), np.asarray(q25_H),
            np.asarray(q75_H), np.asarray(rms_v))


def plot_timeline(
    csv_path: Union[str, Path],
    out_path: Union[str, Path],
    title: str = "Slope Entropy evolution",
    thirds: bool = True,
    dpi: int = DEFAULT_DPI,
) -> Path:
    idx, mean_H, q25_H, q75_H, _ = _read_timeline_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.fill_between(idx, q25_H, q75_H, color="#1f4e79", alpha=0.25, label="IQR (per file)")
    ax.plot(idx, mean_H, color="#1f4e79", linewidth=1.2, label="mean H")
    if thirds and len(idx) > 0:
        n = len(idx)
        for frac in (1 / 3, 2 / 3):
            ax.axvline(idx[int(n * frac)], linestyle="--", color="#888", linewidth=0.7)
    ax.set_xlabel("File index (chronological)")
    ax.set_ylabel("Normalised Slope Entropy")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_turning_point(
    csv_path: Union[str, Path],
    change_point_sweep_idx: Optional[int],
    band: dict,
    out_path: Union[str, Path],
    title: str = "Turning-point detection",
    dpi: int = DEFAULT_DPI,
) -> Path:
    idx, mean_H, q25_H, q75_H, _ = _read_timeline_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.fill_between(idx, q25_H, q75_H, color="#1f4e79", alpha=0.20, label="IQR")
    ax.plot(idx, mean_H, color="#1f4e79", linewidth=1.2, label="mean H")
    ax.axhline(band["mu0"], color="#555", linewidth=0.8, label="healthy mean")
    ax.axhspan(band["lower"], band["upper"], color="#999", alpha=0.15,
               label="healthy control band")
    if change_point_sweep_idx is not None and change_point_sweep_idx < len(idx):
        ax.axvline(idx[change_point_sweep_idx], color="#c0392b", linewidth=1.5,
                   label=f"turning point (file {idx[change_point_sweep_idx]})")
    ax.set_xlabel("File index (chronological)")
    ax.set_ylabel("Normalised Slope Entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_three_stage_raw(
    signals: Sequence[np.ndarray],
    labels: Sequence[str],
    fs: float,
    out_path: Union[str, Path],
    duration_ms: float = 50.0,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Raw signals across stages",
    dpi: int = DEFAULT_DPI,
) -> Path:
    n = int(duration_ms * 1e-3 * fs)
    fig, axes = plt.subplots(len(signals), 1, figsize=(8.0, 2.4 * len(signals)),
                              sharex=True, sharey=True)
    if len(signals) == 1:
        axes = [axes]
    colors = ["#27ae60", "#f39c12", "#c0392b"]
    for ax, sig, lbl, col in zip(axes, signals, labels, colors):
        mm = min(n, len(sig))
        t = np.arange(mm) / fs * 1e3
        ax.plot(t, sig[:mm], linewidth=0.7, color=col)
        ax.set_ylabel(f"{lbl}\nAmplitude (g)")
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_three_stage_entropy(
    distributions: Sequence[np.ndarray],
    labels: Sequence[str],
    out_path: Union[str, Path],
    title: str = "Entropy distribution per stage",
    dpi: int = DEFAULT_DPI,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    parts = ax.violinplot([np.asarray(d, dtype=np.float64) for d in distributions],
                          showmeans=True, showmedians=False)
    colors = ["#27ae60", "#f39c12", "#c0392b"]
    for body, c in zip(parts['bodies'], colors[:len(distributions)]):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalised Slope Entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_sensitivity(
    csv_paths: Sequence[Path],
    panel_labels: Sequence[str],
    change_points: Sequence[Optional[int]],
    out_path: Union[str, Path],
    title: str = "Parameter sensitivity",
    dpi: int = DEFAULT_DPI,
) -> Path:
    fig, axes = plt.subplots(len(csv_paths), 1, figsize=(9.0, 2.4 * len(csv_paths)),
                              sharex=True)
    if len(csv_paths) == 1:
        axes = [axes]
    for ax, csvp, lbl, cp in zip(axes, csv_paths, panel_labels, change_points):
        idx, mean_H, q25_H, q75_H, _ = _read_timeline_csv(csvp)
        ax.fill_between(idx, q25_H, q75_H, alpha=0.25, color="#1f4e79")
        ax.plot(idx, mean_H, color="#1f4e79", linewidth=1.0)
        if cp is not None and cp < len(idx):
            ax.axvline(idx[cp], color="#c0392b", linewidth=1.2,
                       label=f"i* = file {idx[cp]}")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("File index (chronological)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_perf(
    perf_rows: Sequence[Tuple[str, float, float]],
    out_path: Union[str, Path],
    title: str = "Per-file processing time",
    dpi: int = DEFAULT_DPI,
) -> Path:
    labels = [r[0] for r in perf_rows]
    means = np.array([r[1] for r in perf_rows])
    stds = np.array([r[2] for r in perf_rows])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, color="#1f4e79", alpha=0.85, capsize=4)
    ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.0,
               label="Real-time threshold (1 s per 1 s file)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean processing time per file (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()
