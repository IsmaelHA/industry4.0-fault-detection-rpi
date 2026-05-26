"""Matplotlib helpers. Agg backend so the same code works headless on the Pi."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pathlib import Path  # noqa: E402
from typing import Optional, Tuple, Union  # noqa: E402

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
    n_samples = int(duration_ms * 1e-3 * fs)
    n_samples = max(1, min(n_samples, len(signal)))
    t_ms = np.arange(n_samples) / fs * 1e3

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(t_ms, signal[:n_samples], linewidth=0.7, color="#1f4e79")
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
