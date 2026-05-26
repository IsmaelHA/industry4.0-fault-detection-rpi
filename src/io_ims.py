"""I/O for the NASA IMS Bearing Dataset (1 s of 20 kHz vibration per file)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

SAMPLING_RATE_HZ: int = 20_000
SAMPLES_PER_FILE: int = 20_480


def list_files(data_root: Union[str, Path]) -> List[Path]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_root.resolve()}\n"
            "Place the unpacked IMS files at the path defined in config.yaml -> data_root."
        )
    files = sorted(p for p in data_root.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files inside data directory: {data_root.resolve()}")
    return files


def load_signal(path: Union[str, Path], channel: int = 0) -> np.ndarray:
    path = Path(path)
    arr = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    if arr.ndim == 1:
        return arr
    if channel < 0 or channel >= arr.shape[1]:
        raise IndexError(
            f"Channel {channel} requested but file has only {arr.shape[1]} channels: {path.name}"
        )
    return arr[:, channel]


def basic_statistics(signal: np.ndarray) -> dict:
    return {
        "n_samples": int(signal.size),
        "duration_s": float(signal.size / SAMPLING_RATE_HZ),
        "min": float(signal.min()),
        "max": float(signal.max()),
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
    }
