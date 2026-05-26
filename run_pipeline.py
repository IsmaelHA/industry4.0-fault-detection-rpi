"""Pipeline entry point. Reads every parameter from config.yaml."""

from __future__ import annotations

from pathlib import Path

from src.io_ims import SAMPLING_RATE_HZ, basic_statistics, list_files, load_signal
from src.plots import plot_raw_signal
from src.utils import load_config, set_seed


def main() -> None:
    cfg = load_config("config.yaml")
    _ = set_seed(cfg["seed"])

    data_root = Path(cfg["data_root"])
    results_dir = Path(cfg["results_dir"]) / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(data_root)
    print(f"[Phase 1] Found {len(files)} files in {data_root}")
    first_file = files[0]
    print(f"[Phase 1] Loading first file: {first_file.name}")

    signal = load_signal(first_file, channel=cfg["channel"])
    stats = basic_statistics(signal)
    print(
        f"[Phase 1] length={stats['n_samples']} samples "
        f"({stats['duration_s']:.2f} s @ {SAMPLING_RATE_HZ} Hz)"
    )
    print(
        f"[Phase 1] min={stats['min']:.4f} g, max={stats['max']:.4f} g, "
        f"mean={stats['mean']:.4f} g, std={stats['std']:.4f} g, "
        f"RMS={stats['rms']:.4f} g"
    )

    out_path = results_dir / "F1_raw_healthy.png"
    plot_raw_signal(
        signal=signal,
        fs=SAMPLING_RATE_HZ,
        out_path=out_path,
        title="F1 — Raw vibration signal (healthy stage, file 1)",
        duration_ms=cfg["figure"]["duration_ms"],
        dpi=cfg["figure"]["dpi"],
    )
    print(f"[Phase 1] Saved: {out_path}")


if __name__ == "__main__":
    main()
