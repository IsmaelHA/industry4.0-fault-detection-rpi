"""Write results/environment.txt with hardware/software info."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.perf import capture_environment
from src.runtime import load_config


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    out = Path(cfg["results_dir"]) / "environment.txt"
    capture_environment(out)
    print(f"[env] Wrote {out}")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
