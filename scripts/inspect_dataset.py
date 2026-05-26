"""Walk an IMS dataset directory and report files / channels / time span per test.

Usage: python scripts/inspect_dataset.py [path_to_IMS_root]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np


TEST_HINTS = {
    "1st_test": {"channels": 8, "bearings": 4, "channels_per_bearing": 2,
                  "fault_location": "inner race / roller on B3, B4"},
    "2nd_test": {"channels": 4, "bearings": 4, "channels_per_bearing": 1,
                  "fault_location": "outer race on bearing 1"},
    "3rd_test": {"channels": 4, "bearings": 4, "channels_per_bearing": 1,
                  "fault_location": "outer race on bearing 3"},
    "4th_test": {"channels": 4, "bearings": 4, "channels_per_bearing": 1,
                  "fault_location": "outer race on bearing 3 (alt naming)"},
}


def parse_timestamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        return None


def inspect_test(test_dir: Path) -> None:
    files = sorted(p for p in test_dir.iterdir() if p.is_file())
    n = len(files)
    if n == 0:
        print(f"  ! Empty: {test_dir.name}")
        return

    try:
        first = np.loadtxt(files[0])
    except Exception as exc:
        print(f"  ! Could not parse {files[0].name}: {exc}")
        return
    if first.ndim == 1:
        n_channels, n_samples = 1, len(first)
    else:
        n_samples, n_channels = first.shape

    t0 = parse_timestamp(files[0].name)
    t1 = parse_timestamp(files[-1].name)
    span_h = ((t1 - t0).total_seconds() / 3600.0) if (t0 and t1) else None

    hint = TEST_HINTS.get(test_dir.name, {})

    print(f"  {test_dir.name}:")
    print(f"      files            : {n}")
    print(f"      samples per file : {n_samples}")
    print(f"      channels         : {n_channels}"
          + (f"   (expected: {hint.get('channels')})" if 'channels' in hint else ""))
    print(f"      first file       : {files[0].name}")
    print(f"      last  file       : {files[-1].name}")
    if span_h is not None:
        print(f"      time span        : {span_h:.1f} hours ({span_h / 24:.1f} days)")
    if "fault_location" in hint:
        print(f"      documented fault : {hint['fault_location']}")
    if "channels_per_bearing" in hint:
        cpb = hint["channels_per_bearing"]
        print(f"      bearing -> column mapping (0-based):")
        for b in range(hint.get("bearings", 0)):
            cols = ", ".join(str(b * cpb + k) for k in range(cpb))
            tag = "   <- Bearing 1" if b == 0 else ""
            print(f"          bearing {b+1}: column(s) {cols}{tag}")
    print()


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("./data/raw")
    root = root.resolve()
    print(f"Inspecting: {root}\n")

    if not root.exists():
        print(f"ERROR: path does not exist: {root}")
        return 1

    test_dirs = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                for f in child.iterdir():
                    if f.is_file() and parse_timestamp(f.name) is not None:
                        test_dirs.append(child)
                        break

    if not test_dirs:
        if any(parse_timestamp(p.name) for p in root.iterdir() if p.is_file()):
            test_dirs = [root]
        else:
            print("No IMS-style data directories found.")
            print("Expected sub-folders such as 1st_test, 2nd_test, 3rd_test.")
            return 1

    for d in test_dirs:
        inspect_test(d)

    print("Done. Recommended: use 2nd_test, channel 0 (Bearing 1, outer-race fault).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
