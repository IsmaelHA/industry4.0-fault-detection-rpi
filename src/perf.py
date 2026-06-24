"""Timing, memory probes, and environment capture (Linux/Pi)."""

from __future__ import annotations

import gc
import platform
import sys
from pathlib import Path
from time import perf_counter


def _peak_rss_kb() -> int:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0


class Stopwatch:
    def __init__(self, label: str = "") -> None:
        self.label = label

    def __enter__(self):
        gc.collect()
        self.t0 = perf_counter()
        self.r0 = _peak_rss_kb()
        return self

    def __exit__(self, *_):
        self.dt = perf_counter() - self.t0
        self.peak_rss = _peak_rss_kb()
        self.d_rss = self.peak_rss - self.r0


def capture_environment(out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    u = platform.uname()
    lines.append(f"system          : {u.system}")
    lines.append(f"node            : {u.node}")
    lines.append(f"release         : {u.release}")
    lines.append(f"version         : {u.version}")
    lines.append(f"machine         : {u.machine}")
    lines.append(f"processor       : {u.processor}")
    lines.append(f"python          : {sys.version.split()[0]}")
    lines.append(f"python_impl     : {platform.python_implementation()}")

    try:
        model = Path("/proc/device-tree/model").read_text(errors="replace").strip("\x00 \n\t")
        lines.append(f"pi_model        : {model}")
    except Exception:
        lines.append("pi_model        : (unavailable)")

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace")
        for kw in ("Model", "Hardware", "Revision"):
            for line in cpuinfo.splitlines():
                if line.startswith(kw):
                    lines.append(f"{kw.lower():<15} : {line.split(':', 1)[1].strip()}")
                    break
    except Exception:
        pass

    try:
        meminfo = Path("/proc/meminfo").read_text(errors="replace")
        for line in meminfo.splitlines()[:3]:
            lines.append(f"meminfo         : {line.strip()}")
    except Exception:
        pass

    for pkg in ("numpy", "scipy", "matplotlib", "yaml"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            lines.append(f"{pkg:<15} : {ver}")
        except Exception:
            lines.append(f"{pkg:<15} : not installed")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
