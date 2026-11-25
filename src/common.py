"""Small shared helpers: result files, timing, memory, plotting style."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from config import FIG, RES

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

PALETTE = {"blue": "#2f6db5", "orange": "#d9822b", "teal": "#2a9d8f", "red": "#c0392b",
           "grey": "#7f8c8d", "purple": "#7b5ea7"}


def save_result(name: str, obj) -> None:
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def load_result(name: str, default=None):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def median_time(fn, repeat: int = 3, warmup: bool = True):
    """Median wall time (s) of fn(); the first (cold) call is discarded when warmup=True."""
    if warmup:
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Linux/macOS: resource, Windows: psutil)."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024
    except ImportError:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / (1024 * 1024)


def size_mb(path: Path) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def style_axes():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def savefig(fig, name: str) -> str:
    p = FIG / f"{name}.png"
    fig.savefig(p)
    return str(p)
