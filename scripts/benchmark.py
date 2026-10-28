"""Benchmark conversion throughput on real datasets.

Measures wall time, rows/s and MB/s for the babel converters, and extrapolates
a DROID-scale (76k trajectories, ~21M frames) estimate to compare with public
reference points (e.g. the LanceDB DROID full conversion at 34 min).

Usage:
    python scripts/benchmark.py v21-to-v30 SRC DST [--drop-videos] [--runs N]
    python scripts/benchmark.py mcap-to-v30 SRC DST --state-topic T [--runs N]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DROID_TRAJECTORIES = 76_000
DROID_FRAMES_APPROX = 21_000_000  # ~276 frames/traj average


def bench(fn, *, runs: int = 3) -> float:
    """Run fn() `runs` times (fresh dst each run), return best wall seconds."""
    best = None
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def fmt(n: float) -> str:
    return f"{n:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("converter", choices=["v21-to-v30", "mcap-to-v30"])
    ap.add_argument("src", nargs="+")
    ap.add_argument("dst")
    ap.add_argument("--state-topic")
    ap.add_argument("--drop-videos", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    src = args.src[0] if len(args.src) == 1 else args.src
    src_path = Path(src if isinstance(src, str) else src[0])
    src_bytes = sum(
        p.stat().st_size
        for p in (src_path.rglob("*") if src_path.is_dir() else [src_path])
        if p.is_file()
    )

    if args.converter == "v21-to-v30":
        from embodied_lake.babel.v21_to_v30 import convert_v21_to_v30

        def run() -> None:
            dst = Path(args.dst)
            if dst.exists():
                shutil.rmtree(dst)
            report = convert_v21_to_v30(args.src[0], args.dst, drop_videos=args.drop_videos)
            run.rows = report.frames  # type: ignore[attr-defined]
    else:
        from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30

        mapping = McapMapping(state_topic=args.state_topic or "")

        def run() -> None:
            dst = Path(args.dst)
            if dst.exists():
                shutil.rmtree(dst)
            report = convert_mcap_to_v30(args.src, args.dst, mapping)
            run.rows = report.frames  # type: ignore[attr-defined]

    dt = bench(run, runs=args.runs)
    rows = run.rows
    mb = src_bytes / 1e6
    rows_per_s = rows / dt
    print(f"converter      : {args.converter}")
    print(f"wall (best of {args.runs}): {dt:.2f} s")
    print(f"rows           : {fmt(rows)}  ({rows / dt:,.0f} rows/s)")
    print(f"input size     : {mb:.1f} MB  ({mb / dt:.1f} MB/s)")
    est = DROID_FRAMES_APPROX / rows_per_s / 60
    print(f"DROID extrapolation (~{fmt(DROID_FRAMES_APPROX)} frames, tabular): ~{est:,.0f} min")
    print("reference      : LanceDB full DROID conversion 34 min (incl. video/vector pipeline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
