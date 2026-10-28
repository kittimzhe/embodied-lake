"""Visualize a (video-free) LeRobot v3.0 dataset in Rerun.

Reads the parquet shards directly (pyarrow only, no lerobot install) and logs
per-episode state/action traces as scalar time series, saving an .rrd you can
open with the Rerun viewer (`rerun out.rrd`). This is the Phase-2 episode
preview groundwork: dataset → Rerun without any heavy dependencies.

Usage:
    python scripts/visualize_v30.py DATASET_DIR [--out out.rrd] [--episodes 0 1 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

import rerun as rr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="local LeRobot v3.0 dataset directory")
    ap.add_argument("--out", default=None, help="output .rrd path (default: <dataset>.rrd)")
    ap.add_argument("--episodes", nargs="*", type=int, default=None,
                    help="episode indices to include (default: all)")
    args = ap.parse_args()

    root = Path(args.dataset)
    eps_tbl = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    wanted = set(args.episodes) if args.episodes is not None else None

    rr.init("embodied-lake.dataset-preview")
    out = Path(args.out) if args.out else root.parent / f"{root.name}.rrd"

    n_logged = 0
    for row in range(len(eps_tbl["episode_index"])):
        ep = eps_tbl["episode_index"][row]
        if wanted is not None and ep not in wanted:
            continue
        chunk = eps_tbl["data/chunk_index"][row]
        file_idx = eps_tbl["data/file_index"][row]
        path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
        tbl = pq.read_table(path)
        cols = tbl.column_names
        if "timestamp" not in cols or "episode_index" not in cols:
            print(f"skip {path.name}: no timestamp/episode_index columns", file=sys.stderr)
            continue
        d = tbl.select(
            [c for c in cols if c in ("timestamp", "episode_index", "frame_index")]
            + [c for c in cols if c.startswith(("observation.", "action."))]
        ).to_pydict()
        tasks = eps_tbl["tasks"][row]
        rr.log(
            f"episode_{ep:06d}",
            rr.TextLog(f"task={tasks} length={eps_tbl['length'][row]}"),
        )
        for feat in [c for c in cols if c.startswith(("observation.", "action."))]:
            vectors = d[feat]
            width = len(vectors[0]) if vectors and isinstance(vectors[0], (list, tuple)) else 1
            for dim in range(width):
                values = [v[dim] if width > 1 else v for v in vectors]
                rr.send_columns(
                    f"episode_{ep:06d}/{feat}/{dim}",
                    indexes=[rr.TimeColumn("timestamp", timestamp=d["timestamp"])],
                    columns=[rr.ComponentColumn("scalar", rr.components.ScalarBatch(values))],
                )
        n_logged += 1

    rr.save(str(out))
    print(f"logged {n_logged} episode(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
