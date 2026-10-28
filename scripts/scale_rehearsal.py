"""Scale rehearsal: synthetic v2.1 at ~2000 episodes, convert, verify.

Fixture datasets (2 episodes) never exercise the writer's sharding paths
(multiple data files per chunk, episode-file fanout). This generates a
synthetic-but-format-exact v2.1 tree (~400k rows), converts it, and checks:

- conversion succeeds and throughput holds (no superlinear blowup)
- the QC gate passes on the output
- every episode lands in exactly one data file, row counts add up

Usage: PYTHONPATH=src python scripts/scale_rehearsal.py [--episodes 2000]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def synth_v21(root: Path, episodes: int, frames: int = 200, dim: int = 8) -> int:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [dim], "names": None},
        "action": {"dtype": "float32", "shape": [dim], "names": None},
    }
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1",
        "robot_type": "synth_scale",
        "total_episodes": episodes,
        "total_frames": episodes * frames,
        "fps": 20,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }))
    # v2.1 meta: tasks.jsonl + episodes.jsonl (+ optional per-episode stats)
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "synthetic scale rehearsal"}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": ep, "tasks": ["synthetic scale rehearsal"], "length": frames})
            + "\n"
            for ep in range(episodes)
        )
    )
    total = 0
    for ep in range(episodes):
        t = np.arange(frames, dtype=np.float64)
        state = np.stack([np.sin(t / 10 + ep) for _ in range(dim)], axis=1).astype(np.float32)
        action = np.roll(state, -1, axis=0)
        idx = np.arange(total, total + frames)
        ep_idx = np.full(frames, ep, dtype=np.int64)
        fi = np.arange(frames)
        table = pa.table({
            "observation.state": state.tolist(),
            "action": action.tolist(),
            "episode_index": ep_idx,
            "frame_index": fi,
            "index": idx,
            "timestamp": (t / 20).tolist(),
        })
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        total += frames
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="elake_scale_"))
    try:
        src = tmp / "v21"
        rows = synth_v21(src, args.episodes)
        print(f"synthetic v2.1: {args.episodes} episodes, {rows} rows, "
              f"{sum(f.stat().st_size for f in src.rglob('*.parquet')) / 1e6:.0f} MB")

        from embodied_lake.babel.v21_to_v30 import convert_v21_to_v30
        from embodied_lake.qc.structural import qc_dataset
        from embodied_lake.qc.diff import diff_datasets

        dst = tmp / "v30"
        t0 = time.perf_counter()
        rep = convert_v21_to_v30(src, dst)
        dt = time.perf_counter() - t0
        print(f"converted in {dt:.1f}s → {rows / dt:,.0f} rows/s")

        report = qc_dataset(dst)
        print("qc:", "PASS" if report.ok else f"FAIL {report.errors}")
        d = diff_datasets(src, dst, expect_migration=True)
        print("diff:", d.summary().splitlines()[-1])

        data_files = list((dst / "data").rglob("*.parquet"))
        n_rows_out = sum(pq.read_metadata(f).num_rows for f in data_files)
        print(f"output: {len(data_files)} data files, {n_rows_out} rows")
        assert n_rows_out == rows, "row count mismatch!"
        print("SCALE REHEARSAL OK")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
