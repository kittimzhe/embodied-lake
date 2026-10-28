"""Render a local LeRobot v3.0 dataset to a standalone Rerun .rrd file.

Thin wrapper around the same logic as ``scripts/visualize_v30.py`` so it can
be reached from the CLI as ``elake preview``. Requires ``rerun-sdk``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

import rerun as rr


def render_dataset(dataset: str | Path, out: str | Path | None = None, episodes=None) -> Path:
    root = Path(dataset)
    eps_tbl = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    wanted = set(episodes) if episodes is not None else None

    rr.init("embodied_lake.dataset-preview")
    out_path = Path(out) if out else root.parent / f"{root.name}.rrd"

    n = 0
    for row in range(len(eps_tbl["episode_index"])):
        ep = eps_tbl["episode_index"][row]
        if wanted is not None and ep not in wanted:
            continue
        chunk = eps_tbl["data/chunk_index"][row]
        file_idx = eps_tbl["data/file_index"][row]
        path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
        tbl = pq.read_table(path)
        cols = tbl.column_names
        d = tbl.select(
            [c for c in cols if c in ("timestamp", "episode_index")]
            + [c for c in cols if c.startswith(("observation.", "action."))]
        ).to_pydict()
        rr.log(
            f"episode_{ep:06d}",
            rr.TextLog(f"task={eps_tbl['tasks'][row]} length={eps_tbl['length'][row]}"),
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
        n += 1
    rr.save(str(out_path))
    return out_path
