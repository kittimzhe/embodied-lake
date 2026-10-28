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


def render_comparison(
    a: str | Path,
    b: str | Path,
    out: str | Path | None = None,
    episode: int = 0,
    max_frames: int = 5000,
) -> Path:
    """Render two v3.0 datasets into ONE .rrd for visual comparison.

    Entity tree: ``compare/<label>/<feature>/<dim>`` plus
    ``compare/delta/<feature>/<dim>`` (a - b, should be flat when a converted
    output matches its reference). Shared ``timestamp`` timeline — same trick
    posted as a workaround recipe on rerun-io/rerun#12984.
    """
    root_a, root_b = Path(a), Path(b)

    def load(root: Path) -> dict:
        eps_tbl = pq.read_table(
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ).to_pydict()
        for row in range(len(eps_tbl["episode_index"])):
            if eps_tbl["episode_index"][row] != episode:
                continue
            p = root / "data" / f"chunk-{eps_tbl['data/chunk_index'][row]:03d}" / \
                f"file-{eps_tbl['data/file_index'][row]:03d}.parquet"
            tbl = pq.read_table(p)
            d = tbl.to_pydict()
            # 对齐帧数(截到两者较短处,由调用侧统一)
            return d
        raise ValueError(f"episode {episode} not found in {root}")

    da, db = load(root_a), load(root_b)
    n = min(len(da["index"]), len(db["index"]), max_frames)
    if len(da["index"]) != len(db["index"]):
        import sys
        print(f"note: frame counts differ ({len(da['index'])} vs {len(db['index'])}); "
              f"rendering first {n}", file=sys.stderr)
    shared = [c for c in da if c in db and c.startswith(("observation.", "action."))]

    rr.init("embodied_lake.dataset-comparison")
    for feat in shared:
        va, vb = da[feat][:n], db[feat][:n]
        width = len(va[0]) if va and isinstance(va[0], (list, tuple)) else 1
        ts = da["timestamp"][:n]
        for dim in range(width):
            sa = [v[dim] if width > 1 else v for v in va]
            sb = [v[dim] if width > 1 else v for v in vb]
            idx = [rr.TimeColumn("timestamp", timestamp=ts)]
            comp = rr.ComponentColumn("scalar", rr.components.ScalarBatch(sa))
            rr.send_columns(f"compare/{feat}/{dim}/a", indexes=idx, columns=[comp])
            comp = rr.ComponentColumn("scalar", rr.components.ScalarBatch(sb))
            rr.send_columns(f"compare/{feat}/{dim}/b", indexes=idx, columns=[comp])
            delta = [x - y for x, y in zip(sa, sb)]
            comp = rr.ComponentColumn("scalar", rr.components.ScalarBatch(delta))
            rr.send_columns(f"compare/{feat}/{dim}/delta", indexes=idx, columns=[comp])

    out_path = Path(out) if out else root_a.parent / f"compare_{root_a.name}_vs_{root_b.name}.rrd"
    rr.save(str(out_path))
    return out_path
