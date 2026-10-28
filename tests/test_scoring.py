"""Tests for the rule-based quality scorer."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_lake.qc.scoring import score_dataset, score_episode
import numpy as np


def _make_v30(tmp: Path, episodes: list[dict[str, list]]) -> Path:
    """episodes: list of {"observation.state": rows, "action": rows}"""
    root = tmp / "ds"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    rows_meta, g = [], 0
    for i, ep in enumerate(episodes):
        n = len(ep["observation.state"])
        tbl = pa.table({
            "timestamp": pa.array([k / 30.0 for k in range(n)], type=pa.float32()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "episode_index": pa.array([i] * n, type=pa.int64()),
            "index": pa.array(list(range(g, g + n)), type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
            "observation.state": pa.array(ep["observation.state"], type=pa.list_(pa.float32())),
            "action": pa.array(ep["action"], type=pa.list_(pa.float32())),
        })
        pq.write_table(tbl, root / "data" / "chunk-000" / f"file-{i:03d}.parquet")
        rows_meta.append({
            "episode_index": i, "tasks": ["t"], "length": n,
            "data/chunk_index": 0, "data/file_index": i,
            "dataset_from_index": g, "dataset_to_index": g + n,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        })
        g += n
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_meta),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(__import__("json").dumps({
        "codebase_version": "v3.0", "total_episodes": len(episodes),
        "total_frames": g, "fps": 30, "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": None, "fps": 30},
            "action": {"dtype": "float32", "shape": [2], "names": None, "fps": 30},
        },
    }))
    return root


def _healthy_ep(n=40, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 2)).astype(np.float32).tolist()
    return {"observation.state": state, "action": state}


def test_healthy_episodes_score_full(tmp_path: Path) -> None:
    root = _make_v30(tmp_path, [_healthy_ep(40, s) for s in range(4)])
    rep = score_dataset(root)
    assert len(rep.episodes) == 4
    assert all(e.score == 100.0 for e in rep.episodes), [(e.score, e.findings) for e in rep.episodes]


def test_frozen_joint_detected(tmp_path: Path) -> None:
    bad = _healthy_ep(40, 1)
    frozen = [[v[0], 5.0] for v in bad["observation.state"]]
    root = _make_v30(tmp_path, [
        _healthy_ep(40, 0), {"observation.state": frozen, "action": frozen},
    ])
    rep = score_dataset(root)
    bad_ep = rep.episodes[1]
    assert any(f.check == "frozen_dim" for f in bad_ep.findings)
    assert bad_ep.score < 100


def test_saturation_detected(tmp_path: Path) -> None:
    stuck = [[7.5, float(i % 3)] for i in range(30)]  # dim0 pinned at 7.5
    root = _make_v30(tmp_path, [
        _healthy_ep(40, 0), {"observation.state": stuck, "action": stuck},
    ])
    rep = score_dataset(root)
    assert any(f.check == "saturated" for f in rep.episodes[1].findings)


def test_low_motion_detected(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    near_zero = (rng.normal(scale=1e-9, size=(40, 2))).astype(np.float32).tolist()
    root = _make_v30(tmp_path, [
        _healthy_ep(40, 0), _healthy_ep(40, 1),
        {"observation.state": near_zero, "action": near_zero},
    ])
    rep = score_dataset(root)
    assert any(f.check == "low_motion" for f in rep.episodes[2].findings)
