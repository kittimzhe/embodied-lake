"""End-to-end tests for the v2.1 -> v3.0 converter.

Builds a synthetic (tiny, video-free) v2.1 dataset on disk, converts it, and
asserts structural and data-level equivalence with the v3.0 layout.

Requires pyarrow; runs under the project venv (`.venv/bin/python -m pytest`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from embodied_lake.babel.v21_to_v30 import (  # noqa: E402
    ConversionError,
    aggregate_episode_stats,
    convert_v21_to_v30,
)

LENGTHS = [4, 6, 5]
TASKS = ["pick the cube", "place the cube"]


def _make_episode_table(ep_idx: int, length: int, task_idx: int) -> pa.Table:
    return pa.table(
        {
            "timestamp": pa.array([i / 30.0 for i in range(length)], type=pa.float32()),
            "frame_index": pa.array(list(range(length)), type=pa.int64()),
            "episode_index": pa.array([ep_idx] * length, type=pa.int64()),
            "index": pa.array(list(range(ep_idx * 100, ep_idx * 100 + length)), type=pa.int64()),
            "task_index": pa.array([task_idx] * length, type=pa.int64()),
            "observation.state": pa.array(
                [[float(i % 3)] for i in range(length)], type=pa.list_(pa.float32())
            ),
            "action": pa.array(
                [[float(i % 2)] for i in range(length)], type=pa.list_(pa.float32())
            ),
        }
    )


FEATURES = {
    "observation.state": {"dtype": "float32", "shape": [1], "names": None},
    "action": {"dtype": "float32", "shape": [1], "names": None},
}


@pytest.fixture()
def v21_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds_v21"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    episodes_rows, stats_rows = [], []
    for ep_idx, length in enumerate(LENGTHS):
        task_idx = ep_idx % 2
        tbl = _make_episode_table(ep_idx, length, task_idx)
        pq.write_table(tbl, root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        episodes_rows.append(
            {"episode_index": ep_idx, "tasks": [TASKS[task_idx]], "length": length}
        )
        stats_rows.append(
            {
                "episode_index": ep_idx,
                "stats": {
                    "observation.state": {
                        "min": 0.0, "max": 2.0, "mean": 1.0, "std": 0.5,
                        "count": length, "q01": 0.1, "q10": 0.5, "q50": 1.0, "q90": 1.5, "q99": 1.9,
                    },
                    "action": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.25, "count": length},
                },
            }
        )

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "so_follower",
                "total_episodes": len(LENGTHS),
                "total_frames": sum(LENGTHS),
                "total_tasks": len(TASKS),
                "fps": 30.0,
                "splits": {"train": f"0:{len(LENGTHS)}"},
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk}/episode_{episode_index:06d}.parquet",
                "video_path": None,
                "features": FEATURES,
                "total_chunks": 1,
                "total_videos": 0,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(TASKS)),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in episodes_rows), encoding="utf-8"
    )
    (root / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in stats_rows), encoding="utf-8"
    )
    return root


def test_convert_single_file(v21_dataset: tmp_path.__class__ if False else Path, tmp_path: Path) -> None:
    out = tmp_path / "ds_v30"
    report = convert_v21_to_v30(v21_dataset, out)

    assert report.episodes == 3
    assert report.frames == sum(LENGTHS)
    assert report.tasks == 2
    assert report.has_stats

    # v3.0 layout
    assert (out / "meta" / "info.json").is_file()
    assert (out / "meta" / "tasks.parquet").is_file()
    assert (out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").is_file()
    assert (out / "data" / "chunk-000" / "file-000.parquet").is_file()

    # data identity: rows equal to source episodes concatenated in order
    src_tbl = pa.concat_tables(
        [
            pq.read_table(v21_dataset / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
            for i in range(3)
        ]
    )
    dst_tbl = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    assert dst_tbl.num_rows == src_tbl.num_rows == sum(LENGTHS)
    assert dst_tbl.to_pydict() == src_tbl.to_pydict()

    # episodes metadata ranges
    eps = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    assert eps["episode_index"] == [0, 1, 2]
    assert eps["dataset_from_index"] == [0, 4, 10]
    assert eps["dataset_to_index"] == [4, 10, 15]
    assert eps["length"] == LENGTHS
    assert eps["tasks"][0] == [TASKS[0]]
    assert eps["data/file_index"] == [0, 0, 0]
    # flattened per-episode stats present, including quantile pass-through
    assert eps["stats/action/count"] == LENGTHS
    assert eps["stats/observation.state/q50"] == [1.0, 1.0, 1.0]
    # upstream constant columns for the episodes metadata artifact
    assert eps["meta/episodes/chunk_index"] == [0, 0, 0]
    assert eps["meta/episodes/file_index"] == [0, 0, 0]

    # tasks parquet: upstream-compatible pandas index layout when pandas present
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        df = pd.read_parquet(out / "meta" / "tasks.parquet")
        assert df.index.name == "task"
        assert df.loc[TASKS[0], "task_index"] == 0
        assert df.loc[TASKS[1], "task_index"] == 1

    # tasks parquet
    tasks = pq.read_table(out / "meta" / "tasks.parquet").to_pydict()
    assert sorted(zip(tasks["task_index"], tasks["task"])) == [(0, TASKS[0]), (1, TASKS[1])]

    # info.json upgraded
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert "total_chunks" not in info and "total_videos" not in info
    assert info["fps"] == 30  # int
    assert info["data_path"].startswith("data/chunk-")
    assert info["video_path"] is None
    assert info["features"]["action"]["fps"] == 30
    assert info["splits"] == {"train": "0:3"}

    # aggregate stats file
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["action"]["count"] == sum(LENGTHS)
    assert stats["action"]["min"] == 0.0 and stats["action"]["max"] == 1.0


def test_convert_splits_files_on_size_threshold(v21_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "ds_v30_tiny"
    report = convert_v21_to_v30(v21_dataset, out, data_file_size_in_mb=1e-9)
    # threshold smaller than any episode -> every episode lands in its own file
    assert report.data_files_written == 3
    for i in range(3):
        assert (out / "data" / "chunk-000" / f"file-{i:03d}.parquet").is_file()
    eps = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    assert eps["data/file_index"] == [0, 1, 2]


def test_convert_rejects_videos(v21_dataset: Path, tmp_path: Path) -> None:
    info_path = v21_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.front"] = {"dtype": "video", "shape": [3, 480, 640]}
    info_path.write_text(json.dumps(info))
    with pytest.raises(NotImplementedError, match="video"):
        convert_v21_to_v30(v21_dataset, tmp_path / "never")


def test_convert_rejects_wrong_version(v21_dataset: Path, tmp_path: Path) -> None:
    info_path = v21_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ConversionError, match="codebase_version"):
        convert_v21_to_v30(v21_dataset, tmp_path / "never")


def test_aggregate_stats_weighted() -> None:
    agg = aggregate_episode_stats(
        [
            {"f": {"mean": 0.0, "std": 0.0, "count": 4, "min": -1.0, "max": 1.0}},
            {"f": {"mean": 2.0, "std": 0.0, "count": 6, "min": 1.0, "max": 3.0}},
        ]
    )
    assert agg["f"]["count"] == 10
    assert agg["f"]["mean"] == pytest.approx(1.2)
    assert agg["f"]["min"] == -1.0 and agg["f"]["max"] == 3.0
    # combined population variance of {-1x4? no: ep0 all-0 x4, ep1 all-2 x6}
    assert agg["f"]["std"] == pytest.approx(0.9797958971)  # sqrt(0.96)


def test_convert_drop_videos(v21_dataset: Path, tmp_path: Path):
    """video features normally fail fast; drop_videos=True converts tabular only."""
    import shutil as _shutil

    src = tmp_path / "src"
    _shutil.copytree(v21_dataset, src)
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.image"] = {"dtype": "video", "shape": [3, 480, 640], "info": {"video_path": "videos/observation.image", "fps": 30, "pixels_paths": None}}
    info["video_path"] = "videos/{video_key}/episode_{episode_index:06d}.mp4"
    info["total_videos"] = 2
    info_path.write_text(json.dumps(info))

    with pytest.raises(NotImplementedError):
        convert_v21_to_v30(src, tmp_path / "never")

    out = tmp_path / "out"
    report = convert_v21_to_v30(src, out, drop_videos=True)
    assert any("dropped" in n for n in report.notes)
    new_info = json.loads((out / "meta" / "info.json").read_text())
    assert "observation.image" not in new_info["features"]
    assert new_info["video_path"] is None
    assert new_info["video_files_size_in_mb"] is None


def test_aggregate_vector_stats_elementwise():
    """Real v2.1 datasets store vector-feature stats as per-dim lists (bugfix)."""
    from embodied_lake.babel.v21_to_v30 import aggregate_episode_stats

    all_stats = [
        {  # ep0: 2 frames; state dim-2
            "observation.state": {
                "min": [0.0, 10.0], "max": [2.0, 12.0],
                "mean": [1.0, 11.0], "std": [0.5, 0.5], "count": [2],
            }
        },
        {  # ep1: 6 frames
            "observation.state": {
                "min": [4.0, 14.0], "max": [6.0, 16.0],
                "mean": [5.0, 15.0], "std": [0.5, 0.5], "count": [6],
            }
        },
    ]
    agg = aggregate_episode_stats(all_stats)["observation.state"]
    assert agg["min"] == [0.0, 10.0]
    assert agg["max"] == [6.0, 16.0]
    assert agg["count"] == 8
    # weighted mean: (1*2+5*6)/8 = 4, (11*2+15*6)/8 = 14
    assert agg["mean"] == [4.0, 14.0]
    import math
    # combining the CLAIMED per-episode stds: E[x2] = ((0.25+1)*2 + (0.25+25)*6)/8
    # = 19.25; var = 19.25 - 4**2 = 3.25 -> std = sqrt(3.25)
    assert all(math.isclose(v, math.sqrt(3.25), rel_tol=1e-9) for v in agg["std"])
