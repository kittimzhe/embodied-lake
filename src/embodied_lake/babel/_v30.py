"""Shared helpers for writing LeRobot v3.0 dataset directories.

Used by the babel converters (v21_to_v30, mcap_to_lerobot) so every path in
the platform emits structurally identical, upstream-compatible artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

V30 = "v3.0"
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def compute_feature_stats(values: np.ndarray) -> dict:
    """Stats over a 2-D (frames, dims) float array, flattened per feature.

    Matches the stat set observed in real v3.0 datasets on the Hub:
    min/max/mean/std/count + q01/q10/q50/q90/q99 (population std).
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    stats = {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "count": int(flat.size),
    }
    for name, q in QUANTILES.items():
        stats[name] = float(np.quantile(flat, q))
    return stats


def write_tasks(root: Path, tasks: list[str]) -> None:
    """Upstream-compatible tasks.parquet (task strings as named pandas index)."""
    import pandas as pd

    (root / "meta").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")
    ).to_parquet(root / "meta" / "tasks.parquet")


def write_episodes(root: Path, episodes_meta: list[dict]) -> None:
    out = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episodes_meta), out)


def write_stats(root: Path, aggregate: dict) -> None:
    (root / "meta" / "stats.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_info(
    root: Path,
    *,
    features: dict,
    total_episodes: int,
    total_frames: int,
    fps: int,
    robot_type: str | None = None,
) -> None:
    """Write an upgraded v3.0 info.json (video-free datasets)."""
    info: dict = {
        "codebase_version": V30,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": None,  # filled by caller via _finalize if needed
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": None,
        "data_path": DEFAULT_DATA_PATH,
        "video_path": None,
    }
    info["features"] = {
        name: {**ft, "fps": int(fps)} for name, ft in features.items() if ft.get("dtype") != "video"
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return None


def stats_to_flat_columns(feature: str, stats: dict) -> dict[str, object]:
    return {f"stats/{feature}/{k}": v for k, v in stats.items()}
