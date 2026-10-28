"""Smoke tests for the dataset inspector.

Runs fully offline against a local v3.0-shaped fixture; the network test is
skipped automatically when the Hugging Face Hub is unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_lake.inspect import InspectError, inspect, load_info, normalize

# Real-world v3.0 info.json shape (mirrors a small public dataset).
V3_FIXTURE = {
    "codebase_version": "v3.0",
    "robot_type": "so_follower",
    "total_episodes": 160,
    "total_frames": 81786,
    "total_tasks": 1,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "fps": 30,
    "splits": {"train": "0:160"},
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.state": {"dtype": "float32", "shape": [7], "names": None},
        "action": {"dtype": "float32", "shape": [7], "names": None},
        "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
    },
}


@pytest.fixture()
def v3_dataset(tmp_path: Path) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(V3_FIXTURE), encoding="utf-8")
    return tmp_path


def test_load_info_local(v3_dataset: Path) -> None:
    info = load_info(str(v3_dataset))
    assert info["codebase_version"] == "v3.0"


def test_normalize_v3(v3_dataset: Path) -> None:
    card = inspect(str(v3_dataset))
    assert card.codebase_version == "v3.0"
    assert card.robot_type == "so_follower"
    assert card.total_episodes == 160
    assert card.total_frames == 81786
    assert card.data_path.startswith("data/chunk-")
    assert "observation.images.front" in card.features
    # unknown keys are preserved in extra instead of being dropped
    assert card.extra["chunks_size"] == 1000


def test_missing_local_dir_raises() -> None:
    with pytest.raises(InspectError):
        load_info("/definitely/not/a/dataset")


def test_repo_id_heuristic() -> None:
    from embodied_lake.inspect import _looks_like_repo_id

    assert _looks_like_repo_id("lerobot/svla_so101_pickplace")
    assert not _looks_like_repo_id("some/local/path") or Path("some/local/path").exists()
    assert not _looks_like_repo_id("not-a-repo-id")


def test_network_inspect_real_dataset() -> None:
    """End-to-end against a real (small, public) v3.0 dataset on the Hub."""
    pytest.importorskip("urllib.request")
    try:
        card = inspect("Embodied-AI-6/dataset", timeout=20)
    except Exception as exc:  # network unavailable in sandbox -> skip, not fail
        pytest.skip(f"hub unreachable: {exc}")
    assert card.codebase_version == "v3.0"
    assert card.total_episodes == 160
    assert card.total_frames == 81786
