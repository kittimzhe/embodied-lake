"""Tests for the ALOHA HDF5 -> LeRobot v3.0 converter."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("h5py")

from embodied_lake.babel.aloha_to_lerobot import (  # noqa: E402
    AlohaConversionError,
    AlohaMapping,
    convert_aloha_to_v30,
)
from embodied_lake.inspect import inspect as inspect_dataset  # noqa: E402


def _write_aloha(path: Path, n: int, *, with_qvel: bool = True, with_images: bool = False) -> None:
    rng = np.random.default_rng(seed=42)
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = True
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=rng.normal(size=(n, 7)).astype(np.float32))
        if with_qvel:
            obs.create_dataset("qvel", data=rng.normal(size=(n, 7)).astype(np.float32))
        if with_images:
            imgs = obs.create_group("images")
            imgs.create_dataset(
                "cam_high",
                data=np.zeros((n, 2, 2, 3), dtype=np.uint8),
                compression="gzip",
            )
        f.create_dataset("action", data=rng.normal(size=(n, 7)).astype(np.float32))


@pytest.fixture()
def two_aloha_files(tmp_path: Path) -> list[Path]:
    p1, p2 = tmp_path / "episode_0.hdf5", tmp_path / "episode_1.hdf5"
    _write_aloha(p1, 12, with_images=True)
    _write_aloha(p2, 8)
    return [p1, p2]


def test_convert_aloha_basic(two_aloha_files: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "ds"
    report = convert_aloha_to_v30(two_aloha_files, out, AlohaMapping(fps=50))

    assert report.episodes == 2
    assert report.frames == 20
    assert any("image" in n for n in report.notes)

    d0 = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    assert len(d0["observation.state"]) == 12
    assert len(d0["action"][0]) == 7

    # values preserved
    with h5py.File(two_aloha_files[0], "r") as f:
        expected_qpos = f["observations/qpos"][0]
    assert d0["observation.state"][0] == pytest.approx(expected_qpos, abs=1e-6)

    # qvel not included by default
    assert "observation.velocity" not in d0

    # episodes metadata
    eps = pq.read_table(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    assert eps["dataset_from_index"] == [0, 12]
    assert eps["dataset_to_index"] == [12, 20]

    # dogfood
    card = inspect_dataset(str(out))
    assert card.codebase_version == "v3.0"
    assert card.total_frames == 20
    assert card.robot_type == "aloha"
    assert "observation.state" in card.features and "action" in card.features


def test_convert_aloha_with_qvel(two_aloha_files: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "ds"
    convert_aloha_to_v30(two_aloha_files, out, AlohaMapping(include_qvel=True))
    d0 = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    assert len(d0["observation.velocity"][0]) == 7
    with h5py.File(two_aloha_files[0], "r") as f:
        assert d0["observation.velocity"][3] == pytest.approx(
            f["observations/qvel"][3], abs=1e-6
        )


def test_convert_aloha_missing_action(tmp_path: Path) -> None:
    src = tmp_path / "bad.hdf5"
    with h5py.File(src, "w") as f:
        f.create_dataset("observations/qpos", data=np.zeros((4, 7), dtype=np.float32))
    with pytest.raises(AlohaConversionError, match="missing"):
        convert_aloha_to_v30([src], tmp_path / "never")
