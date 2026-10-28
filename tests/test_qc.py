"""Tests for the structural QC module."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30
from embodied_lake.qc.structural import qc_dataset

from test_mcap_to_lerobot import STATE_TOPIC, _write_mcap


@pytest.fixture(scope="module")
def good_dataset(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("qc")
    _write_mcap(tmp / "a.mcap", 8, 0, action=False)
    _write_mcap(tmp / "b.mcap", 6, 1_000_000_000, action=False)
    out = tmp / "ds"
    convert_mcap_to_v30(
        [tmp / "a.mcap", tmp / "b.mcap"], out, McapMapping(state_topic=STATE_TOPIC)
    )
    return out


def test_qc_passes_on_converter_output(good_dataset: Path) -> None:
    report = qc_dataset(good_dataset)
    assert report.ok, [str(i) for i in report.issues]
    assert report.episodes_checked == 2


def test_qc_catches_missing_data_file(good_dataset: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(good_dataset, broken)
    (broken / "data" / "chunk-000" / "file-001.parquet").unlink()
    report = qc_dataset(broken)
    assert not report.ok
    codes = {i.code for i in report.errors}
    assert "data_file_missing" in codes
    assert "frames_count" in codes


def test_qc_catches_frames_mismatch(good_dataset: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken2"
    shutil.copytree(good_dataset, broken)
    import json

    info_path = broken / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 999  # wrong
    info_path.write_text(json.dumps(info))
    report = qc_dataset(broken)
    assert not report.ok
    assert any(i.code == "frames_count" for i in report.errors)


def test_qc_catches_split_mismatch(good_dataset: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken3"
    shutil.copytree(good_dataset, broken)
    import json

    info_path = broken / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": "0:5"}  # wrong end
    info_path.write_text(json.dumps(info))
    report = qc_dataset(broken)
    assert any(i.code == "splits" for i in report.errors)
