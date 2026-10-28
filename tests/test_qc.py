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


def test_qc_catches_frame_gap_and_fps_drift(tmp_path: Path) -> None:
    """需要 >10 帧的 episode 才触发 fps 检查,单独造一个长 episode。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30

    tmp = tmp_path
    _write_mcap(tmp / "long.mcap", 20, 0, action=False)
    broken = tmp / "ds"
    convert_mcap_to_v30([tmp / "long.mcap"], broken, McapMapping(state_topic=STATE_TOPIC, fps=30))
    p = broken / "data" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(p).to_pydict()
    # 破坏 frame_index 连续性 + 拉大时间戳间隔(模拟丢帧/漂移)
    tbl["frame_index"] = [0, 2] + tbl["frame_index"][2:]  # gap
    tbl["timestamp"] = [t * 5 for t in tbl["timestamp"]]  # 5x slower than fps
    pq.write_table(pa.table(tbl), p)
    report = qc_dataset(broken)
    codes = {i.code for i in report.issues}
    assert "frame_index_gap" in codes
    assert "fps_drift" in codes


def test_qc_passes_real_reference_dataset() -> None:
    """若本地有 pusht 验证产物则跑真数据回归(不依赖网络,目录缺失则跳过)。"""
    path = Path("/tmp/pusht_v30_ours")
    if not (path / "meta" / "info.json").exists():
        import pytest

        pytest.skip("local pusht validation output not present")
    report = qc_dataset(path)
    assert report.ok, [str(i) for i in report.issues]
    assert report.episodes_checked == 206
