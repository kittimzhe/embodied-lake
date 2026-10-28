"""Tests for dataset diff (conversion validation)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcap_ros2")

from test_mcap_to_lerobot import STATE_TOPIC, _write_mcap  # noqa: E402

from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30  # noqa: E402
from embodied_lake.qc.diff import diff_datasets  # noqa: E402


@pytest.fixture(scope="module")
def converted_pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("diff")
    _write_mcap(tmp / "a.mcap", 10, 0, action=False)
    _write_mcap(tmp / "b.mcap", 6, 1_000_000_000, action=False)
    ds = tmp / "ds"
    convert_mcap_to_v30(
        [tmp / "a.mcap", tmp / "b.mcap"], ds, McapMapping(state_topic=STATE_TOPIC)
    )
    return ds, tmp


def test_identical_copy_passes(converted_pair) -> None:
    ds, tmp = converted_pair
    copy = tmp / "ds_copy"
    shutil.copytree(ds, copy)
    rep = diff_datasets(ds, copy)
    assert rep.ok, rep.summary()


def test_value_drift_detected(converted_pair) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds, tmp = converted_pair
    mutated = tmp / "ds_mut"
    shutil.copytree(ds, mutated)
    p = mutated / "data" / "chunk-000" / "file-000.parquet"
    d = pq.read_table(p).to_pydict()
    d["action"] = [[v + 1.0 for v in row] for row in d["action"]]
    pq.write_table(pa.table(d), p)
    rep = diff_datasets(ds, mutated)
    assert not rep.ok
    names = {s.name for s in rep.sections if not s.ok}
    assert "data rows" in names


def test_meta_drift_detected(converted_pair) -> None:
    ds, tmp = converted_pair
    mutated = tmp / "ds_meta"
    shutil.copytree(ds, mutated)
    info_path = mutated / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 999
    info_path.write_text(json.dumps(info))
    rep = diff_datasets(ds, mutated)
    assert not rep.ok
    names = {s.name for s in rep.sections if not s.ok}
    assert "info" in names or "data rows" in names


def test_render_comparison_rrd(converted_pair) -> None:
    pytest.importorskip("rerun")
    from embodied_lake.viz.rrd import render_comparison

    ds, tmp = converted_pair
    copy = tmp / "ds_copy_for_cmp"
    shutil.copytree(ds, copy)
    out = render_comparison(ds, copy, tmp / "cmp.rrd", episode=0)
    assert out.stat().st_size > 0
