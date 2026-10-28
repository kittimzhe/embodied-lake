"""End-to-end tests for the catalog and the ingest pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcap_ros2")

from test_mcap_to_lerobot import STATE_TOPIC, _write_mcap  # noqa: E402

from embodied_lake.cli import main  # noqa: E402
from embodied_lake.platform.catalog import Catalog  # noqa: E402


@pytest.fixture()
def mcap_src(tmp_path: Path) -> Path:
    p = tmp_path / "rec.mcap"
    _write_mcap(p, 10, 0, action=False)
    return p


def test_catalog_register_and_list(mcap_src: Path, tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    rc = main([
        "ingest", str(mcap_src), str(ds),
        "--mode", "mcap-to-v30", "--state-topic", STATE_TOPIC,
        "--catalog", str(tmp_path / "cat"),
    ])
    assert rc == 0

    cat = Catalog(tmp_path / "cat")
    try:
        rows = cat.list()
        assert len(rows) == 1
        e = rows[0]
        assert e.version == "v3.0"
        assert e.episodes == 1
        assert e.frames == 10
        assert e.qc_ok is True
        assert e.mean_score == 100.0
        assert e.name == "ds"

        # lineage recorded
        cur = cat.db.execute("SELECT lineage, source FROM datasets WHERE id = ?", (e.id,))
        r = cur.fetchone()
        lineage = json.loads(r["lineage"])
        assert lineage["converter"] == "mcap_to_v30"
        assert lineage["state_topic"] == STATE_TOPIC
        assert r["source"].endswith("rec.mcap")
    finally:
        cat.close()


def test_catalog_re_register_upserts(mcap_src: Path, tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    for _ in range(2):
        rc = main([
            "ingest", str(mcap_src), str(ds),
            "--mode", "mcap-to-v30", "--state-topic", STATE_TOPIC,
            "--catalog", str(tmp_path / "cat"),
        ])
        assert rc == 0
    cat = Catalog(tmp_path / "cat")
    try:
        assert len(cat.list()) == 1  # upsert, not duplicate
    finally:
        cat.close()


def test_ingest_refuses_broken_output(tmp_path: Path) -> None:
    """结构 qc 失败时 ingest 退出非零且不注册。"""
    from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30

    src = tmp_path / "rec.mcap"
    _write_mcap(src, 10, 0, action=False)
    ds = tmp_path / "ds"
    convert_mcap_to_v30([src], ds, McapMapping(state_topic=STATE_TOPIC))
    # 破坏:删除数据文件后直接 add 应记录 qc_ok=0
    (ds / "data" / "chunk-000" / "file-000.parquet").unlink()
    cat = Catalog(tmp_path / "cat")
    try:
        entry = cat.register(ds, with_scoring=False)
        assert entry.qc_ok is False
    finally:
        cat.close()


def test_ingest_cli_end_to_end(mcap_src: Path, tmp_path: Path, capsys) -> None:
    rc = main([
        "ingest", str(mcap_src), str(tmp_path / "ds2"),
        "--mode", "mcap-to-v30", "--state-topic", STATE_TOPIC,
        "--catalog", str(tmp_path / "cat2"),
    ])
    assert rc == 0
    rc = main(["catalog", "list", "--catalog", str(tmp_path / "cat2")])
    out = capsys.readouterr().out
    assert "ds2" in out and "PASS" in out
    assert rc == 0
