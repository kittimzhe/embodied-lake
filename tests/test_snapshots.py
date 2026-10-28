"""Tests for the snapshot integrity ledger."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_lake.platform.snapshots import SnapshotStore


def _mini_ds(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))
    pq.write_table(pa.table({"a": [1, 2, 3]}), root / "data" / "chunk-000" / "file-000.parquet")


def test_create_list_verify_roundtrip(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _mini_ds(ds)
    store = SnapshotStore.open(tmp_path / "cat.db")
    s = store.create(ds, message="baseline")
    assert s.content_hash and len(s.manifest) == 2
    ok, _, drift = store.verify(ds)
    assert ok and drift == []
    listed = store.list(ds)
    assert len(listed) == 1 and listed[0].message == "baseline"


def test_drift_detected_on_data_change(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _mini_ds(ds)
    store = SnapshotStore.open(tmp_path / "cat.db")
    store.create(ds)
    p = ds / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pa.table({"a": [1, 2, 99]}), p)
    ok, _, drift = store.verify(ds)
    assert not ok
    assert any("changed" in d for d in drift)


def test_drift_detected_on_removal_and_verify_fails_clean(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _mini_ds(ds)
    store = SnapshotStore.open(tmp_path / "cat.db")
    store.create(ds)
    (ds / "data" / "chunk-000" / "file-000.parquet").unlink()
    ok, _, drift = store.verify(ds)
    assert not ok and any("missing" in d for d in drift)
    fresh = SnapshotStore.open(tmp_path / "other.db")
    ok2, _, d2 = fresh.verify(ds)
    assert not ok2 and "no snapshot" in d2[0]


def test_meta_change_detected(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _mini_ds(ds)
    store = SnapshotStore.open(tmp_path / "cat.db")
    store.create(ds)
    info = ds / "meta" / "info.json"
    info.write_text(json.dumps({"total_episodes": 2}))
    ok, _, drift = store.verify(ds)
    assert not ok
