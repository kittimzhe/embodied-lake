"""Tests for the stdlib HTTP API over the catalog."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("mcap_ros2")

from test_mcap_to_lerobot import STATE_TOPIC, _write_mcap  # noqa: E402

from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30  # noqa: E402
from embodied_lake.platform.api import make_handler  # noqa: E402


@pytest.fixture()
def api(tmp_path: Path):
    _write_mcap(tmp_path / "rec.mcap", 10, 0, action=False)
    ds = tmp_path / "ds"
    convert_mcap_to_v30([tmp_path / "rec.mcap"], ds, McapMapping(state_topic=STATE_TOPIC))
    from embodied_lake.platform.catalog import Catalog

    cat = Catalog(tmp_path / "cat")
    cat.register(ds, source="rec.mcap")
    cat.close()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path / "cat"))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _get(url: str) -> tuple[int, object]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_healthz(api: str) -> None:
    code, body = _get(f"{api}/healthz")
    assert code == 200 and body["ok"] is True


def test_list_and_detail(api: str) -> None:
    code, rows = _get(f"{api}/api/datasets")
    assert code == 200 and len(rows) == 1
    assert rows[0]["name"] == "ds" and rows[0]["qc_ok"] is True

    code, detail = _get(f"{api}/api/datasets/{rows[0]['id']}")
    assert code == 200
    assert detail["source"] == "rec.mcap"
    assert "converter" in detail["lineage"] or detail["lineage"]
    assert detail["features"]


def test_404s(api: str) -> None:
    for path in ("/api/datasets/999", "/preview/nope.rrd", "/no/such/route"):
        try:
            urllib.request.urlopen(f"{api}{path}", timeout=5)
            ok, status = False, None
        except urllib.error.HTTPError as e:
            ok, status = True, e.code
        assert ok and status == 404, path


def test_post_register(api: str, tmp_path: Path) -> None:
    # build a second dataset and POST it
    _write_mcap(tmp_path / "rec2.mcap", 6, 1_000_000_000, action=False)
    ds2 = tmp_path / "ds2"
    convert_mcap_to_v30([tmp_path / "rec2.mcap"], ds2, McapMapping(state_topic=STATE_TOPIC))
    req = urllib.request.Request(
        f"{api}/api/datasets",
        data=json.dumps({"path": str(ds2), "source": "rec2.mcap"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 201
    _, rows = _get(f"{api}/api/datasets")
    assert len(rows) == 2


def test_index_served(api: str) -> None:
    with urllib.request.urlopen(f"{api}/", timeout=5) as r:
        body = r.read().decode()
    assert r.status == 200
    assert "embodied-lake" in body and "/api/datasets" in body


def test_snapshot_fields_and_history(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "rec.mcap", 10, 0, action=False)
    ds = tmp_path / "ds"
    convert_mcap_to_v30([tmp_path / "rec.mcap"], ds, McapMapping(state_topic=STATE_TOPIC))
    from embodied_lake.platform.catalog import Catalog
    from embodied_lake.platform.snapshots import SnapshotStore

    cat = Catalog(tmp_path / "cat")
    cat.register(ds, source="rec.mcap")
    cat.close()
    store = SnapshotStore.open(tmp_path / "cat" / "catalog.db")
    store.create(ds, message="test pin")
    store._conn.close()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path / "cat"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/api/datasets", timeout=5) as r:
            rows = json.loads(r.read())
        assert rows and rows[0]["snapshot"]["hash"], rows[0]
        assert rows[0]["snapshot_count"] >= 1

        with urllib.request.urlopen(f"{base}/api/datasets/{rows[0]['id']}/snapshots", timeout=5) as r:
            snaps = json.loads(r.read())["snapshots"]
        assert any(s["message"] == "test pin" for s in snaps)
    finally:
        httpd.shutdown()
