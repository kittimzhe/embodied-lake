"""Minimal JSON API over the dataset catalog (stdlib only).

Self-hosting story: ``elake serve`` exposes the catalog as HTTP so a thin web
UI (or another service) can browse datasets, their QC verdicts and lineage.
Zero web-framework dependency — ``http.server`` + the SQLite catalog.

Endpoints:
    GET  /healthz                 -> {"ok": true}
    GET  /api/datasets            -> list of catalog entries
    GET  /api/datasets/{id}       -> full row incl. lineage + qc issues
    POST /api/datasets            -> register/refresh {"path": ...}
    GET  /preview/{name}.rrd      -> static .rrd file next to the catalog
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .catalog import Catalog


def make_handler(catalog_dir: str | Path) -> type[BaseHTTPRequestHandler]:
    catalog_path = Path(catalog_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "embodied-lake/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
            pass  # quiet by default; CLI prints startup line

        def _snapshots_for(self, dataset_path: str) -> list[dict]:
            """Snapshot rows for a dataset, read from the snapshot ledger."""
            import sqlite3

            db = catalog_path / "catalog.db"
            if not db.is_file():
                return []
            try:
                conn = sqlite3.connect(db)
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        "SELECT id, content_hash, message, created_at FROM snapshots"
                        " WHERE dataset_path = ? ORDER BY id",
                        (dataset_path,),
                    ).fetchall()
                finally:
                    conn.close()
            except sqlite3.OperationalError:  # snapshots table not created yet
                return []
            return [
                {
                    "id": r["id"],
                    "hash": r["content_hash"],
                    "message": r["message"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

        # ------------------------------------------------------------ #
        def _json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------------ #
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                page = (Path(__file__).parent / "static" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return None
            if self.path == "/healthz":
                return self._json(200, {"ok": True})
            if self.path == "/api/datasets":
                cat = Catalog(catalog_path)
                try:
                    entries = []
                    for e in cat.list():
                        snaps = self._snapshots_for(e.path)
                        entries.append(
                            {
                                "id": e.id,
                                "name": e.name,
                                "path": e.path,
                                "version": e.version,
                                "episodes": e.episodes,
                                "frames": e.frames,
                                "fps": e.fps,
                                "qc_ok": e.qc_ok,
                                "mean_score": e.mean_score,
                                "snapshot": (
                                    {"id": snaps[-1]["id"], "hash": snaps[-1]["hash"]}
                                    if snaps
                                    else None
                                ),
                                "snapshot_count": len(snaps),
                            }
                        )
                    rows = entries
                finally:
                    cat.close()
                return self._json(200, rows)

            m = re.fullmatch(r"/api/datasets/(\d+)", self.path)
            if m and not self.path.endswith("/snapshots"):
                cat = Catalog(catalog_path)
                try:
                    cur = cat.db.execute(
                        "SELECT * FROM datasets WHERE id = ?", (int(m.group(1)),)
                    )
                    row = cur.fetchone()
                finally:
                    cat.close()
                if row is None:
                    return self._json(404, {"error": "not found"})
                snaps = self._snapshots_for(row["path"])
                if snaps:
                    latest = snaps[-1]
                    payload_extra = {
                        "snapshot": {
                            "id": latest["id"],
                            "hash": latest["content_hash"],
                            "message": latest["message"],
                            "created_at": latest["created_at"],
                        },
                        "snapshot_count": len(snaps),
                    }
                else:
                    payload_extra = {"snapshot": None, "snapshot_count": 0}
                return self._json(
                    200,
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "path": row["path"],
                        "version": row["version"],
                        "episodes": row["episodes"],
                        "frames": row["frames"],
                        "fps": row["fps"],
                        "features": json.loads(row["features"] or "[]"),
                        "robot_type": row["robot_type"],
                        "qc_ok": bool(row["qc_ok"]),
                        "qc_issues": json.loads(row["qc_issues"] or "[]"),
                        "mean_score": row["mean_score"],
                        "source": row["source"],
                        "lineage": json.loads(row["lineage"] or "{}"),
                        **payload_extra,
                    },
                )

            m = re.fullmatch(r"/api/datasets/(\d+)/snapshots", self.path)
            if m:
                cat = Catalog(catalog_path)
                try:
                    cur = cat.db.execute(
                        "SELECT * FROM datasets WHERE id = ?", (int(m.group(1)),)
                    )
                    row = cur.fetchone()
                finally:
                    cat.close()
                if row is None:
                    return self._json(404, {"error": "not found"})
                return self._json(200, {"snapshots": self._snapshots_for(row["path"])})

            m = re.fullmatch(r"/preview/([\w.\-]+\.rrd)", self.path)
            if m:
                f = catalog_path / m.group(1)
                if not f.is_file():
                    return self._json(404, {"error": "no such preview"})
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return None

            return self._json(404, {"error": "unknown route"})

        # ------------------------------------------------------------ #
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/datasets":
                return self._json(404, {"error": "unknown route"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "invalid json"})
            path = payload.get("path")
            if not path:
                return self._json(400, {"error": "path required"})
            if not (Path(path) / "meta" / "info.json").is_file():
                return self._json(422, {"error": f"not a dataset directory: {path}"})
            cat = Catalog(catalog_path)
            try:
                entry = cat.register(
                    path,
                    source=payload.get("source"),
                    lineage=payload.get("lineage"),
                    with_scoring=bool(payload.get("score", False)),
                )
            finally:
                cat.close()
            return self._json(201, {"id": entry.id, "name": entry.name})

    return Handler


def serve(catalog_dir: str | Path, host: str = "127.0.0.1", port: int = 8317) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(catalog_dir))
    httpd.serve_forever()
