"""Dataset snapshots: content-addressed integrity ledger (``elake snapshot``).

A snapshot pins what a dataset looked like when it passed QC — so drift after
the fact (accidental rewrite, partial copy, upstream mutation) is detectable.
This is the "commit" half of dataset versioning; rollback-by-materialization
is intentionally out of scope for v0.1 (datasets are immutable parquet trees —
copying terabytes to undo a metadata typo is the wrong default).

The hash covers, in order:
- ``meta/info.json`` raw bytes
- every ``meta/*.parquet`` + ``meta/episodes/**`` raw bytes (sorted by path)
- every ``data/**`` parquet: (relpath, size, row-count, sha256) sorted by path

Snapshots live in the catalog DB (not inside the dataset), so dataset trees
stay byte-clean for upstream tools.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

_SNAP_TABLE = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_path TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    manifest TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_path ON snapshots(dataset_path, id);
"""


def _file_digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(root: str | Path) -> tuple[str, list[dict]]:
    """Return (content_hash, manifest rows) for a v3.0 (or v2.x) dataset tree."""
    root = Path(root)
    rows: list[dict] = []
    h = hashlib.sha256()

    def feed(part: bytes) -> None:
        h.update(part)

    meta_files = sorted(
        [*(root / "meta").glob("*.parquet"), *(root / "meta").rglob("*.parquet")]
    ) if (root / "meta").is_dir() else []
    info = root / "meta" / "info.json"
    if info.is_file():
        digest = _file_digest(info)
        feed(digest.encode())
        rows.append({"path": "meta/info.json", "bytes": info.stat().st_size, "sha256": digest})
    for p in sorted(set(meta_files)):
        digest = _file_digest(p)
        feed(digest.encode())
        rows.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest})

    data_files = sorted((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else []
    for p in data_files:
        rel = str(p.relative_to(root))
        size, nrows = p.stat().st_size, pq.read_metadata(p).num_rows
        digest = _file_digest(p)
        feed(f"{rel}|{size}|{nrows}|{digest}".encode())
        rows.append({"path": rel, "bytes": size, "rows": nrows, "sha256": digest})
    return h.hexdigest(), rows


@dataclass
class Snapshot:
    id: int
    dataset_path: str
    dataset_name: str
    content_hash: str
    message: str
    created_at: float
    manifest: list[dict]


class SnapshotStore:
    """Snapshot ledger backed by the same SQLite file as the catalog."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        with self._conn:
            self._conn.executescript(_SNAP_TABLE)

    @classmethod
    def open(cls, root: str | Path) -> "SnapshotStore":
        path = Path(root)
        if path.is_dir():
            path = path / "catalog.db"  # serve-style catalog root (ELAKE_CATALOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(path))

    def create(self, dataset_dir: str | Path, message: str = "", name: str | None = None) -> Snapshot:
        root = Path(dataset_dir).resolve()
        content_hash, manifest = dataset_manifest(root)
        name = name or root.name
        now = time.time()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO snapshots (dataset_path, dataset_name, content_hash, message, manifest, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (str(root), name, content_hash, message, json.dumps(manifest), now),
            )
        return Snapshot(cur.lastrowid, str(root), name, content_hash, message, now, manifest)

    def list(self, dataset_dir: str | Path | None = None) -> list[Snapshot]:
        q = "SELECT id, dataset_path, dataset_name, content_hash, message, created_at, manifest FROM snapshots"
        args: tuple = ()
        if dataset_dir is not None:
            q += " WHERE dataset_path = ?"
            args = (str(Path(dataset_dir).resolve()),)
        q += " ORDER BY id"
        return [
            Snapshot(*row[:6], json.loads(row[6]))
            for row in self._conn.execute(q, args).fetchall()
        ]

    def changeset(self, dataset_dir: str | Path, from_id: int | None = None,
                  to_id: int | None = None) -> list[dict]:
        """File-level changes between two pins (default: previous vs latest).

        Returns rows: {file, change: added|removed|changed|renamed?, detail}.
        Content-only comparison via the stored manifests — no dataset access
        needed, so pins can be compared long after the tree is gone.
        """
        snaps = self.list(dataset_dir)
        if len(snaps) < 2:
            raise ValueError("need at least two snapshots to diff")
        a = next((s for s in snaps if s.id == from_id), snaps[-2]) if from_id else snaps[-2]
        b = next((s for s in snaps if s.id == to_id), snaps[-1]) if to_id else snaps[-1]
        old = {r["path"]: r for r in a.manifest}
        new = {r["path"]: r for r in b.manifest}
        out: list[dict] = []
        for path in sorted(set(old) - set(new)):
            out.append({"file": path, "change": "removed"})
        for path in sorted(set(new) - set(old)):
            out.append({"file": path, "change": "added",
                        "detail": f"{new[path].get('rows', '?')} rows"})
        for path in sorted(set(old) & set(new)):
            if old[path] != new[path]:
                d = []
                for k in ("bytes", "rows", "sha256"):
                    if old[path].get(k) != new[path].get(k):
                        d.append(f"{k}: {old[path].get(k)} → {new[path].get(k)}")
                out.append({"file": path, "change": "changed", "detail": "; ".join(d)})
        return out

    def verify(self, dataset_dir: str | Path) -> tuple[bool, str, list[str]]:
        """Current tree vs latest snapshot → (ok, expected_hash, drift_details)."""
        snaps = self.list(dataset_dir)
        if not snaps:
            return False, "", ["no snapshot recorded for this dataset"]
        latest = snaps[-1]
        current_hash, _ = dataset_manifest(dataset_dir)
        if current_hash == latest.content_hash:
            return True, latest.content_hash, []
        _, current_manifest = dataset_manifest(dataset_dir)
        old = {r["path"]: r for r in latest.manifest}
        new = {r["path"]: r for r in current_manifest}
        drift = []
        for path in sorted(set(old) - set(new)):
            drift.append(f"missing: {path}")
        for path in sorted(set(new) - set(old)):
            drift.append(f"added: {path}")
        for path in sorted(set(old) & set(new)):
            if old[path] != new[path]:
                drift.append(f"changed: {path}")
        return False, latest.content_hash, drift or ["root hash differs (manifest equal — reordered?)"]
