"""Dataset catalog: a local registry of known datasets with QC/lineage.

The catalog is the platform's core state: every dataset we convert, validate
or score gets registered with its lineage (what produced it) and its latest
quality verdicts, so "which of my datasets are clean and where did they come
from" is a query, not an archaeology dig.

Storage: single SQLite file (stdlib ``sqlite3`` — no extra dependency,
self-hostable by copying the file). Schema is versioned via ``user_version``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..inspect import inspect as inspect_dataset
from ..qc.scoring import score_dataset
from ..qc.structural import qc_dataset

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    version       TEXT,
    episodes      INTEGER,
    frames        INTEGER,
    fps           INTEGER,
    features      TEXT,           -- JSON list of feature names
    robot_type    TEXT,
    qc_ok         INTEGER,        -- structural qc verdict (1/0)
    qc_issues     TEXT,           -- JSON list of issue strings
    mean_score    REAL,           -- content scoring mean (0-100)
    source        TEXT,           -- e.g. "mcap:/rec_20261028.mcap"
    lineage       TEXT,           -- JSON: converter + params + timestamp
    registered_at REAL
);
"""


@dataclass
class CatalogEntry:
    id: int
    path: str
    name: str
    version: str | None
    episodes: int | None
    frames: int | None
    fps: int | None
    qc_ok: bool | None
    mean_score: float | None


class Catalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "catalog.db")
        self.db.row_factory = sqlite3.Row
        cur = self.db.execute("PRAGMA user_version")
        if cur.fetchone()[0] == 0:
            self.db.executescript(_SCHEMA)
            self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.db.commit()

    # ------------------------------------------------------------------ #
    def register(
        self,
        dataset: str | Path,
        *,
        source: str | None = None,
        lineage: dict | None = None,
        with_scoring: bool = True,
    ) -> CatalogEntry:
        """Inspect + qc (+ score) a dataset and upsert it into the catalog."""
        root = Path(dataset)
        card = inspect_dataset(str(root))
        qc = qc_dataset(root)
        mean_score = None
        if with_scoring:
            try:
                mean_score = score_dataset(root).mean_score
            except Exception:
                mean_score = None  # scoring optional (e.g. odd feature shapes)

        issues = json.dumps([f"{i.severity}:{i.code}:{i.message}" for i in qc.issues])
        row = {
            "path": str(root.resolve()),
            "name": root.name,
            "version": card.codebase_version,
            "episodes": card.total_episodes,
            "frames": card.total_frames,
            "fps": card.fps,
            "features": json.dumps(sorted(card.features or {})),
            "robot_type": card.robot_type,
            "qc_ok": 1 if qc.ok else 0,
            "qc_issues": issues,
            "mean_score": mean_score,
            "source": source,
            "lineage": json.dumps({**(lineage or {}), "registered_at": time.time()}),
            "registered_at": time.time(),
        }
        self.db.execute(
            """
            INSERT INTO datasets (path, name, version, episodes, frames, fps,
                                  features, robot_type, qc_ok, qc_issues,
                                  mean_score, source, lineage, registered_at)
            VALUES (:path, :name, :version, :episodes, :frames, :fps,
                    :features, :robot_type, :qc_ok, :qc_issues,
                    :mean_score, :source, :lineage, :registered_at)
            ON CONFLICT(path) DO UPDATE SET
                version=excluded.version, episodes=excluded.episodes,
                frames=excluded.frames, fps=excluded.fps,
                features=excluded.features, robot_type=excluded.robot_type,
                qc_ok=excluded.qc_ok, qc_issues=excluded.qc_issues,
                mean_score=excluded.mean_score, source=excluded.source,
                lineage=excluded.lineage, registered_at=excluded.registered_at
            """,
            row,
        )
        self.db.commit()
        return self.get_by_path(str(root.resolve()))

    # ------------------------------------------------------------------ #
    def get_by_path(self, path: str) -> CatalogEntry:
        cur = self.db.execute("SELECT * FROM datasets WHERE path = ?", (path,))
        r = cur.fetchone()
        return self._entry(r)

    def list(self) -> list[CatalogEntry]:
        cur = self.db.execute("SELECT * FROM datasets ORDER BY id")
        return [self._entry(r) for r in cur.fetchall()]

    @staticmethod
    def _entry(r: sqlite3.Row | None) -> CatalogEntry | None:
        if r is None:
            raise KeyError("dataset not found in catalog")
        return CatalogEntry(
            id=r["id"],
            path=r["path"],
            name=r["name"],
            version=r["version"],
            episodes=r["episodes"],
            frames=r["frames"],
            fps=r["fps"],
            qc_ok=None if r["qc_ok"] is None else bool(r["qc_ok"]),
            mean_score=r["mean_score"],
        )

    def close(self) -> None:
        self.db.close()
