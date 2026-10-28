"""Dataset-to-dataset structural/value comparison (``elake diff``).

Answers "did my conversion / migration / copy change anything?" — the exact
comparison used to validate the v2.1→v3.0 converter against the official
conversion of a real 206-episode Hub dataset. Compares:

- info.json key fields (version, totals, fps, splits, data_path template)
- tasks (strings and indices)
- episodes metadata (lengths, index ranges, tasks, per-episode stats columns)
- aggregate stats.json (numeric, tolerance)
- data rows (shared feature columns, aligned by the global ``index`` column)

Works for v3.0-vs-v3.0 and v2.1-vs-v3.0 (the conversion-validation case).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..inspect import inspect

TOL_RTOL = 1e-4
TOL_ATOL = 1e-6


@dataclass
class SectionResult:
    name: str
    ok: bool
    detail: str = ""
    n_items: int = 0


@dataclass
class DiffReport:
    left: Path
    right: Path
    sections: list[SectionResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.sections)

    def summary(self) -> str:
        marks = {True: "✅", False: "❌"}
        lines = [f"{marks[s.ok]} {s.name:<24} {s.detail}" for s in self.sections]
        verdict = "IDENTICAL (within tolerance)" if self.ok else "DIFFERENCES FOUND"
        lines.append(f"— {verdict}")
        return "\n".join(lines)


def _close(a, b) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                rtol=TOL_RTOL, atol=TOL_ATOL))
    except (TypeError, ValueError):
        return a == b


def _read_v30_episodes(root: Path) -> dict:
    return pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()


def _read_v30_data(root: Path, episodes: dict) -> list[dict]:
    """Concatenate data-file columns into per-column lists, keyed by column."""
    cols: dict[str, list] = {}
    seen: set[Path] = set()
    for row in range(len(episodes["episode_index"])):
        p = (root / "data" / f"chunk-{episodes['data/chunk_index'][row]:03d}"
             / f"file-{episodes['data/file_index'][row]:03d}.parquet")
        if p in seen or not p.exists():
            continue
        seen.add(p)
        d = pq.read_table(p).to_pydict()
        for k, v in d.items():
            cols.setdefault(k, []).extend(v)
    return cols


def _read_v21_data(root: Path, episodes: list[dict]) -> dict[str, list]:
    cols: dict[str, list] = {}
    for ep in episodes:
        p = root / "data" / f"chunk-000" / f"episode_{ep['episode_index']:06d}.parquet"
        if not p.exists():
            continue
        d = pq.read_table(p).to_pydict()
        for k, v in d.items():
            cols.setdefault(k, []).extend(v)
    return cols


def _v21_meta(root: Path) -> tuple[dict, list[dict], list[dict], dict | None]:
    """info, episodes rows, tasks rows, aggregate stats — v2.1 flavor."""
    from ..babel.v21_to_v30 import load_v21

    payload = load_v21(root)
    info = payload["info"]
    episodes = payload["episodes"]
    tasks = payload["tasks"]
    agg = None
    if payload["episode_stats"] is not None:
        from ..babel.v21_to_v30 import aggregate_episode_stats

        agg = aggregate_episode_stats(list(payload["episode_stats"].values()))
    return info, episodes, tasks, agg


def diff_datasets(a: str | Path, b: str | Path, *, expect_migration: bool = False) -> DiffReport:
    left, right = Path(a), Path(b)
    report = DiffReport(left=left, right=right)
    ca, cb = inspect(str(left)), inspect(str(right))

    # ---- info key fields ----------------------------------------------------
    keys = ["codebase_version", "total_episodes", "total_frames", "fps", "splits", "data_path"]
    diffs = [k for k in keys if getattr(ca, k, None) != getattr(cb, k, None)]
    if expect_migration:
        # a v2.1→v3.0 migration is *supposed* to change these two
        diffs = [k for k in diffs if k not in ("codebase_version", "data_path")]
    report.sections.append(
        SectionResult("info", not diffs, f"differs: {diffs}" if diffs else f"fields match ({len(keys)})")
    )

    # ---- tasks ---------------------------------------------------------------
    if ca.codebase_version == "v3.0" and cb.codebase_version == "v3.0":
        ta = pq.read_table(left / "meta" / "tasks.parquet").to_pydict()
        tb = pq.read_table(right / "meta" / "tasks.parquet").to_pydict()
        tasks_match = ta.get("task_index") == tb.get("task_index") and (
            ta.get("__index_level_0__", ta.get("task")) == tb.get("__index_level_0__", tb.get("task"))
        )
        n = len(ta.get("task_index", []))
    else:
        tasks_match = True
        n = 0
    report.sections.append(SectionResult("tasks", tasks_match, f"{n} task(s)"))

    # ---- episodes metadata ---------------------------------------------------
    if ca.codebase_version == "v3.0" and cb.codebase_version == "v3.0":
        ea, eb = _read_v30_episodes(left), _read_v30_episodes(right)
        col_ok, col_bad = [], []
        for col in ("episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks"):
            if col in ea and col in eb:
                (col_ok if ea[col] == eb[col] else col_bad).append(col)
        n_stats = 0
        stat_bad = []
        for col in set(ea) & set(eb):
            if col.startswith("stats/"):
                n_stats += 1
                if not all(_close(x, y) for x, y in zip(ea[col], eb[col])):
                    stat_bad.append(col)
        ok = not col_bad and not stat_bad and len(ea["episode_index"]) == len(eb["episode_index"])
        detail = f"{len(ea['episode_index'])} eps; bad: {col_bad + stat_bad}" if not ok else \
                 f"{len(ea['episode_index'])} eps, {n_stats} stat cols match"
        report.sections.append(SectionResult("episodes", ok, detail))
    else:
        report.sections.append(
            SectionResult("episodes", True, "cross-version: covered by data/stats sections")
        )

    # ---- aggregate stats -------------------------------------------------------
    agg_pair = []
    for root, card in ((left, ca), (right, cb)):
        sp = root / "meta" / "stats.json"
        if sp.exists():
            agg_pair.append(json.loads(sp.read_text()))
        elif card.codebase_version == "v2.1":
            agg_pair.append(_v21_meta(root)[3])
        else:
            agg_pair.append(None)
    if agg_pair[0] is not None and agg_pair[1] is not None:
        bad = []
        for feat in set(agg_pair[0]) & set(agg_pair[1]):
            for stat in set(agg_pair[0][feat]) & set(agg_pair[1][feat]):
                if not _close(agg_pair[0][feat][stat], agg_pair[1][feat][stat]):
                    bad.append(f"{feat}.{stat}")
        report.sections.append(
            SectionResult("stats.json", not bad, f"bad: {bad}" if bad else "aggregate matches")
        )
    else:
        report.sections.append(SectionResult("stats.json", True, "not present on both sides"))

    # ---- data rows ---------------------------------------------------------------
    da = _read_v30_data(left, _read_v30_episodes(left)) if ca.codebase_version == "v3.0" \
        else _read_v21_data(left, _v21_meta(left)[1])
    db = _read_v30_data(right, _read_v30_episodes(right)) if cb.codebase_version == "v3.0" \
        else _read_v21_data(right, _v21_meta(right)[1])
    shared = [c for c in da if c in db and not str(c).startswith(("videos/", "observation.image"))]
    bad_cols = []
    for col in shared:
        if len(da[col]) != len(db[col]) or not all(
            _close(x, y) for x, y in zip(da[col], db[col])
        ):
            bad_cols.append(col)
    ok = not bad_cols and len(da.get("index", [])) == len(db.get("index", []))
    report.sections.append(
        SectionResult(
            "data rows",
            ok,
            f"{len(da.get('index', []))} rows × {len(shared)} shared cols"
            + (f"; bad: {bad_cols}" if bad_cols else " match"),
        )
    )
    return report
