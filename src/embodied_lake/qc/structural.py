"""Structural quality checks for LeRobot datasets (v2.1 and v3.0).

`elake qc` validates the invariants a dataset must satisfy to be loadable and
trainable: metadata consistency, index continuity, artifact coverage, stats
sanity. Checks are read-only and dependency-light (pyarrow only).

Inspired in spirit by community tools like lerobot-doctor (#3139): converted
data is where structural bugs creep in, so the platform runs these checks on
every conversion output before the dataset enters the catalog.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from ..inspect import DatasetCard, inspect

SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


@dataclass
class QcIssue:
    code: str
    severity: str  # error | warn | info
    message: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class QcReport:
    dataset: Path
    version: str | None = None
    episodes_checked: int = 0
    issues: list[QcIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[QcIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        n_err = sum(1 for i in self.issues if i.severity == "error")
        n_warn = sum(1 for i in self.issues if i.severity == "warn")
        return (
            f"{self.dataset}: {self.version or 'unknown'} — "
            f"{'PASS' if self.ok else 'FAIL'} "
            f"({n_err} error(s), {n_warn} warning(s), {self.episodes_checked} episodes)"
        )


def _check_common(card: DatasetCard, report: QcReport) -> None:
    if card.codebase_version not in ("v2.1", "v3.0"):
        report.issues.append(
            QcIssue("version", "error", f"unsupported format version {card.codebase_version!r}")
        )
    if card.total_episodes is None or card.total_episodes <= 0:
        report.issues.append(QcIssue("episodes", "error", "no episodes declared in info.json"))
    if card.total_frames is None or card.total_frames <= 0:
        report.issues.append(QcIssue("frames", "error", "no frames declared in info.json"))
    if not card.fps or card.fps <= 0:
        report.issues.append(QcIssue("fps", "warn", "fps missing or zero in info.json"))
    splits = card.splits or {}
    if str(splits.get("train", "")).split(":")[-1].isdigit():
        train_end = int(str(splits["train"]).split(":")[-1])
        if train_end != (card.total_episodes or -1):
            report.issues.append(
                QcIssue(
                    "splits",
                    "error",
                    f"train split ends at {train_end} but total_episodes={card.total_episodes}",
                )
            )


def _check_v30_data(root: Path, card: DatasetCard, report: QcReport) -> None:
    """Index continuity + declared-file coverage for a v3.0 dataset."""
    episodes_tbl = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    n = len(episodes_tbl["episode_index"])
    report.episodes_checked = n
    if n != (card.total_episodes or -1):
        report.issues.append(
            QcIssue(
                "episodes_count",
                "error",
                f"episodes parquet has {n} rows, info.json declares {card.total_episodes}",
            )
        )

    prev_to = None
    for i in range(n):
        ep = episodes_tbl["episode_index"][i]
        frm = episodes_tbl["dataset_from_index"][i]
        to = episodes_tbl["dataset_to_index"][i]
        length = episodes_tbl["length"][i]
        if ep != i:
            report.issues.append(
                QcIssue("episode_order", "error", f"row {i} has episode_index={ep}")
            )
        if to - frm != length:
            report.issues.append(
                QcIssue(
                    "episode_range",
                    "error",
                    f"episode {ep}: range [{frm},{to}) != length {length}",
                )
            )
        if prev_to is not None and frm != prev_to:
            report.issues.append(
                QcIssue(
                    "index_gap",
                    "error",
                    f"episode {ep}: from_index {frm} != previous to_index {prev_to}",
                )
            )
        prev_to = to

        chunk = episodes_tbl["data/chunk_index"][i]
        file_idx = episodes_tbl["data/file_index"][i]
        path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
        if not path.exists():
            report.issues.append(
                QcIssue("data_file_missing", "error", f"episode {ep}: missing {path.name}")
            )

    if report.issues:
        pass  # additional aggregate checks below still valuable

    # declared data files all exist and row counts add up
    total_rows = 0
    seen: set[Path] = set()
    for i in range(n):
        path = (
            root
            / "data"
            / f"chunk-{episodes_tbl['data/chunk_index'][i]:03d}"
            / f"file-{episodes_tbl['data/file_index'][i]:03d}.parquet"
        )
        if path in seen or not path.exists():
            continue
        seen.add(path)
    for path in sorted(seen):
        total_rows += pq.read_metadata(path).num_rows
    if total_rows != (card.total_frames or -1):
        report.issues.append(
            QcIssue(
                "frames_count",
                "error",
                f"data files hold {total_rows} rows, info.json declares {card.total_frames}",
            )
        )

    # stats.json present when features exist
    if card.features and not (root / "meta" / "stats.json").exists():
        report.issues.append(QcIssue("stats_missing", "warn", "meta/stats.json not found"))


def _check_tasks(root: Path, card: DatasetCard, report: QcReport) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        report.issues.append(QcIssue("tasks_missing", "error", "meta/tasks.parquet not found"))
        return
    tbl = pq.read_table(tasks_path)
    if "task_index" not in tbl.column_names:
        report.issues.append(QcIssue("tasks_schema", "error", "task_index column missing"))
        return
    indices = tbl.column("task_index").to_pylist()
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        report.issues.append(QcIssue("tasks_index", "warn", "task_index not unique/sorted"))


def qc_dataset(src: str | Path) -> QcReport:
    root = Path(src)
    report = QcReport(dataset=root)
    try:
        card = inspect(str(root))
    except Exception as exc:
        report.issues.append(QcIssue("unreadable", "error", f"inspect failed: {exc}"))
        return report
    report.version = card.codebase_version
    _check_common(card, report)
    if report.errors:
        return report
    _check_tasks(root, card, report)
    if card.codebase_version == "v3.0":
        _check_v30_data(root, card, report)
    else:
        report.episodes_checked = card.total_episodes or 0
        report.issues.append(
            QcIssue("v21", "info", "deep checks currently implemented for v3.0 only")
        )
    return report
