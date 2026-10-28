"""LeRobot dataset format converter: v2.1 -> v3.0 (local, video-free datasets).

A dependency-light reimplementation of the structural conversion performed by
upstream ``src/lerobot/scripts/convert_dataset_v21_to_v30.py``, matching its
observable layout semantics:

- ``meta/tasks.jsonl``            -> ``meta/tasks.parquet``             (task_index, task)
- ``meta/episodes.jsonl`` (+stats)-> ``meta/episodes/chunk-XXX/file-XXX.parquet``
- ``data/chunk-XXX/episode_*.parquet`` -> ``data/chunk-XXX/file-XXX.parquet``
  (episodes concatenated into size-bounded files; chunk/file roll over per
  ``chunks_size`` files per chunk)
- ``meta/info.json``              -> updated to v3.0 (templates, sizes, per-feature fps)

Not yet implemented (fails fast with a clear error):
- video features (would need MP4 concatenation + per-episode timestamp ranges)

Stats aggregation over episodes uses count-weighted population moments; this
matches the trivial cases and is documented in the report emitted per run.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

V21 = "v2.1"
V30 = "v3.0"

DEFAULT_DATA_FILE_SIZE_IN_MB = 100.0
DEFAULT_CHUNK_SIZE = 1000  # files per chunk in v3.0
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

MB = 1024.0 * 1024.0


class ConversionError(RuntimeError):
    """Raised when the source dataset cannot be converted."""


@dataclass
class ConvertReport:
    source: Path
    target: Path
    episodes: int = 0
    frames: int = 0
    data_files_written: int = 0
    tasks: int = 0
    has_stats: bool = False
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# v2.1 readers (stdlib json lines)
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_v21(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ConversionError(f"no meta/info.json under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = info.get("codebase_version", "unknown")
    if version != V21:
        raise ConversionError(f"expected codebase_version {V21}, found '{version}'")

    tasks_rows = _read_jsonl(root / "meta" / "tasks.jsonl")
    episodes_rows = _read_jsonl(root / "meta" / "episodes.jsonl")
    stats_path = root / "meta" / "episodes_stats.jsonl"
    stats_rows = _read_jsonl(stats_path) if stats_path.is_file() else None
    return {
        "info": info,
        "tasks": sorted(tasks_rows, key=lambda r: r["task_index"]),
        "episodes": sorted(episodes_rows, key=lambda r: r["episode_index"]),
        "episode_stats": (
            {r["episode_index"]: r["stats"] for r in stats_rows} if stats_rows is not None else None
        ),
    }


# --------------------------------------------------------------------------- #
# v3.0 writers
# --------------------------------------------------------------------------- #

def _roll_over(chunk_idx: int, file_idx: int, files_per_chunk: int) -> tuple[int, int]:
    file_idx += 1
    if file_idx >= files_per_chunk:
        chunk_idx += 1
        file_idx = 0
    return chunk_idx, file_idx


def _flatten_episode_stats(stats: dict) -> dict[str, object]:
    """Flatten ``{feature: {stat: value}}`` -> ``stats/<feature>/<stat>`` columns.

    Passes through *all* stat keys present in the source (real-world v2.1
    datasets carry min/max/mean/std/count **and** q01/q10/q50/q90/q99).
    """
    flat: dict[str, object] = {}
    for feature, s in stats.items():
        for stat, value in s.items():
            flat[f"stats/{feature}/{stat}"] = value
    return flat


def aggregate_episode_stats(all_stats: list[dict]) -> dict:
    """Count-weighted aggregation of per-episode feature stats.

    min/max: extremum of extremums; count: sum; mean/std: population moments
    combined across episodes. ``std`` stays None when any episode lacks it.
    """
    features: set[str] = set()
    for stats in all_stats:
        features.update(stats.keys())

    out: dict[str, dict] = {}
    for feature in sorted(features):
        mins, maxs, counts, means, stds = [], [], [], [], []
        for stats in all_stats:
            s = stats.get(feature)
            if not s:
                continue
            if "min" in s:
                mins.append(s["min"])
            if "max" in s:
                maxs.append(s["max"])
            if "count" in s:
                counts.append(float(s["count"]))
            if "mean" in s:
                means.append((float(s["mean"]), float(s.get("count", 1))))
            if "std" in s:
                stds.append((float(s["std"]), float(s["mean"]), float(s.get("count", 1))))
        agg: dict = {}
        if mins:
            agg["min"] = min(mins)
        if maxs:
            agg["max"] = max(maxs)
        if counts:
            agg["count"] = sum(counts)
        if means:
            n = sum(c for _, c in means)
            agg["mean"] = sum(m * c for m, c in means) / n if n else None
        if stds and "mean" in agg:
            n = sum(c for _, _, c in stds)
            m = agg["mean"]
            # E[x^2] = var + mean^2 per episode, combined then centered.
            ex2 = sum((sd * sd + mu * mu) * c for sd, mu, c in stds) / n if n else None
            agg["std"] = max(ex2 - m * m, 0.0) ** 0.5 if ex2 is not None else None
        out[feature] = agg
    return out


# --------------------------------------------------------------------------- #
# main conversion
# --------------------------------------------------------------------------- #

def convert_v21_to_v30(
    src: str | Path,
    dst: str | Path,
    *,
    data_file_size_in_mb: float = DEFAULT_DATA_FILE_SIZE_IN_MB,
) -> ConvertReport:
    root = Path(src)
    target = Path(dst)
    payload = load_v21(root)
    info = payload["info"]

    video_keys = [k for k, ft in (info.get("features") or {}).items() if ft.get("dtype") == "video"]
    if video_keys:
        raise NotImplementedError(
            "video features are not supported by this converter yet (mp4 concatenation + "
            f"timestamp ranges pending); found: {video_keys}. "
            "Use upstream convert_dataset_v21_to_v30.py for video datasets."
        )

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    report = ConvertReport(source=root, target=target, has_stats=payload["episode_stats"] is not None)

    # ---- tasks.parquet ----------------------------------------------------- #
    # Upstream writes this artifact via pandas with the task strings as a named
    # index (io_utils.write_tasks/load_tasks); mirror it exactly so the file is
    # byte-compatible with pd.read_parquet-based readers. Fall back to a plain
    # pyarrow table when pandas is unavailable (reported as a note).
    task_indices = [t["task_index"] for t in payload["tasks"]]
    task_strings = [t["task"] for t in payload["tasks"]]
    (target / "meta").mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(
            {"task_index": task_indices}, index=pd.Index(task_strings, name="task")
        ).to_parquet(target / "meta" / "tasks.parquet")
    except ImportError:  # pragma: no cover - exercised only without pandas
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array(task_indices, type=pa.int64()),
                    "task": pa.array(task_strings, type=pa.string()),
                }
            ),
            target / "meta" / "tasks.parquet",
        )
        report.notes.append("pandas unavailable: tasks.parquet written without upstream index layout")
    report.tasks = len(task_indices)

    # ---- data shards + per-episode data metadata --------------------------- #
    ep_parquets = sorted((root / "data").glob("*/*.parquet"))
    if len(ep_parquets) != len(payload["episodes"]):
        raise ConversionError(
            f"data/ parquet count ({len(ep_parquets)}) != episodes in meta ({len(payload['episodes'])})"
        )

    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    chunk_idx = file_idx = 0
    size_mb = 0.0
    from_index = 0
    pending: list[tuple[Path, dict]] = []

    episodes_meta: list[dict] = []

    def flush() -> None:
        nonlocal chunk_idx, file_idx, size_mb, pending
        if not pending:
            return
        tbl = pa.concat_tables([pq.read_table(p) for p, _ in pending])
        out = target / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, out)
        report.data_files_written += 1
        chunk_idx, file_idx = _roll_over(chunk_idx, file_idx, chunks_size)
        size_mb = 0.0
        pending = []

    for ep_path, ep_row in zip(ep_parquets, payload["episodes"]):
        ep_size_mb = ep_path.stat().st_size / MB
        num_frames = pq.ParquetFile(ep_path).metadata.num_rows
        if num_frames != ep_row["length"]:
            raise ConversionError(
                f"episode {ep_row['episode_index']}: meta length {ep_row['length']} != parquet rows {num_frames}"
            )
        if size_mb + ep_size_mb >= data_file_size_in_mb and pending:
            flush()
        episodes_meta.append(
            {
                "episode_index": ep_row["episode_index"],
                "tasks": ep_row.get("tasks", []),
                "length": num_frames,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": from_index,
                "dataset_to_index": from_index + num_frames,
                # upstream always writes episodes metadata as chunk-000/file-000
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                **(
                    _flatten_episode_stats(payload["episode_stats"][ep_row["episode_index"]])
                    if payload["episode_stats"] is not None
                    and ep_row["episode_index"] in payload["episode_stats"]
                    else {}
                ),
            }
        )
        pending.append((ep_path, ep_row))
        size_mb += ep_size_mb
        from_index += num_frames
        report.episodes += 1
        report.frames += num_frames
    flush()

    # ---- meta/episodes parquet --------------------------------------------- #
    episodes_tbl = pa.Table.from_pylist(episodes_meta)
    ep_out = target / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episodes_tbl, ep_out)

    # ---- meta/stats.json (aggregate) --------------------------------------- #
    if payload["episode_stats"] is not None:
        agg = aggregate_episode_stats(list(payload["episode_stats"].values()))
        (target / "meta" / "stats.json").write_text(
            json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report.notes.append(
            "stats aggregation uses count-weighted population moments; verify against "
            "upstream aggregate_stats() before relying on std for non-trivial datasets"
        )

    # ---- info.json --------------------------------------------------------- #
    new_info = dict(info)
    new_info.pop("total_chunks", None)
    new_info.pop("total_videos", None)
    new_info["codebase_version"] = V30
    new_info["data_files_size_in_mb"] = int(data_file_size_in_mb)
    new_info["video_files_size_in_mb"] = None
    new_info["data_path"] = DEFAULT_DATA_PATH
    new_info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") is not None else None
    new_info["fps"] = int(info["fps"])
    new_info["splits"] = {"train": f"0:{len(payload['episodes'])}"}
    new_info["total_episodes"] = report.episodes
    new_info["total_frames"] = report.frames
    for key, ft in (new_info.get("features") or {}).items():
        if ft.get("dtype") != "video":
            ft["fps"] = new_info["fps"]
    (target / "meta" / "info.json").write_text(
        json.dumps(new_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
