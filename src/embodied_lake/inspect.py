"""Inspect LeRobot datasets (local directory or Hugging Face Hub repo).

Pure-stdlib module: no heavy dependencies (no torch / lerobot) so it can run
anywhere, including CI and edge boxes.

Supports both the v2.x and v3.x ``meta/info.json`` layouts:

- v2.x: per-episode files, keys like ``total_episodes``/``total_frames``.
- v3.x: file-based shards with relational metadata and path templates
  (``data_path`` / ``video_path``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
USER_AGENT = "embodied-lake/0.1.0 (+https://github.com/placeholder/embodied-lake)"


class InspectError(RuntimeError):
    """Raised when a dataset source cannot be inspected."""


@dataclass
class DatasetCard:
    """Normalized view over a LeRobot dataset's ``meta/info.json``."""

    source: str
    codebase_version: str = "unknown"
    robot_type: str | None = None
    total_episodes: int | None = None
    total_frames: int | None = None
    total_tasks: int | None = None
    fps: int | float | None = None
    splits: dict = field(default_factory=dict)
    data_path: str | None = None
    video_path: str | None = None
    features: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _looks_like_repo_id(source: str) -> bool:
    """Heuristic: ``org/name`` with no path separators on disk and valid chars."""
    if Path(source).exists():
        return False
    parts = source.split("/")
    if len(parts) != 2 or not all(parts):
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return all(set(part) <= allowed for part in parts)


def load_info(source: str, *, timeout: float = 30.0) -> dict:
    """Load ``meta/info.json`` from a local dataset directory or an HF repo id."""
    local = Path(source) / "meta" / "info.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))

    if _looks_like_repo_id(source):
        url = HF_RESOLVE_URL.format(repo_id=source, filename="meta/info.json")
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:  # gated/private/missing repos all land here
            raise InspectError(
                f"could not fetch meta/info.json for '{source}' "
                f"(HTTP {exc.code}; dataset may be missing, gated, or private)"
            ) from exc

    raise InspectError(
        f"'{source}' is neither a local dataset directory "
        f"(no meta/info.json found) nor a 'org/name' Hugging Face repo id"
    )


def normalize(source: str, info: dict) -> DatasetCard:
    """Map a raw info dict (v2.x or v3.x) onto a DatasetCard."""
    known = {
        "codebase_version", "robot_type", "total_episodes", "total_frames",
        "total_tasks", "fps", "splits", "data_path", "video_path", "features",
    }
    card = DatasetCard(
        source=source,
        codebase_version=str(info.get("codebase_version", "unknown")),
        robot_type=info.get("robot_type"),
        total_episodes=info.get("total_episodes"),
        total_frames=info.get("total_frames"),
        total_tasks=info.get("total_tasks"),
        fps=info.get("fps"),
        splits=info.get("splits", {}) or {},
        data_path=info.get("data_path"),
        video_path=info.get("video_path"),
        features=sorted((info.get("features") or {}).keys()),
    )
    card.extra = {k: v for k, v in info.items() if k not in known}
    return card


def inspect(source: str, *, timeout: float = 30.0) -> DatasetCard:
    """Inspect a dataset and return its normalized card."""
    return normalize(source, load_info(source, timeout=timeout))


def render(card: DatasetCard) -> str:
    """Human-readable rendering of a DatasetCard."""
    lines = [
        f"dataset        : {card.source}",
        f"format version : {card.codebase_version}",
    ]
    if card.robot_type:
        lines.append(f"robot type     : {card.robot_type}")
    lines += [
        f"episodes       : {card.total_frames if card.total_episodes is None else card.total_episodes}",
        f"frames         : {card.total_frames}",
        f"tasks          : {card.total_tasks}",
        f"fps            : {card.fps}",
    ]
    if card.splits:
        lines.append(f"splits         : {json.dumps(card.splits)}")
    if card.data_path:
        lines.append(f"data template  : {card.data_path}")
    if card.video_path:
        lines.append(f"video template : {card.video_path}")
    if card.features:
        lines.append("features       :")
        lines += [f"  - {name}" for name in card.features]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m embodied_lake.inspect <dataset-dir-or-hf-repo-id>")
        return 0 if argv else 2
    try:
        card = inspect(argv[0])
    except InspectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render(card))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
