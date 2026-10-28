"""Tests for the Hub tarball streaming extractor (offline tar.gz fixture)."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_hub_tarball import stream_extract_tabular  # noqa: E402


def _make_task_tarball(path: Path, task: str = "676") -> None:
    """Task-layout tarball: data parquet, meta files, then a fat video."""
    with tarfile.open(path, "w:gz") as tf:
        def add(name: str, data: bytes) -> None:
            b = tarfile.TarInfo(name)
            b.size = len(data)
            tf.addfile(b, io.BytesIO(data))

        add(f"AgiBotWorld-Beta/{task}/data/chunk-000/file-000.parquet", b"PAR1")
        add(f"AgiBotWorld-Beta/{task}/meta/info.json", b"{}")
        add(f"AgiBotWorld-Beta/{task}/meta/tasks.parquet", b"PAR1")
        add(f"AgiBotWorld-Beta/{task}/videos/head/chunk-000/file-000.mp4",
            b"\x00" * 100_000)


def test_extracts_tabular_and_stops_before_videos(tmp_path: Path) -> None:
    tb = tmp_path / "676.tar.gz"
    _make_task_tarball(tb)
    out = tmp_path / "out"
    got = stream_extract_tabular(str(tb), out)
    # stops at tasks.parquet → video member never extracted
    assert "meta/tasks.parquet" in got
    assert not any("videos" in g for g in got)
    assert (out / "meta" / "info.json").read_bytes() == b"{}"
