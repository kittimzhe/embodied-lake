"""Tests for elake report (shareable markdown)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcap_ros2")

from test_mcap_to_lerobot import STATE_TOPIC, _write_mcap  # noqa: E402

from embodied_lake.babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30  # noqa: E402
from embodied_lake.qc.report import render_report  # noqa: E402


def test_report_renders(tmp_path: Path) -> None:
    _write_mcap(tmp_path / "a.mcap", 12, 0, action=False)
    ds = tmp_path / "ds"
    convert_mcap_to_v30([tmp_path / "a.mcap"], ds, McapMapping(state_topic=STATE_TOPIC))
    out = render_report(ds, tmp_path / "r.md")
    text = out.read_text()
    assert "# Dataset report" in text and "Structural QC" in text
    assert "PASS" in text and "Content scores" in text
    assert "episode" in text  # score table present
