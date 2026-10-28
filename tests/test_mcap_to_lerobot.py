"""End-to-end tests for the MCAP -> LeRobot v3.0 converter.

Synthesizes real rosbag2-style MCAP files (ros2msg schemas, CDR payloads) with
mcap-ros2-support, converts them, and validates the v3.0 output — including a
dogfood pass through our own inspector.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

pytest.importorskip("mcap_ros2")

from mcap_ros2.writer import Writer as Ros2Writer  # noqa: E402

from embodied_lake.babel.mcap_to_lerobot import (  # noqa: E402
    McapConversionError,
    McapMapping,
    convert_mcap_to_v30,
)
from embodied_lake.inspect import inspect as inspect_dataset  # noqa: E402

JOINT_STATE_DEF = (
    "std_msgs/Header header\n"
    "string[] name\n"
    "float64[] position\n"
    "float64[] velocity\n"
    "float64[] effort\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
)

STATE_TOPIC = "/arm/joint_states"
ACTION_TOPIC = "/arm/command"


def _write_mcap(path: Path, n: int, t0_ns: int, *, action: bool) -> None:
    with path.open("wb") as fh:
        writer = Ros2Writer(fh)
        schema = writer.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE_DEF)
        action_schema = schema
        for i in range(n):
            t_ns = t0_ns + i * 33_333_333  # ~30 Hz
            header = SimpleNamespace(
                stamp=SimpleNamespace(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000),
                frame_id="base_link",
            )
            msg = SimpleNamespace(
                header=header,
                name=[f"j{k}" for k in range(3)],
                position=[math.sin(i * 0.3 + k) for k in range(3)],
                velocity=[0.0, 0.0, 0.0],
                effort=[0.0, 0.0, 0.0],
            )
            writer.write_message(topic=STATE_TOPIC, schema=schema, message=msg, log_time=t_ns)
            if action:
                cmd = SimpleNamespace(
                    header=header,
                    name=[f"j{k}" for k in range(3)],
                    position=[math.cos(i * 0.3 + k) for k in range(3)],
                    velocity=[0.0, 0.0, 0.0],
                    effort=[0.0, 0.0, 0.0],
                )
                writer.write_message(
                    topic=ACTION_TOPIC, schema=action_schema, message=cmd, log_time=t_ns
                )
        writer.finish()


@pytest.fixture()
def two_mcap_files(tmp_path: Path) -> list[Path]:
    p1 = tmp_path / "ep0.mcap"
    p2 = tmp_path / "ep1.mcap"
    _write_mcap(p1, 10, 1_000_000_000_000, action=False)
    _write_mcap(p2, 6, 5_000_000_000_000, action=False)
    return [p1, p2]


def test_convert_mcap_basic(two_mcap_files: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "ds_v30"
    mapping = McapMapping(state_topic=STATE_TOPIC, task="wave the arm", fps=30)
    report = convert_mcap_to_v30(two_mcap_files, out, mapping)

    assert report.episodes == 2
    assert report.frames == 16
    assert any("action mirrors state" in n for n in report.notes)

    # structure: one data file per episode
    d0 = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    d1 = pq.read_table(out / "data" / "chunk-000" / "file-001.parquet")
    assert d0.num_rows == 10 and d1.num_rows == 6

    # values survived the CDR encode/decode round trip
    states = d0.to_pydict()["observation.state"]
    assert states[0] == pytest.approx([math.sin(k) for k in range(3)], abs=1e-6)
    # action mirrors state by default
    assert d0.to_pydict()["action"][0] == pytest.approx(states[0], abs=1e-6)
    # timestamps start at 0 and increase
    ts = d0.to_pydict()["timestamp"]
    assert ts[0] == pytest.approx(0.0, abs=1e-4)

    # episodes metadata
    eps = pq.read_table(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    assert eps["episode_index"] == [0, 1]
    assert eps["dataset_from_index"] == [0, 10]
    assert eps["dataset_to_index"] == [10, 16]
    assert eps["tasks"] == [["wave the arm"], ["wave the arm"]]
    # stats columns include quantiles like real datasets
    assert "stats/observation.state/q50" in eps

    # dogfood: our inspector reads the produced dataset
    card = inspect_dataset(str(out))
    assert card.codebase_version == "v3.0"
    assert card.total_episodes == 2
    assert card.total_frames == 16
    assert card.robot_type is None
    assert "observation.state" in card.features and "action" in card.features


def test_convert_mcap_with_action_topic(tmp_path: Path) -> None:
    src = tmp_path / "ep.mcap"
    _write_mcap(src, 5, 1_000_000_000_000, action=True)
    out = tmp_path / "ds"
    mapping = McapMapping(state_topic=STATE_TOPIC, action_topic=ACTION_TOPIC)
    convert_mcap_to_v30([src], out, mapping)
    d = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    # action now follows the command topic (cos), not the state (sin)
    assert d["action"][0] == pytest.approx([math.cos(k) for k in range(3)], abs=1e-6)
    assert d["observation.state"][0] == pytest.approx([math.sin(k) for k in range(3)], abs=1e-6)


def test_convert_mcap_missing_topic(tmp_path: Path) -> None:
    src = tmp_path / "ep.mcap"
    _write_mcap(src, 3, 0, action=False)
    mapping = McapMapping(state_topic="/does/not/exist")
    with pytest.raises(McapConversionError, match="state topic"):
        convert_mcap_to_v30([src], tmp_path / "never", mapping)


def test_convert_mcap_stats_values(two_mcap_files: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "ds"
    convert_mcap_to_v30(
        two_mcap_files, out, McapMapping(state_topic=STATE_TOPIC)
    )
    import json

    stats = json.loads((out / "meta" / "stats.json").read_text())
    # sin over i*0.3, k in 0..2 — bounds from the full set
    vals = [math.sin(i * 0.3 + k) for i in range(10) for k in range(3)] + [
        math.sin(i * 0.3 + k) for i in range(6) for k in range(3)
    ]
    assert stats["observation.state"]["min"] == pytest.approx(min(vals))
    assert stats["observation.state"]["max"] == pytest.approx(max(vals))
    assert stats["observation.state"]["count"] == 48
