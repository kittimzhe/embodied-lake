"""Convert MCAP recordings (e.g. rosbag2 output) into LeRobot v3.0 datasets.

Scope of this first revision:

- state/action come from ``sensor_msgs/JointState`` topics (the dominant
  encoding for teleoperated arm recordings), or from JSON-schema channels
  with a configurable field name;
- one MCAP file == one episode; multiple files are concatenated into a
  single v3.0 dataset;
- action defaults to the state stream (demonstrated joints) unless an
  explicit action topic is given;
- video/image topics are not converted yet (reported in the run report).

The output directory is a valid video-free LeRobot v3.0 dataset, structurally
identical to what upstream tooling produces for video-free data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from . import _v30

try:  # optional JSON channel support needs nothing beyond stdlib
    import json as _json
except ImportError:  # pragma: no cover
    _json = None


class McapConversionError(RuntimeError):
    pass


@dataclass
class McapMapping:
    """Topic-to-feature mapping for one conversion run."""

    state_topic: str
    action_topic: str | None = None  # default: mirror state (teleop demos)
    state_field: str = "position"  # JointState field, or JSON dict key
    task: str = "demonstration"
    fps: int = 30
    robot_type: str | None = None


@dataclass
class McapReport:
    files: list[Path] = field(default_factory=list)
    episodes: int = 0
    frames: int = 0
    dropped_frames: int = 0
    skipped_topics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _decode_vector(msg, field: str, topic: str) -> np.ndarray:
    """Extract a float vector from a decoded ROS2 message or JSON dict."""
    if isinstance(msg, dict):
        value = msg.get(field)
    else:
        value = getattr(msg, field, None)
    if value is None:
        raise McapConversionError(f"topic '{topic}' has no field '{field}'")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise McapConversionError(
            f"topic '{topic}' field '{field}' is not a non-empty 1-D vector"
        )
    return arr


def _read_streams(
    path: Path, mapping: McapMapping
) -> tuple[dict[str, list[tuple[int, np.ndarray]]], list[str]]:
    """Read mapped topics from one MCAP file -> ({topic: [(log_time_ns, vec)]}, skipped)."""
    wanted = {mapping.state_topic}
    if mapping.action_topic:
        wanted.add(mapping.action_topic)

    streams: dict[str, list[tuple[int, np.ndarray]]] = {}
    skipped: list[str] = []
    factory = DecoderFactory()

    with path.open("rb") as fh:
        reader = make_reader(fh)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic not in wanted:
                continue
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is not None:
                payload = decoder(message.data)
            elif schema.encoding == "json":
                payload = _json.loads(message.data.decode("utf-8"))
            else:
                if topic not in skipped:
                    skipped.append(topic)
                continue
            streams.setdefault(topic, []).append(
                (message.log_time, _decode_vector(payload, mapping.state_field, topic))
            )
    return streams, skipped


def _forward_fill(
    stream: list[tuple[int, np.ndarray]], times_ns: np.ndarray
) -> np.ndarray:
    """Latest sample at-or-before each query time (strict forward-fill)."""
    out = np.empty((len(times_ns), len(stream[0][1])), dtype=np.float32)
    idx = 0
    current = None
    for i, t in enumerate(times_ns):
        while idx < len(stream) and stream[idx][0] <= t:
            current = stream[idx][1]
            idx += 1
        if current is None:
            raise McapConversionError("action stream starts after the first state sample")
        out[i] = current
    return out


def convert_mcap_to_v30(
    sources: list[str | Path],
    dst: str | Path,
    mapping: McapMapping,
) -> McapReport:
    root_paths = [Path(s) for s in sources]
    for p in root_paths:
        if not p.is_file():
            raise McapConversionError(f"mcap source not found: {p}")
    target = Path(dst)
    if target.exists():
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True)

    report = McapReport(files=root_paths)

    all_state: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    episodes_meta: list[dict] = []
    global_index = 0

    for ep_idx, path in enumerate(root_paths):
        streams, skipped_topics = _read_streams(path, mapping)
        report.skipped_topics.extend(f"{path.name}:{t}" for t in skipped_topics)
        if mapping.state_topic not in streams:
            raise McapConversionError(
                f"{path.name}: state topic '{mapping.state_topic}' not found. "
                f"Available decodable topics: {sorted(streams) or 'none'}"
            )
        state_stream = streams[mapping.state_topic]
        times_ns = np.asarray([t for t, _ in state_stream], dtype=np.int64)
        state = np.stack([v for _, v in state_stream]).astype(np.float32)

        if mapping.action_topic:
            action = _forward_fill(streams[mapping.action_topic], times_ns)
        else:
            action = state.copy()
            report.notes.append(
                f"episode {ep_idx}: action mirrors state topic (teleop demonstration)"
            )

        length = len(state_stream)
        all_state.append(state)
        all_action.append(action)

        frames_tbl = pa.table(
            {
                "timestamp": pa.array(
                    ((times_ns - times_ns[0]) / 1e9).astype(np.float32)
                ),
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([ep_idx] * length, type=pa.int64()),
                "index": pa.array(
                    list(range(global_index, global_index + length)), type=pa.int64()
                ),
                "task_index": pa.array([0] * length, type=pa.int64()),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            }
        )
        out_path = target / _v30.DEFAULT_DATA_PATH.format(chunk_index=0, file_index=ep_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(frames_tbl, out_path)

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [mapping.task],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": ep_idx,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                **_v30.stats_to_flat_columns(
                    "observation.state", _v30.compute_feature_stats(state)
                ),
                **_v30.stats_to_flat_columns(
                    "action", _v30.compute_feature_stats(action)
                ),
            }
        )
        report.episodes += 1
        report.frames += length
        global_index += length

    # ---- meta artifacts ---------------------------------------------------- #
    _v30.write_tasks(target, [mapping.task])
    _v30.write_episodes(target, episodes_meta)
    _v30.write_stats(
        target,
        {
            "observation.state": _v30.compute_feature_stats(np.concatenate(all_state)),
            "action": _v30.compute_feature_stats(np.concatenate(all_action)),
        },
    )
    dim = int(all_state[0].shape[1])
    _v30.write_info(
        target,
        features={
            "observation.state": {"dtype": "float32", "shape": [dim], "names": None},
            "action": {"dtype": "float32", "shape": [dim], "names": None},
        },
        total_episodes=report.episodes,
        total_frames=report.frames,
        fps=mapping.fps,
        robot_type=mapping.robot_type,
    )
    # fill total_tasks after the fact (kept out of write_info signature churn)
    info_path = target / "meta" / "info.json"
    info = _json.loads(info_path.read_text())
    info["total_tasks"] = 1
    info_path.write_text(_json.dumps(info, indent=2))
    report.notes.append("video/image topics are not converted in this revision")
    return report
