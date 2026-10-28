"""Convert ALOHA-style HDF5 episode files into a LeRobot v3.0 dataset.

ALOHA (and most of its derivatives) record one HDF5 file per episode with:

- ``/observations/qpos``   (N, dim)   joint positions -> observation.state
- ``/observations/qvel``   (N, dim)   joint velocities -> observation.velocity (opt-in)
- ``/observations/images/{camera}``   uint8 frames   -> NOT converted yet
- ``/action``              (N, dim)   actions        -> action
- file attrs: ``sim`` (bool), sometimes ``compress``

Variants using ``/data/obs/...`` prefixes are handled via ``--prefix``.

One HDF5 file == one episode; images are skipped with a note in this revision
(state-space datasets convert losslessly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import _v30

try:
    import h5py
except ImportError as _exc:  # pragma: no cover
    raise ImportError("h5py is required for the aloha converter (extra: babel)") from _exc


class AlohaConversionError(RuntimeError):
    pass


@dataclass
class AlohaMapping:
    prefix: str = ""                # e.g. "data/obs" for /data/obs/qpos variants
    include_qvel: bool = False      # also emit observation.velocity
    task: str = "manipulation demo"
    fps: int = 50                   # ALOHA default recording rate
    robot_type: str | None = "aloha"


@dataclass
class AlohaReport:
    files: list[Path] = field(default_factory=list)
    episodes: int = 0
    frames: int = 0
    notes: list[str] = field(default_factory=list)


def _read_group(f: h5py.File, prefix: str) -> dict:
    """Locate observations/action datasets under an optional prefix."""
    qpos = f.get(f"{prefix}/observations/qpos".lstrip("/"))
    if qpos is None:
        qpos = f.get("qpos")  # bare variant
    action = f.get(f"{prefix}/action".lstrip("/"))
    if action is None:
        action = f.get("action")
    qvel = f.get(f"{prefix}/observations/qvel".lstrip("/"))
    if qvel is None:
        qvel = f.get("qvel")
    return {"qpos": qpos, "qvel": qvel, "action": action}


def convert_aloha_to_v30(
    sources: list[str | Path],
    dst: str | Path,
    mapping: AlohaMapping | None = None,
) -> AlohaReport:
    mapping = mapping or AlohaMapping()
    root_paths = [Path(s) for s in sources]
    for p in root_paths:
        if not p.is_file():
            raise AlohaConversionError(f"hdf5 source not found: {p}")
    target = Path(dst)
    if target.exists():
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True)

    report = AlohaReport(files=root_paths)
    episodes_meta: list[dict] = []
    all_state: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    all_qvel: list[np.ndarray] | None = [] if mapping.include_qvel else None
    global_index = 0
    used_image_note = False

    for ep_idx, path in enumerate(root_paths):
        with h5py.File(path, "r") as f:
            groups = _read_group(f, mapping.prefix)
            if groups["qpos"] is None or groups["action"] is None:
                raise AlohaConversionError(
                    f"{path.name}: missing /observations/qpos or /action "
                    f"(available root keys: {list(f.keys())})"
                )
            state = np.asarray(groups["qpos"][()], dtype=np.float32)
            action = np.asarray(groups["action"][()], dtype=np.float32)
            if state.shape != action.shape:
                raise AlohaConversionError(
                    f"{path.name}: qpos {state.shape} and action {action.shape} shapes differ"
                )
            qvel = None
            if mapping.include_qvel:
                if groups["qvel"] is None:
                    raise AlohaConversionError(f"{path.name}: qvel requested but absent")
                qvel = np.asarray(groups["qvel"][()], dtype=np.float32)
                all_qvel.append(qvel)
            if "observations" in f and "images" in (
                f.get("observations") or f.get(f"{mapping.prefix}/observations".lstrip("/")) or {}
            ):
                if not used_image_note:
                    report.notes.append(
                        "image datasets detected but not converted in this revision"
                    )
                    used_image_note = True

        length = int(state.shape[0])
        all_state.append(state)
        all_action.append(action)

        columns: dict[str, pa.Array] = {
            "timestamp": pa.array(
                (np.arange(length, dtype=np.float32) / mapping.fps)
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
        if qvel is not None:
            columns["observation.velocity"] = pa.array(
                qvel.tolist(), type=pa.list_(pa.float32())
            )

        out_path = target / _v30.DEFAULT_DATA_PATH.format(chunk_index=0, file_index=ep_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out_path)

        ep_stats: dict = {
            **_v30.stats_to_flat_columns(
                "observation.state", _v30.compute_feature_stats(state)
            ),
            **_v30.stats_to_flat_columns("action", _v30.compute_feature_stats(action)),
        }
        if qvel is not None:
            ep_stats.update(
                _v30.stats_to_flat_columns(
                    "observation.velocity", _v30.compute_feature_stats(qvel)
                )
            )
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
                **ep_stats,
            }
        )
        report.episodes += 1
        report.frames += length
        global_index += length

    # ---- meta artifacts ---------------------------------------------------- #
    _v30.write_tasks(target, [mapping.task])
    _v30.write_episodes(target, episodes_meta)
    aggregate = {
        "observation.state": _v30.compute_feature_stats(np.concatenate(all_state)),
        "action": _v30.compute_feature_stats(np.concatenate(all_action)),
    }
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": [all_state[0].shape[1]], "names": None},
        "action": {"dtype": "float32", "shape": [all_action[0].shape[1]], "names": None},
    }
    if all_qvel:
        aggregate["observation.velocity"] = _v30.compute_feature_stats(
            np.concatenate(all_qvel)
        )
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": [all_qvel[0].shape[1]],
            "names": None,
        }
    _v30.write_stats(target, aggregate)
    _v30.write_info(
        target,
        features=features,
        total_episodes=report.episodes,
        total_frames=report.frames,
        fps=mapping.fps,
        robot_type=mapping.robot_type,
    )
    import json as _json

    info_path = target / "meta" / "info.json"
    info = _json.loads(info_path.read_text())
    info["total_tasks"] = 1
    info_path.write_text(_json.dumps(info, indent=2))
    return report
