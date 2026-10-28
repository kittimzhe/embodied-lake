# Changelog

All notable changes to embodied-lake are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
adherence to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) starts at v0.1.0.

## [Unreleased]

### Added
- `elake preview`: render a local v3.0 dataset to a standalone Rerun `.rrd`
  (per-episode scalar time series per feature dim) — Phase-2 episode preview
  groundwork.
- `elake serve` + `platform/api.py`: stdlib HTTP API over the catalog
  (`/healthz`, dataset list/detail with lineage + QC verdicts, POST register,
  static `.rrd` serving). Docker Compose deployment under `docker/`, verified
  end-to-end (container build, register via POST, teardown).
- `platform/catalog.py` + `elake catalog add/list`: SQLite dataset registry
  (stdlib `sqlite3`) with lineage, QC verdicts and scores; upsert-by-path.
- `elake ingest`: one command — convert → structural QC gate → score → register.
- `elake score` + `qc/scoring.py`: rule-based content scoring (frozen joints,
  sensor saturation, action spikes via robust z-score, low-motion episodes vs
  dataset-median energy). Validated on a real 206-episode conversion: zero
  false positives.
- `qc/structural.py` + `elake qc`: structural validation — metadata
  consistency, split bounds, index continuity, data-file coverage, frame
  counts, per-episode frame/timestamp monotonicity, fps drift (median-based).
- `scripts/benchmark.py`: conversion throughput harness with DROID-scale
  extrapolation. Real-data numbers: 216k rows/s (v2.1→v3.0 on
  lerobot/pusht), 61k rows/s (MCAP incl. CDR decode).
- `babel/mcap_to_lerobot.py` + `elake convert mcap-to-v30`: rosbag2/MCAP →
  LeRobot v3.0 (`sensor_msgs/JointState` via CDR/ros2msg; JSON-schema
  channels; forward-fill action alignment; one file = one episode).
- `babel/aloha_to_lerobot.py` + `elake convert hdf5-to-v30`: ALOHA-style HDF5 →
  v3.0 (qpos/action, optional qvel, variant prefixes).
- `babel/v21_to_v30.py` + `elake convert v21-to-v30`: LeRobot v2.1 → v3.0
  migration, `--drop-videos` for tabular-only conversion.
- `elake inspect`: dataset card for local & Hub datasets (v2.x/v3.0),
  stdlib-only.

### Fixed
- Element-wise stats aggregation: real v2.1 datasets store per-episode stats
  as per-dimension lists (image stats nested) — scalar-first code crashed;
  aggregation rewritten with numpy, count-weighted population moments.
- Per-episode QC segmentation: v3.0 timestamps restart at each episode
  boundary; monotonicity/fps checks now segment by episode.

### Validation highlights
- v2.1→v3.0 output verified artifact-by-artifact against the **official** v3.0
  conversion of `lerobot/pusht` (206 episodes / 25,650 frames): tasks,
  episodes metadata, 50 stats columns, aggregate stats, data rows — identical.
