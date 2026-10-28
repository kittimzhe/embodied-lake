# Changelog

All notable changes to embodied-lake are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
adherence to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) starts at v0.1.0.

## [Unreleased]

## [0.1.2] — 2026-10-28

### Added
- **`quaternion_nonunit` scoring rule** — flags pose-like features (xyz+quat
  layout, dims [3:7]) whose quaternion norms deviate from 1. Calibrated against
  real data: reproduces the AgiBotWorld-Beta findings
  (OpenDriveLab/AgiBot-World#164) digit-for-digit — ee_base 663/751 frames @
  norm median 1.0070, ee_camera 737/751 @ 1.2481 — and extends them to the
  `action.*` counterparts. Joint-angle groups stay silent (norm-wandering
  discriminator CV < 0.2; real aloha qpos measures 0.59/0.55 and does not
  trigger). Zero false positives across the three validation corpora.
- **`elake report`** — shareable Markdown dataset report (identity + structural
  QC verdict + per-episode score table with findings), for PRs/dataset
  cards/onboarding.
- Streaming validation recipe: task-level tarballs on the Hub (AgiBotWorld
  style) can be validated tabular-only — parquet+meta sit at the tarball head.

## [0.1.1] — 2026-10-28

## [0.1.1] — 2026-10-28

### Added
- **Snapshot integrity ledger** (`elake snapshot create/list/verify/diff`):
  content-addressed pins over info/meta/data (per-file sha256 + row counts),
  stored in the catalog DB so dataset trees stay byte-clean. Drift reports name
  the exact file; `diff` gives file-level changesets between two pins —
  manifest-only, so audits work after the tree is gone. `elake ingest` pins
  automatically once the QC gate passes; API + web UI expose the latest hash.
- **`elake preview --compare A B`**: renders two datasets into ONE `.rrd` with
  `a`/`b`/`delta` channels per feature on a shared timeline (flat delta =
  converter output matches reference). Same pattern shared upstream as a
  workaround recipe on rerun-io/rerun#12984.
- **Validation corpus #2**: `lerobot/aloha_sim_transfer_cube_human` (50 eps /
  20k frames) — tasks, episodes metadata (49 cols), per-episode stats,
  aggregate stats, and all 20,000×8 data rows value-identical vs the official
  v3.0. Portability findings: Hub datasets disagree on tasks column naming
  (`task` vs `__index_level_0__` — tools read both); `--drop-videos` legitimately
  shifts shard boundaries (row set identical).
- **Validation corpus #3**: real mobile-ALOHA HDF5 (`sumo43/mobile-aloha`) —
  `qpos`/`qvel`/`action` round-trip **bit-exact** (max|Δ| = 0.0); the content
  scorer's first real catch: gripper velocity dims (6/13 of 14) frozen/saturated
  across every episode — anatomically correct, systematic.
- **Scale rehearsal** (`scripts/scale_rehearsal.py`): 20k episodes / 4M rows /
  174 MB synthetic v2.1 → 395k rows/s, QC PASS, source-diff IDENTICAL, output
  correctly shards into 2 size-bounded data files.
- Packaging: extras match reality (`babel`/`viz`/`all`); wheel + sdist attached
  to the release; fresh-venv install verified.

### Changed
- Web catalog UI: `pinned` column (snapshot hash) and Rerun-viewer preview links.

## [0.1.0] — 2026-10-28

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
