# Phase 0 notes — LeRobot formats & Rerun ingestion

## LeRobot v2.1 vs v3.0 in practice (from lerobot/pusht, 206 eps)

- v2.1 per-episode stats in `episodes_stats.jsonl` are **per-dimension lists**
  (image features: nested 3-D lists; count: 1-element list) — scalar-first
  assumptions break. Element-wise numpy aggregation handles all shapes.
- v3.0 timestamps **restart at 0 each episode**; QC checks must segment by
  episode.
- Real v3.0 episodes parquet carries `stats/<feature>/{min,max,mean,std,count,
  q01,q10,q50,q90,q99}` and constant `meta/episodes/{chunk,file}_index` columns.
- `tasks.parquet` is a pandas index-layout artifact (`__index_level_0__` =
  task strings); upstream `load_tasks` restores the index by name.

## Rerun ingestion (rerun-sdk 0.40.0, 2026-10-28)

- `scripts/visualize_v30.py`: dataset parquet → per-episode scalar time series
  (each feature dim gets an entity under `episode_XXXXXX/<feature>/<dim>`),
  task/length logged as TextLog; saves standalone `.rrd` openable by the
  Rerun viewer. Works on all converter outputs (validated on real pusht
  conversion: 3 eps → 2.1MB rrd; mcap fixture: 1 ep → 11KB rrd).
- API notes for 0.40 (several breaking changes vs older docs):
  - `rr.TimeColumn(name, timestamp=[...])` — the kwarg is `timestamp`,
    not `timeseries`.
  - `rr.ComponentColumn("scalar", rr.components.ScalarBatch(values))` —
    `ScalarBatch` (component) works; the `rr.Scalars` archetype does not
    (no `as_arrow_array`).
  - App IDs with dots get migrated to entry names (warning, harmless).
- Headless verification: no viewer needed to produce `.rrd`; opening requires
  `rerun file.rrd` on a desktop. Message-count sanity via file size only.
- Phase 2 plan: wrap this as `elake preview DATASET` with blueprint defaults
  (time-series panel per feature group), then embed the web viewer in the
  platform UI. The Rerun web viewer (Wasm) can load `.rrd` via HTTP — aligns
  with the self-hosted deployment story.
- Rerun's native LeRobotReader (Rust viewer side) reads lerobot datasets
  directly; our artifacts are structurally identical to official ones
  (validated), so native loading should work — to be verified with a GUI
  session when available.

## LeRobot record→train→rollout environment

Deferred: lerobot install pulls the full training stack (torch & co.). The
platform's Babel/QC path deliberately avoids it; revisit for the Phase 3
deep-dive if needed (e.g. using the system python which already has torch).
