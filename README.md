# embodied-lake (working title)

> LeRobot-native, self-hostable data platform for embodied AI — an open alternative to the (closed-source) Foxglove Data Platform, built on the [LeRobot dataset format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) (the de-facto standard: 5,750+ datasets on the Hugging Face Hub) and [Rerun](https://github.com/rerun-io/rerun) as the multimodal data layer.

**Status: Phase 1 — Babel layer + QC.** See [docs/roadmap.md](docs/roadmap.md).

## Why

Embodied AI is bottlenecked on data infrastructure. The chain
`collect → annotate → QC → manage → simulate → train` has open gaps
(no format Babel layer, no robot-native annotation tooling, no self-hosted
platform since Foxglove went closed-source in 2024). embodied-lake targets:

1. **Babel layer** — convert MCAP / rosbag2 / HDF5 (ALOHA) / RLDS /
   LeRobot v2.1 ⇄ v3.0, with incremental migration.
2. **QC pipeline** — structural validation now; batch scoring via VLM reward
   models (e.g. NVIDIA Flora) later; quality reports throughout.
3. **Dataset management** — catalog, versioning, multi-tenancy, on-prem
   deployment (Docker Compose), Rerun-powered episode preview.

## Install & quick start

```bash
pip install -e ".[babel]"  # base CLI + converters (pyarrow/pandas/mcap/h5py — no torch)
```

```bash
# --- inspect: dataset card for any local or Hub dataset ---------------------
elake inspect Embodied-AI-6/dataset      # Hub dataset (5,750+ available)
elake inspect ./my_dataset --json        # local dataset, machine-readable

# --- convert: get existing data INTO LeRobot v3.0 ---------------------------
elake convert v21-to-v30 ./ds_v21 ./ds_v30            # LeRobot v2.1 -> v3.0
elake convert v21-to-v30 ./ds_v21 ./ds_v30 --drop-videos   # tabular-only mode
elake convert mcap-to-v30 rec_*.mcap ./ds \
    --state-topic /arm/joint_states --action-topic /arm/command   # rosbag2/MCAP
elake convert hdf5-to-v30 episode_*.hdf5 ./ds         # ALOHA-style HDF5

# --- catalog & ingest: the platform loop -------------------------------------
elake ingest rec_*.mcap ./ds --mode mcap-to-v30 --state-topic /arm/joint_states
#            ^ convert -> structural qc gate -> score -> register (one command)
elake catalog list                            # registry with qc verdicts + lineage
elake serve --port 8317                        # HTTP API over the catalog
# docker/docker-compose.yml                    # one-command on-prem deployment

# --- qc + scoring ------------------------------------------------------------
elake qc ./ds_v30                          # structural PASS/FAIL, exit code 1 on error
elake score ./ds_v30 --min-score 80        # content rules: frozen joints, saturation,
                                           # action spikes, low-motion episodes
```

## Validated against real data

The v2.1→v3.0 converter's output was compared, artifact by artifact, with the
**official** v3.0 conversion of [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht)
(206 episodes / 25,650 frames) on the Hub:

- tasks / episodes metadata / info.json fields — **identical**
- 50 per-episode stats columns (vector features, element-wise) — **identical**
- aggregate `stats.json` (count-weighted moments) — **identical**
- all 25,650 data rows × 10 shared columns — **value-identical**
- only divergence: video stats columns (dropped by explicit `--drop-videos`)

The MCAP and ALOHA converters round-trip real `sensor_msgs/JointState` (CDR)
and HDF5 payloads with 1e-6 precision in the test suite (25 tests).

## Throughput (real data, Apple silicon, best of 3)

| converter | rate | DROID-scale extrapolation (tabular) |
|---|---|---|
| v2.1 → v3.0 (pusht, 25,650 rows) | **216,000 rows/s** | ~2 min |
| MCAP → v3.0 (10,000 frames, incl. CDR decode) | **61,000 rows/s** | ~6 min |

Reference: the LanceDB DROID full conversion (76k trajectories, incl. video +
vector pipelines) runs 34 min — embodied-lake's tabular path is not the
bottleneck; video handling is the open work (see roadmap).

## Design notes

- **Dependency-light by design**: conversion and QC need only
  pyarrow/pandas/numpy (+ `mcap`/`h5py` per source format). No `lerobot`
  install, no torch — safe to run on collection machines and CI.
- **Upstream-compatible artifacts**, not "close enough": tasks.parquet is
  written with the pandas index layout upstream readers expect; per-episode
  stats carry the full key set (min/max/mean/std/count/q01–q99); constant
  `meta/episodes` columns match what real v3.0 datasets on the Hub contain.
- **QC as a first-class product**: every conversion ends with a `elake qc`
  gate (index continuity, file coverage, frame counts, split bounds, stats
  presence) before data enters the catalog.

## Repo layout

```
src/embodied_lake/
  inspect.py       # dataset card inspector (stdlib-only, v2/v3)
  babel/           # format conversion layer (v21_to_v30, mcap, aloha/hdf5)
  qc/              # structural QC pipeline
  platform/        # Phase 2: catalog / multi-tenancy / API (planned)
tests/             # pytest suite (23 tests: offline fixtures + optional network)
docs/roadmap.md    # phased plan & evidence ledger
```

## Roadmap

- [x] Phase 0 — repo skeleton, `elake inspect` (local + Hub)
- [~] Phase 1 — Babel layer (v2.1→v3.0 ✅, MCAP ✅, ALOHA HDF5 ✅; RLDS next) + QC (structural ✅, scoring next)
- [ ] Phase 2 — platform & open-source release (catalog, Rerun preview, Docker Compose)
- [ ] Phase 3 — deep dive: fleet collection orchestration **or** VLM auto-annotation

## License

Apache-2.0 (matching the LeRobot / Rerun ecosystem).

## Acknowledgements

Built on top of — and contributing back to — [huggingface/lerobot](https://github.com/huggingface/lerobot) and [rerun-io/rerun](https://github.com/rerun-io/rerun).
