# Roadmap & evidence ledger

Phased plan mirroring 《具身数据平台-项目与开源贡献路线图.md》 (workspace-level).
Each phase lists deliverables, resume points, and upstream-contribution links.

## Phase 0 — Foundation (weeks 1-3)  [IN PROGRESS]
- [x] Repo skeleton (Apache-2.0, src layout, pytest, ruff)
- [x] `elake inspect`: dataset card for local & Hub datasets (v2/v3), stdlib-only
- [ ] Run LeRobot record→train→rollout loop; document environment (docs/phase0-notes.md)
- [ ] Rerun ingestion of the same dataset; notes on LeRobotReader behavior
- Deliverable: reproducible env doc + "LeRobot v2.1 vs v3.0 in practice" note

## Phase 1 — Babel layer + QC MVP (months 1-2)
- [x] `elake convert v2.1→v3.0` (2026-10-28 r2: structure validated against real v3.0 dataset
      on HF — pandas-index tasks, quantile stats pass-through, meta/episodes const cols)
- [x] `elake convert mcap→lerobot-v3` (2026-10-28 r3: JointState/CDR + JSON channels,
      forward-fill action alignment, real stats incl. quantiles; dogfooded via `elake inspect`)
- [x] `elake convert hdf5(aloha)→lerobot-v3` (2026-10-28 r4: qpos/action/qvel,
      prefix variants, images skipped with note; 17/17 tests)
- [ ] `elake convert rlds→lerobot-v3`
- [ ] Benchmark: DROID 76k trajectories end-to-end (reference: LanceDB full conversion 34 min)
- [ ] `elake qc score` (Flora-Rewarder orchestration) + quality report
- Contribution link: mcap converter has NO upstream equivalent → propose to LeRobot
  (as example + docs first); do NOT compete on distributed conversion (#1998 taken)

## Phase 2 — Platform & open-source release (months 3-4)
- [ ] Catalog + versioning + multi-tenancy (web UI, S3/MinIO backend)
- [ ] Rerun viewer embedding for episode preview
- [ ] Docker Compose one-command on-prem deployment
- [ ] Public release: bilingual README, architecture diagram, demo video
- Contribution link: Rerun integration issues → Rerun PRs (Rust)

## Phase 3 — Deep dive (choose ONE, month 5+)
- A. Fleet-scale collection orchestration (task dispatch, time sync, resume, edge→cloud)
- B. Auto-annotation scheduling (Flora-TempSeg/Captioner + human review queue)

## Evidence ledger
| Date | Artifact | Evidence |
|---|---|---|
| 2026-10-28 | inspector passes offline+network tests (v3.0 dataset card) | `pytest tests/` 5/5 green |
| 2026-10-28 | `elake inspect Embodied-AI-6/dataset` live Hub fetch works | CLI output: v3.0 / 160 ep / 81786 frames |
| 2026-10-28 | git repo initialized, first commit | `8acd588` |
| 2026-10-28 | v2.1→v3.0 converter, 10/10 tests, real-dataset validated | commit `74c1dbc` |
| 2026-10-28 | mcap→v3.0 converter, 14/14 tests, CLI end-to-end + inspector dogfood | commit `6086eb8` |
| 2026-10-28 | hdf5(aloha)→v3.0 converter, 17/17 tests | commit `1828376` |
| 2026-10-28 (r4) | upstream engagement kit drafted (RFC #4368 comment, PR #4443 converter-table proposal, examples-issue draft) | ../upstream-engagement-kit.md; gh CLI found authenticated as `kittimzhe` — awaiting user confirmation |

## Contribution log (/contributor routine)
| Date | Repo | Candidate | Status | Reason |
|---|---|---|---|---|
| 2026-10-28 | huggingface/lerobot | #2061, #1998 (good first) | excluded | both assigned (s1lent4gnt); #1998 = our Babel direction, watch & engage in comments |
| 2026-10-28 | rerun-io/rerun | #11467 (update Rust deps) | excluded | claimed in comments (GeorgeLifinRell, maintainer replied) |
| 2026-10-28 | rerun-io/rerun | #11269 (vcpkg docs) | excluded | maintainer marked blocked on upstream vcpkg package |
| 2026-10-28 | huggingface/lerobot | #5116 (FastWAM docs) | excluded | commenter already wrote fix, PR imminent |
| 2026-10-28 | huggingface/lerobot | #5105 (pi0 pad-in-loss bug) | excluded | PRs #5106/#5108 already open (claimed within 1 day of filing) |
| 2026-10-28 (r2) | huggingface/lerobot | #5118 (camera-slot bug) | excluded | 3 fixes in flight (fork branch + PR #5120 + PR #5129 covering 6 policies) |
| 2026-10-28 (r2) | foxglove/mcap | #1907 (rust fuzz targets) | excluded | issue author already opened PR #1908; maintainer undecided on fuzzing |
| 2026-10-28 (r3) | huggingface/lerobot | documentation-labeled backlog | excluded | only #5116 remains unassigned (already excluded r1: imminent PR) |
| 2026-10-28 (r3) | huggingface/lerobot | #1998 (distributed v2.1→v3.0) | watchlist | active 4-part PR series in flight (#3673/#3675/#3676) — direction taken; stay away |
| 2026-10-28 (r3) | rerun-io/rerun | #12984 (compare precomputed RRDs), #12890 (TextLog filtering) | watchlist | fresh, unassigned, awaiting triage; #12984 aligns with dataset-comparison UX |

**Learnings (2026-10-28, round 3)**: MCAP→LeRobot ingestion has NO upstream equivalent (official scripts cover only v21→v30) — our converter fills a real gap; next contribution move: propose upstream (example + docs first) once GitHub credentials are provided. Distributed conversion (#1998) is taken — do not compete.

**Learnings (2026-10-28, round 2)**: hot-repo bug fixes are a red ocean (3 parallel fixes on one bug) — confirmed twice. Pivot contribution strategy to *feature PRs born from project work* (upstream has NO mcap/hdf5/rlds→lerobot converters; official v21→v30 script is single-machine only) where we hold unique context and there is no race. Next: build mcap→lerobot converter, then propose upstream.

**Learnings (2026-10-28)**: LeRobot fresh well-scoped issues get claimed in ~24h → routine must run at high frequency; prefer issues where our Babel/QC work gives unique context (converters, MCAP/HDF5/RLDS ingestion) over generic drive-by fixes; policy check done (AGENTS.md + CONTRIBUTING.md present, AI-assisted contributions anticipated; uv, Python 3.12+, pre-commit ruff/mypy, pytest marks). Awaiting from user: GitHub username + target role before first fork/PR.

### Real-data validation record (2026-10-28, r5)
- Target: `lerobot/pusht` v2.1 revision (206 episodes / 25,650 frames), real community dataset
- Tool: `elake convert v21-to-v30 --drop-videos` + `elake qc` (PASS, 0 errors)
- Compared against the OFFICIAL v3.0 conversion on `main`:
  tasks ✅ · episodes metadata (all 8 structural columns over 206 rows) ✅ ·
  50/50 shared stats columns (element-wise, vector features) ✅ ·
  stats.json aggregate (count-weighted moments match upstream) ✅ ·
  data rows 25,650 × 10 shared columns value-identical ✅ ·
  info.json key fields ✅ — only video stats columns differ (dropped by request)
- Bug found & fixed by this validation: v2.1 per-episode stats are per-dimension
  LISTS, not scalars — aggregation rewritten element-wise (regression test added)

### Benchmark record (2026-10-28, r8; scripts/benchmark.py, best of 3)
- v2.1→v3.0 on real lerobot/pusht (25,650 rows, videos dropped): 0.12 s →
  **216,233 rows/s**; DROID tabular extrapolation ~2 min
- mcap→v3.0 synthetic 10k JointState frames (CDR decode dominates): 0.16 s →
  **61,369 rows/s**; extrapolation ~6 min
- Public reference: LanceDB full DROID conversion 34 min (video+vector incl.)

### QC scoring record (2026-10-28, r9)
- `elake score`: rule engine live — frozen_dim / saturated / action_spike (robust
  z) / low_motion (dataset-median energy reference); 4 detector tests green
- Real-data calibration: lerobot/pusht conversion output → 206/206 episodes
  score 100.0, zero false positives (clean sim benchmark)

### Platform loop first cut (2026-10-28, r10) — Phase 2 started
- `platform/catalog.py`: SQLite registry (stdlib only) — lineage, qc verdict,
  mean score, upsert-by-path; `elake catalog add/list`
- `elake ingest`: convert -> qc gate (exit 1 on structural failure) -> score
  -> register, one command; 4 e2e tests (33/33 green)
- Real-data drill: pusht conversion output registered (#1, v3.0, qc=PASS, 100.0)

### Self-hosted serving (2026-10-28, r11)
- `elake serve`: stdlib HTTP API over the catalog (healthz, list/detail JSON,
  POST register, static .rrd preview serving) — zero web-framework deps
- `docker/Dockerfile` + `docker-compose.yml`: one-command on-prem
  (volumes: catalog state + datasets RO)
- 4 API tests (live server on ephemeral port); 37/37 green

### Candidate: OpenRAL/openral #250 (2026-10-28, r12) — READY, awaiting user approval
- bug + help-wanted, unassigned, 0 comments, open since 2026-09-08; repo active (pushed 10-27)
- AlohaHAL publishes ros2_control topics a real ALOHA doesn't have; issue author
  established the xs_sdk contract and suggested the fix; repo docs/tests already
  carry "move with the #250 fix" banners — maintainer-aligned
- local fix implemented + verified: 27/27 of their unit tests pass
  (patch: ../contributions/openral-250-aloha-xs-sdk.patch)
- open design point to confirm in comment: gripper as motor group vs joint_single
  (defaults to group "arm"/"gripper" per stock motor_config)

### Phase 1 close-out decisions (2026-10-28, r13)
- **RLDS→v3.0: DEFERRED to Phase 3.** Honest scope call: RLDS datasets are
  TFRecord+protobuf with TFDS metadata; parsing without the TF stack is a
  large surface for marginal benefit — the formats teams actually hand us
  (rosbag2/MCAP, ALOHA HDF5, LeRobot v2.1) are all covered and validated.
  Revisit if a concrete RLDS ingestion request lands.
- Phase 1 declared functionally complete: 3 converters + qc/score + benchmark.

### 2026-10-28 r13 — user approvals received; FIRST PRs OUT
- private repo github.com/kittimzhe/embodied-lake (main, 21+ commits)
- OpenRAL/openral#451 PR opened (fix #250, xs_sdk contract) + design-point comment
- LeRobot #4368 RFC comment posted (converter landscape data point)
- LeRobot #5130 example proposal opened (rosbag2→v3.0 recipe)
- Draft B (#4443 converter-table entry) deferred until repo goes public
