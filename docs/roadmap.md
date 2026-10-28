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
- [ ] `elake convert hdf5(aloha)→lerobot-v3`
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
