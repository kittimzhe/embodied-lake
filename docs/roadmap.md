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
- [ ] `elake convert mcap→lerobot-v3`
- [ ] `elake convert hdf5(aloha)→lerobot-v3`
- [ ] `elake convert v2.1→v3.0` (incremental)
- [ ] `elake convert rlds→lerobot-v3`
- [ ] Benchmark: DROID 76k trajectories end-to-end (reference: LanceDB full conversion 34 min)
- [ ] `elake qc score` (Flora-Rewarder orchestration) + quality report
- Contribution link: generic converters → LeRobot PRs (upstream ships only v2.1→v3.0)

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
| 2026-10-28 | inspector passes offline+network tests (v3.0 dataset card) | `pytest tests/` output |
