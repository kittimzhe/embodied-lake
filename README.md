# embodied-lake (working title)

> LeRobot-native, self-hostable data platform for embodied AI — an open alternative to the (closed-source) Foxglove Data Platform, built on the [LeRobot dataset format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) (the de-facto standard: 5,750+ datasets on the Hugging Face Hub) and [Rerun](https://github.com/rerun-io/rerun) as the multimodal data layer.

**Status: Phase 0 — scaffolding.** See [docs/roadmap.md](docs/roadmap.md).

## Why

Embodied AI is bottlenecked on data infrastructure. The chain
`collect → annotate → QC → manage → simulate → train` has open gaps
(no format Babel layer, no robot-native annotation tooling, no self-hosted
platform since Foxglove went closed-source in 2024). embodied-lake targets:

1. **Babel layer** — convert MCAP / rosbag2 / HDF5 (ALOHA) / RLDS /
   LeRobot v2.1 ⇄ v3.0, with incremental migration.
2. **QC pipeline** — batch scoring via VLM reward models (e.g. NVIDIA Flora),
   rule filters, quality reports.
3. **Dataset management** — catalog, versioning, multi-tenancy, on-prem
   deployment (Docker Compose), Rerun-powered episode preview.

## Install & quick start (Phase 0 tooling)

```bash
pip install -e .          # zero heavy deps for the base CLI
elake inspect Embodied-AI-6/dataset        # inspect a Hub dataset
elake inspect ./my_dataset --json          # inspect a local dataset
```

## Repo layout

```
src/embodied_lake/
  inspect.py       # dataset card inspector (stdlib-only, v2/v3)
  babel/           # Phase 1: format conversion layer
  qc/              # Phase 1: quality-control pipeline
  platform/        # Phase 2: catalog / multi-tenancy / API
tests/             # pytest suite (offline fixtures + optional network tests)
docs/roadmap.md    # phased plan & evidence ledger
```

## License

Apache-2.0 (matching the LeRobot / Rerun ecosystem).

## Acknowledgements

Built on top of — and contributing back to — [huggingface/lerobot](https://github.com/huggingface/lerobot) and [rerun-io/rerun](https://github.com/rerun-io/rerun).
