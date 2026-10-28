# Contributing to embodied-lake

Thanks for considering a contribution!

## Setup

```bash
git clone https://github.com/kittimzhe/embodied-lake.git
cd embodied-lake
python3.12 -m venv .venv
.venv/bin/pip install pyarrow pandas numpy pytest mcap mcap-ros2-support h5py rerun-sdk
PYTHONPATH=src .venv/bin/python -m pytest tests/
```

No heavy ML dependencies are needed for development — conversion and QC
deliberately avoid requiring `lerobot`/`torch`.

## Ground rules

- **Tests required** for every bug fix and feature. The suite is offline by
  default (network/real-data tests skip gracefully when unreachable).
- **Upstream compatibility over convenience**: artifacts this project writes
  (tasks.parquet layout, stats key set, constant metadata columns) match what
  real LeRobot v3.0 datasets on the Hub contain; keep it that way. When in
  doubt, validate against a real dataset (see `docs/roadmap.md` for the
  method used on lerobot/pusht).
- **Dependency discipline**: base CLI stays stdlib-only; converters may use
  pyarrow/pandas/numpy + format-specific readers (`mcap`, `h5py`). Nothing
  pulls in training stacks.
- **DCO**: sign your commits (`git commit -s`).

## Adding a converter

1. Mirror the structure of `babel/mcap_to_lerobot.py` (mapping dataclass,
   report dataclass, fail-fast errors, notes for skipped capabilities).
2. Emit artifacts through the shared v3.0 writer semantics (see
   `babel/_v30.py` users) so every converter's output passes `elake qc`.
3. Add tests: synthetic source (round-trip value precision), error paths,
   and a dogfood pass — `elake inspect`/`elake qc` on your output.

## Reporting bugs

Open an issue with: the command you ran, the dataset/format involved (a
minimal reproducer if possible), and the full traceback.
