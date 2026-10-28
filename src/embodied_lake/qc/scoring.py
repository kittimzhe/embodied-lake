"""Rule-based quality scoring for LeRobot v3.0 datasets.

Complements the structural QC (``qc/structural.py``) with content-level
checks computed from the actual feature values — the failure modes teams hit
most in real collection runs:

- **frozen joints**: dimension whose value never changes over an episode
  (dead encoder, dropped topic, unmapped channel)
- **saturation**: values pinned at an extremum (sensor clipping)
- **action spikes**: frame-to-frame jumps far beyond the episode's typical
  delta (teleop glitch, controller hiccup)
- **low motion**: an episode whose overall motion energy is far below the
  dataset median (nothing happened; failed grasp immediately)

Output: per-episode findings + a 0-100 score, plus a dataset-level JSON
report. All thresholds are heuristic defaults, overridable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..inspect import inspect

# ---- tunable heuristics ---------------------------------------------------- #
FROZEN_ABS_TOL = 1e-6       # a dim is frozen if (max-min) <= this
SPIKE_Z = 12.0              # |delta| > SPIKE_Z * robust-sigma of deltas
LOW_MOTION_FACTOR = 0.05    # episode motion energy < factor * dataset median
SATURATION_MIN_COUNT = 3    # >= this many exact-extremum repeats counts


@dataclass
class Finding:
    check: str
    detail: str
    penalty: float


@dataclass
class EpisodeScore:
    episode_index: int
    length: int
    score: float
    findings: list[Finding] = field(default_factory=list)


@dataclass
class ScoreReport:
    dataset: Path
    episodes: list[EpisodeScore] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return float(np.mean([e.score for e in self.episodes])) if self.episodes else 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": str(self.dataset),
                "mean_score": round(self.mean_score, 1),
                "episodes": [
                    {
                        "episode_index": e.episode_index,
                        "length": e.length,
                        "score": round(e.score, 1),
                        "findings": [
                            {"check": f.check, "detail": f.detail, "penalty": f.penalty}
                            for f in e.findings
                        ],
                    }
                    for e in self.episodes
                ],
            },
            indent=2,
            ensure_ascii=False,
        )


def _feature_matrix(tbl_dict: dict, feature: str) -> np.ndarray | None:
    """Column -> (frames, dims) float matrix; scalar features -> (frames, 1)."""
    col = tbl_dict.get(feature)
    if col is None or len(col) == 0:
        return None
    first = col[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        try:
            m = np.asarray(col, dtype=np.float64)
            if m.ndim == 2:
                return m
        except (ValueError, TypeError):
            pass
        return np.asarray([list(v) for v in col], dtype=np.float64)
    return np.asarray(col, dtype=np.float64).reshape(-1, 1)


def score_episode(
    episode_index: int,
    tables: dict[str, np.ndarray],
    dataset_motion_ref: float | None = None,
) -> EpisodeScore:
    findings: list[Finding] = []
    n_frames = next(iter(tables.values())).shape[0] if tables else 0

    for feat, m in tables.items():
        if m.ndim != 2 or m.shape[0] < 2:
            continue
        mins = m.min(axis=0)
        maxs = m.max(axis=0)
        # frozen joints
        frozen = [d for d in range(m.shape[1]) if maxs[d] - mins[d] <= FROZEN_ABS_TOL]
        if frozen:
            findings.append(
                Finding(
                    "frozen_dim",
                    f"{feat}: dims {frozen[:8]} never change over {m.shape[0]} frames",
                    10.0 * min(len(frozen), 3),
                )
            )
        # saturation: exact repeats of the per-dim extremum
        for d in range(m.shape[1]):
            lo = np.sum(m[:, d] == mins[d])
            hi = np.sum(m[:, d] == maxs[d])
            hit = max(lo, hi)
            if hit >= SATURATION_MIN_COUNT and hit > 0.9 * m.shape[0]:
                findings.append(
                    Finding(
                        "saturated",
                        f"{feat}[{d}]: pinned at {'min' if lo >= hi else 'max'} for "
                        f"{hit}/{m.shape[0]} frames",
                        10.0,
                    )
                )
        # spikes (delta z-score, robust sigma = 1.4826 * MAD)
        if feat.startswith("action."):
            deltas = np.abs(np.diff(m, axis=0))
            if deltas.size:
                sigma = 1.4826 * np.median(np.abs(deltas - np.median(deltas)))
                if sigma > FROZEN_ABS_TOL:
                    worst = float(deltas.max() / sigma)
                    if worst > SPIKE_Z:
                        findings.append(
                            Finding(
                                "action_spike",
                                f"{feat}: max delta is {worst:.1f}x robust sigma",
                                15.0,
                            )
                        )

    # low motion across all features
    if dataset_motion_ref is not None and tables:
        energy = 0.0
        for m in tables.values():
            if m.shape[0] >= 2:
                energy += float(np.mean(np.abs(np.diff(m, axis=0))))
        if energy < LOW_MOTION_FACTOR * dataset_motion_ref:
            findings.append(
                Finding(
                    "low_motion",
                    f"motion energy {energy:.2e} << dataset median {dataset_motion_ref:.2e}",
                    25.0,
                )
            )

    score = max(0.0, 100.0 - sum(f.penalty for f in findings))
    return EpisodeScore(
        episode_index=episode_index, length=int(n_frames), score=score, findings=findings
    )


def score_dataset(src: str | Path) -> ScoreReport:
    root = Path(src)
    card = inspect(str(root))
    if card.codebase_version != "v3.0":
        raise ValueError(f"scoring requires a v3.0 dataset, got {card.codebase_version}")
    eps_tbl = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()

    per_episode: list[tuple[int, dict[str, np.ndarray]]] = []
    for row in range(len(eps_tbl["episode_index"])):
        ep = eps_tbl["episode_index"][row]
        path = (
            root
            / "data"
            / f"chunk-{eps_tbl['data/chunk_index'][row]:03d}"
            / f"file-{eps_tbl['data/file_index'][row]:03d}.parquet"
        )
        d = pq.read_table(path).to_pydict()
        feats = {}
        for feat in [c for c in d if c.startswith(("observation.", "action."))]:
            m = _feature_matrix(d, feat)
            if m is not None and m.shape[0] >= 2:
                feats[feat] = m
        if feats:
            per_episode.append((ep, feats))

    # dataset-level motion reference: median per-episode energy
    def energy(feats: dict[str, np.ndarray]) -> float:
        return sum(
            float(np.mean(np.abs(np.diff(m, axis=0)))) for m in feats.values()
        )

    ref = float(np.median([energy(f) for _, f in per_episode])) if per_episode else None

    report = ScoreReport(dataset=root)
    for ep, feats in per_episode:
        report.episodes.append(score_episode(ep, feats, dataset_motion_ref=ref))
    return report
