"""Validate a Hub task-tarball dataset WITHOUT downloading the videos.

AgiBotWorld-style repos ship one tar.gz per task (GBs, mostly videos), with
the tabular part (data/*.parquet + meta/*) at the head of the archive. This
script streams the tarball, extracts only the tabular members, and runs the
embodied-lake QC + scorer + report over them — a full data-quality audit for
the cost of a few MB of transfer.

Usage:
    python scripts/validate_hub_tarball.py <tarball-url> [--out DIR] [--task N]

Example (AgiBotWorld-Beta task 676 — reproduces the quaternion findings on
OpenDriveLab/AgiBot-World#164):
    python scripts/validate_hub_tarball.py \
        https://huggingface.co/datasets/OpenWAM/AgiBotWorld-Beta/resolve/main/676.tar.gz
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path


def stream_extract_tabular(url: str, out: Path) -> list[str]:
    """Extract data/*.parquet + meta files from a streaming tar.gz."""
    wanted_suffixes = (
        "/data/chunk-", "/meta/info.json", "/meta/stats.json",
        "/meta/tasks.parquet", "/meta/episodes/",
    )
    out.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    if Path(url).exists():  # local fixture path (tests / offline replay)
        resp = open(url, "rb")  # noqa: SIM115
    else:
        resp = urllib.request.urlopen(url, timeout=60)

    class _Counting:
        """Progress-wrapping raw stream (MB read so far)."""

        def __init__(self, r):
            self._r = r
            self.n = 0

        def read(self, n=-1):
            b = self._r.read(n)
            self.n += len(b)
            return b

        def getattr(self, name):
            return getattr(self._r, name)

    raw = _Counting(resp)
    tf = tarfile.open(fileobj=raw, mode="r|gz")
    for m in tf:
        if raw.n - getattr(_Counting, "_last", 0) > 50 * 1024 * 1024:
            print(f"  … {raw.n / 1e6:.0f} MB streamed, at {m.name}", file=sys.stderr)
            _Counting._last = raw.n
        rel = None
        if any(s in m.name for s in wanted_suffixes):
            # strip the leading AgiBotWorld-Beta/<task>/ prefix
            parts = m.name.split("/", 2)
            rel = "/".join(parts[2:]) if len(parts) == 3 else m.name
        if rel is None:
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        f = tf.extractfile(m)
        if f is None:
            continue
        dest.write_bytes(f.read())
        got.append(rel)
        if rel == "meta/tasks.parquet":  # last tabular member in task layout
            break  # closing the stream cancels the rest of the download
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="tarball URL (…/resolve/main/<task>.tar.gz)")
    ap.add_argument("--out", default=None, help="output dir (default: ./<task>_tabular)")
    args = ap.parse_args()

    task = args.url.rsplit("/", 1)[-1].removesuffix(".tar.gz").split(".")[0]
    out = Path(args.out) if args.out else Path(f"{task}_tabular")
    print(f"streaming {args.url} … (task tarballs are GBs; tabular head"
          f" extract stops the download once meta is reached)")
    got = stream_extract_tabular(args.url, out)
    print(f"extracted {len(got)} tabular member(s) → {out}")
    if not any(g.startswith("data/") for g in got):
        print("error: no data parquet found — not a task-tarball layout?", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from embodied_lake.qc.report import render_report  # noqa: E402
    from embodied_lake.qc.scoring import score_dataset  # noqa: E402
    from embodied_lake.qc.structural import qc_dataset  # noqa: E402

    qc = qc_dataset(out)
    print(f"qc: {'PASS' if qc.ok else 'FAIL'} ({len(qc.issues)} issue(s))")
    rep = score_dataset(out)
    if rep.episodes:
        flagged = [e for e in rep.episodes if e.findings]
        print(f"score: mean {rep.mean_score:.1f} over {len(rep.episodes)} episodes; "
              f"{len(flagged)} with findings")
        for e in flagged[:5]:
            for f in e.findings[:4]:
                print(f"  ep {e.episode_index}: {f.check}: {f.detail}")
    md = render_report(out)
    print(f"report: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
