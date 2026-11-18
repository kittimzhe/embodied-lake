"""Sweep AgiBotWorld-Beta task tarballs with the tabular-head validator.

Usage: .venv/bin/python scripts/sweep_agibot.py <task> [<task> ...]
Writes sweep-<ts>.json next to cwd with per-task summaries.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

URL = "https://huggingface.co/datasets/OpenWAM/AgiBotWorld-Beta/resolve/main/{task}.tar.gz"
TIMEOUT_S = int(os.environ.get("SWEEP_TIMEOUT", "240"))


def sweep(tasks: list[str]) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path("/tmp/agibot_sweep")
    out_dir.mkdir(exist_ok=True)
    results = []
    for t in tasks:
        dst = out_dir / t
        if dst.exists():  # a previous failed run may leave partial extraction
            import shutil
            shutil.rmtree(dst)
        cmd = [
            str(root / ".venv/bin/python"), str(root / "scripts/validate_hub_tarball.py"),
            URL.format(task=t), "--out", str(dst),
        ]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
            ok, out, err = p.returncode == 0, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            ok, out, err = False, "", f"timed out after {TIMEOUT_S}s"
        dt = time.time() - t0
        quats, frozen, score_line, qc_line = [], [], "", ""
        for line in out.splitlines():
            if "quaternion_nonunit" in line:
                quats.append(line.split("quaternion_nonunit: ", 1)[-1].strip())
            elif line.startswith("score:"):
                score_line = line.strip()
            elif line.startswith("qc:"):
                qc_line = line.strip()
            elif "frozen_dim" in line and len(frozen) < 6:
                frozen.append(line.split("frozen_dim: ", 1)[-1].strip())
        results.append({
            "task": t, "ok": ok, "seconds": round(dt),
            "qc": qc_line, "score": score_line,
            "quaternion_findings": quats[:8], "frozen_dims": frozen[:6],
        })
        print(f"[{t}] {'OK' if ok else 'FAIL'} {dt:.0f}s | {score_line or err[-60:]}", flush=True)
    return results


if __name__ == "__main__":
    tasks = sys.argv[1:]
    res = sweep(tasks)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(f"sweep-{stamp}.json")
    path.write_text(json.dumps(res, indent=1))
    print("saved:", path)
