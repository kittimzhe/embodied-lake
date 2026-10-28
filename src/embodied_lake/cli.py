"""``elake`` command-line entry point.

Subcommands:
    inspect   Show a dataset card (format version, episodes, frames, features).
"""

from __future__ import annotations

import argparse
import json
import sys

from .inspect import inspect as inspect_dataset
from .inspect import render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elake",
        description="embodied-lake: LeRobot-native embodied-data platform tooling",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect", help="inspect a LeRobot dataset (local dir or 'org/name' HF repo)"
    )
    p_inspect.add_argument("source", help="dataset directory or Hugging Face repo id")
    p_inspect.add_argument(
        "--json", action="store_true", help="emit the normalized card as JSON"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "inspect":
        try:
            card = inspect_dataset(args.source)
        except Exception as exc:  # keep CLI exit codes stable
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "source": card.source,
                        "codebase_version": card.codebase_version,
                        "robot_type": card.robot_type,
                        "total_episodes": card.total_episodes,
                        "total_frames": card.total_frames,
                        "total_tasks": card.total_tasks,
                        "fps": card.fps,
                        "splits": card.splits,
                        "data_path": card.data_path,
                        "video_path": card.video_path,
                        "features": card.features,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(render(card))
        return 0

    return 2  # unreachable: subparsers are required


if __name__ == "__main__":
    raise SystemExit(main())
