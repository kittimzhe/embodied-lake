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

    p_convert = sub.add_parser("convert", help="convert between dataset formats")
    csub = p_convert.add_subparsers(dest="converter", required=True)
    p_v21 = csub.add_parser("v21-to-v30", help="LeRobot v2.1 -> v3.0 (video-free datasets)")
    p_v21.add_argument("src", help="source dataset directory (v2.1)")
    p_v21.add_argument("dst", help="destination dataset directory (v3.0)")
    p_v21.add_argument(
        "--size-mb",
        type=float,
        default=100.0,
        help="target data-file size in MB before rolling over (default: 100)",
    )
    p_v21.add_argument(
        "--drop-videos",
        action="store_true",
        help="convert tabular data only; drop video features instead of failing",
    )

    p_prev = sub.add_parser("preview", help="render a dataset to a Rerun .rrd (headless)")
    p_prev.add_argument("src", help="local dataset directory (v3.0)")
    p_prev.add_argument("--out", default=None, help="output .rrd path (default: <dataset>.rrd)")
    p_prev.add_argument("--episodes", nargs="*", type=int, default=None, help="episode indices (default: all)")

    p_qc = sub.add_parser("qc", help="structural quality checks on a local dataset")
    p_qc.add_argument("src", help="dataset directory")

    p_mcap = csub.add_parser("mcap-to-v30", help="MCAP (rosbag2) -> LeRobot v3.0")
    p_mcap.add_argument("src", nargs="+", help="one or more .mcap files (one episode each)")
    p_mcap.add_argument("dst", help="destination dataset directory (v3.0)")
    p_mcap.add_argument("--state-topic", required=True, help="topic providing observation.state")
    p_mcap.add_argument("--action-topic", default=None, help="topic providing action (default: mirror state)")
    p_mcap.add_argument("--state-field", default="position", help="message field to extract (default: position)")
    p_mcap.add_argument("--task", default="demonstration", help="task label for all episodes")
    p_mcap.add_argument("--fps", type=int, default=30)
    p_mcap.add_argument("--robot-type", default=None)

    p_aloha = csub.add_parser("hdf5-to-v30", help="ALOHA-style HDF5 -> LeRobot v3.0")
    p_aloha.add_argument("src", nargs="+", help="one or more .hdf5 episode files")
    p_aloha.add_argument("dst", help="destination dataset directory (v3.0)")
    p_aloha.add_argument("--task", default="manipulation demo")
    p_aloha.add_argument("--fps", type=int, default=50, help="ALOHA default recording rate")
    p_aloha.add_argument("--include-qvel", action="store_true", help="also emit observation.velocity")
    p_aloha.add_argument("--prefix", default="", help="dataset path prefix for variants (e.g. data/obs)")

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

    if args.command == "preview":
        try:
            import rerun  # noqa: F401
        except ImportError:
            print("error: rerun-sdk required for preview (pip install rerun-sdk)", file=sys.stderr)
            return 1
        from .viz.rrd import render_dataset

        out = render_dataset(args.src, args.out, args.episodes)
        print(f"preview written: {out}")
        return 0

    if args.command == "qc":
        from .qc.structural import qc_dataset

        report = qc_dataset(args.src)
        for issue in report.issues:
            print(str(issue))
        print(report.summary())
        return 0 if report.ok else 1

    if args.command == "convert" and args.converter == "v21-to-v30":
        try:
            from .babel.v21_to_v30 import convert_v21_to_v30
        except ImportError as exc:
            print(f"error: missing optional dependency for conversion: {exc}", file=sys.stderr)
            return 1
        try:
            report = convert_v21_to_v30(
                args.src, args.dst,
                data_file_size_in_mb=args.size_mb,
                drop_videos=args.drop_videos,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"converted {report.episodes} episodes / {report.frames} frames "
            f"-> {report.data_files_written} data file(s), {report.tasks} task(s); "
            f"stats: {'yes' if report.has_stats else 'no'}"
        )
        for note in report.notes:
            print(f"note: {note}")
        return 0

    if args.command == "convert" and args.converter == "mcap-to-v30":
        try:
            from .babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30
        except ImportError as exc:
            print(f"error: missing optional dependency for conversion: {exc}", file=sys.stderr)
            return 1
        mapping = McapMapping(
            state_topic=args.state_topic,
            action_topic=args.action_topic,
            state_field=args.state_field,
            task=args.task,
            fps=args.fps,
            robot_type=args.robot_type,
        )
        try:
            report = convert_mcap_to_v30(args.src, args.dst, mapping)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"converted {len(report.files)} file(s): {report.episodes} episodes / "
            f"{report.frames} frames; task='{args.task}'"
        )
        for t in report.skipped_topics:
            print(f"skipped undecodable topic: {t}")
        for note in report.notes:
            print(f"note: {note}")
        return 0

    if args.command == "convert" and args.converter == "hdf5-to-v30":
        try:
            from .babel.aloha_to_lerobot import AlohaMapping, convert_aloha_to_v30
        except ImportError as exc:
            print(f"error: missing optional dependency for conversion: {exc}", file=sys.stderr)
            return 1
        try:
            report = convert_aloha_to_v30(
                args.src,
                args.dst,
                AlohaMapping(
                    prefix=args.prefix,
                    include_qvel=args.include_qvel,
                    task=args.task,
                    fps=args.fps,
                ),
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"converted {len(report.files)} file(s): {report.episodes} episodes / "
            f"{report.frames} frames"
        )
        for note in report.notes:
            print(f"note: {note}")
        return 0

    return 2  # unreachable: subparsers are required


if __name__ == "__main__":
    raise SystemExit(main())
