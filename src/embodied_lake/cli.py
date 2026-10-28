"""``elake`` command-line entry point.

Subcommands:
    inspect   Show a dataset card (format version, episodes, frames, features).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

    p_cat = sub.add_parser("catalog", help="dataset registry (local SQLite)")
    cat_sub = p_cat.add_subparsers(dest="action", required=True)
    p_cat_add = cat_sub.add_parser("add", help="register/refresh a dataset")
    p_cat_add.add_argument("dataset")
    p_cat_add.add_argument("--source", default=None, help="e.g. mcap:/path/rec.mcap")
    p_cat_add.add_argument("--catalog", default=None, help="catalog dir (default: .elake)")
    p_cat_ls = cat_sub.add_parser("list", help="list registered datasets")
    p_cat_ls.add_argument("--catalog", default=None, help="catalog dir (default: .elake)")

    p_serve = sub.add_parser("serve", help="serve the catalog over HTTP (stdlib only)")
    p_serve.add_argument("--catalog", default=None, help="catalog dir (default: .elake)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8317)

    p_ing = sub.add_parser("ingest", help="convert -> qc -> score -> catalog, one command")
    p_ing.add_argument("src", nargs="+")
    p_ing.add_argument("dst")
    p_ing.add_argument("--mode", required=True, choices=["v21-to-v30", "mcap-to-v30", "hdf5-to-v30"])
    p_ing.add_argument("--state-topic", default=None)
    p_ing.add_argument("--action-topic", default=None)
    p_ing.add_argument("--task", default=None)
    p_ing.add_argument("--fps", type=int, default=None)
    p_ing.add_argument("--drop-videos", action="store_true")
    p_ing.add_argument("--no-score", action="store_true")
    p_ing.add_argument("--catalog", default=None, help="catalog dir (default: .elake)")

    p_snap = sub.add_parser("snapshot", help="dataset snapshots (integrity ledger)")
    ps = p_snap.add_subparsers(dest="snap_command", required=True)
    ps_c = ps.add_parser("create", help="record a content-addressed snapshot")
    ps_c.add_argument("dataset")
    ps_c.add_argument("-m", "--message", default="")
    ps_c.add_argument("--catalog", default="state/catalog.db", help="snapshot DB path")
    ps_l = ps.add_parser("list", help="snapshot history for a dataset")
    ps_l.add_argument("dataset", nargs="?", default=None)
    ps_l.add_argument("--catalog", default="state/catalog.db", help="snapshot DB path")
    ps_v = ps.add_parser("verify", help="current tree vs latest snapshot")
    ps_v.add_argument("dataset")
    ps_v.add_argument("--catalog", default="state/catalog.db", help="snapshot DB path")
    ps_d = ps.add_parser("diff", help="changeset between two pins (default prev→latest)")
    ps_d.add_argument("dataset")
    ps_d.add_argument("--from", dest="from_id", type=int, default=None)
    ps_d.add_argument("--to", dest="to_id", type=int, default=None)
    ps_d.add_argument("--catalog", default="state/catalog.db", help="snapshot DB path")

    p_diff = sub.add_parser("diff", help="compare two datasets (conversion validation)")
    p_diff.add_argument("left", help="dataset A (e.g. v2.1 source)")
    p_diff.add_argument("right", help="dataset B (e.g. converted v3.0)")
    p_diff.add_argument("--expect-migration", action="store_true",
                        help="treat version/path-template drift as expected (v2.1→v3.0)")

    p_score = sub.add_parser("score", help="content-level quality scoring (rules)")
    p_score.add_argument("src", help="local v3.0 dataset directory")
    p_score.add_argument("--json-out", default=None, help="write full report JSON here")
    p_score.add_argument("--min-score", type=float, default=None,
                         help="exit 1 if any episode scores below this")

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

    if args.command == "serve":
        from .platform.api import serve

        print(f"serving catalog on http://{args.host}:{args.port} (ctrl-c to stop)")
        serve(Path(args.catalog or ".elake"), host=args.host, port=args.port)
        return 0

    if args.command == "catalog":
        from .platform.catalog import Catalog

        cat_dir = Path(args.catalog or ".elake")
        cat = Catalog(cat_dir)
        try:
            if args.action == "add":
                entry = cat.register(args.dataset, source=getattr(args, "source", None))
                print(f"registered #{entry.id}: {entry.name} ({entry.version}, "
                      f"qc={'PASS' if entry.qc_ok else 'FAIL'}, score={entry.mean_score})")
                return 0
            if args.action == "list":
                rows = cat.list()
                if not rows:
                    print("(catalog empty)")
                for e in rows:
                    print(f"#{e.id:<3} {e.name:<28} {e.version:<6} eps={e.episodes:<5} "
                          f"frames={e.frames:<7} qc={'PASS' if e.qc_ok else 'FAIL':<4} "
                          f"score={e.mean_score if e.mean_score is not None else '-'}")
                return 0
        finally:
            cat.close()

    if args.command == "ingest":
        from .platform.catalog import Catalog

        lineage: dict = {"mode": args.mode}
        # 1) convert
        if args.mode == "v21-to-v30":
            from .babel.v21_to_v30 import convert_v21_to_v30

            report = convert_v21_to_v30(args.src[0], args.dst, drop_videos=args.drop_videos)
            lineage["source"] = args.src[0]
            lineage["converter"] = "v21_to_v30"
            print(f"converted: {report.episodes} episodes / {report.frames} frames")
        elif args.mode == "mcap-to-v30":
            from .babel.mcap_to_lerobot import McapMapping, convert_mcap_to_v30

            mapping = McapMapping(
                state_topic=args.state_topic or "",
                action_topic=args.action_topic,
                task=args.task or "demonstration",
                fps=args.fps or 30,
            )
            rep = convert_mcap_to_v30(args.src, args.dst, mapping)
            lineage.update({"source": ",".join(args.src), "converter": "mcap_to_v30",
                            "state_topic": args.state_topic})
            print(f"converted: {rep.episodes} episodes / {rep.frames} frames")
        else:
            from .babel.aloha_to_lerobot import AlohaMapping, convert_aloha_to_v30

            mapping = AlohaMapping(
                task=args.task or "manipulation demo", fps=args.fps or 50,
                include_qvel=args.include_qvel,
            )
            rep = convert_aloha_to_v30(args.src, args.dst, mapping)
            lineage.update({"source": ",".join(args.src), "converter": "aloha_to_v30"})
            print(f"converted: {rep.episodes} episodes / {rep.frames} frames")

        # 2) structural qc gate
        from .qc.structural import qc_dataset

        qc = qc_dataset(args.dst)
        for issue in qc.issues:
            print(str(issue))
        print(qc.summary())
        if not qc.ok:
            return 1

        # 3) scoring + 4) catalog
        cat = Catalog(Path(args.catalog or ".elake"))
        try:
            entry = cat.register(
                args.dst, source=lineage.get("source"), lineage=lineage,
                with_scoring=not args.no_score,
            )
            # 5) pin what passed QC: content-addressed snapshot
            from .platform.snapshots import SnapshotStore

            store = SnapshotStore.open(Path(args.catalog or ".elake") / "catalog.db")
            try:
                snap = store.create(
                    args.dst,
                    message=f"ingest: {args.mode} (qc PASS"
                    + (f", score {entry.mean_score:.1f}" if entry.mean_score is not None else "")
                    + ")",
                )
            finally:
                store._conn.close()
        finally:
            cat.close()
        print(f"registered #{entry.id}: {entry.name} qc=PASS score={entry.mean_score}")
        print(f"snapshot #{snap.id}: {snap.content_hash[:16]}… ({len(snap.manifest)} files)")
        return 0

    if args.command == "snapshot":
        from .platform.snapshots import SnapshotStore

        store = SnapshotStore.open(args.catalog)
        if args.snap_command == "create":
            s = store.create(args.dataset, message=args.message)
            print(f"snapshot #{s.id} · {s.dataset_name} · {s.content_hash[:16]}… · {len(s.manifest)} files")
        elif args.snap_command == "list":
            for s in store.list(args.dataset):
                import datetime

                ts = datetime.datetime.fromtimestamp(s.created_at).strftime("%Y-%m-%d %H:%M")
                print(f"#{s.id:<3} {ts}  {s.content_hash[:16]}…  {s.message or '(no message)'}")
        elif args.snap_command == "diff":
            try:
                rows = store.changeset(args.dataset, args.from_id, args.to_id)
            except ValueError as e:
                print(f"error: {e}")
                return 1
            if not rows:
                print("no file-level changes between pins")
            for r in rows:
                print(f"  {r['change']:<8} {r['file']}" + (f"  ({r['detail']})" if r.get("detail") else ""))
            return 0
        elif args.snap_command == "verify":
            ok, expected, drift = store.verify(args.dataset)
            if ok:
                print(f"OK — matches snapshot {expected[:16]}…")
                return 0
            print(f"DRIFT vs {expected[:16]}…:")
            for d in drift[:20]:
                print(f"  {d}")
            return 1
        return 0

    if args.command == "diff":
        from .qc.diff import diff_datasets

        rep = diff_datasets(args.left, args.right, expect_migration=args.expect_migration)
        print(rep.summary())
        return 0 if rep.ok else 1

    if args.command == "score":
        from .qc.scoring import score_dataset

        rep = score_dataset(args.src)
        flagged = [e for e in rep.episodes if e.findings]
        for e in sorted(rep.episodes, key=lambda x: x.score)[:10]:
            for f in e.findings:
                print(f"ep {e.episode_index:<5} score={e.score:<6.1f} {f.check}: {f.detail}")
        print(
            f"{rep.dataset}: {len(rep.episodes)} episodes scored, mean {rep.mean_score:.1f}, "
            f"{len(flagged)} episode(s) with findings"
        )
        if args.json_out:
            Path(args.json_out).write_text(rep.to_json())
            print(f"report: {args.json_out}")
        if args.min_score is not None and any(e.score < args.min_score for e in rep.episodes):
            return 1
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
