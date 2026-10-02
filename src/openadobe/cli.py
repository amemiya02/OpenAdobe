"""Agent-facing CLI. Every command prints one JSON document; failures exit 1 with {"ok": false}.

    openadobe init DIR [--fps 30] [--size 1920x1080]
    openadobe ops [NAME]                          list operations with JSON Schemas
    openadobe op DIR NAME key=value ...           apply one operation
    openadobe batch DIR steps.json                apply several operations atomically
    openadobe inspect DIR | transcript DIR ASSET | log DIR | undo DIR
    openadobe render DIR [--preview] [--deliverable main]
    openadobe perceive DIR frame|sheet|wave|spectrum|loudness|detect [--at T] [--media PATH]
    openadobe qc DIR [--media PATH]
    openadobe export-otio DIR OUT.otio
    openadobe doctor [--build]
    openadobe mcp                                 serve the same tools over MCP (stdio)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, deliver, ops, perceive, qc, store
from .engines.runner import EngineError, Runner
from .timebase import Rate, parse_time


def _emit(payload: dict, code: int = 0) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return code


def _kv(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="openadobe", description="Agent-operated post-production on real engines.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a project")
    p.add_argument("dir")
    p.add_argument("--name")
    p.add_argument("--fps", default="30")
    p.add_argument("--size", default="1920x1080")

    p = sub.add_parser("ops", help="list operations and their parameter schemas")
    p.add_argument("name", nargs="?")

    p = sub.add_parser("effects", help="effect catalogue (clip/track/master effects)")
    p.add_argument("kind", nargs="?")
    p = sub.add_parser("templates", help="comp templates (lottie motion graphics, blender compositing)")
    p.add_argument("engine", nargs="?", choices=["lottie", "blender"])
    sub.add_parser("presets", help="caption, deliverable and audio-repair presets; transition kinds")
    sub.add_parser("audio-steps", help="audio repair steps and presets for asset.process")

    p = sub.add_parser("op", help="apply one operation")
    p.add_argument("dir")
    p.add_argument("name")
    p.add_argument("params", nargs="*", help="key=value (values parsed as JSON when possible)")
    p.add_argument("--json", help="params as a JSON object")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("batch", help="apply a JSON list of {op, params} atomically")
    p.add_argument("dir")
    p.add_argument("file", help="path or - for stdin")
    p.add_argument("--dry-run", action="store_true")

    for name in ("inspect", "log", "undo"):
        p = sub.add_parser(name)
        p.add_argument("dir")
        if name == "log":
            p.add_argument("-n", type=int, default=20)

    p = sub.add_parser("transcript", help="indexed transcript words of an asset")
    p.add_argument("dir")
    p.add_argument("asset")

    p = sub.add_parser("render", help="render a deliverable (runs QC unless --preview)")
    p.add_argument("dir")
    p.add_argument("--deliverable", default="main")
    p.add_argument("--all", action="store_true", help="render every deliverable")
    p.add_argument("--preview", action="store_true", help="fast 640px render, no loudness pass")
    p.add_argument("--no-qc", action="store_true")

    p = sub.add_parser("perceive", help="look at / listen to the work")
    p.add_argument("dir")
    p.add_argument("kind", choices=["frame", "sheet", "wave", "spectrum", "loudness", "detect"])
    p.add_argument("--at", help="timeline time for frame (frames, '2.5s' or timecode)")
    p.add_argument("--media", help="file to analyse (default: latest render)")
    p.add_argument("--n", type=int, default=12, help="frames in a contact sheet")

    p = sub.add_parser("review", help="build a review package (contact sheet, waveform, QC, captions, rubric)")
    p.add_argument("dir")
    p.add_argument("--media")
    p.add_argument("--vlm", action="store_true", help="also ask a Claude vision model (needs ANTHROPIC_API_KEY)")

    p = sub.add_parser("qc", help="quality-check a render")
    p.add_argument("dir")
    p.add_argument("--media")
    p.add_argument("--deliverable", default="main")

    p = sub.add_parser("export-otio", help="export the edit as OpenTimelineIO")
    p.add_argument("dir")
    p.add_argument("out")

    p = sub.add_parser("doctor", help="check engines")
    p.add_argument("--build", action="store_true", help="build the engine image if missing")
    p.add_argument("--verify", action="store_true", help="render the golden reference project and compare")
    p.add_argument("--record", action="store_true", help="with --verify: record a new golden reference")
    p.add_argument("--workdir", help="where to build the golden project (default: a temp dir)")

    sub.add_parser("mcp", help="serve over MCP stdio")
    return ap


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    cmd = args.cmd

    if cmd == "init":
        w, h = (int(x) for x in args.size.lower().split("x"))
        root = Path(args.dir).resolve()
        project = store.init(root, args.name or root.name, Rate.parse(args.fps), w, h)
        return _emit({"ok": True, "project": str(root), "fps": str(project.rate), "size": args.size,
                      "tracks": [t.id for t in project.tracks]})
    if cmd == "ops":
        return _emit({"ok": True, "ops": ops.describe(args.name)})
    if cmd == "op":
        params = json.loads(args.json) if args.json else {}
        params.update(_kv(args.params))
        return _emit(ops.apply(Path(args.dir), [(args.name, params)], runner=runner, dry_run=args.dry_run))
    if cmd == "batch":
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text("utf-8")
        steps = [(s["op"], s.get("params", {})) for s in json.loads(raw)]
        return _emit(ops.apply(Path(args.dir), steps, runner=runner, dry_run=args.dry_run))
    if cmd == "inspect":
        return _emit({"ok": True, **perceive.inspect(store.load(Path(args.dir)), Path(args.dir))})
    if cmd == "transcript":
        return _emit({"ok": True, **perceive.transcript(store.load(Path(args.dir)), args.asset)})
    if cmd == "log":
        return _emit({"ok": True, "log": store.read_log(Path(args.dir), args.n)})
    if cmd == "undo":
        project = store.undo(Path(args.dir))
        return _emit({"ok": True, "revision": project.revision})
    if cmd == "render":
        if args.all:
            return _emit({"ok": True, **deliver.render_all(Path(args.dir), runner)})
        res = deliver.render(Path(args.dir), args.deliverable, args.preview, runner, run_qc=not args.no_qc)
        return _emit({"ok": True, **res})
    if cmd == "effects":
        from . import effects
        return _emit({"ok": True, "effects": effects.describe(args.kind)})
    if cmd == "templates":
        from . import comps
        return _emit({"ok": True, "templates": comps.describe(args.engine)})
    if cmd == "audio-steps":
        from . import audio_repair
        return _emit({"ok": True, **audio_repair.describe()})
    if cmd == "presets":
        from . import captions, timeline
        out = {"ok": True, "captions": captions.PRESETS, "deliverables": deliver.PRESETS,
               "transitions": {"video": timeline.VIDEO_KINDS, "audio": timeline.AUDIO_KINDS}}
        try:
            from . import audio_repair
            out["audio_repair"] = audio_repair.describe()
        except ImportError:
            pass
        return _emit(out)
    if cmd == "perceive":
        root = Path(args.dir).resolve()
        if args.kind == "frame":
            if args.at is None:
                raise ValueError("frame needs --at")
            project = store.load(root)
            res = perceive.frame(runner, root, parse_time(args.at, project.rate))
        elif args.kind == "sheet":
            res = perceive.contact_sheet(runner, root, args.media, n=args.n)
        else:
            fn = {"wave": perceive.waveform, "spectrum": perceive.spectrogram, "loudness": perceive.loudness,
                  "detect": perceive.detect}[args.kind]
            res = fn(runner, root, args.media)
        return _emit({"ok": True, **res})
    if cmd == "review":
        from . import review
        package = review.build(runner, Path(args.dir), args.media)
        if args.vlm:
            package["vlm"] = review.vlm_review(package)
        package.pop("rubric", None)
        return _emit({"ok": True, **package})
    if cmd == "qc":
        root = Path(args.dir).resolve()
        project = store.load(root)
        media = perceive._media(root, args.media)
        return _emit({"ok": True, **qc.run(runner, root, project, project.deliverables[args.deliverable], media)})
    if cmd == "export-otio":
        from .interchange import export_otio
        root = Path(args.dir).resolve()
        out = export_otio(store.load(root), root, Path(args.out))
        return _emit({"ok": True, "otio": str(out)})
    if cmd == "doctor":
        present = runner.image_present()
        if not present and args.build:
            runner.build_image()
            present = runner.image_present()
        report = {"ok": present, "mode": runner.mode, "image": runner.image, "image_present": present}
        if present:
            report["engines"] = runner.versions()
        try:
            import faster_whisper  # noqa: F401
            report["asr"] = True
        except ImportError:
            report["asr"] = False
        if present and args.verify:
            import tempfile

            from . import golden
            work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="openadobe-golden-"))
            report["golden"] = golden.run(runner, work, record=args.record)
            report["ok"] = report["ok"] and report["golden"]["ok"]
        return _emit(report, 0 if report["ok"] else 1)
    if cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0
    raise AssertionError(cmd)


def main(argv: list[str] | None = None) -> None:
    try:
        code = run(argv)
    except ops.OpError as exc:
        code = _emit({"ok": False, "error": str(exc), "issues": exc.issues}, 1)
    except store.ProjectError as exc:
        code = _emit({"ok": False, "error": str(exc), "issues": exc.issues}, 1)
    except EngineError as exc:
        code = _emit({"ok": False, "error": str(exc)}, 1)
    except (KeyError, ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        code = _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 1)
    except Exception as exc:  # noqa: BLE001 - keep the one-JSON-document contract even for bugs
        import traceback
        code = _emit({"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}",
                      "trace": traceback.format_exc(limit=6).splitlines()[-8:]}, 2)
    sys.exit(code)


if __name__ == "__main__":
    main()
