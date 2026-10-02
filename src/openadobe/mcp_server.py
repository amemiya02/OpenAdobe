"""MCP server (optional `mcp` extra) exposing the same API as the CLI.

Edits go through two generic tools (``list_ops`` / ``apply_ops``) instead of one tool per
verb, which keeps the tool list small in the agent's context. Perception tools return real
images so a multimodal model can look at frames and contact sheets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import deliver, ops, perceive, qc, store
from .engines.runner import Runner
from .timebase import Rate, parse_time


def build():
    try:
        from mcp.server.mcpserver import Image, MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer
        from mcp.server.fastmcp import Image

    mcp = MCPServer("openadobe")
    runner = Runner()

    @mcp.tool()
    def init_project(dir: str, name: str | None = None, fps: str = "30", size: str = "1920x1080") -> dict:
        """Create a project directory with V1 (video), A1 (dialogue) and A2 (music) tracks."""
        w, h = (int(x) for x in size.lower().split("x"))
        root = Path(dir).resolve()
        p = store.init(root, name or root.name, Rate.parse(fps), w, h)
        return {"project": str(root), "tracks": [t.id for t in p.tracks]}

    @mcp.tool()
    def list_ops(name: str | None = None) -> list[dict]:
        """List edit operations with their JSON Schemas. Call with a name for one op."""
        return ops.describe(name)

    @mcp.tool()
    def apply_ops(dir: str, steps: list[dict[str, Any]], dry_run: bool = False) -> dict:
        """Apply [{"op": name, "params": {...}}, ...] atomically as one revision. Times accept frames,
        '2.5s' or timecode. Fails without changing anything if the result would be invalid."""
        try:
            return ops.apply(Path(dir), [(s["op"], s.get("params", {})) for s in steps], runner=runner,
                             dry_run=dry_run)
        except ops.OpError as exc:
            return {"ok": False, "error": str(exc), "issues": exc.issues}

    @mcp.tool()
    def inspect(dir: str) -> dict:
        """Timeline, assets, captions, mix settings and lint issues of a project."""
        return perceive.inspect(store.load(Path(dir)), Path(dir))

    @mcp.tool()
    def transcript(dir: str, asset: str) -> dict:
        """Indexed transcript words of an asset (indexes feed edit.cut_words)."""
        return perceive.transcript(store.load(Path(dir)), asset)

    @mcp.tool()
    def undo(dir: str) -> dict:
        """Revert the latest operation."""
        return {"revision": store.undo(Path(dir)).revision}

    @mcp.tool()
    def render(dir: str, deliverable: str = "main", preview: bool = False) -> dict:
        """Render a deliverable. Full renders match loudness and run QC; previews are fast 640px."""
        return deliver.render(Path(dir), deliverable, preview, runner)

    @mcp.tool()
    def see_frame(dir: str, at: str):
        """Render one timeline frame (with captions) from the current edit and return the image."""
        root = Path(dir).resolve()
        res = perceive.frame(runner, root, parse_time(at, store.load(root).rate))
        return Image(path=res["image"])

    @mcp.tool()
    def see_contact_sheet(dir: str, media: str | None = None, n: int = 12):
        """Time-stamped grid of frames from a render (default: latest) — review the whole cut at a glance."""
        return Image(path=perceive.contact_sheet(runner, Path(dir), media, n=n)["image"])

    @mcp.tool()
    def see_audio(dir: str, kind: str = "wave", media: str | None = None):
        """Waveform ('wave') or spectrogram ('spectrum') image of a render or asset."""
        fn = perceive.spectrogram if kind == "spectrum" else perceive.waveform
        return Image(path=fn(runner, Path(dir), media)["image"])

    @mcp.tool()
    def measure(dir: str, kind: str = "loudness", media: str | None = None) -> dict:
        """'loudness' (LUFS / LRA / true peak) or 'detect' (black, freeze, silence spans)."""
        fn = perceive.detect if kind == "detect" else perceive.loudness
        return fn(runner, Path(dir), media)

    @mcp.tool()
    def catalog(kind: str = "effects") -> Any:
        """Reference data: 'effects' (clip/track/master effects + params), 'templates' (lottie/blender comps),
        'presets' (caption styles, deliverables, transitions), 'audio_steps' (asset.process repair chain)."""
        if kind == "effects":
            from . import effects
            return effects.describe()
        if kind == "templates":
            from . import comps
            return comps.describe()
        if kind == "audio_steps":
            from . import audio_repair
            return audio_repair.describe()
        from . import captions, timeline
        return {"captions": captions.PRESETS, "deliverables": deliver.PRESETS,
                "transitions": {"video": timeline.VIDEO_KINDS, "audio": timeline.AUDIO_KINDS}}

    @mcp.tool()
    def review(dir: str, media: str | None = None):
        """Review package for the latest (or given) render: time-stamped contact sheet image + QC, loudness,
        captions and a senior-editor rubric. Look at the image and judge the cut against the rubric."""
        from . import review as rv
        pkg = rv.build(runner, Path(dir), media)
        facts = {k: pkg[k] for k in ("qc", "loudness", "captions", "timeline")}
        return [Image(path=pkg["contact_sheet"]), json.dumps(facts, ensure_ascii=False), pkg["rubric"]]

    @mcp.tool()
    def quality_check(dir: str, media: str | None = None, deliverable: str = "main") -> dict:
        """Run QC on a render against the deliverable spec."""
        root = Path(dir).resolve()
        project = store.load(root)
        return qc.run(runner, root, project, project.deliverables[deliverable], perceive._media(root, media))

    return mcp


def main() -> None:
    build().run()
