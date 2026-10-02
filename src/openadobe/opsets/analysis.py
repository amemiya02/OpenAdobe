"""Media analysis ops: shots / beats / faces and person matting."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from .. import analysis, ingest, matting
from ..model import Asset, Project
from ..ops import Ctx, Params, op

What = Literal["shots", "beats", "faces"]


class AnalyzeP(Params):
    asset: str
    what: list[What] | None = Field(None, description="subset of shots/beats/faces; omitted = everything that "
                                                       "applies (shots+faces need video, beats needs audio)")


@op("asset.analyze", AnalyzeP)
def asset_analyze(ctx: Ctx, p: Project, a: AnalyzeP):
    """Analyse an asset and store the results in asset.analysis: shots (frame ranges), beats (tempo, beat, downbeat
    and onset times in seconds) and faces (boxes plus a smoothed subject track for reframing). Returns a summary."""
    asset = p.asset(a.asset)
    want = a.what or [w for w in ("shots", "beats", "faces")
                      if (w == "beats" and asset.probe.has_audio) or (w != "beats" and asset.kind == "video")]
    summary: dict = {"asset": asset.id}
    for w in want:
        if w == "shots":
            if asset.kind != "video":
                raise ValueError(f"{asset.id} is not a video; shots need video")
            res = analysis.shots(ctx.runner, ctx.root, p, asset)
            asset.analysis["shots"] = res
            summary["shots"] = len(res)
        elif w == "beats":
            if not asset.probe.has_audio:
                raise ValueError(f"{asset.id} has no audio; beats need audio")
            res = analysis.beats(ctx.runner, ctx.root, p, asset)
            asset.analysis["beats"] = res
            summary.update(tempo=res["tempo"], beats=len(res["beats"]))
        else:
            if asset.kind != "video":
                raise ValueError(f"{asset.id} is not a video; faces need video")
            res = analysis.faces(ctx.runner, ctx.root, p, asset)
            asset.analysis["faces"] = res
            summary["face_coverage_pct"] = round(100 * analysis.face_coverage(res), 1)
    return summary


class MatteP(Params):
    asset: str
    name: str | None = Field(None, description="name of the new asset; default '<name> (matte)'")


@op("asset.matte", MatteP)
def asset_matte(ctx: Ctx, p: Project, a: MatteP):
    """Cut the person out of a video (Robust Video Matting). Creates a NEW video asset with real alpha (lossless FFV1)
    and the same duration; place it on an overlay video track (track.add video overlay) above the background."""
    src = p.asset(a.asset)
    if src.kind != "video":
        raise ValueError(f"{src.id} is not a video")
    new_id = p.new_id("m")
    media_dir = ctx.root / "media"
    media_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", src.name).strip("_")[:40] or "media"
    out = media_dir / f"{new_id}_{slug}_matte.mkv"
    matting.matte(ctx.runner, ctx.root, p, src, out)
    probe = ingest.probe(ctx.runner, out)
    asset = Asset(
        id=new_id, name=a.name or f"{src.name} (matte)", kind="video", source=str(out), sha256=ingest.sha256(out),
        media=str(out.relative_to(ctx.root)), probe=probe, duration_frames=src.duration_frames,
        provenance={"from": src.id, "process": [{"op": "matte", "model": "rvm_mobilenetv3"}]},
    )
    p.assets[new_id] = asset
    return {"asset": new_id, "frames": asset.duration_frames, "has_audio": probe.has_audio}
