"""Comp ops: motion graphics (Lottie) and compositing (Blender) used on the timeline like media."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .. import comps, timeline
from ..model import Comp, Project, Track
from ..ops import Ctx, Params, Time, op, resolve_time


def _overlay_track(p: Project) -> str:
    """The topmost overlay video track, created if there is none."""
    overlays = [t for t in p.tracks if t.kind == "video" and t.role == "overlay"]
    if overlays:
        return overlays[-1].id
    tid = p.new_id("V")
    p.tracks.append(Track(id=tid, kind="video", role="overlay", name=f"Graphics {tid}"))
    return tid


class CompAddP(Params):
    engine: Literal["lottie", "blender"]
    template: str = Field(description="see `openadobe templates`")
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict, description="template slot -> asset id (blender)")
    duration: Time = "4s"
    start: Time | None = Field(None, description="place on the timeline at this time (omit to only create)")
    track: str | None = Field(None, description="video track; default: topmost overlay track (created if needed)")


@op("comp.add", CompAddP)
def comp_add(ctx: Ctx, p: Project, a: CompAddP):
    """Create a composition (title, lower third, keyed shot, 3D title…) and optionally place it on the timeline."""
    comp = Comp(id=p.new_id("k"), engine=a.engine, template=a.template, params=a.params, inputs=a.inputs,
                duration=resolve_time(p, a.duration))
    p.comps[comp.id] = comp
    asset = comps.materialise(p, ctx.root, comp)
    out: dict[str, Any] = {"comp": comp.id, "asset": asset.id, "params": comp.params}
    if a.start is not None:
        track = a.track or _overlay_track(p)
        clips = timeline.add_clip(p, asset.id, track, resolve_time(p, a.start), 0, None, None, insert=False)
        out.update({"clip": clips[0].id, "track": track})
    if comp.engine == "blender":
        out["note"] = "renders on the next `openadobe render` (or comp.render)"
    return out


class CompSetP(Params):
    comp: str
    params: dict[str, Any] = Field(default_factory=dict, description="merged into the current params")
    inputs: dict[str, str] | None = None
    duration: Time | None = None


@op("comp.set", CompSetP)
def comp_set(ctx: Ctx, p: Project, a: CompSetP):
    """Edit a composition. Clips using it update automatically (Dynamic Link); Blender comps re-render lazily."""
    comp = p.comps.get(a.comp)
    if comp is None:
        raise KeyError(f"no comp {a.comp!r}")
    comp.params = {**comp.params, **a.params}
    if a.inputs is not None:
        comp.inputs = a.inputs
    if a.duration is not None:
        comp.duration = resolve_time(p, a.duration)
    asset = comps.materialise(p, ctx.root, comp)
    out: dict[str, Any] = {"comp": comp.id, "asset": asset.id, "params": comp.params,
                           "stale": comp.render_key is None}
    if a.duration is not None:  # clips follow the new length (no ripple): shorter comps shorten their clips
        changed = []
        for t in p.tracks:
            for c in t.clips:
                if c.asset_id == asset.id and c.src_in + c.duration > comp.duration:
                    if c.src_in >= comp.duration:
                        raise ValueError(f"clip {c.id} starts at frame {c.src_in} of {comp.id}, past its new length")
                    c.duration = comp.duration - c.src_in
                    changed.append(c.id)
        if changed:
            out["clips_shortened"] = changed
    return out


class CompRenderP(Params):
    preview: bool = False


@op("comp.render", CompRenderP)
def comp_render(ctx: Ctx, p: Project, a: CompRenderP):
    """Render all out-of-date Blender comps now (delivery does this automatically)."""
    return {"rendered": comps.ensure_rendered(ctx.runner, ctx.root, p, preview=a.preview)}
