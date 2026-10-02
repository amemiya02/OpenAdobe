"""Effect ops: clip effects, track inserts and master bus effects from the catalogue (effects.py)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .. import effects as fx
from ..model import Effect, Project
from ..ops import Ctx, Params, op, resolve_time


def _keyframes(p: Project, params: dict[str, Any]) -> dict[str, Any]:
    """Allow keyframe keys as times ('1.5s', '00:00:01:15') as well as frames."""
    out = {}
    for k, v in params.items():
        out[k] = {resolve_time(p, f): x for f, x in v.items()} if isinstance(v, dict) else v
    return out


def _target_list(p: Project, scope: str, ref: str | None) -> tuple[list[Effect], str]:
    if scope == "clip":
        track, clip = p.find_clip(ref or "")
        return clip.effects, track.kind
    if scope == "track":
        track = p.track(ref or "")
        return track.fx, track.kind
    return p.mix.master_fx, "audio"


class EffectAddP(Params):
    kind: str = Field(description="catalogue name, see `openadobe effects`")
    clip: str | None = Field(None, description="clip id (scope clip)")
    track: str | None = Field(None, description="track id (scope track)")
    scope: Literal["clip", "track", "master"] = "clip"
    params: dict[str, Any] = Field(default_factory=dict, description="values, or keyframes {time: value}")
    linked: bool = Field(False, description="clip scope: also add to linked clips of the matching kind")


@op("effect.add", EffectAddP)
def effect_add(ctx: Ctx, p: Project, a: EffectAddP):
    """Add a catalogued effect to a clip, a track (insert) or the master bus. Values are validated."""
    ref = a.clip if a.scope == "clip" else a.track
    lst, kind = _target_list(p, a.scope, ref)
    params = fx.validate(a.kind, _keyframes(p, a.params))
    if fx.target(a.kind) != kind:
        raise ValueError(f"{a.kind} is a {fx.target(a.kind)} effect; {a.scope} {ref or ''} is {kind}")
    eid = p.new_id("e")
    lst.append(Effect(id=eid, kind=a.kind, params=params))
    added = [eid]
    if a.scope == "clip" and a.linked:
        _, clip = p.find_clip(a.clip or "")
        for t, c in p.linked(clip):
            if c.id != clip.id and t.kind == fx.target(a.kind):
                e2 = p.new_id("e")
                c.effects.append(Effect(id=e2, kind=a.kind, params=params))
                added.append(e2)
    return {"effects": added, "params": params}


def _find_effect(p: Project, eid: str) -> tuple[list[Effect], Effect]:
    pools = [c.effects for t in p.tracks for c in t.clips] + [t.fx for t in p.tracks] + [p.mix.master_fx]
    for pool in pools:
        for e in pool:
            if e.id == eid:
                return pool, e
    raise KeyError(f"no effect {eid!r}")


class EffectSetP(Params):
    effect: str
    params: dict[str, Any] = Field(default_factory=dict, description="params to change (merged)")
    enabled: bool | None = None


@op("effect.set", EffectSetP)
def effect_set(ctx: Ctx, p: Project, a: EffectSetP):
    """Change an effect's params (merged with the current ones) or bypass it."""
    _, e = _find_effect(p, a.effect)
    e.params = fx.validate(e.kind, {**e.params, **_keyframes(p, a.params)})
    if a.enabled is not None:
        e.enabled = a.enabled
    return e.model_dump()


class EffectRemoveP(Params):
    effect: str


@op("effect.remove", EffectRemoveP)
def effect_remove(ctx: Ctx, p: Project, a: EffectRemoveP):
    """Remove an effect."""
    pool, e = _find_effect(p, a.effect)
    pool.remove(e)
    return {"removed": e.id}
