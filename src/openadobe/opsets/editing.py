"""Editing ops beyond the P0 core: speed changes and transitions."""

from __future__ import annotations

from pydantic import Field

from .. import timeline
from ..model import Project
from ..ops import Ctx, Params, Time, op, resolve_time


class SpeedP(Params):
    clip: str
    speed: float = Field(description="0.05..20; 0.5 = half speed slow motion, 2 = double speed")
    ripple: bool = Field(True, description="move later clips on the same tracks to follow the new length")


@op("clip.speed", SpeedP)
def clip_speed(ctx: Ctx, p: Project, a: SpeedP):
    """Change playback speed (picture and its linked sound, pitch preserved). Uses the same source range."""
    return timeline.set_speed(p, a.clip, a.speed, a.ripple)


class TransitionAddP(Params):
    clip: str = Field(description="the outgoing clip; the transition sits on the cut at its end")
    duration: Time = "0.5s"
    kind: str = Field("dissolve", description="video: " + ", ".join(timeline.VIDEO_KINDS)
                      + "; audio: " + ", ".join(timeline.AUDIO_KINDS))
    audio: bool = Field(True, description="also crossfade (or dip) the linked sound")


@op("transition.add", TransitionAddP)
def transition_add(ctx: Ctx, p: Project, a: TransitionAddP):
    """Add a transition centred on the cut after a clip. Dissolves/wipes/pushes need unused media (handles)
    on both sides of the cut; dips (dip_black/dip_white) don't."""
    track, _ = p.find_clip(a.clip)
    return {"transitions": timeline.add_transition(p, track.id, a.clip, resolve_time(p, a.duration), a.kind,
                                                   a.audio)}


class TransitionSetP(Params):
    transition: str
    duration: Time | None = None
    kind: str | None = None
    softness: float | None = Field(None, description="wipe edge softness 0..1")


def _find_transition(p: Project, tid: str):
    for t in p.tracks:
        for tr in t.transitions:
            if tr.id == tid:
                return t, tr
    raise KeyError(f"no transition {tid!r}")


@op("transition.set", TransitionSetP)
def transition_set(ctx: Ctx, p: Project, a: TransitionSetP):
    """Change a transition's duration, kind or wipe softness."""
    track, tr = _find_transition(p, a.transition)
    if a.kind is not None:
        kinds = timeline.VIDEO_KINDS if track.kind == "video" else timeline.AUDIO_KINDS
        if a.kind not in kinds:
            raise ValueError(f"{track.kind} transitions: {kinds}")
        tr.kind = a.kind
    if a.duration is not None:
        tr.duration = resolve_time(p, a.duration)
    if a.softness is not None:
        tr.params["softness"] = a.softness
    return tr.model_dump()


class TransitionRemoveP(Params):
    transition: str


@op("transition.remove", TransitionRemoveP)
def transition_remove(ctx: Ctx, p: Project, a: TransitionRemoveP):
    """Remove a transition (back to a straight cut)."""
    track, tr = _find_transition(p, a.transition)
    track.transitions.remove(tr)
    return {"removed": tr.id}


class MontageP(Params):
    clips: list[str] = Field(min_length=1, description="asset ids to cut together (b-roll pool, cycling)")
    music: str = Field(description="music asset with beat analysis (asset.analyze what=['beats'])")
    track: str = "V1"
    start: Time = 0
    length: Time | None = Field(None, description="montage length; default: the music length")
    every: int = Field(2, ge=1, le=16, description="cut every N beats")
    on: str = Field("beats", description="'beats' or 'downbeats'")
    place_music: bool = Field(True, description="also put the music on the first music track")
    pick: str = Field("shots", description="'shots' = use a different detected shot per cut; 'sequential'")


@op("edit.montage", MontageP)
def edit_montage(ctx: Ctx, p: Project, a: MontageP):
    """Beat-synced montage: cut a pool of clips on the music's beats (every N beats), b-roll muted."""
    music = p.asset(a.music)
    beats_info = music.analysis.get("beats")
    if not beats_info or not beats_info.get(a.on):
        raise ValueError(f"{a.music} has no {a.on}; run asset.analyze asset={a.music} what=['beats'] first")
    start = resolve_time(p, a.start)
    length = resolve_time(p, a.length) if a.length is not None else music.duration_frames
    warning = None
    if length > music.duration_frames:
        warning = f"length clamped to the music ({p.rate.clock(music.duration_frames)}); loop or add music to go longer"
        length = music.duration_frames
    fps = float(p.rate.fps)
    beats = [round(t * fps) for t in beats_info[a.on] if 0 < round(t * fps) < length]
    marks = beats[a.every - 1 :: a.every]  # a cut on every Nth beat after the start
    cuts = [0, *marks, length]
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] >= 3]  # no sub-3-frame flashes
    pools = [p.asset(x) for x in a.clips]
    if any(x.kind not in ("video", "image", "comp") for x in pools):
        raise ValueError("montage clips must be video, image or comp assets")
    shot_cursor = {x.id: 0 for x in pools}
    src_cursor = {x.id: 0 for x in pools}
    placed = []
    for i in range(len(cuts) - 1):
        dur = cuts[i + 1] - cuts[i]
        asset = pools[i % len(pools)]
        if asset.kind == "image":
            src_in = 0
        else:
            shots = [s for s in (asset.analysis.get("shots") or []) if s["end"] - s["start"] >= dur]
            if a.pick == "shots" and shots:
                shot = shots[shot_cursor[asset.id] % len(shots)]
                shot_cursor[asset.id] += 1
                slack = shot["end"] - shot["start"] - dur
                src_in = shot["start"] + slack // 3  # skip the very start of a shot
            else:
                if src_cursor[asset.id] + dur > asset.duration_frames:
                    src_cursor[asset.id] = 0
                src_in = src_cursor[asset.id]
                src_cursor[asset.id] += dur
            if src_in + dur > asset.duration_frames:
                raise ValueError(f"{asset.id} is shorter than a {dur}-frame cut; lower 'every' or add clips")
        clips = timeline.add_clip(p, asset.id, a.track, start + cuts[i], src_in, dur, None, insert=False)
        placed.append(clips[0].id)
    out = {"clips": placed, "cuts": len(placed), "tempo": beats_info.get("tempo"), "length": length}
    if warning:
        out["warning"] = warning
    if a.place_music:
        mus = p.tracks_by_role("mus")
        if not mus:
            raise ValueError("no music track; add one with track.add kind=audio role=mus")
        clip = timeline.add_clip(p, music.id, mus[0].id, start, 0, min(length, music.duration_frames), None,
                                 insert=False)[0]
        out["music_clip"] = clip.id
    return out
