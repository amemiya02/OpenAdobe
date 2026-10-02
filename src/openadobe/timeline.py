"""Pure editing algorithms on the project model (no I/O).

Linked clips (picture + its sound) share ``Clip.link`` and are edited as one unit.
Ripple edits shift later clips only on the tracks the edited clips live on, so a music bed on
another track keeps its position (Premiere's behaviour with sync-lock off for that track).
Manual captions follow ripple edits through a frame-remapping function; transcript-derived
captions need no remapping because they are rebuilt from the cut at compile time.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .effects import value_at
from .model import Asset, Clip, Effect, Project, Track

IMAGE_PAD = 100_000  # stills: producer frames are offset so transition handles never go negative

FrameMap = Callable[[int], int | None]  # old timeline frame -> new frame, None if the frame was removed


class EditError(ValueError):
    pass


def producer_in(clip: Clip, asset: Asset | None = None) -> int:
    """First producer frame of the clip (timewarp producers count output frames; stills are padded)."""
    if asset is not None and asset.kind == "image":
        return clip.src_in + IMAGE_PAD
    return clip.src_in if clip.speed == 1.0 else round(clip.src_in / clip.speed)


def producer_length(clip: Clip, asset: Asset) -> int:
    return asset.duration_frames if clip.speed == 1.0 else int(asset.duration_frames / clip.speed)


def src_for_producer(pin: int, speed: float) -> int:
    """A source frame whose timewarp position is exactly ``pin`` (keeps split/trim pieces contiguous)."""
    if speed == 1.0:
        return pin
    guess = round(pin * speed)
    for d in (0, -1, 1, -2, 2):
        if round((guess + d) / speed) == pin:
            return guess + d
    return guess


def fork_effects(project: Project, effects: list[Effect], offset: int, length: int) -> list[Effect]:
    """Independent copies (new ids) of a clip's effects for a piece starting ``offset`` frames in."""
    out = []
    for e in effects:
        params = {}
        for k, v in e.params.items():
            if isinstance(v, dict) and v:
                pts = {int(f) - offset: x for f, x in v.items() if 0 <= int(f) - offset < length}
                pts[0] = value_at(v, offset)
                pts.setdefault(max(length - 1, 0), value_at(v, offset + length - 1))
                params[k] = dict(sorted(pts.items()))
            else:
                params[k] = v
        out.append(Effect(id=project.new_id("e"), kind=e.kind, params=params, enabled=e.enabled))
    return out


def track_end(track: Track) -> int:
    return max((c.end for c in track.clips), default=0)


def shift_after(project: Project, tracks: set[str], frame: int, delta: int, exclude: set[str] = frozenset()) -> None:
    """Move every clip on ``tracks`` starting at or after ``frame`` by ``delta`` frames."""
    for t in project.tracks:
        if t.id in tracks:
            for c in t.clips:
                if c.id not in exclude and c.start >= frame:
                    c.start += delta


def remap_captions(project: Project, fn: FrameMap) -> None:
    """Re-time manual captions; a caption keeps the span of its surviving frames or is dropped."""
    kept = []
    for cap in project.captions.items:
        frames = [m for f in range(cap.start, cap.end) if (m := fn(f)) is not None]
        if frames:
            cap.start, cap.end = min(frames), max(frames) + 1
            kept.append(cap)
    project.captions.items = kept


def add_clip(
    project: Project,
    asset_id: str,
    track_id: str,
    start: int | None,
    src_in: int,
    duration: int | None,
    link_track: str | None,
    insert: bool,
) -> list[Clip]:
    asset = project.asset(asset_id)
    track = project.track(track_id)
    if track.kind == "video" and asset.kind == "audio":
        raise EditError(f"{asset_id} is audio-only; put it on an audio track")
    if track.kind == "audio" and not asset.probe.has_audio:
        raise EditError(f"{asset_id} has no audio stream")
    if src_in < 0:
        raise EditError("src_in must be >= 0")
    if duration is None:
        if asset.kind == "image":
            raise EditError("still images need an explicit duration")
        duration = asset.duration_frames - src_in
    if duration <= 0:
        raise EditError("duration must be positive")
    targets = [track]
    if link_track is not None:
        lt = project.track(link_track)
        if lt.kind != "audio" or track.kind != "video":
            raise EditError("link_track must be an audio track when placing on a video track")
        targets.append(lt)
    if start is None:
        start = max(track_end(t) for t in targets)
    if insert:
        shift_after(project, {t.id for t in targets}, start, duration)
        remap_captions(project, lambda f: f if f < start else f + duration)
    link = project.new_id("L") if len(targets) > 1 else None
    made = []
    for t in targets:
        clip = Clip(id=project.new_id("c"), asset_id=asset_id, start=start, src_in=src_in, duration=duration,
                    link=link)
        t.clips.append(clip)
        made.append(clip)
    return made


def move(project: Project, clip_id: str, start: int, track_id: str | None) -> None:
    track, clip = project.find_clip(clip_id)
    delta = start - clip.start
    for _, c in project.linked(clip):
        c.start += delta
    if track_id and track_id != track.id:
        dest = project.track(track_id)
        if dest.kind != track.kind:
            raise EditError(f"cannot move a {track.kind} clip to {dest.kind} track {dest.id}")
        track.clips.remove(clip)
        dest.clips.append(clip)


def trim(project: Project, clip_id: str, head: int, tail: int, ripple: bool) -> None:
    """head/tail > 0 remove frames from that edge, < 0 extend it.

    With ripple the clip keeps its start and later clips on its tracks close up (or open up).
    Without ripple the trimmed edge simply leaves a gap.
    """
    _, clip = project.find_clip(clip_id)
    group = project.linked(clip)
    start, old_end = clip.start, clip.end
    for _, c in group:
        new_src = c.src_in + head if c.speed == 1.0 else src_for_producer(producer_in(c) + head, c.speed)
        if new_src < 0:
            raise EditError(f"cannot extend head of {c.id} before source start")
        c.src_in = new_src
        c.duration -= head + tail
        if c.duration <= 0:
            raise EditError(f"trim leaves {c.id} with no frames")
        asset = project.asset(c.asset_id)
        if asset.kind != "image" and c.src_out > asset.duration_frames:
            raise EditError(f"cannot extend {c.id} past the end of its media")
        c.fade_in = min(c.fade_in, c.duration)
        c.fade_out = min(c.fade_out, c.duration - c.fade_in)
        if not ripple:
            c.start += head
    if ripple:
        shift_after(project, {t.id for t, _ in group}, old_end, -(head + tail), exclude={c.id for _, c in group})

        def fn(f: int) -> int | None:
            if f < start:
                return f
            if f >= old_end:
                return f - head - tail
            if f - start < head or f >= old_end - tail:
                return None
            return f - head

        remap_captions(project, fn)


def split(project: Project, clip_id: str, at: int) -> list[str]:
    """Split a clip (and its linked clips) at timeline frame ``at``; returns the new right-half ids."""
    _, clip = project.find_clip(clip_id)
    if not clip.start < at < clip.end:
        raise EditError(f"{clip_id} spans {clip.start}..{clip.end}; split point {at} is outside it")
    group = project.linked(clip)
    new_link = project.new_id("L") if clip.link else None
    created = []
    for t, c in group:
        left = at - c.start
        right = c.model_copy(deep=True, update={
            "id": project.new_id("c"), "start": at,
            "src_in": src_for_producer(producer_in(c) + left, c.speed) if c.speed != 1.0 else c.src_in + left,
            "duration": c.duration - left, "fade_in": 0, "fade_out": min(c.fade_out, c.duration - left),
            "link": new_link,
        })
        right.effects = fork_effects(project, c.effects, left, right.duration)
        for tr in t.transitions:
            if tr.a == c.id:
                tr.a = right.id
        c.duration = left
        c.fade_out = 0
        c.fade_in = min(c.fade_in, left)
        t.clips.append(right)
        created.append(right.id)
    return created


def delete(project: Project, clip_id: str, ripple: bool) -> list[str]:
    _, clip = project.find_clip(clip_id)
    group = project.linked(clip)
    start, end = clip.start, clip.end
    for t, c in group:
        t.clips.remove(c)
    if ripple:
        shift_after(project, {t.id for t, _ in group}, end, start - end)
        remap_captions(project, lambda f: f if f < start else (None if f < end else f - (end - start)))
    return [c.id for _, c in group]


def remove_source_ranges(
    project: Project, clip_id: str, ranges: list[tuple[int, int]], ripple: bool, declick: int = 1
) -> dict:
    """Cut source-frame ranges [a, b) out of a clip and its linked clips (silences, flubbed words).

    Audio pieces get ``declick`` frame micro-fades at the new internal cut points.
    Returns the surviving clip ids and the number of frames removed.
    """
    _, clip = project.find_clip(clip_id)
    if clip.speed != 1.0:
        raise EditError(f"{clip_id} plays at speed {clip.speed}; cut source ranges before changing speed")
    lo, hi = clip.src_in, clip.src_out
    cuts = sorted((max(a, lo), min(b, hi)) for a, b in ranges if min(b, hi) > max(a, lo))
    merged: list[list[int]] = []
    for a, b in cuts:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    keep: list[tuple[int, int]] = []
    cursor = lo
    for a, b in merged:
        if a > cursor:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < hi:
        keep.append((cursor, hi))
    if not keep:
        raise EditError("the ranges remove the whole clip; use clip.delete instead")
    removed = (hi - lo) - sum(b - a for a, b in keep)
    group = project.linked(clip)
    if removed == 0:
        return {"clips": [c.id for _, c in group], "removed_frames": 0}

    start, old_end = clip.start, clip.end
    survivors: list[str] = []
    links = [project.new_id("L") if clip.link else None for _ in keep]
    for t, c in group:
        t.clips.remove(c)
        pos = c.start
        for i, (a, b) in enumerate(keep):
            internal = declick if t.kind == "audio" else 0
            length = b - a
            fade_in = min(c.fade_in if a == lo else internal, length)  # original fades stay on original edges
            fade_out = min(c.fade_out if b == hi else internal, length - fade_in)
            piece = c.model_copy(deep=True, update={
                "id": project.new_id("c"), "start": pos if ripple else c.start + (a - lo), "src_in": a,
                "duration": length, "link": links[i], "fade_in": fade_in, "fade_out": fade_out,
            })
            piece.effects = fork_effects(project, c.effects, a - lo, length)
            t.clips.append(piece)
            survivors.append(piece.id)
            pos += b - a
    if ripple:
        shift_after(project, {t.id for t, _ in group}, old_end, -removed, exclude=set(survivors))
        offsets = []  # (src_a, src_b, new timeline start)
        pos = start
        for a, b in keep:
            offsets.append((a, b, pos))
            pos += b - a

        def fn(f: int) -> int | None:
            if f < start:
                return f
            if f >= old_end:
                return f - removed
            src = lo + (f - start)
            for a, b, new in offsets:
                if a <= src < b:
                    return new + (src - a)
            return None

        remap_captions(project, fn)
    return {"clips": survivors, "removed_frames": removed}


def set_speed(project: Project, clip_id: str, speed: float, ripple: bool) -> dict:
    """Change playback speed keeping the same source material; duration scales (linked clips follow)."""
    if not 0.05 <= speed <= 20:
        raise EditError("speed must be between 0.05 and 20")
    _, clip = project.find_clip(clip_id)
    group = project.linked(clip)
    for _, c in group:
        a = project.assets.get(c.asset_id)
        if a is not None and a.kind == "comp" and project.comps[a.comp_id].engine == "lottie":
            raise EditError("speed cannot be changed on a Lottie graphic; change the comp's duration instead")
    old_end = clip.end
    src_len = clip.src_len
    new_dur = max(1, math.floor(src_len / speed))
    for _, c in group:
        c.speed = speed
        c.duration = new_dur
        c.fade_in = min(c.fade_in, new_dur)
        c.fade_out = min(c.fade_out, new_dur - c.fade_in)
    delta = new_dur - (old_end - clip.start)
    if ripple and delta:
        shift_after(project, {t.id for t, _ in group}, old_end, delta, exclude={c.id for _, c in group})
        remap_captions(project, lambda f: f if f < old_end else f + delta)
    return {"clip": clip_id, "speed": speed, "duration": new_dur}


# ---- transitions -----------------------------------------------------------------------

DIP_KINDS = ("dip_black", "dip_white", "dip")
VIDEO_KINDS = ("dissolve", "wipe_left", "wipe_right", "wipe_up", "wipe_down", "wipe_circle", "wipe_diagonal",
               "push_left", "push_right", "push_up", "push_down", "slide_left", "slide_right", "slide_up",
               "slide_down", "dip_black", "dip_white")
AUDIO_KINDS = ("crossfade", "dip")


def halves(duration: int) -> tuple[int, int]:
    """Frames of a centred transition before and after the cut."""
    return duration // 2, duration - duration // 2


def needs_handles(kind: str) -> bool:
    return kind not in DIP_KINDS


def add_transition(project: Project, track_id: str, a_id: str, duration: int, kind: str, audio: bool) -> list[str]:
    """Add a transition on the cut after clip ``a`` (to the clip that starts where it ends)."""
    track = project.track(track_id)
    kinds = VIDEO_KINDS if track.kind == "video" else AUDIO_KINDS
    if kind not in kinds:
        raise EditError(f"{track.kind} transitions: {kinds}")
    a = next((c for c in track.clips if c.id == a_id), None)
    if a is None:
        raise EditError(f"{a_id} is not on {track_id}")
    b = next((c for c in track.clips if c.start == a.end), None)
    if b is None:
        raise EditError(f"no clip starts where {a_id} ends ({a.end}); transitions need adjacent clips")
    if any(t.a == a.id or t.b == b.id for t in track.transitions):
        raise EditError(f"the cut {a.id}|{b.id} already has a transition")
    from .model import Transition

    made = []
    t = Transition(id=project.new_id("T"), a=a.id, b=b.id, duration=duration, kind=kind)
    track.transitions.append(t)
    made.append(t.id)
    if audio and track.kind == "video" and a.link and b.link:
        partners = [(lt, la, lb) for lt in project.tracks if lt.kind == "audio"
                    for la in lt.clips if la.link == a.link for lb in lt.clips if lb.link == b.link]
        for lt, la, lb in partners:
            if la.end == lb.start and not any(x.a == la.id for x in lt.transitions):
                at = Transition(id=project.new_id("T"), a=la.id, b=lb.id, duration=duration,
                                kind="dip" if kind in DIP_KINDS else "crossfade")
                lt.transitions.append(at)
                made.append(at.id)
    return made


def prune_transitions(project: Project) -> list[str]:
    """Drop transitions whose clips were removed or are no longer adjacent; returns their ids."""
    dropped = []
    for t in project.tracks:
        ids = {c.id: c for c in t.clips}
        keep = []
        for tr in t.transitions:
            a, b = ids.get(tr.a), ids.get(tr.b)
            if a is not None and b is not None and a.end == b.start:
                keep.append(tr)
            else:
                dropped.append(tr.id)
        t.transitions = keep
    return dropped
