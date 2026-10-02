"""Gain automation: clip gain + fades + ducking, compiled to MLT keyframes.

Every component is piecewise-linear in dB, so the sum is exact when evaluated at the union of
all breakpoints. Ducking is plain automation computed from where the trigger roles (dialogue)
have material on the timeline — the same result a mixer would draw by hand, and deterministic.
"""

from __future__ import annotations

import math
from itertools import pairwise

from .model import Clip, Ducking, Project, Track

FLOOR_DB = -60.0


def merge_intervals(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def trigger_intervals(project: Project, ducking: Ducking) -> list[tuple[int, int]]:
    """Timeline ranges where trigger roles are audible."""
    spans = [
        (c.start, c.end)
        for t in project.tracks
        if t.kind == "audio" and t.role in ducking.trigger_roles and not t.muted
        for c in t.clips
    ]
    gap = project.rate.to_frames(ducking.merge_gap_ms / 1000)
    return merge_intervals(spans, gap)


def _duck_db(t: float, intervals: list[tuple[int, int]], depth: float, attack: int, release: int) -> float:
    """Ducking contribution at timeline frame t (<= 0)."""
    best = 0.0
    for a, b in intervals:
        if a <= t <= b:
            return depth
        if attack and a - attack < t < a:
            best = min(best, depth * (t - (a - attack)) / attack)
        if release and b < t < b + release:
            best = min(best, depth * (1 - (t - b) / release))
    return best


def _fade_db(x: int, clip: Clip) -> float:
    """Fade contribution at clip-local frame x (<= 0)."""
    last = clip.duration - 1
    db = 0.0
    if clip.fade_in and x < clip.fade_in:
        db = min(db, FLOOR_DB * (1 - x / clip.fade_in))
    if clip.fade_out and x > last - clip.fade_out:
        db = min(db, FLOOR_DB * (1 - (last - x) / clip.fade_out))
    return db


def clip_envelope(project: Project, track: Track, clip: Clip) -> list[tuple[int, float]]:
    """Breakpoints (clip-local frame, dB) for one audio clip."""
    last = clip.duration - 1
    points = {0, last}
    if clip.fade_in:
        points.add(min(clip.fade_in, last))
    if clip.fade_out:
        points.add(max(last - clip.fade_out, 0))

    duck = project.mix.ducking
    intervals: list[tuple[int, int]] = []
    attack = release = 0
    if duck.enabled and track.role in duck.target_roles:
        intervals = trigger_intervals(project, duck)
        attack = project.rate.to_frames(duck.attack_ms / 1000)
        release = project.rate.to_frames(duck.release_ms / 1000)
        crossings = []
        for (_, b1), (a2, _) in pairwise(intervals):
            if attack and release and b1 + release > a2 - attack:
                t = (1 + b1 / release + (a2 - attack) / attack) / (1 / release + 1 / attack)
                crossings += [math.floor(t), math.ceil(t)]
        for a, b in intervals:
            for f in (a - attack, a, b, b + release, *crossings):
                local = f - clip.start
                if 0 < local < last:
                    points.add(local)

    base = clip.gain_db + track.gain_db
    env = []
    for x in sorted(points):
        db = base + _fade_db(x, clip)
        if intervals:
            db += _duck_db(clip.start + x, intervals, duck.depth_db, attack, release)
        env.append((x, max(db, FLOOR_DB)))
    return _simplify(env)


def _simplify(env: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Drop interior points that lie on the line between their neighbours."""
    if len(env) <= 2:
        return env
    out = [env[0]]
    for i in range(1, len(env) - 1):
        (x0, y0), (x1, y1), (x2, y2) = out[-1], env[i], env[i + 1]
        interp = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
        if abs(interp - y1) > 0.01:
            out.append(env[i])
    out.append(env[-1])
    return out


def is_flat_zero(env: list[tuple[int, float]]) -> bool:
    return all(abs(db) < 1e-6 for _, db in env)


def to_keyframes(env: list[tuple[int, float]]) -> str:
    if len({round(db, 2) for _, db in env}) == 1:
        return f"{env[0][1]:.2f}"
    return ";".join(f"{x}={db:.2f}" for x, db in env)


def video_fade_keyframes(clip: Clip) -> str | None:
    """Brightness keyframes (0..1) for fade from/to black, or None."""
    if not clip.fade_in and not clip.fade_out:
        return None
    last = clip.duration - 1
    if last == 0:
        return "0=0"
    pts: dict[int, float] = {0: 1.0, last: 1.0}
    if clip.fade_in:
        pts[0] = 0.0
        pts[min(clip.fade_in, last)] = 1.0
    if clip.fade_out:
        pts[max(last - clip.fade_out, 0)] = min(pts.get(max(last - clip.fade_out, 0), 1.0), 1.0)
        pts[last] = 0.0
    return ";".join(f"{x}={v:g}" for x, v in sorted(pts.items()))
