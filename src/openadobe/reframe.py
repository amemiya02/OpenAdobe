"""Reframing for deliverables in another aspect ratio (16:9 → 9:16 / 1:1 …).

``fit``      letterbox/pillarbox (MLT's default scaling)
``fill``     scale to cover the frame, centred
``subject``  scale to cover and pan to keep the tracked subject (face) in frame — keyframes come from
             ``asset.analysis["faces"]["subject"]`` (normalised cx/cy per time) and are compiled into a
             ``transform`` effect, i.e. a real qtblend animation in the engine, not a pixel crop in Python.
"""

from __future__ import annotations

from .model import Asset, Clip, Effect, Project


def fill_geometry(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[float, float, float]:
    """(scale, fitted width, fitted height): the fitted size is the letterboxed size at scale 1."""
    aspect = src_w / src_h
    w_fit = min(out_w, out_h * aspect)
    h_fit = w_fit / aspect
    return max(out_w / w_fit, out_h / h_fit), w_fit, h_fit


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _smooth(values: list[float], alpha: float = 0.35) -> list[float]:
    out: list[float] = []
    for v in values:
        out.append(v if not out else out[-1] + alpha * (v - out[-1]))
    return out


def clip_transform(project: Project, clip: Clip, asset: Asset, out_w: int, out_h: int, mode: str) -> Effect | None:
    if mode == "fit" or asset.kind == "audio":
        return None
    src_w, src_h = project.width, project.height
    if abs(src_w / src_h - out_w / out_h) < 0.01:
        return None
    s, w_fit, h_fit = fill_geometry(src_w, src_h, out_w, out_h)
    shown_w, shown_h = w_fit * s, h_fit * s
    lim_x = max(0.0, (shown_w - out_w) / 2 / out_w)
    lim_y = max(0.0, (shown_h - out_h) / 2 / out_h)
    params: dict = {"scale": s, "x": 0.0, "y": 0.0}
    track = (asset.analysis.get("faces") or {}).get("subject") or []
    if mode == "subject" and track:
        fps = float(project.rate.fps)
        step = max(1, round(fps / 2))  # a keyframe every ~0.5 s, smoothed: a camera operator, not a jitter
        frames = list(range(0, clip.duration, step))
        if frames[-1] != clip.duration - 1:
            frames.append(clip.duration - 1)
        xs, ys = [], []
        for f in frames:
            t = (clip.src_in + f * clip.speed) / fps
            nearest = min(track, key=lambda p: abs(p["t"] - t))
            xs.append(_clamp(-(nearest["cx"] - 0.5) * shown_w / out_w, lim_x))
            ys.append(_clamp(-(nearest["cy"] - 0.5) * shown_h / out_h, lim_y))
        xs, ys = _smooth(xs), _smooth(ys)
        params["x"] = {f: round(x, 4) for f, x in zip(frames, xs, strict=True)}
        params["y"] = {f: round(y, 4) for f, y in zip(frames, ys, strict=True)}
    return Effect(id=f"reframe_{clip.id}", kind="transform", params=params)


def transforms(project: Project, out_w: int, out_h: int, mode: str) -> dict[str, Effect]:
    """Per-clip transforms for picture on main tracks (graphics/overlays keep their own layout)."""
    out: dict[str, Effect] = {}
    for t in project.tracks:
        if t.kind != "video" or t.role != "main":
            continue
        for c in t.clips:
            e = clip_transform(project, c, project.assets[c.asset_id], out_w, out_h, mode)
            if e is not None:
                out[c.id] = e
    return out
