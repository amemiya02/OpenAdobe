"""Compile the project model to MLT XML and render it with ``melt``.

Layout of the compiled tractor (Kdenlive uses the same structure):

    track 0   black background (also provides silent audio)
    track 1…  video tracks in model order (bottom → top), audio hidden
    track n…  audio tracks, video hidden
    qtblend   composites every video track onto track 0
    mix       sums every audio track into track 0
    tractor filters: master fx → master gain → peak limiter → burned-in captions

Each clip gets its own producer so its fades, gain automation and effects attach as producer
filters spanning the clip's source range plus any transition handles (keyframes are clip-local,
shifted by the leading handle). A transition becomes a small nested tractor placed in the track's
playlist over the cut: the outgoing and incoming material on two tracks with a luma / frei0r /
mix transition between them — exactly how Shotcut and Kdenlive build same-track transitions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import gcd, hypot
from pathlib import Path

from ..effects import compile_effect, target
from ..envelope import clip_envelope, is_flat_zero, to_keyframes, video_fade_keyframes
from ..model import Asset, Clip, Effect, Project, Track, Transition
from ..timeline import DIP_KINDS, halves, needs_handles, producer_in, producer_length
from .runner import Runner

WIPES = {"wipe_left", "wipe_right", "wipe_up", "wipe_down", "wipe_circle", "wipe_diagonal"}
SLIDES = {
    "push_left": "frei0r.sleid0r_push-left", "push_right": "frei0r.sleid0r_push-right",
    "push_up": "frei0r.sleid0r_push-up", "push_down": "frei0r.sleid0r_push-down",
    "slide_left": "frei0r.sleid0r_slide-left", "slide_right": "frei0r.sleid0r_slide-right",
    "slide_up": "frei0r.sleid0r_slide-up", "slide_down": "frei0r.sleid0r_slide-down",
}


def _props(el: ET.Element, **props) -> ET.Element:
    for k, v in props.items():
        if v is None:
            continue
        p = ET.SubElement(el, "property", name=k.replace("__", "."))
        p.text = str(v)
    return el


def _filter(parent: ET.Element, service: str, fid: str, **props) -> ET.Element:
    el = ET.SubElement(parent, "filter", id=fid)
    return _props(el, mlt_service=service, **props)


def _profile(project: Project, width: int, height: int) -> ET.Element:
    g = gcd(width, height)
    return ET.Element("profile", {
        "description": f"openadobe {width}x{height} {project.rate}",
        "width": str(width), "height": str(height), "progressive": "1",
        "sample_aspect_num": "1", "sample_aspect_den": "1",
        "display_aspect_num": str(width // g), "display_aspect_den": str(height // g),
        "frame_rate_num": str(project.rate.num), "frame_rate_den": str(project.rate.den),
        "colorspace": "709",
    })


# ---- helpers ---------------------------------------------------------------------------

def extend_keyframes(kf: str, pre: int, post: int, duration: int) -> str:
    """Shift clip-local keyframes by the leading handle and hold edge values over the handles."""
    if (not pre and not post) or "=" not in kf:
        return kf
    pts = [(int(a) + pre, b) for a, b in (p.split("=", 1) for p in kf.split(";"))]
    if pre:
        pts.insert(0, (0, pts[0][1]))
    if post:
        pts.append((pre + duration + post - 1, pts[-1][1]))
    seen: dict[int, str] = {}
    for f, v in pts:
        seen[f] = v
    return ";".join(f"{f}={v}" for f, v in sorted(seen.items()))


def _shift_params(params: dict, pre: int) -> dict:
    if not pre:
        return params
    return {k: ({int(f) + pre: v for f, v in val.items()} if isinstance(val, dict) else val)
            for k, val in params.items()}


def luma_map(project_dir: Path, kind: str, width: int, height: int) -> Path:
    """Greyscale wipe map (binary PGM): darker pixels switch to the incoming clip first."""
    path = Path(project_dir) / "build" / "lumas" / f"{kind}_{width}x{height}.pgm"
    if path.exists():
        return path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = max(width // 4, 2), max(height // 4, 2)  # luma maps are scaled by MLT; keep them small
    rmax = hypot(w / 2, h / 2)
    rows = bytearray()
    for y in range(h):
        for x in range(w):
            v = {
                "wipe_left": x / (w - 1), "wipe_right": 1 - x / (w - 1),
                "wipe_up": 1 - y / (h - 1), "wipe_down": y / (h - 1),
                "wipe_circle": hypot(x - w / 2, y - h / 2) / rmax,
                "wipe_diagonal": (x / (w - 1) + y / (h - 1)) / 2,
            }[kind]
            rows.append(round(v * 255))
    path.write_bytes(f"P5\n{w} {h}\n255\n".encode() + bytes(rows))
    return path.resolve()


# ---- producers -------------------------------------------------------------------------

def _resource(project: Project, project_dir: Path, asset: Asset, proxies: bool) -> Path:
    rel = asset.proxy if (proxies and asset.proxy) else asset.media
    return (project_dir / rel).resolve()


def _producer(root: ET.Element, project: Project, project_dir: Path, track: Track, clip: Clip, asset: Asset,
              pre: int, post: int, extra_fade: tuple[int, int], proxies: bool, out_size: tuple[int, int]) -> str:
    pid = f"p_{clip.id}"
    resource = _resource(project, project_dir, asset, proxies)
    pin = producer_in(clip, asset)
    if asset.kind == "image":
        length = pin + clip.duration + post
        el = ET.SubElement(root, "producer", id=pid, attrib={"in": "0", "out": str(length - 1)})
        _props(el, resource=resource, length=length, ttl=1, aspect_ratio=1)
    elif asset.kind == "comp" and project.comps[asset.comp_id].engine == "lottie":
        from . import lottie

        length = asset.duration_frames
        el = ET.SubElement(root, "producer", id=pid, attrib={"in": "0", "out": str(length - 1)})
        _props(el, **lottie.mlt_producer_props(resource, length))
    else:
        length = producer_length(clip, asset)
        el = ET.SubElement(root, "producer", id=pid, attrib={"in": "0", "out": str(length - 1)})
        common = {"video_index": 0 if track.kind == "video" else -1,
                  "audio_index": 0 if track.kind == "audio" else -1}
        if clip.speed == 1.0:
            _props(el, resource=resource, mlt_service="avformat", seekable=1, length=length, **common)
        else:
            _props(el, resource=f"{clip.speed:g}:{resource}", mlt_service="timewarp", length=length,
                   warp_pitch=1 if track.kind == "audio" else 0, **common)

    span = {"in": str(pin - pre), "out": str(pin + clip.duration + post - 1)}
    total = pre + clip.duration + post
    if track.kind == "video":
        fade = video_fade_keyframes(clip)
        if fade:
            f = _filter(el, "brightness", f"fade_{clip.id}", level=extend_keyframes(fade, pre, post, clip.duration),
                        alpha=1)
            f.attrib.update(span)
    else:
        eff = clip
        if extra_fade != (0, 0):
            fi = max(clip.fade_in, extra_fade[0])
            eff = clip.model_copy(update={"fade_in": fi, "fade_out": min(max(clip.fade_out, extra_fade[1]),
                                                                         clip.duration - fi)})
        env = clip_envelope(project, track, eff)
        if not is_flat_zero(env):
            f = _filter(el, "volume", f"gain_{clip.id}",
                        level=extend_keyframes(to_keyframes(env), pre, post, clip.duration))
            f.attrib.update(span)
    for e in clip.effects:
        if not e.enabled or target(e.kind) != track.kind:
            continue
        shifted = Effect(id=e.id, kind=e.kind, params=_shift_params(e.params, pre))
        for i, (service, props) in enumerate(compile_effect(shifted, *out_size, total)):
            f = _filter(el, service, f"fx_{clip.id}_{e.id}_{i}")
            _props(f, **props)
            f.attrib.update(span)
    return pid


def _effect_filters(parent: ET.Element, effects: list[Effect], kind: str, out_size: tuple[int, int], prefix: str,
                    duration: int) -> None:
    for e in effects:
        if not e.enabled or target(e.kind) != kind:
            continue
        for i, (service, props) in enumerate(compile_effect(e, *out_size, duration)):
            f = _filter(parent, service, f"{prefix}_{e.id}_{i}")
            _props(f, **props)


# ---- transitions -----------------------------------------------------------------------

def _transition_tractor(root: ET.Element, project: Project, project_dir: Path, track: Track, tr: Transition,
                        a: Clip, b: Clip, pa: str, pb: str, width: int, height: int) -> str:
    d = tr.duration
    h1, h2 = halves(d)
    a_out = producer_in(a, project.assets[a.asset_id]) + a.duration  # first producer frame after a
    b_in = producer_in(b, project.assets[b.asset_id])
    tid = f"tr_{tr.id}"
    # MLT's XML loader silently drops entries whose producer is defined later: define before use
    col = None
    if not needs_handles(tr.kind):
        color = "0xffffffff" if tr.kind == "dip_white" else "0x000000ff"
        col = ET.SubElement(root, "producer", id=f"{tid}_col", attrib={"in": "0", "out": str(d - 1)})
        _props(col, resource=color, mlt_service="color", length=d, mlt_image_format="rgba")
    pl_a = ET.SubElement(root, "playlist", id=f"{tid}_a")
    pl_b = ET.SubElement(root, "playlist", id=f"{tid}_b")
    if col is None:
        ET.SubElement(pl_a, "entry", producer=pa, attrib={"in": str(a_out - h1), "out": str(a_out + h2 - 1)})
        ET.SubElement(pl_b, "entry", producer=pb, attrib={"in": str(b_in - h1), "out": str(b_in + h2 - 1)})
    else:
        ET.SubElement(pl_a, "entry", producer=pa, attrib={"in": str(a_out - h1), "out": str(a_out - 1)})
        ET.SubElement(pl_a, "entry", producer=col.get("id"), attrib={"in": "0", "out": str(h2 - 1)})
        ET.SubElement(pl_b, "entry", producer=col.get("id"), attrib={"in": "0", "out": str(h1 - 1)})
        ET.SubElement(pl_b, "entry", producer=pb, attrib={"in": str(b_in), "out": str(b_in + h2 - 1)})

    tractor = ET.SubElement(root, "tractor", id=tid, attrib={"in": "0", "out": str(d - 1)})
    multi = ET.SubElement(tractor, "multitrack")
    ET.SubElement(multi, "track", producer=pl_a.get("id"))
    ET.SubElement(multi, "track", producer=pl_b.get("id"))

    def add(service: str, first: int, last: int, **props) -> None:
        t = ET.SubElement(tractor, "transition", id=f"{tid}_{service}_{first}",
                          attrib={"in": str(first), "out": str(last)})
        _props(t, mlt_service=service, a_track=0, b_track=1, **props)

    if track.kind == "audio":
        add("mix", 0, d - 1, start=0, end=1, always_active=1)
    elif tr.kind in DIP_KINDS:
        add("luma", 0, h1 - 1)
        add("luma", h1, d - 1)
    elif tr.kind == "dissolve":
        add("luma", 0, d - 1)
    elif tr.kind in WIPES:
        softness = tr.params.get("softness", 0.08)
        add("luma", 0, d - 1, resource=luma_map(project_dir, tr.kind, width, height), softness=softness)
    elif tr.kind in SLIDES:
        add(SLIDES[tr.kind], 0, d - 1, **{"0": f"0=0;{d - 1}=1"})
    else:
        raise ValueError(f"unsupported transition {tr.kind}")
    return tid


# ---- compile ---------------------------------------------------------------------------

def compile_project(
    project: Project,
    project_dir: Path,
    *,
    captions_ass: Path | None = None,
    master_gain_db: float | None = None,
    limiter_db: float | None = None,
    width: int | None = None,
    height: int | None = None,
    proxies: bool = False,
    audio_override: Path | None = None,
    video_transform: dict[str, Effect] | None = None,
) -> str:
    """Compile to MLT XML.

    ``audio_override`` replaces every audio track with one pre-mixed file (e.g. an Ardour mixdown).
    ``video_transform`` maps clip id → an extra transform effect (used by deliverable reframing).
    """
    project_dir = Path(project_dir).resolve()
    total = project.duration()
    if total <= 0:
        raise ValueError("the timeline is empty")
    width, height = width or project.width, height or project.height

    root = ET.Element("mlt", {"LC_NUMERIC": "C", "version": "7.30.0", "title": project.name, "producer": "main"})
    root.append(_profile(project, width, height))

    black = ET.SubElement(root, "producer", id="black", attrib={"in": "0", "out": str(total - 1)})
    _props(black, length=total, eof="pause", resource="0x000000ff", mlt_service="color",
           mlt_image_format="rgba", set__test_audio=0)
    bg = ET.SubElement(root, "playlist", id="background")
    ET.SubElement(bg, "entry", producer="black", attrib={"in": "0", "out": str(total - 1)})

    video = [t for t in project.tracks if t.kind == "video"]
    audio = [] if audio_override else [t for t in project.tracks if t.kind == "audio"]
    playlists: list[tuple[str, str, bool]] = []  # (kind, playlist id, muted)

    for track in video + audio:
        clips = track.sorted_clips()
        by_a = {t.a: t for t in track.transitions}
        by_b = {t.b: t for t in track.transitions}
        crossfades = {t.id for t in track.transitions if not (track.kind == "audio" and t.kind == "dip")}
        producers = {}
        for c in clips:
            tin, tout = by_b.get(c.id), by_a.get(c.id)
            pre = halves(tin.duration)[0] if tin and tin.id in crossfades and needs_handles(tin.kind) else 0
            post = halves(tout.duration)[1] if tout and tout.id in crossfades and needs_handles(tout.kind) else 0
            extra = (halves(tin.duration)[1] if tin and tin.id not in crossfades else 0,
                     halves(tout.duration)[0] if tout and tout.id not in crossfades else 0)
            asset = project.assets[c.asset_id]
            extra_effects = (video_transform or {}).get(c.id)
            clip = c if extra_effects is None else c.model_copy(update={"effects": [*c.effects, extra_effects]})
            producers[c.id] = _producer(root, project, project_dir, track, clip, asset, pre, post, extra, proxies,
                                        (width, height))

        pl = ET.Element("playlist", id=f"pl_{track.id}")
        _props(pl, oa__track=track.id, oa__role=track.role)
        cursor = 0
        for c in clips:
            tin, tout = by_b.get(c.id), by_a.get(c.id)
            head = halves(tin.duration)[1] if tin and tin.id in crossfades else 0
            tail = halves(tout.duration)[0] if tout and tout.id in crossfades else 0
            vis_start, vis_len = c.start + head, c.duration - head - tail
            if vis_start > cursor:
                ET.SubElement(pl, "blank", length=str(vis_start - cursor))
                cursor = vis_start
            if vis_len > 0:
                pin = producer_in(c, project.assets[c.asset_id]) + head
                ET.SubElement(pl, "entry", producer=producers[c.id],
                              attrib={"in": str(pin), "out": str(pin + vis_len - 1)})
                cursor += vis_len
            if tout and tout.id in crossfades:
                b = next(x for x in clips if x.id == tout.b)
                tid = _transition_tractor(root, project, project_dir, track, tout, c, b, producers[c.id],
                                          producers[b.id], width, height)
                ET.SubElement(pl, "entry", producer=tid, attrib={"in": "0", "out": str(tout.duration - 1)})
                cursor += tout.duration
        _effect_filters(pl, track.fx, track.kind, (width, height), f"tfx_{track.id}", total)
        root.append(pl)
        playlists.append((track.kind, pl.get("id"), track.muted))

    if audio_override:
        ap = ET.SubElement(root, "producer", id="mixdown", attrib={"in": "0", "out": str(total - 1)})
        _props(ap, resource=Path(audio_override).resolve(), mlt_service="avformat", video_index=-1, audio_index=0,
               length=total)
        pl = ET.SubElement(root, "playlist", id="pl_mixdown")
        ET.SubElement(pl, "entry", producer="mixdown", attrib={"in": "0", "out": str(total - 1)})
        playlists.append(("audio", "pl_mixdown", False))

    tractor = ET.SubElement(root, "tractor", id="main", attrib={"in": "0", "out": str(total - 1)})
    _props(tractor, oa__revision=project.revision)
    multi = ET.SubElement(tractor, "multitrack")
    ET.SubElement(multi, "track", producer="background")
    for kind, pid, muted in playlists:
        hide = "both" if muted else ("audio" if kind == "video" else "video")
        ET.SubElement(multi, "track", producer=pid, hide=hide)
    for i, (kind, pid, muted) in enumerate(playlists, start=1):
        if muted:
            continue
        name = pid.removeprefix("pl_")
        if kind == "video":
            t = ET.SubElement(tractor, "transition", id=f"comp_{name}")
            _props(t, a_track=0, b_track=i, mlt_service="qtblend", always_active=1, compositing=0, distort=0)
        else:
            t = ET.SubElement(tractor, "transition", id=f"mix_{name}")
            _props(t, a_track=0, b_track=i, mlt_service="mix", always_active=1, sum=1)

    if not audio_override:
        _effect_filters(tractor, project.mix.master_fx, "audio", (width, height), "mfx", total)
    gain = project.mix.master_gain_db if master_gain_db is None else master_gain_db
    if abs(gain) > 1e-6:
        _filter(tractor, "volume", "master_gain", level=f"{gain:.2f}")
    if limiter_db is not None:
        _filter(tractor, "avfilter.alimiter", "master_limiter", av__limit=f"{10 ** (limiter_db / 20):.4f}",
                av__level=0, av__attack=5, av__release=50)
    if captions_ass is not None:
        _filter(tractor, "avfilter.subtitles", "captions", av__filename=str(Path(captions_ass).resolve()))

    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def verify_references(xml_text: str) -> None:
    """Every producer reference must point at an element defined earlier in the document.

    MLT's XML loader silently drops forward references, which renders wrong pictures without an error.
    """
    root = ET.fromstring(xml_text.split("\n", 1)[1] if xml_text.startswith("<?xml") else xml_text)
    defined: set[str] = set()
    for el in root.iter():
        ref = el.get("producer") if el.tag in ("entry", "track") else None
        if ref is not None and ref not in defined:
            raise RuntimeError(f"compiled MLT references {ref!r} before it is defined")
        if el.tag in ("producer", "playlist", "tractor", "chain") and el.get("id"):
            defined.add(el.get("id"))


def write(project: Project, project_dir: Path, name: str, **kwargs) -> Path:
    build = Path(project_dir) / "build"
    build.mkdir(exist_ok=True)
    path = build / f"{name}.mlt"
    xml = compile_project(project, project_dir, **kwargs)
    verify_references(xml)
    path.write_text(xml, "utf-8")
    return path.resolve()


def render(
    runner: Runner,
    mlt_file: Path,
    out: Path,
    project_dir: Path,
    *,
    vcodec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    acodec: str = "aac",
    abitrate: str = "256k",
    sample_rate: int = 48000,
    audio_only: bool = False,
    frame: int | None = None,
    timeout: float | None = None,
    extra_mounts: tuple[Path, ...] = (),
) -> Path:
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["melt", "-silent", str(mlt_file)]
    if frame is not None:
        args += [f"in={frame}", f"out={frame}"]
    args += ["-consumer", f"avformat:{out}", "real_time=-1", "threads=0"]
    if frame is not None:
        args += ["vcodec=png", "f=image2", "update=1", "an=1"]
    elif audio_only:
        args += ["vn=1", "video_off=1", "acodec=pcm_s24le", f"ar={sample_rate}", "channels=2"]
    else:
        args += [f"vcodec={vcodec}", f"crf={crf}", f"preset={preset}", "pix_fmt=yuv420p", f"acodec={acodec}",
                 f"ab={abitrate}", f"ar={sample_rate}", "channels=2", "movflags=+faststart"]
    runner.run(args, mounts=[Path(project_dir).resolve(), *extra_mounts], timeout=timeout)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"melt produced no output at {out}")
    return out
