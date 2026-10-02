"""Effect catalogue: stable, agent-facing effect names mapped onto MLT filters.

Agents use catalogue names and human units ("blur radius 8", "saturation 1.2"); this module
validates them and compiles them to MLT filter properties. Animatable params accept keyframes as
``{"0": 1.0, "30": 1.4}`` (clip-local frames, or times like "1.5s" resolved by the op layer) and are
emitted as MLT animation strings (linear interpolation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .model import Effect

Keyframes = dict[int, float]


@dataclass(frozen=True)
class P:
    name: str
    kind: str  # float | int | bool | color | path | choice
    default: Any
    description: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    animatable: bool = False


@dataclass(frozen=True)
class Spec:
    kind: str
    target: str  # video | audio
    description: str
    params: tuple[P, ...]
    compile: Callable[[dict[str, Any], dict], list[tuple[str, dict[str, str]]]]
    tags: tuple[str, ...] = field(default_factory=tuple)


def _num(v: float) -> str:
    return f"{v:.6g}"


def anim(value: Any, scale: Callable[[float], float] = lambda x: x) -> str:
    """A constant or keyframe dict → MLT property string."""
    if isinstance(value, dict):
        return ";".join(f"{int(k)}={_num(scale(float(v)))}" for k, v in sorted(value.items(), key=lambda kv: int(kv[0])))
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return _num(scale(float(value)))
    return str(value)


def value_at(value: Any, frame: int) -> float:
    """Evaluate a constant or keyframe dict at a clip-local frame (linear, clamped)."""
    if not isinstance(value, dict):
        return float(value)
    pts = sorted((int(k), float(v)) for k, v in value.items())
    if frame <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):  # noqa: RUF007
        if x0 <= frame <= x1:
            return y0 + (y1 - y0) * (frame - x0) / (x1 - x0)
    return pts[-1][1]


def _hex(c: str) -> str:
    c = c.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"colour must be RRGGBB, got {c!r}")
    int(c, 16)
    return c.upper()


# ---- compilers -------------------------------------------------------------------------
# Each returns [(mlt_service, {property: value})]; ctx carries width/height/duration.

def _eq(p, ctx):
    return [("avfilter.eq", {"av.brightness": anim(p["brightness"]), "av.contrast": anim(p["contrast"]),
                             "av.saturation": anim(p["saturation"]), "av.gamma": anim(p["gamma"])})]


def _lgg(p, ctx):
    out = {}
    for part in ("lift", "gamma", "gain"):
        out[f"{part}_r"] = out[f"{part}_g"] = out[f"{part}_b"] = _num(p[part])
    return [("lift_gamma_gain", out)]


def _lut(p, ctx):
    return [("avfilter.lut3d", {"av.file": str(p["file"]), "av.interp": p["interp"]})]


def _blur(p, ctx):
    return [("avfilter.gblur", {"av.sigma": anim(p["radius"]), "av.steps": "2"})]


def _sharpen(p, ctx):
    return [("avfilter.unsharp", {"av.luma_msize_x": "5", "av.luma_msize_y": "5", "av.luma_amount": anim(p["amount"])})]


def _vignette(p, ctx):
    return [("avfilter.vignette", {"av.angle": anim(p["strength"], lambda s: 0.1 + s * 1.2)})]


def _flip(p, ctx):
    return [("avfilter.hflip" if p["axis"] == "horizontal" else "avfilter.vflip", {})]


def _crop(p, ctx):
    w, h = ctx["width"], ctx["height"]
    return [("crop", {"left": str(round(p["left"] * w)), "right": str(round(p["right"] * w)),
                      "top": str(round(p["top"] * h)), "bottom": str(round(p["bottom"] * h)), "center": "0"})]


def _transform(p, ctx):
    """Position/scale/rotation/opacity via qtblend's animated rect (Ken Burns, PIP, punch-in)."""
    w, h, dur = ctx["width"], ctx["height"], ctx["duration"]
    keys = {0, max(dur - 1, 0)}
    for name in ("x", "y", "scale", "opacity", "rotation"):
        if isinstance(p[name], dict):
            keys.update(int(k) for k in p[name])
    rect, rot = [], []
    for f in sorted(keys):
        s = value_at(p["scale"], f)
        cx = w / 2 + value_at(p["x"], f) * w
        cy = h / 2 + value_at(p["y"], f) * h
        rw, rh = w * s, h * s
        rect.append(f"{f}={cx - rw / 2:.2f} {cy - rh / 2:.2f} {rw:.2f} {rh:.2f} {value_at(p['opacity'], f):.4f}")
        rot.append(f"{f}={value_at(p['rotation'], f):.3f}")
    return [("qtblend", {"rect": ";".join(rect), "rotation": ";".join(rot), "compositing": "0",
                         "distort": "0", "rotate_center": "1"})]


def _chroma_key(p, ctx):
    color = "0x" + _hex(p["color"])
    out = [("avfilter.chromakey", {"av.color": color, "av.similarity": anim(p["similarity"]),
                                   "av.blend": anim(p["blend"])})]
    if p["despill"]:
        out.append(("avfilter.despill", {"av.type": "green" if _hex(p["color"])[2:4] > _hex(p["color"])[4:6]
                                         else "blue", "av.mix": "0.5", "av.expand": "0"}))
    return out


def _db_to_lin(db: float) -> float:
    return 10 ** (db / 20)


def _highpass(p, ctx):
    return [("avfilter.highpass", {"av.f": _num(p["freq"]), "av.p": "2"})]


def _lowpass(p, ctx):
    return [("avfilter.lowpass", {"av.f": _num(p["freq"]), "av.p": "2"})]


def _band(p, ctx):
    return [("avfilter.equalizer", {"av.f": _num(p["freq"]), "av.g": _num(p["gain_db"]),
                                    "av.t": "q", "av.w": _num(p["q"])})]


def _comp(p, ctx):
    return [("avfilter.acompressor", {"av.threshold": _num(_db_to_lin(p["threshold_db"])), "av.ratio": _num(p["ratio"]),
                                      "av.attack": _num(p["attack_ms"]), "av.release": _num(p["release_ms"]),
                                      "av.makeup": _num(_db_to_lin(p["makeup_db"]))})]


def _denoise(p, ctx):
    return [("avfilter.afftdn", {"av.nr": _num(p["reduction_db"]), "av.nf": _num(p["floor_db"]), "av.tn": "1"})]


def _echo(p, ctx):
    return [("avfilter.aecho", {"av.in_gain": "0.8", "av.out_gain": _num(p["mix"]),
                                "av.delays": _num(p["delay_ms"]), "av.decays": _num(p["decay"])})]


def _limiter(p, ctx):
    return [("avfilter.alimiter", {"av.limit": _num(_db_to_lin(p["ceiling_db"])), "av.level": "0",
                                   "av.attack": "5", "av.release": "50"})]


def _gain(p, ctx):
    return [("volume", {"level": anim(p["db"])})]


CATALOG: dict[str, Spec] = {s.kind: s for s in [
    Spec("color.adjust", "video", "Brightness, contrast, saturation, gamma (Lumetri basic correction).", (
        P("brightness", "float", 0.0, "-1..1 offset", -1, 1, animatable=True),
        P("contrast", "float", 1.0, "0..3 (1 = unchanged)", 0, 3, animatable=True),
        P("saturation", "float", 1.0, "0..3 (0 = black & white)", 0, 3, animatable=True),
        P("gamma", "float", 1.0, "0.1..3", 0.1, 3, animatable=True),
    ), _eq, ("color",)),
    Spec("color.lift_gamma_gain", "video", "Three-way primary grade (shadows/midtones/highlights, neutral).", (
        P("lift", "float", 0.0, "-0.5..0.5 shadows", -0.5, 0.5),
        P("gamma", "float", 1.0, "0.2..3 midtones", 0.2, 3),
        P("gain", "float", 1.0, "0..3 highlights", 0, 3),
    ), _lgg, ("color",)),
    Spec("color.lut", "video", "Apply a 3D LUT (.cube/.3dl), e.g. a log-to-Rec709 or film look.", (
        P("file", "path", None, "absolute path of the LUT"),
        P("interp", "choice", "tetrahedral", "interpolation", choices=("nearest", "trilinear", "tetrahedral")),
    ), _lut, ("color",)),
    Spec("blur", "video", "Gaussian blur (e.g. background blur, privacy, transitions).", (
        P("radius", "float", 8.0, "sigma in pixels", 0, 200, animatable=True),
    ), _blur, ("stylize",)),
    Spec("sharpen", "video", "Unsharp mask.", (
        P("amount", "float", 0.8, "-2..5", -2, 5, animatable=True),
    ), _sharpen, ("stylize",)),
    Spec("vignette", "video", "Darken the frame edges.", (
        P("strength", "float", 0.5, "0..1", 0, 1, animatable=True),
    ), _vignette, ("stylize",)),
    Spec("flip", "video", "Mirror the picture.", (
        P("axis", "choice", "horizontal", "axis", choices=("horizontal", "vertical")),
    ), _flip, ("transform",)),
    Spec("crop", "video", "Crop edges (fractions of frame size); the rest is transparent.", (
        P("left", "float", 0.0, "0..0.5", 0, 0.5), P("right", "float", 0.0, "0..0.5", 0, 0.5),
        P("top", "float", 0.0, "0..0.5", 0, 0.5), P("bottom", "float", 0.0, "0..0.5", 0, 0.5),
    ), _crop, ("transform",)),
    Spec("transform", "video", "Position / scale / rotation / opacity, keyframable: punch-in, Ken Burns, "
         "picture-in-picture.", (
        P("x", "float", 0.0, "horizontal offset of the centre, fraction of width", -2, 2, animatable=True),
        P("y", "float", 0.0, "vertical offset of the centre, fraction of height", -2, 2, animatable=True),
        P("scale", "float", 1.0, "1 = full frame", 0.01, 10, animatable=True),
        P("rotation", "float", 0.0, "degrees clockwise", -3600, 3600, animatable=True),
        P("opacity", "float", 1.0, "0..1", 0, 1, animatable=True),
    ), _transform, ("transform",)),
    Spec("chroma_key", "video", "Green/blue screen key with spill suppression (put the clip on an overlay track).", (
        P("color", "color", "00FF00", "key colour RRGGBB"),
        P("similarity", "float", 0.18, "0.01..1 how close to the key colour", 0.01, 1, animatable=True),
        P("blend", "float", 0.08, "0..1 edge softness", 0, 1, animatable=True),
        P("despill", "bool", True, "remove colour spill from edges"),
    ), _chroma_key, ("keying",)),
    Spec("audio.gain", "audio", "Gain in dB, keyframable (volume automation).", (
        P("db", "float", 0.0, "-60..24", -60, 24, animatable=True),
    ), _gain, ("audio",)),
    Spec("audio.highpass", "audio", "Remove rumble below a frequency (voice: 80–100 Hz).", (
        P("freq", "float", 90.0, "Hz", 10, 2000),
    ), _highpass, ("audio", "eq")),
    Spec("audio.lowpass", "audio", "Remove content above a frequency (telephone/radio looks, hiss).", (
        P("freq", "float", 12000.0, "Hz", 200, 22000),
    ), _lowpass, ("audio", "eq")),
    Spec("audio.eq_band", "audio", "Parametric EQ band (peaking).", (
        P("freq", "float", 3000.0, "centre Hz", 20, 20000),
        P("gain_db", "float", 2.0, "-24..24", -24, 24),
        P("q", "float", 1.0, "bandwidth Q", 0.1, 20),
    ), _band, ("audio", "eq")),
    Spec("audio.compressor", "audio", "Dynamics compressor (voice: -20 dB, 3:1).", (
        P("threshold_db", "float", -20.0, "dBFS", -60, 0), P("ratio", "float", 3.0, "1..20", 1, 20),
        P("attack_ms", "float", 10.0, "ms", 0.01, 2000), P("release_ms", "float", 150.0, "ms", 0.01, 9000),
        P("makeup_db", "float", 3.0, "dB", 0, 36),
    ), _comp, ("audio", "dynamics")),
    Spec("audio.limiter", "audio", "Brickwall limiter.", (
        P("ceiling_db", "float", -1.0, "dBFS", -24, 0),
    ), _limiter, ("audio", "dynamics")),
    Spec("audio.denoise", "audio", "Broadband noise reduction (FFT).", (
        P("reduction_db", "float", 12.0, "dB of reduction", 0.01, 97),
        P("floor_db", "float", -50.0, "noise floor dB", -80, -20),
    ), _denoise, ("audio", "repair")),
    Spec("audio.echo", "audio", "Echo / slap-back.", (
        P("delay_ms", "float", 180.0, "ms", 1, 5000), P("decay", "float", 0.35, "0..1", 0, 1),
        P("mix", "float", 0.7, "output gain", 0, 1),
    ), _echo, ("audio", "creative")),
]}


def describe(kind: str | None = None) -> list[dict]:
    specs = [CATALOG[kind]] if kind else CATALOG.values()
    return [{
        "kind": s.kind, "target": s.target, "description": s.description, "tags": list(s.tags),
        "params": {p.name: {"type": p.kind, "default": p.default, "description": p.description,
                            **({"min": p.lo, "max": p.hi} if p.lo is not None else {}),
                            **({"choices": list(p.choices)} if p.choices else {}),
                            "keyframes": p.animatable} for p in s.params},
    } for s in specs]


def validate(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults, type-check and range-check; keyframe dicts must use integer frame keys."""
    if kind not in CATALOG:
        raise ValueError(f"unknown effect {kind!r}; see `openadobe effects`")
    spec = CATALOG[kind]
    known = {p.name: p for p in spec.params}
    unknown = set(params) - set(known)
    if unknown:
        raise ValueError(f"{kind}: unknown params {sorted(unknown)}; params: {sorted(known)}")
    out: dict[str, Any] = {}
    for name, p in known.items():
        v = params.get(name, p.default)
        if v is None:
            raise ValueError(f"{kind}: {name} is required")
        if isinstance(v, dict):
            if not v:
                raise ValueError(f"{kind}.{name}: keyframes need at least one point")
            if not p.animatable:
                raise ValueError(f"{kind}.{name} cannot be keyframed")
            v = {int(k): _check(kind, p, x) for k, x in v.items()}
            if any(k < 0 for k in v):
                raise ValueError(f"{kind}.{name}: keyframes must be >= 0")
        else:
            v = _check(kind, p, v)
        out[name] = v
    return out


def _check(kind: str, p: P, v: Any) -> Any:
    if p.kind in ("float", "int"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{kind}.{p.name} must be a number")
        if (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi):
            raise ValueError(f"{kind}.{p.name}={v} outside {p.lo}..{p.hi}")
        return float(v) if p.kind == "float" else int(v)
    if p.kind == "bool":
        return bool(v)
    if p.kind == "color":
        return _hex(str(v))
    if p.kind == "choice":
        if v not in p.choices:
            raise ValueError(f"{kind}.{p.name} must be one of {p.choices}")
        return v
    return str(v)


def compile_effect(effect: Effect, width: int, height: int, duration: int) -> list[tuple[str, dict[str, str]]]:
    spec = CATALOG[effect.kind]
    params = validate(effect.kind, effect.params)
    return spec.compile(params, {"width": width, "height": height, "duration": duration})


def target(kind: str) -> str:
    return CATALOG[kind].target
