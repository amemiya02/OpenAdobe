"""Lottie motion-graphics templates, rendered natively by MLT's `glaxnimate` producer.

Python only fills template parameters into Lottie JSON (an open animation format); it never rasterises
anything. The JSON file is placed on a timeline track as an ordinary MLT producer (see
`mlt_producer_props`) and composited by `qtblend`, whose alpha comes from the Lottie's empty background.

Experimentally established facts (MLT 7.30 / Glaxnimate in the engine image) that shape this module:

* The glaxnimate producer and qtblend both need Qt, which refuses to start without a display. The image
  has no working X server, so run melt with `DISPLAY=:99 QT_QPA_PLATFORM=offscreen` (`MELT_ENV` below):
  MLT only checks that DISPLAY is set, and Qt then renders offscreen.
* Text layers (ty=5) render, with Chinese, through fontconfig. Glaxnimate resolves a text layer's `f` against
  `fonts.list[].fName` to get family + style; a bare family name always gives the Regular face, and an
  unknown name silently falls back to an italic Latin face with no CJK glyphs. So every document declares
  `<font>-Regular` and `<font>-Bold` in `fonts.list` and text layers reference those fNames.
* Glaxnimate reads the text size as points (em = size * 4/3 px); `PT_TO_PX` compensates.
* Glaxnimate ignores text justification (`j`) and text boxes (`ps`/`sz`): text is always left-aligned at the
  layer origin, on the baseline. To centre text we put the layer anchor at the estimated centre of the
  string (`text_width`, a per-character advance heuristic), so scaling pops around the middle too.
* Layer masks (rectangles in layer-local space), trim paths, per-layer transforms and keyframed opacity all
  work. Unknown names or unsupported features are never an error in glaxnimate, they just do not render.
* The Lottie `fr` need not match the project rate: the producer rescales time (48 frames at 24 fps becomes
  60 frames at a 30 fps profile). We still write `fr` = project rate so frame numbers line up 1:1.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

FONT = "Noto Sans CJK SC"
# Glaxnimate treats a text layer's font size `s` as points, not pixels: the em box is s * 4/3 px at 96 dpi.
# All layout in this module is in true pixels; the Lottie `s` is size_px / PT_TO_PX.
PT_TO_PX = 4 / 3
# Environment for running melt on Lottie/qtblend inside the engine image (no X server needed).
MELT_ENV = {"DISPLAY": ":99", "QT_QPA_PLATFORM": "offscreen"}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ---------------------------------------------------------------------------------------------------------
# Parameter specs
# ---------------------------------------------------------------------------------------------------------
def _p(type_: str, default: Any, description: str, **extra: Any) -> dict[str, Any]:
    return {"type": type_, "default": default, "description": description, **extra}


_COMMON = {
    "font": _p("string", FONT, "font family; must be installed in the engine image (CJK: Noto Sans CJK SC)"),
    "color": _p("color", "#FFFFFF", "text colour, #RRGGBB"),
    "accent": _p("color", "#FFC400", "accent colour (bars, lines, highlights), #RRGGBB"),
    "bold": _p("bool", True, "use the bold face for the main text"),
    "shadow": _p("bool", True, "soft drop shadow behind text for legibility over video"),
}


def _frac(default: float, description: str, lo: float = 0.0, hi: float = 1.0) -> dict[str, Any]:
    return _p("number", default, description, min=lo, max=hi)


_SPECS: dict[str, dict[str, Any]] = {
    "title": {
        "description": "Big centred title with optional subtitle; scales and fades in, accent rule draws, fades out.",
        "default_duration_s": 4.0,
        "params": {
            "text": _p("string", "标题 Title", "main title"),
            "subtitle": _p("string", "", "optional subtitle under the title"),
            "size": _frac(0.10, "title font size as a fraction of frame height", 0.01, 0.4),
            "y": _frac(0.5, "vertical centre of the block as a fraction of frame height"),
            **_COMMON,
        },
    },
    "lower_third": {
        "description": "Name plus role in a lower-third: accent bar grows, panel wipes, text reveals, all slide out.",
        "default_duration_s": 5.0,
        "params": {
            "name": _p("string", "张三", "name (first line)"),
            "role": _p("string", "产品经理 Product Lead", "role / description (second line)"),
            "size": _frac(0.05, "name font size as a fraction of frame height", 0.01, 0.2),
            "x": _frac(0.06, "left edge as a fraction of frame width"),
            "y": _frac(0.80, "vertical centre of the block as a fraction of frame height"),
            "panel": _p("bool", True, "translucent dark panel behind the text"),
            **_COMMON,
        },
    },
    "callout": {
        "description": "Label in a rounded box with an animated line and arrow pointing at a target point.",
        "default_duration_s": 4.0,
        "params": {
            "text": _p("string", "重点 Look here", "label text"),
            "target_x": _frac(0.70, "target point x, fraction of frame width"),
            "target_y": _frac(0.35, "target point y, fraction of frame height"),
            "x": _frac(0.30, "label box centre x, fraction of frame width"),
            "y": _frac(0.65, "label box centre y, fraction of frame height"),
            "size": _frac(0.045, "label font size as a fraction of frame height", 0.01, 0.2),
            "box_color": _p("color", "#14161C", "label box fill, #RRGGBB"),
            "arrow": _p("bool", True, "arrow head at the target (otherwise a dot)"),
            **_COMMON,
        },
    },
    "cta": {
        "description": "Pill call-to-action button that pops in with overshoot, pulses once, then shrinks away.",
        "default_duration_s": 3.0,
        "params": {
            "text": _p("string", "关注", "button text, e.g. 关注 / Subscribe"),
            "size": _frac(0.05, "text size as a fraction of frame height", 0.01, 0.2),
            "x": _frac(0.85, "button centre x, fraction of frame width"),
            "y": _frac(0.88, "button centre y, fraction of frame height"),
            **_COMMON, "accent": _p("color", "#FF3B30", "button fill colour, #RRGGBB"),
        },
    },
    "progress_bar": {
        "description": "Thin bar along the top or bottom edge that grows from 0 to 100% over the whole duration.",
        "default_duration_s": 10.0,
        "params": {
            "position": _p("string", "bottom", "top or bottom edge", choices=["top", "bottom"]),
            "thickness": _frac(0.008, "bar height as a fraction of frame height", 0.001, 0.1),
            "accent": _p("color", "#FFC400", "bar colour, #RRGGBB"),
            "track": _p("bool", True, "faint track behind the bar showing the full length"),
            "track_color": _p("color", "#FFFFFF", "track colour (drawn at 25% opacity), #RRGGBB"),
        },
    },
    "kinetic_text": {
        "description": "Words (or characters, for CJK) pop in one after another, hold, then fade out together.",
        "default_duration_s": 4.0,
        "params": {
            "text": _p("string", "让每一个字 都有节奏", "text; spaces separate words, CJK text without spaces animates per character"),
            "mode": _p("string", "auto", "word, char or auto", choices=["auto", "word", "char"]),
            "highlight": _p("string", "", "comma-separated tokens drawn in the accent colour"),
            "stagger_s": _frac(0.14, "delay between tokens in seconds", 0.0, 2.0),
            "size": _frac(0.09, "font size as a fraction of frame height", 0.01, 0.4),
            "x": _frac(0.5, "block centre x, fraction of frame width"),
            "y": _frac(0.5, "block centre y, fraction of frame height"),
            **_COMMON,
        },
    },
}


def describe() -> list[dict]:
    """One entry per template: name, description, params {name: {type, default, description, ...}}, default_duration_s."""
    return [
        {"name": n, "description": s["description"], "params": json.loads(json.dumps(s["params"])),
         "default_duration_s": s["default_duration_s"]}
        for n, s in _SPECS.items()
    ]


def _resolve(template: str, params: dict) -> dict[str, Any]:
    if template not in _SPECS:
        raise ValueError(f"unknown lottie template {template!r}; have {sorted(_SPECS)}")
    spec = _SPECS[template]["params"]
    unknown = sorted(set(params) - set(spec))
    if unknown:
        raise ValueError(f"template {template!r} has no params {unknown}; valid: {sorted(spec)}")
    out: dict[str, Any] = {}
    for name, s in spec.items():
        v = params.get(name, s["default"])
        t = s["type"]
        if t == "string":
            if not isinstance(v, str):
                raise ValueError(f"{name} must be a string")
            if "choices" in s and v not in s["choices"]:
                raise ValueError(f"{name} must be one of {s['choices']}, got {v!r}")
        elif t == "color":
            if not isinstance(v, str) or not _HEX.match(v):
                raise ValueError(f"{name} must be a #RRGGBB colour, got {v!r}")
        elif t == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{name} must be true or false")
        elif t == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a number")
            if not s["min"] <= v <= s["max"]:
                raise ValueError(f"{name} must be within [{s['min']}, {s['max']}], got {v}")
            v = float(v)
        out[name] = v
    return out


# ---------------------------------------------------------------------------------------------------------
# Text width heuristic (glaxnimate cannot centre text, so we need to know how wide a string is)
# ---------------------------------------------------------------------------------------------------------
def _advance(ch: str) -> float:
    """Approximate advance width in em for Noto Sans CJK (Source Han Sans) glyphs."""
    cp = ord(ch)
    if ch == " ":
        return 0.22
    if (0x2E80 <= cp <= 0x9FFF or 0xAC00 <= cp <= 0xD7AF or 0xF900 <= cp <= 0xFAFF
            or 0xFF00 <= cp <= 0xFF60 or 0x3000 <= cp <= 0x303F or 0x20000 <= cp <= 0x3FFFF):
        return 1.0
    if unicodedata.category(ch) in ("So", "Sk") and cp > 0x2000:
        return 1.2  # emoji and symbols
    if ch in "il.,:;'|!jtfI`":
        return 0.27
    if ch in "mM":
        return 0.9
    if ch in "wW":
        return 0.84
    if ch.isdigit():
        return 0.56
    if ch.isupper():
        return 0.645
    if ch.islower():
        return 0.55
    return 0.5


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Estimated rendered width in pixels of a single line of `text` at font size `size` px."""
    return sum(_advance(c) for c in text) * size * (1.06 if bold else 1.0)


# ---------------------------------------------------------------------------------------------------------
# Lottie building blocks
# ---------------------------------------------------------------------------------------------------------
_EASE = {  # name -> (outgoing handle of the keyframe, incoming handle of the next one)
    "linear": ((0.167, 0.167), (0.833, 0.833)),
    "out": ((0.22, 1.0), (0.36, 1.0)),
    "in": ((0.55, 0.0), (1.0, 0.45)),
    "inout": ((0.65, 0.0), (0.35, 1.0)),
}


class Anim(list):
    """Keyframes as [(frame, value, ease_to_next)]; a value is a number or a list of numbers."""


def _vec(v: Any) -> list[float]:
    if isinstance(v, (list, tuple)):
        return list(v) if v and isinstance(v[0], dict) else [float(x) for x in v]  # path keyframes pass through
    return [float(v)]


def _prop(v: Any) -> dict:
    """Lottie property: static, or animated when v is an Anim."""
    if not isinstance(v, Anim):
        return {"a": 0, "k": list(v) if isinstance(v, (list, tuple)) else v}
    kfs = []
    by_time: dict[int, tuple] = {}
    for t, val, ease in sorted(v, key=lambda k: k[0]):
        by_time[round(t)] = (round(t), val, ease)  # a later key at the same frame overrides an earlier one
    pts = list(by_time.values())
    for i, (t, val, ease) in enumerate(pts):
        kf: dict[str, Any] = {"t": t, "s": _vec(val)}
        if i + 1 < len(pts):
            n = len(kf["s"])
            (ox, oy), (ix, iy) = _EASE[ease]
            kf["e"] = _vec(pts[i + 1][1])
            kf["o"] = {"x": [ox] * n, "y": [oy] * n}
            kf["i"] = {"x": [ix] * n, "y": [iy] * n}
        kfs.append(kf)
    return {"a": 1, "k": kfs}


def _times(*ts: float) -> list[int]:
    """Round to frames and force strictly increasing (keyframes at identical times confuse renderers)."""
    out: list[int] = []
    for t in ts:
        t = round(t)
        out.append(t if not out or t > out[-1] else out[-1] + 1)
    return out


def _hexrgb(h: str) -> list[float]:
    return [round(int(h[i:i + 2], 16) / 255, 4) for i in (1, 3, 5)]


class _Doc:
    def __init__(self, name: str, w: int, h: int, fps: float, dur: int, font: str):
        self.name, self.w, self.h, self.fps, self.dur = name, w, h, fps, dur
        self.font = font
        self.layers: list[dict] = []
        self.u = h / 1080.0  # pixel unit that scales with frame height

    # timing helpers --------------------------------------------------------------------------------
    def frames(self, seconds: float) -> float:
        return seconds * self.fps

    def edges(self, intro_s: float = 0.6, outro_s: float = 0.5) -> tuple[int, int, int]:
        """(intro_end, outro_start, outro_end) frames: ease in, hold, ease out; shrinks for short clips."""
        last = max(self.dur - 1, 1)
        intro = min(self.frames(intro_s), last * 0.3)
        outro = min(self.frames(outro_s), last * 0.3)
        a, b, c = _times(intro, last - outro, last)
        return a, max(b, a + 1), max(c, a + 2)

    # layer constructors ----------------------------------------------------------------------------
    def _layer(self, ty: int, name: str, ks: dict) -> dict:
        lay = {"ddd": 0, "ind": len(self.layers) + 1, "ty": ty, "nm": name, "sr": 1, "ks": ks, "ao": 0,
               "ip": 0, "op": self.dur, "st": 0, "bm": 0}
        self.layers.append(lay)
        return lay

    @staticmethod
    def ks(o: Any = 100, p: Any = (0, 0), a: Any = (0, 0), s: Any = (100, 100), r: Any = 0) -> dict:
        def v3(x: Any) -> Any:
            if isinstance(x, Anim):
                return Anim((t, [*_vec(val), 100 if x is s else 0][:3], e) for t, val, e in x)
            return [*x, 100 if x is s else 0]
        return {"o": _prop(o), "r": _prop(r), "p": _prop(v3(p)), "a": _prop(v3(a)), "s": _prop(v3(s))}

    def shape(self, name: str, items: list[dict], ks: dict | None = None) -> dict:
        lay = self._layer(4, name, ks or self.ks())
        lay["shapes"] = [{"ty": "gr", "nm": name, "it": [*items, {
            "ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]},
            "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}]}]
        return lay

    def text(self, name: str, text: str, size: float, color: str, ks: dict, bold: bool = False,
             shadow: bool = False, shadow_ks: dict | None = None, mask: Anim | None = None) -> dict:
        """Text layer whose layer-local origin is the baseline-left of the string. Returns the main layer."""
        fname = f"{self.font}-{'Bold' if bold else 'Regular'}"

        def make(nm: str, col: str, k: dict) -> dict:
            lay = self._layer(5, nm, k)
            lay["t"] = {"d": {"k": [{"s": {"s": round(size / PT_TO_PX, 2), "f": fname, "t": text, "j": 0, "tr": 0,
                                           "lh": round(size * 1.25 / PT_TO_PX, 2), "ls": 0, "fc": _hexrgb(col)}, "t": 0}]},
                        "p": {}, "m": {"g": 1, "a": {"a": 0, "k": [0, 0]}}, "a": []}
            return lay

        def masked(lay: dict) -> dict:
            if mask is not None:
                lay["hasMask"] = True
                lay["masksProperties"] = [{"inv": False, "mode": "a", "nm": "reveal", "o": {"a": 0, "k": 100},
                                           "x": {"a": 0, "k": 0}, "pt": _prop(mask)}]
            return lay

        if shadow:
            masked(make(name + " shadow", "#000000", shadow_ks or ks))
        return masked(make(name, color, ks))

    def fonts(self) -> dict:
        return {"list": [{"fFamily": self.font, "fName": f"{self.font}-{st}", "fStyle": st, "ascent": 75}
                         for st in ("Regular", "Bold")]}

    def document(self) -> dict:
        return {"v": "5.7.0", "fr": self.fps, "ip": 0, "op": self.dur, "w": self.w, "h": self.h, "nm": self.name,
                "ddd": 0, "fonts": self.fonts(), "assets": [], "layers": self.layers, "markers": []}


def _fill(color: str, opacity: float = 100) -> dict:
    return {"ty": "fl", "c": {"a": 0, "k": [*_hexrgb(color), 1]}, "o": _prop(opacity), "r": 1}


def _stroke(color: str, width: float, opacity: float = 100) -> dict:
    return {"ty": "st", "c": {"a": 0, "k": [*_hexrgb(color), 1]}, "o": _prop(opacity), "w": {"a": 0, "k": width},
            "lc": 2, "lj": 2}


def _rect(w: float, h: float, r: float, cx: float = 0, cy: float = 0) -> dict:
    return {"ty": "rc", "d": 1, "s": {"a": 0, "k": [w, h]}, "p": {"a": 0, "k": [cx, cy]}, "r": {"a": 0, "k": r}}


def _path(points: list[tuple[float, float]], closed: bool = False) -> dict:
    z = [[0, 0]] * len(points)
    return {"ty": "sh", "ks": {"a": 0, "k": {"c": closed, "v": [list(p) for p in points], "i": z, "o": z}}}


def _rect_path(x0: float, x1: float, y0: float, y1: float) -> dict:
    z = [[0, 0]] * 4
    return {"c": True, "v": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "i": z, "o": z}


def _fade(d: _Doc, t_in: tuple[int, int], t_out: tuple[int, int], peak: float = 100) -> Anim:
    """Opacity: 0 -> peak across t_in, hold, -> 0 across t_out."""
    return Anim([(0, 0, "linear"), (t_in[0], 0, "out"), (t_in[1], peak, "linear"), (t_out[0], peak, "in"),
                 (t_out[1], 0, "linear")])


def _shadow_ks(d: _Doc, ks: dict, dx: float, dy: float, opacity: float = 38) -> dict:
    """Copy of a text layer's transform, nudged and dimmed, for a drop shadow."""
    out = json.loads(json.dumps(ks))
    p = out["p"]
    if p["a"]:
        for kf in p["k"]:
            kf["s"][0] += dx
            kf["s"][1] += dy
            if "e" in kf:
                kf["e"][0] += dx
                kf["e"][1] += dy
    else:
        p["k"][0] += dx
        p["k"][1] += dy
    o = out["o"]
    if o["a"]:
        for kf in o["k"]:
            kf["s"] = [kf["s"][0] * opacity / 100]
            if "e" in kf:
                kf["e"] = [kf["e"][0] * opacity / 100]
    else:
        o["k"] = o["k"] * opacity / 100
    return out


def _fit(text: str, size: float, bold: bool, max_w: float) -> float:
    w = text_width(text, size, bold)
    return size if w <= max_w else size * max_w / w


def _centered_text(d: _Doc, name: str, text: str, size: float, color: str, cx: float, cy: float, bold: bool,
                   shadow: bool, o: Any, s: Any = (100, 100), dy: Any = None, delta: tuple[float, float] | None = None) -> dict:
    """Text centred (visually) on (cx, cy); scale pivots on that centre. `dy` is an optional Anim of y offset."""
    w = text_width(text, size, bold)
    anchor = (w / 2, -0.36 * size)
    if dy is not None:
        pos: Any = Anim((t, (cx, cy + v), e) for t, v, e in dy)
    else:
        pos = (cx, cy)
    ks = d.ks(o=o, p=pos, a=anchor, s=s)
    sk = _shadow_ks(d, ks, *(delta or (2 * d.u, 3 * d.u))) if shadow else None
    return d.text(name, text, size, color, ks, bold, shadow, sk)


# ---------------------------------------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------------------------------------
def _title(d: _Doc, p: dict) -> None:
    a, b, c = d.edges(0.7, 0.5)
    size = _fit(p["text"], p["size"] * d.h, p["bold"], 0.9 * d.w)
    cx, cy = d.w / 2, p["y"] * d.h
    has_sub = bool(p["subtitle"].strip())
    sub_size = _fit(p["subtitle"], size * 0.42, False, 0.9 * d.w) if has_sub else 0
    rule_h = max(3 * d.u, size * 0.05)
    block_h = size * 0.8 + size * 0.22 + rule_h + (size * 0.2 + sub_size * 0.8 if has_sub else 0)
    ty = cy - block_h / 2 + size * 0.4  # centre of the title line
    rule_y = ty + size * 0.4 + size * 0.22 + rule_h / 2  # accent rule between title and subtitle
    rule_w = min(max(text_width(p["text"], size, p["bold"]) * 0.35, 90 * d.u), 0.5 * d.w)
    rule_ks = d.ks(o=_fade(d, (0, a), (b, c)), p=(cx, rule_y), s=Anim([(0, (0, 100), "linear"), (a * 0.4, (0, 100), "out"),
                                                                     (a, (100, 100), "linear"), (b, (100, 100), "in"),
                                                                     (c, (0, 100), "linear")]))
    title_o = _fade(d, (0, a), (b, c))
    title_s = Anim([(0, (82, 82), "out"), (a, (100, 100), "linear"), (b, (100, 100), "in"), (c, (106, 106), "linear")])
    _centered_text(d, "title", p["text"], size, p["color"], cx, ty, p["bold"], p["shadow"], title_o, title_s)
    d.shape("rule", [_rect(rule_w, rule_h, rule_h / 2), _fill(p["accent"])], rule_ks)
    if has_sub:
        a2, b2, _ = _times(a * 0.7 + d.frames(0.12), b, c)
        sub_o = _fade(d, (a * 0.4, a2), (b2, c))
        sub_dy = Anim([(0, size * 0.25, "linear"), (a * 0.4, size * 0.25, "out"), (a2, 0, "linear"), (b2, 0, "in"), (c, -size * 0.1, "linear")])
        sy = rule_y + rule_h / 2 + size * 0.2 + sub_size * 0.4
        _centered_text(d, "subtitle", p["subtitle"], sub_size, p["color"], cx, sy, False, p["shadow"], sub_o, dy=sub_dy)


def _lower_third(d: _Doc, p: dict) -> None:
    a, b, c = d.edges(0.8, 0.55)
    u = d.u
    s1 = _fit(p["name"], p["size"] * d.h, p["bold"], 0.8 * d.w)
    s2 = _fit(p["role"], s1 * 0.62, False, 0.8 * d.w) if p["role"].strip() else 0
    pad = s1 * 0.5
    bar_w = max(5 * u, s1 * 0.12)
    w1 = text_width(p["name"], s1, p["bold"])
    w2 = text_width(p["role"], s2, False) if s2 else 0
    panel_w = max(w1, w2) * 1.05 + pad * 2.4
    block_h = pad * 0.9 + s1 * 0.78 + (s1 * 0.25 + s2 * 1.0 if s2 else s1 * 0.22)
    x0 = p["x"] * d.w
    cy = p["y"] * d.h
    top = cy - block_h / 2
    tx = x0 + bar_w + pad  # text left edge
    total_w = bar_w + panel_w
    slide = Anim([(0, 0, "linear"), (b, 0, "in"), (c, -(x0 + total_w + 40 * u), "linear")])
    fade_out = Anim([(0, 100, "linear"), (b, 100, "in"), (c, 0, "linear")])
    t_bar, t_panel, t_name, t_role = _times(1, a * 0.45, a * 0.55, a * 0.8)
    t_bar_end = _times(a * 0.5)[0]

    def at(x: float, y: float) -> Anim:
        return Anim((t, (x + v, y), e) for t, v, e in slide)

    # accent bar grows from the bottom edge upwards
    bar_s = Anim([(0, (100, 0), "linear"), (t_bar, (100, 0), "out"), (t_bar_end, (100, 100), "linear"), (c, (100, 100), "linear")])
    d.shape("bar", [_rect(bar_w, block_h, 0, bar_w / 2, -block_h / 2), _fill(p["accent"])],
            d.ks(o=fade_out, p=at(x0, top + block_h), s=bar_s))
    if p["panel"]:
        pan_s = Anim([(0, (0, 100), "linear"), (t_panel, (0, 100), "out"), (a, (100, 100), "linear"), (c, (100, 100), "linear")])
        d.shape("panel", [_rect(panel_w, block_h, 0, panel_w / 2, block_h / 2), _fill("#000000", 55)],
                d.ks(o=fade_out, p=at(x0 + bar_w, top), s=pan_s))

    def reveal(width: float, size: float, t0: int, t1: int) -> Anim:
        y0, y1 = -size * 1.0, size * 0.4
        full = width * 1.2 + pad * 2  # generous: small sizes are hinted wider than the estimate
        return Anim([(0, [_rect_path(-pad, -pad, y0, y1)], "linear"), (t0, [_rect_path(-pad, -pad, y0, y1)], "out"),
                     (t1, [_rect_path(-pad, full, y0, y1)], "linear"), (c, [_rect_path(-pad, full, y0, y1)], "linear")])

    def text_ks(x: float, y: float) -> dict:
        return d.ks(o=fade_out, p=at(x, y))

    ny = top + pad * 0.45 + s1 * 0.78  # baseline of the name line
    nks = text_ks(tx, ny)
    d.text("name", p["name"], s1, p["color"], nks, p["bold"], p["shadow"],
           _shadow_ks(d, nks, 2 * u, 3 * u) if p["shadow"] else None, mask=reveal(w1, s1, t_name, a))
    if s2:
        ry = ny + s1 * 0.25 + s2 * 0.78
        rks = text_ks(tx, ry)
        d.text("role", p["role"], s2, p["accent"], rks, False, p["shadow"],
               _shadow_ks(d, rks, 2 * u, 3 * u) if p["shadow"] else None,
               mask=reveal(w2, s2, t_role, _times(a, a + 1)[0] + 2))


def _callout(d: _Doc, p: dict) -> None:
    a, b, c = d.edges(0.7, 0.5)
    u = d.u
    size = _fit(p["text"], p["size"] * d.h, p["bold"], 0.6 * d.w)
    w = text_width(p["text"], size, p["bold"])
    bw, bh = w + size * 1.3, size * 1.9
    bx, by = p["x"] * d.w, p["y"] * d.h
    tx, ty = p["target_x"] * d.w, p["target_y"] * d.h
    # line from the box edge toward the target
    dx, dy = tx - bx, ty - by
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    k = min((bw / 2) / abs(ux) if ux else math.inf, (bh / 2) / abs(uy) if uy else math.inf)
    sx, sy = bx + ux * k, by + uy * k
    gap = 8 * u
    ex, ey = tx - ux * gap, ty - uy * gap
    length = math.hypot(ex - sx, ey - sy)
    lw = max(3.0, 4 * u)
    t_line0, t_line1 = _times(a * 0.6, a * 0.6 + max(a * 1.1, 3))
    fade = _fade(d, (0, a), (b, c))
    pop_s = Anim([(0, (70, 70), "out"), (a, (100, 100), "linear"), (b, (100, 100), "in"), (c, (92, 92), "linear")])

    if length > 2 * u:
        tm = {"ty": "tm", "s": {"a": 0, "k": 0}, "e": _prop(Anim([(0, 0, "linear"), (t_line0, 0, "inout"), (t_line1, 100, "linear"), (c, 100, "linear")])),
              "o": {"a": 0, "k": 0}, "m": 1}
        d.shape("line", [_path([(sx, sy), (ex, ey)]), _stroke(p["accent"], lw), tm], d.ks(o=_fade(d, (0, 1), (b, c))))
    head = 20 * u + size * 0.25
    if p["arrow"]:
        ang = math.atan2(ey - sy, ex - sx) if length > 1 else math.atan2(dy, dx)
        pts = [(tx - gap * ux + head * math.cos(ang + math.pi - 0.5), ty - gap * uy + head * math.sin(ang + math.pi - 0.5)),
               (tx - gap * ux, ty - gap * uy),
               (tx - gap * ux + head * math.cos(ang + math.pi + 0.5), ty - gap * uy + head * math.sin(ang + math.pi + 0.5))]
        ho = Anim([(0, 0, "linear"), (t_line1 - 1, 0, "linear"), (t_line1 + 2, 100, "linear"), (b, 100, "in"), (c, 0, "linear")])
        d.shape("arrow head", [_path(pts), _stroke(p["accent"], lw)], d.ks(o=ho))
    else:
        r = max(7 * u, size * 0.18)
        ds = Anim([(0, (0, 0), "linear"), (t_line1 - 1, (0, 0), "out"), (t_line1 + 5, (100, 100), "linear"), (b, (100, 100), "in"), (c, (0, 0), "linear")])
        d.shape("dot", [{"ty": "el", "d": 1, "s": {"a": 0, "k": [2 * r, 2 * r]}, "p": {"a": 0, "k": [0, 0]}}, _fill(p["accent"])],
                d.ks(p=(tx, ty), s=ds))
    # box (pops from its centre), border, then the label
    box_ks = d.ks(o=fade, p=(bx, by), s=pop_s)
    d.shape("box", [_rect(bw, bh, bh * 0.3), _fill(p["box_color"], 92), _stroke(p["accent"], max(2.0, 3 * u))], box_ks)
    to = Anim([(0, 0, "linear"), (a * 0.5, 0, "out"), (a, 100, "linear"), (b, 100, "in"), (c, 0, "linear")])
    _centered_text(d, "label", p["text"], size, p["color"], bx, by, p["bold"], False, to, pop_s)


def _cta(d: _Doc, p: dict) -> None:
    a, b, c = d.edges(0.45, 0.35)
    u = d.u
    size = _fit(p["text"], p["size"] * d.h, True, 0.5 * d.w)
    w = text_width(p["text"], size, True)
    bw, bh = w + size * 1.8, size * 2.0
    cx, cy = p["x"] * d.w, p["y"] * d.h
    # clamp so the whole pill stays in frame
    cx = min(max(cx, bw / 2 + 8 * u), d.w - bw / 2 - 8 * u)
    cy = min(max(cy, bh / 2 + 8 * u), d.h - bh / 2 - 8 * u)
    t_over = _times(a * 0.62)[0]
    mid = (a + b) / 2
    t_p0, t_p1, t_p2 = _times(mid - d.frames(0.25), mid, mid + d.frames(0.3))
    t_p0 = max(t_p0, a + 1)
    t_p1, t_p2 = max(t_p1, t_p0 + 1), max(t_p2, t_p1 + 2)
    if t_p2 >= b:  # clip too short to pulse
        pts: list = [(0, (0, 0), "out"), (t_over, (118, 118), "inout"), (a, (100, 100), "linear"), (b, (100, 100), "in"),
                     (c, (0, 0), "linear")]
    else:
        pts = [(0, (0, 0), "out"), (t_over, (118, 118), "inout"), (a, (100, 100), "linear"), (t_p0, (100, 100), "inout"),
               (t_p1, (110, 110), "inout"), (t_p2, (100, 100), "linear"), (b, (100, 100), "in"), (c, (0, 0), "linear")]
    s = Anim(pts)
    o = Anim([(0, 0, "linear"), (max(a * 0.3, 1), 100, "linear"), (b, 100, "in"), (c, 0, "linear")])
    d.shape("pill", [_rect(bw, bh, bh / 2), _fill(p["accent"])], d.ks(o=o, p=(cx, cy), s=s))
    _centered_text(d, "label", p["text"], size, p["color"], cx, cy, True, False, o, s)


def _progress_bar(d: _Doc, p: dict) -> None:
    th = max(2.0, p["thickness"] * d.h)
    y = 0.0 if p["position"] == "top" else d.h - th
    last = max(d.dur - 1, 1)
    if p["track"]:
        d.shape("track", [_rect(d.w, th, 0, d.w / 2, th / 2), _fill(p["track_color"], 25)], d.ks(p=(0, y)))
    s = Anim([(0, (0, 100), "linear"), (last, (100, 100), "linear")])
    d.shape("bar", [_rect(d.w, th, 0, d.w / 2, th / 2), _fill(p["accent"])], d.ks(p=(0, y), s=s))


def _kinetic_text(d: _Doc, p: dict) -> None:
    text = p["text"].strip()
    _, b, c = d.edges(0.4, 0.5)
    mode = p["mode"]
    if mode == "auto":
        mode = "word" if len(text.split()) > 1 else "char"
    if mode == "word":
        tokens = text.split()
        sep = 0.22
    else:
        tokens = [ch for ch in text if not ch.isspace()]
        sep = 0.0
    if not tokens:
        raise ValueError("kinetic_text needs non-empty text")
    size = p["size"] * d.h
    bold = p["bold"]
    widths = [text_width(t, size, bold) for t in tokens]
    # fit: shrink if even the longest single line would exceed the frame, otherwise wrap greedily
    max_w = 0.88 * d.w
    if max(widths) > max_w:
        size *= max_w / max(widths)
        widths = [text_width(t, size, bold) for t in tokens]
    lines: list[list[int]] = [[]]
    cur = 0.0
    for i, w in enumerate(widths):
        add = w + (sep * size if lines[-1] else 0)
        if lines[-1] and cur + add > max_w:
            lines.append([])
            cur = 0.0
            add = w
        lines[-1].append(i)
        cur += add
    lh = size * 1.3
    top = p["y"] * d.h - (len(lines) - 1) * lh / 2
    hl = {h.strip() for h in p["highlight"].split(",") if h.strip()}
    n = len(tokens)
    last = max(d.dur - 1, 1)
    stagger = min(d.frames(p["stagger_s"]), 0.6 * max(b - 1, 1) / max(n, 1))
    pop = max(d.frames(0.3), 2)
    gapx = sep * size
    for li, idxs in enumerate(lines):
        total = sum(widths[i] for i in idxs) + gapx * (len(idxs) - 1)
        x = p["x"] * d.w - total / 2
        cy = top + li * lh
        for i in idxs:
            cx = x + widths[i] / 2
            x += widths[i] + gapx
            t0 = 1 + i * stagger
            t1 = t0 + pop
            t0f, t1f, bo, co = _times(t0, t1, max(b, t1 + 1), max(c, t1 + 2))
            co = min(co, max(last, t1f + 1)) if co > last else co
            o = Anim([(0, 0, "linear"), (t0f, 0, "linear"), (t1f, 100, "linear"), (bo, 100, "in"), (co, 0, "linear")])
            s = Anim([(0, (65, 65), "linear"), (t0f, (65, 65), "out"), (t1f, (100, 100), "linear"), (bo, (100, 100), "in"), (co, (104, 104), "linear")])
            dy = Anim([(0, size * 0.3, "linear"), (t0f, size * 0.3, "out"), (t1f, 0, "linear"), (bo, 0, "in"), (co, -size * 0.1, "linear")])
            color = p["accent"] if tokens[i] in hl else p["color"]
            _centered_text(d, f"tok{i}", tokens[i], size, color, cx, cy, bold, p["shadow"], o, s, dy)


_BUILDERS: dict[str, Callable[[_Doc, dict], None]] = {
    "title": _title, "lower_third": _lower_third, "callout": _callout, "cta": _cta,
    "progress_bar": _progress_bar, "kinetic_text": _kinetic_text,
}


# ---------------------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------------------
def build(template: str, params: dict, width: int, height: int, fps: float, duration_frames: int) -> dict:
    """Return a Lottie JSON document (w x h, `fr`=fps, `op`=duration_frames, transparent background).

    Unknown template names, unknown params and badly typed or out-of-range params raise ValueError.
    """
    p = _resolve(template, params or {})
    if width <= 0 or height <= 0 or fps <= 0 or duration_frames < 2:
        raise ValueError("width, height and fps must be positive and duration_frames >= 2")
    doc = _Doc(template, int(width), int(height), float(fps), int(duration_frames), p.get("font", FONT))
    _BUILDERS[template](doc, p)
    out = doc.document()
    out["layers"] = out["layers"][::-1]  # builders add back-to-front; Lottie lists the top layer first
    for i, lay in enumerate(out["layers"], 1):
        lay["ind"] = i
    return out


def write(template: str, params: dict, width: int, height: int, fps: float, duration_frames: int, out_path: Path) -> Path:
    doc = build(template, params, width, height, fps, duration_frames)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out_path


def mlt_producer_props(path: Path, duration_frames: int) -> dict[str, str]:
    """MLT XML <producer> properties that place this Lottie file on a track.

    Use with `<producer in="0" out="{duration_frames-1}">`. `length` equals the file's own frame count (the
    producer derives the same from `op`); `eof=pause` holds the last frame instead of looping if a clip is
    ever extended past the file. The producer outputs RGBA at the Lottie size (set it equal to the project
    size), background fully transparent, so the track composites with `qtblend` and needs no extra alpha
    handling. The melt process needs a Qt-capable env (see MELT_ENV).
    """
    return {
        "mlt_service": "glaxnimate",
        "resource": str(Path(path).resolve()),
        "length": str(int(duration_frames)),
        "eof": "pause",
        "background": "#00000000",
    }
