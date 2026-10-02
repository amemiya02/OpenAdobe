"""Blender comps: templates built headlessly with bpy and rendered to lossless RGBA video (FFV1 gbrap10le in MKV).

Pipeline (all inside the version-locked engine image):
    input clips -> ffmpeg -> PNG sequences (comp fps/size, straight alpha)
    blender -b --factory-startup --python-exit-code 1 -P blender_scripts/<template>.py -- args.json
                                                 -> PNG RGBA16 sequence + <out>.blend
    ffmpeg PNG sequence -> FFV1 gbrap10le .mkv (MLT composites its alpha; it flattens ProRes 4444)

Renderer choice (measured; see tests/test_blender.py): EEVEE Next does start under
xvfb + llvmpipe (OpenGL 4.3) but is unusable (about 85 s of shader compilation, then ~0.5 s per sample
for an empty 640x360 frame). Cycles CPU works (no OpenImageDenoise in this build, so low sample counts
and noise-friendly lighting); the keying and glow templates are pure compositor trees.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

from .runner import Runner

SCRIPTS_DIR = Path(__file__).resolve().parent / "blender_scripts"
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

TEMPLATES: dict[str, dict[str, Any]] = {
    "keying": {
        "description": "Chroma-key a green/blue-screen clip (Keying node + despill + optional garbage matte), "
        "composited over an optional background plate; transparent where no plate is given.",
        "params": {
            "key_color": {"type": "color", "default": "#00FF00", "description": "Screen colour to remove."},
            "clip_black": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
                           "description": "Matte values below this become fully transparent."},
            "clip_white": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                           "description": "Matte values above this become fully opaque."},
            "despill": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0,
                        "description": "Despill strength (removes screen colour reflected on the subject)."},
            "edge_blur": {"type": "float", "default": 1.0, "min": 0.0, "max": 20.0,
                          "description": "Post blur of the matte in pixels (softens edges)."},
            "edge_grow": {"type": "int", "default": 0, "min": -20, "max": 20,
                          "description": "Grow (+) or shrink (-) the matte in pixels."},
            "garbage": {"type": "list[float]", "default": [0.0, 0.0, 0.0, 0.0],
                        "description": "Garbage-matte crop as fractions of the frame removed: [left, right, top, bottom]."},
        },
        "inputs": {
            "fg": {"description": "Green/blue-screen clip.", "required": True},
            "bg": {"description": "Optional background plate shown behind the keyed subject.", "required": False},
        },
        "default_duration_s": 5.0,
        "engine": "compositor",
    },
    "title_3d": {
        "description": "Extruded, bevelled 3D text (CJK supported) with key/rim lights and a slow camera push, "
        "on a transparent background.",
        "params": {
            "text": {"type": "str", "default": "OpenAdobe", "description": "Main title (supports Chinese/Japanese/Korean)."},
            "subtitle": {"type": "str", "default": "", "description": "Optional smaller line below the title."},
            "font": {"type": "str", "default": "Noto Sans CJK SC:bold",
                     "description": "Font as a fontconfig pattern (e.g. 'Noto Sans CJK SC:bold', resolved with fc-match in the container) or a font file path."},
            "text_color": {"type": "color", "default": "#F2F2F2", "description": "Title material colour."},
            "subtitle_color": {"type": "color", "default": "#FFC857", "description": "Subtitle material colour."},
            "rim_color": {"type": "color", "default": "#5AA9FF", "description": "Colour of the back/rim light."},
            "extrude": {"type": "float", "default": 0.12, "min": 0.0, "max": 1.0,
                        "description": "Extrusion depth relative to the text height."},
            "bevel": {"type": "float", "default": 0.015, "min": 0.0, "max": 0.2,
                      "description": "Bevel size relative to the text height."},
            "metallic": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                         "description": "Material metallic amount (metal is noisier at low sample counts)."},
            "camera_push": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.6,
                            "description": "Fraction of the camera distance travelled over the clip (0 = static)."},
            "samples": {"type": "int", "default": 10, "min": 1, "max": 256,
                        "description": "Cycles samples per pixel (no denoiser in this build: 10 is draft-clean, "
                        "32+ is smooth but several times slower; preview uses a third)."},
            "text_width": {"type": "float", "default": 0.7, "min": 0.1, "max": 0.95,
                           "description": "Width of the title as a fraction of the frame at the start."},
        },
        "inputs": {},
        "default_duration_s": 3.0,
        "engine": "CYCLES",
    },
    "glow": {
        "description": "Bloom / fog glow on a clip using the compositor Glare node, plus a subtle colour lift.",
        "params": {
            "threshold": {"type": "float", "default": 0.8, "min": 0.0, "max": 5.0,
                          "description": "Brightness above which pixels glow."},
            "intensity": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0,
                          "description": "Glow gain added back onto the image."},
            "size": {"type": "int", "default": 7, "min": 1, "max": 9,
                     "description": "Glow radius (1 tight .. 9 very wide, relative to frame size)."},
            "style": {"type": "enum", "default": "bloom", "choices": ["bloom", "fog_glow"],
                      "description": "Glare algorithm."},
            "tint": {"type": "color", "default": "#FFE9CC", "description": "Highlight tint of the colour lift."},
            "color_lift": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0,
                           "description": "Strength of the subtle colour lift (0 = glow only)."},
        },
        "inputs": {"clip": {"description": "Clip to add glow to.", "required": True}},
        "default_duration_s": 5.0,
        "engine": "compositor",
    },
    "card_3d": {
        "description": "Input clip on a 3D card that swings or spins in perspective, with soft shadow and "
        "reflection, over transparency.",
        "params": {
            "rotation": {"type": "float", "default": 35.0, "min": 0.0, "max": 180.0,
                         "description": "Swing amount in degrees (+/- around the facing position)."},
            "direction": {"type": "enum", "default": "right", "choices": ["left", "right", "up", "down"],
                          "description": "Which way the card swings first / which axis it turns about."},
            "motion": {"type": "enum", "default": "swing", "choices": ["swing", "spin", "static"],
                       "description": "swing = ease from -rotation to +rotation, spin = full 360 turn, static = fixed pose."},
            "corner_radius": {"type": "float", "default": 0.04, "min": 0.0, "max": 0.5,
                              "description": "Corner radius as a fraction of the card's shorter side."},
            "shadow": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                       "description": "Soft ground shadow strength."},
            "reflection": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0,
                           "description": "Floor reflection strength."},
            "card_scale": {"type": "float", "default": 0.7, "min": 0.1, "max": 1.0,
                           "description": "Card width as a fraction of the frame when facing the camera."},
            "edge_color": {"type": "color", "default": "#1A1A1F", "description": "Card edge (thickness) colour."},
        },
        "inputs": {"clip": {"description": "Clip shown on the card face.", "required": True}},
        "default_duration_s": 3.0,
        "engine": "CYCLES",
    },
}


# a hung Blender must not hold the project lock forever (CPU Cycles measured ~1.3 s/frame at 1080p under load)
RENDER_TIMEOUT_BASE = 600
RENDER_TIMEOUT_PER_FRAME = 30

def describe() -> list[dict]:
    """Template catalogue for agents: name, description, params, inputs, default_duration_s."""
    out = []
    for name, t in TEMPLATES.items():
        out.append({
            "name": name,
            "description": t["description"],
            "params": {k: {f: v for f, v in spec.items() if f in ("type", "default", "description", "choices", "min", "max")}
                       for k, spec in t["params"].items()},
            "inputs": {k: dict(v) for k, v in t["inputs"].items()},
            "default_duration_s": t["default_duration_s"],
        })
    return out


def _coerce(name: str, spec: dict[str, Any], value: Any) -> Any:
    kind = spec["type"]
    if kind in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"param {name!r} must be a number, got {value!r}")
        value = int(value) if kind == "int" else float(value)
        if value < spec.get("min", -1e18) or value > spec.get("max", 1e18):
            raise ValueError(f"param {name!r}={value} outside [{spec.get('min')}, {spec.get('max')}]")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"param {name!r} must be a string")
        return value
    if kind == "color":
        if not isinstance(value, str) or not _HEX.match(value):
            raise ValueError(f"param {name!r} must be a #RRGGBB colour, got {value!r}")
        return value.upper()
    if kind == "enum":
        if value not in spec["choices"]:
            raise ValueError(f"param {name!r} must be one of {spec['choices']}, got {value!r}")
        return value
    if kind == "list[float]":
        if not isinstance(value, list | tuple) or len(value) != len(spec["default"]) or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and 0 <= v < 1 for v in value
        ):
            raise ValueError(f"param {name!r} must be a list of {len(spec['default'])} numbers in [0, 1)")
        return [float(v) for v in value]
    raise ValueError(f"param {name!r}: unsupported type {kind}")  # pragma: no cover


def validate(template: str, params: dict, inputs: dict) -> dict:
    """Return the params with defaults filled in. Raises ValueError on unknown/missing/bad params or inputs."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown blender template {template!r}; available: {sorted(TEMPLATES)}")
    t = TEMPLATES[template]
    params = dict(params or {})
    inputs = dict(inputs or {})
    unknown = sorted(set(params) - set(t["params"]))
    if unknown:
        raise ValueError(f"unknown params for {template}: {unknown}; valid: {sorted(t['params'])}")
    unknown_in = sorted(set(inputs) - set(t["inputs"]))
    if unknown_in:
        raise ValueError(f"unknown inputs for {template}: {unknown_in}; valid slots: {sorted(t['inputs'])}")
    missing = sorted(k for k, v in t["inputs"].items() if v["required"] and not inputs.get(k))
    if missing:
        raise ValueError(f"missing required inputs for {template}: {missing}")
    filled = {k: _coerce(k, spec, params[k]) if k in params else spec["default"] for k, spec in t["params"].items()}
    if template == "keying" and filled["clip_black"] >= filled["clip_white"]:
        raise ValueError("keying: clip_black must be below clip_white")
    if template == "title_3d" and not filled["text"].strip():
        raise ValueError("title_3d: text must not be empty")
    return filled


def _prep_cmd(src: Path, dst: Path, w: int, h: int, fps_num: int, fps_den: int, frames: int, fit: str) -> str:
    """ffmpeg command line turning an input clip/still into an exact-length, comp-rate ProRes 4444 intermediate."""
    if fit == "cover":
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    else:
        vf = f"scale='min({w},iw)':'min({h},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
    vf = f"fps={fps_num}/{fps_den},{vf},tpad=stop=-1:stop_mode=clone,format=yuva444p10le"
    parts = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-frames:v", str(frames), "-an",
             "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-r", f"{fps_num}/{fps_den}", str(dst)]
    return " ".join(shlex.quote(p) for p in parts)


def render(runner: Runner, root: Path, template: str, params: dict, inputs: dict[str, Path], width: int, height: int,
           fps_num: int, fps_den: int, duration_frames: int, out_path: Path, preview: bool = False) -> dict:
    """Render the comp headlessly to out_path (lossless RGBA, FFV1 .mkv) and save the generated .blend next to it.

    Returns {"video": out_path, "blend": path, "seconds": wall_time, "engine": "CYCLES"|"compositor"}.
    `preview=True` renders at half resolution with fewer samples (same lossless RGBA output).
    Next to the video, `<stem>.inputs/` holds the conformed input clips the .blend references (relative paths),
    so the project re-opens with its media. Everything runs in a single container call; frames are rendered
    to container-local scratch because bind-mounted writes are ~20x slower.
    """
    t0 = time.monotonic()
    filled = validate(template, params, inputs or {})
    slots = {k: Path(p).resolve() for k, p in (inputs or {}).items() if p}
    for slot, src in slots.items():
        if not src.exists():
            raise ValueError(f"input {slot!r} does not exist: {src}")
    if duration_frames < 1:
        raise ValueError("duration_frames must be >= 1")
    spec = TEMPLATES[template]
    out_path = Path(out_path).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    blend = out_path.with_suffix(".blend")
    scale = 0.5 if preview else 1.0
    rw, rh = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)

    inputs_dir = out_dir / f"{out_path.stem}.inputs"
    prepared = {slot: inputs_dir / f"{slot}.mov" for slot in slots}
    if slots:
        inputs_dir.mkdir(parents=True, exist_ok=True)
    args = {"template": template, "params": filled, "inputs": {k: str(p) for k, p in prepared.items()},
            "width": width, "height": height, "fps_num": fps_num, "fps_den": fps_den, "frames": duration_frames,
            "preview": preview, "blend": str(blend)}
    args_file = out_dir / f".{out_path.stem}.args.json"
    args_file.write_text(json.dumps(args), encoding="utf-8")
    fit = "contain" if template == "card_3d" else "cover"
    q = shlex.quote
    # encode next to the target and rename last: an interrupted render never leaves a file that looks complete
    part = out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
    blender_cmd = " ".join(q(x) for x in [
        "blender", "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(SCRIPTS_DIR / f"{template}.py"),
        "--", str(args_file)])
    # Lossless 10-bit planar RGBA (FFV1 in MKV): MLT's avformat producer keeps alpha for rgba codecs but flattens
    # ProRes 4444's yuva onto black, so the cache stays in RGB and is never matrix-converted.
    encode = (
        f"ffmpeg -nostdin -y -v error -framerate {fps_num}/{fps_den} -start_number 1 "
        '-i "$W/frames/f_%05d.png" '
        + " ".join(q(x) for x in [
            "-frames:v", str(duration_frames), "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrap10le",
            "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
            "-r", f"{fps_num}/{fps_den}", str(part)])
    )
    script = ["set -e", "W=$(mktemp -d)", "trap 'rm -rf \"$W\"' EXIT", 'mkdir -p "$W/frames"']
    script += [_prep_cmd(src, prepared[slot], rw, rh, fps_num, fps_den, duration_frames, fit) for slot, src in slots.items()]
    script += [
        f'OPENADOBE_WORK="$W" {blender_cmd} > "$W/blender.log" 2>&1 || {{ grep -v "^Fra:" "$W/blender.log" | tail -40 >&2; exit 1; }}',
        'N=$(ls "$W/frames" | wc -l)',
        f'[ "$N" -eq {duration_frames} ] || {{ echo "blender wrote $N frames, expected {duration_frames}" >&2; exit 1; }}',
        f'rm -f {q(str(part))}',
        encode,
        f'mv -f {q(str(part))} {q(str(out_path))}',
    ]
    try:
        runner.run(["sh", "-c", "\n".join(script)], mounts=[out_dir], ro=[SCRIPTS_DIR, *{p.parent for p in slots.values()}],
                   timeout=RENDER_TIMEOUT_BASE + RENDER_TIMEOUT_PER_FRAME * duration_frames)
    finally:
        args_file.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    return {"video": out_path, "blend": blend, "seconds": round(time.monotonic() - t0, 2), "engine": spec["engine"]}
