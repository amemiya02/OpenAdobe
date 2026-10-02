"""Compositions on the timeline (Dynamic Link).

A ``Comp`` is defined in the model (engine + template + params + input assets) and appears on the
timeline through an ``Asset`` of kind ``comp``. The engine output is a cache entry keyed by a hash
of everything that affects the pixels, so editing a comp re-renders only that comp and an unchanged
comp is never rendered twice.

    lottie   → a Lottie JSON written next to the project; MLT renders it natively (no pre-render)
    blender  → a .blend + lossless RGBA (FFV1 .mkv) render under cache/, made before delivery
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .engines.runner import Runner
from .model import Asset, Comp, Probe, Project

TEMPLATE_VERSION = {"lottie": 1, "blender": 2}  # bump when a template's output changes


def engine_module(engine: str):
    if engine == "lottie":
        from .engines import lottie
        return lottie
    if engine == "blender":
        from .engines import blender
        return blender
    raise ValueError(f"unknown comp engine {engine!r}")


def describe(engine: str | None = None) -> list[dict]:
    out = []
    for eng in ([engine] if engine else ["lottie", "blender"]):
        try:
            mod = engine_module(eng)
        except ImportError:
            continue
        for t in mod.describe():
            out.append({"engine": eng, **t})
    return out


def size(project: Project, comp: Comp) -> tuple[int, int]:
    return comp.width or project.width, comp.height or project.height


def _scripts_hash(engine: str) -> str:
    """Hash of the engine's template sources, so editing a template invalidates its cached renders."""
    base = Path(__file__).parent / "engines"
    files = sorted((base / "blender_scripts").glob("*.py")) + [base / "blender.py"] if engine == "blender" \
        else [base / "lottie.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def render_key(project: Project, comp: Comp, engine_version: str = "") -> str:
    inputs = {slot: project.asset(aid).sha256 for slot, aid in sorted(comp.inputs.items())}
    w, h = size(project, comp)
    payload = {
        "engine": comp.engine, "template": comp.template, "params": comp.params, "inputs": inputs,
        "duration": comp.duration, "size": [w, h], "fps": str(project.rate), "alpha": comp.alpha,
        "engine_version": engine_version, "template_version": TEMPLATE_VERSION[comp.engine],
        "scripts": _scripts_hash(comp.engine),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _probe(project: Project, comp: Comp) -> Probe:
    w, h = size(project, comp)
    return Probe(duration_s=project.rate.to_seconds(comp.duration), has_video=True, has_audio=False, width=w,
                 height=h, fps=str(project.rate), frames=comp.duration)


def validate(project: Project, comp: Comp) -> dict[str, Any]:
    """Fill defaults and check params/inputs against the template."""
    for slot, aid in comp.inputs.items():
        asset = project.asset(aid)
        if asset.kind == "comp" and comp.id in _upstream(project, asset.comp_id) | {asset.comp_id}:
            raise ValueError(f"{comp.id} cannot use {aid} as input {slot!r}: it depends on {comp.id} (cycle)")
        if asset.kind == "comp" and project.comps[asset.comp_id].engine == "lottie":
            raise ValueError(f"input {slot!r}: Lottie comp {asset.comp_id} has no rendered video; "
                             "place it on a track above the Blender comp instead")
    mod = engine_module(comp.engine)
    if comp.engine == "blender":
        return mod.validate(comp.template, comp.params, comp.inputs)
    known = {t["name"]: t for t in mod.describe()}
    if comp.template not in known:
        raise ValueError(f"unknown lottie template {comp.template!r}; templates: {sorted(known)}")
    w, h = size(project, comp)
    mod.build(comp.template, comp.params, w, h, float(project.rate.fps), comp.duration)  # raises on bad params
    return comp.params


def materialise(project: Project, root: Path, comp: Comp) -> Asset:
    """Create/refresh the comp's asset. Lottie JSON is written immediately; Blender renders lazily."""
    root = Path(root).resolve()
    comp.params = validate(project, comp)
    asset = next((a for a in project.assets.values() if a.comp_id == comp.id), None)
    key = render_key(project, comp)
    if comp.engine == "lottie":
        mod = engine_module("lottie")
        rel = Path("graphics") / f"{comp.id}-{key}.json"
        (root / rel).parent.mkdir(exist_ok=True)
        w, h = size(project, comp)
        mod.write(comp.template, comp.params, w, h, float(project.rate.fps), comp.duration, root / rel)
        comp.render_key = key
        media = str(rel)
    else:
        media = asset.media if asset else f"cache/{comp.id}-pending.mkv"
        if comp.render_key != key:
            comp.render_key = None  # stale until ensure_rendered runs
    if asset is None:
        asset = Asset(id=project.new_id("m"), name=f"{comp.template} ({comp.id})", kind="comp",
                      source=f"comp:{comp.id}", sha256=key, media=media, probe=_probe(project, comp),
                      duration_frames=comp.duration, comp_id=comp.id)
        project.assets[asset.id] = asset
    else:
        asset.media, asset.sha256 = media, key
        asset.probe, asset.duration_frames = _probe(project, comp), comp.duration
    return asset


def _upstream(project: Project, comp_id: str, seen: set[str] | None = None) -> set[str]:
    """Comps that ``comp_id`` takes as inputs, transitively."""
    seen = set() if seen is None else seen
    for aid in project.comps[comp_id].inputs.values():
        a = project.assets.get(aid)
        if a is not None and a.kind == "comp" and a.comp_id not in seen:
            seen.add(a.comp_id)
            _upstream(project, a.comp_id, seen)
    return seen


def used(project: Project) -> list[str]:
    """Comps the timeline needs, inputs before the comps that consume them (render order)."""
    on_timeline = {project.assets[c.asset_id].comp_id for t in project.tracks for c in t.clips
                   if c.asset_id in project.assets and project.assets[c.asset_id].kind == "comp"}
    order: list[str] = []

    def visit(cid: str) -> None:
        if cid in order:
            return
        for up in sorted(_upstream(project, cid), key=lambda u: len(_upstream(project, u))):
            if up not in order:
                order.append(up)
        order.append(cid)

    for cid in sorted(on_timeline - {None}):
        visit(cid)
    return order


def stale(project: Project, preview: bool = False) -> list[Comp]:
    """Blender comps the timeline needs that must be (re)rendered, in dependency order. Comps not used on the
    timeline are left alone. A preview render satisfies previews only, never a final delivery."""
    out = []
    for c in (project.comps[cid] for cid in used(project)):
        if c.engine != "blender":
            continue
        key = render_key(project, c)
        ok = {key, f"{key}:preview"} if preview else {key}
        if c.render_key not in ok:
            out.append(c)
    return out


def ensure_rendered(runner: Runner, root: Path, project: Project, preview: bool = False) -> list[dict]:
    """Render every out-of-date Blender comp into the cache and point its asset at the result.

    Returns what was rendered. The caller saves the project (cache state is not an edit revision).
    """
    root = Path(root).resolve()
    done = []
    for comp in stale(project, preview):
        mod = engine_module("blender")
        key = render_key(project, comp)
        out = root / "cache" / f"{comp.id}-{key}{'-preview' if preview else ''}.mkv"
        out.parent.mkdir(exist_ok=True)
        if not out.exists():
            inputs = {slot: (root / project.asset(aid).media).resolve() for slot, aid in comp.inputs.items()}
            w, h = size(project, comp)
            info = mod.render(runner, root, comp.template, comp.params, inputs, w, h, project.rate.num,
                              project.rate.den, comp.duration, out, preview=preview)
            done.append({"comp": comp.id, "render": str(out), "seconds": info.get("seconds"),
                         "engine": info.get("engine"), "blend": str(info.get("blend"))})
        else:
            done.append({"comp": comp.id, "render": str(out), "cached": True})
        comp.render_key = f"{key}:preview" if preview else key
        asset = next(a for a in project.assets.values() if a.comp_id == comp.id)
        asset.media, asset.sha256 = str(out.relative_to(root)), key
    return done
