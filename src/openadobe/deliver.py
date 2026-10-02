"""Rendering deliverables: comps → compile → (mix) → loudness pass → final render → sidecars → QC.

Loudness is matched with a measured two-pass approach: render the mix, measure integrated
loudness (EBU R128), set the master gain to hit the target, then render the final file through
a peak limiter. With ``mix.engine == "ardour"`` the mix comes from an Ardour session compiled from
the same model. Renders are named by revision so every version stays comparable.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

from . import captions as cap
from . import comps, perceive, qc, reframe, store
from .engines import mlt
from .engines.runner import Runner
from .model import Deliverable, Project

PRESETS: dict[str, dict] = {
    "youtube": {"width": 1920, "height": 1080, "loudness_lufs": -14.0, "true_peak_db": -1.0, "reframe": "fit"},
    "youtube_4k": {"width": 3840, "height": 2160, "loudness_lufs": -14.0, "true_peak_db": -1.0, "crf": 16},
    "shorts": {"width": 1080, "height": 1920, "loudness_lufs": -14.0, "true_peak_db": -1.0, "reframe": "subject"},
    "tiktok": {"width": 1080, "height": 1920, "loudness_lufs": -14.0, "true_peak_db": -1.0, "reframe": "subject"},
    "reels": {"width": 1080, "height": 1920, "loudness_lufs": -14.0, "true_peak_db": -1.0, "reframe": "subject"},
    "square": {"width": 1080, "height": 1080, "loudness_lufs": -14.0, "true_peak_db": -1.0, "reframe": "subject"},
    "bilibili": {"width": 1920, "height": 1080, "loudness_lufs": -16.0, "true_peak_db": -1.0},
    "broadcast_ebu": {"width": 1920, "height": 1080, "loudness_lufs": -23.0, "true_peak_db": -1.0,
                      "burn_captions": False, "crf": 14},
    "podcast_video": {"width": 1920, "height": 1080, "loudness_lufs": -16.0, "true_peak_db": -1.0},
}


def preset(name: str, preset_name: str) -> Deliverable:
    if preset_name not in PRESETS:
        raise ValueError(f"unknown deliverable preset {preset_name!r}; presets: {sorted(PRESETS)}")
    return Deliverable(name=name, platform=preset_name, **PRESETS[preset_name])


def prepare(root: Path, runner: Runner, preview: bool = False) -> list[dict]:
    """Bring derived media up to date before compiling: render stale comps (Dynamic Link)."""
    with store.locked(root):
        project = store.load(root)
        if not comps.stale(project, preview):
            return []
        done = comps.ensure_rendered(runner, root, project, preview=preview)
        store.save(root, project)
        return done


def _ardour_mix(runner: Runner, root: Path, project: Project, tag: str) -> Path:
    from .engines import ardour

    res = ardour.mixdown(runner, root, project, root / "build" / f"{tag}-ardour", stems=False)
    return Path(res["mix"]).resolve()


def render(root: Path, deliverable: str = "main", preview: bool = False, runner: Runner | None = None,
           run_qc: bool = True) -> dict:
    root = Path(root).resolve()
    runner = runner or Runner()
    comp_renders = prepare(root, runner, preview=preview)
    project = store.load(root)
    if deliverable not in project.deliverables:
        raise KeyError(f"no deliverable {deliverable!r}; have {list(project.deliverables)}")
    d = project.deliverables[deliverable]
    tag = f"{d.name}-r{project.revision:04d}" + ("-preview" if preview else "")
    started = time.time()

    width, height = d.width or project.width, d.height or project.height
    if preview:
        scale = 640 / max(width, height)
        width, height = round(width * scale / 2) * 2, round(height * scale / 2) * 2
    files = (cap.write_files(project, root, tag, width, height) if (d.burn_captions or d.sidecar_srt)
             else {"ass": None, "srt": None})
    moves = reframe.transforms(project, width, height, d.reframe)

    audio_override, mix_fallback = None, None
    if project.mix.engine == "ardour" and not preview:
        from .engines import ardour

        mix_fallback = ardour.losses(project)
        if not mix_fallback:  # never ship a mix that silently differs from the model: fall back to MLT instead
            audio_override = _ardour_mix(runner, root, project, tag)

    master = 0.0 if audio_override else project.mix.master_gain_db  # the Ardour mix already has the master fader
    measured = None
    if not preview and d.loudness_lufs is not None:
        x = mlt.write(project, root, f"{tag}-mixdown", master_gain_db=master, audio_override=audio_override,
                      width=width, height=height)
        wav = mlt.render(runner, x, root / "build" / f"{tag}-mixdown.wav", root, audio_only=True,
                         sample_rate=project.sample_rate)
        measured = perceive.loudness(runner, root, wav)
        i = measured["integrated_lufs"]
        if i is not None and math.isfinite(i):
            master += d.loudness_lufs - i
        wav.unlink(missing_ok=True)

    x = mlt.write(
        project, root, tag,
        captions_ass=files["ass"] if d.burn_captions else None,
        master_gain_db=master,
        limiter_db=None if preview else d.true_peak_db - 0.5,
        width=width, height=height, proxies=preview, audio_override=audio_override,
        video_transform=moves or None,
    )
    out = root / "renders" / f"{tag}.mp4"
    mlt.render(runner, x, out, root, vcodec=d.vcodec, crf=28 if preview else d.crf,
               preset="veryfast" if preview else d.preset, acodec=d.acodec, abitrate=d.abitrate,
               sample_rate=project.sample_rate)
    result = {
        "render": str(out), "deliverable": d.name, "revision": project.revision, "preview": preview,
        "size": f"{width}x{height}", "master_gain_db": round(master, 2), "mlt": str(x),
        "mix_engine": "ardour" if audio_override else "mlt", "reframed_clips": len(moves), "seconds": None,
    }
    if mix_fallback:
        result["mix_fallback"] = {"reason": "the Ardour adapter cannot reproduce these; mixed with MLT instead",
                                  "items": mix_fallback}
    if comp_renders:
        result["comps_rendered"] = comp_renders
    if measured:
        result["pre_loudness_lufs"] = measured["integrated_lufs"]
    if d.sidecar_srt and files.get("srt"):
        srt = out.with_suffix(".srt")
        shutil.copyfile(files["srt"], srt)
        result["srt"] = str(srt)
    if run_qc and not preview:
        report = qc.run(runner, root, project, d, out)
        out.with_suffix(".qc.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        result["qc"] = {k: report[k] for k in ("pass", "failed", "warnings")}
        result["qc_report"] = str(out.with_suffix(".qc.json"))
    result["seconds"] = round(time.time() - started, 1)
    return result


def render_all(root: Path, runner: Runner | None = None) -> dict:
    """Render every deliverable of the project (e.g. 16:9 master + 9:16 short + square)."""
    project = store.load(root)
    results = {name: render(root, name, runner=runner) for name in project.deliverables}
    return {"renders": results, "all_pass": all(r.get("qc", {}).get("pass", True) for r in results.values())}


def make_proxies(runner: Runner, root: Path, project: Project, asset_ids: list[str] | None = None) -> list[str]:
    """360p H.264 proxies (short GOP for fast seeking) used by preview renders; same frames as the mezzanine."""
    made = []
    for asset in project.assets.values():
        if asset_ids and asset.id not in asset_ids:
            continue
        if asset.kind != "video" or (asset.proxy and (Path(root) / asset.proxy).exists()):
            continue
        rel = Path("proxies") / f"{asset.id}.mp4"
        (Path(root) / "proxies").mkdir(exist_ok=True)
        args = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str((Path(root) / asset.media).resolve()),
                "-map", "0:v:0", "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-g", "12", "-pix_fmt", "yuv420p"]
        if asset.probe.has_audio:
            args += ["-map", "0:a:0", "-c:a", "pcm_s16le"]
        args += [str((Path(root) / rel).resolve())]
        runner.run(args, mounts=[Path(root).resolve()])
        asset.proxy = str(rel)
        made.append(asset.id)
    return made
