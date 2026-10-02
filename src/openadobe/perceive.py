"""Perception (L4): let the agent see and hear its own work.

Images (frames, contact sheets, waveforms, spectrograms) are written under ``perceive/`` and
returned as paths, so a multimodal model can look at them. Measurements come back as JSON.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from . import captions as cap
from . import store
from .engines import mlt
from .engines.runner import Runner
from .lint import lint
from .model import Project

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def latest_render(root: Path) -> Path:
    renders = sorted((Path(root) / "renders").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not renders:
        raise FileNotFoundError("no renders yet; run `openadobe render` (or pass a media path)")
    return renders[-1].resolve()


def _media(root: Path, media: str | Path | None) -> Path:
    if media is None:
        return latest_render(root)
    path = Path(media)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def _ffmpeg(runner: Runner, root: Path, args: list[str], media: Path):
    mounts = [Path(root).resolve()]
    ro = [] if Path(root).resolve() in media.parents else [media.parent]
    return runner.run(["ffmpeg", "-nostdin", "-hide_banner", *args], mounts=mounts, ro=ro)


def duration(runner: Runner, root: Path, media: Path) -> float:
    proc = runner.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(media)],
                      mounts=[Path(root).resolve()], ro=[media.parent])
    return float(proc.stdout.strip() or 0)


def frame(runner: Runner, root: Path, at: int) -> dict:
    """Render one timeline frame (with captions) straight from the current project."""
    from .deliver import prepare

    prepare(Path(root).resolve(), runner, preview=True)
    project = store.load(root)
    tag = f"frame-r{project.revision:04d}"
    files = cap.write_files(project, root, tag)
    x = mlt.write(project, root, tag, captions_ass=files["ass"])
    out = Path(root) / "perceive" / f"{tag}-{at:06d}.png"
    mlt.render(runner, x, out, root, frame=at)
    return {"image": str(out.resolve()), "frame": at, "time": project.rate.clock(at)}


def contact_sheet(runner: Runner, root: Path, media: str | Path | None = None, n: int = 12, cols: int = 4,
                  width: int = 480) -> dict:
    """A grid of evenly spaced frames, each stamped with its time — one image summarises the cut."""
    path = _media(root, media)
    dur = duration(runner, root, path)
    rows = math.ceil(n / cols)
    out = Path(root) / "perceive" / f"{path.stem}-sheet.jpg"
    interval = max(dur, 0.04) / n
    # select the first real frame at/after each interval so every label is that frame's true time
    # (the fps filter would keep the *last* frame of each slot and mislabel it)
    vf = (f"select='if(isnan(prev_selected_t),1,gte(t-prev_selected_t,{interval:.4f}))',scale={width}:-2,"
          f"drawtext=fontfile={FONT}:text='%{{pts\\:hms}}':x=8:y=8:fontsize=22:fontcolor=white:"
          f"box=1:boxcolor=black@0.6:boxborderw=4,tile={cols}x{rows}:padding=4:color=gray")
    _ffmpeg(runner, root, ["-y", "-v", "error", "-i", str(path), "-vf", vf, "-fps_mode", "passthrough",
                           "-frames:v", "1", "-q:v", "3", str(out)], path)
    return {"image": str(out.resolve()), "frames": n, "duration_s": round(dur, 3)}


def waveform(runner: Runner, root: Path, media: str | Path | None = None) -> dict:
    path = _media(root, media)
    out = Path(root) / "perceive" / f"{path.stem}-wave.png"
    _ffmpeg(runner, root, ["-y", "-v", "error", "-i", str(path), "-filter_complex",
                           "showwavespic=s=1600x400:split_channels=1:colors=0x4fc3f7|0x81c784", "-frames:v", "1",
                           str(out)], path)
    return {"image": str(out.resolve())}


def spectrogram(runner: Runner, root: Path, media: str | Path | None = None) -> dict:
    path = _media(root, media)
    out = Path(root) / "perceive" / f"{path.stem}-spectrum.png"
    _ffmpeg(runner, root, ["-y", "-v", "error", "-i", str(path), "-lavfi",
                           "showspectrumpic=s=1600x600:legend=1:scale=log", str(out)], path)
    return {"image": str(out.resolve())}


def _num(pattern: str, text: str) -> float | None:
    m = re.findall(pattern, text)
    if not m:
        return None
    v = m[-1]
    return -math.inf if v == "-inf" else float(v)


def loudness(runner: Runner, root: Path, media: str | Path | None = None) -> dict:
    """EBU R128: integrated loudness (LUFS), loudness range (LU), true peak (dBTP)."""
    path = _media(root, media)
    proc = _ffmpeg(runner, root, ["-nostats", "-i", str(path), "-vn", "-filter_complex", "ebur128=peak=true",
                                  "-f", "null", "-"], path)
    summary = proc.stderr[proc.stderr.rfind("Summary:"):]
    return {
        "integrated_lufs": _num(r"I:\s+(-?[\d.]+|-inf) LUFS", summary),
        "range_lu": _num(r"LRA:\s+(-?[\d.]+) LU", summary),
        "true_peak_db": _num(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary),
        "media": str(path),
    }


def detect(runner: Runner, root: Path, media: str | Path | None = None, black_min: float = 0.5,
           silence_db: float = -50, silence_min: float = 2.0, freeze_min: float = 2.0) -> dict:
    """Black frames, frozen picture and silence intervals (seconds)."""
    path = _media(root, media)
    proc = _ffmpeg(runner, root, [
        "-nostats", "-i", str(path),
        "-vf", f"blackdetect=d={black_min}:pix_th=0.10,freezedetect=n=0.003:d={freeze_min}",
        "-af", f"silencedetect=n={silence_db}dB:d={silence_min}", "-f", "null", "-"], path)
    err = proc.stderr
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]
    fs = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", err)]
    fe = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", err)]
    dur = None
    if len(fe) < len(fs) or "silence_start" in err:
        dur = duration(runner, root, path)
    freeze = list(zip(fs, fe + [dur] * (len(fs) - len(fe))))
    from .ingest import parse_silence
    silence = parse_silence(err, dur or 0.0)
    return {"black": black, "freeze": freeze, "silence": silence, "media": str(path)}


def inspect(project: Project, root: Path) -> dict:
    """Structured view of the timeline for the agent (positions in frames and clock time)."""
    rate = project.rate
    tracks = []
    for t in project.tracks:
        clips = []
        for c in t.sorted_clips():
            a = project.assets[c.asset_id]
            clips.append({
                "clip": c.id, "asset": a.id, "asset_name": a.name, "start": c.start, "end": c.end,
                "at": f"{rate.clock(c.start)}–{rate.clock(c.end)}", "src": f"{rate.clock(c.src_in)}–{rate.clock(c.src_out)}",
                **({"link": c.link} if c.link else {}),
                **({"gain_db": c.gain_db} if c.gain_db else {}),
                **({"fade_in": c.fade_in} if c.fade_in else {}),
                **({"fade_out": c.fade_out} if c.fade_out else {}),
                **({"name": c.name} if c.name else {}),
            })
        tracks.append({"track": t.id, "kind": t.kind, "role": t.role, "name": t.name, "muted": t.muted,
                       "gain_db": t.gain_db, "clips": clips})
    assets = [{"asset": a.id, "name": a.name, "kind": a.kind, "frames": a.duration_frames,
               "duration": rate.clock(a.duration_frames), "has_audio": a.probe.has_audio,
               "size": f"{a.probe.width}x{a.probe.height}" if a.probe.width else None,
               "transcript_words": len(a.transcript.words) if a.transcript else 0, "tags": a.tags}
              for a in project.assets.values()]
    caps = cap.effective(project)
    return {
        "project": project.name, "revision": project.revision, "fps": str(rate),
        "size": f"{project.width}x{project.height}", "duration": rate.clock(project.duration()),
        "duration_frames": project.duration(), "assets": assets, "tracks": tracks,
        "captions": {"enabled": project.captions.enabled, "source": project.captions.source, "count": len(caps),
                     "items": [{"start": rate.clock(c.start), "end": rate.clock(c.end), "text": c.text}
                               for c in caps[:200]]},
        "mix": project.mix.model_dump(),
        "issues": lint(project),
    }


def transcript(project: Project, asset_id: str) -> dict:
    """Indexed words of an asset's transcript, for edit.cut_words."""
    a = project.asset(asset_id)
    if not a.transcript:
        raise ValueError(f"{asset_id} has no transcript; run asset.transcribe")
    return {"asset": a.id, "language": a.transcript.language,
            "words": [{"i": i, "text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
                      for i, w in enumerate(a.transcript.words)]}
