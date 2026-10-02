"""Ingest: probe, hash and normalise source media into the project.

Every video source is transcoded to a constant-frame-rate mezzanine at the project rate
(ProRes 422 LT + 48 kHz PCM). Variable-frame-rate phone footage is the main cause of
audio/video drift, so nothing downstream ever sees the original file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from .engines.runner import Runner
from .model import Asset, Probe, Project

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rate(text: str | None) -> float:
    if not text or text in ("0/0", "0"):
        return 0.0
    num, _, den = text.partition("/")
    return float(num) / float(den or 1)


def probe(runner: Runner, path: Path) -> Probe:
    path = Path(path).resolve()
    proc = runner.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        ro=[path.parent],
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    probe = Probe(duration_s=duration, has_video=video is not None, has_audio=audio is not None)
    if video:
        probe.width, probe.height = video.get("width"), video.get("height")
        probe.fps = video.get("avg_frame_rate") or video.get("r_frame_rate")
        probe.video_codec = video.get("codec_name")
        r, avg = _rate(video.get("r_frame_rate")), _rate(video.get("avg_frame_rate"))
        probe.vfr = bool(r and avg and abs(r - avg) / r > 0.01)
        if str(video.get("nb_frames", "")).isdigit():
            probe.frames = int(video["nb_frames"])
    if audio:
        probe.sample_rate = int(audio.get("sample_rate") or 0) or None
        probe.channels = audio.get("channels")
        probe.audio_codec = audio.get("codec_name")
    return probe


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:40] or "media"


def _stereo(probe: Probe) -> list[str]:
    """Mono sources are copied to both channels at full level (a plain -ac 2 upmix is 3 dB quieter)."""
    if probe.channels == 1:
        return ["-af", "pan=stereo|c0=c0|c1=c0"]
    return ["-ac", "2"]


def ingest(runner: Runner, project: Project, root: Path, src: Path, name: str | None = None) -> Asset:
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"no such file: {src}")
    root = Path(root).resolve()
    digest = sha256(src)
    for existing in project.assets.values():
        if existing.sha256 == digest:
            return existing  # idempotent: same bytes, same asset

    asset_id = project.new_id("m")
    media_dir = root / "media"
    media_dir.mkdir(exist_ok=True)
    stem = f"{asset_id}_{_slug(src.stem)}"
    src_probe = probe(runner, src)
    fps = str(project.rate)

    if src.suffix.lower() in IMAGE_EXT or (src_probe.has_video and src_probe.video_codec in ("png", "mjpeg")
                                             and src_probe.duration_s == 0):
        kind = "image"
        dest = media_dir / f"{stem}{src.suffix.lower()}"
        shutil.copyfile(src, dest)
    elif src_probe.has_video:
        kind = "video"
        dest = media_dir / f"{stem}.mov"
        args = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0"]
        if src_probe.has_audio:
            args += ["-map", "0:a:0", "-c:a", "pcm_s16le", "-ar", str(project.sample_rate), *_stereo(src_probe)]
        args += [
            "-vf", f"fps={fps},setsar=1,format=yuv422p10le",
            "-c:v", "prores_ks", "-profile:v", "1", "-vendor", "apl0",
            "-fps_mode", "cfr", "-r", fps, str(dest),
        ]
        runner.run(args, mounts=[root], ro=[src.parent])
    elif src_probe.has_audio:
        kind = "audio"
        dest = media_dir / f"{stem}.wav"
        runner.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-map", "0:a:0", "-c:a", "pcm_s24le",
             "-ar", str(project.sample_rate), *_stereo(src_probe), str(dest)],
            mounts=[root], ro=[src.parent],
        )
    else:
        raise ValueError(f"{src.name}: no usable video or audio stream")

    media_probe = probe(runner, dest) if kind != "image" else src_probe
    if kind == "image":
        frames = 0
    elif kind == "video":
        frames = media_probe.frames or math.floor(media_probe.duration_s * float(project.rate.fps) + 1e-6)
    else:
        frames = math.floor(media_probe.duration_s * float(project.rate.fps) + 1e-6)
    asset = Asset(
        id=asset_id, name=name or src.stem, kind=kind, source=str(src), sha256=digest,
        media=str(dest.relative_to(root)), probe=media_probe, duration_frames=frames,
    )
    project.assets[asset_id] = asset
    return asset


def detect_silence(runner: Runner, root: Path, asset: Asset, threshold_db: float, min_s: float) -> list[tuple[float, float]]:
    """Silent ranges in source seconds."""
    path = (Path(root) / asset.media).resolve()
    proc = runner.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(path), "-vn", "-af",
         f"silencedetect=n={threshold_db}dB:d={min_s}", "-f", "null", "-"],
        mounts=[Path(root).resolve()],
    )
    return parse_silence(proc.stderr, asset.probe.duration_s)


def parse_silence(stderr: str, duration: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        if m := re.search(r"silence_start: (-?[\d.]+)", line):
            start = max(0.0, float(m.group(1)))
        elif (m := re.search(r"silence_end: ([\d.]+)", line)) and start is not None:
            ranges.append((start, float(m.group(1))))
            start = None
    if start is not None:
        ranges.append((start, duration))
    return ranges
