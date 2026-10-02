"""Media analysis: shot boundaries, beats, faces / subject track.

Everything that decodes media runs ffmpeg inside the engine image (like ``asr.decode_16k``); only the
pure-numpy / librosa / OpenCV post-processing runs on the host. Frames are project-rate frames, times
are seconds. Person matting lives in ``matting.py``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import urllib.request
import uuid
import wave
from pathlib import Path

from .engines.runner import Runner
from .model import Asset, Project

# ---- model cache -----------------------------------------------------------------------

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_NAME
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def model_dir() -> Path:
    d = Path(os.environ.get("OPENADOBE_MODEL_DIR") or Path.home() / ".cache" / "openadobe" / "models")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(name: str, url: str, sha256: str) -> Path:
    """Return the cached model file, downloading it on first use. The hash is always verified."""
    dest = model_dir() / name
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    tmp = dest.with_suffix(f".{uuid.uuid4().hex[:8]}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        got = _sha256(tmp)
        if got != sha256:
            raise RuntimeError(f"{name}: sha256 mismatch (got {got}, expected {sha256})")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _media(root: Path, asset: Asset) -> Path:
    return (Path(root).resolve() / asset.media).resolve()


# ---- shots -----------------------------------------------------------------------------

def parse_scene_metadata(text: str, fps: float) -> list[tuple[int, float]]:
    """(frame, score) pairs from ffmpeg ``metadata=print`` output of ``select=gt(scene,…)``."""
    cuts: list[tuple[int, float]] = []
    frame: int | None = None
    for line in text.splitlines():
        if m := re.search(r"pts_time:([\d.]+)", line):
            frame = round(float(m.group(1)) * fps)
        elif (m := re.search(r"lavfi\.scene_score=([\d.]+)", line)) and frame is not None:
            cuts.append((frame, float(m.group(1))))
            frame = None
    return cuts


def shots_from_cuts(cuts: list[tuple[int, float]], total: int, min_len: int = 1) -> list[dict]:
    """Turn cut frames into contiguous shots covering [0, total). Shot ``score`` is that of its opening cut."""
    kept: list[tuple[int, float]] = []
    for f, s in sorted(cuts):
        if 0 < f < total and (not kept or f - kept[-1][0] >= min_len) and f >= min_len:
            kept.append((f, s))
    starts = [(0, 0.0), *kept]
    return [{"start": f, "end": starts[i + 1][0] if i + 1 < len(starts) else total, "score": round(s, 4)}
            for i, (f, s) in enumerate(starts)]


def shots(runner: Runner, root: Path, project: Project, asset: Asset, threshold: float = 0.3) -> list[dict]:
    """Shot boundaries via ffmpeg's scene score on the mezzanine: ``[{"start", "end", "score"}]`` in frames."""
    media = _media(root, asset)
    fps = float(project.rate.fps)
    proc = runner.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-an", "-vf",
         f"select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"],
        mounts=[Path(root).resolve()],
    )
    total = asset.duration_frames or round(asset.probe.duration_s * fps)
    return shots_from_cuts(parse_scene_metadata(proc.stdout, fps), total)


# ---- beats -----------------------------------------------------------------------------

def pick_downbeats(beats: list[float], strength: list[float], per_bar: int = 4) -> list[float]:
    """Every ``per_bar``-th beat, starting from the phase whose beats have the most onset strength."""
    if len(beats) < per_bar or len(strength) != len(beats):
        return list(beats[:1]) if beats else []
    sums = [sum(strength[p::per_bar]) / len(strength[p::per_bar]) for p in range(per_bar)]
    phase = max(range(per_bar), key=lambda p: sums[p])
    return beats[phase::per_bar]


def beats(runner: Runner, root: Path, project: Project, asset: Asset) -> dict:
    """Tempo, beats, downbeats (best effort) and onsets in seconds, via librosa on container-decoded audio."""
    import librosa
    import numpy as np

    root = Path(root).resolve()
    media = _media(root, asset)
    out = root / "build" / f"{media.stem}.{uuid.uuid4().hex[:6]}.beats.wav"
    out.parent.mkdir(exist_ok=True)
    sr = 22050
    try:
        runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(out)], mounts=[root])
        with wave.open(str(out)) as wf:
            pcm = wf.readframes(wf.getnframes())
    finally:
        out.unlink(missing_ok=True)
    y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if len(y) < sr // 2:
        return {"tempo": 0.0, "beats": [], "downbeats": [], "onsets": []}
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, units="frames")
    onset_frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames")
    beat_t = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)]
    strength = [float(env[min(int(f), len(env) - 1)]) for f in beat_frames]
    return {
        "tempo": round(float(np.atleast_1d(tempo)[0]), 2),
        "beats": [round(t, 3) for t in beat_t],
        "downbeats": [round(t, 3) for t in pick_downbeats(beat_t, strength)],
        "onsets": [round(float(t), 3) for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)],
    }


# ---- faces -----------------------------------------------------------------------------

def normalise_boxes(boxes: list[tuple[float, float, float, float, float]], w: int, h: int) -> list[list[float]]:
    return [[round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4), round(float(s), 3)]
            for x, y, bw, bh, s in boxes]


def track_subject(samples: list[dict], alpha: float = 0.35, jump: float = 0.25) -> list[dict]:
    """Smoothed track of the main face from per-sample normalised boxes.

    The face closest to the previous subject wins (so the track is persistent); the first pick is the
    largest. Positions are exponentially smoothed, held when the face is lost, and leading samples before
    the first detection take the first detected position. With no face at all the track is the frame centre.
    A candidate farther than ``jump`` from the track with a clearly smaller size is ignored as a stray.
    """
    raw: list[tuple[float, float, float] | None] = []
    prev: tuple[float, float, float] | None = None
    for s in samples:
        best = None
        for x, y, w, h, score in s["boxes"]:
            cx, cy, size = x + w / 2, y + h / 2, max(w, h)
            if prev is None:
                cost = -size * score
            else:
                dist = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
                if dist > jump and size < prev[2] * 0.7:
                    continue
                cost = dist - 0.5 * size
            if best is None or cost < best[0]:
                best = (cost, (cx, cy, size))
        if best:
            prev = best[1]
        raw.append(best[1] if best else None)
    cur = next((r for r in raw if r), (0.5, 0.5, 0.0))
    started = False
    out: list[dict] = []
    for s, r in zip(samples, raw, strict=True):
        if r:
            cur = tuple(c + alpha * (n - c) for c, n in zip(cur, r, strict=True)) if started else r
            started = True
        out.append({"t": s["t"], "cx": round(cur[0], 4), "cy": round(cur[1], 4), "size": round(cur[2], 4)})
    return out


def _detector(w: int, h: int):
    import cv2

    try:
        path = fetch_model(YUNET_NAME, YUNET_URL, YUNET_SHA256)
        det = cv2.FaceDetectorYN.create(str(path), "", (w, h), 0.7, 0.3, 5000)

        def run(img):
            det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = det.detect(img)
            return [] if faces is None else [tuple(float(v) for v in (f[0], f[1], f[2], f[3], f[14])) for f in faces]

        return run, "yunet"
    except (OSError, RuntimeError):  # offline / hash failure: Haar cascade shipped with cv2
        casc = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

        def run(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return [(float(x), float(y), float(bw), float(bh), 1.0) for x, y, bw, bh in casc.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))]

        return run, "haar"


def faces(runner: Runner, root: Path, project: Project, asset: Asset, step_s: float = 0.2, width: int = 640) -> dict:
    """Face boxes sampled every ``step_s`` seconds plus a smoothed main-subject track (all normalised 0..1)."""
    import cv2

    root = Path(root).resolve()
    media = _media(root, asset)
    frames_dir = root / "build" / f"faces_{uuid.uuid4().hex[:8]}"
    frames_dir.mkdir(parents=True)
    try:
        runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(media), "-an", "-vf",
                    f"fps={1 / step_s:.6f},scale={width}:-2", "-start_number", "0", str(frames_dir / "f%06d.png")],
                   mounts=[root])
        detect, backend = None, "yunet"
        samples = []
        for i, png in enumerate(sorted(frames_dir.glob("f*.png"))):
            img = cv2.imread(str(png))
            if img is None:
                continue
            if detect is None:
                detect, backend = _detector(img.shape[1], img.shape[0])
            h, w = img.shape[:2]
            samples.append({"t": round(i * step_s, 3), "boxes": normalise_boxes(detect(img), w, h)})
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return {"backend": backend, "step_s": step_s, "samples": samples, "subject": track_subject(samples)}


def face_coverage(result: dict) -> float:
    s = result.get("samples", [])
    return sum(1 for x in s if x["boxes"]) / len(s) if s else 0.0
