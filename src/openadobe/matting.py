"""Person matting with Robust Video Matting (ONNX, recurrent), output as lossless FFV1 RGBA (MKV).

MLT composites FFV1/PNG alpha but flattens ProRes 4444 onto black, so mattes are stored as FFV1.

Frames are decoded and encoded by the engine image's ffmpeg through raw pipes; only the neural network runs
on the host (onnxruntime, CoreML when available else CPU). The network runs at a reduced working size and
its alpha is resized back, so cost scales with ``infer_long`` instead of the delivery resolution.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from .analysis import fetch_model
from .engines.runner import Runner
from .model import Asset, Project

RVM_NAME = "rvm_mobilenetv3_fp32.onnx"
RVM_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/" + RVM_NAME
RVM_SHA256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"


def downsample_ratio(w: int, h: int, target: int = 512) -> float:
    """RVM's recommended ratio: the network's internal low-res pass should have a long side near ``target``."""
    return round(min(1.0, target / max(w, h)), 4)


def working_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Inference size: long side <= ``long_side``, both dimensions even, aspect preserved."""
    if max(w, h) <= long_side:
        return w - w % 2, h - h % 2
    k = long_side / max(w, h)
    return max(2, round(w * k / 2) * 2), max(2, round(h * k / 2) * 2)


def _session():
    import onnxruntime as ort

    path = fetch_model(RVM_NAME, RVM_URL, RVM_SHA256)
    avail = ort.get_available_providers()
    # CoreML's partitioning of RVM is slower than the CPU provider here and breaks dynamic shapes
    providers = [p for p in ("CUDAExecutionProvider",) if p in avail] + ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def matte(runner: Runner, root: Path, project: Project, asset: Asset, out_path: Path, infer_long: int = 512) -> Path:
    """Write ``out_path`` (.mkv, FFV1 bgra lossless at the project rate, source audio kept)."""
    import cv2
    import numpy as np

    root = Path(root).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    media = (root / asset.media).resolve()
    w, h = asset.probe.width, asset.probe.height
    if not (w and h):
        raise ValueError(f"{asset.id} has no video stream")
    fps = str(project.rate)
    mounts = [root, out_path.parent]

    iw, ih = working_size(w, h, infer_long)
    ratio = np.array([downsample_ratio(iw, ih)], dtype=np.float32)
    sess = _session()
    states = {n: np.zeros((1, 1, 1, 1), dtype=np.float32) for n in ("r1i", "r2i", "r3i", "r4i")}

    dec = subprocess.Popen(
        runner.command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-an", "-vf", f"fps={fps}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], mounts=mounts),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    enc_args = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                "-r", fps, "-i", "-"]
    if asset.probe.has_audio:
        enc_args += ["-i", str(media), "-map", "0:v", "-map", "1:a:0", "-c:a", "copy", "-shortest"]
    enc_args += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgra", str(out_path)]
    enc = subprocess.Popen(runner.command(enc_args, mounts=mounts), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # drain stderr in the background so a chatty ffmpeg can never block on a full pipe
    errs: dict[str, bytes] = {}
    for key, proc in (("decode", dec), ("encode", enc)):
        threading.Thread(target=lambda k=key, p=proc: errs.__setitem__(k, p.stderr.read()), daemon=True).start()

    size = w * h * 3
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            small = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA) if (iw, ih) != (w, h) else rgb
            src = (small.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            _, pha, *rec = sess.run(None, {"src": src, **states, "downsample_ratio": ratio})
            states = dict(zip(("r1i", "r2i", "r3i", "r4i"), rec, strict=True))
            alpha = pha[0, 0]
            if (iw, ih) != (w, h):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            a8 = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            enc.stdin.write(np.dstack([rgb, a8]).tobytes())
    except BrokenPipeError:
        pass
    except BaseException:
        dec.kill()
        enc.kill()
        raise
    finally:
        try:
            enc.stdin.close()  # always: an encoder left waiting for input would block enc.wait() forever
        except BrokenPipeError:
            pass
        dec.stdout.close()
        dec.wait()
        enc.wait()
    if enc.returncode != 0 or dec.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"matting failed: decode={errs.get('decode', b'')[-400:]!r} "
                           f"encode={errs.get('encode', b'')[-400:]!r}")
    return out_path
