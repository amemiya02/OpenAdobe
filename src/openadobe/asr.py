"""Speech recognition with word timestamps (optional `asr` extra: faster-whisper).

Audio is decoded by the engine image's ffmpeg to 16 kHz mono PCM and handed to the model as an
array, so recognition never depends on the host's PyAV/ffmpeg build.
"""

from __future__ import annotations

import os
import wave
from functools import lru_cache
from pathlib import Path

from .captions import words_from_segments
from .engines.runner import Runner
from .model import Transcript


@lru_cache(maxsize=2)
def _model(name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("speech recognition needs the asr extra: uv sync --extra asr") from exc
    device = os.environ.get("OPENADOBE_ASR_DEVICE", "auto")
    return WhisperModel(name, device=device, compute_type=os.environ.get("OPENADOBE_ASR_COMPUTE", "int8"))


def decode_16k(runner: Runner, root: Path, media: Path):
    import numpy as np

    out = Path(root).resolve() / "build" / f"{media.stem}.16k.wav"
    out.parent.mkdir(exist_ok=True)
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(out)], mounts=[Path(root).resolve()])
    with wave.open(str(out)) as wf:
        pcm = wf.readframes(wf.getnframes())
    out.unlink(missing_ok=True)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(runner: Runner, root: Path, media: Path, model: str = "small", language: str | None = None,
               vocabulary: list[str] | None = None) -> Transcript:
    audio = decode_16k(runner, root, Path(media).resolve())
    prompt = "以下是普通话的句子，使用简体中文和标点符号。" if language == "zh" else ""
    if vocabulary:
        prompt += (" 关键词：" if language == "zh" else " Terms: ") + "、".join(vocabulary) + "。"
    segments, info = _model(model).transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True, beam_size=5,
        initial_prompt=prompt or None, hotwords=" ".join(vocabulary) if vocabulary else None,
    )
    words = words_from_segments(list(segments))
    return Transcript(language=info.language, model=f"faster-whisper/{model}", words=words)
