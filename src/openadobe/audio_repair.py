"""Audio repair chains (the Audition layer): non-destructive processing into NEW audio assets.

A chain is a list of ``{"step": name, **params}`` dicts. Every step maps to ffmpeg filters that run in the
engine container; the whole chain is ONE ffmpeg pass, except ``loudness`` which needs a measuring pass over
everything before it first and is then applied as a plain linear gain (loudnorm ``linear=true``).

Output is always 48 kHz 24-bit stereo WAV that is sample-aligned with the source: same number of samples, no
latency. Filters that delay their output declare ``delay`` (samples at 48 kHz) and the builder trims it off.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engines.runner import EngineError, Runner
from .model import Asset, Project

SAMPLE_RATE = 48000
RNNN_DIR = "/opt/openadobe/rnnn"
RNNN_MODELS = ("cb", "mp", "sh", "bd", "lq")
HEAD = f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"
HEAD_MONO = f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c0"  # mono -> both sides at 0 dB
# Measured in engines:dev (ffmpeg 7.1) by cross-correlating input and output; verified per step by the tests.
ARNNDN_DELAY = 480  # one RNNoise frame
AFFTDN_DELAY = 1200


class ChainError(ValueError):
    """Invalid chain or step parameters."""


def _lin(db: float) -> float:
    return 10 ** (db / 20)


def _num(x: float) -> str:
    return f"{x:.6g}"


@dataclass
class Param:
    name: str
    default: Any
    description: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None

    @property
    def type(self) -> str:
        if self.choices is not None:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        return "int" if isinstance(self.default, int) else "number"

    def coerce(self, step: str, value: Any) -> Any:
        label = f"{step}.{self.name}"
        if self.choices is not None:
            if isinstance(self.default, str):
                value = str(value)
            if value not in self.choices:
                raise ValueError(f"{label}={value!r}: expected one of {list(self.choices)}")
            return value
        if isinstance(self.default, bool):
            if not isinstance(value, bool):
                raise ChainError(f"{label}={value!r}: expected true/false")
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ChainError(f"{label}={value!r}: expected a number")
        if isinstance(self.default, int):
            if value != int(value):
                raise ValueError(f"{label}={value!r}: expected an integer")
            value = int(value)
        else:
            value = float(value)
        if (self.lo is not None and value < self.lo) or (self.hi is not None and value > self.hi):
            raise ValueError(f"{label}={value}: out of range [{self.lo}, {self.hi}]")
        return value


@dataclass
class Step:
    name: str
    description: str
    params: list[Param]
    build: Callable[[dict], list[str]]
    delay: Callable[[dict], int] = field(default=lambda p: 0)  # samples @48k added by the filters

    def describe(self) -> dict:
        return {
            "description": self.description,
            "params": {
                p.name: {k: v for k, v in {
                    "type": p.type, "default": p.default, "min": p.lo, "max": p.hi,
                    "choices": list(p.choices) if p.choices else None, "description": p.description,
                }.items() if v is not None} for p in self.params
            },
        }


# ---- step builders ---------------------------------------------------------------------

def _denoise(p: dict) -> list[str]:
    if p["method"] == "afftdn":
        f = (f"afftdn=nr={_num(p['reduction_db'])}:nf={_num(p['noise_floor_db'])}"
             f"{':tn=1:tr=1' if p['track_noise'] else ''}")
        return [f] * p["passes"]
    return [f"arnndn=m={RNNN_DIR}/{p['model']}.rnnn:mix={_num(p['mix'])}"] * p["passes"]


def _dehum(p: dict) -> list[str]:
    out = []
    for k in range(1, p["harmonics"] + 1):
        f = p["freq"] * k
        if f < SAMPLE_RATE / 2 - 100:
            out.append(f"bandreject=f={_num(f)}:width_type=q:w={_num(p['q'])}")
    return out


def _declick(p: dict) -> list[str]:
    return [(f"adeclick=t={_num(p['threshold'])}:w={_num(p['window_ms'])}:o={_num(p['overlap_pct'])}"
             f":b={_num(p['burst'])}")]


def _deess(p: dict) -> list[str]:
    return [f"deesser=i={_num(p['intensity'])}:m={_num(p['max_reduction'])}:f={_num(p['freq'])}"]


def _highpass(p: dict) -> list[str]:
    return [f"highpass=f={_num(p['freq'])}:poles={p['poles']}"]


def _voice_eq(p: dict) -> list[str]:
    out = []
    if p["warmth_db"]:
        out.append(f"lowshelf=f=150:g={_num(p['warmth_db'])}")
    if p["mud_db"]:
        out.append(f"equalizer=f=300:t=q:w=1:g={_num(p['mud_db'])}")
    if p["presence_db"]:
        out.append(f"equalizer=f=3200:t=q:w=0.9:g={_num(p['presence_db'])}")
    if p["air_db"]:
        out.append(f"highshelf=f=10000:g={_num(p['air_db'])}")
    return out or ["anull"]


def _gate(p: dict) -> list[str]:
    return [(f"agate=threshold={_num(_lin(p['threshold_db']))}:ratio={_num(p['ratio'])}"
             f":attack={_num(p['attack_ms'])}:release={_num(p['release_ms'])}:range={_num(_lin(p['range_db']))}")]


def _knee(p: dict) -> str:  # acompressor knee is 1..8 (linear); keep param in dB, 0..18
    return _num(min(8.0, max(1.0, _lin(p["knee_db"]))))


def _compress(p: dict) -> list[str]:
    return [(f"acompressor=threshold={_num(_lin(p['threshold_db']))}:ratio={_num(p['ratio'])}"
             f":attack={_num(p['attack_ms'])}:release={_num(p['release_ms'])}"
             f":makeup={_num(min(64.0, max(1.0, _lin(p['makeup_db']))))}:knee={_knee(p)}")]


def _level(p: dict) -> list[str]:
    peak = _num(_lin(p["peak_db"]))
    if p["method"] == "dynaudnorm":
        return [f"dynaudnorm=f={p['frame_ms']}:g={p['gauss']}:p={peak}:m={_num(min(100.0, max(1.0, _lin(p['max_gain_db']))))}"]
    comp = f":c={_num(p['compression'])}" if p["compression"] >= 1 else ""
    return [f"speechnorm=p={peak}:e={_num(min(50.0, max(1.0, _lin(p['max_gain_db']))))}{comp}"]


def _denoise_delay(p: dict) -> int:
    return (AFFTDN_DELAY if p["method"] == "afftdn" else ARNNDN_DELAY) * p["passes"]


def _loudness_placeholder(p: dict) -> list[str]:  # resolved by process(); never built directly
    raise ValueError("loudness is built by process() (two-pass)")


P = Param
STEPS: dict[str, Step] = {s.name: s for s in [
    Step("denoise", "Broadband noise reduction. arnndn = RNNoise neural denoiser (best for speech, models: cb "
         "general, mp, sh, bd, lq); afftdn = FFT denoiser for stationary noise (hiss, fans, air conditioning).", [
        P("method", "arnndn", "arnndn (neural, speech) or afftdn (stationary noise)", choices=("arnndn", "afftdn")),
        P("model", "mp", "RNNoise model (arnndn only): mp is the gentlest on voices; cb general purpose; sh, bd, lq "
          "other training sets", choices=RNNN_MODELS),
        P("mix", 1.0, "arnndn wet amount: 1 fully denoised, 0 untouched, negative keeps only the removed noise",
          -1.0, 1.0),
        P("passes", 1, "run the denoiser this many times in series (2 = strong)", 1, 3),
        P("reduction_db", 25.0, "afftdn noise reduction in dB", 0.01, 97.0),
        P("noise_floor_db", -40.0, "afftdn: the stationary noise floor in dBFS (measure it in a pause)", -80.0, -20.0),
        P("track_noise", False, "afftdn follows a drifting noise floor and residual (tn+tr)"),
    ], _denoise, _denoise_delay),
    Step("dehum", "Remove mains hum (50 Hz Europe/Asia, 60 Hz Americas) and its harmonics with narrow notches.", [
        P("freq", 50.0, "fundamental in Hz (50 or 60)", 20.0, 400.0),
        P("harmonics", 5, "number of harmonics to notch, fundamental included", 1, 16),
        P("q", 25.0, "notch quality factor (higher = narrower, gentler on the voice)", 2.0, 200.0),
    ], _dehum),
    Step("declick", "Remove clicks, pops and mouth noises (adeclick).", [
        P("threshold", 2.0, "detection threshold; lower repairs more", 1.0, 100.0),
        P("window_ms", 55.0, "analysis window in ms", 10.0, 100.0),
        P("overlap_pct", 75.0, "window overlap in percent", 50.0, 95.0),
        P("burst", 2.0, "burst fusion; higher merges neighbouring clicks", 0.0, 10.0),
    ], _declick),
    Step("deess", "Tame sibilance (deesser).", [
        P("intensity", 0.4, "0 off .. 1 strongest, how much the sibilant band is turned down", 0.0, 1.0),
        P("max_reduction", 0.5, "maximum amount of reduction, 0..1", 0.0, 1.0),
        P("freq", 0.5, "sibilance centre as a fraction of Nyquist (0.5 is about 12 kHz; 0.3 is about 7 kHz)",
          0.0, 1.0),
    ], _deess),
    Step("highpass", "Remove rumble and plosives below the cutoff.", [
        P("freq", 80.0, "cutoff in Hz", 10.0, 1000.0),
        P("poles", 2, "1 = 6 dB/oct, 2 = 12 dB/oct", 1, 2),
    ], _highpass),
    Step("voice_eq", "Voice tone preset: low shelf warmth, mud cut, presence peak, air shelf.", [
        P("warmth_db", 1.5, "low shelf at 150 Hz", -12.0, 12.0),
        P("mud_db", -1.5, "cut at 300 Hz", -12.0, 12.0),
        P("presence_db", 2.5, "peak at 3.2 kHz for intelligibility", -12.0, 12.0),
        P("air_db", 1.0, "high shelf at 10 kHz", -12.0, 12.0),
    ], _voice_eq),
    Step("gate", "Downward expander that quietens the gaps between phrases.", [
        P("threshold_db", -45.0, "level below which the gate closes", -90.0, 0.0),
        P("ratio", 2.0, "expansion ratio", 1.0, 9000.0),
        P("attack_ms", 10.0, "attack", 0.01, 9000.0),
        P("release_ms", 250.0, "release", 0.01, 9000.0),
        P("range_db", -18.0, "attenuation applied when closed", -90.0, 0.0),
    ], _gate),
    Step("compress", "Dynamics compressor.", [
        P("threshold_db", -20.0, "threshold", -60.0, 0.0),
        P("ratio", 3.0, "ratio", 1.0, 20.0),
        P("attack_ms", 20.0, "attack", 0.01, 2000.0),
        P("release_ms", 250.0, "release", 0.01, 9000.0),
        P("makeup_db", 3.0, "make-up gain", 0.0, 36.0),
        P("knee_db", 6.0, "soft knee width", 0.0, 18.0),
    ], _compress),
    Step("level", "Even out levels: speechnorm (phrase by phrase, fast) or dynaudnorm (smooth, gaussian windowed).", [
        P("method", "speechnorm", "speechnorm or dynaudnorm", choices=("speechnorm", "dynaudnorm")),
        P("peak_db", -1.5, "target peak", -30.0, 0.0),
        P("max_gain_db", 12.0, "maximum gain applied", 0.0, 30.0),
        P("compression", 0.0, "speechnorm: compression factor, 0 = off, else 1..50", 0.0, 50.0),
        P("frame_ms", 500, "dynaudnorm: frame length", 10, 8000),
        P("gauss", 31, "dynaudnorm: gaussian window, odd", 3, 301),
    ], _level),
    Step("loudness", "Two-pass loudness normalisation: measure, then apply a linear gain (loudnorm linear). "
         "Use last.", [
        P("target_lufs", -16.0, "integrated loudness target", -70.0, -5.0),
        P("true_peak_db", -1.0, "true peak ceiling", -9.0, 0.0),
        P("lra", 11.0, "loudness range target", 1.0, 50.0),
    ], _loudness_placeholder),
]}

PRESETS: dict[str, dict] = {
    "podcast_voice": {
        "description": "Spoken word for podcasts/voice-over: highpass 80, RNNoise, de-ess, compress, level, -16 LUFS.",
        "chain": [{"step": "highpass", "freq": 80}, {"step": "denoise"}, {"step": "deess"}, {"step": "compress"},
                  {"step": "level"}, {"step": "loudness", "target_lufs": -16, "true_peak_db": -1}],
    },
    "noisy_interview": {
        "description": "Rescue noisy location dialogue: dehum, strong denoise, highpass, level.",
        "chain": [{"step": "dehum"}, {"step": "denoise", "passes": 2}, {"step": "highpass"}, {"step": "level"}],
    },
    "music_master": {
        "description": "Gentle music master: soft compression then -14 LUFS, true peak -1.",
        "chain": [{"step": "compress", "threshold_db": -18, "ratio": 1.8, "attack_ms": 30, "release_ms": 300,
                   "makeup_db": 1, "knee_db": 9},
                  {"step": "loudness", "target_lufs": -14, "true_peak_db": -1}],
    },
}


def describe() -> dict:
    return {
        "steps": {name: s.describe() for name, s in STEPS.items()},
        "presets": {name: {"description": v["description"], "chain": v["chain"]} for name, v in PRESETS.items()},
        "output": "48 kHz 24-bit stereo WAV, sample-aligned with the source (same length, zero latency)",
    }


# ---- chain -> filters ------------------------------------------------------------------

def preset_chain(name: str) -> list[dict]:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; presets: {list(PRESETS)}")
    return [dict(s) for s in PRESETS[name]["chain"]]


def normalize_chain(chain: list[dict]) -> list[dict]:
    """Validate a chain and fill defaults: ``[{"step": "denoise"}]`` -> full parameter dicts."""
    if not chain:
        raise ValueError("empty chain")
    out = []
    for i, raw in enumerate(chain):
        if not isinstance(raw, dict) or "step" not in raw:
            raise ValueError(f"chain[{i}] must be an object with a 'step' key")
        name = raw["step"]
        if name not in STEPS:
            raise ValueError(f"chain[{i}]: unknown step {name!r}; steps: {list(STEPS)}")
        spec = STEPS[name]
        known = {p.name: p for p in spec.params}
        extra = set(raw) - set(known) - {"step"}
        if extra:
            raise ValueError(f"chain[{i}] {name}: unknown params {sorted(extra)}; valid: {list(known)}")
        out.append({"step": name, **{p.name: p.coerce(name, raw.get(p.name, p.default)) for p in spec.params}})
    if sum(s["step"] == "loudness" for s in out) > 1:
        raise ValueError("at most one loudness step per chain")
    return out


def build_filters(chain: list[dict]) -> list[str]:
    """Filters for a normalised chain without any loudness step."""
    out: list[str] = []
    for s in chain:
        if s["step"] == "loudness":
            raise ValueError("loudness must be resolved by process()")
        out += STEPS[s["step"]].build(s)
    return out


def chain_delay(chain: list[dict]) -> int:
    return sum(STEPS[s["step"]].delay(s) for s in chain)


def filter_string(chain: list[dict], samples: int | None = None, loudness_gain: dict | None = None,
                  mono: bool = False) -> str:
    """The full -af string. ``loudness_gain`` carries loudnorm's measured values for the second pass; the
    loudness step is taken wherever it sits in the chain. Pure: used by tests."""
    chain = normalize_chain(chain)
    parts = [HEAD_MONO if mono else HEAD]
    delay = sum(STEPS[s["step"]].delay(s) for s in chain)
    if delay:  # silence in so the delayed tail is flushed, then drop the delay at the end: sample aligned
        parts.append(f"apad=pad_len={delay}")
    for s in chain:
        if s["step"] == "loudness":
            if loudness_gain is not None:
                parts.append(_loudnorm(s, loudness_gain))
        else:
            parts += STEPS[s["step"]].build(s)
    if delay:
        parts.append(f"atrim=start_sample={delay},asetpts=PTS-STARTPTS")
    if samples is not None:
        parts.append(f"apad=whole_len={samples}")
        parts.append(f"atrim=end_sample={samples}")
    return ",".join(parts)


def _loudnorm(s: dict, m: dict) -> str:
    lra = min(50.0, max(s["lra"], math.ceil(float(m["input_lra"])) + 1.0))
    return (f"loudnorm=I={_num(s['target_lufs'])}:TP={_num(s['true_peak_db'])}:LRA={_num(lra)}"
            f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
            f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true:print_format=json,"
            f"aresample={SAMPLE_RATE}")


# ---- running ---------------------------------------------------------------------------

def _json_block(stderr: str) -> dict:
    blocks = re.findall(r"\{[^{}]*\}", stderr, re.DOTALL)
    if not blocks:
        raise EngineError("loudnorm printed no measurement", stderr)
    return json.loads(blocks[-1])


def _input_samples(runner: Runner, src: Path) -> int:
    proc = runner.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                       "stream=duration,sample_rate", "-of", "json", str(src)], ro=[src.parent])
    st = json.loads(proc.stdout)["streams"]
    if not st:
        raise ValueError(f"{src.name} has no audio stream")
    dur = float(st[0].get("duration") or 0)
    if dur <= 0:
        proc = runner.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)],
                          ro=[src.parent])
        dur = float(json.loads(proc.stdout)["format"]["duration"])
    return round(dur * SAMPLE_RATE)


def measure(runner: Runner, path: Path, tp: float = -1.0, target: float = -16.0) -> dict:
    """loudnorm analysis of a file (input_i, input_tp, input_lra, input_thresh)."""
    path = Path(path).resolve()
    proc = runner.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-vn", "-af",
                       f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"], ro=[path.parent])
    return _json_block(proc.stderr)


def process(runner: Runner, root: Path, project: Project, asset: Asset, chain: list[dict], out_path: Path) -> dict:
    """Run ``chain`` over the asset's audio and write a 48 kHz 24-bit stereo WAV to ``out_path``.

    The source is never modified. The output has exactly the source's number of samples."""
    chain = normalize_chain(chain)
    root = Path(root).resolve()
    src = (root / asset.media).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path == src:
        raise ValueError("refusing to overwrite the source media")
    if not asset.probe.has_audio:
        raise ValueError(f"{asset.id} has no audio stream")
    samples = _input_samples(runner, src)
    mono = asset.probe.channels == 1
    info: dict[str, Any] = {"samples": samples, "sample_rate": SAMPLE_RATE}

    gain = None
    li = next((i for i, s in enumerate(chain) if s["step"] == "loudness"), None)
    if li is not None:
        # pass 1: everything before the loudness step, measured
        pre = filter_string(chain[:li], mono=mono) if li else (HEAD_MONO if mono else HEAD)
        s = chain[li]
        proc = runner.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(src), "-map", "0:a:0", "-af",
             (f"{pre},loudnorm=I={_num(s['target_lufs'])}:TP={_num(s['true_peak_db'])}:LRA={_num(s['lra'])}"
              ":print_format=json"), "-f", "null", "-"],
            ro=[src.parent])
        gain = _json_block(proc.stderr)
        info["measured"] = {k: gain[k] for k in ("input_i", "input_tp", "input_lra", "input_thresh")}
    af = filter_string(chain, samples, gain, mono)
    proc = runner.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(src), "-map", "0:a:0", "-vn", "-af", af,
         "-c:a", "pcm_s24le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-f", "wav", str(out_path)],
        mounts=[out_path.parent], ro=[src.parent])
    if li is not None:
        try:
            info["normalization_type"] = _json_block(proc.stderr).get("normalization_type")
        except (EngineError, ValueError):
            pass
    info["filter"] = af
    return info
