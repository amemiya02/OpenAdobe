"""Audio repair: pure chain->filter tests, plus engine tests that process and measure real audio."""

from __future__ import annotations

import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from openadobe import audio_repair as ar
from openadobe import ops, store
from openadobe.engines import mlt
from openadobe.engines.runner import Runner
from openadobe.timebase import Rate

SR = 48000
runner = Runner()


def engine_only(fn):
    fn = pytest.mark.engine(fn)
    return pytest.mark.skipif(not runner.image_present(), reason="engine image not built")(fn)


needs_say = pytest.mark.skipif(sys.platform != "darwin", reason="speech fixture needs macOS `say`")


# ---- pure -------------------------------------------------------------------------------

def test_describe_lists_steps_and_presets():
    d = ar.describe()
    for step in ("denoise", "dehum", "declick", "deess", "highpass", "voice_eq", "gate", "compress", "level",
                 "loudness"):
        assert step in d["steps"] and d["steps"][step]["description"] and d["steps"][step]["params"]
    assert set(d["presets"]) >= {"podcast_voice", "noisy_interview", "music_master"}
    assert d["steps"]["highpass"]["params"]["freq"]["default"] == 80.0


def test_defaults_are_filled_and_validated():
    chain = ar.normalize_chain([{"step": "denoise"}])
    assert chain[0]["method"] == "arnndn" and chain[0]["model"] == "mp" and chain[0]["mix"] == 1.0
    with pytest.raises(ValueError, match="unknown step"):
        ar.normalize_chain([{"step": "reverb"}])
    with pytest.raises(ValueError, match="unknown params"):
        ar.normalize_chain([{"step": "highpass", "cutoff": 80}])
    with pytest.raises(ValueError, match="out of range"):
        ar.normalize_chain([{"step": "denoise", "mix": 3}])
    with pytest.raises(ValueError, match="expected one of"):
        ar.normalize_chain([{"step": "denoise", "model": "nope"}])
    with pytest.raises(ValueError, match="at most one loudness"):
        ar.normalize_chain([{"step": "loudness"}, {"step": "loudness"}])
    with pytest.raises(ValueError, match="empty"):
        ar.normalize_chain([])


def test_filter_strings():
    f = ar.filter_string([{"step": "highpass", "freq": 90}, {"step": "denoise", "model": "mp", "mix": 0.8}])
    assert f.startswith("aresample=48000,aformat=")
    assert "highpass=f=90:poles=2" in f
    assert "arnndn=m=/opt/openadobe/rnnn/mp.rnnn:mix=0.8" in f
    assert f.index("highpass") < f.index("arnndn")
    assert "afftdn=nr=25:nf=-40," in ar.filter_string([{"step": "denoise", "method": "afftdn"}, {"step": "highpass"}])
    assert ar.filter_string([{"step": "denoise", "passes": 2}]).count("arnndn=") == 2
    assert ":tn=1:tr=1" in ar.filter_string([{"step": "denoise", "method": "afftdn", "track_noise": True}])
    # RNNoise / afftdn latency is compensated: silence padded in, the delay trimmed off again
    f = ar.filter_string([{"step": "denoise", "passes": 2}])
    assert f"apad=pad_len={2 * ar.ARNNDN_DELAY}" in f and f"atrim=start_sample={2 * ar.ARNNDN_DELAY}" in f
    assert "pan=stereo|c0=c0|c1=c0" in ar.filter_string([{"step": "highpass"}], mono=True)
    # length pinned to the source sample count
    assert ar.filter_string([{"step": "highpass"}], samples=96000).endswith("apad=whole_len=96000,atrim=end_sample=96000")


def test_dehum_notches_harmonics_below_nyquist():
    f = ar.filter_string([{"step": "dehum", "freq": 60, "harmonics": 3}])
    assert [m for m in re.findall(r"bandreject=f=(\d+)", f)] == ["60", "120", "180"]
    assert ar.filter_string([{"step": "dehum", "freq": 400, "harmonics": 16}]).count("bandreject") == 16


def test_loudness_second_pass_uses_measurement_linearly():
    m = {"input_i": "-24.1", "input_tp": "-6.0", "input_lra": "3.2", "input_thresh": "-34.5", "target_offset": "0.2"}
    f = ar.filter_string([{"step": "compress"}, {"step": "loudness", "target_lufs": -14, "true_peak_db": -1}], 100, m)
    assert "loudnorm=I=-14:TP=-1:" in f and "measured_I=-24.1" in f and "linear=true" in f
    assert f.index("acompressor") < f.index("loudnorm")
    assert "loudnorm" not in ar.filter_string([{"step": "loudness"}])  # first pass has no loudnorm in the chain


def test_presets_expand_and_validate():
    for name in ar.PRESETS:
        chain = ar.normalize_chain(ar.preset_chain(name))
        assert chain
    assert [s["step"] for s in ar.preset_chain("podcast_voice")] == [
        "highpass", "denoise", "deess", "compress", "level", "loudness"]
    assert ar.normalize_chain(ar.preset_chain("podcast_voice"))[-1]["target_lufs"] == -16
    assert ar.normalize_chain(ar.preset_chain("music_master"))[-1]["target_lufs"] == -14
    with pytest.raises(ValueError, match="unknown preset"):
        ar.preset_chain("nope")


# ---- helpers (engine) -------------------------------------------------------------------

def read_wav(path: Path) -> np.ndarray:
    """-> float array (samples, channels) in -1..1 for 16/24-bit PCM WAV."""
    with wave.open(str(path), "rb") as w:
        n, ch, width = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64) / (1 << 23)
    else:
        raise AssertionError(f"unexpected sample width {width}")
    return x.reshape(-1, ch)


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / np.sqrt(np.mean(x**2))


def db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.square(x)) + 1e-20)


def lag_ms(ref: np.ndarray, out: np.ndarray) -> float:
    a, b = ref[:, 0], out[: len(ref), 0]
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    k = np.concatenate([np.arange(0, 2400), np.arange(n - 2400, n)])  # +-50 ms
    best = int(k[np.argmax(corr[k])])
    best = best - n if best > n // 2 else best
    return best / SR * 1000


def ebur128(path: Path) -> tuple[float, float]:
    proc = runner.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null",
                       "-"], ro=[path.parent])
    summary = proc.stderr[proc.stderr.rindex("Summary:"):]
    return (float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1)),
            float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1)))


NOISE_S = 2.5  # noise-only lead and tail of the speech fixtures


def say_speech(d: Path) -> np.ndarray:
    text = ("The quick brown fox jumps over the lazy dog. Audio repair keeps every edit intact, "
            "so sixty seconds of sibilant sounds, such as this sentence, stay natural.")
    subprocess.run(["say", "-o", str(d / "say.aiff"), text], check=True)
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(d / "say.aiff"), "-ar", str(SR), "-ac", "1",
                str(d / "say.wav")], mounts=[d])
    x = read_wav(d / "say.wav")[:, 0]
    active = x[np.abs(x) > 0.02]
    return x * (10 ** (-20 / 20) / np.sqrt(np.mean(active**2)))


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("audio_fx").resolve()
    rng = np.random.default_rng(7)
    out: dict = {"dir": d}
    if sys.platform == "darwin":
        speech = say_speech(d)
        n_lead = int(NOISE_S * SR)
        body = np.concatenate([np.zeros(n_lead), speech, np.zeros(n_lead)])
        noise = pink(len(body), rng) * 10 ** (-38 / 20)
        write_wav(d / "speech_noise.wav", body + noise)
        t = np.arange(len(body)) / SR
        hum = sum(0.012 * np.sin(2 * np.pi * 50 * k * t + k) for k in range(1, 6))
        write_wav(d / "speech_hum.wav", body + noise * 0.3 + hum)
        white = np.concatenate([np.zeros(n_lead), speech, np.zeros(n_lead)]) + rng.standard_normal(len(body)) * 10 ** (
            -40 / 20)
        write_wav(d / "speech_white.wav", white)
        out |= {"speech_white": d / "speech_white.wav", "speech_noise": d / "speech_noise.wav", "speech_hum": d / "speech_hum.wav", "speech_len": len(speech)}
    # music-like: chords with slow tremolo plus a little noise
    t = np.arange(SR * 8) / SR
    chord = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([220, 277.2, 329.6, 440, 554.4]))
    music = chord * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)) * 0.3 + 0.01 * pink(len(t), rng)
    write_wav(d / "music.wav", music)
    out["music"] = d / "music.wav"
    # hum on its own, for the narrowband measurement
    t = np.arange(SR * 4) / SR
    hum = sum(0.05 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 6))
    write_wav(d / "hum.wav", hum + pink(len(t), rng) * 10 ** (-45 / 20))
    out["hum"] = d / "hum.wav"
    return out


class Env:
    """A project + assets around fixture wavs, for processing through the real ops."""

    def __init__(self, d: Path):
        self.root = d / "proj"
        store.init(self.root, "audio", Rate(num=30, den=1), 640, 360)

    def import_(self, path: Path) -> str:
        return ops.apply(self.root, [("asset.import", {"path": str(path)})], runner=runner)["results"][0]["asset"]

    def process(self, asset: str, **kw) -> str:
        return ops.apply(self.root, [("asset.process", {"asset": asset, **kw})], runner=runner)["results"][0]["asset"]

    def wav(self, asset: str) -> np.ndarray:
        return read_wav(self.root / store.load(self.root).asset(asset).media)


@pytest.fixture(scope="module")
def env(fx) -> Env:
    return Env(fx["dir"])


@pytest.fixture(scope="module")
def assets(env, fx) -> dict[str, str]:
    return {k: env.import_(fx[k]) for k in ("speech_noise", "speech_white", "speech_hum", "music", "hum") if k in fx}


# ---- engine tests -----------------------------------------------------------------------

@needs_say
@pytest.mark.parametrize("which,step_params", [
    ("speech_noise", {"step": "denoise"}),  # RNNoise on pink noise
    ("speech_noise", {"step": "denoise", "passes": 2}),
    ("speech_white", {"step": "denoise", "method": "afftdn"}),  # FFT denoiser on stationary hiss
])
@engine_only
def test_denoise_lowers_noise_floor_keeps_speech(env, fx, assets, which, step_params):
    src = assets[which]
    new = env.process(src, chain=[step_params])
    before, after = env.wav(src)[:, 0], env.wav(new)[:, 0]
    lead = slice(int(0.3 * SR), int(2.2 * SR))
    speech = slice(int(NOISE_S * SR), int(NOISE_S * SR) + fx["speech_len"])
    drop = db(before[lead]) - db(after[lead])
    change = abs(db(after[speech]) - db(before[speech]))
    print(f"\n{step_params}: noise-only drop {drop:.1f} dB, speech change {change:.2f} dB")
    assert drop >= 10 and change < 3


def band_db(x: np.ndarray, f0: float, sr: int = SR, half: float = 1.5) -> float:
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return 10 * np.log10(spec[np.abs(freqs - f0) <= half].sum() + 1e-20)


@engine_only
def test_dehum_drops_hum_energy(env, assets):
    src = assets["hum"]
    new = env.process(src, chain=[{"step": "dehum", "freq": 50, "harmonics": 5}])
    seg = slice(SR, 3 * SR)
    before, after = env.wav(src)[seg, 0], env.wav(new)[seg, 0]
    drops = [float(band_db(before, 50 * k) - band_db(after, 50 * k)) for k in range(1, 6)]
    total = 10 * np.log10(sum(10 ** (band_db(before, 50 * k) / 10) for k in range(1, 6))
                          / sum(10 ** (band_db(after, 50 * k) / 10) for k in range(1, 6)))
    print(f"\ndehum: per-harmonic drop {[round(x, 1) for x in drops]} dB, total {total:.1f} dB")
    assert min(drops) >= 20 and total >= 20


ALIGN_CHAINS = [[{"step": s}] for s in ("denoise", "dehum", "declick", "deess", "highpass", "voice_eq", "gate",
                                        "compress", "level", "loudness")]
ALIGN_CHAINS += [[{"step": "denoise", "method": "afftdn"}], [{"step": "level", "method": "dynaudnorm"}]]
ALIGN_IDS = [c[0]["step"] + ("-" + c[0].get("method", "") if len(c[0]) > 1 else "") for c in ALIGN_CHAINS]


@engine_only
@needs_say
@pytest.mark.parametrize("chain", ALIGN_CHAINS, ids=ALIGN_IDS)
def test_step_stays_sample_aligned(env, assets, chain):
    src = assets["speech_hum"]
    new = env.process(src, chain=chain)
    before, after = env.wav(src), env.wav(new)
    lag = lag_ms(before, after)
    print(f"\n{ALIGN_IDS[ALIGN_CHAINS.index(chain)]}: lag {lag:.3f} ms, samples {len(before)} -> {len(after)}")
    assert len(after) == len(before)
    assert abs(lag) <= 1.0


@engine_only
@needs_say
@pytest.mark.parametrize("preset", list(ar.PRESETS))
def test_preset_stays_sample_aligned(env, assets, preset):
    src = assets["speech_hum"]
    new = env.process(src, preset=preset)
    before, after = env.wav(src), env.wav(new)
    lag = lag_ms(before, after)
    print(f"\n{preset}: lag {lag:.3f} ms, samples {len(before)} -> {len(after)}")
    assert len(after) == len(before)
    assert abs(lag) <= 1.0
    p = store.load(env.root).asset(new)
    assert p.provenance["from"] == src and p.provenance["process"]


@engine_only
def test_alignment_on_music_with_every_step_chained(env, assets):
    """One long chain through every delay-prone step on a broadband signal."""
    src = assets["music"]
    new = env.process(src, chain=[{"step": s} for s in ("dehum", "declick", "gate", "voice_eq", "compress", "level",
                                                      "deess", "highpass")])
    before, after = env.wav(src), env.wav(new)
    assert len(after) == len(before) and abs(lag_ms(before, after)) <= 1.0


@engine_only
@pytest.mark.parametrize("target,tp,preset", [(-16.0, -1.0, None), (-14.0, -1.0, "music_master"), (-23.0, -2.0, None)])
def test_loudness_hits_target_and_true_peak(env, assets, target, tp, preset):
    src = assets["music"]
    kw = {"preset": preset} if preset else {"chain": [{"step": "loudness", "target_lufs": target, "true_peak_db": tp}]}
    new = env.process(src, **kw)
    lufs, peak = ebur128(env.root / store.load(env.root).asset(new).media)
    print(f"\nloudness target {target}/{tp}: got {lufs} LUFS, true peak {peak}")
    assert abs(lufs - target) <= 0.5
    assert peak <= tp + 0.2


@engine_only
@needs_say
def test_podcast_preset_loudness_on_speech(env, assets):
    new = env.process(assets["speech_noise"], preset="podcast_voice")
    lufs, peak = ebur128(env.root / store.load(env.root).asset(new).media)
    print(f"\npodcast_voice: {lufs} LUFS, true peak {peak}")
    assert abs(lufs - (-16)) <= 0.5 and peak <= -0.8


@engine_only
def test_original_is_untouched_and_provenance_recorded(env, assets):
    p = store.load(env.root)
    src = p.asset(assets["music"])
    from openadobe import ingest
    digest_before = ingest.sha256(env.root / src.media)
    new_id = env.process(src.id, chain=[{"step": "highpass"}], name="HP")
    p = store.load(env.root)
    new = p.asset(new_id)
    assert ingest.sha256(env.root / src.media) == digest_before
    assert new.kind == "audio" and new.name == "HP" and new.media.startswith("media/")
    assert new.sha256 == ingest.sha256(env.root / new.media) != digest_before
    assert new.duration_frames == src.duration_frames
    assert new.provenance["from"] == src.id and new.provenance["process"][0]["step"] == "highpass"
    assert new.probe.sample_rate == SR and new.probe.channels == 2 and new.probe.audio_codec == "pcm_s24le"


@engine_only
def test_process_errors(env, assets):
    with pytest.raises(ops.OpError, match="chain or a preset"):
        env.process(assets["music"])
    with pytest.raises(ops.OpError, match="unknown step"):
        env.process(assets["music"], chain=[{"step": "nope"}])


# ---- end to end: video asset, replace_audio, MLT mixdown ---------------------------------

@engine_only
@needs_say
def test_e2e_process_video_audio_replace_and_mixdown(tmp_path, fx):
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                "-i", str(fx["speech_noise"]), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(d / "talk.mp4")], mounts=[d], ro=[fx["dir"]])
    root = d / "proj"
    store.init(root, "e2e", Rate(num=30, den=1), 320, 180)
    vid = ops.apply(root, [("asset.import", {"path": str(d / "talk.mp4")})], runner=runner)["results"][0]["asset"]
    ops.apply(root, [("clip.add", {"asset": vid, "track": "V1"})], runner=runner)
    p = store.load(root)
    video_clip = p.track("V1").clips[0]
    audio_clip = p.track("A1").clips[0]
    assert audio_clip.link == video_clip.link and audio_clip.asset_id == vid

    def mixdown(name: str) -> np.ndarray:
        x = mlt.write(store.load(root), root, name)
        return read_wav(mlt.render(runner, x, root / "build" / f"{name}.wav", root, audio_only=True))

    before = mixdown("before")
    res = ops.apply(root, [("asset.process", {"asset": vid, "preset": "noisy_interview"})], runner=runner)
    new = res["results"][0]["asset"]
    p = store.load(root)
    assert p.asset(new).kind == "audio" and p.asset(new).duration_frames == p.asset(vid).duration_frames
    assert p.asset(new).provenance["from"] == vid and p.asset(vid).kind == "video"

    rep = ops.apply(root, [("clip.replace_audio", {"clip": video_clip.id, "asset": new})], runner=runner)
    assert rep["results"][0]["replaced"] == [{"clip": audio_clip.id, "from": vid, "to": new}]
    p = store.load(root)
    c = p.track("A1").clips[0]
    assert c.asset_id == new and (c.start, c.src_in, c.duration) == (audio_clip.start, audio_clip.src_in,
                                                                    audio_clip.duration)
    assert p.track("V1").clips[0].asset_id == vid  # picture untouched

    after = mixdown("after")
    lead = slice(int(0.3 * SR), int(2.2 * SR))
    drop = db(before[lead]) - db(after[lead])
    print(f"\ne2e mixdown noise floor drop: {drop:.1f} dB; samples {len(before)} vs {len(after)}")
    assert drop >= 10
    assert abs(len(before) - len(after)) <= 1
    assert abs(lag_ms(before, after)) <= 1.0

    # swapping in an asset that is too short is refused, and undo restores the original audio
    short = ops.apply(root, [("asset.import", {"path": str(fx["music"])})], runner=runner)["results"][0]["asset"]
    p = store.load(root)
    p.asset(short)  # music is 8 s; the clip needs ~ the speech length: still long enough or refused, never silent
    if p.asset(short).duration_frames < p.track("A1").clips[0].src_out:
        with pytest.raises(ops.OpError, match="needs source frames"):
            ops.apply(root, [("clip.replace_audio", {"clip": audio_clip.id, "asset": short})], runner=runner)


def test_replace_audio_validation_without_engine(project, root):
    """Pure model checks (no engine): wrong kinds and too-short assets are refused."""
    from tests.conftest import make_asset
    v = make_asset(project, "video", frames=300)
    short = make_asset(project, "audio", frames=100)
    same = make_asset(project, "audio", frames=300)
    silent = make_asset(project, "audio", frames=300, has_audio=False)
    store.save(root, project)
    ops.apply(root, [("clip.add", {"asset": v.id, "track": "V1"})])
    with pytest.raises(ops.OpError, match="needs source frames"):
        ops.apply(root, [("clip.replace_audio", {"clip": "c1", "asset": short.id})])
    with pytest.raises(ops.OpError, match="no audio"):
        ops.apply(root, [("clip.replace_audio", {"clip": "c1", "asset": silent.id})])
    out = ops.apply(root, [("clip.replace_audio", {"clip": "c1", "asset": same.id})])
    assert out["results"][0]["replaced"][0]["to"] == same.id
    assert store.load(root).track("A1").clips[0].asset_id == same.id

