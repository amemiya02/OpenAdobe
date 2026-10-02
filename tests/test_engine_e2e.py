"""End-to-end through the real engines (needs the openadobe/engines image; `openadobe doctor --build`)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from openadobe import deliver, ops, perceive, store
from openadobe.engines.runner import Runner
from openadobe.timebase import Rate

runner = Runner()
pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(not runner.image_present(), reason="engine image not built"),
]


def ffmpeg(*args: str, cwd: Path) -> None:
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], mounts=[cwd])


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("fixtures").resolve()
    # "talking head": 8 s of picture, speech-like tone bursts with pauses (on 0–1.5, 2.5–4, 5–8)
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25", "-f", "lavfi", "-i",
           "aevalsrc='0.25*sin(2*PI*330*t)*(lt(t,1.5)+between(t,2.5,4)+gte(t,5))':s=48000:d=8",
           "-t", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "talk.mp4"), cwd=d)
    ffmpeg("-f", "lavfi", "-i", "aevalsrc='0.3*sin(2*PI*220*t)':s=44100:d=12", str(d / "bed.wav"), cwd=d)
    return {"talk": d / "talk.mp4", "bed": d / "bed.wav"}


@pytest.fixture(scope="module")
def built(media, tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e").resolve() / "proj"
    store.init(root, "e2e", Rate(num=30, den=1), 1280, 720)
    imp = ops.apply(root, [("asset.import", {"path": str(media["talk"])}),
                           ("asset.import", {"path": str(media["bed"])})])
    talk, bed = (r["asset"] for r in imp["results"])
    ops.apply(root, [("clip.add", {"asset": talk, "track": "V1"})])
    cut = ops.apply(root, [("edit.cut_silence", {"clip": "c1"})])["results"][0]
    ops.apply(root, [
        ("clip.add", {"asset": bed, "track": "A2", "start": 0}),
        ("mix.ducking", {"depth_db": -15}),
        ("captions.add", {"start": "0.2s", "end": "1.4s", "text": "开场字幕 Opening"}),
        ("deliverable.set", {"settings": {"loudness_lufs": -16}}),
    ])
    p = store.load(root)
    first_v = min(p.track("V1").clips, key=lambda c: c.start)
    ops.apply(root, [("clip.set", {"clip": first_v.id, "fade_in": "0.5s"})])
    return root, cut


def test_ingest_conforms_to_cfr_project_rate(built):
    root, _ = built
    p = store.load(root)
    talk = next(a for a in p.assets.values() if a.kind == "video")
    assert talk.probe.fps == "30/1" and talk.duration_frames == 240
    bed = next(a for a in p.assets.values() if a.kind == "audio")
    assert bed.probe.sample_rate == 48000


def test_silence_cut_removes_pauses(built):
    root, cut = built
    assert cut["silences"] == 2
    # two ~1 s pauses minus 0.12 s breaths each side ≈ 1.52 s
    assert 40 <= cut["removed_frames"] <= 50
    p = store.load(root)
    assert len(p.track("V1").clips) == 3 and len(p.track("A1").clips) == 3


def test_render_meets_loudness_and_passes_qc(built):
    root, _ = built
    res = deliver.render(root, "main", runner=runner)
    assert res["qc"]["pass"], json.dumps(json.loads(Path(res["qc_report"]).read_text()), indent=1)
    loud = perceive.loudness(runner, root, res["render"])
    assert abs(loud["integrated_lufs"] - (-16)) <= 1.0
    assert loud["true_peak_db"] <= -0.5
    assert Path(res["srt"]).read_text("utf-8").count("-->") == 1


def test_fade_in_starts_dark_and_perception_images(built):
    root, _ = built
    dark = perceive.frame(runner, root, 1)["image"]
    lit = perceive.frame(runner, root, 30)["image"]

    def luma(path: str) -> float:
        proc = subprocess.run(
            runner.command(["ffmpeg", "-hide_banner", "-i", path, "-vf",
                            "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"],
                           mounts=[root]), capture_output=True, text=True, check=False)
        return float(proc.stderr.split("YAVG=")[-1].split()[0])

    assert luma(dark) < 30 < luma(lit)
    sheet = perceive.contact_sheet(runner, root, n=8)
    assert Path(sheet["image"]).stat().st_size > 10_000


def _window_lufs(root: Path, wav: Path, start: float, end: float) -> float:
    proc = subprocess.run(runner.command(
        ["ffmpeg", "-hide_banner", "-i", str(wav), "-af", f"atrim={start}:{end},astats=metadata=1:reset=0",
         "-f", "null", "-"], mounts=[root]), capture_output=True, text=True, check=False)
    return float(proc.stderr.split("RMS level dB:")[-1].split()[0])


def test_ducking_lowers_music_under_dialogue(built):
    root, _ = built
    p = store.load(root)
    dia_end = p.rate.to_seconds(max(c.end for c in p.track("A1").clips))
    # dialogue fader at -60 dB: still triggers ducking, but the mix is ~only the music bed
    ops.apply(root, [("track.set", {"track": "A1", "gain_db": -60})])
    try:
        x = deliver.mlt.write(store.load(root), root, "duckcheck")
        wav = deliver.mlt.render(runner, x, root / "build" / "duckcheck.wav", root, audio_only=True)
    finally:
        store.undo(root)
    under = _window_lufs(root, wav, 1.0, dia_end - 0.5)
    after = _window_lufs(root, wav, dia_end + 1.0, dia_end + 2.5)
    assert 12 <= after - under <= 17  # depth -15 dB
