"""Media analysis (shots / beats / faces) and person matting."""

from __future__ import annotations

import json
import urllib.request
from itertools import pairwise
from pathlib import Path

import pytest

from openadobe import analysis, ingest, matting, ops, store
from openadobe.engines.runner import Runner
from openadobe.timebase import Rate

TESTDATA = Path.home() / ".cache" / "openadobe" / "testdata"
BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"


# ---- pure helpers ----------------------------------------------------------------------

def test_parse_scene_metadata_and_shots():
    text = ("frame:0    pts:60      pts_time:2.4\nlavfi.scene_score=0.512\n"
            "frame:1    pts:125     pts_time:5\nlavfi.scene_score=0.9\n")
    cuts = analysis.parse_scene_metadata(text, 25)
    assert cuts == [(60, 0.512), (125, 0.9)]
    s = analysis.shots_from_cuts(cuts, 200)
    assert [(x["start"], x["end"]) for x in s] == [(0, 60), (60, 125), (125, 200)]
    assert analysis.shots_from_cuts([], 50) == [{"start": 0, "end": 50, "score": 0.0}]
    assert analysis.shots_from_cuts([(0, 1.0), (300, 1.0)], 50)[0]["end"] == 50  # out-of-range cuts dropped


def test_pick_downbeats_uses_strongest_phase():
    beats = [i * 0.5 for i in range(12)]
    strength = [1, 1, 5, 1] * 3
    assert analysis.pick_downbeats(beats, strength) == [1.0, 3.0, 5.0]
    assert analysis.pick_downbeats([], []) == []


def test_normalise_boxes():
    assert analysis.normalise_boxes([(50, 20, 100, 40, 0.9)], 200, 100) == [[0.25, 0.2, 0.5, 0.4, 0.9]]


def _sample(t, *boxes):
    return {"t": t, "boxes": [list(b) for b in boxes]}


def test_subject_holds_last_position_and_centres_when_never_seen():
    face = (0.4, 0.3, 0.2, 0.2, 0.9)  # centre (0.5, 0.4)
    track = analysis.track_subject([_sample(0, face), _sample(0.2), _sample(0.4)])
    assert [(p["cx"], p["cy"]) for p in track] == [(0.5, 0.4)] * 3
    nothing = analysis.track_subject([_sample(0), _sample(0.2)])
    assert [(p["cx"], p["cy"], p["size"]) for p in nothing] == [(0.5, 0.5, 0.0)] * 2


def test_subject_backfills_leading_gap_and_smooths():
    a, b = (0.1, 0.1, 0.2, 0.2, 0.9), (0.5, 0.1, 0.2, 0.2, 0.9)
    track = analysis.track_subject([_sample(0), _sample(0.2, a), _sample(0.4, b)], alpha=0.5)
    assert track[0]["cx"] == track[1]["cx"] == 0.2
    assert 0.2 < track[2]["cx"] < 0.6 and track[2]["cx"] == pytest.approx(0.4)


def test_subject_prefers_persistent_face_over_stray():
    main = (0.45, 0.35, 0.2, 0.2, 0.9)
    stray = (0.0, 0.0, 0.05, 0.05, 0.95)
    track = analysis.track_subject([_sample(0, main), _sample(0.2, stray, main), _sample(0.4, stray, main)])
    assert all(abs(p["cx"] - 0.55) < 0.01 for p in track)


def test_matting_geometry_helpers():
    assert matting.downsample_ratio(1920, 1080) == 0.2667
    assert matting.downsample_ratio(320, 240) == 1.0
    assert matting.working_size(1920, 1080, 768) == (768, 432)
    assert matting.working_size(1080, 1920, 768) == (432, 768)
    assert matting.working_size(641, 481, 768) == (640, 480)


def test_fetch_model_rejects_wrong_hash(tmp_path, monkeypatch):
    src = tmp_path / "x.bin"
    src.write_bytes(b"not a model")
    monkeypatch.setenv("OPENADOBE_MODEL_DIR", str(tmp_path / "models"))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        analysis.fetch_model("x.bin", src.as_uri(), "0" * 64)
    assert not list((tmp_path / "models").glob("x.bin*"))


# ---- engine tests ----------------------------------------------------------------------

runner = Runner()
engine = [pytest.mark.engine, pytest.mark.skipif(not runner.image_present(), reason="engine image not built")]


def needs_engine(fn):
    for m in engine:
        fn = m(fn)
    return fn


def ffmpeg(*args: str, cwd: Path) -> None:
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], mounts=[cwd])


def fetch_testdata(name: str) -> Path:
    path = TESTDATA / name
    if not path.exists() or path.stat().st_size < 1000:
        TESTDATA.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(BASE + name, timeout=30) as resp:
                path.write_bytes(resp.read())
        except OSError as exc:
            pytest.skip(f"offline, cannot fetch {name}: {exc}")
    return path


def new_project(tmp_path, w=640, h=360, fps=25) -> Path:
    root = tmp_path.resolve() / "proj"
    store.init(root, "an", Rate(num=fps, den=1), w, h)
    return root


def import_asset(root: Path, path: Path):
    res = ops.apply(root, [("asset.import", {"path": str(path)})])
    return res["results"][0]["asset"]


@needs_engine
def test_shots_boundaries_within_one_frame(tmp_path):
    d = tmp_path.resolve()
    # three visually different segments: 1.2 s, 2.0 s, 1.6 s at 25 fps (30, 50, 40 frames)
    for name, src, dur in (("a", "color=c=red:s=320x180:r=25", 1.2), ("b", "testsrc2=s=320x180:r=25", 2.0),
                           ("c", "mandelbrot=s=320x180:r=25", 1.6)):
        ffmpeg("-f", "lavfi", "-i", src, "-t", str(dur), "-pix_fmt", "yuv420p", "-c:v", "libx264", str(d / f"{name}.mp4"),
               cwd=d)
    (d / "list.txt").write_text("".join(f"file '{n}.mp4'\n" for n in "abc"))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(d / "list.txt"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(d / "cat.mp4"), cwd=d)
    root = new_project(d / "p", 320, 180)
    asset = import_asset(root, d / "cat.mp4")
    res = ops.apply(root, [("asset.analyze", {"asset": asset, "what": ["shots"]})])["results"][0]
    assert res["shots"] == 3
    shots = store.load(root).asset(asset).analysis["shots"]
    assert shots[0]["start"] == 0 and shots[-1]["end"] == 120
    assert abs(shots[1]["start"] - 30) <= 1 and abs(shots[2]["start"] - 80) <= 1
    assert all(s["end"] == n["start"] for s, n in pairwise(shots))


@needs_engine
def test_beats_click_track_120bpm(tmp_path):
    d = tmp_path.resolve()
    # 10 s of 120 bpm clicks (every 0.5 s, 30 ms 1 kHz bursts, accent on every 4th)
    expr = ("aevalsrc='sin(2*PI*1000*t)*exp(-60*mod(t,0.5))*(0.5+0.5*lt(mod(t,2),0.5))':s=44100:d=10")
    ffmpeg("-f", "lavfi", "-i", expr, "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=10",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "click.mp4"), cwd=d)
    root = new_project(d / "p")
    asset = import_asset(root, d / "click.mp4")
    res = ops.apply(root, [("asset.analyze", {"asset": asset, "what": ["beats"]})])["results"][0]
    assert abs(res["tempo"] - 120) <= 2
    b = store.load(root).asset(asset).analysis["beats"]
    assert b["onsets"] and b["downbeats"]
    # every detected beat lies within 40 ms of the 0.5 s grid (grid offset taken from the first beat)
    off = b["beats"][0] % 0.5
    assert all(min((t - off) % 0.5, 0.5 - (t - off) % 0.5) < 0.04 for t in b["beats"])
    assert len(b["beats"]) >= 17
    # the first beat sits on a real click (clicks are at multiples of 0.5 s)
    assert min(off, 0.5 - off) < 0.04
    assert all(abs(((t - b["downbeats"][0]) / 2.0) - round((t - b["downbeats"][0]) / 2.0)) < 0.05 for t in b["downbeats"])


def make_person_clip(d: Path, name: str, image: str, seconds: float, fps: int = 25, size=(640, 480)) -> Path:
    img = d / image
    img.write_bytes(fetch_testdata(image).read_bytes())  # the container only sees the mounted scratch dir
    out = d / name
    w, h = size
    ffmpeg("-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", str(img), "-f", "lavfi", "-i",
           f"sine=f=440:d={seconds}", "-vf",
           f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=yuv420p",
           "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out), cwd=d.resolve())
    return out


@needs_engine
def test_faces_found_and_subject_track(tmp_path):
    d = tmp_path.resolve()
    clip = make_person_clip(d, "lena.mp4", "lena.jpg", 2.0, size=(512, 512))
    root = new_project(d / "p", 512, 512)
    asset = import_asset(root, clip)
    res = ops.apply(root, [("asset.analyze", {"asset": asset, "what": ["faces"]})])["results"][0]
    assert res["face_coverage_pct"] >= 80
    f = store.load(root).asset(asset).analysis["faces"]
    assert len(f["samples"]) >= 9 and len(f["subject"]) == len(f["samples"])
    s = f["subject"][-1]
    # lena's face sits in the upper-middle of the frame (x ~ 0.5, y ~ 0.45)
    assert 0.35 < s["cx"] < 0.65 and 0.25 < s["cy"] < 0.6 and 0.15 < s["size"] < 0.6
    box = max(f["samples"][0]["boxes"], key=lambda b: b[2])
    assert all(0 <= v <= 1 for v in box[:4])


@needs_engine
def test_matte_creates_overlay_ready_alpha_asset(tmp_path):
    d = tmp_path.resolve()
    clip = make_person_clip(d, "messi.mp4", "messi5.jpg", 1.0, size=(640, 480))
    root = new_project(d / "p", 640, 480)
    src = import_asset(root, clip)
    res = ops.apply(root, [("asset.matte", {"asset": src, "name": "cutout"})])["results"][0]
    p = store.load(root)
    m = p.asset(res["asset"])
    assert m.id != src and m.kind == "video" and m.name == "cutout"
    assert m.provenance == {"from": src, "process": [{"op": "matte", "model": "rvm_mobilenetv3"}]}
    assert m.duration_frames == p.asset(src).duration_frames
    assert m.media.startswith("media/") and m.sha256 == ingest.sha256(root / m.media)
    assert m.probe.has_audio and m.probe.video_codec == "ffv1"
    out = runner.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                      "-of", "json", str(root / m.media)], ro=[root])
    assert json.loads(out.stdout)["streams"][0]["pix_fmt"] == "bgra"  # MLT keeps this alpha (not ProRes 4444's)

    # alpha is real: person opaque, background transparent
    png = d / "frame.png"
    ffmpeg("-i", str(root / m.media), "-vf", "select=eq(n\\,12)", "-frames:v", "1", "-pix_fmt", "rgba", str(png), cwd=d)
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert rgba.shape[2] == 4
    alpha = rgba[:, :, 3]
    assert (alpha < 25).mean() > 0.3, "expected a transparent background"
    assert (alpha > 230).mean() > 0.02, "expected opaque person pixels"
    assert np.unique(alpha).size > 10  # soft edge, not a binary mask

    # placeable over V1 on an overlay track and composites through the engine
    ops.apply(root, [("clip.add", {"asset": src, "track": "V1", "link_audio": False}),
                     ("track.add", {"kind": "video", "role": "overlay"})])
    ops.apply(root, [("clip.add", {"asset": m.id, "track": "V2", "start": 0, "link_audio": False})])
    clips = store.load(root).track("V2").clips
    assert len(clips) == 1 and clips[0].asset_id == m.id
