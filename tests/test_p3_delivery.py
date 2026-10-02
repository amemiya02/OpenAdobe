"""P3/P4 delivery: presets, subject-following reframe, letterbox fit, proxies, render-all."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from openadobe import deliver, ops, perceive, reframe, store
from openadobe.engines.runner import Runner
from openadobe.timebase import Rate
from tests.conftest import make_asset

runner = Runner()


def test_fill_geometry_16x9_to_9x16():
    s, w_fit, h_fit = reframe.fill_geometry(1920, 1080, 1080, 1920)
    assert w_fit == 1080 and round(h_fit, 1) == 607.5 and round(s, 3) == 3.160


def test_subject_keyframes_pan_towards_subject(project):
    a = make_asset(project, frames=60, aid="m1")
    a.analysis["faces"] = {"subject": [{"t": 0.0, "cx": 0.8, "cy": 0.5, "size": 0.2},
                                       {"t": 2.0, "cx": 0.8, "cy": 0.5, "size": 0.2}]}
    from openadobe import timeline as tl
    v, _ = tl.add_clip(project, "m1", "V1", 0, 0, None, "A1", insert=False)
    e = reframe.clip_transform(project, v, a, 1080, 1920, "subject")
    xs = list(e.params["x"].values())
    assert all(x < 0 for x in xs[1:])            # pans left to bring the right-side subject to centre
    assert max(abs(x) for x in xs) <= (3.16 * 1080 - 1080) / 2 / 1080 + 1e-6  # never reveals an edge
    assert reframe.clip_transform(project, v, a, 1920, 1080, "subject") is None  # same aspect: nothing to do


def test_preset_op(root):
    res = ops.apply(root, [("deliverable.preset", {"name": "short", "preset": "shorts"})])["results"][0]
    assert (res["width"], res["height"], res["reframe"]) == (1080, 1920, "subject")
    assert res["platform"] == "shorts" and res["preset"] == "medium"  # x264 speed preset untouched
    with pytest.raises(ops.OpError):
        ops.apply(root, [("deliverable.preset", {"name": "x", "preset": "betamax"})])


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_vertical_reframe_follows_subject_and_fit_letterboxes(tmp_path):
    d = tmp_path.resolve()
    # grey 16:9 picture with a white square on the right third (centre x = 0.8)
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x404040:s=640x360:r=25:d=2",
                "-f", "lavfi", "-i", "sine=f=440:sample_rate=48000:d=2", "-vf",
                "drawbox=x=482:y=150:w=60:h=60:color=white:t=fill", "-shortest", "-c:v", "libx264", "-pix_fmt",
                "yuv420p", "-c:a", "aac", str(d / "s.mp4")], mounts=[d])
    root = d / "proj"
    store.init(root, "r", Rate(num=25, den=1), 640, 360)
    ops.apply(root, [("asset.import", {"path": str(d / "s.mp4")})])
    p = store.load(root)
    p.assets["m1"].analysis["faces"] = {"subject": [{"t": 0.0, "cx": 0.8, "cy": 0.5, "size": 0.17}]}
    store.save(root, p)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("deliverable.set", {"name": "short", "settings": {"width": 360, "height": 640,
                                                                       "reframe": "subject", "loudness_lufs": None}}),
                     ("deliverable.set", {"name": "box", "settings": {"width": 360, "height": 640, "reframe": "fit",
                                                                     "loudness_lufs": None}})])
    short = deliver.render(root, "short", runner=runner, run_qc=False)
    box = deliver.render(root, "box", runner=runner, run_qc=False)
    assert short["reframed_clips"] == 1 and box["reframed_clips"] == 0

    def px(path: str, x: int, y: int) -> tuple[int, ...]:
        proc = subprocess.run(runner.command(["ffmpeg", "-v", "error", "-ss", "1", "-i", path, "-vf",
                                              f"format=rgb24,crop=1:1:{x}:{y}", "-frames:v", "1", "-f", "rawvideo",
                                              "-"], mounts=[root]), capture_output=True, check=False)
        return tuple(proc.stdout[:3])

    assert min(px(short["render"], 180, 320)) > 200      # the white subject is now centre frame
    assert max(px(box["render"], 180, 30)) < 20          # fit: black letterbox bar at the top
    assert 50 < px(box["render"], 100, 320)[0] < 90       # fit: grey picture in the middle band


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_proxies_used_for_previews(tmp_path):
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "t.mp4")], mounts=[d])
    root = d / "proj"
    store.init(root, "p", Rate(num=25, den=1), 1280, 720)
    ops.apply(root, [("asset.import", {"path": str(d / "t.mp4")})])
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}), ("asset.make_proxy", {})])
    p = store.load(root)
    assert p.assets["m1"].proxy and (root / p.assets["m1"].proxy).exists()
    res = deliver.render(root, preview=True, runner=runner)
    assert "proxies/m1.mp4" in (root / "build" / f"main-r{p.revision:04d}-preview.mlt").read_text()
    assert res["size"] == "640x360"


def test_beat_montage_cuts_on_beats(root):
    p = store.load(root)
    make_asset(p, frames=300, aid="m1")
    make_asset(p, frames=300, aid="m2")
    mus = make_asset(p, kind="audio", frames=240, aid="m3")
    mus.analysis["beats"] = {"tempo": 120.0, "beats": [i * 0.5 for i in range(1, 16)],
                             "downbeats": [i * 2.0 for i in range(1, 4)]}
    p.assets["m1"].analysis["shots"] = [{"start": 0, "end": 100, "score": 1}, {"start": 100, "end": 300, "score": 1}]
    store.save(root, p)
    res = ops.apply(root, [("edit.montage", {"clips": ["m1", "m2"], "music": "m3", "every": 2})])["results"][0]
    p = store.load(root)
    starts = sorted(c.start for c in p.track("V1").clips)
    assert starts == [0, 30, 60, 90, 120, 150, 180, 210]          # every 2 beats at 120 bpm, 30 fps
    assert res["music_clip"] and p.track("A2").clips[0].duration == 240
    assert not p.track("A1").clips                                   # b-roll is silent
    m1_ins = sorted(c.src_in for c in p.track("V1").clips if c.asset_id == "m1")
    assert len(set(m1_ins)) > 1                                      # different shots used
    with pytest.raises(ops.OpError, match="no beats"):
        ops.apply(root, [("edit.montage", {"clips": ["m1"], "music": "m2"})])


def test_vlm_request_contains_images_and_rubric(tmp_path):
    from openadobe import review
    (tmp_path / "s.jpg").write_bytes(b"\xff\xd8jpg")
    (tmp_path / "w.png").write_bytes(b"\x89PNGpng")
    body = review.vlm_request({"contact_sheet": str(tmp_path / "s.jpg"), "waveform": str(tmp_path / "w.png"),
                               "qc": {}, "loudness": None, "captions": [], "timeline": {},
                               "rubric": review.RUBRIC}, "claude-sonnet-5-5")
    kinds = [c["type"] for c in body["messages"][0]["content"]]
    assert kinds == ["image", "image", "text"] and "senior video editor" in body["messages"][0]["content"][2]["text"]


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_review_package(tmp_path):
    from openadobe import review
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=3",
                "-f", "lavfi", "-i", "sine=f=440:sample_rate=48000:d=3", "-shortest", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "t.mp4")], mounts=[d])
    root = d / "proj"
    store.init(root, "rv", Rate(num=25, den=1), 640, 360)
    ops.apply(root, [("asset.import", {"path": str(d / "t.mp4")})])
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("captions.add", {"start": "0.4s", "end": "2s", "text": "测试"})])
    deliver.render(root, runner=runner)
    pkg = review.build(runner, root)
    assert pkg["qc"]["pass"] and pkg["captions"] == ["00:00:00.400 测试"]
    assert (root / "perceive").joinpath(pkg["package"]).exists()


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_delivery_with_ardour_mix(tmp_path):
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=6",
                "-f", "lavfi", "-i", "aevalsrc='0.3*sin(2*PI*330*t)*lt(mod(t,3),2)':s=48000:d=6", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "t.mp4")], mounts=[d])
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:sample_rate=48000:d=6",
                str(d / "m.wav")], mounts=[d])
    root = d / "proj"
    store.init(root, "ar", Rate(num=25, den=1), 640, 360)
    ops.apply(root, [("asset.import", {"path": str(d / "t.mp4")}), ("asset.import", {"path": str(d / "m.wav")})])
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("clip.add", {"asset": "m2", "track": "A2", "start": 0}),
                     ("mix.ducking", {"depth_db": -12}),
                     ("mix.engine", {"engine": "ardour"})])
    res = deliver.render(root, runner=runner)
    assert res["mix_engine"] == "ardour" and res["qc"]["pass"], res
    assert "mixdown" in (root / "build" / f"main-r{store.load(root).revision:04d}.mlt").read_text()


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_ardour_master_fader_applied_once(tmp_path):
    """Without a loudness pass the master fader must land the same in an Ardour mix as in an MLT mix."""
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=25:d=4",
                "-f", "lavfi", "-i", "sine=f=440:sample_rate=48000:d=4", "-shortest", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "t.mp4")], mounts=[d])
    root = d / "proj"
    store.init(root, "g", Rate(num=25, den=1), 320, 180)
    ops.apply(root, [("asset.import", {"path": str(d / "t.mp4")})])
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}), ("mix.master", {"gain_db": -6}),
                     ("deliverable.set", {"settings": {"loudness_lufs": None}})])
    lufs = {}
    for engine in ("mlt", "ardour"):
        ops.apply(root, [("mix.engine", {"engine": engine})])
        res = deliver.render(root, runner=runner, run_qc=False)
        assert res["mix_engine"] == engine
        lufs[engine] = perceive.loudness(runner, root, Path(res["render"]))["integrated_lufs"]
    assert abs(lufs["mlt"] - lufs["ardour"]) < 1.0, lufs
