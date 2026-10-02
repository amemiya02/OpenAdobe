"""P2: comps on the timeline (Dynamic Link) — Lottie graphics rendered natively by MLT."""

from __future__ import annotations

import subprocess

import pytest

from openadobe import comps, deliver, ops, store
from openadobe.engines import mlt
from openadobe.engines.runner import Runner
from openadobe.timebase import Rate
from tests.conftest import make_asset

runner = Runner()


def test_lottie_comp_add_and_dynamic_link(root):
    p = store.load(root)
    make_asset(p, frames=150, aid="m1")
    store.save(root, p)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"})])
    res = ops.apply(root, [("comp.add", {"engine": "lottie", "template": "lower_third", "start": "1s",
                                         "duration": "3s", "params": {"name": "张三", "role": "导演"}})])["results"][0]
    p = store.load(root)
    asset = p.assets[res["asset"]]
    assert asset.kind == "comp" and asset.duration_frames == 90 and (root / asset.media).exists()
    assert p.track(res["track"]).role == "overlay"
    first_media = asset.media
    ops.apply(root, [("comp.set", {"comp": res["comp"], "params": {"name": "李四"}})])
    p = store.load(root)
    assert p.assets[res["asset"]].media != first_media  # new content → new cache entry, clip follows
    assert '"李四"' in (root / p.assets[res["asset"]].media).read_text("utf-8") or "李四" in \
        (root / p.assets[res["asset"]].media).read_text("utf-8")
    xml = mlt.compile_project(p, root)
    assert "glaxnimate" in xml
    with pytest.raises(ops.OpError):
        ops.apply(root, [("comp.add", {"engine": "lottie", "template": "nope"})])


def test_render_key_tracks_inputs(project):
    from openadobe.model import Comp
    make_asset(project, frames=30, aid="m1")
    c = Comp(id="k1", engine="blender", template="glow", inputs={"clip": "m1"}, duration=30)
    k1 = comps.render_key(project, c)
    project.assets["m1"].sha256 = "changed"
    assert comps.render_key(project, c) != k1


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_lottie_title_renders_over_video(tmp_path):
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x204080:s=640x360:r=25:d=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "bg.mp4")], mounts=[d])
    root = d / "proj"
    store.init(root, "c", Rate(num=25, den=1), 640, 360)
    ops.apply(root, [("asset.import", {"path": str(d / "bg.mp4")})])
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("comp.add", {"engine": "lottie", "template": "title", "start": 0, "duration": "3s",
                                   "params": {"text": "开场标题"}})])
    res = deliver.render(root, "main", preview=False, runner=runner, run_qc=False)
    out = res["render"]

    def mean_luma(n: int) -> float:
        proc = subprocess.run(runner.command(["ffmpeg", "-hide_banner", "-i", out, "-vf",
                                              f"select=eq(n\\,{n}),signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                                              "-f", "null", "-"], mounts=[root]), capture_output=True, text=True,
                              check=False)
        return float(proc.stderr.split("YAVG=")[-1].split()[0])

    plain = mean_luma(90)   # after the title ends: background only
    titled = mean_luma(37)  # title on screen
    assert titled > plain + 2, (titled, plain)


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_blender_key_comp_delivers_over_plate(tmp_path):
    """Dynamic Link through Blender: the keyed comp is rendered into the cache, composited by MLT, then reused."""
    d = tmp_path.resolve()
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x2040c0:s=320x180:r=25:d=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "plate.mp4")], mounts=[d])
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x00ff00:s=320x180:r=25:d=2",
                "-vf", "drawbox=x=120:y=60:w=80:h=60:color=0xff0000:t=fill", "-c:v", "libx264", "-pix_fmt", "yuv444p",
                "-crf", "4", str(d / "green.mp4")], mounts=[d])
    root = d / "proj"
    store.init(root, "k", Rate(num=25, den=1), 320, 180)
    ops.apply(root, [("asset.import", {"path": str(d / "plate.mp4")}),
                     ("asset.import", {"path": str(d / "green.mp4")})])
    res = ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                           ("comp.add", {"engine": "blender", "template": "keying", "start": 0, "duration": "1s",
                                         "inputs": {"fg": "m2"}, "params": {"key_color": "#00FF00"}})])
    comp_id = res["results"][1]["comp"]
    assert comps.stale(store.load(root)), "blender comp renders lazily"
    out = deliver.render(root, "main", preview=False, runner=runner, run_qc=False)
    p = store.load(root)
    assert not comps.stale(p) and p.assets[res["results"][1]["asset"]].media.startswith("cache/")

    def rgb(x: int, y: int, n: int = 12) -> tuple[int, ...]:
        raw = subprocess.run(runner.command(["ffmpeg", "-v", "error", "-i", out["render"], "-vf",
                                             f"select=eq(n\\,{n}),format=rgb24,crop=1:1:{x}:{y}", "-frames:v", "1",
                                             "-f", "rawvideo", "-"], mounts=[root]), capture_output=True,
                             check=True).stdout
        return tuple(raw[:3])

    r, g, b = rgb(20, 20)    # keyed green → the plate shows through
    assert b > 150 and g < 110, (r, g, b)
    r, g, b = rgb(160, 90)   # the red box survives the key
    assert r > 180 and g < 80 and b < 80, (r, g, b)
    again = comps.ensure_rendered(runner, root, p)
    assert again == [] and comps.render_key(p, p.comps[comp_id]) == p.comps[comp_id].render_key
