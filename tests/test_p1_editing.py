"""P1: transitions, speed, effects — model logic, compile structure and real renders."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openadobe import ops, store
from openadobe import timeline as tl
from openadobe.engines import mlt
from openadobe.engines.runner import Runner
from openadobe.lint import errors, lint
from openadobe.timebase import Rate
from tests.conftest import make_asset


def two_clips(project, a_frames=100, b_frames=100, a_in=0, a_dur=60, b_in=20, b_dur=60):
    a = make_asset(project, frames=a_frames)
    b = make_asset(project, frames=b_frames)
    va, _ = tl.add_clip(project, a.id, "V1", 0, a_in, a_dur, "A1", insert=False)
    vb, _ = tl.add_clip(project, b.id, "V1", None, b_in, b_dur, "A1", insert=False)
    return va, vb


# ---- model ---------------------------------------------------------------------------

def test_transition_add_links_audio_crossfade(project):
    va, _ = two_clips(project)
    made = tl.add_transition(project, "V1", va.id, 20, "dissolve", audio=True)
    assert len(made) == 2
    assert project.track("A1").transitions[0].kind == "crossfade"
    assert not errors(lint(project))


def test_transition_needs_handles_but_dip_does_not(project):
    va, _ = two_clips(project, a_frames=60, b_in=0)  # no media after a, none before b
    tl.add_transition(project, "V1", va.id, 20, "dissolve", audio=False)
    codes = {i["code"] for i in errors(lint(project))}
    assert codes == {"transition.handles"}
    project.track("V1").transitions[0].kind = "dip_black"
    assert not errors(lint(project))


def test_prune_drops_transitions_on_separated_clips(project):
    va, vb = two_clips(project)
    tl.add_transition(project, "V1", va.id, 20, "dissolve", audio=True)
    tl.move(project, vb.id, 200, None)
    assert sorted(tl.prune_transitions(project)) == ["T1", "T2"]


def test_speed_halves_duration_and_keeps_source(project):
    a = make_asset(project, frames=120)
    v, au = tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
    after = make_asset(project, frames=30)
    nxt, _ = tl.add_clip(project, after.id, "V1", None, 0, None, "A1", insert=False)
    tl.set_speed(project, v.id, 2.0, ripple=True)
    assert (v.duration, au.duration, v.src_out) == (60, 60, 120)
    assert nxt.start == 60
    right = tl.split(project, v.id, 30)
    r = project.find_clip(right[0])[1]
    assert (r.src_in, r.src_len) == (60, 60)
    with pytest.raises(tl.EditError):
        tl.remove_source_ranges(project, v.id, [(0, 10)], ripple=True)


def test_effect_validation_and_ops(root):
    p = store.load(root)
    make_asset(p, frames=90, aid="m1")
    store.save(root, p)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"})])
    res = ops.apply(root, [("effect.add", {"clip": "c1", "kind": "transform",
                                           "params": {"scale": {"0s": 1.0, "3s": 1.2}}})])
    assert res["results"][0]["params"]["scale"] == {0: 1.0, 90: 1.2}
    with pytest.raises(ops.OpError, match="audio effect"):
        ops.apply(root, [("effect.add", {"clip": "c1", "kind": "audio.highpass"})])
    with pytest.raises(ops.OpError, match="outside"):
        ops.apply(root, [("effect.add", {"clip": "c1", "kind": "color.adjust", "params": {"saturation": 9}})])
    with pytest.raises(ops.OpError, match="cannot be keyframed"):
        ops.apply(root, [("effect.add", {"clip": "c1", "kind": "crop", "params": {"left": {"0": 0.1}}})])
    ops.apply(root, [("effect.add", {"clip": "c2", "kind": "audio.highpass", "params": {"freq": 120}}),
                     ("effect.add", {"scope": "track", "track": "A1", "kind": "audio.compressor"}),
                     ("effect.add", {"scope": "master", "kind": "audio.limiter"})])
    xml = mlt.compile_project(store.load(root), root)
    assert "avfilter.highpass" in xml and "avfilter.acompressor" in xml and "qtblend" in xml


def test_compile_transition_structure(project, root):
    va, vb = two_clips(project)
    tl.add_transition(project, "V1", va.id, 20, "wipe_left", audio=True)
    doc = ET.fromstring(mlt.compile_project(project, root).split("\n", 1)[1])
    pl = doc.find("playlist[@id='pl_V1']")
    entries = [(e.get("producer"), e.get("in"), e.get("out")) for e in pl.findall("entry")]
    # a visible 0..49, transition 50..69 (a 50..69 handle-in, b 10..29), b visible from its frame 30
    assert entries == [(f"p_{va.id}", "0", "49"), ("tr_T1", "0", "19"), (f"p_{vb.id}", "30", "79")]
    tr_a = doc.find("playlist[@id='tr_T1_a']/entry")
    tr_b = doc.find("playlist[@id='tr_T1_b']/entry")
    assert (tr_a.get("in"), tr_a.get("out"), tr_b.get("in"), tr_b.get("out")) == ("50", "69", "10", "29")
    assert doc.find("tractor[@id='tr_T1']/transition/property[@name='resource']").text.endswith(".pgm")
    assert doc.find("tractor[@id='tr_T2']/transition/property[@name='mlt_service']").text == "mix"


# ---- engine --------------------------------------------------------------------------

runner = Runner()
engine = pytest.mark.skipif(not runner.image_present(), reason="engine image not built")


def ff(*args: str, cwd: Path) -> None:
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], mounts=[cwd])


def stats(path: Path, root: Path, vf: str, key: str) -> list[float]:
    proc = subprocess.run(runner.command(["ffmpeg", "-hide_banner", "-i", str(path), "-vf",
                                          f"{vf},metadata=print:key={key}", "-f", "null", "-"], mounts=[root]),
                          capture_output=True, text=True, check=False)
    return [float(line.split("=")[-1]) for line in proc.stderr.splitlines() if key in line]


@pytest.fixture(scope="module")
def colors(tmp_path_factory):
    """A red and a blue clip (4 s each, tone audio) plus a project with both imported."""
    d = tmp_path_factory.mktemp("p1").resolve()
    for name, color, freq in (("red", "red", 300), ("blue", "blue", 600)):
        ff("-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=25:d=4", "-f", "lavfi", "-i",
           f"sine=f={freq}:sample_rate=48000:d=4", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(d / f"{name}.mp4"), cwd=d)
    root = d / "proj"
    store.init(root, "p1", Rate(num=25, den=1), 320, 180)
    ops.apply(root, [("asset.import", {"path": str(d / "red.mp4")}), ("asset.import", {"path": str(d / "blue.mp4")})])
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1", "duration": 50}),
                     ("clip.add", {"asset": "m2", "track": "V1", "src_in": 25, "duration": 50})])
    return root


def render_frames(root: Path, name: str) -> Path:
    x = mlt.write(store.load(root), root, name)
    return mlt.render(runner, x, root / "renders" / f"{name}.mkv", root, vcodec="ffv1", acodec="pcm_s16le")


def frame_rgb(root: Path, video: Path, n: int, x: int, y: int) -> tuple[int, int, int]:
    proc = subprocess.run(runner.command(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                                          f"select=eq(n\\,{n}),format=rgb24,crop=1:1:{x}:{y}", "-frames:v", "1",
                                          "-f", "rawvideo", "-"], mounts=[root]), capture_output=True, check=False)
    return tuple(proc.stdout[:3])


@pytest.mark.engine
@engine
@pytest.mark.parametrize("kind,check", [
    ("dissolve", lambda m: 60 < m[0] < 200 and 60 < m[2] < 200),           # midpoint is a red/blue mix
    ("dip_black", lambda m: max(m) < 40),                                   # midpoint is black
    ("wipe_left", None),
    ("push_left", None),
])
def test_video_transitions_render(colors, kind, check):
    root = colors
    res = ops.apply(root, [("transition.add", {"clip": "c1", "duration": 20, "kind": kind})])
    try:
        out = render_frames(root, f"tr_{kind}")
        mid = 50  # the cut; transition spans 40..59
        m = frame_rgb(root, out, mid, 160, 90)
        before, after = frame_rgb(root, out, 30, 160, 90), frame_rgb(root, out, 70, 160, 90)
        assert before[0] > 200 and before[2] < 60, before   # pure red before
        assert after[2] > 200 and after[0] < 60, after      # pure blue after
        if check:
            assert check(m), m
        else:  # wipes/pushes: left and right halves differ mid-way
            left, right = frame_rgb(root, out, mid, 20, 90), frame_rgb(root, out, mid, 300, 90)
            assert left != right, (left, right)
    finally:
        ops.apply(root, [("transition.remove", {"transition": t}) for t in res["results"][0]["transitions"]])


@pytest.mark.engine
@engine
def test_speed_and_effects_render(colors):
    root = colors
    ops.apply(root, [("clip.speed", {"clip": "c1", "speed": 2.0}),
                     ("effect.add", {"clip": "c3", "kind": "color.adjust", "params": {"saturation": 0.0}}),
                     ("effect.add", {"clip": "c3", "kind": "transform", "params": {"scale": 0.5}})])
    p = store.load(root)
    assert p.duration() == 75  # 25 + 50
    out = render_frames(root, "fx")
    r, g, b = frame_rgb(root, out, 40, 160, 90)   # blue clip, desaturated → grey
    assert abs(r - g) < 12 and abs(g - b) < 12 and b < 60, (r, g, b)  # blue → grey (blue has low luma)
    corner = frame_rgb(root, out, 40, 5, 5)        # scaled to half → black border
    assert max(corner) < 20, corner
    proc = subprocess.run(runner.command(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
                                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                                         mounts=[root]), capture_output=True, text=True, check=False)
    assert int(proc.stdout.strip()) == 75


def test_reference_verifier_catches_forward_refs():
    bad = '<mlt><playlist id="pl"><entry producer="later"/></playlist><producer id="later"/></mlt>'
    with pytest.raises(RuntimeError, match="before it is defined"):
        mlt.verify_references(bad)
    mlt.verify_references('<mlt><producer id="p"/><playlist id="pl"><entry producer="p"/></playlist></mlt>')


def tone_level(root: Path, wav: Path, start: float, dur: float, freq: int) -> float:
    """RMS dB of a narrow band around ``freq`` in a window."""
    proc = subprocess.run(runner.command(
        ["ffmpeg", "-hide_banner", "-ss", str(start), "-t", str(dur), "-i", str(wav), "-af",
         f"pan=mono|c0=c0,bandpass=f={freq}:w=60,astats=metadata=1:reset=0", "-f", "null", "-"], mounts=[root]),
        capture_output=True, text=True, check=False)
    return float(proc.stderr.split("RMS level dB:")[-1].split()[0])


@pytest.mark.engine
@engine
def test_audio_crossfade_and_dip(colors):
    root = colors
    p = store.load(root)
    first = min(p.track("V1").clips, key=lambda c: c.start)
    cut = p.rate.to_seconds(first.end)
    res = ops.apply(root, [("transition.add", {"clip": first.id, "duration": 20, "kind": "dissolve"})])
    try:
        x = mlt.write(store.load(root), root, "xfade")
        wav = mlt.render(runner, x, root / "build" / "xfade.wav", root, audio_only=True)
        a_before, b_before = tone_level(root, wav, cut - 0.8, 0.2, 300), tone_level(root, wav, cut - 0.8, 0.2, 600)
        a_mid, b_mid = tone_level(root, wav, cut - 0.06, 0.12, 300), tone_level(root, wav, cut - 0.06, 0.12, 600)
        a_after, b_after = tone_level(root, wav, cut + 0.8, 0.2, 300), tone_level(root, wav, cut + 0.8, 0.2, 600)
        assert a_before - b_before > 30 and b_after - a_after > 30      # A then B
        assert abs(a_mid - b_mid) < 6 and a_mid > a_before - 10         # both present mid-way
    finally:
        ops.apply(root, [("transition.remove", {"transition": t}) for t in res["results"][0]["transitions"]])
    res = ops.apply(root, [("transition.add", {"clip": first.id, "duration": 20, "kind": "dip_black"})])
    try:
        x = mlt.write(store.load(root), root, "adip")
        wav = mlt.render(runner, x, root / "build" / "adip.wav", root, audio_only=True)
        at_cut = max(tone_level(root, wav, cut - 0.04, 0.08, 300), tone_level(root, wav, cut - 0.04, 0.08, 600))
        steady = tone_level(root, wav, cut - 0.8, 0.2, 300)
        assert at_cut < steady - 15, (at_cut, steady)                    # sound dips with the picture
    finally:
        ops.apply(root, [("transition.remove", {"transition": t}) for t in res["results"][0]["transitions"]])
