"""Ardour adapter, end to end: build a session from the model, mix down headlessly and measure the result.

Needs the openadobe/engines image (Ardour 8.12 + MLT + ffmpeg).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openadobe import ops, perceive, store
from openadobe.engines import ardour, mlt
from openadobe.engines.runner import Runner
from openadobe.model import Effect
from openadobe.timebase import Rate

runner = Runner()
pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(not runner.image_present(), reason="engine image not built"),
]

FPS = 30
DURATION_S = 20  # a ~20 s project, also used to time a mixdown


def ffmpeg(*args: str, cwd: Path) -> None:
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], mounts=[cwd])


def rms_db(wav: Path, start: float, end: float, root: Path) -> float:
    proc = runner.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(wav), "-af",
                       f"atrim={start}:{end},astats=metadata=1:reset=0", "-f", "null", "-"],
                      mounts=[root, wav.parent], check=False)
    return float(proc.stderr.split("RMS level dB:")[-1].split()[0])


def residual_db(wav: Path, notch: float, start: float, end: float, root: Path) -> float:
    """RMS left after removing one tone (a 40 Hz wide notch): what the stem holds besides that tone."""
    proc = runner.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(wav), "-af",
                       f"atrim={start}:{end},bandreject=f={notch}:width_type=h:w=40,astats=metadata=1:reset=0",
                       "-f", "null", "-"], mounts=[root, wav.parent], check=False)
    return float(proc.stderr.split("RMS level dB:")[-1].split()[0])


def probe(wav: Path) -> dict[str, str]:
    proc = runner.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                       "stream=sample_rate,channels,codec_name,duration_ts", "-of", "default=nw=1", str(wav)],
                      mounts=[wav.parent])
    return dict(line.split("=", 1) for line in proc.stdout.split())


@pytest.fixture(scope="module")
def workdir():
    # not pytest's tmp_path_factory: its base directories are pruned by any pytest session starting concurrently
    d = Path(tempfile.mkdtemp(prefix="oa-ardour-")).resolve()
    yield d
    if os.environ.get("OPENADOBE_KEEP_TESTDIR"):
        print(f"\nkept {d}")
    else:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def project_root(workdir):
    d = workdir
    media = d / "media_src"
    media.mkdir()
    # dialogue: 330 Hz bursts at 0-1.5, 2.5-4 and 5-8 s, under 8 s of picture
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i",
           "aevalsrc='0.25*sin(2*PI*330*t)*(lt(t,1.5)+between(t,2.5,4)+gte(t,5))':s=48000:d=8",
           "-t", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(media / "talk.mp4"), cwd=media)
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc='0.3*sin(2*PI*220*t)':s=44100:d={DURATION_S}", str(media / "bed.wav"),
           cwd=media)
    root = d / "proj"
    store.init(root, "ardour", Rate(num=FPS, den=1), 640, 360)
    imp = ops.apply(root, [("asset.import", {"path": str(media / "talk.mp4")}),
                           ("asset.import", {"path": str(media / "bed.wav")})])
    talk, bed = (r["asset"] for r in imp["results"])
    ops.apply(root, [("clip.add", {"asset": talk, "track": "V1"}),
                     ("clip.add", {"asset": bed, "track": "A2", "start": 0}),
                     ("mix.ducking", {"depth_db": -15}),
                     ("track.set", {"track": "A1", "gain_db": -3})])
    p = store.load(root)
    mus = p.track("A2").clips[0]
    ops.apply(root, [("clip.set", {"clip": mus.id, "fade_in": "1s"})])
    p = store.load(root)
    assert p.track("A1").role == "dia" and p.track("A2").role == "mus"
    p.track("A1").fx.append(Effect(id="fx1", kind="audio.highpass", params={"freq": 90.0}))
    p.track("A1").fx.append(Effect(id="fx2", kind="audio.denoise", params={}))  # no Ardour equivalent: skipped
    p.track("A2").fx.append(Effect(id="fx3", kind="audio.compressor", params={"threshold_db": -6, "ratio": 2, "makeup_db": 0}))
    store.save(root, p)
    return root


@pytest.fixture(scope="module")
def mixed(project_root):
    p = store.load(project_root)
    assert p.duration() == DURATION_S * FPS
    out = project_root / "renders" / "ardour-test"
    res = ardour.mixdown(runner, project_root, p, out)
    return res, p, project_root


def test_exact_duration_and_format(mixed):
    res, p, _ = mixed
    expected = p.duration() * 48000 // FPS
    wavs = [res["mix"], *res["stems"].values()]
    assert set(res["stems"]) == {"dia", "mus"}
    for wav in wavs:
        info = probe(wav)
        assert int(info["duration_ts"]) == expected, wav
        assert info["sample_rate"] == "48000" and info["channels"] == "2"
        assert info["codec_name"] == "pcm_s24le"


def test_stems_hold_only_their_role(mixed):
    res, _, root = mixed
    dia, mus = res["stems"]["dia"], res["stems"]["mus"]
    # only music plays after 8.5 s: the dialogue stem is silent there, the music stem is not
    assert rms_db(dia, 9, 19, root) < -90
    assert rms_db(mus, 9, 19, root) > -20
    # while both play, each stem holds only its own tone
    assert rms_db(dia, 6, 7.5, root) > -25
    # a stem minus its own tone leaves (nearly) nothing: no trace of the other role's tone
    assert residual_db(mus, 220, 6, 7.5, root) < rms_db(mus, 6, 7.5, root) - 20
    assert residual_db(dia, 330, 6, 7.5, root) < rms_db(dia, 6, 7.5, root) - 20


def test_ducking_lowers_music_under_dialogue(mixed):
    res, _, root = mixed
    mus = res["stems"]["mus"]
    under = rms_db(mus, 5.6, 7.6, root)
    after = rms_db(mus, 12, 18, root)
    assert 13 <= after - under <= 17, (under, after)  # depth -15 dB; the compressor sits far above the music


def test_music_fade_in_ramps(mixed):
    res, _, root = mixed
    mus = res["stems"]["mus"]
    levels = [rms_db(mus, a, a + 0.25, root) for a in (0.0, 0.25, 0.5, 0.75)]
    assert levels == sorted(levels) and levels[-1] - levels[0] > 20, levels


def test_track_gain_is_the_fader_not_doubled(mixed):
    res, _, root = mixed
    p = store.load(root)
    src = ardour.prepare_sources(runner, root, p)[p.track("A1").clips[0].asset_id]  # what Ardour plays (cached)
    # the dialogue stem is the source minus the -3 dB fader, applied once (not -6 dB)
    assert rms_db(res["stems"]["dia"], 5.2, 7.8, root) == pytest.approx(rms_db(src, 5.2, 7.8, root) - 3.0, abs=0.5)


def test_loudness_matches_mlt_mixdown(mixed):
    res, p, root = mixed
    x = mlt.write(p, root, "ardour-compare")
    wav = mlt.render(runner, x, root / "build" / "ardour-compare.wav", root, audio_only=True)
    ardour_lufs = perceive.loudness(runner, root, str(res["mix"]))["integrated_lufs"]
    mlt_lufs = perceive.loudness(runner, root, str(wav))["integrated_lufs"]
    assert abs(ardour_lufs - mlt_lufs) <= 1.5, (ardour_lufs, mlt_lufs)


def test_session_reopens_and_lists_regions(mixed):
    res, p, _ = mixed
    listing = res["listing"]  # produced by export.lua after re-loading the saved session
    assert {n: v["regions"] for n, v in listing["tracks"].items()} == {
        "A1": len(p.track("A1").clips), "A2": len(p.track("A2").clips)}
    # every track feeds exactly its role bus (not Master as well), every bus feeds Master only
    assert {n: v["to"] for n, v in listing["tracks"].items()} == {"A1": ["DIA"], "A2": ["MUS"]}
    assert listing["buses"] == {"DIA": ["Master"], "MUS": ["Master"]}
    assert listing["tracks"]["A1"]["gain"] == pytest.approx(10 ** (-3 / 20), rel=1e-4)
    by_name = {r["region"]: r for r in listing["regions"]}
    for t in (p.track("A1"), p.track("A2")):
        for c in t.clips:
            r = by_name[c.id]
            assert r["track"] == t.id
            assert r["position"] == c.start * 48000 // FPS
            assert r["length"] == c.duration * 48000 // FPS
            assert r["start"] == c.src_in * 48000 // FPS
    again = ardour.inspect_session(runner, res["session"])  # a second, independent re-open
    assert again["regions"] == listing["regions"]


def test_fx_mapping_report(mixed):
    res, _, _ = mixed
    mapped = {(m["where"], m["kind"]): m["plugin"] for m in res["report"]["mapped"]}
    assert mapped[("A1", "audio.highpass")] == "ACE High/Low Pass Filter"
    assert mapped[("A2", "audio.compressor")] == "urn:ardour:a-comp#stereo"
    assert [s["kind"] for s in res["report"]["skipped"]] == ["audio.denoise"]
    # the plugins are in the saved session, in front of the fader
    routes = {r.get("name"): [pr.get("type") for pr in r.findall("Processor")]
              for r in ET.parse(res["session"] / "mix.ardour").getroot().iter("Route")}
    assert routes["A1"].index("luaproc") < routes["A1"].index("amp")
    assert routes["A2"].index("lv2") < routes["A2"].index("amp")


def test_muted_track_is_silent_and_timing(project_root, workdir):
    p = store.load(project_root)
    p.track("A1").muted = True
    res = ardour.mixdown(runner, project_root, p, workdir / "muted", stems=False)
    assert res["stems"] == {}
    # no dialogue is left, so no ducking either: the mix is the plain music bed
    bed = ardour.prepare_sources(runner, project_root, p)[p.track("A2").clips[0].asset_id]
    assert rms_db(res["mix"], 6, 7.5, project_root) == pytest.approx(rms_db(bed, 6, 7.5, project_root), abs=0.7)
    print(f"\nmixdown wall time for a {DURATION_S} s project: {res['seconds']:.1f} s")


def test_session_is_regenerated(project_root):
    p = store.load(project_root)
    s1 = ardour.build_session(runner, project_root, p)
    marker = s1 / "marker.txt"
    marker.write_text("stale")
    s2 = ardour.build_session(runner, project_root, p)
    assert s1 == s2 and not marker.exists()  # always regenerated
    assert s2.name == f"mix-r{p.revision}" and s2.parent == project_root / "build" / "ardour"


def test_mix_ops(project_root):
    root = project_root
    ops.apply(root, [("mix.engine", {"engine": "ardour"})])
    assert store.load(root).mix.engine == "ardour"
    res = ops.apply(root, [("mix.export_stems", {})])["results"][0]
    assert set(res["stems"]) == {"dia", "mus"} and Path(res["mix"]).exists()
    assert Path(res["mix"]).parent.name.startswith("stems-r")
    ops.apply(root, [("mix.engine", {"engine": "mlt"})])
    assert store.load(root).mix.engine == "mlt"
