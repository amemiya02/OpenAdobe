import json
import xml.etree.ElementTree as ET

import pytest

from openadobe import ops, store
from openadobe.engines import mlt
from openadobe.ops import _drop_slivers, _edge_aware
from tests.conftest import make_asset


def seed(root):
    p = store.load(root)
    make_asset(p, frames=300, aid="m1")
    make_asset(p, kind="audio", frames=600, aid="m2")
    store.save(root, p)


def test_apply_logs_snapshots_and_undo(root):
    seed(root)
    r1 = ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"})])
    assert r1["revision"] == 1 and r1["results"][0]["clips"] == ["c1", "c2"]
    ops.apply(root, [("clip.set", {"clip": "c1", "fade_in": "0.5s"})])
    assert store.load(root).find_clip("c2")[1].fade_in == 15  # linked audio follows
    log = store.read_log(root)
    assert [e["op"] for e in log] == ["clip.add", "clip.set"]
    store.undo(root)
    assert store.load(root).find_clip("c1")[1].fade_in == 0
    store.undo(root)
    assert store.load(root).duration() == 0
    with pytest.raises(store.ProjectError):
        store.undo(root)


def test_invalid_edit_is_rejected_atomically(root):
    seed(root)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"})])
    with pytest.raises(ops.OpError) as exc:
        ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1", "start": 100, "link_audio": False})])
    assert exc.value.issues[0]["code"] == "track.overlap"
    assert store.load(root).revision == 1


def test_batch_is_one_revision_and_params_are_validated(root):
    seed(root)
    res = ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                           ("clip.add", {"asset": "m2", "track": "A2"}),
                           ("mix.ducking", {"depth_db": -10})])
    assert res["revision"] == 1
    with pytest.raises(ops.OpError, match="invalid params"):
        ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1", "bogus": 1})])
    with pytest.raises(ops.OpError, match="unknown op"):
        ops.apply(root, [("clip.explode", {})])


def test_dry_run_changes_nothing(root):
    seed(root)
    res = ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"})], dry_run=True)
    assert res["dry_run"] and store.load(root).revision == 0


def test_every_op_publishes_a_schema():
    for spec in ops.describe():
        assert spec["doc"], spec["op"]
        json.dumps(spec["params"])


def test_silence_cut_helpers():
    assert _edge_aware([(0.0, 1.0), (2.0, 3.0), (5.0, 6.0)], 0.0, 6.0, 0.1) == [(0.0, 0.9), (2.1, 2.9), (5.1, 6.0)]
    # a 3-frame sliver before the end is swallowed into the final cut
    assert _drop_slivers([(30, 60), (153, 177)], 0, 180, 8) == [(30, 60), (153, 180)]
    # a sliver between two cuts merges them
    assert _drop_slivers([(30, 60), (63, 90)], 0, 180, 8) == [(30, 90)]


def test_compile_structure(root):
    seed(root)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1", "start": 30}),
                     ("clip.add", {"asset": "m2", "track": "A2", "start": 0, "duration": 400}),
                     ("clip.set", {"clip": "c1", "fade_in": 15}),
                     ("mix.ducking", {})])
    p = store.load(root)
    doc = ET.fromstring(mlt.compile_project(p, root, master_gain_db=-3, limiter_db=-1.5).split("\n", 1)[1])
    pl = doc.find("playlist[@id='pl_V1']")
    assert pl.find("blank").get("length") == "30"
    assert pl.find("entry").get("out") == "299"
    tracks = doc.findall("tractor/multitrack/track")
    assert [t.get("hide") for t in tracks] == [None, "audio", "video", "video"]
    services = [t.find("property[@name='mlt_service']").text for t in doc.findall("tractor/transition")]
    assert services == ["qtblend", "mix", "mix"]
    fade = doc.find("producer[@id='p_c1']/filter")
    assert fade.find("property[@name='mlt_service']").text == "brightness"
    music_gain = doc.find("producer[@id='p_c3']/filter/property[@name='level']").text
    assert "-14.00" in music_gain  # ducked under dialogue
    filters = [f.find("property[@name='mlt_service']").text for f in doc.findall("tractor/filter")]
    assert filters == ["volume", "avfilter.alimiter"]
