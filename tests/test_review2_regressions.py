"""Regressions for defects found in the second review."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from openadobe import captions as cap
from openadobe import comps, ops, store
from openadobe import timeline as tl
from openadobe.engines import mlt
from openadobe.lint import errors, lint
from openadobe.model import Caption, CaptionStyle, Comp, Effect
from openadobe.timebase import Rate
from tests.conftest import make_asset


def test_preview_comp_render_does_not_satisfy_final(project, root):
    make_asset(project, frames=30, aid="m1")
    c = Comp(id="k1", engine="blender", template="glow", inputs={"clip": "m1"}, duration=30)
    project.comps["k1"] = c
    asset = comps.materialise(project, root, c)
    assert comps.stale(project) == []                  # not on the timeline: never rendered
    tl.add_clip(project, asset.id, "V1", 0, 0, None, None, insert=False)
    key = comps.render_key(project, c)
    c.render_key = key + ":preview"
    assert comps.stale(project) == [c]                 # final delivery still needs a full render
    assert comps.stale(project, preview=True) == []    # previews may reuse the preview render


def test_dissolve_between_stills_has_no_negative_in_points(project, root):
    a = make_asset(project, kind="image", frames=0, aid="m1", has_audio=False)
    b = make_asset(project, kind="image", frames=0, aid="m2", has_audio=False)
    va = tl.add_clip(project, a.id, "V1", 0, 0, 50, None, insert=False)[0]
    tl.add_clip(project, b.id, "V1", None, 0, 50, None, insert=False)
    tl.add_transition(project, "V1", va.id, 10, "dissolve", audio=False)
    assert not errors(lint(project))
    doc = ET.fromstring(mlt.compile_project(project, root).split("\n", 1)[1])
    assert all(int(e.get("in")) >= 0 for e in doc.iter("entry"))
    assert all(int(f.get("in", "0")) >= 0 for f in doc.iter("filter"))


def test_split_gives_independent_effects_and_continues_keyframes(project):
    a = make_asset(project, frames=100, aid="m1")
    v, _ = tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
    v.effects.append(Effect(id="e1", kind="transform", params={"scale": {0: 1.0, 99: 2.0}}))
    right_id = tl.split(project, v.id, 50)[0]
    right = project.find_clip(right_id)[1]
    assert right.effects is not v.effects and right.effects[0].id != "e1"
    scale = right.effects[0].params["scale"]
    assert abs(scale[0] - (1.0 + 50 / 99)) < 1e-6 and scale[49] == 2.0
    v.effects.clear()
    assert right.effects


def test_split_keeps_outgoing_transition_on_right_half(project):
    a = make_asset(project, frames=200, aid="m1")
    b = make_asset(project, frames=200, aid="m2")
    va, _ = tl.add_clip(project, a.id, "V1", 0, 0, 100, "A1", insert=False)
    tl.add_clip(project, b.id, "V1", None, 20, 100, "A1", insert=False)
    tl.add_transition(project, "V1", va.id, 10, "dissolve", audio=True)
    right = tl.split(project, va.id, 40)
    assert tl.prune_transitions(project) == []
    assert project.track("V1").transitions[0].a == right[0]


def test_typewriter_keeps_box_opacity():
    c = Caption(id="t", start=0, end=30, text="ab")
    line = cap.to_ass([c], cap.apply_preset(CaptionStyle(), "typewriter"), 640, 360,
                      Rate(num=30, den=1)).splitlines()[-1]
    assert "\\1a&HFF&" in line and "\\alpha" not in line


def test_speed_rounding_never_overruns_media(project):
    a = make_asset(project, frames=100, aid="m1")
    for s in (1.5, 1.8, 1.9, 0.7, 3.3):
        v, _ = tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
        tl.set_speed(project, v.id, s, ripple=False)
        assert not errors(lint(project)), (s, errors(lint(project)))
        assert mlt.producer_in(v) + v.duration <= mlt.producer_length(v, a)
        right = project.find_clip(tl.split(project, v.id, v.start + v.duration // 2)[0])[1]
        assert mlt.producer_in(right) == mlt.producer_in(v) + v.duration
        tl.delete(project, v.id, ripple=False)
        tl.delete(project, right.id, ripple=False)


def test_speed_is_rejected_on_lottie_comps(root):
    p = store.load(root)
    make_asset(p, frames=90, aid="m1")
    store.save(root, p)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("comp.add", {"engine": "lottie", "template": "title", "start": 0, "duration": "2s",
                                   "params": {"text": "x"}})])
    clip = store.load(root).tracks[-1].clips[0].id
    with pytest.raises(ops.OpError, match="speed"):
        ops.apply(root, [("clip.speed", {"clip": clip, "speed": 0.5})])


def test_montage_length_clamped_to_music_and_empty_pool_rejected(root):
    p = store.load(root)
    make_asset(p, frames=900, aid="m1")
    mus = make_asset(p, kind="audio", frames=120, aid="m2")
    mus.analysis["beats"] = {"tempo": 120, "beats": [i * 0.5 for i in range(1, 8)]}
    store.save(root, p)
    res = ops.apply(root, [("edit.montage", {"clips": ["m1"], "music": "m2", "length": "20s"})])["results"][0]
    assert res["length"] == 120 and "warning" in res
    with pytest.raises(ops.OpError):
        ops.apply(root, [("edit.montage", {"clips": [], "music": "m2"})])


def test_portrait_captions_grouped_for_portrait_lines(project):
    words = [(f"字{i}", i * 0.2, i * 0.2 + 0.18) for i in range(30)]
    a = make_asset(project, frames=300, words=words)
    tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
    project.captions.enabled = True
    portrait = project.captions.style.model_copy(update={"max_line_units": 12})
    assert all(cap.text_units(c.text) <= 24 for c in cap.effective(project, portrait))


def test_empty_keyframes_rejected():
    from openadobe import effects
    with pytest.raises(ValueError):
        effects.validate("transform", {"scale": {}})
