"""Regressions for defects found in the review of the Blender/Ardour integration."""

from __future__ import annotations

import pytest

from openadobe import comps, ops, store
from openadobe import timeline as tl
from openadobe.engines import ardour
from openadobe.model import Comp, Effect
from tests.conftest import make_asset


def _blender(project, root, cid, inputs, template="glow"):
    c = Comp(id=cid, engine="blender", template=template, inputs=inputs, duration=30)
    project.comps[cid] = c
    return comps.materialise(project, root, c)


def test_comp_inputs_render_first_and_unused_comps_are_skipped(project, root):
    make_asset(project, frames=60, aid="m1")
    a1 = _blender(project, root, "k1", {"clip": "m1"})
    a2 = _blender(project, root, "k2", {"clip": a1.id})
    _blender(project, root, "k3", {"clip": "m1"})            # never placed: must not cost a render
    tl.add_clip(project, a2.id, "V1", 0, 0, None, None, insert=False)
    assert [c.id for c in comps.stale(project)] == ["k1", "k2"]


def test_comp_input_cycles_and_lottie_inputs_are_rejected(project, root):
    make_asset(project, frames=60, aid="m1")
    a1 = _blender(project, root, "k1", {"clip": "m1"})
    a2 = _blender(project, root, "k2", {"clip": a1.id})
    project.comps["k1"].inputs = {"clip": a2.id}
    with pytest.raises(ValueError, match="cycle"):
        comps.validate(project, project.comps["k1"])
    lot = Comp(id="k9", engine="lottie", template="title", params={"text": "x"}, duration=30)
    project.comps["k9"] = lot
    la = comps.materialise(project, root, lot)
    with pytest.raises(ValueError, match="Lottie"):
        _blender(project, root, "k4", {"clip": la.id})


def test_comp_set_duration_shortens_its_clips(root):
    res = ops.apply(root, [("comp.add", {"engine": "lottie", "template": "title", "start": 0, "duration": "4s",
                                         "params": {"text": "标题"}})])["results"][0]
    out = ops.apply(root, [("comp.set", {"comp": res["comp"], "duration": "2s"})])["results"][0]
    assert out["clips_shortened"] == [res["clip"]]
    clip = store.load(root).track(res["track"]).clips[0]
    assert clip.duration == 60


def test_ardour_losses_flag_what_the_adapter_cannot_reproduce(project, root):
    a = make_asset(project, frames=300, aid="m1")
    _, au = tl.add_clip(project, a.id, "V1", 0, 0, 100, "A1", insert=False)
    assert ardour.losses(project) == []
    au.effects.append(Effect(id="e1", kind="audio.highpass", params={}))
    au.speed = 2.0
    kinds = {x["kind"] for x in ardour.losses(project)}
    assert {"audio.highpass", "clip.speed"} <= kinds
    store.save(root, project)
    res = ops.apply(root, [("mix.engine", {"engine": "ardour"})])["results"][0]
    assert {x["kind"] for x in res["would_fall_back"]} >= {"audio.highpass", "clip.speed"}
