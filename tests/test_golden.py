"""Golden reference: fingerprint comparison logic + a full verify run through the engines."""

from __future__ import annotations

import copy
import json

import pytest

from openadobe import golden
from openadobe.engines.runner import Runner

runner = Runner()


def ref() -> dict:
    return json.loads(golden.REF_FILE.read_text("utf-8"))


def test_reference_is_recorded_and_self_consistent():
    r = ref()
    assert r["frame_count"] == 160 and len(r["frames"]) == len(golden.SAMPLE_FRAMES)
    assert golden.compare(r, r) == []


def test_compare_flags_regressions():
    r = ref()
    broken = copy.deepcopy(r)
    broken["frames"][3]["rgb"][0] += 40          # a transition renders differently
    broken["lufs"] += 2                          # the mix got louder
    broken["frame_count"] -= 1                   # a frame went missing
    problems = golden.compare(broken, r)
    assert any("frame 52 rgb" in p for p in problems)
    assert any("loudness" in p for p in problems)
    assert any("frame count" in p for p in problems)


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
def test_golden_verify(tmp_path):
    res = golden.run(runner, tmp_path / "g")
    assert res["ok"], res["problems"]
