from __future__ import annotations

import pytest

from openadobe import store
from openadobe.model import Asset, Probe, Transcript, Word
from openadobe.timebase import Rate


def make_asset(project, kind="video", frames=300, has_audio=True, words=None, aid=None):
    aid = aid or project.new_id("m")
    probe = Probe(duration_s=frames / 30, has_video=kind != "audio", has_audio=has_audio)
    asset = Asset(id=aid, name=aid, kind=kind, source=f"/src/{aid}", sha256=aid, media=f"media/{aid}.mov",
                  probe=probe, duration_frames=frames)
    if words:
        asset.transcript = Transcript(language="zh", model="test",
                                      words=[Word(text=t, start=s, end=e) for t, s, e in words])
    project.assets[aid] = asset
    return asset


@pytest.fixture
def project(tmp_path):
    return store.init(tmp_path / "proj", "test", Rate(num=30, den=1), 1920, 1080)


@pytest.fixture
def root(project, tmp_path):
    return tmp_path / "proj"
