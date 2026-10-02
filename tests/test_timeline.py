import pytest

from openadobe import timeline as tl
from openadobe.lint import errors, lint
from openadobe.model import Caption
from tests.conftest import make_asset


def place(project, frames=300, start=None):
    a = make_asset(project, frames=frames)
    return tl.add_clip(project, a.id, "V1", start, 0, None, "A1", insert=False)


def test_add_links_picture_and_sound(project):
    v, a = place(project)
    assert v.link == a.link is not None
    assert (v.start, v.duration) == (a.start, a.duration) == (0, 300)
    assert project.track("A1").clips == [a]


def test_append_and_overlap_is_lint_error(project):
    place(project, 100)
    v2, _ = place(project, 100)
    assert v2.start == 100
    a3 = make_asset(project, frames=50)
    tl.add_clip(project, a3.id, "V1", 150, 0, None, None, insert=False)
    assert any(i["code"] == "track.overlap" for i in errors(lint(project)))


def test_insert_pushes_later_clips(project):
    place(project, 100)
    v2, a2 = place(project, 100)
    a3 = make_asset(project, frames=30)
    tl.add_clip(project, a3.id, "V1", 100, 0, None, "A1", insert=True)
    assert v2.start == a2.start == 130
    assert not errors(lint(project))


def test_split_keeps_link_pairs(project):
    v, _ = place(project)
    right = tl.split(project, v.id, 120)
    assert len(right) == 2
    rv = project.find_clip(right[0])[1]
    ra = project.find_clip(right[1])[1]
    assert rv.link == ra.link != v.link
    assert (v.duration, rv.start, rv.src_in, rv.duration) == (120, 120, 120, 180)


def test_ripple_delete_shifts_only_affected_tracks(project):
    v1, _ = place(project, 100)
    v2, a2 = place(project, 100)
    music = make_asset(project, kind="audio", frames=600)
    (m,) = tl.add_clip(project, music.id, "A2", 0, 0, None, None, insert=False)
    tl.delete(project, v1.id, ripple=True)
    assert v2.start == a2.start == 0
    assert m.start == 0 and m.duration == 600


def test_trim_ripple_and_non_ripple(project):
    v1, a1 = place(project, 100)
    v2, _ = place(project, 100)
    tl.trim(project, v1.id, 10, 20, ripple=True)
    assert (v1.start, v1.src_in, v1.duration) == (0, 10, 70)
    assert a1.duration == 70 and v2.start == 70
    tl.trim(project, v2.id, 5, 0, ripple=False)
    assert (v2.start, v2.src_in, v2.duration) == (75, 5, 95)


def test_trim_cannot_extend_past_media(project):
    v, _ = place(project, 100)
    with pytest.raises(tl.EditError):
        tl.trim(project, v.id, 0, -1, ripple=False)


def test_remove_ranges_ripple(project):
    v, _ = place(project, 300)
    after, _ = place(project, 60)
    res = tl.remove_source_ranges(project, v.id, [(30, 60), (100, 130)], ripple=True)
    assert res["removed_frames"] == 60
    vids = sorted((c for c in project.track("V1").clips if c.asset_id == v.asset_id), key=lambda c: c.start)
    assert [(c.start, c.src_in, c.duration) for c in vids] == [(0, 0, 30), (30, 60, 40), (70, 130, 170)]
    auds = sorted((c for c in project.track("A1").clips if c.asset_id == v.asset_id), key=lambda c: c.start)
    assert [c.fade_out for c in auds] == [1, 1, 0]  # declick at internal cuts only
    assert all(x.link == y.link for x, y in zip(vids, auds))
    assert after.start == 240
    assert not errors(lint(project))


def test_remove_ranges_retimes_manual_captions(project):
    v, _ = place(project, 300)
    project.captions.items = [Caption(id="s1", start=40, end=80, text="a"),
                              Caption(id="s2", start=200, end=220, text="b"),
                              Caption(id="s3", start=35, end=55, text="gone")]
    tl.remove_source_ranges(project, v.id, [(30, 60)], ripple=True)
    caps = {c.id: (c.start, c.end) for c in project.captions.items}
    assert caps["s1"] == (30, 50)  # surviving frames 60..79 → 30..49
    assert caps["s2"] == (170, 190)
    assert "s3" not in caps


def test_remove_everything_is_rejected(project):
    v, _ = place(project, 100)
    with pytest.raises(tl.EditError):
        tl.remove_source_ranges(project, v.id, [(0, 100)], ripple=True)


def test_lint_detects_source_overrun(project):
    v, _ = place(project, 100)
    v.duration = 150
    assert any(i["code"] == "clip.source_range" for i in errors(lint(project)))


def test_cut_words_across_clips_deletes_emptied_clips(root):
    from openadobe import ops, store
    p = store.load(root)
    make_asset(p, frames=150, aid="m1", words=[("大家好", 0.0, 0.8), ("嗯", 1.0, 1.4), ("今天", 2.0, 2.5),
                                                 ("聊", 2.5, 2.8), ("剪辑", 3.5, 4.2)])
    store.save(root, p)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("clip.remove_ranges", {"clip": "c1", "ranges": [["0.7s", "1.8s"]]})])
    # the filler is gone already → cutting it again is an error, not a silent no-op
    import pytest as _pytest
    with _pytest.raises(ops.OpError, match="not in the current cut"):
        ops.apply(root, [("edit.cut_words", {"asset": "m1", "words": [[1, 1]]})])
    res = ops.apply(root, [("edit.cut_words", {"asset": "m1", "words": [[2, 3]]})])["results"][0]
    assert res["removed_text"] == ["今天聊"]
    p = store.load(root)
    assert not errors(lint(p))
    assert len(p.track("V1").clips) == len(p.track("A1").clips)
