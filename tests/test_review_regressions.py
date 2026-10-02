"""Regressions for defects found in the P0 review."""

from itertools import pairwise

from openadobe import captions as cap
from openadobe import timeline as tl
from openadobe.envelope import clip_envelope, video_fade_keyframes
from openadobe.lint import errors, lint
from openadobe.model import CaptionStyle, Clip
from tests.conftest import make_asset


def place(project, frames=300):
    a = make_asset(project, frames=frames)
    return tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)


def test_one_frame_piece_keeps_valid_fades(project):
    v, _ = place(project, 100)
    tl.remove_source_ranges(project, v.id, [(10, 20), (21, 30)], ripple=True)
    assert not errors(lint(project))


def test_long_fades_are_clamped_to_surviving_piece(project):
    v, a = place(project, 100)
    for c in (v, a):
        c.fade_in, c.fade_out = 30, 30
    tl.remove_source_ranges(project, v.id, [(0, 60)], ripple=True)
    assert not errors(lint(project))


def test_head_cut_gets_declick_not_original_fade(project):
    v, a = place(project, 100)
    a.fade_in = v.fade_in = 30
    tl.remove_source_ranges(project, v.id, [(0, 20)], ripple=True)
    (audio,) = project.track("A1").clips
    (video,) = project.track("V1").clips
    assert audio.fade_in == 1 and video.fade_in == 0  # new edge mid-speech: declick only


def test_wrap_never_splits_accented_or_cyrillic_words():
    # the balance point falls inside the long accented / Cyrillic word
    for text in ("ab éééééééééééééééé cd", "ab жжжжжжжжжжжжжжжж cd"):
        lines = cap.wrap(text, 6, 2).split("\\N")
        assert all(len(w) in (2, 16) for line in lines for w in line.split()), lines
    for text in ("très bien, c'est une phrase française assez longue", "это довольно длинная русская фраза тест"):
        for line in cap.wrap(text, 12, 2).split("\\N"):
            assert line == line.strip()
        joined = cap.wrap(text, 12, 2).replace("\\N", " ")
        assert joined.split() == text.split()


def test_clean_keeps_latin_punctuation_inside_cjk_captions():
    style = CaptionStyle()
    assert cap._clean("使用 Python 3.5 版本，访问 example.com 下午10:30", style) == \
        "使用 Python 3.5 版本 访问 example.com 下午10:30"
    assert cap._clean("一共1,000人，很多", style) == "一共1,000人 很多"


def test_captions_never_run_past_the_timeline_or_collapse(project):
    a = make_asset(project, frames=30, words=[("你好", 0.6, 0.9)])
    tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
    project.captions.enabled = True
    caps = cap.effective(project)
    assert caps and caps[-1].end <= project.duration()
    assert all(c.end > c.start for c in caps)


def test_simultaneous_dialogue_does_not_emit_zero_length_captions(project):
    # "好。" ends a group; the other speaker's word starts on the same frame → next group starts at 0
    a = make_asset(project, frames=60, words=[("好。", 0.0, 0.3)])
    b = make_asset(project, kind="audio", frames=60, words=[("呀", 0.0, 0.4)])
    tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
    project.tracks.append(project.tracks[1].model_copy(update={"id": "A9", "clips": []}))
    tl.add_clip(project, b.id, "A9", 0, 0, None, None, insert=False)
    project.captions.enabled = True
    assert all(c.end > c.start for c in cap.effective(project))


def test_one_frame_video_fade_survives():
    c = Clip(id="c", asset_id="m", start=0, src_in=0, duration=1, fade_in=5)
    assert video_fade_keyframes(c) == "0=0"


def test_ducking_curve_has_breakpoint_where_ramps_cross(project):
    dia = make_asset(project, frames=300)
    tl.add_clip(project, dia.id, "V1", 30, 0, 30, "A1", insert=False)    # 30..60
    tl.add_clip(project, dia.id, "V1", 85, 60, 30, "A1", insert=False)   # 85..115, gap 25 > merge 24
    music = make_asset(project, kind="audio", frames=300)
    (m,) = tl.add_clip(project, music.id, "A2", 0, 0, None, None, insert=False)
    d = project.mix.ducking
    d.enabled, d.depth_db, d.merge_gap_ms, d.attack_ms, d.release_ms = True, -12, 800, 500, 500
    env = clip_envelope(project, project.track("A2"), m)
    # true curve between 60 and 85: release from 60 (15 f), attack into 85 (15 f) → crossing ≈ 72.5
    xs = [x for x, _ in env]
    assert any(70 <= x <= 75 for x in xs), env
    def at(x):
        for (x0, y0), (x1, y1) in pairwise(env):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    true_72 = min(-12 * (1 - (72 - 60) / 15), -12 * ((72 - 70) / 15))
    assert abs(at(72) - true_72) < 0.6


def test_processed_audio_keeps_transcript_captions_and_text_cuts(root):
    """After clip.replace_audio with a repaired derivative, captions and cut_words still work."""
    from openadobe import ops, store
    p = store.load(root)
    make_asset(p, frames=150, aid="m1", words=[("大家好", 0.0, 0.8), ("嗯", 1.0, 1.4), ("今天", 2.0, 2.5)])
    derived = make_asset(p, kind="audio", frames=150, aid="m2")
    derived.provenance = {"from": "m1", "process": [{"step": "denoise"}]}
    store.save(root, p)
    ops.apply(root, [("clip.add", {"asset": "m1", "track": "V1"}),
                     ("captions.configure", {"enabled": True})])
    p = store.load(root)
    p.track("A1").clips[0].asset_id = "m2"   # what clip.replace_audio does
    store.save(root, p)
    assert [c.text for c in cap.effective(store.load(root))] == ["大家好嗯", "今天"]  # 0.2 s gap: one caption
    res = ops.apply(root, [("edit.cut_words", {"asset": "m1", "words": [[1, 1]]})])["results"][0]
    assert res["removed_frames"] > 0
    p = store.load(root)
    assert "".join(c.text for c in cap.effective(p)) == "大家好今天"  # the filler is gone
    assert len(p.track("V1").clips) == len(p.track("A1").clips) == 2
