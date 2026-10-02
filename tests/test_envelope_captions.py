import pytest

from openadobe import captions as cap
from openadobe import timeline as tl
from openadobe.envelope import FLOOR_DB, clip_envelope, to_keyframes, video_fade_keyframes
from openadobe.model import Caption
from tests.conftest import make_asset


def test_fades_and_gain():
    from openadobe.model import Clip, Project, Track
    p = Project(name="x")
    t = Track(id="A1", kind="audio", role="dia", name="d", gain_db=-3)
    c = Clip(id="c1", asset_id="m1", start=0, src_in=0, duration=100, fade_in=10, fade_out=20, gain_db=2)
    env = clip_envelope(p, t, c)
    assert env == [(0, FLOOR_DB), (10, -1.0), (79, -1.0), (99, FLOOR_DB)]
    assert to_keyframes(env) == "0=-60.00;10=-1.00;79=-1.00;99=-60.00"


def test_ducking_follows_dialogue(project):
    dia = make_asset(project, frames=60)
    tl.add_clip(project, dia.id, "V1", 30, 0, None, "A1", insert=False)  # dialogue 30..90
    music = make_asset(project, kind="audio", frames=300)
    (m,) = tl.add_clip(project, music.id, "A2", 0, 0, None, None, insert=False)
    project.mix.ducking.enabled = True
    project.mix.ducking.depth_db = -12
    env = dict(clip_envelope(project, project.track("A2"), m))
    attack = project.rate.to_frames(0.25)
    release = project.rate.to_frames(0.6)
    assert env[0] == 0.0
    assert env[30 - attack] == 0.0 and env[30] == -12.0 and env[90] == -12.0
    assert env[90 + release] == 0.0
    # dialogue itself is never ducked
    a1 = project.track("A1").clips[0]
    assert all(db == 0.0 for _, db in clip_envelope(project, project.track("A1"), a1))


def test_video_fade_keyframes():
    from openadobe.model import Clip
    c = Clip(id="c", asset_id="m", start=0, src_in=0, duration=60, fade_in=15, fade_out=15)
    assert video_fade_keyframes(c) == "0=0;15=1;44=1;59=0"
    assert video_fade_keyframes(c.model_copy(update={"fade_in": 0, "fade_out": 0})) is None


def test_wrap_cjk_and_latin():
    text = "今天我们来聊一聊如何让智能体真正地剪辑视频"
    wrapped = cap.wrap(text, 12, 2)
    first, second = wrapped.split("\\N")
    assert first + second == text and abs(len(first) - len(second)) <= 2
    latin = cap.wrap("the quick brown fox jumps over the lazy dog again", 12, 2)
    assert all(not part.startswith(" ") for part in latin.split("\\N"))
    assert "jum\\Nps" not in latin


def test_transcript_captions_follow_the_cut(project):
    words = [("大家", 0.1, 0.4), ("好", 0.4, 0.6), ("。", 0.6, 0.6),
             ("呃", 1.0, 1.3),
             ("今天", 2.0, 2.3), ("聊", 2.3, 2.5), ("剪辑", 2.5, 2.9)]
    a = make_asset(project, frames=120, words=words)
    v, _ = tl.add_clip(project, a.id, "V1", 0, 0, None, "A1", insert=False)
    project.captions.enabled = True
    before = cap.effective(project)
    assert [c.text for c in before] == ["大家好", "呃", "今天聊剪辑"]
    tl.remove_source_ranges(project, v.id, [(27, 57)], ripple=True)  # cut 0.9s–1.9s (the "呃")
    after = cap.effective(project)
    assert [c.text for c in after] == ["大家好", "今天聊剪辑"]
    assert after[1].start == project.rate.to_frames(2.0) - 30


def test_ass_and_srt_roundtrip(project):
    caps = [Caption(id="s1", start=30, end=75, text="你好 world")]
    ass = cap.to_ass(caps, project.captions.style, 1920, 1080, project.rate)
    assert "PlayResY: 1080" in ass and "Dialogue: 0,0:00:01.00,0:00:02.50,Default" in ass
    srt = cap.to_srt(caps, project.rate)
    assert cap.parse_srt(srt) == [(1.0, 2.5, "你好 world")]


def test_cjk_pauses_become_spaces_and_lines_never_start_with_punctuation(project):
    from openadobe.model import CaptionStyle
    style = CaptionStyle()
    assert cap._clean("它操作的是真实的剪辑引擎,而不是用代码画出来的动画。", style) == \
        "它操作的是真实的剪辑引擎 而不是用代码画出来的动画"
    wrapped = cap.wrap("它操作的是真实的剪辑引擎 而不是用代码画出来的动画", 16, 2)
    assert wrapped == "它操作的是真实的剪辑引擎\\N而不是用代码画出来的动画"
    assert not any(line[:1] in cap.END_PUNCT for line in cap.wrap("一二三四五六七八，九十一二三四五六七八", 10, 2).split("\\N"))
    assert cap._clean("Hello, world.", style) == "Hello, world."  # Latin text untouched


def test_fix_transcript_merges_words(root):
    from openadobe import ops, store
    p = store.load(root)
    make_asset(p, frames=90, aid="m1", words=[("欢迎", 0.0, 0.5), ("Open", 0.5, 0.8), ("ado。", 0.8, 1.2),
                                                ("今天", 1.5, 2.0)])
    store.save(root, p)
    ops.apply(root, [("asset.fix_transcript", {"asset": "m1", "replace": [[1, 2, "OpenAdobe。"]]})])
    words = store.load(root).assets["m1"].transcript.words
    assert [(w.text, w.start, w.end) for w in words] == [("欢迎", 0.0, 0.5), ("OpenAdobe。", 0.5, 1.2),
                                                         ("今天", 1.5, 2.0)]


def test_caption_presets_and_karaoke(project):
    from openadobe.model import CaptionStyle
    c = Caption(id="t1", start=0, end=90, text="今天我们来聊一聊 怎么让智能体剪辑视频",
                words=[(0, 10, "今天"), (10, 20, "我们"), (20, 25, "来"), (25, 40, "聊一聊"), (42, 50, "怎么"),
                       (50, 55, "让"), (55, 70, "智能体"), (70, 80, "剪辑"), (80, 88, "视频")])
    shorts = cap.apply_preset(CaptionStyle(), "shorts")
    line = cap.to_ass([c], shorts, 1080, 1920, project.rate).splitlines()[-1]
    assert line.endswith("{\\k33}今天{\\k33}我们{\\k17}来{\\k50}聊一聊\\N{\\k33}怎么{\\k17}让{\\k50}智能体{\\k33}剪辑{\\k27}视频")
    style_line = next(x for x in cap.to_ass([c], shorts, 1080, 1920, project.rate).splitlines() if x.startswith("Style"))
    assert ",&H0000D4FF,&H00FFFFFF," in style_line and style_line.endswith(",5,65,65,134,1")  # highlight, middle
    boxed = cap.to_ass([c], cap.apply_preset(CaptionStyle(), "boxed"), 1920, 1080, project.rate)
    assert ",3," in next(x for x in boxed.splitlines() if x.startswith("Style"))  # opaque box border style
    with pytest.raises(ValueError):
        cap.apply_preset(CaptionStyle(), "nope")
