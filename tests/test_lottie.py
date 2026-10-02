"""Lottie templates: JSON structure (fast) and real renders through MLT's glaxnimate producer + qtblend (engine)."""

from __future__ import annotations

import json
import os
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import pytest

from openadobe.engines import lottie
from openadobe.engines.runner import Runner

try:
    import numpy as np
except ImportError:  # numpy ships with the `analysis` extra
    np = None

TEMPLATES = [t["name"] for t in lottie.describe()]


# ------------------------------------------------------------------------------------------ fast tests
def test_describe_lists_all_templates():
    assert set(TEMPLATES) == {"title", "lower_third", "callout", "cta", "progress_bar", "kinetic_text"}
    for t in lottie.describe():
        assert t["description"] and t["default_duration_s"] > 0
        assert t["params"]
        for name, spec in t["params"].items():
            assert {"type", "default", "description"} <= set(spec), (t["name"], name)


def _keyframe_times(node) -> list[list[int]]:
    """Every list of keyframe `t` values found anywhere in the document."""
    found: list[list[int]] = []
    if isinstance(node, dict):
        if node.get("a") == 1 and isinstance(node.get("k"), list):
            found.append([kf["t"] for kf in node["k"]])
        for v in node.values():
            found += _keyframe_times(v)
    elif isinstance(node, list):
        for v in node:
            found += _keyframe_times(v)
    return found


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize(("w", "h", "fps", "dur"), [(1920, 1080, 30, 120), (1280, 720, 29.97, 75), (1080, 1920, 25, 40)])
def test_every_template_builds_valid_lottie(template, w, h, fps, dur):
    doc = lottie.build(template, {}, w, h, fps, dur)
    doc = json.loads(json.dumps(doc))  # must be plain JSON
    assert (doc["w"], doc["h"], doc["fr"], doc["ip"], doc["op"]) == (w, h, fps, 0, dur)
    assert doc["layers"]
    assert all(layer["ty"] != 1 for layer in doc["layers"]), "no solid layer: background must stay transparent"
    assert {f["fName"] for f in doc["fonts"]["list"]} >= {"Noto Sans CJK SC-Regular", "Noto Sans CJK SC-Bold"}
    assert [layer["ind"] for layer in doc["layers"]] == list(range(1, len(doc["layers"]) + 1))
    for layer in doc["layers"]:
        assert layer["ip"] == 0 and layer["op"] == dur
    for ts in _keyframe_times(doc):
        assert ts == sorted(set(ts)), "keyframe times must be strictly increasing"
        assert ts[0] >= 0 and ts[-1] <= dur


@pytest.mark.parametrize("template", TEMPLATES)
def test_tiny_durations_still_valid(template):
    doc = lottie.build(template, {}, 640, 360, 30, 6)
    for ts in _keyframe_times(doc):
        assert ts == sorted(set(ts))


@pytest.mark.parametrize("template", TEMPLATES)
def test_unknown_params_and_bad_values_raise(template):
    with pytest.raises(ValueError, match="no params"):
        lottie.build(template, {"nonsense": 1}, 640, 360, 30, 60)
    colour = next((n for n, s in lottie.describe()[TEMPLATES.index(template)]["params"].items() if s["type"] == "color"), None)
    with pytest.raises(ValueError, match="colour"):
        lottie.build(template, {colour: "red"}, 640, 360, 30, 60)


def test_other_validation_errors():
    with pytest.raises(ValueError, match="unknown lottie template"):
        lottie.build("sparkles", {}, 640, 360, 30, 60)
    with pytest.raises(ValueError, match="within"):
        lottie.build("callout", {"target_x": 1.5}, 640, 360, 30, 60)
    with pytest.raises(ValueError, match="one of"):
        lottie.build("progress_bar", {"position": "middle"}, 640, 360, 30, 60)
    with pytest.raises(ValueError, match="string"):
        lottie.build("title", {"text": 5}, 640, 360, 30, 60)
    with pytest.raises(ValueError, match="duration_frames"):
        lottie.build("title", {}, 640, 360, 30, 1)


def test_params_flow_into_document():
    doc = lottie.build("title", {"text": "你好，世界", "subtitle": "副标题", "font": "DejaVu Sans", "color": "#FF0000"},
                       1920, 1080, 30, 90)
    texts = {lay["t"]["d"]["k"][0]["s"]["t"] for lay in doc["layers"] if lay["ty"] == 5}
    assert texts == {"你好，世界", "副标题"}
    main = next(lay for lay in doc["layers"] if lay["nm"] == "title")
    s = main["t"]["d"]["k"][0]["s"]
    assert s["fc"] == [1.0, 0.0, 0.0] and s["f"] == "DejaVu Sans-Bold"
    assert doc["fonts"]["list"][0]["fFamily"] == "DejaVu Sans"


def test_text_width_heuristic():
    assert lottie.text_width("你好世界", 100) == pytest.approx(400)
    assert lottie.text_width("iiii", 100) < lottie.text_width("MMMM", 100)
    assert lottie.text_width("A", 100, bold=True) > lottie.text_width("A", 100)


def test_progress_bar_grows_to_full_width():
    doc = lottie.build("progress_bar", {}, 1000, 500, 30, 61)
    bar = next(lay for lay in doc["layers"] if lay["nm"] == "bar")
    kfs = bar["ks"]["s"]["k"]
    assert kfs[0]["s"][0] == 0 and kfs[-1]["s"][0] == 100 and kfs[-1]["t"] == 60


def test_write_and_producer_props(tmp_path):
    path = lottie.write("cta", {"text": "Subscribe"}, 640, 360, 30, 90, tmp_path / "sub" / "cta.json")
    assert json.loads(path.read_text(encoding="utf-8"))["op"] == 90
    props = lottie.mlt_producer_props(path, 90)
    assert props["mlt_service"] == "glaxnimate" and props["resource"] == str(path.resolve())
    assert props["length"] == "90" and all(isinstance(v, str) for v in props.values())


# ---------------------------------------------------------------------------------------- engine tests
runner = Runner()
engine = pytest.mark.skipif(not runner.image_present() or np is None, reason="engine image or numpy missing")

W, H, FPS, DUR = 640, 360, 30, 90
HOLD = 45  # inside the hold of every template at 90 frames
BG = (32, 64, 128)
OUT_DIR = Path(os.environ["OPENADOBE_LOTTIE_OUT"]).resolve() if os.environ.get("OPENADOBE_LOTTIE_OUT") else None


def save_png(path: Path, arr) -> Path:
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def render(work: Path, name: str, template: str, params: dict, frames: tuple[int, ...] = (0, HOLD, DUR - 1),
           dur: int = DUR) -> dict[int, np.ndarray]:
    """Composite the template over a solid colour with melt (two tracks + qtblend) and return RGB frames."""
    work = work.resolve()
    json_path = lottie.write(template, params, W, H, FPS, dur, work / f"{name}.json")
    mlt = ET.Element("mlt", {"LC_NUMERIC": "C", "version": "7.30.0", "producer": "main"})
    ET.SubElement(mlt, "profile", width=str(W), height=str(H), frame_rate_num=str(FPS), frame_rate_den="1",
                  sample_aspect_num="1", sample_aspect_den="1", progressive="1", colorspace="709")
    bg = ET.SubElement(mlt, "producer", id="bg", attrib={"in": "0", "out": str(dur - 1)})
    ET.SubElement(bg, "property", name="mlt_service").text = "color"
    ET.SubElement(bg, "property", name="resource").text = "#{:02x}{:02x}{:02x}".format(*BG)
    fg = ET.SubElement(mlt, "producer", id="fg", attrib={"in": "0", "out": str(dur - 1)})
    for k, v in lottie.mlt_producer_props(json_path, dur).items():
        ET.SubElement(fg, "property", name=k).text = v
    tractor = ET.SubElement(mlt, "tractor", id="main", attrib={"in": "0", "out": str(dur - 1)})
    multi = ET.SubElement(tractor, "multitrack")
    ET.SubElement(multi, "track", producer="bg")
    ET.SubElement(multi, "track", producer="fg")
    ET.SubElement(tractor, "transition", mlt_service="qtblend", a_track="0", b_track="1", always_active="1",
                  compositing="0", distort="0")
    xml_path = work / f"{name}.mlt"
    ET.ElementTree(mlt).write(xml_path, encoding="utf-8", xml_declaration=True)
    raw = work / f"{name}.rgb"
    env = [f"{k}={v}" for k, v in lottie.MELT_ENV.items()]
    runner.run(["env", *env, "melt", "-silent", str(xml_path), "-consumer", f"avformat:{raw}", "f=rawvideo",
                "vcodec=rawvideo", "pix_fmt=rgb24"], mounts=[work], timeout=240)
    data = np.fromfile(raw, dtype=np.uint8)
    assert data.size == dur * H * W * 3, f"expected {dur} frames, got {data.size / (H * W * 3):.1f}"
    video = data.reshape(dur, H, W, 3)
    out = {f: video[f].copy() for f in frames}
    for f in frames:
        save_png(work / f"{name}_f{f}.png", out[f])
        if OUT_DIR:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            save_png(OUT_DIR / f"{name}_f{f}.png", out[f])
    return out


def ink(frame, tol: int = 40):
    """Boolean mask of pixels that differ from the background colour."""
    return np.abs(frame.astype(int) - np.array(BG)).max(axis=2) > tol


def bbox(mask):
    ys, xs = np.nonzero(mask)
    assert xs.size, "no graphics pixels at all"
    return xs.min() / W, ys.min() / H, xs.max() / W, ys.max() / H


def white(frame):
    return (frame > 215).all(axis=2)


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("lottie")


@pytest.fixture(scope="module")
def renders(work) -> dict[str, dict[int, np.ndarray]]:
    cases = {
        "title": ("title", {"text": "标题 Title", "subtitle": "副标题 Subtitle"}),
        "lower_third": ("lower_third", {"name": "张三 Zhang San", "role": "产品经理 Product Lead"}),
        "callout": ("callout", {"text": "看这里", "target_x": 0.75, "target_y": 0.3, "x": 0.3, "y": 0.7}),
        "cta": ("cta", {"text": "关注"}),
        "progress_bar": ("progress_bar", {"position": "bottom"}),
        "kinetic_text": ("kinetic_text", {"text": "让每一个字 都有节奏", "stagger_s": 0.3}),
    }
    out = {}
    for name, (tpl, params) in cases.items():
        out[name] = render(work, name, tpl, params, frames=(0, 12, HOLD, DUR - 1))
    return out


@engine
@pytest.mark.engine
@pytest.mark.parametrize("template", TEMPLATES)
def test_background_shows_through_and_animates(renders, template):
    frames = renders[template]
    hold = frames[HOLD]
    # (a) alpha: the quiet top-left corner and most of the frame still show the solid background
    assert np.abs(hold[10:40, 10:40].astype(int) - BG).max() <= 3
    assert (~ink(hold, tol=6)).mean() > 0.80
    # graphics exist at the hold
    assert ink(hold).sum() > 150
    # (c) animates: first frame differs from the hold frame
    assert (np.abs(frames[0].astype(int) - hold.astype(int)).max(axis=2) > 40).sum() > 100
    if template != "progress_bar":
        assert ink(frames[0]).sum() < ink(hold).sum() * 0.05, "starts invisible"
        assert ink(frames[DUR - 1]).sum() < ink(hold).sum() * 0.05, "ends invisible"


@engine
@pytest.mark.engine
def test_title_centred_with_chinese(renders):
    x0, y0, x1, y1 = bbox(ink(renders["title"][HOLD]))
    assert abs((x0 + x1) / 2 - 0.5) < 0.04, "title block must be horizontally centred"
    assert abs((y0 + y1) / 2 - 0.5) < 0.08
    assert x1 - x0 > 0.2


@engine
@pytest.mark.engine
def test_lower_third_region(renders):
    x0, y0, x1, y1 = bbox(ink(renders["lower_third"][HOLD]))
    assert 0.04 < x0 < 0.12 and x1 < 0.75
    assert 0.68 < y0 and y1 < 0.93
    assert white(renders["lower_third"][HOLD]).sum() > 100, "name text drawn"


@engine
@pytest.mark.engine
def test_callout_points_at_target(renders):
    hold = renders["callout"][HOLD]
    m = ink(hold)
    tx, ty = int(0.75 * W), int(0.3 * H)
    assert m[ty - 25:ty + 25, tx - 40:tx + 40].sum() > 20, "line/arrow reaches the target"
    bx, by = int(0.3 * W), int(0.7 * H)
    assert white(hold)[by - 30:by + 30, bx - 80:bx + 80].sum() > 80, "label text inside the box near (0.3, 0.7)"
    assert not m[int(0.05 * H):int(0.2 * H)].any(), "nothing above the target area"


@engine
@pytest.mark.engine
def test_cta_pill(renders):
    hold = renders["cta"][HOLD]
    x0, y0, x1, y1 = bbox(ink(hold))
    assert 0.78 < (x0 + x1) / 2 < 0.92 and 0.8 < (y0 + y1) / 2 < 0.95
    red = (hold[..., 0] > 200) & (hold[..., 1] < 90)
    assert red.sum() > 1500, "red pill"
    assert white(hold).sum() > 80, "white label inside the pill"


@engine
@pytest.mark.engine
def test_progress_bar_grows(renders):
    def accent_width(frame):
        row = frame[H - 2]  # bar sits on the bottom edge
        return ((row[:, 0] > 200) & (row[:, 1] > 150) & (row[:, 2] < 80)).sum() / W

    assert accent_width(renders["progress_bar"][0]) < 0.03
    assert accent_width(renders["progress_bar"][12]) == pytest.approx(12 / (DUR - 1), abs=0.03)
    assert accent_width(renders["progress_bar"][HOLD]) == pytest.approx(HOLD / (DUR - 1), abs=0.03)
    assert accent_width(renders["progress_bar"][DUR - 1]) > 0.97


@engine
@pytest.mark.engine
def test_kinetic_text_appears_progressively(renders):
    f = renders["kinetic_text"]
    early, hold = white(f[12]).sum(), white(f[HOLD]).sum()
    assert 0 < early < hold * 0.8, "words arrive one after another"
    x0, _, x1, _ = bbox(ink(f[HOLD]))
    assert abs((x0 + x1) / 2 - 0.5) < 0.05


@engine
@pytest.mark.engine
def test_chinese_title_renders_real_glyphs(work):
    ref = render(work, "zh_a", "title", {"text": "你好世界", "size": 0.16}, frames=(HOLD,))[HOLD]
    other = render(work, "zh_b", "title", {"text": "春夏秋冬", "size": 0.16}, frames=(HOLD,))[HOLD]
    empty = render(work, "zh_empty", "title", {"text": "", "size": 0.16}, frames=(HOLD,))[HOLD]
    a, b, e = white(ref), white(other), white(empty)
    assert e.sum() < 50, "empty text draws no glyphs"
    assert a.sum() > 1500 and b.sum() > 1500, "four CJK characters draw plenty of ink"
    x0, y0, x1, y1 = bbox(a)
    assert 0.2 < x1 - x0 < 0.6 and abs((x0 + x1) / 2 - 0.5) < 0.04
    # tofu boxes would be identical for different characters; real glyphs differ
    inter = (a & b).sum()
    assert inter / max(a.sum(), b.sum()) < 0.7, "different characters must give different shapes"
    # per-character ink density of real CJK glyphs is well below a solid box
    box = a[int(y0 * H):int(y1 * H) + 1, int(x0 * W):int(x1 * W) + 1]
    assert 0.08 < box.mean() < 0.55
