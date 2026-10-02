"""Blender comp templates: schema (fast) and real headless renders in the engine image (marked `engine`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openadobe.engines import blender
from openadobe.engines.runner import Runner

runner = Runner()
W, H = 640, 360
FPS = (24, 1)


# ---- schema / validation (no engine needed) -----------------------------------------------------------------


def test_describe_lists_all_templates():
    d = {t["name"]: t for t in blender.describe()}
    assert set(d) == {"keying", "title_3d", "glow", "card_3d"}
    for t in d.values():
        assert t["description"] and t["default_duration_s"] > 0
        for spec in t["params"].values():
            assert {"type", "default", "description"} <= set(spec)
        for slot in t["inputs"].values():
            assert isinstance(slot["required"], bool) and slot["description"]
    assert d["keying"]["inputs"]["fg"]["required"] and not d["keying"]["inputs"]["bg"]["required"]
    json.dumps(blender.describe())  # agent-facing: must be JSON-serialisable


def test_validate_fills_defaults_and_rejects_bad_input():
    p = blender.validate("title_3d", {"text": "你好"}, {})
    assert p["text"] == "你好" and p["extrude"] == 0.12 and p["font"].startswith("Noto Sans CJK")
    with pytest.raises(ValueError, match="unknown params"):
        blender.validate("glow", {"nope": 1}, {"clip": "x.mp4"})
    with pytest.raises(ValueError, match="missing required inputs"):
        blender.validate("keying", {}, {"bg": "bg.mp4"})
    with pytest.raises(ValueError, match="unknown inputs"):
        blender.validate("glow", {}, {"clip": "a", "extra": "b"})
    with pytest.raises(ValueError, match="one of"):
        blender.validate("card_3d", {"direction": "sideways"}, {"clip": "a"})
    with pytest.raises(ValueError, match="outside"):
        blender.validate("glow", {"size": 99}, {"clip": "a"})
    with pytest.raises(ValueError, match="#RRGGBB"):
        blender.validate("keying", {"key_color": "green"}, {"fg": "a"})
    with pytest.raises(ValueError, match="clip_black"):
        blender.validate("keying", {"clip_black": 0.9, "clip_white": 0.2}, {"fg": "a"})
    with pytest.raises(ValueError, match="unknown blender template"):
        blender.validate("nope", {}, {})


# ---- engine renders ------------------------------------------------------------------------------------------


def ff(*args: str, cwd: Path, ro: Path | None = None) -> None:
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], mounts=[cwd], ro=[ro] if ro else [])


def probe(path: Path) -> dict:
    out = runner.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        ro=[path.parent],
    ).stdout
    return json.loads(out)["streams"][0]


def frame(path: Path, n: int, w: int, h: int):
    """Decode frame n to an (h, w, 4) uint8 RGBA array."""
    np = pytest.importorskip("numpy")
    raw = path.with_suffix(f".{n}.raw")
    runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n})",
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", str(raw)], mounts=[path.parent])
    return np.frombuffer(raw.read_bytes(), np.uint8).reshape(h, w, 4).astype(int)


def check_output(res: dict, frames: int, w: int = W, h: int = H, fps: tuple[int, int] = FPS) -> dict:
    assert res["video"].exists() and res["blend"].exists() and res["blend"].stat().st_size > 10_000
    assert res["seconds"] > 0 and res["engine"] in ("CYCLES", "compositor")
    s = probe(res["video"])
    assert s["codec_name"] == "ffv1" and s["pix_fmt"] == "gbrap10le"  # MLT keeps rgba alpha (it flattens ProRes 4444)
    assert int(s["nb_read_frames"]) == frames
    assert (s["width"], s["height"]) == (w, h)
    assert s["r_frame_rate"] == f"{fps[0]}/{fps[1]}"
    return s


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("blender_fixtures").resolve()
    # synthetic green-screen shot: #00FF00 background, red box moving right, white disc top right
    ff("-f", "lavfi", "-i", f"color=c=0x00FF00:s={W}x{H}:r=24", "-f", "lavfi", "-i", "color=c=0xFF0000:s=120x120:r=24",
       "-filter_complex",
       "[0][1]overlay=x='40+t*100':y=130,format=gbrp,"
       "geq=r='if(lt(hypot(X-540,Y-70),45),255,r(X,Y))':g='if(lt(hypot(X-540,Y-70),45),255,g(X,Y))':"
       "b='if(lt(hypot(X-540,Y-70),45),255,b(X,Y))',format=yuv444p",
       "-t", "1", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv444p", str(d / "greenscreen.mp4"), cwd=d)
    ff("-f", "lavfi", "-i", f"color=c=0x2040C0:s={W}x{H}:r=24", "-t", "1", "-c:v", "libx264", "-crf", "8",
       "-pix_fmt", "yuv444p", str(d / "plate.mp4"), cwd=d)
    # dark clip with a small bright square (for glow)
    ff("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=24,drawbox=x=300:y=160:w=40:h=40:color=white:t=fill",
       "-t", "1", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv444p", str(d / "dark.mp4"), cwd=d)
    # opaque "screen recording" stand-in for the card
    ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r=24", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       str(d / "screen.mp4"), cwd=d)
    return {k: d / f"{k}.mp4" for k in ("greenscreen", "plate", "dark", "screen")}


@pytest.fixture(scope="module")
def outdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("blender_out").resolve()


@pytest.fixture(scope="module")
def keyed(media, outdir):
    res = blender.render(runner, outdir, "keying", {}, {"fg": media["greenscreen"]}, W, H, *FPS, 24,
                         outdir / "keyed.mkv")
    return res


@pytest.mark.engine
@pytest.mark.skipif(not runner.image_present(), reason="engine image not built")
class TestEngine:
    def test_keying_alpha_and_colour(self, keyed):
        check_output(keyed, 24)
        assert keyed["engine"] == "compositor"
        for n, box_x in ((0, 40), (12, 90), (23, 135)):
            f = frame(keyed["video"], n, W, H)
            assert f[10, 10, 3] <= 3  # green screen -> transparent
            assert f[300, 300, 3] <= 3
            r, g, b, a = f[190, box_x + 60]  # centre of the moving red box
            assert a >= 252 and r >= 245 and g <= 12 and b <= 12, (n, (r, g, b, a))
            r, g, b, a = f[70, 540]  # white disc
            assert a >= 252 and min(r, g, b) >= 240
            # no green fringe left on the opaque disc/box (despill)
            assert f[190, box_x + 60, 1] <= 12
        # the compiled .blend is a real project and keeps its media next to it
        assert (keyed["video"].parent / "keyed.inputs" / "fg.mov").exists()

    def test_keying_over_plate_with_garbage_matte_and_ntsc_rate(self, media, outdir):
        res = blender.render(
            runner, outdir, "keying", {"garbage": [0.0, 0.5, 0.0, 0.0]},
            {"fg": media["greenscreen"], "bg": media["plate"]}, W, H, 30000, 1001, 12, outdir / "keyed_bg.mkv")
        check_output(res, 12, fps=(30000, 1001))
        f = frame(res["video"], 3, W, H)
        assert (f[..., 3] >= 252).all()  # plate fills the frame
        r, g, b, _ = f[10, 10]
        assert abs(r - 0x20) <= 6 and abs(g - 0x40) <= 6 and abs(b - 0xC0) <= 6  # plate shows through
        assert f[190, 100][0] >= 245 and f[190, 100][1] <= 12  # box still on top
        # garbage matte removed the right half, including the disc
        assert abs(f[70, 540][0] - 0x20) <= 6 and abs(f[70, 540][2] - 0xC0) <= 6

    def test_title_3d_english(self, outdir):
        res = blender.render(runner, outdir, "title_3d", {"text": "OpenAdobe", "subtitle": "Agent-operated post"}, {},
                             W, H, *FPS, 12, outdir / "title_en.mkv")
        check_output(res, 12)
        assert res["engine"] == "CYCLES"
        for n in (0, 11):
            f = frame(res["video"], n, W, H)
            a = f[..., 3]
            centre = a[H // 2 - 80:H // 2 + 80, W // 2 - 220:W // 2 + 220]
            assert (centre >= 250).mean() > 0.10  # solid glyph pixels in the middle
            for corner in (a[:30, :30], a[:30, -30:], a[-30:, :30], a[-30:, -30:]):
                assert corner.max() <= 3  # transparent corners
        # camera pushes in: the title covers more pixels at the end than at the start
        first = (frame(res["video"], 0, W, H)[..., 3] >= 128).sum()
        last = (frame(res["video"], 11, W, H)[..., 3] >= 128).sum()
        assert last > first * 1.05

    def test_title_3d_chinese_glyphs_not_tofu(self, outdir):
        a = blender.render(runner, outdir, "title_3d", {"text": "你好，世界"}, {}, W, H, *FPS, 2, outdir / "title_zh1.mkv")
        b = blender.render(runner, outdir, "title_3d", {"text": "永恒春天"}, {}, W, H, *FPS, 2, outdir / "title_zh2.mkv")
        fa, fb = frame(a["video"], 0, W, H), frame(b["video"], 0, W, H)
        for f in (fa, fb):
            assert (f[..., 3] >= 250).sum() > 6000  # real glyph coverage
            assert f[:20, :20, 3].max() <= 3
        # missing-glyph boxes would look identical for any text; real glyphs differ between strings
        diff = abs(fa[..., 3] - fb[..., 3]) > 128
        assert diff.sum() > 2500

    def test_glow_blooms_around_bright_shape(self, media, outdir):
        res = blender.render(runner, outdir, "glow", {}, {"clip": media["dark"]}, W, H, *FPS, 12, outdir / "glow.mkv")
        check_output(res, 12)
        src = outdir / "dark_src.mov"
        ff("-i", str(media["dark"]), "-c:v", "prores_ks", "-profile:v", "4444", str(src), cwd=outdir,
           ro=media["dark"].parent)
        s = frame(src, 5, W, H)
        g = frame(res["video"], 5, W, H)
        lum = lambda f, y, x: f[y, x, :3].mean()
        for dx in (10, 25, 50):  # ring around the 40 px square (x 300..340, y 160..200)
            assert lum(g, 180, 340 + dx) > lum(s, 180, 340 + dx) + 15, dx
        # falls off with distance and leaves the far corners dark
        assert lum(g, 180, 345) > lum(g, 180, 400) > lum(g, 180, 560)
        assert lum(g, 5, 5) < 25
        assert g[180, 320, :3].min() >= 240  # the shape itself stays bright
        assert (g[..., 3] == 255).all()

    def test_card_3d_perspective_quad_over_transparency(self, media, outdir):
        res = blender.render(runner, outdir, "card_3d", {"reflection": 0.0, "shadow": 0.4}, {"clip": media["screen"]},
                             W, H, *FPS, 12, outdir / "card.mkv")
        check_output(res, 12)
        f = frame(res["video"], 0, W, H)  # swing starts at -rotation: card turned in perspective
        solid = f[..., 3] >= 250
        assert solid.sum() > 15000
        for corner in (f[:25, :25, 3], f[:25, -25:, 3], f[-25:, :25, 3], f[-25:, -25:, 3]):
            # transparent corners; the shadow catcher leaves a faint (<= ~5 %) soft-shadow/occlusion haze
            assert corner.mean() <= 10.0 and corner.max() <= 40
        cols = [x for x in range(W) if solid[:, x].sum() > 4]
        left, right = cols[0] + 8, cols[-1] - 8
        hl, hr = solid[:, left].sum(), solid[:, right].sum()
        # perspective: the nearer edge is clearly taller than the far edge -> not a rectangle
        assert max(hl, hr) > 1.12 * min(hl, hr), (hl, hr)
        # the card is not axis-aligned: top edge slopes
        top = lambda x: int(solid[:, x].argmax())
        assert abs(top(left) - top(right)) > 8
        # mid-swing (frame 6) the card faces the camera -> near-rectangular
        g = frame(res["video"], 6, W, H)
        s2 = g[..., 3] >= 250
        cols2 = [x for x in range(W) if s2[:, x].sum() > 4]
        h1, h2 = s2[:, cols2[0] + 8].sum(), s2[:, cols2[-1] - 8].sum()
        assert max(h1, h2) < 1.06 * min(h1, h2)

    def test_preview_is_half_resolution(self, media, outdir):
        res = blender.render(runner, outdir, "glow", {}, {"clip": media["dark"]}, W, H, *FPS, 6,
                             outdir / "glow_prev.mkv", preview=True)
        check_output(res, 6, w=W // 2, h=H // 2)
