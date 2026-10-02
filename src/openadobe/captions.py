"""Captions: derive from transcripts through the cut, wrap CJK/Latin text, write ASS/SRT.

Transcript-derived captions are rebuilt from the current edit every time, so they follow
every cut automatically — no re-timing step can drift.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .model import Caption, CaptionStyle, Project, Word
from .timebase import Rate

END_PUNCT = "，。、,.;；：:!?！？…"
_WIDE = "\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef"  # CJK, kana, hangul, full-width forms
# a "word" is a run of letters/digits from any script except wide CJK, which may break anywhere
_WORD = re.compile(rf"(?:(?![{_WIDE}])[^\W_]|['’\-])+\s*|\s+|.")
_CJK_PAUSE = re.compile(rf"\s*(?:[，、；：]|(?<=[{_WIDE}])[,;:]|[,;:](?=[{_WIDE}]))\s*")
_CJK_STOP = re.compile(rf"\s*(?:。|(?<=[{_WIDE}])\.)\s*(?=\S)")
BREAK_PUNCT = "。！？!?.…；;"


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def text_units(text: str) -> float:
    return sum(1.0 if is_wide(ch) else 0.55 for ch in text)


def wrap(text: str, max_units: float, max_lines: int) -> str:
    """Balance text across at most two lines (\\N separated).

    Never splits a Latin word, never starts a line with punctuation, and prefers breaking at a
    space (where a CJK pause was) when one is close to the middle.
    """
    text = text.strip()
    if text_units(text) <= max_units or max_lines < 2:
        return text
    tokens = _WORD.findall(text)
    target = text_units(text) / 2
    acc, best, best_i = 0.0, None, len(tokens)
    for i, tok in enumerate(tokens[:-1]):
        acc += text_units(tok)
        nxt = tokens[i + 1]
        if nxt.strip() and nxt.strip()[0] in END_PUNCT:
            continue  # the next line would start with punctuation
        score = abs(acc - target) - (0.35 * target if tok.isspace() else 0)
        if best is None or score < best:
            best, best_i = score, i + 1
    first, second = "".join(tokens[:best_i]).strip(), "".join(tokens[best_i:]).strip()
    return f"{first}\\N{second}" if second else first


def join_words(words: list[str]) -> str:
    out = ""
    for w in words:
        w = w.strip()
        if not w:
            continue
        if out and not (is_wide(out[-1]) or is_wide(w[0]) or w[0] in END_PUNCT):
            out += " "
        out += w
    return out


def _clean(text: str, style: CaptionStyle) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if any(is_wide(ch) for ch in text):
        if style.cjk_pause_as_space:
            # Chinese subtitle convention: pauses shown as spaces, not commas
            text = _CJK_PAUSE.sub(" ", text)
            text = _CJK_STOP.sub(" ", text)
        if style.strip_end_punct:
            text = text.rstrip(END_PUNCT + " ")
    return text.strip()


def timeline_words(project: Project) -> list[tuple[int, int, str]]:
    """Every transcript word that survives the cut, as (timeline start, end, text)."""
    out: list[tuple[int, int, str]] = []
    fps = float(project.rate.fps)
    for t in project.tracks:
        if t.kind != "audio" or t.role != "dia" or t.muted:
            continue
        for c in t.sorted_clips():
            tr = project.transcript_for(c.asset_id)
            if not tr:
                continue
            for w in tr.words:
                ws, we = w.start * fps, w.end * fps
                mid = (ws + we) / 2
                if not (c.src_in <= mid < c.src_out):
                    continue
                s = c.start + (max(ws, c.src_in) - c.src_in) / c.speed
                e = c.start + (min(we, c.src_out) - c.src_in) / c.speed
                out.append((round(s), max(round(e), round(s) + 1), w.text))
    out.sort()
    return out


def from_transcript(project: Project, style: CaptionStyle | None = None) -> list[Caption]:
    style = style or project.captions.style
    rate = project.rate
    words = timeline_words(project)
    limit = style.max_line_units * style.max_lines
    max_len = rate.to_frames(6.0)
    pause = rate.to_frames(0.45)
    groups: list[list[tuple[int, int, str]]] = []
    for w in words:
        if groups:
            g = groups[-1]
            text = join_words([x[2] for x in g] + [w[2]])
            if (w[0] - g[-1][1] > pause or text_units(_clean(text, style)) > limit or w[1] - g[0][0] > max_len
                    or (g[-1][2].strip() and g[-1][2].strip()[-1] in BREAK_PUNCT)):
                groups.append([w])
            else:
                g.append(w)
        else:
            groups.append([w])
    caps = []
    total = project.duration()
    carry: list[tuple[int, int, str]] = []
    for i, g in enumerate(groups):
        start, end = g[0][0], g[-1][1]
        end = max(end, start + rate.to_frames(0.7))  # minimum on-screen time
        if i + 1 < len(groups):
            nxt = groups[i + 1][0][0]
            if nxt <= start:  # simultaneous speech: show it with the next caption instead
                carry += g
                continue
            end = min(end, nxt)
        end = min(end, total)
        words = [*carry, *g]
        text = _clean(join_words([x[2] for x in words]), style)
        carry = []
        if text and end > start:
            timed = [(ws, we, _clean(wt, style) or wt.strip()) for ws, we, wt in words]
            caps.append(Caption(id=f"t{i + 1}", start=start, end=end, text=text, words=timed))
    return caps


def effective(project: Project, style: CaptionStyle | None = None) -> list[Caption]:
    if not project.captions.enabled:
        return []
    caps = from_transcript(project, style) if project.captions.source == "transcript" else []
    return sorted(caps + list(project.captions.items), key=lambda c: c.start)


# ---- writers ---------------------------------------------------------------------------

def _ass_time(frames: int, rate: Rate) -> str:
    cs = round(rate.to_seconds(frames) * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _bgr(rgb: str) -> str:
    rgb = rgb.lstrip("#").upper()
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


PRESETS: dict[str, dict] = {
    "default": {},
    "boxed": {"box": True, "box_opacity": 0.55, "outline": 0.0, "bold": False},
    "yellow": {"color": "FFE14D", "outline": 0.005},
    "shorts": {"size": 0.075, "position": "middle", "animation": "karaoke", "highlight_color": "FFD400",
               "max_line_units": 10, "outline": 0.007, "shadow": 0.004},
    "minimal": {"bold": False, "outline": 0.0, "shadow": 0.003, "size": 0.048},
    "news": {"box": True, "box_color": "0B3D91", "box_opacity": 0.9, "outline": 0.0, "position": "bottom",
             "animation": "fade"},
    "pop": {"animation": "pop", "size": 0.065, "outline": 0.006},
    "typewriter": {"animation": "typewriter", "box": True, "box_opacity": 0.5, "outline": 0.0},
}


def apply_preset(style: CaptionStyle, name: str) -> CaptionStyle:
    if name not in PRESETS:
        raise ValueError(f"unknown caption preset {name!r}; presets: {sorted(PRESETS)}")
    return CaptionStyle.model_validate({**CaptionStyle().model_dump(), **PRESETS[name], "preset": name})


def _alpha(opacity: float) -> str:
    return f"{round((1 - max(0.0, min(1.0, opacity))) * 255):02X}"


def _ass_text(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", "\\N")


def _wrap_words(text: str, words: list[tuple[int, int, str]],
                style: CaptionStyle) -> list[tuple[int, int, str, bool]]:
    """Words flagged 'line break before', placed where wrap() breaks the caption's display text."""
    wrapped = wrap(text, style.max_line_units, style.max_lines)
    if "\\N" not in wrapped:
        return [(*w, False) for w in words]
    cut = len(wrapped.split("\\N")[0].replace(" ", ""))  # break after this many visible characters
    out, seen, broke = [], 0, False
    for w in words:
        flag = not broke and seen >= cut and seen > 0
        broke = broke or flag
        out.append((*w, flag))
        seen += len(w[2].replace(" ", ""))
    return out


def _event_text(c: Caption, style: CaptionStyle, rate: Rate) -> str:
    anim = style.animation
    if anim == "karaoke" and c.words:
        parts, prev_end = [], c.start
        for ws, we, wt, brk in _wrap_words(c.text, c.words, style):
            lead = max(0, ws - prev_end)
            cs = max(1, round(rate.to_seconds(we - ws + lead) * 100))
            sep = "\\N" if brk else ("" if (parts and (is_wide(wt[:1] or "a") or not wt)) else (" " if parts else ""))
            parts.append(f"{sep}{{\\k{cs}}}{_ass_text(wt)}")
            prev_end = we
        return "".join(parts)
    text = _ass_text(wrap(c.text, style.max_line_units, style.max_lines))
    if anim == "fade":
        return "{\\fad(150,150)}" + text
    if anim == "pop":
        return "{\\fad(80,100)\\fscx70\\fscy70\\t(0,140,\\fscx108\\fscy108)\\t(140,220,\\fscx100\\fscy100)}" + text
    if anim == "typewriter":
        chars = list(text.replace("\\N", "\n"))
        span = max(1, round(rate.to_seconds(c.end - c.start) * 1000 * 0.6))
        step = span / max(1, len(chars))
        out = []
        for i, ch in enumerate(chars):
            if ch == "\n":
                out.append("\\N")
                continue
            t0 = round(i * step)
            out.append(f"{{\\1a&HFF&\\t({t0},{t0 + 1},\\1a&H00&)}}{ch}")  # text only; keep the box
        return "".join(out)
    return text


def to_ass(caps: list[Caption], style: CaptionStyle, width: int, height: int, rate: Rate) -> str:
    size = round(style.size * height)
    outline = round(style.outline * height)
    shadow = round(style.shadow * height)
    margin = round(style.margin_v * height)
    align = {"bottom": 2, "middle": 5, "top": 8}[style.position]
    karaoke = style.animation == "karaoke"
    # karaoke: PrimaryColour = already spoken (highlight), SecondaryColour = not yet spoken
    primary = _bgr(style.highlight_color) if karaoke else _bgr(style.color)
    secondary = _bgr(style.color) if karaoke else "&H000000FF"
    if style.box:
        border_style, outline_px = 3, max(outline, round(0.012 * height))
        back = f"&H{_alpha(style.box_opacity)}{_bgr(style.box_color)[4:]}"
        outline_col = back
    else:
        border_style, outline_px = 1, outline
        back = f"&H64{_bgr('000000')[4:]}"
        outline_col = _bgr(style.outline_color)
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font},{size},{primary},{secondary},{outline_col},{back},"
        f"{-1 if style.bold else 0},0,0,0,100,100,0,0,{border_style},{outline_px},{shadow},{align},"
        f"{round(width * 0.06)},{round(width * 0.06)},{margin},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [f"Dialogue: 0,{_ass_time(c.start, rate)},{_ass_time(c.end, rate)},Default,,0,0,0,,"
             f"{_event_text(c, style, rate)}" for c in caps]
    return header + "\n".join(lines) + "\n"


def _srt_time(frames: int, rate: Rate) -> str:
    ms = round(rate.to_seconds(frames) * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(caps: list[Caption], rate: Rate, style: CaptionStyle | None = None) -> str:
    blocks = []
    for i, c in enumerate(caps, 1):
        text = wrap(c.text, style.max_line_units, style.max_lines).replace("\\N", "\n") if style else c.text
        blocks.append(f"{i}\n{_srt_time(c.start, rate)} --> {_srt_time(c.end, rate)}\n{text}\n")
    return "\n".join(blocks)


_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            if m := _SRT_TIME.search(line):
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                out.append((start, end, " ".join(lines[i + 1:]).strip()))
                break
    return out


def cps(caption: Caption, rate: Rate) -> float:
    secs = max(rate.to_seconds(caption.end - caption.start), 1e-3)
    return text_units(caption.text.replace(" ", "")) / secs


def write_files(project: Project, root: Path, tag: str, width: int | None = None,
                height: int | None = None) -> dict[str, Path | None]:
    """ASS laid out for the output size (so 9:16 deliverables get native-size captions) + SRT."""
    w, h = width or project.width, height or project.height
    style = project.captions.style
    if w < h and style.max_line_units > 12:  # portrait: shorter lines, grouped for them too
        style = style.model_copy(update={"max_line_units": 12})
    caps = effective(project, style)
    if not caps:
        return {"ass": None, "srt": None, "count": 0}
    build = Path(root) / "build"
    build.mkdir(exist_ok=True)
    ass = build / f"{tag}.ass"
    srt = build / f"{tag}.srt"
    ass.write_text(to_ass(caps, style, w, h, project.rate), "utf-8")
    srt.write_text(to_srt(caps, project.rate, project.captions.style), "utf-8")
    return {"ass": ass.resolve(), "srt": srt.resolve(), "count": len(caps)}


def words_from_segments(segments) -> list[Word]:
    """Flatten faster-whisper segments into Word objects."""
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(text=w.word, start=float(w.start), end=float(w.end)))
    return words
