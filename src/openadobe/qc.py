"""Automatic quality control of a render against its deliverable spec.

``fail`` checks block delivery; ``warn`` checks are things a human editor would look at
(they may be intentional: a dark scene, a deliberate pause).
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from . import captions as cap
from . import perceive
from .engines.runner import Runner
from .lint import errors, lint
from .model import Deliverable, Project


def _check(name: str, ok: bool, severity: str, **info) -> dict:
    return {"check": name, "pass": ok, "severity": severity, **info}


def _intended_dark(project: Project) -> list[tuple[float, float]]:
    """Seconds where black is expected: gaps in main video and fades."""
    rate = project.rate
    spans: list[tuple[int, int]] = []
    mains = sorted((c for t in project.tracks_by_role("main") for c in t.clips), key=lambda c: c.start)
    cursor = 0
    for c in mains:
        if c.start > cursor:
            spans.append((cursor, c.start))
        cursor = max(cursor, c.end)
        if c.fade_in:
            spans.append((c.start, c.start + c.fade_in))
        if c.fade_out:
            spans.append((c.end - c.fade_out, c.end))
    if cursor < project.duration():
        spans.append((cursor, project.duration()))
    return [(rate.to_seconds(a), rate.to_seconds(b)) for a, b in spans]


def _covered(span: tuple[float, float], allowed: list[tuple[float, float]], slack: float = 0.25) -> bool:
    a, b = span
    inside = sum(max(0.0, min(b, y + slack) - max(a, x - slack)) for x, y in allowed)
    return inside >= (b - a) * 0.9


def run(runner: Runner, root: Path, project: Project, deliverable: Deliverable, media: Path) -> dict:
    checks: list[dict] = []
    rate = project.rate

    lint_errors = errors(lint(project))
    checks.append(_check("project.lint", not lint_errors, "fail", issues=lint_errors))

    proc = runner.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                       str(media)], mounts=[Path(root).resolve()])
    info = json.loads(proc.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    want_w, want_h = deliverable.width or project.width, deliverable.height or project.height
    checks.append(_check("format.video", bool(v) and (v["width"], v["height"]) == (want_w, want_h), "fail",
                         value=f"{v['width']}x{v['height']}" if v else None, expected=f"{want_w}x{want_h}"))
    fps_ok = bool(v) and v.get("r_frame_rate") == f"{rate.num}/{rate.den}"
    checks.append(_check("format.fps", fps_ok, "fail", value=v.get("r_frame_rate") if v else None,
                         expected=str(rate)))
    checks.append(_check("format.audio", bool(a) and int(a.get("sample_rate", 0)) == project.sample_rate, "fail",
                         value=a.get("sample_rate") if a else None, expected=project.sample_rate))
    dur = float(info["format"].get("duration", 0))
    want = rate.to_seconds(project.duration())
    checks.append(_check("format.duration", abs(dur - want) <= 0.15, "fail", value=round(dur, 3),
                         expected=round(want, 3)))

    if deliverable.loudness_lufs is not None:
        loud = perceive.loudness(runner, root, media)
        i, tp = loud["integrated_lufs"], loud["true_peak_db"]
        checks.append(_check("audio.loudness", i is not None and abs(i - deliverable.loudness_lufs) <= 1.0, "fail",
                             value=i, expected=deliverable.loudness_lufs, tolerance=1.0))
        checks.append(_check("audio.true_peak", tp is not None and tp <= deliverable.true_peak_db + 0.5, "fail",
                             value=tp, expected=f"<= {deliverable.true_peak_db}"))

    det = perceive.detect(runner, root, media)
    allowed = _intended_dark(project)
    stray_black = [s for s in det["black"] if not _covered(s, allowed)]
    checks.append(_check("video.unintended_black", not stray_black, "warn", spans=stray_black))
    checks.append(_check("video.freeze", not det["freeze"], "warn", spans=det["freeze"]))
    checks.append(_check("audio.silence", not det["silence"], "warn", spans=det["silence"]))

    caps = cap.effective(project)
    style = project.captions.style
    fast = [{"text": c.text, "at": rate.clock(c.start), "cps": round(cap.cps(c, rate), 1)}
            for c in caps if cap.cps(c, rate) > style.max_cps]
    checks.append(_check("captions.reading_speed", not fast, "warn", too_fast=fast, limit=style.max_cps))
    overlap = [rate.clock(b.start) for a_, b in pairwise(caps) if b.start < a_.end]
    late = [rate.clock(c.start) for c in caps if c.end > project.duration()]
    checks.append(_check("captions.timing", not overlap and not late, "fail", overlaps=overlap,
                         beyond_end=late))

    passed = all(c["pass"] for c in checks if c["severity"] == "fail")
    return {"pass": passed, "media": str(media), "revision": project.revision,
            "failed": [c["check"] for c in checks if not c["pass"] and c["severity"] == "fail"],
            "warnings": [c["check"] for c in checks if not c["pass"] and c["severity"] == "warn"],
            "checks": checks}
