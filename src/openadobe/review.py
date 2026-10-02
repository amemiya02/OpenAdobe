"""Review package: everything a reviewer (human or vision model) needs to judge a cut, in one place.

``openadobe review`` collects a time-stamped contact sheet, waveform, loudness + QC results, the
caption text and the timeline summary, plus a rubric. With ``--vlm`` and ANTHROPIC_API_KEY set, the
package is sent to a Claude vision model, which returns time-coded issues with suggested ops.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

from . import perceive, qc, store
from .engines.runner import Runner

RUBRIC = """You are a senior video editor reviewing a cut. Judge it like a broadcast/online editor:
1. Story & pacing: dead air, repeated takes, cuts mid-word, jump cuts that need cover (b-roll/zoom).
2. Picture: black/frozen frames, bad framing in vertical versions, flashes, mismatched colour between shots.
3. Graphics & captions: readability, timing vs speech, typos (check names), safe areas, overlaps.
4. Sound: dialogue clarity vs music, ducking, abrupt starts/ends, clicks at cuts, loudness/peaks.
Return JSON: {"verdict": "ship"|"fix", "score": 0-10, "issues": [{"at": "MM:SS", "severity": "high|medium|low",
"problem": "...", "fix": "openadobe op suggestion"}]}. Only report issues you can see or infer from the material."""


def build(runner: Runner, root: Path, media: str | Path | None = None, n: int = 16) -> dict:
    root = Path(root).resolve()
    project = store.load(root)
    path = perceive._media(root, media)
    out_dir = root / "perceive" / f"review-{path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = perceive.contact_sheet(runner, root, path, n=n)
    wave = perceive.waveform(runner, root, path)
    deliverable = next((d for d in project.deliverables.values() if path.stem.startswith(f"{d.name}-r")),
                       project.deliverables.get("main"))
    report = qc.run(runner, root, project, deliverable, path)
    summary = perceive.inspect(project, root)
    package = {
        "media": str(path), "revision": project.revision, "contact_sheet": sheet["image"], "waveform": wave["image"],
        "qc": {k: report[k] for k in ("pass", "failed", "warnings")},
        "loudness": next((c for c in report["checks"] if c["check"] == "audio.loudness"), None),
        "captions": [f"{c['start']} {c['text']}" for c in summary["captions"]["items"]],
        "timeline": {"duration": summary["duration"],
                     "tracks": [{"track": t["track"], "role": t["role"], "clips": len(t["clips"])}
                                for t in summary["tracks"]]},
        "rubric": RUBRIC,
    }
    (out_dir / "review.json").write_text(json.dumps(package, ensure_ascii=False, indent=1), "utf-8")
    package["package"] = str(out_dir / "review.json")
    return package


def vlm_request(package: dict, model: str) -> dict:
    """The Messages API request body for a vision review (images + facts + rubric)."""
    content = []
    for key in ("contact_sheet", "waveform"):
        data = base64.b64encode(Path(package[key]).read_bytes()).decode()
        mime = "image/jpeg" if package[key].endswith(".jpg") else "image/png"
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
    facts = {k: package[k] for k in ("qc", "loudness", "captions", "timeline")}
    content.append({"type": "text", "text": f"Facts:\n{json.dumps(facts, ensure_ascii=False)}\n\n{package['rubric']}"})
    return {"model": model, "max_tokens": 2000, "messages": [{"role": "user", "content": content}]}


def vlm_review(package: dict, model: str | None = None) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("set ANTHROPIC_API_KEY to run a vision review (or review the package yourself)")
    body = vlm_request(package, model or os.environ.get("OPENADOBE_REVIEW_MODEL", "claude-sonnet-5-5"))
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        reply = json.loads(resp.read())
    text = "".join(b.get("text", "") for b in reply.get("content", []))
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"verdict": "unknown", "raw": text}
