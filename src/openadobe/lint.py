"""Model validation run after every operation. Errors block the operation; warnings are reported."""

from __future__ import annotations

from .model import AUDIO_ROLES, VIDEO_ROLES, Project


def issue(level: str, code: str, message: str, **where) -> dict:
    return {"level": level, "code": code, "message": message, **where}


def lint(project: Project) -> list[dict]:
    out: list[dict] = []
    seen_ids: set[str] = set()
    for t in project.tracks:
        if t.kind == "video" and t.role not in VIDEO_ROLES:
            out.append(issue("error", "track.role", f"video track {t.id} cannot have role {t.role}", track=t.id))
        if t.kind == "audio" and t.role not in AUDIO_ROLES:
            out.append(issue("error", "track.role", f"audio track {t.id} cannot have role {t.role}", track=t.id))
        prev = None
        for c in t.sorted_clips():
            if c.id in seen_ids:
                out.append(issue("error", "clip.duplicate_id", f"duplicate clip id {c.id}", clip=c.id))
            seen_ids.add(c.id)
            if c.asset_id not in project.assets:
                out.append(issue("error", "clip.missing_asset", f"{c.id} references unknown asset {c.asset_id}",
                                 clip=c.id))
                continue
            asset = project.assets[c.asset_id]
            if c.duration <= 0:
                out.append(issue("error", "clip.empty", f"{c.id} has duration {c.duration}", clip=c.id))
            if c.start < 0:
                out.append(issue("error", "clip.negative_start", f"{c.id} starts at {c.start}", clip=c.id))
            from .timeline import producer_in, producer_length
            overrun = (c.src_out > asset.duration_frames if c.speed == 1.0
                       else producer_in(c) + c.duration > producer_length(c, asset))
            if c.src_in < 0 or (asset.kind != "image" and overrun):
                out.append(issue("error", "clip.source_range",
                                 f"{c.id} uses source {c.src_in}..{c.src_out} but {asset.id} has "
                                 f"{asset.duration_frames} frames", clip=c.id))
            if c.fade_in < 0 or c.fade_out < 0 or c.fade_in + c.fade_out > c.duration:
                out.append(issue("error", "clip.fades", f"{c.id} fades exceed its duration", clip=c.id))
            if t.kind == "audio" and not asset.probe.has_audio:
                out.append(issue("error", "clip.no_audio", f"{c.id} on audio track but {asset.id} has no audio",
                                 clip=c.id))
            if t.kind == "video" and asset.kind == "audio":
                out.append(issue("error", "clip.no_video", f"{c.id} on video track but {asset.id} is audio",
                                 clip=c.id))
            if prev is not None and c.start < prev.end:
                out.append(issue("error", "track.overlap",
                                 f"{c.id} overlaps {prev.id} on {t.id} ({c.start} < {prev.end})",
                                 track=t.id, clip=c.id))
            prev = c if prev is None or c.end > prev.end else prev

    # transitions: adjacent clips, room on both sides, media handles for overlapping kinds
    from .timeline import halves, needs_handles
    for t in project.tracks:
        clips = {c.id: c for c in t.clips}
        used_head: dict[str, int] = {}
        used_tail: dict[str, int] = {}
        for tr in t.transitions:
            a, b = clips.get(tr.a), clips.get(tr.b)
            where = {"track": t.id, "transition": tr.id}
            if a is None or b is None or a.end != b.start:
                out.append(issue("error", "transition.not_adjacent", f"{tr.id}: {tr.a} and {tr.b} are not adjacent",
                                 **where))
                continue
            if tr.duration < 2:
                out.append(issue("error", "transition.duration", f"{tr.id} must be at least 2 frames", **where))
                continue
            h1, h2 = halves(tr.duration)
            used_tail[a.id] = h1
            used_head[b.id] = h2
            if needs_handles(tr.kind):
                for c, need, side in ((a, h2, "after"), (b, h1, "before")):
                    asset = project.assets.get(c.asset_id)
                    if asset is None or asset.kind == "image":
                        continue
                    if side == "after":
                        avail = int(asset.duration_frames / c.speed) - round(c.src_in / c.speed) - c.duration
                    else:
                        avail = round(c.src_in / c.speed)
                    if avail < need:
                        out.append(issue("error", "transition.handles",
                                         f"{tr.id} ({tr.kind}) needs {need} unused frames {side} {c.id}, media has "
                                         f"{max(avail, 0)}; shorten it, trim the clip, or use a dip transition",
                                         **where))
        for cid, c in clips.items():
            if used_head.get(cid, 0) + used_tail.get(cid, 0) >= c.duration:
                out.append(issue("error", "transition.overlap", f"transitions consume all of {cid}", track=t.id,
                                 clip=cid))

    for comp in project.comps.values():
        for slot, aid in comp.inputs.items():
            if aid not in project.assets:
                out.append(issue("error", "comp.input", f"{comp.id} input {slot!r} references unknown asset {aid}",
                                 comp=comp.id))
        if comp.engine == "blender" and comp.render_key is None:
            out.append(issue("warning", "comp.stale", f"{comp.id} will render before the next delivery",
                             comp=comp.id))

    # linked clips must stay in sync (same timing) — anything else is a desync
    groups: dict[str, list] = {}
    for t in project.tracks:
        for c in t.clips:
            if c.link:
                groups.setdefault(c.link, []).append(c)
    for link, clips in groups.items():
        if len({(c.start, c.src_in, c.duration, c.speed) for c in clips}) > 1:
            out.append(issue("warning", "link.desync", f"linked clips {[c.id for c in clips]} are out of sync",
                             link=link))

    total = project.duration()
    mains = project.tracks_by_role("main")
    if total and mains:
        covered = sorted((c.start, c.end) for t in mains for c in t.clips)
        cursor = 0
        for a, b in covered:
            if a > cursor:
                out.append(issue("warning", "video.gap",
                                 f"no main video {project.rate.clock(cursor)}–{project.rate.clock(a)}",
                                 start=cursor, end=a))
            cursor = max(cursor, b)
        if cursor < total:
            out.append(issue("warning", "video.gap",
                             f"no main video {project.rate.clock(cursor)}–{project.rate.clock(total)}",
                             start=cursor, end=total))

    caps = sorted(project.captions.items, key=lambda c: c.start)
    for i, cap in enumerate(caps):
        if cap.end <= cap.start:
            out.append(issue("error", "caption.empty", f"caption {cap.id} has no duration", caption=cap.id))
        if not cap.text.strip():
            out.append(issue("error", "caption.text", f"caption {cap.id} has no text", caption=cap.id))
        if i and cap.start < caps[i - 1].end:
            out.append(issue("warning", "caption.overlap", f"caption {cap.id} overlaps {caps[i - 1].id}",
                             caption=cap.id))
    return out


def errors(issues: list[dict]) -> list[dict]:
    return [i for i in issues if i["level"] == "error"]
