"""Emergency exit: export the edit as OpenTimelineIO (Kdenlive ≥ 25.04 imports it natively).

Lossy by design — cuts, gaps and media references survive; fades, gain automation and captions
are carried only as metadata. The OpenAdobe model remains the source of truth.
"""

from __future__ import annotations

from pathlib import Path

from .model import Project


def export_otio(project: Project, root: Path, out: Path) -> Path:
    try:
        import opentimelineio as otio
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("OTIO export needs the otio extra: uv sync --extra otio") from exc

    rate = float(project.rate.fps)

    def rt(frames: int):
        return otio.opentime.RationalTime(frames, rate)

    def span(start: int, dur: int):
        return otio.opentime.TimeRange(rt(start), rt(dur))

    tl = otio.schema.Timeline(name=project.name)
    tl.metadata["openadobe"] = {"revision": project.revision, "fps": str(project.rate)}
    for track in project.tracks:
        kind = otio.schema.TrackKind.Video if track.kind == "video" else otio.schema.TrackKind.Audio
        ot = otio.schema.Track(name=f"{track.id} {track.name}", kind=kind)
        ot.metadata["openadobe"] = {"role": track.role, "muted": track.muted, "gain_db": track.gain_db}
        cursor = 0
        for c in track.sorted_clips():
            if c.start > cursor:
                ot.append(otio.schema.Gap(source_range=span(0, c.start - cursor)))
            asset = project.assets[c.asset_id]
            ref = otio.schema.ExternalReference(
                target_url=str((Path(root) / asset.media).resolve()),
                available_range=span(0, asset.duration_frames) if asset.duration_frames else None,
            )
            clip = otio.schema.Clip(name=c.name or f"{c.id} {asset.name}", media_reference=ref,
                                    source_range=span(c.src_in, c.duration))
            clip.metadata["openadobe"] = c.model_dump()
            ot.append(clip)
            cursor = c.end
        tl.tracks.append(ot)
    out = Path(out).resolve()
    otio.adapters.write_to_file(tl, str(out))
    return out
