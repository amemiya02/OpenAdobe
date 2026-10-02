"""The operation API (L3): the only way agents change a project.

Every op has a typed parameter model (its JSON Schema is published by ``openadobe ops``),
runs under the project lock, is linted before it is committed, snapshotted for undo and
appended to ``ops.jsonl``. A failing op leaves the project untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import captions as cap
from . import store, timeline
from .engines.runner import Runner
from .ingest import detect_silence, ingest
from .lint import errors, lint
from .model import AUDIO_ROLES, Caption, CaptionStyle, Deliverable, Project, Track, now_iso
from .timebase import Rate, parse_time

Time = int | str


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class Ctx:
    root: Path
    runner: Runner


@dataclass
class OpSpec:
    name: str
    params: type[Params]
    fn: Callable[[Ctx, Project, Any], Any]
    doc: str


OPS: dict[str, OpSpec] = {}


def op(name: str, params: type[Params]):
    def deco(fn):
        OPS[name] = OpSpec(name, params, fn, (fn.__doc__ or "").strip())
        return fn
    return deco


class OpError(Exception):
    def __init__(self, message: str, issues: list[dict] | None = None):
        super().__init__(message)
        self.issues = issues or []


def describe(name: str | None = None) -> list[dict]:
    specs = [OPS[name]] if name else OPS.values()
    return [{"op": s.name, "doc": s.doc, "params": s.params.model_json_schema()} for s in specs]


def _run(ctx: Ctx, project: Project, name: str, params: dict) -> Any:
    if name not in OPS:
        raise OpError(f"unknown op {name!r}; run `openadobe ops` for the list")
    spec = OPS[name]
    try:
        parsed = spec.params.model_validate(params or {})
    except ValidationError as exc:
        raise OpError(f"invalid params for {name}: {exc.errors(include_url=False)}") from None
    try:
        return spec.fn(ctx, project, parsed)
    except (timeline.EditError, KeyError, ValueError) as exc:
        raise OpError(f"{name}: {exc.args[0] if exc.args else exc}") from None


def apply(root: Path, steps: list[tuple[str, dict]], *, runner: Runner | None = None, dry_run: bool = False) -> dict:
    """Apply one or more ops atomically as a single revision."""
    root = Path(root).resolve()
    ctx = Ctx(root, runner or Runner())
    with store.locked(root):
        project = store.load(root)
        before = project.model_copy(deep=True)
        results = [_run(ctx, project, name, params) for name, params in steps]
        dropped = timeline.prune_transitions(project)
        issues = lint(project) + [
            {"level": "warning", "code": "transition.removed", "message": f"{t} removed: its clips are no longer "
             "adjacent", "transition": t} for t in dropped]
        if errors(issues):
            raise OpError("rejected: the edit would leave the project invalid", errors(issues))
        out = {
            "ok": True,
            "revision": project.revision if dry_run else project.revision + 1,
            "results": [_jsonable(r) for r in results],
            "warnings": [i for i in issues if i["level"] == "warning"],
            "dry_run": dry_run,
        }
        if dry_run:
            return out
        store.snapshot(root, before)
        project.revision += 1
        store.save(root, project)
        store.append_log(root, {
            "revision": project.revision, "op": steps[0][0] if len(steps) == 1 else "batch",
            "steps": [{"op": n, "params": p} for n, p in steps], "ts": now_iso(),
        })
        return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _t(project: Project, value: Time) -> int:
    return parse_time(value, project.rate)


# ---- project / assets ------------------------------------------------------------------

class ProjectSetP(Params):
    name: str | None = None
    fps: str | None = Field(None, description="e.g. '30', '25', '30000/1001'; only before media is imported")
    width: int | None = None
    height: int | None = None


@op("project.set", ProjectSetP)
def project_set(ctx: Ctx, p: Project, a: ProjectSetP):
    """Change project name, frame rate (only while empty) or frame size."""
    if a.fps is not None:
        if p.assets:
            raise ValueError("frame rate is fixed once media is imported (mezzanines are conformed to it)")
        p.rate = Rate.parse(a.fps)
    for field in ("name", "width", "height"):
        if (v := getattr(a, field)) is not None:
            setattr(p, field, v)
    return {"name": p.name, "fps": str(p.rate), "size": f"{p.width}x{p.height}"}


class ImportP(Params):
    path: str = Field(description="absolute path of a video, audio or image file")
    name: str | None = None


@op("asset.import", ImportP)
def asset_import(ctx: Ctx, p: Project, a: ImportP):
    """Probe, hash and normalise a media file into the project (CFR mezzanine at project rate)."""
    asset = ingest(ctx.runner, p, ctx.root, Path(a.path), a.name)
    return {"asset": asset.id, "kind": asset.kind, "duration": p.rate.clock(asset.duration_frames),
            "frames": asset.duration_frames, "has_audio": asset.probe.has_audio, "vfr_source": asset.probe.vfr}


class TranscribeP(Params):
    asset: str
    model: str = Field("small", description="faster-whisper model: tiny/base/small/medium/large-v3")
    language: str | None = Field(None, description="ISO code, e.g. 'zh' or 'en'; auto-detect if omitted")
    vocabulary: list[str] = Field(default_factory=list,
                                  description="names/terms the speaker uses (brand, people, jargon) to bias recognition")


@op("asset.transcribe", TranscribeP)
def asset_transcribe(ctx: Ctx, p: Project, a: TranscribeP):
    """Speech-to-text with word timestamps (needs the `asr` extra). Enables captions and text-based cuts."""
    from .asr import transcribe

    asset = p.asset(a.asset)
    if not asset.probe.has_audio:
        raise ValueError(f"{asset.id} has no audio")
    asset.transcript = transcribe(ctx.runner, ctx.root, ctx.root / asset.media, a.model, a.language,
                                  a.vocabulary)
    return {"asset": asset.id, "language": asset.transcript.language, "words": len(asset.transcript.words)}


class FixTranscriptP(Params):
    asset: str
    replace: list[tuple[int, int, str]] = Field(
        description="[first, last, text]: words first..last (inclusive) become one word with this text, "
                    "keeping their timing; use '' to blank a word without cutting media")


@op("asset.fix_transcript", FixTranscriptP)
def asset_fix_transcript(ctx: Ctx, p: Project, a: FixTranscriptP):
    """Correct recognition errors (names, terms). Captions update; word indexes after a merge shift down."""
    from .model import Word

    tr = p.asset(a.asset).transcript
    if not tr:
        raise ValueError(f"{a.asset} has no transcript")
    spans = sorted(a.replace, key=lambda r: r[0], reverse=True)
    for (i, j, _), (i2, j2, _) in zip(spans[1:], spans):
        if j >= i2:
            raise ValueError(f"replacement ranges [{i},{j}] and [{i2},{j2}] overlap")
    for i, j, text in spans:
        if not (0 <= i <= j < len(tr.words)):
            raise ValueError(f"word range [{i}, {j}] outside 0..{len(tr.words) - 1}")
        tr.words[i:j + 1] = [Word(text=text, start=tr.words[i].start, end=tr.words[j].end)]
    return {"asset": a.asset, "words": len(tr.words)}


class TagP(Params):
    asset: str
    tags: list[str]


@op("asset.tag", TagP)
def asset_tag(ctx: Ctx, p: Project, a: TagP):
    """Replace an asset's tags (free-form labels for search: 'b-roll', 'interview', 'logo'…)."""
    p.asset(a.asset).tags = a.tags
    return {"asset": a.asset, "tags": a.tags}


# ---- tracks ----------------------------------------------------------------------------

class TrackAddP(Params):
    kind: Literal["video", "audio"]
    role: Literal["main", "overlay", "dia", "mus", "sfx", "amb"]
    name: str | None = None


@op("track.add", TrackAddP)
def track_add(ctx: Ctx, p: Project, a: TrackAddP):
    """Add a track. Video tracks stack upward (later = on top). Audio roles drive mixing and ducking."""
    prefix = "V" if a.kind == "video" else "A"
    tid = p.new_id(prefix)
    p.tracks.append(Track(id=tid, kind=a.kind, role=a.role, name=a.name or f"{a.role} {tid}"))
    return {"track": tid}


class TrackSetP(Params):
    track: str
    name: str | None = None
    muted: bool | None = None
    gain_db: float | None = None


@op("track.set", TrackSetP)
def track_set(ctx: Ctx, p: Project, a: TrackSetP):
    """Rename, mute/unmute or set the fader (dB) of a track."""
    t = p.track(a.track)
    for field in ("name", "muted", "gain_db"):
        if (v := getattr(a, field)) is not None:
            setattr(t, field, v)
    return {"track": t.id, "muted": t.muted, "gain_db": t.gain_db}


# ---- clips -----------------------------------------------------------------------------

class ClipAddP(Params):
    asset: str
    track: str
    start: Time | None = Field(None, description="timeline position; omitted = end of track")
    src_in: Time = Field(0, description="source in-point")
    duration: Time | None = Field(None, description="omitted = to end of media")
    link_audio: bool | str = Field(True, description="for video with sound: also place its audio, linked, on this "
                                                     "audio track (true = first dialogue track)")
    insert: bool = Field(False, description="push later clips right instead of failing on overlap")


@op("clip.add", ClipAddP)
def clip_add(ctx: Ctx, p: Project, a: ClipAddP):
    """Place media on the timeline. Video with sound gets a linked audio clip (picture and sound stay in sync)."""
    asset = p.asset(a.asset)
    track = p.track(a.track)
    link_track = None
    if track.kind == "video" and asset.probe.has_audio and asset.kind == "video" and a.link_audio is not False:
        if isinstance(a.link_audio, str):
            link_track = a.link_audio
        else:
            dia = p.tracks_by_role("dia")
            if not dia:
                raise ValueError("no dialogue track for the linked audio; add one or pass link_audio=false")
            link_track = dia[0].id
    clips = timeline.add_clip(
        p, a.asset, a.track,
        None if a.start is None else _t(p, a.start),
        _t(p, a.src_in),
        None if a.duration is None else _t(p, a.duration),
        link_track, a.insert,
    )
    return {"clips": [c.id for c in clips], "start": clips[0].start, "end": clips[0].end,
            "at": p.rate.clock(clips[0].start)}


class ClipMoveP(Params):
    clip: str
    start: Time
    track: str | None = None


@op("clip.move", ClipMoveP)
def clip_move(ctx: Ctx, p: Project, a: ClipMoveP):
    """Move a clip (and its linked clips) to a new start, optionally to another track of the same kind."""
    timeline.move(p, a.clip, _t(p, a.start), a.track)
    return {"clip": a.clip}


class ClipTrimP(Params):
    clip: str
    head: Time = Field(0, description="frames removed from the start (negative extends)")
    tail: Time = Field(0, description="frames removed from the end (negative extends)")
    ripple: bool = False


@op("clip.trim", ClipTrimP)
def clip_trim(ctx: Ctx, p: Project, a: ClipTrimP):
    """Trim or extend a clip's edges (linked clips follow). Ripple closes/opens the gap on its tracks."""
    timeline.trim(p, a.clip, _t(p, a.head), _t(p, a.tail), a.ripple)
    _, c = p.find_clip(a.clip)
    return {"clip": c.id, "start": c.start, "duration": c.duration}


class ClipSplitP(Params):
    clip: str
    at: Time = Field(description="timeline position inside the clip")


@op("clip.split", ClipSplitP)
def clip_split(ctx: Ctx, p: Project, a: ClipSplitP):
    """Razor: split a clip and its linked clips at a timeline position."""
    return {"new_clips": timeline.split(p, a.clip, _t(p, a.at))}


class ClipDeleteP(Params):
    clip: str
    ripple: bool = False


@op("clip.delete", ClipDeleteP)
def clip_delete(ctx: Ctx, p: Project, a: ClipDeleteP):
    """Remove a clip and its linked clips; ripple closes the gap on their tracks."""
    return {"deleted": timeline.delete(p, a.clip, a.ripple)}


class ClipSetP(Params):
    clip: str
    gain_db: float | None = None
    fade_in: Time | None = None
    fade_out: Time | None = None
    name: str | None = None
    linked: bool = Field(True, description="apply fades to linked clips too")


@op("clip.set", ClipSetP)
def clip_set(ctx: Ctx, p: Project, a: ClipSetP):
    """Set clip gain (dB, audio only), fade in/out (video fades from/to black) or label."""
    _, clip = p.find_clip(a.clip)
    targets = p.linked(clip) if a.linked else [p.find_clip(a.clip)]
    for t, c in targets:
        if a.fade_in is not None:
            c.fade_in = _t(p, a.fade_in)
        if a.fade_out is not None:
            c.fade_out = _t(p, a.fade_out)
        if a.gain_db is not None and t.kind == "audio":
            c.gain_db = a.gain_db
        if a.name is not None:
            c.name = a.name
    return {"clips": [c.id for _, c in targets]}


class RemoveRangesP(Params):
    clip: str
    ranges: list[tuple[Time, Time]] = Field(description="source-time ranges [start, end) to cut, e.g. [['1.2s','1.9s']]")
    ripple: bool = True


@op("clip.remove_ranges", RemoveRangesP)
def clip_remove_ranges(ctx: Ctx, p: Project, a: RemoveRangesP):
    """Cut source-time ranges out of a clip (and its linked clips); ripple closes the gaps."""
    ranges = [(_t(p, s), _t(p, e)) for s, e in a.ranges]
    return timeline.remove_source_ranges(p, a.clip, ranges, a.ripple)


def _edge_aware(ranges: list[tuple[float, float]], lo: float, hi: float, pad: float) -> list[tuple[float, float]]:
    """Shrink cut ranges by ``pad`` so speech keeps a breath, except where they touch the clip edge."""
    out = []
    for s, e in ranges:
        s2 = s if s <= lo + 1e-3 else s + pad
        e2 = e if e >= hi - 1e-3 else e - pad
        if e2 > s2:
            out.append((s2, e2))
    return out


def _drop_slivers(ranges: list[tuple[int, int]], lo: int, hi: int, min_keep: int) -> list[tuple[int, int]]:
    """Extend cuts so no kept piece is shorter than ``min_keep`` frames."""
    ranges = sorted(ranges)
    out: list[list[int]] = []
    cursor = lo
    for s, e in ranges:
        if s - cursor < min_keep:
            s = cursor if not out else out[-1][0]
            if out:
                out.pop()
        out.append([s, e])
        cursor = e
    if out and hi - cursor < min_keep:
        out[-1][1] = hi
    return [(s, e) for s, e in out]


class CutSilenceP(Params):
    clip: str
    threshold_db: float = -35.0
    min_silence: Time = "0.5s"
    pad: Time = Field("0.12s", description="breath kept on each side of a cut")
    min_keep: Time = Field("0.25s", description="kept pieces shorter than this are cut too")
    ripple: bool = True


@op("edit.cut_silence", CutSilenceP)
def edit_cut_silence(ctx: Ctx, p: Project, a: CutSilenceP):
    """Remove pauses from a dialogue clip (detected on its audio), keeping a natural breath at each cut."""
    _, clip = p.find_clip(a.clip)
    asset = p.asset(clip.asset_id)
    if not asset.probe.has_audio:
        raise ValueError(f"{clip.id}'s media has no audio to analyse")
    fps = float(p.rate.fps)
    lo, hi = clip.src_in / fps, clip.src_out / fps
    silences = detect_silence(ctx.runner, ctx.root, asset, a.threshold_db, p.rate.to_seconds(_t(p, a.min_silence)))
    silences = [(max(s, lo), min(e, hi)) for s, e in silences if min(e, hi) > max(s, lo)]
    cuts = _edge_aware(silences, lo, hi, p.rate.to_seconds(_t(p, a.pad)))
    frames = [(round(s * fps), round(e * fps)) for s, e in cuts]
    frames = _drop_slivers(frames, clip.src_in, clip.src_out, _t(p, a.min_keep))
    if not frames:
        return {"clips": [clip.id], "removed_frames": 0, "silences": 0}
    res = timeline.remove_source_ranges(p, a.clip, frames, a.ripple)
    res["silences"] = len(frames)
    res["removed"] = p.rate.clock(res["removed_frames"])
    return res


class CutWordsP(Params):
    asset: str = Field(description="asset whose transcript the word indexes refer to")
    words: list[tuple[int, int]] = Field(description="inclusive [first, last] word indexes in the asset transcript "
                                                     "(see `openadobe transcript`)")
    pad: Time = Field("0.04s", description="extra material removed around the words at the transcript edges")
    ripple: bool = True


@op("edit.cut_words", CutWordsP)
def edit_cut_words(ctx: Ctx, p: Project, a: CutWordsP):
    """Text-based editing: remove spoken words (flubs, filler, false starts) by transcript index, wherever
    they currently sit in the cut. Clips left with nothing are deleted; picture follows linked sound."""
    p.asset(a.asset)
    tr = p.transcript_for(a.asset)
    if not tr:
        raise ValueError(f"{a.asset} has no transcript; run asset.transcribe first")
    fps = float(p.rate.fps)
    pad = p.rate.to_seconds(_t(p, a.pad))
    ranges = []
    for i, j in a.words:
        if not (0 <= i <= j < len(tr.words)):
            raise ValueError(f"word range [{i}, {j}] outside 0..{len(tr.words) - 1}")
        # cut into the gaps on either side (capped at 0.3 s) so no fragment of the word remains
        start = tr.words[i - 1].end if i > 0 else tr.words[i].start - pad
        end = tr.words[j + 1].start if j + 1 < len(tr.words) else tr.words[j].end + pad
        start = max(start, tr.words[i].start - 0.3)
        end = min(end, tr.words[j].end + 0.3)
        ranges.append((max(0, round(start * fps)), round(end * fps)))

    origin = p.source_asset(a.asset)
    targets = [c.id for t in p.tracks if t.kind == "audio" for c in t.clips
               if p.source_asset(c.asset_id) == origin]  # includes repaired derivatives of the recording
    if not targets:
        raise ValueError(f"{a.asset} has no sound on the timeline")
    removed, deleted, touched = 0, [], []
    for cid in targets:
        try:
            _, clip = p.find_clip(cid)
        except KeyError:
            continue  # went away with an earlier linked deletion
        hits = [(max(s, clip.src_in), min(e, clip.src_out)) for s, e in ranges
                if min(e, clip.src_out) > max(s, clip.src_in)]
        if not hits:
            continue
        covered = set()
        for s, e in hits:
            covered.update(range(s, e))
        if len(covered) >= clip.duration:
            removed += clip.duration
            deleted += timeline.delete(p, cid, a.ripple)
        else:
            res = timeline.remove_source_ranges(p, cid, hits, a.ripple)
            removed += res["removed_frames"]
            touched += res["clips"]
    if not removed:
        raise ValueError("those words are not in the current cut")
    return {"removed_frames": removed, "removed": p.rate.clock(removed), "deleted_clips": deleted,
            "clips": touched, "removed_text": [tr_text(tr, i, j) for i, j in a.words]}


def tr_text(tr, i: int, j: int) -> str:
    from .captions import join_words
    return join_words([w.text for w in tr.words[i:j + 1]])


# ---- captions --------------------------------------------------------------------------

class CaptionsConfigureP(Params):
    enabled: bool | None = None
    source: Literal["transcript", "manual"] | None = Field(
        None, description="transcript = rebuilt from dialogue transcripts on every render (follows the cut)")
    style: dict[str, Any] | None = Field(None, description="'preset' (default, boxed, yellow, shorts, minimal, news, "
                                         "pop, typewriter) and/or fields of CaptionStyle: "
                                         f"{list(CaptionStyle.model_fields)}")


@op("captions.configure", CaptionsConfigureP)
def captions_configure(ctx: Ctx, p: Project, a: CaptionsConfigureP):
    """Enable captions, pick their source and adjust style (font, size, colours, line length, reading speed)."""
    if a.enabled is not None:
        p.captions.enabled = a.enabled
    if a.source is not None:
        p.captions.source = a.source
    if a.style:
        base = p.captions.style
        if "preset" in a.style:
            base = cap.apply_preset(base, a.style["preset"])
        p.captions.style = CaptionStyle.model_validate({**base.model_dump(), **a.style})
    return {"enabled": p.captions.enabled, "source": p.captions.source, "count": len(cap.effective(p))}


class CaptionAddP(Params):
    start: Time
    end: Time
    text: str


@op("captions.add", CaptionAddP)
def captions_add(ctx: Ctx, p: Project, a: CaptionAddP):
    """Add a manual caption at timeline times (shown alongside transcript captions)."""
    c = Caption(id=p.new_id("s"), start=_t(p, a.start), end=_t(p, a.end), text=a.text)
    p.captions.items.append(c)
    p.captions.enabled = True
    return {"caption": c.id}


class CaptionRemoveP(Params):
    caption: str


@op("captions.remove", CaptionRemoveP)
def captions_remove(ctx: Ctx, p: Project, a: CaptionRemoveP):
    """Delete a manual caption."""
    before = len(p.captions.items)
    p.captions.items = [c for c in p.captions.items if c.id != a.caption]
    if len(p.captions.items) == before:
        raise ValueError(f"no manual caption {a.caption!r}")
    return {"removed": a.caption}


class CaptionImportP(Params):
    path: str
    offset: Time = 0


@op("captions.import_srt", CaptionImportP)
def captions_import_srt(ctx: Ctx, p: Project, a: CaptionImportP):
    """Import an SRT file as manual captions (timeline times, shifted by offset)."""
    off = _t(p, a.offset)
    items = cap.parse_srt(Path(a.path).read_text("utf-8-sig"))
    for s, e, text in items:
        p.captions.items.append(Caption(id=p.new_id("s"), start=p.rate.to_frames(s) + off,
                                        end=p.rate.to_frames(e) + off, text=text))
    p.captions.enabled = True
    return {"imported": len(items)}


# ---- mix / delivery --------------------------------------------------------------------

class DuckingP(Params):
    enabled: bool = True
    depth_db: float | None = None
    attack_ms: float | None = None
    release_ms: float | None = None
    merge_gap_ms: float | None = None
    target_roles: list[str] | None = None
    trigger_roles: list[str] | None = None


@op("mix.ducking", DuckingP)
def mix_ducking(ctx: Ctx, p: Project, a: DuckingP):
    """Automatically lower target roles (music) while trigger roles (dialogue) are speaking."""
    d = p.mix.ducking
    for field, v in a.model_dump(exclude_none=True).items():
        setattr(d, field, v)
    for role in d.target_roles + d.trigger_roles:
        if role not in AUDIO_ROLES:
            raise ValueError(f"unknown audio role {role!r}; roles: {AUDIO_ROLES}")
    return d.model_dump()


class MasterP(Params):
    gain_db: float


@op("mix.master", MasterP)
def mix_master(ctx: Ctx, p: Project, a: MasterP):
    """Master fader (dB). Deliverables with a loudness target set their own gain on top of this."""
    p.mix.master_gain_db = a.gain_db
    return {"master_gain_db": a.gain_db}


class DeliverableP(Params):
    name: str = "main"
    settings: dict[str, Any] = Field(default_factory=dict,
                                     description=f"fields of Deliverable: {list(Deliverable.model_fields)}")


@op("deliverable.set", DeliverableP)
def deliverable_set(ctx: Ctx, p: Project, a: DeliverableP):
    """Create or update an output spec (size, codec, loudness target, caption burn-in, SRT sidecar)."""
    base = p.deliverables.get(a.name, Deliverable(name=a.name)).model_dump()
    p.deliverables[a.name] = Deliverable.model_validate({**base, **a.settings, "name": a.name})
    return p.deliverables[a.name].model_dump()


def resolve_time(project: Project, value: Time) -> int:
    """Public alias of the agent time parser for op modules."""
    return _t(project, value)


from . import opsets  # noqa: F401  (registers feature ops)
