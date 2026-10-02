"""The OpenAdobe project model: the single source of truth.

Engine files (MLT XML, ASS, later .blend/.ardour) are compiled from this model and never
edited directly. Every position is an integer frame at ``Project.rate``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .timebase import Rate

SCHEMA_VERSION = 1

TrackKind = Literal["video", "audio"]
TrackRole = Literal["main", "overlay", "dia", "mus", "sfx", "amb"]
AUDIO_ROLES: tuple[str, ...] = ("dia", "mus", "sfx", "amb")
VIDEO_ROLES: tuple[str, ...] = ("main", "overlay")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Probe(BaseModel):
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    fps: str | None = None
    vfr: bool = False
    frames: int | None = None  # video stream frame count when the container reports it
    sample_rate: int | None = None
    channels: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


class Word(BaseModel):
    text: str
    start: float  # source seconds
    end: float


class Transcript(BaseModel):
    language: str
    model: str
    words: list[Word]


class Asset(BaseModel):
    id: str
    name: str
    kind: Literal["video", "audio", "image", "comp"]
    source: str  # original absolute path, informational only
    sha256: str
    media: str  # normalised mezzanine path (or comp render), relative to the project dir
    probe: Probe  # of the mezzanine
    duration_frames: int  # at project rate; 0 for stills (unbounded)
    transcript: Transcript | None = None
    tags: list[str] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)  # shots, beats, faces, loudness… (see analysis.py)
    proxy: str | None = None  # low-res proxy for previews, relative to the project dir
    provenance: dict[str, Any] | None = None  # derived assets: {"from": asset id, "process": [...]}
    comp_id: str | None = None  # kind == "comp": the Comp this asset renders


class Effect(BaseModel):
    """A catalogued effect (see effects.py). Param values may be keyframed: {"0": v0, "30": v1} with
    clip-local frame keys, or "0=v0;30=v1" MLT syntax."""

    id: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class Transition(BaseModel):
    """A transition centred on the cut between adjacent clips ``a`` (left) and ``b`` (right)."""

    id: str
    a: str
    b: str
    duration: int
    kind: str = "dissolve"
    params: dict[str, Any] = Field(default_factory=dict)


class Comp(BaseModel):
    """A composition rendered by a compositing engine and used on the timeline like media (Dynamic Link)."""

    id: str
    engine: Literal["blender", "lottie"]
    template: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)  # template slot -> asset id
    duration: int  # frames
    width: int | None = None  # None = project size
    height: int | None = None
    alpha: bool = True
    render_key: str | None = None  # content hash of the last render (cache key)


class Clip(BaseModel):
    id: str
    asset_id: str
    start: int  # timeline frame
    src_in: int  # source frame (project rate)
    duration: int
    gain_db: float = 0.0
    fade_in: int = 0
    fade_out: int = 0
    link: str | None = None  # clips sharing a link move/trim/split together (picture + its sound)
    name: str | None = None
    speed: float = 1.0  # playback speed; source frames used = round(duration * speed)
    effects: list[Effect] = Field(default_factory=list)

    @property
    def end(self) -> int:
        return self.start + self.duration

    @property
    def src_len(self) -> int:
        return self.duration if self.speed == 1.0 else round(self.duration * self.speed)

    @property
    def src_out(self) -> int:
        return self.src_in + self.src_len


class Track(BaseModel):
    id: str
    kind: TrackKind
    role: TrackRole
    name: str
    muted: bool = False
    gain_db: float = 0.0
    clips: list[Clip] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    fx: list[Effect] = Field(default_factory=list)  # track-level effects (audio: mixer insert chain)

    def sorted_clips(self) -> list[Clip]:
        return sorted(self.clips, key=lambda c: c.start)


class CaptionStyle(BaseModel):
    font: str = "Noto Sans CJK SC"
    size: float = 0.055  # fraction of frame height
    bold: bool = True
    color: str = "FFFFFF"  # RRGGBB
    outline_color: str = "000000"
    outline: float = 0.004  # fraction of frame height
    margin_v: float = 0.07  # fraction of frame height from the bottom
    max_line_units: float = 16  # CJK chars per line (Latin letters count ~0.55)
    max_lines: int = 2
    max_cps: float = 9.0  # reading speed limit used by QC (CJK chars/s)
    strip_end_punct: bool = True
    cjk_pause_as_space: bool = True  # ，、；： inside CJK captions become spaces
    preset: str = "default"
    animation: Literal["none", "fade", "pop", "karaoke", "typewriter"] = "none"
    position: Literal["bottom", "middle", "top"] = "bottom"
    box: bool = False  # opaque box behind text
    box_color: str = "000000"
    box_opacity: float = 0.6
    highlight_color: str = "FFD400"  # karaoke: the word being spoken
    shadow: float = 0.0  # fraction of frame height


class Caption(BaseModel):
    id: str
    start: int
    end: int
    text: str
    words: list[tuple[int, int, str]] | None = None  # timeline-frame word timings (transcript captions)


class Captions(BaseModel):
    enabled: bool = False
    source: Literal["transcript", "manual"] = "transcript"
    style: CaptionStyle = Field(default_factory=CaptionStyle)
    items: list[Caption] = Field(default_factory=list)  # manual captions (timeline frames)


class Ducking(BaseModel):
    enabled: bool = False
    target_roles: list[str] = Field(default_factory=lambda: ["mus"])
    trigger_roles: list[str] = Field(default_factory=lambda: ["dia"])
    depth_db: float = -14.0
    attack_ms: float = 250.0
    release_ms: float = 600.0
    merge_gap_ms: float = 800.0  # trigger gaps shorter than this stay ducked


class Mix(BaseModel):
    ducking: Ducking = Field(default_factory=Ducking)
    master_gain_db: float = 0.0
    engine: Literal["mlt", "ardour"] = "mlt"  # who mixes the final audio
    master_fx: list[Effect] = Field(default_factory=list)


class Deliverable(BaseModel):
    name: str = "main"
    platform: str | None = None  # platform preset this was created from (deliverable.preset)
    width: int | None = None  # None = project size
    height: int | None = None
    reframe: Literal["fit", "fill", "subject"] = "fit"  # other aspect ratios: letterbox, centre crop, follow subject
    vcodec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    acodec: str = "aac"
    abitrate: str = "256k"
    loudness_lufs: float | None = -14.0
    true_peak_db: float = -1.0
    burn_captions: bool = True
    sidecar_srt: bool = True


class Project(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str
    rate: Rate = Field(default_factory=Rate)
    width: int = 1920
    height: int = 1080
    sample_rate: int = 48000
    channels: int = 2
    assets: dict[str, Asset] = Field(default_factory=dict)
    tracks: list[Track] = Field(default_factory=list)
    captions: Captions = Field(default_factory=Captions)
    mix: Mix = Field(default_factory=Mix)
    deliverables: dict[str, Deliverable] = Field(default_factory=lambda: {"main": Deliverable()})
    comps: dict[str, Comp] = Field(default_factory=dict)
    revision: int = 0
    counters: dict[str, int] = Field(default_factory=dict)
    created: str = Field(default_factory=now_iso)
    updated: str = Field(default_factory=now_iso)

    # ---- ids -------------------------------------------------------------------------
    def new_id(self, prefix: str) -> str:
        """Next unused id for a prefix (m=asset, c=clip, V/A=track, L=link, T=transition, e=effect, k=comp…)."""
        taken = self._ids()
        n = self.counters.get(prefix, 0)
        while True:
            n += 1
            if f"{prefix}{n}" not in taken:
                break
        self.counters[prefix] = n
        return f"{prefix}{n}"

    def _ids(self) -> set[str]:
        ids = set(self.assets) | set(self.comps) | {t.id for t in self.tracks}
        ids |= {c.id for c in self.captions.items}
        for t in self.tracks:
            ids |= {c.id for c in t.clips} | {c.link for c in t.clips if c.link} | {tr.id for tr in t.transitions}
            ids |= {e.id for e in t.fx} | {e.id for c in t.clips for e in c.effects}
        ids |= {e.id for e in self.mix.master_fx}
        return ids

    # ---- lookup ----------------------------------------------------------------------
    def track(self, track_id: str) -> Track:
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise KeyError(f"no track {track_id!r}; tracks: {[t.id for t in self.tracks]}")

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for t in self.tracks:
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        raise KeyError(f"no clip {clip_id!r}")

    def asset(self, asset_id: str) -> Asset:
        try:
            return self.assets[asset_id]
        except KeyError:
            raise KeyError(f"no asset {asset_id!r}; assets: {list(self.assets)}") from None

    def linked(self, clip: Clip) -> list[tuple[Track, Clip]]:
        """The clip plus every clip sharing its link group."""
        if clip.link is None:
            return [(self.find_clip(clip.id)[0], clip)]
        return [(t, c) for t in self.tracks for c in t.clips if c.link == clip.link]

    def source_asset(self, asset_id: str) -> str:
        """Follow provenance back to the original recording (repaired audio, mattes → their source)."""
        seen = set()
        while asset_id not in seen:
            seen.add(asset_id)
            prov = self.assets[asset_id].provenance if asset_id in self.assets else None
            if not prov or prov.get("from") not in self.assets:
                return asset_id
            asset_id = prov["from"]
        return asset_id

    def transcript_for(self, asset_id: str) -> Transcript | None:
        """An asset's transcript, or its source's (derived media keeps the source timing)."""
        own = self.assets[asset_id].transcript
        return own if own is not None else self.assets[self.source_asset(asset_id)].transcript

    def tracks_by_role(self, *roles: str) -> list[Track]:
        return [t for t in self.tracks if t.role in roles]

    def duration(self) -> int:
        return max((c.end for t in self.tracks for c in t.clips), default=0)
