"""Audio repair ops: non-destructive derived audio assets and swapping them into clips."""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from .. import audio_repair, ingest
from ..model import Asset, Project
from ..ops import Ctx, Params, op


class ProcessP(Params):
    asset: str
    chain: list[dict[str, Any]] | None = Field(
        None, description="steps in order: [{'step': 'denoise', 'model': 'cb'}, {'step': 'loudness', "
                          "'target_lufs': -16}]; step names and parameters come from `openadobe audio-steps` "
                          "(audio_repair.describe())")
    preset: str | None = Field(None, description="podcast_voice | noisy_interview | music_master; runs before `chain`")
    name: str | None = Field(None, description="name of the new asset; default '<name> (<preset or steps>)'")


@op("asset.process", ProcessP)
def asset_process(ctx: Ctx, p: Project, a: ProcessP):
    """Repair / master the audio of an asset (works on video assets too: their audio stream is processed).
    Creates a NEW audio asset with provenance; the original is never modified and the result has exactly the same
    length and sample alignment, so clips can later be pointed at it with clip.replace_audio without disturbing
    any cut. Steps: denoise, dehum, declick, deess, highpass, voice_eq, gate, compress, level, loudness."""
    src = p.asset(a.asset)
    if not src.probe.has_audio:
        raise ValueError(f"{src.id} has no audio to process")
    chain: list[dict] = []
    if a.preset:
        chain += audio_repair.preset_chain(a.preset)
    chain += a.chain or []
    if not chain:
        raise ValueError("give a chain or a preset")
    chain = audio_repair.normalize_chain(chain)
    new_id = p.new_id("m")
    media_dir = ctx.root / "media"
    media_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", src.name).strip("_")[:40] or "media"
    out = media_dir / f"{new_id}_{slug}_{a.preset or 'fx'}.wav"
    info = audio_repair.process(ctx.runner, ctx.root, p, src, chain, out)
    probe = ingest.probe(ctx.runner, out)
    label = a.preset or "+".join(s["step"] for s in chain)
    asset = Asset(
        id=new_id, name=a.name or f"{src.name} ({label})", kind="audio", source=str(out), sha256=ingest.sha256(out),
        media=str(out.relative_to(ctx.root)), probe=probe, duration_frames=src.duration_frames,
        provenance={"from": src.id, "process": chain},
    )
    p.assets[new_id] = asset
    result: dict[str, Any] = {"asset": new_id, "frames": asset.duration_frames, "samples": info["samples"]}
    if "measured" in info:
        result["source_lufs"] = float(info["measured"]["input_i"])
        result["source_true_peak_db"] = float(info["measured"]["input_tp"])
    if info.get("normalization_type"):
        result["normalization"] = info["normalization_type"]
    return result


class ReplaceAudioP(Params):
    clip: str = Field(description="an audio clip, or a video clip whose linked audio clips should be replaced")
    asset: str = Field(description="the processed audio asset (from asset.process)")


@op("clip.replace_audio", ReplaceAudioP)
def clip_replace_audio(ctx: Ctx, p: Project, a: ReplaceAudioP):
    """Point audio clips at a processed asset, keeping their timing. Given an audio clip it replaces that clip's
    asset; given a video clip it replaces the linked audio clips (the picture is untouched). The new asset must
    have the same length as the old one or at least cover each clip's source range, so edits and cuts stay
    intact: edit first, then swap in the repaired dialogue."""
    new = p.asset(a.asset)
    if not new.probe.has_audio:
        raise ValueError(f"{new.id} has no audio")
    track, clip = p.find_clip(a.clip)
    if track.kind == "audio":
        targets = [clip]
    else:
        targets = [c for t, c in p.linked(clip) if t.kind == "audio"]
        if not targets:
            raise ValueError(f"{clip.id} has no linked audio clips; pass the audio clip itself")
    changed = []
    for c in targets:
        old = p.asset(c.asset_id)
        if new.duration_frames != old.duration_frames and new.duration_frames < c.src_out:
            raise ValueError(
                f"{new.id} has {new.duration_frames} frames, {old.id} has {old.duration_frames} and {c.id} "
                f"needs source frames up to {c.src_out}")
        changed.append({"clip": c.id, "from": old.id, "to": new.id})
    for c in targets:
        c.asset_id = new.id
    return {"replaced": changed}
