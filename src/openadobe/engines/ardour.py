"""Compile the project model to an Ardour session and mix it down headlessly.

Headless toolchain (Ardour 8.12 in the engine image)::

    ardour8-lua  = /usr/lib/ardour8/luasession  (Lua scripting without a GUI; dummy audio backend)

* ``ardour_lua/build.lua``   creates the session skeleton through the Lua API: one stereo audio track per project
  audio track, one bus per role present (dia/mus/sfx/amb) feeding Master, faders, mute and the ACE plugin chains.
* this module then adds the sources and regions to the saved ``.ardour`` XML. Ardour's Lua API cannot create a
  source from a file without the GUI editor (``Editor:do_import``), so that part is written directly. It is still a
  one-way compile from the model: the session is regenerated on every build and never read back into the model.
* ``ardour_lua/export.lua``  re-opens the finished session (listing tracks and regions as proof that it loads) and
  renders mix and stems with ``Session:simple_export`` (24-bit WAV at session rate, from an export preset
  staged into the container's HOME).  Stems are exported from the same session with the other role bus faders at -inf,
  which makes each stem exactly that bus's output at master (master stays at unity).

Gain automation (clip gain, fades, ducking) comes from ``envelope.clip_envelope`` and is written as region gain
envelopes; the track fader carries ``Track.gain_db`` (so it is taken out of the envelope again).
"""

from __future__ import annotations

import json
import math
import shlex
import shutil
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

from .. import effects
from ..envelope import FLOOR_DB, clip_envelope
from ..model import AUDIO_ROLES, Clip, Effect, Project, Track
from .runner import EngineError, Runner

RATE = 48000
SUPERCLOCK = 282_240_000  # Temporal::superclock_ticks_per_second in Ardour 8
TICKS = SUPERCLOCK // RATE  # superclock ticks per sample (5880)
LUA_DIR = Path(__file__).with_name("ardour_lua")
EXPORT_PRESET_ID = "3f1c2d54-77aa-4b0e-8e57-6b0c3b5e9a02"
BUS_NAMES = {role: role.upper() for role in AUDIO_ROLES}
RUNS: list[dict[str, Any]] = []  # every ardour8-lua invocation of this process (script, attempt, seconds, exit)
ENV_STEP = RATE // 100  # gain ramps are written as points at most 10 ms apart


EXPORT_FORMAT = """<?xml version="1.0" encoding="UTF-8"?>
<ExportFormatSpecification name="OpenAdobe WAV 24" id="8d3c6f10-0a54-4c1e-9f6a-0a0de0b1ad01" with-cue="false" with-toc="false" with-mp4chaps="false" command="" analyse="false" soundcloud-upload="false">
  <Encoding id="F_WAV" type="T_Sndfile" extension="wav" name="WAV" has-sample-format="true" channel-limit="256"/>
  <SampleRate rate="SR_Session"/>
  <SRCQuality quality="SRC_SincBest"/>
  <EncodingOptions>
    <Option name="sample-format" value="SF_24"/>
    <Option name="dithering" value="D_None"/>
    <Option name="tag-metadata" value="false"/>
    <Option name="tag-support" value="false"/>
    <Option name="broadcast-info" value="false"/>
  </EncodingOptions>
  <Processing>
    <Normalize enabled="false" loudness="no" dbfs="0" lufs="-23" dbtp="-1"/>
    <Silence>
      <Start>
        <Trim enabled="false"/>
        <Add enabled="false">
          <Duration format="Timecode" hours="0" minutes="0" seconds="0" frames="0"/>
        </Add>
      </Start>
      <End>
        <Trim enabled="false"/>
        <Add enabled="false">
          <Duration format="Timecode" hours="0" minutes="0" seconds="0" frames="0"/>
        </Add>
      </End>
    </Silence>
  </Processing>
</ExportFormatSpecification>
"""
EXPORT_PRESET = f"""<?xml version="1.0" encoding="UTF-8"?>
<ExportPreset id="{EXPORT_PRESET_ID}" name="OpenAdobe WAV 24">
  <ExportFormat id="8d3c6f10-0a54-4c1e-9f6a-0a0de0b1ad01"/>
  <ExportFilename>
    <Folder relative="1" path="export"/>
    <Field name="label" enabled="0"/>
    <Field name="session" enabled="0"/>
    <Field name="snapshot" enabled="0"/>
    <Field name="timespan" enabled="0"/>
    <Field name="revision" enabled="0"/>
    <Field name="time" enabled="0" value="T_None"/>
    <Field name="date" enabled="0" value="D_None"/>
  </ExportFilename>
</ExportPreset>
"""


# ---- model -> Ardour plugin mapping -----------------------------------------------------------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _plugin(kind: str, plugin_id: str, params: dict[int, float], note: str = "") -> dict[str, Any]:
    return {"type": kind, "id": plugin_id, "params": params, "note": note}


def _fx_highpass(p: dict) -> dict:
    # ffmpeg highpass p=2 is one biquad = 12 dB/oct
    return _plugin("lua", "ACE High/Low Pass Filter", {0: 1, 1: p["freq"], 2: 0.707, 3: 0})


def _fx_lowpass(p: dict) -> dict:
    return _plugin("lua", "ACE High/Low Pass Filter", {0: 0, 3: 1, 4: p["freq"], 5: 0.707})


def _fx_eq_band(p: dict) -> dict:
    q = p["q"]
    octaves = 2 / math.log(2) * math.asinh(1 / (2 * q))  # bandwidth in octaves for a given Q
    # band 2 of ACE EQ; the other bands stay at 0 dB (flat)
    return _plugin("lv2", "urn:ardour:a-eq", {
        5: _clamp(p["freq"], 20, 20000), 6: _clamp(p["gain_db"], -20, 20), 7: _clamp(octaves, 0.1, 4)})


def _fx_compressor(p: dict) -> dict:
    return _plugin("lv2", "urn:ardour:a-comp#stereo", {
        0: _clamp(p["attack_ms"], 0.1, 100), 1: _clamp(p["release_ms"], 1, 2000), 2: 2.8,
        3: _clamp(p["ratio"], 1, 20), 4: _clamp(p["threshold_db"], -60, 0), 5: _clamp(p["makeup_db"], 0, 30)})


def _fx_limiter(p: dict) -> dict:
    return _plugin("lv2", "urn:ardour:a-comp#stereo", {
        0: 0.1, 1: 50, 2: 0, 3: 20, 4: _clamp(p["ceiling_db"], -60, 0), 5: 0},
        note="approximation: Ardour has no brickwall limiter; ACE Compressor 20:1 with 0.1 ms attack")


def _fx_echo(p: dict) -> dict:
    wet = p["decay"] / (0.8 + p["decay"])
    gain = 20 * math.log10(max(p["mix"] * (0.8 + p["decay"]), 1e-4))
    return _plugin("lv2", "urn:ardour:a-delay", {
        4: 0, 5: _clamp(p["delay_ms"], 1, 8000), 7: 100 * wet, 8: 0, 9: 20000, 10: _clamp(gain, -60, 20)},
        note="approximation of ffmpeg aecho (single tap) with ACE Delay")


def _fx_gain(p: dict) -> dict | None:
    if isinstance(p["db"], dict):
        return None
    return _plugin("lua", "ACE Amplifier", {0: _clamp(p["db"], -20, 20)})


_FX_MAP = {
    "audio.highpass": _fx_highpass, "audio.lowpass": _fx_lowpass, "audio.eq_band": _fx_eq_band,
    "audio.compressor": _fx_compressor, "audio.limiter": _fx_limiter, "audio.echo": _fx_echo,
    "audio.gain": _fx_gain,
}
_FX_SKIP = {"audio.denoise": "Ardour ships no denoiser (use asset.process denoise on the asset instead)"}


def map_effects(fx: Iterable[Effect], where: str, report: dict[str, list]) -> list[dict]:
    """Catalogue audio effects -> Ardour plugin descriptions; fills report['mapped'] / report['skipped']."""
    out: list[dict] = []
    for e in fx:
        if not e.enabled:
            continue
        if e.kind in _FX_SKIP or e.kind not in _FX_MAP or effects.CATALOG.get(e.kind) is None \
                or effects.CATALOG[e.kind].target != "audio":
            report["skipped"].append({"where": where, "kind": e.kind, "reason": _FX_SKIP.get(e.kind, "no Ardour mapping")})
            continue
        plug = _FX_MAP[e.kind](effects.validate(e.kind, e.params))
        if plug is None:
            report["skipped"].append({"where": where, "kind": e.kind, "reason": "keyframed parameter"})
            continue
        entry = {"where": where, "kind": e.kind, "plugin": plug["id"]}
        if plug["note"]:
            entry["note"] = plug["note"]
        report["mapped"].append(entry)
        out.append(plug)
    return out


# ---- helpers ---------------------------------------------------------------------------------------------------
def _lua(value: Any) -> str:
    """Python value -> Lua literal (the container's Lua has no JSON)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(value, dict):
        return "{" + ", ".join(f"[{_lua(k)}]={_lua(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_lua(v) for v in value) + "}"
    if value is None:
        return "nil"
    raise TypeError(f"cannot serialise {type(value).__name__} to Lua")


def _samples(project: Project, frames: int) -> int:
    """Project frames -> samples at 48 kHz (rounded from the exact rational position)."""
    return round(frames * RATE * project.rate.den / project.rate.num)


def _sc(samples: int) -> str:
    return f"a{samples * TICKS}"


def _ardour_dir(root: Path) -> Path:
    return Path(root).resolve() / "build" / "ardour"


def _roles_present(project: Project) -> list[str]:
    used = {t.role for t in project.tracks if t.kind == "audio"}
    return [r for r in AUDIO_ROLES if r in used]


def _audio_tracks(project: Project) -> list[Track]:
    return [t for t in project.tracks if t.kind == "audio"]


def _run_lua(runner: Runner, oa: Path, script: str, *args: Path, mounts: Iterable[Path],
             limit: int = 150, attempts: int = 3, fresh: Path | None = None) -> str:
    """Run one of the driver scripts in ardour8-lua. The export preset directory is copied into the container's
    HOME first (HOME is a throw-away directory in the engine image).

    Two luasession quirks are handled here: it intermittently dies with "FATAL: exception not rethrown" (exit 133)
    while the process tears down, after the script finished and the session was saved, so success is the script's own
    OA-DONE marker; and an export now and then stalls forever, so every run is killed after ``limit`` seconds (inside
    the container, so it cannot leak) and retried."""
    shell = ('mkdir -p "$HOME/.config/ardour8/export" && cp "$0"/*.format "$0"/*.preset '
             '"$HOME/.config/ardour8/export/" && exec timeout -s KILL "$1" ardour8-lua "${@:2}"')
    if fresh is not None:
        # removed inside the container on every attempt: Docker Desktop's file sharing can keep a host-deleted
        # directory "existing" for the next container, and create_session refuses an existing directory
        shell = f"rm -rf {shlex.quote(str(fresh))} && " + shell
    cmd = ["bash", "-c", shell, str(oa / "export"), str(limit),
           str(oa / "lua" / script), str(oa / "lua" / "common.lua"), *map(str, args)]
    failure = ""
    for attempt in range(1, attempts + 1):
        t0 = time.monotonic()
        proc = runner.run(cmd, mounts=[oa, *mounts], timeout=limit + 60, check=False)
        RUNS.append({"script": script, "attempt": attempt, "seconds": round(time.monotonic() - t0, 1),
                     "exit": proc.returncode, "done": "OA-DONE" in proc.stdout})
        if "OA-DONE" in proc.stdout and "OA-ERROR" not in proc.stderr:
            return proc.stdout
        errors = [ln for ln in (proc.stderr + proc.stdout).splitlines() if "OA-ERROR" in ln or "Error" in ln]
        failure = f"ardour {script} failed (exit {proc.returncode}): " + " | ".join(errors[-5:])
        if any("OA-ERROR" in ln for ln in errors):  # a script error is deterministic: do not retry
            break
    raise EngineError(failure)


def _stage_scripts(oa: Path) -> Path:
    """Copy the driver scripts and export preset into ``oa`` (moved into the session dir once it exists)."""
    (oa / "lua").mkdir(parents=True, exist_ok=True)
    (oa / "export").mkdir(parents=True, exist_ok=True)
    for f in LUA_DIR.glob("*.lua"):
        shutil.copy2(f, oa / "lua" / f.name)
    (oa / "export" / "oa24.format").write_text(EXPORT_FORMAT, "utf-8")
    (oa / "export" / "oa24.preset").write_text(EXPORT_PRESET, "utf-8")
    return oa


# ---- sources ---------------------------------------------------------------------------------------------------
def prepare_sources(runner: Runner, root: Path, project: Project) -> dict[str, Path]:
    """48 kHz 24-bit stereo WAV of every asset used on an audio track, cached by asset id + content hash."""
    root = Path(root).resolve()
    cache = _ardour_dir(root) / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for track in _audio_tracks(project):
        for clip in track.clips:
            asset = project.asset(clip.asset_id)
            if clip.asset_id in out or not asset.probe.has_audio:
                continue
            wav = cache / f"{asset.id}-{asset.sha256[:12]}.wav"
            if not wav.exists() or wav.stat().st_size == 0:
                tmp = wav.with_suffix(".part.wav")
                runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(root / asset.media), "-vn",
                            "-map", "0:a:0", "-ac", "2", "-ar", str(RATE), "-c:a", "pcm_s24le", str(tmp)],
                           mounts=[cache], ro=[root])
                tmp.replace(wav)
            out[clip.asset_id] = wav
    return out


# ---- session XML -----------------------------------------------------------------------------------------------
def _envelope_points(project: Project, track: Track, clip: Clip, length: int) -> list[tuple[int, float]]:
    """Region-relative (sample, linear gain) points; the fader carries Track.gain_db so it is removed here."""
    env = clip_envelope(project, track, clip)
    pts = [(_samples(project, x + clip.start) - _samples(project, clip.start), max(db - track.gain_db, FLOOR_DB))
           for x, db in env]
    pts.append((length, pts[-1][1]))  # hold the last value to the end of the region
    dense: list[tuple[int, float]] = [pts[0]]
    for (x0, d0), (x1, d1) in pairwise(pts):
        if abs(d1 - d0) > 1e-6 and x1 - x0 > ENV_STEP:
            n = math.ceil((x1 - x0) / ENV_STEP)
            dense += [(x0 + round((x1 - x0) * i / n), d0 + (d1 - d0) * i / n) for i in range(1, n)]
        dense.append((x1, d1))
    out: list[tuple[int, float]] = []
    for x, d in dense:
        if out and x <= out[-1][0]:  # strictly increasing times
            x = out[-1][0] + 1
        out.append((x, 10 ** (d / 20)))
    return out


def _add_regions(project: Project, session_file: Path, sources: dict[str, Path]) -> dict[str, int]:
    tree = ET.parse(session_file)
    root = tree.getroot()
    ids = [int(e.get("id")) for e in root.iter() if (e.get("id") or "").isdigit()]
    counter = [max(ids, default=0) + 1000]

    def new_id() -> str:
        counter[0] += 1
        return str(counter[0])

    sources_el = root.find("Sources")
    assert sources_el is not None
    route_ids = {r.get("name"): r.get("id") for r in root.iter("Route")}
    playlists = {p.get("orig-track-id"): p for p in root.iter("Playlist")}
    src_ids: dict[Path, tuple[str, str]] = {}
    counts: dict[str, int] = {}
    for track in _audio_tracks(project):
        playlist = playlists[route_ids[track.id]]
        counts[track.id] = 0
        for layer, clip in enumerate(track.sorted_clips()):
            wav = sources.get(clip.asset_id)
            if wav is None:
                continue
            if wav not in src_ids:
                pair = (new_id(), new_id())
                for ch, sid in enumerate(pair):
                    ET.SubElement(sources_el, "Source", {
                        "name": str(wav), "type": "audio", "flags": "", "id": sid, "natural-position": "a0",
                        "captured-for": "", "channel": str(ch), "origin": str(wav), "gain": "1"})
                src_ids[wav] = pair
            s0, s1 = src_ids[wav]
            pos = _samples(project, clip.start)
            length = _samples(project, clip.end) - pos
            start = _samples(project, clip.src_in)
            env = _envelope_points(project, track, clip, length)
            flat = all(abs(g - 1.0) < 1e-6 for _, g in env)
            reg = ET.SubElement(playlist, "Region", {
                "name": clip.id, "muted": "0", "opaque": "1", "locked": "0", "video-locked": "0", "automatic": "0",
                "whole-file": "0", "import": "0", "external": "1", "sync-marked": "0", "left-of-split": "0",
                "right-of-split": "0", "hidden": "0", "position-locked": "0", "valid-transients": "0",
                "start": _sc(start), "length": f"{_sc(length)}@{_sc(pos)}", "sync-position": "a0",
                "ancestral-start": "a0", "ancestral-length": "a0@a0", "stretch": "1", "shift": "1",
                "layering-index": str(layer), "tags": "", "contents": "0", "rgroup": "0",
                "envelope-active": "0" if flat else "1", "default-fade-in": "0", "default-fade-out": "0",
                "fade-in-active": "0", "fade-out-active": "0", "fade-before-fx": "0", "scale-amplitude": "1",
                "id": new_id(), "type": "audio", "first-edit": "nothing", "source-0": s0, "source-1": s1,
                "master-source-0": s0, "master-source-1": s1, "channels": "2"})
            lst = ET.SubElement(ET.SubElement(reg, "Envelope"), "AutomationList", {
                "automation-id": "envelope", "id": new_id(), "interpolation-style": "Exponential",
                "time-domain": "AudioTime", "state": "Off"})
            ET.SubElement(lst, "events").text = "".join(f"{_sc(x)} {g:.9g}\n" for x, g in env)
            counts[track.id] += 1
    tree.write(session_file, encoding="UTF-8", xml_declaration=True)
    return counts


# ---- public API ------------------------------------------------------------------------------------------------
def plan(project: Project) -> tuple[list[dict], list[dict], dict[str, list]]:
    """Pure: the tracks and master chain the session will get, plus the mapped/skipped report."""
    report: dict[str, list] = {"mapped": [], "skipped": []}
    tracks = []
    for t in _audio_tracks(project):
        tracks.append({"name": t.id, "role": t.role, "gain_db": t.gain_db, "muted": t.muted,
                       "fx": map_effects(t.fx, t.id, report)})
    for t in _audio_tracks(project):
        for c in t.clips:
            if c.speed != 1.0:
                report["skipped"].append({"where": c.id, "kind": "clip.speed",
                                          "reason": "Ardour regions are not time-stretched here; played at 1.0"})
            if not project.asset(c.asset_id).probe.has_audio:
                report["skipped"].append({"where": c.id, "kind": "clip", "reason": "asset has no audio"})
            for e in c.effects:
                if e.enabled and e.kind.startswith("audio."):
                    report["skipped"].append({"where": c.id, "kind": e.kind, "reason": "clip-level audio effects "
                                              "are not compiled by the Ardour adapter (use Track.fx)"})
    master_fx = map_effects(project.mix.master_fx, "master", report)
    return tracks, master_fx, report


def losses(project: Project) -> list[dict]:
    """What an Ardour mix would drop or play differently from the model (empty = the Ardour mix is faithful)."""
    return [s for s in plan(project)[2]["skipped"] if s["reason"] != "asset has no audio"]


def build_session(runner: Runner, root: Path, project: Project, name: str = "mix") -> Path:
    """Compile the model into a fresh Ardour session under root/build/ardour/<name>-r<revision>/ and return it.

    Always regenerated, never edited by hand. ``session_report(session_dir)`` lists what was mapped and skipped."""
    root = Path(root).resolve()
    if project.duration() <= 0:
        raise ValueError("the timeline is empty")
    base = _ardour_dir(root)
    session = base / f"{name}-r{project.revision}"
    if session.exists():
        shutil.rmtree(session)  # generated output only: build/ardour/<name>-r<rev>
    base.mkdir(parents=True, exist_ok=True)  # Ardour insists on creating the session directory itself
    sources = prepare_sources(runner, root, project)

    tracks, master_fx, report = plan(project)
    spec = {"dir": str(session), "name": name, "rate": RATE, "master_db": project.mix.master_gain_db,
            "buses": _roles_present(project), "bus_names": BUS_NAMES, "tracks": tracks, "master_fx": master_fx}
    staging = base / f".{session.name}.oa"
    if staging.exists():
        shutil.rmtree(staging)
    oa = _stage_scripts(staging)
    spec_file = oa / "build_spec.lua"
    spec_file.write_text("return " + _lua(spec) + "\n", "utf-8")
    _run_lua(runner, oa, "build.lua", spec_file, mounts=[base], fresh=session)
    _add_regions(project, session / f"{name}.ardour", sources)
    shutil.move(str(staging), session / "oa")
    oa = session / "oa"
    (oa / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
    return session


def session_report(session_dir: Path) -> dict[str, list]:
    """What the last build mapped to Ardour plugins and what it skipped: {"mapped": [...], "skipped": [...]}."""
    return json.loads((Path(session_dir) / "oa" / "report.json").read_text("utf-8"))


def inspect_session(runner: Runner, session_dir: Path, name: str = "mix") -> dict[str, Any]:
    """Re-open the session headlessly and list its tracks, buses and regions (proof that it is a valid session)."""
    session = Path(session_dir).resolve()
    spec = {"dir": str(session), "name": name, "jobs": [], "out_dir": str(session / "export"), "start": 0, "stop": 0,
            "preset": EXPORT_PRESET_ID}
    spec_file = session / "oa" / "inspect_spec.lua"
    spec_file.write_text("return " + _lua(spec) + "\n", "utf-8")
    return _parse_listing(_run_lua(runner, session / "oa", "export.lua", spec_file, mounts=[session.parent]))


def _parse_listing(stdout: str) -> dict[str, Any]:
    """Parse the OA-TRACK / OA-BUS / OA-REGION lines printed by export.lua after re-opening the session."""
    tracks: dict[str, dict] = {}
    regions: list[dict] = []
    buses: dict[str, list[str]] = {}

    def fields(parts: list[str]) -> dict[str, str]:
        return dict(p.split("=", 1) for p in parts)

    def dests(value: str) -> list[str]:
        return value.split(",") if value else []

    for line in stdout.splitlines():
        parts = line.split()
        if line.startswith("OA-TRACK "):
            f = fields(parts[2:])
            tracks[parts[1]] = {"regions": int(f["regions"]), "gain": float(f["gain"]),
                                "muted": f["muted"] == "1", "to": dests(f.get("to", ""))}
        elif line.startswith("OA-BUS "):
            buses[parts[1]] = dests(fields(parts[2:]).get("to", ""))
        elif line.startswith("OA-REGION "):
            f = fields(parts[3:])
            regions.append({"track": parts[1], "region": parts[2], "position": int(f["pos"]),
                            "length": int(f["len"]), "start": int(f["start"])})
    return {"tracks": tracks, "buses": buses, "regions": regions}


def _wav_frames(runner: Runner, wav: Path) -> int:
    proc = runner.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                       "stream=sample_rate,channels,bits_per_raw_sample,duration_ts", "-of",
                       "default=nw=1", str(wav)], mounts=[wav.parent])
    info = dict(line.split("=", 1) for line in proc.stdout.split())
    if info.get("sample_rate") != str(RATE) or info.get("channels") != "2":
        raise EngineError(f"unexpected export format for {wav}: {info}")
    return int(info["duration_ts"])


def mixdown(runner: Runner, root: Path, project: Project, out_dir: Path, stems: bool = True) -> dict:
    """Build the session and export it headlessly.

    Returns ``{"mix": wav, "stems": {role: wav}, "session": dir, "seconds": float, "listing": {...},
    "report": {...}}``. WAVs are 48 kHz 24-bit stereo, exactly ``project.duration()`` frames long."""
    t0 = time.monotonic()
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    session = build_session(runner, root, project)
    roles = _roles_present(project)
    # every job states every bus, so jobs do not inherit each other's fader moves
    jobs = [{"name": "mix", "off": {BUS_NAMES[o]: False for o in roles}}]
    if stems:
        jobs += [{"name": f"stem_{r}", "off": {BUS_NAMES[o]: o != r for o in roles}} for r in roles]
    total = _samples(project, project.duration())
    spec = {"dir": str(session), "name": "mix", "jobs": jobs, "out_dir": str(out_dir), "start": 0, "stop": total,
            "preset": EXPORT_PRESET_ID}
    spec_file = session / "oa" / "export_spec.lua"
    spec_file.write_text("return " + _lua(spec) + "\n", "utf-8")
    for f in [*out_dir.glob("stem_*.wav"), out_dir / "mix.wav"]:
        f.unlink(missing_ok=True)  # so a failed or no-op export can never pass off an older file as this one
    stdout = _run_lua(runner, session / "oa", "export.lua", spec_file, mounts=[_ardour_dir(root), out_dir])
    result: dict[str, Any] = {"mix": out_dir / "mix.wav", "stems": {}, "session": session}
    wavs = [result["mix"]]
    if stems:
        result["stems"] = {role: out_dir / f"stem_{role}.wav" for role in roles}
        wavs += list(result["stems"].values())
    for wav in wavs:
        if not wav.exists():
            raise EngineError(f"Ardour did not write {wav}")
        got = _wav_frames(runner, wav)
        if got != total:
            raise EngineError(f"{wav.name} has {got} samples, expected {total}")
    result["seconds"] = time.monotonic() - t0
    result["listing"] = _parse_listing(stdout)
    result["report"] = session_report(session)
    return result
