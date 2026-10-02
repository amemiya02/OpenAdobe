"""Golden reference: a deterministic project that exercises the whole pipeline, rendered and fingerprinted.

``openadobe doctor --verify`` rebuilds it with the current engine image and compares the render's
fingerprint with ``golden_ref.json``. An engine upgrade (new MLT/ffmpeg/fonts) that silently changes
cuts, transitions, effects, captions, graphics or the mix fails here before it reaches real projects.
``--record`` writes a new reference after a deliberate change.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from . import deliver, ops, store
from .engines.runner import Runner
from .timebase import Rate

REF_FILE = Path(__file__).with_name("golden_ref.json")
SAMPLE_FRAMES = (0, 10, 30, 52, 60, 75, 95, 110, 130, 150)
TOLERANCE = {"rgb": 12.0, "luma_grid": 14.0, "lufs": 0.8, "seconds_rms": 1.5}


def _fixtures(runner: Runner, d: Path) -> None:
    def ff(*args: str) -> None:
        runner.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], mounts=[d])

    ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=4", "-f", "lavfi", "-i",
       "aevalsrc='0.3*sin(2*PI*330*t)*(lt(mod(t,2),1.4))':s=48000:d=4", "-shortest", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "a.mp4"))
    ff("-f", "lavfi", "-i", "color=c=0x2060c0:s=320x180:r=25:d=4", "-f", "lavfi", "-i",
       "sine=f=550:sample_rate=48000:d=4", "-shortest", "-vf", "drawbox=x=120:y=60:w=80:h=60:color=yellow:t=fill",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "b.mp4"))
    ff("-f", "lavfi", "-i", "aevalsrc='0.2*sin(2*PI*220*t)+0.2*sin(2*PI*277*t)':s=48000:d=10", str(d / "m.wav"))


def build_reference(runner: Runner, workdir: Path) -> Path:
    d = Path(workdir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    _fixtures(runner, d)
    root = d / "golden"
    store.init(root, "golden", Rate(num=25, den=1), 320, 180)
    ops.apply(root, [("asset.import", {"path": str(d / n)}) for n in ("a.mp4", "b.mp4", "m.wav")])
    ops.apply(root, [
        ("clip.add", {"asset": "m1", "track": "V1", "duration": 60}),
        ("clip.add", {"asset": "m2", "track": "V1", "src_in": 10, "duration": 50}),
        ("clip.add", {"asset": "m1", "track": "V1", "src_in": 45, "duration": 50}),
        ("transition.add", {"clip": "c1", "duration": 12, "kind": "dissolve"}),
        ("transition.add", {"clip": "c3", "duration": 10, "kind": "wipe_left"}),
        ("effect.add", {"clip": "c3", "kind": "color.adjust", "params": {"saturation": 0.2, "contrast": 1.2}}),
        ("effect.add", {"clip": "c5", "kind": "transform", "params": {"scale": {"0": 1.0, "49": 1.3}}}),
        ("clip.set", {"clip": "c1", "fade_in": 8}),
        ("clip.add", {"asset": "m3", "track": "A2", "start": 0, "duration": 160}),
        ("clip.set", {"clip": "c7", "gain_db": -8, "fade_out": 20}),
        ("mix.ducking", {"depth_db": -12}),
        ("captions.add", {"start": 20, "end": 70, "text": "金标准 Golden"}),
        ("comp.add", {"engine": "lottie", "template": "lower_third", "start": 90, "duration": 60,
                      "params": {"name": "OpenAdobe", "role": "回归"}}),
        ("deliverable.set", {"settings": {"loudness_lufs": -16.0, "crf": 12}}),
    ])
    return root


def fingerprint(runner: Runner, root: Path, media: Path) -> dict:
    root = Path(root).resolve()
    sel = "+".join(f"eq(n\\,{n})" for n in SAMPLE_FRAMES)
    proc = subprocess.run(runner.command(
        ["ffmpeg", "-v", "error", "-i", str(media), "-vf", f"select='{sel}',scale=4:3:flags=area,format=rgb24",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-"], mounts=[root]), capture_output=True, check=False)
    raw = proc.stdout
    frames = []
    for i in range(len(raw) // 36):
        px = raw[i * 36:(i + 1) * 36]
        rgb = [sum(px[c::3]) / 12 for c in range(3)]
        grid = [round(0.2126 * px[j] + 0.7152 * px[j + 1] + 0.0722 * px[j + 2], 1) for j in range(0, 36, 3)]
        frames.append({"rgb": [round(v, 1) for v in rgb], "luma_grid": grid})
    loud = subprocess.run(runner.command(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(media), "-vn", "-af",
         ("ebur128=peak=true,asetnsamples=48000,astats=metadata=1:reset=1,"
          "ametadata=print:key=lavfi.astats.Overall.RMS_level"), "-f", "null", "-"], mounts=[root]),
        capture_output=True, text=True, check=False).stderr
    rms = [round(float(v), 1) if v != "-inf" else -120.0 for v in re.findall(r"RMS_level=(-?[\d.]+|-inf)", loud)]
    summary = loud[loud.rfind("Summary:"):]
    lufs = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", summary)[-1])
    count = subprocess.run(runner.command(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
                                           "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(media)],
                                          mounts=[root]), capture_output=True, text=True, check=False)
    return {"frames": frames, "seconds_rms": rms, "lufs": lufs, "frame_count": int(count.stdout.strip() or 0)}


def compare(fp: dict, ref: dict) -> list[str]:
    problems = []
    if fp["frame_count"] != ref["frame_count"]:
        problems.append(f"frame count {fp['frame_count']} != {ref['frame_count']}")
    if len(fp["frames"]) != len(ref["frames"]):
        problems.append(f"sampled {len(fp['frames'])} frames, reference has {len(ref['frames'])}")
    for n, a, b in zip(SAMPLE_FRAMES, fp["frames"], ref["frames"], strict=False):
        for key in ("rgb", "luma_grid"):
            worst = max(abs(x - y) for x, y in zip(a[key], b[key], strict=True))
            if worst > TOLERANCE[key]:
                problems.append(f"frame {n} {key} differs by {worst:.1f} (> {TOLERANCE[key]})")
    if abs(fp["lufs"] - ref["lufs"]) > TOLERANCE["lufs"]:
        problems.append(f"loudness {fp['lufs']} vs {ref['lufs']} LUFS")
    for i, (a, b) in enumerate(zip(fp["seconds_rms"], ref["seconds_rms"], strict=False)):
        if max(a, b) > -60 and abs(a - b) > TOLERANCE["seconds_rms"]:
            problems.append(f"second {i} RMS {a} vs {b} dB")
    return problems


def run(runner: Runner, workdir: Path, record: bool = False) -> dict:
    started = time.time()
    root = build_reference(runner, workdir)
    res = deliver.render(root, runner=runner)
    fp = fingerprint(runner, root, Path(res["render"]))
    fp["engines"] = runner.versions()
    if record:
        REF_FILE.write_text(json.dumps(fp, indent=1, ensure_ascii=False), "utf-8")
        return {"ok": True, "recorded": str(REF_FILE), "qc_pass": res["qc"]["pass"],
                "seconds": round(time.time() - started, 1)}
    if not REF_FILE.exists():
        return {"ok": False, "error": "no golden reference recorded; run `openadobe doctor --verify --record`"}
    ref = json.loads(REF_FILE.read_text("utf-8"))
    problems = compare(fp, ref)
    if not res["qc"]["pass"]:
        problems.append(f"QC failed: {res['qc']['failed']}")
    return {"ok": not problems, "problems": problems, "render": res["render"], "reference_engines": ref.get("engines"),
            "engines": fp["engines"], "seconds": round(time.time() - started, 1)}
