# OpenAdobe

**Premiere / After Effects / Audition for agents.** An AI agent edits *real* projects on real engines —
MLT for editing, Blender for compositing, Lottie (glaxnimate) for motion graphics, Ardour for mixing,
ffmpeg for repair and delivery — through a typed, validated, undoable operation API, and checks its own
work by looking at frames and measuring audio. No code-rendered video.

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Plan/status: [`docs/PLAN-P1-P4.md`](docs/PLAN-P1-P4.md) ·
Agent playbook: [`skills/openadobe/SKILL.md`](skills/openadobe/SKILL.md)

## Capabilities
| Layer | What the agent can do | Engine |
|---|---|---|
| **Edit (Pr)** | linked picture+sound, insert/ripple/trim/split/move, speed (pitch kept), silence cut, text-based cut, 17 transitions (dissolve, wipes, push/slide, dips) + audio crossfades | MLT |
| **Look** | 18 catalogued effects with keyframes: colour (eq, lift/gamma/gain, LUT), blur, sharpen, vignette, crop, transform (punch-in, Ken Burns, PIP), chroma key | MLT / avfilter |
| **Graphics (AE)** | Lottie templates: title, lower third, callout, CTA, progress bar, kinetic text (CJK) | glaxnimate in MLT |
| **Compositing (AE)** | Blender comps: keying + despill, 3D title, glow, 3D card — cached by content hash (Dynamic Link) | Blender |
| **Audio (Au)** | repair chains (RNNoise/FFT denoise, dehum, declick, de-ess, EQ, gate, compress, level, loudness) as derived assets; ducking; track/master inserts; Ardour session mix + stems | ffmpeg / Ardour |
| **Captions** | word-level transcripts (faster-whisper) that follow every cut; 8 style presets incl. karaoke | libass |
| **AI** | shot detection, beats/tempo, face/subject tracking, person matting (RVM), beat-synced montage, subject-following vertical reframes | ffmpeg / librosa / OpenCV / ONNX |
| **Deliver** | platform presets (YouTube, Shorts/TikTok/Reels, square, Bilibili, EBU broadcast), two-pass loudness + limiter, SRT, QC on every render, proxies for previews, render-all | MLT + ffmpeg |
| **Review** | frames, time-stamped contact sheets, waveform, spectrogram, loudness, black/freeze/silence, review package (+ optional Claude vision critique) | ffmpeg |
| **Trust** | every op linted before commit, logged, undoable; atomic batches; golden reference render (`doctor --verify`) guards engine upgrades | — |

Interfaces: JSON **CLI**, **MCP** server (images returned as image content), **Skill** playbook,
**OTIO** export as an emergency exit (Kdenlive imports it).

## Quick start
```bash
uv sync --extra asr --extra mcp --extra otio --extra analysis
uv run openadobe doctor --build --verify      # build the version-locked engine image, check the golden render
examples/talking_head_demo.sh                 # P0: voice-over → cut → captions → mix → QC
examples/full_showcase.sh                     # P1–P3: graphics, keying, repair, montage, Ardour mix, Shorts
```

## Engines
`docker/engines.Dockerfile` (Debian trixie): melt 7.30, ffmpeg 7.1, frei0r, Blender 4.3, Ardour 8.12,
RNNoise models, Noto CJK. Directories are bind-mounted at identical paths so compiled engine files work on
host and in the container. `OPENADOBE_ENGINE=local` uses host binaries instead.

## Tests
```bash
uv run pytest -m "not engine"   # fast, pure logic
uv run pytest                   # + end-to-end renders through the real engines (slow)
```
