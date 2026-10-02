# P1–P4 execution plan (2026-10-02)

Status keys: [ ] todo · [~] in progress · [x] done (verified by tests / renders)

## Contracts (owner: main session) — done before any delegation
- Model additions (`model.py`): `Clip.speed`, `Clip.effects`, `Track.transitions`, `Track.fx`, `Asset.analysis`,
  `Asset.proxy`, `Asset.provenance`, `Asset.comp_id`, `Project.comps`, `Mix.engine`, `Deliverable.reframe/preset`,
  caption style presets/animation, `Caption.words`.
- Ops are modular: `ops.py` keeps the registry + core ops; feature ops live in `src/openadobe/opsets/<area>.py`
  and register with the same `@op` decorator. Each delegated area owns exactly one opset file + its engine module.
- Engine image includes Ardour 8.12, Blender 4.3.2, RNNoise models (docker/engines.Dockerfile).
- Sub-agents: Sonnet 5.5; must not edit model.py / ops.py / mlt.py / timeline.py / Dockerfile / pyproject.

## P1 — Premiere-level editing + Audition-level audio
- [x] T1 Transitions: dissolve, wipes, push/slide, dip to black/white; audio crossfades on linked sound (main)
- [x] T2 Effects catalog → MLT filters, keyframable params, `effect.*` ops, `effects` catalog listing (main)
- [x] T3 Clip speed (timewarp) incl. caption/ripple math (main)
- [x] T4 Caption style presets + animations (pop, karaoke word highlight, typewriter, boxed) (main)
- [x] T5 Audio repair chains as derived assets: denoise (afftdn/arnndn), de-ess, hum removal, voice EQ,
       leveler, loudness; `asset.process` (sub-agent: audio)
- [x] T6 Ardour adapter: compile model → Ardour session headless (roles → tracks/busses, regions, gains,
       fades, plugins), export stems + mix; `Mix.engine="ardour"` in delivery (sub-agent: ardour)

## P2 — After Effects-level compositing / motion graphics
- [x] T7 Comp infrastructure / Dynamic Link: `Project.comps`, comp assets, content-hash render cache,
       auto (re)render before delivery, `comp.*` ops (main)
- [x] T8 Lottie motion-graphics templates rendered natively by MLT's glaxnimate producer: title, lower third,
       callout, CTA, progress bar, kinetic text (sub-agent: lottie)
- [x] T9 Blender comps: chroma key + despill + background, 3D title, camera shake/glow looks, rendered to
       lossless FFV1 RGBA .mkv (sub-agent: blender; MLT flattens ProRes 4444 alpha)

## P3 — AI operations
- [x] T10 Analysis: shots (scdet), beats/onsets/tempo (librosa), faces/subject track (OpenCV YuNet) →
        `Asset.analysis` (sub-agent: analysis)
- [x] T11 Person matting (Robust Video Matting ONNX) → alpha matte asset; background replace (sub-agent: analysis)
- [x] T12 Reframe: deliverables with other aspect ratios follow the subject (main)
- [x] T13 Beat-synced montage op (main)
- [x] T14 Review package + optional VLM critique (main)
- [x] T15 Deliverable presets + render-all (main)

## P4 — Hardening
- [x] T16 Proxies for fast previews (main)
- [x] T17 Golden regression suite: reference renders with metric fingerprints; `doctor --verify` (main)
- [x] T18 Docs/skill update, independent review, fix findings (main + reviewer)
