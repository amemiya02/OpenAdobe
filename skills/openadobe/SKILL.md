---
name: openadobe
description: Edit real video/audio projects at Premiere/After Effects/Audition level with the OpenAdobe CLI/MCP on real engines (MLT, Blender, Ardour, ffmpeg): cutting, transitions, effects, speed, captions, motion graphics, keying, audio repair, mixing, loudness, vertical reframes, beat montages, QC and review. Use for 剪辑口播/vlog/访谈/短视频、加字幕/标题/包装、配BGM、去停顿口误、降噪修音、混音、竖屏重构、卡点、成片审查. Not for generating video with code.
---

# OpenAdobe editing playbook

You edit a **project directory** through `openadobe` (CLI) or the `openadobe mcp` tools. Never edit
`project.oa.json`, `build/`, `cache/` or media by hand — every change is an op: validated, logged, undoable.
Every command prints one JSON document; `"ok": false` means nothing changed — read `error` / `issues`.

## Ground rules
- **Times**: frames (`120`), seconds (`"4.5s"`, `"800ms"`) or timecode (`"00:00:04:15"`).
- **Discover**: `openadobe ops [name]` (ops + JSON Schemas), `openadobe effects`, `openadobe templates`,
  `openadobe presets`, `openadobe audio-steps`.
- **Batch** related edits: `openadobe batch DIR steps.json` → one revision, all-or-nothing.
- **Look before you claim**: `inspect` after edits; `render --preview` + `perceive sheet` while working;
  `render` (QC) + `review` before you call it done. Open the images you get back.
- `undo` reverts the latest op; `log` shows history. Renders are named by revision (`main-r0012.mp4`).

## 1. Assemble (Premiere)
```bash
openadobe init P --fps 30 --size 1920x1080            # V1 video · A1 dialogue · A2 music
openadobe op P asset.import path=/abs/talk.mp4         # → m1 (CFR mezzanine; VFR phone footage is fixed here)
openadobe op P asset.make_proxy                         # fast previews for long/4K media
openadobe op P clip.add asset=m1 track=V1               # picture + linked sound (A1) stay in sync
openadobe op P clip.split clip=c1 at=12s ; clip.trim / clip.move / clip.delete ripple=true
openadobe op P clip.speed clip=c3 speed=0.5             # slow motion (pitch kept), ripple follows
openadobe op P transition.add clip=c1 kind=dissolve duration=0.5s   # also: wipe_left/right/up/down/circle/
                                                        # diagonal, push_*, slide_*, dip_black, dip_white
```
Overlapping transitions need unused media (handles) on both sides; if lint says `transition.handles`, use a
dip or trim the clips. Transitions disappear automatically (warning) when you separate their clips.

## 2. Dialogue edit
```bash
openadobe op P asset.transcribe asset=m1 language=zh 'vocabulary=["品牌名","人名"]'
openadobe transcript P m1                               # read it, fix names:
openadobe op P asset.fix_transcript asset=m1 'replace=[[5,8,"OpenAdobe"]]'
openadobe op P edit.cut_silence clip=c1                 # pauses out, 0.12 s breaths kept
openadobe op P edit.cut_words asset=m1 'words=[[6,9]]'  # flubs, fillers, false starts — as whole thoughts
```

## 3. Look (Lumetri / effects)
`openadobe op P effect.add clip=c4 kind=color.adjust 'params={"saturation":1.15,"contrast":1.08}'`
Kinds: color.adjust, color.lift_gamma_gain, color.lut (.cube), blur, sharpen, vignette, flip, crop,
transform (x/y/scale/rotation/opacity — keyframe as `{"0s":1.0,"3s":1.15}` for punch-ins/Ken Burns),
chroma_key. `scope=track` / `scope=master` for inserts. `effect.set` / `effect.remove`.

## 4. Graphics & compositing (After Effects)
```bash
openadobe templates                                      # lottie: title, lower_third, callout, cta,
                                                         #   progress_bar, kinetic_text; blender: keying,
                                                         #   title_3d, glow, card_3d
openadobe op P comp.add engine=lottie template=lower_third start=3s duration=4s 'params={"name":"张三","role":"导演"}'
openadobe op P comp.add engine=blender template=keying 'inputs={"fg":"m4","bg":"m5"}' start=10s duration=6s
openadobe op P comp.set comp=k1 'params={"name":"李四"}'  # clips update (Dynamic Link); Blender re-renders lazily
#   only comps placed on the timeline are rendered; a comp may take another Blender comp's asset as input
openadobe op P asset.matte asset=m1                       # person cut-out (alpha) → put on an overlay track
```
Comps land on the topmost overlay track (created if needed). Graphics are real Lottie / .blend projects.

## 5. Sound (Audition)
```bash
openadobe op P asset.process asset=m1 preset=podcast_voice     # or chain=[{"step":"denoise"},{"step":"dehum"}…]
openadobe op P clip.replace_audio clip=c2 asset=m7              # swap in the repaired dialogue, timing intact
openadobe op P mix.ducking depth_db=-16                          # music dips under speech
openadobe op P effect.add scope=track track=A1 kind=audio.compressor
openadobe op P mix.engine engine=ardour                          # final mix in an Ardour session (stems too)
#   result.would_fall_back lists what Ardour can't reproduce (clip speed, clip audio fx, denoise…); delivery then
#   mixes with MLT and reports mix_fallback — move such processing to asset.process / track fx to stay on Ardour
```
Steps: denoise (RNNoise/FFT), dehum, declick, deess, highpass, voice_eq, gate, compress, level, loudness.
Processed audio is a new asset (provenance kept); originals are never touched.

## 6. Music-driven & vertical versions
```bash
openadobe op P asset.analyze asset=m9 'what=["beats"]'           # tempo, beats, downbeats
openadobe op P asset.analyze asset=m1 'what=["shots","faces"]'   # shot list, subject track for reframing
openadobe op P edit.montage 'clips=["m2","m3","m4"]' music=m9 every=2   # cuts on every 2nd beat
openadobe op P deliverable.preset name=short preset=shorts       # 1080x1920, follows the subject
openadobe render P --all                                         # master + short + … each with QC
```

## 7. Captions
`captions.configure enabled=true source=transcript 'style={"preset":"shorts"}'` — presets: default, boxed,
yellow, shorts (karaoke word highlight), minimal, news, pop, typewriter. Transcript captions follow every cut.

## 8. Finish
```bash
openadobe render P --preview && openadobe perceive P sheet       # iterate
openadobe render P                                               # loudness 2-pass + limiter + QC report
openadobe review P [--vlm]                                       # contact sheet + facts + rubric
```
QC `fail` checks (format, duration, loudness ±1 LU, true peak, caption timing, lint) must pass. `warn` checks
(unintended black, freeze, silence, caption reading speed) — fix or state why intended.

## Editing judgement
- Pace: remove dead air but keep breaths; cut flubs as whole thoughts; cover jump cuts with b-roll or a
  `transform` punch-in (scale 1.12–1.2).
- Transitions: straight cuts by default; dissolves for time passing; dips for chapter breaks; don't decorate.
- Music: bed −6 dB under talk, ducking −14…−18 dB, fade in 0.5 s / out 1–2 s, end after the last word.
- Captions: fix names via the transcript; shorts use `shorts`/`pop`; landscape `default`/`boxed`.
- Vertical: run face analysis first so `subject` reframing keeps the speaker centred.

## Engine health
`openadobe doctor --verify` renders the golden reference project and compares fingerprints — run after any
engine image change.
