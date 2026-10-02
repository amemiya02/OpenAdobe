#!/usr/bin/env bash
# P1–P3 acceptance: one agent session, CLI only, real engines.
#   host interview (noisy, with pauses and a flub) → repair → silence/flub cuts → punch-in → captions
#   → Lottie lower third + Blender 3D title → dip to a beat-synced b-roll montage with a Blender green-screen
#   comp → ducking → Ardour mix → 16:9 master + 9:16 Short (subject-following) → QC → review package.
#
#   examples/full_showcase.sh [WORKDIR]
set -euo pipefail
OA="${OA:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/openadobe}"
W="$(cd "${1:-$(mktemp -d)}" && pwd)"
SRC="$W/src"; P="$W/project"; T0=$(date +%s)
mkdir -p "$SRC"
step() { printf '\n\033[1m== [%ss] %s\033[0m\n' "$(( $(date +%s) - T0 ))" "$*"; }
j() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }
ff() { ffmpeg -nostdin -y -v error "$@"; }
FACE="${FACE:-$HOME/.cache/openadobe/testdata/messi5.jpg}"
[ -f "$FACE" ] || curl -fsSL -o "$FACE" https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg

step "fixtures: noisy host interview, b-roll, music with a beat, green-screen plate"
say -v Tingting -o "$SRC/vo.aiff" "欢迎来到 OpenAdobe 完整演示。[[slnc 1200]] 这一期，嗯，那个，[[slnc 800]] 这一期我们用智能体完成一整套后期。[[slnc 1300]] 包括剪辑，包装，抠像，修音，还有混音。"
ff -i "$SRC/vo.aiff" -f lavfi -i "sine=f=50:sample_rate=48000" -f lavfi -i "anoisesrc=color=pink:amplitude=0.015:sample_rate=48000" \
   -filter_complex "[0]aresample=48000,volume=1.0[v];[1]volume=0.05[h];[2]volume=1[n];[v][h][n]amix=inputs=3:duration=first:normalize=0" "$SRC/vo_noisy.wav"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC/vo_noisy.wav")
ff -loop 1 -i "$FACE" -i "$SRC/vo_noisy.wav" -filter_complex \
   "[0]scale=-2:820,pad=1920:1080:1920-iw-160:(oh-ih)/2:color=0x1b1f2e,zoompan=z='min(zoom+0.0003,1.06)':d=1:s=1920x1080:fps=30,format=yuv420p[v]" \
   -map "[v]" -map 1:a -t "$DUR" -c:v libx264 -c:a aac "$SRC/host.mp4"
ff -f lavfi -i "mandelbrot=s=1920x1080:r=30" -t 8 -c:v libx264 -pix_fmt yuv420p "$SRC/broll1.mp4"
ff -f lavfi -i "life=s=480x270:r=30:mold=10:ratio=0.1:death_color=#102030:life_color=#ffb000,scale=1920:1080:flags=neighbor" -t 8 -c:v libx264 -pix_fmt yuv420p "$SRC/broll2.mp4"
ff -f lavfi -i "testsrc2=s=1920x1080:r=30" -t 8 -c:v libx264 -pix_fmt yuv420p "$SRC/broll3.mp4"
ff -f lavfi -i "aevalsrc='0.55*sin(2*PI*55*t)*exp(-18*mod(t,0.6))+0.10*(sin(2*PI*220*t)+sin(2*PI*277.2*t)+sin(2*PI*329.6*t))':s=48000:d=24" "$SRC/music.wav"
ff -f lavfi -i "color=c=0x00FF00:s=1920x1080:r=30:d=6" -vf \
   "drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial Bold.ttf:text='KEYED':fontcolor=0xE03030:fontsize=260:x=200+t*180:y=360" \
   -c:v libx264 -pix_fmt yuv420p "$SRC/greenscreen.mp4" 2>/dev/null || \
ff -f lavfi -i "color=c=0x00FF00:s=1920x1080:r=30:d=6" -vf "drawbox=x=600:y=300:w=700:h=480:color=0xE03030:t=fill" \
   -c:v libx264 -pix_fmt yuv420p "$SRC/greenscreen.mp4"

step "init + import + proxies"
$OA init "$P" --name showcase --fps 30 --size 1920x1080 >/dev/null
$OA batch "$P" - <<EOF | j 'print([(r["asset"], r["kind"], r["duration"]) for r in d["results"]])'
[{"op":"asset.import","params":{"path":"$SRC/host.mp4","name":"主持人"}},
 {"op":"asset.import","params":{"path":"$SRC/broll1.mp4","name":"b-roll 分形"}},
 {"op":"asset.import","params":{"path":"$SRC/broll2.mp4","name":"b-roll 生命游戏"}},
 {"op":"asset.import","params":{"path":"$SRC/broll3.mp4","name":"b-roll 测试图"}},
 {"op":"asset.import","params":{"path":"$SRC/music.wav","name":"配乐"}},
 {"op":"asset.import","params":{"path":"$SRC/greenscreen.mp4","name":"绿幕"}}]
EOF
$OA op "$P" asset.make_proxy | j 'print(d["results"][0])'

step "analysis: transcript, faces, beats, shots"
$OA op "$P" asset.transcribe asset=m1 language=zh model=small 'vocabulary=["OpenAdobe","抠像","混音","剪辑"]' | j 'print(d["results"][0])'
$OA batch "$P" - <<'EOF' | j 'print(d["results"])'
[{"op":"asset.analyze","params":{"asset":"m1","what":["faces"]}},
 {"op":"asset.analyze","params":{"asset":"m5","what":["beats"]}},
 {"op":"asset.analyze","params":{"asset":"m2","what":["shots"]}}]
EOF

step "agent reads the transcript: fix the name, find the flub"
$OA transcript "$P" m1 > "$W/transcript.json"
python3 - "$W/transcript.json" > "$W/decisions.json" <<'PY'
import json, re, sys
words = json.load(open(sys.argv[1]))["words"]
text = [w["text"].strip() for w in words]
norm = [re.sub(r"[^a-z]", "", t.lower()) for t in text]
fixes, i = [], 0
while i < len(words):
    best, acc = None, ""
    for j in range(i, min(i + 5, len(words))):
        if not norm[j] or not "openadobe".startswith(acc + norm[j]):
            break
        acc += norm[j]; best = j
    if best is not None and len(acc) >= 6 and (best > i or acc != "openadobe"):
        fixes.append([i, best, "OpenAdobe" + re.sub(r"[A-Za-z\s]+", "", words[best]["text"])]); i = best + 1
    else:
        i += 1
core = lambda t: t.strip("，。,.!！?？ ")
cut, i = [], 0
while i < len(text):
    if core(text[i]) in {"嗯", "呃", "那个", "啊"}:
        j = i
        while j + 1 < len(text) and core(text[j + 1]) in {"嗯", "呃", "那个", "啊"}:
            j += 1
        k = i
        while k > 0 and not text[k - 1].endswith(("。", "！", "？", ".", "!", "?")) and words[i]["start"] - words[k - 1]["start"] < 2.5:
            k -= 1
        cut.append([k, j]); i = j + 1
    else:
        i += 1
# indexes shift after a merge: apply cuts first (in original indexes), then the name fix
print(json.dumps({"fix": fixes, "cut": cut}))
PY
cat "$W/decisions.json"

step "dialogue: place, repair audio, cut pauses and the flub"
$OA op "$P" clip.add asset=m1 track=V1 | j 'print(d["results"][0])'
$OA op "$P" edit.cut_silence clip=c1 | j 'print({k: v for k, v in d["results"][0].items() if k != "clips"})'
CUT=$(j 'print(json.dumps(d["cut"]))' < "$W/decisions.json")
[ "$CUT" != "[]" ] && $OA op "$P" edit.cut_words asset=m1 words="$CUT" | j 'print(d["results"][0]["removed_text"])'
FIX=$(j 'print(json.dumps(d["fix"]))' < "$W/decisions.json")
[ "$FIX" != "[]" ] && $OA op "$P" asset.fix_transcript asset=m1 replace="$FIX" >/dev/null
$OA op "$P" asset.process asset=m1 preset=noisy_interview name=主持人-修复 | j 'print(d["results"][0])'
REP=$($OA inspect "$P" | j 'print([a["asset"] for a in d["assets"] if a["name"] == "主持人-修复"][0])')
for C in $($OA inspect "$P" | j 'print(" ".join(c["clip"] for c in d["tracks"][0]["clips"]))'); do
  $OA op "$P" clip.replace_audio clip="$C" asset="$REP" >/dev/null
done
HOST_END=$($OA inspect "$P" | j 'print(d["duration_frames"])')
SECOND=$($OA inspect "$P" | j 'print(sorted(d["tracks"][0]["clips"], key=lambda c: c["start"])[1]["clip"])')
LASTHOST=$($OA inspect "$P" | j 'print(sorted(d["tracks"][0]["clips"], key=lambda c: c["start"])[-1]["clip"])')

step "look + graphics: punch-in, captions, Lottie lower third, Blender 3D title"
$OA batch "$P" - <<EOF | j 'print("rev", d["revision"], [w["code"] for w in d["warnings"]])'
[{"op":"effect.add","params":{"clip":"$SECOND","kind":"transform","params":{"scale":{"0":1.0,"0.4s":1.15}}}},
 {"op":"effect.add","params":{"clip":"$SECOND","kind":"color.adjust","params":{"contrast":1.06,"saturation":1.1}}},
 {"op":"captions.configure","params":{"enabled":true,"source":"transcript","style":{"preset":"default"}}},
 {"op":"comp.add","params":{"engine":"lottie","template":"lower_third","start":"1.2s","duration":"4s","params":{"name":"梅西","role":"演示主持人"}}},
 {"op":"track.add","params":{"kind":"video","role":"overlay","name":"Titles"}},
 {"op":"comp.add","params":{"engine":"blender","template":"title_3d","track":"V3","start":0,"duration":"2.5s","params":{"text":"OpenAdobe","subtitle":"智能体后期"}}}]
EOF

step "b-roll montage on the beat + green-screen comp + dip from the interview"
$OA op "$P" edit.montage 'clips=["m2","m3","m4"]' music=m5 start=$HOST_END length=8s every=2 | j 'print(d["results"][0])'
$OA op "$P" transition.add clip="$LASTHOST" kind=dip_black duration=0.6s | j 'print(d["results"][0])'
$OA op "$P" comp.add engine=blender template=keying 'inputs={"fg":"m6"}' start=$((HOST_END + 90)) duration=3s | j 'print(d["results"][0])'
for C in $($OA inspect "$P" | j "print(' '.join(c['clip'] for c in d['tracks'][0]['clips'] if c['start'] >= $HOST_END))"); do
  $OA op "$P" effect.add clip="$C" kind=color.adjust 'params={"saturation":1.25,"contrast":1.05}' >/dev/null
done

step "mix: ducking, voice chain on the dialogue track, Ardour as mix engine"
$OA batch "$P" - <<'EOF' | j 'print("rev", d["revision"])'
[{"op":"mix.ducking","params":{"depth_db":-14}},
 {"op":"effect.add","params":{"scope":"track","track":"A1","kind":"audio.compressor"}},
 {"op":"mix.engine","params":{"engine":"ardour"}},
 {"op":"deliverable.preset","params":{"name":"main","preset":"youtube"}},
 {"op":"deliverable.preset","params":{"name":"short","preset":"shorts"}}]
EOF
$OA inspect "$P" | j 'print({"duration": d["duration"], "tracks": [(t["track"], t["role"], len(t["clips"])) for t in d["tracks"]], "captions": [c["text"] for c in d["captions"]["items"]], "issues": [i["code"] for i in d["issues"]]})'

step "preview → look"
$OA render "$P" --preview | j 'print({k: d[k] for k in ("render", "seconds")}, d.get("comps_rendered"))'
$OA perceive "$P" sheet --n 16 | j 'print(d["image"])'

step "deliver: 16:9 master + 9:16 Short (Ardour mix, loudness, QC)"
$OA render "$P" --all | tee "$W/renders.json" | j 'print({n: {k: r.get(k) for k in ("render", "size", "mix_engine", "reframed_clips", "qc", "seconds")} for n, r in d["renders"].items()}, "all_pass:", d["all_pass"])'

step "review package"
SHORT=$(j 'print(d["renders"]["short"]["render"])' < "$W/renders.json")
$OA review "$P" | j 'print({k: d[k] for k in ("contact_sheet", "qc", "package")})'
$OA perceive "$P" sheet --media "$SHORT" --n 8 | j 'print("short sheet:", d["image"])'
echo; echo "workdir: $W  (total $(( $(date +%s) - T0 ))s)"
