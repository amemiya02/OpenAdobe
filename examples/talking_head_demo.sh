#!/usr/bin/env bash
# P0 acceptance: 口播粗剪 + 字幕 + BGM 闪避 → mp4 + 自动 QC, driven only through the CLI
# (exactly what an agent would call). Needs macOS `say` for the synthetic voice-over.
#
#   examples/talking_head_demo.sh [WORKDIR]
set -euo pipefail
OA="${OA:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/openadobe}"
W="$(cd "${1:-$(mktemp -d)}" && pwd)"
SRC="$W/src"; P="$W/project"
mkdir -p "$SRC"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

step "fixtures: voice-over with pauses and a flub, picture, music bed"
say -v Tingting -o "$SRC/vo.aiff" \
  "大家好，欢迎来到 OpenAdobe。[[slnc 1400]] 今天我们，嗯，那个，[[slnc 900]] 今天我们来聊一聊，怎么让智能体真正地剪辑视频。[[slnc 1600]] 它操作的是真实的剪辑引擎，而不是用代码画出来的动画。[[slnc 1200]] 我们下期再见。" \
  2>/dev/null || say -o "$SRC/vo.aiff" "大家好，欢迎来到 OpenAdobe。[[slnc 1400]] 今天我们来聊一聊，怎么让智能体真正地剪辑视频。"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC/vo.aiff")
ffmpeg -v error -y -f lavfi -i "testsrc2=size=1920x1080:rate=30000/1001" -i "$SRC/vo.aiff" -t "$DUR" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$SRC/talk.mp4"
ffmpeg -v error -y -f lavfi -i "aevalsrc='0.12*(sin(2*PI*196*t)+sin(2*PI*247*t)+sin(2*PI*294*t))*(0.6+0.4*sin(2*PI*0.5*t))':s=44100:d=40" \
  "$SRC/bed.wav"

step "init + import (CFR mezzanines)"
$OA init "$P" --name demo --fps 30 --size 1920x1080 >/dev/null
IMP=$($OA batch "$P" - <<EOF
[{"op":"asset.import","params":{"path":"$SRC/talk.mp4","name":"口播"}},
 {"op":"asset.import","params":{"path":"$SRC/bed.wav","name":"BGM"}}]
EOF
)
echo "$IMP" | field 'd["results"]'
TALK=$(echo "$IMP" | field 'd["results"][0]["asset"]'); BED=$(echo "$IMP" | field 'd["results"][1]["asset"]')

step "transcribe (word timestamps)"
$OA op "$P" asset.transcribe asset="$TALK" language=zh model=small 'vocabulary=["OpenAdobe"]' | field 'd["results"][0]'

step "review transcript: fix a misrecognised name (agent spots it while reading)"
$OA transcript "$P" "$TALK" > "$W/transcript.json"
FIX=$(python3 - "$W/transcript.json" <<'PY'
import json, re, sys
words = json.load(open(sys.argv[1]))["words"]
norm = [re.sub(r"[^a-z]", "", w["text"].lower()) for w in words]
fixes = []
i = 0
while i < len(words):
    best, acc = None, ""
    for j in range(i, min(i + 5, len(words))):
        if not norm[j] or not "openadobe".startswith(acc + norm[j]):
            break
        acc += norm[j]
        best = j
    if best is not None and len(acc) >= 6 and (best > i or acc != "openadobe"):
        tail = re.sub(r"[A-Za-z\s]+", "", words[best]["text"])  # keep trailing punctuation
        fixes.append([i, best, "OpenAdobe" + tail])
        i = best + 1
    else:
        i += 1
print(json.dumps(fixes[:1]))
PY
)
echo "agent decision: fix $FIX"
[ "$FIX" != "[]" ] && $OA op "$P" asset.fix_transcript asset="$TALK" replace="$FIX" | field 'd["results"][0]' || true

step "place picture + linked sound, cut pauses"
$OA op "$P" clip.add asset="$TALK" track=V1 | field 'd["results"][0]'
$OA op "$P" edit.cut_silence clip=c1 | field '{k: v for k, v in d["results"][0].items() if k != "clips"}'

step "text-based cut: remove the flub (stand-in for the agent reading the transcript)"
$OA transcript "$P" "$TALK" > "$W/transcript.json"
RANGES=$(python3 - "$W/transcript.json" <<'PY'
import json, sys
words = json.load(open(sys.argv[1]))["words"]
text = [w["text"].strip() for w in words]
FILLERS = {"嗯", "呃", "那个", "啊"}
core = lambda t: t.strip("，。,.!！?？ ")
cut = []
i = 0
while i < len(text):
    if core(text[i]) in FILLERS:
        j = i
        while j + 1 < len(text) and core(text[j + 1]) in FILLERS:
            j += 1
        # false start: words since the last sentence end, right before the filler
        k = i
        while k > 0 and not text[k - 1].endswith(("。", "！", "？", ".", "!", "?")) and words[i]["start"] - words[k - 1]["start"] < 2.0:
            k -= 1
        cut.append([k, j])
        i = j + 1
    else:
        i += 1
print(json.dumps(cut))
PY
)
echo "agent decision: cut words $RANGES"
if [ "$RANGES" != "[]" ]; then
  $OA op "$P" edit.cut_words asset="$TALK" words="$RANGES" | field 'd["results"][0] if d.get("ok") else d'
fi

step "music bed, ducking, captions from transcript, fades"
END=$($OA inspect "$P" | field 'd["duration_frames"]')
FIRST_V=$($OA inspect "$P" | field 'sorted(d["tracks"][0]["clips"], key=lambda c: c["start"])[0]["clip"]')
LAST_V=$($OA inspect "$P" | field 'sorted(d["tracks"][0]["clips"], key=lambda c: c["start"])[-1]["clip"]')
$OA batch "$P" - <<EOF | field '{"revision": d["revision"], "warnings": [w["code"] for w in d["warnings"]]}'
[{"op":"clip.add","params":{"asset":"$BED","track":"A2","start":0,"duration":$((END + 45))}},
 {"op":"clip.set","params":{"clip":"$FIRST_V","fade_in":"0.5s"}},
 {"op":"clip.set","params":{"clip":"$LAST_V","fade_out":"0.5s"}},
 {"op":"mix.ducking","params":{"depth_db":-16}},
 {"op":"captions.configure","params":{"enabled":true,"source":"transcript"}}]
EOF
BEDCLIP=$($OA inspect "$P" | field 'd["tracks"][2]["clips"][0]["clip"]')
$OA op "$P" clip.set clip="$BEDCLIP" fade_in=0.5s fade_out=1.5s gain_db=-6 >/dev/null

step "inspect"
$OA inspect "$P" | field '{"duration": d["duration"], "captions": [c["text"] for c in d["captions"]["items"]], "issues": d["issues"]}'

step "render (loudness two-pass, limiter, burned captions, SRT) + QC"
$OA render "$P" | tee "$W/render.json" | field '{k: d[k] for k in ("render", "master_gain_db", "qc", "seconds")}'

step "perceive: contact sheet + loudness"
$OA perceive "$P" sheet --n 12 | field 'd["image"]'
$OA perceive "$P" loudness | field '{k: d[k] for k in ("integrated_lufs", "true_peak_db", "range_lu")}'
echo; echo "workdir: $W"
