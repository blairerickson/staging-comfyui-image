#!/bin/bash
KEY=$(cat /tmp/civitai.key)
D=/root/ComfyUI/models/loras
LOG=/tmp/m161_download.log
: > $LOG
dl() {
  vid="$1"; out="$2"
  echo "=== $vid -> $out ===" >> $LOG
  code=$(curl -sS -L -w '%{http_code}' -o "$D/$out.tmp" \
    -H "Authorization: Bearer $KEY" \
    "https://civitai.com/api/download/models/$vid" 2>>$LOG)
  sz=$(stat -c%s "$D/$out.tmp" 2>/dev/null || echo 0)
  echo "http=$code size=$sz" >> $LOG
  if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ]; then
    mv "$D/$out.tmp" "$D/$out"
    echo "OK $out ($sz)" >> $LOG
  else
    echo "BODY:" >> $LOG; head -c 400 "$D/$out.tmp" >> $LOG; echo >> $LOG
    rm -f "$D/$out.tmp"
    echo "BLOCKED $out code=$code size=$sz" >> $LOG
  fi
}
dl 3300533 "Excellent-Full-Nude-MiniMax-H3_By-Sarcastic-Tofu_H3.safetensors"
dl 3329929 "SpankMeH3_H3.safetensors"
dl 3310653 "MiniMax-H3-NSFW-Unlocked_H3.safetensors"
dl 3335116 "Tied-Standing_H3.safetensors"
dl 3326785 "Tied-Spreadeagle_H3.safetensors"
dl 3337665 "Strange-Reverie_H3.safetensors"
echo DONE >> $LOG
