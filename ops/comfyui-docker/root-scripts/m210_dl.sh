#!/bin/bash
KEY=$(cat /tmp/civitai.key)
D=/root/ComfyUI/models/loras
LOG=/tmp/m210-lora-dl.log
: > $LOG
dl() {
  fn="$1"; url="$2"
  if [ -s "$D/$fn" ]; then echo "SKIP present $fn $(stat -c%s "$D/$fn")" >> $LOG; return; fi
  code=$(curl -sL --retry 3 --max-time 1800 -H "Authorization: Bearer $KEY" -o "$D/$fn.part" -w "%{http_code}" "$url" 2>>$LOG)
  sz=$(stat -c%s "$D/$fn.part" 2>/dev/null || echo 0)
  if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ]; then mv "$D/$fn.part" "$D/$fn"; echo "OK $fn http=$code bytes=$sz" >> $LOG
  else echo "FAIL $fn http=$code bytes=$sz body=$(head -c 160 "$D/$fn.part" 2>/dev/null | tr -d '\0' | head -c 160)" >> $LOG; rm -f "$D/$fn.part"; fi
}
dl "80sFantasyMovie_minimaxh3_3333584_epoch_10.safetensors" "https://civitai.com/api/download/models/3337011?fileId=3223514"
dl "Tentacles-3D_minimaxh3_3344593_epoch_7.safetensors"      "https://civitai.com/api/download/models/3347053?fileId=3234017"
dl "1980s_horror_h3_175.safetensors"                          "https://civitai.com/api/download/models/3342936?fileId=3229696"
dl "1980s_horror_krea2-r64_140.safetensors"                   "https://civitai.com/api/download/models/3126334?fileId=3006795"
dl "Rough_Sketch_Nature_epoch_18.safetensors"                 "https://civitai.com/api/download/models/3348767?fileId=3235860"
dl "H3_STYLE_Somethings_Off_-_by_FoS.safetensors"             "https://civitai.com/api/download/models/3347985?fileId=3235033"
echo "DONE $(date -u +%H:%M:%SZ)" >> $LOG
