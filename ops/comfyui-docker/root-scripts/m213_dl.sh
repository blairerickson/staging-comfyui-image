#!/bin/bash
KEY=$(cat /tmp/civitai.key)
D=/root/ComfyUI/models/loras
LOG=/tmp/m213-lora-dl.log
: > $LOG
dl() {
  fn="$1"; url="$2"
  if [ -s "$D/$fn" ]; then echo "SKIP present $fn $(stat -c%s "$D/$fn")" >> $LOG; return; fi
  code=$(curl -sL --retry 3 --max-time 1800 -H "Authorization: Bearer $KEY" -o "$D/$fn.part" -w "%{http_code}" "$url" 2>>$LOG)
  sz=$(stat -c%s "$D/$fn.part" 2>/dev/null || echo 0)
  if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ]; then mv "$D/$fn.part" "$D/$fn"; echo "OK $fn $code bytes=$sz" >> $LOG
  else echo "FAIL $fn http=$code bytes=$sz body=$(head -c 160 "$D/$fn.part" 2>/dev/null | tr -d '\0')" >> $LOG; rm -f "$D/$fn.part"; fi
}
dl "SumiInk_h3_v2.safetensors"                 "https://civitai.com/api/download/models/3343890?fileId=3230727"
dl "h3_fun_copy_000003500.safetensors"         "https://civitai.com/api/download/models/3270361?fileId=3154202"
dl "HvyMtl Style_000001750.safetensors"        "https://civitai.com/api/download/models/3256846?fileId=3140027"
dl "Heavy_Metal_Krea2_000004000.safetensors"   "https://civitai.com/api/download/models/3180732?fileId=3061286"
echo "DONE $(date -u +%H:%M:%SZ)" >> $LOG
