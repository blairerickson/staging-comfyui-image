#!/bin/bash
# M223 (Chief, 2026-09-24): add the "Hairy Pussy" LoRA (civitai model 2744200) for BOTH families.
#   version 3352892 (base MiniMax H3)      fileId 3240176 -> hairy_pussy_h3_v1.safetensors
#   version 3086526 (base Krea 2)          fileId 2965976 -> hairy_pussy_krea2_v1.safetensors
# Provenance appended to /root/m223-lora-dl.log (mirrors m213_dl.sh).
KEY=$(cat /tmp/civitai.key)
D=/root/ComfyUI/models/loras
LOG=/root/m223-lora-dl.log
: > $LOG
dl() {
  fn="$1"; url="$2"; model="$3"; ver="$4"; fid="$5"
  if [ -s "$D/$fn" ]; then echo "SKIP present $fn $(stat -c%s "$D/$fn")" >> $LOG; return; fi
  code=$(curl -sL --retry 3 --max-time 1800 -H "Authorization: Bearer $KEY" -o "$D/$fn.part" -w "%{http_code}" "$url" 2>>$LOG)
  sz=$(stat -c%s "$D/$fn.part" 2>/dev/null || echo 0)
  if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ]; then
    mv "$D/$fn.part" "$D/$fn"
    sha=$(sha256sum "$D/$fn" | awk "{print \$1}")
    echo "OK $fn http=$code bytes=$sz model=$model version=$ver fileId=$fid sha256=$sha url=$url" >> $LOG
  else
    echo "FAIL $fn http=$code bytes=$sz body=$(head -c 160 "$D/$fn.part" 2>/dev/null | tr -d "\0")" >> $LOG
    rm -f "$D/$fn.part"
  fi
}
dl "hairy_pussy_h3_v1.safetensors"    "https://civitai.com/api/download/models/3352892?fileId=3240176" 2744200 3352892 3240176
dl "hairy_pussy_krea2_v1.safetensors" "https://civitai.com/api/download/models/3086526?fileId=2965976" 2744200 3086526 2965976
echo "DONE $(date -u +%H:%M:%SZ)" >> $LOG
