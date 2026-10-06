#!/bin/bash
: "${CIVITAI_API_KEY:?CIVITAI_API_KEY must be set}"
KEY="$CIVITAI_API_KEY"
DST=/root/ComfyUI/models/loras
LOG=/root/krea_asset_logs/m188b_fetch.log
: > "$LOG"
get() { # fn versionId fileId
  fn=$1; vid=$2; fid=$3
  [ -s "$DST/$fn" ] && { echo "SKIP exists $fn" >>"$LOG"; return; }
  echo "GET $fn" >>"$LOG"
  curl -sL --fail --retry 3 --max-time 1800 -H "Authorization: Bearer $KEY" -o "$DST/$fn.part" \
    "https://civitai.com/api/download/models/$vid?fileId=$fid" >>"$LOG" 2>&1 \
    && mv "$DST/$fn.part" "$DST/$fn" && echo "OK $fn $(stat -c%s "$DST/$fn")" >>"$LOG" \
    || { echo "FAIL $fn" >>"$LOG"; rm -f "$DST/$fn.part"; }
}
get donnie-darko-2001.safetensors 3345732 3232670
get cruel-intentions-1999.safetensors 3345686 3232618
get action_minimax_t2v_spa_nk_rev1.safetensors 3329929 3216002
echo DONE >>"$LOG"
