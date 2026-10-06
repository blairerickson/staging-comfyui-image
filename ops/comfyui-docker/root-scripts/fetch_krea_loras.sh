#!/bin/bash
# Krea2 LoRA fetch with SHA256 verification. Staging only.
: "${CIVITAI_API_KEY:?CIVITAI_API_KEY must be set}"
DST=/root/ComfyUI/models/loras
mkdir -p "$DST" /root/krea_asset_logs
LOG=/root/krea_asset_logs/lora_fetch.log
: > "$LOG"
while IFS='|' read -r fn url sha sz; do
  [ -z "$fn" ] && continue
  out="$DST/$fn"
  if [ -s "$out" ]; then
    echo "$(date -u +%FT%TZ) SKIP exists $fn" >> "$LOG"; continue
  fi
  echo "$(date -u +%FT%TZ) GET $fn" >> "$LOG"
  # civitai needs auth token for direct download in some cases
  curl -sL --fail --retry 3 --max-time 900 \
    -H "Authorization: Bearer $CIVITAI_API_KEY" \
    -o "$out.part" "$url" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$(date -u +%FT%TZ) FAIL($rc) $fn" >> "$LOG"; rm -f "$out.part"; continue; fi
  actual=$(sha256sum "$out.part" | awk '{print toupper($1)}')
  if [ "$actual" = "$(echo $sha | tr 'a-z' 'A-Z')" ]; then
    mv "$out.part" "$out"
    echo "$(date -u +%FT%TZ) OK sha-match $fn $(stat -c%s "$out")" >> "$LOG"
  else
    echo "$(date -u +%FT%TZ) SHA-MISMATCH $fn want=$sha got=$actual" >> "$LOG"
    mv "$out.part" "$out.badmismatch"
  fi
done < /tmp/lora_manifest.txt
echo "$(date -u +%FT%TZ) DONE" >> "$LOG"
