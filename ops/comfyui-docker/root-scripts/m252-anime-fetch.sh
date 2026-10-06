#!/bin/bash
# M252 — download Blair's 8 anime Krea2 LoRAs onto the Vast box.
KEY="$1"
D=/root/ComfyUI/models/loras
LOG=/root/m252-anime-dl.log
: > "$LOG"
dl() {
  fn="$1"; url="$2"
  if [ -s "$D/$fn" ]; then echo "SKIP present $fn $(stat -c%s "$D/$fn")" >> "$LOG"; return; fi
  code=$(curl -sL --retry 3 --retry-delay 2 --max-time 1800 -H "Authorization: Bearer $KEY" -o "$D/$fn.part" -w "%{http_code}" "$url" 2>>"$LOG")
  sz=$(stat -c%s "$D/$fn.part" 2>/dev/null || echo 0)
  if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ]; then mv "$D/$fn.part" "$D/$fn"; echo "OK $fn http=$code bytes=$sz" >> "$LOG"
  else echo "FAIL $fn http=$code bytes=$sz body=$(head -c 200 "$D/$fn.part" 2>/dev/null | tr -d '\0')" >> "$LOG"; rm -f "$D/$fn.part"; fi
}
dl "R88YRZ2KRST5RHHEAM8D77NFW0.safetensors"          "https://civitai.com/api/download/models/3187443?fileId=3068667"
dl "Cartoony_Anime_epoch_10.safetensors"             "https://civitai.com/api/download/models/3186331?fileId=3066898"
dl "Vintage_Anime_Style_KREA2.safetensors"           "https://civitai.com/api/download/models/3188940?fileId=3069588"
dl "retro_anime_style_krea2.safetensors"             "https://civitai.com/api/download/models/3118780?fileId=2999053"
dl "ANIME MIX V1.safetensors"                         "https://civitai.com/api/download/models/3328763?fileId=3214750"
dl "mode_niji_anime_krea2_v3.safetensors"            "https://civitai.com/api/download/models/3349914?fileId=3237026"
dl "AquarelleKrea2.safetensors"                       "https://civitai.com/api/download/models/3119209?fileId=2999483"
dl "Gurren_Lagann__Anime__Eyecatch_Style.safetensors" "https://civitai.com/api/download/models/3308897?fileId=3194076"
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
