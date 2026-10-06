set -u
: "${CIVITAI_API_KEY:?CIVITAI_API_KEY must be set}"
KEY="$CIVITAI_API_KEY"
D=/root/ComfyUI/models/loras
cd "$D"
try() {
  fn="$1"; vid="$2"
  if [ -s "$fn" ]; then echo "SKIP present $fn $(stat -c%s "$fn")"; return; fi
  code=$(curl -sL --retry 2 --max-time 600 -H "Authorization: Bearer $KEY" -o "$fn.part" -w "%{http_code}" "https://civitai.com/api/download/models/$vid")
  sz=$(stat -c%s "$fn.part" 2>/dev/null || echo 0)
  head -c 80 "$fn.part" | tr -d '\0' > /tmp/hdr.$$
  if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ]; then mv "$fn.part" "$fn"; echo "OK $fn $code $sz"
  else echo "FAIL $fn http=$code size=$sz head=$(head -c 120 /tmp/hdr.$$)"; rm -f "$fn.part"; fi
  rm -f /tmp/hdr.$$
}
try "hmnsfw_spankme_h3_v1.safetensors" 3329929
try "cruel_intentions_1999_krea2.safetensors" 3345686
try "donnie_darko_2001_krea2.safetensors" 3345732
