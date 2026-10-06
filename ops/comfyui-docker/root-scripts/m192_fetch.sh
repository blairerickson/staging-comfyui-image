set -u
cd /root/ComfyUI/models/loras
URL="https://huggingface.co/sololo-xyz/EmmaMyers_SoloLoRA_Kr2v1/resolve/main/SadieS_SoloLoRA_Kr2v1.safetensors"
OUT=sololo_emmamyers_kr2v1_raw.safetensors
if [ ! -s "$OUT" ]; then
  curl -sL --fail --retry 3 --max-time 900 -o "$OUT.part" "$URL" || { echo "DL FAIL"; rm -f "$OUT.part"; exit 1; }
  mv "$OUT.part" "$OUT"
fi
ls -la "$OUT"
python3 - <<'PY'
import json,struct
p="/root/ComfyUI/models/loras/sololo_emmamyers_kr2v1_raw.safetensors"
f=open(p,"rb"); n=struct.unpack("<Q",f.read(8))[0]; h=json.loads(f.read(n))
meta=h.get("__metadata__",{})
print("header keys:",len(h),"meta keys:",list(meta.keys()))
for k,v in meta.items():
    s=str(v); print(" ",k,"=",(s[:400]+" ...") if len(s)>400 else s)
PY
