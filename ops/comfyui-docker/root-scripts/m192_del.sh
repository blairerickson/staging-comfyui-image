set -u
D=/root/ComfyUI/models/loras
mkdir -p /root/ComfyUI/models/loras_quarantine
mv -v "$D/krea2_emmamyers_v1_onetrainer.safetensors" /root/ComfyUI/models/loras_quarantine/ 2>&1
ls -la /root/ComfyUI/models/loras_quarantine/ | tail -3
echo "--- emmamyers still in loras dir? ---"
ls "$D" | grep -i emmamyers || echo "none (good)"
echo "--- sadie file present ---"
ls -la "$D/krea2_sadiesink_v1_onetrainer.safetensors"
