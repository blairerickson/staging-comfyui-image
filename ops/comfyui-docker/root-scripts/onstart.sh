# ===== prepended: ensure /root/venv exists (fresh image has none) =====
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
log() { printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
if [ ! -x /root/venv/bin/python ]; then
  log "creating /root/venv from system python3"
  # ensure python3-venv + pip available
  (apt-get update -y && apt-get install -y --no-install-recommends python3-venv python3-pip ca-certificates curl git tmux ffmpeg aria2 openssh-server) >/tmp/apt.log 2>&1 || true
  python3 -m venv /root/venv 2>/tmp/venv-make.log || { log "venv create FAILED"; tail -5 /tmp/venv-make.log; }
  /root/venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1 || true
fi
log "venv ready: $(/root/venv/bin/python --version 2>&1)"

#!/bin/bash
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export COMFY_DIR=/root/ComfyUI
export MODELS_DIR=/root/models
export LOG_DIR=/root/boot_logs
export COMFY_LOG="$LOG_DIR/comfyui.log"
export BOOT_LOG="$LOG_DIR/provision.log"
mkdir -p "$LOG_DIR" "$MODELS_DIR" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" "$MODELS_DIR/diffusion_models" "$COMFY_DIR/user/default/workflows"
export PATH="/usr/local/cuda-12.8/bin:/root/venv/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/root/.cache/huggingface

PYTHON=/root/venv/bin/python
[ -x "$PYTHON" ] || PYTHON=/usr/bin/python3
export PYTHON

log() { printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$BOOT_LOG"; }
run() { log "RUN $*"; "$@" >>"$BOOT_LOG" 2>&1; local rc=$?; [ $rc -eq 0 ] || log "FAILED rc=$rc $*"; return $rc; }

log "=== provision2 begin (python=$PYTHON) ==="

if [[ ! -d "$COMFY_DIR/.git" ]]; then
    # FIX: mkdir'd "$COMFY_DIR/user/default/workflows" above makes target non-empty,
    # which makes git clone refuse (rc=128). Clear a stale non-git dir before cloning.
    if [[ -e "$COMFY_DIR" ]] && [ ! -d "$COMFY_DIR/.git" ]; then
        log "clearing stale non-git $COMFY_DIR before clone"
        rm -rf "$COMFY_DIR"
    fi
    run git clone --recursive https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    run git -C "$COMFY_DIR" checkout --detach 5653b4ac8eca01e64df7d8b5a2ee22385f14dabf
else
    log "ComfyUI checkout exists"
fi
# pin comfy rev in case of partial
( cd "$COMFY_DIR" && git checkout --detach 5653b4ac8eca01e64df7d8b5a2ee22385f14dabf 2>/dev/null || true ) >>"$BOOT_LOG" 2>&1

# --- torch/cu128 into venv (idempotent) ---
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    log "installing torch 2.7.0+cu128 into venv (big download)"
    run "$PYTHON" -m pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
"$PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" >"$LOG_DIR/torch-check.txt" 2>&1 || true
cat "$LOG_DIR/torch-check.txt" | tee -a "$BOOT_LOG"

# --- comfy + node python deps into venv ---
run "$PYTHON" -m pip install --no-cache-dir \
    "comfyui-frontend-package==1.51.9" "comfyui-workflow-templates==0.11.48" \
    "comfyui-embedded-docs==0.5.10" "comfy-kitchen==0.2.31" "comfy-aimdo==0.4.15" \
    opencv-python-headless numba PyWavelets aisuite polygraphy pymatting matplotlib \
    color-matcher GitPython PyGithub matrix-nio "huggingface-hub>0.20" typer rich \
    typing-extensions toml uv chardet

clone_detached(){ local n=$1 r=$2 v=$3; [[ -d "$COMFY_DIR/custom_nodes/$n/.git" ]] || run git clone "$r" "$COMFY_DIR/custom_nodes/$n"; ( cd "$COMFY_DIR/custom_nodes/$n" && git checkout --detach "$v" 2>/dev/null || true ); }
clone_detached ComfyUI-Manager            https://github.com/ltdrdata/ComfyUI-Manager f39cbd5
clone_detached comfyui-krea2edit          https://github.com/lbouaraba/comfyui-krea2edit 86f886d
clone_detached ComfyUI-Frame-Interpolation https://github.com/Fannovel16/ComfyUI-Frame-Interpolation 26545cc
clone_detached ComfyUI-Image-Filters      https://github.com/spacepxl/ComfyUI-Image-Filters bbb3fb0
clone_detached ComfyUI-VideoHelperSuite   https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite 115de7a
clone_detached ComfyUI-WAS-Node-Suite     https://github.com/WASasquatch/was-node-suite-comfyui ea935d1
clone_detached ComfyUI-easy-use           https://github.com/yolain/ComfyUI-Easy-Use 58e077a
clone_detached ComfyUI_CFGStar            https://github.com/bvhari/ComfyUI_CFGStar 37c0b2d
clone_detached ComfyUI_essentials         https://github.com/cubiq/ComfyUI_essentials 9d9f4be
clone_detached RES4LYF                    https://github.com/ClownsharkBatwing/RES4LYF 26036f6
clone_detached ComfyUI-Upscaler-Tensorrt  https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt f428364
for spec in "ComfyUI-KJNodes|https://github.com/kijai/ComfyUI-KJNodes" "ComfyUI-LogicUtils|https://github.com/aria1th/ComfyUI-LogicUtils" "ComfyUI-Spectrum-MiniMax-H3|https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3" "Comfyui-minimaxh3-FBcache-shendumao|https://github.com/signerzwb/Comfyui-minimaxh3-FBcache-shendumao" "comfyui-custom-scripts|https://github.com/pythongosssss/ComfyUI-Custom-Scripts" "comfyui_LLM_party|https://github.com/heshengtao/comfyui_LLM_party" "qwen3vl-kvcache|https://github.com/dylan121322/qwen3vl-kvcache" "rgthree-comfy|https://github.com/rgthree/rgthree-comfy" "Civicomfy|https://github.com/MoonGoblinDev/Civicomfy"; do
    name=${spec%%|*}; url=${spec#*|}; [[ -d "$COMFY_DIR/custom_nodes/$name" ]] || run git clone --depth 1 "$url" "$COMFY_DIR/custom_nodes/$name"
done

download(){ local url=$1 dest=$2 min=$3; mkdir -p "$(dirname "$dest")"; if [[ -s "$dest" && $(stat -c %s "$dest") -ge $min ]]; then log "SKIP $(basename "$dest")"; return 0; fi; rm -f "$dest" "$dest.part";
  if ! run aria2c -x 8 -s 8 --retry-wait=2 --max-tries=8 --continue=true --console-log-level=warn --summary-interval=0 -d "$(dirname "$dest")" -o "$(basename "$dest.part")" "$url"; then log "FALLBACK curl $(basename "$dest")"; run curl -fL --retry 8 --retry-delay 2 -o "$dest.part" "$url"; fi
  mv "$(dirname "$dest")/$(basename "$dest.part")" "$dest"; local b=$(stat -c %s "$dest"); (( b >= min )) || { log "SIZE FAIL $dest $b<$min"; return 1; }; log "OK $(basename "$dest") $b"; }

download https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors "$COMFY_DIR/models/checkpoints/krea2_turbo_fp8_scaled.safetensors" 12533000000
download https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors "$MODELS_DIR/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" 5000000000
download https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors "$MODELS_DIR/vae/qwen_image_vae.safetensors" 240000000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors "$MODELS_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors" 19987000000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors "$MODELS_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" 19999000000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "$MODELS_DIR/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" 14960000000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors "$MODELS_DIR/vae/minimax_h3_video_vae_fp16.safetensors" 4967000000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors "$MODELS_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" 577000000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors "$COMFY_DIR/models/loras/minimax_h3_fl2v_turbo_4step_v0.1.safetensors" 1956192000
download https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors "$COMFY_DIR/models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" 1956193000
download https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors "$COMFY_DIR/models/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors" 314800000
download https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors "$COMFY_DIR/models/loras/minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors" 306700000

mkdir -p "$COMFY_DIR/models/diffusion_models" "$COMFY_DIR/models/text_encoders" "$COMFY_DIR/models/vae"
for sub in diffusion_models text_encoders vae; do for file in "$MODELS_DIR/$sub"/*; do [[ -e "$file" ]] || continue; name=$(basename "$file"); t="$COMFY_DIR/models/$sub/$name"; [[ -e "$t" ]] || ln -s "$file" "$t"; done; done

# install manager + each custom node req
run "$PYTHON" -m pip install -e "$COMFY_DIR/custom_nodes/ComfyUI-Manager" || log "manager-install-fail-continuing"
for f in "$COMFY_DIR"/custom_nodes/*/requirements.txt; do b=$(basename "$(dirname "$f")"); [[ "$b" == ComfyUI-Upscaler-Tensorrt ]] && continue; log "req: $b"; "$PYTHON" -m pip install -r "$f" >>"$BOOT_LOG" 2>&1 || log "req-fail: $b"; done

log "provision2 phase complete"

# ===== added tail: fl2va alias + launch ComfyUI exposed on 8188 =====
FP8="$MODELS_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
DM="$COMFY_DIR/models/diffusion_models"
mkdir -p "$DM"
if [ -f "$FP8" ] && [ ! -e "$DM/minimax_h3_fl2va_pruned_int8_convrot.safetensors" ]; then
  ln -s "$FP8" "$DM/minimax_h3_fl2va_pruned_int8_convrot.safetensors" && log "fl2va fp8->int8 alias linked"
fi
if ! tmux has-session -t comfy 2>/dev/null; then
  tmux new-session -d -s comfy "cd $COMFY_DIR && DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-} /root/venv/bin/python main.py --listen 0.0.0.0 --port 8188 --enable-manager 2>&1 | tee $LOG_DIR/comfyui.log"
  log "ComfyUI tmux launched on 0.0.0.0:8188"
else
  log "comfy tmux already exists"
fi
log "=== onstart-expose8188 complete ==="
