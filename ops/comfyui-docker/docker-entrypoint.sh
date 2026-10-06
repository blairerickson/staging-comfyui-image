#!/usr/bin/env bash
set -euo pipefail

# docker-entrypoint.sh — boot-time model pull + ComfyUI launch
# Part of the PlaySpace staging Docker image.
# Secrets from env: CIVITAI_API_KEY, DEEPSEEK_API_KEY

COMFY_DIR="/root/ComfyUI"
VENV_DIR="/root/venv"
MODELS_DIR="/root/models"

echo "[entrypoint] Starting PlaySpace staging box bootstrapper"
echo "[entrypoint] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

setup_symlinks() {
    echo "[entrypoint] Setting up model symlinks..."

    # diffusion_models -> /root/models/diffusion_models
    [ -f "$COMFY_DIR/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors" ] || \
        ln -sf "$MODELS_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors" \
               "$COMFY_DIR/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"

    [ -f "$COMFY_DIR/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" ] || \
        ln -sf "$MODELS_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors" \
               "$COMFY_DIR/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

    [ -f "$COMFY_DIR/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" ] || \
        ln -sf "$MODELS_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
               "$COMFY_DIR/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"

    [ -f "$COMFY_DIR/models/diffusion_models/h3ErosMax_beta5.safetensors" ] || \
        ln -sf "$COMFY_DIR/models/unet/h3ErosMax_beta5.safetensors" \
               "$COMFY_DIR/models/diffusion_models/h3ErosMax_beta5.safetensors"

    [ -f "$COMFY_DIR/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors" ] || \
        ln -sf "$COMFY_DIR/models/checkpoints/krea2_turbo_fp8_scaled.safetensors" \
               "$COMFY_DIR/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors"

    # text_encoders -> /root/models/text_encoders
    for f in qwen3vl_4b_fp8_scaled.safetensors qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
             Qwen3-VL-4B-Instruct-Uncensored-FP8.safetensors; do
        [ -f "$COMFY_DIR/models/text_encoders/$f" ] || \
            ln -sf "$MODELS_DIR/text_encoders/$f" \
                   "$COMFY_DIR/models/text_encoders/$f"
    done

    # vae -> /root/models/vae
    for f in minimax_h3_video_vae_fp16.safetensors minimax_h3_audio_vae_fp32.safetensors \
             qwen_image_vae.safetensors; do
        [ -f "$COMFY_DIR/models/vae/$f" ] || \
            ln -sf "$MODELS_DIR/vae/$f" \
                   "$COMFY_DIR/models/vae/$f"
    done

    # h3_char_tokens
    [ -L "$COMFY_DIR/models/h3_char_tokens" ] || \
        ln -sf ../input/h3_char_tokens "$COMFY_DIR/models/h3_char_tokens"

    echo "[entrypoint] Symlinks done."
}

# download a file if not present and sha256 match optional
dl() {
    local url="$1" dest="$2" sha256="${3:-}"
    local dir; dir=$(dirname "$dest")
    mkdir -p "$dir"
    if [ -f "$dest" ]; then
        if [ -n "$sha256" ]; then
            if echo "$sha256  $dest" | sha256sum -c --quiet 2>/dev/null; then
                echo "[entrypoint] Already have $dest (match)"
                return 0
            fi
            echo "[entrypoint] Re-downloading $dest (sha mismatch)"
        else
            echo "[entrypoint] Already have $dest"
            return 0
        fi
    fi
    echo "[entrypoint] Downloading $url ..."
    aria2c -x8 -s8 --console-log-level=error -d "$dir" -o "$(basename "$dest")" "$url"
}

# --- Phase 1: HuggingFace base models ---
pull_hf_base() {
    echo "[entrypoint] Phase 1: HuggingFace base models..."

    dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors" \
       "$MODELS_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"

    dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
       "$MODELS_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"

    dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
       "$MODELS_DIR/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

    dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
       "$MODELS_DIR/vae/minimax_h3_video_vae_fp16.safetensors"

    dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" \
       "$MODELS_DIR/vae/minimax_h3_audio_vae_fp32.safetensors"

    dl "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors" \
       "$COMFY_DIR/models/checkpoints/krea2_turbo_fp8_scaled.safetensors"

    dl "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
       "$MODELS_DIR/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"

    dl "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors" \
       "$MODELS_DIR/vae/qwen_image_vae.safetensors"

    echo "[entrypoint] Phase 1 done."
}

# --- Phase 2: Civitai LoRAs with known modelVersionId ---
pull_civitai() {
    echo "[entrypoint] Phase 2: Civitai LoRAs..."
    if [ -z "${CIVITAI_API_KEY:-}" ]; then
        echo "[entrypoint] WARNING: CIVITAI_API_KEY not set — skipping Civitai downloads."
        echo "[entrypoint] The box will boot but Civitai LoRAs will be missing."
        return 0
    fi

    local civitai_dl="$VENV_DIR/bin/python -c \"
import sys, os, json, urllib.request
key = os.environ['CIVITAI_API_KEY']
url = sys.argv[1]
dest = sys.argv[2]
if os.path.exists(dest):
    print(f'Already have {dest}')
    sys.exit(0)
os.makedirs(os.path.dirname(dest), exist_ok=True)
req = urllib.request.Request(url.replace('https://civitai.com/api/download/', 'https://civitai.com/api/download/models/'))
req.add_header('Authorization', f'Bearer {key}')
print(f'Downloading {url} -> {dest}')
with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
    while True:
        chunk = r.read(8*1024*1024)
        if not chunk: break
        f.write(chunk)
print(f'Done: {dest}')
\""

    download_civitai_model() {
        local model_id="$1" file_id="$2" dest="$3"
        local url="https://civitai.com/api/download/models/${model_id}?fileId=${file_id}"
        if [ -f "$dest" ]; then
            echo "[entrypoint] Already have $dest"
            return 0
        fi
        mkdir -p "$(dirname "$dest")"
        curl -sL -H "Authorization: Bearer ${CIVITAI_API_KEY}" "$url" -o "$dest" && \
            echo "[entrypoint] Done: $dest"
    }

    download_civitai_model "3294059" "3185154" "$COMFY_DIR/models/unet/h3ErosMax_beta5.safetensors"
    download_civitai_model "3071970" "2951107" "$COMFY_DIR/models/checkpoints/krea2TurboNSFWAIO_v10.safetensors"
    download_civitai_model "3219337" "3101211" "$COMFY_DIR/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    download_civitai_model "3206518" "3088013" "$COMFY_DIR/models/loras/HMNSFW_AIO_V2.safetensors"
    download_civitai_model "3198686" "3079816" "$COMFY_DIR/models/loras/hmmotion_minimax-h3_epoch12.safetensors"
    download_civitai_model "3222484" "3104474" "$COMFY_DIR/models/loras/HMInnie_v1_e50.safetensors"
    download_civitai_model "3218160" "3099984" "$COMFY_DIR/models/loras/HMPenis_v2_e35.safetensors"
    download_civitai_model "3215304" "3097094" "$COMFY_DIR/models/loras/vagassist_e40.safetensors"
    download_civitai_model "3207723" "3089272" "$COMFY_DIR/models/loras/Pussy4nus_Epoch80.safetensors"
    download_civitai_model "3238531" "3121030" "$COMFY_DIR/models/loras/HMCumshot_V2.safetensors"
    download_civitai_model "3213728" "3095449" "$COMFY_DIR/models/loras/vaglokr.safetensors"
    download_civitai_model "3268969" "3152740" "$COMFY_DIR/models/loras/HMBreastsV2.safetensors"
    download_civitai_model "3252213" "3135252" "$COMFY_DIR/models/loras/Vagina.safetensors"
    download_civitai_model "3228089" "3110357" "$COMFY_DIR/models/loras/moawxx_000002000.safetensors"
    download_civitai_model "3294126" "3179683" "$COMFY_DIR/models/loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
    download_civitai_model "3240923" "3123486" "$COMFY_DIR/models/loras/davham_cinema.safetensors"
    download_civitai_model "3278719" "3162803" "$COMFY_DIR/models/loras/Cinema-MH3-V02_000010000.safetensors"
    download_civitai_model "3308674" "3193838" "$COMFY_DIR/models/loras/indie90s_style_h3_ep50.safetensors"
    download_civitai_model "3308677" "3193841" "$COMFY_DIR/models/loras/horror70s_style_h3_ep50.safetensors"
    download_civitai_model "3308670" "3193833" "$COMFY_DIR/models/loras/cybergoth90s_style_h3_ep50.safetensors"
    download_civitai_model "3312531" "3197760" "$COMFY_DIR/models/loras/cinemagrade_style_h3_ep50.safetensors"
    download_civitai_model "3309568" "3194784" "$COMFY_DIR/models/loras/BoobSlider_V2.safetensors"
    download_civitai_model "3308678" "3193901" "$COMFY_DIR/models/loras/Sharpness.safetensors"
    download_civitai_model "3308678" "3193901" "$COMFY_DIR/models/loras/minimax_h3_lms_v1.0_r64.safetensors"
    download_civitai_model "3256084" "3139271" "$COMFY_DIR/models/loras/mvmt_h3_lora_v1_500.safetensors"
    download_civitai_model "3268186" "3151950" "$COMFY_DIR/models/loras/Motion_Repair.safetensors"
    download_civitai_model "3301234" "3186234" "$COMFY_DIR/models/loras/FrameRush-Minimax-V2_c2-st1500.safetensors"
    download_civitai_model "3204862" "3086301" "$COMFY_DIR/models/loras/SynthPussy_H3_closeups_v1-step00008300.safetensors"
    download_civitai_model "3266628" "3150341" "$COMFY_DIR/models/loras/MysticXXX_MMH3-V4.safetensors"
    download_civitai_model "3310653" "3195930" "$COMFY_DIR/models/loras/M3_Unlocked_V2.safetensors"
    download_civitai_model "3314964" "3200403" "$COMFY_DIR/models/loras/DetailSlider-V1.safetensors"
    download_civitai_model "3267949" "3151712" "$COMFY_DIR/models/loras/Minimax H3真实电影质感.safetensors"
    download_civitai_model "3289775" "3174203" "$COMFY_DIR/models/loras/Minimax H3真实电影质感V0.1（解决张量报错）.safetensors"
    download_civitai_model "3294059" "3185144" "$COMFY_DIR/models/unet/h3ErosMax_beta5.cminfo.json"

    echo "[entrypoint] Phase 2 done."
}

# --- Phase 3: HuggingFace character assets (refmods + krea2 character LoRAs) ---
pull_hf_chars() {
    echo "[entrypoint] Phase 3: HF character assets..."

    if [ ! -f "$COMFY_DIR/models/refmods/.done" ]; then
        echo "[entrypoint] Downloading refmods from HF (malcolmrey/minimaxh3)..."
        # Clone the HF repo using git-lfs sparse checkout
        if [ ! -d /tmp/refmods_clone ]; then
            GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/malcolmrey/minimaxh3 /tmp/refmods_clone 2>&1 | tail -3
        fi
        cp /tmp/refmods_clone/*.safetensors "$COMFY_DIR/models/refmods/" 2>/dev/null || true
        touch "$COMFY_DIR/models/refmods/.done"
        echo "[entrypoint] Refmods done."
    fi

    if [ ! -f "$COMFY_DIR/models/loras/.krea2_char_done" ]; then
        echo "[entrypoint] Downloading krea2 character LoRAs from HF (malcolmrey/krea2)..."
        if [ ! -d /tmp/krea2_clone ]; then
            GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/malcolmrey/krea2 /tmp/krea2_clone 2>&1 | tail -3
        fi
        cp /tmp/krea2_clone/*.safetensors "$COMFY_DIR/models/loras/" 2>/dev/null || true
        touch "$COMFY_DIR/models/loras/.krea2_char_done"
        echo "[entrypoint] Krea2 character LoRAs done."
    fi

    # Huihui text encoder
    dl "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated-FP8/resolve/main/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors" \
       "$MODELS_DIR/text_encoders/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors"

    echo "[entrypoint] Phase 3 done."
}

# --- Phase 4: Launch ComfyUI ---
launch_comfy() {
    echo "[entrypoint] Launching ComfyUI..."
    cd "$COMFY_DIR"

    export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
    export CIVITAI_API_KEY="${CIVITAI_API_KEY:-}"

    exec "$VENV_DIR/bin/python" main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-manager
}

# --- Main ---
case "${1:-serve}" in
    serve)
        setup_symlinks
        pull_hf_base
        pull_civitai
        pull_hf_chars
        setup_symlinks
        launch_comfy
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac