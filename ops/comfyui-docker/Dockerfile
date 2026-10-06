# syntax=docker/dockerfile:1
# PlaySpace staging ComfyUI 0.34.0 — H3/Krea2 render box
# CI-friendly build: git clone + pip install, no tarballs committed.
# Uses nvidia/cuda:12.8.1-devel for sageattn3 compilation.
# Model strategy: boot-time pull from HuggingFace + Civitai (see docker-entrypoint.sh)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/root/venv \
    PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential \
        git git-lfs tmux ffmpeg aria2 curl \
        ca-certificates wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /root/venv && \
    /root/venv/bin/pip install --upgrade pip setuptools wheel

RUN /root/venv/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128

# Staging renders on RTX PRO 6000 / 5090 Blackwell, i.e. compute capability 12.0.
# Pinning the arch keeps the CI build down to a single target and lets sageattention
# pick its gencode flags from the env instead of probing for a local GPU, which the
# Actions runner does not have.
ENV TORCH_CUDA_ARCH_LIST=12.0 \
    EXT_PARALLEL=1 \
    MAX_JOBS=4

RUN git clone https://github.com/thu-ml/sageattention.git /tmp/sageattention && \
    cd /tmp/sageattention && git checkout d1a57a5 && \
    # Both setup.py files import torch at module scope, so PEP 517 build isolation
    # has to be off — the isolated env has no torch. That means their build deps
    # must already be in the venv, at the versions pyproject.toml asks for.
    /root/venv/bin/pip install --no-cache-dir \
        "setuptools>=62,<75" "wheel>=0.38,<0.44" "packaging>=21,<24" ninja && \
    /root/venv/bin/pip install --no-cache-dir --no-build-isolation . && \
    # sageattention3_blackwell reads the arch straight off a local GPU and has no
    # TORCH_CUDA_ARCH_LIST path, so pin it to sm_120 for the build.
    cd sageattention3_blackwell && \
    sed -i 's/^    cc_major, cc_minor = torch\.cuda\.get_device_capability()$/    cc_major, cc_minor = (12, 0)  # pinned: no GPU on the build host/' setup.py && \
    grep -q 'cc_major, cc_minor = (12, 0)' setup.py && \
    /root/venv/bin/pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/sageattention

COPY ops/comfyui-docker/requirements.txt /tmp/requirements.txt
RUN /root/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /root

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI && \
    cd /root/ComfyUI && git checkout 5653b4ac8eca01e64df7d8b5a2ee22385f14dabf && \
    rm -rf .git

# Pinned custom nodes, matching the staging box snapshot of 2026-10-06 04:20 UTC.
# The SHAs below are the full 40-char commit ids, each resolved from the 8-char short
# id recorded in the snapshot inventory and verified to exist upstream. The previous
# revision of this file padded those 8-char prefixes out to 40 chars with hex that
# does not belong to any commit, so every checkout died with
# "fatal: reference is not a tree".
# Fetch the pinned SHA directly at depth 1: it is far smaller than a full clone and
# it fails at fetch time on a bad or unreachable pin. `|| exit 1` makes the layer
# abort on the first bad node instead of shipping an image with nodes missing.
RUN mkdir -p /root/ComfyUI/custom_nodes && \
    while IFS=' ' read -r repo dir sha; do \
        echo "Fetching $repo -> $dir @ $sha"; \
        target="/root/ComfyUI/custom_nodes/$dir"; \
        git init -q "$target" && \
        git -C "$target" remote add origin "https://github.com/$repo.git" && \
        git -C "$target" fetch -q --depth 1 origin "$sha" && \
        git -C "$target" checkout -q FETCH_HEAD && \
        rm -rf "$target/.git" || exit 1; \
    done << 'NODES'
MoonGoblinDev/Civicomfy Civicomfy 555e984bbcb02fc955075cb8eb699a6884c51239
Fannovel16/ComfyUI-Frame-Interpolation ComfyUI-Frame-Interpolation 26545cc2dd95bc3d27f056016300673bdeee78f5
spacepxl/ComfyUI-Image-Filters ComfyUI-Image-Filters bbb3fb0045461adf3602faeedaf40af57090d4e2
kijai/ComfyUI-KJNodes ComfyUI-KJNodes 57105374f47d0fbb49c9c3926fb981702e0a4b5c
aria1th/ComfyUI-LogicUtils ComfyUI-LogicUtils f77c699543b8f9977b24581af66ad026d340b663
ltdrdata/ComfyUI-Manager ComfyUI-Manager f39cbd56fecae0b27a446c0cd450cd591f3a8bea
Luisacaotica/ComfyUI-MiniMaxH3Mod ComfyUI-MiniMaxH3Mod 6131c1f62c612a074d6aac8435c26ffa19c00563
ttulttul/ComfyUI-Minimax-H3-Continuation ComfyUI-Minimax-H3-Continuation e1768d5fdfc6f9519d2090dcf78458c2d9625f80
xmarre/ComfyUI-Spectrum-MiniMax-H3 ComfyUI-Spectrum-MiniMax-H3 be95adecec0b85c80d0c9fc5dd8d07386d50aaee
yuvraj108c/ComfyUI-Upscaler-Tensorrt ComfyUI-Upscaler-Tensorrt f428364befb14848c7b6efe357f4b4ff0904009c
Kosinkadink/ComfyUI-VideoHelperSuite ComfyUI-VideoHelperSuite 115de7a9d9e34410cffb9ecfd268e993b11a50fb
WASasquatch/was-node-suite-comfyui ComfyUI-WAS-Node-Suite ea935d1044ae5a26efa54ebeb18fe9020af49a45
yolain/ComfyUI-Easy-Use ComfyUI-easy-use 58e077a7435631301cf7443412515cf958e7f3d1
bvhari/ComfyUI_CFGStar ComfyUI_CFGStar 37c0b2d08aad048e65876874c8c6c95a94e9d6a0
cubiq/ComfyUI_essentials ComfyUI_essentials 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9
signerzwb/Comfyui-minimaxh3-FBcache-shendumao Comfyui-minimaxh3-FBcache-shendumao a19b53c4507192a9cb164263335b238a454c9690
ClownsharkBatwing/RES4LYF RES4LYF 26036f647ca15d3048a193daf99a40cecfc3820d
pythongosssss/ComfyUI-Custom-Scripts comfyui-custom-scripts 609f3afaa74b2f88ef9ce8d939626065e3247469
lbouaraba/comfyui-krea2edit comfyui-krea2edit 86f886dac23013d88996e3a2e99093ba44d322fb
heshengtao/comfyui_LLM_party comfyui_LLM_party 3790b26a438bb8e77c1ad24d93870f710cc8ae41
dylan121322/qwen3vl-kvcache qwen3vl-kvcache 126fc6c914a71a134ad05cfe8a178c9d5cc2407e
rgthree/rgthree-comfy rgthree-comfy 2c5342a8cb0eaecaabf61435a5f37dd594c510ba
NODES

COPY ops/comfyui-docker/hand-placed-nodes/ComfyUI-H3-LongTake /root/ComfyUI/custom_nodes/ComfyUI-H3-LongTake
COPY ops/comfyui-docker/hand-placed-nodes/h3_direct_latent_head /root/ComfyUI/custom_nodes/h3_direct_latent_head
COPY ops/comfyui-docker/hand-placed-nodes/playspace-llm /root/ComfyUI/custom_nodes/playspace-llm
COPY ops/comfyui-docker/hand-placed-nodes/playspace_fleet /root/ComfyUI/custom_nodes/playspace_fleet
COPY ops/comfyui-docker/hand-placed-nodes/websocket_image_save.py /root/ComfyUI/custom_nodes/websocket_image_save.py

COPY ops/comfyui-docker/extra_model_paths.yaml /root/ComfyUI/extra_model_paths.yaml

RUN mkdir -p /root/ComfyUI/user/default && true
COPY ops/comfyui-docker/comfy.settings.json /root/ComfyUI/user/default/comfy.settings.json

COPY ops/comfyui-docker/root-scripts/ /root/
RUN find /root -maxdepth 1 -name '*.sh' -exec chmod +x {} + && \
    find /root -maxdepth 1 -name '*.py' -type f | xargs chmod +x

COPY ops/comfyui-docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

RUN mkdir -p /root/models/diffusion_models \
             /root/models/text_encoders \
             /root/models/vae \
             /root/ComfyUI/models/diffusion_models \
             /root/ComfyUI/models/text_encoders \
             /root/ComfyUI/models/vae \
             /root/ComfyUI/models/unet \
             /root/ComfyUI/models/checkpoints \
             /root/ComfyUI/models/loras \
             /root/ComfyUI/models/refmods \
             /root/ComfyUI/models/clip \
             /root/ComfyUI/models/clip_vision \
             /root/ComfyUI/models/controlnet \
             /root/ComfyUI/models/embeddings \
             /root/ComfyUI/models/upscale_models \
             /root/ComfyUI/models/frame_interpolation \
             /root/ComfyUI/models/style_models \
             /root/ComfyUI/models/configs \
             /root/ComfyUI/models/vae_approx \
             /root/ComfyUI/models/diffusers \
             /root/ComfyUI/models/detection \
             /root/ComfyUI/models/optical_flow \
             /root/ComfyUI/models/photomaker \
             /root/ComfyUI/models/audio_encoders \
             /root/ComfyUI/models/background_removal \
             /root/ComfyUI/models/geometry_estimation \
             /root/ComfyUI/models/hypernetworks \
             /root/ComfyUI/models/latent_upscale_models \
             /root/ComfyUI/models/model_patches \
             /root/ComfyUI/models/motion_models \
             /root/ComfyUI/models/other \
             /root/ComfyUI/models/poses \
             /root/ComfyUI/models/wildcards \
             /root/ComfyUI/models/loras_quarantine \
             /root/ComfyUI/input \
             /root/ComfyUI/output

EXPOSE 8188

ENV CIVITAI_API_KEY="" \
    DEEPSEEK_API_KEY=""

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["serve"]