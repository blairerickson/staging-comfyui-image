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

RUN mkdir -p /root/ComfyUI/custom_nodes && \
    while IFS=' ' read -r repo dir sha; do \
        echo "Cloning $repo -> $dir @ $sha"; \
        git clone "https://github.com/$repo.git" "/root/ComfyUI/custom_nodes/$dir" && \
        cd "/root/ComfyUI/custom_nodes/$dir" && git checkout "$sha" && rm -rf .git; \
    done << 'NODES'
MoonGoblinDev/Civicomfy Civicomfy 555e984bd42d7a18b6e0da8a3e661869b7abec18
Fannovel16/ComfyUI-Frame-Interpolation ComfyUI-Frame-Interpolation 26545cc20ac6ec400d889748ad8b7d9fac3a7c83
spacepxl/ComfyUI-Image-Filters ComfyUI-Image-Filters bbb3fb00cd6549647f77667a25581b3103aa133a
kijai/ComfyUI-KJNodes ComfyUI-KJNodes 57105374ad102984dab97e2b1f841f5e37114ee7
aria1th/ComfyUI-LogicUtils ComfyUI-LogicUtils f77c699550802aec72b26d9d8bdc663facee5d9a
ltdrdata/ComfyUI-Manager ComfyUI-Manager f39cbd56ca80adc1ede8e3d0dbdfc19667c9e250
Luisacaotica/ComfyUI-MiniMaxH3Mod ComfyUI-MiniMaxH3Mod 6131c1f67f30b5f4a9c12e05c966159449cb1598
ttulttul/ComfyUI-Minimax-H3-Continuation ComfyUI-Minimax-H3-Continuation e1768d5faf5e87af7c3f5a3af71037cff804877d
xmarre/ComfyUI-Spectrum-MiniMax-H3 ComfyUI-Spectrum-MiniMax-H3 be95adec6c2411ec5362425deff4bb7148626bd7
yuvraj108c/ComfyUI-Upscaler-Tensorrt ComfyUI-Upscaler-Tensorrt f428364b8de7536244c671ff6c40409f41d7db34
Kosinkadink/ComfyUI-VideoHelperSuite ComfyUI-VideoHelperSuite 115de7a97e42a1d4babee0e8b54da070823aae0a
WASasquatch/was-node-suite-comfyui ComfyUI-WAS-Node-Suite ea935d102071d07f7e1525c573149708ad14e47d
yolain/ComfyUI-Easy-Use ComfyUI-easy-use 58e077a7b4f1ce20165d37ca40d5d18bf2ea7ffe
bvhari/ComfyUI_CFGStar ComfyUI_CFGStar 37c0b2d0f8cb0afdef1880de6027a72af9e566e5
cubiq/ComfyUI_essentials ComfyUI_essentials 9d9f4bedad89c4fd8f5c18372275780ae357851f
signerzwb/Comfyui-minimaxh3-FBcache-shendumao Comfyui-minimaxh3-FBcache-shendumao a19b53c4a5e48a07b73b0edf9a9fd546941b2609
ClownsharkBatwing/RES4LYF RES4LYF 26036f64d249a132aa7ac322f867615d21766a6b
pythongosssss/ComfyUI-Custom-Scripts comfyui-custom-scripts 609f3afaa03c25bf3eddd592fd3f773e3415a442
lbouaraba/comfyui-krea2edit comfyui-krea2edit 86f886da938984e4a4cba6b9d5e9db9bc3a86f6c
heshengtao/comfyui_LLM_party comfyui_LLM_party 3790b26a1aff91e3ee4bef98bc2ce502d3377ede
dylan121322/qwen3vl-kvcache qwen3vl-kvcache 126fc6c90d2c3f94c1ef5767616f0f7cecd62bec
rgthree/rgthree-comfy rgthree-comfy 2c5342a87f2fac6ace49de7ca5bcd467d4e8f976
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