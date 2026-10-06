"""Portable FL2VA/REF2VA loaders that build exactly one active H3 model branch."""

from __future__ import annotations

import re
from typing import Any

import folder_paths
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .native_settings import H3LV_CONFIG


H3LV_STACK = io.Custom("H3LV_MODEL_STACK_V3")

AUTO_DIFFUSION = "AUTO • H3 diffusion matched to MASTER"
LEGACY_AUTO_DIFFUSION = "AUTO • FL2VA diffusion model"
AUTO_TEXT_ENCODER = "AUTO • MiniMax H3 text encoder"
AUTO_VIDEO_VAE = "AUTO • MiniMax H3 video VAE"
AUTO_AUDIO_VAE = "AUTO • MiniMax H3 audio VAE"
AUTO_TURBO = "AUTO • H3 Turbo LoRA matched to MASTER"
LEGACY_AUTO_TURBO = "AUTO • FL2VA Turbo LoRA matched to MASTER"

SAGE_PRESET = "SAGE • Patch Sage Attention KJ"
KITCHEN_PRESET = "COMFY KITCHEN • ModelAttentionBackend"
PYTORCH_PRESET = "PYTORCH • stock attention"
OPTIMIZATION_PRESETS = (SAGE_PRESET, KITCHEN_PRESET, PYTORCH_PRESET)


def _files(category: str) -> list[str]:
    return list(folder_paths.get_filename_list(category))


def _resolve_auto(
    selected: str,
    auto_label: str,
    category: str,
    priorities: tuple[tuple[str, ...], ...],
    *,
    optional: bool = False,
) -> str:
    if selected != auto_label:
        return selected
    files = _files(category)
    lowered = [(value, value.lower().replace("-", "_").replace(" ", "_")) for value in files]
    for terms in priorities:
        matches = [value for value, normalized in lowered if all(term in normalized for term in terms)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            rendered = ", ".join(matches[:4])
            if len(matches) > 4:
                rendered += f", ... ({len(matches)} matches)"
            raise ValueError(
                f"{auto_label} found multiple equally suitable files: {rendered}. "
                "Select the intended file explicitly in H3 MODEL STACK"
            )
    if optional:
        return "NONE"
    raise ValueError(
        f"{auto_label} could not find a compatible file in models/{category}; "
        "select the file explicitly in H3 MODEL STACK"
    )


def _resolve_turbo_lora(selected: str, expected_steps: int, pipeline: str) -> str:
    if selected not in {AUTO_TURBO, LEGACY_AUTO_TURBO}:
        return selected
    step_term = f"{int(expected_steps)}step"
    if pipeline == "ref2va":
        priorities = (
            ("ref2va", "turbo", step_term),
            ("ref2v", "turbo", step_term),
            ("ref2va", "acc", f"{int(expected_steps)}s"),
            ("ref2v", "acc", f"{int(expected_steps)}s"),
            ("ref2va", "acc"),
            ("ref2v", "acc"),
            ("ref2va", "turbo"),
            ("ref2v", "turbo"),
        )
    else:
        priorities = (
            ("fl2v", "turbo", step_term),
            ("fl2va", "turbo", step_term),
            ("fl2v", "acc", f"{int(expected_steps)}s"),
            ("fl2va", "turbo"),
            ("fl2v", "turbo"),
        )
    return _resolve_auto(
        AUTO_TURBO,
        AUTO_TURBO,
        "loras",
        priorities,
        optional=True,
    )


def _declared_turbo_steps(filename: str) -> int | None:
    normalized = str(filename).lower().replace("-", "_").replace(" ", "_")
    found = re.search(r"(?:^|[/_.])([48])(?:_?steps?|s)(?:[/_.]|$)", normalized)
    return int(found.group(1)) if found else None


def _comfy_kitchen_attention_available() -> bool:
    try:
        from comfy.ldm.modules import attention

        return bool(getattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False))
    except (ImportError, AttributeError):
        return False


def _apply_attention(graph, model, preset: str):
    if preset == SAGE_PRESET:
        return graph.node(
            "PathchSageAttentionKJ",
            id="attention_sage",
            model=model,
            sage_attention="auto",
            allow_compile=False,
        ).out(0)
    if preset == KITCHEN_PRESET:
        return graph.node(
            "ModelAttentionBackend",
            id="attention_kitchen",
            model=model,
            attention="comfy kitchen attention",
        ).out(0)
    if preset == PYTORCH_PRESET:
        return graph.node(
            "ModelAttentionBackend",
            id="attention_pytorch",
            model=model,
            attention="pytorch attention",
        ).out(0)
    raise ValueError(f"unknown attention preset: {preset!r}")


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or config.get("schema") != 2:
        raise ValueError("H3 MODEL STACK requires the current MASTER config")
    if str(config.get("pipeline", "fl2va")).lower() not in {"fl2va", "ref2va"}:
        raise ValueError("H3 MODEL STACK received an unknown H3 pipeline")


class H3LVModelStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        unets = _files("diffusion_models")
        clips = _files("text_encoders")
        vaes = _files("vae")
        loras = _files("loras")
        return io.Schema(
            node_id="H3LVModelStack",
            display_name="H3 MODEL STACK • One Active Branch",
            category="MiniMax H3 Long Video",
            description=(
                "Builds only the model branch selected in MASTER: native or Turbo. "
                "Selects exactly one attention backend; Fused Modulation, H3 Memory Efficient Sage, "
                "scheduled Sol and FFN chunk patches are intentionally not used."
            ),
            inputs=[
                H3LV_CONFIG.Input("config"),
                io.Combo.Input(
                    "diffusion_model",
                    options=[AUTO_DIFFUSION, LEGACY_AUTO_DIFFUSION, *unets],
                    default=AUTO_DIFFUSION,
                ),
                io.Combo.Input("text_encoder", options=[AUTO_TEXT_ENCODER, *clips], default=AUTO_TEXT_ENCODER),
                io.Combo.Input("video_vae", options=[AUTO_VIDEO_VAE, *vaes], default=AUTO_VIDEO_VAE),
                io.Combo.Input("audio_vae", options=[AUTO_AUDIO_VAE, *vaes], default=AUTO_AUDIO_VAE),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    default="default",
                    advanced=True,
                ),
                io.Combo.Input("text_encoder_device", options=["default", "cpu"], default="default", advanced=True),
                io.Combo.Input(
                    "turbo_lora",
                    options=[AUTO_TURBO, LEGACY_AUTO_TURBO, "NONE", *loras],
                    default=AUTO_TURBO,
                ),
                io.Float.Input("turbo_strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("secondary_lora", options=["NONE", *loras], default="NONE"),
                io.Float.Input("secondary_strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input(
                    "attention_backend",
                    options=list(OPTIMIZATION_PRESETS),
                    default=SAGE_PRESET,
                    tooltip="Choose one path only. COMFY KITCHEN requires its int8 attention capability.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Clip.Output("clip"),
                io.Vae.Output("video_vae"),
                io.Vae.Output("audio_vae"),
                H3LV_STACK.Output("stack"),
                io.String.Output("summary"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(
        cls,
        config,
        diffusion_model,
        text_encoder,
        video_vae,
        audio_vae,
        weight_dtype,
        text_encoder_device,
        turbo_lora,
        turbo_strength,
        secondary_lora,
        secondary_strength,
        attention_backend,
    ):
        _validate_config(config)
        if attention_backend not in OPTIMIZATION_PRESETS:
            raise ValueError(f"unknown attention backend: {attention_backend!r}")
        if attention_backend == KITCHEN_PRESET and not _comfy_kitchen_attention_available():
            raise ValueError(
                "Comfy Kitchen int8 attention is unavailable. Select SAGE or PYTORCH, "
                "or update ComfyUI/comfy-kitchen."
            )

        pipeline = str(config.get("pipeline", "fl2va")).lower()
        if diffusion_model == LEGACY_AUTO_DIFFUSION:
            diffusion_model = AUTO_DIFFUSION
        diffusion_priorities = (
            (("minimax", "h3", "ref2va"), ("h3", "ref2va"), ("h3", "ref2v"))
            if pipeline == "ref2va"
            else (("minimax", "h3", "fl2va"), ("h3", "fl2"))
        )
        diffusion_model = _resolve_auto(
            diffusion_model,
            AUTO_DIFFUSION,
            "diffusion_models",
            diffusion_priorities,
        )
        text_encoder = _resolve_auto(
            text_encoder,
            AUTO_TEXT_ENCODER,
            "text_encoders",
            (("minimax", "h3"), ("qwen3", "h3"), ("h3",)),
        )
        video_vae = _resolve_auto(
            video_vae,
            AUTO_VIDEO_VAE,
            "vae",
            (("minimax", "h3", "video", "vae"), ("h3", "video")),
        )
        audio_vae = _resolve_auto(
            audio_vae,
            AUTO_AUDIO_VAE,
            "vae",
            (("minimax", "h3", "audio", "vae"), ("h3", "audio")),
        )

        expects_turbo = bool(config["expects_turbo"])
        expected_steps = int(config["steps"])
        resolved_turbo = "NONE"
        if expects_turbo:
            resolved_turbo = _resolve_turbo_lora(turbo_lora, expected_steps, pipeline)
            if resolved_turbo == "NONE":
                raise ValueError(
                    f"MASTER selected Turbo {expected_steps}-step sampling, but no Turbo LoRA was found or selected"
                )
            declared_steps = _declared_turbo_steps(resolved_turbo)
            if declared_steps is not None and declared_steps != expected_steps:
                raise ValueError(
                    f"Turbo LoRA filename declares {declared_steps} steps, but MASTER selected {expected_steps} steps"
                )

        graph = GraphBuilder()
        unet = graph.node("UNETLoader", id="unet", unet_name=diffusion_model, weight_dtype=weight_dtype)
        mask = graph.node("MiniMaxH3DenoiseMaskCompatNative", id="mask_compat", model=unet.out(0))
        model = mask.out(0)
        if expects_turbo:
            model = graph.node(
                "LoraLoaderModelOnly",
                id="turbo_lora",
                model=model,
                lora_name=resolved_turbo,
                strength_model=float(turbo_strength),
            ).out(0)
        if secondary_lora != "NONE":
            model = graph.node(
                "LoraLoaderModelOnly",
                id="secondary_lora",
                model=model,
                lora_name=secondary_lora,
                strength_model=float(secondary_strength),
            ).out(0)
        model = _apply_attention(graph, model, attention_backend)

        clip = graph.node("CLIPLoader", id="clip", clip_name=text_encoder, type="minimax", device=text_encoder_device)
        video = graph.node("VAELoader", id="video_vae", vae_name=video_vae)
        audio = graph.node("VAELoader", id="audio_vae", vae_name=audio_vae)
        spec = {
            "schema": 3,
            "pipeline": pipeline,
            "mode": "turbo" if expects_turbo else "native",
            "sampling_steps": expected_steps,
            "turbo_lora": resolved_turbo,
            "turbo_strength": float(turbo_strength),
            "secondary_lora": secondary_lora,
            "secondary_strength": float(secondary_strength),
            "attention_backend": attention_backend,
        }
        summary = (
            f"{pipeline.upper()} • ONE {spec['mode'].upper()} branch; "
            f"Turbo={'OFF' if resolved_turbo == 'NONE' else resolved_turbo}; "
            f"secondary={'OFF' if secondary_lora == 'NONE' else secondary_lora}; {attention_backend}"
        )
        return io.NodeOutput(
            model,
            clip.out(0),
            video.out(0),
            audio.out(0),
            spec,
            summary,
            expand=graph.finalize(),
        )


__all__ = [
    "AUTO_AUDIO_VAE",
    "AUTO_DIFFUSION",
    "AUTO_TEXT_ENCODER",
    "AUTO_TURBO",
    "AUTO_VIDEO_VAE",
    "H3LV_STACK",
    "H3LVModelStack",
    "LEGACY_AUTO_DIFFUSION",
    "LEGACY_AUTO_TURBO",
    "KITCHEN_PRESET",
    "OPTIMIZATION_PRESETS",
    "PYTORCH_PRESET",
    "SAGE_PRESET",
]
