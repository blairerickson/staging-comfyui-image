"""Compact control surface for native REF2VA long-video generation."""

from __future__ import annotations

from comfy_api.latest import io

from .native_settings import (
    ASPECT_RATIOS,
    AUDIO_HISTORY_PROFILES,
    CONTEXT_PROFILES,
    H3LV_CONFIG,
    MAX_DIMENSION,
    MIN_DIMENSION,
    RESOLUTION_PRESETS,
    SAMPLING_PROFILES,
    SEED_STRATEGIES,
    resolve_native_settings,
)
from .ref_storyboard import DEFAULT_REF_PROMPTS, H3LV_REF_STORY, build_ref_storyboard
from .storyboard import MAX_SECTIONS


REF_IMAGE_SIZE_PROFILES = {
    "MATCH • generation area (recommended)": "match",
    "MAX • 2048px short edge (very heavy)": "max",
}

REFERENCE_POLICIES = {
    "S1 ONLY • best continuity (recommended)": "first_only",
    "EVERY SECTION • strongest reference adherence": "every_section",
}


class H3LVREF2VAMaster(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        prompt_inputs = [
            io.String.Input(
                f"prompt_s{index}",
                multiline=True,
                dynamic_prompts=True,
                default=DEFAULT_REF_PROMPTS[index - 1],
            )
            for index in range(1, MAX_SECTIONS + 1)
        ]
        return io.Schema(
            node_id="H3LVREF2VAMaster",
            display_name="MASTER • REF2VA Native Long Video",
            category="MiniMax H3 Long Video",
            description=(
                "Controls native REF2VA generation. Each section has a complete six-field prompt. "
                "References may be images, 24 FPS video frames, paired video soundtracks, or standalone audio."
            ),
            inputs=[
                io.Int.Input("section_count", default=6, min=1, max=MAX_SECTIONS),
                io.Int.Input("edit_section", default=1, min=1, max=MAX_SECTIONS),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS), default="9:16 portrait"),
                io.Combo.Input(
                    "resolution_preset",
                    options=list(RESOLUTION_PRESETS),
                    default="0.31 MP BALANCED (recommended for 16 GB)",
                ),
                io.Int.Input("manual_width", default=416, min=MIN_DIMENSION, max=MAX_DIMENSION, step=32),
                io.Int.Input("manual_height", default=736, min=MIN_DIMENSION, max=MAX_DIMENSION, step=32),
                io.Int.Input(
                    "section_frames",
                    default=141,
                    min=124,
                    max=362,
                    step=17,
                    tooltip="H3 trained range: 124-362 frames on the 5 + 17*k grid.",
                ),
                io.Combo.Input(
                    "context_profile",
                    options=list(CONTEXT_PROFILES),
                    default="22F balanced (recommended)",
                ),
                io.Combo.Input(
                    "audio_history",
                    options=list(AUDIO_HISTORY_PROFILES),
                    default="0F off (recommended)",
                    advanced=True,
                ),
                io.Combo.Input(
                    "sampling_profile",
                    options=list(SAMPLING_PROFILES),
                    default="TURBO 8 steps / Euler / simple",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=list(REF_IMAGE_SIZE_PROFILES),
                    default="MATCH • generation area (recommended)",
                    tooltip="MAX can improve identity fidelity but may be several times slower and use much more VRAM.",
                ),
                io.Int.Input("base_seed", default=201, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input("seed_strategy", options=list(SEED_STRATEGIES), default="increment per section"),
                *prompt_inputs,
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Combo.Input(
                    "reference_policy",
                    options=list(REFERENCE_POLICIES),
                    default="S1 ONLY • best continuity (recommended)",
                    tooltip=(
                        "S1 ONLY presents references only to section 1 and lets the protected AV latent "
                        "carry identity and motion forward. EVERY SECTION reapplies references and may "
                        "improve adherence, but can replay a source pose, scene, motion, or sound."
                    ),
                ),
            ],
            outputs=[
                H3LV_CONFIG.Output("config"),
                H3LV_REF_STORY.Output("storyboard"),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        section_count,
        edit_section,
        aspect_ratio,
        resolution_preset,
        manual_width,
        manual_height,
        section_frames,
        context_profile,
        audio_history,
        sampling_profile,
        ref_image_size,
        base_seed,
        seed_strategy,
        shift_video,
        shift_audio,
        reference_policy="S1 ONLY • best continuity (recommended)",
        **kwargs,
    ):
        if not 1 <= int(edit_section) <= int(section_count):
            raise ValueError("edit_section must be inside the active section count")
        if ref_image_size not in REF_IMAGE_SIZE_PROFILES:
            raise ValueError(f"unknown reference image size mode: {ref_image_size!r}")
        if reference_policy not in REFERENCE_POLICIES:
            raise ValueError(f"unknown reference policy: {reference_policy!r}")
        config, settings_info = resolve_native_settings(
            aspect_ratio,
            resolution_preset,
            manual_width,
            manual_height,
            section_frames,
            context_profile,
            audio_history,
            sampling_profile,
            shift_video,
            shift_audio,
            base_seed,
            seed_strategy,
            pipeline="ref2va",
        )
        config["ref_image_size"] = REF_IMAGE_SIZE_PROFILES[ref_image_size]
        config["reference_policy"] = REFERENCE_POLICIES[reference_policy]
        prompts = [str(kwargs.get(f"prompt_s{index}", "")) for index in range(1, MAX_SECTIONS + 1)]
        story, story_info = build_ref_storyboard(section_count, prompts)
        final_frames = int(section_frames) + (int(section_count) - 1) * (
            int(section_frames) - int(config["context_frames"])
        )
        summary = (
            f"{settings_info}; reference images={config['ref_image_size']}; "
            f"reference policy={config['reference_policy']}; {story_info}; "
            f"final {final_frames}f / {final_frames / 24.0:.3f}s"
        )
        return io.NodeOutput(config, story, summary)


__all__ = ["H3LVREF2VAMaster", "REFERENCE_POLICIES", "REF_IMAGE_SIZE_PROFILES"]
