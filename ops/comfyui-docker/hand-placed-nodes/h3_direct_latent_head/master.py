"""Single compact control surface for native FL2VA long-video generation."""

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
from .storyboard import DEFAULT_PROMPTS, H3LV_STORY, MAX_SECTIONS, build_storyboard


class H3LVFL2VAMaster(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        prompt_inputs = [
            io.String.Input(
                f"prompt_s{index}",
                multiline=True,
                dynamic_prompts=True,
                default=DEFAULT_PROMPTS[index - 1],
            )
            for index in range(1, MAX_SECTIONS + 1)
        ]
        return io.Schema(
            node_id="H3LVFL2VAMaster",
            display_name="MASTER • FL2VA Native Long Video",
            category="MiniMax H3 Long Video",
            description=(
                "All generation, continuity and storyboard controls in one node. Each section has its own "
                "complete H3 prompt; only the prompt selected by edit_section is shown in the UI."
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
                    tooltip="Experimental extra audio guide before a join. Keep OFF unless A/B testing proves a benefit.",
                ),
                io.Combo.Input(
                    "sampling_profile",
                    options=list(SAMPLING_PROFILES),
                    default="TURBO 8 steps / Euler / simple",
                ),
                io.Int.Input("base_seed", default=201, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input("seed_strategy", options=list(SEED_STRATEGIES), default="increment per section"),
                *prompt_inputs,
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, advanced=True),
            ],
            outputs=[
                H3LV_CONFIG.Output("config"),
                H3LV_STORY.Output("storyboard"),
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
        base_seed,
        seed_strategy,
        shift_video,
        shift_audio,
        **kwargs,
    ):
        if not 1 <= int(edit_section) <= int(section_count):
            raise ValueError("edit_section must be inside the active section count")
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
            pipeline="fl2va",
        )
        prompts = [str(kwargs.get(f"prompt_s{index}", "")) for index in range(1, MAX_SECTIONS + 1)]
        story, story_info = build_storyboard(section_count, prompts)
        final_frames = int(section_frames) + (int(section_count) - 1) * (
            int(section_frames) - int(config["context_frames"])
        )
        summary = (
            f"{settings_info}; {story_info}; final {final_frames}f / "
            f"{final_frames / 24.0:.3f}s"
        )
        return io.NodeOutput(config, story, summary)


__all__ = ["H3LVFL2VAMaster"]
