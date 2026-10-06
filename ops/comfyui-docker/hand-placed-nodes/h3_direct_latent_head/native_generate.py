"""Dynamic native-only FL2VA long-video graph expansion."""

from __future__ import annotations

from typing import Any

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .model_stack import H3LV_STACK
from .native_settings import H3LV_CONFIG
from .storyboard import H3LV_STORY, validate_section_prompt


def _validate_payload(config: dict[str, Any], story: dict[str, Any], stack: dict[str, Any]) -> None:
    if not isinstance(config, dict) or config.get("schema") != 2:
        raise ValueError("invalid or unsupported H3 native settings payload")
    if not isinstance(story, dict) or story.get("schema") != 2:
        raise ValueError("invalid or unsupported H3 storyboard payload")
    if not isinstance(stack, dict) or stack.get("schema") != 3:
        raise ValueError("invalid or unsupported H3 model-stack payload")
    if str(config.get("pipeline", "fl2va")).lower() != "fl2va":
        raise ValueError("FL2VA GENERATE requires an FL2VA MASTER config")
    if str(stack.get("pipeline", "fl2va")).lower() != "fl2va":
        raise ValueError("FL2VA GENERATE requires an FL2VA model stack")
    sections = int(story.get("section_count", 0))
    prompts = story.get("prompts")
    if not 1 <= sections <= 12 or not isinstance(prompts, list) or len(prompts) != sections:
        raise ValueError("H3 storyboard section count and prompt list disagree")
    for index, prompt in enumerate(prompts, 1):
        validate_section_prompt(prompt, index)
    expected_mode = "turbo" if bool(config.get("expects_turbo")) else "native"
    if stack.get("mode") != expected_mode:
        raise ValueError("MASTER and H3 MODEL STACK disagree about native/Turbo mode")


def _section_seed(config: dict[str, Any], index: int) -> int:
    seed = int(config["base_seed"])
    if config["seed_strategy"] == "increment per section":
        seed += index
    return seed & 0xFFFFFFFFFFFFFFFF


def build_native_fl2va_graph(
    model,
    clip,
    video_vae,
    audio_vae,
    stack: dict[str, Any],
    config: dict[str, Any],
    story: dict[str, Any],
    first_frame=None,
    last_frame=None,
):
    _validate_payload(config, story, stack)
    sections = int(story["section_count"])
    prompts = list(story["prompts"])
    width = int(config["width"])
    height = int(config["height"])
    section_frames = int(config["section_frames"])
    context_frames = int(config["context_frames"])
    audio_context_frames = int(config["audio_context_frames"])
    right_seam_trim = context_frames // 2
    left_seam_trim = context_frames - right_seam_trim

    graph = GraphBuilder()
    shifted = graph.node(
        "MiniMaxH3SigmaShift",
        id="sigma_shift",
        model=model,
        shift_video=float(config["shift_video"]),
        shift_audio=float(config["shift_audio"]),
    )
    sigmas = graph.node(
        "BasicScheduler",
        id="scheduler",
        model=shifted.out(0),
        scheduler=config["scheduler"],
        steps=int(config["steps"]),
        denoise=1.0,
    )
    sampler = graph.node("KSamplerSelect", id="sampler", sampler_name=config["sampler"])

    previous_sample = None
    accumulated = None
    visible_batches = []
    for index, prompt in enumerate(prompts):
        section = index + 1
        conditioning_inputs = {
            "clip": clip,
            "vae": video_vae,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": section_frames,
        }
        if section == 1 and first_frame is not None:
            conditioning_inputs["first_frame"] = first_frame
        if section == sections and last_frame is not None:
            conditioning_inputs["last_frame"] = last_frame
        conditioning = graph.node("MiniMaxH3ImageToVideo", id=f"conditioning_s{section}", **conditioning_inputs)
        positive = conditioning.out(0)
        latent = conditioning.out(1)
        if section > 1:
            head = graph.node(
                "MiniMaxH3DirectLatentHead",
                id=f"direct_head_s{section}",
                conditioning=positive,
                target_latent=latent,
                source_latent=previous_sample,
                context_length=context_frames,
                preserve_audio=True,
                audio_context_frames=audio_context_frames,
            )
            positive = head.out(0)
            latent = head.out(1)
        noise = graph.node("RandomNoise", id=f"noise_s{section}", noise_seed=_section_seed(config, index))
        guider = graph.node("BasicGuider", id=f"guider_s{section}", model=shifted.out(0), conditioning=positive)
        sampled = graph.node(
            "SamplerCustomAdvanced",
            id=f"sample_s{section}",
            noise=noise.out(0),
            guider=guider.out(0),
            sampler=sampler.out(0),
            sigmas=sigmas.out(0),
            latent_image=latent,
        )
        previous_sample = sampled.out(0)
        if accumulated is None:
            accumulated = previous_sample
        else:
            stitched = graph.node(
                "MiniMaxH3TimelineLatentStitch",
                id=f"stitch_s{section}",
                accumulated_latent=accumulated,
                next_window_latent=previous_sample,
                context_frames=context_frames,
            )
            accumulated = stitched.out(0)

    for index in range(sections):
        section = index + 1
        window = graph.node(
            "MiniMaxH3TimelineLatentWindow",
            id=f"decode_window_s{section}",
            final_av_latent=accumulated,
            section_frames=section_frames,
            context_frames=context_frames,
            section_index=section,
            sections=sections,
        )
        decoded = graph.node("VAEDecode", id=f"video_decode_s{section}", samples=window.out(0), vae=video_vae)
        trimmed = graph.node(
            "MiniMaxH3TrimImageFrames",
            id=f"visible_frames_s{section}",
            images=decoded.out(0),
            trim_start=0 if section == 1 else left_seam_trim,
            trim_end=0 if section == sections else right_seam_trim,
        )
        visible_batches.append(trimmed.out(0))

    join_inputs = {f"image{index + 1}": value for index, value in enumerate(visible_batches)}
    images = graph.node("MiniMaxH3ImageSequenceJoinMany", id="image_join_once", **join_inputs).out(0)

    decoded_audio = graph.node("VAEDecodeAudio", id="audio_decode_once", samples=accumulated, vae=audio_vae)
    exact_audio = graph.node(
        "MiniMaxH3TimelineAudioTrim",
        id="audio_exact_timeline",
        audio=decoded_audio.out(0),
        timeline_latent=accumulated,
        trim_frames=0,
        max_correction_ms=50.0,
    )
    video = graph.node(
        "CreateVideo",
        id="final_video",
        images=images,
        audio=exact_audio.out(0),
        fps=24.0,
        bit_depth=8,
        color_space="sRGB",
    )
    final_frames = section_frames + (sections - 1) * (section_frames - context_frames)
    if first_frame is None and last_frame is None:
        mode = "T2V"
    elif first_frame is None:
        mode = "last-frame guidance"
    elif last_frame is None:
        mode = "I2V"
    else:
        mode = "FL2V"
    report = (
        f"{mode}; {sections} sections; {width}x{height} native; "
        f"{final_frames} frames / {final_frames / 24.0:.3f}s; "
        f"protected AV context {context_frames}f; audio history {audio_context_frames}f; "
        f"center-cut seams {left_seam_trim}F/{right_seam_trim}F; bounded video decode; "
        "one image join; audio decoded once from the cumulative AV latent"
    )
    return video.out(0), accumulated, final_frames, report, graph.finalize()


class H3LVFL2VAGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LVFL2VAGenerate",
            display_name="GENERATE • FL2VA Long Video",
            category="MiniMax H3 Long Video",
            description=(
                "Dynamically builds 1-12 sequential FL2VA sections, transfers exact protected AV latents, "
                "stitches on the absolute timeline, decodes video in bounded windows and audio once."
            ),
            inputs=[
                io.Model.Input("model", raw_link=True),
                io.Clip.Input("clip", raw_link=True),
                io.Vae.Input("video_vae", raw_link=True),
                io.Vae.Input("audio_vae", raw_link=True),
                H3LV_STACK.Input("stack"),
                H3LV_CONFIG.Input("config"),
                H3LV_STORY.Input("storyboard"),
                io.Image.Input("first_frame", optional=True, raw_link=True),
                io.Image.Input("last_frame", optional=True, raw_link=True),
            ],
            outputs=[
                io.Video.Output("video"),
                io.Latent.Output("final_av_latent"),
                io.Int.Output("final_frames"),
                io.String.Output("report"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, stack, config, storyboard, first_frame=None, last_frame=None):
        video, latent, final_frames, report, expanded = build_native_fl2va_graph(
            model,
            clip,
            video_vae,
            audio_vae,
            stack,
            config,
            storyboard,
            first_frame,
            last_frame,
        )
        return io.NodeOutput(video, latent, final_frames, report, expand=expanded)


__all__ = ["H3LVFL2VAGenerate", "build_native_fl2va_graph"]
