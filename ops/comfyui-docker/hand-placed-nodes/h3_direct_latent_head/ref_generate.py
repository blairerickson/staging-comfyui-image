"""Dynamic native-only REF2VA long-video graph expansion."""

from __future__ import annotations

import re
from typing import Any

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .model_stack import H3LV_STACK
from .native_settings import H3LV_CONFIG
from .ref_storyboard import H3LV_REF_STORY, validate_ref_section_prompt


REFERENCE_LIMITS = {"image": 4, "video": 3, "video_audio": 3, "audio": 3}
REFERENCE_TAG = re.compile(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", re.IGNORECASE)


def _active_consecutive(values: tuple[Any, ...], label: str) -> list[Any]:
    active: list[Any] = []
    gap = False
    for index, value in enumerate(values, 1):
        if value is None:
            gap = True
        elif gap:
            raise ValueError(f"{label} inputs must be enabled consecutively from {label} 1; found a gap before {label} {index}")
        else:
            active.append(value)
    return active


def _validate_reference_tags(
    prompts: list[str],
    image_count: int,
    video_count: int,
    audio_count: int,
    reference_policy: str,
) -> None:
    limits = {"picture": image_count, "video": video_count, "audio": audio_count}
    for section, prompt in enumerate(prompts, 1):
        tags = [(kind.lower(), int(number)) for kind, number in REFERENCE_TAG.findall(prompt)]
        presents_references = section == 1 or reference_policy == "every_section"
        if section == 1 and not tags:
            raise ValueError(
                "section 1 REF2VA prompt must reference at least one active "
                "<Picture N>, <Video N>, or <Audio N>"
            )
        if tags and not presents_references:
            raise ValueError(
                f"section {section} contains a reference tag, but S1 ONLY presents references "
                "only to section 1; remove <Picture N>, <Video N>, and <Audio N> tags from "
                f"section {section} or select EVERY SECTION"
            )
        for kind, number in tags:
            available = limits[kind]
            if number < 1 or number > available:
                raise ValueError(
                    f"section {section} uses <{kind.title()} {number}>, but only {available} active {kind} reference(s) exist"
                )


def _validate_payload(config: dict[str, Any], story: dict[str, Any], stack: dict[str, Any]) -> None:
    if not isinstance(config, dict) or config.get("schema") != 2:
        raise ValueError("invalid or unsupported H3 native settings payload")
    if str(config.get("pipeline", "")).lower() != "ref2va":
        raise ValueError("REF2VA GENERATE requires a REF2VA MASTER config")
    if config.get("ref_image_size") not in {"match", "max"}:
        raise ValueError("REF2VA MASTER reference image size is invalid")
    if config.get("reference_policy") not in {"first_only", "every_section"}:
        raise ValueError("REF2VA MASTER reference policy is invalid")
    if not isinstance(story, dict) or story.get("schema") != 1 or story.get("pipeline") != "ref2va":
        raise ValueError("invalid or unsupported H3 REF2VA storyboard payload")
    if not isinstance(stack, dict) or stack.get("schema") != 3:
        raise ValueError("invalid or unsupported H3 model-stack payload")
    if str(stack.get("pipeline", "")).lower() != "ref2va":
        raise ValueError("REF2VA GENERATE requires a REF2VA model stack")
    sections = int(story.get("section_count", 0))
    prompts = story.get("prompts")
    if not 1 <= sections <= 12 or not isinstance(prompts, list) or len(prompts) != sections:
        raise ValueError("H3 REF2VA storyboard section count and prompt list disagree")
    for index, prompt in enumerate(prompts, 1):
        validate_ref_section_prompt(prompt, index)
    expected_mode = "turbo" if bool(config.get("expects_turbo")) else "native"
    if stack.get("mode") != expected_mode:
        raise ValueError("MASTER and H3 MODEL STACK disagree about native/Turbo mode")


def _section_seed(config: dict[str, Any], index: int) -> int:
    seed = int(config["base_seed"])
    if config["seed_strategy"] == "increment per section":
        seed += index
    return seed & 0xFFFFFFFFFFFFFFFF


def build_native_ref2va_graph(
    model,
    clip,
    video_vae,
    audio_vae,
    stack: dict[str, Any],
    config: dict[str, Any],
    story: dict[str, Any],
    ref_image_1=None,
    ref_image_2=None,
    ref_image_3=None,
    ref_image_4=None,
    ref_video_1=None,
    ref_video_2=None,
    ref_video_3=None,
    ref_video_audio_1=None,
    ref_video_audio_2=None,
    ref_video_audio_3=None,
    ref_audio_1=None,
    ref_audio_2=None,
    ref_audio_3=None,
):
    _validate_payload(config, story, stack)
    images = _active_consecutive(
        (ref_image_1, ref_image_2, ref_image_3, ref_image_4), "REF IMAGE"
    )
    videos = _active_consecutive((ref_video_1, ref_video_2, ref_video_3), "REF VIDEO")
    video_audios = (ref_video_audio_1, ref_video_audio_2, ref_video_audio_3)
    for index, soundtrack in enumerate(video_audios, 1):
        if soundtrack is not None and index > len(videos):
            raise ValueError(f"REF VIDEO AUDIO {index} requires REF VIDEO {index}")
    audios = _active_consecutive((ref_audio_1, ref_audio_2, ref_audio_3), "REF AUDIO")
    paired_audio_count = sum(value is not None for value in video_audios[:len(videos)])
    if not images and not videos and not audios:
        raise ValueError("REF2VA requires at least one active image, video, or audio reference")

    sections = int(story["section_count"])
    prompts = list(story["prompts"])
    reference_policy = str(config["reference_policy"])
    _validate_reference_tags(
        prompts,
        len(images),
        len(videos),
        paired_audio_count + len(audios),
        reference_policy,
    )
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
        presents_references = section == 1 or reference_policy == "every_section"
        conditioning_inputs = {
            "clip": clip,
            "vae": video_vae,
            "audio_vae": audio_vae,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": section_frames,
            "ref_image_size": config["ref_image_size"],
        }
        if presents_references:
            for ref_index, image in enumerate(images):
                conditioning_inputs[f"ref_images.ref_image_{ref_index}"] = image
            for ref_index, video in enumerate(videos):
                conditioning_inputs[f"ref_videos.ref_video_{ref_index}"] = video
                soundtrack = video_audios[ref_index]
                if soundtrack is not None:
                    conditioning_inputs[f"ref_video_audios.ref_video_audio_{ref_index}"] = soundtrack
            for ref_index, audio in enumerate(audios):
                conditioning_inputs[f"ref_audios.ref_audio_{ref_index}"] = audio
        conditioning = graph.node(
            "MiniMaxH3ReferenceToVideo",
            id=f"conditioning_s{section}",
            **conditioning_inputs,
        )
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
    images_out = graph.node("MiniMaxH3ImageSequenceJoinMany", id="image_join_once", **join_inputs).out(0)
    decoded_audio = graph.node("VAEDecodeAudio", id="audio_decode_once", samples=accumulated, vae=audio_vae)
    exact_audio = graph.node(
        "MiniMaxH3TimelineAudioTrim",
        id="audio_exact_timeline",
        audio=decoded_audio.out(0),
        timeline_latent=accumulated,
        trim_frames=0,
        max_correction_ms=50.0,
    )
    video_out = graph.node(
        "CreateVideo",
        id="final_video",
        images=images_out,
        audio=exact_audio.out(0),
        fps=24.0,
        bit_depth=8,
        color_space="sRGB",
    )
    final_frames = section_frames + (sections - 1) * (section_frames - context_frames)
    report = (
        f"REF2VA; {sections} sections; {width}x{height} native; {final_frames} frames / "
        f"{final_frames / 24.0:.3f}s; refs {len(images)} image / {len(videos)} video / "
        f"{paired_audio_count + len(audios)} audio; reference policy {reference_policy}; "
        f"reference image size {config['ref_image_size']}; "
        f"protected AV context {context_frames}f; audio history {audio_context_frames}f; "
        f"center-cut seams {left_seam_trim}F/{right_seam_trim}F; bounded video decode; "
        "one image join; audio decoded once from the cumulative AV latent"
    )
    return video_out.out(0), accumulated, final_frames, report, graph.finalize()


class H3LVREF2VAGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LVREF2VAGenerate",
            display_name="GENERATE • REF2VA Long Video",
            category="MiniMax H3 Long Video",
            description=(
                "Builds 1-12 sequential REF2VA sections with images, 24 FPS reference videos, "
                "paired video soundtracks and standalone audio references."
            ),
            inputs=[
                io.Model.Input("model", raw_link=True),
                io.Clip.Input("clip", raw_link=True),
                io.Vae.Input("video_vae", raw_link=True),
                io.Vae.Input("audio_vae", raw_link=True),
                H3LV_STACK.Input("stack"),
                H3LV_CONFIG.Input("config"),
                H3LV_REF_STORY.Input("storyboard"),
                *[io.Image.Input(f"ref_image_{index}", optional=True, raw_link=True) for index in range(1, 5)],
                *[io.Image.Input(f"ref_video_{index}", optional=True, raw_link=True) for index in range(1, 4)],
                *[io.Audio.Input(f"ref_video_audio_{index}", optional=True, raw_link=True) for index in range(1, 4)],
                *[io.Audio.Input(f"ref_audio_{index}", optional=True, raw_link=True) for index in range(1, 4)],
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
    def execute(cls, model, clip, video_vae, audio_vae, stack, config, storyboard, **references):
        video, latent, final_frames, report, expanded = build_native_ref2va_graph(
            model,
            clip,
            video_vae,
            audio_vae,
            stack,
            config,
            storyboard,
            **references,
        )
        return io.NodeOutput(video, latent, final_frames, report, expand=expanded)


__all__ = ["H3LVREF2VAGenerate", "REFERENCE_LIMITS", "build_native_ref2va_graph"]
