"""Exact native-resolution H3 AV stitching and bounded video decode windows."""

from __future__ import annotations

from typing import Any

import torch
import comfy.nested_tensor
from comfy_api.latest import io

from .minimax_h3_direct_latent_head import (
    _extract_h3_av_streams,
    _mask_streams,
    pixel_frames_for_video_steps,
    video_steps_for_pixel_frames,
)
from .timeline import H3Timeline, LATENT_TIMELINE_KEY, audio_latent_boundary, latent_timeline


FINAL_STITCH_KEY = "h3_native_final_stitch"


def _cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous().clone()


def _audio_length(window: H3Timeline) -> int:
    return audio_latent_boundary(window.end_frame) - audio_latent_boundary(window.start_frame)


def _require_same_stream_layout(left_video, left_audio, right_video, right_audio) -> None:
    if left_video.shape[:2] != right_video.shape[:2] or left_video.shape[3:] != right_video.shape[3:]:
        raise ValueError("H3 video layout differs between stitch inputs")
    if left_video.dtype != right_video.dtype:
        raise ValueError("H3 video dtype differs between stitch inputs")
    if left_audio.shape[:3] != right_audio.shape[:3]:
        raise ValueError("H3 audio layout differs between stitch inputs")
    if left_audio.dtype != right_audio.dtype:
        raise ValueError("H3 audio dtype differs between stitch inputs")


def _overlap_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 1.0e-2
    if dtype == torch.float16:
        return 2.0e-3
    return 1.0e-5


def _require_protected_overlap(previous: torch.Tensor, current: torch.Tensor, label: str) -> float:
    if previous.shape != current.shape:
        raise ValueError(f"{label} protected overlap shape differs")
    left = _cpu_copy(previous)
    right = _cpu_copy(current)
    tolerance = max(_overlap_tolerance(left.dtype), _overlap_tolerance(right.dtype))
    difference = float((left.float() - right.float()).abs().max().item())
    if difference > tolerance:
        raise ValueError(
            f"{label} protected overlap changed: max_abs_diff={difference:.8g}, "
            f"allowed={tolerance:.8g}"
        )
    return difference


def stitch_timeline_latents(accumulated_latent: dict[str, Any], next_window_latent: dict[str, Any], context_frames: int):
    """Remove one verified protected overlap and append the new raw AV tail."""
    left_video, left_audio = _extract_h3_av_streams(accumulated_latent, "accumulated_latent")
    right_video, right_audio = _extract_h3_av_streams(next_window_latent, "next_window_latent")
    _require_same_stream_layout(left_video, left_audio, right_video, right_audio)

    left_time = latent_timeline(accumulated_latent, pixel_frames_for_video_steps(left_video.shape[2]))
    right_time = latent_timeline(next_window_latent, pixel_frames_for_video_steps(right_video.shape[2]))
    context_frames = int(context_frames)
    if left_time.end_frame - right_time.start_frame != context_frames:
        raise ValueError("H3 stitch timeline does not match the requested context")
    if right_time.end_frame <= left_time.end_frame:
        raise ValueError("next H3 window does not extend the accumulated timeline")
    if right_time.continuation_depth != left_time.continuation_depth + 1:
        raise ValueError("H3 continuation depth is not sequential")

    video_trim = video_steps_for_pixel_frames(context_frames)
    audio_trim = audio_latent_boundary(left_time.end_frame) - audio_latent_boundary(right_time.start_frame)
    if video_trim >= right_video.shape[2] or audio_trim <= 0 or audio_trim >= right_audio.shape[-1]:
        raise ValueError("H3 overlap is incompatible with the stitch inputs")
    if left_audio.shape[-1] != _audio_length(left_time) or right_audio.shape[-1] != _audio_length(right_time):
        raise ValueError("H3 audio length differs from its absolute timeline")

    video_difference = _require_protected_overlap(left_video[:, :, -video_trim:], right_video[:, :, :video_trim], "video")
    audio_difference = _require_protected_overlap(left_audio[..., -audio_trim:], right_audio[..., :audio_trim], "audio")
    output_video = torch.cat((_cpu_copy(left_video), _cpu_copy(right_video)[:, :, video_trim:]), dim=2).contiguous()
    output_audio = torch.cat((_cpu_copy(left_audio), _cpu_copy(right_audio)[..., audio_trim:]), dim=-1).contiguous()
    output_time = H3Timeline(left_time.start_frame, right_time.end_frame, right_time.continuation_depth)
    if pixel_frames_for_video_steps(output_video.shape[2]) != output_time.frames or output_audio.shape[-1] != _audio_length(output_time):
        raise RuntimeError("stitched H3 AV latent does not match its union timeline")

    output = accumulated_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((output_video, output_audio))
    if "noise_mask" in accumulated_latent or "noise_mask" in next_window_latent:
        left_vm, left_am = (_cpu_copy(x) for x in _mask_streams(accumulated_latent, left_video, left_audio))
        right_vm, right_am = (_cpu_copy(x) for x in _mask_streams(next_window_latent, right_video, right_audio))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((
            torch.cat((left_vm, right_vm[:, :, video_trim:]), dim=2).contiguous(),
            torch.cat((left_am, right_am[..., audio_trim:]), dim=-1).contiguous(),
        ))
    else:
        output.pop("noise_mask", None)
    output[LATENT_TIMELINE_KEY] = output_time.as_dict()
    sections = int(accumulated_latent.get(FINAL_STITCH_KEY, {}).get("sections", 1)) + 1
    output[FINAL_STITCH_KEY] = {"schema": 1, "sections": sections, "context_frames": context_frames}
    info = (
        f"H3 native AV stitch: {sections} sections, {output_time.frames}f; removed "
        f"{context_frames}f/{video_trim} video steps/{audio_trim} audio steps; "
        f"protected diff video={video_difference:.3g}, audio={audio_difference:.3g}; CPU"
    )
    return output, info


def slice_timeline_window(final_av_latent: dict[str, Any], section_frames: int, context_frames: int, section_index: int, sections: int):
    """Slice one original overlapping window for bounded native video-VAE decode."""
    video, audio = _extract_h3_av_streams(final_av_latent, "final_av_latent")
    final_time = latent_timeline(final_av_latent, pixel_frames_for_video_steps(video.shape[2]))
    section_frames, context_frames, section_index, sections = map(int, (section_frames, context_frames, section_index, sections))
    if final_time.start_frame != 0 or sections < 1 or not 1 <= section_index <= sections:
        raise ValueError("invalid H3 final timeline or section index")
    section_video_steps = video_steps_for_pixel_frames(section_frames)
    video_steps_for_pixel_frames(context_frames)
    extension_frames = section_frames - context_frames
    expected_final_frames = section_frames + (sections - 1) * extension_frames
    if extension_frames <= 0 or extension_frames % 17 or final_time.frames != expected_final_frames:
        raise ValueError("final H3 latent does not match the decode-window plan")

    start_frame = (section_index - 1) * extension_frames
    end_frame = start_frame + section_frames
    start_video_step = (start_frame // 17) * 5
    end_video_step = start_video_step + section_video_steps
    start_audio_step = audio_latent_boundary(start_frame)
    end_audio_step = audio_latent_boundary(end_frame)
    if end_video_step > video.shape[2] or end_audio_step > audio.shape[-1]:
        raise RuntimeError("H3 decode window exceeds the final latent")

    output_video = _cpu_copy(video[:, :, start_video_step:end_video_step])
    output_audio = _cpu_copy(audio[..., start_audio_step:end_audio_step])
    output_time = H3Timeline(start_frame, end_frame, section_index - 1)
    output = final_av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((output_video, output_audio))
    output[LATENT_TIMELINE_KEY] = output_time.as_dict()
    if "noise_mask" in final_av_latent:
        video_mask, audio_mask = _mask_streams(final_av_latent, video, audio)
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((
            _cpu_copy(video_mask[:, :, start_video_step:end_video_step]),
            _cpu_copy(audio_mask[..., start_audio_step:end_audio_step]),
        ))
    return output, f"H3 native decode window S{section_index}/{sections}: [{start_frame},{end_frame})"


class MiniMaxH3TimelineLatentStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimelineLatentStitch",
            display_name="MiniMax H3 Exact AV Stitch",
            category="MiniMax H3 Long Video/internal",
            inputs=[io.Latent.Input("accumulated_latent"), io.Latent.Input("next_window_latent"), io.Int.Input("context_frames", default=22, min=5, max=4085, step=17)],
            outputs=[io.Latent.Output("latent"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, accumulated_latent, next_window_latent, context_frames):
        return io.NodeOutput(*stitch_timeline_latents(accumulated_latent, next_window_latent, context_frames))


class MiniMaxH3TimelineLatentWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimelineLatentWindow",
            display_name="MiniMax H3 Native Decode Window",
            category="MiniMax H3 Long Video/internal",
            inputs=[
                io.Latent.Input("final_av_latent"),
                io.Int.Input("section_frames", default=141, min=5, max=4085, step=17),
                io.Int.Input("context_frames", default=22, min=5, max=4085, step=17),
                io.Int.Input("section_index", default=1, min=1, max=12),
                io.Int.Input("sections", default=6, min=1, max=12),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, final_av_latent, section_frames, context_frames, section_index, sections):
        return io.NodeOutput(*slice_timeline_window(final_av_latent, section_frames, context_frames, section_index, sections))


class MiniMaxH3TrimImageFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TrimImageFrames",
            display_name="MiniMax H3 Trim Decoded Window",
            category="MiniMax H3 Long Video/internal",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("trim_start", default=0, min=0, max=4085),
                io.Int.Input("trim_end", default=0, min=0, max=4085),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, trim_start, trim_end):
        trim_start = int(trim_start)
        trim_end = int(trim_end)
        if trim_start < 0 or trim_end < 0 or trim_start + trim_end >= images.shape[0]:
            raise ValueError(
                f"trim_start={trim_start}, trim_end={trim_end} are outside decoded batch "
                f"of {images.shape[0]} frames"
            )
        end = images.shape[0] - trim_end if trim_end else images.shape[0]
        return io.NodeOutput(images[trim_start:end].contiguous())


class MiniMaxH3ImageSequenceJoinMany(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageSequenceJoinMany",
            display_name="MiniMax H3 Exact Image Sequence Join Many",
            category="MiniMax H3 Long Video/internal",
            inputs=[
                io.Image.Input("image1"),
                *[io.Image.Input(f"image{index}", optional=True) for index in range(2, 13)],
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, image1, **kwargs):
        images = [image1]
        images.extend(
            value for index in range(2, 13)
            if (value := kwargs.get(f"image{index}")) is not None
        )
        reference = images[0]
        if reference.ndim != 4:
            raise ValueError("H3 image sequences must be [frames,height,width,channels]")
        for index, value in enumerate(images[1:], 2):
            if value.ndim != 4 or value.shape[1:] != reference.shape[1:]:
                raise ValueError(f"H3 decoded section {index} geometry differs")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"H3 decoded section {index} dtype/device differs")
        if len(images) == 1:
            return io.NodeOutput(reference.contiguous())
        return io.NodeOutput(torch.cat(images, dim=0).contiguous())


__all__ = [
    "MiniMaxH3TimelineLatentStitch",
    "MiniMaxH3TimelineLatentWindow",
    "MiniMaxH3TrimImageFrames",
    "MiniMaxH3ImageSequenceJoinMany",
    "slice_timeline_window",
    "stitch_timeline_latents",
]
