"""Direct latent-prefix continuation for MiniMax H3 AV latents.

This native edition keeps the proven 22-frame masked video/audio prefix, carries an exact
absolute 24 fps / 40 Hz timeline, and adds an independent end-aligned audio
history block.  The latter lets H3 read what happened immediately *before*
the join instead of treating the copied prefix as the start of a new take.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch

import comfy.nested_tensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io

from .h3_mask_compat import ensure_h3_denoise_mask_velocity_compat

from .timeline import (
    AUDIO_LATENT_FPS,
    FPS,
    H3Timeline,
    LATENT_TIMELINE_KEY,
    audio_latent_boundary,
    latent_timeline,
    valid_window_lengths,
)


_LOG = logging.getLogger(__name__)
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_PLANES = 2
CONTEXT_FRAME_GRID = (5, 22, 39, 56, 73, 90)
MIN_CONTEXT_FRAMES = 5
MAX_CONTEXT_FRAMES = 4085
DEFAULT_CONTEXT_FRAMES = 22
DEFAULT_AUDIO_CONTEXT_FRAMES = 0
MAX_AUDIO_CONTEXT_FRAMES = 240


def pixel_frames_for_video_steps(video_steps: int) -> int:
    video_steps = int(video_steps)
    if video_steps < 1:
        raise ValueError("H3 video latent must contain at least one temporal step")
    period = len(FRAME_PER_TOKEN)
    cycles, remainder = divmod(video_steps, period)
    return cycles * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def video_steps_for_pixel_frames(frames: int) -> int:
    frames = int(frames)
    if frames < 5 or (frames - 5) % 17 != 0:
        raise ValueError(
            f"{frames} frames is not a phase-safe MiniMax H3 continuation run "
            "(expected 5 + 17*k)"
        )
    video_steps = 2 + 5 * ((frames - 5) // 17)
    if pixel_frames_for_video_steps(video_steps) != frames:
        raise RuntimeError(f"internal MiniMax H3 temporal mapping failed for {frames} frames")
    return video_steps


def audio_steps_for_window(start_frame: int, end_frame: int) -> int:
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame <= start_frame:
        raise ValueError("audio timeline window must be positive")
    return audio_latent_boundary(end_frame) - audio_latent_boundary(start_frame)


def audio_steps_for_pixel_frames(frames: int) -> int:
    """Compatibility helper for an initial window starting at frame zero."""
    return audio_steps_for_window(0, int(frames))


def resolve_context_frames(
    requested_context: int,
    available_source_frames: int,
    target_frames: int,
) -> int:
    requested_context = int(requested_context)
    available_source_frames = int(available_source_frames)
    target_frames = int(target_frames)
    if requested_context < MIN_CONTEXT_FRAMES or (requested_context - 5) % 17 != 0:
        raise ValueError(
            "context_length must follow the phase-safe MiniMax H3 grid 5 + 17*k "
            f"(5, 22, 39, 56, 73, 90, ...); got {requested_context}"
        )
    if requested_context > available_source_frames:
        raise ValueError(
            f"source_latent covers only {available_source_frames} frames; "
            f"cannot copy a {requested_context}-frame context"
        )
    if requested_context >= target_frames:
        raise ValueError(
            f"target_latent covers {target_frames} frames; protected context "
            f"must be shorter than the target (got {requested_context})"
        )
    return requested_context


def _require_tail_to_head_phase_alignment(source_steps: int, context_steps: int) -> None:
    source_start = int(source_steps) - int(context_steps)
    period = len(FRAME_PER_TOKEN)
    if source_start < 0:
        raise ValueError("source_latent is shorter than the requested context")
    if source_start % period != 0:
        raise ValueError(
            "MiniMax H3 source tail is not phase-aligned with a target head: "
            f"tail starts at latent step {source_start} (phase {source_start % period}); "
            "use a 5+17*k context"
        )


def _require_h3_av_mask_support() -> None:
    """Fail clearly unless the live ComfyUI has native H3 nested AV masks."""
    import comfy.model_base
    import comfy.samplers
    from comfy.ldm.minimax.model import MiniMaxH3Model

    missing: list[str] = []
    nested_type = getattr(comfy.nested_tensor, "NestedTensor", None)
    if nested_type is None or not callable(getattr(nested_type, "unbind", None)):
        missing.append("comfy.nested_tensor.NestedTensor")

    h3_model_base = getattr(comfy.model_base, "MiniMaxH3", None)
    helpers = (
        "_pool_masks_to_token_grid",
        "_token_grid_masks",
        "_denoise_mask_values",
        "_denoise_mask_conds",
        "scale_latent_inpaint",
    )
    if h3_model_base is None:
        missing.append("comfy.model_base.MiniMaxH3")
    else:
        for helper in helpers:
            if helper not in h3_model_base.__dict__ or not callable(getattr(h3_model_base, helper, None)):
                missing.append(f"MiniMaxH3.{helper}")

    try:
        forward_parameters = inspect.signature(MiniMaxH3Model.forward).parameters
    except (TypeError, ValueError):
        forward_parameters = {}
    if "audio_denoise_mask" not in forward_parameters:
        missing.append("MiniMaxH3Model.forward(audio_denoise_mask=...)")

    sampler_method = getattr(getattr(comfy.samplers, "CFGGuider", None), "sample", None)
    try:
        sampler_parameters = inspect.signature(sampler_method).parameters
        sampler_names = set(inspect.unwrap(sampler_method).__code__.co_names)
    except (AttributeError, TypeError, ValueError):
        sampler_parameters = {}
        sampler_names = set()
    if "denoise_mask" not in sampler_parameters or not {
        "is_nested",
        "prepare_mask",
        "pack_latents",
    }.issubset(sampler_names):
        missing.append("CFGGuider nested denoise-mask preprocessing")

    if missing:
        raise RuntimeError(
            "MiniMax H3 Direct Latent Head requires native H3 AV noise-mask "
            "support in ComfyUI; missing: {}. Update ComfyUI. No Guide fallback "
            "or broad runtime patch was applied.".format(", ".join(missing))
        )


@lru_cache(maxsize=1)
def _require_timeline_audio_guide_support() -> None:
    """Verify the live H3 layout accepts a fractional negative audio anchor.

    Stock ``Add Guide`` cannot create this placement, but ``PackedLayout``
    intentionally uses the numeric keyframe index without rounding.  The
    behavioural check is cheap and fails closed after an incompatible ComfyUI
    update instead of silently putting the soundtrack at the wrong time.
    """
    from comfy.ldm.minimax.model import PackedLayout

    # Probe both the one-second window and the recommended two-second default.
    for audio_steps in (40, 80):
        join_steps = 37
        start_step = join_steps - audio_steps
        resolved_frame_index = start_step / FRAME_RESCALE
        probe = torch.zeros((1, AUDIO_CHANNELS, AUDIO_PLANES, audio_steps))
        layout = PackedLayout(
            text_len=1,
            latent_t=2,
            latent_h=2,
            latent_w=2,
            audio_t=96,
            keyframes=[{
                "resolved_frame_index": resolved_frame_index,
                "audio_latent": probe,
            }],
        )
        cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond_audio"]
        target_spans = [(a, b) for a, b, kind in layout.segments if kind == "audio"]
        if len(cond_spans) != 1 or len(target_spans) != 1:
            raise RuntimeError(
                "MiniMax H3 PackedLayout no longer exposes the expected audio segments"
            )
        cond_a, cond_b = cond_spans[0]
        target_a, _ = target_spans[0]
        target_origin = float(layout.position_ids[target_a, 0])
        cond_time = layout.position_ids[cond_a:cond_b, 0]
        actual_start = float(cond_time.min()) - target_origin
        actual_end = actual_start + audio_steps
        if abs(actual_start - start_step) > 1.0e-9 or abs(actual_end - join_steps) > 1.0e-9:
            raise RuntimeError(
                "MiniMax H3 no longer preserves fractional negative audio-guide "
                "coordinates; update h3_direct_latent_head before rendering"
            )


def _validate_h3_stream_shapes(video: Any, audio: Any, label: str) -> None:
    if not torch.is_tensor(video) or not torch.is_tensor(audio):
        raise ValueError(f"{label} video/audio streams must be torch.Tensor values")
    if video.ndim != 5:
        raise ValueError(f"{label} video must be [B,24,T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 4:
        raise ValueError(f"{label} audio must be [B,32,2,T], got {tuple(audio.shape)}")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError(f"{label} must use MiniMax H3 batch size 1")
    if video.shape[1] != VIDEO_CHANNELS:
        raise ValueError(f"{label} video must have {VIDEO_CHANNELS} channels")
    if audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_PLANES:
        raise ValueError(
            f"{label} audio must be [1,{AUDIO_CHANNELS},{AUDIO_PLANES},T], "
            f"got {tuple(audio.shape)}"
        )
    if video.shape[2] < 2 or video.shape[2] % len(FRAME_PER_TOKEN) != 2:
        raise ValueError(f"{label} video temporal length is not an H3 5k+2 latent run")
    if audio.shape[-1] < 1:
        raise ValueError(f"{label} audio temporal length must be positive")
    if not video.is_floating_point() or not audio.is_floating_point():
        raise ValueError(f"{label} H3 streams must be floating point")


def _extract_h3_av_streams(latent: dict[str, Any], label: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"{label} must be a LATENT dictionary with 'samples'")
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False) or not callable(getattr(samples, "unbind", None)):
        raise ValueError(f"{label} must contain a MiniMax H3 NestedTensor pair")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError(f"{label} must contain exactly video and audio streams")
    video, audio = streams
    _validate_h3_stream_shapes(video, audio, label)
    return video, audio


def _validate_audio_grid(
    video: torch.Tensor,
    audio: torch.Tensor,
    timeline: H3Timeline,
    label: str,
) -> None:
    frames = pixel_frames_for_video_steps(video.shape[2])
    if timeline.frames != frames:
        raise ValueError(f"{label} timeline/latent frame count differs")
    expected = audio_steps_for_window(timeline.start_frame, timeline.end_frame)
    if audio.shape[-1] != expected:
        raise ValueError(
            f"{label} has {audio.shape[-1]} audio steps for absolute frames "
            f"[{timeline.start_frame},{timeline.end_frame}); expected {expected}"
        )


def _require_video_copy_compatibility(source: torch.Tensor, target: torch.Tensor) -> None:
    if (source.shape[1], source.shape[3], source.shape[4]) != (
        target.shape[1], target.shape[3], target.shape[4]
    ):
        raise ValueError(
            "source/target H3 video geometry differs; automatic latent resize is unsupported"
        )
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("source/target video dtype and device must match exactly")


def _require_audio_copy_compatibility(source: torch.Tensor, target: torch.Tensor) -> None:
    if source.shape[1:3] != target.shape[1:3]:
        raise ValueError("source/target H3 audio latent layout differs")
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("source/target audio dtype and device must match exactly")


def _align_target_audio_to_timeline(
    target_audio: torch.Tensor,
    target_time: H3Timeline,
) -> tuple[torch.Tensor, int]:
    """Fit stock H3 audio to its absolute window without resampling.

    ComfyUI creates target audio from ``round(section_frames * 40 / 24)``.
    An absolute rolling window can require the neighbouring integer instead.
    The discrepancy is mathematically limited to one latent step.  The new
    step is open/noisy (mask=1) and will therefore be generated by the sampler.
    """
    actual = int(target_audio.shape[-1])
    local_lengths = valid_window_lengths(target_time.frames)
    if actual not in local_lengths:
        raise ValueError(
            f"target_latent has {actual} audio steps for {target_time.frames} frames; "
            f"expected a stock H3 length in {local_lengths}"
        )
    expected = audio_steps_for_window(target_time.start_frame, target_time.end_frame)
    correction = expected - actual
    if correction == 0:
        return target_audio, 0
    if correction == 1:
        padding = torch.zeros(
            (*target_audio.shape[:-1], 1),
            device=target_audio.device,
            dtype=target_audio.dtype,
        )
        return torch.cat((target_audio, padding), dim=-1), correction
    if correction == -1:
        return target_audio[..., :-1].contiguous(), correction
    raise RuntimeError(
        "absolute H3 audio-grid correction exceeded one latent step: "
        f"actual={actual}, expected={expected}"
    )


def _broadcast_mask(part: torch.Tensor, shape: tuple[int, ...], label: str) -> torch.Tensor:
    try:
        return torch.broadcast_to(part, shape).clone().to(dtype=torch.float32)
    except RuntimeError as exact_error:
        if part.ndim != len(shape) or abs(int(part.shape[-1]) - int(shape[-1])) > 1:
            raise ValueError(f"target_latent {label} mask cannot broadcast to H3 target") from exact_error
        intermediate_shape = (*shape[:-1], int(part.shape[-1]))
        try:
            fitted = torch.broadcast_to(part, intermediate_shape).clone().to(dtype=torch.float32)
        except RuntimeError as error:
            raise ValueError(f"target_latent {label} mask cannot broadcast to H3 target") from error
        if fitted.shape[-1] > shape[-1]:
            return fitted[..., : shape[-1]].contiguous()
        padding = torch.ones(
            (*fitted.shape[:-1], shape[-1] - fitted.shape[-1]),
            device=fitted.device,
            dtype=fitted.dtype,
        )
        return torch.cat((fitted, padding), dim=-1)


def _mask_streams(
    target_latent: dict[str, Any],
    target_video: torch.Tensor,
    target_audio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_shape = (1, 1, target_video.shape[2], target_video.shape[3], target_video.shape[4])
    audio_shape = (1, 1, target_audio.shape[2], target_audio.shape[3])
    existing = target_latent.get("noise_mask")
    if existing is None:
        return (
            torch.ones(video_shape, device=target_video.device, dtype=torch.float32),
            torch.ones(audio_shape, device=target_audio.device, dtype=torch.float32),
        )
    if getattr(existing, "is_nested", False) and callable(getattr(existing, "unbind", None)):
        parts = list(existing.unbind())
    elif isinstance(existing, (tuple, list)):
        parts = list(existing)
    else:
        raise ValueError("target_latent noise_mask must be a nested H3 AV mask")
    if len(parts) != 2 or not all(torch.is_tensor(part) for part in parts):
        raise ValueError("target_latent noise_mask must contain two tensors")
    video_mask = _broadcast_mask(parts[0], video_shape, "video")
    audio_mask = _broadcast_mask(parts[1], audio_shape, "audio")
    return video_mask, audio_mask


def drop_keyframes_inside_prefix(conditioning: Sequence[Any], prefix_frames: int) -> list[list[Any]]:
    """Remove only conflicting keyframes; preserve REF2VA ``minimax_refs``."""
    if not isinstance(conditioning, Sequence) or isinstance(conditioning, (str, bytes)):
        raise ValueError("conditioning must be a sequence")
    output: list[list[Any]] = []
    dropped = 0
    for index, entry in enumerate(conditioning):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"conditioning entry {index} must be [embedding, metadata]")
        embedding, metadata = entry
        if not isinstance(metadata, dict):
            raise ValueError(f"conditioning entry {index} metadata must be a dict")
        copied = metadata.copy()
        if "minimax_keyframes" in metadata:
            keyframes = metadata.get("minimax_keyframes") or []
            if not isinstance(keyframes, (list, tuple)):
                raise ValueError("minimax_keyframes must be a sequence")
            kept = []
            for keyframe in keyframes:
                if not isinstance(keyframe, dict):
                    raise ValueError("each minimax_keyframe must be a dict")
                raw_position = keyframe.get("resolved_frame_index", keyframe.get("frame_index", 0))
                try:
                    position = float(raw_position)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"keyframe position {raw_position!r} is not numeric") from error
                if 0 <= position < int(prefix_frames):
                    dropped += 1
                else:
                    kept.append(keyframe)
            copied["minimax_keyframes"] = kept
        output.append([embedding, copied])
    if dropped:
        _LOG.warning(
            "MiniMax H3 Direct Latent Head removed %d keyframe(s) inside protected prefix",
            dropped,
        )
    return output


def add_end_aligned_audio_context(
    conditioning: Sequence[Any],
    source_audio: torch.Tensor,
    source_time: H3Timeline,
    join_audio_steps: int,
    audio_context_frames: int,
) -> tuple[list[list[Any]], int, float]:
    """Append an audio-only H3 keyframe that ends exactly at the join.

    The copied/masked AV prefix still owns the visible 22-frame head.  This
    extra conditioning block reaches farther backwards and costs no delivered
    frames.  Multiples of three video frames map to whole 40 Hz audio steps.
    """
    audio_context_frames = int(audio_context_frames)
    if audio_context_frames == 0:
        return [list(entry) for entry in conditioning], 0, 0.0
    if audio_context_frames < 3 or audio_context_frames > MAX_AUDIO_CONTEXT_FRAMES:
        raise ValueError(
            f"audio_context_frames must be 0 or 3..{MAX_AUDIO_CONTEXT_FRAMES}"
        )
    if audio_context_frames % 3:
        raise ValueError(
            "audio_context_frames must be divisible by 3 so it lands exactly "
            "on H3's 40 Hz audio grid; recommended: 48 for stronger continuity"
        )
    if audio_context_frames > source_time.frames:
        raise ValueError(
            f"source_latent covers {source_time.frames} frames; cannot carry "
            f"{audio_context_frames} frames of audio history"
        )

    audio_start = source_time.end_frame - audio_context_frames
    context_steps = audio_steps_for_window(audio_start, source_time.end_frame)
    expected_steps = audio_context_frames * AUDIO_LATENT_FPS // FPS
    if context_steps != expected_steps:
        raise RuntimeError("exact H3 audio-context grid calculation disagrees")
    if context_steps > source_audio.shape[-1]:
        raise ValueError("source audio latent is shorter than the requested history")

    _require_timeline_audio_guide_support()
    audio_tail = source_audio[..., -context_steps:].clone()
    start_step = int(join_audio_steps) - context_steps
    resolved_frame_index = start_step / FRAME_RESCALE

    output: list[list[Any]] = []
    for index, entry in enumerate(conditioning):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"conditioning entry {index} must be [embedding, metadata]")
        embedding, metadata = entry
        if not isinstance(metadata, dict):
            raise ValueError(f"conditioning entry {index} metadata must be a dict")
        copied = metadata.copy()
        keyframes = list(copied.get("minimax_keyframes") or [])
        keyframes.append({
            "resolved_frame_index": resolved_frame_index,
            "audio_latent": audio_tail.clone(),
        })
        copied["minimax_keyframes"] = keyframes
        output.append([embedding, copied])
    return output, context_steps, resolved_frame_index


def apply_direct_latent_head(
    conditioning,
    target_latent: dict[str, Any],
    source_latent: dict[str, Any],
    context_length: int,
    preserve_audio: bool,
    audio_context_frames: int = DEFAULT_AUDIO_CONTEXT_FRAMES,
) -> tuple[list[list[Any]], dict[str, Any], int, str]:
    _require_h3_av_mask_support()
    mask_velocity_status = ensure_h3_denoise_mask_velocity_compat()
    target_video, target_audio = _extract_h3_av_streams(target_latent, "target_latent")
    source_video, source_audio = _extract_h3_av_streams(source_latent, "source_latent")
    target_frames = pixel_frames_for_video_steps(target_video.shape[2])
    source_frames = pixel_frames_for_video_steps(source_video.shape[2])
    frames = resolve_context_frames(context_length, source_frames, target_frames)
    video_steps = video_steps_for_pixel_frames(frames)
    _require_tail_to_head_phase_alignment(source_video.shape[2], video_steps)

    source_time = latent_timeline(source_latent, source_frames)
    target_start = source_time.end_frame - frames
    target_time = H3Timeline(
        target_start,
        target_start + target_frames,
        source_time.continuation_depth + 1,
    )

    target_audio, audio_grid_correction = _align_target_audio_to_timeline(
        target_audio, target_time
    )

    _require_video_copy_compatibility(source_video, target_video)
    out_video = target_video.clone()
    out_audio = target_audio.clone()
    out_video[:, :, :video_steps] = source_video[:, :, -video_steps:].clone()

    audio_steps = 0
    if bool(preserve_audio):
        _validate_audio_grid(source_video, source_audio, source_time, "source_latent")
        _validate_audio_grid(target_video, target_audio, target_time, "target_latent")
        audio_steps = audio_steps_for_window(target_time.start_frame, target_time.start_frame + frames)
        source_audio_steps = audio_steps_for_window(source_time.end_frame - frames, source_time.end_frame)
        if source_audio_steps != audio_steps:
            raise RuntimeError("source-tail and target-head audio boundaries disagree")
        if audio_steps >= target_audio.shape[-1] or audio_steps > source_audio.shape[-1]:
            raise ValueError("protected audio prefix is incompatible with source/target length")
        _require_audio_copy_compatibility(source_audio, target_audio)
        out_audio[..., :audio_steps] = source_audio[..., -audio_steps:].clone()

    video_mask, audio_mask = _mask_streams(target_latent, target_video, target_audio)
    video_mask[:, :, :video_steps] = 0.0
    if bool(preserve_audio):
        audio_mask[..., :audio_steps] = 0.0

    out_latent = target_latent.copy()
    out_latent["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
    out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    out_latent[LATENT_TIMELINE_KEY] = target_time.as_dict()
    out_conditioning = drop_keyframes_inside_prefix(conditioning, frames)
    audio_history_steps = 0
    audio_history_index = 0.0
    if bool(preserve_audio) and int(audio_context_frames) > 0:
        out_conditioning, audio_history_steps, audio_history_index = (
            add_end_aligned_audio_context(
                out_conditioning,
                source_audio,
                source_time,
                audio_steps,
                audio_context_frames,
            )
        )

    audio_info = f"audio {audio_steps} steps" if bool(preserve_audio) else "audio open"
    info = (
        f"Direct H3 Native: absolute frames [{target_time.start_frame},{target_time.end_frame}), "
        f"context {frames}f/{video_steps} video steps/{audio_info}, "
        f"audio-grid {audio_grid_correction:+d}, audio history "
        f"{int(audio_context_frames)}f/{audio_history_steps} steps ending at join "
        f"(index {audio_history_index:.6g}), depth {target_time.continuation_depth}; "
        f"{mask_velocity_status}"
    )
    return out_conditioning, out_latent, frames, info


class MiniMaxH3DirectLatentHead(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3DirectLatentHead",
            display_name="MiniMax H3 Direct Latent Head",
            category="model/latent/minimax",
            description=(
                "Copies/protects a phase-aligned 22F H3 AV tail, adds an independent "
                "end-aligned audio history guide, and propagates an exact cumulative "
                "24 fps / 40 Hz timeline."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("target_latent"),
                io.Latent.Input("source_latent"),
                io.Int.Input(
                    "context_length",
                    default=DEFAULT_CONTEXT_FRAMES,
                    min=MIN_CONTEXT_FRAMES,
                    max=MAX_CONTEXT_FRAMES,
                    step=17,
                    tooltip="Phase-safe H3 grid: 5 + 17*k. Recommended: 22.",
                ),
                io.Boolean.Input(
                    "preserve_audio",
                    default=True,
                    tooltip="Copy/protect the exact cumulative audio-latent prefix.",
                ),
                io.Int.Input(
                    "audio_context_frames",
                    default=DEFAULT_AUDIO_CONTEXT_FRAMES,
                    min=0,
                    max=MAX_AUDIO_CONTEXT_FRAMES,
                    step=3,
                    tooltip=(
                        "Independent tail-audio history ending at the video join. "
                        "Use 48 (two exact seconds) for stronger continuity; 24 is "
                        "the lighter one-second profile; 0 disables the extra guide."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
                io.Latent.Output("latent"),
                io.Int.Output("trim_frames"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        target_latent,
        source_latent,
        context_length,
        preserve_audio,
        audio_context_frames=DEFAULT_AUDIO_CONTEXT_FRAMES,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *apply_direct_latent_head(
                conditioning,
                target_latent,
                source_latent,
                context_length,
                preserve_audio,
                audio_context_frames,
            )
        )


__all__ = [
    "MiniMaxH3DirectLatentHead",
    "apply_direct_latent_head",
    "add_end_aligned_audio_context",
    "audio_steps_for_pixel_frames",
    "audio_steps_for_window",
    "drop_keyframes_inside_prefix",
    "pixel_frames_for_video_steps",
    "_align_target_audio_to_timeline",
    "video_steps_for_pixel_frames",
]
