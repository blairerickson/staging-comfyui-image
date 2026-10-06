"""Sample-exact final audio crop on the cumulative H3 frame timeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as torch_functional
from comfy_api.latest import io

from .minimax_h3_direct_latent_head import _extract_h3_av_streams, pixel_frames_for_video_steps
from .timeline import AUDIO_TIMELINE_KEY, H3Timeline, latent_timeline, pcm_boundary


def _validated_audio(audio: dict[str, Any]) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError("audio must be an AUDIO dictionary")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not torch.is_tensor(waveform) or waveform.ndim != 3 or not waveform.is_floating_point():
        raise ValueError("audio waveform must be a floating [batch,channels,samples] tensor")
    if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[-1] < 1 or sample_rate <= 0:
        raise ValueError("audio dimensions and sample rate must be positive")
    return waveform, sample_rate


def trim_audio_to_timeline(audio: dict[str, Any], timeline_latent: dict[str, Any], trim_frames: int, max_correction_ms: float):
    waveform, sample_rate = _validated_audio(audio)
    video, _ = _extract_h3_av_streams(timeline_latent, "timeline_latent")
    window = latent_timeline(timeline_latent, pixel_frames_for_video_steps(video.shape[2]))
    trim_frames = int(trim_frames)
    if trim_frames < 0 or trim_frames >= window.frames:
        raise ValueError(f"trim_frames must be in [0,{window.frames}), got {trim_frames}")
    maximum_correction = int(round(sample_rate * float(max_correction_ms) / 1000.0))
    if maximum_correction < 0:
        raise ValueError("max_correction_ms must be non-negative")

    segment = H3Timeline(window.start_frame + trim_frames, window.end_frame, window.continuation_depth)
    expected_window_samples = pcm_boundary(window.end_frame, sample_rate) - pcm_boundary(window.start_frame, sample_rate)
    actual_window_samples = int(waveform.shape[-1])
    correction = actual_window_samples - expected_window_samples
    if abs(correction) > maximum_correction:
        raise ValueError(
            "decoded H3 audio duration differs too much from its frame timeline: "
            f"actual={actual_window_samples}, expected={expected_window_samples}, limit={maximum_correction}"
        )

    crop_start = pcm_boundary(segment.start_frame, sample_rate) - pcm_boundary(window.start_frame, sample_rate)
    output_samples = pcm_boundary(segment.end_frame, sample_rate) - pcm_boundary(segment.start_frame, sample_rate)
    crop_end = crop_start + output_samples
    if crop_start >= actual_window_samples:
        raise ValueError("decoded H3 audio ends before the exact crop starts")
    trimmed = waveform[..., crop_start:min(crop_end, actual_window_samples)]
    missing = output_samples - int(trimmed.shape[-1])
    if missing > 0:
        if missing > maximum_correction:
            raise ValueError(f"decoded audio is {missing} samples short after exact crop")
        trimmed = torch_functional.pad(trimmed, (0, missing))
    if int(trimmed.shape[-1]) != output_samples:
        raise RuntimeError("internal H3 audio timeline crop mismatch")

    output = audio.copy()
    output["waveform"] = trimmed.contiguous()
    output[AUDIO_TIMELINE_KEY] = segment.as_dict()
    info = (
        f"H3 final audio [{segment.start_frame},{segment.end_frame}) = {output_samples} PCM "
        f"samples @ {sample_rate} Hz; decoder correction {correction:+d}"
    )
    return output, info


class MiniMaxH3TimelineAudioTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimelineAudioTrim",
            display_name="MiniMax H3 Exact Timeline Audio Trim",
            category="MiniMax H3 Long Video/internal",
            inputs=[
                io.Audio.Input("audio"),
                io.Latent.Input("timeline_latent"),
                io.Int.Input("trim_frames", default=0, min=0, max=4085),
                io.Float.Input("max_correction_ms", default=50.0, min=0.0, max=1000.0, step=1.0),
            ],
            outputs=[io.Audio.Output("audio"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, audio, timeline_latent, trim_frames, max_correction_ms):
        return io.NodeOutput(*trim_audio_to_timeline(audio, timeline_latent, trim_frames, max_correction_ms))


__all__ = ["MiniMaxH3TimelineAudioTrim", "trim_audio_to_timeline"]
