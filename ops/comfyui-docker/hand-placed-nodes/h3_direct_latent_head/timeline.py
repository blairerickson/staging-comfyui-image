"""Exact rational timeline helpers for MiniMax H3 long-video continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


FPS = 24
AUDIO_LATENT_FPS = 40
LATENT_TIMELINE_KEY = "h3_direct_latent_timeline"
AUDIO_TIMELINE_KEY = "h3_direct_latent_audio_timeline"
AUDIO_OVERLAP_KEY = "h3_direct_latent_audio_overlap"
TIMELINE_SCHEMA = 1


def round_ratio(numerator: int, denominator: int) -> int:
    """Round an exact rational to nearest integer, ties to even.

    This matches Python's ``round`` policy without passing through binary
    floating point, so every section resolves the same absolute boundary.
    """
    numerator = int(numerator)
    denominator = int(denominator)
    if numerator < 0 or denominator <= 0:
        raise ValueError("round_ratio expects numerator >= 0 and denominator > 0")
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def timeline_boundary(frame: int, rate: int, fps: int = FPS) -> int:
    frame = int(frame)
    rate = int(rate)
    fps = int(fps)
    if frame < 0 or rate <= 0 or fps <= 0:
        raise ValueError("frame must be >= 0 and rate/fps must be positive")
    return round_ratio(frame * rate, fps)


def audio_latent_boundary(frame: int) -> int:
    return timeline_boundary(frame, AUDIO_LATENT_FPS, FPS)


def pcm_boundary(frame: int, sample_rate: int) -> int:
    return timeline_boundary(frame, int(sample_rate), FPS)


def span_on_rate(start_frame: int, end_frame: int, rate: int) -> int:
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame <= start_frame:
        raise ValueError("timeline span must have end_frame > start_frame")
    return timeline_boundary(end_frame, rate) - timeline_boundary(start_frame, rate)


def valid_window_lengths(frames: int, rate: int = AUDIO_LATENT_FPS) -> tuple[int, ...]:
    """Possible discrete lengths for a frame-aligned window on a global grid."""
    frames = int(frames)
    rate = int(rate)
    if frames <= 0 or rate <= 0:
        raise ValueError("frames and rate must be positive")
    lower, remainder = divmod(frames * rate, FPS)
    return (lower,) if remainder == 0 else (lower, lower + 1)


def section_windows(
    section_frames: int,
    context_frames: int,
    sections: int,
) -> tuple[H3Timeline, ...]:
    """Return the absolute rolling windows used by a manual long-video graph."""
    section_frames = int(section_frames)
    context_frames = int(context_frames)
    sections = int(sections)
    if section_frames <= context_frames or context_frames < 0 or sections < 1:
        raise ValueError("invalid section/context/count geometry")
    extension = section_frames - context_frames
    return tuple(
        H3Timeline(index * extension, index * extension + section_frames, index)
        for index in range(sections)
    )


@dataclass(frozen=True)
class H3Timeline:
    start_frame: int
    end_frame: int
    continuation_depth: int = 0

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("timeline start_frame must be non-negative")
        if self.end_frame <= self.start_frame:
            raise ValueError("timeline end_frame must be greater than start_frame")
        if self.continuation_depth < 0:
            raise ValueError("timeline continuation_depth must be non-negative")

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    def as_dict(self) -> dict[str, int]:
        return {
            "schema": TIMELINE_SCHEMA,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fps": FPS,
            "audio_latent_fps": AUDIO_LATENT_FPS,
            "continuation_depth": self.continuation_depth,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "H3Timeline":
        if not isinstance(value, Mapping):
            raise ValueError("timeline metadata must be a mapping")
        if int(value.get("schema", -1)) != TIMELINE_SCHEMA:
            raise ValueError("unsupported H3 timeline metadata schema")
        if int(value.get("fps", -1)) != FPS:
            raise ValueError("H3 timeline metadata must use 24 fps")
        if int(value.get("audio_latent_fps", -1)) != AUDIO_LATENT_FPS:
            raise ValueError("H3 timeline metadata must use a 40 Hz audio latent grid")
        return cls(
            start_frame=int(value["start_frame"]),
            end_frame=int(value["end_frame"]),
            continuation_depth=int(value.get("continuation_depth", 0)),
        )


def latent_timeline(latent: Mapping[str, Any], frames: int) -> H3Timeline:
    """Read a propagated timeline or bootstrap the first section at frame zero."""
    value = latent.get(LATENT_TIMELINE_KEY)
    timeline = H3Timeline(0, int(frames), 0) if value is None else H3Timeline.from_dict(value)
    if timeline.frames != int(frames):
        raise ValueError(
            "latent timeline duration does not match its H3 video stream: "
            f"metadata={timeline.frames} frames, latent={int(frames)} frames"
        )
    return timeline


__all__ = [
    "AUDIO_LATENT_FPS",
    "AUDIO_OVERLAP_KEY",
    "AUDIO_TIMELINE_KEY",
    "FPS",
    "H3Timeline",
    "LATENT_TIMELINE_KEY",
    "audio_latent_boundary",
    "latent_timeline",
    "pcm_boundary",
    "round_ratio",
    "section_windows",
    "span_on_rate",
    "timeline_boundary",
    "valid_window_lengths",
]
