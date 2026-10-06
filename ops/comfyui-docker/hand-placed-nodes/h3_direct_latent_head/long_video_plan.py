"""One-control geometry planner for manual MiniMax H3 long-video graphs."""

from __future__ import annotations

from .timeline import FPS, audio_latent_boundary, pcm_boundary, section_windows


def validate_h3_frame_count(value: int, label: str) -> int:
    value = int(value)
    if value < 5 or (value - 5) % 17 != 0:
        raise ValueError(
            f"{label}={value} is not phase-safe for MiniMax H3; "
            "use 5 + 17*k (for example 124, 141, 158, ..., 260)"
        )
    return value


def plan_long_video(
    section_frames: int,
    context_frames: int,
    sections: int,
) -> tuple[int, int, int, str]:
    section_frames = validate_h3_frame_count(section_frames, "section_frames")
    context_frames = validate_h3_frame_count(context_frames, "context_frames")
    sections = int(sections)
    if sections < 1:
        raise ValueError("sections must be at least 1")
    if context_frames >= section_frames:
        raise ValueError("context_frames must be shorter than section_frames")

    windows = section_windows(section_frames, context_frames, sections)
    extension_frames = section_frames - context_frames
    final_frames = windows[-1].end_frame
    seams = [window.end_frame for window in windows[:-1]]
    seam_text = ", ".join(
        f"{frame}f/{frame / FPS:.6f}s" for frame in seams
    ) or "none"
    audio_schedule = [
        audio_latent_boundary(window.end_frame)
        - audio_latent_boundary(window.start_frame)
        for window in windows
    ]
    final_pcm_32k = pcm_boundary(final_frames, 32000)
    info = (
        f"H3 long-video plan: {sections} x {section_frames}f, context {context_frames}f, "
        f"extension {extension_frames}f; final {final_frames}f/{final_frames / FPS:.6f}s; "
        f"seams [{seam_text}]; audio steps/window {audio_schedule}; "
        f"final PCM@32k {final_pcm_32k}"
    )
    return section_frames, extension_frames, final_frames, info


__all__ = ["plan_long_video", "validate_h3_frame_count"]
