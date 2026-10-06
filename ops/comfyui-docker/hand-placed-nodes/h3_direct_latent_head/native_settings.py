"""Compact, fail-closed native-resolution settings for H3 long video."""

from __future__ import annotations

from math import log, sqrt
from typing import Any

from comfy_api.latest import io

from .long_video_plan import validate_h3_frame_count


H3LV_CONFIG = io.Custom("H3LV_NATIVE_CONFIG_V2")
ALIGN = 32
MIN_DIMENSION = 128
MAX_DIMENSION = 2048
MAX_NATIVE_PIXELS = 1344 * 768

ASPECT_RATIOS: dict[str, float] = {
    "9:16 portrait": 9 / 16,
    "16:9 landscape": 16 / 9,
    "1:1 square": 1.0,
    "3:4 portrait": 3 / 4,
    "4:3 landscape": 4 / 3,
    "2:3 portrait": 2 / 3,
    "3:2 landscape": 3 / 2,
    "4:5 portrait": 4 / 5,
    "5:4 landscape": 5 / 4,
    "21:9 ultrawide": 21 / 9,
    "9:21 tall": 9 / 21,
    "2.39:1 cinema": 2.39,
}

RESOLUTION_PRESETS: dict[str, float | None] = {
    "0.20 MP FAST": 0.20,
    "0.31 MP BALANCED (recommended for 16 GB)": 0.31,
    "0.50 MP QUALITY": 0.50,
    "0.65 MP HEAVY": 0.65,
    "1.00 MP VERY HEAVY": 1.00,
    "CUSTOM native size": None,
}

CONTEXT_PROFILES: dict[str, int] = {
    "5F light": 5,
    "22F balanced (recommended)": 22,
    "39F strong identity / less new footage": 39,
    "56F maximum continuity / experimental": 56,
}

AUDIO_HISTORY_PROFILES: dict[str, int] = {
    "0F off (recommended)": 0,
    "24F / 1 second (experimental)": 24,
    "48F / 2 seconds (experimental)": 48,
}

SAMPLING_PROFILES: dict[str, dict[str, Any]] = {
    "TURBO 4 steps / Euler / simple": {"steps": 4, "sampler": "euler", "scheduler": "simple", "turbo": True},
    "TURBO 8 steps / Euler / simple": {"steps": 8, "sampler": "euler", "scheduler": "simple", "turbo": True},
    "NATIVE 20 steps / res_multistep / simple": {"steps": 20, "sampler": "res_multistep", "scheduler": "simple", "turbo": False},
    "NATIVE 25 steps / res_multistep / simple": {"steps": 25, "sampler": "res_multistep", "scheduler": "simple", "turbo": False},
}

SEED_STRATEGIES = ("increment per section", "fixed for every section")


def _aligned_dimension(value: Any, name: str) -> int:
    value = int(value)
    if not MIN_DIMENSION <= value <= MAX_DIMENSION:
        raise ValueError(f"{name} must be between {MIN_DIMENSION} and {MAX_DIMENSION}")
    if value % ALIGN:
        raise ValueError(f"{name}={value} must be divisible by {ALIGN}")
    return value


def _geometry_score(width: int, height: int, ratio: float, megapixels: float) -> float:
    return 5.0 * log((width / height) / ratio) ** 2 + log((width * height / 1_000_000) / megapixels) ** 2


def native_geometry(aspect_ratio: str, resolution_preset: str, manual_width: int, manual_height: int) -> tuple[int, int]:
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"unknown aspect ratio: {aspect_ratio!r}")
    if resolution_preset not in RESOLUTION_PRESETS:
        raise ValueError(f"unknown native resolution preset: {resolution_preset!r}")
    megapixels = RESOLUTION_PRESETS[resolution_preset]
    if megapixels is None:
        width = _aligned_dimension(manual_width, "manual_width")
        height = _aligned_dimension(manual_height, "manual_height")
        if width * height > MAX_NATIVE_PIXELS:
            raise ValueError(
                f"manual native canvas {width}x{height} exceeds H3's {MAX_NATIVE_PIXELS:,}-pixel area cap"
            )
        return width, height

    ratio = ASPECT_RATIOS[aspect_ratio]
    ideal_height = sqrt(megapixels * 1_000_000 / ratio)
    ideal_width = ideal_height * ratio
    candidates: list[tuple[float, int, int]] = []
    for wi in range(round(ideal_width / ALIGN) - 10, round(ideal_width / ALIGN) + 11):
        width = wi * ALIGN
        if not MIN_DIMENSION <= width <= MAX_DIMENSION:
            continue
        for hi in range(round(ideal_height / ALIGN) - 10, round(ideal_height / ALIGN) + 11):
            height = hi * ALIGN
            if MIN_DIMENSION <= height <= MAX_DIMENSION and width * height <= MAX_NATIVE_PIXELS:
                candidates.append((_geometry_score(width, height, ratio, megapixels), width, height))
    if not candidates:
        raise ValueError("no aligned H3 resolution fits this preset")
    _, width, height = min(candidates)
    return width, height


def resolve_native_settings(
    aspect_ratio: str,
    resolution_preset: str,
    manual_width: int,
    manual_height: int,
    section_frames: int,
    context_profile: str,
    audio_history: str,
    sampling_profile: str,
    shift_video: float,
    shift_audio: float,
    base_seed: int,
    seed_strategy: str,
    *,
    pipeline: str = "fl2va",
) -> tuple[dict[str, Any], str]:
    pipeline = str(pipeline).lower()
    if pipeline not in {"fl2va", "ref2va"}:
        raise ValueError(f"unknown H3 pipeline: {pipeline!r}")
    width, height = native_geometry(aspect_ratio, resolution_preset, manual_width, manual_height)
    section_frames = validate_h3_frame_count(section_frames, "section_frames")
    if not 124 <= section_frames <= 362:
        raise ValueError("section_frames must stay inside H3's trained 124-362 frame range")
    if context_profile not in CONTEXT_PROFILES:
        raise ValueError(f"unknown context profile: {context_profile!r}")
    if audio_history not in AUDIO_HISTORY_PROFILES:
        raise ValueError(f"unknown audio history: {audio_history!r}")
    if sampling_profile not in SAMPLING_PROFILES:
        raise ValueError(f"unknown sampling profile: {sampling_profile!r}")
    if seed_strategy not in SEED_STRATEGIES:
        raise ValueError(f"unknown seed strategy: {seed_strategy!r}")
    context_frames = CONTEXT_PROFILES[context_profile]
    audio_context_frames = AUDIO_HISTORY_PROFILES[audio_history]
    validate_h3_frame_count(context_frames, "context_frames")
    if context_frames >= section_frames:
        raise ValueError("context must be shorter than section_frames")
    if audio_context_frames > section_frames:
        raise ValueError("audio history cannot be longer than a section")
    if float(shift_video) <= 0 or float(shift_audio) <= 0:
        raise ValueError("sigma shifts must be positive")
    profile = dict(SAMPLING_PROFILES[sampling_profile])
    config = {
        "schema": 2,
        "pipeline": pipeline,
        "width": width,
        "height": height,
        "section_frames": section_frames,
        "context_frames": context_frames,
        "audio_context_frames": audio_context_frames,
        "sampling_profile": sampling_profile,
        "steps": profile["steps"],
        "sampler": profile["sampler"],
        "scheduler": profile["scheduler"],
        "expects_turbo": profile["turbo"],
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "base_seed": int(base_seed),
        "seed_strategy": seed_strategy,
        "fps": 24.0,
    }
    actual_mp = width * height / 1_000_000
    info = (
        f"{pipeline.upper()} NATIVE {width}x{height} ({actual_mp:.3f} MP); section {section_frames}f; "
        f"protected AV context {context_frames}f; audio history {audio_context_frames}f; "
        f"{sampling_profile}; seed {base_seed} ({seed_strategy})"
    )
    return config, info


__all__ = ["H3LV_CONFIG", "resolve_native_settings", "native_geometry"]
