"""Validated MiniMax H3 storyboards for one to twelve long-video sections."""

from __future__ import annotations

import re
from typing import Any

from comfy_api.latest import io


H3LV_STORY = io.Custom("H3LV_STORYBOARD_V2")
MAX_SECTIONS = 12
PROMPT_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)


DEFAULT_PROMPTS = (
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, one continuous medium-wide tracking shot at eye level. An adult traveler in a bright yellow raincoat, dark trousers, and black boots walks along a rain-wet old-town market street at blue hour. The camera tracks backward smoothly, keeping the traveler centered. Her shoulder-length dark hair, clothing, body proportions, natural facial appearance, lens, exposure, and the street layout remain consistent. She carries a folded paper map in her right hand and briefly glances toward warm shop-window reflections while continuing forward at a relaxed pace. Fine rain falls continuously and the wet pavement reflects the storefront lights.

overall_soundscape: Continuous light rain, steady footsteps on wet stone, distant traffic, and quiet market ambience remain at a stable volume and acoustic perspective.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a direct continuation of the same unbroken tracking shot. The same traveler in the yellow raincoat continues walking at the same pace along the same wet market street. The camera keeps tracking backward with the same lens, height, exposure, and distance. She approaches a covered arcade, shifts the folded map from her right hand to both hands, and turns smoothly beneath the awning without stopping. Her appearance, clothing, scale, direction of travel, and the geometry of the surrounding storefronts remain unchanged. Rain continues beyond the edge of the arcade while warm reflections slide naturally across the pavement.

overall_soundscape: The same light rain, footsteps, distant traffic, and subdued market ambience continue seamlessly; the rain becomes slightly softer under the awning.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a direct continuation of the same unbroken shot inside the covered arcade. The same traveler remains centered in the same yellow raincoat as the camera tracks backward at the established speed and distance. She slows slightly, unfolds the paper map while walking, studies it for a moment, then raises her gaze toward a small information kiosk ahead. Preserve her natural appearance, clothing details, hand positions, direction of movement, lighting, exposure, and the continuous spatial relationship between the arcade entrance and the kiosk. No cut, jump, or sudden change of color temperature occurs.

overall_soundscape: Footsteps and paper rustle remain clear beneath soft rain outside the arcade, low ventilation hum, and distant market voices.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a direct continuation of the same unbroken arcade shot. The same traveler reaches the information kiosk and stops naturally without changing identity, clothing, scale, or screen position. The camera performs a subtle slow push-in while keeping the established eye-level angle and stable exposure. She rests one hand on the kiosk ledge, traces a route on the unfolded map with the other hand, and compares it with the small street diagram mounted behind the ledge. The yellow raincoat remains the main color anchor; the background geometry, reflected light, and wet surfaces remain continuous with the previous section.

overall_soundscape: The same soft exterior rain and indoor arcade ambience continue, joined by gentle paper movement and one quiet tap on the kiosk ledge.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a direct continuation of the same unbroken shot. The same traveler folds the map along its existing creases, turns away from the kiosk, and resumes walking toward the far exit of the arcade. The camera eases back into the original smooth backward tracking movement, preserving the same lens, camera height, exposure, and subject distance. She steps from the covered walkway back into the light rain and heads toward a pedestrian bridge visible ahead. Her face, hair, yellow raincoat, proportions, movement direction, and the market environment remain consistent. The lighting transition is gradual and the wet pavement continues naturally across the exit.

overall_soundscape: Paper rustle fades as steady footsteps, light rain, distant traffic, and quiet market ambience return to their earlier balance.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a direct continuation and natural ending of the same unbroken tracking shot. The same traveler in the yellow raincoat walks onto the pedestrian bridge, slows beside the railing, and stops while keeping the folded map in her right hand. The camera gradually stops tracking, makes a very small slow push-in, and settles into a stable medium shot with the established lens, eye-level height, and exposure. She looks across the rain-softened city lights, then turns her gaze slightly toward the camera with a calm expression. Preserve her identity, clothing, proportions, the direction of the bridge, the wet surfaces, and the blue-hour color balance through the final frame. No cut or abrupt tonal shift occurs.

overall_soundscape: Light rain, soft wind across the bridge, distant traffic, and quiet residual footsteps form one continuous ambience and settle gently at the end.

non_diegetic_music: N/A""",
    *("",) * (MAX_SECTIONS - 6),
)


def validate_section_prompt(prompt: str, section_index: int) -> str:
    """Require the official three-field H3 prompt body without rewriting it."""
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError(f"section {section_index} prompt is empty")
    positions: list[int] = []
    matches = []
    for field in PROMPT_FIELDS:
        found = list(re.finditer(rf"(?im)^\s*{re.escape(field)}\s*:\s*", prompt))
        if len(found) != 1:
            raise ValueError(
                f"section {section_index} prompt must contain exactly one '{field}:' field"
            )
        positions.append(found[0].start())
        matches.append(found[0])
    if positions != sorted(positions):
        raise ValueError(
            f"section {section_index} prompt fields must be ordered: "
            + " -> ".join(PROMPT_FIELDS)
        )
    for field, match, next_position in zip(PROMPT_FIELDS, matches, positions[1:] + [len(prompt)]):
        if not prompt[match.end():next_position].strip():
            raise ValueError(f"section {section_index} field '{field}' is empty")
    return prompt


def build_storyboard(section_count: int, prompts: list[str]) -> tuple[dict[str, Any], str]:
    section_count = int(section_count)
    if not 1 <= section_count <= MAX_SECTIONS:
        raise ValueError(f"section_count must be 1..{MAX_SECTIONS}")
    if len(prompts) != MAX_SECTIONS:
        raise ValueError("internal storyboard prompt count mismatch")
    validated = [
        validate_section_prompt(prompt, index + 1)
        for index, prompt in enumerate(prompts[:section_count])
    ]
    story = {"schema": 2, "section_count": section_count, "prompts": validated}
    return story, f"Storyboard: {section_count} independent validated H3 prompts"


__all__ = [
    "DEFAULT_PROMPTS",
    "H3LV_STORY",
    "MAX_SECTIONS",
    "PROMPT_FIELDS",
    "build_storyboard",
    "validate_section_prompt",
]
