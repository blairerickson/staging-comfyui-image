"""Validated MiniMax H3 REF2VA storyboards for one to twelve sections."""

from __future__ import annotations

import re
from typing import Any

from comfy_api.latest import io

from .storyboard import MAX_SECTIONS


H3LV_REF_STORY = io.Custom("H3LV_REF_STORYBOARD_V1")
REF_PROMPT_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


DEFAULT_REF_PROMPTS = (
    """subject_definitions:
<Subject 1> is the adult traveler shown in <Picture 1>, preserving the same facial identity, shoulder-length dark hair, yellow raincoat, dark trousers, black boots, and natural body proportions.

summary:
[reference generation] The target section introduces <Subject 1> walking through a rain-wet old-town market in one continuous cinematic tracking shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the facial identity and distinctive appearance from <Picture 1> while allowing natural walking motion and expression changes.

detailed_description:
The target video uses a live-action cinematic style, realistic motion, soft blue-hour daylight, warm storefront reflections, and restrained natural color.
[Shot 1] A medium-wide eye-level shot frames <Subject 1> walking along a rain-wet old-town market street. The camera tracks backward smoothly and keeps her centered. She carries a folded paper map in her right hand, glances briefly toward the shop windows, and continues forward at a relaxed pace. Her referenced face, hair, yellow raincoat, proportions, and clothing details remain recognizable and stable. Fine rain falls continuously; storefronts and wet pavement retain coherent geometry and reflections.

overall_soundscape: Continuous light rain, steady footsteps on wet stone, distant traffic, and quiet market ambience remain at a stable volume and acoustic perspective.

non_diegetic_music: N/A""",
    """subject_definitions:
<Subject 1> is the same adult traveler established in the previous section, preserving the same facial identity, shoulder-length dark hair, yellow raincoat, dark trousers, black boots, and natural body proportions.

summary:
[reference generation] The target section continues the same unbroken market shot as <Subject 1> turns beneath a covered arcade.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the established identity and clothing while preserving the incoming pose, walking direction, framing, and motion state from the previous section.

detailed_description:
The live-action cinematic style, blue-hour lighting, realistic rain, lens, exposure, and restrained color remain unchanged.
[Shot 1] Continue directly from the previous section with no cut or reset. <Subject 1> keeps walking at the established pace while the camera tracks backward at the same height and distance. She approaches a covered arcade, shifts the folded map into both hands, and turns smoothly beneath the awning. Her face and all recognizable features remain stable without forcing the original reference pose. The market geometry and wet reflections move continuously; rain remains visible outside the arcade and becomes softer under the roof.

overall_soundscape: The same light rain, footsteps, distant traffic, and subdued market ambience continue seamlessly; rain becomes slightly softer beneath the awning.

non_diegetic_music: N/A""",
    """subject_definitions:
<Subject 1> is the same adult traveler established in the previous section, preserving the same facial identity, shoulder-length dark hair, yellow raincoat, dark trousers, black boots, and natural body proportions.

summary:
[reference generation] The target section continues inside the arcade as <Subject 1> opens the map and notices an information kiosk.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain the established identity and clothing while carrying forward the exact incoming position, movement, camera relationship, and scene state.

detailed_description:
The target video remains live-action and cinematic with the same lens, exposure, soft blue-hour illumination, and realistic surface reflections.
[Shot 1] Continue the same unbroken shot inside the covered arcade. <Subject 1> remains centered while the camera tracks backward at the established speed. She slows slightly, unfolds the map while walking, studies it briefly, and raises her gaze toward a small information kiosk ahead. Keep the face and distinctive appearance recognizable, but let the current pose follow naturally from the carried motion rather than returning to the source photograph. Preserve the arcade entrance, kiosk placement, yellow-raincoat color, and continuous spatial geometry. No cut or abrupt color change occurs.

overall_soundscape: Footsteps and paper rustle remain clear beneath soft rain outside, a low arcade hum, and distant market voices.

non_diegetic_music: N/A""",
    """subject_definitions:
<Subject 1> is the same adult traveler established in the previous section, preserving the same facial identity, shoulder-length dark hair, yellow raincoat, dark trousers, black boots, and natural body proportions.

summary:
[reference generation] The target section continues as <Subject 1> stops at the kiosk and checks the route on the map.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - preserve the established identity and clothing together with the incoming body position, hand motion, framing, lighting, and environment.

detailed_description:
The same live-action cinematic treatment, realistic motion, blue-hour color, stable exposure, and coherent arcade environment continue.
[Shot 1] Continue directly from the previous section with no new establishing composition. <Subject 1> reaches the information kiosk and stops naturally. The camera performs a subtle slow push-in from the established eye-level position. She rests one hand on the ledge and traces a route on the unfolded map with the other while comparing it with a small street diagram. Her face, hair, yellow raincoat, and proportions remain stable without copying the original reference pose. Background geometry, wet surfaces, and reflected light remain continuous with the preceding section.

overall_soundscape: Soft exterior rain and low arcade ambience continue, joined by gentle paper movement and one quiet tap on the kiosk ledge.

non_diegetic_music: N/A""",
    """subject_definitions:
<Subject 1> is the same adult traveler established in the previous section, preserving the same facial identity, shoulder-length dark hair, yellow raincoat, dark trousers, black boots, and natural body proportions.

summary:
[reference generation] The target section continues as <Subject 1> leaves the kiosk and walks from the arcade toward a pedestrian bridge.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - preserve the established identity and clothing while maintaining the previous section's finishing pose, screen position, camera distance, and environmental continuity.

detailed_description:
The live-action cinematic style, natural blue-hour lighting, established lens and exposure, and restrained color remain constant.
[Shot 1] Continue the same unbroken shot. <Subject 1> folds the map along its existing creases, turns away from the kiosk, and resumes walking toward the far exit. The camera smoothly returns to the established backward tracking motion. She steps from the covered arcade into the light rain and heads toward a pedestrian bridge visible ahead. Keep the established identity recognizable throughout, while allowing the carried pose and motion to evolve naturally. Preserve the yellow-raincoat color, market layout, wet pavement, and gradual lighting transition across the exit.

overall_soundscape: Paper rustle fades as steady footsteps, light rain, distant traffic, and quiet market ambience return to their earlier balance.

non_diegetic_music: N/A""",
    """subject_definitions:
<Subject 1> is the same adult traveler established in the previous section, preserving the same facial identity, shoulder-length dark hair, yellow raincoat, dark trousers, black boots, and natural body proportions.

summary:
[reference generation] The target section completes the continuous sequence as <Subject 1> stops on the pedestrian bridge and looks across the city.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - preserve the established identity and clothing together with the incoming movement, bridge position, camera relationship, lighting, and final scene state.

detailed_description:
The target video retains the same live-action cinematic style, realistic rain, blue-hour color balance, stable lens, exposure, and natural motion.
[Shot 1] Continue directly from the preceding section and end the same unbroken tracking shot naturally. <Subject 1> walks onto the pedestrian bridge, slows beside the railing, and stops with the folded map in her right hand. The camera gradually stops tracking, performs a very small slow push-in, and settles into a stable medium shot. She looks across the rain-softened city lights, then turns her gaze slightly toward the camera with a calm expression. Her face and distinctive features remain stable while her final pose follows from the carried motion. Preserve bridge direction, wet surfaces, and color balance through the last frame.

overall_soundscape: Light rain, soft wind across the bridge, distant traffic, and quiet residual footsteps form one continuous ambience and settle gently at the end.

non_diegetic_music: N/A""",
    *("",) * (MAX_SECTIONS - 6),
)


def validate_ref_section_prompt(prompt: str, section_index: int) -> str:
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError(f"section {section_index} prompt is empty")
    positions: list[int] = []
    matches = []
    for field in REF_PROMPT_FIELDS:
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
            + " -> ".join(REF_PROMPT_FIELDS)
        )
    for field, match, next_position in zip(REF_PROMPT_FIELDS, matches, positions[1:] + [len(prompt)]):
        if not prompt[match.end():next_position].strip():
            raise ValueError(f"section {section_index} field '{field}' is empty")
    return prompt


def build_ref_storyboard(section_count: int, prompts: list[str]) -> tuple[dict[str, Any], str]:
    section_count = int(section_count)
    if not 1 <= section_count <= MAX_SECTIONS:
        raise ValueError(f"section_count must be 1..{MAX_SECTIONS}")
    if len(prompts) != MAX_SECTIONS:
        raise ValueError("internal REF2VA storyboard prompt count mismatch")
    validated = [
        validate_ref_section_prompt(prompt, index + 1)
        for index, prompt in enumerate(prompts[:section_count])
    ]
    story = {"schema": 1, "pipeline": "ref2va", "section_count": section_count, "prompts": validated}
    return story, f"REF2VA storyboard: {section_count} independent validated prompts"


__all__ = [
    "DEFAULT_REF_PROMPTS",
    "H3LV_REF_STORY",
    "REF_PROMPT_FIELDS",
    "build_ref_storyboard",
    "validate_ref_section_prompt",
]
