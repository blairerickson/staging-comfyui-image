"""Compact native-only MiniMax H3 FL2VA and REF2VA long-video extension."""

from comfy_api.latest import ComfyExtension

from .audio_trim import MiniMaxH3TimelineAudioTrim
from .h3_mask_compat import MiniMaxH3DenoiseMaskCompatNative
from .master import H3LVFL2VAMaster
from .minimax_h3_direct_latent_head import MiniMaxH3DirectLatentHead
from .model_stack import H3LVModelStack
from .native_generate import H3LVFL2VAGenerate
from .ref_generate import H3LVREF2VAGenerate
from .ref_master import H3LVREF2VAMaster
from .native_timeline import (
    MiniMaxH3ImageSequenceJoinMany,
    MiniMaxH3TimelineLatentStitch,
    MiniMaxH3TimelineLatentWindow,
    MiniMaxH3TrimImageFrames,
)


WEB_DIRECTORY = "./web"


class H3DirectLatentHeadExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            H3LVModelStack,
            H3LVFL2VAMaster,
            H3LVFL2VAGenerate,
            H3LVREF2VAMaster,
            H3LVREF2VAGenerate,
            MiniMaxH3DenoiseMaskCompatNative,
            MiniMaxH3DirectLatentHead,
            MiniMaxH3TimelineLatentStitch,
            MiniMaxH3TimelineLatentWindow,
            MiniMaxH3TrimImageFrames,
            MiniMaxH3ImageSequenceJoinMany,
            MiniMaxH3TimelineAudioTrim,
        ]


async def comfy_entrypoint() -> H3DirectLatentHeadExtension:
    return H3DirectLatentHeadExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
