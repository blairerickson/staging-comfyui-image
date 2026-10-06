"""Fail-closed MiniMax H3 denoise-mask compatibility for ComfyUI 0.34.x.

ComfyUI PR #15988 fixes a velocity/x0 contract regression by multiplying the
video and audio velocities by their local denoise masks before the outer x0
conversion.  This module applies the same algebra without editing ComfyUI.
It detects the affected forward implementation structurally, stays idle when
the upstream fix is present, and refuses unknown implementations.
"""

from __future__ import annotations

import functools
import inspect
import re
import threading

from comfy_api.latest import io


_PATCH_LOCK = threading.RLock()
_STATUS_ATTR = "_h3_direct_latent_mask_velocity_status_native"
_PATCH_MARKER = "_h3_direct_latent_mask_velocity_patch_native"
_ANNOUNCED: set[str] = set()


def classify_h3_forward_source(source: str) -> str:
    """Classify a MiniMaxH3Model.forward source as fixed, affected or unknown."""
    compact = re.sub(r"\s+", "", str(source))
    video_fixed = (
        "out[0]=out[0]*denoise_mask" in compact
        or "out[0]*=denoise_mask" in compact
    )
    audio_fixed = (
        "out[1]=out[1]*audio_denoise_mask" in compact
        or "out[1]*=audio_denoise_mask" in compact
    )
    if video_fixed and audio_fixed:
        return "upstream_fixed"

    passes_both_masks = (
        "denoise_mask=denoise_mask" in compact
        and "audio_denoise_mask=audio_denoise_mask" in compact
    )
    has_audio_carry = (
        "ifscale!=1.0" in compact
        and "audio_src*carry" in compact
        and "time_shift_sigma" in compact
    )
    if passes_both_masks and has_audio_carry:
        return "affected_0_34"
    return "unknown"


def _apply_velocity_masks(
    output,
    denoise_mask,
    audio_denoise_mask,
    audio_bias=None,
):
    """Apply PR #15988 velocity scaling, reconstructed after audio conversion."""
    if not isinstance(output, list):
        output = list(output)
    if len(output) != 2:
        raise RuntimeError("MiniMax H3 forward must return [video, audio]")
    if denoise_mask is not None:
        output[0] = output[0] * denoise_mask
    if audio_denoise_mask is not None:
        if audio_bias is None:
            output[1] = output[1] * audio_denoise_mask
        else:
            # Original 0.34 forward returns bias + gain*raw_velocity.  Scaling
            # only the velocity term exactly matches inserting the upstream fix
            # before the carry/schedule conversion.
            output[1] = audio_bias + (output[1] - audio_bias) * audio_denoise_mask
    return output


def _make_patched_forward(model_module, original_forward):
    @functools.wraps(original_forward)
    def patched_forward(
        self,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        denoise_mask=None,
        audio_denoise_mask=None,
        **kwargs,
    ):
        output = original_forward(
            self,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
            **kwargs,
        )

        audio_bias = None
        if audio_denoise_mask is not None:
            scale = float((minimax_payload or {}).get("audio_scale", 1.0))
            if scale != 1.0:
                audio_src = x[1]
                shift_v = float(
                    transformer_options.get(
                        "minimax_h3_sigma_shift_video", self.sigma_shift_video
                    )
                )
                shift_a = float(
                    transformer_options.get(
                        "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
                    )
                )
                sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
                sigma_a = model_module.time_shift_sigma(sigma_v, shift_v, shift_a)
                carry = (sigma_a / sigma_v).to(audio_src.dtype)
                audio_bias = (1.0 - scale) * (audio_src * carry)

        return _apply_velocity_masks(
            output,
            denoise_mask,
            audio_denoise_mask,
            audio_bias,
        )

    setattr(patched_forward, _PATCH_MARKER, True)
    return patched_forward


def _announce(status: str) -> None:
    if status in _ANNOUNCED:
        return
    _ANNOUNCED.add(status)
    print(f"[H3 DirectLatent] {status}")


def ensure_h3_denoise_mask_velocity_compat(model_module=None) -> str:
    """Install the narrow PR #15988 equivalent or verify the official fix."""
    if model_module is None:
        from comfy.ldm.minimax import model as model_module

    model_class = getattr(model_module, "MiniMaxH3Model", None)
    if model_class is None:
        raise RuntimeError("MiniMaxH3Model is unavailable; cannot verify mask velocity")

    with _PATCH_LOCK:
        current_forward = model_class.forward
        saved_status = getattr(model_class, _STATUS_ATTR, None)
        if saved_status:
            if saved_status.startswith("custom") and not getattr(
                current_forward, _PATCH_MARKER, False
            ):
                raise RuntimeError(
                    "MiniMaxH3Model.forward changed after the compatibility "
                    "patch; restart ComfyUI and remove the conflicting H3 patch"
                )
            _announce(saved_status)
            return saved_status

        if getattr(current_forward, _PATCH_MARKER, False):
            status = "custom PR #15988-equivalent mask fix already active"
            setattr(model_class, _STATUS_ATTR, status)
            _announce(status)
            return status

        try:
            source = inspect.getsource(inspect.unwrap(current_forward))
        except (OSError, TypeError) as error:
            raise RuntimeError(
                "Cannot inspect MiniMaxH3Model.forward; refusing an unverified "
                "denoise-mask patch"
            ) from error

        state = classify_h3_forward_source(source)
        if state == "upstream_fixed":
            status = "official MiniMax H3 mask-velocity fix detected; custom patch skipped"
            setattr(model_class, _STATUS_ATTR, status)
            _announce(status)
            return status
        if state != "affected_0_34":
            raise RuntimeError(
                "Unknown MiniMaxH3Model.forward layout. H3 DirectLatent Native "
                "will not patch it blindly; update the node pack or use a supported "
                "ComfyUI build"
            )

        model_class.forward = _make_patched_forward(model_module, current_forward)
        status = "custom PR #15988-equivalent mask fix active for affected ComfyUI 0.34.x"
        setattr(model_class, _STATUS_ATTR, status)
        _announce(status)
        return status


class MiniMaxH3DenoiseMaskCompatNative(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3DenoiseMaskCompatNative",
            display_name="MiniMax H3 Denoise Mask Compatibility",
            category="model/latent/minimax",
            description=(
                "Applies the narrow ComfyUI PR #15988 velocity fix only when the "
                "affected H3 0.34.x forward is detected. Skips itself when the "
                "official fix exists and fails closed on unknown code."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output("model"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        status = ensure_h3_denoise_mask_velocity_compat()
        return io.NodeOutput(model, status)


__all__ = [
    "MiniMaxH3DenoiseMaskCompatNative",
    "classify_h3_forward_source",
    "ensure_h3_denoise_mask_velocity_compat",
    "_apply_velocity_masks",
]
