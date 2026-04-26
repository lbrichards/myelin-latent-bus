"""Answer-start injection: edit the last position before generation.

The hook fires on the chosen layer's MLP (or attention) output during the
first `burst_steps` forward passes — i.e., the steps that produce the
opening tokens of the answer. The vector can either be added to the hidden
state at the final sequence position, scaled by alpha, or exactly replace that
position for activation-patching experiments. The first call also records the
vector's projection on the hidden state before and after injection, which is
useful as a sanity check that the edit actually moved the activation in the
expected direction.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from latent_bus.model_io import _get_decoder_layers


def _build_hook(
    vector: torch.Tensor,
    alpha: float,
    effect_data: Dict[str, float],
    burst_steps: int,
    mode: str = "add",
):
    """Closure factory for the injection hook."""
    if mode not in {"add", "replace"}:
        raise ValueError(f"Unsupported injection mode {mode!r}; choose 'add' or 'replace'.")
    call_count = {"n": 0}

    def hook_fn(module, inputs, outputs):
        call_count["n"] += 1
        if call_count["n"] > burst_steps:
            return outputs

        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        if hidden.dim() != 3:
            return outputs

        # Inject at the last sequence position (= answer-start at decode step 0).
        vector_np = vector.detach().to(dtype=torch.float32, device="cpu").numpy()
        if call_count["n"] == 1:
            h_before = hidden[0, -1, :].detach().to(dtype=torch.float32, device="cpu").numpy()
            effect_data["proj_before0"] = float(np.dot(h_before, vector_np))

        v = vector.to(hidden.device).to(hidden.dtype)
        edited = hidden.clone()
        if mode == "replace":
            edited[:, -1, :] = v
        else:
            edited[:, -1, :] = edited[:, -1, :] + alpha * v

        if call_count["n"] == 1:
            h_after = edited[0, -1, :].detach().to(dtype=torch.float32, device="cpu").numpy()
            effect_data["proj_after0"] = float(np.dot(h_after, vector_np))
            effect_data["delta_proj0"] = (
                effect_data["proj_after0"] - effect_data["proj_before0"]
            )
            effect_data["edit_l2_0"] = float(np.linalg.norm(h_after - h_before))

        if isinstance(outputs, tuple):
            return (edited,) + outputs[1:]
        return edited

    return hook_fn


def answer_start_injection(
    model,
    vector: torch.Tensor,
    alpha: float,
    layer_idx: int,
    site: str = "mlp_out",
    burst_steps: int = 1,
    mode: str = "add",
):
    """Register an answer-start injection hook.

    Returns (hook_handle, effect_data). The caller is responsible for
    calling `hook_handle.remove()` after generation. `effect_data` is
    populated during the first hook call with the projection of the
    hidden state on `vector` before and after injection.

    Parameters
    ----------
    model : transformers.PreTrainedModel
    vector : torch.Tensor
        Unit-norm fact vector with shape `(hidden_size,)`.
    alpha : float
        Scalar that scales the vector before addition.
    layer_idx : int
        Decoder layer index to inject at.
    site : {"mlp_out", "attn_out"}
        Which sub-block's output to inject into.
    burst_steps : int
        Number of forward passes to inject for. Default 1 (just the first
        decoded token).
    mode : {"add", "replace"}
        Add `alpha * vector` to the final position, or replace the final
        position exactly with `vector`.
    """
    layers = _get_decoder_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {len(layers)} layers."
        )

    target_layer = layers[layer_idx]
    effect_data: Dict[str, float] = {}
    hook = _build_hook(vector, alpha, effect_data, burst_steps, mode=mode)

    if site == "mlp_out":
        if not hasattr(target_layer, "mlp"):
            raise RuntimeError(f"No `mlp` sub-module on layer {layer_idx}.")
        handle = target_layer.mlp.register_forward_hook(hook)
    elif site == "attn_out":
        if not hasattr(target_layer, "self_attn"):
            raise RuntimeError(f"No `self_attn` sub-module on layer {layer_idx}.")
        handle = target_layer.self_attn.register_forward_hook(hook)
    else:
        raise ValueError(f"Unsupported site {site!r}; choose 'mlp_out' or 'attn_out'.")

    return handle, effect_data
