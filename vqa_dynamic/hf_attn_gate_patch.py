"""Attention temperature patching utilities for HF Qwen3-VL."""

from __future__ import annotations

import logging
from typing import List

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


def find_cross_attn_modules(model) -> List[nn.Module]:
    """Heuristic: pick text attention modules and skip vision blocks."""
    modules = []
    for name, module in model.named_modules():
        name_l = name.lower()
        if "vision" in name_l:
            continue
        if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
            modules.append(module)
    return modules


def set_attn_temperature(modules: List[nn.Module], t_attn: float) -> None:
    for module in modules:
        setattr(module, "attn_temperature", float(t_attn))


def patch_cross_attn_forward() -> bool:
    """Monkey-patch Qwen3-VL eager attention to apply temperature."""
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Could not import qwen3_vl modeling for patching: %s", exc)
        return False

    if getattr(modeling_qwen3_vl, "_attn_temp_patched", False):
        return True

    orig_fn = modeling_qwen3_vl.eager_attention_forward

    def patched_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = modeling_qwen3_vl.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3_vl.repeat_kv(value, module.num_key_value_groups)

        attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        t_attn = getattr(module, "attn_temperature", 1.0)
        if t_attn and abs(float(t_attn) - 1.0) > 1e-6:
            attn_scores = attn_scores / float(t_attn)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_scores = attn_scores + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    modeling_qwen3_vl._orig_eager_attention_forward = orig_fn
    modeling_qwen3_vl.eager_attention_forward = patched_eager_attention_forward
    modeling_qwen3_vl._attn_temp_patched = True
    LOGGER.info("Attention temperature patch active: True")
    return True
