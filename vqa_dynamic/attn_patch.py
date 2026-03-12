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
            continue
        if all(hasattr(module, attr) for attr in ("query", "key", "value")):
            modules.append(module)
    return modules


def set_attn_temperature(modules: List[nn.Module], t_attn: float) -> None:
    for module in modules:
        setattr(module, "attn_temperature", float(t_attn))


def set_attn_key_bias(modules: List[nn.Module], key_indices: List[int] | None, bias: float = 0.0) -> None:
    """Set additive attention-score bias for selected key positions."""
    idx = [int(i) for i in (key_indices or []) if int(i) >= 0]
    for module in modules:
        setattr(module, "attn_key_bias_indices", idx)
        setattr(module, "attn_key_bias", float(bias))


def set_attn_capture_pre_softmax(
    modules: List[nn.Module],
    enabled: bool,
    keep_steps: bool = True,
) -> None:
    """Enable/disable reduced pre-softmax logging on patched attention modules.

    When enabled, each module stores either:
    - `attn_pre_softmax_hist`: list[(B, K)] across steps if keep_steps=True
    - `attn_pre_softmax_last`: (B, K) for the latest step if keep_steps=False
    """
    for module in modules:
        setattr(module, "attn_capture_pre_softmax", bool(enabled))
        setattr(module, "attn_capture_pre_softmax_keep_steps", bool(keep_steps))
        if enabled:
            setattr(module, "attn_pre_softmax_hist", [])
            setattr(module, "attn_pre_softmax_last", None)
        else:
            if hasattr(module, "attn_pre_softmax_hist"):
                delattr(module, "attn_pre_softmax_hist")
            if hasattr(module, "attn_pre_softmax_last"):
                delattr(module, "attn_pre_softmax_last")


def _patch_qwen3_attention() -> bool:
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
        key_bias = float(getattr(module, "attn_key_bias", 0.0) or 0.0)
        key_idx = getattr(module, "attn_key_bias_indices", None) or []
        if key_bias != 0.0 and key_idx:
            valid = [i for i in key_idx if i < attn_scores.shape[-1]]
            if valid:
                idx_t = torch.tensor(valid, device=attn_scores.device, dtype=torch.long)
                attn_scores.index_add_(
                    dim=-1,
                    index=idx_t,
                    source=torch.full(
                        (*attn_scores.shape[:-1], idx_t.numel()),
                        fill_value=key_bias,
                        device=attn_scores.device,
                        dtype=attn_scores.dtype,
                    ),
                )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_scores = attn_scores + causal_mask

        if bool(getattr(module, "attn_capture_pre_softmax", False)):
            reduced = attn_scores[:, :, -1, :].detach().float().mean(dim=1).cpu()  # (B, K)
            if bool(getattr(module, "attn_capture_pre_softmax_keep_steps", True)):
                hist = getattr(module, "attn_pre_softmax_hist", None)
                if hist is None:
                    hist = []
                    setattr(module, "attn_pre_softmax_hist", hist)
                hist.append(reduced)
            else:
                setattr(module, "attn_pre_softmax_last", reduced)

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    modeling_qwen3_vl._orig_eager_attention_forward = orig_fn
    modeling_qwen3_vl.eager_attention_forward = patched_eager_attention_forward
    modeling_qwen3_vl._attn_temp_patched = True
    LOGGER.info("Qwen3-VL attention temperature patch active: True")
    return True


def _patch_llama_attention() -> bool:
    try:
        from transformers.models.llama import modeling_llama
    except Exception as exc:
        LOGGER.warning("Could not import llama modeling for patching: %s", exc)
        return False

    if getattr(modeling_llama, "_attn_temp_patched", False):
        return True

    orig_fn = modeling_llama.eager_attention_forward

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
        key_states = modeling_llama.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_llama.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        t_attn = getattr(module, "attn_temperature", 1.0)
        if t_attn and abs(float(t_attn) - 1.0) > 1e-6:
            attn_weights = attn_weights / float(t_attn)
        key_bias = float(getattr(module, "attn_key_bias", 0.0) or 0.0)
        key_idx = getattr(module, "attn_key_bias_indices", None) or []
        if key_bias != 0.0 and key_idx:
            valid = [i for i in key_idx if i < attn_weights.shape[-1]]
            if valid:
                idx_t = torch.tensor(valid, device=attn_weights.device, dtype=torch.long)
                attn_weights.index_add_(
                    dim=-1,
                    index=idx_t,
                    source=torch.full(
                        (*attn_weights.shape[:-1], idx_t.numel()),
                        fill_value=key_bias,
                        device=attn_weights.device,
                        dtype=attn_weights.dtype,
                    ),
                )
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        if bool(getattr(module, "attn_capture_pre_softmax", False)):
            reduced = attn_weights[:, :, -1, :].detach().float().mean(dim=1).cpu()  # (B, K)
            if bool(getattr(module, "attn_capture_pre_softmax_keep_steps", True)):
                hist = getattr(module, "attn_pre_softmax_hist", None)
                if hist is None:
                    hist = []
                    setattr(module, "attn_pre_softmax_hist", hist)
                hist.append(reduced)
            else:
                setattr(module, "attn_pre_softmax_last", reduced)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    modeling_llama._orig_eager_attention_forward = orig_fn
    modeling_llama.eager_attention_forward = patched_eager_attention_forward
    modeling_llama._attn_temp_patched = True
    LOGGER.info("Llama attention temperature patch active: True")
    return True


def _patch_instructblip_qformer_attention() -> bool:
    try:
        from transformers.models.instructblip import modeling_instructblip
    except Exception as exc:
        LOGGER.warning("Could not import instructblip modeling for patching: %s", exc)
        return False

    cls = modeling_instructblip.InstructBlipQFormerMultiHeadAttention
    if getattr(cls, "_attn_temp_patched", False):
        return True

    orig_forward = cls.forward

    def patched_forward(
        self: nn.Module,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / torch.tensor(self.attention_head_size, device=attention_scores.device).sqrt()
        attention_scores_dtype = attention_scores.dtype

        if is_cross_attention:
            t_attn = getattr(self, "attn_temperature", 1.0)
            if t_attn and abs(float(t_attn) - 1.0) > 1e-6:
                attention_scores = attention_scores / float(t_attn)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores).to(attention_scores_dtype)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

    cls.forward = patched_forward
    cls._attn_temp_patched = True
    LOGGER.info("InstructBLIP Q-Former attention temperature patch active: True")
    return True


def patch_cross_attn_forward() -> bool:
    """Monkey-patch attention modules to apply temperature when supported."""
    patched = False
    if _patch_qwen3_attention():
        patched = True
    if _patch_llama_attention():
        patched = True
    if _patch_instructblip_qformer_attention():
        patched = True
    if patched:
        LOGGER.info("Attention temperature patch active: True")
    return patched
