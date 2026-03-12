"""Custom generate() sampling loop with step-wise temperature control only."""

from __future__ import annotations

from typing import Optional
import re
import time

import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper
from transformers.generation.utils import (
    GenerationConfig,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from vqa_dynamic.attn_patch import set_attn_temperature, set_attn_key_bias


# ── Inline stubs for removed modules (soft_2x2, attn_gate_smoother) ──────────
# These were part of the old attention-gating concept. The code paths that use
# them (temp_mode="soft2x2", gate smoothing) are inactive in current subspace
# intervention experiments but remain in the generate loop for compatibility.

class Soft2x2Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class Soft2x2State:
    pass

def soft_2x2_update(ht, ha, params, state):
    return 1.0, 1.0, {}

class AttnGateSmootherConfig:
    def __init__(self, mode="ema", ema_alpha=0.3, window_size=3):
        self.mode = mode
        self.ema_alpha = ema_alpha
        self.window_size = window_size

class AttnGateSmoother:
    def __init__(self, config):
        self._ema = None
        self._alpha = config.ema_alpha

    def update(self, val):
        if self._ema is None:
            self._ema = val
        else:
            self._ema = self._alpha * val + (1 - self._alpha) * self._ema
        return self._ema


def _find_decoder_layers(model):
    """Find transformer decoder layer ModuleList for hook registration."""
    for path in [
        ("model", "language_model", "layers"),    # LLaVA (llava-hf)
        ("language_model", "model", "layers"),    # LLaVA alt
        ("model", "model", "layers"),             # some wrappers
        ("model", "layers"),                      # direct LLaMA
        ("language_model", "layers"),             # another variant
        ("transformer", "h"),                     # GPT-2 style
    ]:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList):
            return obj
    return None


class SubspaceInterventionHook:
    """Forward hook: risk-gated correct-subspace attraction at a target layer.

    Centered PCA formulation:
        z = h - mu              (center using correct-sample mean)
        z_proj = Uk Uk^T z      (project onto correct subspace)
        z_orth = z - z_proj     (orthogonal residual = mismatch)
        R(h) = ||z_orth|| / ||z||  (risk score)
        h' = h - alpha * z_orth    (reduce mismatch)
    where alpha = lambda * clip((R - tau) / (tau_max - tau), 0, 1).
    """

    def __init__(self, mu: torch.Tensor, uk: torch.Tensor,
                 tau: float, tau_max: float, lambda_max: float):
        self.mu = mu                    # [d]
        self.uk = uk                    # [d, k]
        self.tau = float(tau)
        self.tau_max = float(tau_max)
        self.lambda_max = float(lambda_max)
        self.enabled = False
        self.prefill_risk = None
        self._reset()

    def _reset(self):
        self.last_risk = None
        self.last_alpha = 0.0
        self.last_applied = False
        self.last_orth_before = None
        self.last_orth_after = None

    def reset_step(self):
        self._reset()

    def reset_sample(self):
        """Call before each new sample generation."""
        self._reset()
        self.prefill_risk = None
        self._step_count = 0
        self._decode_count = 0

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        if isinstance(output, tuple):
            h_full = output[0]
        else:
            h_full = output

        # Detect prefill (seq_len > 1) vs decode (seq_len == 1)
        is_prefill = h_full.shape[1] > 1
        self._step_count = getattr(self, "_step_count", 0) + 1

        h_last = h_full[:, -1:, :].to(torch.float32)       # [B, 1, d]
        mu = self.mu.to(device=h_last.device, dtype=torch.float32)
        uk = self.uk.to(device=h_last.device, dtype=torch.float32)

        # Center, project, compute orthogonal residual
        z = h_last - mu.unsqueeze(0).unsqueeze(0)           # [B, 1, d]
        z_proj = (z @ uk) @ uk.T                            # [B, 1, d]
        z_orth = z - z_proj                                 # [B, 1, d]

        # Risk: R(h) = ||z_orth|| / ||z||
        z_norm = torch.linalg.norm(z, dim=-1).clamp(min=1e-8)
        orth_norm = torch.linalg.norm(z_orth, dim=-1)
        risk = float((orth_norm / z_norm).mean().item())

        self.last_risk = risk
        self.last_orth_before = float(orth_norm.mean().item())

        # Record prefill risk separately (matches offline extraction scale)
        if is_prefill:
            self.prefill_risk = risk
        else:
            self._decode_count = getattr(self, "_decode_count", 0) + 1

        # Intervene on PREFILL only (yes/no decision is made here).
        # Decode steps are too late — the answer token is already determined.
        if not is_prefill:
            self.last_applied = False
            self.last_alpha = 0.0
            self.last_orth_after = self.last_orth_before
            return output

        if risk <= self.tau:
            self.last_applied = False
            self.last_alpha = 0.0
            self.last_orth_after = self.last_orth_before
            return output

        # Risk-proportional alpha
        alpha = self.lambda_max * max(0.0, min(1.0,
            (risk - self.tau) / max(self.tau_max - self.tau, 1e-8)))

        self.last_alpha = alpha
        self.last_applied = True

        # Realign: remove alpha fraction of orthogonal mismatch
        h_new = h_last - alpha * z_orth                     # [B, 1, d]

        # Diagnostic: recompute orth after intervention
        z_new = h_new - mu.unsqueeze(0).unsqueeze(0)
        z_new_orth = z_new - (z_new @ uk) @ uk.T
        self.last_orth_after = float(torch.linalg.norm(z_new_orth, dim=-1).mean().item())

        h_modified = h_full.clone()
        h_modified[:, -1:, :] = h_new.to(h_full.dtype)

        if isinstance(output, tuple):
            return (h_modified,) + output[1:]
        return h_modified


def compute_token_entropy(scores: torch.Tensor) -> torch.Tensor:
    """Compute entropy of next-token distribution. scores: (B, V)."""
    probs = nn.functional.softmax(scores, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    return -(probs * logp).sum(dim=-1)


def _slice_attention_for_entropy(
    attn: torch.Tensor,
    vision_start: Optional[int],
    vision_end: Optional[int],
    vision_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    # attn: (B, H, T, S) -> select last query token
    attn = attn[:, :, -1, :]
    if vision_indices is not None and vision_indices.numel() > 0:
        attn = torch.index_select(attn, dim=-1, index=vision_indices)
    elif vision_start is not None and vision_end is not None and vision_end > vision_start:
        attn = attn[:, :, vision_start:vision_end]
    return attn


def compute_attn_entropy(
    outputs,
    vision_start: Optional[int] = None,
    vision_end: Optional[int] = None,
    vision_indices: Optional[torch.Tensor] = None,
    prefer_cross_attn: bool = True,
) -> Optional[torch.Tensor]:
    """Compute attention entropy from model outputs. Returns (B,) or None."""
    attn_stack = None
    if prefer_cross_attn and hasattr(outputs, "cross_attentions") and outputs.cross_attentions is not None:
        attn_stack = outputs.cross_attentions
    elif hasattr(outputs, "attentions") and outputs.attentions is not None:
        attn_stack = outputs.attentions

    if attn_stack is None:
        return None

    # last layer, average over heads
    last = attn_stack[-1]  # (B, H, T, S)
    last = _slice_attention_for_entropy(last, vision_start, vision_end, vision_indices)
    probs = last.mean(dim=1)  # (B, S)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    logp = torch.log(probs.clamp_min(1e-12))
    return -(probs * logp).sum(dim=-1)


def compute_lastlayer_cross_signals(
    outputs,
    vision_start: Optional[int] = None,
    vision_end: Optional[int] = None,
    vision_indices: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    """Compute Ha_last_hat and S_last from last-layer cross-attention only."""
    meta = {"source": "cross_attentions", "kv_len": None, "n_vision_tokens": None, "vision_slice": None}
    if not hasattr(outputs, "cross_attentions") or outputs.cross_attentions is None:
        return None, None, meta
    attn_stack = outputs.cross_attentions
    if len(attn_stack) == 0:
        return None, None, meta
    last = attn_stack[-1]
    if last is None or last.dim() < 4:
        return None, None, meta
    # (B, H, T, S) -> current query token
    last_q = last[:, :, -1, :]
    kv_len = int(last_q.shape[-1])
    meta["kv_len"] = kv_len
    if vision_indices is not None and vision_indices.numel() > 0:
        vis_idx = vision_indices.to(last_q.device)
        vis_idx = vis_idx[(vis_idx >= 0) & (vis_idx < kv_len)]
        if vis_idx.numel() == 0:
            return None, None, meta
        p_all = last_q.mean(dim=1)
        p_vision = torch.index_select(p_all, dim=-1, index=vis_idx)
        meta["n_vision_tokens"] = int(vis_idx.numel())
        meta["vision_slice"] = f"indices[{int(vis_idx.min().item())}:{int(vis_idx.max().item())}]"
    elif vision_start is not None and vision_end is not None and vision_end > vision_start:
        s = max(0, int(vision_start))
        e = min(kv_len, int(vision_end))
        if e <= s:
            return None, None, meta
        p_all = last_q.mean(dim=1)
        p_vision = p_all[:, s:e]
        meta["n_vision_tokens"] = int(e - s)
        meta["vision_slice"] = f"{s}:{e}"
    else:
        return None, None, meta
    s_last = p_vision.sum(dim=-1)
    p_norm = p_vision / p_vision.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    n_vis = int(meta["n_vision_tokens"] or 0)
    if n_vis > 1:
        denom = float(torch.log(torch.tensor(float(n_vis), device=p_norm.device)).item())
        ha_last = (-(p_norm * torch.log(p_norm.clamp_min(1e-12))).sum(dim=-1) / max(denom, 1e-12)).clamp(0.0, 1.0)
    else:
        ha_last = torch.zeros_like(s_last)
    return ha_last, s_last, meta


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12, normalize: bool = True) -> torch.Tensor:
    """Entropy over last dim for probability tensor p (..., N)."""
    p = p.float()
    h = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
    if not normalize:
        return h
    n = p.shape[-1]
    if n <= 1:
        return torch.zeros_like(h)
    denom = torch.log(torch.tensor(float(n), device=p.device, dtype=p.dtype)).clamp_min(eps)
    return (h / denom).clamp(0.0, 1.0)


def _safe_tensor_to_py(x: Optional[torch.Tensor]):
    if x is None:
        return None
    if x.numel() == 1:
        return float(x.item())
    return x.detach().cpu().tolist()


def compute_text_vision_attn_stats_from_last_layer(
    outputs,
    vision_start: Optional[int] = None,
    vision_end: Optional[int] = None,
    vision_indices: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    normalize_entropy: bool = True,
):
    """Compute text/vision attention entropy and masses from last-layer self-attn."""
    if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
        return None
    last = outputs.attentions[-1]
    if last is None or last.dim() < 4:
        return None
    # (B, H, Q, K) -> head-avg for current query token.
    a = last[:, :, -1, :].float().mean(dim=1)  # (B, K)
    bsz, kv_len = int(a.shape[0]), int(a.shape[-1])

    if bsz <= 0 or kv_len <= 1:
        return {
            "H_attn_text": None,
            "H_attn_vision": None,
            "m_text": None,
            "m_vision": None,
            "ratio_TV": None,
            "delta_TV": None,
            "attn_source": "attentions",
            "kv_len": kv_len,
            "n_vision_tokens": 0,
            "attn_max": None,
        }

    vis_mask = torch.zeros(kv_len, dtype=torch.bool, device=a.device)
    if vision_indices is not None and vision_indices.numel() > 0:
        vis_idx = vision_indices.to(a.device)
        vis_idx = vis_idx[(vis_idx >= 0) & (vis_idx < kv_len)]
        if vis_idx.numel() == 0:
            return None
        vis_mask[vis_idx] = True
    elif vision_start is not None and vision_end is not None and vision_end > vision_start:
        s = max(0, int(vision_start))
        e = min(kv_len, int(vision_end))
        if e <= s:
            return None
        vis_mask[s:e] = True
    else:
        return None

    text_mask = ~vis_mask
    n_vision = int(vis_mask.sum().item())
    n_text = int(text_mask.sum().item())
    if n_vision <= 0 or n_text <= 0:
        return {
            "H_attn_text": None,
            "H_attn_vision": None,
            "m_text": None,
            "m_vision": None,
            "ratio_TV": None,
            "delta_TV": None,
            "attn_source": "attentions",
            "kv_len": kv_len,
            "n_vision_tokens": n_vision,
            "attn_max": None,
        }

    p_vis = a[:, vis_mask]
    p_text = a[:, text_mask]
    m_vis = p_vis.sum(dim=-1)
    m_text = p_text.sum(dim=-1)

    p_vis_norm = p_vis / m_vis.unsqueeze(-1).clamp_min(eps)
    p_text_norm = p_text / m_text.unsqueeze(-1).clamp_min(eps)

    h_vis = _entropy_from_probs(p_vis_norm, eps=eps, normalize=normalize_entropy)
    h_text = _entropy_from_probs(p_text_norm, eps=eps, normalize=normalize_entropy)
    ratio_tv = h_text / h_vis.clamp_min(eps)
    delta_tv = h_text - h_vis

    return {
        "H_attn_text": _safe_tensor_to_py(h_text),
        "H_attn_vision": _safe_tensor_to_py(h_vis),
        "m_text": _safe_tensor_to_py(m_text),
        "m_vision": _safe_tensor_to_py(m_vis),
        "ratio_TV": _safe_tensor_to_py(ratio_tv),
        "delta_TV": _safe_tensor_to_py(delta_tv),
        "attn_source": "attentions",
        "kv_len": kv_len,
        "n_vision_tokens": n_vision,
        "attn_max": _safe_tensor_to_py(a.max(dim=-1).values),
    }


def compute_object_attn_stats_from_last_layer(
    outputs,
    keyword_positions: Optional[list[int]] = None,
):
    """Compute attention mass on object keyword token positions from last-layer self-attn."""
    if not keyword_positions:
        return None
    if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
        return None
    last = outputs.attentions[-1]
    if last is None or last.dim() < 4:
        return None
    a = last[:, :, -1, :].float().mean(dim=1)  # (B, K)
    kv_len = int(a.shape[-1])
    pos = torch.tensor(keyword_positions, device=a.device, dtype=torch.long)
    pos = pos[(pos >= 0) & (pos < kv_len)]
    if pos.numel() == 0:
        return None
    obj = torch.index_select(a, dim=-1, index=pos)
    obj_mass = obj.sum(dim=-1)
    obj_peak = obj.max(dim=-1).values
    return {
        "object_attn_mass": _safe_tensor_to_py(obj_mass),
        "object_attn_peak": _safe_tensor_to_py(obj_peak),
        "object_attn_n_tokens": int(pos.numel()),
    }


def schedule_temperature(
    ht: torch.Tensor,
    ha: Optional[torch.Tensor],
    mode: str,
    t_fixed: float,
    t_min: float,
    t_max: float,
    a: float,
    b: float,
    c: float,
    ht_low: float,
    ht_high: float,
    ha_low: float,
    ha_high: float,
    t_text_low: float,
    t_text_high: float,
) -> torch.Tensor:
    if mode == "fixed":
        return torch.full_like(ht, float(t_fixed))
    if mode == "entropy":
        if ha is None:
            temp = a * ht + c
        else:
            temp = a * ht + b * ha + c
        return temp.clamp(min=t_min, max=t_max)
    # 2x2 rule-based mode
    if ha is None:
        return torch.full_like(ht, float(t_fixed))
    ht_low_mask = ht <= ht_low
    ht_high_mask = ht >= ht_high
    ha_low_mask = ha <= ha_low
    ha_high_mask = ha >= ha_high

    # default: keep fixed
    temp = torch.full_like(ht, float(t_fixed))

    # Cross-attn low & Logit low -> keep
    # Cross-attn low & Logit high -> t_text_low (sharpen)
    temp = torch.where(ha_low_mask & ht_high_mask, torch.full_like(ht, float(t_text_low)), temp)
    # Cross-attn high & Logit low -> t_text_high (flatten)
    temp = torch.where(ha_high_mask & ht_low_mask, torch.full_like(ht, float(t_text_high)), temp)
    # Cross-attn high & Logit high -> t_text_low (sharpen)
    temp = torch.where(ha_high_mask & ht_high_mask, torch.full_like(ht, float(t_text_low)), temp)
    return temp


def _strip_temperature_warper(logits_processor: LogitsProcessorList) -> LogitsProcessorList:
    return LogitsProcessorList([p for p in logits_processor if not isinstance(p, TemperatureLogitsWarper)])


def _get_param(model_kwargs, generation_config, name, default):
    if name in model_kwargs:
        return model_kwargs.pop(name, default)
    return getattr(generation_config, name, default)


def _parse_bias_steps(spec: str):
    s = str(spec or "all").strip().lower()
    if s == "all":
        return "all", None
    if s == "prefill":
        return "prefill", None
    try:
        n = int(s)
        return "n", max(0, n)
    except Exception:
        return "all", None


def _build_yesno_allowed_ids(tokenizer) -> list[int]:
    allowed = set()
    candidates = {"yes", "no", " yes", " no", "Yes", "No", " YES", " NO"}
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size > 0:
        for tid in range(vocab_size):
            t = tokenizer.decode([tid], skip_special_tokens=False).strip()
            if t.lower() in {"yes", "no"}:
                allowed.add(tid)
    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            allowed.add(int(ids[0]))
    return sorted(allowed)


_OCR_PHRASES = [
    "read",
    "what does it say",
    "what is written",
    "what is the text",
    "text",
    "word",
    "letters",
    "characters",
    "sentence",
    "sign",
    "label",
    "menu",
    "license plate",
    "plate",
    "barcode",
    "serial",
    "id number",
    "phone number",
    "address",
    "translate",
    "translation",
]

_COUNT_PHRASES = [
    "how many",
    "count",
    "number of",
    "total number",
    "how many are there",
]

_WORD_BOUNDARY_TOKENS = {
    "read",
    "text",
    "count",
    "word",
    "letters",
    "characters",
    "sentence",
    "sign",
    "label",
    "menu",
    "plate",
}


def classify_task(question: str) -> str:
    """Classify question into one of: ocr, count, general."""
    q = (question or "").strip().lower()
    if not q:
        return "general"
    for kw in _OCR_PHRASES:
        if kw in _WORD_BOUNDARY_TOKENS:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                return "ocr"
        elif kw in q:
            return "ocr"
    for kw in _COUNT_PHRASES:
        if kw in _WORD_BOUNDARY_TOKENS:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                return "count"
        elif kw in q:
            return "count"
    return "general"


def _schedule_attn_temperature(
    ht: Optional[torch.Tensor],
    ha: Optional[torch.Tensor],
    ht_low: float,
    ht_high: float,
    ha_low: float,
    ha_high: float,
    t_attn_base: float,
    t_attn_low: float,
    t_attn_high: float,
) -> float:
    if ha is None or ht is None:
        return float(t_attn_base)
    # Use batch mean for a single scalar temperature
    ha_mean = float(ha.mean().item())
    ht_mean = float(ht.mean().item())

    ha_is_low = ha_mean <= ha_low
    ha_is_high = ha_mean >= ha_high
    ht_is_low = ht_mean <= ht_low
    ht_is_high = ht_mean >= ht_high

    # 2x2 rule from the table:
    # Cross-attn low + Logit low -> keep t_attn
    # Cross-attn low + Logit high -> keep t_attn
    # Cross-attn high + Logit low -> t_attn_low (strengthen image focus)
    # Cross-attn high + Logit high -> t_attn_low (strengthen image focus)
    if ha_is_high and (ht_is_low or ht_is_high):
        return float(t_attn_low)
    return float(t_attn_base)


def _compute_attention_mass(
    outputs,
    vision_start: Optional[int],
    vision_end: Optional[int],
    vision_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if vision_indices is None and (vision_start is None or vision_end is None):
        return None
    attn_stack = None
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        attn_stack = outputs.attentions
    if attn_stack is None:
        return None
    last = attn_stack[-1]  # (B, H, T, S)
    last = last[:, :, -1, :]
    if vision_indices is not None and vision_indices.numel() > 0:
        last = torch.index_select(last, dim=-1, index=vision_indices)
    elif vision_start is not None and vision_end is not None and vision_end > vision_start:
        last = last[:, :, vision_start:vision_end]
    probs = last.mean(dim=1)  # (B, S)
    return probs.sum(dim=-1)


def entropy_temp_generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> torch.LongTensor | GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput:
    """Non-beam sampling loop with step-wise temperature control. Minimal diff from _sample."""
    # read custom args (prefer model_kwargs, fallback to generation_config)
    temp_mode = _get_param(model_kwargs, generation_config, "temp_mode", "fixed")
    temp_fixed = float(_get_param(model_kwargs, generation_config, "temp_fixed", 1.0))
    temp_min = float(_get_param(model_kwargs, generation_config, "temp_min", 0.2))
    temp_max = float(_get_param(model_kwargs, generation_config, "temp_max", 1.5))
    temp_a = float(_get_param(model_kwargs, generation_config, "temp_a", 1.0))
    temp_b = float(_get_param(model_kwargs, generation_config, "temp_b", 1.0))
    temp_c = float(_get_param(model_kwargs, generation_config, "temp_c", 0.0))
    temp_log = bool(_get_param(model_kwargs, generation_config, "temp_log", False))
    use_attn_entropy = bool(_get_param(model_kwargs, generation_config, "use_attn_entropy", False))
    vision_start = _get_param(model_kwargs, generation_config, "vision_start_idx", None)
    vision_end = _get_param(model_kwargs, generation_config, "vision_end_idx", None)
    vision_indices = _get_param(model_kwargs, generation_config, "vision_indices", None)
    if vision_indices is not None and not torch.is_tensor(vision_indices):
        try:
            vision_indices = torch.tensor(vision_indices, dtype=torch.long)
        except Exception:
            vision_indices = None
    ht_low = float(_get_param(model_kwargs, generation_config, "ht_low", 0.5))
    ht_high = float(_get_param(model_kwargs, generation_config, "ht_high", 1.5))
    ha_low = float(_get_param(model_kwargs, generation_config, "ha_low", 2.0))
    ha_high = float(_get_param(model_kwargs, generation_config, "ha_high", 4.0))
    t_text_low = float(_get_param(model_kwargs, generation_config, "t_text_low", 0.7))
    t_text_high = float(_get_param(model_kwargs, generation_config, "t_text_high", 1.2))
    soft_ht0 = float(_get_param(model_kwargs, generation_config, "soft2x2_ht0", 1.0))
    soft_ha0 = float(_get_param(model_kwargs, generation_config, "soft2x2_ha0", 3.0))
    soft_k_t = float(_get_param(model_kwargs, generation_config, "soft2x2_k_t", 6.0))
    soft_k_a = float(_get_param(model_kwargs, generation_config, "soft2x2_k_a", 6.0))
    soft_ema = float(_get_param(model_kwargs, generation_config, "soft2x2_ema_alpha", 0.0))
    enable_t_attn = bool(_get_param(model_kwargs, generation_config, "enable_t_attn", False))
    t_attn_base = float(_get_param(model_kwargs, generation_config, "t_attn_base", 1.0))
    t_attn_low = float(_get_param(model_kwargs, generation_config, "t_attn_low", 0.7))
    t_attn_high = float(_get_param(model_kwargs, generation_config, "t_attn_high", 1.2))
    attn_modules = globals().get("_ATTN_MODULES") or _get_param(model_kwargs, generation_config, "attn_modules", None)
    attn_bias_modules = globals().get("_ATTN_BIAS_MODULES") or _get_param(model_kwargs, generation_config, "attn_bias_modules", attn_modules)
    attn_temp_sanity_check = bool(_get_param(model_kwargs, generation_config, "attn_temp_sanity_check", False))
    attn_temp_sanity_probe = float(_get_param(model_kwargs, generation_config, "attn_temp_sanity_probe", 2.0))
    use_gate_for_intervention = bool(_get_param(model_kwargs, generation_config, "use_gate_for_intervention", False))
    force_no_intervention = bool(_get_param(model_kwargs, generation_config, "force_no_intervention", False))
    attn_gate_tau = float(_get_param(model_kwargs, generation_config, "attn_gate_tau", 0.2))
    gate_smooth_mode = _get_param(model_kwargs, generation_config, "attn_gate_smooth_mode", "ema")
    gate_ema_alpha = float(_get_param(model_kwargs, generation_config, "attn_gate_ema_alpha", 0.9))
    gate_window_size = int(_get_param(model_kwargs, generation_config, "attn_gate_window_size", 8))
    gate_adaptive = bool(_get_param(model_kwargs, generation_config, "attn_gate_adaptive", False))
    gate_target = float(_get_param(model_kwargs, generation_config, "attn_gate_target_rate", 0.7))
    gate_adapt_alpha = float(_get_param(model_kwargs, generation_config, "attn_gate_adapt_alpha", 0.9))
    attn_step_reduce_only = bool(_get_param(model_kwargs, generation_config, "attn_step_reduce_only", True))
    attn_keyword_bias = float(_get_param(model_kwargs, generation_config, "attn_keyword_bias", 0.0) or 0.0)
    attn_keyword_bias_max = float(_get_param(model_kwargs, generation_config, "attn_keyword_bias_max", 4.0) or 4.0)
    attn_keyword_positions = _get_param(model_kwargs, generation_config, "attn_keyword_positions", None)
    if attn_keyword_positions is None:
        attn_keyword_positions = []
    bias_steps_mode, bias_steps_n = _parse_bias_steps(_get_param(model_kwargs, generation_config, "bias_steps", "all"))
    bias_anneal = str(_get_param(model_kwargs, generation_config, "bias_anneal", "hard"))
    bias_apply_when = str(_get_param(model_kwargs, generation_config, "bias_apply_when", "always"))
    safe_decode_on_bias = bool(_get_param(model_kwargs, generation_config, "safe_decode_on_bias", False))
    collapse_risk_threshold = float(_get_param(model_kwargs, generation_config, "collapse_risk_threshold", 0.6) or 0.6)
    force_yesno = str(_get_param(model_kwargs, generation_config, "force_yesno", "off"))
    force_yesno_ids = _get_param(model_kwargs, generation_config, "force_yesno_ids", None)
    if force_yesno_ids is None:
        force_yesno_ids = []
    subspace_shrink_enable = bool(_get_param(model_kwargs, generation_config, "subspace_shrink_enable", False))
    subspace_shrink_apply = bool(_get_param(model_kwargs, generation_config, "subspace_shrink_apply", False))
    subspace_shrink_alpha = float(_get_param(model_kwargs, generation_config, "subspace_shrink_alpha", 0.3))
    subspace_risk_score = _get_param(model_kwargs, generation_config, "subspace_risk_score", None)
    subspace_risk_threshold = _get_param(model_kwargs, generation_config, "subspace_risk_threshold", None)
    # Read tensors from module-level global (avoids deepcopy OOM on gen_cfg)
    _st = globals().get("_SUBSPACE_TENSORS", {})
    subspace_mu = _st.get("mu") if _st else _get_param(model_kwargs, generation_config, "subspace_mu", None)
    subspace_uk = _st.get("uk") if _st else _get_param(model_kwargs, generation_config, "subspace_uk", None)
    subspace_basis_list = _st.get("basis_list") if _st else _get_param(model_kwargs, generation_config, "subspace_basis_list", None)
    subspace_online_risk = bool(_get_param(model_kwargs, generation_config, "subspace_online_risk", False))
    subspace_intervention_layer = int(_get_param(model_kwargs, generation_config, "subspace_intervention_layer", 20))
    subspace_tau = float(_get_param(model_kwargs, generation_config, "subspace_tau", 0.5))
    subspace_tau_max = float(_get_param(model_kwargs, generation_config, "subspace_tau_max", 0.7))
    subspace_lambda_max = float(_get_param(model_kwargs, generation_config, "subspace_lambda_max", 0.5))

    du_lam = float(_get_param(model_kwargs, generation_config, "du_lam", 0.9))
    du_k = int(_get_param(model_kwargs, generation_config, "du_k", 2))
    du_ht_low = float(_get_param(model_kwargs, generation_config, "du_ht_low", 0.55))
    du_ht_high = float(_get_param(model_kwargs, generation_config, "du_ht_high", 0.65))
    du_ha_low = float(_get_param(model_kwargs, generation_config, "du_ha_low", 0.55))
    du_ha_high = float(_get_param(model_kwargs, generation_config, "du_ha_high", 0.65))
    du_t_text_base = float(_get_param(model_kwargs, generation_config, "du_t_text_base", 0.7))
    du_t_text_min = float(_get_param(model_kwargs, generation_config, "du_t_text_min", 0.1))
    du_t_text_max = float(_get_param(model_kwargs, generation_config, "du_t_text_max", 1.3))
    du_t_attn_base = float(_get_param(model_kwargs, generation_config, "du_t_attn_base", 1.0))
    du_t_attn_min = float(_get_param(model_kwargs, generation_config, "du_t_attn_min", 0.3))
    du_t_attn_max = float(_get_param(model_kwargs, generation_config, "du_t_attn_max", 2.0))
    du_dt_explore = float(_get_param(model_kwargs, generation_config, "du_dt_explore", 0.2))
    du_dt_conserve = float(_get_param(model_kwargs, generation_config, "du_dt_conserve", 0.1))
    du_alpha_conserve = float(_get_param(model_kwargs, generation_config, "du_alpha_conserve", 0.7))
    du_dt_floor = float(_get_param(model_kwargs, generation_config, "du_dt_floor", 0.05))
    du_gateoff_lam = float(_get_param(model_kwargs, generation_config, "du_gateoff_lam", 0.9))
    du_da_strong = float(_get_param(model_kwargs, generation_config, "du_da_strong", 0.4))
    du_ht_star = float(_get_param(model_kwargs, generation_config, "du_ht_star", 0.6))
    du_ha_star = float(_get_param(model_kwargs, generation_config, "du_ha_star", 0.6))
    du_k_t = float(_get_param(model_kwargs, generation_config, "du_k_t", 0.8))
    du_k_g = float(_get_param(model_kwargs, generation_config, "du_k_g", 0.6))
    du_k_a = float(_get_param(model_kwargs, generation_config, "du_k_a", 0.8))
    du_ha_adversarial = bool(_get_param(model_kwargs, generation_config, "du_ha_adversarial", False))
    du_task_override = str(_get_param(model_kwargs, generation_config, "du_task_override", "precise_c4"))
    du_precise_c4_allow_text_up = bool(
        _get_param(model_kwargs, generation_config, "du_precise_c4_allow_text_up", False)
    )
    du_precise_c4_text_up_delta = float(
        _get_param(model_kwargs, generation_config, "du_precise_c4_text_up_delta", 0.0)
    )
    task_question = str(_get_param(model_kwargs, generation_config, "task_question", ""))
    task_type = classify_task(task_question)

    soft_params = None
    soft_state = None
    if temp_mode == "soft2x2":
        soft_params = Soft2x2Params(
            ht0=soft_ht0,
            ha0=soft_ha0,
            k_t=soft_k_t,
            k_a=soft_k_a,
            t_fixed=temp_fixed,
            t_text_low=t_text_low,
            t_text_high=t_text_high,
            t_attn_base=t_attn_base,
            t_attn_low=t_attn_low,
            ema_alpha=soft_ema,
        )
        soft_state = Soft2x2State()
    gate_smoother = AttnGateSmoother(
        AttnGateSmootherConfig(
            mode=gate_smooth_mode,
            ema_alpha=gate_ema_alpha,
            window_size=gate_window_size,
        )
    )
    gate_rate_ema = 0.0
    gate_tau = attn_gate_tau
    gate_on_count = 0
    gate_total_steps = 0
    gate_on_h_attn_sum = 0.0
    gate_on_h_t_sum = 0.0
    gate_cases: list[str | None] = []
    step_task_types: list[str | None] = []
    step_override_applied: list[bool | None] = []
    step_telemetry: list[dict] = []
    gate_case_counts: dict[str, int] = {}
    n_steps_ha_high = 0
    n_steps_override = 0
    first_step_override = -1
    forward_time_s = 0.0
    entropy_time_s = 0.0
    sampling_time_s = 0.0
    cross_signal_meta_logged = False
    attn_temp_sanity_done = False
    last_gate_on = False
    decode_step_count = 0
    first_tokens_ids: list[int] = []
    first_tokens_str: list[str] = []
    eos_at_step1 = False
    keyword_bias_warned = False
    subspace_applied_any = False
    subspace_res_before_vals: list[float] = []
    subspace_res_after_vals: list[float] = []
    hook_risk_vals: list[float] = []
    hook_alpha_vals: list[float] = []

    # remove built-in temperature warper to avoid double scaling
    logits_processor = _strip_temperature_warper(logits_processor)

    # init values (mirror _sample)
    pad_token_id = generation_config._pad_token_tensor
    eos_token_tensor = getattr(generation_config, "_eos_token_tensor", None)
    output_attentions = generation_config.output_attentions or use_attn_entropy
    output_hidden_states = generation_config.output_hidden_states
    if subspace_basis_list is None:
        subspace_basis_list = []
    if not isinstance(subspace_basis_list, (list, tuple)):
        subspace_basis_list = []
    # Backward-compat: single basis fallback.
    if (not subspace_basis_list) and (subspace_mu is not None) and (subspace_uk is not None):
        subspace_basis_list = [{"layer": None, "mu": subspace_mu, "uk": subspace_uk}]

    # ── Online hook-based intervention ──
    _intervention_hook = None
    _hook_handle = None
    use_subspace_hook = bool(
        subspace_shrink_enable and subspace_shrink_apply
        and subspace_online_risk and len(subspace_basis_list) > 0
    )
    if use_subspace_hook:
        # Find basis (mu + Uk) for the target intervention layer
        target_layer = subspace_intervention_layer
        hook_mu = None
        hook_uk = None
        for basis in subspace_basis_list:
            if isinstance(basis, dict) and basis.get("layer") == target_layer:
                hook_mu = basis.get("mu")
                hook_uk = basis.get("uk")
                break
        if hook_uk is None and len(subspace_basis_list) == 1:
            hook_mu = subspace_basis_list[0].get("mu")
            hook_uk = subspace_basis_list[0].get("uk")
        if hook_uk is None or hook_mu is None:
            avail = [b.get("layer") for b in subspace_basis_list if isinstance(b, dict)]
            raise ValueError(
                f"Subspace online risk enabled but no basis found for "
                f"intervention layer {target_layer}. "
                f"Available layers in basis: {avail}. "
                f"Check --subspace-basis-layer and --subspace-intervention-layer."
            )
        if hook_uk is not None and hook_mu is not None:
            _intervention_hook = SubspaceInterventionHook(
                mu=hook_mu,
                uk=hook_uk,
                tau=subspace_tau,
                tau_max=subspace_tau_max,
                lambda_max=subspace_lambda_max,
            )
            decoder_layers = _find_decoder_layers(model)
            if decoder_layers is not None:
                # hidden_states[L] = output of decoder_layers[L-1]
                hook_layer_idx = target_layer - 1
                if 0 <= hook_layer_idx < len(decoder_layers):
                    _hook_handle = decoder_layers[hook_layer_idx].register_forward_hook(_intervention_hook)
                    _intervention_hook.enabled = True

    # Legacy logit-replacement path (when not using hook)
    need_subspace_hidden = bool(
        subspace_shrink_enable and subspace_shrink_apply
        and len(subspace_basis_list) > 0 and not use_subspace_hook
    )
    output_hidden_states = bool(output_hidden_states or need_subspace_hidden)
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    keep_attn_history = bool(return_dict_in_generate and output_attentions and not attn_step_reduce_only)
    decoder_attentions = () if keep_attn_history else None
    cross_attentions = () if (keep_attn_history and model.config.is_encoder_decoder) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape[:2]
    initial_input_len = int(cur_len)
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = model.__call__
    compile_forward = model._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                generation_config.compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = model._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    current_t_attn = float(t_attn_base)
    t_text_prev = None
    t_attn_prev = None
    ht_ema = None
    ha_ema = None
    step_idx = 0

    # Reset hook step counter for this sample
    if _intervention_hook is not None:
        _intervention_hook.reset_sample()

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # Bias scheduling/conditioning
        bias_mag = float(attn_keyword_bias)
        if abs(bias_mag) > abs(attn_keyword_bias_max):
            bias_mag = float(attn_keyword_bias_max if bias_mag > 0 else -attn_keyword_bias_max)
        if abs(attn_keyword_bias) > abs(attn_keyword_bias_max) and not keyword_bias_warned:
            print(
                {
                    "keyword_bias_warning": True,
                    "requested_bias": float(attn_keyword_bias),
                    "clamped_bias": float(bias_mag),
                    "bias_max": float(attn_keyword_bias_max),
                }
            )
            keyword_bias_warned = True
        if bias_steps_mode == "prefill":
            step_window_on = bool(is_prefill)
        elif bias_steps_mode == "n":
            if bias_steps_n is None:
                step_window_on = True
            else:
                step_window_on = bool(decode_step_count < bias_steps_n)
        else:
            step_window_on = True
        if bias_steps_mode == "n" and bias_steps_n and bias_anneal == "linear" and decode_step_count < bias_steps_n:
            frac = 1.0 - (float(decode_step_count) / float(max(1, bias_steps_n)))
            bias_mag *= max(0.0, frac)
        gate_condition_on = True
        if bias_apply_when == "gate_on":
            gate_condition_on = bool(last_gate_on)
        bias_active = (
            abs(bias_mag) > 1e-12
            and bool(attn_keyword_positions)
            and step_window_on
            and gate_condition_on
        )
        if enable_t_attn and attn_modules is not None:
            set_attn_temperature(attn_modules, current_t_attn)
        if attn_bias_modules is not None:
            if bias_active:
                set_attn_key_bias(attn_bias_modules, attn_keyword_positions, bias_mag)
            else:
                set_attn_key_bias(attn_bias_modules, None, 0.0)
        t0_forward = time.perf_counter()
        model_inputs["output_hidden_states"] = output_hidden_states
        model_inputs["output_attentions"] = output_attentions
        if is_prefill:
            outputs = model(
                **model_inputs,
                return_dict=True,
            )
            is_prefill = False
        else:
            outputs = model_forward(
                **model_inputs,
                return_dict=True,
            )
        forward_time_s += time.perf_counter() - t0_forward

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        subspace_res_before = None
        subspace_res_after = None
        subspace_applied_step = False
        hook_risk_score = None
        hook_alpha = None
        hook_prefill_risk = None
        # ── Online hook path: hook already modified hidden state at target layer ──
        if _intervention_hook is not None:
            subspace_applied_step = _intervention_hook.last_applied
            hook_risk_score = _intervention_hook.last_risk
            hook_alpha = _intervention_hook.last_alpha
            subspace_res_before = _intervention_hook.last_orth_before
            subspace_res_after = _intervention_hook.last_orth_after
            hook_prefill_risk = _intervention_hook.prefill_risk
            _intervention_hook.reset_step()
        # ── Legacy logit-replacement path ──
        elif need_subspace_hidden:
            try:
                hs = getattr(outputs, "hidden_states", None)
                if hs is not None and len(hs) > 0:
                    h_last = hs[-1][:, -1, :].to(dtype=torch.float32, device=input_ids.device)  # [B,d]
                    alpha = float(max(0.0, min(1.0, subspace_shrink_alpha)))
                    h_new_list = []
                    before_list = []
                    after_list = []
                    for basis in subspace_basis_list:
                        uk_t = basis.get("uk") if isinstance(basis, dict) else None
                        if uk_t is None:
                            continue
                        if not torch.is_tensor(uk_t):
                            uk_t = torch.tensor(uk_t, dtype=torch.float32, device=input_ids.device)
                        else:
                            uk_t = uk_t.to(dtype=torch.float32, device=input_ids.device)
                        if not (uk_t.ndim == 2 and uk_t.shape[0] == h_last.shape[-1]):
                            continue
                        # Direct attraction: h' = (1-α)h + α·Uk Uk^T h
                        h_proj = (h_last @ uk_t) @ uk_t.T  # [B,d]
                        h_orth = h_last - h_proj  # orthogonal component
                        h_new_j = (1.0 - alpha) * h_last + alpha * h_proj  # = h - α·h_orth
                        h_new_list.append(h_new_j)
                        before_list.append(float(torch.linalg.norm(h_orth, dim=1).mean().item()))
                        after_list.append(float(torch.linalg.norm((1.0 - alpha) * h_orth, dim=1).mean().item()))

                    if h_new_list:
                        h_new = torch.stack(h_new_list, dim=0).mean(dim=0)
                        subspace_res_before = float(sum(before_list) / len(before_list))
                        subspace_res_after = float(sum(after_list) / len(after_list))
                        lm_head = model.get_output_embeddings()
                        if lm_head is not None:
                            w_dtype = lm_head.weight.dtype if hasattr(lm_head, "weight") else h_new.dtype
                            logits_new = lm_head(h_new.to(dtype=w_dtype))
                            # Some decoder-only models define final_logits_bias.
                            if hasattr(model, "final_logits_bias") and model.final_logits_bias is not None:
                                logits_new = logits_new + model.final_logits_bias.to(logits_new.device, logits_new.dtype)
                            next_token_logits = logits_new.to(dtype=torch.float32, device=input_ids.device)
                            subspace_applied_step = True
            except Exception:
                subspace_applied_step = False
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # apply dynamic temperature after processors/warpers
        t0_entropy = time.perf_counter()
        if vision_indices is not None:
            vision_indices = vision_indices.to(input_ids.device)
        ha = compute_attn_entropy(outputs, vision_start, vision_end, vision_indices) if use_attn_entropy else None
        tv_stats = compute_text_vision_attn_stats_from_last_layer(
            outputs,
            vision_start=vision_start,
            vision_end=vision_end,
            vision_indices=vision_indices,
            eps=1e-12,
            normalize_entropy=True,
        )
        obj_stats = compute_object_attn_stats_from_last_layer(
            outputs,
            keyword_positions=attn_keyword_positions,
        )
        ha_last_hat, s_last, cross_meta = compute_lastlayer_cross_signals(
            outputs, vision_start=vision_start, vision_end=vision_end, vision_indices=vision_indices
        )
        if not cross_signal_meta_logged and cross_meta.get("kv_len") is not None:
            print(
                {
                    "cross_signal_meta": True,
                    "kv_len": cross_meta.get("kv_len"),
                    "n_vision_tokens": cross_meta.get("n_vision_tokens"),
                    "vision_slice": cross_meta.get("vision_slice"),
                    "source": cross_meta.get("source"),
                }
            )
            cross_signal_meta_logged = True
        if (
            attn_temp_sanity_check
            and (not attn_temp_sanity_done)
            and enable_t_attn
            and attn_modules is not None
        ):
            try:
                ha_base = compute_attn_entropy(outputs, vision_start, vision_end, vision_indices)
                set_attn_temperature(attn_modules, float(attn_temp_sanity_probe))
                outputs_probe = model(**model_inputs, return_dict=True)
                ha_probe = compute_attn_entropy(outputs_probe, vision_start, vision_end, vision_indices)
                set_attn_temperature(attn_modules, current_t_attn)
                base_v = None if ha_base is None else float(ha_base.mean().item())
                probe_v = None if ha_probe is None else float(ha_probe.mean().item())
                delta_v = None
                if base_v is not None and probe_v is not None:
                    delta_v = probe_v - base_v
                print(
                    {
                        "attn_temp_sanity": True,
                        "step": step_idx,
                        "t_attn_base": float(current_t_attn),
                        "t_attn_probe": float(attn_temp_sanity_probe),
                        "H_attn_base": base_v,
                        "H_attn_probe": probe_v,
                        "delta": delta_v,
                    }
                )
                if outputs_probe is not None:
                    del outputs_probe
            except Exception as _exc:
                print({"attn_temp_sanity": False, "step": step_idx, "error": str(_exc)})
            finally:
                attn_temp_sanity_done = True
        ht = compute_token_entropy(next_token_scores)
        gate_on = False
        s_t = None
        s_bar = None
        if use_gate_for_intervention:
            s_t = _compute_attention_mass(outputs, vision_start, vision_end, vision_indices)
            if s_t is not None:
                s_bar = gate_smoother.update(s_t.detach()).detach()
                gate_on = bool((s_bar >= gate_tau).all().item())
                if gate_adaptive:
                    gate_rate_ema = gate_adapt_alpha * gate_rate_ema + (1.0 - gate_adapt_alpha) * float(gate_on)
                    gate_tau -= 0.05 * (gate_target - gate_rate_ema)
                    gate_tau = max(0.0, min(1.0, gate_tau))
            else:
                # Gate requested but no valid vision span/mass for this sample.
                gate_on = False
        last_gate_on = bool(gate_on)
        gate_total_steps += 1
        if gate_on:
            gate_on_count += 1
            if ha is not None:
                gate_on_h_attn_sum += float(ha.mean().item())
            gate_on_h_t_sum += float(ht.mean().item())
        if force_no_intervention:
            temp = torch.full_like(ht, float(temp_fixed))
            if enable_t_attn:
                current_t_attn = float(t_attn_base)
            gate_case = "FORCED_OFF"
            override_applied = False
            aux = {
                "Ht_hat": None,
                "Ha_hat": None,
                "Ht_ema": None,
                "Ha_ema": None,
                "gate_case": gate_case,
                "gate_on": gate_on,
                "t_text_final": temp.detach().cpu().tolist(),
                "t_attn_final": [float(current_t_attn)],
                "t_text_before": temp.detach().cpu().tolist(),
                "t_text_after": temp.detach().cpu().tolist(),
                "delta_up": None,
                "task_type": task_type,
                "override_applied": override_applied,
            }
        elif temp_mode == "soft2x2" and soft_params is not None:
            t_text, t_attn, aux = soft_2x2_update(ht, ha, soft_params, soft_state)
            if gate_on:
                temp = torch.full_like(ht, float(t_text))
                if enable_t_attn:
                    current_t_attn = float(t_attn)
            else:
                temp = torch.full_like(ht, float(temp_fixed))
            gate_case = None
            override_applied = None
        elif temp_mode == "dual_uncertainty":
            vocab_size = getattr(model.config, "vocab_size", next_token_scores.size(-1))
            log_vocab = float(torch.log(torch.tensor(float(vocab_size))).item())
            ht_hat = (ht / log_vocab).clamp(0.0, 1.0)
            if ha is not None:
                if vision_indices is not None and vision_indices.numel() > 0:
                    n_vis = int(vision_indices.numel())
                elif vision_start is not None and vision_end is not None and vision_end > vision_start:
                    n_vis = int(vision_end - vision_start)
                else:
                    n_vis = 0
                if n_vis > 1:
                    log_vis = float(torch.log(torch.tensor(float(n_vis))).item())
                    ha_hat = (ha / log_vis).clamp(0.0, 1.0)
                else:
                    ha_hat = torch.zeros_like(ht_hat)
            else:
                ha_hat = torch.zeros_like(ht_hat)

            if step_idx % max(1, du_k) == 0:
                if ht_ema is None:
                    ht_ema = ht_hat.detach()
                    ha_ema = ha_hat.detach()
                else:
                    ht_ema = du_lam * ht_ema + (1.0 - du_lam) * ht_hat.detach()
                    ha_ema = du_lam * ha_ema + (1.0 - du_lam) * ha_hat.detach()

            ht_use = ht_ema if ht_ema is not None else ht_hat
            ha_use = ha_ema if ha_ema is not None else ha_hat

            ht_low = ht_use <= du_ht_low
            ht_high = ht_use >= du_ht_high
            ha_low = ha_use <= du_ha_low
            ha_high = ha_use >= du_ha_high
            ht_low_any = bool(ht_low.any().item())
            ht_high_any = bool(ht_high.any().item())
            ha_low_any = bool(ha_low.any().item())
            ha_high_any = bool(ha_high.any().item())
            if ha_high_any:
                n_steps_ha_high += 1

            # continuous control (base + k_t*Ht - k_g*Ha)
            # adversarial: flip Ha influence (increase T_text when Ha is high)
            if du_ha_adversarial:
                t_text_cont = du_t_text_base + du_k_t * (ht_use - du_ht_star) + du_k_g * (ha_use - du_ha_star)
            else:
                t_text_cont = du_t_text_base + du_k_t * (ht_use - du_ht_star) - du_k_g * (ha_use - du_ha_star)
            # cap exploration when grounding is poor (skip cap in adversarial mode)
            if (not du_ha_adversarial) and ha_high_any:
                t_text_cont = torch.minimum(t_text_cont, torch.full_like(t_text_cont, du_t_text_base))

            # attention temperature: reduce as Ha increases
            # adversarial: increase t_attn when Ha is high
            if du_ha_adversarial:
                t_attn_cont = du_t_attn_base + du_k_a * (ha_use - du_ha_star)
            else:
                t_attn_cont = du_t_attn_base - du_k_a * (ha_use - du_ha_star)

            gate_case = "C1"
            if ha_low_any and ht_high_any:
                gate_case = "C2"

            delta_up = None
            t_text_before = t_text_cont
            t_text_after = t_text_cont

            if ha_high_any and ht_low_any:
                gate_case = "C3"
                # conservative override (or adversarial boost)
                if du_ha_adversarial:
                    t_text_cont = torch.maximum(
                        t_text_cont, torch.full_like(t_text_cont, du_t_text_base + du_dt_explore)
                    )
                    t_attn_cont = torch.maximum(
                        t_attn_cont, torch.full_like(t_attn_cont, du_t_attn_base + du_da_strong)
                    )
                else:
                    # Only roll back the "upward" part above base.
                    # If t_text_cont is already below base, keep it as-is.
                    delta_up = torch.clamp(t_text_cont - du_t_text_base, min=0.0)
                    t_text_cont = torch.where(
                        t_text_cont > du_t_text_base,
                        du_t_text_base + (1.0 - du_alpha_conserve) * delta_up,
                        t_text_cont,
                    )
                    t_attn_cont = torch.minimum(
                        t_attn_cont, torch.full_like(t_attn_cont, du_t_attn_base - du_da_strong)
                    )
            elif ha_high_any and ht_high_any:
                gate_case = "C4"
                if du_ha_adversarial:
                    # default adversarial behavior (kept)
                    t_attn_cont = torch.maximum(
                        t_attn_cont, torch.full_like(t_attn_cont, du_t_attn_base + du_da_strong)
                    )
                else:
                    # default behavior (kept): conservative text + attn downshift
                    t_text_cont = torch.minimum(
                        t_text_cont, torch.full_like(t_text_cont, du_t_text_base - du_dt_conserve)
                    )
                    t_attn_cont = torch.minimum(
                        t_attn_cont, torch.full_like(t_attn_cont, du_t_attn_base - du_da_strong)
                    )

            override_applied = False
            precise_task = task_type in {"ocr", "count"}
            if (
                du_task_override == "precise_c4"
                and precise_task
                and gate_on
                and ha_high_any
                and gate_case == "C4"
            ):
                override_applied = True
                # C4 precise override: do not extra text downshift; keep cap at base.
                t_text_cont = torch.minimum(t_text_cont, torch.full_like(t_text_cont, du_t_text_base))
                if du_precise_c4_allow_text_up:
                    t_text_cont = torch.minimum(
                        t_text_cont + du_precise_c4_text_up_delta,
                        torch.full_like(t_text_cont, du_t_text_max),
                    )
                # Keep strong visual focusing.
                if du_ha_adversarial:
                    t_attn_cont = torch.maximum(
                        t_attn_cont, torch.full_like(t_attn_cont, du_t_attn_base + du_da_strong)
                    )
                else:
                    t_attn_cont = torch.minimum(
                        t_attn_cont, torch.full_like(t_attn_cont, du_t_attn_base - du_da_strong)
                    )
                n_steps_override += 1
                if first_step_override < 0:
                    first_step_override = step_idx

            t_text_after = t_text_cont
            t_text_final = t_text_cont.clamp(min=du_t_text_min, max=du_t_text_max)
            t_attn_final = t_attn_cont.clamp(min=du_t_attn_min, max=du_t_attn_max)

            if gate_on:
                temp = t_text_final
                if enable_t_attn:
                    current_t_attn = float(t_attn_final.mean().item())
            else:
                if t_text_prev is None:
                    t_text_prev = torch.full_like(ht, float(du_t_text_base))
                if t_attn_prev is None:
                    t_attn_prev = torch.full_like(ht, float(du_t_attn_base))
                t_text_final = du_gateoff_lam * t_text_prev + (1.0 - du_gateoff_lam) * du_t_text_base
                t_attn_final = du_gateoff_lam * t_attn_prev + (1.0 - du_gateoff_lam) * du_t_attn_base
                t_text_final = t_text_final.clamp(min=du_t_text_min, max=du_t_text_max)
                t_attn_final = t_attn_final.clamp(min=du_t_attn_min, max=du_t_attn_max)
                temp = t_text_final
                if enable_t_attn:
                    current_t_attn = float(t_attn_final.mean().item())
            t_text_prev = t_text_final.detach()
            t_attn_prev = t_attn_final.detach()
            aux = {
                "Ht_hat": ht_hat.detach().cpu().tolist(),
                "Ha_hat": ha_hat.detach().cpu().tolist(),
                "Ht_ema": None if ht_ema is None else ht_ema.detach().cpu().tolist(),
                "Ha_ema": None if ha_ema is None else ha_ema.detach().cpu().tolist(),
                "gate_case": f"{gate_case}_ADV" if du_ha_adversarial else gate_case,
                "gate_on": gate_on,
                "t_text_final": t_text_final.detach().cpu().tolist() if torch.is_tensor(t_text_final) else t_text_final,
                "t_attn_final": t_attn_final.detach().cpu().tolist() if torch.is_tensor(t_attn_final) else t_attn_final,
                "t_text_before": t_text_before.detach().cpu().tolist() if torch.is_tensor(t_text_before) else t_text_before,
                "t_text_after": t_text_after.detach().cpu().tolist() if torch.is_tensor(t_text_after) else t_text_after,
                "delta_up": None if delta_up is None else delta_up.detach().cpu().tolist(),
                "task_type": task_type,
                "override_applied": override_applied,
            }
            gate_case = aux["gate_case"]
        else:
            temp = schedule_temperature(
                ht,
                ha,
                temp_mode,
                temp_fixed,
                temp_min,
                temp_max,
                temp_a,
                temp_b,
                temp_c,
                ht_low,
                ht_high,
                ha_low,
                ha_high,
                t_text_low,
                t_text_high,
            )
            if not gate_on:
                temp = torch.full_like(ht, float(temp_fixed))
            if enable_t_attn and gate_on:
                current_t_attn = _schedule_attn_temperature(
                    ht,
                    ha,
                    ht_low,
                    ht_high,
                    ha_low,
                    ha_high,
                    t_attn_base,
                    t_attn_low,
                    t_attn_high,
                )
            gate_case = None
            override_applied = None

        gate_cases.append(gate_case)
        step_task_types.append(task_type if temp_mode == "dual_uncertainty" else None)
        step_override_applied.append(override_applied)
        if gate_case is not None:
            gate_case_counts[gate_case] = gate_case_counts.get(gate_case, 0) + 1
        next_token_scores = next_token_scores / temp[:, None]
        # Optional yes/no constrained first-token decoding.
        if force_yesno == "mask_logits" and step_idx == 0 and force_yesno_ids:
            allowed = torch.tensor(force_yesno_ids, dtype=torch.long, device=next_token_scores.device)
            allowed = allowed[(allowed >= 0) & (allowed < next_token_scores.size(-1))]
            if allowed.numel() > 0:
                mask = torch.full_like(next_token_scores, float("-inf"))
                mask.scatter_(1, allowed.unsqueeze(0).expand(next_token_scores.size(0), -1), 0.0)
                next_token_scores = next_token_scores + mask
        entropy_time_s += time.perf_counter() - t0_entropy
        row = {
            "step": step_idx,
            "Ht": float(ht.mean().item()),
            "Ha": None if ha is None else float(ha.mean().item()),
            "Ha_last_hat": None if ha_last_hat is None else float(ha_last_hat.mean().item()),
            "S_last": None if s_last is None else float(s_last.mean().item()),
            "Ht_hat": None if temp_mode != "dual_uncertainty" else (
                float(ht_hat.mean().item()) if "ht_hat" in locals() and ht_hat is not None else None
            ),
            "Ha_hat": None if temp_mode != "dual_uncertainty" else (
                float(ha_hat.mean().item()) if "ha_hat" in locals() and ha_hat is not None else None
            ),
            "gate_on": bool(gate_on),
            "gate_case": gate_case,
            "task_type": task_type if temp_mode == "dual_uncertainty" else None,
            "override_applied": override_applied,
            "t_text_applied": float(temp.mean().item()),
            "t_attn_applied": float(current_t_attn),
            "force_no_intervention": bool(force_no_intervention),
            "subspace_applied": bool(subspace_applied_step),
            "subspace_online_risk": hook_risk_score,
            "subspace_online_alpha": hook_alpha,
            "subspace_risk_score": (hook_risk_score if hook_risk_score is not None
                                    else (None if subspace_risk_score is None else float(subspace_risk_score))),
            "subspace_risk_threshold": (subspace_tau if use_subspace_hook
                                        else (None if subspace_risk_threshold is None else float(subspace_risk_threshold))),
            "subspace_alpha": (hook_alpha if hook_alpha is not None
                               else float(max(0.0, min(1.0, subspace_shrink_alpha)))),
            "subspace_residual_before": subspace_res_before,
            "subspace_residual_after": subspace_res_after,
            "subspace_n_bases": int(len(subspace_basis_list)) if subspace_basis_list is not None else 0,
            "subspace_intervention_layer": (subspace_intervention_layer if use_subspace_hook else None),
        }
        if subspace_applied_step:
            subspace_applied_any = True
            if subspace_res_before is not None:
                subspace_res_before_vals.append(float(subspace_res_before))
            if subspace_res_after is not None:
                subspace_res_after_vals.append(float(subspace_res_after))
        if hook_risk_score is not None:
            hook_risk_vals.append(float(hook_risk_score))
        if hook_alpha is not None:
            hook_alpha_vals.append(float(hook_alpha))
        if tv_stats is None:
            row.update(
                {
                    "H_attn_text": None,
                    "H_attn_vision": None,
                    "m_text": None,
                    "m_vision": None,
                    "ratio_TV": None,
                    "delta_TV": None,
                    "attn_source": None,
                    "kv_len": None,
                    "n_vision_tokens": None,
                    "attn_max": None,
                }
            )
        else:
            row.update(tv_stats)
        if obj_stats is None:
            row.update(
                {
                    "object_attn_mass": None,
                    "object_attn_peak": None,
                    "object_attn_n_tokens": 0,
                }
            )
        else:
            row.update(obj_stats)
        step_telemetry.append(row)
        if temp_log:
            log = {
                "Ht": ht.detach().cpu().tolist(),
                "Ha": None if ha is None else ha.detach().cpu().tolist(),
                "T": temp.detach().cpu().tolist(),
            }
            if temp_mode == "soft2x2":
                log.update(
                    {
                        "t_attn": current_t_attn,
                        "s_t": aux.get("s_t") if temp_mode == "soft2x2" else None,
                        "s_a": aux.get("s_a") if temp_mode == "soft2x2" else None,
                    }
                )
            if temp_mode == "dual_uncertainty":
                log.update(
                    {
                        "t_attn": current_t_attn,
                        "Ht_hat": aux.get("Ht_hat") if temp_mode == "dual_uncertainty" else None,
                        "Ha_hat": aux.get("Ha_hat") if temp_mode == "dual_uncertainty" else None,
                        "Ht_ema": aux.get("Ht_ema") if temp_mode == "dual_uncertainty" else None,
                        "Ha_ema": aux.get("Ha_ema") if temp_mode == "dual_uncertainty" else None,
                        "gate_case": aux.get("gate_case") if temp_mode == "dual_uncertainty" else None,
                        "t_text_final": aux.get("t_text_final") if temp_mode == "dual_uncertainty" else None,
                        "t_attn_final": aux.get("t_attn_final") if temp_mode == "dual_uncertainty" else None,
                        "t_text_before": aux.get("t_text_before") if temp_mode == "dual_uncertainty" else None,
                        "t_text_after": aux.get("t_text_after") if temp_mode == "dual_uncertainty" else None,
                        "delta_up": aux.get("delta_up") if temp_mode == "dual_uncertainty" else None,
                        "gate_on": aux.get("gate_on") if temp_mode == "dual_uncertainty" else None,
                        "task_type": aux.get("task_type") if temp_mode == "dual_uncertainty" else None,
                        "override_applied": aux.get("override_applied") if temp_mode == "dual_uncertainty" else None,
                    }
                )
            if s_t is not None:
                log.update(
                    {
                        "S_t": s_t.detach().cpu().tolist(),
                        "S_bar": None if s_bar is None else s_bar.detach().cpu().tolist(),
                        "gate_on": gate_on,
                        "gate_tau": gate_tau,
                    }
                )
            print(log)
        step_idx += 1

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if keep_attn_history:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        do_sample_step = bool(do_sample)
        if safe_decode_on_bias and bias_active:
            risk = float(ht.mean().item())
            if (collapse_risk_threshold <= 0.0) or (risk >= collapse_risk_threshold):
                do_sample_step = False

        if do_sample_step:
            t0_sampling = time.perf_counter()
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            sampling_time_s += time.perf_counter() - t0_sampling
        else:
            t0_sampling = time.perf_counter()
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            sampling_time_s += time.perf_counter() - t0_sampling

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if next_tokens.numel() > 0 and len(first_tokens_ids) < 5:
            first_tokens_ids.append(int(next_tokens[0].item()))
        if step_idx == 0 and eos_token_tensor is not None:
            eos_ids = eos_token_tensor.tolist() if torch.is_tensor(eos_token_tensor) else [int(eos_token_tensor)]
            eos_at_step1 = bool(int(next_tokens[0].item()) in set(int(x) for x in eos_ids))
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        decode_step_count += 1
        del outputs

        # stash gate stats for final return
        model_kwargs["_gate_on_count"] = gate_on_count
        model_kwargs["_gate_total_steps"] = gate_total_steps
        model_kwargs["_gate_on_h_attn_sum"] = gate_on_h_attn_sum
        model_kwargs["_gate_on_h_t_sum"] = gate_on_h_t_sum

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        gate_on_count = int(model_kwargs.get("_gate_on_count", 0))
        gate_total_steps = int(model_kwargs.get("_gate_total_steps", 0))
        gate_on_h_attn_sum = float(model_kwargs.get("_gate_on_h_attn_sum", 0.0))
        gate_on_h_t_sum = float(model_kwargs.get("_gate_on_h_t_sum", 0.0))
        if model.config.is_encoder_decoder:
            out = GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            out = GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        # Attach gate stats for downstream summary
        out.gate_on_count = gate_on_count
        out.total_steps = gate_total_steps
        out.gate_on_h_attn_sum = gate_on_h_attn_sum
        out.gate_on_h_t_sum = gate_on_h_t_sum
        out.gate_cases = gate_cases
        out.step_task_types = step_task_types
        out.step_override_applied = step_override_applied
        out.gate_case_counts = gate_case_counts
        out.task_type = task_type
        out.override_ever = bool(n_steps_override > 0)
        out.n_steps_Ha_high = int(n_steps_ha_high)
        out.n_steps_override = int(n_steps_override)
        out.first_step_override = int(first_step_override)
        out.step_telemetry = step_telemetry
        ha_vals = [float(r.get("Ha")) for r in step_telemetry if r.get("Ha") is not None]
        out.first_tokens_ids = list(first_tokens_ids)
        out.eos_at_step1 = bool(eos_at_step1)
        out.avg_gen_len = int(input_ids.shape[1] - initial_input_len) if input_ids is not None else None
        out.gate_on_ratio = float(gate_on_count / max(1, gate_total_steps))
        out.attn_entropy_min = min(ha_vals) if ha_vals else None
        out.attn_entropy_mean = (sum(ha_vals) / len(ha_vals)) if ha_vals else None
        out.attn_entropy_max = max(ha_vals) if ha_vals else None
        out.subspace_shrink_applied = bool(subspace_applied_any)
        out.subspace_online_mode = bool(use_subspace_hook)
        # Use prefill risk as the canonical risk score (matches offline scale).
        # Fall back to hook_risk_vals mean only if prefill risk is unavailable.
        _prefill_r = (
            float(_intervention_hook.prefill_risk)
            if _intervention_hook is not None and _intervention_hook.prefill_risk is not None
            else None
        )
        out.subspace_risk_score = (
            _prefill_r if _prefill_r is not None
            else (float(sum(hook_risk_vals) / len(hook_risk_vals)) if hook_risk_vals
                  else (None if subspace_risk_score is None else float(subspace_risk_score)))
        )
        out.subspace_prefill_risk = _prefill_r
        out.subspace_risk_threshold = (subspace_tau if use_subspace_hook
                                       else (None if subspace_risk_threshold is None else float(subspace_risk_threshold)))
        out.subspace_shrink_alpha = float(max(0.0, min(1.0, subspace_shrink_alpha)))
        out.subspace_lambda_max = (subspace_lambda_max if use_subspace_hook else None)
        out.subspace_tau = (subspace_tau if use_subspace_hook else None)
        out.subspace_tau_max = (subspace_tau_max if use_subspace_hook else None)
        out.subspace_intervention_layer = (subspace_intervention_layer if use_subspace_hook else None)
        out.subspace_n_bases = int(len(subspace_basis_list)) if subspace_basis_list is not None else 0
        out.subspace_residual_before_mean = (
            float(sum(subspace_res_before_vals) / len(subspace_res_before_vals)) if subspace_res_before_vals else None
        )
        out.subspace_residual_after_mean = (
            float(sum(subspace_res_after_vals) / len(subspace_res_after_vals)) if subspace_res_after_vals else None
        )
        out.timing = {
            "forward_s": float(forward_time_s),
            "entropy_s": float(entropy_time_s),
            "sampling_s": float(sampling_time_s),
            "total_s": float(forward_time_s + entropy_time_s + sampling_time_s),
        }
        out.force_no_intervention = bool(force_no_intervention)
        # Clean up hook
        if _hook_handle is not None:
            _hook_handle.remove()
        if _intervention_hook is not None:
            _intervention_hook.enabled = False
        return out

    # Clean up hook on non-dict path
    if _hook_handle is not None:
        _hook_handle.remove()
    if _intervention_hook is not None:
        _intervention_hook.enabled = False
    return input_ids
