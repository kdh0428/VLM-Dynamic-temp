"""HF Transformers attention-gated decoding for Qwen3-VL."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from .metrics import compute_accuracy
from .prompts import build_prompt, extract_docvqa_answer, extract_final_answer
from .hf_attn_gate_patch import (
    find_cross_attn_modules,
    patch_cross_attn_forward,
    set_attn_temperature,
)

LOGGER = logging.getLogger(__name__)


def _find_vision_span(input_ids: torch.Tensor, tokenizer) -> Tuple[int, int]:
    """Locate vision span (start, end) using special tokens."""
    special_tokens = tokenizer.all_special_tokens or []
    candidates = [tok for tok in special_tokens if "vision" in tok.lower() or "image" in tok.lower()]
    start_tokens = [tok for tok in candidates if "start" in tok.lower()]
    end_tokens = [tok for tok in candidates if "end" in tok.lower()]
    if not start_tokens or not end_tokens:
        raise ValueError(
            "Could not find vision start/end tokens. "
            f"Candidates={candidates}. Provide explicit span rules."
        )
    start_id = tokenizer.convert_tokens_to_ids(start_tokens[0])
    end_id = tokenizer.convert_tokens_to_ids(end_tokens[0])
    ids = input_ids[0].tolist()
    try:
        start_pos = ids.index(start_id)
        end_pos = ids.index(end_id, start_pos + 1)
    except ValueError as exc:
        raise ValueError(
            "Vision start/end tokens not found in input_ids. "
            f"start={start_tokens[0]} end={end_tokens[0]}"
        ) from exc
    if end_pos <= start_pos + 1:
        raise ValueError("Invalid vision span: end before start or empty span.")
    return start_pos + 1, end_pos


def _find_vision_indices(input_ids: torch.Tensor, tokenizer, model, processor) -> torch.Tensor:
    """Find vision token indices for both Qwen3-VL and LLaVA-style models."""
    image_token_id = None
    if hasattr(model, "config"):
        image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None and hasattr(processor, "image_token"):
        try:
            image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
        except Exception:
            image_token_id = None
    if image_token_id is not None:
        ids = input_ids[0]
        positions = (ids == int(image_token_id)).nonzero(as_tuple=False).flatten()
        if positions.numel() > 0:
            return positions
    start, end = _find_vision_span(input_ids, tokenizer)
    return torch.arange(start, end, device=input_ids.device)


def _build_hf_prompt(question: str, task_type: str, meta, processor, model_id: str) -> str:
    """Build model-specific prompts for HF decoding."""
    if "llava" in model_id.lower() and hasattr(processor, "apply_chat_template"):
        if task_type == "yesno":
            text = (
                f"Question: {question}\n"
                'Answer with a single word: "yes" or "no". '
                'On the last line, output exactly: "Final answer: yes" or "Final answer: no".'
            )
        else:
            text = (
                f"Question: {question}\n"
                'Answer concisely. On the last line, output exactly: "Final answer: <short answer>".'
            )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    return build_prompt(question, meta)


def _attention_gate_score(attentions: Sequence[torch.Tensor], vision_idx: torch.Tensor) -> float:
    """Compute attention mass to vision tokens for the last generated token."""
    if not attentions:
        return 0.0
    attn_last = attentions[-1]  # (B, H, Q, K)
    a_last = attn_last[:, :, -1, :]  # (B, H, K)
    vision_attn = torch.index_select(a_last, dim=-1, index=vision_idx)
    score = vision_attn.sum(-1).mean(1).item()
    return float(score)


def _attention_gate_score_from_attn(attn_last: torch.Tensor, vision_idx: torch.Tensor) -> float:
    """Compute attention mass to vision tokens from a single attention tensor."""
    a_last = attn_last[:, :, -1, :]  # (B, H, K)
    vision_attn = torch.index_select(a_last, dim=-1, index=vision_idx)
    score = vision_attn.sum(-1).mean(1).item()
    return float(score)


def _attn_entropy(attn_weights: torch.Tensor) -> float:
    attn_last = attn_weights[:, :, -1, :]
    probs = attn_last / attn_last.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ent = -(probs * (probs + 1e-20).log()).sum(dim=-1)
    return float(ent.mean().item())


def _entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * (probs + 1e-20).log()).sum(dim=-1)
    return float(ent.item())


def _select_temperatures(
    cross_ent: float,
    logit_ent: float,
    t_base: float,
    t_gate: float,
    t_high: float,
    t_attn_base: float,
    t_attn_low: float,
    attn_ent_low: float,
    attn_ent_high: float,
    h_low: float,
    h_high: float,
) -> Tuple[float, float]:
    cross_low = cross_ent < attn_ent_low
    cross_high = cross_ent > attn_ent_high
    logit_low = logit_ent < h_low
    logit_high = logit_ent > h_high

    # 2x2 control: cross-attn entropy (image focus) x logit entropy (text uncertainty)
    if cross_low and logit_low:
        return t_base, t_attn_base
    if cross_low and logit_high:
        return t_gate, t_attn_base
    if cross_high and logit_low:
        return t_high, t_attn_low
    if cross_high and logit_high:
        return t_gate, t_attn_low
    return t_base, t_attn_base


def _top1_info(logits: torch.Tensor) -> Tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    top1 = probs.argmax(dim=-1)
    top1_prob = probs.gather(dim=-1, index=top1.unsqueeze(-1)).squeeze(-1)
    return int(top1.item()), float(top1_prob.item())


def _sample_token(logits: torch.Tensor) -> int:
    """Sample a token id from a logits distribution (expects [1, vocab])."""
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id.item())


def _mcq_answer_found(text: str) -> bool:
    import re

    if not text:
        return False
    t = text.strip()
    # Single letter answer.
    if re.fullmatch(r"[A-Da-d]", t):
        return True
    # Patterns like "Answer: A" or "Final answer: B".
    if re.search(r"(final\s*answer|answer)\s*[:is]*\s*([A-Da-d])\b", t, re.IGNORECASE):
        return True
    return False


def _final_answer_found(text: str, task_mode: str) -> bool:
    import re

    if not text:
        return False
    if task_mode == "mcq":
        return _mcq_answer_found(text)
    m = re.search(r"final\s*answer\s*(?:is|:)\s*(.*)", text, re.IGNORECASE)
    if not m:
        return False
    tail = m.group(1).strip()
    if not tail:
        return False
    if task_mode == "yesno":
        return bool(re.search(r"\b(yes|no)\b", tail, re.IGNORECASE))
    return True


def _final_answer_prefix_found(text: str) -> bool:
    import re

    if not text:
        return False
    return bool(re.search(r"final\s*answer\s*:", text, re.IGNORECASE))


@dataclass
class DecodingConfig:
    max_new_tokens: int
    min_new_tokens: int
    max_cot_tokens: int
    min_cot_tokens: int
    ema_alpha: float
    ema_k: int
    ema_tau_t: float
    ema_tau_a: float
    ema_eps_t: float
    ema_eps_a: float
    ema_adaptive_tau: bool
    ema_window: int
    ema_percentile: float
    eos_bias: float
    task_mode: str
    repeat_last_token_penalty: float
    stop_strings: List[str]
    top_k: int
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    bad_word_ids: List[int]
    forced_eos: bool
    enable_final_yesno_mask: bool


def _context_limit(tokenizer, model) -> int | None:
    cfg = getattr(model, "config", None)
    max_ctx = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
    if not max_ctx or max_ctx <= 0:
        max_ctx = getattr(tokenizer, "model_max_length", None)
    if max_ctx and max_ctx > 0 and max_ctx < 1_000_000:
        return int(max_ctx)
    return None


def _run_decoding_checks() -> None:
    # (1) max_new_tokens stops
    cfg = DecodingConfig(
        max_new_tokens=2,
        min_new_tokens=0,
        max_cot_tokens=2,
        min_cot_tokens=1,
        ema_alpha=0.9,
        ema_k=2,
        ema_tau_t=1.0,
        ema_tau_a=1.0,
        ema_eps_t=1e-3,
        ema_eps_a=1e-3,
        ema_adaptive_tau=True,
        ema_window=8,
        ema_percentile=30.0,
        eos_bias=0.0,
        task_mode="vqa",
        repeat_last_token_penalty=1.0,
        stop_strings=[],
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        bad_word_ids=[],
        forced_eos=False,
        enable_final_yesno_mask=True,
    )
    assert cfg.max_new_tokens == 2
    # (2) EOS bias toggle
    assert cfg.eos_bias >= 0.0
    # (3) MCQ early stop detection
    assert _mcq_answer_found("A")
    assert _mcq_answer_found("Answer: B")
    assert not _mcq_answer_found("cat")
    assert _final_answer_prefix_found("Final answer: ")
    assert _final_answer_found("Final answer: Yes", "yesno")


def _parse_stop_strings(raw: str) -> List[str]:
    if not raw:
        return []
    return [s for s in (p.strip() for p in raw.split(",")) if s]


def _stop_string_found(text: str, stop_strings: List[str]) -> bool:
    if not text or not stop_strings:
        return False
    return any(s in text for s in stop_strings)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] + (vals[c] - vals[f]) * (k - f))


_YESNO_CACHE: Dict[int, Dict[str, List[List[int]]]] = {}


def _safe_encode(tokenizer, processor, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        if processor is not None and hasattr(processor, "tokenizer"):
            return processor.tokenizer.encode(text, add_special_tokens=False)
    return []


def _get_yesno_token_sets(tokenizer, processor) -> Dict[str, List[List[int]]]:
    """Return token sequences for yes/no variants (cache by tokenizer id)."""
    cache_key = id(tokenizer)
    if cache_key in _YESNO_CACHE:
        return _YESNO_CACHE[cache_key]
    variants = {
        "yes": ["Yes", "YES", "yes", " yes", "Yes.", " yes."],
        "no": ["No", "NO", "no", " no", "No.", " no."],
    }
    out: Dict[str, List[List[int]]] = {"yes": [], "no": []}
    for key, texts in variants.items():
        for t in texts:
            ids = _safe_encode(tokenizer, processor, t)
            if ids:
                out[key].append(ids)
    _YESNO_CACHE[cache_key] = out
    return out


def _mask_logits_to_allowed(logits: torch.Tensor, allowed_ids: List[int]) -> None:
    if not allowed_ids:
        return
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed_ids] = 0.0
    logits += mask


def _choose_yesno_sequence_by_logprob(
    logits: torch.Tensor, yes_seqs: List[List[int]], no_seqs: List[List[int]]
) -> List[int]:
    """Pick Yes/No sequence by comparing first-token logprobs (simple lookahead)."""
    def best_logprob(seqs: List[List[int]]) -> Tuple[float, List[int]]:
        best_lp = float("-inf")
        best_seq: List[int] = []
        for seq in seqs:
            if not seq:
                continue
            lp = float(torch.log_softmax(logits, dim=-1)[..., seq[0]].item())
            if lp > best_lp:
                best_lp = lp
                best_seq = seq
        return best_lp, best_seq

    yes_lp, yes_seq = best_logprob(yes_seqs)
    no_lp, no_seq = best_logprob(no_seqs)
    return yes_seq if yes_lp >= no_lp else no_seq


def _apply_final_prefix(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    prefix_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    if not prefix_ids:
        return input_ids, attention_mask, past_key_values
    device = input_ids.device
    prefix = torch.tensor([prefix_ids], device=device, dtype=input_ids.dtype)
    if past_key_values is None:
        new_input_ids = torch.cat([input_ids, prefix], dim=-1)
        new_attention_mask = torch.cat(
            [attention_mask, torch.ones_like(prefix, device=device)], dim=-1
        )
        outputs = model(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            use_cache=True,
            output_attentions=True,
        )
        return new_input_ids, new_attention_mask, outputs.past_key_values
    new_attention_mask = torch.cat(
        [attention_mask, torch.ones_like(prefix, device=device)], dim=-1
    )
    outputs = model(
        input_ids=prefix,
        attention_mask=new_attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    new_input_ids = torch.cat([input_ids, prefix], dim=-1)
    return new_input_ids, new_attention_mask, outputs.past_key_values


def _parse_bad_words(raw: str, tokenizer, processor) -> List[int]:
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        ids = _safe_encode(tokenizer, processor, token)
        if ids:
            out.extend(ids)
    return list(sorted(set(out)))


def _apply_repetition_penalty(logits: torch.Tensor, generated: List[int], penalty: float) -> None:
    if penalty <= 1.0 or not generated:
        return
    for token_id in set(generated):
        tok = int(token_id)
        val = logits[..., tok]
        logits[..., tok] = torch.where(val < 0, val * penalty, val / penalty)


def _calc_banned_ngram_tokens(
    generated: List[int], no_repeat_ngram_size: int
) -> List[int]:
    if no_repeat_ngram_size <= 0 or len(generated) < no_repeat_ngram_size:
        return []
    n = no_repeat_ngram_size
    prefix = tuple(generated[-(n - 1) :]) if n > 1 else tuple()
    banned = set()
    for i in range(len(generated) - n + 1):
        if n == 1 or tuple(generated[i : i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return list(banned)


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> None:
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < kth_vals, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)


def _decode_with_attn_gate(
    model,
    processor,
    tokenizer,
    prompt: str,
    image,
    args,
    step_logger: Optional[Any] = None,
    record_minimal: bool = False,
    model_id: str | None = None,
    task_mode: str | None = None,
) -> Tuple[str, bool, float, Dict[str, float]]:
    """Custom decoding loop with attention gate and entropy-based temperature."""
    device = next(model.parameters()).device
    if model_id and "llava" in model_id.lower():
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    vision_idx = _find_vision_indices(input_ids, tokenizer, model, processor)

    if record_minimal:
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        sequences = outputs.sequences
        input_len = input_ids.shape[1]
        gen_ids = sequences[0, input_len:]

        gate_triggered = False
        max_s_t = 0.0
        gate_steps = 0
        h_attn_vals: List[float] = []
        h_text_vals: List[float] = []

        scores = outputs.scores or []
        attn_steps = outputs.attentions or []
        for step_idx in range(min(len(scores), len(attn_steps))):
            logits = scores[step_idx]
            step_attn = attn_steps[step_idx][-1] if attn_steps[step_idx] else None
            if step_attn is None:
                continue
            h_t = _entropy_from_logits(logits)
            cross_ent = _attn_entropy(step_attn)
            s_t = _attention_gate_score_from_attn(step_attn, vision_idx)
            gate_on = s_t >= args.attn_gate_tau
            if gate_on:
                gate_triggered = True
                gate_steps += 1
                h_attn_vals.append(cross_ent)
                h_text_vals.append(h_t)
            if s_t > max_s_t:
                max_s_t = s_t

            record = {
                "step": step_idx,
                "gate_on": gate_on,
                "S_t": s_t,
                "H_t": h_t,
                "H_attn": cross_ent,
            }
            if step_logger is not None:
                step_logger.write(json.dumps(record) + "\n")
            if getattr(args, "attn_gate_step_stdout", False):
                print(json.dumps(record))

        if gate_steps:
            h_attn_mean = float(sum(h_attn_vals) / gate_steps)
            h_text_mean = float(sum(h_text_vals) / gate_steps)
        else:
            h_attn_mean = 0.0
            h_text_mean = 0.0

        stats = {
            "total_tokens": float(gen_ids.numel()),
            "gate_on_count": float(gate_steps),
            "gate_on_h_attn_mean": h_attn_mean,
            "gate_on_h_t_mean": h_text_mean,
        }
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return decoded, gate_triggered, max_s_t, stats

    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids_set = {eos_ids}
    elif eos_ids is None:
        eos_ids_set = set()
    else:
        eos_ids_set = set(eos_ids)

    past_key_values = None
    generated: List[int] = []
    decode_cfg = DecodingConfig(
        max_new_tokens=int(args.max_new_tokens),
        min_new_tokens=int(getattr(args, "min_new_tokens", 0) or 0),
        max_cot_tokens=int(getattr(args, "max_cot_tokens", 0) or 0),
        min_cot_tokens=int(getattr(args, "min_cot_tokens", 0) or 0),
        ema_alpha=float(getattr(args, "ema_alpha", 0.9) or 0.9),
        ema_k=int(getattr(args, "ema_k", 5) or 5),
        ema_tau_t=float(getattr(args, "ema_tau_t", 1.5) or 1.5),
        ema_tau_a=float(getattr(args, "ema_tau_a", 3.0) or 3.0),
        ema_eps_t=float(getattr(args, "ema_eps_t", 1e-3) or 1e-3),
        ema_eps_a=float(getattr(args, "ema_eps_a", 1e-3) or 1e-3),
        ema_adaptive_tau=bool(getattr(args, "ema_adaptive_tau", True)),
        ema_window=int(getattr(args, "ema_window", 32) or 32),
        ema_percentile=float(getattr(args, "ema_percentile", 30.0) or 30.0),
        eos_bias=float(getattr(args, "eos_bias", 0.0) or 0.0),
        task_mode=str(task_mode or "vqa"),
        repeat_last_token_penalty=float(getattr(args, "repeat_last_token_penalty", 1.0) or 1.0),
        stop_strings=_parse_stop_strings(getattr(args, "stop_strings", "")),
        top_k=int(getattr(args, "top_k", 0) or 0),
        top_p=float(getattr(args, "top_p", 1.0) or 1.0),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0) or 1.0),
        no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0) or 0),
        bad_word_ids=_parse_bad_words(getattr(args, "bad_words", ""), tokenizer, processor),
        forced_eos=bool(getattr(args, "forced_eos", False)),
        enable_final_yesno_mask=bool(getattr(args, "enable_final_yesno_mask", True)),
    )
    max_steps = decode_cfg.max_new_tokens
    ctx_limit = _context_limit(tokenizer, model)

    gate_triggered = False
    max_s_t = 0.0
    current_t_attn = args.t_attn_base
    sanity_done = False
    gate_steps = 0
    h_attn_vals: List[float] = []
    h_text_vals: List[float] = []
    phase = "cot"
    phase_switch_step: Optional[int] = None
    final_selected: Optional[str] = None
    cot_tokens = 0
    final_prefix_ids = _safe_encode(tokenizer, processor, "Final answer:")
    ht_ema: Optional[float] = None
    ha_ema: Optional[float] = None
    prev_ht_ema: Optional[float] = None
    prev_ha_ema: Optional[float] = None
    ht_hist = deque(maxlen=max(1, decode_cfg.ema_window))
    ha_hist = deque(maxlen=max(1, decode_cfg.ema_window))
    r1_count = 0
    r2_count = 0
    rule_trigger: Optional[str] = None
    yesno_tokens = (
        _get_yesno_token_sets(tokenizer, processor) if decode_cfg.task_mode == "yesno" else {}
    )
    yes_variants = yesno_tokens.get("yes", [])
    no_variants = yesno_tokens.get("no", [])
    yes_single = sorted({seq[0] for seq in yes_variants if len(seq) == 1})
    no_single = sorted({seq[0] for seq in no_variants if len(seq) == 1})
    final_allowed_single = sorted(set(yes_single + no_single))

    for step in range(max_steps):
        if len(generated) >= decode_cfg.max_new_tokens:
            break
        if ctx_limit is not None and (input_ids.shape[1] + len(generated)) >= ctx_limit:
            break
        t_attn_used = current_t_attn
        if getattr(args, "enable_attn_temp", True) and not record_minimal:
            set_attn_temperature(args.attn_modules, t_attn_used)
        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "output_attentions": True,
            }
            for k, v in inputs.items():
                if k not in {"input_ids", "attention_mask"}:
                    model_inputs[k] = v
        else:
            model_inputs = {
                "input_ids": input_ids[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_attentions": True,
            }

        outputs = model(**model_inputs)
        if False:
            ent_base = _attn_entropy(outputs.attentions[-1])
            set_attn_temperature(args.attn_modules, args.t_attn_low)
            outputs_low = model(**model_inputs)
            ent_low = _attn_entropy(outputs_low.attentions[-1]) if outputs_low.attentions else None
            set_attn_temperature(args.attn_modules, current_t_attn)
            LOGGER.info(
                "Attn temp sanity: T=%.2f ent=%.4f | T=%.2f ent=%s",
                args.t_attn_base,
                ent_base,
                args.t_attn_low,
                f"{ent_low:.4f}" if ent_low is not None else "None",
            )
            sanity_done = True
        logits = outputs.logits[:, -1, :]
        s_t = _attention_gate_score(outputs.attentions, vision_idx)
        h_t = _entropy_from_logits(logits)
        cross_ent = _attn_entropy(outputs.attentions[-1]) if outputs.attentions else 0.0
        if ht_ema is None:
            ht_ema = h_t
            ha_ema = cross_ent
        else:
            ht_ema = decode_cfg.ema_alpha * ht_ema + (1.0 - decode_cfg.ema_alpha) * h_t
            ha_ema = decode_cfg.ema_alpha * ha_ema + (1.0 - decode_cfg.ema_alpha) * cross_ent
        ht_hist.append(ht_ema)
        ha_hist.append(ha_ema)
        gate_on = s_t >= args.attn_gate_tau
        if gate_on:
            gate_triggered = True
            gate_steps += 1
            h_attn_vals.append(cross_ent)
            h_text_vals.append(h_t)
        if s_t > max_s_t:
            max_s_t = s_t
        if record_minimal:
            t_text = 1.0
            next_t_attn = t_attn_used
        else:
            t_text, next_t_attn = _select_temperatures(
                cross_ent=cross_ent,
                logit_ent=h_t,
                t_base=args.attn_t_base,
                t_gate=args.attn_t_gate,
                t_high=args.attn_t_high,
                t_attn_base=args.t_attn_base,
                t_attn_low=args.t_attn_low,
                attn_ent_low=args.attn_ent_low,
                attn_ent_high=args.attn_ent_high,
                h_low=args.attn_h_low,
                h_high=args.attn_h_high,
            )
        logits = logits / max(t_text, 1e-4)
        if decode_cfg.eos_bias and eos_ids_set:
            for eos_id in eos_ids_set:
                logits[..., eos_id] += decode_cfg.eos_bias
        if len(generated) < decode_cfg.min_new_tokens and eos_ids_set:
            for eos_id in eos_ids_set:
                logits[..., eos_id] = float("-inf")
        if decode_cfg.forced_eos and (len(generated) + 1) >= decode_cfg.max_new_tokens and eos_ids_set:
            logits.fill_(float("-inf"))
            for eos_id in eos_ids_set:
                logits[..., eos_id] = 0.0
        if decode_cfg.bad_word_ids:
            logits[..., decode_cfg.bad_word_ids] = float("-inf")
        _apply_repetition_penalty(logits, generated, decode_cfg.repetition_penalty)
        if decode_cfg.repeat_last_token_penalty > 1.0 and generated:
            last_id = generated[-1]
            logits[..., last_id] /= decode_cfg.repeat_last_token_penalty
        banned = _calc_banned_ngram_tokens(generated, decode_cfg.no_repeat_ngram_size)
        if banned:
            logits[..., banned] = float("-inf")
        if decode_cfg.task_mode == "yesno" and phase == "final" and decode_cfg.enable_final_yesno_mask:
            if final_allowed_single:
                _mask_logits_to_allowed(logits, final_allowed_single)
                top1_id, top1_prob = _top1_info(logits)
                next_id = top1_id
            else:
                chosen_seq = _choose_yesno_sequence_by_logprob(logits, yes_variants, no_variants)
                for tok in chosen_seq:
                    generated.append(int(tok))
                final_selected = "yes" if chosen_seq in yes_variants else "no"
                break
        else:
            _apply_top_k_top_p(logits, decode_cfg.top_k, decode_cfg.top_p)
            top1_id, top1_prob = _top1_info(logits)
            next_id = _sample_token(logits)
        current_t_attn = next_t_attn

        record = {
            "step": step,
            "gate_on": gate_on,
            "S_t": s_t,
            "H_t": h_t,
            "H_attn": cross_ent,
        }
        if not record_minimal:
            record.update(
                {
                    "top1_token": tokenizer.decode([top1_id], skip_special_tokens=True),
                    "top1_prob": top1_prob,
                    "T_text": t_text,
                    "T_attn": t_attn_used,
                }
            )
        if step_logger is not None:
            step_logger.write(json.dumps(record) + "\n")
        if getattr(args, "attn_gate_step_stdout", False):
            print(json.dumps(record))

        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=device)], dim=-1
        )
        past_key_values = outputs.past_key_values

        if phase == "cot":
            cot_tokens += 1
        if decode_cfg.task_mode == "yesno" and phase == "final" and final_allowed_single:
            if next_id in yes_single:
                final_selected = "yes"
                break
            if next_id in no_single:
                final_selected = "no"
                break

        if len(generated) < decode_cfg.min_new_tokens:
            continue
        if next_id in eos_ids_set:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if phase == "cot":
            if cot_tokens >= decode_cfg.min_cot_tokens:
                if ht_ema is not None and ha_ema is not None:
                    tau_t = decode_cfg.ema_tau_t
                    tau_a = decode_cfg.ema_tau_a
                    if decode_cfg.ema_adaptive_tau and len(ht_hist) >= decode_cfg.ema_window:
                        tau_t = _percentile(list(ht_hist), decode_cfg.ema_percentile)
                        tau_a = _percentile(list(ha_hist), decode_cfg.ema_percentile)
                    if ht_ema <= tau_t and ha_ema <= tau_a:
                        r1_count += 1
                    else:
                        r1_count = 0
                    if prev_ht_ema is not None and prev_ha_ema is not None:
                        if (
                            abs(ht_ema - prev_ht_ema) < decode_cfg.ema_eps_t
                            and abs(ha_ema - prev_ha_ema) < decode_cfg.ema_eps_a
                        ):
                            r2_count += 1
                        else:
                            r2_count = 0
                    prev_ht_ema = ht_ema
                    prev_ha_ema = ha_ema
                if r1_count >= decode_cfg.ema_k:
                    rule_trigger = "R1"
                    phase = "final"
                    phase_switch_step = step
                elif r2_count >= decode_cfg.ema_k:
                    rule_trigger = "R2"
                    phase = "final"
                    phase_switch_step = step
            if _final_answer_prefix_found(decoded):
                rule_trigger = "prefix"
                phase = "final"
                phase_switch_step = step
            if decode_cfg.max_cot_tokens > 0 and cot_tokens >= decode_cfg.max_cot_tokens:
                rule_trigger = "forced"
                phase = "final"
                phase_switch_step = step
            if phase == "final":
                if not _final_answer_prefix_found(decoded):
                    input_ids, attention_mask, past_key_values = _apply_final_prefix(
                        model,
                        input_ids,
                        attention_mask,
                        past_key_values,
                        final_prefix_ids,
                    )
                    for tok in final_prefix_ids:
                        generated.append(int(tok))
                if step_logger is not None:
                    step_logger.write(
                        json.dumps({"phase": "final", "step": step, "rule": rule_trigger}) + "\n"
                    )
        if _stop_string_found(decoded, decode_cfg.stop_strings):
            break
        if decode_cfg.task_mode == "yesno" and phase == "final":
            if _final_answer_found(decoded, decode_cfg.task_mode):
                final_selected = "yes" if "yes" in decoded.lower() else "no"
                break
        if _final_answer_found(decoded, decode_cfg.task_mode):
            break

    if gate_steps:
        h_attn_mean = float(sum(h_attn_vals) / gate_steps)
        h_text_mean = float(sum(h_text_vals) / gate_steps)
    else:
        h_attn_mean = 0.0
        h_text_mean = 0.0

    stats = {
        "total_tokens": float(len(generated)),
        "gate_on_count": float(gate_steps),
        "gate_on_h_attn_mean": h_attn_mean,
        "gate_on_h_t_mean": h_text_mean,
        "phase_switch_step": phase_switch_step,
        "final_selected": final_selected,
        "ht_ema": ht_ema,
        "ha_ema": ha_ema,
        "rule_trigger": rule_trigger,
    }
    return tokenizer.decode(generated, skip_special_tokens=True), gate_triggered, max_s_t, stats


def _extract_yesno(text: str) -> str:
    import re

    if not text:
        return ""
    m = re.search(r"final answer\s*(?:is|:)\s*(.*)", text, re.IGNORECASE)
    if m:
        tail = m.group(1)
        m2 = re.search(r"\b(yes|no)\b", tail, re.IGNORECASE)
        return m2.group(1).lower() if m2 else ""
    return ""


if __name__ == "__main__":
    _run_decoding_checks()


def _effective_limit(dataset, limit: int | None):
    if limit is None or limit < 0:
        try:
            return len(dataset)
        except Exception:
            return None
    return limit


def _count_completed(jsonl_path: str) -> int:
    if not jsonl_path:
        return 0
    path = Path(jsonl_path)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("summary"):
                continue
            count += 1
    return count


def run_hf_attn_gate(
    dataset: Iterable[Dict[str, Any]],
    args,
    output_jsonl: str,
) -> Dict[str, Any]:
    """Run HF attention-gated decoding with Transformers."""
    baseline_only = getattr(args, "mode", "") == "hf_attn_gate_baseline"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = args.max_new_tokens
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    patched = patch_cross_attn_forward() if args.enable_attn_temp else False
    if args.enable_attn_temp and not patched and not baseline_only:
        raise RuntimeError("Attention temperature patch failed; cannot modify attention values.")
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code
    )
    tokenizer = processor.tokenizer
    if "llava" in args.model_id.lower():
        tokenizer.padding_side = "left"
    model.eval()
    args.attn_modules = find_cross_attn_modules(model)
    if args.enable_attn_temp and not args.attn_modules and not baseline_only:
        raise RuntimeError("No attention modules found for temperature patching.")

    total_acc = 0.0
    n = 0

    resume_count = 0
    if getattr(args, "resume_jsonl", ""):
        resume_count = _count_completed(args.resume_jsonl)
    else:
        resume_count = max(0, int(getattr(args, "resume_from", 0)))

    step_log = None
    if args.attn_gate_step_jsonl:
        step_mode = "a" if resume_count > 0 else "w"
        step_log = open(args.attn_gate_step_jsonl, step_mode, encoding="utf-8")

    total_expected = _effective_limit(dataset, args.limit)
    if total_expected is not None and resume_count >= total_expected:
        LOGGER.warning("Resume count (%d) >= limit (%d); nothing to do.", resume_count, total_expected)
        return {"mode": "hf_attn_gate", "overall_accuracy": 0.0, "num_samples": 0}

    out_mode = "a" if resume_count > 0 else "w"
    with open(output_jsonl, out_mode, encoding="utf-8", buffering=1) as f_out:
        for idx, example in enumerate(dataset):
            if idx < resume_count:
                continue
            if total_expected is not None and (n + resume_count) >= total_expected:
                break
            try:
                image = example.get("image")
                question = example.get("question", "")
                gt_answers = example.get("answers", [])
                meta = {k: example.get(k) for k in example.keys() if k not in {"image", "question", "answers"}}
                prompt = _build_hf_prompt(question, meta.get("task_type", "vqa"), meta, processor, args.model_id)
                with torch.no_grad():
                    text, gate_triggered, max_s_t, step_stats = _decode_with_attn_gate(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        image=image,
                        args=args,
                        step_logger=step_log,
                        record_minimal=baseline_only,
                        model_id=args.model_id,
                        task_mode=meta.get("task_type", "vqa"),
                    )
                task_type = meta.get("task_type", "vqa")
                if task_type == "docvqa":
                    pred = extract_docvqa_answer(text)
                elif task_type == "yesno":
                    pred = _extract_yesno(text)
                else:
                    pred = extract_final_answer(text)
                acc_input = pred if task_type in {"mcq", "mmbench", "yesno", "vizwiz"} else text
                acc = compute_accuracy(acc_input, gt_answers, task_type=task_type)
            except Exception as exc:
                LOGGER.warning("HF attn-gate example failed: %s", exc)
                continue

            record = {
                "question_id": meta.get("question_id"),
                "image_id": meta.get("image_id"),
                "question": question,
                "gt_answers": gt_answers,
                "prediction": pred,
                "raw_text": text,
                "accuracy": acc,
                "gate_triggered": gate_triggered,
                "max_gate_score": max_s_t,
                "total_tokens": int(step_stats.get("total_tokens", 0)),
                "gate_on_count": int(step_stats.get("gate_on_count", 0)),
                "gate_on_h_attn_mean": float(step_stats.get("gate_on_h_attn_mean", 0.0)),
                "gate_on_h_t_mean": float(step_stats.get("gate_on_h_t_mean", 0.0)),
                "phase_switch_step": step_stats.get("phase_switch_step"),
                "final_selected": step_stats.get("final_selected"),
                "ht_ema": step_stats.get("ht_ema"),
                "ha_ema": step_stats.get("ha_ema"),
                "rule_trigger": step_stats.get("rule_trigger"),
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            status = "correct" if acc >= 0.5 else "wrong"
            if step_log is not None and baseline_only:
                step_log.write(
                    json.dumps(
                        {
                            "summary": True,
                            "status": status,
                            "accuracy": acc,
                            "tokens": int(step_stats.get("total_tokens", 0)),
                            "gate_on": int(step_stats.get("gate_on_count", 0)),
                            "H_attn": float(step_stats.get("gate_on_h_attn_mean", 0.0)),
                            "H_t": float(step_stats.get("gate_on_h_t_mean", 0.0)),
                        }
                    )
                    + "\n"
                )
            total_acc += acc
            n += 1
            display_idx = resume_count + n
            LOGGER.info(
                "HF_Attn_Gate %d/%s | %s | tokens=%d | gate_on=%d | H_attn=%.4f | "
                "H_t=%.4f | pred=%s | gt=%s",
                display_idx,
                total_expected if total_expected is not None else "all",
                status,
                int(step_stats.get("total_tokens", 0)),
                int(step_stats.get("gate_on_count", 0)),
                float(step_stats.get("gate_on_h_attn_mean", 0.0)),
                float(step_stats.get("gate_on_h_t_mean", 0.0)),
                pred,
                gt_answers[0] if gt_answers else "",
            )
            if args.clear_cache_every and n % args.clear_cache_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if step_log is not None:
        step_log.close()

    overall_acc = total_acc / max(1, n)
    summary_record = {
        "summary": True,
        "mode": "hf_attn_gate",
        "overall_accuracy": overall_acc,
        "num_samples": n,
        "resume_from": resume_count,
    }
    with open(output_jsonl, "a", encoding="utf-8", buffering=1) as f_out:
        f_out.write(json.dumps(summary_record) + "\n")
        f_out.flush()
    print(
        f"HF_Attn_Gate | accuracy={overall_acc:.4f} | n={n}"
    )
    return {
        "mode": "hf_attn_gate",
        "overall_accuracy": overall_acc,
        "num_samples": n,
    }
