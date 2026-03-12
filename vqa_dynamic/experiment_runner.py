"""HF Transformers attention-gated decoding for Qwen3-VL."""

from __future__ import annotations

import gc
import os
import json
import csv
import logging
import re
import time
import io
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    InstructBlipForConditionalGeneration,
)

from intervention_generate import entropy_temp_generate
from intervention_generate import AttnGateSmoother, AttnGateSmootherConfig
from .metrics import compute_accuracy
from .prompts import build_prompt, extract_docvqa_answer, extract_final_answer
from .attn_patch import (
    find_cross_attn_modules,
    patch_cross_attn_forward,
    set_attn_capture_pre_softmax,
    set_attn_key_bias,
    set_attn_temperature,
)

LOGGER = logging.getLogger(__name__)


def _to_pil_image(image) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    try:
        return Image.fromarray(np.array(image))
    except Exception:
        return None


def _apply_visual_perturbation(image, kind: str, strength: float):
    img = _to_pil_image(image)
    if img is None:
        return image
    k = (kind or "blur").strip().lower()
    s = max(0.0, float(strength))
    if k == "blur":
        radius = max(0.1, 0.8 * s)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if k == "jpeg":
        quality = int(max(20, min(95, 85 - 20 * s)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert(img.mode if img.mode else "RGB")
    if k == "noise":
        arr = np.array(img).astype(np.float32)
        sigma = max(2.0, 8.0 * s)
        noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    if k == "color":
        # mild saturation/contrast jitter
        sat = max(0.6, 1.0 - 0.2 * s)
        con = max(0.7, 1.0 - 0.15 * s)
        out = ImageEnhance.Color(img).enhance(sat)
        out = ImageEnhance.Contrast(out).enhance(con)
        return out
    return img


def _create_null_image(image, kind: str = "blank"):
    """Create a null image (same size as original) for VCD pass-2."""
    img = _to_pil_image(image)
    if img is None:
        return image
    w, h = img.size
    if kind == "noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    return Image.new("RGB", (w, h), (128, 128, 128))


def _find_yesno_token_ids(tokenizer):
    """Find token IDs for yes/no variants used by LLaVA-style models."""
    yes_ids = set()
    no_ids = set()
    for word in ["Yes", "yes", " Yes", " yes"]:
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if ids:
            yes_ids.add(ids[0])
    for word in ["No", "no", " No", " no"]:
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if ids:
            no_ids.add(ids[0])
    return sorted(yes_ids), sorted(no_ids)


def _extract_yesno_margin(stats: Dict[str, Any]) -> Optional[float]:
    v = stats.get("top12_prob_margin")
    if v is None:
        v = stats.get("top12_margin")
    try:
        return float(v)
    except Exception:
        return None


def _load_subspace_risk_map(path: str, risk_col: str) -> Dict[int, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"subspace risk csv not found: {path}")
    risk: Dict[int, float] = {}
    with open(p, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            return {}
        if "sample_id" not in rd.fieldnames:
            raise ValueError(f"subspace risk csv missing sample_id: {path}")
        if risk_col not in rd.fieldnames:
            raise ValueError(f"subspace risk csv missing {risk_col}: {path}")
        for row in rd:
            try:
                sid = int(row.get("sample_id"))
                rv = float(row.get(risk_col))
            except Exception:
                continue
            if np.isfinite(rv):
                risk[sid] = rv
    return risk


def _quantile_threshold(values: List[float], top_pct: float) -> float:
    if not values:
        return float("inf")
    p = float(top_pct)
    p = min(max(p, 1e-6), 1.0)
    q = 1.0 - p
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _force_eager_attention_impl(model) -> None:
    """Best-effort: force eager attention path on wrapper and nested language model."""
    targets = [("model", model)]
    lm = getattr(model, "language_model", None)
    if lm is not None:
        targets.append(("language_model", lm))
    inner = getattr(model, "model", None)
    if inner is not None and inner is not lm:
        targets.append(("inner_model", inner))

    for name, obj in targets:
        try:
            if hasattr(obj, "set_attn_implementation"):
                obj.set_attn_implementation("eager")
        except Exception as exc:
            LOGGER.warning("Could not set eager attention on %s: %s", name, exc)
        try:
            cfg = getattr(obj, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                setattr(cfg, "_attn_implementation", "eager")
        except Exception:
            pass

    for name, obj in targets:
        cfg = getattr(obj, "config", None)
        impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
        LOGGER.info("Attention implementation [%s]: %s", name, impl)


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
    try:
        start, end = _find_vision_span(input_ids, tokenizer)
        return torch.arange(start, end, device=input_ids.device)
    except Exception:
        return None


def _find_subsequence_positions(seq: List[int], subseq: List[int]) -> List[int]:
    if not subseq or len(subseq) > len(seq):
        return []
    n = len(subseq)
    out: List[int] = []
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == subseq:
            out.extend(range(i, i + n))
    return sorted(set(out))


def _keyword_token_positions(input_ids: torch.Tensor, tokenizer, word: str) -> List[int]:
    text = (word or "").strip()
    if not text:
        return []
    seq = input_ids[0].tolist()
    toks = tokenizer(text, add_special_tokens=False).input_ids
    if not toks:
        return []
    return _find_subsequence_positions(seq, toks)


def _strip_leading_determiners(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"^(?:any|some|this|that|these|those)\b\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:a|an|the)\b\s*", "", s, flags=re.IGNORECASE).strip()
    return s


def _extract_object_from_question(question: str) -> str:
    """Best-effort object phrase extraction for MME-Hall/POPE yes/no questions."""
    q = (question or "").strip().lower()
    if not q:
        return ""
    q = q.replace("？", "?")
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?.!]+$", "", q).strip()
    q = re.sub(r"\s*please answer yes or no\.?$", "", q).strip()
    q = re.sub(r"\s*answer yes or no only\.?$", "", q).strip()
    q = re.sub(r"\s*please answer with yes or no\.?$", "", q).strip()

    patterns = [
        # Location/relationship style: focus object is the subject NP.
        r"^(?:is|are)\s+(?:(?:a|an|the)\b\s+)?(?P<obj>.+?)\s+(?:on the (?:left|right)(?: side)? of|above|under|behind|in front of|out of|in|on top of|next to)\s+.+$",
        # There-be with count modifiers and optional "appear".
        r"^(?:is|are)\s+there\s+(?:only\s+)?(?:a total of\s+)?(?:no|one|two|three|four|five|six|seven|eight|nine|ten|\d+)?\s*(?P<obj>.+?)\s+(?:appear(?:ing)?\s+)?(?:in|on|at)\s+(?:this|the)\s+(?:image|picture)$",
        # There-be generic with image/picture tail.
        r"^(?:is|are)\s+there\s+(?:only\s+)?(?:a total of\s+)?(?:(?:any|some|a|an|the)\b\s+)?(?P<obj>.+?)\s+(?:in|on|at)\s+(?:this|the)\s+(?:image|picture)$",
        # There-be generic without strict tail.
        r"^(?:is|are)\s+there\s+(?:only\s+)?(?:a total of\s+)?(?:(?:any|some|a|an|the|no)\b\s+)?(?P<obj>.+?)$",
        r"^is there (?:(?:a|an|the)\b\s+)?(?P<obj>.+?)\s+(?:in|on|at)\s+the image$",
        r"^is there (?P<obj>.+?)\s+(?:in|on|at)\s+the image$",
        r"^are there (?P<obj>.+?)\s+(?:in|on|at)\s+the image$",
        r"^does (?:the image|this image) (?:contain|have|include)\s+(?P<obj>.+)$",
        r"^is (?:(?:a|an|the)\b\s+)?(?P<obj>.+?)\s+(?:present|visible)\s+(?:in|on)\s+the image$",
    ]
    for pat in patterns:
        m = re.match(pat, q)
        if m:
            gd = m.groupdict()
            obj = (gd.get("obj") or "").strip()
            obj = re.sub(r"\b(any|some)\b", "", obj).strip()
            obj = re.sub(r"^(?:a|an|the)\b\s*", "", obj).strip()
            obj = re.sub(r"^there\b\s*", "", obj).strip()
            obj = re.sub(r"^(?:a\s+total\s+of\s+)", "", obj).strip()
            obj = re.sub(
                r"^(?:only\s+)?(?:a total of\s+)?(?:no|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+",
                "",
                obj,
            ).strip()
            obj = re.sub(r"\bappear(?:ing)?\b", "", obj).strip()
            # Cut object before trailing relational preposition phrase.
            obj = re.split(
                r"\s+(?:on the (?:left|right)(?: side)? of|above|under|behind|in front of|next to|in|on|at|with|without|of)\s+",
                obj,
                maxsplit=1,
            )[0].strip()
            obj = re.sub(r"\s+", " ", obj)
            return _strip_leading_determiners(obj)
    return ""


def _head_from_count_phrase(phrase: str) -> str:
    """Fallback noun head from count phrase if phrase token matching fails."""
    p = (phrase or "").strip().lower()
    if not p:
        return ""
    p = re.sub(r"^(?:only\s+)?(?:a\s+total\s+of\s+)?", "", p).strip()
    p = re.sub(
        r"^(?:no|zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+",
        "",
        p,
    ).strip()
    # remove leading articles/determiners from fallback
    p = _strip_leading_determiners(p)
    return p


def _build_hf_prompt(question: str, task_type: str, meta, processor, model_id: str) -> str:
    """Build model-specific prompts for HF decoding."""
    if "instructblip" in model_id.lower():
        if task_type == "yesno":
            return f"Question: {question} Answer with exactly one word: yes or no."
        return f"Question: {question} Answer concisely."

    if "llava" in model_id.lower():
        if task_type == "yesno":
            llava_cot = bool(meta.get("llava_cot", False)) if meta else False
            system_msg = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            fewshot = ""
            if meta and meta.get("dataset_id") == "lmms-lab/MME-Hall":
                if llava_cot:
                    fewshot = (
                        "Examples:\n"
                        "Q: Is there a cat in the image?\n"
                        "Reasoning: I can see a cat in the scene.\n"
                        "Final answer: Yes\n"
                        "Q: Is there a bicycle in the image?\n"
                        "Reasoning: No bicycle is visible.\n"
                        "Final answer: No\n"
                    )
                else:
                    fewshot = (
                        "Examples:\n"
                        "Q: Is there a cat in the image? A: yes\n"
                        "Q: Is there a bicycle in the image? A: no\n"
                    )
            prefix = f"{fewshot}\n" if fewshot else ""
            if llava_cot:
                query = (
                    f"{prefix}"
                    f"{question}\n"
                    "First, provide brief reasoning in 1-2 sentences.\n"
                    'Then on the last line output exactly one of:\n'
                    '"Final answer: Yes"\n'
                    '"Final answer: No"'
                )
            else:
                query = f"{prefix}{question}"
            return f"USER: <image>\n{query}\nASSISTANT:"
        if hasattr(processor, "apply_chat_template"):
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


def _attention_gate_score(attentions: Sequence[torch.Tensor], vision_idx: Optional[torch.Tensor]) -> float:
    """Compute attention mass to vision tokens for the last generated token."""
    if not attentions or vision_idx is None:
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


def _attn_entropy(attn_weights: torch.Tensor, vision_idx: Optional[torch.Tensor] = None) -> float:
    attn_last = attn_weights[:, :, -1, :]
    if vision_idx is not None and vision_idx.numel() > 0:
        attn_last = torch.index_select(attn_last, dim=-1, index=vision_idx)
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
    question_text: str | None = None,
    sample_id: int | None = None,
) -> Tuple[str, bool, float, Dict[str, float]]:
    """Custom decoding loop with attention gate and entropy-based temperature."""
    device = next(model.parameters()).device
    model_id_l = (model_id or "").lower()
    if "llava" in model_id_l:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    elif "instructblip" in model_id_l:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    # InstructBLIP occasionally returns pixel_values as 5D (B,1,C,H,W). Force 4D (B,C,H,W).
    if "instructblip" in model_id_l and "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        if torch.is_tensor(pv) and pv.dim() == 5 and pv.size(1) == 1:
            inputs["pixel_values"] = pv.squeeze(1)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    bias_modules = getattr(args, "attn_bias_modules", getattr(args, "attn_modules", []))

    # Optional: force attention toward specific question keyword token positions (e.g., "clock").
    keyword = str(getattr(args, "attn_keyword_bias_word", "") or "").strip()
    requested_keyword = keyword
    if not keyword and bool(getattr(args, "attn_keyword_bias_auto_object", False)):
        keyword = _extract_object_from_question(str(question_text or ""))
        requested_keyword = keyword
    keyword_bias = float(getattr(args, "attn_keyword_bias", 0.0) or 0.0)
    applied_keyword = ""
    keyword_positions: List[int] = []
    if keyword and bias_modules:
        keyword_core = _strip_leading_determiners(keyword)
        if keyword_core:
            keyword = keyword_core
        key_pos = _keyword_token_positions(input_ids, tokenizer, keyword)
        if not key_pos and bool(getattr(args, "attn_keyword_bias_auto_object", False)):
            # Count-phrase first; if unmatched in prompt tokenization, back off to head noun.
            fallback_keyword = _head_from_count_phrase(keyword)
            if fallback_keyword and fallback_keyword != keyword:
                key_pos = _keyword_token_positions(input_ids, tokenizer, fallback_keyword)
                if key_pos:
                    LOGGER.info(
                        "Keyword bias fallback applied: phrase=%r -> head=%r",
                        keyword,
                        fallback_keyword,
                    )
                    keyword = fallback_keyword
        if key_pos:
            keyword_positions = list(key_pos)
            applied_keyword = keyword
            if abs(keyword_bias) > 1e-12:
                set_attn_key_bias(bias_modules, key_pos, keyword_bias)
                LOGGER.info("Keyword attention bias applied: word=%r bias=%.4f n_pos=%d", keyword, keyword_bias, len(key_pos))
            else:
                set_attn_key_bias(bias_modules, None, 0.0)
                LOGGER.info("Keyword attention telemetry enabled: word=%r n_pos=%d (bias=0)", keyword, len(key_pos))
        else:
            set_attn_key_bias(bias_modules, None, 0.0)
            LOGGER.warning("Keyword bias requested but token not found in prompt: %r", keyword)
    else:
        set_attn_key_bias(bias_modules, None, 0.0)

    vision_idx = _find_vision_indices(input_ids, tokenizer, model, processor)
    if vision_idx is None:
        LOGGER.warning("Vision token span not found; disabling gate_on for this sample.")

    if record_minimal:
        need_attn_outputs = bool(
            getattr(args, "use_attn_entropy", False)
            or getattr(args, "use_gate_for_intervention", False)
            or getattr(args, "enable_generate_temp_gate_stats", False)
        )
        need_score_outputs = bool(
            getattr(args, "enable_generate_temp_gate_stats", False)
            or getattr(args, "use_attn_entropy", False)
            or getattr(args, "logit_margin_top12", False)
            or str(getattr(args, "vcd_mode", "off")) != "off"
        )
        if getattr(args, "mode", "") == "hf_attn_gate_generate_temp":
            vision_start = None
            vision_end = None
            if getattr(args, "use_attn_entropy", False):
                try:
                    vision_start, vision_end = _find_vision_span(input_ids, tokenizer)
                except Exception:
                    vision_start = None
                    vision_end = None
            gen_cfg = model.generation_config
            try:
                gen_cfg = gen_cfg.__class__.from_dict(gen_cfg.to_dict())
            except Exception:
                import copy as _copy
                gen_cfg = _copy.deepcopy(gen_cfg)
            gen_cfg.return_dict_in_generate = True
            gen_cfg.output_attentions = need_attn_outputs
            gen_cfg.output_scores = need_score_outputs
            gen_cfg.temperature = 1.0
            gen_cfg.temp_mode = getattr(args, "temp_mode", "fixed")
            gen_cfg.temp_fixed = getattr(args, "temp_fixed", 1.0)
            gen_cfg.temp_min = getattr(args, "temp_min", 0.2)
            gen_cfg.temp_max = getattr(args, "temp_max", 1.5)
            gen_cfg.temp_a = getattr(args, "temp_a", 1.0)
            gen_cfg.temp_b = getattr(args, "temp_b", 1.0)
            gen_cfg.temp_c = getattr(args, "temp_c", 0.0)
            gen_cfg.temp_log = getattr(args, "temp_log", False)
            gen_cfg.use_attn_entropy = getattr(args, "use_attn_entropy", False)
            gen_cfg.vision_start_idx = vision_start
            gen_cfg.vision_end_idx = vision_end
            try:
                gen_cfg.vision_indices = vision_idx.tolist()
            except Exception:
                gen_cfg.vision_indices = None
            gen_cfg.ht_low = getattr(args, "ht_low", 0.5)
            gen_cfg.ht_high = getattr(args, "ht_high", 1.5)
            gen_cfg.ha_low = getattr(args, "ha_low", 2.0)
            gen_cfg.ha_high = getattr(args, "ha_high", 4.0)
            gen_cfg.t_text_low = getattr(args, "t_text_low", 0.7)
            gen_cfg.t_text_high = getattr(args, "t_text_high", 1.2)
            gen_cfg.soft2x2_ht0 = getattr(args, "soft2x2_ht0", 1.0)
            gen_cfg.soft2x2_ha0 = getattr(args, "soft2x2_ha0", 3.0)
            gen_cfg.soft2x2_k_t = getattr(args, "soft2x2_k_t", 6.0)
            gen_cfg.soft2x2_k_a = getattr(args, "soft2x2_k_a", 6.0)
            gen_cfg.soft2x2_ema_alpha = getattr(args, "soft2x2_ema_alpha", 0.0)
            gen_cfg.du_lam = getattr(args, "du_lam", 0.9)
            gen_cfg.du_k = getattr(args, "du_k", 2)
            gen_cfg.du_ht_low = getattr(args, "du_ht_low", 0.55)
            gen_cfg.du_ht_high = getattr(args, "du_ht_high", 0.65)
            gen_cfg.du_ha_low = getattr(args, "du_ha_low", 0.55)
            gen_cfg.du_ha_high = getattr(args, "du_ha_high", 0.65)
            gen_cfg.du_t_text_base = getattr(args, "du_t_text_base", 0.7)
            gen_cfg.du_t_text_min = getattr(args, "du_t_text_min", 0.1)
            gen_cfg.du_t_text_max = getattr(args, "du_t_text_max", 1.3)
            gen_cfg.du_t_attn_base = getattr(args, "du_t_attn_base", 1.0)
            gen_cfg.du_t_attn_min = getattr(args, "du_t_attn_min", 0.3)
            gen_cfg.du_t_attn_max = getattr(args, "du_t_attn_max", 2.0)
            gen_cfg.du_dt_explore = getattr(args, "du_dt_explore", 0.2)
            gen_cfg.du_dt_conserve = getattr(args, "du_dt_conserve", 0.1)
            gen_cfg.du_alpha_conserve = getattr(args, "du_alpha_conserve", 0.7)
            gen_cfg.du_dt_floor = getattr(args, "du_dt_floor", 0.05)
            gen_cfg.du_da_strong = getattr(args, "du_da_strong", 0.4)
            gen_cfg.du_ht_star = getattr(args, "du_ht_star", 0.6)
            gen_cfg.du_ha_star = getattr(args, "du_ha_star", 0.6)
            gen_cfg.du_k_t = getattr(args, "du_k_t", 0.8)
            gen_cfg.du_k_g = getattr(args, "du_k_g", 0.6)
            gen_cfg.du_k_a = getattr(args, "du_k_a", 0.8)
            gen_cfg.du_gateoff_lam = getattr(args, "du_gateoff_lam", 0.9)
            gen_cfg.du_ha_adversarial = bool(getattr(args, "du_ha_adversarial", False))
            gen_cfg.du_task_override = getattr(args, "du_task_override", "precise_c4")
            gen_cfg.du_precise_c4_allow_text_up = bool(getattr(args, "du_precise_c4_allow_text_up", False))
            gen_cfg.du_precise_c4_text_up_delta = float(getattr(args, "du_precise_c4_text_up_delta", 0.0))
            gen_cfg.task_question = str(question_text or "")
            gen_cfg.enable_t_attn = bool(getattr(args, "enable_attn_temp", True))
            gen_cfg.t_attn_base = getattr(args, "t_attn_base", 1.0)
            gen_cfg.t_attn_low = getattr(args, "t_attn_low", 0.7)
            gen_cfg.t_attn_high = getattr(args, "t_attn_high", 1.2)
            gen_cfg.attn_temp_sanity_check = bool(getattr(args, "attn_temp_sanity_check", False))
            gen_cfg.attn_temp_sanity_probe = float(getattr(args, "attn_temp_sanity_probe", 2.0))
            # Pass module refs via global to avoid deepcopy OOM on gen_cfg
            import intervention_generate as _etg
            _etg._ATTN_MODULES = args.attn_modules
            _etg._ATTN_BIAS_MODULES = bias_modules
            # If vision span is not detectable for this model/sample, disable gate intervention
            # to avoid inconsistent gate_on accounting.
            gen_cfg.use_gate_for_intervention = bool(
                getattr(args, "use_gate_for_intervention", False) and (vision_idx is not None)
            )
            gen_cfg.force_no_intervention = bool(getattr(args, "force_no_intervention", False))
            gen_cfg.attn_gate_tau = getattr(args, "attn_gate_tau", 0.2)
            gen_cfg.attn_gate_smooth_mode = getattr(args, "attn_gate_smooth_mode", "ema")
            gen_cfg.attn_gate_ema_alpha = getattr(args, "attn_gate_ema_alpha", 0.9)
            gen_cfg.attn_gate_window_size = getattr(args, "attn_gate_window_size", 8)
            gen_cfg.attn_gate_adaptive = bool(getattr(args, "attn_gate_adaptive", False))
            gen_cfg.attn_gate_target_rate = getattr(args, "attn_gate_target_rate", 0.7)
            gen_cfg.attn_gate_adapt_alpha = getattr(args, "attn_gate_adapt_alpha", 0.9)
            gen_cfg.attn_step_reduce_only = bool(getattr(args, "attn_step_reduce_only", True))
            gen_cfg.attn_keyword_bias = float(getattr(args, "attn_keyword_bias", 0.0) or 0.0)
            gen_cfg.attn_keyword_bias_max = float(getattr(args, "attn_keyword_bias_max", 4.0) or 4.0)
            gen_cfg.attn_keyword_positions = keyword_positions
            gen_cfg.bias_steps = str(getattr(args, "bias_steps", "all"))
            gen_cfg.bias_anneal = str(getattr(args, "bias_anneal", "hard"))
            gen_cfg.bias_apply_when = str(getattr(args, "bias_apply_when", "always"))
            gen_cfg.safe_decode_on_bias = bool(getattr(args, "safe_decode_on_bias", False))
            gen_cfg.collapse_risk_threshold = float(getattr(args, "collapse_risk_threshold", 0.6))
            gen_cfg.force_yesno = str(getattr(args, "force_yesno", "off"))
            gen_cfg.force_yesno_ids = getattr(args, "_force_yesno_ids", None)
            gen_cfg.subspace_shrink_enable = bool(getattr(args, "subspace_shrink_enable", False))
            gen_cfg.subspace_shrink_apply = bool(getattr(args, "_subspace_apply_for_sample", False))
            gen_cfg.subspace_shrink_alpha = float(getattr(args, "subspace_shrink_alpha", 0.3))
            gen_cfg.subspace_risk_score = getattr(args, "_subspace_risk_value_for_sample", None)
            gen_cfg.subspace_risk_threshold = getattr(args, "_subspace_risk_threshold", None)
            # Pass tensors via module-level global to avoid deepcopy OOM on gen_cfg
            import intervention_generate as _etg
            _etg._SUBSPACE_TENSORS = {
                "mu": getattr(args, "_subspace_mu", None),
                "uk": getattr(args, "_subspace_uk", None),
                "basis_list": getattr(args, "_subspace_basis_list", None),
            }
            gen_cfg.subspace_online_risk = bool(getattr(args, "subspace_online_risk", False))
            gen_cfg.subspace_intervention_layer = int(getattr(args, "subspace_intervention_layer", 20))
            gen_cfg.subspace_tau = float(getattr(args, "subspace_tau", 0.5))
            gen_cfg.subspace_tau_max = float(getattr(args, "subspace_tau_max", 0.7))
            gen_cfg.subspace_lambda_max = float(getattr(args, "subspace_lambda", 0.5))
            if "instructblip" in model_id_l and (task_mode or "") == "yesno":
                # InstructBLIP on yes/no tends to collapse under sampling in this path.
                # Keep deterministic decoding unless user explicitly enables sampling.
                do_sample = bool(getattr(args, "gen_temp_do_sample", False))
            elif getattr(args, "no_gen_temp_do_sample", False):
                do_sample = False
            elif getattr(args, "gen_temp_do_sample", False):
                do_sample = True
            else:
                do_sample = True
            outputs = model.generate(
                **inputs,
                generation_config=gen_cfg,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                output_attentions=need_attn_outputs,
                output_scores=need_score_outputs,
                return_dict_in_generate=True,
                custom_generate=entropy_temp_generate,
            )
            if isinstance(outputs, torch.Tensor):
                outputs = type("GenOut", (), {"sequences": outputs, "scores": None, "attentions": None})()
        else:
            if getattr(args, "gen_do_sample", False):
                do_sample = True
            elif getattr(args, "no_gen_do_sample", False):
                do_sample = False
            else:
                do_sample = False
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=getattr(args, "gen_temperature", 1.0),
                output_attentions=need_attn_outputs,
                output_scores=need_score_outputs,
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
        all_h_attn_vals: List[float] = []
        all_h_text_vals: List[float] = []
        all_h_attn_text_vals: List[float] = []
        all_h_attn_vision_vals: List[float] = []
        all_m_text_vals: List[float] = []
        all_m_vision_vals: List[float] = []
        all_attn_max_vals: List[float] = []
        all_obj_attn_mass_vals: List[float] = []
        all_obj_attn_peak_vals: List[float] = []
        all_s_last_vals: List[float] = []

        scores = outputs.scores or []
        top1_logit = None
        top2_logit = None
        top12_margin = None
        top1_prob = None
        top2_prob = None
        top12_prob_margin = None
        top12_prob_ratio = None
        top1_id = None
        top2_id = None
        top1_token = None
        top2_token = None
        if bool(getattr(args, "logit_margin_top12", False) or getattr(args, "risk_2pass", False)) and len(scores) > 0:
            try:
                s0 = scores[0]
                if s0 is not None and s0.shape[-1] >= 2:
                    probs = torch.softmax(s0[0].float(), dim=-1)
                    vals, ids = torch.topk(s0[0], k=2)
                    top1_logit = float(vals[0].item())
                    top2_logit = float(vals[1].item())
                    top12_margin = float(top1_logit - top2_logit)
                    top1_id = int(ids[0].item())
                    top2_id = int(ids[1].item())
                    top1_prob = float(probs[top1_id].item())
                    top2_prob = float(probs[top2_id].item())
                    top12_prob_margin = float(top1_prob - top2_prob)
                    top12_prob_ratio = float(top1_prob / max(top2_prob, 1e-12))
                    try:
                        top1_token = tokenizer.decode([top1_id], skip_special_tokens=False)
                    except Exception:
                        top1_token = None
                    try:
                        top2_token = tokenizer.decode([top2_id], skip_special_tokens=False)
                    except Exception:
                        top2_token = None
            except Exception:
                pass
        # Extract yes/no logits for VCD
        yesno_logit_yes = None
        yesno_logit_no = None
        if len(scores) > 0:
            try:
                s0 = scores[0]
                if s0 is not None:
                    vcd_yesno = getattr(args, "_vcd_yesno_ids", None)
                    if vcd_yesno:
                        yes_ids_list, no_ids_list = vcd_yesno
                        logits_0 = s0[0].float()
                        if yes_ids_list:
                            yesno_logit_yes = float(max(logits_0[i].item() for i in yes_ids_list))
                        if no_ids_list:
                            yesno_logit_no = float(max(logits_0[i].item() for i in no_ids_list))
            except Exception:
                pass
        attn_steps = outputs.attentions or []
        gate_cases = getattr(outputs, "gate_cases", None)
        step_task_types = getattr(outputs, "step_task_types", None)
        step_override_applied = getattr(outputs, "step_override_applied", None)
        smoother = AttnGateSmoother(
            AttnGateSmootherConfig(
                mode=getattr(args, "attn_gate_smooth_mode", "ema"),
                ema_alpha=float(getattr(args, "attn_gate_ema_alpha", 0.9)),
                window_size=int(getattr(args, "attn_gate_window_size", 8)),
            )
        )
        telemetry = getattr(outputs, "step_telemetry", None)
        if telemetry:
            for step_idx, row in enumerate(telemetry):
                h_t = float(row.get("Ht", 0.0) or 0.0)
                cross_ent = float(row.get("Ha", 0.0) or 0.0)
                h_attn_text = row.get("H_attn_text")
                h_attn_vision = row.get("H_attn_vision")
                m_text = row.get("m_text")
                m_vision = row.get("m_vision")
                attn_max = row.get("attn_max")
                obj_attn_mass = row.get("object_attn_mass")
                obj_attn_peak = row.get("object_attn_peak")
                s_last = row.get("S_last")
                gate_on = bool(row.get("gate_on", False))
                s_t = row.get("S_t")
                s_bar = row.get("S_bar")
                if s_t is None:
                    s_t = 0.0
                if s_bar is None:
                    s_bar = 0.0
                all_h_attn_vals.append(cross_ent)
                all_h_text_vals.append(h_t)
                if h_attn_text is not None:
                    all_h_attn_text_vals.append(float(h_attn_text))
                if h_attn_vision is not None:
                    all_h_attn_vision_vals.append(float(h_attn_vision))
                if m_text is not None:
                    all_m_text_vals.append(float(m_text))
                if m_vision is not None:
                    all_m_vision_vals.append(float(m_vision))
                if attn_max is not None:
                    all_attn_max_vals.append(float(attn_max))
                if obj_attn_mass is not None:
                    all_obj_attn_mass_vals.append(float(obj_attn_mass))
                if obj_attn_peak is not None:
                    all_obj_attn_peak_vals.append(float(obj_attn_peak))
                if s_last is not None:
                    all_s_last_vals.append(float(s_last))
                if gate_on:
                    gate_triggered = True
                    gate_steps += 1
                    h_attn_vals.append(cross_ent)
                    h_text_vals.append(h_t)
                    if float(s_t) > max_s_t:
                        max_s_t = float(s_t)
                record = {
                    "sample_id": sample_id,
                    "step": step_idx,
                    "gate_on": gate_on,
                    "S_t": s_t,
                    "S_bar": s_bar,
                    "H_t": h_t,
                    "H_attn": cross_ent,
                    "Ht_hat": row.get("Ht_hat"),
                    "Ha_hat": row.get("Ha_hat"),
                    "t_text_applied": row.get("t_text_applied"),
                    "t_attn_applied": row.get("t_attn_applied"),
                    "gate_case": row.get("gate_case"),
                    "case_id": row.get("gate_case"),
                    "task_type": row.get("task_type"),
                    "override_applied": row.get("override_applied"),
                    "force_no_intervention": row.get("force_no_intervention"),
                    "Ha_last_hat": row.get("Ha_last_hat"),
                    "S_last": row.get("S_last"),
                    "H_attn_text": h_attn_text,
                    "H_attn_vision": h_attn_vision,
                    "m_text": m_text,
                    "m_vision": m_vision,
                    "ratio_TV": row.get("ratio_TV"),
                    "delta_TV": row.get("delta_TV"),
                    "attn_source": row.get("attn_source"),
                    "kv_len": row.get("kv_len"),
                    "n_vision_tokens": row.get("n_vision_tokens"),
                    "attn_max": attn_max,
                    "object_attn_mass": obj_attn_mass,
                    "object_attn_peak": obj_attn_peak,
                    "object_attn_n_tokens": row.get("object_attn_n_tokens"),
                    "subspace_applied": row.get("subspace_applied"),
                    "subspace_risk_score": row.get("subspace_risk_score"),
                    "subspace_prefill_risk": row.get("subspace_prefill_risk"),
                    "subspace_risk_threshold": row.get("subspace_risk_threshold"),
                    "subspace_alpha": row.get("subspace_alpha"),
                    "subspace_residual_before": row.get("subspace_residual_before"),
                    "subspace_residual_after": row.get("subspace_residual_after"),
                    "subspace_n_bases": row.get("subspace_n_bases"),
                }
                if step_logger is not None:
                    step_logger.write(json.dumps(record) + "\n")
                if getattr(args, "attn_gate_step_stdout", False):
                    print(json.dumps(record))
        else:
            for step_idx in range(min(len(scores), len(attn_steps))):
                logits = scores[step_idx]
                step_attn = attn_steps[step_idx][-1] if attn_steps[step_idx] else None
                if step_attn is None:
                    continue
                h_t = _entropy_from_logits(logits)
                cross_ent = _attn_entropy(step_attn, vision_idx)
                all_h_attn_vals.append(cross_ent)
                all_h_text_vals.append(h_t)
                gate_on = False
                s_t = 0.0
                s_bar = 0.0
                if not (
                    getattr(args, "mode", "") == "hf_attn_gate_generate_temp"
                    and not getattr(args, "enable_generate_temp_gate_stats", True)
                ):
                    if vision_idx is not None:
                        s_t = _attention_gate_score_from_attn(step_attn, vision_idx)
                        s_bar = smoother.update(torch.tensor([float(s_t)], device=logits.device))[0].item()
                        gate_on = s_bar >= args.attn_gate_tau
                    else:
                        s_t = None
                        s_bar = None
                        gate_on = False
                if gate_on:
                    gate_triggered = True
                    gate_steps += 1
                    h_attn_vals.append(cross_ent)
                    h_text_vals.append(h_t)
                    if s_t > max_s_t:
                        max_s_t = s_t

                record = {
                    "sample_id": sample_id,
                    "step": step_idx,
                    "gate_on": gate_on,
                    "S_t": s_t,
                    "S_bar": s_bar,
                    "H_t": h_t,
                    "H_attn": cross_ent,
                    "gate_case": gate_cases[step_idx] if gate_cases is not None and step_idx < len(gate_cases) else None,
                    "case_id": gate_cases[step_idx] if gate_cases is not None and step_idx < len(gate_cases) else None,
                    "task_type": step_task_types[step_idx]
                    if step_task_types is not None and step_idx < len(step_task_types)
                    else None,
                    "override_applied": step_override_applied[step_idx]
                    if step_override_applied is not None and step_idx < len(step_override_applied)
                    else None,
                    "H_attn_text": None,
                    "H_attn_vision": None,
                    "m_text": None,
                    "m_vision": None,
                    "ratio_TV": None,
                    "delta_TV": None,
                    "attn_source": None,
                    "kv_len": None,
                    "n_vision_tokens": None,
                    "Ha_last_hat": None,
                    "S_last": None,
                    "attn_max": None,
                    "object_attn_mass": None,
                    "object_attn_peak": None,
                    "object_attn_n_tokens": 0,
                    "subspace_applied": None,
                    "subspace_risk_score": None,
                    "subspace_prefill_risk": None,
                    "subspace_risk_threshold": None,
                    "subspace_alpha": None,
                    "subspace_residual_before": None,
                    "subspace_residual_after": None,
                    "subspace_n_bases": None,
                }
                if step_logger is not None:
                    step_logger.write(json.dumps(record) + "\n")
                if getattr(args, "attn_gate_step_stdout", False):
                    print(json.dumps(record))

        if getattr(args, "mode", "") == "hf_attn_gate_generate_temp" and hasattr(outputs, "gate_on_count"):
            gate_steps = int(outputs.gate_on_count)
            if gate_steps > 0:
                h_attn_mean = float(outputs.gate_on_h_attn_sum) / gate_steps
                h_text_mean = float(outputs.gate_on_h_t_sum) / gate_steps
            else:
                h_attn_mean = 0.0
                h_text_mean = 0.0
            gate_triggered = gate_steps > 0
            gate_case_counts = getattr(outputs, "gate_case_counts", None)
        else:
            if gate_steps:
                h_attn_mean = float(sum(h_attn_vals) / gate_steps)
                h_text_mean = float(sum(h_text_vals) / gate_steps)
            else:
                h_attn_mean = 0.0
                h_text_mean = 0.0
            gate_case_counts = None
        all_h_attn_mean = float(sum(all_h_attn_vals) / max(1, len(all_h_attn_vals)))
        all_h_text_mean = float(sum(all_h_text_vals) / max(1, len(all_h_text_vals)))
        all_h_attn_text_mean = (
            float(sum(all_h_attn_text_vals) / len(all_h_attn_text_vals)) if all_h_attn_text_vals else None
        )
        all_h_attn_vision_mean = (
            float(sum(all_h_attn_vision_vals) / len(all_h_attn_vision_vals)) if all_h_attn_vision_vals else None
        )
        all_m_text_mean = float(sum(all_m_text_vals) / len(all_m_text_vals)) if all_m_text_vals else None
        all_m_vision_mean = float(sum(all_m_vision_vals) / len(all_m_vision_vals)) if all_m_vision_vals else None
        all_attn_max_mean = float(sum(all_attn_max_vals) / len(all_attn_max_vals)) if all_attn_max_vals else None
        all_obj_attn_mean = (
            float(sum(all_obj_attn_mass_vals) / len(all_obj_attn_mass_vals)) if all_obj_attn_mass_vals else None
        )
        all_obj_attn_peak_mean = (
            float(sum(all_obj_attn_peak_vals) / len(all_obj_attn_peak_vals)) if all_obj_attn_peak_vals else None
        )
        all_obj_attn_max = max(all_obj_attn_mass_vals) if all_obj_attn_mass_vals else None
        all_obj_attn_last = all_obj_attn_mass_vals[-1] if all_obj_attn_mass_vals else None
        s_top_first = all_s_last_vals[0] if all_s_last_vals else None
        s_top_mean = float(sum(all_s_last_vals) / len(all_s_last_vals)) if all_s_last_vals else None

        stats = {
            "total_tokens": float(gen_ids.numel()),
            "gate_on_count": float(gate_steps),
            "gate_on_h_attn_mean": h_attn_mean,
            "gate_on_h_t_mean": h_text_mean,
            "all_steps_h_attn_mean": all_h_attn_mean,
            "all_steps_h_t_mean": all_h_text_mean,
            "all_steps_h_attn_text_mean": all_h_attn_text_mean,
            "all_steps_h_attn_vision_mean": all_h_attn_vision_mean,
            "all_steps_m_text_mean": all_m_text_mean,
            "all_steps_m_vision_mean": all_m_vision_mean,
            "all_steps_attn_max_mean": all_attn_max_mean,
            "object_attn_mean": all_obj_attn_mean,
            "object_attn_peak_mean": all_obj_attn_peak_mean,
            "object_attn_max": all_obj_attn_max,
            "object_attn_last": all_obj_attn_last,
            "s_top_first": s_top_first,
            "s_top_mean": s_top_mean,
            "gate_case_counts": gate_case_counts,
            "task_type": getattr(outputs, "task_type", None),
            "override_ever": getattr(outputs, "override_ever", None),
            "n_steps_Ha_high": getattr(outputs, "n_steps_Ha_high", None),
            "n_steps_override": getattr(outputs, "n_steps_override", None),
            "first_step_override": getattr(outputs, "first_step_override", None),
            "force_no_intervention": getattr(outputs, "force_no_intervention", None),
            "subspace_shrink_applied": getattr(outputs, "subspace_shrink_applied", None),
            "subspace_online_mode": getattr(outputs, "subspace_online_mode", None),
            "subspace_risk_score": getattr(outputs, "subspace_risk_score", None),
            "subspace_prefill_risk": getattr(outputs, "subspace_prefill_risk", None),
            "subspace_risk_threshold": getattr(outputs, "subspace_risk_threshold", None),
            "subspace_shrink_alpha": getattr(outputs, "subspace_shrink_alpha", None),
            "subspace_lambda_max": getattr(outputs, "subspace_lambda_max", None),
            "subspace_tau": getattr(outputs, "subspace_tau", None),
            "subspace_tau_max": getattr(outputs, "subspace_tau_max", None),
            "subspace_intervention_layer": getattr(outputs, "subspace_intervention_layer", None),
            "subspace_n_bases": getattr(outputs, "subspace_n_bases", None),
            "subspace_residual_before_mean": getattr(outputs, "subspace_residual_before_mean", None),
            "subspace_residual_after_mean": getattr(outputs, "subspace_residual_after_mean", None),
            "timing_forward_s": getattr(outputs, "timing", {}).get("forward_s") if hasattr(outputs, "timing") else None,
            "timing_entropy_s": getattr(outputs, "timing", {}).get("entropy_s") if hasattr(outputs, "timing") else None,
            "timing_sampling_s": getattr(outputs, "timing", {}).get("sampling_s") if hasattr(outputs, "timing") else None,
            "timing_total_s": getattr(outputs, "timing", {}).get("total_s") if hasattr(outputs, "timing") else None,
            "first_tokens_ids": getattr(outputs, "first_tokens_ids", None),
            "first_tokens_str": getattr(outputs, "first_tokens_str", None),
            "eos_at_step1": getattr(outputs, "eos_at_step1", None),
            "digit_rate": getattr(outputs, "digit_rate", None),
            "prefix_rate": getattr(outputs, "prefix_rate", None),
            "empty_pred": getattr(outputs, "empty_pred", None),
            "avg_gen_len": getattr(outputs, "avg_gen_len", None),
            "gate_on_ratio": getattr(outputs, "gate_on_ratio", None),
            "attn_entropy_min": getattr(outputs, "attn_entropy_min", None),
            "attn_entropy_mean": getattr(outputs, "attn_entropy_mean", None),
            "attn_entropy_max": getattr(outputs, "attn_entropy_max", None),
            "requested_keyword": requested_keyword,
            "applied_keyword": applied_keyword,
            "keyword_bias": keyword_bias,
            "top1_logit": top1_logit,
            "top2_logit": top2_logit,
            "top12_margin": top12_margin,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "top12_prob_margin": top12_prob_margin,
            "top12_prob_ratio": top12_prob_ratio,
            "top1_id": top1_id,
            "top2_id": top2_id,
            "top1_token": top1_token,
            "top2_token": top2_token,
            "yesno_logit_yes": yesno_logit_yes,
            "yesno_logit_no": yesno_logit_no,
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
    smoother = AttnGateSmoother(
        AttnGateSmootherConfig(
            mode=getattr(args, "attn_gate_smooth_mode", "ema"),
            ema_alpha=float(getattr(args, "attn_gate_ema_alpha", 0.9)),
            window_size=int(getattr(args, "attn_gate_window_size", 8)),
        )
    )
    gate_tau = float(getattr(args, "attn_gate_tau", 0.2))
    gate_target = float(getattr(args, "attn_gate_target_rate", 0.7))
    gate_adapt_alpha = float(getattr(args, "attn_gate_adapt_alpha", 0.9))
    gate_rate_ema = 0.0
    gate_adaptive = bool(getattr(args, "attn_gate_adaptive", False))
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
            ent_base = _attn_entropy(outputs.attentions[-1], vision_idx)
            set_attn_temperature(args.attn_modules, args.t_attn_low)
            outputs_low = model(**model_inputs)
            ent_low = _attn_entropy(outputs_low.attentions[-1], vision_idx) if outputs_low.attentions else None
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
        if vision_idx is not None:
            s_t = _attention_gate_score(outputs.attentions, vision_idx)
        else:
            s_t = None
        h_t = _entropy_from_logits(logits)
        cross_ent = _attn_entropy(outputs.attentions[-1], vision_idx) if outputs.attentions else 0.0
        if ht_ema is None:
            ht_ema = h_t
            ha_ema = cross_ent
        else:
            ht_ema = decode_cfg.ema_alpha * ht_ema + (1.0 - decode_cfg.ema_alpha) * h_t
            ha_ema = decode_cfg.ema_alpha * ha_ema + (1.0 - decode_cfg.ema_alpha) * cross_ent
        ht_hist.append(ht_ema)
        ha_hist.append(ha_ema)
        if s_t is not None:
            s_bar = smoother.update(torch.tensor([float(s_t)], device=logits.device))[0].item()
            gate_on = s_bar >= gate_tau
        else:
            s_bar = None
            gate_on = False
        if gate_adaptive:
            gate_rate_ema = gate_adapt_alpha * gate_rate_ema + (1.0 - gate_adapt_alpha) * float(gate_on)
            # Simple proportional adjustment
            gate_tau -= 0.05 * (gate_target - gate_rate_ema)
            gate_tau = max(0.0, min(1.0, gate_tau))
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
            "S_bar": s_bar,
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
    # Prefer first non-empty line / first token for single-word answers
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        m0 = re.match(r"^(yes|no)\b", first.strip(), re.IGNORECASE)
        if m0:
            return m0.group(1).lower()
        # Accept numeric yes/no (common in LLaVA)
        if first in {"0", "1"}:
            return "yes" if first == "1" else "no"
    # Fallback: single-token or short answers like "Yes"/"No"
    m3 = re.search(r"\b(yes|no)\b", text.strip(), re.IGNORECASE)
    return m3.group(1).lower() if m3 else ""


def _build_yesno_token_ids(tokenizer) -> List[int]:
    """Build a conservative single-token allowlist for yes/no masking."""
    allowed = set()
    variants = [
        "yes", "no", " yes", " no", "Yes", "No", " YES", " NO",
        "yes.", "no.", " yes.", " no.", "Yes.", "No.",
    ]
    for s in variants:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) == 1:
            allowed.add(int(ids[0]))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size > 0:
        for tid in range(vocab_size):
            try:
                tok = tokenizer.decode([tid], skip_special_tokens=False).strip().lower()
            except Exception:
                continue
            if tok in {"yes", "no", "yes.", "no."}:
                allowed.add(int(tid))
    return sorted(allowed)


def _build_yesno_token_id_groups(tokenizer) -> Dict[str, List[int]]:
    """Build yes/no single-token id groups for margin logging."""
    groups = {"yes": set(), "no": set()}
    yes_variants = ["yes", " yes", "Yes", " YES", "yes.", " yes.", "Yes."]
    no_variants = ["no", " no", "No", " NO", "no.", " no.", "No."]
    for s in yes_variants:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) == 1:
            groups["yes"].add(int(ids[0]))
    for s in no_variants:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) == 1:
            groups["no"].add(int(ids[0]))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size > 0:
        for tid in range(vocab_size):
            try:
                tok = tokenizer.decode([tid], skip_special_tokens=False).strip().lower()
            except Exception:
                continue
            if tok in {"yes", "yes."}:
                groups["yes"].add(int(tid))
            elif tok in {"no", "no."}:
                groups["no"].add(int(tid))
    return {"yes": sorted(groups["yes"]), "no": sorted(groups["no"])}


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


def _load_sample_id_subset(path: str) -> Optional[set[int]]:
    p = (path or "").strip()
    if not p:
        return None
    fpath = Path(p)
    if not fpath.exists():
        LOGGER.warning("sample-id-file not found: %s", p)
        return None
    out: set[int] = set()
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.add(int(s))
            except Exception:
                continue
    LOGGER.info("Loaded sample-id subset: %d ids from %s", len(out), p)
    return out


def _prefix_collapse_token(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "<empty>"
    s = s.replace("\n", " ")
    return s[:16]


def _run_layerwise_object_attn_analysis(
    dataset: Iterable[Dict[str, Any]],
    args,
    model,
    processor,
    tokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    results_path = str(getattr(args, "layerwise_results_jsonl", "") or "").strip()
    if not results_path:
        raise ValueError("--layerwise-results-jsonl is required with --layerwise-object-attn")
    p = Path(results_path)
    if not p.exists():
        raise FileNotFoundError(f"layerwise results jsonl not found: {results_path}")

    correct_map: Dict[int, int] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("summary"):
                continue
            sid = o.get("sample_id")
            if sid is None:
                continue
            correct_map[int(sid)] = 1 if float(o.get("accuracy", 0.0)) >= 0.5 else 0
    if not correct_map:
        raise RuntimeError("No sample_id/accuracy found in layerwise-results-jsonl.")

    rows: List[Dict[str, Any]] = []
    n_skipped_no_kw = 0
    n_skipped_no_attn = 0
    n_seen = 0
    max_new_tokens = int(getattr(args, "layerwise_max_new_tokens", 16) or 16)
    total_target = int(len(correct_map))
    progress_every = 25
    LOGGER.info(
        "Layerwise object-attn analysis start | target_samples=%d | max_new_tokens=%d",
        total_target,
        max_new_tokens,
    )

    for idx, example in enumerate(dataset):
        if idx not in correct_map:
            continue
        if args.limit >= 0 and n_seen >= int(args.limit):
            break

        image = example.get("image")
        question = example.get("question", "")
        meta = {k: example.get(k) for k in example.keys() if k not in {"image", "question", "answers"}}
        meta["dataset_id"] = args.dataset_id
        prompt = _build_hf_prompt(question, meta.get("task_type", "yesno"), meta, processor, args.model_id)

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        keyword = _extract_object_from_question(question)
        key_pos = _keyword_token_positions(inputs["input_ids"], tokenizer, keyword) if keyword else []
        if not key_pos:
            n_skipped_no_kw += 1
            n_seen += 1
            if (n_seen % progress_every == 0) or (n_seen == total_target):
                LOGGER.info(
                    "Layerwise progress %d/%d | sample_id=%d | rows=%d | skip_no_kw=%d | skip_no_attn=%d",
                    n_seen,
                    total_target,
                    idx,
                    len(rows),
                    n_skipped_no_kw,
                    n_skipped_no_attn,
                )
            continue

        # Capture pre-softmax attention logits per module/step (reduced to head-mean row).
        attn_modules = list(getattr(args, "attn_modules", []) or [])
        if not attn_modules:
            n_skipped_no_attn += 1
            n_seen += 1
            if (n_seen % progress_every == 0) or (n_seen == total_target):
                LOGGER.info(
                    "Layerwise progress %d/%d | sample_id=%d | rows=%d | skip_no_kw=%d | skip_no_attn=%d",
                    n_seen,
                    total_target,
                    idx,
                    len(rows),
                    n_skipped_no_kw,
                    n_skipped_no_attn,
                )
            continue
        set_attn_capture_pre_softmax(attn_modules, enabled=True, keep_steps=True)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_attentions=False,
                return_dict_in_generate=True,
            )
        _ = out  # not used directly; logits are captured via patched attention modules.

        has_any_hist = any(bool(getattr(m, "attn_pre_softmax_hist", [])) for m in attn_modules)
        if not has_any_hist:
            n_skipped_no_attn += 1
            n_seen += 1
            set_attn_capture_pre_softmax(attn_modules, enabled=False, keep_steps=False)
            if (n_seen % progress_every == 0) or (n_seen == total_target):
                LOGGER.info(
                    "Layerwise progress %d/%d | sample_id=%d | rows=%d | skip_no_kw=%d | skip_no_attn=%d",
                    n_seen,
                    total_target,
                    idx,
                    len(rows),
                    n_skipped_no_kw,
                    n_skipped_no_attn,
                )
            continue

        # Use first generated step for consistency with previous probability-based analysis.
        for li, module in enumerate(attn_modules):
            hist = getattr(module, "attn_pre_softmax_hist", None) or []
            if len(hist) == 0:
                continue
            vec = hist[0][0].float()  # (K,) from first decode step, batch=1
            valid = [p for p in key_pos if 0 <= int(p) < int(vec.numel())]
            if not valid:
                continue
            idx_t = torch.tensor(valid, device=vec.device, dtype=torch.long)
            picked = torch.index_select(vec, 0, idx_t)
            logit_mean = float(picked.mean().item())
            logit_max = float(picked.max().item())
            rows.append(
                {
                    "sample_id": idx,
                    "correct": int(correct_map[idx]),
                    "layer": int(li),
                    "keyword": keyword,
                    "n_keyword_tokens": int(len(valid)),
                    "object_attn_logit_mean": logit_mean,
                    "object_attn_logit_max": logit_max,
                }
            )
        set_attn_capture_pre_softmax(attn_modules, enabled=False, keep_steps=False)

        n_seen += 1
        if (n_seen % progress_every == 0) or (n_seen == total_target):
            LOGGER.info(
                "Layerwise progress %d/%d | sample_id=%d | rows=%d | skip_no_kw=%d | skip_no_attn=%d",
                n_seen,
                total_target,
                idx,
                len(rows),
                n_skipped_no_kw,
                n_skipped_no_attn,
            )
        if getattr(args, "clear_cache_every", 0) and n_seen % int(args.clear_cache_every) == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    outdir = Path(str(getattr(args, "layerwise_outdir", "analysis_layerwise_object_attn")))
    outdir.mkdir(parents=True, exist_ok=True)

    per_path = outdir / "per_sample_layer_object_attn.csv"
    if rows:
        with per_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "correct",
                    "layer",
                    "keyword",
                    "n_keyword_tokens",
                    "object_attn_logit_mean",
                    "object_attn_logit_max",
                ],
            )
            w.writeheader()
            w.writerows(rows)

    agg: Dict[Tuple[int, int], Dict[str, float]] = {}
    for r in rows:
        k = (int(r["layer"]), int(r["correct"]))
        s = agg.setdefault(k, {"n": 0.0, "logit_mean_sum": 0.0, "logit_max_sum": 0.0})
        s["n"] += 1.0
        s["logit_mean_sum"] += float(r["object_attn_logit_mean"])
        s["logit_max_sum"] += float(r["object_attn_logit_max"])

    summary_rows = []
    for (layer, correct), s in sorted(agg.items()):
        n = max(1.0, s["n"])
        summary_rows.append(
            {
                "layer": layer,
                "correct": correct,
                "n": int(s["n"]),
                "object_attn_logit_mean_avg": s["logit_mean_sum"] / n,
                "object_attn_logit_max_avg": s["logit_max_sum"] / n,
            }
        )
    with (outdir / "layer_summary_correct_vs_wrong.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["layer", "correct", "n", "object_attn_logit_mean_avg", "object_attn_logit_max_avg"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    by_layer: Dict[int, Dict[int, Dict[str, float]]] = {}
    for r in summary_rows:
        by_layer.setdefault(int(r["layer"]), {})[int(r["correct"])] = r
    diff_rows = []
    for layer in sorted(by_layer.keys()):
        c0 = by_layer[layer].get(0, {})
        c1 = by_layer[layer].get(1, {})
        m0 = c0.get("object_attn_logit_mean_avg")
        m1 = c1.get("object_attn_logit_mean_avg")
        p0 = c0.get("object_attn_logit_max_avg")
        p1 = c1.get("object_attn_logit_max_avg")
        diff_rows.append(
            {
                "layer": layer,
                "logit_mean_correct_0": m0,
                "logit_mean_correct_1": m1,
                "logit_max_correct_0": p0,
                "logit_max_correct_1": p1,
                "logit_mean_diff_correct_minus_wrong": (None if (m0 is None or m1 is None) else (m1 - m0)),
                "logit_max_diff_correct_minus_wrong": (None if (p0 is None or p1 is None) else (p1 - p0)),
            }
        )
    with (outdir / "layer_diff_correct_minus_wrong.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "logit_mean_correct_0",
                "logit_mean_correct_1",
                "logit_max_correct_0",
                "logit_max_correct_1",
                "logit_mean_diff_correct_minus_wrong",
                "logit_max_diff_correct_minus_wrong",
            ],
        )
        w.writeheader()
        w.writerows(diff_rows)

    meta = {
        "n_labeled_samples": len(correct_map),
        "n_seen": n_seen,
        "n_rows": len(rows),
        "n_unique_samples_used": len(set(r["sample_id"] for r in rows)) if rows else 0,
        "n_skipped_no_keyword": n_skipped_no_kw,
        "n_skipped_no_attn": n_skipped_no_attn,
        "outdir": str(outdir),
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info(
        "Layerwise object-attn analysis done | seen=%d/%d | rows=%d | skip_no_kw=%d | skip_no_attn=%d | outdir=%s",
        n_seen,
        total_target,
        len(rows),
        n_skipped_no_kw,
        n_skipped_no_attn,
        str(outdir),
    )
    return {
        "mode": "hf_attn_gate_layerwise_object_attn",
        "overall_accuracy": 0.0,
        "num_samples": int(meta["n_unique_samples_used"]),
        "outdir": str(outdir),
    }


def run_hf_attn_gate(
    dataset: Iterable[Dict[str, Any]],
    args,
    output_jsonl: str,
) -> Dict[str, Any]:
    """Run HF attention-gated decoding with Transformers."""
    if getattr(args, "gpu_ids", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    baseline_only = getattr(args, "mode", "") == "hf_attn_gate_baseline"
    generate_temp_only = getattr(args, "mode", "") == "hf_attn_gate_generate_temp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_id_l = args.model_id.lower()
    if "llava" in model_id_l:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    elif "instructblip" in model_id_l:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    elif "qwen-vl" in model_id_l or "mplug-owl2" in model_id_l:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = args.max_new_tokens
    _force_eager_attention_impl(model)
    patched = patch_cross_attn_forward() if args.enable_attn_temp else False
    if args.enable_attn_temp and not patched:
        LOGGER.warning("Attention temperature patch failed for this model; disabling attn temp.")
        args.enable_attn_temp = False
    if args.enable_attn_temp and not patched and not (baseline_only or generate_temp_only):
        raise RuntimeError("Attention temperature patch failed; cannot modify attention values.")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True if ("qwen-vl" in model_id_l or "mplug-owl2" in model_id_l) else args.trust_remote_code,
    )
    tokenizer = processor.tokenizer
    if str(getattr(args, "force_yesno", "off")) == "mask_logits":
        try:
            args._force_yesno_ids = _build_yesno_token_ids(tokenizer)
            LOGGER.info("force_yesno=mask_logits | allowed_token_ids=%d", len(args._force_yesno_ids))
        except Exception as exc:
            LOGGER.warning("Failed to build yes/no allowlist: %s", exc)
            args._force_yesno_ids = []
    else:
        args._force_yesno_ids = []
    # VCD setup
    vcd_mode = str(getattr(args, "vcd_mode", "off") or "off")
    if vcd_mode != "off":
        args._vcd_yesno_ids = _find_yesno_token_ids(tokenizer)
        LOGGER.info("VCD mode=%s | yes_ids=%s, no_ids=%s", vcd_mode, args._vcd_yesno_ids[0], args._vcd_yesno_ids[1])
    else:
        args._vcd_yesno_ids = None
    if "llava" in args.model_id.lower():
        tokenizer.padding_side = "left"
    model.eval()
    args._subspace_mu = None
    args._subspace_uk = None
    args._subspace_basis_list = []
    args._subspace_risk_map = {}
    args._subspace_risk_threshold = None
    if bool(getattr(args, "subspace_shrink_enable", False)):
        basis_dir = str(getattr(args, "subspace_basis_dir", "") or "").strip()
        basis_layers_str = str(getattr(args, "subspace_basis_layers", "") or "").strip()
        if basis_layers_str:
            basis_layers = []
            for tok in basis_layers_str.split(","):
                tok = tok.strip()
                if tok:
                    basis_layers.append(int(tok))
            basis_layers = sorted(set(basis_layers))
        else:
            basis_layers = [int(getattr(args, "subspace_basis_layer", 25))]
        if not basis_layers:
            raise ValueError("no subspace basis layer specified")
        if not basis_dir:
            raise ValueError("--subspace-shrink-enable requires --subspace-basis-dir")
        basis_list = []
        for basis_layer in basis_layers:
            mu_path = Path(basis_dir) / f"mu_layer{basis_layer}.pt"
            uk_path = Path(basis_dir) / f"basis_Uk_layer{basis_layer}.pt"
            if not uk_path.exists():
                raise FileNotFoundError(f"subspace basis missing: {uk_path}")
            if not mu_path.exists():
                raise FileNotFoundError(f"subspace basis mu missing: {mu_path}")
            uk = torch.load(uk_path, map_location=device).float()
            mu = torch.load(mu_path, map_location=device).float()
            if uk.ndim != 2:
                raise ValueError(f"invalid subspace basis Uk shape at layer {basis_layer}")
            if mu.ndim != 1 or mu.shape[0] != uk.shape[0]:
                raise ValueError(f"mu/Uk dim mismatch at layer {basis_layer}")
            basis_list.append({"layer": int(basis_layer), "mu": mu, "uk": uk})
        args._subspace_basis_list = basis_list
        # Backward-compatible single-basis fields
        args._subspace_mu = basis_list[0].get("mu")
        args._subspace_uk = basis_list[0]["uk"]
        risk_csv = str(getattr(args, "subspace_risk_csv", "") or "").strip()
        risk_col = str(getattr(args, "subspace_risk_col", "r_rel_fused_mean"))
        risk_map = _load_subspace_risk_map(risk_csv, risk_col) if risk_csv else {}
        args._subspace_risk_map = risk_map
        thr = getattr(args, "subspace_risk_threshold", None)
        if thr is None and risk_map:
            thr = _quantile_threshold(list(risk_map.values()), float(getattr(args, "subspace_risk_top_pct", 0.2)))
        args._subspace_risk_threshold = (float(thr) if thr is not None else None)
        LOGGER.info(
            "Subspace attraction enabled | layers=%s alpha=%.4f basis_dim=%d k=%d risk_map=%d thr=%s",
            basis_layers,
            float(getattr(args, "subspace_shrink_alpha", 0.3)),
            int(args._subspace_uk.shape[0]),
            int(args._subspace_uk.shape[1]),
            len(risk_map),
            ("None" if args._subspace_risk_threshold is None else f"{args._subspace_risk_threshold:.6f}"),
        )
    args.attn_modules = find_cross_attn_modules(model)
    bias_target = str(getattr(args, "attn_keyword_bias_target", "all")).lower()
    if bias_target == "last" and args.attn_modules:
        args.attn_bias_modules = [args.attn_modules[-1]]
    else:
        args.attn_bias_modules = list(args.attn_modules)
    LOGGER.info(
        "Keyword-bias target=%s | bias_modules=%d/%d",
        bias_target,
        len(getattr(args, "attn_bias_modules", []) or []),
        len(getattr(args, "attn_modules", []) or []),
    )
    if args.enable_attn_temp and not args.attn_modules and not (baseline_only or generate_temp_only):
        raise RuntimeError("No attention modules found for temperature patching.")
    if bool(getattr(args, "layerwise_object_attn", False)):
        return _run_layerwise_object_attn_analysis(dataset, args, model, processor, tokenizer, device)

    total_acc = 0.0
    n = 0
    total_latency_s = 0.0
    sample_id_subset = _load_sample_id_subset(getattr(args, "sample_id_file", ""))
    empty_pred_count = 0
    eos_at_step1_count = 0
    digit_collapse_count = 0
    prefix_collapse_count = 0
    total_gen_len = 0.0
    gate_on_ratio_vals: List[float] = []
    prefix_hist: Dict[str, int] = {}

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
    margin_csv_path = str(getattr(args, "logit_margin_output_csv", "") or "").strip()
    margin_csv_fp = None
    margin_csv_writer = None
    if margin_csv_path:
        csv_mode = "a" if (resume_count > 0 and Path(margin_csv_path).exists()) else "w"
        margin_csv_fp = open(margin_csv_path, csv_mode, newline="", encoding="utf-8")
        margin_csv_writer = csv.DictWriter(
            margin_csv_fp,
            fieldnames=[
                "sample_id",
                "question",
                "prediction",
                "gt",
                "correct",
                "top1_logit",
                "top2_logit",
                "top12_margin",
                "top1_prob",
                "top2_prob",
                "top12_prob_margin",
                "top12_prob_ratio",
                "top1_id",
                "top2_id",
                "top1_token",
                "top2_token",
                "dataset_id",
                "model_id",
            ],
        )
        if csv_mode == "w":
            margin_csv_writer.writeheader()
    with open(output_jsonl, out_mode, encoding="utf-8", buffering=1) as f_out:
        for idx, example in enumerate(dataset):
            if sample_id_subset is not None and idx not in sample_id_subset:
                continue
            if idx < resume_count:
                continue
            if total_expected is not None and (n + resume_count) >= total_expected:
                break
            try:
                sample_t0 = time.perf_counter()
                image = example.get("image")
                question = example.get("question", "")
                gt_answers = example.get("answers", [])
                meta = {k: example.get(k) for k in example.keys() if k not in {"image", "question", "answers"}}
                meta["dataset_id"] = args.dataset_id
                meta["llava_cot"] = bool(getattr(args, "llava_cot", False))
                prompt = _build_hf_prompt(question, meta.get("task_type", "vqa"), meta, processor, args.model_id)
                # Sample-level gate for subspace intervention.
                online_risk = bool(getattr(args, "subspace_online_risk", False))
                if online_risk:
                    # Online mode: hook handles gating per-step; always enable.
                    apply_subspace = bool(getattr(args, "subspace_shrink_enable", False))
                    risk_val = None
                else:
                    # Legacy offline CSV mode.
                    risk_map = getattr(args, "_subspace_risk_map", {}) or {}
                    risk_val = risk_map.get(int(idx))
                    risk_thr = getattr(args, "_subspace_risk_threshold", None)
                    apply_subspace = bool(
                        getattr(args, "subspace_shrink_enable", False)
                        and (risk_val is not None)
                        and (risk_thr is not None)
                        and (float(risk_val) >= float(risk_thr))
                    )
                args._subspace_apply_for_sample = apply_subspace
                args._subspace_risk_value_for_sample = (None if risk_val is None else float(risk_val))
                with torch.no_grad():
                    ret = _decode_with_attn_gate(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        image=image,
                        args=args,
                        step_logger=step_log,
                        record_minimal=baseline_only or generate_temp_only,
                        model_id=args.model_id,
                        task_mode=meta.get("task_type", "vqa"),
                        question_text=question,
                        sample_id=idx,
                    )
                    if isinstance(ret, (list, tuple)):
                        if len(ret) >= 4:
                            text, gate_triggered, max_s_t, step_stats = ret[:4]
                        else:
                            raise ValueError(f"_decode_with_attn_gate returned {len(ret)} values, expected >=4")
                    else:
                        raise ValueError(f"_decode_with_attn_gate returned non-tuple: {type(ret)}")
                task_type = meta.get("task_type", "vqa")
                if task_type == "docvqa":
                    pred = extract_docvqa_answer(text)
                elif task_type == "yesno":
                    pred = _extract_yesno(text)
                else:
                    pred = extract_final_answer(text)

                # Optional 2-pass perturbation consistency + grounding risk diagnostics.
                risk_row = None
                if bool(getattr(args, "risk_2pass", False)):
                    image_p2 = _apply_visual_perturbation(
                        image,
                        kind=str(getattr(args, "risk_perturb", "blur")),
                        strength=float(getattr(args, "risk_perturb_strength", 1.0)),
                    )
                    with torch.no_grad():
                        ret2 = _decode_with_attn_gate(
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            image=image_p2,
                            args=args,
                            step_logger=None,
                            record_minimal=baseline_only or generate_temp_only,
                            model_id=args.model_id,
                            task_mode=meta.get("task_type", "vqa"),
                            question_text=question,
                            sample_id=idx,
                        )
                    if isinstance(ret2, (list, tuple)) and len(ret2) >= 4:
                        text2, _, _, step_stats2 = ret2[:4]
                    else:
                        text2, step_stats2 = "", {}
                    if task_type == "docvqa":
                        pred2 = extract_docvqa_answer(text2)
                    elif task_type == "yesno":
                        pred2 = _extract_yesno(text2)
                    else:
                        pred2 = extract_final_answer(text2)
                    margin1 = _extract_yesno_margin(step_stats)
                    margin2 = _extract_yesno_margin(step_stats2)
                    s_top = step_stats.get("s_top_first")
                    if s_top is None:
                        s_top = step_stats.get("s_top_mean")
                    disagree = int((pred or "") != (pred2 or ""))
                    tau_high = float(getattr(args, "risk_tau_high", 0.5))
                    tau_s = float(getattr(args, "risk_tau_s", 0.2))
                    stable_both = bool(getattr(args, "risk_stable_both_margins", False))
                    high_conf = False
                    if stable_both:
                        if margin1 is not None and margin2 is not None:
                            high_conf = (margin1 > tau_high) and (margin2 > tau_high)
                    else:
                        if margin1 is not None:
                            high_conf = margin1 > tau_high
                    low_ground = (s_top is not None) and (float(s_top) < tau_s)
                    gate_conf_ungrounded = int(bool(high_conf and low_ground))
                    risk_gate_on = int(bool(disagree or gate_conf_ungrounded))
                    if not bool(getattr(args, "risk_gate_enable", False)):
                        risk_gate_on = 0
                    risk_row = {
                        "sample_id": idx,
                        "task_type": task_type,
                        "pred_pass1": pred,
                        "pred_pass2": pred2,
                        "raw_pass2": text2,
                        "disagree": disagree,
                        "margin1": margin1,
                        "margin2": margin2,
                        "s_top": s_top,
                        "tau_high": tau_high,
                        "tau_s": tau_s,
                        "risk_conf_ungrounded": gate_conf_ungrounded,
                        "risk_gate_on": risk_gate_on,
                        "risk_perturb": str(getattr(args, "risk_perturb", "blur")),
                        "risk_perturb_strength": float(getattr(args, "risk_perturb_strength", 1.0)),
                    }
                # --- VCD (Visual Contrastive Decoding) ---
                vcd_row = None
                if vcd_mode != "off" and task_type == "yesno":
                    null_img = _create_null_image(image, kind=str(getattr(args, "vcd_null_image", "blank")))
                    with torch.no_grad():
                        ret_null = _decode_with_attn_gate(
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            image=null_img,
                            args=args,
                            step_logger=None,
                            record_minimal=True,
                            model_id=args.model_id,
                            task_mode="vqa",
                            question_text=question,
                            sample_id=idx,
                        )
                    if isinstance(ret_null, (list, tuple)) and len(ret_null) >= 4:
                        text_null, _, _, stats_null = ret_null[:4]
                    else:
                        text_null, stats_null = "", {}
                    pred_null = _extract_yesno(text_null)
                    ly_img = step_stats.get("yesno_logit_yes")
                    ln_img = step_stats.get("yesno_logit_no")
                    ly_null = stats_null.get("yesno_logit_yes")
                    ln_null = stats_null.get("yesno_logit_no")
                    vcd_alpha = float(getattr(args, "vcd_alpha", 1.0))
                    vcd_pred_contrastive = None
                    if ly_img is not None and ln_img is not None and ly_null is not None and ln_null is not None:
                        cy = ly_img - vcd_alpha * ly_null
                        cn = ln_img - vcd_alpha * ln_null
                        vcd_pred_contrastive = "yes" if cy > cn else "no"
                    vcd_disagree = int((pred or "") != (pred_null or ""))
                    vcd_row = {
                        "sample_id": idx,
                        "pred_img": pred,
                        "pred_null": pred_null,
                        "vcd_disagree": vcd_disagree,
                        "logit_yes_img": ly_img,
                        "logit_no_img": ln_img,
                        "logit_yes_null": ly_null,
                        "logit_no_null": ln_null,
                        "vcd_alpha": vcd_alpha,
                        "vcd_pred_contrastive": vcd_pred_contrastive,
                    }
                    if vcd_mode == "decode" and vcd_pred_contrastive is not None:
                        pred = vcd_pred_contrastive

                acc_input = pred if task_type in {"mcq", "mmbench", "yesno", "vizwiz"} else text
                acc = compute_accuracy(acc_input, gt_answers, task_type=task_type)
                sample_latency_s = time.perf_counter() - sample_t0
                first_token_ids = step_stats.get("first_tokens_ids") or []
                first_token_strs = []
                for tid in first_token_ids[:5]:
                    try:
                        first_token_strs.append(tokenizer.decode([int(tid)], skip_special_tokens=False))
                    except Exception:
                        first_token_strs.append("")
                text_stripped = (text or "").strip()
                empty_pred = int(len(pred.strip()) == 0 if isinstance(pred, str) else True)
                digit_rate = (sum(ch.isdigit() for ch in text_stripped) / max(1, len(text_stripped))) if text_stripped else 0.0
                prefix_rate = float(("Q:" in (text or "")) or ("A:" in (text or "")))
                eos_at_step1 = bool(step_stats.get("eos_at_step1", False))
                avg_gen_len = int(step_stats.get("avg_gen_len", step_stats.get("total_tokens", 0)) or 0)
                gate_on_ratio = step_stats.get("gate_on_ratio")
            except Exception as exc:
                LOGGER.exception("HF attn-gate example failed: %s", exc)
                continue

            record = {
                "question_id": meta.get("question_id"),
                "image_id": meta.get("image_id"),
                "sample_id": idx,
                "question": question,
                "gt_answers": gt_answers,
                "prediction": pred,
                "raw_text": text,
                "accuracy": acc,
                "temp_fixed": getattr(args, "temp_fixed", None),
                "du_t_text_base": getattr(args, "du_t_text_base", None),
                "gen_temperature": getattr(args, "gen_temperature", None),
                "gate_triggered": gate_triggered,
                "max_gate_score": max_s_t,
                "total_tokens": int(step_stats.get("total_tokens", 0)),
                "gate_on_count": int(step_stats.get("gate_on_count", 0)),
                "gate_on_h_attn_mean": float(step_stats.get("gate_on_h_attn_mean", 0.0)),
                "gate_on_h_t_mean": float(step_stats.get("gate_on_h_t_mean", 0.0)),
                "all_steps_h_attn_mean": float(step_stats.get("all_steps_h_attn_mean", 0.0)),
                "all_steps_h_t_mean": float(step_stats.get("all_steps_h_t_mean", 0.0)),
                "all_steps_h_attn_text_mean": step_stats.get("all_steps_h_attn_text_mean"),
                "all_steps_h_attn_vision_mean": step_stats.get("all_steps_h_attn_vision_mean"),
                "all_steps_m_text_mean": step_stats.get("all_steps_m_text_mean"),
                "all_steps_m_vision_mean": step_stats.get("all_steps_m_vision_mean"),
                "gate_case_counts": step_stats.get("gate_case_counts"),
                "detected_task_type": step_stats.get("task_type"),
                "override_ever": step_stats.get("override_ever"),
                "n_steps_Ha_high": step_stats.get("n_steps_Ha_high"),
                "n_steps_override": step_stats.get("n_steps_override"),
                "first_step_override": step_stats.get("first_step_override"),
                "latency_s": sample_latency_s,
                "timing_forward_s": step_stats.get("timing_forward_s"),
                "timing_entropy_s": step_stats.get("timing_entropy_s"),
                "timing_sampling_s": step_stats.get("timing_sampling_s"),
                "timing_total_s": step_stats.get("timing_total_s"),
                "phase_switch_step": step_stats.get("phase_switch_step"),
                "final_selected": step_stats.get("final_selected"),
                "ht_ema": step_stats.get("ht_ema"),
                "ha_ema": step_stats.get("ha_ema"),
                "rule_trigger": step_stats.get("rule_trigger"),
                "first_tokens_ids": first_token_ids,
                "first_tokens_str": first_token_strs,
                "eos_at_step1": eos_at_step1,
                "digit_rate": float(digit_rate),
                "prefix_rate": float(prefix_rate),
                "empty_pred": int(empty_pred),
                "avg_gen_len": int(avg_gen_len),
                "gate_on_ratio": gate_on_ratio,
                "attn_entropy_min": step_stats.get("attn_entropy_min"),
                "attn_entropy_mean": step_stats.get("attn_entropy_mean"),
                "attn_entropy_max": step_stats.get("attn_entropy_max"),
                "all_steps_attn_max_mean": step_stats.get("all_steps_attn_max_mean"),
                "object_attn_mean": step_stats.get("object_attn_mean"),
                "object_attn_peak_mean": step_stats.get("object_attn_peak_mean"),
                "object_attn_max": step_stats.get("object_attn_max"),
                "object_attn_last": step_stats.get("object_attn_last"),
                "requested_keyword": step_stats.get("requested_keyword"),
                "applied_keyword": step_stats.get("applied_keyword"),
                "keyword_bias": step_stats.get("keyword_bias"),
                "top1_logit": step_stats.get("top1_logit"),
                "top2_logit": step_stats.get("top2_logit"),
                "top12_margin": step_stats.get("top12_margin"),
                "top1_prob": step_stats.get("top1_prob"),
                "top2_prob": step_stats.get("top2_prob"),
                "top12_prob_margin": step_stats.get("top12_prob_margin"),
                "top12_prob_ratio": step_stats.get("top12_prob_ratio"),
                "top1_id": step_stats.get("top1_id"),
                "top2_id": step_stats.get("top2_id"),
                "top1_token": step_stats.get("top1_token"),
                "top2_token": step_stats.get("top2_token"),
                "risk_disagree": (None if risk_row is None else risk_row.get("disagree")),
                "risk_margin1": (None if risk_row is None else risk_row.get("margin1")),
                "risk_margin2": (None if risk_row is None else risk_row.get("margin2")),
                "risk_s_top": (None if risk_row is None else risk_row.get("s_top")),
                "risk_conf_ungrounded": (None if risk_row is None else risk_row.get("risk_conf_ungrounded")),
                "risk_gate_on": (None if risk_row is None else risk_row.get("risk_gate_on")),
                "risk_pred_pass2": (None if risk_row is None else risk_row.get("pred_pass2")),
                "vcd_disagree": (None if vcd_row is None else vcd_row.get("vcd_disagree")),
                "vcd_pred_null": (None if vcd_row is None else vcd_row.get("pred_null")),
                "vcd_logit_yes_img": (None if vcd_row is None else vcd_row.get("logit_yes_img")),
                "vcd_logit_no_img": (None if vcd_row is None else vcd_row.get("logit_no_img")),
                "vcd_logit_yes_null": (None if vcd_row is None else vcd_row.get("logit_yes_null")),
                "vcd_logit_no_null": (None if vcd_row is None else vcd_row.get("logit_no_null")),
                "vcd_pred_contrastive": (None if vcd_row is None else vcd_row.get("vcd_pred_contrastive")),
                "vcd_alpha": (None if vcd_row is None else vcd_row.get("vcd_alpha")),
                "subspace_shrink_enabled": bool(getattr(args, "subspace_shrink_enable", False)),
                "subspace_online_mode": step_stats.get("subspace_online_mode"),
                "subspace_shrink_applied": step_stats.get("subspace_shrink_applied"),
                "subspace_risk_score": step_stats.get("subspace_risk_score"),
                "subspace_prefill_risk": step_stats.get("subspace_prefill_risk"),
                "subspace_risk_threshold": step_stats.get("subspace_risk_threshold"),
                "subspace_shrink_alpha": step_stats.get("subspace_shrink_alpha"),
                "subspace_lambda_max": step_stats.get("subspace_lambda_max"),
                "subspace_tau": step_stats.get("subspace_tau"),
                "subspace_tau_max": step_stats.get("subspace_tau_max"),
                "subspace_intervention_layer": step_stats.get("subspace_intervention_layer"),
                "subspace_n_bases": step_stats.get("subspace_n_bases"),
                "subspace_residual_before_mean": step_stats.get("subspace_residual_before_mean"),
                "subspace_residual_after_mean": step_stats.get("subspace_residual_after_mean"),
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            if margin_csv_writer is not None:
                margin_csv_writer.writerow(
                    {
                        "sample_id": idx,
                        "question": question,
                        "prediction": pred,
                        "gt": (gt_answers[0] if gt_answers else ""),
                        "correct": int(acc >= 0.5),
                        "top1_logit": step_stats.get("top1_logit"),
                        "top2_logit": step_stats.get("top2_logit"),
                        "top12_margin": step_stats.get("top12_margin"),
                        "top1_prob": step_stats.get("top1_prob"),
                        "top2_prob": step_stats.get("top2_prob"),
                        "top12_prob_margin": step_stats.get("top12_prob_margin"),
                        "top12_prob_ratio": step_stats.get("top12_prob_ratio"),
                        "top1_id": step_stats.get("top1_id"),
                        "top2_id": step_stats.get("top2_id"),
                        "top1_token": step_stats.get("top1_token"),
                        "top2_token": step_stats.get("top2_token"),
                        "dataset_id": args.dataset_id,
                        "model_id": args.model_id,
                    }
                )
                margin_csv_fp.flush()
            empty_pred_count += int(empty_pred)
            eos_at_step1_count += int(eos_at_step1)
            if digit_rate > 0.3:
                digit_collapse_count += 1
            if prefix_rate > 0.0:
                prefix_collapse_count += 1
            total_gen_len += float(avg_gen_len)
            if gate_on_ratio is not None:
                gate_on_ratio_vals.append(float(gate_on_ratio))
            pref = _prefix_collapse_token(text)
            prefix_hist[pref] = prefix_hist.get(pref, 0) + 1
            if step_log is not None:
                step_log.write(
                    json.dumps(
                        {
                            "sample_summary": True,
                            "sample_id": idx,
                            "dataset_name": args.dataset_id,
                            "task_type": task_type,
                            "question": question,
                            "pred_norm": pred,
                            "gt": gt_answers[0] if gt_answers else "",
                            "correct": bool(acc >= 0.5),
                            "accuracy": float(acc),
                            "tokens": int(step_stats.get("total_tokens", 0)),
                            "latency_s": float(sample_latency_s),
                            "all_steps_h_attn_text_mean": step_stats.get("all_steps_h_attn_text_mean"),
                            "all_steps_h_attn_vision_mean": step_stats.get("all_steps_h_attn_vision_mean"),
                            "all_steps_m_text_mean": step_stats.get("all_steps_m_text_mean"),
                            "all_steps_m_vision_mean": step_stats.get("all_steps_m_vision_mean"),
                            "first_tokens_ids": first_token_ids,
                            "first_tokens_str": first_token_strs,
                            "eos_at_step1": eos_at_step1,
                            "digit_rate": float(digit_rate),
                            "prefix_rate": float(prefix_rate),
                            "empty_pred": int(empty_pred),
                            "avg_gen_len": int(avg_gen_len),
                            "gate_on_ratio": gate_on_ratio,
                            "all_steps_attn_max_mean": step_stats.get("all_steps_attn_max_mean"),
                            "object_attn_mean": step_stats.get("object_attn_mean"),
                            "object_attn_peak_mean": step_stats.get("object_attn_peak_mean"),
                            "object_attn_max": step_stats.get("object_attn_max"),
                            "object_attn_last": step_stats.get("object_attn_last"),
                        }
                    )
                    + "\n"
                )
            risk_log_path = str(getattr(args, "risk_log_jsonl", "") or "").strip()
            if risk_row is not None and risk_log_path:
                with open(risk_log_path, "a", encoding="utf-8") as f_risk:
                    f_risk.write(json.dumps(risk_row) + "\n")
            vcd_log_path = str(getattr(args, "vcd_log_jsonl", "") or "").strip()
            if vcd_row is not None and vcd_log_path:
                with open(vcd_log_path, "a", encoding="utf-8") as f_vcd:
                    f_vcd.write(json.dumps(vcd_row) + "\n")
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
            total_latency_s += sample_latency_s
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
            if getattr(args, "attn_gate_sample_stdout", False):
                print(
                    "HF_Attn_Gate %d/%s | %s | tokens=%d | gate_on=%d | H_attn=%.4f | "
                    "H_t=%.4f | pred=%s | gt=%s"
                    % (
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
                )
            if args.clear_cache_every and n % args.clear_cache_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if step_log is not None:
        step_log.close()
    if margin_csv_fp is not None:
        margin_csv_fp.close()

    overall_acc = total_acc / max(1, n)
    mode_name = getattr(args, "mode", "hf_attn_gate")
    summary_record = {
        "summary": True,
        "mode": mode_name,
        "overall_accuracy": overall_acc,
        "num_samples": n,
        "resume_from": resume_count,
        "avg_latency_s": (total_latency_s / n) if n else 0.0,
        "empty_pred_rate": (empty_pred_count / n) if n else 0.0,
        "eos_at_step1_rate": (eos_at_step1_count / n) if n else 0.0,
        "digit_collapse_rate": (digit_collapse_count / n) if n else 0.0,
        "prefix_collapse_rate": (prefix_collapse_count / n) if n else 0.0,
        "avg_gen_len": (total_gen_len / n) if n else 0.0,
        "avg_gate_on_ratio": (sum(gate_on_ratio_vals) / len(gate_on_ratio_vals)) if gate_on_ratio_vals else 0.0,
        "top_collapsed_prefixes": sorted(prefix_hist.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }
    with open(output_jsonl, "a", encoding="utf-8", buffering=1) as f_out:
        f_out.write(json.dumps(summary_record) + "\n")
        f_out.flush()
    print(
        f"HF_Attn_Gate | accuracy={overall_acc:.4f} | n={n}"
    )
    result = {
        "mode": mode_name,
        "overall_accuracy": overall_acc,
        "num_samples": n,
        "avg_latency_s": (total_latency_s / n) if n else 0.0,
        "empty_pred_rate": (empty_pred_count / n) if n else 0.0,
        "eos_at_step1_rate": (eos_at_step1_count / n) if n else 0.0,
        "digit_collapse_rate": (digit_collapse_count / n) if n else 0.0,
        "prefix_collapse_rate": (prefix_collapse_count / n) if n else 0.0,
        "avg_gen_len": (total_gen_len / n) if n else 0.0,
        "avg_gate_on_ratio": (sum(gate_on_ratio_vals) / len(gate_on_ratio_vals)) if gate_on_ratio_vals else 0.0,
        "top_collapsed_prefixes": sorted(prefix_hist.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }
    try:
        cfg_obj = dict(vars(args))
        if "attn_modules" in cfg_obj:
            cfg_obj["attn_modules"] = f"<{len(args.attn_modules) if hasattr(args, 'attn_modules') and args.attn_modules is not None else 0} modules>"
        if "attn_bias_modules" in cfg_obj:
            cfg_obj["attn_bias_modules"] = f"<{len(args.attn_bias_modules) if hasattr(args, 'attn_bias_modules') and args.attn_bias_modules is not None else 0} modules>"
        cfg_path = f"{output_jsonl}.config.json"
        with open(cfg_path, "w", encoding="utf-8") as f_cfg:
            json.dump(cfg_obj, f_cfg, indent=2, default=str)
        results_path = f"{output_jsonl}.results.json"
        with open(results_path, "w", encoding="utf-8") as f_res:
            json.dump(result, f_res, indent=2)
        diag_path = f"{output_jsonl}.diag_summary.json"
        with open(diag_path, "w", encoding="utf-8") as f_diag:
            json.dump(summary_record, f_diag, indent=2)
    except Exception as exc:
        LOGGER.warning("Failed to write artifact json files: %s", exc)
    return result
