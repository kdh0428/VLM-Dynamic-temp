"""Uncertainty estimation utilities."""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from vllm import LLM, SamplingParams

from .metrics import normalize_answer
from .prompts import extract_final_answer, make_mm_prompt

LOGGER = logging.getLogger(__name__)


def _coerce_logprob_value(val: Any) -> Optional[float]:
    """Extract float logprob from various vLLM logprob structures."""
    if val is None:
        return None
    if isinstance(val, (float, int)):
        return float(val)
    lp_attr = getattr(val, "logprob", None)
    if lp_attr is not None:
        try:
            return float(lp_attr)
        except Exception:
            return None
    return None


def _logprob_dict_from_token_logprob(token_logprob: Any) -> Dict[str, float]:
    """Convert vLLM token logprob structure to a plain dict."""
    if token_logprob is None:
        return {}

    # Case 1: list of Logprob entries
    if isinstance(token_logprob, list):
        d: Dict[str, float] = {}
        for entry in token_logprob:
            tok = getattr(entry, "token", None) or getattr(entry, "decoded_token", None)
            lp = _coerce_logprob_value(entry)
            if tok is not None and lp is not None:
                d[str(tok)] = lp
        if d:
            return d

    # Case 2: dict of token -> logprob or Logprob
    if isinstance(token_logprob, dict):
        d: Dict[str, float] = {}
        for k, v in token_logprob.items():
            lp = _coerce_logprob_value(v)
            if lp is None:
                continue
            d[str(k)] = lp
        if d:
            return d

    # Case 3: single Logprob-like object
    d: Dict[str, float] = {}
    token = (
        getattr(token_logprob, "token", None)
        or getattr(token_logprob, "decoded_token", None)
    )
    logprob = _coerce_logprob_value(token_logprob)
    if token is not None and logprob is not None:
        d[str(token)] = logprob
    top_logprobs = getattr(token_logprob, "top_logprobs", None)
    if top_logprobs:
        for entry in top_logprobs:
            tok = getattr(entry, "token", None) or getattr(entry, "decoded_token", None)
            lp = _coerce_logprob_value(entry)
            if tok is not None and lp is not None:
                d[str(tok)] = lp
    return d


def _entropy_from_logprob_dict(logprob_dict: Dict[str, float]) -> float:
    """Compute entropy for a token given a logprob dict."""
    if not logprob_dict:
        return 0.0
    logprobs = np.array(list(logprob_dict.values()), dtype=np.float64)
    max_lp = logprobs.max()
    probs = np.exp(logprobs - max_lp)
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy


def normalize_final_answer(raw_text: str) -> str:
    """Extract a stable label from raw generation."""
    import re

    text = raw_text or ""
    # 1) MCQ label A/B/C/D
    match_choice = re.search(r"\b([A-Da-d])\b", text)
    if match_choice:
        return match_choice.group(1).upper()

    # 2) Integer number
    match_int = re.search(r"[-+]?\d+", text)
    if match_int:
        return match_int.group(0)

    # 3) Fallback to existing extractor + aggressive normalization
    extracted = extract_final_answer(text)
    return normalize_answer(extracted)


def run_cot_with_logprobs(llm: LLM, prompt: str, image: Image.Image, args) -> Dict[str, Any]:
    """Run CoT generation with logprobs and compute normalized entropy."""
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.baseline_temperature,
        top_p=0.9,
        logprobs=args.cot_logprobs_k,
        stop=None,
    )
    mm_prompt = make_mm_prompt(prompt, image)
    outputs = llm.generate(
        prompts=[mm_prompt],
        sampling_params=sampling_params,
        use_tqdm=args.vllm_tqdm,
    )
    output = outputs[0].outputs[0]
    text = output.text
    logprob_list = getattr(output, "logprobs", []) or []

    entropies: List[float] = []
    k_values: List[int] = []
    for token_lp in logprob_list:
        lp_dict = _logprob_dict_from_token_logprob(token_lp)
        if not lp_dict:
            continue
        entropies.append(_entropy_from_logprob_dict(lp_dict))
        k_values.append(len(lp_dict))

    if not entropies:
        cot_uncertainty = 0.0
    else:
        mean_entropy = float(np.mean(entropies))
        k = max(1, int(np.median(k_values))) if k_values else args.cot_logprobs_k
        cot_uncertainty = mean_entropy / max(math.log(k), 1e-6)
        cot_uncertainty = float(np.clip(cot_uncertainty, 0.0, 1.0))

    return {"text": text, "cot_uncertainty": cot_uncertainty, "entropies": entropies}


def run_cot_batch_with_logprobs(
    llm: LLM, prompts: List[str], images: List[Image.Image], args
) -> List[Dict[str, Any]]:
    """Batch version of CoT with logprobs for efficiency."""
    if not prompts:
        return []
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.baseline_temperature,
        top_p=0.9,
        logprobs=args.cot_logprobs_k,
        stop=None,
    )
    mm_prompts = [make_mm_prompt(p, img) for p, img in zip(prompts, images)]
    try:
        outputs = llm.generate(
            prompts=mm_prompts,
            sampling_params=sampling_params,
            use_tqdm=args.vllm_tqdm,
        )
    except Exception as exc:  # pragma: no cover - runtime robustness
        LOGGER.warning("CoT batch generation failed (skipping batch): %s", exc)
        return []
    results: List[Dict[str, Any]] = []
    for out in outputs:
        gen = out.outputs[0]
        text = gen.text
        logprob_list = getattr(gen, "logprobs", []) or []
        entropies: List[float] = []
        k_values: List[int] = []
        for token_lp in logprob_list:
            lp_dict = _logprob_dict_from_token_logprob(token_lp)
            if not lp_dict:
                continue
            entropies.append(_entropy_from_logprob_dict(lp_dict))
            k_values.append(len(lp_dict))
        if not entropies:
            cot_uncertainty = 0.0
        else:
            mean_entropy = float(np.mean(entropies))
            k = max(1, int(np.median(k_values))) if k_values else args.cot_logprobs_k
            cot_uncertainty = mean_entropy / max(math.log(k), 1e-6)
            cot_uncertainty = float(np.clip(cot_uncertainty, 0.0, 1.0))
        results.append({"text": text, "cot_uncertainty": cot_uncertainty, "entropies": entropies})
    return results


def estimate_image_uncertainty(
    llm: LLM,
    prompt: str,
    image: Image.Image,
    args,
    reuse_answer: Optional[str] = None,
) -> float:
    """Estimate image uncertainty via answer entropy over multiple samples."""
    answers: List[str] = []
    if reuse_answer:
        answers.append(normalize_final_answer(reuse_answer))
    num_to_sample = max(0, args.num_image_samples - len(answers))

    sampling_params = SamplingParams(
        max_tokens=min(args.max_new_tokens, 32),
        temperature=args.baseline_temperature,
        top_p=0.9,
        logprobs=None,
    )

    for _ in range(num_to_sample):
        mm_prompt = make_mm_prompt(prompt, image)
        outputs = llm.generate(
            prompts=[mm_prompt],
            sampling_params=sampling_params,
            use_tqdm=args.vllm_tqdm,
        )
        text = outputs[0].outputs[0].text
        ans = normalize_final_answer(text)
        answers.append(ans)

    if not answers:
        return 0.0

    freq = Counter(answers)
    total = float(sum(freq.values()))
    probs = np.array([count / total for count in freq.values()], dtype=np.float64)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    num_unique = len(freq)
    if num_unique <= 1:
        return 0.0
    denom = math.log(num_unique)
    image_uncert = float(np.clip(entropy / denom, 0.0, 1.0))
    return image_uncert


def compute_dynamic_temperature(
    image_uncert: float,
    cot_uncert: float,
    t_min: float,
    t_max: float,
    alpha: float,
    beta: float,
) -> float:
    """Map combined uncertainty to a temperature."""
    weight_sum = alpha + beta
    if weight_sum <= 0:
        LOGGER.warning("alpha+beta<=0, resetting to 0.5 each")
        alpha = beta = 0.5
        weight_sum = 1.0
    alpha /= weight_sum
    beta /= weight_sum
    u = alpha * image_uncert + beta * cot_uncert
    u = float(np.clip(u, 0.0, 1.0))
    # Uncertainty가 낮을수록 temp↑, 높을수록 temp↓ (t_mid 기준 양방향 스케일)
    t_mid = 0.5 * (t_min + t_max)
    temp = t_mid + (0.5 - u) * (t_max - t_min)
    temp = float(np.clip(temp, t_min, t_max))
    return temp


def _load_image(image: Image.Image | str) -> Image.Image:
    """Load image from PIL or path and convert to RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image path not found: {image}")
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _weak_augment() -> T.Compose:
    """Light augmentation for view stability."""
    return T.Compose(
        [
            T.RandomResizedCrop(size=384, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ]
    )


def view_stability_to_uncertainty(stats: Dict[str, Any]) -> float:
    """Map view stability stats to [0,1] uncertainty (mu high → low u, sigma2 high → high u)."""
    mu = float(stats.get("mu", 1.0))
    sigma2 = float(stats.get("sigma2", 0.0))
    mu_clamped = max(-1.0, min(1.0, mu))
    mu_uncert = 0.5 * (1.0 - mu_clamped)  # 0 when mu=1, 1 when mu=-1
    sigma_uncert = float(np.clip(sigma2, 0.0, 1.0))
    u = 0.5 * mu_uncert + 0.5 * sigma_uncert
    return float(np.clip(u, 0.0, 1.0))


@torch.no_grad()
def compute_view_stability(
    model,
    processor,
    image,
    num_views: int = 4,
    device: str = "cuda",
    normalize: bool = True,
):
    """
    약한 augmentation을 적용한 여러 뷰의 임베딩을 기반으로
    CLIP-style view stability 지표(μ, σ², s_matrix 등)를 계산한다.

    Args:
        model: HF vision encoder 또는 Qwen3-VL 모델(vision encoder 부분 사용)
        processor: AutoProcessor 또는 vision 전처리기
        image: PIL.Image 또는 경로 문자열
        num_views: 생성할 augmented view 개수 (기본 4)
        device: "cuda" 또는 "cpu"
        normalize: True면 L2 normalize 후 cosine similarity 계산

    Returns:
        {
            "mu": float,        # cosine similarity 평균 (off-diagonal)
            "sigma2": float,    # cosine similarity 분산 (off-diagonal)
            "s_matrix": ndarray # KxK similarity matrix
        }
    """

    base_img = _load_image(image)
    aug = _weak_augment()
    views = [aug(base_img) for _ in range(max(1, num_views))]

    feats: List[torch.Tensor] = []
    for view in views:
        try:
            inputs = processor(images=view, return_tensors="pt")
        except TypeError:
            # Some multimodal processors (e.g., Qwen3-VL) require text as well.
            inputs = processor(images=view, text=[""], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if hasattr(model, "get_image_features"):
            emb = model.get_image_features(**inputs)
        elif hasattr(model, "vision_model"):
            vision_out = model.vision_model(**inputs)
            if isinstance(vision_out, (tuple, list)):
                emb = vision_out[0]
            else:
                emb = vision_out.last_hidden_state
            emb = emb[:, 0]  # CLS/first token
        else:
            raise ValueError("Model does not expose vision encoder outputs.")
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        feats.append(emb.detach())

    feat_mat = torch.cat(feats, dim=0)  # (K, D)
    s_matrix = feat_mat @ feat_mat.T
    k = s_matrix.shape[0]
    if k <= 1:
        sims = torch.tensor([1.0], device=s_matrix.device)
    else:
        tri = torch.triu_indices(k, k, offset=1, device=s_matrix.device)
        sims = s_matrix[tri[0], tri[1]]
    mu = sims.mean().item()
    sigma2 = sims.var(unbiased=False).item()

    return {
        "mu": float(mu),
        "sigma2": float(sigma2),
        "s_matrix": s_matrix.detach().cpu().numpy(),
    }


def _load_image(image: Image.Image | str) -> Image.Image:
    """Load image from PIL or path and convert to RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image path not found: {image}")
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _weak_augment() -> T.Compose:
    """Light augmentation for view stability."""
    return T.Compose(
        [
            T.RandomResizedCrop(size=384, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ]
    )


@torch.no_grad()
def compute_view_stability(
    model,
    processor,
    image,
    num_views: int = 4,
    device: str = "cuda",
    normalize: bool = True,
):
    """
    약한 augmentation을 적용한 여러 뷰의 임베딩을 기반으로
    CLIP-style view stability 지표(μ, σ², s_matrix 등)를 계산한다.

    Args:
        model: HF vision encoder 또는 Qwen3-VL 모델(vision encoder 부분 사용)
        processor: AutoProcessor 또는 vision 전처리기
        image: PIL.Image 또는 경로 문자열
        num_views: 생성할 augmented view 개수 (기본 4)
        device: "cuda" 또는 "cpu"
        normalize: True면 L2 normalize 후 cosine similarity 계산

    Returns:
        {
            "mu": float,        # cosine similarity 평균 (off-diagonal)
            "sigma2": float,    # cosine similarity 분산 (off-diagonal)
            "s_matrix": ndarray # KxK similarity matrix
        }
    """

    base_img = _load_image(image)
    aug = _weak_augment()
    views = [aug(base_img) for _ in range(max(1, num_views))]

    feats: List[torch.Tensor] = []
    for view in views:
        inputs = processor(images=view, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if hasattr(model, "get_image_features"):
            emb = model.get_image_features(**inputs)
        elif hasattr(model, "vision_model"):
            vision_out = model.vision_model(**inputs)
            if isinstance(vision_out, (tuple, list)):
                emb = vision_out[0]
            else:
                emb = vision_out.last_hidden_state
            emb = emb[:, 0]  # CLS/first token
        else:
            raise ValueError("Model does not expose vision encoder outputs.")
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        feats.append(emb.detach())

    feat_mat = torch.cat(feats, dim=0)  # (K, D)
    s_matrix = feat_mat @ feat_mat.T
    k = s_matrix.shape[0]
    if k <= 1:
        sims = torch.tensor([1.0], device=s_matrix.device)
    else:
        tri = torch.triu_indices(k, k, offset=1, device=s_matrix.device)
        sims = s_matrix[tri[0], tri[1]]
    mu = sims.mean().item()
    sigma2 = sims.var(unbiased=False).item()

    return {
        "mu": float(mu),
        "sigma2": float(sigma2),
        "s_matrix": s_matrix.detach().cpu().numpy(),
    }
