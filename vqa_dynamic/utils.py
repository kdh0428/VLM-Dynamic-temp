"""Utility helpers for seeding and LLM creation."""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
from vllm import LLM

LOGGER = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:  # pragma: no cover - torch may be absent
        LOGGER.warning("Torch seed not set: %s", exc)


def create_llm(
    model_id: str,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    gpu_ids: str | None = None,
    logits_processors: list | None = None,
    **kwargs: Any,
) -> LLM:
    """Create a vLLM LLM instance."""
    LOGGER.info("Initializing LLM: %s", model_id)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        LOGGER.info("Using CUDA_VISIBLE_DEVICES=%s", gpu_ids)
    return LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        logits_processors=logits_processors,
        **kwargs,
    )


def load_view_encoder(model_id: str, device: str = "cuda", trust_remote_code: bool = True):
    """Load a HF vision encoder (e.g., SigLIP/CLIP) and its processor for view stability."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()
    return model, processor
