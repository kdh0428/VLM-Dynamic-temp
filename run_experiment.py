"""Entry point for VLM hallucination intervention experiments (HF mode)."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from vqa_dynamic.cli import parse_args
from vqa_dynamic.data import load_vqav2_dataset
from vqa_dynamic.experiment_runner import run_hf_attn_gate


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    _set_seeds(args.seed)

    dataset = load_vqav2_dataset(
        args.dataset_id,
        args.split,
        config=args.dataset_config,
        mmbench_source=args.mmbench_source,
        vizwiz_only_unanswerable=args.vizwiz_only_unanswerable,
        mme_hall_only=args.mme_hall_only,
        mme_hall_categories=args.mme_hall_categories,
        hallusionbench_image_root=args.hallusionbench_image_root,
    )

    result = run_hf_attn_gate(dataset, args, args.output_jsonl)
    print(
        f"HF_Attn_Gate | accuracy={result['overall_accuracy']:.4f} | "
        f"n={result['num_samples']}"
    )


if __name__ == "__main__":
    main()
