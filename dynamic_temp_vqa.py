# pip install "vllm>=0.5.0" "datasets" "Pillow" "tqdm"
"""Entry point for dynamic temperature VQA evaluation using Qwen3-VL and vLLM."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from vqa_dynamic.cli import parse_args
from vqa_dynamic.data import load_vqav2_dataset
from vqa_dynamic.runner import run_baseline, run_dynamic, run_dynamictemp
from vqa_dynamic.hf_attn_gate import run_hf_attn_gate
from vqa_dynamic.utils import create_llm, load_view_encoder, set_seeds
from vqa_dynamic.logitsproc import DynamicTempLogitsProcessor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    set_seeds(args.seed)

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
    # If we need to run both modes, materialize the dataset once to avoid iterator exhaustion.
    if args.mode == "both":
        try:
            dataset_list = list(dataset)
            dataset_for_baseline = dataset_list
            dataset_for_dynamic = dataset_list
        except Exception:
            # Fallback: keep original reference if list conversion fails.
            dataset_for_baseline = dataset
            dataset_for_dynamic = dataset
    else:
        dataset_for_baseline = dataset
        dataset_for_dynamic = dataset
    if args.mode in ("hf_attn_gate", "hf_attn_gate_baseline"):
        result = run_hf_attn_gate(dataset_for_baseline, args, args.output_jsonl)
        print(
            f"HF_Attn_Gate | accuracy={result['overall_accuracy']:.4f} | "
            f"n={result['num_samples']}"
        )
        return

    logits_processors = None
    if args.mode == "dynamictemp":
        logits_processors = [DynamicTempLogitsProcessor]
    llm = create_llm(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        gpu_ids=args.gpu_ids,
        logits_processors=logits_processors,
    )

    # Optional view-stability encoder for image uncertainty
    args.view_encoder = None
    args.view_processor = None
    if args.image_uncert_method == "view_stability":
        view_model_id = args.view_model_id or args.model_id
        args.view_encoder, args.view_processor = load_view_encoder(
            view_model_id, device=args.view_device, trust_remote_code=args.trust_remote_code
        )

    results: List[Dict[str, Any]] = []
    if args.mode in ("baseline", "both", "baseline+dynamictemp"):
        results.append(run_baseline(llm, dataset_for_baseline, args))
    if args.mode in ("dynamic", "both"):
        results.append(run_dynamic(llm, dataset_for_dynamic, args))
    if args.mode in ("dynamictemp", "baseline+dynamictemp"):
        results.append(run_dynamictemp(llm, dataset_for_dynamic, args))

    for res in results:
        if res["mode"] == "baseline":
            print(
                f"Baseline | temp={res['temperature']:.3f} | "
                f"accuracy={res['overall_accuracy']:.4f} | n={res['num_samples']}"
            )
        elif res["mode"] == "dynamic":
            print(
                "Dynamic | "
                f"t_min={res['t_min']:.2f} t_max={res['t_max']:.2f} "
                f"alpha={res['alpha']:.2f} beta={res['beta']:.2f} | "
                f"accuracy={res['overall_accuracy']:.4f} | n={res['num_samples']}"
            )
        elif res["mode"] == "dynamictemp":
            print(
                "DynamicTemp | "
                f"t_min={res['t_min']:.2f} t_max={res['t_max']:.2f} "
                f"mode={res['dyn_temp_mode']} | "
                f"accuracy={res['overall_accuracy']:.4f} | n={res['num_samples']}"
            )


if __name__ == "__main__":
    main()
