"""Command-line interface for dynamic temperature VQA."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Dynamic temperature VQA with vLLM and Qwen3-VL.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--dataset-id", type=str, default="lmms-lab/MMBench_EN")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional dataset config/name (e.g., en/cn for MMBench).")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument(
        "--mmbench-source",
        type=str,
        default=None,
        help="Optional source filter for MMBench TSV/HF (e.g., MMBench_DEV_EN_V11, MMBench_DEV_CN_V11).",
    )
    parser.add_argument(
        "--mme-hall-only",
        action="store_true",
        help="Filter MME to hallucination categories (existence/count/position/color).",
    )
    parser.add_argument(
        "--mme-hall-categories",
        type=str,
        default="existence,count,position,color",
        help="Comma-separated MME hall categories to keep.",
    )
    parser.add_argument(
        "--hallusionbench-image-root",
        type=str,
        default=None,
        help="Root directory for HallusionBench images when dataset provides filenames only.",
    )
    parser.add_argument(
        "--vizwiz-only-unanswerable",
        action="store_true",
        help="Filter VizWiz to only unanswerable samples.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "baseline",
            "dynamic",
            "dynamictemp",
            "hf_attn_gate",
            "hf_attn_gate_baseline",
            "both",
            "baseline+dynamictemp",
        ],
        default="both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1, help="Number of examples to process (-1 for all).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for prompt-level vLLM calls.")
    parser.add_argument("--baseline-temperature", type=float, default=0.7)
    parser.add_argument("--t-min", type=float, default=0.2)
    parser.add_argument("--t-max", type=float, default=1.2)
    parser.add_argument(
        "--dyn-temp-mode",
        type=str,
        choices=["linear", "entropy"],
        default="entropy",
        help="Logits-processor dynamic temperature mode.",
    )
    parser.add_argument(
        "--dyn-temp-target-entropy",
        type=float,
        default=5.0,
        help="Target entropy for logits-processor dynamic temperature.",
    )
    parser.add_argument(
        "--dyn-temp-entropy-gain",
        type=float,
        default=0.25,
        help="Gain for entropy-based temperature adjustment.",
    )
    parser.add_argument(
        "--dyn-temp-max-steps",
        type=int,
        default=0,
        help="Max steps for linear schedule (0 => use max-new-tokens).",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--num-image-samples", type=int, default=4)
    parser.add_argument(
        "--image-uncert-method",
        type=str,
        choices=["answer_entropy", "view_stability"],
        default="answer_entropy",
        help="이미지 불확실성 계산 방식: answer_entropy(답변 분포) 또는 view_stability(embedding 뷰 안정성).",
    )
    parser.add_argument(
        "--img-uncert-scale",
        type=float,
        default=1.0,
        help="이미지 불확실성 값에 곱해 스케일을 키우거나 줄이는 계수 (동적 온도 반응도 조절).",
    )
    parser.add_argument(
        "--cot-uncert-scale",
        type=float,
        default=1.0,
        help="CoT 토큰 불확실성 값에 곱해 스케일을 조절하는 계수.",
    )
    parser.add_argument(
        "--view-model-id",
        type=str,
        default=None,
        help="뷰 안정성용 vision encoder 모델 ID (None이면 model-id 재사용 시도).",
    )
    parser.add_argument(
        "--view-num-views",
        type=int,
        default=4,
        help="뷰 안정성 계산 시 생성할 augmented view 개수.",
    )
    parser.add_argument(
        "--view-device",
        type=str,
        default="cuda",
        help="뷰 안정성 계산 시 사용할 디바이스 (cuda/cpu).",
    )
    parser.add_argument(
        "--no-view-normalize",
        action="store_true",
        help="뷰 안정성 계산에서 임베딩 L2 정규화를 끔.",
    )
    parser.add_argument("--cot-logprobs-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Minimum number of new tokens before early stopping triggers.",
    )
    parser.add_argument(
        "--max-cot-tokens",
        type=int,
        default=64,
        help="Max number of CoT tokens before forcing final answer phase.",
    )
    parser.add_argument(
        "--min-cot-tokens",
        type=int,
        default=16,
        help="Minimum CoT tokens before allowing final answer phase.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.9,
        help="EMA alpha for entropy-based phase switch (higher = smoother).",
    )
    parser.add_argument(
        "--ema-k",
        type=int,
        default=5,
        help="Consecutive steps required to trigger EMA-based phase switch.",
    )
    parser.add_argument(
        "--ema-tau-t",
        type=float,
        default=1.5,
        help="Ht_ema threshold for phase switch (R1).",
    )
    parser.add_argument(
        "--ema-tau-a",
        type=float,
        default=3.0,
        help="Ha_ema threshold for phase switch (R1).",
    )
    parser.add_argument(
        "--ema-eps-t",
        type=float,
        default=1e-3,
        help="Ht_ema delta threshold for plateau switch (R2).",
    )
    parser.add_argument(
        "--ema-eps-a",
        type=float,
        default=1e-3,
        help="Ha_ema delta threshold for plateau switch (R2).",
    )
    parser.add_argument(
        "--ema-adaptive-tau",
        action="store_true",
        default=True,
        help="Use rolling percentile thresholds for Ht/Ha (adaptive R1).",
    )
    parser.add_argument(
        "--no-ema-adaptive-tau",
        dest="ema_adaptive_tau",
        action="store_false",
        help="Disable adaptive tau; use fixed ema-tau-t/ema-tau-a.",
    )
    parser.add_argument(
        "--ema-window",
        type=int,
        default=32,
        help="Rolling window size for adaptive thresholds.",
    )
    parser.add_argument(
        "--ema-percentile",
        type=float,
        default=30.0,
        help="Percentile for adaptive tau (lower => stricter).",
    )
    parser.add_argument(
        "--stop-strings",
        type=str,
        default="",
        help="Comma-separated stop strings to terminate generation when matched.",
    )
    parser.add_argument("--output-jsonl", type=str, default="results_vqa_dynamic_temp.jsonl")
    parser.add_argument(
        "--resume-jsonl",
        type=str,
        default="",
        help="Resume from an existing jsonl by skipping already processed samples.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Skip the first N samples before processing (overridden by --resume-jsonl if set).",
    )
    parser.add_argument(
        "--attn-gate-tau",
        type=float,
        default=0.2,
        help="Attention gate threshold for image-token attention mass.",
    )
    parser.add_argument(
        "--attn-h-low",
        type=float,
        default=2.0,
        help="Lower entropy threshold for gated temperature adjustment.",
    )
    parser.add_argument(
        "--attn-h-high",
        type=float,
        default=6.0,
        help="Upper entropy threshold for gated temperature adjustment.",
    )
    parser.add_argument(
        "--attn-ent-low",
        type=float,
        default=1.0,
        help="Lower cross-attention entropy threshold for image focus.",
    )
    parser.add_argument(
        "--attn-ent-high",
        type=float,
        default=2.0,
        help="Upper cross-attention entropy threshold for image dispersion.",
    )
    parser.add_argument(
        "--attn-t-base",
        type=float,
        default=0.7,
        help="Base text temperature for HF attention-gate decoding.",
    )
    parser.add_argument(
        "--attn-t-gate",
        type=float,
        default=0.4,
        help="Gated text temperature when entropy is low.",
    )
    parser.add_argument(
        "--attn-t-high",
        type=float,
        default=1.0,
        help="Higher text temperature when cross-attention entropy is high.",
    )
    parser.add_argument(
        "--eos-bias",
        type=float,
        default=0.5,
        help="Positive bias added to EOS logit after scaling (0 disables).",
    )
    parser.add_argument(
        "--repeat-last-token-penalty",
        type=float,
        default=1.0,
        help="Penalty >1.0 to discourage repeating the previous token (1.0 disables).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (0 disables).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling cutoff (1.0 disables).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty over generated tokens (>1.0 enables).",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Disallow repeated n-grams of this size (0 disables).",
    )
    parser.add_argument(
        "--bad-words",
        type=str,
        default="",
        help="Comma-separated tokens/strings to ban during generation.",
    )
    parser.add_argument(
        "--forced-eos",
        action="store_true",
        help="Force EOS on the final generation step.",
    )
    parser.add_argument(
        "--enable-final-yesno-mask",
        action="store_true",
        default=True,
        help="Enable final-phase Yes/No vocab masking for yes/no tasks.",
    )
    parser.add_argument(
        "--disable-final-yesno-mask",
        dest="enable_final_yesno_mask",
        action="store_false",
        help="Disable final-phase Yes/No vocab masking.",
    )
    parser.add_argument(
        "--attn-gate-step-jsonl",
        type=str,
        default=None,
        help="Optional per-step JSONL log for attention gate decoding.",
    )
    parser.add_argument(
        "--attn-gate-step-stdout",
        action="store_true",
        help="Print per-step attention gate stats to stdout.",
    )
    parser.add_argument(
        "--enable-attn-temp",
        dest="enable_attn_temp",
        action="store_true",
        default=True,
        help="Enable attention temperature scaling in HF attention-gate mode.",
    )
    parser.add_argument(
        "--disable-attn-temp",
        dest="enable_attn_temp",
        action="store_false",
        help="Disable attention temperature scaling in HF attention-gate mode.",
    )
    parser.add_argument(
        "--t-attn-base",
        type=float,
        default=1.0,
        help="Base attention temperature.",
    )
    parser.add_argument(
        "--t-attn-low",
        type=float,
        default=0.7,
        help="Lower attention temperature when entropy is high.",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        choices=["eager", "sdpa", "flash"],
        default="eager",
        help="Attention implementation to request in HF mode.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="Fraction of GPU memory vLLM may use (0,1]. Lower if OOM persists.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids to use (sets CUDA_VISIBLE_DEVICES). Example: 0,1",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Override max model length to reduce KV cache footprint if memory is tight.",
    )
    parser.add_argument(
        "--vllm-tqdm",
        action="store_true",
        help="Show vLLM internal progress bars (in addition to our own).",
    )
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=0,
        help="If >0, run gc/torch CUDA cache cleanup every N samples (helps long runs).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code (not recommended for Qwen VL models).",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()
