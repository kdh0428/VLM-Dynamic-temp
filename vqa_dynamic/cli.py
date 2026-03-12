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
        "--sample-id-file",
        type=str,
        default="",
        help="Optional path to text file of sample_id indices (one per line) to evaluate as subset.",
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
            "hf_attn_gate_generate_temp",
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
    parser.add_argument(
        "--temp-mode",
        type=str,
        default="fixed",
        choices=["fixed", "entropy", "rule2x2", "soft2x2", "dual_uncertainty"],
        help="Step-wise temperature mode for custom generate.",
    )
    parser.add_argument("--temp-fixed", type=float, default=1.0)
    parser.add_argument("--temp-min", type=float, default=0.2)
    parser.add_argument("--temp-max", type=float, default=1.5)
    parser.add_argument("--temp-a", type=float, default=1.0)
    parser.add_argument("--temp-b", type=float, default=1.0)
    parser.add_argument("--temp-c", type=float, default=0.0)
    parser.add_argument("--temp-log", action="store_true", help="Log Ht/Ha/T per step.")
    parser.add_argument(
        "--gen-temperature",
        type=float,
        default=1.0,
        help="Temperature for HF model.generate baseline (hf_attn_gate_baseline).",
    )
    parser.add_argument(
        "--gen-do-sample",
        action="store_true",
        help="Enable sampling in hf_attn_gate_baseline (default: greedy).",
    )
    parser.add_argument(
        "--no-gen-do-sample",
        action="store_true",
        help="Disable sampling in hf_attn_gate_baseline (greedy).",
    )
    parser.add_argument(
        "--gen-temp-do-sample",
        action="store_true",
        help="Enable sampling in hf_attn_gate_generate_temp (default: on).",
    )
    parser.add_argument(
        "--no-gen-temp-do-sample",
        action="store_true",
        help="Disable sampling in hf_attn_gate_generate_temp (force greedy).",
    )
    parser.add_argument(
        "--llava-cot",
        action="store_true",
        default=False,
        help="For LLaVA yes/no tasks, use CoT-style prompting with mandatory final-answer format.",
    )
    parser.add_argument(
        "--no-llava-cot",
        dest="llava_cot",
        action="store_false",
        help="Disable CoT-style prompting for LLaVA yes/no tasks.",
    )
    parser.add_argument(
        "--use-attn-entropy",
        action="store_true",
        help="Use attention entropy in temperature schedule (requires output_attentions).",
    )
    parser.add_argument("--ht-low", type=float, default=0.5)
    parser.add_argument("--ht-high", type=float, default=1.5)
    parser.add_argument("--ha-low", type=float, default=2.0)
    parser.add_argument("--ha-high", type=float, default=4.0)
    parser.add_argument("--t-text-low", type=float, default=0.7)
    parser.add_argument("--t-text-high", type=float, default=1.2)
    parser.add_argument(
        "--use-gate-for-intervention",
        action="store_true",
        help="Use gate_on to enable/disable temperature and attention interventions.",
    )
    parser.add_argument(
        "--force-no-intervention",
        action="store_true",
        help="For hf_attn_gate_generate_temp: compute signals but force fixed temperatures (no policy intervention).",
    )
    parser.add_argument(
        "--subspace-shrink-enable",
        action="store_true",
        help="Enable correct-subspace attraction on hidden states in hf_attn_gate_generate_temp.",
    )
    parser.add_argument(
        "--subspace-shrink-alpha",
        type=float,
        default=0.3,
        help="Online hook: lambda_max for risk-proportional alpha. "
             "h' = h - alpha*z_orth where alpha = lambda*clip((R-tau)/(tau_max-tau),0,1). "
             "Legacy path: h' = (1-alpha)*h + alpha*(Uk Uk^T h). 0=no change.",
    )
    parser.add_argument(
        "--subspace-basis-dir",
        type=str,
        default="",
        help="Directory containing mu_layer*.pt and basis_Uk_layer*.pt artifacts.",
    )
    parser.add_argument(
        "--subspace-basis-layer",
        type=int,
        default=25,
        help="Layer id used to load subspace basis files (mu_layer{L}.pt, basis_Uk_layer{L}.pt).",
    )
    parser.add_argument(
        "--subspace-basis-layers",
        type=str,
        default="",
        help="Comma-separated basis layers for simultaneous shrink (e.g., 20,25). If set, overrides --subspace-basis-layer.",
    )
    parser.add_argument(
        "--subspace-risk-csv",
        type=str,
        default="",
        help="CSV with per-sample risk score (must include sample_id and risk column).",
    )
    parser.add_argument(
        "--subspace-risk-col",
        type=str,
        default="r_rel_fused_mean",
        help="Risk column name in --subspace-risk-csv.",
    )
    parser.add_argument(
        "--subspace-risk-threshold",
        type=float,
        default=None,
        help="Apply shrink only when risk >= threshold. If unset, use --subspace-risk-top-pct.",
    )
    parser.add_argument(
        "--subspace-risk-top-pct",
        type=float,
        default=0.2,
        help="If threshold unset: use top-pct risk as threshold (e.g., 0.2 = top 20%%).",
    )
    # ── Online risk-gated intervention (Risk-Gated Manifold Realignment) ──
    parser.add_argument(
        "--subspace-online-risk",
        action="store_true",
        help="Compute risk online from hidden state at target layer (no offline CSV needed).",
    )
    parser.add_argument(
        "--subspace-intervention-layer",
        type=int,
        default=20,
        help="Layer index (hidden_states convention) where the intervention hook is applied.",
    )
    parser.add_argument(
        "--subspace-tau",
        type=float,
        default=0.5,
        help="Risk threshold tau: intervene only when R(h) > tau.",
    )
    parser.add_argument(
        "--subspace-tau-max",
        type=float,
        default=0.7,
        help="Risk saturation point for proportional alpha scaling.",
    )
    parser.add_argument(
        "--subspace-lambda",
        type=float,
        default=0.5,
        help="Max attraction strength lambda: alpha = lambda * clip((R-tau)/(tau_max-tau), 0, 1).",
    )
    parser.add_argument("--soft2x2-ht0", type=float, default=1.0)
    parser.add_argument("--soft2x2-ha0", type=float, default=3.0)
    parser.add_argument("--soft2x2-k-t", type=float, default=6.0)
    parser.add_argument("--soft2x2-k-a", type=float, default=6.0)
    parser.add_argument("--soft2x2-ema-alpha", type=float, default=0.0)
    parser.add_argument("--du-lam", type=float, default=0.9)
    parser.add_argument("--du-k", type=int, default=2)
    parser.add_argument("--du-ht-low", type=float, default=0.55)
    parser.add_argument("--du-ht-high", type=float, default=0.65)
    parser.add_argument("--du-ha-low", type=float, default=0.55)
    parser.add_argument("--du-ha-high", type=float, default=0.65)
    parser.add_argument("--du-t-text-base", type=float, default=0.7)
    parser.add_argument("--du-t-text-min", type=float, default=0.1)
    parser.add_argument("--du-t-text-max", type=float, default=1.3)
    parser.add_argument("--du-t-attn-base", type=float, default=1.0)
    parser.add_argument("--du-t-attn-min", type=float, default=0.3)
    parser.add_argument("--du-t-attn-max", type=float, default=2.0)
    parser.add_argument("--du-dt-explore", type=float, default=0.2)
    parser.add_argument("--du-dt-conserve", type=float, default=0.1)
    parser.add_argument("--du-alpha-conserve", type=float, default=0.7)
    parser.add_argument("--du-dt-floor", type=float, default=0.05)
    parser.add_argument("--du-da-strong", type=float, default=0.4)
    parser.add_argument("--du-ht-star", type=float, default=0.6)
    parser.add_argument("--du-ha-star", type=float, default=0.6)
    parser.add_argument("--du-k-t", type=float, default=0.8)
    parser.add_argument("--du-k-g", type=float, default=0.6)
    parser.add_argument("--du-k-a", type=float, default=0.8)
    parser.add_argument("--du-gateoff-lam", type=float, default=0.9)
    parser.add_argument(
        "--du-ha-adversarial",
        action="store_true",
        help="Adversarial ablation: increase T_text/T_attn when Ha is high (flip Ha influence).",
    )
    parser.add_argument(
        "--du-task-override",
        type=str,
        choices=["none", "precise_c4"],
        default="precise_c4",
        help="Task-specific override policy for dual_uncertainty.",
    )
    parser.add_argument(
        "--du-precise-c4-allow-text-up",
        action="store_true",
        help="Allow slight text-temperature recovery in precise C4 override.",
    )
    parser.add_argument(
        "--du-precise-c4-text-up-delta",
        type=float,
        default=0.0,
        help="Text-temperature increase delta used only when --du-precise-c4-allow-text-up is set.",
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
        "--attn-gate-smooth-mode",
        type=str,
        choices=["ema", "window"],
        default="ema",
        help="S_t smoothing mode for gate_on.",
    )
    parser.add_argument("--attn-gate-ema-alpha", type=float, default=0.9)
    parser.add_argument("--attn-gate-window-size", type=int, default=8)
    parser.add_argument(
        "--attn-gate-adaptive",
        action="store_true",
        help="Adapt attn_gate_tau online to reach target gate rate.",
    )
    parser.add_argument("--attn-gate-target-rate", type=float, default=0.7)
    parser.add_argument("--attn-gate-adapt-alpha", type=float, default=0.9)
    parser.add_argument(
        "--enable-generate-temp-gate-stats",
        dest="enable_generate_temp_gate_stats",
        action="store_true",
        default=True,
        help="Compute gate_on stats in hf_attn_gate_generate_temp mode.",
    )
    parser.add_argument(
        "--disable-generate-temp-gate-stats",
        dest="enable_generate_temp_gate_stats",
        action="store_false",
        help="Disable gate_on stats in hf_attn_gate_generate_temp mode.",
    )
    parser.add_argument(
        "--attn-step-reduce-only",
        dest="attn_step_reduce_only",
        action="store_true",
        default=True,
        help="For hf_attn_gate_generate_temp: reduce attentions per step and do not keep full attention history in outputs.",
    )
    parser.add_argument(
        "--no-attn-step-reduce-only",
        dest="attn_step_reduce_only",
        action="store_false",
        help="Keep full attention history in generate outputs (higher memory usage).",
    )
    parser.add_argument(
        "--attn-gate-sample-stdout",
        action="store_true",
        help="Print per-sample summary stats to stdout.",
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
        "--attn-temp-sanity-check",
        action="store_true",
        help="Run one-step sanity check: compare attention entropy at current t_attn vs probe t_attn.",
    )
    parser.add_argument(
        "--attn-temp-sanity-probe",
        type=float,
        default=2.0,
        help="Probe t_attn value used by --attn-temp-sanity-check.",
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
        "--attn-keyword-bias-word",
        type=str,
        default="",
        help="If set, add key-position bias to this keyword in prompt tokens (e.g., clock).",
    )
    parser.add_argument(
        "--attn-keyword-bias-auto-object",
        action="store_true",
        help="If --attn-keyword-bias-word is empty, auto-extract object phrase from question and apply keyword bias.",
    )
    parser.add_argument(
        "--attn-keyword-bias",
        type=float,
        default=0.0,
        help="Additive bias on attention scores for keyword token positions.",
    )
    parser.add_argument(
        "--attn-keyword-bias-target",
        type=str,
        choices=["all", "last"],
        default="all",
        help="Which attention layers receive keyword bias: all modules or last module only.",
    )
    parser.add_argument(
        "--attn-keyword-bias-max",
        type=float,
        default=4.0,
        help="Clamp absolute keyword-bias magnitude to this value during decoding.",
    )
    parser.add_argument(
        "--bias-steps",
        type=str,
        default="all",
        help='Bias schedule: "all", "prefill", or integer N (first N decode steps).',
    )
    parser.add_argument(
        "--bias-anneal",
        type=str,
        choices=["hard", "linear"],
        default="hard",
        help="Bias schedule type for integer --bias-steps.",
    )
    parser.add_argument(
        "--bias-apply-when",
        type=str,
        choices=["always", "gate_on"],
        default="always",
        help="Apply keyword bias always or only when gate_on.",
    )
    parser.add_argument(
        "--safe-decode-on-bias",
        action="store_true",
        help="When bias is active, force deterministic step decoding (argmax).",
    )
    parser.add_argument(
        "--collapse-risk-threshold",
        type=float,
        default=0.6,
        help="Threshold for collapse risk when using --safe-decode-on-bias (0 disables extra trigger).",
    )
    parser.add_argument(
        "--force-yesno",
        type=str,
        choices=["off", "mask_logits", "rerank"],
        default="off",
        help="Optional yes/no constrained decoding in generate_temp path.",
    )
    parser.add_argument(
        "--logit-margin-top12",
        "--logit-margin-yesno",
        dest="logit_margin_top12",
        action="store_true",
        help="Record first-step top1/top2 logit margin per sample in HF attn-gate modes.",
    )
    parser.add_argument(
        "--logit-margin-output-csv",
        type=str,
        default="",
        help="Optional CSV path to dump per-sample yes/no logit margins.",
    )
    parser.add_argument(
        "--layerwise-object-attn",
        action="store_true",
        help="In HF attn-gate modes, run layer-wise object-attention comparison (correct vs wrong) from an existing results JSONL.",
    )
    parser.add_argument(
        "--layerwise-results-jsonl",
        type=str,
        default="",
        help="Existing results JSONL containing sample_id and accuracy for correct/wrong labels.",
    )
    parser.add_argument(
        "--layerwise-outdir",
        type=str,
        default="analysis_layerwise_object_attn",
        help="Output directory for layer-wise object-attention analysis.",
    )
    parser.add_argument(
        "--layerwise-max-new-tokens",
        type=int,
        default=16,
        help="Generation length used when collecting per-layer attentions.",
    )
    parser.add_argument(
        "--risk-2pass",
        action="store_true",
        help="Enable 2-pass visual perturbation consistency risk logging.",
    )
    parser.add_argument(
        "--risk-perturb",
        type=str,
        choices=["blur", "jpeg", "noise", "color"],
        default="blur",
        help="Visual perturbation type for pass-2 when --risk-2pass is enabled.",
    )
    parser.add_argument(
        "--risk-perturb-strength",
        type=float,
        default=1.0,
        help="Perturbation strength for --risk-perturb.",
    )
    parser.add_argument(
        "--risk-topk-layers",
        type=int,
        default=2,
        help="Top-K layers for S_top grounding proxy (currently approximated from step telemetry).",
    )
    parser.add_argument(
        "--risk-tau-high",
        type=float,
        default=0.5,
        help="High-confidence threshold on yes/no margin for risk gate.",
    )
    parser.add_argument(
        "--risk-tau-s",
        type=float,
        default=0.2,
        help="Low-grounding threshold on S_top for risk gate.",
    )
    parser.add_argument(
        "--risk-stable-both-margins",
        action="store_true",
        help="Require both pass margins > tau_high for confident-ungrounded trigger.",
    )
    parser.add_argument(
        "--risk-gate-enable",
        action="store_true",
        help="Enable 2-stage risk gate flagging (disagree OR high-margin+low-S_top).",
    )
    parser.add_argument(
        "--risk-log-jsonl",
        type=str,
        default="",
        help="Optional JSONL path for per-sample 2-pass risk diagnostics.",
    )
    parser.add_argument(
        "--vcd-mode",
        type=str,
        choices=["off", "diagnostic", "decode"],
        default="off",
        help="Visual Contrastive Decoding: off, diagnostic (log only), decode (contrastive prediction).",
    )
    parser.add_argument(
        "--vcd-alpha",
        type=float,
        default=1.0,
        help="Contrastive weight: logits_final = logits_img - alpha * logits_null.",
    )
    parser.add_argument(
        "--vcd-null-image",
        type=str,
        choices=["blank", "noise"],
        default="blank",
        help="Null image type for VCD pass-2.",
    )
    parser.add_argument(
        "--vcd-log-jsonl",
        type=str,
        default="",
        help="Optional JSONL path for per-sample VCD diagnostics.",
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
