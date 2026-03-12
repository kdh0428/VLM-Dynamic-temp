# VLM Hallucination Intervention

Inference-time intervention to reduce hallucination in Vision-Language Models.
Currently targets **LLaVA-1.5-7b** on **POPE** (9000 yes/no samples).

## Core Idea: Risk-Gated Manifold Realignment

1. Extract hidden states from a correct-answer subset and fit a PCA subspace ("correct manifold").
2. At inference, compute a risk score measuring how far the model's hidden state deviates from that manifold.
3. If risk exceeds a threshold, realign the hidden state toward the correct subspace during **prefill** (before the yes/no token is generated).

```
R(h) = ||z_orth|| / ||z||

z = h - mu                     # center around correct mean
z_proj = Uk * Uk^T * z         # project onto correct subspace (PCA top-k)
z_orth = z - z_proj            # orthogonal residual (mismatch)

If R(h) > tau:
  alpha = lambda_max * clip((R - tau) / (tau_max - tau), 0, 1)
  h' = h - alpha * z_orth      # remove mismatch proportionally
```

## Install

```bash
pip install torch transformers pillow datasets
```

## Project Structure

```
run_experiment.py                  # Entry point (main)
intervention_generate.py           # Custom generate loop with SubspaceInterventionHook
vqa_dynamic/
  cli.py                          # CLI argument parser
  experiment_runner.py            # Experiment runner (model loading, main loop)
  data.py                        # Dataset loading
  attn_patch.py                  # Attention patching utilities
  metrics.py                     # Accuracy computation
  prompts.py                     # Prompt building / answer extraction
scripts/
  extract_hidden_states.py       # Extract per-layer hidden states from model
  correct_subspace_common.py     # Shared utilities (PCA fitting, scoring)
  subspace_fit_basis.py          # Fit PCA basis from correct-answer hidden states
  subspace_score_dataset.py      # Score dataset samples against fitted basis
  subspace_evaluate_scores.py    # Evaluate AUC / risk-accuracy correlation
```

## Pipeline

### Step 1: Extract hidden states
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/extract_hidden_states.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --dataset-id lmms-lab/POPE \
  --split test \
  --layers "18,19,20,21,22" \
  --output-jsonl results/hidden_states_pope_L18to22.jsonl \
  --gpu-id 0 --dtype bf16
```

### Step 2: Fit correct subspace (PCA basis)
```bash
cd scripts && python subspace_fit_basis.py \
  --hidden-jsonl ../results/hidden_states_pope_L18to22_train20.jsonl \
  --layers "18,19,20,21,22" \
  --k 128 \
  --outdir ../analysis_subspace/basis_pope_train20_L18to22_k128
```

### Step 3: Score dataset and evaluate
```bash
python subspace_score_dataset.py \
  --hidden-jsonl ../results/hidden_states_pope_L18to22_test80.jsonl \
  --basis-dir ../analysis_subspace/basis_pope_train20_L18to22_k128 \
  --layers "19" --k 128 \
  --output-csv ../analysis_subspace/scores_pope_test80.csv

python subspace_evaluate_scores.py \
  --scores-csv ../analysis_subspace/scores_pope_test80.csv \
  --outdir ../analysis_subspace/eval_pope_test80
```

### Step 4: Run intervention experiment
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --dataset-id lmms-lab/POPE \
  --split test \
  --mode hf_attn_gate_generate_temp \
  --output-jsonl results_pope_intervention.jsonl \
  --seed 42 \
  --attn-impl eager \
  --enable-attn-temp \
  --no-gen-temp-do-sample \
  --max-new-tokens 32 \
  --enable-final-yesno-mask \
  --temp-mode fixed --temp-fixed 1.0 \
  --subspace-shrink-enable \
  --subspace-basis-dir analysis_subspace/basis_pope_train20_L18to22_k128 \
  --subspace-basis-layer 19 \
  --subspace-intervention-layer 19 \
  --subspace-online-risk \
  --subspace-tau 0.25 \
  --subspace-tau-max 0.30 \
  --subspace-lambda 0.5 \
  --subspace-shrink-alpha 0.3
```

Use `--sample-id-file results/pope_test80_sample_ids.txt` to restrict to the test subset (no data leakage).

## Key CLI Flags (Subspace Intervention)

| Flag | Description |
|---|---|
| `--subspace-shrink-enable` | Enable subspace intervention |
| `--subspace-online-risk` | Use online hook (vs legacy logit-replacement) |
| `--subspace-basis-dir` | Directory with `mu_layer*.pt` and `basis_Uk_layer*.pt` |
| `--subspace-basis-layer` | Which layer's basis to load |
| `--subspace-intervention-layer` | Which decoder layer to attach the hook to |
| `--subspace-tau` | Risk threshold (intervene only if R(h) > tau) |
| `--subspace-tau-max` | Upper bound for alpha scaling |
| `--subspace-lambda` | Maximum intervention strength (lambda_max) |
| `--no-gen-temp-do-sample` | Force greedy decoding (recommended) |

## Artifacts

Each experiment run produces:
- `*.jsonl` — per-sample predictions and diagnostics
- `*.config.json` — full configuration snapshot
- `*.results.json` — aggregate accuracy and timing
- `*.diag_summary.json` — run-level diagnostic summary

## Existing Data

| File | Description |
|---|---|
| `results/hidden_states_pope_L18to22.jsonl` | POPE hidden states (L18-22, 9000 samples) |
| `results/hidden_states_mmehall_L18to22.jsonl` | MME-Hall hidden states (L18-22, 240 samples) |
| `analysis_subspace/basis_pope_train20_L18to22_k128/` | PCA basis from POPE train 20% |
| `results/basis_mmehall_L20_k128/` | PCA basis from MME-Hall L20 |

## OOM Notes

- Never put `nn.Module` or large tensors in `generation_config` (deepcopy OOM).
  Use module-level globals (`_SUBSPACE_TENSORS`) instead.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for stable memory.
- Always use `python run_experiment.py`, not `python -m vqa_dynamic.cli`.
