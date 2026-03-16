# VLM Hallucination Intervention

Inference-time intervention to reduce hallucination in Vision-Language Models.
Currently targets **LLaVA-1.5-7b** on **POPE** (9000 yes/no samples).

## Intervention Approaches

### 1. Risk-Gated Manifold Realignment (Subspace)

Fit a PCA subspace from correct-answer hidden states. At inference, measure deviation and realign.

```
R(h) = ||z_orth|| / ||z||          # risk score (ratio)
e(h) = ||z_orth||                   # absolute residual norm

z = h - mu                          # center around correct mean
z_proj = Uk Uk^T z                  # project onto correct subspace
z_orth = z - z_proj                 # orthogonal residual

If gate passes:
  alpha = lambda_max * clip((R - tau) / (tau_max - tau), 0, 1)
  h' = h - alpha * z_orth
```

Gate modes (`--risk-gate-mode`):
- `ratio_only`: R(h) > tau (default)
- `abs_only`: e(h) > tau_abs
- `ratio_and_abs`: both conditions

### 2. Contrastive-Axis Intervention

Steer hidden states along the (correct - wrong) centroid direction from calibration data.

```
s = <h - midpoint, v_hat>           # projection onto contrastive axis
If s < tau_s:
  s' = s + lambda * (tau_s - s)     # push toward correct side
  h' = midpoint + s' * v_hat + r    # preserve orthogonal component
```

Optional risk gate from subspace basis. Supersedes subspace hook when active.

### 3. Logistic Probe Intervention

Use a supervised probe direction (w, b) trained on correct/wrong hidden states.

```
s = w^T h + b                       # probe margin
alpha = lambda * clip((tau_s - s) / (tau_s - tau_min), 0, 1)
h' = h + alpha * w_hat              # push along probe direction
```

Supersedes both contrastive and subspace hooks. Hook hierarchy: **probe > contrastive > subspace**.

All interventions act on **prefill only** (the yes/no decision is determined at prefill; decode steps are too late).

## Install

```bash
pip install -r requirements.txt
```

## Project Structure

```
run_experiment.py                     # Entry point (main)
intervention_generate.py              # Custom generate loop with intervention hooks
vqa_dynamic/
  cli.py                             # CLI argument parser
  experiment_runner.py               # Experiment runner (model loading, main loop)
  data.py                            # Dataset loading
  attn_patch.py                      # Attention patching utilities
  metrics.py                         # Accuracy computation
  prompts.py                         # Prompt building / answer extraction
scripts/
  extract_hidden_states.py           # Extract per-layer hidden states from model
  subspace_fit_basis.py              # Fit PCA basis from correct-answer hidden states
  contrastive_fit_axis.py            # Fit contrastive axis (mu_correct - mu_wrong)
  logistic_probe_fit.py              # Fit logistic probe on hidden states
  cross_dataset_geometry_analysis.py # Cross-dataset geometry analysis (e.g. MME-Hall -> POPE)
  nn_margin_risk_joint_analysis.py   # Joint NN-margin + subspace-risk analysis
  risk_gate_mode_sweep.py            # Compare ratio_only / abs_only / ratio_and_abs gating
  correct_subspace_common.py         # Shared utilities (PCA fitting, scoring)
  subspace_score_dataset.py          # Score dataset samples against fitted basis
  subspace_evaluate_scores.py        # Evaluate AUC / risk-accuracy correlation
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

### Step 2: Fit calibration artifacts

**PCA basis** (for subspace intervention / risk gating):
```bash
python scripts/subspace_fit_basis.py \
  --hidden-jsonl results/hidden_states_pope_L18to22_train20.jsonl \
  --layers "19" --k 128 \
  --outdir results/basis_pope_L19_k128
```

**Contrastive axis**:
```bash
python scripts/contrastive_fit_axis.py \
  --train-jsonl results/hidden_states_pope_L18to22_train20.jsonl \
  --layer 19 \
  --outdir results/contrastive_axis_L19
```

**Logistic probe**:
```bash
python scripts/logistic_probe_fit.py \
  --train-jsonl results/hidden_states_pope_L18to22_train20.jsonl \
  --test-jsonl results/hidden_states_pope_L18to22_test80.jsonl \
  --layer 19 \
  --outdir results/logistic_probe_L19
```

### Step 3: Run intervention experiments

**Subspace intervention** (risk-gated manifold realignment):
```bash
python run_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --dataset-id lmms-lab/POPE --split test \
  --mode hf_attn_gate_generate_temp \
  --output-jsonl results_pope_subspace.jsonl \
  --seed 42 --attn-impl eager --no-gen-temp-do-sample \
  --max-new-tokens 4 --batch-size 16 \
  --enable-attn-temp --enable-final-yesno-mask \
  --temp-mode fixed --temp-fixed 1.0 \
  --subspace-shrink-enable \
  --subspace-basis-dir results/basis_pope_L19_k128 \
  --subspace-basis-layer 19 \
  --subspace-intervention-layer 19 \
  --subspace-online-risk \
  --subspace-tau 0.65 --subspace-tau-max 0.70 --subspace-lambda 0.5 \
  --risk-gate-mode ratio_only \
  --sample-id-file results/pope_test80_sample_ids.txt
```

**Contrastive intervention** (with risk gate):
```bash
python run_experiment.py \
  ... (same base flags) \
  --subspace-shrink-enable \
  --subspace-basis-dir results/basis_pope_L19_k128 \
  --subspace-basis-layer 19 \
  --contrastive-enable \
  --contrastive-dir results/contrastive_axis_L19 \
  --contrastive-layer 19 \
  --contrastive-tau-s 7.0 --contrastive-lambda 0.5
```

**Probe intervention** (with risk gate):
```bash
python run_experiment.py \
  ... (same base flags) \
  --subspace-shrink-enable \
  --subspace-basis-dir results/basis_pope_L19_k128 \
  --subspace-basis-layer 19 \
  --probe-enable \
  --probe-dir results/logistic_probe_L19 \
  --probe-layer 19 \
  --probe-tau-s 0.0 --probe-tau-min -5.0 --probe-lambda 1.0
```

### Step 4: Offline geometry analysis

**Cross-dataset geometry** (MME-Hall -> POPE):
```bash
python scripts/cross_dataset_geometry_analysis.py \
  --fit-jsonl results/hidden_states_mmehall_L18to22.jsonl \
  --eval-jsonl results/hidden_states_pope_L18to22.jsonl \
  --outdir analysis_cross_geometry
```

**Risk gate mode comparison**:
```bash
python scripts/risk_gate_mode_sweep.py \
  --fit-jsonl results/hidden_states_mmehall_L18to22.jsonl \
  --eval-jsonl results/hidden_states_pope_L18to22.jsonl \
  --layers 18 19 20 21 22 \
  --outdir analysis_risk_gate_mode
```

## Key CLI Flags

### Subspace Intervention
| Flag | Description |
|---|---|
| `--subspace-shrink-enable` | Enable subspace intervention |
| `--subspace-online-risk` | Use online hook (vs legacy logit-replacement) |
| `--subspace-basis-dir` | Directory with `mu_layer*.pt` and `basis_Uk_layer*.pt` |
| `--subspace-basis-layer` | Which layer's basis to load |
| `--subspace-intervention-layer` | Decoder layer for hook attachment |
| `--subspace-tau` | Risk ratio threshold (intervene when R(h) > tau) |
| `--subspace-tau-max` | Upper bound for alpha scaling |
| `--subspace-lambda` | Maximum intervention strength |
| `--risk-gate-mode` | `ratio_only` / `abs_only` / `ratio_and_abs` |
| `--subspace-tau-abs` | Absolute residual threshold (for abs_only / ratio_and_abs) |

### Contrastive Intervention
| Flag | Description |
|---|---|
| `--contrastive-enable` | Enable contrastive-axis intervention (supersedes subspace) |
| `--contrastive-dir` | Directory with `contrastive_m.pt`, `contrastive_v_hat.pt` |
| `--contrastive-layer` | Layer for contrastive hook |
| `--contrastive-tau-s` | Projection threshold (intervene when s < tau_s) |
| `--contrastive-lambda` | Steering strength |

### Probe Intervention
| Flag | Description |
|---|---|
| `--probe-enable` | Enable probe intervention (supersedes contrastive & subspace) |
| `--probe-dir` | Directory with `weight.pt`, `bias.pt`, `meta.json` |
| `--probe-layer` | Layer for probe hook (must match meta.json) |
| `--probe-tau-s` | Margin threshold |
| `--probe-tau-min` | Lower margin for alpha scheduling (must be < tau_s) |
| `--probe-lambda` | Max push magnitude |

### General
| Flag | Description |
|---|---|
| `--no-gen-temp-do-sample` | Force greedy decoding (recommended for POPE) |
| `--sample-id-file` | Restrict to subset (e.g. test80% for no train leakage) |
| `--max-new-tokens` | Max generated tokens (4 is sufficient for yes/no) |

## Calibration Source Enforcement

When risk gating is active, same-source calibration is enforced:
- Basis `meta.json`'s `hidden_jsonl` must match the contrastive/probe `meta.json`'s `train_jsonl`
- Mismatch raises `ValueError` at startup (not a silent warning)
- Probe `meta.json`'s `layer` must match `--probe-layer` (error otherwise)

## Artifacts

Each experiment run produces:
- `*.jsonl` — per-sample predictions and diagnostics (includes risk, orth_abs, probe scores)
- `*.config.json` — full configuration snapshot
- `*.results.json` — aggregate accuracy and timing
- `*.diag_summary.json` — run-level diagnostic summary

## Detection Performance (Cross-Domain, MME-Hall -> POPE L19)

| Method | AUROC |
|---|---|
| NN margin | 0.8158 |
| Risk R(h) ratio | 0.7738 |
| LLR | 0.7699 |
| Risk e(h) absolute | 0.5585 |
| Logistic [R, e] combined | 0.7857 |

Same-domain logistic probe (POPE train20 -> test80): AUROC = 0.8739

## OOM Notes

- Never put `nn.Module` or large tensors in `generation_config` (deepcopy OOM).
  Use module-level globals (`_SUBSPACE_TENSORS`, `_CONTRASTIVE_TENSORS`, `_PROBE_TENSORS`).
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for stable memory.
- Always use `python run_experiment.py`, not `python -m vqa_dynamic.experiment_runner`.
