# Risk-Gated Manifold Realignment — Experiment Log

## Project Overview

LLaVA-1.5-7b on POPE (9000 yes/no samples) 할루시네이션 감소를 위한 inference-time intervention 연구.

**Core idea**: 모델의 중간 레이어 hidden state가 "correct decision manifold"에서 벗어난 정도(risk score)를 측정하고, 벗어난 샘플만 선택적으로 realign하여 할루시네이션을 줄인다.

### Risk Score 계산

```
R(h) = ||z_orth|| / ||z||

z = h - μ                    # hidden state에서 correct 평균을 뺌
z_proj = Uk · Uk^T · z      # correct subspace(PCA top-k)에 투영
z_orth = z - z_proj          # 투영 잔차 (subspace 밖 성분)
R(h) = ||z_orth|| / ||z||   # 전체 대비 subspace 밖 비율 (0~1)
```

R(h) > τ이면 할루시네이션 위험 → hidden state를 correct manifold 방향으로 끌어당김:
```
h' = h - α · z_orth          # orthogonal mismatch의 α 비율 제거
α = λ_max · clip((R - τ) / (τ_max - τ), 0, 1)   # risk-proportional
```

---

## 실험 결과 요약

### Baseline
- POPE greedy baseline: **85.46%** (9000 samples)

### Intervention 실험들

| 실험 | Basis Source | Layer | τ | Mode | Accuracy | Net Effect |
|---|---|---|---|---|---|---|
| v3 (all-step) | MME-Hall | L20 | 0.50 | do_sample | 85.52% | +0.06% (무해) |
| τ=0.56 step1 | MME-Hall | L20 | 0.56 | do_sample | 80.79% | -4.67% (sampling noise) |
| τ=0.56 step1 | MME-Hall | L20 | 0.56 | **greedy** | 85.43% | -0.03% (Net -2) |
| τ=0.25 L19 | POPE train20% | L19 | 0.25 | greedy | 85.12% | Net 0 (100% applied, 0 변화) |

### 핵심 발견

#### 1. do_sample vs greedy 차이가 크다
- do_sample 실험(80.79%)에서의 큰 손실은 sampling randomness 때문
- Greedy로 바꾸면 not-applied 샘플에서 flip 0건 (deterministic)
- Applied 샘플도 Net -2 (거의 중립)

#### 2. Prefill vs Decode step의 차이
- **Prefill** (step 1): 전체 프롬프트를 처리하고 yes/no를 결정하는 시점
- **Decode** (step 2+): 이미 생성된 토큰 이후 (comma, period 등)
- v3가 "무해"했던 이유: all-step 개입이지만 decode step 수정은 이미 결정된 답 이후 토큰에만 영향

#### 3. Offline R(h) vs Online R(h) 스케일 불일치 — 중요!

**Offline 추출** (`scripts/extract_hidden_states.py`):
- `model(**inputs, output_hidden_states=True)` → prefill forward pass
- 마지막 토큰의 hidden state에서 R(h) 계산
- MME-Hall basis L20: mean R(h) ≈ 0.53, POPE basis L19: mean R(h) ≈ 0.22

**Online 기록** (`intervention_generate.py`):
- Hook이 **모든 step** (prefill + decode 1 + decode 2 + ...)에서 risk를 기록
- `subspace_risk_score`는 **모든 step의 risk 평균** (`hook_risk_vals`의 mean)
- MME-Hall basis L20: mean ≈ 0.81, POPE basis L19: mean ≈ 0.78
- Decode step의 R(h)가 ~0.82로 매우 높아서 prefill 값을 압도

**결과**: Online `subspace_risk_score`에서 correct/wrong 차이가 거의 없음
- MME-Hall basis: correct 0.8132, wrong 0.8150 (gap 0.0019)
- POPE basis: correct 0.7758, wrong 0.7815 (gap 0.006)

**이것이 τ 캘리브레이션 실패의 원인**:
- τ=0.56 (offline 기준)으로 설정하면 online에서 ~42% 선택 (우연히 적절)
- τ=0.25 (offline 기준)으로 설정하면 online에서 100% 선택 (모두 0.25 초과)

**해결 필요**: prefill step의 risk만 따로 기록/사용해야 offline AUC와 일치 가능

#### 4. Offline AUC (에러 탐지 성능)

| Basis Source | Eval Set | Best Layer | AUC |
|---|---|---|---|
| MME-Hall (cross-domain) | POPE full | L20 | 0.7927 |
| POPE train20% (proper split) | POPE test80% | L19 | **0.8241** |
| MME-Hall | POPE (L18-22 fused) | fused mean | 0.7436 |
| POPE train20% | POPE test80% (fused) | fused mean | 0.8147 |

- **In-domain basis가 +0.03 AUC 향상** (0.7927→0.8241)
- **L19 > L20** (POPE basis 기준에서 일관적)
- **Multi-layer fused는 single best보다 낮음** — L21-22가 희석

#### 5. Intervention 자체가 답을 바꾸지 못함
- τ=0.25 실험: 7200개 전체에 개입, residual을 절반으로 줄임 (41→20)
- **그런데 단 하나의 답도 바뀌지 않음** (baseline과 100% 동일)
- 원인 추정: hook이 현재 **첫 번째 decode step에서만** 개입 (prefill은 skip)
  - `intervention_generate.py` line 110-111: `if is_prefill or self._decode_count > 0: return output`
  - 첫 decode step에서 hidden state를 수정해도, yes/no는 이미 prefill에서 결정됨

---

## 코드 구조

### 주요 파일

1. **`run_experiment.py`** — 진입점 (main)
2. **`vqa_dynamic/cli.py`** — CLI argument parser
3. **`vqa_dynamic/experiment_runner.py`** — 실험 러너
   - `run_hf_attn_gate()`: 메인 루프
   - Subspace basis 로딩 및 설정
   - `_ATTN_MODULES`, `_SUBSPACE_TENSORS`를 module-level globals로 전달 (deepcopy OOM 방지)
4. **`intervention_generate.py`** — 커스텀 generate 함수
   - `SubspaceInterventionHook` class: hook 로직
   - prefill은 skip, 첫 decode step에서만 개입
   - `subspace_risk_score` = 모든 step risk의 평균 ← **문제의 근원**
   - `subspace_prefill_risk` = prefill step의 risk만 별도 기록 (신규 추가)

### 데이터 파일

- `results/hidden_states_pope_L18to22.jsonl` — POPE hidden states (L18-22, 9000 samples)
- `results/hidden_states_mmehall_L18to22.jsonl` — MME-Hall hidden states (L18-22, 240 samples)
- `results/hidden_states_pope_L18to22_train20.jsonl` — POPE train 20% split (1800 samples)
- `results/hidden_states_pope_L18to22_test80.jsonl` — POPE test 80% split (7200 samples)
- `results/pope_test80_sample_ids.txt` — test80% sample ID list

### Basis 파일

- `analysis_subspace/basis_pope_train20_L18to22_k128/` — POPE train20%에서 fit한 basis (L18-22, k=128)
- `results/basis_mmehall_L20_k128/` — MME-Hall L20 basis

### 결과 파일

- `results_pope_baseline.jsonl` — Greedy baseline (85.46%)
- `results_pope_intervention_mmehall_L20_tau056_greedy.jsonl` — MME-Hall basis L20 τ=0.56 greedy (85.43%)
- `results_pope_intervention_popebasis_L19_tau025_greedy_test80.jsonl` — POPE basis L19 τ=0.25 test80% (85.12%)
- `results_pope_intervention_mmehall_L20_allstep.jsonl` — v3 all-step (85.52%)

### Score/Evaluation 파일

- `analysis_subspace/scores_pope_from_mmehall_L20_k128.csv` — MME-Hall basis offline scores
- `analysis_subspace/scores_pope_test80_from_train20_L18to22_k128.csv` — POPE basis offline scores (no leakage)
- `analysis_subspace/eval_pope_test80_from_train20_L18to22_k128/report.json` — Proper evaluation metrics

---

## 다음 단계 (미완료)

### 최우선: Online risk의 prefill-only 기록

**문제**: `subspace_risk_score`가 모든 step의 평균이라 online discriminability가 거의 없음
**해결**: `SubspaceInterventionHook`에서 prefill step의 risk를 따로 기록

```python
# intervention_generate.py SubspaceInterventionHook.__call__ (이미 구현됨)
if is_prefill:
    self.prefill_risk = risk  # prefill risk만 따로 저장
```

출력에 `subspace_prefill_risk`를 추가하고, τ를 offline 스케일에 맞춰 설정.

### 그 다음: Intervention 타이밍 재검토

현재 hook은 **첫 decode step**에서만 개입하는데, yes/no는 이미 prefill에서 결정됨.
선택지:
1. **Prefill에서 개입** — `is_prefill` 조건 제거. 단, 이전에 "prefill intervention is destructive" 결론 (τ=0.56 do_sample에서)이 있었으나, greedy에서는 재검증 필요
2. **Logit-level 개입** — hidden state 대신 yes/no logit을 직접 조정
3. **Multi-pass 방식** — perturbation consistency로 불확실 샘플 탐지 후 답변 수정

### 장기: Modification 전략 개선

현재 `h' = h - α · z_orth` (orthogonal 성분 제거)가 효과가 없음.
- R(h) 50% 감소시켜도 답이 안 바뀜 → yes/no decision boundary와 무관한 방향을 수정하고 있을 수 있음
- **Contrastive direction** (correct vs wrong의 mean difference)으로 개입하면 더 효과적일 수 있음

---

## OOM 방지 주의사항

- `generation_config`에 nn.Module이나 대형 tensor를 넣으면 `model.generate()` 내부 deepcopy에서 OOM
- 해결: module-level globals (`_SUBSPACE_TENSORS`, `_ATTN_MODULES`, `_ATTN_BIAS_MODULES`) 사용
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (또는 `PYTORCH_ALLOC_CONF`) 필요

## 실행 명령 예시

```bash
cd "/root/vlm-hallucination-intervention"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --dataset-id lmms-lab/POPE \
  --split test \
  --mode hf_attn_gate_generate_temp \
  --output-jsonl results_experiment_name.jsonl \
  --seed 42 \
  --attn-impl eager \
  --enable-attn-temp \
  --no-gen-temp-do-sample \
  --max-new-tokens 32 \
  --enable-final-yesno-mask \
  --temp-mode fixed --temp-fixed 1.0 \
  --du-task-override precise_c4 \
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

`--sample-id-file results/pope_test80_sample_ids.txt` 추가하면 test80% 서브셋만 실행.
`python -m vqa_dynamic.cli`는 `__main__.py`가 없어서 동작하지 않음 — 반드시 `python run_experiment.py` 사용.
