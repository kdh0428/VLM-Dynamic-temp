#!/usr/bin/env python3
"""Plot per-sample entropy scatter from hf_attn_gate step logs."""

from __future__ import annotations

import argparse
import json
import math
from statistics import mean
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_steps_by_sample(step_rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group step logs into samples using step==0 as a boundary."""
    samples: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for row in step_rows:
        step = row.get("step", None)
        if step == 0 and current:
            samples.append(current)
            current = []
        current.append(row)
    if current:
        samples.append(current)
    return samples


def _is_correct(result: Dict[str, Any]) -> bool:
    if "status" in result:
        return str(result.get("status", "")).lower() == "correct"
    acc = result.get("accuracy", 0.0)
    try:
        return float(acc) >= 0.5
    except Exception:
        return False


def _summarize_sample(
    steps: List[Dict[str, Any]], gate_only: bool
) -> Tuple[float, float] | None:
    if gate_only:
        steps = [s for s in steps if s.get("gate_on")]
    if not steps:
        return None
    h_t = [float(s.get("H_t", 0.0)) for s in steps]
    h_attn = [float(s.get("H_attn", 0.0)) for s in steps]
    return mean(h_t), mean(h_attn)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-sample logit entropy vs cross-attn entropy."
    )
    parser.add_argument("--steps", required=True, help="hf_attn_gate_steps.jsonl")
    parser.add_argument("--results", required=True, help="results_hf_attn_gate*.jsonl")
    parser.add_argument("--out", default="attn_gate_scatter.png")
    parser.add_argument(
        "--include-gate-off",
        action="store_true",
        help="Include all steps (not only gate_on) in the per-sample mean.",
    )
    args = parser.parse_args()

    step_rows = _load_jsonl(args.steps)
    samples = _group_steps_by_sample(step_rows)
    results = _load_jsonl(args.results)

    n = min(len(samples), len(results))
    if len(samples) != len(results):
        print(
            f"Warning: step samples={len(samples)} results={len(results)}; "
            f"using first {n} pairs."
        )

    correct_x, correct_y = [], []
    wrong_x, wrong_y = [], []
    skipped = 0
    gate_only = not args.include_gate_off

    for i in range(n):
        summary = _summarize_sample(samples[i], gate_only=gate_only)
        if summary is None or any(math.isnan(v) for v in summary):
            skipped += 1
            continue
        x, y = summary
        if _is_correct(results[i]):
            correct_x.append(x)
            correct_y.append(y)
        else:
            wrong_x.append(x)
            wrong_y.append(y)

    plt.figure(figsize=(7, 6))
    plt.scatter(correct_x, correct_y, s=16, c="#2ca02c", alpha=0.7, label="Correct")
    plt.scatter(wrong_x, wrong_y, s=16, c="#d62728", alpha=0.7, label="Wrong")
    plt.xlabel("Logit entropy (mean over gate_on steps)")
    plt.ylabel("Cross-attention entropy (mean over gate_on steps)")
    plt.title("HF Attn Gate: Per-sample Entropy Scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out} (skipped {skipped} samples with no gate_on steps).")


if __name__ == "__main__":
    main()
