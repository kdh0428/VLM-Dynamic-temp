#!/usr/bin/env python3
"""Summarize per-sample gate stats from hf_attn_gate step logs."""

from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any, Dict, List


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize per-sample gate stats from hf_attn_gate_steps.jsonl."
    )
    parser.add_argument("--steps", required=True, help="hf_attn_gate_steps.jsonl")
    args = parser.parse_args()

    step_rows = _load_jsonl(args.steps)
    samples = _group_steps_by_sample(step_rows)

    header = [
        "sample_idx",
        "total_tokens",
        "gate_on_count",
        "gate_on_h_attn_mean",
        "gate_on_h_t_mean",
    ]
    print("\t".join(header))
    for idx, steps in enumerate(samples):
        total = len(steps)
        gate_steps = [s for s in steps if s.get("gate_on")]
        gate_count = len(gate_steps)
        if gate_steps:
            h_attn_mean = mean(float(s.get("H_attn", 0.0)) for s in gate_steps)
            h_t_mean = mean(float(s.get("H_t", 0.0)) for s in gate_steps)
        else:
            h_attn_mean = 0.0
            h_t_mean = 0.0
        print(
            f"{idx}\t{total}\t{gate_count}\t{h_attn_mean:.6f}\t{h_t_mean:.6f}"
        )


if __name__ == "__main__":
    main()
