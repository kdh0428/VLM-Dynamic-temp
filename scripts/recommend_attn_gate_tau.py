#!/usr/bin/env python3
"""Recommend attn-gate tau based on S_t percentiles."""

from __future__ import annotations

import argparse
import json
from typing import List


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] + (vals[c] - vals[f]) * (k - f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend tau from S_t distribution.")
    parser.add_argument("--steps", required=True, help="hf_attn_gate_steps.jsonl")
    parser.add_argument(
        "--target-gate-on",
        type=float,
        default=0.3,
        help="Desired fraction of gate_on steps (0-1).",
    )
    args = parser.parse_args()

    s_vals: List[float] = []
    with open(args.steps, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("summary") or "S_t" not in rec:
                continue
            s_vals.append(float(rec["S_t"]))

    if not s_vals:
        print("No S_t values found.")
        return

    target = max(0.0, min(1.0, args.target_gate_on))
    percentile = 100.0 * (1.0 - target)
    tau = _percentile(s_vals, percentile)
    print(f"samples={len(s_vals)}")
    print(f"target_gate_on={target:.2f} -> tau (p{percentile:.1f}) = {tau:.6f}")


if __name__ == "__main__":
    main()
