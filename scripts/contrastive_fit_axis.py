#!/usr/bin/env python3
"""Fit contrastive axis from train/calibration hidden states.

Computes:
  mu_correct  = mean hidden state of correct samples
  mu_wrong    = mean hidden state of wrong samples
  m           = (mu_correct + mu_wrong) / 2   (midpoint)
  v_hat       = (mu_correct - mu_wrong) / ||mu_correct - mu_wrong||  (unit axis)

Saves m, v_hat as .pt tensors and calibration stats to meta.json.
Must be run on train/calibration split only (no test leakage).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-jsonl", required=True,
                    help="Train/calibration split hidden states JSONL.")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    layer_key = str(args.layer)

    correct_vecs = []
    wrong_vecs = []

    with open(args.hidden_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("summary"):
                continue
            hs = obj.get("hidden_step0_finaltoken", {})
            h = hs.get(layer_key)
            if h is None:
                continue
            vec = np.array(h, dtype=np.float32)
            if int(obj.get("correct", 0)) == 1:
                correct_vecs.append(vec)
            else:
                wrong_vecs.append(vec)

    if not correct_vecs:
        raise ValueError("no correct samples found")
    if not wrong_vecs:
        raise ValueError("no wrong samples found")

    correct_mat = np.stack(correct_vecs)   # [Nc, d]
    wrong_mat = np.stack(wrong_vecs)       # [Nw, d]

    mu_correct = correct_mat.mean(axis=0)
    mu_wrong = wrong_mat.mean(axis=0)

    m = (mu_correct + mu_wrong) / 2.0
    v = mu_correct - mu_wrong
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-12:
        raise ValueError("mu_correct == mu_wrong, no contrastive axis")
    v_hat = v / v_norm

    torch.save(torch.from_numpy(m).float(), outdir / "contrastive_m.pt")
    torch.save(torch.from_numpy(v_hat).float(), outdir / "contrastive_v_hat.pt")

    # Calibration: compute s = <h - m, v_hat> for all train samples
    all_vecs = np.concatenate([correct_mat, wrong_mat], axis=0)
    all_labels = np.array([1] * len(correct_mat) + [0] * len(wrong_mat))
    s_all = (all_vecs - m) @ v_hat
    s_correct = s_all[all_labels == 1]
    s_wrong = s_all[all_labels == 0]

    meta = {
        "hidden_jsonl": args.hidden_jsonl,
        "layer": args.layer,
        "n_correct": len(correct_mat),
        "n_wrong": len(wrong_mat),
        "feature_dim": int(correct_mat.shape[1]),
        "v_norm": v_norm,
        "s_correct_mean": float(s_correct.mean()),
        "s_correct_std": float(s_correct.std()),
        "s_wrong_mean": float(s_wrong.mean()),
        "s_wrong_std": float(s_wrong.std()),
        "s_gap": float(s_correct.mean() - s_wrong.mean()),
        "s_percentiles": {
            str(p): float(np.percentile(s_all, p))
            for p in [5, 10, 25, 50, 75, 90, 95]
        },
    }

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] outdir={outdir}")
    print(f"  n_correct={len(correct_mat)}, n_wrong={len(wrong_mat)}, d={correct_mat.shape[1]}")
    print(f"  ||mu_correct - mu_wrong|| = {v_norm:.4f}")
    print(f"  s_correct: mean={s_correct.mean():.4f} std={s_correct.std():.4f}")
    print(f"  s_wrong:   mean={s_wrong.mean():.4f} std={s_wrong.std():.4f}")
    print(f"  s_gap:     {s_correct.mean() - s_wrong.mean():.4f}")


if __name__ == "__main__":
    main()
