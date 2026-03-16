#!/usr/bin/env python3
"""Fit logistic regression probe on hidden states to find correct/wrong boundary.

Trains on train/calibration split only (no test leakage).
Optionally evaluates on a held-out test split.

Outputs:
  weight.pt   — probe weight vector w [d], float32
  bias.pt     — probe bias scalar b, float32
  meta.json   — train/test metrics, hyperparams
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


def load_hidden_vectors(jsonl_path: str, layer: int):
    """Load hidden state vectors and labels from JSONL.

    Returns:
        X: np.ndarray [N, d]
        y: np.ndarray [N] (1=correct, 0=wrong)
    """
    layer_key = str(layer)
    vecs, labels = [], []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("summary"):
                continue
            hs = obj.get("hidden_step0_finaltoken", {})
            h = hs.get(layer_key)
            if h is None:
                continue
            vecs.append(np.array(h, dtype=np.float32))
            labels.append(int(obj.get("correct", 0)))
    return np.stack(vecs), np.array(labels, dtype=np.int32)


def evaluate(clf, X, y):
    """Compute accuracy, AUROC, AP for a fitted classifier."""
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    auroc = roc_auc_score(y, y_prob)
    ap = average_precision_score(y, y_prob)
    return {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "ap": float(ap),
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_wrong": int((y == 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True,
                    help="Train/calibration hidden states JSONL.")
    ap.add_argument("--test-jsonl", default="",
                    help="Test split hidden states JSONL (evaluation only).")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--C", type=float, default=1.0,
                    help="Inverse regularization strength.")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load train
    print(f"[load] train: {args.train_jsonl}")
    X_train, y_train = load_hidden_vectors(args.train_jsonl, args.layer)
    print(f"  N={len(y_train)} correct={y_train.sum()} wrong={(y_train==0).sum()} d={X_train.shape[1]}")

    # Fit logistic regression
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Extract weight and bias
    w = clf.coef_[0].astype(np.float32)        # [d]
    b = clf.intercept_[0].astype(np.float32)    # scalar

    # Save as torch tensors
    torch.save(torch.from_numpy(w).float(), outdir / "weight.pt")
    torch.save(torch.tensor(float(b)).float(), outdir / "bias.pt")

    # Train metrics
    train_metrics = evaluate(clf, X_train, y_train)
    print(f"[train] acc={train_metrics['accuracy']:.4f} "
          f"AUROC={train_metrics['auroc']:.4f} AP={train_metrics['ap']:.4f}")

    # w direction stats
    w_norm = float(np.linalg.norm(w))
    w_hat = w / w_norm

    # Project train samples onto w direction
    scores_train = X_train @ w + b
    s_correct = scores_train[y_train == 1]
    s_wrong = scores_train[y_train == 0]

    meta = {
        "train_jsonl": args.train_jsonl,
        "layer": args.layer,
        "C": args.C,
        "max_iter": args.max_iter,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "feature_dim": int(X_train.shape[1]),
        "w_norm": w_norm,
        "train": train_metrics,
        "train_score_correct_mean": float(s_correct.mean()),
        "train_score_correct_std": float(s_correct.std()),
        "train_score_wrong_mean": float(s_wrong.mean()),
        "train_score_wrong_std": float(s_wrong.std()),
        "train_score_gap": float(s_correct.mean() - s_wrong.mean()),
    }

    # Test evaluation
    test_jsonl = (args.test_jsonl or "").strip()
    if test_jsonl:
        print(f"[load] test: {test_jsonl}")
        X_test, y_test = load_hidden_vectors(test_jsonl, args.layer)
        print(f"  N={len(y_test)} correct={y_test.sum()} wrong={(y_test==0).sum()}")
        test_metrics = evaluate(clf, X_test, y_test)
        print(f"[test]  acc={test_metrics['accuracy']:.4f} "
              f"AUROC={test_metrics['auroc']:.4f} AP={test_metrics['ap']:.4f}")
        scores_test = X_test @ w + b
        st_correct = scores_test[y_test == 1]
        st_wrong = scores_test[y_test == 0]
        meta["test_jsonl"] = test_jsonl
        meta["test"] = test_metrics
        meta["test_score_correct_mean"] = float(st_correct.mean())
        meta["test_score_correct_std"] = float(st_correct.std())
        meta["test_score_wrong_mean"] = float(st_wrong.mean())
        meta["test_score_wrong_std"] = float(st_wrong.std())
        meta["test_score_gap"] = float(st_correct.mean() - st_wrong.mean())

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] outdir={outdir}")
    print(f"  w_norm={w_norm:.4f}")
    print(f"  train score gap: {s_correct.mean() - s_wrong.mean():.4f}")


if __name__ == "__main__":
    main()
