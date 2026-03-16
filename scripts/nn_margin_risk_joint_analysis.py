#!/usr/bin/env python3
"""Joint NN-margin + subspace-risk wrong-detection analysis.

Fits geometry from calibration source (MME-Hall), evaluates on target (POPE).
Computes:
  - NN margin alone
  - Subspace risk R(h) = ||z_orth|| / ||z|| alone
  - Logistic combination of both
  - Score-level fusion (product, sum, etc.)
  - Scatter, ROC, and reliability plots

Usage:
    python scripts/nn_margin_risk_joint_analysis.py \
        --fit-jsonl results/hidden_states_mmehall_L18to22.jsonl \
        --eval-jsonl results/hidden_states_pope_L18to22.jsonl \
        --layers 18 19 20 21 22 \
        --pca-k 128 \
        --outdir analysis_nn_risk_joint
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Data loading ──────────────────────────────────────────────────────────

def load_hidden(jsonl_path: str):
    """Return {layer_str: (X [N,d], y [N])} from a hidden-states JSONL."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("summary"):
                continue
            rows.append(obj)
    if not rows:
        raise ValueError(f"No data rows in {jsonl_path}")

    layers = sorted(rows[0].get("hidden_step0_finaltoken", {}).keys(), key=int)
    result = {}
    for L in layers:
        vecs, labels = [], []
        for r in rows:
            h = r.get("hidden_step0_finaltoken", {}).get(L)
            if h is None:
                continue
            vecs.append(np.array(h, dtype=np.float32))
            labels.append(int(r.get("correct", 0)))
        result[L] = (np.stack(vecs), np.array(labels, dtype=np.int32))
    return result


# ── Metric computation ────────────────────────────────────────────────────

def compute_nn_margin(fit_X, fit_y, eval_X, nn_k=5):
    """NN margin: avg_dist_to_wrong - avg_dist_to_correct. Higher = more correct."""
    Xc = fit_X[fit_y == 1]
    Xw = fit_X[fit_y == 0]
    k_c = min(nn_k, len(Xc))
    k_w = min(nn_k, len(Xw))
    nn_c = NearestNeighbors(n_neighbors=k_c, metric="euclidean").fit(Xc)
    nn_w = NearestNeighbors(n_neighbors=k_w, metric="euclidean").fit(Xw)
    dist_to_c, _ = nn_c.kneighbors(eval_X)
    dist_to_w, _ = nn_w.kneighbors(eval_X)
    return dist_to_w.mean(axis=1) - dist_to_c.mean(axis=1)


def compute_risk(fit_X, fit_y, eval_X, pca_k=128):
    """Subspace risk R(h) = ||z_orth|| / ||z||. Higher = more anomalous/wrong."""
    Xc = fit_X[fit_y == 1]
    mu = Xc.mean(axis=0)

    # PCA on correct samples
    Xc_centered = Xc - mu
    k = min(pca_k, len(Xc) - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc_centered, full_matrices=False)
    Uk = Vt[:k].T  # [d, k]

    # Risk for eval samples
    z = eval_X - mu  # [N, d]
    z_proj = (z @ Uk) @ Uk.T  # [N, d]
    z_orth = z - z_proj  # [N, d]
    z_norm = np.linalg.norm(z, axis=1).clip(min=1e-8)
    orth_norm = np.linalg.norm(z_orth, axis=1)
    risk = orth_norm / z_norm
    return risk, mu, Uk


def auroc_ap(y_true_wrong, scores):
    """Compute AUROC and AP for wrong detection (y=1 means wrong)."""
    auroc = roc_auc_score(y_true_wrong, scores)
    ap = average_precision_score(y_true_wrong, scores)
    return auroc, ap


# ── Combination methods ───────────────────────────────────────────────────

def logistic_combine_cv(feat_matrix, y_wrong, n_splits=5):
    """Fit logistic regression on feature matrix with stratified CV.
    Returns: mean AUROC, mean AP, per-fold AUROCs, full OOF predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(y_wrong))
    fold_aurocs = []

    for train_idx, val_idx in skf.split(feat_matrix, y_wrong):
        X_tr, X_val = feat_matrix[train_idx], feat_matrix[val_idx]
        y_tr, y_val = y_wrong[train_idx], y_wrong[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_val_s)[:, 1]
        oof_scores[val_idx] = proba
        fold_aurocs.append(roc_auc_score(y_val, proba))

    mean_auroc = roc_auc_score(y_wrong, oof_scores)
    mean_ap = average_precision_score(y_wrong, oof_scores)
    return mean_auroc, mean_ap, fold_aurocs, oof_scores


# ── Analysis per layer ────────────────────────────────────────────────────

def analyse_layer(fit_X, fit_y, eval_X, eval_y, layer, pca_k, nn_k=5):
    y_wrong = 1 - eval_y  # 1 = wrong

    # Individual scores
    nn_margin = compute_nn_margin(fit_X, fit_y, eval_X, nn_k=nn_k)
    risk, mu, Uk = compute_risk(fit_X, fit_y, eval_X, pca_k=pca_k)

    # NN margin: lower → more wrong → flip sign for wrong-detection
    nn_wrong_score = -nn_margin
    risk_wrong_score = risk  # higher → more wrong

    nn_auroc, nn_ap = auroc_ap(y_wrong, nn_wrong_score)
    risk_auroc, risk_ap = auroc_ap(y_wrong, risk_wrong_score)

    # Combination 1: product of rank-normalized scores
    nn_rank = nn_wrong_score.argsort().argsort() / len(nn_wrong_score)
    risk_rank = risk_wrong_score.argsort().argsort() / len(risk_wrong_score)
    prod_score = nn_rank * risk_rank
    prod_auroc, prod_ap = auroc_ap(y_wrong, prod_score)

    # Combination 2: sum of rank-normalized scores
    sum_score = nn_rank + risk_rank
    sum_auroc, sum_ap = auroc_ap(y_wrong, sum_score)

    # Combination 3: max of rank-normalized scores
    max_score = np.maximum(nn_rank, risk_rank)
    max_auroc, max_ap = auroc_ap(y_wrong, max_score)

    # Combination 4: logistic regression (CV) on [nn_score, risk_score]
    feat_2d = np.column_stack([nn_wrong_score, risk_wrong_score])
    lr_auroc, lr_ap, lr_folds, lr_oof = logistic_combine_cv(feat_2d, y_wrong)

    # Combination 5: logistic on [nn, risk, nn*risk] (interaction)
    feat_3d = np.column_stack([nn_wrong_score, risk_wrong_score,
                                nn_wrong_score * risk_wrong_score])
    lr3_auroc, lr3_ap, lr3_folds, lr3_oof = logistic_combine_cv(feat_3d, y_wrong)

    # Statistics
    nn_c = nn_margin[eval_y == 1]
    nn_w = nn_margin[eval_y == 0]
    risk_c = risk[eval_y == 1]
    risk_w = risk[eval_y == 0]

    # Correlation between nn_wrong_score and risk_wrong_score
    corr = float(np.corrcoef(nn_wrong_score, risk_wrong_score)[0, 1])

    m = {
        "layer": layer,
        "pca_k": pca_k,
        "nn_k": nn_k,
        "n_eval": len(eval_y),
        "n_correct": int(eval_y.sum()),
        "n_wrong": int((eval_y == 0).sum()),
        "n_fit": len(fit_y),
        # Individual metrics
        "nn_auroc": nn_auroc,
        "nn_ap": nn_ap,
        "nn_correct_mean": float(nn_c.mean()),
        "nn_correct_std": float(nn_c.std()),
        "nn_wrong_mean": float(nn_w.mean()),
        "nn_wrong_std": float(nn_w.std()),
        "risk_auroc": risk_auroc,
        "risk_ap": risk_ap,
        "risk_correct_mean": float(risk_c.mean()),
        "risk_correct_std": float(risk_c.std()),
        "risk_wrong_mean": float(risk_w.mean()),
        "risk_wrong_std": float(risk_w.std()),
        # Combination metrics
        "rank_product_auroc": prod_auroc,
        "rank_product_ap": prod_ap,
        "rank_sum_auroc": sum_auroc,
        "rank_sum_ap": sum_ap,
        "rank_max_auroc": max_auroc,
        "rank_max_ap": max_ap,
        "logistic_2d_auroc": lr_auroc,
        "logistic_2d_ap": lr_ap,
        "logistic_2d_fold_aurocs": lr_folds,
        "logistic_3d_auroc": lr3_auroc,
        "logistic_3d_ap": lr3_ap,
        "logistic_3d_fold_aurocs": lr3_folds,
        # Correlation
        "nn_risk_correlation": corr,
    }

    # Store arrays for plotting
    m["_nn_margin"] = nn_margin
    m["_risk"] = risk
    m["_y_wrong"] = y_wrong
    m["_lr_oof"] = lr_oof
    m["_lr3_oof"] = lr3_oof
    m["_nn_wrong_score"] = nn_wrong_score
    m["_risk_wrong_score"] = risk_wrong_score

    return m


# ── Plotting ──────────────────────────────────────────────────────────────

def make_plots(results, outdir: Path):
    for r in results:
        layer = r["layer"]
        nn_margin = r["_nn_margin"]
        risk = r["_risk"]
        y_wrong = r["_y_wrong"]
        eval_y = 1 - y_wrong
        lr_oof = r["_lr_oof"]
        lr3_oof = r["_lr3_oof"]

        # ── 1. Scatter: NN margin vs Risk, colored by correct/wrong ──
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(nn_margin[eval_y == 1], risk[eval_y == 1],
                   s=3, alpha=0.15, c="steelblue", label=f"Correct (n={int(eval_y.sum())})")
        ax.scatter(nn_margin[eval_y == 0], risk[eval_y == 0],
                   s=8, alpha=0.5, c="coral", label=f"Wrong (n={int((eval_y==0).sum())})", zorder=5)
        ax.set_xlabel("NN margin (positive = closer to correct)")
        ax.set_ylabel("Risk R(h) = ||z_orth|| / ||z||")
        ax.set_title(f"L{layer}: NN margin vs Subspace Risk (r={r['nn_risk_correlation']:.3f})")
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"scatter_L{layer}.png", dpi=150)
        plt.close(fig)

        # ── 2. ROC curves: individual + combined ──
        fig, ax = plt.subplots(figsize=(8, 6))
        # NN margin
        fpr, tpr, _ = roc_curve(y_wrong, r["_nn_wrong_score"])
        ax.plot(fpr, tpr, label=f"NN margin (AUROC={r['nn_auroc']:.4f})", lw=2)
        # Risk
        fpr, tpr, _ = roc_curve(y_wrong, r["_risk_wrong_score"])
        ax.plot(fpr, tpr, label=f"Risk R(h) (AUROC={r['risk_auroc']:.4f})", lw=2)
        # Rank sum
        nn_rank = r["_nn_wrong_score"].argsort().argsort() / len(y_wrong)
        risk_rank = r["_risk_wrong_score"].argsort().argsort() / len(y_wrong)
        fpr, tpr, _ = roc_curve(y_wrong, nn_rank + risk_rank)
        ax.plot(fpr, tpr, label=f"Rank sum (AUROC={r['rank_sum_auroc']:.4f})", lw=1.5, ls="--")
        # Logistic 2D CV
        fpr, tpr, _ = roc_curve(y_wrong, lr_oof)
        ax.plot(fpr, tpr, label=f"Logistic 2D CV (AUROC={r['logistic_2d_auroc']:.4f})", lw=2, ls="-.")
        # Logistic 3D CV
        fpr, tpr, _ = roc_curve(y_wrong, lr3_oof)
        ax.plot(fpr, tpr, label=f"Logistic 3D CV (AUROC={r['logistic_3d_auroc']:.4f})", lw=2, ls=":")

        ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
        ax.set_xlabel("False Positive Rate (correct misclassified as wrong)")
        ax.set_ylabel("True Positive Rate (wrong correctly detected)")
        ax.set_title(f"L{layer}: Wrong-Detection ROC — Individual vs Combined")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"roc_L{layer}.png", dpi=150)
        plt.close(fig)

        # ── 3. Precision-Recall curves ──
        fig, ax = plt.subplots(figsize=(8, 6))
        prec, rec, _ = precision_recall_curve(y_wrong, r["_nn_wrong_score"])
        ax.plot(rec, prec, label=f"NN margin (AP={r['nn_ap']:.4f})", lw=2)
        prec, rec, _ = precision_recall_curve(y_wrong, r["_risk_wrong_score"])
        ax.plot(rec, prec, label=f"Risk R(h) (AP={r['risk_ap']:.4f})", lw=2)
        prec, rec, _ = precision_recall_curve(y_wrong, lr_oof)
        ax.plot(rec, prec, label=f"Logistic 2D CV (AP={r['logistic_2d_ap']:.4f})", lw=2, ls="-.")
        prec, rec, _ = precision_recall_curve(y_wrong, lr3_oof)
        ax.plot(rec, prec, label=f"Logistic 3D CV (AP={r['logistic_3d_ap']:.4f})", lw=2, ls=":")

        base_rate = y_wrong.mean()
        ax.axhline(base_rate, color="gray", ls="--", lw=0.8, label=f"Base rate ({base_rate:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"L{layer}: Wrong-Detection Precision-Recall")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"pr_L{layer}.png", dpi=150)
        plt.close(fig)

        # ── 4. Distribution histograms (2x2) ──
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # NN margin distribution
        ax = axes[0, 0]
        ax.hist(nn_margin[eval_y == 1], bins=80, alpha=0.6, density=True,
                label="Correct", color="steelblue")
        ax.hist(nn_margin[eval_y == 0], bins=80, alpha=0.6, density=True,
                label="Wrong", color="coral")
        ax.set_xlabel("NN margin")
        ax.set_title(f"NN margin distribution (AUROC={r['nn_auroc']:.4f})")
        ax.legend(fontsize=8)

        # Risk distribution
        ax = axes[0, 1]
        ax.hist(risk[eval_y == 1], bins=80, alpha=0.6, density=True,
                label="Correct", color="steelblue")
        ax.hist(risk[eval_y == 0], bins=80, alpha=0.6, density=True,
                label="Wrong", color="coral")
        ax.set_xlabel("Risk R(h)")
        ax.set_title(f"Risk distribution (AUROC={r['risk_auroc']:.4f})")
        ax.legend(fontsize=8)

        # Logistic 2D OOF score distribution
        ax = axes[1, 0]
        ax.hist(lr_oof[y_wrong == 0], bins=80, alpha=0.6, density=True,
                label="Correct", color="steelblue")
        ax.hist(lr_oof[y_wrong == 1], bins=80, alpha=0.6, density=True,
                label="Wrong", color="coral")
        ax.set_xlabel("Logistic 2D P(wrong)")
        ax.set_title(f"Logistic 2D combined (AUROC={r['logistic_2d_auroc']:.4f})")
        ax.legend(fontsize=8)

        # Logistic 3D OOF score distribution
        ax = axes[1, 1]
        ax.hist(lr3_oof[y_wrong == 0], bins=80, alpha=0.6, density=True,
                label="Correct", color="steelblue")
        ax.hist(lr3_oof[y_wrong == 1], bins=80, alpha=0.6, density=True,
                label="Wrong", color="coral")
        ax.set_xlabel("Logistic 3D P(wrong)")
        ax.set_title(f"Logistic 3D combined (AUROC={r['logistic_3d_auroc']:.4f})")
        ax.legend(fontsize=8)

        fig.suptitle(f"L{layer}: Score Distributions", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"distributions_L{layer}.png", dpi=150)
        plt.close(fig)

    # ── 5. Cross-layer AUROC summary bar chart ──
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        layers = [r["layer"] for r in results]
        x = np.arange(len(layers))
        metrics = [
            ("nn_auroc", "NN margin"),
            ("risk_auroc", "Risk R(h)"),
            ("rank_sum_auroc", "Rank sum"),
            ("logistic_2d_auroc", "Logistic 2D"),
            ("logistic_3d_auroc", "Logistic 3D"),
        ]
        w = 0.15
        for i, (key, label) in enumerate(metrics):
            vals = [r[key] for r in results]
            ax.bar(x + i * w, vals, w, label=label)
        ax.set_xticks(x + 2 * w)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylabel("Wrong-Detection AUROC")
        ax.set_title("NN Margin + Risk: Individual and Combined AUROC by Layer")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "auroc_summary.png", dpi=150)
        plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────

def write_report(results, outdir, fit_path, eval_path):
    lines = []
    lines.append("=" * 78)
    lines.append("Joint NN-margin + Subspace-Risk Wrong-Detection Analysis")
    lines.append(f"Fit (calibration): {fit_path}")
    lines.append(f"Eval (target):     {eval_path}")
    lines.append("=" * 78)
    lines.append("")

    header = (
        f"{'Layer':>5} | {'NN':>7} {'Risk':>7} | "
        f"{'RkSum':>7} {'RkProd':>7} {'RkMax':>7} | "
        f"{'LR_2D':>7} {'LR_3D':>7} | {'Corr':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"  L{r['layer']:<3} | "
            f"{r['nn_auroc']:>7.4f} {r['risk_auroc']:>7.4f} | "
            f"{r['rank_sum_auroc']:>7.4f} {r['rank_product_auroc']:>7.4f} {r['rank_max_auroc']:>7.4f} | "
            f"{r['logistic_2d_auroc']:>7.4f} {r['logistic_3d_auroc']:>7.4f} | "
            f"{r['nn_risk_correlation']:>6.3f}"
        )
    lines.append("")

    # Best per method
    lines.append("Best layer per method (AUROC):")
    for key, name in [
        ("nn_auroc", "NN margin alone"),
        ("risk_auroc", "Risk R(h) alone"),
        ("rank_sum_auroc", "Rank sum (NN+Risk)"),
        ("logistic_2d_auroc", "Logistic [NN, Risk]"),
        ("logistic_3d_auroc", "Logistic [NN, Risk, NN*Risk]"),
    ]:
        best = max(results, key=lambda r: r[key])
        lines.append(f"  {name:<35} L{best['layer']} AUROC={best[key]:.4f}")
    lines.append("")

    # Improvement from combination
    lines.append("Combination gain over best individual:")
    for r in results:
        best_ind = max(r["nn_auroc"], r["risk_auroc"])
        best_comb = max(r["rank_sum_auroc"], r["logistic_2d_auroc"], r["logistic_3d_auroc"])
        delta = best_comb - best_ind
        lines.append(
            f"  L{r['layer']}: best_individual={best_ind:.4f} "
            f"best_combined={best_comb:.4f} delta={delta:+.4f}"
        )
    lines.append("")

    # Distribution stats
    lines.append("Score distributions (mean +/- std):")
    for r in results:
        lines.append(f"  L{r['layer']}:")
        lines.append(
            f"    NN margin:  correct={r['nn_correct_mean']:.3f}+/-{r['nn_correct_std']:.3f}  "
            f"wrong={r['nn_wrong_mean']:.3f}+/-{r['nn_wrong_std']:.3f}"
        )
        lines.append(
            f"    Risk R(h):  correct={r['risk_correct_mean']:.4f}+/-{r['risk_correct_std']:.4f}  "
            f"wrong={r['risk_wrong_mean']:.4f}+/-{r['risk_wrong_std']:.4f}"
        )
    lines.append("")

    # Interpretation
    best_overall = max(results, key=lambda r: max(
        r["nn_auroc"], r["logistic_2d_auroc"], r["logistic_3d_auroc"]
    ))
    best_val = max(best_overall["nn_auroc"],
                   best_overall["logistic_2d_auroc"],
                   best_overall["logistic_3d_auroc"])
    nn_alone = best_overall["nn_auroc"]
    lines.append("Summary:")
    lines.append(f"  Best overall: L{best_overall['layer']} AUROC={best_val:.4f}")
    lines.append(f"  NN margin alone at that layer: AUROC={nn_alone:.4f}")
    delta = best_val - nn_alone
    if delta > 0.005:
        lines.append(f"  Risk adds {delta:+.4f} AUROC on top of NN margin.")
    else:
        lines.append(
            f"  Risk adds negligible gain ({delta:+.4f}). "
            f"NN margin and risk may be capturing similar signal."
        )
    lines.append(f"  NN-Risk correlation: {best_overall['nn_risk_correlation']:.3f}")
    lines.append("")

    report = "\n".join(lines)
    with open(outdir / "report.txt", "w") as f:
        f.write(report)
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-jsonl", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 19, 20, 21, 22])
    ap.add_argument("--pca-k", type=int, default=128)
    ap.add_argument("--nn-k", type=int, default=5)
    ap.add_argument("--outdir", default="analysis_nn_risk_joint")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fit data: {args.fit_jsonl}")
    fit_data = load_hidden(args.fit_jsonl)
    print(f"Loading eval data: {args.eval_jsonl}")
    eval_data = load_hidden(args.eval_jsonl)

    requested = set(str(l) for l in args.layers)
    common = sorted(requested & set(fit_data.keys()) & set(eval_data.keys()), key=int)
    print(f"Layers to analyse: {common}")

    results = []
    for L in common:
        print(f"\n=== Layer {L} ===")
        fit_X, fit_y = fit_data[L]
        eval_X, eval_y = eval_data[L]
        print(f"  fit: {len(fit_y)} (correct={fit_y.sum()}, wrong={(fit_y==0).sum()})")
        print(f"  eval: {len(eval_y)} (correct={eval_y.sum()}, wrong={(eval_y==0).sum()})")

        m = analyse_layer(fit_X, fit_y, eval_X, eval_y, int(L),
                          pca_k=args.pca_k, nn_k=args.nn_k)
        results.append(m)

        print(f"  NN margin AUROC  = {m['nn_auroc']:.4f}  (AP={m['nn_ap']:.4f})")
        print(f"  Risk R(h) AUROC  = {m['risk_auroc']:.4f}  (AP={m['risk_ap']:.4f})")
        print(f"  Rank sum AUROC   = {m['rank_sum_auroc']:.4f}")
        print(f"  Logistic 2D AUROC= {m['logistic_2d_auroc']:.4f}")
        print(f"  Logistic 3D AUROC= {m['logistic_3d_auroc']:.4f}")
        print(f"  NN-Risk corr     = {m['nn_risk_correlation']:.3f}")

    # Save metrics (strip numpy arrays)
    metrics_out = []
    for r in results:
        metrics_out.append({k: v for k, v in r.items() if not k.startswith("_")})
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nGenerating plots...")
    make_plots(results, outdir)

    print()
    write_report(results, outdir, args.fit_jsonl, args.eval_jsonl)

    print(f"\nDone. Output in {outdir}/")


if __name__ == "__main__":
    main()
