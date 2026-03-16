#!/usr/bin/env python3
"""Offline threshold sweep for ratio_only / abs_only / ratio_and_abs gating.

Reads hidden states from MME-Hall (fit) and POPE (eval), computes both
R(h) = ||z_orth||/||z|| and e(h) = ||z_orth|| per sample, then sweeps
thresholds to compare wrong-detection AUROC/AP and gating precision/recall.

Usage:
    python scripts/risk_gate_mode_sweep.py \
        --fit-jsonl results/hidden_states_mmehall_L18to22.jsonl \
        --eval-jsonl results/hidden_states_pope_L18to22.jsonl \
        --layers 19 \
        --pca-k 128 \
        --outdir analysis_risk_gate_mode
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
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_hidden(jsonl_path: str):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("summary"):
                continue
            rows.append(obj)
    if not rows:
        raise ValueError(f"No data in {jsonl_path}")
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


def compute_risk_and_abs(fit_X, fit_y, eval_X, pca_k=128):
    """Return (ratio [N], abs [N]) for eval samples."""
    Xc = fit_X[fit_y == 1]
    mu = Xc.mean(axis=0)
    Xc_c = Xc - mu
    k = min(pca_k, len(Xc) - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc_c, full_matrices=False)
    Uk = Vt[:k].T  # [d, k]

    z = eval_X - mu
    z_proj = (z @ Uk) @ Uk.T
    z_orth = z - z_proj
    z_norm = np.linalg.norm(z, axis=1).clip(min=1e-8)
    orth_norm = np.linalg.norm(z_orth, axis=1)

    ratio = orth_norm / z_norm   # R(h)
    return ratio, orth_norm      # ratio, abs


def sweep_threshold(scores, y_wrong, n_points=200):
    """Sweep threshold, return list of {tau, precision, recall, f1, n_gated}."""
    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.linspace(lo, hi, n_points)
    results = []
    for tau in thresholds:
        gated = scores > tau
        n_gated = int(gated.sum())
        if n_gated == 0:
            results.append({"tau": float(tau), "precision": 0.0, "recall": 0.0,
                            "f1": 0.0, "n_gated": 0, "gate_rate": 0.0})
            continue
        tp = int((gated & (y_wrong == 1)).sum())
        fp = int((gated & (y_wrong == 0)).sum())
        fn = int((~gated & (y_wrong == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        results.append({
            "tau": float(tau),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "n_gated": n_gated,
            "gate_rate": float(n_gated / len(scores)),
        })
    return results


def lr_cv(feats, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(feats, y):
        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(sc.fit_transform(feats[tr]), y[tr])
        oof[va] = clf.predict_proba(sc.transform(feats[va]))[:, 1]
    return oof


def analyse_layer(fit_X, fit_y, eval_X, eval_y, layer, pca_k):
    y_wrong = 1 - eval_y
    ratio, orth_abs = compute_risk_and_abs(fit_X, fit_y, eval_X, pca_k)

    # Individual AUROCs (higher score = more wrong)
    ratio_auroc = roc_auc_score(y_wrong, ratio)
    ratio_ap = average_precision_score(y_wrong, ratio)
    abs_auroc = roc_auc_score(y_wrong, orth_abs)
    abs_ap = average_precision_score(y_wrong, orth_abs)

    # Combined via logistic CV
    feat_2d = np.column_stack([ratio, orth_abs])
    oof_2d = lr_cv(feat_2d, y_wrong)
    combined_auroc = roc_auc_score(y_wrong, oof_2d)
    combined_ap = average_precision_score(y_wrong, oof_2d)

    # Correlation
    corr = float(np.corrcoef(ratio, orth_abs)[0, 1])

    # Threshold sweeps
    ratio_sweep = sweep_threshold(ratio, y_wrong)
    abs_sweep = sweep_threshold(orth_abs, y_wrong)

    # For ratio_and_abs: sweep ratio threshold while fixing abs at median,
    # then sweep abs threshold while fixing ratio at median
    ratio_med = float(np.median(ratio))
    abs_med = float(np.median(orth_abs))

    # AND-gate at various percentiles
    and_results = []
    for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
        tau_r = float(np.percentile(ratio, pct))
        tau_a = float(np.percentile(orth_abs, pct))
        gate_ratio = ratio > tau_r
        gate_abs = orth_abs > tau_a
        gate_and = gate_ratio & gate_abs
        gate_or = gate_ratio | gate_abs
        for name, mask in [("ratio_only", gate_ratio), ("abs_only", gate_abs),
                           ("ratio_and_abs", gate_and), ("ratio_or_abs", gate_or)]:
            n_gated = int(mask.sum())
            if n_gated == 0:
                and_results.append({"pct": pct, "mode": name, "precision": 0, "recall": 0,
                                    "f1": 0, "n_gated": 0, "gate_rate": 0})
                continue
            tp = int((mask & (y_wrong == 1)).sum())
            fp = int((mask & (y_wrong == 0)).sum())
            fn = int((~mask & (y_wrong == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            and_results.append({
                "pct": pct, "mode": name,
                "tau_ratio": float(tau_r), "tau_abs": float(tau_a),
                "precision": float(prec), "recall": float(rec),
                "f1": float(f1), "n_gated": n_gated,
                "gate_rate": float(n_gated / len(y_wrong)),
            })

    return {
        "layer": layer,
        "n_eval": len(eval_y),
        "n_wrong": int((eval_y == 0).sum()),
        "ratio_auroc": ratio_auroc, "ratio_ap": ratio_ap,
        "abs_auroc": abs_auroc, "abs_ap": abs_ap,
        "combined_auroc": combined_auroc, "combined_ap": combined_ap,
        "corr_ratio_abs": corr,
        "ratio_mean_correct": float(ratio[eval_y == 1].mean()),
        "ratio_mean_wrong": float(ratio[eval_y == 0].mean()),
        "ratio_std_correct": float(ratio[eval_y == 1].std()),
        "ratio_std_wrong": float(ratio[eval_y == 0].std()),
        "abs_mean_correct": float(orth_abs[eval_y == 1].mean()),
        "abs_mean_wrong": float(orth_abs[eval_y == 0].mean()),
        "abs_std_correct": float(orth_abs[eval_y == 1].std()),
        "abs_std_wrong": float(orth_abs[eval_y == 0].std()),
        "and_gate_results": and_results,
        "_ratio": ratio, "_abs": orth_abs, "_y_wrong": y_wrong, "_eval_y": eval_y,
        "_ratio_sweep": ratio_sweep, "_abs_sweep": abs_sweep,
        "_oof_2d": oof_2d,
    }


def make_plots(results, outdir):
    for r in results:
        layer = r["layer"]
        ratio = r["_ratio"]
        orth_abs = r["_abs"]
        y_wrong = r["_y_wrong"]
        eval_y = r["_eval_y"]

        # 1. Scatter: ratio vs abs colored by correct/wrong
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ratio[eval_y == 1], orth_abs[eval_y == 1],
                   s=3, alpha=0.15, c="steelblue", label="Correct")
        ax.scatter(ratio[eval_y == 0], orth_abs[eval_y == 0],
                   s=8, alpha=0.5, c="coral", label="Wrong", zorder=5)
        ax.set_xlabel("R(h) = ||z_orth|| / ||z||  (ratio)")
        ax.set_ylabel("e(h) = ||z_orth||  (absolute)")
        ax.set_title(f"L{layer}: Ratio vs Absolute Residual (r={r['corr_ratio_abs']:.3f})")
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"scatter_ratio_abs_L{layer}.png", dpi=150)
        plt.close(fig)

        # 2. ROC curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for scores, name, auroc in [
            (ratio, "R(h) ratio", r["ratio_auroc"]),
            (orth_abs, "e(h) absolute", r["abs_auroc"]),
            (r["_oof_2d"], "Logistic [R,e]", r["combined_auroc"]),
        ]:
            fpr, tpr, _ = roc_curve(y_wrong, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.4f})", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"L{layer}: Wrong-Detection ROC — Ratio vs Absolute")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"roc_L{layer}.png", dpi=150)
        plt.close(fig)

        # 3. Distribution histograms (2x1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.hist(ratio[eval_y == 1], bins=80, alpha=0.6, density=True, color="steelblue", label="Correct")
        ax.hist(ratio[eval_y == 0], bins=80, alpha=0.6, density=True, color="coral", label="Wrong")
        ax.set_xlabel("R(h) = ||z_orth||/||z||")
        ax.set_title(f"Ratio distribution (AUROC={r['ratio_auroc']:.4f})")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.hist(orth_abs[eval_y == 1], bins=80, alpha=0.6, density=True, color="steelblue", label="Correct")
        ax.hist(orth_abs[eval_y == 0], bins=80, alpha=0.6, density=True, color="coral", label="Wrong")
        ax.set_xlabel("e(h) = ||z_orth||")
        ax.set_title(f"Absolute distribution (AUROC={r['abs_auroc']:.4f})")
        ax.legend(fontsize=8)

        fig.suptitle(f"L{layer}: Risk Score Distributions", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"distributions_L{layer}.png", dpi=150)
        plt.close(fig)

        # 4. Threshold sweep: F1 vs gate_rate
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for sweep, name, ax in [
            (r["_ratio_sweep"], "R(h) ratio", axes[0]),
            (r["_abs_sweep"], "e(h) absolute", axes[1]),
        ]:
            taus = [s["tau"] for s in sweep]
            f1s = [s["f1"] for s in sweep]
            precs = [s["precision"] for s in sweep]
            recs = [s["recall"] for s in sweep]
            rates = [s["gate_rate"] for s in sweep]
            ax.plot(taus, f1s, label="F1", lw=2)
            ax.plot(taus, precs, label="Precision", lw=1.5, ls="--")
            ax.plot(taus, recs, label="Recall", lw=1.5, ls="--")
            ax2 = ax.twinx()
            ax2.plot(taus, rates, label="Gate rate", lw=1, ls=":", color="gray")
            ax2.set_ylabel("Gate rate", color="gray")
            ax.set_xlabel(f"Threshold ({name})")
            ax.set_ylabel("Score")
            ax.set_title(f"{name}: Threshold Sweep")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

        fig.suptitle(f"L{layer}: Threshold Sweep — Ratio vs Absolute", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"threshold_sweep_L{layer}.png", dpi=150)
        plt.close(fig)

        # 5. AND-gate comparison bar chart
        and_res = r["and_gate_results"]
        pcts = sorted(set(a["pct"] for a in and_res))
        modes = ["ratio_only", "abs_only", "ratio_and_abs", "ratio_or_abs"]
        colors = {"ratio_only": "steelblue", "abs_only": "coral",
                  "ratio_and_abs": "mediumpurple", "ratio_or_abs": "olive"}

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for metric_idx, metric in enumerate(["precision", "recall", "f1"]):
            ax = axes[metric_idx]
            x = np.arange(len(pcts))
            w = 0.2
            for i, mode in enumerate(modes):
                vals = [next((a[metric] for a in and_res
                              if a["pct"] == pct and a["mode"] == mode), 0)
                        for pct in pcts]
                ax.bar(x + i * w, vals, w, label=mode, color=colors[mode])
            ax.set_xticks(x + 1.5 * w)
            ax.set_xticklabels([f"P{p}" for p in pcts])
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} by Gate Mode & Percentile")
            if metric_idx == 0:
                ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"L{layer}: Gate Mode Comparison at Matched Percentile Thresholds", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f"gate_mode_comparison_L{layer}.png", dpi=150)
        plt.close(fig)


def write_report(results, outdir, fit_path, eval_path):
    lines = []
    lines.append("=" * 78)
    lines.append("Risk Gate Mode Comparison: ratio_only vs abs_only vs ratio_and_abs")
    lines.append(f"Fit: {fit_path}")
    lines.append(f"Eval: {eval_path}")
    lines.append("=" * 78)

    for r in results:
        lines.append(f"\n--- Layer {r['layer']} ---")
        lines.append(f"  N={r['n_eval']} (wrong={r['n_wrong']})")
        lines.append(f"  Correlation(ratio, abs) = {r['corr_ratio_abs']:.4f}")
        lines.append("")
        lines.append("  Wrong-detection AUROC / AP:")
        lines.append(f"    R(h) ratio:          AUROC={r['ratio_auroc']:.4f}  AP={r['ratio_ap']:.4f}")
        lines.append(f"    e(h) absolute:       AUROC={r['abs_auroc']:.4f}  AP={r['abs_ap']:.4f}")
        lines.append(f"    Logistic [R, e]:      AUROC={r['combined_auroc']:.4f}  AP={r['combined_ap']:.4f}")
        lines.append("")
        lines.append("  Distributions:")
        lines.append(f"    R(h): correct={r['ratio_mean_correct']:.4f}+/-{r['ratio_std_correct']:.4f}  "
                     f"wrong={r['ratio_mean_wrong']:.4f}+/-{r['ratio_std_wrong']:.4f}")
        lines.append(f"    e(h): correct={r['abs_mean_correct']:.2f}+/-{r['abs_std_correct']:.2f}  "
                     f"wrong={r['abs_mean_wrong']:.2f}+/-{r['abs_std_wrong']:.2f}")
        lines.append("")

        # Gate comparison table at matched percentiles
        lines.append("  Gate comparison (matched-percentile thresholds):")
        lines.append(f"    {'Pct':>4} {'Mode':<16} {'tau_R':>7} {'tau_e':>7} "
                     f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'Rate':>6}")
        lines.append("    " + "-" * 65)
        for a in r["and_gate_results"]:
            lines.append(
                f"    {a['pct']:>4} {a['mode']:<16} "
                f"{a.get('tau_ratio', 0):>7.4f} {a.get('tau_abs', 0):>7.2f} "
                f"{a['precision']:>6.3f} {a['recall']:>6.3f} {a['f1']:>6.3f} "
                f"{a['gate_rate']:>6.3f}"
            )

    lines.append("")
    report = "\n".join(lines)
    with open(outdir / "report.txt", "w") as f:
        f.write(report)
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-jsonl", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[19])
    ap.add_argument("--pca-k", type=int, default=128)
    ap.add_argument("--outdir", default="analysis_risk_gate_mode")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fit: {args.fit_jsonl}")
    fit_data = load_hidden(args.fit_jsonl)
    print(f"Loading eval: {args.eval_jsonl}")
    eval_data = load_hidden(args.eval_jsonl)

    requested = set(str(l) for l in args.layers)
    common = sorted(requested & set(fit_data.keys()) & set(eval_data.keys()), key=int)
    print(f"Layers: {common}")

    results = []
    for L in common:
        print(f"\n=== Layer {L} ===")
        fit_X, fit_y = fit_data[L]
        eval_X, eval_y = eval_data[L]
        m = analyse_layer(fit_X, fit_y, eval_X, eval_y, int(L), args.pca_k)
        results.append(m)
        print(f"  R(h) AUROC={m['ratio_auroc']:.4f}  e(h) AUROC={m['abs_auroc']:.4f}  "
              f"Combined={m['combined_auroc']:.4f}  corr={m['corr_ratio_abs']:.4f}")

    # Save metrics
    metrics_out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nGenerating plots...")
    make_plots(results, outdir)

    print()
    write_report(results, outdir, args.fit_jsonl, args.eval_jsonl)
    print(f"\nDone. Output in {outdir}/")


if __name__ == "__main__":
    main()
