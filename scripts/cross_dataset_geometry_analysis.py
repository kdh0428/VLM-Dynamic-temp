#!/usr/bin/env python3
"""Cross-dataset hidden-state geometry analysis.

Fits geometry from a calibration source (e.g. MME-Hall) and evaluates
how well it explains correct/wrong differences on a target dataset
(e.g. POPE full).  Produces per-layer metrics, plots, and a summary.

Usage:
    python scripts/cross_dataset_geometry_analysis.py \
        --fit-jsonl results/hidden_states_mmehall_L18to22.jsonl \
        --eval-jsonl results/hidden_states_pope_L18to22.jsonl \
        --outdir analysis_cross_geometry
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
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


# ── Per-layer geometry computation ────────────────────────────────────────

def analyse_layer(fit_X, fit_y, eval_X, eval_y, layer: int, pca_k: int = 64):
    """Compute all geometry metrics for one layer."""
    m = {}
    m["layer"] = layer

    # ── Fit statistics (from calibration source) ──
    Xc = fit_X[fit_y == 1]
    Xw = fit_X[fit_y == 0]
    m["fit_n_correct"] = len(Xc)
    m["fit_n_wrong"] = len(Xw)

    mu_c = Xc.mean(axis=0)
    mu_w = Xw.mean(axis=0)
    midpoint = (mu_c + mu_w) / 2.0
    diff = mu_c - mu_w
    diff_norm = float(np.linalg.norm(diff))
    m["mu_diff_norm"] = diff_norm
    v_hat = diff / max(diff_norm, 1e-12)

    # ── 1. Centroid difference / projection ──
    # Project eval data onto contrastive axis
    eval_proj = (eval_X - midpoint) @ v_hat  # positive = correct side
    ec = eval_proj[eval_y == 1]
    ew = eval_proj[eval_y == 0]
    m["eval_proj_correct_mean"] = float(ec.mean())
    m["eval_proj_correct_std"] = float(ec.std())
    m["eval_proj_wrong_mean"] = float(ew.mean())
    m["eval_proj_wrong_std"] = float(ew.std())

    pooled_std = np.sqrt((ec.var() * len(ec) + ew.var() * len(ew)) / (len(ec) + len(ew)))
    m["cohens_d"] = float((ec.mean() - ew.mean()) / max(pooled_std, 1e-12))
    t_stat, t_pval = stats.ttest_ind(ec, ew, equal_var=False)
    m["welch_t"] = float(t_stat)
    m["welch_p"] = float(t_pval)

    # AUROC/AP (wrong detection: score = -proj, higher = more wrong)
    auroc_proj = roc_auc_score(1 - eval_y, -eval_proj)
    ap_proj = average_precision_score(1 - eval_y, -eval_proj)
    m["proj_wrong_auroc"] = float(auroc_proj)
    m["proj_wrong_ap"] = float(ap_proj)

    # ── 2. Spread difference ──
    var_c = np.var(Xc, axis=0)  # diagonal of covariance
    var_w = np.var(Xw, axis=0)
    trace_c = float(var_c.sum())
    trace_w = float(var_w.sum())
    m["fit_trace_sigma_correct"] = trace_c
    m["fit_trace_sigma_wrong"] = trace_w
    m["fit_var_ratio"] = trace_w / max(trace_c, 1e-12)

    # Class-centered norm on eval
    eval_norm_c = np.linalg.norm(eval_X[eval_y == 1] - mu_c, axis=1)
    eval_norm_w = np.linalg.norm(eval_X[eval_y == 0] - mu_w, axis=1)
    m["eval_centered_norm_correct_mean"] = float(eval_norm_c.mean())
    m["eval_centered_norm_correct_std"] = float(eval_norm_c.std())
    m["eval_centered_norm_wrong_mean"] = float(eval_norm_w.mean())
    m["eval_centered_norm_wrong_std"] = float(eval_norm_w.std())

    # ── 3. Diagonal Mahalanobis to fit-correct distribution ──
    var_c_safe = np.maximum(var_c, 1e-8)
    # distance for each eval sample to fit-correct centroid
    delta = eval_X - mu_c
    mahal = np.sqrt((delta ** 2 / var_c_safe).sum(axis=1))
    mahal_c = mahal[eval_y == 1]
    mahal_w = mahal[eval_y == 0]
    m["mahal_correct_mean"] = float(mahal_c.mean())
    m["mahal_correct_std"] = float(mahal_c.std())
    m["mahal_wrong_mean"] = float(mahal_w.mean())
    m["mahal_wrong_std"] = float(mahal_w.std())
    # wrong detection: higher mahal = more wrong
    auroc_mahal = roc_auc_score(1 - eval_y, mahal)
    ap_mahal = average_precision_score(1 - eval_y, mahal)
    m["mahal_wrong_auroc"] = float(auroc_mahal)
    m["mahal_wrong_ap"] = float(ap_mahal)

    # ── 4. Class-conditional log-likelihood (diagonal Gaussian) ──
    var_w_safe = np.maximum(var_w, 1e-8)
    d = fit_X.shape[1]
    # log p(x|class) = -0.5 * [d*log(2pi) + sum(log(var_i)) + sum((x-mu)^2/var_i)]
    const = -0.5 * d * np.log(2 * np.pi)
    logdet_c = -0.5 * np.log(var_c_safe).sum()
    logdet_w = -0.5 * np.log(var_w_safe).sum()

    delta_c = eval_X - mu_c
    delta_w = eval_X - mu_w
    ll_c = const + logdet_c - 0.5 * (delta_c ** 2 / var_c_safe).sum(axis=1)
    ll_w = const + logdet_w - 0.5 * (delta_w ** 2 / var_w_safe).sum(axis=1)
    llr = ll_c - ll_w  # positive = more likely correct

    llr_c = llr[eval_y == 1]
    llr_w = llr[eval_y == 0]
    m["llr_correct_mean"] = float(llr_c.mean())
    m["llr_wrong_mean"] = float(llr_w.mean())
    # wrong detection: lower llr = more wrong
    auroc_llr = roc_auc_score(1 - eval_y, -llr)
    ap_llr = average_precision_score(1 - eval_y, -llr)
    m["llr_wrong_auroc"] = float(auroc_llr)
    m["llr_wrong_ap"] = float(ap_llr)

    # ── 5. Direction difference: PCA subspace overlap ──
    k = min(pca_k, len(Xc) - 1, len(Xw) - 1, d)
    Uc = _top_k_pca(Xc, k)  # [d, k]
    Uw = _top_k_pca(Xw, k)  # [d, k]
    # Principal angles via SVD of Uc^T Uw
    M = Uc.T @ Uw
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    angles = np.arccos(svals)
    m["pca_k"] = int(k)
    m["principal_angle_mean_deg"] = float(np.degrees(angles.mean()))
    m["principal_angle_median_deg"] = float(np.degrees(np.median(angles)))
    m["principal_angle_max_deg"] = float(np.degrees(angles.max()))
    m["subspace_overlap"] = float(svals.mean())  # 1 = identical subspaces

    # ── 6. Nearest-neighbor margin ──
    nn_k = min(5, len(Xc), len(Xw))
    nn_c = NearestNeighbors(n_neighbors=nn_k, metric="euclidean").fit(Xc)
    nn_w = NearestNeighbors(n_neighbors=nn_k, metric="euclidean").fit(Xw)
    dist_to_c, _ = nn_c.kneighbors(eval_X)
    dist_to_w, _ = nn_w.kneighbors(eval_X)
    avg_dist_c = dist_to_c.mean(axis=1)
    avg_dist_w = dist_to_w.mean(axis=1)
    nn_margin = avg_dist_w - avg_dist_c  # positive = closer to correct
    nn_c_vals = nn_margin[eval_y == 1]
    nn_w_vals = nn_margin[eval_y == 0]
    m["nn_margin_correct_mean"] = float(nn_c_vals.mean())
    m["nn_margin_wrong_mean"] = float(nn_w_vals.mean())
    # wrong detection: lower margin = more wrong
    auroc_nn = roc_auc_score(1 - eval_y, -nn_margin)
    ap_nn = average_precision_score(1 - eval_y, -nn_margin)
    m["nn_wrong_auroc"] = float(auroc_nn)
    m["nn_wrong_ap"] = float(ap_nn)

    # Store arrays for plotting
    m["_eval_proj"] = eval_proj
    m["_eval_y"] = eval_y
    m["_mahal"] = mahal
    m["_llr"] = llr
    m["_nn_margin"] = nn_margin

    return m


def _top_k_pca(X, k):
    """Return top-k PCA directions [d, k] for centered X."""
    X_centered = X - X.mean(axis=0)
    if X_centered.shape[0] < k:
        k = X_centered.shape[0]
    # economy SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return Vt[:k].T  # [d, k]


# ── Plotting ──────────────────────────────────────────────────────────────

def make_plots(results, outdir: Path):
    layers = [r["layer"] for r in results]

    # 1. Layer-wise AUROC bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["proj_wrong_auroc", "mahal_wrong_auroc", "llr_wrong_auroc", "nn_wrong_auroc"]
    labels = ["Projection", "Mahalanobis", "Log-likelihood", "NN margin"]
    x = np.arange(len(layers))
    w = 0.18
    for i, (met, lab) in enumerate(zip(metrics, labels)):
        vals = [r[met] for r in results]
        ax.bar(x + i * w, vals, w, label=lab)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Wrong-detection AUROC")
    ax.set_title("MME-Hall fit → POPE eval: Wrong-detection AUROC by layer")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "auroc_by_layer.png", dpi=150)
    plt.close(fig)

    # 2. Projection histogram for best layer
    best = max(results, key=lambda r: r["proj_wrong_auroc"])
    proj = best["_eval_proj"]
    y = best["_eval_y"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proj[y == 1], bins=80, alpha=0.6, density=True, label="Correct", color="steelblue")
    ax.hist(proj[y == 0], bins=80, alpha=0.6, density=True, label="Wrong", color="coral")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Contrastive projection s")
    ax.set_ylabel("Density")
    ax.set_title(f"Projection histogram (L{best['layer']}, AUROC={best['proj_wrong_auroc']:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "projection_histogram.png", dpi=150)
    plt.close(fig)

    # 3. Mahalanobis distribution
    mahal = best["_mahal"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mahal[y == 1], bins=80, alpha=0.6, density=True, label="Correct", color="steelblue")
    ax.hist(mahal[y == 0], bins=80, alpha=0.6, density=True, label="Wrong", color="coral")
    ax.set_xlabel("Diagonal Mahalanobis distance to fit-correct")
    ax.set_ylabel("Density")
    ax.set_title(f"Mahalanobis distribution (L{best['layer']}, AUROC={best['mahal_wrong_auroc']:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mahalanobis_histogram.png", dpi=150)
    plt.close(fig)

    # 4. Spread / variance comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    trace_c = [r["fit_trace_sigma_correct"] for r in results]
    trace_w = [r["fit_trace_sigma_wrong"] for r in results]
    var_rat = [r["fit_var_ratio"] for r in results]

    axes[0].bar(x - 0.15, trace_c, 0.3, label="Correct", color="steelblue")
    axes[0].bar(x + 0.15, trace_w, 0.3, label="Wrong", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"L{l}" for l in layers])
    axes[0].set_ylabel("tr(Sigma)")
    axes[0].set_title("Covariance trace (MME-Hall fit)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, var_rat, 0.5, color="mediumpurple")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"L{l}" for l in layers])
    axes[1].set_ylabel("tr(Σ_wrong) / tr(Σ_correct)")
    axes[1].set_title("Variance ratio by layer")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)

    fig.tight_layout()
    fig.savefig(outdir / "spread_comparison.png", dpi=150)
    plt.close(fig)

    # 5. Cohen's d by layer
    fig, ax = plt.subplots(figsize=(8, 4))
    ds = [r["cohens_d"] for r in results]
    ax.bar(x, ds, 0.5, color="teal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Cohen's d")
    ax.set_title("Projection separation (Cohen's d, MME-Hall axis → POPE eval)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "cohens_d_by_layer.png", dpi=150)
    plt.close(fig)


# ── Summary report ────────────────────────────────────────────────────────

def write_report(results, outdir: Path, fit_path: str, eval_path: str):
    auroc_metrics = [
        ("proj_wrong_auroc", "Contrastive projection"),
        ("mahal_wrong_auroc", "Diagonal Mahalanobis"),
        ("llr_wrong_auroc", "Log-likelihood ratio"),
        ("nn_wrong_auroc", "NN margin"),
    ]

    lines = []
    lines.append("=" * 70)
    lines.append("Cross-dataset hidden geometry analysis")
    lines.append(f"Fit source:  {fit_path}")
    lines.append(f"Eval source: {eval_path}")
    lines.append("=" * 70)
    lines.append("")

    # Per-layer table
    lines.append(f"{'Layer':>5} {'Proj':>7} {'Mahal':>7} {'LLR':>7} {'NN':>7} {'Cohen_d':>8} {'||Δμ||':>8} {'VarRat':>7}")
    lines.append("-" * 60)
    for r in results:
        lines.append(
            f"  L{r['layer']:<3} "
            f"{r['proj_wrong_auroc']:>7.4f} "
            f"{r['mahal_wrong_auroc']:>7.4f} "
            f"{r['llr_wrong_auroc']:>7.4f} "
            f"{r['nn_wrong_auroc']:>7.4f} "
            f"{r['cohens_d']:>8.3f} "
            f"{r['mu_diff_norm']:>8.2f} "
            f"{r['fit_var_ratio']:>7.3f}"
        )
    lines.append("")

    # Best per metric
    lines.append("Strongest layer per metric (wrong-detection AUROC):")
    for key, name in auroc_metrics:
        best = max(results, key=lambda r: r[key])
        lines.append(f"  {name:<30} → L{best['layer']} (AUROC={best[key]:.4f})")
    lines.append("")

    # Overall best
    all_aurocs = []
    for r in results:
        for key, name in auroc_metrics:
            all_aurocs.append((r[key], name, r["layer"]))
    all_aurocs.sort(reverse=True)
    lines.append("Top-5 (metric, layer) by wrong-detection AUROC:")
    for auroc, name, layer in all_aurocs[:5]:
        lines.append(f"  {name:<30} L{layer}  AUROC={auroc:.4f}")
    lines.append("")

    # Interpretation
    lines.append("Summary:")
    best_metric, best_name, best_layer = all_aurocs[0]
    lines.append(
        f"  Best wrong-detection signal: {best_name} at L{best_layer} "
        f"(AUROC={best_metric:.4f})."
    )
    # Check cross-dataset transfer
    best_proj = max(results, key=lambda r: r["proj_wrong_auroc"])
    lines.append(
        f"  MME-Hall contrastive axis transfers to POPE with "
        f"AUROC={best_proj['proj_wrong_auroc']:.4f} (L{best_proj['layer']}), "
        f"Cohen's d={best_proj['cohens_d']:.3f}."
    )
    best_mahal = max(results, key=lambda r: r["mahal_wrong_auroc"])
    lines.append(
        f"  Mahalanobis outlier detection: AUROC={best_mahal['mahal_wrong_auroc']:.4f} "
        f"(L{best_mahal['layer']})."
    )
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
    ap.add_argument("--outdir", default="analysis_cross_geometry")
    ap.add_argument("--pca-k", type=int, default=32)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fit data: {args.fit_jsonl}")
    fit_data = load_hidden(args.fit_jsonl)
    print(f"Loading eval data: {args.eval_jsonl}")
    eval_data = load_hidden(args.eval_jsonl)

    common_layers = sorted(set(fit_data.keys()) & set(eval_data.keys()), key=int)
    print(f"Common layers: {common_layers}")

    results = []
    for L in common_layers:
        print(f"\n=== Layer {L} ===")
        fit_X, fit_y = fit_data[L]
        eval_X, eval_y = eval_data[L]
        print(f"  fit: {len(fit_y)} samples | eval: {len(eval_y)} samples")
        m = analyse_layer(fit_X, fit_y, eval_X, eval_y, int(L), pca_k=args.pca_k)
        results.append(m)
        print(f"  proj AUROC={m['proj_wrong_auroc']:.4f}  mahal AUROC={m['mahal_wrong_auroc']:.4f}  "
              f"llr AUROC={m['llr_wrong_auroc']:.4f}  nn AUROC={m['nn_wrong_auroc']:.4f}")

    # Save metrics (strip numpy arrays)
    metrics_out = []
    for r in results:
        metrics_out.append({k: v for k, v in r.items() if not k.startswith("_")})
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # CSV
    if metrics_out:
        keys = [k for k in metrics_out[0].keys()]
        with open(outdir / "metrics.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for row in metrics_out:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    # Plots
    print("\nGenerating plots...")
    make_plots(results, outdir)

    # Report
    print()
    write_report(results, outdir, args.fit_jsonl, args.eval_jsonl)

    print(f"\nDone. Output in {outdir}/")


if __name__ == "__main__":
    main()
