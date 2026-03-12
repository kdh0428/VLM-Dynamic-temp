#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def risk_coverage(y: np.ndarray, score: np.ndarray, points: int = 20) -> pd.DataFrame:
    # higher score => higher risk
    n = len(y)
    order = np.argsort(-score)
    y_s = y[order]
    rows = []
    for c in np.linspace(0.05, 1.0, points):
        k = max(1, int(np.ceil(c * n)))
        err = float(y_s[:k].mean())
        rows.append({"coverage": float(c), "error_rate": err, "n_kept": int(k)})
    return pd.DataFrame(rows)


def bootstrap_ci_auc_ap(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    aucs = []
    aps = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss))
        aps.append(average_precision_score(yy, ss))
    if len(aucs) < 5:
        return (np.nan, np.nan), (np.nan, np.nan)
    return (
        (float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))),
        (float(np.quantile(aps, 0.025)), float(np.quantile(aps, 0.975))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-col", default="is_error")
    ap.add_argument("--score-cols", default="", help="comma list; default auto from r_rel/r_abs")
    ap.add_argument("--main-score", default="r_rel_fused_mean")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-scores-csv", default="", help="optional domain-shift diagnostic")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores_csv)
    if args.label_col not in df.columns:
        raise ValueError(f"missing label column: {args.label_col}")
    y = df[args.label_col].astype(int).to_numpy()

    if args.score_cols.strip():
        score_cols = [x.strip() for x in args.score_cols.split(",") if x.strip()]
    else:
        score_cols = [c for c in df.columns if c.startswith("r_rel_") or c.startswith("r_abs_")]
        score_cols = [c for c in score_cols if df[c].notna().any()]
    if not score_cols:
        raise ValueError("no score columns found")

    rows: List[Dict] = []
    for c in score_cols:
        s = df[c].astype(float).to_numpy()
        auc = float(roc_auc_score(y, s))
        apv = float(average_precision_score(y, s))
        (auc_lo, auc_hi), (ap_lo, ap_hi) = bootstrap_ci_auc_ap(y, s, args.bootstrap, args.seed)
        rows.append(
            {
                "score_col": c,
                "auc": auc,
                "auc_ci_lo": auc_lo,
                "auc_ci_hi": auc_hi,
                "ap": apv,
                "ap_ci_lo": ap_lo,
                "ap_ci_hi": ap_hi,
                "mean_correct": float(df.loc[df[args.label_col] == 0, c].mean()),
                "mean_wrong": float(df.loc[df[args.label_col] == 1, c].mean()),
            }
        )
    met = pd.DataFrame(rows).sort_values("auc", ascending=False)
    met.to_csv(outdir / "metrics.csv", index=False)

    main_score = args.main_score if args.main_score in df.columns else met.iloc[0]["score_col"]
    s_main = df[main_score].astype(float).to_numpy()

    # ROC plot
    fpr, tpr, _ = roc_curve(y, s_main)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{main_score} (AUC={roc_auc_score(y, s_main):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_main.png", dpi=180)
    plt.close()

    # PR plot
    p, r, _ = precision_recall_curve(y, s_main)
    plt.figure()
    plt.plot(r, p, label=f"{main_score} (AP={average_precision_score(y, s_main):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pr_main.png", dpi=180)
    plt.close()

    # Distribution plot
    plt.figure()
    plt.hist(df.loc[df[args.label_col] == 0, main_score], bins=40, alpha=0.6, label="correct")
    plt.hist(df.loc[df[args.label_col] == 1, main_score], bins=40, alpha=0.6, label="wrong")
    plt.xlabel(main_score)
    plt.ylabel("count")
    plt.title("Score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "score_distribution_main.png", dpi=180)
    plt.close()

    # Risk-coverage
    rc = risk_coverage(y, s_main, points=20)
    rc.to_csv(outdir / "risk_coverage_main.csv", index=False)
    plt.figure()
    plt.plot(rc["coverage"], rc["error_rate"])
    plt.xlabel("Coverage")
    plt.ylabel("Error rate in high-risk slice")
    plt.title(f"Risk-Coverage ({main_score})")
    plt.tight_layout()
    plt.savefig(outdir / "risk_coverage_main.png", dpi=180)
    plt.close()

    diag = {}
    if args.compare_scores_csv:
        df2 = pd.read_csv(args.compare_scores_csv)
        if main_score in df2.columns and "correct" in df.columns and "correct" in df2.columns:
            a = df.loc[df["correct"].astype(int) == 1, main_score].astype(float).to_numpy()
            b = df2.loc[df2["correct"].astype(int) == 1, main_score].astype(float).to_numpy()
            diag = {
                "domain_shift_main_score_correct_mean_src": float(np.mean(a)) if len(a) else np.nan,
                "domain_shift_main_score_correct_mean_cmp": float(np.mean(b)) if len(b) else np.nan,
                "domain_shift_main_score_correct_mean_diff": float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan,
            }
            pd.DataFrame([diag]).to_csv(outdir / "domain_shift_diagnostic.csv", index=False)

    report = {
        "scores_csv": args.scores_csv,
        "label_col": args.label_col,
        "n_samples": int(len(df)),
        "error_rate": float(y.mean()),
        "main_score": main_score,
        "main_auc": float(roc_auc_score(y, s_main)),
        "main_ap": float(average_precision_score(y, s_main)),
        "best_score_by_auc": str(met.iloc[0]["score_col"]),
        "best_auc": float(met.iloc[0]["auc"]),
        "domain_shift_diagnostic": diag,
    }
    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(outdir / "report.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2))
        f.write("\n")
    print(f"[done] outdir={outdir}")
    print(f"[done] metrics={outdir / 'metrics.csv'}")
    print(f"[done] report={outdir / 'report.json'}")


if __name__ == "__main__":
    main()

