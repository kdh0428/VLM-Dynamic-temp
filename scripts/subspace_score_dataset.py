#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from correct_subspace_common import (
    compute_residual_score_batch,
    layer_matrix,
    load_hidden_jsonl,
    parse_layers,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-jsonl", required=True, help="dataset to score")
    ap.add_argument("--basis-dir", required=True, help="from subspace_fit_basis.py")
    ap.add_argument("--layers", default="", help="override basis layers; default from basis meta files")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    basis_dir = Path(args.basis_dir)
    if not basis_dir.exists():
        raise FileNotFoundError(args.basis_dir)

    if args.layers.strip():
        layers = parse_layers(args.layers)
    else:
        layers = []
        for p in sorted(basis_dir.glob("basis_Uk_layer*.pt")):
            s = p.stem.replace("basis_Uk_layer", "")
            layers.append(int(s))
        layers = sorted(set(layers))
        if not layers:
            raise ValueError("no basis_Uk_layer*.pt found in basis-dir")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    df = load_hidden_jsonl(args.hidden_jsonl).copy()
    out = df[["sample_id", "category", "gt", "pred", "correct"]].copy()
    print(f"[score] hidden={args.hidden_jsonl} n={len(df)} layers={layers} device={device}")

    rel_cols = []
    abs_cols = []

    for L in layers:
        mu = torch.load(basis_dir / f"mu_layer{L}.pt", map_location=device).float()
        uk = torch.load(basis_dir / f"basis_Uk_layer{L}.pt", map_location=device).float()
        x = layer_matrix(df, L).astype(np.float32)  # [N,d]

        r_abs_all = []
        r_rel_all = []
        z_norm_all = []
        proj_norm_all = []
        res_norm_all = []

        n = x.shape[0]
        for st in range(0, n, int(args.batch_size)):
            ed = min(n, st + int(args.batch_size))
            xb = torch.from_numpy(x[st:ed]).to(device=device, dtype=torch.float32)
            sc = compute_residual_score_batch(xb, mu=mu, uk=uk, eps=float(args.eps))
            r_abs_all.append(sc["r_abs"].detach().cpu().numpy())
            r_rel_all.append(sc["r_rel"].detach().cpu().numpy())
            z_norm_all.append(sc["z_norm"].detach().cpu().numpy())
            proj_norm_all.append(sc["proj_norm"].detach().cpu().numpy())
            res_norm_all.append(sc["res_norm"].detach().cpu().numpy())

        r_abs = np.concatenate(r_abs_all, axis=0)
        r_rel = np.concatenate(r_rel_all, axis=0)
        z_norm = np.concatenate(z_norm_all, axis=0)
        proj_norm = np.concatenate(proj_norm_all, axis=0)
        res_norm = np.concatenate(res_norm_all, axis=0)

        c_abs = f"r_abs_layer{L}"
        c_rel = f"r_rel_layer{L}"
        out[c_abs] = r_abs
        out[c_rel] = r_rel
        out[f"z_norm_layer{L}"] = z_norm
        out[f"proj_norm_layer{L}"] = proj_norm
        out[f"res_norm_layer{L}"] = res_norm
        rel_cols.append(c_rel)
        abs_cols.append(c_abs)
        print(f"[score] layer={L} done | mean_r_rel={float(r_rel.mean()):.6f}")

    out["r_abs_fused_mean"] = out[abs_cols].mean(axis=1)
    out["r_rel_fused_mean"] = out[rel_cols].mean(axis=1)
    out["is_error"] = (1 - out["correct"].astype(int)).astype(int)
    out["basis_dir"] = str(basis_dir)
    out["source_hidden_jsonl"] = args.hidden_jsonl

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[done] out_csv={out_csv}")


if __name__ == "__main__":
    main()

