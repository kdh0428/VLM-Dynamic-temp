#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from correct_subspace_common import fit_correct_subspace, layer_matrix, load_hidden_jsonl, parse_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-jsonl", required=True, help="train split hidden jsonl")
    ap.add_argument("--layers", default="25")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--feature-source", default="step0_finaltoken")
    ap.add_argument("--token-position", default="first_answer_token_prelogit")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    df = load_hidden_jsonl(args.hidden_jsonl)
    df_c = df[df["correct"].astype(int) == 1].copy()
    if len(df_c) == 0:
        raise ValueError("no correct samples in fit set")

    print(
        f"[fit] hidden={args.hidden_jsonl} total={len(df)} correct={len(df_c)} "
        f"layers={layers} k={args.k}"
    )

    meta = {
        "hidden_jsonl": args.hidden_jsonl,
        "fit_correct_only": True,
        "layers": layers,
        "requested_k": int(args.k),
        "num_total_samples": int(len(df)),
        "num_correct_samples": int(len(df_c)),
        "feature_source": args.feature_source,
        "token_position": args.token_position,
        "per_layer": {},
    }

    for L in layers:
        x = layer_matrix(df_c, L)  # [N,d]
        mu, uk, m = fit_correct_subspace(x, k=int(args.k))
        torch.save(mu.cpu(), outdir / f"mu_layer{L}.pt")
        torch.save(uk.cpu(), outdir / f"basis_Uk_layer{L}.pt")
        meta["per_layer"][str(L)] = m
        print(
            f"[fit] layer={L} done | N={m['num_samples']} d={m['feature_dim']} "
            f"k={m['k']} evr1={m['explained_variance_ratio'][0]:.6f}"
        )

    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] outdir={outdir}")


if __name__ == "__main__":
    main()

