#!/usr/bin/env python3
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_layers(s: str) -> List[int]:
    out = []
    for t in (s or "").split(","):
        t = t.strip()
        if t:
            out.append(int(t))
    if not out:
        raise ValueError("empty --layers")
    return sorted(set(out))


def load_hidden_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("summary"):
                continue
            rows.append(o)
    if not rows:
        raise ValueError("no sample rows in hidden jsonl")
    df = pd.DataFrame(rows)
    if "correct" not in df.columns:
        raise ValueError("hidden jsonl must include `correct`")
    return df


def layer_matrix(df: pd.DataFrame, layer: int, dtype=np.float32) -> np.ndarray:
    key = str(layer)
    arr = []
    for d in df["hidden_step0_finaltoken"].tolist():
        v = d.get(key)
        if v is None:
            raise ValueError(f"missing layer {layer}")
        arr.append(np.asarray(v, dtype=dtype))
    x = np.stack(arr, axis=0)
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite values in layer {layer}")
    return x


def fit_correct_subspace(hidden_states: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    centered PCA
      mu = mean(h)
      z = h - mu
      z ~= U_k U_k^T z
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [N, d]")
    n, d = hidden_states.shape
    if n < 2:
        raise ValueError("need at least 2 samples")
    k_eff = int(min(k, n, d))
    if k_eff < 1:
        raise ValueError("k must be >= 1")

    x = torch.as_tensor(hidden_states, dtype=torch.float32)
    mu = x.mean(dim=0)  # [d]
    z = x - mu[None, :]  # [N,d]

    # z = U S V^T ; principal basis in feature space is V[:, :k]
    _, s, vh = torch.linalg.svd(z, full_matrices=False)
    uk = vh[:k_eff, :].T.contiguous()  # [d,k]
    evals = (s[:k_eff] ** 2) / max(1, (n - 1))
    evr = evals / torch.clamp(evals.sum(), min=1e-12)
    meta = {
        "num_samples": int(n),
        "feature_dim": int(d),
        "k": int(k_eff),
        "singular_values": s[:k_eff].cpu().tolist(),
        "explained_variance": evals.cpu().tolist(),
        "explained_variance_ratio": evr.cpu().tolist(),
    }
    return mu, uk, meta


def compute_residual_score_batch(
    h: torch.Tensor,
    mu: torch.Tensor,
    uk: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Correct centered residual:
      z = h - mu
      z_proj = U U^T z
      z_res = z - z_proj
    """
    if h.ndim == 1:
        h = h.unsqueeze(0)
    z = h - mu.unsqueeze(0)  # [B,d]
    z_proj = (z @ uk) @ uk.T  # [B,d]
    z_res = z - z_proj

    z_norm = torch.linalg.norm(z, dim=1)
    proj_norm = torch.linalg.norm(z_proj, dim=1)
    res_norm = torch.linalg.norm(z_res, dim=1)
    r_abs = res_norm
    r_rel = res_norm / torch.clamp(z_norm, min=eps)
    return {
        "r_abs": r_abs,
        "r_rel": r_rel,
        "z_norm": z_norm,
        "proj_norm": proj_norm,
        "res_norm": res_norm,
    }

