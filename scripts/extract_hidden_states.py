#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vqa_dynamic.experiment_runner import _build_hf_prompt, _extract_yesno


def norm_yesno(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"yes", "true", "1"}:
        return "yes"
    if s in {"no", "false", "0"}:
        return "no"
    return s


def get_gt(example: Dict) -> str:
    answers = example.get("answers")
    if isinstance(answers, list) and answers:
        return norm_yesno(str(answers[0]))
    if isinstance(answers, str):
        return norm_yesno(answers)
    ans = example.get("answer", "")
    return norm_yesno(str(ans))


def parse_layers(s: str) -> List[int]:
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("empty --layers")
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset-id", default="lmms-lab/POPE")
    ap.add_argument("--split", default="test")
    ap.add_argument("--layers", default="15,20,25")
    ap.add_argument("--output-jsonl", default="results/hidden_states_pope.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=2)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--mme-hall-only", action="store_true")
    ap.add_argument("--mme-hall-categories", type=str, default="existence,count,position,color")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--clear-cache-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = parse_layers(args.layers)
    outp = Path(args.output_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "bf16":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.float32

    if args.dataset_id in {"lmms-lab/MME", "lmms-lab/MME-Hall"}:
        ds = load_dataset("lmms-lab/MME", split=args.split)
        hall_cats = {c.strip() for c in str(args.mme_hall_categories).split(",") if c.strip()}
        if args.dataset_id == "lmms-lab/MME-Hall" or bool(args.mme_hall_only):
            ds = ds.filter(lambda ex: ex.get("category") in hall_cats)
    else:
        ds = load_dataset(args.dataset_id, split=args.split)
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, dtype=dtype, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model.eval()

    n = 0
    n_correct = 0
    n_incorrect = 0
    n_nan = 0

    with outp.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            if args.limit >= 0 and n >= args.limit:
                break
            image = ex.get("image")
            question = ex.get("question", "")
            task_type = ex.get("task_type", "yesno")
            meta = {k: ex.get(k) for k in ex.keys() if k not in {"image", "question", "answers"}}
            meta["dataset_id"] = args.dataset_id
            prompt = _build_hf_prompt(question, task_type, meta, processor, args.model_id)

            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, return_dict=True, output_hidden_states=True)
                hs = out.hidden_states
                seq = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    temperature=0.0,
                )

            in_len = int(inputs["input_ids"].shape[1])
            gen_ids = seq[0, in_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = _extract_yesno(text)
            gt = get_gt(ex)
            correct = int(pred == gt and pred in {"yes", "no"})

            layer_map = {}
            bad = False
            for li in layers:
                if li < 0 or li >= len(hs):
                    raise ValueError(f"layer index out of range: {li} / {len(hs)-1}")
                vec = hs[li][0, -1, :].detach().float().cpu().numpy().astype(np.float16)
                if not np.isfinite(vec).all():
                    bad = True
                layer_map[str(li)] = vec.tolist()
            if bad:
                n_nan += 1

            row = {
                "sample_id": idx,
                "category": ex.get("category", "pope"),
                "question": question,
                "gt": gt,
                "pred": pred,
                "correct": int(correct),
                "layers": layers,
                "hidden_step0_finaltoken": layer_map,
            }
            f.write(json.dumps(row) + "\n")

            n += 1
            n_correct += int(correct == 1)
            n_incorrect += int(correct == 0)

            if (n % 50) == 0:
                print(f"[progress] {n} samples | correct={n_correct} incorrect={n_incorrect}")
            if args.clear_cache_every > 0 and (n % args.clear_cache_every) == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {
        "summary": True,
        "n_total": n,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_nonfinite_hidden": n_nan,
        "layers": layers,
        "output_jsonl": str(outp),
    }
    with outp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print("[done]", summary)


if __name__ == "__main__":
    main()
