# pip install "datasets" "Pillow" "tqdm"
"""Evaluate lmms-lab/MMBench_EN (dev split) with a custom predict_answer."""

from __future__ import annotations

import argparse
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def predict_answer(question: str, options: Dict[str, str], image: Any) -> str:
    """Replace this stub with your model call; must return one of 'A'/'B'/'C'/'D'."""
    raise NotImplementedError("Implement your model inference here.")


def to_pil(img_any: Any) -> Image.Image:
    """Convert dataset image field to PIL.Image.Image."""
    if isinstance(img_any, Image.Image):
        return img_any
    if isinstance(img_any, dict) and "bytes" in img_any:
        return Image.open(BytesIO(img_any["bytes"]))
    if isinstance(img_any, str):
        return Image.open(img_any)
    return Image.fromarray(np.array(img_any))


def normalize_letter(letter: str) -> str:
    return (letter or "").strip().upper()[:1]


def build_question(question: str, hint: Any) -> str:
    """Combine hint (if available) with question."""
    if hint is None:
        return question
    # Some hints are NaN floats
    if isinstance(hint, float) and np.isnan(hint):
        return question
    hint_str = str(hint).strip()
    if not hint_str:
        return question
    return f"{hint_str}\n{question}"


def evaluate(dataset_id: str, split: str, limit: int | None = None) -> Tuple[int, int]:
    ds = load_dataset(dataset_id, split=split)
    iterator = ds if limit is None else ds.select(range(min(limit, len(ds))))
    total = 0
    correct = 0

    for ex in tqdm(iterator, desc="Evaluating"):
        question = build_question(ex.get("question", ""), ex.get("hint"))
        options = {k: ex.get(k, "") for k in ["A", "B", "C", "D"]}
        gt = normalize_letter(ex.get("answer", ""))
        image = to_pil(ex.get("image"))

        pred = normalize_letter(predict_answer(question, options, image))
        if pred == gt:
            correct += 1
        total += 1
    return correct, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MMBench_EN dev split accuracy.")
    parser.add_argument("--dataset-id", type=str, default="lmms-lab/MMBench_EN")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to evaluate (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    correct, total = evaluate(args.dataset_id, args.split, limit=args.limit)
    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
