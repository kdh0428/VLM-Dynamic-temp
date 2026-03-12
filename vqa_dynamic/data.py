"""Dataset loading and example helpers for VQA-style datasets (VQAv2, SimpleVQA, MMBench)."""

from __future__ import annotations

import os
import logging
from io import BytesIO
import base64
import io
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, load_dataset
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _log_dataset_env(dataset_id: str, split: str, config: str | None) -> None:
    """Log datasets-related env to diagnose slow/blocked loads."""
    LOGGER.info(
        "Datasets env: datasets=%s HF_DATASETS_CACHE=%s HF_HOME=%s TRANSFORMERS_CACHE=%s",
        getattr(datasets, "__version__", "unknown"),
        os.environ.get("HF_DATASETS_CACHE"),
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
    )
    LOGGER.info("Load args: dataset_id=%s config=%s split=%s", dataset_id, config, split)


def _decode_base64_image(image_str: str) -> Image.Image:
    """Decode base64 image string to PIL.Image (handles optional data: prefix)."""
    payload = image_str
    if image_str.startswith("data:"):
        payload = image_str.split(",", 1)[-1]
    payload = payload + "=" * (-len(payload) % 4)
    img_bytes = base64.b64decode(payload, validate=False)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def load_tsv_dataset(tsv_path: str) -> Dataset:
    """Load TSV file into a datasets.Dataset."""
    LOGGER.info("Loading TSV dataset from %s", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    # If image field is a numeric id, try to map it to a base64 string from rows that carry the actual image.
    if "index" in df.columns and "image" in df.columns:
        base64_rows = df[df["image"].astype(str).str.len() > 100]
        idx_to_b64 = {
            str(row["index"]): row["image"] for _, row in base64_rows.iterrows()
        }

        def _resolve_image(val):
            if isinstance(val, str) and val.isdigit() and val in idx_to_b64:
                return idx_to_b64[val]
            return val

        df["image"] = df["image"].apply(_resolve_image)

    df["__base_dir"] = os.path.dirname(os.path.abspath(tsv_path))
    return Dataset.from_pandas(df)


def load_vqav2_dataset(
    dataset_id: str,
    split: str,
    config: str | None = None,
    mmbench_source: str | None = None,
    vizwiz_only_unanswerable: bool = False,
    mme_hall_only: bool = False,
    mme_hall_categories: str | None = None,
    hallusionbench_image_root: str | None = None,
):
    """Load dataset split via datasets or local TSV."""
    LOGGER.info("Loading dataset %s config %s split %s", dataset_id, config, split)
    _log_dataset_env(dataset_id, split, config)
    if dataset_id == "m-a-p/SimpleVQA":
        ds = load_dataset(dataset_id, split="test")

        def _map_simplevqa(ex):
            try:
                pil_img = _decode_base64_image(ex["image"])
            except Exception:
                pil_img = ex["image"]
            return {
                "image": pil_img,
                "question": ex.get("question", ""),
                "answers": [ex.get("answer", "")],
                "question_id": str(ex.get("data_id", "")),
                "task_type": "simplevqa",
                "language": ex.get("language"),
                "vqa_category": ex.get("vqa_category"),
                "original_category": ex.get("original_category"),
            }

        return ds.map(_map_simplevqa)
    if dataset_id == "lmms-lab/VizWiz-VQA":
        ds = load_dataset(dataset_id, split=split)

        def _map_vizwiz(ex):
            answers_field = ex.get("answers", [])
            if isinstance(answers_field, list):
                if answers_field and isinstance(answers_field[0], dict):
                    gt_answers = [a.get("answer", "") for a in answers_field]
                else:
                    gt_answers = [str(a) for a in answers_field]
            else:
                gt_answers = [str(answers_field)]

            return {
                "image": ex.get("image"),
                "question": ex.get("question", ""),
                "answers": gt_answers,
                "question_id": ex.get("question_id") or ex.get("image_id") or ex.get("id"),
                "task_type": "vizwiz",
            }

        ds = ds.map(_map_vizwiz)
        if vizwiz_only_unanswerable:
            LOGGER.info("Filtering VizWiz to only unanswerable samples.")

            def _is_unanswerable(ex):
                answers = ex.get("answers", [])
                if not answers:
                    return False
                return any(str(a).strip().lower() == "unanswerable" for a in answers)

            ds = ds.filter(_is_unanswerable)
        return ds
    if dataset_id == "lmms-lab/POPE":
        ds = load_dataset(dataset_id, split=split)

        def _map_pope(ex):
            ans = str(ex.get("answer", "")).strip().lower()
            return {
                "image": ex.get("image"),
                "question": ex.get("question", ""),
                "answers": [ans],
                "question_id": ex.get("question_id") or ex.get("id"),
                "task_type": "yesno",
                "category": ex.get("category"),
                "image_source": ex.get("image_source"),
            }

        return ds.map(_map_pope)
    if dataset_id in {"lmms-lab/MME", "lmms-lab/MME-Hall"}:
        ds = load_dataset("lmms-lab/MME", split=split)
        hall_cats = {"existence", "count", "position", "color"}
        if mme_hall_categories:
            hall_cats = {c.strip() for c in mme_hall_categories.split(",") if c.strip()}
        if dataset_id == "lmms-lab/MME-Hall" or mme_hall_only:
            LOGGER.info("Filtering MME to hall categories: %s", sorted(hall_cats))
            ds = ds.filter(lambda ex: ex.get("category") in hall_cats)

        def _map_mme(ex):
            ans = str(ex.get("answer", "")).strip().lower()
            return {
                "image": ex.get("image"),
                "question": ex.get("question", ""),
                "answers": [ans],
                "question_id": ex.get("question_id") or ex.get("id"),
                "task_type": "yesno",
                "category": ex.get("category"),
            }

        return ds.map(_map_mme)
    if dataset_id == "rayguan/HallusionBench":
        ds = load_dataset(dataset_id, split=split)

        def _map_hallusionbench(ex):
            gt = ex.get("gt_answer", ex.get("answer", ""))
            gt = str(gt).strip()
            if gt in {"1", "yes", "Yes", "YES"}:
                gt_norm = "yes"
            elif gt in {"0", "no", "No", "NO"}:
                gt_norm = "no"
            else:
                gt_norm = str(gt).strip().lower()

            image_val = ex.get("image")
            if image_val is None:
                filename = ex.get("filename") or ex.get("image_path")
                if filename and hallusionbench_image_root:
                    image_val = os.path.join(hallusionbench_image_root, filename)
                else:
                    image_val = filename

            return {
                "image": image_val,
                "question": ex.get("question", ""),
                "answers": [gt_norm],
                "question_id": ex.get("question_id") or ex.get("id") or ex.get("idx"),
                "task_type": "yesno",
                "category": ex.get("category"),
                "subcategory": ex.get("subcategory"),
                "figure_id": ex.get("figure_id"),
                "visual_input": ex.get("visual_input"),
            }

        ds = ds.map(_map_hallusionbench)
        if "visual_input" in ds.column_names:
            ds = ds.filter(lambda ex: str(ex.get("visual_input", "1")).strip() == "1")
        return ds
    if dataset_id == "lmms-lab/DocVQA":
        docvqa_config = config or "DocVQA"
        ds = load_dataset(dataset_id, docvqa_config, split=split)

        def _map_docvqa(ex):
            answers_field = ex.get("answers", [])
            if isinstance(answers_field, list):
                if answers_field and isinstance(answers_field[0], dict):
                    gt_answers = [a.get("answer", a.get("text", "")) for a in answers_field]
                else:
                    gt_answers = [str(a) for a in answers_field]
            elif isinstance(answers_field, dict):
                gt_answers = [str(a) for a in answers_field.get("text", [])]
            else:
                ans = ex.get("answer", answers_field)
                gt_answers = [str(ans)] if ans is not None else []

            return {
                "image": ex.get("image"),
                "question": ex.get("question", ""),
                "answers": gt_answers,
                "question_id": ex.get("question_id") or ex.get("questionId") or ex.get("id"),
                "task_type": "docvqa",
            }

        return ds.map(_map_docvqa)
    if dataset_id == "merve/vqav2-small":
        ds = load_dataset(dataset_id, split=split)

        def _map_vqav2(ex):
            ans = str(ex.get("multiple_choice_answer", "")).strip()
            return {
                "image": ex.get("image"),
                "question": ex.get("question", ""),
                "answers": [ans],
                "question_id": None,
                "task_type": "yesno" if ans.lower() in ("yes", "no") else "vqa",
            }

        ds = ds.map(_map_vqav2)
        ds = ds.filter(lambda ex: ex["task_type"] == "yesno")
        LOGGER.info("VQAv2-small yes/no filtered: %d samples", len(ds))
        return ds
    if dataset_id == "Lin-Chen/MMStar":
        ds = load_dataset(dataset_id, split="val")

        def _map_mmstar(ex):
            return {
                "image": ex.get("image"),
                "question": ex.get("question", ""),
                "answers": [ex.get("answer", "")],
                "question_id": str(ex.get("index", "")),
                "task_type": "mmstar",
                "category": ex.get("category"),
                "l2_category": ex.get("l2_category"),
                "meta_info": ex.get("meta_info"),
            }

        return ds.map(_map_mmstar)
    if dataset_id.endswith(".json"):
        if os.path.isfile(dataset_id):
            LOGGER.info("Loading local JSON dataset: %s", dataset_id)
            ds = load_dataset("json", data_files=dataset_id, split="train")
            if os.path.basename(dataset_id).lower().startswith("hallusionbench"):
                def _map_hallusionbench_local(ex):
                    gt = ex.get("gt_answer", ex.get("answer", ""))
                    gt = str(gt).strip()
                    if gt in {"1", "yes", "Yes", "YES"}:
                        gt_norm = "yes"
                    elif gt in {"0", "no", "No", "NO"}:
                        gt_norm = "no"
                    else:
                        gt_norm = str(gt).strip().lower()

                    image_val = ex.get("image")
                    if image_val is None:
                        filename = ex.get("filename") or ex.get("image_path")
                        if filename and hallusionbench_image_root:
                            image_val = os.path.join(hallusionbench_image_root, filename)
                        else:
                            image_val = filename

                    return {
                        "image": image_val,
                        "question": ex.get("question", ""),
                        "answers": [gt_norm],
                        "question_id": ex.get("question_id") or ex.get("id") or ex.get("idx"),
                        "task_type": "yesno",
                        "category": ex.get("category"),
                        "subcategory": ex.get("subcategory"),
                        "figure_id": ex.get("figure_id"),
                        "visual_input": ex.get("visual_input"),
                    }

                ds = ds.map(_map_hallusionbench_local)
                if "visual_input" in ds.column_names:
                    ds = ds.filter(lambda ex: str(ex.get("visual_input", "1")).strip() == "1")
            return ds
        LOGGER.info("Loading remote JSON via datasets 'json' builder: %s", dataset_id)
        return load_dataset("json", data_files=dataset_id, split="train")
    if dataset_id.endswith(".tsv"):
        # Local TSV file
        if os.path.isfile(dataset_id):
            return load_tsv_dataset(dataset_id)
        # Remote TSV on Hugging Face Hub or URL
        LOGGER.info("Loading remote TSV via datasets 'tsv' builder: %s", dataset_id)
        return load_dataset("tsv", data_files=dataset_id, split="train")
    if config:
        ds = load_dataset(dataset_id, config, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    if mmbench_source and "source" in ds.column_names:
        LOGGER.info("Filtering MMBench by source == %s", mmbench_source)
        ds = ds.filter(lambda ex: ex.get("source") == mmbench_source)
    return ds


def extract_example_fields(example: Dict[str, Any]) -> Tuple[Image.Image, str, List[str], Dict[str, Any]]:
    """Extract PIL image, question text, and list of ground-truth answers."""
    image_raw = example.get("image") or example.get("image_path") or example.get("img_path")
    if isinstance(image_raw, Image.Image):
        pil_image = image_raw
    elif isinstance(image_raw, dict) and "bytes" in image_raw:
        pil_image = Image.open(BytesIO(image_raw["bytes"]))
    elif isinstance(image_raw, str):
        img_path = image_raw
        pil_image = None
        # Try local/relative path first
        if os.path.isfile(img_path):
            pil_image = Image.open(img_path)
        else:
            base_dir = example.get("__base_dir")
            if base_dir:
                candidate = os.path.join(base_dir, img_path)
                if os.path.isfile(candidate):
                    pil_image = Image.open(candidate)
                else:
                    # If the field looks like an id (e.g., "1781"), try common image extensions.
                    if img_path.isdigit():
                        for ext in (".jpg", ".png", ".jpeg"):
                            cand2 = os.path.join(base_dir, img_path + ext)
                            if os.path.isfile(cand2):
                                pil_image = Image.open(cand2)
                                break
        # If still None, attempt base64-decoded image string
        if pil_image is None:
            try:
                padded = img_path + "=" * (-len(img_path) % 4)
                img_bytes = base64.b64decode(padded, validate=False)
                pil_image = Image.open(BytesIO(img_bytes))
            except Exception as exc:
                raise FileNotFoundError(f"Could not resolve image path or decode base64 for {img_path}") from exc
    else:
        pil_image = Image.fromarray(np.array(image_raw))

    question = example.get("question", "")
    answers_field = example.get("answers", [])
    if not answers_field and "answer" in example:
        answers_field = example.get("answer")

    choices: List[Tuple[str, str]] = []
    task_type = example.get("task_type", "vqa")
    gt_answers: List[str] = []

    # Multiple-choice style (e.g., MMBench)
    if example.get("options"):
        opts = example.get("options")
        if isinstance(opts, list):
            for idx, opt in enumerate(opts):
                if isinstance(opt, dict):
                    label = opt.get("label") or opt.get("letter") or opt.get("option") or opt.get("id") or chr(65 + idx)
                    text = opt.get("text") or opt.get("content") or opt.get("choice") or ""
                else:
                    label = chr(65 + idx)
                    text = str(opt)
                choices.append((str(label).upper(), str(text)))
        task_type = "mcq"
        ans = example.get("answer") or example.get("label") or example.get("correct_answer")
        if ans:
            gt_answers = [str(ans)]
        else:
            gt_answers = []
    elif any(k in example for k in ("A", "B", "C", "D")):
        # MMBench_EN style fields
        for key in ("A", "B", "C", "D"):
            if key in example:
                choices.append((key, str(example.get(key, ""))))
        task_type = "mmbench"
        ans = example.get("answer") or example.get("label") or example.get("correct_answer")
        if ans:
            gt_answers = [str(ans)]
        else:
            gt_answers = []
    else:
        if isinstance(answers_field, list):
            if answers_field and isinstance(answers_field[0], dict) and "answer" in answers_field[0]:
                gt_answers = [a.get("answer", "") for a in answers_field]
            else:
                gt_answers = [str(a) for a in answers_field]
        elif isinstance(answers_field, dict) and "text" in answers_field:
            gt_answers = [str(a) for a in answers_field.get("text", [])]
        else:
            gt_answers = [str(answers_field)]

    meta = {
        "question_id": example.get("index") or example.get("question_id") or example.get("id"),
        "image_id": example.get("image_id"),
        "choices": choices,
        "task_type": task_type,
        "hint": example.get("hint"),
        "category": example.get("category"),
        "l2_category": example.get("l2_category"),
        "meta_info": example.get("meta_info"),
    }
    return pil_image, question, gt_answers, meta
