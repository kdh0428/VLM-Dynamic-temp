"""Evaluation loops for baseline and dynamic temperature strategies."""

from __future__ import annotations

import json
import logging
import gc
from typing import Any, Dict, Iterable

from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch

from .data import extract_example_fields
from .metrics import compute_accuracy
from .prompts import build_prompt, extract_docvqa_answer, extract_final_answer, make_mm_prompt
from .uncertainty import (
    compute_dynamic_temperature,
    compute_view_stability,
    estimate_image_uncertainty,
    run_cot_batch_with_logprobs,
    view_stability_to_uncertainty,
)

LOGGER = logging.getLogger(__name__)


def _chunk(iterable, size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _effective_limit(dataset, limit: int | None):
    """Resolve limit=-1 or None to full dataset length if available."""
    if limit is None or limit < 0:
        try:
            return len(dataset)
        except Exception:
            return None
    return limit


def _maybe_clear_cache(args, step: int) -> None:
    """Optionally clear Python/torch caches every N samples."""
    every = getattr(args, "clear_cache_every", 0)
    if not every or step % every != 0:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_baseline(llm: LLM, dataset: Iterable[Dict[str, Any]], args) -> Dict[str, Any]:
    """Run fixed-temperature baseline."""
    total_acc = 0.0
    n = 0
    eff_limit = _effective_limit(dataset, args.limit)
    img_min, img_max = float("inf"), float("-inf")
    cot_min, cot_max = float("inf"), float("-inf")
    # 결과 저장 경로: baseline 전용 파일 (dynamic과 덮어쓰기 방지)
    baseline_out_path = args.output_jsonl
    if getattr(args, "mode", "") != "baseline":
        baseline_out_path = f"baseline_{args.output_jsonl}"
    progress = tqdm(total=eff_limit, desc="Baseline", position=0, leave=True)
    with open(baseline_out_path, "w", encoding="utf-8") as f_out:
        for batch in _chunk(dataset, args.batch_size):
            if eff_limit is not None and n >= eff_limit:
                break
            prompts = []
            images = []
            metas = []
            gts = []
            for example in batch:
                if eff_limit is not None and n + len(prompts) >= eff_limit:
                    break
                try:
                    image, question, gt_answers, meta = extract_example_fields(example)
                    prompts.append(build_prompt(question, meta))
                    images.append(image)
                    metas.append(meta)
                    gts.append((gt_answers, question))
                except Exception as exc:  # pragma: no cover - runtime robustness
                    LOGGER.warning("Baseline example prep failed: %s", exc)
                    continue
            if not prompts:
                continue

            # CoT 한 번으로 토큰 엔트로피까지 계산
            cot_results = run_cot_batch_with_logprobs(llm, prompts, images, args)

            for idx, (prompt, image, meta, gt_question) in enumerate(zip(prompts, images, metas, gts)):
                if eff_limit is not None and n >= eff_limit:
                    break
                if idx >= len(cot_results):
                    break
                cot_result = cot_results[idx]
                gt_answers, question = gt_question
                cot_text = cot_result["text"]
                cot_uncert_raw = float(cot_result["cot_uncertainty"])
                cot_uncert = float(max(0.0, cot_uncert_raw * getattr(args, "cot_uncert_scale", 1.0)))
                if meta.get("task_type") == "docvqa":
                    cot_answer = extract_docvqa_answer(cot_text)
                else:
                    cot_answer = extract_final_answer(cot_text)

                if getattr(args, "image_uncert_method", "answer_entropy") == "view_stability":
                    if args.view_encoder is None or args.view_processor is None:
                        LOGGER.warning("View encoder not loaded; falling back to answer-entropy image uncertainty.")
                        image_uncert = estimate_image_uncertainty(
                            llm, prompt, image, args, reuse_answer=cot_answer
                        )
                    else:
                        stats = compute_view_stability(
                            args.view_encoder,
                            args.view_processor,
                            image,
                            num_views=args.view_num_views,
                            device=args.view_device,
                            normalize=not getattr(args, "no_view_normalize", False),
                        )
                        image_uncert = view_stability_to_uncertainty(stats)
                else:
                    image_uncert = estimate_image_uncertainty(
                        llm, prompt, image, args, reuse_answer=cot_answer
                    )
                raw_image_uncert = float(max(0.0, image_uncert * getattr(args, "img_uncert_scale", 1.0)))
                image_uncert = float(min(1.0, raw_image_uncert))

                task_type = meta.get("task_type", "vqa")
                pred = cot_answer
                acc_input = pred if task_type in {"mcq", "mmbench", "mmstar", "simplevqa", "docvqa", "vizwiz", "yesno"} else cot_text
                acc = compute_accuracy(acc_input, gt_answers, task_type=task_type)
                acc_binary = 1.0 if (task_type == "vizwiz" and acc >= 0.9) else acc
                if task_type == "vizwiz":
                    status = "correct" if acc >= 0.9 else "wrong"
                else:
                    status = "correct" if acc >= 0.5 else "wrong"

                record = {
                    "mode": "baseline",
                    "question_id": meta.get("question_id"),
                    "image_id": meta.get("image_id"),
                    "question": question,
                    "gt_answers": gt_answers,
                    "baseline_temperature": args.baseline_temperature,
                    "cot_uncertainty": cot_uncert,
                    "cot_uncertainty_raw": cot_uncert_raw,
                    "image_uncertainty": raw_image_uncert,
                    "image_uncertainty_clipped": image_uncert,
                    "cot_answer": cot_answer,
                    "accuracy": acc,
                    "accuracy_binary": acc_binary if task_type == "vizwiz" else acc,
                    "status": status,
                }
                f_out.write(json.dumps(record) + "\n")

                total_acc += acc_binary
                n += 1
                progress.update(1)
                progress.set_description(f"Baseline {n}/{eff_limit if eff_limit is not None else 'all'}")
                if task_type == "vizwiz":
                    LOGGER.info(
                        "Baseline %d/%d | %s | acc=%.3f | acc_bin=%.0f | pred=%s | gt=%s | cot_u=%.3f | img_u=%.3f",
                        n,
                        eff_limit if eff_limit is not None else -1,
                        status,
                        acc,
                        acc_binary,
                        pred,
                        gt_answers[0] if gt_answers else "",
                        cot_uncert,
                        raw_image_uncert,
                    )
                else:
                    LOGGER.info(
                        "Baseline %d/%d | %s | acc=%.3f | pred=%s | gt=%s | cot_u=%.3f | img_u=%.3f",
                        n,
                        eff_limit if eff_limit is not None else -1,
                        status,
                        acc,
                        pred,
                        gt_answers[0] if gt_answers else "",
                        cot_uncert,
                        raw_image_uncert,
                    )
                _maybe_clear_cache(args, n)
                if eff_limit is not None and n >= eff_limit:
                    break
        progress.close()

    overall_acc = total_acc / max(1, n)
    return {
        "mode": "baseline",
        "temperature": args.baseline_temperature,
        "overall_accuracy": overall_acc,
        "num_samples": n,
    }


def run_dynamic(llm: LLM, dataset: Iterable[Dict[str, Any]], args) -> Dict[str, Any]:
    """Run dynamic temperature strategy."""
    total_acc = 0.0
    n = 0
    eff_limit = _effective_limit(dataset, args.limit)
    progress = tqdm(total=eff_limit, desc="Dynamic", position=0, leave=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for batch in _chunk(dataset, args.batch_size):
            if eff_limit is not None and n >= eff_limit:
                break
            prompts = []
            images = []
            metas = []
            gts = []
            for example in batch:
                if eff_limit is not None and n + len(prompts) >= eff_limit:
                    break
                try:
                    image, question, gt_answers, meta = extract_example_fields(example)
                    prompts.append(build_prompt(question, meta))
                    images.append(image)
                    metas.append(meta)
                    gts.append((gt_answers, question))
                except Exception as exc:  # pragma: no cover - runtime robustness
                    LOGGER.warning("Dynamic example prep failed: %s", exc)
                    continue
            if not prompts:
                continue

            cot_results = run_cot_batch_with_logprobs(llm, prompts, images, args)

            for idx, (prompt, image, meta, gt_question) in enumerate(zip(prompts, images, metas, gts)):
                if eff_limit is not None and n >= eff_limit:
                    break
                if idx >= len(cot_results):
                    break
                cot_result = cot_results[idx]
                gt_answers, question = gt_question
                cot_text = cot_result["text"]
                cot_uncert_raw = float(cot_result["cot_uncertainty"])
                cot_uncert = float(max(0.0, cot_uncert_raw * getattr(args, "cot_uncert_scale", 1.0)))
                if meta.get("task_type") == "docvqa":
                    cot_answer = extract_docvqa_answer(cot_text)
                else:
                    cot_answer = extract_final_answer(cot_text)

                if getattr(args, "image_uncert_method", "answer_entropy") == "view_stability":
                    if args.view_encoder is None or args.view_processor is None:
                        LOGGER.warning("View encoder not loaded; falling back to answer-entropy image uncertainty.")
                        image_uncert = estimate_image_uncertainty(
                            llm, prompt, image, args, reuse_answer=cot_answer
                        )
                    else:
                        stats = compute_view_stability(
                            args.view_encoder,
                            args.view_processor,
                            image,
                            num_views=args.view_num_views,
                            device=args.view_device,
                            normalize=not getattr(args, "no_view_normalize", False),
                        )
                        image_uncert = view_stability_to_uncertainty(stats)
                else:
                    image_uncert = estimate_image_uncertainty(
                        llm, prompt, image, args, reuse_answer=cot_answer
                    )
                # 스케일 팩터 적용 (상한은 별도로 클립해 temp 계산에 사용, raw는 그대로 기록)
                raw_image_uncert = float(max(0.0, image_uncert * getattr(args, "img_uncert_scale", 1.0)))
                image_uncert_temp = float(min(1.0, raw_image_uncert))

                dyn_temp = compute_dynamic_temperature(
                    image_uncert=image_uncert_temp,
                    cot_uncert=cot_uncert,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    alpha=args.alpha,
                    beta=args.beta,
                )

                sampling_params = SamplingParams(
                    max_tokens=args.max_new_tokens,
                    temperature=dyn_temp,
                    top_p=0.9,
                    logprobs=None,
                )
                mm_prompt = make_mm_prompt(prompt, image)
                try:
                    outputs = llm.generate(
                        prompts=[mm_prompt],
                        sampling_params=sampling_params,
                        use_tqdm=args.vllm_tqdm,
                    )
                except Exception as exc:  # pragma: no cover - runtime robustness
                    LOGGER.warning("Dynamic final generation failed (skipping sample): %s", exc)
                    continue
                dyn_text = outputs[0].outputs[0].text
                task_type = meta.get("task_type", "vqa")
                if task_type == "docvqa":
                    dyn_answer = extract_docvqa_answer(dyn_text)
                else:
                    dyn_answer = extract_final_answer(dyn_text)
                acc_input = dyn_answer if task_type in {"mcq", "mmbench", "mmstar", "simplevqa", "docvqa", "vizwiz", "yesno"} else dyn_text
                acc = compute_accuracy(acc_input, gt_answers, task_type=task_type)
                acc_binary = 1.0 if (task_type == "vizwiz" and acc >= 0.9) else acc
                if task_type == "vizwiz":
                    status = "correct" if acc >= 0.9 else "wrong"
                else:
                    status = "correct" if acc >= 0.5 else "wrong"

                record = {
                    "question_id": meta.get("question_id"),
                    "image_id": meta.get("image_id"),
                    "question": question,
                    "gt_answers": gt_answers,
                    "baseline_temperature": args.baseline_temperature,
                    "cot_uncertainty": cot_uncert,
                    "cot_uncertainty_raw": cot_uncert_raw,
                    "image_uncertainty": raw_image_uncert,
                    "image_uncertainty_clipped": image_uncert_temp,
                    "dynamic_temperature": dyn_temp,
                    "cot_answer": cot_answer,
                    "dynamic_answer": dyn_answer,
                    "accuracy": acc,
                    "accuracy_binary": acc_binary if task_type == "vizwiz" else acc,
                    "status": status,
                }
                f_out.write(json.dumps(record) + "\n")

                total_acc += acc_binary
                n += 1
                progress.update(1)
                progress.set_description(f"Dynamic {n}/{eff_limit if eff_limit is not None else 'all'}")
                if task_type == "vizwiz":
                    LOGGER.info(
                        "Dynamic %d/%d | %s | acc=%.3f | acc_bin=%.0f | temp=%.3f | cot_u=%.3f | img_u=%.3f | pred=%s | gt=%s",
                        n,
                        eff_limit if eff_limit is not None else -1,
                        status,
                        acc,
                        acc_binary,
                        dyn_temp,
                        cot_uncert,
                        raw_image_uncert,
                        dyn_answer,
                        gt_answers[0] if gt_answers else "",
                    )
                else:
                    LOGGER.info(
                        "Dynamic %d/%d | %s | acc=%.3f | temp=%.3f | cot_u=%.3f | img_u=%.3f | pred=%s | gt=%s",
                        n,
                        eff_limit if eff_limit is not None else -1,
                        status,
                        acc,
                        dyn_temp,
                        cot_uncert,
                        raw_image_uncert,
                        dyn_answer,
                        gt_answers[0] if gt_answers else "",
                    )
                _maybe_clear_cache(args, n)
            if eff_limit is not None and n >= eff_limit:
                break
    progress.close()

    overall_acc = total_acc / max(1, n)
    return {
        "mode": "dynamic",
        "overall_accuracy": overall_acc,
        "num_samples": n,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "alpha": args.alpha,
        "beta": args.beta,
    }


def run_dynamictemp(llm: LLM, dataset: Iterable[Dict[str, Any]], args) -> Dict[str, Any]:
    """Run single-pass dynamic temperature via logits processor."""
    total_acc = 0.0
    n = 0
    eff_limit = _effective_limit(dataset, args.limit)
    progress = tqdm(total=eff_limit, desc="DynamicTemp", position=0, leave=True)
    out_path = args.output_jsonl
    if getattr(args, "mode", "") == "baseline+dynamictemp":
        out_path = f"dynamictemp_{args.output_jsonl}"
    with open(out_path, "w", encoding="utf-8") as f_out:
        for batch in _chunk(dataset, args.batch_size):
            if eff_limit is not None and n >= eff_limit:
                break
            prompts = []
            images = []
            metas = []
            gts = []
            for example in batch:
                if eff_limit is not None and n + len(prompts) >= eff_limit:
                    break
                try:
                    image, question, gt_answers, meta = extract_example_fields(example)
                    prompts.append(build_prompt(question, meta))
                    images.append(image)
                    metas.append(meta)
                    gts.append((gt_answers, question))
                except Exception as exc:  # pragma: no cover - runtime robustness
                    LOGGER.warning("DynamicTemp example prep failed: %s", exc)
                    continue
            if not prompts:
                continue

            extra_args = {
                "dyn_temp": True,
                "t_min": args.t_min,
                "t_max": args.t_max,
                "mode": args.dyn_temp_mode,
                "target_entropy": args.dyn_temp_target_entropy,
                "entropy_gain": args.dyn_temp_entropy_gain,
                "max_steps": args.dyn_temp_max_steps or args.max_new_tokens,
            }
            sampling_params = SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=0.9,
                logprobs=None,
                extra_args=extra_args,
            )

            mm_prompts = [make_mm_prompt(p, img) for p, img in zip(prompts, images)]
            try:
                outputs = llm.generate(
                    prompts=mm_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=args.vllm_tqdm,
                )
            except Exception as exc:  # pragma: no cover - runtime robustness
                LOGGER.warning("DynamicTemp generation failed: %s", exc)
                continue

            for out, meta, gt_question in zip(outputs, metas, gts):
                text = out.outputs[0].text
                task_type = meta.get("task_type", "vqa")
                if task_type == "docvqa":
                    pred = extract_docvqa_answer(text)
                else:
                    pred = extract_final_answer(text)
                gt_answers, question = gt_question
                acc_input = pred if task_type in {"mcq", "mmbench", "mmstar", "simplevqa", "docvqa", "vizwiz", "yesno"} else text
                acc = compute_accuracy(acc_input, gt_answers, task_type=task_type)
                acc_binary = 1.0 if (task_type == "vizwiz" and acc >= 0.9) else acc
                if task_type == "vizwiz":
                    status = "correct" if acc >= 0.9 else "wrong"
                else:
                    status = "correct" if acc >= 0.5 else "wrong"

                record = {
                    "mode": "dynamictemp",
                    "question_id": meta.get("question_id"),
                    "image_id": meta.get("image_id"),
                    "question": question,
                    "gt_answers": gt_answers,
                    "t_min": args.t_min,
                    "t_max": args.t_max,
                    "dyn_temp_mode": args.dyn_temp_mode,
                    "pred": pred,
                    "accuracy": acc,
                    "accuracy_binary": acc_binary if task_type == "vizwiz" else acc,
                    "status": status,
                }
                f_out.write(json.dumps(record) + "\n")

                total_acc += acc_binary
                n += 1
                progress.update(1)
                progress.set_description(f"DynamicTemp {n}/{eff_limit if eff_limit is not None else 'all'}")
                if task_type == "vizwiz":
                    LOGGER.info(
                        "DynamicTemp %d/%d | %s | acc=%.3f | acc_bin=%.0f | pred=%s | gt=%s",
                        n,
                        eff_limit if eff_limit is not None else -1,
                        status,
                        acc,
                        acc_binary,
                        pred,
                        gt_answers[0] if gt_answers else "",
                    )
                else:
                    LOGGER.info(
                        "DynamicTemp %d/%d | %s | acc=%.3f | pred=%s | gt=%s",
                        n,
                        eff_limit if eff_limit is not None else -1,
                        status,
                        acc,
                        pred,
                        gt_answers[0] if gt_answers else "",
                    )
                _maybe_clear_cache(args, n)
            if eff_limit is not None and n >= eff_limit:
                break
    progress.close()

    overall_acc = total_acc / max(1, n)
    return {
        "mode": "dynamictemp",
        "overall_accuracy": overall_acc,
        "num_samples": n,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "dyn_temp_mode": args.dyn_temp_mode,
    }
