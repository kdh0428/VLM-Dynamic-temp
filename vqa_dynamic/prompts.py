"""Prompt construction and parsing utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

from PIL import Image

from .metrics import normalize_answer


def _merge_hint(question: str, meta: Optional[Dict[str, Any]]) -> str:
    """Attach hint text to the question if present."""
    if not meta:
        return question
    hint = meta.get("hint")
    if hint is None:
        return question
    hint_str = str(hint).strip()
    if not hint_str or hint_str.lower() == "nan":
        return question
    return f"Hint: {hint_str}\n{question}"


def build_vqa_prompt(question: str) -> str:
    """Construct the VQA prompt with an <image> placeholder."""
    system_msg = (
        "You are a helpful visual question answering assistant. "
        "Use the hint if provided and answer concisely. "
        "If the image does not contain enough information to answer, reply with the word 'unanswerable'. "
        "Always finish with: \"Final answer: <short answer>\" on the last line."
    )
    user_msg = (
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Here is a question about the image. First reason step by step. "
        'Then, on the FINAL line, output exactly: "Final answer: <short answer>" '
        "and nothing after it.\n\n"
        f"Question: {question}"
    )
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_mmbench_prompt(question: str, choices: List[Tuple[str, str]]) -> str:
    """Construct prompt for multiple-choice MMBench-style tasks."""
    letters = ", ".join([c[0] for c in choices])
    options_lines = [f"{lbl}. {txt}" for lbl, txt in choices]
    choices_str = "Options:\n" + "\n".join(options_lines) if options_lines else "Options:\n"
    system_msg = (
        "You are a helpful visual question answering assistant. "
        "Use the hint if provided. Always choose exactly one option from the given list. "
        "Always finish with: \"Final answer: [LETTER]\" on the last line."
    )
    user_msg = (
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Answer the following multiple choice question.\n"
        f"Question: {question}\n\n{choices_str}\n\n"
        "Please select the correct answer from the options above. "
        "Then, on the FINAL line, output exactly: \"Final answer: [LETTER]\" "
        f"where [LETTER] is exactly one of {{{letters}}}. After that line, output nothing else."
    )
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_mmstar_prompt(question: str) -> str:
    """Prompt for MMStar MCQ where options are inline in the question."""
    system_msg = (
        "You are a helpful assistant. "
        "Respond with EXACTLY one line: \"Final answer: [LETTER]\" where [LETTER] is A, B, C, or D. "
        "Do not provide any explanation or extra text."
    )
    user_msg = (
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{question}\n\n"
        "Answer with a single letter (A/B/C/D) only."
    )
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_yesno_prompt(question: str) -> str:
    """Prompt for yes/no hallucination benchmarks."""
    system_msg = (
        "You are a visual question answering assistant. "
        "First, write your reasoning. "
        "Then output a final answer on a new line in exactly this format: "
        "Final answer: Yes or Final answer: No. "
        "The final answer line is mandatory. "
        "Do not use any other answer format."
    )
    user_msg = (
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"Question: {question}\n"
        "Reply with the required format. "
        "The last line must be exactly: \"Final answer: Yes\" or \"Final answer: No\"."
    )
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_docvqa_prompt(question: str) -> str:
    """Prompt for DocVQA short-span answers."""
    system_msg = "You are a helpful assistant for document visual question answering."
    user_msg = (
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"Question: {question}\n"
        "Answer with a short text span. Do not add extra explanation. "
        'On the last line, output exactly: "Final answer: <short answer>".'
    )
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt(question: str, meta: Optional[Dict[str, any]] = None) -> str:
    """Select prompt template based on meta/task."""
    meta = meta or {}
    if meta.get("task_type") == "mmstar":
        return build_mmstar_prompt(question)
    if meta.get("task_type") == "yesno":
        return build_yesno_prompt(question)
    if meta.get("task_type") == "docvqa":
        return build_docvqa_prompt(question)
    if meta.get("task_type") in {"mcq", "mmbench"} and meta.get("choices"):
        q_with_hint = _merge_hint(question, meta)
        return build_mmbench_prompt(q_with_hint, meta["choices"])
    # For VQA/free-form, do not prepend hint; use question as-is.
    return build_vqa_prompt(question)


def extract_final_answer(text: str) -> str:
    """Extract final answer.
    Prefer lines containing 'Final answer:'; if absent, accept 'answer is'
    and fall back to the last non-empty line.
    """
    import re

    if not text:
        return ""
    # Prefer 'final answer:'; if not present, try 'answer is'
    patterns = [
        re.compile(r"final answer\s*:\s*(.*)", re.IGNORECASE),
        re.compile(r"answer\s+is\s*(.*)", re.IGNORECASE),
    ]
    for pat in patterns:
        matches = pat.findall(text)
        if matches:
            ans_line = matches[-1].strip()
            ans_line = ans_line.splitlines()[0] if ans_line else ""
            return normalize_answer(ans_line)
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return normalize_answer(lines[-1]) if lines else ""


def extract_docvqa_answer(text: str) -> str:
    """DocVQA answer parsing: strip and take the first non-empty line."""
    if not text:
        return ""
    import re

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in lines:
        m = re.search(r"final answer\s*:\s*(.*)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: skip pure <think> line
    for line in lines:
        if line.lower() == "<think>":
            continue
        return line
    return ""


def make_mm_prompt(prompt: str, image: Image.Image) -> dict:
    """Build a multi-modal prompt dict for vLLM."""
    # vLLM expects list matching number of image placeholders.
    return {"prompt": prompt, "multi_modal_data": {"image": [image]}}
