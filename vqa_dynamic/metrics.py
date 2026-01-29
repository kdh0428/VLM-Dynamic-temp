"""Normalization and accuracy helpers for VQAv2, MCQ/MMBench, VizWiz, DocVQA, and SimpleVQA."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence
from decimal import Decimal, InvalidOperation, ROUND_DOWN


def normalize_answer(s: str) -> str:
    """Basic normalization: lowercase, strip whitespace, remove trailing punctuation."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\\.!,?]+$", "", s)
    return s.strip()

# ---------------------------------------------------------------------------
# Multiple-choice utilities
# ---------------------------------------------------------------------------

def normalize_mcq_text(text: str) -> str:
    """Lowercase, trim, strip trailing punctuation for MCQ parsing."""
    if text is None:
        return ""
    return normalize_answer(text)


def extract_choice_letter(text: str) -> str:
    """
    Extract robust MCQ choice letter (A-D) from arbitrary text.
    Priority:
      1) Exact single letter a/b/c/d
      2) Parentheses variants (a), a), (a
      3) answer/ans patterns: "answer: a", "answer is b", "ans: c"
      4) option/choice/select patterns: "option a", "choice b", "select c"
      5) First standalone a/b/c/d at word boundary
      6) Otherwise return ""
    """
    if text is None:
        return ""
    t = normalize_mcq_text(text)

    if t in {"a", "b", "c", "d"}:
        return t.upper()

    m = re.search(r"\(?([abcd])\)?", t)
    if m and m.group(1):
        return m.group(1).upper()

    m = re.search(r"(?:answer|ans)\s*(?:is|:)?\s*([abcd])", t)
    if m:
        return m.group(1).upper()

    m = re.search(r"(?:option|choice|select)\s*([abcd])", t)
    if m:
        return m.group(1).upper()

    # 5) Word boundary fallback, only if the text is short (to avoid matching articles in long sentences).
    if len(t.split()) <= 5:
        m = re.search(r"\b([abcd])\b", t)
        if m:
            return m.group(1).upper()

    return ""


def accuracy_mcq(pred_text: str, gt_text: str) -> float:
    """0/1 accuracy for MCQ based on extracted choice letters."""
    pred_letter = extract_choice_letter(pred_text)
    gt_letter = extract_choice_letter(gt_text)
    if not pred_letter or not gt_letter:
        return 0.0
    return 1.0 if pred_letter == gt_letter else 0.0


# ---------------------------------------------------------------------------
# VQAv2 normalization utilities (aligned with official evaluation)
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "he'd've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Im've": "I'm've",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "not've": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed": "she'd",
    "she'd've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebody'd",
    "somebody'd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebody's": "somebody's",
    "someoned": "someone'd",
    "someone'd've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = r";/[]\"(){}<>@&*+=_~`|"


def _process_punctuation(text: str) -> str:
    text = _COMMA_STRIP.sub(r"\1\3", text)
    text = _PERIOD_STRIP.sub("", text)
    for p in _PUNCT:
        if p in text:
            text = text.replace(p, " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _process_contractions(text: str) -> str:
    words = text.split()
    new_words = []
    for w in words:
        new_words.append(_CONTRACTIONS.get(w, w))
    return " ".join(new_words)


def normalize_vqa(text: str) -> str:
    """Normalize text following official VQAv2 eval."""
    if text is None:
        return ""
    text = text.lower().strip()
    text = _process_punctuation(text)
    text = _process_contractions(text)
    words = []
    for w in text.split():
        if w in _ARTICLES:
            continue
        if w in _NUMBER_MAP:
            w = _NUMBER_MAP[w]
        words.append(w)
    return " ".join(words).strip()


def vqa_accuracy(pred_text: str, gt_answers: Iterable[str]) -> float:
    """Compute VQAv2-style accuracy."""
    norm_pred = normalize_vqa(pred_text)
    norm_gt = [normalize_vqa(a) for a in gt_answers]
    n_match = sum(1 for a in norm_gt if a == norm_pred)
    return min(1.0, n_match / 3.0)


# ---------------------------------------------------------------------------
# DocVQA: ANLS (Average Normalized Levenshtein Similarity)
# ---------------------------------------------------------------------------

def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance with a memory-efficient DP."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, replace))
        prev = curr
    return prev[-1]


def anls_score(pred: str, golds: Sequence[str], tau: float = 0.5) -> float:
    """Compute ANLS score for a single prediction against multiple golds."""
    if not golds:
        return 0.0
    pred_norm = (pred or "").strip().lower()
    best = 0.0
    for gold in golds:
        gold_norm = (gold or "").strip().lower()
        if not pred_norm and not gold_norm:
            nl = 0.0
        else:
            denom = max(len(pred_norm), len(gold_norm))
            dist = _levenshtein_distance(pred_norm, gold_norm)
            nl = dist / denom if denom else 0.0
        score = (1.0 - nl) if nl < tau else 0.0
        if score > best:
            best = score
    return float(best)


# ---------------------------------------------------------------------------
# VizWiz (VQA official-style) accuracy
# ---------------------------------------------------------------------------

_VIZWIZ_MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_VIZWIZ_ARTICLES = {"a", "an", "the"}
_VIZWIZ_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_VIZWIZ_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_VIZWIZ_PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def _vizwiz_process_punctuation(text: str) -> str:
    out_text = text
    for p in _VIZWIZ_PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_VIZWIZ_COMMA_STRIP, text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _VIZWIZ_PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def _vizwiz_process_digit_article(text: str) -> str:
    out_words = []
    for word in text.lower().split():
        word = _VIZWIZ_MANUAL_MAP.setdefault(word, word)
        if word not in _VIZWIZ_ARTICLES:
            out_words.append(word)
    for idx, word in enumerate(out_words):
        if word in _CONTRACTIONS:
            out_words[idx] = _CONTRACTIONS[word]
    return " ".join(out_words)


def vizwiz_accuracy(pred_text: str, gt_answers: Sequence[str]) -> float:
    """Compute VizWiz accuracy using official VQA-style normalization."""
    if not gt_answers:
        return 0.0

    gt_list = [str(a) for a in gt_answers]
    gt_list = [a.replace("\n", " ").replace("\t", " ").strip() for a in gt_list]
    pred_ans = str(pred_text or "").replace("\n", " ").replace("\t", " ").strip()

    if len(gt_list) == 1:
        gt_single = gt_list[0]
        if pred_ans == gt_single:
            return 1.0
        pred_norm = _vizwiz_process_digit_article(_vizwiz_process_punctuation(pred_ans))
        gt_norm = _vizwiz_process_digit_article(_vizwiz_process_punctuation(gt_single))
        return 1.0 if pred_norm == gt_norm else 0.0

    if len(set(gt_list)) > 1:
        gt_list = [_vizwiz_process_digit_article(_vizwiz_process_punctuation(a)) for a in gt_list]
        pred_ans = _vizwiz_process_digit_article(_vizwiz_process_punctuation(pred_ans))

    gt_acc = []
    for idx, gt in enumerate(gt_list):
        other = [a for j, a in enumerate(gt_list) if j != idx]
        matching = [a for a in other if a == pred_ans]
        gt_acc.append(min(1.0, len(matching) / 3.0))
    return float(sum(gt_acc) / len(gt_acc))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SIMPLEVQA_NA_PATTERNS = [
    r"\b(i\s*don't\s*know)\b",
    r"\b(i\s*do\s*not\s*know)\b",
    r"\b(unsure)\b",
    r"\b(not\s*sure)\b",
    r"\b(cannot\s*determine)\b",
    r"\b(can't\s*tell)\b",
    r"\b(unknown)\b",
    r"\b(no\s*idea)\b",
    r"\b(need\s*more\s*information)\b",
    r"\b(not\s*provided)\b",
    r"\b(n/a)\b",
    r"\b(na)\b",
    r"\b(모르겠)\b",
    r"\b(알\s*수\s*없)\b",
    r"\b(확인\s*불가)\b",
    r"\b(판단\s*불가)\b",
]
_SIMPLEVQA_NA_RE = re.compile("|".join(_SIMPLEVQA_NA_PATTERNS), re.IGNORECASE)


def _strip_data_prefix_if_any(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(final\s*answer\s*:)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*(answer\s*:)\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _normalize_text_basic(s: str) -> str:
    s = _strip_data_prefix_if_any(s)
    s = s.strip().lower()
    s = s.strip(" \t\n\r\"'`")
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\.$", "", s).strip()
    return s


def _extract_first_number(s: str) -> str | None:
    s = _strip_data_prefix_if_any(s)
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", s)
    return m.group(0) if m else None


def _to_decimal(num_str: str) -> Decimal | None:
    try:
        cleaned = num_str.replace(",", "")
        return Decimal(cleaned)
    except (InvalidOperation, AttributeError):
        return None


def _decimal_places(num_str: str) -> int:
    num_str = num_str.replace(",", "")
    if "." not in num_str:
        return 0
    return max(0, len(num_str.split(".", 1)[1]))


def _numeric_equiv(pred_num_str: str, gold_num_str: str) -> bool:
    pred_dec = _to_decimal(pred_num_str)
    gold_dec = _to_decimal(gold_num_str)
    if pred_dec is None or gold_dec is None:
        return False
    if pred_dec == gold_dec:
        return True
    p_dp = _decimal_places(pred_num_str)
    if p_dp == 0:
        gold_trunc = gold_dec.quantize(Decimal("1"), rounding=ROUND_DOWN)
        return pred_dec == gold_trunc
    quant = Decimal("1." + ("0" * p_dp))
    gold_trunc = gold_dec.quantize(quant, rounding=ROUND_DOWN)
    return pred_dec == gold_trunc


def _is_not_attempted(pred: str) -> bool:
    pred_norm = _normalize_text_basic(pred)
    if not pred_norm:
        return True
    if _SIMPLEVQA_NA_RE.search(pred_norm):
        return True
    if pred_norm in {"idk", "i dont know", "i do not know", "unknown"}:
        return True
    return False


def _exact_correct(pred: str, gold: str) -> bool:
    pred_norm = _normalize_text_basic(pred)
    gold_norm = _normalize_text_basic(gold)
    if not pred_norm:
        return False
    if pred_norm == gold_norm:
        return True
    pred_num = _extract_first_number(pred)
    gold_num = _extract_first_number(gold)
    if pred_num and gold_num:
        return _numeric_equiv(pred_num, gold_num)
    return False


def compute_accuracy(pred_text, gt, task_type: str = "vqa") -> float:
    """
    task_type in {"mmbench", "mcq"} => gt is a single choice string
    else => gt is an iterable/list of VQAv2 answers
    """
    if task_type == "yesno":
        pred = (pred_text or "").strip().lower()
        gt_text = gt if isinstance(gt, str) else (gt[0] if gt else "")
        gt_norm = str(gt_text).strip().lower()
        return 1.0 if pred in {"yes", "no"} and pred == gt_norm else 0.0
    if task_type == "vizwiz":
        gt_texts = gt if isinstance(gt, list) else ([gt] if gt else [])
        return vizwiz_accuracy(pred_text, gt_texts)
    if task_type == "docvqa":
        gt_texts = gt if isinstance(gt, list) else ([gt] if gt else [])
        return anls_score(pred_text, gt_texts)
    if task_type == "simplevqa":
        gt_text = gt if isinstance(gt, str) else (gt[0] if gt else "")
        if _is_not_attempted(pred_text):
            return 0.0
        return 1.0 if _exact_correct(pred_text, gt_text) else 0.0
    if task_type in {"mmbench", "mcq", "mmstar"}:
        gt_text = gt if isinstance(gt, str) else (gt[0] if gt else "")
        return accuracy_mcq(pred_text, gt_text)
    return vqa_accuracy(pred_text, gt)


# ---------------------------------------------------------------------------
# Simple tests
# ---------------------------------------------------------------------------

def _test_mcq():
    assert accuracy_mcq("Answer: (b)", "b") == 1.0
    assert accuracy_mcq("option c", "C") == 1.0
    assert accuracy_mcq("I think it's D.", "a") == 0.0
    assert accuracy_mcq("select a", "A") == 1.0
    assert accuracy_mcq("none", "A") == 0.0


def _test_vqa():
    gts = ["Two", "two.", "2", "three"]
    assert abs(vqa_accuracy("two", gts) - 1.0) < 1e-6
    assert abs(vqa_accuracy("2", gts) - 1.0) < 1e-6
    assert abs(vqa_accuracy("three", gts) - min(1.0, 1 / 3.0)) < 1e-6
    assert abs(vqa_accuracy("five", gts) - 0.0) < 1e-6


if __name__ == "__main__":
    _test_mcq()
    _test_vqa()
    print("All tests passed.")
