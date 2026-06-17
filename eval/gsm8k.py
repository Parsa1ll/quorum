"""GSM8K loading, answer extraction, and scoring.

Gold answers end in a line like '#### 42'. For model outputs we prefer the
number inside \boxed{}; on this model a finished answer almost always has one.
If \boxed is missing and the generation was truncated, we abstain rather than
grab a stray number from the middle of the reasoning. Grabbing that stray number
was corrupting the majority vote, so abstaining on truncation is deliberate.
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class Problem:
    question: str
    answer: float


_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def load_gsm8k(split="test", limit=100) -> list[Problem]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    for row in ds:
        gold = row["answer"].split("####")[-1].strip()
        problems.append(Problem(row["question"], _to_float(gold)))
        if limit and len(problems) >= limit:
            break
    return problems


def _to_float(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _boxed(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return None
    num = _NUM.search(m[-1])
    return _to_float(num.group()) if num else None


def _last_number(text):
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None


def extract_answer(text) -> float | None:
    # text-only extraction (used by the smoke test): boxed, else last number.
    boxed = _boxed(text)
    return boxed if boxed is not None else _last_number(text)


def answer_of(sample) -> float | None:
    # truncation-aware: a cut-off sample with no \boxed abstains.
    boxed = _boxed(sample.text)
    if boxed is not None:
        return boxed
    if sample.truncated:
        return None
    return _last_number(sample.text)


def is_correct(pred, gold, tol=1e-4) -> bool:
    return pred is not None and abs(pred - gold) < tol


def majority_vote(preds) -> float | None:
    preds = [round(p, 4) for p in preds if p is not None]
    if not preds:
        return None
    return Counter(preds).most_common(1)[0][0]


def agreement(preds) -> float:
    # Fraction of (non-abstaining) samples that back the majority answer. Used
    # as the difficulty signal for adaptive: high means easy/confident.
    preds = [round(p, 4) for p in preds if p is not None]
    if not preds:
        return 0.0
    top = Counter(preds).most_common(1)[0][0]
    return sum(1 for p in preds if p == top) / len(preds)
