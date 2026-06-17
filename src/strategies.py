"""Test-time-compute strategies.

These are pure: they take an already-generated pool of samples and decide an
answer plus how much of the pool they "spent". Generation happens once up front
(eval/generate.py) and every strategy reads the same pool, so the accuracy gaps
are purely about allocation, not luck.

  greedy            the single temp-0 sample
  self_consistency  majority vote over the first n pool samples
  adaptive          peek at init_n; if they agree, stop; else use up to max_n
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval.gsm8k import answer_of, majority_vote, agreement  # noqa: E402


@dataclass
class Answer:
    pred: float | None
    samples_used: int          # generations spent (the compute proxy)
    tokens_used: int


def _tokens(samples):
    return sum(s.gen_tokens for s in samples)


def greedy(greedy_sample) -> Answer:
    return Answer(answer_of(greedy_sample), 1, greedy_sample.gen_tokens)


def self_consistency(pool, n=16) -> Answer:
    used = pool[:n]
    preds = [answer_of(s) for s in used]
    return Answer(majority_vote(preds), len(used), _tokens(used))


def adaptive(pool, init_n=4, max_n=16, agree_threshold=0.9) -> Answer:
    used = pool[:init_n]
    preds = [answer_of(s) for s in used]

    if agreement(preds) < agree_threshold and max_n > init_n:
        used = pool[:max_n]
        preds = [answer_of(s) for s in used]

    return Answer(majority_vote(preds), len(used), _tokens(used))
