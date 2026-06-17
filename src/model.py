"""mlx-lm generation helpers.

load_model loads a small reasoning model; sample() draws one completion. Runs
locally, no API. Generation is the whole cost of the study, so this stays thin
and the caching/pooling lives one layer up (see eval/generate.py).
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from src.sample import Sample  # re-exported; defined dependency-free for the offline path


def set_seed(seed: int):
    # one global seed at the start of a run keeps it reproducible while the
    # rng still advances between samples (so sc draws differ).
    mx.random.seed(seed)


def load_model(model_id: str):
    return load(model_id)


def build_prompt(tokenizer, question, system=None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def _run(model, tokenizer, prompt, sampler, max_tokens) -> Sample:
    parts, n, finish = [], 0, None
    for r in stream_generate(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, sampler=sampler):
        parts.append(r.text)
        n = r.generation_tokens
        finish = r.finish_reason
    # finish_reason is "length" when it ran out of budget mid-thought.
    return Sample(text="".join(parts), gen_tokens=n, truncated=(finish == "length"))


def sample(model, tokenizer, question, *, temp=0.7, top_p=0.95,
           max_tokens=1024, system=None) -> Sample:
    # temp=0 gives near-greedy decoding for the baseline.
    prompt = build_prompt(tokenizer, question, system)
    return _run(model, tokenizer, prompt, make_sampler(temp=temp, top_p=top_p), max_tokens)


def raw_complete(model, tokenizer, prompt, *, temp=0.0, top_p=1.0,
                 max_tokens=256) -> Sample:
    # continue from an arbitrary prompt string (used by budget forcing to make
    # the model answer after we splice in a closing </think>).
    return _run(model, tokenizer, prompt, make_sampler(temp=temp, top_p=top_p), max_tokens)
