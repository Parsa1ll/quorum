"""Generate (and cache) the sample pool that the strategies analyse.

For each problem we draw:
  - one greedy sample (temp 0) for the baseline
  - a pool of K samples (temp 0.7) shared by sc@8 / sc@16 / adaptive

Sharing the pool matters: every strategy sees the same draws, so the accuracy
gaps come from how they allocate compute, not from sampling luck. It
also roughly halves generation vs running each strategy separately.

Each sample is seeded from (seed, question, index) so it's reproducible and
order-independent: resuming a half-finished run gives the same pool.
"""
from __future__ import annotations
import hashlib

# src.model is imported lazily inside _draw: it pulls in mlx, but the key helpers
# (_qid/_key/_derive_seed) are pure and reused by the offline analysis scripts,
# which must run without the GPU stack installed.


def _qid(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()[:10]


def _key(model_id, seed, max_tokens, temp, qid, idx) -> str:
    return f"{model_id}|s{seed}|mt{max_tokens}|t{temp}|{qid}|{idx}"


def _derive_seed(seed, qid, idx) -> int:
    h = int(hashlib.md5(f"{qid}|{idx}".encode()).hexdigest()[:8], 16)
    return (seed * 1_000_003 + h) % (2**31 - 1)


def _draw(model, tok, question, *, qid, idx, temp, max_tokens, seed, cache, model_id):
    from src.model import sample, set_seed  # lazy: only generation needs mlx
    key = _key(model_id, seed, max_tokens, temp, qid, idx)
    hit = cache.get(key)
    if hit is not None:
        return hit
    set_seed(_derive_seed(seed, qid, idx))
    s = sample(model, tok, question, temp=temp, max_tokens=max_tokens)
    cache.put(key, s)
    return s


def ensure_pool(model, tok, problems, *, k, max_tokens, model_id, seed, cache, log=print):
    """Make sure every problem has its greedy sample + K-sample pool cached.

    Returns a list of dicts aligned with `problems`:
      {"problem": p, "greedy": Sample, "pool": [Sample, ...]}
    """
    out = []
    for i, p in enumerate(problems):
        qid = _qid(p.question)
        g = _draw(model, tok, p.question, qid=qid, idx="greedy", temp=0.0,
                  max_tokens=max_tokens, seed=seed, cache=cache, model_id=model_id)
        pool = []
        for j in range(k):
            s = _draw(model, tok, p.question, qid=qid, idx=j, temp=0.7,
                      max_tokens=max_tokens, seed=seed, cache=cache, model_id=model_id)
            pool.append(s)
        n_trunc = sum(s.truncated for s in pool)
        log(f"  [{i+1}/{len(problems)}] {qid} pool={k} truncated={n_trunc}", flush=True)
        out.append({"problem": p, "greedy": g, "pool": pool})
    return out
