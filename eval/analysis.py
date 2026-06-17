"""Offline analysis over the cached pool (no model needed).

Two things the honest writeup needs:

1. Compare our plain agreement-threshold stopping against a prior-work style
   posterior rule. Adaptive-Consistency (Aggarwal et al., EMNLP 2023) stops
   sampling once it's confident the current majority is the true majority. For
   the majority-vs-rest case that reduces to a Beta posterior on the top answer's
   share: stop when P(p > 0.5) >= conf. Computed with math.comb, no scipy.

2. Check the premise behind adaptive at all: does early agreement actually
   predict whether the answer is right? If it doesn't, adaptive can't help, and
   that's a finding worth reporting.

  python eval/analysis.py --limit 50
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval.cache import SampleCache                        # noqa: E402
from eval.generate import _qid, _key                     # noqa: E402
from eval.gsm8k import load_gsm8k, answer_of, majority_vote, agreement, is_correct  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_pool_from_cache(problems, cache, model_id, seed, max_tokens, k):
    out = []
    for p in problems:
        qid = _qid(p.question)
        g = cache.get(_key(model_id, seed, max_tokens, 0.0, qid, "greedy"))
        pool = [cache.get(_key(model_id, seed, max_tokens, 0.7, qid, j)) for j in range(k)]
        if g is None or any(s is None for s in pool):
            continue  # not generated yet
        out.append({"problem": p, "greedy": g, "pool": pool})
    return out


def majority_posterior(v1, n):
    # P(p > 0.5) for p ~ Beta(v1+1, n-v1+1), via 1 - I_0.5(a,b) with the
    # binomial-sum form of the regularized incomplete beta (integer params).
    a, b = v1 + 1, n - v1 + 1
    m = a + b - 1
    i_half = sum(comb(m, j) for j in range(a, m + 1)) * 0.5 ** m
    return 1.0 - i_half


def beta_stop(pool, min_n=2, max_n=16, conf=0.95):
    """Prior-work style: keep sampling until the majority is posterior-confident."""
    preds = []
    used = 0
    for s in pool[:max_n]:
        used += 1
        a = answer_of(s)
        if a is not None:
            preds.append(round(a, 4))
        if used >= min_n and preds:
            v1 = Counter(preds).most_common(1)[0][1]
            if majority_posterior(v1, len(preds)) >= conf:
                break
    pred = Counter(preds).most_common(1)[0][0] if preds else None
    return pred, used


def our_adaptive(pool, init_n=4, max_n=16, threshold=0.9):
    used = pool[:init_n]
    preds = [answer_of(s) for s in used]
    if agreement(preds) < threshold and max_n > init_n:
        used = pool[:max_n]
        preds = [answer_of(s) for s in used]
    return majority_vote(preds), len(used)


def _point(name, fn, data):
    correct = used = 0
    for d in data:
        pred, u = fn(d["pool"])
        correct += is_correct(pred, d["problem"].answer)
        used += u
    n = len(data)
    return {"strategy": name, "accuracy": correct / n, "avg_samples": used / n}


def validity(data, init_n=4, k=16):
    # Does early agreement predict correctness? The signal is the agreement among
    # the first init_n samples. The label MUST come from a disjoint set: scoring it
    # against the full first-k vote would be circular, because those init_n samples
    # are a subset of the k-vote and so co-move with it mechanically. We therefore
    # label each problem by the majority vote over the held-out samples [init_n:k].
    import numpy as np
    early, correct = [], []
    for d in data:
        early.append(agreement([answer_of(s) for s in d["pool"][:init_n]]))
        held = majority_vote([answer_of(s) for s in d["pool"][init_n:k]])
        correct.append(int(is_correct(held, d["problem"].answer)))
    early, correct = np.array(early), np.array(correct)
    bins = []
    for lo, hi in [(0.0, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]:
        mask = (early >= lo) & (early < hi)
        if mask.any():
            bins.append({"range": f"[{lo:.2f},{hi:.2f})", "n": int(mask.sum()),
                         "acc": float(correct[mask].mean())})
    r = float(np.corrcoef(early, correct)[0, 1]) if early.std() > 0 else float("nan")
    return {"point_biserial_r": r,
            "mean_agreement_correct": float(early[correct == 1].mean()) if (correct == 1).any() else None,
            "mean_agreement_wrong": float(early[correct == 0].mean()) if (correct == 0).any() else None,
            "bins": bins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    cache = SampleCache(RESULTS / "cache.jsonl")
    problems = load_gsm8k(limit=args.limit)
    data = load_pool_from_cache(problems, cache, args.model, args.seed, args.max_tokens, args.k)
    print(f"loaded {len(data)}/{len(problems)} fully-cached pools")
    if not data:
        return

    points = []
    for th in (0.6, 0.8, 0.9, 1.0):
        points.append(_point(f"ours(t={th})", lambda pool, th=th: our_adaptive(pool, threshold=th), data))
    # min_n=4 so the Beta rule has the same 4-sample floor as our_adaptive's init_n.
    # Without this it could stop at 2 and the comparison would not be apples-to-apples.
    for conf in (0.8, 0.9, 0.95, 0.99):
        points.append(_point(f"beta(c={conf})", lambda pool, c=conf: beta_stop(pool, min_n=4, conf=c), data))

    val = validity(data)
    out = {"meta": {"n": len(data), "model": args.model, "seed": args.seed},
           "stopping_rules": points, "agreement_validity": val}
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2))
    _report(points, val, len(data))


def _report(points, val, n):
    lines = [f"# Stopping rules (n={n})", "",
             "| Rule | Accuracy | Avg samples |", "|------|---------:|------------:|"]
    for p in points:
        lines.append(f"| {p['strategy']} | {p['accuracy']:.3f} | {p['avg_samples']:.2f} |")
    lines += ["", "## Is early agreement a valid difficulty signal?",
              f"- point-biserial r(early agreement, correct) = {val['point_biserial_r']:.3f}",
              f"- mean early agreement when correct: {val['mean_agreement_correct']}",
              f"- mean early agreement when wrong:   {val['mean_agreement_wrong']}", "",
              "| early agreement | n | accuracy |", "|---|---:|---:|"]
    for b in val["bins"]:
        lines.append(f"| {b['range']} | {b['n']} | {b['acc']:.3f} |")
    (RESULTS / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
