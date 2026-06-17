"""Offline robustness checks on a cached run (no model, no GPU).

1. Bootstrap CIs over problems, including the paired sc@8 - sc@16 difference.
   This answers whether 0.86 vs 0.88 is significant at n=50.
2. Truncation confound: hard problems both truncate and disagree, so the
   agreement signal might just be a truncation detector. Re-check the
   agreement->correct relationship while controlling for per-problem truncation.

  python eval/robustness.py --limit 50
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import strategies                                   # noqa: E402
from eval.cache import SampleCache                           # noqa: E402
from eval.gsm8k import load_gsm8k, answer_of, agreement, is_correct  # noqa: E402
from eval.analysis import load_pool_from_cache               # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def correct_vec(data, fn):
    return np.array([float(is_correct(fn(d).pred, d["problem"].answer)) for d in data])


def boot_ci(vec, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(vec)
    means = vec[rng.integers(0, n, (B, n))].mean(1)
    return vec.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_ci(a, b, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    diffs = (a[idx] - b[idx]).mean(1)
    return (a - b).mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


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
    n = len(data)
    print(f"n = {n} cached problems\n")

    strat = {
        "greedy": lambda d: strategies.greedy(d["greedy"]),
        "sc@8": lambda d: strategies.self_consistency(d["pool"], 8),
        "sc@16": lambda d: strategies.self_consistency(d["pool"], 16),
        "adaptive(t=0.6)": lambda d: strategies.adaptive(d["pool"], 4, 16, 0.6),
    }
    vecs = {k: correct_vec(data, fn) for k, fn in strat.items()}

    print("Bootstrap 95% CI on accuracy (10k resamples over problems):")
    cis = {}
    for k, v in vecs.items():
        m, lo, hi = boot_ci(v)
        cis[k] = [m, lo, hi]
        print(f"  {k:16s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    d, lo, hi = paired_ci(vecs["sc@8"], vecs["sc@16"])
    sig = not (lo <= 0 <= hi)
    print(f"\nPaired sc@8 - sc@16 = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"-> {'significant' if sig else 'NOT significant (CI spans 0)'}")

    # truncation confound
    trunc = np.array([np.mean([s.truncated for s in d["pool"]]) for d in data])
    early = np.array([agreement([answer_of(s) for s in d["pool"][:4]]) for d in data])
    # Label from a DISJOINT held-out vote (samples 4:16), not vecs["sc@16"]: the
    # first-4 agreement is a subset of the full 16-vote, so scoring against it would
    # inflate r by construction. Matches the disjoint label used in analysis.validity.
    held = np.array([float(is_correct(
        strategies.self_consistency(d["pool"][4:16], 12).pred, d["problem"].answer))
        for d in data])

    def r(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")

    def r_ci(x, y, B=10000, seed=0):
        # bootstrap CI for the correlation. at n=50 this is wide, which is the point:
        # it shows whether the agreement->correct link is distinguishable from zero.
        rng = np.random.default_rng(seed)
        m = len(x)
        rs = []
        for _ in range(B):
            idx = rng.integers(0, m, m)
            xi, yi = x[idx], y[idx]
            rs.append(np.corrcoef(xi, yi)[0, 1] if xi.std() > 0 and yi.std() > 0 else np.nan)
        rs = np.array(rs)
        return [float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))]

    low = trunc <= np.median(trunc)
    r_all_ci = r_ci(early, held)
    out = {
        "n": n,
        "accuracy_ci": cis,
        "sc8_minus_sc16": {"diff": d, "ci": [lo, hi], "significant": bool(sig)},
        "truncation_confound": {
            "mean_truncation": float(trunc.mean()),
            "r_agreement_correct_all": r(early, held),
            "r_agreement_correct_all_ci": r_all_ci,
            "r_agreement_correct_all_significant": not (r_all_ci[0] <= 0 <= r_all_ci[1]),
            "r_agreement_correct_low_trunc": r(early[low], held[low]),
            "r_truncation_correct": r(trunc, held),
            "n_low_trunc": int(low.sum()),
            "label": "majority vote over held-out samples [4:16]",
        },
    }
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    tc = out["truncation_confound"]
    print(f"\nTruncation confound (mean per-problem truncation {tc['mean_truncation']:.2f}):")
    print(f"  r(agreement, correct), all problems      = {tc['r_agreement_correct_all']:.3f}  "
          f"95% CI [{tc['r_agreement_correct_all_ci'][0]:.2f}, {tc['r_agreement_correct_all_ci'][1]:.2f}]  "
          f"-> {'significant' if tc['r_agreement_correct_all_significant'] else 'NOT significant (CI spans 0)'}")
    print(f"  r(agreement, correct), low-truncation half = {tc['r_agreement_correct_low_trunc']:.3f}  (n={tc['n_low_trunc']})")
    print(f"  r(truncation, correct)                   = {tc['r_truncation_correct']:.3f}")


if __name__ == "__main__":
    main()
