"""Run the study across several seeds and aggregate mean +/- std (error bars).

  python eval/multiseed.py --limit 50 --seeds 0 1 2 3 4

Each seed draws its own pool (different sampling rng), cached and resumable, so
re-running continues from wherever it stopped. The point is honest error bars:
one seed gives a point, several give a distribution, and then "adaptive matches
sc@16" is a claim with a spread instead of a single noisy number.

Writes results/multiseed.json, results/multiseed.md (mean +/- std),
results/pareto_seeds.png (with error bars).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import load_model, set_seed                  # noqa: E402
from src import strategies                                  # noqa: E402
from eval.cache import SampleCache                           # noqa: E402
from eval.generate import ensure_pool                       # noqa: E402
from eval.gsm8k import load_gsm8k, is_correct               # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

STRATS = {
    "greedy": lambda d: strategies.greedy(d["greedy"]),
    "sc@8": lambda d: strategies.self_consistency(d["pool"], 8),
    "sc@16": lambda d: strategies.self_consistency(d["pool"], 16),
    "adaptive(t=0.6)": lambda d: strategies.adaptive(d["pool"], 4, 16, 0.6),
    "adaptive(t=0.8)": lambda d: strategies.adaptive(d["pool"], 4, 16, 0.8),
}


def score_seed(data):
    out = {}
    for name, fn in STRATS.items():
        correct = samples = 0
        for d in data:
            a = fn(d)
            correct += is_correct(a.pred, d["problem"].answer)
            samples += a.samples_used
        n = len(data)
        out[name] = {"accuracy": correct / n, "avg_samples": samples / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    model, tok = load_model(args.model)
    problems = load_gsm8k(limit=args.limit)
    cache = SampleCache(RESULTS / "cache.jsonl")

    per_seed, done = {}, []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        set_seed(seed)
        data = ensure_pool(model, tok, problems, k=args.k, max_tokens=args.max_tokens,
                           model_id=args.model, seed=seed, cache=cache)
        per_seed[seed] = score_seed(data)
        done.append(seed)
        # write after every seed, so an interrupted run still has error bars
        agg = aggregate(per_seed, done)
        out = {"meta": {"model": args.model, "seeds": done, "limit": args.limit,
                        "max_tokens": args.max_tokens, "k": args.k},
               "per_seed": {str(s): per_seed[s] for s in done}, "aggregate": agg}
        (RESULTS / "multiseed.json").write_text(json.dumps(out, indent=2))
        write_table(agg, done)
        plot(agg)
        print(f"  [seeds done: {done}] wrote multiseed.*", flush=True)


def aggregate(per_seed, seeds):
    agg = []
    for name in STRATS:
        acc = np.array([per_seed[s][name]["accuracy"] for s in seeds])
        smp = np.array([per_seed[s][name]["avg_samples"] for s in seeds])
        agg.append({"strategy": name,
                    "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1)) if len(seeds) > 1 else 0.0,
                    "samples_mean": float(smp.mean()), "samples_std": float(smp.std(ddof=1)) if len(seeds) > 1 else 0.0})
    return agg


def write_table(agg, seeds):
    lines = [f"# Across {len(seeds)} seeds {seeds} (mean +/- std)", "",
             "| Strategy | Accuracy | Avg samples |", "|----------|:--------:|:-----------:|"]
    for r in agg:
        lines.append(f"| {r['strategy']} | {r['acc_mean']:.3f} +/- {r['acc_std']:.3f} | "
                     f"{r['samples_mean']:.2f} +/- {r['samples_std']:.2f} |")
    (RESULTS / "multiseed.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(agg):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in agg:
        color = "tab:red" if r["strategy"].startswith("adaptive") else "tab:blue"
        ax.errorbar(r["samples_mean"], r["acc_mean"], yerr=r["acc_std"], xerr=r["samples_std"],
                    fmt="o", color=color, capsize=3)
        ax.annotate(r["strategy"], (r["samples_mean"], r["acc_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("avg samples used")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs samples on GSM8K (mean +/- std over seeds)")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto_seeds.png", dpi=150)


if __name__ == "__main__":
    main()
