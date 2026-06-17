"""Run the TTC study on GSM8K and write the results.

Two passes: generate a shared sample pool once (cached, resumable), then score
each strategy on it. Re-running only re-scores, so sweeping the adaptive
threshold is free.

Writes to results/:
  study.json   meta + per-strategy accuracy / avg compute + diagnostics
  study.md     the same as a markdown table
  pareto.png   accuracy vs average samples

Keep --limit small at first, generation is slow on a laptop:
  python eval/run_study.py --limit 50
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
# src.model (mlx) is imported lazily in main() so the pure scoring helpers here
# (score, headline, headline_heldout) can be reused offline without the GPU stack
# installed.
from src import strategies                               # noqa: E402
from eval.cache import SampleCache                        # noqa: E402
from eval.generate import ensure_pool                    # noqa: E402
from eval.gsm8k import load_gsm8k, answer_of, is_correct  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def score(name, answers, problems):
    correct = total_samples = total_tokens = 0
    for ans, p in zip(answers, problems):
        correct += is_correct(ans.pred, p.answer)
        total_samples += ans.samples_used
        total_tokens += ans.tokens_used
    n = len(problems)
    return {
        "strategy": name,
        "accuracy": correct / n,
        "avg_samples": total_samples / n,
        "avg_tokens": total_tokens / n,
        "n_problems": n,
    }


def boxed_diagnostic(data):
    # sanity-check the abstain policy: how often does a finished sample lack a
    # parseable answer? if this is ~0 the boxed-or-abstain rule is safe.
    finished = no_answer = truncated = 0
    for d in data:
        for s in d["pool"]:
            if s.truncated:
                truncated += 1
                continue
            finished += 1
            if answer_of(s) is None:
                no_answer += 1
    return {"finished": finished, "truncated": truncated,
            "finished_but_no_answer": no_answer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--limit", type=int, default=50, help="number of GSM8K problems")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16, help="pool size (max samples per question)")
    args = ap.parse_args()

    from src.model import load_model, set_seed  # lazy: only generation needs mlx
    RESULTS.mkdir(exist_ok=True)
    set_seed(args.seed)
    model, tok = load_model(args.model)
    problems = load_gsm8k(limit=args.limit)
    cache = SampleCache(RESULTS / "cache.jsonl")

    print(f"generating pool (k={args.k}, max_tokens={args.max_tokens}) ...")
    data = ensure_pool(model, tok, problems, k=args.k, max_tokens=args.max_tokens,
                       model_id=args.model, seed=args.seed, cache=cache)
    probs = [d["problem"] for d in data]

    runs = [
        score("greedy", [strategies.greedy(d["greedy"]) for d in data], probs),
        score("sc@8", [strategies.self_consistency(d["pool"], 8) for d in data], probs),
        score("sc@16", [strategies.self_consistency(d["pool"], 16) for d in data], probs),
    ]
    # adaptive operating points: same pool, just different stopping thresholds.
    for th in (0.6, 0.7, 0.8, 0.9, 1.0):
        answers = [strategies.adaptive(d["pool"], init_n=4, max_n=16, agree_threshold=th)
                   for d in data]
        runs.append(score(f"adaptive(t={th})", answers, probs))

    out = {
        "meta": {
            "model": args.model, "seed": args.seed,
            "max_tokens": args.max_tokens, "k": args.k, "n_problems": len(probs),
        },
        "runs": runs,
        "diagnostics": boxed_diagnostic(data),
        "headline": headline(runs),
        "headline_heldout": headline_heldout(data, probs),
    }
    (RESULTS / "study.json").write_text(json.dumps(out, indent=2))
    write_table(runs, out["headline"], out["headline_heldout"])
    plot(runs)
    print(f"\nwrote results to {RESULTS}/")
    if out["headline"]:
        print("in-sample:  " + out["headline"]["summary"])
    if out["headline_heldout"]:
        print("held-out:   " + out["headline_heldout"]["summary"])


def headline(runs):
    # cheapest adaptive point that matches sc@16 accuracy (within a small margin).
    sc16 = next((r for r in runs if r["strategy"] == "sc@16"), None)
    if not sc16:
        return None
    adapt = [r for r in runs if r["strategy"].startswith("adaptive")]
    matches = [r for r in adapt if r["accuracy"] >= sc16["accuracy"] - 1e-9]
    pick = min(matches, key=lambda r: r["avg_samples"]) if matches else max(
        adapt, key=lambda r: r["accuracy"], default=None)
    if not pick:
        return None
    pct = 100 * pick["avg_samples"] / sc16["avg_samples"]
    return {
        "matched": bool(matches),
        "point": pick["strategy"],
        "adaptive_acc": pick["accuracy"],
        "sc16_acc": sc16["accuracy"],
        "pct_of_samples": pct,
        "summary": (f"{pick['strategy']} reaches {pick['accuracy']:.3f} acc vs "
                    f"sc@16 {sc16['accuracy']:.3f} using {pct:.0f}% of the samples "
                    f"({pick['avg_samples']:.1f} vs {sc16['avg_samples']:.1f})."),
    }


def headline_heldout(data, probs, thresholds=(0.6, 0.7, 0.8, 0.9, 1.0),
                     init_n=4, max_n=16):
    # Honest headline: choose the adaptive threshold on a tuning half, then report
    # it on the held-out half. The plain headline() picks the cheapest threshold
    # that matches sc@16 on the SAME 50 problems it scores, which is selection on
    # the test set and biases the savings optimistically. This avoids that.
    n = len(probs)
    if n < 4:
        return None
    mid = n // 2
    tune_idx, rep_idx = list(range(mid)), list(range(mid, n))

    def acc_samples(idxs, answers):
        c = s = 0
        for i in idxs:
            c += is_correct(answers[i].pred, probs[i].answer)
            s += answers[i].samples_used
        m = len(idxs)
        return c / m, s / m

    sc16 = [strategies.self_consistency(d["pool"], max_n) for d in data]
    sc16_tune_acc, _ = acc_samples(tune_idx, sc16)

    cand = []
    for th in thresholds:
        ans = [strategies.adaptive(d["pool"], init_n, max_n, th) for d in data]
        tune_acc, tune_smp = acc_samples(tune_idx, ans)
        cand.append({"th": th, "tune_acc": tune_acc, "tune_samples": tune_smp, "ans": ans})

    matches = [c for c in cand if c["tune_acc"] >= sc16_tune_acc - 1e-9]
    pick = min(matches or cand, key=lambda c: c["tune_samples"])
    rep_acc, rep_smp = acc_samples(rep_idx, pick["ans"])
    sc16_rep_acc, sc16_rep_smp = acc_samples(rep_idx, sc16)
    pct = 100 * rep_smp / sc16_rep_smp if sc16_rep_smp else None
    return {
        "matched_on_tune": bool(matches),
        "picked_threshold": pick["th"],
        "tune_n": len(tune_idx), "report_n": len(rep_idx),
        "report_adaptive_acc": rep_acc, "report_adaptive_samples": rep_smp,
        "report_sc16_acc": sc16_rep_acc, "report_sc16_samples": sc16_rep_smp,
        "pct_of_samples": pct,
        "summary": (f"tuned t={pick['th']} on first {len(tune_idx)} problems, "
                    f"reported on held-out {len(rep_idx)}: adaptive "
                    f"{rep_acc:.3f} @ {rep_smp:.1f} samples vs sc@16 "
                    f"{sc16_rep_acc:.3f} @ {sc16_rep_smp:.1f}"
                    f"{f' ({pct:.0f}% of samples)' if pct else ''}."),
    }


def write_table(runs, head, heldout=None):
    lines = ["| Strategy | Accuracy | Avg samples | Avg tokens |",
             "|----------|---------:|------------:|-----------:|"]
    for r in runs:
        lines.append(f"| {r['strategy']} | {r['accuracy']:.3f} | "
                     f"{r['avg_samples']:.1f} | {r['avg_tokens']:.0f} |")
    if head:
        lines += ["", "In-sample (threshold chosen and scored on the same 50 problems, "
                  "so the saving is optimistically biased):", head["summary"]]
    if heldout:
        lines += ["", "Held-out (threshold tuned on one half, reported on the other "
                  "half; this is the honest saving):", heldout["summary"]]
    (RESULTS / "study.md").write_text("\n".join(lines) + "\n")


def plot(runs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    fixed = [r for r in runs if not r["strategy"].startswith("adaptive")]
    adapt = [r for r in runs if r["strategy"].startswith("adaptive")]
    ax.scatter([r["avg_samples"] for r in fixed], [r["accuracy"] for r in fixed],
               s=80, label="fixed")
    for r in fixed:
        ax.annotate(r["strategy"], (r["avg_samples"], r["accuracy"]),
                    textcoords="offset points", xytext=(6, 4))
    if adapt:
        adapt = sorted(adapt, key=lambda r: r["avg_samples"])
        ax.plot([r["avg_samples"] for r in adapt], [r["accuracy"] for r in adapt],
                "o-", color="tab:red", label="adaptive (threshold sweep)")
    ax.set_xlabel("avg samples used")
    ax.set_ylabel("accuracy")
    ax.set_title("Test-time compute on GSM8K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=150)


if __name__ == "__main__":
    main()
