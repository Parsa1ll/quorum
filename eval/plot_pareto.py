"""Combined view: accuracy vs tokens across both studies.

avg_samples hides the real cost (a truncated 1024-token sample costs far more
than a 300-token one), so the honest compute axis is average generated tokens.
This puts the sample-count strategies and the budget-forcing curve on one plot
and pulls out the token-efficiency comparison.

  python eval/plot_pareto.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _match_tokens(curve_points, target_acc):
    # cheapest (fewest-token) point that reaches target accuracy
    hits = [p for p in curve_points if p["accuracy"] >= target_acc - 1e-9]
    return min(hits, key=lambda p: p["tokens"]) if hits else None


def main():
    study = _load("study.json")
    over = _load("overthinking.json")
    if not study:
        print("no study.json; run eval/run_study.py first")
        return

    runs = study["runs"]
    fixed = [r for r in runs if r["strategy"] in ("greedy", "sc@8", "sc@16")]
    adapt = sorted((r for r in runs if r["strategy"].startswith("adaptive")),
                   key=lambda r: r["avg_tokens"])

    # token-efficiency headline: cheapest way to hit sc@16 accuracy
    sc16 = next(r for r in runs if r["strategy"] == "sc@16")
    cands = [{"name": r["strategy"], "tokens": r["avg_tokens"], "accuracy": r["accuracy"]}
             for r in runs]
    if over:
        cands += [{"name": f"budget@{p['budget']}", "tokens": p["avg_tokens"],
                   "accuracy": p["accuracy"]} for p in over["curve"]]
    cheapest = _match_tokens(cands, sc16["accuracy"])
    summary = []
    if cheapest:
        ratio = sc16["avg_tokens"] / cheapest["tokens"]
        summary.append(
            f"Cheapest route to sc@16 accuracy ({sc16['accuracy']:.3f}): "
            f"{cheapest['name']} at ~{cheapest['tokens']:.0f} tokens vs sc@16's "
            f"{sc16['avg_tokens']:.0f} -> {ratio:.1f}x fewer tokens. (In-sample: this "
            f"picks the cheapest point on the same 50 problems, so it is optimistic; "
            f"see the held-out figure in RESULTS.md, ~37% of the samples.)")
    if over:
        bf = max(over["curve"], key=lambda p: p["accuracy"])
        best_sc = max((r for r in runs if r["strategy"].startswith("sc")),
                      key=lambda r: r["accuracy"])
        tok_ratio = best_sc["avg_tokens"] / bf["avg_tokens"]
        gap = best_sc["accuracy"] - bf["accuracy"]
        summary.append(
            f"Axis tradeoff: parallel sampling has the higher ceiling "
            f"({best_sc['strategy']} {best_sc['accuracy']:.3f}) vs single-chain budget "
            f"forcing peak {bf['accuracy']:.3f} (+{gap:.3f} for sampling), but budget "
            f"forcing reaches its peak at ~{bf['avg_tokens']:.0f} tokens, {tok_ratio:.0f}x "
            f"fewer than {best_sc['strategy']}'s {best_sc['avg_tokens']:.0f}.")
    (RESULTS / "summary.md").write_text("\n".join("- " + s for s in summary) + "\n")
    for s in summary:
        print(s)

    _plot(fixed, adapt, over)


def _plot(fixed, adapt, over):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([r["avg_tokens"] for r in fixed], [r["accuracy"] for r in fixed],
               s=90, color="tab:blue", zorder=3, label="fixed (greedy / sc)")
    for r in fixed:
        ax.annotate(r["strategy"], (r["avg_tokens"], r["accuracy"]),
                    textcoords="offset points", xytext=(6, 5))
    if adapt:
        ax.plot([r["avg_tokens"] for r in adapt], [r["accuracy"] for r in adapt],
                "s-", color="tab:red", label="adaptive (threshold sweep)")
    if over:
        c = over["curve"]
        ax.plot([p["avg_tokens"] for p in c], [p["accuracy"] for p in c],
                "o-", color="tab:green", label="budget forcing (1 chain)")
    ax.set_xscale("log")
    ax.set_xlabel("avg generated tokens per question  (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs compute on GSM8K (1.5B, on-device)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto_tokens.png", dpi=150)


if __name__ == "__main__":
    main()
