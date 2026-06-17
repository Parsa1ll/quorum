"""Overthinking study (s1-style budget forcing).

Question: past some thinking length, does more reasoning stop helping (or hurt)
on this small model? We trace accuracy vs thinking budget on GSM8K.

Budget forcing: let the model think for B tokens, then force it to answer by
splicing in a closing </think> and a "the answer is \boxed{" cue. Decoding is
greedy so the length axis is what varies, not sampling noise.

Efficiency: we generate ONE long greedy chain per problem (up to max budget)
and reuse it for every smaller budget. If a chain finishes on its own before B,
that natural answer is what budget B would have produced, so we just use it.

  python eval/overthinking.py --limit 30
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import load_model, set_seed, build_prompt, raw_complete, Sample  # noqa: E402
from eval.cache import SampleCache                        # noqa: E402
from eval.generate import _qid                            # noqa: E402
from eval.gsm8k import load_gsm8k, answer_of, is_correct  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGETS = [128, 256, 384, 512, 768, 1024, 1536, 2048]
FORCE = "\n</think>\n\nBased on the work above, the final answer is \\boxed{"
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _force_answer(text):
    # the forced continuation begins right after "\boxed{"; take its first number
    m = _NUM.search(text)
    return float(m.group().replace(",", "")) if m else None


def _long_chain(model, tok, question, qid, max_budget, cache, model_id):
    key = f"{model_id}|think|mt{max_budget}|{qid}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    prompt = build_prompt(tok, question)
    s = raw_complete(model, tok, prompt, temp=0.0, max_tokens=max_budget)
    cache.put(key, s)
    return s


def answer_at_budget(model, tok, question, qid, chain, budget, cache, model_id):
    """Answer + token cost if the model is held to `budget` thinking tokens."""
    # chain finished on its own within the budget -> that's the natural answer.
    if not chain.truncated and chain.gen_tokens <= budget:
        return answer_of(chain), chain.gen_tokens

    key = f"{model_id}|force|b{budget}|{qid}"
    hit = cache.get(key)
    if hit is None:
        ids = tok.encode(chain.text)
        prefix = tok.decode(ids[:budget])
        prompt = build_prompt(tok, question) + prefix + FORCE
        hit = raw_complete(model, tok, prompt, temp=0.0, max_tokens=64)
        cache.put(key, hit)
    return _force_answer(hit.text), budget + hit.gen_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    set_seed(args.seed)
    model, tok = load_model(args.model)
    problems = load_gsm8k(limit=args.limit)
    cache = SampleCache(RESULTS / "cache.jsonl")
    max_b = max(BUDGETS)

    rows = []
    for b in BUDGETS:
        rows.append({"budget": b, "correct": 0, "tokens": 0, "answered": 0})

    for i, p in enumerate(problems):
        qid = _qid(p.question)
        chain = _long_chain(model, tok, p.question, qid, max_b, cache, args.model)
        for row in rows:
            pred, cost = answer_at_budget(model, tok, p.question, qid, chain,
                                          row["budget"], cache, args.model)
            row["correct"] += is_correct(pred, p.answer)
            row["answered"] += int(pred is not None)
            row["tokens"] += cost
        print(f"  [{i+1}/{len(problems)}] {qid} chain={chain.gen_tokens}tok "
              f"truncated={chain.truncated}", flush=True)

    n = len(problems)
    curve = [{"budget": r["budget"], "accuracy": r["correct"] / n,
              "avg_tokens": r["tokens"] / n, "answered": r["answered"] / n}
             for r in rows]
    peak = max(curve, key=lambda r: r["accuracy"])
    out = {"meta": {"model": args.model, "seed": args.seed, "n_problems": n,
                    "budgets": BUDGETS},
           "curve": curve, "peak": peak}
    (RESULTS / "overthinking.json").write_text(json.dumps(out, indent=2))
    _table(curve, peak)
    _plot(curve, peak)
    print(f"\npeak accuracy {peak['accuracy']:.3f} at budget {peak['budget']} "
          f"(~{peak['avg_tokens']:.0f} tokens)")


def _table(curve, peak):
    lines = ["| Budget | Accuracy | Avg tokens | Answered |",
             "|-------:|---------:|-----------:|---------:|"]
    for r in curve:
        star = "  <- peak" if r["budget"] == peak["budget"] else ""
        lines.append(f"| {r['budget']} | {r['accuracy']:.3f} | "
                     f"{r['avg_tokens']:.0f} | {r['answered']:.2f} |{star}")
    (RESULTS / "overthinking.md").write_text("\n".join(lines) + "\n")


def _plot(curve, peak):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["avg_tokens"] for r in curve], [r["accuracy"] for r in curve], "o-")
    ax.axvline(peak["avg_tokens"], color="gray", ls="--", lw=1)
    ax.set_xlabel("avg thinking tokens")
    ax.set_ylabel("accuracy")
    ax.set_title("Overthinking curve (budget forcing, GSM8K)")
    fig.tight_layout()
    fig.savefig(RESULTS / "overthinking.png", dpi=150)


if __name__ == "__main__":
    main()
