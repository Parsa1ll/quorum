"""Export the cached results into docs/data.json for the static site.

No model needed: this reads the seed-0 pool from the cache and the result JSONs,
and writes the per-question answers the in-browser allocator recomputes from, plus
the fixed-strategy points, the overthinking curve, and the headline stats.

  python eval/export_web.py --limit 50
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval.cache import SampleCache                          # noqa: E402
from eval.gsm8k import load_gsm8k, answer_of               # noqa: E402
from eval.analysis import load_pool_from_cache             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def _round(x):
    return None if x is None else round(float(x), 4)


def _maybe(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    DOCS.mkdir(exist_ok=True)
    cache = SampleCache(RESULTS / "cache.jsonl")
    problems = load_gsm8k(limit=args.limit)
    data = load_pool_from_cache(problems, cache, args.model, args.seed,
                                args.max_tokens, args.k)

    pool = []
    for d in data:
        pool.append({
            "gold": _round(d["problem"].answer),
            "answers": [_round(answer_of(s)) for s in d["pool"]],
            "tokens": [s.gen_tokens for s in d["pool"]],
        })

    study = _maybe("study.json")
    fixed = []
    if study:
        for r in study["runs"]:
            if r["strategy"] in ("greedy", "sc@8", "sc@16"):
                fixed.append({"strategy": r["strategy"], "accuracy": r["accuracy"],
                              "avg_samples": r["avg_samples"], "avg_tokens": r["avg_tokens"]})

    over = _maybe("overthinking.json")
    overthinking = over["curve"] if over else []

    analysis = _maybe("analysis.json")
    rules = analysis["stopping_rules"] if analysis else []

    multiseed = _maybe("multiseed.json")
    seeds = multiseed["aggregate"] if multiseed else []

    robustness = _maybe("robustness.json")

    out = {
        "meta": {
            "model": args.model.split("/")[-1],
            "n": len(pool),
            "k": args.k,
            "init_n": 4,
            "seeds": multiseed["meta"]["seeds"] if multiseed else [args.seed],
            "hardware": "Apple M1 Pro, 16 GB",
        },
        "pool": pool,
        "fixed": fixed,
        "overthinking": overthinking,
        "stopping_rules": rules,
        "multiseed": seeds,
        "robustness": robustness,
    }
    # Written as a JS file (not .json) so the page works when opened directly as a
    # local file, without a server. Any static host serves it the same way.
    (DOCS / "data.js").write_text("window.TTC_DATA = " + json.dumps(out) + ";\n")
    print(f"wrote {DOCS/'data.js'}  ({len(pool)} questions, "
          f"{len(fixed)} fixed points, {len(overthinking)} budgets)")


if __name__ == "__main__":
    main()
