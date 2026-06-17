"""Interactive demo: watch the adaptive allocator spend compute per question.

  streamlit run app.py

Reads the cached pool and study results from results/ and recomputes the
adaptive strategy live as you move the threshold, so the numbers are real, not
canned. Generation already happened (cached), so this is instant.
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from src import strategies                                   # noqa: E402
from eval.cache import SampleCache                           # noqa: E402
from eval.gsm8k import load_gsm8k, agreement, answer_of, is_correct  # noqa: E402
from eval.analysis import load_pool_from_cache               # noqa: E402

RESULTS = ROOT / "results"
st.set_page_config(page_title="Adaptive TTC on GSM8K", layout="wide")


@st.cache_data
def load_everything():
    study = json.loads((RESULTS / "study.json").read_text())
    over = _maybe("overthinking.json")
    analysis = _maybe("analysis.json")
    m = study["meta"]
    cache = SampleCache(RESULTS / "cache.jsonl")
    problems = load_gsm8k(limit=m["n_problems"])
    data = load_pool_from_cache(problems, cache, m["model"], m["seed"], m["max_tokens"], m["k"])
    return study, over, analysis, m, data


def _maybe(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def run_adaptive(data, init_n, max_n, threshold):
    rows, correct, samples = [], 0, 0
    for i, d in enumerate(data):
        ans = strategies.adaptive(d["pool"], init_n=init_n, max_n=max_n, agree_threshold=threshold)
        ok = is_correct(ans.pred, d["problem"].answer)
        correct += ok
        samples += ans.samples_used
        rows.append({"i": i, "agreement": agreement([answer_of(s) for s in d["pool"][:init_n]]),
                     "used": ans.samples_used, "ok": ok, "pred": ans.pred,
                     "gold": d["problem"].answer})
    n = len(data)
    return rows, correct / n, samples / n


def grid_html(rows):
    tiles = []
    for r in rows:
        color = "#2e7d32" if r["ok"] else "#c62828"
        tip = (f"Q{r['i']}  agreement {r['agreement']:.2f}  used {r['used']} samples  "
               f"pred {r['pred']} / gold {r['gold']}")
        tiles.append(
            f'<div title="{tip}" style="width:34px;height:34px;border-radius:5px;'
            f'background:{color};color:#fff;display:flex;align-items:center;'
            f'justify-content:center;font-size:13px;font-weight:600;">{r["used"]}</div>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:5px;">' + "".join(tiles) + "</div>")


def pareto_fig(study, live):
    import matplotlib.pyplot as plt
    runs = study["runs"]
    fixed = [r for r in runs if r["strategy"] in ("greedy", "sc@8", "sc@16")]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter([r["avg_samples"] for r in fixed], [r["accuracy"] for r in fixed],
               s=70, color="tab:blue", zorder=3)
    for r in fixed:
        ax.annotate(r["strategy"], (r["avg_samples"], r["accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.scatter([live["samples"]], [live["acc"]], s=160, color="tab:red", zorder=4,
               marker="*", label="adaptive (live)")
    ax.set_xlabel("avg samples used")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


def main():
    st.title("Adaptive test-time compute on GSM8K")
    st.caption("A 1.5B reasoning model on a MacBook. Spend more samples only on hard "
               "questions. Move the threshold and watch the allocator.")

    try:
        study, over, analysis, meta, data = load_everything()
    except FileNotFoundError:
        st.error("No results yet. Run `python eval/run_study.py --limit 50` first.")
        return
    if not data:
        st.warning("Pool not in cache for this run. Re-run the study to populate it.")
        return

    st.sidebar.header("Allocator settings")
    threshold = st.sidebar.slider("agreement threshold (stop if early samples agree this much)",
                                  0.5, 1.0, 0.9, 0.05)
    init_n = st.sidebar.slider("initial samples", 1, 8, 4)
    max_n = meta["k"]
    st.sidebar.caption(f"model: {meta['model'].split('/')[-1]}  ·  n={meta['n_problems']}  ·  "
                       f"seed {meta['seed']}  ·  max {max_n} samples")

    rows, acc, avg_samples = run_adaptive(data, init_n, max_n, threshold)
    sc16 = next(r for r in study["runs"] if r["strategy"] == "sc@16")

    c1, c2, c3 = st.columns(3)
    c1.metric("Adaptive accuracy", f"{acc:.3f}", f"{acc - sc16['accuracy']:+.3f} vs sc@16")
    c2.metric("Avg samples / question", f"{avg_samples:.2f}", f"{16 - avg_samples:.1f} fewer than sc@16")
    c3.metric("Compute vs sc@16", f"{100 * avg_samples / 16:.0f}%", "of the samples")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Per-question allocation")
        st.caption("Each tile is one question. Number = samples the allocator spent. "
                   "Green = correct, red = wrong. Hover for detail.")
        st.markdown(grid_html(rows), unsafe_allow_html=True)
    with right:
        st.subheader("Where it lands")
        st.pyplot(pareto_fig(study, {"acc": acc, "samples": avg_samples}))

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("Strategies (fixed)")
        st.table([{"strategy": r["strategy"], "accuracy": round(r["accuracy"], 3),
                   "avg samples": round(r["avg_samples"], 1)} for r in study["runs"]
                  if r["strategy"] in ("greedy", "sc@8", "sc@16")])
        if analysis:
            st.subheader("Stopping rules (ours vs prior work)")
            st.table([{"rule": p["strategy"], "accuracy": round(p["accuracy"], 3),
                       "avg samples": round(p["avg_samples"], 2)}
                      for p in analysis["stopping_rules"]])
    with b:
        if over and (RESULTS / "overthinking.png").exists():
            st.subheader("Overthinking curve")
            st.image(str(RESULTS / "overthinking.png"))
        if (RESULTS / "summary.md").exists():
            st.subheader("Headline")
            st.markdown((RESULTS / "summary.md").read_text())

    robustness_section()
    multiseed_section()


def robustness_section():
    rob = _maybe("robustness.json")
    if not rob:
        return
    st.divider()
    st.subheader("Robustness (is it significant?)")
    ci = rob["accuracy_ci"]
    st.table([{"strategy": k, "accuracy": f"{v[0]:.3f}", "95% CI": f"[{v[1]:.2f}, {v[2]:.2f}]"}
              for k, v in ci.items()])
    d = rob["sc8_minus_sc16"]
    verdict = "significant" if d["significant"] else "NOT significant (CI spans 0)"
    st.markdown(f"**sc@8 - sc@16 = {d['diff']:+.3f}**, 95% CI "
                f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] -> *{verdict}*.")
    tc = rob["truncation_confound"]
    st.markdown("**Does the agreement signal survive controlling for truncation?** "
                f"r(agreement,correct) all = {tc['r_agreement_correct_all']:.2f}, "
                f"low-truncation half = **{tc['r_agreement_correct_low_trunc']:.2f}**, "
                f"r(truncation,correct) = {tc['r_truncation_correct']:.2f}. "
                "Truncation alone predicts nothing; the signal is stronger without it.")


def multiseed_section():
    ms = _maybe("multiseed.json")
    if not ms:
        return
    st.divider()
    st.subheader(f"Error bars across seeds {ms['meta']['seeds']}")
    st.table([{"strategy": r["strategy"],
               "accuracy": f"{r['acc_mean']:.3f} +/- {r['acc_std']:.3f}",
               "avg samples": f"{r['samples_mean']:.2f} +/- {r['samples_std']:.2f}"}
              for r in ms["aggregate"]])
    if (RESULTS / "pareto_seeds.png").exists():
        st.image(str(RESULTS / "pareto_seeds.png"))


if __name__ == "__main__":
    main()
