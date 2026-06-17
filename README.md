# Quorum

Quorum is a small study of how a 1.5B reasoning model should spend extra compute
when it answers, run start to finish on a MacBook with Apple MLX. No cluster, no
API. I built it to learn how this actually works by measuring it myself.

**Live demo and write-up: [quorum-ml.vercel.app](https://quorum-ml.vercel.app/)**

A 1.5B model will not top any leaderboard, so this is not about raw accuracy. It is
about efficiency: sampling more answers or thinking longer both cost compute, so
which one actually buys accuracy, and where does it stop paying off?

What I found:

1. **Sampling beats answering once.** Drawing several answers and voting lifts
   accuracy about 17 points over a single greedy answer.
2. **Spend only where it is needed.** Taking more samples only on questions where
   the first few disagree hits the same accuracy as always taking 16, for about a
   third of the samples. A known rule from prior work (Adaptive-Consistency) does it
   on a little less, once both rules get the same 4-sample floor.
3. **Overthinking is real.** Make the model think for a fixed budget and sweep it:
   accuracy climbs fast to about 512 tokens, then flattens. More thinking past that
   barely helps here.
4. **The agreement signal is honest, not magic.** On the full 50 questions, "early
   answers agree" is not a statistically significant predictor of a correct answer.
   The method still works, because questions that agree early happen to be right
   about 90% of the time.

Full numbers, error bars, and honest caveats are in [RESULTS.md](RESULTS.md). There
is also a plain-language write-up with an interactive demo in [docs/](docs/), the
Quorum site.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py     # downloads the model on first run, then offline
```

Tested on an Apple M1 Pro (16 GB), Python 3.9+, `mlx`/`mlx-lm` 0.18, with
`mlx-community/DeepSeek-R1-Distill-Qwen-1.5B`.

## Reproduce

Generation is the only slow part and it is cached to `results/cache.jsonl`, so
re-runs only re-score. Start with a small `--limit` and grow it.

```bash
python eval/run_study.py    --limit 50 --seed 0   # greedy / sc / adaptive -> study.*
python eval/overthinking.py --limit 50 --seed 0   # budget-forcing sweep
python eval/analysis.py     --limit 50 --seed 0   # posterior stopping rule + signal check
python eval/robustness.py   --limit 50 --seed 0   # bootstrap CIs, truncation control
python eval/multiseed.py    --limit 50 --seeds 0 1 2   # error bars across seeds
python eval/plot_pareto.py                        # combined accuracy-vs-tokens plot
python eval/export_web.py   --limit 50 --seed 0   # build docs/data.js for the site
streamlit run app.py                              # interactive allocator demo (local)
```

## Demo

A one-page write-up with an interactive version of the allocator is in `docs/`,
deployed as a static site on Vercel. The widget recomputes the strategy in the
browser from the cached pool, so the numbers match the study exactly. `app.py` is
the fuller local version (tables, error bars, robustness) and needs `streamlit run`.

## Layout

```
src/
  model.py        mlx-lm generation: one sampled completion, seeding, token counts
  strategies.py   greedy / self-consistency / adaptive, as pure functions over a pool
eval/
  gsm8k.py        load GSM8K, extract and score answers, the agreement signal
  cache.py        append-only, resumable jsonl cache for generations
  generate.py     draw and cache the shared per-question sample pool
  run_study.py    score the strategies, write the table and Pareto plot
  overthinking.py budget forcing: accuracy vs. thinking length
  analysis.py     posterior (Beta) stopping rule and agreement-signal validity
  robustness.py   bootstrap CIs and the truncation confound check
  multiseed.py    repeat across seeds, aggregate mean and std
  plot_pareto.py  combined accuracy-vs-tokens view
  export_web.py   write docs/data.js for the website
scripts/
  smoke_test.py   one-question sanity check
app.py            streamlit demo of the allocator (local)
docs/             one-page static site (deployed on Vercel)
results/          tables, plots, and the generation cache (kept in the repo)
```

## Method notes

- **Shared pool.** Each question gets one greedy sample and a pool of 16 sampled
  completions. Every strategy reads the same pool, so accuracy differences come
  from how compute is allocated, not from separate lucky draws. Each sample is
  seeded from `(seed, question, index)`, so a run is reproducible and resumable.
- **Abstain on truncation.** A completion that hits the token limit before it
  boxes an answer abstains instead of contributing a number pulled from the
  middle of its reasoning. Across 585 finished samples, 0 finished without a
  parseable answer, so the rule does not silently drop valid answers.
- **Token cost, not sample count.** A truncated 1024-token sample costs far more
  than a 300-token one, so the headline efficiency plot uses average generated
  tokens rather than number of samples.

## References

- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning* (2022).
- Aggarwal et al., *Let's Sample Step by Step: Adaptive-Consistency* (EMNLP 2023).
- Muennighoff et al., *s1: Simple Test-Time Scaling* (2025).

## License

MIT, see [LICENSE](LICENSE).
