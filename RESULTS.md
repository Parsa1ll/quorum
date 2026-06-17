# Results

Test-time-compute study on GSM8K with a 1.5B reasoning model, run on-device.
The claim is about compute efficiency, not raw accuracy: a 1.5B model is not
going to top GSM8K, so the question is how to spend inference compute well and
where spending more stops paying off.

## Setup

- Model: `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B` (MLX, Metal)
- Data: GSM8K `test`, first 50 problems
- Decoding: temp 0.7 / top-p 0.95 for the sampled pool, temp 0 for greedy
- `max_tokens` 1024, pool size 16
- Hardware: Apple M1 Pro, 16 GB, ~46 tok/s

Findings 1 and 2's headline (greedy vs. sampling vs. adaptive) are averaged over
three seeds. The stopping-rule comparison, the overthinking sweep, and the
agreement-signal analysis are single-seed (seed 0, n=50); they are reported as
such and not dressed up as more.

## Method notes

- **Shared pool.** Each question gets one greedy sample and a pool of 16 sampled
  completions. sc@8, sc@16, and the adaptive rules all read the same pool, so the
  accuracy gaps come from allocation, not from different draws.
- **Abstain on truncation.** A completion that runs out of token budget before it
  boxes an answer abstains rather than contributing a stray number from its
  reasoning. Across 585 finished samples, 0 finished without a parseable answer,
  so this cut is clean. 27% of samples truncated at 1024 tokens, almost all on
  the hardest problems.
- **Compute axis.** Sample count hides cost, so the headline plot
  (`results/pareto_tokens.png`) uses average generated tokens.

## Finding 1: test-time compute helps, and the gain is robust

Mean and standard deviation over three seeds:

| Strategy | Accuracy | Avg samples |
|----------|---------:|------------:|
| greedy | 0.680 +/- 0.000 | 1.00 |
| sc@8 | 0.840 +/- 0.035 | 8.00 |
| sc@16 | 0.847 +/- 0.023 | 16.00 |

Sampling and voting add about 17 points over greedy. The greedy number is
identical across seeds because it is deterministic, and sampling stays in the
0.84-0.85 range every seed, so the gain is well outside the per-seed spread.

sc@8 and sc@16 are not distinguishable here. The paired bootstrap difference is
sc@8 - sc@16 = +0.02 with a 95% CI of [0.00, 0.06], which includes zero. The sign
even flips between views: on seed 0, sc@8 (0.88) is above sc@16 (0.86); averaged
over the three seeds, sc@8 (0.840) is below sc@16 (0.847). Both readings say the
same thing, which is that voting past 8 samples does not measurably help on this
subset, not that either one is really better.

Two kinds of error bar, and they are not the same size. The +/- above is the
spread across three generation seeds with the 50 problems held fixed, so it only
captures sampling noise, not which-problems-we-drew noise. A problem-level
bootstrap (10k resamples over problems, seed 0) is much wider because n is only
50:

| Strategy | Accuracy | 95% CI (bootstrap over problems) |
|----------|---------:|:---------------------------------|
| greedy | 0.680 | [0.540, 0.800] |
| sc@8 | 0.880 | [0.780, 0.960] |
| sc@16 | 0.860 | [0.760, 0.940] |
| adaptive(t=0.6) | 0.860 | [0.760, 0.940] |

The roughly +/-0.10 width is the honest uncertainty on any single accuracy number
here. The three-seed +/-0.02 understates it; it is not a confidence interval on
the accuracy, only on the generation draw.

## Finding 2: adaptive allocation, and a prior-work rule that edges it out

Spend a few samples, then draw more only on questions where the early samples
disagree. Two stopping rules, both on the seed-0 pool (n=50), and both held to the
same 4-sample floor so the comparison is fair:

| Rule | Accuracy | Avg samples | Share of sc@16 |
|------|---------:|------------:|---------------:|
| sc@16 (fixed) | 0.860 | 16.00 | 100% |
| ours (agreement, t=0.6) | 0.860 | 6.16 | 38% |
| Beta posterior (Aggarwal-style, c=0.8, min_n=4) | 0.860 | 5.90 | 37% |

Both match sc@16 accuracy for a fraction of the samples, and the adaptive
operating point holds across the three seeds (0.847 +/- 0.023 at 6.16 +/- 0.48
samples vs. sc@16's 0.847 +/- 0.023). The prior-work rule is slightly more
efficient: a Beta posterior on the majority share (a reduction of
Adaptive-Consistency, Aggarwal et al. 2023) reaches the same accuracy at 5.90
samples vs. the naive threshold's 6.16, once both are given the same 4-sample
floor. One caveat on this margin: letting the Beta rule stop at 2 samples instead
of 4 drops it to 4.42, but that is a head start, not a smarter rule, so the matched
floor is the honest comparison. At n=50 the 5.90-vs-6.16 gap is small and within
noise; the fair statement is that the principled rule is at least as efficient, and
it is not my invention.

Two honest caveats on this table. First, the accuracy *parity* with sc@16 is
partly mechanical, not a discovery: with init_n=4 and a low threshold, every
question whose first four samples agree is routed to the same vote sc@16 would
cast, so matching its accuracy is close to guaranteed. The real result is the
compute saving, not equal accuracy. Second, the threshold t=0.6 above was picked
as the cheapest one matching sc@16 *on the same 50 problems it is scored on*,
which biases the saving optimistically. Re-doing it honestly, by tuning the
threshold on the first 25 problems and reporting it on the held-out 25, the saving
survives: adaptive reaches 0.960 at 5.9 samples vs sc@16's 0.960 at 16.0 on the
held-out half, still about 37% of the samples (`headline_heldout` in
`run_study.py`). The held-out half happens to be the easier 25 (hence 0.96), so
read the 37% figure, not the accuracy level.

## Finding 3: overthinking, and which axis to scale

Budget forcing in the style of s1: let the model think for B tokens, then splice
in a closing `</think>` and make it commit to an answer. Greedy decoding, so only
the thinking length varies (single seed, n=50).

| Budget | Accuracy | Avg tokens |
|-------:|---------:|-----------:|
| 128 | 0.280 | 137 |
| 256 | 0.460 | 264 |
| 512 | 0.760 | 448 |
| 1024 | 0.800 | 605 |
| 1536 | 0.820 | 673 |
| 2048 | 0.820 | 705 |

Accuracy climbs steeply to about 512 tokens, then flattens. Going from 512 to
2048 tokens (4x the budget) buys +0.06. More thinking stops helping well before
the model would naturally stop. At this scale it does not clearly hurt, it
saturates.

Putting both axes on one token-cost plot gives the real tradeoff:

- Parallel sampling has the higher ceiling: sc@8 reaches 0.88 on this seed.
- Sequential budget forcing peaks lower, 0.82, but reaches that peak at ~673
  tokens, about 7x fewer than sc@8's ~4966.
- At a matched ~600-token budget, forcing an answer (0.80) beats letting a single
  greedy chain truncate and abstain (0.68). A committed answer from an unfinished
  chain is worth more than a discarded one.

Neither axis dominates. Sampling buys the last few points of accuracy at a large
token cost; a single budget-forced chain is far cheaper but caps lower.

## Finding 4: is early agreement a real difficulty signal?

Adaptive stops early when the first few samples agree, so it is worth asking
whether early agreement tracks correctness. The signal is the agreement among the
first 4 samples; the label has to come from a disjoint set, because scoring it
against the full 16-sample vote would inflate the correlation by construction (the
4 are a subset of the 16). So each problem is labelled by the majority over the
held-out samples [4:16] (single seed, n=50):

- point-biserial r(early agreement, correct) = 0.18, 95% CI [-0.06, 0.52]
- r(truncation, correct) = -0.04
- mean early agreement is 0.88 when right and 0.73 when wrong

On the full set that correlation is **not statistically significant**: the
bootstrap CI includes zero, and at n=50 the critical |r| is about 0.28, which 0.18
does not clear. So the correlation by itself is suggestive, not conclusive, and it
should not be quoted as a real effect. What is solid is the plainer fact in the bin
table below: questions whose first 4 samples fully agree are right about 90% of the
time, and that is what the allocator actually relies on when it stops early. It does
not need a strong correlation to work. Truncation alone predicts nothing (r =
-0.04), so the agreement signal is not merely a truncation detector. On the
low-truncation half the correlation is stronger, r = 0.64 (n=27), but treat that as
exploratory: it is a post-hoc subgroup at half the sample size.

| early agreement | n | accuracy |
|-----------------|--:|---------:|
| [0.00, 0.50) | 5 | 0.80 |
| [0.50, 0.75) | 5 | 0.40 |
| [0.75, 1.00) | 2 | 0.50 |
| [1.00, 1.01) | 38 | 0.90 |

The catch is the distribution. 76% of questions land at full early agreement and
are ~0.90 correct, while the disagreeing minority is where the hard problems sit.
This is also why the allocator's saving does not depend on a significant
correlation: it depends on most questions being confident early and usually right,
which the bins show directly. The same distribution caps how much any adaptive rule
can save here, because most questions are easy and there is little compute to
redirect.

## Limitations

- n = 50. Findings 1-2's core comparison is averaged over three seeds; everything
  else is single-seed. Treat single-problem differences as noise. The per-problem
  bootstrap CIs in Finding 1 (~+/-0.10 wide) are the real uncertainty; the
  three-seed +/-0.02 only varies the generation draw, not which 50 problems were
  used, so it is not a CI on the accuracy.
- The 50 problems are the *first* 50 of the GSM8K test split, not a random sample,
  so they are not guaranteed to be representative of the split.
- One dataset (GSM8K). The overthinking saturation point is specific to this
  difficulty level; harder problems such as MATH would likely move it and could
  turn saturation into an actual decline.
- 27% of samples truncate at 1024 tokens. They abstain rather than corrupt the
  vote, but the hardest problems are under-sampled, which probably caps accuracy
  a little.
- A smaller-sample artifact, kept here as a caution: at n = 30 (the first 30
  problems), sc@16 happened to dip to 0.80, which made budget forcing look like
  it matched self-consistency at far fewer tokens. At n = 50 that apparent parity
  is gone. The early "matches at 15x fewer tokens" reading was an artifact of the
  smaller sample, not a real result.
