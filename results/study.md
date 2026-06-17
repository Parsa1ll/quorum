| Strategy | Accuracy | Avg samples | Avg tokens |
|----------|---------:|------------:|-----------:|
| greedy | 0.680 | 1.0 | 604 |
| sc@8 | 0.880 | 8.0 | 4966 |
| sc@16 | 0.860 | 16.0 | 10159 |
| adaptive(t=0.6) | 0.860 | 6.2 | 4101 |
| adaptive(t=0.7) | 0.860 | 6.4 | 4198 |
| adaptive(t=0.8) | 0.860 | 6.9 | 4435 |
| adaptive(t=0.9) | 0.860 | 6.9 | 4435 |
| adaptive(t=1.0) | 0.860 | 6.9 | 4435 |

In-sample (threshold chosen and scored on the same 50 problems, so the saving is optimistically biased):
adaptive(t=0.6) reaches 0.860 acc vs sc@16 0.860 using 38% of the samples (6.2 vs 16.0).

Held-out (threshold tuned on one half, reported on the other half; this is the honest saving):
tuned t=0.6 on first 25 problems, reported on held-out 25: adaptive 0.960 @ 5.9 samples vs sc@16 0.960 @ 16.0 (37% of samples).
