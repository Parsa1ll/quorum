# Stopping rules (n=50)

| Rule | Accuracy | Avg samples |
|------|---------:|------------:|
| ours(t=0.6) | 0.860 | 6.16 |
| ours(t=0.8) | 0.860 | 6.88 |
| ours(t=0.9) | 0.860 | 6.88 |
| ours(t=1.0) | 0.860 | 6.88 |
| beta(c=0.8) | 0.860 | 5.90 |
| beta(c=0.9) | 0.860 | 6.82 |
| beta(c=0.95) | 0.860 | 7.78 |
| beta(c=0.99) | 0.860 | 10.38 |

## Is early agreement a valid difficulty signal?
- point-biserial r(early agreement, correct) = 0.184
- mean early agreement when correct: 0.8760162601626016
- mean early agreement when wrong:   0.7314814814814814

| early agreement | n | accuracy |
|---|---:|---:|
| [0.00,0.50) | 5 | 0.800 |
| [0.50,0.75) | 5 | 0.400 |
| [0.75,1.00) | 2 | 0.500 |
| [1.00,1.01) | 38 | 0.895 |
