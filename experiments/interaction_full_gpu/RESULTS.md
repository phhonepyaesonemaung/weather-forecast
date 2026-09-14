# Interaction fusion full-run results

All 25 prediction reports and checkpoint paths were checked; MAEs were recomputed.
Metrics below are means of five seed metrics, not ensemble forecast errors.

| Mode | Mean MAE C | Seed SD MAE C | Mean RMSE C |
|---|---:|---:|---:|
| scaled_seasonal | 0.612182 | 0.004314 | 0.810967 |
| scaled_global | 0.613292 | 0.003877 | 0.814373 |
| scaled_average | 0.614640 | 0.004952 | 0.813958 |
| interaction | 0.615284 | 0.005949 | 0.816590 |
| joint | 0.621730 | 0.009467 | 0.821253 |

Paired bootstrap, 30-day blocks, 5,000 samples. Negative favors interaction.

| Reference | Interaction minus reference MAE C | 95% interval C |
|---|---:|---|
| scaled_average | +0.000643 | [-0.004922, +0.006649] |
| scaled_global | +0.001991 | [-0.003904, +0.008771] |
| scaled_seasonal | +0.003102 | [-0.002450, +0.009768] |
| joint | -0.006447 | [-0.012712, +0.001267] |

Validation diagnostics from each selected interaction checkpoint:

| Seed | tanh(lambda) | Correction/base mean norm ratio |
|---|---:|---:|
| 1 | 0.822249 | 0.425778 |
| 2 | 0.745333 | 0.421838 |
| 3 | 0.739321 | 0.531112 |
| 4 | 0.600673 | 0.309851 |
| 5 | 0.742584 | 0.457925 |

See summary.json for 7/90-day block sensitivity and full diagnostics.
Intervals describe date uncertainty conditional on these trained seeds. The test period
has already informed model design; these are exploratory comparisons, with no multiple-comparison correction.
No novelty or independent generalization claim follows from this run.
