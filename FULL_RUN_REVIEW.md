# CPU full-run review

Verified 30 runs in experiments/full_run: six modes, seeds 1–5, matching
configurations, 934 prediction dates per run, saved checkpoints, finite
predictions, consistent monthly counts, and MAEs recomputed from prediction CSVs.
The runs used CPU PyTorch 2.14.0, batch size 32, up to 100 epochs, and the default
chronological split. This review does not retrain the models.

| Mode | Mean MAE, C | Mean RMSE, C |
|---|---:|---:|
| average | 0.635168 | 0.838433 |
| scaled_average | 0.616070 | 0.817759 |
| scaled_global | 0.618165 | 0.817325 |
| scaled_seasonal | 0.621586 | 0.821949 |
| scaled_weather | 0.617403 | 0.818373 |
| temporal_only | 0.619508 | 0.820375 |

These are averages of each seed's metrics, not metrics of an ensemble.
Scaled average has the lowest mean MAE; scaled global has the lowest mean RMSE.

Paired date-block bootstrap (5,000 samples, 30-day blocks, five-seed mean
absolute errors; negative differences favor the candidate):

| Candidate minus reference | MAE difference, C | 95% interval, C |
|---|---:|---|
| scaled_average minus average | -0.019098 | [-0.027612, -0.008624] |
| scaled_weather minus scaled_average | +0.001333 | [-0.004584, +0.007219] |
| temporal_only minus scaled_average | +0.003438 | [-0.004069, +0.009373] |

The scale-correction improvement also has intervals below zero with 7-day
blocks ([-0.027582, -0.010395]) and 90-day blocks ([-0.028762, -0.005837]).
This supports scale correction relative to the reference in this experiment.
It does not demonstrate an advantage for the weather gate or clearly establish
that the feature branch improves on temporal-only attention.

The original-average mean here (0.635168) differs from historical results.jsonl
(approximately 0.6129). The new runner resets the random stream after model
construction and uses a dedicated shuffle generator, and computes sample-weighted
validation loss. The same seed label therefore does not recreate an old run.
Compare modes within this controlled run; these results do not beat the historical
best recorded mean MAE. The historical test period has already informed selection,
so conclusions remain exploratory pending independent chronological evaluation.

The bootstrap quantifies date uncertainty conditional on these trained seeds.
It does not include full training-seed uncertainty. Original-average seed 4 is
noticeably worse than its other seeds, so broader seed replication would help
assess robustness before making a general claim about scale correction.
