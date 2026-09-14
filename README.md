# Mandalay Weather Forecasting — Dual-Attention Fusion LSTM

Forecasting next-day mean temperature for Mandalay, Myanmar using 20 years
(2006-2026) of hourly ERA5 reanalysis data, aggregated to daily resolution.
All models: PyTorch.

## New scale-correction experiments

See [CONDITIONED_FUSION.md](CONDITIONED_FUSION.md) for the six controlled
comparisons, including the five-parameter seasonal/weather gate, chronological
splits, monthly metrics, and paired date-block bootstrap analysis.
These are hypotheses to test, not demonstrated improvements. The historical
interpretation below predates the scale-imbalance finding: uniform feature
attention produces a context 1/64 of the temporal context. Seed consistency
alone does not establish a structural improvement or explain adaptive-gate
performance. Global-scalar runs are already present in results.jsonl; statements
below saying they have not been run are historical.

## Results (5-seed evaluation, real committed runs)

| Model | Order | Mean MAE ± Std (°C) | Mean RMSE ± Std (°C) | Beats naive? |
|---|---|---|---|---|
| Naive persistence | — | 0.674 | — | — |
| Dual-Attention (feature-first) | Input → FeatAttn → LSTM → TempAttn → Dense | 0.700 ± 0.167 (bimodal, see below) | 0.926 ± 0.218 | No on average |
| Dual-Attention (LSTM-first) | Input → LSTM → FeatAttn → TempAttn → Dense | 0.659 ± 0.016 | 0.867 ± 0.021 | Yes |
| Parallel Fusion (adaptive, 8,256 params) | Input → LSTM → [FeatAttn, TempAttn] → learned gate → Dense | 0.638 ± 0.022 | 0.842 ± 0.027 | Yes, 4/5 seeds |
| Parallel Fusion (confidence, 0 learned params) | Input → LSTM → [FeatAttn, TempAttn] → entropy-based gate → Dense | 0.618 ± 0.003 | — | Yes, 5/5 seeds |
| Parallel Fusion (global_scalar, 1 param) | Input → LSTM → [FeatAttn, TempAttn] → single learned ratio → Dense | *not yet run* | — | — |
| **Parallel Fusion (average, 0 params)** | Input → LSTM → [FeatAttn, TempAttn] → 50/50 avg → Dense | **0.613 ± 0.002** | **0.815 ± 0.002** | **Yes, all 5 seeds** |

**Parallel fusion (average) is the best model so far** — lowest MAE and RMSE,
and by far the tightest cross-seed variance of anything tried, meaning the
improvement is structural, not a lucky seed. Notably, the learned adaptive
gate (8,256 parameters) *underperforms* both the simple fixed average (0
parameters) and the entropy-based confidence gate (0 learned parameters) -
plausible explanation: with only ~6,500 training sequences, there isn't
enough data to reliably learn a good per-example gating function, so the
extra parameters mostly add noise. This is reported honestly rather than
only showing the better variant.

**`global_scalar` is a new fourth fusion mode added to test this directly**:
a single learned mixing ratio (1 parameter, shared across every example),
sitting deliberately between `average` (0 params) and `adaptive` (128
params) on the complexity spectrum. If it performs close to `average`, that
supports the theory that per-example gating is what causes `adaptive` to
overfit - a single global ratio would then be the honest answer to "how
much adaptivity does this data actually support." Implemented and
smoke-tested (verified it has exactly 1 parameter, starts at exactly 0.5,
and moves meaningfully after a few training steps) - not yet run for a
real multi-seed comparison.

This is also a meaningfully different result from the earlier TensorFlow/
Keras version of the two sequential architectures (see
`legacy_tensorflow/README.md`: MAE 0.918 and 1.001 there — neither beat the
naive baseline). One thing carried over unchanged: **feature-first still has
a bimodal-outlier seed** (seed 5: MAE 1.034, ~65% worse than its other four
seeds), the same instability pattern seen in the TF version. It has not yet
been re-run with zero-initialized attention weights (the fix that reportedly
helped this in the TF version).

## Architectures

```
Sequential (feature-first / lstm-first):
  raw input -> LSTM -> Feature Attention -> Temporal Attention -> Prediction
  (or the reverse order)

Parallel fusion (train_parallel_fusion_attention.py):
  raw input -> LSTM -> [Feature Attention, Temporal Attention] (parallel)
            -> Fusion -> Prediction
```

Parallel fusion computes both attention mechanisms **independently** from
the same LSTM output (neither feeds into the other), then combines them via
one of four fusion modes, spanning a range of learned complexity:

- `average` (0 params) — fixed 50/50 blend. **Best result so far.**
- `global_scalar` (1 param) — a single learned mixing ratio
  `alpha = sigmoid(w)`, shared across every example and every hidden
  channel. Sits deliberately between `average` and `adaptive` in
  complexity - tests whether any global deviation from 50/50 helps,
  without the overfitting risk of a full per-example gate.
- `adaptive` (8,256 params) — a learned gate `g = sigmoid(W[v_feat; v_temp] + b)`,
  deciding the blend from the *content* of the two attention vectors, per
  example and per hidden channel. Underperformed `average` in testing.
- `confidence` (0 learned params) — a gate derived from the normalized
  entropy of each attention mechanism's own weight distribution: whichever
  branch is more "confident" (sharper, lower-entropy attention) gets more
  weight. Both entropies are normalized by their own max possible value
  (`log(hidden_dim)` for feature attention, `log(seq_len)` for temporal
  attention) before comparing, since they operate over different numbers of
  classes and raw entropy values aren't otherwise comparable.

All attention layers and the learned fusion gates are zero-initialized, so
training starts from a stable, near-uniform state (the same 50/50 point
`average` uses) rather than an arbitrary random skew.

## On novelty (read before writing this up as a paper)

The general pattern of "parallel feature+temporal attention combined by a
fusion/gating step" is **not new** — it traces back to RETAIN (2016) and
appears explicitly in recent time-series forecasting papers (e.g. VTformer's
"adaptive fusion method", a 2025 clinical-modeling paper doing parallel
BiLSTM temporal+feature attention). Framing this as a novel architecture
will not survive a literature-aware reviewer.

What appears to be a genuine gap: **no published ML/deep-learning weather
forecasting study exists for Myanmar** (checked broadly — nothing turned up
even for basic ML approaches, let alone attention-based LSTMs; nearest
regional work found was Bangladesh and Malaysia). The defensible framing is
an **empirical/application contribution**: a rigorous, multi-seed comparison
of established attention-fusion patterns, applied for the first time to
Myanmar meteorological data, including the honest finding that attention
complexity mostly doesn't beat naive persistence for day-ahead temperature
in this climate — except parallel fusion, which does, consistently.

## Setup

```bash
pip install -r requirements.txt
```

See the note at the top of `requirements.txt` for installing a
CUDA-enabled PyTorch build on Windows.

## Running

```bash
python train_feature_first_attention_torch.py --seed 42
python train_lstm_first_attention_torch.py --seed 42
python train_parallel_fusion_attention.py --fusion average --seed 42
python train_parallel_fusion_attention.py --fusion global_scalar --seed 42
python train_parallel_fusion_attention.py --fusion adaptive --seed 42
python train_parallel_fusion_attention.py --fusion confidence --seed 42
```

Common flags: `--epochs`, `--patience`, `--sequence_length`, `--batch_size`.
The parallel-fusion script additionally supports `--lr`, `--weight_decay`,
`--clip_norm`.

Each run appends one line to `results.jsonl`.

## Next steps

1. Run `train_parallel_fusion_attention.py --fusion global_scalar` for a
   real 5-seed comparison (only smoke-tested so far) - if it lands close to
   `average`, that's evidence per-example gating is what causes `adaptive`
   to overfit, not adaptivity itself
2. Re-run feature-first seed 5 (or all 5 seeds) with zero-initialized
   attention weights to check whether it resolves the bimodal instability
3. If `confidence` performs well, inspect its gate values against actual
   weather conditions (e.g. does it favor temporal attention during stable
   periods and feature attention around weather transitions?) — this maps
   directly to a paper interpretability section
4. Literature check specifically confirming the Myanmar/Southeast Asia
   coverage gap before finalizing the paper's contribution claim

## Repository structure

```
data/20years_dataset_mandalay.csv        raw hourly ERA5 data
train_feature_first_attention_torch.py   sequential, feature-attention-first
train_lstm_first_attention_torch.py      sequential, LSTM-first
train_parallel_fusion_attention.py       parallel feature+temporal attention, 3 fusion modes
models/                                  saved checkpoints
results.jsonl                            per-run results log
legacy_tensorflow/                       earlier Keras-based experiments (see its own README)
```
