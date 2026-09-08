# Mandalay Weather Forecasting — Dual-Attention Fusion LSTM

Forecasting next-day mean temperature for Mandalay, Myanmar using 20 years
(2006-2026) of hourly ERA5 reanalysis data, aggregated to daily resolution.
All models: PyTorch.

## Results (5-seed evaluation, real committed runs)

| Model | Order | Mean MAE ± Std (°C) | Mean RMSE ± Std (°C) | Beats naive? |
|---|---|---|---|---|
| Naive persistence | — | 0.674 | — | — |
| Dual-Attention (feature-first) | Input → FeatAttn → LSTM → TempAttn → Dense | 0.700 ± 0.167 (bimodal, see below) | 0.926 ± 0.218 | No on average |
| Dual-Attention (LSTM-first) | Input → LSTM → FeatAttn → TempAttn → Dense | 0.659 ± 0.016 | 0.867 ± 0.021 | Yes |
| Parallel Fusion (adaptive) | Input → LSTM → [FeatAttn, TempAttn] → learned gate → Dense | 0.638 ± 0.022 | 0.842 ± 0.027 | Yes, 4/5 seeds |
| **Parallel Fusion (average)** | Input → LSTM → [FeatAttn, TempAttn] → 50/50 avg → Dense | **0.613 ± 0.002** | **0.815 ± 0.002** | **Yes, all 5 seeds** |

**Parallel fusion (average) is the best model so far** — lowest MAE and RMSE,
and by far the tightest cross-seed variance of anything tried, meaning the
improvement is structural, not a lucky seed. Notably, the learned adaptive
gate *underperforms* the simple fixed average — plausible explanation: with
only ~6,500 training sequences, there isn't enough data to reliably learn a
good per-example gating function, so the extra parameters mostly add noise.
This is reported honestly rather than only showing the better variant.

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
one of three fusion modes:

- `average` — fixed 50/50 blend. **Best result so far.**
- `adaptive` — a learned gate `g = sigmoid(W[v_feat; v_temp] + b)`, deciding
  the blend from the *content* of the two attention vectors. Adds
  parameters; underperformed `average` in testing.
- `confidence` — a **parameter-free** gate derived from the normalized
  entropy of each attention mechanism's own weight distribution: whichever
  branch is more "confident" (sharper, lower-entropy attention) gets more
  weight. Both entropies are normalized by their own max possible value
  (`log(hidden_dim)` for feature attention, `log(seq_len)` for temporal
  attention) before comparing, since they operate over different numbers of
  classes and raw entropy values aren't otherwise comparable. Implemented
  and smoke-tested (verified the gate starts at 0.5 with zero-init
  attention, and moves away from 0.5 once attention sharpens through
  training) — **not yet run for a real multi-seed comparison.**

All attention layers and the adaptive fusion gate are zero-initialized, so
training starts from a stable, near-uniform state rather than an arbitrary
random skew.

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
python train_parallel_fusion_attention.py --fusion adaptive --seed 42
python train_parallel_fusion_attention.py --fusion confidence --seed 42
```

Common flags: `--epochs`, `--patience`, `--sequence_length`, `--batch_size`.
The parallel-fusion script additionally supports `--lr`, `--weight_decay`,
`--clip_norm`.

Each run appends one line to `results.jsonl`.

## Next steps

1. Run `train_parallel_fusion_attention.py --fusion confidence` for a real
   5-seed comparison (only smoke-tested so far)
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
