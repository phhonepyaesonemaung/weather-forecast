# Mandalay Weather Forecasting — Dual-Attention Fusion LSTM

Forecasting next-day mean temperature for Mandalay, Myanmar using 20 years
(2006-2026) of hourly ERA5 reanalysis data, aggregated to daily resolution.
All models: PyTorch.

## Results so far (5-seed evaluation, real committed runs)

| Model | Order | Mean MAE ± Std (°C) | Mean RMSE ± Std (°C) |
|---|---|---|---|
| Naive persistence | — | 0.674 | — |
| Dual-Attention (feature-first) | Input → FeatAttn → LSTM → TempAttn → Dense | 0.700 ± 0.167 (bimodal, see below) | 0.926 ± 0.218 |
| Dual-Attention (feature-first), excl. seed 5 | same | 0.617 ± 0.005 | — |
| Dual-Attention (LSTM-first) | Input → LSTM → FeatAttn → TempAttn → Dense | 0.659 ± 0.016 | 0.867 ± 0.021 |

**This is a meaningfully different result from the earlier TensorFlow/Keras
version of these same two architectures** (see `legacy_tensorflow/README.md`
for those numbers, MAE 0.918 and 1.001 respectively — neither beat the
naive baseline there). In this PyTorch reimplementation, LSTM-first now
reliably beats naive persistence on average, and feature-first does too on
4 of 5 seeds. One thing carried over unchanged: **feature-first still has
a bimodal-outlier seed** (seed 5: MAE 1.034, roughly 65% worse than its
other four seeds) — same instability pattern seen in the TF version. It
was not yet re-run with zero-initialized attention weights, which is the
fix that addressed this in project notes for the TF version; worth trying
here too before drawing final conclusions about feature-first's stability.

## New architecture: parallel fusion (`train_parallel_fusion_attention.py`)

```
raw input -> LSTM -> [Feature Attention, Temporal Attention] (parallel)
          -> Fusion (average or adaptive gating) -> Prediction
```

This is a different structure from both models above: those apply
feature- and temporal-attention **sequentially** (one feeding into the
other, in one of two orders). Here, both attention mechanisms are
computed **independently and in parallel** from the same LSTM output,
then combined by a fusion stage:

- `average`: fixed 50/50 blend of the two attention outputs
- `adaptive`: a learned gate `g = sigmoid(W[v_feat; v_temp] + b)`, so the
  model decides per-example, per-hidden-channel how much to trust the
  feature-attention view vs. the temporal-attention view

Both attention layers and the adaptive fusion gate are zero-initialized,
so training starts from a stable, near-uniform state rather than an
arbitrary random skew.

**Status: implemented and smoke-tested (a few epochs, confirms the code
runs correctly end-to-end with no shape/logic errors), not yet run for a
real multi-seed comparison.** That's the next step — see below.

A parallel fusion of feature- and temporal-attention for LSTM time-series
forecasting is a less common combination in the literature than sequential
dual-attention. Worth a targeted literature check before writing this up
as a novel contribution, but it hasn't been tried in this project before.

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
python train_parallel_fusion_attention.py --fusion adaptive --seed 42
python train_parallel_fusion_attention.py --fusion average --seed 42
```

Common flags: `--epochs`, `--patience`, `--sequence_length`, `--batch_size`.
The parallel-fusion script additionally supports `--lr`, `--weight_decay`,
`--clip_norm`.

Each run appends one line to `results.jsonl`.

## Next steps

1. Run `train_parallel_fusion_attention.py` for real (5 seeds x both
   fusion modes) to get comparable numbers to the table above
2. Re-run `train_feature_first_attention_torch.py` seed 5 (or all 5
   seeds) with zero-initialized attention weights to check whether that
   resolves the bimodal instability, same as it reportedly did for the
   TensorFlow version
3. If `adaptive` beats `average` and both existing architectures
   consistently, inspect the learned gate values to understand what the
   model is actually keying off
4. Literature check on parallel feature+temporal attention fusion for
   LSTM time-series forecasting, to confirm how novel this framing is

## Repository structure

```
data/20years_dataset_mandalay.csv        raw hourly ERA5 data
train_feature_first_attention_torch.py   sequential, feature-attention-first
train_lstm_first_attention_torch.py      sequential, LSTM-first
train_parallel_fusion_attention.py       new: parallel feature+temporal attention, fused
models/                                  saved checkpoints (git-ignored)
results.jsonl                            per-run results log
legacy_tensorflow/                       earlier Keras-based experiments (see its own README)
```
