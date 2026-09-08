# Mandalay Weather Forecasting with LSTM + Dual Attention

Forecasting next-day mean temperature for Mandalay, Myanmar using 20 years (2006–2026)
of hourly ERA5 reanalysis data, aggregated to daily resolution. Four approaches are
implemented and compared in one notebook.

## Structure
- `data/20years_dataset_mandalay.csv` — raw hourly ERA5 data
- `mandalay_lstm_forecast.ipynb`:
  - **Part 1**: cleaning, feature engineering, chronological split, plain LSTM, naive baseline
  - **Part 2**: dual-attention LSTM, **feature-attention-first** ordering
    (Input → Feature Attention → LSTM → Temporal Attention → Prediction)
  - **Part 3**: dual-attention LSTM, **LSTM-first** ordering
    (Input → LSTM → Feature Attention → Temporal Attention → Prediction)
  - **Part 4**: final comparison table across all four models
- `models/` — trained model artifacts (not tracked in git)

## Results — multi-seed comparison (the honest headline numbers)

| Model | Order | Mean MAE ± Std (°C) |
|---|---|---|
| Naive persistence | — | 0.674 ± 0.000 |
| Plain LSTM | Input → LSTM → Dense | 0.752 ± 0.061 |
| Dual-Attention (feature-first) | Input → FeatAttn → LSTM → TempAttn → Dense | 0.918 ± 0.238 (bimodal) |
| Dual-Attention (LSTM-first) | Input → LSTM → FeatAttn → TempAttn → Dense | 1.001 ± 0.162 |

**Key finding:** none of the three LSTM variants reliably beats the naive persistence
baseline on average for 1-day-ahead temperature at this single station — day-ahead
temperature in a tropical climate is dominated by seasonal persistence. Between the two
attention orderings, feature-first performs closer to baseline on average; LSTM-first
(applying feature attention to the LSTM's compressed hidden state rather than the 18
original physical features) landed in a weaker optimum in most seeds. Both orderings are
implemented, evaluated, and kept in the notebook — see Part 4 for full discussion.

## Setup and how to run
```bash
pip install -r requirements.txt
jupyter notebook mandalay_lstm_forecast.ipynb
# Then: Cell -> Run All (or run top to bottom)
```
Runtime: roughly 8–15 minutes on CPU for the full notebook (Parts 1–4), faster on GPU.

## Next steps
See the "Notes / next steps" sections throughout the notebook.
