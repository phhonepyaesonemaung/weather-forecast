"""
Train the parallel dual-attention fusion LSTM for Mandalay temperature
forecasting -- PyTorch.

Architecture: Input -> LSTM -> [Feature Attention, Temporal Attention]
              (parallel, both computed independently from the LSTM's
              output) -> Fusion (average or adaptive gating) -> Prediction

This differs from feature_first / lstm_first: those apply feature- and
temporal-attention SEQUENTIALLY (one feeding into the other). Here both
run in parallel from the same LSTM output and get combined by a fusion
stage. Four fusion modes, spanning a range of learned complexity:
  - "average" (0 params): fixed 50/50 blend
  - "global_scalar" (1 param): a single learned mixing ratio, shared by
    every example - the middle ground between average and adaptive
  - "adaptive" (128 params): a learned gate that decides the blend from
    the content of the two attention vectors, per example and per channel
  - "confidence" (0 learned params): a gate derived from how peaked
    (confident) each attention mechanism's own distribution is, via
    normalized entropy

Run from the weather-forecast/ project root:
    python train_parallel_fusion_attention.py
    python train_parallel_fusion_attention.py --fusion average --seed 7
    python train_parallel_fusion_attention.py --fusion adaptive --lr 0.0005
    python train_parallel_fusion_attention.py --fusion confidence --seed 7
    python train_parallel_fusion_attention.py --fusion global_scalar --seed 7

Expects data/20years_dataset_mandalay.csv relative to where you run this from.
Saves the trained model to models/parallel_fusion_<fusion>_seed<seed>.pt
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/20years_dataset_mandalay.csv")
    p.add_argument("--fusion", type=str, choices=["adaptive", "average", "confidence", "global_scalar"], default="adaptive")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--sequence_length", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="models")
    return p.parse_args()


def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        "valid_time": "date", "u10": "wind_u_10m", "v10": "wind_v_10m",
        "d2m": "dewpoint_2m", "t2m": "temperature_2m", "msl": "sea_level_pressure",
        "sp": "surface_pressure", "tcc": "total_cloud_cover", "tp": "total_precipitation",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["temperature_c"] = df["temperature_2m"] - 273.15
    df["dewpoint_c"] = df["dewpoint_2m"] - 273.15
    df["surface_pressure_hpa"] = df["surface_pressure"] / 100
    df["wind_speed_10m"] = np.sqrt(df["wind_u_10m"] ** 2 + df["wind_v_10m"] ** 2)
    df["precipitation_mm"] = df["total_precipitation"] * 1000
    df["humidity"] = 100 * (
        np.exp((17.625 * df["dewpoint_c"]) / (243.04 + df["dewpoint_c"]))
        / np.exp((17.625 * df["temperature_c"]) / (243.04 + df["temperature_c"]))
    ).clip(0, 100)

    daily = df.set_index("date").resample("D").agg({
        "temperature_c": "mean", "dewpoint_c": "mean", "humidity": "mean",
        "precipitation_mm": "sum", "wind_speed_10m": "mean",
        "total_cloud_cover": "mean", "surface_pressure_hpa": "mean",
    }).dropna()

    daily["month"] = daily.index.month
    daily["day_of_year"] = daily.index.dayofyear
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["day_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["day_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)
    for lag in [1, 2, 3, 7]:
        daily[f"temp_lag_{lag}"] = daily["temperature_c"].shift(lag)
    daily["humidity_lag_1"] = daily["humidity"].shift(1)
    daily["pressure_lag_1"] = daily["surface_pressure_hpa"].shift(1)
    daily["rain_lag_1"] = daily["precipitation_mm"].shift(1)
    daily = daily.dropna()

    feature_columns = [
        "temperature_c", "dewpoint_c", "humidity", "precipitation_mm",
        "wind_speed_10m", "total_cloud_cover", "surface_pressure_hpa",
        "month_sin", "month_cos", "day_sin", "day_cos",
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_7",
        "humidity_lag_1", "pressure_lag_1", "rain_lag_1",
    ]
    target_column = "temperature_c"

    train = daily.loc[:"2021-12-31"]
    val = daily.loc["2022-01-01":"2023-12-31"]
    test = daily.loc["2024-01-01":]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(train[feature_columns])
    X_val_scaled = feature_scaler.transform(val[feature_columns])
    X_test_scaled = feature_scaler.transform(test[feature_columns])
    y_train_scaled = target_scaler.fit_transform(train[[target_column]])
    y_val_scaled = target_scaler.transform(val[[target_column]])
    y_test_scaled = target_scaler.transform(test[[target_column]])

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled, feature_scaler, target_scaler, feature_columns)


def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# -----------------------------------------------------------------
# Feature Attention: per-timestep softmax over the LSTM's hidden
# channels, then mean-pooled over time to a fixed-size vector.
# Zero-initialized so it starts at a uniform (~1/N) softmax instead
# of an arbitrary random skew - same stability fix already used to
# address the bimodal-seed instability seen in feature_first.
# -----------------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, h):
        scores = torch.tanh(self.score(h))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.mean(h * weights, dim=1)
        return pooled, weights


# -----------------------------------------------------------------
# Temporal Attention: Bahdanau-style additive attention over the
# time axis, computed independently from Feature Attention (both
# read the same LSTM output, neither feeds into the other).
# -----------------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, units=32):
        super().__init__()
        self.W = nn.Linear(hidden_dim, units)
        self.V = nn.Linear(units, 1)
        nn.init.zeros_(self.V.weight)
        nn.init.zeros_(self.V.bias)

    def forward(self, h):
        score = self.V(torch.tanh(self.W(h)))
        weights = torch.softmax(score, dim=1)
        context = torch.sum(weights * h, dim=1)
        return context, weights


# -----------------------------------------------------------------
# Fusion: combine the feature-attention and temporal-attention
# vectors. Four modes:
#
#   "average"       - fixed 50/50 blend. 0 learned parameters.
#   "global_scalar" - ONE learned scalar mixing weight, shared across
#                      every example and every hidden channel. The
#                      middle ground between "average" (0 params) and
#                      "adaptive" (a full 128-parameter gating network):
#                      does the data support ANY global deviation from
#                      50/50, without the overfitting risk of letting
#                      the gate vary per example? 1 learned parameter.
#   "adaptive"      - a learned gate (zero-initialized so it starts at
#                      sigmoid(0) = 0.5) that decides the blend from the
#                      CONTENT of v_feat/v_temp, per example and per
#                      hidden channel - a full Linear layer, 128 params.
#   "confidence"    - a PARAMETER-FREE gate derived from how peaked
#                      (confident) each attention mechanism's own weight
#                      distribution is, via normalized entropy. Low
#                      entropy = sharp, confident attention = trusted
#                      more.
# -----------------------------------------------------------------
class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, v_feat, v_temp):
        gate = torch.sigmoid(self.gate(torch.cat([v_feat, v_temp], dim=-1)))
        fused = gate * v_feat + (1 - gate) * v_temp
        return fused, gate


class GlobalScalarFusion(nn.Module):
    """A single learned scalar mixing weight alpha = sigmoid(w), shared by
    every example and every hidden channel - unlike AdaptiveFusion, alpha
    does not depend on the input at all, so there is exactly one learned
    parameter in this whole module. Zero-initialized so it starts at the
    same 50/50 blend as "average" and only drifts away from that if the
    data supports a consistently better global ratio."""
    def __init__(self):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, v_feat, v_temp):
        alpha = torch.sigmoid(self.raw_alpha)
        fused = alpha * v_feat + (1 - alpha) * v_temp
        return fused, alpha


def confidence_fusion(v_feat, v_temp, feat_weights, temp_weights):
    """Parameter-free fusion: blend weight comes from the normalized
    entropy of each attention mechanism's own distribution, not from a
    learned function of the vectors' content.

    feat_weights: (batch, seq_len, hidden_dim) - a softmax distribution
        over hidden_dim, independently at each timestep.
    temp_weights: (batch, seq_len, 1) - a softmax distribution over
        seq_len (one distribution per example).

    The two live on different scales (max possible entropy is log(hidden_dim)
    vs log(seq_len)), so each is normalized by its own maximum before
    comparing - otherwise the branch with more classes would look
    artificially less "confident" regardless of actual sharpness.
    """
    eps = 1e-8
    hidden_dim = feat_weights.shape[-1]
    seq_len = temp_weights.shape[1]

    # Feature attention: one distribution per timestep: average entropy
    # across the window into one scalar per example.
    feat_entropy = -(feat_weights * torch.log(feat_weights + eps)).sum(dim=-1)  # (batch, seq_len)
    feat_entropy = feat_entropy.mean(dim=1) / np.log(hidden_dim)  # (batch,) in [0, 1]

    # Temporal attention: one distribution per example already.
    temp_w = temp_weights.squeeze(-1)  # (batch, seq_len)
    temp_entropy = -(temp_w * torch.log(temp_w + eps)).sum(dim=1) / np.log(seq_len)  # (batch,) in [0, 1]

    # Lower entropy = more confident = more weight. Softmax over the
    # negated entropies turns "which branch is more confident" into a
    # normalized blend weight.
    gate = torch.softmax(torch.stack([-feat_entropy, -temp_entropy], dim=-1), dim=-1)  # (batch, 2)
    g_feat = gate[:, 0:1]  # (batch, 1), broadcasts over hidden_dim

    fused = g_feat * v_feat + (1 - g_feat) * v_temp
    return fused, g_feat


class ParallelFusionDualAttentionLSTM(nn.Module):
    def __init__(self, n_features, lstm_units=64, temp_att_units=32, fusion="adaptive"):
        super().__init__()
        valid_fusions = ("average", "global_scalar", "adaptive", "confidence")
        if fusion not in valid_fusions:
            raise ValueError(f"fusion must be one of {valid_fusions}, got {fusion!r}")
        self.fusion_mode = fusion
        self.lstm = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.feature_attention = FeatureAttention(lstm_units)
        self.temporal_attention = TemporalAttention(lstm_units, temp_att_units)
        if fusion == "adaptive":
            self.fusion = AdaptiveFusion(lstm_units)
        elif fusion == "global_scalar":
            self.fusion = GlobalScalarFusion()
        else:
            self.fusion = None
        self.dense1 = nn.Linear(lstm_units, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)

        v_feat, feat_weights = self.feature_attention(lstm_out)
        v_temp, temp_weights = self.temporal_attention(lstm_out)

        if self.fusion_mode in ("adaptive", "global_scalar"):
            fused, gate = self.fusion(v_feat, v_temp)
        elif self.fusion_mode == "confidence":
            fused, gate = confidence_fusion(v_feat, v_temp, feat_weights, temp_weights)
        else:
            fused = (v_feat + v_temp) / 2
            gate = None

        h = torch.relu(self.dense1(fused))
        h = self.dropout2(h)
        output = self.output_layer(h)
        return output, feat_weights, temp_weights, gate


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (no GPU found, using CPU)"))

    print(f"Loading data from {args.data_path} ...")
    (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s,
     feature_scaler, target_scaler, feature_columns) = load_and_prepare_data(args.data_path)

    X_train, y_train = create_sequences(X_train_s, y_train_s, args.sequence_length)
    X_val, y_val = create_sequences(X_val_s, y_val_s, args.sequence_length)
    X_test, y_test = create_sequences(X_test_s, y_test_s, args.sequence_length)
    n_features = X_train.shape[2]
    print(f"Train/Val/Test sequences: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}, "
          f"{n_features} features, sequence length {args.sequence_length}")

    true_c = target_scaler.inverse_transform(y_test)
    naive_pred = target_scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1))
    naive_mae = mean_absolute_error(true_c, naive_pred)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ParallelFusionDualAttentionLSTM(n_features, fusion=args.fusion).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.patience // 2), min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _, _, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _, _, _ = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        pred_scaled, _, _, _ = model(X_test_t)
        pred_scaled = pred_scaled.cpu().numpy()

    pred_c = target_scaler.inverse_transform(pred_scaled)
    mae = mean_absolute_error(true_c, pred_c)
    rmse = np.sqrt(mean_squared_error(true_c, pred_c))

    print(f"\nParallel-Fusion Dual-Attention LSTM ({args.fusion}) -> MAE: {mae:.3f} C   RMSE: {rmse:.3f} C")
    print(f"Naive persistence baseline                          -> MAE: {naive_mae:.3f} C")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"parallel_fusion_{args.fusion}_seed{args.seed}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    result = {
        "model": f"parallel_fusion_{args.fusion}_torch", "seed": args.seed,
        "mae": mae, "rmse": rmse, "naive_mae": naive_mae,
    }
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
