"""
Train the LSTM-first dual-attention LSTM for Mandalay temperature forecasting
-- PyTorch version.

Architecture: Input -> LSTM -> Feature Attention -> Temporal Attention -> Prediction

Run from the weather-forecast/ project root:
    python train_lstm_first_attention_torch.py
    python train_lstm_first_attention_torch.py --seed 7 --epochs 100 --patience 15

Expects data/20years_dataset_mandalay.csv relative to where you run this from.
Saves the trained model to models/lstm_first_attention_seed<seed>.pt
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--sequence_length", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
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
# Feature Attention: per-timestep softmax over the feature axis.
# Applied AFTER the LSTM in this version, so it reweights the LSTM's
# 64 hidden units rather than the 18 raw input features.
# -----------------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.score = nn.Linear(n_features, n_features)

    def forward(self, x):
        scores = torch.tanh(self.score(x))
        weights = torch.softmax(scores, dim=-1)
        return x * weights, weights


# -----------------------------------------------------------------
# Temporal Attention: Bahdanau-style additive attention over the
# time axis, applied to the feature-attention-weighted sequence.
# -----------------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, units=32):
        super().__init__()
        self.W = nn.Linear(hidden_dim, units)
        self.V = nn.Linear(units, 1)

    def forward(self, h):
        score = self.V(torch.tanh(self.W(h)))
        weights = torch.softmax(score, dim=1)
        context = torch.sum(weights * h, dim=1)
        return context, weights


class LSTMFirstDualAttentionLSTM(nn.Module):
    def __init__(self, n_features, lstm_units=64, temp_att_units=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.feature_attention = FeatureAttention(lstm_units)
        self.temporal_attention = TemporalAttention(lstm_units, temp_att_units)
        self.dense1 = nn.Linear(lstm_units, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)
        feat_weighted, feat_weights = self.feature_attention(lstm_out)
        context, temp_weights = self.temporal_attention(feat_weighted)
        h = torch.relu(self.dense1(context))
        h = self.dropout2(h)
        output = self.output_layer(h)
        return output, feat_weights, temp_weights


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

    model = LSTMFirstDualAttentionLSTM(n_features).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
            pred, _, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _, _ = model(xb)
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
        pred_scaled, _, _ = model(X_test_t)
        pred_scaled = pred_scaled.cpu().numpy()

    pred_c = target_scaler.inverse_transform(pred_scaled)
    mae = mean_absolute_error(true_c, pred_c)
    rmse = np.sqrt(mean_squared_error(true_c, pred_c))

    print(f"\nLSTM-first Dual-Attention LSTM -> MAE: {mae:.3f} C   RMSE: {rmse:.3f} C")
    print(f"Naive persistence baseline     -> MAE: {naive_mae:.3f} C")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"lstm_first_attention_seed{args.seed}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    result = {"model": "lstm_first_torch", "seed": args.seed, "mae": mae, "rmse": rmse, "naive_mae": naive_mae}
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
