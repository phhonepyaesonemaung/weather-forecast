"""
Train the LSTM-first dual-attention LSTM for Mandalay temperature forecasting.

Architecture: Input -> LSTM -> Feature Attention -> Temporal Attention -> Prediction
(this is the exact ordering: LSTM, then feature attention, then temporal attention,
then prediction)

Run from the weather-forecast/ project root:
    python train_lstm_first_attention.py
    python train_lstm_first_attention.py --seed 7 --epochs 100 --patience 15

Expects data/20years_dataset_mandalay.csv relative to where you run this from.
Saves the trained model to models/lstm_first_attention_seed<seed>.keras
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


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
    return np.array(X_seq), np.array(y_seq)


def main():
    args = parse_args()

    import tensorflow as tf
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs detected: {gpus if gpus else 'none (running on CPU)'}")

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

    # -----------------------------------------------------------------
    # Feature Attention layer: per-timestep softmax over the feature
    # axis. Applied AFTER the LSTM in this version, so it reweights the
    # LSTM's 64 hidden units rather than the 18 raw input features.
    # -----------------------------------------------------------------
    class FeatureAttention(layers.Layer):
        def build(self, input_shape):
            n_feat = input_shape[-1]
            self.score_dense = layers.Dense(n_feat, activation="tanh")
            super().build(input_shape)

        def call(self, x):
            scores = self.score_dense(x)
            weights = tf.nn.softmax(scores, axis=-1)
            return x * weights, weights

    # -----------------------------------------------------------------
    # Temporal Attention layer: Bahdanau-style additive attention over
    # the time axis, applied to the feature-attention-weighted sequence.
    # -----------------------------------------------------------------
    class TemporalAttention(layers.Layer):
        def __init__(self, units=32, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.W = layers.Dense(self.units, activation="tanh")
            self.V = layers.Dense(1)
            super().build(input_shape)

        def call(self, h):
            score = self.V(self.W(h))
            weights = tf.nn.softmax(score, axis=1)
            context = tf.reduce_sum(weights * h, axis=1)
            return context, weights

    # Architecture: Input -> LSTM -> Feature Attention -> Temporal Attention -> Prediction
    inputs = keras.Input(shape=(args.sequence_length, n_features), name="input_sequence")
    lstm_out = layers.LSTM(64, return_sequences=True, name="lstm")(inputs)
    lstm_out = layers.Dropout(0.2)(lstm_out)
    feat_weighted, feat_weights = FeatureAttention(name="feature_attention")(lstm_out)
    context, temp_weights = TemporalAttention(units=32, name="temporal_attention")(feat_weighted)
    x = layers.Dense(16, activation="relu")(context)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, name="prediction")(x)

    model = keras.Model(inputs, output, name="lstm_first_dual_attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mse", metrics=["mae"],
    )
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, args.patience // 2), min_lr=1e-5)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    pred_c = target_scaler.inverse_transform(model.predict(X_test, verbose=0))
    mae = mean_absolute_error(true_c, pred_c)
    rmse = np.sqrt(mean_squared_error(true_c, pred_c))

    print(f"\nLSTM-first Dual-Attention LSTM -> MAE: {mae:.3f} C   RMSE: {rmse:.3f} C")
    print(f"Naive persistence baseline     -> MAE: {naive_mae:.3f} C")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"lstm_first_attention_seed{args.seed}.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    result = {"model": "lstm_first", "seed": args.seed, "mae": mae, "rmse": rmse, "naive_mae": naive_mae}
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
