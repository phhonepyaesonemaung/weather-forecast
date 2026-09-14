"""Controlled scale-correction ablations; see CONDITIONED_FUSION.md."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_parallel_fusion_attention import (
    ParallelFusionDualAttentionLSTM, load_and_prepare_data,
)

MODES = ("average", "scaled_average", "scaled_global", "scaled_seasonal",
         "scaled_weather", "temporal_only")


class ConditionedFusionLSTM(ParallelFusionDualAttentionLSTM):
    def __init__(self, n_features, mode="scaled_weather", lstm_units=64):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        super().__init__(n_features, lstm_units=lstm_units, fusion="average")
        self.mode = mode
        if mode == "temporal_only":
            self.feature_attention = None
        self.gate_layer = None
        if mode in ("scaled_seasonal", "scaled_weather"):
            self.gate_layer = nn.Linear(2 if mode == "scaled_seasonal" else 4, 1)
            nn.init.zeros_(self.gate_layer.weight)
            nn.init.zeros_(self.gate_layer.bias)
        if mode == "scaled_global":
            self.raw_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        h, _ = self.lstm(x)
        h = self.dropout1(h)
        vt, _ = self.temporal_attention(h)
        gate = torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)
        if self.mode == "temporal_only":
            fused = vt
        else:
            vf, _ = self.feature_attention(h)
            if self.mode != "average":
                vf = vf * h.shape[-1]
            gate = torch.full_like(gate, 0.5)
            if self.mode == "scaled_global":
                gate = torch.sigmoid(self.raw_gate).expand_as(gate)
            elif self.gate_layer is not None:
                gate = torch.sigmoid(self.gate_layer(context[:, :self.gate_layer.in_features]))
            fused = gate * vf + (1 - gate) * vt
        pred = self.output_layer(self.dropout2(torch.relu(self.dense1(fused))))
        return pred, gate


def prepare(daily, columns, length, train_end, val_end, test_end):
    """Split-contained windows match the original runner's evaluation dates."""
    if length < 1:
        raise ValueError("sequence_length must be positive")
    train_end, val_end, test_end = map(pd.Timestamp, (train_end, val_end, test_end))
    if not train_end < val_end < test_end:
        raise ValueError("Require train_end < val_end < test_end")
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("Daily dates must be unique and increasing")
    if not (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Missing daily observations; windows must contain consecutive days")
    masks = [daily.index <= train_end,
             (daily.index > train_end) & (daily.index <= val_end),
             (daily.index > val_end) & (daily.index <= test_end)]
    if any(mask.sum() <= length for mask in masks):
        raise ValueError("Each split needs more days than sequence_length")
    changes = pd.DataFrame({
        "humidity_change": daily.humidity - daily.humidity_lag_1,
        "pressure_change": daily.surface_pressure_hpa - daily.pressure_lag_1,
    }, index=daily.index)
    xs = MinMaxScaler().fit(daily.loc[masks[0], columns])
    ys = MinMaxScaler().fit(daily.loc[masks[0], ["temperature_c"]])
    gs = StandardScaler().fit(changes.loc[masks[0]])
    context = np.column_stack([daily.day_sin, daily.day_cos, gs.transform(changes)])
    datasets, dates, persistence = [], [], []
    for mask in masks:
        part = daily.loc[mask]
        x = xs.transform(part[columns])
        y = ys.transform(part[["temperature_c"]])
        c = context[mask]
        # Context is from t (last observed day), target is t+1.
        datasets.append(TensorDataset(
            torch.tensor(np.stack([x[i-length:i] for i in range(length, len(x))]), dtype=torch.float32),
            torch.tensor(c[length-1:-1], dtype=torch.float32),
            torch.tensor(y[length:], dtype=torch.float32)))
        dates.append(part.index[length:])
        persistence.append(part.temperature_c.to_numpy()[length-1:-1])
    scalers = {"feature_min": xs.min_.tolist(), "feature_scale": xs.scale_.tolist(),
               "target_min": ys.min_.tolist(), "target_scale": ys.scale_.tolist(),
               "gate_mean": gs.mean_.tolist(), "gate_scale": gs.scale_.tolist()}
    return datasets, dates, persistence, ys, scalers


def metrics(frame):
    error = frame.prediction_c.to_numpy() - frame.actual_c.to_numpy()
    naive = frame.persistence_c.to_numpy() - frame.actual_c.to_numpy()
    return {"n": len(frame), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "persistence_mae": float(np.abs(naive).mean())}


def run(args, model_factory=ConditionedFusionLSTM, diagnostics=None, auxiliary_column="feature_gate"):
    torch.set_num_threads(args.threads)
    daily, columns = load_and_prepare_data(args.data_path, return_daily=True)
    datasets, dates, persistence, ys, scalers = prepare(
        daily, columns, args.sequence_length, args.train_end, args.val_end, args.test_end)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable. Install CUDA-enabled PyTorch in this environment.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else torch.device(requested_device)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    for seed in args.seeds:
        for mode in args.modes:
            name = f"{mode}_seed{seed}"
            if any((out / (name + suffix)).exists() for suffix in (".pt", ".csv", ".json")):
                raise FileExistsError(f"Existing run {name}; choose a fresh output_dir")
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model = model_factory(len(columns), mode).to(device)
            # Reset RNG after construction so all modes share shuffle/dropout streams.
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, generator=generator)
            val = DataLoader(datasets[1], batch_size=args.batch_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=max(3, args.patience // 2), min_lr=1e-5)
            best, state, stale, best_epoch = float("inf"), None, 0, 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                for x, c, y in train:
                    x, c, y = x.to(device), c.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(model(x, c)[0], y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                    optimizer.step()
                model.eval()
                total = 0.0
                with torch.no_grad():
                    for x, c, y in val:
                        pred, _ = model(x.to(device), c.to(device))
                        total += nn.functional.mse_loss(pred, y.to(device), reduction="sum").item()
                score = total / len(datasets[1])
                if not np.isfinite(score):
                    raise RuntimeError("Non-finite validation loss")
                scheduler.step(score)
                print(f"{name} epoch {epoch}: val_mse={score:.6f}", flush=True)
                if score < best - 1e-6:
                    best, state, stale, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            model.load_state_dict(state)
            model.eval()
            predictions, gates = [], []
            with torch.no_grad():
                for x, c, _ in DataLoader(datasets[2], batch_size=args.batch_size):
                    pred, gate = model(x.to(device), c.to(device))
                    predictions.append(pred.cpu().numpy())
                    gates.append(gate.cpu().numpy())
            frame = pd.DataFrame({"date": dates[2],
                "actual_c": ys.inverse_transform(datasets[2].tensors[2].numpy()).ravel(),
                "prediction_c": ys.inverse_transform(np.concatenate(predictions)).ravel(),
                "persistence_c": persistence[2], auxiliary_column: np.concatenate(gates).ravel()})
            result = {"mode": mode, "seed": seed, "best_epoch": best_epoch,
                "best_val_mse": best, "metrics": metrics(frame),
                "monthly": {str(month): metrics(group) for month, group in frame.groupby(frame.date.dt.to_period("M"))},
                "config": vars(args), "features": columns, "scalers": scalers,
                "torch_version": torch.__version__, "device": str(device)}
            if diagnostics is not None:
                result["diagnostics"] = diagnostics(model, datasets[1], args.batch_size, device)
            frame.to_csv(out / f"{name}.csv", index=False)
            (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            torch.save({"state_dict": {k: v.cpu() for k, v in state.items()}, "metadata": result}, out / f"{name}.pt")
            print(f"{name}: {result['metrics']}", flush=True)


def parse_args(modes=MODES, output_dir="experiments/conditioned_fusion"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/20years_dataset_mandalay.csv")
    p.add_argument("--modes", nargs="+", choices=modes, default=list(modes))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--train_end", default="2021-12-31")
    p.add_argument("--val_end", default="2023-12-31")
    p.add_argument("--test_end", default="2026-08-21")
    p.add_argument("--output_dir", default=output_dir)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto uses GPU when available; cuda requires a working GPU")
    for name, default in [("epochs", 100), ("patience", 15), ("sequence_length", 30), ("batch_size", 32), ("threads", 4)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_norm", type=float, default=1.0)
    args = p.parse_args()
    if min(args.epochs, args.patience, args.sequence_length, args.batch_size, args.threads) < 1:
        p.error("Integer training settings must be positive")
    if args.lr <= 0 or args.clip_norm <= 0 or args.weight_decay < 0:
        p.error("Require positive lr/clip_norm and nonnegative weight_decay")
    return args


if __name__ == "__main__":
    run(parse_args())
