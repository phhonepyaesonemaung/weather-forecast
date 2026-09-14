"""Paired moving-block bootstrap of date-level, seed-averaged MAE differences."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_difference(folder, reference, candidate):
    def load(mode):
        runs = {}
        for path in Path(folder).glob(f"{mode}_seed*.csv"):
            seed = int(path.stem.rsplit("seed", 1)[1])
            frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ValueError("Predictions must have unique, sorted dates")
            if not np.isfinite(frame[["actual_c", "prediction_c"]].to_numpy()).all():
                raise ValueError("Predictions contain non-finite values")
            runs[seed] = frame
        return runs
    a, b = load(reference), load(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("Both modes must have the same nonempty set of seeds")
    dates = a[min(a)].index
    if len(dates) < 2 or not (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Bootstrap requires consecutive daily predictions")
    differences = []
    truth = a[min(a)].actual_c.to_numpy()
    for seed in sorted(a):
        for frame in (a[seed], b[seed]):
            if not frame.index.equals(dates) or not np.allclose(frame.actual_c, truth, rtol=0, atol=1e-5):
                raise ValueError("Dates and actual temperatures must match across all runs")
        differences.append(np.abs(b[seed].prediction_c.to_numpy() - truth)
                           - np.abs(a[seed].prediction_c.to_numpy() - truth))
    return np.mean(differences, axis=0), sorted(a)


def bootstrap(difference, block_length=30, samples=5000, seed=42):
    difference = np.asarray(difference)
    n = len(difference)
    if not 1 <= block_length <= n or samples < 1:
        raise ValueError("Require 1 <= block_length <= number of dates and samples >= 1")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for i in range(samples):
        starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
        indices = (starts[:, None] + np.arange(block_length)).ravel()[:n]
        estimates[i] = difference[indices].mean()
    return {"candidate_minus_reference_mae_c": float(difference.mean()),
            "ci95_c": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "dates": n, "block_length": block_length, "bootstrap_samples": samples}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder")
    p.add_argument("--reference", default="scaled_average")
    p.add_argument("--candidate", default="scaled_weather")
    p.add_argument("--block_length", type=int, default=30)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    difference, seeds = paired_difference(args.folder, args.reference, args.candidate)
    result = bootstrap(difference, args.block_length, args.samples, args.seed)
    result.update(reference=args.reference, candidate=args.candidate, seeds=seeds)
    print(json.dumps(result, indent=2))
