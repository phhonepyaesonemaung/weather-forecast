"""
Seasonal Quantile Normalization with Heavy-Tail Handling and Physical
Bounds Enforcement.

WHAT THIS IS (and, importantly, what it is not):

This module combines three preprocessing techniques for multivariate
weather time series:

  1. Per-calendar-month quantile normalization: (x - monthly_median) /
     (monthly_p95 - monthly_p5). This is the standard technique known in
     the climate science literature as "quantile mapping" - a decades-old,
     widely used bias-correction method (see e.g. Cannon et al. 2015,
     "Bias correction of GCM precipitation by quantile mapping"; it is
     explicitly described as a "traditional"/"conventional" method in
     recent deep-learning weather forecasting papers, not a novel one).

  2. A log1p transform applied to heavy-tailed variables (detected via
     skewness or a p99/p95 ratio) before quantile normalization. This is
     textbook practice in statistics and hydrology for handling skewed
     variables like precipitation.

  3. Clamping predictions/inputs to known physical ranges (e.g. humidity
     in [0, 100]) on inverse-transform. This is standard engineering
     hygiene, not a research contribution by itself.

NONE OF THESE THREE COMPONENTS ARE NOVEL, individually or in this
combination - each is a well-established, previously published technique.
If this is described in a paper, it should be presented as "we apply
standard seasonal quantile normalization (quantile mapping), a log
transform for heavy-tailed variables, and physical bounds enforcement" -
not as a new method.

A further terminology note: "physics-informed" is a specific, loaded term
in the ML literature, referring to embedding actual physical equations or
PDE constraints directly into a model's loss function or architecture
(Raissi et al., 2019, "Physics-informed neural networks"). What this
module does - domain-knowledge-based clipping and a log transform - does
not do that, and calling it "physics-informed" in a paper risks a
reviewer objecting on terminology grounds alone, independent of whether
the underlying technique is useful (it is).

This is a genuinely useful preprocessing utility. It is not a proposed
research contribution.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew as scipy_skew

EPS = 1e-8


class PhysicsInformedAdaptiveNormalizer:
    """Seasonal (per-month) quantile normalizer with optional log-transform
    for heavy-tailed variables and physical-bounds enforcement on inverse
    transform.

    (Class name kept as specified; see the module docstring above for why
    the "physics-informed" and "adaptive" labels are more marketing than
    technical description - the mechanism itself is quantile mapping plus
    a log transform plus clipping, all pre-existing techniques.)

    Parameters
    ----------
    variable_names : list of str
        Names of the variables, in the column order they appear in the
        data arrays passed to fit/transform.
    physical_bounds : dict mapping variable name -> (min, max)
        Either bound may be None to mean "no constraint on that side".
        E.g. {"humidity": (0, 100), "rainfall": (0, None)}.
    heavy_tail_threshold : float, default 2.0
        A variable is treated as heavy-tailed (and log1p-transformed
        before quantile normalization) if its skewness exceeds this
        threshold, OR if its 99th/95th percentile ratio exceeds 3.0.

    Example
    -------
    >>> bounds = {"temp": (-50, 60), "humidity": (0, 100), "rainfall": (0, None)}
    >>> scaler = PhysicsInformedAdaptiveNormalizer(list(bounds.keys()), bounds)
    >>> scaler.fit(train_data, train_timestamps)
    >>> normalized = scaler.transform(train_data, train_timestamps)
    >>> recovered = scaler.inverse_transform(normalized, train_timestamps)
    """

    def __init__(
        self,
        variable_names: List[str],
        physical_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        heavy_tail_threshold: float = 2.0,
    ):
        self.variable_names = list(variable_names)
        self.physical_bounds = physical_bounds or {}
        self.heavy_tail_threshold = heavy_tail_threshold

        self._is_fit = False
        self._heavy_tailed: Dict[str, bool] = {}
        # Per-variable arrays indexed [month-1] (0..11), shape (12,)
        self._median: Dict[str, np.ndarray] = {}
        self._p5: Dict[str, np.ndarray] = {}
        self._p95: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _detect_heavy_tailed(self, column: np.ndarray) -> bool:
        s = scipy_skew(column)
        p95, p99 = np.percentile(column, [95, 99])
        # Guard against p95 <= 0 (ratio undefined / meaningless there)
        ratio = (p99 / p95) if p95 > 0 else 0.0
        return bool(s > self.heavy_tail_threshold or ratio > 3.0)

    def fit(self, train_data: np.ndarray, timestamps: pd.Series) -> "PhysicsInformedAdaptiveNormalizer":
        """Compute per-month median/p5/p95 for each variable (after an
        optional log1p transform for variables detected as heavy-tailed).

        train_data : np.ndarray, shape (n_samples, n_variables)
        timestamps : pd.Series of datetime-like values, length n_samples,
            aligned row-for-row with train_data.
        """
        train_data = np.asarray(train_data, dtype=np.float64)
        months = pd.DatetimeIndex(timestamps).month.to_numpy()  # 1..12

        for i, var in enumerate(self.variable_names):
            col = train_data[:, i]

            heavy = self._detect_heavy_tailed(col)
            self._heavy_tailed[var] = heavy
            work_col = np.log1p(np.clip(col, a_min=0, a_max=None)) if heavy else col

            median = np.zeros(12)
            p5 = np.zeros(12)
            p95 = np.zeros(12)
            for m in range(1, 13):
                vals = work_col[months == m]
                if len(vals) == 0:
                    # No training examples for this month (e.g. very short
                    # training window) - fall back to the global stats so
                    # transform() still has something sane to use.
                    vals = work_col
                median[m - 1] = np.median(vals)
                p5[m - 1] = np.percentile(vals, 5)
                p95[m - 1] = np.percentile(vals, 95)

            self._median[var] = median
            self._p5[var] = p5
            self._p95[var] = p95

        self._is_fit = True
        return self

    def _require_fit(self):
        if not self._is_fit:
            raise RuntimeError("Call fit() before transform()/inverse_transform().")

    def transform(self, data: np.ndarray, timestamps: pd.Series) -> np.ndarray:
        """Normalize data using the statistics computed in fit()."""
        self._require_fit()
        data = np.asarray(data, dtype=np.float64)
        months = pd.DatetimeIndex(timestamps).month.to_numpy()
        month_idx = months - 1  # 0..11, for indexing the per-month arrays

        out = np.empty_like(data)
        for i, var in enumerate(self.variable_names):
            col = data[:, i]
            if self._heavy_tailed[var]:
                col = np.log1p(np.clip(col, a_min=0, a_max=None))

            median = self._median[var][month_idx]
            spread = self._p95[var][month_idx] - self._p5[var][month_idx]
            out[:, i] = (col - median) / (spread + EPS)

        return out

    def inverse_transform(self, normalized_data: np.ndarray, timestamps: pd.Series) -> np.ndarray:
        """Reverse transform() and enforce physical bounds."""
        self._require_fit()
        normalized_data = np.asarray(normalized_data, dtype=np.float64)
        months = pd.DatetimeIndex(timestamps).month.to_numpy()
        month_idx = months - 1

        out = np.empty_like(normalized_data)
        for i, var in enumerate(self.variable_names):
            median = self._median[var][month_idx]
            spread = self._p95[var][month_idx] - self._p5[var][month_idx]
            col = normalized_data[:, i] * (spread + EPS) + median

            if self._heavy_tailed[var]:
                col = np.expm1(col)

            lo, hi = self.physical_bounds.get(var, (None, None))
            if lo is not None:
                col = np.clip(col, a_min=lo, a_max=None)
            if hi is not None:
                col = np.clip(col, a_min=None, a_max=hi)

            out[:, i] = col

        return out

    def forward(self, data: np.ndarray, timestamps: pd.Series) -> torch.Tensor:
        """PyTorch-compatible entry point: transform() then wrap as a
        float32 tensor, for use as a preprocessing step ahead of a model."""
        normalized = self.transform(data, timestamps)
        return torch.as_tensor(normalized, dtype=torch.float32)

    # ------------------------------------------------------------------
    def is_heavy_tailed(self, variable_name: str) -> bool:
        self._require_fit()
        return self._heavy_tailed[variable_name]

    def get_scaling_statistics(self) -> Dict[str, dict]:
        """Returns fitted parameters per variable, for debugging/plotting."""
        self._require_fit()
        return {
            var: {
                "heavy_tailed": self._heavy_tailed[var],
                "monthly_median": self._median[var].tolist(),
                "monthly_p5": self._p5[var].tolist(),
                "monthly_p95": self._p95[var].tolist(),
            }
            for var in self.variable_names
        }


if __name__ == "__main__":
    # ---- Synthetic data with seasonal patterns and extreme events ----
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2023-01-01", periods=8760, freq="h")
    month = timestamps.month.to_numpy()

    n = len(timestamps)
    temp = 20 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + rng.normal(0, 2, n)
    humidity = np.clip(60 + 15 * np.sin(2 * np.pi * (month - 6) / 12) + rng.normal(0, 5, n), 0, 100)
    # Heavy-tailed: mostly small, occasional large rain events
    rainfall = rng.exponential(scale=2.0, size=n)
    extreme_mask = rng.random(n) < 0.01
    rainfall[extreme_mask] += rng.uniform(50, 150, size=extreme_mask.sum())
    wind = np.abs(rng.normal(5, 2, n))
    pressure = 1013 + rng.normal(0, 5, n)

    data = np.column_stack([temp, humidity, rainfall, wind, pressure])
    var_names = ["temp", "humidity", "rainfall", "wind", "pressure"]

    physical_bounds = {
        "temp": (-50, 60),
        "humidity": (0, 100),
        "rainfall": (0, None),
        "wind": (0, None),
        "pressure": (800, 1200),
    }

    scaler = PhysicsInformedAdaptiveNormalizer(
        variable_names=var_names, physical_bounds=physical_bounds, heavy_tail_threshold=2.0,
    )
    scaler.fit(data, timestamps)

    normalized = scaler.transform(data, timestamps)
    recovered = scaler.inverse_transform(normalized, timestamps)

    print(f"Reconstruction MSE: {np.mean((data - recovered) ** 2):.6f}")
    print(f"Per-variable reconstruction MAE:")
    for i, var in enumerate(var_names):
        print(f"  {var:>10s}: {np.mean(np.abs(data[:, i] - recovered[:, i])):.6f}")

    print(f"\nHeavy-tailed variables detected: {[v for v in var_names if scaler.is_heavy_tailed(v)]}")

    # Physical bounds sanity checks
    assert recovered[:, var_names.index("humidity")].min() >= 0
    assert recovered[:, var_names.index("humidity")].max() <= 100
    assert recovered[:, var_names.index("wind")].min() >= 0
    assert recovered[:, var_names.index("rainfall")].min() >= 0
    print("\nPhysical bounds enforced correctly (assertions passed).")

    # PyTorch-compatible path
    tensor_out = scaler.forward(data, timestamps)
    print(f"\nforward() output type: {type(tensor_out)}, dtype: {tensor_out.dtype}, shape: {tuple(tensor_out.shape)}")
