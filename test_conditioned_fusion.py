import unittest

import numpy as np
import pandas as pd
import torch

from compare_conditioned_fusion import bootstrap
from train_conditioned_fusion import ConditionedFusionLSTM, MODES, prepare
from train_parallel_fusion_attention import FeatureAttention, TemporalAttention, ParallelFusionDualAttentionLSTM


class FusionTests(unittest.TestCase):
    def test_uniform_scale(self):
        h = torch.randn(3, 30, 64)
        vf, _ = FeatureAttention(64)(h)
        vt, _ = TemporalAttention(64)(h)
        torch.testing.assert_close(vf * 64, vt)

    def test_original_average_equivalence(self):
        torch.manual_seed(7)
        original = ParallelFusionDualAttentionLSTM(18, fusion="average").eval()
        torch.manual_seed(7)
        new = ConditionedFusionLSTM(18, "average").eval()
        x = torch.randn(3, 30, 18)
        torch.testing.assert_close(original(x)[0], new(x, torch.zeros(3, 4))[0])

    def test_gates_and_gradients(self):
        for mode in MODES:
            model = ConditionedFusionLSTM(18, mode)
            pred, gate = model(torch.randn(3, 30, 18), torch.randn(3, 4))
            self.assertEqual(tuple(pred.shape), (3, 1))
            torch.testing.assert_close(gate, torch.full_like(gate, 0 if mode == "temporal_only" else 0.5))
            pred.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if model.gate_layer is not None:
                self.assertEqual(sum(p.numel() for p in model.gate_layer.parameters()),
                                 3 if mode == "scaled_seasonal" else 5)

    def test_context_alignment_and_training_only_scaling(self):
        dates = pd.date_range("2020-01-01", periods=24)
        n = np.arange(24, dtype=float)
        daily = pd.DataFrame({"temperature_c": n, "humidity": n ** 2,
            "humidity_lag_1": (n - 1) ** 2, "surface_pressure_hpa": n * 2,
            "pressure_lag_1": n * 2 - 2, "day_sin": np.sin(n), "day_cos": np.cos(n)}, index=dates)
        args = (["temperature_c"], 2, dates[7], dates[15], dates[23])
        data, target_dates, naive, _, scalers = prepare(daily, *args)
        self.assertEqual(target_dates[0][0], dates[2])
        self.assertEqual(naive[0][0], 1)
        np.testing.assert_allclose(data[0].tensors[1][0, :2], [np.sin(1), np.cos(1)], atol=1e-7)
        changed = daily.copy()
        changed.loc[dates[8]:, "humidity"] += 100000
        _, _, _, _, after = prepare(changed, *args)
        self.assertEqual(scalers, after)
        self.assertAlmostEqual(scalers["gate_mean"][0], 6.0)

    def test_bootstrap_constant_difference(self):
        result = bootstrap(np.full(100, -0.2), block_length=10, samples=100)
        np.testing.assert_allclose(result["ci95_c"], [-0.2, -0.2])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
