import unittest
import torch
from train_conditioned_fusion import ConditionedFusionLSTM
from train_interaction_fusion import InteractionFusionLSTM, interaction_context


class InteractionTests(unittest.TestCase):
    def test_identity_and_uniform_cases(self):
        torch.manual_seed(8)
        h = torch.randn(3, 30, 64, dtype=torch.float64)
        a = torch.randn_like(h).softmax(-1)
        b = torch.randn(3, 30, 1, dtype=h.dtype).softmax(1)
        expected = (64*b*a*h).sum(1) - (64*a*h).mean(1) - (b*h).sum(1) + h.mean(1)
        torch.testing.assert_close(interaction_context(h, a, b), expected)
        for aa, bb in ((torch.full_like(a, 1/64), b), (a, torch.full_like(b, 1/30))):
            torch.testing.assert_close(interaction_context(h, aa, bb), torch.zeros_like(expected))

    def test_initial_baseline_and_parameter_count(self):
        torch.manual_seed(7)
        old = ConditionedFusionLSTM(18, "scaled_average").eval()
        torch.manual_seed(7)
        new = InteractionFusionLSTM(18).eval()
        x, c = torch.randn(4, 30, 18), torch.randn(4, 4)
        torch.testing.assert_close(old(x, c)[0], new(x, c)[0])
        self.assertEqual(sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()), 1)

    def test_coefficient_can_learn(self):
        torch.manual_seed(8)
        model = InteractionFusionLSTM(18).eval()
        # The base branches first learn nonuniform weights; then lambda can learn.
        torch.nn.init.normal_(model.feature_attention.score.weight, std=0.5)
        torch.nn.init.normal_(model.temporal_attention.V.weight, std=0.5)
        pred, _ = model(torch.randn(4, 30, 18), torch.randn(4, 4))
        pred.square().sum().backward()
        self.assertTrue(torch.isfinite(model.raw_lambda.grad).all())
        self.assertGreater(model.raw_lambda.grad.abs().item(), 0)

    def test_controls_match_existing_models(self):
        for mode in ("scaled_average", "scaled_global", "scaled_seasonal"):
            torch.manual_seed(2)
            old = ConditionedFusionLSTM(18, mode).eval()
            torch.manual_seed(2)
            new = InteractionFusionLSTM(18, mode).eval()
            x, c = torch.randn(3, 30, 18), torch.randn(3, 4)
            torch.testing.assert_close(old(x, c)[0], new(x, c)[0])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
