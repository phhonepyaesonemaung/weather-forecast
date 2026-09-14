"""One-parameter interaction correction and direct joint-pooling control.

python train_interaction_fusion.py --device cuda --seeds 1 2 3 4 5
Uses exactly the data preparation and training loop of train_conditioned_fusion.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_conditioned_fusion import ConditionedFusionLSTM, parse_args, run

MODES = ("interaction", "joint", "scaled_average", "scaled_global", "scaled_seasonal")


def interaction_context(h, a, b):
    """I = J - F - T + mean(h), computed without cancellation of large sums."""
    return ((b - 1 / h.shape[1]) * (h.shape[2] * a - 1) * h).sum(dim=1)


class InteractionFusionLSTM(ConditionedFusionLSTM):
    def __init__(self, n_features, mode="interaction", lstm_units=64):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        super().__init__(n_features, "scaled_average" if mode in ("interaction", "joint") else mode,
                         lstm_units=lstm_units)
        self.mode = mode
        if mode == "interaction":
            self.raw_lambda = nn.Parameter(torch.zeros(1))

    def contexts(self, x):
        h, _ = self.lstm(x)
        h = self.dropout1(h)
        vt, b = self.temporal_attention(h)
        vf, a = self.feature_attention(h)
        base = (vf * h.shape[2] + vt) / 2
        interaction = interaction_context(h, a, b)
        joint = (h.shape[2] * b * a * h).sum(dim=1)
        return base, interaction, joint, a, b

    def forward(self, x, context):
        if self.mode not in ("interaction", "joint"):
            return super().forward(x, context)
        base, interaction, joint, _, _ = self.contexts(x)
        coefficient = torch.zeros((len(x), 1), dtype=x.dtype, device=x.device)
        if self.mode == "interaction":
            coefficient = self.raw_lambda.tanh().expand_as(coefficient)
            fused = base + coefficient * interaction
        else:
            fused = joint
        prediction = self.output_layer(self.dropout2(torch.relu(self.dense1(fused))))
        return prediction, coefficient


def validation_diagnostics(model, dataset, batch_size, device):
    """Diagnostics of best checkpoint on validation only, without tuning."""
    if model.mode not in ("interaction", "joint"):
        return {"auxiliary_value": "feature mixing gate"}
    model.eval()
    totals = torch.zeros(4, device=device)
    coefficient = float(model.raw_lambda.detach().tanh().cpu()) if model.mode == "interaction" else None
    with torch.no_grad():
        for x, _, _ in DataLoader(dataset, batch_size=batch_size):
            base, interaction, _, a, b = model.contexts(x.to(device))
            totals += torch.stack([
                base.norm(dim=1).sum(), interaction.norm(dim=1).sum(),
                (a * a.shape[-1] - 1).abs().mean(dim=(1, 2)).sum(),
                (b * b.shape[1] - 1).abs().mean(dim=(1, 2)).sum()])
    base_norm, interaction_norm, feature_deviation, temporal_deviation = (totals / len(dataset)).cpu().tolist()
    ratio = interaction_norm / max(base_norm, 1e-12)
    return {"split": "validation", "lambda_tanh": coefficient,
            "base_mean_norm": base_norm, "interaction_mean_norm": interaction_norm,
            "interaction_to_base_norm_ratio": ratio,
            "correction_to_base_norm_ratio": abs(coefficient) * ratio if coefficient is not None else None,
            "feature_relative_uniform_deviation": feature_deviation,
            "temporal_relative_uniform_deviation": temporal_deviation,
            "auxiliary_value": "tanh(lambda); zero placeholder for direct joint pooling"}


if __name__ == "__main__":
    args = parse_args(modes=MODES, output_dir="experiments/interaction_full_gpu")
    run(args, model_factory=InteractionFusionLSTM, diagnostics=validation_diagnostics,
        auxiliary_column="fusion_coefficient")
