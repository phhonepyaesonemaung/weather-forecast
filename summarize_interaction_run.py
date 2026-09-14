"""Validate and summarize a completed interaction experiment without refitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from compare_conditioned_fusion import bootstrap, paired_difference
from train_interaction_fusion import MODES


def summarize(folder):
    folder = Path(folder)
    rows = []
    for mode in MODES:
        for seed in (1, 2, 3, 4, 5):
            stem = folder / f"{mode}_seed{seed}"
            result = json.loads(stem.with_suffix(".json").read_text())
            frame = pd.read_csv(stem.with_suffix(".csv"))
            if not stem.with_suffix(".pt").is_file():
                raise ValueError(f"Missing checkpoint: {stem}")
            mae = np.abs(frame.prediction_c - frame.actual_c).mean()
            if not np.isclose(mae, result["metrics"]["mae"]) or len(frame) != result["metrics"]["n"]:
                raise ValueError(f"Metrics mismatch: {stem}")
            rows.append(result)
    table = []
    for mode in MODES:
        runs = [r for r in rows if r["mode"] == mode]
        table.append({"mode": mode, "mean_mae": float(np.mean([r["metrics"]["mae"] for r in runs])),
                      "sd_mae": float(np.std([r["metrics"]["mae"] for r in runs], ddof=1)),
                      "mean_rmse": float(np.mean([r["metrics"]["rmse"] for r in runs]))})
    comparisons = []
    for reference in ("scaled_average", "scaled_global", "scaled_seasonal", "joint"):
        difference, seeds = paired_difference(folder, reference, "interaction")
        for block in (7, 30, 90):
            comparisons.append({"reference": reference, "candidate": "interaction", "seeds": seeds,
                                **bootstrap(difference, block_length=block)})
    report = {"metrics": table, "comparisons": comparisons,
              "interaction_validation_diagnostics": [
                  {"seed": r["seed"], **r["diagnostics"]} for r in rows if r["mode"] == "interaction"]}
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Interaction fusion full-run results", "",
             "All 25 prediction reports and checkpoint paths were checked; MAEs were recomputed.",
             "Metrics below are means of five seed metrics, not ensemble forecast errors.", "",
             "| Mode | Mean MAE C | Seed SD MAE C | Mean RMSE C |", "|---|---:|---:|---:|"]
    for r in sorted(table, key=lambda r: r["mean_mae"]):
        lines.append(f"| {r['mode']} | {r['mean_mae']:.6f} | {r['sd_mae']:.6f} | {r['mean_rmse']:.6f} |")
    lines += ["", "Paired bootstrap, 30-day blocks, 5,000 samples. Negative favors interaction.", "",
              "| Reference | Interaction minus reference MAE C | 95% interval C |", "|---|---:|---|"]
    for r in comparisons:
        if r["block_length"] == 30:
            low, high = r["ci95_c"]
            lines.append(f"| {r['reference']} | {r['candidate_minus_reference_mae_c']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ["", "Validation diagnostics from each selected interaction checkpoint:", "",
              "| Seed | tanh(lambda) | Correction/base mean norm ratio |", "|---|---:|---:|"]
    for r in report["interaction_validation_diagnostics"]:
        lines.append(f"| {r['seed']} | {r['lambda_tanh']:.6f} | {r['correction_to_base_norm_ratio']:.6f} |")
    lines += ["", "See summary.json for 7/90-day block sensitivity and full diagnostics.",
              "Intervals describe date uncertainty conditional on these trained seeds. The test period",
              "has already informed model design; these are exploratory comparisons, with no multiple-comparison correction.",
              "No novelty or independent generalization claim follows from this run.", ""]
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="experiments/interaction_full_gpu")
    summarize(parser.parse_args().folder)
