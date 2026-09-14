# Scale-corrected, conditioned fusion experiments

This implements the proposed research hypothesis, not a claim of novelty or
improved accuracy. Existing training commands and historical results are preserved.

The feature context is multiplied by hidden dimension D (64), making it equal
to the temporal context at uniform attention. This does not enforce equal norms
after training. The weather gate is
`sigmoid(b + w1*day_sin + w2*day_cos + w3*z_delta_RH + w4*z_delta_P)`.
It has five parameters, initialized to zero (gate 0.5). Calendar features and
weather changes come from the last observed day, not the target day. Changes
use the existing daily humidity and pressure lag columns. Their mean and standard
deviation are fitted on training days only; constant changes use scale 1.

| Mode | Feature scaling | Gate |
|---|---|---|
| average | Original | Fixed 0.5 |
| scaled_average | Multiply by D | Fixed 0.5 |
| scaled_global | Multiply by D | One learned parameter |
| scaled_seasonal | Multiply by D | Season, three parameters |
| scaled_weather | Multiply by D | Season and weather changes, five parameters |
| temporal_only | No feature branch | Temporal context only |

Run from the project root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m unittest test_conditioned_fusion
.\.venv\Scripts\python.exe train_conditioned_fusion.py --seeds 1 2 3 4 5
.\.venv\Scripts\python.exe compare_conditioned_fusion.py experiments/conditioned_fusion --reference scaled_average --candidate scaled_weather
```

The default command trains all six modes for five seeds, with at most 100 epochs
and validation early stopping. This can take substantial time on CPU. To check
the workflow first, use `--epochs 1 --seeds 1 --output_dir experiments/smoke`.
One epoch is a software check, not a performance experiment.

## GPU training

The runner defaults to `--device auto`, which selects CUDA when available.
Use `--device cuda` to require the NVIDIA GPU and fail clearly if unavailable;
`--device cpu` explicitly selects CPU. The selected device is printed at startup.
For this Windows/Python 3.14 environment, install the CUDA build with:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe train_conditioned_fusion.py --device cuda --seeds 1 2 3 4 5 --epochs 100 --output_dir experiments/full_run_gpu
```

Keep GPU and CPU experiments in separate output folders. GPU results may differ
numerically even with the same seeds. Compare modes within the same environment.

Use distinct output folders for chronological evaluations, for example:

```powershell
.\.venv\Scripts\python.exe train_conditioned_fusion.py --train_end 2017-12-31 --val_end 2019-12-31 --test_end 2021-12-31 --output_dir experiments/period_2020_2021
.\.venv\Scripts\python.exe train_conditioned_fusion.py --train_end 2019-12-31 --val_end 2021-12-31 --test_end 2023-12-31 --output_dir experiments/period_2022_2023
```

Default boundaries reproduce training through 2021, validation 2022–2023,
and testing 2024–21 August 2026. As in the original runner, each split constructs
its own windows, excluding its first 30 days from scoring. Missing daily dates
are rejected. All modes receive identical settings and sample dates. Validation
MSE is weighted by sample count. Test inference runs in batches.

Each run saves a dated prediction CSV (including persistence and feature gate),
a JSON report (overall and year-month MAE/RMSE, configuration, normalization,
and best epoch), and a checkpoint containing weights and metadata. Existing run
files are refused to avoid overwriting results. Historical results.jsonl is not
modified. For inference, instantiate ConditionedFusionLSTM with the saved mode,
load state_dict, and apply the saved feature/target affine transforms
`scaled = raw * scale + min`; weather deltas use `(raw - gate_mean)/gate_scale`.

The comparison tool requires matching seeds, dates, and observations. It averages
paired absolute-error differences across seeds on each date, then resamples
contiguous date blocks to estimate a percentile 95% interval. Negative values
favor the candidate. This measures date uncertainty conditional on the trained
seeds, not independent seed replicates or ensemble prediction accuracy. Try
block lengths such as 7, 30, and 90 days and report sensitivity. Intervals are
exploratory, especially with seasonality, multiple comparisons, and periods
already used for model selection; independent periods/locations are still needed.
