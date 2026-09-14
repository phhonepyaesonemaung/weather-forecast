# Running interaction fusion

The new model lives in train_interaction_fusion.py. It reuses the existing
conditioned runner through optional model/diagnostics arguments, so all modes
share preprocessing, training, early stopping and prediction export. Existing
commands retain their defaults.

```powershell
python -m unittest test_conditioned_fusion test_interaction_fusion
python train_interaction_fusion.py --device cuda --seeds 1 2 3 4 5 --epochs 100 --output_dir experiments/interaction_full_gpu
```

The default runs 25 models: interaction, joint, scaled_average, scaled_global,
scaled_seasonal for each of five seeds. The interaction mode implements exactly
the proposed formula, with one zero-initialized learned scalar lambda:

    I = sum_i (b_i - 1/L) (D*a_i - 1) * h_i
    v = (F + T)/2 + tanh(lambda)*I

The direct joint control uses `v = D*sum_i b_i*a_i*h_i`, with no extra learned
parameter. The three existing controls are freshly trained in the same run.
Tests verify that their initial predictions match their original classes.

Each JSON report additionally records validation-set interaction/base vector
norms, departures of attention from uniform, and tanh(lambda), computed from
the best-validation checkpoint. No tuning uses the test predictions. The CSV
column fusion_coefficient means tanh(lambda) for interaction, a zero placeholder
for joint, and the feature mixing gate for the three existing controls.

```powershell
python compare_conditioned_fusion.py experiments/interaction_full_gpu --reference scaled_average --candidate interaction
python compare_conditioned_fusion.py experiments/interaction_full_gpu --reference joint --candidate interaction
```

The current execution writes progress to interaction_full_gpu.log. In PowerShell:

```powershell
Get-Content interaction_full_gpu.log -Tail 10 -Wait
```

Use a fresh output folder for any later experiment. Existing completed run files
are protected from overwriting. This command does not resume a partial epoch.
The proposal's originality and performance remain unproven until evaluated;
the already examined test period is exploratory evidence.
