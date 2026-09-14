# Proposed attention interaction correction

Status: implemented in train_interaction_fusion.py; see INTERACTION_RUN.md for
execution instructions. Originality and accuracy gains are not established by
the proposal. Prepared after checking all 30 CUDA runs in full_run_gpu.

## Evidence from CUDA runs

Mean seed MAE (C): original average 0.627476; scaled average 0.614640;
scaled global 0.613292; scaled seasonal 0.612182; scaled weather 0.622223;
temporal only 0.620096. Verified matching dates and observed temperatures,
checkpoint existence, CUDA metadata, and MAEs recomputed from predictions.

Paired 30-day-block bootstrap intervals over 934 dates, averaging absolute
errors over five seeds before resampling (5,000 samples):

- Scaled average minus original: -0.012836 C, CI [-0.019740, -0.004579].
- Weather gate minus scaled average: +0.007582 C, CI [+0.002002, +0.012601].
- Seasonal gate minus scaled average: -0.002459 C, CI [-0.007223, +0.002126].

The weather gate is not supported by these runs. Seasonal gating has the
lowest observed MAE but no clear advantage over scaled averaging under this
bootstrap. These intervals are conditional on the trained seeds, and the test
period has already influenced model design.

## Formula

For L observed days and D hidden channels, let h_i be the hidden vector,
a_i the feature softmax vector (sum across channels = 1), and b_i the temporal
softmax scalar (sum across days = 1). All are computed from observed inputs.

Define the existing corrected branches and an unweighted reference:

    F = (D/L) sum_i a_i * h_i
    T = sum_i b_i h_i
    m = (1/L) sum_i h_i

Multiplication between vectors is elementwise. Define:

    J = D sum_i b_i a_i * h_i
    I = J - F - T + m
      = sum_i (b_i - 1/L) (D a_i - 1) * h_i

Proposed fusion and prediction:

    v = (F + T)/2 + tanh(lambda) I
    predicted_temperature = existing_prediction_head(v)

Only lambda is an additional learned scalar; initialize it to zero. The model
starts exactly as scaled_average. Keep the original prediction head, loss,
features and training settings for the first experiment.

I is a second-order contrast of pooling with both attention maps, feature
attention only, temporal attention only, and neither. It explicitly exposes
an interaction before separate pooling discards the maps. This is not a causal
effect, a statistical independence test, or an uncertainty estimate.

At uniform temporal attention I = 0, regardless of feature attention. At
uniform feature attention I = 0, regardless of temporal attention. Thus this
term does not simply add another copy of either branch's individual effect.
The identity and both zero-correction cases passed a numerical algebra check.

The hypothesis is that channel emphasis matters differently on days with high
temporal attention, and that this additional statistic may improve prediction.
The result is a signed residual correction, not a normalized probability
distribution or convex mixture. tanh bounds the coefficient, not the correction
norm. Report interaction norms relative to the base vector during training.

At uniform initialization I and its attention derivatives vanish. The base
branches can still learn normally; lambda can receive a gradient once both
maps become nonuniform. Near-uniform maps may keep I tiny. This is an explicit
failure mode to inspect, not grounds to claim the formula will improve accuracy.

## Related work and limits of the search

- DA-RNN already combines feature/input and temporal attention:
  https://arxiv.org/abs/1704.02971
- Bilinear Attention Networks already model attention interactions:
  https://arxiv.org/abs/1805.07932
- Differential Attention Fusion already exists for time-series forecasting:
  https://arxiv.org/abs/2202.11402
- Persistence initialization with residual gating already exists:
  https://arxiv.org/abs/2208.14236

These sources rule out broad novelty claims about dual attention, interaction
pooling, differential attention, or residual gating. A targeted search did not
identify this exact four-term correction for the current parallel LSTM, but
search absence does not establish novelty. The expansion itself is elementary
algebra. Any contribution would have to be the precisely specified method and
demonstrated behavior, with a closer full-text prior-art comparison.

## Controlled experiment to perform next

Compare scaled_average, scaled_global (one-parameter control), scaled_seasonal,
the proposed one-parameter correction, and direct joint pooling J. The joint
pooling control checks whether subtraction of the individual effects matters.
Train all with the same five seeds/settings on chronological development folds;
use validation only to choose any settings. Inspect lambda, attention departure
from uniform, and interaction/base norm ratios using training/validation data.

Use MAE/RMSE, monthly metrics, and paired block bootstrap with sensitivity to
7/30/90-day blocks. Do not tune lambda on saved test predictions. Retain the
already used 2024-2026 period only as exploratory confirmation. Earlier folds
improve robustness checks but are not pristine independent data after extensive
dataset analysis; reserve a genuinely untouched period or location for a strong
generalization claim. No accuracy gain should be promised in advance.
