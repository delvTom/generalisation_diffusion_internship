# Local wavelet Gaussian scale mixture

This autonomous experiment tests whether a proportional two-state Gaussian
scale mixture (GSM) improves local wavelet-coefficient denoising beyond matched
Gaussian/Wiener baselines.

## Fixed protocol

- official MNIST train/test splits at native `28x28` or resized `64x64`;
- pixels in `[0,1]`, without centering or variance normalization;
- `64x64` uses torchvision's bilinear PIL resize with antialiasing;
- orthogonal Haar DWT, two levels, `periodization`;
- at `28x28`: detail bands `H1,V1,D1` are `14x14`, and level-2 bands are `7x7`;
- at `64x64`: detail bands `H1,V1,D1` are `32x32`, and level-2 bands are `16x16`;
- `A2` is ignored by the coefficient-level fit and handled by a shared affine
  Wiener estimator in the complete-image experiment;
- valid `3x3` neighborhoods in one band only, without wrapping or padding;
- five uniformly sampled valid centers per image and band by default;
- fixed empirical patch mean during GSM fitting;
- `covariance_epsilon=1e-6` relative to average marginal variance;
- `scale_floor=1e-4` under the `E[S]=1` identification constraint.
- CUDA bulk computation in float32 with TF32 disabled, with GSM parameters and
  reductions retained in float64.

The denoising path is

```text
Y_t = t Y_0 + (1-t) Z,    Z ~ N(0,I),
```

with `t=0` pure noise and `t=1` clean data. This first experiment contains no
velocity field, ODE integration, generation, FID, or neural-network comparison.

Five posterior-mean estimators are evaluated using identical clean patches and
noise draws:

1. empirical local Wiener;
2. local Wiener matched to the GSM mean and covariance;
3. scalar/self Wiener matched to the GSM central marginal;
4. self GSM using only the noisy center coefficient;
5. local GSM using the complete noisy `3x3` patch.

The scalar matched Wiener is required to isolate marginal non-Gaussianity:
`GSM self` versus `Wiener matched self`. Context is measured by `GSM local`
versus `GSM self`.

## Commands

Run the smoke test first:

```bash
python3 -u wavelet_gsm_local/run_experiment.py --smoke-test
```

The default smoke-test output is temporary and includes the selected resolution.

Then run the full protocol:

```bash
python3 -u wavelet_gsm_local/run_experiment.py --device cuda
```

Use `--reuse-fits` to reuse completed per-band GSM files from the selected
output directory. Results are written under
`wavelet_gsm_local/results/mnist28_haar2/` by default.

For MNIST-64 using a separately obtained official MNIST cache:

```bash
python3 -u wavelet_gsm_local/run_experiment.py \
  --image-size 64 \
  --data-root /path/to/MNIST-cache \
  --device cuda
```

This refits all six detail-band GSMs and writes them under
`wavelet_gsm_local/results/mnist64_haar2/`.

## Complete-image denoising

After the coefficient-level experiment has produced its six fitted models, run:

```bash
python3 -u wavelet_gsm_local/run_image_denoising.py --device cuda
```

For the completed MNIST-64 coefficient run:

```bash
python3 -u wavelet_gsm_local/run_image_denoising.py \
  --source-dir wavelet_gsm_local/results/mnist64_haar2 \
  --device cuda
```

This second script never refits the detail-band GSMs. It reconstructs complete
test images with matched self/local Wiener, empirical local Wiener, and
self/local GSM posterior means. Every method shares the same full `A2`
affine-Wiener estimate; detail-band borders use their corresponding
self estimator. One pixel-space Gaussian noise draw per test image is shared by
all methods and all values of `t`.

Outputs are stored under the selected source run's `full_image_denoising/`
directory, for example
`wavelet_gsm_local/results/mnist64_haar2/full_image_denoising/`:

- `reconstructions_t*.png`: raw estimates displayed after explicit `[0,1]` clipping;
- `image_mse_vs_t.png`: unclipped complete-image test MSE;
- `interior_border_mse_vs_t.png`: detail-band risk split by applicability of context;
- `relative_image_mse_gains.png`: paired relative gains with 95% intervals;
- `covariance_heldout_diagnostic.png`: train empirical and fitted GSM covariances
  compared with a held-out test covariance;
- `conditional_mse_by_clean_magnitude.png`: MNIST-64 detail-coefficient MSE
  conditioned on `abs(Y_0,q) / sigma_b`, for matched local Wiener, self GSM,
  and local GSM;
- `gsm_local_gain_by_clean_magnitude.png`: relative local-GSM gain over matched
  local Wiener in the same magnitude bins, with the coefficient count shown;
- `magnitude_diagnostic_config.json`: train-only subband standard deviations,
  effective bin edges, counts, and aggregation conventions;
- `sanity_checks.json`: endpoint, border, DWT-linearity, and per-image Parseval audits.

## Interpretation limits

MNIST has many nearly zero background patches and is not spatially stationary.
The inferred regimes are therefore called low-scale and high-scale, never
automatically interpreted as background and contour. Positive fourth-order
excess rejects a matched Gaussian fourth-order identity but does not establish
that the two-state GSM is the correct model.
