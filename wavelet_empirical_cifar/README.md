# Three separate CIFAR-10 wavelet experiments

All runs use CIFAR-10 in `[-1,1]`, `db2` level 3 with periodization, and
`X_t=(1-t) epsilon+t X_1`. The training split is the reference bank; seeded,
disjoint CIFAR-10 test subsets are used for calibration and final evaluation.
The empirical estimator uses the repository's exact streaming softmax. Each
selected wavelet site exposes all three RGB channels. A separate CelebA-HQ
script is included, but its dataset and checkpoint are not redistributed.
The default 8 calibration / 16 evaluation images are for screening, not for
a final quantitative claim. A selected variant can be confirmed with
`--calibration-queries 32 --evaluation-queries 128` and the full bank.

Run from the repository root in the existing environment (`python3` below may
be replaced by `.venv/bin/python`):

```bash
python3 -m wavelet_empirical_cifar.run_wiener_masks --device cuda
python3 -m wavelet_empirical_cifar.run_spatial_sharing --device cuda --benchmark-only
python3 -m wavelet_empirical_cifar.run_spatial_sharing --device cuda --target-stride 16 --no-generation
python3 -m wavelet_empirical_cifar.run_coarse_context --device cuda
```

The CIFAR-10 Artem-style threshold is `--artem-threshold 0.05` on
`run_wiener_masks`. It is a separate, variable-budget comparison: scalar
Wiener rows are divided by their diagonal and thresholded against the global
maximum before scalar coordinates are combined into complete RGB sites.
`artem_mask_sizes.csv` reports how many sites remain at each time. Do not
attribute an improvement over a 9-site mask solely to the location of the
sites when its budget differs. The threshold run uses its own result directory
by default and does not overwrite the equal-budget results.

To add the trained U-Net to an **existing** pair of saved generations without
recomputing the 50,000-image candidate bank:

```bash
python3 -m wavelet_empirical_cifar.compare_saved_generations --device cuda
```

This uses the recorded seed and Euler step count and writes
`generation_comparison_unet.png` in the existing result directory.

For the completed `tau=0.05` run, the following reporting command reuses the
saved empirical generations and FM U-Net samples. It computes the two missing
affine Wiener controls from the same training-bank moments and initial noises,
then writes a 2D wavelet-mask mosaic and a five-method grid with no nearest
neighbors:

```bash
python3 -m wavelet_empirical_cifar.report_wiener_threshold --device cuda
```

Outputs in `results/wiener_masks_tau_0p05/` are
`artem_wavelet_mask_mosaic.png` and `generation_comparison_five.png`.
The mosaic's blue cells are *wavelet coefficients*, not input-image pixels;
each selected site exposes all RGB channels. The two empirical columns use
the posterior over training images, whereas the global and restricted Wiener
columns are affine covariance-based regressors. The comparison uses the saved
seed and 30 Euler steps. `--maps-only` creates only the mask figure.

To regenerate a four-column CIFAR-10 grid from saved samples without the
restricted affine Wiener control or a new GPU run, use
`python3 -m wavelet_empirical_cifar.report_wiener_threshold --four-only`.
It writes `generation_comparison_four.png` alongside the earlier grid.

The matching CelebA-HQ gallery uses the full 30,000-image bank at 64x64,
eight shared initial noises, and four columns (Artem-threshold empirical,
sensitivity-9 empirical, global affine Wiener, FM U-Net). There is no
restricted Wiener column or nearest-neighbor column:

```bash
python3 -u -m wavelet_empirical_cifar.run_celeba_wiener_comparison --device cuda
```

The result is `results/celeba_hq_tau_0p05/generation_comparison_four.png`.
Masks and every Euler state are cached in that directory, so rerunning the
same command resumes after interruption. `--validate-only` checks the dataset
and pretrained assets without starting generation. This is a substantial
GPU run; the empirical columns evaluate all output sites against the complete
reference bank at every Euler step.
Before the potentially expensive threshold posterior, inspect variable mask
sizes with `python3 -m wavelet_empirical_cifar.run_wiener_masks --device cuda
--artem-threshold 0.05 --mask-size-only` (on one shell line).

For the full spatial-sharing comparison, set `--target-stride 1`; omit
`--no-generation` only after checking the benchmark. Full sharing has
`N_train * H_b * W_b` candidates **per target and subband** and is substantially
more expensive than either equal-size bank. A screening cap, if needed, must
be passed explicitly as `--max-reference-images N`; it is logged and must not
be compared as though it were the 50,000-image bank. Similarly, a stride above
one estimates MSE only on the selected sites, not on the entire image.

Each script writes a JSON protocol file, CSV metrics, and a compact MSE plot
to its own result directory. Generated samples, pixel-L2 nearest training
neighbors, and an image grid are written only when generation is enabled.
The stage-one sensitivity masks are determined by the affine **denoiser**
Jacobian; the restricted Wiener control is re-fitted from each submatrix.
During Euler generation, stage-one masks use the nearest of the five measured
times; stage-three contexts are fixed over time. Stage three keeps the
existing `Full` orientation/scale mask by default, then compares four amounts
of extra `cA3` context for fine-detail targets only (`--base-family` can fix
a different existing family in a separate run). The shared-patch experiment
keeps the same-band radius fixed and uses one reproducibly chosen source
position per training image for the equal-size control. The source posterior
mass is aggregated over *all* positions from the same training image before
its concentration is reported.

`--no-unet` allows running without the compatible CIFAR-10 checkpoint. Without
that flag, the script refuses to silently drop the U-Net agreement baseline.
