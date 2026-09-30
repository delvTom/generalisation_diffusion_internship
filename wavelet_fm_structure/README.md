# Wavelet FM Structure

The initial MNIST-32 study measures information accessible to exact empirical
conditional velocity estimators under `self`, `orientation`, `local`, `scale`,
and `all` wavelet masks. A separate radius study then evaluates spatial mask
size and uses the selected radii for generation. The later CIFAR-10 study
calibrates radii against a pretrained FM U-Net.

The first version uses globally normalized MNIST training images, the standard
Flow Matching path `Y_t = (1-t) xi + t Y_0`, a critical orthonormal db2 DWT at
level 3 with periodization, and float64 reference calculations. The official
MNIST test split is not used.

The later calibrated-radius experiment extends those masks to every wavelet
coefficient and uses the resulting vector-valued velocity for generation.

The subband study aggregates targets within each subband: all 16 positions in
the 4x4 bands, all 64 positions in the 8x8 bands, and a spatially stratified
4x4 grid in each 16x16 band. It is run with:

```bash
python3 -m wavelet_fm_structure.run_subband_information
```

By default, the script writes plots, metrics, and its configuration to
`wavelet_fm_structure/results/experiment_a_mnist32_subband_aggregated`.
These generated files are not included in this source snapshot; the curated
SNR figure is available in the top-level `figures/` directory.

The empirical effective SNR of every complete wavelet subband can be plotted
with:

```bash
python3 -m wavelet_fm_structure.plot_subband_snr
```

For the standard FM path, the plotted quantity is
`SNR_b(t) = t^2 lambda_b / (1-t)^2`, where `lambda_b` is the mean empirical
coordinate variance in subband `b`.

## Spatial-radius study

The separate radius experiment keeps the ten central targets and evaluates
four nested mask families over every valid non-aliasing Chebyshev radius. Its
selection metric is the clean-coefficient MSE divided by the empirical
variance of that exact coefficient. The full-mask adaptive radius is the
smallest radius whose risk is within 1 percent of maximal-radius risk.

Run the complete controlled float64 protocol with:

```bash
python3 -m wavelet_fm_structure.run_radius_sweep
```

The default `--device auto` uses CUDA when it is available and otherwise falls
back to CPU. Use `--device cuda` to require a GPU and fail immediately if CUDA
is unavailable.

Results are written to
`wavelet_fm_structure/results/radius_sweep_mnist32`. This experiment is
substantially more expensive than Experiment A because fine-band radii contain
up to 739 visible coefficients. Reduced runs can use
`--max-reference-images`, `--num-evaluation-queries`, and `--times` without
changing the full defaults.

## Generation with calibrated radii

The follow-up experiment turns those diagnostic masks into complete
vector-valued empirical FM estimators. It compares all four radius families
(`same_band`, `orientation_augmented`, `scale_augmented`, and `full`) under two
radius strategies:

- a fixed radius `r=1` in every subband (clamped only when the subband is too
  small);
- the adaptive radius `r*(t)` selected by the `full` family during the radius
  sweep, shared by all four families for a controlled comparison.

Between the seven calibrated times, `r*(t)` is piecewise constant and uses the
nearest calibrated time (the earlier time wins ties). All eight variants start
from the same Gaussian noise and use 30 explicit left-Euler steps. Sampling is
performed in normalized wavelet space without clipping; clipping is used only
to display reconstructed images.

Run generation and the comparison to the unconstrained empirical conditional
FM optimum with:

```bash
python3 -m wavelet_fm_structure.run_radius_mask_generation --device cuda
```

The default implementation uses float32 for practical GPU throughput. Add
`--dtype float64` for a slower numerical audit. The two phases can also be run
separately with `--skip-mse` or `--skip-generation`.

When run, results are written to
`wavelet_fm_structure/results/radius_mask_generation_mnist32`:

- `generation_fixed_r1.png` and `generation_adaptive_rstar.png`: final samples
  for every family;
- `generation_trajectories.png`: one shared-noise trajectory for all eight
  family/radius variants;
- `generated_samples.npz` and `generation_trajectories.npz`: unclipped
  normalized numerical states;
- `mse_vs_empirical_optimum.png`: per-coordinate velocity MSE versus time;
- `mse_vs_empirical_optimum.csv` and `.json`: MSE values, equivalent clean
  error, radii, mask sizes, and Parseval diagnostics;
- `config.json` and `timings.csv`/`.json`: the complete protocol and timings.

For a larger visual audit of the best-performing `full + r*(t)` variant, run:

```bash
python3 -m wavelet_fm_structure.generate_full_adaptive_gallery --device cuda
```

It adds a 20-sample gallery and ten trajectories to the same result directory.
Every final sample is paired with its exact nearest MNIST training image under
pixel L2 distance. The search uses unclipped normalized images and is computed
equivalently in the orthonormal wavelet basis. Neighbor indices, labels, and
pixel MSE values are saved alongside the figures in CSV, JSON, and NPZ form.

## CIFAR-10 calibration against the pretrained FM U-Net

The RGB extension applies the same db2 level-3 DWT independently to `R`, `G`,
and `B`. A spatial site selected by a mask exposes all three channel
coefficients, while one radius is shared across channels. The `Full` family is
calibrated on CIFAR-10 test queries that are disjoint from the training-image
reference bank; the resulting radius schedule is then reused unchanged by all
four families.

Run the complete CUDA experiment with:

```bash
python3 -m wavelet_fm_structure.run_cifar10_fm_calibrated_radius
```

It writes the selected-radius heatmap, the four MSE curves against the
pretrained FM U-Net, and the shared-noise generation/nearest-neighbor grid. It
also compares final `Full` samples against the pretrained FM U-Net and the
affine Wiener velocity estimator, then plots matched trajectories for those
three methods. All three start from exactly the same Gaussian noise and use the
same explicit left-Euler grid. The outputs and supporting numerical values are
written to `wavelet_fm_structure/results/cifar10_fm_calibrated_radius`:

- `selected_radius_heatmap.png`;
- `mse_vs_fm_unet.png`;
- `generations_and_full_neighbors.png`;
- `full_unet_wiener_comparison.png`;
- `full_unet_wiener_trajectories.png`;
- `metrics.json`.

Runs started before the affine-Wiener MSE curve was added can be augmented
without repeating radius calibration or empirical generation. Once the main
run has finished and written `metrics.json`, use:

```bash
python3 -m wavelet_fm_structure.add_wiener_mse_to_cifar10_results
```

This replays the recorded RNG sequence to recover exactly the same test
queries and noises, evaluates only the U-Net and cached affine Wiener model,
then updates `mse_vs_fm_unet.png` and `metrics.json` in place.
