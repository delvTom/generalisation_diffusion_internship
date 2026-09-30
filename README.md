# Analytical locality in flow matching

Research code from a master's internship on analytical velocity estimators for
image generation. The question running through the experiments is: **which
observations and which statistics does an estimator need to retain to avoid
both a purely affine Wiener approximation and a global empirical lookup?**

This repository is organized as a reproducible research record, not a claim
that the analytical samplers match a trained U-Net's image quality. Each
experiment retains its own dataset, target and evaluation protocol; plots from
different sections are not a single benchmark.

## Start here

Create an environment from this directory. Install a PyTorch wheel appropriate
for your CPU/CUDA platform first, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-fm.txt
python scripts/setup_external.py
```

The last command fetches the *separate* Lukoianov et al. repository at commit
`48d69957` into ignored `external/` and links its `src/` and `configs/` into
this directory. It does not copy that repository into this project's history.
For an offline install from a local checkout, pass `--source /path/to/checkout`.
See [DATA.md](DATA.md) for official datasets, PCA caches, model checkpoints,
and normalization conventions. Do not start the full runs until those inputs
are present.

```bash
python scripts/check_environment.py --fm
```

A small CPU check that does not need any pretrained model is:

```bash
python -m wavelet_gsm_local.run_experiment --smoke-test --device cpu --download
```

## Experiment map

| Question | Code and first entry point | Main outcome |
| --- | --- | --- |
| Does a denoiser-derived locality mask remain appropriate for an FM velocity? | `fm_comparisons/`; `python -m fm_comparisons.compare_fm_unet_closed_form_mse --help` | Velocity and denoiser sensitivities select different *relative* spatial masks; teacher-MSE and generation comparisons remain protocol-specific. |
| Can a mixed denoising/velocity target improve optimization? | `asymflow/`; `python -m asymflow.sweep_asymflow_rank_cifar10 --help` | A short, single-seed CIFAR-10 rank sweep favored a 4x4 patch projection at rank 8. This is screening evidence, not a robust final FID claim. |
| Can the probability path control spectral reconstruction? | `spectral_paths/`; `python -m spectral_paths.sequential_linear_schedule.run_affine_experiment --help` | PCA-dependent schedules change trajectories; affine controls and endpoint concentration delimit what this can explain. |
| Which wavelet coefficients should each output observe? | `wavelet_score/`, `wavelet_fm_structure/`, `wavelet_empirical_cifar/`; `python -m wavelet_fm_structure.run_subband_information --help` | Subband, orientation, scale and spatial context are separated; learned radii can improve similarity to an FM U-Net, but generation remains weaker. |
| Is a hard coarse-to-fine information schedule enough? | `wavelet_coarse_to_fine/`; `python -m wavelet_coarse_to_fine.run_cifar10_baseline --help` | No: the tested hard schedule worsened MSE and produced spatial artifacts. |
| Is a larger wavelet feature dictionary useful? | `wavelet_fm_polynomial/`; `python -m wavelet_fm_polynomial.run_mnist_mse --help` | More features reduce fit risk, but may increase held-out risk. |
| Can non-Gaussian local statistics improve Wiener? | `wavelet_gsm_local/`; `python -m wavelet_gsm_local.run_image_denoising --help` | On held-out MNIST-64, local GSM reduced unclipped reconstruction MSE relative to a matched local Wiener estimator; this is **not** a demonstrated generation gain. |

The [experiment index](EXPERIMENTS.md) specifies the exact scripts, saved
results, execution status and caveats for each branch, including exploratory
negative results. Selected aggregate figures are in `figures/`; small numeric
tables are in `results/`. Large generated grids, per-image arrays, source
datasets and model checkpoints are deliberately absent.

## Reproducible entry points

Run commands from this repository root. These examples show the interface;
full GPU runs need the external inputs described in [DATA.md](DATA.md).

```bash
# Local wavelet information diagnostic on MNIST-32
python -m wavelet_fm_structure.run_subband_information --device cuda --download

# CIFAR-10 U-Net-calibrated wavelet radius and generation
python -m wavelet_fm_structure.run_cifar10_fm_calibrated_radius --device cuda

# CIFAR-10 AsymFlow rank screening (trains multiple U-Nets; expensive)
python -m asymflow.sweep_asymflow_rank_cifar10 --projection-mode global

# Fit MNIST-64 GSM, then evaluate full-image denoising
python -m wavelet_gsm_local.run_experiment --image-size 64 --device cuda --download
python -m wavelet_gsm_local.run_image_denoising --source-dir wavelet_gsm_local/results/mnist64_haar2 --device cuda
```

The two GSM stages are intentionally separate; the second loads the model
fitted by the first. The AsymFlow sweep likewise requires its matching PCA
basis. The entry points above accept `--help` for their remaining settings.

## Interpretation and provenance

- A lower MSE against a pretrained U-Net is not necessarily a lower MSE
  against the true conditional velocity or clean image.
- The three polynomial families agree at degree one, but a richer dictionary
  need not generalize better off the fitting images.
- Mask size, source-candidate concentration, and visual quality measure
  different properties. A local posterior can be concentrated while different
  coordinates select different examples.
- The GSM result isolates nonlinear denoising at matched local context and
  first two moments. It has not established better FM generation.

This directory contains student-written experiments and adaptations. The
upstream implementation, Finn's privately shared code, datasets, and weights
are not redistributed. External origins and licensing constraints are listed
in [THIRD_PARTY.md](THIRD_PARTY.md). The author has confirmed permission to
publish this research code and the selected aggregate figures. No reuse
license has been assigned to the internship code.
