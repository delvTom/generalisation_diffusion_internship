# Inputs and runtime dependencies

Run scripts from the repository root. The Python requirements are in
`requirements.txt` (core CPU experiments) and `requirements-fm.txt` (U-Net,
AsymFlow and FID). Install a PyTorch build appropriate for the machine before
installing either file. The development environment used Python 3.13,
torch 2.9, torchvision 0.24, NumPy 2.3, PyWavelets 1.9, Matplotlib 3.10,
`torchcfm` 1.0.7, torchmetrics 1.9 and torch-fidelity 0.4.

## Upstream package

Several scripts use the Lukoianov et al. `local_diffusion` dataset loader,
Wiener preprocessing or reference estimators. Fetch the pinned upstream
checkout separately:

```bash
python scripts/setup_external.py
```

This creates ignored `external/upstream_locality/` at commit
`48d69957a668eeaeff8efad50eb26830299e6484` and links its `src/` and
`configs/` into this repository. It also installs the upstream package in the
active environment. An offline clone may be supplied with `--source PATH`.
The upstream license remains attached to that separate checkout; this project
does not relicense it.

## Data and weights

| Experiment | Required input | Location / command |
| --- | --- | --- |
| MNIST-32 wavelet and MNIST-64 GSM | Official MNIST train/test data; MNIST-64 is resized MNIST, not a different dataset | `data/datasets/` or `--data-root`, with `--download` where supported |
| CIFAR-10 studies | Official CIFAR-10 train/test data | `data/datasets/`; some scripts support `--download`; the upstream dataset config controls normalization to `[-1,1]` for U-Net studies |
| FM U-Net teacher | Checkpoint matching the dataset, time direction and output convention | Default CIFAR-10 path: `data/models/fm_unet/cifar10/fm_cifar10_weights_step_400000.pt`; override with `--checkpoint` where offered |
| CelebA-HQ teacher and data | Separately authorized dataset/checkpoint | Default weight hint: `data/models/fm_unet/celeba_hq/fm_celeba_hq_weights.pt`; obtain from the owner, not this repository |
| Global PCA / Wiener SVD | CIFAR-10 `U.pt` and `LA.pt` from the upstream Wiener preprocessing | `data/models/wiener/cifar10_32/`; AsymFlow global-rank runs require these files |
| Patch PCA for AsymFlow | Basis computed from CIFAR-10 train patches | `python -m asymflow.compute_patchwise_pca_cifar10 --patch-size 4` (likewise 8 and 16) |
| AsymFlow generation | Matching checkpoint produced by this repository's training/sweep script | Pass `--checkpoint`; no trained weights are bundled |

The exact pretrained FM U-Net weights used in the internship were local files
without recorded redistribution permission. They are **not** bundled and no
download URL is asserted here. A different checkpoint can change every
teacher-calibrated mask or teacher-MSE result. Results involving the teacher
are reproducible only once a checkpoint with the same architecture,
normalization, training target and time convention is available.

All output directories under experiment packages are ignored by Git. Curated
aggregate figures and small JSON/CSV results in top-level `figures/` and
`results/` are snapshots of past runs, not reusable training artifacts.
Large caches, per-image predictions, MLflow state and source image grids are
not distributed.

The original path `experiments_tom/...` appearing *inside saved AsymFlow FID
JSON* is historical checkpoint metadata from the completed runs. New runs of
the relocated scripts use `asymflow/results/...` instead.
FID computation may fetch its Inception feature-extractor weights on first use.
Only load checkpoints from a trusted source: PyTorch checkpoint files may
contain serialized objects, and the AsymFlow loader reads its own run metadata.
