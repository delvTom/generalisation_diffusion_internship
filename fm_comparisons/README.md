# Pixel-space Flow Matching comparisons

These experiments adapt empirical conditional estimators, affine Wiener
baselines and sensitivity-derived masks to the straight FM path. They compare
predictions with a pretrained U-Net and, in separate scripts, generate samples
from matched initial noise. They require the upstream `local_diffusion` package
and matching dataset/checkpoint described in [DATA.md](../DATA.md).

| Script | Role |
| --- | --- |
| `plot_fm_mask_sensitivity.py` | Compare denoiser, affine velocity and U-Net sensitivity maps. |
| `compare_fm_unet_closed_form_mse.py` | Evaluate teacher-MSE for empirical masked estimates and the Wiener velocity baseline. |
| `generate_fm_unet_closed_form_samples.py` | Generate comparison grids from shared noise. |
| `streaming_softmax.py` | Numerically stable global candidate accumulation; normalization is over the whole bank, not separately over each chunk. |
| `hybrid_wiener_covariance_cifar10.py` | Secondary negative control for global/local covariance hybrids. |
| `generate_pca_hybrid_artem_wiener_cifar10.py`, `generate_asym_artem_mask_cifar10.py`, `generate_wiener_asym_hybrid_sampler_cifar10.py` | Exploratory target/PCA hybrids; do not interpret their old images as a validated improvement. |

Example interface check:

```bash
python -m fm_comparisons.compare_fm_unet_closed_form_mse --help
```

The corrected `WeightedStreamingSoftmax` implementation and its chunk-size
test were validated in the preparation workspace. Tests are not distributed
in this compact research folder, but the implementation remains local here.
The retained [teacher-MSE plot](../figures/mask_teacher_mse.png) is a
comparison with one teacher checkpoint, not a universal sample-quality metric.
