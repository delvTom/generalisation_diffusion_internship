# Experiment map and evidence

The projects below were run in the original internship workspace. This
repository contains their selected source files and a small number of
aggregate exports; it does **not** contain the original data, checkpoints or
complete run directories. A saved figure is evidence of an executed protocol,
not a guarantee that every dependency is available in a new environment.

| Axis | Primary source | Evidence retained | Supported reading / limit |
| --- | --- | --- | --- |
| FM target and locality | `fm_comparisons/compare_fm_unet_closed_form_mse.py`, `plot_fm_mask_sensitivity.py` | [Teacher-MSE comparison](figures/mask_teacher_mse.png) | **Executed.** Denoiser- and velocity-derived sensitivities need not induce the same thresholded mask. The difference includes the diagonal term; teacher MSE is not intrinsic sample quality. Original results required corrected global posterior accumulation. |
| Asymmetric FM target | `asymflow/compute_patchwise_pca_cifar10.py`, `train_asymflow_unet_cifar10.py`, `sweep_asymflow_rank_cifar10.py`, `generate_asymflow_unet_cifar10.py` | [Global rank sweep](results/asymflow_screening/global/fid_vs_rank.png), [patch-4 rank sweep](results/asymflow_screening/patch_04/fid_vs_rank.png), their `fid_results.json` files | **Executed screening.** CIFAR-10, one seed, 50,000 training steps per rank, 10,000 FID samples, 100 Euler steps. The best observed patch-4 point is rank 8 with FID 32.98; no independent long-budget validation of that point is included. |
| Spectral probability paths | `spectral_paths/generate_kappa_unconstrained_closed_form.py`, `generate_kappa_local_artem_closed_form.py`, `sequential_linear_schedule/run_affine_experiment.py` | [Schedule](figures/spectral_schedule.png), [Euler error](figures/spectral_euler_relative_error.png); large image trajectories omitted | **Executed with diagnostic caveats.** Changing component schedules changes when information enters and can strongly affect Euler error. It does not alone establish avoidance of global empirical-posterior concentration. Some old posterior-neighbor diagnostics evaluated a final state rather than the recorded intermediate state and are intentionally not exported. |
| MNIST wavelet foundations | `wavelet_score/run_mnist32_denoising.py`, `run_mnist32_fm_velocity_generation.py`, `run_mnist32_masked_empirical_fm.py` | Code and protocol notes; full grids omitted | **Executed.** Orthogonal transforms, coefficient indexing and restricted empirical estimators support the later studies. Denoising MSE, velocity MSE and sample grids answer different questions and are not pooled. |
| Wavelet information hierarchy | `wavelet_fm_structure/plot_subband_snr.py` | [Effective SNR](figures/effective_snr_by_subband.png) | Coarse bands tend to become informative earlier on MNIST-32. This is descriptive, not a proof of an optimal hard schedule. |
| Adaptive wavelet masks | `wavelet_fm_structure/run_cifar10_fm_calibrated_radius.py` | [Radii and U-Net MSE](figures/adaptive_radius_and_unet_mse.png) | The `Full` family calibrates a radius by time and subband on separate CIFAR-10 queries; the radius is reused for other families. Teacher MSE depends on the teacher, the small query sample, and the mask budget. Generations were often visibly weaker than U-Net samples. |
| Forced coarse-to-fine mask | `wavelet_coarse_to_fine/run_cifar10_baseline.py` | [Diagnostic](figures/coarse_to_fine_diagnostics.png) | A hard sequence of coarse/intermediate/fine contexts did not yield convincing generations and could produce patch-like artifacts. It is a negative result, not an optimized method. |
| Dense polynomial wavelet dictionaries | `wavelet_fm_polynomial/run_mnist_mse.py` | [Fit vs held-out MSE](figures/polynomial_train_test.png) | Increasing feature capacity lowers fit risk, but held-out risk can worsen; a fit curve is not evidence of a better velocity field off training samples. The figure is on MNIST-32 and is not directly comparable with GSM MNIST-64. |
| Local GSM denoising | `wavelet_gsm_local/run_experiment.py`, then `run_image_denoising.py` | [MSE and gains](figures/gsm_denoising_results.png), [fourth-order diagnostic](figures/fourth_order_excess_heatmaps.png), [CSV metrics](results/gsm_mnist64/image_mse_metrics.csv) | On held-out MNIST-64, GSM local improves unclipped reconstruction MSE relative to a local Wiener with the same context and matched first two moments. This does not by itself imply better FM generation. |
| Wiener-induced wavelet masks | `wavelet_empirical_cifar/run_wiener_masks.py`, `run_celeba_wiener_comparison.py` | [Selected sites](figures/wavelet_wiener_mask_sites.png), [CIFAR-10 MSE](figures/wavelet_wiener_mask_mse.png), [evaluation table](results/wavelet_wiener_masks/evaluation.csv) | **Executed.** The affine denoiser's sensitivity selects visible wavelet sites; an empirical posterior predicts each output from the selected sites. A nine-site mask and a thresholded variable-size mask have different budgets and cannot isolate location alone. The CelebA-HQ image grid is not bundled. |
| Spatial sharing and coarse context | `wavelet_empirical_cifar/run_spatial_sharing.py`, `run_coarse_context.py` | Source retained; large galleries omitted | **Exploratory/negative.** These test translation sharing and extra coarse context separately. Their results do not establish a general improvement over the calibrated wavelet-mask experiment. |
| Hybrid affine controls | `fm_comparisons/hybrid_wiener_covariance_cifar10.py`, `generate_pca_hybrid_artem_wiener_cifar10.py`, `generate_asym_artem_mask_cifar10.py` | Source retained; older result trees omitted | **Exploratory/negative.** Tested PCA/patch covariance and target hybrids do not give a clean generative improvement. Some direct asymmetric-target variants were not valid velocity comparisons; do not use their old figures as primary evidence. |

## Key protocol distinctions

- **FM velocity vs denoising:** FM experiments study the conditional velocity
  or agreement with a pretrained FM U-Net. The local GSM result studies clean
  image reconstruction under `X_t=t X_clean+(1-t) noise`; its MSE is not a
  velocity MSE.
- **Train vs test:** Polynomial fit and held-out curves answer different
  questions. GSM parameters use MNIST train images and are evaluated on the
  official 10,000-image test split with paired noise.
- **Wavelet basis:** MNIST-32 mask studies use orthogonal `db2` at level 3;
  the MNIST-64 GSM experiment uses Haar at level 2. Do not pool their metrics.
- **Matched Wiener:** The local Wiener baseline uses the GSM's fitted mean and
  covariance and the same valid 3x3 detail-band context. `A2` is estimated
  with a shared affine Wiener model for every method; detail-band borders use
  self-only estimators.
- **Display vs metric:** GSM reconstruction figures are displayed after
  clipping to `[0,1]`, but reported image MSE uses raw unclipped predictions.
- **AsymFlow vs analytical estimators:** AsymFlow trains a U-Net and reports FID.
  The polynomial, masked-posterior and GSM experiments estimate explicit
  statistical rules. Their reported errors are not directly comparable.

## Regeneration and status

The two GSM stages have separate entry points: fit subband parameters, then
evaluate image-level denoising without a new fit. The first has a CPU smoke
mode. The CIFAR-10 wavelet experiments need a compatible FM U-Net checkpoint;
`--no-unet` changes the research question and must be reported as such. The
AsymFlow, spectral FID and long empirical runs are expensive and were not
repeated for this repository cleanup. See each module's README and [DATA.md](DATA.md).

The privately supplied Finn notebook and `wavelet_Emma/experiments_tom` runs
are excluded: the code has unverified redistribution rights and several runs
had numerical failures or inconsistent degree-one outputs. The independent
FM polynomial study in `wavelet_fm_polynomial/` is the maintained comparison.
Early scripts in `MSE_loss_project/` are likewise excluded from this paper-style
package: their tiny, heterogeneous candidate banks and occasional inclusion
of evaluation queries in the bank make them unsuitable for the main claims.
They remain in the original workspace, not silently deleted.
