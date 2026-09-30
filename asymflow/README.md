# Asymmetric FM on CIFAR-10

The U-Net predicts a mixed target: velocity on a PCA subspace and clean data
on its orthogonal complement. Its output is converted back to a full velocity
for Euler generation. Global PCA and a shared PCA over image patches are kept
as separate projection modes; changing the patch size changes the experiment.

The scripts are ordered as follows:

1. Obtain the official CIFAR-10 data and set up the pinned upstream package
   using `python scripts/setup_external.py` from the repository root.
2. For global PCA, provide the upstream Wiener SVD `U.pt` and `LA.pt` under
   `data/models/wiener/cifar10_32/`. For patchwise PCA, run
   `python -m asymflow.compute_patchwise_pca_cifar10 --patch-size 4`.
3. Train one point with `python -m asymflow.train_asymflow_unet_cifar10
   --projection-mode patchwise --patch-size 4 --rank 8`, or run the short
   screening with `python -m asymflow.sweep_asymflow_rank_cifar10
   --projection-mode patchwise --patch-sizes 4,8,16`.
4. Generate from a matching saved checkpoint with
   `python -m asymflow.generate_asymflow_unet_cifar10 --checkpoint PATH`.

The historical screening used 50,000 training steps per rank, 100 Euler
sampling steps and 10,000 samples for FID, with one seed. Its saved JSON and
plots are under `results/asymflow_screening/`. Patch 4, rank 8 gave the lowest
*observed* FID (32.98), but this was not replicated across seeds or validated
by a long training run at that point. Checkpoint paths inside those JSON files
refer to the old worktree; newly run scripts write under `asymflow/results/`.

These scripts are the student's adaptation. The U-Net implementation and
dataset loader remain separately installed dependencies; neither is vendored
here. See [DATA.md](../DATA.md) and [THIRD_PARTY.md](../THIRD_PARTY.md).
