# Polynomial wavelet regression for Flow Matching

This directory contains one controlled MNIST-32 experiment. It compares the
analytic affine Flow Matching optimum with three nested polynomial feature
dictionaries in an orthonormal, critically sampled wavelet basis.

The path and target are

```text
X_t = t X + (1 - t) epsilon
V   = X - epsilon.
```

Images are transformed with `db2`, `level=3`, and `mode="periodization"`.
Every family has the exact same degree-one dictionary `[1, Y_t]`; its `D=1`
prediction is therefore the analytic affine Wiener FM estimator. For larger
degrees:

- `independent` adds univariate powers of every wavelet coefficient;
- `band` additionally adds products between H/V/D coefficients at the same
  scale and position;
- `local` additionally adds products between spatial neighbors from the same
  subband.

The readout from each global dictionary to the complete wavelet velocity is
dense. The polynomial weights are obtained by a direct linear-system solve,
not gradient descent. The primary protocol uses no ridge.

The current solver forms centered, standardized normal equations in float64
and uses Cholesky without adding jitter. This is more sensitive to conditioning
than a QR/SVD solve: an unsuccessful factorization stops explicitly. Both the
normal-equation residual and the observed fit-MSE nesting are checked. D=1 is
assigned to the analytic Wiener by construction, not independently fitted.

Run the full experiment with:

```bash
python3 wavelet_fm_polynomial/run_mnist_mse.py --device cuda
```

This uses all 60,000 MNIST training images and all 10,000 official test
images. The largest dictionary is `local`, `D=3`, with 13,837 features
including the intercept, so the unregularized system remains overdetermined.

For a quick pipeline check:

```bash
python3 wavelet_fm_polynomial/run_mnist_mse.py \
  --device cuda --families independent --degrees 1 \
  --times 0.5 --num-fit 512 --num-validation 128 \
  --out-dir /tmp/wavelet_fm_polynomial_smoke
```

The run writes `config.json`, `metrics.csv`, `diagnostics.json`,
`mse_fit.png`, and `mse_validation.png` under the selected output directory.

Degree-three generation uses direct 20-step Euler sampling:

```bash
python3 -u wavelet_fm_polynomial/run_mnist_generation.py --device cuda
```

At each nonzero Euler evaluation time `t_j = j / 20`, the generator fits each
`D=3` regressor directly on the fixed 60,000 clean/noise pairs and immediately
uses it for the corresponding Euler update. There is no temporal interpolation
and no MSE evaluation. At `t=0`, it uses the exact endpoint field
`E[V | X_0] = mu - X_0`. All families start from identical independent
Gaussian noises. The run writes a final image grid, a trajectory grid, raw
arrays, its configuration, and numerical fit diagnostics.
