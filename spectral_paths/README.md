# Spectral flow-matching paths

These experiments change **when** each PCA coordinate receives signal while
keeping the initial noise and final data distribution fixed. They test whether
a coarse-to-fine ordering, by itself, changes the behavior of empirical or
affine analytical estimators. It is a path study, not a new trained model.

## Kappa-indexed path

In the centered PCA basis, coordinate `i` follows

```text
z_t[i] = alpha_i(t) epsilon[i] + beta_i(t) z_clean[i]
r_i = (lambda_i / lambda_star) ** (kappa - 1)
beta_i(t) = t*r_i / (1-t+t*r_i)
alpha_i(t) = 1-beta_i(t)
```

`kappa=1` recovers the straight FM path. `kappa=0` equalizes the average
spectral signal-to-noise ratio, and negative values reverse the usual
hierarchy. The empirical closed-form uses a time-dependent Mahalanobis
distance in this basis. The two scripts compare unconstrained and local
posterior constructions:

```bash
python -m spectral_paths.generate_kappa_unconstrained_closed_form --help
python -m spectral_paths.generate_kappa_local_artem_closed_form --help
```

The old trajectory runs showed that paths can substantially change the order
of reconstruction. They did **not** establish that an unconstrained empirical
posterior stops concentrating on a training candidate near the endpoint.
Some neighbor diagnostics in the original workspace were computed on a final
sample rather than the corresponding saved intermediate state and are not
exported here.

## Sequential schedule and affine control

`sequential_linear_schedule/` contains a separate controlled experiment with
orthogonal PCA paths, exact affine flow, explicit Euler solvers and optional
FID. The point is to distinguish path effects from discretization error. Its
default YAML assumes CIFAR-10 and an external PCA/Wiener cache; see
[DATA.md](../DATA.md). The parser and fast schedule tests can be checked with:

```bash
python -m spectral_paths.sequential_linear_schedule.run_affine_experiment --help
python -m pytest -q spectral_paths/sequential_linear_schedule/tests/test_schedule.py
```

`--pilot --skip-fid` is a reduced run, but it still needs external CIFAR-10
data and compatible upstream configuration. Full FID runs are not part of the
repository-cleanup verification.
