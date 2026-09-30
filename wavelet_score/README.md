# MNIST wavelet baselines

This module contains the orthogonal 2D wavelet transform, coefficient indexing,
three polynomial feature families inspired by Finn et al., and the initial
masked empirical FM estimators. It is the common numerical foundation for the
later subband-radius and coarse-to-fine studies.

| Entry point | Question |
| --- | --- |
| `python -m wavelet_score.run_mnist32_denoising --help` | Can restricted wavelet features denoise MNIST-32? |
| `python -m wavelet_score.run_mnist32_fm_velocity_generation --help` | How do the corresponding velocity estimates integrate under Euler? |
| `python -m wavelet_score.run_mnist32_masked_empirical_fm --help` | How do explicit information masks change FM velocity estimation and generation? |

Read `specification.md`, `fm_specification.md` and
`masked_fm_specification.md` for precise targets and conventions. The
polynomial experiment in `wavelet_fm_polynomial/` is the maintained dense
velocity-regression study. Author-supplied Finn code and unstable historical
dense audits are not copied into this repository.
