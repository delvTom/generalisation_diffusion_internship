# Provenance and third-party material

This repository is a clean-history selection of internship experiments, not a
fork presented under a new name. The original working tree was based on the
[Lukoianov et al. official repository](https://github.com/ottogin/locality-in-diffusion-models)
(base commit `48d6995`). Its `src/local_diffusion`, `configs`, original README,
notebooks, and assets are **not copied** here. Several FM scripts import that
package at runtime; clone and install it separately if those scripts are used.
The upstream `LICENSE.txt` contains a CC BY-NC-SA 4.0 license text, although
its first line abbreviates the designation inconsistently. Check the upstream
license and respect its terms when using the dependency.

| Source | What is used here | License / publication status |
| --- | --- | --- |
| [Lukoianov et al., *Locality in Image Diffusion Models Emerges from Data Statistics*](https://github.com/ottogin/locality-in-diffusion-models) | Analytical locality and sensitivity baseline; runtime `local_diffusion` dependency in selected scripts | Upstream license file as described above; no upstream source copied |
| [Kamb and Ganguli, *An analytic theory of creativity in convolutional diffusion models*](https://arxiv.org/abs/2412.20292) | Locality/equivariance motivation and empirical-posterior comparison | Paper cited; no Kamb source code copied; code license not asserted |
| [Finn et al., *Where the Score Lives*](https://arxiv.org/abs/2606.08309) | Wavelet-feature family definitions and polynomial baseline motivation | Paper cited; privately shared author implementation and email are excluded; its redistribution rights were not established |
| [Conditional Flow Matching (`torchcfm`)](https://github.com/atong01/conditional-flow-matching) | External U-Net implementation imported by FM scripts | MIT according to installed `torchcfm` package metadata; no library code copied |
| [PyWavelets](https://pywavelets.readthedocs.io/) | External orthogonal DWT implementation | Dependency only, no vendored code |

`fm_comparisons/streaming_softmax.py` is a local stable accumulator for the
empirical-posterior computation. Its role and the upstream batch-normalization
issue are documented in the module; its test checks chunk-size invariance.
The wavelet polynomial modules are a reimplementation and FM adaptation of
paper-defined feature families, **not** the authors' private files.

`asymflow/` contains the student's experiment scripts copied from the
`experiments_tom/` worktree of the separate SymFlow checkout and adapted here
to remove machine-specific paths. It uses the external `torchcfm` U-Net and
the separately installed Lukoianov data loader; the SymFlow checkout's
`lakonlab/`, demos and other source files are not copied. The short CIFAR-10
rank-sweep JSON/PNG results are included with their original protocol fields.

The extra PCA/Wiener hybrid scripts in `fm_comparisons/` are student
explorations with negative or ambiguous results. They are retained for
traceability and are not presented as a validated improvement.

The U-Net and Wiener checkpoint files found in the working tree had no
redistribution authorization recorded. They are excluded. The separately
received private Finn implementation is not part of this repository.

No license is assigned to the internship code in this snapshot. Public
release and any later license choice require confirmation of the author's and
host institution's rights. This does not alter third-party licenses.
