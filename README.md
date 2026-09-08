# spec_rt

[![PyPI version](https://img.shields.io/pypi/v/spec-rt.svg)](https://pypi.org/project/spec-rt/)
[![Documentation Status](https://readthedocs.org/projects/specrt/badge/?version=latest)](https://specrt.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21052794.svg)](https://doi.org/10.5281/zenodo.21052794)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Gaussian-based radiative-transfer fitting for absorption and emission spectra.

`spec_rt` fits absorption and emission spectra with Gaussian components and
exports radiative-transfer decomposition results. The PyPI project name is
`spec_rt`; the Python import package is also `spec_rt`.

Version 1.0.2 requires gaussFitSpec 1.0.4 or newer and accelerates complicated absorption spectra with overlap-aware CNM
ordering and analytic emission-model derivatives. The radiative-transfer and
BIC selection method is unchanged, and exhaustive legacy ordering remains
available for validation. See the [performance documentation](https://specrt.readthedocs.io/en/latest/performance.html).

## Links

- PyPI: https://pypi.org/project/spec-rt/
- Documentation: https://specrt.readthedocs.io/en/latest/installation.html
- Source code: https://github.com/retarchen/specRT
- Citation DOI: https://doi.org/10.5281/zenodo.21052794
