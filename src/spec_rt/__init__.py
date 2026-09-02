"""Public package interface for specRT."""

from gaussFitSpec import SpectrumFitResult, fit_spectrum, plot_fit, read_spectrum, save_components_csv
from .spectra_decomposing import SpectraDecomposing
from .spectra_decomposing_io import load_six_column_spectrum, validate_absorption_input
from .spectra_decomposing_plotting import create_legacy_axes

sd = SpectraDecomposing

__all__ = [
    "SpectrumFitResult",
    "SpectraDecomposing",
    "fit_spectrum",
    "plot_fit",
    "read_spectrum",
    "save_components_csv",
    "sd",
    "load_six_column_spectrum",
    "validate_absorption_input",
    "create_legacy_axes",
]
__version__ = "1.0.0"
