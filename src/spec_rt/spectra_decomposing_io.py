"""Input/output helpers for the radiative-transfer workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _to_scalar(value):
    """Return a CSV-safe scalar, converting one-element arrays and empty arrays."""
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.size == 0:
        return np.nan
    return array.reshape(-1)[0].item()


def _as_float_array(values):
    """Normalize possibly nested scalar-like values into a float array."""
    return np.asarray([_to_scalar(value) for value in values], dtype=float)


def load_six_column_spectrum(path, *, absorption_format="transmission"):
    """Load the six-column example table used by the original notebook.

    Args:
        path: Text file containing six columns in the order absorption
            velocity, absorption spectrum, absorption error, emission velocity,
            emission spectrum, emission error.
        absorption_format: ``"transmission"`` expects the second column to be
            ``exp(-tau)`` and converts it to ``1 - exp(-tau)``. Use
            ``"one_minus_exp_tau"`` to keep the column unchanged.

    Returns:
        A tuple ``(x, y, yerr, xemi, yemi, yemi_err)`` with non-finite
        emission rows removed.
    """

    data = np.atleast_2d(np.loadtxt(path))
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("Expected a text table with exactly six columns.")

    x = data[:, 0]
    raw_absorption = data[:, 1]
    yerr = data[:, 2]

    if absorption_format == "transmission":
        y = 1.0 - raw_absorption
    elif absorption_format == "one_minus_exp_tau":
        y = raw_absorption
    else:
        raise ValueError("absorption_format must be 'transmission' or 'one_minus_exp_tau'.")

    emission_mask = np.isfinite(data[:, 3]) & np.isfinite(data[:, 4]) & np.isfinite(data[:, 5])
    xemi = data[emission_mask, 3]
    yemi = data[emission_mask, 4]
    yemi_err = data[emission_mask, 5]
    return x, y, yerr, xemi, yemi, yemi_err


def validate_absorption_input(y, yerr, *, edge_fraction=0.1, sigma_limit=3.0):
    """Validate that the absorption input behaves like ``1 - exp(-tau)``.

    The function checks that the edge channels stay close to a zero baseline
    within a configurable number of noise sigma.

    Returns:
        A dictionary with the measured baseline mean, baseline scatter, and
        reference noise level.
    """

    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if y.ndim != 1 or yerr.ndim != 1 or y.size != yerr.size:
        raise ValueError("y and yerr must be one-dimensional arrays of equal length.")

    n_edge = max(5, int(len(y) * edge_fraction))
    edge = np.concatenate([y[:n_edge], y[-n_edge:]])
    baseline_mean = float(np.nanmean(edge))
    baseline_std = float(np.nanstd(edge))
    noise_level = float(np.nanmedian(np.abs(yerr)))
    limit = max(noise_level * sigma_limit, baseline_std * sigma_limit, 1.0e-6)
    if abs(baseline_mean) > limit:
        raise ValueError(
            "Absorption baseline is too far from zero for a 1 - exp(-tau) input. "
            f"Measured baseline={baseline_mean:.4f}, allowed={limit:.4f}."
        )
    if np.nanmax(y) > 1.5:
        raise ValueError("Absorption input exceeds the expected 1 - exp(-tau) range.")
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "noise_level": noise_level,
    }


def write_table_outputs(
    model,
    *,
    name,
    popt,
    pcov,
    gausf,
    funT,
    xemi,
    Or,
    fit_e,
    mean_Ts,
    sigma_meanTsf,
    v_shift,
    F_values,
):
    """Persist the CSV outputs produced by the legacy workflow."""

    output_dir = Path(model.datapath)
    output_dir.mkdir(parents=True, exist_ok=True)

    NHI_c = 0.0
    NHI_w = 0.0
    sigma_NHIc = 0.0
    sigma_NHIw = 0.0
    K = 1.823e18 / 1e20
    ncold = int(len(popt) / 3)
    mean_Ts = _as_float_array(mean_Ts)
    sigma_meanTsf = _as_float_array(sigma_meanTsf)
    Or = _as_float_array(Or)
    v_shift = _as_float_array(v_shift)
    F_values = _as_float_array(F_values)

    if model.renew:
        for filename in ["CNMonlydata.csv", "Fulldata.csv", "WNMonlydata.csv"]:
            file_path = output_dir / filename
            if file_path.exists():
                df_existing = pd.read_csv(file_path)
                df_existing = df_existing[df_existing["Name"] != name]
                df_existing.reset_index(drop=True, inplace=True)
                df_existing.to_csv(file_path, mode="w", index=False, header=True)

    for i in range(ncold):
        j = i * 3
        _popt = popt[j : j + 3]
        _pcov = pcov[j : j + 3]
        if mean_Ts[i] <= 1000:
            NHI_c += K * mean_Ts[i] * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]
            _d = (
                (K * sigma_meanTsf[i] * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
                + (K * mean_Ts[i] * _pcov[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
                + (K * mean_Ts[i] * _popt[0] * np.sqrt(2 * np.pi) * _pcov[2]) ** 2
            )
            sigma_NHIc += _d
        else:
            NHI_w += K * mean_Ts[i] * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]
            _d = (
                (K * sigma_meanTsf[i] * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
                + (K * mean_Ts[i] * _pcov[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
                + (K * mean_Ts[i] * _popt[0] * np.sqrt(2 * np.pi) * _pcov[2]) ** 2
            )
            sigma_NHIw += _d
    sigma_NHIc = np.sqrt(sigma_NHIc)

    for i in range(int(len(gausf) / 3)):
        j = i * 3
        _popt = gausf[j : j + 3]
        _pcov = fit_e[j : j + 3]
        NHI_w += K * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]
        _d = (K * _popt[0] * np.sqrt(2 * np.pi) * _pcov[2]) ** 2 + (
            K * _pcov[0] * np.sqrt(2 * np.pi) * _popt[2]
        ) ** 2
        sigma_NHIw += _d
    sigma_NHIw = np.sqrt(sigma_NHIw)

    f_c = NHI_c / (NHI_c + NHI_w)
    sigma_fc = np.sqrt(
        ((NHI_c / (NHI_c + NHI_w) ** 2 * sigma_NHIw)) ** 2
        + ((NHI_w / (NHI_c + NHI_w) ** 2 * sigma_NHIc)) ** 2
    )
    NHI_uncorr = K * np.trapezoid(funT, xemi)
    total_NHI_c = NHI_c
    total_sigma_NHIc = sigma_NHIc
    total_NHI_w = NHI_w
    total_sigma_NHIw = sigma_NHIw
    total_f_c = f_c
    total_sigma_fc = sigma_fc

    df = pd.DataFrame(
        [
            {
                "Name": name,
                "NHI_c": total_NHI_c,
                "Sigma_NHIc": total_sigma_NHIc,
                "NHI_w": total_NHI_w,
                "Sigma_NHIw": total_sigma_NHIw,
                "f_c": total_f_c,
                "Sigma_fc": total_sigma_fc,
                "Tsky": model.Tsky,
                "NHI_uncorr_fit": NHI_uncorr,
            }
        ]
    )
    file_path = output_dir / "Fulldata.csv"
    df.to_csv(file_path, mode="a", index=False, header=not file_path.exists())

    for i in range(ncold):
        j = i * 3
        _popt = popt[j : j + 3]
        _pcov = pcov[j : j + 3]
        fwhm_ = 2.35482 * _popt[2]
        NHI_c = K * mean_Ts[i] * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]
        NHI_c_err = np.sqrt(
            (K * sigma_meanTsf[i] * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
            + (K * mean_Ts[i] * _pcov[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
            + (K * mean_Ts[i] * _popt[0] * np.sqrt(2 * np.pi) * _pcov[2]) ** 2
        )
        T_K = 21.866 * fwhm_**2
        df = pd.DataFrame(
            [
                {
                    "Name": name,
                    "tau": _popt[0],
                    "Sigma_tau": _pcov[0],
                    "velocity_tau": _popt[1],
                    "Sigma_velocity_tau": _pcov[1],
                    "fwhm": fwhm_,
                    "Sigma_fwhm": 2.355 * _pcov[2],
                    "mean_Ts": mean_Ts[i],
                    "sigma_mean_Ts": sigma_meanTsf[i],
                    "T_k_max": T_K,
                    "NHI_c": NHI_c,
                    "Sigma_NHI_c": NHI_c_err,
                    "v_shift": v_shift[i],
                    "Order": Or[i],
                }
            ]
        )
        file_path = output_dir / "CNMonlydata.csv"
        df.to_csv(file_path, mode="a", index=False, header=not file_path.exists())

    for i in range(int(len(gausf) / 3)):
        j = i * 3
        _popt = gausf[j : j + 3]
        _popt_err = fit_e[j : j + 3]
        fwhm_ = 2.35482 * _popt[2]
        NHI_w = K * _popt[0] * np.sqrt(2 * np.pi) * _popt[2]
        NHI_w_err = np.sqrt(
            (K * _popt[0] * np.sqrt(2 * np.pi) * _popt_err[2]) ** 2
            + (K * _popt_err[0] * np.sqrt(2 * np.pi) * _popt[2]) ** 2
        )
        T_K = 21.866 * fwhm_**2
        df = pd.DataFrame(
            [
                {
                    "Name": name,
                    "TB_WNM": _popt[0],
                    "Sigma_TB_WNM": _popt_err[0],
                    "velocity_TB": _popt[1],
                    "Sigma_velocity_TB": _popt_err[1],
                    "fwhm": fwhm_,
                    "Sigma_fwhm": 2.355 * _popt_err[2],
                    "T_k_max": T_K,
                    "NHI_w": NHI_w,
                    "Sigma_NHI_w": NHI_w_err,
                    "F_value": F_values[i],
                }
            ]
        )
        file_path = output_dir / "WNMonlydata.csv"
        df.to_csv(file_path, mode="a", index=False, header=not file_path.exists())

    return {
        "NHI_c": total_NHI_c,
        "Sigma_NHIc": total_sigma_NHIc,
        "NHI_w": total_NHI_w,
        "Sigma_NHIw": total_sigma_NHIw,
        "f_c": total_f_c,
        "Sigma_fc": total_sigma_fc,
        "NHI_uncorr_fit": NHI_uncorr,
    }
