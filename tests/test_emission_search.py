from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spec_rt.spectra_decomposing_search import (
    build_emission_model,
    generate_cnm_orderings,
)


def test_overlap_ordering_removes_only_separated_component_permutations():
    cold = np.array(
        [
            [0.08, 116.0, 1.2],
            [0.13, 141.0, 3.0],
            [0.50, 158.0, 4.2],
            [0.19, 166.0, 1.5],
            [0.04, 175.0, 4.8],
        ]
    )

    overlap = generate_cnm_orderings(cold, strategy="overlap", overlap_sigma=2.5)
    exhaustive = generate_cnm_orderings(cold, strategy="exhaustive")

    assert overlap.shape == (6, 5)
    assert exhaustive.shape == (120, 5)
    assert all(sorted(order) == list(range(5)) for order in overlap.tolist())


def test_emission_model_analytic_jacobian_matches_finite_difference():
    x = np.linspace(-8.0, 8.0, 51)
    cold = np.array([[0.3, -1.0, 1.2], [0.5, 1.5, 1.8]])
    order = np.array([1, 0])
    fractions = np.array([0.0, 0.5])
    parameters = np.array(
        [-1.1, 1.4, 45.0, 80.0, 12.0, -2.0, 2.5, 8.0, 3.0, 3.5]
    )
    model, jacobian = build_emission_model(x, cold, order, fractions, 2.73)

    analytic = jacobian(x, *parameters)
    numerical = np.empty_like(analytic)
    for column, value in enumerate(parameters):
        step = 1.0e-6 * max(1.0, abs(value))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[column] += step
        lower[column] -= step
        numerical[:, column] = (
            model(x, *upper) - model(x, *lower)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-5, atol=2.0e-7)


def test_ordering_cap_is_deterministic_for_heavily_blended_components():
    cold = np.array([[0.1, float(index), 10.0] for index in range(7)])

    first = generate_cnm_orderings(cold, strategy="overlap", max_orderings=25)
    second = generate_cnm_orderings(cold, strategy="overlap", max_orderings=25)

    assert first.shape == (25, 7)
    np.testing.assert_array_equal(first, second)
