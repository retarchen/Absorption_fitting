"""Efficient search helpers for the radiative-transfer emission fit."""

from __future__ import annotations

import itertools
import math

import numpy as np


def generate_cnm_orderings(
    cold_parameters,
    *,
    strategy="overlap",
    overlap_sigma=2.5,
    max_orderings=720,
):
    """Return deterministic CNM foreground/background orderings.

    ``strategy="exhaustive"`` returns every permutation, matching the legacy
    implementation.  ``strategy="overlap"`` only permutes connected groups of
    components whose Gaussian profiles overlap appreciably.  Relative order is
    immaterial for separated components, so this avoids factorial duplication
    while retaining the same radiative-transfer model for blended components.

    Args:
        cold_parameters: Flat ``amplitude, center, sigma`` parameter array.
        strategy: ``"overlap"`` (default) or ``"exhaustive"``.
        overlap_sigma: Components are connected when their center separation is
            at most this value times their combined Gaussian width.
        max_orderings: Maximum overlap-aware orderings.  If the overlap graph
            still produces more candidates, a deterministic representative set
            is used.  Set to ``None`` to disable the cap.

    Returns:
        Integer array with shape ``(n_orderings, n_cold)``.
    """
    parameters = np.asarray(cold_parameters, dtype=float).reshape(-1, 3)
    n_cold = len(parameters)
    if n_cold == 0:
        return np.empty((1, 0), dtype=int)
    if strategy not in {"overlap", "exhaustive"}:
        raise ValueError("cnm_order_strategy must be 'overlap' or 'exhaustive'.")
    if overlap_sigma <= 0:
        raise ValueError("cnm_overlap_sigma must be positive.")
    if max_orderings is not None and max_orderings < 1:
        raise ValueError("max_cnm_orderings must be >= 1 or None.")

    indices = tuple(range(n_cold))
    if strategy == "exhaustive":
        return np.asarray(list(itertools.permutations(indices)), dtype=int)

    centers = parameters[:, 1]
    sigmas = np.abs(parameters[:, 2])
    adjacency = [set() for _ in indices]
    for left in indices:
        for right in range(left + 1, n_cold):
            combined_sigma = np.hypot(sigmas[left], sigmas[right])
            if abs(centers[left] - centers[right]) <= overlap_sigma * combined_sigma:
                adjacency[left].add(right)
                adjacency[right].add(left)

    groups = []
    unseen = set(indices)
    while unseen:
        pending = [unseen.pop()]
        group = []
        while pending:
            current = pending.pop()
            group.append(current)
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            pending.extend(neighbors)
        groups.append(tuple(sorted(group, key=lambda item: centers[item])))
    groups.sort(key=lambda group: min(centers[list(group)]))

    group_permutations = [tuple(itertools.permutations(group)) for group in groups]
    total_orderings = math.prod(len(candidates) for candidates in group_permutations)
    if max_orderings is None or total_orderings <= max_orderings:
        orderings = [
            tuple(itertools.chain.from_iterable(parts))
            for parts in itertools.product(*group_permutations)
        ]
        return np.asarray(orderings, dtype=int)

    # Very heavily blended spectra can still have a factorial search.  Include
    # physically natural orders, then fill the cap with reproducible samples.
    natural = tuple(np.argsort(centers).tolist())
    selected = {natural, tuple(reversed(natural)), indices}
    rng = np.random.default_rng(0)
    while len(selected) < max_orderings:
        sampled_groups = [candidates[rng.integers(len(candidates))] for candidates in group_permutations]
        selected.add(tuple(itertools.chain.from_iterable(sampled_groups)))
    return np.asarray(sorted(selected)[:max_orderings], dtype=int)


def build_emission_model(x, cold_parameters, order, fractions, t_sky):
    """Build the legacy radiative-transfer model and its analytic Jacobian.

    The fitted parameter order is unchanged: CNM velocities, CNM spin
    temperatures, then repeating WNM amplitude, center, and sigma values.
    """
    x = np.asarray(x, dtype=float)
    cold = np.asarray(cold_parameters, dtype=float).reshape(-1, 3)
    order = np.asarray(order, dtype=int)
    fractions = np.asarray(fractions, dtype=float)
    n_cold = len(cold)
    amplitudes = cold[:, 0]
    sigmas = cold[:, 2]

    def components(parameters):
        parameters = np.asarray(parameters, dtype=float)
        velocities = parameters[:n_cold]
        temperatures = parameters[n_cold : 2 * n_cold]
        warm = parameters[2 * n_cold :].reshape(-1, 3)

        offsets = x[None, :] - velocities[:, None]
        cold_tau = amplitudes[:, None] * np.exp(
            -0.5 * (offsets / sigmas[:, None]) ** 2
        )
        ordered_tau = cold_tau[order]
        tau_before = np.vstack(
            [np.zeros_like(x), np.cumsum(ordered_tau, axis=0)[:-1]]
        )
        attenuation = np.exp(-tau_before)
        ordered_temperatures = temperatures[order, None]
        cold_source = (
            (1.0 - np.exp(-ordered_tau))
            * ordered_temperatures
            * attenuation
        )
        total_transmission = np.exp(-np.sum(cold_tau, axis=0))

        if len(warm):
            warm_offsets = x[None, :] - warm[:, 1, None]
            warm_gaussians = warm[:, 0, None] * np.exp(
                -0.5 * (warm_offsets / warm[:, 2, None]) ** 2
            )
            warm_transmission = (
                fractions[:, None]
                + (1.0 - fractions[:, None]) * total_transmission
            )
        else:
            warm_offsets = np.empty((0, len(x)))
            warm_gaussians = np.empty((0, len(x)))
            warm_transmission = np.empty((0, len(x)))

        model = (
            np.sum(cold_source, axis=0)
            + np.sum(warm_transmission * warm_gaussians, axis=0)
            + t_sky * (total_transmission - 1.0)
        )
        return (
            model,
            cold_tau,
            offsets,
            cold_source,
            attenuation,
            total_transmission,
            warm,
            warm_offsets,
            warm_gaussians,
            warm_transmission,
        )

    def model(_x, *parameters):
        return components(parameters)[0]

    def jacobian(_x, *parameters):
        (
            _,
            cold_tau,
            offsets,
            cold_source,
            attenuation,
            total_transmission,
            warm,
            warm_offsets,
            warm_gaussians,
            warm_transmission,
        ) = components(parameters)
        temperatures = np.asarray(parameters[n_cold : 2 * n_cold], dtype=float)
        n_parameters = 2 * n_cold + 3 * len(warm)
        derivative = np.zeros((len(x), n_parameters), dtype=float)

        order_position = np.empty(n_cold, dtype=int)
        order_position[order] = np.arange(n_cold)
        d_tau_d_velocity = cold_tau * offsets / sigmas[:, None] ** 2
        warm_absorbed = (
            np.sum((1.0 - fractions[:, None]) * warm_gaussians, axis=0)
            if len(warm)
            else np.zeros_like(x)
        )
        for component in range(n_cold):
            position = order_position[component]
            own_source_derivative = (
                np.exp(-cold_tau[component])
                * d_tau_d_velocity[component]
                * temperatures[component]
                * attenuation[position]
            )
            later_source = np.sum(cold_source[position + 1 :], axis=0)
            transmission_derivative = (
                -total_transmission
                * d_tau_d_velocity[component]
                * (warm_absorbed + t_sky)
            )
            derivative[:, component] = (
                own_source_derivative
                - d_tau_d_velocity[component] * later_source
                + transmission_derivative
            )
            derivative[:, n_cold + component] = (
                (1.0 - np.exp(-cold_tau[component])) * attenuation[position]
            )

        for warm_index, warm_component in enumerate(warm):
            column = 2 * n_cold + 3 * warm_index
            amplitude, _, sigma = warm_component
            gaussian_unit = (
                warm_gaussians[warm_index] / amplitude
                if amplitude != 0
                else np.exp(-0.5 * (warm_offsets[warm_index] / sigma) ** 2)
            )
            derivative[:, column] = warm_transmission[warm_index] * gaussian_unit
            derivative[:, column + 1] = (
                warm_transmission[warm_index]
                * warm_gaussians[warm_index]
                * warm_offsets[warm_index]
                / sigma**2
            )
            derivative[:, column + 2] = (
                warm_transmission[warm_index]
                * warm_gaussians[warm_index]
                * warm_offsets[warm_index] ** 2
                / sigma**3
            )
        return derivative

    return model, jacobian
