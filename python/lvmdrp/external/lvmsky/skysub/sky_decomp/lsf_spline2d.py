"""Split-zodi decomposition with a continuous two-dimensional spline LSF."""

from __future__ import annotations

from dataclasses import asdict
from functools import cache, lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

from .fit import HC_OVER_KB_CMK, LSF_CHANNELS
from .lsf_surface_iterative import (
    LSFSurfaceIterativeConfig,
    LSFSurfaceState,
    SkyDecompLSFSurfaceIterative,
    _channel_mask,
    _configured_knot_vector,
    _fit_bspline_channel,
    _resolved_spline_config,
    _spline_strategy_summary,
    _wave_fingerprint,
    build_bspline_basis,
    evaluate_bspline_basis,
)


LSF_OFFSET_BASIS_COUNT = 11
LSF_OFFSET_DEGREE = 3
LSF_OFFSET_LOWER_ANGSTROM = -3.0
LSF_OFFSET_UPPER_ANGSTROM = 3.0
LSF_SPLINE2D_REPRESENTATION = "continuous_mspline_density"
_OFFSET_KNOTS = np.linspace(
    LSF_OFFSET_LOWER_ANGSTROM,
    LSF_OFFSET_UPPER_ANGSTROM,
    LSF_OFFSET_BASIS_COUNT + LSF_OFFSET_DEGREE + 1,
)
_DIAGNOSTIC_DELTA = np.linspace(
    LSF_OFFSET_LOWER_ANGSTROM,
    LSF_OFFSET_UPPER_ANGSTROM,
    241,
)


@cache
def _offset_splines() -> tuple[tuple[BSpline, ...], tuple[BSpline, ...], np.ndarray]:
    splines = tuple(
        BSpline.basis_element(
            _OFFSET_KNOTS[index : index + LSF_OFFSET_DEGREE + 2],
            extrapolate=False,
        )
        for index in range(LSF_OFFSET_BASIS_COUNT)
    )
    areas = np.array(
        [
            spline.integrate(_OFFSET_KNOTS[index], _OFFSET_KNOTS[index + 4])
            for index, spline in enumerate(splines)
        ]
    )
    return splines, tuple(spline.antiderivative() for spline in splines), areas


def mspline_basis(delta_wavelength: np.ndarray, derivative: int = 0) -> np.ndarray:
    """Evaluate the eleven unit-integral cubic M-splines on [-3, 3] Angstrom."""
    if derivative not in (0, 1):
        raise ValueError("only value and first derivative are supported")
    coordinate = np.asarray(delta_wavelength, dtype=float)
    flat = coordinate.ravel()
    result = np.zeros((flat.size, LSF_OFFSET_BASIS_COUNT))
    splines, _, areas = _offset_splines()
    for index, spline in enumerate(splines):
        lower, upper = _OFFSET_KNOTS[index], _OFFSET_KNOTS[index + 4]
        inside = (flat >= lower) & (flat <= upper)
        evaluator = spline if derivative == 0 else spline.derivative()
        result[inside, index] = evaluator(flat[inside]) / areas[index]
    return result.reshape(coordinate.shape + (LSF_OFFSET_BASIS_COUNT,))


def mspline_bin_integrals(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Analytically integrate every M-spline over paired intervals."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or np.any(upper < lower):
        raise ValueError("bin edges must be conformable and ordered")
    flat_lower, flat_upper = lower.ravel(), upper.ravel()
    result = np.empty((flat_lower.size, LSF_OFFSET_BASIS_COUNT))
    _, antiderivatives, areas = _offset_splines()
    for index, antiderivative in enumerate(antiderivatives):
        support_lower, support_upper = _OFFSET_KNOTS[index], _OFFSET_KNOTS[index + 4]
        lo = np.clip(flat_lower, support_lower, support_upper)
        hi = np.clip(flat_upper, support_lower, support_upper)
        result[:, index] = (antiderivative(hi) - antiderivative(lo)) / areas[index]
    result[np.abs(result) < 2.0e-15] = 0.0
    return np.maximum(result.reshape(lower.shape + (LSF_OFFSET_BASIS_COUNT,)), 0.0)


def native_pixel_edges(wave: np.ndarray) -> np.ndarray:
    """Return midpoint/extrapolated edges of an untouched native grid."""
    wave = np.asarray(wave, dtype=float)
    if (
        wave.ndim != 1
        or wave.size < 2
        or np.any(~np.isfinite(wave))
        or np.any(np.diff(wave) <= 0.0)
    ):
        raise ValueError("wave must be a strictly increasing one-dimensional grid")
    edges = np.empty(wave.size + 1)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return edges


def _integrated_components(
    wave: np.ndarray,
    line_wave: np.ndarray,
    output_mask: np.ndarray,
) -> sp.csc_matrix:
    """Exact native-bin density for every line/M-spline pair.

    Memoised on the exact bytes of its three arguments.  The telluric models
    are reconstructed once per fitted row while the native grid and the line
    catalog are fixed for the whole worker, so without this every row repaid
    the full M-spline bin integration; the row-dependent part of the model
    (the transmission) never enters here.  The returned matrix is shared
    between instances and must not be modified in place -- every caller only
    multiplies it.
    """
    wave = np.ascontiguousarray(wave, dtype=float)
    line_wave = np.ascontiguousarray(line_wave, dtype=float)
    output_mask = np.ascontiguousarray(output_mask, dtype=bool)
    return _integrated_components_cached(
        wave.tobytes(), line_wave.tobytes(), output_mask.tobytes()
    )


@lru_cache(maxsize=16)
def _integrated_components_cached(
    wave_bytes: bytes,
    line_wave_bytes: bytes,
    output_mask_bytes: bytes,
) -> sp.csc_matrix:
    wave = np.frombuffer(wave_bytes, dtype=float)
    line_wave = np.frombuffer(line_wave_bytes, dtype=float)
    output_mask = np.frombuffer(output_mask_bytes, dtype=bool)
    return _build_integrated_components(wave, line_wave, output_mask)


def _build_integrated_components(
    wave: np.ndarray,
    line_wave: np.ndarray,
    output_mask: np.ndarray,
) -> sp.csc_matrix:
    edges = native_pixel_edges(wave)
    widths = np.diff(edges)
    first = np.clip(
        np.searchsorted(edges, line_wave + LSF_OFFSET_LOWER_ANGSTROM, side="right") - 1,
        0,
        wave.size,
    )
    stop = np.clip(
        np.searchsorted(edges, line_wave + LSF_OFFSET_UPPER_ANGSTROM, side="left") + 1,
        0,
        wave.size,
    )
    count = np.maximum(stop - first, 0)
    line = np.repeat(np.arange(line_wave.size), count)
    if line.size == 0:
        return sp.csc_matrix((wave.size, line_wave.size * LSF_OFFSET_BASIS_COUNT))
    start = np.repeat(np.cumsum(count) - count, count)
    pixel = first[line] + np.arange(line.size) - start
    keep = output_mask[pixel]
    line, pixel = line[keep], pixel[keep]
    mass = mspline_bin_integrals(
        edges[pixel] - line_wave[line],
        edges[pixel + 1] - line_wave[line],
    ) / widths[pixel, None]
    pair, basis = np.nonzero(mass)
    return sp.coo_matrix(
        (
            mass[pair, basis],
            (pixel[pair], line[pair] * LSF_OFFSET_BASIS_COUNT + basis),
        ),
        shape=(wave.size, line_wave.size * LSF_OFFSET_BASIS_COUNT),
    ).tocsc()


def _line_design(
    components: sp.csc_matrix,
    line_flux: np.ndarray,
    wavelength_basis: np.ndarray,
) -> sp.csc_matrix:
    """Map tensor coefficients to the exact native-bin line model."""
    weighted = line_flux[:, None] * wavelength_basis
    line, wavelength_component = np.nonzero(weighted)
    offset_component = np.arange(LSF_OFFSET_BASIS_COUNT)
    transform = sp.coo_matrix(
        (
            np.repeat(weighted[line, wavelength_component], LSF_OFFSET_BASIS_COUNT),
            (
                (line[:, None] * LSF_OFFSET_BASIS_COUNT + offset_component).ravel(),
                (
                    offset_component[None, :] * wavelength_basis.shape[1]
                    + wavelength_component[:, None]
                ).ravel(),
            ),
        ),
        shape=(
            line_flux.size * LSF_OFFSET_BASIS_COUNT,
            LSF_OFFSET_BASIS_COUNT * wavelength_basis.shape[1],
        ),
    ).tocsc()
    return (components @ transform).tocsc()


def _component_masses(
    state: LSFSurfaceState,
    channel: str,
    wavelength: np.ndarray,
) -> np.ndarray:
    knots, degree = state.knot_vectors[channel], state.degrees[channel]
    coordinate = np.clip(wavelength, knots[degree], knots[-degree - 1])
    return evaluate_bspline_basis(coordinate, knots, degree) @ state.coefficients[channel].T


def evaluate_lsf_density(
    state: LSFSurfaceState,
    wavelength: np.ndarray,
    delta_wavelength: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted LSF density in inverse Angstrom."""
    if state.legacy_kernel_representation != LSF_SPLINE2D_REPRESENTATION:
        raise ValueError("state is not an lsf-spline2d state")
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    delta_wavelength = np.atleast_1d(np.asarray(delta_wavelength, dtype=float))
    result = np.zeros((wavelength.size, delta_wavelength.size))
    covered = np.zeros(wavelength.size, dtype=bool)
    offset_basis = mspline_basis(delta_wavelength)
    for channel, lower, upper in LSF_CHANNELS:
        mask = _channel_mask(wavelength, lower, upper)
        if np.any(mask):
            result[mask] = _component_masses(state, channel, wavelength[mask]) @ offset_basis.T
            covered |= mask
    if not np.all(covered):
        raise ValueError("state does not cover every requested wavelength")
    return result


def evaluate_lsf_diagnostics(
    state: LSFSurfaceState,
    wavelength: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return barycenter, correction, sigma, and literal FWHM in Angstrom."""
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    step = (_OFFSET_KNOTS[-1] - _OFFSET_KNOTS[0]) / (_OFFSET_KNOTS.size - 1)
    centers = 0.5 * (_OFFSET_KNOTS[:-4] + _OFFSET_KNOTS[4:])
    variance = step**2 / 3.0
    masses = np.zeros((wavelength.size, LSF_OFFSET_BASIS_COUNT))
    for channel, lower, upper in LSF_CHANNELS:
        mask = _channel_mask(wavelength, lower, upper)
        if np.any(mask):
            masses[mask] = _component_masses(state, channel, wavelength[mask])
    centroid = masses @ centers
    sigma = np.sqrt(np.maximum(masses @ (centers**2 + variance) - centroid**2, 0.0))
    delta = np.linspace(LSF_OFFSET_LOWER_ANGSTROM, LSF_OFFSET_UPPER_ANGSTROM, 1201)
    density = masses @ mspline_basis(delta).T
    fwhm = np.full(wavelength.size, np.nan)
    for index, profile in enumerate(density):
        peak = int(np.argmax(profile))
        half = 0.5 * profile[peak]
        left = np.flatnonzero(profile[: peak + 1] <= half)
        right = np.flatnonzero(profile[peak:] <= half)
        if left.size and right.size:
            lo, hi = int(left[-1]), int(peak + right[0])
            x0 = np.interp(half, profile[lo : lo + 2], delta[lo : lo + 2])
            x1 = np.interp(half, profile[hi - 1 : hi + 1][::-1], delta[hi - 1 : hi + 1][::-1])
            fwhm[index] = x1 - x0
    return {
        "centroid_angstrom": centroid,
        "wavelength_correction_angstrom": -centroid,
        "sigma_angstrom": sigma,
        "fwhm_angstrom": fwhm,
    }


class SkyDecompLSFSpline2D(SkyDecompLSFSurfaceIterative):
    """The iterative split-zodi model with a B-spline x M-spline LSF."""

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("split_zodi", True) is not True:
            raise ValueError("SkyDecompLSFSpline2D requires split_zodi=True")
        kwargs["split_zodi"] = True
        if kwargs.get("config") is None:
            kwargs["config"] = LSFSurfaceIterativeConfig(roughness_fraction=1.0e-4)
        self._line_components: dict[str, sp.csc_matrix] = {}
        self._line_indices: dict[str, np.ndarray] = {}
        self._continuum_components: dict[str, sp.csc_matrix] = {}
        super().__init__(*args, **kwargs)
        if np.any(np.diff(self.wave) <= 0.0):
            raise ValueError("SkyDecompLSFSpline2D requires a strictly increasing native grid")
        self._prepare_catalog()

    def _prepare_catalog(self) -> None:
        waves: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        self._group_slices: dict[str, slice] = {}
        group_offset = line_offset = 0
        self._o2_slice = slice(0, 0)
        catalogs = (
            ("oh", self._oh_line_groups),
            ("atom", self._atom_line_groups),
            ("orc", self._orc_line_groups),
            ("o2", [(self.lam_o2, np.ones_like(self.lam_o2))]),
        )
        for key, catalog in catalogs:
            self._group_slices[key] = slice(group_offset, group_offset + len(catalog))
            for local_group, (line_wave, line_weight) in enumerate(catalog):
                line_wave = np.asarray(line_wave, dtype=float)
                line_weight = np.asarray(line_weight, dtype=float)
                valid = np.isfinite(line_wave) & np.isfinite(line_weight) & (line_weight >= 0.0)
                waves.append(line_wave[valid])
                weights.append(line_weight[valid])
                groups.append(np.full(np.count_nonzero(valid), group_offset + local_group))
                if key == "o2":
                    self._o2_slice = slice(line_offset, line_offset + np.count_nonzero(valid))
                line_offset += np.count_nonzero(valid)
            group_offset += len(catalog)
        self._line_wave = np.concatenate(waves)
        self._base_line_weight = np.concatenate(weights)
        self._line_group = np.concatenate(groups).astype(int)
        self._n_groups = group_offset

        for channel, lower, upper in LSF_CHANNELS:
            output_mask = _channel_mask(self.wave, lower, upper)
            indices = np.flatnonzero(_channel_mask(self._line_wave, lower, upper))
            self._line_indices[channel] = indices
            self._line_components[channel] = _integrated_components(
                self.wave,
                self._line_wave[indices],
                output_mask,
            )
            source = np.flatnonzero(output_mask)
            self._continuum_components[channel] = _integrated_components(
                self.wave,
                self.wave[source],
                output_mask,
            )

    def _line_weights(self) -> np.ndarray:
        weights = self._base_line_weight.copy()
        relative = self.aij_o2 * self.gi_o2 * np.exp(
            -HC_OVER_KB_CMK * (self.ei_o2 - self.e0_o2) / self.t_o2
        )
        total = float(np.sum(relative))
        weights[self._o2_slice] = relative / total if total > 0.0 else 0.0
        return weights

    def _line_source(self, line_coefficient: np.ndarray) -> np.ndarray:
        self._active_line_coefficient = np.asarray(line_coefficient)
        return super()._line_source(line_coefficient)

    def _nominal_coefficients(self, channel_mask: np.ndarray) -> np.ndarray:
        sigma = (
            float(np.nanmedian(np.asarray(self.lsf_sigma)[channel_mask]))
            if np.ndim(self.lsf_sigma)
            else float(self.lsf_sigma)
        )
        centers = 0.5 * (_OFFSET_KNOTS[:-4] + _OFFSET_KNOTS[4:])
        coefficient = np.exp(-0.5 * (centers / max(sigma, 0.15)) ** 2)
        coefficient = 0.5 * (coefficient + coefficient[::-1])
        return coefficient / coefficient.sum()

    def _make_state(
        self,
        coefficients: dict[str, np.ndarray],
        knots: dict[str, np.ndarray],
        degrees: dict[str, int],
        metrics: dict[str, dict[str, object]],
        knot_strategy: str,
    ) -> LSFSurfaceState:
        wave_n, wave_min, wave_max, wave_hash = _wave_fingerprint(self.wave)
        return LSFSurfaceState(
            coefficients=coefficients,
            knot_vectors=knots,
            degrees=degrees,
            channel_bounds={name: (lower, upper) for name, lower, upper in LSF_CHANNELS},
            tap_offsets=np.arange(-5, 6),
            config=asdict(self.config),
            metrics=metrics,
            requested_cycles=self.config.n_refinement_cycles,
            completed_cycles=0,
            wave_n=wave_n,
            wave_min=wave_min,
            wave_max=wave_max,
            wave_sha256=wave_hash,
            knot_strategy=knot_strategy,
            legacy_kernel_representation=LSF_SPLINE2D_REPRESENTATION,
        )

    def _nominal_state(self, reason: str) -> LSFSurfaceState:
        coefficients, knots, degrees, metrics = {}, {}, {}, {}
        resolved = _resolved_spline_config(self.config, self.spline_config)
        for channel, lower, upper in LSF_CHANNELS:
            mask = _channel_mask(self.wave, lower, upper)
            fit_mask = mask & ((self.wave >= self.config.blue_fit_lower) if channel == "B" else True)
            spline = resolved.for_channel(channel)
            knot_vector = _configured_knot_vector(self.wave[fit_mask], spline)
            if knot_vector is None:
                _, knot_vector = build_bspline_basis(
                    self.wave[fit_mask], spline.n_basis, spline.degree, np.ones(fit_mask.sum())
                )
            coefficients[channel] = np.repeat(
                self._nominal_coefficients(mask)[:, None], spline.n_basis, axis=1
            )
            knots[channel], degrees[channel] = knot_vector, spline.degree
            metrics[channel] = {
                "status": "fallback",
                "reason": reason,
                "n_pixels": int(fit_mask.sum()),
                "fit_lower": float(self.wave[fit_mask][0]),
                "fit_upper": float(self.wave[fit_mask][-1]),
                "knots_fixed": True,
                "model": "constant_blue_mspline" if channel == "B" else "tensor_bspline_mspline",
            }
        state = self._make_state(coefficients, knots, degrees, metrics, "nominal_mspline")
        state.fit_status = f"failed:{reason}"
        state.failure_reason = reason
        return state

    def _failed_input_lsf_state(self, reason: str) -> LSFSurfaceState:
        return self._nominal_state(reason)

    def _build_continuum_operator(self, state: LSFSurfaceState) -> sp.csr_matrix:
        widths = np.diff(native_pixel_edges(self.wave))
        blocks = []
        for channel, lower, upper in LSF_CHANNELS:
            source = np.flatnonzero(_channel_mask(self.wave, lower, upper))
            masses = _component_masses(state, channel, self.wave[source])
            transform = sp.coo_matrix(
                (
                    (masses * widths[source, None]).ravel(),
                    (
                        np.arange(source.size * LSF_OFFSET_BASIS_COUNT),
                        np.repeat(np.arange(source.size), LSF_OFFSET_BASIS_COUNT),
                    ),
                ),
                shape=(source.size * LSF_OFFSET_BASIS_COUNT, source.size),
            ).tocsc()
            block = (self._continuum_components[channel] @ transform).tocoo()
            blocks.append((block.row, source[block.col], block.data))
        return sp.coo_matrix(
            (
                np.concatenate([block[2] for block in blocks]),
                (
                    np.concatenate([block[0] for block in blocks]),
                    np.concatenate([block[1] for block in blocks]),
                ),
            ),
            shape=(self.wave.size, self.wave.size),
        ).tocsr()

    def _set_lsf_state(self, state: LSFSurfaceState | None) -> None:
        self.lsf_surface_state = state
        if state is None:
            self._lsf_surface = self._lsf_operator = None
            self.lsf_metrics, self.lsf_kernels = {}, {}
            return
        self._lsf_surface = evaluate_lsf_density(state, self.wave, _DIAGNOSTIC_DELTA)
        self._lsf_operator = self._build_continuum_operator(state)
        self.lsf_metrics = state.metrics
        edges = np.linspace(LSF_OFFSET_LOWER_ANGSTROM, LSF_OFFSET_UPPER_ANGSTROM, 12)
        bins = mspline_bin_integrals(edges[:-1], edges[1:])
        self.lsf_kernels = {
            channel: bins
            @ np.median(
                _component_masses(
                    state,
                    channel,
                    self.wave[_channel_mask(self.wave, lower, upper)],
                ),
                axis=0,
            )
            for channel, lower, upper in LSF_CHANNELS
        }

    def _fit_lsf_channels(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        source: np.ndarray,
        fixed_background: np.ndarray,
    ) -> LSFSurfaceState:
        line_coefficient = self._active_line_coefficient
        if np.shape(line_coefficient) != (self._n_groups,):
            raise ValueError("line_coefficient does not match the retained line catalog")
        previous = self.lsf_surface_state
        resolved = _resolved_spline_config(self.config, self.spline_config)
        individual_flux = self._line_weights() * np.maximum(
            np.asarray(line_coefficient)[self._line_group], 0.0
        )
        target = np.asarray(flux) - np.asarray(fixed_background)
        coefficients, knots, degrees, metrics = {}, {}, {}, {}

        for channel, lower, upper in LSF_CHANNELS:
            channel_mask = _channel_mask(self.wave, lower, upper)
            fit_mask = channel_mask & (
                (self.wave >= self.config.blue_fit_lower) if channel == "B" else True
            )
            spline = resolved.for_channel(channel)
            previous_knots = None if previous is None else previous.knot_vectors[channel]
            knot_vector = (
                previous_knots
                if previous_knots is not None
                else _configured_knot_vector(self.wave[fit_mask], spline)
            )
            if knot_vector is None:
                _, knot_vector = build_bspline_basis(
                    self.wave[fit_mask], spline.n_basis, spline.degree, np.ones(fit_mask.sum())
                )
            line_indices = self._line_indices[channel]
            line_wave = self._line_wave[line_indices]
            coordinate = np.clip(
                line_wave,
                knot_vector[spline.degree],
                knot_vector[-spline.degree - 1],
            )
            exact_design = _line_design(
                self._line_components[channel],
                individual_flux[line_indices],
                evaluate_bspline_basis(coordinate, knot_vector, spline.degree),
            )[fit_mask].toarray()
            fallback = (
                self._nominal_coefficients(channel_mask)
                if previous is None
                else np.median(previous.coefficients[channel], axis=1)
            )
            _, coefficient, _, metric = _fit_bspline_channel(
                self.wave[fit_mask],
                np.asarray(source)[fit_mask],
                target[fit_mask],
                np.asarray(ivar)[fit_mask],
                fallback,
                n_basis=spline.n_basis,
                degree=spline.degree,
                roughness_fraction=self.config.roughness_fraction,
                offset_roughness_fraction=self.config.roughness_fraction,
                fallback_prior_fraction=self.config.fallback_prior_fraction,
                information_prior_max_boost=self.config.information_prior_max_boost,
                background_degree=self.config.background_degree,
                knot_vector=knot_vector,
                kernel_design=exact_design,
            )
            if previous is not None and metric["status"] == "fallback":
                coefficient = previous.coefficients[channel].copy()
                metric["reason"] = f"previous_surface:{metric['reason']}"
            metric.update(
                fit_lower=float(self.wave[fit_mask][0]),
                fit_upper=float(self.wave[fit_mask][-1]),
                knots_fixed=previous_knots is not None,
                model="constant_blue_mspline" if channel == "B" else "tensor_bspline_mspline",
            )
            coefficients[channel], knots[channel], degrees[channel], metrics[channel] = (
                coefficient,
                np.asarray(knot_vector).copy(),
                spline.degree,
                metric,
            )
        state = self._make_state(
            coefficients,
            knots,
            degrees,
            metrics,
            _spline_strategy_summary(resolved),
        )
        self._set_lsf_state(state)
        return state

    def _render_lines(self) -> dict[str, np.ndarray]:
        weights = self._line_weights()
        grouped = np.zeros((self._n_groups, self.wave.size))
        for channel, _, _ in LSF_CHANNELS:
            indices = self._line_indices[channel]
            masses = _component_masses(
                self.lsf_surface_state,
                channel,
                self._line_wave[indices],
            )
            transform = sp.coo_matrix(
                (
                    (masses * weights[indices, None]).ravel(),
                    (
                        np.arange(indices.size * LSF_OFFSET_BASIS_COUNT),
                        np.repeat(self._line_group[indices], LSF_OFFSET_BASIS_COUNT),
                    ),
                ),
                shape=(indices.size * LSF_OFFSET_BASIS_COUNT, self._n_groups),
            ).tocsc()
            grouped += (self._line_components[channel] @ transform).T.toarray()
        return {key: grouped[where] for key, where in self._group_slices.items()}

    def _assemble_refined_matrices(self) -> dict[str, np.ndarray]:
        lines = self._render_lines()
        zodi = self._convolve_matrix_channelwise(self.matrix_zodi_hr)
        return self._matrix_bundle(
            lines["oh"],
            self._convolve_matrix_channelwise(self.matrix_moon_hr),
            self.matrix_diffuse,
            lines["atom"],
            lines["orc"],
            lines["o2"],
            matrix_zodi=zodi,
        )

    def _finalize_result(self, *args, **kwargs):
        if self.lsf_surface_state is None:
            run = args[0] if args else kwargs["run"]
            self._set_lsf_state(self._nominal_state(run.failure or "no_completed_lsf_cycle"))
        return super()._finalize_result(*args, **kwargs)


__all__ = [
    "LSF_SPLINE2D_REPRESENTATION",
    "SkyDecompLSFSpline2D",
    "evaluate_lsf_density",
    "evaluate_lsf_diagnostics",
    "mspline_basis",
    "mspline_bin_integrals",
    "native_pixel_edges",
]
