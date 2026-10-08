"""Iterative continuum, line, and wavelength-dependent LSF decomposition.

The blue arm uses one shape-constrained kernel because it has relatively little
isolated line information.  Each red-arm kernel tap is a smooth B-spline of
wavelength.  The quadratic program keeps every evaluated kernel nonnegative,
normalized, and single-peaked at the central tap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
import tracemalloc
from pathlib import Path

import clarabel
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

from .result_io import (
    CHANNEL_NAMES,
    FAILED_INPUT_FIT_STATUS,
    METRIC_COLUMNS,
    STATE_CONFIG_COLUMNS,
)
from .fit import LSF_CHANNELS, LSF_KERNEL_SIZE, SkyDecomp, SkyDecompResult


LSF_STATE_SCHEMA_VERSION = 1
LSF_TAP_OFFSETS = np.arange(-(LSF_KERNEL_SIZE // 2), LSF_KERNEL_SIZE // 2 + 1)


@dataclass(frozen=True, slots=True)
class LSFSurfaceIterativeConfig:
    """Scientific controls for continuum, LSF, and line refinement.

    Fields are grouped below by stage. Wavelength values are in Angstrom;
    Huber thresholds are in robust-noise units; fractions are dimensionless.
    """

    # Continuum reweighting around sky lines and residual outliers.
    line_weight: float = 5.0e-4
    skyline_cumulative_fraction: float = 0.99
    skyline_half_width_angstrom: float = 2.0
    huber_transition_sigma: float = 3.0

    # Refinement count and smooth wavelength-dependent kernel representation.
    n_refinement_cycles: int = 5
    n_basis: int = 6
    degree: int = 3

    # Kernel priors and the low-order nuisance background used by each channel fit.
    roughness_fraction: float = 0.01
    fallback_prior_fraction: float = 1.0e-4
    information_prior_max_boost: float = 50.0
    background_degree: int = 3
    blue_fit_lower: float = 5500.0

    def __post_init__(self) -> None:
        for name in ("n_refinement_cycles", "n_basis", "degree", "background_degree"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"{name} must be an integer")
        if not 0.0 <= self.line_weight <= 1.0:
            raise ValueError("line_weight must lie in [0, 1]")
        if not 0.0 < self.skyline_cumulative_fraction <= 1.0:
            raise ValueError("skyline_cumulative_fraction must lie in (0, 1]")
        if self.skyline_half_width_angstrom <= 0.0:
            raise ValueError("skyline_half_width_angstrom must be positive")
        if self.huber_transition_sigma <= 0.0:
            raise ValueError("huber_transition_sigma must be positive")
        if self.n_refinement_cycles < 1:
            raise ValueError("n_refinement_cycles must be positive")
        if self.degree < 0:
            raise ValueError("degree must be nonnegative")
        if self.n_basis < self.degree + 1:
            raise ValueError("n_basis must be at least degree + 1")
        if self.roughness_fraction < 0.0:
            raise ValueError("roughness_fraction must be nonnegative")
        if self.fallback_prior_fraction < 0.0:
            raise ValueError("fallback_prior_fraction must be nonnegative")
        if self.information_prior_max_boost < 1.0:
            raise ValueError("information_prior_max_boost must be at least one")
        if self.background_degree < 0:
            raise ValueError("background_degree must be nonnegative")
        if not np.isfinite(self.blue_fit_lower):
            raise ValueError("blue_fit_lower must be finite")


@dataclass(frozen=True, slots=True)
class LSFChannelSplineConfig:
    """One channel's wavelength B-spline representation.

    ``information`` reproduces the existing first-cycle information-quantile
    knots. ``uniform`` spaces the interior knots uniformly in wavelength, and
    ``explicit`` uses the supplied interior wavelengths verbatim. The full
    clamped knot vector is persisted in :class:`LSFSurfaceState` in every case.
    """

    n_basis: int = 6
    degree: int = 3
    knot_strategy: str = "information"
    interior_knots: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("n_basis", "degree"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"{name} must be an integer")
        if self.degree < 0:
            raise ValueError("degree must be nonnegative")
        if self.n_basis < self.degree + 1:
            raise ValueError("n_basis must be at least degree + 1")
        if self.knot_strategy not in {"information", "uniform", "explicit"}:
            raise ValueError(
                "knot_strategy must be 'information', 'uniform', or 'explicit'"
            )
        knots = tuple(float(value) for value in self.interior_knots)
        object.__setattr__(self, "interior_knots", knots)
        if any(not np.isfinite(value) for value in knots):
            raise ValueError("interior_knots must be finite")
        if any(right <= left for left, right in zip(knots[:-1], knots[1:])):
            raise ValueError("interior_knots must be strictly increasing")
        expected = self.n_basis - self.degree - 1
        if self.knot_strategy == "explicit" and len(knots) != expected:
            raise ValueError(
                "explicit interior_knots must contain n_basis - degree - 1 values"
            )
        if self.knot_strategy != "explicit" and knots:
            raise ValueError("interior_knots are only valid with knot_strategy='explicit'")


@dataclass(frozen=True, slots=True)
class LSFSplineConfig:
    """Opt-in per-channel B-spline controls for the LSF surface.

    The defaults reproduce the historical constant B-channel kernel and the
    six-basis cubic information-adaptive R/Z surfaces. Omitting this object
    preserves the legacy ``LSFSurfaceIterativeConfig.n_basis`` and ``degree``
    path exactly.
    """

    b: LSFChannelSplineConfig = LSFChannelSplineConfig(
        n_basis=1,
        degree=0,
        knot_strategy="uniform",
    )
    r: LSFChannelSplineConfig = LSFChannelSplineConfig()
    z: LSFChannelSplineConfig = LSFChannelSplineConfig()

    def for_channel(self, channel: str) -> LSFChannelSplineConfig:
        try:
            return {"B": self.b, "R": self.r, "Z": self.z}[channel]
        except KeyError as error:
            raise ValueError(f"Unknown LSF channel: {channel}") from error


@dataclass(slots=True)
class LSFSurfaceState:
    """Compact, self-describing representation of one fitted LSF."""

    coefficients: dict[str, np.ndarray]
    knot_vectors: dict[str, np.ndarray]
    degrees: dict[str, int]
    channel_bounds: dict[str, tuple[float | None, float | None]]
    tap_offsets: np.ndarray
    config: dict[str, float | int]
    metrics: dict[str, dict[str, object]]
    requested_cycles: int
    completed_cycles: int
    wave_n: int
    wave_min: float
    wave_max: float
    wave_sha256: str
    fit_status: str = ""
    failure_reason: str = ""
    final_continuum_status: str = "not_run"
    final_line_status: str = "not_run"
    knot_strategy: str = "first_cycle_adaptive_then_fixed"
    legacy_kernel_representation: str = "native_grid_channel_median"
    schema_version: int = LSF_STATE_SCHEMA_VERSION


@dataclass(slots=True)
class LSFSurfaceIterativeResult(SkyDecompResult):
    """Baseline-compatible decomposition result with one additive LSF field."""

    lsf_state: LSFSurfaceState


@dataclass(slots=True)
class _FitRun:
    """Mutable model state and diagnostics for one iterative fit."""

    matrices: dict[str, np.ndarray]
    continuum_coefficient: np.ndarray
    line_coefficient: np.ndarray
    continuum_coef_err: np.ndarray
    line_coef_err: np.ndarray
    continuum: np.ndarray
    line_model: np.ndarray
    weights: np.ndarray
    channel_noise: dict[str, float]
    metrics: list[dict[str, object]]
    failure: str | None
    solver_status: str
    continuum_status: str = "not_run"
    line_status: str = "not_run"
    # Persisted covariance sub-blocks captured from the final continuum-stage
    # ``_fit_design`` return (2D arrays sized (n_moon, n_moon) and
    # (n_zodi, n_zodi)).  ``None`` when the block was not present in the
    # solve.  These are what get written to disk as ``COEF_COV_*`` HDUs.
    coef_cov_moon: np.ndarray | None = None
    coef_cov_zodi: np.ndarray | None = None


def _wave_fingerprint(wave: np.ndarray) -> tuple[int, float, float, str]:
    value = np.asarray(wave, dtype=np.float64)
    canonical = np.ascontiguousarray(value.astype("<f8", copy=False))
    return (
        int(value.size),
        float(value[0]),
        float(value[-1]),
        hashlib.sha256(canonical.tobytes()).hexdigest(),
    )


def _channel_mask(
    wave: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    mask = np.ones(wave.size, dtype=bool)
    if lower is not None:
        mask &= wave >= lower
    if upper is not None:
        mask &= wave < upper
    return mask


def build_skyline_mask(
    wave: np.ndarray,
    line_stick_matrices: tuple[np.ndarray, ...],
    *,
    cumulative_fraction: float = 0.99,
    half_width_angstrom: float = 2.0,
) -> np.ndarray:
    """Mask the strongest reference lines in every physical line group."""
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("wave must be a nonempty one-dimensional array")
    if np.any(np.diff(wave) <= 0.0):
        raise ValueError("wave must be strictly increasing")
    if not 0.0 < cumulative_fraction <= 1.0:
        raise ValueError("cumulative_fraction must lie in (0, 1]")
    if half_width_angstrom <= 0.0:
        raise ValueError("half_width_angstrom must be positive")

    centers = np.zeros(wave.size, dtype=bool)
    for matrix in line_stick_matrices:
        rows = np.atleast_2d(np.asarray(matrix, dtype=float))
        if rows.shape[1] != wave.size:
            raise ValueError("line stick matrices must match wave")
        for row in rows:
            strength = np.clip(np.nan_to_num(np.abs(row)), 0.0, np.inf)
            positive = np.flatnonzero(strength > 0.0)
            if positive.size == 0:
                continue
            order = positive[np.argsort(strength[positive])[::-1]]
            cumulative = np.cumsum(strength[order])
            stop = int(
                np.searchsorted(
                    cumulative,
                    cumulative_fraction * cumulative[-1],
                    side="left",
                )
            )
            centers[order[: stop + 1]] = True

    center_wave = wave[centers]
    if center_wave.size == 0:
        return centers
    lower = np.searchsorted(
        wave,
        center_wave - half_width_angstrom,
        side="left",
    )
    upper = np.searchsorted(
        wave,
        center_wave + half_width_angstrom,
        side="right",
    )
    difference = np.zeros(wave.size + 1, dtype=int)
    np.add.at(difference, lower, 1)
    np.add.at(difference, upper, -1)
    return np.cumsum(difference[:-1]) > 0


def _robust_mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def continuum_fit_weights(
    wave: np.ndarray,
    residual: np.ndarray,
    ivar: np.ndarray,
    skyline_mask: np.ndarray,
    *,
    line_weight: float = 5.0e-4,
    huber_transition_sigma: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return skyline-protected weights for a continuum-dominated solve.

    The Huber step decides which pixels are outliers, on the residual in
    NOISE units, ``|r| * sqrt(ivar)``. That has to match the weighting it
    feeds: the factor is multiplied INTO ``ivar``, so testing the RAW
    residual instead -- as this did before 2026-09-20 -- mixes an absolute
    outlier test with an inverse-variance weighting. The two agree only when
    ivar is constant across a channel, and under photon weighting ivar spans
    ~90x within one, so a bright pixel was being clipped as an outlier while
    sitting well inside its own, larger, noise.

    Weights are renormalised PER LSF CHANNEL, which makes every channel
    contribute equally regardless of its own noise. That is not a likelihood,
    but it protects the blue, where the moon/zodi colour separation lives:
    normalising once per row instead costs 13.5% of the blue chi2 on dark
    rows (measured 2026-09-20).
    """
    wave = np.asarray(wave, dtype=float)
    residual = np.asarray(residual, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    skyline_mask = np.asarray(skyline_mask, dtype=bool)
    if not (wave.shape == residual.shape == ivar.shape == skyline_mask.shape):
        raise ValueError("wave, residual, ivar, and skyline_mask must match")
    if not 0.0 <= line_weight <= 1.0:
        raise ValueError("line_weight must lie in [0, 1]")
    if huber_transition_sigma <= 0.0:
        raise ValueError("huber_transition_sigma must be positive")

    valid = np.isfinite(residual) & np.isfinite(ivar) & (ivar > 0.0) & ~skyline_mask
    # The quantity the outlier test is applied to. SIGNED, because
    # `_robust_mad` is taken about zero and MAD(|x|) != MAD(x); taking the
    # absolute value here would move the threshold as well as the units. In
    # noise units a correct noise model gives MAD ~ 1, but the MAD is still
    # taken rather than assumed, so a miscalibrated ivar cannot turn the
    # Huber step into a hard cut at k.
    score = residual * np.sqrt(ivar)
    score = np.where(np.isfinite(score), score, np.nan)
    global_sigma = _robust_mad(score[valid])
    if not np.isfinite(global_sigma) or global_sigma <= 0.0:
        global_sigma = 1.0

    multiplier = np.ones_like(residual, dtype=float)
    multiplier[skyline_mask] = line_weight
    channel_noise: dict[str, float] = {}
    for channel, lower, upper in LSF_CHANNELS:
        channel_use = valid & _channel_mask(wave, lower, upper)
        sigma = _robust_mad(score[channel_use])
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = global_sigma
        channel_noise[channel] = float(sigma)

        absolute = np.abs(score[channel_use])
        threshold = huber_transition_sigma * sigma
        huber = np.ones_like(absolute)
        outside = absolute > threshold
        huber[outside] = threshold / absolute[outside]
        multiplier[channel_use] *= huber

    weights = ivar * multiplier
    for _, lower, upper in LSF_CHANNELS:
        channel_use = np.isfinite(weights) & (weights > 0.0) & _channel_mask(wave, lower, upper)
        if np.any(channel_use):
            weights[channel_use] /= float(np.mean(weights[channel_use]))
    weights[~np.isfinite(weights) | (weights < 0.0)] = 0.0
    return weights, channel_noise


def _shifted_source(source: np.ndarray, kernel_size: int) -> np.ndarray:
    offsets = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    shifted = np.zeros((source.size, kernel_size), dtype=float)
    for column, offset in enumerate(offsets):
        if offset == 0:
            shifted[:, column] = source
        elif offset > 0:
            shifted[offset:, column] = source[:-offset]
        else:
            shifted[:offset, column] = source[-offset:]
    return shifted


def _information_adaptive_knots(
    wave: np.ndarray,
    information: np.ndarray,
    n_basis: int,
    degree: int,
) -> np.ndarray:
    n_interior = n_basis - degree - 1
    if n_interior <= 0:
        return np.empty(0, dtype=float)

    weight = np.clip(np.nan_to_num(information), 0.0, np.inf)
    if np.sum(weight) == 0.0:
        return np.linspace(wave[0], wave[-1], n_interior + 2)[1:-1]

    cumulative = np.cumsum(weight) / np.sum(weight)
    quantiles = np.arange(1, n_interior + 1) / (n_interior + 1)
    knots = np.interp(quantiles, cumulative, wave)
    spacing = (wave[-1] - wave[0]) * min(0.08, 0.5 / (n_interior + 1))
    lower = wave[0] + spacing * np.arange(1, n_interior + 1)
    upper = wave[-1] - spacing * np.arange(n_interior, 0, -1)
    knots = np.clip(knots, lower, upper)
    for index in range(1, n_interior):
        knots[index] = max(knots[index], knots[index - 1] + spacing)
    for index in range(n_interior - 2, -1, -1):
        knots[index] = min(knots[index], knots[index + 1] - spacing)
    return knots


def build_bspline_basis(
    wave: np.ndarray,
    n_basis: int,
    degree: int,
    information: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a clamped partition-of-unity basis on one channel."""
    interior = _information_adaptive_knots(
        wave,
        information,
        n_basis,
        degree,
    )
    knots = np.concatenate(
        [
            np.repeat(wave[0], degree + 1),
            interior,
            np.repeat(wave[-1], degree + 1),
        ]
    )
    basis = BSpline.design_matrix(wave, knots, degree).toarray()
    basis = np.maximum(basis, 0.0)
    basis /= np.sum(basis, axis=1, keepdims=True)
    return basis, knots


def _configured_knot_vector(
    wave: np.ndarray,
    spline: LSFChannelSplineConfig,
) -> np.ndarray | None:
    """Return a configured clamped knot vector, or ``None`` for information knots."""
    if spline.knot_strategy == "information":
        return None
    n_interior = spline.n_basis - spline.degree - 1
    if spline.knot_strategy == "uniform":
        interior = np.linspace(wave[0], wave[-1], n_interior + 2)[1:-1]
    else:
        interior = np.asarray(spline.interior_knots, dtype=float)
        if interior.size and (
            float(interior[0]) <= float(wave[0])
            or float(interior[-1]) >= float(wave[-1])
        ):
            raise ValueError(
                "explicit interior knots must lie strictly inside the channel fit domain"
            )
    return np.concatenate(
        [
            np.repeat(wave[0], spline.degree + 1),
            interior,
            np.repeat(wave[-1], spline.degree + 1),
        ]
    )


def _resolved_spline_config(
    config: LSFSurfaceIterativeConfig,
    spline_config: LSFSplineConfig | None,
) -> LSFSplineConfig:
    if spline_config is not None:
        if not isinstance(spline_config, LSFSplineConfig):
            raise TypeError("spline_config must be an LSFSplineConfig")
        return spline_config
    return LSFSplineConfig(
        b=LSFChannelSplineConfig(
            n_basis=1,
            degree=0,
            knot_strategy="information",
        ),
        r=LSFChannelSplineConfig(
            n_basis=config.n_basis,
            degree=config.degree,
            knot_strategy="information",
        ),
        z=LSFChannelSplineConfig(
            n_basis=config.n_basis,
            degree=config.degree,
            knot_strategy="information",
        ),
    )


def _spline_strategy_summary(spline_config: LSFSplineConfig) -> str:
    values = []
    for channel in ("B", "R", "Z"):
        spline = spline_config.for_channel(channel)
        values.append(
            f"{channel}={spline.knot_strategy}:n{spline.n_basis}:k{spline.degree}"
        )
    return "first_cycle_configured_then_fixed;" + ";".join(values)


def evaluate_bspline_basis(
    wave: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Evaluate a persisted clamped B-spline basis without moving its knots."""
    wave = np.asarray(wave, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("wave must be a nonempty one-dimensional array")
    if knots.ndim != 1:
        raise ValueError("knots must be one-dimensional")
    n_basis = knots.size - int(degree) - 1
    if n_basis < 1:
        raise ValueError("knots and degree do not define a B-spline basis")
    basis = BSpline.design_matrix(
        wave,
        knots,
        int(degree),
        extrapolate=False,
    ).toarray()
    basis = np.maximum(basis, 0.0)
    row_sum = np.sum(basis, axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("wave lies outside the persisted B-spline domain")
    return basis / row_sum


def _project_unimodal(kernel: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(kernel, dtype=float), 0.0, np.inf)
    center = value.size // 2
    value[: center + 1] = np.maximum.accumulate(value[: center + 1])
    value[center:] = np.maximum.accumulate(value[center:][::-1])[::-1]
    value /= np.sum(value)
    return value


def _sum_constraints(kernel_size: int, n_basis: int) -> sp.csc_matrix:
    rows = np.tile(np.arange(n_basis), kernel_size)
    columns = np.arange(kernel_size * n_basis)
    return sp.coo_matrix(
        (np.ones(columns.size), (rows, columns)),
        shape=(n_basis, columns.size),
    ).tocsc()


_CURVATURE_GRAM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _curvature_gram_cached(kernel_size: int, n_basis: int) -> np.ndarray:
    """Return ``(I_K ⊗ D²)ᵀ (I_K ⊗ D²)`` as a dense ``(K·n_basis, K·n_basis)`` matrix.

    Depends only on ``(kernel_size, n_basis)``; computed once per configuration.
    """
    key = (int(kernel_size), int(n_basis))
    cached = _CURVATURE_GRAM_CACHE.get(key)
    if cached is not None:
        return cached
    d2 = np.diff(np.eye(n_basis), n=2, axis=0)
    curvature = sp.kron(sp.eye(kernel_size), d2, format="csc")
    gram = (curvature.T @ curvature).toarray()
    _CURVATURE_GRAM_CACHE[key] = gram
    return gram


def _monotonicity_constraints(
    kernel_size: int,
    n_basis: int,
) -> sp.csc_matrix:
    center = kernel_size // 2
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    row = 0
    for basis_index in range(n_basis):
        for tap in range(center):
            rows.extend([row, row])
            columns.extend(
                [
                    tap * n_basis + basis_index,
                    (tap + 1) * n_basis + basis_index,
                ]
            )
            values.extend([1.0, -1.0])
            row += 1
        for tap in range(center, kernel_size - 1):
            rows.extend([row, row])
            columns.extend(
                [
                    (tap + 1) * n_basis + basis_index,
                    tap * n_basis + basis_index,
                ]
            )
            values.extend([1.0, -1.0])
            row += 1
    return sp.coo_matrix(
        (values, (rows, columns)),
        shape=((kernel_size - 1) * n_basis, kernel_size * n_basis),
    ).tocsc()


def kernel_moments(kernel_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return kernel centroid and standard deviation in pixel units."""
    offsets = np.arange(
        -(kernel_surface.shape[1] // 2),
        kernel_surface.shape[1] // 2 + 1,
        dtype=float,
    )
    center = kernel_surface @ offsets
    variance = np.sum(
        kernel_surface * (offsets[None, :] - center[:, None]) ** 2,
        axis=1,
    )
    return center, np.sqrt(np.maximum(variance, 0.0))


def _kernel_metric_summary(
    surface: np.ndarray,
    *,
    include_legacy_aliases: bool = True,
) -> dict[str, float]:
    center, sigma = kernel_moments(surface)
    summary = {
        "center_pix_min": float(np.min(center)),
        "center_pix_median": float(np.median(center)),
        "center_pix_max": float(np.max(center)),
        "sigma_pix_min": float(np.min(sigma)),
        "sigma_pix_median": float(np.median(sigma)),
        "sigma_pix_max": float(np.max(sigma)),
    }
    if include_legacy_aliases:
        summary["center_pix"] = summary["center_pix_median"]
        summary["sigma_pix"] = summary["sigma_pix_median"]
    return summary


def _channel_fallback(
    basis: np.ndarray,
    fallback_kernel: np.ndarray,
    knots: np.ndarray,
    reason: str,
    runtime: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    coefficient = np.repeat(
        fallback_kernel[:, None],
        basis.shape[1],
        axis=1,
    )
    surface = basis @ coefficient.T
    metrics: dict[str, float | str] = {
        "status": "fallback",
        "reason": reason,
        "n_basis": float(basis.shape[1]),
        "chi2_red": np.nan,
        "rms_resid": np.nan,
        **_kernel_metric_summary(surface),
        "runtime_sec": runtime,
    }
    return surface, coefficient, knots, metrics


def _kernel_constraint_system(
    kernel_size: int,
    n_basis: int,
    n_nuisance: int,
    free_amplitude: bool,
) -> tuple[sp.csc_matrix, np.ndarray, list[object]]:
    n_kernel = kernel_size * n_basis
    nonnegative = sp.hstack(
        [-sp.eye(n_kernel), sp.csc_matrix((n_kernel, n_nuisance))],
        format="csc",
    )
    monotonicity_kernel = _monotonicity_constraints(kernel_size, n_basis)
    monotonicity = sp.hstack(
        [monotonicity_kernel, sp.csc_matrix((monotonicity_kernel.shape[0], n_nuisance))],
        format="csc",
    )
    constraints = [nonnegative, monotonicity]
    rhs = [np.zeros(n_kernel), np.zeros(monotonicity.shape[0])]
    cones = [
        clarabel.NonnegativeConeT(n_kernel),
        clarabel.NonnegativeConeT(monotonicity.shape[0]),
    ]
    if not free_amplitude:
        equality = sp.hstack(
            [_sum_constraints(kernel_size, n_basis), sp.csc_matrix((n_basis, n_nuisance))],
            format="csc",
        )
        constraints.insert(0, equality)
        rhs.insert(0, np.ones(n_basis))
        cones.insert(0, clarabel.ZeroConeT(n_basis))
    return sp.vstack(constraints, format="csc"), np.concatenate(rhs), cones


def _fit_bspline_channel(
    wave: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    ivar: np.ndarray,
    fallback_kernel: np.ndarray,
    *,
    n_basis: int = 6,
    degree: int = 3,
    roughness_fraction: float = 0.01,
    fallback_prior_fraction: float = 1.0e-4,
    information_prior_max_boost: float = 50.0,
    background_degree: int = 3,
    free_amplitude: bool = False,
    knot_vector: np.ndarray | None = None,
    kernel_design: np.ndarray | None = None,
    offset_roughness_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Fit one normalized, central-peak B-spline kernel surface.

    The fitted kernel is nonnegative and unimodal at every wavelength. A
    normalized projected ``fallback_kernel`` is returned when the channel has
    insufficient information or the constrained solve fails.

    Returns
    -------
    surface, coefficients, knots, metrics
        Native-grid kernel rows, compact per-tap B-spline coefficients, the
        knot vector, and solver diagnostics.
    """
    wave = np.asarray(wave, dtype=float)
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    fallback = _project_unimodal(fallback_kernel)

    shifted = _shifted_source(source, fallback.size)
    information = np.clip(np.nan_to_num(ivar), 0.0, np.inf) * np.sum(
        shifted**2,
        axis=1,
    )
    if knot_vector is None:
        basis, knots = build_bspline_basis(wave, n_basis, degree, information)
    else:
        knots = np.asarray(knot_vector, dtype=float).copy()
        persisted_n_basis = knots.size - degree - 1
        if persisted_n_basis != n_basis:
            raise ValueError("persisted knot vector does not match n_basis and degree")
        basis = evaluate_bspline_basis(wave, knots, degree)
    if kernel_design is None:
        kernel_design = (shifted[:, :, None] * basis[:, None, :]).reshape(
            wave.size,
            fallback.size * n_basis,
        )
    else:
        kernel_design = np.asarray(kernel_design, dtype=float)
        if kernel_design.shape != (wave.size, fallback.size * n_basis):
            raise ValueError("kernel_design has an incompatible shape")

    midpoint = 0.5 * (wave[0] + wave[-1])
    half_width = max(0.5 * (wave[-1] - wave[0]), 1.0)
    scaled_wave = (wave - midpoint) / half_width
    target_scale = max(float(np.sqrt(np.nanmean(target**2))), 1.0e-30)
    background = (
        np.polynomial.legendre.legvander(scaled_wave, background_degree)
        * target_scale
    )
    design = np.column_stack([kernel_design, background])
    valid = (
        np.isfinite(target)
        & np.isfinite(ivar)
        & (ivar > 0.0)
        & np.all(np.isfinite(design), axis=1)
    )

    n_kernel = fallback.size * n_basis
    n_nuisance = background.shape[1]
    n_parameter = n_kernel + n_nuisance
    insufficient_information = np.count_nonzero(valid) <= n_parameter or not np.any(
        np.abs(kernel_design[valid]) > 0.0
    )
    if insufficient_information:
        return _channel_fallback(basis, fallback, knots, "insufficient_active_pixels")

    root_weight = np.sqrt(ivar[valid])
    weighted_design = design[valid] * root_weight[:, None]
    weighted_target = target[valid] * root_weight
    data_scale = max(float(np.sqrt(np.mean(weighted_target**2))), 1.0e-30)
    weighted_design /= data_scale
    weighted_target /= data_scale

    hessian = weighted_design.T @ weighted_design
    hessian_scale = max(float(np.trace(hessian)) / n_parameter, 1.0e-12)
    if n_basis >= 3 and roughness_fraction > 0.0:
        curvature_gram = _curvature_gram_cached(fallback.size, n_basis)
        hessian[:n_kernel, :n_kernel] += roughness_fraction * hessian_scale * curvature_gram
    if fallback.size >= 3 and offset_roughness_fraction > 0.0:
        d2_offset = np.diff(np.eye(fallback.size), n=2, axis=0)
        operator = np.kron(d2_offset, np.eye(n_basis))
        hessian[:n_kernel, :n_kernel] += (
            offset_roughness_fraction * hessian_scale * (operator.T @ operator)
        )

    basis_information = np.divide(
        basis.T @ information,
        np.sum(basis, axis=0),
        out=np.zeros(n_basis),
        where=np.sum(basis, axis=0) > 0.0,
    )
    relative_information = basis_information / max(
        float(np.max(basis_information)),
        1.0e-30,
    )
    prior_boost = (
        1.0
        + (information_prior_max_boost - 1.0)
        * (1.0 - np.clip(relative_information, 0.0, 1.0)) ** 2
    )
    prior_weight = np.tile(prior_boost, fallback.size)
    prior_scale = fallback_prior_fraction * hessian_scale
    hessian[:n_kernel, :n_kernel] += prior_scale * np.diag(prior_weight)
    hessian += 1.0e-10 * np.eye(n_parameter)

    fallback_vector = np.repeat(fallback[:, None], n_basis, axis=1).reshape(-1)
    linear = -(weighted_design.T @ weighted_target)
    linear[:n_kernel] -= prior_scale * prior_weight * fallback_vector

    constraints, rhs, cones = _kernel_constraint_system(
        fallback.size,
        n_basis,
        n_nuisance,
        free_amplitude,
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    started = time.perf_counter()
    solver = clarabel.DefaultSolver(
        sp.triu(sp.csc_matrix((hessian + hessian.T) * 0.5)).tocsc(),
        np.asarray(linear, dtype=np.float64),
        constraints,
        rhs,
        cones,
        settings,
    )
    solution = solver.solve()
    runtime = time.perf_counter() - started
    status = str(solution.status)
    vector = None if solution.x is None else np.asarray(solution.x, dtype=float)
    if (
        status not in {"Solved", "AlmostSolved"}
        or vector is None
        or not np.all(np.isfinite(vector))
    ):
        return _channel_fallback(basis, fallback, knots, "solver_failed", runtime)

    raw_coefficient = vector[:n_kernel].reshape(fallback.size, n_basis)
    raw_coefficient[np.abs(raw_coefficient) < 1.0e-12] = 0.0
    coefficient = np.maximum(raw_coefficient, 0.0)
    amplitude_scale = np.sum(coefficient, axis=0)
    if np.any(amplitude_scale <= 0.0):
        return _channel_fallback(basis, fallback, knots, "empty_kernel", runtime)
    coefficient /= amplitude_scale[None, :]
    surface = basis @ coefficient.T
    surface /= np.sum(surface, axis=1, keepdims=True)

    model = design @ vector
    residual = target[valid] - model[valid]
    chi2 = float(np.sum(residual**2 * ivar[valid]))
    metrics = {
        "status": status,
        "reason": "",
        "n_pixels": float(wave.size),
        "n_valid_pixels": float(np.count_nonzero(valid)),
        "n_basis": float(n_basis),
        "degree": float(degree),
        "chi2_red": chi2 / max(np.count_nonzero(valid) - n_parameter, 1),
        "rms_resid": float(np.sqrt(np.mean(residual**2))),
        **_kernel_metric_summary(surface),
        "fitted_amplitude_scale": float(np.median(amplitude_scale)),
        "runtime_sec": runtime,
    }
    return surface, coefficient, knots, metrics


def fit_bspline_channel(
    wave: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    ivar: np.ndarray,
    fallback_kernel: np.ndarray,
    *,
    n_basis: int = 6,
    degree: int = 3,
    roughness_fraction: float = 0.01,
    fallback_prior_fraction: float = 1.0e-4,
    information_prior_max_boost: float = 50.0,
    background_degree: int = 3,
    free_amplitude: bool = False,
    knot_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Fit one normalized, central-peak B-spline kernel surface."""
    return _fit_bspline_channel(
        wave,
        source,
        target,
        ivar,
        fallback_kernel,
        n_basis=n_basis,
        degree=degree,
        roughness_fraction=roughness_fraction,
        fallback_prior_fraction=fallback_prior_fraction,
        information_prior_max_boost=information_prior_max_boost,
        background_degree=background_degree,
        free_amplitude=free_amplitude,
        knot_vector=knot_vector,
    )


def fit_lsf_surface(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    source: np.ndarray,
    fixed_background: np.ndarray,
    fallback_kernels: dict[str, np.ndarray],
    config: LSFSurfaceIterativeConfig,
    *,
    previous_state: LSFSurfaceState | None = None,
    spline_config: LSFSplineConfig | None = None,
) -> LSFSurfaceState:
    """Fit the three-channel wavelength-dependent LSF.

    The blue channel uses one constant constrained kernel; red channels use
    smooth B-spline coefficients. When ``previous_state`` is supplied, its
    knots are fixed and any failed channel reuses the previous accepted surface.

    Returns
    -------
    LSFSurfaceState
        Compact coefficients, knots, channel diagnostics, and grid provenance.
    """
    wave = np.asarray(wave, dtype=float)
    target = np.asarray(flux, dtype=float) - np.asarray(
        fixed_background,
        dtype=float,
    )
    surface = np.zeros((wave.size, LSF_KERNEL_SIZE), dtype=float)
    previous_surface = (
        None if previous_state is None else evaluate_lsf_surface(previous_state, wave)
    )
    coefficients: dict[str, np.ndarray] = {}
    knots: dict[str, np.ndarray] = {}
    degrees: dict[str, int] = {}
    bounds: dict[str, tuple[float | None, float | None]] = {}
    metrics: dict[str, dict[str, object]] = {}
    resolved_splines = _resolved_spline_config(config, spline_config)

    for channel, lower, upper in LSF_CHANNELS:
        mask = _channel_mask(wave, lower, upper)
        if not np.any(mask):
            continue
        fit_mask = mask.copy()
        channel_spline = resolved_splines.for_channel(channel)
        n_basis = channel_spline.n_basis
        degree = channel_spline.degree
        free_amplitude = False
        if channel == "B":
            # The blue arm has too little isolated-line information for a smooth surface.
            fit_mask &= wave >= config.blue_fit_lower
            free_amplitude = True

        # Freeze the first accepted knots so later cycles change only coefficients.
        previous_knots = None
        if previous_state is not None:
            previous_knots = previous_state.knot_vectors.get(channel)
        configured_knots = (
            None
            if previous_knots is not None
            else _configured_knot_vector(wave[fit_mask], channel_spline)
        )

        local_surface, coefficient, knot_vector, channel_metrics = fit_bspline_channel(
            wave[fit_mask],
            source[fit_mask],
            target[fit_mask],
            ivar[fit_mask],
            fallback_kernels[channel],
            n_basis=n_basis,
            degree=degree,
            roughness_fraction=(0.0 if channel == "B" else config.roughness_fraction),
            fallback_prior_fraction=config.fallback_prior_fraction,
            information_prior_max_boost=(
                1.0 if channel == "B" else config.information_prior_max_boost
            ),
            background_degree=config.background_degree,
            free_amplitude=free_amplitude,
            knot_vector=(previous_knots if previous_knots is not None else configured_knots),
        )

        restored_previous = previous_state is not None and channel_metrics["status"] == "fallback"
        if restored_previous:
            local_surface = previous_surface[mask].copy()
            coefficient = previous_state.coefficients[channel].copy()
            knot_vector = previous_state.knot_vectors[channel].copy()
            channel_metrics["reason"] = f"previous_surface:{channel_metrics['reason']}"
        if channel == "B":
            surface[mask] = local_surface if restored_previous else local_surface[0]
            channel_metrics["model"] = "constant_blue_kernel"
        else:
            surface[mask] = local_surface
            channel_metrics["model"] = "cubic_bspline_kernel"
        if spline_config is not None:
            if channel == "B" and (n_basis != 1 or degree != 0):
                if not restored_previous:
                    surface[mask] = local_surface[0]
                    surface[fit_mask] = local_surface
                channel_metrics["model"] = "wavelength_dependent_blue_bspline_kernel"
            elif channel != "B":
                channel_metrics["model"] = "wavelength_dependent_bspline_kernel"
            channel_metrics["knot_strategy"] = channel_spline.knot_strategy
        channel_metrics["fit_lower"] = float(np.min(wave[fit_mask]))
        channel_metrics["fit_upper"] = float(np.max(wave[fit_mask]))
        channel_metrics["knots_fixed"] = previous_knots is not None
        coefficients[channel] = coefficient
        knots[channel] = knot_vector
        degrees[channel] = degree
        bounds[channel] = (lower, upper)
        metrics[channel] = channel_metrics

    row_sum = np.sum(surface, axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
        raise RuntimeError("LSF fit did not define a kernel at every wavelength")
    surface /= row_sum
    wave_n, wave_min, wave_max, wave_sha256 = _wave_fingerprint(wave)
    return LSFSurfaceState(
        coefficients=coefficients,
        knot_vectors=knots,
        degrees=degrees,
        channel_bounds=bounds,
        tap_offsets=LSF_TAP_OFFSETS.copy(),
        config=asdict(config),
        metrics=metrics,
        requested_cycles=config.n_refinement_cycles,
        completed_cycles=0,
        wave_n=wave_n,
        wave_min=wave_min,
        wave_max=wave_max,
        wave_sha256=wave_sha256,
        knot_strategy=(
            "first_cycle_adaptive_then_fixed"
            if spline_config is None
            else _spline_strategy_summary(resolved_splines)
        ),
    )


def build_lsf_operator(
    wave: np.ndarray,
    kernel_surface: np.ndarray,
    *,
    channel_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    tap_offsets: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Build a sparse channelwise LSF convolution operator ``L`` with ``output = source @ L.T``.

    ``L[i, j] = kernel_surface[i, tap]`` when ``tap_offsets[tap] == i - j`` and
    both ``i`` and ``j`` fall in the same channel; zero elsewhere.
    """
    wave = np.asarray(wave, dtype=float)
    kernel_surface = np.asarray(kernel_surface, dtype=float)
    if wave.ndim != 1 or wave.size == 0 or np.any(np.diff(wave) <= 0.0):
        raise ValueError("wave must be nonempty and strictly increasing")
    if (
        kernel_surface.ndim != 2
        or kernel_surface.shape[0] != wave.size
        or kernel_surface.shape[1] % 2 != 1
    ):
        raise ValueError("kernel_surface must have shape (wave, odd kernel size)")
    if np.any(~np.isfinite(kernel_surface)) or np.any(kernel_surface < -1.0e-12):
        raise ValueError("kernel_surface must be finite and nonnegative")
    if not np.allclose(np.sum(kernel_surface, axis=1), 1.0, atol=1.0e-10):
        raise ValueError("each LSF kernel must sum to one")
    if tap_offsets is None:
        offsets = np.arange(
            -(kernel_surface.shape[1] // 2),
            kernel_surface.shape[1] // 2 + 1,
        )
    else:
        offsets = np.asarray(tap_offsets)
        if (
            offsets.ndim != 1
            or offsets.size != kernel_surface.shape[1]
            or not np.all(np.isfinite(offsets))
            or not np.all(offsets == offsets.astype(int))
        ):
            raise ValueError("tap_offsets must be finite integer pixel offsets")
        offsets = offsets.astype(int)
    bounds = (
        {channel: (lower, upper) for channel, lower, upper in LSF_CHANNELS}
        if channel_bounds is None
        else channel_bounds
    )

    n = wave.size
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []
    for tap, offset in enumerate(offsets):
        o = int(offset)
        for channel in CHANNEL_NAMES:
            if channel not in bounds:
                continue
            lower, upper = bounds[channel]
            start = 0 if lower is None else int(np.searchsorted(wave, lower, side="left"))
            stop = n if upper is None else int(np.searchsorted(wave, upper, side="left"))
            if stop <= start:
                continue
            i_lo = start + max(0, o)
            i_hi = stop + min(0, o)
            if i_hi <= i_lo:
                continue
            i_idx = np.arange(i_lo, i_hi)
            rows_all.append(i_idx)
            cols_all.append(i_idx - o)
            data_all.append(kernel_surface[i_idx, tap])
    if not data_all:
        return sp.csr_matrix((n, n), dtype=float)
    return sp.csr_matrix(
        (
            np.concatenate(data_all),
            (np.concatenate(rows_all), np.concatenate(cols_all)),
        ),
        shape=(n, n),
    )


def apply_lsf_surface(
    wave: np.ndarray,
    values: np.ndarray,
    kernel_surface: np.ndarray,
    *,
    channel_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    tap_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a target-wavelength-dependent kernel without crossing channels."""
    source = np.asarray(values, dtype=float)
    one_dimensional = source.ndim == 1
    if one_dimensional:
        source = source[None, :]
    if source.ndim != 2 or source.shape[1] != np.asarray(wave).size:
        raise ValueError("values must end with the wavelength dimension")
    operator = build_lsf_operator(
        wave,
        kernel_surface,
        channel_bounds=channel_bounds,
        tap_offsets=tap_offsets,
    )
    output = (operator @ source.T).T
    return output[0] if one_dimensional else output


def evaluate_lsf_surface(
    state: LSFSurfaceState,
    wave: np.ndarray,
) -> np.ndarray:
    """Reconstruct a fitted LSF surface from persisted knots and coefficients."""
    if state.legacy_kernel_representation != "native_grid_channel_median":
        raise ValueError("state does not contain a discrete 11-tap LSF")
    if state.schema_version != LSF_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported LSF state schema version: {state.schema_version}")
    evaluation_wave = np.asarray(wave, dtype=float)
    if (
        evaluation_wave.ndim != 1
        or evaluation_wave.size == 0
        or np.any(np.diff(evaluation_wave) <= 0.0)
    ):
        raise ValueError("wave must be nonempty and strictly increasing")

    surface = np.zeros((evaluation_wave.size, state.tap_offsets.size), dtype=float)
    covered = np.zeros(evaluation_wave.size, dtype=bool)
    for channel in CHANNEL_NAMES:
        if channel not in state.channel_bounds:
            continue
        lower, upper = state.channel_bounds[channel]
        mask = _channel_mask(evaluation_wave, lower, upper)
        if not np.any(mask):
            continue
        if channel not in state.coefficients:
            raise ValueError(f"LSF state does not contain channel {channel}")
        coefficient = np.asarray(state.coefficients[channel], dtype=float)
        if (
            coefficient.ndim != 2
            or coefficient.shape[0] != state.tap_offsets.size
            or np.any(~np.isfinite(coefficient))
            or np.any(coefficient < -1.0e-12)
        ):
            raise ValueError(f"Invalid LSF coefficients for channel {channel}")
        degree = int(state.degrees[channel])
        if degree == 0 and coefficient.shape[1] == 1:
            surface[mask] = coefficient[:, 0]
            covered[mask] = True
            continue
        channel_wave = evaluation_wave[mask]
        knot_vector = np.asarray(state.knot_vectors[channel], dtype=float)
        domain_lower = float(knot_vector[degree])
        domain_upper = float(knot_vector[-degree - 1])
        outside_domain = (channel_wave < domain_lower) | (channel_wave > domain_upper)
        if np.any(outside_domain):
            if channel != "B":
                raise ValueError(
                    f"Evaluation grid lies outside the fitted {channel}-channel spline domain"
                )
            channel_wave = np.clip(channel_wave, domain_lower, domain_upper)
        basis = evaluate_bspline_basis(
            channel_wave,
            knot_vector,
            degree,
        )
        if basis.shape[1] != coefficient.shape[1]:
            raise ValueError(f"LSF coefficient and knot dimensions disagree for channel {channel}")
        surface[mask] = basis @ coefficient.T
        covered[mask] = True

    if not np.all(covered):
        raise ValueError("persisted LSF state does not cover the requested wave")
    if np.any(surface < -1.0e-10) or np.any(~np.isfinite(surface)):
        raise ValueError("persisted LSF state produces invalid kernel weights")
    surface = np.maximum(surface, 0.0)
    row_sum = np.sum(surface, axis=1, keepdims=True)
    if np.any(row_sum <= 0.0) or np.any(~np.isfinite(row_sum)):
        raise ValueError("persisted LSF state does not cover the requested wave")
    if not np.allclose(row_sum, 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("persisted LSF kernels are not normalized")
    return surface / row_sum


def evaluate_lsf_kernel(
    state: LSFSurfaceState,
    wavelength: float,
) -> np.ndarray:
    """Evaluate the persisted 11-tap LSF at one wavelength."""
    if not np.isfinite(wavelength):
        raise ValueError("wavelength must be finite")
    return evaluate_lsf_surface(state, np.array([float(wavelength)]))[0]


def apply_lsf_channelwise(
    state: LSFSurfaceState,
    wave: np.ndarray,
    values: np.ndarray,
    *,
    allow_different_grid: bool = False,
) -> np.ndarray:
    """Reconstruct and apply a persisted LSF state to one or more spectra."""
    evaluation_wave = np.asarray(wave, dtype=float)
    if not allow_different_grid:
        wave_n, wave_min, wave_max, wave_sha256 = _wave_fingerprint(evaluation_wave)
        if (
            wave_n != state.wave_n
            or wave_min != state.wave_min
            or wave_max != state.wave_max
            or wave_sha256 != state.wave_sha256
        ):
            raise ValueError(
                "wave does not match the fitted pixel grid; pass "
                "allow_different_grid=True only when pixel-offset semantics are intended"
            )
    source = np.asarray(values, dtype=float)
    if source.ndim < 1 or source.shape[-1] != evaluation_wave.size:
        raise ValueError("values must end with the wavelength dimension")
    original_shape = source.shape
    flattened = source.reshape(-1, evaluation_wave.size)
    applied = apply_lsf_surface(
        evaluation_wave,
        flattened,
        evaluate_lsf_surface(state, evaluation_wave),
        channel_bounds=state.channel_bounds,
        tap_offsets=state.tap_offsets,
    )
    return applied.reshape(original_shape)


def load_lsf_surface_state(
    filename: str | Path,
    spectrum_index: int = 0,
) -> LSFSurfaceState:
    """Load one compact LSF state through the shared result I/O layer."""
    from .result_io import load_lsf_surface_state as load_state

    return load_state(filename, spectrum_index=spectrum_index)


def _nominal_lsf_state(
    wave: np.ndarray,
    fallback_kernels: dict[str, np.ndarray],
    config: LSFSurfaceIterativeConfig,
    reason: str,
) -> LSFSurfaceState:
    coefficients: dict[str, np.ndarray] = {}
    knots: dict[str, np.ndarray] = {}
    degrees: dict[str, int] = {}
    bounds: dict[str, tuple[float | None, float | None]] = {}
    metrics: dict[str, dict[str, object]] = {}
    for channel, lower, upper in LSF_CHANNELS:
        mask = _channel_mask(wave, lower, upper)
        if not np.any(mask):
            continue
        kernel = _project_unimodal(fallback_kernels[channel])
        coefficients[channel] = kernel[:, None]
        knots[channel] = np.array([wave[mask][0], wave[mask][-1]], dtype=float)
        degrees[channel] = 0
        bounds[channel] = (lower, upper)
        metrics[channel] = {
            "status": "fallback",
            "reason": reason,
            "n_pixels": int(np.sum(mask)),
            "n_valid_pixels": 0,
            "chi2_red": np.nan,
            "rms_resid": np.nan,
            **_kernel_metric_summary(
                kernel[None, :],
                include_legacy_aliases=False,
            ),
            "runtime_sec": 0.0,
            "fit_lower": float(wave[mask][0]),
            "fit_upper": float(wave[mask][-1]),
            "knots_fixed": True,
            "model": "constant_nominal_fallback",
        }
    wave_n, wave_min, wave_max, wave_sha256 = _wave_fingerprint(wave)
    return LSFSurfaceState(
        coefficients=coefficients,
        knot_vectors=knots,
        degrees=degrees,
        channel_bounds=bounds,
        tap_offsets=LSF_TAP_OFFSETS.copy(),
        config=asdict(config),
        metrics=metrics,
        requested_cycles=config.n_refinement_cycles,
        completed_cycles=0,
        wave_n=wave_n,
        wave_min=wave_min,
        wave_max=wave_max,
        wave_sha256=wave_sha256,
        fit_status=f"failed:{reason}",
        failure_reason=reason,
        knot_strategy="nominal_constant_fallback",
    )


class SkyDecompLSFSurfaceIterative(SkyDecomp):
    """Iteratively refine continuum, smooth LSF, and emission-line amplitudes."""

    _LINE_KEYS = ("oh", "atom", "orc", "o2")
    _SOLVED = {"Solved", "AlmostSolved"}

    @property
    def _CONTINUUM_KEYS(self) -> tuple[str, ...]:
        """Continuum block order.  Grows to include ``zodi`` when split_zodi=True."""
        if getattr(self, "split_zodi", False):
            return ("moon", "zodi", "diffuse")
        return ("moon", "diffuse")

    def __init__(
        self,
        *args,
        config: LSFSurfaceIterativeConfig | None = None,
        spline_config: LSFSplineConfig | None = None,
        **kwargs,
    ) -> None:
        if float(kwargs.get("moon_interline_boost", 0.0)) != 0.0:
            raise ValueError("SkyDecompLSFSurfaceIterative requires moon_interline_boost=0")
        kwargs["moon_interline_boost"] = 0.0
        self.config = config or LSFSurfaceIterativeConfig()
        if spline_config is not None and not isinstance(spline_config, LSFSplineConfig):
            raise TypeError("spline_config must be an LSFSplineConfig")
        self.spline_config = spline_config
        self.lsf_surface_state: LSFSurfaceState | None = None
        self._lsf_surface: np.ndarray | None = None
        self._lsf_operator: sp.csr_matrix | None = None
        self.stage_metrics: tuple[dict[str, object], ...] = ()
        self.skyline_mask = np.array([], dtype=bool)
        self.continuum_weights = np.array([], dtype=float)
        self.channel_noise: dict[str, float] = {}
        self.final_continuum = np.array([], dtype=float)
        self.final_line_model = np.array([], dtype=float)
        super().__init__(*args, **kwargs)

    @property
    def lsf_surface(self) -> np.ndarray | None:
        return self._lsf_surface

    @staticmethod
    def _stack_matrices(
        matrices: dict[str, np.ndarray],
        keys: tuple[str, ...],
    ) -> np.ndarray:
        return np.vstack([matrices[key] for key in keys])

    def _line_stick_matrices(self) -> tuple[np.ndarray, ...]:
        return (
            self.matrix_oh_stick,
            self.matrix_atom_stick,
            self.matrix_orc_stick,
            self.matrix_o2_stick,
        )

    def _set_lsf_state(self, state: LSFSurfaceState | None) -> None:
        self.lsf_surface_state = state
        if state is None:
            self._lsf_surface = None
            self._lsf_operator = None
            self.lsf_metrics = {}
            self.lsf_kernels = {}
            return
        # The compact state is authoritative; the dense surface is a runtime cache.
        self._lsf_surface = evaluate_lsf_surface(state, self.wave)
        # Prebuild the sparse channelwise convolution operator once per surface update.
        self._lsf_operator = build_lsf_operator(
            self.wave,
            self._lsf_surface,
            channel_bounds=state.channel_bounds,
            tap_offsets=state.tap_offsets,
        )
        self.lsf_metrics = {
            channel: {
                **metric,
                "legacy_kernel_representation": state.legacy_kernel_representation,
            }
            for channel, metric in state.metrics.items()
        }
        self.lsf_kernels = {
            channel: np.median(
                self._lsf_surface[_channel_mask(self.wave, lower, upper)],
                axis=0,
            )
            for channel, lower, upper in LSF_CHANNELS
            if channel in state.channel_bounds
        }

    def _failed_input_lsf_state(self, reason: str) -> LSFSurfaceState:
        fallback = {
            channel: self._default_channel_kernel(
                _channel_mask(self.wave, lower, upper)
            )
            for channel, lower, upper in LSF_CHANNELS
        }
        return _nominal_lsf_state(self.wave, fallback, self.config, reason)

    def _fit_lsf_channels(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        source: np.ndarray,
        fixed_background: np.ndarray,
    ) -> LSFSurfaceState:
        previous = self.lsf_surface_state
        previous_surface = self._lsf_surface
        fallback = {}
        for channel, lower, upper in LSF_CHANNELS:
            mask = _channel_mask(self.wave, lower, upper)
            if previous is None:
                fallback[channel] = self._default_channel_kernel(mask)
            else:
                fallback[channel] = np.median(
                    previous_surface[mask],
                    axis=0,
                )
        state = fit_lsf_surface(
            self.wave,
            flux,
            ivar,
            source,
            fixed_background,
            fallback,
            self.config,
            previous_state=previous,
            spline_config=self.spline_config,
        )
        self._set_lsf_state(state)
        return state

    def _convolve_matrix_channelwise(self, matrix: np.ndarray) -> np.ndarray:
        if self._lsf_operator is None:
            return super()._convolve_matrix_channelwise(matrix)
        if matrix.size == 0:
            return matrix.copy()
        return (self._lsf_operator @ np.asarray(matrix, dtype=float).T).T

    def _combine_coefficients(
        self,
        matrices: dict[str, np.ndarray],
        continuum_coefficient: np.ndarray,
        line_coefficient: np.ndarray,
    ) -> np.ndarray:
        slices = self._component_slices(matrices)
        coefficient = np.zeros(slices["o2"].stop, dtype=float)
        offset = 0
        for key in self._CONTINUUM_KEYS:
            size = matrices[key].shape[0]
            coefficient[slices[key]] = continuum_coefficient[offset : offset + size]
            offset += size
        offset = 0
        for key in self._LINE_KEYS:
            size = matrices[key].shape[0]
            coefficient[slices[key]] = line_coefficient[offset : offset + size]
            offset += size
        return coefficient

    def _combine_coef_errors(
        self,
        matrices: dict[str, np.ndarray],
        continuum_coef_err: np.ndarray,
        line_coef_err: np.ndarray,
    ) -> np.ndarray:
        """Scatter continuum + line 1σ errors into the combined coefficient layout.

        Every accepted refinement cycle fits the continuum block
        (``_CONTINUUM_KEYS = ('moon', 'diffuse')``) and the line block
        (``_LINE_KEYS = ('oh', 'atom', 'orc', 'o2')``) via two independent
        ``_fit_design`` calls, each of which returns its own active-set
        covariance. Because the two solves are strictly sequential (the line
        stage sees the flux with the fresh continuum subtracted) their KKT
        Hessians share no rows and the cross-block covariance is not defined
        by the model. The reported per-coefficient standard errors therefore
        come from whichever solve produced the value: continuum entries from
        ``continuum_coef_err``, line entries from ``line_coef_err``. Missing
        elements are left as ``NaN`` — consistent with the boundary
        convention used by ``_coef_err_active_set``.

        This is the uncertainty-space analogue of ``_combine_coefficients``
        and uses the same ``_component_slices`` mapping so the two arrays
        always agree column-by-column with ``coef``.
        """
        slices = self._component_slices(matrices)
        coef_err = np.full(slices["o2"].stop, np.nan, dtype=float)
        offset = 0
        for key in self._CONTINUUM_KEYS:
            size = matrices[key].shape[0]
            if continuum_coef_err.size >= offset + size:
                coef_err[slices[key]] = continuum_coef_err[offset : offset + size]
            offset += size
        offset = 0
        for key in self._LINE_KEYS:
            size = matrices[key].shape[0]
            if line_coef_err.size >= offset + size:
                coef_err[slices[key]] = line_coef_err[offset : offset + size]
            offset += size
        return coef_err

    def _line_source(self, line_coefficient: np.ndarray) -> np.ndarray:
        return np.vstack(self._line_stick_matrices()).T @ line_coefficient

    def _continuum_from_components(
        self,
        components: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return the additive continuum represented by the public components."""
        total = components["moon"] + components["diffuse"]
        if "zodi" in components:
            total = total + components["zodi"]
        return total

    @staticmethod
    def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
        denominator = max(float(np.linalg.norm(old)), 1.0e-30)
        return float(np.linalg.norm(new - old) / denominator)

    @staticmethod
    def _metric(
        stage: str,
        status: str,
        flux: np.ndarray,
        model: np.ndarray,
        ivar: np.ndarray,
        skyline_mask: np.ndarray,
        **extra: object,
    ) -> dict[str, object]:
        residual = flux - model
        valid = np.isfinite(residual) & np.isfinite(ivar) & (ivar > 0.0)
        line_free = valid & ~skyline_mask
        skyline = valid & skyline_mask

        def rms(mask: np.ndarray) -> float:
            return float(np.sqrt(np.mean(residual[mask] ** 2))) if np.any(mask) else np.nan

        metric: dict[str, object] = {
            "stage": stage,
            "solver_status": status,
            "chi2_per_valid_pixel": float(
                np.sum(residual[valid] ** 2 * ivar[valid]) / max(int(np.sum(valid)), 1)
            ),
            "rms": rms(valid),
            "line_free_rms": rms(line_free),
            "line_free_mad": _robust_mad(residual[line_free]),
            "skyline_rms": rms(skyline),
            "skyline_mad": _robust_mad(residual[skyline]),
        }
        metric.update(extra)
        return metric

    def _fit_continuum_stage(
        self,
        run: _FitRun,
        flux: np.ndarray,
        ivar: np.ndarray,
        skyline_mask: np.ndarray,
    ) -> tuple[
        dict[str, object],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        dict[str, float],
    ]:
        weights, channel_noise = continuum_fit_weights(
            self.wave,
            flux - run.continuum - run.line_model,
            ivar,
            skyline_mask,
            line_weight=self.config.line_weight,
            huber_transition_sigma=self.config.huber_transition_sigma,
        )
        design = self._stack_matrices(run.matrices, self._CONTINUUM_KEYS)
        n_moon = run.matrices["moon"].shape[0]
        # Continuum stack order: moon [, zodi], diffuse.  Zodi occupies the block
        # immediately after moon when split_zodi is on.
        if getattr(self, "split_zodi", False) and "zodi" in run.matrices and run.matrices["zodi"].shape[0] > 0:
            n_zodi = run.matrices["zodi"].shape[0]
            zodi_slice_local = slice(n_moon, n_moon + n_zodi)
            diffuse_slice_local = slice(n_moon + n_zodi, design.shape[0])
        else:
            zodi_slice_local = None
            diffuse_slice_local = slice(n_moon, design.shape[0])
        # Install this iteration's OH amplitude for the moon-gated FeO bound.
        # OH is in the LINE model, which is held fixed for this solve, so its
        # band integral is a known constant here -- but it changes as the LSF
        # refines, so it has to be re-installed every time.
        if getattr(self, 'diffuse_oh_bound_dex', 0.0) > 0.0:
            _oh = run.matrices.get('oh')
            _lc = np.asarray(run.line_coefficient, dtype=float)
            _amp = None
            if _oh is not None and np.size(_oh) > 0 and _lc.size >= _oh.shape[0]:
                _amp = float(np.nansum(np.asarray(_oh, dtype=float).sum(axis=1)
                                       * _lc[:_oh.shape[0]]))
            self.set_diffuse_oh_reference(_amp)
        fit = self._fit_design(
            design,
            flux - run.line_model,
            weights,
            moon_slice=slice(0, n_moon),
            diffuse_slice=diffuse_slice_local,
            zodi_slice=zodi_slice_local,
        )
        if str(fit["status"]) not in self._SOLVED:
            return fit, None, None, None, weights, channel_noise
        coefficient = np.asarray(fit["coef"], dtype=float)
        coef_err = np.asarray(
            fit.get("coef_err", np.full_like(coefficient, np.nan)),
            dtype=float,
        )
        # Diagnostic only: the UNCLAMPED target RMS this solve saw. The moon
        # and zodi curvature penalties act with strength lambda * data_scale**2
        # (see `_solve_nonnegative_weighted`), so this is what says how hard
        # the nominal lambda actually pulled on this row. Overwritten each
        # refinement cycle, so it ends up describing the solve that produced
        # the continuum that was kept.
        self._data_scale_cont = float(fit.get("data_scale_raw", np.nan))
        continuum = design.T @ coefficient
        return fit, coefficient, coef_err, continuum, weights, channel_noise

    def _run_iterations(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
    ) -> tuple[dict[str, object], np.ndarray, _FitRun]:
        """Run the nominal seed, refinement transactions, and final closure."""
        # Nominal joint seed.
        split_zodi = getattr(self, "split_zodi", False)
        matrices = self._matrix_bundle(
            self.matrix_oh,
            self.matrix_moon,
            self.matrix_diffuse,
            self.matrix_atom,
            self.matrix_orc,
            self.matrix_o2,
            matrix_zodi=self.matrix_zodi if split_zodi else None,
        )
        full_keys = (("oh", "moon", "zodi", "diffuse", "atom", "orc", "o2")
                     if split_zodi else ("oh", "moon", "diffuse", "atom", "orc", "o2"))
        full_design = self._stack_matrices(matrices, full_keys)
        slices = self._component_slices(matrices)
        seed = self._fit_design(
            full_design,
            flux,
            ivar,
            moon_slice=slices["moon"],
            diffuse_slice=slices["diffuse"],
            zodi_slice=slices.get("zodi"),
        )
        self._data_scale_seed = float(seed.get("data_scale_raw", np.nan))
        seed_coefficient = np.asarray(seed["coef"], dtype=float)
        seed_coef_err = np.asarray(
            seed.get("coef_err", np.full_like(seed_coefficient, np.nan)),
            dtype=float,
        )
        continuum_coefficient = np.concatenate(
            [seed_coefficient[slices[key]] for key in self._CONTINUUM_KEYS]
        )
        line_coefficient = np.concatenate(
            [seed_coefficient[slices[key]] for key in self._LINE_KEYS]
        )
        continuum_coef_err = np.concatenate(
            [seed_coef_err[slices[key]] for key in self._CONTINUUM_KEYS]
        )
        line_coef_err = np.concatenate(
            [seed_coef_err[slices[key]] for key in self._LINE_KEYS]
        )
        continuum_design = self._stack_matrices(matrices, self._CONTINUUM_KEYS)
        line_design = self._stack_matrices(matrices, self._LINE_KEYS)
        continuum = continuum_design.T @ continuum_coefficient
        line_model = line_design.T @ line_coefficient
        skyline_mask = build_skyline_mask(
            self.wave,
            self._line_stick_matrices(),
            cumulative_fraction=self.config.skyline_cumulative_fraction,
            half_width_angstrom=self.config.skyline_half_width_angstrom,
        )
        seed_status = str(seed["status"])
        seed_solved = seed_status in self._SOLVED
        run = _FitRun(
            matrices=matrices,
            continuum_coefficient=continuum_coefficient,
            line_coefficient=line_coefficient,
            continuum_coef_err=continuum_coef_err,
            line_coef_err=line_coef_err,
            continuum=continuum,
            line_model=line_model,
            weights=ivar.copy(),
            channel_noise={channel: np.nan for channel, _, _ in LSF_CHANNELS},
            metrics=[
                self._metric(
                    "00_nominal_joint_seed",
                    seed_status,
                    flux,
                    continuum + line_model,
                    ivar,
                    skyline_mask,
                )
            ],
            failure=None if seed_solved else f"nominal_seed:{seed_status}",
            solver_status=seed_status,
        )

        # Each cycle commits only after continuum, LSF, and line stages succeed.
        for cycle in range(1, self.config.n_refinement_cycles + 1):
            if run.failure is not None:
                break
            previous_state = self.lsf_surface_state
            previous_surface = (
                None if self._lsf_surface is None else self._lsf_surface.copy()
            )
            continuum_fit, candidate_coefficient, candidate_coef_err, candidate_continuum, weights, noise = (
                self._fit_continuum_stage(run, flux, ivar, skyline_mask)
            )
            run.weights = weights
            run.channel_noise = noise
            continuum_status = str(continuum_fit["status"])
            if continuum_status not in self._SOLVED:
                run.metrics.append(
                    self._metric(
                        f"{cycle:02d}_continuum_failed",
                        continuum_status,
                        flux,
                        run.continuum + run.line_model,
                        ivar,
                        skyline_mask,
                    )
                )
                run.failure = f"cycle_{cycle}_continuum:{continuum_status}"
                break

            run.metrics.append(
                self._metric(
                    f"{cycle:02d}_continuum",
                    continuum_status,
                    flux,
                    candidate_continuum + run.line_model,
                    ivar,
                    skyline_mask,
                    continuum_relative_change=self._relative_change(
                        candidate_continuum,
                        run.continuum,
                    ),
                )
            )
            state = self._fit_lsf_channels(
                flux,
                ivar,
                self._line_source(run.line_coefficient),
                candidate_continuum,
            )
            matrices = self._assemble_refined_matrices()
            line_design = self._stack_matrices(matrices, self._LINE_KEYS)
            line_fit = self._fit_design(
                line_design,
                flux - candidate_continuum,
                ivar,
            )
            line_status = str(line_fit["status"])
            if line_status not in self._SOLVED:
                self._set_lsf_state(previous_state)
                run.metrics.append(
                    self._metric(
                        f"{cycle:02d}_lines_failed",
                        line_status,
                        flux,
                        run.continuum + run.line_model,
                        ivar,
                        skyline_mask,
                    )
                )
                run.failure = f"cycle_{cycle}_lines:{line_status}"
                break

            candidate_line_coefficient = np.asarray(line_fit["coef"], dtype=float)
            candidate_line_coef_err = np.asarray(
                line_fit.get("coef_err", np.full_like(candidate_line_coefficient, np.nan)),
                dtype=float,
            )
            candidate_line_model = line_design.T @ candidate_line_coefficient
            state.completed_cycles = cycle
            lsf_change = (
                np.nan
                if previous_state is None
                else self._relative_change(self._lsf_surface, previous_surface)
            )
            run.metrics.append(
                self._metric(
                    f"{cycle:02d}_lsf_lines",
                    line_status,
                    flux,
                    candidate_continuum + candidate_line_model,
                    ivar,
                    skyline_mask,
                    line_relative_change=self._relative_change(
                        candidate_line_model,
                        run.line_model,
                    ),
                    lsf_relative_change=lsf_change,
                    lsf_status={
                        channel: metric["status"]
                        for channel, metric in state.metrics.items()
                    },
                )
            )
            run.matrices = matrices
            run.continuum_coefficient = candidate_coefficient
            run.continuum_coef_err = candidate_coef_err
            run.coef_cov_moon = continuum_fit.get("coef_cov_moon")
            run.coef_cov_zodi = continuum_fit.get("coef_cov_zodi")
            run.line_coefficient = candidate_line_coefficient
            run.line_coef_err = candidate_line_coef_err
            run.continuum = candidate_continuum
            run.line_model = candidate_line_model
            run.solver_status = line_status

        # Closure uses the last fully committed model, even after a failed cycle.
        if seed_solved:
            continuum_fit, candidate_coefficient, candidate_coef_err, candidate_continuum, weights, noise = (
                self._fit_continuum_stage(run, flux, ivar, skyline_mask)
            )
            run.weights = weights
            run.channel_noise = noise
            run.continuum_status = str(continuum_fit["status"])
            if run.continuum_status not in self._SOLVED:
                final_failure = f"final_continuum:{run.continuum_status}"
                run.failure = (
                    final_failure
                    if run.failure is None
                    else f"{run.failure};{final_failure}"
                )
            else:
                run.metrics.append(
                    self._metric(
                        "final_continuum",
                        run.continuum_status,
                        flux,
                        candidate_continuum + run.line_model,
                        ivar,
                        skyline_mask,
                        continuum_relative_change=self._relative_change(
                            candidate_continuum,
                            run.continuum,
                        ),
                    )
                )
                line_design = self._stack_matrices(run.matrices, self._LINE_KEYS)
                line_fit = self._fit_design(
                    line_design,
                    flux - candidate_continuum,
                    ivar,
                )
                run.line_status = str(line_fit["status"])
                if run.line_status not in self._SOLVED:
                    final_failure = f"final_lines:{run.line_status}"
                    run.failure = (
                        final_failure
                        if run.failure is None
                        else f"{run.failure};{final_failure}"
                    )
                else:
                    candidate_line_coefficient = np.asarray(
                        line_fit["coef"],
                        dtype=float,
                    )
                    candidate_line_coef_err = np.asarray(
                        line_fit.get(
                            "coef_err",
                            np.full_like(candidate_line_coefficient, np.nan),
                        ),
                        dtype=float,
                    )
                    candidate_line_model = line_design.T @ candidate_line_coefficient
                    run.metrics.append(
                        self._metric(
                            "final_lines",
                            run.line_status,
                            flux,
                            candidate_continuum + candidate_line_model,
                            ivar,
                            skyline_mask,
                            line_relative_change=self._relative_change(
                                candidate_line_model,
                                run.line_model,
                            ),
                        )
                    )
                    run.continuum_coefficient = candidate_coefficient
                    run.continuum_coef_err = candidate_coef_err
                    run.coef_cov_moon = continuum_fit.get("coef_cov_moon")
                    run.coef_cov_zodi = continuum_fit.get("coef_cov_zodi")
                    run.line_coefficient = candidate_line_coefficient
                    run.line_coef_err = candidate_line_coef_err
                    run.continuum = candidate_continuum
                    run.line_model = candidate_line_model
                    run.solver_status = run.line_status

        return seed, skyline_mask, run

    def _finalize_result(
        self,
        run: _FitRun,
        seed: dict[str, object],
        flux: np.ndarray,
        ivar: np.ndarray,
        skyline_mask: np.ndarray,
        started: float,
        trace_started: bool,
        verbose: bool,
    ) -> LSFSurfaceIterativeResult:
        """Build the public result and publish the final instance state."""
        coefficient = self._combine_coefficients(
            run.matrices,
            run.continuum_coefficient,
            run.line_coefficient,
        )
        coefficient_err = self._combine_coef_errors(
            run.matrices,
            run.continuum_coef_err,
            run.line_coef_err,
        )
        components = self._components_from_coef(coefficient, run.matrices)
        # `run.matrices['o2']` is a (1, n_wave) block already convolved by the
        # fitted LSF surface; keep `vector_o2` in step so what we persist
        # matches the O2 basis used to build `components['o2']`.
        if run.matrices["o2"].shape[0] == 1:
            self.vector_o2 = run.matrices["o2"][0].copy()
        # Per-pixel 1σ of the LSF-refined reconstruction: LSF-aware Jacobian
        # propagation using the SAME matrix bundle as `_components_from_coef`
        # so mean and σ share an identical basis.
        _sigma_comps = self._components_sigma_from_coef_err(coefficient_err, run.matrices)
        bestfit_lsf_sigma = np.sqrt(
            _sigma_comps["oh"] ** 2
            + _sigma_comps["moon"] ** 2
            + _sigma_comps["diffuse"] ** 2
            + _sigma_comps["atom"] ** 2
            + _sigma_comps["orc"] ** 2
            + _sigma_comps["o2"] ** 2
            + (_sigma_comps["zodi"] ** 2 if "zodi" in _sigma_comps else 0.0)
        )
        continuum = self._continuum_from_components(components)
        line_model = (
            components["oh"]
            + components["atom"]
            + components["orc"]
            + components["o2"]
        )
        bestfit_lsf = continuum + line_model

        residual = flux - bestfit_lsf
        valid = np.isfinite(residual) & np.isfinite(ivar) & (ivar > 0.0)
        chi2 = float(np.sum(residual[valid] ** 2 * ivar[valid]))
        reduced_chi2 = chi2 / max(int(np.sum(valid)) - coefficient.size, 1)
        rms_resid = (
            float(np.sqrt(np.mean(residual[valid] ** 2)))
            if np.any(valid)
            else np.nan
        )
        y_mean = (
            float(np.average(flux[valid], weights=ivar[valid]))
            if np.any(valid)
            else np.nan
        )
        total_sum = (
            float(np.sum((flux[valid] - y_mean) ** 2 * ivar[valid]))
            if np.any(valid)
            else np.nan
        )
        r2 = (
            1.0 - chi2 / total_sum
            if np.isfinite(total_sum) and total_sum > 0.0
            else np.nan
        )
        elapsed = time.perf_counter() - started
        peak_memory_mb = tracemalloc.get_traced_memory()[1] / 1024**2
        if trace_started:
            tracemalloc.stop()

        self.coef = coefficient
        self.coef_err = coefficient_err
        # Keep the historical seed-valued fields; bestfit_lsf is the refined model.
        self.bestfit = np.asarray(seed["bestfit"], dtype=float)
        self.bestfit_lsf = bestfit_lsf
        self.fit_status = run.solver_status if run.failure is None else f"failed:{run.failure}"
        self.r2 = r2
        self.rms_resid = rms_resid
        self.peak_memory_mb = peak_memory_mb
        self.stage_metrics = tuple(run.metrics)
        self.skyline_mask = skyline_mask
        self.continuum_weights = run.weights
        self.channel_noise = run.channel_noise
        self.final_continuum = continuum
        self.final_line_model = line_model
        if self.lsf_surface_state is None:
            fallback = {
                channel: self._default_channel_kernel(
                    _channel_mask(self.wave, lower, upper)
                )
                for channel, lower, upper in LSF_CHANNELS
            }
            self._set_lsf_state(
                _nominal_lsf_state(
                    self.wave,
                    fallback,
                    self.config,
                    run.failure or "no_completed_lsf_cycle",
                )
            )
        completed_cycles = self.lsf_surface_state.completed_cycles
        self.lsf_surface_state.fit_status = self.fit_status
        self.lsf_surface_state.failure_reason = run.failure or ""
        self.lsf_surface_state.final_continuum_status = run.continuum_status
        self.lsf_surface_state.final_line_status = run.line_status
        self.fit_summary = (
            f"status={self.fit_status} | npar={coefficient.size} | "
            f"ngood={int(np.sum(valid))} | chi2_red={reduced_chi2:.4g} | "
            f"R2={r2:.5f} | line_weight={self.config.line_weight:.3g} | "
            f"skyline_fraction={float(np.mean(skyline_mask)):.3f} | "
            f"refinement_cycles={completed_cycles}/{self.config.n_refinement_cycles} | "
            f"fixed_lsf_knots=True | dt={elapsed:.2f}s"
        )

        if verbose:
            for metric in run.metrics:
                print(
                    f"{metric['stage']}: chi2/pix={metric['chi2_per_valid_pixel']:.5g}, "
                    f"line-free RMS={metric['line_free_rms']:.5g}, "
                    f"skyline RMS={metric['skyline_rms']:.5g}"
                )

        return LSFSurfaceIterativeResult(
            coef=self.coef,
            coef_err=self.coef_err,
            bestfit=np.asarray(seed["bestfit"], dtype=float),
            resid=np.asarray(seed["resid"], dtype=float),
            resid_level=float(seed["resid_level"]),
            fit_status=self.fit_status,
            fit_summary=self.fit_summary,
            reduced_chi2=reduced_chi2,
            fit_elapsed_sec=elapsed,
            components=components,
            design_names=self.design_names,
            t_o2=self.t_o2,
            t_o2_err=self.t_o2_err,
            r2=r2,
            rms_resid=rms_resid,
            peak_memory_mb=peak_memory_mb,
            o2_fit_status=self.o2_fit_status,
            o2_fit_summary=self.o2_fit_summary,
            o2_fit_elapsed_sec=self.o2_fit_elapsed_sec,
            o2_valid_frac=self.o2_valid_frac,
            lsf_kernels=self.lsf_kernels,
            lsf_metrics=self.lsf_metrics,
            bestfit_lsf=bestfit_lsf,
            moon_knots=self.moon_knots_used.copy(),
            moon_boosted_pixels=self.moon_boosted_pixels_used.copy(),
            vector_o2=self.vector_o2.copy(),
            o2_prefit_amp=float(self.o2_prefit_amp),
            bestfit_lsf_sigma=bestfit_lsf_sigma,
            lsf_state=self.lsf_surface_state,
            coef_cov_moon=run.coef_cov_moon,
            coef_cov_zodi=run.coef_cov_zodi,
        )

    def failed_input_result(self, reason: str) -> LSFSurfaceIterativeResult:
        """Return a same-schema NaN row for one known invalid input."""
        reason = str(reason).strip()
        if not reason:
            raise ValueError("failed input reason must not be empty")

        state = self._failed_input_lsf_state(reason)
        for channel, coefficient in state.coefficients.items():
            state.coefficients[channel] = np.full_like(coefficient, np.nan)
            state.metrics[channel].update(
                status="not_run_failed_input",
                reason=reason,
            )
        state.fit_status = FAILED_INPUT_FIT_STATUS
        state.failure_reason = reason
        state.final_continuum_status = "not_run_failed_input"
        state.final_line_status = "not_run_failed_input"
        self._set_lsf_state(None)
        self.lsf_surface_state = state
        self.lsf_metrics = state.metrics

        nan_wave = np.full(self.wave.shape, np.nan, dtype=np.float64)
        nan_coefficient = np.full(len(self.design_names), np.nan, dtype=np.float64)
        matrices = self._matrix_bundle(
            self.matrix_oh,
            self.matrix_moon,
            self.matrix_diffuse,
            self.matrix_atom,
            self.matrix_orc,
            self.matrix_o2,
            matrix_zodi=self.matrix_zodi if self.split_zodi else None,
        )
        components = self._components_from_coef(nan_coefficient, matrices)
        self.lsf_kernels = {
            channel: np.full(state.tap_offsets.size, np.nan, dtype=np.float64)
            for channel in state.coefficients
        }
        summary = f"status={FAILED_INPUT_FIT_STATUS} | reason={reason}"
        result = LSFSurfaceIterativeResult(
            coef=nan_coefficient.copy(),
            coef_err=nan_coefficient.copy(),
            bestfit=nan_wave.copy(),
            resid=nan_wave.copy(),
            resid_level=np.nan,
            fit_status=FAILED_INPUT_FIT_STATUS,
            fit_summary=summary,
            reduced_chi2=np.nan,
            fit_elapsed_sec=np.nan,
            components=components,
            design_names=list(self.design_names),
            t_o2=np.nan,
            t_o2_err=np.nan,
            r2=np.nan,
            rms_resid=np.nan,
            peak_memory_mb=np.nan,
            o2_fit_status="not_run_failed_input",
            o2_fit_summary=reason,
            o2_fit_elapsed_sec=np.nan,
            o2_valid_frac=np.nan,
            lsf_kernels=self.lsf_kernels,
            lsf_metrics=self.lsf_metrics,
            bestfit_lsf=nan_wave.copy(),
            moon_knots=self.moon_knots_used.copy(),
            moon_boosted_pixels=np.array([], dtype=np.float64),
            vector_o2=nan_wave.copy(),
            o2_prefit_amp=np.nan,
            bestfit_lsf_sigma=nan_wave.copy(),
            zodi_names=list(self.zodi_names),
            zodi_knots=self.zodi_knots_used.copy(),
            lsf_state=state,
        )
        self.fit_status = result.fit_status
        self.fit_summary = result.fit_summary
        return result

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
    ) -> LSFSurfaceIterativeResult:
        """Run the approved continuum -> LSF -> lines refinement sequence."""
        flux = np.asarray(flux, dtype=float)
        ivar = np.asarray(ivar, dtype=float)
        if flux.shape != self.wave.shape or ivar.shape != self.wave.shape:
            raise ValueError("flux and ivar must match wave")

        trace_started = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            trace_started = True
        tracemalloc.reset_peak()
        started = time.perf_counter()
        self._set_lsf_state(None)
        self._prefit_o2(flux, ivar)
        seed, skyline_mask, run = self._run_iterations(flux, ivar)

        return self._finalize_result(
            run,
            seed,
            flux,
            ivar,
            skyline_mask,
            started,
            trace_started,
            verbose,
        )


__all__ = [
    "LSFChannelSplineConfig",
    "LSFSplineConfig",
    "LSFSurfaceIterativeConfig",
    "LSFSurfaceIterativeResult",
    "LSFSurfaceState",
    "SkyDecompLSFSurfaceIterative",
    "apply_lsf_channelwise",
    "apply_lsf_surface",
    "build_bspline_basis",
    "build_lsf_operator",
    "build_skyline_mask",
    "continuum_fit_weights",
    "evaluate_bspline_basis",
    "evaluate_lsf_kernel",
    "evaluate_lsf_surface",
    "fit_bspline_channel",
    "fit_lsf_surface",
    "kernel_moments",
    "load_lsf_surface_state",
]
