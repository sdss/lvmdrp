"""LVM sky-decomposition weighted-NNLS fit engine.

`SkyDecomp` builds a non-negative additive spectral model of one arm's flux
against a physically motivated design matrix and solves a curvature-penalised
quadratic program (QP) via Clarabel.  Public families in the default fit are:

    OH lines  +  Moon_bs (continuum spline * solar)  +  diffuse (HO2 + FeO + O2Ac)
    +  atomic airglow lines  +  ORC  +  O2

Two subclasses in this package add per-row iterative LSF refinement:

    `SkyDecompLSFSurfaceIterative` — smooth wavelength-dependent LSF surface,
        used for the p40_p70 corpus decompositions.
    `SkyDecompMoonZodiLSFSurfaceIterative` — same wrapper but with a fully
        physical Moon+Zodi predictor replacing the color-envelope spline.

`decompose_parallel.py` runs any of these at corpus scale via
`--fit-model {baseline, lsf-surface-iterative, lsf-surface-iterative-split-zodi,
moon-zodi-lsf-surface-iterative}`.

split_zodi mode
---------------

`SkyDecomp(split_zodi=True)` and the same flag on `SkyDecompLSFSurfaceIterative`
promote the single continuum spline `Moon_bs` into two color-tagged families:

    Moon_bs = solar_rb(λ) * moon_albedo(λ) * B_K_moon(λ) * c_moon[k]
    Zodi_bs = solar_rb(λ) * zodi_color(λ) * B_K_zodi(λ) * c_zodi[k]

where

    solar_rb(λ)     — LATMOS solar SED rebinned and LSF-convolved onto the LVM
                      wavelength grid (already used by the baseline `Moon_bs`).
    moon_albedo(λ)  — ROLO lunar disk-integrated albedo at a fiducial phase
                      angle (default 30°), median-normalised to unity.
                      Loaded from `moon_albedo_asset_path`, default
                      `<base_dir>/moon_zodi/eso_skycalc_rolo_moon_albedo.dat`.
    zodi_color(λ)   — Leinert-style reddening `(λ/5000Å)^zodi_color_exponent`,
                      default exponent 0.26 (matches `moon_zodi_model.py`),
                      median-normalised to unity.
    B_K(λ)          — cubic B-spline design on `K` uniformly-spaced interior
                      knots.  Default `n_zodi_spline_knots=3` (the zodi spline
                      is intentionally much smoother than the moon spline).

Both families remain non-negative in the QP.  The two color envelopes are
what makes them distinguishable: moon_albedo carries sharp ROLO mineral
bands while zodi_color is a smooth power-law reddening.

Curvature penalties
~~~~~~~~~~~~~~~~~~~
Each family carries a second-difference smoothness penalty:

    moon: ‖D² c_moon‖² * moon_smooth_lambda    (default 0.1)
    zodi: ‖D² c_zodi‖² * zodi_smooth_lambda    (default = 10 * moon)

Bump `zodi_smooth_lambda` (~1e-1) when using a small zodi knot count (K≤3);
a tighter zodi penalty makes the fitted zodi cleanly follow the red-slope
`zodi_color` envelope instead of picking up locally blue features.

Recommended settings
~~~~~~~~~~~~~~~~~~~~
Validated on the p40_p70 every10 corpus (100-row phase-stratified sample):

    n_spline_knots        = 29       (unchanged Moon_bs count)
    n_zodi_spline_knots   = 3
    moon_smooth_lambda    = 1e-3     (baseline value; kept for continuity)
    zodi_smooth_lambda    = 1e-1
    moon_albedo_fiducial_phase_deg = 30.0
    zodi_color_exponent   = 0.26

Backwards compatibility
~~~~~~~~~~~~~~~~~~~~~~~
`split_zodi=False` is the default and produces the exact same design matrix,
components, and solver output as prior versions of this module.  When
`split_zodi=True` the result gains a `"zodi"` component (and matching entry
in `_components_sigma_from_coef_err`); `SkyDecompResult` grows two kw-only
fields `zodi_names` and `zodi_knots`.  83 pre-existing tests pass unchanged.

Documentation and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The design, identifiability analysis, and end-to-end validation on 100
phase-stratified rows live in
`notebooks/moon_zodi_split_identifiability.ipynb`.  Corpus-scale usage:

    python skysub/decompose_parallel.py <every10.fits> \\
        --fit-model lsf-surface-iterative-split-zodi \\
        --n-zodi-spline-knots 3 --zodi-smooth-lambda 0.1 \\
        --n-refinement-cycles 5 --n-workers 8 --output-dir <out>

produces `..._decomp_{sci,sky1,sky2}_lsf_surface_iterative_split_zodi.fits`
each carrying a new `COMP_ZODI` HDU alongside the standard component HDUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import inspect
import pathlib
import re
import time
import tracemalloc

import clarabel
import numpy as np
import scipy.sparse as sp
from astropy.table import Table
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from lvmdrp.core.fluxcal import rebin_and_convolve

from .moon_zodi_model import (
    DEFAULT_PALACE_DIFFUSE_SUFFIX,
    DEFAULT_PALACE_OH_SUFFIX,
)

CAP_WAVE = 5.0
HC_OVER_KB_CMK = 1.4387769
OH_GROUP_KEYS = ("v_upper", "N_upper", "F_upper")
O2_MIN = 8600.0
O2_MAX = 8696.0
T_O2_REF = 191.5
T_O2_HALF_RANGE = 20.0
O2_MIN_VALID_FRAC = 0.5

# Smallest target RMS treated as real when the curvature penalties are divided
# by data_scale**2.  Purely a divide-by-zero guard for an all-zero target:
# anything above it is a legitimately faint row and is normalised, not floored.
# Deliberately far below the 1.0 the old clamp used, which was large enough to
# pin most real rows.
_DATA_SCALE_MIN = 1e-30
SUFFIXED_PMD_TABLES = frozenset({"pmd_popmodel_OH.dat", "pmd_refcont.dat"})
LSF_KERNEL_SIZE = 11
LSF_MIN_VALID_FRAC = 0.5
LSF_CHANNELS = (
    ("B", None, 5787.0),
    ("R", 5787.0, 7454.0),
    ("Z", 7454.0, None),
)

# Production split Moon/Zodiacal-light continuum defaults.  These are the
# calibrated settings from the moon-zodi-shape-priors branch; keeping them next
# to the SkyDecomp continuum parameters makes the direct class API and the
# corpus CLI use one ordinary, physically named configuration.
SPLIT_ZODI_CONTINUUM_DEFAULTS = {
    "n_spline_knots": 11,
    "moon_smooth_lambda": 1.0e-1,
    "split_zodi": True,
    "n_zodi_spline_knots": 1,
    "zodi_smooth_lambda": 1.0e-1,
    "moon_albedo_fiducial_phase_deg": 30.0,
    "zodi_color_exponent": 0.26,
    "moon_ratio_bound": 0.7,
    "zodi_ratio_bound": 0.7,
    "amp_prior_tol": 3.0,
    "amp_prior_floor": 0.02,
    "zodi_amp_bound": 2.0,
    "diffuse_ratio_bound_dex": 0.2,
    "diffuse_ratio_nominal": (0.0396, 0.7026, 0.2578),
    "diffuse_oh_centre_log10": -0.6489,
    "diffuse_oh_bound_dex": 0.15,
    "diffuse_oh_gate_frac": 0.6,
    "diffuse_oh_relax_dex": 0.0,
    "diffuse_oh_scope": "block",
    "moon_interline_boost": 0.0,
}


def vac_to_air(lam_vac_a: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam_vac_a, float)
    s2 = (1e4 / lam) ** 2
    n = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
    return lam / n


def decode_hitran_id(table: Table) -> Table:
    ids = np.asarray(table["ID"].astype(str), dtype=str)
    lengths = np.char.str_len(ids)
    if np.any(lengths != 13):
        index = int(np.flatnonzero(lengths != 13)[0])
        raise ValueError(
            f"HITRAN ID must contain exactly 13 characters: row={index}, "
            f"ID={ids[index]!r}, length={int(lengths[index])}"
        )

    v_up = np.where(np.array([s[4:5] for s in ids]) == "X", "10", np.array([s[4:5] for s in ids])).astype(int)
    v_low = np.array([s[5:6] for s in ids], dtype=int)
    branch_n = np.array([s[6:7] for s in ids])
    branch_j = np.array([s[7:8] for s in ids])
    f_up = np.array([s[8:9] for s in ids], dtype=int)
    f_low = np.array([s[9:10] for s in ids], dtype=int)
    n_up = np.array([s[10:12] for s in ids], dtype=int)
    parity = np.array([s[12:13] for s in ids])

    delta_map = {"O": -2, "P": -1, "Q": 0, "R": 1, "S": 2}
    invalid_branches = sorted(set(branch_n) - set(delta_map))
    if invalid_branches:
        raise ValueError(f"Unsupported HITRAN rotational branches: {invalid_branches}")
    delta_n = np.array([delta_map[branch] for branch in branch_n], dtype=int)
    n_low = n_up - delta_n
    invalid_levels = (n_up < 0) | (n_low < 0)
    if np.any(invalid_levels):
        index = int(np.flatnonzero(invalid_levels)[0])
        raise ValueError(
            f"HITRAN ID yields a negative rotational level: row={index}, "
            f"ID={ids[index]!r}, N_upper={n_up[index]}, N_lower={n_low[index]}"
        )

    decoded_reference_columns = {
        "vi": v_up,
        "Fi": f_up,
        "Ni": n_up,
        "pi": parity,
    }
    for column, decoded in decoded_reference_columns.items():
        if column not in table.colnames:
            continue
        if column == "pi":
            reference = np.char.strip(
                np.asarray(table[column].astype(str), dtype=str)
            )
        else:
            reference = np.asarray(table[column], dtype=int)
        mismatch = reference != decoded
        if np.any(mismatch):
            index = int(np.flatnonzero(mismatch)[0])
            raise ValueError(
                f"HITRAN ID disagrees with PALACE column {column}: row={index}, "
                f"ID={ids[index]!r}, decoded={decoded[index]!r}, "
                f"reference={reference[index]!r}"
            )

    table["v_upper"] = v_up
    table["v_lower"] = v_low
    table["N_upper"] = n_up
    table["N_lower"] = n_low
    table["F_upper"] = f_up
    table["F_lower"] = f_low
    table["branch_N"] = branch_n
    table["branch_J"] = branch_j
    table["parity"] = parity
    table["is_main"] = (f_up == f_low) & np.isin(branch_n, ["P", "Q", "R"])
    return table


@lru_cache(maxsize=None)
def _cached_ascii_table(path: str, fmt: str) -> Table:
    if fmt == "ascii.basic":
        return Table.read(
            path, format=fmt, guess=False, comment="#", fast_reader=False
        )
    return Table.read(path, format=fmt)


def read_static_table(path: str | Path, fmt: str = "ascii.basic") -> Table:
    """Parse a static PMD/PALACE ascii asset once per process.

    The telluric models are constructed once per fitted row, and re-parsing
    these tables was a fifth of that construction.  The cache holds the
    pristine parse and every caller gets its own copy, because the builders
    add and overwrite columns on the table they are handed.
    """
    return _cached_ascii_table(str(path), fmt).copy()


@lru_cache(maxsize=None)
def read_static_matrix(path: str, comments: str) -> np.ndarray:
    """Load a static whitespace-delimited asset once per process.

    The returned array is shared and therefore read-only; copy before any
    in-place arithmetic.
    """
    matrix = np.loadtxt(path, comments=comments)
    matrix.setflags(write=False)
    return matrix


def _solar_templates(
    path: str | Path, wave: np.ndarray, lsf_sigma: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the native-grid solar SED, interpolated and LSF-rebinned.

    Memoised on the asset, the grid and the LSF width, none of which vary
    between rows; the rebin-and-convolve behind it is otherwise repaid on
    every per-row model construction.  Callers get their own copies, so the
    cached templates can never be modified through them.
    """
    grid = np.ascontiguousarray(wave, dtype=float)
    sigma = np.ascontiguousarray(lsf_sigma, dtype=float)
    solar_hr, solar_rb = _solar_templates_cached(
        str(path), grid.tobytes(), sigma.tobytes(), sigma.shape
    )
    return solar_hr.copy(), solar_rb.copy()


@lru_cache(maxsize=8)
def _solar_templates_cached(
    path: str,
    wave_bytes: bytes,
    sigma_bytes: bytes,
    sigma_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.frombuffer(wave_bytes, dtype=float)
    lsf_sigma = np.frombuffer(sigma_bytes, dtype=float).reshape(sigma_shape)
    sol = read_static_matrix(path, ";")
    solar_wave = vac_to_air(sol[:, 0] * 10.0)
    # `sol` is the shared read-only parse; copy before normalising in place.
    solar_flux = np.array(sol[:, 1], dtype=float)
    solar_flux /= np.nanmedian(solar_flux)
    solar_hr = np.nan_to_num(
        np.interp(wave, solar_wave, solar_flux, left=0.0, right=0.0)
    )
    solar_rb = rebin_and_convolve(
        wave,
        solar_wave,
        solar_flux,
        lsf_sigma * 2.355,
        lsf_in_wavelength=True,
    )
    solar_rb /= np.nanmedian(solar_rb)
    return solar_hr, solar_rb


def _frozen_group(*fields):
    """Freeze one cached line group: arrays are shared, so they stay read-only."""
    for field in fields:
        if isinstance(field, np.ndarray):
            field.setflags(write=False)
    return fields


@lru_cache(maxsize=None)
def _oh_line_catalog(
    path: str,
    wave_lo: float,
    wave_hi: float,
    group_keys: tuple[str, ...],
    amplitude,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Tie the OH population model into its fitted line groups, once per process.

    Reading, HITRAN-decoding and grouping this table is the bulk of what is
    left of model construction, and it depends on nothing that varies from row
    to row -- only the asset, the grid span, the grouping keys and the class's
    own amplitude rule, all of which are in the cache key.  The telluric
    models are rebuilt per fitted row, so this is paid thousands of times a
    worker otherwise.
    """
    oh = read_static_table(path)
    oh["wave"] = vac_to_air(np.asarray(oh["lam"], float) * 1e4)
    mask = (oh["wave"] >= wave_lo - CAP_WAVE) & (oh["wave"] <= wave_hi + CAP_WAVE)
    oh = decode_hitran_id(oh[mask])
    return tuple(
        _frozen_group(
            np.asarray(group["wave"], float),
            np.asarray(amplitude(group), float),
        )
        for group in oh.group_by(list(group_keys)).groups
    )


@lru_cache(maxsize=None)
def _atom_line_catalog(
    path: str, wave_lo: float, wave_hi: float
) -> tuple[tuple[np.ndarray, np.ndarray, str], ...]:
    """Group the atomic airglow lines by species, once per process."""
    atom = read_static_table(path)
    atom["wave"] = vac_to_air(np.asarray(atom["lam"], float) * 1e4)
    mask = (atom["wave"] >= wave_lo - CAP_WAVE) & (atom["wave"] <= wave_hi + CAP_WAVE)
    atom = atom[mask]
    atom = atom[~np.isin(np.asarray(atom["class"], str), ["H", "Orc"])]
    catalog = []
    for group in atom.group_by("class").groups:
        amp = np.array(group["I"], dtype=float)
        amp /= np.nansum(amp)
        catalog.append(
            _frozen_group(
                np.asarray(group["wave"], float), amp, f"ATOM_{group['class'][0]}"
            )
        )
    return tuple(catalog)


@lru_cache(maxsize=None)
def _orc_line_catalog(
    path: str, wave_lo: float, wave_hi: float
) -> tuple[tuple[np.ndarray, np.ndarray, str], ...]:
    """Group the O2 Herzberg pseudo-continuum features, once per process."""
    orc = read_static_table(path)
    orc["wave"] = vac_to_air(np.asarray(orc["lam"], float) * 1e4)
    mask = (orc["wave"] >= wave_lo - CAP_WAVE) & (orc["wave"] <= wave_hi + CAP_WAVE)
    return tuple(
        _frozen_group(
            np.asarray(group["wave"], float),
            np.asarray(group["I"], float),
            f"ATOM_Orc_{group['reffeat'][0]}",
        )
        for group in orc[mask].group_by("reffeat").groups
    )


GRP2VECTOR_SIGMA_CUTOFF = 12.0


def grp2vector(
    line_wave: np.ndarray,
    line_amp: np.ndarray,
    wave: np.ndarray,
    lsf_sigma: np.ndarray | float,
) -> np.ndarray:
    """Sum Gaussian line profiles of one group onto the native grid.

    Each line is evaluated only on the pixels within
    ``GRP2VECTOR_SIGMA_CUTOFF`` sigma of its centre instead of on the whole
    grid.  At 12 sigma the Gaussian is exp(-72) ~ 5e-32 of its peak, i.e. some
    16 orders of magnitude below the float64 resolution of the pixel it would
    be added to, so the windowed sum is the full sum to the last bit while the
    cost falls from O(n_wave * n_lines) to O(window * n_lines).  This is the
    single hottest routine in model construction, and the telluric models
    rebuild their line matrices per row, so the window matters there twice.
    """
    cent = np.asarray(line_wave, float)
    amp = np.asarray(line_amp, float)
    wave = np.asarray(wave, float)
    out = np.zeros(wave.size, dtype=float)
    if cent.size == 0 or wave.size == 0:
        return out

    sig = (
        np.interp(cent, wave, lsf_sigma)
        if np.ndim(lsf_sigma) > 0
        else np.full(cent.size, float(lsf_sigma))
    )
    reach = GRP2VECTOR_SIGMA_CUTOFF * sig
    first = np.searchsorted(wave, cent - reach, side="left")
    stop = np.searchsorted(wave, cent + reach, side="right")
    # A non-positive sigma has no profile at all; the dense form used to make
    # it inf/nan, which no caller can use.
    count = np.where(sig > 0.0, stop - first, 0)
    line = np.repeat(np.arange(cent.size), count)
    if line.size == 0:
        return out
    start = np.repeat(np.cumsum(count) - count, count)
    pixel = first[line] + np.arange(line.size) - start
    scaled = (wave[pixel] - cent[line]) / sig[line]
    return np.bincount(
        pixel,
        weights=amp[line] * np.exp(-0.5 * scaled * scaled),
        minlength=wave.size,
    )


def sticks2vector(
    line_wave: np.ndarray,
    line_amp: np.ndarray,
    wave: np.ndarray,
) -> np.ndarray:
    line_wave = np.asarray(line_wave, float)
    line_amp = np.asarray(line_amp, float)
    wave = np.asarray(wave, float)
    out = np.zeros_like(wave, dtype=float)
    if wave.size < 2 or line_wave.size == 0:
        return out

    inside = (line_wave >= wave[0]) & (line_wave <= wave[-1])
    if not np.any(inside):
        return out
    line_wave = line_wave[inside]
    line_amp = line_amp[inside]

    hi = np.searchsorted(wave, line_wave, side="left")
    inner = (hi > 0) & (hi < wave.size)
    if np.any(inner):
        i1 = hi[inner]
        i0 = i1 - 1
        span = wave[i1] - wave[i0]
        frac = np.divide(line_wave[inner] - wave[i0], span, out=np.zeros_like(i0, dtype=float), where=span > 0)
        frac = np.clip(frac, 0.0, 1.0)
        np.add.at(out, i0, line_amp[inner] * (1.0 - frac))
        np.add.at(out, i1, line_amp[inner] * frac)

    np.add.at(out, 0, np.sum(line_amp[hi == 0]))
    np.add.at(out, wave.size - 1, np.sum(line_amp[hi >= wave.size]))
    return out


def _build_d2_operator(n_par: int) -> np.ndarray:
    """Second-difference operator D such that (D c)_i = c_i - 2 c_{i+1} + c_{i+2}."""
    if n_par < 3:
        return np.zeros((0, n_par), dtype=float)
    d2 = np.zeros((n_par - 2, n_par), dtype=float)
    idx = np.arange(n_par - 2)
    d2[idx, idx] = 1.0
    d2[idx, idx + 1] = -2.0
    d2[idx, idx + 2] = 1.0
    return d2


# Rayleigh / aerosol optical depths at LCO.  These mirror the closed forms in
# ``moon_zodi_model`` (``_rayleigh_optical_depth`` and its aerosol companion) so
# the split-zodi colour envelopes below agree with the frozen physical model
# they were calibrated against; they are duplicated rather than imported to keep
# this module free of the astropy-coordinates / IERS import chain.
def _rayleigh_tau(wave_angstrom: np.ndarray, pressure_hpa: float = 744.0) -> np.ndarray:
    wave_micron = np.asarray(wave_angstrom, dtype=float) / 1.0e4
    inv_sq = wave_micron ** -2
    return (
        0.008569
        * wave_micron ** -4
        * (1.0 + 0.0113 * inv_sq + 0.00013 * inv_sq ** 2)
        * (pressure_hpa / 1013.25)
    )


def _aerosol_tau(wave_angstrom: np.ndarray) -> np.ndarray:
    return 0.0336 * (np.asarray(wave_angstrom, dtype=float) / 5000.0) ** -1.38


def _stable_phi(value: np.ndarray) -> np.ndarray:
    """(1 - exp(-t)) / t, series-continued through t -> 0.

    The airmass-difference integral of the scattering source term along the
    line of sight.  Mirrors ``moon_zodi_model._stable_phi``.
    """
    value = np.asarray(value, dtype=np.float64)
    small = np.abs(value) < 1.0e-4
    safe = np.where(small, 1.0, value)
    direct = -np.expm1(-value) / safe
    series = 1.0 - value / 2.0 + value**2 / 6.0 - value**3 / 24.0
    return np.where(small, series, direct)


def _rayleigh_phase(cosine: float, depolarization: float = 0.0148) -> float:
    """Rayleigh scattering phase function.  Mirrors the frozen model."""
    return float(
        3.0 * (1.0 - depolarization)
        / (16.0 * np.pi * (1.0 + 2.0 * depolarization))
        * (1.0 + (1.0 + 3.0 * depolarization) / (1.0 - depolarization) * cosine**2)
    )


def _henyey_greenstein_phase(cosine: float, g: float = 0.8) -> float:
    """Forward-peaked aerosol phase function.  Mirrors the frozen model."""
    return float((1.0 - g**2)
                 / (4.0 * np.pi * (1.0 + g**2 - 2.0 * g * cosine) ** 1.5))


def _unit_median(vec: np.ndarray) -> np.ndarray:
    """Normalise a colour/attenuation shape to unit median.

    Every envelope factor in this module is median-normalised so that the shape
    change does not move the coefficient scale.  For the airmass-dependent zodi
    factor this also makes the envelope SHAPE-ONLY: the fitted amplitude keeps
    the meaning it has today, so Chapter 2 of the ML notebook (which applies
    geometry factor 1.0 to moon and zodi) needs no matching change.
    """
    vec = np.asarray(vec, dtype=float)
    med = float(np.nanmedian(vec))
    return (vec / med) if np.isfinite(med) and med > 0.0 else vec


@dataclass(slots=True)
class SkyDecompResult:
    coef: np.ndarray
    # Per-coefficient 1σ posterior uncertainty in the same units and order as
    # ``coef``. Computed from the active-set Fisher information of the final
    # weighted-NNLS solve (see ``SkyDecomp._coef_err_active_set`` for the full
    # derivation): the (regularised) Hessian sub-block of the strictly positive
    # coefficients is inverted, the internal column/data scaling is undone, and
    # the variance is inflated by ``max(reduced_chi², 1)`` before taking the
    # square root. Boundary coefficients pinned at c = 0 by the c ≥ 0 constraint
    # get ``NaN`` (no symmetric interval is defined there).
    coef_err: np.ndarray
    bestfit: np.ndarray
    resid: np.ndarray
    resid_level: float
    fit_status: str
    fit_summary: str
    reduced_chi2: float
    fit_elapsed_sec: float
    components: dict[str, np.ndarray]
    design_names: list[str]
    t_o2: float
    t_o2_err: float
    r2: float
    rms_resid: float
    peak_memory_mb: float
    o2_fit_status: str
    o2_fit_summary: str
    o2_fit_elapsed_sec: float
    o2_valid_frac: float
    lsf_kernels: dict[str, np.ndarray]
    lsf_metrics: dict[str, dict[str, object]]
    bestfit_lsf: np.ndarray
    moon_knots: np.ndarray
    moon_boosted_pixels: np.ndarray
    # Unit-integrated O2 template on `wave` and the prefit amplitude that
    # scales it; the fitted `coef['O2_b01']` absorbs the amplitude, so
    # `coef * vector_o2` reproduces the model O2 contribution.
    vector_o2: np.ndarray
    o2_prefit_amp: float
    # Per-pixel 1σ of the LSF-refined reconstruction, obtained by first-order
    # (Jacobian) propagation of ``coef_err`` through the LSF-convolved design
    # matrices used to build ``bestfit_lsf`` (see
    # ``SkyDecomp._components_sigma_from_coef_err``).  Same shape as
    # ``bestfit_lsf``.  Written to disk as the ``FLUX_SIGMA_TOTAL`` HDU by the
    # parallel decomposition writer.
    bestfit_lsf_sigma: np.ndarray
    # Zodi family (only populated when `SkyDecomp(split_zodi=True)`).  kw_only so
    # LSFSurfaceIterativeResult can still add positional fields after us.
    zodi_names: list[str] = field(default_factory=list, kw_only=True)
    zodi_knots: np.ndarray = field(default_factory=lambda: np.array([], float), kw_only=True)
    # Full posterior covariance sub-matrices for the moon and (optionally) zodi
    # B-spline blocks -- 2D arrays with shape (n_moon, n_moon) / (n_zodi, n_zodi)
    # in the same physical units as ``coef``. Entries corresponding to inactive
    # (boundary-zero) coefficients are ``NaN`` (no symmetric interval defined).
    # These carry the off-diagonal correlations that ``coef_err`` (= sqrt of the
    # diagonal) discards; downstream code that needs the joint uncertainty on
    # any linear combination of moon or zodi coefficients (e.g. total moon
    # amplitude, PCA scores) should use these matrices instead of assuming
    # independence.  ``None`` when the block was not present in the solve.
    coef_cov_moon: np.ndarray | None = field(default=None, kw_only=True)
    coef_cov_zodi: np.ndarray | None = field(default=None, kw_only=True)


class SkyDecomp:
    oh_group_keys = OH_GROUP_KEYS

    def __init__(
        self,
        wave: np.ndarray,
        *,
        lsf_sigma: np.ndarray | float = 0.5,
        n_spline_knots: int = 25,
        base_dir: str | Path | None = None,
        palace_suffix: str | None = None,
        palace_oh_suffix: str | None = None,
        palace_diffuse_suffix: str | None = None,
        o2_min_valid_frac: float = O2_MIN_VALID_FRAC,
        moon_smooth_lambda: float = 1e-3,
        moon_ratio_bound: float = 0.0,
        zodi_ratio_bound: float = 0.0,
        moon_ratio_relax_window: tuple[float, float] | None = None,
        moon_ratio_relax_bound: float = 0.0,
        amp_prior_tol: float = 0.0,
        amp_prior_floor: float = 0.02,
        zodi_amp_bound: float = 0.0,
        diffuse_ratio_bound_dex: float = 0.0,
        diffuse_ratio_nominal: tuple[float, ...] | None = None,
        diffuse_oh_centre_log10: float | None = None,
        diffuse_oh_bound_dex: float = 0.0,
        diffuse_oh_gate_frac: float = 0.6,
        diffuse_oh_relax_dex: float = 0.0,
        diffuse_oh_scope: str = "block",
        moon_scatter_envelope: bool = False,
        moon_ms_coeff: float = 3.5,
        moon_relax_gate: tuple[float, float] | None = None,
        moon_interline_boost: float = 0.0,
        moon_interline_red_min: float = 7454.0,
        moon_interline_exclusion_a: float = 3.0,
        moon_interline_line_flux_threshold: float = 0.0,
        split_zodi: bool = False,
        n_zodi_spline_knots: int = 3,
        zodi_smooth_lambda: float | None = None,
        moon_albedo_asset_path: str | pathlib.Path | None = None,
        moon_albedo_fiducial_phase_deg: float = 30.0,
        zodi_color_exponent: float = 0.26,
    ) -> None:
        self.wave = np.asarray(wave, float)
        self.lsf_sigma = np.asarray(lsf_sigma, float)
        self.n_spline_knots = int(n_spline_knots)
        self.o2_min_valid_frac = float(np.clip(o2_min_valid_frac, 0.0, 1.0))
        self.moon_smooth_lambda = max(float(moon_smooth_lambda), 0.0)
        # Hard shape bounds, beta in (0, 1); 0 disables.  Preferred over the
        # slope penalties above: scale-invariant and no likelihood trade-off.
        self.moon_ratio_bound = float(moon_ratio_bound)
        self.zodi_ratio_bound = float(zodi_ratio_bound)
        # Per-knot relaxation of the MOON ratio bound over one wavelength
        # window.  Motivation: with the zodi bound applied, the only place the
        # fit degrades is a narrow blue band -- 4000-4200 A carries 61-170% of
        # the net excess residual on all three arms (far/near/sci), with
        # 13-15 of 31 bands actually improving and the net cost ~0% of the
        # total |residual|.  That band is where the solar SED both families are
        # built on has its densest line blanketing (Ca II H&K, H-delta, CN,
        # Fe I), so it looks like a template or blue-throughput defect rather
        # than a missing continuum component.  Relaxing the MOON knots there
        # lets the moon family absorb the artifact while zodi stays pinned
        # everywhere -- deliberately asymmetric, because 4000-4200 A is also
        # where the two colour envelopes are most alike and relaxing zodi would
        # reopen the role swap exactly where it is easiest.
        #
        # This is a WORKAROUND, not a fix: if the excess is a calibration or
        # solar-template defect it should be corrected at source, and this
        # relaxation will hide it until then.
        self.moon_ratio_relax_window = (
            None if moon_ratio_relax_window is None
            else (float(min(moon_ratio_relax_window)),
                  float(max(moon_ratio_relax_window))))
        self.moon_ratio_relax_bound = float(moon_ratio_relax_bound)
        self._moon_full_knots = None
        self._moon_relaxed_basis = None
        self.target_airmass = None
        # Amplitude bracket on the moon's share of the moon+zodi continuum,
        # f = int(moon) / int(moon + zodi).  The ratio bounds above fix the two
        # families' COLOURS but leave their amplitudes free, and measurement on
        # 100 lunation-stratified sky spectra shows the amplitudes are wrong in
        # a way colour alone cannot reach:
        #   * fitted f is flat at 0.20-0.30 from new moon to full, where the
        #     geometry predicts 0.00 -> 0.26 -> 0.52 -> 0.68 -> 0.86 -> 0.94;
        #   * fitted int(zodi) tracks lunar illumination at rho = +0.95 (it must
        #     not: the zodiacal light does not care about the moon) and retains
        #     only rho = 0.26 of its Leinert B500 dependence, against 0.996 for
        #     the prediction -- the "zodi" family is largely moonlight;
        #   * with the moon 37 deg BELOW the horizon the moon family still takes
        #     31-43% of the continuum, where geometry predicts 2e-5.
        # `set_amplitude_prior` supplies f_pred per row; `amp_prior_tol` is the
        # multiplicative tolerance kappa (<=1 disables) allowed either side of
        # it, imposed as two hard linear inequalities on the fitted totals.
        #
        # Constraining the FRACTION, not either absolute amplitude, is what
        # makes this calibration-free: both predictions leave the physical
        # model through the same solid-angle/flux conversion, so their ratio is
        # independent of throughput and of physical_to_fit_flux_scale.  An
        # earlier attempt (removed 2026-08-26) instead put a soft quadratic
        # prior on the absolute zodi amplitude, anchored by a self-calibrated
        # median(int_zodi / B500) taken over moon-down rows -- circular, since
        # those rows carry the 43% spurious moon share measured above.  Being
        # homogeneous in c these rows are also scale-invariant, so unlike a
        # quadratic penalty they carry no hidden data_scale**2 (see the D2 note
        # in the solver) and never bid against the likelihood.
        self.amp_prior_tol = float(amp_prior_tol)
        # Slack kept at both ends of the bracket: the moon may always take up
        # to `amp_prior_floor` of the continuum and the zodi is never squeezed
        # below it.  Guards the two places the prediction is least trustworthy
        # -- the below-horizon suppression is a learned extrapolation fitted
        # only down to -7 deg altitude, and ROLO phase coverage runs out on
        # thin crescents -- and keeps the bracket strictly inside (0, 1).
        self.amp_prior_floor = float(np.clip(amp_prior_floor, 0.0, 0.499))
        self._amp_prior_moon_fraction = None
        # Absolute bracket on int(zodi) against the Leinert prediction, as a
        # multiplicative tolerance (<=1 disables).  The fraction bracket above
        # is RELATIVE, so it cannot restore the zodi's ecliptic dependence: it
        # only divides whatever total the two splines hold between them.  On
        # the same 100 spectra the fitted zodi tracks Leinert B500 at rho=+0.52
        # in dark time but only +0.17 with the moon up, and sits at 4.5x its
        # predicted amplitude there (q90 = 18x) -- the family is fine until a
        # moon is present and then becomes moonlight.  Only an absolute anchor
        # reaches that.
        #
        # Unlike the fraction rows this one has a non-zero right-hand side, so
        # it is stated in native flux units: `set_amplitude_prior` must be
        # given zodi_total on the same scale as the flux being fitted (i.e.
        # multiplied by the same FACTOR), because unlike the fraction the
        # absolute amplitude does NOT cancel the flux calibration.
        #
        # Its empirical licence is the dark-time measurement above: with the
        # shape bounds on and no moon, fitted/predicted int(zodi) is 0.94 with
        # 0.16 dex scatter, so the literature absolute scale is good to ~6%
        # median and a factor ~1.4 spread.  Do not tighten kappa below that
        # spread.
        self.zodi_amp_bound = float(zodi_amp_bound)
        # Diffuse species-ratio bracket.  `_dex` is the HALF-WIDTH in dex
        # allowed either side of `_nominal` (0 disables); `_nominal` is the
        # (HO2, FeO, O2Ac) FLUX-share centre, which must be measured on the
        # corpus being fitted -- see the construction site for why PALACE's
        # own shares are the wrong centre for LCO.
        self.diffuse_ratio_bound_dex = float(max(diffuse_ratio_bound_dex, 0.0))
        if diffuse_ratio_nominal is None:
            self.diffuse_ratio_nominal = None
        else:
            _dn = np.asarray(diffuse_ratio_nominal, dtype=np.float64).ravel()
            if _dn.size < 2 or not np.all(np.isfinite(_dn)) or not np.all(_dn > 0.0):
                raise ValueError(
                    "diffuse_ratio_nominal must be >=2 finite positive flux shares, "
                    f"got {diffuse_ratio_nominal!r}")
            self.diffuse_ratio_nominal = tuple(_dn / _dn.sum())
        # FeO-vs-OH upper bound, moon-gated.  FeO is mesospheric
        # chemiluminescence and cannot depend on the moon, yet measured on
        # gaia-stars-mask-cont its amplitude relative to OH rises a factor 4.5
        # with moon_frac_po (rho +0.69) while airmass and van Rhijn show
        # nothing -- the FeO template is absorbing scattered moonlight, ~74% of
        # the fitted FeO on full-moon rows.  This caps it.
        #
        # UPPER BOUND ONLY and MOON-GATED, both measured: the dark-time scatter
        # of log10(A_FeO/A_OH) is 0.306 dex and is REAL -- clipping it costs
        # 15-32% of the blue chi2 -- so a two-sided or ungated bound is
        # unaffordable.  The allowed width ramps from `+relax` at the gate to
        # `bound` at full moon, so the constraint tightens where the leak is
        # worst and vanishes below the gate.
        self.diffuse_oh_centre_log10 = (None if diffuse_oh_centre_log10 is None
                                    else float(diffuse_oh_centre_log10))
        self.diffuse_oh_bound_dex = float(max(diffuse_oh_bound_dex, 0.0))
        self.diffuse_oh_gate_frac = float(np.clip(diffuse_oh_gate_frac, 0.0, 0.999))
        self.diffuse_oh_relax_dex = float(max(diffuse_oh_relax_dex, 0.0))
        if str(diffuse_oh_scope) not in ("block", "feo"):
            raise ValueError(
                f"diffuse_oh_scope must be 'block' or 'feo', got {diffuse_oh_scope!r}")
        self.diffuse_oh_scope = str(diffuse_oh_scope)
        # Installed per row by `set_diffuse_oh_reference`; None disables the bound.
        self._diffuse_oh_amp = None
        self._amp_prior_zodi_total = None
        # Two-channel (Rayleigh + aerosol) scattered-moonlight envelope with a
        # multiple-scattering enhancement, installed per row by
        # `set_moon_geometry`.  Off by default: the envelope is then the
        # Rayleigh-only shape below and the design matrix is unchanged.
        # `moon_ms_coeff` is the c in the (1 + c X tau_R) enhancement; the
        # frozen model uses 3.5.
        self.moon_scatter_envelope = bool(moon_scatter_envelope)
        self.moon_ms_coeff = float(moon_ms_coeff)
        # Gate the blue moon relaxation on how much the moon actually
        # contributes, as (f_lo, f_hi) on the geometry-predicted moon fraction
        # installed by `set_amplitude_prior`.  Below f_lo the relaxed knots
        # keep the tight bound; above f_hi they get the full
        # `moon_ratio_relax_bound`; between, beta interpolates.
        #
        # The point is that the two regimes are disjoint.  Reversals live in
        # the faint-moon corner -- rho(separation margin, FLI) = +0.62 and every
        # thin-margin row has a faint moon -- while the blue residual cost lives
        # at FLI > 0.6, where the margin's minimum over 56 spectra is +2.81.
        # An ungated relaxation buys blue freedom everywhere and pays for it
        # where the moon component is too weak to determine its own shape; the
        # gate buys it only where the data can support it.
        #
        # Using the PREDICTED fraction rather than the fitted one keeps this
        # from being circular: the fit cannot widen its own bound by putting
        # more flux in the moon.
        self.moon_relax_gate = (
            None if moon_relax_gate is None
            else (float(min(moon_relax_gate)), float(max(moon_relax_gate))))
        self.moon_scatter_shape = None
        self._moon_envelope_base = None
        self._moon_envelope_base_hr = None
        self._moon_bspl = None
        self.moon_interline_boost = max(float(moon_interline_boost), 0.0)
        self.moon_interline_red_min = float(moon_interline_red_min)
        self.moon_interline_exclusion_a = max(float(moon_interline_exclusion_a), 0.0)
        self.moon_interline_line_flux_threshold = float(np.clip(moon_interline_line_flux_threshold, 0.0, 1.0))
        self.base_dir = (
            Path(base_dir).resolve() if base_dir is not None else Path(__file__).resolve().parent.parent
        )
        if (self.base_dir / "bundle_manifest.json").is_file() and palace_suffix is None:
            if palace_oh_suffix is None:
                palace_oh_suffix = DEFAULT_PALACE_OH_SUFFIX
            if palace_diffuse_suffix is None:
                palace_diffuse_suffix = DEFAULT_PALACE_DIFFUSE_SUFFIX
        self.palace_suffix = self._validate_palace_suffix(palace_suffix)
        self.palace_oh_suffix = self._validate_palace_suffix(palace_oh_suffix)
        self.palace_diffuse_suffix = self._validate_palace_suffix(palace_diffuse_suffix)
        self.pmd_dir = self.base_dir / "palace" / "PMD"
        self.solar_path = self.base_dir / "Spectre_HR_LATMOS_Meftah_V1_350_1000nm.txt"

        self.t_o2 = T_O2_REF
        self.t_o2_err = np.nan
        self.o2_prefit_amp = np.nan
        self.vector_o2 = np.zeros_like(self.wave)
        self.matrix_o2 = self.vector_o2[None, :]
        self.vector_o2_stick = np.zeros_like(self.wave)
        self.matrix_o2_stick = self.vector_o2_stick[None, :]
        self.o2_prefit_bestfit = self.vector_o2.copy()
        self.bestfit = np.zeros_like(self.wave)
        self.coef = np.array([], float)
        self.coef_err = np.array([], float)
        self.fit_status = ""
        self.fit_summary = ""
        self.r2 = np.nan
        self.rms_resid = np.nan
        self.peak_memory_mb = np.nan
        self.o2_fit_status = ""
        self.o2_fit_summary = ""
        self.o2_fit_elapsed_sec = 0.0
        self.o2_valid_frac = np.nan
        self.lsf_kernels: dict[str, np.ndarray] = {}
        self.lsf_metrics: dict[str, dict[str, object]] = {}
        self.bestfit_lsf = np.zeros_like(self.wave)
        self.moon_knots_used = np.array([], float)
        self.moon_boosted_pixels_used = np.array([], float)

        # ---- Split-zodi configuration ----
        self.split_zodi = bool(split_zodi)
        self.n_zodi_spline_knots = int(n_zodi_spline_knots)
        if zodi_smooth_lambda is None:
            self.zodi_smooth_lambda = 10.0 * self.moon_smooth_lambda
        else:
            self.zodi_smooth_lambda = max(float(zodi_smooth_lambda), 0.0)
        self.moon_albedo_asset_path = (
            pathlib.Path(moon_albedo_asset_path)
            if moon_albedo_asset_path is not None
            else self.base_dir / 'moon_zodi'
              / 'eso_skycalc_rolo_moon_albedo.dat'
        )
        self.moon_albedo_fiducial_phase_deg = float(moon_albedo_fiducial_phase_deg)
        self.zodi_color_exponent = float(zodi_color_exponent)
        # Populated by _build_static_basis when split_zodi=True; otherwise empty.
        self.matrix_zodi = np.zeros((0, self.wave.size), dtype=float)
        self.matrix_zodi_hr = np.zeros((0, self.wave.size), dtype=float)
        self.zodi_names: list[str] = []
        self.zodi_knots_used = np.array([], float)
        self.moon_albedo_shape = np.ones_like(self.wave)
        self.zodi_color_shape = np.ones_like(self.wave)
        # Cached second-difference operators for the moon / zodi curvature penalties;
        # populated in _build_moon / _build_zodi.  Shape is (n_par - 2, n_par).
        self._d2_moon = np.zeros((0, 0), dtype=float)
        self._d2_zodi = np.zeros((0, 0), dtype=float)
        self._zodi_bspl = None
        self._zodi_envelope_base = None
        self.zodi_extinction_shape = None
        self.moon_rayleigh_shape = None

        self._build_static_basis()

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
        n_lsf_refits: int = 1,
    ) -> SkyDecompResult:
        flux = np.asarray(flux, float)
        ivar = np.asarray(ivar, float)
        n_lsf_refits = max(int(n_lsf_refits), 0)
        trace_started = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            trace_started = True
        tracemalloc.reset_peak()
        t0 = time.perf_counter()

        self._prefit_o2(flux, ivar)
        self.design_matrix = self._assemble_design_matrix()
        comp_slices = self._component_slices(
            self._matrix_bundle(
                self.matrix_oh,
                self.matrix_moon,
                self.matrix_diffuse,
                self.matrix_atom,
                self.matrix_orc,
                self.matrix_o2,
                matrix_zodi=self.matrix_zodi if self.split_zodi else None,
            )
        )
        first = self._fit_design(
            self.design_matrix,
            flux,
            ivar,
            moon_slice=comp_slices["moon"],
            diffuse_slice=comp_slices["diffuse"],
            zodi_slice=comp_slices.get("zodi"),
        )
        self.bestfit = first["bestfit"]
        refined = first
        final_matrices = self._matrix_bundle(
            self.matrix_oh,
            self.matrix_moon,
            self.matrix_diffuse,
            self.matrix_atom,
            self.matrix_orc,
            self.matrix_o2,
            matrix_zodi=self.matrix_zodi if self.split_zodi else None,
        )
        self.lsf_kernels = {}
        self.lsf_metrics = {}
        iter_logs: list[dict[str, object]] = []

        for i_refit in range(1, n_lsf_refits + 1):
            source_lsf, fixed_background = self._build_lsf_source(refined["coef"])
            self._fit_lsf_channels(flux, ivar, source_lsf, fixed_background)
            final_matrices = self._assemble_refined_matrices()
            _refined_names = ["oh", "moon"]
            if "zodi" in comp_slices:
                _refined_names.append("zodi")
            _refined_names.extend(["diffuse", "atom", "orc", "o2"])
            refined_design = np.vstack([final_matrices[name] for name in _refined_names])
            refined = self._fit_design(
                refined_design,
                flux,
                ivar,
                moon_slice=comp_slices["moon"],
                diffuse_slice=comp_slices["diffuse"],
                zodi_slice=comp_slices.get("zodi"),
            )
            iter_logs.append(
                {
                    "iter": i_refit,
                    "lsf": {
                        ch: {
                            "status": met["status"],
                            "chi2_red": met["chi2_red"],
                            "sigma_pix": met["sigma_pix"],
                            "center_pix": met["center_pix"],
                            "runtime_sec": met["runtime_sec"],
                        }
                        for ch, met in self.lsf_metrics.items()
                    },
                    "fit": {
                        "status": refined["status"],
                        "chi2_red": refined["reduced_chi2"],
                        "r2": refined["r2"],
                        "rms": refined["rms_resid"],
                        "qp_dt": refined["qp_elapsed_sec"],
                    },
                }
            )

        components = self._components_from_coef(refined["coef"], final_matrices)
        # `final_matrices['o2']` is a (1, n_wave) block that already incorporates
        # every LSF refit; keep `vector_o2` in step so what we persist matches
        # the O2 basis used to build `components['o2']`.
        if final_matrices["o2"].shape[0] == 1:
            self.vector_o2 = final_matrices["o2"][0].copy()

        # Per-pixel 1σ of the refined reconstruction (LSF-aware Jacobian
        # propagation).  Uses the SAME matrix bundle as `_components_from_coef`
        # above so the mean and σ come from an identical LSF/O2 basis.  Needs
        # the finalised coef_err from `refined`, not `self.coef_err` (which is
        # not filled until a few lines below).
        _sigma_coef_err = np.asarray(
            refined.get("coef_err", np.full(refined["coef"].shape, np.nan)),
            dtype=float,
        )
        _sigma_comps = self._components_sigma_from_coef_err(_sigma_coef_err, final_matrices)
        bestfit_lsf_sigma = np.sqrt(
            _sigma_comps["oh"] ** 2
            + _sigma_comps["moon"] ** 2
            + _sigma_comps["diffuse"] ** 2
            + _sigma_comps["atom"] ** 2
            + _sigma_comps["orc"] ** 2
            + _sigma_comps["o2"] ** 2
        )

        fit_elapsed_sec = time.perf_counter() - t0
        peak_memory_mb = tracemalloc.get_traced_memory()[1] / 1024**2
        if trace_started:
            tracemalloc.stop()
        self.peak_memory_mb = peak_memory_mb
        self.bestfit_lsf = refined["bestfit"]
        self.coef = refined["coef"]
        self.coef_err = np.asarray(
            refined.get("coef_err", np.full_like(self.coef, np.nan)),
            dtype=float,
        )
        self.fit_status = refined["status"]
        self.fit_summary = (
            f"status={refined['status']} | npar={refined['n_par']} | ngood={refined['n_good']} | "
            f"chi2_red={refined['reduced_chi2']:.4g} | R2={refined['r2']:.5f} | "
            f"qp_dt={refined['qp_elapsed_sec']:.2f}s | moon_smooth_lambda={self.moon_smooth_lambda:.3g} | "
            f"moon_mask_downweight_boost={self.moon_interline_boost:.3g} | "
            f"moon_interline_line_flux_threshold={self.moon_interline_line_flux_threshold:.3g} | "
            f"n_lsf_refits={n_lsf_refits} | dt={fit_elapsed_sec:.2f}s"
        )
        self.r2 = refined["r2"]
        self.rms_resid = refined["rms_resid"]

        if verbose:
            print("O2")
            print(f"  T         {self.t_o2:.2f} +/- {self.t_o2_err:.2f} K")
            print(f"  chi2_red  {self._fmt_num(self._extract_o2_chi2_red())}")
            print(f"  dt        {self.o2_fit_elapsed_sec:.3f} s")
            print()
            print("decomp")
            print(f"  init      {self._fmt_num(first['reduced_chi2'])}")
            print()
            print("iterations")
            for log in iter_logs:
                lsf_dt = 0.0
                vals = {}
                for ch in ("B", "R", "Z"):
                    met = log["lsf"].get(ch)
                    vals[ch] = self._fmt_num(np.nan if met is None else met["chi2_red"])
                    if met is not None:
                        lsf_dt += float(met["runtime_sec"])
                fit_log = log["fit"]
                print(
                    f"  [{log['iter']}] LSF     "
                    f"B={vals['B']}   R={vals['R']}   Z={vals['Z']}   dt={lsf_dt:.3f} s"
                )
                print(
                    f"      decomp  chi2_red={self._fmt_num(fit_log['chi2_red'])}   "
                    f"qp={fit_log['qp_dt']:.3f} s"
                )
            if not iter_logs:
                print("  none")
            print()
            print("final")
            print(
                f"  decomp    {self._fmt_num(refined['reduced_chi2'])}\n"
                f"  refits    {n_lsf_refits}\n"
                f"  total_dt  {fit_elapsed_sec:.3f} s\n"
                f"  peak_mem  {peak_memory_mb:.2f} MB"
            )

        return SkyDecompResult(
            coef=self.coef,
            coef_err=self.coef_err,
            bestfit=first["bestfit"],
            resid=first["resid"],
            resid_level=first["resid_level"],
            fit_status=self.fit_status,
            fit_summary=self.fit_summary,
            reduced_chi2=refined["reduced_chi2"],
            fit_elapsed_sec=fit_elapsed_sec,
            components=components,
            design_names=self.design_names,
            t_o2=self.t_o2,
            t_o2_err=self.t_o2_err,
            r2=self.r2,
            rms_resid=self.rms_resid,
            peak_memory_mb=peak_memory_mb,
            o2_fit_status=self.o2_fit_status,
            o2_fit_summary=self.o2_fit_summary,
            o2_fit_elapsed_sec=self.o2_fit_elapsed_sec,
            o2_valid_frac=self.o2_valid_frac,
            lsf_kernels=self.lsf_kernels,
            lsf_metrics=self.lsf_metrics,
            bestfit_lsf=self.bestfit_lsf,
            moon_knots=self.moon_knots_used.copy(),
            moon_boosted_pixels=self.moon_boosted_pixels_used.copy(),
            vector_o2=self.vector_o2.copy(),
            o2_prefit_amp=float(self.o2_prefit_amp),
            bestfit_lsf_sigma=bestfit_lsf_sigma,
            zodi_names=list(self.zodi_names),
            zodi_knots=self.zodi_knots_used.copy(),
            coef_cov_moon=refined.get("coef_cov_moon"),
            coef_cov_zodi=refined.get("coef_cov_zodi"),
        )

    @staticmethod
    def _matrix_bundle(
        matrix_oh: np.ndarray,
        matrix_moon: np.ndarray,
        matrix_diffuse: np.ndarray,
        matrix_atom: np.ndarray,
        matrix_orc: np.ndarray,
        matrix_o2: np.ndarray,
        matrix_zodi: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "oh": matrix_oh,
            "moon": matrix_moon,
            "diffuse": matrix_diffuse,
            "atom": matrix_atom,
            "orc": matrix_orc,
            "o2": matrix_o2,
        }
        if matrix_zodi is not None and matrix_zodi.shape[0] > 0:
            out["zodi"] = matrix_zodi
        return out

    @staticmethod
    def _per_column_chi2_from_residuals(
        design_matrix: np.ndarray,
        resid: np.ndarray,
        ivar: np.ndarray,
        good: np.ndarray,
        *,
        fallback: float,
    ) -> np.ndarray:
        """Per-column pseudo-chi² from residual power weighted by column support.

        For each column ``j`` of the physical design matrix, compute a
        heteroscedasticity-consistent (HC1-style) chi² by weighting the
        pixel-wise weighted residual power by that column's spectral support:

            w_j[i]        = |A[i, j]| / max_i |A[i, j]|
            chi²_j        = Σ_i w_j[i] · ivar[i] · resid[i]²  /  Σ_i w_j[i]

        The ``max_i |A[i, j]|`` normalisation cancels in the ratio, so the
        implementation collapses to two BLAS gemv calls over ``|A[good, :]|``:

            chi²_j        = (Σ_i |A[i, j]| · ivar[i] · resid[i]²) / Σ_i |A[i, j]|

        Columns with vanishing support (all-zero design over the good pixels)
        fall back to ``fallback`` (typically the aggregate ``reduced_chi²``).
        This gives ``_coef_err_active_set`` a per-column inflation factor
        instead of the scalar ``max(reduced_chi², 1)`` used historically,
        matching the MLP-side observation (top methods §7) that the
        miscalibration is column-dependent.
        """
        # design_matrix here is (n_col, n_pix) -- see _fit_design.
        n_par = int(design_matrix.shape[0])
        per_col = np.full(n_par, float(fallback), dtype=float)
        if not good.any():
            return per_col
        # Take absolute value once on the good-pixel slice (n_col, n_good);
        # the transpose in the historical implementation is skipped because
        # gemv over rows is just as fast as gemv over columns and saves the
        # extra (n_good, n_col) copy.
        abs_slice = np.abs(design_matrix[:, good])  # (n_col, n_good)
        r2w = (resid[good] * resid[good]) * ivar[good]  # (n_good,)
        num = abs_slice @ r2w  # (n_col,) -- single gemv
        den = abs_slice.sum(axis=1)  # (n_col,)
        support_mask = den > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2_j = num / np.where(support_mask, den, 1.0)
        chi2_j = np.where(np.isfinite(chi2_j), chi2_j, float(fallback))
        per_col[support_mask] = chi2_j[support_mask]
        return per_col

    @staticmethod
    def _coef_err_active_set(
        coef: np.ndarray,
        source_p: list[np.ndarray],
        source_col_scale: list[np.ndarray],
        source_data_scale: list[float],
        source_local_index: list[int],
        reduced_chi2: float,
        per_column_chi2: np.ndarray | None = None,
        cov_block_slices: dict[str, slice] | None = None,
        unconstrained_indices: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Active-set posterior 1σ uncertainty for a constrained WLS solve.

        Notes
        -----
        The QP minimises

            J(c) = ½‖W(A c − y)‖² + ½ λ ‖L c‖²    subject to    c ≥ 0

        where ``W = diag(ivar_good)`` is the weighting used by ``_fit_design``,
        ``A`` is the physical design matrix and ``L`` is the moon-spline
        second-difference operator (with strength ``λ = 2·moon_smooth_lambda``).
        For readability the internal solve additionally divides the data by a
        scalar ``data_scale`` and rescales columns by ``col_scale``; those
        transformations cancel out of the KKT solution but must be undone here
        before the covariance is reported in native coefficient units.

        At the KKT solution, split coefficients into

            active   := {j : c_j > tol}      inactive := {j : c_j = 0}

        Holding the inactive set at zero collapses the constrained problem to
        an unconstrained WLS on the active columns

            c_a ← argmin ‖W (A_a c_a − y)‖² + λ ‖L_a c_a‖²

        whose Hessian is the active sub-block of the (regularised) Fisher
        information ``H = AᵀWA + λ LᵀL``. The corresponding maximum a
        posteriori covariance is

            Cov(c_a) = H_{a,a}⁻¹                                    (1)

        and it is what the routine reports.  Equivalently, this is the
        classical Cramér–Rao unconstrained covariance of the sub-model that
        contains only the active coefficients.

        Smoothness prior. ``H`` above is the *regularised* Fisher information,
        i.e. it already includes the ``λ LᵀL`` term added at solve time
        (`` _fit_design`` line "p_dense_local[..] += 2·smooth_λ·D²ᵀD² ").  So
        the σ reported here reflects both the data likelihood and the
        smoothness prior on adjacent B-spline coefficients; no separate
        Bayesian tightening step is needed.

        Undoing the internal scaling. What is stored in the QP is the dense
        matrix

            P_scaled = diag(1/col_scale) · (H / data_scale²) · diag(1/col_scale)
                       + regulariser

        so ``H⁻¹ = diag(1/col_scale) · P_scaled⁻¹ · diag(1/col_scale) · data_scale²``.
        The active sub-block of ``P_scaled`` is inverted once per solve group
        (Hessian identity is the group key so a two-stage moon/diffuse refit
        is handled cleanly), the column-scale factors are folded in via
        ``inv_col = 1/col_scale[a]``, and the diagonal gives ``Var(c_j)`` in
        physical units. The 1σ error is ``sqrt(Var)``.

        Noise inflation. Empirical ``reduced_chi²`` above unity indicates that
        the input ``ivar`` under-estimates the noise (or the model is
        mildly under-specified); the variance is therefore inflated by
        ``max(reduced_chi², 1)`` before the square root. This is the standard
        "sigma-hat" adjustment applied to weighted least squares.

        Per-column inflation (2026-08-16). The MLP sigma-calibration diagnostic
        (top methods §7) showed the aggregate ``reduced_chi²`` scalar
        systematically over-inflates well-fit columns and under-inflates poorly
        fit ones. When ``per_column_chi2`` is provided (see ``_fit_design``),
        the inflation factor becomes column-specific ``max(per_column_chi2[j], 1)``
        computed from residual power weighted by each column's spectral support.
        This is a lightweight heteroscedasticity-consistent (HC1-style) fix that
        preserves the active-set covariance structure and matches the physical
        intuition that emission-line columns and continuum-spline columns have
        very different residual behaviour. If ``per_column_chi2`` is ``None`` the
        routine falls back to the scalar ``reduced_chi²`` for backward
        compatibility.

        Boundary treatment. Inactive coefficients are pinned at c_j = 0 by
        the constraint and have no symmetric 1σ interval defined; they are
        reported as ``NaN``. A one-sided upper limit is easy to derive from
        ``|A_jᵀ W (y − A c)| / √(A_jᵀ W A_j)`` but is not written here to
        keep the array shape aligned with ``coef`` and avoid mixing
        interval-type semantics on a per-row basis.

        Caveats. Active-set covariance ignores the constraint's effect on
        coefficients that are close to zero: a small perturbation of the data
        can flip a boundary coefficient into the active set. A bootstrap or
        profile-likelihood analysis is more faithful in that regime but not
        needed for the calibration MLP downstream.

        Joint covariance blocks (2026-08-25). When ``cov_block_slices`` is
        supplied -- a mapping ``{name: slice}`` selecting one or more
        contiguous coefficient ranges (typically ``{'moon': moon_slice,
        'zodi': zodi_slice}``) -- the routine also assembles the full active-
        set covariance restricted to each named block and returns it as a 2D
        matrix (n_block × n_block) alongside the usual 1D ``coef_err``.  Off-
        diagonal correlations are computed with the same column-scale-undo
        and per-column-χ² inflation logic as the diagonal (off-diagonal
        inflation uses ``sqrt(inflate[i] * inflate[j])`` so the correlation
        matrix is preserved and the diagonal reproduces the vector ``coef_err``).
        Inactive-coefficient rows and columns are ``NaN``.  Return type is
        ``(coef_err, {name: cov_2d})`` in this case; when
        ``cov_block_slices`` is None (default) the routine returns just
        ``coef_err`` for backwards compatibility.
        """
        n_par = int(coef.size)
        coef_err = np.full(n_par, np.nan, dtype=float)
        want_blocks = cov_block_slices is not None
        if want_blocks:
            cov_blocks = {
                name: np.full(
                    (sl.stop - sl.start, sl.stop - sl.start), np.nan, dtype=float)
                for name, sl in cov_block_slices.items()
            }
        if n_par == 0:
            if want_blocks:
                return coef_err, cov_blocks
            return coef_err

        active_tol = 1e-8 * float(max(np.max(np.abs(coef)), 1.0))
        active_mask = coef > active_tol
        if unconstrained_indices is not None:
            unconstrained_indices = np.asarray(unconstrained_indices, dtype=int)
            active_mask[unconstrained_indices] = (
                np.abs(coef[unconstrained_indices]) > active_tol
            )
        coef_err[~active_mask] = np.nan

        scalar_inflate = float(max(reduced_chi2, 1.0))
        if per_column_chi2 is not None:
            per_col = np.asarray(per_column_chi2, dtype=float)
            if per_col.shape != coef.shape:
                raise ValueError(
                    f"per_column_chi2 shape {per_col.shape} != coef shape {coef.shape}"
                )
            per_col = np.where(np.isfinite(per_col) & (per_col > 0.0), per_col,
                                scalar_inflate)
            inflate_arr = np.maximum(per_col, 1.0)
        else:
            inflate_arr = np.full(coef.shape, scalar_inflate, dtype=float)

        # Group active coefficients by the solve they came from (identity of the
        # Hessian array); within each group we invert one small sub-block once.
        groups: dict[int, list[int]] = {}
        for j in np.flatnonzero(active_mask):
            groups.setdefault(id(source_p[int(j)]), []).append(int(j))

        for solve_indices in groups.values():
            first = solve_indices[0]
            p_solve = source_p[first]
            cs_solve = source_col_scale[first]
            ds_solve = source_data_scale[first]
            local_idx = np.asarray(
                [source_local_index[j] for j in solve_indices],
                dtype=int,
            )
            H_a = p_solve[np.ix_(local_idx, local_idx)]
            H_a = 0.5 * (H_a + H_a.T) + 1e-12 * np.eye(H_a.shape[0])
            try:
                cov_a = np.linalg.inv(H_a)
            except np.linalg.LinAlgError:
                cov_a = np.linalg.pinv(H_a)
            inv_col = 1.0 / cs_solve[local_idx]
            # Undo column and data scaling; inflate by per-column χ² (falling
            # back to the scalar reduced_chi² when per_column_chi2 is None); clip
            # to nonnegative before the square root to guard against tiny
            # negative eigenvalues from finite-precision arithmetic.
            #
            # diag(D · cov_a · D) = d² · diag(cov_a) when D is diagonal, so the
            # historical outer-product materialisation is skipped.
            solve_indices_arr = np.asarray(solve_indices, dtype=int)
            var_a = np.clip(
                (inv_col * inv_col)
                * np.diagonal(cov_a)
                * (ds_solve ** 2)
                * inflate_arr[solve_indices_arr],
                0.0,
                np.inf,
            )
            coef_err[solve_indices_arr] = np.sqrt(var_a)

            if want_blocks:
                # Materialise the physical-unit covariance for this solve group
                # (small square matrix; only done here in the block path).  This
                # applies the same column-scale-undo and χ² inflation as the
                # diagonal, so the sqrt of its diagonal reproduces coef_err.
                inflate_row = inflate_arr[solve_indices_arr]
                scale_row = inv_col * ds_solve * np.sqrt(inflate_row)
                cov_a_phys = (scale_row[:, None] * scale_row[None, :]) * cov_a
                for name, sl in cov_block_slices.items():
                    block_mask = (
                        (solve_indices_arr >= sl.start)
                        & (solve_indices_arr < sl.stop)
                    )
                    if not block_mask.any():
                        continue
                    block_local = solve_indices_arr[block_mask] - sl.start
                    block_source = np.flatnonzero(block_mask)
                    sub = cov_a_phys[np.ix_(block_source, block_source)]
                    cov_blocks[name][np.ix_(block_local, block_local)] = sub

        if want_blocks:
            return coef_err, cov_blocks
        return coef_err

    def _fit_design(
        self,
        design_matrix: np.ndarray,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        moon_slice: slice | None = None,
        diffuse_slice: slice | None = None,
        zodi_slice: slice | None = None,
        unconstrained_indices: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Weighted constrained least-squares fit of one design matrix.

        Solves the quadratic program

            c* = argmin_{c ≥ 0} ½‖diag(√ivar) (A c − y)‖²
                                + λ ‖L c‖²    (moon-spline curvature)

        on the finite/positive-ivar subset of the data. ``A = design_matrix``
        holds the physical basis; ``λ = moon_smooth_lambda`` (from the model
        config) is nonzero only for the moon block, whose curvature operator
        ``L`` penalises second differences of adjacent spline coefficients.
        Columns listed in ``unconstrained_indices`` may take either sign; all
        other coefficients retain the historical nonnegative constraint.

        For numerical stability the solve is performed on a doubly-rescaled
        version of the problem: ``y'' = y / data_scale`` and
        ``A'' = A / (data_scale · col_scale)`` where ``data_scale`` normalises
        the target RMS to unity and ``col_scale`` normalises each column to
        unit ℓ²-norm. The nonnegative QP is delegated to Clarabel. The
        returned coefficient is the fitted vector rescaled back into physical
        units, ``c = x* / col_scale``.

        An optional second solve refits the moon+diffuse (``target_cols``)
        block with an interline-boosted weight vector while the remaining
        coefficients are held at their first-solve values (subtracted from
        the data). When this happens each moon/diffuse coefficient's covariance
        is taken from the target-refit Hessian, not the first-solve Hessian —
        see the per-column provenance arrays ``source_p``, ``source_col_scale``,
        ``source_data_scale`` and ``source_local_index`` prepared here.

        Uncertainties
        -------------
        ``coef_err`` is filled by ``_coef_err_active_set`` right before the
        result dict is built. See that method for the full derivation; briefly,
        it inverts the active sub-block of the regularised Fisher information
        stored on ``source_p`` (one small linear-algebra call per solve group),
        undoes the ``col_scale`` / ``data_scale`` transformations, inflates
        variance by ``max(reduced_chi², 1)`` for a hat-sigma adjustment and
        reports the square root. Coefficients pinned at c = 0 by the c ≥ 0
        constraint receive ``NaN``.

        Returns
        -------
        dict with ``coef``, ``coef_err``, ``status``, ``bestfit``, ``resid``,
        ``resid_level``, ``n_good``, ``n_par``, ``chi2``, ``reduced_chi2``,
        ``r2``, ``rms_resid``, ``qp_elapsed_sec``, plus the main solve's
        ``data_scale_raw`` (before the 1.0 clamp), ``data_scale_used`` and
        ``data_scale_clamped``, which are diagnostics only.
        """
        good = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
        y = flux[good]
        base_w = np.sqrt(ivar[good])
        self.moon_boosted_pixels_used = np.array([], float)
        boosted_w = None
        if self.moon_interline_boost > 0.0:
            interline_weight = self._moon_interline_weights()
            masked_mask = good & (interline_weight < (1.0 - 1e-12))
            self.moon_boosted_pixels_used = self.wave[masked_mask]
            boosted_w = base_w * np.sqrt(interline_weight[good])
        n_good = int(np.sum(good))

        a = design_matrix[:, good].T
        n_par = int(a.shape[1])
        unconstrained = np.asarray(
            [] if unconstrained_indices is None else unconstrained_indices,
            dtype=int,
        )
        if (
            unconstrained.ndim != 1
            or np.any(unconstrained < 0)
            or np.any(unconstrained >= n_par)
            or np.unique(unconstrained).size != unconstrained.size
        ):
            raise ValueError("unconstrained_indices must contain unique valid columns")

        # Per-solve (raw, used) data_scale pairs, in call order: the main solve
        # first, then the interline refit when one runs. Reported out so the
        # clamp-activation rate can be measured across a corpus -- it decides
        # how badly the brightness coupling documented below actually bites.
        solve_scales: list[tuple[float, float]] = []

        def _solve_nonnegative_weighted(
            a_mat: np.ndarray,
            y_vec: np.ndarray,
            w_vec: np.ndarray,
            moon_slice_local: slice | None,
            zodi_slice_local: slice | None = None,
            diffuse_slice_local: slice | None = None,
            unconstrained_local: np.ndarray | None = None,
        ) -> tuple[np.ndarray, str, float, np.ndarray, np.ndarray, float]:
            aw = a_mat * w_vec[:, None]
            yw = y_vec * w_vec
            raw_data_scale = float(np.sqrt(np.nanmean(yw**2))) if yw.size else 1.0
            # Normalise by the UNCLAMPED scale.  The old `max(raw, 1.0)` clamp
            # was not a second-order detail: it pinned data_scale on 57% of the
            # solves that produce the kept coefficients, and on a pinned row
            # the DATA term stops being brightness-invariant too, so dividing
            # the penalty by the clamped value would have left exactly those
            # rows uncorrected.  The guard below is purely numerical -- an
            # all-zero target, where the solve is degenerate at any scale.
            #
            # Removing the clamp IMPROVES conditioning rather than harming it:
            # column norms are set by col_scale, not data_scale, so `yw` now
            # has exactly unit RMS instead of as little as 0.3.
            data_scale = (raw_data_scale
                          if np.isfinite(raw_data_scale)
                          and raw_data_scale > _DATA_SCALE_MIN
                          else 1.0)
            penalty_scale = data_scale ** 2
            solve_scales.append((raw_data_scale, data_scale))
            aw = aw / data_scale
            yw = yw / data_scale
            col_scale = np.sqrt(np.sum(aw**2, axis=0))
            col_scale = np.where(np.isfinite(col_scale) & (col_scale > 0), col_scale, 1.0)
            aw = aw / col_scale[None, :]

            n_par_local = int(aw.shape[1])
            p_dense_local = aw.T @ aw

            # Penalize curvature of moon-spline coefficients to suppress oscillations.
            #
            # NB these D2 penalties are brightness-dependent in effect.  After
            # the rescaling above `aw` has unit-norm columns and `yw` unit RMS,
            # so the DATA term is brightness-invariant -- but `col_scale` is
            # derived after the data_scale division, so col_scale ~ 1/data_scale
            # and a penalty built as D/col_scale carries an extra data_scale**2.
            # The effective strength of moon_smooth_lambda / zodi_smooth_lambda
            # therefore scales as flux**2.
            #
            # MEASURED, 2026-09-19, 60 rows of gaia-stars-mask-telluric-chi2
            # with FIT_PIXEL_WEIGHTS on, over the continuum solve that produces
            # the coefficients that are kept:  lambda_eff / lambda_nominal is
            # 1x at the median, 47x at p90 and 3240x at the maximum.  The
            # clamp below floors data_scale at 1, so the distribution is
            # one-sided -- lambda_nominal is not a typical strength, it is the
            # FLOOR, and 57% of rows sit exactly on it while the brightest are
            # regularised three orders of magnitude harder.  Since the bright
            # rows are the moon-up rows, the moon spline is smoothed hardest
            # where the moon signal is real and left loosest where the block is
            # a ghost pinned at the amplitude floor.
            #
            # Left as-is because it is the deployed behaviour, but it is a trap
            # for anyone calibrating these against a mixed-brightness sample.
            # `data_scale_raw` is reported out of this method so a run can
            # record its own distribution; see the `data_scale_*` META columns.
            if self.moon_smooth_lambda > 0.0 and moon_slice_local is not None:
                i0 = moon_slice_local.start or 0
                i1 = moon_slice_local.stop or i0
                n_moon = i1 - i0
                # Use cached D2 when it matches; else build on the fly (interline refit
                # solves a submatrix whose moon sub-block may not equal the cached size).
                d2 = self._d2_moon if self._d2_moon.shape == (max(n_moon - 2, 0), n_moon) else _build_d2_operator(n_moon)
                if d2.shape[0] > 0:
                    d2_scaled = d2 / col_scale[i0:i1][None, :]
                    _lam = self.moon_smooth_lambda / penalty_scale
                    p_dense_local[i0:i1, i0:i1] += 2.0 * _lam * (d2_scaled.T @ d2_scaled)

            # Zodi-spline curvature penalty (same construction; usually a heavier lambda).
            if getattr(self, 'zodi_smooth_lambda', 0.0) > 0.0 and zodi_slice_local is not None:
                i0z = zodi_slice_local.start or 0
                i1z = zodi_slice_local.stop or i0z
                n_zodi = i1z - i0z
                d2z = self._d2_zodi if self._d2_zodi.shape == (max(n_zodi - 2, 0), n_zodi) else _build_d2_operator(n_zodi)
                if d2z.shape[0] > 0:
                    d2z_scaled = d2z / col_scale[i0z:i1z][None, :]
                    _lam_z = self.zodi_smooth_lambda / penalty_scale
                    p_dense_local[i0z:i1z, i0z:i1z] += 2.0 * _lam_z * (d2z_scaled.T @ d2z_scaled)

            q_local = -(aw.T @ yw)
            p_local = sp.csc_matrix((p_dense_local + p_dense_local.T) / 2.0)
            p_local = sp.triu(p_local).tocsc()
            # --- Adjacent-knot ratio bounds on the moon / zodi spline shape ---
            # beta * c_k <= c_{k+1} <= c_k / beta on the NATIVE coefficients,
            # which bounds how fast a family's multiplier may vary with
            # wavelength and therefore forbids either family reproducing the
            # other's colour.  Unlike the D1 penalties above these are hard
            # linear inequalities, so they do not bid against the likelihood:
            # the fit is free to place the amplitude anywhere and only the shape
            # excursion is capped.  Being homogeneous in c they are also
            # scale-invariant by construction -- no brightness calibration.
            #
            # The solve runs on x = c * col_scale, so c_k = x_k / s_k and each
            # bound picks up the neighbouring col_scale ratio.  Rows are stated
            # as G x <= 0 and appended to the nonnegative cone: Clarabel solves
            # A x + s = b with s in K, so A = G, b = 0 gives s = -G x >= 0.
            #
            # NB with c >= 0 already enforced, a ratio bound makes a block
            # all-positive or all-zero: if c_k = 0 then c_{k+1} <= 0.  That is
            # physically reasonable (a family is present with a bounded shape,
            # or absent) but it does change the active-set structure that the
            # coefficient covariance of 1.9 is built from.
            ratio_rows: list[np.ndarray] = []
            for _fam_name, _sl_local, _beta in (
                ('moon', moon_slice_local, getattr(self, 'moon_ratio_bound', 0.0)),
                ('zodi', zodi_slice_local, getattr(self, 'zodi_ratio_bound', 0.0)),
            ):
                if _sl_local is None:
                    continue
                _lo = _sl_local.start or 0
                _hi = _sl_local.stop or _lo
                _n_blk = _hi - _lo
                # One beta per adjacent PAIR, so the moon bound can be relaxed
                # over a wavelength window while staying tight elsewhere.  A
                # pair is relaxed when either of its two knots is in the window.
                _betas = np.full(max(_n_blk - 1, 0), float(_beta), dtype=float)
                if _fam_name == 'moon':
                    _mask = getattr(self, '_moon_relaxed_basis', None)
                    # Guard the interline refit, which solves a submatrix whose
                    # moon sub-block may not be the full basis.
                    if (_mask is not None and _mask.size == _n_blk and _mask.any()):
                        _br = self._gated_moon_relax_bound(float(_beta))
                        for _p in range(_betas.size):
                            if _mask[_p] or _mask[_p + 1]:
                                _betas[_p] = _br
                if not np.any((_betas > 0.0) & (_betas < 1.0)):
                    continue
                for _pair_i, _k in enumerate(range(_lo, _hi - 1)):
                    _b = float(_betas[_pair_i])
                    if not (0.0 < _b < 1.0):
                        continue
                    _sk, _sk1 = col_scale[_k], col_scale[_k + 1]
                    # beta * c_k - c_{k+1} <= 0
                    _row = np.zeros(n_par_local, dtype=np.float64)
                    _row[_k] = _b / _sk
                    _row[_k + 1] = -1.0 / _sk1
                    ratio_rows.append(_row)
                    # c_{k+1} - c_k / beta <= 0
                    _row = np.zeros(n_par_local, dtype=np.float64)
                    _row[_k] = -1.0 / (_b * _sk)
                    _row[_k + 1] = 1.0 / _sk1
                    ratio_rows.append(_row)

            # --- Amplitude bracket on the moon share of the moon+zodi total ---
            # f_lo <= u / (u + v) <= f_hi with u = int(moon), v = int(zodi),
            # from the geometry prediction supplied by set_amplitude_prior.
            # Both are linear in c, so each side is one linear inequality:
            #   u/(u+v) <= f_hi   <=>   (1 - f_hi) u - f_hi v <= 0
            #   u/(u+v) >= f_lo   <=>   f_lo v - (1 - f_lo) u <= 0
            # The per-column weights are the UNWEIGHTED design integrals, so u
            # and v are the same band-integrated component fluxes the
            # diagnostics report, not ivar-weighted surrogates.
            _amp_f = getattr(self, '_amp_prior_moon_fraction', None)
            _amp_kappa = float(getattr(self, 'amp_prior_tol', 0.0))
            if (_amp_f is not None and np.isfinite(_amp_f) and _amp_kappa > 1.0
                    and moon_slice_local is not None and zodi_slice_local is not None):
                _eps = float(getattr(self, 'amp_prior_floor', 0.02))
                _f = float(np.clip(_amp_f, 0.0, 1.0))
                _f_hi = min(max(_amp_kappa * _f, _eps), 1.0 - _eps)
                _f_lo = min(max(_f / _amp_kappa, 0.0), _f_hi)
                # Column integrals over the good pixels, mapped into x-space
                # (the solve runs on x = c * col_scale).
                _col_int = np.asarray(a_mat.sum(axis=0), dtype=np.float64) / col_scale
                _m_sel = np.zeros(n_par_local, dtype=np.float64)
                _z_sel = np.zeros(n_par_local, dtype=np.float64)
                _m_sel[moon_slice_local] = _col_int[moon_slice_local]
                _z_sel[zodi_slice_local] = _col_int[zodi_slice_local]
                if np.any(_m_sel > 0.0) and np.any(_z_sel > 0.0):
                    for _row in ((1.0 - _f_hi) * _m_sel - _f_hi * _z_sel,
                                 _f_lo * _z_sel - (1.0 - _f_lo) * _m_sel):
                        _norm = float(np.max(np.abs(_row)))
                        if _norm > 0.0:
                            ratio_rows.append(_row / _norm)

            # Every row above is homogeneous (G x <= 0); the absolute zodi
            # bracket below is the only one with a right-hand side.
            ratio_rhs: list[float] = [0.0] * len(ratio_rows)

            # --- Ratio bracket on the diffuse species (2026-09-10) ----------
            # The three diffuse species are individually unidentifiable in the
            # LVM band: with the canonical PALACE HO2 vector -- a featureless
            # red riser carrying only 4.8% of its emission below 9800 A -- the
            # free fit spreads log10(FeO/HO2) over 8.7 dex, and the THREE ARMS
            # OF ONE EXPOSURE, looking at the same sky a few degrees apart,
            # disagree by 0.536 dex at the median.  Airglow does not vary by
            # x3.5 in a species ratio over 5 deg, so that spread is fitting
            # noise, and the amplitude it moves leaks into the zodi in dark
            # time, where zodi and diffuse are the only continuum players.
            #
            # A_k / A_0 <= R_hi  <=>  A_k - R_hi A_0 <= 0, linear in c, so each
            # side is one inequality -- the same construction as the moon-share
            # bracket above.  Ratios are taken against species 0 (HO2) and use
            # the same UNWEIGHTED column integrals as that bracket, so A_k is
            # the band-integrated component flux the diagnostics report.
            #
            # `diffuse_ratio_nominal` is FLUX shares, and it should come from
            # the corpus being fitted, NOT from PALACE: measured on the hybrid
            # basis, centring on PALACE's own shares costs 25.7% of the blue
            # chi2 while centring on the corpus median costs 0.67%, because
            # PALACE is calibrated for Paranal and we observe from LCO.  The
            # half-width is what the data cannot measure anyway: +/-0.2 dex
            # costs 0.07% of blue chi2 and no row above 5%, and is still wider
            # than the +/-0.12 dex seasonal variability Noll et al. (2024)
            # measure for the FeO continuum.
            _d_w = float(getattr(self, 'diffuse_ratio_bound_dex', 0.0))
            _d_nom = getattr(self, 'diffuse_ratio_nominal', None)
            if (_d_w > 0.0 and _d_nom is not None
                    and diffuse_slice_local is not None):
                _d0 = diffuse_slice_local.start or 0
                _d1 = diffuse_slice_local.stop or _d0
                _nom = np.asarray(_d_nom, dtype=np.float64).ravel()
                if (_d1 - _d0) == _nom.size and _nom.size >= 2 and np.all(_nom > 0):
                    _dcol = np.asarray(a_mat.sum(axis=0), dtype=np.float64) / col_scale
                    _W = 10.0 ** _d_w
                    _ref = _d0                      # species 0 = HO2
                    _i_ref = _dcol[_ref]
                    for _k in range(1, _nom.size):
                        _i_k = _dcol[_d0 + _k]
                        if not (_i_ref > 0.0 and _i_k > 0.0):
                            continue
                        _r = float(_nom[_k] / _nom[0])
                        for _rb, _sgn in ((_r * _W, +1.0), (_r / _W, -1.0)):
                            # +1: A_k - R_hi A_0 <= 0 ; -1: R_lo A_0 - A_k <= 0
                            _row = np.zeros(n_par_local, dtype=np.float64)
                            _row[_d0 + _k] = _sgn * _i_k
                            _row[_ref] = -_sgn * _rb * _i_ref
                            _norm = float(np.max(np.abs(_row)))
                            if _norm > 0.0:
                                ratio_rows.append(_row / _norm)
                                ratio_rhs.append(0.0)

            # --- Moon-gated upper bound on A_FeO / A_OH ---------------------
            # A_FeO <= 10**(centre + W(f)) * A_OH with A_OH a KNOWN constant
            # (OH lives in the line model, which is fixed during this solve),
            # so this is one linear inequality with a right-hand side.
            #   W(f) = bound + relax * (1 - s),  s = (f - gate) / (1 - gate)
            # clipped to [0, 1]; no bound at all for f <= gate.  See the
            # constructor for why it is one-sided and gated.
            _fo_amp = getattr(self, '_diffuse_oh_amp', None)
            _fo_c = getattr(self, 'diffuse_oh_centre_log10', None)
            _fo_w = float(getattr(self, 'diffuse_oh_bound_dex', 0.0))
            _fo_f = getattr(self, '_amp_prior_moon_fraction', None)
            if (_fo_amp is not None
                    and _fo_c is not None and _fo_w > 0.0
                    and _fo_f is not None and np.isfinite(_fo_f)
                    and diffuse_slice_local is not None):
                _g = float(getattr(self, 'diffuse_oh_gate_frac', 0.6))
                _fr = float(np.clip(_fo_f, 0.0, 1.0))
                if _fr > _g:
                    _s = np.clip((_fr - _g) / max(1.0 - _g, 1e-6), 0.0, 1.0)
                    _w_eff = _fo_w + float(getattr(self, 'diffuse_oh_relax_dex', 0.0)) * (1.0 - _s)
                    _d0 = diffuse_slice_local.start or 0
                    _d1 = diffuse_slice_local.stop or _d0
                    if (_d1 - _d0) == 3:          # HO2, FeO, O2Ac
                        _fcol = np.asarray(a_mat.sum(axis=0), dtype=np.float64) / col_scale
                        # 'block' constrains HO2 + FeO + O2Ac against OH.  That
                        # is the right scope: the +/-0.2 dex species-ratio
                        # bracket ties the three together, so an FeO-only cap
                        # is limited by that bracket's lower edge on ~40% of
                        # gated rows (occupancy rose 25.8% -> 40.5% when the
                        # FeO-only version ran).  The block ratio is also the
                        # better-behaved quantity: dark-time robust sigma
                        # 0.216 dex against FeO's 0.306, Theil-Sen slope
                        # against OH +1.16 against +1.40, and a STRONGER moon
                        # correlation (+0.716 against +0.693).
                        if str(getattr(self, 'diffuse_oh_scope', 'block')) == 'block':
                            _cols = np.arange(_d0, _d1)
                        else:
                            _cols = np.array([_d0 + 1])
                        _i_feo = float(np.sum(_fcol[_cols]))
                        if _i_feo > 0.0:
                            _row = np.zeros(n_par_local, dtype=np.float64)
                            _row[_cols] = _fcol[_cols]
                            _cap_ratio = 10.0 ** (_fo_c + _w_eff)
                            _cap = _cap_ratio * float(_fo_amp)
                            _nrm = float(np.max(np.abs(_row)))
                            if _nrm > 0.0:
                                ratio_rows.append(_row / _nrm)
                                ratio_rhs.append(_cap / _nrm)

            # --- Absolute bracket on int(zodi) against the Leinert prediction --
            #   Z_pred / kappa <= v <= kappa * Z_pred
            # stated in native flux units, so `a_mat` enters unscaled and only
            # the x-space mapping (c = x / col_scale) is applied.
            _amp_z = getattr(self, '_amp_prior_zodi_total', None)
            _z_kappa = float(getattr(self, 'zodi_amp_bound', 0.0))
            if (_amp_z is not None and np.isfinite(_amp_z) and _amp_z > 0.0
                    and _z_kappa > 1.0 and zodi_slice_local is not None):
                _zint = np.zeros(n_par_local, dtype=np.float64)
                _zcol = np.asarray(a_mat.sum(axis=0), dtype=np.float64) / col_scale
                _zint[zodi_slice_local] = _zcol[zodi_slice_local]
                _znorm = float(np.max(np.abs(_zint)))
                if _znorm > 0.0:
                    _zint = _zint / _znorm
                    _zt = float(_amp_z) / _znorm
                    # v <= kappa * Z_pred
                    ratio_rows.append(_zint.copy())
                    ratio_rhs.append(_z_kappa * _zt)
                    # -v <= -Z_pred / kappa
                    ratio_rows.append(-_zint)
                    ratio_rhs.append(-_zt / _z_kappa)

            free = np.asarray(
                [] if unconstrained_local is None else unconstrained_local,
                dtype=int,
            )
            if free.size:
                constrained_local = np.setdiff1d(
                    np.arange(n_par_local, dtype=int), free, assume_unique=True
                )
                nonnegative_rows = -sp.eye(n_par_local, format="csc")[
                    constrained_local
                ]
            else:
                constrained_local = np.arange(n_par_local, dtype=int)
                nonnegative_rows = -sp.eye(n_par_local, format="csc")
            if ratio_rows:
                g_mat = np.vstack(ratio_rows)
                a_con = sp.vstack(
                    [nonnegative_rows, sp.csc_matrix(g_mat)],
                    format="csc")
                b_con = np.concatenate([
                    np.zeros(constrained_local.size, dtype=np.float64),
                    np.asarray(ratio_rhs, dtype=np.float64)])
                cones = [clarabel.NonnegativeConeT(constrained_local.size + g_mat.shape[0])]
            else:
                a_con = nonnegative_rows
                b_con = np.zeros(constrained_local.size, dtype=np.float64)
                cones = [clarabel.NonnegativeConeT(constrained_local.size)]
            settings = clarabel.DefaultSettings()
            settings.verbose = False

            t_qp_local = time.perf_counter()
            solver = clarabel.DefaultSolver(p_local, np.asarray(q_local, dtype=np.float64), a_con, b_con, cones, settings)
            qp_result_local = solver.solve()
            retry_regularization = getattr(
                self,
                "qp_retry_static_regularization_constant",
                None,
            )
            if (
                str(qp_result_local.status) == "InsufficientProgress"
                and retry_regularization is not None
            ):
                settings.static_regularization_constant = float(retry_regularization)
                solver = clarabel.DefaultSolver(
                    p_local,
                    np.asarray(q_local, dtype=np.float64),
                    a_con,
                    b_con,
                    cones,
                    settings,
                )
                qp_result_local = solver.solve()
            qp_dt_local = time.perf_counter() - t_qp_local
            coef_local = np.asarray(qp_result_local.x, float) / col_scale
            # `p_dense_local` is the column- and data-scaled Fisher information
            # (plus any regulariser); the caller needs it, `col_scale`, and
            # `data_scale` to back out the native-coordinate covariance.
            return (
                coef_local,
                str(qp_result_local.status),
                qp_dt_local,
                p_dense_local,
                col_scale,
                float(data_scale),
            )

        (
            coef,
            status,
            qp_elapsed_sec,
            p_main,
            col_scale_main,
            data_scale_main,
        ) = _solve_nonnegative_weighted(
            a, y, base_w, moon_slice,
            zodi_slice_local=zodi_slice,
            diffuse_slice_local=diffuse_slice,
            unconstrained_local=unconstrained,
        )

        # Track whichever solve produced the FINAL value of each coefficient,
        # so uncertainties come from that solve's Hessian.
        source_p: list[np.ndarray] = [p_main] * n_par
        source_col_scale: list[np.ndarray] = [col_scale_main] * n_par
        source_data_scale: list[float] = [data_scale_main] * n_par
        source_local_index: list[int] = list(range(n_par))

        target_cols = []
        for comp_slice in (moon_slice, diffuse_slice):
            if comp_slice is None:
                continue
            i0 = comp_slice.start or 0
            i1 = comp_slice.stop or i0
            if i1 > i0:
                target_cols.extend(range(i0, i1))
        target_cols = sorted(set(target_cols))

        if boosted_w is not None and target_cols:
            target_cols_arr = np.asarray(target_cols, dtype=int)
            fixed_cols_arr = np.setdiff1d(np.arange(a.shape[1], dtype=int), target_cols_arr, assume_unique=True)
            y_target = y.copy()
            if fixed_cols_arr.size > 0:
                y_target = y_target - a[:, fixed_cols_arr] @ coef[fixed_cols_arr]

            a_target = a[:, target_cols_arr]
            unconstrained_target = np.flatnonzero(
                np.isin(target_cols_arr, unconstrained)
            )
            moon_slice_local = None
            if moon_slice is not None:
                m0 = moon_slice.start or 0
                m1 = moon_slice.stop or m0
                if m1 > m0:
                    moon_global = np.arange(m0, m1, dtype=int)
                    local_pos = np.searchsorted(target_cols_arr, moon_global)
                    in_bounds = (local_pos >= 0) & (local_pos < target_cols_arr.size)
                    local_pos = local_pos[in_bounds]
                    local_pos = local_pos[target_cols_arr[local_pos] == moon_global[in_bounds]]
                    if local_pos.size > 0:
                        moon_slice_local = slice(int(local_pos.min()), int(local_pos.max()) + 1)
            coef_target, status_target, qp_dt_target, p_target, col_scale_target, data_scale_target = _solve_nonnegative_weighted(
                a_target,
                y_target,
                boosted_w,
                moon_slice_local,
                zodi_slice_local=None,
                unconstrained_local=unconstrained_target,
            )
            coef[target_cols_arr] = coef_target
            for local_pos_i, global_col in enumerate(target_cols_arr):
                source_p[int(global_col)] = p_target
                source_col_scale[int(global_col)] = col_scale_target
                source_data_scale[int(global_col)] = data_scale_target
                source_local_index[int(global_col)] = int(local_pos_i)
            status = f"{status} | md_refit={status_target}"
            qp_elapsed_sec += qp_dt_target

        bestfit = design_matrix.T @ coef
        resid = flux - bestfit
        resid_level = -3.0 * np.nanstd(resid)
        chi2 = float(np.sum(resid[good] ** 2 * ivar[good]))
        dof = max(n_good - n_par, 1)
        reduced_chi2 = chi2 / dof
        rms_resid = float(np.sqrt(np.nanmean(resid[good] ** 2))) if n_good else np.nan
        y_mean = float(np.average(y, weights=ivar[good])) if n_good else np.nan
        sst = float(np.sum((y - y_mean) ** 2 * ivar[good])) if n_good else np.nan
        r2 = 1.0 - chi2 / sst if np.isfinite(sst) and sst > 0 else np.nan
        per_column_chi2 = self._per_column_chi2_from_residuals(
            design_matrix, resid, ivar, good, fallback=reduced_chi2,
        )
        cov_block_slices: dict[str, slice] = {}
        if moon_slice is not None and (moon_slice.stop - moon_slice.start) > 0:
            cov_block_slices["moon"] = moon_slice
        if zodi_slice is not None and (zodi_slice.stop - zodi_slice.start) > 0:
            cov_block_slices["zodi"] = zodi_slice
        _err_ret = self._coef_err_active_set(
            coef,
            source_p,
            source_col_scale,
            source_data_scale,
            source_local_index,
            reduced_chi2,
            per_column_chi2=per_column_chi2,
            cov_block_slices=cov_block_slices or None,
            unconstrained_indices=unconstrained,
        )
        if cov_block_slices:
            coef_err, cov_blocks = _err_ret
        else:
            coef_err = _err_ret
            cov_blocks = {}
        return {
            "coef": coef,
            "coef_err": coef_err,
            "coef_cov_moon": cov_blocks.get("moon"),
            "coef_cov_zodi": cov_blocks.get("zodi"),
            "status": status,
            "bestfit": bestfit,
            "resid": resid,
            "resid_level": resid_level,
            "n_good": n_good,
            "n_par": n_par,
            "chi2": chi2,
            "reduced_chi2": reduced_chi2,
            "r2": r2,
            "rms_resid": rms_resid,
            "qp_elapsed_sec": qp_elapsed_sec,
            "data_scale_raw": solve_scales[0][0] if solve_scales else np.nan,
            "data_scale_used": solve_scales[0][1] if solve_scales else np.nan,
            "data_scale_clamped": bool(solve_scales and solve_scales[0][0] < 1.0),
        }

    @staticmethod
    def _component_slices(mats: dict[str, np.ndarray]) -> dict[str, slice]:
        i0 = 0
        i1 = i0 + mats["oh"].shape[0]
        i2 = i1 + mats["moon"].shape[0]
        has_zodi = ("zodi" in mats) and (mats["zodi"].shape[0] > 0)
        i2z = i2 + (mats["zodi"].shape[0] if has_zodi else 0)
        i3 = i2z + mats["diffuse"].shape[0]
        i4 = i3 + mats["atom"].shape[0]
        i5 = i4 + mats["orc"].shape[0]
        i6 = i5 + mats["o2"].shape[0]
        out = {
            "oh": slice(i0, i1),
            "moon": slice(i1, i2),
            "diffuse": slice(i2z, i3),
            "atom": slice(i3, i4),
            "orc": slice(i4, i5),
            "o2": slice(i5, i6),
        }
        if has_zodi:
            out["zodi"] = slice(i2, i2z)
        return out

    def _components_from_coef(self, coef: np.ndarray, mats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        sl = self._component_slices(mats)
        diffuse_coef = coef[sl["diffuse"]]
        components = {
            "oh": mats["oh"].T @ coef[sl["oh"]],
            "moon": mats["moon"].T @ coef[sl["moon"]],
            "ho2": diffuse_coef[0] * self.vector_ho2,
            "feo": diffuse_coef[1] * self.vector_feo,
            "o2ac": diffuse_coef[2] * self.vector_o2ac,
            "atom": mats["atom"].T @ coef[sl["atom"]],
            "orc": mats["orc"].T @ coef[sl["orc"]],
            "o2": mats["o2"].T @ coef[sl["o2"]],
        }
        components["diffuse"] = components["ho2"] + components["feo"] + components["o2ac"]
        if "zodi" in sl:
            components["zodi"] = mats["zodi"].T @ coef[sl["zodi"]]
        return components

    def _components_sigma_from_coef_err(
        self,
        coef_err: np.ndarray,
        mats: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Propagate per-coefficient 1σ to per-component flux 1σ.

        For every matrix component (``oh``, ``moon``, ``atom``, ``orc``,
        ``o2``) the flux variance at wavelength λ is

            σ²_comp(λ) = Σ_j M[j, λ]² · σ_c[j]²                    (1)

        where ``M`` is the component's design matrix (already
        LSF-convolved by ``_assemble_refined_matrices``) and ``σ_c`` is the
        native-space per-coefficient uncertainty. This is the exact
        first-order (Jacobian) propagation because the reconstruction
        ``flux = M.T @ coef`` is linear in the coefficients.

        Implementation note. The sum is evaluated as ``(M*M).T @ σ²_c`` which
        dispatches to BLAS gemv on the squared matrix. Micro-benchmarks show
        this is faster than ``np.einsum('ij,ij,i->j', M, M, σ²_c)`` for
        typical shapes (n_coef ~ 400, n_pix ~ 3500) because einsum falls back
        to a general-purpose Python-side path for 3-tensor contractions
        instead of the tuned BLAS kernel.

        The diffuse sub-component uncertainties use the rows of the supplied
        diffuse design matrix. This keeps uncertainty propagation tied to the
        same final LSF-aware matrix bundle as the reconstructed mean and also
        supports reduced synthetic matrices used by workflow tests.

        NaN or non-finite ``coef_err`` values are treated as zero
        contribution, matching the "no measured uncertainty available"
        semantics used elsewhere in the pipeline.
        """
        sl = self._component_slices(mats)
        err = np.asarray(coef_err, dtype=np.float64).ravel()
        if err.size != sum(m.shape[0] for m in mats.values()):
            raise ValueError(
                f"coef_err length mismatch: expected "
                f"{sum(m.shape[0] for m in mats.values())}, got {err.size}"
            )
        err2 = np.where(np.isfinite(err), err ** 2, 0.0)

        def _matrix_sigma(name: str) -> np.ndarray:
            m = mats[name]
            # (m*m).T @ vec dispatches to BLAS gemv on the squared matrix.
            var = (m * m).T @ err2[sl[name]]
            return np.sqrt(np.maximum(var, 0.0))

        diffuse_err2 = err2[sl["diffuse"]]
        diffuse_matrix = mats["diffuse"]

        def _diffuse_row_sigma(index: int) -> np.ndarray:
            if index >= diffuse_matrix.shape[0]:
                return np.zeros(diffuse_matrix.shape[1], dtype=np.float64)
            return np.sqrt(diffuse_err2[index]) * np.abs(diffuse_matrix[index])

        sigmas = {
            "oh": _matrix_sigma("oh"),
            "moon": _matrix_sigma("moon"),
            "ho2": _diffuse_row_sigma(0),
            "feo": _diffuse_row_sigma(1),
            "o2ac": _diffuse_row_sigma(2),
            "diffuse": _matrix_sigma("diffuse"),
            "atom": _matrix_sigma("atom"),
            "orc": _matrix_sigma("orc"),
            "o2": _matrix_sigma("o2"),
        }
        if "zodi" in sl:
            sigmas["zodi"] = _matrix_sigma("zodi")
        return sigmas

    def _build_lsf_source(self, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zodi_bundle = self.matrix_zodi_hr if (self.split_zodi and self.matrix_zodi.shape[0] > 0) else None
        mats = self._matrix_bundle(
            self.matrix_oh_stick,
            self.matrix_moon_hr,
            self.matrix_diffuse,
            self.matrix_atom_stick,
            self.matrix_orc_stick,
            self.matrix_o2_stick,
            matrix_zodi=zodi_bundle,
        )
        sl = self._component_slices(mats)
        emission = (
            mats["oh"].T @ coef[sl["oh"]]
            + mats["atom"].T @ coef[sl["atom"]]
            + mats["orc"].T @ coef[sl["orc"]]
            + mats["o2"].T @ coef[sl["o2"]]
        )
        moon = mats["moon"].T @ coef[sl["moon"]]
        if "zodi" in sl:
            moon = moon + mats["zodi"].T @ coef[sl["zodi"]]
        fixed_background = mats["diffuse"].T @ coef[sl["diffuse"]]
        return emission + moon, fixed_background

    def _assemble_refined_matrices(self) -> dict[str, np.ndarray]:
        zodi_hr = None
        if self.split_zodi and self.matrix_zodi.shape[0] > 0:
            zodi_hr = self._convolve_matrix_channelwise(self.matrix_zodi_hr)
        return self._matrix_bundle(
            self._convolve_matrix_channelwise(self.matrix_oh_stick),
            self._convolve_matrix_channelwise(self.matrix_moon_hr),
            self.matrix_diffuse,
            self._convolve_matrix_channelwise(self.matrix_atom_stick),
            self._convolve_matrix_channelwise(self.matrix_orc_stick),
            self._convolve_matrix_channelwise(self.matrix_o2_stick),
            matrix_zodi=zodi_hr,
        )

    def _build_static_basis(self) -> None:
        self.matrix_oh = self._build_oh()
        self.matrix_atom, self.atom_names = self._build_atom()
        self.matrix_orc, self.orc_names = self._build_orc()
        self.o2_names = ["O2_b01"]
        self._load_o2_model()
        self.matrix_moon, self.moon_names = self._build_moon()
        self.matrix_zodi, self.zodi_names = self._build_zodi()
        self.matrix_diffuse, self.diffuse_names = self._build_diffuse()
        self.design_names = (
            [f"OH_{i:03d}" for i in range(self.matrix_oh.shape[0])]
            + self.moon_names
            + self.zodi_names
            + self.diffuse_names
            + self.atom_names
            + self.orc_names
            + self.o2_names
        )
        self.design_matrix = self._assemble_design_matrix()

    def _assemble_design_matrix(self) -> np.ndarray:
        parts = [self.matrix_oh, self.matrix_moon]
        if self.split_zodi and self.matrix_zodi.shape[0] > 0:
            parts.append(self.matrix_zodi)
        parts.extend([self.matrix_diffuse, self.matrix_atom, self.matrix_orc, self.matrix_o2])
        return np.vstack(parts)

    def _build_oh(self) -> np.ndarray:
        # The catalog is cached per amplitude RULE, so the rule has to be
        # identifiable without an instance.  Every subclass states it as a
        # staticmethod; refuse anything else rather than silently calling an
        # unbound method with the group as its `self`.
        amplitude = inspect.getattr_static(type(self), "_oh_amplitude")
        if not isinstance(amplitude, staticmethod):
            raise TypeError(
                f"{type(self).__name__}._oh_amplitude must be a staticmethod"
            )
        groups = _oh_line_catalog(
            str(self._pmd_path("pmd_popmodel_OH.dat")),
            float(self.wave.min()),
            float(self.wave.max()),
            tuple(self.oh_group_keys),
            amplitude.__func__,
        )
        matrix = np.zeros((len(groups), self.wave.size))
        self.matrix_oh_stick = np.zeros_like(matrix)
        self._oh_line_groups = []
        for idx, (line_wave, amp) in enumerate(groups):
            self._oh_line_groups.append((line_wave.copy(), amp.copy()))
            matrix[idx] = grp2vector(line_wave, amp, self.wave, self.lsf_sigma)
            self.matrix_oh_stick[idx] = sticks2vector(line_wave, amp, self.wave)
        return matrix

    @staticmethod
    def _oh_amplitude(group: Table) -> np.ndarray:
        return np.asarray(group["Aij"] * group["gi"], dtype=float)

    def _build_moon(self) -> tuple[np.ndarray, list[str]]:
        solar_hr, solar_rb = _solar_templates(
            self._require_path(self.solar_path), self.wave, self.lsf_sigma
        )
        self.vector_moon = solar_rb

        w0, w1 = self.wave[0], self.wave[-1]
        interior = self._uniform_moon_knots(w0, w1, self.n_spline_knots)

        self.moon_knots_used = interior.copy()

        t_knots = np.r_[(w0,) * 4, interior, (w1,) * 4]
        matrix_bspl = BSpline.design_matrix(self.wave, t_knots, 3).toarray()
        # Colour envelope: solar × moon_albedo × Rayleigh when split_zodi, else
        # solar alone.
        #
        # The Rayleigh factor is what makes a shape prior possible at all.
        # Scattered moonlight is Rayleigh-dominated and therefore steeply blue
        # (the frozen physical model gives log-log slope ~-3.7), while
        # solar × albedo alone is nearly flat (~+0.2).  Without the factor the
        # 15 spline knots have to manufacture the entire lambda^-4 tilt
        # themselves: reproducing the physical moon then needs a 50x
        # coefficient swing, ||D1 c||/mean ~ 2.0, which no tight prior can
        # coexist with.  With it, the same target needs a 1.4x swing and
        # ||D1 c||/mean ~ 0.15-0.24 (0.58 worst case over airmass), so the
        # spline is back to being a mild correction and D1 can be pinned.
        # Deliberately NOT airmass-dependent: the moon's own airmass response
        # lives in the scattering source term, and an extra line-of-sight
        # extinction factor here double-counts it (measured: the requirement
        # rises from 0.18 to 0.51 at X=2.8 when applied).
        if getattr(self, 'split_zodi', False):
            self.moon_albedo_shape = self._load_moon_albedo_shape()
            self.moon_rayleigh_shape = _unit_median(_rayleigh_tau(self.wave))
            # Base = the geometry-free part.  `set_moon_geometry` multiplies in
            # the two-channel scattering shape when moon_scatter_envelope is on;
            # otherwise the Rayleigh-only shape below is used, unchanged.
            self._moon_envelope_base = solar_rb * self.moon_albedo_shape
            self._moon_envelope_base_hr = solar_hr * self.moon_albedo_shape
            self._moon_bspl = matrix_bspl
            shape = (self.moon_scatter_shape
                     if getattr(self, 'moon_scatter_shape', None) is not None
                     else self.moon_rayleigh_shape)
            envelope = self._moon_envelope_base * shape
            envelope_hr = self._moon_envelope_base_hr * shape
        else:
            envelope = solar_rb
            envelope_hr = solar_hr
        matrix_moon = (envelope[:, None] * matrix_bspl).T
        self.matrix_moon_hr = (envelope_hr[:, None] * matrix_bspl).T
        moon_names = [f"Moon_bs{i:02d}" for i in range(matrix_moon.shape[0])]
        # Precompute the moon penalty operators once (depend only on n_par).
        self._d2_moon = _build_d2_operator(matrix_moon.shape[0])
        self._moon_full_knots = np.asarray(t_knots, dtype=float)
        self._moon_relaxed_basis = self._moon_relaxed_mask(matrix_moon.shape[0])
        return matrix_moon, moon_names

    def _gated_moon_relax_bound(self, beta_tight: float) -> float:
        """Blue-relaxation bound for this row, gated on moon dominance.

        Returns ``moon_ratio_relax_bound`` unchanged when no gate is
        configured or no prediction is installed.  With a gate, interpolates
        from ``beta_tight`` (no extra freedom, faint moon) to the relax bound
        (full freedom, moon dominates) across the ``moon_relax_gate`` window in
        predicted moon fraction.
        """
        relax = float(getattr(self, 'moon_ratio_relax_bound', 0.0))
        gate = getattr(self, 'moon_relax_gate', None)
        frac = getattr(self, '_amp_prior_moon_fraction', None)
        if gate is None or frac is None or not np.isfinite(frac):
            return relax
        lo, hi = gate
        span = hi - lo
        s = 1.0 if span <= 0.0 else float(np.clip((float(frac) - lo) / span, 0.0, 1.0))
        return float(beta_tight + (relax - beta_tight) * s)

    def _moon_relaxed_mask(self, n_basis: int) -> np.ndarray:
        """Moon basis functions positioned inside the relaxation window.

        Position is the Greville abscissa, ``xi_j = (t[j+1]+t[j+2]+t[j+3])/3``,
        which is where cubic basis function ``j`` actually sits.  Selecting on
        SUPPORT overlap instead would be far too blunt: with 11 interior knots
        over 3600-9800 A each support spans ~2070 A, so a 3800-4400 A window
        would drag in 5 of 15 knots reaching out to 6183 A.  On the Greville
        abscissa the same window selects the one knot that belongs to it.
        """
        win = getattr(self, 'moon_ratio_relax_window', None)
        knots = getattr(self, '_moon_full_knots', None)
        mask = np.zeros(int(n_basis), dtype=bool)
        if win is None or knots is None:
            return mask
        lo, hi = float(win[0]), float(win[1])
        for j in range(int(n_basis)):
            if j + 3 < knots.size:
                xi = float(knots[j + 1] + knots[j + 2] + knots[j + 3]) / 3.0
                mask[j] = (lo <= xi <= hi)
        return mask

    def _load_moon_albedo_shape(self) -> np.ndarray:
        """Interpolate ROLO lunar albedo at the fiducial phase onto the fit grid."""
        path = self.moon_albedo_asset_path
        with pathlib.Path(path).open('r', encoding='utf-8') as fh:
            lines = [ln.rstrip() for ln in fh if ln.strip() and not ln.startswith('#')]
        constants = np.fromstring(lines[0], sep=' ')
        n_lam = int(lines[1])
        coefs = np.array([np.fromstring(ln, sep=' ') for ln in lines[2:2 + n_lam]])
        phase_deg_abs = abs(self.moon_albedo_fiducial_phase_deg)
        phase_rad = np.deg2rad(phase_deg_abs)
        signed_rad = np.deg2rad(self.moon_albedo_fiducial_phase_deg)
        signed_lim = (signed_rad if phase_deg_abs < 97.0
                      else 97.0 * signed_rad / phase_deg_abs)
        v = coefs[:, 1:]
        poly = (v[:, 0] + v[:, 1] * phase_rad + v[:, 2] * phase_rad**2 + v[:, 3] * phase_rad**3
                + v[:, 4] * signed_lim + v[:, 5] * signed_lim**3 + v[:, 6] * signed_lim**5)
        opp = (v[:, 7] * np.exp(-phase_deg_abs / constants[0])
               + v[:, 8] * np.exp(-phase_deg_abs / constants[1])
               + v[:, 9] * np.cos((phase_deg_abs - constants[2]) / constants[3]))
        tabulated = np.exp(poly + opp) / 0.87
        wave_nm = self.wave / 10.0
        albedo = np.interp(wave_nm, coefs[:, 0], tabulated)
        # Normalise to unit median so Moon_bs coefficient magnitudes
        # stay comparable to the current (no-albedo) fit-time scale.
        med = float(np.nanmedian(albedo))
        return (albedo / med).astype(float) if np.isfinite(med) and med > 0 else albedo

    def _zodi_color_shape_vec(self) -> np.ndarray:
        """Parametric zodi reddening: (λ/5000Å)^exponent, normalised to unit median."""
        shape = (self.wave / 5000.0) ** self.zodi_color_exponent
        med = float(np.nanmedian(shape))
        return (shape / med).astype(float) if np.isfinite(med) and med > 0 else shape

    def _build_zodi(self) -> tuple[np.ndarray, list[str]]:
        """Build Zodi_bs family: solar_rb × zodi_color × B-spline(K_zodi).

        Called only when `split_zodi=True`.  Uses the same solar template as the moon
        family (already computed in `_build_moon`), so must be invoked AFTER `_build_moon`.
        """
        if not self.split_zodi:
            return np.zeros((0, self.wave.size), dtype=float), []
        # Solar template is stored on self.vector_moon (populated in _build_moon).
        solar_rb = self.vector_moon
        self.zodi_color_shape = self._zodi_color_shape_vec()
        w0, w1 = self.wave[0], self.wave[-1]
        interior = self._uniform_moon_knots(w0, w1, self.n_zodi_spline_knots)
        self.zodi_knots_used = interior.copy()
        t_knots = np.r_[(w0,) * 4, interior, (w1,) * 4]
        matrix_bspl = BSpline.design_matrix(self.wave, t_knots, 3).toarray()
        envelope = solar_rb * self.zodi_color_shape
        # Airmass-free base, kept so ``set_target_airmass`` can re-derive the
        # attenuated matrices per row without rebuilding the B-spline basis.
        self._zodi_bspl = matrix_bspl
        self._zodi_envelope_base = envelope
        matrix_zodi = (envelope[:, None] * matrix_bspl).T
        # HR twin: identical for now; LSF refit convolves it channelwise if used.
        self.matrix_zodi_hr = matrix_zodi.copy()
        zodi_names = [f'Zodi_bs{i:02d}' for i in range(matrix_zodi.shape[0])]
        # Precompute the zodi penalty operators (depend only on n_par).
        self._d2_zodi = _build_d2_operator(matrix_zodi.shape[0])
        return matrix_zodi, zodi_names

    def _zodi_extinction_shape(self, target_airmass: float) -> np.ndarray:
        """Median-normalised zodi attenuation shape at one target airmass.

        Zodiacal light is attenuated along the line of sight, so unlike the
        moon's Rayleigh factor this one genuinely depends on airmass, and the
        dependence is large enough to matter: without it the spline shape needed
        to reproduce the physical zodi grows from ||D1 c||/mean = 0.16 at X=1 to
        0.40 at X=2.8, and a prior tight enough to forbid the role swap would
        clip that real extinction colour instead.  With it the requirement is
        0.15-0.18 flat across the whole airmass range.

        The half optical depth follows the frozen physical model
        (``exp(-X (0.5 tau_ray + 0.5 tau_aer))``) and the extended-source
        argument in the notebook's 2.5.2: photons scattered out of the beam are
        largely replaced from adjacent lines of sight, so a stellar curve
        over-corrects.  Target airmass is clipped to the physical range [1, 3];
        the moon's own (unbounded) airmass is not used here.
        """
        x = float(target_airmass)
        if not np.isfinite(x):
            return np.ones_like(self.wave, dtype=float)
        x = min(max(x, 1.0), 3.0)
        tau_half = 0.5 * (_rayleigh_tau(self.wave) + _aerosol_tau(self.wave))
        return _unit_median(np.exp(-(x - 1.0) * tau_half))

    def set_target_airmass(self, target_airmass: float | None) -> None:
        """Install the per-row zodi extinction envelope (split_zodi only).

        Call before ``fit`` for each row.  ``None`` restores the airmass-free
        envelope.  Rebuilds the design matrix, mirroring what the Moon/Zodi
        variant does after installing its per-row prediction.
        """
        if not getattr(self, 'split_zodi', False):
            return
        if getattr(self, '_zodi_envelope_base', None) is None:
            return
        self.target_airmass = (None if target_airmass is None
                               else float(target_airmass))
        shape = (np.ones_like(self.wave, dtype=float) if target_airmass is None
                 else self._zodi_extinction_shape(target_airmass))
        self.zodi_extinction_shape = shape
        envelope = self._zodi_envelope_base * shape
        self.matrix_zodi = (envelope[:, None] * self._zodi_bspl).T
        self.matrix_zodi_hr = self.matrix_zodi.copy()
        self.design_matrix = self._assemble_design_matrix()

    def _moon_scatter_shape(self, moon_sep_deg: float, moon_airmass: float,
                            target_airmass: float) -> np.ndarray:
        """Two-channel scattered-moonlight colour shape, unit-median normalised.

        Reproduces the frozen model's moon carrier wavelength dependence:

            tau_R * P_Rayleigh(cos t) + 0.97 * tau_a * P_HG(cos t)   (channels)
          * (1 + c * X_t * tau_R) / (1 + c * X_t * tau_R(5000))      (multiple
                                                                      scattering)
          * exp(-tau_tot * X_t) * phi(tau_tot * (X_m - X_t))         (path)

        Three things the Rayleigh-only envelope it replaces cannot express:

        * The AEROSOL channel.  tau_a ~ lambda^-1.38 against Rayleigh's
          lambda^-4, so the mixture is much redder than Rayleigh alone, and the
          mixing ratio is set by the two phase functions at the moon-target
          separation.  Henyey-Greenstein (g = 0.8) is strongly forward-peaked,
          so close to the moon the red aerosol channel dominates while at large
          separation Rayleigh does.  The fitted moon colour therefore has to
          vary with separation, which a fixed envelope cannot do -- and the
          worst single fit in the 200-spectrum sample (row 12, both telescopes,
          rms 20x the sample median) sits at 18-25 deg separation, exactly
          where the missing channel matters most.
        * MULTIPLE SCATTERING, the (1 + c X tau_R) enhancement.  Modest at LCO
          (~9% in the blue at X = 1.5) but the right sign and shape.
        * The atmospheric path terms, which carry their own mild colour.

        Measured motivation: with a Rayleigh-only envelope the fit drives the
        moon to log-log slope -4.1 (tighter zodi bounds push it to -5.6)
        against the -3.7 the model predicts, i.e. bluer than physics, which is
        what a missing red channel looks like.  The cost is concentrated where
        the moon dominates: rho(rms cost, FLI) = +0.91, and +49% rms at
        FLI > 0.8 against +1.5% below 0.6.
        """
        cos_t = float(np.cos(np.deg2rad(float(moon_sep_deg))))
        x_t = float(np.clip(target_airmass, 1.0, 3.0))
        x_m = float(moon_airmass) if np.isfinite(moon_airmass) else x_t
        # The moon itself may sit at high airmass (unlike the telescope), but
        # cap it so a moon at/below the horizon cannot produce an absurd path.
        x_m = float(np.clip(x_m, 1.0, 40.0))
        tau_r = _rayleigh_tau(self.wave)
        tau_a = _aerosol_tau(self.wave)
        tau_tot = tau_r + tau_a
        channels = (tau_r * _rayleigh_phase(cos_t)
                    + 0.97 * tau_a * _henyey_greenstein_phase(cos_t))
        c_ms = float(getattr(self, 'moon_ms_coeff', 3.5))
        tau_r_500 = float(_rayleigh_tau(np.asarray([5000.0]))[0])
        multiple = ((1.0 + c_ms * x_t * tau_r)
                    / (1.0 + c_ms * x_t * tau_r_500))
        path = np.exp(-tau_tot * x_t) * _stable_phi(tau_tot * (x_m - x_t))
        return _unit_median(channels * multiple * path)

    def set_moon_geometry(self, moon_sep_deg: float | None,
                          moon_airmass: float | None = None,
                          target_airmass: float | None = None) -> None:
        """Install the per-row moon scattering envelope (split_zodi only).

        Active only when ``moon_scatter_envelope=True``; ``None`` for
        ``moon_sep_deg`` restores the geometry-free Rayleigh-only envelope.
        Call before ``fit`` for each row.  Rebuilds the moon block and the
        design matrix, so it is the expensive setter -- unlike
        ``set_amplitude_prior``, which only stores scalars.

        ``target_airmass`` defaults to whatever ``set_target_airmass`` last
        installed, so the usual per-row order is ``set_target_airmass`` then
        ``set_moon_geometry``.
        """
        if not getattr(self, 'split_zodi', False):
            return
        if not getattr(self, 'moon_scatter_envelope', False):
            return
        if getattr(self, '_moon_envelope_base', None) is None:
            return
        if moon_sep_deg is None or not np.isfinite(moon_sep_deg):
            self.moon_scatter_shape = None
            shape = self.moon_rayleigh_shape
        else:
            if target_airmass is None:
                target_airmass = self.target_airmass
            if target_airmass is None or not np.isfinite(target_airmass):
                target_airmass = 1.0
            if moon_airmass is None or not np.isfinite(moon_airmass):
                moon_airmass = float(target_airmass)
            shape = self._moon_scatter_shape(moon_sep_deg, moon_airmass,
                                             target_airmass)
            self.moon_scatter_shape = shape
        self.matrix_moon = (
            (self._moon_envelope_base * shape)[:, None] * self._moon_bspl).T
        self.matrix_moon_hr = (
            (self._moon_envelope_base_hr * shape)[:, None] * self._moon_bspl).T
        self.design_matrix = self._assemble_design_matrix()

    def set_diffuse_oh_reference(self, oh_amplitude: float | None) -> None:
        """Install this row's band-integrated OH amplitude for the FeO bound.

        ``oh_amplitude`` must be on the same flux scale as the array passed to
        ``fit`` (the caller multiplies by ``physical_to_fit_flux_scale``), and
        is the integral of the CURRENT line model's OH block, so it has to be
        re-installed on every continuum solve as the LSF refines.  ``None``
        clears the bound.  Takes effect only when ``diffuse_oh_bound_dex > 0`` and
        ``diffuse_oh_centre_log10`` is set; the moon gate reads the fraction
        already installed by :meth:`set_amplitude_prior`, so no extra
        per-row plumbing is needed.
        """
        if oh_amplitude is None or not np.isfinite(oh_amplitude) or oh_amplitude <= 0:
            self._diffuse_oh_amp = None
        else:
            self._diffuse_oh_amp = float(oh_amplitude)

    def set_amplitude_prior(self, moon_fraction: float | None,
                            zodi_total: float | None = None) -> None:
        """Install the per-row amplitude prior centres (split_zodi only).

        ``moon_fraction`` is the geometry-predicted
        ``int(moon) / int(moon + zodi)`` over the fitted band; ``None`` (the
        default state) leaves the amplitudes unconstrained.  Call before
        ``fit`` for each row.  Takes effect only when ``amp_prior_tol > 1``.

        ``zodi_total`` is the predicted ``int(zodi)`` for the absolute Leinert
        bracket, active only when ``zodi_amp_bound > 1``.  It must be on the
        same flux scale as the array passed to ``fit`` -- pass the same
        ``physical_to_fit_flux_scale`` to ``predict`` as the FACTOR the flux
        was multiplied by.  The fraction above is calibration-free; this is
        not.

        The caller owns the prediction so this module stays free of ephemeris
        and coordinate machinery: build a ``MoonZodiObservation`` and take
        ``sum(pred.moon) / sum(pred.moon + pred.zodi)`` from
        ``MoonZodiPhysicalModel.predict``.  Only the ratio is used, so the
        prediction's absolute normalisation is irrelevant -- but its moon and
        zodi must come from the SAME predict() call, or the shared conversion
        factor no longer cancels.

        Unlike ``set_target_airmass`` this does not touch the design matrix, so
        it is cheap to call per row.
        """
        if not getattr(self, 'split_zodi', False):
            return
        self._amp_prior_moon_fraction = (
            None if moon_fraction is None or not np.isfinite(moon_fraction)
            else float(np.clip(moon_fraction, 0.0, 1.0)))
        self._amp_prior_zodi_total = (
            None if zodi_total is None or not np.isfinite(zodi_total)
            or float(zodi_total) <= 0.0 else float(zodi_total))

    @staticmethod
    def _uniform_moon_knots(w0: float, w1: float, n_knots: int) -> np.ndarray:
        n_knots = max(int(n_knots), 0)
        if n_knots == 0:
            return np.array([], float)
        return np.linspace(float(w0), float(w1), n_knots + 2)[1:-1]

    def _moon_interline_weights(self) -> np.ndarray:
        line_signal = self._line_density_vector()
        finite = np.isfinite(line_signal)
        positive = finite & (line_signal > 0)
        line_signal_norm = np.zeros_like(line_signal, dtype=float)
        if np.any(positive):
            sig_max = float(np.nanmax(line_signal[positive]))
            if np.isfinite(sig_max) and sig_max > 0:
                line_signal_norm[positive] = line_signal[positive] / sig_max
        # Use a strict binary mask around detected lines; masked pixels are
        # down-weighted while unmasked pixels keep nominal weight.
        raw_line_mask = finite & (line_signal_norm > self.moon_interline_line_flux_threshold)
        line_mask = raw_line_mask.copy()
        if self.moon_interline_exclusion_a > 0 and self.wave.size > 2:
            dw = np.gradient(self.wave)
            dw_med = max(float(np.nanmedian(np.abs(dw))), 1e-6)
            n_pix = int(np.ceil(self.moon_interline_exclusion_a / dw_med))
            if n_pix > 0:
                ker = np.ones(2 * n_pix + 1, dtype=int)
                line_mask = np.convolve(raw_line_mask.astype(int), ker, mode="same") > 0

        red_mask = self.wave >= self.moon_interline_red_min
        masked = line_mask & red_mask

        # Reuse the existing knob: larger values mean stronger suppression of
        # masked pixels, with floor weight 1 / (1 + boost).
        w = np.ones_like(self.wave, dtype=float)
        if self.moon_interline_boost > 0.0:
            w[masked] = 1.0 / (1.0 + self.moon_interline_boost)
        return w

    def _line_density_vector(self) -> np.ndarray:
        out = np.zeros_like(self.wave, dtype=float)
        for name in ("matrix_oh_stick", "matrix_atom_stick", "matrix_orc_stick"):
            mat = getattr(self, name, None)
            if mat is not None and np.size(mat) > 0:
                out += np.nansum(np.abs(np.asarray(mat, float)), axis=0)

        o2_stick = getattr(self, "vector_o2_stick", None)
        if o2_stick is not None and np.size(o2_stick) > 0:
            out += np.abs(np.asarray(o2_stick, float))
        return out

    def _build_diffuse(self) -> tuple[np.ndarray, list[str]]:
        ref = read_static_table(self._pmd_path("pmd_refcont.dat"), "ascii")
        lam_ref = np.asarray(ref["lam"], float) * 1e4
        exact_native_grid = (
            lam_ref.shape == self.wave.shape
            and np.max(np.abs(lam_ref - self.wave)) <= 1.0e-10
        )
        if exact_native_grid:
            vectors = [np.asarray(ref[name], float) for name in ("fcHO2", "fcFeO", "fcO2Ac")]
        else:
            vectors = [
                np.interp(self.wave, lam_ref, np.asarray(ref[name], float))
                for name in ("fcHO2", "fcFeO", "fcO2Ac")
            ]
        self.vector_ho2, self.vector_feo, self.vector_o2ac = [
            np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            for vector in vectors
        ]

        matrix = np.vstack([self.vector_ho2, self.vector_feo, self.vector_o2ac])
        return matrix, ["HO2", "FeO", "O2Ac"]

    def _build_atom(self) -> tuple[np.ndarray, list[str]]:
        groups = _atom_line_catalog(
            str(self._pmd_path("pmd_intdata_atom.dat")),
            float(self.wave.min()),
            float(self.wave.max()),
        )
        matrix = np.zeros((len(groups), self.wave.size))
        self.matrix_atom_stick = np.zeros_like(matrix)
        self._atom_line_groups = []
        names = []
        for idx, (line_wave, amp, name) in enumerate(groups):
            self._atom_line_groups.append((line_wave.copy(), amp.copy()))
            matrix[idx] = grp2vector(line_wave, amp, self.wave, self.lsf_sigma)
            self.matrix_atom_stick[idx] = sticks2vector(line_wave, amp, self.wave)
            names.append(name)
        return matrix, names

    def _build_orc(self) -> tuple[np.ndarray, list[str]]:
        groups = _orc_line_catalog(
            str(self._pmd_path("pmd_intmodel_Orc.dat")),
            float(self.wave.min()),
            float(self.wave.max()),
        )
        matrix = np.zeros((len(groups), self.wave.size))
        self.matrix_orc_stick = np.zeros_like(matrix)
        self._orc_line_groups = []
        names = []
        for idx, (line_wave, amp, name) in enumerate(groups):
            self._orc_line_groups.append((line_wave.copy(), amp.copy()))
            matrix[idx] = grp2vector(line_wave, amp, self.wave, self.lsf_sigma)
            self.matrix_orc_stick[idx] = sticks2vector(line_wave, amp, self.wave)
            names.append(name)
        return matrix, names

    def _load_o2_model(self) -> None:
        pop_o2 = read_static_table(self._pmd_path("pmd_popmodel_O2.dat"))
        pop_o2["wave"] = vac_to_air(np.asarray(pop_o2["lam"], float) * 1e4)
        o2 = pop_o2[
            (pop_o2["wave"] >= O2_MIN)
            & (pop_o2["wave"] <= O2_MAX)
            & (np.asarray(pop_o2["vi"], int) == 0)
        ]
        self.lam_o2 = np.asarray(o2["wave"], float)
        self.ei_o2 = np.asarray(o2["Ei"], float)
        self.aij_o2 = np.asarray(o2["Aij"], float)
        self.gi_o2 = np.asarray(o2["gi"], float)
        self.e0_o2 = float(np.nanmin(self.ei_o2)) if self.ei_o2.size else 0.0
        self.o2_band = (self.wave >= O2_MIN) & (self.wave <= O2_MAX)

    def _prefit_o2(self, flux: np.ndarray, ivar: np.ndarray) -> None:
        if not np.any(self.o2_band) or self.lam_o2.size == 0:
            self.t_o2 = T_O2_REF
            self.t_o2_err = np.nan
            self.o2_prefit_amp = np.nan
            self.vector_o2 = np.zeros_like(self.wave)
            self.matrix_o2 = self.vector_o2[None, :]
            self.vector_o2_stick = np.zeros_like(self.wave)
            self.matrix_o2_stick = self.vector_o2_stick[None, :]
            self.o2_prefit_bestfit = self.vector_o2.copy()
            self.o2_fit_status = "fallback"
            self.o2_fit_summary = "O2 prefit | status=fallback | reason=no_o2_band"
            self.o2_fit_elapsed_sec = 0.0
            self.o2_valid_frac = 0.0
            return

        x = self.wave[self.o2_band]
        y = flux[self.o2_band]
        iv = ivar[self.o2_band]
        moon = self.vector_moon[self.o2_band]
        base_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(iv) & (iv > 0) & np.isfinite(moon)
        min_valid = int(np.ceil(self.o2_min_valid_frac * x.size))
        self.o2_valid_frac = float(np.count_nonzero(base_valid) / x.size) if x.size else 0.0
        xv = x[base_valid]
        yv = y[base_valid]
        ivv = iv[base_valid]
        mv = moon[base_valid]
        x0 = float(np.nanmean(xv)) if xv.size else 0.0
        xs = max(float(np.nanmax(xv) - np.nanmin(xv)), 1.0) if xv.size else 1.0
        sqrt_iv = np.sqrt(ivv) if ivv.size else np.array([], float)
        xlin = (xv - x0) / xs if xv.size else np.array([], float)

        def o2_shape(temp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            rel = self.aij_o2 * self.gi_o2 * np.exp(-HC_OVER_KB_CMK * (self.ei_o2 - self.e0_o2) / temp)
            rel_sum = np.nansum(rel)
            if not np.isfinite(rel_sum) or rel_sum <= 0:
                return np.zeros_like(self.wave), np.zeros_like(self.wave), np.zeros_like(xv)
            rel /= rel_sum
            stick_shape = sticks2vector(self.lam_o2, rel, self.wave)
            full_shape = grp2vector(self.lam_o2, rel, self.wave, self.lsf_sigma)
            return full_shape, stick_shape, full_shape[self.o2_band][base_valid]

        def linear_init(temp: float) -> np.ndarray:
            _, _, sv = o2_shape(temp)
            if sv.size < 3:
                return np.array([0.0, 0.0, 0.0], float)
            design = np.c_[sv, mv, mv * xlin]
            aw = design * sqrt_iv[:, None]
            coef, _, _, _ = np.linalg.lstsq(aw, yv * sqrt_iv, rcond=None)
            return coef

        def set_o2_summary(status: str, dt: float, chi2: float, par: np.ndarray, err: np.ndarray, nvalid: int) -> None:
            dof = max(nvalid - 4, 1)
            chi2_red = chi2 / dof if np.isfinite(chi2) else np.nan
            y_mean = float(np.average(yv, weights=ivv)) if nvalid else np.nan
            sst = float(np.sum((yv - y_mean) ** 2 * ivv)) if nvalid else np.nan
            r2 = 1.0 - chi2 / sst if np.isfinite(sst) and sst > 0 else np.nan
            self.o2_fit_summary = (
                "O2 prefit | "
                f"status={status} | nvalid={nvalid}/{x.size} ({self.o2_valid_frac:.1%}) | "
                f"T={self.t_o2:.2f}+/-{self.t_o2_err:.2f} K | "
                f"chi2_red={chi2_red:.4g} | R2={r2:.5f} | dt={dt:.3f}s"
            )

        init_coef = linear_init(T_O2_REF)

        if np.sum(base_valid) < max(min_valid, 3):
            full_shape, stick_shape, _ = o2_shape(T_O2_REF)
            amp0 = max(float(init_coef[0]), 0.0)
            self.t_o2 = T_O2_REF
            self.t_o2_err = np.nan
            self.o2_prefit_amp = amp0
            self.vector_o2 = np.zeros_like(self.wave)
            self.vector_o2[self.o2_band] = full_shape[self.o2_band]
            self.matrix_o2 = self.vector_o2[None, :]
            self.vector_o2_stick = stick_shape
            self.matrix_o2_stick = self.vector_o2_stick[None, :]
            self.o2_prefit_bestfit = amp0 * self.vector_o2
            self.o2_fit_status = "fallback"
            self.o2_fit_elapsed_sec = 0.0
            set_o2_summary("fallback", 0.0, np.nan, np.r_[self.t_o2, init_coef], np.full(4, np.nan), int(np.sum(base_valid)))
            return

        bounds = ([T_O2_REF - T_O2_HALF_RANGE], [T_O2_REF + T_O2_HALF_RANGE])
        p0 = np.array([T_O2_REF, max(init_coef[0], 0.0), max(init_coef[1], 0.0), init_coef[2]], float)

        def residual(par: np.ndarray) -> np.ndarray:
            _, _, sv = o2_shape(float(par[0]))
            model = par[1] * sv + mv * (par[2] + par[3] * xlin)
            return (yv - model) * sqrt_iv

        t0 = time.perf_counter()
        res = least_squares(
            residual,
            x0=p0,
            bounds=([bounds[0][0], 0.0, 0.0, -np.inf], [bounds[1][0], np.inf, np.inf, np.inf]),
            method="trf",
        )
        dt = time.perf_counter() - t0
        if not res.success or not np.isfinite(res.x[0]):
            full_shape, stick_shape, _ = o2_shape(T_O2_REF)
            amp0 = max(float(init_coef[0]), 0.0)
            self.t_o2 = T_O2_REF
            self.t_o2_err = np.nan
            self.o2_prefit_amp = amp0
            self.vector_o2 = np.zeros_like(self.wave)
            self.vector_o2[self.o2_band] = full_shape[self.o2_band]
            self.matrix_o2 = self.vector_o2[None, :]
            self.vector_o2_stick = stick_shape
            self.matrix_o2_stick = self.vector_o2_stick[None, :]
            self.o2_prefit_bestfit = amp0 * self.vector_o2
            self.o2_fit_status = "fallback"
            self.o2_fit_elapsed_sec = dt
            set_o2_summary("fallback", dt, np.nan, np.r_[self.t_o2, init_coef], np.full(4, np.nan), int(np.sum(base_valid)))
            return

        full_shape, stick_shape, _ = o2_shape(float(res.x[0]))
        self.t_o2 = float(res.x[0])
        jtj = res.jac.T @ res.jac
        dof = max(res.fun.size - res.x.size, 1)
        cov = (2.0 * res.cost / dof) * np.linalg.pinv(jtj)
        err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        self.t_o2_err = float(err[0]) if err.size else np.nan
        self.o2_prefit_amp = float(res.x[1])
        self.vector_o2 = np.zeros_like(self.wave)
        self.vector_o2[self.o2_band] = full_shape[self.o2_band]
        self.matrix_o2 = self.vector_o2[None, :]
        self.vector_o2_stick = stick_shape
        self.matrix_o2_stick = self.vector_o2_stick[None, :]
        self.o2_prefit_bestfit = float(res.x[1]) * self.vector_o2
        self.o2_fit_status = str(res.status)
        self.o2_fit_elapsed_sec = dt
        chi2 = float(np.sum(res.fun**2))
        set_o2_summary(self.o2_fit_status, dt, chi2, res.x, err, int(np.sum(base_valid)))

    def _fit_lsf_channels(self, flux: np.ndarray, ivar: np.ndarray, source: np.ndarray, fixed_background: np.ndarray) -> None:
        offsets = np.arange(-(LSF_KERNEL_SIZE // 2), LSF_KERNEL_SIZE // 2 + 1, dtype=int)

        self.lsf_kernels = {}
        self.lsf_metrics = {}

        for channel, lo, hi in LSF_CHANNELS:
            mask = np.ones_like(self.wave, dtype=bool)
            if lo is not None:
                mask &= self.wave >= lo
            if hi is not None:
                mask &= self.wave < hi

            n_pix = int(np.sum(mask))
            if n_pix == 0:
                continue

            source_ch = np.asarray(source[mask], float)
            obs = np.asarray(flux[mask], float)
            obs_ivar = np.asarray(ivar[mask], float)
            bg = np.asarray(fixed_background[mask], float)
            target = obs - bg
            valid = np.isfinite(source_ch) & np.isfinite(target) & np.isfinite(obs_ivar) & (obs_ivar > 0)
            n_valid = int(np.sum(valid))
            valid_frac = n_valid / n_pix
            min_valid = int(np.ceil(LSF_MIN_VALID_FRAC * n_pix))
            fallback_kernel = self._default_channel_kernel(mask)

            metric: dict[str, object] = {
                "status": "ok",
                "reason": "",
                "n_pixels": n_pix,
                "n_valid_pixels": n_valid,
                "valid_frac": valid_frac,
                "chi2": np.nan,
                "chi2_red": np.nan,
                "r2": np.nan,
                "rms_resid": np.nan,
                "runtime_sec": 0.0,
                "sum_kernel": 1.0,
                "center_pix": 0.0,
                "sigma_pix": 0.0,
            }

            def finalize(kernel: np.ndarray, reason: str = "") -> None:
                kernel = np.asarray(kernel, float)
                model_lsf = self._convolve_with_kernel(source_ch, kernel, offsets)
                resid = target[valid] - model_lsf[valid]
                chi2 = float(np.sum(resid**2 * obs_ivar[valid])) if n_valid else np.nan
                dof = max(n_valid - LSF_KERNEL_SIZE, 1)
                y_mean = float(np.average(target[valid], weights=obs_ivar[valid])) if n_valid else np.nan
                sst = float(np.sum((target[valid] - y_mean) ** 2 * obs_ivar[valid])) if n_valid else np.nan
                metric["reason"] = reason
                metric["chi2"] = chi2
                metric["chi2_red"] = chi2 / dof if np.isfinite(chi2) else np.nan
                metric["r2"] = 1.0 - chi2 / sst if np.isfinite(sst) and sst > 0 else np.nan
                metric["rms_resid"] = float(np.sqrt(np.nanmean(resid**2))) if n_valid else np.nan
                metric["sum_kernel"] = float(np.sum(kernel))
                metric["center_pix"] = float(np.sum(offsets * kernel))
                metric["sigma_pix"] = float(np.sqrt(np.sum((offsets - metric["center_pix"]) ** 2 * kernel)))
                self.lsf_kernels[channel] = kernel
                self.lsf_metrics[channel] = metric

            if n_valid < max(min_valid, 1):
                metric["status"] = "fallback"
                finalize(fallback_kernel, "not_enough_valid_pixels")
                continue

            if not np.isfinite(np.sum(source_ch[valid] ** 2)) or np.sum(source_ch[valid] ** 2) <= 0:
                metric["status"] = "fallback"
                finalize(fallback_kernel, "degenerate_source_model")
                continue

            x = np.column_stack([self._shift_with_zeros(source_ch, int(off)) for off in offsets])
            x_valid = x[valid]
            w = np.sqrt(obs_ivar[valid])
            xw = x_valid * w[:, None]
            yw = target[valid] * w
            data_scale = max(float(np.sqrt(np.nanmean(yw**2))) if yw.size else 1.0, 1.0)
            xw = xw / data_scale
            yw = yw / data_scale

            p_dense = xw.T @ xw
            q = -(xw.T @ yw)
            p = sp.csc_matrix((p_dense + p_dense.T) / 2.0)
            p = sp.triu(p).tocsc()
            aeq = sp.csc_matrix(np.ones((1, LSF_KERNEL_SIZE), dtype=float))
            anonneg = -sp.eye(LSF_KERNEL_SIZE, format="csc")
            acon = sp.vstack([aeq, anonneg], format="csc")
            bcon = np.r_[1.0, np.zeros(LSF_KERNEL_SIZE, dtype=float)]
            cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(LSF_KERNEL_SIZE)]

            settings = clarabel.DefaultSettings()
            settings.verbose = False

            t0 = time.perf_counter()
            solver = clarabel.DefaultSolver(p, np.asarray(q, dtype=np.float64), acon, bcon, cones, settings)
            qp_result = solver.solve()
            metric["runtime_sec"] = time.perf_counter() - t0

            kernel = np.asarray(qp_result.x, float)
            if str(qp_result.status) != "Solved" or (not np.isfinite(kernel).all()) or float(np.sum(kernel)) <= 0:
                metric["status"] = "fallback"
                finalize(fallback_kernel, "solver_failed")
                continue

            kernel = np.clip(kernel, 0.0, np.inf)
            kernel /= np.sum(kernel)
            metric["status"] = str(qp_result.status)
            finalize(kernel)

    @staticmethod
    def _shift_with_zeros(vec: np.ndarray, offset: int) -> np.ndarray:
        out = np.zeros_like(vec, dtype=float)
        if offset == 0:
            out[:] = vec
        elif offset > 0:
            out[offset:] = vec[:-offset]
        else:
            out[:offset] = vec[-offset:]
        return out

    def _convolve_with_kernel(self, vec: np.ndarray, kernel: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return self._convolve_rows(np.asarray(vec, float)[None, :], kernel, offsets)[0]

    def _convolve_matrix_channelwise(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.copy()
        out = np.zeros_like(matrix, dtype=float)
        for channel, lo, hi in LSF_CHANNELS:
            mask = np.ones_like(self.wave, dtype=bool)
            if lo is not None:
                mask &= self.wave >= lo
            if hi is not None:
                mask &= self.wave < hi
            if not np.any(mask):
                continue
            kernel = self.lsf_kernels.get(channel, self._default_channel_kernel(mask))
            offsets = np.arange(-(kernel.size // 2), kernel.size // 2 + 1, dtype=int)
            out[:, mask] = self._convolve_rows(np.asarray(matrix[:, mask], float), kernel, offsets)
        return out

    def _default_channel_kernel(self, mask: np.ndarray) -> np.ndarray:
        offsets = np.arange(-(LSF_KERNEL_SIZE // 2), LSF_KERNEL_SIZE // 2 + 1, dtype=float)
        dw = np.gradient(self.wave[mask]) if np.count_nonzero(mask) > 1 else np.array([1.0], float)
        if np.ndim(self.lsf_sigma) > 0:
            sigma_wave = np.nanmedian(np.asarray(self.lsf_sigma[mask], float))
        else:
            sigma_wave = float(self.lsf_sigma)
        sigma_pix = sigma_wave / max(float(np.nanmedian(dw)), 1e-6)
        sigma_pix = max(float(sigma_pix), 0.5)
        kernel = np.exp(-0.5 * (offsets / sigma_pix) ** 2)
        kernel /= np.sum(kernel)
        return kernel

    def _extract_o2_chi2_red(self) -> float:
        token = "chi2_red="
        if token not in self.o2_fit_summary:
            return np.nan
        tail = self.o2_fit_summary.split(token, 1)[1]
        value = tail.split("|", 1)[0].strip()
        try:
            return float(value)
        except ValueError:
            return np.nan

    @staticmethod
    def _fmt_num(value: float) -> str:
        return f"{value:.4g}" if np.isfinite(value) else "nan"

    @staticmethod
    def _convolve_rows(matrix: np.ndarray, kernel: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0:
            return matrix.copy()
        kernel = np.asarray(kernel, dtype=float)
        offsets = np.asarray(offsets)
        k_size = kernel.size
        half = k_size // 2
        # Fast path when offsets are the canonical symmetric arange used everywhere internally.
        if (
            k_size > 0
            and offsets.size == k_size
            and np.array_equal(offsets.astype(int), np.arange(-half, half + 1))
        ):
            padded = np.pad(matrix, ((0, 0), (half, half)))
            windows = np.lib.stride_tricks.sliding_window_view(padded, k_size, axis=1)
            return np.einsum("riw,w->ri", windows, kernel[::-1])
        out = np.zeros_like(matrix, dtype=float)
        for weight, offset in zip(kernel, offsets):
            o = int(offset)
            if o == 0:
                out += weight * matrix
            elif o > 0:
                out[:, o:] += weight * matrix[:, :-o]
            else:
                out[:, :o] += weight * matrix[:, -o:]
        return out

    def _pmd_path(self, name: str) -> Path:
        suffix = self.palace_suffix
        if name == "pmd_popmodel_OH.dat" and self.palace_oh_suffix is not None:
            suffix = self.palace_oh_suffix
        elif name == "pmd_refcont.dat" and self.palace_diffuse_suffix is not None:
            suffix = self.palace_diffuse_suffix
        if suffix is not None and name in SUFFIXED_PMD_TABLES:
            path = Path(name)
            name = f"{path.stem}{suffix}{path.suffix}"
        return self._require_path(self.pmd_dir / name)

    @staticmethod
    def _validate_palace_suffix(suffix: str | None) -> str | None:
        if suffix is None:
            return None
        if not isinstance(suffix, str):
            raise TypeError("palace_suffix must be a string or None")

        if not suffix:
            raise ValueError("palace_suffix must not be empty")
        if suffix.endswith(".dat"):
            raise ValueError("palace_suffix must not include the .dat extension")
        if (
            re.fullmatch(r"[A-Za-z0-9._-]+", suffix) is None
            or re.search(r"[A-Za-z0-9]", suffix) is None
        ):
            raise ValueError(
                "palace_suffix must contain only letters, digits, '.', '_', or '-'"
            )
        return suffix

    @staticmethod
    def _require_path(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Required reference file not found: {path}")
        return path


def reconstruct_component_spectra(
    wave: np.ndarray,
    coef: np.ndarray,
    lsf_sigma: np.ndarray | float,
    *,
    n_spline_knots: int = 25,
    base_dir: str | Path | None = None,
    palace_suffix: str | None = None,
    palace_oh_suffix: str | None = None,
    palace_diffuse_suffix: str | None = None,
    o2_vector: np.ndarray | None = None,
    coef_err: np.ndarray | None = None,
    moon_interline_boost: float = 0.0,
    moon_interline_red_min: float = 7454.0,
    moon_interline_exclusion_a: float = 3.0,
    moon_interline_line_flux_threshold: float = 0.0,
    split_zodi: bool = False,
    n_zodi_spline_knots: int = 3,
    moon_albedo_asset_path: str | Path | None = None,
    moon_albedo_fiducial_phase_deg: float = 30.0,
    zodi_color_exponent: float = 0.26,
) -> dict[str, np.ndarray]:
    """Reconstruct component spectra from decomposition coefficients.

    Parameters
    ----------
    wave
        Wavelength grid in Angstrom.
    coef
        Coefficient vector matching the internal design-matrix order.
    lsf_sigma
        Gaussian LSF sigma (scalar or per-pixel vector) used to build line bases.
    n_spline_knots
        Number of interior moon B-spline knots.
    base_dir
        Root path containing PALACE/PMD and solar reference files.
    palace_suffix
        Optional suffix selecting versioned ``pmd_popmodel_OH`` and
        ``pmd_refcont`` tables. For example, ``"_adam_v1"`` selects
        ``pmd_popmodel_OH_adam_v1.dat`` and ``pmd_refcont_adam_v1.dat``.
    palace_oh_suffix
        Optional exact suffix overriding ``palace_suffix`` for
        ``pmd_popmodel_OH`` only.
    palace_diffuse_suffix
        Optional exact suffix overriding ``palace_suffix`` for
        ``pmd_refcont`` only.
    o2_vector
        Optional precomputed O2 template on `wave`. If omitted, O2 is set to zero.
    coef_err
        Optional per-coefficient 1σ uncertainty (same shape as ``coef``). When
        provided, the returned dict additionally contains ``sigma`` mapping
        each component name to its per-pixel flux 1σ (first-order Jacobian
        propagation through the LSF-convolved design matrix; see
        ``SkyDecompBase._components_sigma_from_coef_err``), plus
        ``sigma_total`` (quadrature sum of the independent-component
        variances).
    moon_interline_boost
        Strength of down-weighting for masked red pixels in the moon-spline fit.
        Masked-pixel weight is `1 / (1 + moon_interline_boost)`.
    moon_interline_red_min
        Wavelength threshold (Angstrom) above which masking/down-weighting is applied.
    moon_interline_exclusion_a
        Characteristic half-width (Angstrom) used to broaden line influence when
        defining masked line neighborhoods.
    moon_interline_line_flux_threshold
        Normalized threshold in [0, 1] applied to line-signal strength before
        building exclusion windows. Only pixels above this threshold are treated
        as line centers for exclusion.
    split_zodi, n_zodi_spline_knots, moon_albedo_asset_path,
    moon_albedo_fiducial_phase_deg, zodi_color_exponent
        When ``split_zodi=True``, add a separate Zodi_bs block to the design
        matrix (moon carries ROLO albedo × solar, zodi carries a Leinert
        reddening × solar). Must match the settings used when the decomposition
        was fit; the returned dict then also contains a ``"zodi"`` key and
        ``sigma["zodi"]`` when ``coef_err`` is provided.

    Returns
    -------
    dict[str, np.ndarray]
        Component spectra with keys: `oh`, `moon`, `ho2`, `feo`, `o2ac`,
        `diffuse`, `atom`, `orc`, `o2`, and `total`. When ``split_zodi=True``,
        `zodi` is also present.
    """
    model = SkyDecomp(
        wave,
        lsf_sigma=lsf_sigma,
        n_spline_knots=n_spline_knots,
        base_dir=base_dir,
        palace_suffix=palace_suffix,
        palace_oh_suffix=palace_oh_suffix,
        palace_diffuse_suffix=palace_diffuse_suffix,
        moon_interline_boost=moon_interline_boost,
        moon_interline_red_min=moon_interline_red_min,
        moon_interline_exclusion_a=moon_interline_exclusion_a,
        moon_interline_line_flux_threshold=moon_interline_line_flux_threshold,
        split_zodi=split_zodi,
        n_zodi_spline_knots=n_zodi_spline_knots,
        moon_albedo_asset_path=moon_albedo_asset_path,
        moon_albedo_fiducial_phase_deg=moon_albedo_fiducial_phase_deg,
        zodi_color_exponent=zodi_color_exponent,
    )

    coef_arr = np.asarray(coef, float).ravel()

    if o2_vector is not None:
        o2_vec = np.asarray(o2_vector, float).ravel()
        if o2_vec.shape != model.wave.shape:
            raise ValueError(
                f"o2_vector shape mismatch: expected {model.wave.shape}, got {o2_vec.shape}"
            )
        model.vector_o2 = o2_vec
        model.matrix_o2 = o2_vec[None, :]
        model.vector_o2_stick = o2_vec.copy()
        model.matrix_o2_stick = o2_vec[None, :]

    mats = model._matrix_bundle(
        model.matrix_oh,
        model.matrix_moon,
        model.matrix_diffuse,
        model.matrix_atom,
        model.matrix_orc,
        model.matrix_o2,
        matrix_zodi=model.matrix_zodi if split_zodi else None,
    )
    n_expected = sum(m.shape[0] for m in mats.values())
    if coef_arr.size != n_expected:
        raise ValueError(
            f"Coefficient length mismatch: expected {n_expected}, got {coef_arr.size}"
        )

    comps = model._components_from_coef(coef_arr, mats)
    comps["total"] = (
        comps["oh"]
        + comps["moon"]
        + comps["diffuse"]
        + comps["atom"]
        + comps["orc"]
        + comps["o2"]
    )
    if "zodi" in comps:
        comps["total"] = comps["total"] + comps["zodi"]
    if coef_err is not None:
        err_arr = np.asarray(coef_err, dtype=np.float64).ravel()
        if err_arr.size != coef_arr.size:
            raise ValueError(
                f"coef_err length mismatch: expected {coef_arr.size}, "
                f"got {err_arr.size}"
            )
        sigma_comps = model._components_sigma_from_coef_err(err_arr, mats)
        comps["sigma"] = sigma_comps
        comps["sigma_total"] = np.sqrt(
            sigma_comps["oh"] ** 2
            + sigma_comps["moon"] ** 2
            + sigma_comps["diffuse"] ** 2
            + sigma_comps["atom"] ** 2
            + sigma_comps["orc"] ** 2
            + sigma_comps["o2"] ** 2
            + (sigma_comps["zodi"] ** 2 if "zodi" in sigma_comps else 0.0)
        )
    return comps


__all__ = [
    "SkyDecomp",
    "SkyDecompResult",
    "vac_to_air",
    "decode_hitran_id",
    "grp2vector",
    "reconstruct_component_spectra",
]
