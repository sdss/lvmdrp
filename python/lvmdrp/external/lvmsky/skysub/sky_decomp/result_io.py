"""Serialization and restoration for sky-decomposition results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .fit import LSF_CHANNELS

if TYPE_CHECKING:
    from .moon_zodi_model import MoonZodiState


LSF_HDU_NAMES = ("LSF_COEF", "LSF_KNOTS", "LSF_META")
MOON_ZODI_HDU_NAMES = ("MZ_MODEL", "MZ_ASSETS", "MZ_KNOTS", "MZ_META")
INVALID_OBSERVATION_FIT_STATUS = "skipped_invalid_observation"
FAILED_INPUT_FIT_STATUS = "failed_input"
CHANNEL_NAMES = tuple(channel for channel, _, _ in LSF_CHANNELS)

STATE_CONFIG_COLUMNS = (
    ("line_weight", "line_weight", float),
    ("skyline_cumulative_fraction", "skyline_cumulative_fraction", float),
    ("skyline_half_width_angstrom", "skyline_half_width_angstrom", float),
    ("huber_transition_sigma", "huber_transition_sigma", float),
    ("n_refinement_cycles", "requested_cycles", int),
    ("n_basis", "config_n_basis", int),
    ("degree", "config_degree", int),
    ("roughness_fraction", "roughness_fraction", float),
    ("fallback_prior_fraction", "fallback_prior_fraction", float),
    ("information_prior_max_boost", "information_prior_max_boost", float),
    ("background_degree", "background_degree", int),
    ("blue_fit_lower", "blue_fit_lower", float),
)

METRIC_COLUMNS = (
    ("status", "", str),
    ("reason", "", str),
    ("n_pixels", 0, int),
    ("n_valid_pixels", 0, int),
    ("chi2_red", np.nan, float),
    ("rms_resid", np.nan, float),
    ("center_pix_min", np.nan, float),
    ("center_pix_median", np.nan, float),
    ("center_pix_max", np.nan, float),
    ("sigma_pix_min", np.nan, float),
    ("sigma_pix_median", np.nan, float),
    ("sigma_pix_max", np.nan, float),
    ("runtime_sec", 0.0, float),
    ("fit_lower", np.nan, float),
    ("fit_upper", np.nan, float),
    ("knots_fixed", False, bool),
    ("model", "", str),
)


def _text_value(value: object) -> str:
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def _lsf_meta_row(spectrum_index, state, channel, *, available=True, reason=None):
    coefficient = state.coefficients[channel]
    knot_vector = state.knot_vectors[channel]
    metric = state.metrics[channel]
    lower, upper = state.channel_bounds[channel]
    row = {
        "spectrum_index": spectrum_index,
        "channel": channel,
        "available": bool(available),
        "lower": np.nan if lower is None else lower,
        "upper": np.nan if upper is None else upper,
        "degree": state.degrees[channel],
        "n_basis": coefficient.shape[1],
        "n_knots": knot_vector.size,
        "tap_offsets": state.tap_offsets.copy(),
        "status": str(metric.get("status", "")),
        "reason": str(metric.get("reason", "")),
        "schema_version": state.schema_version,
        "requested_cycles": state.requested_cycles,
        "completed_cycles": state.completed_cycles,
        "fit_status": state.fit_status,
        "failure_reason": state.failure_reason,
        "final_continuum_status": state.final_continuum_status,
        "final_line_status": state.final_line_status,
        "knot_strategy": state.knot_strategy,
        "legacy_kernel_representation": state.legacy_kernel_representation,
        "wave_n": state.wave_n,
        "wave_min": state.wave_min,
        "wave_max": state.wave_max,
        "wave_sha256": state.wave_sha256,
    }
    for name, column, converter in STATE_CONFIG_COLUMNS:
        if name != "n_refinement_cycles":
            row[column] = converter(state.config[name])
    for name, default, converter in METRIC_COLUMNS[2:]:
        row[name] = converter(metric.get(name, default))
    if not available:
        row["status"] = "unavailable"
        row["reason"] = str(reason or "fit unavailable")
        row["completed_cycles"] = 0
        row["fit_status"] = "failed_input"
        row["failure_reason"] = row["reason"]
    return row


def build_lsf_hdus(states):
    """Validate compact LSF states and return their ordered FITS HDUs."""
    from astropy.io import fits
    from astropy.table import Table

    if not states:
        raise ValueError("LSF state list must not be empty")
    reference = states[0]
    for state in states:
        if state.schema_version != reference.schema_version:
            raise ValueError("All LSF states must use the same schema version")
        if not np.array_equal(state.tap_offsets, reference.tap_offsets):
            raise ValueError("All LSF states must use the same tap offsets")
        if (
            state.wave_n != reference.wave_n
            or state.wave_min != reference.wave_min
            or state.wave_max != reference.wave_max
            or state.wave_sha256 != reference.wave_sha256
        ):
            raise ValueError("All LSF states must describe the same wavelength grid")
        if set(state.coefficients) != set(CHANNEL_NAMES):
            raise ValueError("Every LSF state must contain B, R, and Z channels")
    max_basis = max(
        state.coefficients[channel].shape[1] for state in states for channel in CHANNEL_NAMES
    )
    max_knots = max(
        state.knot_vectors[channel].size for state in states for channel in CHANNEL_NAMES
    )
    coefficient_cube = np.full(
        (len(states), len(CHANNEL_NAMES), reference.tap_offsets.size, max_basis),
        np.nan,
        dtype=float,
    )
    knot_cube = np.full(
        (len(states), len(CHANNEL_NAMES), max_knots),
        np.nan,
        dtype=float,
    )
    meta_rows = []
    for spectrum_index, state in enumerate(states):
        for channel_index, channel in enumerate(CHANNEL_NAMES):
            coefficient = state.coefficients[channel]
            knot_vector = state.knot_vectors[channel]
            coefficient_cube[spectrum_index, channel_index, :, : coefficient.shape[1]] = (
                coefficient
            )
            knot_cube[spectrum_index, channel_index, : knot_vector.size] = knot_vector
            meta_rows.append(_lsf_meta_row(spectrum_index, state, channel))

    coefficient_hdu = fits.ImageHDU(coefficient_cube, name="LSF_COEF")
    coefficient_hdu.header["TAPMIN"] = int(reference.tap_offsets[0])
    coefficient_hdu.header["TAPMAX"] = int(reference.tap_offsets[-1])
    return [
        coefficient_hdu,
        fits.ImageHDU(knot_cube, name="LSF_KNOTS"),
        fits.BinTableHDU(Table(rows=meta_rows), name="LSF_META"),
    ]


def load_lsf_surface_cubes(filename: str | Path):
    """Read the three LSF HDUs once, for repeated per-row state construction.

    :func:`load_lsf_surface_state` reads LSF_COEF, LSF_KNOTS and LSF_META in
    full on every call.  That is fine for a handful of rows, but a diagnostic
    that walks thousands of them pays it thousands of times -- on a full-corpus
    decomposition those three HDUs are ~30 + 4 + 48 MB, so a 2900-row x 3-arm
    sweep would read of order half a terabyte to extract a few MB of state.
    Read them once with this and pass the result to
    :func:`lsf_surface_state_from_cubes`.
    """
    from astropy.io import fits

    with fits.open(filename, memmap=False) as hdul:
        for name in LSF_HDU_NAMES:
            if name not in hdul:
                raise KeyError(f"Missing {name} extension in {filename}")
        # LSF_META stays a FITS_rec: astropy decodes TFORM 'L' columns to bool
        # on access, and np.asarray() would hand back the raw int8 FITS
        # logicals instead -- where 'F' is 70, i.e. TRUTHY.  That silently
        # defeats the `available` check below, so a row with no fitted LSF
        # would return a garbage state instead of raising.  Safe to use after
        # the close because memmap=False has already read it into memory.
        return (np.asarray(hdul["LSF_COEF"].data, dtype=float),
                np.asarray(hdul["LSF_KNOTS"].data, dtype=float),
                hdul["LSF_META"].data)


def load_lsf_surface_state(filename: str | Path, spectrum_index: int = 0):
    """Load one compact LSF state from an extended decomposition FITS."""
    coefficient_cube, knot_cube, meta = load_lsf_surface_cubes(filename)
    return lsf_surface_state_from_cubes(
        (coefficient_cube, knot_cube, meta), spectrum_index)


def lsf_surface_state_from_cubes(cubes, spectrum_index: int = 0):
    """Build one :class:`LSFSurfaceState` from cubes already in memory.

    ``cubes`` is the ``(coefficient_cube, knot_cube, meta)`` tuple returned by
    :func:`load_lsf_surface_cubes`.  Same result as
    :func:`load_lsf_surface_state`, without the re-read.
    """
    from .lsf_surface_iterative import LSF_STATE_SCHEMA_VERSION, LSFSurfaceState

    coefficient_cube, knot_cube, meta = cubes

    if coefficient_cube.ndim != 4 or knot_cube.ndim != 3:
        raise ValueError("Stored LSF coefficient or knot arrays have invalid shapes")
    if spectrum_index < 0 or spectrum_index >= coefficient_cube.shape[0]:
        raise IndexError("spectrum_index is outside the stored LSF array")
    rows = meta[meta["spectrum_index"] == spectrum_index]
    if len(rows) != len(CHANNEL_NAMES):
        raise ValueError("LSF_META does not contain exactly one row per channel")
    if not np.all(np.asarray(rows["available"], dtype=bool)):
        raise ValueError("No fitted LSF state is available for this spectrum")

    coefficients: dict[str, np.ndarray] = {}
    knots: dict[str, np.ndarray] = {}
    degrees: dict[str, int] = {}
    bounds: dict[str, tuple[float | None, float | None]] = {}
    metrics: dict[str, dict[str, object]] = {}
    first_row = rows[0]
    config = {
        name: converter(first_row[column])
        for name, column, converter in STATE_CONFIG_COLUMNS
    }
    tap_offsets = np.asarray(first_row["tap_offsets"], dtype=int)
    if tap_offsets.ndim != 1 or tap_offsets.size != coefficient_cube.shape[2]:
        raise ValueError("Stored tap offsets do not match LSF_COEF")
    schema_version = int(first_row["schema_version"])
    if schema_version != LSF_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported LSF state schema version: {schema_version}")
    channel_to_index = {name: index for index, name in enumerate(CHANNEL_NAMES)}
    for row in rows:
        channel = _text_value(row["channel"])
        if channel not in channel_to_index or channel in coefficients:
            raise ValueError(f"Invalid or duplicate LSF channel row: {channel}")
        if not np.array_equal(np.asarray(row["tap_offsets"], dtype=int), tap_offsets):
            raise ValueError("Stored LSF tap offsets differ between channels")
        channel_index = channel_to_index[channel]
        n_basis = int(row["n_basis"])
        n_knots = int(row["n_knots"])
        if n_basis < 1 or n_knots < 2:
            raise ValueError(f"Invalid stored LSF dimensions for channel {channel}")
        coefficients[channel] = coefficient_cube[
            spectrum_index, channel_index, :, :n_basis
        ].copy()
        knots[channel] = knot_cube[spectrum_index, channel_index, :n_knots].copy()
        degrees[channel] = int(row["degree"])
        lower_value = float(row["lower"])
        upper_value = float(row["upper"])
        bounds[channel] = (
            None if np.isnan(lower_value) else lower_value,
            None if np.isnan(upper_value) else upper_value,
        )
        metrics[channel] = {
            name: _text_value(row[name]) if converter is str else converter(row[name])
            for name, _, converter in METRIC_COLUMNS
        }
    return LSFSurfaceState(
        coefficients=coefficients,
        knot_vectors=knots,
        degrees=degrees,
        channel_bounds=bounds,
        tap_offsets=tap_offsets,
        config=config,
        metrics=metrics,
        requested_cycles=int(first_row["requested_cycles"]),
        completed_cycles=int(first_row["completed_cycles"]),
        wave_n=int(first_row["wave_n"]),
        wave_min=float(first_row["wave_min"]),
        wave_max=float(first_row["wave_max"]),
        wave_sha256=_text_value(first_row["wave_sha256"]),
        fit_status=_text_value(first_row["fit_status"]),
        failure_reason=_text_value(first_row["failure_reason"]),
        final_continuum_status=_text_value(first_row["final_continuum_status"]),
        final_line_status=_text_value(first_row["final_line_status"]),
        knot_strategy=_text_value(first_row["knot_strategy"]),
        legacy_kernel_representation=_text_value(first_row["legacy_kernel_representation"]),
        schema_version=schema_version,
    )


def _moon_zodi_signature(state: MoonZodiState) -> tuple[object, ...]:
    return (
        state.schema_version,
        state.model_id,
        state.formula_version,
        state.scientific_status,
        state.source_session,
        state.manifest_sha256,
        state.checkpoint_sha256,
        state.parameter_names,
        state.parameter_values,
        state.asset_records,
        state.wave_n,
        state.wave_min,
        state.wave_max,
        state.wave_sha256,
        state.correction_scope,
        state.correction_degree,
        state.correction_knots,
        state.physical_to_fit_flux_scale,
    )


def validate_moon_zodi_states(states: list[MoonZodiState]) -> None:
    """Reject mixed model bundles, grids, correction bases, or flux scales."""
    from .moon_zodi_model import MODEL_PARAMETER_NAMES

    if not states:
        raise ValueError("Moon/Zodi FITS state list must not be empty")
    reference = states[0]
    if reference.parameter_names != MODEL_PARAMETER_NAMES:
        raise ValueError("Moon/Zodi parameter names or order changed")
    if len(reference.asset_records) != 4:
        raise ValueError("Moon/Zodi state must contain exactly four asset records")
    if not reference.correction_knots:
        raise ValueError("Moon/Zodi correction knot vector is empty")
    signature = _moon_zodi_signature(reference)
    for index, state in enumerate(states):
        if _moon_zodi_signature(state) != signature:
            raise ValueError(
                f"Moon/Zodi result {index} uses an incompatible model, grid, or correction contract"
            )


def _moon_zodi_meta_row(spectrum_index: int, state: MoonZodiState) -> dict[str, object]:
    observation = state.observation
    geometry = state.geometry
    flags = set(state.flags)
    return {
        "spectrum_index": spectrum_index,
        "expnum": observation.expnum,
        "role": observation.role,
        "date_obs_utc": observation.date_obs,
        "midpoint_utc": geometry.midpoint_utc,
        "exposure_seconds": observation.exposure_seconds,
        "exposure_seconds_source": observation.exposure_seconds_source,
        "target_ra_deg": observation.target_ra_deg,
        "target_dec_deg": observation.target_dec_deg,
        "target_altitude_deg": geometry.target_altitude_deg,
        "target_airmass": geometry.target_airmass,
        "sun_altitude_deg": geometry.sun_altitude_deg,
        "moon_altitude_deg": geometry.moon_altitude_deg,
        "moon_airmass": geometry.moon_airmass,
        "moon_separation_deg": geometry.moon_separation_deg,
        "signed_phase_deg": geometry.signed_phase_deg,
        "moon_distance_km": geometry.moon_distance_km,
        "sun_moon_distance_km": geometry.sun_moon_distance_km,
        "solar_elongation_deg": geometry.solar_elongation_deg,
        "ecliptic_lon_relative_deg": geometry.ecliptic_lon_relative_deg,
        "ecliptic_latitude_deg": geometry.ecliptic_latitude_deg,
        "moon_velocity_kms": geometry.moon_velocity_kms,
        "zodi_velocity_kms": geometry.zodi_velocity_kms,
        "zodi_b500": geometry.zodi_b500,
        "feature_t": state.feature_t,
        "feature_m": state.feature_m,
        "feature_p": state.feature_p,
        "feature_u": state.feature_u,
        "feature_q": state.feature_q,
        "moon_below_horizon": "moon_below_horizon" in flags,
        "geometry_extrapolated": "geometry_extrapolated" in flags,
        "exptime_assumed": "exposure_time_assumed" in flags,
        "flags_json": json.dumps(state.flags, separators=(",", ":")),
        "physical_to_fit_flux_scale": state.physical_to_fit_flux_scale,
        "predictor_seconds": state.predictor_seconds,
        "correction_scope": state.correction_scope,
        "correction_degree": state.correction_degree,
        "wave_n": state.wave_n,
        "wave_min": state.wave_min,
        "wave_max": state.wave_max,
        "wave_sha256": state.wave_sha256,
        "model_id": state.model_id,
    }


def build_moon_zodi_hdus(states: list[MoonZodiState]):
    """Validate states and return the four ordered Moon/Zodi FITS HDUs."""
    from astropy.io import fits
    from astropy.table import Table

    validate_moon_zodi_states(states)
    reference = states[0]
    model_hdu = fits.BinTableHDU(
        Table(
            {
                "NAME": list(reference.parameter_names),
                "VALUE": np.asarray(reference.parameter_values, dtype=np.float64),
            }
        ),
        name="MZ_MODEL",
    )
    model_hdu.header["SCHEMAV"] = reference.schema_version
    model_hdu.header["MODELID"] = reference.model_id
    model_hdu.header["FORMULAV"] = reference.formula_version
    model_hdu.header["STATUS"] = reference.scientific_status
    model_hdu.header["SRCSESS"] = reference.source_session
    model_hdu.header["CHECKSHA"] = reference.checkpoint_sha256
    model_hdu.header["MANIFSHA"] = reference.manifest_sha256
    model_hdu.header["WAVESHA"] = reference.wave_sha256
    model_hdu.header["CORRSCP"] = reference.correction_scope
    model_hdu.header["CORRDEG"] = reference.correction_degree

    assets_hdu = fits.BinTableHDU(
        Table(
            {
                "NAME": [record[0] for record in reference.asset_records],
                "SHA256": [record[1] for record in reference.asset_records],
                "SOURCE": [record[2] for record in reference.asset_records],
            }
        ),
        name="MZ_ASSETS",
    )
    knots_hdu = fits.ImageHDU(
        np.asarray(reference.correction_knots, dtype=np.float64),
        name="MZ_KNOTS",
    )
    knots_hdu.header["DEGREE"] = reference.correction_degree
    knots_hdu.header["SCOPE"] = reference.correction_scope
    meta_hdu = fits.BinTableHDU(
        Table(rows=[_moon_zodi_meta_row(index, state) for index, state in enumerate(states)]),
        name="MZ_META",
    )
    return [model_hdu, assets_hdu, knots_hdu, meta_hdu]


def load_moon_zodi_state(
    filename: str | Path,
    spectrum_index: int = 0,
) -> MoonZodiState:
    """Load one compact Moon/Zodi state from an extended decomposition FITS."""
    from astropy.io import fits

    from .moon_zodi_model import (
        MODEL_PARAMETER_NAMES,
        MoonZodiGeometry,
        MoonZodiObservation,
        MoonZodiState,
    )

    with fits.open(filename, memmap=False) as hdul:
        present = [name in hdul for name in MOON_ZODI_HDU_NAMES]
        if any(present) and not all(present):
            missing = [
                name
                for name, exists in zip(MOON_ZODI_HDU_NAMES, present)
                if not exists
            ]
            raise KeyError(f"Incomplete Moon/Zodi state in {filename}; missing {missing}")
        if not all(present):
            raise KeyError(f"No Moon/Zodi state extensions in {filename}")
        model_data = np.asarray(hdul["MZ_MODEL"].data).copy()
        model_header = hdul["MZ_MODEL"].header.copy()
        assets_data = np.asarray(hdul["MZ_ASSETS"].data).copy()
        knots = np.asarray(hdul["MZ_KNOTS"].data, dtype=np.float64).copy()
        meta = np.asarray(hdul["MZ_META"].data).copy()

    if model_data.shape[0] != 12:
        raise ValueError("MZ_MODEL must contain exactly 12 parameter rows")
    parameter_names = tuple(_text_value(value) for value in model_data["NAME"])
    if parameter_names != MODEL_PARAMETER_NAMES:
        raise ValueError("Stored Moon/Zodi parameter names or order changed")
    parameter_values = tuple(float(value) for value in model_data["VALUE"])
    if assets_data.shape[0] != 4:
        raise ValueError("MZ_ASSETS must contain exactly four rows")
    asset_records = tuple(
        (_text_value(row["NAME"]), _text_value(row["SHA256"]), _text_value(row["SOURCE"]))
        for row in assets_data
    )
    matches = meta[np.asarray(meta["spectrum_index"], dtype=int) == spectrum_index]
    if len(matches) != 1:
        raise ValueError("MZ_META must contain exactly one row for spectrum_index")
    row = matches[0]
    observation = MoonZodiObservation(
        expnum=int(row["expnum"]),
        date_obs=_text_value(row["date_obs_utc"]),
        role=_text_value(row["role"]),
        target_ra_deg=float(row["target_ra_deg"]),
        target_dec_deg=float(row["target_dec_deg"]),
        exposure_seconds=float(row["exposure_seconds"]),
        exposure_seconds_source=_text_value(row["exposure_seconds_source"]),
    )
    geometry = MoonZodiGeometry(
        midpoint_utc=_text_value(row["midpoint_utc"]),
        target_altitude_deg=float(row["target_altitude_deg"]),
        target_airmass=float(row["target_airmass"]),
        sun_altitude_deg=float(row["sun_altitude_deg"]),
        moon_altitude_deg=float(row["moon_altitude_deg"]),
        moon_airmass=float(row["moon_airmass"]),
        moon_separation_deg=float(row["moon_separation_deg"]),
        signed_phase_deg=float(row["signed_phase_deg"]),
        moon_distance_km=float(row["moon_distance_km"]),
        sun_moon_distance_km=float(row["sun_moon_distance_km"]),
        solar_elongation_deg=float(row["solar_elongation_deg"]),
        ecliptic_lon_relative_deg=float(row["ecliptic_lon_relative_deg"]),
        ecliptic_latitude_deg=float(row["ecliptic_latitude_deg"]),
        moon_velocity_kms=float(row["moon_velocity_kms"]),
        zodi_velocity_kms=float(row["zodi_velocity_kms"]),
        zodi_b500=float(row["zodi_b500"]),
    )
    flags = tuple(str(value) for value in json.loads(_text_value(row["flags_json"])))
    state = MoonZodiState(
        schema_version=int(model_header["SCHEMAV"]),
        model_id=_text_value(model_header["MODELID"]),
        formula_version=_text_value(model_header["FORMULAV"]),
        scientific_status=_text_value(model_header["STATUS"]),
        source_session=_text_value(model_header["SRCSESS"]),
        manifest_sha256=_text_value(model_header["MANIFSHA"]),
        checkpoint_sha256=_text_value(model_header["CHECKSHA"]),
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        asset_records=asset_records,
        observation=observation,
        geometry=geometry,
        feature_t=float(row["feature_t"]),
        feature_m=float(row["feature_m"]),
        feature_p=float(row["feature_p"]),
        feature_u=float(row["feature_u"]),
        feature_q=float(row["feature_q"]),
        wave_n=int(row["wave_n"]),
        wave_min=float(row["wave_min"]),
        wave_max=float(row["wave_max"]),
        wave_sha256=_text_value(row["wave_sha256"]),
        correction_scope=_text_value(row["correction_scope"]),
        correction_degree=int(row["correction_degree"]),
        correction_knots=tuple(float(value) for value in knots),
        physical_to_fit_flux_scale=float(row["physical_to_fit_flux_scale"]),
        predictor_seconds=float(row["predictor_seconds"]),
        flags=flags,
    )
    validate_moon_zodi_states([state])
    return state


def _validate_result_batch(results):
    """Validate one homogeneous result batch before constructing any output HDU."""
    if not results:
        raise ValueError("results must contain at least one decomposition result")
    concrete_type = type(results[0])
    if any(type(result) is not concrete_type for result in results):
        raise ValueError(
            "Cannot mix decomposition result types; all results must have the same concrete type"
        )
    design_names = tuple(results[0].design_names)
    if len(design_names) != len(set(design_names)):
        raise ValueError("design_names must be unique")
    component_keys = tuple(results[0].components)
    n_wave = np.asarray(results[0].bestfit_lsf).size
    if n_wave < 1:
        raise ValueError("result wavelength dimension is empty")

    lsf_signature = None
    for index, result in enumerate(results):
        invalid_observation = result.fit_status == INVALID_OBSERVATION_FIT_STATUS
        failed_input = result.fit_status == FAILED_INPUT_FIT_STATUS
        nan_placeholder = invalid_observation or failed_input
        if tuple(result.design_names) != design_names:
            raise ValueError(f"Result {index} has different ordered design_names")
        if tuple(result.components) != component_keys:
            raise ValueError(f"Result {index} has different ordered component keys")
        if np.asarray(result.coef).shape != (len(design_names),):
            raise ValueError(f"Result {index} coefficient shape does not match design_names")
        if np.asarray(result.coef_err).shape != (len(design_names),):
            raise ValueError(
                f"Result {index} coefficient error shape does not match design_names"
            )
        for name in (
            "bestfit",
            "bestfit_lsf",
            "bestfit_lsf_sigma",
            "resid",
            "vector_o2",
        ):
            if np.asarray(getattr(result, name)).shape != (n_wave,):
                raise ValueError(f"Result {index} {name} has an incompatible wavelength shape")
        for name, component in result.components.items():
            if np.asarray(component).shape != (n_wave,):
                raise ValueError(f"Result {index} component {name} has an incompatible shape")

        if nan_placeholder:
            nan_arrays = (
                "coef",
                "coef_err",
                "bestfit",
                "bestfit_lsf",
                "bestfit_lsf_sigma",
                "resid",
                "vector_o2",
            )
            for name in nan_arrays:
                if not np.all(np.isnan(np.asarray(getattr(result, name), dtype=float))):
                    raise ValueError(
                        f"NaN-placeholder result {index} must store only NaN in {name}"
                    )
            for name, component in result.components.items():
                if not np.all(np.isnan(np.asarray(component, dtype=float))):
                    raise ValueError(
                        "NaN-placeholder result "
                        f"{index} must store only NaN in component {name}"
                    )
            nan_scalars = (
                "resid_level",
                "reduced_chi2",
                "fit_elapsed_sec",
                "t_o2",
                "t_o2_err",
                "r2",
                "rms_resid",
                "peak_memory_mb",
                "o2_fit_elapsed_sec",
                "o2_valid_frac",
                "o2_prefit_amp",
            )
            for name in nan_scalars:
                if not np.isnan(float(getattr(result, name))):
                    raise ValueError(
                        "NaN-placeholder result "
                        f"{index} must store NaN in scalar {name}"
                    )
            if "reason=" not in str(result.fit_summary):
                raise ValueError(
                    f"NaN-placeholder result {index} must preserve a rejection reason"
                )

        o2_indices = [
            position for position, name in enumerate(design_names) if name == "O2_b01"
        ]
        if o2_indices and not nan_placeholder:
            if len(o2_indices) != 1:
                raise ValueError("O2_b01 must occur exactly once in design_names")
            expected_o2 = float(result.coef[o2_indices[0]]) * np.asarray(result.vector_o2)
            o2_tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(expected_o2))))
            if (
                float(np.max(np.abs(np.asarray(result.components["o2"]) - expected_o2)))
                > o2_tolerance
            ):
                raise ValueError(f"Result {index} failed named O2 component closure")

        state = getattr(result, "lsf_state", None)
        if state is not None:
            if nan_placeholder and any(
                not np.all(np.isnan(np.asarray(value, dtype=float)))
                for value in state.coefficients.values()
            ):
                raise ValueError(
                    f"NaN-placeholder result {index} must store NaN LSF coefficients"
                )
            signature = (
                state.schema_version,
                tuple(state.tap_offsets),
                tuple(sorted(state.config.items())),
                state.requested_cycles,
                state.wave_n,
                state.wave_min,
                state.wave_max,
                state.wave_sha256,
                tuple(sorted(state.channel_bounds.items())),
            )
            if lsf_signature is None:
                lsf_signature = signature
            elif signature != lsf_signature:
                raise ValueError(f"Result {index} has an incompatible LSF state")

    has_lsf = [getattr(result, "lsf_state", None) is not None for result in results]
    if any(has_lsf) and not all(has_lsf):
        raise ValueError("Cannot mix results with and without compact LSF state")
    has_moon_zodi = [hasattr(result, "moon_zodi_state") for result in results]
    if any(has_moon_zodi) and not all(has_moon_zodi):
        raise ValueError("Cannot mix Moon/Zodi and non-Moon/Zodi results")
    if all(has_moon_zodi):
        expected_components = ("moon", "zodi", "diffuse", "oh", "atom", "orc", "o2")
        if component_keys != expected_components:
            raise ValueError(f"Moon/Zodi component contract changed: {component_keys}")
        if design_names.count("O2_b01") != 1:
            raise ValueError("Moon/Zodi results require exactly one named O2_b01 coefficient")
        validate_moon_zodi_states([result.moon_zodi_state for result in results])
        for index, result in enumerate(results):
            if result.fit_status == INVALID_OBSERVATION_FIT_STATUS:
                if "invalid_observation" not in result.moon_zodi_state.flags:
                    raise ValueError(
                        f"Invalid-observation result {index} lacks its Moon/Zodi state flag"
                    )
                continue
            reconstructed = sum(
                (np.asarray(result.components[name]) for name in expected_components),
                np.zeros(n_wave, dtype=np.float64),
            )
            tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(result.bestfit_lsf))))
            if float(np.max(np.abs(reconstructed - result.bestfit_lsf))) > tolerance:
                raise ValueError(f"Result {index} failed full Moon/Zodi model closure")
    return design_names, component_keys, all(has_lsf), all(has_moon_zodi)


def extra_meta_columns(extra_meta, n_results, reserved=()):
    """Validate per-row extra META entries and pivot them into columns.

    ``extra_meta`` is a sequence of dicts aligned with the results, all sharing
    one key set -- every row of a FITS column must exist, so a caller that
    computes a flag for only some rows has to fill the rest explicitly (see
    `decompose_parallel._reliability_extra_meta`, which writes -1 rather than
    letting a missing flag read as a clean row).  ``None`` yields no columns.
    """
    if extra_meta is None:
        return {}
    extra_meta = list(extra_meta)
    if len(extra_meta) != int(n_results):
        raise ValueError(
            f"extra_meta has {len(extra_meta)} entries for "
            f"{int(n_results)} results")
    key_sets = {tuple(sorted(entry)) for entry in extra_meta}
    if len(key_sets) > 1:
        raise ValueError(
            f"extra_meta rows disagree on their columns: {sorted(key_sets)}")
    columns = {}
    for key in (key_sets.pop() if key_sets else ()):
        if key in reserved:
            raise ValueError(
                f"extra_meta column {key!r} collides with a result field")
        columns[key] = [entry[key] for entry in extra_meta]
    return columns


def results_to_fits(results, filename, extra_meta=None, primary_meta=None):
    """Write a homogeneous batch of sky-decomposition results to FITS.

    ``primary_meta`` adds or overrides primary-header provenance cards.
    ``extra_meta`` optionally adds per-row META columns that do not live on the
    result object -- the result dataclasses use ``slots=True``, so a caller
    cannot attach anything to them.  It is a sequence aligned with ``results``
    of dicts sharing one key set, which is how `decompose_parallel` carries the
    `sky_decomp.reliability` flags (a decision made by the FITTER, e.g. whether
    a reversal retry was run, and therefore not recoverable from the result).

    ``primary_meta`` optionally adds run-level cards to the primary header,
    as ``{key: value}`` or ``{key: (value, comment)}`` -- e.g. ZODICORR, which
    `moon_model_cache` reads back to match its prior to the decomposition.
    """
    from astropy.io import fits
    from astropy.table import Table

    design_names, comp_keys, is_iterative, is_moon_zodi = _validate_result_batch(results)
    rows = {
        "t_o2": [result.t_o2 for result in results],
        "t_o2_err": [result.t_o2_err for result in results],
        "o2_prefit_amp": [result.o2_prefit_amp for result in results],
        "reduced_chi2": [result.reduced_chi2 for result in results],
        "r2": [result.r2 for result in results],
        "rms_resid": [result.rms_resid for result in results],
        "resid_level": [result.resid_level for result in results],
        "fit_status": [result.fit_status for result in results],
        "fit_summary": [result.fit_summary for result in results],
        "fit_elapsed_sec": [result.fit_elapsed_sec for result in results],
        "peak_memory_mb": [result.peak_memory_mb for result in results],
        "o2_fit_status": [result.o2_fit_status for result in results],
        "o2_fit_summary": [result.o2_fit_summary for result in results],
        "o2_fit_elapsed_sec": [result.o2_fit_elapsed_sec for result in results],
        "o2_valid_frac": [result.o2_valid_frac for result in results],
    }
    rows.update(extra_meta_columns(extra_meta, len(results), reserved=rows))

    def stack(attribute):
        return np.vstack([getattr(result, attribute) for result in results])

    coef_arr = stack("coef")
    if coef_arr.shape[1] != len(design_names):
        raise ValueError(
            f"Coefficient count ({coef_arr.shape[1]}) does not match "
            f"number of design names ({len(design_names)})."
        )
    coef_table = Table(
        {name: coef_arr[:, index] for index, name in enumerate(design_names)}
    )
    coef_err_arr = stack("coef_err")
    if coef_err_arr.shape != coef_arr.shape:
        raise ValueError(
            f"Coefficient error shape {coef_err_arr.shape} does not match "
            f"coefficient shape {coef_arr.shape}."
        )
    coef_err_table = Table(
        {name: coef_err_arr[:, index] for index, name in enumerate(design_names)}
    )
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(Table(rows), name="META"),
            fits.BinTableHDU(coef_table, name="COEF"),
            fits.BinTableHDU(coef_err_table, name="COEF_ERR"),
            fits.ImageHDU(stack("bestfit"), name="BESTFIT"),
            fits.ImageHDU(stack("bestfit_lsf"), name="BESTFIT_LSF"),
            fits.ImageHDU(stack("bestfit_lsf_sigma"), name="FLUX_SIGMA_TOTAL"),
            fits.ImageHDU(stack("resid"), name="RESID"),
            fits.ImageHDU(stack("vector_o2"), name="VECTOR_O2"),
        ]
    )

    def _stack_cov_block(attribute):
        blocks = [getattr(result, attribute, None) for result in results]
        if all(b is None for b in blocks):
            return None
        # Uniform shape check: any present block dictates (n_block, n_block).
        template = next(b for b in blocks if b is not None)
        template = np.asarray(template, dtype=float)
        if template.ndim != 2 or template.shape[0] != template.shape[1]:
            raise ValueError(
                f"{attribute} must be a square 2D matrix per row; "
                f"got shape {template.shape}")
        n_block = int(template.shape[0])
        # A 1x1 block adds no information beyond the corresponding COEF_ERR
        # entry, so skip writing the HDU in that degenerate case (moon-zodi
        # physical model has n_moon=n_zodi=1).
        if n_block <= 1:
            return None
        out = np.full((len(results), n_block, n_block), np.nan, dtype=float)
        for i, block in enumerate(blocks):
            if block is None:
                continue
            arr = np.asarray(block, dtype=float)
            if arr.shape != (n_block, n_block):
                raise ValueError(
                    f"{attribute} shape mismatch on result {i}: "
                    f"expected {(n_block, n_block)}, got {arr.shape}")
            out[i] = arr
        return out

    for attr, hdu_name in (
        ("coef_cov_moon", "COEF_COV_MOON"),
        ("coef_cov_zodi", "COEF_COV_ZODI"),
    ):
        arr = _stack_cov_block(attr)
        if arr is not None:
            hdul.append(fits.ImageHDU(arr, name=hdu_name))
    for key in comp_keys:
        hdul.append(
            fits.ImageHDU(
                np.vstack([result.components[key] for result in results]),
                name=f"COMP_{key.upper()}",
            )
        )

    iterative_result_types = []
    try:
        from .lsf_surface_iterative import LSFSurfaceIterativeResult

        iterative_result_types.append(LSFSurfaceIterativeResult)
    except ModuleNotFoundError:
        pass
    iterative_result_types = tuple(set(iterative_result_types))
    recognised_iterative = [
        bool(iterative_result_types) and isinstance(result, iterative_result_types)
        for result in results
    ]
    if is_iterative and not all(recognised_iterative):
        raise TypeError("Compact LSF state is attached to an unrecognised result type")
    if is_iterative:
        states = [result.lsf_state for result in results]
        if any(state is None for state in states):
            raise ValueError("Every iterative result must contain a compact LSF state")
        hdul.extend(build_lsf_hdus(states))
        if states[0].legacy_kernel_representation == "continuous_mspline_density":
            residual_pca_names = [
                name for name in design_names if name.startswith("ResidualPCA_")
            ]
            line_amplitude_pca_names = [
                name
                for name in design_names
                if name.startswith("LineAmplitudePCA_")
            ]
            line_adjoint_pca_names = [
                name for name in design_names if name.startswith("LineAdjointPCA_")
            ]
            uses_vn_oh_groups = any(name.startswith("OHVN_") for name in design_names)
            vnf_pca_names = [
                name
                for name in design_names
                if name.startswith("OHVNFPCAPrep_")
            ]
            if line_adjoint_pca_names:
                hdul[0].header["DECOMPM"] = (
                    "telluric-corrected-lines-palace-aijc-vnf-line-adjoint-pca"
                )
                hdul[0].header["LADPCAK"] = len(line_adjoint_pca_names) - 1
                hdul[0].header["OHCOEF"] = "PALACE Aijc"
                hdul[0].header["OHGROUP"] = "v_upper,N_upper,F_upper"
            elif line_amplitude_pca_names:
                if vnf_pca_names:
                    hdul[0].header["DECOMPM"] = (
                        "telluric-corrected-lines-palace-aijc-vnf-pca-line-pca"
                    )
                elif uses_vn_oh_groups:
                    hdul[0].header["DECOMPM"] = (
                        "telluric-corrected-lines-palace-aijc-vn-line-amplitude-pca"
                    )
                else:
                    hdul[0].header["DECOMPM"] = (
                        "telluric-corrected-lines-palace-aijc-line-amplitude-pca"
                    )
                hdul[0].header["LAPCAK"] = len(line_amplitude_pca_names) - 1
                hdul[0].header["OHCOEF"] = "PALACE Aijc"
            elif residual_pca_names:
                hdul[0].header["DECOMPM"] = (
                    "telluric-corrected-lines-palace-aijc-residual-pca"
                )
                hdul[0].header["RESPCAK"] = len(residual_pca_names) - 1
                hdul[0].header["OHCOEF"] = "PALACE Aijc"
            elif vnf_pca_names:
                hdul[0].header["DECOMPM"] = (
                    "telluric-corrected-lines-palace-aijc-vnf-pca-prep"
                )
                hdul[0].header["OHCOEF"] = "PALACE Aijc"
            elif uses_vn_oh_groups:
                hdul[0].header["DECOMPM"] = (
                    "telluric-corrected-lines-palace-aijc-vn"
                )
                hdul[0].header["OHCOEF"] = "PALACE Aijc"
            else:
                hdul[0].header["DECOMPM"] = "lsf-spline2d-split-zodi"
            if uses_vn_oh_groups:
                hdul[0].header["OHGROUP"] = "v_upper,N_upper"
            if vnf_pca_names:
                hdul[0].header["OHGROUP"] = "v_upper,N_upper,F_upper"
                hdul[0].header["OHPCAK"] = len(vnf_pca_names) - 1
            hdul["LSF_COEF"].header["BASIS"] = "M-spline"

    if is_moon_zodi:
        hdul[0].header["DECOMPM"] = "moon-zodi-lsf-surface-iterative"
        hdul[0].header["SCHEMAV"] = 1
        hdul.extend(build_moon_zodi_hdus([result.moon_zodi_state for result in results]))

    for key, value in (primary_meta or {}).items():
        hdul[0].header[key] = value
    hdul.writeto(filename, overwrite=True)
    print(
        f"Wrote {len(results)} results, {coef_arr.shape[1]} coefs, "
        f"{len(comp_keys)} components → {filename}"
    )


__all__ = [
    "FAILED_INPUT_FIT_STATUS",
    "INVALID_OBSERVATION_FIT_STATUS",
    "LSF_HDU_NAMES",
    "MOON_ZODI_HDU_NAMES",
    "build_lsf_hdus",
    "build_moon_zodi_hdus",
    "load_lsf_surface_state",
    "load_moon_zodi_state",
    "results_to_fits",
    "extra_meta_columns",
    "validate_moon_zodi_states",
]
