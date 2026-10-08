"""Frozen metadata-driven Moon and zodiacal-light model on the native LVM grid."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    GeocentricTrueEcliptic,
    SkyCoord,
    get_body,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
from astropy.time import Time, TimeDelta
from astropy.utils import iers
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr


C_KMS = 299_792.458
AU_KM = 149_597_870.7
R_MOON_KM = 1_737.4
LSF_FWHM_TO_SIGMA = 2.35482
NATIVE_PIXELS = 12_401
NATIVE_GRID_SHA256 = "78d6785c2ecda6afb2b1d4d02bfc2a28c9803dfca7477ad407d946c46e9e3dd1"
MODEL_SCHEMA_VERSION = 1
MODEL_ID = "lvm_moon_zodi_joint_adam_30000_v1"
FORMULA_VERSION = "moon-zodi-geometry-v1"
CORRECTION_SCOPE = "moon_plus_zodi"
DATA_BUNDLE_SCHEMA_VERSION = 1
DATA_BUNDLE_ID = "sky_decomp_moon_zodi_lsf_surface_iterative_v1"
DEFAULT_PALACE_OH_SUFFIX = "_telluric_upper_parity_lsf_adam_25000_v1"
SKYFAR_LINEAR_RIDGE_LAMBDA = 0.1
SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX = "_skyfar_linear_ridge_0p1_v1"
DEFAULT_PALACE_DIFFUSE_SUFFIX = "_canonhyb_v1"
EPHEMERIS_ASSET = "jpl_de432s_short_planetary_ephemeris.bsp"
SOLAR_ASSET = (
    "meftah_solar_hrs_disk_integrated_v1_1_vacuum_velocity_step_2kms.npz"
)
MOON_ALBEDO_ASSET = "eso_skycalc_rolo_moon_albedo.dat"
ZODIACAL_LIGHT_ASSET = "eso_skycalc_leinert_zodiacal_light.dat"

LCO = EarthLocation.from_geodetic(
    lon=-70.6920 * u.deg,
    lat=-29.0089 * u.deg,
    height=2281.0 * u.m,
)

# 2026-10-08: the environment variable LVMSKY_DATA_ROOT, when set, replaces the
# bundled data root next to this file.  This lets a copy of the code without
# its data/ directory (e.g. the copy vendored into lvmdrp, whose data sit in
# the calibration directory) find the bundle.  It is read once, at import, so
# it must be set before sky_decomp is imported: DEFAULT_DATA_ROOT and
# DEFAULT_DATA_DIR are also the default arguments of several functions.
# Unset, the behaviour is unchanged.
LVMSKY_DATA_ROOT_ENV = "LVMSKY_DATA_ROOT"
DEFAULT_DATA_ROOT = (
    Path(os.environ[LVMSKY_DATA_ROOT_ENV]).expanduser().resolve()
    if os.environ.get(LVMSKY_DATA_ROOT_ENV)
    else Path(__file__).resolve().parent / "data"
)
DEFAULT_DATA_DIR = DEFAULT_DATA_ROOT / "moon_zodi"
MODEL_PARAMETER_NAMES = (
    "moon_rayleigh_intercept",
    "moon_aerosol_intercept",
    "moon_common_target_airmass_log",
    "moon_common_moon_airmass_log",
    "moon_common_phase_abs_fraction",
    "moon_common_phase_signed_fraction",
    "moon_common_cos_separation_centered",
    "moon_rayleigh_extra_target_airmass_log",
    "moon_aerosol_extra_cos_separation_centered",
    "moon_below_horizon_suppression",
    "zodi_intercept",
    "zodi_target_airmass_log",
)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wave_sha256(wave: np.ndarray) -> str:
    """Hash a wavelength array in the approved little-endian float64 encoding."""
    value = np.asarray(wave, dtype=np.float64)
    return sha256(np.ascontiguousarray(value.astype("<f8", copy=False)).tobytes()).hexdigest()


@lru_cache(maxsize=None)
def validate_decomposition_data_root(data_root: str) -> dict[str, object]:
    """Validate the portable PALACE and Moon/Zodi core data contract."""
    root = Path(data_root)
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Sky decomposition data manifest is missing: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != DATA_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported sky decomposition data schema: {payload.get('schema_version')}"
        )
    if payload.get("bundle_id") != DATA_BUNDLE_ID:
        raise ValueError("Sky decomposition data bundle identifier changed")
    palace = payload.get("palace_contract")
    if not isinstance(palace, dict):
        raise ValueError("Sky decomposition data bundle lacks a PALACE contract")
    if (
        palace.get("oh_suffix") != DEFAULT_PALACE_OH_SUFFIX
        or palace.get("diffuse_suffix") != DEFAULT_PALACE_DIFFUSE_SUFFIX
    ):
        raise ValueError("Sky decomposition PALACE suffix contract changed")
    records = palace.get("assets")
    if not isinstance(records, list):
        raise ValueError("Sky decomposition data bundle lacks PALACE asset records")
    required_palace_paths = {
        f"palace/PMD/pmd_popmodel_OH{DEFAULT_PALACE_OH_SUFFIX}.dat",
        f"palace/PMD/pmd_refcont{DEFAULT_PALACE_DIFFUSE_SUFFIX}.dat",
        "palace/PMD/pmd_intdata_atom.dat",
        "palace/PMD/pmd_intmodel_Orc.dat",
        "palace/PMD/pmd_popmodel_O2.dat",
    }
    records_by_path = {str(record.get("path")): record for record in records}
    recorded_palace_paths = set(records_by_path)
    missing_palace_paths = required_palace_paths - recorded_palace_paths
    if missing_palace_paths:
        raise ValueError(
            "Sky decomposition data bundle lacks required PALACE assets: "
            + ", ".join(sorted(missing_palace_paths))
        )
    for relative_path in sorted(required_palace_paths):
        record = records_by_path[relative_path]
        path = root / relative_path
        expected = str(record["sha256"])
        if not path.is_file():
            raise FileNotFoundError(f"Required PALACE asset is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for PALACE asset {path.name}: "
                f"expected {expected}, found {actual}"
            )
    solar = payload.get("legacy_solar_reference")
    if not isinstance(solar, dict):
        raise ValueError("Sky decomposition data bundle lacks the Meftah source record")
    solar_path = root / str(solar["path"])
    if not solar_path.is_file():
        raise FileNotFoundError(f"Meftah solar source is missing: {solar_path}")
    actual_solar = file_sha256(solar_path)
    if actual_solar != str(solar["sha256"]):
        raise ValueError(
            f"Checksum mismatch for Meftah solar source {solar_path.name}: "
            f"expected {solar['sha256']}, found {actual_solar}"
        )
    return payload


@lru_cache(maxsize=None)
def validate_decomposition_asset_contract(
    data_root: str,
    contract_key: str,
    label: str,
) -> dict[str, object]:
    """Validate one method-specific asset without coupling sibling methods."""
    root = Path(data_root)
    payload = validate_decomposition_data_root(data_root)
    contract = payload.get(contract_key)
    if not isinstance(contract, dict):
        raise ValueError(f"Sky decomposition data bundle lacks a {label} contract")
    asset_path = root / str(contract["path"])
    if not asset_path.is_file():
        raise FileNotFoundError(f"Required {label} asset is missing: {asset_path}")
    actual = file_sha256(asset_path)
    if actual != str(contract["sha256"]):
        raise ValueError(
            f"Checksum mismatch for {label} asset {asset_path.name}: "
            f"expected {contract['sha256']}, found {actual}"
        )
    return contract


@dataclass(frozen=True, slots=True)
class MoonZodiObservation:
    """Metadata required to predict Moon and zodiacal light for one exposure."""

    expnum: int
    date_obs: str
    role: Literal["sci", "sky_near", "sky_far"]
    target_ra_deg: float
    target_dec_deg: float
    exposure_seconds: float
    exposure_seconds_source: Literal["metadata", "assumed_900s"]

    def __post_init__(self) -> None:
        if self.role not in {"sci", "sky_near", "sky_far"}:
            raise ValueError(f"Unsupported Moon/Zodi observation role: {self.role!r}")
        if self.exposure_seconds_source not in {"metadata", "assumed_900s"}:
            raise ValueError("exposure_seconds_source must be 'metadata' or 'assumed_900s'")
        if not np.isfinite(self.exposure_seconds) or self.exposure_seconds <= 0.0:
            raise ValueError("exposure_seconds must be positive and finite")
        if not np.isfinite(self.target_ra_deg) or not np.isfinite(self.target_dec_deg):
            raise ValueError("target coordinates must be finite")


@dataclass(frozen=True, slots=True)
class MoonZodiGeometry:
    """Exposure-midpoint geometry used by the frozen physical model."""

    midpoint_utc: str
    target_altitude_deg: float
    target_airmass: float
    sun_altitude_deg: float
    moon_altitude_deg: float
    moon_airmass: float
    moon_separation_deg: float
    signed_phase_deg: float
    moon_distance_km: float
    sun_moon_distance_km: float
    solar_elongation_deg: float
    ecliptic_lon_relative_deg: float
    ecliptic_latitude_deg: float
    moon_velocity_kms: float
    zodi_velocity_kms: float
    zodi_b500: float


@dataclass(frozen=True, slots=True)
class MoonZodiState:
    """Compact, versioned provenance needed to interpret one prediction and fit."""

    schema_version: int
    model_id: str
    formula_version: str
    scientific_status: str
    source_session: str
    manifest_sha256: str
    checkpoint_sha256: str
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    asset_records: tuple[tuple[str, str, str], ...]
    observation: MoonZodiObservation
    geometry: MoonZodiGeometry
    feature_t: float
    feature_m: float
    feature_p: float
    feature_u: float
    feature_q: float
    wave_n: int
    wave_min: float
    wave_max: float
    wave_sha256: str
    correction_scope: str
    correction_degree: int
    correction_knots: tuple[float, ...]
    physical_to_fit_flux_scale: float
    predictor_seconds: float
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MoonZodiPrediction:
    """Physical Moon and Zodi vectors plus complete compact provenance."""

    moon: np.ndarray
    zodi: np.ndarray
    state: MoonZodiState


class MoonZodiInvalidObservationError(ValueError):
    """Expected geometry rejection for an observation that cannot be modelled."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        observation: MoonZodiObservation,
        geometry: MoonZodiGeometry,
    ) -> None:
        super().__init__(message)
        self.reason = str(reason)
        self.observation = observation
        self.geometry = geometry


def _text(value: object) -> str:
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def _numeric_lines(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append([float(token) for token in stripped.split()])
    return rows


@lru_cache(maxsize=None)
def _load_manifest(data_dir: str) -> tuple[dict[str, object], str]:
    path = Path(data_dir) / "model_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Moon/Zodi model manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != MODEL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Moon/Zodi manifest schema: {payload.get('schema_version')}")
    if payload.get("model_id") != MODEL_ID or payload.get("formula_version") != FORMULA_VERSION:
        raise ValueError("Moon/Zodi manifest model or formula identifier changed")
    parameters = payload.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("Moon/Zodi manifest parameters must be an ordered list")
    names = tuple(str(item["name"]) for item in parameters)
    values = np.asarray([item["value"] for item in parameters], dtype=np.float64)
    if names != MODEL_PARAMETER_NAMES or values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("Moon/Zodi manifest global parameter contract failed")
    return payload, file_sha256(path)


@lru_cache(maxsize=None)
def _validate_assets(data_dir: str) -> tuple[tuple[str, str, str], ...]:
    root = Path(data_dir)
    manifest, _ = _load_manifest(data_dir)
    records = manifest.get("assets")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("Moon/Zodi manifest must describe exactly four assets")
    validated: list[tuple[str, str, str]] = []
    for record in records:
        name = str(record["name"])
        expected = str(record["sha256"])
        source = str(record.get("source", ""))
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required Moon/Zodi model asset is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for Moon/Zodi model asset {name}: "
                f"expected {expected}, found {actual}"
            )
        validated.append((name, actual, source))
    return tuple(validated)


@lru_cache(maxsize=None)
def _load_solar(path: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    with np.load(path, allow_pickle=False) as asset:
        wave = np.asarray(asset["wave_vacuum_angstrom"], dtype=np.float64)
        flux = np.asarray(asset["flux_disk_integrated"], dtype=np.float64)
        dv_kms = float(asset["grid_velocity_step_kms"])
        source = _text(asset["source_filename"])
    if (
        wave.shape != (157_365,)
        or flux.shape != wave.shape
        or wave.dtype != np.float64
        or flux.dtype != np.float64
        or not np.all(np.diff(wave) > 0.0)
        or not np.isclose(dv_kms, 2.0)
    ):
        raise ValueError(f"Malformed high-resolution solar asset: {path}")
    return wave, flux, dv_kms, source


@lru_cache(maxsize=None)
def _load_rolo(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _numeric_lines(Path(path))
    constants = np.asarray(rows[0], dtype=np.float64)
    n_wave = int(rows[1][0])
    coefficients = np.asarray(rows[2 : 2 + n_wave], dtype=np.float64)
    if constants.shape != (4,) or coefficients.shape != (n_wave, 11):
        raise ValueError(f"Malformed ROLO coefficient table: {path}")
    return constants, coefficients


@lru_cache(maxsize=None)
def _load_leinert(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _numeric_lines(Path(path))
    n_lon, n_lat = (int(value) for value in rows[0])
    lon = np.asarray(rows[1], dtype=np.float64)
    lat = np.asarray(rows[2], dtype=np.float64)
    values = np.asarray(rows[3 : 3 + n_lon], dtype=np.float64)
    if lon.shape != (n_lon,) or lat.shape != (n_lat,) or values.shape != (n_lon, n_lat):
        raise ValueError(f"Malformed Leinert zodiacal-light table: {path}")
    return lon, lat, values


class _LeinertDomainError(ValueError):
    """Expected physical-domain rejection from the Leinert lookup table."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = str(reason)


def _interpolate_leinert(
    lon_abs_deg: float,
    lat_abs_deg: float,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    lon, lat, values = grid
    x, y = abs(float(lon_abs_deg)), abs(float(lat_abs_deg))
    if not (lon[0] <= x <= lon[-1] and lat[0] <= y <= lat[-1]):
        raise _LeinertDomainError(
            "zodi_outside_leinert_grid",
            f"Coordinates are outside the Leinert grid: lon={x:.3f}, lat={y:.3f}",
        )
    i1 = min(max(int(np.searchsorted(lon, x, side="right")), 1), lon.size - 1)
    j1 = min(max(int(np.searchsorted(lat, y, side="right")), 1), lat.size - 1)
    i0, j0 = i1 - 1, j1 - 1
    corners = values[np.ix_([i0, i1], [j0, j1])]
    if np.any(corners >= 99999):
        raise _LeinertDomainError(
            "zodi_invalid_near_sun_cell",
            f"Invalid near-Sun Leinert cell: lon={x:.3f}, lat={y:.3f}",
        )
    tx = (x - lon[i0]) / (lon[i1] - lon[i0])
    ty = (y - lat[j0]) / (lat[j1] - lat[j0])
    return float(
        (1.0 - tx) * (1.0 - ty) * corners[0, 0]
        + tx * (1.0 - ty) * corners[1, 0]
        + (1.0 - tx) * ty * corners[0, 1]
        + tx * ty * corners[1, 1]
    )


def _airmass(altitude_deg: float) -> float:
    if altitude_deg <= 0.0:
        return np.inf
    return float(
        1.0
        / (
            np.sin(np.deg2rad(altitude_deg))
            + 0.50572 * (altitude_deg + 6.07995) ** -1.6364
        )
    )


def _cartesian_km(cartesian: object) -> np.ndarray:
    return np.asarray(cartesian.xyz.to_value(u.km), dtype=np.float64)


def _cartesian_kms(cartesian: object) -> np.ndarray:
    return np.asarray(cartesian.xyz.to_value(u.km / u.s), dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize an invalid geometry vector")
    return vector / norm


def _reflected_lunar_velocity_kms(obstime: Time) -> float:
    sun_p, sun_v = get_body_barycentric_posvel("sun", obstime)
    moon_p, moon_v = get_body_barycentric_posvel("moon", obstime)
    earth_p, earth_v = get_body_barycentric_posvel("earth", obstime)
    sun_pos, moon_pos, earth_pos = map(_cartesian_km, (sun_p, moon_p, earth_p))
    sun_vel, moon_vel, earth_vel = map(_cartesian_kms, (sun_v, moon_v, earth_v))
    incoming = np.dot(sun_vel - moon_vel, _unit(sun_pos - moon_pos))
    outgoing = np.dot(moon_vel - earth_vel, _unit(moon_pos - earth_pos))
    return float(incoming + outgoing)


@lru_cache(maxsize=8)
def _midpoint_body_geometry(
    date_obs: str, exposure_seconds: float, data_dir: str
) -> dict[str, object]:
    """Sun and Moon at the exposure midpoint -- everything that is not the target.

    The three arms of one exposure share a midpoint, so `moon_model_cache`
    asks for the same ephemeris three (and, with the physics-only pass, six)
    times.  Cached on (date_obs, exposure_seconds, data_dir); the computation
    is exactly the one `compute_midpoint_geometry` used to do inline.
    """
    root = Path(data_dir)
    obstime = Time(date_obs, scale="utc") + TimeDelta(
        0.5 * exposure_seconds,
        format="sec",
    )
    altaz = AltAz(obstime=obstime, location=LCO, pressure=0 * u.hPa)
    with solar_system_ephemeris.set(str(root / EPHEMERIS_ASSET)):
        moon = get_body("moon", obstime, LCO)
        sun = get_body("sun", obstime, LCO)
        moon_geocentric = get_body("moon", obstime)
        sun_geocentric = get_body("sun", obstime)
        moon_altitude = float(moon.transform_to(altaz).alt.deg)
        sun_altitude = float(sun.transform_to(altaz).alt.deg)
        elongation = float(moon_geocentric.separation(sun_geocentric).deg)
        phase_abs = 180.0 - elongation
        ecliptic = GeocentricTrueEcliptic(equinox=obstime)
        sun_ecliptic = sun_geocentric.transform_to(ecliptic)
        moon_ecliptic = moon_geocentric.transform_to(ecliptic)
        moon_relative_longitude = float(
            (moon_ecliptic.lon - sun_ecliptic.lon).wrap_at(180 * u.deg).to_value(u.deg)
        )
        signed_phase = float(np.sign(moon_relative_longitude) * phase_abs)
        sun_position, _ = get_body_barycentric_posvel("sun", obstime)
        moon_position, _ = get_body_barycentric_posvel("moon", obstime)
        sun_moon_distance = float(
            np.linalg.norm(_cartesian_km(sun_position) - _cartesian_km(moon_position))
        )
        moon_velocity = _reflected_lunar_velocity_kms(obstime)
    return {
        "moon": moon,
        "sun": sun,
        "moon_altitude": moon_altitude,
        "sun_altitude": sun_altitude,
        "signed_phase": signed_phase,
        "sun_ecliptic_lon": sun_ecliptic.lon,
        "sun_moon_distance": sun_moon_distance,
        "moon_velocity": moon_velocity,
    }


def compute_midpoint_geometry(
    observation: MoonZodiObservation,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    zodi_velocity_amplitude_kms: float = 12.0,
) -> MoonZodiGeometry:
    """Compute target, Moon, Sun, and ecliptic geometry at exposure midpoint."""
    root = Path(data_dir)
    iers.conf.auto_download = False
    # The frozen experiment intentionally used the packaged IERS table for
    # historical LVM exposures without a network refresh or a 30-day age gate.
    iers.conf.auto_max_age = None
    obstime = Time(observation.date_obs, scale="utc") + TimeDelta(
        0.5 * observation.exposure_seconds,
        format="sec",
    )
    target = SkyCoord(
        ra=observation.target_ra_deg * u.deg,
        dec=observation.target_dec_deg * u.deg,
        frame="icrs",
    )
    altaz = AltAz(obstime=obstime, location=LCO, pressure=0 * u.hPa)
    target_altitude = float(target.transform_to(altaz).alt.deg)
    if target_altitude <= 0.0:
        geometry = MoonZodiGeometry(
            midpoint_utc=obstime.isot,
            target_altitude_deg=target_altitude,
            target_airmass=np.nan,
            sun_altitude_deg=np.nan,
            moon_altitude_deg=np.nan,
            moon_airmass=np.nan,
            moon_separation_deg=np.nan,
            signed_phase_deg=np.nan,
            moon_distance_km=np.nan,
            sun_moon_distance_km=np.nan,
            solar_elongation_deg=np.nan,
            ecliptic_lon_relative_deg=np.nan,
            ecliptic_latitude_deg=np.nan,
            moon_velocity_kms=np.nan,
            zodi_velocity_kms=np.nan,
            zodi_b500=np.nan,
        )
        raise MoonZodiInvalidObservationError(
            "target_below_horizon",
            f"Moon/Zodi target is below the horizon: altitude={target_altitude:.3f} deg",
            observation=observation,
            geometry=geometry,
        )

    _bodies = _midpoint_body_geometry if _MEMOISE else _midpoint_body_geometry.__wrapped__
    bodies = _bodies(
        observation.date_obs, float(observation.exposure_seconds), str(root)
    )
    moon, sun = bodies["moon"], bodies["sun"]
    moon_altitude = bodies["moon_altitude"]
    sun_altitude = bodies["sun_altitude"]
    signed_phase = bodies["signed_phase"]
    sun_moon_distance = bodies["sun_moon_distance"]
    moon_velocity = bodies["moon_velocity"]
    with solar_system_ephemeris.set(str(root / EPHEMERIS_ASSET)):
        target_topocentric = target.transform_to(moon.frame)
        moon_separation = float(target_topocentric.separation(moon).deg)
        ecliptic = GeocentricTrueEcliptic(equinox=obstime)
        target_ecliptic = target.transform_to(ecliptic)
        relative_longitude = float(
            (target_ecliptic.lon - bodies["sun_ecliptic_lon"])
            .wrap_at(180 * u.deg).to_value(u.deg)
        )
        ecliptic_latitude = float(target_ecliptic.lat.to_value(u.deg))
        solar_elongation = float(target_topocentric.separation(sun).deg)

    zodi_velocity = float(
        zodi_velocity_amplitude_kms
        * np.sin(np.deg2rad(relative_longitude))
        * np.cos(np.deg2rad(ecliptic_latitude))
    )
    geometry_values = {
        "midpoint_utc": obstime.isot,
        "target_altitude_deg": target_altitude,
        "target_airmass": _airmass(target_altitude),
        "sun_altitude_deg": sun_altitude,
        "moon_altitude_deg": moon_altitude,
        "moon_airmass": _airmass(moon_altitude),
        "moon_separation_deg": moon_separation,
        "signed_phase_deg": signed_phase,
        "moon_distance_km": float(moon.distance.to_value(u.km)),
        "sun_moon_distance_km": sun_moon_distance,
        "solar_elongation_deg": solar_elongation,
        "ecliptic_lon_relative_deg": relative_longitude,
        "ecliptic_latitude_deg": ecliptic_latitude,
        "moon_velocity_kms": moon_velocity,
        "zodi_velocity_kms": zodi_velocity,
    }
    try:
        zodi_b500 = _interpolate_leinert(
            relative_longitude,
            ecliptic_latitude,
            _load_leinert(str(root / ZODIACAL_LIGHT_ASSET)),
        )
    except _LeinertDomainError as error:
        raise MoonZodiInvalidObservationError(
            error.reason,
            str(error),
            observation=observation,
            geometry=MoonZodiGeometry(**geometry_values, zodi_b500=np.nan),
        ) from error
    return MoonZodiGeometry(
        **geometry_values,
        zodi_b500=zodi_b500,
    )


def air_to_vacuum(wave_air_angstrom: np.ndarray) -> np.ndarray:
    """Convert standard-air wavelengths to vacuum wavelengths."""
    wave_air = np.asarray(wave_air_angstrom, dtype=np.float64)
    wave_vacuum = wave_air.copy()
    for _ in range(4):
        sigma2 = (1.0e4 / wave_vacuum) ** 2
        refractive_index = (
            1.0
            + 8.34254e-5
            + 2.406147e-2 / (130.0 - sigma2)
            + 1.5998e-4 / (38.9 - sigma2)
        )
        wave_vacuum = wave_air * refractive_index
    return wave_vacuum


def build_projection_operator(
    wave_detector_air: np.ndarray,
    lsf_fwhm_air: np.ndarray,
    wave_hr_vacuum: np.ndarray,
    *,
    stencil_size: int = 321,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact detector-pixel integration weights with the measured base LSF."""
    if stencil_size < 3 or stencil_size % 2 != 1:
        raise ValueError("stencil_size must be an odd integer of at least three")
    wave_detector_vacuum = air_to_vacuum(wave_detector_air)
    edges = np.empty(wave_detector_vacuum.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wave_detector_vacuum[:-1] + wave_detector_vacuum[1:])
    edges[0] = wave_detector_vacuum[0] - 0.5 * (
        wave_detector_vacuum[1] - wave_detector_vacuum[0]
    )
    edges[-1] = wave_detector_vacuum[-1] + 0.5 * (
        wave_detector_vacuum[-1] - wave_detector_vacuum[-2]
    )
    source_width = np.gradient(wave_hr_vacuum)
    sigma = (
        np.asarray(lsf_fwhm_air, dtype=np.float64)
        * wave_detector_vacuum
        / np.asarray(wave_detector_air, dtype=np.float64)
        / LSF_FWHM_TO_SIGMA
    )
    nearest = np.searchsorted(wave_hr_vacuum, wave_detector_vacuum)
    nearest = np.clip(nearest, 1, wave_hr_vacuum.size - 1)
    choose_left = np.abs(wave_hr_vacuum[nearest - 1] - wave_detector_vacuum) < np.abs(
        wave_hr_vacuum[nearest] - wave_detector_vacuum
    )
    nearest -= choose_left.astype(np.int64)
    offsets = np.arange(stencil_size, dtype=np.int64) - stencil_size // 2
    indices = np.clip(
        nearest[:, None] + offsets[None, :],
        0,
        wave_hr_vacuum.size - 1,
    )
    weights = np.empty(indices.shape, dtype=np.float64)
    for start in range(0, wave_detector_vacuum.size, chunk_size):
        stop = min(start + chunk_size, wave_detector_vacuum.size)
        index_chunk = indices[start:stop]
        source = wave_hr_vacuum[index_chunk]
        lower = edges[start:stop, None]
        upper = edges[start + 1 : stop + 1, None]
        sigma_chunk = sigma[start:stop, None]
        pixel_width = upper - lower
        weight_chunk = (
            source_width[index_chunk]
            / pixel_width
            * (
                ndtr((upper - source) / sigma_chunk)
                - ndtr((lower - source) / sigma_chunk)
            )
        )
        row_sum = np.sum(weight_chunk, axis=1, keepdims=True)
        if np.any(~np.isfinite(row_sum) | (row_sum <= 0.0)):
            raise ValueError("Moon/Zodi LSF projection has an empty or invalid detector row")
        weights[start:stop] = weight_chunk / row_sum
    return wave_detector_vacuum, indices.astype(np.int32), weights


# --- Opt-in memoisation for repeated evaluations ------------------------------
# `moon_model_cache` evaluates every arm twice (the fitted model and the
# physics-only variant used by `geometry_amplitude_prior`) and all three arms
# of an exposure share one midpoint.  The two most expensive pure steps, the
# LSF projection operator (~0.11 s) and the astropy ephemeris (~0.03 s), depend
# only on those inputs, so with memoisation on they are computed once.  Keys
# are content digests, never object identities, and cached arrays are
# read-only.  It is OFF by default: a process-wide cache would outlive any
# change to module state (a monkeypatched Leinert lookup, say), so only a
# caller whose inputs are fixed -- the cache builder's worker processes --
# should turn it on.
_MEMOISE = False


def set_memoisation(enabled: bool) -> None:
    """Turn the per-process evaluation caches on or off (and empty them)."""
    global _MEMOISE
    _MEMOISE = bool(enabled)
    _PROJECTION_CACHE.clear()
    _cached_midpoint_geometry.cache_clear()
    _midpoint_body_geometry.cache_clear()


_PROJECTION_CACHE_SIZE = 4
_PROJECTION_CACHE: "dict[tuple, tuple]" = {}


def _array_digest(values: np.ndarray) -> bytes:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return sha256(array.tobytes()).digest() + repr(array.shape).encode()


def _cached_projection(
    wave_detector_air: np.ndarray,
    lsf_fwhm_air: np.ndarray,
    wave_hr_vacuum: np.ndarray,
    wave_hr_key: bytes,
):
    """`build_projection_operator` plus its CSR form, memoised on content."""
    key = (wave_hr_key, _array_digest(wave_detector_air), _array_digest(lsf_fwhm_air))
    hit = _PROJECTION_CACHE.get(key)
    if hit is not None:
        return hit
    from scipy import sparse

    _, indices, weights = build_projection_operator(
        wave_detector_air, lsf_fwhm_air, wave_hr_vacuum
    )
    n_rows, stencil = indices.shape
    # Duplicate column indices (clipped at the grid edges) are summed by CSR,
    # exactly as the gather-and-sum formulation did.
    operator = sparse.csr_matrix(
        (weights.ravel(), indices.ravel().astype(np.int64),
         np.arange(0, n_rows * stencil + 1, stencil, dtype=np.int64)),
        shape=(n_rows, wave_hr_vacuum.size),
    )
    for array in (indices, weights):
        array.setflags(write=False)
    hit = (indices, weights, operator)
    if len(_PROJECTION_CACHE) >= _PROJECTION_CACHE_SIZE:
        _PROJECTION_CACHE.pop(next(iter(_PROJECTION_CACHE)))
    _PROJECTION_CACHE[key] = hit
    return hit


@lru_cache(maxsize=16)
def _cached_midpoint_geometry(
    observation: "MoonZodiObservation", data_dir: str
) -> "MoonZodiGeometry":
    """`compute_midpoint_geometry`, memoised on the (frozen) observation.

    Rejections are not cached: the exception propagates and the next call
    recomputes, which only costs time on rows that are refused anyway.
    """
    return compute_midpoint_geometry(observation, data_dir=data_dir)


def _rolo_albedo(
    wave_vacuum_angstrom: np.ndarray,
    signed_phase_deg: float,
    constants: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    wave_nm = np.asarray(wave_vacuum_angstrom, dtype=np.float64) / 10.0
    values = coefficients[:, 1:]
    phase_deg = abs(float(signed_phase_deg))
    phase_rad = np.deg2rad(phase_deg)
    signed_rad = np.deg2rad(float(signed_phase_deg))
    signed_limited = signed_rad if phase_deg < 97.0 else 97.0 * signed_rad / phase_deg
    polynomial = (
        values[:, 0]
        + values[:, 1] * phase_rad
        + values[:, 2] * phase_rad**2
        + values[:, 3] * phase_rad**3
        + values[:, 4] * signed_limited
        + values[:, 5] * signed_limited**3
        + values[:, 6] * signed_limited**5
    )
    opposition = (
        values[:, 7] * np.exp(-phase_deg / constants[0])
        + values[:, 8] * np.exp(-phase_deg / constants[1])
        + values[:, 9] * np.cos((phase_deg - constants[2]) / constants[3])
    )
    tabulated = np.exp(polynomial + opposition) / 0.87
    return np.interp(wave_nm, coefficients[:, 0], tabulated)


def _rayleigh_optical_depth(
    wave_vacuum_angstrom: np.ndarray,
    pressure_hpa: float = 744.0,
) -> np.ndarray:
    wave_micron = np.asarray(wave_vacuum_angstrom, dtype=np.float64) / 1.0e4
    inverse_square = wave_micron**-2
    return (
        0.008569
        * wave_micron**-4
        * (1.0 + 0.0113 * inverse_square + 0.00013 * inverse_square**2)
        * (pressure_hpa / 1013.25)
    )


def _shift_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    velocity_kms: float,
) -> np.ndarray:
    rest_wave = wave / (1.0 + velocity_kms / C_KMS)
    return np.interp(rest_wave, wave, flux)


def _stable_phi(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    small = np.abs(value) < 1.0e-4
    safe = np.where(small, 1.0, value)
    direct = -np.expm1(-value) / safe
    series = 1.0 - value / 2.0 + value**2 / 6.0 - value**3 / 24.0
    return np.where(small, series, direct)


def _rayleigh_phase_function(cosine: float, depolarization: float = 0.0148) -> float:
    return float(
        3.0
        * (1.0 - depolarization)
        / (16.0 * np.pi * (1.0 + 2.0 * depolarization))
        * (1.0 + (1.0 + 3.0 * depolarization) / (1.0 - depolarization) * cosine**2)
    )


def _henyey_greenstein_phase_function(cosine: float, g: float = 0.8) -> float:
    return float((1.0 - g**2) / (4.0 * np.pi * (1.0 + g**2 - 2.0 * g * cosine) ** 1.5))


class MoonZodiPhysicalModel:
    """Evaluate the frozen Moon/Zodi model without JAX or experiment imports."""

    def __init__(self, *, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.manifest, self.manifest_sha256 = _load_manifest(str(self.data_dir))
        self.asset_records = _validate_assets(str(self.data_dir))
        self.wave_hr, self.solar_flux, self.dv_kms, self.solar_source = _load_solar(
            str(self.data_dir / SOLAR_ASSET)
        )
        self.rolo_constants, self.rolo_coefficients = _load_rolo(
            str(self.data_dir / MOON_ALBEDO_ASSET)
        )
        parameters = self.manifest["parameters"]
        self.parameter_values = np.asarray(
            [item["value"] for item in parameters],
            dtype=np.float64,
        )
        # Pointing-independent high-resolution arrays, computed on first use.
        self._zodi_broadened: np.ndarray | None = None
        self._tau_rayleigh: np.ndarray | None = None
        self._wave_hr_key: bytes | None = None

    def validate_wave(self, wave: np.ndarray) -> None:
        value = np.asarray(wave)
        if value.dtype != np.float64:
            raise ValueError("Moon/Zodi wavelength grid must be float64")
        if value.shape != (NATIVE_PIXELS,):
            raise ValueError(
                f"Moon/Zodi requires {NATIVE_PIXELS} native wavelengths, found {value.shape}"
            )
        actual = wave_sha256(value)
        if actual != NATIVE_GRID_SHA256:
            raise ValueError(
                f"Moon/Zodi native wavelength fingerprint changed: {actual}"
            )

    def _geometry_flags(self, geometry: MoonZodiGeometry) -> tuple[str, ...]:
        flags: list[str] = []
        if geometry.moon_altitude_deg < 0.0:
            flags.append("moon_below_horizon")
        if geometry.target_altitude_deg < 30.0:
            flags.append("low_target_altitude")
        if geometry.moon_separation_deg < 20.0:
            flags.append("near_moon")
        if geometry.sun_altitude_deg > -18.0:
            flags.append("astronomical_twilight")
        if abs(geometry.signed_phase_deg) > 97.0:
            flags.append("rolo_phase_extrapolation_limited")

        training_domain = self.manifest.get("training_domain", {})
        values = {
            "target_airmass": geometry.target_airmass,
            "moon_airmass": geometry.moon_airmass if np.isfinite(geometry.moon_airmass) else 1.0,
            "moon_altitude_deg": geometry.moon_altitude_deg,
            "moon_separation_deg": geometry.moon_separation_deg,
            "signed_phase_deg": geometry.signed_phase_deg,
        }
        extrapolated = []
        for name, value in values.items():
            bounds = training_domain.get(name)
            if isinstance(bounds, list) and len(bounds) == 2:
                if value < float(bounds[0]) or value > float(bounds[1]):
                    extrapolated.append(name)
        if extrapolated:
            flags.append("geometry_extrapolated")
            flags.extend(f"geometry_extrapolated:{name}" for name in extrapolated)
        return tuple(flags)

    def predict(
        self,
        wave_air_angstrom: np.ndarray,
        detector_lsf_fwhm_air_angstrom: np.ndarray,
        observation: MoonZodiObservation,
        *,
        physical_to_fit_flux_scale: float,
    ) -> MoonZodiPrediction:
        """Return separate projected Moon and Zodi vectors on the unchanged LVM grid."""
        wave = np.asarray(wave_air_angstrom)
        lsf = np.asarray(detector_lsf_fwhm_air_angstrom)
        self.validate_wave(wave)
        if lsf.dtype != np.float64 or lsf.shape != wave.shape:
            raise ValueError("detector_lsf_fwhm must be float64 and match wave")
        if not np.all(np.isfinite(lsf) & (lsf > 0.0)):
            raise ValueError("detector_lsf_fwhm must be finite and positive")
        if not np.isfinite(physical_to_fit_flux_scale) or physical_to_fit_flux_scale <= 0.0:
            raise ValueError("physical_to_fit_flux_scale must be positive and finite")

        started = time.perf_counter()
        geometry = (
            _cached_midpoint_geometry(observation, str(self.data_dir))
            if _MEMOISE
            else compute_midpoint_geometry(observation, data_dir=self.data_dir)
        )
        moon_solar = _shift_spectrum(
            self.wave_hr,
            self.solar_flux,
            geometry.moon_velocity_kms,
        )
        if self._zodi_broadened is None:
            self._zodi_broadened = gaussian_filter1d(
                self.solar_flux,
                sigma=30.0 / self.dv_kms,
                mode="nearest",
                truncate=5.0,
            )
        zodi_broadened = self._zodi_broadened
        zodi_solar = _shift_spectrum(
            self.wave_hr,
            zodi_broadened,
            geometry.zodi_velocity_kms,
        )
        lunar_albedo = _rolo_albedo(
            self.wave_hr,
            geometry.signed_phase_deg,
            self.rolo_constants,
            self.rolo_coefficients,
        )
        if self._tau_rayleigh is None:
            self._tau_rayleigh = _rayleigh_optical_depth(self.wave_hr)
        tau_rayleigh = self._tau_rayleigh
        if _MEMOISE:
            if self._wave_hr_key is None:
                self._wave_hr_key = _array_digest(self.wave_hr)
            _, _, projection_operator = _cached_projection(
                wave, lsf, self.wave_hr, self._wave_hr_key
            )
        else:
            projection_operator = None
            _, projection_indices, projection_weights = build_projection_operator(
                wave,
                lsf,
                self.wave_hr,
            )

        below_horizon = geometry.moon_altitude_deg < 0.0
        moon_airmass_geometry = (
            geometry.moon_airmass if np.isfinite(geometry.moon_airmass) else 1.0
        )
        moon_airmass_physics = (
            _airmass(0.01) if below_horizon else moon_airmass_geometry
        )
        moon_visible = 1.0 if below_horizon or geometry.moon_altitude_deg > 0.0 else 0.0
        target_airmass = geometry.target_airmass
        wave_ratio = self.wave_hr / 5000.0
        tau_aerosol = 0.0336 * wave_ratio**-1.38
        tau_total = tau_rayleigh + tau_aerosol
        cosine = float(np.cos(np.deg2rad(geometry.moon_separation_deg)))
        phase_rayleigh = _rayleigh_phase_function(cosine)
        phase_aerosol = _henyey_greenstein_phase_function(cosine)
        delta = tau_total * (moon_airmass_physics - target_airmass)
        common = (
            target_airmass
            * np.exp(-tau_total * target_airmass)
            * _stable_phi(delta)
        )
        lunar_irradiance_factor = (R_MOON_KM / geometry.moon_distance_km) ** 2 * (
            AU_KM / geometry.sun_moon_distance_km
        ) ** 2
        carrier = (
            moon_visible
            * moon_solar
            * lunar_albedo
            * lunar_irradiance_factor
        )
        tau_rayleigh_500 = float(_rayleigh_optical_depth(np.asarray([5000.0]))[0])
        blue = (1.0 + 3.5 * target_airmass * tau_rayleigh) / (
            1.0 + 3.5 * target_airmass * tau_rayleigh_500
        )
        rayleigh_hr = carrier * common * tau_rayleigh * phase_rayleigh * blue
        aerosol_hr = (
            carrier
            * common
            * 0.97
            * tau_aerosol
            * phase_aerosol
            * blue
        )
        zodi_attenuation = np.exp(
            -target_airmass * (0.5 * tau_rayleigh + 0.5 * tau_aerosol)
        )
        reference = (self.wave_hr >= 4900.0) & (self.wave_hr <= 5100.0)
        solar_reference_500 = float(np.median(self.solar_flux[reference]))
        zodi_hr = (
            geometry.zodi_b500
            * 1.0e-8
            / 1000.0
            * zodi_solar
            / solar_reference_500
            * wave_ratio**0.26
            * zodi_attenuation
        )
        fibre_solid_angle_sr = np.pi * (0.5 * 35.3 / 206_265.0) ** 2
        conversion = fibre_solid_angle_sr * 100.0

        def project(values: np.ndarray) -> np.ndarray:
            if projection_operator is not None:
                return conversion * (projection_operator @ values)
            return conversion * np.sum(
                np.take(values, projection_indices) * projection_weights,
                axis=1,
            )

        rayleigh = project(rayleigh_hr)
        aerosol = project(aerosol_hr)
        zodi_base = project(zodi_hr)

        theta = self.parameter_values[:10]
        zeta = self.parameter_values[10:]
        feature_t = float(np.log(target_airmass))
        feature_m = float(np.log(max(moon_airmass_geometry, 1.0)))
        feature_p = float(abs(geometry.signed_phase_deg) / 180.0)
        feature_u = float(geometry.signed_phase_deg / 180.0)
        feature_q = float(cosine - 0.5)
        common_geometry = (
            theta[2] * feature_t
            + theta[3] * feature_m
            + theta[4] * feature_p
            + theta[5] * feature_u
            + theta[6] * feature_q
        )
        rayleigh_scale = np.exp(theta[0] + common_geometry + theta[7] * feature_t)
        aerosol_scale = np.exp(theta[1] + common_geometry + theta[8] * feature_q)
        below_depth = np.tanh(max(-geometry.moon_altitude_deg, 0.0) / 5.0)
        horizon_scale = np.exp(theta[9] * below_depth)
        zodi_scale = np.exp(zeta[0] + zeta[1] * feature_t)
        moon = horizon_scale * (rayleigh_scale * rayleigh + aerosol_scale * aerosol)
        zodi = zodi_scale * zodi_base
        moon = np.asarray(moon * physical_to_fit_flux_scale, dtype=np.float64)
        zodi = np.asarray(zodi * physical_to_fit_flux_scale, dtype=np.float64)
        if (
            not np.all(np.isfinite(moon))
            or not np.all(np.isfinite(zodi))
            or np.any(moon < 0.0)
            or np.any(zodi < 0.0)
        ):
            raise RuntimeError("Moon/Zodi predictor produced invalid or negative values")

        flags = list(self._geometry_flags(geometry))
        if observation.exposure_seconds_source == "assumed_900s":
            flags.append("exposure_time_assumed")
        elapsed = time.perf_counter() - started
        state = MoonZodiState(
            schema_version=MODEL_SCHEMA_VERSION,
            model_id=MODEL_ID,
            formula_version=FORMULA_VERSION,
            scientific_status=str(self.manifest["scientific_status"]),
            source_session=str(self.manifest["source_session"]),
            manifest_sha256=self.manifest_sha256,
            checkpoint_sha256=str(self.manifest["source_checkpoint"]["sha256"]),
            parameter_names=MODEL_PARAMETER_NAMES,
            parameter_values=tuple(float(value) for value in self.parameter_values),
            asset_records=self.asset_records,
            observation=observation,
            geometry=geometry,
            feature_t=feature_t,
            feature_m=feature_m,
            feature_p=feature_p,
            feature_u=feature_u,
            feature_q=feature_q,
            wave_n=int(wave.size),
            wave_min=float(wave[0]),
            wave_max=float(wave[-1]),
            wave_sha256=wave_sha256(wave),
            correction_scope=CORRECTION_SCOPE,
            correction_degree=3,
            correction_knots=(),
            physical_to_fit_flux_scale=float(physical_to_fit_flux_scale),
            predictor_seconds=elapsed,
            flags=tuple(flags),
        )
        return MoonZodiPrediction(moon=moon, zodi=zodi, state=state)

    def invalid_observation_state(
        self,
        wave_air_angstrom: np.ndarray,
        observation: MoonZodiObservation,
        error: MoonZodiInvalidObservationError,
        *,
        physical_to_fit_flux_scale: float,
    ) -> MoonZodiState:
        """Return provenance for one expected rejection without fabricating a fit."""
        wave = np.asarray(wave_air_angstrom)
        self.validate_wave(wave)
        if error.observation != observation:
            raise ValueError("Invalid-observation context does not match the observation")
        if (
            not np.isfinite(physical_to_fit_flux_scale)
            or physical_to_fit_flux_scale <= 0.0
        ):
            raise ValueError("physical_to_fit_flux_scale must be positive and finite")
        flags = [
            "invalid_observation",
            f"invalid_observation:{error.reason}",
        ]
        if observation.exposure_seconds_source == "assumed_900s":
            flags.append("exposure_time_assumed")
        return MoonZodiState(
            schema_version=MODEL_SCHEMA_VERSION,
            model_id=MODEL_ID,
            formula_version=FORMULA_VERSION,
            scientific_status=str(self.manifest["scientific_status"]),
            source_session=str(self.manifest["source_session"]),
            manifest_sha256=self.manifest_sha256,
            checkpoint_sha256=str(self.manifest["source_checkpoint"]["sha256"]),
            parameter_names=MODEL_PARAMETER_NAMES,
            parameter_values=tuple(float(value) for value in self.parameter_values),
            asset_records=self.asset_records,
            observation=observation,
            geometry=error.geometry,
            feature_t=np.nan,
            feature_m=np.nan,
            feature_p=np.nan,
            feature_u=np.nan,
            feature_q=np.nan,
            wave_n=int(wave.size),
            wave_min=float(wave[0]),
            wave_max=float(wave[-1]),
            wave_sha256=wave_sha256(wave),
            correction_scope=CORRECTION_SCOPE,
            correction_degree=3,
            correction_knots=(),
            physical_to_fit_flux_scale=float(physical_to_fit_flux_scale),
            predictor_seconds=np.nan,
            flags=tuple(flags),
        )


_AMPLITUDE_PRIOR_MODELS: dict[str, "MoonZodiPhysicalModel"] = {}


def _physics_only_model(data_dir: str | Path) -> "MoonZodiPhysicalModel":
    """Cached predictor with the LEARNED scale factors set to unity.

    Keeps only ``moon_below_horizon_suppression``, which is a geometric taper
    rather than a calibration: without it the moon carrier is built at a
    fictitious 0.01 deg altitude whenever the moon is down and no amount of
    downstream clipping recovers the right answer.

    The other ten parameters are dropped on purpose.  They were fitted with
    ``CORRECTION_SCOPE = 'moon_plus_zodi'``, i.e. the training only ever
    constrained the SUM, so the split they imply is not independently
    calibrated -- and ``zodi_target_airmass_log = 1.28`` (zodi growing as
    X^1.28, where extinction should make it fall) looks like it absorbed the
    same moon-into-zodi leakage the amplitude prior exists to remove.  Using it
    would feed that leakage back in as a prior.  In practice the two variants
    agree on the moon fraction to a few percent, so this costs nothing.
    """
    key = str(Path(data_dir).resolve())
    model = _AMPLITUDE_PRIOR_MODELS.get(key)
    if model is None:
        model = MoonZodiPhysicalModel(data_dir=data_dir)
        values = np.array(model.parameter_values, dtype=float).copy()
        horizon = values[MODEL_PARAMETER_NAMES.index("moon_below_horizon_suppression")]
        values[:] = 0.0
        values[MODEL_PARAMETER_NAMES.index("moon_below_horizon_suppression")] = horizon
        model.parameter_values = values
        _AMPLITUDE_PRIOR_MODELS[key] = model
    return model


# --- Empirical correction to the Leinert zodiacal-light brightness ---------
# Measured 2026-09-24 on the 1.3.2 gaia-stars-mask-telluric-chi2 corpus, from
# the 14,979 DARK pointings (moon_frac_po <= 1/150; sci, near and far arms
# pooled) where the decomposition fits Zodi_bs freely.  Target:
# log10(Z_fit / (SPLIT_ZODI_ZODI_PRIOR_CALIBRATION * Z_leinert)), with the
# +-log10(kappa_z) bracket handled as CENSORING (Tobit likelihood) -- up to a
# third of high-|beta| dark rows sit on the ceiling, so a plain fit to the
# interior would be biased toward the model.  Interior fits do not depend on
# the anchor (it only clips), so they are a genuine measurement.
#
# What was wrong: the Leinert profile falls off too STEEPLY with ecliptic
# latitude for this data -- too bright on the ecliptic, too faint toward the
# poles -- so it OVERSTATES the zodi contrast between pointings a few degrees
# apart (fitted-on-model slope of sci-arm differences: 0.45 near / 0.43 far).
#
# Fitted JOINTLY with a galactic-latitude term and a log-airmass term, and
# only the ecliptic part is kept here.  That matters: Zodi_bs also absorbs
# Galactic continuum (diffuse galactic light / unresolved stars, ~1.5x more in
# the plane, no template in the basis), and an ecliptic-only fit partly
# mistakes it for zodi.  The Galactic part is -0.15 dex plane-to-pole.  The Galactic and
# airmass terms were deliberately NOT deployed -- the Galactic one is not
# zodiacal light, and the airmass one (+0.05 per ln X) looks like airglow
# leaking into Zodi_bs.  The level was re-fitted with the shape fixed.
#
# Validated out-of-fold with WHOLE NIGHTS held out (this corpus leaks within
# a night): dark scatter 0.124 -> 0.102 dex; sci-arm contrast slope
# 0.45 -> 0.74 (near), 0.43 -> 0.94 (far); sci-far residual 0.113 -> 0.088.
# Degree 2 was chosen over higher degrees on the CONTRAST test: degree 5 fits
# absolute levels better but drives the near-arm slope to 0.37, i.e. it fits
# fine structure that does not transfer between neighbouring pointings.
#
# GEOMETRY: |lambda - lambda_sun| is the TRUE geocentric-ecliptic value,
# exactly as `relative_longitude` is computed below.  Do NOT derive it from
# the ML context column `sun_sep`: that column is not the solar elongation
# (mlp_predictor.data computes it as an ICRS separation, which puts the Sun
# at the barycentre -- rho +0.13 with the true elongation).  A first version
# of this fit did exactly that and had the longitude dependence wrong.
#
# Result: pole-vs-ecliptic +0.30 dex at |dlam| = 120; near-Sun vs anti-Sun on
# the ecliptic only -0.03 dex.  Applied range over the 41,064 observed
# pointings: -0.13 to +0.26 dex, median +0.06.
#
# Form: log10 factor = sum_k c_k P_i(u) P_j(v), Legendre P, with
#   u = 2 |lambda - lambda_sun| / 180 - 1,   v = 2 sin|beta| - 1.
ZODI_LEINERT_CORRECTIONS: dict[str, dict | None] = {
    "none": None,
    "lvm-ecl-2026-09": {
        "terms": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
        "coef": (-0.0015446893945962865, 0.14287470227857374, 0.0355828485715581,
                 0.15213766681794777, -0.005288404813077793, -0.025153192741487505,
                 -0.11350842322273286, -0.02537049388355427),
    },
}


def zodi_leinert_correction_log10(
    correction: str, relative_longitude_deg: float, ecliptic_latitude_deg: float
) -> float:
    """log10 multiplicative correction to the Leinert zodi for ``correction``.

    ``"none"`` returns 0.0 exactly, so the uncorrected model is bit-identical.
    Raises ``KeyError`` for an unknown name rather than silently using none: a
    decomposition and the cache built from it must agree on this.
    """
    spec = ZODI_LEINERT_CORRECTIONS[str(correction)]
    if spec is None:
        return 0.0
    dlam = abs(((float(relative_longitude_deg) + 180.0) % 360.0) - 180.0)
    u_ = 2.0 * dlam / 180.0 - 1.0
    v_ = 2.0 * np.sin(np.deg2rad(abs(float(ecliptic_latitude_deg)))) - 1.0
    pu = np.polynomial.legendre.legvander(np.asarray([u_]), 2)[0]
    pv = np.polynomial.legendre.legvander(np.asarray([v_]), 2)[0]
    return float(sum(c * pu[i] * pv[j]
                     for (i, j), c in zip(spec["terms"], spec["coef"])))


def geometry_amplitude_prior(
    wave_air_angstrom: np.ndarray,
    detector_lsf_fwhm_air_angstrom: np.ndarray,
    observation: MoonZodiObservation,
    *,
    physical_to_fit_flux_scale: float,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    zodi_correction: str = "none",
) -> tuple[float, float, float]:
    """Geometry priors for ``SkyDecomp.set_amplitude_prior`` on one spectrum.

    ``zodi_correction`` names an entry of ``ZODI_LEINERT_CORRECTIONS``.  It
    rescales the zodi total AND is carried into ``moon_fraction``, so the two
    constraints stay mutually consistent.  Every consumer -- the decomposition
    anchor, the moon-model cache, the inference path -- must pass the SAME name
    the decomposition was run with; see ``moon_model_cache.load``.

    Returns ``(moon_fraction, zodi_total, target_airmass)``:

    * ``moon_fraction`` -- ``int(moon) / int(moon + zodi)``.  Calibration-free:
      both carriers leave the model through the same solid-angle and flux
      conversion, so the ratio is independent of throughput and of
      ``physical_to_fit_flux_scale``.
    * ``zodi_total`` -- ``int(zodi)`` in FIT flux units, for the absolute
      Leinert bracket.  This one is NOT calibration-free, so
      ``physical_to_fit_flux_scale`` must equal the FACTOR the flux being
      fitted was multiplied by.
    * ``target_airmass`` -- from the same geometry solve, for
      ``set_target_airmass``.

    Both predictions come from a single ``predict`` call so the shared
    conversion cancels in the ratio.  Raises
    ``MoonZodiInvalidObservationError`` for geometry that cannot be modelled
    (target below the horizon); callers should skip the prior for that row
    rather than fail the fit.
    """
    prediction = _physics_only_model(data_dir).predict(
        wave_air_angstrom,
        detector_lsf_fwhm_air_angstrom,
        observation,
        physical_to_fit_flux_scale=physical_to_fit_flux_scale,
    )
    moon_total = float(np.nansum(prediction.moon))
    zodi_total = float(np.nansum(prediction.zodi))
    if zodi_correction != "none":
        _geo = prediction.state.geometry
        zodi_total *= 10.0 ** zodi_leinert_correction_log10(
            zodi_correction, _geo.ecliptic_lon_relative_deg,
            _geo.ecliptic_latitude_deg)
    denominator = moon_total + zodi_total
    fraction = moon_total / denominator if denominator > 0.0 else np.nan
    return fraction, zodi_total, float(prediction.state.geometry.target_airmass)


__all__ = [
    "ZODI_LEINERT_CORRECTIONS",
    "zodi_leinert_correction_log10",
    "CORRECTION_SCOPE",
    "DATA_BUNDLE_ID",
    "DATA_BUNDLE_SCHEMA_VERSION",
    "DEFAULT_DATA_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_PALACE_DIFFUSE_SUFFIX",
    "DEFAULT_PALACE_OH_SUFFIX",
    "EPHEMERIS_ASSET",
    "FORMULA_VERSION",
    "MOON_ALBEDO_ASSET",
    "MODEL_ID",
    "MODEL_PARAMETER_NAMES",
    "MODEL_SCHEMA_VERSION",
    "MoonZodiGeometry",
    "MoonZodiInvalidObservationError",
    "MoonZodiObservation",
    "MoonZodiPhysicalModel",
    "MoonZodiPrediction",
    "MoonZodiState",
    "SKYFAR_LINEAR_RIDGE_LAMBDA",
    "SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX",
    "SOLAR_ASSET",
    "ZODIACAL_LIGHT_ASSET",
    "air_to_vacuum",
    "build_projection_operator",
    "compute_midpoint_geometry",
    "file_sha256",
    "validate_decomposition_asset_contract",
    "geometry_amplitude_prior",
    "validate_decomposition_data_root",
    "wave_sha256",
]
