"""Telluric-corrected line strengths on the continuous 2-D LSF model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

from .fit import SPLIT_ZODI_CONTINUUM_DEFAULTS, grp2vector, sticks2vector
from .lsf_spline2d import SkyDecompLSFSpline2D
from .moon_zodi_model import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PALACE_OH_SUFFIX,
    validate_decomposition_asset_contract,
    validate_decomposition_data_root,
)
TELLURIC_CORRECTED_LINES_FIT_MODEL = "telluric-corrected-lines-lsf-spline2d"
LINE_TELLURIC_ASSET = "palace/PMD/palace_line_telluric_r4m_v1.fits"
LINE_TELLURIC_CONTRACT = "line_telluric_contract"
PALACE_REFERENCE_PWV_MM = 2.5


def _positive_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def calculate_line_transmission(
    tau_non_h2o_ref: np.ndarray,
    tau_h2o_ref: np.ndarray,
    pwv_mm: float,
    airmass: float,
) -> np.ndarray:
    """Evaluate PALACE molecular absorption from its R=4e6 line coefficients."""
    tau_non_h2o_ref = np.asarray(tau_non_h2o_ref, dtype=float)
    tau_h2o_ref = np.asarray(tau_h2o_ref, dtype=float)
    pwv_mm = _positive_scalar(pwv_mm, "pwv_mm")
    airmass = _positive_scalar(airmass, "airmass")
    if tau_non_h2o_ref.shape != tau_h2o_ref.shape:
        raise ValueError("the two line optical-depth vectors must have the same shape")
    if np.any(~np.isfinite(tau_non_h2o_ref)) or np.any(~np.isfinite(tau_h2o_ref)):
        raise ValueError("line optical-depth vectors must be finite")
    transmission = np.exp(
        -airmass
        * (
            tau_non_h2o_ref
            + (pwv_mm / PALACE_REFERENCE_PWV_MM) * tau_h2o_ref
        )
    )
    if np.any(~np.isfinite(transmission)) or np.any(transmission < 0.0):
        raise ValueError("line transmission must be finite and non-negative")
    return transmission


@lru_cache(maxsize=None)
def _load_line_telluric_coefficients(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = Table.read(path)
    required = {"wave_air_model_A", "tau_non_H2O_ref", "tau_H2O_ref"}
    if not required.issubset(table.colnames):
        raise ValueError("PALACE line telluric table is missing required columns")
    if int(table.meta.get("RMAX", 0)) != 4_000_000:
        raise ValueError("PALACE line telluric table is not the R=4e6 product")
    if float(table.meta.get("PWVREF", np.nan)) != PALACE_REFERENCE_PWV_MM:
        raise ValueError("PALACE line telluric table has the wrong reference PWV")
    arrays = tuple(
        np.asarray(table[name], dtype=np.float64)
        for name in ("wave_air_model_A", "tau_non_H2O_ref", "tau_H2O_ref")
    )
    if any(array.ndim != 1 or np.any(~np.isfinite(array)) for array in arrays):
        raise ValueError("PALACE line telluric columns must be finite vectors")
    return arrays


def calculate_drp_transmission(
    wave: np.ndarray,
    all_fiber_lsf: np.ndarray,
    pwv_mm: float,
    reduction_airmass: float,
    telluric_calculator: Any,
) -> np.ndarray:
    """Reproduce the native-pixel transmission used by the DRP."""
    wave = np.asarray(wave, dtype=float)
    lsf = np.asarray(all_fiber_lsf)
    pwv_mm = _positive_scalar(pwv_mm, "pwv_mm")
    reduction_airmass = _positive_scalar(reduction_airmass, "reduction_airmass")
    if wave.ndim != 1 or np.any(~np.isfinite(wave)) or np.any(np.diff(wave) <= 0.0):
        raise ValueError("wave must be finite and strictly increasing")
    if lsf.ndim != 2 or lsf.shape[1] != wave.size:
        raise ValueError("all_fiber_lsf must have shape (n_fibers, n_wave)")
    lsf_median = np.nanmedian(lsf, axis=0)
    if np.any(~np.isfinite(lsf_median)) or np.any(lsf_median <= 0.0):
        raise ValueError("the all-fiber DRP median LSF must be finite and positive")
    transmission = np.asarray(
        telluric_calculator.match_to_data(
            wave,
            lsf_median,
            pwv_mm,
            airmass=reduction_airmass,
            lsf_in_wavelength=True,
        ),
        dtype=float,
    )
    if transmission.shape != wave.shape:
        raise ValueError("DRP transmission does not match the native wavelength grid")
    if np.any(~np.isfinite(transmission)) or np.any(transmission <= 0.0):
        raise ValueError("DRP transmission must be finite and strictly positive")
    if np.any(transmission > 1.00001):
        raise ValueError("DRP transmission exceeds unity")
    return transmission


def restore_drp_input(
    flux_corrected: np.ndarray,
    ivar_corrected: np.ndarray,
    wave: np.ndarray,
    all_fiber_lsf: np.ndarray,
    pwv_mm: float,
    sci_airmass: float,
    telluric_calculator: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undo the DRP pixel-space telluric division without changing the grid."""
    flux = np.asarray(flux_corrected, dtype=float)
    ivar = np.asarray(ivar_corrected, dtype=float)
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1 or flux.shape != wave.shape or ivar.shape != wave.shape:
        raise ValueError("flux, ivar, and wave must be conformable one-dimensional arrays")
    if np.any(~np.isfinite(flux)) or np.any(~np.isfinite(ivar)) or np.any(ivar <= 0.0):
        raise ValueError("flux and positive ivar must be finite on every native pixel")
    transmission = calculate_drp_transmission(
        wave,
        all_fiber_lsf,
        pwv_mm,
        sci_airmass,
        telluric_calculator,
    )
    return flux * transmission, ivar / transmission**2, transmission


class SkyDecompTelluricLinesLSFSpline2D(SkyDecompLSFSpline2D):
    """Attenuate every configured sky line at its line-centre transmission."""

    def __init__(
        self,
        *args: Any,
        telluric_calculator: Any,
        pwv_mm: float,
        line_airmass: float,
        **kwargs: Any,
    ) -> None:
        self.telluric_calculator = telluric_calculator
        self.pwv_mm = _positive_scalar(pwv_mm, "pwv_mm")
        self.line_airmass = _positive_scalar(line_airmass, "line_airmass")
        base_dir = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        if (base_dir / "bundle_manifest.json").is_file():
            validate_decomposition_data_root(str(base_dir))
            validate_decomposition_asset_contract(
                str(base_dir),
                LINE_TELLURIC_CONTRACT,
                "PALACE R=4e6 line telluric",
            )
        kwargs["base_dir"] = base_dir
        super().__init__(*args, **kwargs)

        self._intrinsic_groups = {
            "oh": [(wave.copy(), amp.copy()) for wave, amp in self._oh_line_groups],
            "atom": [(wave.copy(), amp.copy()) for wave, amp in self._atom_line_groups],
            "orc": [(wave.copy(), amp.copy()) for wave, amp in self._orc_line_groups],
        }
        coefficient_wave, tau_non_h2o_ref, tau_h2o_ref = (
            _load_line_telluric_coefficients(str(base_dir / LINE_TELLURIC_ASSET))
        )
        if not np.array_equal(coefficient_wave, self._line_wave):
            raise ValueError("PALACE line telluric table does not match the model line order")
        self._line_transmission_values = calculate_line_transmission(
            tau_non_h2o_ref,
            tau_h2o_ref,
            self.pwv_mm,
            self.line_airmass,
        )

        self._line_transmission(self._line_wave)
        for family in ("oh", "atom", "orc"):
            self._rebuild_family(family)
        self.design_matrix = self._assemble_design_matrix()

    def _line_transmission(self, line_wave_air: np.ndarray) -> np.ndarray:
        line_wave_air = np.asarray(line_wave_air, dtype=float)
        if not np.array_equal(line_wave_air, self._line_wave):
            raise ValueError("line transmission requires the exact ordered model line catalog")
        return self._line_transmission_values.copy()

    def _line_weights(self) -> np.ndarray:
        return super()._line_weights() * self._line_transmission(self._line_wave)

    def _rebuild_family(self, family: str) -> None:
        groups = self._intrinsic_groups[family]
        matrix = np.zeros((len(groups), self.wave.size), dtype=float)
        matrix_stick = np.zeros_like(matrix)
        for index, (line_wave, intrinsic_amplitude) in enumerate(groups):
            group_index = self._group_slices[family].start + index
            line_index = np.flatnonzero(self._line_group == group_index)
            if not np.array_equal(self._line_wave[line_index], line_wave):
                raise ValueError(f"{family} line order changed after catalog construction")
            amplitude = intrinsic_amplitude * self._line_transmission_values[line_index]
            matrix[index] = grp2vector(line_wave, amplitude, self.wave, self.lsf_sigma)
            matrix_stick[index] = sticks2vector(line_wave, amplitude, self.wave)
        setattr(self, f"matrix_{family}", matrix)
        setattr(self, f"matrix_{family}_stick", matrix_stick)

    def _prefit_o2(self, flux: np.ndarray, ivar: np.ndarray) -> None:
        intrinsic_aij = self.aij_o2.copy()
        self.aij_o2 = intrinsic_aij * self._line_transmission_values[self._o2_slice]
        try:
            super()._prefit_o2(flux, ivar)
        finally:
            self.aij_o2 = intrinsic_aij


class SkyDecompTelluricCorrectedLinesLSFSpline2D(
    SkyDecompTelluricLinesLSFSpline2D
):
    """Fit a DRP-corrected spectrum with line-centre telluric attenuation."""

    def __init__(
        self,
        *args: Any,
        telluric_calculator: Any,
        pwv_mm: float,
        source_airmass: float,
        drp_transmission: np.ndarray,
        **kwargs: Any,
    ) -> None:
        wave = np.asarray(args[0] if args else kwargs.get("wave"), dtype=float)
        transmission = np.asarray(drp_transmission, dtype=float)
        if transmission.shape != wave.shape:
            raise ValueError("drp_transmission must match the native wavelength grid")
        if np.any(~np.isfinite(transmission)) or np.any(transmission <= 0.0):
            raise ValueError("drp_transmission must be finite and strictly positive")
        self.drp_transmission = transmission.copy()
        self.source_airmass = _positive_scalar(source_airmass, "source_airmass")
        super().__init__(
            *args,
            telluric_calculator=telluric_calculator,
            pwv_mm=pwv_mm,
            line_airmass=self.source_airmass,
            **kwargs,
        )
        inverse = (1.0 / self.drp_transmission)[:, None]
        self._line_components = {
            channel: matrix.multiply(inverse).tocsc()
            for channel, matrix in self._line_components.items()
        }

    def _rebuild_family(self, family: str) -> None:
        super()._rebuild_family(family)
        setattr(
            self,
            f"matrix_{family}",
            getattr(self, f"matrix_{family}") / self.drp_transmission,
        )

    def _prefit_o2(self, flux: np.ndarray, ivar: np.ndarray) -> None:
        moon = self.vector_moon.copy()
        self.vector_moon = moon * self.drp_transmission
        try:
            super()._prefit_o2(
                np.asarray(flux) * self.drp_transmission,
                np.asarray(ivar) / self.drp_transmission**2,
            )
        finally:
            self.vector_moon = moon
        self.vector_o2 /= self.drp_transmission
        self.matrix_o2 /= self.drp_transmission
        self.o2_prefit_bestfit /= self.drp_transmission


class SkyDecompAdam25kTelluricLSFSpline2D(
    SkyDecompTelluricCorrectedLinesLSFSpline2D
):
    """Fit Adam-25k OH ratios with tellurics and a continuous 2-D LSF."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        requested_suffix = kwargs.get("palace_oh_suffix")
        if requested_suffix not in (None, DEFAULT_PALACE_OH_SUFFIX):
            raise ValueError("Adam-25k requires its bundled OH source table")
        kwargs["palace_oh_suffix"] = DEFAULT_PALACE_OH_SUFFIX
        super().__init__(*args, **kwargs)


class SkyDecompAdam25kTelluricSplitZodiLSFSpline2D(
    SkyDecompAdam25kTelluricLSFSpline2D
):
    """Adam-25k lines with production split-zodi continuum and 2-D LSF."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **(SPLIT_ZODI_CONTINUUM_DEFAULTS | kwargs))

    def _finalize_result(self, *args: Any, **kwargs: Any):
        result = super()._finalize_result(*args, **kwargs)
        result.fit_summary += " | continuum_profile=split-zodi-production-v1"
        self.fit_summary = result.fit_summary
        return result


# Compatibility for executed notebooks and checkpoints created before the
# production method received its physical name.
SkyDecompAdam25kNivContinuumLSFSpline2D = (
    SkyDecompAdam25kTelluricSplitZodiLSFSpline2D
)


__all__ = [
    "SkyDecompAdam25kNivContinuumLSFSpline2D",
    "SkyDecompAdam25kTelluricLSFSpline2D",
    "SkyDecompAdam25kTelluricSplitZodiLSFSpline2D",
    "SkyDecompTelluricCorrectedLinesLSFSpline2D",
    "SkyDecompTelluricLinesLSFSpline2D",
    "TELLURIC_CORRECTED_LINES_FIT_MODEL",
    "calculate_drp_transmission",
    "calculate_line_transmission",
    "restore_drp_input",
]
