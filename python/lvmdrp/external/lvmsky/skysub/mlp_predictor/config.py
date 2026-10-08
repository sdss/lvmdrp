"""User-facing pipeline configuration.

Only the paths and context columns are configurable now; every other knob
(architecture, optimizer, filter thresholds, split policy) is hard-coded
in the module that consumes it because the deployed pipeline has settled
on one configuration.  Older experiment-tracking dataclasses were removed
2026-08-27 during the dead-code sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default context columns pulled from the science + sky rows.  Mirrors the
# split-zodi notebook's ``context_cols`` list (cell "starter data load").
DEFAULT_CONTEXT_COLUMNS: tuple[str, ...] = (
    "alt",
    "az_sin",
    "az_cos",
    "airmass",
    "moon_sep",
    "moon_alt",
    "moon_az_sin",
    "moon_az_cos",
    "moon_phase_sin",
    "moon_phase_cos",
    "sun_sep",
    "sun_alt",
    "sun_az_sin",
    "sun_az_cos",
    "sci_sep",
    "vanrhijn_87km",
    "vanrhijn_95km",
    "vanrhijn_285km",
    "obstime_day_sin",
    "obstime_day_cos",
    "obstime_lunation_sin",
    "obstime_lunation_cos",
    "obstime_year_sin",
    "obstime_year_cos",
    "f107",
    "f107_81d",
    "kp",
    "ew",
)


@dataclass
class DataConfig:
    """Where the decomposition corpus lives and which flavour to read."""

    decomp_data_root: str = "moon_zodi_spline4"
    decomp_stem: str = "lvmsframe_median_stack_1.2.1_p40_p70"
    palace_dir: str = "../"
    factor: float = 1.0e14
    context_columns: tuple[str, ...] = DEFAULT_CONTEXT_COLUMNS

    # The split-zodi LSF-surface-iterative decomposition is the only flavour
    # produced by the current pipeline; the suffix is hard-coded.
    decomp_suffix: str = "_lsf_surface_iterative_split_zodi"

    @property
    def decomp_prefix(self) -> str:
        return f"{self.decomp_data_root}/{self.decomp_stem}"

    @property
    def input_fits_meta(self) -> str:
        return f"{self.decomp_prefix}_meta_only.fits"

    @property
    def input_fits_for_basis(self) -> str:
        return f"{self.decomp_prefix}_every10.fits"

    @property
    def input_fits_flux(self) -> str:
        """Full-corpus stack, the only file carrying every row's FLUX_* arrays.

        ``input_fits_for_basis`` is the every10 subsample and ``input_fits_meta``
        holds no flux at all, so any filter that has to look at the observed
        spectra of the whole training corpus -- currently the science-continuum
        colour gate -- needs this one.
        """
        return f"{self.decomp_prefix}.fits"

    def coef_fits(self, arm: str) -> str:
        return f"{self.decomp_prefix}_{arm}_meta_coef{self.decomp_suffix}.fits"

    def decomp_fits(self, arm: str) -> str:
        return f"{self.decomp_prefix}_decomp_{arm}{self.decomp_suffix}.fits"


@dataclass
class PipelineConfig:
    """Top-level bundle passed to every stage.  Only ``data`` remains."""

    data: DataConfig = field(default_factory=DataConfig)


__all__ = [
    "DEFAULT_CONTEXT_COLUMNS",
    "DataConfig",
    "PipelineConfig",
]
