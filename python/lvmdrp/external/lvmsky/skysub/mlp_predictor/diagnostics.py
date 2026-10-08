"""Diagnostic plots + metrics for the split-zodi coefficient predictor.

Each diagnostic cell lives as a readable Python file under
:mod:`.diagnostics_cells` and is exposed via one method
on :class:`Diagnostics`.  The method reads the file, applies any per-call
regex source patches (used for exposing tunable literals as kwargs), and
:func:`exec` s the body against a persistent globals dict so cross-cell
state (e.g. ``rmse_subset_results`` produced by
:meth:`Diagnostics.full_spectrum_batch_rmse` and consumed by
:meth:`Diagnostics.worst_recon`) propagates from one call to the next.

Notebook usage::

    from mlp_predictor.diagnostics import Diagnostics, DiagnosticsContext
    ctx = DiagnosticsContext(filtered_triplet=..., mlp_artifacts=..., ...)
    diag = Diagnostics(ctx)
    diag.naive_baseline()
    diag.amplitude_error_vs_ctx()
    diag.full_spectrum_batch_rmse()
    diag.headline_summary()   # synthesises the three above; recomputes nothing

Ten of the 28 cells are wired into the notebook as of 2026-09-09; the rest
remain callable from a scratch cell when a specific question needs them.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from astropy.io import fits
from astropy.table import Table
from plotly.subplots import make_subplots

from sky_decomp.fit import reconstruct_component_spectra
from sky_decomp.lsf_surface_iterative import LSFSurfaceState, SkyDecompLSFSurfaceIterative
from sky_decomp.result_io import load_lsf_surface_state

from .compressor import (
    compress_coef_err_to_score_sigma,
    compress_coefs_to_scores,
    expand_scores_to_coefs,
)
from .data import (
    ECLIPTIC_FEATURE_NAMES,
    LMC_EXCLUSION,
    PHYSICS_PRIOR_FEATURE_NAMES,
    SMC_EXCLUSION,
    airglow_extinction_matrix,
    airglow_geometry_scale,
    airglow_van_rhijn_matrix,
    build_triplet_coef_dataset,
    split_zodi_reversal_keep_mask,
    sci_continuum_colour_keep_mask,
    diffuse_zeroed_keep_mask,
    sci_continuum_colour_excess,
    SCI_COLOUR_EXCESS_MAX,
    canonical_row_labels,
    split_zodi_decomp_paths,
    load_lsf_state_if_available,
    load_o2_vector_if_available,
    reconstruct_with_lsf,
    resolve_coef_wavelengths_a,
    _angular_separation_deg_vec,
    _augment_triplet_with_ecliptic,
    _augment_triplet_with_physics_priors,
    _augment_triplet_with_moon_model,
    _build_group_indices,
    _decode_cyclic_context,
    _infer_base_dir_for_reconstruction,
    _sun_ecliptic_longitude_deg,
)
from .metrics import (
    load_pixel_sigma_if_available,
    metric_row,
    per_column_wrmse,
    pixel_wrmse_per_row,
    pixel_wrmse_pointwise,
    weighted_rmse_per_row,
)
from .ml_utils import (
    moon_bs_indices_from_names,
    moon_phase_deg_from_ctx,
    row_spline_roughness,
)
from .trainer import (
    DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP,
    predict_sci_coefficients_default,
)

_CELLS_DIR = Path(__file__).parent / "diagnostics_cells"


def _load_cell_source(cell_stem: str) -> str:
    """Return the readable source of a diagnostic cell body."""
    p = _CELLS_DIR / f"{cell_stem}.py"
    if not p.exists():
        raise FileNotFoundError(
            f"diagnostics cell {cell_stem!r} not found at {p} \u2014 the "
            f"diagnostics_cells directory should ship with the package."
        )
    return p.read_text(encoding="utf-8")


def _base_globals() -> dict:
    """Return a globals dict wired for cell-body ``exec``."""

    return {
        "copy": copy,
        "defaultdict": defaultdict,
        "Path": Path,
        "np": np,
        "pd": pd,
        "plt": plt,
        "px": px,
        "go": go,
        "pio": pio,
        "make_subplots": make_subplots,
        "fits": fits,
        "Table": Table,
        "reconstruct_component_spectra": reconstruct_component_spectra,
        "SkyDecompLSFSurfaceIterative": SkyDecompLSFSurfaceIterative,
        "LSFSurfaceState": LSFSurfaceState,
        "load_lsf_surface_state": load_lsf_surface_state,
        "reconstruct_with_lsf": reconstruct_with_lsf,
        "load_lsf_state_if_available": load_lsf_state_if_available,
        "load_o2_vector_if_available": load_o2_vector_if_available,
        "build_triplet_coef_dataset": build_triplet_coef_dataset,
        "split_zodi_reversal_keep_mask": split_zodi_reversal_keep_mask,
        "sci_continuum_colour_keep_mask": sci_continuum_colour_keep_mask,
        "diffuse_zeroed_keep_mask": diffuse_zeroed_keep_mask,
        "sci_continuum_colour_excess": sci_continuum_colour_excess,
        "SCI_COLOUR_EXCESS_MAX": SCI_COLOUR_EXCESS_MAX,
        "canonical_row_labels": canonical_row_labels,
        "split_zodi_decomp_paths": split_zodi_decomp_paths,
        "_angular_separation_deg_vec": _angular_separation_deg_vec,
        "_augment_triplet_with_ecliptic": _augment_triplet_with_ecliptic,
        "_augment_triplet_with_physics_priors": _augment_triplet_with_physics_priors,
        "_augment_triplet_with_moon_model": _augment_triplet_with_moon_model,
        "_build_group_indices": _build_group_indices,
        "_decode_cyclic_context": _decode_cyclic_context,
        "_infer_base_dir_for_reconstruction": _infer_base_dir_for_reconstruction,
        "_sun_ecliptic_longitude_deg": _sun_ecliptic_longitude_deg,
        "airglow_geometry_scale": airglow_geometry_scale,
        "airglow_van_rhijn_matrix": airglow_van_rhijn_matrix,
        "airglow_extinction_matrix": airglow_extinction_matrix,
        "resolve_coef_wavelengths_a": resolve_coef_wavelengths_a,
        "LMC_EXCLUSION": LMC_EXCLUSION,
        "SMC_EXCLUSION": SMC_EXCLUSION,
        "ECLIPTIC_FEATURE_NAMES": ECLIPTIC_FEATURE_NAMES,
        "PHYSICS_PRIOR_FEATURE_NAMES": PHYSICS_PRIOR_FEATURE_NAMES,
        "compress_coefs_to_scores": compress_coefs_to_scores,
        "compress_coef_err_to_score_sigma": compress_coef_err_to_score_sigma,
        "expand_scores_to_coefs": expand_scores_to_coefs,
        "predict_sci_coefficients_default": predict_sci_coefficients_default,
        "DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP": DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP,
        "metric_row": metric_row,
        "_metric_row": metric_row,
        "per_column_wrmse": per_column_wrmse,
        "_per_column_wrmse": per_column_wrmse,
        "weighted_rmse_per_row": weighted_rmse_per_row,
        "pixel_wrmse_per_row": pixel_wrmse_per_row,
        "pixel_wrmse_pointwise": pixel_wrmse_pointwise,
        "load_pixel_sigma_if_available": load_pixel_sigma_if_available,
        "moon_phase_deg_from_ctx": moon_phase_deg_from_ctx,
        "_moon_phase_deg_from_ctx": moon_phase_deg_from_ctx,
        "moon_bs_indices_from_names": moon_bs_indices_from_names,
        "_moon_bs_indices_from_names": moon_bs_indices_from_names,
        "row_spline_roughness": row_spline_roughness,
        "_row_spline_roughness": row_spline_roughness,
    }


def _float_literal(value) -> str:
    """Source literal for a float that may be infinite."""
    v = float(value)
    return repr(v) if np.isfinite(v) else ("float('inf')" if v > 0 else "float('-inf')")


def _smoothing_literal(value) -> str:
    """Source literal for a sky-arm-correction smoothing argument."""
    if value is None:
        return "0.0"
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"residual_smoothing_A must be 'auto', a width in A, or None; got {value!r}")
        return "'auto'"
    return repr(float(value))


@dataclass
class DiagnosticsContext:
    """Runtime state each diagnostic cell reads.

    All fields default to ``None`` so a caller can build a partial context
    (e.g. right after the data-load cell, with only ``filtered_triplet`` +
    ``extras={"triplet": triplet}``) and still run the pre-training
    diagnostics.  Post-training cells that read ``mlp_artifacts`` etc. will
    obviously fail if those fields are still ``None`` — the exec-based
    dispatch does not gate on which fields are populated.
    """

    filtered_triplet: dict = None
    mlp_artifacts: dict = None
    group_compressors: dict = None
    group_indices: dict = None
    geom_kwargs: dict = None
    ensemble_members: list = None
    coef_pred_det: Any = None
    coef_near_all: Any = None
    coef_far_all: Any = None
    coef_sci_all: Any = None
    ctx_near_all: Any = None
    ctx_far_all: Any = None
    ctx_sci_all: Any = None
    coef_err_near_all: Any = None
    coef_err_far_all: Any = None
    coef_err_sci_all: Any = None
    train_idx: Any = None
    val_idx: Any = None
    test_idx: Any = None
    extras: dict = field(default_factory=dict)

    def as_globals_extension(self) -> dict:
        d = {
            "filtered_triplet": self.filtered_triplet,
            "mlp_artifacts": self.mlp_artifacts,
            "group_compressors": self.group_compressors,
            "_group_indices_compress": self.group_indices,
            "group_indices": self.group_indices,
            "compress_geom_kwargs": self.geom_kwargs,
            "geom_kwargs": self.geom_kwargs,
            "_ensemble_members": self.ensemble_members,
            "ensemble_members": self.ensemble_members,
            "coef_pred_det": self.coef_pred_det,
            "coef_near_all": self.coef_near_all,
            "coef_far_all": self.coef_far_all,
            "coef_sci_all": self.coef_sci_all,
            "ctx_near_all": self.ctx_near_all,
            "ctx_far_all": self.ctx_far_all,
            "ctx_sci_all": self.ctx_sci_all,
            "coef_err_near_all": self.coef_err_near_all,
            "coef_err_far_all": self.coef_err_far_all,
            "coef_err_sci_all": self.coef_err_sci_all,
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
        }
        if self.extras:
            d.update(self.extras)
        return d


class Diagnostics:
    """One method per notebook diagnostic cell.

    Each method loads the corresponding readable ``.py`` cell body from
    :mod:`.diagnostics_cells`, applies any per-call regex source patches
    (used to override tunable literals such as ``REQUESTED_ROW``,
    ``n_sample`` / ``rng_seed``, ``_MAX_POINTS`` and ``dpi``), then
    :func:`exec` s the body against a persistent :attr:`_globals` dict so
    cross-cell state propagates.
    """

    def __init__(self, ctx: DiagnosticsContext) -> None:
        self.ctx = ctx
        self._init_globals()

    def _init_globals(self) -> None:
        self._globals = _base_globals()
        self._globals.update(self.ctx.as_globals_extension())

    def _run(self, cell_stem: str, source_patches: list | None = None) -> dict:
        # Self-heal after %autoreload rebinds the class on an existing instance.
        if not hasattr(self, "_globals"):
            self._init_globals()
        # Re-merge fresh base globals so newly added symbols (after a rebuild)
        # are picked up without rebuilding the Diagnostics instance.
        for k, v in _base_globals().items():
            self._globals.setdefault(k, v)
        src = _load_cell_source(cell_stem)
        if source_patches:
            for pattern, replacement in source_patches:
                src = re.sub(pattern, replacement, src, count=1, flags=re.MULTILINE)
        exec(src, self._globals)
        return self._globals

    def coef_hist_prepost(self) -> dict:
        """Notebook cell id=coef-hist-prepost.  Body lives in ``diagnostics_cells/coef_hist_prepost.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('coef_hist_prepost')

    def relationship_scatter_matrix(self) -> dict:
        """Notebook cell id=353da137.  Body lives in ``diagnostics_cells/relationship_scatter_matrix.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('relationship_scatter_matrix')

    def full_spectrum_single_row(self) -> dict:
        """Notebook cell id=4d9933ba.  Body lives in ``diagnostics_cells/full_spectrum_single_row.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('full_spectrum_single_row')

    def full_spectrum_batch_rmse(self) -> dict:
        """Notebook cell id=76237d92.  Body lives in ``diagnostics_cells/full_spectrum_batch_rmse.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('full_spectrum_batch_rmse')

    def worst_recon(self) -> dict:
        """Notebook cell id=worst-rmse-recon.  Body lives in ``diagnostics_cells/worst_recon.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('worst_recon')

    def pipeline_state_check(self) -> dict:
        """Notebook cell id=4c0df634.  Body lives in ``diagnostics_cells/pipeline_state_check.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('pipeline_state_check')

    def training_history(self) -> dict:
        """Per-epoch train/val loss and blend-alpha curves for every member.

        Reads only what the trainer recorded, so it costs nothing and works
        identically on a live ensemble and on one restored from a ``.pt``
        (``save_ensemble`` persists the history since 2026-09-23; older files
        carry none and this raises a message saying so).

        Leaves ``training_history_result`` -- a per-seed DataFrame of
        epochs_run / best_epoch / best_val / final_val / epochs_after_best --
        in the exec globals, or ``None`` when the ensemble carries no history
        (a pre-2026-09-23 file), in which case the cell prints why and skips
        rather than raising.

        Body lives in ``diagnostics_cells/training_history.py``.
        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('training_history')

    def per_seed_vs_ensemble(self) -> dict:
        """Notebook cell id=071885bd.  Body lives in ``diagnostics_cells/per_seed_vs_ensemble.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('per_seed_vs_ensemble')

    def headline_summary(self) -> dict:
        """Consolidated read of the other cells' persisted results.

        Recomputes nothing: reads ``naive_baseline_result``,
        ``amplitude_error_vs_ctx_result`` and ``rmse_subset_results`` out of
        the shared exec globals, so it cannot drift from the cells it
        summarises.  Run those first.
        """
        return self._run('headline_summary')

    def naive_baseline(self) -> dict:
        """Notebook cell id=naive-baseline.  Body lives in ``diagnostics_cells/naive_baseline.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('naive_baseline')

    def rmse_dual_diagnostic(self) -> dict:
        """Notebook cell id=rmse-dual-diagnostic.  Body lives in ``diagnostics_cells/rmse_dual_diagnostic.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('rmse_dual_diagnostic')

    def rmse_worst_stability(self) -> dict:
        """Notebook cell id=rmse-worst-stability.  Body lives in ``diagnostics_cells/rmse_worst_stability.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('rmse_worst_stability')

    def per_lunation_drift(self) -> dict:
        """Notebook cell id=ed4cab99.  Body lives in ``diagnostics_cells/per_lunation_drift.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('per_lunation_drift')

    def per_context_slice(self) -> dict:
        """Notebook cell id=dac09413.  Body lives in ``diagnostics_cells/per_context_slice.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('per_context_slice')

    def sky_arm_zodi_bias(self) -> dict:
        """Notebook cell id=sky-arm-zodi-bias-diagnostic.  Body lives in ``diagnostics_cells/sky_arm_zodi_bias.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('sky_arm_zodi_bias')

    def predictive_uncertainty_ensemble(self) -> dict:
        """Notebook cell id=83d6200a.  Body lives in ``diagnostics_cells/predictive_uncertainty_ensemble.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('predictive_uncertainty_ensemble')

    def resid_over_sigma_per_group(self) -> dict:
        """Notebook cell id=resid-over-sigma-per-group.  Body lives in ``diagnostics_cells/resid_over_sigma_per_group.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('resid_over_sigma_per_group')

    def amplitude_error_vs_ctx(self) -> dict:
        """Body lives in ``diagnostics_cells/amplitude_error_vs_ctx.py``.

        Moon and zodi integrated-amplitude error against every context feature,
        shown SIGNED (can ctx predict the direction of the miss?) and as a
        MAGNITUDE (can ctx predict which rows are missed?).  Returns the
        persistent exec-globals dict for inspection.
        """
        return self._run('amplitude_error_vs_ctx')

    def resid_vs_sigma_per_decile(self) -> dict:
        """Notebook cell id=083a6c59.  Body lives in ``diagnostics_cells/resid_vs_sigma_per_decile.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('resid_vs_sigma_per_decile')

    def truth_conditioned_sigma_calibration(self) -> dict:
        """Notebook cell id=truth-conditioned-sigma-calibration.  Body lives in ``diagnostics_cells/truth_conditioned_sigma_calibration.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('truth_conditioned_sigma_calibration')

    def moon_sigma_investigation(self) -> dict:
        """Notebook cell id=moon-sigma-investigation.  Body lives in ``diagnostics_cells/moon_sigma_investigation.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('moon_sigma_investigation')

    def physical_space_cap(self) -> dict:
        """Notebook cell id=52f70832.  Body lives in ``diagnostics_cells/physical_space_cap.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('physical_space_cap')

    def swrmse_coef_map(self) -> dict:
        """Notebook cell id=swrmse-coef-map.  Body lives in ``diagnostics_cells/swrmse_coef_map.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('swrmse_coef_map')

    def wrmse_vs_ctx_correlation(self) -> dict:
        """Notebook cell id=a64f3841.  Body lives in ``diagnostics_cells/wrmse_vs_ctx_correlation.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('wrmse_vs_ctx_correlation')

    def wrmse_vs_ctx_scatter(self) -> dict:
        """Notebook cell id=655d5df3.  Body lives in ``diagnostics_cells/wrmse_vs_ctx_scatter.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('wrmse_vs_ctx_scatter')

    def residual_ctx_attribution(self) -> dict:
        """Notebook cell id=residual-ctx-attribution.  Body lives in ``diagnostics_cells/residual_ctx_attribution.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('residual_ctx_attribution')

    def sky_arm_disagreement_floor(self) -> dict:
        """Notebook cell id=sky-arm-disagreement-floor.  Body lives in ``diagnostics_cells/sky_arm_disagreement_floor.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('sky_arm_disagreement_floor')

    def wavelength_residual_atlas(self, size: int = 500,
                                  seed: int = 42) -> dict:
        """Wavelength-resolved residual atlas, split moon-down / moon-up.

        ``size``  rows reconstructed.  The figure splits by moon state, so each
                  panel sees roughly half of this -- which is why the default
                  is 500 rather than the 200 it was before the split.  Cost is
                  linear in time (two LSF reconstructions per row) and in
                  memory (~0.7 MB/row across the residual and per-component
                  stacks); the figure is aggregate lines only, ~7 MB at any n.
        ``seed``  numpy rng seed for the sample.

        Body lives in ``diagnostics_cells/wavelength_residual_atlas.py``.
        Returns the persistent exec-globals dict for inspection.
        """
        return self._run(
            'wavelength_residual_atlas',
            source_patches=[
                (r"^n_sample_atlas\s*=\s*\d+", f"n_sample_atlas = {int(size)}"),
                (r"^rng_seed_atlas\s*=\s*\d+", f"rng_seed_atlas = {int(seed)}"),
            ],
        )

    def ensemble_spread_calibration(self) -> dict:
        """Notebook cell id=ensemble-spread-calibration.  Body lives in ``diagnostics_cells/ensemble_spread_calibration.py``.

        Returns the persistent exec-globals dict for inspection.
        """
        return self._run('ensemble_spread_calibration')


    # ------------------------------------------------------------------
    # Hand-written overrides that accept per-call arguments.
    # ------------------------------------------------------------------

    def _resolve_row_from_expnum(self, expnum: int) -> int:
        """Look up the every10-corpus row index matching ``expnum``."""
        import numpy as _np
        from astropy.io import fits as _fits
        data_root = self._globals.get("DECOMP_DATA_ROOT")
        stem = self._globals.get("DECOMP_STEM")
        if data_root is None or stem is None:
            raise KeyError(
                "expnum lookup needs DECOMP_DATA_ROOT and DECOMP_STEM in the "
                "diagnostics context (add both to DiagnosticsContext.extras)."
            )
        e10_input = f"{data_root}/{stem}_every10.fits"
        with _fits.open(e10_input) as h:
            meta = h["META"].data
            colname = next(
                (c for c in meta.columns.names if c.lower() == "expnum"), None,
            )
            if colname is None:
                raise KeyError(f"{e10_input} META has no expnum column")
            matches = _np.flatnonzero(_np.asarray(meta[colname]) == int(expnum))
        if matches.size == 0:
            raise ValueError(f"expnum {expnum} not found in {e10_input}")
        return int(matches[0])

    def full_spectrum_single_row(self, row: int | None = None,
                                 expnum: int | None = None,
                                 show_moon_zodi_model: bool = True,
                                 residual_correction: bool = False,
                                 residual_smoothing_A: float | str | None = "auto",
                                 residual_max_A: float = 5000.0,
                                 line_scaling: bool = False,
                                 line_highpass_A: float = 25.0,
                                 line_prior_sigma: float = 0.1,
                                 line_tilt: bool = True) -> dict:
        """Reconstruct a single every10 row.  Pass ``row`` (every10 index)
        or ``expnum`` (looked up via the every10 META FITS).

        ``residual_correction`` subtracts, in the 4th panel (observed -
        predicted), the sky arms' own decomposition residual: near and far,
        inverse-variance weighted and scaled per pixel by the predicted
        science model over the arm's model
        (``mlp_predictor.sky_arm_correction``).  Panel 4 then shows the
        uncorrected spectrum in gray and the corrected one in red, and the
        corrected chi2 is printed.  Default False.
        ``residual_smoothing_A`` ``'auto'`` (default: 3-pixel median on faint
        rows only, the measured optimum), a running-median width in Angstrom,
        or 0 for unsmoothed.
        ``residual_max_A`` the correction is applied in full up to this and
        tapered to zero over the next 100 A (default 5000 A); pass
        ``float('inf')`` for the full band.
        ``line_scaling``, ``line_highpass_A``, ``line_prior_sigma``,
        ``line_tilt``: rescale the predicted sky lines on this science spectrum
        after the correction above, exactly as in `full_spectrum_batch_rmse`.
        """
        if expnum is not None and row is not None:
            raise TypeError("pass row OR expnum, not both")
        if expnum is not None:
            row = self._resolve_row_from_expnum(expnum)
        if row is None:
            row = 978
        return self._run(
            "full_spectrum_single_row",
            source_patches=[
                (r"^REQUESTED_ROW\s*=\s*\d+", f"REQUESTED_ROW = {int(row)}"),
                (r"^SHOW_MOON_ZODI_MODEL\s*=\s*(?:True|False)",
                 f"SHOW_MOON_ZODI_MODEL = {bool(show_moon_zodi_model)}"),
                (r"^RESIDUAL_CORRECTION\s*=\s*(?:True|False)",
                 f"RESIDUAL_CORRECTION = {bool(residual_correction)}"),
                (r"^RESIDUAL_SMOOTHING_A\s*=\s*.*$",
                 f"RESIDUAL_SMOOTHING_A = {_smoothing_literal(residual_smoothing_A)}"),
                (r"^RESIDUAL_CORRECTION_MAX_A\s*=\s*.*$",
                 f"RESIDUAL_CORRECTION_MAX_A = {_float_literal(residual_max_A)}"),
                (r"^LINE_SCALING\s*=\s*(?:True|False)",
                 f"LINE_SCALING = {bool(line_scaling)}"),
                (r"^LINE_SCALING_HIGHPASS_A\s*=\s*.*$",
                 f"LINE_SCALING_HIGHPASS_A = {float(line_highpass_A)!r}"),
                (r"^LINE_SCALING_PRIOR\s*=\s*.*$",
                 f"LINE_SCALING_PRIOR = {float(line_prior_sigma)!r}"),
                (r"^LINE_SCALING_TILT\s*=\s*(?:True|False)",
                 f"LINE_SCALING_TILT = {bool(line_tilt)}"),
            ],
        )

    def full_spectrum_batch_rmse(self, size: int | None = None, seed: int = 42,
                                 stroked: int = 60, n_workers: int = 8,
                                 split: str = "heldout",
                                 residual_correction: bool = False,
                                 residual_smoothing_A: float | str | None = "auto",
                                 residual_max_A: float = 5000.0,
                                 line_scaling: bool = False,
                                 line_highpass_A: float = 25.0,
                                 line_prior_sigma: float = 0.1,
                                 line_tilt: bool = True) -> dict:
        """Full-spectrum RMSE over the held-out rows.

        ``size``    rows evaluated.  ``None`` (default) takes EVERY gated row
                    in ``split``; an integer caps it, drawn phase-stratified as
                    before.  Every statistic uses all the rows taken.
        ``seed``    rng seed for the stratified draw.  Irrelevant when ``size``
                    is None, since then there is nothing to draw.
        ``stroked`` how many rows are DRAWN as lines in the residual figure.
                    Only the drawing is capped; the RMS envelope, the
                    histograms and every printed number still use all rows.
                    This is the memory knob: five spectrum panels x 12401
                    points per line means an uncapped 500-row figure
                    serialises to ~758 MB and kills the kernel, against ~49 MB
                    at the default 60.  Budget ~0.8 MB per stroked row.
        ``n_workers`` processes for the reconstruction loop (~0.3 s/row
                    serial).  1 forces the serial path; the pool falls back to
                    serial by itself if forking fails.
        ``split``   which rows to score: ``'heldout'`` (validation + test, the
                    default), ``'test'``, ``'val'``, or ``'all'``.  ``'all'``
                    restores the pre-2026-09-23 behaviour of scoring training
                    rows too, which made the numbers optimistic.
        ``residual_correction`` subtracts each row's sky-arm decomposition
                    residual (``mlp_predictor.sky_arm_correction``: near and
                    far, inverse-variance weighted, scaled per pixel by the
                    predicted science model over the arm model) from the first
                    residual panel, and
                    prints and returns the corrected chi2 next to the
                    uncorrected one.  ``sci_residuals`` and the other
                    statistics stay uncorrected.
        ``residual_smoothing_A`` ``'auto'`` (default: 3-pixel median on faint
                    rows only), a running-median width in Angstrom, or 0.
        ``residual_max_A`` the correction is applied in full up to this and
                    tapered to zero over the next 100 A (default 5000 A);
                    ``float('inf')`` for the full band.
        ``line_scaling`` after that correction, refit the brightness of the
                    predicted sky lines on each science spectrum
                    (``mlp_predictor.sky_line_scaling``): one scale per OH
                    vibrational band, per atomic line family and for O2,
                    on a high-pass that removes the science continuum.
                    Scored and plotted together with the correction above.
        ``line_highpass_A`` width of that high-pass (default 25 A).
        ``line_prior_sigma`` 1-sigma prior pulling each scale toward 1.
        ``line_tilt`` also fit a rotational-temperature tilt per OH band.
        """
        if split not in ("heldout", "test", "val", "all"):
            raise ValueError(
                f"split must be 'heldout', 'test', 'val' or 'all'; got {split!r}")
        return self._run(
            "full_spectrum_batch_rmse",
            source_patches=[
                (r"^\s*n_sample\s*=\s*(?:\d+|None)",
                 f"    n_sample = {'None' if size is None else int(size)}"),
                (r"^\s*rng_seed\s*=\s*\d+", f"    rng_seed = {int(seed)}"),
                (r"^\s*n_workers\s*=\s*\d+", f"    n_workers = {int(n_workers)}"),
                (r"^\s*_EVAL_SPLIT\s*=\s*'[a-z]+'",
                 f"    _EVAL_SPLIT = {split!r}"),
                (r"^\s*MAX_RESID_LINES\s*=\s*\d+",
                 f"    MAX_RESID_LINES = {max(int(stroked), 0)}"),
                (r"^\s*RESIDUAL_CORRECTION\s*=\s*(?:True|False)",
                 f"    RESIDUAL_CORRECTION = {bool(residual_correction)}"),
                (r"^\s*RESIDUAL_SMOOTHING_A\s*=\s*.*$",
                 f"    RESIDUAL_SMOOTHING_A = {_smoothing_literal(residual_smoothing_A)}"),
                (r"^\s*RESIDUAL_CORRECTION_MAX_A\s*=\s*.*$",
                 f"    RESIDUAL_CORRECTION_MAX_A = {_float_literal(residual_max_A)}"),
                (r"^\s*LINE_SCALING\s*=\s*(?:True|False)",
                 f"    LINE_SCALING = {bool(line_scaling)}"),
                (r"^\s*LINE_SCALING_HIGHPASS_A\s*=\s*.*$",
                 f"    LINE_SCALING_HIGHPASS_A = {float(line_highpass_A)!r}"),
                (r"^\s*LINE_SCALING_PRIOR\s*=\s*.*$",
                 f"    LINE_SCALING_PRIOR = {float(line_prior_sigma)!r}"),
                (r"^\s*LINE_SCALING_TILT\s*=\s*(?:True|False)",
                 f"    LINE_SCALING_TILT = {bool(line_tilt)}"),
            ],
        )

    def relationship_scatter_matrix(self, max_points: int = 3000,
                                    dpi: int = 110) -> dict:
        """Coefficient-vs-context scatter matrix.  Defaults produce a very large
        PNG (~350 MB framebuffer at ~30 ctx cols); lower ``dpi`` / ``max_points``
        avoid killing the Jupyter kernel with memory pressure."""
        return self._run(
            "relationship_scatter_matrix",
            source_patches=[
                (r"^_MAX_POINTS\s*=\s*\d+", f"_MAX_POINTS = {int(max_points)}"),
                (r"format='png',\s*dpi=\d+", f"format='png', dpi={int(dpi)}"),
            ],
        )


__all__ = [
    "Diagnostics",
    "DiagnosticsContext",
]
