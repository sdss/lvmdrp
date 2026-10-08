"""Weighted RMSE metrics — coefficient-space and pixel-space.

Consolidates three near-duplicate WRMSE helpers from the notebook:
- ``_per_column_wrmse``      (cell 296658b6)
- ``weighted_rmse_per_row``  (cell wrmse_row_helper)
- ``pixel_wrmse_per_row``    (cell pixel_wrmse_helper)

into three thin wrappers over one shared weight-building routine
``_wrmse_weights_from_sigma``, so future weight-shaping changes touch a
single implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PIXEL_SIGMA_FLOOR_REL = 0.05
"""Relative floor on the per-pixel sigma used by :func:`pixel_wrmse_per_row`."""

PIXEL_SIGMA_HDU_NAMES: tuple[str, ...] = ("FLUX_SIGMA_TOTAL", "FLUX_SIGMA")
"""HDU names accepted by :func:`load_pixel_sigma_if_available` (preference order)."""


def _wrmse_weights_from_sigma(
    sigma: np.ndarray,
    *,
    group_indices: Mapping[str, np.ndarray] | None,
    floor_by_group: Mapping[str, float] | None,
    default_floor_rel: float = 0.05,
) -> np.ndarray:
    """Return WRMSE weights ``1/sigma_eff**2`` mean-normalised column-wise.

    ``sigma`` is (n_row, n_col).  Per-column floor is
    ``floor_rel * median_col(sigma)`` where ``floor_rel`` is looked up from
    ``floor_by_group`` when the coefficient's group is known (via
    ``group_indices``), else ``default_floor_rel``.

    Column-mean normalisation (``w /= mean_row(w)``) keeps the resulting
    WRMSE on the same scale as unweighted RMSE.
    """

    sigma = np.asarray(sigma, dtype=np.float64)
    finite = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan)
    median_col = np.nanmedian(finite, axis=0)
    median_col = np.where(np.isfinite(median_col) & (median_col > 0.0), median_col, 1.0)

    floor_rel_col = np.full(sigma.shape[1], float(default_floor_rel), dtype=np.float64)
    if group_indices is not None and floor_by_group is not None:
        for g, idx in group_indices.items():
            idx = np.asarray(idx, dtype=int)
            if idx.size:
                floor_rel_col[idx] = float(floor_by_group.get(g, default_floor_rel))
    floor_col = floor_rel_col * median_col

    sigma_eff = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, floor_col[None, :])
    sigma_eff = np.maximum(sigma_eff, floor_col[None, :])
    w = 1.0 / (sigma_eff ** 2)
    wcm = np.mean(w, axis=0)
    wcm = np.where(wcm > 0.0, wcm, 1.0)
    return w / wcm[None, :]


def per_column_wrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    group_indices: Mapping[str, np.ndarray],
    floor_by_group: Mapping[str, float],
) -> tuple[np.ndarray, float]:
    """Per-column WRMSE + total (row+column pooled) WRMSE.

    Weights mirror the trainer: sigma is floored per group at
    ``floor_by_group[g] * median_col(sigma)``, per-column normalised to
    ``E[w]=1`` so the mean/median WRMSE stay on the same scale as RMSE.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = y_pred - y_true
    w = _wrmse_weights_from_sigma(
        sigma, group_indices=group_indices, floor_by_group=floor_by_group,
    )
    f = np.isfinite(w) & np.isfinite(resid)
    num_col = np.sum(w * resid ** 2 * f, axis=0)
    den_col = np.sum(w * f, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_col = np.sqrt(np.where(den_col > 0.0, num_col / den_col, np.nan))
    tot_num = float(np.sum(w[f] * resid[f] ** 2))
    tot_den = float(np.sum(w[f]))
    total = float(np.sqrt(tot_num / max(tot_den, 1e-30)))
    return per_col, total


def weighted_rmse_per_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    group_indices: Mapping[str, np.ndarray],
    floor_by_group: Mapping[str, float],
) -> np.ndarray:
    """Per-row coefficient-space WRMSE (rows with no finite weights -> NaN)."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = y_pred - y_true
    w = _wrmse_weights_from_sigma(
        sigma, group_indices=group_indices, floor_by_group=floor_by_group,
    )
    f = np.isfinite(w) & np.isfinite(resid)
    num = np.sum(w * resid ** 2 * f, axis=1)
    den = np.sum(w * f, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(den > 0.0, num / den, np.nan))


def pixel_wrmse_per_row(
    flux_pred: np.ndarray,
    flux_true: np.ndarray,
    sigma_pix: np.ndarray | None = None,
    floor_rel: float = PIXEL_SIGMA_FLOOR_REL,
) -> np.ndarray:
    """Per-row pixel-space WRMSE.

    Parameters
    ----------
    flux_pred, flux_true : (n_row, n_pix) or (n_pix,) arrays
        Predicted and observed spectra, same shape.  A 1-D input is broadcast
        to (1, n_pix).
    sigma_pix : (n_row, n_pix) array or None
        Per-pixel 1 sigma.  ``None`` triggers floor-only weighting (uniform
        weights per row → plain per-row RMSE with median-based floor).
    floor_rel : float
        Relative floor factor — floor per row =
        ``floor_rel * median(|flux_true|)`` per row.
    """

    flux_pred = np.atleast_2d(np.asarray(flux_pred, dtype=np.float64))
    flux_true = np.atleast_2d(np.asarray(flux_true, dtype=np.float64))
    if flux_pred.shape != flux_true.shape:
        raise ValueError(
            f"shape mismatch: pred {flux_pred.shape} vs true {flux_true.shape}"
        )
    resid = flux_pred - flux_true
    w = _pixel_weights(flux_true, sigma_pix, floor_rel)
    mask = np.isfinite(w) & np.isfinite(resid)
    w_safe = np.where(mask, w, 0.0)
    r2_safe = np.where(mask, resid ** 2, 0.0)
    num = np.sum(w_safe * r2_safe, axis=1)
    den = np.sum(w_safe, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(den > 0.0, num / den, np.nan))


def pixel_wrmse_pointwise(
    flux_pred: np.ndarray,
    flux_true: np.ndarray,
    sigma_pix: np.ndarray | None = None,
    floor_rel: float = PIXEL_SIGMA_FLOOR_REL,
) -> np.ndarray:
    """Per-pixel ``w * resid**2`` (before per-row reduction).

    Useful for residual-band diagnostic plots that need the pointwise
    contributions rather than the reduced per-row scalar.
    """

    flux_pred = np.atleast_2d(np.asarray(flux_pred, dtype=np.float64))
    flux_true = np.atleast_2d(np.asarray(flux_true, dtype=np.float64))
    resid = flux_pred - flux_true
    w = _pixel_weights(flux_true, sigma_pix, floor_rel)
    return w * resid ** 2


def _pixel_weights(
    flux_true: np.ndarray,
    sigma_pix: np.ndarray | None,
    floor_rel: float,
) -> np.ndarray:
    """Row-normalised pixel-space weights ``1 / sigma_eff ** 2``."""

    true_abs = np.abs(flux_true)
    median_row = np.nanmedian(true_abs, axis=1, keepdims=True)
    median_row = np.where(np.isfinite(median_row) & (median_row > 0.0), median_row, 1.0)
    floor_row = float(floor_rel) * median_row
    if sigma_pix is None:
        sigma_eff = np.broadcast_to(floor_row, flux_true.shape).copy()
    else:
        s = np.atleast_2d(np.asarray(sigma_pix, dtype=np.float64))
        if s.shape != flux_true.shape:
            raise ValueError(
                f"sigma_pix shape {s.shape} != flux shape {flux_true.shape}"
            )
        sigma_eff = np.where(np.isfinite(s) & (s > 0.0), s, floor_row)
        sigma_eff = np.maximum(sigma_eff, floor_row)
    w = 1.0 / (sigma_eff ** 2)
    wrm = np.nanmean(w, axis=1, keepdims=True)
    wrm = np.where(wrm > 0.0, wrm, 1.0)
    return w / wrm


def load_pixel_sigma_if_available(
    fits_path: str | Path,
    row_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray | None:
    """Return per-pixel sigma from ``fits_path`` or ``None``.

    Returns ``None`` (does NOT raise) when neither HDU name in
    :data:`PIXEL_SIGMA_HDU_NAMES` is present, so upstream reporting can
    degrade gracefully to plain RMSE.  When ``row_indices`` is given, only
    those rows are returned in that order.
    """

    from astropy.io import fits

    try:
        with fits.open(str(fits_path)) as hdul:
            _names = [str(getattr(h, "name", "")).upper() for h in hdul]
            for _wanted in PIXEL_SIGMA_HDU_NAMES:
                if _wanted in _names:
                    _data = np.asarray(hdul[_wanted].data, dtype=np.float64)
                    if row_indices is not None:
                        _data = _data[np.asarray(row_indices, dtype=int)]
                    return _data
    except FileNotFoundError:
        pass
    return None


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    *,
    sigma: np.ndarray | None = None,
    group_indices: Mapping[str, np.ndarray] | None = None,
    floor_by_group: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Distributional summary of per-coefficient error.

    Terminology (also used in §9 of the top notebook doc):
      * eRMSE  — per-coefficient RMSE aggregated over the ensemble of rows
        (axis=0).  ``mean_eRMSE`` / ``median_eRMSE`` are the mean / median
        across the per-coefficient RMSE vector.
      * eWRMSE — weighted counterpart (mirror of the trainer's coef_err
        weighted loss) — reported only when ``sigma`` and per-group
        floor/index info are provided.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    corr = np.empty(y_true.shape[1], dtype=np.float64)
    for j in range(y_true.shape[1]):
        x = y_true[:, j]
        y = y_pred[:, j]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corr[j] = np.nan
        else:
            corr[j] = float(np.corrcoef(x, y)[0, 1])
    out: dict[str, Any] = {
        "model": model_name,
        "mean_eRMSE": float(np.nanmean(rmse)),
        "median_eRMSE": float(np.nanmedian(rmse)),
        "mean_eMAE": float(np.nanmean(mae)),
        "mean_corr": float(np.nanmean(corr)),
        "median_corr": float(np.nanmedian(corr)),
    }
    if sigma is not None and group_indices is not None and floor_by_group is not None:
        wrmse_col, total_wrmse = per_column_wrmse(
            y_true, y_pred, sigma, group_indices, floor_by_group,
        )
        out["mean_eWRMSE"] = float(np.nanmean(wrmse_col))
        out["median_eWRMSE"] = float(np.nanmedian(wrmse_col))
        out["total_eWRMSE"] = float(total_wrmse)
    return out


__all__ = [
    "PIXEL_SIGMA_FLOOR_REL",
    "PIXEL_SIGMA_HDU_NAMES",
    "per_column_wrmse",
    "weighted_rmse_per_row",
    "pixel_wrmse_per_row",
    "pixel_wrmse_pointwise",
    "load_pixel_sigma_if_available",
    "metric_row",
]
