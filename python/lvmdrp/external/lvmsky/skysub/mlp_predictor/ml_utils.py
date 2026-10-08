"""Reproducibility helpers, robust feature scaler, and split-index utilities.

Extracted verbatim (with minimal import fix-ups) from cell 296658b6 of the
split-zodi notebook.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch


class RobustScaler:
    """Median / IQR feature scaler.

    Preferred over ``sklearn.preprocessing.RobustScaler`` inside the trainer
    because it operates purely on numpy + broadcasts without pulling sklearn
    into the notebook environment.
    """

    med_: np.ndarray
    scale_: np.ndarray

    def fit(self, x: np.ndarray) -> "RobustScaler":
        x = np.asarray(x, dtype=np.float32)
        self.med_ = np.nanmedian(x, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        iqr = q75 - q25
        self.scale_ = np.where(iqr > 1e-8, iqr, 1.0).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.med_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * self.scale_ + self.med_


def set_reproducibility(seed: int = 42) -> None:
    """Seed python, numpy, torch (+ cuda if available)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random per-row train / val / test split."""

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def split_indices_by_night(
    obstime_mjd: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-night hold-out split (night_id = floor(mjd - 0.5))."""

    t = np.asarray(obstime_mjd, dtype=np.float64).reshape(-1)
    if t.size == 0:
        raise ValueError("obstime_mjd is empty")
    if not np.isfinite(t).all():
        raise ValueError("obstime_mjd contains non-finite values")

    night_id = np.floor(t - 0.5).astype(int)
    unique_nights = np.unique(night_id)
    rng = np.random.default_rng(seed)
    shuffled_nights = unique_nights.copy()
    rng.shuffle(shuffled_nights)

    n_nights = shuffled_nights.size
    n_train = int(train_frac * n_nights)
    n_val = int(val_frac * n_nights)
    train_nights = shuffled_nights[:n_train]
    val_nights = shuffled_nights[n_train:n_train + n_val]
    test_nights = shuffled_nights[n_train + n_val:]

    train_idx = np.flatnonzero(np.isin(night_id, train_nights))
    val_idx = np.flatnonzero(np.isin(night_id, val_nights))
    test_idx = np.flatnonzero(np.isin(night_id, test_nights))
    return train_idx, val_idx, test_idx


def moon_phase_deg_from_ctx(
    filtered: dict,
    arm: str = "sci",
) -> np.ndarray:
    """Per-row moon phase in degrees [0, 360), reading whichever encoding is present."""

    names = [str(n) for n in filtered["ctx_names"]]
    ctx = np.asarray(filtered[f"ctx_{arm}"], dtype=np.float64)
    if "moon_phase" in names:
        return ctx[:, names.index("moon_phase")]
    s = names.index("moon_phase_sin")
    c = names.index("moon_phase_cos")
    return np.rad2deg(np.arctan2(ctx[:, s], ctx[:, c])) % 360.0


def split_indices_by_moon_phase(
    obstime_mjd: np.ndarray,
    moon_phase: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Night-level split stratified by moon phase.

    Rows are grouped by night (``floor(mjd - 0.5)``).  Each night is assigned
    a representative moon phase (median across its exposures); nights are
    sorted by that phase and cut into ``n_bins`` equal-count quantiles.
    Within each quantile the nights are shuffled and split into train /
    val / test with the requested fractions rounded per-bin, so val and
    test each cover the full moon-phase range roughly uniformly.  Whole
    nights stay together, so within-night airglow autocorrelation is not
    leaked across splits.
    """

    t = np.asarray(obstime_mjd, dtype=np.float64).reshape(-1)
    phase = np.asarray(moon_phase, dtype=np.float64).reshape(-1)
    if t.size == 0:
        raise ValueError("obstime_mjd is empty")
    if t.shape != phase.shape:
        raise ValueError("obstime_mjd and moon_phase must have the same shape")
    if not np.isfinite(t).all():
        raise ValueError("obstime_mjd contains non-finite values")
    if not np.isfinite(phase).all():
        raise ValueError("moon_phase contains non-finite values")

    night_id = np.floor(t - 0.5).astype(int)
    unique_nights = np.unique(night_id)
    n_nights = unique_nights.size

    night_phase = np.empty(n_nights, dtype=np.float64)
    for k, nid in enumerate(unique_nights):
        night_phase[k] = np.median(phase[night_id == nid])

    rng = np.random.default_rng(seed)
    tie_break = rng.random(n_nights)
    order = np.lexsort((tie_break, night_phase))
    sorted_nights = unique_nights[order]

    n_bins_eff = max(1, min(int(n_bins), n_nights))
    bin_edges = np.linspace(0, n_nights, n_bins_eff + 1, dtype=int)
    test_frac = max(0.0, 1.0 - float(train_frac) - float(val_frac))

    train_list: list[int] = []
    val_list: list[int] = []
    test_list: list[int] = []
    for k in range(n_bins_eff):
        lo, hi = int(bin_edges[k]), int(bin_edges[k + 1])
        block = sorted_nights[lo:hi].copy()
        b = block.size
        if b == 0:
            continue
        perm = rng.permutation(b)
        block = block[perm]
        if b >= 3:
            n_val = max(1, int(round(float(val_frac) * b)))
            n_test = max(1, int(round(test_frac * b)))
            if n_val + n_test >= b:
                n_val = max(1, min(n_val, b - 2))
                n_test = max(1, min(n_test, b - 1 - n_val))
        elif b == 2:
            n_val, n_test = 1, 0
        else:
            n_val, n_test = 0, 0
        val_list.extend(block[:n_val].tolist())
        test_list.extend(block[n_val:n_val + n_test].tolist())
        train_list.extend(block[n_val + n_test:].tolist())

    train_nights = np.asarray(train_list, dtype=int)
    val_nights = np.asarray(val_list, dtype=int)
    test_nights = np.asarray(test_list, dtype=int)

    train_idx = np.flatnonzero(np.isin(night_id, train_nights))
    val_idx = np.flatnonzero(np.isin(night_id, val_nights))
    test_idx = np.flatnonzero(np.isin(night_id, test_nights))
    return train_idx, val_idx, test_idx


def moon_bs_indices_from_names(coef_names: Sequence[object]) -> np.ndarray:
    """Column indices of moon spline (``moon_bs*``) coefficients."""

    return np.asarray(
        [i for i, n in enumerate(coef_names) if str(n).lower().startswith("moon_bs")],
        dtype=int,
    )


def row_spline_roughness(vals: np.ndarray) -> float:
    """Root-mean-second-difference of a coefficient vector (spline-noise proxy)."""

    vals = np.asarray(vals, dtype=np.float64)
    if vals.size < 3:
        return float("nan")
    d2 = vals[2:] - 2.0 * vals[1:-1] + vals[:-2]
    return float(np.sqrt(np.nanmean(d2 * d2)))


__all__ = [
    "RobustScaler",
    "set_reproducibility",
    "split_indices",
    "split_indices_by_night",
    "split_indices_by_moon_phase",
    "moon_phase_deg_from_ctx",
    "moon_bs_indices_from_names",
    "row_spline_roughness",
]
