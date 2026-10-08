"""Per-group coefficient compressor (sqrt / asinh / linear + optional PCA).

Extracted from cell ``dca7fb1c`` of the split-zodi notebook.

Pipeline
--------
For each coefficient group ``g`` (moon, zodi, mesospheric, ...):

    em (n_g)  -->  z = transform(em)          (asinh / linear / sqrt)
    z         -->  z_std = (z - mean) / sd    (standardise columns)
    z_std     -->  scores = z_std @ basis     (basis = PCA of correlation)
    scores    -->  scores[:, kept]            (r_xarm > threshold on held-out)

The inverse puts zeros in dropped-component slots.  Non-negativity is
recovered by clipping the emissivity after the inverse transform (also, the
``sqrt`` "identity" branch inverts via square which is non-negative by
construction).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from .data import airglow_geometry_scale as _default_airglow_geometry_scale

COMPRESSION_XARM_THRESHOLD: float = 0.10
"""Cross-arm correlation threshold for PCA-component selection.

Kept at 0.10 (was 0.55 pre-2026-08-19b).  0.10 retains all 29 moon PCs;
0.5 A/B regressed test ``mean_eRMSE`` by +37% (25.6 -> 35.2) and sWRMSE
by +50% in coefficient space — reverted.  Only the moon group has
``use_pca=True``, so no other group changes at any threshold.
"""

COMPRESSION_TRANSFORM_BY_GROUP: dict[str, str] = {
    "mesospheric": "asinh",
    # 2026-08-19b: moon 'asinh' -> 'sqrt'.  asinh compressed moon coefs so
    # aggressively that the round-trip lost precision at extreme-blue knots
    # (where moon flux is large).  sqrt is milder while still bounding the
    # bright-moon-row loss share.  Encoder-free compressor round-trip rms
    # in 3600-3800 A: linear 0.009, sqrt 0.005, asinh 8.25.
    "moon": "sqrt",
    "atomic": "sqrt",
    "continuum": "sqrt",
    "ionospheric": "sqrt",
    "other": "sqrt",
}

COMPRESSION_USE_PCA_BY_GROUP: dict[str, bool] = {
    "zodi": False,
    # 2026-08-11: asinh-identity beats PCA + xarm truncation on OH.
    # 2026-08-21 A/B re-test at wd=1e-4 + fixed-pWRMSE pipeline: PCA still
    # regressed by +50% coefficient-space sWRMSE (though only 3-7% in pixel
    # space); the OH tail dominates eRMSE, not flux.  Kept False.
    "mesospheric": False,
    "moon": True,
    "atomic": False,
    "continuum": False,
    "ionospheric": False,
    "other": False,
}

# Percentiles used to clip asinh / log inverses to the training envelope.
_CLIP_LO_PCT, _CLIP_HI_PCT = 0.1, 99.9


def score_forward(x: np.ndarray, kind: str, per_col_scale: np.ndarray) -> np.ndarray:
    """Element-wise forward transform (linear / sqrt / asinh / log)."""

    x = np.asarray(x, dtype=np.float64)
    if kind == "linear":
        return x / per_col_scale[None, :]
    if kind == "sqrt":
        return np.sqrt(np.clip(x, 0.0, None))
    if kind == "asinh":
        return np.arcsinh(x / per_col_scale[None, :])
    if kind == "log":
        return np.log(np.clip(x / per_col_scale[None, :], 1e-6, None))
    raise ValueError(f"unknown score transform {kind!r}")


def score_inverse(y: np.ndarray, kind: str, per_col_scale: np.ndarray) -> np.ndarray:
    """Element-wise inverse transform (see :func:`score_forward`)."""

    y = np.asarray(y, dtype=np.float64)
    if kind == "linear":
        return y * per_col_scale[None, :]
    if kind == "sqrt":
        return np.square(np.clip(y, 0.0, None))
    if kind == "asinh":
        return np.sinh(y) * per_col_scale[None, :]
    if kind == "log":
        return np.exp(y) * per_col_scale[None, :]
    raise ValueError(f"unknown score transform {kind!r}")


def fit_group_compressor(
    em_near_g: np.ndarray,
    em_far_g: np.ndarray,
    em_sci_g: np.ndarray,
    train_idx: np.ndarray,
    held_idx: np.ndarray,
    kind: str,
    use_pca: bool,
    xarm_threshold: float,
) -> dict[str, Any]:
    """Fit one group's compressor.  Returns a dict of matrices + metadata."""

    em_near_g = np.asarray(em_near_g, dtype=np.float64)
    em_far_g = np.asarray(em_far_g, dtype=np.float64)
    em_sci_g = np.asarray(em_sci_g, dtype=np.float64)
    n_g = em_near_g.shape[1]

    pos = em_near_g[train_idx].copy()
    pos[pos <= 0] = np.nan
    per_col_scale = np.nanmedian(pos, axis=0)
    per_col_scale = np.where(
        np.isfinite(per_col_scale) & (per_col_scale > 0), per_col_scale, 1.0
    )

    z_near = score_forward(em_near_g, kind, per_col_scale)
    z_far = score_forward(em_far_g, kind, per_col_scale)
    z_sci = score_forward(em_sci_g, kind, per_col_scale)

    pooled_tr = np.vstack([z_near[train_idx], z_far[train_idx], z_sci[train_idx]])
    finite = np.isfinite(pooled_tr).all(axis=1)
    pooled_tr = pooled_tr[finite]
    if pooled_tr.shape[0] < max(20, n_g // 2):
        raise RuntimeError("not enough finite training rows to standardise this group")
    mean_vec = pooled_tr.mean(axis=0)
    sd_vec = pooled_tr.std(axis=0)
    sd_vec = np.where(sd_vec > 1e-12, sd_vec, 1.0)
    # Robust percentiles cap asinh/log inverse clip range to avoid tail blowup
    # when a handful of anomalous decomposition rows push the raw min/max.
    y_train_min = np.nanpercentile(pooled_tr, _CLIP_LO_PCT, axis=0)
    y_train_max = np.nanpercentile(pooled_tr, _CLIP_HI_PCT, axis=0)

    if not use_pca:
        return {
            "kind": kind,
            "use_pca": False,
            "per_col_scale": per_col_scale,
            "mean_vec": mean_vec,
            "sd_vec": sd_vec,
            "basis": np.eye(n_g),
            "kept": np.arange(n_g, dtype=int),
            "xarm_kept": np.full(n_g, np.nan),
            "n_coef_group": n_g,
            "y_train_min": y_train_min,
            "y_train_max": y_train_max,
        }

    centered = (pooled_tr - mean_vec) / sd_vec
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order]

    scores_near_h = ((z_near[held_idx] - mean_vec) / sd_vec) @ basis
    scores_far_h = ((z_far[held_idx] - mean_vec) / sd_vec) @ basis
    finite_h = (
        np.isfinite(scores_near_h).all(axis=1)
        & np.isfinite(scores_far_h).all(axis=1)
    )
    a = scores_near_h[finite_h] - scores_near_h[finite_h].mean(axis=0)
    b = scores_far_h[finite_h] - scores_far_h[finite_h].mean(axis=0)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    xarm = np.where(
        denom > 0, (a * b).sum(axis=0) / np.maximum(denom, 1e-300), 0.0
    )
    kept = np.flatnonzero(xarm > float(xarm_threshold))

    return {
        "kind": kind,
        "use_pca": True,
        "per_col_scale": per_col_scale,
        "mean_vec": mean_vec,
        "sd_vec": sd_vec,
        "basis": basis,
        "kept": kept,
        "xarm_kept": xarm[kept],
        "xarm_full": xarm,
        "n_coef_group": n_g,
        "y_train_min": y_train_min,
        "y_train_max": y_train_max,
    }


def apply_group_compressor(comp: Mapping[str, Any], em_g: np.ndarray) -> np.ndarray:
    """Transform emissivity to compressed scores ``(n_rows, len(kept))``."""

    em_g = np.asarray(em_g, dtype=np.float64)
    z = score_forward(em_g, comp["kind"], comp["per_col_scale"])
    centered = (z - comp["mean_vec"]) / comp["sd_vec"]
    scores_full = centered @ comp["basis"]
    return scores_full[:, comp["kept"]]


def inverse_group_compressor(
    comp: Mapping[str, Any],
    scores: np.ndarray,
    jensen_correction: np.ndarray | None = None,
) -> np.ndarray:
    """Compressed scores → emissivity ``(n_rows, n_g)``.

    ``jensen_correction``, when supplied, is a per-coefficient or per-row
    multiplicative factor applied after the naive inverse.  Introduced to
    cancel the sinh-mean shrinkage of asinh compressors
    (``E[sinh(y_true) | y_pred] ≈ sinh(y_pred) * E[cosh(delta)]``), it is
    now used more broadly as an empirical mean-bias null.  Clipped to
    ``[0.5, 2.0]`` in the trainer before it lands here.
    """

    scores = np.asarray(scores, dtype=np.float64)
    n_pc = comp["basis"].shape[1]
    full = np.zeros((scores.shape[0], n_pc), dtype=np.float64)
    full[:, comp["kept"]] = scores
    centered = full @ comp["basis"].T
    z = centered * comp["sd_vec"] + comp["mean_vec"]
    if comp["kind"] in ("asinh", "log") and "y_train_min" in comp:
        _margin = 0.5
        z = np.clip(
            z, comp["y_train_min"] - _margin, comp["y_train_max"] + _margin
        )
    x = score_inverse(z, comp["kind"], comp["per_col_scale"])
    if jensen_correction is not None:
        jc = np.asarray(jensen_correction, dtype=np.float64)
        if jc.ndim == 1:
            if jc.shape != (x.shape[1],):
                raise ValueError(
                    f"jensen_correction shape {jc.shape} must match n_g={x.shape[1]}"
                )
            x = x * jc[None, :]
        elif jc.ndim == 2:
            if jc.shape != x.shape:
                raise ValueError(
                    f"jensen_correction 2D shape {jc.shape} must match x.shape={x.shape}"
                )
            x = x * jc
        else:
            raise ValueError(
                f"jensen_correction must be 1-D or 2-D; got ndim={jc.ndim}"
            )
    return x


def fit_all_group_compressors(
    filtered: Mapping[str, Any],
    group_indices: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    held_idx: np.ndarray,
    airglow_geometry_scale: Callable[..., np.ndarray] | None = None,
    transforms_by_group: Mapping[str, str] | None = None,
    use_pca_by_group: Mapping[str, bool] | None = None,
    xarm_threshold: float = COMPRESSION_XARM_THRESHOLD,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit compressors for every group in ``group_indices``.

    Parameters
    ----------
    filtered
        The corpus dict (``coef_near`` / ``coef_far`` / ``coef_sci`` +
        ``ctx_*`` + ``ctx_names`` + optional ``coef_wavelengths_a`` /
        ``coef_extinction_k``).
    group_indices
        ``{group_name: np.ndarray}`` mapping coefficient names to column
        indices, produced by ``data._build_group_indices``.
    train_idx, held_idx
        Row indices used for fitting (train) and for cross-arm-correlation
        component selection (held-out val).
    airglow_geometry_scale
        Callback with the same signature as ``data.airglow_geometry_scale``;
        injected to avoid a cyclic module import.
    """

    if transforms_by_group is None:
        transforms_by_group = COMPRESSION_TRANSFORM_BY_GROUP
    if use_pca_by_group is None:
        use_pca_by_group = COMPRESSION_USE_PCA_BY_GROUP
    if airglow_geometry_scale is None:
        airglow_geometry_scale = _default_airglow_geometry_scale

    n_coef = filtered["coef_near"].shape[1]
    geom_kwargs = dict(
        ctx_names=filtered["ctx_names"],
        group_indices=group_indices,
        n_coef=n_coef,
        coef_wavelengths_a=filtered.get("coef_wavelengths_a"),
        coef_extinction_k=filtered.get("coef_extinction_k"),
    )
    sc_near = airglow_geometry_scale(filtered["ctx_near"], **geom_kwargs)
    sc_far = airglow_geometry_scale(filtered["ctx_far"], **geom_kwargs)
    sc_sci = airglow_geometry_scale(filtered["ctx_sci"], **geom_kwargs)
    em_near = np.asarray(filtered["coef_near"], dtype=np.float64) / sc_near
    em_far = np.asarray(filtered["coef_far"], dtype=np.float64) / sc_far
    em_sci = np.asarray(filtered["coef_sci"], dtype=np.float64) / sc_sci

    compressors: dict[str, dict[str, Any]] = {}
    if verbose:
        print(
            f"Fitting per-group compressors on {train_idx.size} training rows, "
            f"{held_idx.size} held-out rows (xarm_threshold={xarm_threshold:.2f})."
        )
        print(
            f"  {'group':<14s} {'n_coef':>6s} {'transform':>9s} {'PCA':>4s} "
            f"{'n_score':>7s} {'median r':>8s} {'compression':>11s}"
        )

    total_in, total_out = 0, 0
    for gname, gidx in group_indices.items():
        gidx = np.asarray(gidx, dtype=int)
        n_in = int(gidx.size)
        if n_in == 0:
            continue
        kind = transforms_by_group.get(gname, "sqrt")
        use_pca = bool(use_pca_by_group.get(gname, False)) and n_in >= 3
        comp = fit_group_compressor(
            em_near[:, gidx], em_far[:, gidx], em_sci[:, gidx],
            train_idx=train_idx, held_idx=held_idx,
            kind=kind, use_pca=use_pca, xarm_threshold=xarm_threshold,
        )
        comp["coef_indices"] = gidx
        compressors[gname] = comp
        n_out = int(comp["kept"].size)
        total_in += n_in
        total_out += n_out
        if verbose:
            if comp["use_pca"] and n_out:
                r_med = f"{float(np.median(comp['xarm_kept'])):.3f}"
            else:
                r_med = " n/a  "
            comp_ratio = f"{n_in/max(n_out,1):.1f}x"
            print(
                f"  {gname:<14s} {n_in:>6d} {kind:>9s} "
                f"{('yes' if comp['use_pca'] else 'no'):>4s} "
                f"{n_out:>7d} {r_med:>8s} {comp_ratio:>11s}"
            )

    if verbose:
        print(
            f"  {'TOTAL':<14s} {total_in:>6d} {'':>9s} {'':>4s} "
            f"{total_out:>7d} {'':>8s} "
            f"{total_in/max(total_out,1):>10.1f}x"
        )
    return compressors, geom_kwargs


def compress_coef_err_to_score_sigma(
    coef_phys: np.ndarray,
    coef_err: np.ndarray,
    ctx_phys: np.ndarray,
    compressors: Mapping[str, Mapping[str, Any]],
    geom_kwargs: Mapping[str, Any],
    group_indices: Mapping[str, np.ndarray],
    airglow_geometry_scale: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Propagate per-coef sigma into compressed score space (Jacobian, diagonal only)."""

    if airglow_geometry_scale is None:
        airglow_geometry_scale = _default_airglow_geometry_scale
    coef = np.asarray(coef_phys, dtype=np.float64)
    sig = np.nan_to_num(np.asarray(coef_err, dtype=np.float64), nan=0.0)
    sig = np.clip(sig, 0.0, np.inf)

    scale = airglow_geometry_scale(np.asarray(ctx_phys, dtype=np.float64), **geom_kwargs)
    em = coef / scale
    sig_em2 = (sig / scale) ** 2

    parts: list[np.ndarray] = []
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for gname, gidx in group_indices.items():
        gidx = np.asarray(gidx, dtype=int)
        if gname not in compressors or gidx.size == 0:
            continue
        comp = compressors[gname]
        pcs = np.asarray(comp["per_col_scale"], dtype=np.float64)
        sd_vec = np.asarray(comp["sd_vec"], dtype=np.float64)
        basis = np.asarray(comp["basis"], dtype=np.float64)
        kept = np.asarray(comp["kept"], dtype=int)

        em_g = em[:, gidx]
        sig_em2_g = sig_em2[:, gidx]

        kind = comp["kind"]
        if kind == "linear":
            dzdem = np.broadcast_to(1.0 / pcs[None, :], em_g.shape)
        elif kind == "sqrt":
            dzdem = np.where(
                em_g > 0.0, 0.5 / np.sqrt(np.clip(em_g, 1e-30, None)), 0.0
            )
        elif kind == "asinh":
            dzdem = 1.0 / (
                pcs[None, :] * np.sqrt(1.0 + (em_g / pcs[None, :]) ** 2)
            )
        elif kind == "log":
            dzdem = np.where(
                em_g > 1e-6, 1.0 / (np.clip(em_g, 1e-6, None) * pcs[None, :]), 0.0
            )
        else:
            raise ValueError(f"unknown score transform {kind!r}")

        sig_z2 = (dzdem ** 2) * sig_em2_g
        sig_centered2 = sig_z2 / (sd_vec[None, :] ** 2)

        basis_kept = basis[:, kept]
        sig_scores2 = sig_centered2 @ (basis_kept ** 2)

        parts.append(np.sqrt(np.clip(sig_scores2, 0.0, np.inf)))
        slices[gname] = (offset, offset + parts[-1].shape[1])
        offset += parts[-1].shape[1]

    if not parts:
        return np.empty((coef.shape[0], 0), dtype=np.float64), slices
    return np.concatenate(parts, axis=1).astype(np.float64), slices


def compress_coefs_to_scores(
    coef_phys: np.ndarray,
    ctx_phys: np.ndarray,
    compressors: Mapping[str, Mapping[str, Any]],
    geom_kwargs: Mapping[str, Any],
    group_indices: Mapping[str, np.ndarray],
    airglow_geometry_scale: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Full forward pipeline: physical coefs → concatenated per-group scores."""

    if airglow_geometry_scale is None:
        airglow_geometry_scale = _default_airglow_geometry_scale
    scale = airglow_geometry_scale(np.asarray(ctx_phys, dtype=np.float64), **geom_kwargs)
    em = np.asarray(coef_phys, dtype=np.float64) / scale
    parts: list[np.ndarray] = []
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for gname, gidx in group_indices.items():
        gidx = np.asarray(gidx, dtype=int)
        if gname not in compressors or gidx.size == 0:
            continue
        comp = compressors[gname]
        s = apply_group_compressor(comp, em[:, gidx])
        parts.append(s)
        slices[gname] = (offset, offset + s.shape[1])
        offset += s.shape[1]
    if not parts:
        return np.empty((coef_phys.shape[0], 0), dtype=np.float64), slices
    return np.concatenate(parts, axis=1), slices


def resolve_regime_lift_zodi(
    jc_dict: Mapping[str, Any],
    ctx_phys: np.ndarray,
    ctx_names: list[str] | None,
) -> np.ndarray:
    """Resolve a regime-lift dict to a per-row per-coef lift array.

    Two schemas are supported (auto-detected):

    * ``1d_alt``       — keys ``moon_up`` / ``moon_horizon`` / ``moon_down``.
                         Smooth tanh gate on ``moon_alt`` blends the three
                         regime lifts.
    * ``2d_phase_alt`` — 4 × 3 = 12 keys ``phase_qN_moon_X``.  Row weight
                         is the product of a cyclic Gaussian on ``moon_phase``
                         (against per-quartile midpoints) and the tanh gate
                         on ``moon_alt``.
    """

    # Local import — avoids a hard dependency on ml_utils at module load time.
    from .ml_utils import moon_phase_deg_from_ctx

    n_rows = int(np.asarray(ctx_phys).shape[0])
    schema = "2d_phase_alt" if "phase_q1_moon_up" in jc_dict else "1d_alt"

    ctx_list = list(ctx_names) if ctx_names is not None else []
    if "moon_alt" not in ctx_list:
        for _fallback in ("moon_horizon", "phase_q2_moon_horizon", "phase_q1_moon_horizon"):
            if _fallback in jc_dict:
                _v = np.asarray(jc_dict[_fallback], dtype=np.float64)
                return np.broadcast_to(_v[None, :], (n_rows, _v.size)).copy()
        raise ValueError(
            "regime lift dict has no per-coef vector: " + str(list(jc_dict))
        )

    moon_alt_idx = ctx_list.index("moon_alt")
    alt = np.asarray(ctx_phys[:, moon_alt_idx], dtype=np.float64)
    thr_up = float(jc_dict.get("moon_up_threshold_deg", 10.0))
    thr_dn = float(jc_dict.get("moon_down_threshold_deg", -10.0))
    scale = float(jc_dict.get("boundary_scale_deg", 5.0))
    w_up = 0.5 * (1.0 + np.tanh((alt - thr_up) / scale))
    w_dn = 0.5 * (1.0 - np.tanh((alt - thr_dn) / scale))
    w_h = np.clip(1.0 - w_up - w_dn, 0.0, 1.0)
    _tot_alt = w_up + w_h + w_dn
    _tot_alt = np.where(_tot_alt > 0, _tot_alt, 1.0)
    w_up = w_up / _tot_alt
    w_h = w_h / _tot_alt
    w_dn = w_dn / _tot_alt

    if schema == "1d_alt":
        lift_up = np.asarray(jc_dict["moon_up"], dtype=np.float64)
        lift_h = np.asarray(jc_dict["moon_horizon"], dtype=np.float64)
        lift_d = np.asarray(jc_dict["moon_down"], dtype=np.float64)
        return (
            w_up[:, None] * lift_up[None, :]
            + w_h[:, None] * lift_h[None, :]
            + w_dn[:, None] * lift_d[None, :]
        )

    phase_deg = moon_phase_deg_from_ctx({"ctx_sci": ctx_phys, "ctx_names": ctx_names})
    phase_mids = np.asarray(
        jc_dict.get("phase_quartile_mids_deg", [56.0, 141.0, 218.5, 311.5]),
        dtype=np.float64,
    )
    sigma_phase = float(jc_dict.get("phase_gauss_sigma_deg", 55.0))
    _d = np.abs(phase_deg[:, None] - phase_mids[None, :])
    _d = np.minimum(_d, 360.0 - _d)
    w_phase = np.exp(-0.5 * (_d / sigma_phase) ** 2)
    w_phase = w_phase / np.maximum(w_phase.sum(axis=1, keepdims=True), 1e-30)

    _phase_names = ("phase_q1", "phase_q2", "phase_q3", "phase_q4")
    _alt_names = ("moon_up", "moon_horizon", "moon_down")
    _alt_weights = (w_up, w_h, w_dn)
    n_coef = int(np.asarray(jc_dict["phase_q1_moon_up"]).size)
    out = np.zeros((n_rows, n_coef), dtype=np.float64)
    for _pi, _pname in enumerate(_phase_names):
        for _ai, _aname in enumerate(_alt_names):
            _L = np.asarray(jc_dict[f"{_pname}_{_aname}"], dtype=np.float64)
            out += (w_phase[:, _pi] * _alt_weights[_ai])[:, None] * _L[None, :]
    return out


def expand_scores_to_coefs(
    scores: np.ndarray,
    ctx_phys: np.ndarray,
    compressors: Mapping[str, Mapping[str, Any]],
    group_indices: Mapping[str, np.ndarray],
    geom_kwargs: Mapping[str, Any],
    n_coef: int,
    score_slices: Mapping[str, tuple[int, int]],
    airglow_geometry_scale: Callable[..., np.ndarray] | None = None,
    jensen_corrections: Mapping[str, Any] | None = None,
    coef_upper_bound: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Full inverse pipeline: predicted scores → physical coefs at ctx_phys.

    ``jensen_corrections`` — ``{group_name: per-coefficient factor array}``
    passed through to :func:`inverse_group_compressor`.

    ``coef_upper_bound`` — ``{group_name: per-coefficient upper-bound
    array}``.  Predictions above the bound are clipped to it (§11 item 11):
    defensive guard against asinh-inverse blowups when the trunk output
    lands outside the training envelope; does not touch training.
    """

    if airglow_geometry_scale is None:
        airglow_geometry_scale = _default_airglow_geometry_scale
    scale = airglow_geometry_scale(np.asarray(ctx_phys, dtype=np.float64), **geom_kwargs)
    em = np.zeros((scores.shape[0], n_coef), dtype=np.float64)
    for gname, gidx in group_indices.items():
        gidx = np.asarray(gidx, dtype=int)
        if gname not in compressors or gidx.size == 0:
            continue
        lo, hi = score_slices[gname]
        _jc = None if jensen_corrections is None else jensen_corrections.get(gname)
        if isinstance(_jc, dict) and ("moon_up" in _jc or "phase_q1_moon_up" in _jc):
            _jc = resolve_regime_lift_zodi(_jc, ctx_phys, geom_kwargs.get("ctx_names"))
        em[:, gidx] = inverse_group_compressor(
            compressors[gname], scores[:, lo:hi], jensen_correction=_jc
        )
    coef = em * scale
    coef = np.clip(coef, 0.0, None)
    if isinstance(coef_upper_bound, dict):
        for gname, gidx in group_indices.items():
            _ub = coef_upper_bound.get(gname)
            if _ub is None:
                continue
            gidx = np.asarray(gidx, dtype=int)
            if gidx.size == 0:
                continue
            coef[:, gidx] = np.minimum(
                coef[:, gidx], np.asarray(_ub, dtype=np.float64)[None, :]
            )
    return coef


__all__ = [
    "COMPRESSION_XARM_THRESHOLD",
    "COMPRESSION_TRANSFORM_BY_GROUP",
    "COMPRESSION_USE_PCA_BY_GROUP",
    "score_forward",
    "score_inverse",
    "fit_group_compressor",
    "apply_group_compressor",
    "inverse_group_compressor",
    "fit_all_group_compressors",
    "compress_coef_err_to_score_sigma",
    "compress_coefs_to_scores",
    "resolve_regime_lift_zodi",
    "expand_scores_to_coefs",
]
