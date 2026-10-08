# Plot worst-case reconstructions: SCI target plus sky1/sky2 input self-recon checks.
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.io import fits
from sky_decomp.moon_zodi_model import LSF_FWHM_TO_SIGMA
from sky_decomp.result_io import (load_lsf_surface_cubes,
                                  lsf_surface_state_from_cubes)
from mlp_predictor.data import make_telluric_row_lookup

from pathlib import Path

required = [
    "rmse_subset_results",
    "reconstruct_with_lsf",
    "load_lsf_state_if_available",
    "load_o2_vector_if_available",
    "_infer_base_dir_for_reconstruction",
    "predict_sci_coefficients_default",
    "mlp_artifacts",
    "build_triplet_coef_dataset",
    "context_cols",
    "FACTOR",
    "_DECOMP_SUFFIX",
]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError("Run the batch RMSE subset cell first. Missing: " + ", ".join(missing))

# Telluric variant: see full_spectrum_batch_rmse for why the basis is per row.
# Telluric transmission is looked up BY ROW, so it has to come from the file
# these row indices address.  The shared TELLURIC_ROW_FOR is bound to
# `input_fits_for_basis` (the every10 stack), and the batch-RMSE cell now hands
# us corpus rows -- feeding those to it raises IndexError, or silently returns
# another exposure's transmission if the index happens to be in range.  Bound
# below, once EVERY10_INPUT has been resolved from rmse_subset_results.

n_worst = 15
row_pos = np.asarray(rmse_subset_results["row_positions"], dtype=int)
row_idx = np.asarray(rmse_subset_results["row_indices"], dtype=int)
# `row_idx` are EVERY10 rows and are used to index the every10 arrays below.
# They must NOT be what the labels show: every10 row N is a different
# spectrum from row N of the corpus tables.  `row_labels` carries expnum
# plus the full-corpus row, which is resolvable in any table.
row_label = list(rmse_subset_results.get("row_labels")
                 or [f"src row {int(r)}" for r in row_idx])
sci_rmse = np.asarray(rmse_subset_results["sci_rmse"], dtype=np.float64)
near_rmse = np.asarray(rmse_subset_results.get("near_rmse", np.full_like(sci_rmse, np.nan)), dtype=np.float64)
far_rmse = np.asarray(rmse_subset_results.get("far_rmse", np.full_like(sci_rmse, np.nan)), dtype=np.float64)

finite = np.isfinite(sci_rmse)
if not finite.any():
    raise RuntimeError("No finite sci_rmse values in rmse_subset_results")
worst_local = np.flatnonzero(finite)[np.argsort(sci_rmse[finite])[::-1]][: min(n_worst, int(finite.sum()))]

if "coef_sci_pred" in globals() and np.asarray(coef_sci_pred).shape[0] == sci_rmse.shape[0]:
    coef_pred_subset = np.asarray(coef_sci_pred, dtype=np.float64)
else:
    _e10_stem = f"{DECOMP_DATA_ROOT}/{DECOMP_STEM}_every10"
    _e10_suffix = _DECOMP_SUFFIX
    EVERY10_INPUT = f"{_e10_stem}.fits"
    EVERY10_NEAR = f"{_e10_stem}_decomp_sky1{_e10_suffix}.fits"
    EVERY10_FAR = f"{_e10_stem}_decomp_sky2{_e10_suffix}.fits"
    EVERY10_SCI = f"{_e10_stem}_decomp_sci{_e10_suffix}.fits"
    e10_triplet_plot = build_triplet_coef_dataset(
        input_fits_path=EVERY10_INPUT,
        sky_near_decomp_fits_path=EVERY10_NEAR,
        sky_far_decomp_fits_path=EVERY10_FAR,
        sci_decomp_fits_path=EVERY10_SCI,
        context_columns=context_cols,
        return_chi2=True,
    )
    coef_pred_subset = predict_sci_coefficients_default(
        mlp_artifacts,
        coef_near_phys=e10_triplet_plot["coef_near"][row_pos],
        coef_far_phys=e10_triplet_plot["coef_far"][row_pos],
        ctx_near_phys=e10_triplet_plot["ctx_near"][row_pos],
        ctx_far_phys=e10_triplet_plot["ctx_far"][row_pos],
        ctx_sci_phys=e10_triplet_plot["ctx_sci"][row_pos],
    ).astype(np.float64)

if "coef_near_sel" in globals() and np.asarray(coef_near_sel).shape[0] == sci_rmse.shape[0]:
    coef_near_subset = np.asarray(coef_near_sel, dtype=np.float64)
else:
    coef_near_subset = np.asarray(e10_triplet_plot["coef_near"][row_pos], dtype=np.float64)

if "coef_far_sel" in globals() and np.asarray(coef_far_sel).shape[0] == sci_rmse.shape[0]:
    coef_far_subset = np.asarray(coef_far_sel, dtype=np.float64)
else:
    coef_far_subset = np.asarray(e10_triplet_plot["coef_far"][row_pos], dtype=np.float64)

# 2026-08-16: per-row WRMSE for the same subset.
# Two flavours are reported:
#   * wrmse_coef_subset   -- coefficient-space, weighted by decomposition COEF_ERR
#   * near_wrmse_pix / far_wrmse_pix / sci_wrmse_pix -- pixel-space, pulled from
#     rmse_subset_results (populated by the batch RMSE cell using per-pixel sigma
#     from FLUX_SIGMA_TOTAL when the new decompositions land, else propagated
#     from COEF_ERR on the fly).
try:
    weighted_rmse_per_row  # noqa: F821 -- defined by the WRMSE-helper cell
    # `row_pos` indexes whatever table the batch-RMSE cell selected from.
    # Since 2026-09-23 that is `filtered_triplet` (it no longer builds an e10
    # triplet at all), so prefer the true coefficients it already sliced, then
    # filtered_triplet, and only then a legacy e10 triplet.
    if ("coef_sci_sel" in globals()
            and np.asarray(coef_sci_sel).shape[0] == sci_rmse.shape[0]):
        _coef_true_subset = np.asarray(coef_sci_sel, dtype=np.float64)
        _sigma_subset = (np.asarray(coef_err_sci_all, dtype=np.float64)[row_pos]
                         if "coef_err_sci_all" in globals()
                         else np.full_like(_coef_true_subset, np.nan))
    else:
        _e10_wrmse_src = globals().get("e10_triplet_plot",
                                       globals().get("e10_triplet",
                                                     globals().get("filtered_triplet")))
        if _e10_wrmse_src is None:
            raise NameError("no triplet available for sWRMSE_coef lookup")
        _coef_true_subset = np.asarray(_e10_wrmse_src["coef_sci"][row_pos],
                                       dtype=np.float64)
        _sigma_raw = _e10_wrmse_src.get("coef_err_sci", None)
        if _sigma_raw is None:
            _sigma_subset = np.full_like(_coef_true_subset, np.nan)
        else:
            _sigma_subset = np.asarray(_sigma_raw[row_pos], dtype=np.float64)
    _gidx_map = (_group_indices_compress if "_group_indices_compress" in globals()
                 else group_indices_sf)
    wrmse_coef_subset = weighted_rmse_per_row(
        _coef_true_subset, coef_pred_subset, _sigma_subset,
        _gidx_map, dict(DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP),
    ).astype(np.float64)
except (NameError, KeyError) as _wrmse_exc:
    wrmse_coef_subset = np.full(coef_pred_subset.shape[0], np.nan, dtype=np.float64)
    print(f"(coef-space wrmse unavailable: {_wrmse_exc}; reporting NaN)")

# Pixel-space WRMSE for the same rows.  Cell 25 populates these when it runs;
# if it has not been run this session, use NaN so the reporting stays graceful.
_pix_ok = ("rmse_subset_results" in globals()
           and all(k in rmse_subset_results for k in ("near_wrmse", "far_wrmse", "sci_wrmse")))
if _pix_ok:
    near_wrmse_pix = np.asarray(rmse_subset_results["near_wrmse"], dtype=np.float64)
    far_wrmse_pix  = np.asarray(rmse_subset_results["far_wrmse"], dtype=np.float64)
    sci_wrmse_pix  = np.asarray(rmse_subset_results["sci_wrmse"], dtype=np.float64)
    _pix_source_note = rmse_subset_results.get("pix_sigma_source", "unknown")
    print(f"pixel pWRMSE sigma source: {_pix_source_note}")
else:
    near_wrmse_pix = np.full_like(sci_rmse, np.nan)
    far_wrmse_pix  = np.full_like(sci_rmse, np.nan)
    sci_wrmse_pix  = np.full_like(sci_rmse, np.nan)
    print("(pixel pWRMSE unavailable: run the batch RMSE eval cell first)")

# Follow the files `row_indices` actually came from.  Since 2026-09-23 the
# batch-RMSE cell scores the FULL held-out split off the corpus stack, so its
# indices are corpus rows; opening an every10 stack and indexing it with them
# would silently return a different spectrum per row (every10 row 493 is
# corpus row 4930).  Fall back to every10 only for a results dict written
# before the paths were recorded.
_e10_stem = f"{DECOMP_DATA_ROOT}/{DECOMP_STEM}_every10"
_e10_suffix = _DECOMP_SUFFIX
_src = rmse_subset_results if "rmse_subset_results" in globals() else {}
EVERY10_INPUT = _src.get("source_input", f"{_e10_stem}.fits")
EVERY10_NEAR = _src.get("source_near", f"{_e10_stem}_decomp_sky1{_e10_suffix}.fits")
EVERY10_FAR = _src.get("source_far", f"{_e10_stem}_decomp_sky2{_e10_suffix}.fits")
EVERY10_SCI = _src.get("source_sci", f"{_e10_stem}_decomp_sci{_e10_suffix}.fits")
print(f"  worst-recon reads {Path(EVERY10_INPUT).name} "
      f"(the file the batch-RMSE row indices belong to)")
if globals().get('TELLURIC_ROW_FOR') is not None:
    _telluric_for = make_telluric_row_lookup(EVERY10_INPUT, verbose=False,
                                             decomp_suffix=_DECOMP_SUFFIX)
# LSF cubes once per arm: load_lsf_state_if_available re-reads LSF_COEF +
# LSF_KNOTS + LSF_META (~80 MB on a corpus file) on every call, and this cell
# makes three per plotted row.
_lsf_cubes_by_path = {}
for _p in (EVERY10_NEAR, EVERY10_FAR, EVERY10_SCI):
    try:
        _lsf_cubes_by_path[_p] = load_lsf_surface_cubes(_p)
    except (KeyError, OSError, ValueError):
        _lsf_cubes_by_path[_p] = None


def _lsf_state_for(path, row):
    _c = _lsf_cubes_by_path.get(path)
    if _c is None:
        return load_lsf_state_if_available(path, row)
    try:
        return lsf_surface_state_from_cubes(_c, int(row))
    except (KeyError, IndexError, ValueError):
        return None

# Selected rows only: the corpus FLUX planes are 925 MB each and this cell
# touches at most a few dozen rows.  Indexed by position in `row_idx` below.
with fits.open(EVERY10_INPUT, memmap=True) as hdul:
    flux_near_all = np.asarray(hdul["FLUX_SKY_NEAR"].data[row_idx], dtype=np.float64)
    flux_far_all = np.asarray(hdul["FLUX_SKY_FAR"].data[row_idx], dtype=np.float64)
    flux_sci_all = np.asarray(hdul["FLUX_SCI"].data[row_idx], dtype=np.float64)
    wave_arr = np.asarray(hdul["WAVE"].data, dtype=np.float64)
    _lsf_full = hdul["LSF_SCI"].data
    lsf_sci_arr = (np.asarray(_lsf_full, dtype=np.float64)
                   if np.ndim(_lsf_full) == 1
                   else np.asarray(_lsf_full[row_idx], dtype=np.float64))

base_dir_guess = _infer_base_dir_for_reconstruction()

n_cases = len(worst_local)
fig = make_subplots(
    rows=n_cases,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.01,
    subplot_titles=[
        f"{row_label[j]} | sky1_pRMSE={float(near_rmse[j]):.3g}, sky2_pRMSE={float(far_rmse[j]):.3g}, sci_pRMSE={float(sci_rmse[j]):.3g}, sci_pWRMSE={float(sci_wrmse_pix[j]):.3g}, sci_sWRMSE={float(wrmse_coef_subset[j]):.3g}"
        for j in worst_local
    ],
)

for case_i, j in enumerate(worst_local):
    rr = int(row_idx[j])
    wave_row = wave_arr if wave_arr.ndim == 1 else np.asarray(wave_arr[rr], dtype=np.float64)
    # NB the flux / LSF arrays above were loaded as data[row_idx], so they are
    # addressed by POSITION j, while the decomposition lookups below still take
    # the file row rr.
    lsf_row = lsf_sci_arr if lsf_sci_arr.ndim == 1 else np.asarray(lsf_sci_arr[j], dtype=np.float64)

    lsf_sigma = lsf_row / LSF_FWHM_TO_SIGMA

    lsf_state_near = _lsf_state_for(EVERY10_NEAR, rr)
    lsf_state_far = _lsf_state_for(EVERY10_FAR, rr)
    lsf_state_sci = _lsf_state_for(EVERY10_SCI, rr)

    lsf_arg_near = lsf_state_near if lsf_state_near is not None else lsf_sigma
    lsf_arg_far = lsf_state_far if lsf_state_far is not None else lsf_sigma
    lsf_arg_sci = lsf_state_sci if lsf_state_sci is not None else lsf_sigma

    o2_near = load_o2_vector_if_available(EVERY10_NEAR, rr)
    o2_far = load_o2_vector_if_available(EVERY10_FAR, rr)
    o2_sci = load_o2_vector_if_available(EVERY10_SCI, rr)

    comps_near = reconstruct_with_lsf(
        wave=wave_row, coef=coef_near_subset[j], lsf=lsf_arg_near,
        n_spline_knots=N_MOON_KNOTS, base_dir=base_dir_guess, o2_vector=o2_near,
        split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
        telluric=(None if _telluric_for is None else _telluric_for('sky1', rr)),
    )
    comps_far = reconstruct_with_lsf(
        wave=wave_row, coef=coef_far_subset[j], lsf=lsf_arg_far,
        n_spline_knots=N_MOON_KNOTS, base_dir=base_dir_guess, o2_vector=o2_far,
        split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
        telluric=(None if _telluric_for is None else _telluric_for('sky2', rr)),
    )
    comps_sci = reconstruct_with_lsf(
        wave=wave_row, coef=coef_pred_subset[j], lsf=lsf_arg_sci,
        n_spline_knots=N_MOON_KNOTS, base_dir=base_dir_guess, o2_vector=o2_sci,
        split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
        telluric=(None if _telluric_for is None else _telluric_for('sci', rr)),
    )

    near_obs = np.asarray(flux_near_all[j], dtype=np.float64) * FACTOR
    far_obs = np.asarray(flux_far_all[j], dtype=np.float64) * FACTOR
    sci_obs = np.asarray(flux_sci_all[j], dtype=np.float64) * FACTOR

    near_rec = np.asarray(comps_near["total"], dtype=np.float64)
    far_rec = np.asarray(comps_far["total"], dtype=np.float64)
    sci_rec = np.asarray(comps_sci["total"], dtype=np.float64)

    row_num = case_i + 1
    # All 6 spectra in one panel with different colors
    fig.add_trace(
        go.Scattergl(
            x=wave_row, y=near_obs, mode="lines", name="sky1_obs",
            line=dict(color="#1f77b4", width=1.2),
            showlegend=(row_num == 1),
            legendgroup="sky1_obs",
        ),
        row=row_num, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=wave_row, y=near_rec, mode="lines", name="sky1_rec",
            line=dict(color="#1f77b4", width=0.6, dash="dash"),
            showlegend=(row_num == 1),
            legendgroup="sky1_rec",
        ),
        row=row_num, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=wave_row, y=far_obs, mode="lines", name="sky2_obs",
            line=dict(color="#ff7f0e", width=1.2),
            showlegend=(row_num == 1),
            legendgroup="sky2_obs",
        ),
        row=row_num, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=wave_row, y=far_rec, mode="lines", name="sky2_rec",
            line=dict(color="#ff7f0e", width=0.6, dash="dash"),
            showlegend=(row_num == 1),
            legendgroup="sky2_rec",
        ),
        row=row_num, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=wave_row, y=sci_obs, mode="lines", name="sci_obs",
            line=dict(color="#2ca02c", width=1.2),
            showlegend=(row_num == 1),
            legendgroup="sci_obs",
        ),
        row=row_num, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=wave_row, y=sci_rec, mode="lines", name="sci_rec",
            line=dict(color="#2ca02c", width=0.6, dash="dash"),
            showlegend=(row_num == 1),
            legendgroup="sci_rec",
        ),
        row=row_num, col=1,
    )

fig.update_xaxes(title_text="Wavelength [A]", row=n_cases, col=1)
fig.update_yaxes(title_text=f"Flux (x{FACTOR:.3g})", type='log')
fig.update_layout(
    template="plotly_white",
    title=f"Worst SCI reconstruction cases: All arms overlaid (top {n_cases})",
    height=max(540, 380 * n_cases),
    margin=dict(l=70, r=20, t=90, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
)
fig.show()

print("Worst-case rows plotted (descending sci_pRMSE):")
print("  columns: sRMSE = per-row spectral RMSE (over pixels or coefficients);")
print("           pWRMSE = per-row pixel-space WRMSE;")
print("           sWRMSE_coef = per-row spectral WRMSE in coefficient space.")
for rank, j in enumerate(worst_local, start=1):
    print(
        f"  {rank:2d}. {row_label[j]} "
        f"sky1_pRMSE={float(near_rmse[j]):.6g} "
        f"sky1_pWRMSE={float(near_wrmse_pix[j]):.4g} "
        f"sky2_pRMSE={float(far_rmse[j]):.6g} "
        f"sky2_pWRMSE={float(far_wrmse_pix[j]):.4g} "
        f"sci_pRMSE={float(sci_rmse[j]):.6g} "
        f"sci_pWRMSE={float(sci_wrmse_pix[j]):.4g} "
        f"sci_sWRMSE={float(wrmse_coef_subset[j]):.4g}"
    )
