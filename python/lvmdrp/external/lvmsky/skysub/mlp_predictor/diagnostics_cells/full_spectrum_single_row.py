# Full-spectrum reconstruction test for a single requested row using default coefficients
import plotly.graph_objects as go

_e10_stem = f"{DECOMP_DATA_ROOT}/{DECOMP_STEM}_every10"
_e10_suffix = _DECOMP_SUFFIX  # inherited from cell 6
EVERY10_INPUT = f"{_e10_stem}.fits"
EVERY10_NEAR = f"{_e10_stem}_decomp_sky1{_e10_suffix}.fits"
EVERY10_FAR = f"{_e10_stem}_decomp_sky2{_e10_suffix}.fits"
EVERY10_SCI = f"{_e10_stem}_decomp_sci{_e10_suffix}.fits"

REQUESTED_ROW = 1

# Overlay the frozen physical Moon/Zodi model on the three flux panels
# (see 5b below).  Costs ~0.25 s per arm; set False to skip it entirely.
SHOW_MOON_ZODI_MODEL = True

# Correct panel 4 (observed - predicted) by the sky arms' own decomposition
# residual (mlp_predictor.sky_arm_correction): near and far, inverse-variance
# weighted, scaled per pixel by predicted-science-model / arm-model, applied in
# full up to RESIDUAL_CORRECTION_MAX_A and tapered to zero over the next 100 A.
# RESIDUAL_SMOOTHING_A is 'auto' (3-pixel median on faint rows only, the
# measured optimum), a running-median width in A, or 0.
RESIDUAL_CORRECTION = False
RESIDUAL_CORRECTION_MAX_A = 5000.0
RESIDUAL_SMOOTHING_A = 'auto'
# Then rescale the predicted sky lines on this science spectrum
# (mlp_predictor.sky_line_scaling): one scale per OH vibrational band (+ a
# rotational-temperature tilt when LINE_SCALING_TILT), atomic line family and
# O2, fitted on a LINEAR high-pass of width LINE_SCALING_HIGHPASS_A that
# removes the science continuum, with a LINE_SCALING_PRIOR pull toward 1.
LINE_SCALING = False
LINE_SCALING_HIGHPASS_A = 25.0
LINE_SCALING_PRIOR = 0.1
LINE_SCALING_TILT = True
_ANY_CORR = RESIDUAL_CORRECTION or LINE_SCALING

required = [
    "mlp_artifacts",
    "predict_sci_coefficients_default",
    "context_cols",
    "build_triplet_coef_dataset",
    "reconstruct_component_spectra",
    "reconstruct_with_lsf",
    "load_lsf_state_if_available",
    "load_o2_vector_if_available",
    "_infer_base_dir_for_reconstruction",
    "_moon_bs_indices_from_names",
    "_row_spline_roughness",
    "sci_continuum_colour_excess",
    "SCI_COLOUR_EXCESS_MAX",
]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError("Run the training + residual-correction cells first. Missing: " + ", ".join(missing))

# 1) Load coefficients/context from every10 decomposition products.
e10_triplet = build_triplet_coef_dataset(
    input_fits_path=EVERY10_INPUT,
    sky_near_decomp_fits_path=EVERY10_NEAR,
    sky_far_decomp_fits_path=EVERY10_FAR,
    sci_decomp_fits_path=EVERY10_SCI,
    context_columns=context_cols,
    return_chi2=False,
)
# ECLIPTIC-CTX-V1: match training-time ctx layout on the e10 triplet.
if '_augment_triplet_with_ecliptic' in globals():
    _augment_triplet_with_ecliptic(e10_triplet, meta_fits_path=EVERY10_INPUT)
# PHYSICS-PRIORS-CTX-V1: same augment on the e10 triplet.
if '_augment_triplet_with_physics_priors' in globals():
    _augment_triplet_with_physics_priors(e10_triplet)
    # MOON-MODEL-CTX-V1: must match the training-time ctx layout, or
    # ctx_names will not line up with the trained ensemble.
    if (globals().get('USE_MOON_MODEL_FEATURE', False)
            and '_augment_triplet_with_moon_model' in globals()):
        _augment_triplet_with_moon_model(e10_triplet, _e10_stem)
n_e10 = int(e10_triplet["n_rows"])
row_index_e10 = np.asarray(e10_triplet["row_index"], dtype=np.int64)
coef_names_e10 = [str(n) for n in e10_triplet["coef_names"]]
moon_idx = _moon_bs_indices_from_names(coef_names_e10)
zodi_idx = np.array(
    [i for i, n in enumerate(coef_names_e10) if n.startswith('Zodi_bs')],
    dtype=np.int64,
)

if moon_idx.size < 3:
    raise RuntimeError("Expected at least 3 Moon_bs coefficients for spline diagnostics")

# 2) Load observed spectra, wavelength grid, and LSF from every10 input.
with fits.open(EVERY10_INPUT) as hdul:
    for ext in ("FLUX_SKY_NEAR", "FLUX_SKY_FAR", "FLUX_SCI", "WAVE", "LSF_SCI"):
        if ext not in hdul:
            raise KeyError(f"Missing extension {ext} in {EVERY10_INPUT}")

    wave_arr = np.asarray(hdul["WAVE"].data, dtype=np.float64)
    flux_near_all = np.asarray(hdul["FLUX_SKY_NEAR"].data, dtype=np.float64)
    flux_far_all = np.asarray(hdul["FLUX_SKY_FAR"].data, dtype=np.float64)
    flux_sci_true_all = np.asarray(hdul["FLUX_SCI"].data, dtype=np.float64)
    lsf_sci_arr = np.asarray(hdul["LSF_SCI"].data, dtype=np.float64)

n_spec, n_wave = flux_sci_true_all.shape
if row_index_e10.size != n_e10:
    raise ValueError(f"Triplet row_index length mismatch: {row_index_e10.size} vs n_rows={n_e10}")
if np.any(row_index_e10 < 0) or np.any(row_index_e10 >= n_spec):
    raise ValueError(
        f"Triplet row_index contains values outside [0, {n_spec - 1}] for {EVERY10_INPUT}"
    )

idx_row = int(REQUESTED_ROW)
# REQUESTED_ROW is an EVERY10 row (it indexes the every10 arrays below), but
# every10 row N is a different spectrum from row N of the corpus tables, so the
# displayed identity must be the canonical one.  `expnum` is unique in every
# META and stable across selections, which is what makes a plotted number
# resolvable in any FITS table.
_row_ident = canonical_row_labels(
    EVERY10_INPUT, [idx_row],
    corpus_meta_fits=f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_meta_only.fits')
ROW_LABEL = str(_row_ident['label'][0])
print(f'  row identity: REQUESTED_ROW={idx_row} (every10)  ->  {ROW_LABEL}')
triplet_pos = np.flatnonzero(row_index_e10 == idx_row)
if triplet_pos.size == 0:
    raise IndexError(
        f"REQUESTED_ROW={idx_row} is not available in aligned triplet rows. "
        f"Choose one of e10_triplet['row_index'] (size={row_index_e10.size})."
    )
triplet_pos = int(triplet_pos[0])

# Normalize WAVE/LSF arrays to per-row vectors, then select requested row.
wave_row = wave_arr if wave_arr.ndim == 1 else wave_arr[idx_row]
lsf_row = lsf_sci_arr if lsf_sci_arr.ndim == 1 else lsf_sci_arr[idx_row]

flux_near_row = flux_near_all[idx_row]
flux_far_row = flux_far_all[idx_row]
flux_sci_true_row = flux_sci_true_all[idx_row]

# 3) Predict SCI coefficients for the requested row using global default path.
coef_pred_row_batch = predict_sci_coefficients_default(
    mlp_artifacts,
    coef_near_phys=e10_triplet["coef_near"][triplet_pos: triplet_pos + 1],
    coef_far_phys=e10_triplet["coef_far"][triplet_pos: triplet_pos + 1],
    ctx_near_phys=e10_triplet["ctx_near"][triplet_pos: triplet_pos + 1],
    ctx_far_phys=e10_triplet["ctx_far"][triplet_pos: triplet_pos + 1],
    ctx_sci_phys=e10_triplet["ctx_sci"][triplet_pos: triplet_pos + 1],
)
coef_pred_row = np.asarray(coef_pred_row_batch[0], dtype=np.float64)

# 3a) Context values for the near-sky, far-sky and science pointings at this row.
#     Cyclic sin/cos pairs are folded back to 0-360 degree axes for readability.
_ctx_names_e10 = list(e10_triplet["ctx_names"])
_ctx_row_stack = np.stack([
    e10_triplet["ctx_near"][triplet_pos],
    e10_triplet["ctx_far"][triplet_pos],
    e10_triplet["ctx_sci"][triplet_pos],
], axis=0)
_ctx_disp_names, _ctx_disp_stack = _decode_cyclic_context(_ctx_names_e10, _ctx_row_stack)
_ctx_row_df = pd.DataFrame({
    "feature": _ctx_disp_names,
    "sky_near": _ctx_disp_stack[0],
    "sky_far": _ctx_disp_stack[1],
    "science": _ctx_disp_stack[2],
})
print(f"Context values at {ROW_LABEL} (sin/cos pairs decoded to degrees):")
print(_ctx_row_df.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
print()

# 3b) Moon_bs coefficient diagnostics (global prior already applied).
coef_near_row = np.asarray(e10_triplet["coef_near"][triplet_pos], dtype=np.float64)
coef_far_row = np.asarray(e10_triplet["coef_far"][triplet_pos], dtype=np.float64)
coef_sci_row = np.asarray(e10_triplet["coef_sci"][triplet_pos], dtype=np.float64)

# Per-row coef_err arrays (LSF-propagated sigma feeds off these when the
# decomposition FITS lacks a FLUX_SIGMA_TOTAL HDU).  Missing HDU -> None,
# and the downstream WRMSE degrades to the median-floor path.
def _row_coef_err(triplet_key):
    _arr = e10_triplet.get(triplet_key)
    if _arr is None:
        return None
    _row = np.asarray(_arr[triplet_pos], dtype=np.float64)
    return _row if np.any(np.isfinite(_row)) else None

coef_err_near_row = _row_coef_err("coef_err_near")
coef_err_far_row  = _row_coef_err("coef_err_far")
coef_err_sci_row  = _row_coef_err("coef_err_sci")
moon_pred = coef_pred_row[moon_idx]
zodi_pred = coef_pred_row[zodi_idx] if zodi_idx.size else np.zeros(0)
zodi_near = coef_near_row[zodi_idx] if zodi_idx.size else np.zeros(0)
zodi_far  = coef_far_row [zodi_idx] if zodi_idx.size else np.zeros(0)
zodi_true = coef_sci_row [zodi_idx] if zodi_idx.size else np.zeros(0)
moon_near = coef_near_row[moon_idx]
moon_far = coef_far_row[moon_idx]
moon_true = coef_sci_row[moon_idx]

# 4) Reconstruct this row for SCI prediction and near/far self-consistency checks.
base_dir_guess = _infer_base_dir_for_reconstruction()

# Prefer the fitted wavelength-dependent LSF surface from each decomp file's
# LSF_COEF/LSF_KNOTS/LSF_META extensions (written by sky_decomp.lsf_surface_iterative);
# fall back to the input FITS's Gaussian LSF_SCI if that isn't present.
_lsf_state_near = load_lsf_state_if_available(EVERY10_NEAR, idx_row)
_lsf_state_far  = load_lsf_state_if_available(EVERY10_FAR,  idx_row)
_lsf_state_sci  = load_lsf_state_if_available(EVERY10_SCI,  idx_row)
_lsf_sigma_fallback = lsf_row / 2.35
print(f"  LSF source per arm ({ROW_LABEL}): "
      f"near={'surface' if _lsf_state_near is not None else 'gaussian (LSF_SCI)'}, "
      f"far={'surface' if _lsf_state_far is not None else 'gaussian (LSF_SCI)'}, "
      f"sci={'surface' if _lsf_state_sci is not None else 'gaussian (LSF_SCI)'}")

# Per-arm unit-integrated O2 templates from the decomposition FITS
# (VECTOR_O2 extension). If absent (older decomp), the O2 basis stays at
# zero -- matching pre-2026-08-10 behaviour. For the predicted-sci
# reconstruction, use the sci-arm template so the shape is anchored on the
# same layer temperature the science pointing had; the amplitude comes
# from the predicted coef['O2_b01'].
_o2_vec_near = load_o2_vector_if_available(EVERY10_NEAR, idx_row)
_o2_vec_far  = load_o2_vector_if_available(EVERY10_FAR,  idx_row)
_o2_vec_sci  = load_o2_vector_if_available(EVERY10_SCI,  idx_row)
print(f"  O2 template per arm ({ROW_LABEL}): "
      f"near={'VECTOR_O2' if _o2_vec_near is not None else 'zero'}, "
      f"far={'VECTOR_O2' if _o2_vec_far is not None else 'zero'}, "
      f"sci={'VECTOR_O2' if _o2_vec_sci is not None else 'zero'}")

# Telluric variant: per-row basis; see full_spectrum_batch_rmse for why.
_telluric_for = globals().get('TELLURIC_ROW_FOR')
_tel_sci  = None if _telluric_for is None else _telluric_for('sci',  int(idx_row))
_tel_near = None if _telluric_for is None else _telluric_for('sky1', int(idx_row))
_tel_far  = None if _telluric_for is None else _telluric_for('sky2', int(idx_row))

comps_sci = reconstruct_with_lsf(
    wave=wave_row,
    coef=coef_pred_row,
    lsf=_lsf_state_sci if _lsf_state_sci is not None else _lsf_sigma_fallback,
    n_spline_knots=N_MOON_KNOTS,
    n_zodi_spline_knots=N_ZODI_KNOTS,
    base_dir=base_dir_guess,
    o2_vector=_o2_vec_sci,
    telluric=_tel_sci,
)
comps_near_from_near = reconstruct_with_lsf(
    wave=wave_row,
    coef=coef_near_row,
    lsf=_lsf_state_near if _lsf_state_near is not None else _lsf_sigma_fallback,
    n_spline_knots=N_MOON_KNOTS,
    n_zodi_spline_knots=N_ZODI_KNOTS,
    base_dir=base_dir_guess,
    o2_vector=_o2_vec_near,
    coef_err=coef_err_near_row,
    telluric=_tel_near,
)
comps_far_from_far = reconstruct_with_lsf(
    wave=wave_row,
    coef=coef_far_row,
    lsf=_lsf_state_far if _lsf_state_far is not None else _lsf_sigma_fallback,
    n_spline_knots=N_MOON_KNOTS,
    n_zodi_spline_knots=N_ZODI_KNOTS,
    base_dir=base_dir_guess,
    o2_vector=_o2_vec_far,
    coef_err=coef_err_far_row,
    telluric=_tel_far,
)
# Reconstruction of the observed sci spectrum from the fitted sci coefs, so panel 3
# can separate the sky-decomposition fit residual (obs vs recon-from-sci-coef)
# from our transfer-model error (recon-from-pred vs recon-from-sci-coef).
comps_sci_true = reconstruct_with_lsf(
    wave=wave_row,
    coef=coef_sci_row,
    lsf=_lsf_state_sci if _lsf_state_sci is not None else _lsf_sigma_fallback,
    n_spline_knots=N_MOON_KNOTS,
    n_zodi_spline_knots=N_ZODI_KNOTS,
    base_dir=base_dir_guess,
    o2_vector=_o2_vec_sci,
    coef_err=coef_err_sci_row,
    telluric=_tel_sci,
)

flux_sci_pred_row = np.asarray(comps_sci["total"], dtype=np.float64) / FACTOR
flux_sci_true_recon_row = np.asarray(comps_sci_true["total"], dtype=np.float64) / FACTOR
flux_near_recon_row = np.asarray(comps_near_from_near["total"], dtype=np.float64) / FACTOR
flux_far_recon_row = np.asarray(comps_far_from_far["total"], dtype=np.float64) / FACTOR

# 5) Single-row metrics.
resid_row = flux_sci_pred_row - flux_sci_true_row
rmse_row = float(np.sqrt(np.mean(resid_row ** 2)))
rmse_row_display = float(rmse_row * FACTOR)
mae_row = float(np.mean(np.abs(resid_row)))

rmse_near_recon = float(np.sqrt(np.mean((flux_near_recon_row - flux_near_row) ** 2)))
rmse_far_recon = float(np.sqrt(np.mean((flux_far_recon_row - flux_far_row) ** 2)))

# Pixel-space WRMSE for the same three arms.  Sigma source order:
#   1. FLUX_SIGMA_TOTAL HDU in the decomposition FITS (future LSF-aware
#      propagator output);
#   2. sigma_total returned by `reconstruct_component_spectra(coef_err=...)`
#      when the corresponding COEF_ERR array is available on this row;
#   3. None -> pixel_wrmse_per_row degrades to a per-row median-floor RMSE.
def _sigma_for_single_row(fits_path, hdu_row_idx, comps_dict, comps_true_dict=None):
    """Prefer FITS HDU sigma; else use comps sigma_total (from coef_err)."""
    _fits_sigma = load_pixel_sigma_if_available(fits_path,
                                                row_indices=[int(hdu_row_idx)])
    if _fits_sigma is not None:
        return np.asarray(_fits_sigma[0], dtype=np.float64) / FACTOR
    # Fallback: use the sigma_total attached to the reconstructed comps dict.
    _sig = comps_dict.get("sigma_total") if isinstance(comps_dict, dict) else None
    if _sig is not None:
        return np.asarray(_sig, dtype=np.float64) / FACTOR
    # As a last resort, try the "true" recomposition's sigma (only meaningful
    # for the sci arm where the true decomposition sigma is more directly
    # comparable to the observation noise floor).
    _sig_alt = (comps_true_dict.get("sigma_total")
                if isinstance(comps_true_dict, dict) else None)
    if _sig_alt is not None:
        return np.asarray(_sig_alt, dtype=np.float64) / FACTOR
    return None

_sig_near_pix = _sigma_for_single_row(EVERY10_NEAR, idx_row, comps_near_from_near)
_sig_far_pix  = _sigma_for_single_row(EVERY10_FAR,  idx_row, comps_far_from_far)
_sig_sci_pix  = _sigma_for_single_row(EVERY10_SCI,  idx_row, comps_sci, comps_sci_true)

wrmse_near_recon = float(pixel_wrmse_per_row(
    flux_near_recon_row, flux_near_row, _sig_near_pix)[0])
wrmse_far_recon  = float(pixel_wrmse_per_row(
    flux_far_recon_row,  flux_far_row,  _sig_far_pix)[0])
wrmse_row_pix    = float(pixel_wrmse_per_row(
    flux_sci_pred_row,   flux_sci_true_row, _sig_sci_pix)[0])

# 5a) Optional correction of panel 4 by the sky arms' decomposition residual.
#     Each arm's residual against its OWN reconstruction is the solar-line
#     pattern the moon/zodi basis misses; the prediction misses the same
#     pattern, so the arms measure it simultaneously.  See sky_arm_correction.
sky_arm_corr_row = np.zeros_like(flux_near_row)
if RESIDUAL_CORRECTION:
    from mlp_predictor.sky_arm_correction import (
        SkyArm as _SkyArm, arm_photon_variance as _arm_var,
        sky_arm_residual_correction as _sky_arm_corr)
    with fits.open(EVERY10_INPUT, memmap=True) as _h_rc:
        _meta_rc = _h_rc["META"].data
        _nf_rc = {a: float(_meta_rc[f"fibers_sky_{a}_used"][idx_row]) for a in ("near", "far")}
    _solar = lambda comps: (np.asarray(comps.get("moon", 0.0), dtype=np.float64)
                            + np.asarray(comps.get("zodi", 0.0), dtype=np.float64)) / FACTOR
    _arms_rc = [
        _SkyArm(flux_near_row, flux_near_recon_row, _solar(comps_near_from_near),
                _arm_var(flux_near_row, wave_row, [_nf_rc["near"]])),
        _SkyArm(flux_far_row, flux_far_recon_row, _solar(comps_far_from_far),
                _arm_var(flux_far_row, wave_row, [_nf_rc["far"]])),
    ]
    sky_arm_corr_row, _rc_info = _sky_arm_corr(
        wave_row, flux_sci_pred_row, _solar(comps_sci), _arms_rc,
        max_wavelength=RESIDUAL_CORRECTION_MAX_A,
        smoothing=RESIDUAL_SMOOTHING_A, return_info=True)
    _band_rc = wave_row < RESIDUAL_CORRECTION_MAX_A
    print(f"  residual_correction: sky-arm residual, "
          + ("full band" if not np.isfinite(RESIDUAL_CORRECTION_MAX_A)
             else f"to {RESIDUAL_CORRECTION_MAX_A:.0f} A (+100 A taper)") + "; "
          f"arms used near/far = {bool(_rc_info['arm_used'][0])}/{bool(_rc_info['arm_used'][1])} "
          f"(blue frac-resid rms {_rc_info['arm_frac_rms'][0]:.3f}/{_rc_info['arm_frac_rms'][1]:.3f}), "
          f"S/N {_rc_info['snr']:.0f}, smoothed={bool(_rc_info['smoothed'])}, "
          f"rms {np.sqrt(np.mean(sky_arm_corr_row[_band_rc] ** 2)) * FACTOR:.4g} (display units)")
# 5a') Optional rescaling of the predicted sky lines, fitted on THIS science
#      spectrum after the sky-arm correction above.  Templates come from the
#      predicted coefficients on the reconstruction's own basis, so they sum to
#      exactly its line components.
line_corr_row = np.zeros_like(flux_sci_pred_row)
line_scales_row = None
if LINE_SCALING:
    if _lsf_state_sci is None:
        print("  line_scaling: skipped (no fitted LSF surface for the sci arm)")
    else:
        from mlp_predictor.data import reconstruction_basis as _recon_basis, \
            science_line_mask_rows as _sci_mask_rows_ls
        from mlp_predictor.sky_arm_correction import arm_photon_variance as _arm_var_ls
        from mlp_predictor.sky_line_scaling import (
            line_templates as _line_templates, sky_line_scaling_correction as _line_scale,
            NEBULAR_LINES_RED as _NEB_RED, UNMEASURED_VELOCITY_KM_S as _NEB_V)
        _mdl_ls, _mats_ls = _recon_basis(
            wave_row, _lsf_state_sci, n_spline_knots=N_MOON_KNOTS,
            n_zodi_spline_knots=N_ZODI_KNOTS, base_dir=base_dir_guess,
            o2_vector=_o2_vec_sci, telluric=_tel_sci)
        _tpl_ls = {k: v / FACTOR for k, v in _line_templates(
            _mdl_ls, _mats_ls, coef_pred_row, tilt=LINE_SCALING_TILT).items()}
        line_corr_row, _ls_info = _line_scale(
            wave_row, flux_sci_true_row, flux_sci_pred_row + sky_arm_corr_row, _tpl_ls,
            variance=_arm_var_ls(flux_sci_true_row, wave_row, None),
            mask=_sci_mask_rows_ls(EVERY10_INPUT, wave_row, flux_sci_true_row, flux_near_row,
                                   extra_lines=_NEB_RED, widen_if_unmeasured_km_s=_NEB_V),
            highpass_A=LINE_SCALING_HIGHPASS_A, prior_sigma=LINE_SCALING_PRIOR,
            return_info=True)
        line_scales_row = _ls_info["scales"]
        print("  line_scaling: fitted scale per template (+- 1 sigma): "
              + ", ".join(f"{k} {v:.3f}+-{_ls_info['sigma'][k]:.3f}"
                          for k, v in line_scales_row.items() if not k.endswith("_tilt"))
              + f"; {_ls_info['n_pix']} line pixels, "
              f"{100 * _ls_info['downweighted']:.0f}% down-weighted"
              + ("" if _ls_info["accepted"] else
                 "; NOT APPLIED: the guard dropped it because it raised the chi2"))
obs_minus_pred_uncorr_row = -resid_row
obs_minus_pred_row = obs_minus_pred_uncorr_row - sky_arm_corr_row - line_corr_row

# 5c) Absolute photon chi2 for this row, per pixel.
#
# Two curves, and the second is what makes the first readable. The predicted
# reconstruction's chi2 alone conflates two questions: a median of 4 looks
# like a transfer failure until the DECOMPOSITION's own fit is shown sitting
# at 3.9 on the same pixels under the same noise model. So the panel carries
#
#   chi2 recon(pred)  how well the PREDICTED coefficients describe this row
#   chi2 recon(sci)   how well the FITTED coefficients do -- the ML floor
#
# Their ratio is the honest statement of what the transfer costs here.
# Conventions match `full_spectrum_batch_rmse` so the two panels are
# comparable: single-fibre scale by default, the shared variance floor, and
# NO renormalisation -- the scale is absolute, and dividing by the median
# would throw away the only thing the absolute model bought.
CHI2_SINGLE_FIBRE = True      # per-fibre noise, not the stacked level
CHI2_EXPTIME_S = 900.0
CHI2_BLUE_MAX_A = 6000.0

chi2_pix_pred = None
chi2_pix_self = None
chi2_row_pred = chi2_row_self = float("nan")
chi2_blue_pred = chi2_blue_self = float("nan")
chi2_row_corr = chi2_blue_corr = float("nan")
try:
    from mlp_predictor.noise import (load_absolute_sensitivity as _load_sens_abs,
                                     photon_variance_absolute as _phot_var_abs,
                                     floor_variance as _floor_var)
except Exception as _exc_c2:                                   # pragma: no cover
    print(f"  [chi2] mlp_predictor.noise unavailable ({type(_exc_c2).__name__}); "
          f"the chi2 panel is omitted.")
else:
    _sens_c2 = np.asarray(_load_sens_abs(wave_row), dtype=np.float64)
    _dwave_c2 = float(np.median(np.diff(wave_row)))
    _nfib_c2 = None
    if not CHI2_SINGLE_FIBRE:
        from astropy.io import fits as _fits_c2
        with _fits_c2.open(EVERY10_INPUT, memmap=True) as _h_c2:
            _cols_c2 = {c.lower(): c for c in _h_c2["META"].columns.names}
            _nf_col = next((_cols_c2[c] for c in ("fibers_sci_used", "fibers_sci")
                            if c in _cols_c2), None)
            if _nf_col is not None:
                _v_c2 = float(np.asarray(_h_c2["META"].data[_nf_col])[idx_row])
                _nfib_c2 = _v_c2 if np.isfinite(_v_c2) and _v_c2 > 0 else None
        if _nfib_c2 is None:
            print("  [chi2] META carries no usable fibre count; the stacked "
                  "level is per-fibre and so an OVER-estimate.")
    # `photon_variance_absolute` and `floor_variance` are row-wise and take a
    # 2-D (n_row, n_wave) array; this cell has one row, so widen and squeeze.
    _var_c2 = _floor_var(_phot_var_abs(flux_sci_true_row[None, :], _sens_c2,
                                       exptime=CHI2_EXPTIME_S,
                                       dwave=_dwave_c2,
                                       n_fibres=(None if _nfib_c2 is None
                                                 else [_nfib_c2])))[0]
    # The mask deliberately does NOT require flux > 0: the loss keeps
    # non-positive pixels by mapping them onto the row's median variance, and
    # dropping them here would measure a different noise model from the one
    # being tested.
    _good_c2 = (np.isfinite(flux_sci_true_row) & np.isfinite(_sens_c2)
                & (_sens_c2 > 0) & np.isfinite(_var_c2) & (_var_c2 > 0))
    # The science field's emission lines, masked exactly as the decomposition
    # masked them (stack reference LSF, windows centred on this row's measured
    # Halpha velocity).  They are the target's light, not sky, and on an HII
    # region a single Halpha pixel can outweigh the rest of the row.
    try:
        from mlp_predictor.data import science_line_mask_rows as _sci_mask_rows
        _sci_mask_row = _sci_mask_rows(EVERY10_INPUT, wave_row,
                                       flux_sci_true_row, flux_near_row)
        _good_c2 &= ~_sci_mask_row
    except Exception as _exc_mask:
        _sci_mask_row = None
        print(f"  [chi2] science-line mask unavailable ({type(_exc_mask).__name__}: "
              f"{_exc_mask}); the chi2 includes the science emission lines.")
    _resid_self_row = flux_sci_true_recon_row - flux_sci_true_row
    chi2_pix_pred = np.where(_good_c2, resid_row ** 2 / _var_c2, np.nan)
    chi2_pix_self = np.where(_good_c2, _resid_self_row ** 2 / _var_c2, np.nan)
    _blue_c2 = _good_c2 & (wave_row < CHI2_BLUE_MAX_A)
    chi2_row_pred = float(np.nanmean(chi2_pix_pred))
    chi2_row_self = float(np.nanmean(chi2_pix_self))
    if _blue_c2.any():
        chi2_blue_pred = float(np.nanmean(chi2_pix_pred[_blue_c2]))
        chi2_blue_self = float(np.nanmean(chi2_pix_self[_blue_c2]))
    if _ANY_CORR:
        # Same pixels and noise, with the correction(s) applied (5a, 5a').
        _chi2_pix_corr = np.where(_good_c2, obs_minus_pred_row ** 2 / _var_c2, np.nan)
        chi2_row_corr = float(np.nanmean(_chi2_pix_corr))
        if _blue_c2.any():
            chi2_blue_corr = float(np.nanmean(_chi2_pix_corr[_blue_c2]))

print("Single-row reconstruction summary (every10, default coefficients)")
print(f"  row index (input file) = {idx_row}")
print(f"  row index (triplet pos) = {triplet_pos}")
print(f"  n_wave     = {n_wave}")
print("  predictor  = deep group-head MLP")
print(
    "  Moon_bs roughness: "
    f"pred={_row_spline_roughness(moon_pred):.4g}, "
    f"near={_row_spline_roughness(moon_near):.4g}, "
    f"far={_row_spline_roughness(moon_far):.4g}, "
    f"sci_true={_row_spline_roughness(moon_true):.4g}"
)
print(f"  near self-recon pRMSE  = {rmse_near_recon:.6g}")
print(f"  near self-recon pWRMSE = {wrmse_near_recon:.6g}")
print(f"  far  self-recon pRMSE  = {rmse_far_recon:.6g}")
print(f"  far  self-recon pWRMSE = {wrmse_far_recon:.6g}")
print(f"  sci row pRMSE          = {rmse_row:.6g}")
print(f"  sci row pWRMSE         = {wrmse_row_pix:.6g}")
print(f"  sci row pRMSE (x{FACTOR:.3g} display units) = {rmse_row_display:.6g}")
print(f"  sci row MAE            = {mae_row:.6g}")
if chi2_pix_pred is not None:
    _c2_scale = "single fibre" if CHI2_SINGLE_FIBRE else "median stack of this row's fibres"
    print(f"  photon chi2/pix ({_c2_scale})")
    print(f"    recon(pred) full / blue = {chi2_row_pred:.4g} / {chi2_blue_pred:.4g}")
    if _ANY_CORR:
        print(f"    recon(pred) + " + " + ".join(
                  n for n, on in (("sky-arm correction", RESIDUAL_CORRECTION),
                                  ("line scaling", LINE_SCALING)) if on)
              + " full / blue = "
              f"{chi2_row_corr:.4g} / {chi2_blue_corr:.4g}")
    print(f"    recon(sci)  full / blue = {chi2_row_self:.4g} / {chi2_blue_self:.4g}"
          f"   <- the decomposition's own floor")
    if np.isfinite(chi2_row_self) and chi2_row_self > 0:
        print(f"    ratio pred/sci          = {chi2_row_pred / chi2_row_self:.3f}"
              f"   (1.0 = the transfer costs nothing on this row)")

# 5b) Training-density audit in regime space. This replaces the earlier
# coef_err_sci percentile audit -- which was mostly a brightness proxy --
# with a direct measure of whether this row sits in a sparsely-covered part
# of the training distribution on the physical failure axes. That is the
# real cause of tail errors on rows the decomposition itself fits fine.
_REGIME_AXES = ('moon_alt', 'moon_fli', 'abs_ecl_beta', 'airmass', 'sun_alt')
_REGIME_K = 10
_REGIME_SPARSE_PERCENTILE = 90.0

def _regime_axis_key(axis_name):
    return 'ecl_beta_deg' if axis_name == 'abs_ecl_beta' else axis_name

def _regime_axis_value(vec, axis_name):
    return np.abs(vec) if axis_name == 'abs_ecl_beta' else vec

_pop_ctx_ss = np.asarray(filtered_triplet['ctx_sci'], dtype=np.float64)
_pop_ctx_names_ss = list(filtered_triplet['ctx_names'])
_train_idx_ss = np.asarray(
    mlp_artifacts.get('train_idx', np.arange(_pop_ctx_ss.shape[0])), dtype=int)
_train_idx_ss = _train_idx_ss[(_train_idx_ss >= 0) & (_train_idx_ss < _pop_ctx_ss.shape[0])]

_axis_names_present = []
_train_cols = []
for _a in _REGIME_AXES:
    _key = _regime_axis_key(_a)
    if _key not in _pop_ctx_names_ss:
        continue
    _train_cols.append(_regime_axis_value(
        _pop_ctx_ss[_train_idx_ss, _pop_ctx_names_ss.index(_key)], _a))
    _axis_names_present.append(_a)

sci_row_regime_audit = {}
sci_row_sparse_regime_flag = False
if not _axis_names_present:
    print()
    print('[regime audit skipped: no regime axes present in ctx_names]')
else:
    _train_stack = np.stack(_train_cols, axis=1)
    _finite_train = np.all(np.isfinite(_train_stack), axis=1)
    _train_stack = _train_stack[_finite_train]
    _mu = _train_stack.mean(axis=0)
    _sd = _train_stack.std(axis=0)
    _sd = np.where(_sd > 0, _sd, 1.0)
    _train_z = (_train_stack - _mu) / _sd

    # Cache the train-side kNN work per (axes, n_train, k) across cell re-runs.
    if '_regime_train_cache' not in globals():
        globals()['_regime_train_cache'] = {}
    _cache_key = tuple(_axis_names_present) + (int(_train_z.shape[0]), _REGIME_K)
    if _cache_key not in _regime_train_cache:
        try:
            from scipy.spatial import cKDTree as _cKDTree
            _tree = _cKDTree(_train_z)
            _train_d, _ = _tree.query(_train_z, k=_REGIME_K + 1)
            _train_knn_dist = _train_d[:, _REGIME_K]
        except ImportError:
            _tree = None
            _dsq_tr = ((_train_z[:, None, :] - _train_z[None, :, :]) ** 2).sum(axis=2)
            _dsq_tr.sort(axis=1)
            _train_knn_dist = np.sqrt(_dsq_tr[:, _REGIME_K])
        _regime_train_cache[_cache_key] = dict(
            tree=_tree, mu=_mu, sd=_sd,
            train_z=_train_z,
            train_stack=_train_stack,
            train_p90=float(np.percentile(
                _train_knn_dist, _REGIME_SPARSE_PERCENTILE)),
            train_knn_dist=_train_knn_dist,
        )
    _cache = _regime_train_cache[_cache_key]

    # Build the query row from the sci arm of e10_triplet for this row.
    _q_ctx = np.asarray(e10_triplet['ctx_sci'][triplet_pos], dtype=np.float64)
    _q_names = list(e10_triplet['ctx_names'])
    _q_vals = []
    _per_axis_pct = {}
    for _a in _axis_names_present:
        _key = _regime_axis_key(_a)
        if _key not in _q_names:
            _q_vals.append(np.nan)
            continue
        _raw = float(_q_ctx[_q_names.index(_key)])
        _val = float(np.abs(_raw)) if _a == 'abs_ecl_beta' else _raw
        _q_vals.append(_val)
        _tr_axis = _cache['train_stack'][:, _axis_names_present.index(_a)]
        _per_axis_pct[_a] = dict(
            value=_val,
            pop_percentile=float(100.0 * (_tr_axis < _val).mean()),
        )
    _q_arr = np.asarray(_q_vals, dtype=np.float64)
    if not np.all(np.isfinite(_q_arr)):
        print()
        print('[regime audit skipped: query row has NaN on a regime axis]')
    else:
        _q_z = (_q_arr - _cache['mu']) / _cache['sd']
        if _cache['tree'] is not None:
            _d, _ = _cache['tree'].query(_q_z[None, :], k=_REGIME_K + 1)
            _q_knn_all = _d[0]
        else:
            _dsq = ((_cache['train_z'] - _q_z[None, :]) ** 2).sum(axis=1)
            _q_knn_all = np.sqrt(np.sort(_dsq))
        # A near-zero first distance means the query is a training-set twin.
        _shift = 1 if _q_knn_all[0] < 1e-6 else 0
        _q_knn = float(_q_knn_all[_REGIME_K - 1 + _shift])
        _train_p90 = _cache['train_p90']
        _density_ratio = _q_knn / _train_p90 if _train_p90 > 0 else np.inf
        sci_row_sparse_regime_flag = bool(_q_knn > _train_p90)
        sci_row_regime_audit = dict(
            axis_names=list(_axis_names_present),
            axis_values=list(_q_vals),
            per_axis=_per_axis_pct,
            knn_distance=_q_knn,
            train_p90=_train_p90,
            density_ratio=_density_ratio,
            k=_REGIME_K,
            is_sparse=sci_row_sparse_regime_flag,
        )

        print()
        print('Regime-space training-density audit '
              '(this row vs training corpus):')
        print(f"  axes = {', '.join(_axis_names_present)}   "
              f"(k={_REGIME_K} nearest, {int(_cache['train_z'].shape[0])} train rows)")
        print(f"  {'axis':<16}{'row_value':>14}{'train_pct':>14}")
        for _a in _axis_names_present:
            _info = _per_axis_pct[_a]
            _tag = ' *' if (_info['pop_percentile'] > 95.0
                            or _info['pop_percentile'] < 5.0) else ''
            print(f"  {_a:<16}{_info['value']:>14.3f}"
                  f"{_info['pop_percentile']:>13.1f}%{_tag}")
        print(f"  kNN dist (standardized)     = {_q_knn:.3f}")
        print(f"  train p{_REGIME_SPARSE_PERCENTILE:.0f} threshold          "
              f"= {_train_p90:.3f}")
        print(f"  density ratio               = {_density_ratio:.2f}x")
        if sci_row_sparse_regime_flag:
            print(f'  [SPARSE_REGIME] row is beyond the p{_REGIME_SPARSE_PERCENTILE:.0f} '
                  f'training-density envelope;')
            print('                  the model is extrapolating in this regime; '
                  'residuals expected.')
        else:
            print(f'  [ok] row is inside the p{_REGIME_SPARSE_PERCENTILE:.0f} '
                  'training-density envelope.')

# Compact flag prefix + regime density subline reused in every plot title
# below, so the row's coverage status is always visible on the figure itself.
# Science-continuum colour check for THIS row.  A contaminated row is not a
# prediction failure -- its moon target is partly field continuum absorbed by
# the Moon_bs spline -- so the title has to say so, or the moon panel reads as
# a model error.  This is the row-level view of the gate the training corpus
# and the batch/atlas samples now apply.
_colour_stats = sci_continuum_colour_excess(EVERY10_INPUT, [idx_row])
_colour_excess = float(_colour_stats['excess'][0])
_colour_contaminated = (np.isfinite(_colour_excess)
                        and _colour_excess > SCI_COLOUR_EXCESS_MAX)
print(f'  science-continuum colour excess dC = {_colour_excess:+.4f} dex '
      f'(threshold {SCI_COLOUR_EXCESS_MAX:+.3f}) -> '
      f'{"CONTAMINATED: moon target absorbs field continuum" if _colour_contaminated else "clean"}')

_flag_bits = []
if sci_row_sparse_regime_flag:
    _flag_bits.append('SPARSE_REGIME')
if _colour_contaminated:
    _flag_bits.append(f'CONTAMINATED_SCI_CONTINUUM(dC={_colour_excess:+.3f})')
_flag_prefix = f"[{' '.join(_flag_bits) if _flag_bits else 'ok'}]"
if sci_row_regime_audit:
    _axis_parts = []
    for _a in sci_row_regime_audit['axis_names']:
        _info = sci_row_regime_audit['per_axis'][_a]
        _mark = '!' if (_info['pop_percentile'] > 95.0
                        or _info['pop_percentile'] < 5.0) else ''
        _axis_parts.append(f"{_a}={_info['value']:.1f}({_info['pop_percentile']:.0f}%{_mark})")
    _pct_subline = (f"regime kNN(k={sci_row_regime_audit['k']})"
                    f"={sci_row_regime_audit['knn_distance']:.2f} "
                    f"vs train_p{_REGIME_SPARSE_PERCENTILE:.0f}="
                    f"{sci_row_regime_audit['train_p90']:.2f} "
                    f"({sci_row_regime_audit['density_ratio']:.2f}x) · "
                    + ' · '.join(_axis_parts))
else:
    _pct_subline = ''


# 5b) Frozen physical Moon + Zodi model overlay (sky_decomp/moon_zodi_model.py).
#     This is independent of both the decomposition and the ML: it predicts the
#     scattered-moonlight and zodiacal continua from exposure-midpoint ephemeris
#     geometry alone (ROLO albedo x solar SED x Rayleigh/HG scattering for the
#     moon, Leinert B500 for the zodi).  Overlaying it on the three flux panels
#     gives a physical reference for the amplitudes the QP assigned to Moon_bs /
#     Zodi_bs, and for what the ML predicted at the sci pointing.  Each arm uses
#     its own pointing and its own LSF.
import warnings as _warnings


class _MzOverlayDisabled(Exception):
    """Internal sentinel: overlay switched off, not a failure."""


_mz_overlay = {}
if not SHOW_MOON_ZODI_MODEL:
    print("  moon/zodi physical model overlay: disabled "
          "(SHOW_MOON_ZODI_MODEL = False)")
try:
    if not SHOW_MOON_ZODI_MODEL:
        raise _MzOverlayDisabled
    from sky_decomp.moon_zodi_model import (
        MoonZodiInvalidObservationError,
        MoonZodiObservation,
        MoonZodiPhysicalModel,
    )

    with fits.open(EVERY10_INPUT) as _hdul_mz:
        _mz_meta = _hdul_mz["META"].data[idx_row]
        _mz_meta_names = set(_hdul_mz["META"].columns.names or ())
        _mz_lsf = {}
        for _arm, _ext in (("near", "LSF_SKY_NEAR"),
                           ("far", "LSF_SKY_FAR"),
                           ("sci", "LSF_SCI")):
            _a = (np.asarray(_hdul_mz[_ext].data, dtype=np.float64)
                  if _ext in _hdul_mz else lsf_sci_arr)
            _mz_lsf[_arm] = np.asarray(_a if _a.ndim == 1 else _a[idx_row],
                                       dtype=np.float64)

    # Exposure length: prefer a metadata column, else the pipeline's 900 s
    # default (decompose_parallel._WORKER_EXPOSURE_SECONDS).
    _mz_exp, _mz_exp_src = 900.0, "assumed_900s"
    for _c in ("exposure_seconds", "exptime"):
        if _c in _mz_meta_names:
            _v = float(_mz_meta[_c])
            if np.isfinite(_v) and _v > 0.0:
                _mz_exp, _mz_exp_src = _v, "metadata"
                break

    _mz_date_obs = (_mz_meta["date_obs"].decode().strip()
                    if isinstance(_mz_meta["date_obs"], bytes)
                    else str(_mz_meta["date_obs"]).strip())
    _mz_model = MoonZodiPhysicalModel()
    _mz_roles = {
        "near": ("sky_near", "sky_near_ra", "sky_near_dec"),
        "far":  ("sky_far",  "sky_far_ra",  "sky_far_dec"),
        "sci":  ("sci",      "sci_ra",      "sci_dec"),
    }
    for _arm, (_role, _rac, _decc) in _mz_roles.items():
        _obs = MoonZodiObservation(
            expnum=int(_mz_meta["expnum"]),
            date_obs=_mz_date_obs,
            role=_role,
            target_ra_deg=float(_mz_meta[_rac]),
            target_dec_deg=float(_mz_meta[_decc]),
            exposure_seconds=_mz_exp,
            exposure_seconds_source=_mz_exp_src,
        )
        try:
            # astropy warns about IERS coverage for these epochs; the model
            # deliberately pins the packaged table (compute_midpoint_geometry
            # sets iers.conf.auto_download=False), so the warning is expected.
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _pr = _mz_model.predict(wave_row, _mz_lsf[_arm], _obs,
                                        physical_to_fit_flux_scale=FACTOR)
            # predict() returns fit-flux units (scaled by FACTOR); divide back to
            # physical so these match every other trace in this cell, which is
            # stored physical and multiplied by FACTOR at plot time.
            _mz_overlay[_arm] = {
                "moon": np.asarray(_pr.moon, dtype=np.float64) / FACTOR,
                "zodi": np.asarray(_pr.zodi, dtype=np.float64) / FACTOR,
                "state": _pr.state,
            }
        except MoonZodiInvalidObservationError as _exc:
            print(f"  moon/zodi model: {_arm} arm not modellable ({_exc.reason})")
except _MzOverlayDisabled:
    pass
except Exception as _exc:  # missing data bundle, ephemeris, META columns, ...
    print(f"  moon/zodi model overlay unavailable: "
          f"{type(_exc).__name__}: {_exc}")

if _mz_overlay:
    # Compare the physical model against the amplitudes the QP actually fitted
    # (and, for sci, against what the ML predicted).  Ratios are band-integrated
    # so they are insensitive to per-pixel noise.
    _mz_decomp = {
        "near": comps_near_from_near,
        "far": comps_far_from_far,
        "sci": comps_sci_true,
    }
    _mz_rows = []
    for _arm, _ov in _mz_overlay.items():
        _geo = _ov["state"].geometry
        _row = {
            "arm": _arm,
            "moon_alt": _geo.moon_altitude_deg,
            "moon_sep": _geo.moon_separation_deg,
            "phase": _geo.signed_phase_deg,
            "zodi_b500": _geo.zodi_b500,
        }
        _mod_sum = _fit_sum = 0.0
        for _fam in ("moon", "zodi"):
            _mod = float(np.nansum(_ov[_fam]))
            _fit = float(np.nansum(np.asarray(
                _mz_decomp[_arm].get(_fam, 0.0), dtype=np.float64) / FACTOR))
            _row[f"{_fam}_fit/model"] = (_fit / _mod if abs(_mod) > 0 else np.nan)
            _mod_sum += _mod
            _fit_sum += _fit
        # moon and zodi are both reddened solar continua, so the QP can trade
        # amplitude between them almost freely (§1.2.2).  The combined ratio is
        # the identifiable quantity: if it sits near 1 while the two individual
        # ratios are far off, the disagreement is a *split* problem, not an
        # amplitude problem -- and only the split is degenerate.
        _row["(moon+zodi)_fit/model"] = (_fit_sum / _mod_sum
                                         if abs(_mod_sum) > 0 else np.nan)
        _mz_rows.append(_row)
    # Same ratio for the ML prediction at the sci pointing.
    if "sci" in _mz_overlay:
        _row = {"arm": "sci (ML pred)", "moon_alt": np.nan, "moon_sep": np.nan,
                "phase": np.nan, "zodi_b500": np.nan}
        _mod_sum = _pred_sum = 0.0
        for _fam in ("moon", "zodi"):
            _mod = float(np.nansum(_mz_overlay["sci"][_fam]))
            _pred = float(np.nansum(np.asarray(
                comps_sci.get(_fam, 0.0), dtype=np.float64) / FACTOR))
            _row[f"{_fam}_fit/model"] = (_pred / _mod if abs(_mod) > 0 else np.nan)
            _mod_sum += _mod
            _pred_sum += _pred
        _row["(moon+zodi)_fit/model"] = (_pred_sum / _mod_sum
                                         if abs(_mod_sum) > 0 else np.nan)
        _mz_rows.append(_row)
    _mz_state0 = next(iter(_mz_overlay.values()))["state"]
    print(f"  moon/zodi physical model {_mz_state0.model_id} "
          f"({_mz_state0.formula_version}), exposure {_mz_exp:.0f}s "
          f"[{_mz_exp_src}], flags={_mz_state0.flags}")
    print(f"    scientific_status = {_mz_state0.scientific_status!r}")
    print(f"    correction_scope  = {_mz_state0.correction_scope!r}")
    print(pd.DataFrame(_mz_rows).to_string(
        index=False, float_format=lambda v: f'{v:.3g}', na_rep='-'))
    # Read the ratios with the model's own scope in mind.  The fitted
    # correction is applied to the moon+zodi SUM (CORRECTION_SCOPE =
    # 'moon_plus_zodi'), so the combined column is the only one the model is
    # calibrated to reproduce.  The individual moon and zodi columns compare
    # against vectors the fit never constrained separately, so a large split
    # discrepancy there is NOT evidence that the QP mis-assigned the families
    # -- the two are degenerate in the model exactly as they are in the QP.
    print("    (moon+zodi)_fit/model is the calibrated comparison: ~1 means the "
          "total continuum")
    print("    amplitude agrees with the frozen physical prediction.  The "
          "per-family columns are")
    print("    indicative only -- the model's correction scope is the sum, so it "
          "does not claim to")
    print("    split moon from zodi any better than the decomposition does.")
    print("    NB scientific_status marks this model diagnostic-only; use it as "
          "a sanity reference,")
    print("    not as truth.")
    print()


# 6) Diagnostic plot:
#    row1: near observed vs reconstruction from near coefficients
#    row2: far observed vs reconstruction from far coefficients
#    row3: science true vs science prediction
#    row4: science sky-subtracted spectrum, observed - pred
#    row5: per-pixel photon chi2, prediction against the decomposition's floor
#          (omitted when the noise model could not be loaded)
_SHOW_CHI2_PANEL = chi2_pix_pred is not None
_CHI2_ROW = 5
_panel_titles = [
    "Near: observed vs reconstructed from near coefficients",
    "Far: observed vs reconstructed from far coefficients",
    "Science: observed / recon(sci coef) / recon(pred)",
    "Science sky-subtracted: observed - pred"
    + (" - corrections (" + " + ".join(
        n for n, on in (("sky-arm residual", RESIDUAL_CORRECTION),
                        ("line scaling", LINE_SCALING)) if on) + ")"
       if _ANY_CORR else ""),
]
_panel_heights = [0.22, 0.22, 0.30, 0.13]
if _SHOW_CHI2_PANEL:
    _panel_titles.append(
        f"Photon chi2 per pixel ({'single fibre' if CHI2_SINGLE_FIBRE else 'stacked'})"
        f" -- recon(pred) {chi2_row_pred:.3g} vs decomposition floor {chi2_row_self:.3g}")
    _panel_heights.append(0.13)
fig = make_subplots(
    rows=len(_panel_titles),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=tuple(_panel_titles),
    row_heights=_panel_heights,
)

# Physical Moon/Zodi model overlays, one pair per flux panel.  Dotted so they
# read as an external reference rather than as data or reconstruction, and
# added BEFORE each panel's data traces so plotly draws them underneath: the
# data and reconstructions are what we are reading off these panels, and the
# model curves are thick enough to hide a reconstruction that lands on top of
# them.
def _add_mz_traces(_arm, _row):
    _ov = _mz_overlay.get(_arm)
    if _ov is None:
        return
    _mz_curves = (
        ("moon", _ov["moon"], "#9467bd", "dot", 1.2),
        ("zodi", _ov["zodi"], "#17becf", "dot", 1.2),
        # The model's correction scope is moon_plus_zodi, so the sum is the only
        # calibrated curve here -- drawn heavier than its two parts.
        ("moon+zodi", _ov["moon"] + _ov["zodi"], "#8c564b", "dashdot", 1.7),
    )
    for _fam, _y, _color, _dash, _w in _mz_curves:
        fig.add_trace(
            go.Scattergl(
                x=wave_row,
                y=_y * FACTOR,
                mode="lines",
                name=f"{_fam} physical model",
                legendgroup=f"mz_{_fam}",
                showlegend=(_row == 1),
                line=dict(color=_color, width=_w, dash=_dash),
            ),
            row=_row,
            col=1,
        )


_add_mz_traces("near", 1)

fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_near_row * FACTOR,
        mode="lines",
        name="near true",
        line=dict(color="#7f7f7f", width=1.0),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_near_recon_row * FACTOR,
        mode="lines",
        name="near recon(from near coef)",
        line=dict(color="#e41a1c", width=1.4),
    ),
    row=1,
    col=1,
)
_add_mz_traces("far", 2)

fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_far_row * FACTOR,
        mode="lines",
        name="far true",
        line=dict(color="#7f7f7f", width=1.0),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_far_recon_row * FACTOR,
        mode="lines",
        name="far recon(from far coef)",
        line=dict(color="#ff7f00", width=1.4),
    ),
    row=2,
    col=1,
)
_add_mz_traces("sci", 3)

fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_sci_true_row * FACTOR,
        mode="lines",
        name="science observed",
        line=dict(color="#7f7f7f", width=1.0),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_sci_true_recon_row * FACTOR,
        mode="lines",
        name="science recon(from sci coef)",
        line=dict(color="#2ca02c", width=1.2, dash="dash"),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=flux_sci_pred_row * FACTOR,
        mode="lines",
        name="science recon(pred)",
        line=dict(color="#1f78b4", width=1.4),
    ),
    row=3,
    col=1,
)

if _ANY_CORR:
    fig.add_trace(
        go.Scattergl(
            x=wave_row,
            y=obs_minus_pred_uncorr_row * FACTOR,
            mode="lines",
            name="science sky-subtracted (obs - pred), uncorrected",
            line=dict(color="#9e9e9e", width=1.0),
        ),
        row=4,
        col=1,
    )
fig.add_trace(
    go.Scattergl(
        x=wave_row,
        y=obs_minus_pred_row * FACTOR,
        mode="lines",
        name=("science sky-subtracted (obs - pred), corrected"
              if _ANY_CORR else "science sky-subtracted (obs - pred)"),
        line=dict(color="#d62728", width=1.0),
    ),
    row=4,
    col=1,
)
fig.add_hline(y=0, line=dict(color="black", width=0.8, dash="dash"), row=4, col=1)

if _SHOW_CHI2_PANEL:
    # The decomposition's own fit goes on FIRST so it draws underneath: it is
    # the reference the prediction is read against, not a second result.
    fig.add_trace(
        go.Scattergl(
            x=wave_row,
            y=chi2_pix_self,
            mode="lines",
            name="chi2 recon(sci coef) -- decomposition floor",
            line=dict(color="#7f7f7f", width=0.9),
        ),
        row=_CHI2_ROW,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=wave_row,
            y=chi2_pix_pred,
            mode="lines",
            name="chi2 recon(pred)",
            line=dict(color="#d62728", width=1.0),
        ),
        row=_CHI2_ROW,
        col=1,
    )
    # chi2 = 1 is "residual consistent with the photon noise" on this scale.
    fig.add_hline(y=1.0, line=dict(color="black", width=0.8, dash="dash"),
                  row=_CHI2_ROW, col=1)

fig.update_yaxes(type="log", title_text="Near flux", row=1, col=1)
fig.update_yaxes(type="log", title_text="Far flux", row=2, col=1)
fig.update_yaxes(type="log", title_text="Science flux", row=3, col=1)
fig.update_yaxes(type="linear", title_text=("obs - pred (corrected)" if _ANY_CORR else "obs - pred"), row=4, col=1)
if _SHOW_CHI2_PANEL:
    # Linear (2026-09-25): with the science lines masked the per-pixel chi2 no
    # longer spans the several decades that motivated a log axis.
    fig.update_yaxes(type="linear", title_text="chi2 / pixel", row=_CHI2_ROW, col=1)
# Shade the science-line windows the chi2 excludes, so a gap reads as masked.
if _SHOW_CHI2_PANEL and globals().get("_sci_mask_row") is not None:
    _m = np.asarray(_sci_mask_row, dtype=bool)
    _edges = np.flatnonzero(np.diff(np.concatenate([[0], _m.astype(int), [0]])))
    for _a, _b in zip(_edges[0::2], _edges[1::2]):
        for _r in (4, _CHI2_ROW):
            fig.add_vrect(x0=float(wave_row[_a]), x1=float(wave_row[_b - 1]),
                          fillcolor="rgba(150,150,150,0.18)", line_width=0,
                          layer="below", row=_r, col=1)
fig.update_xaxes(title_text="Wavelength [A]", row=len(_panel_titles), col=1)

if _SHOW_CHI2_PANEL:
    _ratio_txt = (f"{chi2_row_pred / chi2_row_self:.2f}x"
                  if np.isfinite(chi2_row_self) and chi2_row_self > 0 else "n/a")
    _chi2_subline = (
        f"photon chi2/pix full (blue&lt;{CHI2_BLUE_MAX_A:.0f}A): "
        f"recon(pred) {chi2_row_pred:.3g} ({chi2_blue_pred:.3g})  ·  "
        f"decomposition floor {chi2_row_self:.3g} ({chi2_blue_self:.3g})  ·  "
        f"pred/floor = {_ratio_txt}"
        + (f"  ·  corrected {chi2_row_corr:.3g} ({chi2_blue_corr:.3g})"
           if _ANY_CORR else "")
        + "<br>")
else:
    _chi2_subline = ""

fig.update_layout(
    template="plotly_white",
    title=dict(
        text=(
            f"{_flag_prefix} Every10 {ROW_LABEL}<br>"
            f"<sub>pRMSE near / far / sci = {rmse_near_recon:.3g} / "
            f"{rmse_far_recon:.3g} / {rmse_row:.3g}  ·  pWRMSE = "
            f"{wrmse_near_recon:.3g} / {wrmse_far_recon:.3g} / "
            f"{wrmse_row_pix:.3g}  ·  sci display pRMSE = "
            f"{rmse_row_display:.3g}</sub><br>"
            f"<sub>{_chi2_subline}</sub>"
            f"<sub>{_pct_subline}</sub>"
        ),
        font=dict(size=13),
        x=0.02, xanchor='left',
        y=0.995, yanchor='top',
    ),
    height=1220 + (170 if _SHOW_CHI2_PANEL else 0),
    margin=dict(t=110, r=20, l=70, b=90),
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.05,
        xanchor="left", x=0.0,
        font=dict(size=10),
    ),
)
fig.show()

# 7) Moon spline coefficient diagnostic figure (global-prior result).
moon_axis = np.arange(moon_idx.size)
fig_moon = go.Figure()
fig_moon.add_trace(
    go.Scatter(
        x=moon_axis,
        y=moon_near,
        mode="lines+markers",
        name="near",
        line=dict(color="#7f7f7f"),
    )
)
fig_moon.add_trace(
    go.Scatter(
        x=moon_axis,
        y=moon_far,
        mode="lines+markers",
        name="far",
        line=dict(color="#bdbdbd"),
    )
)
fig_moon.add_trace(
    go.Scatter(
        x=moon_axis,
        y=moon_true,
        mode="lines+markers",
        name="sci true",
        line=dict(color="#1f78b4"),
    )
)
fig_moon.add_trace(
    go.Scatter(
        x=moon_axis,
        y=moon_pred,
        mode="lines+markers",
        name="pred default",
        line=dict(color="#e41a1c"),
    )
)
fig_moon.update_layout(
    template="plotly_white",
    title=dict(
        text=(f"{_flag_prefix} Moon spline coefficients — {ROW_LABEL}<br>"
              f"<sub>{_pct_subline}</sub>"),
        font=dict(size=12),
        x=0.02, xanchor='left',
    ),
    xaxis_title="Moon_bs coefficient index",
    yaxis_title="coefficient value",
    height=440,
    margin=dict(t=80),
    legend=dict(font=dict(size=10)),
)
fig_moon.show()

# 7b) Zodi spline coefficient diagnostic figure (split_zodi=True corpus).
if zodi_idx.size:
    zodi_axis = np.arange(zodi_idx.size)
    fig_zodi = go.Figure()
    for _y, _name, _color in (
        (zodi_near, 'near',        '#7f7f7f'),
        (zodi_far,  'far',         '#bdbdbd'),
        (zodi_true, 'sci true',    '#1f78b4'),
        (zodi_pred, 'pred default','#e41a1c'),
    ):
        fig_zodi.add_trace(go.Scatter(
            x=zodi_axis, y=_y, mode='lines+markers',
            name=_name, line=dict(color=_color),
        ))
    fig_zodi.update_layout(
        template='plotly_white',
        title=dict(
            text=(f"{_flag_prefix} Zodi spline coefficients — {ROW_LABEL}<br>"
                  f"<sub>{_pct_subline}</sub>"),
            font=dict(size=12),
            x=0.02, xanchor='left',
        ),
        xaxis_title='coefficient index',
        yaxis_title='coefficient value',
        height=380,
        margin=dict(t=80),
        legend=dict(font=dict(size=10)),
    )
    fig_zodi.show()

# 8) Per-component reconstructions for all four arms, mirroring the
#    four-trace layout of the Moon spline diagnostic above but as
#    spectra over wavelength rather than coefficients over index.
#    comps_sci_true was reconstructed in section 4 so row 3 can use it.
_comps_by_arm = {
    "near": comps_near_from_near,
    "far": comps_far_from_far,
    "sci true": comps_sci_true,
    "pred default": comps_sci,
}
_arm_colors = {
    "near": "#7f7f7f",
    "far": "#bdbdbd",
    "sci true": "#1f78b4",
    "pred default": "#e41a1c",
}

# comps["*"] is in the same display scale as comps["total"], so no *FACTOR here.
def _non_moon_continuum(comps):
    return np.asarray(comps["diffuse"], dtype=np.float64)

def _line_component(comps):
    return (np.asarray(comps["oh"], dtype=np.float64)
            + np.asarray(comps["atom"], dtype=np.float64)
            + np.asarray(comps["orc"], dtype=np.float64)
            + np.asarray(comps["o2"], dtype=np.float64))

def _moon_spectrum(comps):
    return np.asarray(comps["moon"], dtype=np.float64)

def _zodi_spectrum(comps):
    if "zodi" not in comps:
        return np.zeros_like(comps["moon"])
    return np.asarray(comps["zodi"], dtype=np.float64)

# Sanity: total is defined by reconstruct_component_spectra as
#   oh + moon + diffuse + atom + orc + o2
# so lines + moon_spectrum + non_moon_continuum must equal it.
_total_pred = np.asarray(comps_sci["total"], dtype=np.float64)
_sum_pred = (_line_component(comps_sci)
             + _moon_spectrum(comps_sci)
             + _non_moon_continuum(comps_sci))
_max_diff = float(np.nanmax(np.abs(_total_pred - _sum_pred)))
_max_rel = float(np.nanmax(np.abs(_total_pred - _sum_pred)
                          / np.clip(np.abs(_total_pred), 1e-30, None)))
print(f"Component-sum check (pred): max abs diff = {_max_diff:.3g}, "
      f"max rel diff = {_max_rel:.3g}")

fig_continuum = go.Figure()
for _arm, _comps in _comps_by_arm.items():
    fig_continuum.add_trace(
        go.Scattergl(
            x=wave_row,
            y=_non_moon_continuum(_comps),
            mode="lines",
            name=_arm,
            line=dict(color=_arm_colors[_arm], width=1.2),
        )
    )
fig_continuum.update_layout(
    template="plotly_white",
    title=dict(
        text=(f"{_flag_prefix} Reconstructed non-moon continuum "
              f"(diffuse = HO2 + FeO + O2ac) — {ROW_LABEL}"),
        font=dict(size=12),
        x=0.02, xanchor='left',
    ),
    xaxis_title="Wavelength [A]",
    yaxis_title=f"Flux (display units x{FACTOR:.3g})",
    height=440,
    margin=dict(t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
)
fig_continuum.show()

# Physical-model overlay for the single-family spline-spectrum panels below.
# One trace per physical pointing: "sci true" and "pred default" share the sci
# pointing, so the model contributes three curves, not four.
def _add_mz_family_traces(_target_fig, _fam):
    for _arm, _color in (("near", "#7f7f7f"), ("far", "#bdbdbd"), ("sci", "#1f78b4")):
        _ov = _mz_overlay.get(_arm)
        if _ov is None:
            continue
        _target_fig.add_trace(
            go.Scattergl(
                x=wave_row,
                # comps[...] in these panels is plotted in fit units without a
                # *FACTOR, while _mz_overlay is stored physical -- hence *FACTOR.
                y=_ov[_fam] * FACTOR,
                mode="lines",
                name=f"{_arm} physical model",
                line=dict(color=_color, width=1.6, dash="dot"),
            )
        )


fig_moon_spectrum = go.Figure()
_add_mz_family_traces(fig_moon_spectrum, "moon")
for _arm, _comps in _comps_by_arm.items():
    fig_moon_spectrum.add_trace(
        go.Scattergl(
            x=wave_row,
            y=_moon_spectrum(_comps),
            mode="lines",
            name=_arm,
            line=dict(color=_arm_colors[_arm], width=1.2),
        )
    )
fig_moon_spectrum.update_layout(
    template="plotly_white",
    title=dict(
        text=(f"{_flag_prefix} Reconstructed moon spline spectrum "
              f"(comps['moon']) — {ROW_LABEL}"),
        font=dict(size=12),
        x=0.02, xanchor='left',
    ),
    xaxis_title="Wavelength [A]",
    yaxis_title=f"Flux (display units x{FACTOR:.3g})",
    height=440,
    margin=dict(t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
)
fig_moon_spectrum.show()

# Reconstructed zodi spline spectrum (split_zodi Zodi_bs family).
if 'zodi' in comps_sci:
    fig_zodi_spectrum = go.Figure()
    _add_mz_family_traces(fig_zodi_spectrum, 'zodi')
    for _arm, _comps in _comps_by_arm.items():
        # Colour explicitly (as the moon panel does): the model traces now come
        # first, so leaving these to the default colorway would recolour them.
        fig_zodi_spectrum.add_trace(
            go.Scattergl(
                x=wave_row,
                y=_zodi_spectrum(_comps),
                mode='lines',
                name=_arm,
                line=dict(color=_arm_colors[_arm], width=1.4),
            )
        )
    fig_zodi_spectrum.update_layout(
        template='plotly_white',
        title=dict(
            text=(f"{_flag_prefix} Reconstructed zodi spline spectrum "
                  f"(comps['zodi']) — {ROW_LABEL}"),
            font=dict(size=12),
            x=0.02, xanchor='left',
        ),
        xaxis_title='Wavelength (A)',
        yaxis_title='flux (fit units)',
        height=400,
        margin=dict(t=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0, font=dict(size=10)),
    )
    fig_zodi_spectrum.show()

fig_lines = go.Figure()
for _arm, _comps in _comps_by_arm.items():
    fig_lines.add_trace(
        go.Scattergl(
            x=wave_row,
            y=_line_component(_comps),
            mode="lines",
            name=_arm,
            line=dict(color=_arm_colors[_arm], width=1.2),
        )
    )
fig_lines.update_layout(
    template="plotly_white",
    title=dict(
        text=(f"{_flag_prefix} Reconstructed line emission "
              f"(OH + atom + ORC + O2) — {ROW_LABEL}"),
        font=dict(size=12),
        x=0.02, xanchor='left',
    ),
    xaxis_title="Wavelength [A]",
    yaxis_title=f"Flux (display units x{FACTOR:.3g})",
    height=480,
    margin=dict(t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
)
fig_lines.show()

# 9) Per-component (pred - sci_true) residual spectrum. Row 3 shows only
#    the total residual; this decomposes it into moon / diffuse / lines
#    so a small broadband deficit in a component that spans the whole
#    wavelength range is visible even when the component panels above
#    make it look "close" on a linear-y axis. By construction the three
#    traces must sum to the blue-minus-green curve of row 3, and that
#    sum is drawn as a black dashed reference.
_delta_moon = _moon_spectrum(comps_sci) - _moon_spectrum(comps_sci_true)
_delta_zodi = _zodi_spectrum(comps_sci) - _zodi_spectrum(comps_sci_true)
_delta_diffuse = _non_moon_continuum(comps_sci) - _non_moon_continuum(comps_sci_true)
_delta_lines = _line_component(comps_sci) - _line_component(comps_sci_true)
_delta_total = _delta_moon + _delta_zodi + _delta_diffuse + _delta_lines

fig_deltas = go.Figure()
for _label, _y, _color in (
    ("moon (pred - sci recon)", _delta_moon, "#e41a1c"),
    ("diffuse (pred - sci recon)", _delta_diffuse, "#377eb8"),
    ("lines (pred - sci recon)", _delta_lines, "#4daf4a"),
    ("total (pred - sci recon)", _delta_total, "#000000"),
):
    fig_deltas.add_trace(
        go.Scattergl(
            x=wave_row,
            y=_y,
            mode="lines",
            name=_label,
            line=dict(color=_color, width=1.2,
                      dash="dash" if _label.startswith("total") else "solid"),
        )
    )
fig_deltas.add_hline(y=0, line=dict(color="rgba(0,0,0,0.4)", width=0.8, dash="dot"))
fig_deltas.update_layout(
    template="plotly_white",
    title=dict(
        text=(f"{_flag_prefix} Per-component prediction minus sci-arm "
              f"reconstruction (linear) — {ROW_LABEL}"),
        font=dict(size=12),
        x=0.02, xanchor='left',
    ),
    xaxis_title="Wavelength [A]",
    yaxis_title=f"Delta flux (display units x{FACTOR:.3g})",
    height=480,
    margin=dict(t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
)
fig_deltas.show()

# 10) Numeric integrated deltas per component in three wavelength bands, so
#     the sign and magnitude of each contribution to the red deficit is
#     visible even where the plot traces are noisy line-by-line.
_bands = [
    ("blue  (< 5500 A)", wave_row < 5500.0),
    ("green (5500-7500)", (wave_row >= 5500.0) & (wave_row < 7500.0)),
    ("red   (>= 7500 A)", wave_row >= 7500.0),
]
print()
print("Integrated (pred - sci recon) per component and wavelength band:")
print(f"  {'band':<18s} {'moon':>12s} {'zodi':>12s} {'diffuse':>12s} {'lines':>12s} {'total':>12s}")
for _bname, _mask in _bands:
    if not _mask.any():
        continue
    _sm = float(np.nansum(_delta_moon[_mask]))
    _sd = float(np.nansum(_delta_diffuse[_mask]))
    _sl = float(np.nansum(_delta_lines[_mask]))
    _st = float(np.nansum(_delta_total[_mask]))
    print(f"  {_bname:<18s} {_sm:>+12.4g} {_sd:>+12.4g} {_sl:>+12.4g} {_st:>+12.4g}")
