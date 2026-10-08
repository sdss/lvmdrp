# Wavelength-resolved residual atlas (physics-space failure map).
#
# Sample n_sample test rows, reconstruct both the true and the ML-predicted
# sci-arm spectra, and aggregate the flux-space residual per pixel over the
# sample.  The atlas shows *where in wavelength* the model fails: a coherent
# blue tilt = zodi mis-calibration; sharp bumps at OH band centres = OH group
# mis-scaling; ROLO mineral bands = moon mis-projection.
#
# What to look for:
#   * flat, near-zero median residual with narrow 68% band -> healthy across
#     the whole range.
#   * coherent blue lift -> zodi head is under-predicting; check the
#     naive_baseline() close_zodi regime.
#   * OH band spikes -> the OH head is biased at specific transitions;
#     probably worth adding a per-group loss floor.
#   * moon band residuals (700-1000 nm mineral features) -> moon head is
#     picking the wrong knot amplitudes.
#
# Data: by default the per-component reconstructions full_spectrum_batch_rmse
# hands over (batch_recon_for_atlas) -- every held-out row it scored, so the
# two cells describe the same sample.  Only without that handoff does this cell
# fall back to drawing and reconstructing its own every10 sample.
import time as _time
from pathlib import Path
from sky_decomp.moon_zodi_model import LSF_FWHM_TO_SIGMA

required = ['filtered_triplet', 'mlp_artifacts', 'predict_sci_coefficients_default',
            'context_cols', 'build_triplet_coef_dataset',
            'reconstruct_component_spectra', 'reconstruct_with_lsf',
            'load_lsf_state_if_available', 'load_o2_vector_if_available',
            '_infer_base_dir_for_reconstruction']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Missing kernel state: ' + ', '.join(_missing))

# Raised from 200 to 500 on 2026-09-10, when the figure gained the moon-up /
# moon-down split: each panel now sees only half the sample, so 200 left ~100
# rows per regime behind every percentile band.  Cost is linear in both time
# (two LSF reconstructions per row) and memory (~0.7 MB per row across the
# residual and per-component stacks; the figure itself is aggregate lines
# only, ~7 MB regardless of n).  Override with
# `diag.wavelength_residual_atlas(size=..., seed=...)`; edit there, not here.
n_sample_atlas = 500
rng_seed_atlas = 42

_e10_stem = f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_every10'
_e10_suffix = _DECOMP_SUFFIX
EVERY10_INPUT = f'{_e10_stem}.fits'
EVERY10_NEAR  = f'{_e10_stem}_decomp_sky1{_e10_suffix}.fits'
EVERY10_FAR   = f'{_e10_stem}_decomp_sky2{_e10_suffix}.fits'
EVERY10_SCI   = f'{_e10_stem}_decomp_sci{_e10_suffix}.fits'

# Build aligned e10 triplet (matches full_spectrum_batch_rmse).
_ATLAS_COMPONENTS = {
    'moon':                 ('moon',),
    'zodi':                 ('zodi',),
    'mesospheric (OH+O2)':  ('oh', 'o2'),
    'continuum (diffuse)':  ('diffuse',),
    'atomic':               ('atom',),
    'ionospheric (ORC)':    ('orc',),
}


def _group_components(_comps):
    """Sum the raw recon components into the six ML groups."""
    _out = {}
    for _gname, _keys in _ATLAS_COMPONENTS.items():
        _acc = 0.0
        for _k in _keys:
            _acc = _acc + np.asarray(_comps.get(_k, 0.0), dtype=np.float64)
        _out[_gname] = _acc
    return _out



# --- Reuse the batch-RMSE reconstructions when they are available ----------
# `full_spectrum_batch_rmse` reconstructs the same rows with the same basis,
# LSF state, O2 vector and telluric, and since 2026-09-23 it hands the
# per-component result over in this cell's own layout and units.  Reusing it
# is not only cheaper -- it is the only way the two cells describe the SAME
# rows.  Before, this cell drew from the ungated every10 pool while the batch
# cell scored the held-out split, so their residuals were never comparable.
#
# Set batch_recon_for_atlas = None, or ATLAS_HANDOFF_ROWS = 0 in the batch
# cell, to force this cell to reconstruct its own sample again.
_handoff = globals().get('batch_recon_for_atlas')
_USE_HANDOFF = bool(_handoff) and _handoff.get('sel_rows') is not None
if _USE_HANDOFF:
    if tuple(_handoff['components']) != tuple(_ATLAS_COMPONENTS):
        raise RuntimeError(
            "batch_recon_for_atlas was built with components "
            f"{tuple(_handoff['components'])} but this cell expects "
            f"{tuple(_ATLAS_COMPONENTS)}; the two definitions have drifted.")
    # The batch cell selects out of filtered_triplet, so its positions index
    # that table -- `e10_triplet` below is a name, not a claim about every10.
    e10_triplet = filtered_triplet
    sel_pos = np.asarray(_handoff['sel_pos'], dtype=int)
    sel_rows = np.asarray(_handoff['sel_rows'], dtype=int)
    _n_pick = int(sel_rows.size)
    _wave_ref_recon = np.asarray(_handoff['wave'], dtype=np.float64)
    wave_arr = _wave_ref_recon
    _resid = _handoff['resid']
    _truth = _handoff['truth']
    _resid_comp = dict(_handoff['resid_comp'])
    _truth_comp = dict(_handoff['truth_comp'])
    _chi2_max_arm = None
    _k_chi2 = ('chi2_near', 'chi2_far', 'chi2_sci')
    if all(_k in filtered_triplet for _k in _k_chi2):
        _chi2_max_arm = np.nanmax(np.vstack(
            [np.asarray(filtered_triplet[_k], dtype=np.float64)
             for _k in _k_chi2]), axis=0)
    _atlas_source = (f"{_handoff['split']} split, batch-RMSE "
                     f"reconstructions")
    print(f"  [atlas] reusing the batch-RMSE reconstructions: {_n_pick} rows "
          f"of the '{_handoff['split']}' split, no re-reconstruction")
else:
    _atlas_source = 'every10 sample, own reconstructions'
    print('  [atlas] no batch-RMSE handoff (run full_spectrum_batch_rmse first '
          'for the held-out sample); reconstructing an every10 sample instead')
    e10_triplet = build_triplet_coef_dataset(
        input_fits_path=EVERY10_INPUT,
        sky_near_decomp_fits_path=EVERY10_NEAR,
        sky_far_decomp_fits_path=EVERY10_FAR,
        sci_decomp_fits_path=EVERY10_SCI,
        context_columns=context_cols,
        return_chi2=True,
    )
    if '_augment_triplet_with_ecliptic' in globals():
        _augment_triplet_with_ecliptic(e10_triplet, meta_fits_path=EVERY10_INPUT)
    if '_augment_triplet_with_physics_priors' in globals():
        _augment_triplet_with_physics_priors(e10_triplet)
        # MOON-MODEL-CTX-V1: must match the training-time ctx layout, or
        # ctx_names will not line up with the trained ensemble.
        if (globals().get('USE_MOON_MODEL_FEATURE', False)
                and '_augment_triplet_with_moon_model' in globals()):
            _augment_triplet_with_moon_model(e10_triplet, _e10_stem)

    _e10_n0 = int(e10_triplet['n_rows'])
    if _e10_n0 == 0:
        raise RuntimeError('No aligned rows in e10_triplet')

    # Apply LMC/SMC field exclusion (do NOT apply the kappa filters -- the atlas
    # wants a realistic sample, tail included).
    _keep = np.ones(_e10_n0, dtype=bool)
    if 'sci_ra' in e10_triplet and 'sci_dec' in e10_triplet:
        _sci_ra = np.asarray(e10_triplet['sci_ra'], dtype=np.float64)
        _sci_dec = np.asarray(e10_triplet['sci_dec'], dtype=np.float64)
        for _region in (LMC_EXCLUSION, SMC_EXCLUSION):
            _sep = _angular_separation_deg_vec(
                _sci_ra, _sci_dec, _region['ra_deg'], _region['dec_deg'])
            _keep &= ~(np.isfinite(_sep) & (_sep <= float(_region['radius_deg'])))

    _avail = np.where(_keep)[0]
    _rng = np.random.default_rng(rng_seed_atlas)
    _n_pick = int(min(n_sample_atlas, _avail.size))
    sel_pos = _rng.choice(_avail, size=_n_pick, replace=False)
    print(f'  atlas sample: n = {_n_pick} of {_avail.size} available every10 rows')

    # Drop rows whose DECOMPOSITION failed, using the same hard cap the training
    # corpus applies (data.apply_triplet_filters chi2_max).  This is a correctness
    # gate, not the tail-trimming the header warns against: the e10 corpus contains
    # rows with reduced_chi2 of order 1e30 (row 89 on the new-oh corpus), whose
    # "truth" spectrum is a broken QP solve.  Scoring the model against those is
    # meaningless, and because the sample is drawn with a fixed seed such a row sits
    # in every run, dominating any mean-based statistic.  The kappa/percentile
    # filters are still deliberately NOT applied.
    #
    # The gate is applied AFTER the draw, not before: `_rng.choice` indexes into
    # `_avail`, so shrinking the pool would resample every row and break
    # comparability with earlier runs.  Filtering the drawn sample instead keeps the
    # survivors a strict subset of the rows previous runs used.
    ATLAS_CHI2_MAX = 10.0
    _chi2_keys = ('chi2_near', 'chi2_far', 'chi2_sci')
    if all(_k in e10_triplet for _k in _chi2_keys):
        _chi2_max_arm = np.nanmax(np.vstack(
            [np.asarray(e10_triplet[_k], dtype=np.float64) for _k in _chi2_keys]), axis=0)
        _chi2_ok_sel = (np.isfinite(_chi2_max_arm[sel_pos])
                        & (_chi2_max_arm[sel_pos] <= ATLAS_CHI2_MAX))
        _n_bad = int((~_chi2_ok_sel).sum())
        if _n_bad:
            print(f'  atlas chi2 gate (max-arm reduced_chi2 <= {ATLAS_CHI2_MAX:g}): '
                  f'dropped {_n_bad} failed-decomposition row(s) from the sample '
                  f'(worst chi2 = {np.nanmax(_chi2_max_arm[sel_pos][~_chi2_ok_sel]):.4g}, '
                  f'at ' + str(canonical_row_labels(
                      EVERY10_INPUT,
                      [int(np.asarray(e10_triplet["row_index"])[sel_pos][~_chi2_ok_sel][
                          np.nanargmax(_chi2_max_arm[sel_pos][~_chi2_ok_sel])])],
                      corpus_meta_fits=f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_meta_only.fits'
                  )['label'][0]) + ')')
            sel_pos = sel_pos[_chi2_ok_sel]
            _n_pick = int(sel_pos.size)
            print(f'  atlas sample after gate: n = {_n_pick}')
        else:
            print(f'  atlas chi2 gate: all {_n_pick} sampled rows pass '
                  f'(max-arm reduced_chi2 <= {ATLAS_CHI2_MAX:g})')
    else:
        _chi2_max_arm = None
        print('  atlas chi2 gate: SKIPPED (triplet carries no chi2; '
              'build with return_chi2=True)')

    # Moon/zodi role-reversal gate, applied to the drawn sample for the same reason
    # and in the same way as the chi2 gate above: a reversed row's moon
    # coefficients describe zodiacal light, so the per-component attribution below
    # would credit the wrong family.  Filtering after the draw keeps the survivors a
    # strict subset of the rows earlier runs used.
    with fits.open(EVERY10_INPUT) as _hdul_rev:
        _wave_rev = np.asarray(_hdul_rev['WAVE'].data, dtype=np.float64)
    _wave_rev = _wave_rev if _wave_rev.ndim == 1 else _wave_rev[0]
    _rev_keep_sel = split_zodi_reversal_keep_mask(
        {'near': EVERY10_NEAR, 'far': EVERY10_FAR, 'sci': EVERY10_SCI},
        np.asarray(e10_triplet['row_index'], dtype=np.int64)[sel_pos],
        _wave_rev, label='atlas')
    if not bool(np.all(_rev_keep_sel)):
        sel_pos = sel_pos[_rev_keep_sel]
        _n_pick = int(sel_pos.size)
        print(f'  atlas sample after reversal gate: n = {_n_pick}')

    # Science-continuum colour gate, same placement and rationale: where the
    # science fibre's continuum colour disagrees with both sky arms, the Moon_bs
    # spline has absorbed field continuum, so the per-component attribution below
    # would charge the moon for something that is not sky.
    _col_keep_sel = sci_continuum_colour_keep_mask(
        EVERY10_INPUT,
        np.asarray(e10_triplet['row_index'], dtype=np.int64)[sel_pos],
        label='atlas')
    if not bool(np.all(_col_keep_sel)):
        sel_pos = sel_pos[_col_keep_sel]
        _n_pick = int(sel_pos.size)
        print(f'  atlas sample after science-continuum colour gate: n = {_n_pick}')

    # Diffuse-zeroed gate: a collapsed diffuse block makes the per-component
    # attribution below meaningless for that row -- the continuum's share of the
    # residual is computed against a truth of ~0.
    _dz_keep_sel = diffuse_zeroed_keep_mask(
        {'near': np.asarray(e10_triplet['coef_near'])[sel_pos],
         'far': np.asarray(e10_triplet['coef_far'])[sel_pos],
         'sci': np.asarray(e10_triplet['coef_sci'])[sel_pos]},
        e10_triplet['coef_names'], label='atlas')
    if not bool(np.all(_dz_keep_sel)):
        sel_pos = sel_pos[_dz_keep_sel]
        _n_pick = int(sel_pos.size)
        print(f'  atlas sample after diffuse-zeroed gate: n = {_n_pick}')

    sel_rows = np.asarray(e10_triplet['row_index'], dtype=np.int64)[sel_pos]
    # sel_rows index the every10 arrays; labels must use the canonical identity
    # because every10 row N != corpus row N (measured: every10 493 is corpus 4930).
    _row_ident = canonical_row_labels(
        EVERY10_INPUT, sel_rows,
        corpus_meta_fits=f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_meta_only.fits')
    _row_label = list(_row_ident['label'])

    # ML predictions for these rows.
    coef_sci_pred_atlas = predict_sci_coefficients_default(
        mlp_artifacts,
        coef_near_phys=np.asarray(e10_triplet['coef_near'][sel_pos], dtype=np.float32),
        coef_far_phys=np.asarray( e10_triplet['coef_far'][sel_pos],  dtype=np.float32),
        ctx_near_phys=np.asarray( e10_triplet['ctx_near'][sel_pos],  dtype=np.float32),
        ctx_far_phys=np.asarray(  e10_triplet['ctx_far'][sel_pos],   dtype=np.float32),
        ctx_sci_phys=np.asarray(  e10_triplet['ctx_sci'][sel_pos],   dtype=np.float32),
    ).astype(np.float32)
    coef_sci_true_atlas = np.asarray(e10_triplet['coef_sci'][sel_pos], dtype=np.float32)

    # Load wavelength grid + LSF ref from the input FITS (matches the recon cell).
    with fits.open(EVERY10_INPUT) as hdul:
        wave_arr = np.asarray(hdul['WAVE'].data, dtype=np.float64)
        lsf_sci_arr = np.asarray(hdul['LSF_SCI'].data, dtype=np.float64)

    _wave_ref_recon = wave_arr if wave_arr.ndim == 1 else wave_arr[int(sel_rows[0])]
    base_dir_guess = _infer_base_dir_for_reconstruction()

    from mlp_predictor.data import make_reconstruction_decomposer
    _telluric_for = globals().get('TELLURIC_ROW_FOR')
    _lsf_model = SkyDecompLSFSurfaceIterative(
        _wave_ref_recon, lsf_sigma=1.0, n_spline_knots=N_MOON_KNOTS,
        base_dir=base_dir_guess,
        split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
    )


    def _precache_decomp_state(decomp_path):
        _state = {'path': Path(decomp_path), 'has_lsf': False, 'o2_cube': None}
        if not _state['path'].exists():
            return _state
        try:
            with fits.open(str(_state['path']), memmap=False) as _hdul_dec:
                _names = {h.name for h in _hdul_dec}
                _state['has_lsf'] = all(_e in _names for _e in ('LSF_COEF', 'LSF_KNOTS', 'LSF_META'))
                if 'VECTOR_O2' in _names:
                    _data = np.asarray(_hdul_dec['VECTOR_O2'].data, dtype=np.float64)
                    if _data.ndim == 2:
                        _state['o2_cube'] = _data
        except (KeyError, IndexError, ValueError) as _exc:
            print(f'  precache failed for {_state["path"].name}: {_exc}')
        return _state


    _state_sci = _precache_decomp_state(EVERY10_SCI)


    # Reconstruction components -> ML coefficient groups (§3.2).  'oh' and 'o2'
    # both belong to the mesospheric group; the rest are one-to-one.
    def _reconstruct_sci_total(coef_row, row_idx, lsf_sigma_fallback):
        """Return (total_flux, {group: component_flux}) for one row."""
        _lsf_state = None
        if _state_sci['has_lsf']:
            try:
                _lsf_state = load_lsf_surface_state(str(_state_sci['path']), int(row_idx))
            except (KeyError, IndexError, ValueError):
                _lsf_state = None
        _o2 = None
        if _state_sci['o2_cube'] is not None and int(row_idx) < _state_sci['o2_cube'].shape[0]:
            _o2_row = _state_sci['o2_cube'][int(row_idx)]
            if np.isfinite(_o2_row).any() and float(np.nansum(np.abs(_o2_row))) > 0.0:
                _o2 = _o2_row
        # Telluric variant: the basis is per row, so rebuild instead of reusing the
        # hoisted model.  See full_spectrum_batch_rmse for the reasoning.
        _tel = (None if _telluric_for is None
                else _telluric_for('sci', int(row_idx)))
        _mdl = (_lsf_model if _tel is None else make_reconstruction_decomposer(
            _lsf_model.wave, n_spline_knots=N_MOON_KNOTS, base_dir=base_dir_guess,
            split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS, telluric=_tel))
        if isinstance(_lsf_state, LSFSurfaceState):
            _mdl._set_lsf_state(_lsf_state)
            _mats = _mdl._assemble_refined_matrices()
            if _o2 is not None:
                _mats['o2'] = np.asarray(_o2, float).ravel()[None, :]
            _comps = _mdl._components_from_coef(np.asarray(coef_row, float).ravel(), _mats)
        else:
            # Rare fallback.
            _comps = reconstruct_with_lsf(
                wave=_mdl.wave, coef=coef_row, lsf=lsf_sigma_fallback,
                n_spline_knots=N_MOON_KNOTS, base_dir=base_dir_guess, o2_vector=_o2,
                split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
                telluric=_tel)
        _by_group = _group_components(_comps)
        _total = np.zeros_like(next(iter(_by_group.values())), dtype=np.float64)
        for _v in _by_group.values():
            _total = _total + _v
        return np.asarray(_total, dtype=np.float64), _by_group


    # Reconstruct N rows: shape (n_pick, n_pix).
    _n_pix = _wave_ref_recon.size
    # float32 like the per-component stacks below: these feed medians and
    # percentiles for display, and float64 doubles the largest allocation in the
    # cell for no visible difference.
    _resid = np.full((_n_pick, _n_pix), np.nan, dtype=np.float32)
    _truth = np.full((_n_pick, _n_pix), np.nan, dtype=np.float32)

    # Per-component residual + truth stacks (float32: 2 x 6 groups x ~200 rows x
    # n_pix ~ 120 MB).  The truth stack is what lets the table report each group's
    # error in ITS OWN units, not just its share of the total flux.
    _resid_comp = {_g: np.full((_n_pick, _n_pix), np.nan, dtype=np.float32)
                   for _g in _ATLAS_COMPONENTS}
    _truth_comp = {_g: np.full((_n_pick, _n_pix), np.nan, dtype=np.float32)
                   for _g in _ATLAS_COMPONENTS}
    print(f'  atlas stacks: {(2 + 2 * len(_ATLAS_COMPONENTS)) * _n_pick * _n_pix * 4 / 1e6:.0f} MB '
          f'for {_n_pick} rows x {_n_pix} pixels')

    _t0 = _time.perf_counter()
    for _i, _rr in enumerate(sel_rows):
        _lsf_row = lsf_sci_arr if lsf_sci_arr.ndim == 1 else lsf_sci_arr[int(_rr)]
        _lsf_sigma_fb = _lsf_row / LSF_FWHM_TO_SIGMA
        _y_true, _c_true = _reconstruct_sci_total(coef_sci_true_atlas[_i], _rr, _lsf_sigma_fb)
        _y_pred, _c_pred = _reconstruct_sci_total(coef_sci_pred_atlas[_i], _rr, _lsf_sigma_fb)
        _y_true = _y_true
        _y_pred = _y_pred
        _resid[_i, :] = _y_pred - _y_true
        _truth[_i, :] = _y_true
        for _g in _ATLAS_COMPONENTS:
            _resid_comp[_g][_i, :] = (_c_pred[_g] - _c_true[_g])
            _truth_comp[_g][_i, :] = _c_true[_g]
        if _i and _i % 50 == 0:
            print(f'  reconstructed {_i}/{_n_pick} rows ({_time.perf_counter() - _t0:.1f}s)')
    print(f'  atlas reconstruction: {_time.perf_counter() - _t0:.1f}s '
          f'({_n_pick} rows x 2 spectra)')

# Aggregate residuals per pixel -- over ALL rows first.  The per-wavelength
# tables further down consume these, so they keep their meaning; the moon
# split below is additional, not a replacement.
_med   = np.nanmedian(_resid, axis=0)
_p16   = np.nanpercentile(_resid, 16, axis=0)
_p84   = np.nanpercentile(_resid, 84, axis=0)
_truth_med = np.nanmedian(_truth, axis=0)

# Normalise residual by the local truth median so a common frac_residual axis
# is meaningful across the range.
_denom = np.where(np.abs(_truth_med) > 1e-30, _truth_med, np.nan)
_med_frac = _med   / _denom

# ---- Moon-up / moon-down split (2026-09-10) -----------------------------
# The two regimes are different problems, and averaging them hides both.
# Moon-down rows have no moon component to speak of (the block is pinned at
# the amplitude floor) and a much fainter sky, so their single-fibre noise is
# larger in fractional terms; moon-up rows are brighter, so the SAME
# fractional error is far more visible against the photon floor.  One shared
# band would be wrong for both, hence a band per panel.
#
# Gate: `moon_frac_po` when the ctx carries it, `moon_alt > 0` otherwise.
# The proxy-fraction gate is the better one -- it cut moon-down contamination
# from 6.1% to 0.6% where it was measured -- but moon_alt is what
# naive_baseline and ensemble_spread_calibration use, so which one ran is
# printed rather than left implicit.
MOON_DOWN_FRAC_MAX_ATLAS = 0.02/3.0
_ctx_names_split = [str(_c).lower() for _c in e10_triplet['ctx_names']]
_ctx_sci_split = np.asarray(e10_triplet['ctx_sci'], dtype=np.float64)[sel_pos]
_moon_up_atlas = None
_moon_gate_name = None
if 'moon_frac_po' in _ctx_names_split:
    _mfp = _ctx_sci_split[:, _ctx_names_split.index('moon_frac_po')]
    _moon_up_atlas = np.isfinite(_mfp) & (_mfp > MOON_DOWN_FRAC_MAX_ATLAS)
    _moon_gate_name = f'moon_frac_po > {MOON_DOWN_FRAC_MAX_ATLAS:.4g}'
elif 'moon_alt' in _ctx_names_split:
    _malt = _ctx_sci_split[:, _ctx_names_split.index('moon_alt')]
    _moon_up_atlas = np.isfinite(_malt) & (_malt > 0.0)
    _moon_gate_name = 'moon_alt > 0'

_atlas_subsets = []
if _moon_up_atlas is None:
    print('  [atlas] neither moon_frac_po nor moon_alt in ctx_names -> '
          'no moon split; showing all rows in one panel.')
    _atlas_subsets = [('all rows', np.ones(_n_pick, dtype=bool))]
else:
    _n_up = int(_moon_up_atlas.sum())
    _n_dn = int((~_moon_up_atlas).sum())
    print(f'  [atlas] moon split on {_moon_gate_name}: '
          f'{_n_dn} down / {_n_up} up of {_n_pick}')
    _atlas_subsets = [('moon down', ~_moon_up_atlas), ('moon up', _moon_up_atlas)]

# ---- Single-fibre photon noise reference band (2026-09-10) --------------
# The bar this atlas is really being judged against.  The sky model gets
# subtracted from ONE fibre at a time, not from the stack, so the residual
# only matters where it rises above the shot noise of a single fibre:
#
#     sigma(lambda) = sqrt(flux * sens / (exptime * dwave))     [N_eff = 1]
#
# `flux` is the SUBSET's median truth flux per pixel, so each panel's band is
# "the noise of one fibre on a typical row of that regime" -- the same
# aggregation its median residual line uses, which is what makes the two
# comparable.  Wherever the median residual sits inside the band, the model is
# already below the noise it will be subtracted from and nothing is to be
# gained there; a residual that leaves the band is real error.
ATLAS_EXPTIME_S = 900.0   # keep equal to CHI2_EXPTIME_S in the batch-rmse cell
_atlas_sens = None
try:
    from mlp_predictor.noise import (
        load_absolute_sensitivity as _atlas_load_sens,
        photon_variance_absolute as _atlas_phot_var)
    _atlas_sens = np.asarray(_atlas_load_sens(_wave_ref_recon), dtype=np.float64)
    _atlas_dwave = float(np.median(np.diff(np.asarray(_wave_ref_recon,
                                                      dtype=np.float64))))
    # The reconstruction is in FIT units (x FACTOR); the sensitivity is
    # absolute, so the flux has to go back to native before it is used and the
    # sigma has to come back afterwards.
    _atlas_factor = float(globals().get('FACTOR', 1e14))
except Exception as _exc_sig:
    print(f'  [atlas] single-fibre noise band unavailable '
          f'({type(_exc_sig).__name__}: {_exc_sig}); plotting without it.')


def _sigma_one_fibre(_truth_med_sub):
    """+/-1 sigma of ONE fibre, in FIT units, from a median truth spectrum.

    No variance floor: this is a reference band, not a chi2 weight, and a
    floored sigma would draw a band where there is no signal to speak of.
    Pixels with non-positive median truth flux simply break the band.
    """
    if _atlas_sens is None:
        return None
    _fn = np.asarray(_truth_med_sub, dtype=np.float64) / _atlas_factor
    _var = _atlas_phot_var(_fn[None, :], _atlas_sens,
                           exptime=ATLAS_EXPTIME_S, dwave=_atlas_dwave,
                           n_fibres=None)[0]
    return np.where(np.isfinite(_var) & (_var > 0),
                    np.sqrt(np.abs(_var)) * _atlas_factor, np.nan)


# Per-subset aggregates, in the same shape as the all-rows ones above.
_atlas_agg = []
for _sname, _smask in _atlas_subsets:
    _ns = int(np.count_nonzero(_smask))
    if _ns == 0:
        print(f'  [atlas] subset "{_sname}" is empty; skipping its panels.')
        continue
    _r = _resid[_smask]
    _t = _truth[_smask]
    _a = dict(name=_sname, n=_ns)
    _a['med'] = np.nanmedian(_r, axis=0)
    _a['p16'] = np.nanpercentile(_r, 16, axis=0)
    _a['p84'] = np.nanpercentile(_r, 84, axis=0)
    # 2-sigma-equivalent (95.4%) band, drawn behind the 68% one.
    _a['p02'] = np.nanpercentile(_r, 2.275, axis=0)
    _a['p98'] = np.nanpercentile(_r, 97.725, axis=0)
    _a['truth_med'] = np.nanmedian(_t, axis=0)
    _a['denom'] = np.where(np.abs(_a['truth_med']) > 1e-30, _a['truth_med'], np.nan)
    _a['sig'] = _sigma_one_fibre(_a['truth_med'])
    _atlas_agg.append(_a)

# ---- Figure -------------------------------------------------------------
# One absolute panel and one fractional panel PER SUBSET, then a shared
# reference panel showing each subset's median truth flux -- which is what
# sets its noise band, so it belongs on the same figure.
_n_sub = len(_atlas_agg)
_atlas_rows = 2 * _n_sub + 1
_titles = ([f"Median true - pred (absolute) -- {_a['name']} (n={_a['n']}), "
            f"68%/95% bands + 1-fibre photon noise" for _a in _atlas_agg]
           + [f"Median (true - pred) / truth -- {_a['name']} (n={_a['n']}), "
              f"68%/95% bands + 1-fibre photon noise" for _a in _atlas_agg]
           + ['Median truth flux per subset (sets each band above)'])
_fig = make_subplots(rows=_atlas_rows, cols=1, shared_xaxes=True,
                     subplot_titles=tuple(_titles),
                     vertical_spacing=0.035)

_SUB_COLOURS = [('rgba(0,120,220,{a})', 'steelblue'),
                ('rgba(180,50,50,{a})', 'firebrick'),
                ('rgba(120,60,180,{a})', 'purple')]


def _atlas_band_panel(_row, _lo, _hi, _centre, _sig, _cband, _cline,
                      _name_band, _name_line, _show_legend, _hover_line,
                      _lo2=None, _hi2=None, _name_band2=None):
    """Noise band (behind), then the 95% and 68% bands, then the median line."""
    if _sig is not None:
        # Added first so it renders underneath.  Two traces: an invisible
        # upper edge, then the lower edge filling to it.
        _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_sig,
            mode='lines', line=dict(color='rgba(40,140,60,0.0)'),
            hoverinfo='skip', showlegend=False), row=_row, col=1)
        _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=-_sig,
            mode='lines', line=dict(color='rgba(40,140,60,0.0)'),
            fill='tonexty', fillcolor='rgba(40,140,60,0.22)',
            name=f'+/-1 sigma, ONE {ATLAS_EXPTIME_S:.0f} s fibre',
            hovertemplate='%{x:.0f} A<br>sigma_1fib %{y:.4g}<extra></extra>',
            showlegend=_show_legend), row=_row, col=1)
    if _lo2 is not None:
        _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_hi2,
            mode='lines', line=dict(color=_cband.format(a=0.0)),
            hoverinfo='skip', showlegend=False), row=_row, col=1)
        _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_lo2,
            mode='lines', line=dict(color=_cband.format(a=0.0)),
            fill='tonexty', fillcolor=_cband.format(a=0.10),
            name=_name_band2, showlegend=_show_legend), row=_row, col=1)
    _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_hi,
        mode='lines', line=dict(color=_cband.format(a=0.0)),
        hoverinfo='skip', showlegend=False), row=_row, col=1)
    _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_lo,
        mode='lines', line=dict(color=_cband.format(a=0.0)),
        fill='tonexty', fillcolor=_cband.format(a=0.20),
        name=_name_band, showlegend=_show_legend), row=_row, col=1)
    _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_centre,
        mode='lines', line=dict(color=_cline, width=1.5),
        name=_name_line, showlegend=_show_legend,
        hovertemplate=_hover_line), row=_row, col=1)
    _fig.add_shape(type='line', x0=float(_wave_ref_recon[0]),
                   x1=float(_wave_ref_recon[-1]), y0=0.0, y1=0.0,
                   line=dict(color='black', dash='dot', width=1),
                   row=_row, col=1)


# Drawn as true - pred (2026-09-29), the same sense as the batch-RMSE figure's
# panels (observed / decomposition minus prediction). Only the drawing is
# negated -- the stored `_resid` stacks and every printed statistic below keep
# pred - true -- so the 16th/84th (and 2.3/97.7th) percentiles swap roles as
# band edges.
for _k, _a in enumerate(_atlas_agg):
    _cband, _cline = _SUB_COLOURS[_k % len(_SUB_COLOURS)]
    # Absolute.
    _atlas_band_panel(
        _k + 1, -_a['p84'], -_a['p16'], -_a['med'], _a['sig'], _cband, _cline,
        f"68% band -- {_a['name']}", f"median residual -- {_a['name']}",
        _show_legend=(_k == 0),
        _hover_line='%{x:.0f} A<br>median resid %{y:.4g}<extra></extra>',
        _lo2=-_a['p98'], _hi2=-_a['p02'], _name_band2=f"95% band -- {_a['name']}")
    # Fractional.
    _sig_frac = None if _a['sig'] is None else _a['sig'] / _a['denom']
    _atlas_band_panel(
        _n_sub + _k + 1, -_a['p84'] / _a['denom'], -_a['p16'] / _a['denom'],
        -_a['med'] / _a['denom'], _sig_frac, _cband, _cline,
        f"fractional 68% band -- {_a['name']}",
        f"median fractional residual -- {_a['name']}",
        _show_legend=False,
        _hover_line='%{x:.0f} A<br>median frac %{y:.3%}<extra></extra>',
        _lo2=-_a['p98'] / _a['denom'], _hi2=-_a['p02'] / _a['denom'],
        _name_band2=f"fractional 95% band -- {_a['name']}")
    _fig.update_yaxes(title_text='true - pred (1e-14)', row=_k + 1, col=1)
    _fig.update_yaxes(title_text='(true - pred) / truth', range=[-0.5, 0.5],
                      row=_n_sub + _k + 1, col=1)
    # Reference panel: this subset's median truth flux.
    _fig.add_trace(go.Scatter(x=_wave_ref_recon, y=_a['truth_med'],
        mode='lines', line=dict(color=_cline, width=1),
        name=f"truth median -- {_a['name']}", showlegend=True),
        row=_atlas_rows, col=1)

_fig.update_yaxes(title_text='truth median flux (1e-14)', row=_atlas_rows,
                  col=1, type='log')
_fig.update_xaxes(title_text='wavelength [Å]', row=_atlas_rows, col=1)
_fig.update_layout(height=300 * _atlas_rows + 130, width=1200,
                   legend=dict(orientation='h', yanchor='bottom', y=1.015,
                               xanchor='left', x=0.0, font=dict(size=10)),
                   title=f'Wavelength residual atlas (n = {_n_pick} rows, '
                         f'{_atlas_source})')
_fig.show()

# How much of the range is already below the noise it will be subtracted
# from?  Two readings per subset: the MEDIAN residual (systematic error, what
# the coloured line shows) and the 68% half-width (per-row scatter).  A model
# can be unbiased and still noisy, so both matter.
for _a in _atlas_agg:
    if _a['sig'] is None:
        continue
    _sig = _a['sig']
    _half = 0.5 * (np.asarray(_a['p84'], float) - np.asarray(_a['p16'], float))
    _okm = np.isfinite(_sig) & np.isfinite(_a['med'])
    _okb = np.isfinite(_sig) & np.isfinite(_half)
    _f_med = (100.0 * np.mean(np.abs(_a['med'][_okm]) <= _sig[_okm])
              if _okm.any() else np.nan)
    _f_band = (100.0 * np.mean(_half[_okb] <= _sig[_okb])
               if _okb.any() else np.nan)
    print(f"  [atlas] {_a['name']} (n={_a['n']}) vs ONE "
          f"{ATLAS_EXPTIME_S:.0f} s fibre: median residual inside +/-1 sigma "
          f"on {_f_med:.1f}% of pixels; 68% half-width inside on "
          f"{_f_band:.1f}%.")
    _rat = np.abs(_a['med'][_okm]) / _sig[_okm]
    if _rat.size:
        print(f"  [atlas] {_a['name']}: |median residual| / sigma_1fib -- "
              f"p50 {np.median(_rat):.3g}, p90 "
              f"{np.percentile(_rat, 90):.3g}, max {_rat.max():.3g}")

# Compact summary: RMS of the median residual by band.
_bands = [('blue    3600-5500', 3600.0, 5500.0),
          ('mid     5500-7500', 5500.0, 7500.0),
          ('NIR     7500-9800', 7500.0, 9800.0)]
print()
print('Per-band RMS of the median residual and median fractional residual:')
for _name, _lo, _hi in _bands:
    _mask_lam = (_wave_ref_recon >= _lo) & (_wave_ref_recon < _hi)
    if not _mask_lam.any():
        continue
    _rms_abs = float(np.sqrt(np.nanmean(_med[_mask_lam] ** 2)))
    _rms_frac = float(np.sqrt(np.nanmean(_med_frac[_mask_lam] ** 2)))
    _mean_bias_frac = float(np.nanmean(_med_frac[_mask_lam]))
    print(f'  {_name:<20s} RMS|abs| = {_rms_abs:.3g}   '
          f'RMS|frac| = {_rms_frac:.3g}   mean_bias_frac = {_mean_bias_frac:+.3g}')

# --- Per-component attribution -----------------------------------------------
# Which coefficient group owns the residual in each band?  The decomposition is
# done on a per-pixel MEAN rather than the median because the mean is linear:
# sum_g mean_resid_g == mean_resid_total exactly, so the per-group numbers add
# up to the total and each group's share is meaningful.  A plain mean is however
# hostage to a single bad row, so the mean is TRIMMED: the worst TRIM_FRAC of
# rows by mean |residual| are excluded here (they are reported separately in the
# tail block below).  Trimming preserves additivity because it is still a mean,
# just over a subset.  The median headline above is left untouched so it stays
# comparable to earlier runs.
TRIM_FRAC = 0.05

_row_scale = np.nanmean(np.abs(_resid), axis=1)          # (n_pick,)
_n_trim = int(np.ceil(TRIM_FRAC * _n_pick))
_order = np.argsort(-np.nan_to_num(_row_scale, nan=-np.inf))
_tail_rows, _keep_rows = _order[:_n_trim], _order[_n_trim:]

_mean_tot = np.nanmean(_resid[_keep_rows], axis=0)
_mean_comp = {_g: np.nanmean(_v[_keep_rows].astype(np.float64), axis=0)
              for _g, _v in _resid_comp.items()}
_truth_comp_mean = {_g: np.nanmean(_v[_keep_rows].astype(np.float64), axis=0)
                    for _g, _v in _truth_comp.items()}
_truth_mean = np.nanmean(_truth[_keep_rows], axis=0)
_denom_mean = np.where(np.abs(_truth_mean) > 1e-30, _truth_mean, np.nan)

_addit_err = float(np.nanmax(np.abs(sum(_mean_comp.values()) - _mean_tot)))
_addit_scale = float(np.nanmax(np.abs(_mean_tot))) or 1.0
print()
print(f'Per-component attribution of the {int(100 * (1 - TRIM_FRAC))}%-TRIMMED '
      f'MEAN residual, i.e. of the typical row '
      f'({_n_pick - _n_trim} of {_n_pick} rows; additivity check: '
      f'max|sum_g - total| = {_addit_err:.2e}, '
      f'{100.0 * _addit_err / _addit_scale:.2g}% of peak).')
print('  RMS|frac| is each group\'s mean residual over the band divided by the '
      'TOTAL mean truth, so')
print('  the column is directly comparable across groups.  bias_frac_% DOES sum '
      'to the total (the')
print('  mean is linear); share_of_total_% is a ratio of RMS and does NOT -- '
      'independent components')
print('  add in quadrature, so shares near ~40% each mean "no group dominates", '
      'while a share near')
print('  100% means that one group owns the band\'s residual outright.')
print('  flux_share_% is the group\'s share of the band\'s truth flux; '
      'self_bias_% is its summed')
print('  residual over its OWN summed flux -- i.e. how wrong the group is in '
      'its own units, which')
print('  is the number to tune against.  A group can own the total bias while '
      'being only slightly')
print('  wrong in itself (large flux_share) or be badly wrong yet nearly '
      'invisible (small share).')
for _name, _lo, _hi in _bands:
    _mask_lam = (_wave_ref_recon >= _lo) & (_wave_ref_recon < _hi)
    if not _mask_lam.any():
        continue
    _rows_c = []
    for _g in list(_ATLAS_COMPONENTS) + ['TOTAL']:
        _v = _mean_tot if _g == 'TOTAL' else _mean_comp[_g]
        _t = _truth_mean if _g == 'TOTAL' else _truth_comp_mean[_g]
        _vf = _v / _denom_mean
        # Band-integrated self-normalisation: divide the group's summed residual
        # by its own summed truth flux over the band.  Integrating before
        # dividing keeps this finite where a component's flux is ~0 at a given
        # pixel (moon on moon-down rows, atomic away from its lines).
        _t_band = float(np.nansum(np.abs(_t[_mask_lam])))
        _self_bias = (100.0 * float(np.nansum(_v[_mask_lam])) / _t_band
                      if _t_band > 0 else np.nan)
        _rows_c.append({
            'component': _g,
            'RMS|abs|': float(np.sqrt(np.nanmean(_v[_mask_lam] ** 2))),
            'RMS|frac|_%': 100.0 * float(np.sqrt(np.nanmean(_vf[_mask_lam] ** 2))),
            'bias_frac_%': 100.0 * float(np.nanmean(_vf[_mask_lam])),
            'flux_share_%': (100.0 * _t_band
                             / max(float(np.nansum(np.abs(_truth_mean[_mask_lam]))), 1e-300)),
            'self_bias_%': _self_bias,
        })
    _df_c = pd.DataFrame(_rows_c)
    _tot_rms = float(_df_c.loc[_df_c['component'] == 'TOTAL', 'RMS|abs|'].iloc[0])
    _df_c['share_of_total_%'] = np.where(
        _df_c['component'] == 'TOTAL', np.nan,
        100.0 * _df_c['RMS|abs|'] / max(_tot_rms, 1e-300))
    print()
    print(f'  --- band {_name} ---')
    print(_df_c.to_string(index=False, float_format=lambda v: f'{v:.4g}',
                          na_rep='-'))
    _worst = _df_c[_df_c['component'] != 'TOTAL'].sort_values(
        'RMS|abs|', ascending=False).iloc[0]
    print(f"    dominant: {_worst['component']} "
          f"({_worst['share_of_total_%']:.0f}% of the band's total residual RMS, "
          f"bias {_worst['bias_frac_%']:+.2f}%)")

# --- Tail concentration ------------------------------------------------------
# How much did the trim above actually remove?  If the untrimmed mean is far
# from the trimmed mean, a few rows carry the sample and any conclusion drawn
# from an untrimmed statistic is a statement about them, not about the model.
print()
print(f'Tail concentration: per-band fractional bias computed three ways '
      f'(n={_n_pick} rows, trim drops the worst {_n_trim}).')
print('  If trimmed ~ median and both << mean, the mean-based attribution above '
      'is a tail statement,')
print('  not a description of the typical row -- fix the tail, not the group '
      'weights.')
_mean_raw = np.nanmean(_resid, axis=0)
_denom_raw = np.nanmean(_truth, axis=0)
_denom_raw = np.where(np.abs(_denom_raw) > 1e-30, _denom_raw, np.nan)
_rows_t = []
for _name, _lo, _hi in _bands:
    _mask_lam = (_wave_ref_recon >= _lo) & (_wave_ref_recon < _hi)
    if not _mask_lam.any():
        continue
    _rows_t.append({
        'band': _name,
        'bias_mean_%':   100.0 * float(np.nanmean((_mean_raw / _denom_raw)[_mask_lam])),
        f'bias_trim{int(100 * TRIM_FRAC)}_%':
            100.0 * float(np.nanmean((_mean_tot / _denom_mean)[_mask_lam])),
        'bias_median_%': 100.0 * float(np.nanmean(_med_frac[_mask_lam])),
    })
print(pd.DataFrame(_rows_t).to_string(index=False,
                                      float_format=lambda v: f'{v:+.3f}'))

# Which rows, and what regime are they in?
_ctx_names_e10 = list(e10_triplet['ctx_names'])
_ctx_sci_e10 = np.asarray(e10_triplet['ctx_sci'], dtype=np.float64)[sel_pos]


def _ctx_col(_nm):
    return (_ctx_sci_e10[:, _ctx_names_e10.index(_nm)]
            if _nm in _ctx_names_e10 else np.full(_n_pick, np.nan))


_moon_alt_e10 = _ctx_col('moon_alt')
_eclb_e10 = _ctx_col('ecl_beta_deg')
_airm_e10 = _ctx_col('airmass')
# chi2 of the sampled rows, so a tail row that is simply a hard observation can
# be told apart from one whose decomposition failed (these survived the gate,
# so all should be <= ATLAS_CHI2_MAX -- a value near the cap is still a warning).
_chi2_sel = (_chi2_max_arm[sel_pos] if _chi2_max_arm is not None
             else np.full(_n_pick, np.nan))
print()
print(f'Worst {_n_trim} rows by mean |residual| (they set the mean-based table):')
_rows_w = []
for _i in _tail_rows:
    _dom = max(_ATLAS_COMPONENTS,
               key=lambda _g: float(np.nanmean(np.abs(_resid_comp[_g][_i]))))
    _rows_w.append({
        'row_index': int(sel_rows[_i]),
        'mean|resid|': float(_row_scale[_i]),
        'x_median_row': float(_row_scale[_i] / max(np.nanmedian(_row_scale), 1e-300)),
        'dominant_comp': _dom,
        'chi2': float(_chi2_sel[_i]),
        'moon_alt': float(_moon_alt_e10[_i]),
        'ecl_beta': float(_eclb_e10[_i]),
        'airmass': float(_airm_e10[_i]),
    })
print(pd.DataFrame(_rows_w).to_string(index=False,
                                      float_format=lambda v: f'{v:.4g}'))
_dom_counts = pd.Series([_r['dominant_comp'] for _r in _rows_w]).value_counts()
print(f'  dominant component among the tail rows: '
      f'{", ".join(f"{_k} x{_v}" for _k, _v in _dom_counts.items())}')
