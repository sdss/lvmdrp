# Full spectrum 100 rows, RMSE stats
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# Explicit: the variant-aware decomposer factory is newer than this cell's
# `required` contract, so do not rely on it being in the shared namespace.
import time as _time

from sky_decomp.result_io import (load_lsf_surface_cubes,
                                  lsf_surface_state_from_cubes)

from mlp_predictor import cell_parallel as _cell_parallel
from mlp_predictor.data import (make_reconstruction_decomposer,
                                make_telluric_row_lookup,
                                science_line_mask_rows)
from sky_decomp.moon_zodi_model import LSF_FWHM_TO_SIGMA

RUN_RMSE_SUBSET_EVAL = True  # Set True to execute this slower evaluation cell.

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
    "FACTOR",
]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError("Run the training + residual-correction cells first. Missing: " + ", ".join(missing))

if not RUN_RMSE_SUBSET_EVAL:
    print("Cell 19 skipped. Set RUN_RMSE_SUBSET_EVAL = True to run the random-subset RMSE evaluation.")
else:
    # Use the same file inputs as the reconstruction diagnostic cell.
    # FULL-CORPUS inputs (2026-09-23).  This cell used to read the every10
    # products -- a 1-in-10 subsample of the stack.  Intersected with the
    # held-out split those leave only 299 of the 2900 validation+test rows, so
    # it scored a tenth of the sample it claimed to.  The corpus files carry
    # every row.  Their size (28 GB per arm) costs nothing here because nothing
    # reads a whole plane: flux, VECTOR_O2 and FLUX_SIGMA_TOTAL are sliced to
    # the selected rows, and the LSF cubes are read once rather than per row.
    _eval_stem = f"{DECOMP_DATA_ROOT}/{DECOMP_STEM}"
    _eval_suffix = _DECOMP_SUFFIX  # inherited from cell 6
    EVAL_INPUT = f"{_eval_stem}.fits"
    EVAL_NEAR = f"{_eval_stem}_decomp_sky1{_eval_suffix}.fits"
    EVAL_FAR = f"{_eval_stem}_decomp_sky2{_eval_suffix}.fits"
    EVAL_SCI = f"{_eval_stem}_decomp_sci{_eval_suffix}.fits"

    # None = every held-out row that passes the gates.  An integer caps it, and
    # the cap is drawn phase-stratified as before.  Overridden by the `size=`
    # kwarg of Diagnostics.full_spectrum_batch_rmse, which rewrites this
    # literal; edit there, not here.
    n_sample = None
    rng_seed = 42
    # Reconstruction workers.  The loop below is ~0.3 s/row of numpy in a
    # per-row model whose state is mutated in place, so it parallelises cleanly
    # across processes and not at all across threads.  1 forces the serial path.
    n_workers = 8

    # Cap on the number of per-row LINES drawn in the residual figure.
    # Overridden by the `stroked=` kwarg of Diagnostics.full_spectrum_batch_rmse,
    # which rewrites this literal; edit there, not here.  The statistics below
    # always use all n_sample rows -- this only limits what is stroked,
    # because the cost of a line is enormous and its value collapses fast.  Each line carries 12401 points, and the figure has FIVE spectrum
    # panels, so the payload is 5 x n_lines x 12401 numbers: measured as JSON,
    # 152 MB at n=100, 303 MB at n=200 and 758 MB at n=500 -- and the kernel
    # holds the figure, the JSON and the frontend copy at once.  That is what
    # kills the kernel above ~200 rows, not the reconstruction loop.
    #
    # 60 lines is already past the point where more ink adds information:
    # they overlap into a band, and the RMS envelope and the right-column
    # histograms -- both computed from ALL rows -- are what the eye reads.
    MAX_RESID_LINES = 60
    # Sky-arm residual correction (mlp_predictor.sky_arm_correction): each
    # row's near and far decomposition residual, inverse-variance weighted,
    # scaled per pixel by predicted-science-model / arm-model, applied in full
    # up to RESIDUAL_CORRECTION_MAX_A and tapered to zero over the next 100 A.
    # Applied to panel 1 (obs - pred) and scored as a second chi2 next to the
    # uncorrected one; sci_residuals and every other
    # statistic stay uncorrected.  RESIDUAL_SMOOTHING_A is 'auto' (3-pixel
    # median on faint rows only, the measured optimum), a running-median
    # width in A, or 0.
    RESIDUAL_CORRECTION = False
    RESIDUAL_CORRECTION_MAX_A = 5000.0
    RESIDUAL_SMOOTHING_A = 'auto'
    # Sky-line scaling (mlp_predictor.sky_line_scaling): after the correction
    # above, refit the brightness of the predicted sky lines on the science
    # spectrum itself -- one scale per OH vibrational band (+ a rotational-
    # temperature tilt when LINE_SCALING_TILT), and per atomic line family and
    # O2 -- on a LINEAR high-pass of width LINE_SCALING_HIGHPASS_A that removes
    # the science continuum; robust (Huber) with a LINE_SCALING_PRIOR (1 sigma,
    # fractional) pull toward the prediction.  Na D, K I, [O I] 6300/6364 and
    # N I keep their predicted brightness, and the red nebular lines are masked
    # (see the module notes on protecting the science signal).  Scored like
    # the correction above.
    LINE_SCALING = False
    LINE_SCALING_HIGHPASS_A = 25.0
    LINE_SCALING_PRIOR = 0.1
    LINE_SCALING_TILT = True

    # Rows whose PER-COMPONENT reconstruction is kept for wavelength_residual_
    # atlas to reuse.  None (the default since 2026-09-29) hands over EVERY row
    # scored here, so the atlas describes exactly the held-out sample this
    # cell scores, not a subset of it or its own every10 draw.  Cost: six
    # component spectra x (pred, true) per row, ~0.65 MB/row, ~1.9 GB for the
    # 2888 held-out rows.  An integer caps it to an evenly spaced subset; 0
    # disables the handoff and the atlas falls back to reconstructing its own
    # every10 sample.
    ATLAS_HANDOFF_ROWS = None

    # Must match _ATLAS_COMPONENTS in wavelength_residual_atlas.py -- the atlas
    # indexes the handoff by these exact keys, so a mismatch raises rather than
    # silently mis-attributing.
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
        return {_g: sum((np.asarray(_comps.get(_k, 0.0), dtype=np.float64)
                         for _k in _keys), start=np.float64(0.0))
                for _g, _keys in _ATLAS_COMPONENTS.items()}

    # 1) Select the rows to score straight from `filtered_triplet`.
    #
    # No triplet rebuild and no re-gating: `filtered_triplet` IS the gated
    # corpus (chi2, field, reversal, colour, diffuse-zeroed and the kappa
    # filters were all applied in the data-load cell), and the split indices
    # address it directly.  The every10 path had to redo the gates because it
    # built its own ungated triplet; that is gone, and with it the risk of the
    # two gate chains drifting apart.
    _EVAL_SPLIT = 'heldout'   # 'heldout' (val+test) | 'test' | 'val' | 'all'
    _n_ft = int(np.asarray(filtered_triplet["coef_sci"]).shape[0])
    _split_pos = {
        'heldout': lambda: np.concatenate([np.asarray(val_idx, dtype=int),
                                           np.asarray(test_idx, dtype=int)]),
        'val':     lambda: np.asarray(val_idx, dtype=int),
        'test':    lambda: np.asarray(test_idx, dtype=int),
        'all':     lambda: np.arange(_n_ft, dtype=int),
    }[_EVAL_SPLIT]()
    _eval_pos_all = np.unique(np.asarray(_split_pos, dtype=int))
    n_rows = int(_eval_pos_all.size)
    print(f"  scoring the '{_EVAL_SPLIT}' split: {n_rows}/{_n_ft} filtered "
          f"corpus rows"
          + ("  <- TRAINING ROWS INCLUDED; the RMSE below is optimistic"
             if _EVAL_SPLIT == 'all' else ""))
    if n_rows == 0:
        raise RuntimeError(f"The '{_EVAL_SPLIT}' split is empty.")

    n_use = n_rows if n_sample is None else int(min(int(n_sample), n_rows))
    rng = np.random.default_rng(rng_seed)
    _moon_phase_ft = _moon_phase_deg_from_ctx(filtered_triplet)
    if n_use >= n_rows:
        # Taking everything: no draw to stratify, so the result stops being a
        # function of `seed`.
        sel_ft = np.sort(_eval_pos_all)
        print(f"  using ALL {n_use} rows of the split (no subsampling)")
    else:
        # Stratify the cap by lunar phase so a subsample still spans dark ->
        # bright roughly uniformly (matches split_indices_by_moon_phase).  A
        # plain choice would inherit whichever phase quantiles happen to hold
        # more rows, and sci_pred_vs_true would then describe that region
        # rather than deployment conditions.
        _valid_phase = _moon_phase_ft[_eval_pos_all]
        if not np.isfinite(_valid_phase).all():
            raise RuntimeError('Non-finite moon_phase in the split rows; '
                               'cannot stratify by lunar phase.')
        _n_phase_bins = int(min(10, n_use))
        _phase_edges = np.quantile(_valid_phase,
                                   np.linspace(0.0, 1.0, _n_phase_bins + 1))
        _phase_edges[0], _phase_edges[-1] = -np.inf, np.inf
        _bin_id = np.digitize(_valid_phase, _phase_edges[1:-1], right=False)
        # Round-robin quota, +1s scattered so no bin is systematically favored.
        _quota = np.full(_n_phase_bins, n_use // _n_phase_bins, dtype=int)
        _quota[:n_use - int(_quota.sum())] += 1
        rng.shuffle(_quota)
        _picked = []
        for _b in range(_n_phase_bins):
            _in_bin = _eval_pos_all[_bin_id == _b]
            _take = int(min(_quota[_b], _in_bin.size))
            if _take > 0:
                _picked.append(rng.choice(_in_bin, size=_take, replace=False))
        _selected = (np.concatenate(_picked).astype(int)
                     if _picked else np.array([], dtype=int))
        _shortfall = n_use - _selected.size
        if _shortfall > 0:
            _remaining = np.setdiff1d(_eval_pos_all, _selected)
            if _remaining.size >= _shortfall:
                _selected = np.concatenate(
                    [_selected, rng.choice(_remaining, size=_shortfall,
                                           replace=False)])
        sel_ft = np.sort(_selected)
    # Positions in `filtered_triplet` and the corresponding CORPUS FITS rows.
    # These are corpus rows now, not every10 rows -- consumers must not index
    # an every10 file with them, which is why `rmse_subset_results` records the
    # files they belong to.
    sel_pos = sel_ft
    sel_rows = np.asarray(filtered_triplet["row_index"], dtype=int)[sel_ft]
    n_use = int(sel_ft.size)

    _row_ident = canonical_row_labels(EVAL_INPUT, sel_rows)
    _row_label = list(_row_ident['label'])
    print(f'  row identity: labels carry expnum + corpus row')

    _sel_phases = _moon_phase_ft[sel_ft]
    _phase_how = ("every row of the split, unstratified"
                  if n_use >= n_rows
                  else f"stratified across {_n_phase_bins} quantile bins")
    print(f"  sample: n_use={n_use}, {_phase_how}; "
          f"phase deg quartiles (min / 25 / 50 / 75 / max) = "
          f"{float(np.min(_sel_phases)):.1f} / "
          f"{float(np.percentile(_sel_phases, 25)):.1f} / "
          f"{float(np.percentile(_sel_phases, 50)):.1f} / "
          f"{float(np.percentile(_sel_phases, 75)):.1f} / "
          f"{float(np.max(_sel_phases)):.1f}")

    # Rows whose per-component reconstruction we keep for the atlas: a
    # systematic every-k sample of the selection.  Evenly spaced rather than
    # random so it is reproducible without a seed and cannot cluster.
    if ATLAS_HANDOFF_ROWS is None and n_use > 0:
        _atlas_take = np.arange(n_use)
    elif ATLAS_HANDOFF_ROWS is not None and int(ATLAS_HANDOFF_ROWS) > 0 and n_use > 0:
        _atlas_take = (np.arange(n_use) if n_use <= int(ATLAS_HANDOFF_ROWS)
                       else np.unique(np.linspace(0, n_use - 1,
                                                  int(ATLAS_HANDOFF_ROWS)
                                                  ).astype(int)))
    else:
        _atlas_take = np.array([], dtype=int)
    _atlas_keep = np.zeros(n_use, dtype=bool)
    _atlas_keep[_atlas_take] = True
    if _atlas_take.size:
        print(f"  atlas handoff: keeping per-component reconstructions for "
              f"{_atlas_take.size}/{n_use} rows "
              f"({_atlas_take.size * 12401 * 6 * 2 * 4 / 1e6:.0f} MB)")

    # 2) Observed spectra, wavelength grid and LSF -- SELECTED ROWS ONLY.
    #    A whole FLUX plane of the corpus stack is 925 MB; three of them would
    #    be 2.8 GB for rows we mostly do not touch.  Fancy-indexing the memmap
    #    reads just the rows we need, and they are then addressed by position
    #    in the selection (`i`), not by corpus row (`rr`).
    _t_load0 = _time.perf_counter()
    with fits.open(EVAL_INPUT, memmap=True) as hdul:
        flux_near_sel = np.asarray(hdul["FLUX_SKY_NEAR"].data[sel_rows],
                                   dtype=np.float32)
        flux_far_sel = np.asarray(hdul["FLUX_SKY_FAR"].data[sel_rows],
                                  dtype=np.float32)
        flux_sci_sel = np.asarray(hdul["FLUX_SCI"].data[sel_rows],
                                  dtype=np.float32)
        wave_arr = np.asarray(hdul["WAVE"].data, dtype=np.float64)
        _lsf_sci_full = hdul["LSF_SCI"].data
        lsf_sci_sel = (np.asarray(_lsf_sci_full, dtype=np.float32)
                       if np.ndim(_lsf_sci_full) == 1
                       else np.asarray(_lsf_sci_full[sel_rows], dtype=np.float32))
        _expnum_sel = None
        if "META" in hdul:
            _meta_ev = Table(hdul["META"].data)
            _meta_up_ev = {c.upper(): c for c in _meta_ev.colnames}
            _expnum_col = next(
                (_meta_up_ev[k] for k in ("EXPNUM", "EXP_NUM", "EXPOSURE")
                 if k in _meta_up_ev), None)
            if _expnum_col is not None:
                _expnum_sel = np.asarray(_meta_ev[_expnum_col])[sel_rows]
    print(f"  observed spectra: {n_use} rows x 3 arms loaded in "
          f"{_time.perf_counter() - _t_load0:.1f} s "
          f"({3 * flux_sci_sel.nbytes / 1e6:.0f} MB)")
    # The science field's own emission lines, masked exactly as the
    # decomposition masked them (windows from the stack's reference LSF,
    # centred on each row's measured Halpha velocity).  They are the TARGET's
    # light, modelled by neither the prediction nor the decomposition, and on
    # HII-region rows they would otherwise dominate the chi2 below.
    _t_mask0 = _time.perf_counter()
    try:
        sci_line_mask_sel = science_line_mask_rows(
            EVAL_INPUT, wave_arr if np.ndim(wave_arr) == 1 else wave_arr[0],
            flux_sci_sel, flux_near_sel)
        print(f"  science-line mask: median {int(np.median(sci_line_mask_sel.sum(1)))} px "
              f"per row excluded from the chi2 ({_time.perf_counter() - _t_mask0:.1f} s)")
    except Exception as _exc_mask:
        sci_line_mask_sel = None
        print(f"  science-line mask unavailable ({type(_exc_mask).__name__}: "
              f"{_exc_mask}); the chi2 includes the science emission lines.")

    # 3) Predict SCI coefficients for the selected rows.
    coef_near_sel = np.asarray(coef_near_all[sel_ft], dtype=np.float64)
    coef_far_sel = np.asarray(coef_far_all[sel_ft], dtype=np.float64)
    coef_sci_sel = np.asarray(coef_sci_all[sel_ft], dtype=np.float64)
    coef_sci_pred = predict_sci_coefficients_default(
        mlp_artifacts,
        coef_near_phys=coef_near_all[sel_ft],
        coef_far_phys=coef_far_all[sel_ft],
        ctx_near_phys=ctx_near_all[sel_ft],
        ctx_far_phys=ctx_far_all[sel_ft],
        ctx_sci_phys=ctx_sci_all[sel_ft],
    ).astype(np.float64)

    # 4) Reconstruct and compute per-row RMSE + pixel-space WRMSE.
    base_dir_guess = _infer_base_dir_for_reconstruction()
    near_rmse = np.full(n_use, np.nan, dtype=np.float64)
    far_rmse = np.full(n_use, np.nan, dtype=np.float64)
    sci_rmse = np.full(n_use, np.nan, dtype=np.float64)
    near_wrmse = np.full(n_use, np.nan, dtype=np.float64)
    far_wrmse = np.full(n_use, np.nan, dtype=np.float64)
    sci_wrmse = np.full(n_use, np.nan, dtype=np.float64)

    # Try to load per-pixel sigma HDUs from the decomposition FITS files
    # up-front (fast path when new decompositions land).  Falls back to
    # on-the-fly propagation via coef_err inside the loop when absent.
    # Selected rows only: a full FLUX_SIGMA_TOTAL plane is 1.85 GB per arm.
    # Indexed by position in the selection from here on, not by corpus row.
    _pix_sigma_near_all = load_pixel_sigma_if_available(EVAL_NEAR, sel_rows)
    _pix_sigma_far_all  = load_pixel_sigma_if_available(EVAL_FAR, sel_rows)
    _pix_sigma_sci_all  = load_pixel_sigma_if_available(EVAL_SCI, sel_rows)
    _pix_sigma_source = {
        arm: ("FITS HDU" if arr is not None else "coef_err propagation")
        for arm, arr in (("near", _pix_sigma_near_all),
                         ("far",  _pix_sigma_far_all),
                         ("sci",  _pix_sigma_sci_all))
    }
    print(f"  pixel sigma source: near={_pix_sigma_source['near']}, "
          f"far={_pix_sigma_source['far']}, sci={_pix_sigma_source['sci']}")

    # Grab coef_err arrays for the fallback path (may be all-NaN when the
    # decomposition FITS lacks a COEF_ERR HDU; the WRMSE helper falls back
    # to floor-only weighting so nothing breaks).
    _cerr_near_sel = (np.asarray(coef_err_near_all, dtype=np.float64)[sel_ft]
                      if _pix_sigma_near_all is None else None)
    _cerr_far_sel  = (np.asarray(coef_err_far_all, dtype=np.float64)[sel_ft]
                      if _pix_sigma_far_all is None else None)
    _cerr_sci_sel  = (np.asarray(coef_err_sci_all, dtype=np.float64)[sel_ft]
                      if _pix_sigma_sci_all is None else None)
    sci_resid_rows = []
    sci_corr_rows = []         # per-row sky correction(s) to add to the prediction (or None)
    sci_line_scale_rows = []   # per-row {template: scale} from LINE_SCALING (or None)
    sci_corr_arm_rows = []     # per-row sky-arm part of the correction alone (or None)
    sci_wave_rows = []
    sci_obs_rows = []          # observed sci flux, for the photon chi2 below
    # DECOMPOSITION SELF-FIT residual: recon(coef_sci_TRUE) - observed sci.
    # This is the decomposition's own miss on the row it was fitted to, so it
    # is the floor the ML transfer is measured against: the predictor cannot
    # beat the model it is predicting the coefficients OF.  Kept in native
    # units and float32 like the others -- at 500 rows the float64 version of
    # this one array is another 50 MB on top of an already tight cell.
    sci_selfres_rows = []
    # Per-component residuals: comps_sci[<comp>] - comps_sci_true[<comp>] per row,
    # stored in native units (like sci_resid_rows) and multiplied by FACTOR at plot time.
    sci_moon_resid_rows = []
    sci_zodi_resid_rows = []
    sci_diffuse_resid_rows = []
    sci_lines_resid_rows = []

    def _lines_sum(_c):
        return (np.asarray(_c["oh"], dtype=np.float64)
                + np.asarray(_c["atom"], dtype=np.float64)
                + np.asarray(_c["orc"], dtype=np.float64)
                + np.asarray(_c["o2"], dtype=np.float64))

    # 4a) Optimization (2026-08-11): build ONE reconstruction model outside the loop and
    #     precache per-file LSF-availability + full VECTOR_O2 cubes. Previously the loop
    #     rebuilt SkyDecompLSFSurfaceIterative (basis + solar-reference + moon spline)
    #     3 x n_use times and re-read VECTOR_O2 on every call, both of which dominated
    #     the runtime. See §12 (2026-08-11 batch-RMSE cell reconstruction hoisted).
    _t_recon0 = _time.perf_counter()
    _wave_ref_recon = (wave_arr if wave_arr.ndim == 1
                       else np.asarray(wave_arr[int(sel_rows[0])], dtype=np.float64))
    if wave_arr.ndim > 1:
        _wave_probe = np.asarray(wave_arr[int(sel_rows[-1])], dtype=np.float64)
        if _wave_probe.shape != _wave_ref_recon.shape or not np.allclose(
                _wave_probe, _wave_ref_recon, rtol=0.0, atol=1e-8):
            raise RuntimeError(
                "wave_arr rows differ between sampled rows; model hoisting assumes a shared grid. "
                "Fall back to per-row reconstruct_with_lsf if this ever triggers on your dataset.")
    # DECOMPOSITION VARIANT.  `TELLURIC_ROW_FOR` is a callable f(kind, row) that
    # the notebook installs for a telluric corpus (see data.DECOMP_VARIANTS) and
    # leaves as None for the production split-zodi one.  It matters here because
    # the telluric design matrix is divided by that ROW's DRP transmission, so
    # unlike the split-zodi basis it cannot be hoisted -- and because the
    # telluric corpus stores its LSF as a continuous M-spline density, which the
    # iterative class cannot read at all.  Rebuilding per row costs ~0.17 s.
    # Telluric transmission must be looked up in the file whose rows we are
    # indexing.  The shared `TELLURIC_ROW_FOR` is bound to
    # `input_fits_for_basis`, i.e. the EVERY10 stack (1867 rows), so feeding it
    # a corpus row raises IndexError at best and returns another exposure's
    # transmission at worst.  Build this cell's own lookup against EVAL_INPUT.
    _telluric_for = None
    if globals().get('TELLURIC_ROW_FOR') is not None:
        _telluric_for = make_telluric_row_lookup(EVAL_INPUT, verbose=False,
                                                 decomp_suffix=_DECOMP_SUFFIX)
    _model_cache = {}

    def _model_for(telluric):
        """Decomposer whose basis matches how this row was fitted."""
        if telluric is None:
            if 'plain' not in _model_cache:
                _model_cache['plain'] = make_reconstruction_decomposer(
                    _wave_ref_recon, n_spline_knots=N_MOON_KNOTS,
                    base_dir=base_dir_guess, split_zodi=SPLIT_ZODI,
                    n_zodi_spline_knots=N_ZODI_KNOTS, telluric=None)
            return _model_cache['plain']
        return make_reconstruction_decomposer(
            _wave_ref_recon, n_spline_knots=N_MOON_KNOTS,
            base_dir=base_dir_guess, split_zodi=SPLIT_ZODI,
            n_zodi_spline_knots=N_ZODI_KNOTS, telluric=telluric)

    _lsf_model = _model_for(None)
    if _telluric_for is not None:
        print('  reconstruction: TELLURIC variant -- the design matrix is '
              'rebuilt per row (per-row DRP transmission), ~0.17 s/row/arm.')

    def _precache_decomp_state(decomp_path):
        state = {"path": Path(decomp_path), "has_lsf": False, "o2_cube": None,
                 "lsf_cubes": None}
        if not state["path"].exists():
            return state
        try:
            with fits.open(str(state["path"]), memmap=True) as _hdul_dec:
                _ext_names = {h.name for h in _hdul_dec}
                state["has_lsf"] = all(_e in _ext_names
                                       for _e in ("LSF_COEF", "LSF_KNOTS", "LSF_META"))
                if state["has_lsf"]:
                    # Refuse to open an inconsistent LSF cube (observed on the
                    # _p25_every10 decomp files, where LSF_COEF was rewritten
                    # to every10 size but LSF_META was inherited from every1).
                    # Loading anyway would slice cube row k with an n_basis
                    # taken from META row k -- a different fiber -- silently
                    # applying the wrong LSF to ~99% of rows and hitting NaN
                    # padding on the rest.
                    _coef_rows = int(_hdul_dec["LSF_COEF"].data.shape[0])
                    _meta_rows = int(len(_hdul_dec["LSF_META"].data))
                    _expected_meta = 3 * _coef_rows
                    if _meta_rows != _expected_meta:
                        raise RuntimeError(
                            f"Inconsistent LSF HDUs in {state['path'].name}: "
                            f"LSF_COEF has {_coef_rows} rows but LSF_META has "
                            f"{_meta_rows} rows (expected 3 x {_coef_rows} = "
                            f"{_expected_meta}). Regenerate this decomposition "
                            f"file with matching LSF_META; do NOT fall back to "
                            f"LSF_SCI sigma because the per-row LSF would be "
                            f"silently wrong on all other rows."
                        )
                if state["has_lsf"]:
                    # Read LSF_COEF / LSF_KNOTS / LSF_META ONCE.  The per-row
                    # loader re-reads all three on every call (~80 MB on a
                    # corpus file), which at 2900 rows x 3 arms would be of
                    # order half a terabyte of I/O to extract a few MB.
                    state["lsf_cubes"] = load_lsf_surface_cubes(str(state["path"]))
                if "VECTOR_O2" in _ext_names:
                    _o2 = _hdul_dec["VECTOR_O2"].data
                    if np.ndim(_o2) == 2:
                        # Selected rows only: the full cube is 1.85 GB per arm.
                        state["o2_cube"] = np.asarray(_o2[sel_rows],
                                                      dtype=np.float64)
        except (KeyError, IndexError, ValueError) as _exc:
            print(f"  precache failed for {state['path'].name}: "
                  f"{type(_exc).__name__}: {_exc}")
        return state

    _state_near = _precache_decomp_state(EVAL_NEAR)
    _state_far  = _precache_decomp_state(EVAL_FAR)
    _state_sci  = _precache_decomp_state(EVAL_SCI)
    print(f"  precache: has_lsf near/far/sci = "
          f"{_state_near['has_lsf']}/{_state_far['has_lsf']}/{_state_sci['has_lsf']}, "
          f"VECTOR_O2 near/far/sci = "
          f"{_state_near['o2_cube'] is not None}/"
          f"{_state_far['o2_cube'] is not None}/"
          f"{_state_sci['o2_cube'] is not None}")

    def _lsf_state_from_cache(state_dict, row_idx):
        """Build one row's LSF state from the cubes read at precache time."""
        if not state_dict["has_lsf"] or state_dict["lsf_cubes"] is None:
            return None
        try:
            return lsf_surface_state_from_cubes(state_dict["lsf_cubes"],
                                                int(row_idx))
        except (KeyError, IndexError, ValueError) as _exc:
            print(f"  LSF surface state unavailable in {state_dict['path'].name} "
                  f"row {int(row_idx)}: {type(_exc).__name__}: {_exc}")
            return None

    def _o2_vec_from_cache(state_dict, sel_i):
        """VECTOR_O2 for selection position `sel_i` (the cube holds only those)."""
        cube = state_dict["o2_cube"]
        if cube is None or int(sel_i) >= cube.shape[0]:
            return None
        row = cube[int(sel_i)]
        if not np.isfinite(row).any() or float(np.nansum(np.abs(row))) == 0.0:
            return None
        return row

    def _fast_reconstruct(coef, lsf_state, o2_vec, lsf_sigma_fallback,
                          coef_err=None, telluric=None):
        _mdl = _model_for(telluric)
        if isinstance(lsf_state, LSFSurfaceState):
            _mdl._set_lsf_state(lsf_state)
            _mats = _mdl._assemble_refined_matrices()
            if o2_vec is not None:
                _o2_arr = np.asarray(o2_vec, float).ravel()
                if _o2_arr.shape != _mdl.wave.shape:
                    raise ValueError(
                        f"o2_vector shape mismatch: expected {_mdl.wave.shape}, "
                        f"got {_o2_arr.shape}")
                _mats["o2"] = _o2_arr[None, :]
            _coef_arr = np.asarray(coef, float).ravel()
            _comps = _mdl._components_from_coef(_coef_arr, _mats)
            _comps["total"] = (_comps["oh"] + _comps["moon"] + _comps.get("zodi", 0) + _comps["diffuse"]
                                + _comps["atom"] + _comps["orc"] + _comps["o2"])
            if coef_err is not None:
                _err_arr = np.asarray(coef_err, float).ravel()
                _sigmas = _mdl._components_sigma_from_coef_err(_err_arr, _mats)
                _comps["sigma"] = _sigmas
                _comps["sigma_total"] = np.sqrt(
                    _sigmas["oh"] ** 2 + _sigmas["moon"] ** 2
                    + _sigmas.get("zodi", np.zeros_like(_sigmas["moon"])) ** 2
                    + _sigmas["diffuse"] ** 2 + _sigmas["atom"] ** 2
                    + _sigmas["orc"] ** 2 + _sigmas["o2"] ** 2)
            return _comps
        # Rare fallback path: no LSF surface state for this row -- take the slow route.
        return reconstruct_with_lsf(
            wave=_mdl.wave, coef=coef, lsf=lsf_sigma_fallback,
            n_spline_knots=N_MOON_KNOTS, base_dir=base_dir_guess, o2_vector=o2_vec,
            split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
            coef_err=coef_err, telluric=telluric)

    def _row_line_templates(coef, lsf_state, o2_vec, telluric):
        """Sky-line templates (physical units) on the row's own basis, or None.

        Same model, LSF state and O2 vector as `_fast_reconstruct`, so the
        templates sum to exactly the reconstruction's line components.
        """
        if not isinstance(lsf_state, LSFSurfaceState):
            return None
        _mdl = _model_for(telluric)
        _mdl._set_lsf_state(lsf_state)
        _mats = _mdl._assemble_refined_matrices()
        if o2_vec is not None:
            _mats["o2"] = np.asarray(o2_vec, float).ravel()[None, :]
        _t = _line_templates(_mdl, _mats, np.asarray(coef, float).ravel(),
                             tilt=LINE_SCALING_TILT)
        return {k: v / FACTOR for k, v in _t.items()}

    print(f"  recon setup: {_time.perf_counter() - _t_recon0:.2f} s "
          f"(one-time basis build + FITS precache)")

    # Sky-arm residual correction inputs, built once and inherited by the
    # forked workers: the arms' fibre counts (their stack noise) and the
    # absolute sensitivity on the shared grid.
    if RESIDUAL_CORRECTION or LINE_SCALING:
        from mlp_predictor.sky_arm_correction import (
            SkyArm as _SkyArm, arm_photon_variance as _arm_var,
            sky_arm_residual_correction as _sky_arm_corr)
        from mlp_predictor.sky_line_scaling import (
            line_templates as _line_templates,
            sky_line_scaling_correction as _line_scale,
            NEBULAR_LINES_RED as _NEB_RED, UNMEASURED_VELOCITY_KM_S as _NEB_V)
        from mlp_predictor.data import science_line_mask_rows as _sci_mask_ls
        from mlp_predictor.noise import load_absolute_sensitivity as _load_sens_rc
        with fits.open(EVAL_INPUT, memmap=True) as _h_rc:
            _nf_rc = {a: np.asarray(_h_rc["META"].data[f"fibers_sky_{a}_used"],
                                    dtype=np.float64)[np.asarray(sel_rows, dtype=int)]
                      for a in ("near", "far")}
        _sens_rc = (np.asarray(_load_sens_rc(wave_arr), dtype=np.float64)
                    if wave_arr.ndim == 1 else None)

    # The per-row work, factored out of the loop so it can run over a
    # process pool.  It closes over the hoisted model cache, the FITS
    # precache and the telluric lookup; `cell_parallel` forks, so the child
    # inherits all of that instead of rebuilding or shipping it.
    def _row_work(i):
        r = sel_rows[i]
        rr = int(r)
        wave_row = wave_arr if wave_arr.ndim == 1 else np.asarray(wave_arr[rr], dtype=np.float64)
        # Cast in BOTH branches: lsf_sci_sel is float32 on disk, and the
        # 1-D branch would otherwise leak float32 into _lsf_sigma_fallback.
        lsf_row = np.asarray(lsf_sci_sel if lsf_sci_sel.ndim == 1
                             else lsf_sci_sel[i], dtype=np.float64)

        flux_near_true = np.asarray(flux_near_sel[i], dtype=np.float64)
        flux_far_true = np.asarray(flux_far_sel[i], dtype=np.float64)
        flux_sci_true = np.asarray(flux_sci_sel[i], dtype=np.float64)

        _lsf_state_near = _lsf_state_from_cache(_state_near, rr)
        _lsf_state_far  = _lsf_state_from_cache(_state_far,  rr)
        _lsf_state_sci  = _lsf_state_from_cache(_state_sci,  rr)
        _lsf_sigma_fallback = lsf_row / LSF_FWHM_TO_SIGMA

        _o2_vec_near = _o2_vec_from_cache(_state_near, i)
        _o2_vec_far  = _o2_vec_from_cache(_state_far,  i)
        _o2_vec_sci  = _o2_vec_from_cache(_state_sci,  i)

        # For each arm, only ask the reconstructor for sigma when the
        # loaded FITS sigma is absent; otherwise the propagator call would
        # duplicate work already done by the pipeline.
        _cerr_near_row = (_cerr_near_sel[i] if _cerr_near_sel is not None else None)
        _cerr_far_row  = (_cerr_far_sel[i] if _cerr_far_sel is not None else None)
        _cerr_sci_row  = (_cerr_sci_sel[i] if _cerr_sci_sel is not None else None)

        # Per-arm telluric: each arm has its OWN source airmass (the sky lines
        # are attenuated along that arm's line of sight) while the DRP
        # transmission that divides the basis is the SCIENCE one for all three.
        # `_telluric_for` handles both, keyed by kind.
        _tel_near = None if _telluric_for is None else _telluric_for('sky1', rr)
        _tel_far  = None if _telluric_for is None else _telluric_for('sky2', rr)
        _tel_sci  = None if _telluric_for is None else _telluric_for('sci',  rr)
        comps_near = _fast_reconstruct(coef_near_sel[i], _lsf_state_near,
                                        _o2_vec_near, _lsf_sigma_fallback,
                                        coef_err=_cerr_near_row,
                                        telluric=_tel_near)
        comps_far  = _fast_reconstruct(coef_far_sel[i],  _lsf_state_far,
                                        _o2_vec_far,  _lsf_sigma_fallback,
                                        coef_err=_cerr_far_row,
                                        telluric=_tel_far)
        comps_sci  = _fast_reconstruct(coef_sci_pred[i], _lsf_state_sci,
                                        _o2_vec_sci,  _lsf_sigma_fallback,
                                        coef_err=_cerr_sci_row,
                                        telluric=_tel_sci)

        flux_near_recon = np.asarray(comps_near["total"], dtype=np.float64) / FACTOR
        flux_far_recon = np.asarray(comps_far["total"], dtype=np.float64) / FACTOR
        flux_sci_pred = np.asarray(comps_sci["total"], dtype=np.float64) / FACTOR

        # nanmean so isolated NaN pixels (~1 pixel/row on ~40% of every10) don't
        # poison the pRMSE; a whole-row veto lives with the chi2/field filter above.
        _o_near_rmse = float(np.sqrt(np.nanmean((flux_near_recon - flux_near_true) ** 2)))
        _o_far_rmse = float(np.sqrt(np.nanmean((flux_far_recon - flux_far_true) ** 2)))

        sci_resid = flux_sci_pred - flux_sci_true
        _o_sci_rmse = float(np.sqrt(np.nanmean(sci_resid ** 2)))
        _o_corr = None
        if RESIDUAL_CORRECTION:
            _solar = lambda comps: (np.asarray(comps.get("moon", 0.0), dtype=np.float64)
                                    + np.asarray(comps.get("zodi", 0.0), dtype=np.float64)) / FACTOR
            _arms_rc = [
                _SkyArm(flux_near_true, flux_near_recon, _solar(comps_near),
                        _arm_var(flux_near_true, wave_row, [_nf_rc["near"][i]], sens=_sens_rc)),
                _SkyArm(flux_far_true, flux_far_recon, _solar(comps_far),
                        _arm_var(flux_far_true, wave_row, [_nf_rc["far"][i]], sens=_sens_rc)),
            ]
            _o_corr = _sky_arm_corr(
                wave_row, flux_sci_pred, _solar(comps_sci), _arms_rc,
                max_wavelength=RESIDUAL_CORRECTION_MAX_A,
                smoothing=RESIDUAL_SMOOTHING_A).astype(np.float32)
        _o_line_scales = None
        _o_corr_arm = _o_corr          # sky-arm part only, before any line scaling
        if LINE_SCALING:
            _tpl = _row_line_templates(coef_sci_pred[i], _lsf_state_sci, _o2_vec_sci, _tel_sci)
            if _tpl is not None:
                _sky_now = flux_sci_pred + (0.0 if _o_corr is None else _o_corr.astype(np.float64))
                _lc, _li = _line_scale(
                    wave_row, flux_sci_true, _sky_now, _tpl,
                    variance=_arm_var(flux_sci_true, wave_row, None, sens=_sens_rc),
                    # The decomposition's science-line mask plus the red nebular
                    # lines it leaves out, slid by this row's Halpha velocity:
                    # unmasked, a weak line on an OH line leaks into that band.
                    mask=_sci_mask_ls(EVAL_INPUT, wave_row, flux_sci_true, flux_near_true,
                                      extra_lines=_NEB_RED, widen_if_unmeasured_km_s=_NEB_V),
                    highpass_A=LINE_SCALING_HIGHPASS_A, prior_sigma=LINE_SCALING_PRIOR,
                    return_info=True)
                _o_corr = (_lc if _o_corr is None else _o_corr + _lc).astype(np.float32)
                _o_line_scales = dict(_li["scales"], accepted=bool(_li["accepted"]))
        # float32 from here on: at 2900 rows the eight per-row stacks
        # are 2.3 GB in float64 and half that in float32, and they feed
        # medians, percentiles and plots -- 7 significant digits is far
        # more than any of those resolve.  The arithmetic above is all
        # float64; only the stored copy is narrowed.
        _o_resid = np.asarray(sci_resid, dtype=np.float32)
        _o_wave = np.asarray(wave_row, dtype=np.float32)
        _o_obs = np.asarray(flux_sci_true, dtype=np.float32)

        # Reconstruct the sci-arm spectrum from the FITTED sci coefficients so
        # per-component residuals (pred - recon(sci_true)) can be separated in
        # the multi-panel residual plot below (matches cell 26 fig_deltas).
        comps_sci_true_batch = _fast_reconstruct(
            coef_sci_sel[i], _lsf_state_sci, _o2_vec_sci, _lsf_sigma_fallback,
            coef_err=None, telluric=_tel_sci,
        )
        _dmoon    = (np.asarray(comps_sci["moon"], dtype=np.float64)
                     - np.asarray(comps_sci_true_batch["moon"], dtype=np.float64)) / FACTOR
        # comps_*.get('zodi') is missing on pre-split rows, so fall back to zero for backward compatibility.
        _zodi_pred_arr = np.asarray(comps_sci.get("zodi", 0.0), dtype=np.float64)
        _zodi_true_arr = np.asarray(comps_sci_true_batch.get("zodi", 0.0), dtype=np.float64)
        _dzodi    = (_zodi_pred_arr - _zodi_true_arr) / FACTOR
        _ddiffuse = (np.asarray(comps_sci["diffuse"], dtype=np.float64)
                     - np.asarray(comps_sci_true_batch["diffuse"], dtype=np.float64)) / FACTOR
        _dlines   = (_lines_sum(comps_sci) - _lines_sum(comps_sci_true_batch)) / FACTOR
        # Per-component handoff for the atlas, in the RECONSTRUCTION's own
        # units -- NOT divided by FACTOR, because that is the convention
        # wavelength_residual_atlas works in.  (The batch statistics below
        # divide; these deliberately do not.)
        if _atlas_keep[i]:
            _gp = _group_components(comps_sci)
            _gt = _group_components(comps_sci_true_batch)
            # The DIFFERENCE is taken in float64 and only then narrowed.  Storing
            # float32 pred and true and subtracting later cancels badly on rows
            # where a family is predicted almost exactly: the residual is then
            # ~1e-4 of the flux and float32 rounding of each operand is a
            # visible fraction of it (measured: 2e-4 relative on a diffuse row).
            _o_cdelta = {_g: (_gp[_g] - _gt[_g]).astype(np.float32)
                         for _g in _ATLAS_COMPONENTS}
            _o_ctrue = {_g: _gt[_g].astype(np.float32) for _g in _ATLAS_COMPONENTS}
        else:
            _o_cdelta = _o_ctrue = None

        # Same reconstruction path, same LSF state, same O2 vector as the
        # prediction above -- only the coefficients differ (fitted, not
        # predicted) -- so the two chi2 distributions below differ ONLY by the
        # coefficient error and are directly comparable.
        _o_selfres = (
            np.asarray(comps_sci_true_batch["total"], dtype=np.float64) / FACTOR
            - flux_sci_true).astype(np.float32)

        # Pixel-space WRMSE: prefer the FITS-side sigma if present, else use
        # the propagator output from _fast_reconstruct.  All three sources
        # deliver the same LSF-aware sigma; the fallback of last resort is
        # the median-floor path inside pixel_wrmse_per_row.
        _sig_near_row = (_pix_sigma_near_all[i] * FACTOR
                         if _pix_sigma_near_all is not None
                         else comps_near.get("sigma_total"))
        _sig_far_row  = (_pix_sigma_far_all[i] * FACTOR
                         if _pix_sigma_far_all is not None
                         else comps_far.get("sigma_total"))
        _sig_sci_row  = (_pix_sigma_sci_all[i] * FACTOR
                         if _pix_sigma_sci_all is not None
                         else comps_sci.get("sigma_total"))
        # comps_*['sigma_total'] comes out in native units; the flux_* arrays
        # here are already divided by FACTOR, so the sigma from propagation
        # must be divided by FACTOR too for a scale match.
        if _pix_sigma_near_all is None and _sig_near_row is not None:
            _sig_near_row = np.asarray(_sig_near_row) / FACTOR
        if _pix_sigma_far_all is None and _sig_far_row is not None:
            _sig_far_row = np.asarray(_sig_far_row) / FACTOR
        if _pix_sigma_sci_all is None and _sig_sci_row is not None:
            _sig_sci_row = np.asarray(_sig_sci_row) / FACTOR

        _o_near_wrmse = float(pixel_wrmse_per_row(
            flux_near_recon, flux_near_true, _sig_near_row)[0])
        _o_far_wrmse  = float(pixel_wrmse_per_row(
            flux_far_recon,  flux_far_true,  _sig_far_row)[0])
        _o_sci_wrmse  = float(pixel_wrmse_per_row(
            flux_sci_pred,   flux_sci_true,  _sig_sci_row)[0])

        return (_o_near_rmse, _o_far_rmse, _o_sci_rmse,
                _o_near_wrmse, _o_far_wrmse, _o_sci_wrmse,
                _o_resid, _o_wave, _o_obs, _o_selfres,
                _dmoon.astype(np.float32), _dzodi.astype(np.float32),
                _ddiffuse.astype(np.float32), _dlines.astype(np.float32),
                _o_cdelta, _o_ctrue, _o_corr, _o_line_scales, _o_corr_arm)

    _t_loop0 = _time.perf_counter()
    _atlas_cpred, _atlas_ctrue = [], []
    _rows_out = _cell_parallel.map_indexed(
        _row_work, range(n_use), n_workers=n_workers, label="recon")
    for i, _o in enumerate(_rows_out):
        (near_rmse[i], far_rmse[i], sci_rmse[i],
         near_wrmse[i], far_wrmse[i], sci_wrmse[i]) = _o[:6]
        sci_resid_rows.append(_o[6])
        sci_wave_rows.append(_o[7])
        sci_obs_rows.append(_o[8])
        sci_selfres_rows.append(_o[9])
        sci_moon_resid_rows.append(_o[10])
        sci_zodi_resid_rows.append(_o[11])
        sci_diffuse_resid_rows.append(_o[12])
        sci_lines_resid_rows.append(_o[13])
        sci_corr_rows.append(_o[16])
        sci_line_scale_rows.append(_o[17])
        sci_corr_arm_rows.append(_o[18])
        if _o[14] is not None:
            _atlas_cpred.append(_o[14])
            _atlas_ctrue.append(_o[15])

    print(f"  recon loop:  {_time.perf_counter() - _t_loop0:.2f} s "
          f"({n_use} rows x 3 arms = {3 * n_use} reconstructions with hoisted basis)")

    # --- Handoff for wavelength_residual_atlas ------------------------------
    # Stacked in the atlas's own layout and units so it can drop them straight
    # into _resid / _truth / _resid_comp / _truth_comp and skip reconstructing
    # its own sample.  Reconstructing twice was not just wasted time: the atlas
    # drew from the ungated every10 pool, so the two cells described DIFFERENT
    # row populations and their residuals were never comparable.
    batch_recon_for_atlas = None
    if _atlas_cpred:
        _ak = np.flatnonzero(_atlas_keep)
        # _atlas_cpred holds float64-derived DELTAS (pred - true), see _row_work.
        _comp_delta = {_g: np.vstack([_c[_g] for _c in _atlas_cpred])
                       for _g in _ATLAS_COMPONENTS}
        _comp_true = {_g: np.vstack([_c[_g] for _c in _atlas_ctrue])
                      for _g in _ATLAS_COMPONENTS}
        _tot_delta = sum(_c.astype(np.float64) for _c in _comp_delta.values())
        _tot_true = sum(_c.astype(np.float64) for _c in _comp_true.values())
        batch_recon_for_atlas = {
            # Positions into filtered_triplet, and the corpus FITS rows.
            'sel_pos': sel_pos[_ak],
            'sel_rows': sel_rows[_ak],
            'wave': np.asarray(wave_arr, dtype=np.float64),
            'resid': _tot_delta.astype(np.float32),
            'truth': _tot_true.astype(np.float32),
            'resid_comp': dict(_comp_delta),
            'truth_comp': {_g: _comp_true[_g] for _g in _ATLAS_COMPONENTS},
            'components': tuple(_ATLAS_COMPONENTS),
            'split': _EVAL_SPLIT,
            'source_input': EVAL_INPUT,
        }
        # Self-check on a few rows: the handoff must reproduce the
        # per-component residuals this cell keeps for its OWN figure, which
        # are computed on a separate line and in FACTOR-DIVIDED units.  The
        # two conventions are the classic way a handoff like this goes quietly
        # wrong, so check rather than trust -- it costs microseconds.
        _chk = np.unique(np.linspace(0, _ak.size - 1, min(3, _ak.size)).astype(int))
        for _cname, _clist in (('moon', sci_moon_resid_rows),
                               ('zodi', sci_zodi_resid_rows),
                               ('continuum (diffuse)', sci_diffuse_resid_rows)):
            for _c in _chk:
                _mine = np.asarray(_clist[int(_ak[_c])], dtype=np.float64) * FACTOR
                _theirs = np.asarray(batch_recon_for_atlas['resid_comp'][_cname][_c],
                                     dtype=np.float64)
                _scale = float(np.sqrt(np.nanmean(_theirs ** 2))) or 1.0
                _dev = float(np.nanmax(np.abs(_mine - _theirs))) / _scale
                if not _dev < 1e-4:
                    raise RuntimeError(
                        f"atlas handoff disagrees with this cell's own "
                        f"{_cname} residual on row {int(_ak[_c])}: max "
                        f"relative deviation {_dev:.3g}. The handoff is stored "
                        f"in reconstruction units and the figure arrays in "
                        f"FACTOR-divided units -- check that convention first.")
        print(f"  atlas handoff self-check: components agree with this cell's "
              f"own residuals to <1e-4 relative on {_chk.size} probed row(s)")
        print(f"  atlas handoff ready: {_ak.size} rows x "
              f"{len(_ATLAS_COMPONENTS)} components "
              f"({(sum(a.nbytes for a in _comp_delta.values()) * 2 + _tot_true.nbytes) / 1e6:.0f} MB)")
        del _comp_delta, _comp_true, _tot_delta, _tot_true, _atlas_cpred, _atlas_ctrue


    def _rmse_stats(arr):
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p05": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }
        return {
            "count": int(x.size),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p05": float(np.percentile(x, 5.0)),
            "p95": float(np.percentile(x, 95.0)),
            "max": float(np.max(x)),
        }

    summary_df = pd.DataFrame(
        [
            {"series": "near_self_recon_pRMSE",  **_rmse_stats(near_rmse)},
            {"series": "near_self_recon_pWRMSE", **_rmse_stats(near_wrmse)},
            {"series": "far_self_recon_pRMSE",   **_rmse_stats(far_rmse)},
            {"series": "far_self_recon_pWRMSE",  **_rmse_stats(far_wrmse)},
            {"series": "sci_pred_vs_true_pRMSE", **_rmse_stats(sci_rmse)},
            {"series": "sci_pred_vs_true_pWRMSE",**_rmse_stats(sci_wrmse)},
        ]
    )

    summary_disp_df = summary_df.copy()
    for c in ["mean", "median", "std", "min", "p05", "p95", "max"]:
        summary_disp_df[c] = summary_disp_df[c] * FACTOR

    print(f"Per-row pRMSE / pWRMSE on {n_use} {_EVAL_SPLIT} spectra from "
          f"every10 inputs"
          + ("" if n_sample is None else f" (phase-stratified draw, seed={rng_seed})"))
    print("Per-row pixel-space pRMSE / pWRMSE stats in physical flux units:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print("")
    print(f"Per-row pRMSE / pWRMSE stats in display units (x{FACTOR:.3g}):")
    print(summary_disp_df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    # Multi-panel residual figure (2026-08-19): row 1 shows the sky-subtracted
    # sci spectrum (observed - pred, since 2026-09-25); rows 2-5 show per-component
    # residuals (recon(sci_true) - pred, since 2026-09-29, the same sense as row
    # 1: truth minus prediction) so a broadband deficit that lives
    # entirely in one component (e.g. moon spline) shows up separately from
    # a line-emission miss (mesospheric / atomic / ionospheric / O2).  The
    # legend is off; per-line hover shows row_idx + expnum from META.
    from plotly.subplots import make_subplots as _make_subplots_resid

    sci_resid_arr = np.vstack(sci_resid_rows) * FACTOR
    # Panel 1 shows the SKY-SUBTRACTED spectrum, observed - predicted, i.e.
    # what a user of the prediction gets; `sci_resid_arr` itself keeps the
    # pred - observed sign it is stored and returned with.
    sci_skysub_arr = -sci_resid_arr
    _ANY_CORR = (RESIDUAL_CORRECTION or LINE_SCALING) and bool(sci_corr_rows) \
        and all(c is not None for c in sci_corr_rows)
    _CORR_LABEL = " + ".join(
        n for n, on in (("sky-arm correction", RESIDUAL_CORRECTION), ("line scaling", LINE_SCALING)) if on)
    if _ANY_CORR:
        sci_skysub_arr = sci_skysub_arr - np.vstack(sci_corr_rows) * FACTOR
    # Panels 2-5 likewise show decomposition - predicted, the same sense as
    # panel 1. The per-row lists keep pred - recon(true), which is what the
    # atlas handoff self-check above compares against, so only these display
    # arrays are negated.
    sci_moon_arr = -np.vstack(sci_moon_resid_rows) * FACTOR
    sci_zodi_arr = -np.vstack(sci_zodi_resid_rows) * FACTOR
    sci_diffuse_arr = -np.vstack(sci_diffuse_resid_rows) * FACTOR
    sci_lines_arr = -np.vstack(sci_lines_resid_rows) * FACTOR
    # The per-component lists are dead once stacked, and each is n_use x 12401
    # float64 -- 50 MB apiece at n_use=500, 200 MB across the four, held for
    # the whole rest of the cell for nothing.  (sci_resid_rows, sci_obs_rows,
    # sci_wave_rows and sci_selfres_rows are NOT freed here: the chi2 block and
    # the ragged-grid branch still read them, and the chi2 block frees the
    # self-residual list itself once it has stacked it.)
    del sci_moon_resid_rows, sci_zodi_resid_rows
    del sci_diffuse_resid_rows, sci_lines_resid_rows
    wave_ref = sci_wave_rows[0]
    same_grid = all(
        (w.shape == wave_ref.shape) and np.allclose(w, wave_ref, rtol=0.0, atol=1e-8)
        for w in sci_wave_rows[1:]
    )

    # Exposure assumed for every photon-noise quantity in this cell (the
    # sigma band below and the chi2 panels further down).  META carries no
    # exposure time, so 900 s -- the LVM standard science exposure and the
    # trainer's `flux_exptime_s` default -- is assumed.  A wrong value scales
    # the noise linearly, so keep it equal to the trainer's or neither the
    # band nor the chi2 is comparable to the loss.
    CHI2_EXPTIME_S = 900.0

    # ---- +/-1 sigma SINGLE-FIBRE photon noise, as a background band -------
    # The reference every residual panel should be read against: a residual
    # inside this band is consistent with the noise of ONE fibre, which is the
    # bar the decomposition is held to elsewhere (the chi2 panel below uses
    # the same convention and the same floor, so the two are comparable).
    #
    # It is the SINGLE-FIBRE level deliberately, not the stacked level: these
    # are median stacks of tens to hundreds of fibres, so the true noise is
    # well below this and the band is a generous reference rather than a
    # detection threshold. The stacked band would sit ~sqrt(N_eff) lower.
    #
    # Per-pixel sigma varies row to row through the observed flux, so the band
    # is the MEDIAN over the plotted rows -- a typical row, not an envelope.
    #
    # Scale, measured on the 1082 filtered test rows of
    # gaia-stars-mask-telluric-chi2 (median sigma = 0.116 in FACTOR units).
    # The band is COMPARABLE TO OR WIDER THAN the residual envelope, so it
    # reads as a wide grey region that most of the strokes sit inside:
    #
    #   row 1   observed - pred          band / p68|r| = 0.80
    #   rows 2-5  recon(sci coef) - pred band / p68|r| = 1.52
    #
    # That is the message, not a defect: on a typical pixel the transfer
    # error is already below the noise of a single fibre, and rows 2-5 are
    # further inside it than row 1 because row 1 also carries the
    # decomposition's own residual (band / p68 = 1.11 for that alone).
    # Judge excursions OUT of the band, not the bulk inside it.
    #
    # Do NOT calibrate expectations against the RMS envelope drawn on the
    # same panel: RMS across rows is carried by a few bad rows and runs far
    # above both, which is the envelope's job and not this band's.
    _sig1_band = None          # per-pixel, FACTOR units, or None
    _sig1_scalar = None        # median over pixels, for the histogram panels
    try:
        from mlp_predictor.noise import (
            load_absolute_sensitivity as _sig_sens,
            photon_variance_absolute as _sig_var,
            floor_variance as _sig_floor)
    except Exception as _exc_sig:
        print(f"  [sigma band] mlp_predictor.noise unavailable "
              f"({type(_exc_sig).__name__}); the panels get no noise band.")
    else:
        if same_grid and sci_obs_rows:
            _s_obs = np.vstack(sci_obs_rows)
            _s_sens = np.asarray(_sig_sens(wave_ref), dtype=np.float64)
            _s_var = _sig_floor(_sig_var(
                _s_obs, _s_sens, exptime=CHI2_EXPTIME_S,
                dwave=float(np.median(np.diff(wave_ref))), n_fibres=None))
            _sig1_band = (np.median(np.sqrt(_s_var), axis=0) * FACTOR)
            _sig1_scalar = float(np.median(_sig1_band))
            print(f"  [sigma band] +/-1 sigma single-fibre noise: median "
                  f"{_sig1_scalar:.4g} (FACTOR units), "
                  f"p5-p95 {np.percentile(_sig1_band, 5):.4g}-"
                  f"{np.percentile(_sig1_band, 95):.4g} across the grid.")
        elif sci_obs_rows:
            print("  [sigma band] rows are not on a common wavelength grid; "
                  "the band is omitted from the spectrum panels.")

    # With the sky-arm correction on, panel 1 shows the CORRECTED sky-subtracted
    # spectra and panel 2 the same rows UNCORRECTED, on a shared y-axis, so
    # the two can be compared stroke for stroke.
    _show_uncorr = _ANY_CORR
    # Panel 3, with both corrections on: the sky-arm correction alone, i.e.
    # what the line scaling still has to fix.
    _show_arm_only = (_show_uncorr and RESIDUAL_CORRECTION and LINE_SCALING
                      and all(c is not None for c in sci_corr_arm_rows))
    _n_extra = int(_show_uncorr) + int(_show_arm_only)
    _n_resid_rows = 5 + _n_extra
    _resid_titles = [
        (f"SCI sky-subtracted, WITH {_CORR_LABEL}: observed - pred - corr (n={n_use})"
         if _show_uncorr else f"SCI sky-subtracted: observed - pred (n={n_use})"),
        "median residual / row (±1,2,3σ)",
    ]
    if _show_uncorr:
        _resid_titles += [f"SCI sky-subtracted, NO correction: observed - pred (n={n_use})", ""]
    if _show_arm_only:
        _resid_titles += [f"SCI sky-subtracted, sky-arm correction only (NO line scaling) (n={n_use})", ""]
    _resid_titles += [
        "Moon component: recon(sci coef) - pred", "",
        "Zodi component: recon(sci coef) - pred", "",
        "Diffuse continuum (HO2 + FeO + O2ac): recon(sci coef) - pred", "",
        "Lines (OH + atom + ORC + O2): recon(sci coef) - pred", "",
    ]
    fig_resid = _make_subplots_resid(
        rows=_n_resid_rows, cols=2,
        shared_xaxes=False,
        column_widths=[0.82, 0.18],
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
        subplot_titles=tuple(_resid_titles),
    )
    # Share the wavelength x-axis across the four spectrum panels only; the
    # right-column histograms keep independent x-axes because per-component
    # median residuals live on very different scales.
    for _r in range(2, _n_resid_rows + 1):
        fig_resid.update_xaxes(matches="x", row=_r, col=1)
    if _show_uncorr:
        # Same y-range for corrected and uncorrected: the comparison is the point.
        for _r in range(2, 2 + _n_extra):
            fig_resid.update_yaxes(matches="y", row=_r, col=1)

    def _expnum_str(i):
        if _expnum_sel is None:
            return ""
        try:
            _e = int(_expnum_sel[i])
        except (IndexError, ValueError, TypeError):
            return ""
        return f" | expnum {_e}"

    _wave_ref32 = np.asarray(wave_ref, dtype=np.float32)
    _panels = [("total",   sci_skysub_arr)]
    if _show_uncorr:
        _panels.append(("total_uncorrected", -sci_resid_arr))
    if _show_arm_only:
        _panels.append(("total_arm_only", -sci_resid_arr - np.vstack(sci_corr_arm_rows) * FACTOR))
    _panels += [
        ("moon",    sci_moon_arr),
        ("zodi",    sci_zodi_arr),
        ("diffuse", sci_diffuse_arr),
        ("lines",   sci_lines_arr),
    ]
    # Stable per-row colors so row i has the same color in every panel.
    from plotly.colors import sample_colorscale as _sample_colorscale
    _row_colors = _sample_colorscale(
        "turbo",
        [j / max(n_use - 1, 1) for j in range(n_use)],
    )
    # Which rows get STROKED.  Everything below that aggregates -- the RMS
    # envelope, the right-column histograms, every printed statistic -- still
    # uses all n_use rows; see MAX_RESID_LINES for why the drawing is capped.
    # np.linspace over the row order is deliberate rather than random: the
    # sample is phase-stratified and sel_pos is sorted, so an even stride
    # spans the lunation range instead of clumping in whichever bin a random
    # draw favoured.
    if n_use <= MAX_RESID_LINES:
        _line_idx = np.arange(n_use)
    else:
        _line_idx = np.unique(
            np.linspace(0, n_use - 1, MAX_RESID_LINES).round().astype(int))
    _n_lines = int(_line_idx.size)
    if _n_lines < n_use:
        print(f"  residual figure: stroking {_n_lines} of {n_use} rows "
              f"(MAX_RESID_LINES); the RMS band, the histograms and every "
              f"number use all {n_use}.")
    # (percent enclosed, color, dash) for the per-row median-residual histograms.
    _sigma_specs = [
        (68.27, "rgba(200, 60, 60, 0.95)", "solid"),
        (95.45, "rgba(230,140, 40, 0.85)", "dash"),
        (99.73, "rgba( 90, 90, 90, 0.75)", "dot"),
    ]
    for _row_i, (_pname, _arr) in enumerate(_panels, start=1):
        # Added BEFORE the strokes so plotly draws it underneath: it is a
        # reference, not data.
        if _sig1_band is not None:
            fig_resid.add_trace(
                go.Scatter(
                    x=wave_ref, y=-_sig1_band,
                    mode="lines", line=dict(width=0),
                    hoverinfo="skip", showlegend=False,
                ),
                row=_row_i, col=1,
            )
            fig_resid.add_trace(
                go.Scatter(
                    x=wave_ref, y=_sig1_band,
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(130,130,130,0.22)",
                    name="±1σ Single-fiber noise",
                    hovertemplate=("λ=%{x:.1f} Å<br>±1σ Noise="
                                   "%{y:.4g}<extra></extra>"),
                    showlegend=bool(_row_i == 1),
                ),
                row=_row_i, col=1,
            )
        if same_grid:
            for i in _line_idx:
                i = int(i)
                _rid = int(sel_rows[i])
                _hover = (
                    f"{_row_label[i]}<br>"
                    f"λ=%{{x:.1f}} Å<br>"
                    f"Δ_{_pname}=%{{y:.4g}}"
                    "<extra></extra>"
                )
                # float32 on the wire: these are display strokes, and the
                # payload is the binding constraint here, not precision.
                fig_resid.add_trace(
                    go.Scattergl(
                        x=_wave_ref32, y=_arr[i].astype(np.float32),
                        mode="lines",
                        line=dict(width=0.8, color=_row_colors[i]),
                        opacity=0.7,
                        hovertemplate=_hover,
                        showlegend=False,
                    ),
                    row=_row_i, col=1,
                )
            _rms_band = np.sqrt(np.mean(_arr ** 2, axis=0))
            fig_resid.add_trace(
                go.Scatter(
                    x=wave_ref, y=-_rms_band,
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.6)", width=1.0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=_row_i, col=1,
            )
            fig_resid.add_trace(
                go.Scatter(
                    x=wave_ref, y=_rms_band,
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.6)", width=1.0),
                    fill="tonexty",
                    fillcolor="rgba(120,120,120,0.15)",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=_row_i, col=1,
            )
        else:
            for i in _line_idx:
                i = int(i)
                _rid = int(sel_rows[i])
                _hover = (
                    f"{_row_label[i]}<br>"
                    f"λ=%{{x:.1f}} Å<br>"
                    f"Δ_{_pname}=%{{y:.4g}}"
                    "<extra></extra>"
                )
                fig_resid.add_trace(
                    go.Scattergl(
                        x=sci_wave_rows[i].astype(np.float32),
                        y=_arr[i].astype(np.float32),
                        mode="lines",
                        line=dict(width=0.8, color=_row_colors[i]),
                        opacity=0.7,
                        hovertemplate=_hover,
                        showlegend=False,
                    ),
                    row=_row_i, col=1,
                )
            _global_rms = float(np.sqrt(np.mean(_arr ** 2)))
            fig_resid.add_hrect(
                y0=-_global_rms, y1=_global_rms,
                fillcolor="rgba(120,120,120,0.15)",
                line_width=0, layer="above",
                row=_row_i, col=1,
            )
        fig_resid.add_hline(
            y=0.0, line=dict(color="rgba(0,0,0,0.5)", width=0.8, dash="dash"),
            row=_row_i, col=1,
        )

        # Right-column histogram: median residual per row (median over pixels)
        # with empirical two-tailed 1σ / 2σ / 3σ percentile bars.
        # The "lines" panel (OH + atom + ORC + O2) is LSF-convolved, so it
        # never hits a literal 0.0 away from a line core -- it decays
        # continuously into a long, tiny tail instead (measured: only 0.85%
        # of pixels are bit-exact zero, even restricted to the OH-free blue
        # half of the spectrum). Those tail pixels carry no real line signal,
        # but there are so many of them (~99% of the 12401-pixel grid) that
        # left in, they swamp the median and pull it toward 0 regardless of
        # how the actual line residuals behave. Masked here (this panel
        # only) below a fixed 1e-3 (FACTOR units) magnitude, so a pixel only
        # counts as "line-bearing" if its residual clears that floor; median
        # then taken over those alone.
        _LINES_HIST_FLOOR = 1e-2  # FACTOR units; see comment above
        _hist_arr = _arr
        if _pname == "lines":
            _hist_arr = np.where(np.abs(_arr) <= _LINES_HIST_FLOOR, np.nan, _arr)
        _med_per_row = np.nanmedian(_hist_arr, axis=1)
        _med_per_row = _med_per_row[np.isfinite(_med_per_row)]
        if _med_per_row.size > 0:
            _nb_hist = int(max(10, min(40, np.sqrt(_med_per_row.size) * 2.0)))
            fig_resid.add_trace(
                go.Histogram(
                    x=_med_per_row,
                    nbinsx=_nb_hist,
                    marker=dict(
                        color="rgba(70,110,170,0.65)",
                        line=dict(color="rgba(35, 60,120,1.0)", width=0.4),
                    ),
                    hovertemplate=("median Δ_" + _pname
                                   + "=%{x:.4g}<br>count=%{y}<extra></extra>"),
                    showlegend=False,
                ),
                row=_row_i, col=2,
            )
            # Same reference on the histogram: a row whose MEDIAN residual
            # falls inside this band is, on a typical pixel, within one
            # fibre's noise.  Most rows land inside, so it is the ones
            # OUTSIDE that the panel is for -- a median over ~12k pixels
            # would beat single-fibre noise by ~sqrt(n) if the errors were
            # independent, so being inside is a weak statement and being
            # outside is a strong one.
            if _sig1_scalar is not None:
                fig_resid.add_vrect(
                    x0=-_sig1_scalar, x1=_sig1_scalar,
                    fillcolor="rgba(130,130,130,0.22)",
                    line_width=0, layer="below",
                    row=_row_i, col=2,
                )
            _row_median = float(np.median(_med_per_row))
            fig_resid.add_vline(
                x=_row_median,
                line=dict(color="rgba(0,0,0,0.7)", width=0.9, dash="dashdot"),
                row=_row_i, col=2,
            )
            for _pct, _color, _dash in _sigma_specs:
                _q_lo = (100.0 - _pct) / 2.0
                _q_hi = 100.0 - _q_lo
                _v_lo = float(np.percentile(_med_per_row, _q_lo))
                _v_hi = float(np.percentile(_med_per_row, _q_hi))
                fig_resid.add_vline(
                    x=_v_lo,
                    line=dict(color=_color, width=1.0, dash=_dash),
                    row=_row_i, col=2,
                )
                fig_resid.add_vline(
                    x=_v_hi,
                    line=dict(color=_color, width=1.0, dash=_dash),
                    row=_row_i, col=2,
                )

    _o = _n_extra                      # row offset of the component panels
    fig_resid.update_xaxes(title_text="Wavelength [Å]", row=_n_resid_rows, col=1)
    fig_resid.update_xaxes(title_text="Median residual", row=_n_resid_rows, col=2)
    fig_resid.update_yaxes(title_text=("Obs - pred - corr" if _show_uncorr
                                       else "Observed - Predicted sky"), row=1, col=1)
    if _show_uncorr:
        fig_resid.update_yaxes(title_text="Obs - pred (uncorrected)", row=2, col=1)
    if _show_arm_only:
        fig_resid.update_yaxes(title_text="Obs - pred - arm corr", row=3, col=1)
    fig_resid.update_yaxes(title_text="Decomposition - predicted moon",    row=2 + _o, col=1)
    fig_resid.update_yaxes(title_text="Decomposition - predicted zodi",    row=3 + _o, col=1)
    fig_resid.update_yaxes(title_text="Decomposition - predicted diffuse", row=4 + _o, col=1)
    fig_resid.update_yaxes(title_text="Decomposition - predicted lines",   row=5 + _o, col=1)
    fig_resid.update_layout(
        template="plotly_white",
        title=(f"SCI + per-component residuals (n={n_use} spectra"
               + (f", {_n_lines} stroked" if _n_lines < n_use else "")
               + f", display units x{FACTOR:.3g}). Hover any line for row_idx "
               f"+ expnum. Gray band = ± RMS(λ) across ALL {n_use} rows; "
               f"right-column histograms show the per-row median residual "
               f"across ALL {n_use} rows, with empirical ±1σ/2σ/3σ "
               f"percentiles."),
        height=1500 + 300 * _n_extra,
        margin=dict(l=80, r=20, t=110, b=60),
        showlegend=False,
        bargap=0.05,
    )
    fig_resid.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.02,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=11),
        text=("<span style='color:rgb(200,60,60)'>─ 1σ</span>"
              " &nbsp; <span style='color:rgb(230,140,40)'>-- 2σ</span>"
              " &nbsp; <span style='color:rgb(90,90,90)'>·· 3σ</span>"
              " &nbsp; <span style='color:rgb(50,50,50)'>-·- median</span>"),
    )
    fig_resid.show()

    # ---- Photon chi2 of the reconstruction (2026-09-09) ---------------------
    # How badly is each row reconstructed, measured against the PHOTON noise
    # model that weights the flux-space training loss?  Uses
    # `photon_pixel_variance`, the same call `photon_pixel_weight` makes, so
    # this cannot drift from the loss.  Deriving sqrt(flux*sens) here instead
    # drops the 5%-of-row-median floor, and 10.9% of observed sci pixels are
    # non-positive, so those pixels would take a near-zero sigma and an
    # unbounded chi2.
    #
    # THE SCALE IS NOW ABSOLUTE (2026-09-09).  `sens_percentiles-{arm}.csv`
    # carries the sensitivity in [erg/s/cm^2/A] per [e-/s/A] -- the normalised
    # `mean-sens` curves cannot, since avgsens divides each arm by its own
    # weighted mean -- so with the exposure time, the dispersion and the
    # per-row fibre count the Poisson variance is fully determined:
    #     var(flux) = flux * sens / (exptime * dwave * N_eff)
    # and a reduced chi2 of 1 means the reconstruction is at the photon limit.
    # Nothing is normalised away here, so the ABSOLUTE level is now readable:
    # values >> 1 are real reconstruction error above the noise.
    #
    # SINGLE FIBRE, deliberately (2026-09-10).  The corpus rows are median
    # stacks of a median 536 sci fibres, but the sky model is going to be
    # subtracted from ONE fibre at a time, and it is that subtraction the
    # error budget has to survive.  So N_eff is left at 1 here: sigma is the
    # noise of a single 900 s fibre observing this sky, which is sqrt(536 *
    # 2/pi) ~ 18x larger than the stack's, and the reduced chi2 is
    # correspondingly ~340x smaller than the stacked-noise version.
    #
    # Read it as: chi2 < 1 means the reconstruction error is buried under the
    # shot noise of the single fibre it will be subtracted from, which is the
    # only bar that matters downstream.  Set CHI2_SINGLE_FIBRE = False to get
    # the stacked-noise view instead -- that is the right comparison against
    # the DECOMPOSITION's own residual, which was fitted to the stack.
    CHI2_SINGLE_FIBRE = True
    #
    # CHI2_EXPTIME_S is set once, above, where the sigma band first needs it.
    #
    # Blue cut for the second panel.  Redward of this the OH forest dominates
    # the pixel budget, so a full-band chi2 is largely a statement about OH
    # residuals and hides how the continuum -- moon, zodi, diffuse, which is
    # what this predictor is actually for -- is doing.  4800 of 12401 pixels
    # survive the cut.
    #
    # Do NOT read a lower blue chi2 as a better blue reconstruction.  The sky
    # is fainter blueward of 6000 A, so the single-fibre sigma/flux is LARGER
    # there and the same fractional error buys a smaller chi2.  Measured on
    # the decomposition's own fit (100 rows, single-fibre noise): full band
    # median 4.0, blue 1.3.  The panels answer "is this row's error above the
    # noise", separately in each region; they are not a like-for-like ranking
    # of the two regions.
    CHI2_BLUE_MAX_A = 6000.0
    try:
        from mlp_predictor.noise import (load_absolute_sensitivity as _load_sens_abs,
                                         photon_variance_absolute as _phot_var_abs,
                                         floor_variance as _floor_var)
        _chi2_ok = True
    except Exception as _exc_chi2:
        print(f"  [chi2] mlp_predictor.noise unavailable ({type(_exc_chi2).__name__}); "
              f"skipping the photon chi2 panel.")
        _chi2_ok = False
    if _chi2_ok and sci_obs_rows and not same_grid:
        print('  [chi2] rows are not on a common wavelength grid; '
              'skipping the photon chi2 panel.')
        _chi2_ok = False
    _chi2_ph = None
    _chi2_blue = None
    _chi2_self = None
    _chi2_self_blue = None
    _chi2_corr = None
    _chi2_corr_blue = None
    _chi2_red = None
    _chi2_corr_red = None
    CHI2_RED_MIN_A = 6000.0
    if _chi2_ok and sci_obs_rows:
        _obs_a = np.vstack(sci_obs_rows)
        _res_a = np.vstack(sci_resid_rows)
        _sens = np.asarray(_load_sens_abs(wave_ref), dtype=np.float64)
        # NATIVE dispersion: these rows are reconstructed on the full grid, not
        # the strided one the flux loss subsamples, so no stride factor here.
        _dwave_c2 = float(np.median(np.diff(wave_ref)))
        # Per-row fibre count, same column preference as the trainer.  Read
        # even in single-fibre mode: it is not used for the variance there,
        # but it is what says how far below the stack this sigma sits, and a
        # reader needs that number to relate the panel to the decomposition.
        _nfib = None
        _meta_c2 = globals().get('_meta_ev')   # full-corpus META; indexed by sel_rows below
        if _meta_c2 is not None:
            _cu = {c.lower(): c for c in _meta_c2.colnames}
            for _c in ('fibers_sci_used', 'fibers_sci'):
                if _c in _cu:
                    _nfib = np.asarray(_meta_c2[_cu[_c]],
                                       dtype=np.float64)[np.asarray(sel_rows, dtype=int)]
                    break
        if _nfib is None and not CHI2_SINGLE_FIBRE:
            print('  [chi2] META carries no fibre count; the stacked level '
                  'below is per-fibre and so is an OVER-estimate.')
        _nfib_var = None if CHI2_SINGLE_FIBRE else _nfib
        _var_c2 = _floor_var(_phot_var_abs(_obs_a, _sens,
                                           exptime=CHI2_EXPTIME_S,
                                           dwave=_dwave_c2, n_fibres=_nfib_var))
        _c2_mode = ('single fibre' if CHI2_SINGLE_FIBRE
                    else 'median stack of the row\'s fibres')
        # The mask does NOT require obs > 0: the loss keeps non-positive pixels
        # by mapping them to the row's median variance, and excluding them here
        # would measure a different noise model from the one being tested.
        _good = (np.isfinite(_obs_a) & np.isfinite(_res_a)
                 & np.isfinite(_sens)[None, :] & (_sens > 0)[None, :])
        # Science emission lines out, for the prediction AND the self-fit
        # (both use `_good`), so the two stay on identical pixels.
        if sci_line_mask_sel is not None and sci_line_mask_sel.shape == _good.shape:
            _good &= ~sci_line_mask_sel
        _pull2 = np.where(_good, _res_a ** 2 / _var_c2, np.nan)
        # NOT renormalised to a median of 1.  The scale is absolute now, and
        # dividing by the median would throw away the only thing this panel
        # gained -- it would force the median to 1 and make the axis a
        # relative spread again, which is what it was before 2026-09-09.
        _chi2_ph = np.nanmean(_pull2, axis=1)
        # LINEAR chi2 (2026-09-10).  On the single-fibre scale the whole
        # distribution lives inside one decade, so a log axis buys nothing and
        # costs the reader the ability to see how far past 1 the bulk sits.
        _c2f = _chi2_ph[np.isfinite(_chi2_ph)]

        # SECOND PANEL: the same chi2 over the blue side only (2026-09-10),
        # which is the continuum-sensitive one.  See CHI2_BLUE_MAX_A above.
        _wr = np.asarray(wave_ref, dtype=np.float64)
        _blue_m = np.isfinite(_wr) & (_wr < CHI2_BLUE_MAX_A)
        _n_blue_pix = int(_blue_m.sum())
        _chi2_blue = (np.nanmean(_pull2[:, _blue_m], axis=1) if _n_blue_pix
                      else np.full(_chi2_ph.shape, np.nan))
        _c2b = _chi2_blue[np.isfinite(_chi2_blue)]
        # Red side, where the OH lines are: what LINE_SCALING acts on.
        _red_m = np.isfinite(_wr) & (_wr >= CHI2_RED_MIN_A)
        _chi2_red = np.nanmean(_pull2[:, _red_m], axis=1)

        # The same chi2 with the sky-arm residual correction applied: the
        # stored residual is pred - obs, so obs - pred - corr = -(res + corr).
        if _ANY_CORR:
            _pull2_c = np.where(_good, (_res_a + np.vstack(sci_corr_rows)) ** 2 / _var_c2, np.nan)
            _chi2_corr = np.nanmean(_pull2_c, axis=1)
            _chi2_corr_blue = (np.nanmean(_pull2_c[:, _blue_m], axis=1) if _n_blue_pix
                               else np.full(_chi2_corr.shape, np.nan))
            _chi2_corr_red = np.nanmean(_pull2_c[:, _red_m], axis=1)
            del _pull2_c

        # DECOMPOSITION SELF-FIT chi2, the reference distribution (2026-09-11).
        # Identical noise model, identical pixels, identical mask -- the only
        # difference is that the coefficients are the FITTED ones rather than
        # the predicted ones.  Overlaid on both panels below so the reader can
        # see the two questions separately:
        #
        #   self-fit chi2   how well the DECOMPOSITION describes this spectrum
        #                   (its own residual; the ML floor)
        #   recon chi2      how well the PREDICTED coefficients describe it
        #
        # Their ratio is the only honest statement of how much the transfer
        # costs.  Reading the reconstruction chi2 alone conflates the two: a
        # median of 4 looks like a prediction failure until the decomposition
        # itself is shown sitting at 3.9 on the same rows.
        if sci_selfres_rows:
            _self_a = np.vstack(sci_selfres_rows).astype(np.float64)
            _pull2_self = np.where(_good, _self_a ** 2 / _var_c2, np.nan)
            _chi2_self = np.nanmean(_pull2_self, axis=1)
            _chi2_self_blue = (np.nanmean(_pull2_self[:, _blue_m], axis=1)
                               if _n_blue_pix
                               else np.full(_chi2_self.shape, np.nan))
            del _pull2_self, _self_a
            # float32 but still n_use x 12401 -- 25 MB at n_use=500, and the
            # figure below has yet to be built.  Nothing reads it again.
            sci_selfres_rows = []
        _c2fs = (_chi2_self[np.isfinite(_chi2_self)] if _chi2_self is not None
                 else np.empty(0))
        _c2bs = (_chi2_self_blue[np.isfinite(_chi2_self_blue)]
                 if _chi2_self_blue is not None else np.empty(0))
        # Both chi2 vectors are now n_use-long scalars; the n_use x 12401
        # intermediates behind them are dead.  Four of them at 50 MB each on
        # a 500-row sample, and the figure below still has to be built.
        # Take the row count first -- the figure title below needs it.
        _n_c2_rows = int(_obs_a.shape[0])
        del _pull2, _var_c2, _res_a, _obs_a, _good

        # What the chi2 FIGURE plots as "Prediction": the corrected chi2 when the
        # sky-arm correction is on (the uncorrected one is overlaid as a red
        # outline), the plain one otherwise.  The returned chi2_photon /
        # chi2_photon_blue stay uncorrected either way.
        _c2_corr_on = _chi2_corr is not None
        _plot_ph = _chi2_corr if _c2_corr_on else _chi2_ph
        _plot_blue = _chi2_corr_blue if _c2_corr_on else _chi2_blue
        _c2f_plot = _plot_ph[np.isfinite(_plot_ph)]
        _c2b_plot = _plot_blue[np.isfinite(_plot_blue)]
        _pred_label = f"Prediction + {_CORR_LABEL}" if _c2_corr_on else "Prediction"

        _figc = _make_subplots_resid(
            rows=2, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.10,
            vertical_spacing=0.13,
            subplot_titles=(
                f"full band  ({_wr.size} px)",
                f"blue of {CHI2_BLUE_MAX_A:.0f} A only, OH-poor  "
                f"({_n_blue_pix} px)",
                "full band: decomposition vs prediction, per row",
                f"blue < {CHI2_BLUE_MAX_A:.0f} A: decomposition vs prediction"))

        def _c2_panel(_v, _vs, _col, _colour, _vu=None):
            """Two overlaid linear chi2 histograms: reconstruction vs self-fit.

            `_v`  = chi2 of the ML reconstruction against the photon noise.
            `_vs` = chi2 of the DECOMPOSITION'S OWN fit on the same rows, same
                    pixels, same sigma.  It is the floor: the predictor is
                    predicting the coefficients of that model, so it cannot do
                    better than the model does, and the gap between the two
                    histograms is the whole cost of the sky-to-science
                    transfer.  Drawn behind, in grey.

            Upper edge from TUKEY'S FENCE, q75 + 3*IQR, not a percentile.  The
            distribution has a hard tail -- a handful of rows run 100x the
            median -- and on a 100-row sample the 99th percentile IS
            essentially the second-largest value, so a percentile cut puts the
            whole bulk in the first bin.  Measured on the decomposition
            residual the fence lands at 17-22 across seeds and keeps 87-91% of
            rows on-scale; p99 landed at 336-5150.  Clipped rows are counted
            in the annotation, never dropped from the statistics.

            The fence is taken over the UNION of the two series so both are on
            the same bins; a fence from the reconstruction alone would clip
            whichever distribution happens to be broader and make the overlay
            a comparison of two different axes.
            """
            _vf = np.asarray(_v, dtype=np.float64)
            _vf = _vf[np.isfinite(_vf)]
            _vsf = np.asarray(_vs, dtype=np.float64)
            _vsf = _vsf[np.isfinite(_vsf)]
            _vuf = (np.asarray(_vu, dtype=np.float64) if _vu is not None
                    else np.empty(0))
            _vuf = _vuf[np.isfinite(_vuf)]
            _both = np.concatenate([_vf, _vsf, _vuf])
            _lo = 0.0
            if _both.size:
                _q25, _q75 = np.percentile(_both, [25, 75])
                _hi = float(_q75 + 3.0 * (_q75 - _q25))
                _p99 = float(np.percentile(_both, 99))
                if np.isfinite(_p99):
                    _hi = min(_hi, _p99)   # a tight distribution needs no fence
                if not np.isfinite(_hi) or _hi <= 0:
                    _hi = float(np.nanmax(_both))
            else:
                _hi = 1.2
            _hi = max(_hi, 1.2)   # always keep the photon limit on-scale
            _bins = dict(start=_lo, end=_hi, size=(_hi - _lo) / 36.0)
            if _vsf.size:
                _figc.add_trace(go.Histogram(
                    x=_vsf, xbins=_bins, name="Decomposition",
                    marker=dict(color="#999999"), opacity=0.55,
                    legendgroup="self", showlegend=(_col == 1),
                    hovertemplate=("self-fit chi2_red=%{x:.3g}"
                                   "<br>n=%{y}<extra></extra>")),
                    row=1, col=_col)
            if _vuf.size:
                # Uncorrected prediction as an outline, for the before/after.
                _figc.add_trace(go.Histogram(
                    x=_vuf, xbins=_bins, name="Prediction, no correction",
                    marker=dict(color="rgba(0,0,0,0)",
                                line=dict(color="#e31a1c", width=1.5)),
                    legendgroup="uncorr", showlegend=(_col == 1),
                    hovertemplate=("uncorrected chi2_red=%{x:.3g}"
                                   "<br>n=%{y}<extra></extra>")),
                    row=1, col=_col)
            _figc.add_trace(go.Histogram(
                x=_vf, xbins=_bins, name=_pred_label,
                marker=dict(color=_colour), opacity=0.75,
                legendgroup="recon", showlegend=(_col == 1),
                hovertemplate=("recon chi2_red=%{x:.3g}"
                               "<br>n=%{y}<extra></extra>")),
                row=1, col=_col)
            _figc.update_xaxes(range=[_lo, _hi * 1.02], row=1, col=_col)
            # 1 = the reconstruction error equals the noise it is measured
            # against; kept on-scale by the max() above so the reference mark
            # is always there.
            _figc.add_vline(x=1.0, line=dict(color="black", width=1,
                                             dash="dash"), row=1, col=_col)
            if _vf.size:
                _n_out = int(np.sum(_vf > _hi))
                _med_r = float(np.median(_vf))
                _txt = f"Pred. median {_med_r:.3g}"
                if _vuf.size:
                    _txt += f"<br>No-corr. median {float(np.median(_vuf)):.3g}"
                if _vsf.size:
                    _med_s = float(np.median(_vsf))
                    _txt += (f"<br>Decomp. median {_med_s:.3g}"
                             f"<br>ratio {_med_r / _med_s:.2f}x"
                             if _med_s > 0 else
                             f"<br>Decomp. median {_med_s:.3g}")
                if _n_out:
                    _txt += (f"<br>{_n_out} rows above {_hi:.3g}"
                             f"<br>(max {float(_vf.max()):.3g})")
                _figc.add_annotation(
                    x=0.98, y=0.94, xref="x domain", yref="y domain",
                    text=_txt, showarrow=False, xanchor="right", align="right",
                    font=dict(size=10, color="#666666"), row=1, col=_col)

        def _c2_scatter(_v, _vs, _col, _colour):
            """Per-row decomposition self-fit chi2 (x) against prediction (y).

            The histograms above show the two DISTRIBUTIONS; this shows the two
            numbers ROW BY ROW, which is what separates the failure modes:

              * on the diagonal   the prediction is at the decomposition's own
                                  floor -- this row is as good as it can be, no
                                  matter how large both numbers are
              * far above it      the DECOMPOSITION describes this row but the
                                  PREDICTION does not: a transfer failure, and
                                  the only kind this predictor can fix
              * far to the right  the decomposition itself failed, so the
                                  "truth" this row is scored against is not
                                  trustworthy and a large y is not the
                                  predictor's fault

            A marginal histogram cannot tell the second from the third: a row
            at chi2 40 looks equally bad in both, and only the pairing says
            whether its target was ever any good.

            LOG-LOG here, unlike the linear histograms above.  The histograms
            are fenced to the bulk and deliberately hide the tail; this panel
            exists FOR the tail, which spans three decades.
            """
            _x = np.asarray(_vs, dtype=np.float64)
            _y = np.asarray(_v, dtype=np.float64)
            if _x.size != _y.size or _x.size == 0:
                return
            try:                      # a local of this cell, not a global
                _lab = list(_row_label)[:_x.size]
            except NameError:
                _lab = [f'row {i}' for i in range(_x.size)]
            if len(_lab) != _x.size:   # never mislabel a point
                _lab = [f'row {i}' for i in range(_x.size)]
            _ok = np.isfinite(_x) & np.isfinite(_y) & (_x > 0) & (_y > 0)
            if not _ok.any():
                return
            _xo, _yo = _x[_ok], _y[_ok]
            _lo_o = [l for l, k in zip(_lab, _ok) if k]
            _rat = _yo / _xo
            # Split at 2x so the failures are visually separate, not a colour ramp
            # the eye has to decode.
            _bad = _rat > 2.0
            for _m, _nm, _cc, _sz in ((~_bad, "at the floor (<= 2x)", _colour, 4),
                                      (_bad, "transfer failure (> 2x)", "#e31a1c", 6)):
                if not _m.any():
                    continue
                _figc.add_trace(go.Scattergl(
                    x=_xo[_m], y=_yo[_m], mode="markers", name=_nm,
                    marker=dict(color=_cc, size=_sz, opacity=0.55,
                                line=dict(width=0)),
                    legendgroup=_nm, showlegend=(_col == 1),
                    text=[f"{l}<br>Ratio {r:.2f}x"
                          for l, r in zip([_l for _l, _k in zip(_lo_o, _m) if _k],
                                          _rat[_m])],
                    hovertemplate=("%{text}<br>decomposition %{x:.3g}"
                                   "<br>prediction %{y:.3g}<extra></extra>")),
                    row=2, col=_col)
            _lo = float(min(_xo.min(), _yo.min())) * 0.7
            _hi = float(max(_xo.max(), _yo.max())) * 1.4
            _ref = np.array([_lo, _hi])
            for _f, _dash, _w in ((1.0, "solid", 1.5), (2.0, "dash", 1.0),
                                  (10.0, "dot", 1.0)):
                _figc.add_trace(go.Scattergl(
                    x=_ref, y=_ref * _f, mode="lines", showlegend=False,
                    line=dict(color="#444444", width=_w, dash=_dash),
                    hoverinfo="skip"), row=2, col=_col)
            _figc.update_xaxes(type="log", range=[np.log10(_lo), np.log10(_hi)],
                               row=2, col=_col)
            _figc.update_yaxes(type="log", range=[np.log10(_lo), np.log10(_hi)],
                               row=2, col=_col)
            _figc.add_annotation(
                x=0.02, y=0.97, xref="x domain", yref="y domain",
                text=(f"Median ratio {float(np.median(_rat)):.2f}x<br>"
                      f"{int((_rat > 2).sum())} rows > 2x, "
                      f"{int((_rat > 10).sum())} > 10x<br>"
                      f"Lines: 1x (floor), 2x, 10x"),
                showarrow=False, xanchor="left", align="left",
                font=dict(size=10, color="#666666"), row=2, col=_col)

        _c2_panel(_c2f_plot, _c2fs, 1, "#1f78b4", _c2f if _c2_corr_on else None)
        _c2_panel(_c2b_plot, _c2bs, 2, "#6a3d9a", _c2b if _c2_corr_on else None)
        # NOT _c2f/_c2fs: those are each filtered by their OWN finite mask, so
        # they can differ in length and are not row-aligned.  The scatter pairs
        # rows, so it must take the raw n_use-long vectors and apply one JOINT
        # mask itself.
        if _chi2_self is not None:
            _c2_scatter(_plot_ph, _chi2_self, 1, "#1f78b4")
        if _chi2_self_blue is not None:
            _c2_scatter(_plot_blue, _chi2_self_blue, 2, "#6a3d9a")
        _q = np.nanpercentile(_plot_ph, [10, 90])
        _figc.update_layout(
            template="plotly_white",
            height=(420 if _chi2_self is None else 840), barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1.0, font=dict(size=10)),
            title=dict(text=(f"SCI reconstruction vs the PHOTON noise of a "
                             f"{_c2_mode.upper()} "
                             f"(n={_n_c2_rows} rows)<br><sub>ABSOLUTE reduced "
                             f"chi2 -- 1 = the reconstruction error equals the "
                             f"shot noise of one {CHI2_EXPTIME_S:.0f} s "
                             f"{_c2_mode}, which is what the sky model will be "
                             f"subtracted from.  p10/p50/p90 = "
                             f"{_q[0]:.3g} / {float(np.nanmedian(_plot_ph)):.3g} / "
                             f"{_q[1]:.3g}"
                             + (f" WITH {_CORR_LABEL}; red outline = "
                                f"the same rows uncorrected, median "
                                f"{float(np.nanmedian(_chi2_ph)):.3g}."
                                if _c2_corr_on else ".")
                             + ("" if _chi2_self is None else
                                f"  Grey = the DECOMPOSITION'S OWN fit on the "
                                f"same rows and the same noise, median "
                                f"{float(np.nanmedian(_chi2_self)):.3g} -- the "
                                f"floor the transfer is measured against.  "
                                f"Bottom row pairs the two PER ROW: on the "
                                f"diagonal = at the floor, above it = a "
                                f"transfer failure, far right = the "
                                f"decomposition failed and the target is not "
                                f"trustworthy.")
                             + "</sub>"),
                       font=dict(size=13), x=0.02, xanchor="left"),
            margin=dict(t=110))
        # Whole label inside ONE $...$ span (\text{} for the words) -- Plotly's
        # MathJax rendering drops a plain-text prefix mixed with a separate
        # $...$ span in the same title (e.g. "Reduced $\chi^2$" rendered as
        # just "chi^2", the "Reduced " silently gone).
        _figc.update_xaxes(title_text=r"$\text{Reduced }\chi^2$", row=1, col=1)
        _figc.update_yaxes(title_text="Rows", row=1, col=1)
        _figc.update_xaxes(title_text=r"$\text{Reduced }\chi^2$", row=1, col=2)
        _figc.update_yaxes(title_text="Rows", row=1, col=2)
        if _chi2_self is not None:
            for _cc in (1, 2):
                _figc.update_xaxes(title_text=r"$\text{Decomposition self-fit }\chi^2$",
                                   row=2, col=_cc)
                _figc.update_yaxes(title_text=(r"$\text{Prediction + corr. }\chi^2$" if _c2_corr_on
                                               else r"$\text{Prediction }\chi^2$"), row=2, col=_cc)
        _figc.show()
        print(f"  [chi2] ABSOLUTE reduced chi2 vs the {_c2_mode} photon "
              f"model: median {float(np.nanmedian(_chi2_ph)):.4g}, "
              f"p10 {_q[0]:.4g}, p90 {_q[1]:.4g}, "
              f"max {float(np.nanmax(_chi2_ph)):.4g}"
              + ("   (science emission-line windows excluded)"
                 if sci_line_mask_sel is not None else ""))
        if _c2b.size:
            _qb = np.nanpercentile(_chi2_blue, [10, 90])
            print(f"  [chi2] blue of {CHI2_BLUE_MAX_A:.0f} A only "
                  f"({_n_blue_pix} of {_wr.size} px, OH-poor): median "
                  f"{float(np.nanmedian(_chi2_blue)):.4g}, p10 {_qb[0]:.4g}, "
                  f"p90 {_qb[1]:.4g}   <- continuum-sensitive; not "
                  f"comparable to the full-band number above (fainter sky "
                  f"there means a larger sigma/flux)")
        if _chi2_self is not None:
            _ms = float(np.nanmedian(_chi2_self))
            _mr = float(np.nanmedian(_chi2_ph))
            print(f"  [chi2] DECOMPOSITION SELF-FIT on the same rows/noise: "
                  f"median {_ms:.4g} full band"
                  + (f", {float(np.nanmedian(_chi2_self_blue)):.4g} blue"
                     if _chi2_self_blue is not None else "")
                  + f"   -> the reconstruction costs "
                    f"{(_mr / _ms if _ms > 0 else float('nan')):.2f}x the "
                    f"decomposition's own residual full band"
                  + ("" if (_chi2_self_blue is None or _chi2_blue is None)
                     else f", {(float(np.nanmedian(_chi2_blue)) / float(np.nanmedian(_chi2_self_blue))):.2f}x blue"))
            print(f"  [chi2] a ratio near 1 means the ML transfer is no longer "
                  f"the error source on these rows -- the decomposition is.")
        if _chi2_corr is not None:
            _parts = []
            if RESIDUAL_CORRECTION:
                _parts.append("sky-arm correction "
                              + ("full band" if not np.isfinite(RESIDUAL_CORRECTION_MAX_A)
                                 else f"to {RESIDUAL_CORRECTION_MAX_A:.0f} A + 100 A taper")
                              + f", smoothing {RESIDUAL_SMOOTHING_A!r}")
            if LINE_SCALING:
                _parts.append(f"line scaling (high-pass {LINE_SCALING_HIGHPASS_A:g} A, "
                              f"prior {LINE_SCALING_PRIOR:g}, tilt {LINE_SCALING_TILT})")
            print(f"  [chi2] WITH {' + '.join(_parts).upper()}: median "
                  f"{float(np.nanmedian(_chi2_corr)):.4g} full band, "
                  f"{float(np.nanmedian(_chi2_corr_blue)):.4g} blue, "
                  f"{float(np.nanmedian(_chi2_corr_red)):.4g} red >= {CHI2_RED_MIN_A:.0f} A"
                  f"   (uncorrected {float(np.nanmedian(_chi2_ph)):.4g} / "
                  f"{float(np.nanmedian(_chi2_blue)):.4g} / {float(np.nanmedian(_chi2_red)):.4g}; "
                  f"rows made worse: {100 * np.nanmean(_chi2_corr > _chi2_ph):.1f}% full, "
                  f"{100 * np.nanmean(_chi2_corr_blue > _chi2_blue):.1f}% blue, "
                  f"{100 * np.nanmean(_chi2_corr_red > _chi2_red):.1f}% red)")
            if LINE_SCALING and any(d is not None for d in sci_line_scale_rows):
                _names = sorted({k for d in sci_line_scale_rows if d for k in d} - {"accepted"})
                _txt = ", ".join(
                    f"{k} {np.nanmedian([d.get(k, np.nan) for d in sci_line_scale_rows if d]):.3f}"
                    for k in _names if not k.endswith("_tilt"))
                print(f"  [line scaling] median fitted scale per template: {_txt}")

        if CHI2_SINGLE_FIBRE and _nfib is not None:
            _fac = float(np.nanmedian(_nfib)) * (2.0 / np.pi)
            print(f"  [chi2] single-fibre sigma; the rows are stacks of a "
                  f"median {np.nanmedian(_nfib):.0f} fibres, so against the "
                  f"STACK this chi2 would be ~{_fac:.0f}x larger.")

    rmse_subset_results = {
        "row_positions": sel_pos,
        "row_indices": sel_rows,
        # CORPUS rows since 2026-09-23 (they were every10 rows before), and the
        # files they index.  A consumer that opens its own every10 stack and
        # indexes it with these gets a different spectrum per row -- the exact
        # failure canonical_row_labels exists to document -- so the paths
        # travel with the indices rather than being assumed.
        "source_input": EVAL_INPUT,
        "source_near": EVAL_NEAR,
        "source_far": EVAL_FAR,
        "source_sci": EVAL_SCI,
        "split": _EVAL_SPLIT,
        "row_ident": _row_ident,
        "row_labels": _row_label,
        "near_rmse": near_rmse,
        "far_rmse": far_rmse,
        "sci_rmse": sci_rmse,
        "near_wrmse": near_wrmse,
        "far_wrmse": far_wrmse,
        "sci_wrmse": sci_wrmse,
        "pix_sigma_source": _pix_sigma_source,
        "summary": summary_df,
        "summary_display": summary_disp_df,
        "sci_residuals": sci_resid_arr,
        "chi2_photon": _chi2_ph,   # ABSOLUTE reduced chi2; see the panel
        # Same, over lambda < CHI2_BLUE_MAX_A only: OH dominates the red half
        # and swamps the continuum this predictor exists to get right.
        "chi2_photon_blue": _chi2_blue,
        "chi2_blue_max_a": CHI2_BLUE_MAX_A,
        # Decomposition self-fit chi2 on the SAME rows, pixels and sigma.
        # `headline_summary` reads these so the reported floor is measured on
        # this corpus rather than carried over as a hard-coded constant.
        "chi2_self": _chi2_self,
        "chi2_self_blue": _chi2_self_blue,
        "chi2_photon_red": _chi2_red,
        # With the sky-arm correction and/or the line scaling (None unless
        # RESIDUAL_CORRECTION or LINE_SCALING).
        "chi2_photon_corrected": _chi2_corr,
        "chi2_photon_blue_corrected": _chi2_corr_blue,
        "chi2_photon_red_corrected": _chi2_corr_red,
        # Per-row {template: fitted scale} from LINE_SCALING (None per row otherwise).
        "line_scales": sci_line_scale_rows,
    }
