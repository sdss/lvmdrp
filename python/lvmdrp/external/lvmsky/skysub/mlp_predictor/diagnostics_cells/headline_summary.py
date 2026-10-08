# Targets: one consolidated table -- the numbers worth reading first.
# Look for: a group losing to its naive baseline (gain <= 0); a moon-down MAD
#           far above moon-up; a reduced chi2 that has moved by more than the
#           seed noise since the last run.
#
# This cell RECOMPUTES NOTHING.  It reads what `naive_baseline`,
# `amplitude_error_vs_ctx` and `full_spectrum_batch_rmse` already left in the
# shared diagnostics globals, so it cannot disagree with the cells it
# summarises -- which a re-implementation eventually would.  Run those first;
# any that has not run is reported as missing rather than silently skipped.

_have = {
    'naive_baseline': 'naive_baseline_result' in globals(),
    'amplitude_error_vs_ctx': 'amplitude_error_vs_ctx_result' in globals(),
    'full_spectrum_batch_rmse': 'rmse_subset_results' in globals(),
}
_missing = [k for k, v in _have.items() if not v]
print('=' * 78)
print('HEADLINE SUMMARY')
print('=' * 78)
if _missing:
    print('  not run yet, so omitted below: ' + ', '.join(_missing))

# --- coefficient space: ML against the best naive baseline -----------------
if _have['naive_baseline']:
    _nb = naive_baseline_result
    print(f"\ncoefficient-space sRMSE on {_nb['n_test']} test rows"
          "   (gain > 0 = ML beats every naive variant)")
    print(f"  {'group':<13s} {'ML':>10s} {'best naive':>11s} {'which':>14s} {'gain':>8s}")
    for g, d in _nb['per_group'].items():
        _flag = '' if d['gain_pct'] > 0 else '   <-- LOSES'
        print(f"  {g:<13s} {d['ml']:>10.4g} {d['best_naive']:>11.4g} "
              f"{d['best_naive_name']:>14s} {d['gain_pct']:>+7.1f}%{_flag}")
    _ge = _nb['group_equal']
    print(f"  {'GROUP-EQUAL':<13s} {_ge['ml']:>10.5g} {_ge['best_naive']:>11.5g} "
          f"{_ge['best_naive_name']:>14s} {_ge['gain_pct']:>+7.1f}%")
    print('  NB a gain can improve because the BASELINE got worse on a harder '
          'test set.\n     Compare ML columns across runs, not gains alone.')

# --- amplitude accuracy, split by moon state -------------------------------
if _have['amplitude_error_vs_ctx']:
    _df = amplitude_error_vs_ctx_result['summary']
    _cols = {c.lower(): c for c in _df.columns}
    _mad = _cols.get('mad'); _cmp = _cols.get('component')
    _med = _cols.get('median'); _fam = _cols.get('family')
    if _mad and _cmp:
        print(f"\nintegrated-amplitude error, log10(pred/true)   "
              f"[MAD is robust; std is not]")
        print(f"  {'family':<12s} {'component':<20s} {'MAD':>9s} {'median':>9s}")
        for _, r in _df.iterrows():
            print(f"  {(str(r[_fam]) if _fam else ''):<12s} "
                  f"{str(r[_cmp]):<20s} {float(r[_mad]):>9.5f} "
                  + (f"{float(r[_med]):>+9.5f}" if _med else ''))
    else:
        print('\n  (amplitude summary present but its columns are not the '
              f'expected ones: {list(_df.columns)})')

# --- spectrum space --------------------------------------------------------
if _have['full_spectrum_batch_rmse']:
    _rs = rmse_subset_results
    _sci = np.asarray(_rs.get('sci_rmse', []), dtype=float)
    _sci = _sci[np.isfinite(_sci)]
    if _sci.size:
        print(f"\nspectrum space, {_sci.size}-row sample")
        print(f"  sci pRMSE      median {np.median(_sci):.4g}  "
              f"p90 {np.percentile(_sci, 90):.4g}")
    # The floor is MEASURED, not carried over.  `full_spectrum_batch_rmse`
    # now computes the decomposition's own self-fit chi2 on the same rows,
    # pixels and sigma, so the floor quoted here always belongs to the corpus
    # actually loaded.  The previous hard-coded 3.96 / 1.20 were measured on
    # gaia-stars-mask-cont and would have silently misreported any other one.
    _c2s = _rs.get('chi2_self')
    _c2s = (np.asarray(_c2s, dtype=float) if _c2s is not None else None)
    if _c2s is not None:
        _c2s = _c2s[np.isfinite(_c2s)]
    _c2 = _rs.get('chi2_photon')
    if _c2 is not None:
        _c2 = np.asarray(_c2, dtype=float); _c2 = _c2[np.isfinite(_c2)]
        if _c2.size:
            print(f"  reduced chi2   median {np.median(_c2):.4g}  "
                  f"p90 {np.percentile(_c2, 90):.4g}   (ABSOLUTE, vs ONE "
                  f"900 s fibre's shot noise)")
            print(f"                 1 = the reconstruction error is at the "
                  f"noise of the single fibre it will be subtracted from.")
            if _c2s is not None and _c2s.size:
                _fl = float(np.median(_c2s))
                print(f"  decomp self-fit median {_fl:.4g}  "
                      f"p10 {np.percentile(_c2s, 10):.3g}  "
                      f"p90 {np.percentile(_c2s, 90):.3g}   <- the FLOOR: "
                      f"the same model fitted to")
                print(f"                 this very row.  A coefficient "
                      f"prediction cannot go below it, so read the\n"
                      f"                 ratio, not the absolute value: "
                      f"reconstruction / self-fit = "
                      f"{(np.median(_c2) / _fl if _fl > 0 else float('nan')):.2f}x.")
            else:
                print(f"                 (decomposition self-fit floor "
                      f"unavailable -- re-run the batch RMSE cell)")
    _c2b = _rs.get('chi2_photon_blue')
    if _c2b is not None:
        _c2b = np.asarray(_c2b, dtype=float); _c2b = _c2b[np.isfinite(_c2b)]
        if _c2b.size:
            _cut = _rs.get('chi2_blue_max_a') or 6000.0
            print(f"  blue < {_cut:.0f} A  median {np.median(_c2b):.4g}  "
                  f"p90 {np.percentile(_c2b, 90):.4g}   (OH-poor, so this is "
                  f"the CONTINUUM's chi2)")
            print(f"                 not comparable to the full-band number: "
                  f"the sky is fainter blueward, so\n"
                  f"                 sigma/flux is larger and the same "
                  f"fractional error scores lower.")
            _c2sb = _rs.get('chi2_self_blue')
            if _c2sb is not None:
                _c2sb = np.asarray(_c2sb, dtype=float)
                _c2sb = _c2sb[np.isfinite(_c2sb)]
                if _c2sb.size:
                    _flb = float(np.median(_c2sb))
                    print(f"  decomp self-fit median {_flb:.4g}  "
                          f"p10 {np.percentile(_c2sb, 10):.3g}  "
                          f"p90 {np.percentile(_c2sb, 90):.3g}   <- the blue "
                          f"floor; ratio "
                          f"{(np.median(_c2b) / _flb if _flb > 0 else float('nan')):.2f}x.")

print('=' * 78)
headline_summary_result = dict(available=_have)
