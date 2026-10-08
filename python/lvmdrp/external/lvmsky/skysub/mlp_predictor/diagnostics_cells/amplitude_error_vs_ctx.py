# Per-family AMPLITUDE error against the context that physically drives that
# family.  Replaces the old per-coefficient (pred - true) vs true scatter,
# which showed absolute coefficient residuals and so was dominated by whichever
# coefficients happen to be large.
#
# HOW THE AMPLITUDE IS DEFINED
# ----------------------------
# Every family is a set of coefficients on its own pre-LSF template basis:
#
#     moon        matrix_moon      15 rows, one per Moon_bs knot
#     zodi        matrix_zodi       5 rows, one per Zodi_bs knot
#     diffuse     matrix_diffuse    3 rows: HO2, FeO, O2Ac
#     OH          matrix_oh       357 rows, one per OH stick
#
# O2_b01 is deliberately EXCLUDED.  Its coefficient is nonzero on every row
# (median 371.5) but `matrix_o2` from the static build path is identically zero
# -- the O2 band's template is fitted PER ROW and stored as VECTOR_O2 in the
# decomposition products, so there is no static basis to integrate.  Including
# it silently contributes nothing and would mislabel an OH-only trace.
#
# A row's family spectrum is f_g(lambda) = sum_k c_gk * B_gk(lambda), and its
# AMPLITUDE is that spectrum integrated over the band,
#
#     A_g = sum_lambda f_g(lambda) = c_g . v_g ,      v_g = B_g.sum(axis=1)
#
# a LINEAR FUNCTIONAL of the coefficients.  Amplitude rather than the raw
# coefficients because the moon/zodi error is overwhelmingly a brightness
# error: 83-89% of the tail's mean-squared error is removed by a single per-row
# rescale, and that rescale agrees with this integrated ratio to rho =
# 0.99-1.00, with tail shape error of only 2-5%.
#
# The basis row order is asserted against the coefficient names below.  It
# matters: the loss indexes each group positionally, so a reordering upstream
# would silently pair coefficients with the wrong basis rows, and a count check
# alone cannot see that.
#
# HOW THE ERROR IS DEFINED
# ------------------------
#     Delta_g = log10( A_g^pred / A_g^true )        [+ = over-predicted]
#
# Log because the error is multiplicative: symmetric in over- and
# under-prediction, scale-free, and comparable across the ~16x amplitude range
# between dark and bright-moon rows.
#
# No error bars are drawn on the truth.  The persisted COEF_COV_* blocks do
# propagate exactly to sigma_A = sqrt(v^T Sigma v), but they are uncalibrated
# by 50-180x -- the |z| and reliability cells report `sigma_scale hint 0.02x`,
# a truth-conditioned median|z| of 0.0037 against a target of 0.6745, and a
# reliability slope of 0.53-0.65 instead of 1.  They order rows correctly and
# cannot be read as error bars.
#
# WHAT TO LOOK FOR
# ----------------
# Each family is plotted only against the context that physically drives IT --
# moon geometry for the moon, ecliptic geometry for the zodi, airglow and
# solar-activity terms for the diffuse continuum and OH.  A trace that departs
# from zero in some regime is a bias the head has not learned; a flat trace at
# zero means the family's brightness is being transferred correctly there.
#
# For the moon, the traces are flat against everything, and that is the result
# rather than a null: regressing the signed moon amplitude error on all 37
# context features gives R^2 = -0.007 (RandomForest, 8276 training rows), so no
# context-driven correction of any parametrisation can work.  Five attempts
# confirmed it -- see ablations.RETIRED.

required = ['filtered_triplet', 'mlp_artifacts', 'test_idx', 'group_indices',
            'predict_sci_coefficients_default', 'SkyDecompLSFSurfaceIterative',
            '_infer_base_dir_for_reconstruction', 'N_MOON_KNOTS', 'SPLIT_ZODI',
            'N_ZODI_KNOTS', 'DECOMP_DATA_ROOT', 'DECOMP_STEM']
_missing = [k for k in required if k not in globals() or globals()[k] is None]
if _missing:
    raise RuntimeError('Run the training + diagnostics-context cells first. '
                       'Missing: ' + ', '.join(_missing))

N_BINS = 8

# Per-family context.  Airglow families get the van Rhijn slant factors, solar
# activity and the seasonal/nightly clocks; the scattered/zodiacal families get
# their own illumination geometry.
_CTX_MOON = ('moon_alt', 'moon_sep', 'moon_fli', 'moon_signal_proxy',
             'moon_airmass_up', 'moon_up_smooth', 'airmass', 'alt',
             'sun_alt', 'sun_sep', 'ecl_beta_deg', 'vanrhijn_285km')
_CTX_ZODI = ('ecl_beta_deg', 'ecl_lon_sin', 'ecl_lon_cos', 'zodi_log10_v',
             'sun_sep', 'sun_alt', 'airmass', 'alt',
             'moon_alt', 'moon_fli', 'obstime_year_sin', 'obstime_year_cos')
_CTX_AIRGLOW = ('vanrhijn_87km', 'vanrhijn_95km', 'airmass', 'alt',
                'f107', 'f107_81d', 'kp', 'ew',
                'obstime_day_sin', 'obstime_day_cos',
                'obstime_year_sin', 'obstime_year_cos')

_te = np.asarray(test_idx, dtype=int)
_ctx_names = [str(n).strip().lower() for n in filtered_triplet['ctx_names']]
_ctx = np.asarray(filtered_triplet['ctx_sci'], dtype=np.float64)[_te]
_coef_names = [str(n) for n in filtered_triplet['coef_names']]

_pred = np.asarray(predict_sci_coefficients_default(
    mlp_artifacts,
    coef_near_phys=np.asarray(filtered_triplet['coef_near'], np.float32)[_te],
    coef_far_phys=np.asarray(filtered_triplet['coef_far'], np.float32)[_te],
    ctx_near_phys=np.asarray(filtered_triplet['ctx_near'], np.float32)[_te],
    ctx_far_phys=np.asarray(filtered_triplet['ctx_far'], np.float32)[_te],
    ctx_sci_phys=np.asarray(filtered_triplet['ctx_sci'], np.float32)[_te]),
    dtype=np.float64)
_true = np.asarray(filtered_triplet['coef_sci'], dtype=np.float64)[_te]

with fits.open(f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_every10.fits') as _h:
    _w = np.asarray(_h['WAVE'].data, dtype=np.float64)
    _wave = _w if _w.ndim == 1 else _w[0]
# The amplitude weights v_g must come from the basis THIS corpus was fitted
# with.  Moon / zodi / diffuse are byte-identical between the decomposition
# variants (verified: max |diff| exactly 0), but OH is NOT -- the telluric fit
# groups it differently -- so an OH amplitude built on the split-zodi basis
# would be the wrong linear functional.  `TELLURIC_BASIS_KW` is the
# representative-row telluric bundle the notebook installs, or None.
from mlp_predictor.data import make_reconstruction_decomposer as _mk_decomp
_basis_tel = globals().get('TELLURIC_BASIS_KW')
if _basis_tel is not None:
    _model = _mk_decomp(
        _wave, n_spline_knots=N_MOON_KNOTS,
        base_dir=_infer_base_dir_for_reconstruction(),
        split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS,
        telluric=_basis_tel)
    print('  amplitude basis: TELLURIC variant (representative row)')
else:
    _model = SkyDecompLSFSurfaceIterative(
        _wave, lsf_sigma=1.0, n_spline_knots=N_MOON_KNOTS,
        base_dir=_infer_base_dir_for_reconstruction(),
        split_zodi=SPLIT_ZODI, n_zodi_spline_knots=N_ZODI_KNOTS)

# Per-family basis, with the row order asserted against the coefficient names.
_mes_idx = np.asarray(group_indices['mesospheric'], dtype=int)
_mes_names = [_coef_names[j] for j in _mes_idx]
_n_oh = sum(1 for n in _mes_names if n.startswith('OH_'))
if _mes_names[:_n_oh] != sorted(_mes_names[:_n_oh]) or _mes_names[_n_oh:] != ['O2_b01']:
    raise RuntimeError(
        f'mesospheric coefficient order is not [OH_* ascending, O2_b01]: '
        f'{_mes_names[:2]} ... {_mes_names[-2:]}.  Stacking matrix_oh on '
        f'matrix_o2 would pair coefficients with the wrong basis rows.')
_cont_names = [_coef_names[j] for j in np.asarray(group_indices['continuum'], int)]
if _cont_names != list(getattr(_model, 'diffuse_names', _cont_names)):
    raise RuntimeError(f'diffuse order mismatch: coefficients {_cont_names} vs '
                       f'basis {list(_model.diffuse_names)}')

_BASIS = {
    'moon': np.asarray(_model.matrix_moon, dtype=np.float64),
    'zodi': np.asarray(_model.matrix_zodi, dtype=np.float64),
    'continuum': np.asarray(_model.matrix_diffuse, dtype=np.float64),
    'oh': np.asarray(_model.matrix_oh, dtype=np.float64),
}
_oh_idx = _mes_idx[:_n_oh]                      # OH_* only; O2_b01 dropped
if _BASIS['oh'].shape[0] != _oh_idx.size:
    raise RuntimeError(f"matrix_oh has {_BASIS['oh'].shape[0]} rows but there "
                       f"are {_oh_idx.size} OH coefficients")
if not np.any(np.asarray(_model.matrix_o2, dtype=np.float64)):
    print('  note: O2_b01 excluded -- matrix_o2 from the static build is '
          'identically zero (its template is the per-row VECTOR_O2).')

print('Amplitude  A_g = sum_lambda f_g(lambda) = c_g . v_g,  v_g = B_g.sum(axis=1)')
print('Error      Delta_g = log10(A_g^pred / A_g^true)      [+ = over-predicted]')
# Moon-up mask for the per-subset MAD above.  From the moon/zodi model cache
# when it exists (that is the same gate the moon_model_log_ratio feature uses);
# otherwise fall back to the moon_alt ctx column, and to None if neither is
# available, in which case the split is simply not printed.
_MOON_UP_TEST = None
try:
    from mlp_predictor import moon_model_cache as _mmc_amp
    _c_amp = _mmc_amp.load(f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}')
    _MOON_UP_TEST = (np.asarray(_c_amp['sci_moon_alt_deg'], dtype=float)[
        np.asarray(filtered_triplet['row_index'], dtype=np.int64)[test_idx]] > 0.0)
except Exception as _exc_amp:
    _ctx_l = [str(n).lower() for n in filtered_triplet['ctx_names']]
    if 'moon_alt' in _ctx_l:
        _MOON_UP_TEST = (np.asarray(filtered_triplet['ctx_sci'],
                                    dtype=float)[test_idx,
                                                 _ctx_l.index('moon_alt')] > 0.0)
    else:
        print(f'  (moon-up split unavailable: {type(_exc_amp).__name__})')

print(f'Test rows: {_te.size}   basis row order asserted against coef_names\n')


def _delta(idx, basis):
    """Signed log amplitude error for one coefficient subset."""
    v = np.asarray(basis, dtype=np.float64).sum(axis=1)
    at = _true[:, idx] @ v
    ap = _pred[:, idx] @ v
    ok = np.isfinite(at) & np.isfinite(ap) & (at > 0) & (ap > 0)
    d = np.full(_te.size, np.nan)
    d[ok] = np.log10(ap[ok] / at[ok])
    return d, ok


# Components per figure: (label, colour, coefficient positions, basis rows).
_mes_wave = None
if 'coef_wavelengths_a' in globals() and coef_wavelengths_a is not None:
    _mes_wave = np.asarray(coef_wavelengths_a, dtype=np.float64)[_mes_idx]

_FIGS = []
_FIGS.append(('moon', _CTX_MOON, [
    ('moon', '#1f78b4', np.asarray(group_indices['moon'], int), _BASIS['moon'])]))
_FIGS.append(('zodi', _CTX_ZODI, [
    ('zodi', '#e41a1c', np.asarray(group_indices['zodi'], int), _BASIS['zodi'])]))

# Diffuse continuum: the three species separately, plus their sum.
_cont_idx = np.asarray(group_indices['continuum'], int)
_cont_comps = [('HO2 + FeO + O2Ac', '#333333', _cont_idx, _BASIS['continuum'])]
for _k, (_nm, _col) in enumerate(zip(_cont_names, ('#1b9e77', '#d95f02', '#7570b3'))):
    _cont_comps.append((_nm, _col, _cont_idx[[_k]], _BASIS['continuum'][[_k]]))
_FIGS.append(('diffuse continuum', _CTX_AIRGLOW, _cont_comps))

# OH: total, then split by wavelength band where the vibrational bands and the
# emitting layer response differ, plus O2_b01 as its own species.
_mes_comps = [('OH total', '#333333', _oh_idx, _BASIS['oh'])]
if _mes_wave is not None and np.isfinite(_mes_wave[:_n_oh]).all():
    for (_lo, _hi, _col) in ((3600, 5500, '#1f78b4'), (5500, 7500, '#33a02c'),
                             (7500, 9800, '#ff7f00')):
        _sel = np.flatnonzero((_mes_wave[:_n_oh] >= _lo) & (_mes_wave[:_n_oh] < _hi))
        if _sel.size >= 3:
            _mes_comps.append((f'OH {_lo}-{_hi} A', _col,
                               _oh_idx[_sel], _BASIS['oh'][_sel]))
_FIGS.append(('OH', _CTX_AIRGLOW, _mes_comps))


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(pd.Series(a[m]).rank().to_numpy(),
                             pd.Series(b[m]).rank().to_numpy())[0, 1])


_summary = []
for _fam, _feats_all, _comps in _FIGS:
    # Drop features that carry no information on these rows (e.g. `ew` is
    # constant in this corpus), otherwise their Spearman rho is NaN and sorts
    # to the top of the "strongest" ranking.
    _feats = []
    _dropped = []
    for f in _feats_all:
        if f not in _ctx_names:
            continue
        _v = _ctx[:, _ctx_names.index(f)]
        if np.isfinite(_v).sum() < 40 or np.nanstd(_v) == 0:
            _dropped.append(f)
        else:
            _feats.append(f)
    print(f'--- {_fam} ---'
          + (f'   (dropped, no variance: {", ".join(_dropped)})' if _dropped else ''))
    for _nm, _col, _idx, _bas in _comps:
        _d, _ok = _delta(_idx, _bas)
        _rho = [(abs(_spearman(_ctx[:, _ctx_names.index(f)], _d)), f) for f in _feats]
        _rho = [(r, f) for r, f in _rho if np.isfinite(r)]
        _rho.sort(reverse=True)
        if not _rho:
            _rho = [(float('nan'), 'n/a')]
        # MAD, and the moon-up / moon-down split, are printed because MEDIAN +
        # STD cannot see the thing this cell exists to measure.  The median is
        # a BIAS and sits near 0 whether the scatter is good or bad; the std is
        # dominated by moon-down rows, where the moon sits at the share floor
        # and log-amplitude ratios explode -- measured 0.649 there against
        # 0.038 moon-up, a 17x difference.  Adding the moon-model ctx feature
        # cut the moon-up MAD from 0.01492 to 0.00760 (-49%) and lifted the
        # naive-baseline gain from +24.3% to +37.3%, while the numbers this
        # line used to print moved from median +0.0043 -> -0.0002 and std
        # 0.4593 -> 0.4415, i.e. looked like noise.
        _mad = lambda _v: float(np.nanmedian(np.abs(_v - np.nanmedian(_v))))
        _split = ''
        if _MOON_UP_TEST is not None:
            # Index the FULL-length arrays.  `_d` is _te.size long with NaN
            # where the row is unusable and `_mad` is nan-aware, so the mask
            # must stay _te.size too -- subsetting it by `_ok` first silently
            # works whenever every row is usable (moon, zodi, HO2 here) and
            # then fails on the first component that drops one (FeO drops a
            # single row whose fitted amplitude is non-positive).
            _u = np.asarray(_MOON_UP_TEST, dtype=bool)
            if _u.size != _d.size:
                print(f'    (moon-up split skipped for {_nm}: mask is '
                      f'{_u.size} rows against {_d.size} test rows)')
            elif (_u & _ok).any() and ((~_u) & _ok).any():
                _split = (f'   [moon-up n={int((_u & _ok).sum())} '
                          f'MAD {_mad(_d[_u]):.5f}'
                          f' | moon-down n={int(((~_u) & _ok).sum())} '
                          f'MAD {_mad(_d[~_u]):.5f}]')
        print(f'  {_nm:<20s} rows {int(_ok.sum()):>5d}  median Delta '
              f'{np.nanmedian(_d):+.4f} dex  MAD {_mad(_d):.5f}  '
              f'std {np.nanstd(_d):.4f}  '
              f'strongest |rho|: {_rho[0][1]} {_rho[0][0]:.3f}{_split}')
        _summary.append(dict(family=_fam, component=_nm,
                             mad=_mad(_d),
                             median=float(np.nanmedian(_d)),
                             std=float(np.nanstd(_d)),
                             top_feature=_rho[0][1], top_rho=float(_rho[0][0])))

    def _rgba(_hex, _a):
        """'#rrggbb' -> 'rgba(r,g,b,a)' so the band can be translucent."""
        _h = str(_hex).lstrip('#')
        return (f'rgba({int(_h[0:2], 16)},{int(_h[2:4], 16)},'
                f'{int(_h[4:6], 16)},{_a})')

    _nrow = int(np.ceil(len(_feats) / 3))
    _fig = make_subplots(rows=_nrow, cols=3, subplot_titles=_feats,
                         vertical_spacing=0.09, horizontal_spacing=0.07)
    for _ci, (_nm, _col, _idx, _bas) in enumerate(_comps):
        _d, _ok = _delta(_idx, _bas)
        for _k, _f in enumerate(_feats):
            _x = _ctx[:, _ctx_names.index(_f)]
            _m = np.isfinite(_x) & np.isfinite(_d)
            if _m.sum() < 40:
                continue
            _qs = np.unique(np.nanpercentile(_x[_m], np.linspace(0, 100, N_BINS + 1)))
            _cx, _md, _plo, _phi = [], [], [], []
            for _b in range(len(_qs) - 1):
                _sel = _m & (_x >= _qs[_b]) & (_x <= _qs[_b + 1])
                if _sel.sum() < 8:
                    continue
                _cx.append(float(np.median(_x[_sel])))
                _md.append(float(np.median(_d[_sel])))
                # +/-1 SIGMA RANGE of the rows in the bin (16th-84th
                # percentile), not a standard error on the median.  The median
                # line alone hides the thing these panels exist to show: a
                # family can be unbiased in every bin and still be wildly
                # uncertain in all of them, which is exactly the moon's
                # situation in dark time.
                #
                # PERCENTILES, not mean +/- std: these distributions have heavy
                # tails -- FeO's std is 0.9 dex against a MAD of 0.038 -- so a
                # std-based band is set by a handful of rows.  16/84 is the
                # Gaussian-equivalent 1 sigma, and on a heavy-tailed sample it
                # tracks the bulk instead.
                _plo.append(float(np.nanpercentile(_d[_sel], 15.865)))
                _phi.append(float(np.nanpercentile(_d[_sel], 84.135)))
            _r, _c = _k // 3 + 1, _k % 3 + 1
            if len(_cx) > 1:
                _fig.add_trace(go.Scatter(
                    x=_cx + _cx[::-1], y=_phi + _plo[::-1],
                    fill='toself', fillcolor=_rgba(_col, 0.13),
                    line=dict(width=0), mode='lines',
                    name=f'{_nm} +/-1 sigma', legendgroup=_nm, showlegend=False,
                    hoverinfo='skip'),
                    row=_r, col=_c)
            _fig.add_trace(go.Scatter(
                x=_cx, y=_md, mode='lines+markers', name=_nm,
                legendgroup=_nm, showlegend=(_k == 0),
                line=dict(color=_col, width=1.6),
                marker=dict(size=5),
                hovertemplate=(f'{_nm}<br>{_f}=%{{x:.4g}}<br>'
                               'median Delta=%{y:+.4f} dex<extra></extra>')),
                row=_r, col=_c)
            if _ci == 0:
                _fig.add_hline(y=0.0, line=dict(color='black', width=0.8,
                                                dash='dash'), row=_r, col=_c)
    _fig.update_layout(
        template='plotly_white', height=240 * _nrow + 130,
        legend=dict(orientation='h', yanchor='bottom', y=1.03,
                    xanchor='left', x=0.0, font=dict(size=10)),
        title=dict(text=(f'{_fam}: signed amplitude error vs its own context '
                         f'(median per quantile bin, {N_BINS} bins)<br>'
                         f'<sub>A = sum_lambda f(lambda) = c . B.sum(axis=1); '
                         f'Delta = log10(A_pred/A_true), + = over-predicted. '
                         f'Flat at zero = brightness transferred correctly; '
                         f'the band is the +/-1 sigma (16-84 pct) range of '
                         f'rows in the bin, so a '
                         f'flat line inside a wide band means unbiased but '
                         f'uncertain.</sub>'),
                   font=dict(size=13), x=0.02, xanchor='left'),
        margin=dict(t=150))
    _fig.update_yaxes(title_text='Delta [dex]', row=1, col=1)
    _fig.show()

amplitude_error_vs_ctx_result = dict(summary=pd.DataFrame(_summary))
