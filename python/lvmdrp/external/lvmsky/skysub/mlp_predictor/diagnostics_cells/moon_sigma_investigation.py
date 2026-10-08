# Investigation: why do so few moon coefficients have SNR > 1?
# Break down per-row median |coef_moon| vs median sigma_moon by moon
# state (alt+phase) so we can see whether it is (a) sigma always big,
# (b) moon coefs always small (spline split across 29 knots), or
# (c) both.
import numpy as np
import pandas as pd

required = ['coef_sci_all', 'coef_err_sci_all', 'ctx_sci_all',
            '_group_indices_compress', 'filtered_triplet']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Missing kernel state: ' + ', '.join(_missing))

_gidx_moon = np.asarray(_group_indices_compress['moon'], dtype=int)
_coef_names_ss = [str(n) for n in filtered_triplet['coef_names']]
print(f"Moon block: {_gidx_moon.size} coefs "
      f"({_coef_names_ss[_gidx_moon[0]]!r} ... {_coef_names_ss[_gidx_moon[-1]]!r})")

_ctx_names_ss = list(filtered_triplet['ctx_names'])
_ma_idx = _ctx_names_ss.index('moon_alt')
_ma = np.asarray(ctx_sci_all[:, _ma_idx], dtype=np.float64)

# moon_fli from ctx if physics augment added it, else derive from moon_phase
if 'moon_fli' in _ctx_names_ss:
    _fli = np.asarray(ctx_sci_all[:, _ctx_names_ss.index('moon_fli')],
                      dtype=np.float64)
elif ('moon_phase_sin' in _ctx_names_ss
      and 'moon_phase_cos' in _ctx_names_ss):
    _pc = np.asarray(ctx_sci_all[:, _ctx_names_ss.index('moon_phase_cos')],
                     dtype=np.float64)
    _fli = np.clip(0.5 * (1.0 - _pc), 0.0, 1.0)
else:
    raise RuntimeError('No moon_fli or moon_phase in ctx.')

_c = np.asarray(coef_sci_all[:, _gidx_moon], dtype=np.float64)
_s = np.asarray(coef_err_sci_all[:, _gidx_moon], dtype=np.float64)
_c_abs = np.abs(_c)

# Per-row summary stats (median across the 29 knots).
def _row_med_finite(arr):
    _mask = np.isfinite(arr) & (arr > 0.0)
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        m = _mask[i]
        if m.any():
            out[i] = np.median(arr[i, m])
    return out

_med_c = _row_med_finite(_c_abs)
_med_s = _row_med_finite(_s)
_max_c = np.nanmax(_c_abs, axis=1)
_min_s = np.nanmin(np.where(np.isfinite(_s) & (_s > 0), _s, np.inf), axis=1)
_min_s = np.where(np.isfinite(_min_s), _min_s, np.nan)

# Per-row SNR of the row's PEAK coef (a bright-moon knot should carry
# most of the amplitude; median-SNR under-represents that because it
# averages a bright peak with 28 shoulder knots).
_snr_med = _med_c / _med_s
_snr_peak = _max_c / _min_s

# Bin by moon state.
_bins = [
    ('moon down (alt<=0)',       (_ma <= 0.0)),
    ('moon up, fli<0.2',         (_ma > 0.0) & (_fli < 0.2)),
    ('moon up, 0.2<=fli<0.5',    (_ma > 0.0) & (_fli >= 0.2) & (_fli < 0.5)),
    ('moon up, 0.5<=fli<0.8',    (_ma > 0.0) & (_fli >= 0.5) & (_fli < 0.8)),
    ('moon up, fli>=0.8',        (_ma > 0.0) & (_fli >= 0.8)),
    ('moon up, alt>30, fli>0.85',(_ma > 30.0) & (_fli > 0.85)),
]

_rows = []
for _label, _mask in _bins:
    _n = int(_mask.sum())
    if _n == 0:
        _rows.append(dict(bin=_label, n=0))
        continue
    _rows.append(dict(
        bin=_label, n=_n,
        med_coef_med=float(np.nanmedian(_med_c[_mask])),
        med_sigma_med=float(np.nanmedian(_med_s[_mask])),
        med_snr_med=float(np.nanmedian(_snr_med[_mask])),
        med_coef_peak=float(np.nanmedian(_max_c[_mask])),
        med_sigma_min=float(np.nanmedian(_min_s[_mask])),
        med_snr_peak=float(np.nanmedian(_snr_peak[_mask])),
    ))

_df = pd.DataFrame(_rows)
print()
print("Per-row moon block: coef and sigma stats binned by moon state")
print("  med_coef_med   = median|coef| across 29 knots, per row, then binned median")
print("  med_sigma_med  = median sigma across 29 knots, per row, then binned median")
print("  med_snr_med    = med_coef_med / med_sigma_med per row, then binned median")
print("  med_coef_peak  = max|coef|  across 29 knots, per row, then binned median")
print("  med_sigma_min  = min(sigma) across 29 knots, per row, then binned median")
print("  med_snr_peak   = max|coef| / min(sigma) per row, then binned median")
print()
_cols = ['bin', 'n', 'med_coef_med', 'med_sigma_med', 'med_snr_med',
         'med_coef_peak', 'med_sigma_min', 'med_snr_peak']
_fmt = {c: (lambda v: f'{v:.4g}') for c in _cols if c not in ('bin', 'n')}
_fmt['n'] = (lambda v: f'{int(v):,}')
with pd.option_context('display.width', 220):
    print(_df[_cols].to_string(index=False, formatters=_fmt))

# Individual coefficient scatter: for the BRIGHTEST-MOON bin, dump the
# per-knot |coef| / sigma histogram so we can see whether ANY knots
# individually clear SNR > 1 there.
print()
_bright_mask = (_ma > 30.0) & (_fli > 0.85)
if _bright_mask.any():
    print(f"Brightest-moon bin (moon_alt>30 & fli>0.85, n={int(_bright_mask.sum())}):")
    _c_b = _c_abs[_bright_mask]
    _s_b = _s[_bright_mask]
    _snr_flat = _c_b / np.where(_s_b > 0, _s_b, np.nan)
    _snr_flat = _snr_flat[np.isfinite(_snr_flat)]
    print(f"  per-coef SNR quantiles (n={_snr_flat.size:,}):")
    print(f"     p5={np.percentile(_snr_flat, 5):.3f}   "
          f"p25={np.percentile(_snr_flat, 25):.3f}   "
          f"p50={np.percentile(_snr_flat, 50):.3f}   "
          f"p75={np.percentile(_snr_flat, 75):.3f}   "
          f"p95={np.percentile(_snr_flat, 95):.3f}   "
          f"max={_snr_flat.max():.3f}")
    _n_snr1 = int((_snr_flat > 1.0).sum())
    _n_snr3 = int((_snr_flat > 3.0).sum())
    print(f"  SNR > 1 count: {_n_snr1:,} / {_snr_flat.size:,} "
          f"({100*_n_snr1/_snr_flat.size:.2f}%)")
    print(f"  SNR > 3 count: {_n_snr3:,} / {_snr_flat.size:,} "
          f"({100*_n_snr3/_snr_flat.size:.2f}%)")

    # Which KNOTS carry the SNR? Aggregate SNR by knot index within the block.
    _snr_by_knot = np.full(_gidx_moon.size, np.nan)
    for _k in range(_gidx_moon.size):
        _sn = (_c_b[:, _k] / np.where(_s_b[:, _k] > 0, _s_b[:, _k], np.nan))
        _sn = _sn[np.isfinite(_sn)]
        if _sn.size:
            _snr_by_knot[_k] = float(np.median(_sn))
    print()
    print("  median SNR per knot (0..28, brightest-moon bin):")
    for _k in range(0, _gidx_moon.size, 4):
        _slice = _snr_by_knot[_k:_k + 4]
        _idx_slice = list(range(_k, min(_k + 4, _gidx_moon.size)))
        print("     " + "  ".join(
            f"[{_i:2d}]={_v:.3f}" for _i, _v in zip(_idx_slice, _slice)
            if np.isfinite(_v)))


# ---- (2) Adjacent-knot correlation check ------------------------------
# If B-spline neighbors are near-collinear, coef_j and coef_{j+1} will be
# highly (positively OR negatively) correlated across rows, and the
# marginal sigma from fit.py's diag(H^-1) will vastly overstate the
# per-coef uncertainty while the joint amplitude stays well-constrained.
_bright = (_ma > 30.0) & (_fli > 0.85)
if _bright.sum() >= 30:
    _cb = _c[_bright]                  # (n_bright, 29) signed coefs
    _corrs = np.full(_gidx_moon.size - 1, np.nan)
    for _k in range(_gidx_moon.size - 1):
        _a = _cb[:, _k]; _b = _cb[:, _k + 1]
        _m = np.isfinite(_a) & np.isfinite(_b)
        if _m.sum() >= 30 and _a[_m].std() > 0 and _b[_m].std() > 0:
            _corrs[_k] = float(np.corrcoef(_a[_m], _b[_m])[0, 1])
    print()
    print(f"Adjacent-knot correlation across n={int(_bright.sum())} brightest-moon rows:")
    print(f"  median |corr| = {np.nanmedian(np.abs(_corrs)):.3f}")
    print(f"  n with |corr| > 0.90: {int(np.sum(np.abs(_corrs) > 0.90))} / {_corrs.size}")
    print(f"  n with |corr| > 0.99: {int(np.sum(np.abs(_corrs) > 0.99))} / {_corrs.size}")
    print(f"  per-pair corr(coef_{{k}}, coef_{{k+1}}):")
    for _k in range(0, _corrs.size, 4):
        _slice = _corrs[_k:_k + 4]
        _idx_slice = list(range(_k, min(_k + 4, _corrs.size)))
        print("     " + "  ".join(
            f"[{_i:2d}↔{_i+1:2d}]={_v:+.2f}"
            for _i, _v in zip(_idx_slice, _slice)
            if np.isfinite(_v)))

    # Joint amplitude uncertainty: fit a scalar amplitude across all 29 knots
    # per row, sigma of that scalar << typical per-knot sigma if the fit is
    # really much tighter jointly than marginally.
    _row_amp = np.linalg.norm(_cb, axis=1)          # ||coef|| per row (L2 norm)
    _sig_b = _s[_bright]
    _row_sig = np.sqrt(np.nansum(_sig_b ** 2, axis=1))  # if independent
    print()
    print(f"Row-level amplitude vs propagated sigma (assuming independence):")
    print(f"  ||coef||  quartiles: "
          f"p25={np.nanpercentile(_row_amp, 25):.3f}, "
          f"p50={np.nanpercentile(_row_amp, 50):.3f}, "
          f"p75={np.nanpercentile(_row_amp, 75):.3f}")
    print(f"  sqrt(sum sigma^2) quartiles: "
          f"p25={np.nanpercentile(_row_sig, 25):.3f}, "
          f"p50={np.nanpercentile(_row_sig, 50):.3f}, "
          f"p75={np.nanpercentile(_row_sig, 75):.3f}")
    _amp_snr = _row_amp / _row_sig
    print(f"  ||coef|| / sqrt(sum sigma^2)  quartiles: "
          f"p25={np.nanpercentile(_amp_snr, 25):.3f}, "
          f"p50={np.nanpercentile(_amp_snr, 50):.3f}, "
          f"p75={np.nanpercentile(_amp_snr, 75):.3f}")
    print("  (independence-assuming SNR; TRUE joint SNR is higher when neighbors are correlated.)")


# Joint SNR = sqrt(c^T Sigma^-1 c / n_active) per row -- uses the persisted
# COEF_COV_MOON block. This is the CORRECT SNR that captures the near-
# collinearity of adjacent knots (whereas ||coef||/sqrt(Sigma sigma^2)
# above assumes independence and therefore UNDER-estimates the SNR).
_cov_moon_here = filtered_triplet.get('coef_cov_moon_sci')
if _cov_moon_here is not None:
    _c_moon = np.asarray(coef_sci_all[:, _gidx_moon], dtype=np.float64)
    _snr_joint = np.full(_c_moon.shape[0], np.nan, dtype=np.float64)
    for _i in range(_c_moon.shape[0]):
        _cvec = _c_moon[_i]
        _cmat = np.asarray(_cov_moon_here[_i], dtype=np.float64)
        _good = np.isfinite(_cvec) & np.isfinite(np.diagonal(_cmat)) & (np.diagonal(_cmat) > 0)
        if _good.sum() < 2:
            continue
        _cg = _cvec[_good]
        _Cg = _cmat[np.ix_(_good, _good)]
        _Cg = 0.5 * (_Cg + _Cg.T) + 1e-24 * np.eye(_Cg.shape[0])
        try:
            _q2 = float(_cg @ np.linalg.solve(_Cg, _cg))
        except np.linalg.LinAlgError:
            continue
        _snr_joint[_i] = np.sqrt(max(_q2, 0.0) / _good.sum())

    print()
    print('Joint SNR from COEF_COV_MOON per row (correct, not independence-assuming):')
    print('  snr_joint = sqrt(c^T Sigma^-1 c / n_active_knots)  per row, then binned median')
    _rows_joint = []
    for _label, _mask in _bins:
        _finite = np.isfinite(_snr_joint) & _mask
        if _finite.sum() == 0:
            continue
        _rows_joint.append(dict(
            bin=_label, n=int(_finite.sum()),
            med_snr_joint=float(np.nanmedian(_snr_joint[_finite])),
        ))
    _df_joint = pd.DataFrame(_rows_joint)
    if not _df_joint.empty:
        with pd.option_context('display.width', 180):
            print(_df_joint.to_string(index=False,
                                       formatters={'med_snr_joint': (lambda v: f'{v:.3f}')}))