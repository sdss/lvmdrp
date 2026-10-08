# Residual -> context regression per group (missing-feature detector).
#
# For each group g:
#   1. Compute per-row per-group residual RMSE on the test split:
#          r_g(row) = sqrt(mean_{j in g} (pred[row, j] - true[row, j])**2)
#   2. Fit a linear regression on r_g(row) ~ ctx_sci features.
#      Report 5-fold CV R^2 and the top-|coef| features (standardised X).
#   3. Fit a random forest for non-linear structure; report CV R^2 and top
#      permutation importances.
#
# What the numbers mean:
#   linear R^2 > 0.1  -> ctx features carry recoverable signal the head is
#                        not modelling; broaden its input set or capacity.
#   RF R^2 - lin R^2 > 0.1 -> the residual is non-linearly related to
#                        an existing feature; the head is under-parameterised
#                        for that variable.
#   both near 0        -> residual is either irreducible physics (gravity
#                        waves, per-night atmospheric variability) or living
#                        in a non-context space (row-to-row time correlation,
#                        input noise).
#
# Requires sklearn (usually available in this env).

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

required = ['filtered_triplet', '_group_indices_compress',
            'coef_sci_all', 'ctx_sci_all', 'test_idx', 'coef_pred_det']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))

_te = np.asarray(test_idx, dtype=int)
_ctx_names = list(filtered_triplet['ctx_names'])
_X_full = np.asarray(ctx_sci_all[_te], dtype=np.float64)
_y_true = np.asarray(coef_sci_all[_te], dtype=np.float64)
_y_pred = np.asarray(coef_pred_det, dtype=np.float64)

# --- DIAGNOSTIC-ONLY probe features: helio-relative ecliptic longitude -------
# The model is never given lambda - lambda_sun.  `zodi_ctx_restriction` carries
# absolute ecl_lon_sin/cos but neither lambda_sun nor the obstime_year_* terms,
# so the zodi branch cannot construct the helio-relative angle that the Leinert
# lookup is actually a function of -- it only gets the coarse table output
# `zodi_log10_v`.  `ecl_lon_sin` has been the #1 RF importance on the zodi
# residual across every configuration tried, which is what a missing feature
# looks like.  But it could equally be a seasonal / observing-pattern proxy,
# since LVM visits given fields at given times of year.
#
# Appending helio_lon here (RF input only -- the network is NOT changed and no
# retrain is implied) separates the two:
#   helio_lon outranks ecl_lon      -> ecl_lon was a proxy for the real angle;
#                                      adding the feature to the model is worth
#                                      a training run.
#   ecl_lon still outranks helio_lon -> seasonal confound; adding the feature
#                                      will not help, run saved.
_PROBE_NAMES = []
if all(_n in _ctx_names for _n in ('ecl_lon_sin', 'ecl_lon_cos')):
    _lon_s = _X_full[:, _ctx_names.index('ecl_lon_sin')]
    _lon_c = _X_full[:, _ctx_names.index('ecl_lon_cos')]
    _lon_deg = np.rad2deg(np.arctan2(_lon_s, _lon_c)) % 360.0
    _sun_lon_deg = _sun_ecliptic_longitude_deg(
        np.asarray(filtered_triplet['obstime_mjd'], dtype=np.float64)[_te])
    # Same convention as data._augment_triplet_with_physics_priors (-180, 180].
    _helio_lon_deg = ((_lon_deg - _sun_lon_deg + 180.0) % 360.0) - 180.0
    _helio_rad = np.deg2rad(_helio_lon_deg)
    _probe = np.column_stack([np.sin(_helio_rad), np.cos(_helio_rad),
                              np.abs(_helio_lon_deg)])
    # |helio_lon| is the actual axis of the Leinert table (it is symmetric about
    # the sun), so it is included alongside the cyclic pair.
    _PROBE_NAMES = ['helio_lon_sin*', 'helio_lon_cos*', 'abs_helio_lon_deg*']
    _X_full = np.column_stack([_X_full, _probe])
    _ctx_names = _ctx_names + _PROBE_NAMES
    print(f'  probe features appended (diagnostic only, * suffix): '
          f'{", ".join(_PROBE_NAMES)}')
    print(f'    helio_lon range [{_helio_lon_deg.min():.1f}, '
          f'{_helio_lon_deg.max():.1f}] deg, '
          f'|corr(helio_lon_sin, ecl_lon_sin)| = '
          f'{abs(np.corrcoef(np.sin(_helio_rad), _lon_s)[0, 1]):.3f}')
else:
    print('  probe features SKIPPED (ecl_lon_sin/cos not in ctx)')

# Drop constant / near-constant columns (would trip the standardiser).
_std_x = np.nanstd(_X_full, axis=0)
_keep_cols = np.where(_std_x > 1e-12)[0]
if len(_keep_cols) < len(_std_x):
    print(f'  dropping {len(_std_x) - len(_keep_cols)} constant ctx columns')
_X = _X_full[:, _keep_cols]
_ctx_names_k = [_ctx_names[i] for i in _keep_cols]

_kf = KFold(n_splits=5, shuffle=True, random_state=42)


def _cv_r2(_model, X, y):
    return float(np.mean(cross_val_score(_model, X, y, cv=_kf, scoring='r2', n_jobs=1)))


print('=' * 90)
print('Residual -> context regression per group (test split, 5-fold CV R^2)')
print('=' * 90)

_scaler = StandardScaler()
_X_std = _scaler.fit_transform(_X)

_summary = []
for gname, idx in _group_indices_compress.items():
    idx = np.asarray(idx, dtype=int)
    _r_g = np.sqrt(np.mean((_y_pred[:, idx] - _y_true[:, idx]) ** 2, axis=1))
    _finite = np.isfinite(_r_g)
    if _finite.sum() < 20:
        print(f'  {gname}: too few finite residuals; skipping')
        continue
    _yy = _r_g[_finite]
    _XX = _X_std[_finite]

    _lin = LinearRegression()
    _lin_r2 = _cv_r2(_lin, _XX, _yy)
    _lin.fit(_XX, _yy)
    _lin_coefs = _lin.coef_
    _lin_top = np.argsort(np.abs(_lin_coefs))[::-1][:5]

    _rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    _rf_r2 = _cv_r2(_rf, _XX, _yy)
    _rf.fit(_XX, _yy)
    _rf_top = np.argsort(_rf.feature_importances_)[::-1][:5]

    print()
    print(f'--- group: {gname}  (n_valid_rows = {int(_finite.sum())})  '
          f'residual mean = {float(np.mean(_yy)):.4g}  median = {float(np.median(_yy)):.4g}')
    print(f'  LinearRegression  CV R^2 = {_lin_r2:+.3f}')
    print('    top |coef| (standardised X):')
    for _i in _lin_top:
        print(f'      {_ctx_names_k[_i]:<22s} coef = {_lin_coefs[_i]:+.3g}')
    print(f'  RandomForest      CV R^2 = {_rf_r2:+.3f}   '
          f'(non-linear gain = {_rf_r2 - _lin_r2:+.3f})')
    print('    top importances:')
    for _i in _rf_top:
        print(f'      {_ctx_names_k[_i]:<22s} imp  = {_rf.feature_importances_[_i]:.3g}')

    # Head-to-head: absolute vs helio-relative ecliptic longitude.
    _fam_ecl = _fam_helio = np.nan
    if _PROBE_NAMES:
        def _fam_imp(_names):
            return float(sum(_rf.feature_importances_[_ctx_names_k.index(_n)]
                             for _n in _names if _n in _ctx_names_k))
        _fam_ecl = _fam_imp(['ecl_lon_sin', 'ecl_lon_cos'])
        _fam_helio = _fam_imp(_PROBE_NAMES)
        _verdict = ('helio WINS' if _fam_helio > 1.2 * _fam_ecl else
                    'ecl WINS' if _fam_ecl > 1.2 * _fam_helio else 'tied')
        print(f'    ecl_lon family imp = {_fam_ecl:.3g}   '
              f'helio_lon family imp = {_fam_helio:.3g}   -> {_verdict}')

    _summary.append({
        'group': gname,
        'n': int(_finite.sum()),
        'residual_mean': float(np.mean(_yy)),
        'lin_R2': _lin_r2,
        'rf_R2':  _rf_r2,
        'nonlin_gain': _rf_r2 - _lin_r2,
        'imp_ecl_lon': _fam_ecl,
        'imp_helio_lon': _fam_helio,
    })

if _summary:
    print()
    print('Summary:')
    print(pd.DataFrame(_summary).to_string(index=False,
          float_format=lambda v: f'{v:.4g}'))
    print()
    print('Interpretation:')
    print('  lin_R2 > 0.10        -> ctx features linearly predict residual; head '
          'is under-using them.  Retrain with a wider trunk or head, or '
          'reconsider the compressor for that group.')
    print('  nonlin_gain > 0.10   -> RF finds structure that the linear model '
          'does not; head is under-parameterised for that feature (or the '
          'feature enters non-linearly, e.g. sign or threshold).')
    print('  both < 0.02          -> residual is irreducible from ctx.  Look at '
          'sky_arm_disagreement_floor() to check the noise floor and '
          'wavelength_residual_atlas() to see where the failure lives.')

    if _PROBE_NAMES:
        _zrow = [_r for _r in _summary if _r['group'] == 'zodi']
        print()
        print('helio-relative longitude probe (diagnostic only; the model was '
              'NOT given these):')
        if _zrow:
            _e, _h = _zrow[0]['imp_ecl_lon'], _zrow[0]['imp_helio_lon']
            print(f'  zodi: ecl_lon family = {_e:.3g}, helio_lon family = {_h:.3g}')
            if _h > 1.2 * _e:
                print('  GO: the helio-relative angle outranks absolute ecliptic '
                      'longitude, so ecl_lon was standing in for it.  Adding '
                      'helio_lon_sin/cos per arm in '
                      '_augment_triplet_with_physics_priors and routing them '
                      'into zodi_ctx_restriction + moon_zodi_ctx_restriction is '
                      'worth a training run.')
            elif _e > 1.2 * _h:
                print('  NO-GO: absolute ecliptic longitude still outranks the '
                      'helio-relative angle, so the zodi residual is tracking '
                      'season / observing pattern rather than zodiacal geometry.  '
                      'Adding the feature is unlikely to help; spend the run '
                      'elsewhere.')
            else:
                print('  INCONCLUSIVE: the two families rank within 20% of each '
                      'other.  The probe does not settle it; weigh the cost of '
                      'the training run against the small remaining zodi headroom '
                      '(err/delta is already at or below the sky-arm floor).')
        print('  NB RF importance on residual MAGNITUDE is partly amplitude-driven.  '
              'Read it alongside zodi_log10_v (the Leinert amplitude proxy the '
              'branch already has): if that ranks low while a longitude feature '
              'ranks high, the ranking is not simply tracking zodi brightness.')
