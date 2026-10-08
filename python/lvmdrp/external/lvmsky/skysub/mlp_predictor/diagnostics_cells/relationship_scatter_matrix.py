# Enhanced coefficient-vs-context scatter matrix (2026-08-19 replacement of the
# 2026-08-10 raw-coefficient version). Improvements (1-10 from the discussion of
# "hidden correlations in raw coefficients"):
#   (1) shape rows (log_amp, red/blue tilt, PC1, PC2) for the Moon_bs spline and
#       the OH block replace the amplitude-dominated `median` aggregates so
#       zodi- / shape-scale signals stop being drowned by the common brightness
#       scalar;
#   (2) Y_TRANSFORM='log' switches raw-coefficient rows to log10; shape rows are
#       already dimensionless;
#   (3) Y_TRANSFORM='geom' divides coefficients by airglow_geometry_scale per
#       arm (same path the loader feeds the encoder);
#   (4) Y_TRANSFORM='residual' regresses y on [moon_alt, moon_sep, airmass,
#       sun_alt, moon_phase] fit on train rows and plots the residual (partial-
#       correlation panel) -- this is the transform that exposed the ecl->
#       Moon_bs signal invisible on raw coefficients;
#   (5) three extra ecliptic display columns (|ecl_beta|, cos(ecl_beta), signed
#       ecl_beta) appended to the context axis so the zodi-relevant projections
#       are directly on the grid;
#   (6) cyclic sin/cos pairs still decoded to degrees for readability;
#   (7) per-panel Pearson r annotation for the STAT_SERIES ('sci_true' default);
#   (8) 95% Fisher CI on r and sample size n annotated inline;
#   (9) partial-r shown when Y_TRANSFORM='residual' (the residualisation happens
#       upstream of the correlation, so the annotated `r` IS the partial r);
#  (10) Spearman rho and non-linearity indicator delta = rho^2 - r^2 annotated
#       alongside r so monotone-but-nonlinear trends stand out. Bolded when
#       |r|>0.15 or |rho|>0.15.
#
# Toggle Y_TRANSFORM at the top of the cell; the four modes are:
#   'raw'      -- coefficients as-fitted (matches the 2026-08-10 version).
#   'log'      -- log10(clip(coef, floor, None)); makes multiplicative scaling
#                 visible.
#   'geom'     -- coef / airglow_geometry_scale(ctx_arm, **compress_geom_kwargs);
#                 removes van-Rhijn * extinction so residual scatter is
#                 intrinsic emissivity.
#   'residual' -- nuisance-regressed partial residual (train-fit LSQ).
import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from IPython.display import Image, display

required = ["mlp_artifacts", "filtered_triplet", "predict_sci_coefficients_default",
            "_decode_cyclic_context"]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError("Run training + prediction cells first. Missing: "
                       + ", ".join(missing))

# --- Configuration -------------------------------------------------------
Y_TRANSFORM = 'residual'  # one of: 'raw', 'log', 'geom', 'residual'
STAT_SERIES = 'sci_true'  # which series drives the per-panel r/rho annotation
_MAX_POINTS = 3000        # scatter markers per panel; stats always use all rows
_TREND_ONLY_STAT = True   # draw the black running-median only for STAT_SERIES

_Y_LABEL = {
    'raw': 'raw coef',
    'log': 'log10(coef)',
    'geom': 'coef / airglow_geometry_scale',
    'residual': 'partial residual (train-fit LSQ vs moon geom)',
}[Y_TRANSFORM]

# --- Load per-arm coefficient + context matrices --------------------------
coef_near_all = np.asarray(filtered_triplet["coef_near"], dtype=np.float64)
coef_far_all = np.asarray(filtered_triplet["coef_far"], dtype=np.float64)
coef_sci_all = np.asarray(filtered_triplet["coef_sci"], dtype=np.float64)
ctx_near_all = np.asarray(filtered_triplet["ctx_near"], dtype=np.float32)
ctx_far_all = np.asarray(filtered_triplet["ctx_far"], dtype=np.float32)
ctx_sci_all = np.asarray(filtered_triplet["ctx_sci"], dtype=np.float32)
coef_names_all = [str(n) for n in filtered_triplet["coef_names"]]
ctx_names_all = [str(n) for n in filtered_triplet["ctx_names"]]
n_all = coef_sci_all.shape[0]

coef_pred_all = predict_sci_coefficients_default(
    mlp_artifacts,
    coef_near_phys=coef_near_all.astype(np.float32),
    coef_far_phys=coef_far_all.astype(np.float32),
    ctx_near_phys=ctx_near_all,
    ctx_far_phys=ctx_far_all,
    ctx_sci_phys=ctx_sci_all,
).astype(np.float64)

# --- (3) Geometry-normalise if requested --------------------------------
def _geom_scale_or_ones(ctx):
    if 'airglow_geometry_scale' not in globals() or 'compress_geom_kwargs' not in globals():
        return None
    try:
        return airglow_geometry_scale(np.asarray(ctx, dtype=np.float64),
                                      **compress_geom_kwargs)
    except (KeyError, ValueError) as exc:
        print(f'  geometry-scale unavailable ({type(exc).__name__}: {exc}); '
              'geom transform will pass through unchanged.')
        return None

_apply_geom = (Y_TRANSFORM == 'geom')
if _apply_geom:
    _scale_near = _geom_scale_or_ones(ctx_near_all)
    _scale_far = _geom_scale_or_ones(ctx_far_all)
    _scale_sci = _geom_scale_or_ones(ctx_sci_all)
    _apply_geom = _scale_near is not None and _scale_far is not None and _scale_sci is not None

if _apply_geom:
    coef_near_use = coef_near_all / np.where(_scale_near > 0, _scale_near, 1.0)
    coef_far_use = coef_far_all / np.where(_scale_far > 0, _scale_far, 1.0)
    coef_sci_use = coef_sci_all / np.where(_scale_sci > 0, _scale_sci, 1.0)
    coef_pred_use = coef_pred_all / np.where(_scale_sci > 0, _scale_sci, 1.0)
elif Y_TRANSFORM == 'log':
    _floor = 1e-6
    coef_near_use = np.log10(np.clip(coef_near_all, _floor, None))
    coef_far_use = np.log10(np.clip(coef_far_all, _floor, None))
    coef_sci_use = np.log10(np.clip(coef_sci_all, _floor, None))
    coef_pred_use = np.log10(np.clip(coef_pred_all, _floor, None))
else:
    coef_near_use = coef_near_all
    coef_far_use = coef_far_all
    coef_sci_use = coef_sci_all
    coef_pred_use = coef_pred_all

# --- (6) Cyclic pairs -> degrees --------------------------------------
_ctx_names_display, _ctx_near_disp = _decode_cyclic_context(ctx_names_all, ctx_near_all)
_, _ctx_far_disp = _decode_cyclic_context(ctx_names_all, ctx_far_all)
_, _ctx_sci_disp = _decode_cyclic_context(ctx_names_all, ctx_sci_all)

# --- (5) Ecliptic display column enrichment ------------------------------
def _augment_ecl_columns(names_disp, ctx_disp):
    if 'ecl_beta_deg' not in names_disp:
        return names_disp, ctx_disp
    _idx = names_disp.index('ecl_beta_deg')
    _b = np.asarray(ctx_disp[:, _idx], dtype=np.float64)
    _extra = np.column_stack([np.abs(_b), np.cos(np.deg2rad(_b))]).astype(np.float32)
    return (list(names_disp) + ['|ecl_beta|', 'cos(ecl_beta)'],
            np.concatenate([ctx_disp, _extra], axis=1))

_orig_display_names = list(_ctx_names_display)  # ecl column-add is fragile on [:-2] slice.
_ctx_names_display, _ctx_near_disp = _augment_ecl_columns(_orig_display_names, _ctx_near_disp)
_, _ctx_far_disp = _augment_ecl_columns(_orig_display_names, _ctx_far_disp)
_, _ctx_sci_disp = _augment_ecl_columns(_orig_display_names, _ctx_sci_disp)

# --- Coefficient index bookkeeping -----------------------------------
_names_lower = [n.lower() for n in coef_names_all]
_moon_idx = np.array([i for i, n in enumerate(_names_lower)
                      if n.startswith("moon_bs")], dtype=int)
_zodi_idx = np.array([i for i, n in enumerate(_names_lower)
                      if n.startswith("zodi_bs")], dtype=int)
_oh_idx = np.array([i for i, n in enumerate(_names_lower)
                    if n.startswith("oh_")], dtype=int)

# --- Train mask for PCA fit + nuisance regression --------------------
_train_idx_local = np.asarray(mlp_artifacts.get('train_idx', np.arange(n_all)),
                              dtype=int)
_train_mask = np.zeros(n_all, dtype=bool)
_train_mask[_train_idx_local[(_train_idx_local >= 0) & (_train_idx_local < n_all)]] = True
if _train_mask.sum() < 10:
    print('  train mask empty; using all rows for PCA / nuisance regression.')
    _train_mask[:] = True

# --- (1) Shape features: log-amp, red/blue tilt, PC1/PC2 -----------------
def _fit_shape_pca(block_idx, k=2):
    """PCA on training-set L2-normalised block; returns (mean, components)."""
    idx = np.asarray(block_idx, dtype=int)
    _b = np.clip(coef_sci_all[_train_mask][:, idx], 0.0, None)
    _n = np.linalg.norm(_b, axis=1, keepdims=True)
    _n = np.where(_n > 0, _n, 1.0)
    _shape = _b / _n
    _mean = _shape.mean(axis=0)
    _centered = _shape - _mean
    _u, _s, _vt = np.linalg.svd(_centered, full_matrices=False)
    return _mean, _vt[:k]


def _make_shape_functions(block_idx, tilt_frac=0.35):
    """Return dict of (label -> function(coef_mat)) producing all-row series."""
    idx = np.asarray(block_idx, dtype=int)
    if idx.size < 4:
        return {}
    lo_end = max(1, int(round(idx.size * tilt_frac)))
    hi_start = max(lo_end, int(round(idx.size * (1 - tilt_frac))))
    _mean, _components = _fit_shape_pca(idx, k=2)

    def _log_amp(coef_mat):
        _b = np.clip(coef_mat[:, idx], 0.0, None)
        return np.log10(np.clip(_b.sum(axis=1), 1e-6, None))

    def _tilt(coef_mat):
        _b = np.clip(coef_mat[:, idx], 0.0, None)
        _lo = np.clip(_b[:, :lo_end].mean(axis=1), 1e-6, None)
        _hi = np.clip(_b[:, hi_start:].mean(axis=1), 1e-6, None)
        return np.log10(_hi / _lo)

    def _pc(coef_mat, k):
        _b = np.clip(coef_mat[:, idx], 0.0, None)
        _n = np.linalg.norm(_b, axis=1, keepdims=True)
        _n = np.where(_n > 0, _n, 1.0)
        return (_b / _n - _mean) @ _components[k]

    return {
        'log_amp': _log_amp,
        'red_blue_tilt': _tilt,
        'PC1': (lambda cm: _pc(cm, 0)),
        'PC2': (lambda cm: _pc(cm, 1)),
    }


_moon_shape = _make_shape_functions(_moon_idx)
_zodi_shape = _make_shape_functions(_zodi_idx)
_oh_shape = _make_shape_functions(_oh_idx)

# Data-driven amplitude floor: NaN-out moon shape features on rows where
# sum(Moon_bs) is below 1% of the moon-up training median. Rules out the
# garbage log_amp/tilt/PC values that come from L2-normalising noise on
# moon-down rows where the block is numerically ~0.
_MOON_AMP_FLOOR = None
if _moon_idx.size > 0 and 'moon_alt' in ctx_names_all:
    _sum_all = np.clip(coef_sci_all[:, _moon_idx].sum(axis=1), 0.0, None)
    _alt_all = ctx_sci_all[:, ctx_names_all.index('moon_alt')]
    _amp_up = _sum_all[_train_mask & (_alt_all > 5.0)]
    if _amp_up.size >= 20:
        _MOON_AMP_FLOOR = 0.01 * float(np.median(_amp_up))
        print(f'  moon shape mask: rows with sum(Moon_bs) < {_MOON_AMP_FLOOR:.4g} '
              f'get NaN shape features (1% of median moon-up training amplitude).')

# --- Individual coefficient rows (unchanged from 2026-08-10) ----------
_INDIVIDUAL = [
    "HO2", "FeO", "O2Ac",
    "ATOM_K", "ATOM_Na", "ATOM_Og",
    "ATOM_N", "ATOM_Or", "ATOM_Orc_OI0777", "ATOM_Orc_OI0845",
    "O2_b01",
]


def _find_col(name):
    lname = name.lower()
    matches = [i for i, n in enumerate(_names_lower) if n == lname]
    if not matches:
        raise KeyError(f"coefficient {name!r} not in coef_names_all")
    return matches[0]


_individual_idx = {name: _find_col(name) for name in _INDIVIDUAL}

# --- Row definitions -------------------------------------------------
_ROW_DEFS = []
if _moon_shape:
    _ROW_DEFS += [
        (f'moon log_amp (n={_moon_idx.size})', 'shape_moon_log_amp'),
        ('moon red/blue tilt', 'shape_moon_red_blue_tilt'),
        ('moon PC1', 'shape_moon_PC1'),
        ('moon PC2', 'shape_moon_PC2'),
    ]
if _zodi_shape:
    _ROW_DEFS += [
        (f'zodi log_amp (n={_zodi_idx.size})', 'shape_zodi_log_amp'),
        ('zodi red/blue tilt', 'shape_zodi_red_blue_tilt'),
        ('zodi PC1', 'shape_zodi_PC1'),
        ('zodi PC2', 'shape_zodi_PC2'),
    ]
if _oh_shape:
    _ROW_DEFS += [
        (f'OH log_amp (n={_oh_idx.size})', 'shape_oh_log_amp'),
        ('OH PC1', 'shape_oh_PC1'),
        ('OH PC2', 'shape_oh_PC2'),
    ]
for _n in _INDIVIDUAL:
    _ROW_DEFS.append((_n, _n))


# Raw coef matrices per series so shape features (PCA / L2-norm / red-blue
# tilt) always operate on physical coefficients, independently of Y_TRANSFORM.
_RAW_BY_SERIES = {
    'near': coef_near_all,
    'far': coef_far_all,
    'sci_true': coef_sci_all,
    'sci_pred_default': coef_pred_all,
}


def _series_full(series_name, coef_mat_use, kind):
    """Series value per row on the FULL dataset (all n_all rows)."""
    if kind.startswith('shape_moon_'):
        _raw = _RAW_BY_SERIES[series_name]
        _y = _moon_shape[kind[len('shape_moon_'):]](_raw)
        # NaN-mask rows where Moon_bs amplitude is negligible (moon-down noise).
        if _MOON_AMP_FLOOR is not None:
            _sum_raw = np.clip(_raw[:, _moon_idx].sum(axis=1), 0.0, None)
            _y = np.where(_sum_raw >= _MOON_AMP_FLOOR, _y, np.nan)
        return _y
    if kind.startswith('shape_zodi_'):
        return _zodi_shape[kind[len('shape_zodi_'):]](_RAW_BY_SERIES[series_name])
    if kind.startswith('shape_oh_'):
        return _oh_shape[kind[len('shape_oh_'):]](_RAW_BY_SERIES[series_name])
    return coef_mat_use[:, _individual_idx[kind]]


# --- (4) Nuisance regression ------------------------------------------
# 2026-08-24: moon-gated basis. Moon contribution physically vanishes for
# moon_alt <= 0, so sharing moon_sep / moon_phase / airmass_moon coefficients
# across up- AND down-rows produced a step-like residual at moon_alt=0.
# Fix: multiply every moon term by a smooth up-indicator (linear ramp
# 0..1 across [0, 5] deg). Airmass (target airmass) and sun_alt stay ungated.
# Basis columns:
#   up * ReLU(moon_alt)  amplitude ramp
#   up * cos(phase), up * sin(phase)  phase gated
#   up * moon_sep        separation gated
#   airmass, sun_alt     ungated (affect brightness regardless of moon)
_NUISANCE_MOON_UP_DEG = 5.0


def _nuisance_matrix(ctx_names, ctx_disp):
    if 'moon_alt' in ctx_names:
        _alt = np.asarray(ctx_disp[:, ctx_names.index('moon_alt')], dtype=np.float64)
        _up = np.clip(_alt / _NUISANCE_MOON_UP_DEG, 0.0, 1.0)
    else:
        _alt = None
        _up = np.ones(ctx_disp.shape[0], dtype=np.float64)
    _cols = []
    if _alt is not None:
        _cols.append(_up * np.maximum(_alt, 0.0))
    if 'moon_phase' in ctx_names:
        _phi = np.deg2rad(np.asarray(ctx_disp[:, ctx_names.index('moon_phase')],
                                     dtype=np.float64))
        _cols.append(_up * np.cos(_phi))
        _cols.append(_up * np.sin(_phi))
    if 'moon_sep' in ctx_names:
        _sep = np.asarray(ctx_disp[:, ctx_names.index('moon_sep')], dtype=np.float64)
        _cols.append(_up * _sep)
    if 'airmass' in ctx_names:
        _cols.append(np.asarray(ctx_disp[:, ctx_names.index('airmass')], dtype=np.float64))
    if 'sun_alt' in ctx_names:
        _cols.append(np.asarray(ctx_disp[:, ctx_names.index('sun_alt')], dtype=np.float64))
    if not _cols:
        return None
    return np.column_stack(_cols)


_N_near = _nuisance_matrix(_ctx_names_display, _ctx_near_disp)
_N_far = _nuisance_matrix(_ctx_names_display, _ctx_far_disp)
_N_sci = _nuisance_matrix(_ctx_names_display, _ctx_sci_disp)


def _residualise(y_all, N_all):
    """Fit y ~ 1 + N on train rows; return residuals on ALL rows."""
    if N_all is None:
        return y_all
    _tr = _train_mask & np.isfinite(y_all) & np.all(np.isfinite(N_all), axis=1)
    if _tr.sum() < 10:
        return y_all
    X_tr = np.column_stack([np.ones(_tr.sum()), N_all[_tr]])
    beta, *_ = np.linalg.lstsq(X_tr, y_all[_tr], rcond=None)
    X_all = np.column_stack([np.ones(n_all), N_all])
    return y_all - X_all @ beta

# --- Downsample rows for markers only (stats always use full data) -----
if n_all > _MAX_POINTS:
    _rng = np.random.default_rng(42)
    _row_sel = np.sort(_rng.choice(n_all, size=_MAX_POINTS, replace=False))
else:
    _row_sel = np.arange(n_all)

_SERIES = [
    ('near',             coef_near_use, _ctx_near_disp, _N_near, "#1f77b4"),
    ('far',              coef_far_use,  _ctx_far_disp,  _N_far,  "#9467bd"),
    ('sci_true',         coef_sci_use,  _ctx_sci_disp,  _N_sci,  "#2ca02c"),
    ('sci_pred_default', coef_pred_use, _ctx_sci_disp,  _N_sci,  "#d62728"),
]

# --- Trend line: 15-bin running median between p2..p98 of x -----------
def _running_median(x, y, n_bins=15):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.array([]), np.array([])
    x_ = x[m]; y_ = y[m]
    lo, hi = np.percentile(x_, [2, 98])
    if not (hi > lo):
        return np.array([]), np.array([])
    edges = np.linspace(lo, hi, n_bins + 1)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    y_med = np.full(n_bins, np.nan)
    for b in range(n_bins):
        _in = (x_ >= edges[b]) & (x_ <= edges[b + 1])
        if _in.sum() >= 10:
            y_med[b] = np.median(y_[_in])
    return x_mid, y_med


def _fisher_ci(r, n, z_=1.96):
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    _z = 0.5 * np.log((1 + r) / (1 - r))
    _se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(_z - z_ * _se)), float(np.tanh(_z + z_ * _se)))


def _stat_pair(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5:
        return dict(n=n, r=np.nan, r_lo=np.nan, r_hi=np.nan, rho=np.nan)
    r, _ = pearsonr(x[m], y[m])
    rho, _ = spearmanr(x[m], y[m])
    lo, hi = _fisher_ci(float(r), n)
    return dict(n=n, r=float(r), r_lo=lo, r_hi=hi, rho=float(rho))


# --- Build the figure (static matplotlib PNG output) -----------------
# Rasterized scatter keeps output size small and avoids the multi-million-marker
# plotly JSON payload. Loses interactivity; keeps the panel layout, stats, and
# running-median trend line.
n_rows = len(_ROW_DEFS)
n_cols = len(_ctx_names_display)

# Precompute y_full per (row_def, series) once so we can reuse for stats + trend.
_y_full_cache = {}
for row_label, kind in _ROW_DEFS:
    for series_name, coef_mat_use, _ctx, N_mat, _color in _SERIES:
        y_full = _series_full(series_name, coef_mat_use, kind)
        if Y_TRANSFORM == 'residual':
            y_full = _residualise(y_full, N_mat)
        _y_full_cache[(row_label, series_name)] = y_full

# STAT_SERIES ctx lookup for the per-panel annotation x-vectors.
_stat_ctx_disp = next(_ctx for _sn, _cm, _ctx, _N, _col in _SERIES
                       if _sn == STAT_SERIES)

_panel_w_in = 3.9  # inches per column (2.5x scaled 2026-08-24)
_panel_h_in = 3.25  # inches per row (2.5x scaled 2026-08-24)
_fig_w = _panel_w_in * n_cols + 2.5
_fig_h = _panel_h_in * n_rows + 3.0

fig, axes = plt.subplots(
    n_rows, n_cols,
    sharex='col', sharey='row',
    figsize=(_fig_w, _fig_h),
    squeeze=False,
)

for r_i, (row_label, kind) in enumerate(_ROW_DEFS):
    for c_i in range(n_cols):
        ax = axes[r_i, c_i]
        col_name = _ctx_names_display[c_i]
        # Scatter each series (rasterized so PNG stays lightweight).
        for series_name, _cm, ctx_disp, _N, color in _SERIES:
            if (col_name == 'sci_sep'
                    and series_name in ('sci_true', 'sci_pred_default')):
                continue
            y_full = _y_full_cache[(row_label, series_name)]
            ax.scatter(
                ctx_disp[_row_sel, c_i], y_full[_row_sel],
                s=10.0, c=color, alpha=0.22,
                linewidths=0, marker='.', rasterized=True,
            )
            if _TREND_ONLY_STAT and series_name != STAT_SERIES:
                continue
            _xm, _ym = _running_median(ctx_disp[:, c_i], y_full, n_bins=15)
            if _xm.size:
                ax.plot(_xm, _ym, color='#111111', linewidth=1.6,
                        rasterized=True)

        # Per-panel r / rho / CI annotation for STAT_SERIES.
        y_full_stat = _y_full_cache.get((row_label, STAT_SERIES))
        _skip_stat = (col_name == 'sci_sep'
                      and STAT_SERIES in ('sci_true', 'sci_pred_default'))
        if y_full_stat is not None and not _skip_stat:
            st = _stat_pair(_stat_ctx_disp[:, c_i], y_full_stat)
            if not np.isfinite(st['r']):
                _r_str = 'r=n/a'
                _err = 0.0
            else:
                _err = max(st['r_hi'] - st['r'], st['r'] - st['r_lo']) \
                    if np.isfinite(st['r_lo']) else 0.0
                _r_str = f"r={st['r']:+.2f}±{_err:.2f}" if _err > 0 else f"r={st['r']:+.2f}"
            _rho_str = (', ρ=n/a' if not np.isfinite(st['rho'])
                        else f", ρ={st['rho']:+.2f}")
            _nl = (''
                   if not (np.isfinite(st['rho']) and np.isfinite(st['r']))
                   else f", Δ={st['rho']**2 - st['r']**2:+.02f}")
            _n_str = (f" (n={st['n']//1000}k)" if st['n'] >= 1000
                      else f" (n={st['n']})")
            _txt = f"{_r_str}{_rho_str}{_nl}{_n_str}"
            _strong = ((np.isfinite(st['r']) and abs(st['r']) > 0.15)
                       or (np.isfinite(st['rho']) and abs(st['rho']) > 0.15))
            ax.text(
                0.02, 0.98, _txt,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11.0, color='#000000' if _strong else '#7a7a7a',
                weight='bold' if _strong else 'normal', zorder=5,
            )

        # Edge labels only.
        if r_i == 0:
            ax.set_title(col_name.replace('_', ' '), fontsize=13)
        if r_i == n_rows - 1:
            ax.set_xlabel(col_name.replace('_', ' '), fontsize=13)
            ax.tick_params(axis='x', labelrotation=30)
        if c_i == 0:
            ax.set_ylabel(row_label, fontsize=13)
        ax.tick_params(axis='both', labelsize=9, length=4, pad=2)
        for _sp in ax.spines.values():
            _sp.set_linewidth(0.5)

# Legend at top-left.
_handles = [Line2D([], [], marker='.', linestyle='None', color=color,
                    markersize=10, label=series_name)
            for series_name, _cm, _ctx, _N, color in _SERIES]
_handles.append(Line2D([], [], color='#111111', linewidth=1.6,
                        label=f'{STAT_SERIES} running median'))
fig.legend(handles=_handles, loc='upper left',
           ncol=len(_handles), bbox_to_anchor=(0.01, 0.995),
           fontsize=12, frameon=False)

fig.suptitle(
    (f"Coefficient vs context (n_rows={n_rows}, n_cols={n_cols}, "
     f"scatter n={_row_sel.size}/full n={n_all}). "
     f"y-transform: {Y_TRANSFORM} [{_Y_LABEL}]. "
     f"black line = 15-bin running median of {STAT_SERIES}. "
     f"panel annotation: Pearson r ±95% Fisher CI, Spearman ρ, "
     f"Δ=ρ²−r² (non-linearity), n (rows used for r, on full data)."),
    fontsize=14, y=0.995,
)

plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.05,
                    wspace=0.05, hspace=0.06)
# VS Code notebooks cap image display at the cell width regardless of
# IPython.display.Image(width=...). Workaround: emit HTML with an
# overflow-x: auto wrapper so the <img> gets its explicit pixel width
# and the user scrolls horizontally within the cell. Also save a full-
# resolution PNG next to the notebook for external viewing.
import base64
from pathlib import Path as _Path
from IPython.display import HTML

_buf = io.BytesIO()
fig.savefig(_buf, format='png', dpi=110, bbox_inches='tight')
plt.close(fig)
_buf.seek(0)
_png_bytes = _buf.getvalue()

_out_png = _Path.cwd() / 'coef_vs_context_scatter.png'
_out_png.write_bytes(_png_bytes)
print(f'  saved full-resolution PNG to {_out_png} '
      f'({len(_png_bytes)/1e6:.1f} MB; open externally for full detail).')

# ~65 px per figure-inch keeps panels legible while horizontal scroll stays
# manageable (~7000 px wide for the current column count).
_display_px = int(_fig_w * 65)
_b64 = base64.b64encode(_png_bytes).decode('ascii')
_html = (
    f'<div style="overflow-x: auto; overflow-y: hidden; width: 100%;">'
    f'<img src="data:image/png;base64,{_b64}" '
    f'style="width: {_display_px}px; max-width: none; height: auto; '
    f'display: block;" />'
    f'</div>'
)
display(HTML(_html))
