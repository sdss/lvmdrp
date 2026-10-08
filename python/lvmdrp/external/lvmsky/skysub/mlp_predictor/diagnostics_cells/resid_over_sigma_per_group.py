# |residual| / sigma per coefficient group -- calibration test for the
# LSF-aware decomposition sigma (COEF_ERR HDU, aleatoric branch).  For each
# coefficient j: z_j = |coef_pred - coef_true| / sigma_j, aggregated per group.
# Under a well-calibrated Gaussian sigma the |z| distribution is half-N(1)
# with median 0.6745, mean 0.7979, p95 = 1.96.  Median|z| systematically above
# 1 means sigma is under-estimated; below 0.5 means over-estimated.
#
# Compares against the item-6 ensemble-spread sigma (epistemic branch) when
# available and reports the combined predictive sigma
# sigma_pred = sqrt(sigma_aleatoric^2 + sigma_epistemic^2) as the third source.
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from scipy.stats import halfnorm

required = ['coef_pred_det', 'coef_sci_all', 'test_idx',
            '_group_indices_compress', 'coef_err_sci_all']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run trainer + coef_err loading first. Missing: '
                       + ', '.join(_missing))

_te = np.asarray(test_idx, dtype=int)
_y_true = np.asarray(coef_sci_all[_te], dtype=np.float64)
_y_pred = np.asarray(coef_pred_det, dtype=np.float64)
_resid = _y_pred - _y_true

# Aleatoric sigma from the decomposition COEF_ERR HDU.
_sigma_alea = np.asarray(coef_err_sci_all[_te], dtype=np.float64)

# Epistemic sigma from item-6 (ensemble spread across seeds), if the cell ran.
_sigma_epi = None
if ('mlp_artifacts' in globals()
        and mlp_artifacts.get('is_ensemble', False)
        and 'predict_sci_coefficients_default' in globals()):
    _members = mlp_artifacts['members']
    _stack = np.stack([
        predict_sci_coefficients_default(
            _m,
            coef_near_phys=coef_near_all[_te], coef_far_phys=coef_far_all[_te],
            ctx_near_phys=ctx_near_all[_te], ctx_far_phys=ctx_far_all[_te],
            ctx_sci_phys=ctx_sci_all[_te],
        ).astype(np.float32) for _m in _members], axis=0)
    _sigma_epi = _stack.std(axis=0, ddof=1).astype(np.float64)

# Combined predictive sigma = sqrt(aleatoric^2 + epistemic^2)
_sigma_combined = None
if _sigma_epi is not None:
    _valid_mask = np.isfinite(_sigma_alea) & np.isfinite(_sigma_epi)
    _sigma_combined = np.sqrt(
        np.where(_valid_mask, _sigma_alea, 0.0) ** 2
        + np.where(_valid_mask, _sigma_epi, 0.0) ** 2
    )
    _sigma_combined[~_valid_mask] = np.nan

_sources = [('COEF_ERR (aleatoric)', _sigma_alea, '#1f77b4')]
if _sigma_epi is not None:
    _sources.append(('ensemble spread (epistemic)', _sigma_epi, '#ff7f0e'))
    _sources.append(('combined (quadrature)', _sigma_combined, '#2ca02c'))
print(f'sigma sources exercised: {[s[0] for s in _sources]}')


def _z_per_group(sigma_arr, gidx):
    r = _resid[:, gidx]
    s = sigma_arr[:, gidx]
    ok = np.isfinite(r) & np.isfinite(s) & (s > 0.0)
    return np.abs(r[ok]) / s[ok] if ok.any() else np.array([], dtype=np.float64)


_groups = list(_group_indices_compress.keys())
_n_groups = len(_groups)
_n_cols = min(3, _n_groups)
_n_rows = (_n_groups + _n_cols - 1) // _n_cols

fig = make_subplots(
    rows=_n_rows, cols=_n_cols,
    subplot_titles=[f'{g} (n_coef={len(_group_indices_compress[g])})'
                    for g in _groups],
    vertical_spacing=0.10, horizontal_spacing=0.08,
)

_z_by_source = {label: {} for label, _s, _c in _sources}
for panel_i, gname in enumerate(_groups):
    r = panel_i // _n_cols + 1
    c = panel_i % _n_cols + 1
    gidx = np.asarray(_group_indices_compress[gname], dtype=int)

    _panel_max = 3.0
    for label, sigma_arr, color in _sources:
        zvals = _z_per_group(sigma_arr, gidx)
        _z_by_source[label][gname] = zvals
        if zvals.size == 0:
            continue
        _panel_max = max(_panel_max, float(np.percentile(zvals, 95)) * 1.5)
        fig.add_trace(go.Histogram(
            x=zvals, nbinsx=60, name=label,
            marker_color=color, opacity=0.55,
            histnorm='probability density',
            showlegend=(panel_i == 0),
        ), row=r, col=c)

    # Half-N(1) target overlay.
    _x = np.linspace(0.0, _panel_max, 200)
    fig.add_trace(go.Scatter(
        x=_x, y=halfnorm.pdf(_x), mode='lines',
        line=dict(color='#d62728', width=2.2, dash='dash'),
        name='half-N(1) target', showlegend=(panel_i == 0),
    ), row=r, col=c)
    # Reference vertical: 0.6745 = target median.
    fig.add_vline(x=0.6745, line=dict(color='#d62728', dash='dot', width=1),
                  row=r, col=c)
    fig.update_xaxes(range=[0.0, min(6.0, _panel_max)], row=r, col=c,
                     title_text='|residual|/sigma' if r == _n_rows else '')
    fig.update_yaxes(title_text='density' if c == 1 else '', row=r, col=c)

fig.update_layout(
    template='plotly_white',
    barmode='overlay',
    height=280 * _n_rows + 100,
    title=('|residual| / sigma per coefficient group. '
           'Well-calibrated sigma -> histogram follows half-N(1) (red dashed); '
           'red dotted line at 0.6745 is the target median.'),
    margin=dict(l=60, r=30, t=90, b=60),
)
fig.show()

# Tabular summary per (source, group).
_summary_rows = []
for label, zdict in _z_by_source.items():
    for gname, zvals in zdict.items():
        if zvals.size == 0:
            _summary_rows.append({
                'sigma_source': label, 'group': gname, 'n_valid': 0,
                'median|z|': np.nan, 'mean|z|': np.nan, 'p95|z|': np.nan,
                'sigma_scale_hint': np.nan,
            })
            continue
        _p50 = float(np.median(zvals))
        _p95 = float(np.percentile(zvals, 95))
        _summary_rows.append({
            'sigma_source': label,
            'group': gname,
            'n_valid': int(zvals.size),
            'median|z|': _p50,
            'mean|z|': float(np.mean(zvals)),
            'p95|z|': _p95,
            # If sigma is off by scalar factor f, median|z| = 0.6745 * f, so
            # f = median|z|/0.6745 gives the per-group sigma-scale correction.
            'sigma_scale_hint': _p50 / 0.6745,
        })

_pd_df = pd.DataFrame(_summary_rows)
print()
print('Per-group |z| calibration diagnostic '
      '(half-N target: median|z|=0.6745, mean|z|=0.7979, p95|z|=1.96):')
print(_pd_df.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

# Verdict per source.
print()
for label in [s[0] for s in _sources]:
    _rows = [r for r in _summary_rows if r['sigma_source'] == label and r['n_valid'] > 0]
    if not _rows:
        continue
    _hints = np.array([r['sigma_scale_hint'] for r in _rows], dtype=np.float64)
    _mean_hint = float(np.nanmean(_hints))
    _max_dev = float(np.nanmax(np.abs(_hints - 1.0)))
    _verdict = ('CALIBRATED' if _max_dev < 0.25
                else 'DRIFT' if _max_dev < 0.5
                else 'UNCALIBRATED')
    print(f'  {label}: sigma_scale hint {_mean_hint:.2f}x on average, '
          f'max per-group drift {_max_dev * 100:.1f}%  -> {_verdict}')
