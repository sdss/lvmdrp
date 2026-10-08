# mean|residual| AND median|residual| vs COEF_ERR sigma per decile.
#
# The mean-vs-median comparison directly tests whether the σ over-estimation
# seen in cell 41 is driven by heavy-tailed outliers or by the bulk of the
# distribution. For a well-calibrated Gaussian σ:
#     E[|r|] / σ = √(2/π)  ≈ 0.7979
#     median|r| / σ        = 0.6745
#     E[|r|] / median|r|   = 0.7979 / 0.6745 ≈ 1.183
# A per-decile mean/median ratio well above 1.183 means outliers inflate the
# mean; if both mean and median lines sit far below the Gaussian references,
# even the bulk is calibrated wrong (σ is genuinely too large).
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

required = ['coef_pred_det', 'coef_sci_all', 'test_idx',
            '_group_indices_compress', 'coef_err_sci_all',
            'coef_near_all', 'coef_far_all', 'ctx_sci_all',
            'ctx_near_all', 'ctx_far_all',
            'predict_sci_coefficients_default', 'mlp_artifacts']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run trainer + coef_err loading first. Missing: '
                       + ', '.join(_missing))

_variants = {}
for _name in ('mlp_artifacts', 'art_pergroup', 'art_global'):
    _obj = globals().get(_name)
    if isinstance(_obj, dict) and _obj.get('is_ensemble', False):
        _variants[_name] = _obj

_te = np.asarray(test_idx, dtype=int)
_y_true = np.asarray(coef_sci_all[_te], dtype=np.float64)
_sig = np.asarray(coef_err_sci_all[_te], dtype=np.float64)

_pred_by_variant = {}
for _name, _arts in _variants.items():
    if (_name == 'mlp_artifacts' and 'coef_pred_det' in globals()
            and np.asarray(coef_pred_det).shape == _y_true.shape):
        _pred_by_variant[_name] = np.asarray(coef_pred_det, dtype=np.float64)
    else:
        _pred_by_variant[_name] = predict_sci_coefficients_default(
            _arts,
            coef_near_phys=coef_near_all[_te], coef_far_phys=coef_far_all[_te],
            ctx_near_phys=ctx_near_all[_te], ctx_far_phys=ctx_far_all[_te],
            ctx_sci_phys=ctx_sci_all[_te],
        ).astype(np.float64)

_variant_colors = {
    'mlp_artifacts': '#2ca02c',
    'art_pergroup':  '#1f77b4',
    'art_global':    '#ff7f0e',
}

_groups = list(_group_indices_compress.keys())
_n_groups = len(_groups)
fig = make_subplots(
    rows=1, cols=_n_groups,
    subplot_titles=[f'{g}  (n_coef={len(_group_indices_compress[g])})'
                    for g in _groups],
    horizontal_spacing=0.05,
)

_n_bins = 10
_summary_rows = []
_GAUSS_MEAN = float(np.sqrt(2.0 / np.pi))   # 0.7979
_GAUSS_MEDIAN = 0.6744897501960817          # scipy.stats.halfnorm.median()

for panel_i, gname in enumerate(_groups):
    col = panel_i + 1
    gidx = np.asarray(_group_indices_compress[gname], dtype=int)

    _sig_g_full = _sig[:, gidx].ravel()
    _ok = np.isfinite(_sig_g_full) & (_sig_g_full > 0.0)
    _sig_g = _sig_g_full[_ok]
    if _sig_g.size < _n_bins * 2:
        continue

    _edges = np.quantile(_sig_g, np.linspace(0.0, 1.0, _n_bins + 1))
    _bin_center = np.sqrt(_edges[:-1] * _edges[1:])
    _bin_id = np.digitize(_sig_g, _edges[1:-1], right=False)

    _sig_x = np.geomspace(max(_sig_g.min(), 1e-30), _sig_g.max(), 200)
    fig.add_trace(go.Scatter(
        x=_sig_x, y=_sig_x, mode='lines',
        line=dict(color='black', width=1.4, dash='dash'),
        name='y = σ (slope 1)', showlegend=(panel_i == 0),
    ), row=1, col=col)
    fig.add_trace(go.Scatter(
        x=_sig_x, y=_GAUSS_MEAN * _sig_x, mode='lines',
        line=dict(color='#d62728', width=1.4, dash='dot'),
        name=f'mean target: √(2/π) σ = {_GAUSS_MEAN:.4f} σ',
        showlegend=(panel_i == 0),
    ), row=1, col=col)
    fig.add_trace(go.Scatter(
        x=_sig_x, y=_GAUSS_MEDIAN * _sig_x, mode='lines',
        line=dict(color='#9467bd', width=1.4, dash='dot'),
        name=f'median target: 0.6745 σ',
        showlegend=(panel_i == 0),
    ), row=1, col=col)

    for _var_name, _pred in _pred_by_variant.items():
        _abs_resid = np.abs((_pred - _y_true)[:, gidx].ravel()[_ok])
        _mean_r = np.array([
            float(_abs_resid[_bin_id == b].mean())
            if np.any(_bin_id == b) else np.nan
            for b in range(_n_bins)
        ], dtype=np.float64)
        _median_r = np.array([
            float(np.median(_abs_resid[_bin_id == b]))
            if np.any(_bin_id == b) else np.nan
            for b in range(_n_bins)
        ], dtype=np.float64)

        _fin_mean = np.isfinite(_mean_r) & (_mean_r > 0.0) & (_bin_center > 0.0)
        _slope_mean = (np.polyfit(np.log10(_bin_center[_fin_mean]),
                                  np.log10(_mean_r[_fin_mean]), 1)[0]
                       if _fin_mean.sum() >= 3 else np.nan)
        _fin_med = np.isfinite(_median_r) & (_median_r > 0.0) & (_bin_center > 0.0)
        _slope_median = (np.polyfit(np.log10(_bin_center[_fin_med]),
                                    np.log10(_median_r[_fin_med]), 1)[0]
                         if _fin_med.sum() >= 3 else np.nan)

        _color = _variant_colors.get(_var_name, '#7f7f7f')
        fig.add_trace(go.Scatter(
            x=_bin_center, y=_mean_r, mode='lines+markers',
            marker=dict(size=6, symbol='circle'),
            line=dict(color=_color, width=2),
            name=f'{_var_name}  mean  slope={_slope_mean:.2f}',
            showlegend=True,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=_bin_center, y=_median_r, mode='lines+markers',
            marker=dict(size=6, symbol='square'),
            line=dict(color=_color, width=2, dash='dash'),
            name=f'{_var_name}  median  slope={_slope_median:.2f}',
            showlegend=True,
        ), row=1, col=col)

        _mid = len(_mean_r) // 2
        _ratio_mean_mid = (float(_mean_r[_mid] / max(_bin_center[_mid], 1e-30))
                           if np.isfinite(_mean_r[_mid]) else np.nan)
        _ratio_median_mid = (float(_median_r[_mid] / max(_bin_center[_mid], 1e-30))
                             if np.isfinite(_median_r[_mid]) else np.nan)
        # Outlier proxy: per-bin mean/median, then median across bins so we
        # aren't fooled by one anomalous decile.
        _both = _fin_mean & _fin_med
        _outlier_ratio = (float(np.median(_mean_r[_both] / _median_r[_both]))
                          if _both.any() else np.nan)
        _summary_rows.append({
            'group': gname,
            'variant': _var_name,
            'slope_mean': _slope_mean,
            'slope_median': _slope_median,
            'mean/σ@med_bin': _ratio_mean_mid,
            'median/σ@med_bin': _ratio_median_mid,
            'mean/median ratio': _outlier_ratio,
            'n_pts': int(_sig_g.size),
        })

    fig.update_xaxes(type='log', title_text='σ (COEF_ERR, native units)',
                     row=1, col=col)
    fig.update_yaxes(type='log',
                     title_text=('|residual|' if panel_i == 0 else ''),
                     row=1, col=col)

fig.update_layout(
    template='plotly_white',
    height=520,
    width=280 * _n_groups + 100,
    title=(f'COEF_ERR σ-calibration: |residual| vs σ per decile '
           f'(mean = circles, median = squares; test split, n_row={_te.size})'),
    margin=dict(l=60, r=30, t=90, b=100),
    legend=dict(orientation='h', y=-0.22, x=0.0),
)
fig.show()

print()
print(f'Gaussian targets: mean/σ = √(2/π) = {_GAUSS_MEAN:.4f},  '
      f'median/σ = {_GAUSS_MEDIAN:.4f},  '
      f'expected mean/median = {_GAUSS_MEAN/_GAUSS_MEDIAN:.3f}')
print()
_df = pd.DataFrame(_summary_rows)
print(_df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print()
print('Interpretation:')
print('  slope=1 for BOTH mean and median  -> σ scales correctly across deciles')
print('  slopes differ  -> outliers shift the mean trace relative to the median trace')
print('  mean/median ratio  ≈ 1.18  -> Gaussian bulk (mean gap explained by tail);')
print('                     >> 1.18 -> heavy outliers inflate mean|resid|')
print('  If median|residual| is still ≪ 0.6745·σ, σ over-estimation is real,')
print('    not just an outlier artefact of the mean statistic.')
