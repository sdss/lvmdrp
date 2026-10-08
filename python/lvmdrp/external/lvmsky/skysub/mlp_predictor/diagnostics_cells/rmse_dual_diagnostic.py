# Coefficient-vs-spectrum RMSE diagnostics: reconcile mean/median behavior.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

required = ['coef_sci_all', 'coef_pred_det', 'test_idx']
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError('Run the training + prediction cells first. Missing: ' + ', '.join(missing))

# 1) Coefficient-space RMSE distribution on the night-held-out test split
_te = np.asarray(test_idx, dtype=int)
_y_true = np.asarray(coef_sci_all[_te], dtype=np.float64)
_y_pred = np.asarray(coef_pred_det, dtype=np.float64)
if _y_pred.shape != _y_true.shape:
    raise RuntimeError(f'coef_pred_det shape {_y_pred.shape} does not match y_true {_y_true.shape}')

rmse_coef = np.sqrt(np.mean((_y_pred - _y_true) ** 2, axis=0))
rmse_coef = rmse_coef[np.isfinite(rmse_coef)]
if rmse_coef.size == 0:
    raise RuntimeError('No finite eRMSE values')

# 2) Row-wise spectral RMSE + pixel-space WRMSE from the reconstruction subset cell
rmse_row_spec = None
wrmse_row_pix = None
if 'rmse_subset_results' in globals() and isinstance(rmse_subset_results, dict):
    _arr = np.asarray(rmse_subset_results.get('sci_rmse', []), dtype=np.float64)
    _arr = _arr[np.isfinite(_arr)]
    if _arr.size:
        rmse_row_spec = _arr
    _warr = np.asarray(rmse_subset_results.get('sci_wrmse', []), dtype=np.float64)
    _warr = _warr[np.isfinite(_warr)]
    if _warr.size:
        wrmse_row_pix = _warr

def _pct_table(arr, label):
    q = [0, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(arr, q)
    return pd.DataFrame({
        'metric': [label] * len(q),
        'percentile': q,
        'value': vals,
    })

tbl = [_pct_table(rmse_coef, 'eRMSE')]
if rmse_row_spec is not None:
    tbl.append(_pct_table(rmse_row_spec, 'sci_pRMSE'))
if wrmse_row_pix is not None:
    tbl.append(_pct_table(wrmse_row_pix, 'sci_pWRMSE'))
pct_df = pd.concat(tbl, ignore_index=True)

print('sRMSE / eRMSE / sWRMSE distribution summary:')
print(pct_df.to_string(index=False, float_format=lambda v: f'{v:.6g}'))

print('')
print('Per-coefficient (eRMSE) aggregate stats (Cell 33 now uses median_eRMSE as primary):')
print(f'  mean   = {float(np.mean(rmse_coef)):.6g}')
print(f'  median = {float(np.median(rmse_coef)):.6g}')
print(f'  p95    = {float(np.percentile(rmse_coef, 95)):.6g}')
print(f'  p99    = {float(np.percentile(rmse_coef, 99)):.6g}')
_tail_ratio = float(np.percentile(rmse_coef, 99) / max(np.median(rmse_coef), 1e-30))
print(f'  p99/median = {_tail_ratio:.3g}')
if _tail_ratio > 100.0:
    print('Loss note: extremely heavy-tailed coefficient errors. A pure MSE-style objective '
          'is likely over-weighting a small outlier set in model selection.')
elif _tail_ratio > 20.0:
    print('Loss note: heavy-tailed coefficient errors. MSE can still over-emphasize the tail; '
          'consider robust alternatives for selection or training.')
else:
    print('Loss note: tail is moderate; current loss likely not severely tail-dominated.')

if rmse_row_spec is not None:
    print('')
    print('Per-row pixel-space pRMSE stats (from rmse_subset_results["sci_rmse"]):')
    print(f'  mean   = {float(np.mean(rmse_row_spec)):.6g}')
    print(f'  median = {float(np.median(rmse_row_spec)):.6g}')
    print(f'  p95    = {float(np.percentile(rmse_row_spec, 95)):.6g}')
    print(f'  p99    = {float(np.percentile(rmse_row_spec, 99)):.6g}')
else:
    print('')
    print('Per-row sci_pRMSE not available: run the batch RMSE subset evaluation cell first.')

if wrmse_row_pix is not None:
    print('')
    print('Per-row pixel-space pWRMSE stats (from rmse_subset_results["sci_wrmse"]):')
    print(f'  mean   = {float(np.mean(wrmse_row_pix)):.6g}')
    print(f'  median = {float(np.median(wrmse_row_pix)):.6g}')
    print(f'  p95    = {float(np.percentile(wrmse_row_pix, 95)):.6g}')
    print(f'  p99    = {float(np.percentile(wrmse_row_pix, 99)):.6g}')
    _pix_src = (rmse_subset_results.get('pix_sigma_source', {})
                if isinstance(rmse_subset_results, dict) else {})
    if _pix_src:
        print(f'  sigma sources: {_pix_src}')
else:
    print('')
    print('Per-row sci_pWRMSE not available: batch RMSE subset cell has not populated sci_wrmse.')

# 2b) Worst coefficients by RMSE (largest errors)
n_worst_coef = 15
rmse_coef_full = np.sqrt(np.mean((_y_pred - _y_true) ** 2, axis=0))
coef_idx = np.arange(rmse_coef_full.size, dtype=int)

if 'coef_names_all' in globals() and len(coef_names_all) == rmse_coef_full.size:
    coef_names = [str(x) for x in coef_names_all]
elif 'filtered_triplet' in globals() and 'coef_names' in filtered_triplet and len(filtered_triplet['coef_names']) == rmse_coef_full.size:
    coef_names = [str(x) for x in filtered_triplet['coef_names']]
else:
    coef_names = [f'coef_{j}' for j in coef_idx]

worst_idx = np.argsort(rmse_coef_full)[::-1][:min(n_worst_coef, rmse_coef_full.size)]
worst_df = pd.DataFrame({
    'rank': np.arange(1, worst_idx.size + 1, dtype=int),
    'coef_index': coef_idx[worst_idx],
    'coef_name': [coef_names[j] for j in worst_idx],
    'rmse': rmse_coef_full[worst_idx],
    'true_mean': np.mean(_y_true[:, worst_idx], axis=0),
    'pred_mean': np.mean(_y_pred[:, worst_idx], axis=0),
    'mean_bias': np.mean(_y_pred[:, worst_idx] - _y_true[:, worst_idx], axis=0),
})
print('')
print(f'Top {worst_idx.size} worst coefficients by eRMSE (test split):')
print(worst_df.to_string(index=False, float_format=lambda v: f'{v:.6g}'))

# 3) Side-by-side histograms: coefficient RMSE, spectral RMSE, pixel WRMSE.
fig = make_subplots(rows=1, cols=3, subplot_titles=(
    'Per-coefficient eRMSE (test split)',
    'Per-row pixel pRMSE (subset)',
    'Per-row pixel pWRMSE (subset)',
))

fig.add_trace(
    go.Histogram(x=rmse_coef, nbinsx=80, name='eRMSE', marker_color='#1f77b4', opacity=0.85),
    row=1, col=1,
)
fig.add_vline(x=float(np.median(rmse_coef)), line=dict(color='#1f77b4', dash='dash'), row=1, col=1)
fig.add_vline(x=float(np.mean(rmse_coef)), line=dict(color='#d62728', dash='dot'), row=1, col=1)

if rmse_row_spec is not None:
    fig.add_trace(
        go.Histogram(x=rmse_row_spec, nbinsx=50, name='sci_pRMSE', marker_color='#2ca02c', opacity=0.85),
        row=1, col=2,
    )
    fig.add_vline(x=float(np.median(rmse_row_spec)), line=dict(color='#2ca02c', dash='dash'), row=1, col=2)
    fig.add_vline(x=float(np.mean(rmse_row_spec)), line=dict(color='#d62728', dash='dot'), row=1, col=2)
else:
    fig.add_annotation(x=0.5, y=0.5, xref='x2 domain', yref='y2 domain',
                       text='Run RMSE subset cell to populate this panel', showarrow=False)

if wrmse_row_pix is not None:
    fig.add_trace(
        go.Histogram(x=wrmse_row_pix, nbinsx=50, name='sci_pWRMSE',
                     marker_color='#ff7f0e', opacity=0.85),
        row=1, col=3,
    )
    fig.add_vline(x=float(np.median(wrmse_row_pix)), line=dict(color='#ff7f0e', dash='dash'), row=1, col=3)
    fig.add_vline(x=float(np.mean(wrmse_row_pix)), line=dict(color='#d62728', dash='dot'), row=1, col=3)
else:
    fig.add_annotation(x=0.5, y=0.5, xref='x3 domain', yref='y3 domain',
                       text='Run RMSE subset cell to populate this panel',
                       showarrow=False)

fig.update_layout(
    template='plotly_white',
    barmode='overlay',
    height=460,
    title='Error diagnostics: coefficient-space eRMSE, pixel-space pRMSE, pixel-space pWRMSE',
    margin=dict(l=60, r=20, t=70, b=55),
)
fig.update_xaxes(title_text='eRMSE value', row=1, col=1)
fig.update_xaxes(title_text='sRMSE value', row=1, col=2)
fig.update_xaxes(title_text='sWRMSE value', row=1, col=3)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=1, col=2)
fig.update_yaxes(title_text='Count', row=1, col=3)
fig.show()
