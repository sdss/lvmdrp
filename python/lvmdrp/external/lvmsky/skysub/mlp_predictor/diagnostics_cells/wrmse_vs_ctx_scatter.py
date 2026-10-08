# Scatter of WRMSE vs each sci-pointing context column, colored by subset.
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

required = ['ctx_sci_full', 'ctx_names_full', 'wrmse_arr',
            'train_mask_t', 'valtest_mask_t', 'lmc_mask_t']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError(
        'Missing kernel state: ' + ', '.join(_missing)
        + '. Run the sWRMSE_coef correlation cell above first.'
    )

subsets = [
    ('Train', train_mask_t, 'rgba(31,119,180,0.30)'),
    ('Val/Test', valtest_mask_t, 'rgba(44,160,44,0.55)'),
    ('LMC/SMC', lmc_mask_t, 'rgba(214,39,40,0.50)'),
]

useful = [(j, cname) for j, cname in enumerate(ctx_names_full)
          if np.any(np.isfinite(ctx_sci_full[:, j])) and np.nanstd(ctx_sci_full[:, j]) > 0]

n_cols_grid = 4
n_rows_grid = math.ceil(len(useful) / n_cols_grid)

fig = make_subplots(rows=n_rows_grid, cols=n_cols_grid,
                    subplot_titles=[c for _, c in useful],
                    horizontal_spacing=0.04, vertical_spacing=0.06)

for panel_i, (j, cname) in enumerate(useful):
    r = panel_i // n_cols_grid + 1
    c = panel_i % n_cols_grid + 1
    x_full = ctx_sci_full[:, j]
    for label, mask, color in subsets:
        m = mask & np.isfinite(x_full) & np.isfinite(wrmse_arr)
        if not np.any(m):
            continue
        fig.add_trace(go.Scattergl(
            x=x_full[m], y=wrmse_arr[m], mode='markers',
            marker=dict(color=color, size=3),
            name=label, legendgroup=label,
            showlegend=(panel_i == 0),
            hoverinfo='skip',
        ), row=r, col=c)

# Y-range from in-domain 0.5-99.5 percentile so extreme tail WRMSE does not
# compress the visible range.  WRMSE is unbounded above, so no ceiling clamp.
in_domain = wrmse_arr[(train_mask_t | valtest_mask_t) & np.isfinite(wrmse_arr)]
y_lo = float(np.percentile(in_domain, 0.5))
y_hi = float(np.percentile(in_domain, 99.5))
y_pad = 0.05 * max(y_hi - y_lo, 1e-6)
y_range = [max(0.0, y_lo - y_pad), y_hi + y_pad]
for panel_i in range(len(useful)):
    fig.update_yaxes(range=y_range,
                     row=panel_i // n_cols_grid + 1,
                     col=panel_i % n_cols_grid + 1)

fig.update_layout(
    title='sWRMSE_coef vs sci-pointing context columns (colored by subset)',
    width=1500, height=260 * n_rows_grid + 100,
    template='plotly_white',
    legend=dict(orientation='h', y=-0.02),
)
fig.show()
