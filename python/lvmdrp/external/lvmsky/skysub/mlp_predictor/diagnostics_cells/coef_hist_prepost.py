# Coefficient distributions by field, pre- vs post-filtering.
# Row 1: raw triplet (all rows returned by the loader).
# Row 2: filtered_triplet (rows surviving field / chi2 / hard-clip / OH-MAD /
# kappa-sigma). Companion to the context histogram above; three step-line
# traces per panel (sky_near / sky_far / science) so per-arm amplitude
# shifts stay visible in the same physical units the decomposition writes
# to disk. x-range is per-row (pre-filter shows the outlier tail; post-filter
# shows only rows that enter training).
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_coef_names_hist = [str(n) for n in filtered_triplet['coef_names']]
_names_l_hist = [n.lower() for n in _coef_names_hist]
_moon_hist_idx = np.array(
    [i for i, n in enumerate(_names_l_hist)
     if n.startswith('moon_bs')], dtype=int)
_zodi_hist_idx = np.array(
    [i for i, n in enumerate(_names_l_hist)
     if n.startswith('zodi_bs')], dtype=int)
_oh_hist_idx = np.array(
    [i for i, n in enumerate(_names_l_hist) if n.startswith('oh_')], dtype=int)


def _find_coef_col_hist(name):
    lname = name.lower()
    hits = [i for i, n in enumerate(_names_l_hist) if n == lname]
    return int(hits[0]) if hits else None


_individual_coef_names_hist = ['HO2', 'FeO', 'O2Ac']
_individual_coef_idx_hist = {n: _find_coef_col_hist(n)
                             for n in _individual_coef_names_hist}


def _per_row_scalar_hist(coef_mat, kind):
    _mat = np.asarray(coef_mat, dtype=np.float64)
    if kind == 'mean_moon':
        if _moon_hist_idx.size == 0:
            return None
        return np.nanmean(_mat[:, _moon_hist_idx], axis=1)
    if kind == 'mean_zodi':
        if _zodi_hist_idx.size == 0:
            return None
        return np.nanmean(_mat[:, _zodi_hist_idx], axis=1)
    if kind == 'mean_oh':
        if _oh_hist_idx.size == 0:
            return None
        return np.nanmean(_mat[:, _oh_hist_idx], axis=1)
    j = _individual_coef_idx_hist.get(kind)
    return _mat[:, j] if j is not None else None


_coef_panels_hist = [
    (f"mean moon (n={_moon_hist_idx.size})", 'mean_moon'),
    (f"mean zodi (n={_zodi_hist_idx.size})", 'mean_zodi'),
    (f"mean OH (n={_oh_hist_idx.size})", 'mean_oh'),
]
for _n in _individual_coef_names_hist:
    if _individual_coef_idx_hist.get(_n) is not None:
        _coef_panels_hist.append((_n, _n))

_hist_field_colors_coef = {
    'sky_near': '#1f77b4',
    'sky_far': '#9467bd',
    'science': '#2ca02c',
}
_hist_field_order_coef = ['sky_near', 'sky_far', 'science']
_hist_n_bins_coef = 60

_hist_source_arrays = {
    'pre-filter': {
        'sky_near': triplet['coef_near'],
        'sky_far': triplet['coef_far'],
        'science': triplet['coef_sci'],
        'n_rows': int(np.asarray(triplet['coef_near']).shape[0]),
    },
    'post-filter': {
        'sky_near': filtered_triplet['coef_near'],
        'sky_far': filtered_triplet['coef_far'],
        'science': filtered_triplet['coef_sci'],
        'n_rows': int(np.asarray(filtered_triplet['coef_near']).shape[0]),
    },
}
_hist_source_order = ['pre-filter', 'post-filter']

_n_panels_coef = len(_coef_panels_hist)
_n_rows_coef = len(_hist_source_order)
_n_cols_coef = _n_panels_coef

_subplot_titles_coef = []
for _r in range(_n_rows_coef):
    for _label, _ in _coef_panels_hist:
        _subplot_titles_coef.append(_label if _r == 0 else '')

fig_coef_hist = make_subplots(
    rows=_n_rows_coef,
    cols=_n_cols_coef,
    subplot_titles=_subplot_titles_coef,
    horizontal_spacing=0.06,
    vertical_spacing=0.14,
)

for _r_i, _src in enumerate(_hist_source_order, start=1):
    _bundle = _hist_source_arrays[_src]
    _row_label = f"{_src} (n={_bundle['n_rows']})"
    for _c_i, (_label, _kind) in enumerate(_coef_panels_hist, start=1):
        _per_field_vals = {}
        for _f in _hist_field_order_coef:
            _v = _per_row_scalar_hist(_bundle[_f], _kind)
            if _v is None:
                continue
            _v = np.asarray(_v, dtype=np.float64)
            _per_field_vals[_f] = _v[np.isfinite(_v)]
        if not _per_field_vals:
            continue
        _combined = np.concatenate(list(_per_field_vals.values()))
        if _combined.size < 2:
            continue
        _lo = float(np.min(_combined))
        _hi = float(np.max(_combined))
        if not np.isfinite(_lo) or not np.isfinite(_hi) or _hi <= _lo:
            continue
        _edges = np.linspace(_lo, _hi, _hist_n_bins_coef + 1)
        _step_x = np.repeat(_edges, 2)[1:-1]
        _first_legend_slot = (_r_i == 1 and _c_i == 1)
        for _f in _hist_field_order_coef:
            _v = _per_field_vals.get(_f)
            if _v is None or _v.size == 0:
                continue
            _counts, _ = np.histogram(_v, bins=_edges)
            _step_y = np.repeat(_counts, 2)
            fig_coef_hist.add_trace(
                go.Scattergl(
                    x=_step_x,
                    y=_step_y,
                    mode='lines',
                    line=dict(color=_hist_field_colors_coef[_f], width=1.4),
                    name=_f,
                    legendgroup=_f,
                    showlegend=_first_legend_slot,
                    hovertemplate=_f + ': %{y}<extra></extra>',
                ),
                row=_r_i,
                col=_c_i,
            )
    fig_coef_hist.update_yaxes(
        title_text=_row_label, row=_r_i, col=1,
        title_font=dict(size=10))

fig_coef_hist.for_each_annotation(lambda a: a.update(font=dict(size=11)))
fig_coef_hist.update_xaxes(showline=True, mirror=True, ticks='outside',
                            ticklen=4, showticklabels=True,
                            exponentformat='e', separatethousands=False)
fig_coef_hist.update_yaxes(showline=True, mirror=True, ticks='outside',
                            ticklen=4, showticklabels=True,
                            exponentformat='e', separatethousands=False)
fig_coef_hist.update_layout(
    template='plotly_white',
    title=('Sky-decomposition coefficient distributions by field, '
           'pre- (top) vs post- (bottom) filtering. '
           "x-range per row: full min-max of each row's finite values "
           '(pre-filter shows the outlier tail; post-filter shows only rows '
           'that enter training).'),
    height=_n_rows_coef * 260 + 200,
    width=min(2000, _n_cols_coef * 380 + 140),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
    margin=dict(l=90, r=20, t=110, b=50),
)
fig_coef_hist.show()
