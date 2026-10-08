# Stability check: are the current top-15 worst coefficients always bad?
import numpy as np
import pandas as pd
import plotly.graph_objects as go

required = ['coef_sci_all', 'test_idx']
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError('Run the training + prediction cells first. Missing: ' + ', '.join(missing))

N_EXPERIMENTS = 10
N_WORST = 15
RNG_SEED = 12345

y_true_all = np.asarray(coef_sci_all, dtype=np.float64)
n_rows, n_coef = y_true_all.shape

# Prefer full-dataset predictions if available so random splits are cheap.
if 'coef_pred_all' in globals() and np.asarray(coef_pred_all).shape == y_true_all.shape:
    y_pred_all = np.asarray(coef_pred_all, dtype=np.float64)
else:
    req_pred = ['predict_sci_coefficients_default', 'mlp_artifacts',
                'coef_near_all', 'coef_far_all', 'ctx_near_all', 'ctx_far_all', 'ctx_sci_all']
    miss_pred = [k for k in req_pred if k not in globals()]
    if miss_pred:
        raise RuntimeError('Need predictions on all rows for randomized experiments. Missing: ' + ', '.join(miss_pred))
    y_pred_all = predict_sci_coefficients_default(
        mlp_artifacts,
        coef_near_phys=np.asarray(coef_near_all, dtype=np.float32),
        coef_far_phys=np.asarray(coef_far_all, dtype=np.float32),
        ctx_near_phys=np.asarray(ctx_near_all, dtype=np.float32),
        ctx_far_phys=np.asarray(ctx_far_all, dtype=np.float32),
        ctx_sci_phys=np.asarray(ctx_sci_all, dtype=np.float32),
    ).astype(np.float64)

if y_pred_all.shape != y_true_all.shape:
    raise RuntimeError(f'Prediction shape mismatch: y_pred_all={y_pred_all.shape}, y_true_all={y_true_all.shape}')

test_idx_arr = np.asarray(test_idx, dtype=int)
if test_idx_arr.size == 0:
    raise RuntimeError('test_idx is empty')
n_test = int(test_idx_arr.size)

# Baseline top-15 worst coefficients from the current held-out test split.
rmse_base = np.sqrt(np.mean((y_pred_all[test_idx_arr] - y_true_all[test_idx_arr]) ** 2, axis=0))
base_worst_idx = np.argsort(rmse_base)[::-1][:min(N_WORST, n_coef)]

if 'coef_names_all' in globals() and len(coef_names_all) == n_coef:
    coef_names = [str(x) for x in coef_names_all]
elif 'filtered_triplet' in globals() and 'coef_names' in filtered_triplet and len(filtered_triplet['coef_names']) == n_coef:
    coef_names = [str(x) for x in filtered_triplet['coef_names']]
else:
    coef_names = [f'coef_{j}' for j in range(n_coef)]

rng = np.random.default_rng(RNG_SEED)
all_top_sets = []
all_top_ranks = []

for exp in range(N_EXPERIMENTS):
    idx = rng.choice(n_rows, size=n_test, replace=False)
    rmse = np.sqrt(np.mean((y_pred_all[idx] - y_true_all[idx]) ** 2, axis=0))
    top_idx = np.argsort(rmse)[::-1][:min(N_WORST, n_coef)]
    all_top_sets.append(set(int(j) for j in top_idx))
    all_top_ranks.append({int(j): (r + 1) for r, j in enumerate(top_idx)})

rows = []
for j in base_worst_idx:
    appears = [j in s for s in all_top_sets]
    count = int(np.sum(appears))
    ranks = [d[j] for d in all_top_ranks if j in d]
    rows.append({
        'coef_index': int(j),
        'coef_name': coef_names[int(j)],
        'baseline_rmse': float(rmse_base[int(j)]),
        'appear_count': count,
        'appear_frac': float(count / N_EXPERIMENTS),
        'always_bad': bool(count == N_EXPERIMENTS),
        'mean_rank_when_present': (float(np.mean(ranks)) if ranks else np.nan),
    })

stability_df = (pd.DataFrame(rows)
                .sort_values(['appear_count', 'baseline_rmse'], ascending=[False, False])
                .reset_index(drop=True))

print(f'Randomized test-index stability check: N_EXPERIMENTS={N_EXPERIMENTS}, N_WORST={N_WORST}, n_test={n_test}')
print('Baseline set = top-15 worst coefficients from the current held-out test split.')
print('')
print(stability_df.to_string(index=False, float_format=lambda v: f'{v:.6g}'))
print('')
_n_always = int(stability_df['always_bad'].sum())
print(f'Always bad (present in top-{N_WORST} for all {N_EXPERIMENTS} experiments): {_n_always}/{len(stability_df)}')

# Also report any coefficients that were consistently bad even if not in the baseline top-15.
global_counts = np.zeros(n_coef, dtype=int)
for s in all_top_sets:
    for j in s:
        global_counts[j] += 1
always_global = np.flatnonzero(global_counts == N_EXPERIMENTS)
if always_global.size:
    _base_set = set(int(x) for x in base_worst_idx.tolist())
    global_df = pd.DataFrame({
        'coef_index': always_global.astype(int),
        'coef_name': [coef_names[int(j)] for j in always_global],
        'appear_count': global_counts[always_global],
        'baseline_in_top15': [bool(int(j) in _base_set) for j in always_global],
    }).sort_values('coef_index').reset_index(drop=True)
    print('')
    print('Coefficients that are in top-15 worst for ALL randomized experiments:')
    print(global_df.to_string(index=False))
else:
    print('')
    print('No coefficient is in the top-15 worst set for all randomized experiments.')

# Quick visual: appearance count for the baseline top-15 coefficients.
fig = go.Figure()
fig.add_trace(go.Bar(
    x=stability_df['coef_name'],
    y=stability_df['appear_count'],
    marker_color=['#1f77b4' if not a else '#d62728' for a in stability_df['always_bad']],
    text=stability_df['appear_count'],
    textposition='outside',
))
fig.update_layout(
    template='plotly_white',
    title=f'How often baseline top-{N_WORST} coefficients stay in top-{N_WORST} (N={N_EXPERIMENTS} randomized tests)',
    xaxis_title='Coefficient',
    yaxis_title='Appearance count across experiments',
    yaxis=dict(range=[0, N_EXPERIMENTS + 1]),
    height=420,
    margin=dict(l=60, r=20, t=70, b=120),
)
fig.show()

rmse_worst_stability_results = {
    'N_EXPERIMENTS': N_EXPERIMENTS,
    'N_WORST': N_WORST,
    'n_test': n_test,
    'baseline_worst_idx': base_worst_idx,
    'stability_df': stability_df,
    'global_counts': global_counts,
}
