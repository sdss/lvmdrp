# Correlation of prediction quality (WRMSE) with sci-pointing context variables.
# Reports Pearson (linear) and Spearman (rank) for the full row set and
# separately for train, val/test and LMC/SMC so the same axes can be compared
# in and out of the training domain.
import pandas as pd
from scipy.stats import pearsonr, spearmanr

required = ['triplet_full', 'wrmse_full', 'full_rows', 'is_train',
            'is_valtest', 'is_region_excluded']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError(
        'Missing kernel state: ' + ', '.join(_missing)
        + '. Run the sWRMSE_coef map cell above first.'
    )

ctx_sci_full = np.asarray(triplet_full['ctx_sci'], dtype=np.float64)
ctx_names_full = list(triplet_full['ctx_names'])
wrmse_arr = np.asarray(wrmse_full, dtype=np.float64)  # 2026-08-16: WRMSE

train_mask_t = is_train[full_rows]
valtest_mask_t = is_valtest[full_rows]
lmc_mask_t = is_region_excluded[full_rows]

subsets = [
    ('all', np.ones_like(train_mask_t, dtype=bool)),
    ('train', train_mask_t),
    ('valtest', valtest_mask_t),
    ('lmcsmc', lmc_mask_t),
]

rows = []
for j, cname in enumerate(ctx_names_full):
    x = ctx_sci_full[:, j]
    row = {'ctx': cname}
    for label, submask in subsets:
        m = submask & np.isfinite(x) & np.isfinite(wrmse_arr)
        n_pts = int(m.sum())
        row[f'n_{label}'] = n_pts
        if n_pts < 20 or np.std(x[m]) == 0.0 or np.std(wrmse_arr[m]) == 0.0:
            row[f'pearson_{label}'] = np.nan
            row[f'spearman_{label}'] = np.nan
        else:
            r_p, _ = pearsonr(x[m], wrmse_arr[m])
            r_s, _ = spearmanr(x[m], wrmse_arr[m])
            row[f'pearson_{label}'] = float(r_p)
            row[f'spearman_{label}'] = float(r_s)
    rows.append(row)

wrmse_ctx_corr_df = pd.DataFrame(rows).sort_values(
    by='pearson_all', key=lambda s: s.abs(), ascending=False
).reset_index(drop=True)

col_order = ['ctx',
             'pearson_all', 'spearman_all', 'n_all',
             'pearson_train', 'spearman_train', 'n_train',
             'pearson_valtest', 'spearman_valtest', 'n_valtest',
             'pearson_lmcsmc', 'spearman_lmcsmc', 'n_lmcsmc']
wrmse_ctx_corr_df = wrmse_ctx_corr_df[col_order]

print('Correlation of sWRMSE_coef vs sci-pointing context columns '
      '(sorted by |Pearson (all)|):')
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', 240,
                       'display.float_format', lambda v: f'{v: .3f}'):
    print(wrmse_ctx_corr_df.to_string(index=False))
