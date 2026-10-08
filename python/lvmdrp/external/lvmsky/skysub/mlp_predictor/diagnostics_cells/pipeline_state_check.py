# --- check pipeline state after airmass filter ---
print('=== Pipeline state check ===')
print(f'filtered_triplet rows       : {filtered_triplet["coef_near"].shape[0]}')
print(f'  has compress_train_idx?   : {"compress_train_idx" in filtered_triplet}')
if 'compress_train_idx' in filtered_triplet:
    print(f'  compress_train_idx max    : {int(np.max(filtered_triplet["compress_train_idx"]))}')
    print(f'  compress_test_idx  max    : {int(np.max(filtered_triplet["compress_test_idx"]))}')

print()
print('=== mlp_artifacts (trained model) ===')
print(f'train_idx max               : {int(mlp_artifacts["train_idx"].max())}')
print(f'val_idx max                 : {int(mlp_artifacts["val_idx"].max())}')
print(f'test_idx max                : {int(mlp_artifacts["test_idx"].max())}')
print(f'group_score_dims            : {mlp_artifacts["group_score_dims"]}')
_gsd = mlp_artifacts['group_score_dims']
for g, n in _gsd.items():
    w = 1.0 / (float(n) ** 0.5)
    print(f'  loss weight w_{g:<12s} = 1/sqrt({n:>3d}) = {w:.4f}')

print()
print('=== group_compressors (fit state) ===')
for g, comp in group_compressors.items():
    print(f'  {g:<14s} n_input={comp["n_coef_group"]:>4d} n_kept={comp["kept"].size:>3d} transform={comp["kind"]:>8s}')

print()
print('=== index alignment check ===')
cur_n = filtered_triplet['coef_near'].shape[0]
mlp_max = int(mlp_artifacts['test_idx'].max())
if 'compress_train_idx' in filtered_triplet:
    ftr_max = int(np.max(filtered_triplet['compress_train_idx']))
    if ftr_max >= cur_n or mlp_max >= cur_n:
        print(f'  MISMATCH: filtered_triplet has {cur_n} rows but mlp_artifacts indices go up to {mlp_max}, filtered_triplet indices to {ftr_max}')
    else:
        print(f'  OK: mlp_artifacts test_idx.max={mlp_max} < filtered_triplet rows={cur_n}')
else:
    if mlp_max >= cur_n:
        print(f'  MISMATCH: current filtered_triplet has {cur_n} rows, but mlp_artifacts.test_idx.max={mlp_max}. Downstream ops using test_idx on filtered_triplet will index out of range.')
    else:
        print(f'  Model was trained on a different filtered_triplet (compress_train_idx lost). Current filtered_triplet has {cur_n} rows; model expects up to {mlp_max}.')

# Continuum-relevant group check
print()
print('=== continuum-related predictions (sanity check on median coef amplitude) ===')
_coef_true = np.asarray(filtered_triplet['coef_sci'], dtype=np.float32)
_coef_pred = predict_sci_coefficients_default(
    mlp_artifacts,
    coef_near_phys=np.asarray(filtered_triplet['coef_near'], dtype=np.float32),
    coef_far_phys=np.asarray(filtered_triplet['coef_far'], dtype=np.float32),
    ctx_near_phys=np.asarray(filtered_triplet['ctx_near'], dtype=np.float32),
    ctx_far_phys=np.asarray(filtered_triplet['ctx_far'], dtype=np.float32),
    ctx_sci_phys=np.asarray(filtered_triplet['ctx_sci'], dtype=np.float32),
).astype(np.float32)
_names = [str(n).lower() for n in filtered_triplet['coef_names']]
_gidx = _build_group_indices(filtered_triplet['coef_names'])
for gname, idx in _gidx.items():
    idx = np.asarray(idx, dtype=int)
    med_true = float(np.median(_coef_true[:, idx]))
    med_pred = float(np.median(_coef_pred[:, idx]))
    mean_true = float(np.mean(_coef_true[:, idx]))
    mean_pred = float(np.mean(_coef_pred[:, idx]))
    bias = mean_pred - mean_true
    print(f'  {gname:<12s} n={idx.size:>3d}  median true/pred = {med_true:.3g} / {med_pred:.3g}   mean true/pred = {mean_true:.3g} / {mean_pred:.3g}   mean bias = {bias:+.3g} ({100*bias/max(abs(mean_true), 1e-30):+.1f}%)')
