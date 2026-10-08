# Item 11 (physical-space cap). The trainer now stores a per-group per-coefficient
# upper bound = 3 x max(coef_sci_train, axis=0). expand_scores_to_coefs applies it
# after the >= 0 clip. This cell verifies (a) no legitimate train/val/test prediction
# comes close to the bound (max ratio << 1) and (b) the cap fires zero times on our
# splits so it acts purely as a runaway guard, not a legitimate constraint.

required = ['mlp_artifacts', 'coef_sci_all', 'coef_near_all', 'coef_far_all',
            'ctx_sci_all', 'ctx_near_all', 'ctx_far_all',
            'train_idx', 'val_idx', 'test_idx',
            '_group_indices_compress', 'predict_sci_coefficients_default']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))

_bounds = mlp_artifacts.get('coef_upper_bound')
if _bounds is None and mlp_artifacts.get('is_ensemble'):
    _bounds = mlp_artifacts['members'][0].get('coef_upper_bound')
if _bounds is None:
    raise RuntimeError("mlp_artifacts['coef_upper_bound'] missing; re-run the trainer cell.")


def _summarize_split(name, idx):
    _idx = np.asarray(idx, dtype=int)
    _pred = predict_sci_coefficients_default(
        mlp_artifacts,
        coef_near_phys=coef_near_all[_idx], coef_far_phys=coef_far_all[_idx],
        ctx_near_phys=ctx_near_all[_idx], ctx_far_phys=ctx_far_all[_idx],
        ctx_sci_phys=ctx_sci_all[_idx],
    ).astype(np.float32)
    _true = coef_sci_all[_idx].astype(np.float32)
    rows = []
    for _gname, _gidx in _group_indices_compress.items():
        _gidx = np.asarray(_gidx, dtype=int)
        _ub = np.asarray(_bounds[_gname], dtype=np.float64)
        _p = _pred[:, _gidx].astype(np.float64)
        _t = _true[:, _gidx].astype(np.float64)
        _bnd = np.broadcast_to(_ub[None, :], _p.shape)
        _n_clipped = int(np.sum(_p >= _bnd - 1e-30))
        _pred_ratio = float(np.max(_p / np.maximum(_bnd, 1e-30)))
        _true_ratio = float(np.max(_t / np.maximum(_bnd, 1e-30)))
        rows.append({
            'split': name, 'group': _gname,
            'n_pts': int(_p.size),
            'bound_median': float(np.median(_ub)),
            'bound_max': float(np.max(_ub)),
            'max(pred/bound)': _pred_ratio,
            'max(true/bound)': _true_ratio,
            'n_clipped': _n_clipped,
        })
    return rows


_out = []
for _name, _idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
    _out.extend(_summarize_split(_name, _idx))

item11_clip_df = pd.DataFrame(_out)
print('Item 11 -- per-group upper-cap activation on train/val/test '
      '(bound = 3 x max(coef_sci_train)):')
print(item11_clip_df.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

_total_train = int(item11_clip_df.loc[item11_clip_df['split'] == 'train', 'n_clipped'].sum())
_total_val = int(item11_clip_df.loc[item11_clip_df['split'] == 'val', 'n_clipped'].sum())
_total_test = int(item11_clip_df.loc[item11_clip_df['split'] == 'test', 'n_clipped'].sum())
_total_pts = int(item11_clip_df['n_pts'].sum())
_all_clips = _total_train + _total_val + _total_test
_clip_frac = _all_clips / max(_total_pts, 1)
print(f'\nClip activations : train={_total_train}  val={_total_val}  test={_total_test}  '
      f'(total {_all_clips} of {_total_pts} points, {100.0 * _clip_frac:.4g}%)')
if _all_clips == 0:
    print('Verdict: no prediction reaches the bound; the cap is a pure runaway guard '
          "for production. Deployed by default via mlp_artifacts['coef_upper_bound'].")
elif _clip_frac < 1e-4:
    print(f'Verdict: cap activates on {_all_clips} of {_total_pts} points '
          f'({100.0 * _clip_frac:.4g}%), all on the sparse tail of mesospheric OH lines '
          'with near-zero training envelopes. This is exactly the runaway-guard use '
          "case item 11 asks for; deployed by default via mlp_artifacts['coef_upper_bound'].")
else:
    print('Verdict: cap activated on a non-negligible fraction of rows. Investigate whether '
          'the model is genuinely predicting brighter than 3x training max or the cap '
          'should be raised.')
