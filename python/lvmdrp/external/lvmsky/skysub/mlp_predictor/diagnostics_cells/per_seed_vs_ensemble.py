# Per-seed vs ensemble diagnostic reader (no retraining).
# Since 2026-08-11 the trainer cell above already fits the 4-seed ensemble; this
# cell reads mlp_artifacts['members'] to show whether individual seeds carry
# per-group biases the ensemble averages out (calibrated uncertainty for §11.7).

required = ['filtered_triplet', 'mlp_artifacts', 'predict_sci_coefficients_default',
            '_group_indices_compress', '_metric_row',
            'coef_near_all', 'coef_far_all', 'coef_sci_all',
            'ctx_near_all', 'ctx_far_all', 'ctx_sci_all', 'test_idx']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run the trainer cell first. Missing: ' + ', '.join(_missing))
if not mlp_artifacts.get('is_ensemble', False):
    raise RuntimeError('mlp_artifacts is not an ensemble; the trainer cell should build it.')

_members = mlp_artifacts['members']
_seeds = mlp_artifacts['seeds']
_te = np.asarray(test_idx, dtype=int)

# Per-seed predictions on the full filtered set for group-bias tables.
_per_seed_pred_all = {}
for _seed, _member in zip(_seeds, _members):
    _per_seed_pred_all[_seed] = predict_sci_coefficients_default(
        _member,
        coef_near_phys=coef_near_all, coef_far_phys=coef_far_all,
        ctx_near_phys=ctx_near_all, ctx_far_phys=ctx_far_all,
        ctx_sci_phys=ctx_sci_all).astype(np.float32)
_ensemble_all = np.mean(np.stack([_per_seed_pred_all[s] for s in _seeds]),
                        axis=0).astype(np.float32)

_bias_rows = []
for gname, idx in _group_indices_compress.items():
    idx = np.asarray(idx, dtype=int)
    mean_true = float(np.mean(coef_sci_all[:, idx]))
    row = {'group': gname, 'n': int(idx.size), 'mean_true': mean_true}
    for _seed in _seeds:
        _pred = _per_seed_pred_all[_seed]
        row[f'bias_s{_seed}_%'] = (
            100.0 * (float(np.mean(_pred[:, idx])) - mean_true)
            / max(abs(mean_true), 1e-30))
    row['bias_ens_%'] = (
        100.0 * (float(np.mean(_ensemble_all[:, idx])) - mean_true)
        / max(abs(mean_true), 1e-30))
    _bias_rows.append(row)

print(f'Per-group mean coefficient bias across {len(_seeds)} seeds + ensemble mean '
      f'(all {coef_sci_all.shape[0]} filtered rows):')
print(pd.DataFrame(_bias_rows).to_string(index=False, float_format=lambda v: f'{v:.3g}'))

_seed_max_biases = [max(abs(r[f'bias_s{s}_%']) for r in _bias_rows) for s in _seeds]
_ens_max_bias = max(abs(r['bias_ens_%']) for r in _bias_rows)
print(f'\nMax |bias| per single seed: min={min(_seed_max_biases):.2f}%, '
      f'max={max(_seed_max_biases):.2f}%, mean={float(np.mean(_seed_max_biases)):.2f}%')
print(f'Max |bias| of ensemble mean:  {_ens_max_bias:.2f}%')
if _ens_max_bias < 1.0:
    print(f'Verdict: ensemble drives max |bias| below 1% -- N={len(_seeds)} seeds are '
          f'sufficient for the group-level unbiasedness that §11.1 requires.')
elif _ens_max_bias < 0.7 * min(_seed_max_biases):
    print(f'Verdict: ensemble helps ({_ens_max_bias:.2f}% vs single-seed min '
          f'{min(_seed_max_biases):.2f}%) but does not fall below 1%. '
          f'Bumping ensemble_seeds to 8 in default_dual_group_config would help.')
else:
    print('Verdict: ensemble does not meaningfully reduce max |bias|; the bias is '
          'systematic (not seed variance). Investigate calibration / loss balance.')
