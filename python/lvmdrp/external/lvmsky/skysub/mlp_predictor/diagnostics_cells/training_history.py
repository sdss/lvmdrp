# Training progress against epoch, per ensemble member.
#
# Reads the per-member `history` (train/val loss) and `blend_history` (the
# near-arm blend alpha per group) that the trainer records and, since
# 2026-09-23, `serialization.save_ensemble` persists -- so this works the same
# on a freshly trained ensemble and on one restored from a .pt.
#
# What to look for:
#   * val still falling at the last epoch -> the budget truncated the run;
#     raise n_epochs (the recorded stall before the eventual best epoch is
#     28-131 epochs, which is why patience is 100).
#   * best-epoch markers clustered far from the end -> patience is firing
#     inside plateau noise and the restored weights are early ones.
#   * val flat while train keeps falling -> overfitting; the loop restores
#     `best_state`, so this costs compute rather than quality.
#   * alpha panel: a group whose alpha never leaves its init is not learning
#     the near/far blend (see the alpha_lr_mult note in the trainer config).
#
# Cheap: it plots recorded numbers and recomputes nothing.

required = ['mlp_artifacts']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run the trainer cell first. Missing: ' + ', '.join(_missing))

_members_th = mlp_artifacts.get('members') or []
if not _members_th:
    raise RuntimeError('mlp_artifacts carries no members to plot.')

_hist_th = [list(m.get('history') or []) for m in _members_th]
# A missing history is a benign state, not a kernel-state error: every .pt
# written before 2026-09-23 lacks it, and such an ensemble is still a correct
# predictor.  Skip with a message instead of raising, so restoring an older
# ensemble does not break a notebook run.  (A missing `mlp_artifacts`, above,
# IS an error -- that means the trainer cell has not run.)
training_history_result = None
if not any(_hist_th):
    print('No per-epoch training history on this ensemble: it was saved before '
          '2026-09-23, when save_ensemble began persisting it. Predictions are '
          'unaffected; retrain to record the curves. Skipping the plot.')
else:

    _seeds_th = list(mlp_artifacts.get('seeds') or range(len(_members_th)))
    _best_ep_th = list(mlp_artifacts.get('best_epochs') or
                       [m.get('best_epoch') for m in _members_th])
    _best_vl_th = list(mlp_artifacts.get('best_val_losses') or
                       [m.get('best_val_loss') for m in _members_th])

    # Alpha groups, in the order the trainer snapshots them.  Since 2026-09-23
    # the ctx-dependent groups (moon / zodi / continuum) also carry
    # `<group>_p16` / `_p84`: their alpha is per row, so the group key holds the
    # median over the validation rows and the pair gives its spread.  Filter the
    # suffixed keys out of the group list or they plot as separate series.
    _alpha_groups_th, _has_band_th = [], {}
    for _bh in (list(m.get('blend_history') or []) for m in _members_th):
        if _bh:
            _keys_th = list(_bh[0].keys())
            _alpha_groups_th = [k for k in _keys_th
                                if k != 'epoch' and not k.endswith(('_p16', '_p84'))]
            _has_band_th = {g: (f'{g}_p16' in _keys_th and f'{g}_p84' in _keys_th)
                            for g in _alpha_groups_th}
            break

    _fig_th = make_subplots(
        rows=1, cols=2 if _alpha_groups_th else 1,
        subplot_titles=(['loss vs epoch'] +
                        (['near-arm blend alpha vs epoch'] if _alpha_groups_th else [])),
        horizontal_spacing=0.09)

    _pal_th = px.colors.qualitative.Plotly


    def _series_th(rows, key):
        return ([r['epoch'] for r in rows if key in r],
                [r[key] for r in rows if key in r])


    for _i_th, (_sd_th, _h_th) in enumerate(zip(_seeds_th, _hist_th)):
        if not _h_th:
            continue
        _c_th = _pal_th[_i_th % len(_pal_th)]
        _x_th, _y_th = _series_th(_h_th, 'train_loss')
        _fig_th.add_trace(go.Scatter(
            x=_x_th, y=_y_th, mode='lines', name=f'seed {_sd_th} train',
            legendgroup=f's{_sd_th}', line=dict(color=_c_th, width=1, dash='dot'),
            opacity=0.55, hovertemplate='epoch %{x}<br>train %{y:.6f}<extra></extra>'),
            row=1, col=1)
        _x_th, _y_th = _series_th(_h_th, 'val_loss')
        _fig_th.add_trace(go.Scatter(
            x=_x_th, y=_y_th, mode='lines', name=f'seed {_sd_th} val',
            legendgroup=f's{_sd_th}', line=dict(color=_c_th, width=1.6),
            hovertemplate='epoch %{x}<br>val %{y:.6f}<extra></extra>'), row=1, col=1)
        # Best epoch: the weights actually kept, which is what the ensemble uses.
        _be_th = _best_ep_th[_i_th] if _i_th < len(_best_ep_th) else None
        _bv_th = _best_vl_th[_i_th] if _i_th < len(_best_vl_th) else None
        if _be_th is not None and _bv_th is not None and _be_th > 0:
            _fig_th.add_trace(go.Scatter(
                x=[_be_th], y=[_bv_th], mode='markers', showlegend=False,
                legendgroup=f's{_sd_th}', marker=dict(color=_c_th, size=9,
                                                      symbol='circle-open',
                                                      line=dict(width=2)),
                hovertemplate=(f'seed {_sd_th} best<br>epoch %{{x}}'
                               '<br>val %{y:.6f}<extra></extra>')), row=1, col=1)

    if _alpha_groups_th:
        # Mean over members: the per-seed spread is small next to the drift, and
        # one line per group per seed would be unreadable at 10 seeds.
        for _gi_th, _g_th in enumerate(_alpha_groups_th):
            _by_ep_th = defaultdict(list)
            for m in _members_th:
                for _r_th in (m.get('blend_history') or []):
                    if _g_th in _r_th:
                        _by_ep_th[int(_r_th['epoch'])].append(float(_r_th[_g_th]))
            if not _by_ep_th:
                continue
            _eps_th = sorted(_by_ep_th)
            _col_th = _pal_th[_gi_th % len(_pal_th)]
            # p16-p84 band for the ctx groups: their alpha varies row to row,
            # and a median alone hides whether the context term is doing
            # anything.  A flat median with a widening band is still learning.
            if _has_band_th.get(_g_th):
                _lo_th, _hi_th = {}, {}
                for m in _members_th:
                    for _r_th in (m.get('blend_history') or []):
                        if f'{_g_th}_p16' in _r_th:
                            _lo_th.setdefault(int(_r_th['epoch']), []).append(
                                float(_r_th[f'{_g_th}_p16']))
                            _hi_th.setdefault(int(_r_th['epoch']), []).append(
                                float(_r_th[f'{_g_th}_p84']))
                _eb_th = sorted(_lo_th)
                if _eb_th:
                    _rgba_th = (f"rgba({int(_col_th[1:3],16)},{int(_col_th[3:5],16)},"
                                f"{int(_col_th[5:7],16)},0.15)")
                    _fig_th.add_trace(go.Scatter(
                        x=_eb_th + _eb_th[::-1],
                        y=([float(np.mean(_hi_th[e])) for e in _eb_th]
                           + [float(np.mean(_lo_th[e])) for e in _eb_th[::-1]]),
                        fill='toself', fillcolor=_rgba_th,
                        line=dict(width=0), hoverinfo='skip',
                        showlegend=False, legendgroup=f'a{_g_th}'), row=1, col=2)
            _fig_th.add_trace(go.Scatter(
                x=_eps_th, y=[float(np.mean(_by_ep_th[e])) for e in _eps_th],
                mode='lines', legendgroup=f'a{_g_th}',
                name=(f'alpha {_g_th}'
                      + (' (ctx, median)' if _has_band_th.get(_g_th) else '')),
                line=dict(color=_col_th, width=1.6),
                hovertemplate=(f'{_g_th}<br>epoch %{{x}}'
                               '<br>alpha %{y:.3f}<extra></extra>')), row=1, col=2)

    _fig_th.update_xaxes(title_text='epoch', row=1, col=1)
    # Log y: the first few epochs are orders of magnitude above the plateau, and
    # on a linear axis they flatten everything that matters into the baseline.
    _fig_th.update_yaxes(title_text='loss (log)', type='log', row=1, col=1)
    if _alpha_groups_th:
        _fig_th.update_xaxes(title_text='epoch', row=1, col=2)
        _fig_th.update_yaxes(title_text='alpha (1 = near arm only)', row=1, col=2)
    _fig_th.update_layout(
        height=430, width=1150,
        title_text=(f'Training progress -- {len(_hist_th)} member(s), '
                    f'{max(len(h) for h in _hist_th)} epochs recorded'),
        margin=dict(t=70, b=50))
    _fig_th.show()

    _rows_th = []
    for _i_th, (_sd_th, _h_th) in enumerate(zip(_seeds_th, _hist_th)):
        if not _h_th:
            continue
        _v_th = [r['val_loss'] for r in _h_th if 'val_loss' in r]
        _be_th = _best_ep_th[_i_th] if _i_th < len(_best_ep_th) else None
        _n_th = len(_h_th)
        _rows_th.append({
            'seed': _sd_th,
            'epochs_run': _n_th,
            'best_epoch': _be_th,
            'best_val': (_best_vl_th[_i_th] if _i_th < len(_best_vl_th) else None),
            'final_val': _v_th[-1] if _v_th else None,
            # How much of the budget ran after the kept weights. Large = patience
            # never fired and the tail was wasted; ~0 = the run was still improving
            # at the end and n_epochs is the binding constraint.
            'epochs_after_best': (None if _be_th is None else _n_th - _be_th),
            'early_stopped': _n_th < int(mlp_artifacts.get('config', {})
                                         .get('n_epochs', _n_th)),
        })
    training_history_result = pd.DataFrame(_rows_th)
    print(f'Per-seed training summary (n_epochs budget = '
          f'{mlp_artifacts.get("config", {}).get("n_epochs", "?")}, '
          f'patience = {mlp_artifacts.get("config", {}).get("patience", "?")}):')
    print(training_history_result.to_string(index=False,
                                            float_format=lambda v: f'{v:.6g}'))

    _still_falling_th = [r['seed'] for r in _rows_th
                         if r['epochs_after_best'] is not None
                         and r['epochs_after_best'] <= 5]
    if _still_falling_th:
        print(f'\n{len(_still_falling_th)}/{len(_rows_th)} seed(s) had their best '
              f'epoch within 5 of the end ({_still_falling_th}): the epoch budget '
              f'is binding, not the patience. Raising n_epochs should still help.')
    else:
        print(f'\nEvery seed found its best epoch with room to spare '
              f'(median {int(np.median([r["epochs_after_best"] for r in _rows_th]))} '
              f'epochs of budget left after it), so n_epochs is not the limit.')
