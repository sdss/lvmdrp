# encoding: utf-8
"""Tests of the sky-line measurement of the large-scale flat-field error (flatgradient)."""

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from lvmdrp.core.rss import RSS
from lvmdrp.functions import flatgradient as fg

# the plane and factors put into the synthetic frames
GX, GY = 6.0, -3.0
FACTORS = np.array([1.03, 1.00, 0.97])
WAVE = np.arange(3600.0, 9800.0, 0.5)


def _positions(n, rng, radius=15.0):
    """Random fiber positions in a disk of `radius` mm."""
    r = radius * np.sqrt(rng.uniform(size=n))
    t = rng.uniform(0, 2 * np.pi, size=n)
    return r * np.cos(t), r * np.sin(t)


def _throughput(x, y, spec, gx=GX, gy=GY, factors=FACTORS):
    return factors[spec - 1] * (1 + (gx * x + gy * y) / 1000)


def _line_centers():
    centers = [c for c, _ in fg.SKYLINES.values()]
    centers += [0.5 * sum(window) for _, window, _ in fg.HELDOUT_LINES.values()]
    return centers


def _spectra(throughput, rng, noise=0.002):
    """Continuum of 1 plus Gaussian sky lines, all scaled by the throughput of each fiber."""
    spec = np.ones((throughput.size, WAVE.size))
    for c in _line_centers():
        spec += 20.0 * np.exp(-0.5 * ((WAVE - c) / 1.0) ** 2)
    spec *= throughput[:, None]
    return spec * (1 + noise * rng.standard_normal(spec.shape))


@pytest.fixture
def frame():
    """A synthetic channel-combined frame: 1200 science fibers, 60 fibers in each sky IFU."""
    rng = np.random.default_rng(42)
    nsci, nsky = 1200, 60
    xs, ys = _positions(nsci, rng)
    spec_sci = np.repeat([1, 2, 3], nsci // 3)
    spec_sky = np.tile([1, 2, 3], nsky // 3)
    xk, yk = _positions(nsky, rng, radius=1.0)
    telescope = np.array(['Sci'] * nsci + ['SkyE'] * nsky + ['SkyW'] * nsky)
    x = np.concatenate([xs, xk, xk])
    y = np.concatenate([ys, yk, yk])
    spec = np.concatenate([spec_sci, spec_sky, spec_sky])
    # the sky IFUs see the factors, but not the plane across the science IFU
    thr = np.where(telescope == 'Sci', _throughput(x, y, spec), FACTORS[spec - 1])
    flux = _spectra(thr, rng)
    # a few bright stars in the science IFU, which must be rejected
    stars = rng.choice(nsci, 15, replace=False)
    flux[stars] += 30.0
    slitmap = Table({'telescope': telescope, 'fibstatus': np.zeros(x.size, dtype=int),
                     'spectrographid': spec, 'xpmm': x, 'ypmm': y})
    return flux, slitmap


def test_line_flux_removes_continuum():
    wave = np.arange(5500.0, 5650.0, 0.5)
    area = 12.0
    flux = 3.0 + area / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((wave - 5577.35) / 1.0) ** 2)
    z = fg.line_flux(flux[None, :], wave, (5573.35, 5581.35), fg.SKYLINES['b'][1])
    assert z[0] == pytest.approx(area, rel=0.01)


def test_line_flux_nan_in_window():
    wave = np.arange(5500.0, 5650.0, 0.5)
    flux = np.ones((2, wave.size))
    flux[1, np.argmin(np.abs(wave - 5577.0))] = np.nan
    z = fg.line_flux(flux, wave, (5573.35, 5581.35), fg.SKYLINES['b'][1])
    assert np.isfinite(z[0]) and np.isnan(z[1])


def test_star_mask_flags_bright_fibers():
    rng = np.random.default_rng(1)
    flux = 1 + 0.01 * rng.standard_normal((500, WAVE.size))
    flux[[3, 77, 400]] += 50.0
    mask = fg.star_mask(flux, WAVE, 'r')
    assert mask[[3, 77, 400]].all()
    assert mask.mean() < 0.1


def test_fit_plane_recovers_plane_and_factors():
    rng = np.random.default_rng(2)
    x, y = _positions(1500, rng)
    spec = rng.integers(1, 4, size=1500)
    z = 50.0 * _throughput(x, y, spec) * (1 + 0.01 * rng.standard_normal(1500))
    r = fg.fit_plane(x, y, z, spec)
    assert r['gx'] == pytest.approx(GX, abs=0.1)
    assert r['gy'] == pytest.approx(GY, abs=0.1)
    assert [r['f1'], r['f2'], r['f3']] == pytest.approx(list(FACTORS / FACTORS.mean()), abs=0.002)
    assert r['rms'] == pytest.approx(1.0, abs=0.2)
    assert r['n'] > 1450


def test_fit_plane_clips_outliers():
    rng = np.random.default_rng(3)
    x, y = _positions(1500, rng)
    spec = rng.integers(1, 4, size=1500)
    z = 50.0 * _throughput(x, y, spec) * (1 + 0.01 * rng.standard_normal(1500))
    bad = rng.choice(1500, 20, replace=False)
    z[bad] *= 3.0
    r = fg.fit_plane(x, y, z, spec)
    assert r['gx'] == pytest.approx(GX, abs=0.1)
    assert r['gy'] == pytest.approx(GY, abs=0.1)
    assert r['n'] <= 1480


def test_fit_plane_too_few_fibers():
    rng = np.random.default_rng(4)
    x, y = _positions(50, rng)
    assert fg.fit_plane(x, y, np.ones(50), np.ones(50, dtype=int)) is None


def test_sky_fiber_factors():
    rng = np.random.default_rng(5)
    spec = np.tile([1, 2, 3], 20)
    z = 10.0 * FACTORS[spec - 1] * (1 + 0.005 * rng.standard_normal(spec.size))
    r = fg.sky_fiber_factors(z, spec)
    assert [r['f1'], r['f2'], r['f3']] == pytest.approx(list(FACTORS / FACTORS.mean()), abs=0.005)
    assert r['n'] == spec.size
    assert fg.sky_fiber_factors(z[:9], spec[:9]) is None


def test_measure_skyline_planes(frame):
    flux, slitmap = frame
    res = fg.measure_skyline_planes(flux, WAVE, slitmap)
    for key in ('b', 'r', 'z', 'r_oh', 'z_oh'):
        assert res[key]['gx'] == pytest.approx(GX, abs=0.1), key
        assert res[key]['gy'] == pytest.approx(GY, abs=0.1), key
        assert res[key]['f1'] / res[key]['f3'] == pytest.approx(FACTORS[0] / FACTORS[2], abs=0.003)
        # the bright stars are not fitted
        assert res[key]['n'] <= 1185
    for key in ('skye_b', 'skye_r', 'skye_z', 'skyw_b', 'skyw_r', 'skyw_z'):
        assert res[key]['f1'] / res[key]['f3'] == pytest.approx(FACTORS[0] / FACTORS[2], abs=0.005)


def test_measure_skyline_planes_options(frame):
    flux, slitmap = frame
    res = fg.measure_skyline_planes(flux, WAVE, slitmap, heldout=False, sky_fibers=False)
    assert set(res) == {'b', 'r', 'z'}


def test_skyline_plane_cards(frame):
    flux, slitmap = frame
    res = fg.measure_skyline_planes(flux, WAVE, slitmap)
    res['r_oh'] = None
    cards = {kw: (value, comment) for kw, value, comment in fg.skyline_plane_cards(res)}
    assert cards['HIERARCH SKYPLANE Z GX'][0] == pytest.approx(GX, abs=0.1)
    assert cards['HIERARCH SKYPLANE SKYE B F1'][0] == pytest.approx(res['skye_b']['f1'], abs=1e-5)
    assert np.isnan(cards['HIERARCH SKYPLANE ROH GX'][0])
    assert cards['HIERARCH SKYPLANE ROH N'][0] == 0
    assert len(cards) == 5 * 7 + 6 * 4


def test_measure_cframe_writes_header(frame):
    flux, slitmap = frame
    rss = RSS(data=flux, wave=WAVE, slitmap=slitmap, header=fits.Header())
    res = fg.measure_cframe(rss)
    assert rss._header['HIERARCH SKYPLANE Z GY'] == pytest.approx(res['z']['gy'], abs=1e-5)
    assert rss._header['HIERARCH SKYPLANE ZOH N'] == res['z_oh']['n']


def test_measure_cframe_needs_1d_wave(frame):
    flux, slitmap = frame
    rss = RSS(data=flux, wave=np.tile(WAVE, (flux.shape[0], 1)), slitmap=slitmap, header=fits.Header())
    with pytest.raises(ValueError):
        fg.measure_cframe(rss)


def test_measure_cframe_failure_does_not_raise(frame):
    flux, slitmap = frame
    rss = RSS(data=flux, wave=np.tile(WAVE, (flux.shape[0], 1)), slitmap=slitmap, header=fits.Header())
    assert fg.measure_cframe(rss, raise_errors=False) is None
    assert 'HIERARCH SKYPLANE Z GX' not in rss._header


def test_skyline_plane_columns_match_cards(frame):
    flux, slitmap = frame
    res = fg.measure_skyline_planes(flux, WAVE, slitmap)
    keywords = {kw.replace('HIERARCH ', '') for kw, _, _ in fg.skyline_plane_cards(res)}
    columns = fg.skyline_plane_columns()
    assert set(columns.values()) == keywords
    assert columns['skyplane_z_gx'] == 'SKYPLANE Z GX'
    assert columns['skyplane_skyw_r_f3'] == 'SKYPLANE SKYW R F3'


# ------------------------------------------------------------------ the model of the flat-field error
EPOCHS = (60000, 60100)
# true coefficients: per epoch (constant, drift t1) and shared (focus, sinK), per component
TRUE = {'f1': {'c0': (1.0, 1.5), 't1': (0.3, -0.2), 'focus': 0.4, 'sinK': 0.2},
        'f2': {'c0': (-0.5, -1.0), 't1': (0.0, 0.1), 'focus': -0.1, 'sinK': -0.3},
        'gx': {'c0': (4.0, 6.0), 't1': (0.5, 0.3), 'focus': 1.0, 'sinK': 1.2},
        'gy': {'c0': (2.0, 3.0), 't1': (-0.4, 0.6), 'focus': -4.0, 'sinK': 0.8}}


def _true_component(comp, mjd, focus, kmir, epoch):
    i = (np.asarray(epoch) == EPOCHS[1]).astype(int)
    lo = np.where(i == 1, 60100.0, 60000.0)
    u = 2 * (mjd - lo) / 90.0 - 1
    c = TRUE[comp]
    return (np.choose(i, c['c0']) + np.choose(i, c['t1']) * u + c['focus'] * (focus - fg.FOCUS0)
            + c['sinK'] * np.sin(np.radians(kmir)))


def _drpall(n=1500, seed=7, noise=0.05):
    """A synthetic drpall table: exposures in two epochs, measured planes from the known model."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    epoch = rng.choice(EPOCHS, n)
    mjd = epoch + rng.uniform(0, 90, n)
    focus = rng.normal(35.3, 0.3, n)
    kmir = rng.uniform(-120, 120, n)
    d = {'expnum': np.arange(n), 'mjd': mjd, 'exptime': np.full(n, 900.0), 'drpqual': np.zeros(n, int),
         'sci_focpos': focus, 'sci_kmpos': kmir}
    for ch in 'brz':
        F1 = _true_component('f1', mjd, focus, kmir, epoch) + noise * rng.standard_normal(n)
        F2 = _true_component('f2', mjd, focus, kmir, epoch) + noise * rng.standard_normal(n)
        F = np.column_stack([F1, F2, -F1 - F2])
        for k in range(3):
            d[f'skyplane_{ch}_f{k + 1}'] = 1 + F[:, k] / 100
        for comp in ('gx', 'gy'):
            d[f'skyplane_{ch}_{comp}'] = _true_component(comp, mjd, focus, kmir, epoch) + noise * rng.standard_normal(n)
        d[f'skyplane_{ch}_n'] = np.full(n, 1550)
    return pd.DataFrame(d)


def test_model_design():
    mjd = np.array([60010.0, 60150.0])
    A, names = fg.model_design(mjd, np.array([35.5, 35.0]), np.array([0.0, 90.0]), np.array(EPOCHS),
                               {60000: (60000.0, 60090.0), 60100: (60100.0, 60190.0)})
    assert A.shape == (2, 2 * len(fg.EPOCH_TERMS) + len(fg.SHARED_TERMS))
    assert names[:4] == ['c0_60000', 't1_60000', 't2_60000', 't3_60000']
    assert A[0, names.index('c0_60000')] == 1 and A[0, names.index('c0_60100')] == 0
    assert A[1, names.index('focus_hinge')] == pytest.approx(0.5)
    assert A[1, names.index('sinK')] == pytest.approx(1.0)


def test_fit_flat_model_recovers_model():
    table = _drpall()
    model = fg.fit_flat_model(table, EPOCHS)
    for ch in 'brz':
        assert sum(model['nexp'][ch].values()) == len(table)
        for comp in ('f1', 'f2', 'gx', 'gy'):
            c = model['coeffs'][ch][comp]
            assert c['focus'] == pytest.approx(TRUE[comp]['focus'], abs=0.03)
            assert c['sinK'] == pytest.approx(TRUE[comp]['sinK'], abs=0.03)
            assert c['cosK'] == pytest.approx(0.0, abs=0.03)
    # the predictions match the true model (the drift ranges come from the data, not exactly 90 days)
    mjd = table['mjd'].to_numpy()
    epoch = fg.assign_epochs(mjd, EPOCHS)
    A, names = fg.model_design(mjd, table['sci_focpos'].to_numpy(), table['sci_kmpos'].to_numpy(), epoch,
                               model['ranges'])
    for comp in ('gx', 'gy', 'f1'):
        pred = A @ np.array([model['coeffs']['z'][comp][n] for n in names])
        true = _true_component(comp, mjd, table['sci_focpos'].to_numpy(), table['sci_kmpos'].to_numpy(), epoch)
        assert np.max(np.abs(pred - true)) < 0.05


def test_select_exposures():
    table = _drpall(n=300)
    table.loc[0, 'drpqual'] = 1
    table.loc[1, 'exptime'] = 100.0
    table.loc[2, 'sci_focpos'] = -999.0
    table.loc[3, 'skyplane_z_gx'] = -999.9
    table.loc[4, 'skyplane_z_gy'] += 50.0
    table.loc[5, 'skyplane_z_n'] = 500
    keep = fg.select_exposures(table, 'z', fg.assign_epochs(table['mjd'], EPOCHS))
    assert not keep[:6].any()
    assert keep[6:].mean() > 0.97


def test_fit_flat_model_too_few_exposures():
    with pytest.raises(ValueError):
        fg.fit_flat_model(_drpall(n=12), EPOCHS)


def test_write_read_evaluate(tmp_path):
    table = _drpall()
    model = fg.fit_flat_model(table, EPOCHS, channels='z')
    path = tmp_path / 'lvm-mfiberflat_gradient-z.fits'
    fg.write_flat_model(model, EPOCHS[1], 'z', str(path), drpver='test')
    m = fg.read_flat_model(str(path))
    assert m['channel'] == 'z' and m['epoch'] == EPOCHS[1]
    assert m['range'] == model['ranges'][EPOCHS[1]]
    assert fits.getheader(str(path))['DRPVER'] == 'test'
    mjd, focus, kmir = 60150.0, 35.1, 40.0
    A, names = fg.model_design(np.array([mjd]), np.array([focus]), np.array([kmir]), np.array([EPOCHS[1]]),
                               model['ranges'])
    val = fg.evaluate_flat_model(m, mjd, focus, kmir)
    for comp in fg.MODEL_COMPONENTS:
        assert val[comp] == pytest.approx(float(A[0] @ [model['coeffs']['z'][comp][n] for n in names]), abs=1e-9)
    assert val['gx'] == pytest.approx(_true_component('gx', mjd, focus, kmir, EPOCHS[1]), abs=0.05)
    # the drift is held at its end value beyond the range of the data
    hi = m['range'][1]
    assert fg.evaluate_flat_model(m, hi + 50, focus, kmir) == fg.evaluate_flat_model(m, hi, focus, kmir)


def test_create_flat_models(tmp_path):
    drpall = str(tmp_path / 'drpall-test.h5')
    _drpall().to_hdf(drpall, key='summary', format='table')
    out = tmp_path / 'models'
    paths = fg.create_flat_models(drpall, EPOCHS, drpver='test', out_dir=str(out), dry_run=True)
    assert len(paths) == 6 and not out.exists()
    paths = fg.create_flat_models(drpall, EPOCHS, drpver='test', out_dir=str(out))
    assert paths[(EPOCHS[0], 'b')] == str(out / str(EPOCHS[0]) / 'lvm-mfiberflat_gradient-b.fits')
    assert all(os.path.exists(p) for p in paths.values())


# ------------------------------------------------------------------ applying the model to a frame
def _constant_model(channel, gx, gy, factors=FACTORS, epoch=60000):
    """A model that predicts the same factors and plane for every exposure."""
    f = 100 * (np.asarray(factors) / np.mean(factors) - 1)
    coeffs = {comp: {t: 0.0 for t in fg.EPOCH_TERMS + fg.SHARED_TERMS} for comp in fg.MODEL_COMPONENTS}
    for comp, v in zip(fg.MODEL_COMPONENTS, (f[0], f[1], f[2], gx, gy)):
        coeffs[comp]['c0'] = v
    return {'channel': channel, 'epoch': epoch, 'range': (60000.0, 60090.0), 'focus0': fg.FOCUS0,
            'coeffs': coeffs}


def _cframe(frame, header=None, gx_z=None):
    """An RSS from the synthetic frame, with the header values the correction needs.

    With `gx_z`, the science fibers redward of 7600 A get a different x slope, so b and z disagree.
    """
    flux, slitmap = frame
    flux = flux.copy()
    if gx_z is not None:
        sci = np.asarray(slitmap['telescope']) == 'Sci'
        x, y = np.asarray(slitmap['xpmm'])[sci], np.asarray(slitmap['ypmm'])[sci]
        spec = np.asarray(slitmap['spectrographid'])[sci]
        red = WAVE > 7600
        flux[np.ix_(sci, red)] *= (_throughput(x, y, spec, gx=gx_z) / _throughput(x, y, spec))[:, None]
    hdr = fits.Header({'MJD': 60050, 'TESCIFO': 35.4, 'TESCIKM': 10.0})
    if header:
        hdr.update(header)
    return RSS(data=flux, error=0.01 * flux, wave=WAVE, slitmap=slitmap, header=hdr)


def test_channel_weights():
    w = fg.channel_weights(WAVE)
    assert np.allclose(w['b'] + w['r'] + w['z'], 1.0)
    assert w['b'][WAVE < 5000].min() == 1 and w['r'][(WAVE > 6000) & (WAVE < 7000)].min() == 1
    assert w['z'][WAVE > 8000].min() == 1


def test_fiber_throughput():
    rng = np.random.default_rng(8)
    x, y = _positions(500, rng)
    spec = rng.integers(1, 4, size=500)
    k = fg.fiber_throughput(FACTORS, GX, GY, x, y, spec)
    assert np.mean(k) == pytest.approx(1.0)
    assert np.allclose(k / _throughput(x, y, spec), np.mean(_throughput(x, y, spec)) ** -1)


def test_apply_flat_models_removes_the_plane(frame):
    rss = _cframe(frame)
    sky = rss._data[np.asarray(rss._slitmap['telescope']) != 'Sci'].copy()
    models = {ch: _constant_model(ch, GX, GY) for ch in 'brz'}
    res = fg.apply_flat_models(rss, models, model_files={'z': '/x/lvm-mfiberflat_gradient-z.fits'})
    assert res['model']['z']['gx'] == pytest.approx(GX)
    after = fg.measure_skyline_planes(rss._data, WAVE, rss._slitmap, sky_fibers=False)
    for key in ('b', 'r', 'z', 'z_oh'):
        assert abs(after[key]['gx']) < 0.1 and abs(after[key]['gy']) < 0.1, key
        assert after[key]['f1'] == pytest.approx(after[key]['f3'], abs=0.002)
    # the sky fibers are not changed; the record is in the header
    assert np.array_equal(rss._data[np.asarray(rss._slitmap['telescope']) != 'Sci'], sky)
    assert rss._header['HIERARCH FLATGRAD APPLIED'] is True
    assert rss._header['HIERARCH FLATGRAD Z GX'] == pytest.approx(GX)
    assert rss._header['HIERARCH FLATGRAD Z FILE'] == 'lvm-mfiberflat_gradient-z.fits'
    assert rss._header['HIERARCH FLATGRAD DIRECT'] is False


def test_apply_flat_models_scales_the_error(frame):
    rss = _cframe(frame)
    ratio = rss._error / rss._data
    fg.apply_flat_models(rss, {ch: _constant_model(ch, GX, GY) for ch in 'brz'})
    assert np.allclose(rss._error / rss._data, ratio)


def test_apply_flat_models_only_once(frame):
    rss = _cframe(frame)
    models = {ch: _constant_model(ch, GX, GY) for ch in 'brz'}
    fg.apply_flat_models(rss, models)
    once = rss._data.copy()
    assert fg.apply_flat_models(rss, models) is None
    assert np.array_equal(rss._data, once)


def test_apply_flat_models_direct_step(frame):
    rss = _cframe(frame)
    models = {ch: _constant_model(ch, 0.5 * GX, 0.5 * GY) for ch in 'brz'}
    res = fg.apply_flat_models(rss, models, direct=True)
    assert res['direct'] and res['rmode'] == 'oh+bz'
    after = fg.measure_skyline_planes(rss._data, WAVE, rss._slitmap, sky_fibers=False)
    for key in ('b', 'r', 'z', 'z_oh'):
        assert abs(after[key]['gx']) < 0.1 and abs(after[key]['gy']) < 0.1, key
    assert rss._header['HIERARCH FLATGRAD DIRECT'] is True
    assert rss._header['HIERARCH FLATGRAD DRMODE'] == 'oh+bz'


def test_apply_flat_models_direct_guard(frame):
    # b and z disagree by 3 %/10 mm: the direct step is not applied, the model is
    rss = _cframe(frame, gx_z=GX + 3.0)
    res = fg.apply_flat_models(rss, {ch: _constant_model(ch, GX, GY) for ch in 'brz'}, direct=True)
    assert not res['direct'] and res['guard'] == pytest.approx(3.0, abs=0.2)
    assert rss._header['HIERARCH FLATGRAD APPLIED'] is True
    assert rss._header['HIERARCH FLATGRAD DIRECT'] is False


def test_apply_flat_models_not_possible(frame):
    models = {ch: _constant_model(ch, GX, GY) for ch in 'brz'}
    rss = _cframe(frame, header={'TESCIFO': -999.0})
    data = rss._data.copy()
    assert fg.apply_flat_models(rss, models) is None
    assert np.array_equal(rss._data, data)
    assert fg.apply_flat_models(_cframe(frame), {'b': models['b']}) is None


def test_correct_cframe(frame, tmp_path, monkeypatch):
    rss = _cframe(frame)
    monkeypatch.setattr(fg, 'find_flat_models', lambda *a, **k: {})
    assert fg.correct_cframe(rss, mjd=60050) is None
    assert 'HIERARCH FLATGRAD APPLIED' not in rss._header

    table = _drpall()
    model = fg.fit_flat_model(table, EPOCHS)
    files = {}
    for ch in 'brz':
        files[ch] = str(tmp_path / f'lvm-mfiberflat_gradient-{ch}.fits')
        fg.write_flat_model(model, EPOCHS[0], ch, files[ch])
    monkeypatch.setattr(fg, 'find_flat_models', lambda *a, **k: files)
    res = fg.correct_cframe(rss, mjd=60050)
    assert res is not None and rss._header['HIERARCH FLATGRAD APPLIED'] is True
    assert rss._header['HIERARCH FLATGRAD EPOCH'] == EPOCHS[0]


def test_flatgrad_test_copy_names():
    from lvmdrp.main import _flatgrad_name
    assert _flatgrad_name('/a/lvmCFrame-00044604.fits') == '/a/lvmCFrame-00044604.flatgrad.fits'
    assert _flatgrad_name('/a/lvmSFrame-00044604.fits.gz') == '/a/lvmSFrame-00044604.flatgrad.fits.gz'


# ------------------------------------------------------------------ flats from the twilights
TWI = {60000: (3.0, 5.0), 60100: (4.0, 7.0)}
SUN = (1.5, -0.15)


def _twilight_summary(n=200, seed=11, noise=0.2):
    """A synthetic twilight table: planes from the known per-epoch gradient plus the Sun term."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    epoch = rng.choice(EPOCHS, n)
    mjd = epoch + rng.uniform(0, 90, n)
    sun_alt = rng.uniform(-12, -2, n)
    dir_sun = rng.uniform(-180, 180, n)
    depth = fg.TWILIGHT_SUN_ALT_REF - sun_alt
    ux, uy = np.cos(np.radians(dir_sun)), np.sin(np.radians(dir_sun))
    amp = SUN[0] + SUN[1] * depth
    gx = np.array([TWI[e][0] for e in epoch]) + amp * ux + noise * rng.standard_normal(n)
    gy = np.array([TWI[e][1] for e in epoch]) + amp * uy + noise * rng.standard_normal(n)
    # stage 1 records unnormalized coefficients, possibly large and negative: only ratios matter
    g1 = rng.choice([0.5, 13.8, -7.5e4], n)
    rows = []
    for ch in 'rz':
        rows.append(pd.DataFrame({'expnum': np.arange(n), 'channel': ch, 'mjd': mjd, 'sun_alt': sun_alt,
                                  'dir_sun': dir_sun, 'gcoeff1': g1, 'gcoeff2': g1 * gy / 1000,
                                  'gcoeff3': g1 * gx / 1000, 'gcoeff4': 0.0}))
    return pd.concat(rows, ignore_index=True)


def test_twilight_planes():
    summary = _twilight_summary()
    tw = fg.twilight_planes(summary, 'z')
    assert tw['mjd'].size == 200
    assert np.all(tw['depth'] == fg.TWILIGHT_SUN_ALT_REF - summary[summary.channel == 'z']['sun_alt'].to_numpy())
    assert fg.twilight_planes(summary, 'b')['mjd'].size == 0


def test_fit_twilight_model():
    fit = fg.fit_twilight_model(_twilight_summary(), EPOCHS, channels='rz')
    for ch in 'rz':
        for e, (gx, gy) in TWI.items():
            assert fit['planes'][ch][e] == pytest.approx((gx, gy), abs=0.1)
        assert fit['sun'][ch] == pytest.approx(SUN, abs=0.1)
        assert sum(fit['nexp'][ch].values()) == 200 and fit['rejected'][ch] == []
    with pytest.raises(ValueError):
        fg.fit_twilight_model(_twilight_summary(), EPOCHS, channels='b')


def _write_flat(path, frame):
    flux, slitmap = frame
    flat = np.ones_like(flux)
    RSS(data=flat, error=0.01 * flat, wave=WAVE, slitmap=slitmap, header=fits.Header({'CCD': 'z1'})).writeFitsData(str(path))


def test_apply_undo_twilight_gradient(frame, tmp_path):
    path = tmp_path / 'lvm-mfiberflat_twilight-z.fits'
    _write_flat(path, frame)
    out = tmp_path / 'new.fits'
    _, changed = fg.apply_twilight_gradient(str(path), GX, GY, 60100, out_mflat=str(out), write_output=True)
    assert changed
    new, old = RSS.from_file(str(out)), RSS.from_file(str(path))
    sci = np.asarray(new._slitmap['telescope']).astype(str) == 'Sci'
    x, y = np.asarray(new._slitmap['xpmm'])[sci], np.asarray(new._slitmap['ypmm'])[sci]
    plane = 1 + (GX * x + GY * y) / 1000
    assert np.allclose(new._data[sci, 0] / old._data[sci, 0], plane / plane.mean())
    assert np.array_equal(new._data[~sci], old._data[~sci])
    assert new._header['Z FIBERFLAT TWIGRAD'] is True and new._header['Z FIBERFLAT TWIEPOCH'] == 60100
    # never twice, and undone exactly
    assert not fg.apply_twilight_gradient(str(out), GX, GY, 60100)[1]
    _, undone = fg.undo_twilight_gradient(str(out), write_output=True)
    assert undone and np.allclose(RSS.from_file(str(out))._data, old._data)


def test_new_flat_replaces_a_link_not_its_target(frame, tmp_path):
    src = tmp_path / 'calib' / '60100'
    src.mkdir(parents=True)
    _write_flat(src / 'lvm-mfiberflat_twilight-z.fits', frame)
    (src / 'lvm-mbias-z1.fits').write_text('bias')
    before = (src / 'lvm-mfiberflat_twilight-z.fits').read_bytes()
    assert fg.link_calibration_tree(str(tmp_path / 'calib'), str(tmp_path / 'new')) == 2
    dst = tmp_path / 'new' / '60100' / 'lvm-mfiberflat_twilight-z.fits'
    assert dst.is_symlink()
    fg.apply_twilight_gradient(str(src / 'lvm-mfiberflat_twilight-z.fits'), GX, GY, 60100, out_mflat=str(dst),
                               write_output=True)
    assert not dst.is_symlink() and (tmp_path / 'new' / '60100' / 'lvm-mbias-z1.fits').is_symlink()
    assert (src / 'lvm-mfiberflat_twilight-z.fits').read_bytes() == before
    with pytest.raises(ValueError):
        fg.link_calibration_tree(str(tmp_path / 'calib'), str(tmp_path / 'calib' / 'inside'))


def test_create_twilight_gradients(frame, tmp_path, monkeypatch):
    import lvmdrp.core.constants
    import lvmdrp.utils.paths
    calib = tmp_path / 'calib'
    for cal in (60000, 60050, 60100):
        (calib / str(cal)).mkdir(parents=True)
        _write_flat(calib / str(cal) / 'lvm-mfiberflat_twilight-z.fits', frame)
    monkeypatch.setenv('LVM_MASTER_DIR', str(calib))
    monkeypatch.setattr(lvmdrp.utils.paths, 'get_calib_paths', lambda mjd, **k: {
        'fiberflat_twilight': {'z': str(calib / str(mjd) / 'lvm-mfiberflat_twilight-z.fits')}})
    summary = tmp_path / 'twilights.csv'
    _twilight_summary().to_csv(summary, index=False)
    epochs = {e: {} for e in EPOCHS}
    cals = {60000: {}, 60050: {}, 60100: {}}
    with pytest.raises(ValueError):
        fg.create_twilight_gradients(str(summary), epochs, cals, channels='z')
    new = tmp_path / 'new'
    fit = fg.create_twilight_gradients(str(summary), epochs, cals, channels='z', out_calib_dir=str(new))
    for cal, e in ((60000, 60000), (60050, 60000), (60100, 60100)):
        h = fits.getheader(str(new / str(cal) / 'lvm-mfiberflat_twilight-z.fits'))
        assert h['Z FIBERFLAT TWIEPOCH'] == e
        assert h['Z FIBERFLAT TWIGX'] == pytest.approx(fit['planes']['z'][e][0], abs=1e-3)
        assert 'Z FIBERFLAT TWIGRAD' not in fits.getheader(str(calib / str(cal) / 'lvm-mfiberflat_twilight-z.fits'))
