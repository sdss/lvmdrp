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
