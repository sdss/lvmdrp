# encoding: utf-8
"""Tests of the sky-line measurement of the large-scale flat-field error (flatgradient)."""

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
