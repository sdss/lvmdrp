# encoding: utf-8
#
# test_fiber_profiles.py
#
# Tests of the cached fiber profiles used by the optimal extraction (Spectrum1D.extract_flux)

import numpy as np
import pytest

from lvmdrp.core.spectrum1d import FiberProfileCache, Spectrum1D


@pytest.fixture(scope='module')
def cache():
    return FiberProfileCache(fiber_radius=1.4, oversampling_factor=100, npixels=15)


def _reference_profiles(cache, centroids, sigmas):
    """ the profiles as FiberProfileCache.__call__ computed them before it read the cache directly """
    lines = np.clip((np.round(cache.nprofiles / (cache.sigma_max - cache.sigma_min) * (sigmas - cache.sigma_min))).astype(int),
                    a_min=0, a_max=cache.nprofiles - 1)
    cumsums = cache.profile_cumsum[:, lines]
    cen_fracs = centroids - np.trunc(centroids) + 0.5
    centers = cache.x_os[:, lines] + cen_fracs
    bins = cache.x[:, lines]
    bin_starts = np.clip(((bins - centers[0, :]) * cache.oversampling_factor).astype(int), a_min=0, a_max=None)
    bin_ends = np.clip(bin_starts + cache.oversampling_factor, a_min=None, a_max=len(cumsums[:, 0]) - 1)
    cols = np.arange(bin_starts.shape[1])[None, :]
    return cumsums[bin_ends, cols] - cumsums[bin_starts, cols]


def test_profiles_identical_to_reference(cache):
    rng = np.random.default_rng(4)
    centroids = rng.uniform(20, 4060, 648)
    # include centroids on pixel boundaries and widths outside the cached range (clipped)
    centroids[:4] = [100.0, 100.5, 101.0 - 1e-9, 2000.25]
    sigmas = rng.uniform(0.3, 2.8, 648)
    sigmas[:3] = [0.1, 2.5, 5.0]
    new = cache(centroids, sigmas)
    assert new.shape == (31, 648)
    assert np.array_equal(new, _reference_profiles(cache, centroids, sigmas))


def test_profiles_normalized(cache):
    # each pixel-integrated profile holds (nearly) all of the fiber's light
    prof = cache(np.array([500.3, 800.7]), np.array([1.0, 1.5]))
    assert np.allclose(prof.sum(axis=0), 1.0, atol=0.01)


def test_extract_flux_recovers_fluxes(cache, monkeypatch):
    monkeypatch.setattr(Spectrum1D, 'fiberProfileCache', cache)
    rng = np.random.default_rng(5)
    npix, nfib = 400, 50
    centroids = 30 + 6.5 * np.arange(nfib) + rng.uniform(-0.3, 0.3, nfib)
    sigmas = rng.uniform(1.0, 1.4, nfib)
    flux = rng.uniform(500, 2000, nfib)
    prof = cache(centroids, sigmas)
    rows = np.trunc(centroids).astype(int)[None, :] - 15 + np.arange(31)[:, None]
    data = np.zeros(npix)
    np.add.at(data, rows, prof * flux[None, :])
    spec = Spectrum1D(wave=np.arange(npix, dtype=float), data=data, error=np.ones(npix),
                      mask=np.zeros(npix, dtype=bool))
    out, err, mask = spec.extract_flux(centroids, sigmas)
    # lsmr stops at atol = btol = 1e-3, which leaves errors of up to about 0.2% even without noise
    assert np.allclose(out, flux, rtol=5e-3)
    assert np.all(np.isfinite(err)) and not mask.any()
