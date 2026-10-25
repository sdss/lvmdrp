
import numpy as np

import pytest
from lvmdrp.core.rss import RSS

@pytest.fixture(scope='module')
def rss():
    # Create an instance of the RSS class for testing
    wave = np.arange(5500, 8500, 1)
    rss = RSS(
        wave=wave,
        data=np.random.randint(0, 100, size=(2, len(wave))),
        error=np.random.randn(2, len(wave))
    )
    yield rss

def test_extendData_same(rss):
    # Test case 3: Extend data with same wave values
    new_wave = np.arange(5500, 8500, 1)
    rss.extendData(new_wave)
    assert np.all(rss._wave == new_wave)

def test_extendData_nans(rss):
    # Test case 1: Extend data with new wave values
    new_wave = np.arange(4000, 9000, 1)
    out = rss.extendData(new_wave)
    assert np.isnan(out._data).any()

def test_extendData_empty(rss):
    # Test case 2: Extend data with empty wave values
    new_wave = np.asarray([])
    with pytest.raises(ValueError):
        rss.extendData(new_wave)

def test_extendData_wrong(rss):
    # Test case 4: Extend data with wrong wave values
    new_wave = np.arange(5500, 8500, 0.3)
    with pytest.raises(ValueError):
        rss.extendData(new_wave)


class _FakeTelluric:
    """telluric model with a constant transmission for each airmass"""

    def match_to_data(self, wave, lsf, pwv, airmass=None, lsf_in_wavelength=False):
        return np.full(wave.size, 1.0 / airmass)


AIRMASS = {"SCIAM": 2.0, "SKYEAM": 4.0, "SKYWAM": 5.0}


def _calibrated_rss(method="MOD"):
    from astropy.io import fits
    from astropy.table import Table
    npix = 50
    counts = np.full((3, npix), 100.0)
    sens = {"std": np.linspace(1.0, 2.0, npix), "mod": np.linspace(2.0, 1.0, npix), "sci": np.full(npix, 3.0)}
    tables = {key: Table({"STD1SEN": value, "mean": value, "rms": 0 * value}) for key, value in sens.items()}

    def applied(key):
        # as in apply_fluxcal: the MOD sensitivity is divided by the telluric transmission of each airmass
        return sens[method.lower()] * (AIRMASS[key] if method == "MOD" else 1.0)

    rss = RSS(data=counts * applied("SCIAM"), error=0.1 * counts * applied("SCIAM"), wave=np.arange(npix, dtype=float),
              lsf=np.ones(npix), sky_east=counts * applied("SKYEAM"), sky_west=counts * applied("SKYWAM"),
              fluxcal_std=tables["std"], fluxcal_sci=tables["sci"], fluxcal_mod=tables["mod"],
              header=fits.Header({"FLUXCAL": method, "PWV_MED": 2.0, **AIRMASS}))
    return rss, counts, sens


def test_swap_fluxcal_applies_the_new_sensitivity():
    rss, counts, sens = _calibrated_rss("MOD")
    swapped = rss.swap_fluxcal("STD", inplace=False, telluric_corrector=_FakeTelluric())
    assert np.allclose(swapped._data, counts * sens["std"])
    assert np.allclose(swapped._error, 0.1 * counts * sens["std"])
    assert np.allclose(swapped._sky_east, counts * sens["std"]) and np.allclose(swapped._sky_west, counts * sens["std"])
    assert swapped._header["FLUXCAL"] == "STD"
    # the original is left untouched
    assert np.allclose(rss._data, counts * sens["mod"] * AIRMASS["SCIAM"]) and rss._header["FLUXCAL"] == "MOD"


def test_swap_fluxcal_applies_the_telluric_correction_of_mod():
    rss, counts, sens = _calibrated_rss("SCI")
    rss.swap_fluxcal("MOD", telluric_corrector=_FakeTelluric())
    assert np.allclose(rss._data, counts * sens["mod"] * AIRMASS["SCIAM"])
    assert np.allclose(rss._sky_east, counts * sens["mod"] * AIRMASS["SKYEAM"])
    assert np.allclose(rss._sky_west, counts * sens["mod"] * AIRMASS["SKYWAM"])


def test_swap_fluxcal_inplace_and_none():
    rss, counts, sens = _calibrated_rss("STD")
    rss.swap_fluxcal("NONE")
    assert np.allclose(rss._data, counts) and rss._header["FLUXCAL"] == "NONE"
    assert rss._header["BUNIT"] == "electron / (Angstrom s)"
    rss.swap_fluxcal("SCI")
    assert np.allclose(rss._data, counts * sens["sci"]) and rss._header["BUNIT"] == "erg / (Angstrom s cm2)"


def _slit_rss(trend, plane, nspec=3, nper=120, seed=1):
    """counts of a flat with a trend along each slit and a plane across the science IFU, with every 10th fiber a sky fiber"""
    from astropy.io import fits
    from astropy.table import Table
    rng = np.random.default_rng(seed)
    nfibers = nspec * nper
    specid = np.repeat(np.arange(1, nspec + 1), nper)
    telescope = np.where(np.arange(nfibers) % 10 == 5, np.where(np.arange(nfibers) % 20 == 5, "SkyE", "SkyW"), "Sci")
    # science fibers along each slit sweep across the IFU, so that a slit trend looks like a plane on the IFU
    x = np.where(telescope == "Sci", np.tile(np.linspace(-7, 7, nper), nspec) + rng.normal(0, 1, nfibers), rng.normal(0, 1, nfibers))
    y = np.where(telescope == "Sci", 2.0 * specid - 4 + rng.normal(0, 2, nfibers), rng.normal(0, 1, nfibers))
    position = np.tile(np.linspace(-1, 1, nper), nspec)
    ypix = np.tile(np.linspace(10, 4070, nper), nspec)
    sky = np.select([telescope == "SkyE", telescope == "SkyW"], [0.8, 1.3], 1.0)
    counts = 1000 * sky * trend(position, ypix) * np.exp(np.where(telescope == "Sci", plane[0] * x + plane[1] * y, 0.0))
    wave = np.arange(5000, 5100, 1.0)
    data = counts[:, None] * np.ones(wave.size) * (1 + rng.normal(0, 1e-4, (nfibers, wave.size)))
    slitmap = Table({"fiberid": np.arange(1, nfibers + 1), "spectrographid": specid, "telescope": telescope,
                     "xpmm": x, "ypmm": y, "ypix_b": ypix})
    return RSS(data=data, error=np.ones_like(data), wave=wave, header=fits.Header({"CCD": "b"}), slitmap=slitmap), specid


def test_fit_slit_trends_tells_the_slit_trend_from_the_ifu_plane():
    def trend(position, ypix):
        return np.exp(-0.06 * position**2 + 0.03 * position**3 + 0.01 * (ypix >= 2040))
    plane = (0.004, -0.002)
    rss, specid = _slit_rss(trend, plane)
    fitted, gradient = rss.fit_slit_trends(cwave=5050, dwave=40)

    expected = trend(np.tile(np.linspace(-1, 1, 120), 3), np.tile(np.linspace(10, 4070, 120), 3))
    for spid in (1, 2, 3):
        expected[specid == spid] /= expected[specid == spid].mean()
    assert np.allclose(fitted, expected, rtol=1e-3)
    assert np.allclose(gradient, plane, atol=1e-4)


def test_fit_slit_trends_finds_no_trend_in_a_plane():
    rss, _ = _slit_rss(lambda position, ypix: np.ones_like(position), (0.004, -0.002))
    fitted, gradient = rss.fit_slit_trends(cwave=5050, dwave=40)
    assert np.allclose(fitted, 1, atol=1e-3)
    assert np.allclose(gradient, (0.004, -0.002), atol=1e-4)


def test_fit_slit_trend_model_without_trends_puts_the_slit_trend_in_the_plane():
    from lvmdrp.core.rss import fit_slit_trend_model

    def trend(position, ypix):
        return np.exp(0.08 * position)
    rss, _ = _slit_rss(trend, (0.0, 0.0))
    counts = rss.coadd_flux(cwave=5050, dwave=40)
    with_trends = fit_slit_trend_model(counts, rss._slitmap, upper=np.asarray(rss._slitmap["ypix_b"]) >= 2040)
    plain = fit_slit_trend_model(counts, rss._slitmap, with_trends=False)
    # the science fibers sweep x along each slit, so the trend looks like a plane along x without the sky fibers
    assert np.allclose(with_trends["plane"], 0, atol=1e-4)
    assert plain["plane"][0] > 0.004
    assert np.allclose(plain["trend"], 1) and np.allclose(plain["coeffs"], 0)
    assert np.allclose(with_trends["coeffs"][:, 1], 0.08, atol=1e-3)


def test_fit_slit_trend_model_returns_the_residuals_of_the_fitted_fibers():
    from lvmdrp.core.rss import fit_slit_trend_model
    rss, _ = _slit_rss(lambda position, ypix: np.ones_like(position), (0.004, -0.002))
    counts = rss.coadd_flux(cwave=5050, dwave=40)
    counts[0] = np.nan
    model = fit_slit_trend_model(counts, rss._slitmap)
    assert np.isnan(model["residuals"][0])
    assert np.nanmax(np.abs(model["residuals"])) < 1e-3
