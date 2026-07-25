import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from lvmdrp.core.rss import RSS

C_KMS = 299792.458


def _make_rss(wave_native, flux, header=None):
    if header is None:
        header = fits.Header()
        header["BUNIT"] = "erg / (s cm2 Angstrom)"
    error = np.sqrt(np.abs(flux)) + 1e-6
    mask = np.zeros_like(flux, dtype=bool)
    slitmap = Table({"targettype": ["science"] * wave_native.shape[0],
                      "telescope": ["Sci"] * wave_native.shape[0]})
    return RSS(wave=wave_native, data=flux, error=error, mask=mask,
               header=header, slitmap=slitmap)


def _gaussian(x, center, sigma=0.3):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.fixture
def true_center():
    return 5000.0


@pytest.fixture
def rv_corr_value():
    return 20.0  # km/s


@pytest.fixture
def native_wave():
    # two fibers, native (observed/topocentric) per-pixel wavelength labels,
    # finely sampled around the line
    base = np.arange(4990.0, 5010.0, 0.01)
    return np.stack([base, base + 0.05])  # slightly different per fiber


@pytest.fixture
def observed_flux(native_wave, true_center, rv_corr_value):
    # physical relation: flux recorded at native (observed) wavelength label
    # lam_obs equals the true/barycentric-frame flux at
    # lam_bary = lam_obs * (1 + rv_corr/c)
    return _gaussian(native_wave * (1.0 + rv_corr_value / C_KMS), true_center)


def test_rectify_wave_no_correction_is_unchanged(native_wave, observed_flux):
    """rv_corr=0.0 (default) must reproduce today's unmodified behavior."""
    rss = _make_rss(native_wave, observed_flux)
    out_grid = np.arange(4995.0, 5005.0, 0.01)

    default_call = rss.rectify_wave(wave=out_grid.copy(), method="linear")
    explicit_zero = rss.rectify_wave(wave=out_grid.copy(), method="linear", rv_corr=0.0)

    np.testing.assert_allclose(default_call._data, explicit_zero._data, equal_nan=True)
    np.testing.assert_allclose(default_call._wave, out_grid)
    np.testing.assert_allclose(explicit_zero._wave, out_grid)


def test_rectify_wave_rv_corr_recovers_shifted_line(native_wave, observed_flux, true_center, rv_corr_value):
    """Applying rv_corr should recover the line at its true wavelength on the
    unchanged output grid, undoing the injected shift."""
    rss = _make_rss(native_wave, observed_flux)
    out_grid = np.arange(4995.0, 5005.0, 0.01)

    uncorrected = rss.rectify_wave(wave=out_grid.copy(), method="linear", rv_corr=0.0)
    corrected = rss.rectify_wave(wave=out_grid.copy(), method="linear", rv_corr=rv_corr_value)

    # output wavelength grid must be identical regardless of rv_corr
    np.testing.assert_allclose(corrected._wave, out_grid)
    np.testing.assert_allclose(corrected._wave, uncorrected._wave)

    peak_uncorrected = out_grid[np.nanargmax(uncorrected._data[0])]
    peak_corrected = out_grid[np.nanargmax(corrected._data[0])]

    expected_uncorrected_peak = true_center / (1.0 + rv_corr_value / C_KMS)

    assert peak_uncorrected == pytest.approx(expected_uncorrected_peak, abs=0.02)
    assert peak_corrected == pytest.approx(true_center, abs=0.02)
    # sanity: correction actually moved the peak
    assert abs(peak_corrected - peak_uncorrected) > 0.05

    # provenance header written only when a correction was applied
    assert "HIERARCH WAVE RVCORR_APPLIED" in corrected._header
    assert "HIERARCH WAVE RVCORR_APPLIED" not in uncorrected._header


def _bary_header(scira=180.0, scidec=0.0, obstime="2026-03-20T00:00:00", imagetyp="object"):
    header = fits.Header()
    if imagetyp is not None:
        header["IMAGETYP"] = imagetyp
    if scira is not None:
        header["SCIRA"] = scira
    if scidec is not None:
        header["SCIDEC"] = scidec
    if obstime is not None:
        header["OBSTIME"] = obstime
    return header


def test_get_bary_rv_returns_plausible_value():
    wave = np.arange(5000.0, 5010.0, 1.0)[None, :]
    flux = np.ones_like(wave)
    rss = _make_rss(wave, flux, header=_bary_header())

    bary_rv = rss.get_bary_rv()

    assert isinstance(bary_rv, float)
    # Earth's orbital + rotational velocity envelope
    assert abs(bary_rv) < 35.0


@pytest.mark.parametrize("missing", ["scira", "scidec", "obstime", "imagetyp"])
def test_get_bary_rv_missing_header_info_returns_zero(missing):
    wave = np.arange(5000.0, 5010.0, 1.0)[None, :]
    flux = np.ones_like(wave)
    kwargs = {"scira": 180.0, "scidec": 0.0, "obstime": "2026-03-20T00:00:00", "imagetyp": "object"}
    kwargs[missing] = None
    rss = _make_rss(wave, flux, header=_bary_header(**kwargs))

    assert rss.get_bary_rv() == 0.0
