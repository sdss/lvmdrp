
import pytest
import numpy as np
from astropy.table import Table

from lvmdrp.core.rss import RSS
from lvmdrp.core.sky import select_sky_fibers, fit_supersky


@pytest.fixture
def rss():
    # Create an instance of the RSS class for testing
    wave = np.repeat([np.arange(5500, 8500, 0.5)], 4, axis=0)
    wave += np.linspace(0, 4, 4)[:, None]
    data = np.asarray([np.polyval([0.1, 0.01, 0.001], w) for w in wave])
    error = np.sqrt(np.abs(data))
    fibermap = Table({"targettype": ["SKY", "SKY", "science", "standard"], "telescope": ["SkyE", "SkyW", "Sci", "Spec"]})
    rss = RSS(
        wave=wave,
        data=data,
        error=error,
        mask=np.zeros_like(data, dtype=bool),
        slitmap=fibermap
    )
    yield rss


def test_select_sky_fibers(rss):
    sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data = select_sky_fibers(rss, telescope="east")
    assert sky_wave.shape == (1, 6000)
    assert sky_data.shape == (1, 6000)
    assert sky_vars.shape == (1, 6000)
    assert sky_mask.shape == (1, 6000)
    assert sci_wave.shape == (2, 6000)
    assert sci_data.shape == (2, 6000)
    sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data = select_sky_fibers(rss, telescope="west")
    assert sky_wave.shape == (1, 6000)
    assert sky_data.shape == (1, 6000)
    assert sky_vars.shape == (1, 6000)
    assert sky_mask.shape == (1, 6000)
    assert sci_wave.shape == (2, 6000)
    assert sci_data.shape == (2, 6000)
    sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data = select_sky_fibers(rss, telescope="both")
    assert sky_wave.shape == (2, 6000)
    assert sky_data.shape == (2, 6000)
    assert sky_vars.shape == (2, 6000)
    assert sky_mask.shape == (2, 6000)
    assert sci_wave.shape == (2, 6000)
    assert sci_data.shape == (2, 6000)


def test_fit_supersky(rss):
    sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data = select_sky_fibers(rss, telescope="both")
    f_data, f_error, f_mask, swave, ssky, svars, smask = fit_supersky(sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data)
    assert ssky.shape == (2*6000,)
    assert svars.shape == (2*6000,)
    assert smask.shape == (2*6000,)
    assert f_data(swave) == pytest.approx(ssky, rel=1e-5)
