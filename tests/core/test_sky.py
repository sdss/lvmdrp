
import pytest
import numpy as np
from astropy.table import Table
from astropy.io import fits

from lvmdrp.core.rss import RSS
from lvmdrp.core.sky import select_sky_fibers, fit_supersky, sky_pars_header


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

def test_sky_pars_header(make_framefits):

    #create mock data and header for test
    sframe = make_framefits(tileid=1054755, mjd=60297, expnum=9532)
    with fits.open(sframe) as hdulist:
        hdr = hdulist['PRIMARY'].header
    
    #running skymodel_pars_header on example header
    test_sky_pars = sky_pars_header(hdr)

    #expected dictionary output from skymodel_pars_header for mock header
    exp_sky_pars = {'HIERARCH SKY SCI_ALT': (32.5605, 'altitude of object above horizon [deg]'),
        'HIERARCH SKY SKYE_ALT': (80.9755, 'altitude of object above horizon [deg]'),
        'HIERARCH SKY SKYW_ALT': (40.0142, 'altitude of object above horizon [deg]'),
        'HIERARCH SKY SCI_SKYE_SEP': (58.2987, 'separation of SCI and SkyE [deg]'),
        'HIERARCH SKY SCI_SKYW_SEP': (8.4839, 'separation of SCI and SkyW [deg]'),
        'HIERARCH SKY SCI_MOON_SEP': (80.1128, 'separation of Moon and object [deg]'),
        'HIERARCH SKY SKYE_MOON_SEP': (34.8634, 'separation of Moon and object [deg]'),
        'HIERARCH SKY SKYW_MOON_SEP': (71.8498, 'separation of Moon and object [deg]'),
        'HIERARCH SKY MOON_ALT': (46.7507, 'altitude of Moon above horizon [deg]'),
        'HIERARCH SKY SUN_ALT': (-13.4857, 'altitude of Sun above horizon [deg]'),
        'HIERARCH SKY MOON_RA': (348.4256, 'RA of the Moon [deg]'),
        'HIERARCH SKY MOON_DEC': (-7.55656, 'DEC of the Moon [deg]'),
        'HIERARCH SKY MOON_PHASE': (79.92, 'Moon phase (0=N,90=1Q,180=F,270=3Q)[deg]'),
        'HIERARCH SKY MOON_FLI': (0.4137, 'Moon fraction lunar illumination'),
        'HIERARCH SKY SCI_SH_HGHT': (265.72376, "height of Earth's shadow [km]"),
        'HIERARCH SKY SKYE_SH_HGHT': (168.94919, "height of Earth's shadow [km]"),
        'HIERARCH SKY SKYW_SH_HGHT': (230.23836, "height of Earth's shadow [km]"),
        'HIERARCH SKYMODEL SM_H': (2.38, 'observatory height [km]'),
        'HIERARCH SKYMODEL SM_HMIN': (2.0, 'lower height limit [km]'),
        'HIERARCH SKYMODEL MOONDIST': (0.9485, 'ratio of distance over mean dist to Moon'),
        'HIERARCH SKYMODEL PRES': (744.0, 'pressure at observer altitude, set: 744 [hPa]'),
        'HIERARCH SKYMODEL SSA': (0.97, "aerosols' single scattering albedo, set: 0.97"),
        'HIERARCH SKYMODEL CALCDS': ('N', 'cal double scattering of Moon (Y or N)'),
        'HIERARCH SKYMODEL O2COLUMN': (1.0, 'relative ozone column density (1->258) [DU]'),
        'HIERARCH SKYMODEL MOONSCAL': (1.0, 'scaling factor for scattered moonlight'),
        'HIERARCH SKYMODEL SCI_LON_ECL': (-199.77931, 'heliocen ecliptic longitude [deg]'), 
        'HIERARCH SKYMODEL SCI_LAT_ECL': (-6.16801, 'ecliptic latitude [deg]'),
        'HIERARCH SKYMODEL SKYE_LON_ECL': (-256.73754, 'heliocen ecliptic longitude [deg]'),
        'HIERARCH SKYMODEL SKYE_LAT_ECL': (-29.26712, 'ecliptic latitude [deg]'),
        'HIERARCH SKYMODEL SKYW_LON_ECL': (-208.0864, 'heliocen ecliptic longitude [deg]'),
        'HIERARCH SKYMODEL SKYW_LAT_ECL': (-8.18363, 'ecliptic latitude [deg]'),
        'HIERARCH SKYMODEL EMIS_STR': (0.2, 'grey-body emissivity'),
        'HIERARCH SKYMODEL TEMP_STR': (290.0, 'grey-body temperature [K]'),
        'HIERARCH SKYMODEL MSOLFLUX': (130.0, 'monthly-averaged solar radio flux, set: 130'),
        'HIERARCH SKYMODEL SEASON': (1, 'bimonthly period (1:Dec/Jan, 6:Oct/Nov; 0:year)'),
        'HIERARCH SKYMODEL TIME': (1, 'period of night (x/3 night, x=1,2,3; 0:night)')
    }
 
    assert test_sky_pars == exp_sky_pars
