#tests for sky.py
#first test is for the sky model headers

import pytest
from astropy.io import fits
from lvmdrp.core.sky import sky_pars_header


@pytest.fixture(autouse=True)

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
        'HIERARCH SKY SCI_SH_HGHT': (265.7293, "height of Earth's shadow [km]"),
        'HIERARCH SKY SKYE_SH_HGHT': (168.952, "height of Earth's shadow [km]"),
        'HIERARCH SKY SKYW_SH_HGHT': (230.24289, "height of Earth's shadow [km]"),
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