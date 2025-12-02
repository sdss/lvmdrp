# encoding: utf-8
#
# conftest.py
#

import itertools
import os
import pathlib
import pytest
import lvmdrp
import importlib

import numpy as np
import pandas as pd
from astropy.io import fits

from lvmdrp.core.image import _parse_ccd_section
from lvmdrp.core.constants import LVM_NPRE
from lvmdrp.functions.imageMethod import DEFAULT_BIASSEC, DEFAULT_TRIMSEC


"""
Here you can add fixtures that will be used for all the tests in this
directory. You can also add conftest.py files in underlying subdirectories.
Those conftest.py will only be applies to the tests in that subdirectory and
underlying directories. See
https://docs.pytest.org/en/latest/how-to/writing_plugins.html#conftest-py-plugins for
more information.
"""


@pytest.fixture(autouse=True, scope='session')
def mock_sas(tmp_path_factory, session_mocker):
    """ Mock the main SAS_BASE_DIR """

    path = str(tmp_path_factory.mktemp("sas"))
    os_copy = os.environ.copy()
    os.environ['SAS_BASE_DIR'] = path
    os.environ["LVMDRP_VERSION"] = 'test'
    session_mocker.patch('lvmdrp.__version__', autospec=True)
    importlib.reload(lvmdrp)
    yield
    os.environ = os_copy


@pytest.fixture(scope='module')
def datadir():
    """ fixture factory to create an empty data directory for an mjd """

    path = pathlib.Path(os.getenv("LVM_DATA_S"))
    path.mkdir(parents=True, exist_ok=True)

    def _datadir(mjd):
        out = (path / f'{mjd}')
        out.mkdir(exist_ok=True)
        return out
    yield _datadir


@pytest.fixture(scope='module')
def datadirs(datadir):
    """ fixture factory to create empty data directories for a list of mjds """
    out = []
    def _make(mjds):
        out.append([datadir(m) for m in mjds])
        return out
    yield _make


@pytest.fixture(scope='module')
def reduxdir():
    """ fixture factory to create empty redux directories for a list of mjds """

    path = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX"))
    path.mkdir(parents=True, exist_ok=True)

    def _reduxdir(tileid, mjd):
        tg = '{:0>4d}XX'.format(int(tileid) // 1000)
        out = (path / os.getenv("LVMDRP_VERSION") / tg / f'{tileid}' / f'{mjd}')
        out.mkdir(parents=True, exist_ok=True)
        return out
    yield _reduxdir


@pytest.fixture(scope='module')
def make_meta():
    """ fixture factory to make a fake metadata frame """
    def _make_meta(expnum=6817, tileid=11111, mjd=61234):
        defaults = {'hemi': 's', 'status': 0, 'stage': 1, 'qual': 0, 'quality': 'excellent',
                    'quartz': False, 'ldls': False, 'argon': False, 'xenon': False, 'krypton': False,
                    'hgne': False, 'neon': False, 'exptime': 900.0, 'imgtype': 'object'}
        camera = ((''.join(i) for i in itertools.product(('b', 'r', 'z'), ('1', '2', '3'))))
        spec = (('sp' + i for i in map(str, (1, 2, 3) * 3)))
        defaults['camera'] = camera
        defaults['spec'] = spec

        defaults['expnum'] = expnum
        defaults['tileid'] = tileid
        defaults['tilegrp'] = '0011XX'
        defaults['mjd'] = mjd
        defaults['rmjd'] = mjd
        return pd.DataFrame(defaults)
    yield _make_meta


@pytest.fixture(scope='module')
def meta(make_meta):
    """ fixture to create a default metadata frame """
    yield make_meta()


@pytest.fixture(scope='module')
def make_multi(make_meta):
    """ fixture factory to create a multi metadata frame"""
    def _make_multi(expnum=[]):
        df = make_meta()
        for i in expnum:
            df = pd.concat([df, make_meta(expnum=i)])
        return df
    yield _make_multi


@pytest.fixture(scope='module')
def multimeta(make_multi):
    """ fixture to create a metadata frame with multiple exposures """
    yield make_multi(expnum=[6818, 6819])


def create_fake_raw_fits(path, tileid=11111, mjd=61234, expnum=6817, cameras=None,
                         pre_levels=[500, 200], bias_levels=[980, 960, 1000, 1020], sci_level=60000,
                         leak=False, shift_rows=[]):
    """ create a fake raw frame FITS file """
    out = {}
    cameras = cameras or ['b1']
    for cam in cameras:
        # create fake header
        filename = f'sdR-s-{cam}-{expnum:0>8}.fits.gz'
        hdr = {'TILE_ID': tileid, 'MJD': mjd, 'EXPOSURE': expnum, "CCD": cam,
               'EXPTIME': 900.0,
               'IMAGETYP': 'object', 'FILENAME': filename, 'SPEC': f'sp{cam[-1]}',
               'OBSTIME': '2023-10-18T07:56:23.289', 'OBSERVAT': 'LCO',
               'TELESCOP': 'SDSS 0.16m', 'SURVEY': 'LVM',
               "TRIMSEC1": DEFAULT_TRIMSEC[0], "TRIMSEC2": DEFAULT_TRIMSEC[1],
               "TRIMSEC3": DEFAULT_TRIMSEC[2], "TRIMSEC4": DEFAULT_TRIMSEC[3],
               "BIASSEC1": DEFAULT_BIASSEC[0], "BIASSEC2": DEFAULT_BIASSEC[1],
               "BIASSEC3": DEFAULT_BIASSEC[2], "BIASSEC4": DEFAULT_BIASSEC[3]}

        # mock data
        data = np.zeros((4080, 4120), dtype=int)
        for iquad in range(4):
            (os_ix,os_fx), (os_iy,os_fy) = _parse_ccd_section(DEFAULT_BIASSEC[iquad])
            (sc_ix,sc_fx), (sc_iy,sc_fy) = _parse_ccd_section(DEFAULT_TRIMSEC[iquad])
            data[os_iy:os_fy, os_ix:os_fx] += bias_levels[iquad]
            data[sc_iy:sc_fy, sc_ix:sc_fx] += bias_levels[iquad] + sci_level

            # simulate leaks in overscan region
            if leak:
                data[:, (os_ix if iquad in {0,2} else os_fx-1)] += sci_level

        # simulate pre-scan regions
        data[:, :LVM_NPRE] = pre_levels[0]
        data[:, -LVM_NPRE:] = pre_levels[1]

        # simulate pixel shifts
        for irow in shift_rows:
            data[irow:] = np.roll(data[irow:], -2, axis=1)

        # create fake file
        prim = fits.PrimaryHDU(header=fits.Header(hdr), data=data)
        full = path / filename
        hdulist = fits.HDUList(prim)
        hdulist.writeto(full)
        out[cam] = full
    return out


@pytest.fixture(scope='module')
def make_fits(datadir):
    """ fixture to create fake fits files """
    def _make_fits(tileid=None, mjd=None, cameras=["b1"], expnum=None, bias_levels=[980, 960, 1000, 1020],
                   sci_level=60000, leak=False, shift_rows=[]):
        path = datadir(mjd)
        return create_fake_raw_fits(path, tileid=tileid, mjd=mjd, cameras=cameras, expnum=expnum,
                                    bias_levels=bias_levels, sci_level=sci_level, leak=leak,
                                    shift_rows=shift_rows)
    yield _make_fits


def create_fake_frame_fits(path, kind='S', tileid=11111, mjd=61234, expnum=6817, camera='b'):
    """ create a fake frame FITS file """

    if kind in ['S', 'C']:
        filename = f'lvm{kind.upper()}Frame-{expnum:0>8}.fits'
    else:
        kind = '' if not kind else kind
        filename = f'lvm{kind.upper()}Frame-{camera}-{expnum:0>8}.fits'

    hdr = {'TILE_ID': tileid, 'MJD': mjd, 'EXPOSURE': expnum, 'FILENAME': filename,
           'OBSTIME': '2023-12-19T00:47:39.095', 'OBSERVAT': 'LCO', 'DRPVER': '0.1.1',
           'TELESCOP': 'SDSS 0.16m', 'SURVEY': 'LVM', 'DPOS': 0, 'DRPQUAL': 0, 'OBJECT': f'tile_id={tileid}',
           'SCIRA': 65.949555, 'SCIDEC': 15.348684, 'SCIPA': 66.7, 'SCIALT': 32.5605, 'SCIAM': 1.857,
           'TESCIKM': -87.5, 'TESCIFO': 36.58,
           'SKY SCI_ALT': 32.5605, 'SKY SCI_SH_HGHT': 165.52885, 'SKY SCI_MOON_SEP': 29.8097,
           'SKYERA': 21.008216, 'SKYEDEC': -22.933382, 'SKYEPA': 0.0, 'SKYEALT': 80.9755, 'SKYEAM': 1.013,
           'TESKYEKM': -37.5, 'TESKYEFO': 36.19, 'SKYENAME': 'WHAM_south_02',
           'SKY SKYE_ALT': 80.9755, 'SKY SCI_SKYE_SEP': 58.2987, 'SKY SKYE_SH_HGHT': 164.79533,
           'SKY SKYE_MOON_SEP': 18.4705,
           'SKYWRA': 58.011871, 'SKYWDEC': 11.817184, 'SKYWPA': 0.0, 'SKYWALT': 40.0142, 'SKYWAM': 1.555,
           'TESKYWKM': -54.51, 'TESKYWFO': 37.11, 'SKYWNAME': 'grid087', 'SKY SKYW_ALT': 40.0142,
           'SKY SCI_SKYW_SEP': 8.4839,'SKY SKYW_SH_HGHT': 165.41738, 'SKY SKYW_MOON_SEP': 27.7977,
           'SKY MOON_RA': 348.42157, 'SKY MOON_DEC': -7.55955,
           'SKY MOON_PHASE': 79.91, 'SKY MOON_FLI': 0.4136,
           'SKY MOON_ALT': 46.8744, 'SKY SUN_ALT': -13.3779,
           'STDSENMB': 2.0e-14, 'STDSENMR': 2.0e-14, 'STDSENMZ': 2.0e-14,
           'SCISENMB': 2.0e-14, 'SCISENMR': 2.0e-14, 'SCISENMZ': 2.0e-14,
           'MODSENMB': 2.0e-14, 'MODSENMR': 2.0e-14, 'MODSENMZ': 2.0e-14,
           'FLUXCAL': "MOD"}

    # create fake file
    prim = fits.PrimaryHDU(header=fits.Header(hdr))
    full = path / filename
    hdulist = fits.HDUList(prim)
    hdulist.writeto(full, overwrite=True)
    return full


@pytest.fixture()
def make_framefits(reduxdir):
    """ fixture to create fake fits files """
    def _make_fits(kind='S', tileid=11111, mjd=61234, expnum=6817, camera='b'):
        paths = reduxdir(tileid, mjd)
        return create_fake_frame_fits(paths, kind=kind, tileid=tileid, mjd=mjd, expnum=expnum, camera=camera)
    yield _make_fits