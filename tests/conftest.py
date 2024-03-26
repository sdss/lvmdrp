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
    """ fixture factory to create empty data directories for a list of mjds """

    path = pathlib.Path(os.getenv("LVM_DATA_S"))
    path.mkdir(parents=True, exist_ok=True)
    outs = []

    def _datadir(mjds):
        for mjd in mjds:
            (path / f'{mjd}').mkdir(exist_ok=True)
            outs.append(path / f'{mjd}')
        return outs
    yield _datadir


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


def create_fake_fits(path, tileid=11111, mjd=61234, expnum=6817, cameras=None, bias_levels=[980, 960, 1000, 1020], sci_level=60000, leak=False, shift_rows=[]):
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
    def _make_fits(mjd=61234, cameras=["b1"], expnum=6817, bias_levels=[980, 960, 1000, 1020], sci_level=60000, leak=False, shift_rows=[]):
        paths = datadir([mjd])
        return create_fake_fits(paths[0], mjd=mjd, cameras=cameras, expnum=expnum, leak=leak, shift_rows=shift_rows)
    yield _make_fits
