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

import pandas as pd
from astropy.io import fits

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
    def _make_meta(expnum=6817, tileid=1111, mjd=61234):
        defaults = {'hemi': 's', 'status': 0, 'stage': 1, 'qual': 0, 'quality': 'excellent',
                    'quartz': False, 'ldls': False, 'argon': False, 'xenon': False, 'krypton': False,
                    'hgne': False, 'neon': False, 'exptime': 900.0, 'imgtype': 'object'}
        camera = ((''.join(i) for i in itertools.product(('b', 'r', 'z'), ('1', '2', '3'))))
        spec = (('sp' + i for i in map(str, (1, 2, 3) * 3)))
        defaults['camera'] = camera
        defaults['spec'] = spec

        defaults['expnum'] = expnum
        defaults['tileid'] = tileid
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


def create_fake_fits(path, tileid=1111, mjd=61234, expnum=6817, cameras=None):
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
               'TELESCOP': 'SDSS 0.16m', 'SURVEY': 'LVM'}
        prim = fits.PrimaryHDU(header=fits.Header(hdr))

        # create fake file
        full = path / filename
        hdulist = fits.HDUList(prim)
        hdulist.writeto(full)
        out[cam] = full
    return out


@pytest.fixture(scope='module')
def make_fits(datadir):
    """ fixture to create fake fits files """
    mjd = 61234
    paths = datadir([mjd])
    create_fake_fits(paths[0], mjd=mjd, cameras=['b1', 'b2', 'b3'])