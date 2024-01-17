# encoding: utf-8
#

import os
import pathlib
import pytest

from lvmdrp.utils.convert import (tileid_grp, correct_sjd, sjd_to_mjd,
                                  mjd_to_sjd, dateobs_to_sjd, dateobs_to_mjd)


@pytest.mark.parametrize('tileid, exp',
                         [('11111', '0011XX'),
                          (11111, '0011XX'),
                          (1055360, '1055XX'),
                          ('*', '*XX')],
                         ids=['str', 'int', 'main', 'pattern'])
def test_tileid_grp(tileid, exp):
    """ test we can correctly get the tile id group """
    grp = tileid_grp(tileid)
    assert grp == exp


def test_sjd_to_mjd():
    """ test we can covert from SJD to MJD """
    out = sjd_to_mjd(60125)
    assert out == 60124.6


def test_mjd_to_sjd():
    """ test we can covert from MJD to SJD """
    out = mjd_to_sjd(60125)
    assert out == 60125.4


def test_dateobs_to_mjd():
    """ test we can covert from date time to MJD """
    out = int(dateobs_to_mjd('2023-06-19T22:52:00.981'))
    assert out == 60114


def test_dateobs_to_sjd():
    """ test we can covert from date time to SJD """
    out = int(dateobs_to_sjd('2023-06-19T22:52:00.981'))
    assert out == 60115


@pytest.mark.parametrize('stem',
                         ['sdR-s-b1-00006817.fits.gz',
                          'calib/lvm-mbias-r1.fits'],
                         ids=['raw', 'master'])
def test_correct_sjd(stem):
    """ test we can correct the SJD """
    path = pathlib.Path(os.getenv("LVM_DATA_S")) / '60112' / stem
    out = correct_sjd(path, '60113')
    assert out == 60112
