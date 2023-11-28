# encoding: utf-8
#

import os
import pathlib
import pytest

from astropy.time import Time
from lvmdrp import __version__ as drpver
from lvmdrp.functions.run_drp import (create_status_file, remove_status_file,
                                      status_file_exists, update_error_file,
                                      should_run, check_daily_mjd, parse_mjds,
                                      filter_expnum)


@pytest.fixture()
def status_file():
    """ fixture to create a status file """
    create_status_file(1111, 61234, status='started')
    yield (pathlib.Path(os.getenv("LVM_SPECTRO_REDUX"))
           / f'{drpver}/1111/logs/lvm-drp-1111-61234.started')


@pytest.fixture()
def transfer():
    """ fixture to create a transfer file """
    done = pathlib.Path(os.getenv("LCO_STAGING_DATA")) / 'log/lvm/61234/transfer-61234.done'
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()
    yield done
    done.unlink()


def test_create_status_file(status_file):
    """ test we can create the status file """
    assert status_file.exists()


def test_remove_status_file(status_file):
    """ test we can remove the status file """
    remove_status_file(1111, 61234)
    assert not status_file.exists()


def test_remove_status_all(status_file):
    """ test we can remove all the status files """
    create_status_file(1111, 61235, status='started')
    remove_status_file(1111, 61234, remove_all=True)

    path2 = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}/1111/logs/lvm-drp-1111-61235.started'

    assert not status_file.exists()
    assert not path2.exists()


def test_status_exists(status_file):
    """ test we can check the status file """
    assert status_file_exists(1111, 61234, status='started')


def test_update_error_file():
    """ test we can update the error file """
    path = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}/drp_errors.txt'

    errors = """ Traceback (most recent call last):
  File "/Users/Brian/Work/github_projects/sdss/lvm/lvmdrp/python/lvmdrp/functions/run_drp.py", line 1517, in run_drp
    quick_science_reduction(expnum, use_fiducial_master=True)
  File "/Users/Brian/Work/github_projects/sdss/lvm/lvmdrp/python/lvmdrp/functions/run_quickdrp.py", line 225, in quick_science_reduction
    raise ValueError('This is a bad error on b2')
ValueError: This is a bad error on b2
    """

    update_error_file(1111, 61234, 1011, errors)
    assert path.exists()

    with open(path) as f:
        data = f.read()
        assert 'ERROR on tileid, mjd, exposure: 1111, 61234, 1011' in data
        assert 'This is a bad error on b2' in data


def test_should_run_no():
    """ test we should not run the drp """
    assert not should_run(61234)


def test_should_run_yes(transfer):
    """ test that we should now run the drp """
    assert should_run(61234)


@pytest.fixture()
def check(mocker, caplog):
    """ fixture to capture output from the check daily """
    mocker.patch('lvmdrp.functions.run_drp.Time.now', return_value=Time(61234, format='mjd'))
    check_daily_mjd(test=True)
    out = '\n'.join(caplog.messages)
    assert 'The MJD is 61234.' in out
    yield out


def test_check_daily_no(check):
    """ test we are not running drp """
    assert 'Data transfer not yet complete for MJD' in check


def test_check_daily_yes(transfer, check):
    """ test we are running the drp """
    assert 'Running DRP for mjd 61234' in check


@pytest.mark.parametrize('mjd, exp',
                         [(61234, 61234),
                          ('61234', 61234),
                          ([61230, 61234, 61237], [61230, 61234, 61237]),
                          ('61230-61240', [61231, 61234, 61238]),
                          ('-61234', [61231, 61234]),
                          ('61234-', [61234, 61238])],
                         ids=['int', 'str', 'list', 'range', 'upper', 'lower'])
def test_parse_mjds(datadir, mjd, exp):
    """ test we can parse different mjd inputs """
    datadir([61231, 61234, 61238])
    out = parse_mjds(mjd)
    assert out == exp


@pytest.mark.parametrize('expnum, exp',
                         [(6817, [6817]),
                          ('6817', [6817]),
                          ([6817, 6819], [6817, 6819]),
                          ('6817-6818', [6817, 6818]),
                          ('-6818', [6817, 6818]),
                          ('6818-', [6818, 6819]),
                          ],
                         ids=['int', 'str', 'list', 'range', 'upper', 'lower'])
def test_filter_expnum(multimeta, expnum, exp):
    """ test we can filter out exposures from a metaframe """
    out = filter_expnum(multimeta, expnum)
    uni = out['expnum'].unique()
    assert all(uni == exp)


