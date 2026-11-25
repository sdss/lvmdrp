# encoding: utf-8
#
# test_calibdir.py
#
# Tests of the choice of calibration directory (sdss/lvmdrp#272)

import os
import subprocess

import pandas as pd
import pytest

from lvmdrp.core import constants
from lvmdrp.core.constants import get_lvmcore_version, get_masters_dir, set_masters_dir
from lvmdrp.main import run_drp
from lvmdrp.utils.metadata import safe_hdf_append
from lvmdrp.utils.paths import get_calib_paths


@pytest.fixture
def calib_tree(tmp_path, monkeypatch):
    """ an empty calibration directory with one epoch, and $LVM_MASTER_DIR restored afterwards """
    monkeypatch.setenv('LVM_MASTER_DIR', os.getenv('LVM_MASTER_DIR', str(tmp_path / 'default')))
    root = tmp_path / 'calib_test'
    (root / '60889').mkdir(parents=True)
    (root / 'pixelmasks').mkdir()
    return root


def test_get_masters_dir_follows_env(monkeypatch):
    monkeypatch.setenv('LVM_MASTER_DIR', '/some/calib')
    assert get_masters_dir() == '/some/calib'
    assert constants.MASTERS_DIR == '/some/calib'
    monkeypatch.setenv('LVM_MASTER_DIR', '/other/calib')
    assert get_masters_dir() == '/other/calib'
    assert constants.MASTERS_DIR == '/other/calib'


def test_set_masters_dir(calib_tree):
    out = set_masters_dir(str(calib_tree) + os.sep)
    assert out == str(calib_tree)
    assert os.environ['LVM_MASTER_DIR'] == str(calib_tree)
    assert get_masters_dir() == str(calib_tree)


def test_set_masters_dir_missing(calib_tree):
    before = get_masters_dir()
    with pytest.raises(FileNotFoundError):
        set_masters_dir(str(calib_tree / 'nothere'))
    assert get_masters_dir() == before


def test_get_calib_paths_follow_calib_dir(calib_tree):
    set_masters_dir(str(calib_tree))
    calibs = get_calib_paths(mjd=60894, cameras='b1', flavors={'bias', 'pixmask', 'fiberflat_twilight'})
    assert calibs['bias']['b1'] == str(calib_tree / '60889' / 'lvm-mbias-b1.fits')
    assert calibs['pixmask']['b1'] == str(calib_tree / 'pixelmasks' / 'lvm-mpixmask-b1.fits')
    assert calibs['fiberflat_twilight']['b'] == str(calib_tree / '60889' / 'lvm-mfiberflat_twilight-b.fits')


def test_run_drp_calib_dir(calib_tree):
    # sets the calibration directory before anything else (and stops: nothing to reduce)
    run_drp(mjd=None, expnum=None, calib_dir=str(calib_tree))
    assert get_masters_dir() == str(calib_tree)


def test_run_drp_calib_dir_missing(calib_tree):
    before = get_masters_dir()
    run_drp(mjd=60894, calib_dir=str(calib_tree / 'nothere'))
    assert get_masters_dir() == before


def test_configured_calib_dir(tmp_path, monkeypatch):
    """ calib_dir in lvmdrp.yaml: none, relative to $SAS_BASE_DIR/sdsswork/lvm, or absolute """
    monkeypatch.setenv('SAS_BASE_DIR', str(tmp_path))
    assert constants.configured_calib_dir({}) is None
    assert constants.configured_calib_dir({'calib_dir': None}) is None
    assert constants.configured_calib_dir({'calib_dir': 'sandbox/calib_x'}) == \
        str(tmp_path / 'sdsswork' / 'lvm' / 'sandbox' / 'calib_x')
    assert constants.configured_calib_dir({'calib_dir': str(tmp_path / 'abs')}) == str(tmp_path / 'abs')


def test_apply_configured_calib_dir(calib_tree, monkeypatch):
    """ the directory of lvmdrp.yaml is used, unless --calib-dir is given; a missing one is an error """
    before = get_masters_dir()
    assert constants.apply_configured_calib_dir({}) is None and get_masters_dir() == before
    cfg = {'calib_dir': str(calib_tree)}
    assert constants.apply_configured_calib_dir(cfg, explicit=True) is None and get_masters_dir() == before
    assert constants.apply_configured_calib_dir(cfg) == str(calib_tree) and get_masters_dir() == str(calib_tree)
    monkeypatch.setenv('LVM_MASTER_DIR', before)
    with pytest.raises(FileNotFoundError, match='lvmdrp.yaml'):
        constants.apply_configured_calib_dir({'calib_dir': str(calib_tree / 'nothere')})
    assert get_masters_dir() == before


def test_calib_dir_in_args():
    assert constants.calib_dir_in_args(['run', '-e', '1', '--calib-dir', '/x'])
    assert constants.calib_dir_in_args(['run', '--calib-dir=/x'])
    assert not constants.calib_dir_in_args(['run', '-e', '1'])


def test_help_requested():
    """ help needs no calibration directory, so the drp group skips its check for it """
    import click

    @click.group()
    def grp():
        pass

    @grp.command()
    def run():
        pass

    @grp.group()
    def calibrations():
        pass

    @calibrations.command()
    def nightly():
        pass

    ctx = click.Context(grp)
    assert constants.help_requested(grp, ctx, ['run', '--help'])
    assert constants.help_requested(grp, ctx, ['calibrations', 'nightly', '--help'])
    assert constants.help_requested(grp, ctx, ['calibrations'])
    assert not constants.help_requested(grp, ctx, ['run'])
    assert not constants.help_requested(grp, ctx, ['calibrations', 'nightly'])


def test_version_label():
    """ $LVMDRP_VERSION, else drpver in lvmdrp.yaml, else the package version """
    from lvmdrp import version_label
    assert version_label('env', 'cfg', '1.3.3dev') == 'env'
    assert version_label(None, 'cfg', '1.3.3dev') == 'cfg'
    assert version_label('', None, '1.3.3dev') == '1.3.3dev'


def test_lvmdrp_yaml_has_calib_dir_and_drpver():
    """ lvmdrp.yaml has both entries (empty in dev; a branch such as xdev may set them) """
    from lvmdrp import config
    assert 'calib_dir' in config and 'drpver' in config


def test_safe_hdf_append_new_column(tmp_path):
    # a drpall written before the calibdir column existed can still be appended to
    h5 = str(tmp_path / 'drpall.h5')
    safe_hdf_append(pd.DataFrame({'expnum': [1], 'object': ['a']}), h5)
    safe_hdf_append(pd.DataFrame({'expnum': [2], 'object': ['b'], 'calibdir': ['/x/calib']}), h5)
    df = pd.read_hdf(h5, key='summary')
    assert list(df['expnum']) == [1, 2]
    assert df['calibdir'].iloc[1] == '/x/calib'
    assert pd.isna(df['calibdir'].iloc[0])


@pytest.fixture
def git_repo(tmp_path):
    """ a git repository with one commit, standing in for lvmcore """
    repo = tmp_path / 'lvmcore'
    repo.mkdir()
    (repo / 'epochs.yaml').write_text('epochs: {}\n')
    git = ['git', '-c', 'user.name=test', '-c', 'user.email=test@example.com']
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(git + ['add', '.'], cwd=repo, check=True)
    subprocess.run(git + ['commit', '-q', '-m', 'first'], cwd=repo, check=True)
    head = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    return repo, head


def test_lvmcore_version(git_repo, monkeypatch):
    repo, head = git_repo
    monkeypatch.setenv('LVMCORE_DIR', str(repo))
    constants._git_version.cache_clear()
    assert get_lvmcore_version() == head


def test_lvmcore_version_dirty(git_repo, monkeypatch):
    repo, head = git_repo
    (repo / 'epochs.yaml').write_text('epochs: {61301: {}}\n')
    (repo / 'untracked.txt').write_text('not part of lvmcore\n')
    monkeypatch.setenv('LVMCORE_DIR', str(repo))
    constants._git_version.cache_clear()
    assert get_lvmcore_version() == head + '-dirty'


def test_lvmcore_version_not_git(tmp_path, monkeypatch):
    monkeypatch.setenv('LVMCORE_DIR', str(tmp_path))
    constants._git_version.cache_clear()
    assert get_lvmcore_version() == 'unknown'
    monkeypatch.setenv('LVMCORE_DIR', str(tmp_path / 'nothere'))
    assert get_lvmcore_version() == 'unknown'
