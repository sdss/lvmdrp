# encoding: utf-8
#
# test_provenance.py
#
# Tests of what the products record to reproduce them: the lvmdrp commit and the FRAMES
# extension of the master fiber flats (sdss/lvmdrp#268, #272)

import os
import re
import subprocess

import numpy as np
import pandas as pd
from astropy.io import fits

from lvmdrp.core import constants
from lvmdrp.core.constants import add_provenance, get_lvmdrp_version
from lvmdrp.core.header import metadata_frames_table, read_frames_extension, write_frames_extension


def test_lvmdrp_version_of_checkout(tmp_path, monkeypatch):
    # a git repository standing in for the lvmdrp checkout (CI installs lvmdrp as a package)
    repo = tmp_path / 'lvmdrp'
    repo.mkdir()
    (repo / 'setup.cfg').write_text('[metadata]\n')
    git = ['git', '-c', 'user.name=test', '-c', 'user.email=test@example.com']
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(git + ['add', '.'], cwd=repo, check=True)
    subprocess.run(git + ['commit', '-q', '-m', 'first'], cwd=repo, check=True)
    head = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(constants, 'LVMDRP_REPO_DIR', str(repo))
    constants._git_version.cache_clear()
    assert re.fullmatch(r'[0-9a-f]{12}', get_lvmdrp_version())
    assert get_lvmdrp_version() == head
    (repo / 'setup.cfg').write_text('[metadata]\nname = changed\n')
    constants._git_version.cache_clear()
    assert get_lvmdrp_version() == head + '-dirty'


def test_lvmdrp_version_not_git(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, 'LVMDRP_REPO_DIR', str(tmp_path))
    assert get_lvmdrp_version() == 'unknown'


def test_add_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv('LVM_MASTER_DIR', str(tmp_path))
    hdr = add_provenance(fits.Header())
    assert hdr['CALIBDIR'] == str(tmp_path)
    assert hdr['LVMCORE'] == constants.get_lvmcore_version()
    assert hdr['LVMDRP'] == get_lvmdrp_version()


def _frames():
    return pd.DataFrame({'tileid': [11111] * 4, 'mjd': [60469] * 4, 'expnum': [18939, 18938, 18938, 18939],
                         'spec': ['sp1', 'sp2', 'sp1', 'sp1'], 'camera': ['b1', 'r2', 'b1', 'r1'],
                         'exptime': [3.0, 1.0, 1.0, 3.0], 'imagetyp': ['flat'] * 4})


def test_metadata_frames_table():
    table = metadata_frames_table(_frames())
    assert table.colnames == ['TILEID', 'MJD', 'EXPNUM', 'SPEC', 'CAMERA', 'EXPTIME']
    assert list(table['EXPNUM']) == [18938, 18938, 18939, 18939]
    assert list(table['CAMERA']) == ['b1', 'r2', 'b1', 'r1']
    assert list(table['MJD']) == [60469] * 4


def _flat(path):
    fits.HDUList([fits.PrimaryHDU(np.ones((3, 4), dtype='float32')),
                  fits.ImageHDU(np.zeros((3, 4), dtype='float32'), name='ERROR')]).writeto(path)


def test_write_read_frames_extension(tmp_path):
    path = str(tmp_path / 'lvm-mfiberflat_twilight-b.fits')
    _flat(path)
    assert read_frames_extension(path) is None
    frames = _frames()
    write_frames_extension(path, metadata_frames_table(frames))
    table = read_frames_extension(path)
    assert list(table['EXPNUM']) == [18938, 18938, 18939, 18939]
    with fits.open(path) as hdul:
        assert hdul[0].header['NFRAMES'] == 2
        assert hdul['ERROR'].data.shape == (3, 4)
    # written again: replaced, not appended
    write_frames_extension(path, metadata_frames_table(frames.iloc[:1]))
    assert len(read_frames_extension(path)) == 1
    with fits.open(path) as hdul:
        assert [h.name for h in hdul].count('FRAMES') == 1
        assert hdul[0].header['NFRAMES'] == 1


class _FakeFlat:
    """ stands in for an RSS, whose writer keeps only its own extensions """
    def writeFitsData(self, out_path):
        if os.path.exists(out_path):
            os.remove(out_path)
        _flat(out_path)


def test_write_flat_keeps_frames(tmp_path):
    from lvmdrp.functions.flatgradient import _write_flat
    src = str(tmp_path / 'in.fits')
    _flat(src)
    write_frames_extension(src, metadata_frames_table(_frames()))
    # in place
    _write_flat(_FakeFlat(), src)
    assert len(read_frames_extension(src)) == 4
    # to another file, replacing a link to the input (never writing through it)
    dst = str(tmp_path / 'out.fits')
    os.symlink(src, dst)
    _write_flat(_FakeFlat(), dst, in_path=src)
    assert not os.path.islink(dst)
    assert len(read_frames_extension(dst)) == 4


def test_combine_fiberflat_sequence_frames(tmp_path, monkeypatch):
    from lvmdrp.functions import run_calseq
    mflat = str(tmp_path / 'lvm-mfiberflat_twilight-b.fits')
    monkeypatch.setattr(run_calseq, 'combine_twilight_sequence', lambda **kwargs: _flat(kwargs['out_mflat']))
    calibs = {flavor: {cam: f'/c/lvm-m{flavor}-{cam}.fits' for cam in ('b1', 'r1')}
              for flavor in ('centroids', 'sigmas', 'wave', 'lsf')}
    run_calseq._combine_fiberflat_sequence('b', calibs, [], [], [], mflat, frames=_frames())
    table = read_frames_extension(mflat)
    assert list(zip(table['EXPNUM'], table['CAMERA'])) == [(18938, 'b1'), (18939, 'b1')]
    # without frames, no extension (as before)
    mflat2 = str(tmp_path / 'lvm-mfiberflat_twilight-r.fits')
    run_calseq._combine_fiberflat_sequence('r', calibs, [], [], [], mflat2)
    assert read_frames_extension(mflat2) is None
