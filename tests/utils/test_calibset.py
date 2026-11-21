# encoding: utf-8
#
# test_calibset.py
#
# Tests of building calibration sets and linking the stage-1 masters (sdss/lvmdrp#268, #272)

import os

import pytest

from lvmdrp.utils import calibset


@pytest.fixture
def source_set(tmp_path):
    """ a small calibration set: two epochs of masters, pixel masks and a Gaia cache """
    src = tmp_path / 'calib'
    for epoch in (60465, 60511):
        (src / str(epoch)).mkdir(parents=True)
        for kind in calibset.STAGE1_MASTERS:
            for cam in calibset.CAMERAS:
                (src / str(epoch) / f'lvm-m{kind}-{cam}.fits').write_text('master')
        for ch in 'brz':
            (src / str(epoch) / f'lvm-mfiberflat_twilight-{ch}.fits').write_text('old flat')
    (src / 'pixelmasks').mkdir()
    (src / 'pixelmasks' / 'lvm-mpixmask-b1.fits').write_text('mask')
    (src / 'gaia_cache').mkdir()
    (src / 'gaia_cache' / 'star.fits').write_text('xp')
    return src


def _stage1_flats(drpver, epoch):
    """ write the three stage-1 twilight flats of an epoch into the reduction tree """
    for ch in 'brz':
        path = calibset.stage1_flat(drpver, epoch, ch)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(f'new flat {epoch} {ch}')


def test_link_epoch_masters(source_set):
    n, missing = calibset.link_epoch_masters(str(source_set), 'cstest', 60465)
    assert n == len(calibset.STAGE1_MASTERS) * 9 and not missing
    out = calibset.stage1_calib_dir('cstest', 60465)
    link = os.path.join(out, 'lvm-mbias-b1.fits')
    assert os.path.islink(link) and os.readlink(link) == str(source_set / '60465' / 'lvm-mbias-b1.fits')
    # the twilight flats are what stage 1 makes: not linked
    assert not os.path.lexists(os.path.join(out, 'lvm-mfiberflat_twilight-b.fits'))
    masks = os.path.join(os.path.dirname(os.path.dirname(out)), 'pixelmasks')
    assert os.path.islink(masks)
    # again: the links are replaced, nothing breaks
    assert calibset.link_epoch_masters(str(source_set), 'cstest', 60465)[0] == n


def test_link_epoch_masters_missing_and_dry_run(source_set):
    (source_set / '60511' / 'lvm-mlsf-z3.fits').unlink()
    n, missing = calibset.link_epoch_masters(str(source_set), 'cstest2', 60511, dry_run=True)
    assert missing == [str(source_set / '60511' / 'lvm-mlsf-z3.fits')]
    assert not os.path.exists(calibset.stage1_calib_dir('cstest2', 60511))


def test_link_epoch_masters_keeps_files(source_set):
    out = calibset.stage1_calib_dir('cstest3', 60465)
    os.makedirs(out)
    with open(os.path.join(out, 'lvm-mbias-b1.fits'), 'w') as f:
        f.write('a real file')
    with pytest.raises(FileExistsError):
        calibset.link_epoch_masters(str(source_set), 'cstest3', 60465)


def test_build_calibration_set(source_set, tmp_path):
    out = tmp_path / 'calib_new'
    _stage1_flats('cstest4', 60465)
    assert calibset.stage1_epochs('cstest4') == [60465]
    res = calibset.build_calibration_set(str(out), source_dir=str(source_set), flats_from='cstest4')
    # everything linked, the new flats copied, the Gaia cache shared as one directory
    assert os.path.islink(out / '60511' / 'lvm-mbias-b1.fits')
    assert os.path.islink(out / '60511' / 'lvm-mfiberflat_twilight-b.fits')
    flat = out / '60465' / 'lvm-mfiberflat_twilight-r.fits'
    assert not os.path.islink(flat) and flat.read_text() == 'new flat 60465 r'
    assert os.path.islink(out / 'gaia_cache') and os.readlink(out / 'gaia_cache') == str(source_set / 'gaia_cache')
    assert len(res['copied']) == 3 and not res['missing']
    # the source set is not changed
    assert (source_set / '60465' / 'lvm-mfiberflat_twilight-r.fits').read_text() == 'old flat'
    assert 'cstest4' in (out / 'README.txt').read_text()
    # an update keeps the copies and adds links for new source files
    (source_set / '60511' / 'lvm-mnew-b1.fits').write_text('new')
    res = calibset.build_calibration_set(str(out), source_dir=str(source_set))
    assert res['links'] == 1 and flat.read_text() == 'new flat 60465 r'
    assert (out / 'README.txt').read_text().count('source:') == 2


def test_build_calibration_set_checks(source_set, tmp_path):
    with pytest.raises(ValueError):
        calibset.build_calibration_set(str(source_set / 'inside'), source_dir=str(source_set))
    with pytest.raises(FileNotFoundError):
        calibset.build_calibration_set(str(tmp_path / 'x'), source_dir=str(tmp_path / 'nothere'))
    out = tmp_path / 'calib_dry'
    res = calibset.build_calibration_set(str(out), source_dir=str(source_set), flats_from='cstest5',
                                         epochs=[60465], dry_run=True)
    assert res['missing'] == [60465] and not out.exists()


def test_link_calibration_tree_links_linked_directories(source_set, tmp_path):
    """ a directory that is itself a link in the source set is linked, not left out """
    from lvmdrp.functions.flatgradient import link_calibration_tree
    other = tmp_path / 'elsewhere'
    other.mkdir()
    (other / 'model.fits').write_text('m')
    os.symlink(other, source_set / 'sky_models')
    out = tmp_path / 'calib_links'
    link_calibration_tree(str(source_set), str(out))
    assert os.path.islink(out / 'sky_models') and (out / 'sky_models' / 'model.fits').read_text() == 'm'
