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


def test_manifest_records_the_set(source_set, tmp_path):
    out = tmp_path / 'calib_m'
    _stage1_flats('cstest6', 60465)
    calibset.build_calibration_set(str(out), source_dir=str(source_set), flats_from='cstest6', checksum=True)
    rows = calibset.read_manifest(str(out))
    link = rows[os.path.join('60511', 'lvm-mbias-b1.fits')]
    assert link['type'] == 'link' and link['target'] == os.path.realpath(source_set / '60511' / 'lvm-mbias-b1.fits')
    assert link['size'] == str(len('master')) and len(link['md5']) == 32
    assert rows[os.path.join('60465', 'lvm-mfiberflat_twilight-b.fits')]['type'] == 'file'
    # the set's own records and the Gaia cache are left out
    assert not any(p.startswith(('gaia_cache', 'README', 'MANIFEST')) for p in rows)
    assert calibset.check_calibration_set(str(out)) == {'changed': [], 'missing': [], 'added': []}


def test_check_finds_changes_underneath(source_set, tmp_path):
    out = tmp_path / 'calib_c'
    calibset.build_calibration_set(str(out), source_dir=str(source_set), checksum=True)
    # a master replaced in the source set (same size, so only the checksum or time tells)
    master = source_set / '60465' / 'lvm-mwave-r2.fits'
    master.write_text('MASTER')
    os.utime(master, (1e9, 1e9))
    (source_set / '60511' / 'lvm-mlsf-b3.fits').unlink()
    (out / '60511' / 'extra.fits').write_text('x')
    diff = calibset.check_calibration_set(str(out), checksum=True)
    assert diff['changed'] == [os.path.join('60465', 'lvm-mwave-r2.fits')]
    assert diff['missing'] == [os.path.join('60511', 'lvm-mlsf-b3.fits')]
    assert diff['added'] == [os.path.join('60511', 'extra.fits')]
    # a new build stops, unless the changes are accepted, which records the new state
    with pytest.raises(calibset.CalibrationSetChanged, match='differs from its manifest'):
        calibset.build_calibration_set(str(out), source_dir=str(source_set))
    calibset.build_calibration_set(str(out), source_dir=str(source_set), accept_changes=True)
    assert 'accepted changes' in (out / 'README.txt').read_text()
    assert not any(calibset.check_calibration_set(str(out)).values())


def test_check_without_manifest(source_set, tmp_path):
    assert calibset.check_calibration_set(str(source_set)) is None


def _reduction_masters(drpver, epoch, kinds=calibset.EPOCH_MASTERS, skip=()):
    """ write masters of an epoch into the reduction tree, as drp calibrations long-term / fiberflats -e do """
    calib = calibset.stage1_calib_dir(drpver, epoch)
    os.makedirs(calib, exist_ok=True)
    for kind in kinds:
        units = 'brz' if kind in calibset.PER_CHANNEL else calibset.CAMERAS
        for u in units:
            name = f'lvm-m{kind}-{u}.fits'
            if name not in skip:
                with open(os.path.join(calib, name), 'w') as f:
                    f.write(f'rebuilt {kind} {u}')
    return calib


def test_build_calibration_set_every_master(source_set, tmp_path):
    calib = _reduction_masters('cstest7', 60465)
    # an input linked in from a calibration set (as stage 1 does) is not a master of the reduction
    os.symlink(str(source_set / '60465' / 'lvm-mbias-b1.fits'), os.path.join(calib, 'lvm-mextra-b1.fits'))
    assert calibset.reduction_epochs('cstest7') == [60465]
    out = tmp_path / 'calib_all'
    res = calibset.build_calibration_set(str(out), source_dir=str(source_set), masters_from='cstest7')
    assert res['kinds'] == {60465: sorted(calibset.EPOCH_MASTERS)}
    assert len(res['copied']) == 7 * 9 + 3
    for name in ('lvm-mbias-b1.fits', 'lvm-mwave-z3.fits', 'lvm-mcounts-r2.fits', 'lvm-mfiberflat_twilight-b.fits'):
        copied = out / '60465' / name
        assert not os.path.islink(copied) and copied.read_text().startswith('rebuilt')
    assert not any('lvm-mextra' in p for p in res['copied'])
    # the other epoch stays links to the source set, which is unchanged
    assert os.path.islink(out / '60511' / 'lvm-mwave-z3.fits')
    assert (source_set / '60465' / 'lvm-mwave-z3.fits').read_text() == 'master'
    readme = (out / 'README.txt').read_text()
    assert 'epoch 60465: copied from cstest7: bias, centroids, counts' in readme
    assert calibset.read_manifest(str(out))[os.path.join('60465', 'lvm-mwave-z3.fits')]['type'] == 'file'


def test_build_calibration_set_some_kinds(source_set, tmp_path):
    # the reduction remade only the wavelength solutions: the rest stays linked
    _reduction_masters('cstest8', 60511, kinds=('wave', 'lsf', 'bias'))
    out = tmp_path / 'calib_wave'
    res = calibset.build_calibration_set(str(out), source_dir=str(source_set), masters_from='cstest8',
                                         kinds=['wave', 'lsf'])
    assert res['kinds'] == {60511: ['lsf', 'wave']}
    assert not os.path.islink(out / '60511' / 'lvm-mwave-b1.fits')
    assert os.path.islink(out / '60511' / 'lvm-mbias-b1.fits')       # not asked for
    assert os.path.islink(out / '60511' / 'lvm-mcentroids-b1.fits')  # not remade


def test_build_calibration_set_incomplete_masters(source_set, tmp_path):
    _reduction_masters('cstest9', 60465, kinds=('wave', 'lsf'), skip=('lvm-mlsf-z3.fits',))
    out = tmp_path / 'calib_bad'
    with pytest.raises(ValueError, match='epoch 60465: lsf 8 files'):
        calibset.build_calibration_set(str(out), source_dir=str(source_set), masters_from='cstest9')
    assert not out.exists()


def test_build_calibration_set_argument_checks(source_set, tmp_path):
    with pytest.raises(ValueError, match='not both'):
        calibset.build_calibration_set(str(tmp_path / 'a'), source_dir=str(source_set),
                                       masters_from='x', flats_from='y')
    with pytest.raises(ValueError, match='unknown kinds'):
        calibset.build_calibration_set(str(tmp_path / 'b'), source_dir=str(source_set),
                                       masters_from='x', kinds=['pixflat'])
    res = calibset.build_calibration_set(str(tmp_path / 'c'), source_dir=str(source_set),
                                         masters_from='cstest10', epochs=[60465], dry_run=True)
    assert res['missing'] == [60465]
