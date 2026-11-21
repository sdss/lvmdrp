# encoding: utf-8
#
# calibset.py
"""Calibration sets: a new calibration directory built from an existing one, and the masters
an epoch's stage 1 needs linked into a reduction tree (sdss/lvmdrp#268, #272).

A calibration set is a directory laid out like $LVM_MASTER_DIR (``<epoch>/lvm-m*.fits``,
``pixelmasks/``, ``gaia_cache/``, ...). A new set is made of symbolic links to every file of
an existing one, with copies of the new files (the stage-1 twilight flats, the flat-field
models) in place of their links, so that the existing set is never changed.
"""

import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from lvmdrp import log

# the masters that stage 1 (the twilight flats) of a calibration epoch reads, per camera
STAGE1_MASTERS = ('bias', 'centroids', 'sigmas', 'model', 'wave', 'lsf')
CAMERAS = tuple(f'{c}{s}' for c in 'brz' for s in (1, 2, 3))
README = 'README.txt'


def default_source_dir() -> str:
    """The standard calibration set, $SAS_BASE_DIR/sdsswork/lvm/sandbox/calib."""
    return os.path.join(os.getenv('SAS_BASE_DIR', ''), 'sdsswork', 'lvm', 'sandbox', 'calib')


def stage1_calib_dir(drpver: str, epoch: int) -> str:
    """Where stage 1 of an epoch reads its masters and writes its twilight flats:
    ``redux/<drpver>/0011XX/11111/<epoch>/calib``."""
    from lvmdrp import path
    return os.path.dirname(path.full('lvm_master', drpver=drpver, tileid=11111, mjd=int(epoch),
                                     kind='mbias', camera='b1'))


def stage1_flat(drpver: str, epoch: int, channel: str) -> str:
    """The stage-1 master twilight flat of an epoch and channel in the `drpver` tree."""
    from lvmdrp import path
    return path.full('lvm_master', drpver=drpver, tileid=11111, mjd=int(epoch),
                     kind='mfiberflat_twilight', camera=channel)


def _link(src: str, dst: str) -> None:
    """Points `dst` at `src`, replacing an earlier link but never a real file."""
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.exists(dst):
        raise FileExistsError(f'{dst} is a file, not a link; not replaced')
    os.symlink(src, dst)


def link_epoch_masters(calib_dir: str, drpver: str, epoch: int,
                       dry_run: bool = False) -> Tuple[int, List[str]]:
    """Links the masters that stage 1 of an epoch needs from a calibration set into the reduction tree.

    Stage 1 (``create_twilight_fiberflats``) reads its masters from the tree of the
    reduction (``redux/<drpver>/0011XX/11111/<epoch>/calib``), not from the calibration
    set, so they are linked there first: bias, centroids, sigmas, model, wave and lsf of
    the 9 cameras, and the pixel masks.

    Args:
        calib_dir: the calibration set to link from.
        drpver: the version label of the reduction tree.
        epoch: the calibration epoch.
        dry_run: report what would be linked, link nothing.

    Returns:
        The number of masters linked and the list of those missing in `calib_dir`.
    """
    out = stage1_calib_dir(drpver, epoch)
    root = os.path.dirname(os.path.dirname(out))
    missing, n = [], 0
    if not dry_run:
        os.makedirs(out, exist_ok=True)
        masks = os.path.join(root, 'pixelmasks')
        if not os.path.lexists(masks):
            os.symlink(os.path.join(calib_dir, 'pixelmasks'), masks)
    for kind in STAGE1_MASTERS:
        for cam in CAMERAS:
            name = f'lvm-m{kind}-{cam}.fits'
            src = os.path.join(calib_dir, str(epoch), name)
            if not os.path.exists(src):
                missing.append(src)
                continue
            if not dry_run:
                _link(src, os.path.join(out, name))
            n += 1
    log.info(f'epoch {epoch}: {n} masters {"to link" if dry_run else "linked"} from {calib_dir} into {out}'
             + (f'; {len(missing)} missing' if missing else ''))
    return n, missing


def stage1_epochs(drpver: str) -> List[int]:
    """The epochs with all three stage-1 twilight flats in the `drpver` tree."""
    root = os.path.dirname(os.path.dirname(stage1_calib_dir(drpver, 0)))
    if not os.path.isdir(root):
        return []
    return sorted(int(d) for d in os.listdir(root)
                  if d.isdigit() and all(os.path.isfile(stage1_flat(drpver, int(d), ch)) for ch in 'brz'))


def build_calibration_set(out_dir: str, source_dir: Optional[str] = None, flats_from: Optional[str] = None,
                          epochs: Optional[Sequence[int]] = None, dry_run: bool = False) -> Dict:
    """Builds (or updates) a calibration set from an existing one, with new twilight flats.

    Every file of `source_dir` gets a symbolic link at the same place in `out_dir`, except
    for entries `out_dir` already has, which are kept (so copies made earlier stay); the
    Gaia cache is linked as one directory, so that both sets share it. With `flats_from`,
    the stage-1 twilight flats of that reduction (``redux/<flats_from>/0011XX/11111/
    <epoch>/calib``) are copied into `out_dir` in place of their links. `source_dir` is
    never changed. Each run is recorded in ``README.txt`` in `out_dir`.

    Args:
        out_dir: the calibration set to build; created if needed.
        source_dir: the set to link from; default: ``$SAS_BASE_DIR/sdsswork/lvm/sandbox/calib``.
        flats_from: version label of the reduction whose stage-1 twilight flats are copied in.
        epochs: the epochs whose flats are copied; default: all with flats in `flats_from`.
        dry_run: report what would be done, change nothing.

    Returns:
        A dict with 'links' (number made), 'copied' (paths of the flats copied) and 'missing'
        (epochs asked for without the three flats).

    Raises:
        ValueError: if `out_dir` is `source_dir` or inside it.
        FileNotFoundError: if `source_dir` doesn't exist.
    """
    from lvmdrp.core.constants import get_lvmcore_version, get_lvmdrp_version
    from lvmdrp.functions.flatgradient import link_calibration_tree

    source_dir = os.path.abspath(source_dir or default_source_dir())
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f'calibration set {source_dir} not found')
    real_src, real_out = os.path.realpath(source_dir), os.path.realpath(out_dir)
    if real_out == real_src or real_out.startswith(real_src + os.sep):
        raise ValueError(f'the new calibration set {out_dir} must not be {source_dir} or inside it')

    result = {'links': 0, 'copied': [], 'missing': []}
    if flats_from:
        available = stage1_epochs(flats_from)
        wanted = sorted(int(e) for e in epochs) if epochs else available
        result['missing'] = [e for e in wanted if e not in available]
        todo = [e for e in wanted if e in available]
    else:
        todo = []

    log.info(f'calibration set {out_dir}: links to {source_dir}'
             + (f'; stage-1 twilight flats of {flats_from} for {len(todo)} epochs' if flats_from else ''))
    if result['missing']:
        log.warning(f'no stage-1 twilight flats in {flats_from} for epochs {result["missing"]}')
    if dry_run:
        for e in todo:
            log.info(f'would copy the twilight flats of epoch {e} from {stage1_calib_dir(flats_from, e)}')
        return result

    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(source_dir, 'gaia_cache')
    if os.path.isdir(cache) and not os.path.lexists(os.path.join(out_dir, 'gaia_cache')):
        os.symlink(cache, os.path.join(out_dir, 'gaia_cache'))
    result['links'] = link_calibration_tree(source_dir, out_dir)
    for e in todo:
        os.makedirs(os.path.join(out_dir, str(e)), exist_ok=True)
        for ch in 'brz':
            src = stage1_flat(flats_from, e, ch)
            dst = os.path.join(out_dir, str(e), os.path.basename(src))
            if os.path.islink(dst):
                os.remove(dst)
            shutil.copy2(src, dst)
            result['copied'].append(dst)
    log.info(f'{result["links"]} links made, {len(result["copied"])} twilight flats copied into {out_dir}')

    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    new = not os.path.exists(os.path.join(out_dir, README))
    with open(os.path.join(out_dir, README), 'a') as f:
        if new:
            f.write('Calibration set built with lvmdrp (drp calibrations build-calib-set).\n'
                    'Files are symbolic links to the source set, except those listed as copied below.\n'
                    'The source set is never changed.\n')
        f.write(f'\n{stamp}  lvmdrp {get_lvmdrp_version()}  lvmcore {get_lvmcore_version()}\n'
                f'  source: {source_dir} ({result["links"]} new links)\n')
        if todo:
            f.write(f'  stage-1 twilight flats copied from {flats_from} for epochs: '
                    f'{" ".join(map(str, todo))}\n')
        if result['missing']:
            f.write(f'  epochs asked for without stage-1 flats: {" ".join(map(str, result["missing"]))}\n')
    return result
