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

import csv
import hashlib
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from lvmdrp import log

# the masters that stage 1 (the twilight flats) of a calibration epoch reads, per camera
STAGE1_MASTERS = ('bias', 'centroids', 'sigmas', 'model', 'wave', 'lsf')
CAMERAS = tuple(f'{c}{s}' for c in 'brz' for s in (1, 2, 3))
README = 'README.txt'
# the record of every file of a set (what each link points to, size, time, optional checksum),
# so that a change underneath the set (a master replaced in the source) can be found (sdss/lvmdrp#280)
MANIFEST = 'MANIFEST.csv'
MANIFEST_COLUMNS = ('path', 'type', 'target', 'size', 'mtime', 'md5')
# left out of the manifest: the set's own records, and the Gaia cache (large, only added to, shared)
MANIFEST_SKIP = (README, MANIFEST, 'gaia_cache')


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def manifest_entries(set_dir: str, checksum: bool = False) -> Dict[str, Dict[str, str]]:
    """Describes every file of a calibration set as it is now.

    Args:
        set_dir: the calibration set.
        checksum: also compute the MD5 checksum of each file (reads every file once).

    Returns:
        {path relative to the set: {'type': 'link' or 'file', 'target': the file a link
        resolves to ('' for a file), 'size', 'mtime' (of the file itself, links followed;
        '' for a broken link), 'md5' ('' unless `checksum`)}}.
    """
    set_dir = os.path.abspath(set_dir)
    entries = {}
    for dirpath, dirnames, filenames in os.walk(set_dir):
        rel_dir = os.path.relpath(dirpath, set_dir)
        if rel_dir == os.curdir:
            dirnames[:] = [d for d in dirnames if d not in MANIFEST_SKIP]
        # linked directories are not walked: describe them as links
        names = filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]
        for name in names:
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if rel in MANIFEST_SKIP:
                continue
            full = os.path.join(dirpath, name)
            is_link = os.path.islink(full)
            entry = {'type': 'link' if is_link else 'file', 'target': os.path.realpath(full) if is_link else '',
                     'size': '', 'mtime': '', 'md5': ''}
            if os.path.exists(full):
                st = os.stat(full)
                entry['size'], entry['mtime'] = str(st.st_size), f'{st.st_mtime:.3f}'
                if checksum and os.path.isfile(full):
                    entry['md5'] = _md5(full)
            entries[rel] = entry
    return entries


def write_manifest(set_dir: str, checksum: bool = False) -> int:
    """Writes MANIFEST.csv of a calibration set (see :func:`manifest_entries`); returns the number of rows."""
    entries = manifest_entries(set_dir, checksum=checksum)
    with open(os.path.join(set_dir, MANIFEST), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLUMNS)
        for rel in sorted(entries):
            writer.writerow([rel] + [entries[rel][c] for c in MANIFEST_COLUMNS[1:]])
    return len(entries)


def read_manifest(set_dir: str) -> Optional[Dict[str, Dict[str, str]]]:
    """Reads MANIFEST.csv of a calibration set, or returns None if it has none."""
    path = os.path.join(set_dir, MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path, newline='') as f:
        return {row['path']: {c: row[c] for c in MANIFEST_COLUMNS[1:]} for row in csv.DictReader(f)}


def check_calibration_set(set_dir: str, checksum: bool = False) -> Optional[Dict[str, List[str]]]:
    """Compares a calibration set with its manifest.

    A file counts as changed if a link now points elsewhere, or its size or time
    differs, or (with `checksum`, where the manifest has one) its checksum differs.

    Args:
        set_dir: the calibration set.
        checksum: compare checksums too, where the manifest has them.

    Returns:
        None if the set has no manifest; otherwise {'changed', 'missing' (in the manifest,
        gone or a broken link now), 'added' (not in the manifest)}: lists of paths
        relative to the set, all empty if nothing changed.
    """
    recorded = read_manifest(set_dir)
    if recorded is None:
        return None
    want_md5 = checksum and any(e['md5'] for e in recorded.values())
    now = manifest_entries(set_dir, checksum=want_md5)
    result = {'changed': [], 'missing': [], 'added': sorted(set(now) - set(recorded))}
    for rel, old in sorted(recorded.items()):
        new = now.get(rel)
        if new is None or (old['size'] and not new['size']):
            result['missing'].append(rel)
            continue
        keys = ['type', 'target', 'size', 'mtime'] + (['md5'] if want_md5 and old['md5'] else [])
        if any(old[k] != new[k] for k in keys):
            result['changed'].append(rel)
    return result


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


class CalibrationSetChanged(RuntimeError):
    """A calibration set differs from its manifest: something changed underneath it."""

    def __init__(self, set_dir: str, diff: Dict[str, List[str]]):
        self.diff = diff
        lines = [f'{k}: {len(v)} ({", ".join(v[:5])}{", ..." if len(v) > 5 else ""})' for k, v in diff.items() if v]
        super().__init__(f'calibration set {set_dir} differs from its manifest; ' + '; '.join(lines))


def build_calibration_set(out_dir: str, source_dir: Optional[str] = None, flats_from: Optional[str] = None,
                          epochs: Optional[Sequence[int]] = None, dry_run: bool = False,
                          checksum: bool = False, accept_changes: bool = False) -> Dict:
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
        checksum: record the MD5 checksum of every file in the manifest (reads every file once).
        accept_changes: go ahead although the set differs from its manifest (and record the
            new state); otherwise that is an error.

    Returns:
        A dict with 'links' (number made), 'copied' (paths of the flats copied), 'missing'
        (epochs asked for without the three flats) and 'check' (the comparison with the
        manifest before the build, None for a new set).

    Raises:
        ValueError: if `out_dir` is `source_dir` or inside it.
        FileNotFoundError: if `source_dir` doesn't exist.
        CalibrationSetChanged: if the set differs from its manifest and not `accept_changes`.

    Every build ends by writing ``MANIFEST.csv`` (:func:`write_manifest`): what each link
    points to, with size and time, so that :func:`check_calibration_set` (``drp calibrations
    check-calib-set``) can tell later whether anything changed underneath the set.
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

    result = {'links': 0, 'copied': [], 'missing': [], 'check': None}
    if os.path.isdir(out_dir):
        result['check'] = check_calibration_set(out_dir, checksum=checksum)
        if result['check'] and any(result['check'].values()):
            err = CalibrationSetChanged(out_dir, result['check'])
            if not accept_changes:
                raise err
            log.warning(f'{err}; going ahead (accept_changes)')
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
        if result['check'] and any(result['check'].values()):
            f.write('  accepted changes underneath the set: '
                    + '; '.join(f'{k} {len(v)}' for k, v in result['check'].items() if v) + '\n')
    nrows = write_manifest(out_dir, checksum=checksum)
    log.info(f'{MANIFEST}: {nrows} files recorded{" with checksums" if checksum else ""}')
    return result
