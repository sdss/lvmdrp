# encoding: utf-8
#
# test_calseq.py

import numpy as np
import pandas as pd

from lvmdrp.core.constants import LVM_NBLOCKS
from lvmdrp.functions.run_calseq import SLITMAP, choose_sequence, get_sequence_iterator


def _frames(expnums, calibfibs=None, cameras=("b1", "r1", "z1")):
    calibfibs = calibfibs or [None] * len(expnums)
    rows = [{"mjd": 60500, "tileid": 11111, "expnum": expnum, "camera": camera, "calibfib": calibfib,
             "imagetyp": "flat", "ldls": True, "quartz": False}
            for expnum, calibfib in zip(expnums, calibfibs) for camera in cameras]
    return pd.DataFrame(rows)


def test_choose_sequence_nightly_keeps_all_cameras_of_first_exposures():
    frames = _frames([103, 101, 102])
    chosen, expnums = choose_sequence(frames, calibration="wave", kind="nightly")
    np.testing.assert_array_equal(expnums, [101, 102])
    assert len(chosen) == 2 * 3
    assert set(chosen.camera) == {"b1", "r1", "z1"}


def test_get_sequence_iterator_without_exposed_standards_fits_all_blocks():
    frames = _frames([101, 102], cameras=("b1",))
    exposed, unexposed = get_sequence_iterator(frames, camera="b1")
    assert list(exposed) == [101]
    std_fiberid, block_idxs = exposed[101]
    assert std_fiberid is None
    assert block_idxs == list(range(LVM_NBLOCKS))
    nstds = ((SLITMAP["spectrographid"] == 1) & (SLITMAP["telescope"] == "Spec")).sum()
    assert len(unexposed) == nstds


def test_get_sequence_iterator_assigns_missing_blocks_to_first_exposed_standard():
    slitmap = SLITMAP[(SLITMAP["spectrographid"] == 1) & (SLITMAP["telescope"] == "Spec")]
    std = slitmap["orig_ifulabel"][0]
    block = int(slitmap["blockid"][0][1:]) - 1
    frames = _frames([101, 102], calibfibs=[None, std], cameras=("b1",))
    exposed, unexposed = get_sequence_iterator(frames, camera="b1")
    assert list(exposed) == [102]
    assert exposed[102][0] == std
    assert sorted(exposed[102][1]) == list(range(LVM_NBLOCKS))
    assert block in exposed[102][1]
    assert std not in unexposed
