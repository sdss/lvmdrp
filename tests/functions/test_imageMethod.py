
import os
import numpy as np
import pytest

from lvmdrp import path
from lvmdrp.core.image import Image, loadImage
from lvmdrp.functions import imageMethod


@pytest.fixture
def mask_2d():
    mask_2d = Image(data=np.ones((4080, 3*4120), dtype=int))
    yield mask_2d

def test_fix_pixel_shifts_noshift(make_fits, mask_2d):
    make_fits(mjd=61231, cameras=['b1', 'r1', 'z1'], expnum=3, leak=False, shift_rows=[])
    make_fits(mjd=61231, cameras=['b1', 'r1', 'z1'], expnum=4, leak=False, shift_rows=[])
    rpaths = sorted(path.expand("lvm_raw", hemi="s", camspec="?1", mjd=61231, expnum=3))
    ipaths = sorted(path.expand("lvm_raw", hemi="s", camspec="?1", mjd=61231, expnum=4))
    opaths = [path.full("lvm_anc", drpver="test", imagetype="object", tileid=11111, mjd=61231, camera=f"{channel}1", expnum=4, kind="e") for channel in "brz"]
    mask_2d_path = path.full("lvm_anc", drpver="test", imagetype="mask2d", tileid=11111, mjd=61231, camera="sp1", expnum=0, kind="")
    os.makedirs(os.path.dirname(opaths[0]), exist_ok=True)

    mask_2d.writeFitsData(mask_2d_path)

    images_ori = [loadImage(rpath) for rpath in rpaths]
    shift_columns, corrs, images_fixed = imageMethod.fix_pixel_shifts(in_images=ipaths, out_images=opaths, ref_images=rpaths, in_mask=mask_2d_path)

    for image_fixed, image_ori in zip(images_fixed, images_ori):
        assert (shift_columns == 0).all()
        assert (image_fixed._data == image_ori._data).all()


def test_fix_pixel_shifts(make_fits, mask_2d):
    make_fits(mjd=61231, cameras=['b1', 'r1', 'z1'], expnum=5, leak=False, shift_rows=[])
    make_fits(mjd=61231, cameras=['b1', 'r1', 'z1'], expnum=6, leak=False, shift_rows=[1500])
    rpaths = sorted(path.expand("lvm_raw", hemi="s", camspec="?1", mjd=61231, expnum=5))
    ipaths = sorted(path.expand("lvm_raw", hemi="s", camspec="?1", mjd=61231, expnum=6))
    opaths = [path.full("lvm_anc", drpver="test", imagetype="object", tileid=11111, mjd=61231, camera=f"{channel}1", expnum=6, kind="e") for channel in "brz"]
    mask_2d_path = path.full("lvm_anc", drpver="test", imagetype="mask2d", tileid=11111, mjd=61231, camera="sp1", expnum=0, kind="")
    os.makedirs(os.path.dirname(opaths[0]), exist_ok=True)

    mask_2d.writeFitsData(mask_2d_path)

    images_ori = [loadImage(rpath) for rpath in rpaths]
    shift_columns, corrs, images_fixed = imageMethod.fix_pixel_shifts(in_images=ipaths, out_images=opaths, ref_images=rpaths, in_mask=mask_2d_path)
    expected_shifts = np.zeros_like(shift_columns)
    expected_shifts[1500:] = 2

    assert (shift_columns == expected_shifts).all()
    for image_fixed, image_ori in zip(images_fixed, images_ori):
        assert (image_fixed._data == image_ori._data).all()
