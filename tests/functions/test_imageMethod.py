
import numpy as np

from lvmdrp import path
from lvmdrp.core.image import Image
from lvmdrp.functions import imageMethod


def test_fix_pixel_shifts_noshift(make_fits):
    make_fits(mjd=61231, cameras=['b1'], expnum=3, leak=False, shift_rows=[])
    make_fits(mjd=61231, cameras=['b1'], expnum=4, leak=False, shift_rows=[])
    rpath = path.full("lvm_raw", hemi="s", camspec="b1", mjd=61231, expnum=3)
    ipath = path.full("lvm_raw", hemi="s", camspec="b1", mjd=61231, expnum=4)

    image_ori = Image()
    image_ori.loadFitsData(ipath)
    shift_columns, image_fixed = imageMethod.fix_pixel_shifts(in_image=ipath, ref_image=rpath)

    assert (shift_columns == 0).all()
    assert (image_fixed._data == image_ori._data).all()


def test_fix_pixel_shifts(make_fits):
    make_fits(mjd=61231, cameras=['b1'], expnum=5, leak=False, shift_rows=[])
    make_fits(mjd=61231, cameras=['b1'], expnum=6, leak=False, shift_rows=[1500])
    rpath = path.full("lvm_raw", hemi="s", camspec="b1", mjd=61231, expnum=5)
    ipath = path.full("lvm_raw", hemi="s", camspec="b1", mjd=61231, expnum=6)

    image_ori = Image()
    image_ori.loadFitsData(rpath)
    shift_columns, image_fixed = imageMethod.fix_pixel_shifts(in_image=ipath, ref_image=rpath)
    expected_shifts = np.zeros_like(shift_columns)
    expected_shifts[1500:] = 2

    assert (shift_columns == expected_shifts).all()
    assert (image_fixed._data == image_ori._data).all()
