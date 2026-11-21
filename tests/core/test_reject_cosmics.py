# encoding: utf-8
#
# test_reject_cosmics.py
#
# Tests of Image.reject_cosmics (cosmic-ray rejection in detrend_frame)

import numpy as np
import pytest
from scipy import ndimage

from lvmdrp.core.image import Image, LinearSelectionElement

KWARGS = dict(rlim=1.3, iterations=5, fwhm_gauss=[2.75, 2.75], replace_box=[10, 2], replace_error=1e6,
              increase_radius=1)


def _reference(data, rdnoise, sigma_det=5, rlim=1.2, iterations=5, fwhm_gauss=[2.0, 2.0], replace_box=[5, 5],
               replace_error=1e6, increase_radius=0, binary_closure=True):
    """ reject_cosmics as it was before the Laplacian of the input was computed once (returns data, mask) """
    sigma_x, sigma_y = fwhm_gauss[0] / 2.354, fwhm_gauss[1] / 2.354
    box_x, box_y = int(replace_box[0]), int(replace_box[1])
    LA_kernel = 0.25*np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    img_original = Image(data=data.copy())
    img = Image(data=data.copy())
    img_original._error = np.sqrt((np.clip(img_original._data, a_min=0.0, a_max=None) + rdnoise**2))
    select = np.zeros(img._dim, dtype=bool)
    img_original._mask = np.zeros(img._dim, dtype=bool)
    img._mask = np.zeros(img._dim, dtype=bool)
    out = img
    for i in range(iterations):
        noise = out.medianImg((box_y, box_x))
        noise.replace_subselect(noise._data <= 0, data=0)
        noise = (noise + rdnoise ** 2).sqrt()
        conv = img.subsampleImg().convolveImg(LA_kernel)
        conv.replace_subselect(conv < 0, data=0)
        Lap = conv.rebin(2, 2)
        S = Lap/(noise*2)
        S_prime = S-S.medianImg((5, 5))
        fine = out.convolveGaussImg(sigma_x, sigma_y, mask=True)
        fine_norm = out/fine
        fine_norm.replace_subselect(fine_norm < 0, data=0)
        Lap2 = fine_norm.subsampleImg().convolveImg(LA_kernel).rebin(2, 2)
        select = np.logical_or(np.logical_and(Lap2 > rlim, S_prime > sigma_det), select)
        if i == iterations-1:
            img_original.replace_subselect(select, mask=True)
            if increase_radius > 0:
                kernel = np.ones((2*increase_radius+1, 2*increase_radius+1))
                kernel /= kernel.sum()
                img_original._mask = Image(data=img_original._mask).convolveImg(kernel=kernel)._data >= 2/kernel.size
            if binary_closure:
                bmask = img_original._mask > 0
                bc_mask = np.zeros(bmask.shape, dtype=img_original._mask.dtype)
                for ang in [20, 45, 70, 90, 110, 135, 160]:
                    bc_mask = bc_mask | ndimage.binary_closing(bmask, structure=LinearSelectionElement(11, 11, ang).se)
                img_original._mask = bc_mask
            out = img_original.replaceMaskMedian(box_x, box_y, replace_error=replace_error)
        else:
            out.replace_subselect(select, mask=True)
            out = out.replaceMaskMedian(box_x, box_y, replace_error=None)
    return out._data, out._mask


@pytest.fixture(scope='module')
def frame():
    """ a small frame with fiber-like rows, noise and cosmic rays (two-pixel hits and a track) """
    rng = np.random.default_rng(7)
    ny, nx = 160, 200
    y = np.arange(ny)[:, None]
    rows = sum(800 * np.exp(-0.5 * ((y - c) / 1.3) ** 2) for c in range(8, ny, 7))
    data = (rows + 50) * np.ones((1, nx))
    data = rng.normal(data, np.sqrt(data + 9.0)).astype(np.float32)
    hits = [(20, 30), (75, 120), (130, 60), (100, 180)]
    for yy, xx in hits:                       # two pixels each: with increase_radius=1 a detection
        data[yy, xx:xx + 2] += 20000          # needs a flagged neighbour to be kept
    for k in range(12):                       # a diagonal track
        data[40 + k, 150 + k] += 15000
    return data, hits


def test_reject_cosmics_identical_to_reference(frame):
    data, _ = frame
    img = Image(data=data.copy(), mask=np.zeros(data.shape, dtype=bool))
    img.reject_cosmics(gain=1.0, rdnoise=3.0, inplace=True, **KWARGS)
    ref_data, ref_mask = _reference(data, 3.0, **KWARGS)
    assert np.array_equal(img._data, ref_data)
    assert np.array_equal(img._mask, ref_mask)


def test_reject_cosmics_finds_hits(frame):
    data, hits = frame
    img = Image(data=data.copy(), mask=np.zeros(data.shape, dtype=bool))
    img.reject_cosmics(gain=1.0, rdnoise=3.0, inplace=True, **KWARGS)
    for yy, xx in hits:
        assert img._mask[yy, xx:xx + 2].all()
    assert img._mask[40:52, 150:162].diagonal().all()
    assert img._mask.mean() < 0.05
