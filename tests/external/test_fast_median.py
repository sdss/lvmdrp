import numpy as np
from lvmdrp.external.fast_median import fast_median_filter_2d, fast_median_filter_1d
from scipy.ndimage import median_filter
import time
import pytest


def test_median_filter_1d_speed():
    np.random.seed(123)
    img = np.random.random(size=100)*3 + 1000
    img = img.astype(np.float32)

    t = time.time()
    img_fm = fast_median_filter_1d(img, size=5)
    dt_fm = time.time()-t

    t = time.time()
    img_om = median_filter(img, size=5)
    dt_om = time.time()-t

    assert dt_om > dt_fm
    assert img_om == pytest.approx(img_fm, rel=0.01)


def test_median_filter_2d_speed():
    np.random.seed(456)
    img = np.random.random(size=[100,100])*3 + 1000
    img = img.astype(np.float32)

    t = time.time()
    img_fm = fast_median_filter_2d(img, size=(5,5))
    dt_fm = time.time()-t

    t = time.time()
    img_om = median_filter(img, size=(5,5))
    dt_om = time.time()-t

    assert dt_om > dt_fm
    assert img_om == pytest.approx(img_fm, rel=0.01)

