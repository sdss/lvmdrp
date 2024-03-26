
import pytest
import numpy as np

from lvmdrp import path
from lvmdrp.core.image import Image, _model_overscan
from lvmdrp.functions.imageMethod import DEFAULT_BIASSEC


def test_model_overscan_nomasking(make_fits):
    mjd = 61231
    bias_levels = [980, 960, 1000, 1020]
    sci_level = 100000
    make_fits(mjd=mjd, cameras=['b1'], expnum=0, bias_levels=bias_levels, sci_level=sci_level)
    ipath = path.full("lvm_raw", hemi="s", camspec="b1", mjd=61231, expnum=0)

    image = Image()
    image.loadFitsData(ipath)

    for iquad in range(4):
        os_quad = image.getSection(DEFAULT_BIASSEC[iquad])
        masked_data, profile, model = _model_overscan(os_quad, axis=1, threshold=None)
        assert profile == pytest.approx(np.ones_like(profile) * bias_levels[iquad], abs=2)
        assert model == pytest.approx(np.ones_like(model) * bias_levels[iquad], abs=2)


def test_model_overscan_masking(make_fits):
    mjd = 61231
    bias_levels = [980, 960, 1000, 1020]
    sci_level = 100000
    make_fits(mjd=mjd, cameras=['b1'], expnum=1, bias_levels=bias_levels, sci_level=sci_level, leak=True)
    ipath = path.full("lvm_raw", hemi="s", camspec="b1", mjd=61231, expnum=1)

    image = Image()
    image.loadFitsData(ipath)

    for iquad in range(4):
        os_quad = image.getSection(DEFAULT_BIASSEC[iquad])
        masked_data, profile, model = _model_overscan(os_quad, axis=1, threshold=3.0)
        assert profile == pytest.approx(np.ones_like(profile) * bias_levels[iquad], abs=2)
        assert model == pytest.approx(np.ones_like(model) * bias_levels[iquad], abs=2)
        assert np.isnan(masked_data).sum() == profile.size
        assert np.nanstd(masked_data) == pytest.approx(0.0, abs=0)
