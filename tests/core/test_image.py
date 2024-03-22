
import pytest
import numpy as np
from astropy.io import fits

from lvmdrp.core.image import Image, _parse_ccd_section, _model_overscan
from lvmdrp.functions.imageMethod import DEFAULT_BIASSEC, DEFAULT_TRIMSEC


def create_fake_raw(path, tileid=11111, mjd=61234, expnum=6817, cameras=None, leak=True):
    """ create a fake raw frame FITS file """
    out = {}
    cameras = cameras or ['b1']
    for cam in cameras:
        # create fake header
        filename = f'sdR-s-{cam}-{expnum:0>8}.fits.gz'
        hdr = {'TILE_ID': tileid, 'MJD': mjd, 'EXPOSURE': expnum, "CCD": cam,
               'EXPTIME': 900.0,
               'IMAGETYP': 'object', 'FILENAME': filename, 'SPEC': f'sp{cam[-1]}',
               'OBSTIME': '2023-10-18T07:56:23.289', 'OBSERVAT': 'LCO',
               'TELESCOP': 'SDSS 0.16m', 'SURVEY': 'LVM'}

        # mock data
        data = np.zeros((4080, 4120), dtype=int)
        bias_levels = [980, 960, 1000, 1020]
        sci_level = 10000
        for iquad in range(4):
            (os_ix,os_fx), (os_iy,os_fy) = _parse_ccd_section(DEFAULT_BIASSEC[iquad])
            (sc_ix,sc_fx), (sc_iy,sc_fy) = _parse_ccd_section(DEFAULT_TRIMSEC[iquad])
            data[os_iy:os_fy, os_ix:os_fx] += bias_levels[iquad]
            data[sc_iy:sc_fy, sc_ix:sc_fx] += bias_levels[iquad] + sci_level
            if leak:
                data[:, (os_ix if iquad in {0,2} else os_fx-1)] += sci_level
        prim = fits.PrimaryHDU(header=fits.Header(hdr), data=data)

        # create fake file
        full = path / filename
        hdulist = fits.HDUList(prim)
        hdulist.writeto(full)
        out[cam] = full
    return out, bias_levels, sci_level


def test_model_overscan_nomasking(datadir):
    mjd = 61231
    paths = datadir([mjd])
    name, bias_levels, sci_level = create_fake_raw(paths[0], mjd=mjd, cameras=['b1'], expnum=0)
    image = Image()
    image.loadFitsData(str(name['b1']))
    for iquad in range(4):
        os_quad = image.getSection(DEFAULT_BIASSEC[iquad])
        masked_data, profile, model = _model_overscan(os_quad, axis=1, threshold=None)
        assert profile == pytest.approx(np.ones_like(profile) * bias_levels[iquad], abs=2)
        assert model == pytest.approx(np.ones_like(model) * bias_levels[iquad], abs=2)


def test_model_overscan_masking(datadir):
    mjd = 61231
    paths = datadir([mjd])
    name, bias_levels, sci_level = create_fake_raw(paths[0], mjd=mjd, cameras=['b1'], leak=True, expnum=1)
    image = Image()
    image.loadFitsData(str(name['b1']))
    for iquad in range(4):
        os_quad = image.getSection(DEFAULT_BIASSEC[iquad])
        masked_data, profile, model = _model_overscan(os_quad, axis=1, threshold=1.0)
        assert profile == pytest.approx(np.ones_like(profile) * bias_levels[iquad], abs=2)
        assert model == pytest.approx(np.ones_like(model) * bias_levels[iquad], abs=2)
        assert np.isnan(masked_data).sum() == profile.size
        assert np.nanstd(masked_data) == pytest.approx(0.0, abs=0)
