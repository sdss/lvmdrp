
import numpy as np
from astropy.io import fits
import pytest

from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS

from lvmdrp.functions.run_twilights import fit_continuum


def create_fake_twilight(path, tileid=11111, mjd=61234, expnum=6817, cameras=None, imagetyp=None):
    """ create a fake raw frame FITS file """
    out = {}
    cameras = cameras or ['b1']
    imagetyp = imagetyp or 'object'
    for cam in cameras:
        # create fake header
        filename = f'sdR-s-{cam}-{expnum:0>8}.fits.gz'
        hdr = {'TILE_ID': tileid, 'MJD': mjd, 'EXPOSURE': expnum, "CCD": cam,
               'EXPTIME': 900.0,
               'IMAGETYP': imagetyp, 'FILENAME': filename, 'SPEC': f'sp{cam[-1]}',
               'OBSTIME': '2023-10-18T07:56:23.289', 'OBSERVAT': 'LCO',
               'TELESCOP': 'SDSS 0.16m', 'SURVEY': 'LVM'}
        prim = fits.PrimaryHDU(header=fits.Header(hdr))

        # create fake file
        full = path / filename
        hdulist = fits.HDUList(prim)
        hdulist.writeto(full)
        out[cam] = full
    return out


@pytest.fixture
def make_twilight_sequence(datadir):
    """ fixture to create fake fits files """
    mjd = 61235
    paths = datadir([mjd])
    for i in range(12):
        create_fake_twilight(paths[0], expnum=7700+i, mjd=mjd, cameras=['b1', 'b2', 'b3', 'r1', 'r2', 'r3', 'z1', 'z2', 'z3'], imagetyp="flat")


def make_fake_spectrum():
    w = np.arange(3000, 5000+1, dtype=float)
    f = 1000**2 - (w-4000)**2
    f[[300, 1000]+list(i for i in range(1500,1550,5))] *= 0.1
    m = np.zeros_like(w, dtype=bool)
    return Spectrum1D(wave=w, data=f, error=np.sqrt(f), mask=m)

def make_fake_rss(tileid=11111, mjd=61234, expnum=6817, cameras=None, imagetyp=None):
    cameras = cameras or ['b1']
    imagetyp = imagetyp or 'object'
    out = []
    for cam in cameras:
        # create fake header
        filename = f'sdR-s-{cam}-{expnum:0>8}.fits.gz'
        hdr = {'TILE_ID': tileid, 'MJD': mjd, 'EXPOSURE': expnum, "CCD": cam,
               'EXPTIME': 900.0,
               'IMAGETYP': imagetyp, 'FILENAME': filename, 'SPEC': f'sp{cam[-1]}',
               'OBSTIME': '2023-10-18T07:56:23.289', 'OBSERVAT': 'LCO',
               'TELESCOP': 'SDSS 0.16m', 'SURVEY': 'LVM',
               'BUNIT': "electron"}

        s = make_fake_spectrum()
        w = s._wave
        f = np.repeat([s._data], 10, axis=0)
        m = np.repeat([s._mask], 10, axis=0)
        x = fits.PrimaryHDU(header=fits.Header(hdr), data=f)
        rss = RSS(wave=w, data=f, error=np.sqrt(f), mask=m, header=x.header)
        out.append(rss)
    return out


def test_fit_continuum():
    """ test continuum fitting """
    spectrum = make_fake_spectrum()
    best_continuum, models, masked_pixels, tck = fit_continuum(spectrum, mask_bands=[(3100,3200)], median_box=1, niter=10, threshold=0.5)
    assert best_continuum.size == spectrum._data.size
    assert len(models) == 0
    assert masked_pixels.sum() == 12
    assert len(tck[0]) == 100+3
    assert best_continuum[0] == pytest.approx(0, abs=1e-10)
    assert best_continuum[-1] == pytest.approx(0, abs=1e-10)


# def test_fit_fiberflat():
#     """ test fitting a fiberflat """
#     rsss = make_fake_rss(imagetyp="flat", cameras=["b1", "b2", "b3"])
#     new_rss = fit_fiberflat(rsss, plot_fibers=[0,5,9], interpolate_bad=False)
#     assert len(new_rss) == 3
#     assert new_rss[0]._data.shape == (10, 2001)
#     assert new_rss[0]._data == pytest.approx(np.ones_like(new_rss[0]._data), abs=1e-10)
#     assert new_rss[1]._data == pytest.approx(np.ones_like(new_rss[1]._data), abs=1e-10)
#     assert new_rss[2]._data == pytest.approx(np.ones_like(new_rss[2]._data), abs=1e-10)


# def test_fit_fiberflat_bad():
#     """ test fitting a fiberflat """
#     rsss = make_fake_rss(imagetyp="flat", cameras=["b1", "b2", "b3"])
#     rsss[0]._data[0] = 0
#     rsss[0]._data[5] = 0
#     rsss[0]._data[9] = 0
#     new_rss = fit_fiberflat(rsss, plot_fibers=[0,5,9], interpolate_bad=False)
#     assert len(new_rss) == 3
#     assert new_rss[0]._data.shape == (10, 2001)
#     assert new_rss[0]._data[0] == pytest.approx(0, abs=1e-10)
#     assert new_rss[0]._data[5] == pytest.approx(0, abs=1e-10)
#     assert new_rss[0]._data[9] == pytest.approx(0, abs=1e-10)
#     assert new_rss[1]._data == pytest.approx(np.ones_like(new_rss[1]._data), abs=1e-10)
#     assert new_rss[2]._data == pytest.approx(np.ones_like(new_rss[2]._data), abs=1e-10)

