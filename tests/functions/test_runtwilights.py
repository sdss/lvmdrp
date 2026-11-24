
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
    best_continuum, models, masked_pixels, tck = fit_continuum(spectrum, mask_bands=[(3100,3200)], median_box=1, niter=10, threshold=0.5, knots=100)
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



def make_fake_mflat(path, nfibers=30, npix=50, channel="r"):
    """ write a flat (unit) master fiberflat with a slitmap of three spectrograph groups """
    from astropy.table import Table
    rng = np.random.default_rng(1)
    telescope = np.array(["Sci"] * nfibers, dtype="<U4")
    telescope[[0, 10, 20]] = "Spec"
    telescope[[5, 15]] = "SkyE"
    telescope[[9, 25]] = "SkyW"
    slitmap = Table({"fiberid": np.arange(1, nfibers + 1),
                     "spectrographid": np.repeat([1, 2, 3], nfibers // 3),
                     "telescope": telescope,
                     "xpmm": rng.uniform(-15, 15, nfibers),
                     "ypmm": rng.uniform(-15, 15, nfibers)})
    data = np.ones((nfibers, npix), dtype=np.float32)
    header = fits.Header({"CCD": channel, "IMAGETYP": "flat"})
    header[f"HIERARCH {channel.upper()} FIBERFLAT SKYCORR"] = False
    header[f"HIERARCH {channel.upper()} FIBERFLAT GROUPBY"] = "spec"
    rss = RSS(data=data, error=0.01 * data, mask=np.zeros(data.shape, dtype=bool), header=header,
              slitmap=slitmap, wave=np.linspace(6000, 7000, npix))
    rss.writeFitsData(str(path))
    return slitmap


def test_ffactor_correction_applies_gradient_to_sci_only(tmp_path):
    """ the IFU gradient is applied to Sci fibers only and undone exactly """
    from lvmdrp.core.fit_profile import IFUGradient
    from lvmdrp.functions.run_twilights import do_ffactor_correction, undo_ffactor_correction

    path = tmp_path / "lvm-mfiberflat_twilight-r.fits"
    slitmap = make_fake_mflat(path)
    coeffs, factors = [2.5, 0.024, 0.012, 0.0], [0.95, 1.01, 1.04]

    do_ffactor_correction(path, coeffs=coeffs, factor=factors, sky_cwave=6363.78, dwave=10.0, coadd_method="fit", write_output=True)
    mflat = RSS.from_file(path)
    assert mflat._header["R FIBERFLAT GRADAPP"] is True

    groups = np.repeat([1, 2, 3], 10)
    sci = np.asarray(slitmap["telescope"] == "Sci")
    x, y = slitmap["xpmm"].data, slitmap["ypmm"].data
    gradient = coeffs[0] + coeffs[1] * y + coeffs[2] * x + coeffs[3] * x * y
    expected = IFUGradient.ifu_factors(factors, groups)
    expected[sci] *= gradient[sci] / gradient[sci].mean()
    assert mflat._data[:, 0] == pytest.approx(expected, rel=1e-6)
    assert np.all(mflat._data == mflat._data[:, :1])

    undo_ffactor_correction(path, write_output=True)
    mflat = RSS.from_file(path)
    assert mflat._data == pytest.approx(np.ones_like(mflat._data), rel=1e-6)
    assert "R FIBERFLAT GRADAPP" not in mflat._header
    assert "R FIBERFLAT GCOEFF1" not in mflat._header
    assert mflat._header["R FIBERFLAT SKYCORR"] is False


def test_undo_ffactor_correction_legacy_flat(tmp_path):
    """ flats corrected without the gradient (no GRADAPP) are undone with the factors only """
    from lvmdrp.core.fit_profile import IFUGradient
    from lvmdrp.functions.run_twilights import undo_ffactor_correction

    path = tmp_path / "lvm-mfiberflat_twilight-r.fits"
    make_fake_mflat(path)
    factors = [0.95, 1.01, 1.04]
    legacy = RSS.from_file(path)
    legacy *= IFUGradient.ifu_factors(factors, np.repeat([1, 2, 3], 10))[:, None]
    legacy.setHdrValue("HIERARCH R FIBERFLAT SKYCORR", True)
    legacy.setHdrValue("HIERARCH R FIBERFLAT CWAVE", 6363.78)
    legacy.setHdrValue("HIERARCH R FIBERFLAT DWAVE", 10.0)
    legacy.setHdrValue("HIERARCH R FIBERFLAT COADD", "fit")
    for i, f in enumerate(factors):
        legacy.setHdrValue(f"HIERARCH R FIBERFLAT FACTOR{i+1}", f)
    for i, c in enumerate([2.5, 0.024, 0.012, 0.0]):
        legacy.setHdrValue(f"HIERARCH R FIBERFLAT GCOEFF{i+1}", c)
    legacy.writeFitsData(str(path))

    undo_ffactor_correction(path, write_output=True)
    mflat = RSS.from_file(path)
    assert mflat._data == pytest.approx(np.ones_like(mflat._data), rel=1e-6)
    assert mflat._header["R FIBERFLAT SKYCORR"] is False
