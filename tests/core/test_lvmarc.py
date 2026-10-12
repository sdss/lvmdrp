# encoding: utf-8
#
# tests of the arc products: line measurements stored in lvmArc, the FRAMES table of combined
# products and the normalization of RSS combinations

import numpy as np
from astropy.io import fits

from lvmdrp.core.header import individual_frames_table
from lvmdrp.core.rss import RSS, lvmArc
from lvmdrp.core.tracemask import TraceMask


def make_lvmarc(slitmap, nfibers=8, npix=50, waves=(5000.0, 5100.0, 5200.0), wave_trace=None):
    rng = np.random.default_rng(3)
    nlines = len(waves)
    cent = rng.uniform(5, npix - 5, (nfibers, nlines))
    mask = np.zeros((nfibers, nlines), dtype=bool)
    mask[1, 2] = True
    cent[1, 2] = np.nan
    return lvmArc(data=rng.normal(0, 1, (nfibers, npix)).astype(np.float32), error=np.ones((nfibers, npix), np.float32),
                  mask=np.zeros((nfibers, npix), bool), header=fits.Header({"CCD": "b1", "EXPOSURE": 7}),
                  ref_wave=list(waves), cent_line=cent, fwhm_line=np.full((nfibers, nlines), 2.5),
                  flux_line=np.full((nfibers, nlines), 100.0), mask_line=mask, bkg_line=np.full((nfibers, nlines), 3.0),
                  wave_trace=wave_trace, slitmap=slitmap[:nfibers])


def test_lvmarc_lines_roundtrip(tmp_path, make_arc):
    """ line tables, rejection mask and backgrounds are written and read back as they were """
    lvmarc = make_lvmarc(make_arc.slitmap())
    path = tmp_path / "lvmArc-b1-00000007.fits"
    lvmarc.writeFitsData(str(path))

    with fits.open(path) as hdus:
        names = [hdu.name for hdu in hdus]
    for extname in ("LINE_CENT", "LINE_FWHM", "LINE_FLUX", "LINE_MASK", "LINE_BKG", "SLITMAP"):
        assert extname in names

    expected = lvmarc.get_lines()
    read = lvmArc.from_file(str(path))
    for a, b in zip(read.get_lines(), expected):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(read.get_line_table("LINE_BKG"), 3.0)
    # a rejected line keeps its (here NaN) measurement and is flagged in the mask
    assert read.get_lines()[4][1, 2] and not read.get_lines()[4][0, 2]


def test_lvmarc_without_traces(tmp_path, make_arc):
    """ an lvmArc whose individual fit failed (no traces) can be read back """
    lvmarc = make_lvmarc(make_arc.slitmap(), wave_trace=None)
    path = tmp_path / "lvmArc-b1-00000007.fits"
    lvmarc.writeFitsData(str(path))
    read = lvmArc.from_file(str(path))
    assert read._wave_trace is None and read._lsf_trace is None
    assert read.get_lines()[0].size == 3


def test_lvmarc_legacy_extensions(tmp_path, make_arc):
    """ files written with the LXPEAK/LXFWHM/LXFLUX/LXMASK names are still read """
    lvmarc = make_lvmarc(make_arc.slitmap())
    path = tmp_path / "lvmArc-b1-00000007.fits"
    lvmarc.writeFitsData(str(path))
    with fits.open(path) as hdus:
        for new, old in (("LINE_CENT", "LXPEAK"), ("LINE_FWHM", "LXFWHM"), ("LINE_FLUX", "LXFLUX"), ("LINE_MASK", "LXMASK")):
            hdus[new].name = old
        del hdus["LINE_BKG"]
        hdus.writeto(tmp_path / "legacy.fits")

    read = lvmArc.from_file(str(tmp_path / "legacy.fits"))
    np.testing.assert_array_equal(read.get_lines()[1], lvmarc.get_lines()[1])
    # without a stored background, the background table is missing (NaN)
    assert np.isnan(read.get_line_table("LINE_BKG")).all()


def test_frames_table_roundtrip(tmp_path):
    """ the exposures combined in a master trace are listed in its FRAMES extension """
    headers = [fits.Header({"TILEID": 11111, "MJD": 61000 + k, "EXPOSURE": 100 + k, "SPEC": "sp1", "CCD": "b1", "EXPTIME": 10.0})
               for k in range(3)]
    frames = individual_frames_table(headers)
    assert list(frames["EXPNUM"]) == [100, 101, 102] and list(frames["MJD"]) == [61000, 61001, 61002]

    trace = TraceMask(data=np.zeros((4, 20), np.float32), mask=np.zeros((4, 20), bool), coeffs=np.zeros((4, 3)),
                      header=fits.Header({"CCD": "b1"}))
    trace.setIndividualFrames(headers)
    path = tmp_path / "lvm-mwave-b1.fits"
    trace.writeFitsData(str(path))
    read = TraceMask.from_file(str(path)).getIndividualFrames()
    assert list(read["EXPNUM"]) == [100, 101, 102]
    assert [str(camera).strip() for camera in read["CAMERA"]] == ["b1"] * 3


def test_frames_table_uses_sjd():
    """ the MJD column holds the SJD (SMJD) of the directory paths, not the MJD after the rollover """
    frames = individual_frames_table([fits.Header({"MJD": 60999, "SMJD": 61000, "EXPOSURE": 5}),
                                      fits.Header({"MJD": 61000, "EXPOSURE": 6})])
    assert list(frames["MJD"]) == [61000, 61000]


def test_frames_table_missing_values():
    """ missing header values get placeholders instead of failing """
    frames = individual_frames_table([fits.Header({"EXPOSURE": 5})])
    assert frames["TILEID"][0] == 11111 and frames["MJD"][0] == -1 and np.isnan(frames["EXPTIME"][0])


def test_combine_rss_normalize():
    """ RSS frames at different levels are scaled to a common level before combining """
    data = np.random.default_rng(1).uniform(1, 2, (4, 30)).astype(np.float32)
    header = fits.Header({"IMAGETYP": "arc"})
    rsss = [RSS(data=data * k, error=np.sqrt(data * k), mask=np.zeros(data.shape, bool), header=header.copy()) for k in (1, 5)]
    combined = RSS()
    combined.combineRSS(rsss, method="median", normalize=True, normalize_percentile=99)
    np.testing.assert_allclose(combined._data, 3 * data, rtol=1e-5)
