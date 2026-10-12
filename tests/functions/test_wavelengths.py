# encoding: utf-8
#
# tests of the wavelength and LSF calibration from arc lines measured in each exposure

import numpy as np
from astropy.io import fits

import lvmdrp.functions.rssMethod as rm
from lvmdrp.core.rss import lvmArc, _read_pixwav_map
from lvmdrp.core.tracemask import TraceMask


def test_combine_arc_lines_removes_offsets():
    """ shifts between exposures are removed before combining, keeping the mean zero point """
    rng = np.random.default_rng(0)
    narcs, nfibers, nlines = 3, 6, 10
    truth = np.linspace(50, 950, nlines)[None].repeat(nfibers, 0)
    shifts = np.array([0.0, 0.4, -0.2])
    cent = truth[None] + shifts[:, None, None] + rng.normal(0, 0.005, (narcs, nfibers, nlines))
    masked = np.zeros(cent.shape, dtype=bool)
    masked[1, :, :4] = True          # lines missing in the shifted exposure
    masked[:, 5] = True              # a dead fiber
    flux, fwhm = np.full(cent.shape, 1e4), np.full(cent.shape, 2.5)

    _, cent_c, fwhm_c, masked_c, offsets = rm._combine_arc_lines(flux, cent, fwhm, masked)

    np.testing.assert_allclose(cent_c[:5], truth[:5] + shifts.mean(), atol=0.02)
    np.testing.assert_allclose(offsets[:, :5] - offsets[:, :5].mean(axis=0), (shifts - shifts.mean())[:, None].repeat(5, 1), atol=0.02)
    assert masked_c[5].all() and not masked_c[:5].any()
    np.testing.assert_allclose(fwhm_c[:5], 2.5)


def test_exposed_std_from_header(make_arc):
    """ the exposed standard fiber is taken from CALIBFIB """
    arc = make_arc(lit_std=2)
    exposed, nonexposed = rm._get_exposed_std_rss(arc, return_nonexposed=True)
    assert list(exposed) == [make_arc.std[2]]
    assert sorted(nonexposed) == sorted(set(make_arc.std) - {make_arc.std[2]})


def test_exposed_std_from_data(make_arc):
    """ without CALIBFIB, each standard fiber is tested on its own (not all at once) """
    arc = make_arc(lit_std=1)
    del arc._header["CALIBFIB"]
    exposed, nonexposed = rm._get_exposed_std_rss(arc, return_nonexposed=True)
    assert list(exposed) == [make_arc.std[1]]
    assert len(nonexposed) == len(make_arc.std) - 1


def test_mask_unlit_std_fibers(make_arc):
    """ unlit standard fibers are masked so they don't bias combinations """
    arc = rm._mask_unlit_std_fibers(make_arc(lit_std=0))
    for k, fiber in enumerate(make_arc.std):
        assert arc._mask[fiber].all() == (k != 0)
        assert np.isnan(arc._data[fiber]).all() == (k != 0)


def test_select_lamps_lines():
    """ given lines are kept for the lamps that were on, and lines in no lamp map are always kept """
    camera = "z1"
    _, _, _, argon, _ = _read_pixwav_map("argon", camera)
    _, _, _, neon, _ = _read_pixwav_map("neon", camera)
    neon_only = np.setdiff1d(np.round(neon, 3), np.round(argon, 3))
    neon_only = neon_only[~np.isclose(neon_only[:, None], np.concatenate([_read_pixwav_map(lamp, camera)[3]
                                      for lamp in ("hgne", "xenon")])[None], atol=0.1).any(axis=1)][:3]
    unknown = np.array([20001.0])
    waves = np.concatenate([argon[:3], neon_only, unknown])
    pixels = np.arange(waves.size, dtype=float)

    _, selected, _ = rm._select_lamps_lines(pixels, waves, np.ones(waves.size, bool), {"argon"}, camera)

    assert set(np.round(argon[:3], 3)) <= set(np.round(selected, 3))
    assert not set(np.round(neon_only, 3)) & set(np.round(selected, 3))
    assert 20001.0 in selected


def test_measure_arc_lines_product(measured_arcs, make_arc):
    """ each exposure gets an lvmArc with its line measurements, backgrounds and fitting parameters """
    lvmarc = lvmArc.from_file(measured_arcs.larcs[1])
    waves, cent, fwhm, flux, mask = lvmarc.get_lines()
    np.testing.assert_allclose(waves, make_arc.waves, atol=1e-3)
    sci = [fiber for fiber in range(make_arc.nfibers) if fiber not in make_arc.std]
    assert not mask[sci].any()
    np.testing.assert_allclose(np.median(cent[sci], axis=0), make_arc.pixels + 1.0, atol=0.05)
    np.testing.assert_allclose(np.median(fwhm[sci]), make_arc.fwhm, rtol=0.03)
    # the lit standard fiber is measured, the others are rejected
    assert not mask[make_arc.std[1]].any() and mask[make_arc.std[0]].all()
    assert np.isfinite(lvmarc.get_line_table("LINE_BKG")[sci]).all()
    header = lvmarc._header
    assert header["b1 WAVE APERTURE"] == 12 and header["b1 WAVE REFFIB"] == 30
    assert lvmarc._wave_trace is not None


def test_fit_wavelength_solution(measured_arcs, make_arc, tmp_path):
    """ the master solution combines the lines of both exposures without the broadening of their shift """
    xarcs, larcs, kws = measured_arcs.xarcs, measured_arcs.larcs, measured_arcs.kws
    out_wave, out_lsf = str(tmp_path / "lvm-mwave-b1.fits"), str(tmp_path / "lvm-mlsf-b1.fits")
    rm.fit_wavelength_solution(in_arcs=larcs, out_wave=out_wave, out_lsf=out_lsf, ref_fiber=30,
                               poly_disp=kws["poly_disp"], poly_fwhm=kws["poly_fwhm"])

    wave, lsf = TraceMask.from_file(out_wave), TraceMask.from_file(out_lsf)
    header = fits.getheader(out_wave)
    assert header["NFRAMES"] == 2 and "b1 WAVE DISP RMS MEDIAN" in header
    assert not any(key.startswith("b1 WAVE CC") or key.startswith("PIPE") for key in header)
    assert list(wave.getIndividualFrames()["EXPNUM"]) == [1, 2]
    # the night's SJD, not the MJD of exposures taken before the rollover
    assert list(wave.getIndividualFrames()["MJD"]) == [61000, 61000]

    sci = [fiber for fiber in range(make_arc.nfibers) if fiber not in make_arc.std]
    pix = np.arange(make_arc.npix)
    # zero point at the mean of the exposures (half a pixel); the synthetic arcs have wave = 20000 + 0.5 x + 2e-5 x^2
    x = pix - 0.5
    true_wave = 20000.0 + 0.5 * x + 2e-5 * x**2
    np.testing.assert_allclose(np.median(wave._data[sci], axis=0)[40:-40], true_wave[40:-40], atol=0.01)
    true_lsf = make_arc.fwhm * (0.5 + 4e-5 * x)
    np.testing.assert_allclose(np.median(lsf._data[sci], axis=0)[40:-40], true_lsf[40:-40], rtol=0.03)

    # the arcs summed in pixels give a broader LSF, from the shift between them
    sum_wave, sum_lsf = str(tmp_path / "sum-mwave.fits"), str(tmp_path / "sum-mlsf.fits")
    rm.determine_wavelength_solution(in_arcs=xarcs, out_wave=sum_wave, out_lsf=sum_lsf, arcs_combination="pixels", **kws)
    broadening = np.median(TraceMask.from_file(sum_lsf)._data[sci]) / np.median(lsf._data[sci]) - 1
    assert broadening > 0.03
