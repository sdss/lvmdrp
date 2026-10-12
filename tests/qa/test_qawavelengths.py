# encoding: utf-8
#
# tests of the wavelength and LSF QA dashboard built from lvmArc products

import os
import re
import shutil

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from scipy.special import erf

import lvmdrp.functions.rssMethod as rm
import lvmdrp.qa.wavelengths as qw
from lvmdrp.core.rss import lvmArc


ARCS = [{"tileid": 11111, "mjd": 61000, "expnum": expnum, "camera": "b1"} for expnum in (1, 2)]


@pytest.fixture(scope="module")
def masters(measured_arcs, tmp_path_factory):
    """ master solutions of the synthetic arcs, as the pipeline writes them """
    path = tmp_path_factory.mktemp("calib")
    out = {"wave": str(path / "lvm-mwave-b1.fits"), "lsf": str(path / "lvm-mlsf-b1.fits")}
    rm.fit_wavelength_solution(in_arcs=measured_arcs.larcs, out_wave=out["wave"], out_lsf=out["lsf"], ref_fiber=30,
                               poly_disp=3, poly_fwhm=2)
    return out


@pytest.fixture()
def products(monkeypatch, measured_arcs):
    """ the pipeline paths of the synthetic arcs, and the line list of the synthetic arcs for anything measured """
    by_expnum = {arc["expnum"]: {"lvmarc": larc, "xarc": xarc}
                 for arc, larc, xarc in zip(ARCS, measured_arcs.larcs, measured_arcs.xarcs)}
    monkeypatch.setattr(qw, "arc_products", lambda arc: dict(by_expnum[arc["expnum"]]))
    measure, determine, kws = rm.measure_arc_lines, rm.determine_wavelength_solution, measured_arcs.kws
    monkeypatch.setattr(rm, "measure_arc_lines", lambda in_arc, out_arc, **k: measure(in_arc=in_arc, out_arc=out_arc, **{**kws, **k}))
    monkeypatch.setattr(rm, "determine_wavelength_solution",
                        lambda in_arcs, out_wave, out_lsf, **k: determine(in_arcs, out_wave, out_lsf, **{**kws, **k}))
    return by_expnum


def sections(html):
    return re.findall(r"<h2>([^<]+)</h2>", html)


def test_profile_residual_exact_model(make_arc):
    """ the profile residuals vanish for data equal to the fitted model """
    nfibers, npix = 10, 200
    x = np.arange(npix, dtype=float)
    cent = np.array([[50.3, 120.77]] * nfibers) + np.linspace(0, 1, nfibers)[:, None]
    fwhm, flux, bkg = np.full(cent.shape, 2.6), np.full(cent.shape, 5000.0), np.full(cent.shape, 12.0)
    sigma = fwhm / 2.3548
    data = 12.0 + sum(0.5 * flux[:, j, None] * (erf((x + 0.5 - cent[:, j, None]) / (np.sqrt(2) * sigma[:, j, None]))
                                               - erf((x - 0.5 - cent[:, j, None]) / (np.sqrt(2) * sigma[:, j, None])))
                      for j in range(2))
    arc = lvmArc(data=data.astype(np.float32), error=np.ones_like(data, dtype=np.float32), mask=np.zeros(data.shape, bool),
                 header=fits.Header({"CCD": "b1"}), ref_wave=[5000.0, 5100.0], cent_line=cent, fwhm_line=fwhm, flux_line=flux,
                 mask_line=np.zeros(cent.shape, bool), bkg_line=bkg, slitmap=make_arc.slitmap()[:nfibers])
    fibers, lines, dx, residual = qw._profile_points(arc, {5000.0: 0, 5100.0: 1})
    assert 0 < residual.size <= nfibers * 2 * 13
    assert np.abs(residual).max() < 1e-4
    assert np.abs(dx).max() <= 6


def test_asinh_ticks():
    """ ticks of the arcsinh axis include zero, are sorted and spread out """
    tickvals, ticktext = qw._asinh_ticks(5.0, -20.0, 8000.0)
    assert "0" in ticktext and "5k" in ticktext
    assert np.all(np.diff(tickvals) >= 0.8 - 1e-9)


def test_spread_fibers():
    """ the default fibers are spread along the slit, skipping standard and bad fibers """
    fibers = pd.DataFrame({"fiber": np.arange(100), "good": np.arange(100) != 50, "telescope": ["Sci"] * 100,
                           "wave_scatter": np.zeros(100)})
    fibers.loc[[0, 99], "telescope"] = "Spec"
    chosen = qw._spread_fibers(fibers)
    assert len(chosen) == 4 and chosen == sorted(chosen)
    assert not {0, 50, 99} & set(chosen)


def test_master_arcs(masters):
    """ the exposures of a master solution come from its FRAMES extension """
    arcs = qw.master_arcs(masters["wave"])
    assert [(arc["expnum"], arc["camera"]) for arc in arcs] == [(1, "b1"), (2, "b1")]
    # paths of the individual arcs are built with the night's SJD
    assert [arc["mjd"] for arc in arcs] == [61000, 61000]
    assert "/61000/" in qw.arc_products(arcs[0])["lvmarc"]
    assert qw.master_arcs(masters["wave"] + ".missing") is None


def test_dashboard_with_sum(products, masters, tmp_path, make_arc):
    """ the dashboard is built from the pipeline products, measuring only the summed arcs """
    paths = qw.compare_camera("b1", ARCS, masters, str(tmp_path), with_sum=True)
    assert paths["reused"] == 2 and paths["measured"] == 0 and paths["measurements"]["master"] == "pipeline"
    assert paths["pixels"] is not None and os.path.isfile(paths["pixels"]["arcs"][0])

    metrics = qw.camera_metrics(paths, fibers=[10, 40])
    assert metrics["spectrum_fibers"] == {"ref": 30, "1": 10, "2": 40}
    report = str(tmp_path / "wave-qa.html")
    summary = qw.write_report([metrics], report, 61000, "longterm", [61000])
    html = open(report).read()

    assert sections(html) == ["Summary by camera", "Arc spectra and fitted lines", "Line profile residuals", "Polynomial fits",
                              "Fit residuals across the slit", "Polynomial coefficients", "Combining exposures",
                              "How the quantities are computed"]
    assert summary.products[0] == "2 from the pipeline, 0 measured here; master pipeline"
    # the sum of exposures is broader, by the one-pixel shift between them
    assert summary.bias_pixels[0] > summary.bias_measurements[0]
    assert 0.5 < summary.explained_ratio[0] < 1.5
    for key in ("spectrum|b1|1|ref", "spectrum|b1|2|2", "spectrum|b1|sum|1", "fig-explained", "coeffs|b1|wave|pixels"):
        assert f'"{key}"' in html
    assert '<option value="ref">Reference fiber</option>' in html and "Fiber 40" in html


def test_exposure_trends(products, masters, tmp_path):
    """ line widths of each exposure relative to all exposures, and their trend with time and temperature """
    metrics = qw.camera_metrics(qw.compare_camera("b1", ARCS, masters, str(tmp_path)))
    dfwhm, good = metrics["dfwhm"], metrics["good"]
    assert dfwhm.shape == (2, good.size)
    # both synthetic exposures have the same line widths
    assert np.isfinite(dfwhm[:, good]).mean() > 0.9
    assert np.abs(np.nanmedian(dfwhm[:, good])) < 0.05
    assert list(metrics["exposure_info"].expnum) == ["1", "2"]
    # line by line flux and width changes, one row per exposure
    assert metrics["line_dflux"].shape == metrics["line_dfwhm"].shape == (2, metrics["ref_lines"].size)
    assert np.isfinite(metrics["line_dflux"]).mean() > 0.9 and np.isfinite(metrics["exposure_dflux"]).all()
    report = str(tmp_path / "wave-qa.html")
    qw.write_report([metrics], report, 61000, "longterm", [61000])
    html = open(report).read()
    nights = sorted(set(metrics["exposure_info"].night))
    keys = ["fwhms|b1", "fluxwidths|b1"] + [f"trends|{quantity}|{night}" for quantity in ("fwhm", "flux", "offset") for night in nights + ["all"]]
    for key in keys:
        assert f'"{key}"' in html


def test_dashboard_without_sum(products, masters, tmp_path):
    """ without the sum, the dashboard measures nothing and shows only the per-arc solutions """
    paths = qw.compare_camera("b1", ARCS, masters, str(tmp_path))
    assert paths["pixels"] is None and not os.path.isdir(tmp_path / "pixels")
    metrics = qw.camera_metrics(paths)
    assert list(metrics["spectrum_fibers"]) == ["ref", "1", "2", "3", "4"]
    report = str(tmp_path / "wave-qa.html")
    qw.write_report([metrics], report, 61000, "longterm", [61000])
    html = open(report).read()
    assert '"fig-explained"' not in html and "Sum of exposures" not in html and "|pixels" not in html
    assert "% along the slit" in html


def test_failed_sum_keeps_camera(products, masters, tmp_path, monkeypatch):
    """ a camera whose summed arcs cannot be measured keeps its own solutions """
    def fail(*args, **kwargs):
        raise ValueError("forced failure")
    monkeypatch.setattr(rm, "determine_wavelength_solution", fail)
    paths = qw.compare_camera("b1", ARCS, masters, str(tmp_path), with_sum=True)
    assert paths["pixels"] is None and "forced failure" in paths["sum_problem"]
    summary = qw.write_report([qw.camera_metrics(paths)], str(tmp_path / "wave-qa.html"), 61000, "longterm", [61000])
    assert "summed arcs skipped" in summary.products[0]


def test_missing_lvmarc_is_measured(products, masters, tmp_path):
    """ an arc without lvmArc is measured into the dashboard directory, the others are reused """
    missing = tmp_path / "pipeline" / "lvmArc-b1-00000002.fits"
    products[2]["lvmarc"] = str(missing)
    paths = qw.compare_camera("b1", ARCS, masters, str(tmp_path / "qa"))
    assert paths["reused"] == 1 and paths["measured"] == 1 and not missing.exists()
    local = tmp_path / "qa" / "measurements" / "lvmArc-b1-00000002.fits"
    assert str(local) in paths["measurements"]["arcs"] and local.exists()
    # measured again with the same parameters, the lines agree with the pipeline product
    np.testing.assert_allclose(lvmArc.from_file(str(local)).get_lines()[1][:30],
                               lvmArc.from_file(products[1]["lvmarc"].replace("00000001", "00000002")).get_lines()[1][:30], atol=1e-4)
    # a second run reuses what this dashboard measured
    again = qw.compare_camera("b1", ARCS, masters, str(tmp_path / "qa"))
    assert again["previous"] == 1 and again["measured"] == 0
    shutil.rmtree(tmp_path / "qa")
