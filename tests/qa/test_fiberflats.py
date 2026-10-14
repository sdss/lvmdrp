# encoding: utf-8
#

import collections
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.special import erf

from lvmdrp.qa.fiberflats import (_pixel_window_weights, measure_skylines,
                                 measure_continuum, summarize_flatfield_qa,
                                 measure_flatfield_qa, find_frames, fit_gaussian_floor,
                                 aggregate_flatfield_qa, write_flatfield_qa_report,
                                 report_flatfield_qa)

import lvmdrp.qa.fiberflats as ff


LINES = [6300.30, 6363.78, 7243.36]
WINDOWS = [(6005, 6025), (7015, 7035)]


def make_frame(flat_residual, nfibers=300, npix=4000, noise=True, seed=1, sky_gradient=None):
    """creates a fake flat fielded r-channel frame with sky lines and continuum"""
    rng = np.random.default_rng(seed)
    spec = np.repeat([1, 2, 3], nfibers // 3)
    telescope = np.array(["Sci", "Sci", "SkyE", "SkyW"])[np.arange(nfibers) % 4]
    # science IFU split in three spectrograph wedges, small sky IFUs with fibers from all spectrographs
    angle = np.radians(120 * (spec - 1)) + rng.uniform(0, np.radians(120), nfibers)
    radius = np.sqrt(rng.uniform(0, 1, nfibers)) * np.where(telescope == "Sci", 8.0, 1.3)
    disp = 0.5 + rng.normal(0, 0.002, nfibers)
    wave = 5650 + rng.normal(0, 0.3, nfibers)[:, None] + disp[:, None] * np.arange(npix)[None, :]

    sigma = 1.6 / 2.355
    sky = 500.0 + 0.05 * (wave - 6000)
    for cwave in LINES:
        sky = sky + 30000 * disp[:, None] * np.exp(-0.5 * ((wave - cwave) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

    data = sky * flat_residual(wave, spec)
    if sky_gradient is not None:
        # multiplicative sky gradient across the science IFU, (gx, gy) at the IFU edge
        sci = telescope == "Sci"
        data[sci] *= (1 + sky_gradient[0] * radius[sci] / 8.0 * np.cos(angle[sci])
                      + sky_gradient[1] * radius[sci] / 8.0 * np.sin(angle[sci]))[:, None]
    error = np.sqrt(data + 9)
    if noise:
        data = data + rng.normal(0, 1, data.shape) * error

    slitmap = Table({"fiberid": np.arange(1, nfibers + 1), "spectrographid": spec,
                     "telescope": telescope, "fibstatus": np.zeros(nfibers, dtype=int),
                     "xpmm": radius * np.cos(angle), "ypmm": radius * np.sin(angle)})
    return SimpleNamespace(_data=data, _error=error, _mask=np.zeros_like(data, dtype=bool), _wave=wave,
                           _slitmap=slitmap, _header={"CCD": "r1", "EXPOSURE": 1})


def test_pixel_window_weights():
    wave = np.arange(10, dtype=float)[None, :]
    weights = _pixel_window_weights(wave, 2.25, 5.0)
    assert weights.sum() == pytest.approx(2.75)
    assert weights[0, 2] == pytest.approx(0.25)
    assert weights[0, 3] == pytest.approx(1.0)
    assert weights[0, 5] == pytest.approx(0.5)


def test_perfect_flatfield():
    frame = make_frame(lambda wave, spec: np.ones_like(wave), noise=False)
    lines = measure_skylines(frame, LINES)
    cont = measure_continuum(frame, WINDOWS)
    for table in (lines, cont):
        summary = summarize_flatfield_qa(frame, table)
        assert np.all(summary["scatter"] < 1e-3)
        assert np.all(np.abs(summary["offset_sp1"]) < 1e-3)

    # recovered line fluxes are close to the injected ones
    for cwave in LINES:
        assert np.nanmedian(lines[f"L{cwave:.2f}_flux"]) == pytest.approx(30000, rel=0.02)


def test_spectrograph_offset():
    frame = make_frame(lambda wave, spec: 1 + 0.02 * (spec == 2)[:, None] * np.ones_like(wave))
    _, lines_summary, _, cont_summary = measure_flatfield_qa(frame, skylines=LINES, cont_windows=WINDOWS, plot=False)
    for summary in (lines_summary, cont_summary):
        assert np.all(summary["reliable"])
        offsets = summary["offset_sp2"] - summary["offset_sp1"]
        assert np.allclose(offsets, 0.02, atol=0.006)


def test_chromatic_residual():
    rng = np.random.default_rng(2)
    tilt = rng.normal(0, 0.05, 300)
    frame = make_frame(lambda wave, spec: 1 + tilt[:, None] * (wave - 5650) / 2000)
    _, lines_summary, _, cont_summary = measure_flatfield_qa(frame, skylines=LINES, cont_windows=WINDOWS, plot=False)
    for summary in (lines_summary, cont_summary):
        # excess scatter follows the injected wavelength dependent residual
        expected = np.std(tilt) * (summary["wave"] - 5650) / 2000
        assert np.allclose(summary["excess"], expected, atol=0.005)


def test_masked_and_bad_fibers():
    frame = make_frame(lambda wave, spec: np.ones_like(wave), noise=False)
    frame._slitmap["fibstatus"][0] = 1
    frame._slitmap["telescope"][1] = "Spec"
    sel = np.abs(frame._wave[2] - LINES[0]) < 1
    frame._mask[2, sel] = True
    lines = measure_skylines(frame, LINES)
    cont = measure_continuum(frame, WINDOWS)
    name = f"L{LINES[0]:.2f}"
    # bad and non-science fibers are not measured, partially masked line windows are not good fits
    assert np.isnan(lines[f"{name}_flux"][[0, 1]]).all()
    assert not lines[f"{name}_good"][2] and np.isnan(lines[f"{name}_norm"][2])
    assert lines[f"L{LINES[1]:.2f}_good"][2]
    assert np.isnan(cont[f"C{WINDOWS[0][0]}-{WINDOWS[0][1]}_flux"][[0, 1]]).all()

    lines = measure_skylines(frame, LINES, method="integrate")
    assert np.isnan(lines[f"{name}_flux"][[0, 1, 2]]).all()
    assert np.isfinite(lines[f"L{LINES[1]:.2f}_flux"][2])


def test_low_snr_is_flagged():
    frame = make_frame(lambda wave, spec: np.ones_like(wave))
    frame._error = frame._error * 100
    summary = summarize_flatfield_qa(frame, measure_skylines(frame, LINES), min_snr=10)
    assert not np.any(summary["reliable"])


def make_batch_summary(nframes=6):
    """creates a fake per frame summary as produced by `run_flatfield_qa_batch`"""
    frame = make_frame(lambda wave, spec: np.ones_like(wave))
    _, lines_summary, _, cont_summary = measure_flatfield_qa(frame, skylines=LINES, cont_windows=WINDOWS, plot=False, verbose=False)
    one = vstack([lines_summary, cont_summary])
    summaries = []
    for i in range(nframes):
        summary = one.copy()
        summary["excess"] = 0.01 * (i + 1)
        summary["reliable"][0] = i > 0
        for column, value in [("filename", f"lvmFrame-r-{i:08d}.fits"), ("expnum", i), ("mjd", 61000 + i // 2),
                              ("tileid", 1028000), ("channel", "r"), ("drpver", "1.3.2")]:
            summary.add_column(value, name=column, index=0)
        summaries.append(summary)
    return vstack(summaries)


def test_find_frames(tmp_path):
    for tileid, mjd, channel, expnum in [(1028000, 61300, "b", 1), (1028000, 61300, "r", 1),
                                         (1028001, 61301, "b", 2), (11111, 61302, "z", 3)]:
        path = tmp_path / "1.3.2" / f"{str(tileid)[:-2]}XX" / str(tileid) / str(mjd) / f"lvmFrame-{channel}-{expnum:08d}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    paths = find_frames("1.3.2", redux_dir=str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["lvmFrame-b-00000001.fits", "lvmFrame-r-00000001.fits",
                                                    "lvmFrame-b-00000002.fits", "lvmFrame-z-00000003.fits"]
    assert len(find_frames("1.3.2", channels="b", redux_dir=str(tmp_path))) == 2
    assert len(find_frames("1.3.2", mjd_range=(61301, 61302), redux_dir=str(tmp_path))) == 2
    assert len(find_frames("1.3.2", tileids=[11111], redux_dir=str(tmp_path))) == 1


def test_aggregate_and_plots():
    summary = make_batch_summary()
    aggregate = aggregate_flatfield_qa(summary)
    assert len(aggregate) == len(LINES) + len(WINDOWS)
    first = aggregate[aggregate["name"] == f"L{LINES[0]:.2f}"][0]
    # the first feature is unreliable in the first frame
    assert first["nframes"] == 6 and first["nreliable"] == 5
    assert first["excess_median"] == pytest.approx(0.04)
    assert aggregate[aggregate["name"] == f"L{LINES[1]:.2f}"][0]["excess_median"] == pytest.approx(0.035)



def test_report(tmp_path):
    summary = make_batch_summary()
    aggregate = aggregate_flatfield_qa(summary)
    out_html = write_flatfield_qa_report(summary, aggregate, str(tmp_path / "qa.html"), run_info={"nskipped": 2, "nfailed": 1})
    html = open(out_html, encoding="utf-8").read()
    assert html.startswith('<meta charset="utf-8">\n<title>LVM Flat-Field QA</title>')
    # pure ASCII so that it renders correctly regardless of the declared encoding
    assert html.isascii()
    assert "cdn.jsdelivr.net/npm/mathjax@" in html and 'id="definitions"' in html
    assert r"e = \sqrt{\max\left(s^2 - n^2,\ 0\right)}" in html
    for div in ("fig-excess", "fig-offsets", "fig-outliers", "fig-timeline", "fig-offsets-time", "fig-gradient-time"):
        assert f'id="{div}"' in html
    assert "cdn.jsdelivr.net/npm/plotly.js-dist-min@" in html
    assert "lvmFrame-r-00000005.fits" in html

    # regenerate the report from a written table
    table_path = str(tmp_path / "qa.fits")
    summary.write(table_path, format="fits")
    hdus = fits.open(table_path)
    hdus[1].name = "FRAMES"
    hdus.writeto(table_path, overwrite=True)
    assert os.path.isfile(report_flatfield_qa(table_path))


def make_frame_gradients(persistent, slope_per_year=(0.0, 0.0), nnights=150, sky=0.03, seed=3):
    """Per frame science IFU gradients: a persistent gradient changing linearly with time plus random sky gradients"""
    rng = np.random.default_rng(seed)
    rows = []
    for channel in "brz":
        for night in range(nnights):
            mjd = 60300 + 4 * night
            years = (mjd - 60300) / 365.25
            gx, gy = (p + s * years for p, s in zip(persistent, slope_per_year))
            for frame in range(rng.integers(1, 5)):
                angle, size = rng.uniform(0, 2 * np.pi), abs(rng.normal(0, sky))
                rows.append({"filename": f"{channel}-{night}-{frame}", "channel": channel, "mjd": mjd, "expnum": 10 * night + frame,
                             "nfeatures": 5, "grad_x": gx + size * np.cos(angle), "grad_y": gy + size * np.sin(angle), "grad_amp": 0.01})
    return Table(rows=rows)


def test_persistent_gradient_and_trend():
    nightly = ff.nightly_gradients(make_frame_gradients(persistent=(0.02, -0.01), slope_per_year=(0.01, 0.0)))
    assert set(nightly["channel"]) == set("brz") and len(nightly) == 3 * 150
    for row in ff.gradient_trends(nightly):
        # the persistent x component is the median over the run, at its middle
        assert row["grad_x"] == pytest.approx(0.02 + 0.01 * (4 * 149 / 2) / 365.25, abs=0.004)
        assert row["grad_y"] == pytest.approx(-0.01, abs=0.004)
        assert row["significance"] > 10 and row["consistency"] > 0.7 and row["rayleigh_p"] < 1e-10
        # the slopes scatter by ~0.2% per year with these numbers of nights and sky gradients
        assert row["trend_grad_x"] == pytest.approx(0.01, abs=0.006) and row["trend_grad_x_lo"] > 0
        assert row["trend_grad_y"] == pytest.approx(0.0, abs=0.006)


def test_sky_gradients_are_not_persistent():
    nightly = ff.nightly_gradients(make_frame_gradients(persistent=(0.0, 0.0)))
    for row in ff.gradient_trends(nightly):
        assert row["significance"] < 3 and row["consistency"] < 0.3 and row["rayleigh_p"] > 1e-3
        for column in ("grad_x", "grad_y"):
            assert row[f"trend_{column}"] == pytest.approx(0.0, abs=0.006)
    assert len(ff.figure_gradient_timeline(nightly, ff.gradient_trends(nightly)).data) >= 6


def test_zero_gradients_have_no_direction():
    gradients = make_frame_gradients(persistent=(0.0, 0.0), sky=0.0, nnights=5)
    row = ff.gradient_trends(ff.nightly_gradients(gradients))[0]
    assert row["strength"] == 0 and np.isnan(row["consistency"]) and np.isnan(row["rayleigh_p"])


def test_gradient_trends_without_data():
    nightly = ff.nightly_gradients(Table(names=["filename", "channel", "mjd", "expnum", "nfeatures", "grad_x", "grad_y", "grad_amp"]))
    assert len(nightly) == 0 and ff.gradient_trends(nightly) == []
    ff.figure_gradient_timeline(nightly, [])


def test_fit_gaussian_floor():
    rng = np.random.default_rng(3)
    nfibers, npix, cwave, total = 500, 200, 6300.0, 20000.0
    disp = 0.5 + rng.normal(0, 0.003, nfibers)
    wave = 6250 + rng.normal(0, 0.3, nfibers)[:, None] + disp[:, None] * np.arange(npix)[None, :]
    sigma = 1.8 / 2.355 * (1 + rng.normal(0, 0.1, nfibers))
    center = cwave + rng.normal(0, 0.3, nfibers)

    mid = 0.5 * (wave[:, 1:] + wave[:, :-1])
    left = np.concatenate([wave[:, :1] - (mid[:, :1] - wave[:, :1]), mid], axis=1)
    right = np.concatenate([mid, wave[:, -1:] + (wave[:, -1:] - mid[:, -1:])], axis=1)
    cdf = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))  # noqa: E731
    truth = total * (cdf((right - center[:, None]) / sigma[:, None]) - cdf((left - center[:, None]) / sigma[:, None]))
    truth += 400 + 2.0 * (wave - cwave)
    error = np.sqrt(truth + 16)
    data = truth + rng.normal(0, 1, truth.shape) * error

    # start from a 10% off LSF guess
    result = fit_gaussian_floor(wave, data, error, np.zeros_like(data, dtype=bool), cwave, sigma_guess=1.6 / 2.355)
    assert result["converged"].all()
    assert np.median(result["flux"]) == pytest.approx(total, rel=0.003)
    assert np.median(result["cont"]) == pytest.approx(400, rel=0.003)
    assert np.median(result["sigma"] / sigma) == pytest.approx(1, abs=0.005)
    pulls = (result["flux"] - total) / result["flux_error"]
    assert np.std(pulls) == pytest.approx(1, abs=0.1)


def test_line_too_weak_over_continuum():
    frame = make_frame(lambda wave, spec: np.ones_like(wave), noise=False)
    # bright continuum (e.g., twilight) under the same sky lines
    frame._data = frame._data + 50000.0
    frame._error = np.sqrt(frame._data + 9)
    summary = summarize_flatfield_qa(frame, measure_skylines(frame, LINES), min_snr=10, min_contrast=1.0)
    assert not np.any(summary["reliable"])
    assert all("line/continuum" in reason for reason in summary["reason"])


def test_sky_gradient():
    frame = make_frame(lambda wave, spec: np.ones_like(wave), nfibers=600, sky_gradient=(0.04, -0.03))
    kwargs = dict(skylines=LINES[:2], cont_windows=WINDOWS[:1], plot=False, verbose=False)

    # without the correction the sky gradient looks like flat field errors
    _, lines_summary, _, _ = measure_flatfield_qa(frame, gradient_deg=None, **kwargs)
    assert np.all(lines_summary["excess"] > 0.01)
    assert np.all(np.nanmax(np.abs([lines_summary[f"offset_sp{i}"] for i in (1, 2, 3)]), axis=0) > 0.005)
    assert "gcoeff1_Sci" not in lines_summary.colnames

    _, lines_summary, _, cont_summary = measure_flatfield_qa(frame, gradient_deg=1, **kwargs)
    for summary in (lines_summary, cont_summary):
        assert np.all(summary["excess"] < 0.004)
        assert np.all(np.abs([summary[f"offset_sp{i}"] for i in (1, 2, 3)]) < 0.003)
        assert np.allclose(summary["grad_x_Sci"], 0.04, atol=0.004)
        assert np.allclose(summary["grad_y_Sci"], -0.03, atol=0.004)
        # pipeline coefficients over the mean gradient, in the GCOEFF order (1, y, x, xy), per mm
        assert np.allclose(summary["gcoeff1_Sci"], 1.0, atol=0.01)
        assert np.allclose(summary["gcoeff2_Sci"], -0.03 / 8, atol=5e-4)
        assert np.allclose(summary["gcoeff3_Sci"], 0.04 / 8, atol=5e-4)
        assert np.all(summary["gcoeff4_Sci"] == 0)


def test_gradient_does_not_absorb_flat_errors():
    offsets = np.array([0.0, 0.02, -0.01])
    frame = make_frame(lambda wave, spec: 1 + offsets[spec - 1][:, None] * np.ones_like(wave), nfibers=600)
    _, lines_summary, _, _ = measure_flatfield_qa(frame, skylines=LINES[:2], cont_windows=WINDOWS[:1], plot=False, verbose=False)
    expected = offsets - offsets.mean()
    for row in lines_summary:
        assert np.allclose([row[f"offset_sp{i}"] for i in (1, 2, 3)], expected, atol=0.003)
        assert abs(row["grad_x_Sci"]) < 0.004 and abs(row["grad_y_Sci"]) < 0.004


# master fiber flats across calibration epochs

def make_flat(nfibers=120, npix=1500, shift=0.0, scale=None, seed=0):
    """creates a fake fiber flat with a smooth wavelength dependence and a per-fiber throughput"""
    rng = np.random.default_rng(seed)
    wave = 5700 + shift + 0.6 * np.arange(npix)[None, :] + rng.normal(0, 0.2, nfibers)[:, None]
    throughput = 1 + 0.05 * np.sin(np.arange(nfibers))
    data = throughput[:, None] * (1 + 0.1 * np.cos((wave - 5700) / 300))
    if scale is not None:
        data = data * scale[:, None]
    mask = np.zeros_like(data, dtype=bool)
    mask[:, :20] = True

    telescope = np.array(["Sci"] * (nfibers - 20) + ["SkyE"] * 10 + ["SkyW"] * 10)
    angle = np.linspace(0, 2 * np.pi, nfibers, endpoint=False)
    radius = np.sqrt(np.linspace(0, 1, nfibers))
    slitmap = Table({"fiberid": np.arange(1, nfibers + 1), "telescope": telescope, "fibstatus": np.zeros(nfibers, dtype=int),
                     "spectrographid": np.repeat([1, 2, 3], nfibers // 3),
                     "xpmm": radius * np.cos(angle), "ypmm": radius * np.sin(angle),
                     "orig_ifulabel": [f"F{i}" for i in range(nfibers)]})
    return ff.RSS(data=data, mask=mask, wave=wave, slitmap=slitmap)


def test_bin_fiberflat_aligns_wavelengths():
    # same flat sampled with two different wavelength solutions
    flat_a, flat_b = make_flat(shift=0.0), make_flat(shift=0.7)
    edges = ff.wavelength_edges(flat_a, bin_width=50)
    binned_a, binned_b = ff.bin_fiberflat(flat_a, edges), ff.bin_fiberflat(flat_b, edges)
    assert binned_a.shape == (120, len(edges) - 1)
    ratio = binned_b / binned_a
    assert np.nanmax(np.abs(ratio - 1)) < 1e-3


def test_ratio_stats():
    rng = np.random.default_rng(1)
    values = 1.02 + rng.normal(0, 0.005, 100000)
    values[:10] = np.nan
    stats = ff.ratio_stats(values, max_deviation=0.01)
    assert stats["nvalues"] == 100000 - 10
    assert stats["median"] == pytest.approx(1.02, abs=1e-3)
    assert stats["sigma"] == pytest.approx(0.005, rel=0.05)
    assert stats["p25"] < stats["median"] < stats["p75"]
    assert stats["frac_off"] == pytest.approx(0.977, abs=0.01)
    assert ff.ratio_stats(np.array([np.nan]))["nvalues"] == 0


def test_load_calibration_epochs(tmp_path):
    path = tmp_path / "calibration-epochs.yaml"
    path.write_text("epochs:\n  60255:\n    flavors:\n      twilight: 60255, 60256\n    trigger: Survey start\n    comment: null\n"
                    "  60321:\n    trigger: Normal operations\n")
    epochs = ff.load_calibration_epochs(str(path))
    assert sorted(epochs) == [60255, 60321]
    assert epochs[60255]["trigger"] == "Survey start"


def test_measure_pipeline_gradient():
    # flat with a gradient along x across the science IFU and different spectrograph throughputs
    flat = make_flat()
    sci = np.asarray(flat._slitmap["telescope"]) == "Sci"
    x = np.asarray(flat._slitmap["xpmm"], dtype=float)
    radius = np.max(np.hypot(x[sci] - x[sci].mean(), np.asarray(flat._slitmap["ypmm"])[sci] - np.asarray(flat._slitmap["ypmm"])[sci].mean()))
    factors = np.repeat([1.0, 1.03, 0.98], 40)
    flat._data = flat._data / (1 + 0.05 * np.sin(np.arange(120)))[:, None] * (1 + 0.04 * x / radius)[:, None] * factors[:, None]

    result = ff.measure_pipeline_gradient(flat, "r", cwave=6000.0)
    assert result["grad_x"] == pytest.approx(0.04, abs=0.003)
    assert result["grad_y"] == pytest.approx(0.0, abs=0.003)
    assert np.mean(result["factors"]) == pytest.approx(1.0)
    assert result["factors"][1] / result["factors"][0] == pytest.approx(1.03, abs=2e-3)
    assert result["factors"][2] / result["factors"][0] == pytest.approx(0.98, abs=2e-3)
    assert result["coeffs"][3] == 0
    assert np.isnan(result["gradient"][~sci]).all() and np.nanmean(result["gradient"][sci]) == pytest.approx(1.0, abs=1e-3)


def test_measure_pipeline_gradient_outside_window():
    with pytest.raises(ValueError, match="does not cover"):
        ff.measure_pipeline_gradient(make_flat(), "b")


def test_ifu_outlines():
    # ideal hexagonal IFU of 7 rings split in three 120 deg wedges, one per spectrograph
    pitch, rings = 1.0, 7
    points = [(pitch * (i + j / 2), pitch * j * np.sqrt(3) / 2)
              for i in range(-rings, rings + 1) for j in range(-rings, rings + 1) if abs(i + j) <= rings]
    x, y = np.array(points).T
    angle = np.degrees(np.arctan2(y, x)) % 360
    specid = np.where(np.hypot(x, y) < 1e-9, 1, 1 + (angle // 120).astype(int))
    spec_edges, ifu_edge = ff._ifu_outlines(x, y, specid, pitch, 0.0)

    def vertex_degrees(segments):
        ends = [tuple(np.round(end, 6)) for seg in segments for end in (seg[:2], seg[2:])]
        return collections.Counter(collections.Counter(ends).values())

    # closed outline: 6 sides per ring step around the IFU, every corner shared by two sides
    assert len(ifu_edge) == 6 * (2 * rings + 1)
    assert set(vertex_degrees(ifu_edge)) == {2}
    # three boundaries from a junction near the center to the IFU edge
    degrees = vertex_degrees(spec_edges)
    assert degrees[3] == 1 and degrees[1] == 3


def test_epoch_gradients():
    import pandas as pd
    summary = pd.DataFrame({
        "epoch": [60255, 60321, 60339, 60339], "channel": ["r", "r", "r", "b"], "nvalues": [0, 100, 100, 100],
        "trigger": ["Survey start", "Normal operations", "Seismic event", "Seismic event"],
        "grad_x": [np.nan, 0.03, 0.0, 0.01], "grad_y": [np.nan, 0.04, -0.02, 0.0],
        "pipe_grad_x": [0.01, 0.02, np.nan, 0.0], "pipe_grad_y": [0.0, 0.0, np.nan, 0.01]})

    # ratio to the reference (60255): the reference is left out
    ratio = ff.epoch_gradients(summary, mjd_ref=60255, source="ratio")
    assert list(ratio["mjd"]) == [60321, 60339, 60339]
    r = ratio[(ratio["channel"] == "r") & (ratio["mjd"] == 60321)][0]
    assert r["strength"] == pytest.approx(0.05) and r["direction"] == pytest.approx(np.degrees(np.arctan2(0.04, 0.03)))
    assert r["symbol"] == "circle" and ratio[(ratio["channel"] == "r") & (ratio["mjd"] == 60339)][0]["symbol"] == "diamond"

    # each flat on its own: the reference is included, flats without a measurement are left out
    flat = ff.epoch_gradients(summary, mjd_ref=60255, source="flat")
    assert sorted(zip(flat["channel"], flat["mjd"])) == [("b", 60339), ("r", 60255), ("r", 60321)]
    assert ff.gradient_trends(flat)[0]["nights"] >= 1


def test_header_corrections():
    slitmap = make_flat()._slitmap
    header = {"B FIBERFLAT FACTOR1": 0.95, "B FIBERFLAT FACTOR2": 1.0, "B FIBERFLAT FACTOR3": 1.05,
              "B FIBERFLAT GCOEFF1": 2.5, "B FIBERFLAT GCOEFF2": 0.025, "B FIBERFLAT GCOEFF3": 0.0125, "B FIBERFLAT GCOEFF4": 0.0}
    corrections = ff._header_corrections(header, "b", slitmap)
    assert corrections["hdr_factor3"] == 1.05 and corrections["hdr_gcoeff2"] == 0.025

    # factors relative to their mean over the science fibers (40, 40 and 20 fibers per spectrograph)
    mean = (0.95 * 40 + 1.0 * 40 + 1.05 * 20) / 100
    assert [corrections[f"hdr_offset_sp{i}"] for i in (1, 2, 3)] == pytest.approx([0.95 / mean - 1, 1 / mean - 1, 1.05 / mean - 1])
    # gradient at the science IFU radius over the mean gradient; GCOEFF2 multiplies y and GCOEFF3 x
    sci = np.asarray(slitmap["telescope"]) == "Sci"
    x, y = np.asarray(slitmap["xpmm"])[sci], np.asarray(slitmap["ypmm"])[sci]
    radius, gradient_mean = np.max(np.hypot(x - x.mean(), y - y.mean())), np.mean(2.5 + 0.025 * y + 0.0125 * x)
    assert corrections["hdr_grad_x"] == pytest.approx(0.0125 * radius / gradient_mean)
    assert corrections["hdr_grad_y"] == pytest.approx(0.025 * radius / gradient_mean)

    missing = ff._header_corrections({}, "b", slitmap)
    assert np.isnan(missing["hdr_offset_sp1"]) and np.isnan(missing["hdr_grad_x"])


def test_qa_fiberflat_epochs(tmp_path, monkeypatch):
    epochs = {60255: {"trigger": "Survey start"}, 60321: {"trigger": "Normal operations"},
              60339: {"trigger": "Instrument intervention", "comment": "replaced a motor"}, 60355: {"trigger": "Normal operations"}}
    # epoch 60321 only changes the spectrograph factors, epoch 60339 has 2% more throughput in
    # the first half of the fibers, epoch 60355 has no flats
    spec_offsets = np.repeat([1.02, 1.0, 0.99], 40)
    scale = np.where(np.arange(120) < 60, 1.02, 1.0)
    flats = {60255: make_flat(), 60321: make_flat(shift=0.5, scale=spec_offsets), 60339: make_flat(shift=-0.4, scale=scale)}
    # the flat of epoch 60321 records the spectrograph factors and gradient applied to it
    flats[60321]._header = {f"{c} FIBERFLAT {key}": value for c in "BR" for key, value in
                            [("FACTOR1", 1.02), ("FACTOR2", 1.0), ("FACTOR3", 0.99),
                             ("GCOEFF1", 1.0), ("GCOEFF2", 0.0), ("GCOEFF3", 0.0), ("GCOEFF4", 0.0)]}
    loaded = {}
    for mjd, flat in flats.items():
        for channel in "br":
            path = ff.fiberflat_path(mjd, channel, flats_dir=str(tmp_path / "calib"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
            loaded[path] = flat
    monkeypatch.setattr(ff.RSS, "from_file", staticmethod(lambda path: loaded[path]))

    # windows of the pipeline measurement within the wavelength range of the fake flats
    result = ff.qa_fiberflat_epochs(mjd_ref=60255, channels="br", epochs=epochs, flats_dir=str(tmp_path / "calib"),
                                    output_dir=str(tmp_path / "qa"), pipeline_cwaves={"b": 6000.0, "r": 6100.0})
    summary = result["summary"].set_index(["epoch", "channel"])
    # as measured the spectrograph factors spread the ratios; the decomposition removes them
    assert summary.loc[(60321, "b"), "sigma"] > 0.005
    assert summary.loc[(60321, "b"), "corr_sigma"] < 1e-3
    offsets = [summary.loc[(60321, "b"), f"offset_sp{i}"] for i in (1, 2, 3)]
    assert offsets[0] - offsets[1] == pytest.approx(0.02, abs=2e-3)
    assert offsets[2] - offsets[1] == pytest.approx(-0.01, abs=2e-3)
    assert summary.loc[(60339, "r"), "p75"] == pytest.approx(1.02, abs=1e-3)
    assert summary.loc[(60339, "r"), "frac_off"] == pytest.approx(0.5, abs=0.05)
    assert not summary.loc[(60355, "b"), "exists"] and summary.loc[(60355, "b"), "nvalues"] == 0
    assert summary.loc[(60255, "b"), "reference"]
    # the header factors in the normalization of the measured offsets, which they explain here
    header_offsets = [summary.loc[(60321, "b"), f"hdr_offset_sp{i}"] for i in (1, 2, 3)]
    assert np.allclose(np.diff(header_offsets), np.diff(offsets), atol=2e-3)
    assert summary.loc[(60321, "r"), "hdr_grad_x"] == 0 and np.isnan(summary.loc[(60339, "b"), "hdr_offset_sp1"])
    # each flat on its own, measured as in the pipeline: the factors follow the spectrograph changes of 60321
    pipe = {mjd: np.array([summary.loc[(mjd, "b"), f"pipe_factor{i}"] for i in (1, 2, 3)]) for mjd in (60255, 60321)}
    change = pipe[60321] / pipe[60255]
    assert change[0] / change[1] == pytest.approx(1.02, abs=2e-3) and change[2] / change[1] == pytest.approx(0.99, abs=2e-3)
    assert np.isfinite(summary.loc[(60255, "b"), "pipe_grad_x"])

    html = open(result["report"], encoding="utf-8").read()
    assert html.isascii()
    assert 'data-fig="ratios|corr"' in html and 'data-fig="offsets"' in html and 'id="ifu-grid"' in html
    assert 'data-fig="gradient_time|ratio"' in html and "Table view: persistent gradient and trends" in html
    assert set(result["figures"]) == {"ratios|raw", "ratios|corr", "offsets", "gradients", "gradient_time|ratio", "gradient_time|flat"}
    assert "replaced a motor" in html and "Header gradient x / y (%)" in html
    assert 'value="pgrad"' in html and 'value="factors"' in html and '"numbers"' in html
    assert '"specEdges"' in html and '"ifuEdge"' in html and '"specCenters"' in html
    assert os.path.isfile(result["report"].replace(".html", ".csv"))


def test_qa_fiberflat_epochs_default_output(tmp_path, monkeypatch):
    epochs = {60255: {"trigger": "Survey start"}}
    # sandbox flats: the dashboard goes next to them, in the master calibrations directory
    monkeypatch.setenv("LVM_MASTER_DIR", str(tmp_path / "sandbox"))
    for channel in "br":
        path = ff.fiberflat_path(60255, channel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
    result = ff.qa_fiberflat_epochs(channels="br", epochs=epochs, dry_run=True)
    assert result["output_dir"] == str(tmp_path / "sandbox" / "qa" / "fiberflat" / "twilight_vs_60255")

    # flats of a pipeline version: in the ancillary directory of the reference epoch
    for channel in "br":
        path = ff.fiberflat_path(60255, channel, drpver="1.2.3", redux_dir=str(tmp_path / "redux"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
    result = ff.qa_fiberflat_epochs(channels="br", epochs=epochs, drpver="1.2.3", redux_dir=str(tmp_path / "redux"), dry_run=True)
    assert result["output_dir"] == str(tmp_path / "redux" / "1.2.3" / "0011XX" / "11111" / "60255" / "ancillary" / "fiberflat_qa" / "twilight_vs_60255")


@pytest.mark.parametrize("selection, name", [
    ({}, "all"),
    ({"mjds": [61313]}, "mjd_61313"),
    ({"mjds": [61330, 61300, 61313]}, "mjd_61300-61330"),
    ({"mjd_range": (61330, 60300)}, "mjd_60300-61330"),
    ({"mjds": [61313], "tileids": [1028325, 1027988]}, "mjd_61313_tiles_1027988-1028325"),
    ({"mjd_range": (60300, 61330), "tileids": list(range(5))}, "mjd_60300-61330_5tiles"),
    ({"frames_file": "/some/where/twilight_frames.txt", "mjds": [61313]}, "frames_twilight_frames"),
])
def test_flatfield_qa_name(selection, name):
    assert ff.flatfield_qa_name(**selection) == name


def test_default_flatfield_qa_dir(tmp_path, monkeypatch):
    assert ff.default_flatfield_qa_dir("1.2.3", "mjd_61313", redux_dir="/redux") == "/redux/1.2.3/qa/flatfield/mjd_61313"
    monkeypatch.setenv("LVM_SPECTRO_REDUX", str(tmp_path))
    assert ff.default_flatfield_qa_dir("1.2.3", "all") == str(tmp_path / "1.2.3" / "qa" / "flatfield" / "all")


def test_reference_epoch_without_flats(tmp_path):
    with pytest.raises(FileNotFoundError):
        ff.qa_fiberflat_epochs(mjd_ref=60255, channels="b", epochs={60255: {}}, flats_dir=str(tmp_path), dry_run=True)


# twilight sequences

def _twilight_header(t, alt, az, pa, expnum=1, channel="b", lamps_on=(), gcoeffs=(1.0, 0.0, 0.0, 0.0), factors=(1.0, 1.0, 1.0)):
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy import units as u
    from lvmdrp.core.constants import LVM_LAT, LVM_LON, LVM_ELEVATION

    location = EarthLocation(lat=LVM_LAT, lon=LVM_LON, height=LVM_ELEVATION * u.m)
    pointing = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame=AltAz(obstime=t, location=location)).icrs
    header = fits.Header({"OBSTIME": t.isot, "EXPTIME": 0.0, "EXPOSURE": expnum, "MJD": 61000, "CCD": channel,
                          "POSCIRA": pointing.ra.deg, "POSCIDE": pointing.dec.deg, "POSCIPA": pa})
    for lamp in ("NEON", "HGNE", "ARGON", "XENON", "LDLS", "QUARTZ"):
        header[lamp] = "ON" if lamp in lamps_on else "OFF"
    for k, c in enumerate(gcoeffs, start=1):
        header[f"HIERARCH {channel.upper()} FIBERFLAT GCOEFF{k}"] = c
    for i, f in enumerate(factors, start=1):
        header[f"HIERARCH {channel.upper()} FIBERFLAT FACTOR{i}"] = f
    return header


def _sun_position(t):
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy import units as u
    from lvmdrp.core.constants import LVM_LAT, LVM_LON, LVM_ELEVATION

    location = EarthLocation(lat=LVM_LAT, lon=LVM_LON, height=LVM_ELEVATION * u.m)
    return get_body("sun", t, location).transform_to(AltAz(obstime=t, location=location))


def test_twilight_geometry():
    from astropy.time import Time

    slitmap = make_flat()._slitmap
    t = Time("2025-06-15T23:05:00", scale="utc")
    sun = _sun_position(t)
    diff = lambda a, b: (a - b + 180) % 360 - 180  # noqa: E731

    towards = ff.twilight_geometry(_twilight_header(t, 45, sun.az.deg, 0.0), slitmap)
    assert towards["sun_alt"] == pytest.approx(sun.alt.deg, abs=0.05) and towards["point_alt"] == pytest.approx(45, abs=0.01)
    # looking towards the Sun below the horizon, the Sun and the horizon are in the same direction across the IFU
    assert diff(towards["dir_sun"], towards["dir_horizon"]) == pytest.approx(0, abs=1)
    # looking away from the Sun, the Sun is towards the zenith
    away = ff.twilight_geometry(_twilight_header(t, 45, sun.az.deg + 180, 0.0), slitmap)
    assert abs(diff(away["dir_sun"], away["dir_horizon"])) == pytest.approx(180, abs=1)
    # rotating the IFU rotates both directions by the same angle
    pa0 = ff.twilight_geometry(_twilight_header(t, 45, sun.az.deg + 60, 0.0), slitmap)
    pa30 = ff.twilight_geometry(_twilight_header(t, 45, sun.az.deg + 60, 30.0), slitmap)
    assert abs(diff(pa30["dir_sun"], pa0["dir_sun"])) == pytest.approx(30, abs=0.5)
    assert diff(pa30["dir_sun"], pa0["dir_sun"]) == pytest.approx(diff(pa30["dir_horizon"], pa0["dir_horizon"]), abs=0.5)

    # bytes columns (as read from FITS) and existing fiber coordinates don't change the result
    stale = Table(slitmap, copy=True)
    stale["telescope"] = np.char.encode(np.asarray(stale["telescope"]).astype(str))
    stale["ra"], stale["dec"] = 0.0, 0.0
    assert ff.twilight_geometry(_twilight_header(t, 45, sun.az.deg + 60, 0.0), stale)["dir_sun"] == pytest.approx(pa0["dir_sun"])

    # without a commanded position only the Sun and the Moon are computed
    header = _twilight_header(t, 45, sun.az.deg, 0.0)
    header["POSCIRA"], header["POSCIDE"] = -999.0, -999.0
    missing = ff.twilight_geometry(header, slitmap)
    assert np.isfinite(missing["sun_alt"]) and np.isnan(missing["dir_sun"]) and np.isnan(missing["point_alt"])


def test_is_twilight():
    from astropy.time import Time

    t = Time("2025-06-15T23:05:00", scale="utc")
    assert ff._is_twilight(fits.Header({"LDLS": "OFF", "QUARTZ": "OFF"}))
    assert not ff._is_twilight(fits.Header({"LDLS": "ON", "QUARTZ": "OFF"}))
    assert not ff._is_twilight(_twilight_header(t, 45, 0, 0, lamps_on=("QUARTZ",)))


def test_qa_twilight_sequence(tmp_path):
    from astropy.time import Time

    slitmap = make_flat()._slitmap
    start = Time("2025-06-15T23:00:00", scale="utc")
    paths = []
    # three twilight exposures with the same removed gradient in two channels, and a dome flat that is left out
    for expnum, minutes in ((101, 0), (102, 5), (103, 10)):
        for channel in "br":
            header = _twilight_header(start + minutes * 60 * ff.u.s, 60, 100, 0.0, expnum=expnum, channel=channel,
                                      gcoeffs=(2.0, 0.02, 0.01, 0.0), factors=(0.95, 1.0, 1.05))
            path = tmp_path / f"lvm-fflat-{channel}-{expnum:08d}.fits"
            fits.HDUList([fits.PrimaryHDU(header=header), fits.table_to_hdu(slitmap)]).writeto(path)
            paths.append(str(path))
    dome = _twilight_header(start, 60, 100, 0.0, expnum=104, channel="b", lamps_on=("LDLS",))
    fits.HDUList([fits.PrimaryHDU(header=dome), fits.table_to_hdu(slitmap)]).writeto(tmp_path / "lvm-fflat-b-00000104.fits")
    paths.append(str(tmp_path / "lvm-fflat-b-00000104.fits"))
    for hdul_path in paths:
        with fits.open(hdul_path, mode="update") as hdul:
            hdul[1].name = "SLITMAP"

    result = ff.qa_twilight_sequence(61000, channels="br", flat_paths=paths, include_raw=False, output_dir=str(tmp_path / "qa"))
    summary = result["summary"]
    assert sorted(summary["expnum"].unique()) == [101, 102, 103]
    assert len(summary) == 6

    # removed gradient as in the other dashboards
    gx, gy, _ = ff._gradient_at_edge([2.0, 0.02, 0.01, 0.0], slitmap)
    assert np.allclose(summary["grad_x"], gx) and np.allclose(summary["grad_y"], gy)
    assert np.allclose(summary["factor1"], 0.95) and np.allclose(summary["factor3"], 1.05)
    assert np.allclose(summary["dtheta_sun"], (summary["direction"] - summary["dir_sun"] + 180) % 360 - 180)
    # the Sun sets along the sequence
    assert np.all(np.diff(summary.drop_duplicates("expnum")["sun_alt"]) < 0)

    html = open(result["report"], encoding="utf-8").read()
    assert html.isascii() and 'data-fig="sky|61000"' in html and 'data-fig="gradient|b|61000"' in html
    assert set(result["figures"]) == {"sky|61000", "geometry|61000", "gradient|b|61000", "gradient|r|61000",
                                      "factors|b|61000", "factors|r|61000"}
    # IFU tiles: the gradient model of each exposure and the directions of the Sun and the horizon
    assert 'id="twi-grid"' in html and ".ifu-grid" in html
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", html).group(1))
    assert set(payload["values"]) == {"b", "r"} and set(payload["values"]["b"]) == {"101", "102", "103"}
    model = np.array(payload["values"]["b"]["101"], dtype=float) / 1e4
    assert np.nanmean(model) == pytest.approx(1.0, abs=1e-3) and np.nanstd(model) > 0
    assert all(e["dirSun"] is not None and e["dirHorizon"] is not None for e in payload["exposures"])
    assert os.path.isfile(result["report"].replace(".html", ".csv"))


def test_qa_twilight_sequence_without_flats(tmp_path):
    with pytest.raises(FileNotFoundError, match="fiberflats --skip-combination"):
        ff.qa_twilight_sequence(61000, flat_paths=[], include_raw=False, output_dir=str(tmp_path))
# dome sequences

def test_is_dome():
    assert ff._is_dome(fits.Header({"LDLS": "ON", "QUARTZ": "OFF"}))
    assert ff._is_dome(fits.Header({"LDLS": "OFF", "QUARTZ": "ON"}))
    assert not ff._is_dome(fits.Header({"LDLS": "OFF", "QUARTZ": "OFF"}))
    assert not ff._is_dome(fits.Header({"LDLS": "ON", "NEON": "ON"}))


def test_circular_spread():
    mean, spread = ff._circular_spread([179.0, -179.0, np.nan])
    assert abs(abs(mean) - 180) < 1e-6 and spread == pytest.approx(1.0)
    assert np.isnan(ff._circular_spread([np.nan])[0])


def test_qa_dome_sequence(tmp_path):
    from astropy.time import Time

    slitmap = make_flat()._slitmap
    start = Time("2025-06-15T22:00:00", scale="utc")
    paths = []

    def write(expnum, channel, lamps_on, gcoeffs, factors, calibfib):
        header = _twilight_header(start + (expnum - 200) * 60 * ff.u.s, 60, 100, 0.0, expnum=expnum, channel=channel,
                                  lamps_on=lamps_on, gcoeffs=gcoeffs, factors=factors)
        header["CALIBFIB"] = calibfib
        header["EXPTIME"] = 90.0
        path = tmp_path / f"lvm-fflat-{channel}-{expnum:08d}.fits"
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.table_to_hdu(slitmap)])
        hdul[1].name = "SLITMAP"
        hdul.writeto(path)
        paths.append(str(path))

    # LDLS exposures for b and r, quartz exposures for z, and a twilight that is left out
    for i, expnum in enumerate((201, 202, 203)):
        for channel in "br":
            write(expnum, channel, ("LDLS",), (2.0, 0.02, 0.01, 0.0), (0.98, 1.0, 1.02 + 0.01 * i), f"P1-{i + 1}")
    for i, expnum in enumerate((211, 212)):
        write(expnum, "z", ("QUARTZ",), (2.0, -0.01, 0.0, 0.0), (1.0, 1.0, 1.0), f"P1-{i + 1}")
    write(220, "b", (), (2.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0), "P1-1")

    result = ff.qa_dome_sequence(61000, channels="brz", flat_paths=paths, epoch_mjd=60999, output_dir=str(tmp_path / "qa"))
    summary = result["summary"]
    assert sorted(summary["expnum"].unique()) == [201, 202, 203, 211, 212]
    assert len(summary) == 8
    assert set(summary.loc[summary["channel"] == "z", "lamp"]) == {"quartz"}
    assert set(summary.loc[summary["channel"] != "z", "lamp"]) == {"ldls"}
    assert list(summary.loc[summary["channel"] == "b", "calibfib"]) == ["P1-1", "P1-2", "P1-3"]

    # removed gradient and factors as in the twilight dashboard
    gx, gy, _ = ff._gradient_at_edge([2.0, 0.02, 0.01, 0.0], slitmap)
    b = summary[summary["channel"] == "b"]
    assert np.allclose(b["grad_x"], gx) and np.allclose(b["grad_y"], gy)
    assert np.allclose(b[["factor1", "factor2", "factor3"]].mean(axis=1), 1.0)
    # the z gradient points towards -y
    assert np.allclose(summary.loc[summary["channel"] == "z", "direction"], -90)

    assert set(result["figures"]) == {"vectors", "gradient", "factors|b", "factors|r", "factors|z"}
    assert list(result["figures"]["gradient"].layout.xaxis.categoryarray) == ["201", "202", "203", "211", "212"]
    html = open(result["report"], encoding="utf-8").read()
    assert html.isascii() and 'data-fig="vectors"' in html and 'data-fig="factors|b"' in html
    assert result["report"].endswith("domes_60999.html") and os.path.isfile(result["report"].replace(".html", ".csv"))
    # IFU tiles of each channel, without arrows
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", html).group(1))
    assert set(payload["values"]) == {"b", "r", "z"} and set(payload["values"]["z"]) == {"211", "212"}
    assert all(e["dirSun"] is None and e["dirHorizon"] is None for e in payload["exposures"])
    assert payload["exposures"][0]["subtitle"].endswith("ldls · P1-1")
    assert "towards the Sun" not in html

    with pytest.raises(FileNotFoundError):
        ff.qa_dome_sequence(61000, flat_paths=[paths[-1]], output_dir=str(tmp_path / "none"))
