# encoding: utf-8
#

import os
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
    for div in ("fig-excess", "fig-offsets", "fig-outliers", "fig-timeline", "fig-offsets-time"):
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

    _, lines_summary, _, cont_summary = measure_flatfield_qa(frame, gradient_deg=1, **kwargs)
    for summary in (lines_summary, cont_summary):
        assert np.all(summary["excess"] < 0.004)
        assert np.all(np.abs([summary[f"offset_sp{i}"] for i in (1, 2, 3)]) < 0.003)
        assert np.allclose(summary["grad_x_Sci"], 0.04, atol=0.004)
        assert np.allclose(summary["grad_y_Sci"], -0.03, atol=0.004)


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
    return SimpleNamespace(_data=data, _wave=wave, _mask=mask, _slitmap=slitmap)


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


def test_qa_fiberflat_epochs(tmp_path, monkeypatch):
    epochs = {60255: {"trigger": "Survey start"}, 60321: {"trigger": "Normal operations"},
              60339: {"trigger": "Instrument intervention", "comment": "replaced a motor"}, 60355: {"trigger": "Normal operations"}}
    # epoch 60321 only changes the spectrograph factors, epoch 60339 has 2% more throughput in
    # the first half of the fibers, epoch 60355 has no flats
    spec_offsets = np.repeat([1.02, 1.0, 0.99], 40)
    scale = np.where(np.arange(120) < 60, 1.02, 1.0)
    flats = {60255: make_flat(), 60321: make_flat(shift=0.5, scale=spec_offsets), 60339: make_flat(shift=-0.4, scale=scale)}
    loaded = {}
    for mjd, flat in flats.items():
        for channel in "br":
            path = ff.fiberflat_path(mjd, channel, flats_dir=str(tmp_path / "calib"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
            loaded[path] = flat
    monkeypatch.setattr(ff.RSS, "from_file", staticmethod(lambda path: loaded[path]))

    result = ff.qa_fiberflat_epochs(mjd_ref=60255, channels="br", epochs=epochs, flats_dir=str(tmp_path / "calib"),
                                    output_dir=str(tmp_path / "qa"))
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

    html = open(result["report"], encoding="utf-8").read()
    assert html.isascii()
    assert 'data-fig="ratios|corr"' in html and 'data-fig="offsets"' in html and 'id="ifu-grid"' in html
    assert set(result["figures"]) == {"ratios|raw", "ratios|corr", "offsets", "gradients"}
    assert "replaced a motor" in html
    assert os.path.isfile(result["report"].replace(".html", ".csv"))


def test_reference_epoch_without_flats(tmp_path):
    with pytest.raises(FileNotFoundError):
        ff.qa_fiberflat_epochs(mjd_ref=60255, channels="b", epochs={60255: {}}, flats_dir=str(tmp_path), dry_run=True)
