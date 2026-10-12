# encoding: utf-8
#

import base64
import io
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from PIL import Image

import lvmdrp.qa.science as sci


def test_line_crossings():
    pixels = np.arange(100, dtype=float)
    wave = np.stack([5000 + 2 * pixels, 5010 + 2 * pixels, 6000 - 2 * pixels])
    cent = np.stack([np.full(100, 10.0), 20.0 + 0.1 * pixels, np.full(100, 30.0)])

    x, y = sci.line_crossings(wave, cent, 5050.0)

    assert np.allclose(x[:2], [25.0, 20.0])
    assert np.allclose(y[:2], [10.0, 22.0])
    # descending wavelengths, and a wavelength out of range
    assert np.isnan(x[2]) and np.isnan(y[2])
    x, _ = sci.line_crossings(wave[2:], cent[2:], 5900.0)
    assert np.isclose(x[0], 50.0)


def test_measure_camera_traces():
    pixels = np.arange(4086, dtype=float)
    nfibers = 6
    master_cent = np.tile(np.arange(nfibers, dtype=float)[:, None] * 10 + 100, (1, pixels.size))
    master_wave = np.tile(5000 + 0.2 * pixels, (nfibers, 1))
    shift = np.linspace(-0.1, 0.1, nfibers)
    offset = 0.05

    traces = sci.measure_camera_traces("b1", master_cent, master_wave, master_cent + shift[:, None], master_wave - offset)

    assert np.allclose(traces["shift"], shift)
    assert np.allclose(traces["wave_offset"], offset)
    line = traces["lines"][0]
    assert line["wave"] == sci.REF_SKYLINES["b"][0]
    assert "wavelength correction" in line["use"] and "fiber flat check" in line["use"]
    # the corrected solution reaches the line 0.05 / 0.2 pixels later
    assert np.allclose(line["x_frame"] - line["x_master"], offset / 0.2)
    assert np.allclose(line["y_frame"] - line["y_master"], shift)


def test_encode_image_puts_first_row_at_the_bottom():
    data = np.zeros((4, 3))
    data[0] = 10.0
    uri = sci.encode_image(data, 0.0, 10.0)
    assert uri.startswith("data:image/jpeg;base64,")
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))).astype(int)
    assert image.shape == (4, 3)
    # lossy: compare with a tolerance
    assert (image[-1] > 200).all() and (image[:-1] < 55).all()


def test_fig_frame_views():
    data = np.random.default_rng(0).normal(100, 5, (400, 4086))
    traces = sci.measure_camera_traces("b1", np.full((3, 4086), 50.0), np.tile(5400 + 0.05 * np.arange(4086), (3, 1)))
    assert sci.fig_frame(data, 90, 110, traces, "full") is not None
    fig = sci.fig_frame(data, 90, 110, traces, "line1")
    assert fig.layout.xaxis.range[1] - fig.layout.xaxis.range[0] == 2 * sci.CUTOUT_HALF_WIDTH + 1
    assert len(fig.layout.images) == 1
    # b has a single reference sky line
    assert sci.fig_frame(data, 90, 110, traces, "line2") is None


def test_measure_sky_residuals():
    wave = np.arange(6250, 6400, 0.5)
    nfibers = 40
    sigma = 1.0
    sky = 50 + 1000 * np.exp(-0.5 * ((wave - 6300.3) / sigma) ** 2)[None, :].repeat(nfibers, axis=0)
    fraction = np.linspace(-0.05, 0.05, nfibers)
    flux = 20 + fraction[:, None] * (sky - 50)
    select = np.ones(nfibers, dtype=bool)
    select[0] = False

    result = sci.measure_sky_residuals(wave, flux, sky, np.ones_like(flux), select, [6300.3, 6363.78])

    assert np.isnan(result["fraction"][0, 0])
    assert np.allclose(result["fraction"][1:, 0], fraction[1:], atol=1e-3)
    # no line at 6363.78: no sky flux to compare with
    assert np.isnan(result["fraction"][:, 1]).all()
    peak = np.nanmax(result["profiles"][0], axis=1)
    assert np.allclose(peak[1:][fraction[1:] > 0], fraction[1:][fraction[1:] > 0], atol=1e-3)

    centers, binned = sci._fiber_bins(result["fraction"])
    assert binned.shape == (nfibers // sci.FIBER_BIN, 2)
    assert np.allclose(centers[:2], [3.5, 11.5])


def test_standard_stars_and_synthetic_mags():
    header = fits.Header({"STD1ACQ": True, "STD1FIB": "P1-2", "STD1ID": 123, "STD1EXP": 90.0,
                          "STD2ACQ": False, "STD2FIB": "P1-3", "STD2ID": 456,
                          "STD3ACQ": True, "STD3FIB": "P1-9", "STD3ID": 789})
    slitmap = Table({"orig_ifulabel": ["P1-1", "P1-2", "P1-3"]})
    stars = sci.standard_stars(header, slitmap)
    assert [(s["index"], s["row"], s["gaia_id"]) for s in stars] == [(1, 1, 123)]

    # flat f_nu spectrum of AB magnitude 10
    wave = np.arange(3600, 9800, 0.5)
    flux = 10 ** (-0.4 * (10 + 48.6)) * 2.99792458e18 / wave ** 2
    mags = sci.synthetic_mags(wave, flux)
    assert np.allclose([mags[c] for c in "brz"], 10.0, atol=1e-3)


def _products(tmp_path, remeasure):
    products = object.__new__(sci.ExposureProducts)
    products.work_dir = str(tmp_path / "products")
    products.remeasure = remeasure
    products.missing = []
    products._produced = {}
    products.anc = lambda kind, camera, imagetype=None: str(tmp_path / "ancillary" / f"lvm-{kind}object-{camera}.fits")
    return products


def test_missing_ancillary_is_skipped(tmp_path):
    products = _products(tmp_path, remeasure=False)
    calls = []
    assert products._get("d", "b1", lambda: calls.append(1)) is None
    assert calls == [] and len(products.missing) == 1

    # an ancillary product produced by a previous run of the dashboard is reused
    os.makedirs(products.work_dir)
    work = os.path.join(products.work_dir, "lvm-dobject-b1.fits")
    open(work, "w").close()
    assert products._get("d", "b1", lambda: calls.append(1)) == work


def test_remeasure_produces_once(tmp_path):
    products = _products(tmp_path, remeasure=True)
    os.makedirs(tmp_path / "ancillary")
    open(tmp_path / "ancillary" / "lvm-dobject-b1.fits", "w").close()
    work = os.path.join(products.work_dir, "lvm-dobject-b1.fits")
    calls = []

    def produce():
        calls.append(1)
        open(work, "w").close()

    # the ancillary product exists but is measured again
    assert products._get("d", "b1", produce) == work
    assert products._get("d", "b1", produce) == work
    assert calls == [1]

    def fail():
        raise RuntimeError("boom")

    assert products._get("l", "b1", fail) is None
    assert "boom" in products.missing[-1]


def test_thermal_shifts_from_slitmaps():
    master = Table({"spectrographid": [1, 1, 2, 2], "ypix_b": [10.0, 20.0, 30.0, 40.0]})
    frame = Table({"spectrographid": [1, 1, 2, 2], "ypix_b": np.array([10.1, 19.9, 30.5, 40.5], dtype=">f4")})
    assert np.allclose(sci.thermal_shifts(frame, master, "b1"), [0.1, -0.1], atol=1e-5)
    assert np.allclose(sci.thermal_shifts(frame, master, "b2"), [0.5, 0.5], atol=1e-5)


def test_calibrate_with_methods():
    npix = 20
    counts = np.full((2, npix), 10.0)
    curves = {"mod": np.full(npix, 2.0), "std": np.full(npix, 3.0), "sci": np.full(npix, np.nan)}
    tables = {key: Table({"STD1SEN": value, "mean": value, "rms": 0 * value}) for key, value in curves.items()}
    cframe = sci.lvmCFrame(data=counts * 2.0, error=np.ones_like(counts), mask=np.zeros_like(counts, dtype=bool),
                           header=fits.Header({"FLUXCAL": "MOD", "BUNIT": "erg / (Angstrom s cm2)"}), wave=np.arange(npix, dtype=float),
                           fluxcal_std=tables["std"], fluxcal_sci=tables["sci"], fluxcal_mod=tables["mod"])
    spectra, unavailable = sci.calibrate_with_methods(cframe)
    assert set(spectra) == {"MOD", "STD"} and set(unavailable) == {"SCI"}
    assert np.allclose(spectra["MOD"], 20.0) and np.allclose(spectra["STD"], 30.0)
    # the frame keeps its calibration
    assert cframe._header["FLUXCAL"] == "MOD" and np.allclose(cframe._data, 20.0)


def test_fig_sensitivity_divides_by_mod():
    wave = np.linspace(3600, 9800, 100)
    curves = {"MOD": (np.full(100, 2.0), np.zeros(100)), "STD": (np.full(100, 3.0), np.zeros(100))}
    fig = sci.fig_sensitivity(wave, curves, "MOD")
    ratios = [trace for trace in fig.data if trace.name == "STD / MOD"]
    assert len(ratios) == 1 and np.allclose(ratios[0].y, 1.5)
    # without a MOD curve there is nothing to divide by
    fig = sci.fig_sensitivity(wave, {"STD": curves["STD"]}, "STD")
    assert not any("/" in (trace.name or "") for trace in fig.data)
