import numpy as np
from astropy.table import Table

from lvmdrp.core.rss import RSS
from lvmdrp.functions.skyMethod import combine_skies


def _make_sci_sky_rss(out_rss, n_sci=5, line_center=6300.0, line_sigma=0.3,
                       line_peak=500.0, continuum=10.0, fiber_offset=1.2):
    """Builds a synthetic RSS with a per-fiber wavelength solution offset --
    mimicking the real, non-rectified per-fiber wavelength grids that exist
    at the pipeline stage where combine_skies actually runs -- and a narrow
    emission line injected at a fixed *wavelength* (not fixed pixel column)
    in every fiber.

    A per-column combination across fibers is only correct once every fiber
    shares a common wavelength grid; here each fiber's grid is deliberately
    shifted, so a bug that combines by column instead of by wavelength
    smears the narrow line away while leaving the flat continuum untouched.
    """
    n_fib = n_sci + 2  # + SkyE, SkyW (unused by the SCIMED default, but present in a real slitmap)
    n_pix = 1400

    base_wave = np.arange(6000.0, 6000.0 + n_pix * 0.5, 0.5)
    offsets = np.linspace(0, fiber_offset * (n_fib - 1), n_fib)
    wave = np.repeat([base_wave], n_fib, axis=0) + offsets[:, None]

    data = continuum + line_peak * np.exp(-0.5 * ((wave - line_center) / line_sigma) ** 2)
    error = np.sqrt(np.abs(data))
    mask = np.zeros_like(data, dtype=bool)

    targettype = ["SKY", "SKY"] + ["science"] * n_sci
    telescope = ["SkyE", "SkyW"] + ["Sci"] * n_sci
    slitmap = Table({"targettype": targettype, "telescope": telescope})

    rss = RSS(wave=wave, data=data, error=error, mask=mask, slitmap=slitmap)
    rss.setHdrValue("SKYERA", 10.0)
    rss.setHdrValue("SKYEDEC", -10.0)
    rss.setHdrValue("SKYWRA", 20.0)
    rss.setHdrValue("SKYWDEC", -20.0)
    rss.setHdrValue("SCIRA", 15.0)
    rss.setHdrValue("SCIDEC", -15.0)
    rss.writeFitsData(out_rss)

    return wave, line_center


def test_combine_skies_scimed_preserves_emission_line(tmp_path):
    """Regression test for the bug fixed alongside issue #250: the SCIMED
    default in combine_skies used to take np.nanmedian(rss._data[sci_idx, :],
    axis=0), a per-pixel-*column* median, before the RSS is wavelength
    rectified. Since every fiber has its own native wavelength-to-pixel
    mapping at that stage, this mixed flux from different wavelengths and
    washed out anything as narrow as a night-sky emission line while leaving
    the smooth continuum intact.
    """
    in_rss = str(tmp_path / "in_sci.fits")
    out_rss = str(tmp_path / "out_sci.fits")
    wave, line_center = _make_sci_sky_rss(in_rss)

    combine_skies(in_rss=in_rss, out_rss=out_rss, sky_weights=None)

    rss = RSS.from_file(out_rss)
    assert rss._header.get("SKYSRC") == "SCIMED"

    fiber = 0
    fwave = wave[fiber]
    sky = rss._sky_east[fiber]

    line_idx = np.argmin(np.abs(fwave - line_center))
    peak = np.nanmax(sky[line_idx - 3:line_idx + 4])
    continuum = np.nanmedian(np.concatenate([sky[:line_idx - 50], sky[line_idx + 50:]]))

    # a per-column (wavelength-unaware) median would leave peak ~= continuum;
    # a correctly wavelength-aligned median recovers most of the injected
    # ~50x peak-to-continuum line.
    assert peak / continuum > 10
