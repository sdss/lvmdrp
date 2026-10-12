import numpy as np
from astropy.table import Table

from lvmdrp.core.rss import RSS
from lvmdrp.core.sky import select_sky_fibers, fit_supersky
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

    Also attaches a SUPERSKY/SUPERSKY_ERROR table exactly as `interpolate_sky`
    would, since `combine_skies` always evaluates it (for SKY_EAST/SKY_WEST)
    regardless of the flux-calibration sky method. Two fibers per sky
    telescope (rather than one) so fit_supersky's per-fiber outlier rejection
    -- which needs a nonzero biweight scale across fibers -- doesn't degenerate
    and reject the only fiber of that telescope.
    """
    n_sky_per_tel = 2
    n_fib = n_sci + 2 * n_sky_per_tel
    n_pix = 4086  # eval_supersky hardcodes a reshape to (-1, 4086), matching the real detector

    base_wave = np.arange(6000.0, 6000.0 + n_pix * 0.5, 0.5)
    offsets = np.linspace(0, fiber_offset * (n_fib - 1), n_fib)
    wave = np.repeat([base_wave], n_fib, axis=0) + offsets[:, None]

    data = continuum + line_peak * np.exp(-0.5 * ((wave - line_center) / line_sigma) ** 2)
    # give the sky-telescope fibers a distinguishable per-fiber continuum offset
    # so fit_supersky's biweight-based outlier rejection sees a nonzero spread
    # across fibers instead of degenerating (a single near-identical value would
    # give biweight_scale == 0, rejecting every fiber as an "outlier")
    n_sky_fibers = 2 * n_sky_per_tel
    data[:n_sky_fibers] += 5.0 * np.arange(n_sky_fibers)[:, None]
    error = np.sqrt(np.abs(data))
    mask = np.zeros_like(data, dtype=bool)

    targettype = ["SKY"] * (2 * n_sky_per_tel) + ["science"] * n_sci
    telescope = ["SkyE"] * n_sky_per_tel + ["SkyW"] * n_sky_per_tel + ["Sci"] * n_sci
    slitmap = Table({"targettype": targettype, "telescope": telescope})

    rss = RSS(wave=wave, data=data, error=error, mask=mask, slitmap=slitmap)
    rss.setHdrValue("BUNIT", "electron")
    rss.setHdrValue("SKYERA", 10.0)
    rss.setHdrValue("SKYEDEC", -10.0)
    rss.setHdrValue("SKYWRA", 20.0)
    rss.setHdrValue("SKYWDEC", -20.0)
    rss.setHdrValue("SCIRA", 15.0)
    rss.setHdrValue("SCIDEC", -15.0)

    # replicate interpolate_sky's SUPERSKY/SUPERSKY_ERROR construction so
    # combine_skies's unconditional eval_supersky() call has something to evaluate
    supersky, supererror = {}, {}
    for telescope_name in ("east", "west"):
        sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data = select_sky_fibers(
            rss, telescope=telescope_name
        )
        s_ssky, s_error, _, _, _, _, _ = fit_supersky(
            sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data
        )
        supersky[telescope_name] = s_ssky
        supererror[telescope_name] = s_error

    superskies = rss.stack_supersky([
        (wave, *supersky["east"].tck, "east"),
        (wave, *supersky["west"].tck, "west"),
    ])
    supererrors = rss.stack_supersky([
        (wave, *supererror["east"].tck, "east"),
        (wave, *supererror["west"].tck, "west"),
    ])
    rss.set_supersky(superskies)
    rss.set_supersky_error(supererrors)

    rss.writeFitsData(out_rss)

    return wave, line_center


def test_combine_skies_scimed_preserves_emission_line(tmp_path):
    """Regression test for the bug fixed alongside issue #250: the SCIMED
    flux-calibration sky in combine_skies used to take
    np.nanmedian(rss._data[sci_idx, :], axis=0), a per-pixel-*column* median,
    before the RSS is wavelength rectified. Since every fiber has its own
    native wavelength-to-pixel mapping at that stage, this mixed flux from
    different wavelengths and washed out anything as narrow as a night-sky
    emission line while leaving the smooth continuum intact.
    """
    in_rss = str(tmp_path / "in_sci.fits")
    out_rss = str(tmp_path / "out_sci.fits")
    wave, line_center = _make_sci_sky_rss(in_rss)

    combine_skies(in_rss=in_rss, out_rss=out_rss, sky_weights=None)

    rss = RSS.from_file(out_rss)
    assert rss._header.get("SKYSRC") == "SCIMED"

    fiber = 0
    fwave = wave[fiber]
    sky = rss._sky[fiber]

    line_idx = np.argmin(np.abs(fwave - line_center))
    peak = np.nanmax(sky[line_idx - 3:line_idx + 4])
    continuum = np.nanmedian(np.concatenate([sky[:line_idx - 50], sky[line_idx + 50:]]))

    # a per-column (wavelength-unaware) median would leave peak ~= continuum;
    # a correctly wavelength-aligned median recovers most of the injected
    # ~50x peak-to-continuum line.
    assert peak / continuum > 10


def test_combine_skies_sky_east_west_independent_of_fluxcal_sky(tmp_path):
    """Regression test for a second bug found alongside the one above: SKY_EAST/
    SKY_WEST used to be aliased to the same array as the flux-calibration sky
    (sky_w = sky_e), so the standard-star exposure-time rescaling loop -- which
    scales sky_east and sky_west separately, each with its own `*=` -- silently
    double-applied the correction whenever both names pointed at the same
    memory. SKY_EAST/SKY_WEST must be their own independent arrays, computed
    from eval_supersky() the same way as on master, regardless of the
    flux-calibration sky method.

    Checks array identity on the in-memory RSS returned by combine_skies
    directly -- a round trip through FITS (RSS.from_file) always produces
    fresh, distinct arrays per extension, so it would never expose aliasing
    even if the bug were reintroduced.
    """
    in_rss = str(tmp_path / "in_sci.fits")
    out_rss = str(tmp_path / "out_sci.fits")
    _make_sci_sky_rss(in_rss)

    rss, _ = combine_skies(in_rss=in_rss, out_rss=out_rss, sky_weights=None)

    assert rss._sky_east is not rss._sky_west
    assert rss._sky is not rss._sky_east
    assert rss._sky is not rss._sky_west
