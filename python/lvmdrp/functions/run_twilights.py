# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 4, 2024
# @Filename: run_twilights.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from __future__ import annotations

from typing import Tuple, List
from copy import deepcopy as copy
import numpy as np
from astropy.table import Table
from scipy import interpolate
from scipy.optimize import least_squares
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

import itertools as it
import bottleneck as bn
from lvmdrp import log
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS, lvmFrame
from lvmdrp.core.fluxcal import butter_lowpass_filter
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.plot import plt, create_subplots, display_ifu, save_fig
from lvmdrp import main as drp
from astropy import wcs
from astropy.io import fits
from astropy.stats import biweight_location
from astropy.visualization import simple_norm


MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
SLITMAP = Table(drp.fibermap.data)


class lvmFlat(lvmFrame):
    """lvmFlat class"""

    def __init__(self, data=None, error=None, mask=None,
                 cent_trace=None, width_trace=None, wave_trace=None, lsf_trace=None,
                 header=None, slitmap=None, superflat=None, **kwargs):
        lvmFrame.__init__(self, data=data, error=error, mask=mask,
                     cent_trace=cent_trace, width_trace=width_trace,
                     wave_trace=wave_trace, lsf_trace=lsf_trace,
                     header=header, slitmap=slitmap, superflat=superflat)

        self._blueprint = dp.load_blueprint(name="lvmFlat")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)


def mkifuimage(
    x, y, flux, fibid, posang=0, RAobs=0, DECobs=0,
    platescale=112.36748321030637, # Focal plane platescale in "/mm
    pscale=0.01 # IFU image pixel scale in mm/pix
):

    # Create fiber image
    rspaxel=35.3/platescale/2 # spaxel radius in mm assuming 35.3" diameter chromium mask
    npix=flux.size # size of IFU image
    ima=np.zeros((npix,npix))+np.nan
    xima=x*pscale # x coordinate in mm of each pixel in image
    yima=y*pscale # y coordinate in mm of each pixel in image
    for i in range(len(flux)):
        sel=(xima-x[i])**2+(yima-y[i])**2<=rspaxel**2
        ima[sel]=flux[i]
    # flag CRPIX for visual reference
    ima[int(npix/2), int(npix/2)]=0

    # Create WCS for IFU image
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [int(npix/2)+1, int(npix/2)+1]
    skypscale=pscale*platescale/3600 # IFU image pixel scale in deg/pix
    posangrad=posang*np.pi/180
    w.wcs.cd=np.array([[skypscale*np.cos(posangrad), -1*skypscale*np.sin(posangrad)],[-1*skypscale*np.sin(posangrad), -1*skypscale*np.cos(posangrad)]])
    w.wcs.crval = [RAobs,DECobs]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()

    # create image
    hdu = fits.PrimaryHDU(ima, header=header)

    return ima, hdu


def remove_field_gradient(in_hflat, out_gflat, wrange, deg=1, display_plots=False):

    fflat = RSS.from_file(in_hflat)
    channel = fflat._header["CCD"]

    x, y, flux, grad_model, grad_model_e, grad_model_w, grad_model_s = fflat.fit_field_gradient(wrange, poly_deg=deg)

    flux_c = flux / grad_model
    flux_med = bn.nanmedian(flux)
    flux_std = np.nanstd(flux)

    fig, axs = create_subplots(to_display=display_plots, nrows=1, ncols=3, figsize=(15,6), sharex=True, sharey=True, layout="constrained")
    sc_or = axs[0].scatter(x, y, s=40, marker="H", vmin=flux_med-4*flux_std, vmax=flux_med+4*flux_std, lw=0, c=flux, cmap="rainbow")
    sc_gr = axs[1].scatter(x, y, s=40, marker="H", lw=0, c=grad_model, cmap="coolwarm", norm=simple_norm(grad_model, stretch="log"))

    axi_or = inset_axes(
        axs[0],
        width="80%",  # width: 50% of parent_bbox width
        height="2%",  # height: 5%
        loc="upper right",
    )
    axi_or.tick_params(labelleft=False, labelsize="x-small")
    fig.colorbar(sc_or, cax=axi_or, orientation="horizontal")

    ax_gr = inset_axes(
        axs[1],
        width="80%",  # width: 50% of parent_bbox width
        height="2%",  # height: 5%
        loc="upper right",
    )
    ax_gr.tick_params(labelleft=False, labelsize="x-small")
    fig.colorbar(sc_gr, cax=ax_gr, orientation="horizontal")

    axs[2].scatter(x, y, s=40, marker="H", vmin=flux_med-2*flux_std, vmax=flux_med+2*flux_std, lw=0, c=flux_c, cmap="rainbow")
    axs[0].set_title("Original", loc="left")
    axs[1].set_title("Gradient", loc="left")
    axs[2].set_title("Original / Gradient", loc="left")
    fig.suptitle(f"Gradient field fitting for {channel = }")
    save_fig(fig, out_gflat, to_display=display_plots, figure_path="qa", label="field_gradient")

    telescope = fflat._slitmap["telescope"]
    rss = fflat._data[telescope=="Sci"]
    rss_e = fflat._data[telescope=="SkyE"]
    rss_w = fflat._data[telescope=="SkyW"]
    rss_s = fflat._data[telescope=="Spec"]

    rss = rss / grad_model[:, None]
    rss_e = rss_e / grad_model_e[:, None]
    rss_w = rss_w / grad_model_w[:, None]
    rss_s = rss_s / grad_model_s[:, None]

    new_fflat = copy(fflat)
    new_fflat._data[telescope == "Sci"] = rss
    new_fflat._data[telescope == "SkyE"] = rss_e
    new_fflat._data[telescope == "SkyW"] = rss_w
    new_fflat._data[telescope == "Spec"] = rss_s
    new_fflat.writeFitsData(out_gflat)


def fit_continuum(spectrum: Spectrum1D, mask_bands: List[Tuple[float,float]],
                  median_box: int, niter: int, threshold: Tuple[float,float]|float, knots: int|np.ndarray):
    """Fit a continuum to a spectrum using a spline interpolation

    Given a spectrum, this function fits a continuum using a spline
    interpolation and iteratively masks outliers below a given threshold of the
    fitted spline.

    Parameters
    ----------
    spectrum : lvmdrp.core.spectrum1d.Spectrum1D
        Spectrum to fit the continuum
    mask_bands : list
        List of wavelength bands to mask
    median_box : int
        Size of the median filter box
    niter : int
        Number of iterations to fit the continuum
    threshold : float or tuple of floats
        Threshold to mask outliers, if tuple, the first element is the lower
        threshold and the second element is the upper threshold
    knots : int or np.ndarray[float]
        Number of knots or actual knots to use in the spline fitting

    Returns
    -------
    best_continuum : np.ndarray
        Best fit continuum
    continuum_models : list
        List of continuum models for each iteration
    masked_pixels : np.ndarray
        Masked pixels in all iterations
    tck : tuple
        Spline parameters
    """
    # early return if no good pixels
    continuum_models = []
    masked_pixels = copy(spectrum._mask)
    good_pix = ~masked_pixels
    if good_pix.sum() == 0:
        return np.ones_like(spectrum._wave) * np.nan, continuum_models, masked_pixels, np.array([])

    # define main arrays
    wave = spectrum._wave[good_pix]
    data = spectrum._data[good_pix]

    # define spline fitting parameters
    if isinstance(knots, int):
        nknots = knots
        knots = np.linspace(wave[wave.size // nknots], wave[-1 * wave.size // nknots], nknots)
    elif isinstance(knots, (list, tuple, np.ndarray)):
        knots = np.asarray(knots)
    else:
        raise TypeError(f"invalid type for {knots = }, {type(knots)}")
    if mask_bands:
        mask = np.ones_like(knots, dtype="bool")
        for iwave, fwave in mask_bands:
            mask[(iwave<=knots)&(knots<=fwave)] = False
        knots = knots[mask]

    # fit first spline
    tck = interpolate.splrep(wave, data, t=knots, task=-1)
    spline = interpolate.splev(spectrum._wave, tck)

    # iterate to mask outliers and update spline
    if threshold is not None and isinstance(threshold, (float, int)):
        threshold = (threshold, np.inf)
    for i in range(niter):
        residuals = spline - spectrum._data
        mask = spline - threshold[0]*np.nanstd(residuals) > spectrum._data
        mask |= spline + threshold[1]*np.nanstd(residuals) < spectrum._data

        # add new outliers to mask
        masked_pixels |= mask

        # update spline
        tck = interpolate.splrep(spectrum._wave[~masked_pixels], spectrum._data[~masked_pixels], t=knots, task=-1)
        new_spline = interpolate.splev(spectrum._wave, tck)
        continuum_models.append(new_spline)
        if np.mean(np.abs(new_spline - spline) / spline) <= 0.01:
            break
        else:
            spline = new_spline

    best_continuum = continuum_models.pop(-1)
    return best_continuum, continuum_models, masked_pixels, tck


def fit_fiberflat(in_twilight: str, out_flat: str, out_twilight: str,
                  remove_gradient: bool = True, niter: int = 4,
                  plot_fibers: List[int] = [0,300,600,900,1200,1400,1700],
                  display_plots: bool = False) -> RSS:
    """Fit fiber flat field given a spectrograph-stacked and rectified twilight exposure

    Parameters
    ----------
    in_twilight : str
        Path to twilight exposure
    out_flat : str
        Path to output fiberflat
    out_twilight : str
        Path to flat-fielded twilight
    remove_gradient : bool, optional
        Whether to fit and remove IFU field gradient or not, by default True
    niter : int, optional
        Number of iterations while fitting gradient, by default 4
    plot_fibers : list[int], optional
        List of fiber indices to plot, by default [0,300,600,900,1200,1400,1700]
    display_plots : bool, optional
        Display plots, by default False

    Returns
    -------
    new_flat : lvmdrp.core.rss.RSS
        Fitted flat field
    """

    twilight = RSS.from_file(in_twilight)
    channel = twilight._header["CCD"][0]
    expnum = twilight._header["EXPOSURE"]
    unit = twilight._header["BUNIT"]

    niter = max(1, niter)
    for i in range(niter):
        median_fiber = biweight_location(twilight._data, axis=0, ignore_nan=True)
        mask = np.isfinite(median_fiber)
        median_fiber = np.interp(twilight._wave, twilight._wave[mask], median_fiber[mask])
        new_flat = copy(twilight)
        new_flat._data = twilight._data / median_fiber[None]
        # for ifiber in range(new_flat._fibers):
        #     f = new_flat.getSpec(ifiber)
        #     if f._mask.all():
        #         continue

        #     # get selection of good pixels
        #     mask = np.isfinite(f._data)

        #     # TODO: implement this as a separated function, skipping boundaries of fibers
        #     # correct median fiber to current fiber wavelength and normalize
        #     # _, shift, _ = _cross_match_float(ref_spec=median_fiber, obs_spec=f._data, stretch_factors=[1.0], shift_range=[-5,+5])
        #     # f._data /= np.interp(f._wave, f._wave+shift, median_fiber)

        #     # # first filtering of high-frequency features
        #     # f._data[mask] = butter_lowpass_filter(f._data[mask], 0.1, 2)
        #     # new_flat._data[ifiber] = f._data

        #     # # further smoothing of remaining unwanted features
        #     # tck = interpolate.splrep(f._wave[mask], f._data[mask], s=0.1)
        #     # new_flat._data[ifiber] = interpolate.splev(f._wave, tck)

        #     # GET THE RAW FLATFIELD
        #     new_flat._data[ifiber] = f._data#np.interp(f._wave, f._wave[mask], f._data[mask])

        # reset pixel mask
        new_flat._mask[:] = new_flat._mask.all(axis=1)[:, None]

        flat_twilight = copy(twilight)
        flat_twilight._data = flat_twilight._data / new_flat._data

        # _, axs = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True, layout="constrained")
        # x, y, flux, grad_model, grad_model_e, grad_model_w, grad_model_s = flat_twilight.fit_field_gradient(wrange=twilight._wave[[500, 700]], poly_deg=1)
        # # norm = simple_norm(flux, stretch="linear", min_percent=5, max_cut=0.5e-13)
        # sc = axs[0].scatter(x, y, s=90, lw=0, c=flux, cmap="coolwarm")
        # plt.colorbar(sc)
        # x, y, flux, grad_model, grad_model_e, grad_model_w, grad_model_s = flat_twilight.fit_field_gradient(wrange=twilight._wave[[2000, 2300]], poly_deg=1)
        # sc = axs[1].scatter(x, y, s=90, lw=0, c=flux, cmap="coolwarm")
        # plt.colorbar(sc)
        # x, y, flux, grad_model, grad_model_e, grad_model_w, grad_model_s = flat_twilight.fit_field_gradient(wrange=twilight._wave[[3500, 3700]], poly_deg=1)
        # sc = axs[2].scatter(x, y, s=90, lw=0, c=flux, cmap="coolwarm")
        # plt.colorbar(sc)
        # plt.show()
        telescope = twilight._slitmap["telescope"]
        rss = twilight._data[telescope=="Sci"]
        rss_e = twilight._data[telescope=="SkyE"]
        rss_w = twilight._data[telescope=="SkyW"]
        rss_s = twilight._data[telescope=="Spec"]

        # rss = rss / grad_model[:, None]
        # rss_e = rss_e / grad_model_e[:, None]
        # rss_w = rss_w / grad_model_w[:, None]
        # rss_s = rss_s / grad_model_s[:, None]

        twilight._data[telescope == "Sci"] = rss
        twilight._data[telescope == "SkyE"] = rss_e
        twilight._data[telescope == "SkyW"] = rss_w
        twilight._data[telescope == "Spec"] = rss_s

    fig = plt.figure(figsize=(15, 10), layout="constrained")
    gs = GridSpec(3, 15, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax_1 = fig.add_subplot(gs[2, :-3])
    ax_2 = fig.add_subplot(gs[2, -3:])
    axs = [ax1, ax2, ax_1, ax_2]

    # fig, axs = create_subplots(to_display=display_plots, nrows=3, figsize=(15,7), layout="constrained")
    fig.suptitle(f"Flat fielding for {channel = }, {expnum = }")
    axs[0].set_title("Twilight", loc="left")
    axs[1].set_title(f"Flat field for fibers {','.join(map(str, plot_fibers))}", loc="left")
    median_fiber = biweight_location(twilight._data, axis=0, ignore_nan=True)
    for ifiber in plot_fibers:
        # _, shift, _ = _cross_match_float(ref_spec=median_fiber, obs_spec=np.nan_to_num(twilight._data[ifiber]), stretch_factors=[1.0], shift_range=[-5,+5])
        ln, = axs[0].step(twilight._wave, twilight._data[ifiber], lw=1, where="mid")
        axs[1].step(twilight._wave, twilight._data[ifiber] / median_fiber, lw=1, color=ln.get_color(), where="mid", alpha=0.5)
        axs[1].plot(new_flat._wave, new_flat._data[ifiber], lw=1, color=ln.get_color(), label=ifiber)
    axs[0].step(twilight._wave, median_fiber, lw=2, color="0.2", where="mid", label="median fiber")
    axs[0].legend(loc=1, frameon=False)
    axs[1].legend(loc=1, frameon=False, title="Fiber Idx", ncols=7)
    axs[1].set_xlabel("Wavelength (Angstrom)")
    axs[0].set_ylabel(f"Counts ({unit})")
    axs[1].set_ylabel("Normalized counts")
    axs[1].set_xlim(*axs[0].get_xlim())

    ypixels = np.split(np.arange(flat_twilight._fibers) + 1, 3)
    median =  np.asarray(np.split(bn.nanmedian(flat_twilight._data, axis=1), 3))
    mu = bn.nanmedian(flat_twilight._data)
    median = (median / mu - 1) * 100
    axs[2].axhspan(-0.5, 0.5, lw=0, alpha=0.3, color="0.7")
    axs[2].axhline(0, ls="--", lw=1, color="0.7")
    colors = ["tab:blue", "tab:red", "tab:green"]
    for ispec in range(3):
        axs[2].step(ypixels[ispec], median[ispec], lw=1, where="mid", label=f"Spec. {ispec+1}", color=colors[ispec])
    axs[2].legend(loc=1, frameon=False, ncols=3)
    axs[2].set_xlabel("Fiber ID")
    axs[2].set_ylabel("Flat-fielded flat (%)")
    axs[2].set_xlim(1, twilight._fibers)

    x = (flat_twilight._data/flat_twilight._data[0]).ravel()
    axs[3].axvspan(np.nanmean(x) - np.nanstd(x), np.nanmean(x) + np.nanstd(x), lw=0, alpha=0.1, color="0.2")
    axs[3].hist(x, bins=1000, range=(0.95, 1.05), color="tab:orange")
    axs[3].tick_params(labelleft=False)
    axs[3].set_xlabel("Twilight / Twilight[0]")

    save_fig(fig, to_display=display_plots, product_path=out_flat, figure_path="qa", label="twilight_fiberflat")

    # write output fiberflat
    log.info(f"writing flat field to {out_flat}")
    new_flat.writeFitsData(out_flat)

    # write output faltfielded explosure
    log.info(f"writing flatfielded twilight to {out_twilight}")
    flat_twilight.writeFitsData(out_twilight)

    return new_flat


def to_native_wave(rss, wave=None):

    # get native wavelength grid or use the one given
    if wave is None and rss._wave_trace is not None:
        trace = TraceMask.from_coeff_table(rss._wave_trace)
        wave = trace.eval_coeffs()
    elif wave is not None:
        pass
    else:
        raise ValueError(f"missing wavelength trace information: {rss._wave_trace = }")

    new_rss = RSS(
        data=np.zeros((rss._fibers, wave.shape[1]), dtype="float32"),
        error=np.zeros((rss._fibers, wave.shape[1]), dtype="float32"),
        mask=np.zeros((rss._fibers, wave.shape[1]), dtype="bool"),
        wave=wave,
        cent_trace=rss._cent_trace,
        width_trace=rss._width_trace,
        wave_trace=rss._wave_trace,
        lsf_trace=rss._lsf_trace,
        slitmap=rss._slitmap,
        header=rss._header
    )

    # reset header keywords to match original wavelength grid state
    new_rss._header["WAVREC"] = False
    if "CRPIX1" in new_rss._header:
        del new_rss._header["CRPIX1"]
    if "CRVAL1" in new_rss._header:
        del new_rss._header["CRVAL1"]
    if "CDELT1" in new_rss._header:
        del new_rss._header["CDELT1"]
    if "CTYPE1" in new_rss._header:
        del new_rss._header["CTYPE1"]

    # interpolate data, error, mask and sky arrays from rectified grid to original grid
    for ifiber in range(rss._fibers):
        f = interpolate.interp1d(rss._wave, rss._data[ifiber], kind="linear", bounds_error=False, fill_value=np.nan)
        new_rss._data[ifiber] = f(wave[ifiber]).astype("float32")
        f = interpolate.interp1d(rss._wave, rss._error[ifiber], kind="linear", bounds_error=False, fill_value=np.nan)
        new_rss._error[ifiber] = f(wave[ifiber]).astype("float32")
        f = interpolate.interp1d(rss._wave, rss._mask[ifiber], kind="nearest", bounds_error=False, fill_value=1)
        new_rss._mask[ifiber] = f(wave[ifiber]).astype("bool")

    return new_rss


def create_lvmflat(in_twilight: str, out_lvmflat: str, in_fiberflat: str,
                   in_cents: List[str], in_widths: List[str],
                   in_waves: List[str], in_lsfs: List[str]) -> lvmFlat:
    """Creates lvmFlat product from given flat-fielded twilight and fiberflat

    This routine takes in a flatfielded twilight exposure and the used
    flatfield. These RSS objects are expected to be spectrograph-stacked and
    rectified. In order to build an lvmFlat object the fiber and wavelength
    traces are also needed. The final lvmFlat object will be in the native
    pixel grid.

    Parameters
    ----------
    in_twilight : str
        Path to flat-fielded twilight
    out_lvmflat : str
        Path to output lvmFlat product
    in_fiberflat : str
        Path to fiberflat
    {in_cents, in_widths} : List[str]
        Paths to fiber centroids/widths for corresponding twilight spectrograph channel
    {in_waves, in_lsfs} : List[str]
        Paths to wavelengths/LSFs for corresponding twilight spectrograph channel

    Returns
    -------
    lvmflat : lvmdrp.functions.run_twilights.lvmFlat
        lvmFlat product
    """

    # load flatfielded twilight
    twilight = RSS.from_file(in_twilight)
    # load flatfield
    fflat = RSS.from_file(in_fiberflat)

    # load fiber and wavelength traces
    mcent = TraceMask.from_spectrographs(*[TraceMask.from_file(master_cent) for master_cent in in_cents])
    mwidth = TraceMask.from_spectrographs(*[TraceMask.from_file(master_width) for master_width in in_widths])
    mwave = TraceMask.from_spectrographs(*[TraceMask.from_file(master_wave) for master_wave in in_waves])
    mlsf = TraceMask.from_spectrographs(*[TraceMask.from_file(master_lsf) for master_lsf in in_lsfs])

    # build lvmFlat
    twilight.set_wave_trace(mwave)
    twilight.set_lsf_trace(mlsf)
    twilight = to_native_wave(twilight)
    fflat.set_wave_trace(mwave)
    fflat.set_lsf_trace(mlsf)
    fflat = to_native_wave(fflat)
    lvmflat = lvmFlat(data=twilight._data, error=twilight._error, mask=twilight._mask, header=twilight._header,
                      cent_trace=mcent, width_trace=mwidth,
                      wave_trace=mwave, lsf_trace=mlsf,
                      superflat=fflat._data, slitmap=twilight._slitmap)
    lvmflat.writeFitsData(out_lvmflat)

    return lvmflat


def combine_twilight_sequence(in_fiberflats: List[str], out_fiberflat: str,
                              in_waves: List[str], in_lsfs: List[str]) -> RSS:
    """Combine twilight exposures into a single RSS object

    Given a list of RSS objects of fiberflats from twilight exposures, this
    function combines them into a single RSS object by averaging the fiber
    throughput of all non-standard fibers and putting the standard fibers in
    their respective positions.

    Parameters
    ----------
    in_fiberflats : list[str]
        List of paths to individual fiberflat exposures
    out_fiberflat : str
        Output path to master fiberflat
    in_waves : list[str]
        List of wavelength solution path for each channel
    in_lsfs : list[str]
        Lost of LSF solution path for each channel


    Returns
    -------
    mflat : RSS
        Master twilight flat
    """

    fflats = [RSS.from_file(in_fiberflat) for in_fiberflat in in_fiberflats]

    # combine RSS exposures using an average
    mflat = RSS(data=np.zeros_like(fflats[0]._data), error=np.zeros_like(fflats[0]._error), mask=np.ones_like(fflats[0]._mask, dtype=bool),
                header=copy(fflats[0]._header), wave=copy(fflats[0]._wave), lsf=copy(fflats[0]._lsf), slitmap=copy(fflats[0]._slitmap))
    # select non-std fibers
    fibermap =  mflat._slitmap
    select_allstd = fibermap["telescope"] == "Spec"
    # select_nonstd = ~select_allstd
    for i, fflat in enumerate(fflats):
        # get exposed standard fiber ID
        fiber_id = fflat._header.get("CALIBFIB")
        if fiber_id is None:
            snrs = bn.nanmedian(fflat._data / fflat._error, axis=1)
            select_nonexposed = snrs < 50
            #plt.figure(figsize=(15,5))
            #plt.plot(snrs[select_allstd])
            #ids_std = fibermap[select_allstd]["orig_ifulabel"]
            #idx_std = np.arange(ids_std.size)
            #plt.gca().set_xticks(idx_std)
            #plt.gca().set_xticklabels(ids_std)
        else:
            select_nonexposed = fibermap["orig_ifulabel"] != fiber_id

        # put std fibers in the right position
        fflat._data[select_allstd&select_nonexposed] = np.nan
        fflat._error[select_allstd&select_nonexposed] = np.nan
        fflat._mask[select_allstd&select_nonexposed] = True

    mflat = copy(fflats[0])
    mflat._data = biweight_location([fflat._data for fflat in fflats], axis=0, ignore_nan=True)
    mflat._error = np.sqrt(biweight_location([fflat._error**2 for fflat in fflats], axis=0, ignore_nan=True)) / len(fflats)

    # mask invalid pixels
    mflat._mask |= np.isnan(mflat._data) | (mflat._data <= 0) | np.isinf(mflat._data)
    mflat._mask |= np.isnan(mflat._error) | (mflat._error <= 0) | np.isinf(mflat._error)

    # interpolate masked fibers if any remaining
    mflat = mflat.interpolate_data(axis="X")
    mflat = mflat.interpolate_data(axis="Y")

    mflat.set_wave_trace(TraceMask.from_spectrographs(*[TraceMask.from_file(in_wave) for in_wave in in_waves]))
    mflat.set_lsf_trace(TraceMask.from_spectrographs(*[TraceMask.from_file(in_lsf) for in_lsf in in_lsfs]))
    mflat = to_native_wave(mflat)
    mflat._error = None
    mflat._mask = None
    mflat.writeFitsData(out_fiberflat, replace_masked=False)

    return mflat

def _reference_fiber(rss, ref_kind):

    if isinstance(rss, RSS):
        data = rss._data.copy()
    elif isinstance(rss, np.ndarray):
        data = np.atleast_2d(rss).T
    else:
        raise TypeError(f"Invalid type for `rss`: {type(rss)}. Expected lvmdrp.core.rss.RSS or numpy array")

    if callable(ref_kind):
        ref_fiber = ref_kind(data, axis=0)
    elif isinstance(ref_kind, int):
        ref_fiber = data[ref_kind, :]
    else:
        raise TypeError(f"Invalid type for `ref_kind`: {type(ref_kind)}. Expected an integer or a callable(x, axis)")
    return ref_fiber

def get_flatfield(rss, ref_kind=lambda x: biweight_location(x, ignore_nan=True), norm_column=None, norm_kind=np.nanmean):
    ref_fiber = _reference_fiber(rss, ref_kind=ref_kind)

    flatfield = rss / ref_fiber
    if norm_column is not None:
        if callable(norm_kind):
            normalization = norm_kind(flatfield._data[:, norm_column], axis=0)
        elif isinstance(norm_kind, int):
            normalization = flatfield._data[norm_kind, norm_column]
        else:
            raise TypeError(f"Invalid type for `norm_kind`: {type(norm_kind)}. Expected an integer or a callable(x, axis)")
    else:
        normalization = 1.0

    flatfield /= normalization
    return flatfield, ref_fiber, normalization

def get_flatfield_sequence(rsss, ref_kind=lambda x: biweight_location(x, ignore_nan=True), norm_column=None, norm_kind=np.nanmean):
    flatfields = []
    ref_fibers = np.zeros((len(rsss), rsss[0]._pixels.size))
    normalizations = np.zeros(len(rsss))
    for i, rss_ in enumerate(rsss):
        flatfield, ref_fibers[i], normalizations[i] = get_flatfield(rss_, ref_kind=ref_kind, norm_column=norm_column, norm_kind=norm_kind)
        flatfields.append(flatfield)
    return flatfields, ref_fibers, normalizations

def normalize_spec(rss, cwave, dwave=6, norm_stat=np.nanmean):
    slitmap = rss._slitmap
    sp1_sel = slitmap["spectrographid"] == 1
    sp2_sel = slitmap["spectrographid"] == 2
    sp3_sel = slitmap["spectrographid"] == 3

    hw = dwave // 2
    wave_sel = (cwave-hw < rss._wave)&(rss._wave < cwave+hw)
    data = rss._data.copy()
    data[:, ~wave_sel] = np.nan
    sp1_norm = norm_stat(data[sp1_sel])
    sp2_norm = norm_stat(data[sp2_sel])
    sp3_norm = norm_stat(data[sp3_sel])

    rss_n = copy(rss)
    rss_n._data[sp1_sel] /= sp1_norm
    rss_n._data[sp2_sel] /= sp2_norm
    rss_n._data[sp3_sel] /= sp3_norm
    return rss_n

def fit_lines(spec, cwaves, dwave=6, display_plots=False):

    cwaves_ = np.atleast_1d(cwaves) if np.isscalar(cwaves) else cwaves

    if spec._lsf is None:
        fwhm_guess = 2.5
    else:
        fwhm_guess = np.nanmean(np.interp(cwaves_, spec._wave, spec._lsf))

    if display_plots:
        fig, axs = plt.subplots(1, len(cwaves_), figsize=(5, 5*len(cwaves_)), layout="constrained")
        axs = np.atleast_1d(axs)
    else:
        axs = None

    flux, sky_wave, fwhm, bg = spec.fitSepGauss(cwaves_, dwave,
                                                fwhm_guess, 0.0,
                                                [0, np.inf],
                                                [-2.5, 2.5],
                                                [fwhm_guess - 1.5, fwhm_guess + 1.5],
                                                [0.0, np.inf],
                                                axs=axs)
    return flux, sky_wave, fwhm, bg

def fit_lines_slit(rss, cwaves, dwave=6, return_xy=False, select_fibers=None, display_plots=False):

    cwaves_ = np.atleast_1d(cwaves)

    if isinstance(select_fibers, str):
        if select_fibers not in rss._slitmap["telescope"]:
            raise ValueError(f"Invalid value for `select_fibers`: {select_fibers}. Expected either 'Sci', 'SkyE', 'SkyW' or 'Spec'")
        select_fibers = rss._slitmap["telescope"] == select_fibers
        log.info(f"selecting {select_fibers.sum()} fibers")
    elif isinstance(select_fibers, np.ndarray) and select_fibers.size == rss._fibers and select_fibers.dtype == bool:
        log.info(f"selecting {select_fibers.sum()} fibers")
    elif select_fibers is None:
        log.info(f"selecting all {rss._fibers} fibers")
        select_fibers = np.ones(rss._fibers, dtype="bool")
    else:
        raise TypeError(f"Invalid type for `select_fibers`: {type(select_fibers)}. Expected either None, string or boolean array matching number of fibers in `rss`")

    flux_slit = np.zeros((rss._fibers, cwaves_.size)) + np.nan
    for ifiber in range(rss._fibers):
        spec = rss[ifiber]
        if not select_fibers[ifiber] or spec._mask.all():
            continue

        try:
            flux, _, _, _ = fit_lines(spec, cwaves_, dwave=dwave, display_plots=display_plots)
        except ValueError as e:
            log.error(f"error while fitting fiber {ifiber}: {e}")
        flux_slit[ifiber] = flux

    if return_xy:
        return flux_slit.squeeze(), rss._slitmap["xpmm"].data, rss._slitmap["ypmm"].data
    return flux_slit.squeeze()

def ifu_factors(factors, fibgroups):
    iid, fid = min(fibgroups), max(fibgroups)
    ifu = np.ones_like(fibgroups, dtype="float")
    for spid in range(iid, fid+1):
        ifu[fibgroups == spid] *= factors[spid-1]
    return ifu

def ifu_gradient(coeffs, x, y):
    ncoeffs = len(coeffs)
    order = int(np.sqrt(ncoeffs))
    ij = it.product(range(order), repeat=2)

    G = np.zeros((x.size, ncoeffs))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    ifu = np.dot(G, coeffs)
    return ifu

def ifu_joint_model(pars, x, y, fibgroups):
    iid, fid = min(fibgroups), max(fibgroups)
    nids = fid - iid + 1
    coeffs, factors = pars[:-nids], pars[-nids:]

    # get IFU gradient model
    gradient_model = ifu_gradient(coeffs=coeffs, x=x, y=y)
    # get IFU factors model
    factors_model = ifu_factors(factors=factors, fibgroups=fibgroups)
    # joint model
    model = gradient_model * factors_model
    return model

def residual(pars, x, y, z, fibgroups):
    model = ifu_joint_model(pars, x, y, fibgroups)
    return model - z

def display_ifu_gradient_fit(x, y, z, fibgroups, coeffs, factors, telescope="Sci",
                             labels=["Spec. factors", "Gradient model", "Original image", "Factor-corrected image", "Fully corrected image"]):
    gradient_model = ifu_gradient(coeffs=coeffs, x=x, y=y)
    factors_model = ifu_factors(factors=factors, fibgroups=fibgroups)

    model = gradient_model * factors_model
    gradient_model_ = gradient_model.copy()
    gradient_model_ /= gradient_model_.mean()
    factors_model_ = factors_model.copy()
    factors_model_ /= factors_model_.mean()

    ifus = [factors_model_, gradient_model_, z, z/factors_model_, z/model]

    fig, axs = plt.subplots(1, 5, figsize=(15, 4), sharex=True, sharey=True, layout="constrained")
    fig.supxlabel("X (spaxel)", fontsize="xx-large")
    fig.supylabel("Y (spaxel)", fontsize="xx-large")
    for i in range(len(ifus)):
        axs[i].set_title(labels[i], loc="left", fontsize="large")
        display_ifu(x=x, y=y, z=ifus[i], ax=axs[i], marker_size=17)

    return fig, axs

def fit_ifu_gradient(rss, cwave, dwave=6, guess_coeffs=[1,2,3,0], fib_groupby="spec", coadd_method="average", display_plots=False):

    if fib_groupby == "spec":
        groups = [1,2,3]
    elif fib_groupby == "quad":
        groups = [1,2,3,4,5,6]
    fibgroups = np.repeat(groups, rss._fibers // len(groups))

    if coadd_method == "average":
        z, x, y = rss.coadd_flux(cwave=cwave, dwave=dwave, return_xy=True, telescope="Sci")
    elif coadd_method == "fit":
        z, x, y = fit_lines_slit(rss=rss, cwaves=cwave, return_xy=True, select_fibers="Sci")
    z /= np.nanmean(z)

    # mask invalid spaxels and spaxels from IFUs other than Sci's
    mask = np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    fibgroups = fibgroups[mask]

    # define guess and boundary values
    guess_factors = len(groups) * [1]
    guess = guess_coeffs + guess_factors
    bound_lower = len(guess_coeffs) * [-np.inf] + len(guess_factors) * [0.0]
    bound_upper = len(guess_coeffs) * [+np.inf] + len(guess_factors) * [1.0]
    fit = least_squares(residual, x0=guess, args=(x, y, z, fibgroups), bounds=(bound_lower, bound_upper))

    coeffs = fit.x[:len(guess_coeffs)]
    factors = fit.x[len(guess_coeffs):]

    if display_plots:
        display_ifu_gradient_fit(x, y, z, fibgroups, coeffs=coeffs, factors=factors)

    return x, y, z, coeffs, factors

def remove_ifu_gradient(rss, coeffs, factors=None):

    if factors is None:
        factors = np.ones(rss._fibers, dtype="float")
    else:
        factors = np.repeat(factors, rss._fibers//factors.size)

    rss_corr = copy(rss)

    telescopes = ["Sci", "SkyE", "SkyW", "Spec"]
    for tel in telescopes:
        tel_select = rss._slitmap["telescope"] == tel
        slitmap = rss._slitmap[tel_select]
        x, y = slitmap["xpmm"].data, slitmap["ypmm"].data

        gradient = ifu_gradient(coeffs=coeffs, x=x, y=y)
        gradient *= factors[tel_select]

        rss_corr._data[slitmap["fiberid"]-1] /= gradient[:, None]
        if rss_corr._error is not None:
            rss_corr._error[slitmap["fiberid"]-1] /= gradient[:, None]

    return rss_corr

def fit_flatfield(twilights, ref_kind=600, norm_cwave=None, display_plots=False):
    """Creates a master fiber flatfield given a set of twilight exposures

    The following steps are followed:
        - For each exposure:
            * Normalize by the chosen reference fiber
            * Normalize by spectrograph at `norm_cwave`
        - Combine the resulting flatfields into a master
        - For each exposure:
            * Apply flatfield to each twilight exposure
            * Fit gradient and correct each telescope IFU by it
        - Combine gradient corrected flatfields into final master

    **NOTE:** This algorithm proposed by Guillermo Blanc, will produce a
    flatfield that fixes the spectrograph shutter timing issue at the cost of a
    flatfield that fully accounts for spectrograph to spectrograph throughput
    variations. As a consequence the resulting flatfield will have to be
    corrected using the same sky line at `norm_cwave` extracted and measured
    during science reductions, where we expect shutter timing issues to be
    within 1%.

    **NOTE:** This same procedure could be applied to dome flats, provided we
    can use >80s exposures to reliably measure the spectrograph to spectrograph
    throughput. LDLS seem to be the best option.

    Parameters
    ----------
    twilights : list[lvmdrp.core.rss.RSS]
        List of twilight exposures
    ref_kind : int|callable, optional
        Position of the reference fiber in RSS or a callable to produce one, by default 600
    norm_cwave : int|None, optional
        Normalization wavelength, by default None

    Returns
    -------
    lvmdrp.core.rss.RSS
        Master fiber flatfield
    """

    log.info(f"calculating raw flatfields out of {len(twilights)} exposures")
    flats, ref_fibers, normalizations = get_flatfield_sequence(rsss=twilights, ref_kind=ref_kind)

    if display_plots:
        fig, ax = plt.subplots(figsize=(14,5), layout="constrained")
        ax.step(flats[0]._wave, ref_fibers.T, where="mid", lw=1)
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel(f"Counts ({flats[0]._header['BUNIT']})")
        ax.set_title("Reference fiber", loc="left", fontsize="xx-large")

    log.info(f"fitting and normalizing IFU gradient at {norm_cwave = :.2f}")
    twilights_g = []
    for twilight, flat in zip(twilights, flats):
        log.info(f"processing {twilight._header['IMAGETYP']} exposure {twilight._header['EXPOSURE']}")
        # fit gradient with spectrograph normalizations (make n-iterations of this or stop when gradient is <1% across) ------
        x, y, z, coeffs, factors = fit_ifu_gradient(flat, fib_groupby="spec", cwave=norm_cwave, display_plots=display_plots)
        log.info(f" fitted spectrograph factors = {np.round(factors, 4)}")
        log.info(f" fitted gradient coeffs      = {np.round(coeffs, 7)}")
        # apply gradient correction
        twilight_g = remove_ifu_gradient(twilight, coeffs=coeffs, factors=factors)
        twilights_g.append(twilight_g)
        # --------------------------------------------------------------------------------------------------------------------

    flats_g, _, _ = get_flatfield_sequence(rsss=twilights_g, ref_kind=ref_kind)

    log.info("calculating gradient-corrected combined flatfield")
    mflat = RSS()
    mflat.combineRSS(flats_g, method="biweight")

    mflat.interpolate_data(axis="X")
    mflat.interpolate_data(axis="Y")

    return mflat, flats, flats_g

def _choose_sky(rss):
    telescope = "SkyE" if abs(rss._header["WAVE HELIORV_SKYE"]) < abs(rss._header["WAVE HELIORV_SKYW"]) else "SkyW"
    return telescope

def fit_skyline_flatfield(sciences, mflat, ref_kind, norm_cwave, norm_fibers=None, display_plots=False):

    if isinstance(sciences, list):
        log.info(f"fitting sky line correction using {len(sciences)} science frames")
        science = RSS()
        science.combineRSS([science / mflat for science in sciences], method="median")
    elif isinstance(sciences, RSS):
        log.info(f"fitting sky line correction using a single science exposure: {sciences}")
        science = sciences / mflat
    else:
        raise TypeError(f"Invalid type for `sciences`: {type(sciences)}. Valid types are lvmdrp.core.rss.RSS and list[lvmdrp.core.rss.RSS]")

    science.apply_pixelmask()

    log.info(f"measuring sky line {norm_cwave:.2f} Angstrom")
    skyline_slit = fit_lines_slit(science, cwaves=norm_cwave, norm_fibers=norm_fibers)

    log.info(f"calculating flatfield slit using {ref_kind = }")
    ref_skyline = _reference_fiber(skyline_slit, ref_kind=ref_kind)
    flatfield_slit = skyline_slit / ref_skyline


    log.info("calculating per-spectrograph flatfield correction")
    flatfield_corr = np.asarray([bn.nanmedian(flatfield_slit[science._slitmap["spectrographid"] == i+1]) for i in range(3)])
    log.info(f"resulting per-spectrograph corrections: {flatfield_corr.round(4) = }")

    if display_plots:
        x = mflat._slitmap["fiberid"]
        # y = (skyline_slit / flatfield_slit / np.repeat(flatfield_corr[:, None], 648))
        y = skyline_slit / np.repeat(flatfield_corr, 648)
        mu = np.nanmedian(y)
        y /= mu

        plt.figure(figsize=(14, 4), layout="constrained")
        plt.axhspan(0.95, 1.05, lw=0, color="0.7", alpha=0.5)
        plt.axhspan(0.98, 1.02, lw=0, color="0.7", alpha=0.5)
        plt.axhspan(0.99, 1.01, lw=0, color="0.7", alpha=0.5)
        plt.plot(x, y, ".-", lw=1)
        plt.ylim(0.92, 1.08)
        plt.xlabel("Fiber ID")
        plt.ylabel(f"Counts ({science._header['BUNIT']})")

    return skyline_slit, flatfield_corr
