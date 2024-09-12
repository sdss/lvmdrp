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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

import bottleneck as bn
from lvmdrp import log
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS, lvmFrame
from lvmdrp.core.fluxcal import butter_lowpass_filter
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.plot import plt, create_subplots, save_fig
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
        median_fiber = np.nanmedian(twilight._data, axis=0)
        mask = np.isfinite(median_fiber)
        median_fiber = np.interp(twilight._wave, twilight._wave[mask], median_fiber[mask])
        new_flat = copy(twilight)
        new_flat._data = twilight._data / median_fiber[None]
        for ifiber in range(new_flat._fibers):
            f = new_flat.getSpec(ifiber)
            if f._mask.all():
                continue

            # get selection of good pixels
            mask = np.isfinite(f._data)

            # TODO: implement this as a separated function, skipping boundaries of fibers
            # correct median fiber to current fiber wavelength and normalize
            # _, shift, _ = _cross_match_float(ref_spec=median_fiber, obs_spec=f._data, stretch_factors=[1.0], shift_range=[-5,+5])
            # f._data /= np.interp(f._wave, f._wave+shift, median_fiber)

            # first filtering of high-frequency features
            f._data[mask] = butter_lowpass_filter(f._data[mask], 0.1, 2)
            new_flat._data[ifiber] = f._data

            # further smoothing of remaining unwanted features
            tck = interpolate.splrep(f._wave[mask], f._data[mask], s=0.1)
            new_flat._data[ifiber] = interpolate.splev(f._wave, tck)

        # reset pixel mask
        new_flat._mask[:] = new_flat._mask.all(axis=1)[:, None]

        flat_twilight = copy(twilight)
        flat_twilight._data = flat_twilight._data / new_flat._data

        x, y, flux, grad_model, grad_model_e, grad_model_w, grad_model_s = flat_twilight.fit_field_gradient(wrange=SPEC_CHANNELS[channel], poly_deg=1)
        telescope = twilight._slitmap["telescope"]
        rss = twilight._data[telescope=="Sci"]
        rss_e = twilight._data[telescope=="SkyE"]
        rss_w = twilight._data[telescope=="SkyW"]
        rss_s = twilight._data[telescope=="Spec"]

        rss = rss / grad_model[:, None]
        rss_e = rss_e / grad_model_e[:, None]
        rss_w = rss_w / grad_model_w[:, None]
        rss_s = rss_s / grad_model_s[:, None]

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
    median =  np.split(bn.nanmedian(flat_twilight._data, axis=1), 3)
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

    x = (flat_twilight._data/flat_twilight._data[0]).flatten()
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
    twilight = twilight.to_native_wave(method="linear", interp_density=True, return_density=False)
    fflat.set_wave_trace(mwave)
    fflat.set_lsf_trace(mlsf)
    fflat = fflat.to_native_wave(method="linear", interp_density=False, return_density=False)
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
    mflat = mflat.to_native_wave(method="linear", interp_density=False, return_density=False)
    mflat._error = None
    mflat._mask = None
    mflat.writeFitsData(out_fiberflat, replace_masked=False)

    return mflat
