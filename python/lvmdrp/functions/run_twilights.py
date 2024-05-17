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
import bottleneck as bn
import pandas as pd
from astropy.table import Table
from scipy.ndimage import median_filter
from scipy import interpolate

from lvmdrp import path, log, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS
from lvmdrp.core.plot import create_subplots, save_fig
from lvmdrp import main as drp
from astropy import wcs
from astropy.io import fits
from astropy.stats import biweight_location
import itertools


MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
SLITMAP = Table(drp.fibermap.data)


def polyfit2d(x, y, z, order=3):
    """
    Fit 2D polynomial
    """
    ncols = (order + 1) ** 2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j
    m, null, null, null = np.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(x, y, m):
    """
    Generate 2D polynomial
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z

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

def sumflux(rsshdu, wrange):
    naxis1 = rsshdu[0].data.shape[1]
    naxis2=rsshdu[0].data.shape[0]
    w = wcs.WCS(rsshdu[0].header)
    wave = w.spectral.pixel_to_world(np.arange(naxis1)).value*1e10
    selwave=(wave>=wrange[0])*(wave<=wrange[1])
    selwavemask=np.tile(selwave, (naxis2,1))

    flux = rsshdu[0].data
    mask = rsshdu["BADPIX"].data.astype(bool)
    flux[mask] = np.nan
    rssmasked=flux*selwavemask

    coadded_flux = np.nanmean(rssmasked,axis=1)
    return coadded_flux

def remove_field_gradient(in_hflat, out_gflat, wrange, deg=1, display_plots=False):

    fflat = fits.open(in_hflat)
    channel = fflat[0].header["CCD"]
    fibermap = Table(fflat["SLITMAP"].data)
    telescope=fibermap["telescope"]

    flux = sumflux(fflat, wrange=wrange)

    rss = fflat[0].data[telescope=="Sci"]
    rss_e = fflat[0].data[telescope=="SkyE"]
    rss_w = fflat[0].data[telescope=="SkyW"]
    rss_s = fflat[0].data[telescope=="Spec"]
    x_e=fibermap["xpmm"].astype(float)[telescope=="SkyE"]
    y_e=fibermap["ypmm"].astype(float)[telescope=="SkyE"]
    x_w=fibermap["xpmm"].astype(float)[telescope=="SkyW"]
    y_w=fibermap["ypmm"].astype(float)[telescope=="SkyW"]
    x_s=fibermap["xpmm"].astype(float)[telescope=="Spec"]
    y_s=fibermap["ypmm"].astype(float)[telescope=="Spec"]

    flux = flux[telescope=="Sci"]
    x=fibermap["xpmm"].astype(float)[telescope=="Sci"]
    y=fibermap["ypmm"].astype(float)[telescope=="Sci"]

    flux_med = np.nanmedian(flux)
    flux_fact = flux / flux_med
    select = np.isfinite(flux_fact)
    coeffs = polyfit2d(x[select], y[select], flux_fact[select], deg)

    grad_model = polyval2d(x, y, coeffs)
    grad_model_e = polyval2d(x_e, y_e, coeffs)
    grad_model_w = polyval2d(x_w, y_w, coeffs)
    grad_model_s = polyval2d(x_s, y_s, coeffs)

    fig, axs = create_subplots(to_display=display_plots, nrows=1, ncols=3, figsize=(15,5))
    flux_c = flux / grad_model
    flux_std = np.nanstd(flux)
    axs[0].scatter(x, y, s=10, vmin=flux_med-2*flux_std, vmax=flux_med+2*flux_std, lw=0, c=flux, cmap="rainbow")
    axs[1].scatter(x, y, s=10, vmin=0.95, vmax=1.05, lw=0, c=grad_model, cmap="coolwarm")
    axs[2].scatter(x, y, s=10, vmin=flux_med-2*flux_std, vmax=flux_med+2*flux_std, lw=0, c=flux_c, cmap="rainbow")
    axs[0].set_title("Original", loc="left")
    axs[1].set_title("Gradient", loc="left")
    axs[2].set_title("Original / Gradient", loc="left")
    fig.suptitle(f"Gradient field fitting for {channel = }")
    save_fig(fig, out_gflat, to_display=display_plots, figure_path="qa", label="field_gradient")

    rss = rss / grad_model[:, None]
    rss_e = rss_e / grad_model_e[:, None]
    rss_w = rss_w / grad_model_w[:, None]
    rss_s = rss_s / grad_model_s[:, None]

    new_fflat = copy(fflat)
    new_fflat[0].data[telescope == "Sci"] = rss
    new_fflat[0].data[telescope == "SkyE"] = rss_e
    new_fflat[0].data[telescope == "SkyW"] = rss_w
    new_fflat[0].data[telescope == "Spec"] = rss_s
    new_fflat.writeto(out_gflat, overwrite=True)

def get_sequence_metadata(expnums: List[int]) -> pd.DataFrame:
    """Returns metadata for a sequence of exposures

    Parameters
    ----------
    expnums : list
        List of exposure numbers

    Returns
    -------
    metadata : pd.DataFrame
        DataFrame with metadata for the sequence of exposures
    """
    paths = []
    for expnum in expnums:
        paths.extend(path.expand("lvm_raw", mjd="*", hemi="s", camspec="*", expnum=expnum))
    paths = sorted(paths)

    mjds = []
    for p in paths:
        mjds.append(int(path.extract("lvm_raw", p)["mjd"]))
    mjds = np.unique(mjds)

    metadata = []
    for mjd in mjds:
        metadata.append(md.get_frames_metadata(mjd=mjd))

    if len(metadata) == 0:
        return pd.DataFrame()

    metadata = pd.concat(metadata, ignore_index=True)
    metadata.query("expnum in @expnums", inplace=True)
    metadata.sort_values(["camera", "expnum"], inplace=True)
    return metadata

def fit_continuum(spectrum: Spectrum1D, mask_bands: List[Tuple[float,float]],
                  median_box: int, niter: int, threshold: Tuple[float,float]|float, **kwargs):
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
    nknots = kwargs.pop("nknots", 100)
    knots = np.linspace(wave[wave.size // nknots], wave[-1 * wave.size // nknots], nknots)
    if mask_bands:
        mask = np.ones_like(knots, dtype="bool")
        for iwave, fwave in mask_bands:
            mask[(iwave<=knots)&(knots<=fwave)] = False
        knots = knots[mask]
    kwargs.update({"t": knots, "task": -1})

    # fit first spline
    tck = interpolate.splrep(wave, data, **kwargs)
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
        tck = interpolate.splrep(spectrum._wave[~masked_pixels], spectrum._data[~masked_pixels], **kwargs)
        new_spline = interpolate.splev(spectrum._wave, tck)
        continuum_models.append(new_spline)
        if np.mean(np.abs(new_spline - spline) / spline) <= 0.01:
            break
        else:
            spline = new_spline

    best_continuum = continuum_models.pop(-1)
    return best_continuum, continuum_models, masked_pixels, tck

def fit_fiberflat(in_twilight: str, out_flat: str, out_rss: str, interpolate_bad: bool = True, mask_bands: List[Tuple[float,float]] = [],
                  median_box: int = 5, niter: int = 1000, threshold: Tuple[float,float]|float = (0.5,2.0),
                  plot_fibers: List[int] = [0,300,600,900,1200,1400,1700],
                  display_plots: bool = False, **kwargs) -> List[RSS]:
    """Fit fiber throughput for a twilight sequence

    Given a list of three extracted and wavelength calibrated twilight
    exposures (spec 1, 2, and 3), this function fits the fiber throughput
    across the entire spectrograph channel using an iterative spline fitting
    method.

    Parameters
    ----------
    in_twilight : str
        Input path for the twilight exposure
    out_flat : str
        Output path for the fitted fiberflat
    out_rss : str
        Output path for the flatfielded twilight
    interpolate_bad : bool, optional
        Interpolate bad pixels, by default True
    mask_bands : list, optional
        List of wavelength bands to mask, by default []
    median_box : int, optional
        Size of the median filter box, by default 5
    niter : int, optional
        Number of iterations to fit the continuum, by default 1000
    threshold : float, optional
        Threshold to mask outliers, by default (0.5,2.0)
    plot_fibers : list, optional
        List of fibers to plot, by default [0,300,600,900,1200,1400,1700]
    display_plots : bool, optional
        Display plots, by default False

    Returns
    -------
    new_flats : list
        List of RSS objects for each twilight exposure with the fitted fiber throughput
    """
    twilight = RSS.from_file(in_twilight)

    channel = twilight._header["CCD"]
    expnum = twilight._header["EXPOSURE"]
    unit = twilight._header["BUNIT"]

    ori_flat = copy(twilight)
    new_flat = copy(twilight)

    # mask bad pixels
    twilight._mask |= np.isnan(twilight._data) | (twilight._data < 0) | np.isinf(twilight._data)
    twilight._data[twilight._mask] = np.nan

    # interpolate bad pixels
    if interpolate_bad:
        twilight.interpolate_data(axis="X", reset_mask=False)

    # mask wavelength bands
    if mask_bands:
        for iwave, fwave in mask_bands:
            twilight._mask |= (iwave <= twilight._wave) & (twilight._wave <= fwave)
            twilight._data[twilight._mask] = np.nan

    # remove high-frequency features and update mask
    twilight._data = median_filter(twilight._data, (1,median_box))
    twilight._mask |= np.isnan(twilight._data)

    # diplay plots
    fig, axs = create_subplots(to_display=display_plots,
                               nrows=len(plot_fibers), ncols=1, sharex=True,
                               figsize=(15,3*len(plot_fibers)), layout="constrained")
    fig.suptitle(f"Twilight flat for {channel = } and {expnum = }")
    fig.supxlabel("Wavelength (Angstrom)")
    fig.supylabel(f"Counts ({unit})")

    if mask_bands is not None:
        for mask in mask_bands:
            for ax in axs:
                ax.axvspan(*mask, color="0.9")

    for ifiber in range(twilight._fibers):
        fiber = twilight[ifiber]
        ori_fiber = ori_flat[ifiber]

        try:
            best_continuum, continuum_models, masked_pixels, tck = fit_continuum(
                spectrum=fiber, mask_bands=mask_bands,
                median_box=median_box, niter=niter, threshold=threshold, **kwargs
            )
        except (ValueError, TypeError) as e:
            log.error(f"while fitting fiber throughput for fiber {ifiber}: {e}")
            new_flat._data[ifiber] = np.nan
            new_flat._mask[ifiber] = True
            continue

        if ifiber in plot_fibers:
            good_pix = ~fiber._mask
            iax = list(plot_fibers).index(ifiber)

            # plot original fiber and processed fiber
            axs[iax].set_title(f"Fiber {ifiber+1}", loc="left")
            axs[iax].step(fiber._wave[good_pix], ori_fiber._data[good_pix], color="0.7", lw=1)
            axs[iax].step(fiber._wave[good_pix], fiber._data[good_pix], color="0.2", lw=1)

            # plot masked pixels and fitted splines
            for continuum_model in continuum_models:
                axs[iax].plot(fiber._wave[masked_pixels], fiber._data[masked_pixels], ".", color="tab:blue", ms=5, mew=0)
                axs[iax].plot(fiber._wave, continuum_model, color="tab:red", lw=1, alpha=0.5, zorder=niter)
            axs[iax].plot(tck[0], np.zeros_like(tck[0]), ".k")
            axs[iax].step(fiber._wave, best_continuum, color="tab:red", lw=2)

        new_flat._data[ifiber] = best_continuum

    save_fig(
        fig,
        product_path=out_rss,
        to_display=display_plots,
        label="twilight_continuum_fit",
        figure_path="qa"
    )

    # normalize by median fiber
    median_fiber = bn.nanmedian(new_flat._data, axis=0)
    # median_fiber = bn.nanmean(new_flat._data, axis=0)
    new_flat._data = new_flat._data / median_fiber
    new_flat._error = new_flat._error / median_fiber
    # new_flat._data[~np.isfinite(new_flat._data)] = 1
    # new_flat._mask[...] = False

    # flattield original twilight
    ori_flat._data = ori_flat._data / new_flat._data
    med_fiberflat = np.median(ori_flat._data, axis=0)

    # plot flatfielded twilight flat
    fig, axs = create_subplots(to_display=display_plots, figsize=(15,7), sharex=True, layout="constrained")
    axs.set_title(f"Flatfielded twilight for camera = {channel}", loc="left")
    fig.supxlabel("Wavelength (Angstrom)")
    fig.supylabel("Normalized counts")

    flat_error = ori_flat._data / med_fiberflat
    med_flat_error = np.median(flat_error, axis=0)
    std_flat_error = np.std(flat_error, axis=0)
    med_error = np.median(ori_flat._error, axis=0) / med_fiberflat
    for ifiber in range(twilight._fibers):
        if ifiber in plot_fibers:
            axs.step(ori_flat._wave, flat_error[ifiber], color="0.2", alpha=0.5, lw=1)
    axs.step(ori_flat._wave, med_flat_error, color="tab:red", lw=2)
    axs.step(ori_flat._wave, med_flat_error - std_flat_error, color="tab:blue", lw=2)
    axs.step(ori_flat._wave, med_flat_error + std_flat_error, color="tab:blue", lw=2)
    axs.step(ori_flat._wave, med_flat_error - med_error, color="tab:green", lw=2)
    axs.step(ori_flat._wave, med_flat_error + med_error, color="tab:green", lw=2)
    axs.set_ylim(0.8, 1.2)

    save_fig(
        fig,
        product_path=out_rss,
        to_display=display_plots,
        label="twilight_flatfielded",
        figure_path="qa"
    )

    # new_flats = new_flat.splitRSS(parts=len(rsss), axis=1)
    # [new_flat.setSlitmap(rsss[0]._slitmap) for new_flat in new_flats]

    # write output fiberflat
    log.info(f"writing twilight flat to {out_flat}")
    new_flat.writeFitsData(out_flat)

    # write output faltfielded explosure
    log.info(f"writing flatfielded twilight to {out_rss}")
    ori_flat.writeFitsData(out_rss)

    return new_flat

def combine_twilight_sequence(fflats: List[RSS]) -> RSS:
    """Combine twilight exposures into a single RSS object

    Given a list of RSS objects of fiberflats from twilight exposures, this
    function combines them into a single RSS object by averaging the fiber
    throughput of all non-standard fibers and putting the standard fibers in
    their respective positions.

    Parameters
    ----------
    fflats : list
        List of fiberflat individual exposures

    Returns
    -------
    mflat : RSS
        Master twilight flat
    """

    # data = [fflat._data for fflat in fflats]
    # median_fiber = bn.nanmedian(data, axis=0)
    # for fflat in fflats:
    #     fflat._data = fflat._data / median_fiber
    #     fflat._error = fflat._error / median_fiber

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
            snrs = np.nanmedian(fflat._data / fflat._error, axis=1)
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

    return mflat

def resample_fiberflat(mflat: RSS, channel: str, mwave_paths: str,
             plot_fibers: List[int] = [0,300,600,900,1200,1400,1700],
             display_plots: bool = False) -> RSS:
    """Fit master twilight flat

    Parameters
    ----------
    mflat : RSS
        Master twilight flat
    channel : str
        LVM channel (b1, b2, r1, r2, z1, z2, etc.)
    mwave_path : str
        Path to master wavelength
    plot_fibers : list, optional
        List of fibers to plot, by default [0,300,600,900,1200,1400,1700]
    display_plots : bool, optional
        Display plots, by default False

    Returns
    -------
    new_flat : RSS
        Master twilight flat with fitted fiber throughput
    """
    tileid = mflat._header.get("TILE_ID", 11111) or 11111
    mjd = mflat._header["MJD"]
    # load master wavelength
    if mwave_paths is not None:
        mwaves = [TraceMask.from_file(mwave_path) for mwave_path in mwave_paths]
        mwave = TraceMask.from_spectrographs(*mwaves)
        mwave = mwave._data[:, :-1]
    else:
        mwave = np.repeat([mflat._wave], mflat._fibers, axis=0)

    # fit fiberflat
    new_flat = RSS(data=np.zeros_like(mwave),
                   wave=mwave,
                   mask=np.zeros_like(mwave, dtype="bool"),
                   lsf=mflat._lsf,
                   slitmap=mflat._slitmap,
                   header=copy(mflat._header))
    for kw in ["WCSAXES", "CRPIX1", "CDELT1", "CUNIT1", "CTYPE1", "CRVAL1", "LATPOLE", "MJDREF", "METREC", "WAVREC"]:
        if kw in new_flat._header:
            del new_flat._header[kw]
    new_flat._good_fibers = mflat._good_fibers

    fig, ax = create_subplots(to_display=display_plots, figsize=(15,7))
    fig.suptitle(f"Flat for {channel = }, fibers = {','.join(map(str,plot_fibers))}")
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized counts")
    for ifiber in range(mflat._fibers):
        if mflat._mask[ifiber].all():
            continue

        f = interpolate.interp1d(mflat._wave, mflat._data[ifiber], bounds_error=False, fill_value=0.0)
        new_flat._data[ifiber] = f(mwave[ifiber])

        if plot_fibers:
            if ifiber in plot_fibers:
                ax.plot(mwave[ifiber], new_flat._data[ifiber], color="tab:blue", lw=2, label="interpolated" if ifiber == plot_fibers[0] else None)
                ax.plot(mflat._wave, mflat._data[ifiber], color="0.7", lw=1, label="original" if ifiber == plot_fibers[0] else None)

    ax.legend(loc=1, frameon=False)
    save_fig(
        fig,
        product_path=path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, kind="f", imagetype="flat", camera=channel),
        to_display=display_plots,
        label="twilight_fiberflat_fit",
        figure_path="qa"
    )

    return new_flat
