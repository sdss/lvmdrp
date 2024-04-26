# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 4, 2024
# @Filename: run_twilights.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from __future__ import annotations

import os
from glob import glob
from typing import Tuple, List, Dict
from copy import deepcopy as copy
import numpy as np
import bottleneck as bn
import pandas as pd
from astropy.table import Table
from scipy.ndimage import median_filter
from scipy import interpolate

from lvmdrp import path, log, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS
from lvmdrp.core.plot import create_subplots, save_fig
from lvmdrp.functions import run_drp as drp
from lvmdrp.functions import run_quickdrp as qdrp
from lvmdrp.functions import run_calseq as calseq

from lvmdrp.functions import imageMethod, rssMethod


ORIG_MASTER_DIR = os.getenv("LVM_MASTER_DIR")
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
# MASTER_ARC_LAMPS = {"b": "hgne", "r": "neon", "z": "neon"}
MASTER_ARC_LAMPS = {"b": "neon_hgne_argon_xenon", "r": "neon_hgne_argon_xenon", "z": "neon_hgne_argon_xenon"}
SLITMAP = Table(drp.fibermap.data)

MASK_BANDS = {
        "b": [(3910, 4000), (4260, 4330)],
        "r": [(6840,6960)],
        "z": [(7570, 7700)]
    }


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
    knots : np.ndarray
        Spline knots
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
    f = interpolate.splrep(wave, data, **kwargs)
    spline = interpolate.splev(spectrum._wave, f)

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
        f = interpolate.splrep(spectrum._wave[~masked_pixels], spectrum._data[~masked_pixels], **kwargs)
        new_spline = interpolate.splev(spectrum._wave, f)
        continuum_models.append(new_spline)
        if np.mean(np.abs(new_spline - spline) / spline) <= 0.01:
            break
        else:
            spline = new_spline

    best_continuum = continuum_models.pop(-1)
    return best_continuum, continuum_models, masked_pixels, knots

def fit_fiberflat(rsss: List[RSS], out_flat: str, out_rss: str, interpolate_bad: bool = True, mask_bands: List[Tuple[float,float]] = [],
                  median_box:int = 5, niter: int = 1000, threshold: Tuple[float,float]|float = (0.5,2.0),
                  plot_fibers: List[int] = list(range(0,648,10)),#[0,300,600,900,1200,1400,1700],
                  display_plots: bool = False, **kwargs) -> List[RSS]:
    """Fit fiber throughput for a twilight sequence

    Given a list of three extracted and wavelength calibrated twilight
    exposures (spec 1, 2, and 3), this function fits the fiber throughput
    across the entire spectrograph channel using an iterative spline fitting
    method.

    Parameters
    ----------
    rsss : list
        List of RSS objects for each twilight exposure
    out_flat : str
        Output path for the master twilight flat
    out_rss : str
        Output path for the master twilight flat
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
    camera = rsss[0]._header["CCD"]
    expnum = rsss[0]._header["EXPOSURE"]
    unit = rsss[0]._header["BUNIT"]

    # stack rsss
    data, error, mask = [], [], []
    for rss in rsss:
        data.append(rss._data)
        error.append(rss._error)
        mask.append(rss._mask)

    flat = RSS(
        wave=rsss[0]._wave,
        data=np.row_stack(data),
        error=np.row_stack(error),
        mask=np.row_stack(mask),
        header=rsss[0]._header,
        slitmap=rsss[0]._slitmap
    )

    ori_flat = copy(flat)
    new_flat = copy(flat)

    # mask bad pixels
    flat._mask |= np.isnan(flat._data) | (flat._data < 0) | np.isinf(flat._data)
    flat._data[flat._mask] = np.nan

    # interpolate bad pixels
    if interpolate_bad:
        flat.interpolate_data(axis="X", reset_mask=False)

    # mask wavelength bands
    if mask_bands:
        for iwave, fwave in mask_bands:
            flat._mask |= (iwave <= flat._wave) & (flat._wave <= fwave)
            flat._data[flat._mask] = np.nan

    # remove high-frequency features and update mask
    flat._data = median_filter(flat._data, (1,median_box))
    flat._mask |= np.isnan(flat._data)

    # diplay plots
    fig, axs = create_subplots(to_display=display_plots,
                               nrows=len(plot_fibers), ncols=1, sharex=True,
                               figsize=(15,3*len(plot_fibers)), layout="constrained")
    fig.suptitle(f"Twilight flat for {camera = } and {expnum = }")
    fig.supxlabel("Wavelength (Angstrom)")
    fig.supylabel(f"Counts ({unit})")

    if mask_bands is not None:
        for mask in mask_bands:
            for ax in axs:
                ax.axvspan(*mask, color="0.9")

    for ifiber in range(flat._fibers):
        twilight = flat[ifiber]
        ori_twilight = ori_flat[ifiber]

        try:
            best_continuum, continuum_models, masked_pixels, knots = fit_continuum(
                spectrum=twilight, mask_bands=mask_bands,
                median_box=median_box, niter=niter, threshold=threshold, **kwargs
            )
        except ValueError:
            log.error(f"while fitting fiber throughput for fiber {ifiber+1}")
            new_flat._data[ifiber] = 0.0
            new_flat._mask[ifiber] = True
            continue

        if ifiber in plot_fibers:
            good_pix = ~twilight._mask
            iax = list(plot_fibers).index(ifiber)

            # plot original twilight and processed twilight
            axs[iax].set_title(f"Fiber {ifiber+1}", loc="left")
            axs[iax].step(twilight._wave[good_pix], ori_twilight._data[good_pix], color="0.7", lw=1)
            axs[iax].step(twilight._wave[good_pix], twilight._data[good_pix], color="0.2", lw=1)

            # plot masked pixels and fitted splines
            for continuum_model in continuum_models:
                axs[iax].plot(twilight._wave[masked_pixels], twilight._data[masked_pixels], ".", color="tab:blue", ms=5, mew=0)
                axs[iax].plot(twilight._wave, continuum_model, color="tab:red", lw=1, alpha=0.5, zorder=niter)
            axs[iax].plot(knots, np.zeros_like(knots), ".k")
            axs[iax].step(twilight._wave, best_continuum, color="tab:red", lw=2)

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
    new_flat._data[~np.isfinite(new_flat._data)] = 1
    new_flat._mask[...] = False

    # flattield original twilight
    ori_flat._data = ori_flat._data / new_flat._data
    med_fiberflat = np.median(ori_flat._data, axis=0)

    # plot flatfielded twilight flat
    fig, axs = create_subplots(to_display=display_plots, figsize=(15,7), sharex=True, layout="constrained")
    axs.set_title(f"Flatfielded twilight for camera = {camera}", loc="left")
    fig.supxlabel("Wavelength (Angstrom)")
    fig.supylabel("Normalized counts")

    flat_error = ori_flat._data / med_fiberflat
    med_flat_error = np.median(flat_error, axis=0)
    std_flat_error = np.std(flat_error, axis=0)
    med_error = np.median(ori_flat._error, axis=0) / med_fiberflat
    for ifiber in range(flat._fibers):
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
    log.info(f"writing twilight flat to {out_rss}")
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
    select_nonstd = ~select_allstd
    for i, fflat in enumerate(fflats):
        # coadding all non-std fibers
        mflat._data[select_nonstd] = mflat._data[select_nonstd] + fflat._data[select_nonstd]
        mflat._error[select_nonstd] = np.sqrt(mflat._error[select_nonstd]**2 + fflat._error[select_nonstd]**2)
        mflat._mask[select_nonstd] = mflat._mask[select_nonstd] & fflat._mask[select_nonstd]

        # get exposed standard fiber ID
        default_fiber_id = f"P1-{i+1}"
        fiber_id = mflat._header.get("CALIBFIB", default_fiber_id) or default_fiber_id
        # put std fibers in the right position
        idx = np.where(fibermap["orig_ifulabel"].value == fiber_id)
        mflat._data[idx] = fflat._data[idx]
        mflat._error[idx] = fflat._error[idx]
        mflat._mask[idx] = mflat._mask[idx] & fflat._mask[idx]
        mflat._header.update(fflat._header["STD*"])

    # compute average of non-std fibers
    mflat._data[select_nonstd] = mflat._data[select_nonstd] / len(fflats)
    mflat._error[select_nonstd] = mflat._error[select_nonstd] / np.sqrt(len(fflats))

    # mask invalid pixels
    mflat._mask |= np.isnan(mflat._data) | (mflat._data <= 0) | np.isinf(mflat._data)

    return mflat

def resample_fiberflat(mflat: RSS, channel: str, mwave_paths: str,
             plot_fibers: List[int] = list(range(0,648,10)),
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
        List of fibers to plot, by default [0, 300, 647]
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

def reduce_twilight_sequence(expnums: List[int], median_box: int = 10, niter: bool = 1000,
                             threshold: Tuple[float,float]|float = (0.5,1.5), nknots: bool = 50,
                             b_mask: List[Tuple[float,float]] = MASK_BANDS["b"],
                             r_mask: List[Tuple[float,float]] = MASK_BANDS["r"],
                             z_mask: List[Tuple[float,float]] = MASK_BANDS["z"],
                             skip_done: bool = False,
                             display_plots: bool = False) -> Dict[str, RSS]:
    """Reduce the twilight sequence and produces master twilight flats

    Given a sequence of twilight exposures, this function reduces them and
    produces master twilight flats for each camera.

    Parameters
    ----------
    expnums : list
        List of twilight exposure numbers
    median_box : int, optional
        Size of the median filter box, by default 5
    niter : int, optional
        Number of iterations to fit the continuum, by default 1000
    threshold : float, optional
        Threshold to mask outliers, by default 0.5
    nknots : int, optional
        Number of knots for the spline fitting, by default 50
    b_mask : list, optional
        List of wavelength bands to mask in the blue channel, by default []
    r_mask : list, optional
        List of wavelength bands to mask in the red channel, by default []
    z_mask : list, optional
        List of wavelength bands to mask in the NIR channel, by default []
    skip_done : bool, optional
        Skip files that already exist, by default False
    display_plots : bool, optional
        Display plots, by default False

    Returns
    -------
    new_flats : dict
        Dictionary with the master twilight flats for each channel
    """
    # get metadata
    flats = get_sequence_metadata(expnums)

    # 2D reduction of twilight sequence
    calseq.reduce_2d(mjds=flats.rmjd.iloc[0], target_mjd=flats.rmjd.iloc[0], expnums=flats.expnum.unique(), reject_cr=False, skip_done=True)

    for flat in flats.to_dict("records"):

        # master calibration paths
        camera = flat["camera"]
        mjd = flat["mjd"]
        masters_mjd = qdrp.get_master_mjd(mjd)
        masters_path = os.path.join(ORIG_MASTER_DIR, f"{masters_mjd}")
        master_cals = {
            "pixelmask" : os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits"),
            "bias" : os.path.join(masters_path, f"lvm-mbias-{camera}.fits"),
            "dark" : os.path.join(masters_path, f"lvm-mdark-{camera}.fits"),
            "pixelflat" : os.path.join(masters_path, f"lvm-mpixflat-{camera}.fits"),
            "cent" : os.path.join(masters_path, f"lvm-mtrace-{camera}.fits"),
            "width" : os.path.join(masters_path, f"lvm-mwidth-{camera}.fits"),
            "wave" : os.path.join(masters_path, f"lvm-mwave-{camera}.fits"),
            "lsf" : os.path.join(masters_path, f"lvm-mlsf-{camera}.fits")
        }

        # extract 1D spectra for each frame
        lflat_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype=flat["imagetyp"], **flat)
        xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=flat["imagetyp"], **flat)
        if skip_done and os.path.isfile(xflat_path):
            log.info(f"skipping {xflat_path}, file already exist")
        else:
            imageMethod.extract_spectra(in_image=lflat_path, out_rss=xflat_path,
                                        in_trace=master_cals.get("cent"), in_fwhm=master_cals.get("width"),
                                        method="optimal", parallel=10)

    # decompose twilight spectra into sun continuum and twilight components
    channels = "brz"
    channels = "r"
    mask_bands = dict(zip(channels, [b_mask, r_mask, z_mask]))
    new_flats = dict.fromkeys(channels)
    flat_channels = flats.groupby(flats.camera.str.__getitem__(0))
    tileid = flats.tileid.min()
    for channel in channels:
        flat_expnums = flat_channels.get_group(channel).groupby("expnum")
        fflats = []
        for expnum in flat_expnums.groups:
            flat = flat_expnums.get_group(expnum).iloc[0]

            xflat_paths = sorted(path.expand("lvm_anc", drpver=drpver, kind="x", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=f"{channel}?", expnum=expnum))
            fflat_flatfielded_path = path.full("lvm_anc", drpver=drpver, kind="flatfielded_",
                                   imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"],
                                   camera=channel, expnum=expnum)
            fflat_path = path.full("lvm_anc", drpver=drpver, kind="f",
                                   imagetype="flat", tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)

            mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave-{channel}?.fits")))
            mlsf_paths = sorted(glob(os.path.join(masters_path, f"lvm-mlsf-{channel}?.fits")))

            # spectrograph stack xflats
            xflat = RSS.from_spectrographs(*[rssMethod.loadRSS(xflat_path) for xflat_path in xflat_paths])
            xflat_path = path.full("lvm_anc", drpver=drpver, kind="x",
                                   imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"],
                                   camera=channel, expnum=expnum)
            xflat.writeFitsData(xflat_path)

            # calibrate in wavelength
            wflat_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rssMethod.create_pixel_table(in_rss=xflat_path, out_rss=wflat_path,
                                         in_waves=mwave_paths, in_lsfs=mlsf_paths)

            # rectify in wavelength
            hflat_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rssMethod.resample_wavelength(in_rss=wflat_path, out_rss=hflat_path, wave_disp=0.5, wave_range=SPEC_CHANNELS[channel])

            # fit fiber throughput
            hflat = rssMethod.loadRSS(hflat_path)
            hflats = hflat.splitRSS(parts=len(mwave_paths), axis=1)
            fflat = fit_fiberflat(rsss=hflats, out_flat=fflat_path, out_rss=fflat_flatfielded_path, median_box=median_box, niter=niter,
                                   threshold=threshold, mask_bands=mask_bands.get(channel, []),
                                   display_plots=display_plots, nknots=nknots)
            fflats.append(fflat)

        mrss = combine_twilight_sequence(fflats=fflats)
        mrss.writeFitsData(path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="mfiberflat", camera=channel))

        mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave-{channel}?.fits")))
        new_flat = resample_fiberflat(mrss, channel=channel, mwave_paths=mwave_paths, display_plots=display_plots)
        mflat_path = os.path.join(masters_path, f"lvm-mfiberflat_twilight-{channel}.fits")
        new_flat.writeFitsData(mflat_path)
        new_flats[channel] = new_flat

    return new_flats


if __name__ == "__main__":

    expnums = [7231]
    expnums = np.arange(7341, 7352+1)
    reduce_twilight_sequence(expnums=expnums, median_box=10, niter=1000, threshold=(0.5,2.5), nknots=60, skip_done=True, display_plots=False)
