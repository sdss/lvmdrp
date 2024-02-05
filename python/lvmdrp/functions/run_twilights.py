# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 4, 2024
# @Filename: run_twilights.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import os
from copy import deepcopy as copy
import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import interpolate
from astropy.stats import biweight_location, biweight_midvariance

from lvmdrp import path, log, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS
from lvmdrp.functions import run_drp as drp
from lvmdrp.functions import run_quickdrp as qdrp

from lvmdrp.functions import imageMethod, rssMethod


ORIG_MASTER_DIR = os.getenv("LVM_MASTER_DIR")
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
# MASTER_ARC_LAMPS = {"b": "hgne", "r": "neon", "z": "neon"}
MASTER_ARC_LAMPS = {"b": "neon_hgne_argon_xenon", "r": "neon_hgne_argon_xenon", "z": "neon_hgne_argon_xenon"}
SLITMAP = Table(drp.fibermap.data)


def fit_continuum(spec, median_box=30, thresh=1.2, niter=5, poly_deg=10, wave_range=None, wave_masks=None, reset_mask=True):

    # mask bad pixels
    # spec._data[spec._mask] = np.nan
    spec._mask[:] = False

    # mask wavelength regions
    if wave_range is not None:
        iwave, fwave = wave_range
        spec._mask = ~((iwave <= spec._wave) & (spec._wave <= fwave))
        spec._data[spec._mask] = np.nan

    # mask wavelength regions
    if wave_masks is not None:
        for iwave, fwave in wave_masks:
            spec._mask |= (iwave <= spec._wave) & (spec._wave <= fwave)
            spec._data[spec._mask] = np.nan

    # copy original spectrum object
    spec_s = copy(spec)

    # replace bad pixels with NaNs
    spec_s._data[spec_s._mask] = np.nan
    spec_s._data = median_filter(spec_s._data, size=median_box)

    for i in range(niter):
        mask = np.divide(spec._data, spec_s._data, where=spec_s._data != 0, out=np.zeros_like(spec_s._data)) > thresh

        spec_s = Spectrum1D(data=np.interp(spec._pixels, spec._pixels[mask], spec._data[mask]))
        spec_s._data = median_filter(spec_s._data, median_box)


    # define continuum with last iteration's mask
    mask = np.divide(spec._data, spec_s._data, where=spec_s._data != 0, out=np.zeros_like(spec_s._data)) > thresh

    # create continuum spectrum and update mask
    out_con = copy(spec)
    out_con._mask |= ~mask

    # fit polynomial function
    coeffs = out_con.smoothSpec(size=poly_deg, method="BSpline")
    # coeffs = out_con.smoothPoly(deg=poly_deg)

    # reset mask
    if reset_mask:
        out_con._mask[:] = False

    return coeffs, out_con

def get_sequence_metadata(expnums):
    """Returns metadata for a sequence of exposures"""
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

    metadata = pd.concat(metadata, ignore_index=True)
    metadata.query("expnum in @expnums", inplace=True)
    metadata.sort_values(["camera", "expnum"], inplace=True)
    return metadata

def continuum_twilight(rsss, interpolate_bad=True, mask_bands=[],
                       median_box=5, niter=100, threshold=1,
                       plot_fibers=np.arange(20), display_plots=True, **kwargs):

    camera = rsss[0]._header["CCD"]
    expnum = rsss[0]._header["EXPOSURE"]

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
    if display_plots:
        fig, axs = plt.subplots(len(plot_fibers), figsize=(15,3*len(plot_fibers)), sharex=True, layout="constrained")
        fig.suptitle(f"Twilight flat for {camera = } and {expnum = }")
        fig.supxlabel("Wavelength (Angstrom)")
        fig.supylabel("Normalized counts")

        if mask_bands is not None:
            for mask in mask_bands:
                for ax in axs:
                    ax.axvspan(*mask, color="0.9")

    for ifiber in range(flat._fibers):
        good_pix = ~flat._mask[ifiber]
        wave = flat._wave[good_pix]
        ori_data = ori_flat._data[ifiber][good_pix]
        data = flat._data[ifiber][good_pix]
        if display_plots and ifiber in plot_fibers:
            iax = list(plot_fibers).index(ifiber)
            axs[iax].set_title(f"Fiber {ifiber+1}", loc="left")
            axs[iax].step(wave, ori_data, color="0.7", lw=1)
            axs[iax].step(wave, data, color="0.2", lw=1)

        if good_pix.sum() == 0:
            continue
        nknots = kwargs.pop("nknots", 100)
        knots = np.linspace(
                    wave[wave.size // nknots],
                    wave[-1 * wave.size // nknots],
                    nknots,
                )
        if mask_bands:
            mask = np.ones_like(knots, dtype="bool")
            for iwave, fwave in mask_bands:
                mask[(iwave<=knots)&(knots<=fwave)] = False
            knots = knots[mask]
        kwargs.update([("t", knots)])
        kwargs.update([("task", -1)])
        f = interpolate.splrep(wave, data, **kwargs)
        spline = interpolate.splev(flat._wave, f)
        for i in range(niter):

            residuals = spline - flat._data[ifiber]
            mask = spline - threshold*np.nanstd(residuals) > flat._data[ifiber]

            if display_plots and ifiber in plot_fibers:
                axs[iax].plot(flat._wave[mask], flat._data[ifiber][mask], ".", color="tab:blue", ms=5, mew=0)
                axs[iax].step(flat._wave, spline, color="tab:red", lw=1, alpha=0.5, zorder=niter+1)

            # add new outliers to mask
            flat._mask[ifiber] |= mask

            # update spline
            good_pix = ~flat._mask[ifiber]
            f = interpolate.splrep(flat._wave[good_pix], flat._data[ifiber][good_pix], **kwargs)
            new_spline = interpolate.splev(flat._wave, f)
            if np.mean(np.abs(new_spline - spline) / spline) <= 0.01:
                break
            else:
                spline = new_spline

        if display_plots and ifiber in plot_fibers:
            axs[iax].step(kwargs["t"], np.zeros_like(kwargs["t"]), ".k")
            axs[iax].step(flat._wave, spline, color="tab:red", lw=2)

        new_flat._data[ifiber] = spline

    # normalize by median fiber
    median_fiber = np.median(new_flat._data, axis=0)
    new_flat._data = new_flat._data / median_fiber
    new_flat._error = new_flat._error / median_fiber
    new_flat._data[~np.isfinite(new_flat._data)] = 1
    new_flat._mask[...] = False

    # flattield original twilight
    ori_flat._data = ori_flat._data / new_flat._data

    # plot flatfielded twilight flat
    if display_plots:
        fig, axs = plt.subplots(figsize=(15,7), sharex=True, layout="constrained")
        axs.set_title(f"Flatfielded twilight for camera = {camera}", loc="left")
        fig.supxlabel("Wavelength (Angstrom)")
        fig.supylabel("Normalized counts")

        for ifiber in range(flat._fibers):
            if ifiber in plot_fibers:
                axs.step(ori_flat._wave, ori_flat._data[ifiber], lw=1)

    new_flats = new_flat.splitRSS(parts=len(rsss), axis=1)
    [new_flat.setSlitmap(rsss[0]._slitmap) for new_flat in new_flats]

    return new_flats

def combine_twilight_sequence(expnums, camera, output_dir):
    hflats = [rssMethod.loadRSS(path.expand("lvm_anc", drpver=drpver, tileid="*", mjd="*", kind="s", imagetype="flat", camera=camera, expnum=expnum)[0]) for expnum in expnums]

    # combine RSS exposures using an average
    mflat = RSS(data=np.zeros_like(hflats[0]._data), error=np.zeros_like(hflats[0]._error), mask=np.ones_like(hflats[0]._mask, dtype=bool),
                header=copy(hflats[0]._header), slitmap=copy(hflats[0]._slitmap))
    # select non-std fibers
    fibermap =  mflat._slitmap[mflat._slitmap["spectrographid"] == int(camera[1])]
    select_allstd = fibermap["telescope"] == "Spec"
    select_nonstd = ~select_allstd
    for i, hflat in enumerate(hflats):
        # coadding all non-std fibers
        mflat._data[select_nonstd] = mflat._data[select_nonstd] + hflat._data[select_nonstd]
        mflat._error[select_nonstd] = np.sqrt(mflat._error[select_nonstd]**2 + hflat._error[select_nonstd]**2)
        mflat._mask[select_nonstd] = mflat._mask[select_nonstd] & hflat._mask[select_nonstd]

        # put std fibers in the right position
        fiber_id = f"P1-{i+1}"
        idx = np.where(fibermap["orig_ifulabel"].value == fiber_id)
        mflat._data[idx] = hflat._data[idx]
        mflat._error[idx] = hflat._error[idx]
        mflat._mask[idx] = mflat._mask[idx] & hflat._mask[idx]
        mflat._header.update(hflat._header["STD*"])

    # compute average of non-std fibers
    mflat._data[select_nonstd] = mflat._data[select_nonstd] / len(hflats)
    mflat._error[select_nonstd] = mflat._error[select_nonstd] / np.sqrt(len(hflats))

    # mask invalid pixels
    mflat._mask |= np.isnan(mflat._data) | (mflat._data <= 0) | np.isinf(mflat._data)

    return mflat

def fit_flat(mflat, camera, mwave_path=None, plot_fibers=[0, 300, 647], display_plots=True):
    channel = camera[0]
    # load master wavelength
    if mwave_path is not None:
        mwave = FiberRows()
        mwave.loadFitsData(mwave_path)
        mwave = mwave._data[:, :-1]
    else:
        mwave = np.repeat([mflat._wave], mflat._fibers, axis=0)

    # fit fiberflat
    new_flat = RSS(data=np.zeros_like(mwave),
                   wave=mwave,
                   mask=np.zeros_like(mwave, dtype="bool"),
                   inst_fwhm=mflat._inst_fwhm)
    new_flat._header = mflat._header
    new_flat._slitmap = mflat._slitmap
    new_flat._good_fibers = mflat._good_fibers

    if display_plots and plot_fibers:
        plt.figure(figsize=(20,7))
        plt.suptitle(f"Flat for {channel = }, fibers = {','.join(map(str,plot_fibers))}")
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Normalized counts")
    for ifiber in range(mflat._fibers):
        if mflat._mask[ifiber].all():
            continue

        f = interpolate.interp1d(mflat._wave, mflat._data[ifiber], bounds_error=False, fill_value=0.0)
        new_flat._data[ifiber] = f(mwave[ifiber])

        if display_plots and plot_fibers:
            if ifiber in plot_fibers:
                plt.plot(mwave[ifiber], new_flat._data[ifiber], color="tab:blue", lw=2, label="interpolated" if ifiber == plot_fibers[0] else None)
                plt.plot(mflat._wave, mflat._data[ifiber], color="0.7", lw=1, label="original" if ifiber == plot_fibers[0] else None)

    if display_plots and plot_fibers:
        plt.legend(loc=1, frameon=False)

    return new_flat

def reduce_twilight_sequence(expnums, median_box=5, niter=1000, threshold=0.5, nknots=50,
                             b_mask=[], r_mask=[], z_mask=[], display_plots=True):
    """Reduce the twilight sequence and produces master twilight flats"""
    # get metadata
    flats = get_sequence_metadata(expnums)
    for flat in flats.to_dict("records"):

        # master calibration paths
        camera = flat["camera"]
        mjd = flat["mjd"]
        arc_lamp = MASTER_ARC_LAMPS[camera[0]]
        masters_mjd = qdrp.get_master_mjd(mjd)
        masters_path = os.path.join(ORIG_MASTER_DIR, f"{masters_mjd}")
        master_cals = {
            "pixelmask" : os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits"),
            "bias" : os.path.join(masters_path, f"lvm-mbias-{camera}.fits"),
            "dark" : os.path.join(masters_path, f"lvm-mdark-{camera}.fits"),
            "pixelflat" : os.path.join(masters_path, f"lvm-mpixflat-{camera}.fits"),
            "cent" : os.path.join(masters_path, f"lvm-mtrace-{camera}.fits"),
            "width" : os.path.join(masters_path, f"lvm-mwidth-{camera}.fits"),
            "wave" : os.path.join(masters_path, f"lvm-mwave_{arc_lamp}-{camera}.fits"),
            "lsf" : os.path.join(masters_path, f"lvm-mlsf_{arc_lamp}-{camera}.fits")
        }

        # preprocess and detrend each frame
        flat_path = path.full("lvm_raw", camspec=flat["camera"], **flat)
        pflat_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=flat["imagetyp"], **flat)
        dflat_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=flat["imagetyp"], **flat)
        os.makedirs(os.path.dirname(pflat_path), exist_ok=True)
        if os.path.isfile(dflat_path):
            log.info(f"skipping {dflat_path}, file already exist")
        else:
            imageMethod.preproc_raw_frame(in_image=flat_path, out_image=pflat_path, in_mask=master_cals.get("pixelmask"))
            imageMethod.detrend_frame(in_image=pflat_path, out_image=dflat_path,
                                        in_bias=master_cals.get("bias"), in_dark=master_cals.get("dark"),
                                        in_pixelflat=master_cals.get("pixelflat"), in_slitmap=SLITMAP)

        # extract 1D spectra for each frame
        xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=flat["imagetyp"], **flat)
        if os.path.isfile(xflat_path):
            log.info(f"skipping {xflat_path}, file already exist")
        else:
            imageMethod.extract_spectra(in_image=dflat_path, out_rss=xflat_path,
                                        in_trace=master_cals.get("cent"), in_fwhm=master_cals.get("width"),
                                        method="optimal", parallel=10)

        wflat_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=flat["imagetyp"], **flat)
        if os.path.isfile(wflat_path):
            log.info(f"skipping {wflat_path}, file already exist")
        else:
            rssMethod.create_pixel_table(in_rss=xflat_path, out_rss=wflat_path,
                                            arc_wave=master_cals.get("wave"), arc_fwhm=master_cals.get("lsf"))

        hflat_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=flat["imagetyp"], **flat)
        if os.path.isfile(hflat_path):
            log.info(f"skipping {hflat_path}, file already exist")
        else:
            iwave, fwave = SPEC_CHANNELS[camera[0]]
            rssMethod.resample_wavelength(in_rss=wflat_path, out_rss=hflat_path,
                                            disp_pix=0.5, start_wave=iwave, end_wave=fwave,
                                            err_sim=5, parallel=0, extrapolate=False)

    # decompose twilight spectra into sun continuum and twilight components
    channels = "brz"
    mask_bands = dict(zip(channels, [b_mask, r_mask, z_mask]))
    new_flats = dict.fromkeys(channels)
    flat_channels = flats.groupby(flats.camera.str.__getitem__(0))
    for channel in flat_channels.groups:
        flat_expnums = flat_channels.get_group(channel).groupby("expnum")
        for expnum in flat_expnums.groups:
            flat_specs = flat_expnums.get_group(expnum)
            hflat_paths = [path.full("lvm_anc", drpver=drpver, kind="h", imagetype=flat["imagetyp"], **flat) for flat in flat_specs.to_dict("records")]
            sflat_paths = [path.full("lvm_anc", drpver=drpver, kind="s", imagetype=flat["imagetyp"], **flat) for flat in flat_specs.to_dict("records")]

            # fit fiber throughput
            hflats = [rssMethod.loadRSS(hflat_path) for hflat_path in hflat_paths]
            sflats = continuum_twilight(rsss=hflats, median_box=median_box, niter=niter, threshold=threshold, mask_bands=mask_bands[channel],
                                        display_plots=display_plots, nknots=nknots)

            # write output to disk
            for sflat, sflat_path in zip(sflats, sflat_paths):
                sflat.writeFitsData(sflat_path)

    # combine twilights and fit master fiberflat
    new_flats = dict.fromkeys(flats.camera.unique())
    flat_camera = flats.groupby("camera")
    for camera in flat_camera.groups:
        channel = camera[0]
        flat_expnums = flat_camera.get_group(camera)
        tileid = flat_expnums.tileid.min()
        mrss = combine_twilight_sequence(expnums=flat_expnums.expnum.values, camera=camera, output_dir=masters_path)
        mrss.writeFitsData(path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="mfiberflat", camera=camera))

        mwave_path = os.path.join(masters_path, f"lvm-mwave_{MASTER_ARC_LAMPS[channel]}-{camera}.fits")
        new_flat = fit_flat(mrss, camera=camera, mwave_path=mwave_path, display_plots=display_plots)
        mflat_path = os.path.join(masters_path, f"lvm-mfiberflat_twilight-{camera}.fits")
        new_flat.writeFitsData(mflat_path)
        new_flats[camera] = new_flat

    return new_flats


if __name__ == "__main__":
    expnums = list(range(7832, 7832+12))
    expnums = list(range(8027, 8038+1))
    expnums = list(range(7341, 7352+1))

    b_mask = [(3910, 4000), (4260, 4330)]
    r_mask = []
    z_mask = [(7570, 7700)]
    flats = reduce_twilight_sequence(expnums=expnums, threshold=0.5, nknots=80, b_mask=b_mask, r_mask=r_mask, z_mask=z_mask, display_plots=True)