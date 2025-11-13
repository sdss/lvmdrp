# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 4, 2024
# @Filename: run_twilights.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from __future__ import annotations

import warnings
from typing import Tuple, List
from copy import deepcopy as copy
import numpy as np
from astropy.table import Table
from scipy.ndimage import median_filter
from scipy import interpolate
from matplotlib.gridspec import GridSpec

import bottleneck as bn
from lvmdrp import log
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS, lvmFrame
from lvmdrp.core.fluxcal import butter_lowpass_filter
from lvmdrp.core.fit_profile import IFUGradient
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.plot import plt, slit, plot_gradient_fit, plot_flatfield_validation, plot_flat_consistency, create_subplots, save_fig
from lvmdrp import main as drp
from astropy import wcs
from astropy.io import fits
from astropy.stats import biweight_location


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


def to_native_wave(rss, wave=None):

    if not rss._header.get("WAVREC", False):
        warnings.warn("input RSS is not rectified, nothing to do")
        return rss

    # get native wavelength grid or use the one given
    if wave is None and rss._wave_trace is not None:
        trace = TraceMask.from_coeff_table(rss._wave_trace)
        wave = trace.eval_coeffs()
    elif wave is not None:
        pass
    else:
        raise ValueError(f"missing wavelength trace information: {rss._wave_trace = }")

    new_rss = copy(rss)
    new_rss.setData(
        data=np.zeros((rss._fibers, wave.shape[1]), dtype="float32"),
        error=np.zeros((rss._fibers, wave.shape[1]), dtype="float32"),
        mask=np.zeros((rss._fibers, wave.shape[1]), dtype="bool")
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


def combine_twilight_sequence(in_twilights: list[str], in_fflats: List[str], out_mflat: str, out_lvmflats: List[str],
                              in_cents: List[str], in_widths: List[str],
                              in_waves: List[str], in_lsfs: List[str], comb_method: str = "biweight",
                              display_plots: bool = False) -> RSS:
    """Combine twilight exposures into a master fiberflat RSS object and create lvmFlat products.

    This function processes a sequence of twilight exposures and their
    corresponding fiberflats to create a master fiberflat RSS object. It
    averages the fiber throughput of all non-standard fibers while ensuring
    that standard fibers are placed in their respective positions. The
    resulting master fiberflat is resampled to a native wavelength grid,
    interpolated to handle masked fibers, and saved to the specified output
    path. Additionally, lvmFlat products are created for each twilight
    exposure, incorporating calibration information.

    Parameters
    ----------
    in_twilights : list[str]
        List of paths to twilight RSS exposures.
    in_fflats : list[str]
        List of paths to individual fiberflat exposures.
    out_mflat : str
        Output path to save the master fiberflat RSS object.
    out_lvmflats : list[str]
        List of output paths to save the lvmFlat objects for each twilight exposure.
    in_cents : list[str]
        List of paths to fiber centroid traces for each channel.
    in_widths : list[str]
        List of paths to fiber width traces for each channel.
    in_waves : list[str]
        List of paths to wavelength solution traces for each channel.
    in_lsfs : list[str]
        List of paths to LSF solution traces for each channel.
    comb_method : str, optional
        Method used to combine the individual fiberflats, by default "biweight".
    display_plots : bool, optional
        Whether to display diagnostic plots during processing, by default False.

    Returns
    -------
    mflat : RSS
        Master fiberflat RSS object.
    fflats : list[RSS]
        List of individual fiberflat RSS objects.
    twilights : list[RSS]
        List of twilight RSS objects.
    lvmflats : list[lvmFlat]
        List of lvmFlat objects created for each twilight exposure.
    """

    log.info(f"loading {len(in_twilights)} twilights")
    twilights = [RSS.from_file(in_twilight) for in_twilight in in_twilights]

    log.info(f"loading {len(in_fflats)} flat fields")
    fflats = [RSS.from_file(in_fflat) for in_fflat in in_fflats]

    log.info(f"loading fiber centroids from {in_cents}")
    mcent = TraceMask.from_spectrographs(*[TraceMask.from_file(in_cent) for in_cent in in_cents])
    log.info(f"loading fiber widths from {in_widths}")
    mwidth = TraceMask.from_spectrographs(*[TraceMask.from_file(in_width) for in_width in in_widths])
    log.info(f"loading wavelength traces from {in_waves}")
    mwave = TraceMask.from_spectrographs(*[TraceMask.from_file(in_wave) for in_wave in in_waves])
    log.info(f"loading LSF traces from {in_lsfs}")
    mlsf = TraceMask.from_spectrographs(*[TraceMask.from_file(in_lsf) for in_lsf in in_lsfs])

    # select non-std fibers
    fibermap =  fflats[0]._slitmap
    select_allstd = fibermap["telescope"] == "Spec"
    # select_nonstd = ~select_allstd
    log.info(f"identifying exposed standard fibers in sequence of {len(fflats)} flat fields:")
    for twilight, fflat in zip(twilights, fflats):
        expnum = twilight._header["EXPOSURE"]

        # get exposed standard fiber ID
        fiber_id = twilight._header.get("CALIBFIB")
        if fiber_id is None:
            snrs = bn.nanmedian(twilight._data / twilight._error, axis=1)
            select_nonexposed = select_allstd & (snrs < 50)
            fiber_id = fibermap[~select_nonexposed]["orig_ifulabel"]
            log.info(f"  for {expnum = } no CALIBFIB header keyword found, inferred {fiber_id = }")
        else:
            select_nonexposed = select_allstd & (fibermap["orig_ifulabel"] != fiber_id)
            log.info(f"  for {expnum = } exposed {fiber_id = }")
        # put std fibers in the right position
        fflat._data[select_nonexposed] = np.nan
        fflat._error[select_nonexposed] = np.nan
        fflat._mask[select_nonexposed] = True

    # TODO: reject outlying flat fields before combining
    log.info(f"combining {len(fflats)} individual flat fields using {comb_method = }")
    mflat = RSS()
    mflat.combineRSS(fflats, method=comb_method)
    channel = mflat._header["CCD"]
    cwave, dwave = mflat._header[f"{channel} FIBERFLAT CWAVE"], mflat._header[f"{channel} FIBERFLAT DWAVE"]
    groupby = mflat._header[f"{channel} FIBERFLAT GROUPBY"]
    coadd_method = mflat._header[f"{channel} FIBERFLAT COADD"]

    # mask invalid pixels
    mflat._mask |= np.isnan(mflat._data) | (mflat._data <= 0) | np.isinf(mflat._data)
    mflat._mask |= np.isnan(mflat._error) | (mflat._error <= 0) | np.isinf(mflat._error)
    log.info(f"interpolating {mflat._mask.sum()} masked pixels ({mflat._mask.all(axis=1).sum()} fibers)")
    # interpolate masked fibers if any remaining
    mflat = mflat.interpolate_data(axis="X")
    mflat = mflat.interpolate_data(axis="Y")

    ncols = 6
    nrows = int(np.ceil(len(fflats)/ncols))
    fig = plt.figure(figsize=(14,3*(3+len(fflats))))
    fig.suptitle(f"Combined flat field consistency for {channel = }", fontsize="xx-large")
    gs = GridSpec(nrows + len(fflats), ncols, hspace=0.3, wspace=0.05, left=0.07, right=0.99, bottom=0.1, top=0.97, figure=fig)
    axs_con = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
    [ax.tick_params(labelbottom=ax.get_subplotspec().is_last_row(), labelleft=ax.get_subplotspec().is_first_col()) for ax in axs_con]
    [(ax.sharex(axs_con[0]), ax.sharey(axs_con[0])) for ax in axs_con[1:]]
    [ax.set_ylabel("Frequency", fontsize="large") for ax in axs_con[::ncols]]
    [ax.set_xlabel("mflat / flat", fontsize="large") for i, ax in enumerate(axs_con[-ncols:])]
    plot_flat_consistency(fflats=fflats, mflat=mflat, log_scale=False, spec_wise=True, labels=True, axs=axs_con)

    mflat.set_wave_trace(mwave)
    mflat.set_lsf_trace(mlsf)
    mflat = to_native_wave(mflat)
    log.info(f"writing master flat field to {out_mflat}")
    mflat.writeFitsData(out_mflat, replace_masked=False)

    # create lvmFlat objects
    log.info(f"creating lvmTFlat products for {len(twilights)}:")
    lvmflats = []
    for i, twilight in enumerate(twilights):
        expnum = twilight._header["EXPOSURE"]
        twilight.set_wave_trace(mwave)
        twilight.set_lsf_trace(mlsf)
        if twilight._header.get("WAVEREC", False):
            raise ValueError(f"twilight of {expnum = } is wavelength-rectified")

        twilight /= mflat
        lvmflat = lvmFlat(data=twilight._data, error=twilight._error, mask=twilight._mask, header=twilight._header,
                          cent_trace=mcent, width_trace=mwidth,
                          wave_trace=mwave, lsf_trace=mlsf,
                          superflat=mflat._data, slitmap=twilight._slitmap)
        log.info(f"  writing lvmTFlat to {out_lvmflats[i]}")
        lvmflat.writeFitsData(out_lvmflats[i])
        lvmflats.append(lvmflat)

        log.info(f"  resampling exposure {expnum = } to rectified wavelength grid")
        lvmflat_r = lvmflat.rectify_wave(wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)

        log.info(f"  removing factors with parameters: {cwave = :.2f}, {dwave = :.2f} Angstrom, {coadd_method = } and fibers {groupby = }")
        x, y, z, coeffs, factors = lvmflat_r.fit_ifu_gradient(
            guess_coeffs=[1,0,0,0], fixed_coeffs=[0,1,2,3],
            cwave=cwave, dwave=dwave, coadd_method=coadd_method, groupby=groupby)
        lvmflat_r = lvmflat_r.remove_ifu_gradient(coeffs=coeffs, factors=factors, groupby=groupby)

        ax_twi = fig.add_subplot(gs[nrows+i, :], sharex=locals().get("ax_twi"))
        ax_twi.tick_params(labelbottom=ax_twi.get_subplotspec().is_last_row())
        ax_twi.set_ylim(0.98, 1.02)
        if ax_twi.get_subplotspec().is_last_row():
            ax_twi.set_xlabel("Fiber ID", fontsize="large")
        ax_twi.set_ylabel("Normalized counts")
        ax_twi.set_title(f"{expnum = }", fontsize="large", loc="left")
        slit(rss=lvmflat_r, cwave=cwave, dwave=dwave, comb_stat=np.nanmean, data=lvmflat_r._data, ax=ax_twi, margins_percent=[0.2,0.5,1.0])
    save_fig(fig, out_mflat, to_display=display_plots, figure_path="qa", label="fiberflat_consistency")

    return mflat, fflats, twilights, lvmflats

def _reference_error(rss, ref_kind, interpolate_invalid=False):
    if isinstance(rss, RSS):
        error = rss._error.copy()
    elif isinstance(rss, np.ndarray):
        error = np.atleast_2d(rss).T * 0.0
    else:
        raise TypeError(f"Invalid type for `rss`: {type(rss)}. Expected lvmdrp.core.rss.RSS or numpy array")

    if callable(ref_kind):
        ref_error = np.sqrt(ref_kind(error**2, axis=0))
    elif isinstance(ref_kind, int):
        ref_error = error[ref_kind, :]
    else:
        raise TypeError(f"Invalid type for `ref_kind`: {type(ref_kind)}. Expected an integer or a callable(x, axis)")

    if interpolate_invalid:
        mask = np.isfinite(ref_error)
        ref_error = np.interp(rss._wave, rss._wave[mask], ref_error[mask])

    return ref_error

def _reference_fiber(rss, ref_kind, interpolate_invalid=True, ax=None):

    if isinstance(rss, RSS):
        data = rss._data.copy()
    elif isinstance(rss, np.ndarray):
        data = np.atleast_2d(rss).T
    else:
        raise TypeError(f"Invalid type for `rss`: {type(rss)}. Expected lvmdrp.core.rss.RSS or numpy array")

    if callable(ref_kind):
        ref_spectrum = ref_kind(data, axis=0)
    elif isinstance(ref_kind, int):
        ref_spectrum = data[ref_kind, :]
    else:
        raise TypeError(f"Invalid type for `ref_kind`: {type(ref_kind)}. Expected an integer or a callable(x, axis)")

    if interpolate_invalid:
        mask = np.isfinite(ref_spectrum)
        ref_spectrum = np.interp(rss._wave, rss._wave[mask], ref_spectrum[mask])

    ref_error = _reference_error(rss, ref_kind, interpolate_invalid)

    if ax is not None:
        ax.step(rss._wave, ref_spectrum, where="mid", color="0.2", lw=1)
        ylims = ax.get_ylim()
        ax.fill_between(rss._wave, ref_spectrum-ref_error, ref_spectrum+ref_error, step="mid", color="0.2", lw=0, alpha=0.2, zorder=-999)
        if "mask" in locals():
            ax.vlines(rss._wave[~mask], *ax.get_ylim(), lw=2, color="0.8", zorder=-9)
        ax.set_ylim(*ylims)

    return ref_spectrum

def get_flatfield(rss, ref_kind=lambda x: biweight_location(x, ignore_nan=True), norm_column=None, norm_kind=np.nanmean, interpolate_invalid=True, smoothing=0.1, axs={}):

    ref_fiber = _reference_fiber(rss, ref_kind=ref_kind, interpolate_invalid=interpolate_invalid, ax=axs.pop("ref_fiber", None))

    flatfield_r = rss / ref_fiber
    if norm_column is not None:
        if callable(norm_kind):
            normalization = norm_kind(flatfield_r._data[:, norm_column], axis=0)
        elif isinstance(norm_kind, int):
            normalization = flatfield_r._data[norm_kind, norm_column]
        else:
            raise TypeError(f"Invalid type for `norm_kind`: {type(norm_kind)}. Expected an integer or a callable(x, axis)")
    else:
        normalization = 1.0

    flatfield_r /= normalization
    flatfield = copy(flatfield_r)

    if interpolate_invalid or smoothing > 0:
        flatfield = flatfield.interpolate_data(axis="X")
    if smoothing > 0:
        nyq = 1 / (rss._wave[1] - rss._wave[0])
        for ifiber in range(flatfield_r._fibers):
            spec = flatfield.getSpec(ifiber)
            if spec._mask.all():
                continue

            select = np.isfinite(spec._data)
            spec._data[select] = butter_lowpass_filter(median_filter(spec._data[select], 51), smoothing, nyq)
            spec._error[select] = np.sqrt(butter_lowpass_filter(median_filter(spec._error[select]**2, 51), smoothing, nyq))

            flatfield.setSpec(ifiber, spec)

    axs_sm = axs.get("smoothing", None)
    if axs_sm is not None:
        wave_ = np.repeat([rss._wave], rss._fibers, axis=0)
        zscores = (flatfield._data - flatfield_r._data) / np.sqrt(flatfield_r._error**2 + flatfield._error**2)
        mean = bn.nanmean(zscores, axis=0)
        stddev = bn.nanstd(zscores, axis=0)
        axs_sm.axhspan(-1.0, +1.0, lw=0, fc="0.7", alpha=0.5)
        axs_sm.axhline(ls="--", lw=0.5, color="w")
        axs_sm.plot(wave_.T, zscores.T, ",", color="0.2", alpha=0.1)
        axs_sm.plot(rss._wave, mean, lw=0.5, color="0.8")
        axs_sm.plot(rss._wave, mean-stddev, lw=0.5, color="0.8")
        axs_sm.plot(rss._wave, mean+stddev, lw=0.5, color="0.8")
        axs_sm.set_ylim(-1.5, +1.5)

    return flatfield, flatfield_r, ref_fiber, normalization

def get_flatfield_sequence(rsss, ref_kind=lambda x: biweight_location(x, ignore_nan=True), norm_column=None, norm_kind=np.nanmean, interpolate_invalid=True):
    flatfields, flatfields_r = [], []
    ref_fibers = np.zeros((len(rsss), rsss[0]._pixels.size))
    normalizations = np.zeros(len(rsss))
    for i, rss in enumerate(rsss):
        flatfield, flatfield_r, ref_fibers[i], normalizations[i] = get_flatfield(rss, ref_kind=ref_kind, norm_column=norm_column, norm_kind=norm_kind, interpolate_invalid=interpolate_invalid)
        flatfields.append(flatfield)
        flatfields_r.append(flatfield_r)
    return flatfields, flatfields_r, ref_fibers, normalizations

def normalize_spec(rss, cwave, dwave=8, norm_stat=np.nanmean):
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

def iterate_gradient_fit(rss, cwave, dwave=8, guess_coeffs=[1,2,3,0], fixed_coeffs=[3], groupby="spec", coadd_method="average", niter=10, thresholds=(0.005, 0.005), axs=None):

    rss_g = copy(rss)

    # first gradient fit
    x, y, z, coeffs, factors = rss_g.fit_ifu_gradient(cwave=cwave, dwave=dwave, guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs, groupby=groupby, coadd_method=coadd_method)
    _, gradient_final, factors_final = rss_g.eval_ifu_gradient(coeffs, factors, groupby, normalize=True)
    log.info(f"initial factors         = {np.round(factors, 4)}")
    log.info(f"initial gradient across = {gradient_final.max()/gradient_final.min():.4f}")

    fthr, gthr = thresholds

    i = 0
    log.info(f"iteratively fitting gradient for up to {niter = }")
    while True:
        # gradient residual
        rss_g = rss_g.remove_ifu_gradient(coeffs=coeffs, factors=factors, groupby=groupby)
        x, y, z_res, coeffs, factors = rss_g.fit_ifu_gradient(cwave=cwave, dwave=dwave,
                                                              guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs,
                                                              groupby=groupby, coadd_method=coadd_method)
        _, gradient_residual, factors_residual = rss_g.eval_ifu_gradient(coeffs, factors, groupby, normalize=True)
        fres, gres = np.abs(factors_residual.max() - 1), np.abs(gradient_residual.max() / gradient_residual.min() - 1)

        log.info(f" iteration {i+1}/{niter}")
        log.info(f"     factors            = {np.round(factors_residual[::rss._fibers//factors.size], 4)}")
        log.info(f"     gradient across    = {gradient_residual.max()/gradient_residual.min():.4f}")
        log.info(f"     factors residuals  = {fres:.4f}")
        log.info(f"     gradient residuals = {gres:.4f}")
        if (fthr > fres and gthr > gres) or i+1 == niter:
            break

        # propagate residuals
        gradient_final *= gradient_residual
        factors_final *= factors_residual
        i += 1
    if i == 0:
        log.info("no further iterations needed")

    log.info(f"fitted factors          = {np.round(factors_final[::rss._fibers//factors.size], 4)}")
    log.info(f"fitted gradient across  = {gradient_final.max()/gradient_final.min():.4f}")

    rss_gradient = rss / rss_g
    _, _, _, coeffs, factors = rss_gradient.fit_ifu_gradient(cwave=cwave, dwave=dwave, guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs, groupby=groupby, coadd_method=coadd_method)

    if isinstance(axs, (list, tuple)) and len(axs) == 2:
        axs_fin, axs_res = axs
        plot_gradient_fit(rss._slitmap, z, gradient_final, factors_final, telescope="Sci", marker_size=15, axs=axs_fin, labels=True)
        plot_gradient_fit(rss_g._slitmap, z_res, gradient_residual, factors_residual, telescope="Sci", marker_size=15, axs=axs_res, labels=False)
    elif isinstance(axs, plt.Axes):
        plot_gradient_fit(rss._slitmap, z, gradient_final, factors_final, telescope="Sci", marker_size=15, axs=axs_fin, labels=True)

    return x, y, z, coeffs, factors

def fit_fiberflat(in_rss, out_flat, out_rss, ref_kind=600, guess_coeffs=[1,2,3,0], fixed_coeffs=[3], groupby="spec",
                  norm_cwave=None, norm_dwave=8, coadd_method="average", smoothing=0.1, interpolate_invalid=True, display_plots=False):
    """Creates a flatfield given a flat (twilight, dome) exposure

    The input RSS needs to be wavelength calibrated, rectified and LSF matched

    The following steps are followed:
        - For each exposure:
            * Normalize by the chosen reference fiber
            * Normalize by spectrograph at `norm_cwave` (chosen to be a strong sky line)
        - Combine the resulting flatfields into a master
        - For each exposure:
            * Apply flatfield to each twilight exposure
            * Fit gradient and correct each telescope IFU by it
        - Combine gradient corrected flatfields into final master

    **NOTE:** This algorithm proposed by Guillermo Blanc, will produce a
    flatfield that accounts for the spectrograph shutter timing issue at the
    cost of a flatfield that doesn't fully account for spectrograph to
    spectrograph throughput variations. As a consequence the resulting
    flatfield will have to be corrected using the same sky line at `norm_cwave`
    extracted and measured during science reductions, where we expect shutter
    timing issues to be within 1%.

    **NOTE:** This same procedure could be applied to dome flats, provided we
    can use >80s exposures to reliably measure the spectrograph to spectrograph
    throughput. LDLS seem to be the best option.

    Parameters
    ----------
    in_rss : str
        Path to inpput flat exposure (e.g., twilight, dome)
    out_flat : str
        Path to output flatfield
    out_rss : str, optional
        Path to self-flatfielded flat exposure
    ref_kind : int|callable, optional
        Position of the reference fiber in RSS or a callable to produce one, by default 600
    groupby : str, optional
        Fit factors to fiber groups by 'spec' or 'quad', by default 'spec'
    norm_cwave : int|None, optional
        Normalization wavelength, by default None
    norm_dwave : int, optional
        Normalization wavelength window around `norm_cwave`, by default 8 Angstroms
    coadd_method : str, optional
        Coadding method used during IFU gradient fitting, by default 'average'
    interpolate_invalid : bool, optional
        Interpolate invalid pixels (NaN, infinity), by default True
    smoothing : float, optional
        Perform a low-pass filtering to remove artifacts and denoise, by default 0.1
    display_plots : bool, optional
        Whether to display plots or not, by default False

    Returns
    -------
    lvmdrp.core.rss.RSS
        Flatfield with not gradient/factor corrections
    lvmdrp.core.rss.RSS
        Flatfield with gradient/factor corrections
    lvmdrp.core.rss.RSS
        Flatfielded flat exposure
    """

    log.info(f"loading flat exposure at {in_rss}")
    rss = RSS.from_file(in_rss)
    expnum = rss._header["EXPOSURE"]
    imagetyp = rss._header["IMAGETYP"]
    channel = rss._header["CCD"]
    unit = rss._header['BUNIT']

    fig = plt.figure(figsize=(14,3*5))
    fig.suptitle(f"Fiber flatfield from '{imagetyp}' exposure, {channel = }, {expnum = }", fontsize="xx-large")
    gs_sed = GridSpec(5, 5, hspace=0.3, wspace=0.01, left=0.07, right=0.99, bottom=0.1, top=0.9, figure=fig)
    gs_ifu = GridSpec(5, 5, hspace=0.01, wspace=0.01, left=0.07, right=0.99, bottom=0.01, top=0.9, figure=fig)
    ax_ref = fig.add_subplot(gs_sed[:2, :])
    ax_ref.tick_params(labelbottom=False)
    ax_ref.set_title(f"Reference fiber using {ref_kind.__name__ if callable(ref_kind) else ref_kind}", loc="left", fontsize="large")
    ax_ref.set_ylabel(f"Counts ({unit})", fontsize="large")
    ax_ref.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))
    ax_ref.axvspan(norm_cwave-norm_dwave//2, norm_cwave+norm_dwave//2, color="tab:blue", alpha=0.5, lw=0)
    ax_smo = fig.add_subplot(gs_sed[-3, :], sharex=ax_ref)
    ax_smo.set_title(f"Smoothing quality with {smoothing = }", loc="left", fontsize="large")
    ax_smo.set_ylabel("Z-scores", fontsize="large")
    ax_smo.set_xlabel("Wavelength (Angstroms)", fontsize="large")
    axs_fin = [fig.add_subplot(gs_ifu[-2, j]) for j in range(5)]
    axs_fin[0].set_ylabel("gradient correction", fontsize="large")
    axs_res = [fig.add_subplot(gs_ifu[-1, j]) for j in range(5)]
    axs_res[0].set_ylabel("gradient residual", fontsize="large")

    log.info(f"calculating flatfield from '{imagetyp}' exposure, {channel = }, {expnum = }, with parameters: {ref_kind = }, {interpolate_invalid = }, {smoothing = }")
    flat, _, ref_fiber, _ = get_flatfield(rss=rss, ref_kind=ref_kind, interpolate_invalid=interpolate_invalid, smoothing=smoothing, axs={"ref_fiber": ax_ref, "smoothing": ax_smo})

    log.info(f"fitting and correcting IFU gradient and '{groupby}' factors @ {norm_cwave:.2f} Angstroms")
    # fit gradient with spectrograph normalizations (make n-iterations of this or stop when gradient is <1% across)
    x, y, z_ori, coeffs, factors = iterate_gradient_fit(rss, cwave=norm_cwave, dwave=norm_dwave, coadd_method=coadd_method, groupby=groupby,
                                                        guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs, axs=(axs_fin,axs_res))

    # apply gradient correction
    rss_g = rss.remove_ifu_gradient(coeffs=coeffs, factors=factors, groupby=groupby)
    # get corrected flatfield
    flat_g, _, _, _ = get_flatfield(rss=rss_g, ref_kind=ref_kind, interpolate_invalid=interpolate_invalid, smoothing=smoothing)

    save_fig(fig, out_flat, to_display=display_plots, figure_path="qa", label="twilight_fiberflat")

    log.info(f"writing fiber flatfield to {out_flat}")
    flat_g.setHdrValue(f"HIERARCH {channel} FIBERFLAT CWAVE", norm_cwave, "norm. wavelength [Angstrom]")
    flat_g.setHdrValue(f"HIERARCH {channel} FIBERFLAT DWAVE", norm_dwave, "norm. window width [Angstrom]")
    flat_g.setHdrValue(f"HIERARCH {channel} FIBERFLAT COADD", coadd_method, "coadding method")
    flat_g.setHdrValue(f"HIERARCH {channel} FIBERFLAT SKYCORR", False, "fiberflat skyline-corrected?")
    flat_g.setHdrValue(f"HIERARCH {channel} FIBERFLAT GROUPBY", groupby, "fiber grouping")
    flat_g.writeFitsData(out_flat)
    log.info(f"writing flatfielded exposure to {out_rss}")
    (rss_g/flat_g).writeFitsData(out_rss)

    return flat, flat_g, rss_g, coeffs, factors

def fit_skyline_flatfield(in_sciences, in_mflat, out_mflat, sky_cwave, cont_cwave, dwave=8, guess_coeffs=[1,2,3,0], fixed_coeffs=[3], groupby="spec",
                          quantiles=(5,97), nsigma=1, comb_method="median", sky_fibers_only=False, force_correction=False, display_plots=False):

    log.info(f"loading master fiberflat at {in_mflat}")
    mflat = RSS.from_file(in_mflat)
    channel = mflat._header["CCD"]
    coadd_method = mflat._header[f"{channel} FIBERFLAT COADD"]

    # verify groupy
    groupby_hdr = mflat._header.get(f"{channel} FIBERFLAT GROUPBY")
    if groupby != groupby_hdr:
        log.warning(f"requested {groupby = } but header says {groupby_hdr}, assuming header value")
        groupby = groupby_hdr
    fiber_groups = mflat._get_fiber_groups(by=groupby)

    # skip correction if already done and no force is required
    # undo correction if done and force is required
    if mflat._header.get(f"{channel} FIBERFLAT SKYCORR"):
        if not force_correction:
            log.info("fiber flat already corrected using sky lines, skipping")
            return mflat, np.ones(mflat._fibers, dtype="float")
        else:
            factors = list(mflat._header[f"{channel} FIBERFLAT FACTOR?"].values())
            log.info(f"requested {force_correction = }; undoing '{groupby}' correction with: {factors}")
            flatfield_corr = IFUGradient.ifu_factors(factors, fiber_groups)
            mflat /= flatfield_corr[:, None]
            mflat.setHdrValue("HIERARCH {channel} FIBERFLAT SKYCORR", False)

    log.info(f"loading {len(in_sciences)} science exposures")
    sciences = [RSS.from_file(in_science) for in_science in in_sciences]

    if isinstance(sciences, list):
        log.info(f"fitting sky line correction using {len(sciences)} science frames")
    elif isinstance(sciences, RSS):
        log.info(f"fitting sky line correction using a single science exposure: {sciences}")
        science = [sciences]
    else:
        raise TypeError(f"Invalid type for `sciences`: {type(sciences)}. Valid types are lvmdrp.core.rss.RSS and list[lvmdrp.core.rss.RSS]")

    fig = plt.figure(figsize=(14,3*(3+len(sciences))))
    fig.suptitle(f"Fiber flatfield correction for {channel = } around sky line @ {sky_cwave:.2f} Angstroms", fontsize="xx-large")
    gs_gra = GridSpec(3+len(sciences), 5, hspace=0.01, wspace=0.01, left=0.07, right=0.99, figure=fig)
    gs_cor = GridSpec(3+len(sciences), 5, hspace=0.5, wspace=0.01, left=0.07, right=0.99, figure=fig)

    sciences_g, factors = [], []
    log.info(f"going to process {len(sciences)} science exposures in {channel = }:")
    for i, science in enumerate(sciences):
        axs = [fig.add_subplot(gs_gra[i, j]) for j in range(5)]

        x, y, _, coeffs, factor, science_g = science.measure_skyline_flatfield(
            mflat=mflat, sky_cwave=sky_cwave, cont_cwave=cont_cwave, dwave=dwave,
            quantiles=None, guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs, groupby=groupby,
            axs=axs, labels=i==0)

        factors.append(factor)
        sciences_g.append(science_g)

    factor_mean = np.mean(factors, axis=0)
    factor_sdev = np.std(factors, axis=0)
    log.info(f"average factors = {np.round(factor_mean, 4)} +/- {np.round(factor_sdev, 4)}")

    if len(sciences) > 2:
        zscore = np.abs(np.asarray(factors) - factor_mean) / factor_sdev
        keep = (zscore <= nsigma).all(axis=1)
        nkeep = keep.sum()
        log.info(f"rejecting {len(sciences) - nkeep} outlying (> {nsigma} sigma) science exposures")
    else:
        keep = np.ones(len(sciences), dtype="bool")
        nkeep = keep.sum()

    log.info(f"combining {nkeep} gradient corrected science frames using {comb_method = }")
    science = RSS()
    science.combineRSS([science_g for i, science_g in enumerate(sciences_g) if keep[i]], method=comb_method)
    science.apply_pixelmask()

    log.info(f"measuring continuum @ {cont_cwave:.2f} Angstroms")
    ax_con = fig.add_subplot(gs_cor[-2, :])
    ax_con.axvline(sky_cwave, lw=1, ls="--", color="0.2")
    rejects = science.reject_fibers(cwave=cont_cwave, quantiles=quantiles, ax=ax_con)
    science._data[rejects, :] = np.nan
    science._error[rejects, :] = np.nan
    science._mask[rejects, :] = True
    log.info(f"rejected {rejects.sum()} fibers outside {quantiles = }")

    log.info(f"validating gradient removal around sky line @ {sky_cwave:.2f} Angstroms")
    axs = [fig.add_subplot(gs_gra[-3, j]) for j in range(5)]
    axs[0].set_ylabel("combined exposure", fontsize="large")

    x, y, skyline_slit, coeffs, factor = science.fit_ifu_gradient(cwave=sky_cwave, dwave=dwave, groupby=groupby,
                                                                  guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs, coadd_method="fit")
    gradient_res = IFUGradient.ifu_gradient(coeffs, x=x, y=y, normalize=True)
    factors_final = IFUGradient.ifu_factors(factor, fiber_groups, normalize=True)
    plot_gradient_fit(science._slitmap, skyline_slit, gradient_res, factors_final, telescope="Sci", marker_size=15, axs=axs, labels=False)
    log.info(f"all fibers factors       = {np.round(factor, 4)}")
    log.info(f"residual gradient across = {bn.nanmax(gradient_res)/bn.nanmin(gradient_res):.4f}")

    if sky_fibers_only:
        log.info(f"measuring sky line @ {sky_cwave:.2f} Angstroms on combined science frame")
        skyline_slit, x, y = science.fit_lines_slit(cwaves=sky_cwave, dwave=dwave, select_fibers=science._slitmap["targettype"]=="SKY", return_xy=True)
        skyline_slit /= bn.nanmedian(skyline_slit)

        log.info("calculating spectrograph corrections using only sky fibers")
        slitmap = science._slitmap
        factor = np.zeros(3, dtype="float")
        for i in range(3):
            factor[i] = bn.nanmedian(skyline_slit[slitmap["spectrographid"]==i+1])
        log.info(f"sky fiber factors    = {np.round(factor, 4)}")

    flatfield_corr = IFUGradient.ifu_factors(factor, fiber_groups)

    science_corr = science / flatfield_corr[:, None]
    skyline_slit = science_corr.fit_lines_slit(cwaves=sky_cwave, select_fibers="Sci")
    fiberids = science_corr._slitmap["fiberid"].data
    ax_cor = fig.add_subplot(gs_cor[-1, :])
    ax_cor.set_title("Flatfielded slit of combined frame", loc="left")
    ax_cor.set_xlabel("Fiber ID", fontsize="large")
    ax_cor.set_ylabel("Normalized counts", fontsize="large")
    ax_cor.set_ylim(0.92, 1.08)
    slit(x=fiberids, y=skyline_slit, data=science_corr._data, ax=ax_cor)

    save_fig(fig, out_mflat, to_display=display_plots, figure_path="qa", label="flat_correction")

    mflat_corr = mflat * flatfield_corr[:, None]
    mflat_corr.setHdrValue(f"HIERARCH {channel} FIBERFLAT SKYCORR", True, "fiberflat skyline-corrected?")
    mflat_corr.setHdrValue(f"HIERARCH {channel} FIBERFLAT GROUPBY", groupby, "fiber grouping")
    for i, f in enumerate(factor):
        mflat_corr.setHdrValue(f"HIERARCH {channel} FIBERFLAT FACTOR{i+1}", f, f"{groupby}{i+1} factor")
    log.info(f"writing corrected master fiberflat to {out_mflat}")
    mflat_corr.writeFitsData(out_mflat)

    # plot flatfielding validation on science exposures
    TEST_WAVES = {
        "b": [3700, 4200, 4800, 5300],
        "r": [5900, 6300, 6800, 7200],
        "z": [7800, 8300, 8900, 9500]
    }
    test_cwaves = TEST_WAVES[channel]
    test_sciences = [science_g / flatfield_corr[:, None] for science_g in sciences_g] + [science_corr]

    fig, axs = create_subplots(
        to_display=display_plots,
        nrows=len(test_sciences), ncols=len(test_cwaves),
        figsize=(4*len(test_cwaves),4*len(test_sciences)),
        sharex=True, sharey=True, layout="constrained", flatten_axes=False)
    if axs.ndim == 1:
        axs = np.atleast_2d(axs).T
    fig.suptitle(f"validating flat-fielded science exposures in {channel = }", fontsize="xx-large")
    for i in range(len(test_sciences)):
        plot_flatfield_validation(fframe=test_sciences[i], cwaves=test_cwaves, dwave=dwave, axs=axs[i], coadd_method=coadd_method)
        expnum = test_sciences[i]._header["EXPOSURE"]
        if i == len(test_sciences) - 1:
            axs[i,0].set_ylabel("combined exposure", fontsize="large")
        else:
            axs[i,0].set_ylabel(f"{expnum = }", fontsize="large")
    save_fig(fig, out_mflat, to_display=display_plots, figure_path="qa", label="fiberflat_validation")

    return mflat_corr, flatfield_corr
