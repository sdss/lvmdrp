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
from matplotlib.gridspec import GridSpec

import bottleneck as bn
from lvmdrp import log
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.rss import RSS, lvmFrame
from lvmdrp.core.fluxcal import butter_lowpass_filter
from lvmdrp.core.fit_profile import IFUGradient
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.plot import plt, slit, plot_gradient_fit, save_fig
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
    mflat._error = np.sqrt(biweight_location([fflat._error**2 for fflat in fflats], axis=0, ignore_nan=True))

    # mask invalid pixels
    mflat._mask |= np.isnan(mflat._data) | (mflat._data <= 0) | np.isinf(mflat._data)
    mflat._mask |= np.isnan(mflat._error) | (mflat._error <= 0) | np.isinf(mflat._error)

    # interpolate masked fibers if any remaining
    mflat = mflat.interpolate_data(axis="X")
    mflat = mflat.interpolate_data(axis="Y")

    mflat.set_wave_trace(TraceMask.from_spectrographs(*[TraceMask.from_file(in_wave) for in_wave in in_waves]))
    mflat.set_lsf_trace(TraceMask.from_spectrographs(*[TraceMask.from_file(in_lsf) for in_lsf in in_lsfs]))
    mflat = to_native_wave(mflat)
    mflat.writeFitsData(out_fiberflat, replace_masked=False)

    return mflat

def _reference_fiber(rss, ref_kind, interpolate_invalid=False, ax=None):

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

    if interpolate_invalid:
        mask = np.isfinite(ref_fiber)
        ref_fiber = np.interp(rss._wave, rss._wave[mask], ref_fiber[mask])

    return ref_fiber

def get_flatfield(rss, ref_kind=lambda x: biweight_location(x, ignore_nan=True), norm_column=None, norm_kind=np.nanmean, interpolate_invalid=False, smooth=False):
    ref_fiber = _reference_fiber(rss, ref_kind=ref_kind, interpolate_invalid=interpolate_invalid)

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

    if interpolate_invalid or smooth:
        flatfield = flatfield.interpolate_data(axis="X")
    if smooth:
        for ifiber in range(flatfield._fibers):
            spec = flatfield.getSpec(ifiber)
            if spec._mask.all():
                continue

            select = np.isfinite(spec._data)
            spec._data[select] = butter_lowpass_filter(spec._data[select], 0.1, 2)

            flatfield.setSpec(ifiber, spec)

    return flatfield, ref_fiber, normalization

def get_flatfield_sequence(rsss, ref_kind=lambda x: biweight_location(x, ignore_nan=True), norm_column=None, norm_kind=np.nanmean, interpolate_invalid=False):
    flatfields = []
    ref_fibers = np.zeros((len(rsss), rsss[0]._pixels.size))
    normalizations = np.zeros(len(rsss))
    for i, rss in enumerate(rsss):
        flatfield, ref_fibers[i], normalizations[i] = get_flatfield(rss, ref_kind=ref_kind, norm_column=norm_column, norm_kind=norm_kind, interpolate_invalid=interpolate_invalid)
        flatfields.append(flatfield)
    return flatfields, ref_fibers, normalizations

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

def iterate_gradient_fit(rss, cwave, dwave=8, groupby="spec", coadd_method="average", niter=10, thresholds=(0.005, 0.005), axs=None):

    rss_g = copy(rss)

    # first gradient fit
    x, y, z, coeffs, factors = rss_g.fit_ifu_gradient(cwave=cwave, dwave=dwave, groupby=groupby, coadd_method=coadd_method)
    _, gradient_final, factors_final = rss_g.eval_ifu_gradient(coeffs, factors, groupby, normalize=True)
    log.info(f"initial factors         = {np.round(factors, 4)}")
    log.info(f"initial gradient across = {gradient_final.max()/gradient_final.min():.4f}")

    fthr, gthr = thresholds

    i = 0
    log.info(f"iteratively fitting gradient for up to {niter = }")
    while True:
        # gradient residual
        rss_g = rss_g.remove_ifu_gradient(coeffs=coeffs, factors=factors, groupby=groupby)
        x, y, z_res, coeffs, factors = rss_g.fit_ifu_gradient(cwave=cwave, dwave=dwave, groupby=groupby, coadd_method=coadd_method)
        _, gradient_residual, factors_residual = rss_g.eval_ifu_gradient(coeffs, factors, groupby, normalize=True)
        fres, gres = np.abs(factors_residual.max() - 1), np.abs(gradient_residual.max() / gradient_residual.min() - 1)

        log.info(f" iteration {i+1}/{niter}")
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
    _, _, _, coeffs, factors = rss_gradient.fit_ifu_gradient(cwave=cwave, dwave=dwave, groupby=groupby, coadd_method=coadd_method)

    if isinstance(axs, (list, tuple)) and len(axs) == 2:
        axs_fin, axs_res = axs
        plot_gradient_fit(rss._slitmap, z, gradient_final, factors_final, telescope="Sci", marker_size=15, axs=axs_fin, labels=True)
        plot_gradient_fit(rss_g._slitmap, z_res, gradient_residual, factors_residual, telescope="Sci", marker_size=15, axs=axs_res, labels=False)
    elif isinstance(axs, plt.Axes):
        plot_gradient_fit(rss._slitmap, z, gradient_final, factors_final, telescope="Sci", marker_size=15, axs=axs_fin, labels=True)

    return x, y, z, coeffs, factors

def fit_fiberflat(in_rss, out_flat, out_rss, ref_kind=600, groupby="spec", norm_cwave=None, norm_dwave=8, interpolate_invalid=True, smooth=True, display_plots=False):
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
        Normalization wavelength window around `norm_cwave`, by default 8 Angstrom
    interpolate_invalid : bool, optional
        Interpolate invalid pixels (NaN, infinity), by default True
    smooth : bool, optional
        Perform a low-pass filtering to remove artifacts and denoise, by default True
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

    log.info(f"calculating flatfield from '{imagetyp}' exposure, {channel = }, {expnum = }")
    flat, ref_fiber, _ = get_flatfield(rss=rss, ref_kind=ref_kind, interpolate_invalid=interpolate_invalid, smooth=smooth)

    fig = plt.figure(figsize=(14,3*3))
    fig.suptitle(f"Fiber flatfield from '{imagetyp}' exposure, {channel = }, {expnum = }", fontsize="xx-large")
    gs_ref = GridSpec(3, 5, hspace=0.7, wspace=0.01, left=0.07, right=0.99, bottom=0.01, top=0.9, figure=fig)
    gs_ifu = GridSpec(3, 5, hspace=0.01, wspace=0.01, left=0.07, right=0.99, bottom=0.01, top=0.9, figure=fig)
    ax_ref = fig.add_subplot(gs_ref[0, :])
    axs_fin = [fig.add_subplot(gs_ifu[1, j]) for j in range(5)]
    axs_fin[0].set_ylabel("gradient correction", fontsize="large")
    axs_res = [fig.add_subplot(gs_ifu[2, j]) for j in range(5)]
    axs_res[0].set_ylabel("gradient residual", fontsize="large")
    ax_ref.step(flat._wave, ref_fiber, where="mid", lw=1)

    log.info(f"fitting and correcting IFU gradient and '{groupby}' factors @ {norm_cwave:.2f} Angstrom")
    # fit gradient with spectrograph normalizations (make n-iterations of this or stop when gradient is <1% across)
    x, y, z_ori, coeffs, factors = iterate_gradient_fit(flat, groupby=groupby, cwave=norm_cwave, dwave=norm_dwave, axs=(axs_fin,axs_res))
    # apply gradient correction
    rss_g = rss.remove_ifu_gradient(coeffs=coeffs, factors=factors, groupby=groupby)
    # get corrected flatfield
    flat_g, _, _ = get_flatfield(rss=rss_g, ref_kind=ref_kind, interpolate_invalid=interpolate_invalid, smooth=smooth)

    ax_ref.set_xlabel("Wavelength (Angstrom)", fontsize="large")
    ax_ref.set_ylabel(f"Counts ({unit})", fontsize="large")
    ax_ref.set_title("Reference fiber", loc="left", fontsize="large")
    ax_ref.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))
    save_fig(fig, out_flat, to_display=display_plots, figure_path="qa", label="twilight_fiberflat")

    log.info(f"writing fiber flatfield to {out_flat}")
    flat_g.writeFitsData(out_flat)
    log.info(f"writing flatfielded exposure to {out_rss}")
    (rss_g/flat_g).writeFitsData(out_rss)

    return flat, flat_g, rss_g, coeffs, factors

def _choose_sky(rss):
    telescope = "SkyE" if abs(rss._header["WAVE HELIORV_SKYE"]) < abs(rss._header["WAVE HELIORV_SKYW"]) else "SkyW"
    return telescope

def fit_skyline_flatfield(in_sciences, in_mflat, out_mflat, cwave, dwave=8, norm_fibers=None, method="median", sky_fibers_only=False, display_plots=False):

    log.info(f"loading {len(in_sciences)} science exposures")
    sciences = [RSS.from_file(in_science) for in_science in in_sciences]

    log.info(f"loading master fiberflat at {in_mflat}")
    mflat = RSS.from_file(in_mflat)
    channel = mflat._header["CCD"]

    if isinstance(sciences, list):
        log.info(f"fitting sky line correction using {len(sciences)} science frames")
    elif isinstance(sciences, RSS):
        log.info(f"fitting sky line correction using a single science exposure: {sciences}")
        science = [sciences]
    else:
        raise TypeError(f"Invalid type for `sciences`: {type(sciences)}. Valid types are lvmdrp.core.rss.RSS and list[lvmdrp.core.rss.RSS]")

    fig = plt.figure(figsize=(14,3*(1+len(sciences))))
    fig.suptitle(f"Fiber flatfield correction for {channel = } around sky line @ {cwave:.2f} Angstrom", fontsize="xx-large")
    gs_gra = GridSpec(2+len(sciences), 5, hspace=0.01, wspace=0.01, left=0.07, right=0.99, bottom=0.03, top=0.97, figure=fig)
    gs_cor = GridSpec(2+len(sciences), 5, hspace=0.7, wspace=0.01, left=0.07, right=0.99, bottom=0.03, top=0.97, figure=fig)

    fiber_groups = mflat._get_fiber_groups(by="spec")
    sciences_g, factors = [], []
    for i, science in enumerate(sciences):
        expnum = science._header["EXPOSURE"]
        imagetyp = science._header["IMAGETYP"]

        fscience = science / mflat

        log.info("measuring and correcting wavelength shifts")
        _, offsets_model = fscience.measure_wave_shifts(cwaves=cwave, dwave=dwave, smooth=True)
        mean_offset, std_offset = bn.nanmean(offsets_model), bn.nanstd(offsets_model)
        log.info(f"measured wavelength offsets: {mean_offset:.4f} +/- {std_offset:.4f}")
        fscience._wave_trace['COEFF'].data[:, 0] -= offsets_model
        wave_trace = TraceMask.from_coeff_table(fscience._wave_trace)
        fscience._wave = wave_trace.eval_coeffs()

        log.info(f"fitting gradient and factors around sky line @ {cwave:.2f} Angstrom for '{imagetyp}' exposure {expnum = }")
        x, y, z, coeffs, factor = fscience.fit_ifu_gradient(cwave=cwave, dwave=dwave, coadd_method="fit")
        gradient_model = IFUGradient.ifu_gradient(coeffs, x=x, y=y, normalize=True)
        factors.append(factor)
        log.info(f" factors         = {np.round(factor, 4)}")
        log.info(f" gradient across = {bn.nanmax(gradient_model)/bn.nanmin(gradient_model):.4f}")

        science_g = fscience.remove_ifu_gradient(coeffs=coeffs, factors=None)
        sciences_g.append(science_g)

        axs = [fig.add_subplot(gs_gra[i, j]) for j in range(5)]
        axs[0].set_ylabel(f"{expnum = }", fontsize="large")
        gradient_model = IFUGradient.ifu_gradient(coeffs, x, y, normalize=True)
        factors_model = IFUGradient.ifu_factors(factor, fiber_groups, normalize=True)
        plot_gradient_fit(fscience._slitmap, z, gradient_model, factors_model, telescope="Sci", marker_size=15, axs=axs, labels=i==0)

    factor_mean = np.mean(factors, axis=0)
    factor_sdev = np.std(factors, axis=0)
    log.info(f"average factors = {np.round(factor_mean, 4)} +/- {np.round(factor_sdev, 4)}")

    zscore = np.abs(np.asarray(factors) - factor_mean) / factor_sdev
    keep = (zscore <= 1).all(axis=1)
    log.info(f"rejecting {len(sciences) - keep.sum()} outlying (> 1sigma) science exposures")

    log.info(f"combining {keep.sum()} gradient corrected science frames using {method = }")
    science = RSS()
    science.combineRSS([science_g for i, science_g in enumerate(sciences_g) if keep[i]], method=method)
    science.apply_pixelmask()

    log.info(f"validating gradient removal around sky line @ {cwave:.2f} Angstrom")
    axs = [fig.add_subplot(gs_gra[-2, j]) for j in range(5)]
    axs[0].set_ylabel("combined exposure", fontsize="large")
    x, y, skyline_slit, coeffs, factor = science.fit_ifu_gradient(cwave=cwave, dwave=dwave, groupby="spec", coadd_method="fit")
    gradient_res = IFUGradient.ifu_gradient(coeffs, x=x, y=y, normalize=True)
    factors_final = IFUGradient.ifu_factors(factor, fiber_groups, normalize=True)
    plot_gradient_fit(science._slitmap, skyline_slit, gradient_res, factors_final, telescope="Sci", marker_size=15, axs=axs, labels=False)
    log.info(f"all fibers factors       = {np.round(factor, 4)}")
    log.info(f"residual gradient across = {bn.nanmax(gradient_res)/bn.nanmin(gradient_res):.4f}")

    if sky_fibers_only:
        log.info(f"measuring sky line @ {cwave:.2f} Angstrom on combined science frame")
        skyline_slit, x, y = science.fit_lines_slit(cwaves=cwave, dwave=dwave, select_fibers=science._slitmap["targettype"]=="SKY", return_xy=True)
        skyline_slit /= bn.nanmedian(skyline_slit)

        log.info("calculating spectrograph corrections using only sky fibers")
        slitmap = science._slitmap
        factor = np.zeros(3, dtype="float")
        for i in range(3):
            factor[i] = bn.nanmedian(skyline_slit[slitmap["spectrographid"]==i+1])
        log.info(f"sky fiber factors    = {np.round(factor, 4)}")

    flatfield_corr = IFUGradient.ifu_factors(factor, fiber_groups)
    flatfield_corr = np.repeat(factor, science._fibers / factor.size)

    science_corr = science / flatfield_corr[:, None]
    skyline_slit = science_corr.fit_lines_slit(cwaves=cwave, select_fibers="Sci")
    fiberids = science_corr._slitmap["fiberid"].data
    ax_cor = fig.add_subplot(gs_cor[-1, :])
    ax_cor.set_title("Flatfielded slit of combined frame", loc="left")
    ax_cor.set_xlabel("Fiber ID")
    ax_cor.set_ylabel("Normalized counts")
    ax_cor.set_ylim(0.92, 1.08)
    slit(x=fiberids, y=skyline_slit, ax=ax_cor)

    log.info(f"writing corrected master fiberflat to {out_mflat}")
    mflat_corr = mflat * flatfield_corr[:, None]
    mflat_corr.writeFitsData(out_mflat)

    return mflat_corr, flatfield_corr
