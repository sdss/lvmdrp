from copy import deepcopy
import warnings
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import numpy
import bottleneck as bn
from astropy.io import fits as pyfits
from astropy.stats import biweight_location
from numpy import polynomial
from scipy.linalg import norm
from scipy import signal, interpolate, integrate, ndimage, sparse
from scipy.ndimage import zoom, median_filter
from typing import List, Tuple

from lvmdrp.utils import gaussian
from lvmdrp.core import fit_profile
from lvmdrp.core import plot
from lvmdrp.core.header import Header

def adaptive_smooth(data, start_width, end_width):
    """
    Smooth an array with a filter that adapts in size from start_width to end_width.
    Parameters:
    - data (numpy.array): The input array to be smoothed.
    - start_width (int): The width of the smoothing kernel at the beginning of the array.
    - end_width (int): The width of the smoothing kernel at the end of the array.
    Returns:
    - numpy.array: The smoothed array.
    """
    # Create an array of kernel sizes changing linearly from start_width to end_width
    n_points = len(data)
    kernel_sizes = numpy.linspace(start_width, end_width, n_points).astype(int)
    # Output array initialization
    smoothed_data = numpy.zeros_like(data)
    # Apply varying filter
    for i in range(n_points):
        # Handle boundary effects by determining effective kernel size
        half_width = kernel_sizes[i] // 2
        start_index = max(0, i - half_width)
        end_index = min(n_points, i + half_width + 1)
        # Apply uniform filter to the local segment of the data
        smoothed_data[i] = bn.median(data[start_index:end_index])
    return smoothed_data

def find_continuum(spec_s,niter=15,thresh=0.8,median_box_max=100,median_box_min=1):
    """
    find the continuum from a spectrum by smoothing and masking the values above
    the smoothed version in an iterative way.
    Parameters:
    - data (numpy.array): The input array from which we would like to find the continuum.
    - niter  (int): Maximum number of iterations
    - thresh (float): Threshold to compare the smoothed an unsmoother version
    - median_box_max (float): Maximum size of the smoothing box
    - median_box_min (float): Minumum size of the smoothing box
    Returns:
    - numpy.array: continuum spectrum.
    """
    median_box=median_box_max
    spec_s_org = spec_s.copy()
    mask = (spec_s>(-1)*numpy.abs(numpy.min(spec_s)))
    #m_spec_s = adaptive_smooth(spec_s, median_box, int(median_box_max*0.5))
    m_spec_s = median_filter(spec_s, median_box)
    pixels = numpy.arange(0,spec_s.shape[0])
    i_len_in = len(spec_s_org[mask])
    for i in range(niter):
        mask = mask & (numpy.divide(m_spec_s, spec_s, where=spec_s != 0, out=numpy.zeros_like(spec_s)) > thresh)
        i_len = len(spec_s_org[mask])
        if (i_len==i_len_in):
            break
        else:
            i_len_in=i_len
        spec_s = numpy.interp(pixels, pixels[mask], spec_s[mask])
        m_spec_s = adaptive_smooth(spec_s, median_box, median_box_max)
#        m_spec_s = median_filter(median_box, spec_s)
        median_box = int(median_box*0.5)
        if (median_box<median_box_min):
            median_box=median_box_min
    spec_s = numpy.interp(pixels, pixels[mask], spec_s_org[mask])
    spec_s_out = spec_s
#    s_spec_s = adaptive_smooth(spec_s, median_box_min, int(median_box_max*0.5))
#    w1 = 1/(1+pixels)
#    w2 = pixels
#    wN = w1+w2
#    w1 = w1/wN
#    w2 = w2/wN
#    print(w1,w2)
#    spec_s_out = w1*spec_s + w2*s_spec_s
    return spec_s_out, mask


def _spec_from_lines(
    lines: numpy.ndarray,
    sigma: float,
    wavelength: numpy.ndarray,
    heights: numpy.ndarray = None,
    names: numpy.ndarray = None,
):
    rss = numpy.zeros((len(lines), wavelength.size))
    for i, line in enumerate(lines):
        rss[i] = gaussian(wavelength, mean=line, stddev=sigma)
    if heights is not None:
        rss = rss / rss.max() * heights[:, None]
    return rss.sum(axis=0)


def _shift_spectrum(spectrum: numpy.ndarray, shift: int) -> numpy.ndarray:
    """
    Shifts a spectrum by a given number of bins.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The spectrum to shift.
    shift : int
        The number of bins to shift the spectrum. Positive values shift the
        spectrum to the right, negative values shift it to the left.

    Returns
    -------
    numpy.ndarray
        The shifted spectrum.
    """
    if shift > 0:
        return numpy.pad(spectrum, (shift, 0), "constant")[:-shift]
    elif shift < 0:
        return numpy.pad(spectrum, (0, -shift), "constant")[-shift:]
    else:
        return spectrum


def _cross_match(
    ref_spec: numpy.ndarray,
    obs_spec: numpy.ndarray,
    stretch_factors: numpy.ndarray,
    shift_range: List[int],
    peak_num: int = None,
) -> Tuple[float, int, float]:
    """Find the best integer-offset cross correlation between two spectra.

    This function finds the best cross correlation between two spectra by
    stretching and shifting the first spectrum and computing the cross
    correlation with the second spectrum. The best cross correlation is
    defined as the integer offset with the highest correlation value and the correct
    number of peaks.

    Parameters
    ----------
    ref_spec : ndarray
        The reference spectrum.
    obs_spec : ndarray
        The observed spectrum.
    stretch_factors : ndarray
        The stretch factors to use.
    shift_range : tuple
        The range of shifts to use.
    peak_num : int, optional
        The number of peaks to match.

    Returns
    -------
    max_correlation : float
        The maximum correlation value.
    best_shift : int
        The best shift.
    best_stretch_factor : float
        The best stretch factor.
    """
    min_shift, max_shift = shift_range
    max_correlation = -numpy.inf
    best_shift = 0
    best_stretch_factor = 1

    for factor in stretch_factors:
        # Stretch the first signal
        stretched_signal1 = zoom(ref_spec, factor, mode="constant", prefilter=True)

        # Make the lengths equal
        len_diff = len(obs_spec) - len(stretched_signal1)
        if len_diff > 0:
            # Zero pad the stretched signal at the end if it's shorter
            stretched_signal1 = numpy.pad(stretched_signal1, (0, len_diff))
        elif len_diff < 0:
            # Or crop the stretched signal at the end if it's longer
            stretched_signal1 = stretched_signal1[:len_diff]

        # Compute the cross correlation
        cross_corr = signal.correlate(obs_spec, stretched_signal1, mode="same")

        # Normalize the cross correlation
        cross_corr = cross_corr.astype(numpy.float32)
        cross_corr /= norm(stretched_signal1) * norm(obs_spec)

        # Get the correlation shifts
        shifts = signal.correlation_lags(
            len(obs_spec), len(stretched_signal1), mode="same"
        )

        # Constrain the cross_corr and shifts to the shift_range
        mask = (shifts >= min_shift) & (shifts <= max_shift)
        cross_corr = cross_corr[mask]
        shifts = shifts[mask]

        # Find the max correlation and the corresponding shift for this stretch factor
        idx_max_corr = numpy.argmax(cross_corr)
        max_corr = cross_corr[idx_max_corr]
        shift = shifts[idx_max_corr]

        # Shift the stretched signal1
        shifted_signal1 = _shift_spectrum(stretched_signal1, shift)

        # Find the peaks in the shifted and stretched signal1
        peaks1, _ = signal.find_peaks(shifted_signal1)

        # Check if the number of peaks matches peak_num, if given
        if peak_num is not None:
            condition = max_corr > max_correlation and len(peaks1) == peak_num
        else:
            condition = max_corr > max_correlation

        if condition:
            max_correlation = max_corr
            best_shift = shift
            best_stretch_factor = factor

    return max_correlation, best_shift, best_stretch_factor


def _normalize_peaks(data, ref, min_peak_dist):
    data_ = numpy.asarray(data).copy()
    dat_peaks, dat_peak_pars = signal.find_peaks(data_, distance=min_peak_dist)

    ref_ = numpy.asarray(ref).copy()
    ref_peaks, ref_peak_pars = signal.find_peaks(ref_, distance=min_peak_dist, rel_height=0.5, width=(2,4), prominence=1.5)

    if dat_peaks.size == 0 or ref_peaks.size == 0:
        return data_, ref_, None, None, None, None

    # dat_norm = interpolate.interp1d(dat_peaks, data_[dat_peaks], kind="linear", bounds_error=False, fill_value=0.0)(numpy.arange(data_.shape[0]))
    # ref_norm = interpolate.interp1d(ref_peaks, ref_[ref_peaks], kind="linear", bounds_error=False, fill_value=0.0)(numpy.arange(data_.shape[0]))
    dat_norm = numpy.interp(numpy.arange(data_.shape[0]), dat_peaks, data_[dat_peaks])
    ref_norm = numpy.interp(numpy.arange(data_.shape[0]), ref_peaks, ref_[ref_peaks])


    ref_ = ref_ / ref_norm
    data_ = data_ / dat_norm
    # ref_ = ref_ / ref_norm * dat_norm / numpy.median(data_)
    # data_ = data_ / numpy.median(data_)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(dat_norm, "-b")
    # plt.vlines(dat_peaks, 0, 1, lw=1, color="tab:blue")
    # # plt.plot(ref_norm, "-r")
    # # plt.vlines(ref_peaks, 0, 1, lw=1, color="tab:red")

    return data_, ref_, dat_peaks, dat_peak_pars, ref_peaks, ref_peak_pars


def _choose_cc_peak(cc, shifts, min_shift, max_shift):
    mask = (shifts >= min_shift) & (max_shift >= shifts)

    ccp, _ = signal.find_peaks(cc[mask])

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15,5))
    # plt.plot(shifts[mask], cc[mask], "-o")
    # plt.vlines(shifts[mask][ccp], 0, 1)

    sum_cc = []
    for p in ccp:
        sum_cc.append(max((cc[mask][p-1], cc[mask][p+1])) + cc[mask][p])
        # plt.vlines(shifts[mask][p-2], 0, 1, ls="--", color="k")
        # plt.vlines(shifts[mask][p+3], 0, 1, ls="--", color="k")
        # mask = (shifts >= p-7) & (shifts <= p+7)
        # guess = [numpy.trapz(cc[mask], shifts[mask]), p, 1.0, 0.0]
        # bound_lower = [0.0, p+min_shift, 2, -numpy.inf]
        # bound_upper = [numpy.inf, p+max_shift, 5, numpy.inf]
        # best_gauss = fit_profile.Gaussian_const(guess)
        # best_gauss.fit(
        #     shifts[mask],
        #     cc[mask],
        #     sigma=1.0,
        #     p0=guess,
        #     bounds=(bound_lower, bound_upper)
        # )
        # area, best_shift_sp, sigma, bg = best_gauss.getPar()
        # sum_cc.append(area)


    # print(ccp, sum_cc)
    return ccp[numpy.argmax(sum_cc)]


def _align_fiber_blocks(ref_spec, obs_spec, median_box=21, clip_factor=0.7, axs=None):
    """Cross-correlate median-filtered versions of fiber profile data and model to get coarse alignment"""
    # sigma-clip spectra to half median to remove fiber features
    obs_avg = biweight_location(obs_spec, ignore_nan=True)
    ref_avg = biweight_location(ref_spec, ignore_nan=True)
    obs_spec = numpy.clip(obs_spec, 0, clip_factor*obs_avg)
    ref_spec = numpy.clip(ref_spec, 0, clip_factor*ref_avg)

    obs_median = signal.medfilt(obs_spec, median_box)
    ref_median = signal.medfilt(ref_spec, median_box)

    obs_median /= numpy.median(obs_median)
    ref_median /= numpy.median(ref_median)

    cc = signal.correlate(obs_median, ref_median, mode="same")

    shifts = signal.correlation_lags(len(obs_spec), len(ref_spec), mode="same")
    best_shift = shifts[numpy.argmax(cc)]

    if axs is not None and len(axs) >= 2:
        pixels = numpy.arange(obs_spec.size)
        axs[0].step(pixels, obs_median, where="mid", color="k", lw=2, label="obs. profile")
        axs[0].step(pixels, ref_median, where="mid", color="tab:blue", lw=1, label="ref. profile")
        axs[0].set_xlabel("Y (pix)")
        axs[0].set_ylabel("Fiber blocks")
        axs[0].legend(loc=2, frameon=False, ncols=2)
        axs[0].set_ylim(-0.1, 1.2)
        axs[1].step(shifts, cc, where="mid")
        axs[1].set_xlabel("Shifts (pix)")
        axs[1].set_ylabel("Cross-correlation")

    return best_shift


def _cross_match_float(
    ref_spec: numpy.ndarray,
    obs_spec: numpy.ndarray,
    stretch_factors: numpy.ndarray,
    shift_range: List[int],
    min_peak_dist: float = 5.0,
    gauss_window: List[int] = [-5, 5],
    gauss_sigmas: List[float] = [0.0, 5.0],
    normalize_spectra : bool = True,
    ax: None|plot.plt.Axes = None
) -> Tuple[float, float, float]:
    """Find the best fractional-pixel cross correlation between two spectra.

    This function finds the best cross correlation between two spectra by
    stretching and shifting the first spectrum and computing the cross
    correlation with the second spectrum. The best cross correlation is
    defined as the fractional-pixel offset with the highest correlation value.
    The spectra are "peak-normalized" before correlating, making all peaks
    about 1 unit in height.

    This is used for measuring fiber shifts during the night

    Parameters
    ----------
    ref_spec : ndarray
        The reference spectrum.
    obs_spec : ndarray
        The observed spectrum.
    stretch_factors : ndarray
        The stretch factors to use.
    shift_range : tuple
        The range of shifts to use.
    min_peak_dist : float, optional
        Minimum distance between two consecutive peaks to be considered signal, by default 5.0
    gauss_window : list[int], optional
        Range of pixels to consider in Gaussian fitting relative to peak, by default [-5, 5]
    gauss_sigmas : list[float], optional
        Gaussian sigma boundaries, by default [0.0, 5.0]
    normalize_spectra : bool, optional
        Normalize both spectrum to have peaks ~1, by default True
    ax : None|plt.Axes, optional
        The matplotlib axes where to draw the CC and the

    Returns
    -------
    max_correlation : float
        The maximum correlation value.
    best_shift : float
        The fractional pixel shift that maximizes the correlation
    best_stretch_factor : float
        The best stretch factor.
    """
    min_shift, max_shift = shift_range
    max_correlation = -numpy.inf
    best_shift = 0
    best_stretch_factor = 1

    # normalize the peaks to roughly magnitude 1, so that individual very bright
    # fibers do not dominate the signal
    if normalize_spectra:
        ref_spec_ = _normalize_peaks(ref_spec, min_peak_dist=min_peak_dist)
        obs_spec_ = _normalize_peaks(obs_spec, min_peak_dist=min_peak_dist)
    else:
        ref_spec_ = ref_spec.copy()
        obs_spec_ = obs_spec.copy()

    for factor in stretch_factors:
        # Stretch the first signal
        stretched_signal1 = zoom(ref_spec_, factor, mode="constant", prefilter=True)

        # Make the lengths equal
        len_diff = len(obs_spec_) - len(stretched_signal1)
        if len_diff > 0:
            # Zero pad the stretched signal at the end if it's shorter
            stretched_signal1 = numpy.pad(stretched_signal1, (0, len_diff))
        elif len_diff < 0:
            # Or crop the stretched signal at the end if it's longer
            stretched_signal1 = stretched_signal1[:len_diff]

        # Compute the cross correlation
        cross_corr = signal.correlate(obs_spec_, stretched_signal1, mode="same")

        # Normalize the cross correlation
        cross_corr = cross_corr.astype(numpy.float32)
        cross_corr /= norm(stretched_signal1) * norm(obs_spec_)
        cross_corr = numpy.nan_to_num(cross_corr)

        # Get the correlation shifts
        shifts = signal.correlation_lags(
            len(obs_spec_), len(stretched_signal1), mode="same"
        )

        # Find the max correlation and the corresponding shift for this stretch factor
        mask = (shifts >= min_shift) & (shifts <= max_shift)
        idx_max_corr = numpy.argmax(cross_corr[mask])
        max_corr = cross_corr[mask][idx_max_corr]
        shift = shifts[mask][idx_max_corr]

        if ax is not None:
            mask_cc = (shifts >= shift+2*gauss_window[0]) & (shifts <= shift+2*gauss_window[1])
            ax.step(shifts[mask_cc], cross_corr[mask_cc], color="0.7", lw=1, where="mid", alpha=0.3)

        condition = max_corr > max_correlation

        if condition:
            best_shifts, best_cross_corr = shifts, cross_corr
            max_correlation = max_corr
            best_shift = shift
            best_stretch_factor = factor

    # Fit Gaussian around maximum cross-correlation peak
    mask = (best_shifts >= best_shift+gauss_window[0]) & (best_shifts <= best_shift+gauss_window[1])
    guess = [numpy.trapz(best_cross_corr[mask], best_shifts[mask]), best_shift, 1.0, 0.0]
    bound_lower = [0.0, best_shift+min_shift, gauss_sigmas[0], -numpy.inf]
    bound_upper = [numpy.inf, best_shift+max_shift, gauss_sigmas[1], numpy.inf]
    best_gauss = fit_profile.Gaussian_const(guess)
    best_gauss.fit(
        best_shifts[mask],
        best_cross_corr[mask],
        sigma=1.0,
        p0=guess,
        bounds=(bound_lower, bound_upper)
    )
    area, best_shift_sp, sigma, bg = best_gauss.getPar()

    # display best match
    if ax is not None:
        mask = (best_shifts >= best_shift+gauss_window[0]) & (best_shifts <= best_shift+gauss_window[1])
        mask_cc = (best_shifts >= best_shift+2*gauss_window[0]) & (best_shifts <= best_shift+2*gauss_window[1])
        ax.step(best_shifts[mask_cc], best_cross_corr[mask_cc], color="0.2", lw=2, where="mid")
        ax.step(best_shifts[mask], best_gauss(best_shifts[mask]), color="tab:red", lw=2, where="mid")
        ax.axvline(best_shift, color="tab:blue", lw=1, ls="--")
        ax.axvline(best_shift_sp, color="tab:red", lw=1)
        ax.text(best_shift, (best_cross_corr[mask_cc]).min(), f"shift = {best_shift}", va="bottom", ha="left", color="tab:blue")
        ax.text(best_shift_sp, (best_cross_corr[mask_cc]).min(), f"subpix. shift = {best_shift_sp:.3f}", va="top", ha="right", color="tab:red")

    return max_correlation, best_shift_sp, best_stretch_factor


def _fiber_cc_match(
    ref_spec: numpy.ndarray,
    obs_spec: numpy.ndarray,
    guess_shift : int,
    shift_range: List[int],
    min_peak_dist: float = 5.0,
    gauss_window: List[int] = [-10, 10],
    gauss_sigmas: List[float] = [0.0, 5.0],
    normalize_spectra : bool = True,
    ax: None|plot.plt.Axes = None
) -> Tuple[float, float, float]:
    """Find the best fractional-pixel cross correlation between two fiber profiles.

    This function finds the best cross correlation between two fiber profiles
    by shifting the first spectrum and computing the cross correlation with the
    second spectrum. The best cross correlation is defined as the
    fractional-pixel offset with the highest correlation value. The spectra are
    "peak-normalized" before correlating, making all peaks about 1 unit in
    height.

    This is used for measuring fiber shifts during the night

    Parameters
    ----------
    ref_spec : ndarray
        The reference spectrum.
    obs_spec : ndarray
        The observed spectrum.
    guess_shift : int
        Guess for the best CC shift
    shift_range : tuple
        The range of shifts to use.
    min_peak_dist : float, optional
        Minimum distance between two consecutive peaks to be considered signal, by default 5.0
    gauss_window : list[int], optional
        Range of pixels to consider in Gaussian fitting relative to peak, by default [-5, 5]
    gauss_sigmas : list[float], optional
        Gaussian sigma boundaries, by default [0.0, 5.0]
    normalize_spectra : bool, optional
        Normalize both spectrum to have peaks ~1, by default True
    ax : None|plt.Axes, optional
        The matplotlib axes where to draw the CC and the

    Returns
    -------
    max_correlation : float
        The maximum correlation value.
    best_shift : float
        The fractional pixel shift that maximizes the correlation
    best_stretch_factor : float
        The best stretch factor.
    """

    min_shift, max_shift = shift_range
    max_correlation = -numpy.inf
    best_shift = 0
    best_stretch_factor = 1

    # normalize the peaks to roughly magnitude 1, so that individual very bright
    # fibers do not dominate the signal
    if normalize_spectra:
        obs_spec_, ref_spec_, obs_peak, obs_peak_pars, ref_peaks, ref_peak_pars = _normalize_peaks(obs_spec, ref_spec, min_peak_dist=min_peak_dist)
    else:
        ref_spec_ = ref_spec.copy()
        obs_spec_ = obs_spec.copy()

    # Get the correlation shifts
    shifts = signal.correlation_lags(
        len(obs_spec_), len(ref_spec_), mode="same"
    )
    cross_corr = signal.correlate(obs_spec_, ref_spec_, mode="same")

    # import matplotlib.pyplot as plt
    # plt.figure()
    # pixels = numpy.arange(obs_spec_.size)
    # plt.step(pixels, obs_spec_, where="mid")
    # plt.step(pixels, ref_spec_, where="mid")
    # plt.figure()
    # plt.step(shifts, cross_corr, where="mid")

    # Normalize the cross correlation
    cross_corr = cross_corr.astype(numpy.float32)
    cross_corr /= norm(ref_spec_) * norm(obs_spec_)
    cross_corr = numpy.nan_to_num(cross_corr)

    # Find the max correlation and the corresponding shift for this stretch factor
    mask = (shifts >= min_shift+guess_shift) & (shifts <= max_shift+guess_shift)
    idx_max_corr = numpy.argmax(cross_corr[mask])
    max_corr = cross_corr[mask][idx_max_corr]
    shift = shifts[mask][idx_max_corr]

    if ax is not None:
        ax.step(shifts[mask], cross_corr[mask], color="0.7", lw=1, where="mid", alpha=0.3)

    condition = max_corr > max_correlation

    if condition:
        best_shifts, best_cross_corr = shifts, cross_corr
        max_correlation = max_corr
        best_shift = shift

    # Fit Gaussian around maximum cross-correlation peak
    mask = (shifts >= min_shift+best_shift) & (shifts <= max_shift+best_shift)
    guess = [numpy.trapz(best_cross_corr[mask], best_shifts[mask]), best_shift, 1.0, 0.0]
    bound_lower = [0.0, best_shift+min_shift, gauss_sigmas[0], -numpy.inf]
    bound_upper = [numpy.inf, best_shift+max_shift, gauss_sigmas[1], numpy.inf]
    best_gauss = fit_profile.Gaussian_const(guess)
    best_gauss.fit(
        best_shifts[mask],
        best_cross_corr[mask],
        sigma=1.0,
        p0=guess,
        bounds=(bound_lower, bound_upper)
    )
    area, best_shift_sp, sigma, bg = best_gauss.getPar()

    # display best match
    if ax is not None:
        mask = (best_shifts >= best_shift+gauss_window[0]) & (best_shifts <= best_shift+gauss_window[1])
        ax.step(best_shifts[mask], best_cross_corr[mask], color="0.2", lw=2, where="mid")
        ax.step(best_shifts[mask], best_gauss(best_shifts[mask]), color="tab:red", lw=2, where="mid")
        ax.axvline(best_shift, color="tab:blue", lw=1, ls="--")
        ax.axvline(best_shift_sp, color="tab:red", lw=1)
        ax.text(best_shift, (best_cross_corr[mask]).min(), f"shift = {best_shift}", va="bottom", ha="left", color="tab:blue")
        ax.text(best_shift_sp, (best_cross_corr[mask]).min(), f"subpix. shift = {best_shift_sp:.3f}", va="top", ha="right", color="tab:red")

    return max_correlation, best_shift_sp, best_stretch_factor



def _apply_shift_and_stretch(
    spectrum: numpy.ndarray, shift: int, stretch_factor: float
) -> numpy.ndarray:
    """Apply a shift and stretch to a spectrum.

    This function applies a shift and stretch to a spectrum.

    Parameters
    ----------
    spectrum : ndarray
        The spectrum.
    shift : int
        The shift to apply.
    stretch_factor : float
        The stretch factor to apply.

    Returns
    -------
    shifted_stretched_spectrum : ndarray
        The shifted and stretched spectrum.
    """
    # Stretch the spectrum
    stretched_spectrum = zoom(spectrum, stretch_factor, mode="constant", prefilter=True)

    # Shift the stretched spectrum
    shifted_stretched_spectrum = _shift_spectrum(stretched_spectrum, shift)

    # If the shifted and stretched spectrum is shorter than the original spectrum, pad it with zeros
    if len(shifted_stretched_spectrum) < len(spectrum):
        shifted_stretched_spectrum = numpy.pad(
            shifted_stretched_spectrum,
            (0, len(spectrum) - len(shifted_stretched_spectrum)),
        )

    # If the shifted and stretched spectrum is longer than the original spectrum, crop it
    elif len(shifted_stretched_spectrum) > len(spectrum):
        shifted_stretched_spectrum = shifted_stretched_spectrum[: len(spectrum)]

    return shifted_stretched_spectrum


def wave_little_interpol(wavelist):
    """Make a wavelengths array for merging echelle orders with little interpolation.

    In echelle spectra we often have the situation that neighboring orders overlap
    a little in wavelength space::

        aaaaaaaaaaaa
                 bbbbbbbbbbbbb
                          ccccccccccccc

    When merging those spectra, we want to keep the original wavelength grid where possible.
    This way, we only need to interpolate on a new wavelength grid where different orders
    overlap (here ``ab`` or ``bc``) and can avoid the dangers of flux interpolation in
    those wavelength region where only one order contributes.

    This algorithm has limitations, some are fundamental, some are just due to the
    implementation and may be removed in future versions:

    - The resulting grid is **not** equally spaced, but the step size should not vary too much.
    - The wavelength arrays need to be sorted in increasing order.
    - There has to be overlap between every order and every order has to have some overlap
      free region in the middle.

    # NOTE: taken from https://bit.ly/3qpRFIp

    Parameters
    ----------
    wavelist : list of 1-dim ndarrays
        input list of wavelength

    Returns
    -------
    waveout : ndarray
        wavelength array that can be used to co-adding all echelle orders.
    """
    mins = numpy.array([min(w) for w in wavelist])
    maxs = numpy.array([max(w) for w in wavelist])

    if numpy.any(numpy.argsort(mins) != numpy.arange(len(wavelist))):
        raise ValueError("List of wavelengths must be sorted in increasing order.")
    if numpy.any(numpy.argsort(mins) != numpy.arange(len(wavelist))):
        raise ValueError("List of wavelengths must be sorted in increasing order.")
    if not numpy.all(maxs[:-1] > mins[1:]):
        raise ValueError("Not all orders overlap.")
    if numpy.any(mins[2:] < maxs[:-2]):
        raise ValueError("No order can be completely overlapped.")

    waveout = [wavelist[0][wavelist[0] < mins[1]]]
    for i in range(len(wavelist) - 1):
        #### overlap region ####
        # No assumptions on how bin edges of different orders match up
        # overlap start and stop are the last and first "clean" points.
        overlap_start = numpy.max(waveout[-1])
        overlap_end = numpy.min(wavelist[i + 1][wavelist[i + 1] > maxs[i]])
        # In overlap region patch in a linear scale with slightly different step.
        dw = overlap_end - overlap_start
        step = 0.5 * (
            bn.mean(numpy.diff(wavelist[i]))
            + bn.mean(numpy.diff(wavelist[i + 1]))
        )
        n_steps = int(dw / step + 0.5)

        wave_overlap = numpy.linspace(
            overlap_start + step, overlap_end - step, n_steps - 1
        )
        waveout.append(wave_overlap)

        #### next region without overlap ####
        if i < (len(wavelist) - 2):  # normal case
            waveout.append(
                wavelist[i + 1][
                    (wavelist[i + 1] > maxs[i]) & (wavelist[i + 1] < mins[i + 2])
                ]
            )
        else:  # last array - no more overlap behind that
            waveout.append(wavelist[i + 1][(wavelist[i + 1] > maxs[i])])

    return numpy.hstack(waveout)


def convolution_matrix(kernel, normalize=True):
    """Helper function to construct a kernel matrix for a convolution

    Parameters
    ----------
    kernel : np.ndarray[float]
        Matrix containing kernels for each pixel, row-wise
    normalize : bool, optional
        Normalizes over rows if the matrix, by default True

    Returns
    -------
    new_kernel : scipy.sparse.csr_array
        Compresed sparse kernel
    """
    if len(kernel.shape) == 2:
        nrows = kernel.shape[0]
    elif len(kernel.shape) == 1:
        nrows = kernel.size
        kernel = numpy.repeat([kernel], nrows, axis=0)

    kernelLength = kernel.shape[1]

    rowIdxFirst = numpy.floor(kernelLength / 2)
    rowIdxLast  = rowIdxFirst + nrows

    vI = []
    vJ = []
    vV = []
    for jj in range(nrows):
        kernel_row = kernel[jj] / numpy.sum(kernel[jj])
        for ii in range(kernelLength):
            if (ii + jj >= rowIdxFirst) and (ii + jj < rowIdxLast):
                # Valid otuput matrix row index
                vI.append(int(ii + jj - rowIdxFirst))
                vJ.append(int(jj))
                vV.append(kernel_row[ii])

    # vI, vJ = numpy.where(kernel>1e-5)
    # vV = kernel[vI, vJ]
    new_kernel = sparse.csr_array((vV, (vJ, vI)))
    if normalize:
        new_kernel = new_kernel.multiply(1 / numpy.sum(new_kernel, axis=1))
    return new_kernel


class Spectrum1D(Header):

    @classmethod
    def select_poly_class(cls, poly_kind=None):
        """Returns the polynomial class to use for the given kind of polynomial

        Parameters
        ----------
        poly_kind : string, optional with default None

        Returns
        -------
        poly_cls : numpy.polynomial.Polynomial
        """
        if poly_kind == "poly" or poly_kind is None or poly_kind == "None":
            poly_cls = numpy.polynomial.Polynomial
        elif poly_kind == "chebyshev":
            poly_cls = numpy.polynomial.Chebyshev
        elif poly_kind == "legendre":
            poly_cls = numpy.polynomial.Legendre
        else:
            raise ValueError(f"Invalid polynomial kind: '{poly_kind}', valid options are: 'poly', 'legendre', 'chebyshev'")
        return poly_cls

    def __init__(
        self, wave=None, data=None, error=None, mask=None,
        lsf=None, wave_trace=None, lsf_trace=None,
        sky=None, sky_error=None, header=None
    ):
        self._data = data
        if data is not None:
            self._dim = self._data.shape[0]
            self._pixels = numpy.arange(self._dim)
        self._error = error
        self._mask = mask
        self._sky = sky
        self._sky_error = sky_error
        self._header = header

        self.set_wave_and_lsf_traces(wave=wave, wave_trace=wave_trace, lsf_trace=lsf_trace, lsf=lsf)

    def __sub__(self, other):
        if isinstance(other, Spectrum1D):
            # verify wavelength and LSF arrays are the same
            if not numpy.array_equal(self._wave, other._wave):
                raise ValueError("wavelength arrays are not the same")
            if not numpy.array_equal(self._wave_trace, other._wave_trace):
                raise ValueError("wavelength trace arrays are not the same")

            data = numpy.zeros_like(self._data)
            select_zero = self._data == 0
            data = self._data - other._data

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero] = 0
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask = self._mask
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero] = 0
            else:
                mask = None

            if self._error is not None and other._error is not None:
                error = numpy.sqrt(self._error**2 + other._error**2)
            elif self._error is not None:
                error = self._error
            elif other._error is not None:
                error = other._error
            else:
                error = None

            if self._sky is not None and other._sky is not None:
                sky = self._sky - other._sky
            elif self._sky is not None:
                sky = self._sky
            elif other._sky is not None:
                sky = other._sky
            else:
                sky = None

            if self._sky_error is not None and other._sky_error is not None:
                sky_error = numpy.sqrt(self._sky_error**2 + other._sky_error**2)
            elif self._sky_error is not None:
                sky_error = self._sky_error
            elif other._sky_error is not None:
                sky_error = other._sky_error
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec

        elif isinstance(other, numpy.ndarray):
            data = self._data - other
            error = self._error
            mask = self._mask
            sky = self._sky
            sky_error = self._sky_error

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data - other
                error = self._error
                mask = self._mask
                sky = self._sky
                sky_error = self._sky_error

                if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                    data = data.astype(numpy.float32)
                if error is not None:
                    if error.dtype == numpy.float64 or error.dtype == numpy.dtype(
                        ">f8"
                    ):
                        error = error.astype(numpy.float32)
                if sky is not None:
                    if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                        sky = sky.astype(numpy.float32)
                if sky_error is not None:
                    if (
                        sky_error.dtype == numpy.float64
                        or sky_error.dtype == numpy.dtype(">f8")
                    ):
                        sky_error = sky_error.astype(numpy.float32)

                spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

                return spec
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for -: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __add__(self, other):
        if isinstance(other, Spectrum1D):
            # verify wavelength and LSF arrays are the same
            if not numpy.array_equal(self._wave, other._wave):
                raise ValueError("wavelength arrays are not the same")
            if not numpy.array_equal(self._wave_trace, other._wave_trace):
                raise ValueError("wavelength trace arrays are not the same")

            other._data.astype(numpy.float32)
            data = numpy.zeros_like(self._data)
            select_zero = self._data == 0
            data = self._data + other._data

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero] = 0
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask = self._mask
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero] = 0
            else:
                mask = None

            if self._error is not None and other._error is not None:
                error = numpy.sqrt(self._error**2 + other._error**2)
            elif self._error is not None:
                error = self._error
            elif other._error is not None:
                error = other._error
            else:
                error = None

            if self._sky is not None and other._sky is not None:
                sky = self._sky + other._sky
            elif self._sky is not None:
                sky = self._sky
            elif other._sky is not None:
                sky = other._sky
            else:
                sky = None

            if self._sky_error is not None and other._sky_error is not None:
                sky_error = numpy.sqrt(self._sky_error**2 + other._sky_error**2)
            elif self._sky_error is not None:
                sky_error = self._sky_error
            elif other._sky_error is not None:
                sky_error = other._sky_error
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec

        elif isinstance(other, numpy.ndarray):
            data = self._data + other

            if self._error is not None:
                error = self._error + other
            else:
                error = None

            if self._mask is not None:
                mask = self._mask
            else:
                mask = None

            if self._sky is not None:
                sky = self._sky + other
            else:
                sky = None

            if self._sky_error is not None:
                sky_error = self._sky_error + other
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data + other

                if self._error is not None:
                    error = self._error + other
                else:
                    error = None

                if self._mask is not None:
                    mask = self._mask
                else:
                    mask = None

                if self._sky is not None:
                    sky = self._sky + other
                else:
                    sky = None

                if self._sky_error is not None:
                    sky_error = self._sky_error + other
                else:
                    sky_error = None

                if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                    data = data.astype(numpy.float32)
                if error is not None:
                    if error.dtype == numpy.float64 or error.dtype == numpy.dtype(
                        ">f8"
                    ):
                        error = error.astype(numpy.float32)
                if sky is not None:
                    if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                        sky = sky.astype(numpy.float32)
                if sky_error is not None:
                    if (
                        sky_error.dtype == numpy.float64
                        or sky_error.dtype == numpy.dtype(">f8")
                    ):
                        sky_error = sky_error.astype(numpy.float32)

                spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

                return spec
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for -: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __truediv__(self, other):
        if isinstance(other, Spectrum1D):
            # verify wavelength and LSF arrays are the same
            if not numpy.isclose(self._wave, other._wave).all():
                raise ValueError("wavelength arrays are not the same")
            if (self._wave_trace is not None and other._wave_trace is not None) and not self._wave_trace == other._wave_trace:
                raise ValueError("wavelength trace arrays are not the same")

            other._data = other._data.astype(numpy.float32)
            select = other._data != 0.0
            data = numpy.divide(
                self._data, other._data, out=numpy.zeros_like(self._data), where=select
            )

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
                mask[~select] = True
            elif other._mask is not None:
                mask = other._mask
                mask[~select] = True
            elif self._mask is not None:
                mask = self._mask
                mask[~select] = True
            else:
                mask = None

            if self._error is not None and other._error is not None:
                error = numpy.zeros_like(self._error)
                error_a = (
                    numpy.divide(self._error, other._data, out=error, where=select) ** 2
                )
                error_b = (
                    numpy.divide(
                        self._data * other._error,
                        other._data**2,
                        out=error,
                        where=select,
                    )
                    ** 2
                )
                error = numpy.sqrt(error_a + error_b)
            elif self._error is not None:
                error = numpy.divide(
                    self._error,
                    other._data,
                    out=numpy.zeros_like(self._error),
                    where=select,
                )
            elif other._error is not None:
                error = numpy.divide(
                    self._data * other._error,
                    other._data**2,
                    out=numpy.zeros_like(self._error),
                    where=select,
                )
            else:
                error = None

            if self._sky is not None and other._sky is not None:
                sky = numpy.divide(
                    self._sky,
                    other._sky,
                    out=numpy.zeros_like(self._sky),
                    where=other._sky != 0.0,
                )
            elif self._sky is not None:
                sky = self._sky
            elif other._sky is not None:
                sky = other._sky
            else:
                sky = None

            if self._sky_error is not None and other._sky_error is not None:
                sky_error = numpy.zeros_like(self._sky_error)
                sky_error_a = (
                    numpy.divide(
                        self._sky_error, other._data, out=sky_error, where=select
                    )
                    ** 2
                )
                sky_error_b = (
                    numpy.divide(
                        self._data * other._sky_error,
                        other._data**2,
                        out=sky_error,
                        where=select,
                    )
                    ** 2
                )
                sky_error = numpy.sqrt(sky_error_a + sky_error_b)
            elif self._sky_error is not None:
                sky_error = numpy.divide(
                    self._sky_error,
                    other._data,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            elif other._sky_error is not None:
                sky_error = numpy.divide(
                    self._data * other._sky_error,
                    other._data**2,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, wave_trace=self._wave_trace, lsf_trace=self._lsf_trace, sky=sky, sky_error=sky_error)

            return spec

        elif isinstance(other, numpy.ndarray):
            select = other != 0.0
            data = numpy.divide(self._data, other, out=numpy.zeros_like(self._data), where=select)

            if self._error is not None:
                error = numpy.divide(
                    self._error, other, out=numpy.zeros_like(self._error), where=select
                )
            else:
                error = None

            if self._mask is not None:
                mask = self._mask
                mask[~select] = True
            else:
                mask = None

            if self._sky is not None:
                sky = numpy.divide(
                    self._sky, other, out=numpy.zeros_like(self._sky), where=select
                )
            else:
                sky = None

            if self._sky_error is not None:
                sky_error = numpy.divide(
                    self._sky_error,
                    other,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                select = other != 0.0
                data = numpy.divide(
                    self._data, other, out=numpy.zeros_like(self._data), where=select
                )

                if self._error is not None:
                    error = numpy.divide(
                        self._error,
                        other,
                        out=numpy.zeros_like(self._error),
                        where=select,
                    )
                else:
                    error = None

                if self._mask is not None:
                    mask = self._mask
                    mask[~select] = True

                if self._sky is not None:
                    sky = numpy.divide(
                        self._sky, other, out=numpy.zeros_like(self._sky), where=select
                    )
                else:
                    sky = None

                if self._sky_error is not None:
                    sky_error = numpy.divide(
                        self._sky_error,
                        other,
                        out=numpy.zeros_like(self._sky_error),
                        where=select,
                    )
                else:
                    sky_error = None

                if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                    data = data.astype(numpy.float32)
                if error is not None:
                    if error.dtype == numpy.float64 or error.dtype == numpy.dtype(
                        ">f8"
                    ):
                        error = error.astype(numpy.float32)
                if sky is not None:
                    if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                        sky = sky.astype(numpy.float32)
                if sky_error is not None:
                    if (
                        sky_error.dtype == numpy.float64
                        or sky_error.dtype == numpy.dtype(">f8")
                    ):
                        sky_error = sky_error.astype(numpy.float32)

                spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

                return spec
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for /: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __rtruediv__(self, other):
        if isinstance(other, Spectrum1D):
            # verify wavelength and LSF arrays are the same
            if not numpy.array_equal(self._wave, other._wave):
                raise ValueError("wavelength arrays are not the same")
            if not numpy.array_equal(self._wave_trace, other._wave_trace):
                raise ValueError("wavelength trace arrays are not the same")

            other._data = other._data.astype(numpy.float32)
            select = self._data != 0.0
            data = numpy.divide(
                other._data, self._data, out=numpy.zeros_like(self._data), where=select
            )

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
                mask[~select] = True
            elif other._mask is not None:
                mask = other._mask
                mask[~select] = True
            elif self._mask is not None:
                mask = self._mask
                mask[~select] = True
            else:
                mask = None

            if self._error is not None and other._error is not None:
                error = numpy.zeros_like(self._error)
                error_a = (
                    numpy.divide(other._error, self._data, out=error, where=select) ** 2
                )
                error_b = (
                    numpy.divide(
                        other._data * self._error,
                        self._data**2,
                        out=error,
                        where=select,
                    )
                    ** 2
                )
                error = numpy.sqrt(error_a + error_b)
            elif self._error is not None:
                error = numpy.divide(
                    other._error,
                    self._data,
                    out=numpy.zeros_like(self._error),
                    where=select,
                )
            elif other._error is not None:
                error = numpy.divide(
                    other._data * self._error,
                    self._data**2,
                    out=numpy.zeros_like(self._error),
                    where=select,
                )
            else:
                error = None

            if other._sky is not None:
                sky = numpy.divide(
                    other._sky,
                    self._sky,
                    out=numpy.zeros_like(self._sky),
                    where=self._sky != 0.0,
                )
            elif self._sky is not None:
                sky = self._sky
            elif other._sky is not None:
                sky = other._sky
            else:
                sky = None

            if self._sky_error is not None and other._sky_error is not None:
                sky_error = numpy.zeros_like(self._sky_error)
                sky_error_a = (
                    numpy.divide(
                        other._sky_error, self._data, out=sky_error, where=select
                    )
                    ** 2
                )
                sky_error_b = (
                    numpy.divide(
                        other._data * self._sky_error,
                        self._data**2,
                        out=sky_error,
                        where=select,
                    )
                    ** 2
                )
                sky_error = numpy.sqrt(sky_error_a + sky_error_b)
            elif self._sky_error is not None:
                sky_error = numpy.divide(
                    other._sky_error,
                    self._data,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            elif other._sky_error is not None:
                sky_error = numpy.divide(
                    other._data * self._sky_error,
                    self._data**2,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec

        elif isinstance(other, numpy.ndarray):
            select = self._data != 0.0
            data = numpy.divide(other, self._data, out=numpy.zeros_like(self._data), where=select)

            if self._error is not None:
                error = numpy.divide(
                    other * self._error,
                    self._data**2,
                    out=numpy.zeros_like(self._error),
                    where=select,
                )
            else:
                error = None

            if self._mask is not None:
                mask = self._mask
                mask[~select] = True
            else:
                mask = None

            if self._sky is not None:
                sky = numpy.divide(
                    other,
                    self._sky,
                    out=numpy.zeros_like(self._sky),
                    where=self._sky != 0.0,
                )
            else:
                sky = None

            if self._sky_error is not None:
                sky_error = numpy.divide(
                    other * self._sky_error,
                    self._data**2,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(
                wave=self._wave,
                data=data,
                error=error,
                mask=mask,
                sky=sky,
                sky_error=sky_error,
            )

            return spec
        else:
            select = self._data != 0.0
            data = numpy.divide(
                other, self._data, out=numpy.zeros_like(self._data), where=select
            )

            if self._error is not None:
                error = numpy.divide(
                    other * self._error,
                    self._data**2,
                    out=numpy.zeros_like(self._error),
                    where=select,
                )
            else:
                error = None

            if self._mask is not None:
                mask = self._mask
                mask[~select] = True
            else:
                mask = None

            if self._sky is not None:
                sky = numpy.divide(
                    other,
                    self._sky,
                    out=numpy.zeros_like(self._sky),
                    where=self._sky != 0.0,
                )
            else:
                sky = None

            if self._sky_error is not None:
                sky_error = numpy.divide(
                    other * self._sky_error,
                    self._data**2,
                    out=numpy.zeros_like(self._sky_error),
                    where=select,
                )
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec

    def __mul__(self, other):
        if isinstance(other, Spectrum1D):
            # verify wavelength and LSF arrays are the same
            if not numpy.array_equal(self._wave, other._wave):
                raise ValueError("wavelength arrays are not the same")
            if not numpy.array_equal(self._wave_trace, other._wave_trace):
                raise ValueError("wavelength trace arrays are not the same")

            data = self._data * other._data

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is not None:
                mask = self._mask
            elif other._mask is not None:
                mask = other._mask
            else:
                mask = None

            if self._error is not None and other._error is not None:
                error_a = self._error * other._data
                error_b = self._data * other._error
                error = numpy.sqrt(error_a**2 + error_b**2)
            elif self._error is not None:
                error = self._error
            elif other._error is not None:
                error = other._error
            else:
                error = None

            if self._sky is not None:
                sky = self._sky * other._data
            elif self._sky is not None:
                sky = self._sky
            elif other._sky is not None:
                sky = other._sky
            else:
                sky = None

            if self._sky_error is not None and other._sky_error is not None:
                sky_error_a = self._sky_error * other._data
                sky_error_b = self._data * other._sky_error
                sky_error = numpy.sqrt(sky_error_a**2 + sky_error_b**2)
            elif self._sky_error is not None:
                sky_error = self._sky_error
            elif other._sky_error is not None:
                sky_error = other._sky_error
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec

        elif isinstance(other, numpy.ndarray):
            data = self._data * other

            if self._mask is not None:
                mask = self._mask
            else:
                mask = None

            if self._error is not None:
                error = self._error * other
            else:
                error = None

            if self._sky is not None:
                sky = self._sky * other
            else:
                sky = None

            if self._sky_error is not None:
                sky_error = self._sky_error * other
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            #  try:
            data = self._data * other

            if self._mask is not None:
                mask = self._mask
            else:
                mask = None

            if self._error is not None:
                error = self._error * other
            else:
                error = None

            if self._sky is not None:
                sky = self._sky * other
            else:
                sky = None

            if self._sky_error is not None:
                sky_error = self._sky_error * other
            else:
                sky_error = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            if sky is not None:
                if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                    sky = sky.astype(numpy.float32)
            if sky_error is not None:
                if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                    ">f8"
                ):
                    sky_error = sky_error.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

            return spec

    def __pow__(self, other):
        data = self._data ** other

        if self._error is not None:
            error = 1.0 / float(other) * self._data ** (other - 1) * self._error
        else:
            error = None

        if self._mask is not None:
            mask = self._mask
        else:
            mask = None

        if self._sky is not None:
            sky = self._sky**other
        else:
            sky = None

        if self._sky_error is not None:
            sky_error = 1.0 / float(other) * self._data ** (other - 1) * self._sky_error
        else:
            sky_error = None

        if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
            data = data.astype(numpy.float32)
        if error is not None:
            if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                error = error.astype(numpy.float32)
        if sky is not None:
            if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                sky = sky.astype(numpy.float32)
        if sky_error is not None:
            if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                ">f8"
            ):
                sky_error = sky_error.astype(numpy.float32)

        spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

        return spec

    def __rpow__(self, other):
        data = other**self._data

        if self._error is not None:
            error = numpy.log(other) * data * self._error
        else:
            error = None

        if self._mask is not None:
            mask = self._mask
        else:
            mask = None

        if self._sky is not None:
            sky = other**self._sky
        else:
            sky = None

        if self._sky_error is not None:
            sky_error = numpy.log(other) * data * self._sky_error
        else:
            sky_error = None

        if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
            data = data.astype(numpy.float32)
        if error is not None:
            if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                error = error.astype(numpy.float32)
        if sky is not None:
            if sky.dtype == numpy.float64 or sky.dtype == numpy.dtype(">f8"):
                sky = sky.astype(numpy.float32)
        if sky_error is not None:
            if sky_error.dtype == numpy.float64 or sky_error.dtype == numpy.dtype(
                ">f8"
            ):
                sky_error = sky_error.astype(numpy.float32)

        spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask, sky=sky, sky_error=sky_error)

        return spec

    def __lt__(self, other):
        return self._data < other

    def __le__(self, other):
        return self._data <= other

    def __eq__(self, other):
        return self._data == other

    def __ne__(self, other):
        return self._data != other

    def __gt__(self, other):
        return self._data > other

    def __ge__(self, other):
        return self._data >= other

    def add_header_comment(self, comstr):
        '''
        Append a COMMENT card at the end of the FITS header.
        '''
        if self._header is None:
            return
        self._header.append(('COMMENT', comstr), bottom=True)

    def eval_wave_and_lsf_traces(self, wave, wave_trace, lsf_trace):
        """Evaluates the wavelength and LSF traces at the given wavelength array.

        Given a wavelength array, this method evaluates the wavelength and LSF
        traces at the given wavelength array. The wavelength trace is evaluated
        using the polynomial coefficients and the LSF trace is evaluated using
        the LSF coefficients. The wavelength and LSF traces are evaluated using
        the same polynomial class as the one used to fit the traces.

        If the wavelength array is not the same as the one fitted by the
        polynomial class, the wavelength and the LSF traces are interpolated to
        the new wavelength array.

        Parameters
        ----------
        wave : numpy.ndarray (float)
            New wavelength scale
        wave_trace : astropy.table.row.Row
            Wavelength trace parameters
        lsf_trace : astropy.table.row.Row
            LSF trace parameters

        Returns
        -------
        wave : numpy.ndarray (float)
            New wavelength scale
        lsf : numpy.ndarray (float)
            New LSF array

        Raises
        ------
        ValueError
            If the new wavelength array is outside the old wavelength array
        ValueError
            If the new LSF array is outside the old LSF array
        ValueError
            If the new wavelength array does not match the input wavelength array
        """
        # eval wavelength and LSF polynomial traces
        if wave_trace is not None:
            wave_coeffs = wave_trace["COEFF"]
            old_wave_pixels = numpy.arange(wave_trace["XMIN"], wave_trace["XMAX"] + 1)
            wave_poly_cls = self.select_poly_class(poly_kind=wave_trace["FUNC"])
            wave_poly = wave_poly_cls(wave_coeffs)
            old_wave = wave_poly(old_wave_pixels)
        else:
            old_wave = wave

        if lsf_trace is not None:
            lsf_coeffs = lsf_trace["COEFF"]
            old_lsf_pixels = numpy.arange(lsf_trace["XMIN"], lsf_trace["XMAX"] + 1)
            lsf_poly_cls = self.select_poly_class(poly_kind=lsf_trace["FUNC"])
            lsf_poly = lsf_poly_cls(lsf_coeffs)
            old_lsf = lsf_poly(old_lsf_pixels)
        else:
            old_lsf = None

        # check if interpolation is needed
        if old_wave.size == wave.size and numpy.allclose(old_wave, wave, rtol=1e-2):
            return old_wave, old_lsf
        else:
            new_wave_pixels = numpy.interp(wave, old_wave, old_wave_pixels)
            # verify that new pixels are within the old pixel range
            if numpy.any(new_wave_pixels < old_wave_pixels[0]) or numpy.any(new_wave_pixels > old_wave_pixels[-1]):
                raise ValueError("New wavelength pixels are outside the old wavelength pixel range")
            new_wave = wave_poly(new_wave_pixels)
            # verify that the new wavelength is equivalent to the input wavelength
            if not numpy.allclose(new_wave, wave, rtol=1e-2):
                raise ValueError("New wavelength pixels do not match the input wavelength")

            # if no LSF trace is provided, return the new wavelength array
            if old_lsf is None:
                new_lsf = None
                return new_wave, new_lsf

            new_lsf_pixels = numpy.interp(wave, old_wave, old_lsf_pixels)
            # verify that new pixels are within the old pixel range
            if numpy.any(new_lsf_pixels < old_lsf_pixels[0]) or numpy.any(new_lsf_pixels > old_lsf_pixels[-1]):
                raise ValueError("New LSF pixels are outside the old LSF pixel range")
            new_lsf = lsf_poly(new_lsf_pixels)

            return new_wave, new_lsf

    def set_wave_and_lsf_traces(self, wave, wave_trace, lsf_trace, lsf=None):
        """Sets the wavelength and LSF traces.

        Parameters
        ----------
        wave : numpy.ndarray (float)
            Wavelength array
        wave_trace : astropy.table.row.Row
            Wavelength trace parameters
        lsf_trace : astropy.table.row.Row
            LSF trace parameters
        lsf : numpy.ndarray (float), optional
            LSF array
        """

        self._wave_trace = wave_trace
        self._lsf_trace = lsf_trace
        self._wave, self._lsf = self.eval_wave_and_lsf_traces(
            wave=wave, wave_trace=self._wave_trace, lsf_trace=self._lsf_trace
        )
        # set LSF only if no trace information is provided
        if self._lsf is None:
            self._lsf = lsf

    def loadFitsData(
        self,
        file,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_wave=None,
        extension_fwhm=None,
        extension_sky=None,
        extension_skyerror=None,
        extension_hdr=None,
        logwave=False,
    ):
        """
        load information from a FITS file into a Spectrum1D object.
        A single or multiple extension FITS file are possible inputs.

        Parameters
        --------------
        filename : string
            Name or Path of the FITS image from which the data shall be loaded

        extension_data : int, optional with default: None
            Number of the FITS extension containing the data

        extension_mask : int, optional with default: None
            Number of the FITS extension containing the masked pixels

        extension_error : int, optional with default: None
            Number of the FITS extension containing the errors for the values
        """

        hdu = pyfits.open(file, uint=True, do_not_scale_image_data=True, memmap=False)
        if (
            extension_data is None
            and extension_mask is None
            and extension_error is None
            and extension_wave is None
            and extension_fwhm is None
            and extension_sky is None
            and extension_skyerror is None
            and extension_hdr is None
        ):
            self._data = hdu[0].data
            self._header = hdu[0].header
            self._dim = self._data.shape[0]  # set shape
            self._pixels = numpy.arange(self._dim)
            # self.setHeader(header = hdu[0].header, origin=file)
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header["EXTNAME"].split()[0] == "ERROR":
                        self._error = hdu[i].data
                    elif hdu[i].header["EXTNAME"].split()[0] == "BADPIX":
                        self._mask = hdu[i].data.astype("bool")
                    elif hdu[i].header["EXTNAME"].split()[0] == "WAVE":
                        self._wave = hdu[i].data
                    elif hdu[i].header["EXTNAME"].split()[0] == "INSTFWHM":
                        self._lsf = hdu[i].data
                    elif hdu[i].header["EXTNAME"].split()[0] == "SKY":
                        self._sky = hdu[i].data
                    elif hdu[i].header["EXTNAME"].split()[0] == "SKY_ERROR":
                        self._sky_error = hdu[i].data
            if self._wave is None:
                self._wave = (
                    self._pixels * self._header["CDELT1"] + self._header["CRVAL1"]
                )
        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data
            if extension_error is not None:
                self._error = hdu[extension_error].data
            if extension_wave is not None:
                self._wave = hdu[extension_wave].data
            if extension_fwhm is not None:
                self._lsf = hdu[i].data
            if extension_sky is not None:
                self._sky = hdu[i].data
            if extension_skyerror:
                self._sky_error = hdu[i].data

        hdu.close()

        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header, origin=file)

    def writeFitsData(
        self,
        filename,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_wave=None,
        extension_fwhm=None,
        extension_sky=None,
        extension_skyerror=None,
        extension_hdr=None,
    ):
        """
        Save information from a Spectrum1D object into a FITS file.
        A single or multiple extension file are possible to create.

        Parameters
        --------------
        filename : string
            Name or Path of the FITS image from which the data shall be loaded

        extension_data : int (0, 1, or 2), optional with default: None
            Number of the FITS extension containing the data

        extension_mask : int (0, 1, or 2), optional with default: None
            Number of the FITS extension containing the masked pixels

        extension_error : int (0, 1, or 2), optional with default: None
            Number of the FITS extension containing the errors for the values
        """
        # convert all to single precision
        self._data = self._data.astype("float32")
        if self._error is not None:
            self._error = self._error.astype("float32")
        if self._wave is not None:
            self._wave = self._wave.astype("float32")
        if self._lsf is not None:
            self._lsf = self._lsf.astype("float32")
        if self._sky is not None:
            self._sky = self._sky.astype("float32")
        if self._sky_error is not None:
            self._sky_error = self._sky_error.astype("float32")

        hdus = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if (
            extension_data is None
            and extension_error is None
            and extension_mask is None
            and extension_wave is None
            and extension_fwhm is None
            and extension_sky is None
            and extension_skyerror is None
            and extension_hdr is None
        ):
            hdus[0] = pyfits.PrimaryHDU(self._data, header=self._header)
            if self._wave is not None:
                hdus[1] = pyfits.ImageHDU(self._wave, name="WAVE")
            if self._lsf is not None:
                hdus[2] = pyfits.ImageHDU(self._lsf, name="INSTFWHM")
            if self._error is not None:
                hdus[3] = pyfits.ImageHDU(self._error, name="ERROR")
            if self._mask is not None:
                hdus[4] = pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX")
            if self._sky is not None:
                hdus[5] = pyfits.ImageHDU(self._sky, name="SKY")
            if self._sky_error is not None:
                hdus[6] = pyfits.ImageHDU(self._sky_error, name="SKY_ERROR")
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data > 0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name="DATA")

            # wavelength hdu
            if extension_wave == 0:
                hdu = pyfits.PrimaryHDU(self._wave)
            elif extension_wave > 0 and extension_wave is not None:
                hdus[extension_wave] = pyfits.ImageHDU(self._wave, name="WAVE")

            # instrumental FWHM hdu
            if extension_fwhm == 0:
                hdu = pyfits.PrimaryHDU(self._lsf)
            elif extension_fwhm > 0 and extension_fwhm is not None:
                hdus[extension_fwhm] = pyfits.ImageHDU(self._lsf, name="INSTFWHM")

            # mask hdu
            if extension_mask == 0:
                hdu = pyfits.PrimaryHDU(self._mask.astype("uint8"))
            elif extension_mask > 0 and extension_mask is not None:
                hdus[extension_mask] = pyfits.ImageHDU(
                    self._mask.astype("uint8"), name="BADPIX"
                )

            # error hdu
            if extension_error == 0:
                hdu = pyfits.PrimaryHDU(self._error)
            elif extension_error > 0 and extension_error is not None:
                hdus[extension_error] = pyfits.ImageHDU(self._error, name="ERROR")

            # sky extension
            if extension_sky == 0:
                hdu = pyfits.PrimaryHDU(self._sky)
            elif extension_sky > 0 and extension_sky is not None:
                hdus[extension_sky] = pyfits.ImageHDU(self._sky, name="SKY")

            # sky error extension
            if extension_skyerror == 0:
                hdu = pyfits.PrimaryHDU(self._sky_error)
            elif extension_skyerror > 0 and extension_skyerror is not None:
                hdus[extension_skyerror] = pyfits.ImageHDU(
                    self._sky_error, name="SKY_ERROR"
                )

            # header hdu
            if extension_hdr == 0:
                hdu = pyfits.PrimaryHDU(header=self._header)
            elif extension_hdr > 0 and extension_hdr is not None:
                hdus[extension_hdr].header = self._header

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except Exception:
                break

        if len(hdus) > 0:
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
        hdu.writeto(filename, overwrite=True)  # write FITS file to disc

    def writeTxtData(self, filename):
        out = open(filename, "w")
        for i in range(self._dim):
            if self._error is None and self._mask is None:
                out.write("%i %f %f\n" % (i + 1, self._wave[i], self._data[i]))
            elif self._error is not None and self._mask is None:
                out.write(
                    "%i %f %f %f\n"
                    % (i + 1, self._wave[i], self._data[i], self._error[i])
                )
            elif self._error is not None and self._mask is not None:
                out.write(
                    "%i %f %f %f %d\n"
                    % (
                        i + 1,
                        self._wave[i],
                        self._data[i],
                        self._error[i],
                        int(self._mask[i]),
                    )
                )

        out.close()

    def loadTxtData(self, filename):
        infile = open(filename, "r")
        lines = infile.readlines()
        wave = numpy.zeros(len(lines), dtype=numpy.float32)
        data = numpy.zeros(len(lines), dtype=numpy.float32)
        for i in range(len(lines)):
            line = lines[i].split()
            wave[i] = float(line[1])
            data[i] = float(line[2])
        self._wave = wave
        self._data = data
        self._dim = len(data)
        infile.close()

    def loadSTDref(
        self, ref_file, column_wave=0, column_flux=1, delimiter="", header=1
    ):
        dat = open(ref_file, "r")
        lines = dat.readlines()
        wave = numpy.zeros(len(lines) - header, dtype=numpy.float32)
        data = numpy.zeros(len(lines) - header, dtype=numpy.float32)
        for i in range(header, len(lines)):
            if delimiter == "":
                line = lines[i].split()
            else:
                line = lines.split(delimiter)
            wave[i - header] = line[column_wave]
            data[i - header] = line[column_flux]
        self._data = data
        self._wave = wave
        self._dim = self._data.shape[0]
        self._pixels = numpy.arange(self._dim)

    def max(self):
        """
        Determines the maximum data value of the spectrum

        Returns: (max,max_wave,max_pos)
        -----------
        max : float
            Maximum data value

        max_wave : float
            Wavelength of maximum data value

        max_pos : int
            Pixel position of the maximum data value

        """
        max = bn.nanmax(self._data)  # get max
        select = self._data == max  # select max value
        max_wave = self._wave[select][0]  # get corresponding wavelength
        max_pos = self._pixels[select][0]  # get corresponding position
        return max, max_wave, max_pos

    def min(self):
        """
        Determines the minimum data value of the spectrum

        Returns: (min,min_wave,min_pos)
        -----------
        min : float
            Minimum data value

        min_wave : float
            Wavelength of minimum data value

        min_pos : int
            Pixel position of the minimum data value

        """
        min = bn.nanmin(self._data)  # get min
        select = self._data == min  # select min value
        min_wave = self._wave[select][0]  # get corresponding waveength
        min_pos = self._pixels[select][0]  # get corresponding position
        return min, min_wave, min_pos

    def getData(self):
        """
        Return the content of the spectrum

        Returns
        -------
        pix : numpy.ndarray (int)
            Array of the pixel positions
        wave : numpy.ndarray (float)
            Array of the wavelength
        data : numpy.ndarray (float)
            Array of the data value
        error : numpy.ndarray (float)
            Array of the corresponding errors
        mask : numpy.ndarray (bool)
            Array of the bad pixel mask
        sky : numpy.ndarray (float)
            Array of the sky spectrum
        """

        return self._pixels, self._wave, self._data, self._error, self._mask, self._sky

    def resampleSpec(
        self,
        ref_wave,
        method="spline",
        err_sim=500,
        replace_error=1e10,
        extrapolate=None,
    ):
        # flip spectrum along dispersion axis if needed
        if self._wave[-1] < self._wave[0]:
            self._wave = numpy.flipud(self._wave)
            self._data = numpy.flipud(self._data)
            if self._error is not None:
                self._error = numpy.flipud(self._error)
            if self._mask is not None:
                self._mask = numpy.flipud(self._mask)
            if self._sky is not None:
                self._sky = numpy.flipud(self._sky)
            if self._sky_error is not None:
                self._sky_error = numpy.flipud(self._sky_error)

        # case where input spectrum has more than half the pixels masked
        if bn.nansum(self._data) == 0.0 or (
            self._mask is not None and numpy.sum(self._mask) > self._dim / 2
        ):
            # all pixels masked
            new_mask = numpy.ones(len(ref_wave), dtype=bool)
            # all data points to zero
            new_data = numpy.zeros(len(ref_wave), numpy.float32)
            # all LSF pixels zero (if present)
            if self._lsf is not None:
                new_lsf = numpy.zeros(len(ref_wave), numpy.float32)
            else:
                new_lsf = None
            # all error pixels replaced with replace_error
            if self._error is None or err_sim == 0:
                new_error = None
            else:
                new_error = numpy.ones(len(ref_wave), numpy.float32) * replace_error
            # all sky pixels zero (if present)
            if self._sky is None:
                new_sky = None
            else:
                new_sky = numpy.zeros(len(ref_wave), numpy.float32)
            # all sky error pixels zero (if present)
            if self._sky_error is None:
                new_sky_error = None
            else:
                new_sky_error = numpy.zeros(len(ref_wave), numpy.float32)

            # return masked spectrum
            return Spectrum1D(
                data=new_data,
                wave=ref_wave,
                error=new_error,
                mask=new_mask,
                lsf=new_lsf,
                sky=new_sky,
                sky_error=new_sky_error,
                header=self._header,
            )

        else:
            # good pixels selection
            if self._mask is not None:
                select_badpix = self._mask
                select_goodpix = numpy.logical_not(self._mask)
            else:
                select_badpix = numpy.zeros(self._dim, dtype=bool)
                select_goodpix = numpy.ones(self._dim, dtype=bool)

            # interpolate LSF ---------------------------------------------------------------------------------------------------------------------------------
            if self._lsf_trace is None and self._lsf is not None:
                intp = interpolate.interp1d(
                    self._wave[select_goodpix],
                    self._lsf[select_goodpix],
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=(0.0, 0.0),
                )
                clean_lsf = intp(self._wave)

                # select pixels that were interpolated (excluding extrapolated ones)
                select_interp = clean_lsf != 0
                # wave_interp = self._wave[select_interp]
                # perform the interpolation on the data
                if method == "spline":
                    intp = interpolate.UnivariateSpline(
                        self._wave[select_interp],
                        clean_lsf[select_interp],
                        s=0,
                        ext="zeros",
                    )
                    new_lsf = intp(ref_wave)
                elif method == "linear":
                    intp = interpolate.interp1d(
                        self._wave[select_interp],
                        clean_lsf[select_interp],
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=(0.0, 0.0),
                    )
                    new_lsf = intp(ref_wave)
            else:
                new_lsf = None

            # interpolate data --------------------------------------------------------------------------------------------------------------------------------
            # replace bad pixels within the spectrum with linear interpolated values
            intp = interpolate.interp1d(
                self._wave[select_goodpix],
                self._data[select_goodpix],
                bounds_error=False,
                assume_sorted=True,
                fill_value=(0.0, 0.0),
            )
            clean_data = intp(self._wave)

            # select pixels that were interpolated (excluding extrapolated ones)
            select_interp = clean_data != 0
            # wave_interp = self._wave[select_interp]
            # perform the interpolation on the data
            if method == "spline":
                intp = interpolate.UnivariateSpline(
                    self._wave[select_interp],
                    clean_data[select_interp],
                    s=0,
                    ext="zeros",
                )
                new_data = intp(ref_wave)
            elif method == "linear":
                intp = interpolate.interp1d(
                    self._wave[select_interp],
                    clean_data[select_interp],
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=(0.0, 0.0),
                )
                new_data = intp(ref_wave)

            # interpolate error -------------------------------------------------------------------------------------------------------------------------------
            select_in = numpy.logical_and(
                ref_wave >= numpy.min(self._wave), ref_wave <= numpy.max(self._wave)
            )
            if self._error is not None and err_sim > 0:
                # replace the error of bad pixels within the spectrum temporarily to ~zero for the MC simulation
                replace_pix = numpy.logical_and(select_badpix, clean_data != 0.0)
                self._error[replace_pix] = 1e-20

                # prepare arrays
                errors = numpy.zeros((err_sim, len(ref_wave)), dtype=numpy.float32)
                error = numpy.zeros(len(self._wave), dtype=numpy.float32)

                # propagate errors using MC simulation
                for i in range(err_sim):
                    error[select_goodpix] = numpy.random.normal(
                        # NOTE: patching negative errors
                        clean_data[select_goodpix],
                        numpy.abs(self._error[select_goodpix]),
                    ).astype(numpy.float32)

                    if method == "spline":
                        intp = interpolate.UnivariateSpline(
                            self._wave[select_interp],
                            error[select_interp],
                            s=0,
                            ext="zeros",
                        )
                        out = intp(ref_wave)
                    elif method == "linear":
                        intp = interpolate.interpolate.interp1d(
                            self._wave[select_interp],
                            error[select_interp],
                            bounds_error=False,
                            assume_sorted=True,
                            fill_value=(0.0, 0.0),
                        )
                        out = intp(ref_wave)
                    errors[i, select_in] = out[select_in]
                new_error = numpy.std(errors, axis=0)
            else:
                new_error = None

            # interpolate mask --------------------------------------------------------------------------------------------------------------------------------
            if self._mask is not None:
                badpix = numpy.zeros(ref_wave.shape[0], dtype=bool)
                indices = numpy.arange(self._wave.shape[0])
                nbadpix = numpy.sum(self._mask)
                if nbadpix > 0:
                    badpix_id = indices[self._mask]
                    for i in range(len(badpix_id)):
                        badpix_min = badpix_id[i] - 2
                        badpix_max = badpix_id[i] + 2
                        bound = numpy.clip(
                            numpy.array([badpix_min, badpix_max]), 0, self._dim - 1
                        )
                        select_bad = numpy.logical_and(
                            ref_wave >= self._wave[bound[0]],
                            ref_wave <= self._wave[bound[1]],
                        )
                        badpix = numpy.logical_or(badpix, select_bad)
                new_mask = numpy.logical_or(
                    badpix,
                    numpy.logical_or(
                        ref_wave < self._wave[0], ref_wave > self._wave[-1]
                    ),
                )
            else:
                new_mask = numpy.logical_or(
                    ref_wave < self._wave[0], ref_wave > self._wave[-1]
                )
                # replace error values in masked pixels
                if new_error is not None:
                    new_error[new_mask] = replace_error

            # interpolate sky ---------------------------------------------------------------------------------------------------------------------------------
            if self._sky is not None:
                intp = interpolate.interp1d(
                    self._wave[select_goodpix],
                    self._sky[select_goodpix],
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=(0.0, 0.0),
                )
                clean_sky = intp(self._wave)

                # select pixels that were interpolated (excluding extrapolated ones)
                select_interp = clean_sky != 0
                # wave_interp = self._wave[select_interp]
                # perform the interpolation on the data
                if method == "spline":
                    intp = interpolate.UnivariateSpline(
                        self._wave[select_interp],
                        clean_sky[select_interp],
                        s=0,
                        ext="zeros",
                    )
                    new_sky = intp(ref_wave)
                elif method == "linear":
                    intp = interpolate.interp1d(
                        self._wave[select_interp],
                        clean_sky[select_interp],
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=(0.0, 0.0),
                    )
                    new_sky = intp(ref_wave)
            else:
                new_sky = None

            # interpolate sky error --------------------------------------------------------------------------------------------------------------------------
            select_in = numpy.logical_and(
                ref_wave >= numpy.min(self._wave), ref_wave <= numpy.max(self._wave)
            )
            if self._sky_error is not None and err_sim > 0:
                # replace the error of bad pixels within the spectrum temporarily to ~zero for the MC simulation
                replace_pix = numpy.logical_and(select_badpix, clean_data != 0.0)
                self._sky_error[replace_pix] = 1e-20

                # prepare arrays
                sky_errors = numpy.zeros((err_sim, len(ref_wave)), dtype=numpy.float32)
                sky_error = numpy.zeros(len(self._wave), dtype=numpy.float32)

                # propagate sky_errors using MC simulation
                for i in range(err_sim):
                    sky_error[select_goodpix] = numpy.random.normal(
                        # NOTE: patching negative sky_errors
                        clean_data[select_goodpix],
                        numpy.abs(self._sky_error[select_goodpix]),
                    ).astype(numpy.float32)

                    if method == "spline":
                        intp = interpolate.UnivariateSpline(
                            self._wave[select_interp],
                            sky_error[select_interp],
                            s=0,
                            ext="zeros",
                        )
                        out = intp(ref_wave)
                    elif method == "linear":
                        intp = interpolate.interpolate.interp1d(
                            self._wave[select_interp],
                            sky_error[select_interp],
                            bounds_error=False,
                            assume_sorted=True,
                            fill_value=(0.0, 0.0),
                        )
                        out = intp(ref_wave)
                    sky_errors[i, select_in] = out[select_in]
                new_sky_error = numpy.std(sky_errors, axis=0)
            else:
                new_sky_error = None

        if extrapolate is not None:
            select_out = numpy.logical_or(
                ref_wave < self._wave[0], ref_wave > self._wave[-1]
            )
            new_data = numpy.where(select_out, extrapolate._data, new_data)
            new_mask = numpy.where(select_out, extrapolate._mask, new_mask)
            if new_error is not None:
                new_error = numpy.where(select_out, extrapolate._error, new_error)
            if new_lsf is not None:
                new_lsf = numpy.where(
                    select_out, extrapolate._lsf, new_lsf
                )
            if new_sky is not None:
                new_sky = numpy.where(select_out, extrapolate._sky, new_sky)
            if new_sky_error is not None:
                new_sky_error = numpy.where(
                    select_out, extrapolate._sky_error, new_error
                )

        spec_out = Spectrum1D(
            data=new_data,
            error=new_error,
            mask=new_mask,
            wave=ref_wave,
            wave_trace=self._wave_trace,
            lsf=new_lsf,
            lsf_trace=self._lsf_trace,
            sky=new_sky,
            sky_error=new_sky_error
        )
        return spec_out

    def resampleSpec_flux_conserving(
        self,
        ref_wave,
        method="spline",
        err_sim=500,
        replace_error=1e10,
        extrapolate=None,
    ):
        old_dlambda = numpy.interp(ref_wave, self._wave[:-1], numpy.diff(self._wave))

        # plt.plot(self._wave, self._data, lw=1, color="k")
        # plt.plot(self._wave, )

        new_dlambda = numpy.diff(ref_wave, append=ref_wave[-1])
        new_spec = self.resampleSpec(
            ref_wave,
            method=method,
            err_sim=err_sim,
            replace_error=replace_error,
            extrapolate=extrapolate,
        )
        # print(self._data)
        # print(new_spec._data)
        new_spec._data *= old_dlambda / new_dlambda
        if self._error is not None:
            new_spec._error *= old_dlambda / new_dlambda

        # print(old_dlambda, new_dlambda, old_dlambda / new_dlambda)
        # print(new_spec._data)
        # plt.plot(ref_wave, new_spec._data, lw=1, color="r")
        # plt.show()

        return new_spec

    def apply_pixelmask(self, mask=None, inplace=False):
        if mask is None:
            mask = self._mask
        if mask is None:
            return self

        if inplace:
            new_spec = self
        else:
            new_spec = deepcopy(self)

        new_spec._data[mask] = numpy.nan
        if new_spec._error is not None:
            new_spec._error[mask] = numpy.nan
        if new_spec._sky is not None:
            new_spec._sky[mask] = numpy.nan
        if new_spec._sky_error is not None:
            new_spec._sky_error[mask] = numpy.nan

        return new_spec

    def interpolate_masked(self, mask=None, inplace=False):
        mask = mask if mask is not None else self._mask
        if mask is None or mask.all():
            return self

        if inplace:
            new_spec = self
        else:
            new_spec = deepcopy(self)

        good_pix = ~mask
        new_spec._data = numpy.interp(new_spec._wave, new_spec._wave[good_pix], new_spec._data[good_pix], left=new_spec._data[good_pix][0], right=new_spec._data[good_pix][-1])
        if new_spec._error is not None:
            new_spec._error = numpy.interp(new_spec._wave, new_spec._wave[good_pix], new_spec._error[good_pix], left=new_spec._error[good_pix][0], right=new_spec._error[good_pix][-1])
        if new_spec._sky is not None:
            new_spec._sky = numpy.interp(new_spec._wave, new_spec._wave[good_pix], new_spec._sky[good_pix], left=new_spec._sky[good_pix][0], right=new_spec._sky[good_pix][-1])
        if new_spec._sky_error is not None:
            new_spec._sky_error = numpy.interp(new_spec._wave, new_spec._wave[good_pix], new_spec._sky_error[good_pix], left=new_spec._sky_error[good_pix][0], right=new_spec._sky_error[good_pix][-1])

        return new_spec

    def flatten_lsf(self, target_fwhm, min_fwhm=0.5*2.354, interpolate_bad=True, inplace=False):
        """Degrades spectral resolution to match a constant resolution in FWHM

        Parameters
        ----------
        target_fwhm : float
            Spectral resolution in FWHM to degrade to
        min_fwhm : float, optional
            Minimum resolution to allow in case any target_fwhm <= fwhm, by default 0.5
        interpolate_bad : bool, optional
            Interpolate bad pixels before convolution, by default True
        inplace : bool, optional
            Degrade resolution in place

        Returns
        -------
        new_spec : lvmdrp.core.spectrum1d.Spectrum1D
            New spectrum with constant LSF
        """
        if self._lsf is None:
            return self

        # make a copy of spectrum if not inplace
        if inplace:
            new_spec = self
        else:
            new_spec = deepcopy(self)

        # interpolate masked pixels
        if interpolate_bad:
            new_spec = new_spec.interpolate_masked(inplace=inplace)

        data = new_spec._data
        wave = new_spec._wave
        fwhm = new_spec._lsf
        if new_spec._error is not None:
            error = new_spec._error
        else:
            error = None
        if new_spec._sky is not None:
            sky = new_spec._sky
        else:
            sky = None
        if new_spec._sky_error is not None:
            sky_error = new_spec._sky_error
        else:
            sky_error = None

        # define Gaussian sigmas
        dfwhm = target_fwhm - fwhm
        if numpy.any(dfwhm <= 0):
            # correcting given resolution to match minimum value allowed
            target_fwhm += min_fwhm - min(dfwhm)
        sigmas = numpy.sqrt(target_fwhm**2 - fwhm**2) / 2.354 / numpy.gradient(wave)

        # setup kernel
        pixels = numpy.ceil(3 * max(sigmas))
        pixels = numpy.arange(-pixels, pixels)
        kernel = numpy.asarray([numpy.exp(-0.5 * (pixels / sigmas[iw]) ** 2) for iw in range(wave.size)])
        kernel = convolution_matrix(kernel)
        new_data = kernel @ data

        # import matplotlib.pyplot as plt
        # from astropy.visualization import simple_norm
        # plt.figure(figsize=(10,10), layout="constrained")
        # plt.imshow(kernel.toarray(), cmap="coolwarm", norm=simple_norm(kernel.toarray(), stretch="log"))
        # plt.show()

        # gauss_sig = numpy.zeros_like(fwhm)
        # select = target_fwhm > fwhm
        # gauss_sig[select] = numpy.sqrt(target_fwhm**2 - fwhm[select] ** 2) / 2.354
        # fact = numpy.sqrt(2.0 * numpy.pi)
        # kernel = numpy.exp(
        #     -0.5
        #     * (
        #         (wave[:, numpy.newaxis] - wave[numpy.newaxis, :])
        #         / gauss_sig[numpy.newaxis, :]
        #     )
        #     ** 2
        # ) / (fact * gauss_sig[numpy.newaxis, :])
        # multiplied = data[:, numpy.newaxis] * kernel
        # new_data = bn.nansum(multiplied, axis=0) / bn.nansum(kernel, 0)

        new_spec._data = new_data
        new_spec._lsf[:] = target_fwhm
        if error is not None:
            new_spec._error = numpy.sqrt((kernel @ error) ** 2)
        if sky is not None:
            new_spec._sky = kernel @ sky
        if sky_error is not None:
            new_spec._sky_error = numpy.sqrt((kernel @ sky_error) ** 2)

        new_spec = new_spec.apply_pixelmask(inplace=inplace)

        return new_spec

    def binSpec(self, new_wave):
        new_disp = new_wave[1:] - new_wave[:-1]
        new_disp = numpy.insert(new_disp, 0, new_disp[0])
        data_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
        mask_out = numpy.zeros(len(new_wave), dtype="bool")
        sky_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
        sky_error_out = numpy.zeros(len(new_wave), dtype=numpy.float32)

        if self._mask is not None:
            mask_in = numpy.logical_and(self._mask)
        else:
            mask_in = numpy.ones(len(self._wave), dtype="bool")
        # masked_data = self._wave[mask_in]
        masked_wave = self._wave[mask_in]

        if self._error is not None:
            error_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
            masked_error = self._error[mask_in]
        else:
            error_out = None

        if self._sky_error is not None:
            sky_error_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
            masked_sky_error = self._sky_error[mask_in]

        bound_min = new_wave - new_disp / 2.0
        bound_max = new_wave + new_disp / 2.0

        for i in range(len(new_wave)):
            select = numpy.logical_and(
                masked_wave >= bound_min[i], masked_wave <= bound_max[i]
            )
            if numpy.sum(select) > 0:
                data_out[i] = numpy.sum(
                    numpy.abs(masked_wave[select] - new_wave[i])
                    * self._data[mask_in][select]
                ) / numpy.sum(numpy.abs(masked_wave[select] - new_wave[i]))
                if self._error is not None:
                    error_out[i] = numpy.sqrt(
                        numpy.sum(masked_error[select] ** 2) / numpy.sum(select) ** 2
                    )
                if self._sky is not None:
                    sky_out[i] = numpy.sum(
                        numpy.abs(masked_wave[select] - new_wave[i])
                        * self._sky[mask_in][select]
                    ) / numpy.sum(numpy.abs(masked_wave[select] - new_wave[i]))
                if self._sky_error is not None:
                    sky_error_out[i] = numpy.sqrt(
                        numpy.sum(masked_sky_error[select] ** 2)
                        / numpy.sum(select) ** 2
                    )
            else:
                mask_out[i] = True
        data_out = numpy.interp(new_wave, masked_wave, self._data[mask_in])
        if self._sky is not None:
            sky_out = numpy.interp(new_wave, masked_wave, self._sky[mask_in])

        spec = Spectrum1D(data=data_out, wave=new_wave, error=error_out, mask=mask_out, sky=sky_out, sky_error=sky_error_out)

        return spec

    def smoothSpec(self, size, method="gauss", mode="nearest"):
        """
        Smooth the spectrum

        Parameters
        --------------
        size : float or int
            Size of the smooth window or Gaussian width (sigma)
        method : string, optional with default: 'gauss'
            Available methods are 'gauss' - convolution with Gaussian kernel or
            'median' - median smoothing of the spectrum
        mode :  string, optional with default: 'nearest'
            Set the mode how to handle the boundarys within the convolution
            Possilbe modes are: reflect, constant, nearest, mirror,  wrap
        """
        if method == "gauss":
            # filter with Gaussian kernel
            self._data = ndimage.filters.gaussian_filter1d(self._data, size, mode=mode)
        elif method == "median":
            # filter with median filter
            self._data = ndimage.filters.median_filter(self._data, size, mode=mode)
        elif method == "BSpline":
            smooth = interpolate.splrep(
                self._wave[~self._mask],
                self._data[~self._mask],
                w=1.0 / numpy.sqrt(numpy.fabs(self._data[~self._mask])),
                s=size,
            )
            self._data = interpolate.splev(self._wave, smooth, der=0)

    def smoothGaussVariable(self, diff_fwhm):
        fact = numpy.sqrt(2.0 * numpy.pi)
        if self._mask is not None:
            mask = numpy.logical_not(self._mask)
        else:
            mask = numpy.ones(self._data.shape[0], dtype="bool")

        if isinstance(diff_fwhm, float) or isinstance(diff_fwhm, int):
            diff_fwhm = numpy.ones(self._data.shape[0], dtype=numpy.float32) * diff_fwhm
        select = diff_fwhm > 0.0
        mask = numpy.logical_and(mask, select)

        data = numpy.zeros_like(self._data)
        data[:] = self._data
        GaussKernels = (
            1.0
            * numpy.exp(
                -0.5
                * (
                    (
                        self._wave[mask][:, numpy.newaxis]
                        - self._wave[mask][numpy.newaxis, :]
                    )
                    / numpy.abs(diff_fwhm[mask][numpy.newaxis, :] / 2.354)
                )
                ** 2
            )
            / (fact * numpy.abs(diff_fwhm[mask][numpy.newaxis, :] / 2.354))
        )
        data[mask] = numpy.sum(
            self._data[mask][:, numpy.newaxis] * GaussKernels, 0
        ) / numpy.sum(GaussKernels, 0)

        if self._error is not None:
            error = numpy.zeros_like(self._error)
            error[:] = self._error
            error[mask] = numpy.sqrt(
                numpy.sum((self._error[mask] * GaussKernels) ** 2, 0)
            ) / numpy.sum(GaussKernels, 0)
            # scale = Spectrum1D(wave=self._wave, data=error/self._error)
            # scale.smoothSpec(40, method='median')
            # error[mask]=error[mask]/scale._data[mask]
        else:
            error = None

        if self._lsf is not None:
            lsf = numpy.sqrt(self._lsf**2 + diff_fwhm**2)
        else:
            lsf = diff_fwhm

        if self._sky is not None:
            sky = numpy.zeros_like(self._sky)
            sky[:] = self._sky
            sky[mask] = numpy.sum(
                self._sky[mask][:, numpy.newaxis] * GaussKernels, 0
            ) / numpy.sum(GaussKernels, 0)
        if self._sky_error is not None:
            sky_error = numpy.zeros_like(self._sky_error)
            sky_error[:] = self._sky_error
            sky_error[mask] = numpy.sqrt(
                numpy.sum((self._sky_error[mask] * GaussKernels) ** 2, 0)
            ) / numpy.sum(GaussKernels, 0)

        spec = Spectrum1D(
            wave=self._wave,
            data=data,
            error=error,
            mask=self._mask,
            lsf=lsf,
            sky=sky,
            sky_error=sky_error,
        )
        return spec

    def smoothPoly(
        self, deg=5, poly_kind="legendre", start_wave=None, end_wave=None, ref_base=None
    ):
        if self._mask is not None:
            mask = numpy.logical_not(self._mask)
        else:
            mask = numpy.ones(self._dim, dtype="bool")
        if start_wave is not None:
            mask = numpy.logical_and(mask, self._wave >= start_wave)
        if end_wave is not None:
            mask = numpy.logical_and(mask, self._wave <= end_wave)
        mask = numpy.logical_and(mask, numpy.logical_not(numpy.isnan(self._data)))
        if numpy.sum(mask) > numpy.fabs(deg):
            if poly_kind == "poly":
                poly = polynomial.Polynomial.fit(
                    self._wave[mask], self._data[mask], deg=deg
                )
            elif poly_kind == "legendre":
                poly = polynomial.Legendre.fit(
                    self._wave[mask], self._data[mask], deg=deg
                )
            out_par = poly.convert().coef

            if ref_base is None:
                self._data = poly(self._wave)
            else:
                self._data = poly(ref_base)
                self._wave = ref_base
                self._dim = len(ref_base)
                self._pixels = numpy.arange(self._dim)
                if self._mask is not None:
                    mask = numpy.zeros(self._dim, dtype="bool")
        else:
            self._data[:] = 0
            if self._mask is not None:
                self._mask[:] = True
            out_par = 0
        return out_par

    def findPeaks(
        self,
        pix_range=None,
        min_dwave=5.0,
        threshold=100.0,
        npeaks=None,
        add_doubles=1e-1,
        maxiter=400,
    ):
        """
        Select local maxima in a Spectrum without taken subpixels into account.

        Parameters
        --------------
        pix_range : tuple, optional with default=None
            Tuple of the pixel range to be considered.
        min_dwave : float, optional with default=3.5
            Minimum distance between two maxima in pixels.
        threshold : float, optional with default=100.0
            Threshold above all pixels are assumed to be maxima,
            it is not used if an expected number of peaks is given.
        npeaks : int, optional with default=0
            Number of expected maxima that should be matched.
            If 0 is given the number of maxima is not constrained.
        add_doubles : float, optional with defaul=1e-3
        maxiter : int, optional with default=400
            Maximum number of iterations to find the peaks, when npeaks is set.

        Returns (pixel, wave, data)
        -----------
        pixels :  numpy.ndarray (int)
            Array of the pixel peak positions
        wave :  numpy.ndarray (float)
            Array of the wavelength peak positions
        data : numpy.ndarray (float)
            Array of the data values at the peak position

        """

        # select pixels within given range
        if pix_range is not None:
            ini = max(0, pix_range[0] - 1)
            fin = min(self._data.size, pix_range[1] - 2)
            s = slice(ini, fin)
            data = self._data[s]
            wave = self._wave[s]
            pixels = self._pixels[s]
            if self._mask is not None:
                mask = self._mask[s]
            else:
                mask = numpy.zeros_like(data, dtype=bool)

        # check for identical adjacent values to use derivative for maxima detection
        doubles = data[1:] == data[:-1]
        doubles = numpy.insert(doubles, 0, False)
        idx = numpy.arange(len(doubles))
        # add some value to one of those adjacent data points
        if numpy.sum(doubles) > 0:
            double_idx = idx[doubles]
            data[double_idx] += add_doubles

        # mask bad pixels
        data = data[~mask]
        wave = wave[~mask]
        pixels = pixels[~mask]

        # compute the discrete derivative
        dwave = wave[1:] - wave[:-1]
        pos_diff = (data[1:] - data[:-1]) / dwave
        # all peaks selected by derivative sign change
        select_peaks = (pos_diff[1:] < 0) & (pos_diff[:-1] > 0)

        # define initial peaks
        peaks = wave[1:-1][select_peaks]

        # peaks selection by minimum distance in pixels
        if min_dwave is not None:
            select_dwave = numpy.diff(peaks) >= min_dwave
        else:
            select_dwave = numpy.ones(peaks.size - 1, dtype=bool)

        # if no number of peaks are given select all maxima over a given threshold
        if npeaks is None or npeaks <= 0:
            select_thres = data[1:-1][select_peaks][:-1][select_dwave] > threshold
        # if a specific number of peaks are expected iterate until correct number of peaks are found
        else:
            matched_peaks = True
            threshold = self.max()[0] / 10.0  # set starting threshold
            m = 0
            while matched_peaks and m < maxiter:
                # select all maxima above threshold
                select_thres = data[1:-1][select_peaks][:-1][select_dwave] > threshold
                # check if the number of peaks match expectation
                peaks = numpy.sum(select_thres)
                # if the number of peaks mismatch adjust threshold value to a new value
                if peaks < npeaks:
                    threshold = threshold / 2.0
                elif peaks > npeaks:
                    threshold = threshold * 1.5
                else:
                    # indicate that the correct number of peaks are not  found
                    matched_peaks = False
                m += 1

        # select pixel positions of peaks
        pixels = pixels[1:-1][select_peaks][:-1][select_dwave][select_thres]
        # select wavelength positions of peaks
        wave = wave[1:-1][select_peaks][:-1][select_dwave][select_thres]
        # select data valuesof peaks
        data = data[1:-1][select_peaks][:-1][select_dwave][select_thres]

        return pixels, wave, data


    def measure_fibers_profile(self, centroids_guess, fwhms_guess=2.5, counts_range=[0,numpy.inf], centroids_range=[-5,5], fwhms_range=[1.0,3.5],
                               npixels=2, ftol=1e-3, xtol=1e-3, solver="dogbox"):
        """Finds the subpixel centers for local maxima in a spectrum by fitting a Gaussian to each peak.

        Parameters
        ----------
        centroids_guess : numpy.ndarray
            Initial guess for the peak centers (pixel positions).
        fwhms_guess : float, optional
            Initial guess for the Gaussian width (sigma) used for modeling each peak (default: 2.5).
        bounds : tuple of numpy.ndarray or float, optional
            Lower and upper bounds for the fit parameters (amplitude, center, sigma) for each peak.
            Should be a tuple (lower, upper), where each is an array of length 3*N or a scalar (default: (-inf, inf)).
        npixels : int, optional
            Number of pixels around the peak to use in the fitting, by default +/-2
        ftol : float, optional
            Relative tolerance for the fit optimization (default: 1e-3).
        xtol : float, optional
            Absolute tolerance for the fit optimization (default: 1e-3).
        solver : str, optional
            Optimization algorithm to use (default: "dogbox").

        Returns
        -------
        centroids : numpy.ndarray
            Array of subpixel peak positions (float).
        mask : numpy.ndarray
            Boolean array indicating which peaks have uncertain or invalid measurements.

        Notes
        -----
        For each initial peak position, a Gaussian is fit to the 3 brightest pixels around the peak.
        Peaks for which the fit fails or returns NaN are masked as invalid.
        """
        fact = numpy.sqrt(2 * numpy.pi)
        sigmas_guess = fwhms_guess / 2.354

        counts = numpy.full(len(centroids_guess), numpy.nan, dtype="float32")
        centroids = numpy.full(len(centroids_guess), numpy.nan, dtype="float32")
        sigmas = numpy.full(len(centroids_guess), numpy.nan, dtype="float32")
        mask = numpy.isnan(centroids_guess)

        bounds = self._parse_gaussians_boundaries(
            ngaussians=centroids.size, centroids=centroids_guess, counts_range=counts_range, centroids_range=centroids_range, fwhms_range=fwhms_range, to_sigmas=True)

        counts_lower, centroids_lower, sigmas_lower = numpy.split(bounds[0], 3)
        counts_upper, centroids_upper, sigmas_upper = numpy.split(bounds[1], 3)
        for j in range(len(centroids_guess)):
            if mask[j]:
                continue

            pixels_selection = numpy.logical_and(
                self._wave > centroids_guess[j] - npixels,
                self._wave < centroids_guess[j] + npixels,
            )
            counts_guess = numpy.interp(centroids_guess[j], self._wave, self._data) * fact * sigmas_guess

            guess_par = [counts_guess, centroids_guess[j], sigmas_guess]
            gauss = fit_profile.Gaussian(guess_par)
            gauss.fit(
                self._wave[pixels_selection],
                self._data[pixels_selection],
                sigma=self._error[pixels_selection],
                p0=guess_par,
                bounds=([counts_lower[j], centroids_lower[j], sigmas_lower[j]], [counts_upper[j], centroids_upper[j], sigmas_upper[j]]),
                ftol=ftol, xtol=xtol,
                solver=solver)

            params = gauss.getPar()
            counts[j] = params[0]
            centroids[j] = params[1]
            sigmas[j] = params[2]

        mask = mask | numpy.isnan(counts) | numpy.isnan(centroids)# | numpy.isnan(sigmas)
        centroids[mask] = numpy.nan

        return counts, centroids, sigmas*2.354, mask

    def measureFWHMPeaks(
        self, pos, nblocks, init_fwhm=2.4, threshold_flux=None, plot=-1
    ):
        """
        Measures the FWHM of emission lines in the spectrum by a Gaussian modelling assuming that the FWHM is constant for a certain number of fibers (blocks).
        If the spectrum provides an error vector, it is taken into account in the modelling.

        Parameters
        --------------
        pos : numpy.ndarray
            Central peak positions for each peak which are fixed during the modelling
        nblocks : integer
            Number of fibers which are modelled simultaneously with a common FWHM.
            Remaining fibers are included in the last block
        init_fwhm: float, optional with default=2.4
            Initial guess of the Gaussian FWHM used for the modelling.
        threshold_flux: float, optional with default = None
            Mininmum flux included in each peak for an accurate measurement.
            If 20% of the fiber to not full fill this requirement the block FWHM is masked

        Returns (fwhm, mask)
        -----------
        fwhm :  numpy.ndarray (float)
            Array of Gaussian FWHM for each peak
        mask : numpy.ndarray (bool)
            Array of uncertain FWHM measurements
        """
        # create empty fwhm and mask arrays
        fibers = len(pos)
        fwhm = numpy.ones(fibers, dtype=numpy.float32)
        mask = numpy.ones(fibers, dtype="bool")

        # setup the blocks of peaks to be modelled simulatenously
        brackets = numpy.arange(0, fibers, nblocks)
        res = fibers % nblocks
        if res == 0:
            brackets = numpy.append(brackets, [fibers + 1])
        else:
            brackets[-1] += res

        # iterate over the blocks
        for i in range(len(brackets) - 1):
            pos_block = pos[
                brackets[i] : brackets[i + 1]
            ]  # cut out the corresponding peak positions
            median_dist = bn.nanmedian(
                pos_block[1:] - pos_block[:-1]
            )  # compute median distance between peaks
            flux = (
                self._data[numpy.round(pos_block).astype("int16")]
                * numpy.sqrt(2)
                * init_fwhm
                / 2.354
            )  # initial guess for the flux

            # compute lower and upper bounds of the positions for each block
            lo = int(bn.nanmin(pos_block) - median_dist)
            if lo <= 0:
                lo = 0
            hi = int(bn.nanmax(pos_block) + median_dist)
            if hi >= self._wave[-1]:
                hi = self._wave[-1]

            # modell each block of peaks with Gaussians with and without associate errors
            par = numpy.insert(
                flux.astype(numpy.float32), 0, init_fwhm / 2.354
            )  # set initial paramters
            gaussians_fix_width = fit_profile.Gaussians_width(
                par, pos_block
            )  # define profile with initial paramter
            if self._error is not None:
                gaussians_fix_width.fit(
                    self._wave[lo:hi],
                    self._data[lo:hi],
                    sigma=self._error[lo:hi],
                    maxfev=1000,
                    xtol=1e-4,
                    ftol=1e-4,
                )  # fit with errors
            else:
                gaussians_fix_width.fit(
                    self._wave[lo:hi],
                    self._data[lo:hi],
                    maxfev=1000,
                    xtol=1e-4,
                    ftol=1e-4,
                )  # fit without errors
            fit_par = gaussians_fix_width.getPar()
            if plot == i:
                gaussians_fix_width.plot(self._wave[lo:hi], self._data[lo:hi])

            fwhm[brackets[i] : brackets[i + 1]] = (
                fit_par[0] * 2.354
            )  # convert Gaussian sigma to FWHM
            # create the bad pixel mask
            if threshold_flux is not None:
                masked = numpy.logical_or(
                    numpy.sum(fit_par[1:] < 0) > 0,
                    numpy.sum(fit_par[1:] > threshold_flux) < 0.2 * len(fit_par[1:]),
                )
                mask[brackets[i] : brackets[i + 1]] = masked
            else:
                mask[brackets[i] : brackets[i + 1]] = numpy.sum(fit_par[1:] < 0) > 0

        # return results
        return fwhm, mask

    def measureOffsetPeaks(
        self, pos, mask, nblocks, init_fwhm=2.0, init_offset=0.0, plot=-1
    ):
        """ """
        # create empty fwhm and mask arrays
        fibers = len(pos)
        if mask is None:
            good = numpy.ones(fibers, dtype="bool") & (pos <= (self._data.shape[0] - 1))
        else:
            good = numpy.logical_not(mask) & (pos <= (self._data.shape[0] - 1))

        # setup the blocks of peaks to be modelled simulatenously
        blocks = numpy.array_split(numpy.arange(0, fibers), nblocks)

        offsets = numpy.zeros(len(blocks), dtype=numpy.float32)
        med_pos = numpy.zeros_like(offsets)

        # iterate over the blocks
        for i in range(len(blocks)):
            pos_block = pos[blocks[i]]  # cut out the corresponding peak positions
            pos_mask = good[blocks[i]]
            if numpy.sum(pos_mask) > 0:
                median_dist = bn.median(
                    pos_block[pos_mask][1:] - pos_block[pos_mask][:-1]
                )  # compute median distance between peaks
                flux = (
                    self._data[numpy.round(pos_block[pos_mask]).astype("int16")]
                    * numpy.sqrt(2)
                    * init_fwhm
                    / 2.354
                )  # initial guess for the flux

                # compute lower and upper bounds of the positions for each block
                lo = int(pos_block[pos_mask][0] - median_dist)
                if lo <= 0:
                    lo = 0
                hi = int(pos_block[pos_mask][-1] + median_dist)
                if hi >= self._wave[-1]:
                    hi = self._wave[-1]

                # modell each block of peaks with Gaussians with and without associate errors
                par = numpy.insert(
                    flux.astype(numpy.float32), 0, init_fwhm / 2.354
                )  # set initial paramters
                par = numpy.append(par, init_offset)  # set initial paramters
                gaussians_offset = fit_profile.Gaussians_offset(
                    par, pos_block[pos_mask]
                )  # define profile with initial paramters
                if self._error is not None:
                    gaussians_offset.fit(
                        self._wave[lo:hi],
                        self._data[lo:hi],
                        sigma=self._error[lo:hi],
                        maxfev=4000,
                        xtol=1e-8,
                        ftol=1e-8,
                    )  # fit with errors
                else:
                    gaussians_offset.fit(
                        self._wave[lo:hi],
                        self._data[lo:hi],
                        maxfev=4000,
                        xtol=1e-8,
                        ftol=1e-8,
                    )  # fit without errors
                fit_par = gaussians_offset.getPar()
                if plot == i:
                    gaussians_offset.plot(self._wave[lo:hi], self._data[lo:hi])

                offsets[i] = fit_par[-1]  # get offset position
                med_pos[i] = bn.mean(self._wave[lo:hi])
            else:
                offsets[i] = 0.0
                med_pos[i] = 0.0

        return offsets, med_pos

    def measureOffsetPeaks2(
        self, pos, mask, fwhm, nblocks, min_offset, max_offset, step_offset, plot=0
    ):
        fibers = len(pos)
        if mask is None:
            good = numpy.ones(fibers, dtype="bool") & (pos <= (self._data.shape[0] - 1))
        else:
            good = numpy.logical_not(mask) & (pos <= (self._data.shape[0] - 1))

        # setup the blocks of peaks to be modelled simulatenously
        blocks = numpy.array_split(numpy.arange(0, fibers), nblocks)

        offsets = numpy.zeros(len(blocks), dtype=numpy.float32)
        med_pos = numpy.zeros_like(offsets)

        # iterate over the blocks
        for i in range(len(blocks)):
            pos_block = pos[blocks[i]]  # cut out the corresponding peak positions
            pos_mask = good[blocks[i]]
            pos_fwhm = fwhm[blocks[i]]
            if numpy.sum(pos_mask) > 0:
                median_dist = bn.median(
                    pos_block[pos_mask][1:] - pos_block[pos_mask][:-1]
                )  # compute median distance between peaks

                # compute lower and upper bounds of the positions for each block
                lo = int(pos_block[pos_mask][0] - median_dist)
                if lo <= 0:
                    lo = 0
                hi = int(pos_block[pos_mask][-1] + median_dist)
                if hi >= self._wave[-1]:
                    hi = self._wave[-1]
                Gaussian_vec = numpy.zeros(
                    (numpy.sum(pos_mask), hi - lo), dtype=numpy.float32
                )
                x = numpy.arange(hi - lo) + lo
                offset = numpy.arange(min_offset, max_offset, step_offset)
                chisq = numpy.zeros(len(offset))
                max_flux = numpy.zeros(len(offset))
                for o in range(len(offset)):
                    for g in range(numpy.sum(pos_mask)):
                        Gaussian_vec[g, :] = numpy.exp(
                            -0.5
                            * (
                                (x - (pos_block[pos_mask][g] + offset[o]))
                                / (pos_fwhm[pos_mask][g] / 2.354)
                            )
                            ** 2
                        ) / (
                            numpy.sqrt(2.0 * numpy.pi)
                            * abs((pos_fwhm[pos_mask][g] / 2.354))
                        )
                    result = numpy.linalg.lstsq(Gaussian_vec.T, self._data[lo:hi])
                    chisq[o] = result[1][0]
                    max_flux[o] = numpy.sum(result[0])
                find_max = numpy.argsort(max_flux)[-1]
                offsets[i] = offset[find_max]  # get offset position
                med_pos[i] = bn.mean(self._wave[lo:hi])
            else:
                offsets[i] = 0.0
                med_pos[i] = 0.0
        return offsets, med_pos

    def _guess_gaussians_integral(self, centroids, fwhms, nsigma=6, return_pixels_selection=False):
        fact = numpy.sqrt(2 * numpy.pi)
        integrals = numpy.zeros(len(centroids), dtype=numpy.float32)
        sigmas = fwhms / 2.354

        select = numpy.zeros(self._dim, dtype="bool")
        for i in range(len(centroids)):
            select_ = numpy.logical_and(
                self._wave > centroids[i] - nsigma * sigmas[i],
                self._wave < centroids[i] + nsigma * sigmas[i],
            )
            integrals[i] = numpy.interp(centroids[i], self._wave, self._data) * fact * sigmas[i]
            select = numpy.logical_or(select, select_)
        if return_pixels_selection:
            return integrals, select
        return integrals

    def _parse_gaussians_params(self, counts=None, centroids=None, sigmas=None, fwhms=None, to_sigmas=False, to_fwhms=False):
        if fwhms is not None and sigmas is not None:
            raise ValueError(f"Invalid values for `fwhms` or `sigmas`: {sigmas = }, {fwhms = }. Only one or none has to be given")

        params = []
        if counts is not None:
            params.append(counts)
        if centroids is not None:
            params.append(centroids)
        if sigmas is not None:
            params.append(sigmas * (2.354 if to_fwhms else 1.0))
        if fwhms is not None:
            params.append(fwhms / (2.354 if to_sigmas else 1.0))

        return numpy.concatenate(params)

    def _parse_gaussians_boundaries(self, ngaussians, counts=None, centroids=None, fwhms=None, counts_range=None, centroids_range=None, fwhms_range=None, to_sigmas=False):

        bounds_lower, bounds_upper = [], []
        _ = numpy.ones(ngaussians)
        def _set_boundaries(x, x_range, to_sigmas=False, clip=None):
            if x is not None and x_range is None:
                raise ValueError(f"Invalid value for `x_range`: {x_range = }. Expected `x_range` when `x` is given")

            if x is not None and x_range is not None:
                lower = x + x_range[0]
                upper = x + x_range[1]
            elif x_range is not None:
                lower = _ * x_range[0]
                upper = _ * x_range[1]
            else:
                lower = numpy.array([])
                upper = numpy.array([])

            if clip is not None and isinstance(clip, (tuple,list)) and len(clip) == 2:
                if clip[0] is not None:
                    lower = numpy.clip(lower, a_min=clip[0], a_max=None)
                if clip[1] is not None:
                    upper = numpy.clip(upper, a_min=None, a_max=clip[1])

            if to_sigmas:
                lower /= 2.354
                upper /= 2.354

            bounds_lower.append(lower)
            bounds_upper.append(upper)

        _set_boundaries(counts, counts_range, clip=(0.0,None))
        _set_boundaries(centroids, centroids_range, clip=(self._wave.min(),self._wave.max()))
        _set_boundaries(fwhms, fwhms_range, clip=(0.0,None), to_sigmas=to_sigmas)

        if len(bounds_lower) == 0 or len(bounds_upper) == 0:
            return [-numpy.inf, +numpy.inf]

        return [numpy.concatenate(bounds_lower), numpy.concatenate(bounds_upper)]

    def fitMultiGauss(self, pixels_selection, counts_guess, centroids_guess, fwhms_guess, counts_range=[0.0,numpy.inf], centroids_range=[-5,+5], fwhms_range=[1.0,3.5],
                      ftol=1e-3, xtol=1e-3, solver="trf", loss="linear"):
        error = numpy.ones(self._dim, dtype=numpy.float32) if self._error is None else self._error

        guess = self._parse_gaussians_params(counts=counts_guess, centroids=centroids_guess, fwhms=fwhms_guess, to_sigmas=True)
        bounds = self._parse_gaussians_boundaries(
            ngaussians=counts_guess.size, centroids=centroids_guess,
            counts_range=counts_range, centroids_range=centroids_range, fwhms_range=fwhms_range, to_sigmas=True)

        gauss_multi = fit_profile.Gaussians(guess)
        gauss_multi.fit(
            self._wave[pixels_selection], self._data[pixels_selection], sigma=error[pixels_selection],
            bounds=bounds, ftol=ftol, xtol=xtol, solver=solver, loss=loss)

        counts, centroids, sigmas = numpy.split(gauss_multi.getPar(), 3)
        params = self._parse_gaussians_params(counts, centroids, sigmas=sigmas, to_fwhms=True)

        return gauss_multi, params

    def fitMultiGauss_fixed_counts(self, pixels_selection, counts, centroids, fwhms_guess, fwhms_range=[1.0,3.5], ftol=1e-3, xtol=1e-3, solver="trf", loss="linear"):
        error = numpy.ones(self._dim, dtype=numpy.float32) if self._error is None else self._error

        guess = self._parse_gaussians_params(fwhms=fwhms_guess, to_sigmas=True)
        fixed = self._parse_gaussians_params(counts=counts, centroids=centroids)
        bounds = self._parse_gaussians_boundaries(ngaussians=counts.size, fwhms_range=fwhms_range, to_sigmas=True)

        gauss_multi = fit_profile.Gaussians_width(guess, args=fixed)
        gauss_multi.fit(
            self._wave[pixels_selection], self._data[pixels_selection], sigma=error[pixels_selection],
            bounds=bounds, ftol=ftol, xtol=xtol, solver=solver, loss=loss)

        sigmas = gauss_multi.getPar()
        params = self._parse_gaussians_params(counts, centroids, sigmas, to_fwhms=True)
        return gauss_multi, params

    def fitMultiGauss_centroids(self, pixels_selection, counts, centroids_guess, fwhms, centroids_range=[-5,+5], ftol=1e-3, xtol=1e-3, solver="trf", loss="linear"):
        error = numpy.ones(self._dim, dtype=numpy.float32) if self._error is None else self._error

        guess = self._parse_gaussians_params(centroids=centroids_guess)
        fixed = self._parse_gaussians_params(counts=counts, fwhms=fwhms, to_sigmas=True)
        bounds = self._parse_gaussians_boundaries(ngaussians=counts.size, centroids=centroids_guess, centroids_range=centroids_range, to_sigmas=True)

        gauss_multi = fit_profile.Gaussians_centroids(guess, args=fixed)
        gauss_multi.fit(
            self._wave[pixels_selection], self._data[pixels_selection], sigma=error[pixels_selection],
            bounds=bounds, ftol=ftol, xtol=xtol, solver=solver, loss=loss)

        centroids = gauss_multi.getPar()
        params = self._parse_gaussians_params(counts, centroids, fwhms, to_fwhms=False)
        return gauss_multi, params

    def fitMultiGauss_fixed_width(self, pixels_selection, counts_guess, centroids, fwhms, counts_range=[0.0,numpy.inf], ftol=1e-3, xtol=1e-3, solver="trf", loss="linear"):
        error = numpy.ones(self._dim, dtype=numpy.float32) if self._error is None else self._error

        guess = self._parse_gaussians_params(counts=counts_guess)
        fixed = self._parse_gaussians_params(centroids=centroids, fwhms=fwhms, to_sigmas=True)
        bounds = self._parse_gaussians_boundaries(ngaussians=counts_guess.size, counts_range=counts_range)

        gauss_multi = fit_profile.Gaussians_counts(guess, args=fixed)
        gauss_multi.fit(
            self._wave[pixels_selection], self._data[pixels_selection], sigma=error[pixels_selection],
            bounds=bounds, ftol=ftol, xtol=xtol, solver=solver, loss=loss)

        counts = gauss_multi.getPar()
        params = self._parse_gaussians_params(counts, centroids, fwhms, to_fwhms=False)
        return gauss_multi, params

    def fitMultiGauss_alphas(self, pixels_selection, counts, centroids, fwhms, alphas, alphas_range=[-1.0,+1.0], ftol=1e-3, xtol=1e-3, solver="trf", loss="linear"):
        error = numpy.ones(self._dim, dtype=numpy.float32) if self._error is None else self._error

        guess = alphas
        _ = numpy.ones_like(alphas)
        fixed = self._parse_gaussians_params(counts=counts, centroids=centroids, fwhms=fwhms, to_sigmas=True)
        bounds = [_ * alphas_range[0], _ * alphas_range[1]]

        gauss_multi = fit_profile.SkewedGaussians(guess, args=fixed)
        gauss_multi.fit(
            self._wave[pixels_selection], self._data[pixels_selection], sigma=error[pixels_selection],
            bounds=bounds, ftol=ftol, xtol=xtol, solver=solver, loss=loss)

        alphas = gauss_multi.getPar()
        params = numpy.concatenate([counts, centroids, fwhms, alphas])
        return gauss_multi, params

    def fit_gaussians(self, pars_guess, pars_fixed, bounds, fitting_params, profile="mexhat", npixels=4, oversampling_factor=10, axs=None):
        profile_class = fit_profile.PROFILES.get(profile)
        if profile_class is None:
            raise ValueError(f"Invalid value for `profile`: {profile}. Expected one of: {fit_profile.PROFILES}")

        error = numpy.ones(self._dim, dtype=numpy.float32) if self._error is None else self._error

        centroids = pars_guess.get("centroids", pars_fixed.get("centroids"))
        lower = numpy.nanmin(centroids) - npixels
        upper = numpy.nanmax(centroids) + npixels
        pixels_selection = (lower <= self._wave) & (self._wave <= upper)

        model = profile_class(pars=pars_guess, fixed=pars_fixed, bounds=bounds, oversampling_factor=oversampling_factor)
        kwargs = fitting_params.copy()
        args = kwargs.pop("args", ())
        model.fit(self._wave[pixels_selection], self._data[pixels_selection], error[pixels_selection], *args, **kwargs)

        params = model._pars
        errors = model._errs

        if axs is not None:
            axs = model.plot(
                x=self._wave[pixels_selection], y=self._data[pixels_selection],
                sigma=self._error[pixels_selection], mask=self._mask[pixels_selection], axs=axs)

        return model, params, errors

    def fitParFile(
        self, par, err_sim=0, ftol=1e-8, xtol=1e-8, method="leastsq", parallel="auto"
    ):
        static_par = deepcopy(par)

        if self._error is not None:
            sigma = self._error
        else:
            sigma = 1.0
        par.fit(
            self._wave,
            self._data,
            sigma=sigma,
            err_sim=err_sim,
            maxfev=1000,
            method=method,
            ftol=ftol,
            xtol=xtol,
            parallel=parallel,
        )
        par.restoreResult()
        if err_sim > 0 and self._error is not None:
            par_err = deepcopy(static_par)
            par_err._par = par._par_err
            par_err.restoreResult()
            par._parameters_err = par_err._parameters

    def fitSepGauss(
        self,
        cent_guess,
        aperture,
        fwhm_guess=3,
        bg_guess=0.0,
        flux_range=[0.0, numpy.inf],
        cent_range=[-2.0, 2.0],
        fwhm_range=[0, 7],
        bg_range=[0, numpy.inf],
        badpix_threshold=4,
        ftol=1e-8,
        xtol=1e-8,
        axs=None,
        fit_bg=True
    ):
        # copy main arrays to avoid side effects
        data = self._data.copy()
        error = self._error.copy() if self._error is not None else numpy.ones(self._dim, dtype=numpy.float32)
        mask = self._mask.copy() if self._mask is not None else numpy.zeros(self._dim, dtype=bool)

        # update mask to account for unmasked invalid pixels
        # mask |= (~numpy.isfinite(data) | ~numpy.isfinite(error))

        # reset bad pixels in data and error
        error[mask] = numpy.inf
        data[mask] = 0.0

        flux = numpy.ones(len(cent_guess)) * numpy.nan
        cent = numpy.ones(len(cent_guess)) * numpy.nan
        fwhm = numpy.ones(len(cent_guess)) * numpy.nan
        bg = numpy.ones(len(cent_guess)) * numpy.nan

        fact = numpy.sqrt(2 * numpy.pi)
        hw = aperture // 2
        for i, centre in enumerate(cent_guess):
            if numpy.isnan(centre) or (centre - hw < self._wave[0] or centre + hw > self._wave[-1]):
                continue

            select = (self._wave >= centre - hw) & (self._wave <= centre + hw)
            # print(i, centre, self._wave.min(), self._wave.max(), select.sum())
            if mask[select].sum() >= badpix_threshold:
                warnings.warn(f"skipping line @ {centre:.2f} with {mask[select].sum()} >= {badpix_threshold = } bad pixels")
                self.add_header_comment(f"skipping line @ {centre:.2f} with {mask[select].sum()} >= {badpix_threshold = } bad pixels")
                continue

            flux_guess = numpy.interp(centre, self._wave[select], data[select]) * fact * fwhm_guess / 2.354
            if fit_bg:
                guess = [flux_guess, centre, fwhm_guess / 2.354, bg_guess]
                bound_lower = [flux_range[0], centre+cent_range[0], fwhm_range[0]/2.354, bg_range[0]]
                bound_upper = [flux_range[1], centre+cent_range[1], fwhm_range[1]/2.354, bg_range[1]]
                gauss = fit_profile.Gaussian_const(guess)
            else:
                guess = [flux_guess, centre, fwhm_guess / 2.354]
                gauss = fit_profile.Gaussian(guess)
                bound_lower = [flux_range[0], centre+cent_range[0], fwhm_range[0]/2.354]
                bound_upper = [flux_range[1], centre+cent_range[1], fwhm_range[1]/2.354]

            gauss.fit(
                self._wave[select],
                data[select],
                sigma=error[select],
                p0=guess,
                bounds=(bound_lower, bound_upper),
                ftol=ftol,
                xtol=xtol
            )

            if fit_bg:
                flux[i], cent[i], fwhm[i], bg[i] = gauss.getPar()
            else:
                flux[i], cent[i], fwhm[i] = gauss.getPar()
            fwhm[i] *= 2.354

            if axs is not None:
                select_2 = (self._wave>=cent[i]-3.5*fwhm[i]/2.354) & (self._wave<=cent[i]+3.5*fwhm[i]/2.354)
                x = self._wave[select_2]
                axs[i].plot(self._wave, (select)*numpy.nan+bn.nanmin(data), "ok")
                axs_ = gauss.plot(self._wave[select], self._data[select], mask=self._mask[select], axs={"mod": axs[i]})
                axs[i] = axs_["mod"]
                axs[i].axhline(bg[i], ls="--", color="tab:blue", lw=1)
                axs[i].axvspan(x[0], x[-1], alpha=0.1, fc="0.5", label="reg. of masking")
                axs[i].axvline(cent_guess[i], ls="--", lw=1, color="tab:red", label="cent. guess")
                axs[i].axvline(cent[i], ls="--", lw=1, color="tab:blue", label="cent. model")
                axs[i].set_title(f"{axs[i].get_title()} @ {cent[i]:.1f} {'Angstroms' if self._pixels[0]!=self._wave[0] else 'pixels'}")
                axs[i].text(0.05, 0.9, f"flux = {flux[i]:.2f}", va="bottom", ha="left", transform=axs[i].transAxes, fontsize=11)
                axs[i].text(0.05, 0.8, f"cent = {cent[i]:.2f}", va="bottom", ha="left", transform=axs[i].transAxes, fontsize=11)
                axs[i].text(0.05, 0.7, f"fwhm = {fwhm[i]:.2f}", va="bottom", ha="left", transform=axs[i].transAxes, fontsize=11)
                axs[i].text(0.05, 0.6, f"bg   = {bg[i]:.2f}", va="bottom", ha="left", transform=axs[i].transAxes, fontsize=11)
                axs[i].legend(loc="upper right", frameon=False, fontsize=11)

            # mask line if >=2 pixels are masked within 3.5sigma
            model_badpix = data[select] == 0
            if not numpy.isnan([cent[i], fwhm[i]]).any():
                select_2 = (self._wave>=cent[i]-3.5*fwhm[i]/2.354) & (self._wave<=cent[i]+3.5*fwhm[i]/2.354)
                model_badpix = mask[select_2]
                if model_badpix.sum() >= 2:
                    warnings.warn(f"masking line @ {centre:.2f} with >= 2 masked pixels within a 3.5 sigma window")
                    self.add_header_comment(f"masking line @ {centre:.2f} with >= 2 masked pixels within a 3.5 sigma window")
                    flux[i] = cent[i] = fwhm[i] = bg[i] = numpy.nan

        return flux, cent, fwhm, bg

    def extract_flux(self, centroids, sigmas, fiber_radius=1.4, npixels=20, replace_error=numpy.inf):

        def _gen_mexhat_basis(x, centroids, sigmas, fiber_radius, oversampling_factor):
            dx = x[1, 0] - x[0, 0]
            x_os = fit_profile.oversample(x, oversampling_factor)
            dx_os = dx / oversampling_factor

            x_kernel = numpy.arange(0, 2*fiber_radius + dx_os, dx_os)
            kernel = fit_profile.fiber_profile(centroids=fiber_radius, radii=fiber_radius, x=x_kernel)
            psfs = fit_profile.gaussians((numpy.ones_like(centroids), centroids, sigmas), x_os.T, alpha=2, collapse=False)[0].T

            profiles = signal.fftconvolve(psfs, kernel.T, mode="same", axes=0)
            profiles /= integrate.trapezoid(profiles, x_os, axis=0)[None, :]

            # reshape model into oversampled bins: (x, oversampling_factor)
            profiles_binned = profiles.reshape((x.shape[0], oversampling_factor, x.shape[1]))
            profiles = integrate.trapezoid(profiles_binned, dx=dx_os, axis=1)
            return profiles

        nfibers = centroids.size
        # round up fiber locations
        pixels = numpy.round(centroids[:, None] + numpy.arange(-npixels / 2.0, npixels / 2.0, 1.0)[None, :]).astype("int")
        # defining bad pixels for each fiber if needed
        if self._mask is not None:
            # select: fibers in the boundary of the chip
            mask = numpy.zeros(nfibers, dtype="bool")
            select = bn.nansum(pixels >= self._mask.shape[0], 1)
            nselect = numpy.logical_not(select)
            mask[select] = True

            # masking fibers if all pixels are bad within npixels
            mask[nselect] = bn.nansum(self._mask[pixels[nselect, :]], 1) == npixels
        else:
            mask = None

        # evaluate basis
        xx = numpy.repeat(numpy.arange(nfibers, dtype="int"), 2*npixels+1)
        # pixel ranges of fiber images
        pos_t = numpy.trunc(centroids)
        yyv = numpy.linspace(pos_t-npixels, pos_t+npixels, 2*npixels+1, endpoint=True)

        v = _gen_mexhat_basis(yyv, centroids, sigmas, fiber_radius=fiber_radius, oversampling_factor=100)

        yyv = yyv.T.ravel()
        v = v.T.ravel() / self._error[yyv.astype("int")]

        B = sparse.csc_matrix((v, (yyv, xx)), shape=(len(self._data), nfibers))

        # invert the projection matrix and solve
        ypixels = numpy.arange(self._data.size)
        guess_flux = numpy.interp(centroids, ypixels, self._data) * fit_profile.fact * sigmas
        out = sparse.linalg.lsmr(B, self._data / self._error, atol=1e-3, btol=1e-3, x0=guess_flux)
        flux = out[0]

        error = numpy.sqrt(1 / ((B.multiply(B)).sum(axis=0))).A
        error = error[0,:]
        if mask is not None and bn.nansum(mask) > 0:
            error[mask] = replace_error

        return flux, error, mask

    def obtainGaussFluxPeaks(self, pos, sigma, replace_error=1e10, plot=False):
        """returns Gaussian peaks parameters, flux error and mask

        this runs fiber fitting assuming that we only need to know the sigma of the Gaussian,
        this runs in a full image column

        Parameters
        ----------
        pos : array_like
            peaks positions
        sigma : array_like
            Gaussian widths
        replace_error : float, optional
            replace error in bad pixels with this value, by default 1e10
        plot : bool, optional
            whether to plot or not the results, by default False

        Returns
        -------
        flux : array_like
            measured flux
        error : array_like
            propagated error
        mask : array_like
            propagated pixel mask
        """

        nfibers = len(pos)
        aperture = 3
        # round up fiber locations
        pixels = numpy.round(
            pos[:, None] + numpy.arange(-aperture / 2.0, aperture / 2.0, 1.0)[None, :]
        ).astype("int")
        # defining bad pixels for each fiber if needed
        if self._mask is not None:
            # select: fibers in the boundary of the chip
            bad_pix = numpy.zeros(nfibers, dtype="bool")
            select = bn.nansum(pixels >= self._mask.shape[0], 1)
            nselect = numpy.logical_not(select)
            bad_pix[select] = True

            # masking fibers if all pixels are bad within aperture
            bad_pix[nselect] = bn.nansum(self._mask[pixels[nselect, :]], 1) == aperture
        else:
            bad_pix = None
        if self._error is None:
            self._error = numpy.ones_like(self._data)

        # construct sparse projection matrix
        fact = numpy.sqrt(2.0 * numpy.pi)
        kernel_width = 7 # should exceed 4 sigma
        # vI = []
        # vJ = []
        # vV = []
        # for xx in range(nfibers):
        #     for yy in range(int(pos[xx]-kernel_width),int(pos[xx]+kernel_width)+1):
        #         v = numpy.exp(-0.5 * ((yy-pos[xx]) / sigma[xx]) ** 2) / (fact * sigma[xx])
        #         if v>=0.0000:   # make non-zero and positive definite
        #             vI.append(xx)
        #             vJ.append(yy)
        #             vV.append(v / self._error[yy])
        # B = sparse.csc_matrix((vV, (vJ, vI)), shape=(len(self._data), nfibers))

        # nfibers x kernel_size
        xx = numpy.repeat(numpy.array(range(nfibers)), 2*kernel_width+1)
        # pixel ranges of fiber images
        pos_t = numpy.trunc(pos)
        yyv = numpy.linspace(pos_t-kernel_width, pos_t+kernel_width, 2*kernel_width+1, endpoint=True)
        # nfibers x kernel_size pixel values
        v = numpy.exp(-0.5 * ((yyv-pos) / sigma) ** 2) / (fact * sigma)
        yyv = yyv.T.ravel()
        v = v.T.ravel() / self._error[yyv.astype(numpy.int32)]
        B = sparse.csc_matrix((v, (yyv, xx)), shape=(len(self._data), nfibers))

        # invert the projection matrix and solve
        ypixels = numpy.arange(self._data.size)
        guess_flux = numpy.interp(pos, ypixels, self._data) * fact * sigma
        out = sparse.linalg.lsmr(B, self._data / self._error, atol=1e-3, btol=1e-3, x0=guess_flux)
        flux = out[0]

        error = numpy.sqrt(1 / ((B.multiply(B)).sum(axis=0))).A
        error = error[0,:]
        if bad_pix is not None and bn.nansum(bad_pix) > 0:
            error[bad_pix] = replace_error

        # pyfits.writeto('B.fits', B.toarray(), overwrite=True)
        # if plot:
        #     plt.plot(self._data, "ok")
        #     plt.plot(numpy.dot(A * self._error[:, None], out[0]), "-r")
        #     # plt.plot(numpy.dot(A, out[0]), '-r')
        #     plt.show()
        return flux, error, bad_pix

    def collapseSpec(self, method="mean", start=None, end=None, transmission_func=None):
        if start is not None:
            select_start = self._wave >= start
        else:
            select_start = numpy.ones(self._dim, dtype="bool")
        if end is not None:
            select_end = self._wave <= end
        else:
            select_end = numpy.ones(self._dim, dtype="bool")
        select = numpy.logical_and(select_start, select_end)
        if self._mask is not None:
            select = numpy.logical_and(select, numpy.logical_not(self._mask))

        if method != "mean" and method != "median" and method != "sum":
            raise ValueError("method must be either 'mean', 'median' or 'sum'")
        elif method == "mean":
            flux = bn.mean(self._data[select])
            if self._error is not None:
                error = numpy.sqrt(
                    numpy.sum(self._error[select] ** 2) / numpy.sum(select) ** 2
                )
            else:
                error = None
            if self._sky is not None:
                sky = bn.mean(self._sky[select])
            else:
                sky = None
            if self._sky_error is not None:
                sky_error = numpy.sqrt(
                    numpy.sum(self._sky_error[select] ** 2) / numpy.sum(select) ** 2
                )
            else:
                sky_error = None
        return flux, error, sky, sky_error

    def coaddSpec(self, other, wave=None):
        """Coadds spectrum with another one with the possibility of overlaping wavelength ranges

        This method is perfect for computing the joint spectrum from two spectrograph channels.

        NOTE: taken from https://bit.ly/3qpRFIp
        """
        # check if other is Spectrum1D instance
        # find best/optimal joint wavelength vector
        if isinstance(other, Spectrum1D):
            spectra = [self, other]
        else:
            raise NotImplementedError("'other' need to be of 'Spectrum1D' type")

        if wave is None:
            wave = wave_little_interpol([self._wave, other._wave])

        fluxes = numpy.ma.zeros((2, len(wave)))
        errors = numpy.zeros_like(fluxes)
        fwhms = numpy.zeros_like(fluxes)
        masks = numpy.zeros_like(fluxes, dtype=bool)
        skies = numpy.zeros_like(fluxes)
        sky_errors = numpy.zeros_like(fluxes)
        for i, s in enumerate(spectra):
            s_new = s.resampleSpec(wave, method="linear")
            fluxes[i, :] = s_new._data
            if s._error is None:
                raise ValueError("s.uncertainty needs to be set for all spectra")
            else:
                errors[i, :] = s_new._error

            if s._lsf is not None:
                fwhms[i, :] = s_new._lsf
            if s._mask is not None:
                masks[i, :] = s_new._mask
            if s._sky is not None:
                skies[i, :] = s_new._sky
            if s._sky_error is not None:
                sky_errors[i, :] = s_new._sky_error

        fluxes[masks] = numpy.nan
        errors[masks] = numpy.nan
        fwhms[masks] = numpy.nan
        skies[masks] = numpy.nan
        sky_errors[masks] = numpy.nan

        # First, make sure there is no flux defined if there is no error.
        # errors = numpy.ma.fix_invalid(errors)
        # if numpy.ma.is_masked(errors):
        #     fluxes[errors.mask] = numpy.ma.masked
        # This can be simplified considerably as soon as masked quantities exist.
        # fluxes = numpy.ma.fix_invalid(fluxes)
        # There are no masked quantities yet, so make sure they are filled here.
        weights = 1.0 / errors**2
        norm = bn.nansum(weights, axis=0)
        weights = weights / norm[None, :]
        fluxes = bn.nansum(fluxes * weights, axis=0)
        fwhms = bn.nansum(fwhms * weights, axis=0)
        errors = numpy.sqrt(1.0 / bn.nansum(weights * norm, axis=0))
        skies = bn.nansum(skies * weights, axis=0)
        sky_errors = numpy.sqrt(bn.nansum(sky_errors**2 * weights**2), axis=0)

        masks = numpy.logical_and(masks[0], masks[1])
        masks = numpy.logical_or(masks, numpy.isnan(fluxes))
        masks = numpy.logical_or(masks, numpy.isnan(fwhms))
        masks = numpy.logical_or(masks, numpy.isnan(errors))
        masks = numpy.logical_or(masks, numpy.isnan(skies))
        masks = numpy.logical_or(masks, numpy.isnan(sky_errors))

        return Spectrum1D(wave=wave, data=fluxes, error=errors, lsf=fwhms, mask=masks, sky=skies, sky_error=sky_errors)

    def fit_lines(self, cwaves, dwave=8, axs=None):

        cwaves_ = numpy.atleast_1d(cwaves)

        if self._lsf is None:
            fwhm_guess = 2.5
        else:
            fwhm_guess = numpy.nanmean(numpy.interp(cwaves_, self._wave, self._lsf))

        if axs is not None:
            axs = numpy.atleast_1d(axs)
        flux, sky_wave, fwhm, bg = self.fitSepGauss(cwaves_, dwave,
                                                    fwhm_guess, 0.0,
                                                    [0, numpy.inf],
                                                    [-2.5, 2.5],
                                                    [max(fwhm_guess - 1.5, 0), fwhm_guess + 1.5],
                                                    [0.0, numpy.inf],
                                                    axs=axs)
        return flux, sky_wave, fwhm, bg
