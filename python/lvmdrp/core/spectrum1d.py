from copy import deepcopy

import matplotlib.pyplot as plt
import numpy
import bottleneck as bn
from astropy.io import fits as pyfits
from numpy import polynomial
from scipy.linalg import norm
from scipy import signal, interpolate, ndimage, sparse, linalg
from scipy.ndimage import zoom
from typing import List, Tuple

from lvmdrp.utils import gaussian
from lvmdrp.core import fit_profile
from lvmdrp.core.header import Header


def _spec_from_lines(lines: numpy.ndarray, sigma: float, wavelength: numpy.ndarray, heights: numpy.ndarray = None, names: numpy.ndarray = None):
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
    """Find the best cross correlation between two spectra.

    This function finds the best cross correlation between two spectra by
    stretching and shifting the first spectrum and computing the cross
    correlation with the second spectrum. The best cross correlation is
    defined as the one with the highest correlation value and the correct
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
            numpy.mean(numpy.diff(wavelist[i]))
            + numpy.mean(numpy.diff(wavelist[i + 1]))
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


class Spectrum1D(Header):
    def __init__(
        self, wave=None, data=None, error=None, mask=None, inst_fwhm=None, header=None
    ):
        self._wave = wave
        self._data = data
        if data is not None:
            self._dim = self._data.shape[0]
            self._pixels = numpy.arange(self._dim)
        self._error = error
        self._mask = mask
        self._inst_fwhm = inst_fwhm
        self._header = header

    def __sub__(self, other):
        if isinstance(other, Spectrum1D):
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
            else:
                error = None
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

        elif isinstance(other, numpy.ndarray):
            data = self._data - other
            if self._error is not None:
                error = self._error
            else:
                error = None
            mask = self._mask
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data - other
                if self._error is not None:
                    error = self._error
                else:
                    error = None
                mask = self._mask
                if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                    data = data.astype(numpy.float32)
                if error is not None:
                    if error.dtype == numpy.float64 or error.dtype == numpy.dtype(
                        ">f8"
                    ):
                        error = error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
                return spec
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for -: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __add__(self, other):
        if isinstance(other, Spectrum1D):
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
            else:
                error = None
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

        elif isinstance(other, numpy.ndarray):
            data = self._data + other
            if self._error is not None:
                error = self._error
            else:
                error = None
            mask = self._mask
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data + other
                if self._error is not None:
                    error = self._error
                else:
                    error = None
                mask = self._mask
                if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                    data = data.astype(numpy.float32)
                if error is not None:
                    if error.dtype == numpy.float64 or error.dtype == numpy.dtype(
                        ">f8"
                    ):
                        error = error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
                return spec
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for -: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __truediv__(self, other):
        if isinstance(other, Spectrum1D):
            other._data = other._data.astype(numpy.float32)
            select = other._data != 0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select) > 0:
                data[select] = self._data[select] / other._data[select].astype(
                    numpy.float32
                )

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask = self._mask
            else:
                mask = None
            if self._error is not None and other._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select) > 0:
                    error[select] = numpy.sqrt(
                        (self._error[select] / other._data[select]) ** 2
                        + (
                            self._data[select]
                            * other._error[select]
                            / other._data[select] ** 2
                        )
                        ** 2
                    )
            elif self._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select) > 0:
                    error[select] = self._error[select] / other._data[select]
                    error[numpy.logical_not(select)] = numpy.max(self._error)
            else:
                error = None
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

        elif isinstance(other, numpy.ndarray):
            if other != 0:
                data = self._data / other
                if self._error is not None:
                    error = self._error / other
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                if other != 0.0:
                    data = self._data / other
                    if self._error is not None:
                        error = self._error / other
                    else:
                        error = None
                    mask = self._mask
                else:
                    data = numpy.zeros_like(self._data)
                    if self._error is not None:
                        error = numpy.zeros_like(self._data)
                    else:
                        error = None
                    if self._mask is not None:
                        mask = numpy.zeros(self._data.shape[0], dtype="bool")
                    else:
                        mask = None
                if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                    data = data.astype(numpy.float32)
                if error is not None:
                    if error.dtype == numpy.float64 or error.dtype == numpy.dtype(
                        ">f8"
                    ):
                        error = error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
                return spec
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for /: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __rtruediv__(self, other):
        if isinstance(other, Spectrum1D):
            other._data = other._data.astype(numpy.float32)
            select = self._data != 0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select) > 0:
                data[select] = (
                    other._data[select].astype(numpy.float32) / self._data[select]
                )

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask = self._mask
            else:
                mask = None
            if self._error is not None and other._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select) > 0:
                    error[select] = numpy.sqrt(
                        (other._error[select] / self._data[select]) ** 2
                        + (
                            other._data[select]
                            * self._error[select]
                            / self._data[select] ** 2
                        )
                        ** 2
                    )
            elif self._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select) > 0:
                    error[select] = other._error[select] / self._data[select]
                    error[numpy.logical_not(select)] = numpy.max(self._error)
            else:
                error = None
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

        elif isinstance(other, numpy.ndarray):
            select = self._data != 0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select) > 0:
                data[select] = other[select] / self._data[select]
                if self._error is not None:
                    error = numpy.zeros_like(self._error)
                    if numpy.sum(select) > 0:
                        error[select] = (
                            other[select]
                            * self._error[select]
                            / self._data[select] ** 2
                        )
                    else:
                        error = None
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec
        else:
            select = self._data != 0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select) > 0:
                data[select] = other / self._data[select]
                if self._error is not None:
                    error = numpy.zeros_like(self._error)
                    if numpy.sum(select) > 0:
                        error[select] = (
                            other * self._error[select] / self._data[select] ** 2
                        )
                    else:
                        error = None
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                else:
                    error = None
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None

            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

    def __mul__(self, other):
        if isinstance(other, Spectrum1D):
            other._data.astype(numpy.float32)
            data = self._data * other._data.astype(numpy.float32)

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is not None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask = self._mask
            else:
                mask = None
            if self._error is not None and other._error is not None:
                error = numpy.sqrt(
                    (self._error * other._data + self._data * other._error) ** 2
                )
            elif self._error is not None and other._error is None:
                error = self._error * other._data
            else:
                error = None
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

        elif isinstance(other, numpy.ndarray):
            if other != 0:
                data = self._data * other
                if self._error is not None:
                    error = self._error * other
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)
            if error is not None:
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            #  try:
            data = self._data * other
            if self._error is not None:
                error = self._error * other
                if error.dtype == numpy.float64 or error.dtype == numpy.dtype(">f8"):
                    error = error.astype(numpy.float32)
            else:
                error = None
            mask = self._mask
            if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
                data = data.astype(numpy.float32)

            spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
            return spec

    def __pow__(self, other):
        data = self._data**other
        if self._error is not None:
            error = 1.0 / float(other) * self._data ** (other - 1) * self._error
        else:
            error = None
        mask = self._mask

        if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
            data = data.astype(numpy.float32)
        spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)
        return spec

    def __rpow__(self, other):
        data = other**self._data
        error = None
        mask = self._mask

        if data.dtype == numpy.float64 or data.dtype == numpy.dtype(">f8"):
            data = data.astype(numpy.float32)
        spec = Spectrum1D(wave=self._wave, data=data, error=error, mask=mask)

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
        return self._data() > other

    def __ge__(self, other):
        return self._data() >= other

    def loadFitsData(
        self,
        file,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_wave=None,
        extension_fwhm=None,
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
            and extension_fwhm is None is None
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
                        self._inst_fwhm = hdu[i].data
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
                self._inst_fwhm = hdu[i].data
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
        if self._inst_fwhm is not None:
            self._inst_fwhm = self._inst_fwhm.astype("float32")

        hdus = [None, None, None, None, None]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if (
            extension_data is None
            and extension_error is None
            and extension_mask is None
            and extension_fwhm is None
            and extension_hdr is None
        ):
            hdus[0] = pyfits.PrimaryHDU(self._data, header=self._header)
            if self._wave is not None:
                hdus[1] = pyfits.ImageHDU(self._wave, name="WAVE")
            if self._inst_fwhm is not None:
                hdus[2] = pyfits.ImageHDU(self._inst_fwhm, name="INSTFWHM")
            if self._error is not None:
                hdus[3] = pyfits.ImageHDU(self._error, name="ERROR")
            if self._mask is not None:
                hdus[4] = pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX")
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
                hdu = pyfits.PrimaryHDU(self._inst_fwhm)
            elif extension_fwhm > 0 and extension_fwhm is not None:
                hdus[extension_fwhm] = pyfits.ImageHDU(self._inst_fwhm, name="INSTFWHM")

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
        max = numpy.nanmax(self._data)  # get max
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
        min = numpy.nanmin(self._data)  # get min
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
        """

        return self._pixels, self._wave, self._data, self._error, self._mask

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

        # case where input spectrum has more than half the pixels masked
        if numpy.nansum(self._data) == 0.0 or (
            self._mask is not None and numpy.sum(self._mask) > self._dim / 2
        ):
            # all pixels masked
            new_mask = numpy.ones(len(ref_wave), dtype=bool)
            # all data points to zero
            new_data = numpy.zeros(len(ref_wave), numpy.float32)
            # all LSF pixels zero (if present)
            if self._inst_fwhm is not None:
                new_inst_fwhm = numpy.zeros(len(ref_wave), numpy.float32)
            else:
                new_inst_fwhm = None
            # all error pixels replaced with replace_error
            if self._error is None or err_sim == 0:
                new_error = None
            else:
                new_error = numpy.ones(len(ref_wave), numpy.float32) * replace_error

            # return masked spectrum
            return Spectrum1D(
                data=new_data,
                wave=ref_wave,
                error=new_error,
                mask=new_mask,
                inst_fwhm=new_inst_fwhm,
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
            if self._inst_fwhm is not None:
                intp = interpolate.interp1d(
                    self._wave[select_goodpix],
                    self._inst_fwhm[select_goodpix],
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=(0.0, 0.0),
                )
                clean_inst_fwhm = intp(self._wave)

                select_interp = clean_inst_fwhm != 0
                # wave_interp = self._wave[select_interp]
                # perform the interpolation on the data
                if method == "spline":
                    intp = interpolate.UnivariateSpline(
                        self._wave[select_interp],
                        clean_inst_fwhm[select_interp],
                        s=0,
                        ext="zeros",
                    )
                    new_inst_fwhm = intp(ref_wave)
                elif method == "linear":
                    intp = interpolate.interp1d(
                        self._wave[select_interp],
                        clean_inst_fwhm[select_interp],
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=(0.0, 0.0),
                    )
                    new_inst_fwhm = intp(ref_wave)
            else:
                new_inst_fwhm = None

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
                        clean_data[select_goodpix], numpy.abs(self._error[select_goodpix])
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

        if extrapolate is not None:
            select_out = numpy.logical_or(
                ref_wave < self._wave[0], ref_wave > self._wave[-1]
            )
            new_data = numpy.where(select_out, extrapolate._data, new_data)
            new_mask = numpy.where(select_out, extrapolate._mask, new_mask)
            if new_error is not None:
                new_error = numpy.where(select_out, extrapolate._error, new_error)
            if new_inst_fwhm is not None:
                new_inst_fwhm = numpy.where(
                    select_out, extrapolate._inst_fwhm, new_inst_fwhm
                )

        spec_out = Spectrum1D(
            wave=ref_wave,
            data=new_data,
            error=new_error,
            mask=new_mask,
            inst_fwhm=new_inst_fwhm,
        )
        return spec_out

    def resampleSpec_flux_conserving(self, ref_wave, method="spline",
        err_sim=500,
        replace_error=1e10,
        extrapolate=None):

        old_dlambda = numpy.interp(ref_wave, self._wave[:-1], numpy.diff(self._wave))

        # plt.plot(self._wave, self._data, lw=1, color="k")
        # plt.plot(self._wave, )

        new_dlambda = numpy.diff(ref_wave, append=ref_wave[-1])
        new_spec = self.resampleSpec(ref_wave, method=method, err_sim=err_sim, replace_error=replace_error, extrapolate=extrapolate)
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

    def matchFWHM(self, target_fwhm, inplace=False):
        if self._inst_fwhm is not None:
            if self._mask is not None:
                good_pix = numpy.logical_not(self._mask)
                data = self._data[good_pix]
                wave = self._wave[good_pix]
                fwhm = self._inst_fwhm[good_pix]
                if self._error is not None:
                    error = self._error[good_pix]
                else:
                    error = None
            else:
                data = self._data
                wave = self._wave
                fwhm = self._inst_fwhm
                error = self._error

            if inplace:
                new_spec = self
            else:
                new_spec = deepcopy(self)

            gauss_sig = numpy.zeros_like(fwhm)
            select = target_fwhm > fwhm
            gauss_sig[select] = numpy.sqrt(target_fwhm**2 - fwhm[select] ** 2) / 2.354
            fact = numpy.sqrt(2.0 * numpy.pi)
            kernel = numpy.exp(
                -0.5
                * (
                    (wave[:, numpy.newaxis] - wave[numpy.newaxis, :])
                    / gauss_sig[numpy.newaxis, :]
                )
                ** 2
            ) / (fact * gauss_sig[numpy.newaxis, :])
            multiplied = data[:, numpy.newaxis] * kernel
            new_data = numpy.sum(multiplied, axis=0) / numpy.sum(kernel, 0)
            if new_spec._mask is not None:
                new_spec._data[good_pix] = new_data
                new_spec._inst_fwhm[:] = target_fwhm
            if error is not None:
                new_error = numpy.sqrt(
                    numpy.sum((error[:, numpy.newaxis] * kernel) ** 2, axis=0)
                ) / numpy.sum(kernel, 0)
                if new_spec._mask is not None:
                    new_spec._error[good_pix] = new_error
                else:
                    new_spec._error = new_error

            return new_spec

    def binSpec(self, new_wave):
        new_disp = new_wave[1:] - new_wave[:-1]
        new_disp = numpy.insert(new_disp, 0, new_disp[0])
        data_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
        mask_out = numpy.zeros(len(new_wave), dtype="bool")
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
        bound_min = new_wave - new_disp / 2.0
        bound_max = new_wave + new_disp / 2.0

        # disp = bound_max - bound_min
        for i in range(len(new_wave)):
            select = numpy.logical_and(
                masked_wave >= bound_min[i], masked_wave <= bound_max[i]
            )
            if numpy.sum(select) > 0:
                #    data_out[i] = numpy.mean(self._data[mask_in][select])
                data_out[i] = numpy.sum(
                    numpy.abs(masked_wave[select] - new_wave[i])
                    * self._data[mask_in][select]
                ) / numpy.sum(numpy.abs(masked_wave[select] - new_wave[i]))
                if self._error is not None:
                    error_out[i] = numpy.sqrt(
                        numpy.sum(masked_error[select] ** 2) / numpy.sum(select) ** 2
                    )
            else:
                data_out[i] = 0.0
                mask_out[i] = True
        data_out = numpy.interp(new_wave, masked_wave, self._data[mask_in])
        #    numpy.delete(masked_wave, select)
        #    numpy.delete(masked_data,  select)
        #    numpy.delete(masked_error, select)
        spec = Spectrum1D(data=data_out, wave=new_wave, error=error_out, mask=mask_out)
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
                self._wave,
                self._data,
                w=1.0 / numpy.sqrt(numpy.fabs(self._data)),
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

        if self._inst_fwhm is not None:
            inst_fwhm = numpy.sqrt(self._inst_fwhm**2 + diff_fwhm**2)
        else:
            inst_fwhm = diff_fwhm

        spec = Spectrum1D(
            wave=self._wave,
            data=data,
            error=error,
            mask=self._mask,
            inst_fwhm=inst_fwhm,
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
            out_par = poly.coef

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

    def measurePeaks(
        self, init_pos, method="gauss", init_sigma=1.0, threshold=0, max_diff=0
    ):
        """
        Find the subpixel centre for the local maxima in a Spectrum.

        Parameters
        --------------
        init_pos : numpy.ndarray
            Initial guess for the peak centres

        method : string, optional with default='gauss'
            Select the method to measure the peaks, either 'gauss' or 'hyperbolic'.
            The first one fits a Gaussian to the 3 brightest pixels around each peak,
            the second one uses a hyperbolic approximation to the 3 brightest pixels around each peak.

        init_sigma: float, optional with default=1.0
            Initial guess of the Gaussian width used for the modelling.
            Only used with the method 'gauss'

        threshold: float, optional with default = 0
            It defines the contrast between the minmum of maximum values for the
            3 brightest pixel around a peak for which an estimated centre is assumed to be valid.

        max_diff: float, optional with default = 0
            If greater than zero, all peak centres which are different from the initial guess position by
            this value are assumed to be invalid.

        Returns (positions, mask)
        -----------
        positions :  numpy.ndarray (float)
            Array of subpixel peaks positions
        mask : numpy.ndarray (bool)
            Array of pixels with uncertain measurements
        """
        # compute the minimum and maximum value for the 3 pixels around all peaks
        # selection of fibers within the boundaries of the detector
        select = numpy.logical_and(
            init_pos - 1 >= [0], init_pos + 1 <= self._data.shape[0] - 1
        )
        mask = numpy.zeros(len(init_pos), dtype="bool")
        # minimum counts of three pixels around each peak
        min = numpy.amin(
            [
                numpy.take(self._data, init_pos[select] + 1),
                numpy.take(self._data, init_pos[select]),
                numpy.take(self._data, init_pos[select] - 1),
            ],
            axis=0,
        )
        # minimum counts of three pixels around each peak
        max = numpy.amax(
            [
                numpy.take(self._data, init_pos[select] + 1),
                numpy.take(self._data, init_pos[select]),
                numpy.take(self._data, init_pos[select] - 1),
            ],
            axis=0,
        )
        # print(init_pos, max)
        # mask all peaks where the contrast between maximum and minimum is below a threshold
        mask[select] = (max) < threshold
        # masking fibers outside the detector
        mask[numpy.logical_not(select)] = True

        if method == "hyperbolic":
            # compute the subpixel peak position using the hyperbolic
            d = (
                numpy.take(self._data, init_pos + 1)
                - 2 * numpy.take(self._data, init_pos)
                + numpy.take(self._data, init_pos - 1)
            )
            positions = (
                init_pos
                + 1
                - (
                    (
                        numpy.take(self._data, init_pos + 1)
                        - numpy.take(self._data, init_pos)
                    )
                    / d
                    + 0.5
                )
            )

        elif method == "gauss":
            # compute the subpixel peak position by fitting a gaussian to all peaks (3 pixel to get a unique solution
            positions = numpy.zeros(
                len(init_pos), dtype="float32"
            )  # create empty array
            for j in range(len(init_pos)):
                # only pixels with enough contrast are fitted
                if not mask[j]:
                    gauss = fit_profile.Gaussian(
                        [
                            self._data[init_pos[j]] * numpy.sqrt(2 * numpy.pi),
                            init_pos[j],
                            init_sigma,
                        ]
                    )  # set initial parameters for Gaussian profile

                    gauss.fit(
                        self._pixels[init_pos[j] - 1 : init_pos[j] + 2],
                        self._data[init_pos[j] - 1 : init_pos[j] + 2],
                        warning=False,
                    )  # perform fitting
                    positions[j] = gauss.getPar()[1]

        mask = numpy.logical_or(
            mask, numpy.isnan(positions)
        )  # masked all corrupt subpixel peak positions

        if max_diff != 0:
            # mask all pixels that are away from the initial guess of peak positions by a certain difference
            mask = numpy.logical_or(
                numpy.logical_or(
                    positions > init_pos + max_diff, positions < init_pos - max_diff
                ),
                mask,
            )

        if numpy.sum(mask) > 0:
            # replace the estimated position of all masekd peak position by the corresponding initial guess peak positions
            positions[mask] = init_pos[mask].astype("float32")
        return positions, mask

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
            median_dist = numpy.nanmedian(
                pos_block[1:] - pos_block[:-1]
            )  # compute median distance between peaks
            flux = (
                self._data[numpy.round(pos_block).astype("int16")]
                * numpy.sqrt(2)
                * init_fwhm
                / 2.354
            )  # initial guess for the flux

            # compute lower and upper bounds of the positions for each block
            lo = int(numpy.nanmin(pos_block) - median_dist)
            if lo <= 0:
                lo = 0
            hi = int(numpy.nanmax(pos_block) + median_dist)
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
                median_dist = numpy.median(
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
                med_pos[i] = numpy.mean(self._wave[lo:hi])
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
                median_dist = numpy.median(
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
                med_pos[i] = numpy.mean(self._wave[lo:hi])
            else:
                offsets[i] = 0.0
                med_pos[i] = 0.0
        return offsets, med_pos

    def fitMultiGauss(self, centres, init_fwhm):
        select = numpy.zeros(self._dim, dtype="bool")
        flux_in = numpy.zeros(len(centres), dtype=numpy.float32)
        sig_in = numpy.ones_like(flux_in) * init_fwhm / 2.354
        cent = numpy.zeros(len(centres), dtype=numpy.float32)
        if self._error is not None:
            error = self._error
        else:
            error = numpy.ones_like(self._dim, dtype=numpy.float32)
        for i in range(len(centres)):
            select_line = numpy.logical_and(
                self._wave > centres[i] - 2 * init_fwhm,
                self._wave < centres[i] + 2 * init_fwhm,
            )
            flux_in[i] = numpy.sum(self._data[select_line])
            select = numpy.logical_or(select, select_line)
            cent[i] = centres[i]
        par = numpy.concatenate([flux_in, cent, sig_in])
        gauss_multi = fit_profile.Gaussians(par)
        gauss_multi.fit(self._wave[select], self._data[select], sigma=error[select])
        return gauss_multi, gauss_multi.getPar()

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
        centres,
        aperture,
        init_back=0.0,
        ftol=1e-8,
        xtol=1e-8,
        axs=None,
        warning=False,
    ):
        ncomp = len(centres)

        out = numpy.zeros(3 * ncomp, dtype=numpy.float32)
        back = [deepcopy(init_back) for _ in centres]

        error = self._error if self._error is not None else numpy.ones(self._dim, dtype=numpy.float32)
        mask = self._mask if self._mask is not None else numpy.zeros(self._dim, dtype=bool)

        for i, centre in enumerate(centres):
            select = self._get_select(centre, aperture, mask)
            if numpy.sum(select) > 0:
                max = numpy.max(self._data[select])
                cent = numpy.median(self._wave[select][self._data[select] == max])
                select = self._get_select(cent, aperture, mask)

                gauss = self._fit_gaussian(select, back[i], error, ftol, xtol, warning)

                out_fit = gauss.getPar()
                out[i] = out_fit[0]
                out[ncomp + i] = out_fit[1]
                out[2 * ncomp + i] = out_fit[2]

                if axs is not None:
                    axs[i] = gauss.plot(self._wave[select], self._data[select], ax=axs[i])
                    axs[i].axvline(centres[i], ls="--", lw=1, color="tab:red")
            else:
                out[i:ncomp + i + 1] = 0.0

        return out

    def _get_select(self, centre, aperture, mask):
        return numpy.logical_and(
            numpy.logical_and(
                self._wave >= centre - aperture / 2.0,
                self._wave <= centre + aperture / 2.0,
            ),
            numpy.logical_not(mask),
        )

    def _fit_gaussian(self, select, back, error, ftol, xtol, warning):
        if back == 0.0:
            par = [0.0, 0.0, 0.0]
            gauss = fit_profile.Gaussian(par)
        else:
            par = [0.0, 0.0, 0.0, 0.0]
            gauss = fit_profile.Gaussian_const(par)

        gauss.fit(
            self._wave[select],
            self._data[select],
            sigma=error[select],
            ftol=ftol,
            xtol=xtol,
            warning=warning,
        )

        return gauss

    def obtainGaussFluxPeaks(self, pos, sigma, indices, replace_error=1e10, plot=False):
        """returns Gaussian peaks parameters, flux error and mask

        this runs fiber fitting assuming that we only need to know the sigma of the Gaussian,
        this runs in a full image column

        Parameters
        ----------
        pos : array_like
            peaks positions
        sigma : array_like
            Gaussian widths
        indices : array_like
            peaks indices
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

        fibers = len(pos)
        aperture = 3
        # round up fiber locations
        pixels = numpy.round(
            pos[:, None] + numpy.arange(-aperture / 2.0, aperture / 2.0, 1.0)[None, :]
        ).astype("int")
        # defining bad pixels for each fiber if needed
        if self._mask is not None:
            bad_pix = numpy.zeros(fibers, dtype="bool")
            select = bn.nansum(pixels >= self._mask.shape[0], 1)
            nselect = numpy.logical_not(select)
            bad_pix[select] = True
            bad_pix[nselect] = bn.nansum(self._mask[pixels[nselect, :]], 1) == aperture
        else:
            bad_pix = None
        if self._error is None:
            self._error = numpy.ones_like(self._data)

        fact = numpy.sqrt(2.0 * numpy.pi)
        A = (
            1.0
            * numpy.exp(
                -0.5 * ((self._wave[:, None] - pos[None, :]) / sigma[None, :]) ** 2
            )
            / (fact * sigma[None, :])
        )
        # making positive definite
        select = A > 0.0001
        A = A / self._error[:, None]

        # plt.figure(figsize=(10, 10))
        # plt.imshow(A, origin="lower")
        # plt.show()

        B = sparse.csr_matrix(
            (A[select], (indices[0][select], indices[1][select])),
            shape=(self._dim, fibers),
        ).todense()
        # print(B)
        out = sparse.linalg.lsmr(
            B, self._data / self._error, atol=1e-4, btol=1e-4
        )
        # out = linalg.lstsq(A, self._data / self._error, lapack_driver='gelsy', check_finite=False)
        # print(out)

        error = numpy.sqrt(1 / bn.nansum((A**2), 0))
        if bad_pix is not None and bn.nansum(bad_pix) > 0:
            error[bad_pix] = replace_error
        # if plot:
        #     plt.figure(figsize=(15, 10))
        #     plt.plot(self._data, "ok")
        #     plt.plot(numpy.dot(A * self._error[:, None], out[0]), "-r")
        #     # plt.plot(numpy.dot(A, out[0]), '-r')
        #     plt.show()
        return out[0], error, bad_pix, B, A

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
            pass
        elif method == "mean":
            flux = numpy.mean(self._data[select])
            if self._error is not None:
                error = numpy.sqrt(
                    numpy.sum(self._error[select] ** 2) / numpy.sum(select) ** 2
                )
            else:
                error = 0
        return flux, error

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
        for i, s in enumerate(spectra):
            s_new = s.resampleSpec(wave, method="linear")
            fluxes[i, :] = s_new._data
            if s._error is None:
                raise ValueError("s.uncertainty needs to be set for all spectra")
            else:
                errors[i, :] = s_new._error

            if s._inst_fwhm is not None:
                fwhms[i, :] = s_new._inst_fwhm
            if s._mask is not None:
                masks[i, :] = s_new._mask

        fluxes[masks] = numpy.nan
        errors[masks] = numpy.nan
        fwhms[masks] = numpy.nan

        # First, make sure there is no flux defined if there is no error.
        # errors = numpy.ma.fix_invalid(errors)
        # if numpy.ma.is_masked(errors):
        #     fluxes[errors.mask] = numpy.ma.masked
        # This can be simplified considerably as soon as masked quantities exist.
        # fluxes = numpy.ma.fix_invalid(fluxes)
        # There are no masked quantities yet, so make sure they are filled here.
        weights = 1.0 / errors**2
        norm = bn.nansum(weights, axis=0)
        weights = weights / norm[None,:]
        fluxes = bn.nansum(fluxes * weights, axis=0)
        fwhms = bn.nansum(fwhms * weights, axis=0)
        errors = numpy.sqrt(1.0 / bn.nansum(weights * norm, axis=0))

        masks = numpy.logical_and(masks[0], masks[1])
        masks = numpy.logical_or(masks, numpy.isnan(fluxes))
        masks = numpy.logical_or(masks, numpy.isnan(fwhms))
        masks = numpy.logical_or(masks, numpy.isnan(errors))

        return Spectrum1D(
            wave=wave, data=fluxes, error=errors, inst_fwhm=fwhms, mask=masks
        )
