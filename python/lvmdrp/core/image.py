from copy import deepcopy as copy
from multiprocessing import Pool, cpu_count
import warnings

from functools import partial
from typing import List
from tqdm import tqdm

import os
import numpy
import bottleneck as bn
from astropy.table import Table
from astropy.io import fits as pyfits
from astropy.modeling import fitting, models
from astropy.stats.biweight import biweight_location, biweight_scale
from scipy import ndimage, signal
from scipy import interpolate

from lvmdrp import log
from lvmdrp.core.constants import CON_LAMPS, ARC_LAMPS
from lvmdrp.core.plot import plt
from lvmdrp.core.fit_profile import gaussians, Gaussians
from lvmdrp.core.apertures import Apertures
from lvmdrp.core.header import Header
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.spectrum1d import Spectrum1D, _cross_match_float, _cross_match, _spec_from_lines

from lvmdrp import __version__ as drpver

def _fill_column_list(columns, width):
    """Adds # width columns around the given columns list

    Parameters
    ----------
    columns : list
        list of columns to add width
    width : int
        number of columns to add around the given columns

    Returns
    -------
    list
        list of columns with width
    """
    new_columns = []
    for icol in columns:
        new_columns.extend(range(icol - width, icol + width))
    return new_columns


def _parse_ccd_section(section):
    """Parse a CCD section in the format [1:NCOL, 1:NROW] to python tuples"""
    slice_x, slice_y = section.strip("[]").split(",")
    slice_x = list(map(lambda str: int(str), slice_x.split(":")))
    slice_y = list(map(lambda str: int(str), slice_y.split(":")))
    slice_x[0] -= 1
    slice_y[0] -= 1
    return slice_x, slice_y


def _zscore(x, axis=1):
    """computes the zscore of a given array along a given axis"""
    if axis == 0:
        zscore = (x - biweight_location(x, axis=axis, ignore_nan=True)[None, :]) / biweight_scale(x, axis=axis, ignore_nan=True)[None, :]
    elif axis == 1:
        zscore = (x - biweight_location(x, axis=axis, ignore_nan=True)[:, None]) / biweight_scale(x, axis=axis, ignore_nan=True)[:, None]
    else:
        raise ValueError("axis must be 0 or 1")
    return zscore


def _model_overscan(os_quad, axis=1, overscan_stat="biweight", threshold=None, model="spline", **kwargs):
    """fits a parametric model to the given overscan region

    Given an overscan section corresponding to a quadrant in a raw frame, this function
    coadds the counts along a given `axis` using a given statistics `stat`. Additionally,
    a model can be fitted to the resulting profile, which options are:
        * const: a constant model by further collapsing along using the same `stat`
        * profile: the raw profile (`os_model = os_profile`)
        * polynomial: a polynomial model fitted on `os_profile`
        * spline: a cubic-spline fitting on `os_profile`

    if `threshold` is given, pixels in the overscan region will be masked if
    above `threshold` standard deviations from the mean.

    Additional keyword parameters are passed to the fitted model.

    Parameters
    ----------
    os_quad : lvmdrp.core.image.Image
        image section corresponding to a overscan quadrant
    axis : int, optional
        axis along which the overscan will be fitted, by default 1
    overscan_stat : str, optional
        function name to use for coadding pixels along `axis`, by default "biweight"
    threshold : float, optional
        threshold to mask columns in the overscan region, by default None
    model : str, optional
        parametric function to fit ("const", "profile", "poly", "spline"), by default "spline"

    Returns
    -------
    os_profile : array_like
        overscan profile after coadding pixels along `axis`
    os_model : array_like, float
        overscan model
    """
    assert axis == 0 or axis == 1

    if overscan_stat == "biweight":
        stat = partial(biweight_location, ignore_nan=True)
    elif overscan_stat == "median":
        stat = numpy.nanmedian
    else:
        warnings.warn(
            f"overscan statistic '{overscan_stat}' not implemented, "
            "falling back to 'biweight'"
        )
        stat = partial(biweight_location, ignore_nan=True)

    if threshold is not None:
        os_data = os_quad._data
        os_zscore = _zscore(os_data, axis=1)
        mask = numpy.abs(os_zscore) > threshold
        # reject the whole column if more than 30% of the pixels are masked
        mask_columns = mask.sum(axis=0) > 0.3 * os_data.shape[0]
        if mask_columns.any():
            mask[:, mask_columns] = True
        os_data[mask] = numpy.nan
    else:
        os_data = os_quad._data

    os_profile = stat(os_data, axis=axis)
    pixels = numpy.arange(os_profile.size)
    if model == "const":
        os_model = numpy.ones_like(pixels) * stat(os_profile)
    elif model == "profile":
        os_model = os_profile
    elif model == "poly":
        model = numpy.polynomial.Polynomial.fit(pixels, os_profile, **kwargs)
        os_model = model(pixels)
    elif model == "spline":
        nknots = kwargs.pop("nknots", 300)
        kwargs.setdefault(
            "t",
            numpy.linspace(
                pixels[len(pixels) // nknots],
                pixels[-1 * len(pixels) // nknots],
                nknots,
            ),
        )
        kwargs.setdefault("task", -1)
        model = interpolate.splrep(pixels, os_profile, **kwargs)
        os_model = interpolate.splev(pixels, model)

    if axis == 1:
        os_model = os_model[:, None]
    elif axis == 0:
        os_model = os_model[None, :]

    return os_data, os_profile, os_model


def _percentile_normalize(images, pct=75):
    """percentile normalize a given stacked images

    Parameters
    ----------
    images : array_like
        3-dimensional array with first axis the image index
    pct : float, optional
        percentile at which the calculate the normalization factor, by default 75

    Returns
    -------
    array_like
        3-dimensional array of normalized images
    array_like
        vector containing normalization factors for each image
    """
    # calculate normalization factor
    pcts = numpy.nanpercentile(images, pct, axis=(1, 2))
    norm = bn.nanmedian(pcts) / pcts

    return norm[:, None, None] * images, norm


def _bg_subtraction(images, quad_sections, bg_sections):
    """returns a background subtracted set of images

    The background is calculated as the median in the given `bg_sections` of
    the given `images`. The actual sections used to calculate the background,
    the median background and the standard deviation of the background are also
    returned matching the shape of the given `images`.


    Parameters
    ----------
    images : array_like
        3-dimensional array of stacked images
    quad_sections : list_like
        4-element list containing the FITS formatted sections of each quadrant
    bg_sections : list_like
        4-element list containing the FITS formatted sections for the background

    Returns
    -------
    array_like
        3-dimensional background subtracted images
    array_like
        3-dimensional median and standard deviation background images
    list_like
        4-element list containing the sections used to calculate the background
    """
    bg_images_med = numpy.ma._masked_array(
        numpy.zeros_like(images), mask=images.mask, fill_value=numpy.nan, hard_mask=True
    )
    bg_images_std = numpy.ma.masked_array(
        numpy.zeros_like(images), mask=images.mask, fill_value=numpy.nan, hard_mask=True
    )
    bg_sections = []
    for i, quad_sec in enumerate(quad_sections):
        xquad, yquad = [slice(idx) for idx in _parse_ccd_section(quad_sec)]
        xbg, ybg = [slice(idx) for idx in _parse_ccd_section(bg_sections[i])]
        # extract quad sections for BG calculation
        bg_array = images[:, xbg, ybg]
        bg_sections.append(bg_array)
        # calculate median and standard deviation BG
        bg_med = bn.nanmedian(bg_array, axis=(1, 2))
        bg_std = bn.nanstd(bg_array, axis=(1, 2))
        # set background sections in corresponding images
        bg_images_med[:, yquad, xquad] = bg_med[:, None, None]
        bg_images_std[:, yquad, xquad] = bg_std[:, None, None]
    images_bgcorr = images - bg_images_med
    # update mask to propagate NaNs in resulting images
    images_bgcorr.mask = images.mask | numpy.isnan(images_bgcorr)

    return images_bgcorr, bg_images_med, bg_images_std, bg_sections


def _remove_spikes(data, width=11, threshold=0.5):
    """Returns a data array with spikes removed

    Parameters
    ----------
    data : array_like
        1-dimensional array of data
    width : int, optional
        width of the window where spikes are located, by default 11
    threshold : float, optional
        threshold to remove spikes, by default 0.5

    Returns
    -------
    array_like
        1-dimensional array with spikes removed
    """
    data_ = copy(data)
    hw = width // 2
    for irow in range(hw, data.size - hw):
        chunk = data[irow-hw:irow+hw+1]
        has_peaks = (chunk[0] == chunk[-1]) and (numpy.abs(chunk) > chunk[0]).any()
        if has_peaks and (chunk != 0).sum() / width < threshold:
            data_[irow-hw:irow+hw+1] = chunk[0]
    return data_


def _fillin_valleys(data, width=18):
    """fills in valleys in the data array

    Parameters
    ----------
    data : array_like
        1-dimensional array of data
    width : int, optional
        width of the valley to fill, by default 18

    Returns
    -------
    array_like
        1-dimensional array with filled valleys
    """
    data_out = copy(data)
    top = data[0]
    for i in range(data.size):
        if data[i] > top:
            top = data[i]
        if data[i] == top:
            continue
        if data[i] < top:
            j_ini = i
        j_fin = j_ini
        for j in range(j_ini, data.size):
            if data[j] < top:
                continue
            if data[j] == top:
                j_fin = j
                break
        if j_fin - j_ini < width:
            data_out[j_ini:j_fin] = top
    return data_out


def _no_stepdowns(data):
    """Removes stepdowns in the data array

    Parameters
    ----------
    data : array_like
        1-dimensional array of data

    Returns
    -------
    array_like
        1-dimensional array with stepdowns removed
    """
    data_out = copy(data)
    top = data[0]
    for i in range(data.size):
        if data[i] > top:
            top = data[i]
        if data[i] == top:
            continue
        if data[i] < top:
            data_out[i] = top
    return data_out


class LinearSelectionElement:
    """Define a selection element for morphological binary image processing.
       Used, e.g. for binary closure of cosmic ray tracks.
    """

    def __init__(self, n, m, angle):
        """This will produce an n x m selection element with a line going
        through the center according to some angle.

        Parameters
        ----------
        n : int
            Number of rows in selection element.
        m : int
            Number of columns in selection element.
        angle : float
            Angle of line through center, in deg [0,180].
        """
        self.se = None
        self.angle = angle

        se = numpy.zeros((m,n), dtype=int)
        xc, yc = n//2, m//2 # row, col

        if angle >= 0 and angle < 45:
            b = numpy.tan(numpy.deg2rad(angle))
        elif angle >= 45 and angle < 90:
            b = numpy.tan(numpy.deg2rad(90 - angle))
        elif angle >= 90 and angle < 135:
            b = numpy.tan(numpy.deg2rad(angle-90))
        elif angle >= 135 and angle < 180:
            b = numpy.tan(numpy.deg2rad(180-angle))
        else:
            raise ValueError('Angle ({}) must be in [0,180]'.format(angle))

        for x in range(0, n):
            y = int(yc + b*(x-xc))
            if y >= 0 and y < m:
                se[y,x] = 1

        if angle < 45:
            self.se = se
        elif angle >= 45 and angle < 90:
            self.se = se.T
        elif angle >= 90 and angle < 135:
            self.se = se.T[:,::-1]
        else:
            self.se = se[:,::-1]

    def plot(self):
        """Return a plot of the selection element (a bitmap).

        Returns
        -------
        fig : matplotlib.Figure
            Figure object for plotting/saving.
        """
        #- Isolated mpl imports to work in batch with no $DISPLAY
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        n, m = self.se.shape
        fig, ax = plt.subplots(1,1, figsize=(0.2*n, 0.2*m), tight_layout=True)
        ax.imshow(self.se, cmap='gray', origin='lower',
                  interpolation='nearest', vmin=0, vmax=1)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(n+1))
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(m+1))
        ax.set(xticklabels=[], yticklabels=[])
        ax.grid(color='gray')
        ax.tick_params(axis='both', length=0)
        return fig


class Image(Header):
    def __init__(self, data=None, header=None, mask=None, error=None, origin=None, individual_frames=None, slitmap=None):
        Header.__init__(self, header=header, origin=origin)
        self._data = data
        if self._data is not None:
            self._data = self._data.astype("float32")
            self._dim = self._data.shape
        else:
            self._dim = None
        self._mask = mask
        self._error = error
        if self._error is not None:
            self._error = self._error.astype("float32")
        self._origin = origin
        # individual frames that went into the master creation
        self._individual_frames = individual_frames
        # set slit map extension
        self._slitmap = slitmap

    def __add__(self, other):
        """
        Operator to add two Images or add another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
            img = Image(header=self._header, origin=self._origin)

            # add data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data + other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(
                    self._error.astype(numpy.float32) ** 2
                    + other._error.astype(numpy.float32) ** 2
                )
                img.setData(error=new_error.astype(numpy.float32))
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img

        elif isinstance(other, numpy.ndarray):
            img = copy(self)
            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                # add ndarray according do its dimensions
                if self._dim == dim:
                    new_data = self._data + other
                elif len(dim) == 1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data + other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
                        new_data = self._data + other[numpy.newaxis, :]
                else:
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                img = copy(self)
                img.setData(self._data + other)
                return img
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for +: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __radd__(self, other):
        self.__add__(other)

    def __sub__(self, other):
        """
        Operator to subtract two Images or subtract another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
            img = copy(self)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data - other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(
                    self._error.astype(numpy.float32) ** 2
                    + other._error.astype(numpy.float32) ** 2
                )
                img.setData(error=new_error.astype(numpy.float32))
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img

        elif isinstance(other, numpy.ndarray):
            img = copy(self)
            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                # add ndarray according do its dimensions
                if self._dim == dim:
                    new_data = self._data - other
                elif len(dim) == 1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data - other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
                        new_data = self._data - other[numpy.newaxis, :]
                else:
                    new_data = self._data - other
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                img = copy(self)
                img.setData(self._data - other)
                return img
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for -: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __truediv__(self, other):
        """
        Operator to divide two Images or divide by another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
            img = copy(self)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data / other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(
                    (self._error / other._data) ** 2
                    + ((self._data / other._data) * (other._error / other._data**2)) ** 2)
                img.setData(error=new_error)
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img

        elif isinstance(other, numpy.ndarray):
            img = copy(self)
            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                # add ndarray according do its dimensions
                if self._dim == dim:
                    new_data = self._data / other
                    if self._error is not None:
                        new_error = self._error / other
                    else:
                        new_error = None
                elif len(dim) == 1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data / other[:, numpy.newaxis]
                        if self._error is not None:
                            new_error = self._error / other[:, numpy.newaxis]
                        else:
                            new_error is not None
                    elif self._dim[1] == dim[0]:
                        new_data = self._data / other[numpy.newaxis, :]
                        if self._error is not None:
                            new_error = self._error / other[numpy.newaxis, :]
                        else:
                            new_error is not None
                else:
                    new_data = self._data
                img.setData(data=new_data, error=new_error)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data / other
                if self._error is not None:
                    new_error = self._error / other
                else:
                    new_error = None

                img = copy(self)
                img.setData(data=new_data, error=new_error)
                return img
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for /: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __mul__(self, other):
        """
        Operator to divide two Images or divide by another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
            img = copy(self)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data * other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(
                    (self._error * other._data) ** 2 + (self._data * other._error) ** 2
                )
                img.setData(error=new_error)
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img

        elif isinstance(other, numpy.ndarray):
            img = copy(self)
            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                # add ndarray according do its dimensions
                if self._dim == dim:
                    new_data = self._data * other
                elif len(dim) == 1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data * other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
                        new_data = self._data * other[numpy.newaxis, :]
                else:
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                img = copy(self)
                img.setData(data=self._data * other)
                return img
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for *: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __rmul__(self, other):
        self.__mul__(other)

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
        self._header.append(('COMMENT', comstr), bottom=True)

    def measure_fiber_shifts(self, ref_image, columns=[500, 1000, 1500, 2000, 2500, 3000], column_width=25, shift_range=[-5,5], axs=None):
        '''Measure the (thermal, flexure, ...) shift between the fiber (traces) in 2 detrended images in the y (cross dispersion) direction.

        Uses cross-correlations between (medians of a number of) columns to determine
        the shift between the fibers in image2 relative to ref_image. The measurement is performed
        independently at each column in columns= using a median of +-column_width columns.

        Parameters
        ----------
        ref_image: Image or numpy.ndarray
            2D reference image
        columns:  List[int]
            List of columns to cross correlate.
        column_width: int
            window width around each value in columns to use
        shift_range: List[int]
            minimal and maximal value for shift

        Returns
        -------
        numpy.ndarray[float]:
            pixel shifts in columns
        '''
        if isinstance(ref_image, Image):
            ref_data = ref_image._data
        elif isinstance(ref_image, numpy.ndarray):
            ref_data = ref_image

        shifts = numpy.zeros(len(columns))
        for j,c in enumerate(columns):
            s1 = numpy.nanmedian(ref_data[50:-50,c-column_width:c+column_width], axis=1)
            s2 = numpy.nanmedian(self._data[50:-50,c-column_width:c+column_width], axis=1)
            snr = numpy.sqrt(numpy.nanmedian(self._data[50:-50,c-column_width:c+column_width], axis=1))

            min_snr = 5.0
            if numpy.nanmedian(snr) > min_snr:
                _, shifts[j], _ = _cross_match_float(s1, s2, numpy.array([1.0]), shift_range, gauss_window=[-3,3], min_peak_dist=5.0, ax=axs[j])
            else:
                comstr = f"low SNR (<={min_snr}) for thermal shift at column {c}: {numpy.nanmedian(snr):.4f}, assuming = 0.0"
                log.warning(comstr)
                self.add_header_comment(comstr)
                shifts[j] = 0.0

        return shifts

    def apply_pixelmask(self, mask=None):
        """Applies the mask to the data and error arrays, setting to nan when True and leaving the same value otherwise"""
        if mask is None:
            mask = self._mask
        if mask is None:
            return self._data, self._error

        self._data[mask] = numpy.nan
        if self._error is not None:
            self._error[mask] = numpy.nan

        self.is_masked = True

        return self._data, self._error

    def getSection(self, section):
        """get image section"""
        sec_x, sec_y = _parse_ccd_section(section)

        data = self._data[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]]
        if self._error is not None:
            error = self._error[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]]
        else:
            mask = None

        header = self._header

        return Image(data=data, error=error, mask=mask, header=header, origin=self._origin, individual_frames=self._individual_frames, slitmap=self._slitmap)

    def setSection(self, section, subimg, update_header=False, inplace=True):
        """replaces a section in the current frame with given subimage

        this function will replace the information in `section` of the current
        frame with the information contained in the `subimg`. If the current
        frame does not contain an extension present in the `subimg`, it will be
        created in the new frame, filling in the remaining sections with dummy
        data. The rest of the sections will be kept if already present in the
        original frame.

        Parameters
        ----------
        section : str
            CCD section for which to set the new subimage
        subimg : lvmdrp.core.image.Image
            subimage to be added to the current image
        update_header : bool, optional
            whether to update the header with `subimg` header, by default False
        inplace : bool, optional
            whether to create a copy of the current image or not, by default True

        Returns
        -------
        lvmdrp.core.image.Image
            image with the section information replaced by the given subimage
        """
        # initialize new image
        if inplace:
            new_image = self
        else:
            new_image = copy(self)

        # parse frame section
        sec_x, sec_y = _parse_ccd_section(section)

        # create dummy mask and error images in case those are not in the original image
        # and subimg contains that information
        if new_image._mask is None and subimg._mask is not None:
            new_image.setData(mask=numpy.zeros_like(new_image._data), inplace=True)
        if new_image._error is None and subimg._error is not None:
            new_image.setData(error=numpy.zeros_like(new_image._data), inplace=True)

        # replace original image section with given subimg
        new_image._data[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]] = subimg._data
        if new_image._error is not None:
            new_image._error[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]] = subimg._error
        if new_image._mask is not None:
            new_image._mask[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]] = subimg._mask

        # update header if needed
        if update_header:
            new_image._header.update(subimg._header)

        return new_image

    def sqrt(self):
        """
        Computes the square root  of the image

        Returns
        -----------
        Image : data_model.Image object
            A full Image object

        """
        if self._data is not None:
            new_data = numpy.sqrt(self._data)  # sqrt of the data
        else:
            new_data = None

        if self._error is not None and self._data is not None:
            new_error = (
                1 / (2 * numpy.sqrt(self._data)) * self._error
            )  # corresponding error
        else:
            new_error = None

        return Image(
            data=new_data,
            error=new_error,
            mask=self._mask,
            header=self._header,
            origin=self._origin,
            individual_frames=self._individual_frames,
            slitmap=self._slitmap,
        )  # return new Image object with corresponding data

    def swapaxes(self):
        """
        Change the axes of the Image object in place (x,y) -> (y,x)

        """
        if self._data is not None:
            self._data = self._data.T
        if self._mask is not None:
            self._mask = self._mask.T
        if self._error is not None:
            self._error = self._error.T
        self._dim = (self._dim[1], self._dim[0])

    def getDim(self):
        """
        Returns the dimension of the image

        Returns
        -----------
        _dim :  tuple
            The dimension of the image (y,x)

        """
        return self._dim

    def getData(self):
        """
        Returns the stored data of the image

        Returns
        -----------
        _data :  numpy.ndarray
            The stored data of the image

        """
        return self._data

    def getMask(self):
        """
        Returns the bad pixel mask of the image

        Returns
        -----------
        _mask :  numpy.ndarray
            The bad pixel mask of the image

        """
        return self._mask

    def getError(self):
        """
        Returns the associated error of the image

        Returns
        -----------
        _error :  numpy.ndarray
            The associated error of the image

        """
        return self._error

    def getPixel(self, y, x):
        """
        Returns the information for a single pixel of the image.
        x,y are numpy array coordinates starting with 0

        Parameters
        --------------
        y : int
            pixel coordinate in y direction of the image
        x : int
            pixel coordinate in x direction of the image

        Returns
        -----------
        (data, error, mask) :  tuple
            Tuple of data value, its associated error, and bad pixel masked.
            If one of those values are not definied, None is returned for its specific value
        """

        if self._data is not None:
            out_data = self._data[y, x]
        else:
            out_data = None

        if self._error is not None:
            out_error = self._error[y, x]
        else:
            out_error = None

        if self._mask is not None:
            out_mask = self._mask[y, x]
        else:
            out_mask = None
        return out_data, out_error, out_mask

    def getSlice(self, slice, axis):
        if axis == "X" or axis == "x" or axis == 0:
            if self._error is not None:
                error = self._error[slice, :]
            else:
                error = None
            if self._mask is not None:
                mask = self._mask[slice, :]
            else:
                mask = None
            return Spectrum1D(
                numpy.arange(self._dim[1]), self._data[slice, :], error, mask
            )
        elif axis == "Y" or axis == "y" or axis == 1:
            if self._error is not None:
                error = self._error[:, slice]
            else:
                error = None
            if self._mask is not None:
                mask = self._mask[:, slice]
            else:
                mask = None
            return Spectrum1D(
                numpy.arange(self._dim[0]), self._data[:, slice], error, mask
            )

    def setData(
        self, data=None, error=None, mask=None, header=None, select=None, inplace=True
    ):
        """sets data for the current frame

        Parameters
        ----------
        data : array_like, optional
            image to be set in the `data` extension, by default None
        error : array_like, optional
            image to be set in the `error` extension, by default None
        mask : array_like, optional
            image to be set in the `mask` extension, by default None
        header : lvmdrp.core.header.Header, optional
            header object to be set, by default None
        select : array_like, optional
            boolean image to select pixels to be replaced, by default None
        inplace : bool, optional
            whether the original image is overwritten or not, by default True

        Returns
        -------
        lvmdrp.core.image.Image
            image with the given data replaced
        """
        # initialize new image
        if inplace:
            new_image = self
        else:
            new_image = copy(self)

        # if not select given set the full image
        if select is None:
            if data is not None:
                new_image._data = data  # set data if given
                new_image._dim = data.shape  # set dimension

            if mask is not None:
                new_image._mask = mask  # set mask if given
                new_image._dim = mask.shape  # set dimension

            if error is not None:
                new_image._error = error  # set mask if given
                new_image._dim = error.shape  # set dimension
            if header is not None:
                new_image.setHeader(header)  # set header
        else:
            # with select definied only partial data are set
            if data is not None:
                new_image._data[select] = data
            if mask is not None:
                new_image._mask[select] = mask
            if error is not None:
                new_image._error[select] = error
            if header is not None:
                new_image.setHeader(header)  # set header

        return new_image

    def convertUnit(self, to, assume="adu", gain_field="GAIN", inplace=False):
        """converts the unit of the image

        Parameters
        ----------
        to : str
            unit to convert to
        assume : str, optional
            unit to assume the current image is in, by default "adu"
        gain_field : str, optional
            header keyword containing the gain value, by default "GAIN"
        inplace : bool, optional
            whether to overwrite the current image or not, by default False

        Returns
        -------
        lvmdrp.core.image.Image
            image with the given unit
        """
        new_image = self if inplace else copy(self)

        # early return if no data or header to compute conversion
        if new_image._header is None or new_image._data is None:
            return new_image

        current = self._header.get("BUNIT", assume)
        if current == to:
            return new_image

        if current != to:
            exptime = self.getHdrValue("EXPTIME")
            gains = self.getHdrValue(f"AMP? {gain_field}")
            sects = self.getHdrValue("AMP? TRIMSEC")
            n_amp = len(gains)
            for i in range(n_amp):
                if current == "adu" and to == "electron":
                    factor = gains[i]
                elif current == "adu" and to == "electron/s":
                    factor = gains[i] / exptime
                elif current == "electron" and to == "adu":
                    factor = 1 / gains[i]
                elif current == "electron" and to == "electron/s":
                    factor = 1 / exptime
                elif current == "electron/s" and to == "adu":
                    factor = gains[i] * exptime
                elif current == "electron/s" and to == "electron":
                    factor = exptime
                else:
                    raise ValueError(f"Cannot convert from {current} to {to}")

                new_image.setSection(
                    section=sects[i],
                    subimg=new_image.getSection(section=sects[i]) * factor,
                    update_header=False,
                    inplace=True,
                )

            new_image._header["BUNIT"] = to

        return new_image

    def removeMask(self):
        self._mask = None

    def removeError(self):
        self._error = None

    def cutOverscan(self, bound_x, bound_y, subtract=True):
        """
        Cut out a certain region from the image within certain pixels in x and y direction (first and last pixel included starting with 1).
        A median of the remaining region can be subtracted from each pixel of the cutted out image

        Parameters
        --------------
        bound_x : list of two strings/integers
            Limiting boundaries in pixels position along the x axes to be included in the cut out image
        bound_y : list of two strings/integers
            Limiting boundaries in pixels position along the y axes to be included in the cut out image
        subtract : bool, optional with default: True
            Decides whether the median of the remaining pixels is subtraced from each pixel of the cutted out image.

        """

        idx = numpy.indices(
            self.getDim()
        )  # generate the index array for the x and y positions
        # select data outside the cut out region (overscan)
        select = numpy.logical_and(
            numpy.logical_and(
                idx[0] >= int(bound_y[0]) - 1, idx[0] <= int(bound_y[1]) - 1
            ),
            numpy.logical_and(
                idx[1] >= int(bound_x[0]) - 1, idx[1] <= int(bound_x[1]) - 1
            ),
        )
        overscan = self._data[numpy.logical_not(select)]
        # compute the median of the ovserscan
        bias_overscan = bn.nanmedian(overscan)
        # get the data of the cut out region
        self._data = self._data[
            int(bound_y[0]) - 1 : int(bound_y[1]), int(bound_x[0]) - 1 : int(bound_x[1])
        ]
        self._dim = self._data.shape
        if self._error is not None:
            # get the error of the cut out region
            self._error = self._error[
                int(bound_y[0]) - 1 : int(bound_y[1]),
                int(bound_x[0]) - 1 : int(bound_x[1]),
            ]
        if self._mask is not None:
            # get the mask of the cut out region
            self._mask = self._mask[
                int(bound_y[0]) - 1 : int(bound_y[1]),
                int(bound_x[0]) - 1 : int(bound_x[1]),
            ]
        if self._header is not None:
            # adjust the axis information in the header if present
            self.setHdrValue("NAXIS1", self._dim[1])
            self.setHdrValue("NAXIS2", self._dim[0])
        if subtract:
            # subtract the median of the overscan region
            self._data = self._data - bias_overscan
        # return the median of the overscan region
        return bias_overscan

    def orientImage(self, orient):
        """
        Changes the orientation of the Image. This is also applied to the error and mask images if present.

        Parameters
        --------------
        orient : string
            Defines how the image should be oriented. Possible values are 'S','T','X','Y','90','180'', and 270'
            Their meaning are:
                'S' : orientation is unchanged
                'T' : the x and y axes are swapped
                'X' : mirrored along the x axis
                'Y' : mirrored along the y axis
                '90' : rotated by 90 degrees
                '180' : rotated by 180 degrees
                '270' : rotated by 270 degrees
        """
        if orient == "S":
            pass  # no change of orietnation

        elif orient == "T":
            # swap the x and y axis
            self.swapaxes

        elif orient == "X":
            # mirror along x axis
            self._data = numpy.fliplr(self._data)
            self._dim = self._data.shape
            if self._error is not None:
                self._error = numpy.fliplr(self._error)
            if self._mask is not None:
                self._mask = numpy.fliplr(self._mask)

        elif orient == "Y":
            # mirror along y axis
            self._data = numpy.flipud(self._data)
            self._dim = self._data.shape
            if self._error is not None:
                self._error = numpy.flipud(self._error)
            if self._mask is not None:
                self._mask = numpy.flipud(self._mask)

        elif orient == "90":
            # rotate by 90 degrees
            self._data = numpy.rot90(self._data)
            self._dim = self._data.shape
            if self._error is not None:
                self._error = numpy.rot90(self._error)
            if self._mask is not None:
                self._mask = numpy.rot90(self._mask)

        elif orient == "180":
            # rotate by 180 degrees
            self._data = numpy.rot90(self._data, 2)
            self._dim = self._data.shape
            if self._error is not None:
                self._error = numpy.rot90(self._error, 2)
            if self._mask is not None:
                self._mask = numpy.rot90(self._mask, 2)

        elif orient == "270":
            # rotate by 270 degrees
            self._data = numpy.rot90(self._data, 3)
            self._dim = self._data.shape
            if self._error is not None:
                self._error = numpy.rot90(self._error, 3)
            if self._mask is not None:
                self._mask = numpy.rot90(self._mask, 3)

        if self._header is not None:
            # adjust the axis information of the header if present
            self.setHdrValue("NAXIS1", self._dim[1])
            self.setHdrValue("NAXIS2", self._dim[0])

    def loadFitsData(
        self,
        filename,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_frames=None,
        extension_slitmap=None,
        extension_header=0,
    ):
        """
        Load data from a FITS image into an Image object, If no specific extensions are given, the  primary extensio is
        assumed to contain the data. All previous extension will be associated according to the EXTNAME keyword either
        as an error image or a bad pixel mask.

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
        self.filename = filename
        # open FITS file
        hdu = pyfits.open(filename, ignore_missing_end=True, uint=False, memmap=False)
        if ".fz" in filename[-4:]:
            extension_data = 1
            extension_header = 1
        if (
            extension_data is None
            and extension_mask is None
            and extension_error is None
            and extension_frames is None
            and extension_slitmap is None
        ):
            self._data = hdu[0].data.astype("float32")
            self._dim = self._data.shape  # set dimension
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header["EXTNAME"].split()[0] == "ERROR":
                        self._error = hdu[i].data.astype("float32")
                    elif hdu[i].header["EXTNAME"].split()[0] == "BADPIX":
                        self._mask = hdu[i].data.astype("bool")
                    elif hdu[i].header["EXTNAME"].split()[0] == "FRAMES":
                        self._individual_frames = Table(hdu[i].data)
                    elif hdu[i].header["EXTNAME"].split()[0] == "SLITMAP":
                        self._slitmap = Table(hdu[i].data)

        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data.astype("float32")
                self._dim = self._data.shape  # set dimension

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data.astype("bool")  # take data
                self._dim = self._mask.shape  # set dimension

            if extension_error is not None:
                self._error = hdu[extension_error].data.astype("flaoat32")
                self._dim = self._error.shape  # set dimension
            if extension_frames is not None:
                self._individual_frames = Table(hdu[extension_frames].data)
            if extension_slitmap is not None:
                self._slitmap = Table(hdu[extension_slitmap].data)

        # set is_masked attribute
        self.is_masked = numpy.isnan(self._data).any()

        # get header from the first FITS extension
        self.setHeader(hdu[extension_header].header)
        hdu.close()

    def writeFitsData(
        self, filename, extension_data=None, extension_mask=None, extension_error=None, extension_frames=None, extension_slitmap=None
    ):
        """
        Save information from an Image into a FITS file. A single or multiple extension file can be created.
        If all optional paramters are set to None, all data if contained will be stored in to extension of the FITS file.

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

        hdus = [None, None, None, None, None]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if (
            extension_data is None
            and extension_error is None
            and extension_mask is None
            and extension_frames is None
            and extension_slitmap is None
        ):
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name="ERROR")
            if self._mask is not None:
                hdus[2] = pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX")
            if self._individual_frames is not None:
                hdus[3] = pyfits.BinTableHDU(self._individual_frames, name="FRAMES")
            if self._slitmap is not None:
                hdus[4] = pyfits.BinTableHDU(self._slitmap, name="SLITMAP")
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data > 0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name="DATA")

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

            # frames hdu
            if extension_frames == 0:
                hdu = pyfits.PrimaryHDU(self._individual_frames)
            elif extension_frames > 0 and extension_frames is not None:
                hdus[extension_frames] = pyfits.BinTableHDU(self._individual_frames, name="FRAMES")

            # slitmap hdu
            if extension_slitmap == 0:
                hdu = pyfits.PrimaryHDU(self._slitmap)
            elif extension_slitmap > 0 and extension_slitmap is not None:
                hdus[extension_slitmap] = pyfits.BinTableHDU(self._slitmap, name="SLITMAP")

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except Exception:
                break
        # if len(hdus)>1:
        #    hdus[0].update_ext_name('T')

        if len(hdus) > 0:
            hdus[0].header['DRPVER'] = drpver
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
            if self._header is not None:
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                try:
                    hdu[0].header["BZERO"] = 0
                except KeyError:
                    pass
                hdu[0].update_header()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        hdu.writeto(filename, output_verify="silentfix", overwrite=True)

    def computePoissonError(self, rdnoise):
        self._error = numpy.zeros_like(self._data)
        select = self._data > 0
        self._error[select] = numpy.sqrt(self._data[select] + rdnoise**2)
        self._error[numpy.logical_not(select)] = rdnoise

    def replace_subselect(self, select, data=None, error=None, mask=None):
        """
            Set data for an Image. Specific data values can replaced according to a specific selection.

            Parameters
            --------------
            select : numpy.ndarray(bool)
                array defining the selection of pixel to be set
            data : numpy.ndarray(float), optional with default = None
                array corresponding to the data to be set
            error : numpy.ndarray(float), optional with default = None
                array corresponding to the data to be set
            mask : numpy.ndarray(bool), optional with default = None
                array corresponding to the bad pixel to be set
        """
        if data is not None:
            self._data[select] = data
        if mask is not None:
            self._mask[select] = mask
        if error is not None:
            self._error[select] = error

    def replaceMaskMedian(self, box_x, box_y, replace_error=1e20):
        """
            Replace bad pixels with the median value of pixel in a rectangular filter window

            Parameters
            --------------
            box_x : int
                Pixel size of filter window in x direction
            box_y : int
                Pixel size of filter window in y direction
            replace_error : float, optional with default: None
                Error that should be set for bad pixel

            Returns
            -----------
            new_image :  Image object
                Subsampled image
        """

        if self._data is None:
            raise RuntimeError("Image object is empty. Nothing to process.")

        idx = numpy.indices(self._dim)  # create an index array
        # get x and y coordinates of bad pixels

        y_cors = idx[0][self._mask]
        x_cors = idx[1][self._mask]

        out_data = copy(self._data)
        msk_data = copy(self._data)
        msk_data[self._mask] = numpy.nan
        out_error = copy(self._error)

        # esimate the pixel distance form the bad pixel to the filter window boundary
        delta_x = numpy.ceil(box_x/2.0)
        delta_y = numpy.ceil(box_y/2.0)

        # iterate over bad pixels
        for m in range(len(y_cors)):
            # computes the min and max pixels of the filter window in x and y
            range_y = numpy.clip([y_cors[m]-delta_y, y_cors[m]+delta_y+1], 0, self._dim[0]-1).astype(numpy.uint16)
            range_x = (numpy.clip([x_cors[m]-delta_x, x_cors[m]+delta_x+1], 0, self._dim[1]-1)).astype(numpy.uint16)
            # compute the masked median within the filter window and replace data
            out_data[y_cors[m], x_cors[m]] = bn.nanmedian(msk_data[range_y[0]:range_y[1],
                                                          range_x[0]:range_x[1]])
            if self._error is not None and replace_error is not None:
                # replace the error of bad pixel if defined
                out_error[y_cors[m], x_cors[m]] = replace_error

        # create new Image object
        new_image = Image(data=out_data, error=out_error,  mask=self._mask, header=self._header, slitmap=self._slitmap)
        return new_image

    def calibrateSDSS(self, fieldPhot, subtractSky=True):
        exptime = float(self.getHdrValue("exptime"))
        softbias = float(self.getHdrValue("softbias"))
        filter = self.getHdrValue("FILTER")

        ## Read photometric information from SDSS Field table
        photHeader = Header()
        photHeader.loadFitsHeader(fieldPhot)
        filters = numpy.array(photHeader.getHdrValue("filters").split(" "))
        filter_select = filters == filter
        f = pyfits.open(fieldPhot, memmap=False)
        tbfield = f[1].data
        aa = float(tbfield.field("aa")[0][filter_select])
        kk = float(tbfield.field("kk")[0][filter_select])
        airmass = float(tbfield.field("airmass")[0][filter_select])
        gain = float(tbfield.field("gain")[0][filter_select])
        dark_var = float(tbfield.field("dark_variance")[0][filter_select])

        ## Photometric conversion factor
        factor = (10 ** (0.4 * (aa + (kk * airmass)))) / exptime
        calibratedImage = self._data - softbias

        if subtractSky:
            try:
                sky = self.getHdrValue("sky")
            except KeyError:
                sky = bn.nanmedian(calibratedImage)
            # print('Sky Background %s: %.2f Counts' %(filters[filter_select][0],sky))
            calibratedImage = calibratedImage - sky
            error = numpy.sqrt((calibratedImage + sky) / gain + dark_var)
        else:
            error = numpy.sqrt((calibratedImage) / gain + dark_var)
        self.setHdrValue("FIELD", 0)

        sdssImage = copy(self)
        sdssImage.setData(data=calibratedImage * factor, error=error * factor, inplace=True)
        return sdssImage

    def split(self, fragments, axis="X"):
        image_list = []
        if axis == "X" or axis == "x" or axis == 1:
            axis_split = 1
        elif axis == "Y" or axis == "y" or axis == 0:
            axis_split = 0

        split_data = numpy.array_split(self._data, fragments, axis=axis_split)
        if self._error is not None:
            split_error = numpy.array_split(self._error, fragments, axis=axis_split)
        else:
            split_error = [None] * fragments
        if self._mask is not None:
            split_mask = numpy.array_split(self._mask, fragments, axis=axis_split)
        else:
            split_mask = [None] * fragments
        for i in range(fragments):
            image_list.append(
                Image(data=split_data[i], error=split_error[i], mask=split_mask[i], header=self._header, origin=self._origin, individual_frames=self._individual_frames, slitmap=self._slitmap)
            )

        return image_list

    def unsplit(self, image_list, axis="X"):
        if axis == "X" or axis == "x" or axis == 1:
            axis_split = 1
        elif axis == "Y" or axis == "y" or axis == 0:
            axis_split = 0

        data = []
        error = []
        mask = []
        for i in range(len(image_list)):
            data.append(image_list[i]._data)
            error.append(image_list[i]._error)
            mask.append(image_list[i]._mask)
        self._data = numpy.concatenate(data, axis_split)
        self._dim = self._data.shape
        if error[0] is not None:
            self._error = numpy.concatenate(error, axis_split)
        if mask[0] is not None:
            self._mask = numpy.concatenate(mask, axis_split)
        if image_list[0]._header is not None:
            self._header = image_list[0]._header

    def subsampleImg(self):
        """
        Subsample the image by a factor of 2, e.g. each pixel is divided into 4 pixel so that their sum is 4 times the original one.

        Returns
        -----------
        new_image :  Image object
            Subsampled image

        """
        # create empty array with 2 time larger size in both axes
        new_dim = (self._dim[0] * 2, self._dim[1] * 2)
        if self._data is not None:
            new_data = numpy.zeros(new_dim, dtype=numpy.float32)
        else:
            new_data = None
        if self._error is not None:
            new_error = numpy.zeros(new_dim, dtype=numpy.float32)
        else:
            new_error = None
        if self._mask is not None:
            new_mask = numpy.zeros(new_dim, dtype="bool")
        else:
            new_mask = None

        # create index array of the new
        indices = numpy.indices(new_dim) + 1
        # define selection for the the 4 different subpixels in which to store the original data
        select1 = numpy.logical_and(indices[0] % 2 == 1, indices[1] % 2 == 1)
        select2 = numpy.logical_and(indices[0] % 2 == 1, indices[1] % 2 == 0)
        select3 = numpy.logical_and(indices[0] % 2 == 0, indices[1] % 2 == 1)
        select4 = numpy.logical_and(indices[0] % 2 == 0, indices[1] % 2 == 0)
        # set pixel for the subsampled data, error and mask
        if self._data is not None:
            new_data[select1] = self._data.flatten()
            new_data[select2] = self._data.flatten()
            new_data[select3] = self._data.flatten()
            new_data[select4] = self._data.flatten()
        if self._error is not None:
            new_error[select1] = self._error.flatten()
            new_error[select2] = self._error.flatten()
            new_error[select3] = self._error.flatten()
            new_error[select4] = self._error.flatten()
        if self._mask is not None:
            new_mask[select1] = self._mask.flatten()
            new_mask[select2] = self._mask.flatten()
            new_mask[select3] = self._mask.flatten()
            new_mask[select4] = self._mask.flatten()
        # create new Image object with the new subsample data
        new_image = copy(self)
        new_image.setData(data=new_data, error=new_error, mask=new_mask, inplace=True)
        return new_image

    def rebin(self, bin_x, bin_y):
        """
        Rebin the image by regullarly summing up the pixel in a regual rectangular binning window with size bin_x times bin_y.
        Make sure that the size of the binning window matches with the total number of pixel in the original image.

        Parameters
        --------------
        bin_x : int
            Pixel size of the binning window in x direction
        bin_y : int
            Pixel size of the binning window in y direction

        Returns
        -----------
        new_image :  Image object
            Subsampled image
        """
        # sum over the data array over each axis by the given pixel
        new = numpy.sum(
            numpy.reshape(self._data, (self._dim[0], self._dim[1] // bin_x, bin_x)), 2
        )
        new2 = numpy.sum(
            numpy.reshape(new, (self._dim[0] // bin_y, bin_y, self._dim[1] // bin_x)), 1
        )

        if self._error is not None:
            # sum over the error array (converted to variance and back) over each axis by the given pixel
            error_new = numpy.sum(
                numpy.reshape(
                    self._error**2, (self._dim[0], self._dim[1] // bin_x, bin_x)
                ),
                2,
            )
            error_new2 = numpy.sqrt(
                numpy.sum(
                    numpy.reshape(
                        error_new, (self._dim[0] // bin_y, bin_y, self._dim[1] // bin_x)
                    ),
                    1,
                )
            )
        else:
            error_new2 = None

        if self._mask is not None:
            # create the new  bad pixel mask
            mask_new = numpy.sum(
                numpy.reshape(self._mask, (self._dim[0], self._dim[1] // bin_x, bin_x)),
                2,
            )
            mask_new2 = numpy.sum(
                numpy.reshape(
                    mask_new, (self._dim[0] // bin_y, bin_y, self._dim[1] // bin_x)
                ),
                1,
            )
            # if only one bad pixel in the binning pixel exists the binned pixel will have the bad pixel status
            new_mask = mask_new2 > 0
        else:
            new_mask = None
        # create new Image object and return
        new_img = Image(
            data=new2,
            error=error_new2,
            mask=new_mask,
            header=self._header,
            origin=self._origin,
            individual_frames=self._individual_frames,
            slitmap=self._slitmap,
        )
        return new_img

    def convolveImg(self, kernel, mode="nearest"):
        """
        Convolves the data of the Image with a given kernel. The mask and error information will be unchanged.

        Parameters
        --------------
        kernel : ndarray
            Convolution kernel
        mode :  string, optional with default: 'nearest'
            Set the mode how to handle the boundarys within the convolution


        Returns
        -----------
        new_image :  Image object
            Convolved image
        """

        # convolve the data array with the given convolution kernel
        new = ndimage.filters.convolve(self._data, kernel, mode=mode)
        if self._error is not None:
            new_error = numpy.sqrt(
                ndimage.filters.convolve(self._error**2, kernel, mode=mode)
            )
        else:
            new_error = None
        # create new Image object with the error and the mask unchanged and return
        new_image = copy(self)
        new_image.setData(data=new, error=new_error, inplace=True)
        return new_image

    def convolveGaussImg(self, sigma_x, sigma_y, mode="nearest", mask=False):
        """
        Convolves the data of the Image with a given kernel. The mask and error information will be unchanged.

        Parameters
        --------------
        sigma_x : float
            With of the Gaussian in pixels along the x direction
        sigma_y : float
            With of the Gaussian in pixels along the y direction
        mode :  string, optional with default: 'nearest'
            Set the mode how to handle the boundarys within the convolution


        Returns
        -----------
        new_image :  Image object
            Convolved Image
        """
        # convolve the data array with the 2D Gaussian convolution kernel

        if self._mask is not None and mask is True:
            mask_data = self._data[self._mask]
            self._data[self._mask] = 0
            gauss = ndimage.filters.gaussian_filter(
                self._data, (sigma_y, sigma_x), mode=mode
            )
            scale = ndimage.filters.gaussian_filter(
                (~self._mask).astype('float32'), (sigma_y, sigma_x), mode=mode
            )
            new = gauss / scale
            self._data[self._mask] = mask_data
        else:
            new = ndimage.filters.gaussian_filter(
                self._data, (sigma_y, sigma_x), mode=mode
            )
        # create new Image object with the error and the mask unchanged and return
        new_image = copy(self)
        new_image.setData(data=new, inplace=True)
        return new_image

    def medianImg(self, size, mode="nearest", use_mask=False, propagate_error=False):
        """return median filtered image with the given kernel size

        optionally the method for handling boundary can be set with the `mode`
        parameter (see documentation for `scipy.ndimage.median_filter`). Masked
        pixels are handledby setting `use_mask=True`. In this last case, the
        `mode` is ignored (see documentation for `scipy.signal.medfilt2d`).

        Parameters
        ----------
        size : tuple
            2-value tuple for the size of the median box
        mode : str, optional
            method to handle boundary pixels, by default "nearest"
        use_mask : bool, optional
            whether to take into account masked pixels or not, by default False
        propagate_error : bool, optional
            whether to propagate the error or not, by default False

        Returns
        -------
        lvmdrp.core.image.Image
            median filtered image
        """
        if self._mask is None and use_mask:
            new_data = copy(self._data)
            new_data[self._mask] = numpy.nan
            new_data = ndimage.median_filter(new_data, size, mode=mode)
            new_mask = None
            new_error = None
            if propagate_error and self._error is not None:
                new_error = numpy.sqrt(ndimage.median_filter(self._error ** 2, size, mode=mode))
        elif self._mask is not None and not use_mask:
            new_data = ndimage.median_filter(self._data, size, mode=mode)
            new_mask = self._mask
            new_error = None
            if propagate_error and self._error is not None:
                new_error = numpy.sqrt(ndimage.median_filter(self._error ** 2, size, mode=mode))
        else:
            # copy data and replace masked with nans
            new_data = copy(self._data)
            new_data[self._mask] = numpy.nan
            # perform median filter
            new_data = signal.medfilt2d(new_data, size)
            # update mask
            new_mask = numpy.isnan(new_data)
            # reset original masked values in new array
            new_data[new_mask] = self._data[new_mask]
            # update error
            new_error = None
            if propagate_error and self._error is not None:
                new_error = copy(self._error)
                new_error[self._mask] = numpy.nan
                new_error = numpy.sqrt(signal.medfilt2d(new_error ** 2, size))
                # reset masked errors in new array
                new_error[new_mask] = self._error[new_mask]

        image = copy(self)
        image.setData(data=new_data, error=new_error, mask=new_mask)
        return image

    def collapseImg(self, axis, mode="mean"):
        """
        Return a collapsed cut as a spectrum object along one axis.

        Parameters
        --------------
        axis : string or int
            Define the axis along which the image is collpase, either 'X','x', or 0 for the
            x axis or 'Y','y', or 1 for the y axis.
        mode : string, optional with default: 'mean'
            set the mode of the collpasing, which are: mean, sum, nansum, median,  min or max

        Returns
        -----------
        spec :  Spectrum1D object
            A Spetrum1D object containing the collapse information
        """

        # set the axis for the collapsing
        if axis == "X" or axis == "x" or axis == 0:
            axis = 1
            dim = self._dim[0]
        elif axis == "Y" or axis == "y" or axis == 1:
            axis = 0
            dim = self._dim[1]
        # collapse the image to Spectrum1D object with requested operation
        if mode == "mean":
            return Spectrum1D(numpy.arange(dim), bn.nanmean(self._data, axis))
        elif mode == "sum":
            return Spectrum1D(numpy.arange(dim), bn.nansum(self._data, axis))
        elif mode == "median":
            return Spectrum1D(numpy.arange(dim), bn.nanmedian(self._data, axis))
        elif mode == "min":
            return Spectrum1D(numpy.arange(dim), bn.nanmin(self._data, axis))
        elif mode == "max":
            return Spectrum1D(numpy.arange(dim), bn.nanmax(self._data, axis))

    def fitPoly(self, axis="y", order=4, plot=-1):
        """
        Fits the image with polynomial function along a given axis.

        Parameters
        --------------
        axis : string or int
            Define the axis along which the  polynomial fit is performed either 'X', 'x', or 0 for the
            x axis or 'Y',' y', or 1 for the y axis.

        order : integer, optional with default: 4
            Order of the polynomials to be fitted

        Returns
        -----------
        new_image :  Image object
            An Image object containing the polynomial modelled data
        """

        # create empty arrays to store the results
        fit_result = numpy.zeros(self._dim, dtype=numpy.float32)
        fit_par = numpy.zeros((order + 1, self._dim[1]), dtype=numpy.float32)

        # match orientation of the image
        if axis == "y" or axis == "Y" or axis == 0:
            pass
        else:
            self.swapaxes()
        # setup the base line for the polynomial fitting
        slices = self._dim[1]
        x = numpy.arange(self._dim[0])
        x = x - bn.nanmean(x)
        # if self._mask is not None:
        #    self._mask = numpy.logical_and(self._mask, numpy.logical_not(numpy.isnan(self._data)))
        valid = ~self._mask.astype("bool")
        # iterate over the image
        for i in range(slices):
            # decide on the bad pixel mask
            # fit the polynomial and clip pixels that are 3sigma away from the initial fit
            # redo the polynomial fitting with clipped data
            if self._mask is not None:
                if numpy.sum(valid[:, i]) > order + 5:
                    fit_par[:, i] = numpy.polyfit(
                        x[valid[:, i]], self._data[valid[:, i], i], order
                    )  # fit polynom
                    res = (
                        numpy.polyval(fit_par[:, i], x[valid[:, i]])
                        - self._data[valid[:, i], i]
                    )  # compute residuals
                    std = numpy.std(res)  # compute RMS deviation
                    select = numpy.logical_and(
                        res > -3 * std, res < 3 * std
                    )  # select good pixels
                    # print(std, i, numpy.sum(numpy.isnan(self._data[self._mask[:, i], i])))
                    fit_par[:, i] = numpy.polyfit(
                        x[valid[:, i]][select],
                        self._data[valid[:, i], i][select],
                        order,
                    )  # fit #refit polynomial with clipped data
                    if plot == i:
                        plt.plot(x, self._data[:, i], "-b")
                        plt.plot(
                            x[valid[:, i]][select],
                            self._data[valid[:, i], i][select],
                            "ok",
                        )
                        max = bn.nanmax(self._data[valid[:, i], i][select])
                    fit_result[:, i] = numpy.polyval(
                        fit_par[:, i], x
                    )  # evalute the polynom
                else:
                    fit_result[:, i] = numpy.nan
            else:
                fit_par[:, i] = numpy.polyfit(x, self._data[:, i], order)  # fit polynom
                res = (
                    numpy.polyval(fit_par[:, i], x) - self._data[:, i]
                )  # compute residuals
                std = numpy.std(res)  # compute RMS deviation
                select = numpy.logical_and(
                    res > -3 * std, res < 3 * std
                )  # select good pixels
                fit_par[:, i] = numpy.polyfit(
                    x[select], self._data[:, i][select], order
                )  # refit polynomial with clipped data
                if plot == i:
                    plt.plot(x[select], self._data[:, i][select], "ok")
                fit_result[:, i] = numpy.polyval(
                    fit_par[:, i], x
                )  # evalute the polynom
            if plot == i:
                plt.plot(x, fit_result[:, i], "-r")
                plt.ylim([0, max])
                plt.show()
        # match orientation of the output array
        if axis == "y" or axis == "Y" or axis == 0:
            pass
        else:
            fit_result = fit_result.T
            self.swapaxes()

        # create image object and return
        if self._mask is not None:
            new_mask = numpy.isnan(fit_result)
        else:
            new_mask = None

        new_img = copy(self)
        new_img.setData(data=fit_result, mask=new_mask)
        return new_img

    def fitSpline(self, axis="y", degree=3, smoothing=0, use_weights=False, clip=None, interpolate_missing=True):
        """Fits a spline to the image along a given axis

        Parameters
        ----------
        axis : string or int
            Define the axis along which the spline fit is performed either 'X', 'x', or 0 for the
            x axis or 'Y',' y', or 1 for the y axis.
        degree : int, optional
            degree of the spline fit, by default 3
        smoothing : float, optional
            smoothing factor for the spline fit, by default 0
        use_weights : bool, optional
            whether to use the inverse variance as weights for the spline fit or not, by default False
        clip : tuple, optional
            minimum and maximum values to clip the spline model, by default None
        interpolate_missing : bool, optional
            interpolate coefficients if spline fitting failed

        Returns
        -------
        lvmdrp.core.image.Image
            An Image object containing the spline modelled data
        """
        # match orientation of the image
        if axis == "y" or axis == "Y" or axis == 0:
            pass
        else:
            self.swapaxes()

        pixels = numpy.arange(self._dim[0])
        models = numpy.zeros(self._dim)
        for i in range(self._dim[1]):
            good_pix = ~self._mask[:,i] if self._mask is not None else ~numpy.isnan(self._data[:,i])

            # skip column if all pixels are masked
            if good_pix.sum() == 0:
                warnings.warn(f"Skipping column {i} due to all pixels being masked", RuntimeWarning)
                continue

            # define spline fitting parameters
            masked_pixels = pixels[good_pix]
            data = self._data[good_pix, i]
            vars = self._error[good_pix, i] ** 2

            # group pixels into continuous segments
            groups, indices = [], []
            for j in range(len(masked_pixels)-1):
                delta = masked_pixels[j+1] - masked_pixels[j]
                if delta > 1:
                    if len(indices) > 0:
                        indices.append(j)
                        groups.append(indices)
                        indices = []
                    continue
                elif j == len(masked_pixels)-2:
                    indices.append(j+1)
                    groups.append(indices)
                else:
                    indices.append(j)

            if len(groups) <= degree+1:
                warnings.warn(f"Skipping column {i} due to insufficient data for spline fit", RuntimeWarning)
                continue

            # collapse groups into single pixel
            new_masked_pixels, new_data, new_vars = [], [], []
            for group in groups:
                new_masked_pixels.append(numpy.nanmean(masked_pixels[group]))
                new_data.append(numpy.nanmedian(data[group]))
                new_vars.append(numpy.nanmean(vars[group]))
            masked_pixels = numpy.asarray(new_masked_pixels)
            data = numpy.asarray(new_data)
            vars = numpy.asarray(new_vars)

            # fit spline
            if use_weights:
                weights = numpy.divide(1, vars, out=numpy.zeros_like(vars), where=vars!=0)
                spline_pars = interpolate.splrep(masked_pixels, data, w=weights, s=smoothing)
            else:
                spline_pars = interpolate.splrep(masked_pixels, data, s=smoothing)
            models[:, i] = interpolate.splev(pixels, spline_pars)

        # clip spline fit if required
        if clip is not None:
            models = numpy.clip(models, clip[0], clip[1])

        # interpolate failed columns if requested
        masked_columns = numpy.count_nonzero((models < 0)|numpy.isnan(models), axis=0) >= 0.1*self._dim[0]
        if interpolate_missing and masked_columns.any():
            log.info(f"interpolating spline fit in {masked_columns.sum()} columns")
            x_pixels = numpy.arange(self._dim[1])
            f = interpolate.interp1d(x_pixels[~masked_columns], models[:, ~masked_columns], axis=1, bounds_error=False, fill_value="extrapolate")
            models[:, masked_columns] = f(x_pixels[masked_columns])

            # clip spline fit if required
            if clip is not None:
                models = numpy.clip(models, clip[0], clip[1])

        # match orientation of the output array
        if axis == "y" or axis == "Y" or axis == 0:
            pass
        else:
            models = models.T
            self.swapaxes()

        new_img = copy(self)
        new_img.setData(data=models)
        return new_img

    def match_reference_column(self, ref_column=2000, ref_centroids=None, stretch_range=[0.7, 1.3], shift_range=[-100, 100], return_pars=False):
        """Returns the reference centroids matched against the current image

        Parameters
        ----------
        ref_column : int, optional
            column to use as reference, by default 2000
        ref_centroids : numpy.ndarray, optional
            reference centroids to use for matching, by default None
        stretch_range : list, tuple, optional
            range of stretch factors to try, by default [0.7, 1.3]
        shift_range : list, tuple, optional
            range of shifts to try in pixels, by default [-100, 100]
        return_pars : bool, optional
            also returns a tuple with the strech and shift parameters, by default False

        Returns
        -------
        numpy.ndarray
            matched reference centroids
        """
        if self._header is None:
            raise ValueError("No header available")
        if self._slitmap is None:
            raise ValueError("No slitmap available")
        channel, spec = list(self._header.get("CCD", [None, None]))
        spec = int(spec)
        if channel is None:
            raise ValueError("No CCD information available in header")

        # extract guess positions from fibermap
        slitmap = self._slitmap[self._slitmap["spectrographid"] == spec]
        if ref_centroids is None and self._slitmap is not None:
            ref_centroids = slitmap[f"ypix_{channel}"].data
        else:
            raise ValueError("No reference profile provided and no slitmap available")

        # define stretch factors
        s_min, s_max = stretch_range
        s_del = 0.1/self._dim[0]
        stretch_factors = numpy.arange(s_min, s_max+s_del, s_del)

        # correct reference fiber positions
        profile = self.getSlice(ref_column, axis="y")
        profile._data = numpy.nan_to_num(profile._data, nan=0, neginf=0, posinf=0)
        pixels = profile._pixels
        pixels = numpy.arange(pixels.size)
        guess_heights = numpy.ones_like(ref_centroids) * numpy.nanmax(profile._data)
        ref_profile = _spec_from_lines(ref_centroids, sigma=1.2, wavelength=pixels, heights=guess_heights)
        log.info(f"correcting guess positions for column {ref_column}")
        cc, bhat, mhat = _cross_match(
            ref_spec=ref_profile,
            obs_spec=profile._data,
            stretch_factors=stretch_factors,
            shift_range=shift_range)
        log.info(f"stretch factor: {mhat:.3f}, shift: {bhat:.3f}")
        ref_centroids = ref_centroids * mhat + bhat
        if return_pars:
            return ref_centroids, mhat, bhat
        return ref_centroids

    def trace_fiber_centroids(self, ref_column=2000, ref_centroids=None, mask_fibstatus=1,
                              ncolumns=140, method="gauss", fwhm_guess=2.5, fwhm_range=[1.0, 3.5],
                              counts_threshold=5000, max_diff=1.5):
        if self._header is None:
            raise ValueError("No header available")
        if self._slitmap is None:
            raise ValueError("No slitmap available")
        channel, spec = list(self._header.get("CCD", [None, None]))
        spec = int(spec)
        if channel is None:
            raise ValueError("No CCD information available in header")

        # extract guess positions from fibermap
        slitmap = self._slitmap[self._slitmap["spectrographid"] == spec]
        if ref_centroids is None and self._slitmap is not None:
            ref_centroids = slitmap[f"ypix_{channel}"].data
        elif ref_centroids is None:
            raise ValueError("No reference profile provided and no slitmap available")

        if isinstance(mask_fibstatus, int):
            mask_fibstatus = [mask_fibstatus]
        elif isinstance(mask_fibstatus, list, tuple, numpy.ndarray):
            mask_fibstatus = mask_fibstatus.astype(int)

        # set mask
        bad_fibers = numpy.isin(slitmap["fibstatus"].data, mask_fibstatus)
        good_fibers = numpy.where(numpy.logical_not(bad_fibers))[0]

        # select columns to measure centroids
        step = self._dim[1] // ncolumns
        columns = numpy.concatenate((numpy.arange(ref_column, 0, -step), numpy.arange(ref_column, self._dim[1], step)))
        log.info(f"selecting {len(columns)-1} columns within range [{min(columns)}, {max(columns)}]")

        # create empty traces mask for the image
        fibers = ref_centroids.size
        dim = self.getDim()
        centroids = TraceMask()
        centroids.createEmpty(data_dim=(fibers, dim[1]), samples_columns=sorted(set(columns)))
        centroids.setFibers(fibers)
        centroids._good_fibers = good_fibers
        centroids.setHeader(self._header.copy())
        centroids._header["IMAGETYP"] = "trace_centroid"

        # set positions of fibers along reference column
        centroids.setSlice(ref_column, axis="y", data=ref_centroids, mask=numpy.zeros_like(ref_centroids, dtype="bool"))

        # trace centroids in each column
        iterator = tqdm(enumerate(columns), total=len(columns), desc="tracing centroids", unit="column", ascii=True)
        for i, icolumn in iterator:
            # extract column profile
            img_slice = self.getSlice(icolumn, axis="y")

            # get fiber positions along previous column
            if icolumn == ref_column:
                # trace reference column first or skip if already traced
                if i == 0:
                    cent_guess, _, mask_guess = centroids.getSlice(ref_column, axis="y")
                else:
                    continue
            else:
                cent_guess, _, mask_guess = centroids.getSlice(columns[i-1], axis="y")

            # cast fiber positions to integers
            cent_guess = cent_guess.round().astype("int16")

            # measure fiber positions
            bound_lower = numpy.array([0]*cent_guess.size + (cent_guess-max_diff).tolist() + [fwhm_range[0]/2.354]*cent_guess.size)
            bound_upper = numpy.array([numpy.inf]*cent_guess.size + (cent_guess+max_diff).tolist() + [fwhm_range[1]/2.354]*cent_guess.size)
            cen_slice, msk_slice = img_slice.measurePeaks(cent_guess, method, init_sigma=fwhm_guess / 2.354, threshold=counts_threshold, bounds=(bound_lower, bound_upper))

            centroids._samples[f"{icolumn}"][~msk_slice] = cen_slice[~msk_slice]
            centroids.setSlice(icolumn, axis="y", data=cen_slice, mask=msk_slice, samples=cen_slice)

        return centroids

    def trace_fiber_widths(self, fiber_centroids, ref_column=2000, ncolumns=40, nblocks=18, iblocks=[],
                           fwhm_guess=2.5, fwhm_range=[1.0,3.5], max_diff=1.5, counts_threshold=5000):

        if self._header is None:
            raise ValueError("No header available")
        unit = self._header["BUNIT"]

        # select columns to fit for amplitudes, fiber_centroids and FWHMs per fiber block
        step = self._dim[1] // ncolumns
        columns = numpy.concatenate((numpy.arange(ref_column, 0, -step), numpy.arange(ref_column+step, self._dim[1], step)))
        log.info(f"tracing fibers in {len(columns)} columns within range [{min(columns)}, {max(columns)}]")

        # initialize flux and FWHM traces
        trace_cent = TraceMask()
        trace_cent.createEmpty(data_dim=(fiber_centroids._fibers, self._dim[1]), samples_columns=sorted(set(columns)))
        trace_cent.setFibers(fiber_centroids._fibers)
        trace_cent._good_fibers = fiber_centroids._good_fibers
        trace_cent.setHeader(self._header.copy())
        trace_amp = copy(trace_cent)
        trace_fwhm = copy(trace_cent)
        trace_cent._header["IMAGETYP"] = "trace_centroid"
        trace_amp._header["IMAGETYP"] = "trace_amplitude"
        trace_fwhm._header["IMAGETYP"] = "trace_fwhm"

        # fit peaks, fiber_centroids and FWHM in each column
        mod_columns, residuals = [], []
        for i, icolumn in enumerate(columns):
            log.info(f"tracing column {icolumn} ({i+1}/{len(columns)})")
            # get slice of data and trace
            cen_slice, _, msk_slice = fiber_centroids.getSlice(icolumn, axis="y")
            img_slice = self.getSlice(icolumn, axis="y")

            # define fiber blocks
            if iblocks and isinstance(iblocks, (list, tuple, numpy.ndarray)):
                cen_blocks = numpy.split(cen_slice, nblocks)
                cen_blocks = numpy.asarray(cen_blocks)[iblocks]
                msk_blocks = numpy.split(msk_slice, nblocks)
                msk_blocks = numpy.asarray(msk_blocks)[iblocks]
            else:
                cen_blocks = numpy.split(cen_slice, nblocks)
                msk_blocks = numpy.split(msk_slice, nblocks)

            # fit each block
            par_blocks = []
            iterator = tqdm(enumerate(zip(cen_blocks, msk_blocks)), total=len(cen_blocks), desc="fitting fibers (00/00 good fibers)", ascii=True, unit="block")
            for j, (cen_block, msk_block) in iterator:
                # apply flux threshold
                cen_idx = cen_block.round().astype("int16")
                msk_block |= (img_slice._data[cen_idx] < counts_threshold)

                # mask bad fibers
                cen_block = cen_block[~msk_block]
                # initialize parameters with the full block size
                par_block = numpy.ones(3 * msk_block.size) * numpy.nan
                par_mask = numpy.tile(msk_block, 3)

                # skip block if all fibers are masked
                if msk_block.sum() > 0.5 * msk_block.size:
                    log.info(f"skipping fiber block {j+1}/{nblocks} (most fibers masked)")
                else:
                    # fit gaussian models to each fiber profile
                    iterator.set_description(f"fitting fibers ({cen_block.size:02d}/{msk_block.size:02d} good fibers)")
                    iterator.refresh()
                    # log.info(f"fitting fiber block {j+1}/{nblocks} ({cen_block.size}/{msk_block.size} good fibers)")
                    bound_lower = numpy.array([0]*cen_block.size + (cen_block-max_diff).tolist() + [fwhm_range[0]/2.354]*cen_block.size)
                    bound_upper = numpy.array([numpy.inf]*cen_block.size + (cen_block+max_diff).tolist() + [fwhm_range[1]/2.354]*cen_block.size)
                    _, par_block[~par_mask] = img_slice.fitMultiGauss(cen_block, init_fwhm=fwhm_guess, bounds=(bound_lower, bound_upper))

                par_blocks.append(par_block)

            # combine all parameters in a single array
            par_joint = numpy.asarray([numpy.split(par_block, 3) for par_block in par_blocks])
            par_joint = par_joint.transpose(1, 0, 2).reshape(3, -1)
            # define joint gaussian model
            mod_joint = Gaussians(par=par_joint.ravel())

            # store joint model
            mod_columns.append(mod_joint)

            # get parameters of joint model
            amp_slice = par_joint[0]
            cent_slice = par_joint[1]
            fwhm_slice = par_joint[2] * 2.354

            # mask fibers with invalid values
            amp_off = (amp_slice <= counts_threshold)
            log.info(f"masking {amp_off.sum()} samples with amplitude < {counts_threshold} {unit}")
            cent_off = numpy.abs(1 - cent_slice / numpy.concatenate(cen_blocks)) > 0.01
            log.info(f"masking {cent_off.sum()} samples with centroids refined by > 1 %")
            fwhm_off = (fwhm_slice < fwhm_range[0]) | (fwhm_slice > fwhm_range[1])
            log.info(f"masking {fwhm_off.sum()} samples with FWHM outside {fwhm_range} pixels")
            amp_mask = numpy.isnan(amp_slice) | amp_off | cent_off | fwhm_off
            cent_mask = numpy.isnan(cent_slice) | amp_off | cent_off | fwhm_off
            fwhm_mask = numpy.isnan(fwhm_slice) | amp_off | cent_off | fwhm_off

            if amp_slice.size != trace_amp._data.shape[0]:
                dummy_amp = numpy.split(numpy.zeros(trace_amp._data.shape[0]), nblocks)
                dummy_cent = numpy.split(numpy.zeros(trace_cent._data.shape[0]), nblocks)
                dummy_fwhm = numpy.split(numpy.zeros(trace_fwhm._data.shape[0]), nblocks)
                dummy_amp_mask = numpy.split(numpy.ones(trace_amp._data.shape[0], dtype=bool), nblocks)
                dummy_cent_mask = numpy.split(numpy.ones(trace_cent._data.shape[0], dtype=bool), nblocks)
                dummy_fwhm_mask = numpy.split(numpy.ones(trace_fwhm._data.shape[0], dtype=bool), nblocks)

                amp_split = numpy.split(amp_slice, len(iblocks))
                cent_split = numpy.split(cent_slice, len(iblocks))
                fwhm_split = numpy.split(fwhm_slice, len(iblocks))
                amp_mask_split = numpy.split(amp_mask, len(iblocks))
                cent_mask_split = numpy.split(cent_mask, len(iblocks))
                fwhm_mask_split = numpy.split(fwhm_mask, len(iblocks))
                for j, iblock in enumerate(iblocks):
                    dummy_amp[iblock] = amp_split[j]
                    dummy_cent[iblock] = cent_split[j]
                    dummy_fwhm[iblock] = fwhm_split[j]
                    dummy_amp_mask[iblock] = amp_mask_split[j]
                    dummy_cent_mask[iblock] = cent_mask_split[j]
                    dummy_fwhm_mask[iblock] = fwhm_mask_split[j]

                # update traces
                trace_amp._samples[f"{icolumn}"] = numpy.concatenate(dummy_amp)
                trace_cent._samples[f"{icolumn}"] = numpy.concatenate(dummy_cent)
                trace_fwhm._samples[f"{icolumn}"] = numpy.concatenate(dummy_fwhm)
                trace_amp.setSlice(icolumn, axis="y", data=numpy.concatenate(dummy_amp), mask=numpy.concatenate(dummy_amp_mask))
                trace_cent.setSlice(icolumn, axis="y", data=numpy.concatenate(dummy_cent), mask=numpy.concatenate(dummy_cent_mask))
                trace_fwhm.setSlice(icolumn, axis="y", data=numpy.concatenate(dummy_fwhm), mask=numpy.concatenate(dummy_fwhm_mask))
                trace_amp._good_fibers = numpy.arange(trace_amp._fibers)[~numpy.all(trace_amp._mask, axis=1)]
                trace_cent._good_fibers = numpy.arange(trace_cent._fibers)[~numpy.all(trace_cent._mask, axis=1)]
                trace_fwhm._good_fibers = numpy.arange(trace_fwhm._fibers)[~numpy.all(trace_fwhm._mask, axis=1)]
            else:
                # update traces
                trace_amp._samples[f"{icolumn}"] = amp_slice
                trace_cent._samples[f"{icolumn}"] = cent_slice
                trace_fwhm._samples[f"{icolumn}"] = fwhm_slice
                trace_amp.setSlice(icolumn, axis="y", data=amp_slice, mask=amp_mask)
                trace_cent.setSlice(icolumn, axis="y", data=cent_slice, mask=cent_mask)
                trace_fwhm.setSlice(icolumn, axis="y", data=fwhm_slice, mask=fwhm_mask)

            # compute model column
            mod_slice = mod_joint(img_slice._pixels)

            # compute residuals
            integral_mod = numpy.trapz(mod_slice, img_slice._pixels)
            integral_mod = integral_mod if integral_mod != 0 else numpy.nan
            # NOTE: this is a hack to avoid integrating the whole column when tracing a few blocks
            integral_dat = numpy.trapz(img_slice._data * (mod_slice>0), img_slice._pixels)
            residuals.append((integral_mod - integral_dat) / integral_dat * 100)

            # compute fitted model stats
            chisq_red = bn.nansum((mod_slice - img_slice._data)[~img_slice._mask]**2 / img_slice._error[~img_slice._mask]**2) / (self._dim[0] - 1 - 3)
            log.info(f"joint model {chisq_red = :.2f}")
            if amp_mask.all() or cent_mask.all() or fwhm_mask.all():
                continue
            min_amp, max_amp, median_amp = bn.nanmin(amp_slice[~amp_mask]), bn.nanmax(amp_slice[~amp_mask]), bn.nanmedian(amp_slice[~amp_mask])
            min_cent, max_cent, median_cent = bn.nanmin(cent_slice[~cent_mask]), bn.nanmax(cent_slice[~cent_mask]), bn.nanmedian(cent_slice[~cent_mask])
            min_fwhm, max_fwhm, median_fwhm = bn.nanmin(fwhm_slice[~fwhm_mask]), bn.nanmax(fwhm_slice[~fwhm_mask]), bn.nanmedian(fwhm_slice[~fwhm_mask])
            log.info(f"joint model amplitudes: {min_amp = :.2f}, {max_amp = :.2f}, {median_amp = :.2f}")
            log.info(f"joint model centroids: {min_cent = :.2f}, {max_cent = :.2f}, {median_cent = :.2f}")
            log.info(f"joint model FWHMs: {min_fwhm = :.2f}, {max_fwhm = :.2f}, {median_fwhm = :.2f}")

        return trace_amp, trace_cent, trace_fwhm, columns, mod_columns, residuals

    def trace_fiber_widths_opt(self, fiber_centroids, ref_column=2000, ncolumns=40, nblocks=18, iblocks=[],
                           fwhm_guess=2.5, fwhm_range=[1.0,3.5], max_diff=1.5, counts_threshold=5000):

        if self._header is None:
            raise ValueError("No header available")
        unit = self._header["BUNIT"]

        # initialize flux and FWHM traces
        trace_cent = TraceMask()
        trace_cent.createEmpty(data_dim=(fiber_centroids._fibers, self._dim[1]))
        trace_cent.setFibers(fiber_centroids._fibers)
        trace_cent._good_fibers = fiber_centroids._good_fibers
        trace_cent.setHeader(self._header.copy())
        trace_amp = copy(trace_cent)
        trace_fwhm = copy(trace_cent)
        trace_cent._header["IMAGETYP"] = "trace_centroid"
        trace_amp._header["IMAGETYP"] = "trace_amplitude"
        trace_fwhm._header["IMAGETYP"] = "trace_fwhm"

        # select columns to fit for amplitudes, fiber_centroids and FWHMs per fiber block
        step = self._dim[1] // ncolumns
        columns = numpy.concatenate((numpy.arange(ref_column, 0, -step), numpy.arange(ref_column+step, self._dim[1], step)))
        log.info(f"tracing fibers in {len(columns)} columns: {','.join(map(str, columns))}")

        # fit peaks, fiber_centroids and FWHM in each column
        mod_columns, residuals = [], []
        for i, icolumn in enumerate(columns):
            log.info(f"tracing column {icolumn} ({i+1}/{len(columns)})")
            # get slice of data and trace
            cen_slice, _, msk_slice = fiber_centroids.getSlice(icolumn, axis="y")
            img_slice = self.getSlice(icolumn, axis="y")

            # mask blocks we don't want to trace
            if iblocks and isinstance(iblocks, (list, tuple, numpy.ndarray)):
                cen_blocks = numpy.asarray(numpy.split(cen_slice, nblocks))
                msk_blocks = numpy.asarray(numpy.split(msk_slice, nblocks))
                cen_blocks = cen_blocks[iblocks]
                msk_blocks = msk_blocks[iblocks]
                msk_slice = numpy.concatenate(msk_blocks)
                cen_slice = numpy.concatenate(cen_blocks)

            cen_idx = cen_slice.round().astype("int16")
            msk_slice |= (img_slice._data[cen_idx] < counts_threshold)

            # initialize parameters with the full block size
            par_slice = numpy.ones(3 * msk_slice.size) * numpy.nan
            par_mask = numpy.tile(msk_slice, 3)

            # fit gaussian models to each fiber profile
            log.info(f"fitting fibers in column {i+1}/{ncolumns} ({cen_slice.size}/{msk_slice.size} selected fibers)")
            bound_lower = numpy.array([0]*cen_slice.size + (cen_slice-max_diff).tolist() + [fwhm_range[0]/2.354]*cen_slice.size)
            bound_upper = numpy.array([numpy.inf]*cen_slice.size + (cen_slice+max_diff).tolist() + [fwhm_range[1]/2.354]*cen_slice.size)
            _, par_slice[~par_mask] = img_slice.fitMultiGauss(cen_slice, init_fwhm=fwhm_guess, bounds=(bound_lower, bound_upper))

            # define joint gaussian model
            mod_joint = Gaussians(par=par_slice)

            # store joint model
            mod_columns.append(mod_joint)

            # get parameters of joint model
            amp_slice = par_slice[0]
            cent_slice = par_slice[1]
            fwhm_slice = par_slice[2] * 2.354

            # mask fibers with invalid values
            amp_off = (amp_slice <= counts_threshold)
            log.info(f"masking {amp_off.sum()} samples with amplitude < {counts_threshold} {unit}")
            cent_off = numpy.abs(1 - cent_slice / numpy.concatenate(cen_blocks)) > 0.01
            log.info(f"masking {cent_off.sum()} samples with centroids refined by > 1 %")
            fwhm_off = (fwhm_slice < fwhm_range[0]) | (fwhm_slice > fwhm_range[1])
            log.info(f"masking {fwhm_off.sum()} samples with FWHM outside {fwhm_range} pixels")
            amp_mask = numpy.isnan(amp_slice) | amp_off | cent_off | fwhm_off
            cent_mask = numpy.isnan(cent_slice) | amp_off | cent_off | fwhm_off
            fwhm_mask = numpy.isnan(fwhm_slice) | amp_off | cent_off | fwhm_off

            if amp_slice.size != trace_amp._data.shape[0]:
                dummy_amp = numpy.split(numpy.zeros(trace_amp._data.shape[0]), nblocks)
                dummy_cent = numpy.split(numpy.zeros(trace_cent._data.shape[0]), nblocks)
                dummy_fwhm = numpy.split(numpy.zeros(trace_fwhm._data.shape[0]), nblocks)
                dummy_amp_mask = numpy.split(numpy.ones(trace_amp._data.shape[0], dtype=bool), nblocks)
                dummy_cent_mask = numpy.split(numpy.ones(trace_cent._data.shape[0], dtype=bool), nblocks)
                dummy_fwhm_mask = numpy.split(numpy.ones(trace_fwhm._data.shape[0], dtype=bool), nblocks)

                amp_split = numpy.split(amp_slice, len(iblocks))
                cent_split = numpy.split(cent_slice, len(iblocks))
                fwhm_split = numpy.split(fwhm_slice, len(iblocks))
                amp_mask_split = numpy.split(amp_mask, len(iblocks))
                cent_mask_split = numpy.split(cent_mask, len(iblocks))
                fwhm_mask_split = numpy.split(fwhm_mask, len(iblocks))
                for j, iblock in enumerate(iblocks):
                    dummy_amp[iblock] = amp_split[j]
                    dummy_cent[iblock] = cent_split[j]
                    dummy_fwhm[iblock] = fwhm_split[j]
                    dummy_amp_mask[iblock] = amp_mask_split[j]
                    dummy_cent_mask[iblock] = cent_mask_split[j]
                    dummy_fwhm_mask[iblock] = fwhm_mask_split[j]

                # update traces
                trace_amp.setSlice(icolumn, axis="y", data=numpy.concatenate(dummy_amp), mask=numpy.concatenate(dummy_amp_mask))
                trace_cent.setSlice(icolumn, axis="y", data=numpy.concatenate(dummy_cent), mask=numpy.concatenate(dummy_cent_mask))
                trace_fwhm.setSlice(icolumn, axis="y", data=numpy.concatenate(dummy_fwhm), mask=numpy.concatenate(dummy_fwhm_mask))
                trace_amp._good_fibers = numpy.arange(trace_amp._fibers)[~numpy.all(trace_amp._mask, axis=1)]
                trace_cent._good_fibers = numpy.arange(trace_cent._fibers)[~numpy.all(trace_cent._mask, axis=1)]
                trace_fwhm._good_fibers = numpy.arange(trace_fwhm._fibers)[~numpy.all(trace_fwhm._mask, axis=1)]
            else:
                # update traces
                trace_amp.setSlice(icolumn, axis="y", data=amp_slice, mask=amp_mask)
                trace_cent.setSlice(icolumn, axis="y", data=cent_slice, mask=cent_mask)
                trace_fwhm.setSlice(icolumn, axis="y", data=fwhm_slice, mask=fwhm_mask)

            # compute model column
            mod_slice = mod_joint(img_slice._pixels)

            # compute residuals
            integral_mod = numpy.trapz(mod_slice, img_slice._pixels)
            integral_mod = integral_mod if integral_mod != 0 else numpy.nan
            # NOTE: this is a hack to avoid integrating the whole column when tracing a few blocks
            integral_dat = numpy.trapz(img_slice._data * (mod_slice>0), img_slice._pixels)
            residuals.append((integral_mod - integral_dat) / integral_dat * 100)

            # compute fitted model stats
            chisq_red = bn.nansum((mod_slice - img_slice._data)[~img_slice._mask]**2 / img_slice._error[~img_slice._mask]**2) / (self._dim[0] - 1 - 3)
            log.info(f"joint model {chisq_red = :.2f}")
            if amp_mask.all() or cent_mask.all() or fwhm_mask.all():
                continue
            min_amp, max_amp, median_amp = bn.nanmin(amp_slice[~amp_mask]), bn.nanmax(amp_slice[~amp_mask]), bn.nanmedian(amp_slice[~amp_mask])
            min_cent, max_cent, median_cent = bn.nanmin(cent_slice[~cent_mask]), bn.nanmax(cent_slice[~cent_mask]), bn.nanmedian(cent_slice[~cent_mask])
            min_fwhm, max_fwhm, median_fwhm = bn.nanmin(fwhm_slice[~fwhm_mask]), bn.nanmax(fwhm_slice[~fwhm_mask]), bn.nanmedian(fwhm_slice[~fwhm_mask])
            log.info(f"joint model amplitudes: {min_amp = :.2f}, {max_amp = :.2f}, {median_amp = :.2f}")
            log.info(f"joint model centroids: {min_cent = :.2f}, {max_cent = :.2f}, {median_cent = :.2f}")
            log.info(f"joint model FWHMs: {min_fwhm = :.2f}, {max_fwhm = :.2f}, {median_fwhm = :.2f}")

        return trace_amp, trace_cent, trace_fwhm, columns, mod_columns, residuals

    def traceFWHM(
        self, axis_select, TraceMask, blocks, init_fwhm, threshold_flux, max_pix=None
    ):
        # create an empty trace  of the given size
        fwhm = numpy.zeros(
            (TraceMask._fibers, TraceMask._data.shape[1]), dtype=numpy.float32
        )
        mask = numpy.ones((TraceMask._fibers, TraceMask._data.shape[1]), dtype="bool")
        pixels = numpy.arange(fwhm.shape[1])
        for i in pixels[axis_select]:
            # print(i,fwhm.shape[1])
            slice_img = self.getSlice(i, axis="y")  # extract cross-dispersion slice
            slice_trace = TraceMask.getSlice(
                i, axis="y"
            )  # extract positions in cross-dispersion
            if max_pix is None:
                good = slice_trace[2] == 0
            else:
                good = (
                    (slice_trace[2] == 0)
                    & (numpy.round(slice_trace[0]) < max_pix)
                    & (numpy.round(slice_trace[0]) > 0)
                )
            trace = slice_trace[0][good]

            #   if i==pixels[axis_select][10]:
            #     plot=4
            #  else:
            #    plot=-1
            plot = -1
            #   print(slice_img._data.shape, trace)
            fwhm_fit = slice_img.measureFWHMPeaks(
                trace,
                blocks,
                init_fwhm=init_fwhm,
                threshold_flux=threshold_flux,
                plot=plot,
            )  # obtain the fiber profile FWHM for each slice
            fwhm[good, i] = fwhm_fit[0]
            mask[good, i] = fwhm_fit[
                1
            ]  #    traceFWHM.setSlice(i, axis='y', data = fwhm_fit[0], mask = fwhm_fit[1]) # insert the result into the trace mask
            # return traceFWHM
        return (fwhm, mask)

    def extractSpecAperture(self, TraceMask, aperture):
        pos = TraceMask._data
        bad_pix = TraceMask._mask
        good_pix = numpy.logical_not(bad_pix)
        data = numpy.zeros((TraceMask._fibers, self._dim[1]), dtype=numpy.float32)
        if self._error is not None:
            error = numpy.zeros((TraceMask._fibers, self._dim[1]), dtype=numpy.float32)
        else:
            error = None
        mask = numpy.zeros((TraceMask._fibers, self._dim[1]), dtype="bool")
        for i in range(self._dim[1]):
            pixels = numpy.round(
                pos[:, i][:, numpy.newaxis]
                + numpy.arange(
                    -(aperture - 1) / 2.0, (aperture - 1) / 2.0 + 0.001, 1.0
                )[numpy.newaxis, :]
            ).astype("int")
            bad = numpy.logical_or(
                numpy.sum(pixels > self._dim[0] - 1, 1) > 0,
                numpy.sum(pixels < 2, 1) > 0,
            )

            good_pix[bad, i] = False
            bad_pix[bad, i] = True
            pixels = numpy.round(
                pos[good_pix[:, i], i][:, numpy.newaxis]
                + numpy.arange(
                    -(aperture - 1) / 2.0, (aperture - 1) / 2.0 + 0.001, 1.0
                )[numpy.newaxis, :]
            ).astype("int")
            bad = numpy.sum(pixels > self._dim[1] - 1, 1) > 0
            data[good_pix[:, i], i] = numpy.sum(self._data[:, i][pixels], 1)
            if self._error is not None:
                error[good_pix[:, i], i] = numpy.sqrt(
                    numpy.sum(self._error[:, i][pixels] ** 2, 1)
                )
            if self._mask is not None:
                mask[good_pix[:, i], i] = numpy.sum(self._mask[:, i][pixels], 1) > 0

        # update mask with trace mask
        mask |= bad_pix
        return data, error, mask

    def extractSpecOptimal(self, cent_trace, trace_fwhm, plot_fig=False):
        # initialize RSS arrays
        data = numpy.zeros((cent_trace._fibers, self._dim[1]), dtype=numpy.float32)
        if self._error is not None:
            error = numpy.zeros((cent_trace._fibers, self._dim[1]), dtype=numpy.float32)
        else:
            error = None
        mask = numpy.zeros((cent_trace._fibers, self._dim[1]), dtype="bool")

        self._data = numpy.nan_to_num(self._data)
        self._error = numpy.nan_to_num(self._error, nan=numpy.inf)

        # convert FWHM trace to sigma
        trace_sigma = trace_fwhm / 2.354

        for i in range(self._dim[1]):
            # get i-column from image and trace
            slice_img = self.getSlice(i, axis="y")
            slice_cent = cent_trace.getSlice(i, axis="y")
            cent = slice_cent[0]

            # define fiber mask
            bad_fiber = (slice_cent[2] == 1) | (
                (slice_cent[0] < 0) | (slice_cent[0] > len(slice_img._data) - 1)
            )
            # bad_fiber = numpy.logical_or(
            #     (slice_cent[2] == 1),
            #     numpy.logical_or(
            #         slice_cent[0] < 0, slice_cent[0] > len(slice_img._data) - 1
            #     ),
            # )
            good_fiber = ~bad_fiber

            # get i-column from sigma trace
            sigma = trace_sigma.getSlice(i, axis="y")[0]

            # set NaNs to zero in image slice
            select_nan = numpy.isnan(slice_img._data)
            slice_img._data[select_nan] = 0

            # define fiber index
            indices = numpy.indices((self._dim[0], numpy.sum(good_fiber)))

            # measure flux along the given columns
            result = slice_img.obtainGaussFluxPeaks(
                cent[good_fiber], sigma[good_fiber], indices, plot=plot_fig
            )
            data[good_fiber, i] = result[0]
            if self._error is not None:
                error[good_fiber, i] = result[1]
            if self._mask is not None:
                mask[good_fiber, i] = result[2]
            mask[bad_fiber, i] = True
        return data, error, mask

    def maskFiberTraces(self, TraceMask, aperture=3, parallel="auto"):
        n0 = 0.5 * aperture
        dx = numpy.arange(-n0, n0 + 1, 0.5)
        trace = TraceMask.getData()[0]
        if self._mask is None:
            self._mask = numpy.zeros(self._dim, dtype="bool")
        if parallel == "auto":
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus)
            mask_pixels = []
            for i in range(self._dim[1]):
                mask_pixels.append(
                    pool.apply_async(
                        numpy.unique,
                        (
                            [
                                numpy.clip(
                                    numpy.round(
                                        trace[:, i][numpy.newaxis, :]
                                        + dx[:, numpy.newaxis]
                                    ),
                                    0,
                                    self._dim[0] - 1,
                                ).astype("int16")
                            ]
                        ),
                    )
                )
            pool.close()
            pool.join()

        for i in range(self._dim[1]):
            if cpus > 1:
                self._mask[mask_pixels[i].get(), i] = True
            else:
                select = numpy.unique(
                    numpy.clip(
                        numpy.round(
                            trace[:, i][numpy.newaxis, :] + dx[:, numpy.newaxis]
                        ),
                        0,
                        self._dim[0] - 1,
                    )
                    .astype("int16")
                    .flatten()
                )
                self._mask[select, i] = True

    def peakPosition(self, guess_x=None, guess_y=None, box_x=None, box_y=None):
        image = self._data * numpy.logical_not(self._mask)
        if guess_x is not None and guess_y is not None:
            cent_x = guess_x
            cent_y = guess_y
        else:
            cent_x = numpy.rint(self._dim[1] / 2.0)
            cent_y = numpy.rint(self._dim[0] / 2.0)
        if box_x is not None and box_y is not None:
            min_x = cent_x - numpy.rint(box_x / 2.0)
            max_x = cent_x + numpy.rint(box_x / 2.0)
            min_y = cent_y - numpy.rint(box_y / 2.0)
            max_y = cent_y + numpy.rint(box_y / 2.0)
            position = numpy.array(
                ndimage.maximum_position(image[min_y:max_y, min_x:max_x])
            )
            position[0] = position[0] + min_y
            position[1] = position[1] + min_x
        else:
            position = ndimage.maximum_position(image)

        return position

    def centroidGauss(self, pix_x, pix_y, aperture=8, init_back=0.1, plot=False):
        range_x = numpy.arange(self._dim[1])
        range_y = numpy.arange(self._dim[0])

        spec_x = Spectrum1D(data=self._data[pix_y, :], wave=range_x)
        spec_y = Spectrum1D(data=self._data[:, pix_x], wave=range_y)

        if numpy.sum(spec_x._data) != 0 and numpy.sum(spec_y._data) != 0:
            out_x = spec_x.fitSepGauss(
                [pix_x], aperture=aperture, init_back=init_back, plot=plot
            )
            cent_x = out_x[1]

            out_y = spec_y.fitSepGauss(
                [pix_y], aperture=aperture, init_back=init_back, plot=False
            )
            cent_y = out_y[1]
        else:
            cent_x = 0
            cent_y = 0
        return cent_x, cent_y

    def centreBary(self, guess_x, guess_y, box_size, exponent=4):
        region = self._data[
            int(guess_y - box_size / 2.0) : int(guess_y + box_size / 2.0),
            int(guess_x - box_size / 2.0) : int(guess_x + box_size / 2.0),
        ]
        indices = numpy.indices(self._dim)
        idx_x = indices[1][
            int(guess_y - box_size / 2.0) : int(guess_y + box_size / 2.0),
            int(guess_x - box_size / 2.0) : int(guess_x + box_size / 2.0),
        ]
        idx_y = indices[0][
            int(guess_y - box_size / 2.0) : int(guess_y + box_size / 2.0),
            int(guess_x - box_size / 2.0) : int(guess_x + box_size / 2.0),
        ]
        if self._mask is not None:
            good_pix = numpy.logical_not(
                self._mask[
                    int(guess_y - box_size / 2.0) : int(guess_y + box_size / 2.0),
                    int(guess_x - box_size / 2.0) : int(guess_x + box_size / 2.0),
                ]
            )

            cent_x = (
                numpy.sum((idx_x * region**exponent)[good_pix])
                / numpy.sum(region[good_pix] ** exponent)
                + 1
            )
            cent_y = (
                numpy.sum((idx_y * region**exponent)[good_pix])
                / numpy.sum(region[good_pix] ** exponent)
                + 1
            )
        else:
            cent_x = (
                numpy.sum((idx_x * region**exponent)) / numpy.sum(region**exponent)
                + 1
            )
            cent_y = (
                numpy.sum((idx_y * region**exponent)) / numpy.sum(region**exponent)
                + 1
            )
        return cent_x, cent_y

    def centreMax(self, cent_x=None, cent_y=None, box_size=None):
        if cent_x is not None and cent_y is not None and box_size is not None:
            cent = ndimage.measurements.maximum_position(
                self._data[
                    int(cent_y - box_size / 2.0) : int(cent_y + box_size / 2.0),
                    int(cent_x - box_size / 2.0) : int(cent_x + box_size / 2.0),
                ]
            )
            cent_x = cent[1] + 1 + int(cent_x - box_size / 2.0)
            cent_y = cent[0] + 1 + int(cent_y - box_size / 2.0)
        else:
            cent = ndimage.measurements.maximum_position(self._data)
            cent_x = cent[1] + 1
            cent_y = cent[0] + 1
        return cent_x, cent_y

    def centreFit(self, model="Gauss", cent_x=None, cent_y=None, box_size=None):
        fit_centre = fitting.LevMarLSQFitter()

        if cent_x is not None and cent_y is not None and box_size is not None:
            if model == "Gauss":
                fit_model = models.Gaussian2D(
                    amplitude=10.0,
                    x_mean=float(box_size) / 2.0,
                    y_mean=float(box_size) / 2.0,
                    x_stddev=2.0,
                    y_stddev=2.0,
                )

            image_cut = self._data[
                int(cent_y - box_size / 2.0) : int(cent_y + box_size / 2.0),
                int(cent_x - box_size / 2.0) : int(cent_x + box_size / 2.0),
            ]
            yi, xi = numpy.indices(image_cut.shape)
            best_fit = fit_centre(fit_model, xi, yi, image_cut)

            # cent = ndimage.measurements.maximum_position(self._data[int(cent_y-box_size/2.0):int(cent_y+box_size/2.0), int(cent_x-box_size/2.0):int(cent_x+box_size/2.0)])
            cent = (best_fit.y_mean.value, best_fit.x_mean.value)
            cent_x = cent[1] + 1 + int(cent_x - box_size / 2.0)
            cent_y = cent[0] + 1 + int(cent_y - box_size / 2.0)
        else:
            if model == "Gauss":
                fit_model = models.Gaussian2D(
                    amplitude=10.0,
                    x_mean=self._dim[1] / 2.0,
                    y_mean=self._dim[0] / 2.0,
                    x_stddev=2.0,
                    y_stddev=2.0,
                )
            yi, xi = numpy.indices(self._data.shape)
            best_fit = fit_centre(fit_model, xi, yi, self._data)
            # cent = ndimage.measurements.maximum_position(self._data)
            cent = (best_fit.y_mean.value, best_fit.x_mean.value)
            cent_x = cent[1] + 1
            cent_y = cent[0] + 1
        return cent_x, cent_y

    def extractAper(
        self,
        cent_x,
        cent_y,
        aperture,
        kmax=1000,
        correct_masked=False,
        ignore_mask=True,
    ):
        def inside(xx, yy, r_ap):
            r = numpy.sqrt(numpy.square(xx) + numpy.square(yy))
            select = r <= r_ap
            return numpy.sum(select)

        def get_pixel_area(x, y, r_ap, qrn):
            xx = x - 0.5 + qrn[:, 1]
            yy = y - 0.5 + qrn[:, 0]
            pixel_area = inside(xx, yy, r_ap) / float(qrn.shape[0])
            return pixel_area

        def get_aperture_cover(i, j, xc, yc, r_ap, qrn_seq):
            x = (j + 1) - xc
            y = (i + 1) - yc
            rxy = numpy.sqrt((x * x) + (y * y))
            area = numpy.zeros(i.shape, dtype=numpy.float)
            select = rxy < (r_ap - 0.71)
            area[select] = 1.0
            select = numpy.logical_and(rxy >= (r_ap - 0.71), rxy <= (r_ap + 0.71))
            xx = (x[select] - 0.5)[numpy.newaxis, :] + qrn_seq[:, 1][:, numpy.newaxis]
            yy = (y[select] - 0.5)[numpy.newaxis, :] + qrn_seq[:, 0][:, numpy.newaxis]
            area[select] = numpy.sum(
                numpy.sqrt(numpy.square(xx) + numpy.square(yy)) <= r_ap, 0
            ) / float(qrn_seq.shape[0])
            return area

        def integrate_aperture(
            image,
            xc,
            yc,
            r_ap,
            kmax,
            error=None,
            mask=None,
            ignore_mask=ignore_mask,
            correct_masked=correct_masked,
        ):
            dim = image.shape
            qrn_seq = numpy.random.random_sample((kmax, 2))
            area_mask = numpy.zeros(dim)

            area_mask = numpy.fromfunction(
                get_aperture_cover, dim, xc=xc, yc=yc, r_ap=r_ap, qrn_seq=qrn_seq
            )

            if numpy.sum(mask) > 0 and not ignore_mask:
                area_mask[mask] = 0
            select = numpy.logical_or(numpy.isnan(image), numpy.isinf(image))
            image[select] = 0.0
            flux_mask = area_mask * image
            total_area = numpy.sum(area_mask.flatten())
            total_flux = numpy.sum(flux_mask.flatten())
            if error is not None:
                error_mask = area_mask * error
                total_error = numpy.sqrt(numpy.sum((error_mask**2).flatten()))
            else:
                total_error = None
            expected_area = numpy.pi * r_ap**2
            correction_factor = expected_area / total_area
            coverage = total_area / expected_area
            if correct_masked:
                total_flux = total_flux * correction_factor

            return total_flux, total_error, total_area, flux_mask, area_mask, coverage

        aperture_result = integrate_aperture(
            self._data,
            cent_x,
            cent_y,
            aperture,
            kmax,
            error=self._error,
            mask=self._mask,
        )
        return aperture_result

    def extractApertures(
        self,
        posTab,
        ref_pixel_x,
        ref_pixel_y,
        arc_scale,
        angle=0,
        offset_arc_x=0.0,
        offset_arc_y=0.0,
    ):
        new_posTab = posTab.rotatePosTab(angle).scalePosTab(1.0 / arc_scale)
        new_posTab.offsetPosTab(ref_pixel_x, ref_pixel_y)

        offset_x = (
            offset_arc_x * numpy.cos(float(angle) / 180.0 * numpy.pi)
            - offset_arc_y * numpy.sin(float(angle) / 180.0 * numpy.pi)
        ) / arc_scale
        offset_y = (
            offset_arc_x * numpy.sin(float(angle) / 180.0 * numpy.pi)
            + offset_arc_y * numpy.cos(float(angle) / 180.0 * numpy.pi)
        ) / arc_scale
        new_posTab.offsetPosTab(-offset_x, -offset_y)

        apertures = Apertures(
            new_posTab._arc_position_x,
            new_posTab._arc_position_y,
            numpy.ones(len(new_posTab._arc_position_x)) * new_posTab._size[0],
        )
        flux = apertures.integratedFlux(self)
        return flux

    def reject_cosmics(self, sigma_det=5, rlim=1.2, iterations=5, fwhm_gauss=[2.0,2.0], replace_box=[5, 5],
            replace_error=1e6, increase_radius=0, binary_closure=True,
            gain=1.0, rdnoise=1.0, bias=0.0, verbose=False, inplace=True):
        """
            Detects and removes cosmics from astronomical images based on Laplacian edge
            detection scheme combined with a PSF convolution approach.

            IMPORTANT:
            The image and the readout noise are assumed to be in units of electrons.
            The image also needs to be BIAS subtracted! The gain can be entered to convert the image from ADUs to
            electros, when this is down already set gain=1.0 as the default. If ncessary a homegnous bias level can
            be subtracted if necessary but default is 0.0.

                Parameters
                --------------
                data: ndarray
                        Two-dimensional array representing the input image in which cosmic rays are detected.
                sigma_det: float, default: 5.0
                        Detection limit of edge pixel above the noise in (sigma units) to be detected as comiscs
                rlim: float, default: 1.2
                        Detection threshold between Laplacian edged and Gaussian smoothed image
                iterations: integer, default: 5
                        Number of iterations. Should be >1 to fully detect extended cosmics
                fwhm_gauss: list of floats, default: [2.0, 2.0]
                        FWHM of the Gaussian smoothing kernel in x and y direction on the CCD
                replace_box: list integers, default: [5,5]
                        median box size in x and y to estimate replacement values from valid pixels
                replace_error: float, default: 1e6
                        Error value for bad pixels in the comupted error image
                increase_radius: integer, default: 0
                        Increase the boundary of each detected cosmic ray pixel by the given number of pixels.
                binary_closure: booean, default: True
                        Apply binary closure to final mask to merge long cosmic ray traces that were separated
                        along the propagation direction
                gain: float, default=1.0
                        Value of the gain in units of electrons/ADUs
                rdnoise: float, default=1.0
                        Value of the readout noise in electrons
                bias: float, default=0.0
                        Optional subtraction of a bias level.
                verbose: boolean, default: False
                        Flag for providing information during the processing on the command line
                inplace: boolean, default: True
                        Flag to indicate whether the code should modify the existing data or return
                        a new Image instance with the modified data. In the latter case the mask and error
                        extensions ONLY contain the cosmic-related pixels.

                Ouput
                -------------
                out: Image class instance
                    Result of the detection process is an Image which contains .data, .error, .mask as attributes for the
                    cleaned image, the internally computed error image and a mask image with flags for cosmic ray pixels.

                Reference
                --------------
                Husemann et al. 2012, A&A, Volume 545, A137 (https://ui.adsabs.harvard.edu/abs/2012A%26A...545A.137)

        """

        # convert all parameters to proper type
        sigma_x = fwhm_gauss[0] / 2.354
        sigma_y = fwhm_gauss[1] / 2.354
        box_x = int(replace_box[0])
        box_y = int(replace_box[1])

        # define Laplacian convolution kernal
        LA_kernel = numpy.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])/4.0

        # Initiate image instances
        img_original = Image(data=self._data)
        img = Image(data=self._data)

        # subtract bias if applicable
        if (bias > 0.0) and verbose:
            log.info(f'Subtract bias level {bias:.2f} from image')
        img = img - bias
        img_original = img_original - bias

        # apply gain factor to data if applicable
        if (gain != 1.0) and verbose:
            log.info(f'  Convert image from ADUs to electrons using a gain factor of {gain:.2f}')
        img = img * gain
        img_original = img_original * gain

        # compute noise using read-noise value
        if (rdnoise > 0.0) and verbose:
            log.info(f'  A value of {rdnoise:.2f} is used for the electron read-out noise')
        img_original._error = numpy.sqrt((numpy.clip(img_original._data, a_min=0.0, a_max=None) + rdnoise**2))

        select = numpy.zeros(img._dim, dtype=bool)
        img_original._mask = numpy.zeros(img._dim, dtype=bool)
        img._mask = numpy.zeros(img._dim, dtype=bool)

        # start iteration
        out = img
        for i in range(iterations):
            if verbose:
                log.info(f'  Start iteration {i+1}')

            # create smoothed noise fromimage
            noise = out.medianImg((box_y, box_x))
            select_neg2 = noise._data <= 0
            noise.replace_subselect(select_neg2, data=0)
            noise = (noise + rdnoise ** 2).sqrt()

            sub = img.subsampleImg()  # subsample image
            conv = sub.convolveImg(LA_kernel)  # convolve subsampled image with kernel
            select_neg = conv < 0
            conv.replace_subselect(select_neg, data=0)  # replace all negative values with 0
            Lap = conv.rebin(2, 2)  # rebin the data to original resolution
            S = Lap/(noise*2)  # normalize Laplacian image by the noise
            S_prime = S-S.medianImg((5, 5))  # cleaning of the normalized Laplacian image

            # Perform additional clean using a 2D Gaussian smoothing kernel
            fine = out.convolveGaussImg(sigma_x, sigma_y, mask=True)  # convolve image with a 2D Gaussian
            fine_norm = out/fine
            select_neg = fine_norm < 0
            fine_norm.replace_subselect(select_neg, data=0)
            sub_norm = fine_norm.subsampleImg()  # subsample image
            Lap2 = sub_norm.convolveImg(LA_kernel)
            Lap2 = Lap2.rebin(2, 2)  # rebin the data to original resolution

            select = numpy.logical_or(numpy.logical_and(Lap2 > rlim, S_prime > sigma_det), select)

            if verbose:
                dim = img_original._dim
                det_pix = numpy.sum(select)
                log.info(f'  Total number of detected cosmics: {det_pix} out of {dim[0] * dim[1]} pixels')

            if i == iterations-1:
                img_original.replace_subselect(select, mask=True)  # set the new mask
                if increase_radius > 0:
                    mask_img = Image(data=img_original._mask)
                    mask_new = mask_img.convolveImg(kernel=numpy.ones((2*increase_radius+1, 2*increase_radius+1)))
                    img_original._mask = mask_new
                if binary_closure:
                    bmask = img_original._mask > 0
                    bc_mask = numpy.zeros(bmask.shape, dtype=img_original._mask.dtype)
                    for ang in [20, 45, 70, 90, 110, 135, 160]:
                        # leave out the dispersion direction (0 degrees), see DESI, Guy et al., ApJ, 2023, 165, 144
                        lse = LinearSelectionElement(11, 11, ang)
                        bc_mask = bc_mask | ndimage.binary_closing(bmask, structure=lse.se)
                    img_original._mask = bc_mask
                    if verbose:
                        log.info(f'  Total number after binary closing: {numpy.sum(bc_mask)} pixels')

                # replace possible corrput pixel with median for final output
                out = img_original.replaceMaskMedian(box_x, box_y, replace_error=replace_error)
            else:
                out.replace_subselect(select, mask=True)  # set the new mask
                out = out.replaceMaskMedian(box_x, box_y, replace_error=None)  # replace possible corrput pixel with median

        if inplace:
            self._data = out._data
            if self._error is None:
                self._error = out._error
            else:
                self._error += out._error
            if self._mask is None:
                self._mask = out._mask
            else:
                self._mask |= out._mask
        else:
            return out

    def getIndividualFrames(self):
        return self._individual_frames

    def setIndividualFrames(self, images):
        self._individual_frames = Table(names=["TILEID", "MJD", "EXPNUM", "SPEC", "CAMERA", "EXPTIME"], dtype=(int, int, int, str, str, float))
        for img in images:
            self._individual_frames.add_row([
                img._header.get("TILEID", 11111),
                img._header.get("MJD"),
                img._header.get("EXPOSURE"),
                img._header.get("SPEC"),
                img._header.get("CCD"),
                img._header.get("EXPTIME"),
            ])

    def getSlitmap(self):
        return self._slitmap

    def setSlitmap(self, slitmap):
        if isinstance(slitmap, pyfits.BinTableHDU):
            self._slitmap = Table(slitmap.data)
        else:
            self._slitmap = slitmap

    def eval_fiber_model(self, trace_cent, trace_width=None, trace_amp=None, columns=None, column_width=None):
        """Returns the evaluated fiber model from the given fiber centroids, widths and amplitudes

        Parameters
        ----------
        trace_cent : TraceMask
            the fiber trace centroids
        trace_width : TraceMask
            the fiber trace widths, defaults to None
        trace_amp : TraceMask
            the fiber trace amplitudes, defaults to None
        nrows : int
            number of rows in the image, defaults to 4080
        columns : list
            list of columns to evaluate the continuum model, defaults to None
        column_width : int
            number of columns to add around the given columns, defaults to None

        Returns
        -------
        Image
            the evaluated continuum model
        Image
            the ratio of the model to the original image
        """
        if trace_width is None or trace_amp is None or trace_cent is None:
            raise ValueError(f"nothing to do, with provided fiber trace information {trace_cent = } {trace_width = }, {trace_amp = }")

        if isinstance(trace_width, (int, float, numpy.float32)):
            trace_width = TraceMask(data=numpy.ones_like(trace_cent._data) * trace_width, mask=numpy.zeros_like(trace_cent._data, dtype=bool))
        elif isinstance(trace_width, TraceMask):
                pass
        else:
            raise ValueError("trace_width must be a TraceMask instance or an int/float")

        if isinstance(trace_amp, (int, float, numpy.float32)):
            trace_amp = TraceMask(data=numpy.ones_like(trace_cent._data) * trace_amp, mask=numpy.zeros_like(trace_cent._data, dtype=bool))
        elif isinstance(trace_amp, TraceMask):
                pass
        else:
            raise ValueError("trace_amp must be a TraceMask instance or an int/float")

        if columns is None:
            columns = numpy.arange(trace_cent._data.shape[1])
        else:
            columns = _fill_column_list(columns, column_width)

        # initialize the continuum model
        nrows = self._dim[0]
        ncols = self._dim[1]
        model = Image(data=numpy.zeros((nrows, ncols)), mask=numpy.ones((nrows, ncols), dtype=bool))
        model._mask[:, columns] = False

        # evaluate continuum model
        y_axis = numpy.arange(nrows)
        for icolumn in tqdm(columns, desc="evaluating fiber model", unit="column", ascii=True):
            pars = (trace_amp._data[:, icolumn], trace_cent._data[:, icolumn], trace_width._data[:, icolumn] / 2.354)
            model._data[:, icolumn] = gaussians(pars=pars, x=y_axis)

        return model, model / self


def loadImage(
    infile,
    extension_data=None,
    extension_mask=None,
    extension_error=None,
    extension_frames=None,
    extension_slitmap=None,
    extension_header=0,
):
    image = Image()
    image.loadFitsData(
        infile,
        extension_data=extension_data,
        extension_mask=extension_mask,
        extension_error=extension_error,
        extension_frames=extension_frames,
        extension_slitmap=extension_slitmap,
        extension_header=extension_header,
    )

    return image


def glueImages(images, positions):
    """
    Merge several images into a new Image. The positions are defined as colum and row positions (single integers).
    The positions of the subimages need to be regular and must have the right shape to fully fill the rectangle of the final image.
    The header from the first subimages is used as the header for the combined Image.

    Parameters
    --------------
    images : list of Image
        A list of data_model.Image objects
    positions : list of strings
        A list of strings to indicate the position of the subimage within the mosaic of subimages for the combined image.
        The string contains of two digits indicating the column and row position, where '00' corresponds to the lower left corner of the combined image.

    Example
    ------------
    >>> from lvmdrp.core import data_model
    >>> full_Image = data_model.Image()
    >>> full_Image.mergeImages((Img1,Img2,Img3,Img4),('00,'01','10','11'))
    """
    # create list of row and column position
    pos_x = []
    pos_y = []
    for i in range(len(images)):
        pos_x.append(int(positions[i][0]))
        pos_y.append(int(positions[i][1]))
    pos_x = numpy.array(pos_x)
    pos_y = numpy.array(pos_y)
    # number of rows and column to position the subimages
    max_x = max(pos_x)
    max_y = max(pos_y)

    # merge the subimages along the rows and then on the columns
    idx = numpy.arange(len(images))
    columns = []
    for i in range(max_x + 1):
        rows = []
        for j in range(max_y + 1):
            select = numpy.logical_and(pos_x == i, pos_y == j)  # select images
            rows.append(images[idx[select][0]]._data)  # add to row list
        columns.append(
            numpy.concatenate(rows)
        )  # combine images for each row and store to column list
    full_CCD_data = numpy.concatenate(columns, axis=1)  # create full CCD

    # same process but for the error data
    if images[0]._error is not None:
        columns = []
        for i in range(max_x + 1):
            rows = []
            for j in range(max_y + 1):
                select = numpy.logical_and(pos_x == i, pos_y == j)
                rows.append(images[idx[select][0]]._error)
            columns.append(numpy.concatenate(rows))
        full_CCD_error = numpy.concatenate(columns, axis=1)
    else:
        full_CCD_error = None

    # same process but for the bad pixel mask
    if images[0]._mask is not None:
        columns = []
        for i in range(max_x + 1):
            rows = []
            for j in range(max_y + 1):
                select = numpy.logical_and(pos_x == i, pos_y == j)
                rows.append(images[idx[select]]._mask)
            columns.append(numpy.concatenate(rows))
        full_CCD_mask = numpy.concatenate(columns, axis=1)
    else:
        full_CCD_mask = None
    # ingest the combined data to the object attribute
    out_image = Image(
        data=full_CCD_data,
        error=full_CCD_error,
        mask=full_CCD_mask,
        header=images[0].getHeader(),
        individual_frames=images[0].getIndividualFrames(),
        slitmap=images[0].getSlitmap(),
    )

    return out_image


def combineImages(
    images: List[Image],
    method: str = "median",
    k: int = 3,
    normalize: bool = True,
    normalize_percentile: int = 75,
    background_subtract: bool = False,
    background_sections: List[str] = None,
    replace_with_nan: bool = True,
):
    """
    Combines several image to a single one according to a certain average methods

    Parameters
    --------------
    images : list of Image
        A list of data_model.Image objects
    method : string
        Method used to average the given images, available are 'median', 'sum', 'mean', 'nansum','clipped_mean'
    k : float
        Only used for the clipped_mean method. Only values within k*sigma around the median value are averaged.
    """
    # TODO: I think medians are fine, as long as we are careful of subtracting a
    # background, and scaling images robustly (i.e. with outlier rejection).
    # You might want to do different things when dealing with different types of frames.
    # We should discuss this.,

    # creates an empty empty array to store the images in a stack
    dim = images[0].getDim()
    stack_image = numpy.zeros((len(images), dim[0], dim[1]), dtype=float)
    stack_error = numpy.zeros((len(images), dim[0], dim[1]), dtype=float)
    stack_mask = numpy.zeros((len(images), dim[0], dim[1]), dtype=bool)

    # load image data in to stack
    for i in range(len(images)):
        stack_image[i, :, :] = images[i].getData()

        # read error image if not a bias image
        if images[0]._header["IMAGETYP"] != "bias" and images[0]._error is not None:
            stack_error[i, :, :] = images[i].getError()

        # read pixel mask image
        if images[i]._mask is not None:
            stack_mask[i, :, :] = images[i].getMask()

    # mask invalid values
    stack_image[stack_mask] = numpy.nan
    stack_error[stack_mask] = numpy.nan

    if background_subtract:
        quad_sections = images[0].getHdrValues("AMP? TRIMSEC")
        stack_image, _, _, _ = _bg_subtraction(
            images=stack_image,
            quad_sections=quad_sections,
            bg_sections=background_sections,
        )

    if normalize:
        # plot distribution of pixels (detect outliers e.g., CR)
        # select pixels that are exposed
        # calculate the median of the selected pixels
        # scale illuminated pixels to a common scale, for the whole image
        stack_image, _ = _percentile_normalize(stack_image, normalize_percentile)

    # combine the images according to the selected method
    if method == "median":
        new_image = bn.nanmedian(stack_image, 0)
        new_error = numpy.sqrt(bn.nanmedian(stack_error ** 2, 0))
    elif method == "sum":
        new_image = bn.nansum(stack_image, 0)
        new_error = numpy.sqrt(bn.nansum(stack_error ** 2, 0))
    elif method == "mean":
        new_image = bn.nanmean(stack_image, 0)
        new_error = numpy.sqrt(bn.nanmean(stack_error ** 2, 0))
    elif method == "clipped_median":
        median = bn.nanmedian(stack_image, 0)
        rms = bn.nanstd(stack_image, 0)
        # select pixels within given sigma limits around the median
        select = (stack_image < median + k * rms) & (
            stack_image > median - k * rms
        )
        # compute the number of good pixels
        good_pixels = bn.nansum(select, 0).astype(bool)
        # set all bad pixel to 0 to compute the mean
        # TODO: make this optional, by default not replacement
        stack_image[:, ~good_pixels] = 0
        new_image = bn.nansum(stack_image, 0) / good_pixels

    # return new image and error to normal array
    new_mask = numpy.all(stack_mask, 0)

    # mask bad pixels
    new_mask = new_mask | numpy.isnan(new_image) | numpy.isnan(new_error)

    # define new header
    if images[0]._header is not None:
        new_header = images[0]._header

        nexp = len(images)
        new_header["ISMASTER"] = (True, "Is this a combined (master) frame")
        new_header["NFRAMES"] = (nexp, "Number of exposures combined")
        new_header["STATCOMB"] = (method, "Statistic used to combine images")

        # add combined lamps to header
        if images[0]._header["IMAGETYP"] == "flat":
            lamps = CON_LAMPS
        elif images[0]._header["IMAGETYP"] == "arc":
            lamps = ARC_LAMPS
        else:
            lamps = []

        if lamps:
            new_lamps = set()
            for image in images:
                for lamp in lamps:
                    if image._header.get(lamp) == "ON":
                        new_lamps.add(lamp)
            for lamp in new_lamps:
                new_header[lamp] = "ON"
    else:
        new_header = None

    # create combined image
    combined_image = Image(
        data=new_image, error=new_error, mask=new_mask, header=new_header, slitmap=images[0].getSlitmap()
    )
    # update masked pixels if needed
    if replace_with_nan:
        combined_image.apply_pixelmask()

    # add metadata of individual images
    combined_image.setIndividualFrames(images)

    return combined_image
