from copy import deepcopy as copy
from multiprocessing import Pool, cpu_count
import warnings

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from functools import partial
from typing import List
from tqdm import tqdm

import os
import numpy
import itertools as it
import bottleneck as bn
import pandas as pd
from astropy.table import Table
from astropy.io import fits as pyfits
from astropy.modeling import fitting, models
from astropy.stats.biweight import biweight_location, biweight_scale
from astropy.visualization import simple_norm
from scipy import ndimage
from scipy import interpolate

from lvmdrp import log
from lvmdrp.core.constants import CON_LAMPS, ARC_LAMPS, LVM_NBLOCKS, LVM_BLOCKSIZE
from lvmdrp.core.plot import plt, plot_fiber_residuals
from lvmdrp.core.fit_profile import gaussians, PROFILES
from lvmdrp.core.apertures import Apertures
from lvmdrp.core.header import Header
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.spectrum1d import Spectrum1D, _normalize_peaks, _fiber_cc_match, _cross_match, _spec_from_lines, _align_fiber_blocks

from lvmdrp.external.fast_median import fast_median_filter_2d

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
        stat = bn.nanmedian
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


def interpolate_mask(x, y, mask, kind="linear", fill_value=0):
    """
    :param x, y: numpy arrays, samples and values
    :param mask: boolean mask, True for masked values
    :param method: interpolation method, one of linear, nearest,
    nearest-up, zero, slinear, quadratic, cubic, previous, or next.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    if not numpy.any(mask):
        return y
    known_x, known_v = x[~mask], y[~mask]
    missing_x = x[mask]
    missing_idx = numpy.where(mask)

    f = interpolate.interp1d(known_x, known_v, kind=kind, fill_value=fill_value, bounds_error=False)
    yy = y.copy()
    yy[missing_idx] = f(missing_x)

    return yy


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

    def _propagate_error(self, other, operation):
        """ Error propagation for different operations. """
        if self._error is None and getattr(other, "error", None) is None:
            return None

        err1 = self._error if self._error is not None else 0
        err2 = other._error if isinstance(other, Image) and other._error is not None else 0

        if operation in ('add', 'sub'):
            return numpy.sqrt(err1**2 + err2**2)
        elif operation == 'mul':
            return numpy.sqrt((err1 * other._data)**2 + (self._data * err2)**2)
        elif operation == 'div':
            return numpy.sqrt((err1 / other._data)**2 + (self._data * err2 / other._data**2)**2)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _apply_operation(self, other, op, op_name):
        if isinstance(other, Image):
            new_data = op(self._data, other._data)
            new_error = self._propagate_error(other, op_name)
            new_mask = numpy.logical_or(self._mask, other._mask) if self._mask is not None and other._mask is not None else None
        elif isinstance(other, numpy.ndarray) or numpy.isscalar(other):
            new_data = op(self._data, other)
            new_error = (op(self._error, other) if self._error is not None else None)
            new_mask = self._mask
        else:
            return NotImplemented

        new_image = copy(self)
        new_image.setData(data=new_data, error=new_error, mask=new_mask)
        return new_image

    def __add__(self, other):
        return self._apply_operation(other, numpy.add, 'add')

    def __sub__(self, other):
        return self._apply_operation(other, numpy.subtract, 'sub')

    def __mul__(self, other):
        return self._apply_operation(other, numpy.multiply, 'mul')

    def __truediv__(self, other):
        return self._apply_operation(other, numpy.divide, 'div')
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

    def measure_fiber_shifts(self, ref_image, trace_cent, columns=[500, 1000, 1500, 2000, 2500, 3000], column_width=25, shift_range=[-5,5], axs=None):
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

        # unpack axes
        axs_cc, axs_fb = axs

        # calculate shift guess along central wide column
        s1 = bn.nanmedian(ref_data[50:-50,2000-500:2000+500], axis=1)
        s2 = bn.nanmedian(self._data[50:-50,2000-500:2000+500], axis=1)
        # fig_guess, axs_guess = plt.subplots(nrows=2, ncols=1, layout="constrained")
        # fig_guess.suptitle("Fiber block cross-correlation match")
        guess_shift = _align_fiber_blocks(s1, s2, axs=None)

        if numpy.abs(guess_shift) > 6:
            log.warning(f"measuring guess fiber thermal shift too large {guess_shift = } pixels, setting guess shift to zero")
            guess_shift = 0
        else:
            log.info(f"measured guess fiber thermal shift {guess_shift = } pixels")

        shifts = numpy.zeros(len(columns))
        select_blocks = [9]
        for j,c in enumerate(columns):
            # collapse columns
            s1 = bn.nanmedian(ref_data[50:-50,c-column_width:c+column_width], axis=1)
            s2 = bn.nanmedian(self._data[50:-50,c-column_width:c+column_width], axis=1)
            # clean remaining NaNs from masked rows
            s2 = numpy.nan_to_num(s2)
            snr = numpy.sqrt(s2)
            median_snr = bn.nanmedian(snr)

            min_snr = 1.0
            if median_snr <= min_snr:
                comstr = f"low SNR (<={min_snr}) for thermal shift at column {c}: {median_snr:.4f}, assuming = NaN"
                log.warning(comstr)
                self.add_header_comment(comstr)
                shifts[j] = numpy.nan
                continue

            _, shifts[j], _ = _fiber_cc_match(s1, s2, guess_shift, shift_range, gauss_window=[-3,3], min_peak_dist=5.0, ax=axs_cc[j])

            blocks_pos = numpy.asarray(numpy.split(trace_cent._data[:, c], 18))[select_blocks]
            blocks_bounds = [(int(bpos.min())-10, int(bpos.max())+10) for bpos in blocks_pos]

            for i, (bmin, bmax) in enumerate(blocks_bounds):
                x = numpy.arange(bmax-bmin) + i*(bmax-bmin) + 10
                y_model = bn.nanmedian(ref_data[bmin:bmax, c-column_width:c+column_width], axis=1)
                y_data = bn.nanmedian(self._data[bmin:bmax, c-column_width:c+column_width], axis=1)
                y_data, y_model, _, _, _, _ = _normalize_peaks(y_data, y_model, min_peak_dist=5.0)
                # y_data, _, _ = _normalize_peaks(y_data, min_peak_dist=5.0)
                axs_fb[j].step(x, y_data, color="0.2", lw=1.5, label="data" if i == 0 else None)
                axs_fb[j].step(x, y_model, color="tab:blue", lw=1, label="model" if i == 0 else None)
                # axs_fb[j].step(x+shifts[j], numpy.interp(x+shifts[j], x, y_model), color="tab:red", lw=1, label="corr. model" if i == 0 else None)
                axs_fb[j].step(x, numpy.interp(x, x+shifts[j], y_model), color="tab:red", lw=1, label="corr. model" if i == 0 else None)
            axs_fb[j].set_title(f"measured shift {shifts[j]:.4f} pixel @ column {c} with SNR = {median_snr:.2f}")
            axs_fb[j].set_ylim(-0.05, 1.3)
        axs_fb[0].legend(loc=1, frameon=False, ncols=3)

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
            camera = self.getHdrValue("CCD").upper()
            exptime = self.getHdrValue("EXPTIME")
            gains = self.getHdrValue(f"{camera} AMP? {gain_field}")
            sects = self.getHdrValue(f"{camera} AMP? TRIMSEC")
            n_amp = len(gains)
            for i in range(n_amp):
                if current == "adu" and to == "electron":
                    factor = gains[i]
                elif current == "adu" and to == "electron / s":
                    factor = gains[i] / exptime
                elif current == "electron" and to == "adu":
                    factor = 1 / gains[i]
                elif current == "electron" and to == "electron / s":
                    factor = 1 / exptime
                elif current == "electron / s" and to == "adu":
                    factor = gains[i] * exptime
                elif current == "electron / s" and to == "electron":
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

    def get_exposed_std(self, ref_column, fiber_pos=None, snr_threshold=5, trust_errors=True, ax=None):
        if self._slitmap is None:
            raise ValueError(f"Slitmap attribute `self._slitmap` has to be defined: {self._slitmap = }")

        if fiber_pos is None:
            fiber_pos = self.match_reference_column(ref_column)

        slitmap = self._slitmap[self._slitmap["spectrographid"] == int(self._header["SPEC"][-1])]
        spec_select = slitmap["telescope"] == "Spec"

        ids_std = slitmap[spec_select]["orig_ifulabel"]
        pos_std = fiber_pos[spec_select].round().astype("int")
        idx_std = numpy.arange(pos_std.size)

        expnum = self._header["EXPOSURE"]
        column = self.getSlice(ref_column, axis="Y")
        snr = (column._data / (column._error if trust_errors else numpy.sqrt(column._data)))
        snr_med = biweight_location(snr[fiber_pos.round().astype("int")], ignore_nan=True)
        snr_std = biweight_scale(snr[fiber_pos.round().astype("int")], ignore_nan=True)
        snr_std_med = biweight_location(snr[pos_std], ignore_nan=True)
        snr_std_std = biweight_scale(snr[pos_std], ignore_nan=True)

        ax.set_title(f"{expnum = }", loc="left")
        ax.axhspan(snr_med-snr_std, snr_med+snr_std, lw=0, fc="0.7", alpha=0.5)
        ax.axhline(snr_med, lw=1, color="0.7")
        ax.axhline(snr_std_med+snr_threshold*snr_std_std, ls="--", lw=1, color="tab:red")
        ax.axhline(snr_std_med, lw=1, color="0.7")
        ax.bar(idx_std, snr[pos_std], hatch="///////", lw=0, ec="tab:blue", fc="none", zorder=999)
        ax.set_xticks(idx_std)
        ax.set_xticklabels(ids_std)
        ax.text(-0.7, snr_med, "Global median SNR", ha="left", va="bottom")
        ax.text(-0.7, snr_std_med, "Stds. median SNR", ha="left", va="bottom")
        ax.text(-0.7, snr_std_med+snr_threshold*snr_std_std, "Exposed threshold", ha="left", va="bottom", color="tab:red")

        # select standard fiber exposed if any
        select_std = numpy.abs(snr[pos_std] - snr_std_med) / snr_std_std > snr_threshold
        exposed_std = ids_std[select_std]
        if select_std.sum() > 1:
            exposed_std_ = exposed_std[numpy.argmax(snr[pos_std[select_std]])]
            warnings.warn(f"More than one standard fiber selected in {expnum = }: {','.join(exposed_std)}, selecting highest SNR: '{exposed_std_}'")
            exposed_std = exposed_std_
        elif select_std.sum() > 0:
            exposed_std = exposed_std[0]
        else:
            return None, snr, snr_std, snr_std_med, snr_std_std

        # highlight exposed fiber in plot
        select_exposed = ids_std == exposed_std
        ax.bar(idx_std[select_exposed], snr[pos_std][select_exposed], hatch="///////", lw=0, ec="tab:red", fc="none", zorder=999)

        return exposed_std, snr, snr_std, snr_std_med, snr_std_std

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
                        self._slitmap = Table.read(hdu[i])

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
                self._slitmap = Table.read(hdu[extension_slitmap])

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
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
            if self._header is not None:
                hdu[0].header = self.getHeader()
                hdu[0].header['DRPVER'] = drpver
                hdu[0].update_header()
            hdu[0].scale(bzero=0, bscale=1)

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
        if self._data is not None:
            new_data = self._data.repeat(2, axis=0).repeat(2, axis=1)
        else:
            new_data = None
        if self._error is not None:
            new_error = self._error.repeat(2, axis=0).repeat(2, axis=1)
        else:
            new_error = None
        if self._mask is not None:
            new_mask = numpy.zeros(new_data.shape, dtype="bool")
        else:
            new_mask = None

        # create index array of the new
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

    def medianImg(self, size, propagate_error=False):
        """return median filtered image with the given kernel size

        Parameters
        ----------
        size : tuple
            2-value tuple for the size of the median box
        propagate_error : bool, optional
            whether to propagate the error or not, by default False

        Returns
        -------
        lvmdrp.core.image.Image
            median filtered image
        """
        new_data = copy(self._data)
        new_error = copy(self._error)

        new_data = fast_median_filter_2d(new_data, size)
        if propagate_error and new_error is not None:
            new_error = numpy.sqrt(fast_median_filter_2d(new_error ** 2, size))

        image = Image(data=new_data, error=new_error, mask=self._mask, header=self._header,
                      origin=self._origin, individual_frames=self._individual_frames, slitmap=self._slitmap)
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

    def fitSpline(self, axis="y", degree=3, smoothing=0, use_weights=False, clip=None, interpolate_missing=True, display_plots=False):
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
        display_plot: bool, optional
            display plots for spline fitting

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
        colors = plt.cm.coolwarm(numpy.linspace(0, 1, self._dim[1]))
        if display_plots:
            fig, axs = plt.subplots(2, 1, figsize=(15,5), sharex=True, layout="constrained")
            axs[1].axhline(ls=":", color="0.7")
            axs[1].set_xlabel("Y axis (pix)")
            axs[0].set_ylabel("Counts (e-)")
            axs[1].set_ylabel("(model - data) / data")
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
                new_masked_pixels.append(bn.nanmean(masked_pixels[group]))
                new_data.append(bn.nanmedian(data[group]))
                new_vars.append(bn.nanmean(vars[group]))
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

            if display_plots:
                if i % 100 == 0:
                    axs[0].plot(pixels, models[:, i], color=colors[i])
                    axs[0].plot(masked_pixels, data, "o", ms=7, color=colors[i])
                axs[1].plot(masked_pixels, interpolate.splev(masked_pixels, spline_pars) / data - 1, "o", ms=7, color=colors[i])

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

    def enhance(self, median_box=None, coadd=None, trust_errors=True, apply_mask=True, replace_errors=numpy.inf):

        img = copy(self)
        img.setData(data=0.0, error=replace_errors, select=img._mask)

        if median_box is not None:
            img = img.replaceMaskMedian(*median_box, replace_error=None)
            img._data = numpy.nan_to_num(img._data)
            img = img.medianImg(median_box, propagate_error=True)

        # coadd images along the dispersion axis to increase the S/N of the peaks
        if coadd is not None:
            coadd_kernel = numpy.ones((1, coadd), dtype="uint8")
            img = img.convolveImg(coadd_kernel)
            # counts_threshold = counts_threshold * coadd

        # mask overscan columns
        img._mask[:, :3] = True
        img._mask[:, -3:] = True
        if apply_mask:
            img.apply_pixelmask()

        # handle invalid error values
        if not trust_errors:
            img._error = numpy.sqrt(numpy.abs(img._data))
        img._error[img._mask|(img._error<=0)|(numpy.isnan(img._error))] = replace_errors

        return img

    def _get_bins(self, data, error, mask, bins, x_bounds=(None,None), y_bounds=(None,None), x_nbound=11, y_nbound=3):

        x_nbins, y_nbins = bins
        x_range, y_range = (0, self._dim[1]), (0, self._dim[0])
        x_pixels = numpy.arange(x_range[1], dtype="int")
        y_pixels = numpy.arange(y_range[1], dtype="int")
        left = right = bottom = top = 0

        # set left and right boundaries if given (offset by 3 pixels to account for pre-scan regions)
        l_bound, r_bound = x_bounds
        if l_bound is not None:
            left = x_nbound
        if r_bound is not None:
            right = x_nbound

        # set top and bottom boundaries if given
        b_bound, t_bound = y_bounds
        if b_bound is not None:
            bottom = y_nbound
        if t_bound is not None:
            top = y_nbound

        x_bins = numpy.histogram_bin_edges(x_pixels, bins=x_nbins, range=(x_range[0]+left,x_range[1]-right))
        y_bins = numpy.histogram_bin_edges(y_pixels, bins=y_nbins, range=(y_range[0]+bottom,y_range[1]-top))

        # add extra bins
        if l_bound is not None:
            x_bins = numpy.insert(x_bins, 0, 0.0)
        if r_bound is not None:
            x_bins = numpy.append(x_bins, self._dim[1])
        if b_bound is not None:
            y_bins = numpy.insert(y_bins, 0, 0)
        if t_bound is not None:
            y_bins = numpy.append(y_bins, self._dim[0])

        # offset by 3 pixels to account for pre-scan regions
        if isinstance(l_bound, (float, int)):
            data[:, (x_nbound)] = l_bound
            error[:, (x_nbound)] = 0.1
            mask[:, (x_nbound)] = False
        elif l_bound == "data":
            pass
        if isinstance(r_bound, (float, int)):
            data[:, -(x_nbound):] = r_bound
            error[:, -(x_nbound):] = 0.1
            mask[:, -(x_nbound):] = False
        elif r_bound == "data":
            pass

        if isinstance(b_bound, (float, int)):
            data[:y_nbound, :] = b_bound
            error[:y_nbound, :] = 0.1
            mask[:y_nbound, :] = False
        elif b_bound == "data":
            pass
        if isinstance(b_bound, (float, int)):
            data[-y_nbound, :] = t_bound
            error[-y_nbound, :] = 0.1
            mask[-y_nbound, :] = False
        elif t_bound == "data":
            pass

        return data, error, mask, x_bins, y_bins

    def histogram(self, bins, nsigma=5.0, stat=bn.nanmedian, x_bounds=(None,None), y_bounds=(None,None), x_nbound=3, y_nbound=3, clip=None, use_mask=True):

        x_nbins, y_nbins = bins
        x_pixels = numpy.arange(self._dim[1], dtype="int")
        y_pixels = numpy.arange(self._dim[0], dtype="int")
        X, Y = numpy.meshgrid(x_pixels, y_pixels, indexing="xy")
        xx, yy = X.ravel(), Y.ravel()

        img_data = self._data.copy()
        img_error = numpy.sqrt(self._data).copy()
        img_mask = self._mask.copy()

        img_data, img_error, img_mask, x_bins, y_bins = self._get_bins(
            data=img_data, error=img_error, mask=img_mask,
            bins=bins, x_bounds=x_bounds, x_nbound=x_nbound, y_bounds=y_bounds, y_nbound=y_nbound)

        if use_mask:
            img_data[img_mask] = numpy.nan
            img_error[img_mask] = numpy.nan
        data = img_data.ravel()
        error = img_error.ravel()

        ix = numpy.digitize(xx, x_bins) - 1
        iy = numpy.digitize(yy, y_bins) - 1
        df = pd.DataFrame({'ix': ix, 'iy': iy, 'data': data, 'variance': error**2})
        groups = df.groupby(['ix', 'iy'])

        zscore = groups.data.apply(lambda g: numpy.abs(g.mean() - g) / g.std(), include_groups=False)
        invalid = zscore > nsigma

        data[invalid] = numpy.nan
        error[invalid] = numpy.nan
        img_data = data.reshape(self._dim)
        img_error = error.reshape(self._dim)

        data_binned = groups.data.agg(stat).unstack().to_numpy()
        error_binned = numpy.sqrt(groups.variance.agg(stat).unstack().to_numpy())
        data_binned = data_binned.T
        error_binned = error_binned.T
        if clip is not None and isinstance(clip, tuple) and len(clip) == 2:
            data_binned = numpy.clip(data_binned, *clip)

        x_cent = (x_bins[:-1]+x_bins[1:]) / 2
        y_cent = (y_bins[:-1]+y_bins[1:]) / 2
        x, y = numpy.meshgrid(x_cent, y_cent, indexing="xy")

        return (ix,iy), x_bins, y_bins, x, y, data_binned, error_binned, X, Y, img_data, img_error, data, error

    def fit_spline2d(self, bins, x_bounds=("data","data"), y_bounds=(0.0,0.0), x_nbound=3, y_nbound=3, nsigma=None, clip=None, smoothing=None, use_weights=True, use_mask=True, axs=None):
        """Fits a 2D bivariate spline to the image data, using binned statistics and sigma clipping.

        The image is divided into bins along both axes, and the median value in each bin is computed.
        Outlier bins are rejected based on a sigma threshold. A 2D spline is then fit to the valid bins,
        optionally using inverse variance weights. The resulting smooth background model can be used for
        tasks such as stray light subtraction.

        Parameters
        ----------
        bins : tuple of int
            Number of bins along the (X, Y) axes, e.g., (x_bins, y_bins).
        nsigma : float
            Sigma threshold for clipping outlier bins. If None, no rejection is performed.
        smoothing : float, optional
            Smoothing parameter for the spline fit. If None, the default is used.
        use_weights : bool, optional
            If True, use inverse variance of the binned errors as weights in the spline fit (default: True).
        axs : dict of matplotlib.axes.Axes, optional
            Dictionary of axes for diagnostic plotting (default: None).

        Returns
        -------
        stray_img : Image
            Image object containing the fitted 2D spline model.
        data_binned : numpy.ndarray
            2D array of binned median values used for the fit.
        error_binned : numpy.ndarray
            2D array of binned errors.
        valid_bins : numpy.ndarray
            Boolean mask indicating which bins were used in the fit.
        """
        x_pixels = numpy.arange(self._dim[1])
        y_pixels = numpy.arange(self._dim[0])

        # get 2D histogram
        xybins, x_bins, y_bins, x, y, data_binned, error_binned, X, Y, img_data, img_error, data, error = self.histogram(
            bins=bins, nsigma=nsigma,
            x_bounds=x_bounds, x_nbound=x_nbound,
            y_bounds=y_bounds, y_nbound=y_nbound,
            clip=clip, use_mask=use_mask)
        y_cent = (y_bins[:-1]+y_bins[1:]) / 2
        x_cent = (x_bins[:-1]+x_bins[1:]) / 2
        y_nbins = y_cent.size

        # select valid bins
        valid_bins = numpy.isfinite(data_binned) & numpy.isfinite(error_binned)

        # fit 2D smoothing spline
        tck = interpolate.bisplrep(
            x[valid_bins].ravel(), y[valid_bins].ravel(), data_binned[valid_bins].ravel(),
            w=1.0/error_binned[valid_bins].ravel() if use_weights else None,
            s=smoothing, xb=0, xe=4086, yb=0, ye=4080, eps=1e-8)
        model_data = interpolate.bisplev(x_pixels, y_pixels, tck).T
        if clip is not None and isinstance(clip, tuple) and len(clip) == 2:
            model_data = numpy.clip(model_data, *clip)

        # calculate binned residuals & model systematic errors
        model_binned = interpolate.bisplev(x_cent, y_cent, tck).T
        model_residuals = (model_binned - data_binned) / error_binned

        model_error = interpolate.griddata(
            points=(x[valid_bins].ravel(), y[valid_bins].ravel()), values=model_residuals[valid_bins].ravel(), xi=(X.ravel(), Y.ravel()),
            method="nearest", rescale=True).reshape(self._dim)

        if axs is not None:
            y_pixels = numpy.arange(self._data.shape[0])
            x_pixels = numpy.arange(self._data.shape[1])
            unit = self._header["BUNIT"]
            norm = simple_norm(data=model_data, stretch="asinh")
            im = axs["img"].imshow(model_data, origin="lower", cmap="Greys_r", norm=norm, interpolation="none")
            cbar = plt.colorbar(im, cax=axs["col"], orientation="horizontal")
            cbar.set_label(f"Counts ({unit})", fontsize="small", color="tab:red")
            axs["img"].set_aspect("auto")

            axs["img"].plot(x[valid_bins].ravel(), y[valid_bins].ravel(), "o", mew=0.5, ms=4, mec="tab:blue", mfc="none")

            # CS = axs["img"].contour(X, Y, model_data, levels=numpy.percentile(model_data, q=(25,50,75)), cmap="Greys", linewidths=1)
            # axs["img"].clabel(CS, fontsize=9)

            colors_x = plt.cm.coolwarm(numpy.linspace(0, 1, self._data.shape[0]))
            colors_y = plt.cm.coolwarm(numpy.linspace(0, 1, self._data.shape[1]))
            for iy in y_pixels:
                axs["xma"].plot(x_pixels, model_data[iy], ",", color=colors_x[iy], alpha=0.2)
            axs["xma"].step(x_pixels, numpy.sqrt(bn.nanmedian(self._error**2, axis=0)), lw=1, color="0.8", where="mid")
            for ix in x_pixels:
                axs["yma"].plot(model_data[:, ix], y_pixels, ",", color=colors_y[ix], alpha=0.2)
            axs["yma"].step(numpy.sqrt(bn.nanmedian(self._error, axis=1)), y_pixels, lw=1, color="0.8", where="mid")

            model_ = interpolate.bisplev(x_pixels, y_cent, tck).T
            if clip is not None and isinstance(clip, tuple) and len(clip) == 2:
                model_ = numpy.clip(model_, *clip)
            for i in range(y_nbins):
                data_ = data[xybins[1]==i].reshape((-1,self._dim[1]))
                error_ = error[xybins[1]==i].reshape((-1,self._dim[1]))
                residuals = (model_[i] - data_) / error_
                mu = numpy.nanmean(residuals, axis=0)

                axs["res"][i].set_title(f"Y-bin = [{y_bins[i]:.1f},{y_bins[i+1]:.1f})", fontsize="large", loc="left")
                axs["res"][i].set_ylabel(f"Counts ({unit})", fontsize="large")

                axs["res"][i].errorbar(
                    x_cent[valid_bins[i]], data_binned[i][valid_bins[i]], yerr=error_binned[i][valid_bins[i]],
                    fmt=".", color="tab:red", ecolor="tab:red", lw=1, elinewidth=1)
                ylims = axs["res"][i].get_ylim()
                axs["res"][i].errorbar(
                    x_pixels, bn.nanmean(data_, axis=0), yerr=numpy.sqrt(bn.nanmean(error_**2, axis=0)),
                    fmt=",", color="0.2", ecolor="0.2", elinewidth=0.5, zorder=-1)
                axs["res"][i].plot(x_pixels, model_[i], "-", color="tab:blue")

                f = numpy.abs(ylims).max()*0.03
                axs["res"][i].plot(x_pixels, residuals.T*f, ",", color="0.2")
                axs["res"][i].step(x_pixels, mu*f, "-", color="tab:blue", lw=1, where="mid")
                axs["res"][i].axhline(-f, ls=":", lw=1, color="0.2")
                axs["res"][i].axhline(+f, ls=":", lw=1, color="0.2")
                axs["res"][i].axhline(ls="--", lw=1, color="0.2")
                axs["res"][i].set_ylim(-f*2, ylims[1])

        stray_img = copy(self)
        stray_img.setData(data=model_data, error=model_error, mask=None)

        return stray_img, data_binned, error_binned, valid_bins

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
        guess_heights = numpy.ones_like(ref_centroids) * bn.nanmax(profile._data)
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

    def guess_fibers(self, ref_column=2000, ref_centroids=None, fwhms_guess=2.5,
                     counts_range=[1e3, numpy.inf], centroids_range=[-1.0,+1.0], fwhms_range=[1.0, 3.5],
                     ncolumns=140, mask_fibstatus=1, solver="dogbox"):
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
        columns = numpy.concatenate((numpy.arange(ref_column, 4, -step), numpy.arange(ref_column, self._dim[1]-3, step)))

        # create empty traces mask for the image
        fibers = ref_centroids.size
        dim = self.getDim()
        centroids = TraceMask.create_empty(data_dim=(fibers, dim[1]), samples_columns=sorted(set(columns)))
        centroids._good_fibers = good_fibers
        centroids._mask[good_fibers, :] = False
        centroids.setHeader(self._header.copy())
        centroids.setSlitmap(self._slitmap)
        centroids._header["IMAGETYP"] = "fiber_centroids"
        counts = TraceMask.create_empty(data_dim=(fibers, dim[1]), samples_columns=sorted(set(columns)))
        counts._good_fibers = good_fibers
        counts._mask[good_fibers, :] = False
        counts.setHeader(self._header.copy())
        counts.setSlitmap(self._slitmap)
        counts._header["IMAGETYP"] = "fiber_counts"
        fwhms = TraceMask.create_empty(data_dim=(fibers, dim[1]), samples_columns=sorted(set(columns)))
        fwhms._good_fibers = good_fibers
        fwhms._mask[good_fibers, :] = False
        fwhms.setHeader(self._header.copy())
        fwhms.setSlitmap(self._slitmap)
        fwhms._header["IMAGETYP"] = "fiber_fwhm"

        # set positions of fibers along reference column
        centroids._samples[str(ref_column)] = ref_centroids

        # trace centroids in each column
        iterator = tqdm(enumerate(columns), total=len(columns), desc="measuring fibers", unit="column", ascii=True)
        for i, icolumn in iterator:
            # extract column profile
            img_slice = self.getSlice(icolumn, axis="y")

            # get fiber positions along previous column
            # trace reference column first or skip if already traced
            if icolumn == ref_column:
                if i == 0:
                    cent_guess = centroids._samples[str(icolumn)].data
                else:
                    continue
            else:
                cent_guess = centroids._samples[str(columns[i-1])].data

            # measure fiber positions
            counts_slice, centroids_slice, fwhms_slice, msk_slice = img_slice.measure_fibers_profile(centroids_guess=cent_guess, fwhms_guess=fwhms_guess,
                                                                                                     counts_range=counts_range, centroids_range=centroids_range,
                                                                                                     fwhms_range=fwhms_range, solver=solver)

            counts._samples[str(icolumn)] = counts_slice
            centroids._samples[str(icolumn)] = centroids_slice
            fwhms._samples[str(icolumn)] = fwhms_slice

        return counts, centroids, fwhms

    def _get_fwhms_trace(self, fwhms):
        if isinstance(fwhms, (TraceMask, FiberRows)):
            pass
        elif isinstance(fwhms, (int, float)):
            _ = TraceMask()
            _.createEmpty(data_dim=(LVM_NBLOCKS*LVM_BLOCKSIZE, self._dim[1]), header=self._header, slitmap=self._slitmap)
            _._data[:] = fwhms
            _._mask[:] = False
            _._error = None
            fwhms = _
        else:
            raise TypeError(f"Invalid type for `fwhms_guess`: {type(fwhms)}. Expected either float/int or TraceMask")
        return fwhms

    def _measure_block_fwhms(self, counts, centroids, fwhms_guess, iblock, columns, fwhms_range=[1.0,3.5], nsigma=6, solver="trf", loss="linear", axs=None):
        counts_block = counts.get_block(iblock=iblock)
        centroids_block = centroids.get_block(iblock=iblock)
        fwhms_block = fwhms_guess.get_block(iblock=iblock)

        counts_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        centroids_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        fwhms_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        iterator = tqdm(enumerate(columns), total=len(columns), desc=f"measuring fiber widths in block    {iblock+1:>2d}/{LVM_NBLOCKS}", ascii=True, unit="column")

        axs = axs if axs is not None else {}
        for i, icolumn in iterator:
            img_slice = self.getSlice(icolumn, axis="Y")
            counts_slice, _, _ = counts_block.getSlice(icolumn, axis="Y")
            centroids_slice, _, mask = centroids_block.getSlice(icolumn, axis="Y")
            fwhms_slice, _, _ = fwhms_block.getSlice(icolumn, axis="Y")

            select = ~mask
            if select.sum() == 0:
                continue
            lower = (centroids_slice[select] - nsigma/2.354*fwhms_slice[select]).min()
            upper = (centroids_slice[select] + nsigma/2.354*fwhms_slice[select]).max()
            pixels_selection = (lower <= img_slice._pixels) & (img_slice._pixels <= upper)


            model_block, par_block = img_slice.fitMultiGauss_fixed_counts(
                pixels_selection, counts_slice[select], centroids_slice[select], fwhms_slice[select], fwhms_range=fwhms_range, solver=solver, loss=loss)

            counts, centroids, fwhms = numpy.split(par_block, 3)
            counts_samples[select, i] = counts
            centroids_samples[select, i] = centroids
            fwhms_samples[select, i] = fwhms

            axs_ = axs.get(icolumn)
            if axs_ is not None:
                axs_ = model_block.plot(
                    x=img_slice._pixels[pixels_selection], y=img_slice._data[pixels_selection],
                    sigma=img_slice._error[pixels_selection], mask=img_slice._mask[pixels_selection], axs=axs_)
                # axs_["mod"].vlines(centroids, *axs_["mod"].get_ylim(), lw=1, color="0.7")
                # axs_["res"].vlines(centroids, *axs_["res"].get_ylim(), lw=1, color="0.7")
                axs[icolumn] = axs_

        fwhms = TraceMask.from_samples(data_dim=fwhms_block._data.shape, samples=fwhms_samples, samples_columns=columns)
        return fwhms

    def _measure_block_centroids(self, counts, centroids_guess, fwhms, iblock, columns, centroids_range=[-5,+5], nsigma=6, solver="trf", loss="linear", axs=None):
        counts_block = counts.get_block(iblock=iblock)
        centroids_block = centroids_guess.get_block(iblock=iblock)
        fwhms_block = fwhms.get_block(iblock=iblock)

        counts_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        centroids_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        fwhms_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        iterator = tqdm(enumerate(columns), total=len(columns), desc=f"measuring fiber centroids in block {iblock+1:>2d}/{LVM_NBLOCKS}", ascii=True, unit="column")

        axs = axs if axs is not None else {}
        for i, icolumn in iterator:
            img_slice = self.getSlice(icolumn, axis="Y")
            counts_slice, _, _ = counts_block.getSlice(icolumn, axis="Y")
            centroids_slice, _, mask = centroids_block.getSlice(icolumn, axis="Y")
            fwhms_slice, _, _ = fwhms_block.getSlice(icolumn, axis="Y")

            select = ~mask
            if select.sum() == 0:
                continue
            lower = (centroids_slice[select] - nsigma/2.354*fwhms_slice[select]).min()
            upper = (centroids_slice[select] + nsigma/2.354*fwhms_slice[select]).max()
            pixels_selection = (lower <= img_slice._pixels) & (img_slice._pixels <= upper)

            model_block, par_block = img_slice.fitMultiGauss_centroids(
                pixels_selection, counts_slice[select], centroids_slice[select], fwhms_slice[select], centroids_range=centroids_range, solver=solver, loss=loss)

            counts, centroids, fwhms = numpy.split(par_block, 3)
            counts_samples[select, i] = counts
            centroids_samples[select, i] = centroids
            fwhms_samples[select, i] = fwhms

            axs_ = axs.get(icolumn)
            if axs_ is not None:
                axs_ = model_block.plot(
                    x=img_slice._pixels[pixels_selection], y=img_slice._data[pixels_selection],
                    sigma=img_slice._error[pixels_selection], mask=img_slice._mask[pixels_selection], axs=axs_)
                # axs_["mod"].vlines(centroids, *axs_["mod"].get_ylim(), lw=1, color="0.7")
                # axs_["res"].vlines(centroids, *axs_["res"].get_ylim(), lw=1, color="0.7")
                axs[icolumn] = axs_

        centroids = TraceMask.from_samples(data_dim=centroids_block._data.shape, samples=centroids_samples, samples_columns=columns)
        return centroids

    def _measure_block_counts(self, counts_guess, centroids, fwhms, iblock, columns, counts_range=[1000,numpy.inf], nsigma=6, solver="trf", loss="linear", axs=None):
        counts_block = counts_guess.get_block(iblock=iblock)
        centroids_block = centroids.get_block(iblock=iblock)
        fwhms_block = fwhms.get_block(iblock=iblock)

        counts_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        centroids_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        fwhms_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        iterator = tqdm(enumerate(columns), total=len(columns), desc=f"measuring fiber counts in block    {iblock+1:>2d}/{LVM_NBLOCKS}", ascii=True, unit="column")

        axs = axs if axs is not None else {}
        for i, icolumn in iterator:
            img_slice = self.getSlice(icolumn, axis="Y")
            counts_slice, _, _ = counts_block.getSlice(icolumn, axis="Y")
            centroids_slice, _, mask = centroids_block.getSlice(icolumn, axis="Y")
            fwhms_slice, _, _ = fwhms_block.getSlice(icolumn, axis="Y")

            select = ~mask
            if select.sum() == 0:
                continue
            lower = (centroids_slice[select] - nsigma/2.354*fwhms_slice[select]).min()
            upper = (centroids_slice[select] + nsigma/2.354*fwhms_slice[select]).max()
            pixels_selection = (lower <= img_slice._pixels) & (img_slice._pixels <= upper)

            model_block, par_block = img_slice.fitMultiGauss_fixed_width(
                pixels_selection, counts_slice[select], centroids_slice[select], fwhms_slice[select], counts_range=counts_range, solver=solver, loss=loss)

            counts, centroids, fwhms = numpy.split(par_block, 3)
            counts_samples[select, i] = counts
            centroids_samples[select, i] = centroids
            fwhms_samples[select, i] = fwhms

            axs_ = axs.get(icolumn)
            if axs_ is not None:
                axs_ = model_block.plot(
                    x=img_slice._pixels[pixels_selection], y=img_slice._data[pixels_selection],
                    sigma=img_slice._error[pixels_selection], mask=img_slice._mask[pixels_selection], axs=axs_)
                # axs_["mod"].vlines(centroids, *axs_["mod"].get_ylim(), lw=1, color="0.7")
                # axs_["res"].vlines(centroids, *axs_["res"].get_ylim(), lw=1, color="0.7")
                axs[icolumn] = axs_

        counts = TraceMask.from_samples(data_dim=counts_block._data.shape, samples=counts_samples, samples_columns=columns)
        return counts

    def _measure_block_alphas(self, counts, centroids, fwhms, alphas_guess, iblock, columns, alphas_range=[-1.0,+1.0], nsigma=6, solver="trf", loss="linear", axs=None):
        counts_block = counts.get_block(iblock=iblock)
        centroids_block = centroids.get_block(iblock=iblock)
        fwhms_block = fwhms.get_block(iblock=iblock)
        alphas_block = alphas_guess.get_block(iblock=iblock)

        counts_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        centroids_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        fwhms_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        alphas_samples = numpy.full((centroids_block._fibers, columns.size), numpy.nan)
        iterator = tqdm(enumerate(columns), total=len(columns), desc=f"measuring fiber alphas in block    {iblock+1:>2d}/{LVM_NBLOCKS}", ascii=True, unit="column")

        axs = axs if axs is not None else {}
        for i, icolumn in iterator:
            img_slice = self.getSlice(icolumn, axis="Y")
            counts_slice, _, _ = counts_block.getSlice(icolumn, axis="Y")
            centroids_slice, _, mask = centroids_block.getSlice(icolumn, axis="Y")
            fwhms_slice, _, _ = fwhms_block.getSlice(icolumn, axis="Y")
            alphas_slice, _, _ = alphas_block.getSlice(icolumn, axis="Y")

            select = ~mask
            if select.sum() == 0:
                continue
            lower = (centroids_slice[select] - nsigma/2.354*fwhms_slice[select]).min()
            upper = (centroids_slice[select] + nsigma/2.354*fwhms_slice[select]).max()
            pixels_selection = (lower <= img_slice._pixels) & (img_slice._pixels <= upper)

            model_block, par_block = img_slice.fitMultiGauss_alphas(
                pixels_selection, counts_slice[select], centroids_slice[select], fwhms_slice[select], alphas_slice[select], alphas_range=alphas_range, solver=solver, loss=loss)

            counts, centroids, fwhms, alphas = numpy.split(par_block, 4)
            counts_samples[select, i] = counts
            centroids_samples[select, i] = centroids
            fwhms_samples[select, i] = fwhms
            alphas_samples[select, i] = alphas

            axs_ = axs.get(icolumn)
            if axs_ is not None:
                axs_ = model_block.plot(
                    x=img_slice._pixels[pixels_selection], y=img_slice._data[pixels_selection],
                    sigma=img_slice._error[pixels_selection], mask=img_slice._mask[pixels_selection], axs=axs_)
                # axs_["mod"].vlines(centroids, *axs_["mod"].get_ylim(), lw=1, color="0.7")
                # axs_["res"].vlines(centroids, *axs_["res"].get_ylim(), lw=1, color="0.7")
                axs[icolumn] = axs_

        alphas = TraceMask.from_samples(data_dim=alphas_block._data.shape, samples=alphas_samples, samples_columns=columns)
        return alphas

    def measure_fiber_block(self, profile, traces_guess, traces_fixed, iblock, columns, bounds, measuring_conf, npixels=4, oversampling_factor=50, axs=None):

        guess_block = {name: traces_guess[name].get_block(iblock) for name in traces_guess}
        fixed_block = {name: traces_fixed[name].get_block(iblock) for name in traces_fixed}
        free_names = list(traces_guess.keys())
        fixed_names = list(traces_fixed.keys())

        models = {icolumn: [] for icolumn in columns}
        samples = {name: numpy.full((block._fibers,columns.size), numpy.nan) for name, block in guess_block.items()}
        errors = copy(samples)

        # TODO: implement updating guess with previous fit

        axs = axs if axs is not None else {}
        iterator = tqdm(enumerate(columns), total=len(columns), desc=f"measuring {free_names} with fixed {fixed_names} @ block {iblock+1:>2d}/{LVM_NBLOCKS}", ascii=True, unit="column")
        for i, icolumn in iterator:
            guess = {name: guess_block[name].getSlice(icolumn, axis="Y")[0] for name in guess_block}
            fixed = {name: fixed_block[name].getSlice(icolumn, axis="Y")[0] for name in fixed_block}
            img_slice = self.getSlice(icolumn, axis="Y")

            model_column, fitted_pars, fitted_errs = img_slice.fit_gaussians(
                guess, fixed, bounds, profile=profile, fitting_params=measuring_conf, npixels=npixels, oversampling_factor=oversampling_factor)
            models[icolumn] = model_column

            axs_column = axs.get(icolumn)
            if axs_column is not None:
                centroids = guess.get("centroids", fixed.get("centroids"))
                lower = numpy.nanmin(centroids - npixels)
                upper = numpy.nanmax(centroids + npixels)
                pixels_selection = (lower <= img_slice._wave) & (img_slice._wave <= upper)
                axs_column = model_column.plot(
                    x=img_slice._pixels[pixels_selection], y=img_slice._data[pixels_selection],
                    sigma=img_slice._error[pixels_selection], mask=img_slice._mask[pixels_selection], axs=axs_column)
                axs[icolumn] = axs_column

            for name in fitted_pars:
                samples[name][:, i] = fitted_pars[name]
                errors[name][:, i] = fitted_errs[name]

        traces = {}
        for name, block in guess_block.items():
            traces[name] = TraceMask.from_samples(
                data_dim=block._data.shape, samples=samples[name], samples_error=errors[name], samples_columns=columns, header=guess_block[name]._header, slitmap=guess_block[name]._slitmap)
        return traces

    def iterative_block_trace(self, profile, guess_traces, fixed_traces, iblock, columns, bounds, measuring_conf, smoothing_conf, npixels=4, oversampling_factor=50, niter=10, axs=None):
        def _set_plot_alphas(axs, niter_done):
            if axs is None or niter_done < 2:
                return
            alphas = numpy.linspace(0.1, 1.0, niter_done-1, endpoint=False)
            for _, axs_column in axs.items():
                for key in axs_column:
                    lines = numpy.asarray(axs_column[key].get_lines())
                    lines = numpy.split(lines, niter_done)
                    if key == "mod":
                        # modify only versions of the model (oversampled, pixelated, final)
                        nlast = 3
                    elif key == "res":
                        # modify only residuals lines (last one)
                        nlast = 1

                    lines_last = lines.pop()
                    [[(line.set_alpha((line.get_alpha() or 1.0)*alpha), line.set_linewidth(1.0)) for line in lines[i][-nlast:]] for i, alpha in enumerate(alphas)]
                    [line.set_linewidth(1.5) for line in lines_last[1:][-nlast:]]

                    # [[line.set_visible(False) for line in lines[i][:nlast]] for i in range(alphas.size)]
                    # [line.set_visible(False) for line in lines_last[1:][:nlast]]
        def _block_cycle(parnames, niter):
            npars = len(parnames)
            names_cycle = it.chain.from_iterable(it.repeat(parnames, niter))
            return ((i//npars, free, [fixed for fixed in parnames if fixed != free]) for i, free in enumerate(names_cycle))

        axs = axs or {}
        axs_xmodels = axs.get("xmodels", {})
        axs_ymodels = axs.get("ymodels", {})

        # TODO: implement burn-in iterations to refine guess traces using Gaussian fitting
        fitted_traces = copy(guess_traces)


        log.info(f"initiating iterative fiber tracing with parameters: {list(fitted_traces.keys())}")
        for i, free_name, fixed_names in _block_cycle(fitted_traces.keys(), niter=niter):
            # TODO: set boundary constraints at image edges to avoid overshoots
            log.info(f"   iteration {i+1:3d}/{niter}:")
            axs_xfree = axs_xmodels.get(free_name, [])
            axs_yfree = axs_ymodels.get(free_name, {})

            free_trace = {free_name: fitted_traces.get(free_name)}
            free_bounds = {free_name: bounds.get(free_name)}
            measuring_conf_ = measuring_conf.get(free_name)

            fixed_traces_ = {fixed_name: fitted_traces.get(fixed_name) for fixed_name in fixed_names}
            fixed_traces_.update(fixed_traces)

            fitted_block = self.measure_fiber_block(
                profile, free_trace, fixed_traces_, iblock, columns, free_bounds,
                measuring_conf=measuring_conf_, npixels=npixels, oversampling_factor=oversampling_factor, axs=axs_yfree)

            smoothing_model, smoothing_conf_ = smoothing_conf.get(free_name)
            smoothing_method = getattr(fitted_block[free_name], f"fit_{smoothing_model}")
            smoothing_method(**smoothing_conf_)
            free_trace[free_name].set_block(iblock=iblock, from_instance=fitted_block[free_name])
            free_trace[free_name]._coeffs = None

            fitted_traces.update(free_trace)

            _set_plot_alphas(axs=axs_yfree, niter_done=i+1)
            if len(axs_xfree) != 0:
                free_trace[free_name].plot_block(iblock=iblock, show_model_samples=False, axs={"mod": axs_xfree[i]})

        return fitted_traces

    def trace_fibers_full(self, centroids_guess, fwhms_guess=2.5, centroids_range=[-5,5], fwhms_range=[1.0,3.5], counts_range=[1e3,numpy.inf],
                          columns=[], iblocks=[], solver="trf"):

        if self._header is None:
            raise ValueError("Invalid value of attribute `_header`: {self._header}. Expected FITS header object")

        ncolumns = len(columns)

        fwhms_guess = self._get_fwhms_trace(fwhms=fwhms_guess)

        # initialize flux and FWHM traces
        centroids_trace = TraceMask.create_empty(
            data_dim=(centroids_guess._fibers, self._dim[1]), samples_columns=sorted(set(columns)), header=self._header.copy(), slitmap=self._slitmap)
        centroids_trace.setFibers(centroids_guess._fibers)
        centroids_trace._good_fibers = centroids_guess._good_fibers
        counts_trace = copy(centroids_trace)
        fwhms_trace = copy(centroids_trace)
        centroids_trace._header["IMAGETYP"] = "fiber_centroids"
        counts_trace._header["IMAGETYP"] = "fiber_counts"
        fwhms_trace._header["IMAGETYP"] = "fiber_fwhms"

        # define fiber blocks
        if iblocks and isinstance(iblocks, (list, tuple, numpy.ndarray)):
            pass
        else:
            iblocks = numpy.arange(LVM_NBLOCKS, dtype="int")

        # fit each block
        for iblock in iblocks:
            centroids_block = centroids_guess.get_block(iblock=iblock)
            fwhms_block = fwhms_guess.get_block(iblock=iblock)

            counts_samples = numpy.full((centroids_block._fibers, ncolumns), numpy.nan)
            centroids_samples = numpy.full((centroids_block._fibers, ncolumns), numpy.nan)
            fwhms_samples = numpy.full((centroids_block._fibers, ncolumns), numpy.nan)
            iterator = tqdm(enumerate(columns), total=len(columns), desc=f"fitting fibers in block: {iblock+1:>2d}/{LVM_NBLOCKS}", ascii=True, unit="column")
            for i, icolumn in iterator:
                img_slice = self.getSlice(icolumn, axis="Y")
                centroids_slice, _, _ = centroids_block.getSlice(icolumn, axis="Y")
                fwhms_slice, _, _ = fwhms_block.getSlice(icolumn, axis="Y")

                counts_slice, pixels_selection = img_slice._guess_gaussians_integral(centroids_slice, fwhms_slice / 2.354, return_pixels_selection=True)

                model_block, par_block = img_slice.fitMultiGauss(
                    pixels_selection, counts_guess=counts_slice, centroids_guess=centroids_slice, fwhms_guess=fwhms_slice,
                    counts_range=counts_range, centroids_range=centroids_range, fwhms_range=fwhms_range, solver=solver)

                counts, centroids, fwhms = numpy.split(par_block, 3)
                counts_samples[:, i] = counts
                centroids_samples[:, i] = centroids
                fwhms_samples[:, i] = fwhms

            counts_mask = numpy.isnan(counts_samples)
            centroids_mask = numpy.isnan(centroids_samples)
            fwhms_mask = numpy.isnan(fwhms_samples)
            block_mask = numpy.tile(numpy.atleast_2d((counts_mask | centroids_mask | fwhms_mask).all(axis=1)).T, self._dim[1])

            # mask invalid samples in samples
            counts_samples[counts_mask] = numpy.nan
            centroids_samples[centroids_mask] = numpy.nan
            fwhms_samples[fwhms_mask] = numpy.nan

            # update tracemasks for this fiber block
            counts_trace.set_block(iblock=iblock, samples=counts_samples, mask=block_mask)
            centroids_trace.set_block(iblock=iblock, samples=centroids_samples, mask=block_mask)
            fwhms_trace.set_block(iblock=iblock, samples=fwhms_samples, mask=block_mask)

        return counts_trace, centroids_trace, fwhms_trace, columns

    def _get_block_pixels(self, centroids, iblock, npixels=5):
        nrows, ncols = self._dim
        x_pixels = numpy.arange(ncols, dtype="int")
        y_pixels = numpy.arange(nrows, dtype="int")
        X, Y = numpy.meshgrid(x_pixels, y_pixels, indexing="xy")

        centroids_block = centroids
        if iblock is not None:
            centroids_block = centroids.get_block(iblock=iblock)

        lower = numpy.nanmin(centroids_block._data, 0) - npixels
        upper = numpy.nanmax(centroids_block._data, 0) + npixels
        pixels_selection = (lower <= Y) & (Y <= upper)
        return X, Y, pixels_selection

    def evaluate_fiber_model(self, traces, profile="normal", iblock=None, blockid=None, oversampling_factor=10, columns=None, npixels=5, verbose=True, axs=None):
        nrows, ncols = self._dim
        if columns is None:
            columns = numpy.arange(ncols, dtype="int")

        blocks = traces.copy()
        if iblock is not None or blockid is not None:
            blocks = {name: trace.get_block(iblock, blockid) for name, trace in traces.items()}

        X, Y, pixels_selection = self._get_block_pixels(centroids=blocks["centroids"], iblock=iblock, npixels=npixels)

        profile_model = PROFILES.get(profile)
        if profile_model is None:
            raise ValueError(f"Invalid value for `profile`: {profile}. Expected one of {PROFILES}")

        model_array = numpy.full((nrows, ncols), numpy.nan)
        if verbose:
            iterator = tqdm(columns, desc=f"evaluating fiber profile '{profile}'", ascii=True, unit="column")
        else:
            iterator = columns
        for icolumn in iterator:
            selection = pixels_selection[:, icolumn]
            pixels = Y[selection, icolumn]
            pars_column = {name: block.getSlice(icolumn, axis="Y")[0] for name, block in blocks.items()}
            model_array[selection, icolumn] = profile_model(pars_column, {}, {}, oversampling_factor=oversampling_factor)._pixelate(pixels)
        model = Image(data=model_array, mask=numpy.isnan(model_array))

        axs = plot_fiber_residuals(model, self, blocks["centroids"], iblock, X=X, Y=Y, axs=axs)

        return model, X, Y, pixels_selection


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

    def extractSpecOptimal(self, cent_trace, trace_sigma, plot_fig=False):
        # initialize RSS arrays
        data = numpy.zeros((cent_trace._fibers, self._dim[1]), dtype=numpy.float32)
        if self._error is not None:
            error = numpy.zeros((cent_trace._fibers, self._dim[1]), dtype=numpy.float32)
        else:
            error = None
        mask = numpy.zeros((cent_trace._fibers, self._dim[1]), dtype="bool")

        self._data = numpy.nan_to_num(self._data)
        self._error = numpy.nan_to_num(self._error, nan=numpy.inf)

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

            # measure flux along the given columns
            # result = slice_img.obtainGaussFluxPeaks(cent[good_fiber], sigma[good_fiber], plot=plot_fig)
            result = slice_img.extract_flux(cent[good_fiber], sigma[good_fiber])
            # try:
            data[good_fiber, i] = result[0]
            if self._error is not None:
                error[good_fiber, i] = result[1]
            if self._mask is not None:
                mask[good_fiber, i] = result[2]
            mask[bad_fiber, i] = True
            # except Exception as e:
            #     print(e)
            #     print(i, result[0].shape, result[1].shape, result.shape[2])
            #     print(error.shape, mask.shape)
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
        LA_kernel = 0.25*numpy.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=numpy.float32)

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
            self._slitmap = Table.read(slitmap)
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
        camera = images[0].getHdrValue("CCD").upper()
        quad_sections = images[0].getHdrValues(f"{camera} AMP? TRIMSEC")
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
