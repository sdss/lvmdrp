import numpy
from astropy.io import fits as pyfits
from scipy import interpolate
from tqdm import tqdm
from copy import deepcopy as copy
import warnings

import bottleneck as bn
import pandas as pd
from lvmdrp import log
from astropy.table import Table

from lvmdrp.core.constants import LVM_NBLOCKS, LVM_BLOCKSIZE, LVM_NFIBERS
from lvmdrp.core.header import Header, combineHdr
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.spectrum1d import Spectrum1D, _cross_match_float
from lvmdrp.core import plot


def fillin_gap(x, distances, inplace=False):
    """Fills in NaN values maintaining given distances
    """
    if inplace:
        x_ = x
    else:
        x_ = x.copy()
    igaps, = numpy.where(numpy.isnan(x))
    for igap in igaps:
        if igap == 0:
            x_[igap] = x_[igap+1] - distances[igap+1]
            continue
        x_[igap] = x_[igap-1] + distances[igap]
    return x_


def _read_fiber_ypix(peaks_file):
    """
    Read peaks file and return the fiber number, pixel position, subpixel position
    and quality flag.
    """
    peaks = pyfits.open(peaks_file, memmap=False)
    xpos = peaks[1].header["XPIX"]
    fiber = peaks[1].data["FIBER"]
    pixel = peaks[1].data["PIXEL"]
    subpix = peaks[1].data["SUBPIX"]
    qual = peaks[1].data["QUALITY"].astype(bool)
    return xpos, fiber, pixel, subpix, qual


def _guess_spline(x, y, k, s, w=None):
    """Do an ordinary spline fit to provide knots"""
    return interpolate.splrep(x, y, w, k=k, s=s)

def _residual_spline(c, x, y, t, k, w=None):
    """The error function to minimize"""
    diff = y - interpolate.splev(x, (t, c, k))
    if w is None:
        diff = numpy.einsum('...i,...i', diff, diff)
    else:
        diff = numpy.dot(diff*diff, w)
    return numpy.abs(diff)


class FiberRows(Header, PositionTable):

    @classmethod
    def create_empty(cls, data_dim, poly_kind=None, poly_deg=None, samples_columns=None, header=None, slitmap=None):
        data = numpy.full(data_dim, numpy.nan, dtype=numpy.float32)
        fibers = data.shape[0]
        error = numpy.full(data_dim, numpy.nan, dtype=numpy.float32)
        mask = numpy.ones(data_dim, dtype="bool")
        if samples_columns is not None:
            samples = Table(data=numpy.full((data_dim[0], len(samples_columns)), numpy.nan), names=samples_columns)
        else:
            samples = None
        if poly_deg is not None:
            coeffs = numpy.full((data_dim[0], poly_deg+1), numpy.nan, dtype=numpy.float32)
        else:
            coeffs = None

        new_fiberrows = cls(data=data, error=error, mask=mask, samples=samples, poly_kind=poly_kind, coeffs=coeffs, header=header, slitmap=slitmap)
        new_fiberrows.setFibers(fibers)
        return new_fiberrows

    @classmethod
    def from_samples(cls, data_dim, samples, samples_columns=None):
        new_fiberrows = cls.create_empty(data_dim=data_dim, samples_columns=samples_columns)
        new_fiberrows.set_samples(samples, columns=samples_columns)
        return new_fiberrows

    @classmethod
    def from_coeff_table(cls, coeff_table, **kwargs):
        """Creates an FiberRows instance from a table of coefficients"""

        if coeff_table is None:
            return None

        nfibers = len(coeff_table)
        npixels = coeff_table["XMAX"].max() - coeff_table["XMIN"].min() + 1
        x_pixels = numpy.arange(npixels)

        data = numpy.zeros((nfibers, npixels), dtype=numpy.float32)
        coeffs = numpy.zeros((nfibers, coeff_table["COEFF"].shape[1]), dtype=numpy.float32)
        for ifiber in range(nfibers):
            poly_cls = Spectrum1D.select_poly_class(poly_kind=coeff_table[ifiber]["FUNC"])
            data[ifiber] = poly_cls(coeff_table[ifiber]["COEFF"])(x_pixels)
            coeffs[ifiber] = coeff_table[ifiber]["COEFF"]

        return cls(data=data, coeffs=coeffs, **kwargs)

    @classmethod
    def from_spectrographs(cls, spec1, spec2, spec3):
        hdrs = []
        fiberrows = [spec1, spec2, spec3]
        for i in range(len(fiberrows)):
            fiberrow = fiberrows[i]
            if i == 0:
                data_out = fiberrow._data
                if fiberrow._error is not None:
                    error_out = fiberrow._error
                if fiberrow._mask is not None:
                    mask_out = fiberrow._mask
                if fiberrow._coeffs is not None:
                    coeffs_out = fiberrow._coeffs
            else:
                data_out = numpy.concatenate((data_out, fiberrow._data), axis=0)
                if fiberrow._error is not None:
                    error_out = numpy.concatenate((error_out, fiberrow._error), axis=0)
                else:
                    error_out = None
                if fiberrow._mask is not None:
                    mask_out = numpy.concatenate((mask_out, fiberrow._mask), axis=0)
                else:
                    mask_out = None
                if fiberrow._coeffs is not None:
                    coeffs_out = numpy.concatenate((coeffs_out, fiberrow._coeffs), axis=0)
                else:
                    coeffs_out = None

        # update header
        if len(hdrs) > 0:
            hdr_out = combineHdr(hdrs)
        else:
            hdr_out = None

        fiberrow_out = cls(
            data=data_out,
            error=error_out,
            mask=mask_out,
            coeffs=coeffs_out
        )
        fiberrow_out.setHeader(hdr_out)

        return fiberrow_out

    def __init__(
        self,
        data=None,
        header=None,
        error=None,
        mask=None,
        slitmap=None,
        samples=None,
        shape=None,
        size=None,
        arc_position_x=None,
        arc_position_y=None,
        good_fibers=None,
        fiber_type=None,
        coeffs=None,
        poly_kind=None,
        poly_deg=None
    ):
        Header.__init__(self, header=header)
        PositionTable.__init__(
            self,
            shape=shape,
            size=size,
            arc_position_x=arc_position_x,
            arc_position_y=arc_position_y,
            good_fibers=good_fibers,
            fiber_type=fiber_type,
        )
        self.setData(data=data, error=error, mask=mask)
        self.set_samples(samples)
        self.set_coeffs(coeffs=coeffs, poly_kind=poly_kind)
        if self._data is None and self._coeffs is not None:
            self.eval_coeffs()

        self.setSlitmap(slitmap)

    def __len__(self):
        return self._fibers

    def _propagate_error(self, other, operation):
        """ Error propagation for different operations. """
        if self._error is None and getattr(other, "_error", None) is None:
            return None

        err1 = self._error if self._error is not None else 0
        err2 = other._error if isinstance(other, self.__class__) and other._error is not None else 0

        if operation in ('add', 'sub'):
            return numpy.sqrt(err1**2 + err2**2)
        elif operation == 'mul':
            return numpy.sqrt((err1 * other._data)**2 + (self._data * err2)**2)
        elif operation == 'div':
            return numpy.sqrt((err1 / other._data)**2 + (self._data * err2 / other._data**2)**2)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _apply_operation(self, other, op, op_name):
        if isinstance(other, self.__class__):
            new_data = op(self._data, other._data)
            new_error = self._propagate_error(other, op_name)
            new_mask = numpy.logical_or(self._mask, other._mask) if self._mask is not None and other._mask is not None else None
        elif isinstance(other, numpy.ndarray) or numpy.isscalar(other):
            new_data = op(self._data, other)
            new_error = (op(self._error, other) if self._error is not None else None)
            new_mask = self._mask
        else:
            raise NotImplementedError(f"operation '{op_name}' between {self.__class__} and {type(other)} is not implemented")

        new = copy(self)
        new.setData(data=new_data, error=new_error, mask=new_mask)
        return new

    def __add__(self, other):
        return self._apply_operation(other, numpy.add, 'add')

    def __sub__(self, other):
        return self._apply_operation(other, numpy.subtract, 'sub')

    def __mul__(self, other):
        return self._apply_operation(other, numpy.multiply, 'mul')

    def __truediv__(self, other):
        return self._apply_operation(other, numpy.divide, 'div')

    def __getitem__(self, fiber):
        if not isinstance(fiber, int):
            raise TypeError("Fiber index need to be an integer")
        if fiber >= self._fibers or fiber < self._fibers * -1:
            raise IndexError(
                "The Object contains only %i Fibers for which the index %i is invalid"
                % (self._fibers, fiber)
            )
        data = self._data[fiber, :]
        if self._error is not None:
            error = self._error[fiber, :]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[fiber, :]
        else:
            mask = None
        spec = Spectrum1D(
            numpy.arange(self._data.shape[1]), data, error=error, mask=mask
        )
        return spec

    def __setitem__(self, fiber, spec):
        if not isinstance(fiber, int):
            raise TypeError("Fiber index need to be an integer")

        if fiber >= self._fibers or fiber < self._fibers * -1:
            raise IndexError(
                "The Object contains only %i Fibers for which the index %i is invalid"
                % (self._fibers, fiber)
            )

        self._data[fiber, :] = spec._data

        if self._error is not None and spec._error is not None:
            self._error[fiber, :] = spec._error

        if self._mask is not None and spec._mask is not None:
            self._mask[fiber, :] = spec._mask

    def __getslice__(self, i, j):
        if not isinstance(i, int) and not isinstance(j, int):
            raise TypeError("Fiber indices need to be integers")

        data = self._data[i:j, :]
        if self._error is not None:
            error = self._error[i:j, :]
        else:
            error = None

        if self._mask is not None:
            mask = self._mask[i:j, :]
        else:
            mask = None

        return self.__class__(data=data, error=error, mask=mask)

    def _get_iblocks(self):
        slitmap = self._filter_slitmap().to_pandas()
        iblocks = slitmap["blockid"].str[1:].drop_duplicates().values.astype("int") - 1
        return iblocks

    def _validate_blockid(self, iblock, blockid, slitmap):
        if blockid is not None:
            pass
        elif iblock is not None:
            blockid = f"B{iblock+1}"
        else:
            raise ValueError(f"Either `iblock` or `blockid` needs to be given: {iblock = }, {blockid = }")
        if blockid not in slitmap["blockid"]:
            raise ValueError(f"Given {blockid = } not found in slitmap column `blockid`")

        return blockid

    def get_block(self, iblock=None, blockid=None):
        slitmap = self._filter_slitmap()
        blockid = self._validate_blockid(iblock, blockid, slitmap=slitmap)
        block_selection = slitmap["blockid"] == blockid
        if block_selection.sum() == 0:
            raise ValueError(f"Requested block: {iblock = }, {blockid = }, is not present in slitmap")

        new_trace = copy(self)
        new_trace._data = self._data[block_selection]
        new_trace._error = self._error[block_selection] if self._error is not None else None
        new_trace._mask = self._mask[block_selection] if self._mask is not None else None
        new_trace._samples = self._samples[block_selection] if self._samples is not None else None
        new_trace.set_coeffs(self._coeffs[block_selection] if self._coeffs is not None else None, poly_kind=self._poly_kind)
        new_trace.setFibers(block_selection.sum())
        new_trace.setSlitmap(slitmap[block_selection])

        return new_trace

    def set_block(self, data=None, iblock=None, blockid=None, error=None, mask=None, samples=None, coeffs=None, poly_kind=None, from_instance=None):

        if from_instance is not None:
            samples_o = from_instance.get_samples(as_pandas=True)
            samples_o = samples_o.values if samples_o is not None else None
            self.set_block(
                data=from_instance._data, iblock=iblock, blockid=blockid,
                error=from_instance._error, mask=from_instance._mask,
                samples=samples_o, coeffs=from_instance._coeffs, poly_kind=from_instance._poly_kind)

        slitmap = self._filter_slitmap()
        blockid = self._validate_blockid(iblock, blockid, slitmap=slitmap)
        block_selection = slitmap["blockid"] == blockid
        nfibers = block_selection.sum()

        if data is not None:
            if data.shape[0] != nfibers:
                raise ValueError(f"Incompatible data shapes. Trying to set a block of {data.shape[0]} fibers to a selection of {nfibers} fibers")
            self._data[block_selection] = data
        if error is not None and self._error is not None:
            self._error[block_selection] = error
        if mask is not None and self._error is not None:
            self._mask[block_selection] = mask
        if samples is not None and self._samples is not None:
            samples_i = self._samples.to_pandas()
            if samples.shape[1] != samples_i.shape[1]:
                raise ValueError(f"Incompatible column sizes for samples. Trying to set samples with {samples.shape[1]} columns to {samples_i.columns.size} columns")
            if samples.shape[0] != nfibers:
                raise ValueError(f"Incompatible sample sizes. Trying to set samples with {samples.shape[0]} fibers to {nfibers} fibers")
            for i, column in enumerate(samples_i.columns):
                samples_i.loc[block_selection, column] = samples[:, i]
            self.set_samples(samples_i)
        if coeffs is not None and poly_kind is not None and self._coeffs is not None:
            if self._poly_kind != poly_kind:
                raise ValueError(f"Incompatible polynomial kinds. Trying to set {poly_kind} to a tracemask of {self._poly_kind}")
            poly_deg = coeffs.shape[1] - 1
            if self._poly_deg != poly_deg:
                raise ValueError(f"Incompatible polynomial degree. Trying to set {poly_deg} to a tracemask of {self._poly_deg}")
            self.set_coeffs(coeffs, poly_kind=poly_kind)

    def get_distances(self):
        samples = self.get_samples(as_pandas=True)
        if samples is not None:
            sample_distances = numpy.gradient(samples, axis=0)
        else:
            sample_distances = None
        if self._data is not None:
            model_distances = numpy.gradient(self._data, axis=0)
        else:
            model_distances = None
        return sample_distances, model_distances

    def createEmpty(self, data_dim, poly_deg=None, samples_columns=None, header=None, slitmap=None):
        """
        Fill the FiberRows object with empty data

        Parameters
        --------------
        data_dim: tuple, optional with default: None
            Dimension of the empty data array to be created
        poly_deg: int, optional with default: None
            Degree of the polynomial trace to be created
        """
        self._data = numpy.full(data_dim, numpy.nan, dtype=numpy.float32)
        self._fibers = self._data.shape[0]
        self._error = numpy.full(data_dim, numpy.nan, dtype=numpy.float32)
        self._mask = numpy.ones(data_dim, dtype="bool")
        if samples_columns is not None:
            self._samples = Table(data=numpy.full((data_dim[0], len(samples_columns)), numpy.nan), names=samples_columns)
        if poly_deg is not None:
            self._coeffs = numpy.full((data_dim[0], poly_deg+1), numpy.nan, dtype=numpy.float32)
        self.setHeader(header)
        self.setSlitmap(slitmap)

    def setFibers(self, fibers):
        """
        Set the number of fibers

        Parameters
        --------------
        fibers: int
            Number of fibers
        """
        self._fibers = fibers

    def setSlice(self, slice, axis="x", data=None, error=None, mask=None, samples=None, select=None):
        """
        Insert data to a slice of the trace mask

        Parameters
        --------------
        slice: int
            Pixell position of the slice

        axis : string or ing (0 or 1), optional with default: 'x'
            Defines the axis of the slice to be inserted, 'X', 'x', or 1 for the x-axis or
            'Y','y', or 0 for the y-axis.

        data : numpy.ndarray (float), optional with default: None
            1D data array to be inserted

        error : numpy.ndarray (float), optional with default: None
            1D error array to be inserted

        mask : numpy.ndarray bool), optional with default: None
            1D array of masked pixel to be inserted

        select : numpy.ndarray bool), optional with default: None
            Subselection of pixels along the slice that should be inserted
        """
        if axis == "X" or axis == "x" or axis == 1:
            if select is not None:
                if data is not None:
                    self._data[slice, select] = data
                if error is not None:
                    self._error[slice, select] = error
                if mask is not None:
                    self._mask[slice, select] = mask
            else:
                if data is not None:
                    self._data[slice, :] = data
                if error is not None:
                    self._error[slice, :] = error
                if mask is not None:
                    self._mask[slice, :] = mask
        elif axis == "Y" or axis == "y" or axis == 0:
            if select is not None:
                if data is not None:
                    self._data[select, slice] = data
                if error is not None:
                    self._error[select, slice] = error
                if mask is not None:
                    self._mask[select, slice] = mask
            else:
                if data is not None:
                    self._data[:, slice] = data
                if error is not None:
                    self._error[:, slice] = error
                if mask is not None:
                    self._mask[:, slice] = mask

    def getSlice(self, slice, axis="x"):
        if axis == "X" or axis == "x" or axis == 1:
            if self._data is not None:
                slice_data = self._data[slice, :]
            else:
                slice_data = None
            if self._error is not None:
                slice_error = self._error[slice, :]
            else:
                slice_error = None
            if self._mask is not None:
                slice_mask = self._mask[slice, :]
            else:
                slice_mask = None

        elif axis == "Y" or axis == "y" or axis == 0:
            if self._data is not None:
                slice_data = self._data[:, slice]
            else:
                slice_data = None
            if self._error is not None:
                slice_error = self._error[:, slice]
            else:
                slice_error = None
            if self._mask is not None:
                slice_mask = self._mask[:, slice]
            else:
                slice_mask = None
        else:
            return None, None, None
        return slice_data, slice_error, slice_mask

    def getSpec(self, fiber):
        data = self._data[fiber, :]
        if self._error is not None:
            error = self._error[fiber, :]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[fiber, :]
        else:
            mask = None
        spec = Spectrum1D(
            numpy.arange(self._data.shape[1]), data, error=error, mask=mask, header=self._header
        )

        return spec

    def getData(self):
        """
        Return the content of the FiberRows object

        Returns: (data, error mask)
        -----------
        data : numpy.ndarray (float)
            Array of the data value

        error : numpy.ndarray (float)
            Array of the corresponding errors

        mask : numpy.ndarray (bool)
            Array of the bad pixel mask
        """
        data = self._data
        error = self._error
        mask = self._mask
        return data, error, mask

    def setData(self, select=None, data=None, mask=None, error=None):
        if select is not None:
            if data is not None:
                self._data[select] = data
            if mask is not None:
                self._mask[select] = mask
            if error is not None:
                self._error[select] = error
        else:
            nfibers, npixels = None, None
            if data is not None:
                self._data = data
                nfibers, npixels = data.shape
            elif not hasattr(self, "_data"):
                self._data = None
            if mask is not None:
                self._mask = mask
                nfibers, npixels = self._mask.shape
                self._good_fibers = numpy.where(numpy.sum(self._mask, axis=1) != self._mask.shape[1])[0]
            elif not hasattr(self, "_mask"):
                self._mask = None
                self._good_fibers = None
            if error is not None:
                self._error = error
                nfibers, npixels = error.shape
            elif not hasattr(self, "_error"):
                self._error = None

            if nfibers is not None:
                self._fibers = nfibers
            elif not hasattr(self, "_fibers"):
                self._fibers = None
            if npixels is not None:
                self._pixels = numpy.arange(npixels) if npixels is not None else npixels
            elif not hasattr(self, "_pixels"):
                self._pixels = None

    def _filter_slitmap(self):
        if self._slitmap is None:
                raise ValueError(f"Attribute `_slitmap` needs to be set: {self._slitmap = }")
        if self._header is None:
            raise ValueError(f"Attribute `_header` needs to be set: {self._header = }")

        slitmap = self._slitmap
        if self._fibers == LVM_NFIBERS:
            slitmap = self._slitmap[self._slitmap["spectrographid"]==int(self._header["SPEC"][-1])]
        return slitmap

    def getSlitmap(self):
        return self._slitmap

    def setSlitmap(self, slitmap):
        if slitmap is None:
            self._slitmap = None
            return
        if isinstance(slitmap, pyfits.BinTableHDU):
            self._slitmap = Table.read(slitmap)
        elif isinstance(slitmap, Table):
            self._slitmap = slitmap
        else:
            raise TypeError(f"Invalid slitmap table type '{type(slitmap)}'")

    def set_samples(self, samples=None, columns=None):
        if isinstance(samples, Table):
            self._samples = samples
        elif isinstance(samples, pd.DataFrame):
            samples.columns = samples.columns.astype("str")
            self._samples = Table.from_pandas(samples)
        elif isinstance(samples, numpy.ndarray) and columns is not None:
            self._samples = Table(data=samples, names=columns)
        elif columns is not None:
            self._samples = Table(data=numpy.full((self._fibers, len(columns)), numpy.nan), names=columns)
        else:
            self._samples = None

        return self._samples

    def get_samples(self, as_pandas=False):
        if self._samples is None:
            return None
        if as_pandas:
            df = self._samples.to_pandas()
            df.columns = df.columns.astype("int")
            return df
        return self._samples

    def apply_pixelmask(self, mask=None):
        if mask is None:
            mask = self._mask
        if mask is None:
            return self

        self._data[mask] = numpy.nan
        self._error[mask] = numpy.nan
        return self

    def split(self, fragments, axis="x"):
        list = []
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
            list.append(
                self.__class__(data=split_data[i], error=split_error[i], mask=split_mask[i])
            )

        return list

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

    def applyFibers(self, function, args):
        result = []
        for i in range(len(self)):
            result.append(function(args))
        return result

    def measureArcLines(
        self,
        ref_fiber,
        ref_cent,
        aperture=12,
        fwhm_guess=3,
        bg_guess=0.0,
        flux_range=[0.0, numpy.inf],
        cent_range=[-2.0, 2.0],
        fwhm_range=[0, 7],
        bg_range=[0, numpy.inf],
        axs=None,
    ):
        nlines = len(ref_cent)
        flux = numpy.ones((self._fibers, nlines), dtype=numpy.float32) * numpy.nan
        cent_wave = numpy.ones((self._fibers, nlines), dtype=numpy.float32) * numpy.nan
        fwhm = numpy.ones((self._fibers, nlines), dtype=numpy.float32) * numpy.nan
        bg = numpy.ones((self._fibers, nlines), dtype=numpy.float32) * numpy.nan
        masked = numpy.zeros((self._fibers, nlines), dtype="bool")

        # define pixel distance between lines, fairly constant fiber-to-fiber
        lines_dist = numpy.asarray([0.0] + numpy.diff(ref_cent).tolist())

        stretch_min, stretch_max, stretch_steps = 0.998, 1.002, 40

        spec = self.getSpec(ref_fiber)
        flux[ref_fiber], cent_wave[ref_fiber], fwhm[ref_fiber], bg[ref_fiber] = spec.fitSepGauss(ref_cent, aperture, fwhm_guess, bg_guess, flux_range, cent_range, fwhm_range, bg_range, axs=axs[ref_fiber][1])
        masked[ref_fiber] = numpy.isnan(flux[ref_fiber])|numpy.isnan(cent_wave[ref_fiber])|numpy.isnan(fwhm[ref_fiber])
        first = numpy.arange(ref_fiber - 1, -1, -1)
        second = numpy.arange(ref_fiber + 1, self._fibers, 1)

        last_spec = copy(self.getSpec(ref_fiber))
        last_cent = copy(cent_wave[ref_fiber])
        iterator = tqdm(
            first,
            total=first.size,
            desc=f"measuring arc lines   upwards from {ref_fiber = }",
            ascii=True,
            unit="fiber",
        )
        for i in iterator:
            spec = self.getSpec(i)
            if spec._mask.all():
                masked[i] = True
                continue

            if axs is not None and i in axs:
                _, axs_fiber = axs[i]
            else:
                axs_fiber = None

            cc, bhat, mhat = _cross_match_float(
                ref_spec=last_spec._data,
                obs_spec=spec._data,
                stretch_factors=numpy.linspace(stretch_min, stretch_max, stretch_steps),
                shift_range=[-10, 10],
                normalize_spectra=False,
            )
            if mhat == stretch_min or mhat == stretch_max:
                log.warning(f"boundary of stretch factors: {mhat = } ({stretch_min, stretch_max = })")
            cent_guess = mhat * last_cent + bhat
            flux[i], cent_wave[i], fwhm[i], bg[i] = spec.fitSepGauss(cent_guess, aperture, fwhm_guess, bg_guess, flux_range, cent_range, fwhm_range, bg_range, axs=axs_fiber)
            masked[i] = numpy.isnan(flux[i])|numpy.isnan(cent_wave[i])|numpy.isnan(fwhm[i])
            if masked[i].any():
                log.warning(f"some lines were not fitted properly in fiber {i}: ")
                log.warning(f"  guess = {numpy.round(cent_guess, 3)} ({mhat = :.5f}, {bhat = :.5f})")
                log.warning(f"   mask = {masked[i]}")
                log.warning(f"   flux = {numpy.round(flux[i],3)}")
                log.warning(f"   cent = {numpy.round(cent_wave[i],3)}")
                log.warning(f"   fwhm = {numpy.round(fwhm[i],3)}")
                log.warning(f"   bg   = {numpy.round(bg[i],3)}")

            last_spec = copy(spec)
            last_cent = fillin_gap(cent_wave[i], distances=lines_dist, inplace=False)

        last_spec = copy(self.getSpec(ref_fiber))
        last_cent = copy(cent_wave[ref_fiber])
        iterator = tqdm(
            second,
            total=second.size,
            desc=f"measuring arc lines downwards from {ref_fiber = }",
            ascii=True,
            unit="fiber",
        )
        for i in iterator:
            spec = self.getSpec(i)
            if spec._mask.all():
                masked[i] = True
                continue

            if axs is not None and i in axs:
                _, axs_fiber = axs[i]
            else:
                axs_fiber = None

            cc, bhat, mhat = _cross_match_float(
                ref_spec=last_spec._data,
                obs_spec=spec._data,
                stretch_factors=numpy.linspace(stretch_min, stretch_max, stretch_steps),
                shift_range=[-10, 10],
                normalize_spectra=False,
            )
            if mhat == stretch_min or mhat == stretch_max:
                log.warning(f"boundary of stretch factors: {mhat = } ({stretch_min, stretch_max = })")
            cent_guess = mhat * last_cent + bhat
            flux[i], cent_wave[i], fwhm[i], bg[i] = spec.fitSepGauss(cent_guess, aperture, fwhm_guess, bg_guess, flux_range, cent_range, fwhm_range, bg_range, axs=axs_fiber)
            masked[i] = numpy.isnan(flux[i])|numpy.isnan(cent_wave[i])|numpy.isnan(fwhm[i])
            if masked[i].any():
                log.warning(f"some lines were not fitted properly in fiber {i}: ")
                log.warning(f"  guess = {numpy.round(cent_guess, 3)} ({mhat = :.5f}, {bhat = :.5f})")
                log.warning(f"   mask = {masked[i]}")
                log.warning(f"   flux = {numpy.round(flux[i],3)}")
                log.warning(f"   cent = {numpy.round(cent_wave[i],3)}")
                log.warning(f"   fwhm = {numpy.round(fwhm[i],3)}")
                log.warning(f"   bg   = {numpy.round(bg[i],3)}")

            last_spec = copy(spec)
            last_cent = fillin_gap(cent_wave[i], distances=lines_dist, inplace=False)

        fibers = numpy.arange(self._fibers)
        return fibers, flux, cent_wave, fwhm, masked

    def append(self, rows, append_hdr=False):
        #  print(self._error,  rows._error)
        if self._data is not None and rows._data is not None:
            self._data = numpy.concatenate((self._data, rows._data))
            if self._header is not None:
                self.setHdrValue("NAXIS2", self._fibers + rows._fibers)
        if self._error is not None and rows._error is not None:
            self._error = numpy.concatenate((self._error, rows._error))
        if self._mask is not None and rows._mask is not None:
            self._mask = numpy.concatenate((self._mask, rows._mask))
        try:
            self._arc_position_x = numpy.concatenate(
                (self._arc_position_x, rows._arc_position_x)
            )
        except ValueError:
            self._arc_position_x = None
        try:
            self._arc_position_y = numpy.concatenate(
                (self._arc_position_y, rows._arc_position_y)
            )
        except ValueError:
            self._arc_position_y = None
        try:
            self._good_fibers = numpy.concatenate(
                (self._good_fibers, rows._good_fibers)
            )
        except ValueError:
            self._good_fibers = None
        try:
            self._fiber_type = numpy.concatenate((self._fiber_type, rows._fiber_type))
        except ValueError:
            self._fiber_type = None

        if append_hdr:
            combined_hdr = combineHdr([self, rows])
            self.setHeader(combined_hdr._header)

    def fit_spline(self, deg=3, nknots=5, knots=None, smoothing=None, weights=None, clip=None, min_samples_frac=0.0):
        """
        smooths the traces along the dispersion direction with a spline function for each individual fiber

        Parameters
        ----------
        nknots: int, optional with default None
            number of knots to use in the spline function
        knots: numpy.ndarray, optional with default None
            array of knots to use in the spline function
        clip : 2-tuple of int, optional with default None
            clip data around this values, defaults to no clipping

        Returns
        -------
        pix_table : numpy.ndarray
            table of measured values
        poly_table : numpy.ndarray
            table of spline values at measured values
        poly_all_table : numpy.ndarray
            table of spline values for all pixels in the fibers
        """
        pixels = numpy.arange(self._data.shape[1])
        if nknots is not None and knots is None:
            knots = numpy.linspace(pixels[len(pixels) // nknots], pixels[-1 * len(pixels) // nknots], nknots)
        elif knots is not None:
            nknots = len(knots)
        else:
            knots = None

        _ = self._samples.to_pandas()
        columns = _.columns.astype("int")
        samples = _.values
        coeffs = numpy.full(self._data.shape[0], numpy.nan, dtype=object)

        pix_table = []
        poly_table = []
        poly_all_table = []
        for i in range(self._fibers):
            good_sam = numpy.isfinite(samples[i, :])
            n_goodsam = good_sam.sum()
            if n_goodsam == 0:
                self._mask[i, :] = True
                continue

            nsamples = good_sam.size
            can_fit = n_goodsam >= deg + 1
            enough_samples = n_goodsam / nsamples > min_samples_frac
            if can_fit and enough_samples:

                # try to fit
                try:
                    tck = interpolate.splrep(columns[good_sam], samples[i, good_sam], s=smoothing)

                    pix_table.extend(numpy.column_stack([columns[good_sam], samples[i, good_sam]]).tolist())
                    poly_table.extend(numpy.column_stack([pixels[columns], interpolate.splev(pixels[columns], tck)]).tolist())
                    poly_all_table.extend(numpy.column_stack([pixels, interpolate.splev(pixels, tck)]).tolist())
                except Exception as e:
                    warnings.warn(f'Fiber trace failure at fiber {i}: {e}')
                    self._mask[i, :] = True
                    continue

                coeffs[i] = tck
                self._data[i, :] = interpolate.splev(pixels, tck)

                if clip is not None:
                    self._data = numpy.clip(self._data, clip[0], clip[1])
                self._mask[i, :] = False
            else:
                if not can_fit:
                    warnings.warn(f"fiber {i} does not meet criterium: {n_goodsam = } >= {deg + 1 = }")
                elif not enough_samples:
                    warnings.warn(f"fiber {i} does not meet criterium: {n_goodsam / nsamples = :.3f} > {min_samples_frac = :.3f}")
                self._mask[i, :] = True

        # TODO: port existing code to deal with tck objects as tables
        self._coeffs = coeffs

        return numpy.asarray(pix_table), numpy.asarray(poly_table), numpy.asarray(poly_all_table)

    def fit_polynomial(self, deg, poly_kind="poly", clip=None, min_samples_frac=0.0):
        """
        smooths the traces along the dispersion direction with a polynomical function for each individual fiber

        Parameters
        ----------
        deg: int
            degree of the polynomial function to describe the trace along diserpsion direction
        poly_kind : string, optional with default 'poly'
            the kind of polynomial to use when smoothing the trace, valid options are: 'poly' (power series, default), 'legendre', 'chebyshev'
        clip : 2-tuple of int, optional with default None
            clip data around this values, defaults to no clipping
        min_samples_frac : float, optional
            minimum fraction of valid samples, by default 0.0 (no threshold)
        """
        pixels = numpy.arange(self._data.shape[1])
        _ = self._samples.to_pandas()
        columns = _.columns.astype("int")
        samples = _.values
        coeffs = numpy.full((self._data.shape[0], numpy.abs(deg) + 1), numpy.nan)
        # iterate over each fiber
        pix_table = []
        poly_table = []
        poly_all_table = []
        for i in range(self._fibers):
            good_sam = numpy.isfinite(samples[i, :])
            n_goodsam = good_sam.sum()
            if n_goodsam == 0:
                self._mask[i, :] = True
                continue

            nsamples = good_sam.size
            can_fit = n_goodsam >= deg + 1
            enough_samples = n_goodsam / nsamples > min_samples_frac
            if can_fit and enough_samples:
                # select the polynomial class
                poly_cls = Spectrum1D.select_poly_class(poly_kind)

                # try to fit
                try:
                    poly = poly_cls.fit(columns[good_sam], samples[i, good_sam], deg=deg)
                    pix_table.extend(numpy.column_stack([columns[good_sam], samples[i, good_sam]]).tolist())
                    poly_table.extend(numpy.column_stack([pixels[columns], poly(pixels[columns])]).tolist())
                    poly_all_table.extend(numpy.column_stack([pixels, poly(pixels)]).tolist())
                except numpy.linalg.LinAlgError as e:
                    warnings.warn(f'Fiber trace failure at fiber {i}: {e}')
                    self._mask[i, :] = True
                    continue

                coeffs[i, :] = poly.convert().coef
                self._data[i, :] = poly(pixels)

                if clip is not None:
                    self._data = numpy.clip(self._data, clip[0], clip[1])
                self._mask[i, :] = False
            else:
                if not can_fit:
                    warnings.warn(f"fiber {i} does not meet criterium: {n_goodsam = } >= {deg + 1 = }")
                elif not enough_samples:
                    warnings.warn(f"fiber {i} does not meet criterium: {n_goodsam / nsamples = :.3f} > {min_samples_frac = :.3f}")
                self._mask[i, :] = True

        self.set_coeffs(coeffs, poly_kind=poly_kind)

        return numpy.asarray(pix_table), numpy.asarray(poly_table), numpy.asarray(poly_all_table)

    def fit_spline2d(self, deg_x=3, deg_y=3, smoothing=None, clip=None, use_weights=True, min_samples_frac=0.0, min_fibers_frac=0.0):
        iblocks = self._get_iblocks()

        columns = numpy.asarray(self._samples.colnames).astype("int")
        ifibers = numpy.arange(LVM_BLOCKSIZE)
        X, Y = numpy.meshgrid(columns, ifibers, indexing="xy")
        x, y = X.ravel(), Y.ravel()
        x_pixels = numpy.arange(self._data.shape[1])

        nsamples = columns.size
        npixels = x_pixels.size
        nfibers = LVM_BLOCKSIZE

        for iblock in tqdm(iblocks, desc="fitting 2D spline to fiber blocks", ascii=True, unit="block"):
            block = self.get_block(iblock=iblock)

            samples = block.get_samples(as_pandas=True).values

            good_sam = numpy.isfinite(samples)
            n_goodsam_per_fiber = good_sam.sum(axis=1)
            n_goodsam = n_goodsam_per_fiber.sum()
            if n_goodsam == 0:
                warnings.warn(f"skipping fiber block B{iblock+1}, no good samples found")
                block._mask[:] = True
                continue

            good_fib = (n_goodsam_per_fiber / nsamples > min_samples_frac)
            n_goodfib = good_fib.sum()
            enough_fibers = n_goodfib / nfibers > min_fibers_frac
            can_fit = n_goodsam >= (deg_x + 1) * (deg_y + 1)

            block._mask[~good_fib] = True

            if not can_fit:
                warnings.warn(f"fiber block B{iblock+1} does not meet criterium: {n_goodsam = } >= {(deg_x + 1) * (deg_y + 1) = }")
                block._mask[:] = True
                continue
            elif not enough_fibers:
                warnings.warn(f"fiber block B{iblock+1} does not meet criterium: {n_goodfib / nfibers = :.3f} > {min_fibers_frac = :.3f}")
                block._mask[:] = True
                continue

            good_sam[good_fib] = True
            select = good_sam.ravel()

            z = samples.ravel()
            x_, y_, z_ = x[select], y[select], z[select]

            tck = interpolate.bisplrep(x_, y_, z_, s=smoothing, xb=0, xe=npixels, yb=0, ye=nfibers, eps=1e-8)
            block_model = interpolate.bisplev(x_pixels, ifibers, tck).T

            self.set_block(iblock=iblock, data=block_model)

    def smoothTraceDist(
        self, start_slice, poly_cross=[4, 1, 4], poly_disp=8, bound=[350, 2000]
    ):
        """
        Smooth the traces along the dispersion direction assuming that their distance is a smooth function of wavelength.
        From a reference slice the distances are measured between the fibers are measured. In cross-dispersion direction the
        change of the distance between the fibers are modelled with a polynomial. The parameters of this polynomial are assumed
        to vary smoothly with wavelength and are modelled with a polynomial along dispersion axis. Uncertain TraceMask pixels are
        excluded from the modelling

        Parameters
        --------------
        start_slice : int
            Dispersion pixel position of the reference cross-dispersion position of fibers to compute their initial distances

        poly_cross : list of integers
            The length of the list correspond to the order of the polynomial used to fit the cross-disperion profile of the relative change in fiber distance ,
            and the value correspond to the order of the polynomial to smooth the corresponding fit parameter in cross-dispersion along the dispersion axis.

        poly_disp : int
            Order of the polynomial used to model correct the absolute positioning (zero-point) of the traces  along dispersion direction
        """

        select_wave = numpy.sum(self._mask, axis=0) / float(self._fibers) <= 0.05

        if bound is not None:
            wave = numpy.arange(self._data.shape[1])
            select_wave = numpy.logical_and(
                numpy.logical_and(wave >= bound[0], wave <= bound[1]), select_wave
            )
        change_dist = numpy.zeros(
            (self._fibers - 1, self._data.shape[1]), dtype=numpy.float32
        )  # empty array to store the fiber distances
        (init_dist, init_mask) = self.getFiberDist(
            start_slice
        )  # get initial fiber distances
        change_dist[
            :, start_slice
        ] = init_dist  # insert the initial distance into array
        first = numpy.arange(start_slice, -1, -1)
        x = (
            numpy.arange(self._fibers - 1) - self._fibers / 2.0
        )  # define the cross-disperion coordinate system
        fit_par = numpy.zeros(
            (len(poly_cross), self._data.shape[1]), dtype=numpy.float32
        )  # empty array to store the poly fit parameters in cross-dispersion direction
        fit_par[-1, start_slice] = 1.0
        # start iteration towards the dispersion column 0
        for i in first:
            if select_wave[i]:
                (dist, bad_mask) = self.getFiberDist(
                    i
                )  # get the fiber distance for the dispersion column i
                change = (
                    init_dist / dist
                )  # compute the relative change in the fiber distance compared to the reference dispersion column
                change_dist[:, i] = change  # store the changes into array
                good_mask = numpy.logical_not(bad_mask)
                select_good = numpy.logical_and(
                    numpy.logical_and(change > 0.5, change < 2.0), good_mask
                )  # masked unrealstic changes

                # select_good = numpy.logical_and(change>0.5, change<1.5) # masked unrealstic changes
                # fit = numpy.polyfit(x[select_good], change[select_good],  len(poly_cross)-1) # fit the relative change in the fiber distance with a polynomial of given order
                # res = change[select_good]-numpy.polyval(fit, x[select_good])
                # select = numpy.abs(res)<=3*numpy.std(res)
                # fit = numpy.polyfit(x[select_good][select], change[select_good][select],  len(poly_cross)-1) # fit the relative change in the fiber distance with a polynomial of given order
                select = numpy.logical_and(
                    numpy.logical_and(change > 0.5, change < 2.0), good_mask
                )  # masked unrealstic changes
                fit = numpy.polyfit(
                    x[select], change[select], len(poly_cross) - 1
                )  # fit the relative change in the fiber distance with a polynomial of given order
                fit_par[:, i] = fit  # store parameters into array

        second = numpy.arange(start_slice, self._data.shape[1], 1)
        # start iteration towards the last dispersion column
        for i in second:
            if select_wave[i]:
                (dist, mask) = self.getFiberDist(
                    i
                )  # get the fiber distance for the dispersion column i
                change = (
                    init_dist / dist
                )  # compute the relative change in the fiber distance compared to the reference dispersion column
                change_dist[:, i] = change  # store the changes into array
                good_mask = numpy.logical_not(bad_mask)
                select_good = numpy.logical_and(
                    numpy.logical_and(change > 0.5, change < 2.0), good_mask
                )  # masked unrealstic changes
                # select_good = numpy.logical_and(change>0.5, change<1.5) # masked unrealstic changes
                # fit = numpy.polyfit(x[select_good], change[select_good],  len(poly_cross)-1) # fit the relative change in the fiber distance with a polynomial of given order
                # res = change[select_good]-numpy.polyval(fit, x[select_good])
                # select = numpy.abs(res)<=3*numpy.std(res)
                # fit = numpy.polyfit(x[select_good][select], change[select_good][select],  len(poly_cross)-1) # fit the relative change in the fiber distance with a polynomial of given order
                select = numpy.logical_and(
                    numpy.logical_and(change > 0.5, change < 2.0), good_mask
                )  # masked unrealstic changes
                fit = numpy.polyfit(
                    x[select], change[select], len(poly_cross) - 1
                )  # fit the relative change in the fiber distance with a polynomial of given order
                fit_par[:, i] = fit  # store parameters into array

            if i == -3930:
                print(change)
                print(good_mask)
                plot.plt.plot(
                    x[select_good][select], change_dist[select_good, i][select], "ok"
                )
                plot.plt.plot(x, numpy.polyval(fit_par[:, i], x), "r")
                plot.plt.show()

        wave = numpy.arange(
            fit_par.shape[1]
        )  # create coordinates in dispersion direction
        #   print(wave[select_wave])
        fit_par_smooth = numpy.zeros_like(
            fit_par
        )  # empty array for the smooth polynomial fit parameters
        # iterate over the order of the fitted polynomial in cross-dispersion direction
        for j in range(len(poly_cross)):
            fit = numpy.polyfit(
                wave[select_wave] - fit_par.shape[1] / 2.0,
                fit_par[j, select_wave],
                poly_cross[len(poly_cross) - j - 1],
            )  # fit polynomial along dispersion axis
            res = fit_par[j, select_wave] - numpy.polyval(
                fit, wave[select_wave] - fit_par.shape[1] / 2.0
            )
            rms = numpy.std(res)
            select = numpy.abs(res) <= 2 * rms
            fit = numpy.polyfit(
                wave[select_wave][select] - fit_par.shape[1] / 2.0,
                fit_par[j, select_wave][select],
                poly_cross[len(poly_cross) - j - 1],
            )  # fit polynomial along dispersion axis
            fit_par_smooth[j, :] = numpy.polyval(
                fit, wave - fit_par.shape[1] / 2.0
            )  # store the resulting polynomial

            # plt.subplot(len(poly_cross), 1, len(poly_cross)-j)
            # plt.plot(wave[select_wave], res, '-k')
            # plt.plot(wave[select_wave], fit_par[j, select_wave], 'ok')
            # plt.plot(wave[select_wave][select], fit_par[j, select_wave][select], 'or')
            # plt.plot(wave, fit_par_smooth[j, :], '-r')
        # plt.show()

        for i in range(len(wave)):
            change_dist[:, i] = numpy.polyval(
                fit_par_smooth[:, i], x
            )  # replace the relative fiber distance with their polynomial smoothed values

        dist_new = (
            init_dist[:, numpy.newaxis] / change_dist
        )  # convert relative fiber distance back to absolute fiber distance with the reference
        new_trace = numpy.zeros_like(
            self._data
        )  # create empty array for the full trace mask
        new_trace[1:, :] = numpy.cumsum(
            dist_new, axis=0
        )  # create absolute positions with an arbitrary zero-point
        select_wave = numpy.sum(self._mask, axis=0) < self._fibers

        # offset1 = self._data[150, select_wave] - new_trace[150, select_wave]
        # offset2 = self._data[200, select_wave] - new_trace[200, select_wave]
        offset_mean = bn.median(
            self._data[:, select_wave] - new_trace[:, select_wave], axis=0
        )  # computes that absolut trace position between the initially measured and estimated trace to compute the zero-point
        # offset_rms = numpy.std(
        #     self._data[:, select_wave] - new_trace[:, select_wave], axis=0
        # )  # compute the rms scatter of the measured positions for each dispersion column
        fit_offset = numpy.polyfit(wave[select_wave], offset_mean, poly_disp)
        ext_offset = numpy.polyval(fit_offset, wave)
        #   plt.plot(wave[select_wave], offset_mean, 'ok')
        #  plt.plot(wave[select_wave], offset1, '-b')
        #  plt.plot(wave[select_wave], offset2, '-g')
        #  plt.plot(wave, ext_offset, '-r')
        #  plt.show()
        out_trace = new_trace + ext_offset[numpy.newaxis, :]  # match the trace offsets
        self._data = out_trace

    def get_coeffs(self):
        """Returns the polynomial coefficients"""
        return self._coeffs

    def set_coeffs(self, coeffs, poly_kind):
        """Sets the polynomial coefficients"""
        if coeffs is not None:
            self._coeffs = coeffs
            self._poly_kind = poly_kind
            self._poly_deg = coeffs.shape[1] - 1
        else:
            self._coeffs = None
            self._poly_kind = None
            self._poly_deg = None

    def eval_coeffs(self, pixels=None):
        """Evaluates the polynomial coefficients to the corresponding data values"""
        poly_cls = Spectrum1D.select_poly_class(self._poly_kind)

        if pixels is None:
            pixels = self._pixels

        if self._data is None:
            self._data = numpy.full((self._fibers, pixels.size), numpy.nan)
        for i in range(self._fibers):
            coeffs = self._coeffs[i, :]
            if not numpy.isfinite(coeffs).all():
                continue
            poly = poly_cls(coeffs)
            self._data[i, :] = poly(pixels)

        return self._data

    def interpolate_coeffs(self):
        """Interpolate coefficients or data of bad fibers

        Returns
        -------
        FiberRows
            Interpolated FiberRows object
        """
        # early return if no coefficients are available
        if self._coeffs is None:
            return self
        # early return if all fibers are masked
        bad_fibers = self._mask.all(axis=1)
        if bad_fibers.sum() == self._fibers:
            return self

        # define coordinates
        x_pixels = numpy.arange(self._data.shape[1])
        y_pixels = numpy.arange(self._fibers)

        # interpolate coefficients
        f_coeffs = interpolate.interp1d(y_pixels[~bad_fibers], self._coeffs[~bad_fibers, :], axis=0, bounds_error=False, fill_value="extrapolate")
        self._coeffs = f_coeffs(y_pixels)

        # evaluate trace at interpolated fibers
        for ifiber in y_pixels[bad_fibers]:
            poly = numpy.polynomial.Polynomial(self._coeffs[ifiber, :])
            self._data[ifiber, :] = poly(x_pixels)
            self._mask[ifiber, :] = False

        return self

    def interpolate_data(self, axis="Y", reset_mask=True):
        """Interpolate data of bad fibers (axis='Y') or bad pixels along the dispersion axis (axis='X')

        Parameters
        ----------
        axis : string or int, optional with default: 'Y'
            Defines the axis of the slice to be inserted, 'X', 'x', or 1 for the x-axis or
            'Y','y', or 0 for the y-axis.
        reset_mask : bool, optional with default: True
            If True, reset the mask of interpolated fibers to False

        Returns
        -------
        FiberRows
            Interpolated FiberRows object

        Raises
        ------
        ValueError
            If axis is not 'X', 'x', 1, 'Y', 'y', or 0
        """
        if self._mask is None:
            raise ValueError(f"Attribute `_mask` needs to be set: {self._mask = }")

        # define coordinates
        x_pixels = numpy.arange(self._data.shape[1])
        y_pixels = numpy.arange(self._fibers)

        # interpolate data
        if axis == "Y" or axis == "y" or axis == 0:
            slitmap = self._filter_slitmap()

            for block_idx in range(LVM_NBLOCKS):
                select_block = slitmap["blockid"] == f"B{block_idx+1}"
                y = y_pixels[select_block]
                data = self._data[select_block]
                mask = self._mask[select_block]

                bad_fibers = mask.all(axis=1)
                if bad_fibers.sum() == 0 or bad_fibers.sum() == LVM_BLOCKSIZE:
                    continue

                f_data = interpolate.interp1d(y[~bad_fibers], data[~bad_fibers], axis=0, bounds_error=False, fill_value="extrapolate")
                self._data[select_block] = f_data(y)
                if self._error is not None:
                    error = self._error[select_block]
                    f_error = interpolate.interp1d(y[~bad_fibers], error[~bad_fibers], axis=0, bounds_error=False, fill_value="extrapolate")
                    self._error[select_block] = f_error(y)

                # unmask interpolated fibers
                if reset_mask:
                    self._mask[select_block] = False
        elif axis == "X" or axis == "x" or axis == 1:
            for ifiber in y_pixels:
                bad_pixels = (self._data[ifiber] <= 0) | (self._mask[ifiber, :])
                # skip fiber if all pixels are bad and set mask to True
                if bad_pixels.all():
                    self._mask[ifiber] = True
                    continue
                # skip fiber if no bad pixels are present, no need to interpolate
                if bad_pixels.sum() == 0:
                    continue
                f_data = interpolate.interp1d(x_pixels[~bad_pixels], self._data[ifiber, ~bad_pixels], bounds_error=False, fill_value="extrapolate")
                self._data[ifiber, :] = f_data(x_pixels)
                if self._error is not None:
                    f_error = interpolate.interp1d(x_pixels[~bad_pixels], self._error[ifiber, ~bad_pixels], bounds_error=False, fill_value="extrapolate")
                    self._error[ifiber, :] = f_error(x_pixels)
                if self._mask is not None and reset_mask:
                    self._mask[ifiber, bad_pixels] = False
        else:
            raise ValueError(f"axis {axis} not supported")

        return self

    def plot_block(self, iblock=None, blockid=None, ref_column=None, show_samples=True, show_model_samples=True, show_model=True, axs=None):
        if iblock is None and blockid is None:
            block = copy(self)
        else:
            block = self.get_block(iblock=iblock, blockid=blockid)

        pixels = numpy.arange(block._data.shape[1], dtype="int")
        samples = block.get_samples(as_pandas=True)

        if axs is None:
            _, axs = plot.create_subplots(to_display=True, figsize=(15,5), layout="constrained")
        if not isinstance(axs, dict) or "mod" not in axs:
            axs = {"mod": axs}

        if ref_column is not None:
            axs["mod"].axvline(ref_column, ls=":", lw=1, color="0.7")
        if samples is not None:
            if show_samples:
                axs["mod"].plot(samples.columns, samples.T, ".", ms=5, mew=0, mfc="0.2", label="data")
            if show_model_samples:
                axs["mod"].plot(samples.columns, block._data[:, samples.columns].T, "s", ms=5, mew=1, mec="0.2", mfc="none", label="model@data")
        ylims = axs["mod"].get_ylim()
        if show_model:
            axs["mod"].plot(pixels, block._data.T, "-", lw=1, label="model")
        axs["mod"].set_ylim(*ylims)

        if "res" not in axs:
            axs["mod"].tick_params(labelbottom=False)

            ax_divider = plot.make_axes_locatable(axs["mod"])
            ax_res = ax_divider.append_axes("bottom", size="30%", pad="5%")
            ax_res.sharex(axs["mod"])

            axs["res"] = ax_res

        axs["res"].axhline(ls="--", lw=1, color="0.4")
        axs["res"].axhline(-0.01, ls=":", lw=1, color="0.4")
        axs["res"].axhline(+0.01, ls=":", lw=1, color="0.4")
        axs["res"].plot(samples.columns, ((block._data[:, samples.columns] - samples)/samples).T, ".-", lw=0.2, ms=5, mew=0)
        axs["res"].set_ylim(-0.05, +0.05)
        plot.plt.gcf().tight_layout()

        return axs

    def plot_block_distances(self, iblock=None, blockid=None, show_samples=True, show_model_samples=True, show_model=True, axs=None):
        if iblock is None and blockid is None:
            block = copy(self)
        else:
            block = self.get_block(iblock=iblock, blockid=blockid)

        pixels = numpy.arange(block._data.shape[1], dtype="int")
        samples = block.get_samples(as_pandas=True)

        sample_distances, model_distances = block.get_distances()

        if axs is None:
            _, axs = plot.create_subplots(to_display=True, figsize=(15,5), layout="constrained")

        if show_model:
            axs.plot(pixels, model_distances.T, "-", lw=1, label="model")
        if samples is not None:
            if show_samples:
                axs.plot(samples.columns, sample_distances.T, ".", ms=5, mew=0, mfc="0.2", label="data")
            if show_model_samples:
                axs.plot(samples.columns, model_distances[:, samples.columns].T, "s", ms=5, mew=1, mec="0.2", mfc="none", label="model@data")

    def plot_fiber(self, ifiber, show_samples=True, axs=None):
        fiber = self[ifiber]

        pixels = fiber._wave
        data = fiber._data
        error = fiber._error
        mask = fiber._mask

        if axs is None:
            _, axs = plot.create_subplots(to_display=True, figsize=(15,5), layout="constrained")

        if mask is not None:
            selection = ~mask
            axs.vlines(pixels[mask], numpy.nanmin(data[selection]), numpy.nanmax(data[selection]), lw=1, color="0.7")
        if error is not None:
            axs.errorbar(pixels, data, yerr=error, fmt="", elinewidth=1, ecolor="0.5")
        axs.step(pixels, data, lw=1, color="0.2")

        if show_samples and self._samples is not None:
            samples = self.get_samples(as_pandas=True)
            axs.plot(samples.columns, samples.iloc[ifiber], ".", color="0.2", mew=0, ms=7)

        return axs
