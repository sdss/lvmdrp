import numpy
from numpy import polynomial
from astropy.io import fits as pyfits
from scipy import interpolate
from tqdm import tqdm

from lvmdrp import log
from lvmdrp.core.header import Header, combineHdr
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.plot import plt


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


class FiberRows(Header, PositionTable):
    def __init__(
        self,
        data=None,
        header=None,
        error=None,
        mask=None,
        shape=None,
        size=None,
        arc_position_x=None,
        arc_position_y=None,
        good_fibers=None,
        fiber_type=None,
        coeffs=None,
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
        if data is None:
            self._data = None
        else:
            self._data = data.astype("float32")
            self._fibers = data.shape[0]
            self._pixels = numpy.arange(data.shape[1])

        if error is None:
            self._error = None
        else:
            self._error = numpy.array(error).astype("float32")

        if mask is None:
            self._mask = None
        else:
            self._mask = numpy.array(mask)

        if coeffs is None:
            self._coeffs = None
        else:
            self._coeffs = coeffs.astype("float32")

    def __len__(self):
        return self._fibers

    def __truediv__(self, other):
        """
        Operator to divide two Images or divide by another type if possible
        """
        if isinstance(other, self.__class__):
            # define behaviour if the other is of the same instance

            img = self.__class__(
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

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
                    + (self._data * other._error / other._data**2) ** 2
                )
                img.setData(error=new_error)
            elif self._error is not None and other._error is None:
                new_error = self._error / other._data
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
            img = self.__class__(
                error=self._error,
                mask=self._mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                # add ndarray according do its dimensions
                if self._data.shape == dim:
                    new_data = self._data / other
                elif len(dim) == 1:
                    if self._data.shape[0] == dim[0]:
                        new_data = self._data / other[:, numpy.newaxis]
                    elif self._data.shape[1] == dim[0]:
                        new_data = self._data / other[numpy.newaxis, :]
                else:
                    new_data = self._data
                if self._error is not None:
                    new_error = self._error / other
                else:
                    new_error = None
                img.setData(data=new_data, error=new_error)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            # try:
            new_data = self._data / other
            if self._error is not None:
                new_error = self._error / other
            else:
                new_error = None
            img = self.__class__(
                data=new_data,
                error=new_error,
                mask=self._mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )
            return img
        # except:
        # raise exception if the type are not matching in general
        #   raise exceptions.TypeError("unsupported operand type(s) for /: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __add__(self, other):
        """
        Operator to add two FiberRow or divide by another type if possible
        """
        if isinstance(other, self.__class__):
            # define behaviour if the other is of the same instance

            img = self.__class__(
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data + other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(self._error**2 + other._error**2)
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
            img = self.__class__(
                error=self._error,
                mask=self._mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

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

        elif isinstance(other, Spectrum1D):
            img = self.__class__(
                error=self._error,
                mask=self._mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

            if self._data is not None:  # check if there is data in the object
                # add ndarray according do its dimensions
                if self._fibers == other._dim:
                    new_data = self._data + other._data[:, numpy.newaxis]
                elif self._data.shape[1] == other._dim:
                    new_data = self._data + other._data[numpy.newaxis, :]
                else:
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data + other
                img = self.__class__(
                    data=new_data,
                    error=self._error,
                    mask=self._mask,
                    header=self._header,
                    shape=self._shape,
                    size=self._size,
                    arc_position_x=self._arc_position_x,
                    arc_position_y=self._arc_position_y,
                    good_fibers=self._good_fibers,
                    fiber_type=self._fiber_type,
                )
                return img
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for +: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

    def __mul__(self, other):
        """
        Operator to add two FiberRow or divide by another type if possible
        """
        if isinstance(other, self.__class__):
            # define behaviour if the other is of the same instance

            img = self.__class__(
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data * other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(
                    other._data**2 * self._error**2
                    + self._data**2 * other._error**2
                )
                img.setData(error=new_error)
            elif self._error is not None:
                new_error = other._data * self._error
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
            img = self.__class__(
                error=self._error,
                mask=self._mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

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
                new_data = self._data * other
                if self._error is not None:
                    new_error = self._error * other
                else:
                    new_error = self._error
                img = self.__class__(
                    data=new_data,
                    error=new_error,
                    mask=self._mask,
                    header=self._header,
                    shape=self._shape,
                    size=self._size,
                    arc_position_x=self._arc_position_x,
                    arc_position_y=self._arc_position_y,
                    good_fibers=self._good_fibers,
                    fiber_type=self._fiber_type,
                )
                return img
            except Exception:
                # raise exception if the type are not matching in general
                raise TypeError(
                    "unsupported operand type(s) for *: %s and %s"
                    % (str(type(self)).split("'")[1], str(type(other)).split("'")[1])
                )

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

    def createEmpty(self, data_dim=None, error_dim=None, mask_dim=None):
        """
        Fill the FiberRows object with empty data

        Parameters
        --------------
        data_dim: tuple, optional with default: None
            Dimension of the empty data array to be created

        error_dim : tuple, optional with default: None
            Dimension of the empty error array to be created

        mask_dim : tuple, optional with default: None
            Dimension of the bad pixel mask to be created (all pixel masked bad)

        """
        if data_dim is not None:
            # create empty  data array and set number of fibers
            self._data = numpy.zeros(data_dim, dtype=numpy.float32)
            self._fibers = self._data.shape[0]

        if error_dim is not None:
            # create empty  error array
            self._error = numpy.zeros(error_dim, dtype=numpy.float32)

        if mask_dim is not None:
            # create empty mask all pixel assigned bad
            self._mask = numpy.ones(mask_dim, dtype="bool")

    def setFibers(self, fibers):
        """
        Set the number of fibers

        Parameters
        --------------
        fibers: int
            Number of fibers
        """
        self._fibers = fibers

    def setSlice(self, slice, axis="x", data=None, error=None, mask=None, select=None):
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
            numpy.arange(self._data.shape[1]), data, error=error, mask=mask
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
            if data is not None:
                self._data = data
                nfibers, npixels = data.shape
            if mask is not None:
                self._mask = mask
                nfibers, npixels = mask.shape
            if error is not None:
                self._error = error
                nfibers, npixels = error.shape

            self._fibers = nfibers
            self._pixels = numpy.arange(npixels)

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

    def loadFitsData(
        self,
        file,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_coeffs=None,
        extension_hdr=None,
    ):
        """
        Load data from a FITS image into an FiberRows object (Fibers in y-direction, dispersion in x-direction)

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
            and extension_coeffs is None
        ):
            self._data = hdu[0].data.astype("float32")
            self._fibers = self._data.shape[0]
            self._pixels = numpy.arange(self._data.shape[1])
            self.setHeader(hdu[0].header)
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header["EXTNAME"].split()[0] == "ERROR":
                        self._error = hdu[i].data.astype("float32")
                    elif hdu[i].header["EXTNAME"].split()[0] == "BADPIX":
                        self._mask = hdu[i].data.astype("bool")
                        self._good_fibers = numpy.where(numpy.sum(self._mask, axis=1) != self._data.shape[1])[0]
                    elif hdu[i].header["EXTNAME"].split()[0] == "COEFFS":
                        self._coeffs = hdu[i].data.astype("float32")

        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data.astype("float32")
                self._fibers = self._data.shape[0]
                self._pixels = numpy.arange(self._data.shape[1])
            if extension_mask is not None:
                self._mask = hdu[extension_mask].data.astype("bool")
                self._good_fibers = numpy.where(numpy.sum(self._mask, axis=1) != self._data.shape[1])[0]
            if extension_error is not None:
                self._error = hdu[extension_error].data.astype("float32")
            if extension_coeffs is not None:
                self._coeffs = hdu[extension_coeffs].data.astype("float32")
        
        hdu.close()
        
        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header)

    def applyFibers(self, function, args):
        result = []
        for i in range(len(self)):
            result.append(function(args))
        return result

    def writeFitsData(
        self,
        filename,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_coeffs=None,
    ):
        """
        Save information from a FiberRows object into a FITS file.
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
        if self._coeffs is not None:
            self._coeffs = self._coeffs.astype("float32")

        hdus = [None, None, None, None]

        # create primary hdus and image hdus
        # data hdu
        if (
            extension_data is None
            and extension_error is None
            and extension_mask is None
            and extension_coeffs is None
        ):
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name="ERROR")
            if self._mask is not None:
                hdus[2] = pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX")
            if self._coeffs is not None:
                hdus[3] = pyfits.ImageHDU(self._coeffs, name="COEFFS")
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

            # polynomial trace hdu
            if extension_coeffs == 0:
                hdu = pyfits.PrimaryHDU(self._coeffs)
            elif extension_coeffs > 0 and extension_coeffs is not None:
                hdus[extension_coeffs] = pyfits.ImageHDU(self._coeffs, name="COEFFS")

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except Exception:
                break

        if len(hdus) > 0:
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
            if self._header is not None:
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                hdu[0].update_header()
        try:
            del hdu[0]._header["COMMENT"]
        except KeyError:
            pass
        try:
            del hdu[0]._header["HISTORY"]
        except KeyError:
            pass
        hdu.writeto(filename, output_verify="silentfix", overwrite=True)

    def measureArcLines(
        self,
        ref_fiber,
        ref_cent,
        aperture=12,
        init_back=30.0,
        flux_min=100,
        fwhm_max=10,
        rel_flux_limits=[0.2, 5],
        axs=None,
    ):
        nlines = len(ref_cent)
        cent_wave = numpy.zeros((self._fibers, nlines), dtype=numpy.float32)
        fwhm = numpy.zeros((self._fibers, nlines), dtype=numpy.float32)
        flux = numpy.zeros((self._fibers, nlines), dtype=numpy.float32)
        masked = numpy.zeros((self._fibers, nlines), dtype="bool")

        spec = self.getSpec(ref_fiber)
        fit = spec.fitSepGauss(ref_cent, aperture, init_back, axs=axs)
        masked[ref_fiber, :] = False
        flux[ref_fiber, :] = fit[:nlines]
        ref_flux = flux[ref_fiber, :]
        cent_wave[ref_fiber, :] = fit[nlines : 2 * nlines]
        fwhm[ref_fiber, :] = fit[2 * nlines : 3 * nlines] * 2.354
        first = numpy.arange(ref_fiber - 1, -1, -1)
        second = numpy.arange(ref_fiber + 1, self._fibers, 1)

        iterator = tqdm(
            first,
            total=first.size,
            desc=f"measuring arc lines upwards from {ref_fiber = }",
            ascii=True,
            unit="fiber",
        )
        for i in iterator:
            spec = self.getSpec(i)

            fit = spec.fitSepGauss(cent_wave[i + 1], aperture, init_back, axs=None)
            flux[i, :] = numpy.fabs(fit[:nlines])
            cent_wave[i, :] = fit[nlines : 2 * nlines]
            fwhm[i, :] = fit[2 * nlines : 3 * nlines] * 2.354

            rel_flux_med = numpy.nanmedian(flux[i, :] / ref_flux)
            if (
                rel_flux_med < rel_flux_limits[0]
                or rel_flux_med > rel_flux_limits[1]
                or numpy.nanmedian(fwhm[i, :]) > fwhm_max
            ):
                select = numpy.ones(len(flux[i, :]), dtype="bool")
            else:
                select = numpy.logical_or(
                    numpy.logical_or(
                        flux[i, :] < flux_min,
                        flux[i, :] / ref_flux > rel_flux_limits[1],
                    ),
                    fwhm[i, :] > fwhm_max,
                )

            if numpy.nansum(select) > 0:
                cent_wave[i, select] = cent_wave[i + 1, select]
                fwhm[i, select] = fwhm[i + 1, select]
                masked[i, select] = True

        iterator = tqdm(
            second,
            total=second.size,
            desc=f"measuring arc lines downwards from {ref_fiber = }",
            ascii=True,
            unit="fiber",
        )
        for i in iterator:
            spec = self.getSpec(i)
            
            fit = spec.fitSepGauss(cent_wave[i - 1], aperture, init_back, axs=None)
            flux[i, :] = numpy.fabs(fit[:nlines])
            cent_wave[i, :] = fit[nlines : 2 * nlines]
            fwhm[i, :] = fit[2 * nlines : 3 * nlines] * 2.354

            rel_flux_med = numpy.nanmedian(flux[i, :] / ref_flux)
            if (
                rel_flux_med < rel_flux_limits[0]
                or rel_flux_med > rel_flux_limits[1]
                or numpy.nanmedian(fwhm[i, :]) > fwhm_max
            ):
                select = numpy.ones(len(flux[i, :]), dtype="bool")
            else:
                select = numpy.logical_or(
                    numpy.logical_or(
                        flux[i, :] < flux_min,
                        flux[i, :] / ref_flux > rel_flux_limits[1],
                    ),
                    fwhm[i, :] > fwhm_max,
                )

            if numpy.nansum(select) > 0:
                cent_wave[i, select] = cent_wave[i - 1, select]
                fwhm[i, select] = fwhm[i - 1, select]
                masked[i, select] = True

        fibers = numpy.arange(self._fibers)
        for i in range(nlines):
            select_line = masked[:, i]
            bad_fibers = fibers[select_line]
            good_fibers = fibers[numpy.logical_not(select_line)]
            for j in bad_fibers:
                nearest = numpy.abs(good_fibers - j)
                sorted = numpy.argsort(nearest)
                greater = good_fibers[sorted][good_fibers[sorted] > j]
                smaller = good_fibers[sorted][good_fibers[sorted] < j]

                if len(smaller) == 0:
                    cent_wave[j, i] = cent_wave[greater[0], i]
                    fwhm[j, i] = fwhm[greater[0], i]
                elif len(greater) == 0:
                    cent_wave[j, i] = cent_wave[smaller[0], i]
                    fwhm[j, i] = fwhm[smaller[0], i]
                else:
                    cent_wave[j, i] = (
                        cent_wave[smaller[0], i] + cent_wave[greater[0], i]
                    ) / 2.0
                    fwhm[j, i] = (fwhm[smaller[0], i] + fwhm[greater[0], i]) / 2.0

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

    def fit_polynomial(self, deg, poly_kind="poly", clip=None):
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
        """
        pixels = numpy.arange(
            self._data.shape[1]
        )  # pixel position in dispersion direction
        self._coeffs = numpy.zeros((self._data.shape[0], numpy.abs(deg) + 1))
        # iterate over each fiber
        pix_table = []
        poly_table = []
        poly_all_table = []
        for i in range(self._fibers):
            good_pix = numpy.logical_not(self._mask[i, :])
            if numpy.sum(good_pix) >= deg + 1:
                # select the polynomial class
                if poly_kind == "poly":
                    poly_cls = polynomial.Polynomial
                elif poly_kind == "legendre":
                    poly_cls = polynomial.Legendre
                elif poly_kind == "chebyshev":
                    poly_cls = polynomial.Chebyshev

                # try to fit
                try:
                    poly = poly_cls.fit(pixels[good_pix], self._data[i, good_pix], deg=deg)
                    pix_table.extend(numpy.column_stack([pixels[good_pix], self._data[i, good_pix]]).tolist())
                    poly_table.extend(numpy.column_stack([pixels[good_pix], poly(pixels[good_pix])]).tolist())
                    poly_all_table.extend(numpy.column_stack([pixels, poly(pixels)]).tolist())
                except numpy.linalg.LinAlgError as e:
                    log.error(f'Fiber trace failure at fiber {i}: {e}')
                    self._mask[i, :] = True
                    continue

                self._coeffs[i, :] = poly.convert().coef
                self._data[i, :] = poly(pixels)

                if clip is not None:
                    self._data = numpy.clip(self._data, clip[0], clip[1])
                self._mask[i, :] = False
            else:
                self._mask[i, :] = True

        return pix_table, poly_table, poly_all_table

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
                plt.plot(
                    x[select_good][select], change_dist[select_good, i][select], "ok"
                )
                plt.plot(x, numpy.polyval(fit_par[:, i], x), "r")
                plt.show()

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
        offset_mean = numpy.median(
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
        
        # define coordinates
        x_pixels = numpy.arange(self._data.shape[1])
        y_pixels = numpy.arange(self._fibers)

        # interpolate coefficients
        bad_fibers = self._mask.all(axis=1)
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
        # define coordinates
        x_pixels = numpy.arange(self._data.shape[1])
        y_pixels = numpy.arange(self._fibers)

        # interpolate data
        if axis == "Y" or axis == "y" or axis == 0:
            bad_fibers = self._mask.all(axis=1)
            f_data = interpolate.interp1d(y_pixels[~bad_fibers], self._data[~bad_fibers, :], axis=0, bounds_error=False)
            self._data = f_data(y_pixels)
            if self._error is not None:
                f_error = interpolate.interp1d(y_pixels[~bad_fibers], self._error[~bad_fibers, :], axis=0, bounds_error=False)
                self._error = f_error(y_pixels)

            # unmask interpolated fibers
            if self._mask is not None:
                self._mask[bad_fibers, :] = False
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
                f_data = interpolate.interp1d(x_pixels[~bad_pixels], self._data[ifiber, ~bad_pixels], bounds_error=False)
                self._data[ifiber, :] = f_data(x_pixels)
                if self._error is not None:
                    f_error = interpolate.interp1d(x_pixels[~bad_pixels], self._error[ifiber, ~bad_pixels], bounds_error=False)
                    self._error[ifiber, :] = f_error(x_pixels)
                if self._mask is not None and reset_mask:
                    self._mask[ifiber, bad_pixels] = False
        else:
            raise ValueError(f"axis {axis} not supported")

        return self