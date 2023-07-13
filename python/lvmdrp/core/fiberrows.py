import numpy
from astropy.io import fits as pyfits
from tqdm import tqdm

from lvmdrp.core.header import Header, combineHdr
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.spectrum1d import Spectrum1D


def _read_fiber_ypix(peaks_file):
    """
    Read peaks file and return the fiber number, pixel position, subpixel position
    and quality flag.
    """
    peaks = pyfits.open(peaks_file)
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
            self._data = data
            self._fibers = data.shape[0]
            self._pixels = numpy.arange(data.shape[1])

        if error is None:
            self._error = None
        else:
            self._error = numpy.array(error)

        if mask is None:
            self._mask = None
        else:
            self._mask = numpy.array(mask)

        if coeffs is None:
            self._coeffs = None
        else:
            self._coeffs = coeffs

    def __len__(self):
        return self._fibers

    def __truediv__(self, other):
        """
        Operator to divide two Images or divide by another type if possible
        """
        if isinstance(other, FiberRows):
            # define behaviour if the other is of the same instance

            img = FiberRows(
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
            img = FiberRows(
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
                    new_data = self._data / other
                elif len(dim) == 1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data / other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
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
            img = FiberRows(
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
        if isinstance(other, FiberRows):
            # define behaviour if the other is of the same instance

            img = FiberRows(
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
            img = FiberRows(
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
            img = FiberRows(
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
                img = FiberRows(
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
        if isinstance(other, FiberRows):
            # define behaviour if the other is of the same instance

            img = FiberRows(
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
            img = FiberRows(
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
                img = FiberRows(
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

        return FiberRows(data=data, error=error, mask=mask)

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
                nfibers, npixels = data.shape
            if mask is not None:
                self._mask[select] = mask
                nfibers, npixels = mask.shape
            if error is not None:
                self._error[select] = error
                nfibers, npixels = error.shape
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
                FiberRows(data=split_data[i], error=split_error[i], mask=split_mask[i])
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
        hdu = pyfits.open(file, uint=True, do_not_scale_image_data=True)
        if (
            extension_data is None
            and extension_mask is None
            and extension_error is None
            and extension_coeffs is None
        ):
            self._data = hdu[0].data
            self._fibers = self._data.shape[0]
            self._pixels = numpy.arange(self._data.shape[1])
            self.setHeader(hdu[0].header)
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header["EXTNAME"].split()[0] == "ERROR":
                        self._error = hdu[i].data
                    elif hdu[i].header["EXTNAME"].split()[0] == "BADPIX":
                        self._mask = hdu[i].data.astype(bool)
                    elif hdu[i].header["EXTNAME"].split()[0] == "COEFFS":
                        self._coeffs = hdu[i].data

        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data
                self._fibers = self._data.shape[0]
                self._pixels = numpy.arange(self._data.shape[1])
            if extension_mask is not None:
                self._mask = hdu[extension_mask].data.astype(bool)
            if extension_error is not None:
                self._error = hdu[extension_error].data
            if extension_coeffs is not None:
                self._coeffs = hdu[extension_coeffs].data
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
        hdu.writeto(
            filename, output_verify="silentfix", overwrite=True
        )  # write FITS file to disc

    def measureArcLines(
        self,
        ref_fiber,
        ref_cent,
        aperture=12,
        init_back=30.0,
        flux_min=100,
        fwhm_max=10,
        rel_flux_limits=[0.2, 5],
        verbose=True,
    ):
        nlines = len(ref_cent)
        cent_wave = numpy.zeros((self._fibers, nlines), dtype=numpy.float32)
        fwhm = numpy.zeros((self._fibers, nlines), dtype=numpy.float32)
        flux = numpy.zeros((self._fibers, nlines), dtype=numpy.float32)
        masked = numpy.zeros((self._fibers, nlines), dtype="bool")

        spec = self.getSpec(ref_fiber)
        fit = spec.fitSepGauss(ref_cent, aperture, init_back)
        masked[ref_fiber, :] = False
        flux[ref_fiber, :] = fit[:nlines]
        ref_flux = flux[ref_fiber, :]
        cent_wave[ref_fiber, :] = fit[nlines : 2 * nlines]
        fwhm[ref_fiber, :] = fit[2 * nlines : 3 * nlines] * 2.354
        first = numpy.arange(ref_fiber - 1, -1, -1)
        second = numpy.arange(ref_fiber + 1, self._fibers, 1)

        if verbose:
            iterator = tqdm(
                first,
                total=first.size,
                desc=f"measuring arc lines upwards from {ref_fiber = }",
                ascii=True,
                unit="fiber",
            )
        else:
            iterator = first
        plot = False
        for i in iterator:
            spec = self.getSpec(i)

            fit = spec.fitSepGauss(cent_wave[i + 1], aperture, init_back, plot=plot)
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
            else:
                plot = False

        if verbose:
            iterator = tqdm(
                second,
                total=second.size,
                desc=f"measuring arc lines downwards from {ref_fiber = }",
                ascii=True,
                unit="fiber",
            )
        else:
            iterator = second
        for i in iterator:
            spec = self.getSpec(i)
            if i == 10:
                plot = True
            else:
                plot = False
            fit = spec.fitSepGauss(cent_wave[i - 1], aperture, init_back, plot=plot)
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
