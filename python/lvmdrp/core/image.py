import numpy
from astropy.io import fits as pyfits

from lvmdrp.core.apertures import *
from lvmdrp.core.header import *
from lvmdrp.core.spectrum1d import Spectrum1D


try:
    import pylab
except:
    pass
from multiprocessing import Pool, cpu_count

from astropy.modeling import fitting, models
from scipy import ndimage


def _parse_ccd_section(section):
    """Parse a CCD section in the format [1:NCOL, 1:NROW] to python tuples"""
    slice_x, slice_y = section.strip("[]").split(",")
    slice_x = list(map(lambda str: int(str) - 1, slice_x.split(":")))
    slice_y = list(map(lambda str: int(str) - 1, slice_y.split(":")))
    return slice_x, slice_y


class Image(Header):
    def __init__(self, data=None, header=None, mask=None, error=None, origin=None):
        Header.__init__(self, header=header, origin=origin)
        self._data = data
        if self._data is not None:
            self._dim = self._data.shape
        else:
            self._dim = None
        self._mask = mask
        self._error = error
        self._origin = origin

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
                    self._error.astype(numpy.float64) ** 2
                    + other._error.astype(numpy.float64) ** 2
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
            img = Image(
                error=self._error,
                mask=self._mask,
                header=self._header,
                origin=self._origin,
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
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data + other
                img = Image(
                    data=new_data,
                    error=self._error,
                    mask=self._mask,
                    header=self._header,
                    origin=self._origin,
                )
                return img
            except:
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

            img = Image(header=self._header, origin=self._origin)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data - other._data
                img.setData(data=new_data)
            else:
                img.setData(data=self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(
                    self._error.astype(numpy.float64) ** 2
                    + other._error.astype(numpy.float64) ** 2
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
            img = Image(
                error=self._error,
                mask=self._mask,
                header=self._header,
                origin=self._origin,
            )

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
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data - other
                img = Image(
                    data=new_data,
                    error=self._error,
                    mask=self._mask,
                    header=self._header,
                    origin=self._origin,
                )
                return img
            except:
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

            img = Image(header=self._header, origin=self._origin)

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
            img = Image(
                error=self._error,
                mask=self._mask,
                header=self._header,
                origin=self._origin,
            )

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
                img = Image(
                    data=new_data,
                    error=new_error,
                    mask=self._mask,
                    header=self._header,
                    origin=self._origin,
                )
                return img
            except:
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

            img = Image(header=self._header, origin=self._origin)

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
            img = Image(
                error=self._error,
                mask=self._mask,
                header=self._header,
                origin=self._origin,
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
                img = Image(
                    data=new_data,
                    error=self._error,
                    mask=self._mask,
                    header=self._header,
                    origin=self._origin,
                )
                return img
            except:
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

        return Image(data=data, header=header, error=error, mask=mask)

    def setSection(self, section, subimg, update_header=False, inplace=True):
        sec_x, sec_y = _parse_ccd_section(section)

        new_image = (
            self
            if inplace
            else Image(
                data=self._data,
                header=self._header,
                mask=self._mask,
                error=self._error,
                origin=self._origin,
            )
        )

        new_image._data[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]] = subimg._data
        if new_image._error is not None:
            new_image._error[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]] = subimg._error
        if new_image._mask is not None:
            new_image._mask[sec_y[0] : sec_y[1], sec_x[0] : sec_x[1]] = subimg._mask

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

    def setData(self, data=None, error=None, mask=None, header=None, select=None):
        """
        Set data for an Image. Specific data values can replaced according to a specific selection.

        Parameters
        --------------
        data : numpy.ndarray(float), optional with default = None
            array corresponding to the data to be set
        error : numpy.ndarray(float), optional with default = None
            array corresponding to the data to be set
        mask : numpy.ndarray(bool), optional with default = None
            array corresponding to the bad pixel to be set
        header : Header object, optional with default = None
        select : numpy.ndarray(bool), optional with default = None
            array defining the selection of pixel to be set

        """
        # if not select given set the full image
        if select is None:
            if data is not None:
                self._data = data  # set data if given
                self._dim = data.shape  # set dimension

            if mask is not None:
                self._mask = mask  # set mask if given
                self._dim = mask.shape  # set dimension

            if error is not None:
                self._error = error  # set mask if given
                self._dim = error.shape  # set dimension
            if header is not None:
                self.setHeader(header)  # set header
        else:
            # with select definied only partial data are set
            if data is not None:
                self._data[select] = data
            if mask is not None:
                self._mask[select] = mask
            if error is not None:
                self._error[select] = error
            if header is not None:
                self.setHeader(header)  # set header

    def convertUnit(
        self, unit, assume="adu", gain_field="GAIN", assume_gain=1.0, inplace=True
    ):
        current = self._header.get("BUNIT", assume)

        new_image = (
            self
            if inplace
            else Image(
                data=self._data,
                header=self._header,
                mask=self._mask,
                error=self._error,
                origin=self._origin,
            )
        )
        if current != unit:
            gains = self._header[f"AMP? {gain_field}"]
            sects = self._header[f"AMP? TRIMSEC"]
            n_amp = len(gains)
            for i in range(n_amp):
                factor = (
                    gains[f"AMP{i+1} {gain_field}"]
                    if current == "adu"
                    else 1 / gains[f"AMP{i+1} {gain_field}"]
                )
                new_image.setSection(
                    section=sects[i],
                    subimg=new_image.getSection(section=sects[i]) * factor,
                    update_header=False,
                    inplace=True,
                )
            else:
                factor = (
                    self._header.get(gain_field, assume_gain)
                    if current == "adu"
                    else 1 / self._header.get(gain_field, assume_gain)
                )
                new_image *= factor

            new_image._header["BUNIT"] = unit

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
        bias_overscan = numpy.median(overscan)
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
        hdu = pyfits.open(
            filename, ignore_missing_end=True, uint=False
        )  # open FITS file
        if ".fz" in filename[-4:]:
            extension_data = 1
            extension_header = 1
        if (
            extension_data is None
            and extension_mask is None
            and extension_error is None
        ):
            self._data = hdu[0].data
            self._dim = self._data.shape  # set dimension
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header["EXTNAME"].split()[0] == "ERROR":
                        self._error = hdu[i].data
                    elif hdu[i].header["EXTNAME"].split()[0] == "BADPIX":
                        self._mask = hdu[i].data.astype("bool")

        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data  # take data
                self._dim = self._data.shape  # set dimension

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data.astype("bool")  # take data
                self._dim = self._mask.shape  # set dimension

            if extension_error is not None:
                self._error = hdu[extension_error].data  # take data
                self._dim = self._error.shape  # set dimension

        self.setHeader(
            hdu[extension_header].header
        )  # get header  from the first FITS extension
        hdu.close()

    def writeFitsData(
        self, filename, extension_data=None, extension_mask=None, extension_error=None
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
        hdus = [None, None, None]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if (
            extension_data is None
            and extension_error is None
            and extension_mask is None
        ):
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name="ERROR")
            if self._mask is not None:
                hdus[2] = pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX")
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

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except:
                break
        # if len(hdus)>1:
        #    hdus[0].update_ext_name('T')

        if len(hdus) > 0:
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
            if self._header is not None:
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                try:
                    hdu[0].header["BZERO"] = 0
                except KeyError:
                    pass
                hdu[0].update_header()

        hdu.writeto(
            filename, output_verify="silentfix", overwrite=True
        )  # write FITS file to disc

    def computePoissonError(self, rdnoise=0.0, replace_masked=1e20):
        image = self._data
        self._error = numpy.zeros_like(image)
        select = image > 0
        self._error[select] = numpy.sqrt(image[select] + rdnoise**2)
        self._error[numpy.logical_not(select)] = rdnoise
        if self._mask is not None and replace_masked != 0:
            self._error[self._mask] = replace_masked

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
        idx = numpy.indices(self._dim)  # create an index array
        # get x and y coordinates of bad pixels

        y_cors = idx[0, self._mask]
        x_cors = idx[1, self._mask]

        out_data = self._data
        out_error = self._error

        # esimate the pixel distance form the bad pixel to the filter window boundary
        delta_x = int(numpy.ceil(box_x / 2.0))
        delta_y = int(numpy.ceil(box_y / 2.0))

        # iterate over bad pixels
        for m in range(len(y_cors)):
            # computes the min and max pixels of the filter window in x and y
            range_y = numpy.clip(
                [y_cors[m] - delta_y, y_cors[m] + delta_y + 1], 0, self._dim[0] - 1
            )
            range_x = numpy.clip(
                [x_cors[m] - delta_x, x_cors[m] + delta_x + 1], 0, self._dim[1] - 1
            )
            # compute the masked median within the filter window and replace data
            select = self._mask[range_y[0] : range_y[1], range_x[0] : range_x[1]] == 0
            out_data[y_cors[m], x_cors[m]] = numpy.median(
                self._data[range_y[0] : range_y[1], range_x[0] : range_x[1]][select]
            )
            if self._error is not None and replace_error is not None:
                # replace the error of bad pixel if defined
                out_error[y_cors[m], x_cors[m]] = replace_error

        # fill nan values and update mask
        out_mask = numpy.logical_or(self._mask, numpy.isnan(out_data))
        out_data = numpy.nan_to_num(out_data)
        # create new Image object
        new_image = Image(
            data=out_data, error=out_error, mask=out_mask, header=self._header
        )
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
        f = pyfits.open(fieldPhot)
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
                sky = numpy.median(calibratedImage)
            # print('Sky Background %s: %.2f Counts' %(filters[filter_select][0],sky))
            calibratedImage = calibratedImage - sky
            error = numpy.sqrt((calibratedImage + sky) / gain + dark_var)
        else:
            error = numpy.sqrt((calibratedImage) / gain + dark_var)
        self.setHdrValue("FIELD", 0)
        sdssImage = Image(
            data=calibratedImage * factor, error=error * factor, header=self._header
        )
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
                Image(data=split_data[i], error=split_error[i], mask=split_mask[i])
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
        new_image = Image(data=new_data, error=new_error, mask=new_mask)
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
        new_image = Image(
            data=new,
            error=new_error,
            mask=self._mask,
            header=self._header,
            origin=self._origin,
        )
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

        if self._mask is not None and mask == True:
            mask_data = self._data[self._mask]
            self._data[self._mask] = 0
            gauss = ndimage.filters.gaussian_filter(
                self._data, (sigma_y, sigma_x), mode=mode
            )
            scale = ndimage.filters.gaussian_filter(
                (self._mask == False).astype("float32"), (sigma_y, sigma_x), mode=mode
            )
            new = gauss / scale
            self._data[self._mask] = mask_data
        else:
            new = ndimage.filters.gaussian_filter(
                self._data, (sigma_y, sigma_x), mode=mode
            )
        # create new Image object with the error and the mask unchanged and return
        new_image = Image(
            data=new,
            error=self._error,
            mask=self._mask,
            header=self._header,
            origin=self._origin,
        )
        return new_image

    def medianImg(self, size, mode="nearest", use_mask=False):
        """
        Return a new Image that has been median filtered with a filter window of given size.

        Parameters
        --------------
        size : tuple of int
            Size of the filter window
        mode : string, optional with default: nearest
            Set the mode how to handle the boundarys within the convolution
            Possilbe modes are: reflect, constant, nearest, mirror,  wrap

        Returns
        -----------
        image :  Image object
            An Image object with the median filter data
        """
        if self._mask is None and use_mask is True:
            new_data = ndimage.filters.median_filter(
                self._data, size, mode=mode
            )  # applying the median filter
            new_mask = None
        elif self._mask is not None and use_mask is False:
            new_data = ndimage.filters.median_filter(
                self._data, size, mode=mode
            )  # applying the median filter
            new_mask = self._mask
        else:
            self._data[self._mask == 1] = numpy.nan
            new_data = ndimage.filters.generic_filter(
                self._data, numpy.nanmedian, size, mode=mode
            )
            new_mask = numpy.isnan(new_data)

        image = Image(
            data=new_data, header=self._header, error=self._error, mask=new_mask
        )  # create a new Image object
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
            return Spectrum1D(numpy.arange(dim), numpy.mean(self._data, axis))
        elif mode == "sum":
            return Spectrum1D(numpy.arange(dim), numpy.sum(self._data, axis))
        elif mode == "nansum":
            return Spectrum1D(numpy.arange(dim), numpy.nansum(self._data, axis))
        elif mode == "median":
            return Spectrum1D(numpy.arange(dim), numpy.median(self._data, axis))
        elif mode == "min":
            return Spectrum1D(numpy.arange(dim), numpy.amin(self._data, axis))
        elif mode == "max":
            return Spectrum1D(numpy.arange(dim), numpy.amax(self._data, axis))

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
        x = x - numpy.mean(x)
        # if self._mask is not None:
        #    self._mask = numpy.logical_and(self._mask, numpy.logical_not(numpy.isnan(self._data)))
        valid = self._mask.astype("bool") == False
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
                        pylab.plot(x, self._data[:, i], "-b")
                        pylab.plot(
                            x[valid[:, i]][select],
                            self._data[valid[:, i], i][select],
                            "ok",
                        )
                        max = numpy.max(self._data[valid[:, i], i][select])
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
                    pylab.plot(x[select], self._data[:, i][select], "ok")
                fit_result[:, i] = numpy.polyval(
                    fit_par[:, i], x
                )  # evalute the polynom
            if plot == i:
                pylab.plot(x, fit_result[:, i], "-r")
                pylab.ylim([0, max])
                pylab.show()
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
        new_img = Image(
            data=fit_result, error=self._error, header=self._header, mask=new_mask
        )
        return new_img

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
            mask[:, i] = bad_pix[:, i]
        return data, error, mask

    def extractSpecOptimal(self, TraceMask, TraceFWHM, plot_fig=False):
        data = numpy.zeros((TraceMask._fibers, self._dim[1]), dtype=numpy.float32)
        if self._error is not None:
            error = numpy.zeros((TraceMask._fibers, self._dim[1]), dtype=numpy.float32)
        else:
            error = None
        mask = numpy.zeros((TraceMask._fibers, self._dim[1]), dtype="bool")

        TraceFWHM = TraceFWHM / 2.354

        for i in range(self._dim[1]):
            slice_img = self.getSlice(i, axis="y")
            slice_trace = TraceMask.getSlice(i, axis="y")
            trace = slice_trace[0]
            bad_fiber = numpy.logical_or(
                (slice_trace[2] == 1),
                numpy.logical_or(
                    slice_trace[0] < 0, slice_trace[0] > len(slice_img._data) - 1
                ),
            )
            good_fiber = numpy.logical_not(bad_fiber)
            fwhm = TraceFWHM.getSlice(i, axis="y")[0]
            select_nan = numpy.isnan(slice_img._data)
            slice_img._data[select_nan] = 0
            indices = numpy.indices((self._dim[0], numpy.sum(good_fiber)))
            result = slice_img.obtainGaussFluxPeaks(
                trace[good_fiber], fwhm[good_fiber], indices, plot=plot_fig)
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

    def createCosmicMask(
        self,
        sigma_det=5,
        flim=1.1,
        iter=3,
        sig_gauss=(0.8, 0.8),
        error_box=(20, 2),
        replace_box=(20, 2),
        parallel="auto",
    ):
        """Return the cosmic ray pixel mask computed using the LA algorithm"""
        err_box_x = error_box[0]
        err_box_y = error_box[1]
        sigma_x = sig_gauss[0]
        sigma_y = sig_gauss[1]
        box_x = replace_box[0]
        box_y = replace_box[1]

        # create a new Image instance to store the initial data array
        out = Image(
            data=self.getData(),
            header=self.getHeader(),
            error=None,
            mask=numpy.zeros(self.getDim(), dtype=bool),
        )
        out.convertUnit("e-")
        # out.removeError()

        # initial CR selection
        select = numpy.zeros_like(out._mask, dtype=bool)

        # define Laplacian convolution kernel
        LA_kernel = (
            numpy.array(
                [
                    [
                        0,
                        -1,
                        0,
                    ],
                    [-1, 4, -1],
                    [0, -1, 0],
                ]
            )
            / 4.0
        )

        if parallel == "auto":
            cpus = cpu_count()
        else:
            cpus = int(parallel)

        # get rdnoise from header or assume given value
        quads = list(self._header["AMP? TRIMSEC"].values())
        rdnoises = list(self._header["AMP? RDNOISE"].values())

        # start iteration
        for i in range(iter):
            # quick and dirty CRR on current iteration
            noise = out.medianImg((err_box_y, err_box_x))
            for iquad in range(len(quads)):
                quad = noise.getSection(quads[iquad])
                select_noise = quad.getData() <= 0
                quad.setData(data=0, select=select_noise)
                quad = (quad + rdnoises[iquad] ** 2).sqrt()
                noise = noise.setSection(
                    quads[iquad], subimg=quad, update_header=False, inplace=False
                )
            noise.setData(
                data=noise._data[noise._data > 0].min(), select=noise._data <= 0
            )
            if cpus > 1:
                result = []
                fine = out.convolveGaussImg(sigma_x, sigma_y)
                fine_norm = out / fine
                select_neg = fine_norm < 0
                fine_norm.setData(data=0, select=select_neg)

                pool = Pool(cpus)
                result.append(pool.apply_async(out.subsampleImg))
                result.append(pool.apply_async(fine_norm.subsampleImg))
                pool.close()
                pool.join()
                sub = result[0].get()
                sub_norm = result[1].get()
                pool.terminate()

                pool = Pool(cpus)
                result[0] = pool.apply_async(sub.convolveImg, args=([LA_kernel]))
                result[1] = pool.apply_async(sub_norm.convolveImg, args=([LA_kernel]))
                pool.close()
                pool.join()
                conv = result[0].get()
                select_neg = conv < 0
                conv.setData(
                    data=0, select=select_neg
                )  # replace all negative values with 0
                Lap2 = result[1].get()
                pool.terminate()

                pool = Pool(cpus)
                result[0] = pool.apply_async(conv.rebin, args=(2, 2))
                result[1] = pool.apply_async(Lap2.rebin, args=(2, 2))
                pool.close()
                pool.join()
                Lap = result[0].get()
                Lap2 = result[1].get()
                pool.terminate()

                S = Lap / (noise * 4)  # normalize Laplacian image by the noise
                S_prime = S - S.medianImg(
                    (err_box_y, err_box_x)
                )  # cleaning of the normalized Laplacian image
            else:
                sub = out.subsampleImg()  # subsample image
                conv = sub.convolveImg(
                    LA_kernel
                )  # convolve subsampled image with kernel
                select_neg = conv < 0
                conv.setData(
                    data=0, select=select_neg
                )  # replace all negative values with 0
                Lap = conv.rebin(2, 2)  # rebin the data to original resolution
                S = Lap / (noise * 4)  # normalize Laplacian image by the noise
                S_prime = S - S.medianImg(
                    (err_box_y, err_box_x)
                )  # cleaning of the normalized Laplacian image
                fine = out.convolveGaussImg(
                    sigma_x, sigma_y
                )  # convolve image with a 2D Gaussian
                fine_norm = out / fine
                select_neg = fine_norm < 0
                fine_norm.setData(data=0, select=select_neg)
                sub_norm = fine_norm.subsampleImg()  # subsample image
                Lap2 = (sub_norm).convolveImg(LA_kernel)
                Lap2 = Lap2.rebin(2, 2)  # rebin the data to original resolution

            # define cosmic ray selection
            select = numpy.logical_or(
                numpy.logical_and((Lap2) > flim, S_prime > sigma_det), select
            )
            # update mask in clean image for next iteration
            out.setData(mask=True, select=select)
            out = out.replaceMaskMedian(box_x, box_y, replace_error=None)

        return select


def loadImage(
    infile,
    extension_data=None,
    extension_mask=None,
    extension_error=None,
    extension_header=0,
):
    image = Image()
    image.loadFitsData(
        infile,
        extension_data=extension_data,
        extension_mask=extension_mask,
        extension_error=extension_error,
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
    )

    return out_image


def combineImages(images, method="median", k=3):
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
    # creates an empty empty array to store the images in a stack
    dim = images[0].getDim()
    stack_image = numpy.zeros((len(images), dim[0], dim[1]), dtype=numpy.float32)
    stack_mask = numpy.zeros((len(images), dim[0], dim[1]), dtype=bool)

    # load image data in to stack
    for i in range(len(images)):
        stack_image[i, :, :] = images[i].getData()
        if images[i]._mask is not None:
            stack_mask[i, :, :] = images[i].getMask()

    # combine the images according to the selected method
    if method == "median":
        new_image = numpy.median(stack_image, 0)
    elif method == "sum":
        new_image = numpy.sum(stack_image, 0)
    elif method == "mean":
        new_image = numpy.mean(stack_image, 0)
    elif method == "nansum":
        new_image = numpy.nansum(stack_image, 0)
    elif method == "clipped_mean":
        median = numpy.median(stack_image, 0)
        rms = numpy.std(stack_image, 0)
        # select pixels within given sigma limits around the median
        select = numpy.logical_and(
            stack_image < median + k * rms, stack_image > median - k * rms
        )
        # compute the number of good pixels
        good_pixels = numpy.sum(select, 0).astype(bool)
        # set all bad pixel to 0 to compute the mean
        stack_image[:, numpy.logical_not(good_pixels)] = 0
        new_image = numpy.sum(stack_image, 0) / good_pixels

    # mask bad pixels
    old_mask = numpy.sum(stack_mask, 0).astype(bool)
    new_mask = numpy.logical_or(old_mask, numpy.isnan(new_image))
    # replace masked pixels
    # new_image[new_mask] = 0

    # TODO: add new header keywords:
    #   - NCOMBINE: number of frames combined
    #   - STATCOMB: statistic used to combine
    if images[0]._header is not None:
        new_header = images[0]._header
    else:
        new_header = None

    outImage = Image(data=new_image, mask=new_mask, header=new_header)

    return outImage
