#!/usr/bin/env python
# encoding: utf-8

import os
import numpy
from astropy.io import fits as pyfits
from astropy.table import Table
from lvmdrp.core.fiberrows import FiberRows


class TraceMask(FiberRows):

    @classmethod
    def from_file(cls, in_tracemask):
        """Returns an TraceMask instance given a FITS file

        Parameters
        ----------
        in_rss : str
            Name or Path of the FITS image from which the data shall be loaded

        Returns
        -------
        TraceMask
            TraceMask instance
        """
        header = None
        data, error, mask = None, None, None
        coeffs, poly_kind, poly_deg = None, None, None
        with pyfits.open(in_tracemask, uint=True, do_not_scale_image_data=True, memmap=False) as hdus:
            header = hdus["PRIMARY"].header
            for hdu in hdus:
                if hdu.name == "PRIMARY":
                    data = hdu.data.astype("float32")
                if hdu.name == "ERROR":
                    error = hdu.data.astype("float32")
                if hdu.name == "BADPIX":
                    mask = hdu.data.astype("bool")
                if hdu.name == "SAMPLES":
                    samples = Table(hdu.data)
                if hdu.name == "COEFFS":
                    coeffs = hdu.data.astype("float32")
                    poly_kind = header.get("POLYKIND")
                    poly_deg = header.get("POLYDEG")

            trace = cls(data=data, error=error, mask=mask, samples=samples, coeffs=coeffs, poly_kind=poly_kind, poly_deg=poly_deg, header=header)

        return trace

    def __init__(
        self,
        data=None,
        header=None,
        error=None,
        mask=None,
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
        FiberRows.__init__(
            self,
            data,
            header,
            error,
            mask,
            samples,
            shape,
            size,
            arc_position_x,
            arc_position_y,
            good_fibers,
            fiber_type,
            coeffs,
            poly_kind,
            poly_deg
        )

    def getRound(self):
        """
        Insert data to a slice of the trace mask

        Returns: (round)
        --------------
        round : numpy.ndarray (int)
            Trace position rounded to the nearest integer
        """
        round = numpy.round(self._data).astype("int16")  # round the traces to integer
        return round

    def getFiberDist(self, slice):
        cut = self._data[:, slice]
        dist = cut[1:] - cut[:-1]
        if self._mask is not None:
            slice_mask = self._mask[:, slice]
            dist_mask = numpy.logical_and(slice_mask[1:], slice_mask[:-1])
            return dist, dist_mask
        else:
            return dist, None

    def getPixelCoor(self):
        x_cor = numpy.zeros((self._nfibers, self._data.shape[1]), dtype="int16")
        x_cor = numpy.arange(self._data.shape[1])[numpy.newaxis, :]
        y_cor = self._data
        return (x_cor, y_cor)

    def clipTrace(self, dim):
        self._data = numpy.clip(self._data, 0, dim)
        self._data = numpy.clip(self._data, 0, dim)

    def writeFitsData(self, out_trace):
        """Writes information from a FiberRows object into a FITS file.

        A single or multiple extension file are possible to create.

        Parameters
        --------------
        out_trace : string
            Name or Path of the FITS image from which the data shall be loaded
        """
        hdus = pyfits.HDUList()

        hdus.append(pyfits.PrimaryHDU(self._data.astype("float32")))
        if self._error is not None:
            hdus.append(pyfits.ImageHDU(self._error.astype("float32"), name="ERROR"))
        if self._mask is not None:
            hdus.append(pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX"))
        if self._samples is not None:
            hdus.append(pyfits.BinTableHDU(self._samples, name="SAMPLES"))
        if self._coeffs is not None:
            hdus.append(pyfits.ImageHDU(self._coeffs.astype("float32"), name="COEFFS"))
            hdus[0].header["POLYKIND"] = (self._poly_kind, "polynomial kind")
            hdus[0].header["POLYDEG"] = (self._poly_deg, "polynomial degree")

        if len(hdus) > 0:
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
            if self._header is not None:
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                hdu[0].update_header()

        os.makedirs(os.path.dirname(out_trace), exist_ok=True)
        hdu.writeto(out_trace, output_verify="silentfix", overwrite=True)