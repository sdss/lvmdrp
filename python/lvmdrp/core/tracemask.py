#!/usr/bin/env python
# encoding: utf-8

import os
import numpy
import bottleneck as bn
from copy import deepcopy as copy
from astropy.io import fits as pyfits
from astropy.table import Table
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core import plot


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
        coeffs, smoothing_kind = None, None
        samples = None
        samples_error = None
        slitmap = None
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
                    samples = Table.read(hdu)
                if hdu.name == "SAMPLES_ERROR":
                    samples_error = Table.read(hdu)
                if hdu.name == "COEFFS":
                    coeffs = hdu.data.astype("float32")
                    smoothing_kind = header.get("SMOOKIND")
                if hdu.name == "SLITMAP":
                    slitmap = hdu

            trace = cls(
                data=data,
                header=header,
                error=error,
                mask=mask,
                slitmap=slitmap,
                samples=samples,
                samples_error=samples_error,
                coeffs=coeffs,
                smoothing_kind=smoothing_kind
            )

        return trace

    def __init__(
        self,
        data=None,
        header=None,
        error=None,
        mask=None,
        slitmap=None,
        samples=None,
        samples_error=None,
        samples_columns=None,
        shape=None,
        size=None,
        arc_position_x=None,
        arc_position_y=None,
        good_fibers=None,
        fiber_type=None,
        coeffs=None,
        smoothing_kind=None
    ):
        FiberRows.__init__(
            self,
            data=data,
            header=header,
            error=error,
            mask=mask,
            slitmap=slitmap,
            samples=samples,
            samples_error=samples_error,
            samples_columns=samples_columns,
            shape=shape,
            size=size,
            arc_position_x=arc_position_x,
            arc_position_y=arc_position_y,
            good_fibers=good_fibers,
            fiber_type=fiber_type,
            coeffs=coeffs,
            smoothing_kind=smoothing_kind
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

        hdus.append(pyfits.PrimaryHDU(self._data.astype("float32"), header=self.getHeader()))
        if self._error is not None:
            hdus.append(pyfits.ImageHDU(self._error.astype("float32"), name="ERROR"))
        if self._mask is not None:
            hdus.append(pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX"))
        if self._samples is not None:
            hdus.append(pyfits.BinTableHDU(self._samples, name="SAMPLES"))
        if self._samples_error is not None:
            hdus.append(pyfits.BinTableHDU(self._samples_error, name="SAMPLES_ERROR"))
        if self._coeffs is not None:
            hdus.append(pyfits.ImageHDU(self._coeffs.astype("float32"), name="COEFFS"))
            hdus[0].header["SMOOKIND"] = (self._smoothing_kind, "smoothing kind")
            hdus[0].update_header()
        if self._slitmap is not None:
            hdus.append(pyfits.BinTableHDU(self._slitmap, name="SLITMAP"))

        os.makedirs(os.path.dirname(out_trace), exist_ok=True)
        hdus.writeto(out_trace, output_verify="silentfix", overwrite=True)
