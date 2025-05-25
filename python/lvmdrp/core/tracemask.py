#!/usr/bin/env python
# encoding: utf-8

import os
import numpy
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
        coeffs, poly_kind, poly_deg = None, None, None
        samples = None
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
                    samples = Table(hdu.data)
                if hdu.name == "COEFFS":
                    coeffs = hdu.data.astype("float32")
                    poly_kind = header.get("POLYKIND")
                    poly_deg = header.get("POLYDEG")
                if hdu.name == "SLITMAP":
                    slitmap = hdu

            trace = cls(data=data, error=error, mask=mask, slitmap=slitmap, samples=samples, coeffs=coeffs, poly_kind=poly_kind, poly_deg=poly_deg, header=header)

        return trace

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
        FiberRows.__init__(
            self,
            data,
            header,
            error,
            mask,
            slitmap,
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

        new_trace = copy(self)
        new_trace._data = self._data[block_selection]
        new_trace._error = self._error[block_selection] if self._error is not None else None
        new_trace._mask = self._mask[block_selection] if self._mask is not None else None
        new_trace._samples = self._samples[block_selection] if self._samples is not None else None
        new_trace.set_coeffs(self._coeffs[block_selection] if self._coeffs is not None else None, poly_kind=self._poly_kind)
        new_trace.setFibers(block_selection.sum())

        return new_trace

    def set_block(self, data=None, iblock=None, blockid=None, error=None, mask=None, samples=None, coeffs=None, poly_kind=None):
        slitmap = self._filter_slitmap()
        blockid = self._validate_blockid(iblock, blockid, slitmap=slitmap)
        block_selection = slitmap["blockid"] == blockid
        nfibers = block_selection.sum()

        if data is not None:
            if data.shape[0] != nfibers:
                raise ValueError(f"Incompatible data shapes. Trying to set a block of {data.shape[0]} fibers to a selection of {nfibers}")
            self._data[block_selection] = data
        if error is not None and self._error is not None:
            self._error[block_selection] = error
        if mask is not None and self._error is not None:
            self._mask[block_selection] = mask
        if samples is not None and self._samples is not None:
            for i, column in enumerate(self._samples.colnames):
                self._samples[column][block_selection] = samples[:, i]
        if coeffs is not None and poly_kind is not None and self._coeffs is not None:
            if self._poly_kind != poly_kind:
                raise ValueError(f"Incompatible polynomial kinds. Trying to set {poly_kind} to a tracemask of {self._poly_kind}")
            poly_deg = coeffs.shape[1] - 1
            if self._poly_deg != poly_deg:
                raise ValueError(f"Incompatible polynomial degree. Trying to set {poly_deg} to a tracemask of {self._poly_deg}")
            self.set_coeffs(coeffs, poly_kind=poly_kind)

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

        hdus.append(pyfits.PrimaryHDU(self._data.astype("float32"), header=self.getHeader()))
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
            hdus[0].update_header()
        if self._slitmap is not None:
            hdus.append(pyfits.BinTableHDU(self._slitmap, name="SLITMAP"))

        os.makedirs(os.path.dirname(out_trace), exist_ok=True)
        hdus.writeto(out_trace, output_verify="silentfix", overwrite=True)

    def plot_block(self, iblock=None, blockid=None, ref_column=None, axs=None):
        block = self.get_block(iblock=iblock, blockid=blockid)

        pixels = numpy.arange(block._data.shape[1], dtype="int")
        samples = block.get_samples(as_pandas=True)

        if axs is None:
            _, ax = plot.create_subplots(to_display=True, figsize=(15,5), layout="constrained")
            ax.tick_params(labelbottom=False)

            ax_divider = plot.make_axes_locatable(ax)
            ax_res = ax_divider.append_axes("bottom", size="30%", pad="5%")
            ax_res.sharex(ax)

            axs = {"mod": ax, "res": ax_res}

        if "mod" in axs:
            if ref_column is not None:
                axs["mod"].axvline(ref_column, ls=":", lw=1, color="0.7")
            axs["mod"].plot(pixels, block._data.T, "-", lw=1, label="model")
            if samples is not None:
                axs["mod"].plot(samples.columns, samples.T, ".", ms=5, mew=0, mfc="0.2", label="data")
                axs["mod"].plot(samples.columns, block._data[:, samples.columns].T, "s", ms=5, mew=1, mec="0.2", mfc="none", label="model@data")
        if "res" in axs:
            axs["res"].axhline(ls="--", lw=1, color="0.4")
            axs["res"].axhline(-0.01, ls=":", lw=1, color="0.4")
            axs["res"].axhline(+0.01, ls=":", lw=1, color="0.4")
            axs["res"].plot(samples.columns, ((block._data[:, samples.columns] - samples)/samples).T, ".-", lw=0.2, ms=5, mew=0)

        return axs

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
