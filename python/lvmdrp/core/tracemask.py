#!/usr/bin/env python
# encoding: utf-8

import numpy
from lvmdrp.core.fiberrows import FiberRows


class TraceMask(FiberRows):
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
    ):
        FiberRows.__init__(
            self,
            data,
            header,
            error,
            mask,
            shape,
            size,
            arc_position_x,
            arc_position_y,
            good_fibers,
            fiber_type,
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
