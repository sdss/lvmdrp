#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy
from numpy import polynomial

from lvmdrp.core.fiberrows import FiberRows
from lvmdrp import log


class TraceMask(FiberRows):
    def __init__(
        self,
        data=None,
        header=None,
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
            None,
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

    def smoothTracePoly(self, deg, poly_kind="poly", clip=None):
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
        log.info(f'Fitting "{poly_kind}" class of polynomial.')
        # iterate over each fiber
        for i in range(self._fibers):
            good_pix = numpy.logical_not(self._mask[i, :])
            if numpy.sum(good_pix) != 0:
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
                except numpy.linalg.LinAlgError as e:
                    log.error(f'Fiber trace failure at fiber {i}: {e}')

                self._coeffs[i, :] = poly.coef
                self._data[i, :] = poly(pixels)

                if clip is not None:
                    self._data = numpy.clip(self._data, clip[0], clip[1])
                self._mask[i, :] = False
            else:
                self._mask[i, :] = True

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

    def getPixelCoor(self):
        x_cor = numpy.zeros((self._nfibers, self._data.shape[1]), dtype="int16")
        x_cor = numpy.arange(self._data.shape[1])[numpy.newaxis, :]
        y_cor = self._data
        return (x_cor, y_cor)

    def clipTrace(self, dim):
        self._data = numpy.clip(self._data, 0, dim)
        self._data = numpy.clip(self._data, 0, dim)
