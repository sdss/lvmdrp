from os import path

import numpy
from scipy import interpolate

from lvmdrp.core.spectrum1d import Spectrum1D


resources_dir = path.join(path.dirname(__file__), "../resources")


class PassBand(Spectrum1D):
    def __init__(self, data=None, wave=None):
        Spectrum1D.__init__(self, wave, data, error=None, mask=None, lsf=None)

    def loadTxtFile(self, file, wave_col=2, trans_col=3, delimiter="", header_col=0):
        filter_dat = open(file, "r")
        lines = filter_dat.readlines()
        wave = numpy.zeros(len(lines) - header_col, dtype=numpy.float32)
        data = numpy.zeros(len(lines) - header_col, dtype=numpy.float32)
        m = 0
        for i in range(header_col, len(lines)):
            if delimiter == "":
                line = lines[i].split()
            else:
                line = lines[i].split(delimiter)
            wave[m] = float(line[wave_col])
            data[m] = float(line[trans_col])
            m += 1
        self.__init__(data, wave)

    def effectiveWave(self):
        a = numpy.trapz(self._data * self._wave)
        b = numpy.trapz(self._data / self._wave)
        eff_wave = numpy.sqrt(a / b)

        return eff_wave

    def getFluxPass(self, spec, resamp="linear"):
        if resamp == "spline":
            intp = interpolate.UnivariateSpline(self._wave, self._data, s=0)
            new_trans = intp(spec._wave)
        elif resamp == "linear":
            intp = interpolate.UnivariateSpline(self._wave, self._data, k=1, s=0)
            new_trans = intp(spec._wave)

        lowCut = numpy.greater(spec._wave, numpy.min(self._wave))
        highCut = numpy.less(spec._wave, numpy.max(self._wave))
        total_cut = numpy.logical_and(lowCut, highCut)

        if spec._mask is not None:
            goodpix = numpy.logical_and(numpy.logical_not(spec._mask), total_cut)
            flux = numpy.trapz(
                spec._data[goodpix] * new_trans[goodpix] * spec._wave[goodpix],
                spec._wave[goodpix],
            ) / numpy.trapz(
                new_trans[goodpix] * spec._wave[goodpix], spec._wave[goodpix]
            )
        else:
            flux = numpy.trapz(
                spec._data[total_cut] * new_trans[total_cut] * spec._wave[total_cut],
                spec._wave[total_cut],
            ) / numpy.trapz(
                new_trans[total_cut] * spec._wave[total_cut], spec._wave[total_cut]
            )
            # print spec._wave[total_cut], spec._data[total_cut]
        if spec._error is not None:
            if spec._mask is not None:
                error = numpy.sqrt(
                    numpy.trapz(
                        spec._error[goodpix] ** 2
                        * new_trans[goodpix]
                        * spec._wave[goodpix],
                        spec._wave[goodpix],
                    )
                    / numpy.trapz(
                        new_trans[goodpix] * spec._wave[goodpix], spec._wave[goodpix]
                    )
                )
            else:
                error = numpy.sqrt(
                    numpy.trapz(
                        spec._error[total_cut] ** 2
                        * new_trans[total_cut]
                        * spec._wave[total_cut],
                        spec._wave[total_cut],
                    )
                    / numpy.trapz(
                        new_trans[total_cut] * spec._wave[total_cut],
                        spec._wave[total_cut],
                    )
                )
        else:
            error = None
        # print spec._data[total_cut],spec._error[total_cut]
        return flux, error

    def getFluxRSS(self, rss, resamp="linear"):
        flux = numpy.zeros(rss._fibers, dtype=numpy.float32)
        if rss._error is not None:
            error = numpy.zeros(rss._fibers, dtype=numpy.float32)
        else:
            error = None

        for i in range(rss._fibers):
            spec = rss[i]
            photo = self.getFluxPass(spec, resamp)
            flux[i] = photo[0]
            if rss._error is not None:
                error[i] = photo[1]
        min = numpy.min(flux)
        max = numpy.max(flux)
        std = numpy.std(flux)
        return flux, error, min, max, std

    def getFluxCube(self, cube, resamp="linear", use_mask=True):
        dim = cube._data.shape[1:]
        flux = numpy.zeros(dim, dtype=numpy.float32)
        if cube._error is not None:
            error = numpy.zeros(dim, dtype=numpy.float32)
        else:
            error = None
        mask = None
        if use_mask and cube._mask is None:
            use_mask = False
        for i in range(dim[0]):
            for j in range(dim[1]):
                if use_mask:
                    mask = cube._mask[:, i, j]
                if cube._error is not None:
                    spec = Spectrum1D(
                        wave=cube._wave,
                        data=cube._data[:, i, j],
                        error=cube._error[:, i, j],
                        mask=mask,
                    )
                else:
                    spec = Spectrum1D(
                        wave=cube._wave, data=cube._data[:, i, j], mask=mask
                    )
                photo = self.getFluxPass(spec, resamp)
                # print photo[0],photo[1]
                flux[i, j] = photo[0]
                if cube._error is not None:
                    error[i, j] = photo[1]
        return flux, error

    def fluxToMag(self, flux, error=None, system="AB", units=1.0e-16, wave=None):
        if system == "AB":
            aLambda = 3e-13  # for conversion to erg s-1 cm-2 angstrom-1 with lambda in microns
            effLMicron = self.effectiveWave() * (1e-10 / 1e-6)

            fluxJy = (flux * units * effLMicron**2) / aLambda
            mag = -2.5 * numpy.log10(fluxJy / 1e23) - 48.6
        if system == "Vega":
            vega_spec = Spectrum1D()
            vega_spec.loadTxtData(resources_dir + "/vega.txt")
            (vega_flux, vega_err) = self.getFluxPass(vega_spec)
            mag = -2.5 * numpy.log10(flux * units / vega_flux)
        # if error is not None:
        #    error = 1.0857*error/flux
        #    return mag, error
        # else:
        return mag

    def magToFlux(self, mag, error=None, system="AB", units=1.0e-16, wave=None):
        if system == "AB":
            aLambda = 3e-13  # for conversion to erg s-1 cm-2 angstrom-1 with lambda in microns
            effLMicron = self.effectiveWave() * (1e-10 / 1e-6)
            fluxJy = 10 ** ((mag + 48.6) / -2.5) * 1e23
            flux = (fluxJy * aLambda) / units / effLMicron**2
        if system == "Vega":
            vega_spec = Spectrum1D()
            vega_spec.loadTxtData(resources_dir + "/vega.txt")
            (vega_flux, vega_err) = self.getFluxPass(vega_spec)
            flux = (10 ** (mag / -2.5)) * (vega_flux / units)
        # if error is not None:
        #    error = 1.0857*error/flux
        #    return mag, error
        # else:
        return flux
