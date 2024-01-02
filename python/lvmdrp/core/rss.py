import os
import numpy
import bottleneck as bn
from scipy import interpolate
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u

from lvmdrp import log
from lvmdrp.core.constants import CONFIG_PATH
from lvmdrp.core.apertures import Aperture
from lvmdrp.core.cube import Cube
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.header import Header, combineHdr
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.spectrum1d import Spectrum1D, wave_little_interpol
from lvmdrp.core import dataproducts as dp


def _read_pixwav_map(lamp: str, camera: str, pixels=None, waves=None):
    """read pixel-wavelength map from a lamp and camera

    Parameters
    ----------
    lamp : str
        arc lamp name
    camera : str
        one of cameras (e.g., b1, r1, z1)

    Returns
    -------
    ref_fiber : int
        reference fiber used to build the pixel-wavelength map
    ref_lines : numpy.ndarray
        reference wavelength of the emission lines
    pixel : numpy.ndarray
        pixel position of the emission lines
    use_line : numpy.ndarray
        mask to select which lines to use
    """
    pixwav_map_path = os.path.join(CONFIG_PATH, "wavelength", f"lvm-pixwav-{lamp}_{camera}.txt")

    if os.path.isfile(pixwav_map_path):
        # load initial pixel positions and reference wavelength from txt config file
        log.info(f"pixel-to-wavelength map in file '{pixwav_map_path}'")
        with open(pixwav_map_path, "r") as file_in:
            ref_fiber = int(file_in.readline()[:-1])
            log.info(f"going to use fiber {ref_fiber} as reference")
            pixels, waves, use_line = numpy.loadtxt(
                file_in, dtype=float, unpack=True
            )
        use_line = use_line.astype(bool)
        log.info(
            f"number of lines in file {pixels.size} percentage masked {(~use_line).sum() / pixels.size * 100: g} %"
        )

        nlines = use_line.sum()
        log.info(f"{nlines} good lines found")
    elif pixels is not None and waves is not None:
        # get the reference spectrum number and the guess pixel map
        ref_fiber = int(ref_fiber)
        log.info(f"going to use fiber {ref_fiber} as reference")
        pixels = numpy.asarray(list(map(float, pixels.split(","))))
        waves = numpy.asarray(list(map(float, waves.split(","))))
        use_line = numpy.ones(len(waves), dtype=bool)
        nlines = len(pixels)
        log.info(
            f"{nlines} good lines found ({(~use_line).sum()} lines masked)"
        )
    else:
        log.warning(f"no pixel-to-wavelength map found for {lamp = } in {camera = }")
        # initialize new table to create a new pixel-to-wave map
        ref_fiber = None
        pixels = numpy.empty((0,))
        waves = numpy.empty((0,))
        use_line = numpy.empty((0,), dtype=bool)

    return pixwav_map_path, ref_fiber, pixels, waves, use_line


class RSS(FiberRows):

    @classmethod
    def from_file(cls, in_rss):
        """Returns an RSS instance given a FITS file

        Parameters
        ----------
        in_rss : str
            Name or Path of the FITS image from which the data shall be loaded

        Returns
        -------
        RSS
            RSS instance
        """
        header = None
        data, error, mask = None, None, None
        wave_trace, lsf_trace = None, None
        wave, lsf = None, None
        cent_trace, width_trace = None, None
        sky, sky_error = None, None
        fluxcal = None
        slitmap = None
        with pyfits.open(in_rss, uint=True, do_not_scale_image_data=True, memmap=False) as hdus:
            header = hdus["PRIMARY"].header
            for hdu in hdus:
                if hdu.name == "PRIMARY":
                    data = hdu.data.astype("float32")
                if hdu.name == "ERROR":
                    error = hdu.data.astype("float32")
                if hdu.name == "BADPIX":
                    mask = hdu.data.astype("bool")
                if hdu.name == "WAVE_TRACE":
                    wave_trace = hdu
                elif hdu.name == "WAVE":
                    wave = hdu.data.astype("float32")
                if hdu.name == "LSF_TRACE":
                    lsf_trace = hdu
                elif hdu.name == "LSF" or hdu.name == "INSTFWHM":
                    lsf = hdu.data.astype("float32")
                if hdu.name == "CENT_TRACE":
                    cent_trace = hdu
                if hdu.name == "WIDTH_TRACE":
                    width_trace = hdu
                if hdu.name == "SKY":
                    sky = hdu.data.astype("float32")
                if hdu.name == "SKY_ERROR":
                    sky_error = hdu.data.astype("float32")
                if hdu.name == "FLUXCAL":
                    fluxcal = hdu.data
                if hdu.name == "SLITMAP":
                    slitmap = hdu.data
            
            rss = cls(
                data=data,
                error=error,
                mask=mask,
                wave_trace=wave_trace,
                lsf_trace=lsf_trace,
                wave=wave,
                lsf=lsf,
                cent_trace=cent_trace,
                width_trace=width_trace,
                sky=sky,
                sky_error=sky_error,
                header=header,
                slitmap=slitmap,
                fluxcal=fluxcal
            )
            
        return rss

    @classmethod
    def from_spectrographs(cls, rss_sp1, rss_sp2, rss_sp3):
        """Stacks together RSS objects from the three spectrographs

        Parameters
        ----------
        rss_sp1 : RSS
            RSS object for spectrograph 1
        rss_sp2 : RSS
            RSS object for spectrograph 2
        rss_sp3 : RSS
            RSS object for spectrograph 3

        Returns
        -------
        RSS
            RSS object with data from all three spectrographs
        """
        # load and stack each extension
        hdrs = []
        rsss = [rss_sp1, rss_sp2, rss_sp3]
        for i in range(len(rsss)):
            rss = rsss[i]
            if i == 0:
                data_out = rss._data
                if rss._error is not None:
                    error_out = rss._error
                if rss._mask is not None:
                    mask_out = rss._mask
                if rss._wave is not None:
                    wave_out = rss._wave
                if rss._lsf is not None:
                    fwhm_out = rss._lsf
                if rss._sky is not None:
                    sky_out = rss._sky
                if rss._sky_error is not None:
                    sky_error_out = rss._sky_error
                if rss._header is not None:
                    hdrs.append(Header(rss.getHeader()))
                if rss._fluxcal is not None:
                    fluxcal_out = rss._fluxcal
            else:
                data_out = numpy.concatenate((data_out, rss._data), axis=0)
                if rss._wave is not None:
                    if len(wave_out.shape) == 2 and len(rss._wave.shape) == 2:
                        wave_out = numpy.concatenate((wave_out, rss._wave), axis=0)
                    elif len(wave_out.shape) == 1 and len(rss._wave.shape) == 1 and numpy.isclose(wave_out, rss._wave).all():
                        wave_out = wave_out
                    else:
                        raise ValueError(f"Cannot concatenate wavelength arrays of different shapes: {wave_out.shape} and {rss._wave.shape} or inhomogeneous wavelength arrays")
                else:
                    wave_out = None
                if rss._lsf is not None:
                    if len(fwhm_out.shape) == 2 and len(rss._lsf.shape) == 2:
                        fwhm_out = numpy.concatenate((fwhm_out, rss._lsf), axis=0)
                    elif len(fwhm_out.shape) == 1 and len(rss._lsf.shape) == 1 and numpy.isclose(fwhm_out, rss._lsf).all():
                        fwhm_out = fwhm_out
                    else:
                        raise ValueError(f"Cannot concatenate FWHM arrays of different shapes: {fwhm_out.shape} and {rss._lsf.shape} or inhomogeneous FWHM arrays")
                else:
                    fwhm_out = None
                if rss._error is not None:
                    error_out = numpy.concatenate((error_out, rss._error), axis=0)
                else:
                    error_out = None
                if rss._mask is not None:
                    mask_out = numpy.concatenate((mask_out, rss._mask), axis=0)
                else:
                    mask_out = None
                if rss._sky is not None:
                    sky_out = numpy.concatenate((sky_out, rss._sky), axis=0)
                else:
                    sky_out = None
                if rss._sky_error is not None:
                    sky_error_out = numpy.concatenate((sky_error_out, rss._sky_error), axis=0)
                else:
                    sky_error_out = None
                if rss._header is not None:
                    hdrs.append(Header(rss.getHeader()))
                if rss._fluxcal is not None:
                    f = fluxcal_out.to_pandas()
                    fluxcal_out = Table.from_pandas(f.combine_first(rss._fluxcal.to_pandas()))
                else:
                    fluxcal_out = None

        # update header
        if len(hdrs) > 0:
            hdr_out = combineHdr(hdrs)
        else:
            hdr_out = None
        
        # update slitmap
        slitmap_out = rss._slitmap

        return cls(
            data=data_out,
            error=error_out,
            mask=mask_out,
            wave=wave_out,
            lsf=fwhm_out,
            sky=sky_out,
            sky_error=sky_error_out,
            header=hdr_out._header,
            slitmap=slitmap_out,
            fluxcal=fluxcal_out,
        )

    @classmethod
    def from_channels(cls, rss_b, rss_r, rss_z, use_weights=True):
        """Stitch together RSS channels into a single RSS object

        Parameters
        ----------
        rss_b : RSS
            RSS object for the b channel
        rss_r : RSS
            RSS object for the r channel
        rss_z : RSS
            RSS object for the z channel
        use_weights : bool, optional
            use inverse variance weights for channel combination, by default True

        Returns
        -------
        RSS
            RSS object with data from all three channels
        """
        
        rsss = [rss_b, rss_r, rss_z]

        # get wavelengths
        log.info("merging wavelength arrays")
        waves = [rss._wave for rss in rsss]
        new_wave = numpy.unique(numpy.concatenate(waves))
        sampling = numpy.diff(new_wave)
        
        # optionally interpolate if the merged wavelengths are not monotonic
        if numpy.all(numpy.isclose(sampling, sampling[0])):
            log.info(f"current wavelength sampling: min = {sampling.min():.2f}, max = {sampling.max():.2f}")
            # extend rss._data to new_wave filling with NaNs
            fluxes, errors, masks, lsfs, skies, sky_errors = [], [], [], [], [], []
            for rss in rsss:
                rss = rss.extendData(new_wave)
                fluxes.append(rss._data)
                errors.append(rss._error)
                masks.append(rss._mask)
                lsfs.append(rss._lsf)
                skies.append(rss._sky)
                sky_errors.append(rss._sky_error)
            fluxes = numpy.asarray(fluxes)
            errors = numpy.asarray(errors)
            masks = numpy.asarray(masks)
            lsfs = numpy.asarray(lsfs)
            skies = numpy.asarray(skies)
            sky_errors = numpy.asarray(sky_errors)
        else:
            log.warning("merged wavelengths are not monotonic, interpolation needed")
            # compute the combined wavelengths
            new_wave = wave_little_interpol(waves)
            sampling = numpy.diff(new_wave)
            log.info(f"new wavelength sampling: min = {sampling.min():.2f}, max = {sampling.max():.2f}")

            # define interpolators
            log.info("interpolating RSS data in new wavelength array")
            fluxes, errors, masks, lsfs, skies, sky_errors = [], [], [], [], [], []
            for rss in rsss:
                f = interpolate.interp1d(rss._wave, rss._data, axis=1, bounds_error=False, fill_value=numpy.nan)
                fluxes.append(f(new_wave).astype("float32"))
                f = interpolate.interp1d(rss._wave, rss._error, axis=1, bounds_error=False, fill_value=numpy.nan)
                errors.append(f(new_wave).astype("float32"))
                f = interpolate.interp1d(rss._wave, rss._mask, axis=1, kind="nearest", bounds_error=False, fill_value=0)
                masks.append(f(new_wave).astype("uint8"))
                f = interpolate.interp1d(rss._wave, rss._lsf, axis=1, bounds_error=False, fill_value=numpy.nan)
                lsfs.append(f(new_wave).astype("float32"))
                f = interpolate.interp1d(rss._wave, rss._sky, axis=1, bounds_error=False, fill_value=numpy.nan)
                skies.append(f(new_wave).astype("float32"))
                f = interpolate.interp1d(rss._wave, rss._sky_error, axis=1, bounds_error=False, fill_value=numpy.nan)
                sky_errors.append(f(new_wave).astype("float32"))
            fluxes = numpy.asarray(fluxes)
            errors = numpy.asarray(errors)
            masks = numpy.asarray(masks)
            lsfs = numpy.asarray(lsfs)
            skies = numpy.asarray(skies)
            sky_errors = numpy.asarray(sky_errors)

        # define weights for channel combination
        vars = errors ** 2
        log.info("combining channel data")
        if use_weights:
            weights = 1.0 / vars
            weights = weights / bn.nansum(weights, axis=0)[None]

            new_data = bn.nansum(fluxes * weights, axis=0)
            new_lsf = bn.nansum(lsfs * weights, axis=0)
            new_error = numpy.sqrt(bn.nansum(vars, axis=0))
            new_mask = numpy.sum(masks, axis=0).astype("bool")
            new_sky = bn.nansum(skies * weights, axis=0)
            new_sky_error = numpy.sqrt(bn.nansum(sky_errors ** 2 * weights ** 2, axis=0))
        else:
            # channel-combine RSS data
            new_data = bn.nanmean(fluxes, axis=0)
            new_lsf = bn.nanmean(lsfs, axis=0)
            new_error = numpy.sqrt(bn.nanmean(vars, axis=0))
            new_mask = numpy.sum(masks, axis=0).astype("bool")
            new_sky = bn.nansum(skies, axis=0)
            new_sky_error = numpy.sqrt(bn.nanmean(sky_errors ** 2, axis=0))

        # create RSS
        new_hdr = rsss[0]._header.copy()
        for rss in rsss[1:]:
            new_hdr.update(rss._header)
        new_hdr["NAXIS1"] = new_data.shape[1]
        new_hdr["NAXIS2"] = new_data.shape[0]
        new_hdr["CCD"] = ",".join([rss._header["CCD"][0] for rss in rsss])
        wcs = WCS(new_hdr)
        wcs.spectral.wcs.cdelt[0] = new_wave[1] - new_wave[0]
        wcs.spectral.wcs.crval[0] = new_wave[0]
        new_hdr.update(wcs.to_header())
        return RSS(
            data=new_data,
            error=new_error,
            mask=new_mask,
            wave=new_wave,
            lsf=new_lsf,
            sky=new_sky,
            sky_error=new_sky_error,
            header=new_hdr
        )

    @classmethod
    def from_spectra1d(
        cls,
        spectra_list,
        header=None,
        shape=None,
        size=None,
        arc_position_x=None,
        arc_position_y=None,
        good_fibers=None,
        fiber_type=None,
        logwave=False,
    ):
        """
        Returns an RSS instance given a list of Spectrum1D instances
        """
        n_spectra = len(spectra_list)
        if n_spectra <= 0:
            raise ValueError("cannot create RSS from an empty list of spectra")
        ref_spec = spectra_list[0]

        rss = cls(
            data=numpy.zeros((n_spectra, ref_spec._data.size)),
            wave=numpy.zeros((n_spectra, ref_spec._wave.size))
            if ref_spec._wave is not None
            else None,
            lsf=numpy.zeros((n_spectra, ref_spec._lsf.size))
            if ref_spec._lsf is not None
            else None,
            error=numpy.zeros((n_spectra, ref_spec._error.size))
            if ref_spec._error is not None
            else None,
            mask=numpy.zeros((n_spectra, ref_spec._mask.size), dtype=bool)
            if ref_spec._mask is not None
            else None,
            sky=numpy.zeros((n_spectra, ref_spec._data.size))
            if ref_spec._sky is not None
            else None,
            sky_error=numpy.zeros((n_spectra, ref_spec._data.size))
            if ref_spec._sky_error is not None
            else None,
            header=header,
            shape=shape,
            size=size,
            arc_position_x=arc_position_x,
            arc_position_y=arc_position_y,
            good_fibers=good_fibers,
            fiber_type=fiber_type,
            logwave=logwave,
        )
        for i in range(n_spectra):
            rss[i] = spectra_list[i]

        # set wavelength and LSF in RSS object
        if numpy.allclose(
            numpy.repeat(rss._wave[0][None, :], rss._fibers, axis=0), rss._wave
        ):
            rss.setWave(rss._wave[0])
        else:
            rss.setWave(rss._wave)
        if numpy.allclose(
            numpy.repeat(rss._lsf[0][None, :], rss._fibers, axis=0),
            rss._lsf,
        ):
            rss.set_lsf(rss._lsf[0])
        else:
            rss.set_lsf(rss._lsf)
        return rss

    def __init__(
        self,
        data=None,
        header=None,
        error=None,
        mask=None,
        sky=None,
        sky_error=None,
        shape=None,
        size=None,
        cent_trace=None,
        width_trace=None,
        wave=None,
        lsf=None,
        wave_trace=None,
        lsf_trace=None,
        arc_position_x=None,
        arc_position_y=None,
        slitmap=None,
        fluxcal=None,
        good_fibers=None,
        fiber_type=None,
        logwave=False,
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
        self._sky = None
        self._sky_error = None
        if sky is not None:
            self._sky = sky
        if sky_error is not None:
            self._sky_error = sky_error

        # set fiber traces information if available
        self.set_cent_trace(cent_trace)
        self.set_width_trace(width_trace)
        
        # set wavelength and LSF traces information if available
        self.set_wave_trace(wave_trace)
        self.set_lsf_trace(lsf_trace)
        # evaluate wavelength and LSF traces
        self.set_wave_array(wave=wave)
        self.set_lsf_array(lsf=lsf)
        
        self.setSlitmap(slitmap)
        self.set_fluxcal(fluxcal)

    def _trace_to_coeff_table(self, trace):
        """Converts a given trace into its polynomial coefficients representation as an Astropy Table"""
        if isinstance(trace, TraceMask):
            coeffs = trace._coeffs
            columns = [
                pyfits.Column(name="FUNC", format="A10", array=numpy.asarray([trace._poly_kind] * self._fibers)),
                pyfits.Column(name="XMIN", format="I", unit="pix", array=numpy.asarray([0] * self._fibers)),
                pyfits.Column(name="XMAX", format="I", unit="pix", array=numpy.asarray([self._data.shape[1]-1] * self._fibers)),
                pyfits.Column(name="COEFF", format=f"{coeffs.shape[1]}E", dim=f"({coeffs.shape[0]},)", array=trace._coeffs)
            ]
            self._trace = Table(pyfits.BinTableHDU.from_columns(columns).data)
            return self._trace
        elif isinstance(trace, pyfits.BinTableHDU):
            self._trace = Table(trace.data)
            return self._trace
        elif trace is None:
            return None
        else:
            raise TypeError(f"trace must be lvmdrp.core.tracemask.TraceMask or None, instead got {type(trace)}")

    def _coeff_table_to_trace(self, table):
        """Converts a given Astropy Table with polynomial coefficients into a TraceMask"""
        if isinstance(table, Table):
            self._trace = TraceMask.from_coeff_table(table)
            return self._trace
        elif table is None:
            return None
        else:
            raise TypeError("table must be astropy.table.Table or None")

    def __mul__(self, other):
        """
        Operator to add two FiberRow or divide by another type if possible
        """
        if isinstance(other, RSS):
            # define behaviour if the other is of the same instance

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                data = self._data * other._data
            else:
                data = self._data

            # add error if contained in both
            if self._error is not None and other._error is not None:
                error = numpy.sqrt(
                    other._data**2 * self._error**2
                    + self._data**2 * other._error**2
                )
            elif self._error is not None:
                error = other._data * self._error
            else:
                error = self._error

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            else:
                mask = self._mask
            if data.dtype == numpy.float64:
                data = data.astype(numpy.float32)
            if error is not None and error.dtype == numpy.float64:
                error = error.astype(numpy.float32)
            rss = RSS(
                data=data,
                error=error,
                mask=mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )
            return rss

        if isinstance(other, Spectrum1D):
            # define behaviour if the other is a Spectrum1D object

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                data = self._data * other._data[numpy.newaxis, :]
            else:
                data = self._data

            # add error if contained in both
            if self._error is not None and other._error is not None:
                error = numpy.sqrt(
                    other._data[numpy.newaxis, :] ** 2 * self._error**2
                    + self._data**2 * other._error[numpy.newaxis, :] ** 2
                )
            elif self._error is not None:
                error = other._data[numpy.newaxis, :] * self._error
            else:
                error = self._error

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask[numpy.newaxis, :])
            else:
                mask = self._mask

            if data.dtype == numpy.float64:
                data = data.astype(numpy.float32)
            if error is not None and error.dtype == numpy.float64:
                error = error.astype(numpy.float32)
            rss = RSS(
                data=data,
                error=error,
                mask=mask,
                header=self._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )

            return rss

        elif isinstance(other, numpy.ndarray):
            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                # add ndarray according do its dimensions
                if self._data.shape == dim:
                    data = self._data * other
                elif len(dim) == 1:
                    if self._data.shape[0] == dim[0]:
                        data = self._data * other[:, numpy.newaxis]
                    elif self._data.shape[1] == dim[0]:
                        data = self._data * other[numpy.newaxis, :]
                else:
                    data = self._data
                if data.dtype == numpy.float64:
                    data = data.astype(numpy.float32)

                rss = RSS(
                    data=data,
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
            return rss
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data * other
                if self._error is not None:
                    error = self._error * other
                else:
                    error = self._error
                if data.dtype == numpy.float64:
                    data = data.astype(numpy.float32)
                if error is not None and error.dtype == numpy.float64:
                    error = error.astype(numpy.float32)
                rss = RSS(
                    data=data,
                    error=error,
                    mask=self._mask,
                    header=self._header,
                    shape=self._shape,
                    size=self._size,
                    arc_position_x=self._arc_position_x,
                    arc_position_y=self._arc_position_y,
                    good_fibers=self._good_fibers,
                    fiber_type=self._fiber_type,
                )
                return rss
            except (TypeError, ValueError):
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

        if self._wave is not None:
            if len(self._wave.shape) == 1:
                wave = self._wave
            else:
                wave = self._wave[fiber, :]
        else:
            wave = numpy.arange(data.size)

        if self._lsf is not None:
            if len(self._lsf.shape) == 1:
                lsf = self._lsf
            else:
                lsf = self._lsf[fiber, :]
        else:
            lsf = None

        if self._error is not None:
            error = self._error[fiber, :]
        else:
            error = None

        if self._mask is not None:
            mask = self._mask[fiber, :]
        else:
            mask = None

        spec = Spectrum1D(wave, data, error=error, mask=mask, lsf=lsf)
        return spec

    def __setitem__(self, fiber, spec):
        self._data[fiber, :] = spec._data

        if self._wave is not None and len(self._wave.shape) == 2:
            self._wave[fiber, :] = spec._wave

        if self._lsf is not None and len(self._lsf.shape) == 2:
            self._lsf[fiber, :] = spec._lsf

        if self._error is not None and spec._error is not None:
            self._error[fiber, :] = spec._error

        if self._mask is not None and spec._mask is not None:
            self._mask[fiber, :] = spec._mask

    def set_wave_array(self, wave=None):
        """Sets the wavelength array for the RSS object
        
        if wave is None, the wavelength array will be created from the trace information if available
        otherwise it will be set from the given array

        Parameters
        ----------
        wave : numpy.ndarray, optional
            Wavelength array to be set, by default None
        """
        self._wave = None
        self._wave_disp = None
        self._wave_start = None
        self._res_elements = None
        if wave is not None:
            self._wave = numpy.asarray(wave)
            if len(wave.shape) == 1:
                self._wave_disp = self._wave[1] - self._wave[0]
                self._wave_start = self._wave[0]
                self._res_elements = self._wave.shape[0]
                if self._header is not None:
                    wcs = WCS(header={
                        "CDELT1": self._wave_disp, "CRVAL1": self._wave_start,
                        "CUNIT1": "angstrom", "CTYPE1": "WAVE", "CRPIX1": 1.0})
                    self._header.update(wcs.to_header())
            elif len(wave.shape) == 2:
                self._res_elements = self._wave.shape[1]
            else:
                raise ValueError("Invalid wavelength array shape")
        elif self._wave_trace is not None:
            trace = TraceMask.from_coeff_table(self._wave_trace)
            self._wave = trace.eval_coeffs()
        else:
            self._wave = None
        
        return self._wave

    def set_lsf_array(self, lsf=None):
        if lsf is not None:
            self._lsf = lsf
        elif self._lsf_trace is not None:
            trace = TraceMask.from_coeff_table(self._lsf_trace)
            self._lsf = trace.eval_coeffs()
        else:
            self._lsf = None

        return self._lsf

    def maskFiber(self, fiber, replace_error=1e10):
        self._data[fiber, :] = 0
        if self._mask is not None:
            self._mask[fiber, :] = True
        if self._error is not None:
            self._error[fiber, :] = replace_error

    def createWavefromHdr(self):
        if self._header is not None:
            wcs = WCS(self._header)
            if wcs.spectral.array_shape:
                self._res_elements = wcs.spectral.array_shape[0]
                wl = wcs.spectral.all_pix2world(numpy.arange(self._res_elements), 0)[0]
                self._wave = (wl * u.m).to(u.angstrom).value
                self._wave_disp = self._wave[1] - self._wave[0]
                self._wave_start = self._wave[0]
            else:
                self._wave = None

    def set_sky(self, rss_sky):
        assert rss_sky._data.shape == self._data.shape
        self._sky = rss_sky._data
        if rss_sky._error is not None:
            self._sky_error = rss_sky._error

    def get_sky(self):
        header = self._header
        if header is not None:
            header["IMAGETYP"] = "sky"
            header["OBJECT"] = "sky"
        return RSS(data=self._sky, error=self._sky_error, mask=self._mask, wave=self._wave, lsf=self._lsf, header=header)

    def getSpec(self, fiber):
        data = self._data[fiber, :]
        if self._wave is not None:
            if len(self._wave.shape) == 1:
                wave = self._wave
            else:
                wave = self._wave[fiber, :]
        else:
            wave = numpy.arange(data.size)
        if self._lsf is not None:
            if len(self._lsf.shape) == 1:
                lsf = self._lsf
            else:
                lsf = self._lsf[fiber, :]
        else:
            lsf = None
        if self._error is not None:
            error = self._error[fiber, :]
        else:
            error = None

        if self._mask is not None:
            mask = self._mask[fiber, :]
        else:
            mask = None
        
        if self._sky is not None:
            sky = self._sky[fiber, :]
        else:
            sky = None
        
        if self._sky_error is not None:
            sky_error = self._sky_error[fiber, :]
        else:
            sky_error = None

        spec = Spectrum1D(wave, data, error=error, mask=mask, lsf=lsf, sky=sky, sky_error=sky_error)
        
        return spec

    def extendData(self, new_wave):
        """Extends data, error, mask, and sky to new wavelength array
        
        Given a new wavelength array `new_wave`, this function extends
        the data, error, mask, and sky arrays to the new wavelength array,
        filling in the new pixels with NaNs.

        Parameters
        ----------
        new_wave : array-like
            New wavelength array to extend to

        Returns
        -------
        self : RSS
            Returns self with extended arrays
        """
        if self._wave is None:
            raise ValueError("No wavelength array found in RSS object")
        
        if self._data is None:
            raise ValueError("No data array found in RSS object")
        
        if len(new_wave) == 0:
            raise ValueError("New wavelength array is empty")

        # find positions in new wavelength array that contain self._wave
        ipix, fpix = numpy.searchsorted(new_wave, self._wave[[0, -1]], side="left")

        # define new arrays filled with NaNs
        new_data = numpy.full((self._data.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
        new_data[:, ipix:fpix+1] = self._data
        if self._error is not None:
            new_error = numpy.full((self._error.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_error[:, ipix:fpix+1] = self._error
        else:
            new_error = None
        if self._mask is not None:
            new_mask = numpy.full((self._mask.shape[0], new_wave.size), False, dtype=bool)
            new_mask[:, ipix:fpix+1] = self._mask
        else:
            new_mask = None
        if self._sky is not None:
            new_sky = numpy.full((self._sky.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_sky[:, ipix:fpix+1] = self._sky
        else:
            new_sky = None
        if self._sky_error is not None:
            new_sky_error = numpy.full((self._sky_error.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_sky_error[:, ipix:fpix+1] = self._sky_error
        else:
            new_sky_error = None
        if self._lsf is not None:
            new_lsf = numpy.full((self._lsf.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_lsf[:, ipix:fpix+1] = self._lsf
        else:
            new_lsf = None

        # set new arrays
        self._data = new_data
        self._error = new_error
        self._mask = new_mask
        self._sky = new_sky
        self._sky_error = new_sky_error
        self._lsf = new_lsf
        self._wave = new_wave
    
        return self

    def combineRSS(self, rss_in, method="mean", replace_error=1e10):
        dim = rss_in[0]._data.shape
        data = numpy.zeros((len(rss_in), dim[0], dim[1]), dtype=numpy.float32)
        if rss_in[0]._mask is not None:
            mask = numpy.zeros((len(rss_in), dim[0], dim[1]), dtype="bool")
        else:
            mask = None
       
        if rss_in[0]._error is not None:
            error = numpy.zeros((len(rss_in), dim[0], dim[1]), dtype=numpy.float32)
        else:
            error = None
        
        if rss_in[0]._sky is not None:
            sky = numpy.zeros((len(rss_in), dim[0], dim[1]), dtype=numpy.float32)
        else:
            sky = None
        
        for i in range(len(rss_in)):
            data[i, :, :] = rss_in[i]._data
            if mask is not None:
                mask[i, :, :] = rss_in[i]._mask
            if error is not None:
                error[i, :, :] = rss_in[i]._error
            if sky is not None:
                sky[i, :, :] = rss_in[i]._sky

        combined_data = numpy.zeros(dim, dtype=numpy.float32)
        combined_error = numpy.zeros(dim, dtype=numpy.float32)
        combined_sky = numpy.zeros(dim, dtype=numpy.float32)
        
        if method == "sum":
            if mask is not None:
                data[mask] = 0
                good_pix = bn.nansum(numpy.logical_not(mask), 0)
                select_mean = good_pix > 0
                combined_data[select_mean] = bn.nansum(data, 0)[select_mean]
                combined_mask = good_pix == 0
                if error is not None:
                    error[mask] = replace_error
                    combined_error[select_mean] = numpy.sqrt(
                        bn.nansum(error**2, 0)[select_mean]
                    )
                else:
                    combined_error = None
                if sky is not None:
                    sky[mask] = 0
                    combined_sky[select_mean] = bn.nansum(sky, 0)[select_mean]
            else:
                combined_mask = None
                combined_data = bn.nansum(data, 0) / data.shape[0]
                if error is not None:
                    combined_error = numpy.sqrt(
                        bn.nansum(error**2, 0) / error.shape[0]
                    )
                else:
                    combined_error = None
                if sky is not None:
                    combined_sky = bn.nansum(sky, 0) / sky.shape[0]
                else:
                    combined_sky = None

        elif method == "mean":
            if mask is not None:
                data[mask] = 0
                good_pix = bn.nansum(numpy.logical_not(mask), 0)
                select_mean = good_pix > 0
                combined_data[select_mean] = (
                    bn.nansum(data, 0)[select_mean] / good_pix[select_mean]
                )
                combined_mask = good_pix == 0
                if error is not None:
                    error[mask] = replace_error
                    combined_error[select_mean] = numpy.sqrt(
                        bn.nansum(error**2, 0)[select_mean]
                        / good_pix[select_mean] ** 2
                    )
                else:
                    combined_error = None
                if sky is not None:
                    sky[mask] = 0
                    combined_sky[select_mean] = (
                        bn.nansum(sky, 0)[select_mean] / good_pix[select_mean]
                    )
                else:
                    combined_sky = None
            else:
                combined_mask = None
                combined_data = bn.nansum(data, 0) / data.shape[0]
                if error is not None:
                    combined_error = numpy.sqrt(
                        bn.nansum(error**2, 0) / error.shape[0]
                    )
                else:
                    combined_error = None
                if sky is not None:
                    combined_sky = bn.nansum(sky, 0) / sky.shape[0]
                else:
                    combined_sky = None

        elif method == "weighted_mean" and error is not None:
            if mask is not None:
                good_pix = bn.nansum(numpy.logical_not(mask), 0)
                select_mean = good_pix > 0
                
                var = error**2
                weights = numpy.divide(1, var, out=numpy.zeros_like(var), where=var != 0)
                weights /= bn.nansum(weights, 0)
                combined_data[good_pix] = bn.nansum(data[good_pix] * var[good_pix], 0)
                combined_error[good_pix] = numpy.sqrt(bn.nansum(var[good_pix], 0))
                combined_mask = ~good_pix
                combined_error[combined_mask] = replace_error
                if sky is not None:
                    combined_sky[good_pix] = bn.nansum(sky[good_pix] * var[good_pix], 0)
                else:
                    combined_sky = None
            else:
                var = error**2
                weights = numpy.divide(1, var, out=numpy.zeros_like(var), where=var != 0)
                weights /= bn.nansum(weights, 0)
                combined_data = bn.nansum(data * weights, 0)
                combined_error = numpy.sqrt(bn.nansum(var, 0))
                combined_mask = None
                if sky is not None:
                    combined_sky = bn.nansum(sky * weights, 0)
                else:
                    combined_sky = None

        elif method == "median":
            if mask is not None:
                good_pix = bn.nansum(numpy.logical_not(mask), 0)
                combined_data[good_pix] = bn.nanmedian(data[good_pix], 0)
                combined_mask = ~good_pix
                if error is not None:
                    combined_error[good_pix] = numpy.sqrt(bn.nanmedian(error[good_pix] ** 2, 0))
                    combined_error[combined_mask] = replace_error
                else:
                    combined_error = None
                if sky is not None:
                    combined_sky[good_pix] = bn.nanmedian(sky[good_pix], 0)
                else:
                    combined_sky = None
            else:
                combined_data = bn.nanmedian(data, 0)
                if error is not None:
                    combined_error = numpy.sqrt(bn.nanmedian(error**2, 0))
                else:
                    combined_error = None
                combined_mask = None
                if sky is not None:
                    combined_sky = bn.nanmedian(sky, 0)
                else:
                    combined_sky = None
            
        else:
            if method == "weighted_mean":
                raise ValueError(f"Method {method} is not supported when error is None")
            raise ValueError(f"Method {method} is not supported")

        self._data = combined_data
        self._wave = rss_in[0]._wave
        self._lsf = rss_in[0]._lsf
        self._header = rss_in[0]._header
        self._mask = combined_mask
        self._error = combined_error
        self._arc_position_x = rss_in[i]._arc_position_x
        self._arc_position_y = rss_in[i]._arc_position_y
        self._shape = rss_in[i]._shape
        self._size = rss_in[i]._size
        self._pixels = rss_in[i]._pixels
        self._fibers = rss_in[i]._fibers
        self._good_fibers = rss_in[i]._good_fibers
        self._fiber_type = rss_in[i]._fiber_type
        self._slitmap = rss_in[i]._slitmap
        self._sky = combined_sky

    def setSpec(self, fiber, spec):
        if spec._data is not None and self._data is not None:
            self._data[fiber, :] = spec._data

        if spec._error is not None and self._error is not None:
            self._error[fiber, :] = spec._error

        if spec._mask is not None and self._mask is not None:
            self._mask[fiber, :] = spec._mask

        if spec._sky is not None and self._sky is not None:
            self._sky[fiber, :] = spec._sky

    def createAperSpec(self, cent_x, cent_y, radius):
        if self._arc_position_x is not None and self._arc_position_y is not None:
            distance = numpy.sqrt(
                (self._arc_position_x - cent_x) ** 2
                + (self._arc_position_y - cent_y) ** 2
            )
            select_rad = distance <= radius
            #     print(select_rad, distance, radius)
            subRSS = self.subRSS(select_rad)
            combined_spec = subRSS.create1DSpec(method="sum")
        return combined_spec

    def create1DSpec(self, method="mean"):
        if self._wave is not None and len(self._wave.shape) == 2:
            if self._mask is not None:
                select = numpy.logical_not(self._mask)
            else:
                select = numpy.ones(self._data.shape, dtype="bool")
            disp = self._wave[:, 1:] - self._wave[:, :-1]
            disp = numpy.insert(disp, 0, disp[:, 0], 1)
            wave = self._wave[select].flatten()
            disp = disp[select].flatten()
            # idx = numpy.argsort(wave)
            _, idx = numpy.unique(wave, return_index=True)
            wave = wave[idx]
            
            data = self._data[select].flatten()[idx]
            
            if self._error is not None:
                error = self._error[select].flatten()[idx]
            else:
                error = None
            if self._lsf is not None:
                lsf = self._lsf[select].flatten()[idx]
            else:
                lsf = None
            if self._mask is not None:
                mask = self._mask[select].flatten()[idx]
            else:
                mask = None
            if self._sky is not None:
                sky = self._sky[select].flatten()[idx]
        else:
            if self._mask is not None:
                select = numpy.logical_not(self._mask)
            else:
                select = numpy.ones(self._data.shape, dtype="bool")
            
            data = numpy.zeros(self._data.shape[1], dtype=numpy.float32)
            
            if self._error is not None:
                error = numpy.zeros(self._data.shape[1], dtype=numpy.float32)
            else:
                error = None
            if self._sky is not None:
                sky = numpy.zeros(self._data.shape[1], dtype=numpy.float32)
            else:
                sky = None
            
            for i in range(self._data.shape[1]):
                if numpy.sum(select[:, i]) > 0:
                    if method == "mean":
                        data[i] = numpy.mean(self._data[select[:, i], i])
                        if error is not None:
                            error[i] = numpy.sqrt(
                                numpy.sum(self._error[select[:, i], i] ** 2)
                                / numpy.sum(select[:, i]) ** 2
                            )
                        if sky is not None:
                            sky[i] = numpy.mean(self._sky[select[:, i], i])
                    elif method == "sum":
                        data[i] = numpy.sum(self._data[select[:, i], i])
                        if error is not None:
                            error[i] = numpy.sqrt(
                                numpy.sum(self._error[select[:, i], i] ** 2)
                            )
                        if sky is not None:
                            sky[i] = numpy.sum(self._sky[select[:, i], i])
            
            if self._mask is not None:
                bad = numpy.sum(self._mask, 0)
                mask = bad == self._fibers
            else:
                mask = None
            
            wave = self._wave
            if self._lsf is not None and len(self._lsf.shape) == 2:
                lsf = numpy.mean(self._lsf, 0)
            else:
                lsf = self._lsf
        
        header = self._header
        
        spec = Spectrum1D(
            wave=wave,
            data=data,
            error=error,
            lsf=lsf,
            mask=mask,
            sky=sky,
            header=header
        )
        return spec

    def selectSpec(self, min=0, max=0, method="median"):
        collapsed = numpy.zeros(self._fibers, dtype=numpy.float32)
        for i in range(self._fibers):
            spec = self[i]
            
            if spec._mask is not None:
                goodpix = numpy.logical_not(spec._mask)
            else:
                goodpix = numpy.ones(spec._data.dim[0], dtype=numpy.float32)
            
            if numpy.sum(goodpix) > 0:
                if method == "median":
                    collapsed[i] = numpy.median(spec._data[goodpix])
                elif method == "sum":
                    collapsed[i] = numpy.sum(spec._data[goodpix])
                elif method == "mean":
                    collapsed[i] = numpy.mean(spec._data[goodpix])
        arg = numpy.argsort(collapsed)
        numbers = numpy.arange(self._fibers)
        select = numpy.logical_or(numbers < min, numbers > numbers[-1] - max)
        return arg[select]

    def createCubeInterpolation(
        self,
        mode="inverseDistance",
        sigma=1.0,
        radius_limit=5,
        resolution=1.0,
        min_fibers=3,
        slope=2.0,
        bad_threshold=0.1,
        replace_error=1e10,
        store_cover=False,
    ):
        if self._shape == "C":
            min_x = numpy.min(self._arc_position_x) - self._size[0]
            max_x = numpy.max(self._arc_position_x) + self._size[0]
            min_y = numpy.min(self._arc_position_y) - self._size[1]
            max_y = numpy.max(self._arc_position_y) + self._size[1]
            dim_x = int(numpy.rint(float(max_x - min_x) / resolution))
            dim_y = int(numpy.rint(float(max_y - min_y) / resolution))

            good_pix = self._data != 0
            cube = numpy.zeros((self._res_elements, dim_y, dim_x), dtype=numpy.float32)
            # print(self._error)
            # print("ERROR")
            if self._error is not None:
                error = numpy.zeros(cube.shape, dtype=numpy.float32)
                corr_cube = numpy.zeros(cube.shape, dtype=numpy.float32)
            mask = numpy.zeros(cube.shape, dtype="bool")
            mask2 = numpy.zeros(cube.shape, dtype="bool")
            weights = numpy.zeros(cube.shape, dtype=numpy.float32)

            fiber_area = numpy.pi * self._size[0] ** 2
            if self._error is not None:
                var = self._error**2
                inv_var = numpy.zeros_like(var)
                good_pix = numpy.logical_not(self._mask)
                inv_var[good_pix] = 1.0 / var[good_pix]
            else:
                inv_var = numpy.ones_like(self._data)
            if mode == "inverseDistance":
                cover = numpy.zeros(cube.shape, dtype=numpy.float32)
                weights_0 = numpy.zeros(
                    (dim_y, dim_x, self._fibers), dtype=numpy.float32
                )

                position = numpy.indices((dim_y, dim_x))
                position_y = position[0].astype(numpy.float32) + min_y
                position_x = position[1].astype(numpy.float32) + min_x

                dist = numpy.sqrt(
                    (
                        position_x[:, :, numpy.newaxis]
                        - self._arc_position_x[numpy.newaxis, numpy.newaxis, :]
                    )
                    ** 2
                    + (
                        position_y[:, :, numpy.newaxis]
                        - self._arc_position_y[numpy.newaxis, numpy.newaxis, :]
                    )
                    ** 2
                )
                select = dist <= radius_limit
                weights_0[select] = numpy.exp(-0.5 * (dist[select] / sigma) ** slope)

                for i in range(self._fibers):
                    #            print(i)
                    select = weights_0[:, :, i] > 0
                    select_bad = (
                        weights_0[:, :, i] / numpy.sum(weights_0[:, :, i].flatten())
                        > bad_threshold
                    )
                    # weight_temp = weights_0[select, i][numpy.newaxis, :]*good_pix[i, :][:, numpy.newaxis]*inv_var[i, :][:, numpy.newaxis]
                    weight_temp = (
                        weights_0[select, i][numpy.newaxis, :]
                        * good_pix[i, :][:, numpy.newaxis]
                    )
                    mask[:, select_bad] = numpy.logical_or(
                        mask[:, select_bad], self._mask[i, :][:, numpy.newaxis]
                    )
                    mask2[:, select_bad] = numpy.logical_or(
                        mask2[:, select_bad],
                        numpy.logical_and(
                            self._mask[i, :][:, numpy.newaxis],
                            self._data[i, :][:, numpy.newaxis] == 0,
                        ),
                    )
                    temp = numpy.sum(weight_temp > 0, 1)[:, numpy.newaxis]
                    if self._error is not None:
                        corr_cube[:, select] += temp * weight_temp
                    weights[:, select] += weight_temp
                    cover[:, select] += (weight_temp > 0).astype("int16")
                    cube[:, select] += self._data[i, :][:, numpy.newaxis] * weight_temp
                    if self._error is not None:
                        error[:, select] += (
                            self._error[i, :][:, numpy.newaxis]
                            * weight_temp
                            * numpy.logical_not(self._mask[i, :][:, numpy.newaxis])
                        ) ** 2
                select = weights > 0
                # cube[select] = (cube[select]/weights[select])*(resolution**2/fiber_area)
                cube[select] = (cube[select] / weights[select]) * resolution**2
                select_cover = cover <= min_fibers
                mask2 = numpy.logical_or(select_cover, mask2)
                cube[select_cover] = 0
                mask[select_cover] = True
                if self._error is not None:
                    error[select] = (
                        numpy.sqrt(error[select]) / weights[select] * (resolution**2)
                    )
                    error[mask] = replace_error
                    error[select_cover] = replace_error
                    corr_cube[select] = numpy.sqrt(corr_cube[select] / weights[select])
                    corr_cube[select_cover] = 0
                else:
                    error = None

            elif mode == "drizzle":
                for i in range(self._fibers):
                    #               print(i)
                    aperture = Aperture(
                        (self._arc_position_x[i] - min_x) / resolution,
                        (self._arc_position_y[i] - min_y) / resolution,
                        self._size[0] / resolution,
                    )
                    cover_fraction = aperture.cover_mask((dim_y, dim_x)) / fiber_area
                    select = cover_fraction > 0
                    # select_cos = cover_fraction>crit_cos
                    # fiber_mask = numpy.logical_and(select_cos[numpy.newaxis, :, :], (self._mask[i, :])[:, numpy.newaxis, numpy.newaxis])
                    # fiber_mask = (self._mask[i, :])[:, numpy.newaxis, numpy.newaxis]

                    mask[:, select] = numpy.logical_or(
                        mask[:, select], self._mask[i, :][:, numpy.newaxis]
                    )
                    mask2[:, select] = numpy.logical_or(
                        mask2[:, select],
                        numpy.logical_and(
                            self._mask[i, :][:, numpy.newaxis],
                            self._data[i, :][:, numpy.newaxis] == 0,
                        ),
                    )
                    weight_0 = (
                        cover_fraction[select][numpy.newaxis, :]
                        * good_pix[i, :].astype("int16")[:, numpy.newaxis]
                    )
                    temp = numpy.sum(weight_0 > 0, 1)[:, numpy.newaxis]

                    weights[:, select] += weight_0
                    # weights2[:,  select]+= inv_var[i,:][:,numpy.newaxis]**2
                    cube[:, select] += self._data[i, :][:, numpy.newaxis] * weight_0
                    if self._error is not None:
                        error[:, select] += var[i, :][:, numpy.newaxis] * weight_0**2
                        corr_cube[:, select] += temp
                    #   error2[:, select] +=1
                select2 = weights > 0
                cube[select2] = cube[select2] / weights[select2] * (resolution**2)
                if bad_threshold > 0.0:
                    cube[mask2] = 0
                # corr_cube[select2]=corr_cube[select2]/weights[select2]

                if self._error is not None:
                    corr_cube = numpy.sqrt(corr_cube)
                    error = numpy.sqrt(error)
                    error[select2] = (
                        error[select2] / weights[select2] * (resolution**2)
                    )

                    error[mask] = replace_error
                else:
                    error = None

                cover = None

        elif self._shape == "R":
            min_x = numpy.round(numpy.min(self._arc_position_x[:1600]), 4)
            max_x = numpy.round(numpy.max(self._arc_position_x[:1600]), 4)
            min_y = numpy.round(numpy.min(self._arc_position_y[:1600]), 4)
            max_y = numpy.round(numpy.max(self._arc_position_y[:1600]), 4)
            dim_x = int(
                numpy.round(numpy.rint(float(max_x - min_x) / resolution), 4) + 1
            )
            dim_y = int(
                numpy.round(numpy.rint(float(max_y - min_y) / resolution), 4) + 1
            )

            good_pix = self._data != 0
            print(min_x, max_x, min_y, max_y, self._res_elements, dim_y, dim_x)
            cube = numpy.zeros((self._res_elements, dim_y, dim_x), dtype=numpy.float32)
            if self._error is not None:
                error = numpy.zeros(cube.shape, dtype=numpy.float32)
                corr_cube = numpy.zeros(cube.shape, dtype=numpy.float32)
            mask = numpy.zeros(cube.shape, dtype="bool")
            mask2 = numpy.zeros(cube.shape, dtype="bool")
            weights = numpy.zeros(cube.shape, dtype=numpy.float32)
            if self._error is not None:
                var = self._error**2
                inv_var = numpy.zeros_like(var)
                good_pix = numpy.logical_not(self._mask)
                inv_var[good_pix] = 1.0 / var[good_pix]
            else:
                inv_var = numpy.ones_like(self._data)

            if mode == "drizzle":
                cover_fraction = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)
                for i in range(self._fibers):
                    indices = numpy.indices((dim_y, dim_x))
                    index_y = numpy.round(indices[0] * resolution + min_y, 4)
                    index_x = numpy.round(indices[1] * resolution + min_x, 4)
                    dist_x = numpy.round(index_x - self._arc_position_x[i], 4)
                    dist_y = numpy.round(index_y - self._arc_position_y[i], 4)
                    if resolution == self._size[0] and resolution == self._size[1]:
                        select = numpy.logical_and(
                            numpy.fabs(dist_x) + 0.001
                            < numpy.round(resolution / 2.0 + self._size[0] / 2.0, 4),
                            numpy.fabs(dist_y) + 0.001
                            < numpy.round(resolution / 2.0 + self._size[1] / 2.0, 4),
                        )
                        area = (
                            resolution**2
                            - numpy.fabs(dist_x[select]) * resolution
                            - numpy.fabs(dist_y[select])
                            * (resolution - numpy.fabs(dist_x[select]))
                        )
                        cover_fraction[select] = area
                    mask[:, select] = numpy.logical_or(
                        mask[:, select], self._mask[i, :][:, numpy.newaxis]
                    )
                    mask2[:, select] = numpy.logical_or(
                        mask2[:, select],
                        numpy.logical_and(
                            self._mask[i, :][:, numpy.newaxis],
                            self._data[i, :][:, numpy.newaxis] == 0,
                        ),
                    )
                    weight_0 = (
                        cover_fraction[select][numpy.newaxis, :]
                        * good_pix[i, :].astype("int16")[:, numpy.newaxis]
                    )

                    weights[:, select] += weight_0
                    # weights2[:,  select]+= inv_var[i,:][:,numpy.newaxis]**2
                    cube[:, select] += self._data[i, :][:, numpy.newaxis] * weight_0
                    if self._error is not None:
                        error[:, select] += var[i, :][:, numpy.newaxis] * weight_0**2
                select2 = weights > 0
                cube[select2] = cube[select2] / weights[select2] * (resolution**2)
                # cube[mask2]=0

                if self._error is not None:
                    error = numpy.sqrt(error)
                    error[select2] = (
                        error[select2] / weights[select2] * (resolution**2)
                    )

                    error[mask] = replace_error
                else:
                    error = None
                cover = None

        if self._header is not None:
            self.setHdrValue("CRVAL3", self.getHdrValue("CRVAL1"))
            self.setHdrValue("CDELT3", self.getHdrValue("CDELT1"))
            self.setHdrValue("CRPIX3", 1.0)
            self.setHdrValue("CRVAL1", 1.0)
            self.setHdrValue("CDELT1", resolution)
            self.setHdrValue("CRVAL2", 1.0)
            self.setHdrValue("CDELT2", resolution)
            self.setHdrValue("CRPIX2", 1.0)
            self.setHdrValue("CRPIX1", 1.0)
            self.setHdrValue("DISPAXIS", 3)
        if self._error is not None:
            corr_cube = corr_cube**0.5
        else:
            corr_cube = None
        if not store_cover:
            cover = None
        Cube_out = Cube(
            data=cube,
            error=error,
            mask=mask,
            error_weight=corr_cube,
            header=self._header,
            cover=cover,
        )
        return Cube_out

    def createCubeInterDAR(
        self,
        offset_x,
        offset_y,
        mode="inverseDistance",
        sigma=1.0,
        radius_limit=5,
        resolution=1.0,
        min_fibers=3,
        slope=2.0,
        bad_threshold=0.1,
        replace_error=1e10,
    ):
        min_x = numpy.min(self._arc_position_x) - self._size[0]
        max_x = numpy.max(self._arc_position_x) + self._size[0]
        min_y = numpy.min(self._arc_position_y) - self._size[0]
        max_y = numpy.max(self._arc_position_y) + self._size[0]
        dim_x = numpy.rint(float(max_x - min_x) / resolution)
        dim_y = numpy.rint(float(max_y - min_y) / resolution)
        good_pix = self._data != 0

        cube = numpy.zeros((self._res_elements, dim_y, dim_x), dtype=numpy.float32)
        if self._error is not None:
            error = numpy.zeros(cube.shape, dtype=numpy.float32)
            corr_cube = numpy.zeros(cube.shape, dtype=numpy.float32)
        mask = numpy.zeros(cube.shape, dtype="bool")
        mask2 = numpy.zeros(cube.shape, dtype="bool")
        weights = numpy.zeros(cube.shape, dtype=numpy.float32)
        fiber_area = numpy.pi * self._size[0] ** 2

        if self._error is not None:
            var = self._error**2
            inv_var = numpy.zeros_like(var)
            good_pix = numpy.logical_not(self._mask)
            inv_var[good_pix] = 1.0 / var[good_pix]
        else:
            inv_var = numpy.ones_like(self._data)

        if mode == "inverseDistance":
            cover = numpy.zeros(cube.shape, dtype=numpy.float32)
            weights_0 = numpy.zeros(cube.shape, dtype=numpy.float32)

            position = numpy.indices((dim_y, dim_x))
            position_y = position[0].astype(numpy.float32) + min_y
            position_x = position[1].astype(numpy.float32) + min_x

            for j in range(self._fibers):
                weights_0[:, :, :] = 0
                for i in range(self._res_elements):
                    dist = numpy.sqrt(
                        (position_x - (self._arc_position_x[j] + offset_x[i])) ** 2
                        + (position_y - (self._arc_position_y[j] + offset_y[i])) ** 2
                    )
                    select = dist <= radius_limit
                    weights_0[i, select] = numpy.exp(
                        -0.5 * (dist[select] / sigma) ** slope
                    )
                    select_bad = (
                        weights_0[i, :, :] / numpy.sum(weights_0[i, :, :].flatten())
                        > bad_threshold
                    )
                    mask[i, select_bad] = numpy.logical_or(
                        mask[i, select_bad],
                        numpy.logical_and(
                            self._mask[j, i], weights_0[i, select_bad] > 0
                        ),
                    )
                    mask2[i, select_bad] = numpy.logical_or(
                        mask2[i, select_bad],
                        numpy.logical_and(
                            numpy.logical_and(self._mask[j, i], self._data[j, i] == 0),
                            weights_0[i, select_bad] > 0,
                        ),
                    )
                # weight_temp = weights_0*good_pix[j, :][:, numpy.newaxis, numpy.newaxis]*inv_var[j, :][:, numpy.newaxis, numpy.newaxis]
                weight_temp = (
                    weights_0 * good_pix[j, :][:, numpy.newaxis, numpy.newaxis]
                )

                temp = numpy.sum(numpy.sum(weight_temp > 0, 1), 1)
                corr_cube += temp[:, numpy.newaxis, numpy.newaxis] * weight_temp
                weights += weight_temp
                cover += (weight_temp > 0).astype("int16")
                cube += self._data[j, :][:, numpy.newaxis, numpy.newaxis] * weight_temp
                if self._error is not None:
                    error += (
                        self._error[j, :][:, numpy.newaxis, numpy.newaxis]
                        * weight_temp
                        * numpy.logical_not(
                            self._mask[j, :][:, numpy.newaxis, numpy.newaxis]
                        )
                    ) ** 2

            select = weights > 0
            cube[select] = (cube[select] / weights[select]) * resolution**2
            select_cover = cover <= min_fibers
            mask2[select_cover] = True
            mask[select_cover] = True
            cube[mask2] = 0
            if self._error is not None:
                error[select] = (
                    numpy.sqrt(error[select]) / weights[select] * (resolution**2)
                )
                error[mask] = replace_error
                corr_cube[select] = numpy.sqrt(corr_cube[select] / weights[select])
            else:
                error = None

        elif mode == "drizzle":
            cover_fraction = numpy.zeros(cube.shape, dtype=numpy.float32)
            for j in range(self._fibers):
                #     print(j)
                cover_fraction[:, :, :] = 0
                for i in range(self._res_elements):
                    aperture = Aperture(
                        self._arc_position_x[j] + offset_x[i] - min_x,
                        self._arc_position_y[j] + offset_y[i] - min_y,
                        self._size[0],
                        grid_fixed=True,
                    )
                    cover_fraction[i, :, :] = (
                        aperture.cover_mask((dim_y, dim_x)) / fiber_area
                    )
                # select_cos = cover_fraction>crit_cos
                # fiber_mask = numpy.logical_and(select_cos, (self._mask[i, :])[:, numpy.newaxis, numpy.newaxis])
                mask = numpy.logical_or(
                    mask,
                    numpy.logical_and(
                        cover_fraction > 0,
                        self._mask[j, :][:, numpy.newaxis, numpy.newaxis],
                    ),
                )
                mask2 = numpy.logical_or(
                    mask2,
                    numpy.logical_and(
                        cover_fraction > 0,
                        numpy.logical_and(self._mask[j, :], (self._data[j, :] == 0))[
                            :, numpy.newaxis, numpy.newaxis
                        ],
                    ),
                )
                # weights_temp = cover_fraction*inv_var[j, :][:, numpy.newaxis, numpy.newaxis]*numpy.logical_not(fiber_mask)
                # weights_temp = cover_fraction*numpy.logical_not(fiber_mask)
                weights_temp = (
                    cover_fraction * good_pix[j, :][:, numpy.newaxis, numpy.newaxis]
                )
                temp = numpy.sum(numpy.sum(weights_temp > 0, 1), 1)
                corr_cube += temp[:, numpy.newaxis, numpy.newaxis] * (weights_temp > 0)
                weights += weights_temp
                cube += self._data[j, :][:, numpy.newaxis, numpy.newaxis] * weights_temp
                if self._error is not None:
                    error += (
                        self._error[j, :][:, numpy.newaxis, numpy.newaxis]
                        * weights_temp
                    ) ** 2
            select2 = weights > 0
            cube[select2] = cube[select2] / weights[select2] * (resolution**2)
            mask = numpy.logical_or(mask, numpy.logical_not(select2))
            cube[mask2] = 0
            if self._error is not None:
                error = numpy.sqrt(error)
                corr_cube = numpy.sqrt(corr_cube)
                # error[select2] = error[select2]/weights[select2]*(resolution**2/fiber_area)*1
                error[select2] = error[select2] / weights[select2] * (resolution**2)
                error[mask] = replace_error
            else:
                error = None

        if self._header is not None:
            self.setHdrValue("CRVAL3", self.getHdrValue("CRVAL1"))
            self.setHdrValue("CDELT3", self.getHdrValue("CDELT1"))
            self.setHdrValue("CRPIX3", 1.0)
            self.setHdrValue("CRVAL1", 1.0)
            self.setHdrValue("CDELT1", 1.0)
            self.setHdrValue("CRVAL2", 1.0)
            self.setHdrValue("CDELT2", 1.0)
            self.setHdrValue("CRPIX2", 1.0)
            self.setHdrValue("CRPIX1", 1.0)
            self.setHdrValue("DISPAXIS", 3)
        Cube_out = Cube(
            data=cube,
            error=error,
            mask=mask,
            header=self._header,
            error_weight=corr_cube**0.5,
        )
        return Cube_out

    def createCubeInterDAR_new(
        self,
        offset_x,
        offset_y,
        min_x,
        max_x,
        min_y,
        max_y,
        dim_x,
        dim_y,
        mode="inverseDistance",
        sigma=1.0,
        radius_limit=5,
        resolution=1.0,
        min_fibers=3,
        slope=2.0,
        bad_threshold=0.1,
        full_field=False,
        replace_error=1e10,
        store_cover=False,
    ):
        if self._shape == "C":
            good_pix = self._data != 0
            cube = numpy.zeros((self._res_elements, dim_y, dim_x), dtype=numpy.float32)
            mask = numpy.zeros(cube.shape, dtype="bool")
            mask2 = numpy.zeros(cube.shape, dtype="bool")
            weights = numpy.zeros(cube.shape, dtype=numpy.float32)
            fiber_area = numpy.pi * self._size[0] ** 2

            if self._error is not None:
                var = self._error**2
                inv_var = numpy.zeros_like(var)
                good_pix = numpy.logical_not(self._mask)
                inv_var[good_pix] = 1.0 / var[good_pix]
            else:
                inv_var = numpy.ones_like(self._data)

            if mode == "inverseDistance":
                cover = numpy.zeros(cube.shape, dtype=numpy.float32)

                error = numpy.zeros(cube.shape, dtype=numpy.float32)
                corr_cube = numpy.zeros(cube.shape, dtype=numpy.float32)

                # position = numpy.indices((dim_y, dim_x))
                # position_y = position[0].astype(numpy.float32) * resolution + min_y
                # position_x = position[1].astype(numpy.float32) * resolution + min_x
                # dist_test = numpy.sqrt(
                #     (position_x - position_x[dim_x / 2.0, dim_y / 2.0]) ** 2
                #     + (position_y - position_y[dim_x / 2.0, dim_y / 2.0]) ** 2
                # )
                # select = dist_test <= radius_limit
                # int_kernel = float(
                #     numpy.sum(numpy.exp(-0.5 * (dist_test[select] / sigma) ** slope))
                # )
                dim_x = int(dim_x)
                dim_y = int(dim_y)
                # fibers = self._fibers
                # points = self._res_elements
                # arc_position_x = self._arc_position_x.astype(numpy.float32)
                # arc_position_y = self._arc_position_y.astype(numpy.float32)
                good_pix = good_pix.astype(numpy.uint8)
                # data = self._data.astype(numpy.float32)

                # if self._mask is not None:
                #     mask_in = self._mask.astype(numpy.uint8)
                # else:
                #     mask_in = numpy.zeros_like(good_pix)
                # if self._error is not None:
                #     error_in = self._error.astype(numpy.float32)
                # else:
                #     error_in = numpy.zeros_like(self._data)
                # mask = numpy.zeros(cube.shape, dtype=numpy.uint8)
                # weights_0 = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)
                # cover_img = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)
                # temp2 = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)
                # c_code = r"""
                # int j,i,k,l;
                # float distance;
                # int coadd;
                # for(i=0; i<points; i++) {
                #     temp2(blitz::Range::all(),blitz::Range::all())=0;
                #     cover_img(blitz::Range::all(),blitz::Range::all())=0;
                #     for(j = 0; j<fibers; j++) {
                #         weights_0(blitz::Range::all(),blitz::Range::all()) = 0;
                #         coadd=0;
                #         for(k=0; k<dim_y;k++) {
                #             for(l=0; l<dim_x; l++) {
                #                 distance = sqrt(pow(position_x(k,l)- (arc_position_x(j)+offset_x(j,i)),2)+pow(position_y(k,l)- (arc_position_y(j)+offset_y(j,i)),2));
                #                 if  (distance<radius_limit) {
                #                     weights_0(k,l) = exp(-0.5*pow(distance/sigma,slope));
                #                     if (good_pix(j,i)==1) {
                #                         temp2(k,l)+=weights_0(k,l);
                #                         cube(i,k,l)+=weights_0(k,l)*data(j,i);
                #                         cover_img(k,l)+=1;
                #                         coadd++;
                #                         if (mask_in(j,i)==0) {
                #                             error(i,k,l)+=pow(weights_0(k,l)*error_in(j,i),2);
                #                         }
                #                     }
                #                     if ((mask_in(j,i)==1) && ((weights_0(k,l)/int_kernel)>bad_threshold)) {
                #                         mask(i,k,l)=1;
                #                     }
                #                 }
                #             }
                #         }

                #         for(k=0; k<dim_y;k++) {
                #             for(l=0; l<dim_x; l++) {
                #                 corr_cube(i,k,l)+=coadd*weights_0(k,l);
                #             }
                #         }
                #     }

                #     for(k=0; k<dim_y;k++) {
                #         for(l=0; l<dim_x; l++) {
                #             if (temp2(k,l)>0) {
                #                 cube(i,k,l) = cube(i,k,l)/temp2(k,l)*pow(resolution,2);
                #                 corr_cube(i,k,l) = sqrt(corr_cube(i,k,l)/temp2(k,l));
                #                 if (mask(i,k,l)==0) {
                #                     error(i,k,l) = sqrt(error(i,k,l))/temp2(k,l)*pow(resolution,2);
                #                 }
                #                 else {
                #                     error(i,k,l) = replace_error;
                #                 }
                #             }
                #            cover(i,k,l)=cover_img(k,l);
                #            if (cover_img(k,l)<(min_fibers+1)) {
                #                 cube(i,k,l) = 0;
                #                 error(i,k,l) = replace_error;
                #                 corr_cube(i,k,l)=0;
                #                 mask(i,k,l)=1;
                #            }
                #         }
                #     }
                #    // if (i==200) break;
                # }
                # """

                # distance = sqrt(pow(position_x(k,l)- (arc_position_x(j)+offset_x(i)),2)+pow(position_y(k,l)- (arc_position_y(j)+offset_y(i)),2);
                # weave.inline(
                #     c_code,
                #     [
                #         "fibers",
                #         "points",
                #         "dim_y",
                #         "dim_x",
                #         "position_x",
                #         "position_y",
                #         "arc_position_x",
                #         "arc_position_y",
                #         "offset_x",
                #         "offset_y",
                #         "radius_limit",
                #         "sigma",
                #         "slope",
                #         "min_fibers",
                #         "bad_threshold",
                #         "replace_error",
                #         "weights_0",
                #         "good_pix",
                #         "resolution",
                #         "data",
                #         "error_in",
                #         "cube",
                #         "error",
                #         "mask_in",
                #         "mask",
                #         "corr_cube",
                #         "temp2",
                #         "cover_img",
                #         "cover",
                #         "int_kernel",
                #     ],
                #     headers=["<math.h>"],
                #     type_converters=converters.blitz,
                #     compiler="gcc",
                # )

            elif mode == "drizzle":
                if self._error is not None:
                    error = numpy.zeros(cube.shape, dtype=numpy.float32)
                    corr_cube = numpy.zeros(cube.shape, dtype=numpy.float32)

                cover_fraction = numpy.zeros(cube.shape, dtype=numpy.float32)

                for j in range(self._fibers):
                    #     print(j)
                    cover_fraction[:, :, :] = 0
                    for i in range(self._res_elements):
                        aperture = Aperture(
                            (
                                self._arc_position_x[j]
                                + offset_x[j, i] * resolution
                                - min_x
                            )
                            / resolution,
                            (
                                self._arc_position_y[j]
                                + offset_y[j, i] * resolution
                                - min_y
                            )
                            / resolution,
                            self._size[0] / resolution,
                            grid_fixed=True,
                        )
                        cover_fraction[i, :, :] = (
                            aperture.cover_mask((dim_y, dim_x)) / fiber_area
                        )
                    # select_cos = cover_fraction>crit_cos
                    # fiber_mask = numpy.logical_and(select_cos, (self._mask[i, :])[:, numpy.newaxis, numpy.newaxis])
                    mask = numpy.logical_or(
                        mask,
                        numpy.logical_and(
                            cover_fraction > 0,
                            self._mask[j, :][:, numpy.newaxis, numpy.newaxis],
                        ),
                    )
                    mask2 = numpy.logical_or(
                        mask2,
                        numpy.logical_and(
                            cover_fraction > 0,
                            numpy.logical_and(
                                self._mask[j, :], (self._data[j, :] == 0)
                            )[:, numpy.newaxis, numpy.newaxis],
                        ),
                    )
                    # weights_temp = cover_fraction*inv_var[j, :][:, numpy.newaxis, numpy.newaxis]*numpy.logical_not(fiber_mask)
                    # weights_temp = cover_fraction*numpy.logical_not(fiber_mask)
                    weights_temp = (
                        cover_fraction * good_pix[j, :][:, numpy.newaxis, numpy.newaxis]
                    )
                    temp = numpy.sum(numpy.sum(weights_temp > 0, 1), 1)

                    weights += weights_temp
                    cube += (
                        self._data[j, :][:, numpy.newaxis, numpy.newaxis] * weights_temp
                    )
                    if self._error is not None:
                        error += (
                            self._error[j, :][:, numpy.newaxis, numpy.newaxis]
                            * weights_temp
                        ) ** 2
                        corr_cube += temp[:, numpy.newaxis, numpy.newaxis] * (
                            weights_temp > 0
                        )
                select2 = weights > 0
                cube[select2] = cube[select2] / weights[select2] * (resolution**2)
                mask = numpy.logical_not(
                    select2
                )  # numpy.logical_or(mask, numpy.logical_not(select2))
                # cube[mask2]=0
                if self._error is not None:
                    error = numpy.sqrt(error)
                    corr_cube = numpy.sqrt(corr_cube)
                    # error[select2] = error[select2]/weights[select2]*(resolution**2/fiber_area)*1
                    error[select2] = (
                        error[select2] / weights[select2] * (resolution**2)
                    )
                    error[mask] = replace_error
                else:
                    error = None
                cover = None

        elif self._shape == "R":
            resolution = float(resolution)
            # points = self._res_elements
            # fibers = self._fibers
            good_pix = numpy.logical_not(self._mask)
            # arc_position_x = self._arc_position_x.astype(numpy.float32)
            # arc_position_y = self._arc_position_y.astype(numpy.float32)
            # size_x = self._size[0]
            # size_y = self._size[1]
            # data = self._data.astype(numpy.float32)
            cube = numpy.zeros((self._res_elements, dim_y, dim_x), dtype=numpy.float32)
            if self._error is not None:
                error = numpy.zeros(cube.shape, dtype=numpy.float32)
                corr_cube = numpy.zeros(cube.shape, dtype=numpy.float32)
                var = self._error**2
                inv_var = numpy.zeros_like(var)
                inv_var[good_pix] = 1.0 / var[good_pix]
                # error_in = self._error.astype(numpy.float32)
            else:
                # error_in = numpy.zeros_like(self._data)
                inv_var = numpy.ones_like(self._data)

            # if self._mask is not None:
            #     mask_in = self._mask.astype(numpy.uint8)
            # else:
            #     mask_in = numpy.zeros_like(good_pix)

            mask = numpy.zeros(cube.shape, dtype=numpy.uint8)
            # weights_0 = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)
            cover = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)
            # temp2 = numpy.zeros((dim_y, dim_x), dtype=numpy.float32)

            if mode == "drizzle":
                # c_code = r"""
                #     int j,i,k,l;
                #     float dist_x;
                #     float dist_y;
                #     int coadd;
                #     for(i=0; i<points; i++) {
                #         temp2(blitz::Range::all(),blitz::Range::all())=0;
                #         cover(blitz::Range::all(),blitz::Range::all())=0;
                #         for(j = 0; j<fibers; j++) {
                #             weights_0(blitz::Range::all(),blitz::Range::all()) = 0;
                #             coadd=0;
                #             for(k=0; k<dim_y; k++) {
                #                 for(l=0; l<dim_x; l++) {
                #                     dist_x= (l*resolution)+min_x-(arc_position_x(j)+(offset_x(j,i)*resolution));
                #                     dist_y= (k*resolution)+min_y-(arc_position_y(j)+(offset_y(j,i)*resolution));
                #                     if  ((resolution==size_x) && (resolution==size_y)) {
                #                         if  ((fabs(dist_x)<resolution) && (fabs(dist_y)<resolution)) {
                #                             weights_0(k,l) = resolution*resolution-fabs(dist_x)*resolution-fabs(dist_y)*(resolution-fabs(dist_x));
                #                             if (good_pix(j,i)==1) {
                #                                 temp2(k,l)+=weights_0(k,l);
                #                                 cube(i,k,l)+=weights_0(k,l)*data(j,i);
                #                                 error(i,k,l)+=pow(weights_0(k,l)*error_in(j,i),2);
                #                                 cover(k,l)+=1;
                #                                 coadd++;
                #                                // if (mask_in(j,i)==0) {
                #                                //     error(i,k,l)+=pow(weights_0(k,l)*error_in(j,i),2);
                #                                // }
                #                             }

                #                             //if ((mask_in(j,i)==1) && ((weights_0(k,l))>0)) {
                #                             //    mask(i,k,l)=1;
                #                             //}
                #                         }
                #                     }
                #                 }
                #             }

                #             for(k=0; k<dim_y;k++) {
                #                 for(l=0; l<dim_x; l++) {
                #                     corr_cube(i,k,l)+=coadd*weights_0(k,l);
                #                 }
                #             }
                #         }
                #         for(k=0; k<dim_y;k++) {
                #             for(l=0; l<dim_x; l++) {
                #                 if (temp2(k,l)>0) {
                #                     cube(i,k,l) = cube(i,k,l)/temp2(k,l)*pow(resolution,2);
                #                     corr_cube(i,k,l) = sqrt(corr_cube(i,k,l)/temp2(k,l));
                #                     error(i,k,l) = sqrt(error(i,k,l))/temp2(k,l)*pow(resolution,2);
                #                  }
                #                  else {
                #     mask(i,k,l) = 1;
                #     error(i,k,l) = replace_error;
                #                 }
                #             }
                #         }
                #     }
                # """
                # weave.inline(
                #     c_code,
                #     [
                #         "fibers",
                #         "points",
                #         "dim_y",
                #         "dim_x",
                #         "min_x",
                #         "min_y",
                #         "size_x",
                #         "size_y",
                #         "arc_position_x",
                #         "arc_position_y",
                #         "offset_x",
                #         "offset_y",
                #         "replace_error",
                #         "weights_0",
                #         "good_pix",
                #         "resolution",
                #         "data",
                #         "error_in",
                #         "cube",
                #         "error",
                #         "mask_in",
                #         "mask",
                #         "corr_cube",
                #         "temp2",
                #         "cover",
                #     ],
                #     headers=["<math.h>"],
                #     type_converters=converters.blitz,
                #     compiler="gcc",
                # )
                cover = None

        # TODO: use WCS module from astropy
        if self._header is not None:
            self.setHdrValue("CRVAL3", self.getHdrValue("CRVAL1"))
            self.setHdrValue("CDELT3", self.getHdrValue("CDELT1"))
            self.setHdrValue("CRPIX3", 1.0)
            self.setHdrValue("CRVAL1", 1.0)
            self.setHdrValue("CDELT1", resolution)
            self.setHdrValue("CRVAL2", 1.0)
            self.setHdrValue("CDELT2", resolution)
            self.setHdrValue("CRPIX2", 1.0)
            self.setHdrValue("CRPIX1", 1.0)
            self.setHdrValue("DISPAXIS", 3)
        if not store_cover:
            cover = None
        if self._error is None:
            Cube_out = Cube(data=cube, mask=mask, header=self._header, cover=cover)
        else:
            Cube_out = Cube(
                data=cube,
                error=error,
                mask=mask,
                header=self._header,
                error_weight=corr_cube**0.5,
                cover=cover,
            )
        return Cube_out

    def subRSS(self, select):
        if self._data is not None:
            data = self._data[select]
        else:
            data = None

        if self._error is not None:
            error = self._error[select]
        else:
            error = None

        if self._mask is not None:
            mask = self._mask[select]
        else:
            mask = None

        if self._arc_position_x is not None:
            arc_position_x = self._arc_position_x[select]
        else:
            arc_position_x = None

        if self._arc_position_y is not None:
            arc_position_y = self._arc_position_y[select]
        else:
            arc_position_y = None

        if self._good_fibers is not None:
            good_fibers = self._good_fibers[select[self._good_fibers]]
        else:
            good_fibers = None

        try:
            fiber_type = self._fiber_type[select]
        except Exception:
            fiber_type = None

        if self._wave is not None:
            if len(self._wave.shape) == 2:
                wave = self._wave[select, :]
            else:
                wave = self._wave
        else:
            wave = None

        if self._lsf is not None:
            if len(self._lsf.shape) == 2:
                lsf = self._lsf[select, :]
            else:
                lsf = self._lsf
        else:
            lsf = None
        
        if self._sky is not None:
            sky = self._sky[select, :]
        else:
            sky = None

        rss = RSS(
            data=data,
            wave=wave,
            lsf=lsf,
            header=self.getHeader(),
            error=error,
            mask=mask,
            sky=sky,
            shape=self._shape,
            size=self._size,
            arc_position_x=arc_position_x,
            arc_position_y=arc_position_y,
            good_fibers=good_fibers,
            fiber_type=fiber_type,
        )
        return rss

    def splitRSS(self, parts, axis=0):
        # print(self._res_elements)
        if axis == 0:
            indices = numpy.arange(self._res_elements)
        elif axis == 1:
            indices = numpy.arange(self._fibers)
        parts = numpy.array_split(indices, parts)
        rss_parts = []
        for i in range(len(parts)):
            if axis == 0:
                data = self._data[:, parts[i]]
                if self._error is not None:
                    error = self._error[:, parts[i]]
                else:
                    error = None
                if self._mask is not None:
                    mask = self._mask[:, parts[i]]
                else:
                    mask = None
                if self._sky is not None:
                    sky = self._sky[:, parts[i]]
                else:
                    sky = None
                if self._wave is not None:
                    if len(self._wave.shape) == 2:
                        wave = self._wave[:, parts[i]]
                    else:
                        wave = self._wave[parts[i]]
                else:
                    wave = None
                if self._lsf is not None:
                    if len(self._lsf.shape) == 2:
                        lsf = self._lsf[:, parts[i]]
                    else:
                        lsf = self._lsf[parts[i]]
                else:
                    lsf = None
            elif axis == 1:
                data = self._data[parts[i]]
                if self._error is not None:
                    error = self._error[parts[i]]
                else:
                    error = None
                if self._mask is not None:
                    mask = self._mask[parts[i]]
                else:
                    mask = None
                if self._sky is not None:
                    sky = self._sky[parts[i]]
                else:
                    sky = None
                if self._wave is not None:
                    if len(self._wave.shape) == 2:
                        wave = self._wave[parts[i]]
                    else:
                        wave = self._wave
                else:
                    wave = None
                if self._lsf is not None:
                    if len(self._lsf.shape) == 2:
                        lsf = self._lsf[parts[i]]
                    else:
                        lsf = self._lsf[parts[i]]
                else:
                    lsf = None

            rss = RSS(
                data=data,
                error=error,
                mask=mask,
                wave=wave,
                lsf=lsf,
                sky=sky,
                header=Header(header=self._header)._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
            )
            rss_parts.append(rss)
        return rss_parts

    def splitFiberType(self, contains=["CAL", "SKY", "OBJ"]):
        splitted_rss = []
        # try:
        for types in contains:
            type = types.split(";")
            select = numpy.zeros(self._fibers, dtype="bool")
            for i in range(len(type)):
                select = numpy.logical_or(select, self._fiber_type == type[i])

            splitted_rss.append(self.subRSS(select))
        return splitted_rss

    def centreBary(self, guess_x, guess_y, radius, exponent=4):
        dist = self.distance(guess_x, guess_y)
        select = dist <= radius
        bary_x = numpy.sum(
            self._data[select, :] ** exponent
            * self._arc_position_x[select][:, numpy.newaxis],
            0,
        ) / numpy.sum(self._data[select, :] ** exponent, 0)
        bary_y = numpy.sum(
            self._data[select, :] ** exponent
            * self._arc_position_y[select][:, numpy.newaxis],
            0,
        ) / numpy.sum(self._data[select, :] ** exponent, 0)
        return bary_x, bary_y

    def getPositionTable(self):
        posTab = PositionTable(
            shape=self._shape,
            size=self._size,
            arc_position_x=self._arc_position_x,
            arc_position_y=self._arc_position_y,
            good_fibers=self._good_fibers,
            fiber_type=self._fiber_type,
        )
        return posTab

    def getSlitmap(self):
        return self._slitmap
    
    def setSlitmap(self, slitmap):
        self._slitmap = Table(slitmap)

        # define fiber positions in WCS
        if self._header is not None:
            wcs = WCS(header=self._header).to_header()
            wcs.update({"NAXIS": 2, "NAXIS2": self._header["NAXIS2"], "CRPIX2": 1,
                        "CRVAL2": 1, "CDELT2": 1, "CTYPE2": "LINEAR"})
            self._header.update(wcs)

    def apply_pixelmask(self, mask=None):
        if mask is None:
            mask = self._mask
        if mask is None:
            return self._data, self._error, self._lsf

        if self._mask is not None:
            self._data[self._mask] = numpy.nan
            if self._error is not None:
                self._error[self._mask] = numpy.nan
            if self._lsf is not None:
                self._lsf[self._mask] = numpy.nan

        return self._data, self._error, self._lsf

    def set_fluxcal(self, fluxcal):
        self._fluxcal = fluxcal
    
    def get_fluxcal(self):
        return self._fluxcal

    def get_cent_trace(self):
        return self._cent_trace
    
    def set_cent_trace(self, cent_trace):
        self._cent_trace = self._trace_to_coeff_table(cent_trace)
        return self._cent_trace

    def get_width_trace(self):
        return self._width_trace
    
    def set_width_trace(self, width_trace):
        self._width_trace = self._trace_to_coeff_table(width_trace)
        return self._width_trace

    def get_wave_trace(self):
        return self._wave_trace
    
    def set_wave_trace(self, wave_trace):
        self._wave_trace = self._trace_to_coeff_table(wave_trace)
        return self._wave_trace
    
    def get_lsf_trace(self):
        return self._lsf_trace

    def set_lsf_trace(self, lsf_trace):
        self._lsf_trace = self._trace_to_coeff_table(lsf_trace)
        return self._lsf_trace

    def writeFitsData(self, out_rss, include_PT=False):
        """Writes information from a RSS object into a FITS file.
        
        Parameters
        ----------
        out_rss : str
            Name or Path of the FITS file to which the data shall be written
        include_PT : bool, optional
            If True, the position table will be included in the FITS file, by default False

        Raises
        ------
        ValueError
            Invalid wavelength array shape
        ValueError
            Invalid LSF array shape
        """
        hdus = pyfits.HDUList()

        hdus.append(pyfits.PrimaryHDU(self._data.astype("float32")))
        if self._error is not None:
            hdus.append(pyfits.ImageHDU(self._error.astype("float32"), name="ERROR"))
        if self._mask is not None:
            hdus.append(pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX"))
        
        if self._wave_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._wave_trace, name="WAVE_TRACE"))
        elif self._wave is not None:
            if len(self._wave.shape) == 1:
                wcs = WCS(
                    header={"CDELT1": self._wave_disp, "CRVAL1": self._wave_start,
                    "CUNIT1": "angstrom", "CTYPE1": "WAVE", "CRPIX1": 1.0})
                self._header.update(wcs.to_header())
            elif len(self._wave.shape) == 2:
                raise ValueError("Missing wavelength trace information")
            else:
                raise ValueError("Invalid wavelength array shape")
        
        if self._lsf_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._lsf_trace, name="LSF_TRACE"))
        elif self._lsf is not None:
            if len(self._lsf.shape) == 1:
                hdus.append(pyfits.ImageHDU(self._lsf.astype("float32"), name="LSF"))
            elif len(self._lsf.shape) == 2:
                raise ValueError("Missing LSF trace information")
            else:
                raise ValueError("Invalid LSF array shape")
        
        if self._cent_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._cent_trace, name="CENT_TRACE"))
        if self._width_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._width_trace, name="WIDTH_TRACE"))
        if self._sky is not None:
            hdus.append(pyfits.ImageHDU(self._sky.astype("float32"), name="SKY"))
        if self._sky_error is not None:
            hdus.append(pyfits.ImageHDU(self._sky_error.astype("float32"), name="SKY_ERROR"))
        if self._fluxcal is not None:
            hdus.append(pyfits.BinTableHDU(self._fluxcal, name="FLUXCAL"))
        if self._slitmap is not None:
            hdus.append(pyfits.BinTableHDU(self._slitmap, name="SLITMAP"))

        if include_PT:
            try:
                table = self.writeFitsPosTable()
                hdus.append(pyfits.BinTableHDU(data=table.data, header=table.header, name="PosTable"))
            except (IndexError, ValueError, AttributeError):
                pass

        if self._header is not None:
            hdus[0].header = self.getHeader()
            hdus[0].update_header()
        hdus.writeto(out_rss, overwrite=True, output_verify="silentfix")

def loadRSS(in_rss):
    rss = RSS.from_file(in_rss)
    return rss


class lvmFrame(RSS):
    """lvmFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = hdulist["PRIMARY"].header
        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data
        wave_trace = Table(hdulist["WAVE_TRACE"].data)
        lsf_trace = Table(hdulist["LSF_TRACE"].data)
        cent_trace = Table(hdulist["CENT_TRACE"].data)
        width_trace = Table(hdulist["WIDTH_TRACE"].data)
        superflat = hdulist["SUPERFLAT"].data
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave_trace=wave_trace, lsf_trace=lsf_trace,
                   cent_trace=cent_trace, width_trace=width_trace,
                   superflat=superflat, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, slitmap=None, wave_trace=None, superflat=None, **kwargs):        
        RSS.__init__(self, data=data, error=error, mask=mask, header=header, slitmap=slitmap)

        self._blueprint = dp.load_blueprint(name="lvmFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)
    
        if wave_trace is not None:
            self.setWaveTrace(wave_trace)
        else:
            self._wave_trace = None
        if superflat is not None:
            self.setSuperflat(superflat)
        else:
            self._superflat = None
        if header is not None:
            self.setHeader(header, **kwargs)
    
    def setHeader(self, orig_header, **kwargs):
        """Set header"""
        blueprint = dp.load_blueprint(name="lvmFrame")
        new_header = orig_header
        new_cards = []
        for card in blueprint["hdu0"]["header"]:
            kw = card["key"]
            cm = card["comment"]
            if kw.lower() in kwargs:
                new_cards.append((kw, kwargs[kw.lower()], cm))
        new_header.update(new_cards)
        self._header = new_header
        return self._header

    def getWaveTrace(self):
        """Wavelength trace representation as FiberRows"""
        if self._wave_trace is not None:
            return FiberRows.from_table(self._wave_trace)
        else:
            return None

        if self._lsf_trace is not None:
            return FiberRows.from_table(self._lsf_trace)
        else:
            return None

    def setWaveTrace(self, wave_trace, lsf_trace):
        """Set wavelength/LSF trace representation"""
        self._wave_trace = self._trace_to_coeff_table(wave_trace)
        self._lsf_trace = self._trace_to_coeff_table(lsf_trace)
        return self._wave_trace, self._lsf_trace

    def getSuperflat(self):
        """Get superflat representation as numpy array"""
        return self._superflat

    def setSuperflat(self, superflat):
        """Set superflat representation"""
        self._superflat = superflat
        return self._superflat

    def getFiberTrace(self):
        """Get fiber centroid/width trace representation as FiberRows"""
        if self._cent_trace is not None:
            cent_trace = TraceMask.from_coeff_table(self._cent_trace)
        else:
            cent_trace = None
        if self._width_trace is not None:
            width_trace = TraceMask.from_coeff_table(self._wave_trace)
        else:
            width_trace = None
        return cent_trace, width_trace

    def setFiberTrace(self, cent_trace, width_trace):
        self._cent_trace = self._trace_to_coeff_table(cent_trace)
        self._width_trace = self._trace_to_coeff_table(width_trace)
        return self._cent_trace, self._width_trace

    def loadFitsData(self, in_file):
        with pyfits.open(in_file) as hdulist:
            self._header = hdulist["PRIMARY"].header
            self._data = hdulist["FLUX"].data
            self._error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
            self._error = numpy.sqrt(self._error)
            self._mask = hdulist["MASK"].data.astype("bool")
            self._wave_trace = Table(hdulist["WAVE_TRACE"].data)
            self._lsf_trace = Table(hdulist["LSF_TRACE"].data)
            self._cent_trace = Table(hdulist["CENT_TRACE"].data)
            self._width_trace = Table(hdulist["WIDTH_TRACE"].data)
            self._superflat = hdulist["SUPERFLAT"].data
            self._slitmap = Table(hdulist["SLITMAP"].data)

    def writeFitsData(self, out_file):
        # update flux header
        for kw in ["BUNIT", "BSCALE", "BZERO"]:
            if kw in self._header:
                self._template["FLUX"].header[kw] = self._header.pop(kw, None)
        # update primary header
        self._template["PRIMARY"].header.update(self._header)
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE_TRACE"] = pyfits.BinTableHDU(data=self._wave_trace, name="WAVE_TRACE")
        self._template["LSF_TRACE"] = pyfits.BinTableHDU(data=self._lsf_trace, name="LSF_TRACE")
        self._template["CENT_TRACE"] = pyfits.BinTableHDU(data=self._cent_trace, name="CENT_TRACE")
        self._template["WIDTH_TRACE"] = pyfits.BinTableHDU(data=self._width_trace, name="WIDTH_TRACE")
        self._template["SUPERFLAT"].data = self._superflat
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        self._template.writeto(out_file, overwrite=True)


class lvmCFrame(RSS):
    """lvmCFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = hdulist["PRIMARY"].header
        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data
        wave = hdulist["WAVE"].data
        superflat = hdulist["SUPERFLAT"].data
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave=wave, superflat=superflat, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, slitmap=None, wave=None, superflat=None, **kwargs):
        RSS.__init__(self, data=data, error=error, mask=mask, header=header, slitmap=slitmap, wave=wave)
    
        self._blueprint = dp.load_blueprint(name="lvmCFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        if superflat is not None:
            self.setSuperflat(superflat)
        else:
            self._superflat = None
        if header is not None:
            self.setHeader(header, **kwargs)
        else:
            self._header = None

    def setHeader(self, orig_header, **kwargs):
        """Set header"""
        blueprint = dp.load_blueprint(name="lvmCFrame")
        new_header = orig_header
        new_cards = []
        # iterate over PRIMARY and FLUX headers
        for i in range(2):
            for card in blueprint[f"hdu{i}"]["header"]:
                kw = card["key"]
                cm = card["comment"]
                if kw.lower() in kwargs:
                    new_cards.append((kw, kwargs[kw.lower()], cm))
        new_header.update(new_cards)

        new_header["CCD"] = ",".join([channel for channel in kwargs.get("channels", [])])
        new_header["NAXIS1"], new_header["NAXIS2"] = self._data.shape[1], self._data.shape[0]
        # update header with WCS
        if self._wave is not None:
            wcs = WCS(new_header)
            wcs.spectral.wcs.cdelt[0] = self._wave[1] - self._wave[0]
            wcs.spectral.wcs.crval[0] = self._wave[0]
            new_header.update(wcs.to_header())
        self._header = new_header
        return self._header

    def getSuperflat(self):
        """Get superflat representation as numpy array"""
        return self._superflat

    def setSuperflat(self, superflat):
        """Set superflat representation"""
        self._superflat = superflat
        return self._superflat
    
    def loadFitsData(self, in_file):
        with pyfits.open(in_file) as f:
            self._data = f["FLUX"].data
            self._error = numpy.divide(1, f["IVAR"].data, where=f["IVAR"].data != 0, out=numpy.zeros_like(f["IVAR"].data))
            self._error = numpy.sqrt(self._error)
            self._mask = f["MASK"].data.astype("bool")
            self._wave = f["WAVE"].data.astype("float32")
            self._superflat = f["SUPERFLAT"].data
            self._slitmap = Table(f["SLITMAP"].data)
            self._header = f["PRIMARY"].header
            for kw in ["BUNIT", "BSCALE", "BZERO"]:
                if kw in f["FLUX"].header:
                    self._header[kw] = f["FLUX"].header.get(kw)
    
    def writeFitsData(self, out_file):
        # update flux header
        for kw in ["BUNIT", "BSCALE", "BZERO"]:
            if kw in self._header:
                self._template["FLUX"].header[kw] = self._header.get(kw)
        # update primary header
        self._template["PRIMARY"].header.update(self._header)
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE"].data = self._wave.astype("float32")
        self._template["SUPERFLAT"].data = self._superflat
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        # write template
        self._template.writeto(out_file, overwrite=True)


class lvmFFrame(RSS):
    """lvmFFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = hdulist["PRIMARY"].header
        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data
        wave = hdulist["WAVE"].data
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave=wave, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, slitmap=None, wave=None, **kwargs):
        RSS.__init__(self, data=data, error=error, mask=mask, header=header, slitmap=slitmap, wave=wave)
    
        self._blueprint = dp.load_blueprint(name="lvmFFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        if header is not None:
            self.setHeader(header, **kwargs)
        else:
            self._header = None

    def setHeader(self, orig_header, **kwargs):
        """Set header"""
        blueprint = dp.load_blueprint(name="lvmFFrame")
        new_header = orig_header
        new_cards = []
        # iterate over PRIMARY and FLUX headers
        for i in range(2):
            for card in blueprint[f"hdu{i}"]["header"]:
                kw = card["key"]
                cm = card["comment"]
                if kw.lower() in kwargs:
                    new_cards.append((kw, kwargs[kw.lower()], cm))
        new_header.update(new_cards)

        new_header["CCD"] = ",".join([channel for channel in kwargs.get("channels", [])])
        new_header["NAXIS1"], new_header["NAXIS2"] = self._data.shape[1], self._data.shape[0]
        # update header with WCS
        if self._wave is not None:
            wcs = WCS(new_header)
            wcs.spectral.wcs.cdelt[0] = self._wave[1] - self._wave[0]
            wcs.spectral.wcs.crval[0] = self._wave[0]
            new_header.update(wcs.to_header())
        self._header = new_header
        return self._header

    def loadFitsData(self, in_file):
        with pyfits.open(in_file) as f:
            self._data = f["FLUX"].data
            self._error = numpy.divide(1, f["IVAR"].data, where=f["IVAR"].data != 0, out=numpy.zeros_like(f["IVAR"].data))
            self._error = numpy.sqrt(self._error)
            self._mask = f["MASK"].data.astype("bool")
            self._wave = f["WAVE"].data.astype("float32")
            self._slitmap = Table(f["SLITMAP"].data)
            self._header = f["PRIMARY"].header
            for kw in ["BUNIT", "BSCALE", "BZERO"]:
                if kw in f["FLUX"].header:
                    self._header[kw] = f["FLUX"].header.get(kw)
    
    def writeFitsData(self, out_file):
        # update flux header
        for kw in ["BUNIT", "BSCALE", "BZERO"]:
            if kw in self._header:
                self._template["FLUX"].header[kw] = self._header.get(kw)
        # update primary header
        self._template["PRIMARY"].header.update(self._header)
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE"].data = self._wave.astype("float32")
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        # write template
        self._template.writeto(out_file, overwrite=True)


class lvmSFrame(RSS):
    """lvmSFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = hdulist["PRIMARY"].header
        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data
        wave = hdulist["WAVE"].data
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave=wave, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, slitmap=None, wave=None, superflat=None, **kwargs):
        RSS.__init__(self, data=data, error=error, mask=mask, header=header, slitmap=slitmap, wave=wave)
    
        self._blueprint = dp.load_blueprint(name="lvmSFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        if header is not None:
            self.setHeader(header, **kwargs)
        else:
            self._header = None

    def setHeader(self, orig_header, **kwargs):
        """Set header"""
        blueprint = dp.load_blueprint(name="lvmSFrame")
        new_header = orig_header
        new_cards = []
        # iterate over PRIMARY and FLUX headers
        for i in range(2):
            for card in blueprint[f"hdu{i}"]["header"]:
                kw = card["key"]
                cm = card["comment"]
                if kw.lower() in kwargs:
                    new_cards.append((kw, kwargs[kw.lower()], cm))
        new_header.update(new_cards)

        new_header["CCD"] = ",".join([channel for channel in kwargs.get("channels", [])])
        new_header["NAXIS1"], new_header["NAXIS2"] = self._data.shape[1], self._data.shape[0]
        # update header with WCS
        if self._wave is not None:
            wcs = WCS(new_header)
            wcs.spectral.wcs.cdelt[0] = self._wave[1] - self._wave[0]
            wcs.spectral.wcs.crval[0] = self._wave[0]
            new_header.update(wcs.to_header())
        self._header = new_header
        return self._header

    def getSky(self):
        """Get sky representation as numpy array"""
        return self._sky
    
    def setSky(self, sky):
        """Set sky representation"""
        self._sky = sky
        return self._sky
    
    def getSupersky(self):
        """Get supersky representation as numpy array"""
        return self._supersky

    def setSupersky(self, supersky):
        """Set supersky representation"""
        self._supersky = supersky
        return self._supersky

    def loadFitsData(self, in_file):
        with pyfits.open(in_file) as f:
            self._data = f["FLUX"].data
            self._error = numpy.divide(1, f["IVAR"].data, where=f["IVAR"].data != 0, out=numpy.zeros_like(f["IVAR"].data))
            self._error = numpy.sqrt(self._error)
            self._mask = f["MASK"].data.astype("bool")
            self._wave = f["WAVE"].data.astype("float32")
            self._sky = f["SKY"].data.astype("float32")
            self._supersky = Table(f["SUPERSKY"].data)
            self._slitmap = Table(f["SLITMAP"].data)
            self._header = f["PRIMARY"].header
            for kw in ["BUNIT", "BSCALE", "BZERO"]:
                if kw in f["FLUX"].header:
                    self._header[kw] = f["FLUX"].header.get(kw)
    
    def writeFitsData(self, out_file):
        # update flux header
        for kw in ["BUNIT", "BSCALE", "BZERO"]:
            if kw in self._header:
                self._template["FLUX"].header[kw] = self._header.get(kw)
        # update primary header
        self._template["PRIMARY"].header.update(self._header)
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE"].data = self._wave.astype("float32")
        self._template["SKY"].data = self._sky.astype("float32")
        self._template["SUPERSKY"] = pyfits.BinTableHDU(data=self._supersky, name="SUPERSKY")
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        # write template
        self._template.writeto(out_file, overwrite=True)


class lvmRSS(RSS):
    pass