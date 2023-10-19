import os
import numpy
import bottleneck as bn
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u

from lvmdrp import log
from lvmdrp.core.constants import CONFIG_PATH
from lvmdrp.core.apertures import Aperture
from lvmdrp.core.cube import Cube
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.header import Header
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.spectrum1d import Spectrum1D


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


def _chain_join(b, r, z):
    ii = [i for i in [b, r, z] if i]
    x = ii[0]
    for e in ii[1:]:
        x = x.coaddSpec(e)
    return x

class RSS(FiberRows):
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
            inst_fwhm=numpy.zeros((n_spectra, ref_spec._inst_fwhm.size))
            if ref_spec._inst_fwhm is not None
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
            numpy.repeat(rss._inst_fwhm[0][None, :], rss._fibers, axis=0),
            rss._inst_fwhm,
        ):
            rss.setInstFWHM(rss._inst_fwhm[0])
        else:
            rss.setInstFWHM(rss._inst_fwhm)
        return rss

    def __init__(
        self,
        data=None,
        wave=None,
        inst_fwhm=None,
        header=None,
        error=None,
        mask=None,
        sky=None,
        sky_error=None,
        shape=None,
        size=None,
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
        self._wave = None
        self._wave_disp = None
        self._wave_start = None
        self._res_elements = None
        self._inst_fwhm = None
        self._sky = None
        self._sky_error = None
        if wave is not None:
            self.setWave(wave)
        else:
            self.createWavefromHdr(logwave=logwave)
        if inst_fwhm is not None:
            self.setInstFWHM(inst_fwhm)
        if sky is not None:
            self._sky = sky
        if sky_error is not None:
            self._sky_error = sky_error
        
        self.setSlitmap(slitmap)
        self.set_fluxcal(fluxcal)

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
                if self._dim == dim:
                    data = self._data * other
                elif len(dim) == 1:
                    if self._dim[0] == dim[0]:
                        data = self._data * other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
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

        if self._inst_fwhm is not None:
            if len(self._inst_fwhm.shape) == 1:
                inst_fwhm = self._inst_fwhm
            else:
                inst_fwhm = self._inst_fwhm[fiber, :]
        else:
            inst_fwhm = None

        if self._error is not None:
            error = self._error[fiber, :]
        else:
            error = None

        if self._mask is not None:
            mask = self._mask[fiber, :]
        else:
            mask = None

        spec = Spectrum1D(wave, data, error=error, mask=mask, inst_fwhm=inst_fwhm)
        return spec

    def __setitem__(self, fiber, spec):
        self._data[fiber, :] = spec._data

        if self._wave is not None and len(self._wave.shape) == 2:
            self._wave[fiber, :] = spec._wave

        if self._inst_fwhm is not None and len(self._inst_fwhm.shape) == 2:
            self._inst_fwhm[fiber, :] = spec._inst_fwhm

        if self._error is not None and spec._error is not None:
            self._error[fiber, :] = spec._error

        if self._mask is not None and spec._mask is not None:
            self._mask[fiber, :] = spec._mask

    def setWave(self, wave, unit="Angstrom"):
        self._wave = numpy.array(wave)

        if len(wave.shape) == 1:
            self._wave_disp = self._wave[1] - self._wave[0]
            self._wave_start = self._wave[0]
            self._res_elements = self._wave.shape[0]
            if self._header is not None:
                wcs = WCS(header={
                    "CDELT1": self._wave_disp, "CRVAL1": self._wave_start,
                    "CUNIT1": unit, "CTYPE1": "WAVE", "CRPIX1": 1.0})
                self._header.update(wcs.to_header())
        if len(wave.shape) == 2:
            self._res_elements = self._wave.shape[1]

    def setInstFWHM(self, inst_fwhm):
        try:
            if len(inst_fwhm) > 0:
                self._inst_fwhm = numpy.array(inst_fwhm)
        except Exception:
            self._inst_fwhm = inst_fwhm

    def maskFiber(self, fiber, replace_error=1e10):
        self._data[fiber, :] = 0
        if self._mask is not None:
            self._mask[fiber, :] = True
        if self._error is not None:
            self._error[fiber, :] = replace_error

    def createWavefromHdr(self, logwave=False):
        if self._header is not None:
            wcs = WCS(self._header)
            if wcs.spectral.array_shape:
                self._res_elements = wcs.spectral.array_shape[0]
                wl = wcs.spectral.all_pix2world(numpy.arange(self._res_elements), 0)[0]
                self._wave = (wl * u.m).to(u.angstrom).value
                self._wave_disp = self._wave[1] - self._wave[0]
                self._wave_start = self._wave[0]
                if logwave:
                    self._wave = 10 ** (self._wave)

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
        return RSS(data=self._sky, error=self._sky_error, mask=self._mask, wave=self._wave, inst_fwhm=self._inst_fwhm, header=header)

    def loadFitsData(
        self,
        file,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_wave=None,
        extension_fwhm=None,
        extension_sky=None,
        extension_skyerror=None,
        extension_hdr=None,
        extension_PT=None,
        extension_slitmap=None,
        extension_fluxcal=None,
        logwave=False,
    ):
        """
        Load data from a FITS image into an RSS object (Fibers in y-direction, dispersion in x-direction)

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
            and extension_wave is None
            and extension_fwhm is None
            and extension_sky is None
            and extension_skyerror is None
            and extension_slitmap is None
            and extension_fluxcal is None
        ):
            self._data = hdu[0].data.astype("float32")
            self.setHeader(header=hdu[0].header, origin=file)
            self.createWavefromHdr(logwave=logwave)
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header["EXTNAME"].split()[0] == "ERROR":
                        self._error = hdu[i].data.astype("float32")
                    if hdu[i].header["EXTNAME"].split()[0] == "BADPIX":
                        self._mask = hdu[i].data.astype("bool")
                        self._good_fibers = numpy.where(numpy.sum(self._mask, axis=1) != self._data.shape[1])[0]
                    if hdu[i].header["EXTNAME"].split()[0] == "WAVE":
                        self.setWave(hdu[i].data.astype("float32"))
                    if hdu[i].header["EXTNAME"].split()[0] == "INSTFWHM":
                        self.setInstFWHM(hdu[i].data.astype("float32"))
                    if hdu[i].header["EXTNAME"].split()[0] == "SLITMAP":
                        self.setSlitmap(Table(hdu[i].data))
                    if hdu[i].header["EXTNAME"].split()[0] == "SKY":
                        self._sky = hdu[i].data.astype("float32")
                    if hdu[i].header["EXTNAME"].split()[0] == "SKY_ERROR":
                        self._sky_error = hdu[i].data.astype("float32")
                    if hdu[i].header["EXTNAME"].split()[0] == "FLUXCAL":
                        self.set_fluxcal(Table(hdu[i].data))
                    if hdu[i].header["EXTNAME"].split()[0] == "POSTABLE":
                        self.loadFitsPosTable(hdu[i])
        else:
            if extension_hdr is not None:
                self.setHeader(hdu[extension_hdr].header, origin=file)
            if extension_data is not None:
                self._data = hdu[extension_data].data.astype("float32")
            if extension_mask is not None:
                self._mask = hdu[extension_mask].data
                self._good_fibers = numpy.where(numpy.sum(self._mask, axis=1) != self._data.shape[1])[0]
            if extension_error is not None:
                self._error = hdu[extension_error].data.astype("float32")
            if extension_wave is not None:
                self.setWave(hdu[extension_wave].data.astype("float32"))
            if extension_fwhm is not None:
                self.setInstFWHM(hdu[extension_fwhm].data.astype("float32"))
            if extension_slitmap is not None:
                self.setSlitmap(Table(hdu[extension_slitmap].data))
            if extension_sky is not None:
                self._sky = hdu[extension_sky].data.astype("float32")
            if extension_skyerror is not None:
                self._sky_error = hdu[extension_skyerror].data.astype("float32")
            if extension_fluxcal is not None:
                self.set_fluxcal(Table(hdu[extension_fluxcal].data))
        
        self._fibers = self._data.shape[0]
        self._pixels = numpy.arange(self._data.shape[1])

        hdu.close()

    def writeFitsData(
        self,
        filename,
        extension_data=None,
        extension_mask=None,
        extension_error=None,
        extension_wave=None,
        extension_fwhm=None,
        extension_sky=None,
        extension_skyerror=None,
        extension_slitmap=None,
        extension_fluxcal=None,
        include_PT=True,
    ):
        """
        Save information from a RSS object into a FITS file.
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
        self._data = self._data.astype(numpy.float32)
        if self._error is not None:
            self._error = self._error.astype(numpy.float32)
        if self._wave is not None:
            self._wave = self._wave.astype(numpy.float32)
        if self._inst_fwhm is not None:
            self._inst_fwhm = self._inst_fwhm.astype(numpy.float32)
        if self._sky is not None:
            self._sky = self._sky.astype(numpy.float32)
        if self._sky_error is not None:
            self._sky_error = self._sky_error.astype(numpy.float32)

        hdus = [None, None, None, None, None, None, None, None, None]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if (
            extension_data is None
            and extension_error is None
            and extension_mask is None
            and extension_wave is None
            and extension_slitmap is None
            and extension_sky is None
            and extension_skyerror is None
            and extension_fluxcal is None
        ):
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._wave is not None:
                if len(self._wave.shape) > 1:
                    hdus[1] = pyfits.ImageHDU(self._wave, name="WAVE")
            if self._inst_fwhm is not None:
                hdus[2] = pyfits.ImageHDU(self._inst_fwhm, name="INSTFWHM")
            if self._error is not None:
                hdus[3] = pyfits.ImageHDU(self._error, name="ERROR")
            if self._mask is not None:
                hdus[4] = pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX")
            if self._slitmap is not None:
                hdus[5] = pyfits.BinTableHDU(self._slitmap, name="SLITMAP")
            if self._sky is not None:
                hdus[6] = pyfits.ImageHDU(self._sky, name="SKY")
            if self._sky_error is not None:
                hdus[7] = pyfits.ImageHDU(self._sky_error, name="SKY_ERROR")
            if self._fluxcal is not None:
                hdus[8] = pyfits.BinTableHDU(self._fluxcal, name="FLUXCAL")
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data > 0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name="DATA")

            # wavelength hdu
            if extension_wave == 0:
                hdu = pyfits.PrimaryHDU(self._wave)
            elif extension_wave > 0 and extension_wave is not None:
                hdus[extension_wave] = pyfits.ImageHDU(self._wave, name="WAVE")

            # instrumental FWHM hdu
            if extension_fwhm == 0:
                hdu = pyfits.PrimaryHDU(self._inst_fwhm)
            elif extension_fwhm > 0 and extension_fwhm is not None:
                hdus[extension_fwhm] = pyfits.ImageHDU(self._inst_fwhm, name="INSTFWHM")

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

            # slitmap hdu
            if extension_slitmap == 0:
                hdu = pyfits.PrimaryHDU(self._slitmap)
            elif extension_slitmap > 0 and extension_slitmap is not None:
                hdus[extension_slitmap] = pyfits.BinTableHDU(self._slitmap, name="SLITMAP")

            # sky hdu
            if extension_sky == 0:
                hdu = pyfits.PrimaryHDU(self._sky)
            elif extension_sky > 0 and extension_sky is not None:
                hdus[extension_sky] = pyfits.ImageHDU(self._sky, name="SKY")
            
            # sky error hdu
            if extension_skyerror == 0:
                hdu = pyfits.PrimaryHDU(self._sky_error)
            elif extension_skyerror > 0 and extension_skyerror is not None:
                hdus[extension_skyerror] = pyfits.ImageHDU(self._sky_error, name="SKY_ERROR")
            
            # fluxcal hdu
            if extension_fluxcal == 0:
                hdu = pyfits.PrimaryHDU(self._fluxcal)
            elif extension_fluxcal > 0 and extension_fluxcal is not None:
                hdus[extension_fluxcal] = pyfits.BinTableHDU(self._fluxcal, name="FLUXCAL")

        if include_PT:
            try:
                table = self.writeFitsPosTable()
                hdus[-1] = pyfits.BinTableHDU(
                    data=table.data, header=table.header, name="PosTable"
                )
            except (IndexError, ValueError, AttributeError):
                pass

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
        hdu.writeto(
            filename, overwrite=True, output_verify="silentfix"
        )  # write FITS file to disc

    def getSpec(self, fiber):
        data = self._data[fiber, :]
        if self._wave is not None:
            if len(self._wave.shape) == 1:
                wave = self._wave
            else:
                wave = self._wave[fiber, :]
        else:
            wave = numpy.arange(data.size)
        if self._inst_fwhm is not None:
            if len(self._inst_fwhm.shape) == 1:
                inst_fwhm = self._inst_fwhm
            else:
                inst_fwhm = self._inst_fwhm[fiber, :]
        else:
            inst_fwhm = None
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

        spec = Spectrum1D(wave, data, error=error, mask=mask, inst_fwhm=inst_fwhm, sky=sky, sky_error=sky_error)
        
        return spec

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

        if method == "mean":
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

        if method == "weighted_mean" and error is not None:
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

        if method == "median":
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
        self._inst_fwhm = rss_in[0]._inst_fwhm
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
            if self._inst_fwhm is not None:
                inst_fwhm = self._inst_fwhm[select].flatten()[idx]
            else:
                inst_fwhm = None
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
            if self._inst_fwhm is not None and len(self._inst_fwhm.shape) == 2:
                inst_fwhm = numpy.mean(self._inst_fwhm, 0)
            else:
                inst_fwhm = self._inst_fwhm
        
        header = self._header
        
        spec = Spectrum1D(
            wave=wave,
            data=data,
            error=error,
            inst_fwhm=inst_fwhm,
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

        if self._inst_fwhm is not None:
            if len(self._inst_fwhm.shape) == 2:
                inst_fwhm = self._inst_fwhm[select, :]
            else:
                inst_fwhm = self._inst_fwhm
        else:
            inst_fwhm = None
        
        if self._sky is not None:
            sky = self._sky[select, :]
        else:
            sky = None

        rss = RSS(
            data=data,
            wave=wave,
            inst_fwhm=inst_fwhm,
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
                if self._inst_fwhm is not None:
                    if len(self._inst_fwhm.shape) == 2:
                        inst_fwhm = self._inst_fwhm[:, parts[i]]
                    else:
                        inst_fwhm = self._inst_fwhm[parts[i]]
                else:
                    inst_fwhm = None
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
                        wave = self._wave[parts[i]]
                else:
                    wave = None
                if self._inst_fwhm is not None:
                    if len(self._inst_fwhm.shape) == 2:
                        inst_fwhm = self._inst_fwhm[parts[i]]
                    else:
                        inst_fwhm = self._inst_fwhm[parts[i]]
                else:
                    inst_fwhm = None

            rss = RSS(
                data=data,
                error=error,
                mask=mask,
                wave=wave,
                inst_fwhm=inst_fwhm,
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
        self._slitmap = slitmap

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
            return self._data, self._error, self._inst_fwhm

        if self._mask is not None:
            self._data[self._mask] = numpy.nan
            if self._error is not None:
                self._error[self._mask] = numpy.nan
            if self._inst_fwhm is not None:
                self._inst_fwhm[self._mask] = numpy.nan

        return self._data, self._error, self._inst_fwhm

    def set_fluxcal(self, fluxcal):
        self._fluxcal = fluxcal
    
    def get_fluxcal(self):
        return self._fluxcal


def loadRSS(infile, extension_data=None, extension_mask=None, extension_error=None, extension_sky=None):
    rss = RSS()
    rss.loadFitsData(
        infile, extension_data=None, extension_mask=None, extension_error=None, extension_sky=None
    )

    return rss
