import os
import numpy
import bottleneck as bn
from copy import deepcopy as copy
from tqdm import tqdm
from scipy import interpolate
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy import units as u

from lvmdrp import log
from lvmdrp.core.constants import CONFIG_PATH
from lvmdrp.core.apertures import Aperture
from lvmdrp.core.cube import Cube
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.header import Header, combineHdr
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.spectrum1d import Spectrum1D, find_continuum, wave_little_interpol
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.fit_profile import polyfit2d, polyval2d
from lvmdrp.core.resample import resample_flux, rebin_spectra

from lvmdrp import __version__ as drpver



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
        sky_east, sky_east_error = None, None
        sky_west, sky_west_error = None, None
        supersky, supersky_error = None, None
        fluxcal_std, fluxcal_sci = None, None
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
                if hdu.name == "LSF_TRACE":
                    lsf_trace = hdu
                if hdu.name == "WAVE":
                    wave = hdu.data.astype("float32")
                if hdu.name == "LSF":
                    lsf = hdu.data.astype("float32")
                if hdu.name == "CENT_TRACE":
                    cent_trace = hdu
                if hdu.name == "WIDTH_TRACE":
                    width_trace = hdu
                if hdu.name == "SKY":
                    sky = hdu.data.astype("float32")
                if hdu.name == "SKY_ERROR":
                    sky_error = hdu.data.astype("float32")
                if hdu.name == "SKY_EAST":
                    sky_east = hdu.data.astype("float32")
                if hdu.name == "SKY_EAST_ERROR":
                    sky_east_error = hdu.data.astype("float32")
                if hdu.name == "SKY_WEST":
                    sky_west = hdu.data.astype("float32")
                if hdu.name == "SKY_WEST_ERROR":
                    sky_west_error = hdu.data.astype("float32")
                if hdu.name == "SUPERSKY":
                    supersky = hdu
                if hdu.name == "SUPERSKY_ERROR":
                    supersky_error = hdu
                if hdu.name == "FLUXCAL_STD":
                    fluxcal_std = hdu
                if hdu.name == "FLUXCAL_SCI":
                    fluxcal_sci = hdu
                if hdu.name == "SLITMAP":
                    slitmap = hdu

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
                sky_east=sky_east,
                sky_east_error=sky_east_error,
                sky_west=sky_west,
                sky_west_error=sky_west_error,
                supersky=supersky,
                supersky_error=supersky_error,
                header=header,
                slitmap=slitmap,
                fluxcal_std=fluxcal_std,
                fluxcal_sci=fluxcal_sci
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
                if rss._cent_trace is not None:
                    cent_trace_out = rss._cent_trace
                if rss._width_trace is not None:
                    width_trace_out = rss._width_trace
                if rss._wave_trace is not None:
                    wave_trace_out = rss._wave_trace
                if rss._lsf_trace is not None:
                    lsf_trace_out = rss._lsf_trace
                if rss._sky is not None:
                    sky_out = rss._sky
                if rss._sky_error is not None:
                    sky_error_out = rss._sky_error
                if rss._supersky is not None:
                    supersky_out = rss._supersky
                if rss._supersky_error is not None:
                    supersky_error_out = rss._supersky_error
                if rss._header is not None:
                    hdrs.append(Header(rss.getHeader()))
                if rss._fluxcal_std is not None:
                    fluxcal_std_out = rss._fluxcal_std
                if rss._fluxcal_sci is not None:
                    fluxcal_sci_out = rss._fluxcal_sci
            else:
                data_out = numpy.concatenate((data_out, rss._data), axis=0)

                if rss._cent_trace is not None:
                    cent_trace_out = rss.stack_trace((cent_trace_out, rss._cent_trace))
                else:
                    cent_trace_out = None
                if rss._width_trace is not None:
                    width_trace_out = rss.stack_trace((width_trace_out, rss._width_trace))
                else:
                    width_trace_out = None
                if rss._wave_trace is not None:
                    wave_trace_out = rss.stack_trace((wave_trace_out, rss._wave_trace))
                else:
                    wave_trace_out = None
                if rss._lsf_trace is not None:
                    lsf_trace_out = rss.stack_trace((lsf_trace_out, rss._lsf_trace))
                else:
                    lsf_trace_out = None
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
                if rss._supersky is not None:
                    supersky_out = rss.stack_supersky((supersky_out, rss._supersky))
                else:
                    supersky_out = None
                if rss._supersky_error is not None:
                    supersky_error_out = rss.stack_supersky((supersky_error_out, rss._supersky_error))
                else:
                    supersky_error_out = None
                if rss._header is not None:
                    hdrs.append(Header(rss.getHeader()))
                if rss._fluxcal_std is not None:
                    f = fluxcal_std_out.to_pandas()
                    fluxcal_std_out = Table.from_pandas(f.combine_first(rss._fluxcal_std.to_pandas()))
                else:
                    fluxcal_std_out = None
                if rss._fluxcal_sci is not None:
                    f = fluxcal_sci_out.to_pandas()
                    fluxcal_sci_out = Table.from_pandas(f.combine_first(rss._fluxcal_sci.to_pandas()))
                else:
                    fluxcal_sci_out = None

        # update header
        if len(hdrs) > 0:
            hdr_out = combineHdr(hdrs)
            hdr_out._header["CCD"] = hdr_out._header["CCD"][0]
        else:
            hdr_out = None


        # update slitmap
        slitmap_out = rss._slitmap

        return cls(
            data=data_out,
            error=error_out,
            mask=mask_out,
            cent_trace=cent_trace_out,
            width_trace=width_trace_out,
            wave_trace=wave_trace_out,
            lsf_trace=lsf_trace_out,
            sky=sky_out,
            sky_error=sky_error_out,
            supersky=supersky_out,
            supersky_error=supersky_error_out,
            header=hdr_out._header,
            slitmap=slitmap_out,
            fluxcal_std=fluxcal_std_out,
            fluxcal_sci=fluxcal_sci_out

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

        rsss = [copy(rss_b), copy(rss_r), copy(rss_z)]

        # get wavelengths
        log.info("merging wavelength arrays")
        waves = [numpy.round(rss._wave, 6) for rss in rsss]
        new_wave = numpy.unique(numpy.concatenate(waves))
        sampling = numpy.diff(new_wave)

        # optionally interpolate if the merged wavelengths are not monotonic
        fluxes, errors, masks, lsfs, skies, sky_errors = [], [], [], [], [], []
        skies_e, skies_w, sky_e_errors, sky_w_errors = [], [], [], []
        if numpy.all(numpy.isclose(sampling, sampling[0], atol=1e-6)):
            log.info(f"current wavelength sampling: min = {sampling.min():.2f}, max = {sampling.max():.2f}")
            # extend rss._data to new_wave filling with NaNs
            for rss in rsss:
                rss = rss.extendData(new_wave)
                fluxes.append(rss._data)
                errors.append(rss._error)
                masks.append(rss._mask)
                lsfs.append(rss._lsf)
                skies.append(rss._sky)
                sky_errors.append(rss._sky_error)
                skies_e.append(rss._sky_east)
                sky_e_errors.append(rss._sky_east_error)
                skies_w.append(rss._sky_west)
                sky_w_errors.append(rss._sky_west_error)
            fluxes = numpy.asarray(fluxes)
            errors = numpy.asarray(errors)
            masks = numpy.asarray(masks)
            lsfs = numpy.asarray(lsfs)
            skies = numpy.asarray(skies)
            sky_errors = numpy.asarray(sky_errors)
            skies_e = numpy.asarray(skies_e)
            sky_e_errors = numpy.asarray(sky_e_errors)
            skies_w = numpy.asarray(skies_w)
            sky_w_errors = numpy.asarray(sky_w_errors)
        else:
            log.warning("merged wavelengths are not monotonic, interpolation needed")
            rsss[0].add_header_comment("merged wavelengths are not monotonic, interpolation needed")
            # compute the combined wavelengths
            new_wave = wave_little_interpol(waves)
            sampling = numpy.diff(new_wave)
            log.info(f"new wavelength sampling: min = {sampling.min():.2f}, max = {sampling.max():.2f}")

            # define interpolators
            log.info("interpolating RSS data in new wavelength array")
            for rss in rsss:
                f = rebin_spectra(new_wave, rss._wave, rss._data, fill=numpy.nan)
                fluxes.append(f.astype("float32"))
                f = rebin_spectra(new_wave, rss._wave, rss._error, fill=numpy.nan)
                errors.append(f.astype("float32"))
                f = rebin_spectra(new_wave, rss._wave, rss._mask, axis=1, kind="nearest", bounds_error=False, fill_value=0)
                masks.append(f(new_wave).astype("uint8"))
                f = rebin_spectra(new_wave, rss._wave, rss._lsf, fill=numpy.nan)
                lsfs.append(f.astype("float32"))
                if rss._sky is not None:
                    f = rebin_spectra(new_wave, rss._wave, rss._sky, fill=numpy.nan)
                    skies.append(f.astype("float32"))
                if rss._sky_error is not None:
                    f = rebin_spectra(new_wave, rss._wave, rss._sky_error, fill=numpy.nan)
                    sky_errors.append(f.astype("float32"))
                if rss._sky_east is not None:
                    f = rebin_spectra(new_wave, rss._wave, rss._sky_east, fill=numpy.nan)
                    skies_e.append(f.astype("float32"))
                if rss._sky_east_error is not None:
                    f = rebin_spectra(new_wave, rss._wave, rss._sky_east_error, fill=numpy.nan)
                    sky_e_errors.append(f.astype("float32"))
                if rss._sky_west is not None:
                    f = rebin_spectra(new_wave, rss._wave, rss._sky_west, fill=numpy.nan)
                    skies_w.append(f.astype("float32"))
                if rss._sky_west_error is not None:
                    f = rebin_spectra(new_wave, rss._wave, rss._sky_west_error, fill=numpy.nan)
                    sky_w_errors.append(f.astype("float32"))
            fluxes = numpy.asarray(fluxes)
            errors = numpy.asarray(errors)
            masks = numpy.asarray(masks)
            lsfs = numpy.asarray(lsfs)
            skies = numpy.asarray(skies)
            sky_errors = numpy.asarray(sky_errors)
            skies_e = numpy.asarray(skies_e)
            sky_e_errors = numpy.asarray(sky_e_errors)
            skies_w = numpy.asarray(skies_w)
            sky_w_errors = numpy.asarray(sky_w_errors)

        # get overlapping ranges
        mask_overlap_br = (new_wave >= rss_r._wave[0]) & (new_wave <= rss_b._wave[-1])
        mask_overlap_rz = (new_wave >= rss_z._wave[0]) & (new_wave <= rss_r._wave[-1])
        # get channel ranges (excluding overlapping regions)
        mask_b = ((new_wave >= rss_b._wave[0]) & (new_wave <= rss_b._wave[-1])) & (~mask_overlap_br)
        mask_r = ((new_wave >= rss_r._wave[0]) & (new_wave <= rss_r._wave[-1])) & (~mask_overlap_br) & (~mask_overlap_rz)
        mask_z = ((new_wave >= rss_z._wave[0]) & (new_wave <= rss_z._wave[-1])) & (~mask_overlap_rz)

        # define weights for channel combination
        vars = errors ** 2
        log.info("combining channel data")
        if use_weights:
            weights = numpy.zeros_like(vars)
            weights[:, :, mask_overlap_br|mask_overlap_rz] = 1.0 / vars[:, :, mask_overlap_br|mask_overlap_rz]
            weights[numpy.isnan(weights)] = 0.0
            # normalize weights
            weights[:, :, mask_overlap_br] = weights[:, :, mask_overlap_br] / bn.nansum(weights[:, :, mask_overlap_br], axis=0)[None]
            weights[:, :, mask_overlap_rz] = weights[:, :, mask_overlap_rz] / bn.nansum(weights[:, :, mask_overlap_rz], axis=0)[None]
            # set weights for non-overlapping regions to 1 in their corresponding channels
            weights[0, :, mask_b] = 1.0
            weights[1, :, mask_r] = 1.0
            weights[2, :, mask_z] = 1.0

            new_data = bn.nansum(fluxes * weights, axis=0)
            new_lsf = bn.nansum(lsfs * weights, axis=0)
            new_error = numpy.sqrt(bn.nansum(vars, axis=0))
            new_mask = (bn.nansum(masks, axis=0)>0)
            if rss._sky is not None:
                new_sky = bn.nansum(skies * weights, axis=0)
            else:
                new_sky = None
            if rss._sky_error is not None:
                new_sky_error = numpy.sqrt(bn.nansum(sky_errors ** 2 * weights ** 2, axis=0))
            else:
                new_sky_error = None
            if rss._sky_east is not None:
                new_skye = bn.nansum(skies_e * weights, axis=0)
            else:
                new_skye = None
            if rss._sky_east_error is not None:
                new_skye_error = numpy.sqrt(bn.nansum(sky_e_errors ** 2 * weights ** 2, axis=0))
            else:
                new_skye_error = None
            if rss._sky_west is not None:
                new_skyw = bn.nansum(skies_w * weights, axis=0)
            else:
                new_skyw = None
            if rss._sky_west_error is not None:
                new_skyw_error = numpy.sqrt(bn.nansum(sky_w_errors ** 2 * weights ** 2, axis=0))
            else:
                new_skyw_error = None
        else:
            # channel-combine RSS data
            new_data = bn.nanmean(fluxes, axis=0)
            new_lsf = bn.nanmean(lsfs, axis=0)
            new_error = numpy.sqrt(bn.nanmean(vars, axis=0))
            new_mask = bn.nansum(masks, axis=0).astype("bool")
            if skies.size != 0:
                new_sky = bn.nansum(skies, axis=0)
            else:
                new_sky = None
            if sky_errors.size != 0:
                new_sky_error = numpy.sqrt(bn.nanmean(sky_errors ** 2, axis=0))
            else:
                new_sky_error = None
            if skies_e.size != 0:
                new_skye = bn.nanmean(skies_e, axis=0)
            else:
                new_skye = None
            if sky_e_errors.size != 0:
                new_skye_error = numpy.sqrt(bn.nanmean(sky_e_errors ** 2, axis=0))
            else:
                new_skye_error = None
            if skies_w.size != 0:
                new_skyw = bn.nanmean(skies_w, axis=0)
            else:
                new_skyw = None
            if sky_w_errors.size != 0:
                new_skyw_error = numpy.sqrt(bn.nanmean(sky_w_errors ** 2, axis=0))
            else:
                new_skyw_error = None

        # create RSS
        new_hdr = rsss[0]._header.copy()
        for rss in rsss[1:]:
            new_hdr.update(rss._header)

        new_rss = RSS(
            data=new_data,
            error=new_error,
            mask=new_mask,
            wave=new_wave,
            lsf=new_lsf,
            sky=new_sky,
            sky_error=new_sky_error,
            sky_east=new_skye,
            sky_east_error=new_skye_error,
            sky_west=new_skyw,
            sky_west_error=new_skyw_error,
            header=new_hdr,
            slitmap=rsss[0]._slitmap
        )
        return new_rss

    @classmethod
    def from_spectra1d(
        cls,
        spectra_list,
        header=None,
        shape=None,
        size=None,
        arc_position_x=None,
        arc_position_y=None,
        slitmap=None,
        good_fibers=None,
        fiber_type=None,
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
            slitmap=slitmap,
            good_fibers=good_fibers,
            fiber_type=fiber_type,
        )
        for i in range(n_spectra):
            rss[i] = spectra_list[i]

        # set wavelength and LSF in RSS object
        if numpy.allclose(
            numpy.repeat(rss._wave[0][None, :], rss._fibers, axis=0), rss._wave
        ):
            rss.set_wave_array(rss._wave[0])
        else:
            rss.set_wave_array(rss._wave)
        if numpy.allclose(
            numpy.repeat(rss._lsf[0][None, :], rss._fibers, axis=0),
            rss._lsf,
        ):
            rss.set_lsf_array(rss._lsf[0])
        else:
            rss.set_lsf_array(rss._lsf)
        return rss

    def __init__(
        self,
        data=None,
        header=None,
        error=None,
        mask=None,
        sky=None,
        sky_error=None,
        supersky=None,
        supersky_error=None,
        sky_east=None,
        sky_east_error=None,
        sky_west=None,
        sky_west_error=None,
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
        fluxcal_std=None,
        fluxcal_sci=None,
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
        self._supersky = None
        self._supersky_error = None
        self.set_sky(sky_master=sky, sky_master_error=sky_error,
                     sky_east=sky_east, sky_east_error=sky_east_error,
                     sky_west=sky_west, sky_west_error=sky_west_error)

        # set fiber traces information if available
        self.set_cent_trace(cent_trace)
        self.set_width_trace(width_trace)

        # set wavelength and LSF information if available
        self.set_wave_trace(wave_trace)
        self.set_wave_array(wave)

        self.set_lsf_trace(lsf_trace)
        self.set_lsf_array(lsf)

        # set supersky information if available
        if supersky is not None:
            self.set_supersky(supersky)
        if supersky_error is not None:
            self.set_supersky_error(supersky_error)

        self.setSlitmap(slitmap)
        self.set_fluxcal(fluxcal_std, source="std")
        self.set_fluxcal(fluxcal_sci, source="sci")

    def _trace_to_coeff_table(self, trace, default_poly_deg=4):
        """Converts a given trace into its polynomial coefficients representation as an Astropy Table"""
        if isinstance(trace, TraceMask):
            coeffs = trace._coeffs if trace._coeffs is not None else numpy.zeros((self._fibers, default_poly_deg+1))
            columns = [
                pyfits.Column(name="FUNC", format="A10", array=numpy.asarray([trace._poly_kind] * self._fibers)),
                pyfits.Column(name="XMIN", format="I", unit="pix", array=numpy.asarray([0] * self._fibers)),
                pyfits.Column(name="XMAX", format="I", unit="pix", array=numpy.asarray([trace._data.shape[1]-1] * self._fibers)),
                pyfits.Column(name="COEFF", format=f"{coeffs.shape[1]}E", dim=f"({self._fibers},)", array=trace._coeffs)
            ]
            self._trace = Table(pyfits.BinTableHDU.from_columns(columns).data)
            return self._trace
        elif isinstance(trace, pyfits.BinTableHDU):
            self._trace = Table(trace.data)
            return self._trace
        elif isinstance(trace, Table):
            self._trace = trace
            return self._trace
        elif trace is None:
            return None
        else:
            raise TypeError(f"trace must be lvmdrp.core.tracemask.TraceMask, astropy.table.Table or None, instead got {type(trace)}")

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

        data = self._data[fiber]

        if self._wave_trace is not None:
            wave_trace = self._wave_trace[fiber]
        else:
            wave_trace = None

        if self._lsf_trace is not None:
            lsf_trace = self._lsf_trace[fiber]
        else:
            lsf_trace = None

        if self._wave is not None:
            if len(self._wave.shape) == 1:
                wave = self._wave
            else:
                wave = self._wave[fiber]
        else:
            wave = numpy.arange(data.size)

        if self._lsf is not None:
            if len(self._lsf.shape) == 1:
                lsf = self._lsf
            else:
                lsf = self._lsf[fiber]
        else:
            lsf = None

        if self._error is not None:
            error = self._error[fiber]
        else:
            error = None

        if self._mask is not None:
            mask = self._mask[fiber]
        else:
            mask = None

        spec = Spectrum1D(data=data, error=error, mask=mask, wave=wave, lsf=lsf, wave_trace=wave_trace, lsf_trace=lsf_trace)
        return spec

    def __setitem__(self, fiber, spec):
        self._data[fiber, :] = spec._data

        if self._wave_trace is not None and spec._wave_trace is not None:
            self._wave_trace[fiber] = spec._wave_trace

        if self._lsf_trace is not None and spec._lsf_trace is not None:
            self._lsf_trace[fiber] = spec._lsf_trace

        if self._wave is not None and len(self._wave.shape) == 2:
            self._wave[fiber, :] = spec._wave

        if self._lsf is not None and len(self._lsf.shape) == 2:
            self._lsf[fiber, :] = spec._lsf

        if self._error is not None and spec._error is not None:
            self._error[fiber, :] = spec._error

        if self._mask is not None and spec._mask is not None:
            self._mask[fiber, :] = spec._mask

        if self._sky is not None and spec._sky is not None:
            self._sky[fiber] = spec._sky

        if self._sky_error is not None and spec._sky_error is not None:
            self._sky_error[fiber] = spec._sky_error

    def add_header_comment(self, comstr):
        '''
        Append a COMMENT card at the end of the FITS header.
        '''
        self._header.append(('COMMENT', comstr), bottom=True)

    def eval_wcs(self, wave=None, data=None, as_dict=True):
        """Returns the WCS object from the current wavelength and fibers arrays"""
        wave = wave or self._wave
        data = data or self._data

        if wave is not None and len(wave.shape) == 1:
            wcs_dict = {"NAXIS": 2, "NAXIS1": data.shape[1], "NAXIS2": data.shape[0],
                        "CDELT1": wave[1]-wave[0],
                        "CRVAL1": wave[0],
                        "CUNIT1": "Angstrom", "CTYPE1": "WAVE", "CRPIX1": 1,
                        "CDELT2": 1,
                        "CRVAL2": 1,
                        "CUNIT2": "", "CTYPE2": "FIBERID", "CRPIX2": 1}

        elif wave is None or len(wave.shape) == 2:
            wcs_dict = {"NAXIS": 2, "NAXIS1": data.shape[1], "NAXIS2": data.shape[0],
                        "CDELT1": 1,
                        "CRVAL1": 1,
                        "CUNIT1": "", "CTYPE1": "XAXIS", "CRPIX1": 1,
                        "CDELT2": 1,
                        "CRVAL2": 1,
                        "CUNIT2": "", "CTYPE2": "FIBERID", "CRPIX2": 1}
        if as_dict:
            return wcs_dict

        return WCS(header=wcs_dict)

    def set_cent_trace(self, cent_trace):
        self._cent_trace = self._trace_to_coeff_table(cent_trace)
        return self._cent_trace

    def get_cent_trace(self, as_tracemask=False):
        if not as_tracemask:
            return self._cent_trace
        else:
            return TraceMask.from_coeff_table(self._cent_trace)

    def get_width_trace(self, as_tracemask=False):
        if not as_tracemask:
            return self._width_trace
        else:
            return TraceMask.from_coeff_table(self._width_trace)

    def set_width_trace(self, width_trace):
        self._width_trace = self._trace_to_coeff_table(width_trace)
        return self._width_trace

    def get_wave_trace(self, as_tracemask=False):
        if not as_tracemask:
            return self.get_wave_trace()
        else:
            return TraceMask.from_coeff_table(self._wave_trace)

    def set_wave_trace(self, wave_trace):
        self._wave_trace = self._trace_to_coeff_table(wave_trace)
        return self._wave_trace

    def get_lsf_trace(self, as_tracemask=False):
        if not as_tracemask:
            return self.get_lsf_trace()
        else:
            return TraceMask.from_coeff_table(self._lsf_trace)

    def set_lsf_trace(self, lsf_trace):
        self._lsf_trace = self._trace_to_coeff_table(lsf_trace)
        return self._lsf_trace

    def set_wave_array(self, wave=None):
        """Sets the wavelength array for the RSS object

        This method tries to set the wavelength array from three different
        sources, in the following order:

            - from the input `wave` array, in which case it expects it to be a
            one-dimensional array with the same number of elements as the
            wavelength dimension of the data array.

            - from the header, in which case the wavelength the resulting
            wavelength will be a one-dimensional array.

            - from the wavelength trace, in which case the resulting wavelength
            array will be a two-dimensional array with the same shape as the
            data array.

        Parameters
        ----------
        wave : numpy.ndarray, optional
            Wavelength array to be set, by default None
        """
        # initialize wavelength attributes
        self._wave = None
        self._wave_disp = None
        self._wave_start = None
        self._res_elements = None

        # set new wavelength array if given
        if wave is not None:
            self._wave = numpy.asarray(wave)
            if len(wave.shape) == 1:
                if wave.size != self._data.shape[1]:
                    raise ValueError(f"Input wavelength array shape {wave.size} does not match the data shape {self._data.shape}")

                self._wave_disp = self._wave[1] - self._wave[0]
                self._wave_start = self._wave[0]
                self._res_elements = self._wave.shape[0]
            elif len(wave.shape) == 2:
                self._wave = numpy.array(wave)
            else:
                raise ValueError("Invalid wavelength array shape")
        elif self._header is not None and self._header.get("WAVREC", False):
            self._wave, self._res_elements, self._wave_start, self._wave_disp = self.get_wave_from_header()
        elif self._wave_trace is not None:
            trace = TraceMask.from_coeff_table(self._wave_trace)
            self._wave = trace.eval_coeffs()

        return self._wave

    def set_lsf_array(self, lsf=None):
        self._lsf = None

        if self._wave is None:
            return self._lsf

        if lsf is not None:
            self._lsf = lsf
        elif self._lsf_trace is not None:
            lsf = numpy.zeros_like(self._data)
            for ifiber in range(self._fibers):
                spec = self[ifiber]
                _, lsf[ifiber] = spec.eval_wave_and_lsf_traces(wave=spec._wave, wave_trace=spec._wave_trace, lsf_trace=spec._lsf_trace)
            self._lsf = lsf

        return self._lsf

    def match_lsf(self, target_fwhm=None, min_fwhm=0.1):
        """Downgrade spectral resolution to match LSF in all fibers

        This function will degrade the resolution of the RSS to match all
        fibers given a scalar value for `target_fwhm` in FWHM or a callable to
        generate one. If None is given, the resolution will be matched to the
        worst value in the LSF.

        Parameters
        ----------
        target_fwhm : float|callable[lsf], optional
            Target resolution or function to apply to current LSF to get target resoltion, by default None
        min_fwhm : float, optional
            Minimum FWHM allowed, by default 0.1

        Returns
        -------
        new_rss : lvmdrp.core.rss.RSS
            A copy of the RSS with the LSF matched to the given value
        """
        target_fwhm = target_fwhm or numpy.max(self._lsf)

        new_specs = []
        for ifiber in tqdm(range(self._fibers), desc="matching LSF", ascii=True, unit="fiber"):
            spec = self.getSpec(ifiber)
            if spec._mask.all():
                new_specs.append(spec)
                continue
            new_specs.append(spec.flatten_lsf(target_fwhm, min_fwhm=min_fwhm))

        return RSS.from_spectra1d(new_specs, header=self._header, slitmap=self._slitmap, good_fibers=self._good_fibers)

    def maskFiber(self, fiber, replace_error=1e10):
        self._data[fiber, :] = 0
        if self._mask is not None:
            self._mask[fiber, :] = True
        if self._error is not None:
            self._error[fiber, :] = replace_error

    def get_wave_from_header(self):
        wave, res_elements, wave_start, wave_disp = None, None, None, None
        if self._header is None:
            return wave, res_elements, wave_start, wave_disp

        header = self._header.copy()
        header["NAXIS"] = len(self._data.shape)
        header["NAXIS1"] = self._data.shape[1]
        header["NAXIS2"] = self._data.shape[0]
        wcs = WCS(header)
        if wcs.spectral.array_shape:
            res_elements = wcs.spectral.array_shape[0]
            wl = wcs.spectral.all_pix2world(numpy.arange(res_elements), 0)[0]
            wave = (wl * u.m).to(u.AA).value
            wave_disp = wave[1] - wave[0]
            wave_start = wave[0]

        return wave, res_elements, wave_start, wave_disp

    def set_sky(self, sky_master=None, sky_master_error=None, sky_east=None, sky_east_error=None, sky_west=None, sky_west_error=None):
        self._sky_east = sky_east
        self._sky_east_error = sky_east_error
        self._sky_west = sky_west
        self._sky_west_error = sky_west_error
        self._sky = sky_master
        self._sky_error = sky_master_error

    def get_sky(self):
        header = self._header
        if header is not None:
            header["IMAGETYP"] = "sky"
            header["OBJECT"] = "sky"
        return RSS(data=self._sky, error=self._sky_error, mask=self._mask, wave=self._wave, lsf=self._lsf, header=header)

    def set_supersky(self, supersky):
        if isinstance(supersky, pyfits.BinTableHDU):
            self._supersky = Table(supersky.data)
        elif isinstance(supersky, Table):
            self._supersky = supersky
        elif isinstance(supersky, tuple):
            wave, knots, coeffs, degree, telescope = supersky
            self._supersky = self.tck_to_table(wave, knots, coeffs, degree, telescope)
        else:
            raise TypeError(f"Invalid {supersky} value. Valid types are 'astropy.io.fits.BinTableHDU', 'astropy.table.Table' and 'tuple'")

    def set_supersky_error(self, supersky_error):
        if isinstance(supersky_error, pyfits.BinTableHDU):
            self._supersky_error = Table(supersky_error.data)
        elif isinstance(supersky_error, Table):
            self._supersky_error = supersky_error
        elif isinstance(supersky_error, tuple):
            wave, knots, coeffs, degree, telescope = supersky_error
            self._supersky_error = self.tck_to_table(wave, knots, coeffs, degree, telescope)
        else:
            raise TypeError(f"Invalid {supersky_error} value. Valid types are 'astropy.io.fits.BinTableHDU', 'astropy.table.Table' and 'tuple'")

    def get_supersky(self):
        return self._supersky

    def get_supersky_error(self):
        return self._supersky_error

    def eval_supersky(self, supersky=None, supersky_error=None):

        # get supersky spline parameters
        supersky = self._supersky if supersky is None else supersky
        supersky_error = self._supersky_error if supersky_error is None else supersky_error
        if supersky is None:
            raise ValueError("Cannot evaluate super sky, spline parameters are None")
        if supersky_error is None:
            raise ValueError("Cannot evaluate super sky error, spline parameters are None")

        telescopes = sorted(set(supersky["telescope"]))

        # separate east and west
        waves = dict(east=[], west=[])
        supersky_spline = dict(east=[], west=[])
        supersky_error_spline = dict(east=[], west=[])
        for telescope in telescopes:
            # separate by telescope
            select_telescope = supersky["telescope"] == telescope
            tcks, = tuple(zip(
                supersky["wave"][select_telescope],
                supersky["knots"][select_telescope],
                supersky["coeffs"][select_telescope],
                supersky["degree"][select_telescope]))

            tcks_error, = tuple(zip(
                supersky_error["wave"][select_telescope],
                supersky_error["knots"][select_telescope],
                supersky_error["coeffs"][select_telescope],
                supersky_error["degree"][select_telescope]))

            # separate wavelenths from spline parameters
            wave = tcks[0]
            wave = wave.reshape((-1, 4086))
            tck = tcks[1:]
            tck_error = tcks_error[1:]

            # evaluate supersky
            dlambda = numpy.diff(wave, axis=1)
            dlambda = numpy.column_stack((dlambda, dlambda[:, -1]))
            sky = numpy.zeros(wave.shape)
            error = numpy.zeros(wave.shape)
            for i in range(self._fibers):
                sky[i, :] = interpolate.splev(wave[i, :], tck)
                error[i, :] = interpolate.splev(wave[i, :], tck_error)

            # store supersky in dictionary
            waves[telescope] = wave
            supersky_spline[telescope] = sky
            supersky_error_spline[telescope] = error

        return waves, supersky_spline, supersky_error_spline

    def eval_master_sky(self, sky_east=None, sky_east_error=None, sky_west=None, sky_west_error=None, weights=None):
        w_e, w_w = weights or (self._header.get("SKYEW"), self._header.get("SKYWW"))
        if w_e is None or w_w is None:
            return None

        sky_east = sky_east or self._sky_east
        sky_east_error = sky_east_error or self._sky_east_error
        sky_west = sky_west or self._sky_west
        sky_west_error = sky_west_error or self._sky_west_error

        if sky_east is not None or sky_west is not None:
            sky_e = RSS(data=sky_east, error=sky_east_error, wave=self._wave)
            sky_w = RSS(data=sky_west, error=sky_west_error, wave=self._wave)
            return sky_e * w_e + sky_w * w_w

        return None

    def tck_to_table(self, wave, knots, coeffs, degree, telescope):
        # pack arguments for validation
        args = (wave, knots, coeffs, degree, telescope)
        types = (numpy.ndarray, numpy.ndarray, numpy.ndarray, int, str)

        tck_dict = dict(wave=[], knots=[], coeffs=[], degree=[], telescope=[])
        if isinstance(wave, list):
            # validate arguments list
            assert all(isinstance(arg, list) for arg in args), "All objects must be instances of list"
            # validate lists length
            length_set = {len(arg) for arg in args}
            assert len(length_set) == 1, "All lists must have the same length"

            for knots, coeffs, degree, telescope in zip(args):
                tck_dict["wave"].append(wave.ravel())
                tck_dict["knots"].append(knots)
                tck_dict["coeffs"].append(coeffs)
                tck_dict["degree"].append(degree)
                tck_dict["telescope"].append(telescope)
        else:
            # validate argument types
            assert all(isinstance(arg, type_) for arg, type_ in zip(args, types)), "Invalid argument type"
            tck_dict["wave"].append(wave.ravel())
            tck_dict["knots"].append(knots)
            tck_dict["coeffs"].append(coeffs)
            tck_dict["degree"].append(degree)
            tck_dict["telescope"].append(telescope)

        return Table(tck_dict)

    def stack_supersky(self, superskies):
        if isinstance(superskies, list):
            iterator = superskies
        elif isinstance(superskies, Table):
            iterator = []
            for supersky in superskies:
                iterator.extend(list(supersky.iterrows()))
        else:
            raise TypeError("superskies must be a list of tuples or an astropy.table.Table")

        tck_dict = dict(wave=[], knots=[], coeffs=[], degree=[], telescope=[])
        for wave, knots, coeffs, degree, telescope in iterator:
            tck_dict["wave"].append(wave)
            tck_dict["knots"].append(knots)
            tck_dict["coeffs"].append(coeffs)
            tck_dict["degree"].append(degree)
            tck_dict["telescope"].append(telescope)

        return Table(tck_dict)

    def getSpec(self, fiber):
        data = self._data[fiber, :]
        if self._wave is not None:
            if len(self._wave.shape) == 1:
                wave = self._wave
            else:
                wave = self._wave[fiber, :]
        else:
            wave = numpy.arange(data.size)
        if self._wave_trace is not None:
            wave_trace = self._wave_trace[fiber]
        else:
            wave_trace = None
        if self._lsf is not None:
            if len(self._lsf.shape) == 1:
                lsf = self._lsf
            else:
                lsf = self._lsf[fiber, :]
        else:
            lsf = None
        if self._lsf_trace is not None:
            lsf_trace = self._lsf_trace[fiber]
        else:
            lsf_trace = None
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

        spec = Spectrum1D(
            data=data,
            error=error,
            mask=mask,
            wave=wave,
            wave_trace=wave_trace,
            lsf=lsf,
            lsf_trace=lsf_trace,
            sky=sky,
            sky_error=sky_error)

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
        if self._sky_east is not None:
            new_sky_east = numpy.full((self._sky_east.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_sky_east[:, ipix:fpix+1] = self._sky_east
        else:
            new_sky_east = None
        if self._sky_east_error is not None:
            new_sky_east_error = numpy.full((self._sky_east_error.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_sky_east_error[:, ipix:fpix+1] = self._sky_east_error
        else:
            new_sky_east_error = None
        if self._sky_west is not None:
            new_sky_west = numpy.full((self._sky_west.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_sky_west[:, ipix:fpix+1] = self._sky_west
        else:
            new_sky_west = None
        if self._sky_west_error is not None:
            new_sky_west_error = numpy.full((self._sky_west_error.shape[0], new_wave.size), numpy.nan, dtype=numpy.float32)
            new_sky_west_error[:, ipix:fpix+1] = self._sky_west_error
        else:
            new_sky_west_error = None
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
        self._sky_east = new_sky_east
        self._sky_east_error = new_sky_east_error
        self._sky_west = new_sky_west
        self._sky_west_error = new_sky_west_error
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
                    collapsed[i] = bn.median(spec._data[goodpix])
                elif method == "sum":
                    collapsed[i] = numpy.sum(spec._data[goodpix])
                elif method == "mean":
                    collapsed[i] = numpy.mean(spec._data[goodpix])
        arg = numpy.argsort(collapsed)
        numbers = numpy.arange(self._fibers)
        select = numpy.logical_or(numbers < min, numbers > numbers[-1] - max)
        return arg[select]

    def rectify_wave(self, wave=None, wave_range=None, wave_disp=None):
        """Wavelength rectifies the RSS object

        This method rectifies the RSS object to an uniform wavelength grid. The
        wavelength grid can be specified in three different ways:

            - by providing a `wave` array, in which case it expects it to be a
            one-dimensional array with the same number of elements as the
            wavelength dimension of the data array and with uniform sampling.

            - by providing a `wave_range` and `wave_disp` values, in which case
            it expects `wave_range` to be a tuple with the lower and upper
            limits of the wavelength range, and `wave_disp` to be the
            wavelength dispersion.

        NOTE: all operations are perfomed in a copy of the RSS object, so the
        original object is not modified.

        Parameters
        ----------
        wave : array-like
            Wavelength array to rectify to, by default None
        wave_range : tuple, optional
            Wavelength range to rectify to, by default None
        wave_disp : float, optional
            Wavelength dispersion to rectify to, by default None

        Returns
        -------
        RSS
            Rectified RSS object
        """
        if self._wave is None and self._wave_trace is None:
            raise ValueError("No wavelength information found in RSS object")
        elif self._wave is None and self._wave_trace is not None:
            trace = TraceMask.from_coeff_table(self._wave_trace)
            self._wave = trace.eval_coeffs()

        if self._header is not None and self._header.get("WAVREC", False) or len(self._wave.shape) == 1:
            return self

        if wave is not None:
            # verify uniform sampling
            if not numpy.allclose(numpy.diff(wave), numpy.diff(wave)[0]):
                raise ValueError("Wavelength array must have uniform sampling")
        elif wave_range is not None and wave_disp is not None:
            wave = numpy.arange(wave_range[0], wave_range[1] + wave_disp, wave_disp)
        elif wave is None and wave_range is None and wave_disp is None:
            raise ValueError("No wavelength information provided to perform rectification")

        # make sure the interpolation happens in density space
        rss = copy(self)
        if rss._header is None:
            rss._header = pyfits.Header()
        unit = rss._header["BUNIT"]
        if not unit.endswith("/angstrom"):
            dlambda = numpy.gradient(rss._wave, axis=1)
            rss._data /= dlambda
            if rss._error is not None:
                rss._error /= dlambda
            if rss._sky is not None:
                rss._sky /= dlambda
            if rss._sky_error is not None:
                rss._sky_error /= dlambda
            unit = unit + "/angstrom"

        rss._header["BUNIT"] = unit
        rss._header["WAVREC"] = True
        # create output RSS
        new_rss = RSS(
            data=numpy.zeros((rss._fibers, wave.size), dtype="float32"),
            error=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._error is not None else None,
            mask=numpy.zeros((rss._fibers, wave.size), dtype="bool"),
            sky=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._sky is not None else None,
            sky_error=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._sky_error is not None else None,
            sky_east=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._sky_east is not None else None,
            sky_east_error=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._sky_east_error is not None else None,
            sky_west=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._sky_west is not None else None,
            sky_west_error=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._sky_west_error is not None else None,
            wave=wave,
            lsf=numpy.zeros((rss._fibers, wave.size), dtype="float32") if rss._lsf is not None else None,
            cent_trace=rss._cent_trace,
            width_trace=rss._width_trace,
            slitmap=rss._slitmap,
            header=rss._header
        )

        # fit and evaluate interpolators
        for ifiber in range(rss._fibers):
            f = resample_flux(wave, rss._wave[ifiber], rss._data[ifiber])
            new_rss._data[ifiber] = f.astype("float32")
            if rss._error is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._error[ifiber])
                new_rss._error[ifiber] = f.astype("float32")
            f = resample_flux(wave, rss._wave[ifiber], rss._mask[ifiber])
            new_rss._mask[ifiber] = f.astype("bool")
            new_rss._mask[ifiber] |= numpy.isnan(new_rss._data[ifiber])|(new_rss._data[ifiber]==0)
            new_rss._mask[ifiber] |= ~numpy.isfinite(new_rss._error[ifiber])
            if rss._lsf is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._lsf[ifiber])
                new_rss._lsf[ifiber] = f.astype("float32")
            if rss._sky is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._sky[ifiber])
                new_rss._sky[ifiber] = f.astype("float32")
            if rss._sky_error is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._sky_error[ifiber])
                new_rss._sky_error[ifiber] = f.astype("float32")
            if rss._sky_east is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._sky_east[ifiber])
                new_rss._sky_east[ifiber] = f.astype("float32")
            if rss._sky_east_error is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._sky_east_error[ifiber])
                new_rss._sky_east_error[ifiber] = f.astype("float32")
            if rss._sky_west is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._sky_west[ifiber])
                new_rss._sky_west[ifiber] = f.astype("float32")
            if rss._sky_west_error is not None:
                f = resample_flux(wave, rss._wave[ifiber], rss._sky_west_error[ifiber])
                new_rss._sky_west_error[ifiber] = f.astype("float32")
        # add supersky information if available
        if rss._supersky is not None:
            new_rss.set_supersky(rss._supersky)
            new_rss.set_supersky_error(rss._supersky_error)

        return new_rss

    # TODO: what do we want here in terms of densities?
    def to_native_wave(self, method="linear", interp_density=True, return_density=False):
        """Converts the wavelength to the native wavelength grid

        This method de-rectifies the RSS object to the native wavelength grid.
        The native wavelength grid is defined by the wavelength trace
        coefficients. If no wavelength trace is found, it returns the RSS
        object unchanged.

        NOTE: all operations are perfomed in a copy of the RSS object, so the
        original object is not modified.

        NOTE: this method should be used to de-rectify RSS objects that
        represent functions of wavelength, instead of samples. For example, RSS
        objects that represent a fiberflat, or a supersky.

        Parameters
        ----------
        method : str, optional
            Interpolation method, by default "linear"
        return_density : bool, optional
            If True, returns the density of the rectification, by default False

        Returns
        -------
        RSS
            De-rectified RSS object
        """
        if self._wave is None and self._header is None:
            raise ValueError("No wavelength information found in RSS object")
        elif self._wave is None and self._header is not None:
            self._wave, _, _, _ = self.get_wave_from_header()

        if self._header is not None and not self._header.get("WAVREC", False) or len(self._wave.shape) == 2:
            return self

        # get native wavelength grid
        trace = TraceMask.from_coeff_table(self._wave_trace)
        wave = trace.eval_coeffs()

        rss = copy(self)
        if rss._header is None:
            rss._header = pyfits.Header()
        unit = rss._header["BUNIT"]
        if not unit.endswith("/angstrom") and interp_density:
            dlambda = numpy.gradient(rss._wave)
            rss._data /= dlambda
            rss._error /= dlambda
            if rss._sky is not None:
                rss._sky /= dlambda
            if rss._sky_error is not None:
                rss._sky_error /= dlambda
            unit = unit + "/angstrom"

        rss._header["BUNIT"] = unit
        rss._header["WAVREC"] = False
        rss._header["METREC"] = (method, "Wavelength rectification method")
        if "CRPIX1" in rss._header:
            del rss._header["CRPIX1"]
        if "CRVAL1" in rss._header:
            del rss._header["CRVAL1"]
        if "CDELT1" in rss._header:
            del rss._header["CDELT1"]
        if "CTYPE1" in rss._header:
            del rss._header["CTYPE1"]
        # create output RSS
        new_rss = RSS(
            data=numpy.zeros((rss._fibers, wave.shape[1]), dtype="float32"),
            error=numpy.zeros((rss._fibers, wave.shape[1]), dtype="float32"),
            mask=numpy.zeros((rss._fibers, wave.shape[1]), dtype="bool"),
            sky=numpy.zeros((rss._fibers, wave.shape[1]), dtype="float32") if rss._sky is not None else None,
            sky_error=numpy.zeros((rss._fibers, wave.shape[1]), dtype="float32") if rss._sky_error is not None else None,
            cent_trace=rss._cent_trace,
            width_trace=rss._width_trace,
            wave_trace=rss._wave_trace,
            lsf_trace=rss._lsf_trace,
            slitmap=rss._slitmap,
            header=rss._header
        )

        # interpolate data, error, mask and sky arrays from rectified grid to original grid
        for ifiber in range(rss._fibers):
            f = resample_flux(wave[ifiber], rss._wave, rss._data[ifiber])
            new_rss._data[ifiber] = f.astype("float32")
            f = resample_flux(wave[ifiber][ifiber], rss._wave, rss._error[ifiber])
            new_rss._error[ifiber] = f.astype("float32")
            f = resample_flux(wave[ifiber][ifiber], rss._wave, rss._mask[ifiber], kind="nearest", bounds_error=False, fill_value=1)
            new_rss._mask[ifiber] = f(wave[ifiber]).astype("bool")
            if rss._sky is not None:
                f = resample_flux(wave[ifiber], rss._wave, rss._sky[ifiber])
                new_rss._sky[ifiber] = f.astype("float32")
            if rss._sky_error is not None:
                f = resample_flux(wave[ifiber], rss._wave, rss._sky_error[ifiber])
                new_rss._sky_error[ifiber] = f.astype("float32")
            if rss._sky_east is not None:
                f = resample_flux(wave[ifiber], rss._wave, rss._sky_east[ifiber])
                new_rss._sky_east[ifiber] = f.astype("float32")
            if rss._sky_east_error is not None:
                f = resample_flux(wave[ifiber], rss._wave, rss._sky_east_error[ifiber])
                new_rss._sky_east_error[ifiber] = f.astype("float32")
            if rss._sky_west is not None:
                f = resample_flux(wave[ifiber], rss._wave, rss._sky_west[ifiber])
                new_rss._sky_west[ifiber] = f.astype("float32")
            if rss._sky_west_error is not None:
                f = resample_flux(wave[ifiber], rss._wave, rss._sky_west_error[ifiber])
                new_rss._sky_west_error[ifiber] = f.astype("float32")

        if not return_density and unit.endswith("/angstrom"):
            dlambda = numpy.gradient(wave, axis=1)
            new_rss._data *= dlambda
            new_rss._error *= dlambda
            if new_rss._sky is not None:
                new_rss._sky *= dlambda
            if new_rss._sky_error is not None:
                new_rss._sky_error *= dlambda
            if new_rss._sky_east is not None:
                new_rss._sky_east *= dlambda
            if new_rss._sky_east_error is not None:
                new_rss._sky_east_error *= dlambda
            if new_rss._sky_west is not None:
                new_rss._sky_west *= dlambda
            if new_rss._sky_west_error is not None:
                new_rss._sky_west_error *= dlambda
            new_rss._header["BUNIT"] = unit.replace("/angstrom", "")
        elif return_density and not unit.endswith("/angstrom"):
            dlambda = numpy.gradient(wave, axis=1)
            new_rss._data /= dlambda
            new_rss._error /= dlambda
            if new_rss._sky is not None:
                new_rss._sky /= dlambda
            if new_rss._sky_error is not None:
                new_rss._sky_error /= dlambda
            if new_rss._sky_east is not None:
                new_rss._sky_east /= dlambda
            if new_rss._sky_east_error is not None:
                new_rss._sky_east_error /= dlambda
            if new_rss._sky_west is not None:
                new_rss._sky_west /= dlambda
            if new_rss._sky_west_error is not None:
                new_rss._sky_west_error /= dlambda
            new_rss._header["BUNIT"] = unit + "/angstrom"

        return new_rss

    def subtract_continuum(self, niter=5, thresh=0.999, median_box_range=(50, 300)):
        """Fits and subtracts the continuum contribution in each fiber

        Iteratively rejects pixels that are above `thresh` in the array

            median_filter(spec) / spec

        where spec is the spectrum in one fiber and the median_filter is an adaptive
        version of median filtering whereby the median box increases in size as a
        function of wavelength.

        Parameters
        ----------
        niter : int, optional
            Number of iterations, by default 5
        thresh : float, optional
            Threshold in median_filter(spec) / spec above which pixels get rejected, by default 0.999
        median_box_range : tuple[int], optional
            range of box sizes in adaptive median filtering, by default (50, 300)

        Returns
        -------
        new_rss : lvmdrp.core.rss.RSS
            Copy of self without continuum contribution
        cont_data : numpy.ndarray
            Continuum fitted for each fiber
        cont_select : numpy.ndarray
            Selection of continuum pixels
        """
        cont_select = numpy.zeros_like(self._data, dtype=bool)
        cont_data = numpy.zeros_like(self._data, dtype="float32")
        for ifiber in range(self._fibers):
            if self._mask[ifiber].sum() == self._data.shape[1]:
                continue
            cont_data[ifiber], cont_select[ifiber] = find_continuum(self._data[ifiber],
                                                                    niter=niter, thresh=thresh,
                                                                    median_box_max=median_box_range[1],
                                                                    median_box_min=median_box_range[0])

        new_rss = copy(self)
        new_rss._data = new_rss._data - cont_data
        return new_rss, cont_data, cont_select

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

    def stackRSS(self, rsss, axis=0):
        if axis == 0:
            axis = 1
        elif axis == 1:
            axis = 0

        data = numpy.concatenate([rss._data for rss in rsss], axis=axis)
        if self._error is not None:
            error = numpy.concatenate([rss._error for rss in rsss], axis=axis)
        else:
            error = None
        if self._mask is not None:
            mask = numpy.concatenate([rss._mask for rss in rsss], axis=axis)
        else:
            mask = None
        if self._sky is not None:
            sky = numpy.concatenate([rss._sky for rss in rsss], axis=axis)
        else:
            sky = None
        if self._wave is not None:
            if len(self._wave.shape) == 2:
                wave = numpy.concatenate([rss._wave for rss in rsss], axis=axis)
            else:
                wave = numpy.concatenate([numpy.repeat([rss._wave], self._fibers, axis=0) for rss in rsss], axis=axis)
        else:
            wave = None
        if self._inst_fwhm is not None:
            if len(self._inst_fwhm.shape) == 2:
                inst_fwhm = numpy.concatenate(
                    [rss._inst_fwhm for rss in rsss], axis=axis
                )
            else:
                inst_fwhm = numpy.concatenate(
                    [numpy.repeat([rss._inst_fwhm], self._fibers, axis=0) for rss in rsss], axis=axis
                )
        else:
            inst_fwhm = None
        if self._arc_position_x is not None:
            arc_position_x = numpy.concatenate(
                [rss._arc_position_x for rss in rsss], axis=0
            )
        else:
            arc_position_x = None
        if self._arc_position_y is not None:
            arc_position_y = numpy.concatenate(
                [rss._arc_position_y for rss in rsss], axis=0
            )
        else:
            arc_position_y = None
        if self._good_fibers is not None:
            good_fibers = numpy.concatenate(
                [rss._good_fibers+self._fibers for rss in rsss], axis=0
            )
        else:
            good_fibers = None

        rss = RSS(
            data=data,
            error=error,
            mask=mask,
            sky=sky,
            wave=wave,
            inst_fwhm=inst_fwhm,
            header=self._header,
            shape=self._shape,
            size=self._size,
            arc_position_x=arc_position_x,
            arc_position_y=arc_position_y,
            slitmap=self._slitmap,
            good_fibers=good_fibers
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
                if self._wave_trace is not None:
                    wave_trace = self._wave_trace[parts[i]]
                else:
                    wave_trace = None
                if self._lsf_trace is not None:
                    lsf_trace = self._lsf_trace[parts[i]]
                else:
                    lsf_trace = None
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
                        lsf = self._lsf
                else:
                    lsf = None
                if self._wave_trace is not None:
                    wave_trace = self._wave_trace[parts[i]]
                else:
                    wave_trace = None
                if self._lsf_trace is not None:
                    lsf_trace = self._lsf_trace[parts[i]]
                else:
                    lsf_trace = None

            rss = RSS(
                data=data,
                error=error,
                mask=mask,
                wave=wave,
                lsf=lsf,
                wave_trace=wave_trace,
                lsf_trace=lsf_trace,
                sky=sky,
                header=Header(header=self._header)._header,
                shape=self._shape,
                size=self._size,
                arc_position_x=self._arc_position_x,
                arc_position_y=self._arc_position_y,
                good_fibers=self._good_fibers,
                fiber_type=self._fiber_type,
                slitmap=self._slitmap
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
        if slitmap is None:
            self._slitmap = None
            return
        if isinstance(slitmap, pyfits.BinTableHDU):
            self._slitmap = Table(slitmap.data)
        elif isinstance(slitmap, Table):
            self._slitmap = slitmap
        else:
            raise TypeError(f"Invalid slitmap table type '{type(slitmap)}'")

        # # define fiber positions in WCS
        # if self._header is not None:
        #     wcs = WCS(header=self._header).to_header()
        #     wcs.update({"NAXIS": 2, "NAXIS2": self._header["NAXIS2"], "CRPIX2": 1,
        #                 "CRVAL2": 1, "CDELT2": 1, "CTYPE2": "LINEAR"})
        #     self._header.update(wcs)

    def apply_pixelmask(self, mask=None):
        """Replaces masked pixels in RSS by NaN values"""
        if mask is None:
            mask = self._mask
        if mask is None:
            return self._data, self._error

        if self._mask is not None:
            self._data[self._mask] = numpy.nan
            if self._error is not None:
                self._error[self._mask] = numpy.nan
            if self._sky is not None:
                self._sky[self._mask] = numpy.nan
            if self._sky_error is not None:
                self._sky_error[self._mask] = numpy.nan
            if self._sky_east is not None:
                self._sky_east[self._mask] = numpy.nan
            if self._sky_east_error is not None:
                self._sky_east_error[self._mask] = numpy.nan
            if self._sky_west is not None:
                self._sky_west[self._mask] = numpy.nan
            if self._sky_west_error is not None:
                self._sky_west_error[self._mask] = numpy.nan

        return self._data, self._error

    def set_fluxcal(self, fluxcal, source="std"):
        if fluxcal is None:
            setattr(self, f"_fluxcal_{source}", None)
            return
        if isinstance(fluxcal, pyfits.BinTableHDU):
            setattr(self, f"_fluxcal_{source}", Table(fluxcal.data))
        elif isinstance(fluxcal, Table):
            setattr(self, f"_fluxcal_{source}", fluxcal)
        else:
            raise TypeError(f"Invalid flux calibration table type '{type(fluxcal)}'")

    def get_fluxcal(self, source="std"):
        return getattr(self, f"_fluxcal_{source}")

    def stack_trace(self, traces):
        trace_dict = dict(FUNC=[], XMIN=[], XMAX=[], COEFF=[])
        iterator = []
        for trace in traces:
            iterator.extend(list(trace.iterrows()))
        for func, xmin, xmax, coeff in iterator:
            trace_dict["FUNC"].append(func)
            trace_dict["XMIN"].append(xmin)
            trace_dict["XMAX"].append(xmax)
            trace_dict["COEFF"].append(coeff)

        return Table(trace_dict)

    def coadd_flux(self, wrange):
        """Return the coadded flux along a given wavelength window for all fibers"""

        if self._wave is None:
            log.warning("missing wavelength information, not able to consistently coadd flux")
            return self

        naxis1 = self._data.shape[1]
        naxis2=self._data.shape[0]
        w = WCS(self._header)
        wave = w.spectral.pixel_to_world(numpy.arange(naxis1)).value*1e10
        selwave=(wave>=wrange[0])*(wave<=wrange[1])
        selwavemask=numpy.tile(selwave, (naxis2,1))

        flux = self._data
        mask = self._mask
        flux[mask] = numpy.nan
        masked = flux*selwavemask

        coadded_flux = numpy.nanmean(masked, axis=1)
        return coadded_flux

    def get_helio_rv(self, apply_hrv_corr=False):
        """Calculates heliocentric velocity corrections for each telescope and standard fiber

        Parameters
        ----------
        apply_heliorv : bool, optional
            Apply heliocentric correction to all fibers

        Returns
        -------
        hrv_corrs : dict[str, float]
            Dictionary containing heliocentric velocity corrections
        """
        if self._header is None or self._header["IMAGETYP"] != "object" or not self._header["PO*RA"] or not self._header["PO*DE"]:
            return

        # calculate heliocentric velocity
        obs_time = Time(self._header['OBSTIME'])
        hrv_corrs = {}
        for tel in ["SCI", "SKYE", "SKYW"]:
            ra = self._header.get(f"PO{tel}RA", self._header.get(f"{tel}RA", self._header.get(f"TE{tel}RA"))) or 0
            dec = self._header.get(f"PO{tel}DE", self._header.get(f"{tel}DE", self._header.get(f"TE{tel}DE"))) or 0
            if ra == 0 or dec == 0:
                log.warning(f"on heliocentric velocity correction, missing RA/Dec information in header, assuming: {ra = }, {dec = }")
                self.add_header_comment(f"on heliocentric velocity correction, missing RA/Dec information in header, assuming: {ra = }, {dec = }")
                self._header[f"HIERARCH WAVE HELIORV_{tel}"] = (numpy.round(0.0, 4), f"Heliocentric velocity correction for {tel} [km/s]")
                hrv_corrs[tel] = numpy.round(0.0, 4)
            else:
                radec = SkyCoord(ra, dec, unit="deg") # center of the pointing or coordinates of the fiber
                hrv_corr = radec.radial_velocity_correction(kind='heliocentric', obstime=obs_time, location=EarthLocation.of_site('lco')).to(u.km / u.s).value
                self._header[f"HIERARCH WAVE HELIORV_{tel}"] = (numpy.round(hrv_corr, 4), f"Heliocentric velocity correction for {tel} [km/s]")
                hrv_corrs[tel] = numpy.round(hrv_corr, 4)

        # calculate standard stars heliocentric corrections
        for istd in range(1, 15+1):
            is_acq = self._header[f"STD{istd}ACQ"]
            if not is_acq:
                continue

            std_obstime = Time(self._header[f"STD{istd}T0"])
            std_ra, std_dec = self._header.get(f"STD{istd}RA", 0.0), self._header.get(f"STD{istd}DE", 0.0)
            if std_ra == 0 or std_dec == 0:
                self._header[f"STD{istd}HRV"] = (0.0, f"Standard {istd} heliocentric vel. corr. [km/s]")
                continue
            std_radec = SkyCoord(std_ra, std_dec, unit="deg")
            std_hrv_corr = std_radec.radial_velocity_correction(kind="heliocentric", obstime=std_obstime, location=EarthLocation.of_site("lco")).to(u.km / u.s).value
            self._header[f"STD{istd}HRV"] = (numpy.round(std_hrv_corr, 4), f"Standard {istd} heliocentric vel. corr. [km/s]")

        # TODO: implement apply_heliorv
        if apply_hrv_corr: ...
            # if helio_vel is None or helio_vel == 0.0:
            #     helio_vel = rss._header.get(helio_vel_keyword)
            #     if helio_vel is None:
            #         helio_vel = 0.0
            #         log.warning(f"no heliocentric velocity found in header by keywords {helio_vel_keyword = }, assuming {helio_vel = } km/s")
            #         rss.add_header_comment(f"no heliocentric velocity {helio_vel_keyword = }, assuming {helio_vel = } km/s")
            # else:
            #     log.info(f"applying heliocentric velocity correction of {helio_vel = } km/s")

            # rss._wave = rss._wave * (1 + helio_vel / c.to("km/s").value)

        return hrv_corrs

    def fit_field_gradient(self, wrange, poly_deg):
        """Fits a polynomial function to the IFU field"""
        if self._slitmap is None:
            log.warning("not able to fit gradient without fibermap information")
            return self

        fibermap = self._slitmap
        telescope = fibermap["telescope"]

        flux = self.coadd_flux(wrange=wrange)

        x_e=fibermap["xpmm"].astype(float)[telescope=="SkyE"]
        y_e=fibermap["ypmm"].astype(float)[telescope=="SkyE"]
        x_w=fibermap["xpmm"].astype(float)[telescope=="SkyW"]
        y_w=fibermap["ypmm"].astype(float)[telescope=="SkyW"]
        x_s=fibermap["xpmm"].astype(float)[telescope=="Spec"]
        y_s=fibermap["ypmm"].astype(float)[telescope=="Spec"]

        flux = flux[telescope=="Sci"]
        x=fibermap["xpmm"].astype(float)[telescope=="Sci"]
        y=fibermap["ypmm"].astype(float)[telescope=="Sci"]

        flux_med = bn.nanmedian(flux)
        flux_fact = flux / flux_med
        select = numpy.isfinite(flux_fact)
        coeffs = polyfit2d(x[select], y[select], flux_fact[select], poly_deg)

        grad_model = polyval2d(x, y, coeffs)
        grad_model_e = polyval2d(x_e, y_e, coeffs)
        grad_model_w = polyval2d(x_w, y_w, coeffs)
        grad_model_s = polyval2d(x_s, y_s, coeffs)

        return x, y, flux, grad_model, grad_model_e, grad_model_w, grad_model_s

    def writeFitsData(self, out_rss, replace_masked=True, include_wave=False):
        """Writes information from a RSS object into a FITS file.

        Parameters
        ----------
        out_rss : str
            Name or Path of the FITS file to which the data shall be written
        replace_masked : bool
            If True, replaces masked values with NaN before writing RSS file
        include_wave : bool, optional
            If True, the wavelength array is included in the FITS file, by default False

        Raises
        ------
        ValueError
            Invalid wavelength array shape
        ValueError
            Invalid LSF array shape
        """
        if replace_masked:
            self.apply_pixelmask()

        hdus = pyfits.HDUList()

        hdus.append(pyfits.PrimaryHDU(self._data.astype("float32")))
        if self._error is not None:
            hdus.append(pyfits.ImageHDU(self._error.astype("float32"), name="ERROR"))
        if self._mask is not None:
            hdus.append(pyfits.ImageHDU(self._mask.astype("uint8"), name="BADPIX"))

        # include wavelength extension for rectified RSSs
        if include_wave and self._wave and len(self._wave.shape) == 1:
            hdus.append(pyfits.ImageHDU(self._wave.astype("float32"), name="WAVE"))

        if self._wave_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._wave_trace, name="WAVE_TRACE"))
        elif self._wave is not None:
            if len(self._wave.shape) == 1:
                # wcs = WCS(
                #     header={"CDELT1": self._wave_disp, "CRVAL1": self._wave_start,
                #     "CUNIT1": "Angstrom", "CTYPE1": "WAVE", "CRPIX1": 1.0})
                self._header.update({"CDELT1": self._wave_disp, "CRVAL1": self._wave_start,
                    "CUNIT1": "Angstrom", "CTYPE1": "WAVE", "CRPIX1": 1.0})
            elif len(self._wave.shape) == 2:
                hdus.append(pyfits.ImageHDU(self._wave.astype("float32"), name="WAVE"))
            else:
                raise ValueError("Missing wavelength trace information")

        if self._lsf_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._lsf_trace, name="LSF_TRACE"))
        elif self._lsf is not None:
            if len(self._lsf.shape) in {1, 2}:
                hdus.append(pyfits.ImageHDU(self._lsf.astype("float32"), name="LSF"))
            else:
                raise ValueError("Missing LSF trace information")

        if self._cent_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._cent_trace, name="CENT_TRACE"))
        if self._width_trace is not None:
            hdus.append(pyfits.BinTableHDU(self._width_trace, name="WIDTH_TRACE"))
        if self._sky is not None:
            hdus.append(pyfits.ImageHDU(self._sky.astype("float32"), name="SKY"))
        if self._sky_error is not None:
            hdus.append(pyfits.ImageHDU(self._sky_error.astype("float32"), name="SKY_ERROR"))
        if self._sky_east is not None:
            hdus.append(pyfits.ImageHDU(self._sky_east.astype("float32"), name="SKY_EAST"))
        if self._sky_east_error is not None:
            hdus.append(pyfits.ImageHDU(self._sky_east_error.astype("float32"), name="SKY_EAST_ERROR"))
        if self._sky_west is not None:
            hdus.append(pyfits.ImageHDU(self._sky_west.astype("float32"), name="SKY_WEST"))
        if self._sky_west_error is not None:
            hdus.append(pyfits.ImageHDU(self._sky_west_error.astype("float32"), name="SKY_WEST_ERROR"))
        if self._supersky is not None:
            hdus.append(pyfits.BinTableHDU(self._supersky, name="SUPERSKY"))
        if self._supersky_error is not None:
            hdus.append(pyfits.BinTableHDU(self._supersky_error, name="SUPERSKY_ERROR"))
        if self._fluxcal_std is not None:
            hdus.append(pyfits.BinTableHDU(self._fluxcal_std, name="FLUXCAL_STD"))
        if self._fluxcal_sci is not None:
            hdus.append(pyfits.BinTableHDU(self._fluxcal_sci, name="FLUXCAL_SCI"))
        if self._slitmap is not None:
            hdus.append(pyfits.BinTableHDU(self._slitmap, name="SLITMAP"))

        if self._header is not None:
            hdus[0].header = self.getHeader()
            hdus[0].update_header()

        os.makedirs(os.path.dirname(out_rss), exist_ok=True)
        hdus[0].header["FILENAME"] = os.path.basename(out_rss)
        hdus[0].header['DRPVER'] = drpver
        hdus.writeto(out_rss, overwrite=True, output_verify="silentfix")

def loadRSS(in_rss):
    rss = RSS.from_file(in_rss)
    return rss


class lvmBaseProduct(RSS):
    """Base class to define an LVM product"""

    _BPARS = {"BUNIT": None, "BSCALE": 1.0, "BZERO": 0.0}

    @classmethod
    def header_from_hdulist(cls, hdulist):
        header = hdulist["PRIMARY"].header
        for kw, vl in cls._BPARS.items():
            header[kw] = hdulist["FLUX"].header.pop(kw, vl)
        return header

    @classmethod
    def from_file(cls, in_file):
        with pyfits.open(in_file) as hdulist:
            return cls.from_hdulist(hdulist)

    def set_header(self, orig_header, **kwargs):
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

    def update_header(self):
        # update flux header
        for kw in ["BUNIT", "BSCALE", "BZERO"]:
            if kw in self._header:
                self._template["FLUX"].header[kw] = self._header.get(kw)

        # update primary header
        self._template["PRIMARY"].header.update(self._header)
        del self._template["PRIMARY"].header["WCS*"]
        del self._template["PRIMARY"].header["CDELT*"]
        del self._template["PRIMARY"].header["CRVAL*"]
        del self._template["PRIMARY"].header["CRPIX*"]
        del self._template["PRIMARY"].header["CTYPE*"]
        del self._template["PRIMARY"].header["CUNIT*"]
        if "DISPAXIS" in self._template["PRIMARY"].header:
            del self._template["PRIMARY"].header["DISPAXIS"]

        # update WCS
        wcs = self.eval_wcs()
        [hdu.header.update(wcs) for hdu in self._template if hdu.is_image and hdu.name != "PRIMARY"]


class lvmFrame(lvmBaseProduct):
    """lvmFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = cls.header_from_hdulist(hdulist)

        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data.astype("bool")
        cent_trace = Table(hdulist["CENT_TRACE"].data)
        width_trace = Table(hdulist["WIDTH_TRACE"].data)
        wave_trace = Table(hdulist["WAVE_TRACE"].data)
        lsf_trace = Table(hdulist["LSF_TRACE"].data)
        superflat = hdulist["SUPERFLAT"].data
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave_trace=wave_trace, lsf_trace=lsf_trace,
                   cent_trace=cent_trace, width_trace=width_trace,
                   superflat=superflat, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None,
                 cent_trace=None, width_trace=None, wave_trace=None, lsf_trace=None,
                 header=None, slitmap=None, superflat=None, **kwargs):
        lvmBaseProduct.__init__(self, data=data, error=error, mask=mask,
                     cent_trace=cent_trace, width_trace=width_trace,
                     wave_trace=wave_trace, lsf_trace=lsf_trace, header=header, slitmap=slitmap)

        self._blueprint = dp.load_blueprint(name="lvmFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        self.set_superflat(superflat)
        if header is not None:
            self.set_header(header, **kwargs)

    def get_superflat(self):
        """Get superflat representation as numpy array"""
        return self._superflat

    def set_superflat(self, superflat):
        """Set superflat representation"""
        self._superflat = superflat
        return self._superflat

    def loadFitsData(self, in_file):
        self = lvmFrame.from_file(in_file)
        return self

    def writeFitsData(self, out_file, replace_masked=True):
        # replace masked pixels
        if replace_masked:
            self.apply_pixelmask()

        # update headers
        self.update_header()
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
        self._template.verify("silentfix")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self._template[0].header["FILENAME"] = os.path.basename(out_file)
        self._template[0].header['DRPVER'] = drpver
        self._template.writeto(out_file, overwrite=True)


class lvmFFrame(lvmBaseProduct):
    """lvmFFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = cls.header_from_hdulist(hdulist)

        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data.astype("bool")
        wave = hdulist["WAVE"].data
        lsf = hdulist["LSF"].data
        sky_east = hdulist["SKY_EAST"].data
        sky_east_error = numpy.divide(1, hdulist["SKY_EAST_IVAR"].data, where=hdulist["SKY_EAST_IVAR"].data != 0, out=numpy.zeros_like(hdulist["SKY_EAST_IVAR"].data))
        sky_east_error = numpy.sqrt(sky_east_error)
        sky_west = hdulist["SKY_WEST"].data
        sky_west_error = numpy.divide(1, hdulist["SKY_WEST_IVAR"].data, where=hdulist["SKY_WEST_IVAR"].data != 0, out=numpy.zeros_like(hdulist["SKY_WEST_IVAR"].data))
        sky_west_error = numpy.sqrt(sky_west_error)
        fluxcal_std = Table(hdulist["FLUXCAL_STD"].data)
        fluxcal_sci = Table(hdulist["FLUXCAL_SCI"].data)
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave=wave, lsf=lsf,
                   sky_east=sky_east, sky_east_error=sky_east_error,
                   sky_west=sky_west, sky_west_error=sky_west_error,
                   fluxcal_std=fluxcal_std, fluxcal_sci=fluxcal_sci, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, wave=None, lsf=None,
                 sky_east=None, sky_east_error=None,
                 sky_west=None, sky_west_error=None,
                 fluxcal_std=None, fluxcal_sci=None, slitmap=None, **kwargs):
        lvmBaseProduct.__init__(self, data=data, error=error, mask=mask, header=header,
                     wave=wave, lsf=lsf,
                     sky_east=sky_east, sky_east_error=sky_east_error,
                     sky_west=sky_west, sky_west_error=sky_west_error,
                     fluxcal_std=fluxcal_std, fluxcal_sci=fluxcal_sci, slitmap=slitmap)

        self._blueprint = dp.load_blueprint(name="lvmFFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        if header is not None:
            self.set_header(header, **kwargs)
        else:
            self._header = None

    def loadFitsData(self, in_file):
        self = lvmFFrame.from_file(in_file)
        return self

    def writeFitsData(self, out_file, replace_masked=True):
        # replace masked pixels
        if replace_masked:
            self.apply_pixelmask()

        # update headers
        self.update_header()
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE"].data = self._wave
        self._template["LSF"].data = self._lsf
        self._template["SKY_EAST"].data = self._sky_east
        self._template["SKY_EAST_IVAR"].data = numpy.divide(1, self._sky_east_error**2, where=self._sky_east_error != 0, out=numpy.zeros_like(self._sky_east_error))
        self._template["SKY_WEST"].data = self._sky_west
        self._template["SKY_WEST_IVAR"].data = numpy.divide(1, self._sky_west_error**2, where=self._sky_west_error != 0, out=numpy.zeros_like(self._sky_west_error))
        self._template["FLUXCAL_STD"] = pyfits.BinTableHDU(data=self._fluxcal_std, name="FLUXCAL_STD")
        self._template["FLUXCAL_SCI"] = pyfits.BinTableHDU(data=self._fluxcal_sci, name="FLUXCAL_SCI")
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        self._template.verify("silentfix")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self._template[0].header["FILENAME"] = os.path.basename(out_file)
        self._template[0].header['DRPVER'] = drpver
        self._template.writeto(out_file, overwrite=True)


class lvmCFrame(lvmBaseProduct):
    """lvmCFrame class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = cls.header_from_hdulist(hdulist)

        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data.astype("bool")
        wave = hdulist["WAVE"].data
        lsf = hdulist["LSF"].data
        sky_east = hdulist["SKY_EAST"].data
        sky_east_error = numpy.divide(1, hdulist["SKY_EAST_IVAR"].data, where=hdulist["SKY_EAST_IVAR"].data != 0, out=numpy.zeros_like(hdulist["SKY_EAST_IVAR"].data))
        sky_east_error = numpy.sqrt(sky_east_error)
        sky_west = hdulist["SKY_WEST"].data
        sky_west_error = numpy.divide(1, hdulist["SKY_WEST_IVAR"].data, where=hdulist["SKY_WEST_IVAR"].data != 0, out=numpy.zeros_like(hdulist["SKY_WEST_IVAR"].data))
        sky_west_error = numpy.sqrt(sky_west_error)
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave=wave, lsf=lsf,
                   sky_east=sky_east, sky_east_error=sky_east_error,
                   sky_west=sky_west, sky_west_error=sky_west_error,
                   slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, slitmap=None, wave=None, lsf=None,
                 sky_east=None, sky_east_error=None, sky_west=None, sky_west_error=None, **kwargs):
        lvmBaseProduct.__init__(self, data=data, error=error, mask=mask, header=header,
                     wave=wave, lsf=lsf,
                     sky_east=sky_east, sky_east_error=sky_east_error,
                     sky_west=sky_west, sky_west_error=sky_west_error,
                     slitmap=slitmap)

        self._blueprint = dp.load_blueprint(name="lvmCFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        if header is not None:
            self.set_header(header, **kwargs)
        else:
            self._header = None

    def loadFitsData(self, in_file):
        self = lvmCFrame.from_file(in_file)
        return self

    def writeFitsData(self, out_file, replace_masked=True):
        # replace masked pixels
        if replace_masked:
            self.apply_pixelmask()

        # update headers
        self.update_header()
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE"].data = self._wave
        self._template["LSF"].data = self._lsf
        self._template["SKY_EAST"].data = self._sky_east
        self._template["SKY_EAST_IVAR"].data = numpy.divide(1, self._sky_east_error**2, where=self._sky_east_error != 0, out=numpy.zeros_like(self._sky_east_error))
        self._template["SKY_WEST"].data = self._sky_west
        self._template["SKY_WEST_IVAR"].data = numpy.divide(1, self._sky_west_error**2, where=self._sky_west_error != 0, out=numpy.zeros_like(self._sky_west_error))
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        self._template.verify("silentfix")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self._template[0].header["FILENAME"] = os.path.basename(out_file)
        self._template[0].header['DRPVER'] = drpver
        self._template.writeto(out_file, overwrite=True)


class lvmSFrame(lvmBaseProduct):
    """LVM SFrame product class for extracted sky data"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = cls.header_from_hdulist(hdulist)

        data = hdulist["FLUX"].data
        error = numpy.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=numpy.zeros_like(hdulist["IVAR"].data))
        error = numpy.sqrt(error)
        mask = hdulist["MASK"].data.astype("bool")
        wave = hdulist["WAVE"].data
        lsf = hdulist["LSF"].data
        sky = hdulist["SKY"].data
        sky_error = numpy.divide(1, hdulist["SKY_IVAR"].data, where=hdulist["SKY_IVAR"].data != 0, out=numpy.zeros_like(hdulist["SKY_IVAR"].data))
        sky_error = numpy.sqrt(sky_error)
        slitmap = Table(hdulist["SLITMAP"].data)
        return cls(data=data, error=error, mask=mask, header=header,
                   wave=wave, lsf=lsf, sky=sky, sky_error=sky_error, slitmap=slitmap)

    def __init__(self, data=None, error=None, mask=None, header=None, slitmap=None, wave=None, lsf=None, sky=None, sky_error=None, **kwargs):
        lvmBaseProduct.__init__(self, data=data, error=error, mask=mask, header=header,
                     wave=wave, lsf=lsf,
                     sky=sky, sky_error=sky_error, slitmap=slitmap)

        self._blueprint = dp.load_blueprint(name="lvmSFrame")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

        if header is not None:
            self.set_header(header, **kwargs)
        else:
            self._header = None

    def loadFitsData(self, in_file):
        self = lvmSFrame.from_file(in_file)
        return self

    def writeFitsData(self, out_file, replace_masked=True):
        # replace masked pixels
        if replace_masked:
            self.apply_pixelmask()

        # update headers
        self.update_header()
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = numpy.divide(1, self._error**2, where=self._error != 0, out=numpy.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["WAVE"].data = self._wave
        self._template["LSF"].data = self._lsf
        self._template["SKY"].data = self._sky.astype("float32")
        self._template["SKY_IVAR"].data = numpy.divide(1, self._sky_error**2, where=self._sky_error != 0, out=numpy.zeros_like(self._sky_error))
        self._template["SLITMAP"] = pyfits.BinTableHDU(data=self._slitmap, name="SLITMAP")
        self._template.verify("silentfix")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self._template[0].header["FILENAME"] = os.path.basename(out_file)
        self._template[0].header['DRPVER'] = drpver
        self._template.writeto(out_file, overwrite=True)


class lvmRSS(RSS):
    pass