# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jun 7, 2022
# @Filename: bismask.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from copy import copy
from enum import IntFlag, auto
import numpy as np


class classproperty(object):
    """taken from https://bit.ly/3yrErQr"""

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class BaseBitmask(IntFlag):
    def _as_bitmask(self):
        fmt_string = "{:0" + str(len(self.__class__.__members__)) + "b}"
        return fmt_string.format(self.value)

    def get_name(self):
        if self.name is not None:
            return self.name
        else:
            bits = list(map(int, self._as_bitmask()))[::-1]
            names = list(self.__class__.__members__.keys())
            return ",".join([names[i] for i in range(len(names)) if bits[i]])

    def __str__(self):
        return f"{self.value}"

    def __eq__(self, flag):
        if isinstance(flag, self.__class__):
            pass
        elif isinstance(flag, str):
            flag = self.__class__[flag.upper()]
        elif isinstance(flag, int):
            flag = self.__class__(flag)
        else:
            try:
                return super().__eq__(flag)
            except:
                raise

        return self.value == flag.value

    def __ne__(self, flag):
        if isinstance(flag, self.__class__):
            pass
        elif isinstance(flag, str):
            flag = self.__class__[flag.upper()]
        elif isinstance(flag, int):
            flag = self.__class__(flag)
        else:
            try:
                return super().__ne__(flag)
            except:
                raise

        return self.value != flag.value

    def __add__(self, flag):
        if isinstance(flag, self.__class__):
            pass
        elif isinstance(flag, str):
            flag = self.__class__[flag.upper()]
        elif isinstance(flag, int):
            flag = self.__class__(flag)
        else:
            try:
                return super().__add__(flag)
            except:
                raise

        return self | flag

    def __contains__(self, flag):
        if isinstance(flag, self.__class__):
            pass
        elif isinstance(flag, str):
            flag = self.__class__[flag.upper()]
        elif isinstance(flag, int):
            flag = self.__class__(flag)
        else:
            try:
                return super().__contains__(flag)
            except:
                raise

        return (self & flag) == flag


class RawFrameQuality(BaseBitmask):
    # TODO: repurpose this to use Dmitry's QC flags
    GOOD = auto()  # bit whether a raw frame is good for reduction
    TEST = auto()  # bit whether a raw frame is for instrument testing purposes
    BAD = auto()  # bit whether a raw frame is bad for reduction


class ReductionStatus(BaseBitmask):
    # mutually exclusive bits
    IN_PROGRESS = auto()  # bit whether a reduction is in progress
    FINISHED = auto()  # bit whether a reduction has succesfully finished
    FAILED = auto()  # bit whether a reduction has failed

    @classproperty
    def MUTUALLY_EXCLUSIVE_BITS(cls):
        return ("IN_PROGRESS", "FINISHED", "FAILED")

    def __add__(self, flag):
        if isinstance(flag, self.__class__):
            pass
        elif isinstance(flag, str):
            flag = self.__class__[flag.upper()]
        elif isinstance(flag, int):
            flag = self.__class__(flag)
        else:
            try:
                return super().__add__(flag)
            except:
                raise

        new = copy(self)
        flag_exclusive = set(flag.get_name().split(",")).intersection(
            self.MUTUALLY_EXCLUSIVE_BITS
        )
        if flag_exclusive:
            to_remove = set(self.MUTUALLY_EXCLUSIVE_BITS).difference(flag_exclusive)
            for bit in to_remove:
                bit = self.__class__[bit]
                if bit in self:
                    new = self ^ bit
        return new | flag


class ReductionStage(BaseBitmask):
    # completed reduction steps
    UNREDUCED = auto()                # exposure in raw stage
    HDRFIX_APPLIED = auto()           # header fix applied
    OVERSCAN_SUBTRACTED = auto()      # trimmed overscan region, overscan-subtracted
    PIXELMASK_ADDED = auto()          # fiducial pixel mask was added
    PREPROCESSED = auto()
    GAIN_CORRECTED = auto()           # gain correction applied
    POISSON_ERROR_CALCULATED = auto() # calculated poisson error
    LINEARITY_CORRECTED = auto()      # linearity correction applied
    DETRENDED = auto()                # bias subtracted and pixel flat-fielded
    COSMIC_CLEANED = auto()           # cosmic ray cleaned
    SCI_ASTROMETRY_ADDED = auto()     # added astrometric solution for science telescope
    SPEC_ASTROMETRY_ADDED = auto()    # added astrometric solution for spectrophotometric telescope
    SKYE_ASTROMETRY_ADDED = auto()    # added astrometric solution for sky east telescope
    SKYW_ASTROMETRY_ADDED = auto()    # added astrometric solution for sky west telescope
    FIBERS_ASTROMETRY_ADDED = auto()  # added astrometric solution for all fibers in the system
    STRAYLIGHT_SUBTRACTED = auto()    # stray light subtracted
    FIBERS_SHIFTED = auto()           # fibers positions corrected for thermal shifts
    SPECTRA_EXTRACTED =  auto()       # extracted spectra
    SPECTROGRAPH_STACKED = auto()     # stacked spectrograph wise
    WAVELENGTH_CALIBRATED = auto()    # arc fiber wavelength solution found
    FLATFIELDED = auto()              # fiber flat-fielded
    WAVELENGTH_SHIFTED = auto()       # wavelength corrected for thermal shifts
    SKY_SUPERSAMPLED = auto()         # extrapolated sky fibers along slit
    SKY_TELESCOPES_COMBINED = auto()  # telescope-combined extrapolated sky fibers
    WAVELENGTH_RECTIFIED = auto()     # wavelength resampled to common grid along slit
    MEASURED_SENS_STD = auto()        # measured sensitivity curve using standard stars
    MEASURED_SENS_SCI = auto()        # scaled fiducial sensitivity using science field stars
    FLUX_CALIBRATED = auto()          # flux calibrated all fibers in the system
    CHANNEL_COMBINED = auto()         # channel stitched together
    SKY_SUBTRACTED = auto()           # sky-subtracted all fibers in the system


    def __add__(self, flag):
        if isinstance(flag, self.__class__):
            pass
        elif isinstance(flag, str):
            flag = self.__class__[flag.upper()]
        elif isinstance(flag, int):
            flag = self.__class__(flag)
        else:
            try:
                return super().__add__(flag)
            except:
                raise

        new = copy(self)
        new = self ^ self.__class__["UNREDUCED"]
        return new | flag

class QualityFlag(BaseBitmask):
    # TODO: add flag for overscan quality
    OSFEATURES = auto()  # Overscan region has features.
    EXTRACTBAD = auto()  # Many bad values in extracted frame.
    EXTRACTBRIGHT = auto()  # Extracted spectra abnormally bright.
    LOWEXPTIME = auto()  # Exposure time less than 10 minutes.
    BADIFU = auto()  # One or more IFUs missing/bad in this frame.
    HIGHSCAT = auto()  # High scattered light levels.
    SCATFAIL = auto()  # Failure to correct high scattered light levels.
    BADDITHER = auto()  # Bad dither location information.
    ARCFOCUS = auto()  # Bad focus on arc frames.
    RAMPAGINGBUNNY = auto()  # Rampaging dust bunnies in IFU flats.
    SKYSUBBAD = auto()  # Bad sky subtraction.
    SKYSUBFAIL = auto()  # Failed sky subtraction.
    FULLCLOUD = auto()  # Completely cloudy exposure.
    BADFLEXURE = auto()  # Abnormally high flexure LSF correction.
    BGROOVEFAIL = auto()  # Possible B-groove glue failure.
    RGROOVEFAIL = auto()  # Possible R-groove glue failure.
    NOGUIDER = auto()  # No guider data available.
    NOSPEC1 = auto()  # No data from spec1.
    NOSPEC2 = auto()  # No data from spec2.
    NOSPEC3 = auto()  # No data from spec3.
    BLOWTORCH = auto()  # Blowtorch artifact detected.
    SEVEREBT = auto()  # Severe blowtorch artifact.
    SATURATED = auto()  # X% of saturated pixels in this frame.


class PixMask(BaseBitmask):
    # pixel bitmasks ------------------------------------------------------
    # from pixelmasks
    BADPIX = auto()
    NEARBADPIXEL = auto()

    # from raw data preprocessing
    SATURATED = auto()

    # from CR rejection
    COSMIC = auto()

    # outlying in pixelflat? possible dust spec
    LOWFLAT = auto()

    # clipped straylight polynomial fit
    STRAYLIGHT = auto()

    CROSSTALK = auto()

    # missing sky lines?
    NOSKY = auto()
    # too bright sky lines?
    BRIGHTSKY = auto()

    # pixels with sensitivities too deviant from instrumental sensitivity
    BADFLUXFACTOR = auto()

    # large sky residuals
    BADSKYFIT = auto()

    # fiber bitmasks ------------------------------------------------------
    NONEXPOSED = auto()
    WEAKFIBER = auto()
    DEADFIBER = auto()
    INTERPOLATED = auto()

    # measure quality of tracing using polynomial fit - samples residuals
    FAILEDPOLY = auto()
    FAILEDSPLINE = auto()
    FAILEDINTERP = auto()
    BADTRACE = auto()
    BADARC = auto()

    # measure offset of the fiber from the median flatfielded fiber
    BADFLAT = auto()

    # offset from a preset fiber shift value
    LARGESHIFT = auto()

    BADSTDFIBER = auto()
    BADSKYFIBER = auto()

    # pixels with no useful information
    NODATA = auto()

    # TODO: bright pixels on top and bottom edges

    # set this if X% close to saturation level
    SATURATION = auto()


# define flag name constants
# RAW_QUALITIES = list(RawFrameQuality.__members__.keys())
STAGES = list(ReductionStage.__members__.keys())
FLAGS = list(QualityFlag.__members__.keys())
DRPQUALITIES = list()


def _parse_bitmask(bitmask, kind):
    if isinstance(bitmask, kind):
        return bitmask
    elif isinstance(bitmask, str):
        bitmask = kind[bitmask]
    elif isinstance(bitmask, int):
        bitmask = kind(bitmask)
    else:
        raise ValueError(f"Wrong type for {bitmask = }: {type(bitmask)}; expected {kind}, string or integer")

    return bitmask


def _parse_where(where, mask_shape):
    if where is not None:
        assert isinstance(where, np.ndarray), f"Wrong type for `where` {type(where)}, expected `numpy.ndarray`"
        assert where.shape == mask_shape, f"Wrong `where` shape {where.shape} not matching `mask_image` shape {mask_shape}"
        assert isinstance(where[0,0], np.bool_), f"Wrong `where` Numpy dtype {type(where[0,0])}, expected a boolean array"
    else:
        where = np.ones(mask_shape, dtype=bool)

    return where


def add_bitmask(mask_image, pixmask, where=None):
    pixmask = _parse_bitmask(pixmask, kind=PixMask)
    where = _parse_where(where, mask_shape=mask_image.shape)

    mask_image[where] |= pixmask
    return mask_image


def toggle_bitmask(mask_image, pixmask, where=None):
    pixmask = _parse_bitmask(pixmask, kind=PixMask)
    where = _parse_where(where, mask_shape=mask_image.shape)

    mask_image[where] ^= pixmask
    return mask_image


def update_header_bitmask(header, kind, bitmask, key, comment):
    bitmask = _parse_bitmask(bitmask, kind=kind)
    if key not in header:
        header[key] = (bitmask.value, comment)
    header[key] |= bitmask.value

    return header


def print_bitmasks(mask_array, logger=None):
    uniques, counts = np.unique(mask_array, return_counts=True)
    bitmasks = dict(zip(map(lambda p: PixMask(int(p)).get_name() if p>0 else "GOODPIX", uniques), counts))
    if logger:
        logger.info(f"{bitmasks}")
        return
    print(bitmasks)
