# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jun 7, 2022
# @Filename: bismask.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from copy import copy
from enum import IntFlag, auto


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
    UNREDUCED = auto()  # exposure not reduced
    PREPROCESSED = auto()  # trimmed overscan region
    CALIBRATED = auto()  # bias, dark and pixelflat calibrated
    COSMIC_CLEAN = auto()  # cosmic ray cleaned
    STRAY_CLEAN = auto()  # stray light subtracted
    FIBERS_FOUND = auto()  # fiberflat fibers located along the column
    FIBERS_TRACED = auto()  # fiberflat fibers traces along the dispersion axis
    SPECTRA_EXTRACTED = (
        auto()
    )  # extracted the fiber spectra of any arc, flat, or science frames
    WAVELENGTH_SOLVED = auto()  # arc fiber wavelength solution found
    WAVELENGTH_RESAMPLED = (
        auto()
    )  # fiber wavelength resampled to common wavelength/LSF vector


class QualityFlag(BaseBitmask):
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


class PixMask(BaseBitmask):
    # fiber bitmasks
    NOPLUG = auto()
    BADTRACE = auto()
    BADFLAT = auto()
    BADARC = auto()
    MANYBADCOLUMNS = auto()
    MANYREJECTED = auto()
    LARGESHIFT = auto()
    BADSKYFIBER = auto()
    NEARWHOPPER = auto()
    WHOPPER = auto()
    SMEARIMAGE = auto()
    SMEARHIGHSN = auto()
    SMEARMEDSN = auto()
    DEADFIBER = auto()
    # pixel bitmasks
    SATURATION = auto()
    BADPIX = auto()
    COSMIC = auto()
    NEARBADPIXEL = auto()
    LOWFLAT = auto()
    FULLREJECT = auto()
    PARTIALREJECT = auto()
    SCATTEREDLIGHT = auto()
    CROSSTALK = auto()
    NOSKY = auto()
    BRIGHTSKY = auto()
    NODATA = auto()
    COMBINEREJ = auto()
    BADFLUXFACTOR = auto()
    BADSKYCHI = auto()


# define flag name constants
STATUS = list(ReductionStatus.__members__.keys())
STAGE = list(ReductionStage.__members__.keys())
FLAGS = list(QualityFlag.__members__.keys())

if __name__ == "__main__":
    status = ReductionStatus(0)
    stage = ReductionStage.UNREDUCED
    print(status.get_name(), stage.get_name())
    status += "IN_PROGRESS"
    print(status.get_name(), stage.get_name())
    stage += ReductionStage.PREPROCESSED
    print(status.get_name(), stage.get_name())
    stage += ReductionStage.CALIBRATED
    print(status.get_name(), stage.get_name())
    status += ReductionStatus.FINISHED
    print(status.get_name(), stage.get_name())
    status += ReductionStatus.IN_PROGRESS
    print(status.get_name(), stage.get_name())
    stage += ReductionStage.FIBERS_FOUND
    print(status.get_name(), stage.get_name())
    status += ReductionStatus.FINISHED
    print(status.get_name(), stage.get_name())
    print("finished" in status)
    status = ReductionStatus.IN_PROGRESS
    print(status.get_name(), stage.get_name())
    stage += ReductionStage.PREPROCESSED | ReductionStage.CALIBRATED
    print(status.get_name(), stage.get_name())
