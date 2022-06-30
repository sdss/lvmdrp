# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jun 7, 2022
# @Filename: bismask.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from enum import IntFlag, auto
from copy import copy


class classproperty(object):
    """taken from https://bit.ly/3yrErQr"""
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

# TODO:
#   - add flag for each step in the reduction, for example: "calib", "cosmic", "stray", etc.
class ReductionStatus(IntFlag):
    # mutually exclusive bits
    RAW = auto()
    IN_PROGRESS = auto()
    FINISHED = auto()
    FAILED = auto()
    # completed reduction steps
    PREPROCESSED = auto()
    CALIBRATED = auto()
    COSMIC_CLEAN = auto()
    STRAY_CLEAN = auto()
    FIBERS_FOUND = auto()
    FIBERS_TRACED = auto()
    SPECTRA_EXTRACTED = auto()
    WAVELENGTH_SOLVED = auto()
    WAVELENGTH_RESAMPLED = auto()

    @classproperty
    def MUTUALLY_EXCLUSIVE_BITS(cls):
        return ("RAW", "IN_PROGRESS", "FINISHED", "FAILED")
    
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

        new = copy(self)
        flag_exclusive = set(flag.get_name().split(",")).intersection(self.MUTUALLY_EXCLUSIVE_BITS)
        if flag_exclusive:
            to_remove = set(self.MUTUALLY_EXCLUSIVE_BITS).difference(flag_exclusive)
            for bit in to_remove:
                bit = self.__class__[bit]
                if bit in self: new = self ^ bit
        return new | flag

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

class QualityFlag(IntFlag):
    # mutually exclusive flag: if OK, OK only, else not OK
    OK = auto()
    # general flags
    MISSING_METADATA = auto()
    BAD_FRAME_SHAPE = auto()
    # during calibration flags
    BAD_CALIBRATION_FRAMES = auto()
    POORLY_DEFINED_MASTER = auto()
    BAD_FIBERS = auto()
    HAS_STRUCTURE = auto()
    BAD_EXTRACTION = auto()
    BAD_WAVELENGTH = auto()
    # products quality flags
    LOW_SNR = auto()

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

        ok_bit = self.__class__.OK
        if flag.name != ok_bit and self == ok_bit:
            new = self & 0
        elif flag.name == ok_bit and self != ok_bit:
            new = copy(self)
            flag = self.__class__(0)
        else:
            new = copy(self)
        return new | flag

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

# define flag name constants
STATUS = list(ReductionStatus.__members__.keys())
FLAGS = list(QualityFlag.__members__.keys())

if __name__ == "__main__":
    status = ReductionStatus.RAW
    print(status.get_name())
    status += "IN_PROGRESS"
    print(status.get_name())
    status += ReductionStatus.PREPROCESSED
    print(status.get_name())
    status += ReductionStatus.CALIBRATED
    print(status.get_name())
    status += ReductionStatus.FINISHED
    print(status.get_name())
    status += ReductionStatus.IN_PROGRESS
    print(status.get_name())
    status += ReductionStatus.FIBERS_FOUND
    print(status.get_name())
    status += ReductionStatus.FINISHED
    print(status.get_name())
    print("finished" in status)
    status = ReductionStatus.IN_PROGRESS
    print(status.get_name())
    status += ReductionStatus.PREPROCESSED|ReductionStatus.CALIBRATED
    print(status.get_name())

    flag = QualityFlag(0)
    print(flag.get_name())
    flag = QualityFlag.OK
    print(flag.get_name())
    flag += QualityFlag.OK
    print(flag.get_name())
    flag += QualityFlag.BAD_FIBERS
    print(flag.get_name())
    flag += QualityFlag.MISSING_METADATA
    print(flag.get_name())
    print("bad_fibers" in flag)
    print("missing_metadata" in flag)
    flag += QualityFlag.OK
    print(flag.get_name())
    flag = QualityFlag.OK
    print(flag.get_name())
    flag += QualityFlag.BAD_CALIBRATION_FRAMES|QualityFlag.BAD_EXTRACTION
    print(flag.get_name())
