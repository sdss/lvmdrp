# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jun 7, 2022
# @Filename: bismask.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from enum import IntFlag, auto


class ReductionStatus(IntFlag):
    RAW = auto()
    IN_PROGRESS = auto()
    FINISHED = auto()
    FAILED = auto()
    
    def _as_bitmask(self):
        fmt_string = "{:0" + str(len(self.__class__.__members__)) + "b}"
        return fmt_string.format(self.value)
    
    def __str__(self):
        return f"{self.value}"

    def __eq__(self, flag):
        if isinstance(flag, self.__class__):
            return self.value == flag.value
        elif isinstance(flag, str):
            return self.value == self.__class__[flag.upper()].value
        elif isinstance(flag, int):
            return self.value == self.__class__(flag)
        else:
            try:
                return super().__eq__(flag)
            except:
                raise# TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(flag)}'")

    def __add__(self, flag):
        if isinstance(flag, self.__class__):
            return (self & 0) | flag
        elif isinstance(flag, str):
            return (self & 0) | self.__class__[flag.upper()]
        elif isinstance(flag, int):
            return (self & 0) | self.__class__(flag)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(flag)}'")


# BUG: once a bad flag is set, adding an OK flag should produce the same input flag
# OK flag should not be able to change what was wrong before
class QualityFlag(IntFlag):
    # mutually exclusive flag: if OK, OK only, else not OK
    OK = auto()
    # general flags
    MISSING_METADATA = auto()
    BAD_FRAME_SHAPE = auto()
    # during calibration flags
    BAD_CALIBRATION_FRAMES = auto()
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
            return self.value == flag.value
        elif isinstance(flag, str):
            return self.value == self.__class__[flag.upper()].value
        elif isinstance(flag, int):
            return self.value == self.__class__(flag)
        else:
            try:
                return super().__eq__(flag)
            except:
                raise# TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(flag)}'")

    def __add__(self, flag):
        if isinstance(flag, self.__class__):
            if flag.name == "OK" or self.name == "OK":
                return (self & 0) | flag
            else:
                return self | flag
        elif isinstance(flag, str):
            if flag.upper() == "OK" or self.name == "OK":
                return (self & 0) | self.__class__[flag.upper()]
            else:
                return self | self.__class__[flag.upper()]
        elif isinstance(flag, int):
            if flag == 1 or self.value == 1:
                return (self & 0) | self.__class__(flag)
            else:
                return self | self.__class__(flag)
        else:
            try:
                return super().__add__(flag)
            except:
                raise# TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(flag)}'")
    
    def __contains__(self, flag):
        if isinstance(flag, self.__class__):
            return (self & flag) == flag
        elif isinstance(flag, str):
            return (self & self.__class__[flag.upper()]) == self.__class__[flag.upper()]
        elif isinstance(flag, int):
            return (self & self.__class__(flag)) == self.__class__(flag)
        else:
            try:
                return super().__contains__(flag)
            except:
                raise# TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(flag)}'")

# define flag name constants
STATUS = list(ReductionStatus.__members__.keys())
FLAGS = list(QualityFlag.__members__.keys())

if __name__ == "__main__":
    status = ReductionStatus(1)
    print(status.name, status)
    status += "IN_PROGRESS"
    print(status.name, status)
    status += "FINISHED"
    print(status.name, status)
    print(ReductionStatus["FINISHED"] == "FINISHED")

    flag = QualityFlag(1)
    print(flag.name, flag)
    flag = flag + QualityFlag(6)
    print(flag.get_name(), flag)
    print("bad_fibers" in flag, flag.value)
    print("missing_metadata" in flag, flag)
