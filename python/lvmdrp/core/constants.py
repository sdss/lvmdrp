# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: constants.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os

from lvmdrp.utils import get_env_lib_directory


# sources server URL
LVM_UNAM_URL = "http://ifs.astroscu.unam.mx/LVM"
LVM_SRC_URL = f"{LVM_UNAM_URL}/lvmdrp_src.zip"

# installation path
INS_PATH = os.getenv("LVM_ESOSKY_DIR", get_env_lib_directory())

LIB_PATH = os.path.join(INS_PATH, "lib")
BIN_PATH = os.path.join(INS_PATH, "bin")
INC_PATH = os.path.join(INS_PATH, "include")

# sources path
SRC_PATH = os.path.abspath("src")
SKYCORR_SRC_PATH = os.path.join(SRC_PATH, "skycorr.tar.gz")
SKYMODEL_SRC_PATH = os.path.join(SRC_PATH, "SM-01.tar.gz")

# installation path for ESO routines
SKYCORR_INST_PATH = os.path.join(LIB_PATH, "skycorr")
SKYMODEL_INST_PATH = os.path.join(LIB_PATH, "skymodel")

# root package path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# configuration paths
CONFIG_PATH = os.path.join(ROOT_PATH, "etc")

# dataproduct blueprints path
DATAPRODUCT_BP_PATH = os.path.join(CONFIG_PATH, "dataproducts")

# main DRP configuration path
MASTER_CONFIG_PATH = os.path.join(CONFIG_PATH, "drp_master_config.yaml")

# ESO sky model (web version) configuration files
SKYCALC_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "skycalc.json")
ALMANAC_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "almanac.json")

# ESO sky model master configuration file
SKYMODEL_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "skymodel_config.yml")
# high-level ESO sky model configuration files
SKYMODEL_INST_CONFIG_PATH = os.path.join(
    SKYMODEL_INST_PATH, "sm-01_mod2", "config", "instrument_etc_ref.par"
)
SKYMODEL_MODEL_CONFIG_PATH = os.path.join(
    SKYMODEL_INST_PATH, "sm-01_mod2", "config", "skymodel_etc_ref.par"
)

# ESO skycorr configuration file
SKYCORR_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "skycorr_config.yml")

# ephemeris path
EPHEMERIS_DIR = os.path.join(os.getenv("LVMCORE_DIR"), "etc")

# fiducial calibrations directory
MASTERS_DIR = os.getenv("LVM_MASTER_DIR")

SKYCORR_PAR_MAP = {
    "INPUT_OBJECT_SPECTRUM": "objfile",
    "INPUT_SKY_SPECTRUM": "skyfile",
    "OUTPUT_NAME": "outfile",
    "OUTPUT_DIR": "outdir",
    "COL_NAMES": "colnames",
    "INST_DIR": "install",
    "DEFAULT_ERROR": "defaultError",
    "WLG_TO_MICRON": "wave2Micron",
    "VAC_AIR": "vacOrAir",
    "DATE_KEY": "dateKey",
    "TIME_KEY": "timeKey",
    "TELALT_KEY": "telAltKey",
    "LINETABNAME": "linetab",
    "VARDATNAME": "vardat",
    "SOLDATURL": "soldaturl",
    "SOLFLUX": "solflux",
    "FWHM": "fwhm",
    "VARFWHM": "varfwhm",
    "LTOL": "ltol",
    "MIN_LINE_DIST": "minLineDist",
    "FLUXLIM": "fluxLim",
    "FTOL": "ftol",
    "XTOL": "xtol",
    "WTOL": "wtol",
    "CHEBY_MAX": "chebyMax",
    "CHEBY_MIN": "chebyMin",
    "CHEBY_CONST": "chebyConst",
    "REBINTYPE": "rebinType",
    "WEIGHTLIM": "weightLim",
    "SIGLIM": "sigLim",
    "FITLIM": "fitLim",
    "PLOT_TYPE": "plotType",
}

FRAMES_CALIB_NEEDS = {
    "bias": [],
    "dark": ["bias"],
    "pixflat": ["bias", "dark"],
    "pixmask": ["bias", "dark", "pixflat"],
    "flat": ["pixmask", "bias", "dark", "pixflat", "trace", "fwhm", "wave", "lsf"],
    "arc": ["pixmask", "bias", "dark", "pixflat", "trace", "fwhm", "wave", "lsf"],
    "object": ["pixmask", "bias", "dark", "pixflat", "fiberflat", "trace", "fwhm", "wave", "lsf"],
}
CALIBRATION_NAMES = {"pixmask", "pixflat", "bias", "trace_guess", "trace", "width", "amp", "model", "wave", "lsf", "fiberflat_dome", "fiberflat_twilight"}
CALIBRATION_MATCH = {
    "trace": ["trace", "width", "model"],
    "wave": ["wave", "lsf"],
    "dome": ["fiberflat_dome"],
    "twilight": ["fiberflat_twilight"]}

CAMERAS = ["b1", "b2", "b3", "r1", "r2", "r3", "z1", "z2", "z3"]

# spectrograph channels as spec
# SPEC_CHANNELS = {"b": (3600, 5930), "r": (5660, 7720), "z": (7470, 9800)}
# spectrograph channels as built, removing regions past the dichroics passbands
SPEC_CHANNELS = {"b": (3600, 5800), "r": (5775, 7570), "z": (7520, 9800)}

ARC_LAMPS = ["NEON", "HGNE", "ARGON", "XENON"]
CON_LAMPS = ["LDLS", "QUARTZ"]

LVM_NBLOCKS = 18
LVM_REFERENCE_COLUMN = 2000
FIDUCIAL_PLATESCALE = 112.36748321030637 # Focal plane platescale in "/mm


# stellar templates
STELLAR_TEMP_PATH = os.path.join(os.getenv("LVM_SANDBOX"), "stellar_models")
