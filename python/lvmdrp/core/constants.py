# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: constants.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import sys


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PYTHON_PATH = os.path.dirname(sys.path[0])
CONFIG_PATH = os.path.join(ROOT_PATH, "python", "lvmdrp", "etc")
MASTER_CONFIG_PATH = os.path.join(CONFIG_PATH, "drp.yml")
SKYMODEL_CONFIG_PATH = os.path.join(CONFIG_PATH, "skymodel_config.yml")
SKYCALC_CONFIG_PATH = os.path.join(CONFIG_PATH, "skycalc.json")
ALMANAC_CONFIG_PATH = os.path.join(CONFIG_PATH, "almanac.json")
SKYCORR_CONFIG_PATH = os.path.join(CONFIG_PATH, "skycorr_config.yml")

SKYMODEL_INST_PATH = os.path.join(PYTHON_PATH, "lib", "skymodel")
SKYCORR_INST_PATH = os.path.join(PYTHON_PATH, "lib", "skycorr")

# dict of dicts containing parameter names for each .par file used in ESO sky model 
SKYMODEL_CONFIG_PARS = {}

SKYCORR_PAR_MAP = {'INPUT_OBJECT_SPECTRUM': 'objfile',
             'INPUT_SKY_SPECTRUM': 'skyfile',
             'OUTPUT_NAME': 'outfile',
             'OUTPUT_DIR': 'outdir',
             'COL_NAMES': 'colnames',
             'INST_DIR': 'install',
             'DEFAULT_ERROR': 'defaultError',
             'WLG_TO_MICRON': 'wave2Micron',
             'VAC_AIR': 'vacOrAir',
             'DATE_KEY': 'dateKey',
             'TIME_KEY': 'timeKey',
             'TELALT_KEY': 'telAltKey',
             'LINETABNAME': 'linetab',
             'VARDATNAME': 'vardat',
             'SOLDATURL': 'soldaturl',
             'SOLFLUX': 'solflux',
             'FWHM': 'fwhm',
             'VARFWHM': 'varfwhm',
             'LTOL': 'ltol',
             'MIN_LINE_DIST': 'minLineDist',
             'FLUXLIM': 'fluxLim',
             'FTOL': 'ftol',
             'XTOL': 'xtol',
             'WTOL': 'wtol',
             'CHEBY_MAX': 'chebyMax',
             'CHEBY_MIN': 'chebyMin',
             'CHEBY_CONST': 'chebyConst',
             'REBINTYPE': 'rebinType',
             'WEIGHTLIM': 'weightLim',
             'SIGLIM': 'sigLim',
             'FITLIM': 'fitLim',
             'PLOT_TYPE': 'plotType'
}

BASIC_CALIBRATION_TYPES = [
    "bias",
    "dark",
    "flat"
]
CALIBRATION_TYPES = BASIC_CALIBRATION_TYPES + [
    "continuum",
    "arc"
]
FRAMES_PRIORITY = CALIBRATION_TYPES + ["object"]
FRAMES_CALIB_NEEDS = {
    "bias": [],
    "dark": ["bias"],
    "flat": ["bias", "dark"],
    "continuum": ["bias", "dark", "flat"],
    "arc": ["bias", "dark", "flat", "continuum"],
    "object": ["bias", "dark", "flat", "continuum", "arc"],
}
