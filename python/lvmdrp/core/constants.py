# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: constants.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_PATH = os.path.join(ROOT_PATH, "python", "lvmdrp", "etc")
MASTER_CONFIG_PATH = os.path.join(CONFIG_PATH, "drp.yml")
SKYCALC_CONFIG_PATH = os.path.join(CONFIG_PATH, "skycalc.json")
ALMANAC_CONFIG_PATH = os.path.join(CONFIG_PATH, "almanac.json")
SKYCORR_CONFIG_PATH = os.path.join(CONFIG_PATH, "skycorr_config.yml")

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
