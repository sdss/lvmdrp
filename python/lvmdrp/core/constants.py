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

CALIBRATION_TYPES = [
    "bias",
    "dark",
    "flat"
]
FRAMES_PRIORITY = CALIBRATION_TYPES + [
    "continuum",
    "arc",
    "object"
]
FRAMES_CALIB_NEEDS = {
    "bias": [],
    "dark": ["bias"],
    "flat": ["bias", "dark"],
    "continuum": ["bias", "dark", "flat"],
    "arc": ["bias", "dark", "flat", "continuum"],
    "object": ["bias", "dark", "flat", "continuum", "arc"],
}

# define product path pattern
INPUT_PATH = os.path.join("{input_path}", "{mjd}", "{label}.fits.gz")
PRODUCT_PATH = os.path.join("{path}", "{label}.{kind}.fits")

# TODO:
#   - add frame regions (overscan, prescan, science) to check if those are correct
#   - bring back file path because the data file system is all messed up. Can't construct the correct path from header information in some cases
BASIC_FIELDS = {
    "id": "INT AUTO_INCREMENT PRIMARY KEY",
    "datetime": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    "mjd": "INT",
    "spec": "VARCHAR(20)",
    "ccd": "VARCHAR(4)",
    "exptime": "FLOAT(4)",
    "imagetyp": "VARCHAR(50)",
    "obstime": "DATETIME",
    "observat": "VARCHAR(20)",
    "label": "VARCHAR(250)",
    "path": "VARCHAR(250)",
    "naxis1": "INT",
    "naxis2": "INT"
}
FOREIGN_FIELDS = {
    "master_id": "INT NULL"
}
LAB_FIELDS = {
    "ccdtemp1": "FLOAT(4) NULL",
    "ccdtemp2": "FLOAT(4) NULL",
    "presure": "FLOAT(4) NULL",
    "labtemp": "FLOAT(4) NULL",
    "labhumid": "FLOAT(4) NULL"
}
ARC_FIELDS = {
    "argon": "BOOL",
    "xenon": "BOOL",
    "hgar": "BOOL",
    "krypton": "BOOL",
    "neon": "BOOL",
    "hgne": "BOOL"
}
CONT_FIELDS = {
    "m625l4": "BOOL",
    "ffs": "BOOL",
    "mi150": "BOOL",
    "ts": "BOOL",
    "ldls": "BOOL",
    "nirled": "BOOL"
}
FLAG_FIELDS = {
    "reduction_started": "TIMESTAMP NULL",
    "reduction_finished": "TIMESTAMP NULL",
    "status": "INT",
    "flags": "INT"
}

ALL_DTYPES = {**BASIC_FIELDS, **FOREIGN_FIELDS, **LAB_FIELDS, **ARC_FIELDS, **CONT_FIELDS, **FLAG_FIELDS}
ALL_DEFAULTS = dict.fromkeys(ALL_DTYPES, None)
ALL_DEFAULTS.update(dict.fromkeys(list(ARC_FIELDS.keys())+list(CONT_FIELDS.keys()), False))

RAW_NAMES = list(ALL_DTYPES.keys())
CALIBRATION_NAMES = list(BASIC_FIELDS.keys()) + list(FLAG_FIELDS.keys())
# these are the fields that are automatically set on adding a new record to the DB
FIELDS_TO_SKIP = ["id", "master_id", "datetime"]

# TODO: add table for DRP products, from preprocessed frames to final science-ready frames
# TODO: add weather table for data quality monitoring purposes
CREATE_RAW_FRAMES = f"""
CREATE TABLE RAW_FRAMES({','.join([f"{field} {ALL_DTYPES[field]}" for field in RAW_NAMES])},
FOREIGN KEY(master_id) REFERENCES CALIBRATION_FRAMES(id)
)"""
CREATE_CALIBRATION_FRAMES = f"""CREATE TABLE CALIBRATION_FRAMES({','.join([f"{field} {ALL_DTYPES[field]}" for field in CALIBRATION_NAMES])})"""
