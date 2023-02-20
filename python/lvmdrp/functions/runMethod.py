# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 14, 2023
# @Filename: runMethod.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import re
import yaml
import numpy as np
from copy import deepcopy as copy

import lvmdrp
from lvmdrp.core.constants import CONFIG_PATH, DATAPRODUCT_BP_PATH
from lvmdrp.core.image import Image, loadImage
from lvmdrp.core.rss import RSS, loadRSS

from lvmdrp.main import load_master_config
import lvmdrp.utils.database as db


description = 'provides tasks for running the DRP'

__all__ = [
    "prepQuick_drp", "prepFull_drp", "fromConfig_drp"
]


def _get_path_from_bp(bp_path):
    """Return dataproduct BP path information"""
    # build dataproduct path template
    bp = yaml.safe_load(open(bp_path, "r"))
    loc = os.path.expandvars(bp["location"])
    name = bp["naming_convention"].split(",")[0]
    dataproduct_path = os.path.join(loc, name).replace("[", "{").replace("]", "}")
    kws_path = re.findall(r"\{(\w+)\}", name)

    return dataproduct_path, kws_path


    return dataproduct_path, keywords

# TODO: allow for several MJDs
def prepQuick_drp(mjd=None, exposure=None, spec=None, ccd=None, quick_config="lvm_quick_config"):
    """

        Returns a list of configuration to perform the quick DRP
        
        Steps carried out by this task:
            * read a quick configuration template
            * find target frame(s) in DB
            * match with calibration frames from DB
            * update quick configuration template(s)
            * return quick configuration .YAML file(s)
        
        all parameters (mjd, exposure. spec and ccd) are optional and are used to constrain the
        search for *raw* science frames in the database. Once the intended frame(s) is(are) found,
        this task will locate in the database the matching calibration frames needed to carry out
        the quick reduction. The i/o file paths for each reduction step will be automatically
        updated as well.

        Parameters
        ----------
        mjd : int, optional
            the MJD constrain to add to the raw frames list
        exposure : int, optional
            the exposure number to target for quick reduction
        spec : string of 'spec1', 'spec2' or 'spec3', optional
            the spectrograph to target for quick reduction
        ccd : string of b1, r1, z1, b2, r2, z2, b3, r3, or z3
            the CCD to target for quick reduction. Note that setting ccd also constrains spec
        quick_config: string
            the name of the quick configuration template
    
    """
    # get quick DRP configuration template
    quick_config_template = _load_template(template_path=os.path.join(CONFIG_PATH, f"{quick_config}.yaml"))

    # connect to DB
    master_config = load_master_config()
    db.create_or_connect_db(master_config)

    target_frames = db.get_raws_metadata_where(mjd=mjd, exposure=exposure, spec=spec, ccd=ccd)
    for target_frame in target_frames:
        master_calib = db.get_master_metadata(metadata=target_frame)

        # copy quick configuration template
        _ = copy(quick_config_template)

        # fill-in location
        hemi = "s" if target_frame.OBSERVATORY=="LCO" else "n"
        _["location"] = os.path.expandvars(_["location"].format(HEMI=hemi, DRPVER=lvmdrp.__version__))
        # fill-in naming_convention
        _["naming_convention"] = _["naming_convention"].format(
            HEMI=hemi,
            CAMERA=target_frame.ccd,
            EXPNUM=target_frame.expnum
        )
        # fill-in target_frame
        path, path_kws = _get_path_from_bp(_["target_frame"])
        _["target_frame"] = path.format(**{key: target_frame.__dict__[key.lower()] for key in path_kws})

        # fill-in calibration frames
        if master_calib["bias"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["bias"])
            _["calib_frames"]["bias"] = path.format(**{key: master_calib["bias"].__dict__[key.lower()] for key in path_kws})
        if master_calib["dark"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["dark"])
            _["calib_frames"]["dark"] = path.format(**{key: master_calib["dark"].__dict__[key.lower()] for key in path_kws})
        if master_calib["pixelflat"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["pixelflat"])
            _["calib_frames"]["pixelflat"] = path.format(**{key: master_calib["pixelflat"].__dict__[key.lower()] for key in path_kws})
        if master_calib["fiberflat"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["fiberflat"])
            _["calib_frames"]["fiberflat"] = path.format(**{key: master_calib["fiberflat"].__dict__[key.lower()] for key in path_kws})
        if master_calib["arc"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["arc"])
            _["calib_frames"]["arc"] = path.format(**{key: master_calib["arc"].__dict__[key.lower()] for key in path_kws})

        # fill-in reduction steps
        for step in _["reduction_steps"]:
            for par in step:
                if par.startswith("in_") or par.startswith("out_"):
                    if "in_bias" == par: _["reduction_steps"][step][par] = _["calibration_frames"]["bias"]
                    elif "in_dark" == par: _["reduction_steps"][step][par] = _["calibration_frames"]["dark"]
                    elif "in_pixelflat" == par: _["reduction_steps"][step][par] = _["calibration_frames"]["pixelflat"]
                    elif "in_fiberflat" == par: _["reduction_steps"][step][par] = _["calibration_frames"]["fiberflat"]
                    elif "in_arc" == par: _["reduction_steps"][step][par] = _["calibration_frames"]["arc"]
                    else:
                        path, path_kws = _get_path_from_bp(par)
                        _["reduction_steps"][step][par] = path.format(**{key: target_frame.__dict__[key.lower()] for key in path_kws})

        # dump quick reduction configuration file
        yaml.safe_dump(_, open(os.path.join(_["location"], _["naming_convention"]), "w"))

# TODO: define prepFull_drp(spec, channel, exposure, mjd):
#   * read a full configuration template
#   * find target frame(s) in DB
#   * match with calibration frames
#   * update full configuration template(s)
#   * return full configuration .YAML file(s)
def prepFull_drp(spec, channel, exposure, mjd):
    pass

# TODO: define fromConfig_drp(config, **registered_modules):
#   * read config
#   * parse each DRP step in config (match config.steps to each module.step in registered_modules)
#   * run each DRP step
def fromConfig_drp(config, **registered_modules):
    config = yaml.safe_load(config)

    # TODO: show target frame info
    # TODO: show calibration frames info
    
    reduction_steps = config["reduction_steps"]
    for step in reduction_steps:
        module_name, task_name = list(step.keys())[0].split(".")

        if module_name not in registered_modules:
            # TODO: show error message
            # TODO: try to import the module.task
            pass

        task = getattr(registered_modules[module_name], task_name)
        # TODO: show running step info
        task(**reduction_steps[step])