
import os
import re
import yaml
import numpy as np
from copy import deepcopy as copy

from lvmdrp.core.constants import CONFIG_PATH, DATAPRODUCT_BP_PATH
from lvmdrp.core.image import Image, loadImage
from lvmdrp.core.rss import RSS, loadRSS

from lvmdrp.main import load_master_config
import lvmdrp.utils.database as db


description = 'provides tasks for running the DRP'

__all__ = [
    "prepQuick_drp", "prepFull_drp", "fromConfig_drp"
]


def _parse_dataproduct_bp(dataproduct_bp):
    """Return dataproduct BP information"""
    # build dataproduct path template
    bp = yaml.safe_load(open(dataproduct_bp, "r"))
    path = os.path.expandvars(bp["location"])
    name = bp["naming_convetion"].split(",")[0]
    dataproduct_path = os.path.join(path, name).replace("[", "{").replace("]", "}")
    keywords = re.findall(r"\[(\w+)\]", name)

    return dataproduct_path, keywords

# TODO: allow for several MJDs
def prepQuick_drp(mjd=None, exposure=None, spec=None, ccd=None, config_name="lvm_quick_config.yaml"):
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
    
    """
    # get quick DRP configuration template
    quick_config_template = yaml.safe_load(open(os.path.join(CONFIG_PATH, config_name), "r"))

    # connect to DB
    config = load_master_config()
    db.create_or_connect_db(config)

    target_frames = db.get_raws_metadata_where(mjd, exposure, spec, ccd)
    quick_configs = []
    for target_frame in target_frames:
        master_calib = db.get_master_metadata(metadata=target_frame)

        # fill-in config
        _ = copy(quick_config_template)

        # calibration frames
        if master_calib["bias"] is not None:
            io_path, keys = _parse_dataproduct_bp(_["calib_frames"]["bias"])
            _["calib_frames"]["bias"] = io_path.format(**{key: master_calib["bias"].__dict__[key.lower()] for key in keys})
        if master_calib["dark"] is not None:
            io_path, keys = _parse_dataproduct_bp(_["calib_frames"]["dark"])
            _["calib_frames"]["dark"] = io_path.format(**{key: master_calib["dark"].__dict__[key.lower()] for key in keys})
        if master_calib["pixelflat"] is not None:
            io_path, keys = _parse_dataproduct_bp(_["calib_frames"]["pixelflat"])
            _["calib_frames"]["pixelflat"] = io_path.format(**{key: master_calib["pixelflat"].__dict__[key.lower()] for key in keys})
        if master_calib["fiberflat"] is not None:
            io_path, keys = _parse_dataproduct_bp(_["calib_frames"]["fiberflat"])
            _["calib_frames"]["fiberflat"] = io_path.format(**{key: master_calib["fiberflat"].__dict__[key.lower()] for key in keys})
        if master_calib["arc"] is not None:
            io_path, keys = _parse_dataproduct_bp(_["calib_frames"]["arc"])
            _["calib_frames"]["arc"] = io_path.format(**{key: master_calib["arc"].__dict__[key.lower()] for key in keys})

        # DRP steps
        for step in _["reduction_steps"]:
            for par in step:
                if par.startswith("in_") or par.startswith("out_"):
                    if "bias" in par.lower(): _["reduction_steps"][step][par] = _["calibration_frames"]["bias"]
                    elif "dark" in par.lower(): _["reduction_steps"][step][par] = _["calibration_frames"]["dark"]
                    elif "pixelflat" in par.lower(): _["reduction_steps"][step][par] = _["calibration_frames"]["pixelflat"]
                    elif "fiberflat" in par.lower(): _["reduction_steps"][step][par] = _["calibration_frames"]["fiberflat"]
                    elif "arc" in par.lower(): _["reduction_steps"][step][par] = _["calibration_frames"]["arc"]
                    else:
                        io_path, keys = _parse_dataproduct_bp(par)
                        _["reduction_steps"][step][par] = io_path.format(**{key: target_frame.__dict__[key.lower()] for key in keys})

        quick_configs.append(_)
    
    return quick_configs

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