# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 14, 2023
# @Filename: runMethod.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import io
import os
import re
from copy import deepcopy as copy

import yaml

import lvmdrp
import lvmdrp.utils.database as db
from lvmdrp.core.constants import CONFIG_PATH, DATAPRODUCT_BP_PATH
from lvmdrp.utils.configuration import load_master_config


description = "provides tasks for running the DRP"

__all__ = [
    "prepCalib_drp",
    "prepMasterCalib_drp",
    "prepReduction_drp",
    "fromConfig_drp",
    "checkDone_drp",
    "dumpScript_drp",
    "getConfigs_drp",
]


def _get_missing_fields_in(template):
    """Return config template keywords that need to be filled in

    This function creates an (dictionary) structure to parse fields in the
    given configuration template. Those fields are:

        * location
        * naming_convention

    The structure is as follows:

        {
            location: (template, [kw1, kw2, ...]),
            naming_convention: (template, [kw1, kw2, ...])
        }

    where template is the original value of the field (e.g., location) in the
    configuration template and the list of kw corresponds to the missing values
    to evaluate the corresponding template field.

    This structure can then be used to fill in the missing fields using
    information from other sources such as DBs and/or FITS headers.

    Parameters
    ----------
    template : string or dict_like
        the path to the template file to parse information from or the already
        loaded YAML template.

    Returns
    -------
    missing_fields : dict_like
        a structure to evaluate the missing fields in the given configuration
        template
    """
    if isinstance(template, (list, dict)):
        temp = template
    else:
        temp = yaml.safe_load(open(template, "r"))

    # parse location
    location = temp["location"].replace("[", "{").replace("]", "}")
    location_kws = list(map(str.lower, re.findall(r"\{(\w+)\}", location)))
    # parse naming_convention
    naming = temp["naming_convetions"].replace("[", "{").replace("]", "}")
    naming_kws = list(map(str.lower, re.findall(r"\{(\w+)\}", naming)))

    missing_fields = {
        "location": (location, location_kws),
        "naming_convention": (naming, naming_kws),
    }
    return missing_fields


def _load_template(template_path):
    """Return a configuration template with missing information as formatter string"""
    with open(template_path, "r") as raw_temp:
        lines = [
            line.replace("[", "{").replace("]", "}") for line in raw_temp.readlines()
        ]

    temp = yaml.safe_load(io.StringIO("".join(lines)))
    return temp


def _get_path_from_bp(bp_name):
    """Return dataproduct BP path information"""
    # define BP path
    bp_path = os.path.join(DATAPRODUCT_BP_PATH, f"{bp_name}_bp.yaml")
    # build dataproduct path template
    bp = yaml.safe_load(open(bp_path, "r"))
    loc = os.path.expandvars(bp["location"])
    name = bp["naming_convention"].split(",")[0]
    dataproduct_path = os.path.join(loc, name).replace("[", "{").replace("]", "}")
    kws_path = re.findall(r"\{(\w+)\}", name)

    return dataproduct_path, kws_path


# TODO:
#   * read preproc configuration template
#   * read master calibration configuration template
#   * find target frame(s) in DB
#   * update calibration frame configuration template(s)
#   * write configuration file(s) to disk
def prepCalib_drp(
    path,
    calib_type,
    mjd=None,
    exposure=None,
    spec=None,
    ccd=None,
    calib_config="lvm_{imagetyp}_config",
):
    # expand path
    path = os.path.expandvars(path)

    # get calibration DRP configuration template
    calib_config_path = os.path.join(CONFIG_PATH, f"{calib_config}.yaml")
    calib_config_path = calib_config_path.format(imagetyp=calib_type)
    calib_config_template = _load_template(template_path=calib_config_path)

    # connect to DB
    master_config = load_master_config()
    db.create_or_connect_db(master_config)

    # get target calibration frames from DB
    calib_frames = db.get_raws_metadata_where(
        imagetyp=calib_type, mjd=mjd, exposure=exposure, spec=spec, ccd=ccd
    )
    # create preproc configuration files
    for calib_frame in calib_frames:
        # copy preproc template for modification
        _ = copy(calib_config_template)
        # fill-in location
        _["location"] = _["location"].format(CALIB_PATH=path, DRPVER=lvmdrp.__version__)
        # fill-in naming_convention
        _["naming_convention"] = _["naming_convention"].format(
            IMAGETYP=calib_type, CAMERA=calib_frame.ccd, EXPNUM=calib_frame.expnum
        )
        # fill-in target_frame
        path, path_kws = _get_path_from_bp(_["target_frame"])
        _["target_frame"] = path.format(
            **{key: calib_frame.__dict__[key.lower()] for key in path_kws}
        )
        # fill-in image type
        _["reduction_steps"]["imageMethod.preprocRawFrame"]["out_image"] = _[
            "reduction_steps"
        ]["imageMethod.preprocRawFrame"]["out_image"].format(IMAGETYP=calib_type)

        # fill-in calibration_frames
        master_calib = db.get_master_metadata(metadata=calib_frame)
        if master_calib["bias"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["bias"])
            _["calib_frames"]["bias"] = path.format(
                **{key: master_calib["bias"].__dict__[key.lower()] for key in path_kws}
            )
        if master_calib["dark"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["dark"])
            _["calib_frames"]["dark"] = path.format(
                **{key: master_calib["dark"].__dict__[key.lower()] for key in path_kws}
            )
        if master_calib["pixelflat"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["pixelflat"])
            _["calib_frames"]["pixelflat"] = path.format(
                **{
                    key: master_calib["pixelflat"].__dict__[key.lower()]
                    for key in path_kws
                }
            )
        if master_calib["fiberflat"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["fiberflat"])
            _["calib_frames"]["fiberflat"] = path.format(
                **{
                    key: master_calib["fiberflat"].__dict__[key.lower()]
                    for key in path_kws
                }
            )
        if master_calib["arc"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["arc"])
            _["calib_frames"]["arc"] = path.format(
                **{key: master_calib["arc"].__dict__[key.lower()] for key in path_kws}
            )

        # fill-in reduction steps
        for step in _["reduction_steps"]:
            for par in step:
                if par.startswith("in_") or par.startswith("out_"):
                    if "in_bias" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "bias"
                        ]
                    elif "in_dark" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "dark"
                        ]
                    elif "in_pixelflat" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "pixelflat"
                        ]
                    elif "in_fiberflat" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "fiberflat"
                        ]
                    elif "in_arc" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"]["arc"]
                    else:
                        path, path_kws = _get_path_from_bp(par)
                        _["reduction_steps"][step][par] = path.format(
                            **{
                                key: calib_frame.__dict__[key.lower()]
                                for key in path_kws
                            }
                        )

        # dump calibration configuration file(s)
        yaml.safe_dump(
            _, open(os.path.join(_["location"], _["naming_convention"]), "w")
        )


def prepMasterCalib_drp(
    path,
    calib_type,
    mjd=None,
    exposure=None,
    spec=None,
    ccd=None,
    mcalib_config="lvm_mcalib_config",
):
    # expand path
    path = os.path.expandvars(path)

    # get calibration DRP configuration template
    mcalib_config_path = os.path.join(CONFIG_PATH, f"{mcalib_config}.yaml")
    mcalib_config_template = _load_template(template_path=mcalib_config_path)

    # connect to DB
    master_config = load_master_config()
    db.create_or_connect_db(master_config)

    # get target calibration frames from DB
    calib_frames = db.get_raws_metadata_where(
        imagetyp=calib_type, mjd=mjd, exposure=exposure, spec=spec, ccd=ccd
    )

    # separate non-analog frames in target list
    non_analogs = db.get_analogs_groups(metadata=calib_frames)
    # get analog calibration frames for each target frame
    # conditions to be analog:
    #   * be the same calibration type (bias, dark, etc.)
    #   * be of the same camera/spectrograph
    #   * have the same exposure time if applies
    for non_analog in non_analogs:
        # get analog for each non-analog
        analogs = db.get_analogs_metadata(metadata=non_analog)

        # copy calib template for modification
        _ = copy(mcalib_config_template)
        # fill-in location
        _["location"] = _["location"].format(CALIB_PATH=path, DRPVER=lvmdrp.__version__)
        # fill-in naming_convention
        _["naming_convention"] = _["naming_convention"].format(
            IMAGETYP=calib_type, CAMERA=non_analog.ccd, EXPNUM=non_analog.expnum
        )
        # fill-in target_frame
        path, path_kws = _get_path_from_bp(
            _["target_frame"].format(IMAGETYP=calib_type)
        )
        _["target_frame"] = [
            path.format(**{key: analog.__dict__[key.lower()] for key in path_kws})
            for analog in analogs
        ]
        # fill-in calibration steps
        _["reduction_steps"]["imageMethod.createMasterFrame"]["in_images"] = _[
            "target_frame"
        ]

        path, path_kws = _get_path_from_bp(
            _["reduction_steps"]["imageMethod.createMasterFrame"]["out_image"].format(
                calib_type
            )
        )
        _["reduction_steps"]["imageMethod.createMasterFrame"][
            "out_image"
        ] = path.format(**{key: non_analog.__dict__[key.lower()] for key in path_kws})

    # dump master creation configuration file(s)
    yaml.safe_dump(_, open(os.path.join(_["location"], _["naming_convention"]), "w"))


# TODO: allow for several MJDs
def prepReduction_drp(config_template, mjd=None, exposure=None, spec=None, ccd=None):
    """

    Writes to disk configuration file(s) to perform the reduction of the target
    raw frames

    Steps carried out by this task:
        * read a configuration template
        * find target frame(s) in DB
        * match with calibration frames from DB
        * update configuration template(s)
        * write the filled-in configuration to YAML file(s)

    all parameters (mjd, exposure. spec and ccd) are optional and are used to
    constrain the search for *raw* target frames in the database. Once the
    intended frame(s) is(are) found, this task will locate in the database the
    matching calibration frames needed to carry out the reduction. The i/o file
    paths for each reduction step will be automatically filled-in as well.

    Parameters
    ----------
    config_template : string
        the name of the configuration template to prepare the reduction of the
        target frame(s)
    mjd : int, optional
        the MJD constrain to add to the raw frames list
    exposure : int, optional
        the exposure number to target for reduction
    spec : string of 'spec1', 'spec2' or 'spec3', optional
        the spectrograph to target for reduction
    ccd : string of b1, r1, z1, b2, r2, z2, b3, r3, or z3
        the CCD to target for reduction. Note that setting ccd also constrains
        spec

    """
    # get DRP configuration template
    config_template = _load_template(
        template_path=os.path.join(CONFIG_PATH, f"{config_template}.yaml")
    )

    # connect to DB
    master_config = load_master_config()
    db.create_or_connect_db(master_config)

    target_frames = db.get_raws_metadata_where(
        mjd=mjd, exposure=exposure, spec=spec, ccd=ccd
    )
    for target_frame in target_frames:
        master_calib = db.get_master_metadata(metadata=target_frame)

        # copy configuration template
        _ = copy(config_template)

        # fill-in location
        hemi = "s" if target_frame.OBSERVATORY == "LCO" else "n"
        _["location"] = os.path.expandvars(
            _["location"].format(HEMI=hemi, DRPVER=lvmdrp.__version__)
        )
        # fill-in naming_convention
        _["naming_convention"] = _["naming_convention"].format(
            HEMI=hemi, CAMERA=target_frame.ccd, EXPNUM=target_frame.expnum
        )
        # fill-in target_frame
        path, path_kws = _get_path_from_bp(_["target_frame"])
        _["target_frame"] = path.format(
            **{key: target_frame.__dict__[key.lower()] for key in path_kws}
        )

        # fill-in calibration_frames
        if master_calib["bias"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["bias"])
            _["calib_frames"]["bias"] = path.format(
                **{key: master_calib["bias"].__dict__[key.lower()] for key in path_kws}
            )
        if master_calib["dark"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["dark"])
            _["calib_frames"]["dark"] = path.format(
                **{key: master_calib["dark"].__dict__[key.lower()] for key in path_kws}
            )
        if master_calib["pixelflat"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["pixelflat"])
            _["calib_frames"]["pixelflat"] = path.format(
                **{
                    key: master_calib["pixelflat"].__dict__[key.lower()]
                    for key in path_kws
                }
            )
        if master_calib["fiberflat"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["fiberflat"])
            _["calib_frames"]["fiberflat"] = path.format(
                **{
                    key: master_calib["fiberflat"].__dict__[key.lower()]
                    for key in path_kws
                }
            )
        if master_calib["arc"] is not None:
            path, path_kws = _get_path_from_bp(_["calib_frames"]["arc"])
            _["calib_frames"]["arc"] = path.format(
                **{key: master_calib["arc"].__dict__[key.lower()] for key in path_kws}
            )

        # fill-in reduction steps
        for step in _["reduction_steps"]:
            for par in step:
                if par.startswith("in_") or par.startswith("out_"):
                    if "in_bias" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "bias"
                        ]
                    elif "in_dark" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "dark"
                        ]
                    elif "in_pixelflat" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "pixelflat"
                        ]
                    elif "in_fiberflat" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"][
                            "fiberflat"
                        ]
                    elif "in_arc" == par:
                        _["reduction_steps"][step][par] = _["calibration_frames"]["arc"]
                    else:
                        path, path_kws = _get_path_from_bp(par)
                        _["reduction_steps"][step][par] = path.format(
                            **{
                                key: target_frame.__dict__[key.lower()]
                                for key in path_kws
                            }
                        )

        # dump reduction configuration file
        yaml.safe_dump(
            _, open(os.path.join(_["location"], _["naming_convention"]), "w")
        )


# TODO: define fromConfig_drp(config, **registered_modules):
#   * read config
#   * parse each DRP step in config
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
        # TODO: add records to corresponding tables in DB


# TODO: define task for verifying completness of the run from a configuration script
def checkDone_drp(config):
    pass


# TODO: define task for dumping script from configs
def dumpScript_drp(configs, **registered_modules):
    """

    Print to screen a commandline script version of the given configuration file

    """
    pass


# TODO: define task for recovering configs from script
def getConfigs_drp(script, **registered_modules):
    pass
