# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 14, 2023
# @Filename: runMethod.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

"""DRP module to prepare and run several types of reductions

The first step is to prepare the configuration file(s) given target frame(s).
The frame(s) can be targeted using a number of parameters:
    * MJD
    * exposure number
    * spectrograph
    * CCD / camera

and for each target frame a configuration file will be written specifying the
corresponding reduction steps. Such steps will depend on the type of frame
(e.g., bias, flat, science), for each of which configuration templates are in
place under lvmdrp/etc. The functions destined to write such configuration
files are prefixed with 'prep', e.g. prepCalib for calibration frames (bias,
dark, flat, arc, etc.) or prepReduction for science, sky, etc.

The second functionality of this module is to actually run the DRP from those
configuration files. For that the 'fromConfig' function is defined.

This module can also be imported and contains the following functions:

    * prepCalib - writes configuration files for calibration frames
    * prepMasterCalib - writes configuration files for master calibration creation
    * prepReduction - writes configuration files for reduction of science frames
    * fromConfig - runs DRP tasks given in a configuration file
    * checkDone - verifies if a DRP run is done given a configuration file
    * dumpScript - writes a bash script for running DRP tasks in a configuration file
    * getConfigs - writes the corresponding configuration file from a given bash script
"""


import io
import os
import re
from copy import deepcopy as copy

import h5py
import yaml
from astropy.io import fits
from tqdm import tqdm
import pandas as pd

import lvmdrp
import lvmdrp.utils.database as db
from lvmdrp.core.constants import CONFIG_PATH, DATAPRODUCT_BP_PATH
from lvmdrp.utils.configuration import load_master_config
from lvmdrp.utils.examples import parse_sdr_name
from lvmdrp.utils.logger import get_logger


description = (
    "provides tasks for writing configuration files and running DRP steps from those"
)

__all__ = [
    "prepCalib_drp",
    "prepMasterCalib_drp",
    "prepReduction_drp",
    "fromConfig_drp",
    "checkDone_drp",
    "dumpScript_drp",
    "getConfigs_drp",
    "metadataCaching_drp",
]

logger = get_logger(__name__)


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


def metadataCaching_drp(path, observatory, mjd, overwrite="0"):

    """caches the header metadata into a HD5 table given a target MJD

    this task will write an HD5 table where the quick data reduction
    information will be available for running the actual reduction.

    Parameters
    ----------
    path : str
        path where the raw frames are stored
    observatory : str
        name of the observatory in the data path structure (e.g., lco, lab)
    mjd : int
        target MJD from which to extract frames metadata
    overwrite : bool
        whether to overwrite the metadata table or not, by default False
    """

    overwrite = bool(int(overwrite))

    metadata_path = os.path.join(path, "metadata.hdf5")

    # remove metadata store if overwrite == True
    if overwrite and os.path.isfile(metadata_path):
        logger.info(f"removing metadata store '{metadata_path}'")
        os.remove(metadata_path)

    # load or create metadata store
    if os.path.isfile(metadata_path):
        logger.info(f"loading metadata store from '{metadata_path}'")
        store = h5py.File(metadata_path, mode="a")
    else:
        logger.info(f"creating metadata store '{metadata_path}'")
        store = h5py.File(metadata_path, mode="w")

    # add observatory group if needed
    if observatory not in store:
        store.create_group(observatory)

    # get existing metadata
    if str(mjd) in store[observatory]:
        metadata_old = pd.DataFrame(store[observatory][str(mjd)][()])
        metadata_old.set_index(["mjd", "camera", "expnum"], inplace=True)
    else:
        metadata_old = pd.DataFrame()

    # filter frames path list
    frames_indices = [
        (os.path.join(root, frame_name), (mjd, *parse_sdr_name(frame_name)))
        for root, _, frame_names in os.walk(path)
        for frame_name in frame_names
        if str(mjd) in root and frame_name.endswith(".fits.gz")
    ]
    ntotal_frames = len(frames_indices)
    # filter out frames in store
    frames_indices = list(
        filter(
            lambda item: tuple(map(lambda s: s.encode("utf-8"), item[1]))
            not in metadata_old.index,
            frames_indices,
        )
    )
    # frames_indices = frames_indices[:119]
    nfilter_frames = len(frames_indices)

    logger.info(
        (
            f"found {ntotal_frames}, skipping {ntotal_frames-nfilter_frames} "
            f"({(ntotal_frames-nfilter_frames)/ntotal_frames*100:g} %) "
            "already present in store"
        )
    )
    if nfilter_frames == 0:
        return

    # initialize table
    metadata = {}
    # extract metadata
    logger.info(f"extracting metadata from {len(frames_indices)} frames")
    iterator = tqdm(
        enumerate(frames_indices),
        total=nfilter_frames,
        desc=f"extracting metadata from MJD = {mjd}",
        ascii=True,
        unit="file",
    )
    for i, (frame_path, (mjd, camera, expnum)) in iterator:
        header = fits.getheader(frame_path, ext=0)
        imagetyp = header.get("FLAVOR", header.get("IMAGETYP"))
        spec = int(camera[-1])
        exptime = header["EXPTIME"]

        metadata[i] = [mjd, camera, expnum, imagetyp, spec, exptime, frame_path]

    # set index
    metadata = pd.DataFrame.from_dict(metadata, orient="index")
    metadata.columns = [
        "mjd",
        "camera",
        "expnum",
        "imagetyp",
        "spec",
        "exptime",
        "path",
    ]
    logger.info("successfully extracted metadata")

    # merge metadata with existing one
    if str(mjd) in store[observatory]:
        logger.info("updating store with new metadata")
        metadata_old.reset_index(inplace=True)
        metadata = pd.concat(
            (metadata_old, metadata), axis="index", join="inner", ignore_index=True
        )
        array = metadata.to_records(index=False)
        dtypes = array.dtype
        array = array.astype(
            [
                (n, dtypes[n])
                if dtypes[n] != object
                else (n, h5py.string_dtype("utf-8", length=None))
                for n in dtypes.names
            ]
        )
        dataset = store[observatory][str(mjd)]
        dataset.resize(dataset.shape[0] + array.shape[0], axis=0)
        dataset[-array.shape[0] :] = array
    else:
        logger.info("adding new data to store")
        array = metadata.to_records(index=False)
        dtypes = array.dtype
        array = array.astype(
            [
                (n, dtypes[n])
                if dtypes[n] != object
                else (n, h5py.string_dtype("utf-8", length=None))
                for n in dtypes.names
            ]
        )
        store[observatory].create_dataset(
            name=str(mjd), data=array, maxshape=(None,), chunks=True
        )

    # write to disk metadata in HDF5 format
    logger.info(f"writing metadata to store '{metadata_path}'")
    store.close()


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
    """writes a configuration file for a calibration frame reduction

    Given a set of parameters to determine the target frame(s) and a
    configuration template, this task will parse and fill in the missing
    fields in the configuration template to produce a configuration file

    Parameters
    ----------
    path : str
        file path where the raw frames are located
    calib_type : str
        frame type corresponding to a calibration frame
    mjd : int, optional
        MJD of the target calibration frame(s), by default None
    exposure : int, optional
        exposure number of the target calibration frame(s), by default None
    spec : str, optional
        spectrograph of the target calibration frame(s) (e.g., spec1), by default None
    ccd : str, optional
        CCD of the target calibration frame(s) (e.g., r1, z3), by default None
    calib_config : str, optional
        configuration file template to use, by default "lvm_{imagetyp}_config"
    """
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
    """writes a configuration file for a master calibration frame creation

    Given a set of parameters to determine the target processed calibration
    frame(s) and a configuration template, this task will parse and fill in the
    missing fields in the configuration template to produce a configuration
    file

    Parameters
    ----------
    path : str
        file path where the raw frames are located
    calib_type : str
        frame type corresponding to a calibration frame
    mjd : int, optional
        MJD of the target calibration frame(s), by default None
    exposure : int, optional
        exposure number of the target calibration frame(s), by default None
    spec : str, optional
        spectrograph of the target calibration frame(s) (e.g., spec1), by default None
    ccd : str, optional
        CCD of the target calibration frame(s) (e.g., r1, z3), by default None
    calib_config : str, optional
        configuration file template to use, by default "lvm_mcalib_config"
    """
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
    """writes to disk configuration file(s) to reduce the target frame(s)

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
    """runs DRP tasks defined in the given configuration file

    Parameters
    ----------
    config : str
        path to the target configuration file
    """
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
