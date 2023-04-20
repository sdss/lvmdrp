# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: database.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os

import h5py
import numpy as np
import pandas as pd
from astropy.table import Table

from lvmdrp.core.constants import CALIBRATION_TYPES, FRAMES_CALIB_NEEDS
from lvmdrp.utils.bitmask import QualityFlag, ReductionStage, ReductionStatus
from lvmdrp.utils.logger import get_logger

# NOTE: replace these lines with Brian's integration of sdss_access and sdss_tree
from sdss_access import Access
from sdss_access.path import Path


path = Path(release="sdss5")
access = Access(release="sdss5")
access.set_base_dir()
# -------------------------------------------------------------------------------


logger = get_logger(__name__)


def _load_or_create_store(observatory, overwrite=False):
    """return the metadata store given a path

    Parameters
    ----------
    observatory: str
        name of the observatory from which data will be retrieved/cached
    overwrite: bool, optional
        whether to overwrite the store or not, by default False

    Returns
    -------
    h5py.Group
        store found in the given path
    """
    metadata_path = os.path.join(access.base_dir, "metadata.hdf5")

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
        store.create_group(f"{observatory}/raw")
        store.create_group(f"{observatory}/master")

    return store[observatory]


def record_db(config, target_paths=None, ignore_cache=False):
    pass


def get_metadata(
    observatory="lco",
    return_masters=False,
    imagetyp=None,
    mjd=None,
    expnum=None,
    spec=None,
    camera=None,
    stage=None,
    status=None,
    quality=None,
):
    """return raw frames metadata from precached HDF5 store

    Parameters
    ----------
    observatory : str
        observatory where the data was observed
    return_masters : bool
        whether the target frames are master calibration frames or not, by default False
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    mjd : int, optional
        MJD where the target frames is located, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None
    """

    store = _load_or_create_store(observatory=observatory)

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(store[f"raw/{mjd}"][()])
    else:
        metadata = pd.DataFrame()
        for mjd in store.keys():
            metadata.append(store[mjd])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))

    # retreive master calibration frames if requested
    if return_masters:
        try:
            master_metadata = pd.DataFrame(store[f"master/{mjd}"][()])
        except KeyError:
            master_metadata = pd.DataFrame()

    # close store
    store.file.close()

    # convert bytes to literal strings
    metadata_str = metadata.select_dtypes(object).apply(
        lambda s: s.str.decode("utf-8"), axis="columns"
    )
    metadata[metadata_str.columns] = metadata_str

    master_str = master_metadata.select_dtypes(object).apply(
        lambda s: s.str.decode("utf-8"), axis="columns"
    )
    master_metadata[master_str.columns] = master_str
    del metadata_str, master_str

    logger.info(f"found {len(metadata)} frames in store")

    # filter by exposure number, spectrograph and/or camera
    query = []
    if imagetyp is not None:
        logger.info(f"filtering by {imagetyp = }")
        query.append("imagetyp == @imagetyp")
    if spec is not None:
        logger.info(f"filtering by {spec = }")
        query.append("spec == @spec")
    if expnum is not None:
        logger.info(f"filtering by {expnum = }")
        query.append("expnum == @expnum")
    if camera is not None:
        logger.info(f"filtering by {camera = }")
        query.append("camera == @camera")
    if stage is not None:
        logger.info(f"filtering by {stage = }")
        query.append("stage == @stage")
    if status is not None:
        logger.info(f"filtering by {status = }")
        query.append("status == @status")
    if quality is not None:
        logger.info(f"filtering by {quality = }")
        query.append("quality == @quality")

    if query:
        query = " and ".join(query)
        metadata.query(query, inplace=True)

    logger.info(f"final number of frames after filtering {len(metadata)}")

    if return_masters:
        return metadata, master_metadata
    return metadata


def get_nonanalogs_groups(metadata_list):
    """Return filtered metadata list filtered by analog attributes: imagetyp, mjd, camera and exptime"""
    # build metadata table containing columns relevant for analog selection: imagetyp, mjd, camera, exptime
    metadata_table = Table(data=[metadata.__dict__ for metadata in metadata_list])
    metadata_group = metadata_table.group_by(["imagetyp", "mjd", "camera", "exptime"])
    # return filtered list containing non-analog frames metadata for which to find analogs
    return [metadata_list[idx] for idx in metadata_group.groups.indices]


def get_analogs_metadata(target_metadata, calib_metadata):
    """return analog frames given a target frame metadata

    This function will match a target calibration frame metadata `target_metadata`
    against a list of calibration frames metadata to find its analogs in order to
    build a master calibration frame. The criteria used to match calibration frames
    are:
        * reduction stage has to be `CALIBRATED`
        * quality_flag == 0
        * imagetyp, camera, mjd and exptime must be the same

    Parameters
    ----------
    target_metadata : pandas.Series
        the metadata of the target calibration frame
    calib_metadata : pandas.DataFrame
        the metadata of the calibration frames from which to pick analogs

    Returns
    -------
    pandas.DataFrame
        the analog frames metadata
    """
    # define empty metadata in case current frame has already a master
    analogs_metadata = []
    if target_metadata.imagetyp in CALIBRATION_TYPES:
        # define query in calib_metadata to match target frame metadata
        bmask = ReductionStage.PREPROCESSED | ReductionStage.CALIBRATED
        q = (
            "@calib_metadata.stage == @bmask "
            "and @calib_metadata.quality_flag == 0 "
            "and @calib_metadata.imagetyp == @target_metadata.imagetyp "
            "and @calib_metadata.camera == @target_metadata.camera "
            "and @calib_metadata.mjd == @target_metadata.mjd "
            "and @calib_metadata.exptime == @target_metadata.exptime"
        )
    else:
        # no analogs needed for this type of target frame
        logger.info(
            (
                f"target frame of type '{target_metadata.imagetyp}' "
                "does not need calibration frames"
            )
        )
        return analogs_metadata

    # filter calibration metadata find matching analogs
    analogs_metadata = calib_metadata.query(q)
    logger.info(f"found {len(analogs_metadata)}:")
    logger.info(f"{analogs_metadata.to_string()}")

    return analogs_metadata


def get_master_metadata(target_metadata, masters_metadata):
    """return the matched master calibration frames given a target frame

    Depending on the type of the target frame, a set of calibration frames may
    be needed. These are stored in lvmdrp.core.constants.FRAMES_CALIB_NEEDS.
    This function retrieves the closest in time set of calibration frames
    according to that mapping.

    Parameters
    ----------
    target_metadata : pandas.Series
        the target frame metadata
    masters_metadata : pandas.DataFrame
        the master calibration frames metadata to pick from

    Returns
    -------
    dict_like
        a dictionary containing the matched master calibration frames
    """
    # retrieve calibration needs
    frame_needs = FRAMES_CALIB_NEEDS.get(target_metadata.imagetyp)
    logger.info(
        (
            f"target frame of type '{target_metadata.imagetyp}' "
            f"needs calibration frames: {', '.join(frame_needs) or None}"
        )
    )
    # raise error in case current frame is not recognized in FRAMES_CALIB_NEEDS
    if frame_needs is None:
        logger.error(
            f"no calibration frames found for '{target_metadata.imagetyp}' type"
        )

    calib_frames = dict.fromkeys(CALIBRATION_TYPES)
    # handle empty list cases (e.g., bias)
    if not frame_needs:
        return calib_frames
    # handle unrecognized frame type

    for calib_type in frame_needs:
        bmask = ReductionStage.PREPROCESSED | ReductionStage.CALIBRATED
        if calib_type in ["flat", "arc"]:
            bmask += (
                ReductionStage.COSMIC_CLEAN
                | ReductionStage.STRAY_CLEAN
                | ReductionStage.FIBERS_FOUND
                | ReductionStage.FIBERS_TRACED
                | ReductionStage.SPECTRA_EXTRACTED
                | ReductionStage.WAVELENGTH_SOLVED
            )
        q = (
            "@masters_metadata.stage == @bmask "
            "and @masters_metadata.quality_flag == 0 "
            "and @masters_metadata.imagetyp == @calib_type "
            "and @masters_metadata.camera == @target_metadata.camera "
            "and @masters_metadata.calib_id is not None"
        )

        # TODO: handle the case in which the retrieved frame is stale and/or has quality
        # flags
        # BUG: there may be cases in which no frame is found
        # BUG: this is retrieving only the first (closest) calibration frame, not
        #      necessarily the best. Should retrieve all possible calibration frames
        #      & decide which one is the best based on quality
        calib_frame = masters_metadata.query(q)
        calib_frame["mjd_diff"] = calib_frame.mjd.apply(
            lambda v: abs(v - target_metadata.mjd)
        )
        calib_frame = (
            calib_frame.sort_values(by="mjd_diff", ascending=True)
            .drop(columns="mjd_diff")
            .iloc[0]
        )
        if len(calib_frame) == 0:
            logger.error(f"no master {calib_type} frame found")
        else:
            logger.info(f"found master {calib_type}")
            calib_frames[calib_type] = calib_frame
    return calib_frames


def put_redux_stage(metadata, stage=None):
    if stage is not None:
        if isinstance(stage, str):
            metadata.status += ReductionStage[stage]
        elif isinstance(stage, int):
            metadata.status += ReductionStage(stage)
        elif isinstance(stage, ReductionStage):
            metadata.status += stage
        else:
            ValueError(f"unknown status type '{type(stage)}'")
    try:
        if isinstance(metadata, (LVMFrames, CalibrationFrames)):
            if "IN_PROGRESS" in metadata.status:
                metadata.reduction_started = dt.datetime.now()
            elif "FINISHED" in metadata.status or "FAILED" in metadata.status:
                metadata.reduction_finished = dt.datetime.now()
            metadata.save()
        elif isinstance(metadata, list):
            for md in metadata:
                if "IN_PROGRESS" in md.status:
                    md.reduction_started = dt.datetime.now()
                elif "FINISHED" in md.status or "FAILED" in md.status:
                    md.reduction_finished = dt.datetime.now()
                md.save()
        else:
            raise ValueError(f"unknown metadata type '{type(metadata)}'")
    except Error as e:
        print(e)
    return metadata


def add_calib(calib_metadata, raw_metadata, stage=None):
    if stage is not None:
        if isinstance(stage, str):
            calib_metadata.status += ReductionStage[stage]
        elif isinstance(stage, int):
            calib_metadata.status += ReductionStage(stage)
        elif isinstance(stage, ReductionStage):
            calib_metadata.status += stage
        else:
            ValueError(f"unknown status type '{type(stage)}'")

    if calib_metadata.status == "IN_PROGRESS":
        calib_metadata.reduction_started = dt.datetime.now()
    elif "FINISHED" in calib_metadata.status or "FAILED" in calib_metadata.status:
        calib_metadata.reduction_finished = dt.datetime.now()
    try:
        calib_metadata.save()
        raw_metadata.calib_id = calib_metadata.id
        raw_metadata.save()
    except Error as e:
        print(e)
        print(calib_metadata)
    return calib_metadata


def add_master(master_metadata, analogs_metadata, stage=None):
    if stage is not None:
        if isinstance(stage, str):
            master_metadata.status += ReductionStage[stage]
        elif isinstance(stage, int):
            master_metadata.status += ReductionStage(stage)
        elif isinstance(stage, ReductionStage):
            master_metadata.status += stage
        else:
            ValueError(f"unknown status type '{type(stage)}'")

    if master_metadata.status == "IN_PROGRESS":
        master_metadata.reduction_started = dt.datetime.now()
    elif "FINISHED" in master_metadata.status or "FAILED" in master_metadata.status:
        master_metadata.reduction_finished = dt.datetime.now()
    try:
        master_metadata.save()
        for analog_metadata in analogs_metadata:
            analog_metadata.calib_id = master_metadata.id
            analog_metadata.update()
    except Error as e:
        print(e)
        print(master_metadata)
    return master_metadata


if __name__ == "__main__":
    from lvmdrp.utils.configuration import load_master_config

    config = load_master_config()
    db = create_or_connect_db(config)

    new_frames = get_raws_metadata()
    for new_frame in new_frames:
        print(new_frame.label, new_frame.flags.get_name())
