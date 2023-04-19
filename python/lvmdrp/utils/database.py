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
# -------------------------------------------------------------------------------


logger = get_logger(__name__)


def load_or_create_store(observatory, overwrite=False):
    """return the metadata store given a path

    Parameters
    ----------
    observatory: str
        name of the observatory from which data will be retrieved/cached
    overwrite: bool, optional
        whether to overwrite the store or not, by default False

    Returns
    -------
    h5py.File object
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
        store.create_group(observatory)

    return store


def get_old_metadata(store, mjd, observatory="lco"):
    """return existing metadata from store given an observatory and MJD

    Parameters
    ----------
    store : h5py.File
        store from which the existing dataset will be retrieved
    mjd : int
        MJD of the target dataset
    observatory : str, optional
        name of the observatory, by default 'lco'

    Returns
    -------
    pandas.DataFrame
        existing metadata for the given observatory and MJD
    """
    if str(mjd) in store[observatory]:
        metadata = pd.DataFrame(store[observatory][str(mjd)][()])
        metadata.set_index(["mjd", "ccd", "expnum"], inplace=True)
    else:
        metadata = pd.DataFrame()

    return metadata

def get_raws_metadata(
    path,
    observatory="lco",
    imagetyp=None,
    mjd=None,
    expnum=None,
    spec=None,
    ccd=None,
):
    """return raw frames metadata from precached HDF5 store

    Parameters
    ----------
    path : str
        path where the raw frames are located
    observatory : str
        observatory where the data was observed
    imagetyp : str, optional
        type/flavor of frame to locate`IMAGETYP`, by default None
    mjd : int, optional
        MJD where the target frames is located, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    ccd : str, optional
        CCD ID of the target frames, by default None
    """

    logger.info(f"loading store from '{path}'")
    metadata_path = os.path.join(path, "metadata.hdf5")

    store = h5py.File(metadata_path, "r")
    dataset = store[observatory]

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(dataset[str(mjd)][()])
    else:
        metadata = []
        for mjd in dataset.keys():
            metadata.append(dataset[mjd])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))
    # close store
    store.close()

    logger.info(f"found {len(metadata)} frames in store")

    # filter by exposure number, spectrograph and/or CCD
    query = []
    if spec is not None:
        logger.info(f"filtering by {spec = }")
        query.append("spec == @spec")
    if expnum is not None:
        logger.info(f"filtering by {expnum = }")
        query.append("expnum == @expnum")
    if ccd is not None:
        logger.info(f"filtering by {ccd = }")
        query.append("ccd == @ccd")

    if query:
        query = " | ".join(query)
        metadata.query(query, inplace=True)

    logger.info(f"final number of frames after filtering {len(metadata)}")

    return metadata


def get_calib_metadata():
    try:
        query = CalibrationFrames.select().where(
            (~CalibrationFrames.is_master) & (CalibrationFrames.flags == 0)
        )
    except Error as e:
        print(e)

    calibration_metadata = [calib_metadata for calib_metadata in query]
    return calibration_metadata


def get_nonanalogs_groups(metadata_list):
    """Return filtered metadata list filtered by analog attributes: imagetyp, mjd, ccd and exptime"""
    # build metadata table containing columns relevant for analog selection: imagetyp, mjd, ccd, exptime
    metadata_table = Table(data=[metadata.__dict__ for metadata in metadata_list])
    metadata_group = metadata_table.group_by(["imagetyp", "mjd", "ccd", "exptime"])
    # return filtered list containing non-analog frames metadata for which to find analogs
    return [metadata_list[idx] for idx in metadata_group.groups.indices]


def get_analogs_metadata(metadata):
    # define empty metadata in case current frame has already a master
    analogs_metadata = []
    if metadata.imagetyp in CALIBRATION_TYPES and not metadata.calib.is_master:
        try:
            query = LVMFrames.select().where(
                (
                    LVMFrames.stage
                    == ReductionStage.PREPROCESSED | ReductionStage.CALIBRATED
                )
                & (LVMFrames.flags == 0)
                & (LVMFrames.imagetyp == metadata.imagetyp)
                & (LVMFrames.camera == metadata.camera)
                & (LVMFrames.mjd == metadata.mjd)
                & (LVMFrames.exptime == metadata.exptime)
            )
        except Error as e:
            print(f"{metadata.imagetyp}: {e}")
    else:
        return analogs_metadata
    analogs_metadata = [analog_metadata for analog_metadata in query]
    return analogs_metadata


def get_master_metadata(metadata):
    """finds and retrieve calibration frames given a target frame

    Depending on the type of the target frame, a set of calibration
    frames may be needed. These are stored in lvmdrp.core.constants.FRAMES_CALIB_NEEDS.
    This function retrieves the closest in time set of calibration frames
    according to that mapping.

    NOTE: When frame_type=='bias', an empty list is returned.

    Parameters
    ----------
    db: mysql.connection object
        connection to DB from which calibration frames can be retrieved
    metadata: namespace
        the metadata for the target frame

    Returns
    -------
    calib_frames: list_like
        list containing the calibration frames needed by the target frame
    """
    # retrieve calibration needs
    frame_needs = FRAMES_CALIB_NEEDS.get(metadata.imagetyp)
    # raise error in case current frame is not recognized in FRAMES_CALIB_NEEDS
    if frame_needs is None:
        raise ValueError(f"Unrecognized frame type '{metadata.imagetyp}'")

    calib_frames = dict.fromkeys(frame_needs)
    # handle empty list cases (e.g., bias)
    if not frame_needs:
        return calib_frames
    # handle unrecognized frame type

    for calib_type in frame_needs:
        bmask = ReductionStage.PREPROCESSED | ReductionStage.CALIBRATED
        if calib_type in ["continuum", "arc"]:
            bmask += (
                ReductionStage.COSMIC_CLEAN
                | ReductionStage.STRAY_CLEAN
                | ReductionStage.FIBERS_FOUND
                | ReductionStage.FIBERS_TRACED
                | ReductionStage.SPECTRA_EXTRACTED
                | ReductionStage.WAVELENGTH_SOLVED
            )
        try:
            query = (
                LVMFrames.select()
                .where(
                    (LVMFrames.stage == bmask)
                    & (LVMFrames.flags == 0)
                    & (LVMFrames.imagetyp == calib_type)
                    & (LVMFrames.camera == metadata.camera)
                    & (LVMFrames.calib_id is not None)
                )
                .order_by(fn.ABS(metadata.mjd - LVMFrames.mjd).asc())
            )
        except Error as e:
            print(f"{calib_type}: {e}")

        # TODO: handle the case in which the retrieved frame is stale and/or has quality flags
        # BUG: there may be cases in which no frame is found
        # BUG: this is retrieving only the first (closest) calibration frame, not necessarily the best
        #      Should retrieve all possible calibration frames & decide which one is the best based on
        #      quality
        calib_frame = query.get_or_none()
        if calib_frame is not None:
            calib_frames[calib_type] = calib_frame.calib
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
