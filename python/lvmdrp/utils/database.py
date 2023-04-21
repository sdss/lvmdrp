# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: database.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
from tqdm import tqdm
from astropy.io import fits

import h5py
import numpy as np
import pandas as pd

from lvmdrp.core.constants import CALIBRATION_TYPES, FRAMES_CALIB_NEEDS
from lvmdrp.utils.bitmask import ReductionStage, ReductionStatus, QualityFlag
from lvmdrp.utils.logger import get_logger

# NOTE: replace these lines with Brian's integration of sdss_access and sdss_tree
from sdss_access import Access
from sdss_access.path import Path


path = Path(release="sdss5")
access = Access(release="sdss5")
access.set_base_dir()
# -------------------------------------------------------------------------------

RAW_METADATA_COLUMNS = [
    ("hemi", str),
    ("imagetyp", str),
    ("camera", str),
    ("expnum", int),
    ("exptime", float),
    ("neon", bool),
    ("hgne", bool),
    ("krypton", bool),
    ("xenon", bool),
    ("ldls", bool),
    ("quartz", bool),
    ("quality", QualityFlag),
    ("stage", ReductionStage),
    ("status", ReductionStatus),
]
MASTER_METADATA_COLUMNS = [
    ("imagetyp", str),
    ("camera", str),
    ("exptime", float),
    ("neon", bool),
    ("hgne", bool),
    ("krypton", bool),
    ("xenon", bool),
    ("ldls", bool),
    ("quartz", bool),
    ("quality", QualityFlag),
    ("stage", ReductionStage),
    ("status", ReductionStatus),
]


logger = get_logger(__name__)


def _decode_string(metadata):
    """return a dataframe with bytes columns turn into literal strings

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe for which bytes columns will be turned into strings

    Returns
    -------
    pandas.DataFrame
        dataframe with all bytes columns turned into literal strings
    """
    df_str = metadata.select_dtypes(object).apply(
        lambda s: s.str.decode("utf-8"), axis="columns"
    )
    metadata[df_str.columns] = df_str
    return metadata


def _filter_metadata(
    metadata,
    mjd=None,
    imagetyp=None,
    spec=None,
    camera=None,
    expnum=None,
    stage=None,
    status=None,
    quality=None,
):
    """return filtered metadata dataframe

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe to filter out using the given criteria
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None

    Returns
    -------
    pandas.DataFrame
        filtered dataframe following the given criteria
    """
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

    return metadata


def _load_or_create_store(observatory, overwrite=False):
    """return the metadata store given a path

    Parameters
    ----------
    observatory: str
        name of the observatory from/for which data will be retrieved/cached
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


def extract_metadata(mjd, frames_paths):
    """return dataframe with metadata extracted from given frames list

    this function will extract metadata from FITS headers given a list of
    frames.

    Parameters
    ----------
    mjd : int
        MJD from which the metadata will be extracted
    frames_paths : list_like
        list of frames paths in the local mirror of SAS

    Returns
    -------
    pandas.DataFrame
        dataframe containing the extracted metadata
    """
    new_metadata = {}
    # extract metadata
    nframes = len(frames_paths)
    logger.info(f"extracting metadata from {nframes} frames")
    iterator = tqdm(
        enumerate(frames_paths),
        total=nframes,
        desc=f"extracting metadata from MJD = {mjd}",
        ascii=True,
        unit="frame",
    )
    for i, frame_path in iterator:
        header = fits.getheader(frame_path, ext=0)
        new_metadata[i] = [
            "n" if header.get("OBSERVAT") != "LCO" else "s",
            header.get("IMAGETYP"),
            header.get("CCD"),
            header.get("EXPOSURE"),
            header.get("EXPTIME"),
            header.get("NEON", "OFF") == "ON",
            header.get("HGNE", "OFF") == "ON",
            header.get("KRYPTON", "OFF") == "ON",
            header.get("XENON", "OFF") == "ON",
            header.get("LDLS", "OFF") == "ON",
            header.get("QUARTZ", "OFF") == "ON",
            header.get("QUALITY", QualityFlag(0)),
            ReductionStage.UNREDUCED,
            ReductionStatus(0),
        ]

    # define dataframe
    new_metadata = pd.DataFrame.from_dict(new_metadata, orient="index")
    new_metadata.columns = list(zip(*RAW_METADATA_COLUMNS))[0]
    return new_metadata


def put_metadata(observatory, mjd, metadata):
    """add new metadata to store

    Parameters
    ----------
    observatory: str
        name of the observatory for which data will be cached
    mjd : int
        MJD where the target frames is located
    metadata : pandas.DataFrame
        dataframe to containing new metadata to add to store
    """
    store = _load_or_create_store(observatory=observatory)
    raw = store["raw"]

    if str(mjd) in raw:
        logger.info("updating store with new metadata")
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
        dataset = raw[str(mjd)]
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
        raw.create_dataset(name=str(mjd), data=array, maxshape=(None,), chunks=True)

    # write to disk metadata in HDF5 format
    logger.info(f"writing metadata to store '{access.base_dir}'")
    store.file.close()


def del_metadata(observatory, mjd=None):
    """delete dataset(s) from a target store

    Parameters
    ----------
    observatory: str
        name of the observatory for which data will be cached
    mjd : int
        MJD where the target dataset is located
    """
    if mjd is None:
        store = _load_or_create_store(observatory=observatory, overwrite=True)
    else:
        store = _load_or_create_store(observatory=observatory)
        if f"raw/{mjd}" in store:
            del store[f"raw/{mjd}"]
        if f"master/{mjd}" in store:
            del store[f"master/{mjd}"]

    store.file.close()


def get_metadata(
    observatory="lco",
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
    observatory : str, optional
        name of the observatory, by default 'lco'
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None

    Returns
    -------
    pandas.DataFrame
        the metadata dataframe filtered following the given criteria
    """

    store = _load_or_create_store(observatory=observatory)

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(store[f"raw/{mjd}"][()])
    else:
        metadata = []
        for mjd in store.keys():
            metadata.append(store[str(mjd)][()])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))

    # close store
    store.file.close()

    # convert bytes to literal strings
    metadata = _decode_string(metadata)

    logger.info(f"found {len(metadata)} frames in store")

    # filter by exposure number, spectrograph and/or camera
    metadata = _filter_metadata(
        metadata, mjd, imagetyp, spec, camera, expnum, stage, status, quality
    )
    logger.info(f"final number of frames after filtering {len(metadata)}")

    return metadata


def get_analog_groups(
    observatory="lco",
    imagetyp=None,
    mjd=None,
    expnum=None,
    spec=None,
    camera=None,
    stage=None,
    status=None,
    quality=None,
):
    """return a list of metadata groups considered to be analogs

    the given metadata dataframe is grouped in analog frames using
    the following criteria:
        * mjd
        * imagetyp
        * camera
        * exptime

    Parameters
    ----------
    observatory : str, optional
        name of the observatory, by default 'lco'
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None

    Returns
    -------
    pandas.DataFrame
        the grouped metadata filtered following the given criteria
    """
    store = _load_or_create_store(observatory=observatory)

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(store[f"raw/{mjd}"][()])
    else:
        metadata = []
        for mjd in store.keys():
            metadata.append(store[str(mjd)][()])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))
    # close store
    store.file.close()

    # convert bytes to literal strings
    metadata = _decode_string(metadata)

    logger.info(f"found {len(metadata)} frames in store")

    # filter by exposure number, spectrograph and/or camera
    metadata = _filter_metadata(
        metadata, mjd, imagetyp, spec, camera, expnum, stage, status, quality
    )
    logger.info(f"final number of frames after filtering {len(metadata)}")

    logger.info("grouping analogs")
    metadata_groups = metadata.groupby(["imagetyp", "mjd", "camera", "exptime"])

    logger.info(f"found {len(metadata_groups)} groups of analogs:")
    analogs = []
    for g in metadata_groups.groups:
        logger.info(g)
        analogs.append(metadata_groups.get_group(g))
    return analogs


def get_master_metadata(
    observatory="lco",
    imagetyp=None,
    mjd=None,
    expnum=None,
    spec=None,
    camera=None,
    stage=None,
    status=None,
    quality=None,
):
    """return the matched master calibration frames given a target frame

    Depending on the type of the target frame, a set of calibration frames may
    be needed. These are stored in lvmdrp.core.constants.FRAMES_CALIB_NEEDS.
    This function retrieves the closest in time set of calibration frames
    according to that mapping.

    Parameters
    ----------
    observatory : str, optional
        name of the observatory, by default 'lco'
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None

    Returns
    -------
    dict_like
        a dictionary containing the matched master calibration frames
    """
    # locate calibration needs
    frame_needs = FRAMES_CALIB_NEEDS.get(imagetyp)
    # initialize master calibration matches
    calib_frames = dict.fromkeys(CALIBRATION_TYPES)

    store = _load_or_create_store(observatory=observatory)

    masters = store["master"]
    if len(masters) == 0:
        store.file.close()
        logger.error("no master calibration frames found in store")
        return calib_frames

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        masters_metadata = pd.DataFrame(masters[str(mjd)][()])
    else:
        masters_metadata = []
        for mjd in masters.keys():
            masters_metadata.append(masters[str(mjd)][()])
        masters_metadata = pd.DataFrame(np.concatenate(masters_metadata, axis=0))
    # close store
    store.file.close()

    # convert bytes to literal strings
    masters_metadata = _decode_string(masters_metadata)

    logger.info(f"found {len(masters_metadata)} master frames in store")

    # filter by exposure number, spectrograph and/or camera
    masters_metadata = _filter_metadata(
        masters_metadata,
        mjd=mjd,
        spec=spec,
        camera=camera,
        expnum=expnum,
        stage=stage,
        status=status,
        quality=quality,
    )
    logger.info(
        f"final number of master frames after filtering {len(masters_metadata)}"
    )
    logger.info(
        (
            f"target frame of type '{imagetyp}' "
            f"needs calibration frames: {', '.join(frame_needs) or None}"
        )
    )
    # raise error in case current frame is not recognized in FRAMES_CALIB_NEEDS
    if frame_needs is None:
        logger.error(f"no calibration frames found for '{imagetyp}' type")
        return calib_frames
    # handle empty list cases (e.g., bias)
    if not frame_needs:
        return calib_frames

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
        q = "@masters_metadata.stage == @bmask"

        # TODO: handle the case in which the retrieved frame is stale and/or has quality
        # flags
        # BUG: there may be cases in which no frame is found
        # BUG: this is retrieving only the first (closest) calibration frame, not
        #      necessarily the best. Should retrieve all possible calibration frames
        #      & decide which one is the best based on quality
        calib_frame = masters_metadata.query(q)
        calib_frame["mjd_diff"] = calib_frame.mjd.apply(lambda v: abs(v - mjd))
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


def put_reduction_stage(
    stage,
    mjd,
    camera,
    expnum,
    observatory="lco",
):
    """update frame metadata with given reduction stage

    Given a values of mjd, camera and exposure number to uniquely identify a
    frame, this function will update the corresponding metadata reduction
    stage.

    Parameters
    ----------
    stage : ReductionStage
        the reduction stage value
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : int, optional
        spectrograph of the target frames, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : str, optional
        zero-padded exposure number of the target frames, by default None
    observatory : str, optional
        name of the observatory, by default 'lco'
    """
    store = _load_or_create_store(observatory=observatory)

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(store[f"raw/{mjd}"][()])
    else:
        metadata = []
        for mjd in store.keys():
            metadata.append(store[str(mjd)][()])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))

    # update stage to a subset of the metadata
    selection = (metadata.camera == camera) & (metadata.expnum == expnum)
    metadata.loc[selection, "stage"] = stage

    # update store
    store[f"raw/{mjd}"][...] = metadata.to_records()
    store.file.close()
