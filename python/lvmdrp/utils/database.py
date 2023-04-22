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
    ("argon", bool),
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
    ("argon", bool),
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
    camera=None,
    expnum=None,
    exptime=None,
    neon=None,
    hgne=None,
    krypton=None,
    xenon=None,
    argon=None,
    ldls=None,
    quartz=None,
    quality=None,
    stage=None,
    status=None,
):
    """return filtered metadata dataframe

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe to filter out using the given criteria
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : int, optional
        exposure number of the target frames, by default None
    exptime : float, optional
        exposure time of the target frames, by default None
    neon : bool, optional
        whether is Neon lamp on or not, by default None
    hgne : bool, optional
        whether is HGNE lamp on or not, by default None
    krypton : bool, optional
        whether is Krypton lamp on or not, by default None
    xenon : bool, optional
        whether is Xenon lamp on or not, by default None
    argon : bool, optional
        whether is Argon lamp on or not, by default None
    ldls : bool, optional
        whether is LDLS lamp on or not, by default None
    quartz : bool, optional
        whether is Quartz lamp on or not, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None

    Returns
    -------
    pandas.DataFrame
        filtered dataframe following the given criteria
    """
    query = []
    if imagetyp is not None:
        logger.info(f"filtering by {imagetyp = }")
        query.append("imagetyp == @imagetyp")
    if camera is not None:
        logger.info(f"filtering by {camera = }")
        query.append("camera == @camera")
    if expnum is not None:
        logger.info(f"filtering by {expnum = }")
        query.append("expnum == @expnum")
    if exptime is not None:
        logger.info(f"filtering by {exptime = }")
        query.append("exptime == @exptime")
    if neon is not None:
        logger.info(f"filtering by {neon = }")
        query.append("neon == @neon")
    if hgne is not None:
        logger.info(f"filtering by {hgne = }")
        query.append("hgne == @hgne")
    if krypton is not None:
        logger.info(f"filtering by {krypton = }")
        query.append("krypton == @krypton")
    if xenon is not None:
        logger.info(f"filtering by {xenon = }")
        query.append("xenon == @xenon")
    if argon is not None:
        logger.info(f"filtering by {argon = }")
        query.append("argon == @argon")
    if ldls is not None:
        logger.info(f"filtering by {ldls = }")
        query.append("ldls == @ldls")
    if quartz is not None:
        logger.info(f"filtering by {quartz = }")
        query.append("quartz == @quartz")
    if quality is not None:
        logger.info(f"filtering by {quality = }")
        query.append("quality == @quality")
    if stage is not None:
        logger.info(f"filtering by {stage = }")
        query.append("stage == @stage")
    if status is not None:
        logger.info(f"filtering by {status = }")
        query.append("status == @status")

    if query:
        query = " and ".join(query)
        metadata.query(query, inplace=True)

    return metadata


def _create_store(observatory):
    """return the metadata store given a path

    Parameters
    ----------
    observatory: str
        observatory for which a metadata store will be created
    """
    metadata_path = os.path.join(access.base_dir, "metadata.hdf5")

    logger.info(f"creating metadata store '{metadata_path}'")
    store = h5py.File(metadata_path, mode="w")

    # add observatory group if needed
    if observatory not in store:
        store.create_group(f"{observatory}/raw")
        store.create_group(f"{observatory}/master")

    store.close()


def _del_store():
    """delete the entire HDF store if exists"""
    metadata_path = os.path.join(access.base_dir, "metadata.hdf5")

    if os.path.isfile(metadata_path):
        logger.info(f"removing metadata store '{metadata_path}'")
        os.remove(metadata_path)
    else:
        logger.warning(f"no '{metadata_path}' store found")


def _load_store(observatory, mode="r"):
    """return the metadata store given a observatory

    Parameters
    ----------
    observatory : str
        metadata observatory from which a store will be loaded
    mode : str, optional
        the mode in which to open the HDF5 store, by default 'r'

    Returns
    -------
    h5py.Group
        the metadata store for the given observatory
    """
    metadata_path = os.path.join(access.base_dir, "metadata.hdf5")

    logger.info(f"loading metadata store from '{metadata_path}'")
    store = h5py.File(metadata_path, mode=mode)

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
            header.get("ARGON", "OFF") == "ON",
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


def add_metadata(
    metadata,
    mjd,
    observatory="lco",
    kind="raw",
):
    """add new metadata to store

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe to containing new metadata to add to store
    mjd : int
        MJD where the target frames is located
    observatory: str, optional
        name of the observatory for which data will be cached, by default 'lco'
    kind : str, optional
        name of the dataset to add data to, by default 'raw'
    """

    # extract target dataset from store
    store = _load_store(observatory=observatory, mode="a")
    if kind not in ["raw", "master"]:
        logger.warning(
            f"unrecognised dataset {kind = }, falling back to kind = 'master'"
        )
        kind = "master"
    dataset = store[kind]

    # prepare metadata to be added to the store
    if kind == "raw":
        columns = list(zip(*RAW_METADATA_COLUMNS))[0]
        array = metadata.filter(items=columns).to_records(index=False)
    else:
        columns = list(zip(*MASTER_METADATA_COLUMNS))[0]
        array = metadata.filter(items=columns).to_records(index=False)

    # convert to proper dtypes
    dtypes = array.dtype
    array = array.astype(
        [
            (n, dtypes[n])
            if dtypes[n] != object
            else (n, h5py.string_dtype("utf-8", length=None))
            for n in dtypes.names
        ]
    )

    if str(mjd) in dataset:
        logger.info(f"updating store with new metadata for MJD = {mjd}")
        dataset = dataset[str(mjd)]
        dataset.resize(dataset.shape[0] + array.shape[0], axis=0)
        dataset[-array.shape[0] :] = array
    else:
        logger.info(f"adding new data to store for MJD = {mjd}")
        dataset.create_dataset(name=str(mjd), data=array, maxshape=(None,), chunks=True)

    # write metadata in HDF5 format
    logger.info(f"writing metadata to store '{access.base_dir}'")
    store.file.close()


def del_metadata(observatory, mjd):
    """delete dataset(s) from a target store

    Parameters
    ----------
    observatory: str
        name of the observatory for which data will be cached
    mjd : int
        MJD where the target dataset is located
    """
    store = _load_store(observatory=observatory, mode="a")
    logger.info(f"deleting MJD = {mjd} from '{observatory}' metadata store")
    if f"raw/{mjd}" in store:
        del store[f"raw/{mjd}"]
    if f"master/{mjd}" in store:
        del store[f"master/{mjd}"]

    store.file.close()


def get_metadata(
    mjd=None,
    imagetyp=None,
    camera=None,
    expnum=None,
    exptime=None,
    neon=None,
    hgne=None,
    krypton=None,
    xenon=None,
    argon=None,
    ldls=None,
    quartz=None,
    quality=None,
    stage=None,
    status=None,
    observatory="lco",
    kind="raw",
):
    """return raw frames metadata from precached HDF5 store

    Parameters
    ----------
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    expnum : int, optional
        exposure number of the target frames, by default None
    exptime : float, optional
        exposure time of the target frames, by default None
    neon : bool, optional
        whether is Neon lamp on or not, by default None
    hgne : bool, optional
        whether is HGNE lamp on or not, by default None
    krypton : bool, optional
        whether is Krypton lamp on or not, by default None
    xenon : bool, optional
        whether is Xenon lamp on or not, by default None
    argon : bool, optional
        whether is Argon lamp on or not, by default None
    ldls : bool, optional
        whether is LDLS lamp on or not, by default None
    quartz : bool, optional
        whether is Quartz lamp on or not, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    observatory : str, optional
        name of the observatory, by default 'lco'
    kind : str, optional
        name of the dataset to get data from, by default 'raw'

    Returns
    -------
    pandas.DataFrame
        the metadata dataframe filtered following the given criteria
    """

    # extract metadata
    store = _load_store(observatory=observatory)
    dataset = store[kind]

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(dataset[str(mjd)][()])
    else:
        metadata = []
        for mjd in dataset.keys():
            metadata.append(dataset[str(mjd)][()])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))

    # close store
    store.file.close()

    # convert bytes to literal strings
    metadata = _decode_string(metadata)

    logger.info(f"found {len(metadata)} frames in store")

    # filter by exposure number, spectrograph and/or camera
    metadata = _filter_metadata(
        metadata=metadata,
        mjd=mjd,
        imagetyp=imagetyp,
        camera=camera,
        expnum=expnum,
        exptime=exptime,
        neon=neon,
        hgne=hgne,
        krypton=krypton,
        xenon=xenon,
        argon=argon,
        ldls=ldls,
        quartz=quartz,
        quality=quality,
        stage=stage,
        status=status,
    )
    logger.info(f"final number of frames after filtering {len(metadata)}")

    return metadata


def get_analog_groups(
    mjd=None,
    imagetyp=None,
    camera=None,
    exptime=None,
    neon=None,
    hgne=None,
    krypton=None,
    xenon=None,
    argon=None,
    ldls=None,
    quartz=None,
    quality=None,
    stage=None,
    status=None,
    observatory="lco",
):
    """return a list of metadata groups considered to be analogs

    the given metadata dataframe is grouped in analog frames using
    the following criteria:
        * imagetyp
        * camera
        * exptime

    Parameters
    ----------
    mjd : int, optional
        MJD where the target frames is located, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    camera : str, optional
        camera ID of the target frames, by default None
    exptime : float, optional
        exposure time of the target frames, by default None
    neon : bool, optional
        whether is Neon lamp on or not, by default None
    hgne : bool, optional
        whether is HGNE lamp on or not, by default None
    krypton : bool, optional
        whether is Krypton lamp on or not, by default None
    xenon : bool, optional
        whether is Xenon lamp on or not, by default None
    argon : bool, optional
        whether is Argon lamp on or not, by default None
    ldls : bool, optional
        whether is LDLS lamp on or not, by default None
    quartz : bool, optional
        whether is Quartz lamp on or not, by default None
    quality : int, optional
        bitmask representing quality of the recution, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    observatory : str, optional
        name of the observatory, by default 'lco'

    Returns
    -------
    pandas.DataFrame
        the grouped metadata filtered following the given criteria
    """
    # extract raw frame metadata
    store = _load_store(observatory=observatory)
    dataset = store["raw"]

    # extract MJD if given, else extract all MJDs
    if mjd is not None:
        metadata = pd.DataFrame(dataset[f"{mjd}"][()])
    else:
        metadata = []
        for mjd in dataset.keys():
            metadata.append(dataset[str(mjd)][()])
        metadata = pd.DataFrame(np.concatenate(metadata, axis=0))
    # close store
    store.file.close()

    # convert bytes to literal strings
    metadata = _decode_string(metadata)

    logger.info(f"found {len(metadata)} frames in store")

    # filter by exposure number, spectrograph and/or camera
    metadata = _filter_metadata(
        metadata=metadata,
        mjd=mjd,
        imagetyp=imagetyp,
        camera=camera,
        exptime=exptime,
        neon=neon,
        hgne=hgne,
        krypton=krypton,
        xenon=xenon,
        argon=argon,
        ldls=ldls,
        quartz=quartz,
        quality=quality,
        stage=stage,
        status=status,
    )
    logger.info(f"final number of frames after filtering {len(metadata)}")

    logger.info("grouping analogs")
    metadata_groups = metadata.groupby(["imagetyp", "camera", "exptime"])

    logger.info(f"found {len(metadata_groups)} groups of analogs:")
    analogs = []
    for g in metadata_groups.groups:
        logger.info(g)
        analogs.append(metadata_groups.get_group(g))
    return analogs


def match_master_metadata(
    target_mjd,
    target_imagetyp,
    target_camera,
    target_exptime,
    quality=None,
    stage=None,
    status=None,
    observatory="lco",
):
    """return the matched master calibration frames given a target frame metadata

    Depending on the type of the target frame, a set of calibration frames may
    be needed. These are stored in lvmdrp.core.constants.FRAMES_CALIB_NEEDS.
    This function retrieves the closest in time set of calibration frames
    according to that mapping.

    Parameters
    ----------
    target_mjd : int
        MJD where the target frames is located
    target_imagetyp : str
        type/flavor of frame to locate `IMAGETYP`
    target_camera : str
        camera ID of the target frames
    target_exptime : float
        exposure time of the target frames
    quality : int, optional
        bitmask representing quality of the recution, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    observatory : str, optional
        name of the observatory, by default 'lco'

    Returns
    -------
    dict_like
        a dictionary containing the matched master calibration frames
    """
    # locate calibration needs
    frame_needs = FRAMES_CALIB_NEEDS.get(target_imagetyp)
    # initialize master calibration matches
    calib_frames = dict.fromkeys(CALIBRATION_TYPES)

    # extract master calibration frames metadata
    store = _load_store(observatory=observatory)
    masters = store["master"]
    if len(masters) == 0:
        store.file.close()
        logger.error("no master calibration frames found in store")
        return calib_frames

    # extract MJD if given, else extract all MJDs
    masters_metadata = pd.DataFrame(masters[str(target_mjd)][()])
    # close store
    store.file.close()

    # convert bytes to literal strings
    masters_metadata = _decode_string(masters_metadata)

    logger.info(f"found {len(masters_metadata)} master frames in store")

    # filter by exposure number, spectrograph and/or camera
    masters_metadata = _filter_metadata(
        metadata=masters_metadata,
        mjd=target_mjd,
        imagetyp=target_imagetyp,
        camera=target_camera,
        exptime=target_exptime,
        quality=quality,
        stage=stage,
        status=status,
    )
    logger.info(
        f"final number of master frames after filtering {len(masters_metadata)}"
    )
    logger.info(
        (
            f"target frame of type '{target_imagetyp}' "
            f"needs calibration frames: {', '.join(frame_needs) or None}"
        )
    )
    # raise error in case current frame is not recognized in FRAMES_CALIB_NEEDS
    if frame_needs is None:
        logger.error(f"no calibration frames found for '{target_imagetyp}' type")
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
        calib_frame["mjd_diff"] = calib_frame.mjd.apply(
            lambda mjd: abs(mjd - target_mjd)
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
    store = _load_store(observatory=observatory, mode="a")

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
