# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: metadata.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import pathlib
from glob import glob, has_magic
from tqdm import tqdm
from astropy.io import fits

import h5py
import pandas as pd

from lvmdrp.core.constants import FRAMES_CALIB_NEEDS
from lvmdrp.utils.bitmask import (
    RawFrameQuality,
    ReductionStage,
    ReductionStatus,
    QualityFlag,
)
from lvmdrp import log, __version__

# # NOTE: replace these lines with Brian's integration of sdss_access and sdss_tree
# from sdss_access import Access
# from sdss_access.path import Path


# path = Path(release="sdss5")
# access = Access(release="sdss5")
# access.set_base_dir()

DRPVER = __version__


METADATA_PATH = os.path.join(os.getenv("LVM_SPECTRO_REDUX"), DRPVER)
# -------------------------------------------------------------------------------

RAW_METADATA_COLUMNS = [
    ("hemi", str),
    ("tileid", int),
    ("mjd", int),
    ("imagetyp", str),
    ("spec", str),
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
    ('quality', str),
    ("qual", RawFrameQuality),
    ("stage", ReductionStage),
    ("status", ReductionStatus),
    ("drpqual", QualityFlag),
    ('name', str)
]
MASTER_METADATA_COLUMNS = [
    ("mjd", int),
    ("imagetyp", str),
    ("spec", str),
    ("camera", str),
    ("exptime", float),
    ("neon", bool),
    ("hgne", bool),
    ("krypton", bool),
    ("xenon", bool),
    ("argon", bool),
    ("ldls", bool),
    ("quartz", bool),
    ("quality", RawFrameQuality),
    ("stage", ReductionStage),
    ("status", ReductionStatus),
    ("drpqual", QualityFlag),
    ('name', str)
]


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


def _get_metadata_paths(tileid=None, mjd=None, kind="raw"):
    """return metadata path depending on the kind

    this function will define a path for a metadata store
    depending on the kind of the metadata that will be stored
    ("raw" or "master").

    Parameters
    ----------
    tileid : int, optional
        tile ID of the target frames, by default None
    mjd : int, optional
        MJD of the target frames, by default None
    kind : str, optional
        metadata kind for which to define a store path, by default "raw"

    Returns
    -------
    str
        path for the corresponding metadata store

    Raises
    ------
    ValueError
        if `tileid` and/or `mjd` are not given when `kind=="raw"`
    ValueError
        if kind is not "raw" or "master"
    """
    if kind == "raw":
        if tileid is None or mjd is None:
            raise ValueError(
                "`tileid` and `mjd` are needed to define a path for raw metadata"
            )
        path_pattern = os.path.join(
            METADATA_PATH, str(tileid), str(mjd), "raw_metadata.hdf5"
        )
    elif kind == "master":
        path_pattern = os.path.join(METADATA_PATH, "master_metadata.hdf5")
    else:
        raise ValueError("valid values for `kind` are: 'raw' and 'master'")

    if has_magic(path_pattern):
        metadata_paths = [pathlib.Path(i) for i in glob(path_pattern)]
    else:
        metadata_paths = [pathlib.Path(path_pattern)]

    return metadata_paths


def _filter_metadata(
    metadata,
    hemi=None,
    tileid=None,
    mjd=None,
    imagetyp=None,
    spec=None,
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
    qual=None,
    stage=None,
    status=None,
    drpqual=None,
):
    """return filtered metadata dataframe

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe to filter out using the given criteria
    hemi : str, optional
        hemisphere of the target frames ('s' or 'n'), by default None
    tileid : int, optional
        tile ID of the target frames, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : str, optional
        name of the spectrograph ('sp1', 'sp2' or 'sp3'), by default None
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
    quality : str, optional
        string of original raw frame quality, by default None
    qual : int, optional
        bitmask representing quality of the recution, by default None
    stage : int, optional
        bitmask representing stage of the raw frames, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    drpqual : int, optional
        bitmask representing the quality of the reduction, by default None

    Returns
    -------
    pandas.DataFrame
        filtered dataframe following the given criteria
    """
    query = []
    if hemi is not None:
        log.info(f"filtering by {hemi = }")
        query.append("hemi == @hemi")
    if tileid is not None:
        log.info(f"filtering by {tileid = }")
        query.append("tileid == @tileid")
    if mjd is not None:
        log.info(f"filtering by {mjd = }")
        query.append("mjd == @mjd")
    if imagetyp is not None:
        log.info(f"filtering by {imagetyp = }")
        query.append("imagetyp == @imagetyp")
    if spec is not None:
        log.info(f"filtering by {spec = }")
        query.append("spec == @spec")
    if camera is not None:
        log.info(f"filtering by {camera = }")
        query.append("camera == @camera")
    if expnum is not None:
        log.info(f"filtering by {expnum = }")
        query.append("expnum == @expnum")
    if exptime is not None:
        log.info(f"filtering by {exptime = }")
        query.append("exptime == @exptime")
    if neon is not None:
        log.info(f"filtering by {neon = }")
        query.append("neon == @neon")
    if hgne is not None:
        log.info(f"filtering by {hgne = }")
        query.append("hgne == @hgne")
    if krypton is not None:
        log.info(f"filtering by {krypton = }")
        query.append("krypton == @krypton")
    if xenon is not None:
        log.info(f"filtering by {xenon = }")
        query.append("xenon == @xenon")
    if argon is not None:
        log.info(f"filtering by {argon = }")
        query.append("argon == @argon")
    if ldls is not None:
        log.info(f"filtering by {ldls = }")
        query.append("ldls == @ldls")
    if quartz is not None:
        log.info(f"filtering by {quartz = }")
        query.append("quartz == @quartz")
    if quality is not None:
        log.info(f"filtering by {quality = }")
        query.append("quality == @quality")
    if qual is not None:
        log.info(f"filtering by {qual = }")
        query.append("qual == @qual")
    if stage is not None:
        log.info(f"filtering by {stage = }")
        query.append("stage == @stage")
    if status is not None:
        log.info(f"filtering by {status = }")
        query.append("status == @status")
    if drpqual is not None:
        log.info(f"filtering by {drpqual = }")
        query.append("drpqual == @drpqual")

    if query:
        query = " and ".join(query)
        metadata.query(query, inplace=True)

    return metadata


def _load_or_create_store(tileid=None, mjd=None, kind="raw", mode="a"):
    """return the metadata store given a tile ID and an MJD

    if loading/creating a store for raw frames metadata, this function will
    require `tileid` and `mjd` to be passed with not values. Multiple stores
    can be created/loaded at the same time using wildcards in `tileid` and/or
    `mjd`.

    Parameters
    ----------
    tileid : int, optional
        tile ID for which a store will be loaded, by default None
    mjd : int, optional
        MJD for which a store will be loaded, by default None
    kind : str, optional
        metadata kind for which a store will be loaded/created, by default "raw"
    mode : str, optional
        instantiate store in read/write mode ("r", "a"), by default "a"

    Returns
    -------
    h5py.Group
        the metadata store for the given observatory
    """
    if mode not in {'r', 'a'}:
        raise ValueError(f"invalid value for {mode = }")

    if not tileid and not mjd:
        raise ValueError(f"specific values for {tileid = } and {mjd = } are needed")

    # define metadata path depending on the kind
    metadata_paths = _get_metadata_paths(tileid=tileid, mjd=mjd, kind=kind)

    stores = []
    for metadata_path in metadata_paths:
        # create the directory if needed
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        msg = 'loading' if metadata_path.exists() else 'creating'
        log.info(f"{msg} metadata store of {kind = }, {tileid = } and {mjd = }")
        stores.append(h5py.File(metadata_path, mode=mode))

    return stores


def _del_store(tileid=None, mjd=None, kind="raw"):
    """delete the entire HDF store if exists

    Parameters
    ----------
    tileid: int, optional
        tile ID for which a metadata store will be deleted, by default None
    mjd: int, optional
        MJD for which a metadata store will be deleted, by default None
    kind : str, optional
        metadata kind for which a store will be deleted, by default "raw"
    """
    # define metadata path depending on the kind
    metadata_paths = _get_metadata_paths(tileid=tileid, mjd=mjd, kind=kind)

    for metadata_path in metadata_paths:
        if os.path.exists(metadata_path):
            log.info(
                f"removing metadata store of {kind = }, {tileid = } and {mjd = }"
            )
            os.remove(metadata_path)
        else:
            log.warning(
                f"no metadata store of {kind = }, {tileid = } and {mjd = } found, "
                "nothing to do"
            )

from typing import Union
from lvmdrp.utils.hdrfix import apply_hdrfix


def get_frames_metadata(mjd: Union[str, int] = None, suffix: str = "fits",
                        overwrite: bool = None) -> pd.DataFrame:
    """ Extract metadata from the 2d raw frames

    Builds an Pandas DataFrame table containing extracted metadata for each of the 2d raw sdR
    frame files.  Globs for all files in the ``mjd`` subdirectory of the raw LVM data.
    If no mjd is specified, searches all of them. Writes the dataframe to an HDF store
    and if found, will load the cached content from there.  The cache is written
    into the MJD subdirectory of the LVM_SPECTRO_REDUX reduction folder

    Parameters
    ----------
    mjd : Union[str, int], optional
        The MJD of the data sub-directory to search in, by default None
    suffix : str, optional
        The raw data file suffix, by default "fits"
    overwrite : bool, optional
        Flag to ignore the HDF cache, by default None

    Returns
    -------
    pd.DataFrame
        a Pandas DataFrame of metadata
    """
    # look up raw data in the relevant MJD path
    raw_data_path = os.getenv("LVM_DATA_S")
    raw_frame = f'{mjd}/sdR*{suffix}*' if mjd else f'*/sdR*{suffix}*'
    frames = list(pathlib.Path(raw_data_path).rglob(raw_frame))

    if _load_or_create_store(tileid='*', mjd=mjd, kind='raw') and not overwrite:
        log.info("Loading existing metadata store.")
        meta = get_metadata(mjd=mjd, tileid='*')
    else:
        if overwrite:
            _del_store(mjd=mjd, tileid='*')

        log.info('Creating new metadata store.')
        meta = extract_metadata(frames)
        add_raws(meta)

    return meta


def extract_metadata(frames_paths: list) -> pd.DataFrame:
    """return dataframe with metadata extracted from given frames list

    this function will extract metadata from FITS headers given a list of
    frames.

    Parameters
    ----------
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
    log.info(f"going to extract metadata from {nframes} frames")
    iterator = tqdm(
        enumerate(frames_paths),
        total=nframes,
        desc="extracting metadata",
        ascii=True,
        unit="frame",
    )
    for i, frame_path in iterator:
        try:
            header = fits.getheader(frame_path, ext=0)
        except OSError as e:
            log.error(f'Cannot read FITS header: {e}')
            continue

        frame_path = pathlib.Path(frame_path)

        # apply any header fix or if none, use old header
        mjd = header.get("MJD")
        header = apply_hdrfix(mjd, hdr=header) or header

        new_metadata[i] = [
            "n" if header.get("OBSERVAT") != "LCO" else "s",
            header.get("TILEID", 1111),
            header.get("MJD"),
            header.get("IMAGETYP"),
            header.get("SPEC"),
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
            header.get("QUALITY", 'excellent'),
            header.get("QUAL", RawFrameQuality(0)),
            ReductionStage.UNREDUCED,
            ReductionStatus(0),
            QualityFlag(0),
            frame_path.stem
        ]

    # define dataframe
    new_metadata = pd.DataFrame.from_dict(new_metadata, orient="index")
    new_metadata.columns = list(zip(*RAW_METADATA_COLUMNS))[0]
    return new_metadata


def add_raws(metadata):
    """add new metadata to store

    this function will add new records to a HDF5 store given a metadata
    dataframe. If several MJD/TILEIDs are present in the metadata, the
    dataframe will be arranged into MJD/TILEID groups to be added to the
    corresponding MJD.

    Parameters
    ----------
    metadata : pandas.DataFrame
        dataframe to containing new metadata to add to store
    """

    # group metadata in unique MJDs and tile IDs
    metadata_groups = metadata.groupby(["tileid", "mjd"])
    for tileid, mjd in metadata_groups.groups:
        # define current group
        metadata_group = metadata_groups.get_group((tileid, mjd))

        # extract target dataset from store
        store = _load_or_create_store(tileid=tileid, mjd=mjd, kind="raw", mode="a")[0]

        # prepare metadata to be added to the store
        columns = list(zip(*RAW_METADATA_COLUMNS))[0]
        array = metadata_group.filter(items=columns).to_records(index=False)

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

        if "raw" in store:
            log.info(
                f"updating metadata store for {tileid = } and {mjd = } "
                f"with {len(array)} new rows"
            )
            dataset = store["raw"]
            dataset.resize(dataset.shape[0] + array.shape[0], axis=0)
            dataset[-array.shape[0] :] = array
            log.info(f"final number of rows {dataset.size}")
        else:
            log.info(
                f"creating metadata store for {tileid = } and {mjd = } "
                f"with {len(array)} new rows"
            )
            dataset = store.create_dataset(
                "raw", data=array, maxshape=(None,), chunks=True
            )

        # write metadata in HDF5 format
        log.info("writing raw metadata store to disk")
        dataset.file.close()


def add_masters(metadata):
    """add master calibration frame metadata to store

    Parameters
    ----------
    metadata : pd.DataFrame
        metadata dataframe to be added to the corresponding store
    """
    store = _load_or_create_store(kind="master", mode="a")[0]

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

    if "master" in store:
        log.info(
            f"updating metadata store for master frames with {len(array)} new rows"
        )
        dataset = store["master"]
        dataset.resize(dataset.shape[0] + array.shape[0], axis=0)
        dataset[-array.shape[0] :] = array
        log.info(f"final number of rows {dataset.size}")
    else:
        log.info(
            f"creating metadata store for master frames with {len(array)} new rows"
        )
        dataset = store.create_dataset(
            "master", data=array, maxshape=(None,), chunks=True
        )

    # write metadata in HDF5 format
    log.info("writing master metadata store to disk")
    store.close()


def del_metadata(tileid=None, mjd=None, kind="raw"):
    """delete dataset(s) from a target store

    Parameters
    ----------
    tileid : int, optional
        tile ID where the target dataset is located, by default None
    mjd : int, optional
        MJD where the target dataset is located, by default None
    kind : str, optional
        name of the dataset to delete: 'raw', 'master', by default 'raw'
    """
    stores = _load_or_create_store(tileid=tileid, mjd=mjd, kind=kind, mode="a")
    for store in stores:
        if kind in store:
            log.info(
                f"deleting metadata from store for {kind = }, {tileid = } and {mjd = }"
            )
            del store[kind]
        else:
            log.warning(
                f"no metadata of {kind = }, {tileid = } and {mjd = }, nothing to do"
            )

        store.close()


def get_metadata(
    tileid=None,
    mjd=None,
    hemi=None,
    imagetyp=None,
    spec=None,
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
    drpqual=None,
    kind="raw",
):
    """return raw frames metadata from precached HDF5 store

    Parameters
    ----------
    tileid : int, optional
        tile ID of the target frames, by default None
    mjd : int, optional
        MJD where the target frames is located, by default None
    hemi : str, optional
        hemisphere where the target frames were taken, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : str, optional
        name of the spectrograph ('sp1', 'sp2' or 'sp3'), by default None
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
        bitmask representing quality of the raw frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    drpqual : int, optional
        bitmask representing the quality of the reduction, by default None
    kind : str, optional
        name of the dataset to get data from, by default 'raw'

    Returns
    -------
    pandas.DataFrame
        the metadata dataframe filtered following the given criteria
    """
    # default output
    default_output = pd.DataFrame(
        columns=list(zip(*RAW_METADATA_COLUMNS))[0]
        if kind == "raw"
        else list(zip(*MASTER_METADATA_COLUMNS))[0]
    )

    # extract metadata
    stores = _load_or_create_store(tileid=tileid, mjd=mjd, kind=kind)

    metadatas = []
    for store in stores:
        if kind not in store:
            log.warning(f"no metadata found of {kind = }, {tileid = } and {mjd = }")
            return default_output
        else:
            dataset = store[kind]

        # extract metadata as dataframe
        metadata = pd.DataFrame(dataset[()])
        log.info(f"found {len(metadata)} frames in store '{store.file.filename}'")

        # close store
        store.close()

        # convert bytes to literal strings
        metadata = _decode_string(metadata)

        # filter by exposure number, spectrograph and/or camera
        metadata = _filter_metadata(
            metadata=metadata,
            hemi=hemi,
            imagetyp=imagetyp,
            spec=spec,
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
            drpqual=drpqual,
        )
        log.info(f"number of frames after filtering {len(metadata)}")

        metadatas.append(metadata)

    metadata = pd.concat(metadatas, axis="index", ignore_index=True)

    log.info(f"total number of frames found {len(metadata)}")

    return metadata


def get_analog_groups(
    tileid,
    mjd,
    hemi=None,
    imagetyp=None,
    spec=None,
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
    drpqual=None,
):
    """return a list of metadata groups considered to be analogs

    the given metadata dataframe is grouped in analog frames using
    the following criteria:
        * imagetyp
        * camera
        * exptime

    Parameters
    ----------
    tileid : int
        tile ID of the target frames
    mjd : int
        MJD where the target frames is located
    hemi : str, optional
        hemisphere where the target frames were taken, by default None
    imagetyp : str, optional
        type/flavor of frame to locate `IMAGETYP`, by default None
    spec : str, optional
        name of the spectrograph ('sp1', 'sp2' or 'sp3'), by default None
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
        bitmask representing quality of the raw frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    drpqual : int, optional
        bitmask representing the quality of the reduction, by default None

    Returns
    -------
    pandas.DataFrame
        the grouped metadata filtered following the given criteria
    """
    # default output
    default_output = [pd.DataFrame(columns=list(zip(*RAW_METADATA_COLUMNS))[0])]

    # extract raw frame metadata
    stores = _load_or_create_store(tileid=tileid, mjd=mjd, kind="raw", mode="r")

    metadatas = []
    for store in stores:
        if "raw" not in store:
            log.warning(f"no metadata found for {tileid = } and {mjd = }")
            return default_output
        else:
            dataset = store["raw"]

        # extract metadata as dataframe
        metadata = pd.DataFrame(dataset[()])
        log.info(f"found {len(metadata)} frames in store")

        # close store
        store.close()

        # convert bytes to literal strings
        metadata = _decode_string(metadata)

        # filter by exposure number, spectrograph and/or camera
        metadata = _filter_metadata(
            metadata=metadata,
            hemi=hemi,
            tileid=tileid,
            mjd=mjd,
            imagetyp=imagetyp,
            spec=spec,
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
            drpqual=drpqual,
        )
        log.info(f"final number of frames after filtering {len(metadata)}")

    metadatas.append(metadata)

    log.info("grouping analogs")
    metadata_groups = metadata.groupby(["imagetyp", "camera", "exptime"])

    log.info(f"found {len(metadata_groups)} groups of analogs:")
    analogs = []
    for g in metadata_groups.groups:
        log.info(g)
        analogs.append(metadata_groups.get_group(g))
    return analogs


def match_master_metadata(
    target_imagetyp,
    target_camera,
    target_exptime,
    mjd=None,
    hemi=None,
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
    drpqual=None,
):
    """return the matched master calibration frames given a target frame metadata

    Depending on the type of the target frame, a set of calibration frames may
    be needed. These are stored in lvmdrp.core.constants.FRAMES_CALIB_NEEDS.
    This function retrieves the closest in time set of calibration frames
    according to that mapping.

    Parameters
    ----------
    target_imagetyp : str
        type/flavor of frame to locate `IMAGETYP`
    target_camera : str
        camera ID of the target frames
    target_exptime : float
        exposure time of the target frames
    mjd : int, optional
        MJD where the target frames is located, by default None
    hemi : str, optional
        hemisphere where the target frames were taken, by default None
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
        bitmask representing quality of the raw frames, by default None
    stage : int, optional
        bitmask representing stage of the reduction, by default None
    status : int, optional
        bitmask representing status of the reduction, by default None
    drpqual : int, optional
        bitmask representing quality of the reduction, by default None

    Returns
    -------
    dict_like
        a dictionary containing the matched master calibration frames
    """
    # locate calibration needs
    frame_needs = FRAMES_CALIB_NEEDS.get(target_imagetyp)
    log.info(
        (
            f"target frame of type '{target_imagetyp}' "
            f"needs calibration frames: {', '.join(frame_needs) or None}"
        )
    )
    # initialize master calibration matches
    calib_frames = dict.fromkeys(frame_needs)

    # extract master calibration frames metadata
    store = _load_or_create_store(kind="master")[0]
    if "master" not in store:
        log.warning("no metadata found for master calibration frames")
        return calib_frames
    else:
        masters = store["master"]

    if len(masters) == 0:
        masters.file.close()
        log.error("no master calibration frames found in store")
        return calib_frames

    # extract MJD if given, else extract all MJDs
    masters_metadata = pd.DataFrame(masters[()])
    # close store
    masters.file.close()

    # convert bytes to literal strings
    masters_metadata = _decode_string(masters_metadata)

    log.info(f"found {len(masters_metadata)} master frames in store")

    # filter by exposure number, spectrograph and/or camera
    log.info(
        f"final number of master frames after filtering {len(masters_metadata)}"
    )
    # raise error in case current frame is not recognized in FRAMES_CALIB_NEEDS
    if frame_needs is None:
        log.error(f"no calibration frames found for '{target_imagetyp}' type")
        return calib_frames
    # handle empty list cases (e.g., bias)
    if not frame_needs:
        return calib_frames

    for calib_type in frame_needs:
        calib_metadata = _filter_metadata(
            metadata=masters_metadata,
            mjd=mjd,
            imagetyp=calib_type,
            camera=target_camera,
            exptime=target_exptime if calib_type != "bias" else None,
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
        if len(calib_metadata) == 0:
            log.error(f"no master {calib_type} frame found")
        else:
            log.info(f"found master {calib_type}")
            calib_frames[calib_type] = calib_metadata.iloc[0]
    return calib_frames


def put_reduction_stage(
    stage,
    tileid,
    mjd,
    camera,
    expnum,
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
    expnum : int, optional
        exposure number of the target frames, by default None
    """
    stores = _load_or_create_store(tileid=tileid, mjd=mjd, kind="raw")

    for store in stores:
        if "raw" not in store:
            log.warning(
                f"no metadata found for {tileid = } and {mjd = }, nothing to do"
            )
            return
        else:
            dataset = store["raw"]

        # extract raw frames metadata
        metadata = pd.DataFrame(dataset[()])

        # update stage to a subset of the metadata
        selection = (metadata.camera == camera) & (metadata.expnum == expnum)
        metadata.loc[selection, "stage"] = stage

        # update store
        dataset[...] = metadata.to_records()
        store.close()