# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: metadata.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import itertools
import os
import pathlib
from glob import glob, has_magic
from typing import Union
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from filelock import FileLock, Timeout
from tqdm import tqdm

from lvmdrp.core.constants import FRAMES_CALIB_NEEDS, CAMERAS
from lvmdrp.utils.bitmask import (
    QualityFlag,
    ReductionStage,
    ReductionStatus,
)
from lvmdrp import log, __version__, path
from lvmdrp.utils.hdrfix import apply_hdrfix
from lvmdrp.utils.convert import dateobs_to_sjd, correct_sjd, tileid_grp


DRPVER = __version__


# -------------------------------------------------------------------------------

RAW_METADATA_COLUMNS = [
    ("hemi", str),
    ("tileid", int),
    ("mjd", int),  # actually SJD
    ("rmjd", int),  # the real MJD
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
    ("hartmann", str),
    ("qaqual", str),
    ("stage", ReductionStage),
    ("status", ReductionStatus),
    ("drpqual", QualityFlag),
    ("name", str),
    ("tilegrp", str)
]
MASTER_METADATA_COLUMNS = [
    ("tileid", int),
    ("mjd", int),  # actually SJD
    ("rmjd", int),  # the real MJD
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
    ("hartmann", str),
    ("qaqual", str),
    ("stage", ReductionStage),
    ("status", ReductionStatus),
    ("drpqual", QualityFlag),
    ("nframes", int),
    ("name", str),
    ("tilegrp", str)
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
    df_str = metadata.select_dtypes([object]).stack().str.decode('utf-8').unstack()
    metadata[df_str.columns] = df_str
    return metadata


def _get_metadata_paths(drpver=None, tileid=None, mjd=None, kind="raw", filter_exist=True):
    """return metadata path depending on the kind

    this function will define a path for a metadata store
    depending on the kind of the metadata that will be stored
    ("raw" or "master").

    Parameters
    ----------
    drpver : str, optional
        DRP version, by default None (current version)
    tileid : int, optional
        tile ID of the target frames, by default None
    mjd : int, optional
        MJD of the target frames, by default None
    kind : str, optional
        metadata kind for which to define a store path, by default "raw"
    filter_exist : bool, optional
        whether the paths should be filtered by existence, by default True

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
    # define DRP version
    drpver = drpver or DRPVER

    # define metadata path
    METADATA_PATH = os.path.join(os.getenv("LVM_SPECTRO_REDUX"), drpver)

    if kind == "raw":
        if tileid is None or mjd is None:
            raise ValueError(
                "`tileid` and `mjd` are needed to define a path for raw metadata"
            )
        tilegrp = tileid_grp(tileid)
        path_pattern = os.path.join(
            METADATA_PATH, tilegrp, str(tileid), str(mjd), "raw_metadata.hdf5"
        )
    elif kind == "master":
        path_pattern = os.path.join(METADATA_PATH, "master_metadata.hdf5")
    else:
        raise ValueError("valid values for `kind` are: 'raw' and 'master'")

    if has_magic(path_pattern):
        metadata_paths = [pathlib.Path(i) for i in glob(path_pattern)]
    else:
        metadata_paths = [pathlib.Path(path_pattern)]

    # return list of existing paths
    if filter_exist:
        metadata_paths = list(filter(os.path.exists, metadata_paths))

    return metadata_paths


def _filter_metadata(
    metadata,
    hemi=None,
    tileid=None,
    mjd=None,
    rmjd=None,
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
    if rmjd is not None:
        log.info(f"filtering by {rmjd = }")
        query.append("rmjd == @rmjd")
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
        return metadata.query(query)

    return metadata


def _load_or_create_store(drpver=None, tileid=None, mjd=None, kind="raw", mode="a"):
    """return the metadata store given a tile ID and an MJD

    if loading/creating a store for raw frames metadata, this function will
    require `tileid` and `mjd` to be passed with not values. Multiple stores
    can be created/loaded at the same time using wildcards in `tileid` and/or
    `mjd`.

    Parameters
    ----------
    drpver : str, optional
        DRP version, by default None (current version)
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
    # define DRP version
    drpver = drpver or DRPVER

    if mode not in {"r", "a"}:
        raise ValueError(f"invalid value for {mode = }")

    if kind == "raw" and not tileid and not mjd:
        raise ValueError(f"specific values for {tileid = } and {mjd = } are needed")

    # define metadata path depending on the kind
    metadata_paths = _get_metadata_paths(
        drpver=drpver, tileid=tileid, mjd=mjd, kind=kind, filter_exist=(mode == "r")
    )

    stores = []
    log.info(f"loading/creating metadata store with parameters {tileid = }, {mjd = } and {kind = }")
    for metadata_path in metadata_paths:
        # create the directory if needed
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

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
            log.info(f"removing metadata store at {metadata_path}")
            os.remove(metadata_path)
        else:
            log.warning(f"no metadata store at {metadata_path} found, " "nothing to do")


def locate_new_frames(hemi, camera, mjd, expnum, return_excluded=False):
    """return paths to new frames not present in the metadata store

    this function will expand the path to raw frames in the local SAS and
    filter out those paths to frames already present in the metadata stores.

    Parameters
    ----------
    hemi : str
        hemisphere of the observatory where the data was taken
    camera : str
        camera ID of the target frames
    mjd : int
        MJD of the target frames
    expnum : int
        exposure number of the target frames
    return_excluded : bool, optional
        whether to return the excluded paths or not, by default False

    Returns
    -------
    array_like
        list of raw frame paths not present in metadata stores
    """
    keys = ["mjd", "hemi", "camera", "expnum"]
    paths = sorted(
        path.expand("lvm_raw", hemi=hemi, camspec=camera, mjd=mjd, expnum=expnum)
    )
    npath = len(paths)
    log.info(f"found {npath} potentially new raw frame paths in local SAS")

    # extract path parameters
    new_path_params = pd.DataFrame(
        [path.extract(name="lvm_raw", example=p) for p in paths]
    )
    new_path_params.rename(columns={"camspec": "camera"}, inplace=True)
    new_path_params[["mjd", "expnum"]] = new_path_params[["mjd", "expnum"]].astype(int)
    new_path_params["x"] = 0
    new_path_params.set_index(keys, inplace=True)

    # load all stores if they exist
    log.info("locating all existing metadata stores")
    try:
        stores = _load_or_create_store(tileid="*", mjd="*", mode="r")
        log.info(f"found {len(stores)} metadata stores")
    except FileNotFoundError:
        log.info(f"no metadata stores found, returning {npath} new paths")
        return paths

    # convert to dataframe
    gen_path_params = map(lambda store: pd.DataFrame(store["raw"][()])[keys], stores)
    old_path_params = pd.concat(gen_path_params, ignore_index=True)
    old_path_params = _decode_string(old_path_params)
    old_path_params["x"] = 0
    old_path_params.set_index(keys, inplace=True)
    # filter out paths in stores
    news = ~new_path_params.isin(old_path_params).x.values
    log.info(f"filtered {news.sum()} paths of new frames present in stores")

    new_paths = np.asarray(paths)[news].tolist()
    if return_excluded:
        return new_paths, np.asarray(paths)[~news].tolist()
    return new_paths


def get_master_metadata(overwrite: bool = None) -> pd.DataFrame:
    """Extract metadata from the master calibration files

    Builds an Pandas DataFrame table containing extracted metadata for all the
    master calibration files found in the calib/ subdirectory of LVM_SPECTRO_REDUX.
    Globs for all files matching the pattern "*calib/*lvm-m*". Writes the dataframe to
    an HDF store and if found, will load the cached content from there.  The cache is written
    at the top level LVM_SPECTRO_REDUX/[DRPVER] reduction folder.

    Parameters
    ----------
    overwrite : bool, optional
        Flag to ignore the HDF cache, by default None

    Returns
    -------
    pd.DataFrame
        a Pandas DataFrame of metadata
    """

    # glob for all file master calibration files, only include bias,darks,arcs,flats
    files = list(pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")).rglob("*calib/*lvm-m[bdpaftwl]*"))

    if _load_or_create_store(kind="master", mode="r") and not overwrite:
        log.info("Loading existing metadata store.")
        meta = get_metadata(kind="master")
    else:
        if overwrite:
            _del_store(kind="master")

        log.info("Creating new metadata store.")
        meta = extract_metadata(files, kind="master")
        add_masters(meta)

    return meta


def get_frames_metadata(
    mjd: Union[str, int] = None, suffix: str = "fits", overwrite: bool = None
) -> pd.DataFrame:
    """Extract metadata from the 2d raw frames

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
    raw_frame = f"{mjd}/sdR*{suffix}*" if mjd else f"*/sdR*{suffix}*"
    frames = list(pathlib.Path(raw_data_path).rglob(raw_frame))

    metadata_paths = _get_metadata_paths(tileid="*", mjd=mjd, kind="raw", filter_exist=True)
    if any(metadata_paths) and not overwrite:
        log.info("Loading existing metadata store.")
        meta = get_metadata(mjd=mjd, tileid="*")
    else:
        if overwrite:
            _del_store(mjd=mjd, tileid="*")

        log.info("Creating new metadata store.")
        meta = extract_metadata(frames, kind="raw")

    return meta


def extract_metadata(frames_paths: list, kind: str = "raw") -> pd.DataFrame:
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
    # define target columns
    if kind == "raw":
        columns = RAW_METADATA_COLUMNS
    elif kind == "master":
        columns = MASTER_METADATA_COLUMNS
    else:
        pass
    # extract metadata
    nframes = len(frames_paths)
    if nframes == 0:
        log.warning("zero paths given, nothing to do")
        return pd.DataFrame(columns=[column for column, _ in columns])
    log.info(f"going to extract metadata from {nframes} frames")
    new_metadata = {}
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
            log.error(f"Cannot read FITS header of {frame_path}: {e}")
            continue

        frame_path = pathlib.Path(frame_path)

        # convert real MJD to SJD
        sjd = int(dateobs_to_sjd(header.get("OBSTIME")))
        sjd = correct_sjd(frame_path, sjd)

        # apply any header fix or if none, use old header
        header = apply_hdrfix(sjd, hdr=header) or header

        # set on-lamp conditions
        onlamp = ["ON", True, 'T', 1]

        # get the tile id; set null tile ids -999 to 11111
        tileid = header.get("TILE_ID") or header.get("TILEID", 11111)
        tileid = 11111 if tileid in (-999, 999, None) else tileid

        # get the tile group
        tilegrp = tileid_grp(tileid)

        if kind == "raw":
            new_metadata[i] = [
                "n" if header.get("OBSERVAT") != "LCO" else "s",
                tileid,
                sjd,
                header.get("MJD"),
                header.get("IMAGETYP"),
                header.get("SPEC"),
                header.get("CCD"),
                header.get("EXPOSURE"),
                header.get("EXPTIME"),
                header.get("NEON", "OFF") in onlamp,
                header.get("HGNE", "OFF") in onlamp,
                header.get("KRYPTON", "OFF") in onlamp,
                header.get("XENON", "OFF") in onlamp,
                header.get("ARGON", "OFF") in onlamp,
                header.get("LDLS", "OFF") in onlamp,
                header.get("QUARTZ", "OFF") in onlamp,
                header.get("HARTMANN", "0 0"),
                # header.get("QUALITY", "excellent"),
                # QC pipeline keywords
                header.get("QAQUAL", "GOOD"),
                # header.get("QAFLAG", QAFlag(0)),
                # DRP quality keywords
                header.get("DRPSTAGE", ReductionStage.UNREDUCED),
                header.get("DRPSTAT", ReductionStatus(0)),
                header.get("DRPQUAL", QualityFlag(0)),
                frame_path.stem,
                tilegrp,
            ]
        elif kind == "master":
            new_metadata[i] = [
                tileid,
                sjd,
                header.get("MJD"),
                header.get("IMAGETYP"),
                header.get("SPEC"),
                header.get("CCD"),
                header.get("EXPTIME"),
                header.get("NEON", "OFF") in onlamp,
                header.get("HGNE", "OFF") in onlamp,
                header.get("KRYPTON", "OFF") in onlamp,
                header.get("XENON", "OFF") in onlamp,
                header.get("ARGON", "OFF") in onlamp,
                header.get("LDLS", "OFF") in onlamp,
                header.get("QUARTZ", "OFF") in onlamp,
                header.get("HARTMANN", "0 0"),
                # TODO: QUALITY may be redundant, double check and remove if it is
                # header.get("QUALITY", "excellent"),
                header.get("QAQUAL", "GOOD"),
                # header.get("QAFLAG", QAFlag(0)),
                header.get("DRPSTAGE", ReductionStage.UNREDUCED),
                header.get("DRPSTAT", ReductionStatus(0)),
                header.get("DRPQUAL", QualityFlag(0)),
                header.get("NFRAMES", 1),
                frame_path.stem,
                tilegrp,
            ]

    # define dataframe
    new_metadata = pd.DataFrame.from_dict(new_metadata, orient="index")
    new_metadata.columns = list(zip(*columns))[0]

    # store metadata in HDF5 store
    if kind == "raw":
        add_raws(new_metadata)
    elif kind == "master":
        add_masters(new_metadata)

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
                if dtypes[n] is not np.dtype("O")
                else (n, h5py.string_dtype("utf-8", length=None))
                for n in dtypes.names
            ]
        )

        if "raw" in store:
            dataset = store["raw"]
            nolds = len(dataset)

            # filter out existing frames using the paths
            old_paths = dataset["name"].astype(str)
            new_paths = array["name"].astype(str)
            fil_paths = ~np.isin(new_paths, old_paths)
            array = array[fil_paths]
            nnews = len(array)

            log.info(
                f"updating metadata store for {tileid = } and {mjd = } "
                f"with {nnews} new rows"
            )

            if nnews > 0:
                dataset.resize(nolds + nnews, axis=0)
                dataset[-nnews:] = array
            log.info(f"final number of rows {nolds + nnews}")
        else:
            nnews = len(array)
            log.info(
                f"creating metadata store for {tileid = } and {mjd = } "
                f"with {nnews} new rows"
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
            if dtypes[n] is not np.dtype("O")
            else (n, h5py.string_dtype("utf-8", length=None))
            for n in dtypes.names
        ]
    )

    if "master" in store:
        dataset = store["master"]
        nolds = len(dataset)

        # filter out existing frames using the paths
        old_paths = dataset["name"].astype(str)
        new_paths = array["name"].astype(str)
        fil_paths = ~np.isin(new_paths, old_paths)
        array = array[fil_paths]
        nnews = len(array)

        log.info(f"updating metadata store for masters with {nnews} new rows")
        if nnews > 0:
            dataset.resize(nolds + nnews, axis=0)
            dataset[-nnews:] = array
        log.info(f"final number of rows {nolds+nnews}")
    else:
        nnews = len(array)
        log.info(f"creating metadata store for masters with {nnews} new rows")
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
    drpver=None,
    tileid=None,
    mjd=None,
    rmjd=None,
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
    drpver : str, optional
        DRP version, by default None (current version)
    tileid : int, optional
        tile ID of the target frames, by default None
    mjd : int, optional
        SJD where the target frames is located, by default None
    rmjd : int, optional
        the real MJD, by default None
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

    # define DRP version
    drpver = drpver or DRPVER

    # default output
    default_output = pd.DataFrame(
        columns=list(zip(*RAW_METADATA_COLUMNS))[0]
        if kind == "raw"
        else list(zip(*MASTER_METADATA_COLUMNS))[0]
    )

    # extract metadata
    stores = _load_or_create_store(drpver=drpver, tileid=tileid, mjd=mjd, kind=kind, mode="r")

    metadatas = []
    for store in stores:
        if kind not in store:
            log.warning(f"no metadata found of {kind = }, {tileid = } and {mjd = }")
            return default_output
        else:
            dataset = store[kind]

        # extract metadata as dataframe
        metadata = pd.DataFrame(dataset[()])

        # close store
        store.close()

        # convert bytes to literal strings
        metadata = _decode_string(metadata)

        metadatas.append(metadata)

    if not metadatas:
        return

    metadata = pd.concat(metadatas, axis="index", ignore_index=True)
    log.info(f"found {len(metadata)} frames in stores")
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

    return metadata


def get_sequence_metadata(mjd, expnums=None, exptime=None, cameras=CAMERAS, for_cals={"bias", "trace", "wave", "dome", "twilight"}, extract_metadata=False):
    """Get frames metadata for a given sequence

    Given a set of MJDs and (optionally) exposure numbers, get the frames
    metadata for the given sequence. This routine will return the frames
    metadata for the given MJDs and the MJD for the master frames.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    expnums : list
        List of exposure numbers to reduce
    exptime : int
        Filter frames metadata by exposure
    cameras : list
        List of cameras (e.g., "b1", "r3") to filter by
    for_cals : list, tuple or set, optional
        Only return frames meant to produce given calibrations, {'bias', 'trace', 'wave', 'dome', 'twilight'}
    extract_metadata : bool
        Whether to extract metadata or not, by default False

    Returns:
    -------
    frames : pd.DataFrame
        Frames metadata
    masters_mjd : float
        MJD for master frames
    """
    # get frames metadata
    frames = get_frames_metadata(mjd=mjd, overwrite=extract_metadata)
    frames.query("imagetyp in ['bias', 'flat', 'arc']", inplace=True)
    if len(frames) == 0:
        log.error(f"no calibration frames found for MJD = {mjd}")
        return frames, {}

    # filter by given expnums
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)

    # filter by given exptime
    if exptime is not None:
        frames.query("exptime == @exptime", inplace=True)

    if cameras:
        frames.query("camera in @cameras", inplace=True)

    # simple fix of imagetyp, some images have the wrong type in the header
    bias_selection = (frames.imagetyp == "bias")
    twilight_selection = (frames.imagetyp == "flat") & ~(frames.ldls|frames.quartz) & ~(frames.neon|frames.hgne|frames.argon|frames.xenon)
    domeflat_selection = (frames.ldls|frames.quartz) & ~(frames.neon|frames.hgne|frames.argon|frames.xenon)
    arc_selection = (frames.neon|frames.hgne|frames.argon|frames.xenon) & ~(frames.ldls|frames.quartz)
    frames.loc[twilight_selection, "imagetyp"] = "flat"
    frames.loc[domeflat_selection, "imagetyp"] = "flat"
    frames.loc[arc_selection, "imagetyp"] = "arc"

    found_cals = {'bias', 'trace', 'wave', 'dome', 'twilight'}
    if bias_selection.sum() == 0 and "bias" in for_cals:
        log.error("no bias exposures found")
        found_cals.remove("bias")
    elif domeflat_selection.sum() == 0 and "trace" in for_cals:
        log.error("no dome flat exposures found")
        found_cals.remove("trace")
    elif arc_selection.sum() == 0 and "wave" in for_cals:
        log.error("no arc exposures found")
        found_cals.remove("wave")
    elif domeflat_selection.sum() == 0 and "dome" in for_cals:
        log.error("no dome flat exposures found")
    elif twilight_selection.sum() == 0 and "twilight" in for_cals:
        log.error("no twilight exposures found")
        found_cals.remove("twilight")

    frames.sort_values(["expnum", "camera"], inplace=True)

    return frames, found_cals


# TODO: implement matching of analogs and calibration masters
# in Brian's run_drp code
def get_analog_groups(
    tileid,
    mjd,
    rmjd=None,
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
    include_fields=[],
    analog_fields=[],
):
    """return a list of metadata groups considered to be analogs

    the given metadata dataframe is grouped in analog frames using
    the following criteria:
        * mjd
        * imagetyp
        * camera
    Optionally, these criteria can be expanded using the `include_fields`.
    The final critaria will always include the above mentioned fields.

    Parameters
    ----------
    tileid : int
        tile ID of the target frames
    mjd : int
        SJD where the target frames is located
    rmjd : int
        the real MJD
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
    include_fields : list, optional
        a list of additional fields to include when grouping analogs, by default []

    Returns
    -------
    dict_like
        the grouped metadata filtered following the given criteria
    dict_like
        the grouped analog frame paths to be combined into masters
    list_like
        the list of master paths
    """
    # default fields
    default_fields = ["tileid", "mjd", "imagetyp", "camera"]
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
            rmjd=rmjd,
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

    metadata = pd.concat(metadatas, axis="index", ignore_index=True)

    log.info("grouping analogs")
    include_fields = default_fields + list(
        filter(lambda i: i not in default_fields, include_fields)
    )
    metadata_groups = metadata.groupby(include_fields)

    log.info(f"found {len(metadata_groups)} groups of analogs:")
    analogs = {}
    analog_paths = {}
    master_paths = []
    for g in metadata_groups.groups:
        log.info(f"{g}")

        # create groups dictionary
        metadata = metadata_groups.get_group(g)
        analogs[g] = metadata

        # create input paths for master creation task
        analog_paths[g] = [
            path.full(
                "lvm_anc",
                drpver=DRPVER,
                tileid=row.tileid,
                mjd=row.mjd,
                kind="c" if row.imagetyp != "bias" else "p",
                imagetype="fiberflat" if row.imagetyp == "flat" else row.imagetyp,
                camera=row.camera,
                expnum=row.expnum,
            )
            for _, row in metadata.iterrows()
        ]

        # create output path for master frame
        row = metadata.iloc[0]
        master_paths.append(create_master_path(row))

    return analogs, analog_paths, master_paths


def create_master_path(row: pd.Series) -> str:
    """Construct the path to a master frame

    Parameters
    ----------
    row : pd.Series
        A pandas row of metadata

    Returns
    -------
    str
        A fully resolved path to the master frame
    """
    return path.full("lvm_master", drpver=DRPVER, kind=f'm{row.imagetyp}',
                     tileid=row.tileid, mjd=row.mjd, camera=row.camera)


def match_master_metadata(
    target_mjd,
    target_imagetyp,
    target_camera,
    hemi=None,
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
    """return the matched master calibration frames given a target frame metadata

    Depending on the type of the target frame, a set of calibration frames may
    be needed. These are stored in lvmdrp.core.constants.FRAMES_CALIB_NEEDS.
    This function retrieves the closest in time set of calibration frames
    according to that mapping.

    Parameters
    ----------
    target_mjd : int
        SJD where the target frames is located
    target_imagetyp : str
        type/flavor of frame to locate `IMAGETYP`
    target_camera : str
        camera ID of the target frames
    hemi : str, optional
        hemisphere where the target frames were taken, by default None
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
        bitmask representing quality of the reduction, by default None

    Returns
    -------
    dict_like
        a dictionary containing the matched master calibration frames
    """
    # normalize flat flavor
    if target_imagetyp in {"fiberflat", "flat"}:
        target_imagetyp = "flat"

    # locate calibration needs
    frame_needs = FRAMES_CALIB_NEEDS.get(target_imagetyp)
    log.info(
        f"target frame of type '{target_imagetyp}' "
        f"needs calibration frames: {', '.join(frame_needs) or None}"
    )
    # initialize master calibration matches
    calib_frames = dict.fromkeys(frame_needs)

    # extract master calibration frames metadata
    store = _load_or_create_store(kind="master", mode="r")
    if not store:
        log.warning("No master store found.")
        return {}
    store = store[0]

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
    masters_metadata = _decode_string(masters_metadata.copy())

    log.info(f"found {len(masters_metadata)} master frames in store")

    # filter by exposure number, spectrograph and/or camera
    log.info(f"final number of master frames after filtering {len(masters_metadata)}")

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
            imagetyp=calib_type,
            camera=target_camera,
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
        if len(calib_metadata) == 0:
            log.warning(f"no master {calib_type} frame found")
        else:
            log.info(f"found master {calib_type}")
            # keep calibration frames with the closest MJD to target_mjd
            calib_frames[calib_type] = calib_metadata.iloc[
                np.argmin(np.abs(calib_metadata.mjd - target_mjd))
            ]

    return calib_frames


# TODO: implement update of reduction status
# in Brian's run_drp code
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


def _collect_header_data(filename: str) -> dict:
    """ Collect the relevant header information from a file

    Get the relevant header keys from the lvmSFrame file.  Remaps
    some of the keys into new, cleaner column names for the summary
    file.

    Parameters
    ----------
    filename : str
        the lvmSFrame filename

    Returns
    -------
    dict
        the extracted header key/values
    """
    hdr_dict_mapping = {'drpver': 'DRPVER', 'drpqual': 'DRPQUAL', 'dpos': 'DPOS', 'object': 'OBJECT',
                        'obstime': 'OBSTIME',
                        # sci
                        'sci_ra': 'TESCIRA', 'sci_dec': 'TESCIDE', 'sci_amass': 'TESCIAM',
                        'sci_kmpos': 'TESCIKM', 'sci_focpos': 'TESCIFO', 'sci_alt: SKY SCI_ALT',
                        'sci_sh_hght': 'SKY SCI_SH_HGHT', 'sci_moon_sep': 'SKY SCI_MOON_SEP',
                        # skye
                        'skye_ra': 'TESKYERA', 'skye_dec': 'TESKYEDE', 'skye_amass': 'TESKYEAM',
                        'skye_kmpos': 'TESKYEKM', 'skye_focpos': 'TESKYEFO', 'skye_name': 'SKYENAME',
                        'skye_alt': 'SKY SKYE_ALT', 'sci_skye_sep': 'SKY SCI_SKYE_SEP',
                        'skye_sh_hght': 'SKY SKYE_SH_HGHT', 'skye_moon_sep': 'SKY SKYE_MOON_SEP',
                        # skyw
                        'skyw_ra': 'TESKYWRA', 'skyw_dec': 'TESKYWDE', 'skyw_amass': 'TESKYWAM',
                        'skyw_kmpos': 'TESKYWKM', 'skyw_focpos': 'TESKYWFO', 'skyw_name': 'SKYWNAME',
                        'skyw_alt': 'SKY SKYW_ALT', 'sci_skyw_sep': 'SKY SCI_SKYW_SEP',
                        'skyw_sh_hght': 'SKY SKYW_SH_HGHT', 'skyw_moon_sep': 'SKY SKYW_MOON_SEP',
                        # sky parameters
                        'moon_ra': 'SKY MOON_RA', 'moon_dec': 'SKY MOON_DEC',
                        'moon_phase': 'SKY MOON_PHASE', 'moon_fli': 'SKY MOON_FLI',
                        'sun_alt': 'SKY SUN_ALT', 'moon_alt': 'SKY MOON_ALT'
                        }

    with fits.open(filename) as hdulist:
        hdr = hdulist['PRIMARY'].header

        hdrrow = {k: hdr.get(v) for k, v in hdr_dict_mapping.items()}

        return hdrrow


def update_summary_file(filename: str, tileid: int = None, mjd: int = None, expnum: int = None,
                        master_mjd: int = None, drpver: str = None):
    """ Update the DRPall summary file

    Update the LVM DRPall summary file with a new row of data for a given lvmSFrame file.
    This writes out the summary file as an HDF5 file using pandas built-in "to_hdf", which
    uses pytables.  This allows for efficient read/writes, updates, and handles file creation.

    Parameters
    ----------
    filename : str
        the lvmSFrame filepath
    tileid : int, optional
        the tileid of the exposure, by default None
    mjd : int, optional
        the mjd of the exposure, by default None
    expnum : int, optional
        the exposure number, by default None
    master_mjd : int, optional
        the master calibration MJD, by default None
    """
    # get DRP version
    drpver = drpver or DRPVER

    # get the row(s) from the raw frames metadata
    df = get_metadata(drpver=drpver, tileid=int(tileid), mjd=int(mjd), expnum=int(expnum), imagetyp='object')
    if df is None or df.empty:
        log.info(f'No metadata found for {tileid=}, {mjd=}, {expnum=}. Exiting.')
        return

    # select unique expnum row, i.e. remove duplicates from camera/spec rows
    # select a subset of columns from frames metadata
    row = df.drop_duplicates(['tileid', 'mjd', 'expnum']).reset_index(drop=True).sort_values(['mjd', 'expnum'])
    row = row[['tilegrp', 'tileid', 'mjd', 'expnum', 'exptime', 'stage', 'status', 'drpqual']]

    # collect header info
    hdr_data = _collect_header_data(filename)

    # add additional metadata
    # get SAS location and name
    location = path.location("lvm_frame", mjd=mjd, drpver=drpver, tileid=tileid, expnum=expnum, kind='SFrame')
    name = path.name("lvm_frame", mjd=mjd, drpver=drpver, tileid=tileid, expnum=expnum, kind='SFrame')
    gdr_location = path.location('lvm_agcam_coadd', mjd=mjd, specframe=expnum, tel='sci')
    hdr_data['filename'] = name
    hdr_data['location'] = location
    hdr_data['agcam_location'] = gdr_location
    hdr_data['calib_mjd'] = master_mjd
    hdr_data['drpver'] = drpver

    # add new columns
    df = row.assign(**hdr_data)

    # explicitly set some column dtypes to try and handle cases with null data
    # sci, skye, skye keys
    tels = {'sci', 'skye', 'skyw'}
    keys = {'ra', 'dec', 'amass', 'kmpos', 'focpos', 'sh_hght', 'moon_sep'}
    dtypes = {f'{i}_{j}': 'float64' for i, j in itertools.product(tels, keys)}
    dtypes.update({colname: 'float64' for colname in ('moon_ra', 'moon_dec', 'moon_phase', 'moon_fli', 'sun_alt', 'moon_alt')})
    dtypes['calib_mjd'] = 'int64'
    df = df.astype(dtypes)

    # replace empty strings in object column with None
    df['object'] = df['object'].fillna('None')
    df['skye_name'] = df['skye_name'].fillna('None')
    df['skyw_name'] = df['skyw_name'].fillna('None')
    # replace NaN values by invalid value -999 (NaNs will be casted to strings)
    df.fillna(-999, inplace=True)

    # create drpall h5 filepath
    drpall = path.full('lvm_drpall', drpver=drpver)
    drpall = drpall.replace('.fits', '.h5')
    # log.info(f'Updating the drpall summary file {drpall}')
    lock = FileLock(drpall.replace('.h5', '.h5.lock'), timeout=5)

    # set min column sizes for some columns
    min_itemsize = {'skye_name': 20, 'skyw_name': 20, 'location': 120, 'agcam_location': 120,
                    'object': 24, 'obstime': 23}

    # write to pytables hdf5
    try:
        with lock:
            df.to_hdf(drpall, key='summary', mode='a', append=True, data_columns=True, min_itemsize=min_itemsize)
    except ImportError:
        log.error('Missing pytables dependency. Install with `pip install "pandas[hdf5]"`. '
                      'On macs, you may first need to first run "brew install hdf5".')
    except Timeout:
        log.error("Another instance of the drp currently holds the drpall lock.")
    except ValueError as e:
        log.error(f"Error while updating drpall file: {e}")
        log.error(f"You may need to remove {drpall} and run this code again with flags -2d -1d -p1d")
    else:
        log.info(f'Updating drpall file {drpall}.')


def convert_h5_to_fits(h5file: str):
    """ Convert a pandas HDF5 file to a FITS file

    Convert the drpall hdf5 file to more astro-friendly
    FITS format.  This function is useful to run once
    the summary HDF5 file for the entire tagged DRP run
    has completed.

    Parameters
    ----------
    h5file : str
        the path to the h5 file
    """
    # read in the dataframe
    df = pd.read_hdf(h5file, key='summary')
    df = df.sort_values(['mjd', 'expnum'])
    df.reset_index(drop=True, inplace=True)
    df.to_hdf(h5file, key='summary', data_columns=True)

    # write FITS file
    fitsfile = h5file.replace('.h5', '.fits')
    table = Table.from_pandas(df)
    table.write(fitsfile, overwrite=True)


def extract_from_filename(filename: str | pathlib.Path) -> tuple:
    """ Extract metadata from a reduced frame filename

    Extract metadata from a reduced lvmXFrame filename.  This is a helper
    function to extract the tileid, mjd, and expnum from a filename.

    Parameters
    ----------
    filename : str
        the filename

    Returns
    -------
    tuple
        a tuple with the extracted metadata
    """
    path = pathlib.Path(filename)
    expnum = path.parts[-1].split('.')[0].split('-')[-1].lstrip('0')
    mjd = path.parts[-2]
    tileid = path.parts[-3]
    return tileid, mjd, expnum
