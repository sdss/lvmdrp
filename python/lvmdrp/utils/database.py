# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: database.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import pickle
import numpy as np
import datetime as dt
from tqdm import tqdm
from sqlite3 import Error
from peewee import *

from astropy.io import fits

from lvmsurveysim.utils.sqlite2astropy import *

from lvmdrp.core.constants import CONFIG_PATH
from lvmdrp.core.constants import FRAMES_PRIORITY, CALIBRATION_TYPES, FRAMES_CALIB_NEEDS
from lvmdrp.utils.bitmask import ReductionStatus, QualityFlag


SQLITE_MAX_VARIABLE_NUMBER = 32766
# TODO:
#   - add frame regions (overscan, prescan, science) to check if those are correct
# TODO: add table for DRP products, from preprocessed frames to final science-ready frames
# TODO: add weather table for data quality monitoring purposes

db = SqliteDatabase(None)

class FlagsField(IntegerField):
    def db_value(self, flags_val):
        if not isinstance(flags_val, QualityFlag):
            raise TypeError(f"Wrong type '{type(flags_val)}', '{QualityFlag}' instance expected")
        return super().adapt(flags_val.value)

    def python_value(self, db_val):
        return QualityFlag(db_val)

class StatusField(IntegerField):
    def db_value(self, status_val):
        if not isinstance(status_val, ReductionStatus):
            raise TypeError(f"Wrong type '{type(status_val)}', '{ReductionStatus}' instance expected")
        return super().adapt(status_val.value)

    def python_value(self, db_val):
        return ReductionStatus(db_val)

class BaseModel(Model):
    class Meta:
        database = db

class BasicMixin(Model):
    datetime = DateTimeField(default=dt.datetime.now)
    mjd = IntegerField(null=True)
    spec = CharField(null=True)
    ccd = CharField(null=True)
    exptime = FloatField(null=True)
    imagetyp = CharField(null=True)
    obstime = DateTimeField(null=True)
    observat = CharField(null=True)
    label = CharField(null=True)
    path = CharField(null=True)
    naxis1 = IntegerField(null=True)
    naxis2 = IntegerField(null=True)

class LabMixin(Model):
    ccdtemp1 = FloatField(null=True)
    ccdtemp2 = FloatField(null=True)
    presure = FloatField(null=True)
    labtemp = FloatField(null=True)
    labhumid = FloatField(null=True)

class ArcMixin(Model):
    argon = BooleanField(default=False)
    xenon = BooleanField(default=False)
    hgar = BooleanField(default=False)
    krypton = BooleanField(default=False)
    neon = BooleanField(default=False)
    hgne = BooleanField(default=False)

class ContMixin(Model):
    m625l4 = BooleanField(default=False)
    ffs = BooleanField(default=False)
    mi150 = BooleanField(default=False)
    ts = BooleanField(default=False)
    ldls = BooleanField(default=False)
    nirled = BooleanField(default=False)

class StatusMixin(Model):
    reduction_started = DateTimeField(null=True)
    reduction_finished = DateTimeField(null=True)
    status = StatusField(default=ReductionStatus["RAW"])
    flags = FlagsField(default=QualityFlag["OK"])

# TODO:
#   - turn this into a master calibration frame
#   - store normal calibration frames in 'lvm_frames' table
class CalibrationFrames(BaseModel, StatusMixin):
    id = IntegerField(primary_key=True)
    is_master = BooleanField(default=False)

    class Meta:
        table_name = "calibration_frames"

class LVMFrames(BaseModel, BasicMixin, LabMixin, ArcMixin, ContMixin, StatusMixin):
    id = IntegerField(primary_key=True)
    calib = ForeignKeyField(CalibrationFrames, backref="frames", null=True)

    class Meta:
        table_name = "lvm_frames"

# define auto columns
AUTO_COLUMNS = ["id", "calib", "datetime"]
# define mandatory columns
MANDATORY_COLUMNS = [name for name in BasicMixin._meta.columns if name not in AUTO_COLUMNS]
# define raw columns excluding auto columns
FRAME_COLUMNS = [name for name in LVMFrames._meta.columns if name not in AUTO_COLUMNS]
# define arc/continuum names using peewee model definitions
ARC_NAMES = list(ArcMixin._meta.columns.keys())
CON_NAMES = list(ContMixin._meta.columns.keys())
ARC_NAMES.remove("id")
CON_NAMES.remove("id")
LAMP_NAMES = ARC_NAMES + CON_NAMES
# define calibration table columns excluding auto columns
CALIBRATION_COLUMNS = [name for name in CalibrationFrames._meta.columns if name not in AUTO_COLUMNS]

def create_or_connect_db(config):
    try:
        db.init(os.path.join(CONFIG_PATH, config.LVM_DB_NAME))
        db.connect()
        db.create_tables([LVMFrames, CalibrationFrames])
    except Error as e:
        print(e)
    return db

def delete_tables_db(config):
    try:
        db.init(os.path.join(CONFIG_PATH, config.LVM_DB_NAME))
        db.connect()
        db.drop_tables([LVMFrames, CalibrationFrames])
    except Error as e:
        print(e)
    return None

def delete_db(config):
    os.remove(os.path.join(CONFIG_PATH, config.LVM_DB_NAME))
    return None

def record_db(config, target_paths=None, ignore_cache=False):
    if target_paths is None: target_paths = config.RAW_DATA_PATHS
    # extract records from frames header
    if os.path.isfile(config.DB_PATH) and not ignore_cache:
        metadata = pickle.load(open(config.DB_PATH, "rb"))
        with db.atomic():
            for i, batch in enumerate(chunked(metadata, n=SQLITE_MAX_VARIABLE_NUMBER//len(metadata[0]))):
                try:
                    LVMFrames.insert_many(batch).execute()
                except Error as e:
                    print(e)
                    print(f"in chunk={i}, {batch}")
    else:
        metadata = []
        for target_path in target_paths:
            for root, _, frames in os.walk(target_path):
                tqdm.write(f"exploring path '{os.path.basename(root)}'")
                for frame in tqdm(frames, total=len(frames), desc="extracting metadata", ascii=True, unit="frame"):
                    new_frame_path = os.path.join(root, frame)
                    new_frame_label = frame.replace(".fits.gz", "")
                    if frame.endswith(".fits.gz"):
                        # NOTE: remove this once testing is finished
                        # if len(metadata) >= 1000: break
                        try:
                            header = fits.getheader(new_frame_path, ext=0)
                        except OSError as e:
                            print(f"{new_frame_path}: {e}")
                            continue
                        # update/add metadata keywords
                        for key in LAMP_NAMES:
                            header[key] = True if header.get(key) == "ON" else False
                        
                        # BUG: fix imagetyp key because the header is messed up. This will be done only for lab data (hopefully!)
                        if header.get("IMAGETYP") and header["IMAGETYP"] in ["continuum", "arc", "object"]:
                            lamps = [lamp for lamp in LAMP_NAMES if header[lamp]]
                            if lamps:
                                has_cont = np.isin(CON_NAMES, lamps)
                                has_arcs = np.isin(ARC_NAMES, lamps)
                                if header["IMAGETYP"] == "continuum" or has_cont.any() or has_arcs.all():
                                    header["IMAGETYP"] = "continuum"
                                elif header["IMAGETYP"] == "arc" or (not has_cont.any() and has_arcs.any() and not has_arcs.all()):
                                    header["IMAGETYP"] = "arc"
                                else:
                                    raise ValueError(f"unrecognized case for lamps: '{lamps}'.")
                            else:
                                header["IMAGETYP"] = "object"
                        
                        header["LABEL"] = new_frame_label
                        header["PATH"] = new_frame_path
                        header["STATUS"] = ReductionStatus["RAW"]
                        header["FLAGS"] = QualityFlag["OK"]

                        record = {key: header.get(key, LVMFrames._meta.columns[key].default) for key in FRAME_COLUMNS}
                        # update status in case there are missing metadata with the exception of those fields that are allowed to be NULL
                        nonnull_values = [record[name] for name in MANDATORY_COLUMNS]
                        record["flags"] += "MISSING_METADATA" if None in nonnull_values else "OK"
                        # append to list for cache
                        metadata.append(record)
        # store cache
        pickle.dump(metadata, open(config.DB_PATH, "wb"))
        # add record to DB
        with db.atomic():
            for i, batch in enumerate(chunked(metadata, n=SQLITE_MAX_VARIABLE_NUMBER//len(metadata[0]))):
                try:
                    LVMFrames.insert_many(batch).execute()
                except Error as e:
                    print(e)
                    print(f"in chunk={i}, {batch}")
    return None

def get_raws_metadata():
    try:
        priority = Case(LVMFrames.imagetyp, tuple((frame_type, i) for i, frame_type in enumerate(FRAMES_PRIORITY)))
        query = LVMFrames.select().where(
            (LVMFrames.status == ReductionStatus["RAW"]) &
            (LVMFrames.flags == QualityFlag["OK"]) &
            (LVMFrames.imagetyp << FRAMES_PRIORITY)
        ).order_by(priority)
    except Error as e:
        print(e)
    
    new_frames = [new_frame for new_frame in query]
    return new_frames

def get_calib_metadata():
    try:
        query = LVMFrames.select().where(
            (ReductionStatus["PREPROCESSED"] << LVMFrames.status) &
            (LVMFrames.flags == QualityFlag["OK"]) &
            (LVMFrames.imagetyp << CALIBRATION_TYPES)
        )
    except Error as e:
        print(e)
    
    calibration_metadata = [calib_metadata for calib_metadata in query]
    return calibration_metadata

def get_analogs_metadata(metadata):
    # define empty metadata in case current frame has already a master
    analogs_metadata = []
    if metadata.imagetyp in CALIBRATION_TYPES and metadata.calib_id is None:
        try:
            query = LVMFrames.select().where(
                (LVMFrames.calib_id == None) &
                (LVMFrames.imagetyp == metadata.imagetyp) &
                (LVMFrames.ccd == metadata.ccd) &
                (LVMFrames.mjd == metadata.mjd) &
                (LVMFrames.exptime == metadata.exptime)
            )
        except Error as e:
            print(f"{metadata.imagetyp}: {e}")
        
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
    if frame_needs is None: raise ValueError(f"Unrecognized frame type '{metadata.imagetyp}'")
    
    calib_frames = dict.fromkeys(frame_needs)
    # handle empty list cases (e.g., bias)
    if not frame_needs:
        return calib_frames
    # handle unrecognized frame type

    for calib_type in frame_needs:
        try:
            query = CalibrationFrames.select().where(
                (CalibrationFrames.imagetyp == calib_type) &
                (CalibrationFrames.ccd == metadata.ccd)
            ).order_by(fn.ABS(metadata.mjd - CalibrationFrames.mjd).asc())
        except Error as e:
            print(f"{calib_type}: {e}")
        
        # TODO: handle the case in which the retrieved frame is stale and/or has quality flags
        # BUG: there may be cases in which no frame is found
        # BUG: this is retrieving only the first (closest) calibration frame, not necessarily the best
        #      Should retrieve all possible calibration frames & decide which one is the best based on
        #      quality
        calib_frame = query.get_or_none()
        if calib_frame is not None:
            calib_frames[calib_type] = calib_frame
    return calib_frames

def put_redux_state(metadata, status=None):
    if status is not None:
        if isinstance(status, str):
            metadata.status = ReductionStatus[status]
        elif isinstance(status, int):
            metadata.status = ReductionStatus(status)
        elif isinstance(status, ReductionStatus):
            metadata.status = status
        else:
            ValueError(f"unknown status type '{type(status)}'")
    try:
        if isinstance(metadata, (LVMFrames, CalibrationFrames)):
            if metadata.status == "IN_PROGRESS": metadata.reduction_started = dt.datetime.now()
            elif metadata.status in ["FINISHED", "FAILED"]: metadata.reduction_finished = dt.datetime.now()
            metadata.save()
        elif isinstance(metadata, list):
            for md in metadata:
                if md.status == "IN_PROGRESS": md.reduction_started = dt.datetime.now()
                elif md.status in ["FINISHED", "FAILED"]: md.reduction_finished = dt.datetime.now()
                md.save()
        else:
            raise ValueError(f"unknown metadata type '{type(metadata)}'")
    except Error as e:
        print(e)
    return metadata

def add_calib(calib_metadata, status=None):
    if status is not None:
        if isinstance(status, str):
            calib_metadata.status = ReductionStatus[status]
        elif isinstance(status, int):
            calib_metadata.status = ReductionStatus(status)
        elif isinstance(status, ReductionStatus):
            calib_metadata.status = status
        else:
            ValueError(f"unknown status type '{type(status)}'")
    
    if calib_metadata.status == "IN_PROGRESS": calib_metadata.reduction_started = dt.datetime.now()
    elif calib_metadata.status in ["FINISHED", "FAILED"]: calib_metadata.reduction_finished = dt.datetime.now()
    try:
        calib_metadata.save()
    except Error as e:
        print(e)
        print(calib_metadata)
    return calib_metadata

def add_master(master_metadata, analogs_metadata, status=None):
    if status is not None:
        if isinstance(status, str):
            master_metadata.status = ReductionStatus[status]
        elif isinstance(status, int):
            master_metadata.status = ReductionStatus(status)
        elif isinstance(status, ReductionStatus):
            master_metadata.status = status
        else:
            ValueError(f"unknown status type '{type(status)}'")
    
    if master_metadata.status == "IN_PROGRESS": master_metadata.reduction_started = dt.datetime.now()
    elif master_metadata.status in ["FINISHED", "FAILED"]: master_metadata.reduction_finished = dt.datetime.now()
    try:
        master_metadata.save()
        for analog_metadata in analogs_metadata:
            analog_metadata.calib_id = master_metadata.id
            analog_metadata.save()
    except Error as e:
        print(e)
        print(master_metadata)
    return master_metadata


if __name__ == "__main__":
    from lvmdrp.main import load_master_config

    
    config = load_master_config()
    db = create_or_connect_db(config)
    
    new_frames = get_raws_metadata()
    for new_frame in new_frames:
        print(new_frame.label, new_frame.flags.get_name())