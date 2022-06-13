# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: database.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

# NOTE: to start using a database, first run these commands in the mysql console as superuser:
#       CREATE USER 'sammy'@'localhost' IDENTIFIED BY 'password';
#       GRANT ALL PRIVILEGES ON *.* TO 'sammy'@'localhost' WITH GRANT OPTION;
#       FLUSH PRIVILEGES;


import os
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
from argparse import Namespace
import mysql.connector as mysql
from astropy.io import fits

from lvmdrp.core.constants import ARC_FIELDS, CALIBRATION_TYPES, CONT_FIELDS, FIELDS_TO_SKIP, FRAMES_PRIORITY, FRAMES_CALIB_NEEDS, ALL_DTYPES, RAW_NAMES, ALL_DEFAULTS, CALIBRATION_NAMES, CREATE_RAW_FRAMES, CREATE_CALIBRATION_FRAMES
from lvmdrp.utils.bitmask import ReductionStatus, QualityFlag


ALL_CONVERTERS = dict.fromkeys(RAW_NAMES, lambda v: v)
ALL_CONVERTERS["status"] = lambda v: ReductionStatus(v)
ALL_CONVERTERS["flags"] = lambda v: QualityFlag(v)


def _parse_db_data(data, column_names):
    _column_names = list(map(str.upper, column_names))
    if isinstance(data, list):
        metadata = []
        for j in range(len(data)):
            data_j = list(data[j])
            for i, name in enumerate(column_names):
                data_j[i] = ALL_CONVERTERS[name](data_j[i])
            metadata.append(Namespace(**dict(zip(_column_names, data_j))))
    elif isinstance(data, tuple):
        metadata = Namespace(**dict(zip(_column_names, data)))
    else:
        raise ValueError(f"unexpected data type '{type(data)}'")
    return metadata

def create_db(config):
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME

    try:
        with mysql.connect(host=_host, user=_user, password=_pass) as db:
            sql = f"CREATE DATABASE {_name}"
            # create new DB
            with db.cursor() as cursor:
                cursor.execute(sql)
    except mysql.Error as e:
        print(e)
    return None

def create_tables_db(config):
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME

    try:
        with mysql.connect(host=_host, user=_user, password=_pass, database=_name) as db:
            # create new tables
            with db.cursor() as cursor:
                for sql in [CREATE_CALIBRATION_FRAMES, CREATE_RAW_FRAMES]:
                    cursor.execute(sql)
                db.commit()
    except mysql.Error as e:
        print(e)
    return None

def delete_tables_db(config):
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME

    try:
        with mysql.connect(host=_host, user=_user, password=_pass, database=_name) as db:
            # create new tables
            with db.cursor() as cursor:
                cursor.execute(f"DROP TABLE {', '.join(['CALIBRATION_FRAMES', 'RAW_FRAMES'])}")
            db.commit()
    except mysql.Error as e:
        print(e)
    return None

def delete_db(config):
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME
    try:
        with mysql.connect(host=_host, user=_user, password=_pass, database=_name) as db:
            with db.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS {_name}")
            db.commit()
    except mysql.Error as e:
        print(e)
    return None

def record_db(config, target_paths=None, ignore_cache=False):
    if target_paths is None:
        target_paths = config.RAW_DATA_PATHS
    # prepare fields skipping 'id' and 'datetime'
    _metadata_fields = list(filter(lambda field: field not in FIELDS_TO_SKIP, RAW_NAMES))
    _metadata_dtypes = [ALL_DTYPES[field] for field in _metadata_fields]
    _lamps = list(ARC_FIELDS.keys())+list(CONT_FIELDS.keys())
    # extract records from frames header
    if os.path.isfile(config.DB_PATH) and not ignore_cache:
        metadata = pickle.load(open(config.DB_PATH, "rb"))
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
                        # if len(metadata) >= 10: break
                        try:
                            header = fits.getheader(new_frame_path, ext=0)
                        except OSError as e:
                            print(f"{new_frame_path}: {e}")
                            continue
                        # update/add metadata keywords
                        for key in _lamps:
                            header[key.upper()] = True if header.get(key.upper()) == "ON" else False
                        header["LABEL"] = new_frame_label
                        header["PATH"] = new_frame_path
                        header["STATUS"] = ReductionStatus["RAW"]
                        header["FLAGS"] = QualityFlag["OK"]

                        # create new record, but ignore default columns 'id' and 'datetime'
                        record = list(header.get(k.upper(), ALL_DEFAULTS[k]) for k in _metadata_fields)
                        # update status in case there are missing metadata with the exception of those fields that are allowed to be NULL
                        nonnull_values = [record[i] for i in range(len(record)) if "NULL" not in _metadata_dtypes[i]]
                        record[_metadata_fields.index("flags")] += "MISSING_METADATA" if None in nonnull_values else "OK"
                        # add record to DB
                        metadata.append(record)
        pickle.dump(metadata, open(config.DB_PATH, "wb"))

    # add new records in DB
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME

    RECORD_RAW_FRAMES = f"""INSERT IGNORE INTO RAW_FRAMES
    ({','.join(_metadata_fields)})
    VALUES ({','.join(len(_metadata_fields)*['%s'])})"""
    try:
        with mysql.connect(host=_host, user=_user, password=_pass, database=_name) as db:
            with db.cursor() as cursor:
                cursor.executemany(RECORD_RAW_FRAMES, metadata)
            db.commit()
    except mysql.ProgrammingError as e:
        print(e)
        print(metadata)
    return None

def integrity_db(config):
    pass

def load_db(config):
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME
    try:
        db = mysql.connect(
            host=_host,
            user=_user,
            password=_pass,
            database=_name
        )
    except mysql.Error as e:
        print(e)

    return db

def get_raws_metadata(db):
    try:
        priorities = ",".join(map(repr, FRAMES_PRIORITY))
        with db.cursor() as cursor:
            cursor.execute(f"""
            SELECT *
            FROM RAW_FRAMES
            WHERE
                status = {ReductionStatus['RAW']} AND flags = {QualityFlag['OK']} AND imagetyp IN ({priorities})
            ORDER BY
                FIELD(imagetyp, {priorities}), mjd ASC
            """)
            all_data = cursor.fetchall()
    except mysql.Error as e:
        print(e)
    
    new_frames = _parse_db_data(data=all_data, column_names=RAW_NAMES)

    return new_frames

# BUG: make these functions return the input (potentially updated) metadata to avoid memory issues
def get_analogs_metadata(db, metadata):
    analogs_metadata = []
    if metadata.IMAGETYP in CALIBRATION_TYPES and metadata.MASTER_ID is None:
        try:
            with db.cursor(buffered=True) as cursor:
                cursor.execute(f"""
                SELECT *
                FROM RAW_FRAMES
                WHERE
                    imagetyp = '{metadata.IMAGETYP}' AND ccd = '{metadata.CCD}' AND mjd = '{metadata.MJD}' AND ABS(exptime-{metadata.EXPTIME}) <= 1e-6
                """)
                analog_data = cursor.fetchall()
        except mysql.Error as e:
            print(f"{metadata.IMAGETYP}: {e}")
        
        analogs_metadata = _parse_db_data(data=analog_data, column_names=RAW_NAMES)
        # set status in progress
        for analog_metadata in analogs_metadata:
            analog_metadata.STATUS += "IN_PROGRESS"
    return analogs_metadata

def get_calib_metadata(db, metadata):
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
    frame_needs = FRAMES_CALIB_NEEDS.get(metadata.IMAGETYP)
    calib_frames = dict.fromkeys(frame_needs, Namespace(**dict([(field.upper(), ALL_DEFAULTS[field]) for field in CALIBRATION_NAMES])))
    # handle empty list cases (e.g., bias)
    if frame_needs == []:
        return calib_frames
    # handle unrecognized frame type
    if frame_needs is None:
        raise ValueError(f"Unrecognized frame type '{metadata.IMAGETYP}'")

    # BUG: change sorting to use 'mjd' instead of 'obstime', since master will be represented by that parameter
    # BUG: remove 'exptime' constrain when looking for bias frames, since all of them have exptime = 0
    for calib_type in frame_needs:
        try:
            with db.cursor(buffered=True) as cursor:
                cursor.execute(f"""
                SELECT *
                FROM CALIBRATION_FRAMES
                WHERE
                    imagetyp = '{calib_type}' AND ccd = '{metadata.CCD}' AND ABS(exptime-{metadata.EXPTIME}) <= 1e-6
                ORDER BY ABS( DATEDIFF( obstime, '{metadata.OBSTIME}' ) )
                """)
                # TODO: handle the case in which the retrieved frame is stale and/or has quality flags
                # BUG: there may be cases in which no frame is found
                data = cursor.fetchone()
                if data is not None:
                    calib_frames[calib_type] = _parse_db_data(data=data, column_names=CALIBRATION_NAMES)
        except mysql.Error as e:
            print(f"{calib_type}: {e}")
    return calib_frames

def put_redux_state(db, metadata, table):
    try:
        with db.cursor() as cursor:
            if isinstance(metadata, Namespace):
                if metadata.STATUS == "IN_PROGRESS": metadata.REDUCTION_STARTED = datetime.now()
                elif metadata.STATUS == "FINISHED": metadata.REDUCTION_FINISHED = datetime.now()
                cursor.execute(f"""
                UPDATE IGNORE
                    {table}
                SET
                    status = '{metadata.STATUS}',
                    flags = '{metadata.FLAGS}',
                    reduction_started = '{metadata.REDUCTION_STARTED}',
                    reduction_finished = '{metadata.REDUCTION_FINISHED}'
                WHERE
                    id = {metadata.ID}
                """)
                db.commit()
            elif isinstance(metadata, list):
                for _ in metadata:
                    if _.STATUS == "IN_PROGRESS": _.REDUCTION_STARTED = datetime.now()
                    elif _.STATUS == "FINISHED": _.REDUCTION_FINISHED = datetime.now()
                cursor.executemany(f"""
                UPDATE IGNORE
                    {table}
                SET
                    status = %s,
                    flags = %s,
                    reduction_started = %s,
                    reduction_finished = %s
                WHERE
                    id = %s
                """, [(_.STATUS, _.FLAGS, _.REDUCTION_STARTED, _.REDUCTION_FINISHED, _.ID) for _ in metadata])
                db.commit()
            else:
                raise ValueError(f"unknown metadata type '{type(metadata)}'")
    except mysql.Error as e:
        print(f"{metadata.ID}: {e}")
    return None

def add_master(db, master_metadata, analogs_metadata):
    # BUG: there is DB overwriting happening in CALIBRATION_FRAMES table
    if master_metadata.STATUS == "IN_PROGRESS": master_metadata.REDUCTION_STARTED = datetime.now()
    elif master_metadata.STATUS == "FINISHED": master_metadata.REDUCTION_FINISHED = datetime.now()
    _metadata_fields, _metadata_values = zip(*[(field.lower(), value) for field, value in master_metadata.__dict__.items()])
    try:
        with db.cursor(buffered=True) as cursor:
            # insert new master
            cursor.execute(f"""
            INSERT IGNORE INTO CALIBRATION_FRAMES
                ({','.join(_metadata_fields)})
            VALUES
                ({','.join(len(_metadata_values)*['%s'])})
            """, _metadata_values)
            db.commit()

            # get last master id
            master_metadata.ID = cursor.lastrowid
            
            # update master reference in raw frames
            cursor.execute(f"""
            UPDATE
                RAW_FRAMES
            SET
                master_id = {master_metadata.ID}
            WHERE
                id IN ({','.join([str(analog_frame.ID) for analog_frame in analogs_metadata])})
            """
            )
            db.commit()
    except mysql.Error as e:
        print(e)
        print(master_metadata)
    return None

if __name__ == "__main__":
    from lvmdrp.main import load_master_config

    config = load_master_config()
    _host = config.LVM_DB_HOST
    _user = config.LVM_DB_USER
    _pass = config.LVM_DB_PASS
    _name = config.LVM_DB_NAME
    try:
        db = mysql.connect(
            host=_host,
            user=_user,
            password=_pass,
            database=_name
        )
    except mysql.Error as e:
        print(e)
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM RAW_FRAMES")
        r = cursor.fetchall()
        print(np.array(r), len(r))
    except mysql.Error as e:
        print(e)
