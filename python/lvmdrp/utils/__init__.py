
from .configuration import *
from .logger import *
import os
import collections.abc


def get_input_path(pattern, mjd, label, paths):
    for path in paths:
        full_path = pattern.format(input_path=path, mjd=mjd, label=label)
        if os.path.isfile(full_path): break
    else:
        return None
    return full_path

def get_calibration_paths(calib_metadata, path, kinds=None):
    if kinds is None:
        calib_paths = dict.fromkeys(calib_metadata.keys())
        for calib_type, metadata in calib_metadata.items():
            if metadata.LABEL is None: continue
            calib_paths[calib_type] = os.path.join(path, f"{metadata.LABEL}.fits")
    else:
        calib_paths = dict.fromkeys(calib_metadata.keys())
        for calib_type, metadata in calib_metadata.items():
            if metadata.LABEL is None: continue
            calib_paths[calib_type] = {os.path.join(path, f"{metadata.LABEL}.{kind}.fits") for kind in kinds[calib_type]}
            calib_paths
    return calib_paths

def get_master_name(label, image_type, mjd):
    master_name = label.split("-")
    master_name[0] = str(mjd)
    master_name = "-".join(master_name)
    return f"{master_name}.m{image_type.upper()}"

def dict_update(d, u):
    """returns the updated version of a nested dictionary

    NOTE: taken from https://bit.ly/3x75Wg9
    """
    if not d: d = {}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
