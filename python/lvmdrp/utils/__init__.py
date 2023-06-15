import collections
import collections.abc
import os
import site
import sys

import numpy as np


def gaussian(x, mean=0, stddev=1):
    return np.exp(-0.5 * (x - mean) ** 2 / stddev**2) / np.sqrt(2 * np.pi) / stddev


def spec_from_lines(lines, sigma, wavelength, heights=None, names=None):
    rss = np.zeros((len(lines), wavelength.size))
    for i, line in enumerate(lines):
        rss[i] = gaussian(wavelength, mean=line, stddev=sigma)
    if heights is not None:
        rss * heights[None]
    return rss.sum(axis=0)


def flatten(iterable):
    for el in iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            yield from flatten(el)
        else:
            yield el


def get_input_path(pattern, mjd, label, paths):
    for path in paths:
        full_path = pattern.format(input_path=path, mjd=mjd, label=label)
        if os.path.isfile(full_path):
            break
    else:
        return None
    return full_path


def get_calibration_paths(calib_metadata, path, kinds=None):
    if kinds is None:
        calib_paths = dict.fromkeys(calib_metadata.keys())
        for calib_type, metadata in calib_metadata.items():
            if metadata.LABEL is None:
                continue
            calib_paths[calib_type] = os.path.join(path, f"{metadata.LABEL}.fits")
    else:
        calib_paths = dict.fromkeys(calib_metadata.keys())
        for calib_type, metadata in calib_metadata.items():
            if metadata.LABEL is None:
                continue
            calib_paths[calib_type] = {
                os.path.join(path, f"{metadata.LABEL}.{kind}.fits")
                for kind in kinds[calib_type]
            }
            calib_paths
    return calib_paths


def get_master_name(label, image_type, mjd):
    master_name = label.split("-")
    master_name[-1] = str(mjd)
    master_name = "-".join(master_name)
    return f"{master_name}.m{image_type.upper()}"


def dict_update(d, u):
    """returns the updated version of a nested dictionary

    NOTE: taken from https://bit.ly/3x75Wg9
    """
    if not d:
        d = {}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def rc_symlink(src, dst):
    """Forces creation of symbolic link if already exists"""
    # create directory if does not exist
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # bypass existing link
    if os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)
    return None


def get_env_lib_directory():
    """Return the installation directory, or None

    taken from: https://bit.ly/3BAOpQK
    """
    if "--user" in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        py_version = "%s.%s" % (sys.version_info[0], sys.version_info[1])
        paths = (
            s % (py_version)
            for s in (
                sys.prefix + "/lib/python%s/dist-packages/",
                sys.prefix + "/lib/python%s/site-packages/",
                sys.prefix + "/local/lib/python%s/dist-packages/",
                sys.prefix + "/local/lib/python%s/site-packages/",
                "/Library/Python/%s/site-packages/",
            )
        )

    for path in paths:
        if os.path.exists(path):
            return os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(path)))
            )

    return None
