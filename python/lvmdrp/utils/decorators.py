# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: decorators.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import inspect
from functools import wraps
from typing import List

from astropy.io import fits

from lvmdrp import log
from lvmdrp.utils.bitmask import QualityFlag


def skip_on_missing_input_path(input_file_args: list, reset_missing_optionals: bool = True):
    """decorator to skip a task if any of the input files is missing

    Parameters
    ----------
    input_file_args : list
        list of input arguments corresponding to the input file paths

    Returns
    -------
    function
        decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, name in enumerate(input_file_args):
                # skip argument if not present in kwargs
                if name not in kwargs:
                    continue
                # get function parameters
                pars = inspect.signature(func).parameters
                # silently continue if input file is optional, is set to None and has default None
                if kwargs[name] is None and pars[name].default is None:
                    continue
                # warning for optional input files that are missing
                elif kwargs[name] is not None and pars[name].default is None and not os.path.isfile(kwargs[name]):
                    log.warning(f"optional input {name} = '{kwargs[name]}' at {func.__name__} is missing")
                    if reset_missing_optionals:
                        kwargs[name] = None
                    continue
                # skip task if input file is required and is set to None
                elif kwargs[name] is None and pars[name].default == inspect._empty:
                    log.error(f"required input {name} is set to None at {func.__name__}")
                    return
                # skip task if input file is missing
                if not os.path.isfile(file_path := kwargs[name]):
                    log.error(f"missing input {name} = '{file_path}' at {func.__name__}")
                    return
            return func(*args, **kwargs)

        return wrapper

    return decorator


def drop_missing_input_paths(input_file_args: List[list]):
    """decorator to drop input files that are missing

    Parameters
    ----------
    input_file_args : list
        list of input arguments corresponding to the input file paths

    Returns
    -------
    function
        decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, name in enumerate(input_file_args):
                # skip argument if not present in kwargs
                if name not in kwargs:
                    continue
                org_file_paths = kwargs[name]
                file_paths = list(filter(os.path.isfile, org_file_paths))
                if len(file_paths) == 0:
                    log.error(f"no input paths found for {name} = '{org_file_paths}' at {func.__name__}")
                    return
                elif len(file_paths) < len(org_file_paths):
                    log.warning(
                        f"dropping {len(org_file_paths) - len(file_paths)} "
                        f"missing input paths: '{set(org_file_paths).difference(file_paths)}' for '{name}' at {func.__name__}"
                    )
                kwargs[name] = file_paths
            return func(*args, **kwargs)

        return wrapper

    return decorator


def skip_if_drpqual_flags(flags: List[str], input_file_arg: str, reset_missing_optionals: bool = True):
    """decorator to skip a task if any of the drpqual flags is True

    Parameters
    ----------
    flags : List[str]
        list of drpqual flag names

    Returns
    -------
    function
        decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # skip argument if not present in kwargs
            if input_file_arg not in kwargs:
                return func(*args, **kwargs)
            # get function parameters
            pars = inspect.signature(func).parameters
            # silently return if input file is optional, is set to None and has default None
            if kwargs[input_file_arg] is None and pars[input_file_arg].default is None:
                return func(*args, **kwargs)
            # warning for optional input file that is missing
            elif kwargs[input_file_arg] is not None and pars[input_file_arg].default is None and not os.path.isfile(kwargs[input_file_arg]):
                log.warning(f"optional input {input_file_arg} = '{kwargs[input_file_arg]}' at {func.__name__} is missing")
                if reset_missing_optionals:
                    kwargs[input_file_arg] = None
                return func(*args, **kwargs)
            # skip task if input file is required and is set to None
            elif kwargs[input_file_arg] is None and pars[input_file_arg].default == inspect._empty:
                log.error(f"required input {input_file_arg} is set to None at {func.__name__}")
                return
            # quickly extract the drpqual bitmaks from the header
            drpqual = QualityFlag(fits.getheader(kwargs[input_file_arg]).get("DRPQUAL", QualityFlag(0)))
            if len(flags_set := set(flags).intersection(drpqual.get_name().split(","))) > 0:
                log.error(f"skipping {func.__name__} due to drpqual flags: {flags_set}")
                return
            return func(*args, **kwargs)

        return wrapper

    return decorator

# TODO: implement a decorator for validating outputs
# it should validate the following characteristics:
#   - exists
#   - number of rows and columns (in text files)
#   - shape of the image
#   - the image has structures
#   - image stats (mean, median, std, etc)
#   - continuum & arcs have the correct number of fibers
# TODO: add validation flags according to the calling function, e.g.: {f.__name__: flags}
# this way there will be no overwriting of the flags and the DRP should be able to track
# where things go wrong
# TODO: implement (non-)existing files as flags = {"missing": {(par_name, filename): True/False}}
# this way I can keep track of the files that went missing and report back using the logger


