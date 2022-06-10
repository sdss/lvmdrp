# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: decorators.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os

from lvmdrp.utils.bitmask import QualityFlag


# TODO: implement a decorator for validating outputs
# it should validate the following characteristics:
#   - exists
#   - number of rows and columns (in text files)
#   - shape of the image
#   - the image has structures
#   - image stats (mean, median, std, etc)
# TODO: add validation flags according to the calling function, e.g.: {f.__name__: flags}
# this way there will be no overwriting of the flags and the DRP should be able to track
# where things go wrong
# TODO: implement (non-)existing files as flags = {"missing": {(par_name, filename): True/False}}
# this way I can keep track of the files that went missing and report back using the logger
def missing_files(potential_flags, *par_files):
    def decorator(f):
        def inner(*args, **kwargs):
            """decorator for bypassing calibration step if a needed file is missing

            This decorator checks for the existence of a given set of file paths.
            If all the files exist, the decorated function proceeds normally and
            returns a successful state of the step, otherwise the decorator bypasses
            the current step and returns the list of missing files

            Parameters
            ----------
            f: function object
                the target function to decorate

            Returns
            -------
            function returns
            flags: dict_like
                a dictionary containing the flags of the run and a list of missing files
            """
            files_exist = dict.fromkeys(par_files, False)
            for par_name in par_files:
                file_path = kwargs.get(par_name)
                files_exist[par_name] = os.path.isfile(file_path) if isinstance(file_path, str) else False
            
            # define OK flag
            flags = QualityFlag["OK"]
            if all(files_exist.values()):
                result = f(*args, **kwargs)
                return (*result, flags) if isinstance(result, tuple) else (result, flags)
            else:
                for flag in potential_flags:
                    flags += flag
                return None, flags
        return inner
    return decorator

if __name__ == "__main__":
    @missing_files(["BAD_CALIBRATION_FRAMES"], "in_file")
    def simple_func(in_file, a, b, c):
        print(in_file)
        print(a, b, c)
        return a
    
    r = simple_func(in_file="a_file.txt", a=1, b=2, c=3)
    print(r)
    

