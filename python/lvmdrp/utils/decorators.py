# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: decorators.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os


def missing_files(*files):
    def decorator(f):
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
        status: dict_like
            a dictionary containing the status of the run and a list of missing files
        """
        def inner(*args, **kwargs):
            exist = dict.fromkeys(files, False)
            for par_name in files:
                file_path = kwargs.get(par_name)
                exist[par_name] = os.path.isfile(file_path) if isinstance(file_path, str) else False
            if all(exist.values()):
                status = {"missing": {}}
                result = f(*args, **kwargs)
                return (*result, status) if isinstance(result, tuple) else (result, status)
            else:
                status = {"missing": {par_name: kwargs.get(par_name) for par_name in exist if not exist[par_name]}}
                return None, status
        return inner
    return decorator

if __name__ == "__main__":
    @missing_files("in_file")
    def simple_func(in_file, a, b, c):
        print(in_file)
        print(a, b, c)
        return a
    
    r = simple_func(in_file="a_file.txt", a=1, b=2, c=3)
    print(r)
    

