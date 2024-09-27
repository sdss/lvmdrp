# encoding: utf-8
#
# setup.py
#

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import shutil
import os
import sys
import site
import subprocess


py_version = '%s.%s' % (sys.version_info[0], sys.version_info[1])
def get_env_lib_directory():
    """Return the installation directory, or None

    taken from: https://bit.ly/3BAOpQK
    """
    if '--user' in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        paths = (s % (py_version) for s in (
            sys.prefix + '/lib/python%s/dist-packages/',
            sys.prefix + '/lib/python%s/site-packages/',
            sys.prefix + '/local/lib/python%s/dist-packages/',
            sys.prefix + '/local/lib/python%s/site-packages/',
            '/Library/Python/%s/site-packages/',
        ))
    for path in paths:
        if os.path.exists(path):
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))


install_path = get_env_lib_directory()
fast_median_src = os.path.join(install_path, f"lib/python{py_version}/site-packages/cextern/fast_median/src")
cwd = os.getcwd()

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)

        shutil.copytree("python/cextern/fast_median/src", fast_median_src, dirs_exist_ok=True)
        os.chdir(fast_median_src)
        subprocess.run("make")
        os.chdir(cwd)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        shutil.copytree("python/cextern/fast_median/src", fast_median_src, dirs_exist_ok=True)
        os.chdir(fast_median_src)
        subprocess.run("make")
        os.chdir(cwd)

setup(cmdclass={'install': PostInstallCommand})
