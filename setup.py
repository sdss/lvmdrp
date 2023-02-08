# encoding: utf-8
#
# setup.py
#
# BUG: this script should take an optional master configuration template, other wise use the one shipped with the package

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

import os
import sys
import site
import shutil
import zipfile
import subprocess
import pexpect
import struct
import logging as setup_logger
import requests
from html.parser import HTMLParser
from urllib.parse import urlparse, parse_qs


def rc_symlink(src, dst):
    """Forces creation of symbolic link if already exists"""
    if os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)
    return None


def get_env_lib_directory():
    """Return the installation directory, or None
    
    taken from: https://bit.ly/3BAOpQK
    """
    if '--user' in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        py_version = '%s.%s' % (sys.version_info[0], sys.version_info[1])
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
    
    setup_logger.error('no installation path found')
    return None

# The NAME variable should be of the format "sdss-drp".
# Please check your NAME adheres to that format.
NAME = 'lvmdrp'
VERSION = '0.1.0'
RELEASE = 'dev' in VERSION

SYSTEM = struct.calcsize("P") * 8

SRC_PATH = os.path.abspath("src")
INS_PATH = get_env_lib_directory()
LIB_PATH = os.path.join(INS_PATH, "lib")
BIN_PATH = os.path.join(INS_PATH, "bin")
INC_PATH = os.path.join(INS_PATH, "include")

SKYCORR_SRC_PATH = os.path.join(SRC_PATH, "skycorr.tar.gz")
SKYMODEL_SRC_PATH = os.path.join(SRC_PATH, "SM-01.tar.gz")

SKYCORR_INST_PATH = os.path.join(LIB_PATH, "skycorr")
SKYMODEL_INST_PATH = os.path.join(LIB_PATH, "skymodel")

LVM_SRC_URL = "http://ifs.astroscu.unam.mx/LVM/lvmdrp_src.zip"


# TODO: implement installation path parameters and defaults
def install_eso_routines():
    # get original current directory
    initial_path = os.getcwd()

    # NOTE: this is probably an unsafe solution, but for now will have to do!!
    # - download compressed source files ----------------------------------------------------------------------------
    os.makedirs(SRC_PATH, exist_ok=True)
    os.chdir(SRC_PATH)
    out = subprocess.run(f"wget {LVM_SRC_URL}".split(), capture_output=True)
    if out.returncode == 0:
        setup_logger.info("successfully downloaded ESO source files")
    else:
        setup_logger.error("error while downloading source files")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode("utf-8"))
    with zipfile.ZipFile("lvmdrp_src.zip", "r") as src_compressed:
        src_compressed.extractall(os.path.curdir)
    os.remove("lvmdrp_src.zip")
    # ---------------------------------------------------------------------------------------------------------------

    # - install skycorr ---------------------------------------------------------------------------------------------
    setup_logger.info("preparing to install skycorr")
    out = subprocess.run(f"tar xzvf {SKYCORR_SRC_PATH}".split(), capture_output=True)
    if out.returncode == 0:
        setup_logger.info("successfully extracted skycorr installer")
    else:
        setup_logger.error("error while preparing skycorr files")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode("utf-8"))

    os.chdir("skycorr")
    if sys.platform == "linux":
        if SYSTEM == 32:
            skycorr_installer = pexpect.spawn("bash skycorr_installer_linux_i686-1.1.2.run", encoding="utf-8")
        elif SYSTEM == 64:
            skycorr_installer = pexpect.spawn("bash skycorr_installer_linux_x86_64-1.1.2.run", encoding="utf-8")
    elif sys.platform == "darwin":
        skycorr_installer = pexpect.spawn("bash skycorr_installer_macos_x86_64-1.1.2.run", encoding="utf-8")
    else:
        raise NotImplementedError(f"installation not implemented for '{sys.platform}' OS")

    skycorr_installer.logfile_read = sys.stdout
    skycorr_installer.expect("root installation directory")
    skycorr_installer.sendline(SKYCORR_INST_PATH)
    skycorr_installer.expect("Is this OK [Y/n]?")
    skycorr_installer.sendline("y")
    prompt = skycorr_installer.expect("Proceed with this installation directory")
    if prompt == 0:
        skycorr_installer.sendline("y")
    prompt = skycorr_installer.expect("will overwrite existing files without further warning")
    if prompt == 0:
        skycorr_installer.sendline("y")
    skycorr_installer.wait()
    skycorr_installer.close()

    if skycorr_installer.exitstatus == 0:
        setup_logger.info("successfully installed skycorr")
    else:
        setup_logger.error(f"error while installing skycorr")
    
    out = subprocess.run(f"{os.path.join(SKYCORR_INST_PATH, 'bin', 'skycorr')} {os.path.join(SKYCORR_INST_PATH, 'examples', 'config', 'sctest_sinfo_H.par')}".split(), capture_output=True)
    if out.returncode == 0:
        setup_logger.info("successfully tested skycorr")
    else:
        setup_logger.error("error while testing skycorr")
        setup_logger.error(f"rolling back changes in {SKYCORR_INST_PATH}")
        shutil.rmtree(SKYCORR_INST_PATH, ignore_errors=True)
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))

    # - install skymodel ---------------------------------------------------------------------------------------------
    setup_logger.info("preparing to install skymodel")
    os.chdir(SRC_PATH)
    out = subprocess.run(f"tar xzvf {SKYMODEL_SRC_PATH}".split(), capture_output=True)
    if out.returncode == 0:
        setup_logger.info("successfully extracted sky model installer")
    else:
        setup_logger.error("error while extracting skymodel")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))

    # create directory structure in installation path
    os.chdir(os.path.join(SRC_PATH, "SM-01", "sm-01_mod1"))
    # copy line database to installation-dir/data
    os.makedirs(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "data"), exist_ok=True)
    # copy config, data and doc dirs into installation-dir/data
    shutil.copytree("config", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "config"), dirs_exist_ok=True)
    shutil.copytree("data", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "data"), dirs_exist_ok=True)
    shutil.copytree("doc", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "doc"), dirs_exist_ok=True)
    shutil.copytree("third_party_code", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code"), dirs_exist_ok=True)
    shutil.copytree("bin", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin"), dirs_exist_ok=True)
    shutil.copytree("src", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "src"), dirs_exist_ok=True)
    shutil.copytree("m4macros", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "m4macros"), dirs_exist_ok=True)
    shutil.copyfile("bootstrap", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bootstrap"))
    shutil.copyfile("configure.ac", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "configure.ac"))
    shutil.copyfile("do_all_compilations.sh", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "do_all_compilations.sh"))
    shutil.copyfile("Makefile.am", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "Makefile.am"))

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code"))
    out = subprocess.run("tar xzvf lnfl_lblrtm_aer.tar.gz".split(), capture_output=True)
    if out.returncode == 0:
        setup_logger.info("successfully extracted third-party codes")
    else:
        setup_logger.error("error while extracting third-party codes")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))

    shutil.copyfile(
        os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "aer_v_3.2", "line_file", "aer_v_3.2"),
        os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "data", "aer_v_3.2")
    )

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lnfl", "build"))
    setup_logger.info("installing LNFL")
    if sys.platform == "linux":
        out = subprocess.run("make -f make_lnfl linuxGNUsgl".split(), capture_output=True)
    elif sys.platform == "darwin":
        out = subprocess.run("make -f make_lnfl osxGNUsgl".split(), capture_output=True)
    else:
        raise NotImplementedError(f"installation not implemented for '{sys.platform}' OS")

    if out.returncode == 0:
        setup_logger.info("successfully installed LNFL")
    else:
        setup_logger.error("error while installing LNFL")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))
    
    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lnfl"))
    shutil.move("lnfl_v2.6_linux_gnu_sgl", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_linux_gnu_sgl"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_linux_gnu_sgl"), os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl"))

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lblrtm", "build"))
    setup_logger.info("installing LBLRTM")
    if sys.platform == "linux":
        out = subprocess.run("make -f make_lblrtm linuxGNUsgl".split(), capture_output=True)
    elif sys.platform == "darwin":
        out = subprocess.run("make -f make_lblrtm osxGNUsgl".split(), capture_output=True)
    else:
        raise NotImplementedError(f"installation not implemented for '{sys.platform}' OS")      

    if out.returncode == 0:
        setup_logger.info("successfully installed LBLRTM")
    else:
        setup_logger.error("error while installing LBLRTM")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lblrtm"))
    shutil.move("lblrtm_v12.2_linux_gnu_sgl", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lblrtm_v12.2_linux_gnu_sgl"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lblrtm_v12.2_linux_gnu_sgl"), os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lblrtm"))

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1"))
    out = subprocess.run("bash bootstrap".split(), capture_output=True)
    out = subprocess.run(f"bash configure --prefix={os.path.join(SKYMODEL_INST_PATH, 'sm-01_mod1')} --with-cpl={SKYCORR_INST_PATH}".split(), capture_output=True)
    out = subprocess.run("make".split(), capture_output=True)
    out = subprocess.run("make install".split(), capture_output=True)

    if out.returncode == 0:
        setup_logger.info("successfully installed skymodel module 01")
        # setup_logger.info(out.stdout.decode("utf-8"))
    else:
        setup_logger.error("error while installing skymodel module 01")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))

    # make symbolic links of binary files in python_dir/bin
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_linux_gnu_sgl"), os.path.join(BIN_PATH, "lnfl"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lblrtm_v12.2_linux_gnu_sgl"), os.path.join(BIN_PATH, "lblrtm"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "create_spec"), os.path.join(BIN_PATH, "create_spec"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "create_speclib"), os.path.join(BIN_PATH, "create_speclib"))
    # NOTE: the following 2 TODOs will be useful if we want to run ESO routines from different directories
    # TODO: copy library files in python_dir/lib
    # TODO: copy header files in python_dir/include

    os.chdir(os.path.join(SRC_PATH, "SM-01", "sm-01_mod2"))
    shutil.copytree("config", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "config"), dirs_exist_ok=True)
    shutil.copytree("data", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "data"), dirs_exist_ok=True)
    shutil.copytree("test", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "test"), dirs_exist_ok=True)
    shutil.copytree("doc", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "doc"), dirs_exist_ok=True)
    shutil.copytree("src", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "src"), dirs_exist_ok=True)
    shutil.copytree("m4macros", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "m4macros"), dirs_exist_ok=True)
    shutil.copyfile("bootstrap", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "bootstrap"))
    shutil.copyfile("configure.ac", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "configure.ac"))
    shutil.copyfile("Makefile.am", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "Makefile.am"))
    os.makedirs(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "output"), exist_ok=True)

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2"))
    out = subprocess.run("bash bootstrap".split(), capture_output=True)
    out = subprocess.run(f"bash configure --prefix={os.path.join(SKYMODEL_INST_PATH, 'sm-01_mod2')} --with-cpl={SKYCORR_INST_PATH}".split(), capture_output=True)
    out = subprocess.run("make install".split(), capture_output=True)

    if out.returncode == 0:
        setup_logger.info("successfully installed skymodel module 02")
        # setup_logger.info(out.stdout.decode("utf-8"))
    else:
        setup_logger.error("error while installing skymodel module 02")
        setup_logger.error("full report:")
        setup_logger.error(out.stderr.decode('utf-8'))

    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "bin", "preplinetrans"), os.path.join(BIN_PATH, "preplinetrans"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "bin", "calcskymodel"), os.path.join(BIN_PATH, "calcskymodel"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "bin", "estmultiscat"), os.path.join(BIN_PATH, "estmultiscat"))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod2", "bin", "testskymodel"), os.path.join(BIN_PATH, "testskymodel"))
    # TODO: copy library files in python_dir/lib
    # TODO: copy header files in python_dir/include
    
    # clean 'src' directory
    shutil.rmtree(os.path.join(SRC_PATH, "SM-01"))
    shutil.rmtree(os.path.join(SRC_PATH, "skycorr"))

    # return to original directory
    os.chdir(initial_path)


class DevCommand(develop):
    def run(self):
        develop.run(self)

        # post-install stuff
        install_eso_routines()


class InsCommand(install):
    def run(self):
        install.run(self)

        # post-install stuff
        install_eso_routines()


def run(packages, install_requires):

    setup(
        name=NAME,
        version=VERSION,
        license='BSD3',
        description='SDSSV-LVM Data Reduction Pipeline',
        long_description=open('README.rst').read(),
        author='Eric Pellegrini',
        author_email='ericpellegrini@outlook.com',
        keywords='astronomy software',
        url='https://github.com/sdss/lvmdrp',
        include_package_data=True,
        packages=packages,
        install_requires=install_requires,
        package_dir={'': 'python'},
        scripts=[
            'bin/drp',
            'bin/pix2wave',
            "bin/prime-db"
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.8',
            'Topic :: Documentation :: Sphinx',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        cmdclass={
            "develop": DevCommand,
            "install": InsCommand
        }
    )


def parse_requirements(reqfile_path):
    """
        Returns the parsed requirements from a requirements .txt file
    """
    install_requires = []
    with open(reqfile_path, "r") as r:
        requirement = r.readline()[:-1].strip()
        if requirement.startswith("-r"):
            install_requires.extend(parse_requirements(requirement.replace("-r ", "")))
        else:
            install_requires.append(requirement)
    return install_requires


def get_requirements():
    ''' Get the proper requirements file based on the optional argument '''

    requirements_file = os.path.join(os.path.dirname(__file__), "requirements_all.txt")
    install_requires = parse_requirements(requirements_file)
    return install_requires


def remove_args(parser):
    ''' Remove custom arguments from the parser '''

    arguments = []
    for action in list(parser._get_optional_actions()):
        if '--help' not in action.option_strings:
            arguments += action.option_strings

    for arg in arguments:
        if arg in sys.argv:
            sys.argv.remove(arg)


if __name__ == '__main__':

    # Get the proper requirements file
    install_requires = get_requirements()

    # Have distutils find the packages
    packages = find_packages(where='python')

    # Runs distutils
    run(packages, install_requires)
