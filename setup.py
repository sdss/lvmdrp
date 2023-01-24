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
import argparse
import shutil
import subprocess
import struct
import logging as setup_logger


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
NAME = 'drp'
VERSION = '0.1.0dev'
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


# TODO: implement installation path parameters and defaults
def install_eso_routines():
    # - install skycorr ---------------------------------------------------------------------------------------------
    setup_logger.info("preparing to install skycorr")
    os.chdir(SRC_PATH)
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
            out = subprocess.run("bash skycorr_installer_linux_i686-1.1.2.run".split(), capture_output=True, text=True, input=f"{SKYCORR_INST_PATH}\ny\ny\n")
        elif SYSTEM == 64:
            out = subprocess.run("bash skycorr_installer_linux_x86_64-1.1.2.run".split(), capture_output=True, text=True, input=f"{SKYCORR_INST_PATH}\ny\ny\n")
    elif sys.platform == "darwin":
        out = subprocess.run("bash skycorr_installer_macos_x86_64-1.1.2.run".split(), capture_output=True, text=True, input=f"{SKYCORR_INST_PATH}\ny\ny\n")
    else:
        raise NotImplementedError(f"installation not implemented for '{sys.platform}' OS")

    if out.returncode == 0:
        setup_logger.info("successfully installed skycorr")
    else:
        setup_logger.error(f"error while installing skycorr")
        setup_logger.error(f"full report:")
        setup_logger.error(f"{out.stderr}")
    
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

    setup(name=NAME,
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


def get_requirements(opts):
    ''' Get the proper requirements file based on the optional argument '''

    if opts.dev:
        name = 'requirements_dev.txt'
    elif opts.doc:
        name = 'requirements_doc.txt'
    else:
        name = 'requirements.txt'

    requirements_file = os.path.join(os.path.dirname(__file__), name)
    install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                        if not line.strip().startswith('#') and line.strip() != '']
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

    # Custom parser to decide whether which requirements to install
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument('-d', '--dev', dest='dev', default=False, action='store_true',
                        help='Install all packages for development')
    parser.add_argument('-o', '--doc', dest='doc', default=False, action='store_true',
                        help='Install only core + documentation packages')

    # We use parse_known_args because we want to leave the remaining args for distutils
    args = parser.parse_known_args()[0]

    # Get the proper requirements file
    install_requires = get_requirements(args)

    # Now we remove all our custom arguments to make sure they don't interfere with distutils
    remove_args(parser)

    # Have distutils find the packages
    packages = find_packages(where='python')

    # Runs distutils
    run(packages, install_requires)
