# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: skyMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import yaml
import subprocess
import itertools as it
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing import Pool
from scipy import optimize
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

import sys
import os
import shutil
import zipfile
import subprocess
import pexpect
import struct

from lvmdrp.core.constants import LVM_SRC_URL, SRC_PATH, SKYCORR_SRC_PATH, SKYMODEL_SRC_PATH, SKYCORR_INST_PATH, SKYMODEL_INST_PATH
from lvmdrp.core.constants import BIN_PATH, SKYCORR_CONFIG_PATH, SKYMODEL_CONFIG_PATH
from lvmdrp.utils import rc_symlink
from lvmdrp.utils.logger import get_logger
from lvmdrp.core.sky import run_skycorr, run_skymodel, skymodel_pars_from_header, optimize_sky, ang_distance
from lvmdrp.core.passband import PassBand
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.header import Header
from lvmdrp.core.rss import RSS


description = "Provides methods for sky subtraction"

__all__ = [
    "installESOSky_drp", "configureSkyModel_drp", "createMasterSky_drp",
    "sepContinuumLine_drp", "evalESOSky_drp", "subtractGeocoronal_drp",
    "corrSkyLine_drp", "corrSkyContinuum_drp", "coaddContinuumLine_drp",
    "subtractSky_drp", "refineContinuum_drp", "subtractPCAResiduals_drp"
]


sky_logger = get_logger(name=__name__)


SYSTEM = struct.calcsize("P") * 8


# TODO: implement installation path parameters and defaults
def installESOSky_drp():
    """
        Installs the ESO sky routines in the current python environment


    """
    # get original current directory
    initial_path = os.getcwd()
    # - download compressed source files ----------------------------------------------------------------------------
    os.makedirs(SRC_PATH, exist_ok=True)
    os.chdir(SRC_PATH)
    sky_logger.info(f"downloading ESO sky source code from '{LVM_SRC_URL}'")
    out = subprocess.run(f"curl {LVM_SRC_URL} --output lvmdrp_src.zip".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully downloaded ESO source files")
    else:
        sky_logger.error("error while downloading source files")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode("utf-8"))
    with zipfile.ZipFile("lvmdrp_src.zip", "r") as src_compressed:
        src_compressed.extractall(os.path.curdir)
    os.remove("lvmdrp_src.zip")
    # ---------------------------------------------------------------------------------------------------------------

    # - install skycorr ---------------------------------------------------------------------------------------------
    sky_logger.info(f"preparing to install skycorr at '{SKYCORR_INST_PATH}'")
    out = subprocess.run(f"tar xzvf {SKYCORR_SRC_PATH}".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully extracted skycorr installer")
    else:
        sky_logger.error("error while preparing skycorr files")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode("utf-8"))

    sky_logger.info(f"installing skycorr on a {SYSTEM}-bit {sys.platform.capitalize()} system")
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

    # skycorr_installer.delaybeforesend = 1.0
    # skycorr_installer.logfile_read = sys.stdout
    skycorr_installer.expect("root installation directory")
    skycorr_installer.sendline(SKYCORR_INST_PATH)
    skycorr_installer.expect("Is this OK")
    skycorr_installer.sendline("y")
    if len(SKYCORR_INST_PATH) > 40:
        skycorr_installer.expect("Proceed with this installation directory")
        skycorr_installer.sendline("y")
    if os.path.exists(SKYCORR_INST_PATH):
        skycorr_installer.expect("will overwrite existing files without further warning")
        skycorr_installer.sendline("y")
    skycorr_installer.wait()
    skycorr_installer.close()

    if skycorr_installer.exitstatus == 0:
        sky_logger.info("successfully installed skycorr")
    else:
        sky_logger.error(f"error while installing skycorr")
    
    out = subprocess.run(f"{os.path.join(SKYCORR_INST_PATH, 'bin', 'skycorr')} {os.path.join(SKYCORR_INST_PATH, 'examples', 'config', 'sctest_sinfo_H.par')}".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully tested skycorr")
    else:
        sky_logger.error("error while testing skycorr")
        sky_logger.error(f"rolling back changes in '{SKYCORR_INST_PATH}'")
        shutil.rmtree(SKYCORR_INST_PATH, ignore_errors=True)
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))

    # defining environment variables for CPL
    cpl_path = os.path.join(SKYCORR_INST_PATH, "lib")
    sky_logger.info(f"setting $LD_LIBRARY_PATH={cpl_path} for CPL discovery")
    os.environ["LD_LIBRARY_PATH"] = cpl_path

    # - install skymodel ---------------------------------------------------------------------------------------------
    sky_logger.info(f"preparing to install skymodel at '{SKYMODEL_INST_PATH}'")
    os.chdir(SRC_PATH)
    out = subprocess.run(f"tar xzvf {SKYMODEL_SRC_PATH}".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully extracted skymodel installer")
    else:
        sky_logger.error("error while extracting skymodel")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))

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
        sky_logger.info("successfully extracted third-party codes")
    else:
        sky_logger.error("error while extracting third-party codes")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))

    shutil.copyfile(
        os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "aer_v_3.2", "line_file", "aer_v_3.2"),
        os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "data", "aer_v_3.2")
    )

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lnfl", "build"))
    sky_logger.info("installing LNFL")
    if sys.platform == "linux":
        out = subprocess.run("make -f make_lnfl linuxGNUsgl".split(), capture_output=True)
    elif sys.platform == "darwin":
        out = subprocess.run("make -f make_lnfl osxGNUsgl".split(), capture_output=True)
    else:
        raise NotImplementedError(f"installation not implemented for '{sys.platform}' OS")

    if out.returncode == 0:
        sky_logger.info("successfully installed LNFL")
    else:
        sky_logger.error("error while installing LNFL")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))
    
    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lnfl"))

    if sys.platform == "linux":
        shutil.move("lnfl_v2.6_linux_gnu_sgl", os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_linux_gnu_sgl"))
        rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_linux_gnu_sgl"), os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl"))
    elif sys.platform == "darwin":
        shutil.move("lnfl_v2.6_OS_X_gnu_sgl",
                    os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_OS_X_gnu_sgl"))
        rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl_v2.6_OS_X_gnu_sgl"),
                   os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lnfl"))

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lblrtm", "build"))
    sky_logger.info("installing LBLRTM")
    if sys.platform == "linux":
        out = subprocess.run("make -f make_lblrtm linuxGNUsgl".split(), capture_output=True)
        lblrtm_sgl_file="lblrtm_v12.2_linux_gnu_sgl"
        cpl_path = SKYCORR_INST_PATH
    elif sys.platform == "darwin":
        out = subprocess.run("make -f make_lblrtm osxGNUsgl".split(), capture_output=True)
        lblrtm_sgl_file = "lblrtm_v12.2_OS_X_gnu_sgl"
        cpl_path = "/usr/local"
    else:
        raise NotImplementedError(f"installation not implemented for '{sys.platform}' OS")      

    if out.returncode == 0:
        sky_logger.info("successfully installed LBLRTM")
    else:
        sky_logger.error("error while installing LBLRTM")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))

    sky_logger.info("installing skymodel")
    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "third_party_code", "lblrtm"))
    shutil.move(lblrtm_sgl_file, os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", lblrtm_sgl_file))
    rc_symlink(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", lblrtm_sgl_file), os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1", "bin", "lblrtm"))

    os.chdir(os.path.join(SKYMODEL_INST_PATH, "sm-01_mod1"))
    out = subprocess.run("bash bootstrap".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully finished bootstrap for module 01")
    else:
        sky_logger.error("error while running bootstrap for module 01")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decor("utf-8"))
    out = subprocess.run(f"bash configure --prefix={os.path.join(SKYMODEL_INST_PATH, 'sm-01_mod1')} --with-cpl={cpl_path}".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully finished configure for module 01")
    else:
        sky_logger.error("error while running configure for module 01")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode("utf-8"))
    out = subprocess.run("make".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully finished make for module 01")
    else:
        sky_logger.error("error while running make for module 01")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode("utf-8"))
    out = subprocess.run("make install".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully installed skymodel module 01")
    else:
        sky_logger.error("error while installing skymodel module 01")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))

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
    if out.returncode == 0:
        sky_logger.info("successfully finished bootstrap for module 02")
    else:
        sky_logger.error("error while running bootstrap for module 02")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode("utf-8"))
    out = subprocess.run(f"bash configure --prefix={os.path.join(SKYMODEL_INST_PATH, 'sm-01_mod2')} --with-cpl={cpl_path}".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully finished configure for module 02")
    else:
        sky_logger.error("error while running configure for module 02")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode("utf-8"))
    out = subprocess.run("make install".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully installed skymodel module 02")
    else:
        sky_logger.error("error while installing skymodel module 02")
        sky_logger.error("full report:")
        sky_logger.error(out.stderr.decode('utf-8'))

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


def configureSkyModel_drp(skymodel_config_path=SKYMODEL_CONFIG_PATH, skymodel_path=SKYMODEL_INST_PATH, method="run", run_library=False, run_multiscat=False, pwvs="-1", source=""):
    """
        Runs/downloads the configuration files of the sky module
        
        If method='run' mode, the following ESO configuration files will be written:
            - lblrtm_setup
            - libstruct.dat
            - sm_filenames.dat
            - instrument_etc.par
            - skymodel_etc.par

        Then, if run_library=True, this function will execute the following ESO routines:
            > create_spec <airmass> <time> <seasson> <output_path> <spectra_resolution> <pwv>
            > preplinetrans

        Additionally, the ESO routine for updating the multiple scattering corrections component
        will be executed if run_multiscat=True
            > estmultiscat

        If method='download', this function will download the neccessary files to run the
        ESO sky models. Additionally you can specify the source from which these files should
        be downloaded. NOTE: this method is not implemented yet.

        Parameters
        ----------
        skymodel_config_path : string
            path to master ESO sky model configuration file. Defaults to {SKYMODEL_CONFIG_PATH}
        skymodel_inst_path : string
            path to ESO sky model installation path. Defaults to {SKYMODEL_INST_PATH}
        method : string
            which method to use for the ESO sky model configuration:
                - 'run' : will write the configuration files and (optionally) pre-build a library
                - 'download' : will download all configuration files and corresponding library files
        run_library : boolean
            whether to run or not the ESO routines to build a spectral library using the specified
            configuration files and a set of precipitable water vapor scalings (see 'pwv')
        run_multiscat : boolean
            whether to run or not the ESO 'estmultiscat' routine for the multiple scattering
            corrections
        pwvs : string of floats
            the precipitable water vapor values (in mm) to use. Defaults to -1 which means no PWV
            scaling is applied

        Examples
        --------

        user:> drp sky configureSkyModel # to write the configuration files only
        user:> drp sky configureSkyModel method=run run_library=True run_multiscat=False pwvs=0.5,1.0,2.5

    """

    if method == "run":
        sky_logger.info(f"writing configuration files using '{skymodel_config_path}' as source")
        # read master configuration file
        skymodel_master_config = yaml.load(open(skymodel_config_path, "r"), Loader=yaml.Loader)

        # write default parameters for the ESO skymodel
        config_names = list(skymodel_master_config.keys())
        with open(os.path.join(skymodel_path, "sm-01_mod1", "config", config_names[0]), "w") as cf:
            for key, val in skymodel_master_config[config_names[0]].items():
                cf.write(f"{key} = {val}\n")
        with open(os.path.join(skymodel_path, "sm-01_mod2", "data", config_names[1]), "w") as cf:
            for par in skymodel_master_config[config_names[1]]:
                cf.write(f"{par}\n")
        with open(os.path.join(skymodel_path, "sm-01_mod2", "data", config_names[2]), "w") as cf:
            for key, val in skymodel_master_config[config_names[2]].items():
                cf.write(f"{key} = {val}\n")
        with open(os.path.join(skymodel_path, "sm-01_mod2", "config", config_names[3]), "w") as cf:
            for key, val in skymodel_master_config[config_names[3]].items():
                cf.write(f"{key} = {val}\n")
        with open(os.path.join(skymodel_path, "sm-01_mod2", "config", config_names[4]), "w") as cf:
            for key, val in skymodel_master_config[config_names[4]].items():
                cf.write(f"{key} = {val}\n")
        sky_logger.info("successfully written config files")

        # create sky library
        if run_library:
            sky_logger.info("creating sky radiative models library")
            # TODO: parse create_spec parameters
            os.chdir(os.path.join(skymodel_path, "sm-01_mod1"))
            lib_path = os.path.abspath(skymodel_master_config["sm_filenames.dat"]["libpath"])
            fact = dict(map(lambda s: s.split()[1:], skymodel_master_config["libstruct.dat"][::1]))
            fact = {k: 10**eval(v) for k, v in fact.items()}
            pars = dict(zip(fact.keys(), skymodel_master_config["libstruct.dat"][1::1].split()))
            
            airmasses = map(lambda f, p: f*eval(p), fact["airmass"], pars["airmass"])
            times = map(lambda f, p: f * eval(p), fact["time"], pars["time"])
            seasons = map(lambda f, p: f * eval(p), fact["season"], pars["season"])
            resols = map(lambda f, p: f * eval(p), fact["resol"], pars["resol"])
            pwvs = pwvs.split()
            create_spec_pars = it.product(airmasses, times, seasons, resols, pwvs)

            # run create_spec across all parameter grid
            for airmass, time, season, res, pwv in tqdm(create_spec_pars, desc="creating sky library", unit="grid step", ascii=True):
                out = subprocess.run(f"{os.path.join('bin', 'create_spec')} {airmass} {time} {season} {lib_path} {res} {pwv}".split(), capture_output=True)
                if out.returncode == 0:
                    sky_logger.info(f"successfully finished 'create_spec' with parameters: {airmass=}, {time=}, {season=}, {res=}, {pwv=}")
                else:
                    sky_logger.error("failed while running 'create_spec'")
                    sky_logger.error(f"with parameters: {airmass=}, {time=}, {season=}, {res=}, {pwv=}")
                    sky_logger.error(out.stderr.decode("utf-8"))

            # create library destination path
            os.makedirs(os.path.join(skymodel_path, "sm-01_mod2", "data", "lib"), exist_ok=True)
            # copy library to destination path as specified in sm_filenames.dat
            shutil.copytree(os.path.join(skymodel_path, "sm-01_mod1", "data"), os.path.join(skymodel_path, "sm-01_mod2", "data", "lib"), dirs_exist_ok=True)

            # run prelinetrans
            os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
            out = subprocess.run(os.path.join("bin", "preplinetrans").split(), capture_output=True)
            if out.returncode == 0:
                sky_logger.info("sucessfully finished 'preplinetrans'")
            else:
                sky_logger.error("failed while running 'preplinetrans'")
                sky_logger.error(out.stderr.decode("utf-8"))
            
            if run_multiscat:
                out = subprocess.run(os.path.join("bin", "estmultiscat").split(), capture_output=True)
                if out.returncode == 0:
                    sky_logger.info("successfully finished 'estmultiscat'")
                else:
                    sky_logger.error("failed while running 'estmultiscat'")
                    sky_logger.error(out.stderr.decode("utf-8"))
    elif method == "download":
        # TODO: download master configuration file and overwrite current one
        # TODO: write individual configuration files (as above)
        # TODO: download create_spec outputs and overwrite current ones
        # TODO: download preplinetrans outputs and overwrite current ones
        # TODO: download multiscat outputs and overwrite current ones
        raise NotImplementedError(f"'{method}' is not implemented yet. Please try again using the 'run' method")
    else:
        raise ValueError(f"unknown method '{method}'. Valid values are: 'run' and 'download'")
        

def createMasterSky_drp(in_rss, out_sky, clip_sigma='3.0', nsky='0', filter='', non_neg='1', plot='0'):
    """
        Creates a mean (sky) spectrum from the RSS, which stored either as a FITS or an ASCII file.
        Spectra may be rejected from the median computation. Bad pixel in the RSS are not included
        in the median computation.

        TODO: implement fiber rejection for science pointings which should make other considerations

        Parameters
        --------------
        in_rss : string
            Input RSS FITS file with a pixel table for the spectral resolution
        out_sky : string
            Output Sky spectrum. Either in FITS format (if *.fits) or in ASCII format (if *.txt)
        clip_sigma : string of float, optional with default: '3.0'
            Sigma value used to reject outlier sky spectra identified in the collapsed median value
            along the dispersion axis. Only used if the nsky value is set to 0 and clip_sigma>0
        nsky : string of integer (>0), optional with default: '0'
            Selects the number of brightest sky spectra to be used for creating the median sky spec.
        filter : string of tuple, optional with default: ''
            Path to file containing the response function of a filter, and the wavelength and
            transmission columns
        plot : string of integer (0 or 1)
            If set to 1, the sky spectrum will be display on screen.

        Examples
        ----------------
        user:> drp sky constructSkySpec IN_RSS.fits OUT_SKY.fits 3.0
        user:> drp sky constructSkySpec IN_RSS.fits OUT_SKY.txt
    """
    sky_logger.info(f"preparing to create master 'sky' from '{in_rss}'")

    clip_sigma=float(clip_sigma)
    nsky = int(nsky)
    non_neg = int(non_neg)
    plot = int(plot)
    filter=filter.split(',')
    
    rss = RSS()
    rss.loadFitsData(in_rss)

    sky_logger.info("calculating median value for each fiber")
    median = np.zeros(len(rss), dtype=np.float32)
    for i in range(len(rss)):
        spec = rss[i]
        
        if spec._mask is not None:
            good_pixels = np.logical_not(spec._mask)
            if np.sum(good_pixels)!=0:
                median[i] = np.median(spec._data[good_pixels])
            else:
                median[i]=0
        else:
            median[i] = np.median(spec._data)
    # mask for fibers with valid sky spectra
    select_good = median!=0

    # sigma clipping around the median sky spectrum
    if clip_sigma>0.0 and nsky==0:
        sky_logger.info(f"calculating sigma clipping with sigma = {clip_sigma} within {select_good.sum()} fibers")
        select = np.logical_and(np.logical_and(median<np.median(median[select_good])+clip_sigma*np.std(median[select_good])/2.0, median>np.median(median[select_good])-clip_sigma*np.std(median[select_good])/2.0), select_good)
        sky_fib = np.sum(select)
    # select fibers that are below the maximum median spectrum within the top nsky fibers
    elif nsky>0:
        idx=np.argsort(median[select_good])
        max_value = np.max(median[select_good][idx[:nsky]])
        if non_neg==1:
            sky_logger.info(f"selecting non-negative (maximum) {nsky} fibers")
            select = (median<=max_value) & (median>0.0)
        else:
            sky_logger.info(f"selecting (maximum) {nsky} fibers with median below {max_value = }")
            select = (median<=max_value)
        sky_fib = np.sum(select)
    rss.setHdrValue('HIERARCH PIPE NSKY FIB', sky_fib, 'Number of averaged sky fibers')

    # selection of sky fibers to build master sky
    subRSS = rss.subRSS(select)

    # calculates the sky magnitude within a given filter response function
    if filter[0] != '':
        sky_logger.info(f"calculating 'sky' magnitude in Vega system using filter in {filter[0]}")
        passband = PassBand()
        passband.loadTxtFile(filter[0], wave_col=int(filter[1]),  trans_col=int(filter[2]))
        (flux_rss, error_rss, min_rss, max_rss, std_rss) = passband.getFluxRSS(subRSS)
        mag_flux = np.zeros(len(flux_rss))
        for m in range(len(flux_rss)):
            if flux_rss[m]>0.0:
                mag_flux[m] = passband.fluxToMag(flux_rss[m], system='Vega')

        mag_mean = np.mean(mag_flux[mag_flux>0.0])
        mag_min = np.min(mag_flux[mag_flux>0.0])
        mag_max = np.max(mag_flux[mag_flux>0.0])
        mag_std = np.std(mag_flux[mag_flux>0.0])
        rss.setHdrValue('HIERARCH PIPE SKY MEAN', float('%.2f'%mag_mean), 'Mean sky brightness of sky fibers')
        rss.setHdrValue('HIERARCH PIPE SKY MIN', float('%.2f'%mag_min), 'Minimum sky brightness of sky fibers')
        rss.setHdrValue('HIERARCH PIPE SKY MAX', float('%.2f'%mag_max), 'Maximum sky brightness of sky fibers')
        rss.setHdrValue('HIERARCH PIPE SKY RMS', float('%.2f'%mag_std), 'RMS sky brightness of sky fibers')
        sky_logger.info(f"{mag_mean = }, {mag_min = }, {mag_max = }, {mag_std = }")

    # create master sky spectrum by computing the average spectrum across selected fibers
    sky_logger.info(f"creating master (averaged) sky out of {subRSS._fibers}")
    skySpec = subRSS.create1DSpec()
    
    if plot==1:
        plt.figure(figsize=(20,5))
        plt.step(skySpec._wave, skySpec._data, color='k')
        plt.show()
    
    sky_logger.info(f"storing master sky in '{out_sky}'")
    skySpec.writeFitsData(out_sky)


def sepContinuumLine_drp(sky_ref, out_cont_line, method="skycorr", sky_sci="", skycorr_config=SKYCORR_CONFIG_PATH, is_science=False):
    """

        Separates the continuum from the sky line contribution using the specified method. The
        output spectra (continuum and line) is stored in a RSS format, with the continuum in the
        first row.
        
        If method='skycorr' (default), this function will use the ESO skycorr routine to fit for
        the line and continuum contribution of the given spectrum in 'sky_ref'. To be able to run
        this method, 'sky_sci' should be given and contain a 1D version of the science spectrum.
        Optionally a YAML file containing skycorr parameter definitions could also be given.

        If method='model', this function will use the ESO sky model to calculate a sky spectrum
        matching the 'sky_ref' observing conditions (ephemeris, airmass, etc.). The continuum
        contribution from the target sky spectrum is set to be the continuum component of the
        calculated model.

        If method='fit', this function will run a tradicional spectral fitting method to
        dissentangle the continuum and line contributions using a set of pre-built continuum/line
        templates.

        NOTE: by using the 'skycorr' method, we get for free a first fitting of the line
        contribution for the 'sky_ref' spectrum. By using the 'model' method, we get all calculated
        components for the target sky spectrum. This information could be use later

        Parameters
        ----------
        sky_ref : string
            path to the 1D target sky spectrum. It should be readable as a
            lvmdrp.core.spectrum1d.Spectrum1D
        out_cont_line : string
            path where the output RSS file will be stored. It will be saved using the methods in
            lvmdrp.core.rss.RSS
         method : string of 'skycorr' (default), 'model' or 'fit'
            the method to be used for the continuum line separation.
        sky_sci : string, optional
            path to the 1D science sky spectrum in the same format as 'sky_ref'. This parameter is
            only requiered if method='skycorr'
        skycorr_config : string, optional with default {SKYCORR_CONFIG_PATH}
            path to a file containing the skycorr parameter definitions in YAML format
        
    
        Examples
        ----------------
        user:> drp sky sepContinumLine SKY_REF.fits OUT_CONT_LINE.fits method='model'
        user:> drp sky sepContinumLine SKY_REF.fits OUT_CONT_LINE.fits sky_sci='SKY_SCI.fits'

    """
    # TODO: if science, then remove/mask out science lines from a predefined list
    # TODO: if science, then select wavelength ranges dominated by sky
    if is_science:
        pass

    # read sky spectrum
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky_ref, extension_hdr=0)
    
    # run skycorr
    if method == "skycorr":
        prefix = "SC"
        if sky_sci != "":
            sci_spec = Spectrum1D()
            sci_spec.loadFitsData(sky_sci, extension_hdr=0)
        else:
            raise ValueError(f"You need to provide a science spectrum to perform the continuum/line separation using skycorr.")
        if np.any(sky_spec._wave != sci_spec._wave):
            sky_spec = sky_spec.resampleSpec(ref_wave=sci_spec._wave, method="linear")
        if np.any(sky_spec._inst_fwhm != sci_spec._inst_fwhm):
            sky_spec = sky_spec.matchFWHM(target_FWHM=sci_spec._inst_fwhm)
        
        output_path = os.path.abspath(os.path.dirname(out_cont_line))
        pars_out, skycorr_fit = run_skycorr(
            skycorr_config_path=skycorr_config,
            sci_spec=sci_spec,
            sky_spec=sky_spec,
            specs_dir=output_path,
            out_dir=output_path,
            spec_label=os.path.basename(out_cont_line).replace(".fits", ""),
            MJD=sky_spec._header["MJD"],
            TIME=(Time(sky_spec._header["MJD"], format="mjd").to_datetime() - datetime.fromisoformat('1970-01-01 00:00:00')).days*24*3600,
            TELALT=sky_spec._header["ALT"],
            WLG_TO_MICRON=1e-4,
            FWHM=sky_spec._inst_fwhm.max()/np.diff(sky_spec._wave).min(),
        )

        wavelength = skycorr_fit["lambda"]
        sky_cont = Spectrum1D(wave=wavelength, data=skycorr_fit["mcflux"], error=None, mask=None, inst_fwhm=sky_spec._inst_fwhm)
        sky_line = Spectrum1D(wave=wavelength, data=skycorr_fit["mlflux"], error=None, mask=None, inst_fwhm=sky_spec._inst_fwhm)
        # TODO: implement skycorr method output

    # run model
    elif method == "model":
        prefix = "SM"
        # TODO: use the master sky parameters (datetime, observing conditions: lunation, moon distance, etc.) evaluate a sky model
        # TODO: use the resulting model continuum as physical representation of the target sky continuum
        # TODO: remove continuum contribution from original sky spectrum
        resample_step, resolving_power = np.diff(sky_spec._wave).min(), int(np.ceil((sky_spec._wave/np.diff(sky_spec._wave).min()).max()))
        # BUG: implement missing parameters in this call of run_skymodel
        skymodel_pars = skymodel_pars_from_header(sky_spec._head)
        inst_pars, model_pars, sky_model = run_skymodel(
            limlam=[sky_spec._wave.min()/1e4, sky_spec._wave.max()/1e4],
            dlam=resample_step/1e4,
            **skymodel_pars
        )
        pars_out = {}
        pars_out.update(inst_pars)
        pars_out.update(model_pars)
        # TODO: the predicted continuum would be the full radiative component - airglow line
        # TODO: scale the predicted continuum with the sky_ref
        sky_cont = Spectrum1D(
            wave=sky_model["lam"].value,
            data=sky_model["flux"].value - sky_model["flux_ael"].value,
            error=(sky_model["dflux2"] - sky_model["dflux1"]).value/2,
            inst_fwhm=sky_model["lam"].value / resolving_power
        )
        sky_cont._mask = np.isnan(sky_cont._data)
        # resample and match in spectral resolution sky model as needed
        if np.any(sky_cont._wave != sky_spec._wave):
            sky_cont = sky_cont.resampleSpec(ref_wave=sky_spec._wave, method="linear")
        if np.any(sky_cont._inst_fwhm != sky_spec._inst_fwhm):
            sky_cont = sky_cont.smoothGaussVariable(diff_fwhm=np.sqrt(sky_spec._inst_fwhm**2 - sky_cont._inst_fwhm**2))

        # calculate the line component
        sky_line = sky_spec - sky_cont
    # run fit
    elif method == "fit":
        # TODO: build a sky model library with continuum and line separated (ESO skycalc)
        # TODO: use this library as templates to fit master skies
        # TODO: check if we can recover observing condition parameters from this fit
        raise NotImplementedError("This method of continuum/line separation is not implemented yet.")
    else:
        raise ValueError(f"Unknown method '{method}'. Valid mehods are: 'skycorr' (default), 'model' and 'fit'.")
    
    # TODO: explore the MaNGA way: sigma-clipping the lines and then smooth high-frequency features so that we get a continuum estimate
    # pack outputs in FITS file
    rss_cont_line = RSS.from_spectra1d((sky_cont, sky_line))
    
    header = sky_spec._header
    for key, val in pars_out.items():
        if isinstance(val, (list,tuple)):
            val = ",".join(map(str, val))
        elif isinstance(val, str) and (os.path.isfile(val) or os.path.isdir(val)):
            val = os.path.basename(val)
        header.append((f"HIERARCH {prefix} {key.upper()}", val))

    rss_cont_line.setHeader(header, origin=sky_ref)
    rss_cont_line.writeFitsData(out_cont_line)


def evalESOSky_drp(sky_ref, out_rss, resample_step="optimal", resample_method="linear", err_sim='500', replace_error='1e10', parallel="auto"):
    """
    
        Evaluates the ESO sky model following the observing conditions in the given sky reference.
        The output contains the calculated components of the sky in a RSS format. In addition a
        'fibermap' table is stored in the second HDU, to keep track of the meaning of each row.

        The wavelength sampling and resolution of the returned model components will always match
        that of the input 'sky_ref'. However, the sampling and resolution of the original sky model
        can be controlled by the user. It is always desirable that the sampling and resolution of
        this original model exceeds those of the reference spectrum, so there is no loss of
        information when matching the wavelength vector to the reference. The user can control the
        wavelength sampling of the sky model components by specifying the 'sampling_step'. By
        setting sampling_step='optimal' (default), sampling will be defined using the input
        'sky_ref' spectrum in two possible ways. If the input spectrum contains the LSF, the
        optimal sampling will be computed to be 1/3 of the maximum resolution following the
        criteria in the sampling theorem. Otherwise, the optimal sampling will be computed to be
        the smallest sampling step in the reference spectrum. For a more seasoned users, the
        sampling_step can also take a floating point value, which is going to be used to produce a
        model for the sky spectra components.
        
        The original model resolution will be either the best resolution from the reference spectrum
        (if the LSF is present), or max( wavelength_ref / sampling_step ). Again, this ensures there
        is a minimum loss of information when matching the original model wavelength vector to the
        reference.

        When resampling the original model components, the user can specify if this is done linearly
        (resample_method='linear') or using a spline (='spline'). To accurately propagate the errors
        during the resampling process, a Monte Carlo method is addopted and the user can specify the
        number of realisations using 'err_sim'. Missing values in the error can be replaced with the
        'replace_error' parameter.

        Parameters
        ----------
        sky_ref : string
            path to the reference spectrum from which observing conditions and ephemeris can be
            inferred to evaluate a ESO model spectrum
        out_rss : string
            path where the output RSS file will be saved
        resample_step : string, optional with default 'optimal'
            the resample step or method to use when interpolating the model in the sky reference
            wavelength
        resample_method : string of 'linear' (default) or 'spline'
            interpolation method to use
        err_sim : float, optional with default 500
            number of MC to propagate the error in the spectrum when interpolating
        replace_error : float, optional with default 1e10
            value to replace missing error values
        parallel : string or integer with default 'auto'
            whether to run the interpolation in parallel in a given number of threads or
            in a serial way (parallel=1)

        Examples
        --------

        user:> drp sky evalESOSky SKY_REF.fits out_rss.fits
    """

    # read sky spectrum
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky_ref, extension_hdr=0)
    
    eval_failed = False
    if resample_step != "optimal":
        try:
            resample_step = eval(resample_step)
        except ValueError:
            eval_failed = True
            sky_logger.error(f"resample_step should be either 'optimal' or a floating point. '{resample_step}' is none.")
            sky_logger.warning("falling back to resample_step='optimal'")
    if eval_failed or resample_step == "optimal":
        # determine sampling based on wavelength resolution
        # if not present LSF in reference spectrum, use the reference sampling step
        if sky_spec._inst_fwhm is not None:
            resample_step = np.min(sky_spec._inst_fwhm) / 3
        else:
            resample_step = np.min(np.diff(sky_spec._wave))
    
    new_wave = np.arange(sky_spec._wave.min(), sky_spec._wave.max() + resample_step, resample_step)
    
    # get skymodel parameters from header
    skymodel_pars = skymodel_pars_from_header(header=sky_spec._header)

    # TODO: move unit and data type conversions to within the run_skymodel routine
    inst_pars, model_pars, sky_model = run_skymodel(
        skymodel_path=SKYMODEL_INST_PATH,
        # instrument parameters
        limlam=[new_wave.min()/1e4, new_wave.max()/1e4],
        dlam=resample_step/1e4,
        # sky model parameters
        **skymodel_pars
    )
    pars_out = {}
    pars_out.update(inst_pars)
    pars_out.update(model_pars)
    
    # create RSS
    wav_comp = sky_model["lam"].value
    lsf_comp = sky_model["lam"].value / pars_out["resol"].value
    sky_model.remove_column("lam")

    msk_comp = np.zeros_like(wav_comp, dtype=bool)
    err_radi = (sky_model["dflux2"] - sky_model["dflux1"]) / 2
    err_tran = (sky_model["dtrans2"] - sky_model["dtrans2"]) / 2
    sky_model.remove_columns(["dflux1", "dflux2", "dtrans1", "dtrans2"])

    sed_comp = sky_model.as_array().T

    nradi = len(list(filter(lambda c: c.startswith("flux"), sky_model.columns)))
    ntran = len(list(filter(lambda c: c.startswith("trans"), sky_model.columns)))
    err_comp = np.row_stack((
        np.tile(err_radi, (nradi, 1)),
        np.tile(err_tran, (ntran, 1))
    ))
    # create initial RSS containing the sky model components
    spectra_list = [Spectrum1D(wave=wav_comp, data=sed, error=err, mask=msk_comp, inst_fwhm=lsf_comp) for sed, err in zip(sed_comp, err_comp)]
    
    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    # resample RSS to reference wavelength sampling
    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        for i in range(len(spectra_list)):
            threads.append(pool.apply_async(spectra_list[i].resampleSpec, (new_wave, resample_method, err_sim, replace_error)))

        for i in range(len(spectra_list)):
            spectra_list[i] = threads[i].get()
        pool.close()
        pool.join()
    else:
        for i in range(len(spectra_list)):
            spectra_list[i] = spectra_list[i].resampleSpec(new_wave)
    
    # convolve RSS to reference LSF
    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        for i in range(len(spectra_list)):
            threads.append(pool.apply_async(spectra_list[i].matchFWHM, (sky_spec._inst_fwhm)))

        for i in range(len(spectra_list)):
            spectra_list[i] = threads[i].get()
        pool.close()
        pool.join()
    else:
        for i in range(len(spectra_list)):
            spectra_list[i] = spectra_list[i].matchFWHM(sky_spec._inst_fwhm)
    
    # build RSS
    rss = RSS.from_spectra1d(spectra_list=spectra_list)
    rss.setHeader(fits.Header(pars_out))
    # dump RSS file containing the
    rss.writeFitsData(filename=out_rss)


def subtractGeocoronal_drp():
    pass


def corrSkyLine_drp(sky1_line_in, sky2_line_in, sci_line_in, line_corr_out, method="distance", sky_models_in="", sci_model_in="", skycorr_config=SKYCORR_CONFIG_PATH):
    """

        Combines the two master sky line components a as weighted average, where the weights are
        determined depending on the given method. Then it runs the ESO skycorr routine to return
        the final version of the sky line component for the science pointing.

        If method='distance', this function will calculate the spherical distance between the sky
        pointings and the science pointing and define the weights as the inverse of those
        distances.

        If method='model', this method will use the given sky models in paths 'sky_models_in' and
        'sci_model_in' to calculate the weights as a scaling factor between the models of each sky
        pointing and the science pointing.

        The extrapolated sky line component will be calculated as a weighted average:

            sky_line = w_1 * sky1_line + w_2 * sky2_line

        Then 'sky_line' will be passed to the ESO skycorr routine to produce the final sky line
        component for the science pointing

        Parameters
        ----------
        sky1_line_in, sky2_line_in, sci_line_in : string
            paths to the sky line components for the sky and the science pointings, respectively
        line_corr_out : string
            path to file where the output line component will be stored
        method : string of 'distance' (default) or 'model'
            method used to calculate the weights
        sky_models_in, sci_model_in : strings
            needed to calculate the weights if method='model'
        skycorr_config : string, optional with default {SKYCORR_CONFIG_PATH}
            path to skycorr configuration file
        
        Examples
        --------
        user:> drp sky corrSkyLine SKY1_LINE.fits SKY2_LINE.fits SCI_LINE.fits LINE_OUT.fits
    
    """
    # BUG: skycorr should be run on each sky pointing and then we have to figure out how to combine them to produce the final sky_line_corr

    # read sky spectra
    sky1_line = Spectrum1D()
    sky1_line.loadFitsData(sky1_line_in)
    sky1_head = Header()
    sky1_head.loadFitsHeader(sky1_line_in)

    sky2_line = Spectrum1D()
    sky2_line.loadFitsData(sky2_line_in)
    sky2_head = Header()
    sky2_head.loadFitsHeader(sky2_line_in)

    # read science spectra
    sci_line = Spectrum1D()
    sci_line.loadFitsData(sci_line_in)
    sci_head = Header()
    sci_head.loadFitsHeader(sci_line_in)

    # sky1 position
    if method == "distance":
        ra_1, dec_1 = sky1_head["RA"], sky1_head["DEC"]
        # sky2 position
        ra_2, dec_2 = sky2_head["RA"], sky2_head["DEC"]
        # sci position
        ra_s, dec_s = sci_head["RA"], sci_head["DEC"]

        w_1 = 1 / ang_distance(ra_1, dec_1, ra_s, dec_s)
        w_2 = 1 / ang_distance(ra_2, dec_2, ra_s, dec_s)
        w_norm = w_1 + w_2
        w_1, w_2 = w_1 / w_norm, w_2 / w_norm
    elif method == "model":
        if sky_models_in != "":
            sky_models_in = sky_models_in.split(",")
            if len(sky_models_in) == 1:
                sky_models_in = 2 * sky_models_in

            # BUG: I cannot index xxx_model.loadFitsData(...) because that is an in-place operation
            sky1_model = RSS()
            sky1_model.loadFitsData(sky_models_in[0])[1]
            sky2_model = RSS()
            sky2_model.loadFitsData(sky_models_in[1])[1]
        else:
            # TODO: fall back to closest sky model if not given filenames
            pass
    
        if sci_model_in != "":
            sci_model = RSS()
            sci_model.loadFitsData(sci_model_in)[1]
        else:
            # TODO: fall back to closest sky model to science target
            pass

        w_1 = sci_model / sky1_model
        w_2 = sci_model / sky2_model
    elif method == "interpolate":
        raise NotImplementedError(f"method '{method}' is not implemented yet")
    else:
        raise ValueError(f"Unknown method '{method}'. Valid mehods are: 'distance' (default), 'model' and 'interpolate'.")

    # TODO: make sure all these spectra are in the same wavelength sampling
    wl_master_sky = sci_line._wave

    # compute a weighted average using as weights the inverse distance to science
    sky_line = w_1 * sky1_line + w_2 * sky2_line
    
    # run skycorr on averaged line spectrum
    pars_out, line_fit = run_skycorr(skycorr_config=skycorr_config, wl=wl_master_sky, sci_spec=sci_line, sky_spec=sky_line)

    # create RSS
    wav_fit = line_fit["lambda"].value
    lsf_fit = line_fit["lambda"].value / pars_out["wres"].value
    sed_fit = line_fit.as_array()[:,1].T
    hdr_fit = fits.Header(pars_out)
    rss = RSS(data=sed_fit, wave=wav_fit, inst_fwhm=lsf_fit, header=hdr_fit)

    # dump RSS file containing the model sky line spectrum
    rss.writeFitsData(filename=line_corr_out)


def corrSkyContinuum_drp(sky1_cont_in, sky2_cont_in, sci_cont_in, cont_corr_out, method="model", sky_models_in="", sci_model_in="", model_fiber=1):
    """

        Combines the sky continuum components from the sky pointings into a final model for the science pointing.

        Given the sky models for the sky and the science pointings, this function will extrapolate the sky continuum components
        in the science pointing as a weighted average, where the weights are a scaling factor between the sky pointings and the
        science pointing:

            w_1 = sci_model / sky1_model
            w_2 = sci_model / sky2_model

        So that the final continuum model for the science pointing is:

            sky_cont = 0.5 * (w_1 * sky1_cont + w_2 * sky2_cont)
        
        Parameters
        ----------
        sky1_cont_in, sky2_cont_in, sky1_model_in : strings
            path to the sky continuum component for the sky and the science pointings, respectively
        sky1_model_in, sky2_model_in, sci_odel_in : strings
            path to the sky model for the sky and the science pointings, respectively
        cont_corr_out : string
            path to output file where to store the extrapolated sky continuum component
        model_fiber : integer, with default 1
            fiber that represents the model sky spectrum in the given files
        
        Examples
        --------
        user:> drp sky corrSkyContinuum SKY1_CONT.fits SKY2_CONT.fits SKY1_MODEL.fits SKY2_MODEL.fits SCI_MODEL.fits CONT_OUT.fits

    """

    # read sky continuum from both sky telescopes
    sky1_cont = Spectrum1D()
    sky1_cont.loadFitsData(sky1_cont_in)
    sky1_head = Header()
    sky1_head.loadFitsHeader(sky1_cont_in)

    sky2_cont = Spectrum1D()
    sky2_cont.loadFitsData(sky2_cont_in)
    sky2_head = Header()
    sky2_head.loadFitsHeader(sky2_cont_in)

    # read sky continuum from science telescope
    sci_cont = Spectrum1D()
    sci_cont.loadFitsData(sci_cont_in)
    sci_head = Header()
    sci_head.loadFitsHeader(sci_cont_in)

    # read sky models for all pointings
    if method == "model":
        if sky_models_in != "":
            sky_models_in = sky_models_in.split(",")
            if len(sky_models_in) == 1:
                sky_models_in = 2 * sky_models_in

            sky1_model = RSS()
            sky1_model.loadFitsData(sky_models_in[0])
            sky2_model = RSS()
            sky2_model.loadFitsData(sky_models_in[1])
        else:
            # TODO: fall back to closest sky model if not given filenames
            pass
    
        if sci_model_in != "":
            sci_model = RSS()
            sci_model.loadFitsData(sci_model_in)
        else:
            # TODO: fall back to closest sky model to science target
            pass

        sky1_model = sky1_model.getSpec(model_fiber)
        sky2_model = sky2_model.getSpec(model_fiber)
        sci_model = sci_model.getSpec(model_fiber)

        # match wavelength resolution and wavelenth across telescopes using science pointing as reference
        if np.any(sky1_model._wave != sci_model._wave):
            sky1_model = sky1_model.resampleSpec(sci_model._wave)
        if np.any(sky2_model._wave != sci_model._wave):
            sky2_model = sky2_model.resampleSpec(sci_model._wave)

        if np.any(sky1_model._inst_fwhm != sci_model._inst_fwhm):
            sky1_model.matchFWHM(sci_model._inst_fwhm)
        if np.any(sky2_model._inst_fwhm != sci_model._inst_fwhm):
            sky2_model.matchFWHM(sci_model._inst_fwhm)

        # TODO: weight the continuum components of each sky telescope depending on the sky quality (darker, airmass)
        # extrapolate sky pointings into science pointing
        w_1 = sci_model / sky1_model
        w_2 = sci_model / sky2_model
        # TODO: smooth high frequency features in weights

    # TODO: implement sky coordinates interpolation
    elif method == "distance":
        ra_1, dec_1 = sky1_head["RA"], sky1_head["DEC"]
        # sky2 position
        ra_2, dec_2 = sky2_head["RA"], sky2_head["DEC"]
        # sci position
        ra_s, dec_s = sci_head["RA"], sci_head["DEC"]

        w_1 = 1 / ang_distance(ra_1, dec_1, ra_s, dec_s)
        w_2 = 1 / ang_distance(ra_2, dec_2, ra_s, dec_s)
        w_norm = w_1 + w_2
        w_1, w_2 = w_1 / w_norm, w_2 / w_norm

    # TODO: implement interpolation in the parameter space
    elif method == "interpolate":
        raise NotImplementedError(f"method '{method}' is not implemented yet")
    else:
        raise ValueError(f"Unknown method '{method}'. Valid mehods are: 'distance', 'model' (default) and 'interpolate'.")

    # TODO: propagate error in continuum correction
    # TODO: propagate mask
    # TODO: propagate LSF
    cont_fit = 0.5 * (w_1 * sky1_cont + w_2 * sky2_cont)
    cont_fit.writeFitsData(cont_corr_out)


def coaddContinuumLine_drp(sky_cont_corr_in, sky_line_corr_in, sky_corr_out, line_fiber=9):
    """
    
        Coadds the corrected line and continuum components into the joint sky spectrum:
            
            sky_corr = sky_cont_corr + sky_line_corr
        
        Parameters
        ----------
        sky_cont_corr_in, sky_line_corr_in : strings
            paths to the corrected sky continuum and line components
        sky_corr_out : string
            path to output file where to store the joint sky spectrum
        line_fiber : integer with default 9
            row in the sky line RSS file that represents the model line component
        
        Examples
        --------
        user:> drp sky coadContinuumLine SKY_CONT_CORR.fits SKY_LINE_CORR.fits SKY_CORR_OUT.fits
        
    """

    # read RSS sky line contribution
    sky_cont_corr = Spectrum1D()
    sky_cont_corr.loadFitsData(sky_cont_corr_in)
    # read RSS continuum contribution
    sky_line_corr = RSS()
    sky_line_corr.loadFitsData(sky_line_corr_in)
    sky_line_corr = sky_line_corr[line_fiber]
    # coadd to build joint sky model

    sky_corr = sky_cont_corr + sky_line_corr
    # dump final sky model
    sky_corr.writeFitsData(sky_corr_out)


def subtractSky_drp(in_rss, out_rss, sky_ref, out_sky, factor='1', scale_region='', scale_ind=False, parallel='auto'):
    """

        Subtracts a (sky) spectrum, which was stored as a FITS file, from the whole RSS.
        The error will be propagated if the spectrum AND the RSS contain error information.

        Parameters
        --------------
        in_rss : string
            input RSS FITS file
        out_rss : string
            output RSS FITS file with spectrum subtracted
        sky_ref : string
            input sky spectrum in FITS format.
        out_sky : string
            output file to store the RSS sky spectra.
        factor : string of float, optional with default: '1'
            the default value for the flux scale factor in case the fitting fails
        scale_region : string of tuple of floats, optional with default: ''
            the wavelength range within which the 'factor' will be fit
        scale_ind : boolean, optional with deafult: False
            whether apply factors individually or apply the median of good factors
        parallel : either string of integer (>0) or  'auto', optional with default: 'auto'
            number of CPU cores used in parallel for the computation. If set to 'auto', the maximum
            number of CPUs for the given system is used

        Examples
        ----------------
        user:> drp sky subtractSkySpec in_rss.fits out_rss.fits SKY_SPEC.fits

    """

    factor = np.array(factor).astype(np.float32)
    scale_ind = bool(scale_ind)
    if scale_region != '':
        region = scale_region.split(',')
        wave_region = [float(region[0]), float(region[1])]
    rss = RSS()
    rss.loadFitsData(in_rss)
    
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky_ref)
    
    sky_head = Header()
    sky_head.loadFitsHeader(sky_ref)
    
    sky_rss = RSS(
        data=np.zeros_like(rss._data),
        wave=np.zeros_like(rss._wave),
        inst_fwhm=np.zeros_like(rss._inst_fwhm),
        error=np.zeros_like(rss._error),
        mask=np.zeros_like(rss._mask, dtype=bool),
        header=sky_head
    )

    if np.all(rss._wave==sky_spec._wave) and scale_region != '':
        factors=np.zeros(len(rss), dtype=np.float32)
        for i in range(len(rss)):
            try:
                optimum = optimize.fmin(optimize_sky, [1.0], args=(rss[i], sky_spec, wave_region[0], wave_region[1]), disp=0)
                factors[i] = optimum[0]
            except RuntimeError:
                factors[i] = 1.0
                rss._mask[i, :] = True
        select_good = factors > 0.0
        scale_factor = np.median(factors[select_good])
        for i in range(len(rss)):
            if scale_ind:
                sky_rss[i] = sky_spec * factors[i]
                rss[i] = rss[i] - sky_rss[i]
            else:
                if factors[i] > 0:
                    sky_rss[i] = sky_spec * np.median(factors[select_good])
                    rss[i] = rss[i] - sky_rss[i]
    elif np.all(rss._wave == sky_spec._wave) and scale_region == '':
        for i in range(len(rss)):
            sky_rss[i] = sky_spec * factor
            rss[i] = rss[i] - sky_rss[i]
        scale_factor = factor

    if len(rss._wave) == 2:
        if parallel == 'auto':
            pool = Pool(cpu_count())
        else:
            pool = Pool(int(parallel))
        threads = []
        for i in range(len(rss)):
            threads.append(pool.apply_async(sky_spec.binSpec, args=([rss[i]._wave])))
        pool.close()
        pool.join()

        for i in range(len(rss)):
            if scale_ind:
                sky_rss[i] = threads[i].get() * factors[i]
                rss[i] = rss[i] - sky_rss[i]
            else:
                sky_rss[i] = threads[i].get() * np.median(factors[select_good])
                if factors[i] > 0:
                    rss[i] = rss[i] - sky_rss[i]

    if scale_region != '':
        rss.setHdrValue('HIERARCH PIPE SKY SCALE', float('%.3f'%scale_factor), 'sky spectrum scale factor')
    rss.writeFitsData(out_rss)
    sky_rss.writeFitsData(out_sky)


def refineContinuum_drp():
    """
    optionally apply an extra residual continuum subtraction using the faintest (i.e. with no stellar light detection) spaxels in the science IFU

    This relies in the availability of dark enough spaxels in the science pointing.
    """
    pass


def subtractPCAResiduals_drp():
    """PCA residual subtraction"""
    pass
