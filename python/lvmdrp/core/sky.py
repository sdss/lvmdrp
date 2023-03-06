# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: drp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import os
import json
import yaml
from io import BytesIO
import shutil
import subprocess
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from astropy import units as u
from skycalc_cli.skycalc import SkyModel, AlmanacQuery
from skycalc_cli.skycalc_cli import fixObservatory

from lvmdrp.core.constants import ALMANAC_CONFIG_PATH, SKYCALC_CONFIG_PATH
from lvmdrp.core.constants import SKYMODEL_INST_PATH, SKYCORR_INST_PATH, SKYCORR_CONFIG_PATH, SKYCORR_PAR_MAP
from lvmdrp.core.constants import SKYMODEL_INST_CONFIG_PATH, SKYMODEL_MODEL_CONFIG_PATH
from lvmdrp.external.skycorr import fitstabSkyCorrWrapper, createParFile, runSkyCorr

from lvmdrp.utils.logger import get_logger


sky_logger = get_logger(name=__name__)


def ang_distance(r1, d1, r2, d2):
    '''
    distance(r1,d1,r2,d2)
    Return the angular offset between two ra,dec positions
    All variables are expected to be in degrees.
    Output is in degrees

    Note - This routine could easily be made more general
    
    author: Knox Long

    '''
    RADIAN = 57.29578

    r1 = r1 / RADIAN
    d1 = d1 / RADIAN
    r2 = r2 / RADIAN
    d2 = d2 / RADIAN
    xlambda = np.sin(d1)*np.sin(d2)+np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
    if xlambda >= 1.0:
        xlambda = 0.0
    else:
        xlambda = np.arccos(xlambda)

    xlambda = xlambda * RADIAN

    return xlambda


def read_skymodel_par(parfile_path, verify=True):
    """Returns a dictionary with the ESO skymodel input .par file contents
    
    Parameters
    ----------
    parfile: string
        path to the configuration (.par) file
    verify: boolean
        whether to verify or not the integrity of the configuration file. Defaults to True
    """
    config = {}
    with open(parfile_path, "r") as f:        
        lines = f.readlines()
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            key, val = list(map(str.strip, line.split("=")))
            vals = val.split()
            if len(vals) != 1:
                val_new = []
                for val in vals:
                    if val.replace(".", "").isnumeric():
                        val_new.append(eval(val))
                    else:
                        val_new.append(val)
            else:
                if val.replace(".", "").isnumeric():
                    val_new = eval(val)
                else:
                    val_new = val

            config[key] = val_new
    
    # TODO: verify integrity of the configuration parameters
    if verify:
        pass
    
    # TODO: add units support

    return config


def write_skymodel_par(parfile_path, config, verify=True):
    """Writes the configuration dictionary in a given .par file(s)
    
    Parameters
    ----------
    par_path: string
        path to the output configuration file(s)
    config_dict: dict-like
        configuration dictionary to save in the given .par file
    verify: boolean
        whether to verify or not the integrity of the parameters dictionary. Dafaults to True.
    """
    # TODO: add units support

    with open(parfile_path, "w") as f:
        for key, val in config.items():
            if isinstance(val, (list, tuple)):
                vals = list(map(str, val))
                f.write(f"{key} = {' '.join(vals)}\n")
            elif isinstance(val, (str, int, float)):
                f.write(f"{key} = {val}\n")


def get_bright_fiber_selection(rss):
    """Returns a selection (mask) of fibers with known stars given an RSS object
    
    TODO: ask Kathryn about this, she and Max have made progress on identifying bright fibers
    """
    # extract tile information from rss header
    # look in Gaia tables (DB) for stars in that tile
    # identify fibers with stellar sources and build a mask (1: bright, 0: dark)
    pass


def run_skymodel(skymodel_path=SKYMODEL_INST_PATH, **kwargs):
    """run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc)
    
    Parameters
    ----------
    skymodel_path: string
        path where the main ESO sky model configuration files and scripts are installed
    **kwargs: dict_like
        configuration parameters within instrument_etc.par and skymodel_etc.par to overwrite

    Returns
    -------
    sky_metadata: fits.Header
        metadata describing sky components
    sky_components: fits.BinTableDU
        table contaning different components of the sky

    """
    # store initial current path
    curdir = os.path.abspath(os.curdir)
    # load original configuration file
    skymodel_inst_par = {}
    skymodel_model_par = {}
    skymodel_inst_par.update(read_skymodel_par(SKYMODEL_INST_CONFIG_PATH))
    skymodel_model_par.update(read_skymodel_par(SKYMODEL_MODEL_CONFIG_PATH))
    # ---------------------------------------------------------------------------------------------
     
    # update original configuration settings with kwargs ------------------------------------------
    skymodel_inst_par.update((k, kwargs[k]) for k in skymodel_inst_par.keys() & kwargs.keys())
    skymodel_inst_par.update((k, kwargs[k]) for k in skymodel_inst_par.keys() & kwargs.keys())
    # save configuration files with the names expected by calcskymodel
    write_skymodel_par(parfile_path=SKYMODEL_INST_CONFIG_PATH.replace("_ref", ""), config=skymodel_inst_par)
    write_skymodel_par(parfile_path=SKYMODEL_MODEL_CONFIG_PATH.replace("_ref", ""), config=skymodel_model_par)
    # ---------------------------------------------------------------------------------------------

    # run calcskymodel with the requested input parameters ----------------------------------------
    os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
    # clean output directory
    # shutil.rmtree("output", ignore_errors=True)
    os.makedirs("output", exist_ok=True)
    
    sky_logger.info("running skymodel from pre-computed airglow lines")
    out = subprocess.run(f"bin/calcskymodel".split(), capture_output=True)
    if out.returncode != 0 or "error" in out.stderr.decode("utf-8").lower():
        sky_logger.warning("no suitable airglow spectrum found")
        
        # extract parameters from config for radiative transfer run
        alt, time, season, resol, pwv = skymodel_model_par["alt"], skymodel_model_par["time"], skymodel_model_par["season"], skymodel_model_par["resol"], skymodel_model_par["pwv"]
        airmass = np.round(1/np.cos((90 - alt) * np.pi / 180), 1)
        resol = int(float(resol))
        pwv = int(pwv) if pwv == "-1" else float(pwv)

        sky_logger.info(f"calculating airglow lines with parameters {airmass = }, {time = }, {season = }, {resol = } {pwv = }")
        os.chdir(os.path.join(skymodel_path, "sm-01_mod1"))
        out = subprocess.run(f"bin/create_spec {airmass} {time} {season} . {resol} {pwv}".split(), capture_output=True)
        if out.returncode == 0:
            sky_logger.info("successfully finished airglow lines calculations")
        else:
            sky_logger.error("failed while running airglow lines calculations")
            sky_logger.error(out.stderr.decode("utf-8"))

        # copy library files to corresponding path according to libpath
        shutil.copytree(os.path.join(skymodel_path, "sm-01_mod1", "output"), os.path.join(skymodel_path, "sm-01_mod2", "data", "lib"))

        sky_logger.info("calculating effective atmospheric transmission")
        os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
        out = subprocess.run(f"bin/preplinetrans".split(), capture_output=True)
        if out.returncode == 0:
            sky_logger.info("successfully finished effective atmospheric transmission calculations")
        else:
            sky_logger.error("failed while running effective atmospheric transmission calculations")
            sky_logger.error(out.stderr.decode("utf-8"))

        out = subprocess.run(f"bin/calcskymodel".split(), capture_output=True)
        if out.returncode == 0:
            sky_logger.info("successfully finished 'calcskymodel'")
        else:
            os.chdir(curdir)
            sky_logger.error("failed while running 'calcskymodel'")
            sky_logger.error(out.stderr.decode("utf-8"))
            return skymodel_inst_par, skymodel_model_par, None    
    # ---------------------------------------------------------------------------------------------

    # read output files and organize in a FITS table ----------------------------------------------
    trans_table = Table(fits.getdata(os.path.join("output/transspec.fits"), ext=1))
    lines_table = Table(fits.getdata(os.path.join("output/radspec.fits"), ext=1))
    os.chdir(curdir)

    trans_table.remove_column("lam")
    sky_comps = hstack([lines_table, trans_table])
    sky_comps["lam"] = sky_comps["lam"]*1e4

    return skymodel_inst_par, skymodel_model_par, sky_comps


def run_skycorr(sci_spec, sky_spec, spec_label, skycorr_path=SKYCORR_INST_PATH, specs_dir="./", out_dir="./", **kwargs):
    # invert masks if present
    if sci_spec._mask is not None:
        sci_spec._mask = ~sci_spec._mask
    if sky_spec._mask is not None:
        sky_spec._mask = ~sci_spec._mask

    skycorr_config_ = yaml.safe_load(open(SKYCORR_CONFIG_PATH, "r"))
    # write each spectrum in skycorr individual format
    # TODO: look for actual meaning of timeVal and telAltVal (see examples)
    # TODO: deactivate the Halpha geocoronal subtraction in skycorr (is not well implemented)
    sci_fits_file, sky_fits_file = fitstabSkyCorrWrapper(
        wave=sci_spec._wave,
        objflux=sci_spec._data,
        skyflux=sky_spec._data,
        dateVal=kwargs.pop("MJD"),
        timeVal=kwargs.pop("TIME"),
        telAltVal=kwargs.pop("TELALT"),
        label=spec_label,
        specs_dir=specs_dir
    )
    
    # update config pars with keyword arguments
    skycorr_config_.update((k, kwargs[k]) for k in skycorr_config_.keys() & kwargs.keys())
    # convert from yaml to skycorr keys
    skycorr_config = {val: skycorr_config_[key] for key, val in SKYCORR_PAR_MAP.items()}
    skycorr_config["install"] = skycorr_path
    skycorr_config["objfile"] = sci_fits_file
    skycorr_config["skyfile"] = sky_fits_file

    out_file = os.path.basename(sci_fits_file.replace(".fits", "_out"))
    skycorr_config["outfile"] = out_file
    skycorr_config["outdir"] = out_dir

    # OPTIONAL CHANGE OF THESE PARS: parfile = None, timeVal = None, dateVal = None, telAltVal = None
    par_file = createParFile(**skycorr_config)
    runSkyCorr(parfile=par_file)

    # read outputs
    skycorr_fit = Table(fits.getdata(os.path.join(out_dir, f"{out_file}_fit.fits"), ext=1))
    skycorr_fit["lambda"] = skycorr_fit["lambda"] * 1e4
    
    return skycorr_config, skycorr_fit


def run_skycalc(skycalc_config=SKYCALC_CONFIG_PATH, almanac_config=ALMANAC_CONFIG_PATH, **kwargs):
    """run web version of the ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc)
    
    Parameters
    ----------
    skycalc_config: str path
        path to configuration file to run ESO skycalc
    almanac_config: str path
        path to ESO almanac configuration file
    **kwargs: dict_like
        configuration parameters to overwrite
    Returns
    -------
    sky_metadata: fits.Header
        metadata describing sky components
    sky_components: fits.BinTableDU
        table contaning different components of the sky
    """
    with open(skycalc_config, 'r') as f:
        inputdic = json.load(f)
    with open(almanac_config, 'r') as f:
        inputalmdic = json.load(f)
    
    # update dictionary with parameters passed as keyword arguments
    inputdic.update((k, kwargs[k]) for k in inputdic.keys() & kwargs.keys())
    inputalmdic.update((k, kwargs[k]) for k in inputalmdic.keys() & kwargs.keys())

    alm = AlmanacQuery(inputalmdic)
    dic = alm.query()
    # TODO: investigate if this parameter has data in the future (this may be important for geocoronal subtraction)
    if dic["msolflux"] < 0: dic["msolflux"] = 130.0

    for key, value in dic.items():
        inputdic[key] = value
    
    try:
        dic = fixObservatory(inputdic)
    except ValueError:
        raise

    sky_model = SkyModel()
    sky_model.callwith(dic)

    
    sky_hdus = fits.open(BytesIO(sky_model.data))
    sky_metadata = sky_hdus[0].header
    sky_components = Table(sky_hdus[1].data)

    # TODO: convert to flux units / normalize to apply scaling factors
    # TODO: convert wavelengths to Angstroms
    sky_components["lam"] = sky_components["lam"]*10*u.AA
    sky_components["flux"] = sky_components["flux"] * (1/u.s/u.m**2*u.arcsec**2) #photons/s/m2/μm/arcsec2

    return sky_metadata, dic, sky_components


def optimize_sky(factor, test_spec, sky_spec, start_wave, end_wave):
    """Returns the residual statistic between the test (science) spectrum and the sky spectrum within a given wavelength range

    This is a helper function to fit the 'factor', the flux scale by which the residuals between the science spectrum and the
    sky spectrum are minimized.

    """
    wave = test_spec._wave
    if test_spec._mask is not None:
        good_pix = np.logical_not(test_spec._mask)
        select1 = np.logical_and(wave>start_wave, wave<end_wave)
        if np.sum(good_pix[select1])>1:
            select = np.logical_and(select1, good_pix)
        else:
            select = select1
    else:
        select = np.logical_and(wave>start_wave, wave<end_wave)
    if (np.sum(select)==0) or (np.sum(test_spec._data[select])==0):
        raise RuntimeError
    rms = np.std(test_spec._data[select]-sky_spec._data[select]*factor)
    return rms
