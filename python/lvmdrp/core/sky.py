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

from lvmdrp.core.constants import SKYMODEL_INST_PATH, SKYCORR_PAR_MAP, SKYMODEL_CONFIG_PARS
from lvmdrp.core.constants import ALMANAC_CONFIG_PATH, SKYCALC_CONFIG_PATH, SKYMODEL_CONFIG_PATH
from lvmdrp.external.skycorr import fitstabSkyCorrWrapper, createParFile, runSkyCorr

from lvmdrp.utils.logger import get_logger


sky_logger = get_logger("sky module")


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


def read_skymodel_par(parfile, verify=True):
    """Returns a dictionary with the ESO skymodel input .par file contents
    
    Parameters
    ----------
    parfile: string
        path to the configuration (.par) file
    verify: boolean
        whether to verify or not the integrity of the configuration file. Defaults to True
    """
    with open(parfile, "r") as f:
        line = f.realine()[:-1].strip()
        pars = {}
        while line:
            if line.startswith("#"): continue
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

            pars[key] = val_new

            line = f.readline()[:-1].strip()
    
    # TODO: verify integrity of the configuration parameters
    if verify:
        pass

    return pars


def write_skymodel_par(par_path, config_dict, verify=True):
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
    # split dictionary in different .par file(s)
    # verify all keywords are present for each given parameter file
    # if all parameters present, write corresponding .par file

    for parfile, expected_keys in SKYMODEL_CONFIG_PARS.items():
        if all(map(lambda key: key in config_dict.keys(), expected_keys)):
            config_par = {key: val for key, val in config_dict.items() if key in expected_keys}
            with open(os.path.join(par_path, parfile), "w") as f:
                for key, val in config_par.items():
                    if isinstance(val, list, tuple):
                        f.write(f"{key} = {' '.join(val)}\n")
                    elif isinstance(val, str):
                        f.write(f"{key} = {val}\n")


def get_bright_fiber_selection(rss):
    """Returns a selection (mask) of fibers with known stars given an RSS object
    
    TODO: ask Kathryn about this, she and Max have made progress on identifying bright fibers
    """
    # extract tile information from rss header
    # look in Gaia tables (DB) for stars in that tile
    # identify fibers with stellar sources and build a mask (1: bright, 0: dark)
    pass


# configuration files to look into:
# - instrument instrument_etc.par file (constant, LSF kernel, wavelength sampling)
# - sm_filenames.dat (paths to atmospheric library, names of tables containing data that depends on the observing conditions)
# - skymodel_etc.par (observing conditions, output columns: moon, etc.)
# - estmultiscat (run once, more than once to improve the quality scattering component)
# - preplinetrans (just once)
# - calcskymodel (within drp, looking for skymodel_etc.par)
# - outputs: radspec.fits and transspec.fits (contains same columns as skycalc)
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
    # load master configuration to get original configuration file names --------------------------
    skymodel_config_names = list(yaml.load(SKYMODEL_CONFIG_PATH, Loader=yaml.Loader).keys())
    instrument_par_name = skymodel_config_names[3]
    skymodel_par_name = skymodel_config_names[4]

    # load original configuration file
    skymodel_config = {}
    skymodel_config.update(read_skymodel_par(os.path.join("config", instrument_par_name)))
    skymodel_config.update(read_skymodel_par(os.path.join("config", skymodel_par_name)))
    alt, time, season, resol, pwv = skymodel_config["alt"], skymodel_config["time"], skymodel_config["season"], skymodel_config["resol"], skymodel_config["pwv"]
    airmass = np.sec((90 - alt) * np.pi / 180)
    # ---------------------------------------------------------------------------------------------
     
    # update original configuration settings with kwargs ------------------------------------------
    skymodel_config.update((k, kwargs[k]) for k in skymodel_config.keys() & kwargs.keys())
    # save configuration files with the names expected by calcskymodel
    write_skymodel_par(par_path="./config", config_dict=skymodel_config)
    # ---------------------------------------------------------------------------------------------

    # run calcskymodel with the requested input parameters ----------------------------------------
    os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
    # clean output directory
    shutil.rmtree("output")
    
    out = subprocess.run(f"bin/calcskymodel".split(), capture_output=True)
    if out.returncode == 0:
        sky_logger.info("successfully finished sky model calculation")
    elif "File opening failed" in out.stderr.decode("utf-8"):
        os.chdir(skymodel_path, "sm-01_mod1")
        out = subprocess.run(f"bin/create_spec {airmass} {time} {season} . {resol} {pwv}".split(), capture_output=True)
        if out.returncode == 0:
            sky_logger.info("successfully finished 'create_spec'")
        else:
            sky_logger.error("failed while running 'create_spec'")
            sky_logger.error(out.stderr.decode("utf-8"))

        # copy library files to corresponding path according to libpath
        shutil.copytree(os.path.join(skymodel_path, "sm-01_mod1", "output"), os.path.join(skymodel_path, "sm-01_mod2", "data", "lib"))

        os.chdir(skymodel_path, "sm-01_mod2")
        out = subprocess.run(f"bin/preplinetrans".split(), capture_output=True)
        if out.returncode == 0:
            sky_logger.info("successfully finished 'preplinetrans'")
        else:
            sky_logger.error("failed while running 'preplinetrans'")
            sky_logger.error(out.stderr.decode("utf-8"))

        out = subprocess.run(f"bin/calcskymodel".split(), capture_output=True)
        if out.returncode == 0:
            sky_logger.info("successfully finished 'calcskymodel'")
        else:
            sky_logger.error("failed while running 'calcskymodel'")
            sky_logger.error(out.stderr.decode("utf-8"))
    else:
        sky_logger.error("failed while running 'calcskymodel'")
        sky_logger.error(out.stderr.decode("utf-8"))

        return skymodel_config, None 
    # ---------------------------------------------------------------------------------------------

    # read output files and organize in a FITS table ----------------------------------------------
    trans_table = Table(fits.getdata(os.path.join("output/transspec.fits"), ext=1))
    lines_table = Table(fits.getdata(os.path.join("output/radspec.fits"), ext=1))

    trans_table.remove_column("lam")
    sky_comps = hstack([lines_table, trans_table])
    sky_comps["lam"] = sky_comps["lam"]*1e4

    return skymodel_config, sky_comps


def run_skycorr(skycorr_config_path, sci_spec, sky_spec, spec_label, specs_dir="./", out_dir="./", **kwargs):

    skycorr_config_ = yaml.safe_load(open(skycorr_config_path, "r"))
    # write each spectrum in skycorr individual format
    # TODO: look for actual meaning of timeVal and telAltVal (see examples)
    # TODO: deactivate the Halpha geocoronal subtraction in skycorr (is not well implemented)
    sci_fits_file, sky_fits_file = fitstabSkyCorrWrapper(
        wave=sci_spec._wave,
        objflux=sci_spec._data,
        skyflux=sky_spec._data,
        dateVal=kwargs.get("MJD"),
        timeVal=kwargs.get("TIME"),
        telAltVal=kwargs.get("TELALT"),
        label=spec_label,
        specs_dir=specs_dir
    )
    
    # convert from yaml to skycorr keys
    skycorr_config = {val: skycorr_config_[key] for key, val in SKYCORR_PAR_MAP.items()}
    skycorr_config["objfile"] = sci_fits_file
    skycorr_config["skyfile"] = sky_fits_file

    out_file = os.path.basename(sci_fits_file.replace(".fits", f"_out"))
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
