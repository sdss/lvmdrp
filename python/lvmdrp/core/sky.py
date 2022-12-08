# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: drp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import os
import json, yaml
from io import BytesIO
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

from lvmdrp.core.constants import SKYCALC_CONFIG_PATH, ALMANAC_CONFIG_PATH, SKYCORR_PAR_MAP
from lvmdrp.external.skycorr import fitstabSkyCorrWrapper, createParFile, runSkyCorr
from skycalc_cli.skycalc import SkyModel, AlmanacQuery
from skycalc_cli.skycalc_cli import fixObservatory


def get_bright_fiber_selection(rss):
    """Returns a selection (mask) of fibers with known stars given an RSS object
    
    TODO: ask Kathryn about this, she and Max have made progress on identifying bright fibers
    """
    # extract tile information from rss header
    # look in Gaia tables (DB) for stars in that tile
    # identify fibers with stellar sources and build a mask (1: bright, 0: dark)
    pass


def run_skymodel(skycalc_config=SKYCALC_CONFIG_PATH, almanac_config=ALMANAC_CONFIG_PATH, return_pars=False, **kwargs):
    """run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc) to evaluate sky spectrum at each telescope pointing (model_sky1, model_sky2, model_skysci)
    
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

    if return_pars:
        return sky_metadata, sky_components, dic
    return sky_metadata, sky_components


def run_skycorr(skycorr_config, sci_spec, sky_spec, spec_label, specs_dir="./", out_dir="./", metadata={}):

    skycorr_config = yaml.safe_load(open(skycorr_config, "r"))
    # write each spectrum in skycorr individual format
    # TODO: look for actual meaning of timeVal and telAltVal (see examples)
    sci_fits_file, sky_fits_file = fitstabSkyCorrWrapper(
        wave=sci_spec._wave,
        objflux=sci_spec._data,
        skyflux=sky_spec._data,
        dateVal=metadata.get("MJD"),
        timeVal=metadata.get("TIME"),
        telAltVal=metadata.get("TELALT"),
        label=spec_label,
        specs_dir=specs_dir
    )
    
    # convert from yaml to skycorr keys
    skycorr_config_ = {val: skycorr_config[key] for key, val in SKYCORR_PAR_MAP.items()}
    skycorr_config_["objfile"] = sci_fits_file
    skycorr_config_["skyfile"] = sky_fits_file

    out_file = os.path.basename(sci_fits_file.replace(".fits", f"_out"))
    skycorr_config_["outfile"] = out_file
    skycorr_config_["outdir"] = out_dir

    # OPTIONAL CHANGE OF THESE PARS: parfile = None, timeVal = None, dateVal = None, telAltVal = None
    par_file = createParFile(**skycorr_config_)
    runSkyCorr(parfile=par_file)

    # read outputs
    skycorr_fit = Table(fits.getdata(os.path.join(out_dir, f"{out_file}_fit.fits"), ext=1))
    
    return skycorr_config_, par_file, skycorr_fit


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
