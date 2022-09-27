import os
import json, yaml
import numpy as np
from io import BytesIO
from astropy.io import fits
from astropy.table import Table
from astropy import units as u 

from lvmdrp.core.constants import SKYCALC_CONFIG_PATH, ALMANAC_CONFIG_PATH
from lvmdrp.external.skycorr import fitstabSkyCorrWrapper, createParFile, runSkyCorr
from skycalc_cli.skycalc import SkyModel, AlmanacQuery
from skycalc_cli.skycalc_cli import fixObservatory


# PAR_MAP = {
#     "objfile": "INPUT_OBJECT_SPECTRUM", "skyfile": "INPUT_SKY_SPECTRIM", "outfile": "OUTPUT_NAME", "outdir": "OUTPUT_DIR",
#     "colnames": "COL_NAMES",
#     "install": "INST_DIR", "defaultError": "DEFAULT_ERROR",
#     "wave2Micron": "WLG_TO_MICRON", "vacOrAir": "VAC_AIR", "dateKey": "DATE_KEY", "timeKey": "TIME_KEY",
#     "telAltKey": "TELALT_KEY",
#     "linetab": "LINETABNAME", "vardat": "VARDATNAME",
#     "soldaturl": "SOLDATURL",
#     "solflux": "SOLFLUX", "fwhm": "FWHM", "varfwhm": "VARFWHM", "ltol": "LTOL", "minLineDist": "MIN_LINE_DIST", "fluxLim": "FLUXLIM",
#     "ftol": "FTOL", "xtol": "XTOL", "wtol": "WTOL", "chebyMax": "CHEBY_MAX", "chebyMin": "CHEBY_MIN", "chebyConst": "CHEBY_CONST",
#     "rebinType": "REBINTYPE", "weightLim": "WEIGHTLIM", "sigLim": "SIGLIM", "fitLim": "FITLIM", "plotType": "PLOT_TYPE"}

PAR_MAP_R = {'INPUT_OBJECT_SPECTRUM': 'objfile',
             'INPUT_SKY_SPECTRUM': 'skyfile',
             'OUTPUT_NAME': 'outfile',
             'OUTPUT_DIR': 'outdir',
             'COL_NAMES': 'colnames',
             'INST_DIR': 'install',
             'DEFAULT_ERROR': 'defaultError',
             'WLG_TO_MICRON': 'wave2Micron',
             'VAC_AIR': 'vacOrAir',
             'DATE_KEY': 'dateKey',
             'TIME_KEY': 'timeKey',
             'TELALT_KEY': 'telAltKey',
             'LINETABNAME': 'linetab',
             'VARDATNAME': 'vardat',
             'SOLDATURL': 'soldaturl',
             'SOLFLUX': 'solflux',
             'FWHM': 'fwhm',
             'VARFWHM': 'varfwhm',
             'LTOL': 'ltol',
             'MIN_LINE_DIST': 'minLineDist',
             'FLUXLIM': 'fluxLim',
             'FTOL': 'ftol',
             'XTOL': 'xtol',
             'WTOL': 'wtol',
             'CHEBY_MAX': 'chebyMax',
             'CHEBY_MIN': 'chebyMin',
             'CHEBY_CONST': 'chebyConst',
             'REBINTYPE': 'rebinType',
             'WEIGHTLIM': 'weightLim',
             'SIGLIM': 'sigLim',
             'FITLIM': 'fitLim',
             'PLOT_TYPE': 'plotType'}

def _load_fiber_map():
    pass


def _load_sky_lines():
    pass


def get_sky_model(skycalc_config=SKYCALC_CONFIG_PATH, almanac_config=ALMANAC_CONFIG_PATH, return_pars=False, **kwargs):
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


def run_skycorr(config_file, wl, sci_rss, sky_rss, metadata=None):

    skycorr_config = yaml.safe_load(open(config_file, "r"))
    # write each spectrum in skycorr individual format
    # TODO: look for actual meaning of timeVal and telAltVal (see examples)
    sky_corr_rss = np.zeros_like(sci_rss)
    for i, flux in enumerate(sci_rss):
        if  metadata is not None:
            sci_fits_file, sky_fits_file = fitstabSkyCorrWrapper(wave=wl, objflux=flux, skyflux=sky_rss[i], dateVal=metadata["MJD"], timeVal=metadata["TIME"], telAltVal=metadata["TELALT"])
        else:
            sci_fits_file, sky_fits_file = fitstabSkyCorrWrapper(wave=wl, objflux=flux, skyflux=sky_rss[i], dateVal=None, timeVal=None, telAltVal=None)

        # convert from yaml to skycorr keys
        skycorr_config_ = {val: skycorr_config[key] for key, val in PAR_MAP_R.items()}
        skycorr_config_["objfile"] = sci_fits_file
        skycorr_config_["skyfile"] = sky_fits_file

        out_file = os.path.basename(sci_fits_file.replace(".fits", f"_out_{i}.fits"))
        skycorr_config_["outfile"] = out_file

        # OPTIONAL CHANGE OF THESE PARS: parfile = None, timeVal = None, dateVal = None, telAltVal = None
        par_file = createParFile(**skycorr_config_)
        runSkyCorr(parfile=par_file)

        # read outputs
        # sky_corr_fits = fits.open(os.path.join(skycorr_config_["outdir"], skycorr_config_["outfile"]))
    # store outputs in RSS

    return skycorr_config_, par_file


def select_sky_fibers(rss, fiber_map_path):
    """select sky fibers to be used to build master skies 1 and 2"""
    fiber_map = _load_fiber_map(fiber_map_path)
    sky1_fibers, sky2_fibers, sci_fibers, std_fibers = rss[fiber_map["SKY_1"]], rss[fiber_map["SKY_2"]], rss[fiber_map["SCI"]], rss[fiber_map["STD"]]
    return sky1_fibers, sky2_fibers, sci_fibers, std_fibers


def build_master_sky(sky1_fibers, sky2_fibers, metadata, method="naive"):
    """run continuum/line separation algorithm on master sky 1 and 2 and master science and produce line-only (sky1_line, sky2_line, sci_line) and continuum-only (sky1_cont, sky2_cont, sci_cont) spectra for each
    
    Parameters
    ----------
    metadata: list_like
        observing conditions for each pointing needed to evaluate ESO skycalc
    """
    if method == "naive":
        # deal with bad pixels (pixmask, pixel outliers)
        # deal with outlying fibers (remove fibers with stellar contribution)
        # match wavelength sampling between fibers
        # average the cleaned fibers
        pass
    elif method == "smart":
        # ESO way
        pass
    else:
        raise ValueError
    

def cont_line_separation(wl_master_sky, master_sky1, master_sky2, sky_lines_list, window_width):
    """average sky1_line and sky2_line into "sky_line", and run skycorr on "sky_line" and "sci_line" to produce "sky_line_corr"
    """
    # TODO: ask Brian about the sky lines list, I think they used something similar for MaNGA
    wl_sky_lines, sky_lines = _load_sky_lines(sky_lines_list)

    # the window size may be a function of the wavelength (there may be sky line groupings)
    # hw = window_width / 2
    # lines_mask = np.ones_like(wl_sky, dtype=bool)
    # for wl_sky in wl_sky_lines:
    #     lines_mask &= (wl_sky-hw<=wl_master_sky)&(wl_master_sky<=wl_sky+hw)
    # return master_sky2_cont, master_sky2_cont, master_sky1_line, master_sky2_line
    pass


def eval_eso_sky(sky1_fibers, sky2_fibers, sci_fibers, metadata):
    """run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc) to evaluate sky spectrum at each telescope pointing (model_sky1, model_sky2, model_skysci)
    


    Returns
    -------
    sky_line_corr, sky_cont_corr, model_skysci

    """
    sky_metas, sky_comps = [], []
    for meta in metadata:
        sky_meta, sky_comp = get_sky_model(**meta.__dict__)
        sky_metas.append(sky_meta)
        sky_comps.append(sky_comp)
    
    # one scaling for lines and one for the cont
    sky1_scale = None
    sky1_line_corr = sky1_scale * sky_comps[0]["flux"]/sky_comps[0]["trans"]
    sky1_cont_corr = sky1_scale * sky_comps[0]["trans"]

    sky2_scale = None
    sky2_line_corr = sky2_scale * sky_comps[1]["flux"]/sky_comps[1]["trans"]
    sky2_cont_corr = sky2_scale * sky_comps[1]["trans"]

    # eval ESO sky in sci pointings
    model_skysci = None

    # compute sky_line_corr and sky_cont_corr
    sky_line_corr = None
    sky_cont_corr = None

    return sky_line_corr, sky_cont_corr, model_skysci


def sky_cont_correct(master_sky1_cont, master_sky2_cont, model_skysci):
    """correct and combine continuum only spectra by doing:   sky_cont_corr=0.5*( sky1_cont*(model_skysci/model_sky1) + sky2_cont*(model_skysci/model_sky2)) """
    # return sky_cont_corr
    pass


def coadd_sky(sky_cont_corr, sky_line_corr):
    """coadd corrected line and continuum combined sky frames:    sky_corr=sky_cont_corr+sky_line_corr"""
    sky_corr = sky_cont_corr+sky_line_corr
    return sky_corr


def sky_lsf_matching():
    """do LSF matching of sky_corr to each science fiber and subtract the LSF-matched corrected sky spectra"""
    pass


def refine_cont_subtraction():
    """optionally apply an extra residual continuum subtraction using the faintest (i.e. with no stellar light detection) spaxels in the science IFU"""
    pass


def pca_residual_subtraction():
    """PCA residual subtraction"""
    pass


def geocoronal_subtraction():
    pass
