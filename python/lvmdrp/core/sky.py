# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: drp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import json
import os
import shutil
import subprocess
from datetime import timedelta
from io import BytesIO

import numpy as np
from scipy import interpolate
import yaml
from astropy import units as u
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table, hstack
from astropy.time import Time
from astropy.coordinates import SkyCoord
from skycalc_cli.skycalc import AlmanacQuery, SkyModel
from skycalc_cli.skycalc_cli import fixObservatory
from skyfield import almanac
from skyfield.positionlib import ICRS
from skyfield.api import Loader, Star, load, wgs84
from skyfield.framelib import ecliptic_frame

from lvmdrp.external import shadow_height_lib as sh

from lvmdrp.core.constants import (
    ALMANAC_CONFIG_PATH,
    EPHEMERIS_DIR,
    SKYCALC_CONFIG_PATH,
    SKYCORR_CONFIG_PATH,
    SKYCORR_INST_PATH,
    SKYCORR_PAR_MAP,
    SKYMODEL_INST_CONFIG_PATH,
    SKYMODEL_INST_PATH,
    SKYMODEL_MODEL_CONFIG_PATH,
)
from lvmdrp.external.skycorr import createParFile, fitstabSkyCorrWrapper, runSkyCorr
from lvmdrp import log


# define shadow height SH_CALCULATOR
SH_CALCULATOR = sh.shadow_calc()

# average moon distance from earth
MEAN_MOON_DIST = 384979000 * u.m
# define environment variable for CPL discovery
os.environ["LD_LIBRARY_PATH"] = os.path.join(SKYCORR_INST_PATH, "lib")


def get_sky_mask_uves(wave, width=3, threshold=2):
    """
    Generate a mask for the bright sky lines.
    mask every line at +-width, where width in same units as wave (Angstroms)
    Only lines with a flux larger than threshold (in 10E-16 ergs/cm^2/s/A) are masked
    The line list is from https://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html

    Returns a bool np.array the same size as wave with sky line wavelengths marked as True
    """
    p = os.path.join(os.getenv('LVMCORE_DIR'), 'etc', 'UVES_sky_lines.txt')
    txt = np.genfromtxt(p)
    skyw, skyf = txt[:,1], txt[:,4]
    #plt.plot(skyw, skyw*0, 'k.')
    #plt.plot(skyw, skyf, 'k.')
    select = (skyf>threshold)
    lines = skyw[select]
    # do NOT mask Ha if it is present in the sky table
    ha = (lines>6562) & (lines<6564)
    lines = lines[~ha]
    mask = np.zeros_like(wave, dtype=bool)
    if width > 0.0:
        for line in lines :
            if (line<=wave[0]) or (line>=wave[-1]):
                continue
            ii=np.where((wave>=line-width)&(wave<=line+width))[0]
            mask[ii]=True

    return mask


def get_z_continuum_mask(w):
    '''
    Some clean regions at the red edge of the NIR channel (hand picked)
    '''
    good = [[9230,9280], [9408,9415], [9464,9472], [9608,9512], [9575,9590], [9593,9603], [9640,9650], [9760,9775]]
    mask = np.zeros_like(w, dtype=bool)
    for r in good :
        if (r[0]<=w[0]) or (r[1]>=w[-1]):
            continue
        ii=np.where((w>=r[0])&(w<=r[1]))[0]
        mask[ii]=True

    # do not mask before first region
    mask[np.where(w<=good[0][0])] = True
    return mask


def moon_phase(jd, ephemeris):
    """Returns lunation for a given ephemeris and JD

    NOTES: original source from https://bit.ly/3VppeZo
    """

    ts = load.timescale()
    t = ts.tt_jd(jd)

    sun, moon, earth = ephemeris['sun'], ephemeris['moon'], ephemeris['earth']

    e = earth.at(t)
    s = e.observe(sun).apparent()
    m = e.observe(moon).apparent()

    _, slon, _ = s.frame_latlon(ecliptic_frame)
    _, mlon, _ = m.frame_latlon(ecliptic_frame)
    phase = (mlon.degrees - slon.degrees) #% 360.0

    return phase, m.fraction_illuminated(sun)


def moon_separation(jd, ephemeris, ra, dec):
    """Returns moon separation from given target"""

    e = ephemeris["earth"].at(load.timescale().tt_jd(jd))

    dk = ICRS.from_radec(ra, dec)

    om = e.observe(ephemeris["moon"])
    ra, dec, _ = om.radec()
    sep = dk.separation_from(ICRS.from_radec(ra_hours=ra.hours, dec_degrees=dec.degrees)).degrees

    return sep


def ang_distance(r1, d1, r2, d2):
    """
    distance(r1,d1,r2,d2)
    Return the angular offset between two ra,dec positions
    All variables are expected to be in degrees.
    Output is in degrees

    Note - This routine could easily be made more general

    author: Knox Long

    """
    RADIAN = 57.29578

    r1 = r1 / RADIAN
    d1 = d1 / RADIAN
    r2 = r2 / RADIAN
    d2 = d2 / RADIAN
    xlambda = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
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
                    if val.lower().replace(".", "").replace("e", "").isnumeric():
                        val_new.append(eval(val))
                    else:
                        val_new.append(val)
            else:
                if val.lower().replace(".", "").replace("e", "").isnumeric():
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
            else:
                f.write(f"{key} = {val}\n")


def skymodel_pars_header(header):
    """Writes skymodel and useful sky info into header
    To run the ESO skymodel, several parameters are needed.  These are calculated
    and stored in the output (see ESO skymodel docu for more details).
    Additionally useful parameters needed for analyzing sky subtraction are provided

    Parameters
    ----------
    header: fits header
        header with needed metadata info for given observation
    skymodel_pars: dict
        output dict with header keywords, values and comments
        None/NAN/Null values replaced with -999

    """

    # extract useful header information
    sci_ra, sci_dec = header["TESCIRA"], header["TESCIDE"]
    skye_ra, skye_dec = header["TESKYERA"], header["TESKYEDE"]
    skyw_ra, skyw_dec = header["TESKYWRA"], header["TESKYWDE"]
    obstime = Time(header["OBSTIME"], scale="tai")

    # define ephemeris object
    ephemeris_loader = Loader(EPHEMERIS_DIR)
    astros = ephemeris_loader("de421.bsp")
    sun, earth, moon = astros["sun"], astros["earth"], astros["moon"]
    # define location
    obs_topos = wgs84.latlon(
        latitude_degrees=SH_CALCULATOR.observatory_topo.latitude.degrees,#.to(u.deg),
        longitude_degrees=SH_CALCULATOR.observatory_topo.longitude.degrees,#.to(u.deg),
        elevation_m=SH_CALCULATOR.observatory_elevation.value,
    )
    obs = earth + obs_topos
    # define observation datetime
    ts = ephemeris_loader.timescale()
    obs_time = ts.from_astropy(obstime)
    # define observatory object
    obs = obs.at(obs_time)

    # define astros
    s, m = obs.observe(sun).apparent(), obs.observe(moon).apparent()

    # define targets for sci, skye, skyw
    try:
        sci_target_ra, sci_target_dec = sci_ra * u.deg, sci_dec * u.deg
        sci_target = Star(ra_hours=sci_target_ra.to(u.hourangle), dec_degrees=sci_target_dec.to(u.deg))
        sci_t = obs.observe(sci_target).apparent()
    except Exception:
        sci_t = np.nan

    try:
        skye_target_ra, skye_target_dec = skye_ra * u.deg, skye_dec * u.deg
        skye_target = Star(ra_hours=skye_target_ra.to(u.hourangle), dec_degrees=skye_target_dec.to(u.deg))
        skye_t = obs.observe(skye_target).apparent()
    except Exception:
        skye_t = np.nan

    try:
        skyw_target_ra, skyw_target_dec = skyw_ra * u.deg, skyw_dec * u.deg
        skyw_target = Star(ra_hours=skyw_target_ra.to(u.hourangle), dec_degrees=skyw_target_dec.to(u.deg))
        skyw_t = obs.observe(skyw_target).apparent()
    except Exception:
        skyw_t = np.nan

    # observatory height ('sm_h' in km)
    sm_h = SH_CALCULATOR.observatory_elevation

    # altitude of object above the horizon (alt, 0 -- 90)
    sci_alt, sci_az, _ = sci_t.altaz()
    skye_alt, skye_az, _ = skye_t.altaz()
    skyw_alt, skyw_az, _ = skyw_t.altaz()

    # separation between moon and sun from earth ('alpha', 0 -- 360, >180 for waning moon)
    alpha = s.separation_from(m)

    # separation between moon and object ('rho', 0 -- 180)
    sci_rho = sci_t.separation_from(m)
    skye_rho = skye_t.separation_from(m)
    skyw_rho = skyw_t.separation_from(m)

    # altitude of moon ('altmoon', -90 -- 90)
    altmoon, _, moondist = m.altaz()

    # RA and dec of moon (moon_RA, moon_dec)
    moonra, moondec, _ = m.radec()

    # Moon phase in deg (0:new, 90: first qt, 180: full, 270: 3rd qt)
    _, slon, _ = s.frame_latlon(ecliptic_frame)
    _, mlon, _ = m.frame_latlon(ecliptic_frame)
    moon_phase = (mlon.degrees - slon.degrees) % 360.0

    # Moon illumination fraction
    moon_fli = m.fraction_illuminated(sun)

    # altitude of sun ('altsun', -90 -- 90)
    altsun, _, _ = s.altaz()

    # distance to moon ('moondist', 0.91 -- 1.08; 1: mean distance)
    moondist = moondist.to(u.m) / MEAN_MOON_DIST

    # TODO: - ** pressure at observatory altitude ('pres' in hPa)
    # TODO: - ** calculation of double scattering of moonlight ('calcds', Y or N)

    # heliocentric ecliptic longitude of object ('lon_ecl', -180 -- 180)
    # heliocentric ecliptic latitude of object ('lat_ecl', -90 -- 90)
    sci_lon_ecl, sci_lat_ecl, _ = sci_t.frame_latlon(ecliptic_frame)
    skye_lon_ecl, skye_lat_ecl, _ = skye_t.frame_latlon(ecliptic_frame)
    skyw_lon_ecl, skyw_lat_ecl, _ = skyw_t.frame_latlon(ecliptic_frame)

    # TODO: - ** monthly-averaged solar radio flux ('msolflux' in sfu)
    # TODO: pull this from header if it already exists

    # bimonthly period ('season'; 1: Dec/Jan, ..., 6: Oct/Nov; 0 entire year)
    month = obs_time.to_astropy().to_datetime().month
    if month in [12, 1]:
        season = 1
    elif month in [2, 3]:
        season = 2
    elif month in [4, 5]:
        season = 3
    elif month in [6, 7]:
        season = 4
    elif month in [8, 9]:
        season = 5
    elif month in [10, 11]:
        season = 6
    else:
        season = 0

    # time of the observation ('time' in x/3 of the night; 0: entire night)
    t_ini, t_fin = obs_time - timedelta(days=2), obs_time + timedelta(days=2)

    risings_and_settings, _ = almanac.find_discrete(
        t_ini, t_fin, almanac.sunrise_sunset(ephemeris=astros, topos=obs_topos)
    )
    i = np.digitize(obs_time.tt, bins=risings_and_settings.tt, right=False)
    risings_and_settings[i - 1].tt <= obs_time.tt < risings_and_settings[i].tt, _[
        [i - 1, i]
    ]

    night_thirds = np.linspace(*risings_and_settings[[i - 1, i]].tt, 4)
    time = np.digitize(obs_time.tt, bins=night_thirds)
    # assume whole night if the target obstime was observed during 'daylight'
    if time == 4:
        time = 0

    # vacuum or air wavelengths ('vac_air', vac or air)
    # precipitable water vapour ('pwv' in mm; -1: bimonthly mean)
    # NOTE: the following is for MOD1 when we run the RT code
    # TODO: - ** radiative transfer code for molecular spectra ('rtcode', L or R)
    # TODO: - ** resolving power of molecular spectra in library ('resol')
    # TODO: - ** sky model components


    skymodel_pars = {
        "HIERARCH SKYMODEL SM_H": (sm_h.to(u.km).value, "observatory height [km]"),
        "HIERARCH SKYMODEL SM_HMIN": ((2.0 * u.km).value, "lower height limit [km]"),
        "HIERARCH SKYMODEL SCI_ALT": (np.round(sci_alt.to(u.deg).value, 4), "altitude of object above horizon [deg]"),
        "HIERARCH SKYMODEL SKYE_ALT": (np.round(skye_alt.to(u.deg).value, 4), "altitude of object above horizon [deg]"),
        "HIERARCH SKYMODEL SKYW_ALT": (np.round(skyw_alt.to(u.deg).value, 4), "altitude of object above horizon [deg]"),
        "HIERARCH SKYMODEL ALPHA": (np.round(alpha.to(u.deg).value, 4), "separation of Sun and Moon from Earth [deg]"),
        "HIERARCH SKYMODEL SCI_RHO": (np.round(sci_rho.to(u.deg).value, 4), "separation of Moon and object [deg]"),
        "HIERARCH SKYMODEL SKYE_RHO": (np.round(skye_rho.to(u.deg).value, 4), "separation of Moon and object [deg]"),
        "HIERARCH SKYMODEL SKYW_RHO": (np.round(skyw_rho.to(u.deg).value, 4), "separation of Moon and object [deg]"),
        "HIERARCH SKYMODEL MOONALT": (np.round(altmoon.to(u.deg).value, 4), "altitude of Moon above horizon [deg]"),
        "HIERARCH SKYMODEL SUNALT": (np.round(altsun.to(u.deg).value, 4), "altitude of Sun above horizon [deg]"),
        "HIERARCH SKYMODEL MOONDIST": (np.round(moondist.value, 4), "distance to Moon (mean distance=1)"),
        "HIERARCH SKYMODEL PRES": ((744 * u.hPa).value, "pressure at observer altitude, set: 744 [hPa]"),
        "HIERARCH SKYMODEL SSA": (0.97, "aerosols' single scattering albedo, set: 0.97"),
        "HIERARCH SKYMODEL CALCDS": ( "N", "cal double scattering of Moon (Y or N)"),
        "HIERARCH SKYMODEL O2COLUMN": (1.0, "relative ozone column density (1->258) [DU]"),
        "HIERARCH SKYMODEL MOONSCAL": (1.0, "scaling factor for scattered moonlight"),
        "HIERARCH SKYMODEL SCI_LON_ECL": (np.round(sci_lon_ecl.to(u.deg).value, 5), "heliocen ecliptic longitude [deg]"),
        "HIERARCH SKYMODEL SCI_LAT_ECL": (np.round(sci_lat_ecl.to(u.deg).value, 5), "ecliptic latitude [deg]"),
        "HIERARCH SKYMODEL SKYE_LON_ECL": (np.round(skye_lon_ecl.to(u.deg).value, 5), "heliocen ecliptic longitude [deg]"),
        "HIERARCH SKYMODEL SKYE_LAT_ECL": (np.round(skye_lat_ecl.to(u.deg).value, 5), "ecliptic latitude [deg]"),
        "HIERARCH SKYMODEL SKYW_LON_ECL": (np.round(skyw_lon_ecl.to(u.deg).value, 5), "heliocen ecliptic longitude [deg]"),
        "HIERARCH SKYMODEL SKYW_LAT_ECL": (np.round(skyw_lat_ecl.to(u.deg).value, 5), "ecliptic latitude [deg]"),
        "HIERARCH SKYMODEL EMIS_STR": (0.2, "grey-body emissivity"),
        "HIERARCH SKYMODEL TEMP_STR": ((290.0 * u.K).value, "grey-body temperature [K]"),
        "HIERARCH SKYMODEL MSOLFLUX": (130.0, "monthly-averaged solar radio flux, set: 130"),
        "HIERARCH SKYMODEL SEASON": (season, "bimonthly period (1:Dec/Jan, 6:Oct/Nov.; 0:year)"),
        "HIERARCH SKYMODEL TIME": (time, "period of night (x/3 night, x=1,2,3; 0:night)"),
        "HIERARCH SKYMODEL MOON_RA": (np.round(moonra.to(u.deg).value, 5), "RA of the Moon [deg]"),
        "HIERARCH SKYMODEL MOON_DEC": (np.round(moondec.to(u.deg).value, 5), "DEC of the Moon [deg]"),
        "HIERARCH SKYMODEL MOON_PHASE": (np.round(moon_phase, 2), "phase of Moon (0=new, 90=1st qt) [deg]"),
        "HIERARCH SKYMODEL MOON_FLI": (np.round(moon_fli, 4), "Moon fraction lunar illumination")
    }

    # checking for any empty, None, or NAN values and replacing with -999
    for key, value in skymodel_pars.items():
        if value[0] == '' or value[0] is None or value[0] == np.nan:
           skymodel_pars[key]=(-999, value[1])

    return skymodel_pars


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
    skymodel_inst_par.update(
        (k, kwargs[k]) for k in skymodel_inst_par.keys() & kwargs.keys()
    )
    skymodel_model_par.update(
        (k, kwargs[k]) for k in skymodel_model_par.keys() & kwargs.keys()
    )
    # save configuration files with the names expected by calcskymodel
    write_skymodel_par(
        parfile_path=SKYMODEL_INST_CONFIG_PATH.replace("_ref", ""),
        config=skymodel_inst_par,
    )
    write_skymodel_par(
        parfile_path=SKYMODEL_MODEL_CONFIG_PATH.replace("_ref", ""),
        config=skymodel_model_par,
    )
    # ---------------------------------------------------------------------------------------------

    # run calcskymodel with the requested input parameters ----------------------------------------
    os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
    # clean output directory
    # shutil.rmtree("output", ignore_errors=True)
    os.makedirs("output", exist_ok=True)

    log.info("trying skymodel from pre-computed airglow lines")
    out = subprocess.run("bin/calcskymodel".split(), capture_output=True)
    if out.returncode != 0 or "error" in out.stderr.decode("utf-8").lower():
        log.warning("no suitable airglow spectrum found")

        # extract parameters from config for radiative transfer run
        alt, time, season, resol, pwv = (
            skymodel_model_par["alt"],
            skymodel_model_par["time"],
            skymodel_model_par["season"],
            skymodel_model_par["resol"],
            skymodel_model_par["pwv"],
        )
        airmass = np.round(1 / np.cos((90 - alt) * np.pi / 180), 1)
        resol = int(float(resol))
        pwv = int(pwv) if pwv == "-1" else float(pwv)

        log.info(
            f"calculating airglow lines with parameters {airmass = }, {time = }, {season = }, {resol = } {pwv = }"
        )
        os.chdir(os.path.join(skymodel_path, "sm-01_mod1"))
        out = subprocess.run(
            f"bin/create_spec {airmass} {time} {season} . {resol} {pwv}".split(),
            capture_output=True,
        )
        if out.returncode == 0:
            log.info("successfully finished airglow lines calculations")
        else:
            log.error("failed while running airglow lines calculations")
            log.error(out.stderr.decode("utf-8"))

        # copy library files to corresponding path according to libpath
        try:
            shutil.copytree(
                os.path.join(skymodel_path, "sm-01_mod1", "output"),
                os.path.join(skymodel_path, "sm-01_mod2", "data", "lib"),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
            )
        except shutil.Error as e:
            log.warning(e.args[0])

        log.info("calculating effective atmospheric transmission")
        os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
        out = subprocess.run("bin/preplinetrans".split(), capture_output=True)
        if out.returncode == 0:
            log.info(
                "successfully finished effective atmospheric transmission calculations"
            )
        else:
            log.error(
                "failed while running effective atmospheric transmission calculations"
            )
            log.error(out.stderr.decode("utf-8"))

        out = subprocess.run("bin/calcskymodel".split(), capture_output=True)
        if out.returncode == 0:
            log.info("successfully finished 'calcskymodel'")
        else:
            os.chdir(curdir)
            log.error("failed while running 'calcskymodel'")
            log.error(out.stderr.decode("utf-8"))
            return skymodel_inst_par, skymodel_model_par, None
    else:
        log.info("successfully finished 'calcskymodel'")
    # ---------------------------------------------------------------------------------------------

    # read output files and organize in a FITS table ----------------------------------------------
    trans_table = Table(fits.getdata(os.path.join("output", "transspec.fits"), ext=1))
    lines_table = Table(fits.getdata(os.path.join("output", "radspec.fits"), ext=1))
    os.chdir(curdir)

    trans_table.remove_column("lam")
    sky_comps = hstack([lines_table, trans_table])
    sky_comps["lam"] = sky_comps["lam"] * 1e4

    return skymodel_inst_par, skymodel_model_par, sky_comps


# TODO: list a set of parameters I want the users
# to modify
def run_skycorr(
    sci_spec,
    sky_spec,
    spec_label,
    skycorr_path=SKYCORR_INST_PATH,
    specs_dir="./",
    out_dir="./",
    **kwargs,
):
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
        specs_dir=specs_dir,
    )

    # update config pars with keyword arguments
    skycorr_config_.update(
        (k, kwargs[k]) for k in skycorr_config_.keys() & kwargs.keys()
    )
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
    skycorr_fit = Table(
        fits.getdata(os.path.join(out_dir, f"{out_file}_fit.fits"), ext=1)
    )
    skycorr_fit["lambda"] = skycorr_fit["lambda"] * 1e4

    return skycorr_config, skycorr_fit


def run_skycalc(
    skycalc_config=SKYCALC_CONFIG_PATH, almanac_config=ALMANAC_CONFIG_PATH, **kwargs
):
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
    with open(skycalc_config, "r") as f:
        inputdic = json.load(f)
    with open(almanac_config, "r") as f:
        inputalmdic = json.load(f)

    # update dictionary with parameters passed as keyword arguments
    inputdic.update((k, kwargs[k]) for k in inputdic.keys() & kwargs.keys())
    inputalmdic.update((k, kwargs[k]) for k in inputalmdic.keys() & kwargs.keys())

    alm = AlmanacQuery(inputalmdic)
    dic = alm.query()
    # TODO: investigate if this parameter has data in the future (this may be important for geocoronal subtraction)
    if dic["msolflux"] < 0:
        dic["msolflux"] = 130.0

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
    sky_components["lam"] = sky_components["lam"] * 10 * u.AA
    sky_components["flux"] = sky_components["flux"] * (
        1 / u.s / u.m**2 * u.arcsec**2
    )  # photons/s/m2/μm/arcsec2

    return sky_metadata, dic, sky_components


def optimize_sky(factor, test_spec, sky_spec, start_wave, end_wave):
    """Returns the residual statistic between the test (science) spectrum and the sky spectrum within a given wavelength range

    This is a helper function to fit the 'factor', the flux scale by which the residuals between the science spectrum and the
    sky spectrum are minimized.

    """
    wave = test_spec._wave
    if test_spec._mask is not None:
        good_pix = np.logical_not(test_spec._mask)
        select1 = np.logical_and(wave > start_wave, wave < end_wave)
        if np.sum(good_pix[select1]) > 1:
            select = np.logical_and(select1, good_pix)
        else:
            select = select1
    else:
        select = np.logical_and(wave > start_wave, wave < end_wave)
    if (np.sum(select) == 0) or (np.sum(test_spec._data[select]) == 0):
        raise RuntimeError
    rms = np.std(test_spec._data[select] - sky_spec._data[select] * factor)
    return rms


def select_sky_fibers(rss, telescope, fibermap=None):
    # define fibermap
    if fibermap is None:
        fibermap = rss._slitmap
    elif fibermap is None and rss._slitmap is None:
        raise ValueError("no fibermap information found")

    # select sky fibers
    telescope = telescope.lower()
    rst_selection = fibermap["targettype"] != "SKY"
    if telescope == "both":
        sky_selection = fibermap["targettype"] == "SKY"
    elif telescope in {"east", "e", "skye"}:
        sky_selection = fibermap["telescope"] == "SkyE"
    elif telescope in {"west", "w", "skyw"}:
        sky_selection = fibermap["telescope"] == "SkyW"
    else:
        raise ValueError(f"invalid value for 'telescope' parameter: '{telescope}'")

    # define wavelength, flux and variances
    sky_wave = rss._wave[sky_selection]
    sky_data = rss._data[sky_selection]
    sky_vars = rss._error[sky_selection] ** 2
    sky_mask = rss._mask[sky_selection]
    rst_wave = rss._wave[rst_selection]
    rst_data = rss._data[rst_selection]

    return sky_wave, sky_data, sky_vars, sky_mask, rst_wave, rst_data


def fit_supersky(sky_wave, sky_data, sky_vars, sky_mask, sci_wave, sci_data):

    # plt.plot(sky_wave[0].ravel(), sky_data[0].ravel(), ".k")
    # plt.plot(sky_wave[2].ravel(), sky_data[2].ravel(), ".r")
    # plt.show()

    # remove outlying sky fibers
    # TODO: this rejection needs to be done on all-channels data
    mean_sky_data = np.nanmean(sky_data, axis=1)
    mean_sky_fiber = biweight_location(mean_sky_data, ignore_nan=True)
    std_sky_fiber = biweight_scale(mean_sky_data, ignore_nan=True)
    mask = np.abs(mean_sky_data - mean_sky_fiber) < 3 * std_sky_fiber
    nsky_fibers = mask.shape[0]
    sky_data = sky_data[mask]
    sky_wave = sky_wave[mask]
    sky_vars = sky_vars[mask]
    sky_mask = sky_mask[mask]

    # update mask
    sky_mask = sky_mask | (~np.isfinite(sky_data))
    sky_mask = sky_mask | (~np.isfinite(sky_vars))

    # build super-sky spectrum
    swave = sky_wave.flatten()
    ssky = sky_data.flatten()
    svars = sky_vars.flatten()
    smask = sky_mask.flatten()
    # build super-science spectrum
    ssci = sci_data.flatten()
    idx = np.unique(swave, return_index=True)[1]

    # sort arrays by wavelength
    swave = swave[idx]
    ssky = ssky[idx]
    svars = svars[idx]
    smask = smask[idx]
    ssci = ssci[idx]

    # calculate weights
    weights = 1 / svars

    # define interpolation functions
    # NOTE: store a super sampled version of the splines as an extension of the sky RSS
    f_data = interpolate.make_smoothing_spline(swave[~smask], ssky[~smask], w=weights[~smask], lam=1e-6)
    # NOTE: verify that the evaluated errors are not in variance at this stage
    f_error = interpolate.make_smoothing_spline(swave[~smask], svars[~smask] / nsky_fibers, w=weights[~smask], lam=1e-6)
    f_mask = interpolate.interp1d(swave, smask, kind="nearest", bounds_error=False, fill_value=0)

    return f_data, f_error, f_mask, swave, ssky, svars, smask


def get_telescope_shadowheight(header, telescope):
    """Calculates shadow height for different LVM telescopes

    Parameters
    ----------
    rss: lvmdrp.core.rss.RSS
        RSS object containing the header information for telescopes
    telescope: str
        telescope name (e.g. 'SKYE', 'SKYW', 'SCI', or 'SPEC')

    Returns
    -------
    shadow_height: float
        height of the shadow of the telescope
    """
    if telescope not in {"SKYE", "SKYW", "SCI", "SPEC"}:
        raise ValueError(f"invalid value for 'telescope' parameter: '{telescope}', valid values are 'SKYE', 'SKYW', 'SCI', or 'SPEC'")

    ra, dec = header[f"TE{telescope}RA"], header[f"TE{telescope}DE"]
    time = Time(header["OBSTIME"],format='isot', scale='utc')
    jd = time.jd

    SH_CALCULATOR.update_time(jd=jd)
    sk = SkyCoord(ra=ra, dec=dec, unit="deg")
    SH_CALCULATOR.set_coordinates(sk.ra, sk.dec)

    shadow_height = SH_CALCULATOR.get_heights()[0]

    # check if the shadow height is a nan value and replace with -999
    isnan = np.isnan(shadow_height)
    if isnan:
        shadow_height = -999

    return shadow_height
