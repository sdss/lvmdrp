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
from io import BytesIO

import numpy as np
from scipy import interpolate
import yaml
from astropy import units as u
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table, hstack
from astropy.time import Time
from astropy.coordinates import get_body, solar_system_ephemeris, AltAz, EarthLocation, SkyCoord 
from skycalc_cli.skycalc import AlmanacQuery, SkyModel
from skycalc_cli.skycalc_cli import fixObservatory
from skyfield.positionlib import ICRS
from skyfield.api import load
from skyfield.framelib import ecliptic_frame

from lvmdrp.external import shadow_height_lib as sh

from lvmdrp.core.constants import (
    ALMANAC_CONFIG_PATH,
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


def sky_pars_header(header):
    """Writes skymodel and useful sky info into header
    To run the ESO skymodel, several parameters are needed.  These are calculated
    and stored in the output (see ESO skymodel docu for more details).
    Additionally useful parameters needed for analyzing sky subtraction are provided

    Updated March 2025 AJ:
        removed SkyFields dependencies, 
        split sky and skymodel (re-named function), 
        included geo sh hght calc here,
        made header keywords more consistent with each other and drpall

    Parameters
    ----------
    header: fits header
        header with needed metadata info for given observation
    sky_pars: dict
        output dict with header keywords, values and comments
        None/NAN/Null values replaced with -999

    """

    # extract useful header information,
    sci_ra = header.get("SCIRA", header.get("POSCIRA", np.nan)) 
    sci_dec = header.get("SCIDEC", header.get("POSCIDE", np.nan))
    skye_ra = header.get("SKYERA", header.get("POSKYERA", np.nan)) 
    skye_dec = header.get("SKYEDEC", header.get("POSKYEDE", np.nan))
    skyw_ra = header.get("SKYWRA", header.get("POSKYWRA", np.nan)) 
    skyw_dec = header.get("SKYWDEC", header.get("POSKYWDE", np.nan))    
 
    obstime = Time(header["OBSTIME"])

   
    # define location of LCO using shadow heigh calculator library
    observatory_location = EarthLocation(lat=SH_CALCULATOR.observatory_topo.latitude.degrees*u.deg, 
                                     lon=SH_CALCULATOR.observatory_topo.longitude.degrees*u.deg, 
                                     height=SH_CALCULATOR.observatory_elevation.value*u.m)
   
    #use astropy Time class for the observing time
    obs_time = Time(obstime)

    #use astropy's builtin emphermis for the locations of the sun and moon
    with solar_system_ephemeris.set('builtin'):
        # Get the Moon's and Sun's coordinates at the specified time, and one hour later for moon phase 
        moon_coord = get_body('moon', obs_time,location=observatory_location)
        sun_coord = get_body('sun',obs_time,location=observatory_location)
        moon_coord_next = get_body('moon', obs_time+1*u.hour,location=observatory_location)
        sun_coord_next = get_body('sun',obs_time+1*u.hour,location=observatory_location)

    #find alt-az frame/coordinates for observation    
    altaz_frame = AltAz(obstime=obs_time, location=observatory_location)
    
    #use astropy SkyCoord class
    sci_coord = SkyCoord(sci_ra, sci_dec, unit='deg')
    skye_coord = SkyCoord(skye_ra, skye_dec, unit='deg')
    skyw_coord = SkyCoord(skyw_ra, skyw_dec, unit='deg')

    # observatory height ('sm_h' in km)
    sm_h = SH_CALCULATOR.observatory_elevation

    # RA and dec of moon (moonra, moondec) and SkyCoord position for moon
    moon_ra = moon_coord.ra.deg
    moon_dec = moon_coord.dec.deg
    moon_pos = SkyCoord(moon_ra*u.deg, moon_dec*u.deg)
    
    # altitude of objects above the horizon (alt, 0 -- 90)
    sci_alt = sci_coord.transform_to(altaz_frame).alt
    skye_alt = skye_coord.transform_to(altaz_frame).alt
    skyw_alt = skyw_coord.transform_to(altaz_frame).alt

    # altitude of moon ('moon_alt') and sun (sun_alt) [ -90 -- 90]
    moon_alt=moon_coord.transform_to(altaz_frame).alt
    sun_alt=sun_coord.transform_to(altaz_frame).alt

    # separation between moon and object ('rho', 0 -- 180)
    sci_rho = sci_coord.separation(moon_pos)
    skye_rho = skye_coord.separation(moon_pos)
    skyw_rho = skyw_coord.separation(moon_pos)

    # separation between SCI and Sky telescopes
    sci_skye = sci_coord.separation(skye_coord)
    sci_skyw = sci_coord.separation(skyw_coord)

    # Moon phase in deg (0:new, 90: first qt, 180: full, 270: 3rd qt)
    alpha = sun_coord.separation(moon_coord)
    alpha_next = sun_coord_next.separation(moon_coord_next)
    if alpha_next.value > alpha.value:
        moon_phase = alpha.to(u.deg)
    else:
        moon_phase = 360 * u.deg - alpha.to(u.deg)

    # Moon illumination fraction
    moon_phase_angle = np.arctan2(sun_coord.distance*np.sin(alpha), moon_coord.distance - sun_coord.distance*np.cos(alpha))
    moon_fli = (1 + np.cos(moon_phase_angle))/2.0

    # distance to moon ('moondist', 0.91 -- 1.08; 1: mean distance)
    moondist=moon_coord.distance.to(u.m)/MEAN_MOON_DIST

    # heliocentric ecliptic longitude of object ('lon_ecl', -180 -- 180)
    # heliocentric ecliptic latitude of object ('lat_ecl', -90 -- 90)
    sci_lon_ecl = sci_coord.geocentrictrueecliptic.lon - sun_coord.geocentrictrueecliptic.lon
    skye_lon_ecl = skye_coord.geocentrictrueecliptic.lon - sun_coord.geocentrictrueecliptic.lon
    skyw_lon_ecl = skyw_coord.geocentrictrueecliptic.lon - sun_coord.geocentrictrueecliptic.lon
    sci_lat_ecl = sci_coord.geocentrictrueecliptic.lat
    skye_lat_ecl = skye_coord.geocentrictrueecliptic.lat
    skyw_lat_ecl = skyw_coord.geocentrictrueecliptic.lat

    # TODO: add MSOLFLUX to headers. Pull data from here:
    # https://spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-5-en.php
    # now set to 130 sfu

    # bimonthly period ('season'; 1: Dec/Jan, ..., 6: Oct/Nov; 0 entire year)
    month = obs_time.to_datetime().month
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
    hour = obs_time.to_datetime().hour
    if season in [1, 2, 6]:
        if hour in [0, 1, 2, 3]:
            time = 1
        elif hour in [4, 5, 6]:
            time = 2
        elif hour in [7, 8, 9]:
            time = 3
        else:
            time = 0
    else:  
        if hour in [23, 0, 1, 2, 3]:
            time = 1
        elif hour in [4, 5, 6, 7]:
            time = 2
        elif hour in [8, 9, 10, 11]:
            time = 3
        else:
            time = 0

    sci_sh_hght = get_telescope_shadowheight(header, telescope="SCI")
    skye_sh_hght = get_telescope_shadowheight(header, telescope="SKYE")
    skyw_sh_hght = get_telescope_shadowheight(header, telescope="SKYW")

    # vacuum or air wavelengths ('vac_air', vac or air)
    # precipitable water vapour ('pwv' in mm; -1: bimonthly mean)
    # NOTE: the following is for MOD1 when we run the RT code
    # TODO: - ** radiative transfer code for molecular spectra ('rtcode', L or R)
    # TODO: - ** resolving power of molecular spectra in library ('resol')
    # TODO: - ** sky model components

    #header keywords SKY = parameters used for sky subtraction testing (incl geocoronal)
    #header keywords SKYMODEL = additional parameters needed to run the ESO sky model
    sky_pars = {
        "HIERARCH SKY SCI_ALT": (np.round(sci_alt.to(u.deg).value, 4), "altitude of object above horizon [deg]"),
        "HIERARCH SKY SKYE_ALT": (np.round(skye_alt.to(u.deg).value, 4), "altitude of object above horizon [deg]"),
        "HIERARCH SKY SKYW_ALT": (np.round(skyw_alt.to(u.deg).value, 4), "altitude of object above horizon [deg]"),
        "HIERARCH SKY SCI_SKYE_SEP": (np.round(sci_skye.to(u.deg).value, 4), "separation of SCI and SkyE [deg]"),
        "HIERARCH SKY SCI_SKYW_SEP": (np.round(sci_skyw.to(u.deg).value, 4), "separation of SCI and SkyW [deg]"),
        "HIERARCH SKY SCI_MOON_SEP": (np.round(sci_rho.to(u.deg).value, 4), "separation of Moon and object [deg]"),
        "HIERARCH SKY SKYE_MOON_SEP": (np.round(skye_rho.to(u.deg).value, 4), "separation of Moon and object [deg]"),
        "HIERARCH SKY SKYW_MOON_SEP": (np.round(skyw_rho.to(u.deg).value, 4), "separation of Moon and object [deg]"),
        "HIERARCH SKY MOON_ALT": (np.round(moon_alt.to(u.deg).value, 4), "altitude of Moon above horizon [deg]"),
        "HIERARCH SKY SUN_ALT": (np.round(sun_alt.to(u.deg).value, 4), "altitude of Sun above horizon [deg]"),
        "HIERARCH SKY MOON_RA": (np.round(moon_ra, 5), "RA of the Moon [deg]"),
        "HIERARCH SKY MOON_DEC": (np.round(moon_dec, 5), "DEC of the Moon [deg]"),
        "HIERARCH SKY MOON_PHASE": (np.round(moon_phase.value, 2), "Moon phase (0=N,90=1Q,180=F,270=3Q)[deg]"),
        "HIERARCH SKY MOON_FLI": (np.round(moon_fli.value, 4), "Moon fraction lunar illumination"),
        "HIERARCH SKY SCI_SH_HGHT": (np.round(sci_sh_hght, 5), "height of Earth's shadow [km]"),
        "HIERARCH SKY SKYE_SH_HGHT": (np.round(skye_sh_hght, 5), "height of Earth's shadow [km]"),
        "HIERARCH SKY SKYW_SH_HGHT": (np.round(skyw_sh_hght, 5), "height of Earth's shadow [km]"),
        "HIERARCH SKYMODEL SM_H": (sm_h.to(u.km).value, "observatory height [km]"),
        "HIERARCH SKYMODEL SM_HMIN": ((2.0 * u.km).value, "lower height limit [km]"),
        "HIERARCH SKYMODEL MOONDIST": (np.round(moondist.value, 4), "ratio of distance over mean dist to Moon"),
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
        "HIERARCH SKYMODEL SEASON": (season, "bimonthly period (1:Dec/Jan, 6:Oct/Nov; 0:year)"),
        "HIERARCH SKYMODEL TIME": (time, "period of night (x/3 night, x=1,2,3; 0:night)")
    }

    # checking for any NAN values and replacing with -999
    for key, value in sky_pars.items():
        if type(value[0]) is not str:
            if np.isnan(value[0]):
               sky_pars[key]=(-999, value[1])

    return sky_pars


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

    ra = header.get(f"{telescope}RA", header.get(f"PO{telescope}RA", np.nan))
    dec = header.get(f"{telescope}DEC", header.get(f"PO{telescope}DE", np.nan))

    time = Time(header["OBSTIME"],format='isot', scale='utc')
    jd = time.jd

    SH_CALCULATOR.update_time(jd=jd)
    sk = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    SH_CALCULATOR.set_coordinates(sk.ra.deg, sk.dec.deg)

    shadow_height = SH_CALCULATOR.get_heights(return_heights=True, unit="km")[0]

    # check if the shadow height is a nan value and replace with -999
    isnan = np.isnan(shadow_height)
    if isnan:
        shadow_height = -999

    return shadow_height
