# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: skyMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import itertools as it
import os
import re
import subprocess
from datetime import datetime
from multiprocessing import Pool, cpu_count
from copy import deepcopy as copy
from typing import Tuple

import numpy as np
import bottleneck as bn
import yaml
from astropy.io import fits
from astropy.time import Time
from scipy import optimize
from scipy import interpolate
from astropy.stats import biweight_location, biweight_scale

from lvmdrp.core.constants import (
    LVM_UNAM_URL,
    SKYCORR_CONFIG_PATH,
    SKYMODEL_CONFIG_PATH,
    SKYMODEL_INST_PATH,
)
from lvmdrp.core.plot import plt, create_subplots, save_fig
from lvmdrp.core.header import Header
from lvmdrp.core.passband import PassBand
from lvmdrp.core.rss import RSS
from lvmdrp.core.sky import (
    ang_distance,
    optimize_sky,
    run_skycorr,
    run_skymodel,
    skymodel_pars_from_header,
)
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp import log
from lvmdrp.utils.examples import fetch_example_data


description = "Provides methods for sky subtraction"

__all__ = [
    "configureSkyModel_drp",
    "createMasterSky_drp",
    "sepContinuumLine_drp",
    "evalESOSky_drp",
    "subtractGeocoronal_drp",
    "corrSkyLine_drp",
    "corrSkyContinuum_drp",
    "coaddContinuumLine_drp",
    "subtractSky_drp",
    "refineContinuum_drp",
    "subtractPCAResiduals_drp",
]


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


def configureSkyModel_drp(
    skymodel_config_path=SKYMODEL_CONFIG_PATH,
    skymodel_path=SKYMODEL_INST_PATH,
    method="run",
    run_library="0",
    run_multiscat="0",
    source="",
    parallel="auto",
):
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
        configuration files
    run_multiscat : boolean
        whether to run or not the ESO 'estmultiscat' routine for the multiple scattering
        corrections
    parallel : string or int
        whether to run the library generation in parallel or not. Valid values are 'auto' (default)
        and integers representing the number of threads to use

    Examples
    --------

    user:> drp sky configureSkyModel # to write the configuration files only
    user:> drp sky configureSkyModel method=run run_library=True run_multiscat=False

    """

    run_library = bool(int(run_library))
    run_multiscat = bool(int(run_multiscat))

    print(run_library, run_library, method)

    ori_path = os.path.abspath(os.curdir)

    log.info(
        f"writing configuration files using '{skymodel_config_path}' as source"
    )
    # read master configuration file
    skymodel_master_config = yaml.load(
        open(skymodel_config_path, "r"), Loader=yaml.Loader
    )

    # write default parameters for the ESO skymodel
    config_names = list(skymodel_master_config.keys())
    with open(
        os.path.join(skymodel_path, "sm-01_mod1", "config", config_names[0]), "w"
    ) as cf:
        for key, val in skymodel_master_config[config_names[0]].items():
            cf.write(f"{key} = {val}\n")
    with open(
        os.path.join(skymodel_path, "sm-01_mod2", "data", config_names[1]), "w"
    ) as cf:
        for par in skymodel_master_config[config_names[1]]:
            cf.write(f"{par}\n")
    with open(
        os.path.join(skymodel_path, "sm-01_mod2", "data", config_names[2]), "w"
    ) as cf:
        for key, val in skymodel_master_config[config_names[2]].items():
            cf.write(f"{key} = {val}\n")
    with open(
        os.path.join(skymodel_path, "sm-01_mod2", "config", config_names[3]), "w"
    ) as cf:
        for key, val in skymodel_master_config[config_names[3]].items():
            cf.write(f"{key} = {val}\n")
    with open(
        os.path.join(skymodel_path, "sm-01_mod2", "config", config_names[4]), "w"
    ) as cf:
        for key, val in skymodel_master_config[config_names[4]].items():
            cf.write(f"{key} = {val}\n")
    log.info("successfully written config files")

    # parse library path
    lib_path = os.path.abspath(
        os.path.join(
            skymodel_path,
            "sm-01_mod2",
            "data",
            skymodel_master_config["sm_filenames.dat"]["libpath"],
        )
    )

    if method == "run":
        # create sky library
        if run_library:
            cur_path = os.path.join(skymodel_path, "sm-01_mod1")
            # set hard-coded pwv (no scaling)
            pwv = -1
            # parse create_spec parameters
            spec_name = skymodel_master_config["libstruct.dat"][0]
            fact, pars = {}, {}
            ipar = {}
            for conv, values in zip(
                skymodel_master_config["libstruct.dat"][1::2],
                skymodel_master_config["libstruct.dat"][2::2],
            ):
                pos, name, exp = conv.split()
                pars[name] = values.split()
                fact[name] = 10 ** eval(exp)
                ipar[name] = pos
                spec_name = re.sub(f"{pos}+", f"{{{name}}}", spec_name)

            spec_nams, spec_pars = [], []
            filt_nams = list(
                filter(lambda name: name not in ["rtcode", "spectype"], pars.keys())
            )
            for i, values in enumerate(
                it.product(*tuple(v for k, v in pars.items() if k in filt_nams))
            ):
                # create output spectra names
                cur_values = {name: value for name, value in zip(filt_nams, values)}
                cur_values["rtcode"] = pars["rtcode"][0]
                cur_values_r, cur_values_t = cur_values.copy(), cur_values.copy()
                cur_values_r["spectype"], cur_values_t["spectype"] = "R", "T"
                spec_nams.append(
                    (
                        os.path.join(lib_path, spec_name.format(**cur_values_r)),
                        os.path.join(lib_path, spec_name.format(**cur_values_t)),
                    )
                )

                # parse parameters
                airmass, time, season, res = values
                spec_pars.append(
                    (
                        np.round(fact["airmass"] * int(airmass), 1),
                        fact["time"] * int(time),
                        fact["season"] * int(season),
                        fact["resol"] * int(res),
                    )
                )

            nlib = len(spec_pars)

            # run create_spec across all parameter grid
            os.chdir(cur_path)
            if parallel == "auto":
                cpus = cpu_count()
            else:
                cpus = int(parallel)
            if cpus > 1:
                log.info(
                    f"going to generate an airglow lines library of {nlib} spectra with {cpus} concurrent workers"
                )
                pool = Pool(cpus)
                result = []
                for i, (airmass, time, season, res) in enumerate(spec_pars):
                    # TODO: build output file names
                    if all(map(os.path.isfile, spec_nams[i])):
                        result.append(None)
                        continue
                    # add task to worker
                    result.append(
                        pool.apply_async(
                            subprocess.run,
                            args=(
                                f"{os.path.join('bin', 'create_spec')} {airmass} {time} {season} {cur_path} {res} {pwv}".split(),
                            ),
                            kwds={"capture_output": True},
                        )
                    )
                pool.close()
                pool.join()
            else:
                log.info(
                    f"going to generate an airglow lines library of {nlib} spectra"
                )

            for i, (airmass, time, season, res) in enumerate(spec_pars):
                if all(map(os.path.isfile, spec_nams[i])):
                    log.info(
                        f"skipping parameters {airmass = }, {time = }, {season = }, {res = }, {pwv = }, files {spec_nams[i]} already exist"
                    )
                    continue
                if cpus > 1:
                    log.info(
                        f"[{i+1:04d}/{nlib:04d}] retrieving airglow lines with parameters: {airmass = }, {time = }, {season = }, {res = }, {pwv = }"
                    )
                    out = result[i].get()
                else:
                    log.info(
                        f"[{i+1:04d}/{nlib:04d}] creating airglow lines with parameters: {airmass = }, {time = }, {season = }, {res = }, {pwv = }"
                    )
                    out = subprocess.run(
                        f"{os.path.join('bin', 'create_spec')} {airmass} {time} {season} {cur_path} {res} {pwv}".split(),
                        capture_output=True,
                    )
                if out.returncode == 0:
                    log.info("successfully finished airglow lines calculations")
                else:
                    log.error("failed while running airglow lines calculations")
                    log.error(out.stderr.decode("utf-8"))

            # copy airglow library to intended destination
            out = subprocess.run(
                f"mv output/*.fits {lib_path}/.".split(), capture_output=True
            )
            if out.returncode == 0:
                log.info("successfully copied airglow library")
            else:
                log.error("failed while copying airglow library")
                log.error(out.stderr.decode("utf-8"))

            # run prelinetrans
            log.info("calculating effective atmospheric transmission")
            os.chdir(os.path.join(skymodel_path, "sm-01_mod2"))
            out = subprocess.run(
                os.path.join("bin", "preplinetrans").split(), capture_output=True
            )
            if out.returncode == 0:
                log.info(
                    "successfully finished effective atmospheric transmission calculations"
                )
            else:
                log.error(
                    "failed while running effective atmospheric transmission calculations"
                )
                log.error(out.stderr.decode("utf-8"))

            if run_multiscat:
                out = subprocess.run(
                    os.path.join("bin", "estmultiscat").split(), capture_output=True
                )
                if out.returncode == 0:
                    log.info("successfully finished 'estmultiscat'")
                else:
                    log.error("failed while running 'estmultiscat'")
                    log.error(out.stderr.decode("utf-8"))
        # return to original path
        os.chdir(ori_path)
    elif method == "download":
        fetch_example_data(
            url=LVM_UNAM_URL,
            name="skymodel_lib",
            dest_path=lib_path,
            ext="zip",
        )
    else:
        raise ValueError(
            f"unknown method '{method}'. Valid values are: 'run' and 'download'"
        )


def createMasterSky_drp(
    in_rss, out_sky, clip_sigma="3.0", nsky="0", filter="", non_neg="1", plot="0"
):
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
    log.info(f"preparing to create master 'sky' from '{in_rss}'")

    clip_sigma = float(clip_sigma)
    nsky = int(nsky)
    non_neg = int(non_neg)
    plot = int(plot)
    filter = filter.split(",")

    rss = RSS()
    rss.loadFitsData(in_rss)

    log.info("calculating median value for each fiber")
    median = np.zeros(len(rss), dtype=np.float32)
    for i in range(len(rss)):
        spec = rss[i]

        if spec._mask is not None:
            good_pixels = np.logical_not(spec._mask)
            if np.sum(good_pixels) != 0:
                median[i] = np.median(spec._data[good_pixels])
            else:
                median[i] = 0
        else:
            median[i] = np.median(spec._data)
    # mask for fibers with valid sky spectra
    select_good = median != 0

    # sigma clipping around the median sky spectrum
    if clip_sigma > 0.0 and nsky == 0:
        log.info(
            f"calculating sigma clipping with sigma = {clip_sigma} within {select_good.sum()} fibers"
        )
        select = np.logical_and(
            np.logical_and(
                median
                < np.median(median[select_good])
                + clip_sigma * np.std(median[select_good]) / 2.0,
                median
                > np.median(median[select_good])
                - clip_sigma * np.std(median[select_good]) / 2.0,
            ),
            select_good,
        )
        sky_fib = np.sum(select)
    # select fibers that are below the maximum median spectrum within the top nsky fibers
    elif nsky > 0:
        idx = np.argsort(median[select_good])
        max_value = np.max(median[select_good][idx[:nsky]])
        if non_neg == 1:
            log.info(f"selecting non-negative (maximum) {nsky} fibers")
            select = (median <= max_value) & (median > 0.0)
        else:
            log.info(
                f"selecting (maximum) {nsky} fibers with median below {max_value = }"
            )
            select = median <= max_value
        sky_fib = np.sum(select)
    rss.setHdrValue("HIERARCH PIPE NSKY FIB", sky_fib, "Number of averaged sky fibers")

    # selection of sky fibers to build master sky
    subRSS = rss.subRSS(select)

    # calculates the sky magnitude within a given filter response function
    if filter[0] != "":
        log.info(
            f"calculating 'sky' magnitude in Vega system using filter in {filter[0]}"
        )
        passband = PassBand()
        passband.loadTxtFile(
            filter[0], wave_col=int(filter[1]), trans_col=int(filter[2])
        )
        (flux_rss, error_rss, min_rss, max_rss, std_rss) = passband.getFluxRSS(subRSS)
        mag_flux = np.zeros(len(flux_rss))
        for m in range(len(flux_rss)):
            if flux_rss[m] > 0.0:
                mag_flux[m] = passband.fluxToMag(flux_rss[m], system="Vega")

        mag_mean = np.mean(mag_flux[mag_flux > 0.0])
        mag_min = np.min(mag_flux[mag_flux > 0.0])
        mag_max = np.max(mag_flux[mag_flux > 0.0])
        mag_std = np.std(mag_flux[mag_flux > 0.0])
        rss.setHdrValue(
            "HIERARCH PIPE SKY MEAN",
            float("%.2f" % mag_mean),
            "Mean sky brightness of sky fibers",
        )
        rss.setHdrValue(
            "HIERARCH PIPE SKY MIN",
            float("%.2f" % mag_min),
            "Minimum sky brightness of sky fibers",
        )
        rss.setHdrValue(
            "HIERARCH PIPE SKY MAX",
            float("%.2f" % mag_max),
            "Maximum sky brightness of sky fibers",
        )
        rss.setHdrValue(
            "HIERARCH PIPE SKY RMS",
            float("%.2f" % mag_std),
            "RMS sky brightness of sky fibers",
        )
        log.info(f"{mag_mean = }, {mag_min = }, {mag_max = }, {mag_std = }")

    # create master sky spectrum by computing the average spectrum across selected fibers
    log.info(f"creating master (averaged) sky out of {subRSS._fibers}")
    skySpec = subRSS.create1DSpec()

    if plot == 1:
        plt.figure(figsize=(20, 5))
        plt.step(skySpec._wave, skySpec._data, color="k")
        plt.show()

    log.info(f"storing master sky in '{out_sky}'")
    skySpec.writeFitsData(out_sky)


def sepContinuumLine_drp(
    sky_ref,
    out_cont_line,
    method="skycorr",
    sky_sci="",
    skycorr_config=SKYCORR_CONFIG_PATH,
    is_science=False,
):
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
            raise ValueError(
                f"You need to provide a science spectrum to perform the continuum/line separation using skycorr."
            )
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
            TIME=(
                Time(sky_spec._header["MJD"], format="mjd").to_datetime()
                - datetime.fromisoformat("1970-01-01 00:00:00")
            ).days
            * 24
            * 3600,
            TELALT=sky_spec._header["ALT"],
            WLG_TO_MICRON=1e-4,
            FWHM=sky_spec._inst_fwhm.max() / np.diff(sky_spec._wave).min(),
        )

        wavelength = skycorr_fit["lambda"]
        sky_cont = Spectrum1D(
            wave=wavelength,
            data=skycorr_fit["mcflux"],
            error=None,
            mask=None,
            inst_fwhm=sky_spec._inst_fwhm,
        )
        sky_line = Spectrum1D(
            wave=wavelength,
            data=skycorr_fit["mlflux"],
            error=None,
            mask=None,
            inst_fwhm=sky_spec._inst_fwhm,
        )
        # TODO: implement skycorr method output

    # run model
    elif method == "model":
        prefix = "SM"
        # TODO: use the master sky parameters (datetime, observing conditions: lunation, moon distance, etc.) evaluate a sky model
        # TODO: use the resulting model continuum as physical representation of the target sky continuum
        # TODO: remove continuum contribution from original sky spectrum
        resample_step, resolving_power = np.diff(sky_spec._wave).min(), int(
            np.ceil((sky_spec._wave / np.diff(sky_spec._wave).min()).max())
        )
        # BUG: implement missing parameters in this call of run_skymodel
        skymodel_pars = skymodel_pars_from_header(sky_spec._header)
        inst_pars, model_pars, sky_model = run_skymodel(
            limlam=[sky_spec._wave.min() / 1e4, sky_spec._wave.max() / 1e4],
            dlam=resample_step / 1e4,
            **skymodel_pars,
        )
        pars_out = {}
        pars_out.update(inst_pars)
        pars_out.update(model_pars)
        # TODO: the predicted continuum would be the full radiative component - airglow line
        # TODO: scale the predicted continuum with the sky_ref
        sky_cont = Spectrum1D(
            wave=sky_model["lam"].value,
            data=sky_model["flux"].value - sky_model["flux_ael"].value,
            error=(sky_model["dflux2"] - sky_model["dflux1"]).value / 2,
            inst_fwhm=sky_model["lam"].value / resolving_power,
        )
        sky_cont._mask = np.isnan(sky_cont._data)
        # resample and match in spectral resolution sky model as needed
        if np.any(sky_cont._wave != sky_spec._wave):
            sky_cont = sky_cont.resampleSpec(ref_wave=sky_spec._wave, method="linear")
        if np.any(sky_cont._inst_fwhm != sky_spec._inst_fwhm):
            sky_cont = sky_cont.smoothGaussVariable(
                diff_fwhm=np.sqrt(sky_spec._inst_fwhm**2 - sky_cont._inst_fwhm**2)
            )

        # calculate the line component
        sky_line = sky_spec - sky_cont
    # run fit
    elif method == "fit":
        # TODO: build a sky model library with continuum and line separated (ESO skycalc)
        # TODO: use this library as templates to fit master skies
        # TODO: check if we can recover observing condition parameters from this fit
        raise NotImplementedError(
            "This method of continuum/line separation is not implemented yet."
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid mehods are: 'skycorr' (default), 'model' and 'fit'."
        )

    # TODO: explore the MaNGA way: sigma-clipping the lines and then smooth high-frequency features so that we get a continuum estimate
    # pack outputs in FITS file
    rss_cont_line = RSS.from_spectra1d((sky_cont, sky_line))

    header = sky_spec._header
    for key, val in pars_out.items():
        if isinstance(val, (list, tuple)):
            val = ",".join(map(str, val))
        elif isinstance(val, str) and (os.path.isfile(val) or os.path.isdir(val)):
            val = os.path.basename(val)
        header.append((f"HIERARCH {prefix} {key.upper()}", val))

    rss_cont_line.setHeader(header, origin=sky_ref)
    rss_cont_line.writeFitsData(out_cont_line)


def evalESOSky_drp(
    sky_ref,
    out_rss,
    resample_step="optimal",
    resample_method="linear",
    err_sim="500",
    replace_error="1e10",
    parallel="auto",
):
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
    err_sim = int(err_sim)
    replace_error = float(replace_error)

    # read sky spectrum
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky_ref, extension_hdr=0)

    eval_failed = False
    if resample_step != "optimal":
        try:
            resample_step = eval(resample_step)
        except ValueError:
            eval_failed = True
            log.error(
                f"resample_step should be either 'optimal' or a floating point. '{resample_step}' is none of those."
            )
            log.warning("falling back to resample_step='optimal'")
    if eval_failed or resample_step == "optimal":
        # determine sampling based on wavelength resolution
        # if not present LSF in reference spectrum, use the reference sampling step
        if sky_spec._inst_fwhm is not None:
            resample_step = np.min(sky_spec._inst_fwhm) / 3
        else:
            resample_step = np.min(np.diff(sky_spec._wave))

    new_wave = np.arange(
        sky_spec._wave.min(), sky_spec._wave.max() + resample_step, resample_step
    )

    # get skymodel parameters from header
    skymodel_pars = skymodel_pars_from_header(header=sky_spec._header)

    # TODO: move unit and data type conversions to within the run_skymodel routine
    inst_pars, model_pars, sky_model = run_skymodel(
        skymodel_path=SKYMODEL_INST_PATH,
        # instrument parameters
        limlam=[new_wave.min() / 1e4, new_wave.max() / 1e4],
        dlam=resample_step / 1e4,
        # sky model parameters
        **skymodel_pars,
    )
    pars_out = {}
    pars_out.update(inst_pars)
    pars_out.update(model_pars)

    # create RSS
    wav_comp = sky_model["lam"].value
    lsf_comp = sky_model["lam"].value / pars_out["resol"]
    sky_model.remove_column("lam")

    err_radi = (sky_model["dflux2"] - sky_model["dflux1"]) / 2
    err_tran = (sky_model["dtrans2"] - sky_model["dtrans2"]) / 2
    sky_model.remove_columns(["dflux1", "dflux2", "dtrans1", "dtrans2"])
    sed_comp = np.asarray(sky_model.as_array().tolist()).T
    msk_comp = np.isnan(sed_comp)

    nradi = len(list(filter(lambda c: c.startswith("flux"), sky_model.columns)))
    ntran = len(list(filter(lambda c: c.startswith("trans"), sky_model.columns)))
    err_comp = np.row_stack(
        (np.tile(err_radi, (nradi, 1)), np.tile(err_tran, (ntran, 1)))
    )
    # create initial RSS containing the sky model components
    spectra_list = [
        Spectrum1D(wave=wav_comp, data=sed, error=err, mask=msk, inst_fwhm=lsf_comp)
        for sed, err, msk in zip(sed_comp, err_comp, msk_comp)
    ]

    if parallel == "auto":
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    # resample RSS to reference wavelength sampling
    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        for i in range(len(spectra_list)):
            threads.append(
                pool.apply_async(
                    spectra_list[i].resampleSpec,
                    (sky_spec._wave, resample_method, err_sim, replace_error),
                )
            )

        for i in range(len(spectra_list)):
            spectra_list[i] = threads[i].get()
        pool.close()
        pool.join()
    else:
        for i in range(len(spectra_list)):
            spectra_list[i] = spectra_list[i].resampleSpec(
                sky_spec._wave, resample_method, err_sim, replace_error
            )

    # convolve RSS to reference LSF
    diff_fwhm = np.sqrt(sky_spec._inst_fwhm**2 - spectra_list[0]._inst_fwhm ** 2)
    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        for i in range(len(spectra_list)):
            threads.append(
                pool.apply_async(spectra_list[i].smoothGaussVariable, (diff_fwhm,))
            )

        for i in range(len(spectra_list)):
            spectra_list[i] = threads[i].get()
        pool.close()
        pool.join()
    else:
        for i in range(len(spectra_list)):
            spectra_list[i] = spectra_list[i].smoothGaussVariable(diff_fwhm)

    # build RSS
    rss = RSS.from_spectra1d(spectra_list=spectra_list)
    header = sky_spec._header
    for key, val in pars_out.items():
        if isinstance(val, (list, tuple)):
            val = ",".join(map(str, val))
        elif isinstance(val, str) and (os.path.isfile(val) or os.path.isdir(val)):
            val = os.path.basename(val)
        header.append((f"HIERARCH SM {key.upper()}", val))

    rss.setHeader(header, origin=sky_ref)
    # dump RSS file containing the
    rss.writeFitsData(filename=out_rss)


def subtractGeocoronal_drp():
    pass


def corrSkyLine_drp(
    sky1_line_in,
    sky2_line_in,
    sci_line_in,
    line_corr_out,
    method="distance",
    sky_models_in="",
    sci_model_in="",
    skycorr_config=SKYCORR_CONFIG_PATH,
):
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
        raise ValueError(
            f"Unknown method '{method}'. Valid mehods are: 'distance' (default), 'model' and 'interpolate'."
        )

    # TODO: make sure all these spectra are in the same wavelength sampling
    wl_master_sky = sci_line._wave

    # compute a weighted average using as weights the inverse distance to science
    sky_line = w_1 * sky1_line + w_2 * sky2_line

    # run skycorr on averaged line spectrum
    pars_out, line_fit = run_skycorr(
        skycorr_config=skycorr_config,
        wl=wl_master_sky,
        sci_spec=sci_line,
        sky_spec=sky_line,
    )

    # create RSS
    wav_fit = line_fit["lambda"].value
    lsf_fit = line_fit["lambda"].value / pars_out["wres"].value
    sed_fit = np.asarray(line_fit.as_array().tolist())[:, 1].T
    hdr_fit = fits.Header(pars_out)
    rss = RSS(data=sed_fit, wave=wav_fit, inst_fwhm=lsf_fit, header=hdr_fit)

    # dump RSS file containing the model sky line spectrum
    rss.writeFitsData(filename=line_corr_out)


def corrSkyContinuum_drp(
    sky1_cont_in,
    sky2_cont_in,
    sci_cont_in,
    cont_corr_out,
    method="model",
    sky_models_in="",
    sci_model_in="",
    model_fiber=1,
):
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
        raise ValueError(
            f"Unknown method '{method}'. Valid mehods are: 'distance', 'model' (default) and 'interpolate'."
        )

    # TODO: propagate error in continuum correction
    # TODO: propagate mask
    # TODO: propagate LSF
    cont_fit = 0.5 * (w_1 * sky1_cont + w_2 * sky2_cont)
    cont_fit.writeFitsData(cont_corr_out)


def coaddContinuumLine_drp(
    sky_cont_corr_in, sky_line_corr_in, sky_corr_out, line_fiber=9
):
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


def subtractSky_drp(
    in_rss,
    out_rss,
    sky_ref,
    out_sky,
    factor="1",
    scale_region="",
    scale_ind=False,
    parallel="auto",
):
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
    if scale_region != "":
        region = scale_region.split(",")
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
        header=sky_head,
    )

    if np.all(rss._wave == sky_spec._wave) and scale_region != "":
        factors = np.zeros(len(rss), dtype=np.float32)
        for i in range(len(rss)):
            try:
                optimum = optimize.fmin(
                    optimize_sky,
                    [1.0],
                    args=(rss[i], sky_spec, wave_region[0], wave_region[1]),
                    disp=0,
                )
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
    elif np.all(rss._wave == sky_spec._wave) and scale_region == "":
        for i in range(len(rss)):
            sky_rss[i] = sky_spec * factor
            rss[i] = rss[i] - sky_rss[i]
        scale_factor = factor

    if len(rss._wave) == 2:
        if parallel == "auto":
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

    if scale_region != "":
        rss.setHdrValue(
            "HIERARCH PIPE SKY SCALE",
            float("%.3f" % scale_factor),
            "sky spectrum scale factor",
        )
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


def interpolate_sky(in_rss: str, out_sky: str, out_rss: str = None, which: str = "both",
                    subtract: bool = False, display_plots: bool = False) -> RSS:
    """Interpolate sky fibers in a given RSS file

    Parameters
    ----------
    in_rss : str
        input RSS file
    out_sky : str
        output sky RSS file
    out_rss : str, optional
        output sky-subtracted RSS file, by default None
    which : str, optional
        for which sky telescope fibers are going to be interpolated, by default "both"
    subtract : bool, optional
        whether to subtract or not interpolated sky from original data, by default False
    display_plots : bool, optional
        whether to display plots or not, by default False

    Returns
    -------
    RSS
        interpolated sky RSS object

    Raises
    ------
    ValueError
        if subtract is True and out_rss is None or if which is not one of {"both", "east", "e", "skye", "west", "w", "skyw"}
    """

    if subtract and out_rss is None:
        raise ValueError(f"need to provide an output file to write sky-subtracted data")
    
    # load input RSS
    log.info(f"loading input RSS file '{os.path.basename(in_rss)}'")
    rss = RSS()
    rss.loadFitsData(in_rss)

    # extract fibermap for current spectrograph
    fibermap = rss._slitmap[rss._slitmap["spectrographid"] == int(rss._header["CCD"][1])]
    
    # select sky fibers
    which = which.lower()
    if which == "both":
        sky_selection = fibermap["targettype"] == "SKY"
    elif which in {"east", "e", "skye"}:
        sky_selection = fibermap["telescope"] == "SkyE"
    elif which in {"west", "w", "skyw"}:
        sky_selection = fibermap["telescope"] == "SkyW"
    else:
        raise ValueError(f"invalid value for 'which' parameter: '{which}'")

    # define wavelength, flux and variances
    log.info(f"interpolating sky fibers for '{which}' sky telescope(s)")
    sky_wave = rss._wave[sky_selection]
    sky_data = rss._data[sky_selection]
    sky_vars = rss._error[sky_selection] ** 2
    sky_mask = rss._mask[sky_selection]
    sci_wave = rss._wave[~sky_selection]
    sci_data = rss._data[~sky_selection]

    # plt.plot(sky_wave[0].ravel(), sky_data[0].ravel(), ".k")
    # plt.plot(sky_wave[2].ravel(), sky_data[2].ravel(), ".r")
    # plt.show()

    # remove outlying sky fibers
    # TODO: this rejection needs to be done on all-channels data
    mean_sky_data = np.nanmean(sky_data, axis=1)
    mean_sky_fiber = biweight_location(mean_sky_data)
    std_sky_fiber = biweight_scale(mean_sky_data)
    mask = np.abs(mean_sky_data - mean_sky_fiber) < 3 * std_sky_fiber
    sky_data = sky_data[mask]
    sky_wave = sky_wave[mask]
    sky_vars = sky_vars[mask]
    sky_mask = sky_mask[mask]

    # plt.plot(sky_wave.ravel(), sky_data.ravel(), ".r")
    # plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(20,5), sharex=True)
    axs = axs.flatten()
    axs[0].plot(sky_wave[0], sky_data[0], "k", lw=1, label="sky")
    axs[0].plot(sci_wave[10], sci_data[10], "0.5", lw=1, label="sci")
    axs[0].legend(loc=2)

    axs[0].plot(sky_wave[0], sky_data[0], "r", lw=1, label="flatfielded sky")
    axs[0].plot(sci_wave[10], sci_data[10], "b", lw=1, label="flatfielded sci")
    axs[0].legend(loc=2)
    # divide by the wavelength sampling step at each pixel
    dlambda = np.diff(sky_wave, axis=1)
    dlambda = np.column_stack((dlambda, dlambda[:, -1]))
    sky_data = sky_data / dlambda
    sky_vars = sky_vars / dlambda

    axs[1].plot(sky_wave[0], sky_data[0], "b", lw=1, label="densities")
    axs[1].legend(loc=2)
    save_fig(
        fig,
        product_path=out_sky,
        to_display=display_plots,
        figure_path="qa",
        label="diag",
    )

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
    
    # model sky using skycorr
    # pars_out, skycorr_fit = run_skycorr(
    #     sci_spec=Spectrum1D(wave=swave, data=ssci, error=np.sqrt(svars), mask=smask),
    #     sky_spec=Spectrum1D(wave=swave, data=ssky, error=np.sqrt(svars), mask=smask),
    #     spec_label=os.path.basename(out_sky).replace(".fits", ""),
    #     specs_dir=os.path.abspath("."),
    #     out_dir=os.path.abspath("."),
    #     MJD=rss._header["MJD"],
    #     TIME=(
    #         Time(rss._header["MJD"], format="mjd").to_datetime()
    #         - datetime.fromisoformat("1970-01-01 00:00:00")
    #     ).days
    #     * 24
    #     * 3600,
    #     TELALT=(np.pi/2 - np.arccos(1/rss._header["SCIAIRM"])) * 180/np.pi,
    #     WLG_TO_MICRON=1e-4,
    #     FWHM=1.5,
    # )
    # ssky = skycorr_fit["mflux"]
    
    # calculate weights
    weights = 1 / svars

    # define interpolation functions
    # NOTE: store a super sampled version of the splines as an extension of the sky RSS
    f_data = interpolate.make_smoothing_spline(swave[~smask], ssky[~smask], w=weights[~smask], lam=1e-6)
    f_error = interpolate.make_smoothing_spline(swave[~smask], svars[~smask], w=weights[~smask], lam=1e-6)
    f_mask = interpolate.interp1d(swave, smask, kind="nearest", bounds_error=False, fill_value=0)

    fig, axs = create_subplots(to_display=display_plots, figsize=(20,10), nrows=2, ncols=1, sharex=True)
    axs[0].scatter(swave, ssky, s=1, color="tab:blue", label="super sky")
    axs[0].plot(swave[~smask], f_data(swave[~smask]).astype("float32"), lw=1, color="k", label="spline")

    # plot residuals
    residuals = (f_data(sky_wave).astype("float32") - sky_data)
    residuals = residuals.flatten()
    axs[1].scatter(sky_wave.flatten(), residuals)
    axs[1].axhline(ls="--", lw=1, color="k")
    axs[1].set_label("lambda (Angstrom)")
    fig.supylabel("counts (e/s)")

    save_fig(
        fig,
        product_path=out_sky,
        to_display=display_plots,
        figure_path="qa",
        label="super_sky",
    )

    # interpolated sky
    dlambda = np.diff(rss._wave, axis=1)
    dlambda = np.column_stack((dlambda, dlambda[:, -1]))
    new_sky = f_data(rss._wave).astype("float32") * dlambda
    new_error = np.sqrt(f_error(rss._wave).astype("float32")) * dlambda
    new_mask = f_mask(rss._wave).astype(bool)
    # update mask with new bad pixels
    # new_mask |= rss._mask
    new_mask = (new_sky<0) | np.isnan(new_sky)
    new_mask |= (new_error<0) | np.isnan(new_error)
    
    fig, axs = plt.subplots(1, 1, figsize=(20,5), sharex=True)
    axs.plot(rss._wave[0], new_sky[0], "k", lw=1, label="sky")
    # axs.plot(sky_wave[10], sci_data[10], "0.5", lw=1, label="sci")
    save_fig(
        fig,
        product_path=out_sky,
        to_display=display_plots,
        figure_path="qa",
        label="sky_comp",
    )

    # define sky RSS
    sky_rss = copy(rss)
    sky_rss.setData(data=new_sky, error=new_error, mask=new_mask)
    sky_rss._header["IMAGETYP"] = "sky"

    # extract standard star metadata if exists
    std_acq = np.asarray(list(rss._header["STD*ACQ"].values()))
    if std_acq.size == 0:
        log.warning("no standard star metadata found, skipping sky reescaling")
    else:
        # filter by acquired
        std_ids = np.asarray(list(rss._header["STD*FIB"].values()))[std_acq]
        std_exp = np.asarray(list(rss._header["STD*EXP"].values()))[std_acq]
        # select only standard star in current exposure
        std_idx = np.where(np.isin(fibermap["orig_ifulabel"], std_ids))
        log.info(f"calculating correction factors for standard star: {fibermap[std_idx]['orig_ifulabel'].value}")
        # calculate scaling factors for standard star
        std_fac = ((stdid, stdexp / rss._header["EXPTIME"]) for stdid, stdexp in zip(std_ids, std_exp) if stdid in fibermap["orig_ifulabel"])
        std_fac = {stdid: np.round(factor,4) for stdid, factor in sorted(std_fac, key=lambda item: int(item[0].split("-")[1]))}
        log.info(f"correction factors for standard star: {std_fac}")
        # apply factors to standard star sky
        sky_rss._data[std_idx] *= np.asarray(list(std_fac.values()))[:, None]
        sky_rss._error[std_idx] *= np.asarray(list(std_fac.values()))[:, None]
    
    # write output sky RSS
    log.info(f"writing output sky RSS file '{os.path.basename(out_sky)}'")
    sky_rss.writeFitsData(out_sky)

    if subtract:
        # subtract interpolated sky from original data
        log.info("subtracting interpolated sky from original data")
        new_data = rss._data - new_sky
        new_error = np.sqrt(rss._error ** 2 + new_error ** 2)
        new_mask = rss._mask | new_mask
        # write output sky-subtracted RSS
        log.info(f"writing output sky-subtracted RSS file '{os.path.basename(out_rss)}'")

        rss_sub = copy(rss)
        rss_sub.setData(data=new_data, error=new_error, mask=new_mask)
        rss_sub.writeFitsData(out_rss)
    
    return f_data, f_error, sky_rss, swave, ssky, svars, smask


def quick_sky_subtraction(in_rss: str, out_rss, in_skye: str, in_skyw: str, sky_weights: Tuple[float, float] = None, skip_subtraction: bool = False) -> RSS:
    """Quick sky subtraction using the sky fibers from both telescopes

    Parameters
    ----------
    in_rss : str
        input RSS file
    out_rss : str
        output sky-subtracted RSS file
    in_skye : str
        input SkyE RSS file
    in_skyw : str
        input SkyW RSS file
    sky_weights : Tuple[float, float]
        weights for each telescope when master_sky = 'combine', by default None
    skip_subtraction : bool, optional
        whether to skip sky subtraction or not, by default False

    Returns
    -------
    RSS
        sky-subtracted RSS object

    """
    # load input RSS
    log.info(f"loading input RSS file '{os.path.basename(in_rss)}'")
    rss = RSS()
    rss.loadFitsData(in_rss)

    # load sky RSS
    log.info(f"loading input SkyE RSS file '{os.path.basename(in_skye)}'")
    sky_e = RSS()
    sky_e.loadFitsData(in_skye)
    log.info(f"loading input SkyW RSS file '{os.path.basename(in_skyw)}'")
    sky_w = RSS()
    sky_w.loadFitsData(in_skyw)

    # linearly interpolate in sky coordinates
    log.info("interpolating sky fibers for both telescopes")
    ra_e = rss._header.get("TESKYERA", rss._header.get("SKYERA"))
    dec_e = rss._header.get("TESKYEDE", rss._header.get("SKYEDEC"))
    ra_w = rss._header.get("TESKYWRA", rss._header.get("SKYWRA"))
    dec_w = rss._header.get("TESKYWDE", rss._header.get("SKYWDEC"))
    ra_s = rss._header.get("TESCIRA", rss._header.get("SCIRA"))
    dec_s = rss._header.get("TESCIDE", rss._header.get("SCIDEC"))

    log.info("interpolating sky telescopes pointings "
        f"(SKYERA, SKYEDEC: {ra_e:.2f}, {dec_e:.2f}; SKYWRA, SKYWDEC: {ra_w:.2f}, {dec_w:.2f}) "
        f"in science telescope pointing (SCIRA, SCIDEC: {ra_s:.2f}, {dec_s:.2f})")

    if sky_weights is None:
        ad = ang_distance(ra_e, dec_e, ra_s, dec_s)
        w_e = 1 / (ad if ad>0 else 1)
        ad = ang_distance(ra_w, dec_w, ra_s, dec_s)
        w_w = 1 / (ad if ad>0 else 1)
        w_norm = w_e + w_w
        w_e, w_w = w_e / w_norm, w_w / w_norm
        log.info(f"calculated weights SkyE: {w_e:.3f}, SkyW: {w_w:.3f}")
    elif len(sky_weights) == 2:
        w_e, w_w = sky_weights
        w_norm = w_e + w_w
        if w_norm != 1:
            w_e, w_w = w_e / w_norm, w_w / w_norm
        log.info(f"assuming user-provided weights SkyE: {w_e:.3f}, SkyW: {w_w:.3f}")
    else:
        raise ValueError(f"invalid value for 'sky_weights' parameter: '{sky_weights}'")

    # define master sky
    sky = sky_e * w_e + sky_w * w_w
    
    # subtract master sky from data
    if skip_subtraction:
        log.info(f"skipping sky subtraction, saving master sky in '{os.path.basename(out_rss)}'")
        new_data = rss._data
        new_error = rss._error
        new_mask = rss._mask
    else:
        log.info("subtracting interpolated sky from original data")
        new_data = rss._data - sky._data
        new_error = np.sqrt(rss._error ** 2 + sky._error ** 2)
        new_mask = rss._mask

    # write output sky-subtracted RSS
    log.info(f"writing output RSS file '{os.path.basename(out_rss)}'")
    rss.setHdrValue("SKYSUB", not skip_subtraction, "sky subtracted?")
    rss.setHdrValue("SKYEW", w_e, "SkyE weight")
    rss.setHdrValue("SKYWW", w_w, "SkyW weight")
    rss.setData(data=new_data, error=new_error, mask=new_mask)
    rss.set_sky(rss_sky=sky)
    rss.writeFitsData(out_rss)

    return rss


def quick_sky_refinement(in_cframe, band=np.array((7238,7242,7074,7084,7194,7265)), skip_subtraction=False):
    """Quick sky refinement using the model in the final CFrame
    
    Parameters
    ----------
    in_cframe : str
        input CFrame file
    band : np.array, optional
        wavelength range to use for sky refinement, by default np.array((7238,7242,7074,7084,7194,7265))
    """
    
    cframe = fits.open(in_cframe)
    wave = cframe["WAVE"].data
    flux = cframe["FLUX"].data
    error = cframe["ERROR"].data
    sky = cframe["SKY"].data

    crval = wave[0]
    cdelt = wave[1] - wave[0]
    i_band = ((band - crval) / cdelt).astype(int)

    map_b = bn.nanmean(flux[:, i_band[0]:i_band[1]], axis=1)
    map_0 = bn.nanmean(flux[:, i_band[2]:i_band[3]], axis=1)
    map_1 = bn.nanmean(flux[:, i_band[4]:i_band[5]], axis=1)
    map_c = map_b - 0.5 * (map_0 + map_1)

    smap_b = bn.nanmean(sky[:, i_band[0]:i_band[1]], axis=1)
    smap_0 = bn.nanmean(sky[:, i_band[2]:i_band[3]], axis=1)
    smap_1 = bn.nanmean(sky[:, i_band[4]:i_band[5]], axis=1)
    smap_c = smap_b - 0.5 * (smap_0 + smap_1)

    scale = map_c / smap_c
    sky_c = np.nan_to_num(sky * scale[:, None])
    if not skip_subtraction:
        data_c = np.nan_to_num(flux - sky_c)
        error_c = np.nan_to_num(error - sky_c)
    else:
        data_c = flux
        error_c = error

    cframe["FLUX"].data = data_c
    cframe["ERROR"].data = error_c
    cframe["SKY"].data = sky_c
    cframe.writeto(in_cframe, overwrite=True)
