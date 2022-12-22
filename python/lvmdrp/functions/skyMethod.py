# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: skyMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from multiprocessing import Pool
from scipy import optimize
from astropy.io import fits

from lvmdrp.core.constants import SKYCORR_CONFIG_PATH, SKYCALC_CONFIG_PATH, ALMANAC_CONFIG_PATH
from lvmdrp.core.sky import run_skycorr, run_skymodel, optimize_sky, ang_distance
from lvmdrp.core.passband import PassBand
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.header import Header
from lvmdrp.core.rss import RSS


description = "Provides methods for sky subtraction"


def createMasterSky_drp(rss_in, sky_out, clip_sigma='3.0', nsky='0', filter='', non_neg='1', plot='0'):
	"""
        Creates an average (sky) spectrum from the RSS, which stored either as a FITS or an ASCII file.
        Spectra may be rejected from the median computation. Bad pixel in the RSS are not included
        in the median computation.

        TODO: implement fiber rejection for science pointings which should make other considerations

        Parameters
        --------------
        rss_in : string
            Input RSS FITS file with a pixel table for the spectral resolution
        sky_out : string
            Output Sky spectrum. Either in FITS format (if *.fits) or in ASCII format (if *.txt)
        clip_sigma : string of float, optiional with default: '3.0'
            Sigma value used to reject outlier sky spectra identified in the collapsed median value along the dispersion axis
            Only used if the nsky value is set to 0 and clip_sigma>0.
        nsky : string of integer (>0), optional with default: '0'
            Selects the number of brightest sky spectra to be used for creating the median sky spec.
        filter : string of tuple, optional with default: ''
            Path to file containing the response function of a filter, and the wavelength and transmission columns
        plot : string of integer (0 or 1)
            If set to 1, the sky spectrum will be display on screen.

        Examples
        ----------------
        user:> drp sky constructSkySpec RSS_IN.fits SKY_OUT.fits 3.0
        user:> drp sky constructSkySpec RSS_IN.fits SKY_OUT.txt
	"""
	clip_sigma=float(clip_sigma)
	nsky = int(nsky)
	non_neg = int(non_neg)
	plot = int(plot)
	filter=filter.split(',')
	rss = RSS()
	rss.loadFitsData(rss_in)
	median = np.zeros(len(rss), dtype=np.float32)
	for i in range(len(rss)):
		spec = rss[i]
		
		if spec._mask is not None:
			if np.sum(np.logical_not(spec._mask))!=0:
				median[i] = np.median(spec._data[np.logical_not(spec._mask)])
			else:
				median[i]=0
		else:
			median[i] = np.median(spec._data)
	# mask for fibers with valid sky spectra
	select_good = median!=0

	# sigma clipping around the median sky spectrum
	if clip_sigma>0.0 and nsky==0:
		select = np.logical_and(np.logical_and(median<np.median(median[select_good])+clip_sigma*np.std(median[select_good])/2.0, median>np.median(median[select_good])-clip_sigma*np.std(median[select_good])/2.0), select_good)
		sky_fib = np.sum(select)
	# select fibers that are below the maximum median spectrum within the top nsky fibers
	elif nsky>0:
		idx=np.argsort(median[select_good])
		max_value = np.max(median[select_good][idx[:nsky]])
		if non_neg==1:
			select = (median<=max_value) & (median>0.0)
		else:
			select = (median<=max_value)
		sky_fib = np.sum(select)
	rss.setHdrValue('hierarch PIPE NSKY FIB', sky_fib, 'Number of averaged sky fibers')
	
	# selection of sky fibers to build master sky
	subRSS = rss.subRSS(select)

	# calculates the sky magnitude within a given filter response function
	if filter[0] != '':
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
		rss.setHdrValue('hierarch PIPE SKY MEAN', float('%.2f'%mag_mean), 'Mean sky brightness of sky fibers')
		rss.setHdrValue('hierarch PIPE SKY MIN', float('%.2f'%mag_min), 'Minium sky brightness of sky fibers')
		rss.setHdrValue('hierarch PIPE SKY MAX', float('%.2f'%mag_max), 'Maximum sky brightness of sky fibers')
		rss.setHdrValue('hierarch PIPE SKY RMS', float('%.2f'%mag_std), 'RMS sky brightness of sky fibers')

	rss.writeFitsHeader(rss_in)
	# create master sky spectrum by computing the average spectrum across selected fibers
	skySpec = subRSS.create1DSpec()

	if plot==1:
		plt.plot(skySpec._wave, skySpec._data, 'ok')
		plt.show()
	if '.fits' in sky_out:
		skySpec.writeFitsData(sky_out)
	if '.txt' in sky_out:
		skySpec.writeTxtData(sky_out)


def sepContinuumLine_drp(sky_ref, cont_out, line_out, method="skycorr", sky_sci="", skycorr_config=SKYCORR_CONFIG_PATH):
    """Separates the continuum from the sky line contribution using the specified method
    
        Run the chosen continuum/line separation algorithm on master sky 1 and 2 and master science and produce
        line-only (sky1_line, sky2_line, sci_line) and continuum-only (sky1_cont, sky2_cont, sci_cont) spectra for each
    """

    # read sky spectrum
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky_ref)
    
    # run skycorr
    if method == "skycorr":
        if sky_sci != "":
            sci_spec = Spectrum1D()
            sci_spec.loadFitsData(sky_sci)
        else:
            raise ValueError(f"You need to provide a science spectrum to perform the continuum/line separation using skycorr.")
        # TODO: match wavelength sampling and resolution if needed
        if np.any(sky_spec._wave != sci_spec._wave):
            sky_spec = sky_spec.binSpec(new_wave=sci_spec._wave)
        pars_out, par_file, skycorr_fit = run_skycorr(skycorr_config=skycorr_config, sci_spec=sci_spec, sky_spec=sky_spec)

        wavelength = skycorr_fit["lambda"]
        # TODO: include propagated errors from the continuum fitting
        # TODO: include propagated pixel masks
        # TODO: include LSF
        sky_cont = Spectrum1D(wave=wavelength, data=skycorr_fit["mcflux"])
        sky_line = Spectrum1D(wave=wavelength, data=skycorr_fit["mlflux"])
    # run physical
    elif method == "physical":
        # NOTE: build a sky model library with continuum and line separated (ESO skycalc)
        # NOTE: use this library as templates to fit master skies
        # NOTE: check if we can recover observing condition parameters from this fit
        raise NotImplementedError("This method of continuum/line separation is not implemented yet.")
    else:
        raise ValueError(f"Unknown method '{method}'. Valid mehods are: skycorr (default) and physical.")
    
    # pack outputs in FITS file
    sky_cont.writeFitsData(cont_out)
    sky_line.writeFitsData(line_out)


def evalESOSky_drp(sky_ref, rss_out, skymodel_config=SKYCALC_CONFIG_PATH, almanac_config=ALMANAC_CONFIG_PATH, resample_step="optimal", resample_method="linear", err_sim='500', replace_error='1e10', parallel="auto"):
    """
    
    run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc) to evaluate sky spectrum at each
    telescope pointing (model_sky1, model_sky2, model_skysci)

    """

    # read sky spectrum
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky_ref)

    eval_failed = False
    if resample_step != "optimal":
        try:
            resample_step = eval(resample_step)
        except ValueError:
            # TODO: add logger info to screen
            pass
    if eval_failed or resample_step == "optimal":
        # NOTE: determine sampling based on wavelength resolution
        # NOTE: if not present LSF in reference spectrum, use the reference sampling step
        if sky_spec._inst_fwhm is not None:
            resample_step = np.min(sky_spec._inst_fwhm) / 3
        else:
            resample_step = np.min(sky_spec._wave)
    
    new_wave = np.arange(sky_spec._wave.min(), sky_spec._wave.max() + resample_step, resample_step)

    pars_out, par_file, sky_model = run_skymodel(
        skycalc_config=skymodel_config,
        almanac_config=almanac_config,
        wmin=new_wave.min(),
        wmax=new_wave.max(),
        wdelta=resample_step,
        wres=(new_wave/resample_step).max()
    )
    
    # create RSS
    wav_comp = sky_model["lam"].value
    lsf_comp = sky_model["lam"].value / pars_out["wres"].value
    sed_comp = sky_model.as_array()[:,1].T
    hdr_comp = fits.Header(pars_out)
    hdr_comp["ASMCONF"] = (par_file, "ESO Advanced Sky Model config file")
    rss = RSS(data=sed_comp, wave=wav_comp, inst_fwhm=lsf_comp, header=hdr_comp)
    
    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    # resample RSS to reference wavelength sampling
    spectra_list = []
    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        for i in range(len(rss)):
            threads.append(pool.apply_async(rss[i].resampleSpec, (new_wave, resample_method, err_sim, replace_error)))

        for i in range(len(rss)):
            spectra_list.append(threads[i].get())
        pool.close()
        pool.join()
    else:
        for i in range(len(rss)):
            spectra_list.append(rss[i].resampleSpec(new_wave))
    
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
    # dump RSS file containing the
    rss.writeFitsData(filename=rss_out)


def subtractGeocoronal_drp():
    pass


def corrSkyLine_drp(sky1_line_in, sky2_line_in, sci_line_in, line_corr_out, skycorr_config=SKYCORR_CONFIG_PATH):
    """
    
    average sky1_line and sky2_line into 'sky_line', and run skycorr on 'sky_line' and 'sci_line' to produce 'sky_line_corr'
    """

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
    ra_1, dec_1 = sky1_head["RA"], sky1_head["DEC"]
    # sky2 position
    ra_2, dec_2 = sky2_head["RA"], sky2_head["DEC"]
    # sci position
    ra_s, dec_s = sci_head["RA"], sci_head["DEC"]

    w_1 = ang_distance(ra_1, dec_1, ra_s, dec_s)
    w_2 = ang_distance(ra_2, dec_2, ra_s, dec_s)
    w_norm = w_1 + w_2
    w_1, w_2 = w_1 / w_norm, w_2 / w_norm
    
    # TODO: make sure all these spectra are in the same wavelength sampling
    wl_master_sky = sci_line._wave

    # compute a weighted average using as weights the inverse distance distance to science
    sky_line = w_1 * sky1_line + w_2 * sky2_line
    
    # run skycorr on averaged line spectrum
    pars_out, par_file, line_fit = run_skycorr(skycorr_config=skycorr_config, wl=wl_master_sky, sci_spec=sci_line, sky_spec=sky_line)

    # create RSS
    wav_fit = line_fit["lambda"].value
    lsf_fit = line_fit["lambda"].value / pars_out["wres"].value
    sed_fit = line_fit.as_array()[:,1].T
    hdr_fit = fits.Header(pars_out)
    hdr_fit["ESCCONF"] = (par_file, "ESO Skycorr config file")
    rss = RSS(data=sed_fit, wave=wav_fit, inst_fwhm=lsf_fit, header=hdr_fit)

    # dump RSS file containing the model sky line spectrum
    rss.writeFitsData(filename=line_corr_out)


def corrSkyContinuum_drp(sky1_cont_in, sky2_cont_in, sky1_model_in, sky2_model_in, sci_model_in, cont_corr_out, model_fiber=2):
    """
    
    correct and combine continuum only spectra by doing:   sky_cont_corr=0.5*( sky1_cont*(model_skysci/model_sky1) + sky2_cont*(model_skysci/model_sky2))
    """

    # read sky continuum from both telescopes
    sky1_cont = Spectrum1D()
    sky1_cont.loadFitsData(sky1_cont_in)

    sky2_cont = Spectrum1D()
    sky2_cont.loadFitsData(sky2_cont_in)

    # read sky models for all pointings
    sky1_rss = RSS()
    sky1_rss.loadFitsData(sky1_model_in)
    sky2_rss = RSS()
    sky2_rss.loadFitsData(sky2_model_in)
    sci_rss = RSS()
    sci_rss.loadFitsData(sci_model_in)

    sky1_model = sky1_rss[model_fiber]
    sky2_model = sky2_rss[model_fiber]
    sci_model = sci_rss[model_fiber]

    # match wavelength resolution and wavelenth across telescopes using science pointing as reference
    if np.all(sky1_model._wave != sci_model._wave):
        sky1_model = sky1_model.resampleSpec(sci_model._wave)
    if np.all(sky2_model._wave != sci_model._wave):
        sky2_model = sky2_model.resampleSpec(sci_model._wave)

    if np.all(sky1_model._inst_fwhm != sci_model._inst_fwhm):
        sky1_model.matchFWHM(sci_model._inst_fwhm)
    if np.all(sky2_model._inst_fwhm != sci_model._inst_fwhm):
        sky2_model.matchFWHM(sci_model._inst_fwhm)    

    # extrapolate sky pointings into science pointing
    w_1 = sci_model / sky1_model
    w_2 = sci_model / sky2_model
    # TODO: smooth high frequency features in weights

    # TODO: implement sky coordinates interpolation
    # TODO: implement interpolation in the parameter space

    # TODO: propagate error in continuum correction
    # TODO: propagate mask
    # TODO: propagate LSF
    cont_fit = 0.5 * (w_1 * sky1_cont + w_2 * sky2_cont)
    cont_fit.writeFitsData(cont_corr_out)


def coaddContinuumLine_drp(sky_cont_corr, sky_line_corr):
    """coadd corrected line and continuum combined sky frames: sky_corr=sky_cont_corr+sky_line_corr"""
    sky_corr = sky_cont_corr + sky_line_corr
    return sky_corr


def subtractSky_drp(rss_in, rss_out, sky, sky_out, factor='1', scale_region='', scale_ind=False, parallel='auto'):
    """
        Subtracts a (sky) spectrum, which was stored as a FITS file, from the whole RSS.
        The error will be propagated if the spectrum AND the RSS contain error information.

        Parameters
        --------------
        rss_in : string
                Input RSS FITS file
        rss_out : string
                Output RSS FITS file with spectrum subtracted
        sky : string
                Input sky spectrum in FITS format.
        sky_out: string
                Output file to store the RSS sky spectra.
        factor: string of float, optional with default: '1'
                The default value for the flux scale factor in case the fitting fails
        scale_region: string of tuple of floats, optional with default: ''
                The wavelength range within which the 'factor' will be fit
        scale_ind: boolean, optional with deafult: False
                Whether apply factors individually or apply the median of good factors
        parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
                Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
                for the given system is used.

        Examples
        ----------------
        user:> drp sky subtractSkySpec RSS_IN.fits RSS_OUT.fits SKY_SPEC.fits
    """

    factor = np.array(factor).astype(np.float32)
    scale_ind = bool(scale_ind)
    if scale_region != '':
        region = scale_region.split(',')
        wave_region = [float(region[0]), float(region[1])]
    rss = RSS()
    rss.loadFitsData(rss_in)
    
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky)
    
    sky_head = Header()
    sky_head.loadFitsHeader()
    
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
    rss.writeFitsData(rss_out)
    sky_rss.writeFitsData(sky_out)


def refineContinuum_drp():
    """
    optionally apply an extra residual continuum subtraction using the faintest (i.e. with no stellar light detection) spaxels in the science IFU

    This relies in the availability of dark enough spaxels in the science pointing.
    """
    pass


def subtractPCAResiduals_drp():
    """PCA residual subtraction"""
    pass
