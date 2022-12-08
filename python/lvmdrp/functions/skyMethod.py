# encoding: utf-8
#
# @Author: The LVM Sky Team
# @Date: Dec 7, 2022
# @Filename: drp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from multiprocessing import Pool
from scipy import optimize

from lvmdrp.core.constants import SKYCORR_CONFIG_PATH
from lvmdrp.core.sky import run_skycorr, run_skymodel, optimize_sky
from lvmdrp.core.passband import PassBand
from lvmdrp.core.spectrum1d import Spectrum1D
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
            sky_spec.binSpec(new_wave=sci_spec._wave)
        sky_line, sky_cont = run_skycorr(skycorr_config=skycorr_config, sci_spec=sci_spec, sky_spec=sky_spec)
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


def evalESOSky_drp(sky1_par, sky2_par, sci_par):
    """run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc) to evaluate sky spectrum at each
    telescope pointing (model_sky1, model_sky2, model_skysci)
    
    Parameters:
    -----------
    {sky1, sky2, sci}_par: dictionary_like
        Parameters to evaluate/calculate the ESO sky corresponding to each pointing

    Returns
    -------
    sky1_model, sky2_model, sci_model

    """
    _, sky1_model = run_skymodel(**sky1_par)
    _, sky2_model = run_skymodel(**sky2_par)
    _, sci_model = run_skymodel(**sci_par)    

    return sky1_model, sky2_model, sci_model


def subtractGeocoronal_drp():
    pass


def corrSkyLine_drp(wl_master_sky, sky1_line, sky2_line, sci_line, config):
    """average sky1_line and sky2_line into 'sky_line', and run skycorr on 'sky_line' and 'sci_line' to produce 'sky_line_corr'"""
    # compute a weighted average using as weights the inverse distance distance to science
    w_1, w_2 = None, None
    sky_line = w_1 * sky1_line + w_2 * sky2_line
    # run skycorr on averaged line spectrum
    sky_line_corr = run_skycorr(skycorr_config=config, wl=wl_master_sky, sci_spec=sci_line, sky_spec=sky_line)
    
    return sky_line_corr


def corrSkyContinuum_drp(sky1_cont, sky2_cont, sky1_model, sky2_model, sci_model):
    """correct and combine continuum only spectra by doing:   sky_cont_corr=0.5*( sky1_cont*(model_skysci/model_sky1) + sky2_cont*(model_skysci/model_sky2)) """
    sky_cont_corr = 0.5 * (sky1_cont * (sci_model/sky1_model) + sky2_cont * (sci_model/sky2_model))
    return sky_cont_corr


def coaddContinuumLine_drp(sky_cont_corr, sky_line_corr):
    """coadd corrected line and continuum combined sky frames: sky_corr=sky_cont_corr+sky_line_corr"""
    sky_corr = sky_cont_corr + sky_line_corr
    return sky_corr


def subtractSky_drp(rss_in, rss_out, sky, factor='1', scale_region='', scale_ind=False, parallel='auto'):
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

    factor=np.array(factor).astype(np.float32)
    scale_ind = bool(scale_ind)
    if scale_region != '':
        region = scale_region.split(',')
        wave_region=[float(region[0]), float(region[1])]
    rss = RSS()
    rss.loadFitsData(rss_in)
    sky_spec = Spectrum1D()
    sky_spec.loadFitsData(sky)

    if np.all(rss._wave==sky_spec._wave) and scale_region != '':
        factors=np.zeros(len(rss), dtype=np.float32)
        for i in range(len(rss)):
            try:
                optimum= optimize.fmin(optimize_sky, [1.0], args=(rss[i], sky_spec, wave_region[0], wave_region[1]), disp=0)
                factors[i]=optimum[0]
            except RuntimeError:
                factors[i]=1.0
                rss._mask[i,:] = True
        select_good = factors>0.0
        scale_factor = np.median(factors[select_good])
        for i in range(len(rss)):
            if scale_ind:
                rss[i] = rss[i]/factors[i]-sky_spec
            else:
                if factors[i]>0:
                    rss[i] = rss[i]-sky_spec*np.median(factors[select_good])
    elif np.all(rss._wave==sky_spec._wave) and scale_region=='':
        for i in range(len(rss)):
            rss[i] = rss[i]-sky_spec*factor
        scale_factor=factor

    if len(rss._wave)==2:
        if parallel=='auto':
            pool = Pool(cpu_count())
        else:
            pool = Pool(int(parallel))
        threads=[]
        for i in range(len(rss)):
            threads.append(pool.apply_async(sky_spec.binSpec, args=([rss[i]._wave])))
        pool.close()
        pool.join()

        for i in range(len(rss)):
            # rss[i] = rss[i]-threads[i].get()
            if scale_ind:
                rss[i] = rss[i]/factors[i]-threads[i].get()
            else:
                if factors[i]>0:
                    rss[i] = rss[i]-threads[i].get()*np.median(factors[select_good])

    if scale_region != '':
        rss.setHdrValue('hierarch PIPE SKY SCALE',float('%.3f'%scale_factor),'sky spectrum scale factor')
    rss.writeFitsData(rss_out)


def refineContinuum_drp():
    """optionally apply an extra residual continuum subtraction using the faintest (i.e. with no stellar light detection) spaxels in the science IFU"""
    pass


def subtractPCAResiduals_drp():
    """PCA residual subtraction"""
    pass
