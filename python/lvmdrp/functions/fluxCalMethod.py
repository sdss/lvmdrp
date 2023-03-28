# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 25, 2023
# @Filename: fluxCalMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

# FROM RENBIN'S PAPER (RENBIN+2016) ---------------------------------------------------------------
# TODO: measure the PSF from the guider
#   * stack image guider to obtain the effective PSF for the science frames
#   * fit with a double Gaussian each guiding star with varying A and sigma
#   * pick the sharpest PSF among the N in-focus guiding stars as reference PSF, (p_0) (check that conditions justifying this choice hold for LVM)
#   * choose a effective guiding wavelength (5400 AA)
# TODO: predict wavelength-dependent PSF
#   * seeing as a function of wavelength:
#       p_lambda(r) = p_0 * [r * (lambda/lambda_0)^(1/5)]
#   * interpolate Gunn+2006 to obtain focus offset as a function of wavelength
#   * convolve PSF with a ring kernel (k_[lambda,d](r)) of radii offset*1/n offset*1/N, where f/n is the telescope beam, N is the size of the secondary mirror and d is the distance to the center of the plate
#   * convolve again with a circular step function with radius = radius of the fiber
# TODO: stellar typing
#   * divide the sky-subtracted std stars by the throughput vector (average throughput from tens of previous run)
#   * select fiber with the maximum integrated flux across the wavelength range for each mini-bundle as the reference spectrum
#   * fit a model to each reference spectrum
#   * scale each model spectrum to match the PSF r-band magnitude of individual std stars
# TODO: fitting flux ratios
#   * select fiber with the maximum integrated flux from 3500 to 10500AA
#   * integrate the flux in eight wide wavelength windows
#   * calculate the ratio between each fiber and the reference fiber in each window
#   * run a MCMC to fit for x,y of the star and scaling and rotation of the DAR vector
#   * given a set of the above parameters, compute the expected flux ratios from the PSF model and repeat until minimum chi-square is reached
#   * scale the PSF to smaller and larger sizes and compute the minimum chi-square by fitting a quadratic function, the best fit is found among all the fibers
#   * run the MCMC again using the best PSF
# TODO: deriving throughput loss
#   * compute the fraction of the flux of the PSF that is covered by the fiber as a function of wavelength
#   * compute the expected flux of the star (frac_PSF * model_star)
#   * divide the observed spectra by the expected flux (obs_star / (frac_PSF * model_star))
# -------------------------------------------------------------------------------------------------

# FROM THE CODE BASE (PY3D) -----------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from lvmdrp.core.rss import RSS, loadRSS
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.external import ancillary_func


description = "provides flux calibration tasks"

__all__ = ["createSensFunction_drp", "createSensFunction2_drp", "quickFluxCalibration_drp", "correctTelluric_drp"]


def createSensFunction_drp(in_rss, out_throughput, ref_spec, airmass, exptime, smooth_poly='5', smooth_ref='6.0', smooth_ref2='6.0', median_filt='0',coadd='1', extinct_v='0.0', extinct_curve='mean', aper_correct='1.0',  ref_units='1e-16', target_units='1e-16', column_wave='0', column_flux='1', delimiter='', header='1' , split='', mask_wave='', mask_telluric='', overlap='100', out_star='', verbose='0'):
    smooth_poly=int(smooth_poly)
    smooth_ref=float(smooth_ref)
    smooth_ref2=float(smooth_ref2)
    median_filt=int(median_filt)
    coadd = int(coadd)
    ref_units = float(ref_units)
    target_units = float(target_units)
    aper_correct = float(aper_correct)
    column_wave = int(column_wave)
    column_flux = int(column_flux)
    header = int(header)
    if mask_wave != '':
        mask_wave = np.array(mask_wave.split(',')).astype('float32')
    else:
        mask_wave = None

    if mask_telluric != '':
        mask_telluric = np.array(mask_telluric.split(',')).astype('float32')
    else:
        mask_telluric = None
    verbose=int(verbose)


    ref_star_spec = Spectrum1D()

    if coadd>0:
        rss = RSS()
        rss.loadFitsData(in_rss)
        select = rss.selectSpec(min=0, max=coadd, method='median')
        star_rss=rss.subRSS(select)
        star_spec = star_rss.create1DSpec(method='sum')/aper_correct
    else:
        star_spec = Spectrum1D()
        if '.fits' in in_rss:
            star_spec.loadFitsData(in_rss)
        elif '.txt' in in_rss:
            star_spec.loadTxtData(in_rss)

    try:
        extinct_v = rss.getHdrValue(extinct_v)
    except:
        extinct_v = float(extinct_v)

    try:
        airmass = rss.getHdrValue(airmass)
    except:
        airmass = float(airmass)

    try:
        exptime = rss.getHdrValue(exptime)
    except:
        exptime = float(exptime)

    if extinct_curve=='mean' or extinct_curve=='summer' or extinct_curve=='winter':
        extinct = 10**(ancillary_func.extinctCAHA(star_spec._wave, extinct_v, type=extinct_curve)*airmass*-0.4)
    elif extinct_curve=='Paranal':
        extinct = 10**(ancillary_func.extinctParanal(star_spec._wave)*airmass*-0.4)
    else:
        extinct=Spectrum1D()
        extinct.loadTxtData(extinct_curve)
        extinct = 10**(extinct*airmass*-0.4)
        extinct=extinct.resampleSpec(star_spec._wave)
    ref_star_spec.loadSTDref(ref_spec,column_wave=column_wave, column_flux=column_flux, delimiter=delimiter, header=header)
    ref_star_resamp = ref_star_spec.resampleSpec(star_spec._wave, method='linear')

    ref_star_resamp.smoothSpec(smooth_ref/2.354/(star_spec._wave[1]-star_spec._wave[0]))
    if out_star != '':
        star_out = open(out_star, 'w')
        for i in range(star_spec._dim):
            star_out.write('%i %.3f %e\n'%(i, star_spec._wave[i], star_spec._data[i]))
        star_out.close()

    star_spec.smoothSpec(smooth_ref)
    #print(exptime,extinct._wave,star_spec._wave)
    star_corr = star_spec/extinct/exptime

    throughput = ref_star_resamp/star_corr
    if mask_wave is not None:
        regions = len(mask_wave)/2
        for i in range(regions):
            select_region = np.logical_and(throughput._wave>mask_wave[i*2], throughput._wave<mask_wave[i*2+1])
            select_blue = np.logical_and(throughput._wave>mask_wave[i*2]-20, throughput._wave<mask_wave[i*2])
            select_red = np.logical_and(throughput._wave>mask_wave[i*2+1], throughput._wave<mask_wave[i*2+1]+20)
            line_par = stats.linregress([mask_wave[i*2]-10,mask_wave[i*2+1]+10], [np.median(throughput._data[select_blue]), np.median(throughput._data[select_red])])

            throughput._data[select_region] = (line_par[0]*throughput._wave[select_region]+line_par[1]).astype('float32')
            #select = np.logical_and(throughput._wave>mask_wave[i*2], throughput._wave<mask_wave[i*2+1])
            #throughput._mask[select]=True
    if mask_telluric is not None:
        star_telluric1 = star_rss.create1DSpec(method='sum')
        star_telluric2 = star_rss.create1DSpec(method='sum')
        regions = len(mask_telluric)/2
        for i in range(regions):
            select_region = np.logical_and(star_telluric1._wave>mask_telluric[i*2], star_telluric1._wave<mask_telluric[i*2+1])
            select_blue = np.logical_and(star_telluric1._wave>mask_telluric[i*2]-20, star_telluric1._wave<mask_telluric[i*2])
            select_red = np.logical_and(star_telluric1._wave>mask_telluric[i*2+1], star_telluric1._wave<mask_telluric[i*2+1]+20)
            line_par = stats.linregress([mask_telluric[i*2]-10,mask_telluric[i*2+1]+10], [np.median(star_telluric1._data[select_blue]), np.median(star_telluric1._data[select_red])])
            star_telluric2._data[select_region] = (line_par[0]*star_telluric1._wave[select_region]+line_par[1]).astype('float32')
        telluric_spec = (star_telluric1 / star_telluric2)**(1.0/airmass)
        telluric_spec.writeFitsData('telluric_spec.fits')
    good_pix = np.logical_not(throughput._mask)
    if median_filt>0:
        throughput.smoothSpec(median_filt, method='median')
    if verbose==1:
        plt.plot(throughput._wave[good_pix][10:-10], throughput._data[good_pix][10:-10], '-k')
    if split=='':
        mask = throughput._mask
        throughput_s = 1.0/Spectrum1D(wave=throughput._wave, data=throughput._data, mask=mask)        
        throughput_s.smoothPoly(smooth_poly)
        mask = np.logical_or(throughput_s._mask, throughput_s._data<=0)
        throughput_s = 1.0/throughput_s
        throughput_s._mask = mask
        if verbose==1:
            plt.plot(throughput_s._wave,  throughput_s._data, '-r')
            plt.plot(throughput_s._wave,  throughput._data/throughput_s._data, '-g')
            # sens_test_out = open('test_sens.txt', 'w')
            # for i in range(throughput_s._dim):
            #     sens_test_out.write('%i %.2f %e %e %e\n'%(i, throughput_s._wave[i], throughput._data[i], throughput_s._data[i], throughput._data[i]/throughput_s._data[i]))
            # sens_test_out.close()
    else:
        split = float(split)
        overlap = float(overlap)
        select = throughput._wave>split
        mask = throughput._mask[select]
        mask[-10:]=True
        throughput_s2 = Spectrum1D(wave=throughput._wave[select], data=throughput._data[select], mask=mask)
        throughput_s2.smoothPoly(smooth_poly)

        select = throughput._wave<split+overlap
        mask = throughput._mask[select]
        mask[-10:]=True
        throughput_s1 = Spectrum1D(wave=throughput._wave[select], data=throughput._data[select], mask=mask)
        throughput_s1.smoothPoly(smooth_poly)

        if verbose==1:
            plt.plot(throughput_s1._wave,  throughput_s1._data, '-r')
            plt.plot(throughput_s2._wave,  throughput_s2._data, '-r')
    if verbose==1:
        plt.show()

    throughput_s.writeFitsData(out_throughput)


def createSensFunction2_drp(in_rss, out_sens, ref_spec, airmass, exptime, smooth_bspline='0.3', smooth_ref='6.0', smooth_ref2='6.0', median_filt='0',coadd='1', extinct_v='0.0', extinct_curve='mean', aper_correct='1.0',  ref_units='1e-16', target_units='1e-16',column_wave='0', column_flux='1', delimiter='', header='1' , mask_wave='', out_star='', verbose='0'):
	smooth_bspline=float(smooth_bspline)
	smooth_ref=float(smooth_ref)
	smooth_ref2=float(smooth_ref2)
	median_filt=int(median_filt)
	coadd = int(coadd)
	ref_units=float(ref_units)
	target_units=float(target_units)
	aper_correct=float(aper_correct)
	column_wave = int(column_wave)
	column_flux = int(column_flux)
	header = int(header)
	if mask_wave != '':
		mask_wave = np.array(mask_wave.split(',')).astype('float32')
	else:
		mask_wave=None
	verbose=int(verbose)

	rss = RSS()
	if coadd>0:

		rss.loadFitsData(in_rss)
		select = rss.selectSpec(min=0, max=coadd, method='median')
		star_rss=rss.subRSS(select)
		star_spec = star_rss.create1DSpec(method='sum')/aper_correct
	else:
		star_spec = Spectrum1D()
		if '.fits' in in_rss:
			star_spec.loadFitsData(in_rss)
		elif '.txt' in in_rss:
			star_spec.loadTxtData(in_rss)

	try:
		extinct_v = rss.getHdrValue(extinct_v)
	except: #KeyError or TypeError:
		extinct_v= float(extinct_v)

	try:
		airmass = rss.getHdrValue(airmass)
	except:# KeyError or TypeError:
		airmass = float(airmass)

	try:
		exptime = rss.getHdrValue(exptime)
	except: # KeyError or TypeError:
		exptime = float(exptime)

	if extinct_curve=='mean' or extinct_curve=='summer' or extinct_curve=='winter':
		extinct = 10**(ancillary_func.extinctCAHA(star_spec._wave, extinct_v, type=extinct_curve)*airmass*-0.4)
	elif extinct_curve=='Paranal':
		extinct = 10**(ancillary_func.extinctParanal(star_spec._wave)*airmass*-0.4)
	else:
		extinct=Spectrum1D()
		extinct.loadTxtData(extinct_curve)
		extinct = 10**(extinct*airmass*-0.4)
		extinct=extinct.resampleSpec(star_spec._wave)

	ref_star_spec = Spectrum1D()
	ref_star_spec.loadSTDref(ref_spec,column_wave=column_wave, column_flux=column_flux, delimiter=delimiter, header=header)
	ref_star_resamp = ref_star_spec.resampleSpec(star_spec._wave, method='linear')

	ref_star_resamp.smoothSpec(smooth_ref/2.354/(star_spec._wave[1]-star_spec._wave[0]))
	if out_star != '':
		star_out = open(out_star, 'w')
		for i in range(star_spec._dim):
			star_out.write('%i %.3f %e\n'%(i, star_spec._wave[i], star_spec._data[i]))
		star_out.close()

	star_spec.smoothSpec(smooth_ref)
	star_corr = star_spec/extinct/exptime

	throughput = ref_star_resamp/star_corr
	if mask_wave is not None:
		regions = len(mask_wave)/2
		for i in range(regions):
			select_region = np.logical_and(throughput._wave>mask_wave[i*2], throughput._wave<mask_wave[i*2+1])
			select_blue = np.logical_and(throughput._wave>mask_wave[i*2]-20, throughput._wave<mask_wave[i*2])
			select_red = np.logical_and(throughput._wave>mask_wave[i*2+1], throughput._wave<mask_wave[i*2+1]+20)
			line_par = stats.linregress([mask_wave[i*2]-10,mask_wave[i*2+1]+10], [np.median(throughput._data[select_blue]), np.median(throughput._data[select_red])])

			throughput._data[select_region] = (line_par[0]*throughput._wave[select_region]+line_par[1]).astype('float32')

	good_pix = np.logical_not(throughput._mask)
	if median_filt>0:
		throughput.smoothSpec(median_filt,method='median')
	if verbose==1:
		plt.plot(throughput._wave[good_pix][10:-10], 1.0/throughput._data[good_pix][10:-10], '-k')

	mask = throughput._mask
	#mask[:10]=True
	#mask[-10:]=True
	throughput_s = Spectrum1D(wave=throughput._wave, data=1.0/throughput._data, mask=mask)
	throughput_s.smoothSpec(smooth_bspline,method='BSpline')
	if verbose==1:
		plt.plot(throughput_s._wave,  throughput_s._data, '-r')
		plt.plot(throughput_s._wave,  (1.0/throughput._data)/throughput_s._data, '-g')
		sens_test_out = open('test_sens.txt', 'w')
		for i in range(throughput_s._dim):
			sens_test_out.write('%i %.2f %e %e %e\n'%(i, throughput_s._wave[i], throughput._data[i], throughput_s._data[i], throughput._data[i]/throughput_s._data[i]))
		sens_test_out.close()
		plt.show()
	throughput_s = 1.0/throughput_s


	# need to replace with XML output
	out = open(out_sens, 'w')
	for i in range(throughput._dim):
		out.write('%i %.3f %e\n'%(i,  throughput_s._wave[i], throughput_s._data[i]))
	out.close()


def quickFluxCalibration_drp(in_rss, out_rss, in_throughput, airmass, exptime, extinct_v='0.0', extinct_curve='mean', ref_units='1e-16', target_units='1e-16', norm_sb_fib=''):
	ref_units=float(ref_units)
	target_units=float(target_units)
	rss = RSS()
	rss.loadFitsData(in_rss)
	if norm_sb_fib=='':
		norm_sb_fib=1.0
	else:
		norm_sb_fib=np.pi*float(norm_sb_fib)**2
	try:
		airmass = rss.getHdrValue(airmass)
	except KeyError:
		airmass = float(airmass)

	try:
		exptime = rss.getHdrValue(exptime)
	except KeyError:
		exptime = float(exptime)

	try:
		extinct_v = rss.getHdrValue(extinct_v)
	except KeyError:
		extinct_v = float(extinct_v)

	troughput_file = open(in_throughput, 'r')
	lines = troughput_file.readlines()
	wave_sens = np.zeros(len(lines), dtype=np.float32)
	sens_dat = np.zeros(len(lines), dtype=np.float32)
	for i in range(len(lines)):
		line = lines[i].split()
		if len(line)==3:
			wave_sens[i]=float(line[1])
			sens_dat[i]=float(line[2])
		elif len(line)==2:
			wave_sens[i]=float(line[0])
			sens_dat[i]=float(line[1])
	throughput = Spectrum1D(wave=wave_sens, data=sens_dat)
	if len(rss._wave.shape)==1:
		if extinct_curve=='mean' or extinct_curve=='summer' or extinct_curve=='winter':
			extinct = 10**(ancillary_func.extinctCAHA(rss._wave, extinct_v, type=extinct_curve)*airmass*-0.4)
		elif extinct_curve=='Paranal':
			extinct = 10**(ancillary_func.extinctParanal(rss._wave)*airmass*-0.4)
		else:
			extinct=Spectrum1D()
			extinct.loadTxtData(extinct_curve)
			extinct = 10**(extinct*airmass*-0.4)
			extinct = extinct.resampleSpec(rss._wave, method='spline')
		throughput_resamp = throughput.resampleSpec(rss._wave, method='spline')

		for j in range(rss._fibers):
			rss[j] = (rss[j]/extinct/exptime/norm_sb_fib)*throughput_resamp*(ref_units/target_units)
	#        print exptime
	rss.writeFitsData(out_rss)


def correctTelluric_drp(in_rss, out_rss, telluric_spectrum, airmass='AIRMASS'):
	"""
        Corrects the wavelength calibrated RSS for the effect of telluric absoroption using
        a transmission spectrum generated from a star.

        Parameters
        --------------
        in_rss : string
                Input RSS FITS file
        out_rss : string
                Output RSS FITS file with the corrected spectra
        telluric_spectrum : string
                FITS file of the telluric transmission spectrum
        airmass : string or string of float, optional with default: 'AIRMASS'
                Airmass for the target observation.
                Either a corresponding header keyword or a float value may be used.

        Examples
        ----------------
        user:> lvmdrp rss correctTelluric in_rss.fits out_rss.fits TELL_SPEC.fits
        user:> lvmdrp rss correctTelluric in_rss.fits out_rss.fits TELL_SPEC.fits  1.4
	"""
	rss = loadRSS(in_rss)
	telluric = Spectrum1D()
	telluric.loadFitsData(telluric_spectrum)
	telluric._mask = None
	telluric._error = None

	try:
		airmass = rss.getHdrValue(airmass)
	except KeyError:
		airmass = float(airmass)
	if len(rss._wave.shape)==1:
		telluric_resamp = telluric.resampleSpec(rss._wave)
		rss_corr = rss*(1.0/(telluric_resamp**(airmass)))

	elif len(rss._wave.shape)==2:
		rss_corr = rss
		for i in range(len(rss._fibers)):
			spec = rss[i]
			telluric_resamp = telluric.resampleSpec(spec._wave)
			rss_corr[i] = spec*(1.0/(telluric_resamp**(airmass)))
	rss_corr.writeFitsData(out_rss)

# -------------------------------------------------------------------------------------------------