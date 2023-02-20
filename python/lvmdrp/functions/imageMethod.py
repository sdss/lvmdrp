import os, sys, numpy
from astropy.io import fits as pyfits
try:
  import pylab
  from matplotlib import pyplot as plt
except:
  pass
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy import interpolate
from lvmdrp.core.image import loadImage, Image, glueImages, combineImages
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.rss import RSS
from lvmdrp.utils.decorators import missing_files

import multiprocessing
from types import *


description = 'Provides Methods to process 2D images'

__all__ = [
	"LACosmic_drp", "findPeaksAuto_drp", "tracePeaks_drp",
	"subtractStraylight_drp", "traceFWHM_drp", "extractSpec_drp",
	"subtractBias_drp", "preprocRawFrame_drp", "basicCalibration_drp",
	"createMasterFrame_drp"
]


def detCos_drp(image,  out_image,   rdnoise='2.9', sigma_det='5', rlim='1.2', iter='5', fwhm_gauss='2.0', replace_box='5,5',  error_box='5,5', replace_error='1e10', increase_radius='0', gain='1.0', verbose='0', parallel='auto'):
	"""
			Detects and removes cosmics from astronomical images based on Laplacian edge
			detection scheme combined with a PSF convolution approach (Husemann  et al. in prep.).

			IMPORTANT:
			The image and the readout noise are assumed to be in units of electrons.
			The image also needs to be BIAS subtracted! The gain can be entered to convert the image from ADUs to electros, when this is down already set gain=1.0 as the default.

			Parameters
			--------------
			image: string
					Name of the FITS file for which the comsics should be detected
		out_mask: string
					Name of the  FITS file with the bad pixel mask
			out_clean: string
					Name of the  FITS file with the cleaned image
			rdnoise: float or string of header keyword
					Value or FITS header keyword for the readout noise in electrons
			sigma_det: float, optional  with default: 5.0
					Detection limit of edge pixel above the noise in (sigma units) to be detected as comiscs
			rlim: float, optional  with default: 1.2
					Detection threshold between Laplacian edged and Gaussian smoothed image
			iter: integer, optional with default: 5
					Number of iterations. Should be >1 to fully detect extended cosmics
			fwhm_gauss: float, optional with default: 2.0
					FWHM of the Gaussian smoothing kernel in x and y direction on the CCD
			replace_box: array of two integers, optional with default: [5,5]
					median box size in x and y to estimate replacement values from valid pixels
			replace_error: float, optional with default: 1e10
					Error value for bad pixels in the comupted error image, will be ignored if empty
			increase_radius: integer, optional with default: 0
					Increase the boundary of each detected cosmic ray pixel by the given number of pixels.
			verbose: bollean, optional  with default: True
					Show information during the processing on the command line (0 - no, 1 - yes)


			References
			--------------
			B. Husemann et al. 2012  "", A&A, ??, ???

	"""
	# convert all parameters to proper type
	sigma_det = float(sigma_det)
	rlim= float(rlim)
	iterations = int(iter)
	sigma= float(fwhm_gauss)/2.354
	error_box = replace_box.split(',')
	err_box_x = int(error_box[0])
	err_box_y = int(error_box[1])
	replace_box = replace_box.split(',')
	box_x = int(replace_box[0])
	box_y = int(replace_box[1])
	increase_radius=int(increase_radius)
	verbose=int(verbose)
	try:
		replace_error = float(replace_error)
	except:
		replace_error = None

	# load image from FITS file
	img = loadImage(image)
	try:
		gain=img.getHdrValue(gain)
	except KeyError:
		pass
	gain = float(gain)

	if gain!=1.0 and verbose==True:
		print('Convert image from ADUs to electrons using a gain factor of %f' %(gain))

	img = img*gain
	#img.writeFitsData('test.fits')

	# create empty mask if no mask is present in original image
	if img._mask is not None:
		mask_orig=img.getMask()
	else:
		mask_orig=numpy.zeros(img.getDim(), dtype=numpy.bool)

	# create a new Image instance to store the initial data array
	img_original = Image(data=img.getData(), header=img.getHeader(), error = img.getError(),  mask=mask_orig)
	img.setData(mask=numpy.zeros(img.getDim(), dtype=numpy.bool))
	img.removeError()

	# estimate Poisson noise after roughly cleaning cosmics using a median filter
	try:
		rdnoise=float(img.getHdrValue(rdnoise))
	except KeyError:
		rdnoise=float(rdnoise)
	if verbose==True:
		print('A value of %f is used for the electron read-out noise.'%(rdnoise))


	# create empty mask
	select = numpy.zeros(img.getDim(),dtype=numpy.bool)

	# define Laplacian convolution kernal
	LA_kernel=numpy.array([[0,-1,0,],[-1,4,-1],[0,-1,0]])/4.0
	out=img

	if parallel:
		try:
			from multiprocessing import Pool
			from multiprocessing import cpu_count
			cpus = cpu_count()
			if cpus>1:
				cpus=2
		except:
			cpus=1

	else:
		cpus = 1
	# start iteration
	if verbose:
		print('Start the detection process using %d CPU cores.'%(cpus))
	for i in range(iterations):
		if verbose:
			print('Start iteration %i'%(i+1))
		# follow the LACosmic scheme to select pixel
		noise =out.medianImg((err_box_x, err_box_y))
		select_neg2 = noise.getData()<=0
		noise.setData(data=0, select=select_neg2)
		noise=(noise+rdnoise**2).sqrt()
		result = []
		if cpus>1:
			fine=out.convolveGaussImg(sigma, sigma, mask=True)
			fine_norm = out/fine
			select_neg = fine_norm<0
			fine_norm.setData(data=0, select=select_neg)
			pool = Pool(cpus)
			result.append(pool.apply_async(out.subsampleImg, args=([2])))
			result.append(pool.apply_async(fine_norm.subsampleImg, args=([2])))
			pool.close()
			pool.join()
			sub = result[0].get()
			sub_norm = result[1].get()
			pool.terminate()
			pool = Pool(cpus)
			result[0]=pool.apply_async(sub.convolveImg, args=([LA_kernel]))
			result[1]=pool.apply_async(sub_norm.convolveImg, args=([LA_kernel]))
			pool.close()
			pool.join()
			conv = result[0].get()
			select_neg = conv<0
			conv.setData(data=0, select=select_neg)  # replace all negative values with 0
			Lap2 = result[1].get()
			pool.terminate()
			pool = Pool(cpus)
			result[0]=pool.apply_async(conv.rebin, args=(2, 2))
			result[1]=pool.apply_async(Lap2.rebin, args=(2, 2))
			pool.close()
			pool.join()
			Lap = result[0].get()
			Lap2 = result[1].get()
			pool.terminate()
			S = Lap/(noise*2) # normalize Laplacian image by the noise
			S_prime = S-S.medianImg((5, 5)) # cleaning of the normalized Laplacian image
		else:
			sub = out.subsampleImg(2) # subsample image
			conv= sub.convolveImg(LA_kernel) # convolve subsampled image with kernel
			select_neg = conv<0
			conv.setData(data=0, select=select_neg)  # replace all negative values with 0
			Lap = conv.rebin(2, 2) # rebin the data to original resolution
			S = Lap/(noise*2) # normalize Laplacian image by the noise
			S_prime = S-S.medianImg((5, 5)) # cleaning of the normalized Laplacian image
			fine=out.convolveGaussImg(sigma, sigma, mask=True) # convolve image with a 2D Gaussian

			fine_norm = out/fine
			select_neg = fine_norm<0
			fine_norm.setData(data=0, select=select_neg)
			sub_norm = fine_norm.subsampleImg(2) # subsample image
			Lap2 = (sub_norm).convolveImg(LA_kernel)
			Lap2 = Lap2.rebin(2, 2) # rebin the data to original resolution

		select = numpy.logical_or(numpy.logical_and((Lap2)>rlim, S_prime>sigma_det),  select)

		# print information on the screen if demanded
		if verbose:
			dim = img_original.getDim()
			det_pix = numpy.sum(select)
			print('Total number of detected cosmics: %i out of %i pixels'%(numpy.sum(select), dim[0]*dim[1]))

		if i==iterations-1:
			img_original.setData(mask=True, select=select) # set the new mask
			if increase_radius>0:
				mask_img = Image(data=img_original._mask)
				mask_new=mask_img.convolveImg(kernel=numpy.ones((2*increase_radius+1, 2*increase_radius+1)))
				img_original.setData(mask=mask_new._data)
			out=img_original.replaceMaskMedian(box_x, box_y, replace_error=replace_error) # replace possible corrput pixel with zeros for final output
		else:
			out.setData(mask=True, select=select)# set the new mask
			out = out.replaceMaskMedian(box_x, box_y, replace_error=None)  # replace possible corrput pixel with zeros
	out.writeFitsData(out_image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image")
def LACosmic_drp(in_image, out_image, sigma_det='5', flim='1.1', iter='3', sig_gauss='0.8,0.8', error_box='20,1', replace_box='20,1',  replace_error='1e10',  increase_radius='0', parallel='2'):
	# convert all parameters to proper type
	sigma_det = float(sigma_det)
	flim = float(flim)
	iter = int(iter)
	error_box = replace_box.split(',')
	error_box = int(error_box[0]), int(error_box[1])
	sig_gauss = sig_gauss.split(',')
	sig_gauss = float(sig_gauss[0]), float(sig_gauss[1])
	replace_box = replace_box.split(',')
	replace_box = int(replace_box[0]), int(replace_box[1])
	increase_radius = int(increase_radius)
	try:
		replace_error = float(replace_error)
	except:
		replace_error = None

	# load image from FITS file
	img = loadImage(in_image)

	# create empty mask if no mask is present in original image
	if img._mask is not None:
		mask_orig = img.getMask()
	else:
		mask_orig = numpy.zeros(img.getDim(), dtype=numpy.bool)

	# create a new Image instance to store the initial data array
	img_original = Image(data=img.getData(), header=img.getHeader(), error=img.getError(),  mask=mask_orig)
	img.setData(mask=numpy.zeros(img.getDim(), dtype=numpy.bool))
	img.removeError()

	cr_select = img.createCosmicMask(sigma_det=sigma_det, flim=flim, iter=iter, sig_gauss=sig_gauss, error_box=error_box, replace_box=replace_box, parallel=parallel)

	# update mask in original image
	img_original.setData(mask=True, select=cr_select)

	# refine CR selection
	if increase_radius>0:
		mask_img = Image(data=img_original._mask)
		mask_new = mask_img.convolveImg(kernel=numpy.ones((2*increase_radius+1, 2*increase_radius+1)))
		mask_new = mask_new._data>0
		img_original.setData(mask=mask_new)
	out = img_original.replaceMaskMedian(replace_box[0], replace_box[1], replace_error=replace_error)
	out.writeFitsData(out_image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image")
def old_LACosmic_drp(in_image,  out_image,  sigma_det='5', flim='1.1', iter='3', sig_gauss='0.8,0.8', error_box='20,1', replace_box='20,1',  replace_error='1e10',  rdnoise='2.9',  increase_radius='0', verbose='0', parallel='2'):
	"""
			Detects and removes cosmic rays from astronomical images based on a modified Laplacian edge
			detection method introduced by van Dokkum (2005) and modified by B. Husemann (2012, in prep.).

			IMPORTANT:
			The image and the readout noise are assumed to be in units of electrons.
			The image also need to be bias subtracted so that the proper Poisson noise image can be estimated.

			Parameters
			--------------
			image: string
					Name of the FITS file for which the comsics should be detected
			out_image: string
					Name of the  FITS file containing the cleaned image, a bad pixel mask extension and the error image if contained in the  input
			sigma_det: string of float, optional  with default: '5.0'
					Detection limit of edge pixel above the noise in (sigma units) to be detected as comiscs
			flim: string of float, optional  with default: '1.1'
					Detection threshold between Laplacian edged and Gaussian smoothed image (should be >1)
			iter: string of integer, optional with default: '3'
					Number of iterations. Should be >1 to fully detect extended cosmics
			sig_gauss: string of two comma separated floats, optional with default: '0.8,0.8'
					Sigma width of the Gaussian smoothing kernel in x and y direction on the CCD
			error_box: string of two comma separated integers, optional with default: '20,1'
					Pixel box width (x and y width on raw image) used to estimate the electron counts for a given pixel by taken a median to estimate the noise level.
					It may be elongated along the dispersion axis of the CCD.
			replace_box: string of two comma separated integers, optional with default: '5,5'
					median box size in x and y to estimate replacement values from valid pixels
			replace_error: strong of float, optional with default: '1e10'
					Error value for bad pixels in the comupted error image, will be ignored if empty
			rdnoise: string of float or string of header keyword, optional with default: 2.9
					Value or FITS header keyword for the readout noise in electrons, not used if an error image of a FITS extension is used
			increase_radius: string of int, optional with default: '0'
					Increase the boundary of each detected cosmic ray pixel by the given number of pixels.
			verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Notes
			-------
			As described in the article by van Dokkum a fine structure images is created to distinguish
			between cosmic rays hits and compact real signals that are almost undersampled on the CCD.
			For IFU data these are mainly emission lines from the sky or more importantly from the
			target objects itself.
			We defined a new fine structure map as the ratio between the Lapacian image and a Gaussian
			smoothed image with a width of the Gaussian matching the PSF of the spectrograph in dispersion
			AND cross-dispersion direction. If the given PSF is perfectly matching wtih the true PSF, a limit of
			flim>1 is the cutting line between true signal and cosmic ray hits.

			References
			--------------
			van Dokkum, Pieter G. 2001, "Cosmic-Ray Rejection by Laplacian Edge Detection",
			PASP, 113, 1420

			Examples
			----------------
			user:> lvmdrp image LACosmic IMAGE.fits MASK.fits CLEAN.fits 5 flim=1.1 sig_gauss=0.8,0.8 replace_box=5,5 increase_radius=1
	"""
	# convert all parameters to proper type
	sigma_det = float(sigma_det)
	flim= float(flim)
	iterations = int(iter)
	error_box = replace_box.split(',')
	err_box_x = int(error_box[0])
	err_box_y = int(error_box[1])
	sig_gauss = sig_gauss.split(',')
	sigma_x = float(sig_gauss[0])
	sigma_y = float(sig_gauss[1])
	replace_box = replace_box.split(',')
	box_x = int(replace_box[0])
	box_y = int(replace_box[1])
	increase_radius=int(increase_radius)
	verbose=int(verbose)
	try:
		replace_error = float(replace_error)
	except:
		replace_error = None

	# load image from FITS file
	img = loadImage(in_image)

	# create empty mask if no mask is present in original image
	if img._mask is not None:
		mask_orig=img.getMask()
	else:
		mask_orig=numpy.zeros(img.getDim(), dtype=numpy.bool)

	# create a new Image instance to store the initial data array
	img_original = Image(data=img.getData(), header=img.getHeader(), error = img.getError(),  mask=mask_orig)
	img.setData(mask=numpy.zeros(img.getDim(), dtype=numpy.bool))
	img.removeError()

	# estimate Poisson noise after roughly cleaning cosmics using a median filter
	try:
		rdnoise = float(rdnoise)
	except:
		rdnoise=img.getHdrValue(rdnoise)


	# create empty mask
	select = numpy.zeros(img.getDim(),dtype=numpy.bool)

	# define Laplacian convolution kernal
	LA_kernel=numpy.array([[0,-1,0,],[-1,4,-1],[0,-1,0]])/4.0
	out=img

	if parallel=='auto':
		cpus = cpu_count()
	else:
		cpus = int(parallel)
	# start iteration
	for i in range(iterations):
		if verbose==1:
			print('iteration %i'%(i+1))
		# follow the LACosmic scheme to select pixel
		noise =out.medianImg((err_box_y, err_box_x))
		select_noise = noise.getData()<=0
		noise.setData(data=0, select=select_noise)
		noise=(noise+rdnoise**2).sqrt()
		result = []
		if cpus>1:
			fine=out.convolveGaussImg(sigma_x, sigma_y)
			fine_norm = out/fine
			select_neg = fine_norm<0
			fine_norm.setData(data=0, select=select_neg)
			pool = Pool(cpus)
			result.append(pool.apply_async(out.subsampleImg))
			result.append(pool.apply_async(fine_norm.subsampleImg))
			pool.close()
			pool.join()
			sub = result[0].get()
			sub_norm = result[1].get()
			pool.terminate()
			pool = Pool(cpus)
			result[0]=pool.apply_async(sub.convolveImg, args=([LA_kernel]))
			result[1]=pool.apply_async(sub_norm.convolveImg, args=([LA_kernel]))
			pool.close()
			pool.join()
			conv = result[0].get()
			select_neg = conv<0
			conv.setData(data=0, select=select_neg)  # replace all negative values with 0
			Lap2 = result[1].get()
			pool.terminate()
			pool = Pool(cpus)
			result[0]=pool.apply_async(conv.rebin, args=(2, 2))
			result[1]=pool.apply_async(Lap2.rebin, args=(2, 2))
			pool.close()
			pool.join()
			Lap = result[0].get()
			Lap2 = result[1].get()
			pool.terminate()
			S = Lap/(noise*4) # normalize Laplacian image by the noise
			S_prime = S-S.medianImg((err_box_y, err_box_x)) # cleaning of the normalized Laplacian image


		else:
			sub = out.subsampleImg() # subsample image
			conv= sub.convolveImg(LA_kernel) # convolve subsampled image with kernel
			select_neg = conv<0
			conv.setData(data=0, select=select_neg)  # replace all negative values with 0
			Lap = conv.rebin(2, 2) # rebin the data to original resolution
			S = Lap/(noise*4) # normalize Laplacian image by the noise
			S_prime = S-S.medianImg((err_box_y, err_box_x)) # cleaning of the normalized Laplacian image
			fine=out.convolveGaussImg(sigma_x, sigma_y) # convolve image with a 2D Gaussian
	#        fine.writeFitsData('s_prime.fits')
			fine_norm = out/fine
			select_neg = fine_norm<0
			fine_norm.setData(data=0, select=select_neg)
			sub_norm = fine_norm.subsampleImg() # subsample image
			Lap2 = (sub_norm).convolveImg(LA_kernel)
			Lap2 = Lap2.rebin(2, 2) # rebin the data to original resolution

		##select = numpy.logical_or(numpy.logical_and(S_prime>sigma_det,(Lap/fine)>flim),select) # select bad pixels
		select = numpy.logical_or(numpy.logical_and((Lap2)>flim, S_prime>sigma_det),  select)

		# print information on the screen if demanded
		if verbose==1:
			dim = img_original.getDim()
			det_pix = numpy.sum(select)
			print('Detected pixels: %i out of %i '%(numpy.sum(select), dim[0]*dim[1]))

		if i==iterations-1:
			img_original.setData(mask=True, select=select) # set the new mask
			if increase_radius>0:
				# print numpy.sum(img_original._mask)
				mask_img = Image(data=img_original._mask)
				mask_new=mask_img.convolveImg(kernel=numpy.ones((2*increase_radius+1, 2*increase_radius+1)))
				# print numpy.sum(mask_new._data)
				mask_new = mask_new._data>0
				img_original.setData(mask=mask_new)
			out=img_original.replaceMaskMedian(box_x, box_y, replace_error=replace_error) # replace possible corrput pixel with zeros for final output
		else:
			out.setData(mask=True, select=select)# set the new mask
			out = out.replaceMaskMedian(box_x, box_y, replace_error=None)  # replace possible corrput pixel with zeros
	out.writeFitsData(out_image)

def addCCDMask_drp(image, mask, replaceError='1e10'):
	"""
		   Adds a mask image (containing only zeros and ones) as new FITS extension to the original image.
		   Values of 1 in the mask image are considered as bad pixels. If the image contains already and error image as an
		   extension the bad pixels can be replaced in the error array by a user defined value.

			Parameters
			--------------
			image: string
					Name of the FITS file to which the mask should be added
			mask: string
					Name of the  FITS file containing the mask of a bad pixel (1 if bad pixel) to be added
			replace_error: strong of float, optional with default: '1e10'
					Error value for bad pixels in a possible error extension, will be ignored if empty

			Examples
			----------------
            user:> lvmdrp image addCDDMask IMAGE.fits MASK.fits
	"""

	replaceError = float(replaceError)
	img = loadImage(image)
	bad_pixel = loadImage(mask, extension_mask=0)
	if img._mask is not None:
		mask_comb = numpy.logical_or(img._mask, bad_pixel._mask)
	else:
		mask_comb = bad_pixel._mask
	if img._error is not None:
		img.setData(error=replaceError, select=mask_comb)
	img.setData(mask=mask_comb)
	img.writeFitsData(image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image")
def findPeaksAuto_drp(in_image, out_peaks_file, nfibers,  disp_axis='X', threshold='5000',median_box='8', median_cross='1', slice='', method='gauss',  init_sigma='1.0', verbose='1'):
	"""
		   Finds the exact subpixel cross-dispersion position of a given number of fibers at a certain dispersion column on the raw CCD frame.
		   If a predefined number of pixel are expected, the initial threshold value for the minimum peak height will varied until the expected number
		   pixels are detected.
		   If instead the number of fibers is set to 0, all peaks above the threshold peak height value will be consider as fibers without further iterations.
		   The results are stored in an ASCII file for further processing.

			Parameters
			--------------
			image: string
					Name of the Continuum FITS file in which the fiber position along cross-dispersion direction will be measured
			out_peaks_file : string
					Name of the ASCII file in which the resulting fiber peak positions are stored
			nfibers: string of integer > 0
					Number of fibers for which need to be identified in cross-dispersion
			disp_axies: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X', 'x', or 0 for the  x axis or 'Y', 'y', or 1 for the y axis.
			threshold: string of float or integer  > 0
					Init threshold for the peak heights to be considered as a fiber peak.
			median_box: string of integer, optional  with default: '8'
					Defines a median smoothing box along dispersion axis to  reduce effects of cosmics or bad pixels
			slice: string of integer, optional with default: ''
					Traces the peaks along a given dispersion slice column number. If empty, the dispersion column with the average maximum counts will be used
			method: string, optional with default: 'gauss'
				Set the method to measure the peaks positions, either 'gauss' or 'hyperbolic'.
			init_sigma: string of  float, optional with default: '1.0'
					Init guess for the  sigma width (in pixels units)  for the Gaussian fitting, only used if method 'gauss' is selected
			verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Examples
			----------------
            user:> lvmdrp image findPeaksAuto IMAGE.fits OUT_PEAKS.txt 382  method='gauss', init_sigma=1.3
	"""
	# convert all parameters to proper type
	npeaks=int(nfibers)
	threshold=float(threshold)
	median_box=int(median_box)
	median_cross=int(median_cross)
	init_sigma = float(init_sigma)
	verbose = int(verbose)

	# Load Image
	img = loadImage(in_image)

	# swap axes so that the dispersion axis goes along the x axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()

	# perform median filtering along the dispersion axis to clean cosmic rays
	img = img.medianImg((median_cross, median_box))

	# if no slice is given find the cross-dispersion cut with the highest signal
	if slice=='':
		median_cut=img.collapseImg(axis='y', mode='median') #median collapse of image along cross-dispersion axis
		maximum = median_cut.max() #get maximum value along dispersion axis
		column=maximum[2] # pixel position of maximum value
		cut = img.getSlice(column, axis='y') # extract this column from image
	else:
		column = int(slice) # convert column to integer value
		cut = img.getSlice(column, axis='y') # extract this column from image

	# find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks
	peaks = cut.findPeaks(threshold=threshold, npeaks=npeaks)

	# find the subpixel centroids of the peaks from the central 3 pixels using either a hyperbolic approximation
	# or perform a leastsq fit with a Gaussian
	centers = cut.measurePeaks(peaks[0], method, init_sigma, threshold=0, max_diff=1.0)[0]
	round_cent = numpy.round(centers).astype('int16') # round the subpixel peak positions to their nearest integer value
	# write number of peaks and their position to an ASCII file NEED TO BE REPLACE WITH XML OUTPUT
	file_out = open(out_peaks_file, 'w')
	file_out.write('%i\n' %(column))
	for i in range(len(centers)):
		file_out.write('%i %i %e %i\n'%(i, round_cent[i], centers[i], 0))
	file_out.close()
	if verbose==1:
		# control plot for the peaks NEED TO BE REPLACE BY A PROPER VERSION AND POSSIBLE IMPLEMENTAION FOR A GUI
		print('%i Fibers found'%(len(centers)))
		pylab.plot(cut._data, '-k')
		pylab.plot(peaks[0],peaks[2] ,'or')
		pylab.plot(centers, numpy.ones(len(centers))*4000.0, 'xg')
		pylab.show()

def findPeaksOffset_drp(image, peaks_master, out_peaks_file, disp_axis='X', threshold='1500', median_box='8', median_cross='1', slice='', method='gauss',  init_sigma='1.0',accuracy=1.2):

    threshold=float(threshold)
    median_box=int(median_box)
    median_cross=int(median_cross)
    init_sigma = float(init_sigma)

    # Load Image
    img = loadImage(image)

    # swap axes so that the dispersion axis goes along the x axis
    if disp_axis=='X' or disp_axis=='x':
        pass
    elif disp_axis=='Y' or disp_axis=='y':
       img.swapaxes()

    # perform median filtering along the dispersion axis to clean cosmic rays
    img = img.medianImg((median_cross, median_box))

    # if no slice is given find the cross-dispersion cut with the highest signal
    if slice=='':
        median_cut=img.collapseImg(axis='y', mode='median') #median collapse of image along cross-dispersion axis
        maximum = median_cut.max() #get maximum value along dispersion axis
        column=maximum[2] # pixel position of maximum value
        cut = img.getSlice(column, axis='y') # extract this column from image
    else:
        column = int(slice) # convert column to integer value
        cut = img.getSlice(column, axis='y') # extract this column from image

    master_file = open(peaks_master, 'r')
    lines = master_file.readlines()
    fiber = numpy.zeros(len(lines), dtype=numpy.int16)
    pixel = numpy.zeros(len(lines), dtype=numpy.int16)
    ref_pos = numpy.zeros(len(lines), dtype=numpy.float32)
    fib_qual = []
    for i in range(len(lines)):
        line = lines[i].split()
        fiber[i] = int(line[0])
        pixel[i] = int(line[1])
        ref_pos[i] = float(line[2])
        fib_qual.append(line[3])
    fib_qual=numpy.array(fib_qual)

    select_good = fib_qual=='GOOD'
    #npeaks=numpy.sum(select_good)
    # find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks
    
    accepted = False
    offset = 0.0
    slope = 0.0
    
    ref_positions = ref_pos[select_good]
    while accepted is False:
        #if numpy.sum(select_weak)>0:
        #    select = numpy.logical_and(select_good_weak, select_weak)
        #    pylab.plot(peaks_weak_good[0][select[select_good_weak]],peaks_weak_good[2] [select[select_good_weak]],'ob')
        peaks = cut.findPeaks(threshold=threshold)
        centers = cut.measurePeaks(peaks[0], method, init_sigma, threshold=0, max_diff=1.0)[0]
        plt.clf()
        plt.plot(cut._data, '-k')
        plt.plot(peaks[0],peaks[2] ,'or')
        plt.plot(centers, numpy.ones(len(centers))*2000.0, 'xg')
        mpeaks = plt.plot(ref_positions+(ref_positions-ref_positions[0])*slope+offset,numpy.ones(numpy.sum(select_good))*2000.0,'+b')
        plt.show()
        print('New Threshold (%.1f):'%(threshold))
        line = sys.stdin.readline()
        try:
            threshold = float(line)
        except:
            accepted=True
        print('New Offset (%.1f):'%(offset))
        line = sys.stdin.readline()
        try:
            offset = float(line)
        except:
            pass
        print('New slope (%.1f):'%(slope))
        line = sys.stdin.readline()
        try:
            slope = float(line)
        except:
            pass
        
    #expect_first = ref_pos[select_good][0]
    #shift_peaks=peaks_good[0][0]-expect_first
    #if expect_first>=5 and shift_peaks>5:
    #    idx = numpy.indices(cut._data.shape)[0]
        #while ref_pos[select_good][-1]+shift_peaks+2>cut._data.shape[0]:
        #    last_fiber = idx[select_good][-1]
    
    
    #npeaks = numpy.sum(select_good)
    #peaks_good = cut.findPeaks(threshold=threshold, npeaks=npeaks)
    
    #centers = peaks_ref


    #round_cent = numpy.round(centers._data).astype('int16') # round the subpixel peak positions to their nearest integer value
    file_out = open(out_peaks_file, 'w')
    
    file_out.write('%i\n' %(column))
    for i in range(len(ref_pos)):
        position = (ref_pos[i]-ref_positions[0])*slope+offset+ref_pos[i]
        if select_good[i]:
            diff_arg = numpy.argmin(numpy.fabs(position-centers))
            diff = position-centers[diff_arg]
            if numpy.fabs(diff)<accuracy:
                file_out.write('%i %i %e %i\n'%(i+1, numpy.round(centers[diff_arg]).astype('int16'), centers[diff_arg], 0))
            else:
                file_out.write('%i %i %e %i\n'%(i+1, numpy.round(position).astype('int16'), position, 1))
        else:
            file_out.write('%i %i %e %i\n'%(i+1, numpy.round(position).astype('int16'), position, 1))
    file_out.close()

def findPeaksMaster_drp(image, peaks_master, out_peaks_file, disp_axis='X', threshold='1500', threshold_weak='500', median_box='8', median_cross='1', slice='', method='gauss',  init_sigma='1.0', verbose='1'):

	threshold=float(threshold)
	threshold_weak=float(threshold_weak)
	median_box=int(median_box)
	median_cross=int(median_cross)
	init_sigma = float(init_sigma)
	verbose = int(verbose)

	# Load Image
	img = loadImage(image)

	# swap axes so that the dispersion axis goes along the x axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()

	# perform median filtering along the dispersion axis to clean cosmic rays
	img = img.medianImg((median_cross, median_box))

	# if no slice is given find the cross-dispersion cut with the highest signal
	if slice=='':
		median_cut=img.collapseImg(axis='y', mode='median') #median collapse of image along cross-dispersion axis
		maximum = median_cut.max() #get maximum value along dispersion axis
		column=maximum[2] # pixel position of maximum value
		cut = img.getSlice(column, axis='y') # extract this column from image
	else:
		column = int(slice) # convert column to integer value
		cut = img.getSlice(column, axis='y') # extract this column from image

	master_file = open(peaks_master, 'r')
	lines = master_file.readlines()
	fiber = numpy.zeros(len(lines), dtype=numpy.int16)
	pixel = numpy.zeros(len(lines), dtype=numpy.int16)
	ref_pos = numpy.zeros(len(lines), dtype=numpy.float32)
	fib_qual = []
	for i in range(len(lines)):
		line = lines[i].split()
		fiber[i] = int(line[0])
		pixel[i] = int(line[1])
		ref_pos[i] = float(line[2])
		fib_qual.append(line[3])
	fib_qual=numpy.array(fib_qual)

	select_good = fib_qual=='GOOD'
	npeaks=numpy.sum(select_good)
	# find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks

	peaks_good = cut.findPeaks(threshold=threshold, npeaks=npeaks)
	expect_first = ref_pos[select_good][0]
	shift_peaks=peaks_good[0][0]-expect_first
	if expect_first>=5 and shift_peaks>5:
		idx = numpy.indices(cut._data.shape)[0]
		while ref_pos[select_good][-1]+shift_peaks+2>cut._data.shape[0]:
			last_fiber = idx[select_good][-1]
			select_good[last_fiber]=False
	npeaks = numpy.sum(select_good)
	peaks_good = cut.findPeaks(threshold=threshold, npeaks=npeaks)
	centers_good = cut.measurePeaks(peaks_good[0], method, init_sigma, threshold=0, max_diff=1.0)[0]
	peaks_ref = Spectrum1D(wave=fiber, data=ref_pos )

	shift_spec = Spectrum1D(wave=fiber[select_good], data=ref_pos[select_good]-centers_good)
	shift_spec.smoothPoly(order=-3, ref_base=fiber)
	centers = peaks_ref-shift_spec
	centers._data[select_good] = centers_good


	select_good_weak = numpy.logical_or(select_good, fib_qual=='WEAK')
	select_weak = fib_qual=='WEAK'
	npeaks=numpy.sum(select_good_weak)
	peaks_weak_good = cut.findPeaks(threshold=threshold_weak, npeaks=npeaks)
	centers_weak_good = cut.measurePeaks(peaks_weak_good[0], method, init_sigma, threshold=0, max_diff=1.0)[0]
	offset_weak = centers._data[select_good_weak]-centers_weak_good
	select_wrong = numpy.logical_not(numpy.logical_and(offset_weak>-0.5, offset_weak<0.5))
	offset_weak[select_wrong]=0
	centers._data[select_good_weak]= centers._data[select_good_weak]-offset_weak
	round_cent = numpy.round(centers._data).astype('int16') # round the subpixel peak positions to their nearest integer value
	file_out = open(out_peaks_file, 'w')
	select_bad = numpy.logical_not(select_good_weak)
	file_out.write('%i\n' %(column))
	for i in range(len(round_cent)):
		file_out.write('%i %i %e %i\n'%(i, round_cent[i], centers._data[i],  int(select_bad[i])))
	file_out.close()

	if verbose==1:
		# control plot for the peaks NEED TO BE REPLACE BY A PROPER VERSION AND POSSIBLE IMPLEMENTAION FOR A GUI
		print('%i Fibers found'%(len(centers._data)))
		pylab.plot(cut._data, '-k')
		pylab.plot(peaks_good[0],peaks_good[2] ,'or')
		if numpy.sum(select_weak)>0:
			select = numpy.logical_and(select_good_weak, select_weak)
			pylab.plot(peaks_weak_good[0][select[select_good_weak]],peaks_weak_good[2] [select[select_good_weak]],'ob')
		pylab.plot(centers._data, numpy.ones(len(centers._data))*2000.0, 'xg')
		pylab.show()

def findPeaksMaster2_drp(image, peaks_master, out_peaks_file, disp_axis='X', threshold='1500', threshold_weak='500', median_box='8', median_cross='1', slice='', method='gauss',  init_sigma='1.0', border='4', verbose='1'):

	threshold=float(threshold)
	threshold_weak=float(threshold_weak)
	border=int(border)
	median_box=int(median_box)
	median_cross=int(median_cross)
	init_sigma = float(init_sigma)
	verbose = int(verbose)

	# Load Image
	img = loadImage(image)

	# swap axes so that the dispersion axis goes along the x axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()

	# perform median filtering along the dispersion axis to clean cosmic rays
	img = img.medianImg((median_cross, median_box))

	# if no slice is given find the cross-dispersion cut with the highest signal
	if slice=='':
		median_cut=img.collapseImg(axis='y', mode='median') #median collapse of image along cross-dispersion axis
		maximum = median_cut.max() #get maximum value along dispersion axis
		column=maximum[2] # pixel position of maximum value
		cut = img.getSlice(column, axis='y') # extract this column from image
	else:
		column = int(slice) # convert column to integer value
		cut = img.getSlice(column, axis='y') # extract this column from image

	master_file = open(peaks_master, 'r')
	lines = master_file.readlines()
	fiber = numpy.zeros(len(lines), dtype=numpy.int16)
	pixel = numpy.zeros(len(lines), dtype=numpy.int16)
	ref_pos = numpy.zeros(len(lines), dtype=numpy.float32)
	fib_qual = []
	for i in range(len(lines)):
		line = lines[i].split()
        #print(line)
		fiber[i] = int(line[0])
		pixel[i] = int(line[1])
		ref_pos[i] = float(line[2])
		fib_qual.append(line[3])
	fib_qual=numpy.array(fib_qual)

	select_good = fib_qual=='GOOD'
	npeaks=numpy.sum(select_good)
	# find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks

	peaks_good=[]
	if numpy.max(cut._data)<threshold:
		threshold = numpy.max(cut._data)*0.8
	while len(peaks_good) != numpy.sum(select_good):
		(peaks_good,temp,peaks_flux) = cut.findPeaks(threshold=threshold, npeaks=0)
		if peaks_good[0]<border:
			peaks_good = peaks_good[1:]
			peaks_flux = peaks_flux[1:]
		if peaks_good[-1]>len(cut._data)-border-1:
			peaks_good = peaks_good[:-1]
			peaks_flux = peaks_flux[:-1]
		if peaks_good[0]>10:
			expect_first = ref_pos[select_good][0]
			shift_peaks=peaks_good[0]-expect_first
		elif peaks_good[-1]+10<=len(cut._data)-1:
			expect_last = ref_pos[select_good][-1]
			shift_peaks=peaks_good[-1]-expect_last
	
		#print peaks_good
		ref_pos_temp = ref_pos[:]+shift_peaks
		select_good = (fib_qual=='GOOD') & (numpy.rint(ref_pos+shift_peaks)>border) & (numpy.rint(ref_pos+shift_peaks)<len(cut._data)-border-1)
		#print (ref_pos+shift_peaks)[select_good],len(cut._data)
		if numpy.sum(select_good)>len(peaks_good):
			threshold = threshold/1.02
		elif numpy.sum(select_good)<len(peaks_good):
			threshold = threshold*1.05
      #print(threshold,numpy.sum(select_good),len(peaks_good),shift_peaks)
		#break
	centers_good = cut.measurePeaks(peaks_good, method, init_sigma, threshold=0, max_diff=1.0)[0]
	peaks_ref = Spectrum1D(wave=fiber, data=ref_pos )

	shift_spec = Spectrum1D(wave=fiber[select_good], data=ref_pos[select_good]-centers_good)
	shift_spec.smoothPoly(order=-3, ref_base=fiber)
	centers = peaks_ref-shift_spec
	centers._data[select_good] = centers_good


	round_cent = numpy.round(centers._data).astype('int16') # round the subpixel peak positions to their nearest integer value
	file_out = open(out_peaks_file, 'w')
	select_bad = numpy.logical_not(select_good)
	file_out.write('%i\n' %(column))
	for i in range(len(round_cent)):
		file_out.write('%i %i %e %i\n'%(i, round_cent[i], centers._data[i],  int(select_bad[i])))
	file_out.close()

	if verbose==1:
		# control plot for the peaks NEED TO BE REPLACE BY A PROPER VERSION AND POSSIBLE IMPLEMENTAION FOR A GUI
		print('%i Fibers found'%(len(centers._data)))
		pylab.plot(cut._data, '-k')
		pylab.plot(peaks_good,peaks_flux ,'or')
		pylab.plot(centers._data, numpy.ones(len(centers._data))*2000.0, 'xg')
		pylab.show()

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image", "peaks_file")
def tracePeaks_drp(in_image, peaks_file, trace_out, disp_axis='X', method='gauss', median_box='7', median_cross='1', steps='30', coadd='30', poly_disp='-6', init_sigma='1.0', threshold_peak='100.0', max_diff='2', verbose='1'):
	"""
			Traces the peaks of fibers along the dispersion axis. The peaks at a specific dispersion column had to be determined before.
			Two scheme of measuring the subpixel peak positionare available: A hyperbolic approximation or fitting a Gaussian profile to the brightest 3 pixels of a peak.
			In both cases the resulting fiber traces along the dispersion axis are smoothed by modelling it with a polynomial function.

			Parameters
			--------------
			image: string
					Name of the Continuum exposure FITS file  used to trace the fibre positions
			peaks_file : string
					Name of peaks file containing previously estimated peak position for a certain cross-disperion profile at a specific dispersion column
			trace_out: string
					Name of the  FITS file in which the trace mask will be stored
			disp_axis: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
			method: string, optional with default: 'gauss'
				Set the method to measure the peaks positions, either 'gauss' or 'hyperbolic'.
			median_box: string of integer, optional  with default: '7'
				Set a median box size for a median filtering in cross-dispersion direction (reduces artifiacts)
			steps : string of int, optional with default :'30'
				Steps in dispersions direction for which to measure the cross-dispersion fibre positions (saves times)
			coadd: string of integer, optional with default: '30'
				Coadd number of pixels in dispersion direction to increase the S/N of the data
			poly_disp: string of integer, optional with default: '-6'
				Order of the polynomial used to smooth the measured peak position along dispersion axis (positiv: normal polynomial, negative: Legandre polynomial)
			init_sigma: string of float, optional with default: '1.0'
				Initial guess for the width of the Gaussian profiles to measure the peak positions (only used in with method 'gauss')
			threshold_peak: string of float, optional  with default: '100.0'
				Minimum contrast between peak height and the adjacent continuuml counts to be considered as a good measurement
			max_diff: string of float, optional with default: '1.0'
				Maximum difference between the peak position of each fiber in adjacent measurements (steps) along  dispersion direction
			verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Examples
			----------------
            user:> lvmdrp image tracePeaks IMAGE.fits OUT_PEAKS.txt x method=gauss steps=40 coadd=20 smooth_poly=-8
	"""

	# convert all parameters to proper type
	coadd=int(coadd)
	poly_disp=int(poly_disp)
	steps=int(steps)
	median_box=int(median_box)
	median_cross=int(median_cross)
	threshold_peak=float(threshold_peak)
	max_diff = float(max_diff)
	init_sigma = float(init_sigma)

	# load continuum image  from file
	img = loadImage(in_image)


	# orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()

	dim = img.getDim()
	# perform median filtering along the dispersion axis to clean cosmic rays
	img = img.medianImg((median_cross, median_box))

	# coadd images along the dispersion axis to increase the S/N of the peaks
	if coadd!=0:
		coadd_kernel = numpy.ones((1, coadd), dtype='uint8') # create convolution kernel array for coaddition
		img = img.convolveImg(coadd_kernel) # perform convolution to coadd the signal
		threshold=threshold_peak*coadd # adjust the minimum contrast threshold for the peaks

	# load the initial positions of the fibers at a certain column NEED TO BE REPLACED WITH XML handling
	file_in_peaks = open(peaks_file, 'r') # load file
	lines = file_in_peaks.readlines()   # read lines
	column = int(lines[0]) #read the pixel column of the initially measured fiber positions as a starting value
	fibers = len(lines)-1  # number of fibers
	positions = numpy.zeros(fibers, dtype=numpy.float32) # empty array to store positions
	bad_fibers = numpy.zeros(fibers, dtype='bool') # empty array to store positions
	for i in range(1, fibers+1):
		line = lines[i].split()
		positions[i-1]=float(line[2])
		bad_fibers[i-1] = bool(int(line[3]))
	good_fibers= numpy.logical_not(bad_fibers)
	# choose between  methods to measure the subpixel peak
	if method=='hyperbolic':
		steps = 1
	elif method=='gauss':
		steps = int(steps)

	# create empty trace mask for the image
	trace = TraceMask()
	trace.createEmpty(data_dim=(fibers, dim[1]), mask_dim=(fibers, dim[1]))
	trace.setFibers(fibers)
	# add the positions of the previous identified peaks
	trace.setSlice(column, axis='y',  data=positions, mask=numpy.zeros(len(positions), dtype='bool'))

	# select cross-dispersion slice for the measurements of the peaks
	first = numpy.arange(column-1, -1, -1)
	select_first = first%steps==0
	second = numpy.arange(column+1, dim[1], 1)
	select_second = second%steps==0
	nslice = numpy.sum(select_first)+numpy.sum(select_second)
	m=1
	# iterate towards index 0 along dispersion axis
	if verbose=='1':
		print('Trace peaks along dispersion axis:')
	for i in first[select_first]:
		if verbose=='1':
			sys.stdout.write('Processing....%.0f%%\r'%(m/float(nslice)*100))
			sys.stdout.flush()
		cut_iter = img.getSlice(i, axis='y') # extract cross-dispersion slice
		# infer pixel position of the previous slice
		if i==first[select_first][0]:
			pix = numpy.round(trace.getData()[0][:, column]).astype('int16')
		else:
			pix = numpy.round(trace.getData()[0][:, i+steps]).astype('int16')

		#measure the peaks for the slice and store it in the trace
		centers = cut_iter.measurePeaks(pix, method,  init_sigma, threshold=threshold, max_diff=float(max_diff))
		if numpy.sum(bad_fibers)>0:
			diff = Spectrum1D(wave=positions,  data=(centers[0]-positions), mask=bad_fibers)
			diff.smoothPoly(-1, ref_base=positions)
			centers[0][bad_fibers]=diff._data[bad_fibers]+positions[bad_fibers]
			centers[1][bad_fibers]=False
		trace.setSlice(i, axis='y', data = centers[0], mask = centers[1])
		m+=1


	# iterate towards the last index along dispersion axis
	for i in second[select_second]:
		if verbose=='1':
			sys.stdout.write('Processing....%.0f%%\r'%(m/float(nslice)*100))
			sys.stdout.flush()
		cut_iter = img.getSlice(i, axis='y')# extract cross-dispersion slice
		# infer pixel position of the previous slice
		if i==second[select_second][0]:
			pix = numpy.round(trace.getData()[0][:, column]).astype('int16')
		else:
			pix = numpy.round(trace.getData()[0][:, i-steps]).astype('int16')

		#measure the peaks for the slice and store it in the trace
		centers = cut_iter.measurePeaks(pix, method, init_sigma, threshold=threshold, max_diff=float(max_diff))
		if numpy.sum(bad_fibers)>0:
			diff = Spectrum1D(wave=positions,  data=(centers[0]-positions), mask=bad_fibers)
			diff.smoothPoly(-1, ref_base=positions)
			centers[0][bad_fibers]=diff._data[bad_fibers]+positions[bad_fibers]
			centers[1][bad_fibers]=False
		trace.setSlice(i, axis='y', data = centers[0], mask = centers[1])
		m+=1

	# smooth all trace by a polynomial
	trace.smoothTracePoly(poly_disp)

	for i in range(fibers):
		if bad_fibers[i]==False:
			trace._mask[i, :] = False
		else:
			trace._mask[i, :] = True

	#This part is not working correctly and therefore taken out at the moment
	## smooth all traces assuming that their distances are changing smoothly along dispersion axis
	##if poly_cross!='':
	##    poly_cross = numpy.array(poly_cross.split(',')).astype('int16')
	##    trace.smoothTraceDist(column, poly_cross=poly_cross, poly_disp=poly_disp)

	trace.writeFitsData(trace_out)

def glueCCDFrames_drp(images, out_image, boundary_x, boundary_y, positions, orientation, subtract_overscan='1',compute_error='1', gain='', rdnoise=''):
	"""
			Glue CCD subimages of different amplifiers  to a full science CCD images. The orientations of the sub images are taken into account as well as their overscan regions.
			A Poission error image can be automatically computed during this process. This requires that the GAIN and the Read-Out Noise are stored as header keywords in each
			subimage.

			Parameters
			--------------
			images: string
					Comma-separated names of the FITS images containing the subimage to be combined
			out_image: string
					Name of the FITS file  in which the combined image will be stored
			boundary_x : string of two comma-separated integers
					Pixel boundaries of the subimages EXCLUDING the overscan regions along x axis (first pixel has index 1)
			boundary_y : string of two comma-separated integers
					Pixel boundaries of the subimages EXCLUDING the overscan regionsalong y axis (first pixel has index 1)
			positions : string of two comma-separated  integer digits,
					Describes the position of each sub image in colum/row format where the first digit describes the row and the second the column position.
					'00' would correspond to the lower left corner in the combined CCD frame
			orientation: comma-separated strings
					Describes how each subimage should be oriented before place into the glued CCD frame. Possible options are: 'S','T','X','Y','90','180'', and 270'
					Their meaning are:
					'S' : orientation is unchanged
					'T' : the x and y axes are swapped
					'X' : mirrored along the x axis
					'Y' : mirrored along the y axis
					'90' : rotated by 90 degrees
					'180' : rotated by 180 degrees
					'270' : rotated by 270 degrees
			subtract_overscan : string of integer ('0' or '1'), optional  with default: '1'
					Should the median value of the overscan region be subtracted from the subimage before glueing, '1' - Yes, '0' - No
			compute_error : string of integer ('0' or '1'), optional  with default: '1'
					Should the Poisson error included into the second extension, '1' - Yes, '0' - No
			gain : string, optional with default :''
					Name of the FITS Header keyword for the gain value of the CCD, will be multiplied
			rdnoise: string, optional with default: ''
					Name of the FITS Header keyword for the read out noise value

			Examples
			----------------
			user:>  lvmdrp image glueCCDFrame FRAME1.fits, FRAME2.fits, FRAME3.fits, FRAME4.fits  FULLFRAME.fits  50,800 1,900  00,10,01,11 X,90,Y,180 gain='GAIN'
			"""
	# convert input parameters to proper type
	list_imgs= images.split(',')
	bound_x = boundary_x.split(',')
	bound_y = boundary_y.split(',')
	orient = orientation.split(',')
	pos = positions.split(',')
	subtract_overscan = bool(int(subtract_overscan))
	compute_error = bool(int(compute_error))
	# create empty lists
	imgs = [] # list of images
	gains = [] # list of gains
	rdnoises=[] # list of read-out noises
	bias = []  # list of biasses

	for i in list_imgs:
		#load subimages from disc and append them to a list
		img = loadImage(i, extension_data=0)
		imgs.append(img)
		if gain!='':
			# get gain value
			try:
				gains.append(img.getHdrValue(gain))
			except KeyError:
				gains.append(float(gain))
		if rdnoise!='':
			# get read out noise value
			try:
				rdnoises.append(img.getHdrValue(rdnoise))
			except KeyError:
				rdnoises.append(float(rdnoise))
		else:
			rdnoises.append(0.0)

	for i in range(len(list_imgs)):
		# append the bias from the overscane region
		bias.append(imgs[i].cutOverscan(bound_x, bound_y,subtract_overscan))
		# multiplication with the gain factor
		if gain=='':
			mult=1.0
		else:
			mult=gains[i]
		imgs[i]=imgs[i]*mult

		# change orientation of subimages
		imgs[i].orientImage(orient[i])
		if compute_error:
			imgs[i].computePoissonError(rdnoise=rdnoises[i])

	# create glued image
	full_img = glueImages(imgs, pos)

	# adjust FITS header information
	full_img.removeHdrEntries(['GAIN','RDNOISE', 'COMMENT',''])
	# add gain keywords for the different subimages (CDDs/Amplifies)
	if gain!='':
		for i in range(len(imgs)):
			full_img.setHdrValue('hierarch AMP%i GAIN'%(i+1), gains[i], 'Gain value of CCD amplifier %i'%(i+1))
	# add read-out noise keywords for the different subimages (CDDs/Amplifies)
	if rdnoise!='':
		for i in range(len(imgs)):
			full_img.setHdrValue('hierarch AMP%i RDNOISE'%(i+1), rdnoises[i], 'Read-out noise of CCD amplifier %i'%(i+1))
	# add bias of overscan region for the different subimages (CDDs/Amplifies)
	for i in range(len(imgs)):
		if subtract_overscan:
			full_img.setHdrValue('hierarch AMP%i OVERSCAN'%(i+1), bias[i], 'Overscan median (bias) of CCD amplifier %i'%(i+1))
	##full_img.setHeader(header=header) # set the modified FITS Header
	#write out FITS file
	if compute_error:
		extension_error=1
	else:
		extension_error=None
	full_img.writeFitsData(out_image)

def combineImages_drp(images, out_image, method='median', k='3.0'):
	# convert input parameters to proper type
	list_imgs= images.split(',')
	if len(list_imgs)==1:
			file_list = open(images, 'r')
			list_imgs=file_list.readlines()
			file_list.close()
	k = float(k)
	imgs=[]
	for i in list_imgs:
		#load subimages from disc and append them to a list
		imgs.append(loadImage(i.replace('\n','')))

	combined_img = combineImages(imgs, method=method, k=k)
	#write out FITS file
	combined_img.writeFitsData(out_image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image", "trace")
def subtractStraylight_drp(in_image, trace, stray_image, clean_image, disp_axis='X',  aperture='7', poly_cross='4', smooth_disp='5', smooth_gauss='10.0', parallel='auto'):
	"""
			Subtracts a diffuse background signal (stray light) from the raw data. It uses the regions between fiber to estimate the stray light signal and
			smoothes the result by a polyon in cross-disperion direction and afterwards a wide 2D Gaussian filter to reduce the introduction of low frequency noise.

			Parameters
			--------------
			image: string
					Name of the FITS image from which the stray light should be subtracted
			trace: string
					Name of the  FITS file with the trace mask of the fibers
			stray_image: string
					Name of the FITS file in which the pure straylight image is stored
			clean_image: string
					Name of the FITS file in which the straylight subtracted image is stored
			disp_axis: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
			aperture: string of integer, optional  with default: '7'
					Size of the aperture around each fiber in cross-disperion direction assumed to contain signal from fibers
			poly_cross: string of integer, optional with default: '4'
				Order of the polynomial used to interpolate the background signal in cross-dispersion direction (positiv: normal polynomial, negativ: Legandre polynomial)
			smooth_gauss : string of float, optional with default :'10.0'
				Width of the 2D Gaussian filter to smooth the measured background signal

			Examples
			----------------
            user:> lvmdrp image subtractStrylight IMAGE.fits TRACE.fits CLEAN.fits x aperture=9 poly_cross=6 smooth_gauss=20.0
	"""
	# convert input parameters to proper type
	aperture = int(aperture)
	poly_cross=int(poly_cross)
	smooth_gauss = float(smooth_gauss)
	smooth_disp = int(smooth_disp)


	# load image data
	img = loadImage(in_image)


	# orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()
    
	# load trace mask
	trace_mask = TraceMask()
	trace_mask.loadFitsData(trace, extension_data=0)
   # trace_mask.clipTrace(img._dim[0])


    #if img._mask is not None:
    #    img._data[img._mask==1] = numpy.nan

    # smooth image along dispersion axis with a median filter excluded NaN values bas
	img_median = img.medianImg((1, smooth_disp),use_mask=True)

	# mask regions around each fiber within a given cross-dispersion aperture
	img_median.maskFiberTraces(trace_mask, aperture=aperture, parallel=parallel)

	# fit the signal in unmaksed areas along cross-dispersion axis independently with a polynom of a given order
	img_fit = img_median.fitPoly(order=poly_cross, plot=-1)

	# smooth the results by 2D Gaussian filter of given with (cross- and dispersion axis have equal width)
	img_smooth =img_fit.convolveGaussImg(smooth_gauss, smooth_gauss)

	# subtract smoothed background signal from origianal image
	img = loadImage(in_image)

	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()
		
	img_out = img-img_smooth
	##img_out = img-img_fit


	#restore original orientation of image
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img_out.swapaxes()
		img_smooth.swapaxes()



	# include header and write out file
	img_out.setHeader(header=img.getHeader())
	img_out.writeFitsData(clean_image)
	img_smooth.writeFitsData(stray_image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image", "trace")
def traceFWHM_drp(in_image, trace, fwhm_out, disp_axis='X', blocks='20', steps='100', coadd='10', poly_disp='5', threshold_flux='50.0', init_fwhm='2.0', clip='', parallel='auto'):
	"""
			Measures the FWHM of the cross-dispersion fiber profile across the CCD.  It assumes that the profiles have a Gaussian shape and that the width  is CONSTANT for
			a BLOCK of fibers in cross-dispersion direction.  If the FITS image contains an extension with the error, the error frame will be taken into account in the Gaussian fitting.
			To increase the speed only the cross-dispersion profiles at certain position along the dispersion axis with a certain distance (steps)
			in pixels are modelled. The FWHM are then extrapolate by fitting a polynomial of given order along the dispersion axis.

			Parameters
			--------------
			image: string
					Name of the Continuum FITS image from which the fiber profile width should be estimate.
			trace: string
					Name of the  FITS file representing the trace mask of the fibers
			fwhm_out: string
					Name of the FITS file in which the FWHM trace image will be stored
			disp_axis: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
			blocks: string of integer, optional  with default: '20'
					Number of fiber blocks that are modelled simultaneously with the same FWHM in cross-dispersion direction.
					The actual number of fibers per block is roughly the total number of fibers divided by the number of blocks.
			steps : string of int, optional with default :'100'
					Steps in dispersions direction columns to measure the cross-dispersion fibre positions
			coadd: string of integer, optional with default: '10'
					Coadd number of pixels in dispersion direction to increase the S/N of the data
			poly_disp: string of integer, optional with default: '5'
					Order of the polynomial used to extrapolate the FWHM  values along dispersion direction for each block
					(positiv: normal polynomial, negativ: Legandre polynomial)
			threshold_flux: sting of float, optional with default: '50.0'
					Minimum integrated counts for a valid fiber per dispersion element
			init_fwhm: string of float, optional with default: '2.0'
					Initial guess of the cross-dispersion fiber FWHM
			clip: string of two comma separated floats, optional with default: ''
					Minimum and maximum number of FWHM in the resulting FWHM trace image. If some value are below or above
					the given limits they are replaced by those limits.
			parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
					Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
					for the given system is used.

			Examples
			----------------
            user:> lvmdrp image traceFWHM IMAGE.fits TRACE.fits FWHM.fits x blocks=32 steps=50 poly_disp=20 clip=2,6 parallel=2
	"""

	# convert input parameters to proper type
	steps=int(steps)
	blocks=int(blocks)
	poly_disp = int(poly_disp)
	init_fwhm = float(init_fwhm)
	coadd = int(coadd)
	threshold_flux = float(threshold_flux)
	if clip!='':
		clip=clip.split(',')
		clip=[float(clip[0]), float(clip[1])]
	else:
		clip=None

	# load image data
	img = loadImage(in_image)

	# orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()
	dim=img.getDim()

	# coadd images along the dispersion axis to increase the S/N of the peaks
	if coadd!=0:
		coadd_kernel = numpy.ones((1, coadd), dtype='uint8') # create convolution kernel array for coaddition
		img = img.convolveImg(coadd_kernel) # perform convolution to coadd the signal
		threshold_flux = threshold_flux*coadd #adjust threshold flux to the coadded signal

	# load trace
	trace_mask = TraceMask()
	trace_mask.loadFitsData(trace)

	# create a trace mask for the image
	traceFWHM = TraceMask()

	# define the cross-dispersion slices to be modelled with Gaussian profiles
	axis = numpy.arange(dim[1])
	select_steps = axis%steps==0
	select_steps[-1] = True

	if parallel=='auto':
		fragments = multiprocessing.cpu_count()
	else:
		fragments = int(parallel)
	if fragments>1:
		split_img = img.split(fragments, axis='X')
		split_trace = trace_mask.split(fragments, axis='X')
		pool = Pool()
		threads=[]
		fwhm=[]
		mask=[]
		select = numpy.array_split(select_steps, fragments)
		for i in range(fragments):
			threads.append(pool.apply_async(split_img[i].traceFWHM, (select[i], split_trace[i], blocks,  init_fwhm, threshold_flux,dim[0])))

		for i in range(fragments):
			result = threads[i].get()
			fwhm.append(result[0])
			mask.append(result[1])

		pool.close()
		pool.join()
		traceFWHM = TraceMask(data=numpy.concatenate(fwhm, axis=1), mask=numpy.concatenate(mask, axis=1))
	else:
		result=img.traceFWHM(select_steps, trace_mask, blocks, init_fwhm, threshold_flux,max_pix=dim[0])
		traceFWHM = TraceMask(data=result[0], mask=result[1])
	#traceFWHM = img.traceFWHM()

	#smooth the FWHM trace with a polynomial fit along dispersion axis (uncertain pixels are not used)
	traceFWHM.smoothTracePoly(poly_disp, clip=clip)

	# write out FWHM trace to FITS file
	traceFWHM.writeFitsData(fwhm_out)

def offsetTrace_drp(image, trace, disp, lines, logfile,  blocks='15', disp_axis='X',  init_offset='0.0', size='20'):
	"""
			Measures the offset in the fiber trace in  cross-dispersion direction in an object raw frame compared to the traces measured from a continuum lamp frame.
			The measurements are stored in a ASCII logfile for futher processing and usage.

			Parameters
			--------------
			image: string
					Name of the target FITS image which should be test for an offset in the tracing
			trace: string
					Name of the  RSS FITS file representing the trace mask of the fibers
			disp: string
					Name of the  RSS FITS file representing containing the wavelength solution for each pixel
			lines: comma separeted string of floats
					Wavelength sequence of bright lines that can be used to compare the tracing
			logfile: string
					Name of the output log file in which the measurements are stored in ASCII format
			blocks: string of integer, optional  with default: '20'
					Number of fiber blocks that are modelled simultaneously with the same FWHM but with a variable offset compare to
					the original central positions. The actual number of fibers per block is roughly the total number of fibers divided by
					the number of blocks.
			disp_axis: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
			init_offset: string of float, optional with default: '0.0'
				Initial guess of the cross-dispersion fiber trace offset in pixels
			size:  string of int (>0), optional with default: '20'
				Number of pixels being coadd in dispersion direction centered on the wavelength corresponding to the wavelengthes
				of the input lines

			Examples
			----------------
            user:> lvmdrp image offsetTrace IMAGE.fits TRACE.fits DISP.fits  blocks=32 size=30
	"""
	lines = lines.split(',')
	size = float(size)
	blocks = int(blocks)
	init_offset=float(init_offset)
	img = Image()
	img.loadFitsData(image)
	# orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()

	trace_mask = TraceMask()
	trace_mask.loadFitsData(trace)

	dispersion_sol = FiberRows()
	dispersion_sol.loadFitsData(disp)

	#read log file to guess offset position
	try:
		log = open(logfile, 'r')
		log_lines = log.readlines()
		l=0
		offset_files=[]
		while l<len(log_lines):
			if len(log_lines[l].split())==1:
				l+=1
				offsets=[]
			else:
				offsets.append(numpy.median(numpy.array(log_lines[l+2].split()[1:]).astype('float32')))
				l+=3
		log.close()
	except IOError:
		offsets=[]
		for i in range(len(lines)):
			offsets.append(init_offset)

	log = open(logfile, 'a')
	log.write('%s\n'%(image))
	off_trace_all =[]
	for i in range(len(lines)):
		wave_line = float(lines[i])
		distance = numpy.abs(dispersion_sol._data-wave_line)
		central_pix = numpy.argmin(distance, 1)
		central_pos = trace_mask._data[numpy.arange(len(central_pix)), central_pix]
		fit = numpy.polyfit(central_pos, central_pix, 4)
		poly = numpy.polyval(fit, numpy.arange(img._data.shape[0]))
		line_pos = numpy.rint(poly).astype('int16')
		collapsed_data = numpy.zeros(len(line_pos), dtype=numpy.float32)

		for j in range(len(line_pos)):
			collapsed_data[j] = numpy.sum(img._data[j, line_pos[j]-size:line_pos[j]+size])
		if img._error is not None:
			collapsed_error = numpy.zeros(len(line_pos), dtype=numpy.float32)
			for j in range(len(line_pos)):
				collapsed_error[j] = numpy.sqrt(numpy.sum(img._error[j, line_pos[j]-size:line_pos[j]+size]**2))
		else:
			collapsed_error = None
		trace_spec = Spectrum1D(wave = numpy.arange(len(collapsed_data)), data=collapsed_data, error=collapsed_error)
		if trace_mask._mask is not None:
			mask=trace_mask._mask[numpy.arange(len(central_pix)), central_pix]
		else:
			mask=None
		out=trace_spec.measureOffsetPeaks(trace_mask._data[numpy.arange(len(central_pix)), central_pix], mask, blocks, init_offset=offsets[i], plot=-1)
		off_trace_all.append(out[0])
		string_x ="%.3f"%(wave_line)
		string_y ="%.3f"%(wave_line)
		string_pix ="%.3f"%(wave_line)
		block_line_pos = numpy.array_split(line_pos, blocks)
		for j in range(len(out[0])):
			string_x+=" %.3f"%(out[1][j])
			string_y+=" %.3f"%(out[0][j])
			string_pix+=" %.3f"%(numpy.median(block_line_pos[j]))
		log.write(string_x+"\n")
		log.write(string_pix+"\n")
		log.write(string_y+"\n")
	off_trace_median= numpy.median(numpy.array(off_trace_all))
	off_trace_rms = numpy.std(numpy.array(off_trace_all))
	off_trace_rms = '%.4f' % off_trace_rms if numpy.isfinite(off_trace_rms) else 'NAN'
	img.setHdrValue('hierarch PIPE FLEX YOFF', float('%.4f'%off_trace_median)*-1, 'flexure offset in y-direction')
	img.setHdrValue('hierarch PIPE FLEX YRMS', off_trace_rms, 'flexure rms in y-direction')
	img.writeFitsHeader(image)
	log.close()

def offsetTrace2_drp(image, trace, trace_fwhm, disp, lines, logfile,  blocks='15', disp_axis='X', min_offset='-2',max_offset='2',step_offset='0.1', size='20'):
	"""
			Measures the offset in the fiber trace in  cross-dispersion direction in an object raw frame compared to the traces measured from a continuum lamp frame.
			The measurements are stored in a ASCII logfile for futher processing and usage.

			Parameters
			--------------
			image: string
					Name of the target FITS image which should be test for an offset in the tracing
			trace: string
					Name of the  RSS FITS file representing the trace mask of the fibers
			disp: string
					Name of the  RSS FITS file representing containing the wavelength solution for each pixel
			lines: comma separeted string of floats
					Wavelength sequence of bright lines that can be used to compare the tracing
			logfile: string
					Name of the output log file in which the measurements are stored in ASCII format
			blocks: string of integer, optional  with default: '20'
					Number of fiber blocks that are modelled simultaneously with the same FWHM but with a variable offset compare to
					the original central positions. The actual number of fibers per block is roughly the total number of fibers divided by
					the number of blocks.
			disp_axis: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
			init_offset: string of float, optional with default: '0.0'
				Initial guess of the cross-dispersion fiber trace offset in pixels
			size:  string of int (>0), optional with default: '20'
				Number of pixels being coadd in dispersion direction centered on the wavelength corresponding to the wavelengthes
				of the input lines

			Examples
			----------------
			user:> lvmdrp image offsetTrace IMAGE.fits TRACE.fits DISP.fits  blocks=32 size=30
	"""
	lines = lines.split(',')
	size = int(size)
	blocks = int(blocks)
	min_offset = float(min_offset)
	max_offset = float(max_offset)
	step_offset = float(step_offset)

	img = Image()
	img.loadFitsData(image)
	# orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()

	trace_mask = TraceMask()
	trace_mask.loadFitsData(trace)

	trace_fwhm_mask = TraceMask()
	trace_fwhm_mask.loadFitsData(trace_fwhm)

	dispersion_sol = FiberRows()
	dispersion_sol.loadFitsData(disp)

	log = open(logfile, 'a')
	log.write('%s\n'%(image))
	off_trace_all =[]
	for i in range(len(lines)):
		wave_line = float(lines[i])
		distance = numpy.abs(dispersion_sol._data-wave_line)
		central_pix = numpy.argmin(distance, 1)
		central_pos = trace_mask._data[numpy.arange(len(central_pix)), central_pix]
		central_fwhm = trace_fwhm_mask._data[numpy.arange(len(central_pix)), central_pix]
		fit = numpy.polyfit(central_pos, central_pix, 4)
		poly = numpy.polyval(fit, numpy.arange(img._data.shape[0]))
		line_pos = numpy.rint(poly).astype('int16')
		collapsed_data = numpy.zeros(len(line_pos), dtype=numpy.float32)

		for j in range(len(line_pos)):
			collapsed_data[j] = numpy.sum(img._data[j, line_pos[j]-size:line_pos[j]+size])
		if img._error is not None:
			collapsed_error = numpy.zeros(len(line_pos), dtype=numpy.float32)
			for j in range(len(line_pos)):
				collapsed_error[j] = numpy.sqrt(numpy.sum(img._error[j, line_pos[j]-size:line_pos[j]+size]**2))
		else:
			collapsed_error = None
		trace_spec = Spectrum1D(wave = numpy.arange(len(collapsed_data)), data=collapsed_data, error=collapsed_error)
		if trace_mask._mask is not None:
			mask=trace_mask._mask[numpy.arange(len(central_pix)), central_pix]
		else:
			mask=None
		out=trace_spec.measureOffsetPeaks2(central_pos, mask, central_fwhm, blocks, min_offset, max_offset, step_offset, plot=-1)
		off_trace_all.append(out[0]*-1)
		string_x ="%.3f"%(wave_line)
		string_y ="%.3f"%(wave_line)
		string_pix ="%.3f"%(wave_line)
		block_line_pos = numpy.array_split(line_pos, blocks)
		for j in range(len(out[0])):
			string_x+=" %.3f"%(out[1][j])
			string_y+=" %.3f"%(out[0][j]*-1)
			string_pix+=" %.3f"%(numpy.median(block_line_pos[j]))
		log.write(string_x+"\n")
		log.write(string_pix+"\n")
		log.write(string_y+"\n")

	off_trace_median= numpy.median(numpy.array(off_trace_all))
	off_trace_rms = numpy.std(numpy.array(off_trace_all))
	img.setHdrValue('hierarch PIPE FLEX YOFF', float('%.4f'%off_trace_median)*-1, 'flexure offset in y-direction')
	img.setHdrValue('hierarch PIPE FLEX YRMS', float('%.4f'%off_trace_rms), 'flexure rms in y-direction')
	img.writeFitsHeader(image)
	log.close()

# TODO: suggestion from Oleg: test a voigt profile for the flux extraction
# it might be better in dealing with cross-talk
# TODO:
# * define lvm-frame ancillary product to replace for out_rss
# * set in_arc = lvm(M)Arc
@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image", "in_arc")
def extractSpec_drp(in_image, out_rss, in_arc,  method='optimal',  aperture='7', fwhm='2.5', disp_axis='X',  replace_error='1e10', plot='-1', parallel='auto'):
	"""
			Extracts the flux for each fiber along the dispersion direction which is written into an RSS FITS file format.
			Either a simple aperture or an optimal extraction scheme may be used.
			In the optimal extraction scheme each cross-dispersion profile is fitted with independent Gaussian for each fiber where
			the position and fwhm is fixed according to the input parameters. This allows for a linear fitting scheme where only the fluxes per fiber
			are the only free parameter.

			Parameters
			--------------
			image: string
					Name of the Continuum FITS image from which the fiber profile width should be estimate.
			trace: string
					Name of the  FITS file representing the trace mask of the fibers
			out_rss: string
					Name of the extracted RSS FITS file
			method: string, optional with default: 'optimal'
					Available methods are either
					1. 'optimal': using Gaussian profile fitting to extract the flux. The fwhm parameter needs to be set properly
					2. 'aperture': simple aperture extraction. The aperture parameter needs to be set as desired
			aperture: string of integer (>0), optional with default: '7'
					Size of the aperture around the peak position in cross-dispersion direction as used to integrate the flux.
					Only used if method is set to 'aperture' otherwise this parameter is ignored.
			fwhm: string or string of float, optional with default: '2.5'
					Set the FWHM in case of the Gaussian profile fitting in optimal method. Either a signle value or the name of a fits file containing
					a spatially resolved FWHM map is provided. Only used if method is set to 'optimal', otherwise this parameter is ignored.
			disp_axis: string of float, optional  with default: 'X'
					Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
			parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
				Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
				for the given system is used.

			Examples
			----------------
			user:> lvmdrp image extractSpec IMAGE.fits TRACE.fits RSS.fits optimal fwhm=FWHM.fits
	"""


	aperture = int(aperture)
	replace_error = float(replace_error)
	img = loadImage(in_image)
	plot=int(plot)

	# orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
	if disp_axis=='X' or disp_axis=='x':
		pass
	elif disp_axis=='Y' or disp_axis=='y':
		img.swapaxes()


	trace_mask = TraceMask()
	trace_mask.loadFitsData(in_arc)
	trace_fwhm = TraceMask()

	if method=='optimal':
		try:
			fwhm = float(fwhm)
			trace_fwhm.setData(data=numpy.ones(trace_mask._data.shape)*fwhm)
		except ValueError:
			trace_fwhm.loadFitsData(fwhm, extension_data=0)
		if parallel=='auto':
			fragments = multiprocessing.cpu_count()
		else:
			fragments = int(parallel)
		if fragments>1:
			split_img = img.split(fragments)
			split_trace = trace_mask.split(fragments)
			split_fwhm = trace_fwhm.split(fragments)
			pool = Pool()
			threads=[]
			data=[]
			error=[]
			mask=[]
			for i in range(fragments):
				threads.append(pool.apply_async(split_img[i].extractSpecOptimal, (split_trace[i], split_fwhm[i])))
			for i in range(fragments):
				result = threads[i].get()
				data.append(result[0])
				error.append(result[1])
				mask.append(result[2])
			pool.close()
			pool.join()
			data = numpy.concatenate(data, axis=1)
			if error[0] is not None:
				error = numpy.concatenate(error, axis=1)
			else:
				error = None
			if mask[0] is not None:
				mask = numpy.concatenate(mask, axis=1)
			else:
				mask = None
		else:
			(data, error, mask) = img.extractSpecOptimal(trace_mask, trace_fwhm, plot=plot)
	elif method=='aperture':
		(data, error, mask) = img.extractSpecAperture(trace_mask, aperture)

	if error is not None:
		error[mask]=replace_error
	rss= FiberRows(data=data, mask=mask, error=error, header = img.getHeader())
	rss.setHdrValue('NAXIS2',  data.shape[0])
	rss.setHdrValue('NAXIS1',  data.shape[1])
	rss.setHdrValue('DISPAXIS',  1)
	if method=='optimal':
		rss.setHdrValue('hierarch PIPE CDISP FWHM MIN',numpy.min(trace_fwhm._data[trace_mask._mask==False], initial=0))
		rss.setHdrValue('hierarch PIPE CDISP FWHM MAX',numpy.max(trace_fwhm._data[trace_mask._mask==False], initial=0))
		rss.setHdrValue('hierarch PIPE CDISP FWHM AVG',numpy.mean(trace_fwhm._data[trace_mask._mask==False]) if data.size != 0 else 0)
		rss.setHdrValue('hierarch PIPE CDISP FWHM MED',numpy.median(trace_fwhm._data[trace_mask._mask==False]) if data.size != 0 else 0)
		rss.setHdrValue('hierarch PIPE CDISP FWHM SIG',numpy.std(trace_fwhm._data[trace_mask._mask==False]) if data.size != 0 else 0)
	rss.writeFitsData(out_rss)

def calibrateSDSSImage_drp(file_in, file_out, field_file):
	"""
			Converts the original SDSS image as retrieved from the DR into a photometrically calibrated image
			in untis of ??. Photometric information are taken from the corresponding SDSS field FITS file.

			Parameters
			--------------
			file_in: string
					Name of the original SDSS image file
			file_out: string
					Name of the  photometric calibrated FITS output file in units of ????
			field_file: string
					Name of the corresponding SDSS field FITS file containing the photometric information

			Examples
			----------------
            user:> lvmdrp image calibrateSDSSImage fpC-001453-g4-0030.fit.gz SDSS_calib.fits drField-001453-4-40-0030.fit
	"""
	image = loadImage(file_in)
	calImage = image.calibrateSDSS(field_file)
	calImage.writeFitsData(file_out)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image", "in_bias")
def subtractBias_drp(in_image, out_image, in_bias, compute_error='1', boundary_x='', boundary_y='', gain='', rdnoise='', subtract_light='0'):
	subtract_light= bool(int(subtract_light))
	compute_error = bool(int(compute_error))
	image = loadImage(in_image)
    #print('image',image._data)
	bias_frame = loadImage(in_bias)
    #print('bias',bias_frame._data)
    
	clean=image-bias_frame
    #print('clean',clean._data)
	if boundary_x!='':
		bound_x = boundary_x.split(',')
	else:
		bound_x=[1, clean._dim[1]]
	if boundary_y!='':
		bound_y = boundary_y.split(',')
	else:
		bound_y=[1, clean._dim[0]]

	if gain!='':
		# get gain value
		try:
			gain = image.getHdrValue(gain)
		except KeyError:
			gain=float(gain)
		clean=clean*gain
        #print(clean._dim)

	if rdnoise!='':
		# get gain value
		try:
			rdnoise = image.getHdrValue(rdnoise)
		except KeyError:
			rdnoise=float(rdnoise)

	if compute_error:
		clean.computePoissonError(rdnoise=rdnoise)

	if boundary_x!='' or boundary_y!='':
		straylight=clean.cutOverscan(bound_x, bound_y,subtract_light)
        #print(straylight)


	clean.writeFitsData(out_image)

def reprojectRSS_drp(stray, trace, fwhm_cross, fwhm_spect, wave, flux, sim_fwhm=0.5, method="linear"):
	"""
			Historic task used for debugging of the the extraction routine...
	"""
	# label for outputs
	out_path = os.path.dirname(flux) or "./"
	out_name = os.path.basename(flux).replace(".fits", "")

	# read stray light map
	trace_stray = TraceMask()
	trace_stray.loadFitsData(stray, extension_data=0)
	# read trace
	trace_mask = TraceMask()
	trace_mask.loadFitsData(trace, extension_data=0)
	# read spatial fwhm
	trace_fwhm = TraceMask()
	trace_fwhm.loadFitsData(fwhm_cross, extension_data=0)
	# read spectral fwhm (lsf)
	spect_fwhm = TraceMask()
	spect_fwhm.loadFitsData(fwhm_spect, extension_data=0)
	# read wavelength solution
	trace_wave = TraceMask()
	trace_wave.loadFitsData(wave, extension_data=0)
	# read simulated RSS
	rss_flux = RSS()
	rss_flux.loadFitsData(flux)

	# TODO: implement interpolation in the cross-dispersion direction for:
	# 	- trace_fwhm
	# 	- spect_fwhm
	# 	- trace_wave
	# 	- trace_mask
	if trace_mask._data.shape[1] != rss_flux._data.shape[1]:
		trace_fwhm_res = numpy.zeros((rss_flux._data.shape[0], trace_mask._data.shape[1]))
		spect_fwhm_res = numpy.zeros((rss_flux._data.shape[0], trace_mask._data.shape[1]))
		trace_wave_res = numpy.zeros((rss_flux._data.shape[0], trace_mask._data.shape[1]))
		trace_mask_res = numpy.zeros((rss_flux._data.shape[0], trace_mask._data.shape[1]))
		cross_pixel = numpy.arange(trace_mask._data.shape[0])
		cross_pixel_res = numpy.linspace(0, trace_mask._data.shape[0]-1, rss_flux._data.shape[0])
		for i in range(trace_mask._data.shape[1]):
			if method == "spline":
				interp_trace_fwhm = interpolate.UnivariateSpline(cross_pixel, trace_fwhm._data[:, i], s=0)
				interp_spect_fwhm = interpolate.UnivariateSpline(cross_pixel, spect_fwhm._data[:, i], s=0)
				interp_wave = interpolate.UnivariateSpline(cross_pixel, trace_wave._data[:, i], s=0)
				interp_mask = interpolate.UnivariateSpline(cross_pixel, trace_mask._data[:, i], s=0)
				trace_fwhm_res[:, i] = interp_trace_fwhm(cross_pixel_res)
				spect_fwhm_res[:, i] = interp_spect_fwhm(cross_pixel_res)
				trace_wave_res[:, i] = interp_wave(cross_pixel_res)
				trace_mask_res[:, i] = interp_mask(cross_pixel_res)
			elif method == "linear":
				interp_trace_fwhm = interpolate.interpolate.interp1d(cross_pixel, trace_fwhm._data[:, i])
				interp_spect_fwhm = interpolate.interpolate.interp1d(cross_pixel, spect_fwhm._data[:, i])
				interp_wave = interpolate.interpolate.interp1d(cross_pixel, trace_wave._data[:, i])
				interp_mask = interpolate.interpolate.interp1d(cross_pixel, trace_mask._data[:, i])
				trace_fwhm_res[:, i] = interp_trace_fwhm(cross_pixel_res)
				spect_fwhm_res[:, i] = interp_spect_fwhm(cross_pixel_res)
				trace_wave_res[:, i] = interp_wave(cross_pixel_res)
				trace_mask_res[:, i] = interp_mask(cross_pixel_res)
			else:
				raise NotImplementedError(f"interpolation method '{method}' not implemented")
	else:
		trace_fwhm_res = trace_fwhm
		spect_fwhm_res = spect_fwhm
		trace_wave_res = trace_wave
		trace_mask_res = trace_mask
	
	trace_fwhm.setData(data=trace_fwhm_res)
	spect_fwhm.setData(data=spect_fwhm_res)
	trace_wave.setData(data=trace_wave_res)
	trace_mask.setData(data=trace_mask_res)
	# write new trace frames
	trace_fwhm.writeFitsData(filename=f"{out_path}/{out_name}.fwhm.fits")
	spect_fwhm.writeFitsData(filename=f"{out_path}/{out_name}.res.fits")
	trace_wave.writeFitsData(filename=f"{out_path}/{out_name}.disp.fits")
	trace_mask.writeFitsData(filename=f"{out_path}/{out_name}.trc.fits")

	# TODO: convert physical units into electrons
	# 	- read flux calibration factor
	# 	- apply factor to simulated spectra

	rss_flux_out = numpy.zeros((rss_flux._data.shape[0], trace_mask._data.shape[1]))
	for j in range(rss_flux._data.shape[0]):
		# extract the j-spectrum & set the original (simulated) fwhm
		spectrum = rss_flux[j]
		# BUG: resampling should be done after applying LSF to ensure the later is done in the most well-sampled data possible
		# resample to instrumental sampling
		spectrum = spectrum.resampleSpec(trace_wave_res[j], method="spline")
		# degrade spectral resolution to instrumental fwhm
		# BUG: there are cases in which instrumental resolution is better than simulation resolution
		spectrum = spectrum.smoothGaussVariable(numpy.sqrt(numpy.abs(spect_fwhm_res[j]**2 - sim_fwhm**2)))
		# transform to pixel space
		rss_flux_out[j] = spectrum._data

	out_2d = numpy.zeros(trace_stray._data.shape)
	pixel = numpy.arange(spect_fwhm_res.shape[1])
	fact = numpy.sqrt(2.*numpy.pi)
	for i in range(trace_mask_res.shape[1]):
		# re-project spectrum using the given instrumental setup
		sigma = trace_fwhm_res[:, i][None, :] / 2.354
		A = numpy.exp(-0.5*((pixel[:, None]-trace_mask_res[:, i][None, :]) / abs(sigma))**2) / (fact*abs(sigma))
		out_2d[:, i] = numpy.dot(A, rss_flux_out[:, i])

	# add stray light map
	out_2d = out_2d + trace_stray._data
	# TODO: add fiber-to-fiber transmission (fiberflat)
	# TODO: add random poissonian noise (bias+dark)
	# TODO: convert to ADU
	# store re-projected in FITS
	rep = pyfits.PrimaryHDU(out_2d)
	rep.writeto(f"{out_path}/{out_name}_2d.fits", overwrite=True)

def testres_drp(image, trace, fwhm, flux):
	"""
			Historic task used for debugging of the the extraction routine...
	"""
	img = Image()
	t1 = time.time()
	img.loadFitsData(image, extension_data=0)
	trace_mask = TraceMask()
	trace_mask.loadFitsData(trace, extension_data=0)
	trace_fwhm = TraceMask()
	#   trace_fwhm.setData(data=numpy.ones(trace_mask._data.shape)*2.5)
	trace_fwhm.loadFitsData(fwhm, extension_data=0)

	trace_flux = TraceMask()
	trace_flux.loadFitsData(flux, extension_data=0)
	x = numpy.arange(img._dim[0])
	out = numpy.zeros(img._dim)
	fact = numpy.sqrt(2.*numpy.pi)
	for i in range(img._dim[1]):
		#  print i
		A=1.0*numpy.exp(-0.5*((x[:, numpy.newaxis]-trace_mask._data[:, i][numpy.newaxis, :])/abs(trace_fwhm._data[:, i][numpy.newaxis, :]/2.354))**2)/(fact*abs(trace_fwhm._data[:, i][numpy.newaxis, :]/2.354))
		spec = numpy.dot(A, trace_flux._data[:, i])
		out[:, i] = spec
		if i==1000:
			pylab.plot(spec, '-r')
			pylab.plot(img._data[:, i], 'ok')
			pylab.show()

	hdu = pyfits.PrimaryHDU(img._data-out)
	hdu.writeto('res.fits', clobber=True)
	hdu = pyfits.PrimaryHDU(out)
	hdu.writeto('fit.fits', overwrite=True)

	hdu = pyfits.PrimaryHDU((img._data-out) / img._data)
	hdu.writeto('res_rel.fits', overwrite=True)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image")
def old_preprocRawFrame_drp(in_image, out_image, boundary_x, boundary_y, positions, orientation, subtract_overscan='1', compute_error='1', gain="none", rdnoise="none", gain_field='GAIN', rdnoise_field='RDNOISE'):
	"""
		Preprocess LVM raw image with different amplifiers to a full science CCD images. The orientations of the sub images are taken into account as well as their
		overscan regions. A Poission error image can be automatically computed during this process. This requires that the {gain_field} and the Read-Out Noise are stored
		as header keywords in the raw image.

		Parameters
		--------------
		in_image: string
				name of the FITS raw image containing the subimage to be preprocessed
		out_image: string
				Name of the FITS file  in which the preprocessed image will be stored
		boundary_x : string of two comma-separated integers
				Pixel boundaries of the subimages EXCLUDING the overscan regions along x axis (first pixel has index 1)
		boundary_y : string of two comma-separated integers
				Pixel boundaries of the subimages EXCLUDING the overscan regionsalong y axis (first pixel has index 1)
		positions : string of two comma-separated  integer digits,
				Describes the position of each sub image in colum/row format where the first digit describes the row and the second the column position.
				'00' would correspond to the lower left corner in the preprocessed CCD frame
		orientation: comma-separated strings
				Describes how each subimage should be oriented before place into the glued CCD frame. Possible options are: 'S','T','X','Y','90','180'', and 270'
				Their meaning are:
				'S' : orientation is unchanged
				'T' : the x and y axes are swapped
				'X' : mirrored along the x axis
				'Y' : mirrored along the y axis
				'90' : rotated by 90 degrees
				'180' : rotated by 180 degrees
				'270' : rotated by 270 degrees
		subtract_overscan : string of integer ('0' or '1'), optional  with default: '1'
				Should the median value of the overscan region be subtracted from the subimage before glueing, '1' - Yes, '0' - No
		compute_error : string of integer ('0' or '1'), optional  with default: '1'
				Should the Poisson error included into the second extension, '1' - Yes, '0' - No
		gain : string, optional with default :''
				Name of the FITS Header keyword for the gain value of the CCD, will be multiplied
		rdnoise: string, optional with default: ''
				Name of the FITS Header keyword for the read out noise value
	"""
	# convert input parameters to proper type
	bound_x = boundary_x.split(',')
	bound_y = boundary_y.split(',')
	orient = orientation.split(',')
	pos = positions.split(',')
	subtract_overscan = bool(int(subtract_overscan))
	compute_error = bool(int(compute_error))

	org_image = loadImage(in_image)
	ab, cd = org_image.split(2, axis="Y")
	(a, b), (c, d) = ab.split(2, axis="X"), cd.split(2, axis="X")
	# reflect b and d amplifiers to have the overscan regions in the last columns
	b.orientImage("X")
	d.orientImage("X")
	images = [a, b, c, d]

	try:
		gains = [org_image.getHdrValue(f"{gain_field}{i+1}") for i in range(len(images))]
	except KeyError:
		try:
			gain = org_image.getHdrValue(gain_field)
		except KeyError:
			if gain != "":
				try:
					gain = float(gain)
				except ValueError:
					gain = "1.0"
		gains = len(images)*[gain]  # list of gains
	
	try:
		rdnoises = [org_image.getHdrValue(f"{rdnoise_field}{i+1}") for i in range(len(images))]
	except KeyError:
		try:
			rdnoise = org_image.getHdrValue(rdnoise_field)
		except KeyError:
			if rdnoise != "":
				try:
					rdnoise = float(rdnoise)
				except ValueError:
					rdnoise = "1.0"
		rdnoises = len(images)*[rdnoise]  # list of read-out noises

	# create empty lists
	bias = []  # list of biasses
	for i in range(len(images)):
		# append the bias from the overscane region
		bias_ = images[i].cutOverscan(bound_x, bound_y, subtract_overscan)
		bias.append(0.0 if numpy.isnan(bias_) else bias_)
		# return to original orientation
		if i == 1 or i == 3:
			images[i].orientImage("X")
		# multiplication with the gain factor
		images[i] *= gains[i]

		# change orientation of subimages
		images[i].orientImage(orient[i])
		if compute_error:
			images[i].computePoissonError(rdnoise=rdnoises[i])

		# copy original geader into each image
		images[i].setHeader(org_image.getHeader())

	# create glued image
	full_img = glueImages(images, pos)
	# flip along dispersion axis
	ccd = org_image._header["CCD"]
	if ccd.startswith("z") or ccd.startswith("b"):
		full_img.orientImage("X")

	# adjust FITS header information
	full_img.removeHdrEntries(['{gain_field}', f'{rdnoise_field}', ''])
	# add gain keywords for the different subimages (CCDs/Amplifiers)
	for i in range(len(images)):
		full_img.setHdrValue(f'HIERARCH AMP%i {gain_field}' % (i+1), gains[i], 'Gain value of CCD amplifier %i' % (i+1))
	# add read-out noise keywords for the different subimages (CCDs/Amplifiers)
	for i in range(len(images)):
		full_img.setHdrValue(f'HIERARCH AMP%i {rdnoise_field}' % (i+1), rdnoises[i], 'Read-out noise of CCD amplifier %i' % (i+1))
	# add bias of overscan region for the different subimages (CCDs/Amplifiers)
	for i in range(len(images)):
		full_img.setHdrValue('HIERARCH AMP%i OVERSCAN' % (i+1), bias[i], 'Overscan median (bias) of CCD amplifier %i' % (i+1))
	#write out FITS file
	full_img.writeFitsData(out_image)
	return full_img

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image")
def preprocRawFrame_drp(in_image, out_image, positions="00,10,01,11", orientation="S,S,S,S", subtract_overscan='1', os_bound_x='', os_bound_y='', compute_error='1', gain="1.0", rdnoise="5", gain_field='GAIN', rdnoise_field='RDNOISE', unit="ADU"):
	# convert input parameters to proper type
	orient = orientation.split(',')
	pos = positions.split(',')
	subtract_overscan = bool(int(subtract_overscan))
	compute_error = bool(int(compute_error))

	# load image
	org_image = loadImage(in_image)

	# parse overscan (OS) section:
	# * extract BIASSEC
	if "BIASSEC" in org_image._header:
		os_x, os_y = org_image._header["BIASSEC"].strip("[]").split(",")
		os_y, os_x = os_y.split(":"), os_x.split(":")
	elif os_bound_x and os_bound_y:
		os_y, os_x = os_bound_y.split(','), os_bound_x.split(',')
	else:
		pass
	os_y, os_x = (int(os_y[0])-1, int(os_y[1])), (int(os_x[0])-1, int(os_x[1]))

	if "TRIMSEC" in org_image._header:
		sc_x_i, sc_y_i, sc_x_f, sc_y_f = org_image._header["TRIMSEC"].replace("[", "").replace("]", "").split(",")
		sc_y_i, sc_x_i, sc_y_f, sc_x_f = sc_y_i.split(":"), sc_x_i.split(":"), sc_y_f.split(":"), sc_x_f.split(":")
	else:
		# assume OS region is in the middle column
		ysize, xsize = org_image._dim
		xsize /= 2
		os_xsize = (os_x[1] - os_x[0]) / 2
		sc_y_i, sc_x_i = ["1", str(ysize)], ["1", str(xsize-os_xsize)]
		sc_y_f, sc_x_f = ["1", str(ysize)], [str(xsize-2*os_xsize+1), str(xsize)]
	sc_i, sc_f = ((int(sc_y_i[0])-1, int(sc_y_i[1])), (int(sc_x_i[0])-1, int(sc_x_i[1]))), ((int(sc_y_f[0])-1, int(sc_y_f[1])), (int(sc_x_f[0])-1, int(sc_x_f[1])))

	# select data outside the cut out region (overscan)
	os_region = Image(data=org_image._data[os_y[0]:os_y[1], os_x[0]:os_x[1]])
	# * split OS in four amplifier sections
	os_ab, os_cd = os_region.split(2, axis="Y")
	(os_a, os_b), (os_c, os_d) = os_ab.split(2, "X"), os_cd.split(2, "X")
	os_quads = [os_a, os_b, os_c, os_d]
	# * compute statistics on each OS section
	os_bias = [numpy.nanmedian(os_quad._data) for os_quad in os_quads]
	os_bias_std = [numpy.nanstd(os_quad._data) for os_quad in os_quads]

	# parse science section:
	# * extract science section
	sc_regions = [org_image._data[sc_y[0]:sc_y[1], sc_x[0]:sc_x[1]] for sc_y, sc_x in (sc_i, sc_f)]
	sc_region = Image(data=numpy.column_stack(sc_regions))
	# * split in four amplifiers
	sc_ab, sc_cd = sc_region.split(2, axis="Y")
	(sc_a, sc_b), (sc_c, sc_d) = sc_ab.split(2, "X"), sc_cd.split(2, "X")
	sc_quads = [sc_a, sc_b, sc_c, sc_d]
	# * apply bias subtraction
	quads = [quad - os_bias[i] for i, quad in enumerate(sc_quads)]

	# parse gain and read noise from header if possible
	try:
		gains = [org_image.getHdrValue(f"{gain_field}{i+1}") for i in range(len(quads))]
	except KeyError:
		try:
			gain = org_image.getHdrValue(gain_field)
		except KeyError:
			if gain != "":
				try:
					gain = float(gain)
				except ValueError:
					gain = 1.0
		gains = len(quads)*[gain]

	try:
		rdnoises = [org_image.getHdrValue(f"{rdnoise_field}{i+1}") for i in range(len(quads))]
	except KeyError:
		try:
			rdnoise = org_image.getHdrValue(rdnoise_field)
		except KeyError:
			if rdnoise != "":
				try:
					rdnoise = float(rdnoise)
				except ValueError:
					rdnoise = 5.0
		rdnoises = len(quads)*[rdnoise]

	# orient quadrants as requested
	[quad.orientImage(orient[i]) for i, quad in enumerate(quads)]
	# convert to specified unit
	if unit == "e-":
		for i in range(len(quads)):
			quads[i] *= gains[i]
			if compute_error:
				quads[i].computePoissonError(rdnoise=rdnoises[i])
	elif unit.upper() == "ADU":
		unit = unit.upper()
	else:
		pass

	# join images
	preproc_image = glueImages(quads, pos)
	# flip along dispersion axis
	ccd = org_image._header["CCD"]
	if ccd.startswith("z") or ccd.startswith("b"):
		preproc_image.orientImage("X")

	# update header
	preproc_image.setHeader(org_image.getHeader())
	# update/set unit
	preproc_image.setHdrValue("BUNIT", unit, "physical units of the array values")
	# add amplifier quadrants
	for i in range(len(quads)):
		ysize, xsize = quads[i]._dim
		x, y =int(pos[i][0]), int(pos[i][1])
		# flip y-axis
		y = 1 if y == 0 else 0
		preproc_image.setHdrValue(f"HIERARCH AMP{i+1} TRIMSEC", f"[{x*xsize+1}:{xsize*(x+1)}, {y*ysize+1}:{ysize*(y+1)}]", f"Region of CCD amplifier {i+1}")
	# add gain keywords for the different subimages (CCDs/Amplifiers)
	for i in range(len(quads)):
		preproc_image.setHdrValue(f'HIERARCH AMP{i+1} {gain_field}', gains[i], f'Gain value of CCD amplifier {i+1} [e-/ADU]')
	# add read-out noise keywords for the different subimages (CCDs/Amplifiers)
	for i in range(len(quads)):
		preproc_image.setHdrValue(f'HIERARCH AMP{i+1} {rdnoise_field}', rdnoises[i], f'Read-out noise of CCD amplifier {i+1} [e-]')
	# add bias of overscan region for the different subimages (CCDs/Amplifiers)
	for i in range(len(quads)):
		preproc_image.setHdrValue(f'HIERARCH AMP{i+1} OVERSCAN', os_bias[i], f'Overscan median of CCD amplifier {i+1} [ADU]')
	# add bias std of overscan region for the different subimages (CCDs/Amplifiers)
	for i in range(len(quads)):
		preproc_image.setHdrValue(f"HIERARCH AMP{i+1} OVERSCAN_STD", os_bias_std[i], f"Overscan std of CCD amplifier {i+1} [ADU]")
	#write out FITS file
	preproc_image.writeFitsData(out_image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_image", "in_bias", "in_dark", "in_pixelflat")
def basicCalibration_drp(in_image, out_image, in_bias=None, in_dark=None, in_pixelflat=None):

    proc_image = loadImage(in_image).convertUnit(unit="e-")
    exptime = proc_image._header["EXPTIME"]
    img_type = proc_image._header["IMAGETYP"].lower()

    # dummy calibration images
    dummy_bias = Image(data=numpy.zeros_like(proc_image._data))
    dummy_dark = Image(data=numpy.zeros_like(proc_image._data))
    dummy_flat = Image(data=numpy.ones_like(proc_image._data))

    # read master bias
    if img_type in ["bias"] or in_bias is None or not os.path.isfile(in_bias):
        master_bias = dummy_bias
    else:
        master_bias = loadImage(in_bias).convertUnit(unit="e-")
    
    # read master dark
    if img_type in ["bias", "dark"] or in_dark is None or not os.path.isfile(in_dark):
        master_dark = dummy_dark
    else:
        master_dark = loadImage(in_dark).convertUnit(unit="e-")
        # scale down the dark if needed
        factor = exptime / master_dark._header["EXPTIME"]
        if factor > 1.0:
            # WARNING: scaling up
            pass 
        master_dark *= factor

    # read master flat
    if img_type in ["bias", "dark", "flat", "flatfield", "fiberflat", "object"] or in_pixelflat is None or not os.path.isfile(in_pixelflat):
        master_flat = dummy_flat
    else:
        master_flat = loadImage(in_pixelflat).convertUnit(unit="e-")

    # normalize in case of flat calibration
    if img_type == "bias":
        calib_image = proc_image
    if img_type == "flat" or img_type == "flatfield":
        proc_image = proc_image / numpy.median(proc_image._data)

    # run basic calibration
    calib_image = (proc_image - master_dark - master_bias) / master_flat
    calib_image.writeFitsData(out_image)

@missing_files(["BAD_CALIBRATION_FRAMES"], "in_images")
def createMasterFrame_drp(in_images, out_image, reject_cr=False, exptime_thresh=5, **cr_kwargs):
	"""

		Combines the given calibration frames (bias, dark, or pixelflat) into a master calibration frame.
		
		Optionally this task will apply a cosmic ray rejection algorithm (reject_cr=True) if the
		following conditions apply:

			* exposure time < exptime_thresh OR
			* number of frames is <= 2
		
		The combination of the images will be carried out using a sigma clipped median statistic if the
		number of exposures is > 2. If the number of exposures <= 2, a simple average statistic is
		applied. In the special case that CR rejection is needed, the combination of images is selective:
		
			* where cosmic ray in one frame, select the other
			* where cosmic ray in none of the frames, calculate an average of both.
		
		When only one frame is given, it is still flagged as master, but a warning will be thrown.

		Parameters
		----------


		Examples
		--------


	"""
	if not isinstance(in_images, (list, tuple)): in_images = [in_images]

	nexp = len(in_images)
	proc_images, exptimes, img_types = [], [], []
	for in_image in in_images:
		proc_image = loadImage(in_image).convertUnit(unit="ADU")
		exptimes.append(proc_image._header["EXPTIME"])
		img_types.append(proc_image._header["IMAGETYP"].lower())
		proc_images.append(proc_image)

	master_type = img_types[0]
	if any(master_frame != img_types):
		# TODO: drop minority type
		# TODO: throw warning: dropping frames != frames[0]
		pass

	master_exptime = exptimes[0]
	if any(master_exptime != exptimes):
		# TODO: scale frames to a common exptime
		# TODO: throw warning: scale frames to a common exptime
		pass

	if reject_cr and (master_exptime < exptime_thresh or nexp <= 2):
		cr_select_1 = proc_images[0].createCosmicMask(**cr_kwargs)
		if nexp == 2:
			cr_select_2 = proc_images[1].createCosmicMask(**cr_kwargs)
			
			# filter out cosmic rays by selecting pixels where no CR were detected
			# normalize counts if pixelflat
			new_data_1 = numpy.where(cr_select_1, x=proc_images[1]._data, y=proc_images[0]._data)
			new_data_2 = numpy.where(cr_select_2, x=proc_images[0]._data, y=proc_images[1]._data)
			if master_type == "flat" or master_type == "flatfield":
				new_data = [new_data_1/numpy.nanmedian(new_data_1), new_data_2/numpy.nanmedian(new_data_2)]
			else:
				new_data = [new_data_1, new_data_2]

			# average images
			new_data = numpy.mean(new_data, axis=0)

			new_header = proc_images[0]._header
			# combine CR pixel selection
			cr_mask = numpy.logical_and(cr_select_1, cr_select_2)
			# combine original masks
			new_mask = numpy.logical_and(proc_images[0]._mask, proc_images[1]._mask)

		elif nexp == 1:
			new_data = proc_images[0]._data
			new_header = proc_images[0]._header
			cr_mask = cr_select_1
			new_mask = proc_images[0]._mask

		# prepare image for CRR
		master_frame = Image(data=new_data, header=new_header, mask=cr_mask)
		replace_box = cr_kwargs.get("replace_box", (20,1))
		replace_error = cr_kwargs.get("replace_error", 1e10)
		# filter out remaining cosmic rays
		master_frame.replaceMaskMedian(*replace_box, replace_error=replace_error)
		# add original masks
		master_frame.setData(mask=new_mask)
	else:
		if master_type == "bias":
			master_frame = combineImages(proc_images, method="clipped_mean", k=3)
		elif master_type == "dark":
			master_frame = combineImages(proc_images, method="clipped_mean", k=3)
		elif master_type == "flat" or master_type == "flatfield":
			master_frame = combineImages([proc_image / numpy.nanmedian(proc_image._data) for proc_image in proc_images], method="clipped_mean", k=3)

	# TODO:
	# * add binary table with columns: MJD, EXPNUM, SPEC, CHANNEL, EXPTIME
	master_frame._header["EXPTIME"] = master_exptime
	master_frame._header["ISMASTER"] = (True, "Is this a combined (master) frame")
	master_frame._header["NFRAMES"] = (nexp, "Number of exposures combined")

	master_frame.writeFitsData(out_image)
    
# TODO: for fiberflats, calculate an average over an X range (around the center) of the extracted fibers and normalize by it
# TODO: then combine them using the RSS method implemented