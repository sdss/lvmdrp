from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import collections
import numpy
try:
  import matplotlib
  from matplotlib import pyplot as plt
except:
  pass
from multiprocessing import cpu_count
from multiprocessing import Pool
from scipy import special
from scipy import ndimage
from scipy import optimize
from scipy import stats
from scipy import interpolate
from lvmdrp.core.rss  import *
from lvmdrp.core.cube  import *
from lvmdrp.core.image import *
from lvmdrp.core.passband import *

from lvmdrp.core import fit_profile
from lvmdrp.external import ancillary_func

from lvmdrp.utils.decorators import missing_files


description='Provides Methods to process Row Stacked Spectra (RSS) files'

def mergeRSS_drp(files_in, file_out,  mergeHdr='1'):
	"""
			Different RSS are merged into a common file by extending the number of fibers.
			Note that the number of spectral pixel need to be the same and that all input RSS must have the same extension.

			Parameters
			--------------
			files_in : string
					Comma-separates name of RSS FITS files to be merged into a signal RSS
			file_out : string
					Name of the merged RSS FITS file
			mergeHdr : string of integer (0 or 1), optional with default: '1'
					Flag to indicate if the header of the input RSS files are also merger and stored in the merged RSS.
					1 if yes, 0 if not

			Examples
			----------------
			user:> lvmdrp rss mergeRSS RSS1.fits,RSS2.fits,RSS3.fits RSS_OUT.fits
	"""

	files = files_in.split(',')

	for i in range(len(files)):
		if i==0:
			rss = loadRSS(files[i])
		else:
			rss_add = loadRSS(files[i])
			if mergeHdr=='0':
				rss.append(rss_add, append_hdr=False)
			else:
				rss.append(rss_add, append_hdr=True)
	rss.writeFitsData(file_out)

@missing_files(["BAD_CALIBRATION_FRAMES"], "arc_rss", "ref_line_file")
def detWaveSolution_drp(arc_rss, disp_rss, res_rss, ref_line_file='', ref_spec='', pixel='', ref_lines='', poly_dispersion='-5', poly_fwhm='-3,-5', init_back='10.0',  aperture='13', flux_min='200.0', fwhm_max='10.0', rel_flux_limits='0.1,5.0', negative=False, verbose='1' ):
	"""
			Measures the pixel position of emission lines in wavelength UNCALIBRATED for all fibers of the RSS.
			Starting from the initial guess of pixel positions for a given fiber, the program measures the position using
			Gaussian fitting to the first and last fiber of the RSS. The best fit emission line position of the previous fiber
			are used as guess parameters. Certain criterion can be imposed to reject certain measurements and flag those
			as bad. They will be ignored for the dispersion solution, which is estimated for each fiber independently.
			Two RSS FITS file containing the wavelength pixel table and the FWHM pixel table will be stored.

			Parameters
			--------------
			arc_rss : string
					Input RSS FITS file name of the uncalibrated arc lamp exposure
			disp_rss: string
					Name of the FITS file in which the wavelength RSS pixel table is stored
			res_rss: string
					Name of the FITS file in which the spectral resolution (FWHM) RSS pixel table is stored
			ref_line_file : string, optional with default: ''
					ASCII file name containing the number of the reference fiber in the first row,
					reference wavelength of emission line, its rough centroid pixel position a flag if the width of the
					line should be considered for the spectral resolution measurements (space separated) in
					each subsquent row.
					If no ASCII file is provided those information must be given in the ref_spec, pixel and ref_lines parameters.
			ref_spec : string of integer, optional with default: ''
					Number of the fiber in the RSS for which the rough guess for their centroid pixel position (x-direction) are given.
					Only used if no ASCII file is given.
			pixel : string of integers, optional with default: ''
					Comma-separated list of rough centroid pixel position for each emission line for the corresponding reference fiber.
					Only used if no ASCII file is given.
			ref_lines : string of floats, optional with default: ''
					Comma-separated list of reference emission-line wavelength. Need to be same number of values as for the pixel guess
					Only used if no ASCII file is given.
			poly_dispersion : string of integer, optional with default: '-5'
					Degree of polynomial used to construct the wavelength solution for each fiber. (positiv: normal polynomial, negative: Legandre polynomial)
			poly_fwhm : string of two integers, optional with default: '-3,-5'
					First integer is the degree of polynomial used to smooth the measured FWHM of each line as a function of fiber number (cross-dispersion).
					Second integer is the degree of polynomial used to subsquently extrapolate the line FWHM across the disperion direction,
					(positiv: normal polynomial, negative: Legandre polynomial)
			init_back : string of float, optinal with default: '10.0'
					Initial guess for the constant background level that can be fitted in addition to the Gaussian for each line.
					If this parameter is left empty, the background level is fixed to zero.
			aperture : string of integer, optional with default: '13'
					Aperture centered on the guess of the pixel position from which pixel with the maximum flux is used as the guess for the Gaussian fitting.
					This is also the size of the fitted region for each line.
			flux_min : string of float, optional with default: '200.0'
					Required minimum integrated flux of the best-fit Gaussian model to be considered as a reliable value.
					The measurement for this emission line for the specific fiber is masked if it falls below this threshold.
			fwhm_max : string of float, optional with default: '10.0'
					Maximum FWHM of the best-fit Gaussian model to be considered as a reliable value.
			rel_flux_limits : string of two floats, optional with default: '0.1,5.0'
					Required relative integrated fluxes with respect to the measured fluxes  for the reference fiber.
					If relative fluxes are outside this range, they will be masked.
			negative :
			verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Examples
			----------------
			user:> lvmdrp rss detWaveSolution ARC_RSS.fits arc REF_FILE.txt poly_dispersion='-7' poly_fwhm='-4,-5'
			user:> lvmdrp rss detWaveSolution ARC_RSS.fits arc ref_spec=100 pixel=200,500,1000 ref_lines=3000.0,5000.0,8000.0 flux_min=100.0
	"""


	# convert parameters to the correct type
	flux_min=float(flux_min)
	fwhm_max=float(fwhm_max)
	init_back=float(init_back)
	aperture=float(aperture)
	poly_dispersion=int(poly_dispersion)
	poly_fwhm_cross=int(poly_fwhm.split(',')[0])
	poly_fwhm_disp=int(poly_fwhm.split(',')[1])

	limits = rel_flux_limits.split(',')
	rel_flux_limits=[float(limits[0]), float(limits[1])]
	negative=bool(negative)
	verbose=int(verbose)

	if ref_line_file!='':
		# load initial pixel positions and reference wavelength from txt config file NEED TO BE REPLACE BY XML SCHEMA
		file_in = open(ref_line_file, 'r') # load file
		lines = file_in.readlines() # read lines
		nlines = len(lines)-1
		pixel = numpy.zeros(nlines, dtype=numpy.float32) # empty for pixel position
		ref_lines = numpy.zeros(nlines, dtype=numpy.float32) # empty for reference wavelength
		use_fwhm = numpy.zeros(nlines, dtype="bool") # empty for reference wavelength
		ref_spec = int(lines[0]) # the reference fiber for the initial positions
		# read the information from file
		for i in range(1, nlines+1):
			line = lines[i].split()
			pixel[i-1] = float(line[0])
			ref_lines[i-1] =float(line[1])
			use_fwhm[i-1] = int(line[2])
	else:
		# get the reference spectrum number, the inital pixel positions and their reference wavelength from the parameter list
		ref_spec = int(ref_spec) # reference fiber
		pixels = pixel.split(',') # split pixel list
		ref = ref_lines.split(',') # split reference wavelength list
		nlines = len(pixels)
		pixel = numpy.zeros(nlines, dtype=numpy.float32) # empty for pixel position
		ref_lines = numpy.zeros(nlines, dtype=numpy.float32) # empty for reference wavelength
		# iterate over the different lines and convert to float
		for i in range(len(ref)):
			ref_lines[i] = float(ref[i])
			pixel[i] = float(pixels[i])

	# initialize the extracted arc line frame
	arc = FiberRows() # create object
	arc.loadFitsData(arc_rss) # load data

	if negative==True:
		arc = arc*-1+numpy.median(arc._data)

	# setup storage array
	wave_sol = numpy.zeros((arc._fibers, arc._data.shape[1]), dtype=numpy.float32) # empty for wavelength solution
	fwhm_sol = numpy.zeros((arc._fibers, arc._data.shape[1]), dtype=numpy.float32) # empty for FWHM spectral resolution
	rms = numpy.zeros(arc._fibers, dtype=numpy.float32) # empty for rms of wavelength solution for each fiber

	# measure the ARC lines with individual Gaussian across the CCD


	(fibers, flux, cent_wave, fwhm, masked) = arc.measureArcLines(ref_spec, pixel, aperture=aperture, init_back=init_back, flux_min=flux_min, fwhm_max=fwhm_max, rel_flux_limits=rel_flux_limits, verbose=bool(verbose))


	# smooth the FWHM values for each ARC line in cross-dispersion direction
	for i in range(nlines): # iterate over modelled emission lines
		select = numpy.logical_and(numpy.logical_not(masked[:, i]), flux[:, i]>flux_min)
		fwhm_med = ndimage.filters.median_filter(numpy.fabs(fwhm[select, i]), 4)
		if poly_fwhm_cross>0:
			fit_fwhm = numpy.polyfit(fibers[select], fwhm_med, poly_fwhm_cross) # fit the profile with a polynomial of given order
			fwhm[:, i]  = numpy.polyval(fit_fwhm, fibers)  # replace the smooth profile
		elif poly_fwhm_cross<0:
			legandre_fwhm = fit_profile.LegandrePoly(numpy.zeros(-1*poly_fwhm_cross+1), min_x=0, max_x=arc._fibers-1)
			legandre_fwhm.fit((fibers[select]),  fwhm_med)
			fwhm[:, i] = legandre_fwhm(fibers)
		if i==-1:
			#pylab.plot(fibers[select], fwhm_med, 'ok')
			pylab.plot(fibers, masked[:, i])
			#pylab.plot(fibers, cent_wave[:, i])
			pylab.show()
   # Determine the wavelength solution
	select_ref_lines = ref_lines>0.0 # select the lines for the wavelength calibration with
	# Iterate over the fibers
	good_fibers=numpy.zeros(len(fibers), dtype="bool")
	masked_fib = numpy.zeros(len(fibers), dtype="uint16")
	for i in fibers:
		select = masked[i, select_ref_lines]
		masked_fib[i] = numpy.sum(select)

		if numpy.sum(select)==0:
			good_fibers[i]=True
		elif numpy.sum(select)==len(select):
			select[:]=False
		select=numpy.logical_not(select)

		if poly_dispersion>0:
			fit_wave = numpy.polyfit(cent_wave[i, select_ref_lines][select], ref_lines[select_ref_lines][select], poly_dispersion) # fit with a polynomial
			rms[i]= numpy.std(ref_lines[select_ref_lines][select]-numpy.polyval(fit_wave, cent_wave[i, select_ref_lines][select])) # compute the rms of the polynomial
			wave_sol[i, : ] = numpy.polyval(fit_wave, numpy.arange(arc._data.shape[1])) # write wavelength solution
		elif poly_dispersion<0:
			legandre_wave = fit_profile.LegandrePoly(numpy.zeros(-1*poly_dispersion+1), min_x=0, max_x=arc._data.shape[1]-1)
			legandre_wave.fit(cent_wave[i, select_ref_lines][select],   ref_lines[select_ref_lines][select])

			rms[i]= numpy.std(ref_lines[select_ref_lines][select]-legandre_wave(cent_wave[i, select_ref_lines][select])) # compute the rms of the polynomial
			wave_sol[i, :] = legandre_wave(numpy.arange(arc._data.shape[1]))
			if verbose>0 and i==verbose:
				pylab.figure(figsize=(17, 5))
				#pylab.plot(ref_lines[select_ref_lines][select], ref_lines[select_ref_lines][select]-legandre_wave(cent_wave[i, select_ref_lines][select]), 'ok')
				pylab.plot(cent_wave[i, select_ref_lines][select],   ref_lines[select_ref_lines][select], 'ok', label="data")
				pylab.plot(numpy.arange(arc._data.shape[1]), wave_sol[i, :], label="wave solution (AA)")
				pylab.xlabel("dispersion axis (pixel)")
				pylab.ylabel("wavelength (AA)")
				pylab.show()
	##pylab.plot(masked_fib)
	##pylab.plot(rms[good_fibers])
	##pylab.plot(good_fibers)
	##pylab.show()

	# Estimate the spectral resoltuion patern

	dwave = wave_sol[:, 1:]-wave_sol[:, :-1] # compute the pixel size in wavelength units accross the CCD
	cent_round = numpy.round(cent_wave).astype('int16') # get the pixel coordinates of the lines

	select_lines = use_fwhm

	# Iterate over the fibers
	for i in fibers:
		fwhm_wave = dwave[i, cent_round[i, :]]*fwhm[i, :]
		if poly_fwhm_disp>0:
			fit_fwhm= numpy.polyfit(cent_wave[i, select_lines], fwhm_wave[select_lines], poly_fwhm_disp )
			fwhm_sol[i, : ] = numpy.polyval(fit_fwhm, numpy.arange(arc._data.shape[1]))
		elif poly_fwhm_disp<0:
			leg_poly_fwhm = fit_profile.LegandrePoly(numpy.zeros(-1*poly_fwhm_disp+1), min_x=0, max_x=arc._data.shape[1]-1 )
			leg_poly_fwhm.fit(cent_wave[i, select_lines], fwhm_wave[select_lines])
			fwhm_sol[i, :]=leg_poly_fwhm(numpy.arange(arc._data.shape[1]))
		if verbose>0 and i==verbose:
			pylab.figure(figsize=(17, 5))
			pylab.plot(numpy.arange(arc._data.shape[1]-1),  dwave[i, :], '-r', label="wave sampling (AA)")
			# print(cent_round)
			# print(dwave[i, cent_round])
			pylab.plot(cent_wave[i, select_lines], fwhm[i, select_lines], 'or', label="data (pixel)")
			pylab.plot(cent_wave[i, select_lines], fwhm_wave[select_lines], 'ok', label="data (AA)")
			pylab.plot(numpy.arange(arc._data.shape[1]), fwhm_sol[i, :], "b", label="LSF (AA)")
			pylab.legend(loc=0, frameon=False)
			pylab.xlabel("dispersion axis (pixel)")
			# pylab.ylabel("LSF (AA)")
			pylab.show()
	arc.setHdrValue('HIERARCH PIPE FWHM POLY', '%d'%(numpy.abs(poly_fwhm_disp)), 'Order of the resolution polynomial')
	fwhm_trace = FiberRows(data = fwhm_sol, header = arc.getHeader())
	#arc.removeHdrEntries(keywords=['PIPE FWHM POLY'])
	arc.setHdrValue('HIERARCH PIPE DISP POLY', '%d'%(numpy.abs(poly_dispersion)), 'Order of the dispersion polynomial')
	arc.setHdrValue('HIERARCH PIPE DISP RMS MEDIAN', '%.4f'%(numpy.median(rms[good_fibers])), 'Median RMS of disp sol')
	arc.setHdrValue('HIERARCH PIPE DISP RMS MIN', '%.4f'%(numpy.min(rms[good_fibers])), 'Min RMS of disp sol')
	arc.setHdrValue('HIERARCH PIPE DISP RMS MAX', '%.4f'%(numpy.max(rms[good_fibers])), 'Max RMS of disp sol')

	wave_trace = FiberRows(data = wave_sol, header = arc.getHeader())

	wave_trace.writeFitsData(disp_rss)
	fwhm_trace.writeFitsData(res_rss)

	return cent_wave[:, select_lines], fwhm_wave[select_lines]

@missing_files(["BAD_CALIBRATION_FRAMES"], "rss_in")
def createPixTable_drp(rss_in, rss_out, arc_wave, arc_fwhm='', cropping=''):
	"""
			Adds the wavelength and possibly also the spectral resolution (FWHM) pixel table as new extension to
			the RSS that is stored as a seperate RSS file.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			rss_out : string
					Output RSS FITS file with the wavelength and spectral resolution pixel table added as extensions
			arc_wave : string
					RSS FITS file containing the wavelength pixel table in its primary (0th) extension
			arc_fwhm : string, optional with default: ''
					RSS FITS file containing the spectral resolution (FWHM) pixel table in its primary (0th) extension.
					No spectral resolution will not be added if the string is empty.

			Examples
			----------------
			user:> lvmdrp rss createPixTable RSS_IN.fits RSS_OUT.fits WAVE.fits
			user:> lvmdrp rss createPixTable RSS_IN.fits RSS_OUT.fits WAVE.fits FWHM.fits
	"""
	rss = RSS()
	rss.loadFitsData(rss_in)
	if cropping!='':
		crop_start=int(cropping.split(',')[0])-1
		crop_end=int(cropping.split(',')[1])-1
	else:
		crop_start=0
		crop_end=rss._data.shape[1]-1
	wave_trace =FiberRows()
	wave_trace.loadFitsData(arc_wave)
	rss.setWave(wave_trace.getData()[0][:, crop_start:crop_end])
	rss._data = rss._data[:, crop_start:crop_end]
	if rss._error is not None:
		rss._error = rss._error[:, crop_start:crop_end]
	if rss._mask is not None:
		rss._mask = rss._mask[:, crop_start:crop_end]

	try:
		rss.copyHdrKey(wave_trace, 'HIERARCH PIPE DISP RMS MEDIAN')
		rss.copyHdrKey(wave_trace, 'HIERARCH PIPE DISP RMS MIN')
		rss.copyHdrKey(wave_trace, 'HIERARCH PIPE DISP RMS MAX')
	except KeyError:
		pass

	if arc_fwhm!='':
		fwhm_trace =FiberRows()
		fwhm_trace.loadFitsData(arc_fwhm)
		rss.setInstFWHM(fwhm_trace.getData()[0][:, crop_start:crop_end])
	rss.writeFitsData(rss_out)

def checkPixTable_drp(rss_in, ref_lines, logfile, blocks='15',  init_back='100.0', aperture='10'):
	"""
			Measures the offset in dispersion direction between the object frame and the used calibration frames.
			It compares the wavelength of emission lines (i.e. night sky line) in the RSS as measured by Gaussian fitting
			with their reference wavelength. The offset in pixels is computed and stored in a log file for future processing.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			ref_lines : string
					Comma-separated list of emission lines to be fit in the RSS for each fiber
			logfile : string
					Output ASCII logfile that stores the position and deviation from the expected wavelength for
					each reference emission line
			blocks: string of integer, optional  with default: '20'
					Number of fiber blocks over which the fitted central wavelength are averaged to increase the accuary of the measurement.
					The actual number of fibers per block is roughly the total number of fibers divided by the number of blocks.
			init_back : string of float, optional with default: '100.0'
					The initial guess for a constant background level included in the Gaussian model.
					If this parameter is empty, no background level is fitted and set to zero instead.
			aperture : string of integer (>0), optional with default: '10'
					Number of pixel used for the fitting of each emission line centered on the pixel position of the line's expected wavelength

			Examples
			----------------
			user:> lvmdrp rss checkPixTable RSS_IN.fits 4500.0,5577.4,6300.3 OFFSETWAVE.log
			user:> lvmdrp rss checkPixTable RSS_IN.fits 4500.0,5577.4,6300.3 OFFSETWAVE.log aperture=14
	"""
	centres = numpy.array(ref_lines.split(',')).astype('float')
	init_back = float(init_back)
	aperture=float(aperture)
	nblocks = int(blocks)
	rss = RSS()
	rss.loadFitsData(rss_in)
	fit_wave = numpy.zeros((len(rss), len(centres)), dtype=numpy.float32)
	good_fiber = numpy.zeros(len(rss), dtype="bool")
	offset_pix = numpy.zeros((len(rss), len(centres)), dtype=numpy.float32)
	log = open(logfile, 'a')
	log.write('%s\n'%(rss_in))

	for i in range(len(rss)):
		spec = rss[i]
		disp_pix = (spec._wave[1:]-spec._wave[:-1])
		numpy.insert(disp_pix, 0, disp_pix[0])

		plot=False
		if i==-1:
			plot=True
		else:
			plot=False
		out=spec.fitSepGauss(centres, aperture, init_back=init_back, plot=plot, warning=False)
		good_fiber[i] = out[0]!=0.0
		fit_wave[i, :] = out[len(centres):2*len(centres)]
		for j in range(len(centres)):
			idx=numpy.argmin(numpy.abs(fit_wave[i, j]-spec._wave))
			offset_pix[i, j] = old_div((fit_wave[i, j]-centres[j]),disp_pix[idx])


	blocks = numpy.array_split(numpy.arange(0, len(rss)), nblocks)
	blocks_good = numpy.array_split(good_fiber, nblocks)
	for j in range(len(centres)):
		log.write('%.3f %.3f %.3f %.3f \n' %(centres[j], numpy.median(fit_wave[good_fiber, j]), numpy.median(fit_wave[good_fiber, j])-centres[j], numpy.std(fit_wave[good_fiber, j])))
		for i in range(len(blocks)):
			log.write(' %.3f'%numpy.mean(blocks[i]))
		log.write('\n')
		for i in range(len(blocks)):
			if numpy.sum(blocks_good[i])>0:
				log.write(' %.3f'%numpy.median(offset_pix[blocks[i][blocks_good[i]], j]))
			else:
				log.write(' 0.0')
		log.write('\n')
		for i in range(len(blocks)):
			if numpy.sum(blocks_good[i])>0:
				log.write(' %.3f'%(numpy.median(fit_wave[blocks[i][blocks_good[i]], j])-centres[j]))
			else:
				log.write(' 0.0')
		log.write('\n')

	off_disp_median=numpy.median(offset_pix[good_fiber, :])
	off_disp_rms=numpy.std(offset_pix[good_fiber, :])
	off_disp_median = float('%.4f'%off_disp_median) if numpy.isfinite(off_disp_median) else str(off_disp_median)
	off_disp_rms    = float('%.4f'%off_disp_rms) if numpy.isfinite(off_disp_rms) else str(off_disp_rms)
	rss.setHdrValue('hierarch PIPE FLEX XOFF', off_disp_median, 'flexure offset in x-direction')
	rss.setHdrValue('hierarch PIPE FLEX XRMS', off_disp_rms, 'flexure rms in x-direction')
	rss.writeFitsHeader(rss_in)
	log.close()

def correctPixTable_drp(rss_in, rss_out, logfile, ref_id, smooth_poly_cross='', smooth_poly_disp='', poly_disp='6', verbose='0'):
	"""
		   Corrects the RSS wavelength pixel table for possible offsets in dispersion direction due to flexure effects
		   with respect to the calibration frames taken for this object. The offfsets need to be determined beforehand
		   via the checkPixTable task. The offsets can be smoothed and/or extrapolated along the dispersion axis, but
		   a global median offset is strongly recommended due to measurement inaccuracies.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			rss_out : string
					Output RSS FITS file with corrected wavelength pixel table
			logfile : string
					Input ASCII logfile containing the previously measured offset for certain reference emission line
					in dispersion direction
			ref_id : string
					Reference ID under which the offsets are stored in the logfile for this specific RSS
			smooth_poly_cross : string of integer, optional with default: ''
					Degree of the polynomial which is used to smooth the offset value for each reference emission line
					as a function of fiber number (i.e. along cross-disperion direction on the CCD)
					(positiv: normal polynomial, negative: Legandre polynomial)
					No smoothing is performed if this parameter is empty.
			smooth_poly_disp : string of integer, optional with default: ''
					Degree of the polynomial which is used to extrapolated the offsets along the wavelength direction
					for each block of fibers individually. (positiv: normal polynomial, negative: Legandre polynomial)
					A median value of all measured shifts is used if this parameter is empty.
			poly_disp : string of integer (>0), optional with default: '6'
					Degree of polynomial used to construct the wavelength solution. This is needed to properly shift
					the wavelength table according to the offset in pixel units.
			verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Examples
			----------------
			user:> lvmdrp rss correctPixTable RSS_in.fits RSS_out.fits OFFSETWAVE.log RSS_REF_ID poly_disp=7

	"""


	poly_disp = int(poly_disp)
	verbose = int(verbose)

	rss = loadRSS(rss_in)
	log = open(logfile, 'r')
	log_lines = log.readlines()
	m=0
	offsets = []
	for i in range(len(log_lines)):
		if ref_id in log_lines[i]:
			ref_wave=[]
			offsets=[]
			m=1
		if m==1 and len(log_lines[i].split())==4:
			ref_wave.append(float(log_lines[i].split()[0]))
			fibers = numpy.array(log_lines[i+1].split()).astype('float')
			offset_pix = numpy.array(log_lines[i+2].split()).astype('float')
			offset_wave=numpy.array(log_lines[i+3].split()).astype('float')

			if  smooth_poly_cross=='':
				offsets.append(offset_pix)

			else:
				smooth_poly_cross = int(smooth_poly_cross)
				spec = Spectrum1D(data=offset_pix, wave=fibers)
				spec.smoothPoly(order=smooth_poly_cross, ref_base=numpy.arange(rss._fibers))
				if verbose==1:
					pylab.plot(fibers, offset_pix, 'o')
					pylab.plot(numpy.arange(rss._fibers), spec._data)
				offsets.append(spec._data)

		if len(log_lines[i].split())==1 and not (ref_id  in log_lines[i]) and m==1:
			m=0
	if verbose==1:
		pylab.show()
	offsets = numpy.array(offsets)
	ref_wave=numpy.array(ref_wave)
	for i in range(rss._fibers):
		spec = rss[i]
		if smooth_poly_disp=='':
			off = numpy.median(offsets.flatten())
		else:
			smooth_poly_disp=int(smooth_poly_disp)
			if smooth_poly_disp=='':
				off = numpy.median(offsets[i])
			else:
				off = Spectrum1D(wave=ref_wave,  data=offsets[:, i])
				off.smoothPoly(smooth_poly_disp, ref_base=spec._wave)
				if i==-1:
					pylab.plot(ref_wave, offsets[:, i], 'ok')
					pylab.plot(off._wave, off._data, '-r')
					pylab.show()
				off = off._data
		new_wave = Spectrum1D(spec._pixels+off,spec._wave )
		new_wave.smoothPoly(poly_disp, ref_base=spec._pixels)
		spec._wave=new_wave._data
		rss[i]=spec
	rss.writeFitsData(rss_out)

@missing_files(["BAD_CALIBRATION_FRAMES"], "rss_in")
def resampleWave_drp(rss_in, rss_out, method='spline', start_wave='', end_wave='', disp_pix='', err_sim='500', replace_error='1e10', correctHvel='',parallel='auto'):
	"""
			Resamples the RSS with a wavelength in pixel table format to an RSS with a common wavelength solution for each fiber.
			A Monte Carlo scheme can be used to propagte the error to the resample spectrum. Note that correlated noise is not taken
			into account with the procedure.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file where the wavelength is stored as a pixel table
			rss_out : string
					Output RSS FITS file with a common wavelength solution
			method : string, optional with default: 'spline'
					Interpolation scheme used for the spectral resampling of the data.
					Available are
					1. spline
					2. linear
			start_wave : string of float, optional with default: ''
					Start wavelength for the common resampled wavelength solution.
					The "optimal" wavelength will be used if the paramter is empty.
			endt_wave : string of float, optional with default: ''
					End wavelength for the common resampled wavelength solution
					The "optimal" wavelength will be used if the paramter is empty.
			disp_pix : string of float, optional with default: ''
					Dispersion per pixel for the common resampled wavelength solution.
					The "optimal" dispersion will be used if the paramter is empty.
			err_sim : string of integer (>0), optional with default: '500'
					Number of Monte Carlo simulation per fiber in the RSS to estimate the error of the resampled spectrum.
					If err_sim is set to 0, no error will be estimated for the resampled RSS.
			replace_error: strong of float, optional with default: '1e10'
					Error value for bad pixels resampled data, will be ignored if empty
			parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
					Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
					for the given system is used.

			Examples
			----------------
			user:> lvmdrp rss resampleWave RSS_in.fits RSS_out.fits
			user:> lvmdrp rss resampleWave RSS_in.fits RSS_out.fits start_wave=3700.0 end_wave=7000.0 disp_pix=2.0 err_sim=0
	"""
	err_sim=int(err_sim)
	replace_error=float(replace_error)

	rss = loadRSS(rss_in)
	if start_wave=='':
		start_wave=numpy.min(rss._wave)
	else:
		start_wave=float(start_wave)
	if disp_pix=='':
		disp_pix=numpy.min(rss._wave[:, 1:]-rss._wave[:, :-1])
	else:
		disp_pix=float(disp_pix)

	if end_wave=='':
		end_wave=numpy.max(rss._wave)
	else:
		end_wave=float(end_wave)

	if correctHvel=='':
		offset_vel=0.0
	else:
		try:
			offset_vel=float(correctHvel)
		except ValueError:
			offset_vel=rss.getHdrValue(correctHvel)

	ref_wave = numpy.arange(start_wave, end_wave+disp_pix-0.001, disp_pix)
	rss._wave = rss._wave*(1+offset_vel/300000.0)

	data = numpy.zeros((rss._fibers, len(ref_wave)), dtype=numpy.float32)
	if rss._error is not None and err_sim!=0:
		error = numpy.zeros((rss._fibers, len(ref_wave)), dtype=numpy.float32)
	else:
		error = None
	mask = numpy.zeros((rss._fibers, len(ref_wave)), dtype='bool')

	if rss._wave is not None and len(rss._wave.shape)==2:
		if parallel=='auto':
			cpus = cpu_count()
		else:
			cpus = int(parallel)
		if cpus>1:
			pool = Pool(cpus)
			result_spec=[]
			for i in range(rss._fibers):
				spec = rss.getSpec(i)
				result_spec.append(pool.apply_async(spec.resampleSpec, args=(ref_wave, method, err_sim, replace_error)))
	  #      spec.resampleSpec(ref_wave, method, err_sim)
			pool.close()
			pool.join()

		for i in range(rss._fibers):
			if cpus>1:
				spec = result_spec[i].get()
			else:
				spec = rss.getSpec(i)
				spec=spec.resampleSpec(ref_wave, method, err_sim, replace_error)
			data[i, :] = spec._data
			if rss._error is not None and err_sim!=0:
				error[i, :] = spec._error
			mask[i, :] = spec._mask
		resamp_rss = RSS(data=data, wave=ref_wave, header = rss.getHeader(), error=error, mask=mask)

	resamp_rss.writeFitsData(rss_out)

def matchResolution_drp(rss_in, rss_out, targetFWHM, parallel='auto'):
	"""
			Homogenise the spectral resolution of the RSS to a common spectral resolution (FWHM) by smoothing
			with a corresponding Gaussian. A pixel table with the spectral resolution needs to be present in the RSS.
			If the spectral resolution is higher than than the target spectral resolution for certain pixel, no smoothing
			is applied for those pixels.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file with a pixel table for the spectral resolution
			rss_out : string
					Output RSS FITS file with a homogenised spectral resolution
			targetFWHM : string of float
					Spectral resolution in FWHM to which the RSS shall be homogenised
			parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
					Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
					for the given system is used.

			Examples
			----------------
			user:> lvmdrp rss matchResolution RSS_in.fits RSS_out.fits 6.0
	"""
	targetFWHM = float(targetFWHM)
	rss = RSS()
	rss.loadFitsData(rss_in)

	smoothFWHM = numpy.zeros_like(rss._inst_fwhm)
	select = rss._inst_fwhm<targetFWHM
	smoothFWHM[select] = numpy.sqrt(targetFWHM**2-rss._inst_fwhm[select]**2)


	if parallel=='auto':
		cpus= cpu_count()
	else:
		cpus=int(parallel)

	if cpus>1:
		pool = Pool(cpus)
		threads=[]
		for i in range(len(rss)):
			spec = rss[i]
			threads.append(pool.apply_async(rss[i].smoothGaussVariable, ([smoothFWHM[i, :]])))

		for i in range(len(rss)):
			rss[i] = threads[i].get()
		pool.close()
		pool.join()
	else:
		for i in range(len(rss)):
			rss[i]=rss[i].smoothGaussVariable(smoothFWHM[i, :])
	rss._inst_fwhm=None
	rss.setHdrValue('hierarch PIPE SPEC RES', targetFWHM, 'FWHM in A of spectral resolution')
	rss.writeFitsData(rss_out)

def constructSkySpec_drp(rss_in, sky_out, clip_sigma='3.0', nsky='0', filter='', non_neg='1', plot='0'):
	"""
			Creates a average (sky) spectrum from the RSS, which stored either as a FITS or an ASCII file.
			Spectra may be rejected from the median computation. Bad pixel in the RSS are not included
			in the median computation.

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
			plot : string of integer (0 or 1)
					If set to 1, the sky spectrum will be display on screen.

			Examples
			----------------
			user:> lvmdrp rss constructSkySpec RSS_IN.fits SKY_OUT.fits 3.0
			user:> lvmdrp rss constructSkySpec RSS_IN.fits SKY_OUT.txt
	"""
	clip_sigma=float(clip_sigma)
	nsky = int(nsky)
	non_neg = int(non_neg)
	plot = int(plot)
	filter=filter.split(',')
	rss = RSS()
	rss.loadFitsData(rss_in)
	median = numpy.zeros(len(rss), dtype=numpy.float32)
	for i in range(len(rss)):
		spec = rss[i]
		##pylab.plot(spec._wave, spec._data)
		if spec._mask is not None:
			if numpy.sum(numpy.logical_not(spec._mask))!=0:
				median[i] = numpy.median(spec._data[numpy.logical_not(spec._mask)])
			else:
				median[i]=0
		else:
			median[i] = numpy.median(spec._data)
	select_good = median!=0
	if clip_sigma>0.0 and nsky==0:
		select = numpy.logical_and(numpy.logical_and(median<numpy.median(median[select_good])+clip_sigma*numpy.std(median[select_good])/2.0, median>numpy.median(median[select_good])-clip_sigma*numpy.std(median[select_good])/2.0), select_good)
		sky_fib = numpy.sum(select)
	elif nsky>0:
		idx=numpy.argsort(median[select_good])
		max_value = numpy.max(median[select_good][idx[:nsky]])
		if non_neg==1:
			select = (median<=max_value) & (median>0.0)
		else:
			select = (median<=max_value)
		sky_fib = numpy.sum(select)
	rss.setHdrValue('hierarch PIPE NSKY FIB', sky_fib, 'Number of averaged sky fibers')
	subRSS = rss.subRSS(select)
	print(filter[0])
	if filter[0]!='':
		passband = PassBand()
		passband.loadTxtFile(filter[0], wave_col=int(filter[1]),  trans_col=int(filter[2]))
		(flux_rss, error_rss, min_rss, max_rss, std_rss) = passband.getFluxRSS(subRSS)
		mag_flux = numpy.zeros(len(flux_rss))
		for m in range(len(flux_rss)):
			if flux_rss[m]>0.0:
				mag_flux[m] = passband.fluxToMag(flux_rss[m], system='Vega')
		mag_mean = numpy.mean(mag_flux[mag_flux>0.0])
		mag_min = numpy.min(mag_flux[mag_flux>0.0])
		mag_max = numpy.max(mag_flux[mag_flux>0.0])
		mag_std = numpy.std(mag_flux[mag_flux>0.0])
		rss.setHdrValue('hierarch PIPE SKY MEAN', float('%.2f'%mag_mean), 'Mean sky brightness of sky fibers')
		rss.setHdrValue('hierarch PIPE SKY MIN', float('%.2f'%mag_min), 'Minium sky brightness of sky fibers')
		rss.setHdrValue('hierarch PIPE SKY MAX', float('%.2f'%mag_max), 'Maximum sky brightness of sky fibers')
		rss.setHdrValue('hierarch PIPE SKY RMS', float('%.2f'%mag_std), 'RMS sky brightness of sky fibers')

	rss.writeFitsHeader(rss_in)
	skySpec = subRSS.create1DSpec()

	if plot==1:
		pylab.plot(skySpec._wave, skySpec._data, 'ok')
		pylab.show()
	if '.fits' in sky_out:
		skySpec.writeFitsData(sky_out)
	if '.txt' in sky_out:
		skySpec.writeTxtData(sky_out)

def subtractSkySpec_drp(rss_in, rss_out, sky, factor='1', scale_region='', scale_ind=False,parallel='auto'):
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
			parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
					Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
					for the given system is used.

			Examples
			----------------
			user:> lvmdrp rss subtractSkySpec RSS_IN.fits RSS_OUT.fits SKY_SPEC.fits
	"""

	factor=numpy.array(factor).astype(numpy.float32)
	scale_ind = bool(scale_ind)
	if scale_region!='':
		region = scale_region.split( ',')
		wave_region=[float(region[0]), float(region[1])]
	rss = RSS()
	rss.loadFitsData(rss_in)
	sky_spec = Spectrum1D()
	sky_spec.loadFitsData(sky)
	def optimize_sky(factor, test_spec, sky_spec, start_wave, end_wave):
		wave = test_spec._wave
		if test_spec._mask is not None:
			good_pix = numpy.logical_not(test_spec._mask)
			select1 = numpy.logical_and(wave>start_wave, wave<end_wave)
			if numpy.sum(good_pix[select1])>1:
				select = numpy.logical_and(select1, good_pix)
			else:
				select = select1
		else:
			select = numpy.logical_and(wave>start_wave, wave<end_wave)
		rms = numpy.std(test_spec._data[select]-sky_spec._data[select]*factor)
		return rms

	if numpy.all(rss._wave==sky_spec._wave) and scale_region!='':
		factors=numpy.zeros(len(rss), dtype=numpy.float32)
		for i in range(len(rss)):
			optimum= optimize.fmin(optimize_sky, [1.0], args=(rss[i], sky_spec, wave_region[0], wave_region[1]), disp=0)
			factors[i]=optimum[0]
		select_good = factors>0.0
		scale_factor = numpy.median(factors[select_good])
		for i in range(len(rss)):
			if scale_ind==True:
				rss[i] = old_div(rss[i],factors[i])-sky_spec
			else:

				if factors[i]>0:
					rss[i] = rss[i]-sky_spec*numpy.median(factors[select_good])



	elif  numpy.all(rss._wave==sky_spec._wave) and scale_region=='':
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
			rss[i] = rss[i]-threads[i].get()


		#if i==0:
			#pylab.plot(threads[i].get()._wave, threads[i].get()._data, '-r')
			#pylab.show()

	if scale_region!='':
		rss.setHdrValue('hierarch PIPE SKY SCALE',float('%.3f'%scale_factor),'sky spectrum scale factor')
	rss.writeFitsData(rss_out)

def splitFibers_drp(rss_in, splitted_out, contains):
	"""
			Subtracts a (sky) spectrum, which was stored as a FITS file, from the whole RSS.
			The error will be propagated if the spectrum AND the RSS contain error information.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file including a position table as an extension
			splitted_out : string
					Comma-separated list of output RSS FITS files
			contains : string
					Comma-Separated list of fiber "types" included in the respective output RSS file.
					Available fiber types are OBJ, SKY and CAL, corresponding to target fibers, dedicated
					sky fibers and calibration fibers, respectively.
					If more than one type of fibers should be contained in one of the splitted RSS, they need to
					be ";" separated.

			Examples
			----------------
			user:> lvmdrp rss splitFibers RSS_IN.fits RSS_OBJ.fits,RSS_SKY.fits SKY,OBJ
			user:> lvmdrp rss splitFibers RSS_IN.fits RSS_OBJ_SKY.fits,RSS_CAL.fits SKY;OBJ,SKY
	"""
	contains = contains.split(',')
	splitted_out = splitted_out.split(',')
	rss = RSS()
	rss.loadFitsData(rss_in)
	splitted_rss = rss.splitFiberType(contains)
	for i in range(len(splitted_rss)):
		splitted_rss[i].writeFitsData(splitted_out[i])

def createFiberFlat_drp(rss_in, rss_out, smooth_poly='0', clip='', valid=''):
	"""
			Creates a fiberflat from a wavelength calibrated skyflat RSS by computing the
			relative transmission to the median spectrum.


			Parameters
			--------------
			rss_in : string
					Input RSS FITS of a skyflat observations
			rss_out : string
					Output RSS FITS file with fiberflat RSS
			smooth_poly : string of integer (>0), optional with default: '-5'
					Degree of polynomial with which the fiberflat may be fitted along
					the dispersion axis for each fiber indepenently. (positiv: normal polynomial, negative: Legandre polynomial)
					However, not recommended....
			clip : string of two comma separated floats, optional with default: ''
					Minimum and maximum number of relative transmission in the resulting fiberflat. If some value are below or above
					the given limits they are replaced by zeros and added to the mask as bad pixels.
			valid : string of two comma separated integers, optional with default: ''
					Minimum and maximum fiber number used to create the reference median spectrum.
					This is mainly required if there is a wavelength dependent vignetting effect, so that those
					fibers can be rejected from the median spectrum.

			Examples
			----------------
			user:> lvmdrp rss createFiberFlat RSS_IN.fits FIBERFLAT.fits clip=0.3,1.5 valid=100,250
			user:> lvmdrp rss createFiberFlat RSS_IN.fits FIBERFLAT.fits -6 clip=0.1,2.0
	"""
	smooth_poly=int(smooth_poly)
	if valid=='':
		valid=None
	else:
		valid=numpy.array(valid.split(',')).astype('int16')
	if clip!='':
		clip = clip.split(',')
		clip[0] = float(clip[0])
		clip[1] = float(clip[1])
	else:
		clip=None

	rss = loadRSS(rss_in)
#    print rss._wave
	fiberflat=rss.createFiberFlat(smooth_poly, clip, valid=valid)

	# perform some statistic about the fiberflat
	if fiberflat._mask is not None:
		select = fiberflat._mask==False
	else:
		select = fiberflat._data==fiberflat._data
	min = numpy.min(fiberflat._data[select])
	max = numpy.max(fiberflat._data[select])
	mean = numpy.mean(fiberflat._data[select])
	median = numpy.median(fiberflat._data[select])
	std = numpy.std(fiberflat._data[select])

	fiberflat.setHdrValue('hierarch PIPE FLAT MIN', float('%.3f'%(min)), 'Mininum fiberflat value')
	fiberflat.setHdrValue('hierarch PIPE FLAT MAX', float('%.3f'%(max)), 'Maximum fiberflat value')
	fiberflat.setHdrValue('hierarch PIPE FLAT AVR', float('%.2f'%(mean)), 'Mean fiberflat value')
	fiberflat.setHdrValue('hierarch PIPE FLAT MED', float('%.2f'%(median)), 'Median fiberflat value')
	fiberflat.setHdrValue('hierarch PIPE FLAT STD', float('%.3f'%(std)), 'rms of fiberflat values')

	if fiberflat is None:
		print('Please resample the RSS frame to a common wavelength solution!')
	else:
		fiberflat.writeFitsData(rss_out)

def correctTraceMask_drp(trace_in, trace_out, logfile, ref_file, poly_smooth=''):
	"""
		   Corrects the trace mask of the central fiber position for possible offsets in cross-dispersion direction due to
		   flexure effects with respect to the calibration frames taken for this object. The offfsets need to be determined
		   beforehand via the offsetTrace task. The offsets can be smoothed and/or extrapolated along the dispersion axis.

			Parameters
			--------------
			trace_in : string
					Input RSS FITS file containing the traces of the fiber position on the CCD
			trace_out : string
					Output RSS FITS file with offset corrected fiber position traces
			logfile : string
					Input ASCII logfile containing the previously measured offset for certain reference emission line
					in cross-dispersion direction
			ref_file : string
					Reference file under which the offsets are stored in the logfile for this specific RSS
			poly_smooth: string of integer, optional with default: ''
					Degree of the polynomial which is used to smooth/extrapolate the offsets as a function
					of wavelength (positiv: normal polynomial, negative: Legandre polynomial)
					No smoothing is performed if this parameter is empty an a median offset is used instead.

			Examples
			----------------
			user:> lvmdrp rss correctTraceMask TRACE_IN.fits TRACE_OUT.fits OFFSET_TRACE.log REF_File_name
			user:> lvmdrp rss correctTraceMask TRACE_IN.fits TRACE_OUT.fits OFFSET_TRACE.log REF_File_name poly_smooth= -6
	"""
	log = open(logfile, 'r')
	log_lines = log.readlines()
	l=0
	while l<len(log_lines):
		split = log_lines[l].split()
		z=1
		if len(split)==1 and split[0]==ref_file:

			offsets=[]
			lines=[]
			cross_pos=[]
			disp_pos=[]
			while l+z<len(log_lines):
				split1 = log_lines[l+z].split()
				split2 = log_lines[l+z+1].split()
				split3 = log_lines[l+z+2].split()
				if len(split1)>1:
					offsets.append(numpy.array(split3[1:]).astype('float32'))
					cross_pos.append(numpy.array(split1[1:]).astype('float32'))
					disp_pos.append(numpy.array(split2[1:]).astype('float32'))
					lines.append(float(split3[0]))
				else:
					incomplete=False
					break
				z+=3
			break
		l+=1

		log.close()
	offsets= numpy.array(offsets)
	cross_pos = numpy.array(cross_pos)
	disp_pos = numpy.array(disp_pos)
	trace = FiberRows()
	trace.loadFitsData(trace_in)

	if poly_smooth=='':
		trace = trace+(numpy.median(offsets.flatten())*-1)
	else:
		split_trace = trace.split(offsets.shape[1], axis='y')
		for j in range(len(split_trace)):
			offset_spec = Spectrum1D(wave=disp_pos[:, j], data=offsets[:, j])
			wave=numpy.arange(trace._data.shape[1])
			offset_spec.smoothPoly(order=int(poly_smooth), start_wave=wave[0], end_wave=wave[-1], ref_base=wave)
			if j>0:
				corr_trace=split_trace[j]+(offset_spec*-1)
				offset_trace.append(corr_trace)
			else:
				offset_trace=split_trace[j]+(offset_spec*-1)

		trace=offset_trace

	trace.writeFitsData(trace_out)

def correctFiberFlat_drp(rss_in, rss_out, fiberflat, clip='0.2'):
	"""
		   Correct an RSS frame for the effect of the different fiber transmission as measured by a fiberflat.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			rss_out : string
					Output RSS FITS file which is fiberflat corrected.
			fiberflat : string
					Fiberflat RSS FITS file containing the relative transmission of each fiber
			clip : string of float, optional with default: ''
					Minimum relative transmission considered for the used fiberflat. Value below the given limits are replaced
					by zeros and added to the mask as bad pixels in the output RSS.

			Examples
			----------------
			user:> lvmdrp rss correctFiberFlat RSS_IN.fits RSS_OUT.fits FIBERFLAT_IN.fits
			user:> lvmdrp rss correctFiberFlat RSS_IN.fits RSS_OUT.fits FIBERFLAT_IN.fits clip='0.4'
	"""
	clip=float(clip)
	rss = RSS()
	rss.loadFitsData(rss_in)
	flat = RSS()
	flat.loadFitsData(fiberflat)

	for i in range(flat._fibers):
		spec_flat = flat.getSpec(i)
		spec_data = rss.getSpec(i)
		flat_resamp=spec_flat.resampleSpec(spec_data._wave, err_sim=0)
		select_clip=numpy.logical_or((flat_resamp<clip) , (numpy.isnan(flat_resamp._data)))
		flat_resamp._data[select_clip]=0
		flat_resamp._mask[select_clip]=True
		spec_new = old_div(spec_data,flat_resamp)
		rss.setSpec(i, spec_new)
	rss.writeFitsData(rss_out)

def createSensFunction_drp(rss_in, out_sens,  ref_spec, airmass, exptime, smooth_poly='5', smooth_ref='6.0', smooth_ref2='6.0', median_filt='0',coadd='1', extinct_v='0.0', extinct_curve='mean', aper_correct='1.0',  ref_units='1e-16', target_units='1e-16',column_wave='0', column_flux='1', delimiter='', header='1' , split='', mask_wave='', mask_telluric='', overlap='100', out_star='', verbose='0'):
	smooth_poly=int(smooth_poly)
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
	if mask_wave!='':
		mask_wave = numpy.array(mask_wave.split(',')).astype('float32')
	else:
		mask_wave=None
		
	if mask_telluric!='':
		mask_telluric = numpy.array(mask_telluric.split(',')).astype('float32')
	else:
		mask_telluric=None
	verbose=int(verbose)


	ref_star_spec = Spectrum1D()

	if coadd>0:
		rss = RSS()
		rss.loadFitsData(rss_in)
		select = rss.selectSpec(min=0, max=coadd, method='median')
		star_rss=rss.subRSS(select)
		star_spec = old_div(star_rss.create1DSpec(method='sum'),aper_correct)
	else:
		star_spec = Spectrum1D()
		if '.fits' in rss_in:
			star_spec.loadFitsData(rss_in)
		elif '.txt' in rss_in:
			star_spec.loadTxtData(rss_in)

	try:
		extinct_v = rss.getHdrValue(extinct_v)
	except:
		extinct_v= float(extinct_v)

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
	if out_star!='':
		star_out = open(out_star, 'w')
		for i in range(star_spec._dim):
			star_out.write('%i %.3f %e\n'%(i, star_spec._wave[i], star_spec._data[i]))
		star_out.close()

	star_spec.smoothSpec(smooth_ref)
	print(exptime,extinct._wave,star_spec._wave)
	star_corr = old_div(old_div(star_spec,extinct),exptime)

	sens_func = old_div(ref_star_resamp,star_corr)
	if mask_wave is not None:
		regions = old_div(len(mask_wave),2)
		for i in range(regions):
			select_region = numpy.logical_and(sens_func._wave>mask_wave[i*2], sens_func._wave<mask_wave[i*2+1])
			select_blue = numpy.logical_and(sens_func._wave>mask_wave[i*2]-20, sens_func._wave<mask_wave[i*2])
			select_red = numpy.logical_and(sens_func._wave>mask_wave[i*2+1], sens_func._wave<mask_wave[i*2+1]+20)
			line_par = stats.linregress([mask_wave[i*2]-10,mask_wave[i*2+1]+10], [numpy.median(sens_func._data[select_blue]), numpy.median(sens_func._data[select_red])])

			sens_func._data[select_region] = (line_par[0]*sens_func._wave[select_region]+line_par[1]).astype('float32')
			#select = numpy.logical_and(sens_func._wave>mask_wave[i*2], sens_func._wave<mask_wave[i*2+1])
			#sens_func._mask[select]=True
	if mask_telluric is not None:
		star_telluric1 = star_rss.create1DSpec(method='sum')
		star_telluric2 = star_rss.create1DSpec(method='sum')
		regions = old_div(len(mask_telluric),2)
		for i in range(regions):
			select_region = numpy.logical_and(star_telluric1._wave>mask_telluric[i*2], star_telluric1._wave<mask_telluric[i*2+1])
			select_blue = numpy.logical_and(star_telluric1._wave>mask_telluric[i*2]-20, star_telluric1._wave<mask_telluric[i*2])
			select_red = numpy.logical_and(star_telluric1._wave>mask_telluric[i*2+1], star_telluric1._wave<mask_telluric[i*2+1]+20)
			line_par = stats.linregress([mask_telluric[i*2]-10,mask_telluric[i*2+1]+10], [numpy.median(star_telluric1._data[select_blue]), numpy.median(star_telluric1._data[select_red])])
			star_telluric2._data[select_region] = (line_par[0]*star_telluric1._wave[select_region]+line_par[1]).astype('float32')
		telluric_spec = (old_div(star_telluric1, star_telluric2))**(1.0/airmass)
		telluric_spec.writeFitsData('telluric_spec.fits')
	good_pix = numpy.logical_not(sens_func._mask)
	if median_filt>0:
		sens_func.smoothSpec(median_filt,method='median')
	if verbose==1:
		pylab.plot(sens_func._wave[good_pix][10:-10], sens_func._data[good_pix][10:-10], '-k')
	if split=='':
		mask = sens_func._mask
		#mask[:10]=True
		#mask[-10:]=True
		sens_func_smooth = 1.0/Spectrum1D(wave=sens_func._wave, data=sens_func._data, mask=mask)
		sens_func_smooth.smoothPoly(smooth_poly)
		sens_func_smooth = 1.0/sens_func_smooth
		if verbose==1:
			pylab.plot(sens_func_smooth._wave,  sens_func_smooth._data, '-r')
			pylab.plot(sens_func_smooth._wave,  old_div(sens_func._data,sens_func_smooth._data), '-g')
			sens_test_out = open('test_sens.txt', 'w')
			for i in range(sens_func_smooth._dim):
				sens_test_out.write('%i %.2f %e %e %e\n'%(i, sens_func_smooth._wave[i], sens_func._data[i], sens_func_smooth._data[i], old_div(sens_func._data[i],sens_func_smooth._data[i])))
			sens_test_out.close()
	else:
		split = float(split)
		overlap = float(overlap)
		select = sens_func._wave>split
		mask = sens_func._mask[select]
		mask[-10:]=True
		sens_func_smooth2 = Spectrum1D(wave=sens_func._wave[select], data=sens_func._data[select], mask=mask)
		sens_func_smooth2.smoothPoly(smooth_poly)

		select = sens_func._wave<split+overlap
		mask = sens_func._mask[select]
		mask[-10:]=True
		sens_func_smooth1 = Spectrum1D(wave=sens_func._wave[select], data=sens_func._data[select], mask=mask)
		sens_func_smooth1.smoothPoly(smooth_poly)

		if verbose==1:
			pylab.plot(sens_func_smooth1._wave,  sens_func_smooth1._data, '-r')
			pylab.plot(sens_func_smooth2._wave,  sens_func_smooth2._data, '-r')
	if verbose==1:
		pylab.show()

	# need to replace with XML output
	out = open(out_sens, 'w')

	if split=='':
		for i in range(sens_func._dim):
			out.write('%i %.3f %e\n'%(i,  sens_func_smooth._wave[i], sens_func_smooth._data[i]))
	else:
		min = numpy.argmin(numpy.abs(sens_func_smooth1._data[sens_func_smooth1._wave>split]-sens_func_smooth2._data[sens_func_smooth2._wave<split+overlap]))
	#    print min
		start2 = numpy.sum(sens_func._wave<=split)
		change = min+start2
		for i in range(sens_func._dim):
			if i<change:
				out.write('%i %.3f %e\n'%(i, sens_func_smooth1._wave[i], sens_func_smooth1._data[i]))
			else:
				out.write('%i %.3f %e\n'%(i, sens_func_smooth2._wave[i-start2], sens_func_smooth2._data[i-start2]))

	out.close()

def createSensFunction2_drp(rss_in, out_sens, ref_spec, airmass, exptime, smooth_bspline='0.3', smooth_ref='6.0', smooth_ref2='6.0', median_filt='0',coadd='1', extinct_v='0.0', extinct_curve='mean', aper_correct='1.0',  ref_units='1e-16', target_units='1e-16',column_wave='0', column_flux='1', delimiter='', header='1' , mask_wave='', out_star='', verbose='0'):
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
	if mask_wave!='':
		mask_wave = numpy.array(mask_wave.split(',')).astype('float32')
	else:
		mask_wave=None
	verbose=int(verbose)

	rss = RSS()
	if coadd>0:

		rss.loadFitsData(rss_in)
		select = rss.selectSpec(min=0, max=coadd, method='median')
		star_rss=rss.subRSS(select)
		star_spec = old_div(star_rss.create1DSpec(method='sum'),aper_correct)
	else:
		star_spec = Spectrum1D()
		if '.fits' in rss_in:
			star_spec.loadFitsData(rss_in)
		elif '.txt' in rss_in:
			star_spec.loadTxtData(rss_in)

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
	if out_star!='':
		star_out = open(out_star, 'w')
		for i in range(star_spec._dim):
			star_out.write('%i %.3f %e\n'%(i, star_spec._wave[i], star_spec._data[i]))
		star_out.close()

	star_spec.smoothSpec(smooth_ref)
	star_corr = old_div(old_div(star_spec,extinct),exptime)

	sens_func = old_div(ref_star_resamp,star_corr)
	if mask_wave is not None:
		regions = old_div(len(mask_wave),2)
		for i in range(regions):
			select_region = numpy.logical_and(sens_func._wave>mask_wave[i*2], sens_func._wave<mask_wave[i*2+1])
			select_blue = numpy.logical_and(sens_func._wave>mask_wave[i*2]-20, sens_func._wave<mask_wave[i*2])
			select_red = numpy.logical_and(sens_func._wave>mask_wave[i*2+1], sens_func._wave<mask_wave[i*2+1]+20)
			line_par = stats.linregress([mask_wave[i*2]-10,mask_wave[i*2+1]+10], [numpy.median(sens_func._data[select_blue]), numpy.median(sens_func._data[select_red])])

			sens_func._data[select_region] = (line_par[0]*sens_func._wave[select_region]+line_par[1]).astype('float32')

	good_pix = numpy.logical_not(sens_func._mask)
	if median_filt>0:
		sens_func.smoothSpec(median_filt,method='median')
	if verbose==1:
		pylab.plot(sens_func._wave[good_pix][10:-10], 1.0/sens_func._data[good_pix][10:-10], '-k')

	mask = sens_func._mask
	#mask[:10]=True
	#mask[-10:]=True
	sens_func_smooth = Spectrum1D(wave=sens_func._wave, data=1.0/sens_func._data, mask=mask)
	sens_func_smooth.smoothSpec(smooth_bspline,method='BSpline')
	if verbose==1:
		pylab.plot(sens_func_smooth._wave,  sens_func_smooth._data, '-r')
		pylab.plot(sens_func_smooth._wave,  old_div((1.0/sens_func._data),sens_func_smooth._data), '-g')
		sens_test_out = open('test_sens.txt', 'w')
		for i in range(sens_func_smooth._dim):
			sens_test_out.write('%i %.2f %e %e %e\n'%(i, sens_func_smooth._wave[i], sens_func._data[i], sens_func_smooth._data[i], old_div(sens_func._data[i],sens_func_smooth._data[i])))
		sens_test_out.close()
		pylab.show()
	sens_func_smooth = 1.0/sens_func_smooth


	# need to replace with XML output
	out = open(out_sens, 'w')
	for i in range(sens_func._dim):
		out.write('%i %.3f %e\n'%(i,  sens_func_smooth._wave[i], sens_func_smooth._data[i]))
	out.close()

def fluxCalibration_drp(rss_in, rss_out, sens_func, airmass, exptime, extinct_v='0.0', extinct_curve='mean', ref_units='1e-16', target_units='1e-16', norm_sb_fib=''):
	ref_units=float(ref_units)
	target_units=float(target_units)
	rss = RSS()
	rss.loadFitsData(rss_in)
	if norm_sb_fib=='':
		norm_sb_fib=1.0
	else:
		norm_sb_fib=numpy.pi*float(norm_sb_fib)**2
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

	sens_in = open(sens_func, 'r')
	lines = sens_in.readlines()
	wave_sens = numpy.zeros(len(lines), dtype=numpy.float32)
	sens_dat = numpy.zeros(len(lines), dtype=numpy.float32)
	for i in range(len(lines)):
		line = lines[i].split()
		if len(line)==3:
			wave_sens[i]=float(line[1])
			sens_dat[i]=float(line[2])
		elif len(line)==2:
			wave_sens[i]=float(line[0])
			sens_dat[i]=float(line[1])
	sens_func = Spectrum1D(wave=wave_sens, data=sens_dat)
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
		sens_func_resamp = sens_func.resampleSpec(rss._wave, method='spline')

		for j in range(rss._fibers):
			rss[j] = (old_div(old_div(old_div(rss[j],extinct),exptime),norm_sb_fib))*sens_func_resamp*(old_div(ref_units,target_units))
	#        print exptime
	rss.writeFitsData(rss_out)

def combineRSS_drp(rsss, rss_out, method='mean'):
	# convert input parameters to proper type
	list_rss= rsss.split(',')

	rss_list=[]
	for i in list_rss:
		#load subimages from disc and append them to a list
		rss= loadRSS(i)
		rss_list.append(rss)
	combined_header = combineHdr(rss_list)
	combined_rss = RSS()
	combined_rss.combineRSS(rss_list, method=method)
	combined_rss.setHeader(header=combined_header._header)
	#write out FITS file
	combined_rss.writeFitsData(rss_out)

def glueRSS_drp(rsss,rss_out):
	list_rss= rsss.split(',')
	glueRSS(list_rss,rss_out)

def apertureFluxRSS_drp(rss_in, center_x, center_y, hdr_prefix, arc_radius, flux_type='mean,3900,4600'):
	flux_type=flux_type.split(',')
	center_x = float(center_x)
	center_y = float(center_y)
	arc_radius=float(arc_radius)
	#load subimages from disc and append them to a list
	rss= loadRSS(rss_in)

	spec = rss.createAperSpec(center_x, center_y, arc_radius)
	if flux_type[0]=='mean' or flux_type[0]=='sum' or flux_type[0]=='median':
		start_wave = float(flux_type[1])
		end_wave = float(flux_type[2])
		flux_spec = spec.collapseSpec(method=flux_type[0], start=start_wave, end=end_wave)

		rss.setHdrValue(hdr_prefix+' APER FLUX', flux_spec[0], flux_type[0]+' flux from %.0f to %.0f'%(start_wave, end_wave))
		rss.setHdrValue(hdr_prefix+' APER ERROR', flux_spec[1], flux_type[0]+' error from %.0f to %.0f'%(start_wave, end_wave))
	else:
		passband = PassBand()
		passband.loadTxtFile(flux_type[0], wave_col=int(flux_type[1]),  trans_col=int(flux_type[2]))
		flux_spec=passband.getFluxPass(spec)
		#  print flux_spec
		rss.setHdrValue(hdr_prefix+' APER FLUX', float('%.3f'%flux_spec[0]), flux_type[0].split('/')[-1].split('.')[0]+' band flux (%.1farcsec diameter)'%(2*arc_radius) )
		if flux_spec[1] is not None:
			rss.setHdrValue(hdr_prefix+' APER ERR', float('%.3f'%flux_spec[1]), flux_type[0].split('/')[-1].split('.')[0]+' band error (%.1farcsec diameter)'%(2*arc_radius) )
	rss.writeFitsData(rss_in)

def matchFluxRSS_drp(rsss, center_x, center_y, hdr_prefixes, arc_radius, start_wave='3800', end_wave='4600', polyorder='2', verbose='0'):
	verbose=int(verbose)
	list_rss= rsss.split(',')
	center_x = float(center_x)
	center_y = float(center_y)
	if start_wave!='':
		start_wave=float(start_wave)
	else:
		start_wave=None
	if end_wave!='':
		end_wave=float(end_wave)
	else:
		end_Wave=None
	hdr_prefixes=hdr_prefixes.split(',')
	arc_radius=float(arc_radius)
	polyorder=int(polyorder)
	specs = []
	fluxes =[]
	for i in range(len(list_rss)):
		#load subimages from disc and append them to a list
		rss= loadRSS(list_rss[i])
		specs.append(rss.createAperSpec(center_x, center_y, arc_radius))
		fluxes.append(numpy.median(specs[i]._data))


	order = numpy.argsort(fluxes)
 #   print fluxes, order
	for i in range(len(list_rss)):
		rss=loadRSS(list_rss[i])
		ratio = old_div(specs[order[-1]],specs[i])
		coeff=ratio.smoothPoly(order=polyorder, start_wave=start_wave, end_wave=end_wave)
		rss=rss*ratio
		rss._data=rss._data.astype(numpy.float32)
		if rss._error is not None:
			rss._error=rss._error.astype(numpy.float32)
		if start_wave is not None:
			rss.setHdrValue(hdr_prefixes[i]+' RELFLUX START',  start_wave, 'Start wave for poly fit')
		if end_wave is not None:
			rss.setHdrValue(hdr_prefixes[i]+' RELFLUX END',  end_wave, 'End wave for poly fit')
		for m in range(len(coeff)):
			rss.setHdrValue(hdr_prefixes[i]+' RELFLUX POLY%i'%(m), '%.3E'%(coeff[len(coeff)-1-m]), 'Polynomial coefficient')

		rss.writeFitsData(list_rss[i])
		if verbose==1:
			pylab.plot(specs[i]._wave,(old_div(specs[order[-1]],specs[i]))._data,'-k')
			pylab.plot(specs[i]._wave,ratio._data,'-r')
			#pylab.plot((specs[i])._data,'-k')
			#pylab.plot((specs[i]*ratio)._data,'-r')
	if verbose==1:
		pylab.show()

@missing_files(["BAD_CALIBRATION_FRAMES"], "rss_in", "position_table")
def includePosTab_drp(rss_in, position_table,  offset_x='0.0', offset_y='0.0'):
	"""
		   Adds an ASCII file position table as a FITS table extension to the RSS file.
		   An offset may be applied to the fiber positions in x and y direction independently.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file for which the position table will be added
			position_table : string
					Input position table ASCII file name
			offset_x : string of float, optional with default: '0.0'
					Offset applied to the fiber positions in x direction before being added to the RSS.
			offset_y : string of float, optional with default: '0.0'
					Offset applied to the fiber positions in y direction before being added to the RSS.

			Examples
			----------------
			user:> lvmdrp rss includePosTab RSS.fits POSTAB.txt
			user:> lvmdrp rss includePosTab RSS.fits POSTAB.txt  offset_x=-5.0 offset_y=3.0
	"""
	offset_x=float(offset_x)
	offset_y=float(offset_y)
	rss = RSS()
	rss.loadFitsData(rss_in)
	rss.loadTxtPosTab(position_table)
	rss.offsetPosTab(offset_x, offset_y)
	rss.writeFitsData(rss_in)

def copyPosTab_drp(rss_in, rss_out):
	"""
		 Copies the position table FITS extension from one RSS to another RSS FITS file.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file from which the position table will be taken.
			rss_out : string
					Output RSS FITS file (must already exist) in which the position table will be added.

			Examples
			----------------
			user:> lvmdrp rss copyPosTab RSS1.fits RSS2.fits
	"""
	rss1 = RSS()
	rss1.loadFitsData(rss_in)
	rss2 = RSS()
	rss2.loadFitsData(rss_out)
	rss2._shape = rss1._shape
	rss2._size= rss1._size

	rss2._arc_position_x = rss1._arc_position_x
	rss2._arc_position_y = rss1._arc_position_y
	rss2._good_fibers = rss1._good_fibers
	rss2._fiber_type = rss1._fiber_type
	rss2.writeFitsData(rss_out)

def offsetPosTab_drp(rss_in, offset_x, offset_y):
	"""
			Applies an offset to the fiber positions in x and y direction independently.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file in which the position table will be changed by fiber offsets
			offset_x : string of float, optional with default: '0.0'
					Offset applied to the fiber positions in x direction.
			offset_y : string of float, optional with default: '0.0'
					Offset applied to the fiber positions in y direction.

			Examples
			----------------
			user:> lvmdrp rss offsetPosTab RSS.fits offset_x=-5.0 offset_y=3.0
	"""
	offset_x = float(offset_x)
	offset_y = float(offset_y)
	rss = RSS()
	rss.loadFitsData(rss_in)
	rss.offsetPosTab(offset_x, offset_y)
	rss.writeFitsData(rss_in)

def rotatePosTab_drp(rss_in, angle='0.0'):
	"""
			Applies an offset to the fiber positions in x and y direction independently.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file in which the position table will be rotated around the bundle zero-point 0,0 by an angle
			angle : string of float, optional with default: '0.0'
					Angle applied to rotate the fiber positions counter-clockwise

			Examples
			----------------
			user:> lvmdrp rss  RSS.fits rotate=152.0
	"""
	angle = float(angle)
	rss = loadRSS(rss_in)
	new_pos=rss.rotatePosTab(angle)
	rss.setPosTab(new_pos)
	rss.writeFitsData(rss_in)

def createCube_drp(rss_in, cube_out, position_x='', position_y='', ref_pos_wave='', int_ref='1', mode='inverseDistance', resolution='1.0', sigma='1.0', radius_limit='5.0', min_fibers='3', slope='2', bad_threshold='0.01',replace_error='1e10', flip_x='0', flip_y='0', full_field='0', store_cover='0', parallel='auto', verbose='0'):
	resolution=float(resolution)
	sigma = float(sigma)
	radius_limit=float(radius_limit)
	min_fibers=int(min_fibers)
	slope = float(slope)
	bad_threshold=float(bad_threshold)
	flip_x=int(flip_x)
	flip_y=int(flip_y)
	int_ref=int(int_ref)
	replace_error = float(replace_error)
	verbose=int(verbose)
	store_cover=bool(store_cover)
	if position_x=='':
		pos_x=None
	else:
		#pos_x = Spectrum1D()
		#pos_x.loadFitsData(position_x)
		pos_x=loadRSS(position_x)
	if position_y=='':
		pos_y=None
	else:
		#pos_y = Spectrum1D()
		#pos_y.loadFitsData(position_y)
		pos_y=loadRSS(position_y)
	if ref_pos_wave!='':
		ref_pos_wave=float(ref_pos_wave)
	else:
		ref_pos_wave=None
	rss= loadRSS(rss_in)
	if flip_x==1:
		rss._arc_position_x=-1*rss._arc_position_x
	if flip_y==1:
		rss._arc_position_y=-1*rss._arc_position_y

	if int(full_field)==0:
		full_field=False
	elif int(full_field)==1:
		full_field=True

	if parallel=='auto':
		cpus = cpu_count()
	else:
		cpus = int(parallel)

	if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
		idx = numpy.argmin(numpy.fabs(pos_x._wave-ref_pos_wave))
		ref_x = pos_x._data[0, idx]
		ref_y = pos_y._data[0, idx]
		if int_ref==1:
			ref_x = numpy.rint(ref_x)
			ref_y = numpy.rint(ref_y)
		offset_x = (pos_x._data-ref_x)*-1
		if flip_x==1:
			offset_x = offset_x*-1
		offset_y = (pos_y._data-ref_y)*-1
		if flip_y==1:
			offset_y = offset_y*-1

		if verbose==1:
			print(ref_x, ref_y)
			pylab.plot(pos_x._wave, offset_x[0, :], '-k')
			pylab.plot(pos_y._wave, offset_y[0, :], '-b')
			pylab.show()

		if rss._shape=='C':
			min_x = numpy.min(rss._arc_position_x) - rss._size[0]
			max_x = numpy.max(rss._arc_position_x) + rss._size[0]
			min_y = numpy.min(rss._arc_position_y) - rss._size[1]
			max_y = numpy.max(rss._arc_position_y) + rss._size[1]
			dim_x = numpy.rint(float(max_x - min_x) / resolution)
			dim_y = numpy.rint(float(max_y - min_y) / resolution)
			if int_ref == 2:
				ref_x = numpy.argsort(numpy.fabs(min_x + numpy.arange(dim_x)*resolution - offset_x[0,idx]))[0]
				ref_y = numpy.argsort(numpy.fabs(min_y + numpy.arange(dim_y)*resolution - offset_y[0,idx]))[0]
				off_x = (min_x +numpy.arange(dim_x)[ref_x]*resolution) - offset_x[0,idx]
				off_y = (min_y + numpy.arange(dim_y)[ref_y]*resolution) - offset_y[0,idx]
				offset_x = offset_x + off_x
				offset_y = offset_y + off_y
				ref_x+=1
				ref_y+=1
		elif rss._shape=='R':
			if full_field==False:
				min_x = numpy.round(numpy.min(rss._arc_position_x) , 4)
				max_x = numpy.round(numpy.max(rss._arc_position_x),  4)
				min_y = numpy.round(numpy.min(rss._arc_position_y) , 4)
				max_y = numpy.round(numpy.max(rss._arc_position_y), 4)
				dim_x = numpy.round(numpy.rint(float(max_x-min_x)/resolution), 4)+1
				dim_y = numpy.round(numpy.rint(float(max_y-min_y)/resolution), 4)+1
				dim_x = int(dim_x)
				dim_y = int(dim_y)
				min_x=float(min_x)
				min_y=float(min_y)
			else:
				min_x = numpy.round(numpy.min(rss._arc_position_x[:,numpy.newaxis]+offset_x*resolution) , 4)
				max_x = numpy.round(numpy.max(rss._arc_position_x[:,numpy.newaxis]+offset_x*resolution),  4)
				min_y = numpy.round(numpy.min(rss._arc_position_y[:,numpy.newaxis]+offset_y*resolution) , 4)
				max_y = numpy.round(numpy.max(rss._arc_position_y[:,numpy.newaxis]+offset_y*resolution), 4)
				dim_x = numpy.round(numpy.rint(float(max_x-min_x)/resolution), 4)+1
				dim_y = numpy.round(numpy.rint(float(max_y-min_y)/resolution), 4)+1
				dim_x = int(dim_x)
				dim_y = int(dim_y)
				min_x=float(min_x)
				min_y=float(min_y)


		# needed to make sure the the c-code is compiled
		dummy_rss = RSS(data=numpy.zeros((rss._fibers, 2), dtype=numpy.float32), error = numpy.zeros((rss._fibers, 2), dtype=numpy.float32),  mask = numpy.zeros((rss._fibers, 2), dtype=bool), wave = rss._wave[:2], shape=rss._shape, size=rss._size, arc_position_x=rss._arc_position_x, arc_position_y=rss._arc_position_y, good_fibers=rss._good_fibers, fiber_type=rss._fiber_type)
		dummy_rss.createCubeInterDAR_new(offset_x, offset_y, min_x,max_x,min_y,max_y,dim_x,dim_y, mode=mode, sigma=sigma, resolution=resolution, radius_limit=radius_limit, min_fibers=min_fibers,slope=slope, bad_threshold=bad_threshold,  full_field=full_field,replace_error=replace_error,store_cover=store_cover)
	#set the dimension for the final array\



	if cpus>1:
		pool = Pool(cpus)
		threads=[]
		part_rss = rss.splitRSS(cpus)
		if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
			offset_x = RSS(data=offset_x,wave=rss._wave)
			offset_y = RSS(data=offset_y,wave=rss._wave)
			part_offsets_x = offset_x.splitRSS(cpus)
			part_offsets_y = offset_y.splitRSS(cpus)

		data=[]
		error=[]
		error_weight=[]
		mask=[]
		cover=[]

		for i in range(cpus):
			if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
	
		#cube = part_rss[i].createCubeInterDAR_new(part_offsets_x[i]._data, part_offsets_y[i]._data, mode=mode, sigma=sigma, resolution=resolution, radius_limit=radius_limit, min_fibers=min_fibers,slope=slope, bad_threshold=bad_threshold,  replace_error=replace_error)
				threads.append(pool.apply_async(part_rss[i].createCubeInterDAR_new, args=(part_offsets_x[i]._data, part_offsets_y[i]._data, min_x,max_x,min_y,max_y,dim_x,dim_y,mode, sigma,  radius_limit, resolution, min_fibers, slope, bad_threshold, full_field,replace_error, store_cover)))
			else:
				threads.append(pool.apply_async(part_rss[i].createCubeInterpolation, args=(mode, sigma,  radius_limit, resolution, min_fibers, slope, bad_threshold, replace_error, store_cover)))
		pool.close()
		pool.join()

		for i in range(cpus):
			cube = threads[i].get()
			if i==0:
				header = cube._header
			data.append(cube._data)
			error.append(cube._error)
			error_weight.append(cube._error_weight)
			mask.append(cube._mask)
			cover.append(cube._cover)

		data = numpy.concatenate(data)
		if rss._error is not None:
			error = numpy.concatenate(error)
			error_weight = numpy.concatenate(error_weight)
		else:
			error = None
			error_weight = None
		mask = numpy.concatenate(mask)
		if store_cover and mode=='inverseDistance':
			cover = numpy.concatenate(cover)
		else:
			cover=None

		cube = Cube(data=data, error=error, mask=mask, wave=rss._wave, error_weight=error_weight, header=header,cover=cover)
	else:
		if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
			print(rss.getHdrValue('CRVAL1'), rss.getHdrValue('CDELT1'))
			cube = rss.createCubeInterDAR_new(offset_x, offset_y,min_x,max_x,min_y,max_y,dim_x,dim_y, mode=mode, sigma=sigma, resolution=resolution, radius_limit=radius_limit, min_fibers=min_fibers,slope=slope, bad_threshold=bad_threshold, replace_error=replace_error, store_cover=store_cover)
		else:
			cube = rss.createCubeInterpolation(mode=mode, sigma=sigma, resolution=resolution, radius_limit=radius_limit, min_fibers=min_fibers,slope=slope, bad_threshold=bad_threshold, replace_error=replace_error, store_cover=store_cover)

 #   Cube.writeFitsData('dat_'+cube_out, extension_data=0)
  #  Cube.writeFitsData('err_'+cube_out, extension_error=0)
  #  Cube.writeFitsData('mask_'+cube_out, extension_mask=0)
	if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
		cube.setHdrValue('CRPIX1',  ref_x, 'Ref pixel for WCS')
		cube.setHdrValue('CRPIX2',  ref_y,  'Ref pixel for WCS')
	cube.writeFitsData(cube_out)

def correctGalExtinct_drp(rss_in, rss_out, Av, Rv='3.1', verbose='0'):
	"""
			Corrects the wavelength calibrated RSS for the effect of galactic extinction using
			the galactic extinction curve from Cardelli et al. (1989).

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			rss_out : string
					Output RSS FITS file with the corrected spectra
			Av : string of float
					V-band galactic extinction in magnitudes along the line of sight
			Rv : string of float, optional with default: '3.1'
					average E(B-V)/A(V) ratio
			verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Examples
			----------------
			user:> lvmdrp rss correctGalExtinct RSS_IN.fits RSS_OUT.fits 0.33
	"""

	Av=float(Av)
	Rv=float(Rv)

	verbose = int(verbose)
	rss = loadRSS(rss_in)

	if len(rss._wave.shape)==1:
		galExtCurve = ancillary_func.galExtinct(rss._wave, Rv)
		Alambda = galExtCurve*Av
		if verbose==1:
			pylab.plot(1.0/10**(old_div(Alambda._data,-2.5)) )
			pylab.show()
		rss_corr = rss*(1.0/10**(old_div(Alambda,-2.5)))
	rss_corr.writeFitsData(rss_out)

def correctTelluric_drp(rss_in, rss_out, telluric_spectrum, airmass='AIRMASS'):
	"""
			Corrects the wavelength calibrated RSS for the effect of telluric absoroption using
			a transmission spectrum generated from a star.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			rss_out : string
					Output RSS FITS file with the corrected spectra
			telluric_spectrum : string
					FITS file of the telluric transmission spectrum
			airmass : string or string of float, optional with default: 'AIRMASS'
					Airmass for the target observation.
					Either a corresponding header keyword or a float value may be used.

			Examples
			----------------
			user:> lvmdrp rss correctTelluric RSS_IN.fits RSS_OUT.fits TELL_SPEC.fits
			user:> lvmdrp rss correctTelluric RSS_IN.fits RSS_OUT.fits TELL_SPEC.fits  1.4
	"""
	rss = loadRSS(rss_in)
	telluric = Spectrum1D()
	telluric.loadFitsData(telluric_spectrum)

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
	rss_corr.writeFitsData(rss_out)

def splitFile_drp(rss_in, data='', error='', mask='', wave='', fwhm='', position_table=''):
	"""
			Copies the different extension of the RSS into separate files.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			data : string, optional with default: ''
					Ouput FITS file name containing only the data RSS in its primary extension
			error : string, optional with default: ''
					Ouput FITS file name containing only the error RSS in its primary extension
			mask : string, optional with default: ''
					Ouput FITS file name containing only the bad pixel mask RSS in its primary extension
			wave : string, optional with default: ''
					Ouput FITS file name containing only the wavelength RSS in its primary extension
			fwhm : string, optional with default: ''
					Ouput FITS file name containing only the spectral resolution RSS in its primary extension
			position_table : string, optional with default: ''
					Ouput ASCII file of the position table in E3D format

			Examples
			----------------
			user:> lvmdrp rss splitFile RSS_IN.fits DATA_RSS.fits
			user:> lvmdrp rss splitFile RSS_IN.fits mask=MASK_RSS.fits position_table=POSTAB.txt
	"""
	rss = loadRSS(rss_in)

	if data!='' and rss._data is not None:
		rss.writeFitsData(data, extension_data=0, include_PT=False)

	if error!='' and rss._error is not None:
		rss.writeFitsData(error, extension_error=0, include_PT=False)

	if mask!='' and rss._mask is not None:
		rss.writeFitsData(mask, extension_mask=0, include_PT=False)

	if position_table!='' and rss._arc_position_x is not None:
		rss.writeTxtPosTab(position_table)

def maskFibers_drp(rss_in,rss_out,fibers,replace_error='1e10'):
	replace_error = float(replace_error)
	mask_fibers = fibers.split(',')

	rss =loadRSS(rss_in)
	for i in range(len(mask_fibers)):
		mfibers=mask_fibers[i].split('-')
		if len(mfibers)==2:
			for f in range(int(mfibers[0]),int(mfibers[1])+1,1):
				rss.maskFiber(f+1,replace_error=replace_error)
		else:
			rss.maskFiber(int(mask_fibers[i])-1,replace_error=replace_error)
	rss.writeFitsData(rss_out)

def maskNAN_drp(rss_in, replace_error='1e12'):
	rss = loadRSS(rss_in)
	select = numpy.isnan(rss._data)
	if numpy.sum(select)>0:
		for i in range(rss._fibers):
			if numpy.sum(select[i, :]):
				rss._data[i, :]=0
				if rss._error is not None:
					rss._error[i, :]=float(replace_error)
				if rss._mask is not None:
					rss._mask[i, :]=True
		rss.writeFitsData(rss_in)

def flatten(x):
	# taken from: https://bit.ly/3BgHitH
	result = []
	for el in x:
		if isinstance(x, collections.Iterable) and not isinstance(el, str):
			result.extend(flatten(el))
		else:
			result.append(el)
	return result

def registerSDSS_drp(rss_in, rss_out, sdss_file, sdss_field, filter, ra, dec, hdr_prefix,  search_box='20.0,2.6', step='1.0,0.2', offset_x ='0.0',  offset_y='0.0', quality_figure='',  angle_key='SPA', parallel='auto', verbose='0'):
	"""
			Copies the different extension of the RSS into separate files.

			Parameters
			--------------
			rss_in : string
					Input RSS FITS file
			rss_out : string
					Output RSS FITS file
			sdss_file : string
					Original SDSS file in a given filter band that contains the object given in rss_in
			sdss_field : string
					Corresponding SDSS field calibration field for photometric calibration
			filter : string
					Filter response curve correponding to the SDSS file and covered by the data.
					The number of columns containing the wavelength and transmission are followed comma separated
			ra : string of float
					Right ascension of reference point to center the IFU in degrees
			dec: string of float
					Declination of reference point to center the IFU in degrees
			hdr_prefix : string
					Prefix for the FITS keywords in which the measurement parameters are stored. Need to start with 'hierarch'
			search_box : string list of floats with default '20.0,2.6'
					Search box size  for subsequent iterations to construct the chi-square plane of the matching
			step : string list of floats with default '1.0,0.2'
					Sampling for subsequent iterations to construct the chi-square plane of the matching
			offset_x : string of float with default '0.0'
					Inital guess for the offset in x (right ascension ) direction
			offset_y : string of float with default '0.0'
					Inital guess for the offset in y (declination ) direction
			quality_figure : string with default  ''
					Name of the output quality control figure. If empty no figure will be produced
			 parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
					Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
					for the given system is used.
			 verbose: string of integer (0 or 1), optional  with default: 1
					Show information during the processing on the command line (0 - no, 1 - yes)

			Examples
			----------------
			user:> lvmdrp rss registerSDSS RSS_IN.fits RSS_OUT.fits SDSS_r_IMG.fits SDSS_FIELD.fit sloan_r.dat,0,1 234.0 20.3 'hierarch TEST'
			user:> lvmdrp rss registerSDSS RSS_IN.fits RSS_OUT.fits SDSS_r_IMG.fits SDSS_FIELD.fit sloan_r.dat,0,1 234.0 20.3 'hierarch TEST'  search_box=20,2 step=2,0.5 quality_figure='test.png' parralel=3 verbose=1
	"""

	import astLib

	search_box = numpy.array(search_box.split(',')).astype(numpy.float32)
	step =  numpy.array(step.split(',')).astype(numpy.float32)
	offset_x=float(offset_x)
	offset_y = float(offset_y)
	verbose=int(verbose)

	rss = loadRSS(rss_in)

	filter=filter.split(',')
	posTab = rss.getPositionTable()
	fiber_area = numpy.pi*posTab._size[0]**2
	img = loadImage(sdss_file)
	sdssimg = img.calibrateSDSS(sdss_field)
	spa = -1*img.getHdrValue(angle_key)
	scale=0.396
	#sdssimg._header.verify('fix')
	wcs = astLib.astWCS.WCS(sdssimg._header,mode='pyfits')
	pix_coordinates = flatten(wcs.wcs2pix(ra,dec))
	passband = PassBand()
	passband.loadTxtFile(filter[0], wave_col=int(filter[1]),  trans_col=int(filter[2]))

	for i in range(len(search_box)):
		if verbose==1:
			print('Start iteration %d'%(i+1))
			print('Searchbox %.2f arcsec with sampling of %.2f arcsec'%(search_box[i], step[i]))
		if i>0:
			offset_x = best_offset_x
			offset_y = best_offset_y
		(offsets_xIFU, offsets_yIFU, chisq, scale_flux, AB_flux,valid_fibers) = rss.registerImage(sdssimg, passband, search_box[i], step[i], pix_coordinates[0]+1, pix_coordinates[1]+1, scale, spa, offset_x, offset_y, parallel=parallel)
		idx = numpy.indices(chisq.shape)
		#select_valid = numpy.max(valid_fibers)-2<valid_fibers
		#select_best=numpy.min(chisq[select_valid])==chisq
		select_best = numpy.min(chisq) == chisq
		best_offset_x = offsets_xIFU[select_best][0]
		best_offset_y = offsets_yIFU[select_best][0]
		best_chisq = chisq[select_best][0]
		best_scale = scale_flux[select_best][0]
		best_valid = valid_fibers[select_best][0]

		if verbose==1:
			print('Best offset in RA: %.2f'%(-1*best_offset_x))
			print('Best offset in DEC: %.2f'%(-1*best_offset_y))
			print('Minimum Chi-square: %.1f'%(best_chisq))
			print('Valid fibers: %.1f'%(best_valid))
			print('Photometric scale factor: %.3f'%(best_scale))

	rss = loadRSS(rss_in)
	if rss._size is not None:
		rss.offsetPosTab(-1*best_offset_x, -1*best_offset_y)
	rss=rss*best_scale
	rss._data=rss._data.astype(numpy.float32)
	rss_error=rss._error.astype(numpy.float32)
	rss.setHdrValue(hdr_prefix+' PIPE OFFX', float('%.2f'%(best_offset_x)),  'IFU RA offset from ref coordinate')
	rss.setHdrValue(hdr_prefix+' PIPE OFFY', float('%.2f'%(best_offset_y)),  'IFU DEC offset from ref coordinate')
	rss.setHdrValue(hdr_prefix+' PIPE CHISQ', float('%.2f'%(best_chisq)),  'CHISQ of image matching')
	rss.setHdrValue(hdr_prefix+' PIPE VALIDFIB', int('%d'%(best_valid)),  'Valid fibers for image matching')
	rss.setHdrValue(hdr_prefix+' PIPE PHOTSCL', float('%.3f'%(best_scale)),   'photometric scale factor')
	rss.writeFitsData(rss_out)

	if quality_figure!='' or verbose==1:
		flux =  sdssimg.extractApertures(posTab, pix_coordinates[0], pix_coordinates[1], scale, angle=spa, offset_arc_x=best_offset_x, offset_arc_y=best_offset_y)

		fig = plt.figure(figsize=(16,6))
		ax1 = fig.add_axes([0.01,0.08,0.3,0.79])
		ax2 = fig.add_axes([0.32,0.08,0.3,0.79])
		x_pos = rss._arc_position_x+best_offset_x
		y_pos = rss._arc_position_y+best_offset_y
		select_nan = numpy.isnan(AB_flux)
		norm = matplotlib.colors.LogNorm(vmin=numpy.min(AB_flux[numpy.logical_not(select_nan)]),vmax=numpy.max(AB_flux[numpy.logical_not(select_nan)]))
		XY = numpy.hstack(((x_pos).ravel()[:, numpy.newaxis], (y_pos).ravel()[:, numpy.newaxis]))
		circ = matplotlib.collections.CircleCollection([60]*len(y_pos), offsets=XY, transOffset=ax1.transData,norm=norm,cmap=matplotlib.cm.gist_stern_r)

		AB_flux[select_nan] = 1e-30
		circ.set_array(AB_flux.ravel())
		ax1.add_collection(circ)
		ax1.autoscale_view()
		ax1.set_xlim(-40, 40)
		ax1.set_ylim(-40, 40)
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax1.set_title('CALIFA r band',fontsize=18,fontweight='bold')

		XY = numpy.hstack(((x_pos).ravel()[:, numpy.newaxis], (y_pos).ravel()[:, numpy.newaxis]))
		circ2 = matplotlib.collections.CircleCollection([60]*len(y_pos), offsets=XY, transOffset=ax2.transData,norm=norm,cmap=matplotlib.cm.gist_stern_r)
		select_nan=numpy.isnan(flux[0])
		flux[0][select_nan] = 1e-30
		circ2.set_array((old_div(flux[0],best_scale)).ravel())
		ax2.add_collection(circ2)
		ax2.autoscale_view()
		ax2.set_xlim(-40, 40)
		ax2.set_ylim(-40, 40)
		ax2.set_xticks([])
		ax2.set_yticks([])
		ax2.set_title('SDSS best-match CALIFA map',fontsize=18,fontweight='bold')

		ax3 = fig.add_axes([0.66,0.08,0.35,0.83])	
		norm = matplotlib.colors.LogNorm(vmin=100.0/float(best_valid),vmax=numpy.max(chisq))	
		chi_map=ax3.imshow(chisq.T,origin='lower',interpolation='nearest',norm=norm,extent=[offsets_xIFU[0, 0]-step[i]/2.0,offsets_xIFU[-1, 0]+step[i]/2.0, offsets_yIFU[0, 0]-step[i]/2.0,offsets_yIFU[0, -1]+step[i]/2.0])
		ax3.plot(best_offset_x, best_offset_y, 'ok', ms=8)
		cb = plt.colorbar(chi_map,ax=ax3,pad=0.0)
		ax3.set_xlabel('offset in RA [arcsec]',fontsize=18)
		ax3.set_ylabel('offset in DEC [arcsec]',fontsize=18)
		ax3.minorticks_on()
		for line in ax3.xaxis.get_ticklines()+ax3.yaxis.get_ticklines()+ax3.xaxis.get_minorticklines()+ax3.yaxis.get_minorticklines():
			line.set_markeredgewidth(2.0)
		ax3.set_title('$\mathbf{\chi^2}$ matching for offsets',fontsize=18,fontweight='bold')
		ax3.set_xlim([offsets_xIFU[0, 0]-step[i]/2.0,offsets_xIFU[-1, 0]+step[i]/2.0])
		ax3.set_ylim([offsets_yIFU[0, 0]-step[i]/2.0,offsets_yIFU[0, -1]+step[i]/2.0])
		if quality_figure!='':
			plt.savefig(quality_figure)
		if verbose==1:
			plt.show()

def DAR_registerSDSS_drp(rss_in, sdss_file, sdss_field, ra, dec, out_prefix,  ref_wave, coadd='150', step='150', smooth_poly='3', resolution='0.3,0.05', guess_x ='0.0',  guess_y='0.0',start_wave='', end_wave='',  parallel='auto', verbose='0'):
	"""
			NOT YET TEST.
			Requires the AstLib python package to be used
	"""
	import astLib

	resolution = numpy.array(resolution.split(',')).astype(numpy.float32)
	search_box = resolution*5
	guess_x=float(guess_x)
	guess_y = float(guess_y)
	coadd= int(coadd)
	step= int(step)
	smooth_poly=int(smooth_poly)
	ref_wave = float(ref_wave)
	verbose=int(verbose)

	if start_wave=='':
		start_wave=None
	else:
		start_wave=float(start_wave)
	if end_wave=='':
		end_wave=None
	else:
		end_wave=float(end_wave)

	rss = loadRSS(rss_in)
	posTab = rss.getPositionTable()
	fiber_area = numpy.pi*posTab._size[0]**2
	img = loadImage(sdss_file)
	sdssimg = img.calibrateSDSS(sdss_field)
	spa = -1*img.getHdrValue('spa')
	scale=0.396
	wcs = astLib.astWCS.WCS(sdssimg._header,mode='pyfits')
	pix_coordinates = wcs.wcs2pix(ra,dec)
	steps = int(numpy.rint(old_div(rss._res_elements,step)))
	mean_wave = numpy.zeros(steps)
	position_x = numpy.zeros(steps, dtype=numpy.float32)
	position_y = numpy.zeros(steps, dtype=numpy.float32)

	passbands=[]
	for m in range(steps):
		filter= numpy.zeros(rss._res_elements)
		filter[step*m:step*(m+1)] = 1.0
		select_wave = filter>0.0
		mean_wave[m] = numpy.mean(rss._wave[select_wave])
		passbands.append(PassBand(wave=rss._wave, data=filter))

	diff = (mean_wave-ref_wave)**2
	select_start = diff==numpy.min(diff)
	idx_pass = numpy.arange(steps)
	select_pass=idx_pass[select_start][0]
	passband = passbands[select_pass]
	(flux_rss, error_rss) = passband.getFluxRSS(rss)
	flux_rss = flux_rss*fiber_area
	error_rss = error_rss*fiber_area

	rss_mag = passband.fluxToMag(flux_rss)
	AB_flux =10**(old_div(rss_mag,-2.5))
	AB_eflux = error_rss*(old_div(AB_flux,flux_rss))
	good_rss = old_div(flux_rss,error_rss)>3.0
	for i in range(len(search_box)):
		result = rss.registerImage(sdssimg, passband, search_box[i], resolution[i], pix_coordinates[0], pix_coordinates[1], scale, spa, guess_x, guess_y, parallel=parallel)
		guess_x = result[0]
		guess_y = result[1]
	position_x[select_start]=guess_x
	position_y[select_start]=guess_y

	select_blue = mean_wave<mean_wave[select_start]
	select_red =  mean_wave>mean_wave[select_start]
	for m in range(idx_pass[select_blue][-1], idx_pass[select_blue][0]-1, -1):
		result = rss.registerImage(sdssimg, passbands[m], search_box[-1], resolution[-1], pix_coordinates[0], pix_coordinates[1], scale, spa, position_x[m+1], position_y[m+1], parallel=parallel)
		position_x[m]=result[0]
		position_y[m]=result[1]

	for m in range(idx_pass[select_red][0], idx_pass[select_red][-1]+1, 1):
		result = rss.registerImage(sdssimg, passbands[m], search_box[-1], resolution[-1], pix_coordinates[0], pix_coordinates[1], scale, spa, position_x[m-1], position_y[m-1], parallel=parallel)
		position_x[m]=result[0]
		position_y[m]=result[1]

	spec_y =Spectrum1D(data=position_y, wave=mean_wave)
	spec_x =Spectrum1D(data=position_x, wave=mean_wave)
	poly_x = spec_x.smoothPoly(order=smooth_poly, ref_base=rss._wave, start_wave=start_wave, end_wave=end_wave)
	poly_y = spec_y.smoothPoly(order=smooth_poly, ref_base=rss._wave, start_wave=start_wave, end_wave=end_wave)
	spec_x.writeFitsData(out_prefix+'.cent_x.fits')
	spec_y.writeFitsData(out_prefix+'.cent_y.fits')
	if verbose==1:
		pylab.plot(mean_wave, position_x, 'ob')
		pylab.plot(spec_x._wave, spec_x._data, '-b')
		pylab.plot(mean_wave, position_y, 'ok')
		pylab.plot(spec_y._wave, spec_y._data, '-k')
		pylab.show()









