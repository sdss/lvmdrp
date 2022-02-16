from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy
from scipy import stats
from lvmdrp.core.rss import *
from lvmdrp.core.header  import Header
from lvmdrp.core.spectrum1d  import Spectrum1D
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.fit_profile import Exponential_constant

description='Provides Methods for dedicated CALIFA data reduction tasks'

def extinctCAVEX_drp(file,cavex_file, time_average='2', date_key='Date', extinct_key='EXT_V', min_extinct='0.1', missing_extinct='0.2'):
	"""
			Reads the ASCII file produced by the CAVEX monitor at Calar Alto to estimate the V-band atmospheric extinction at the observing time of the target and
			adds a header keyword to the FITS file. It averages the measurements within a given time centered at the time of observation. If this is not possible a
			median value of the whole night is used. This is different for objects taken at the beginning of the night, where the median value of the end of last night
			is used. If no information are available because the monitor was off both nights, the user defined value is taken instead.

			Parameters
			--------------
			file : string
					Input target FITS file name
			caves_file : string
					Input CAVEX file
			time_average : string of float, optional with default: '2'
					time span in units of hours over which the measured extinction should be averaged centered on the observing time
			date_key : string, optional with default: 'Date'
					Header keyword with the information of the date and time of observation
			extinct_key : string, optional with default: 'EXT_V'
					Output header keyword in which the estimated atmospheric extinction is stored
			min_extinct : string of float (>0), optional with default: '0.1'
					Minimum extinction. If a lower extinction is estimated it will be replaced by this value
			missing_extinct : string of float(>0), optional with default: '0.2'
					This extinction value is used in case no extinction could be estimated from CAVEX data.

			Example
			-----------
			user:> lvmdrp special FILE.fits CAVEX.dat time_average=1.5 min_extinct=0.15
		"""
	time_average = float(time_average)
	min_extinct = float(min_extinct)
	missing_extinct = float(missing_extinct)

	header = Header()
	header.loadFitsHeader(file)

	date_hdr = header.getHdrValue(date_key)
	date_in = int(date_hdr.split('T')[0].replace('-', ''))
	time = date_hdr.split('T')[1].split(':')
	UThour = float(time[0])+float(time[1])/60.0+float(time[2])/3600.0

	cavex = open(cavex_file, 'r')
	cavex_lines = cavex.readlines()

	if len(cavex_lines)>0:
		UThours=numpy.zeros(len(cavex_lines), dtype=numpy.float32)
		dates=numpy.zeros(len(cavex_lines), dtype=numpy.int32)
		av=numpy.zeros(len(cavex_lines), dtype=numpy.float32)
		for i in range(len(cavex_lines)):
			line = cavex_lines[i].split()
			try:
				UThours[i] = float(line[0])
				dates[i] = int(line[1])
				av[i] = float(line[5])
			except ValueError:
				pass

		select = av>0
		UThours = UThours[select]
		dates = dates[select]
		av = av[select]


		date_min = date_in
		date_max = date_in
		time_min = UThour-float(time_average/2.0)
		time_max = UThour+float(time_average/2.0)

		if time_min<0:
			date_min-=1
			time_min = 24.0+time_min

		if time_max>24.0:
			date_max+=1
			time_max = time_max-24.0

		select_time = numpy.logical_and(numpy.logical_and(dates==date_min, UThours>=time_min), numpy.logical_and(dates==date_max, UThours<=time_max))

		if numpy.sum(select_time)>0:
			out_av = numpy.mean(av[select_time])
			std_av = numpy.std(av[select_time])
		else:
			std_av = 0.0
			if UThour>18.0 and UThour<23.00:
				select_time = numpy.logical_and(dates==date_in, numpy.logical_and(UThours>=0.0, UThours<=14.0))
				if numpy.sum(select_time)>0:
					out_av = numpy.median(av[select_time])
				else:
					select_time = numpy.logical_or(numpy.logical_and(dates==date_in, UThours>14.0), numpy.logical_and(dates==date_in+1, UThours<14.0))
					if numpy.sum(select_time)>0:
						out_av = numpy.median(av[select_time])
					else:
						out_av = missing_extinct
			else:
				if UThour<14.0:
					date_min = date_in-1
					date_max = date_in
				else:
					date_min = date_in
					date_max = date_in+1

				select_time = numpy.logical_or(numpy.logical_and(dates==date_min, UThours>14.0), numpy.logical_and(dates==date_max, UThours<14.0))
				if numpy.sum(select_time)>0:
					out_av = numpy.median(av[select_time])
				else:
					out_av = missing_extinct

	else:
		out_av = missing_extinct
		std_av = 0.0

	if out_av<min_extinct:
		out_av = min_extinct


	header.setHdrValue(extinct_key, float("%.3f"%(out_av)), 'V-band atmospheric extinction')
	header.setHdrValue(extinct_key+'_SD', float("%.3f"%(std_av)), 'std of V-band atmospheric extinction')
	header.writeFitsHeader()

def matchMasterTrace_drp(CALIB_trace, Master_trace, out_trace, poly_cross=-2):
	poly_cross=int(poly_cross)
	calib_trc = loadRSS(CALIB_trace)
	master_trc = loadRSS(Master_trace)
	master_calib = master_trc.splitFiberType(['CAL'])[0]
	shift = calib_trc._data-master_calib._data

	for i in range(shift.shape[1]):
		spec = Spectrum1D(wave=master_calib._data[:, i], data=shift[:, i])
		if i==-1:
			pylab.plot(spec._wave,spec._data,'ok')
		spec.smoothPoly(order=poly_cross, ref_base=master_trc._data[:, i])
		if i==-1:
			pylab.plot(spec._wave,spec._data,'-r')
			pylab.show()
		master_trc._data[:, i] = master_trc._data[:, i]+spec._data
	master_trc.writeFitsData(out_trace)

def matchARCLamp_drp(arc_rss, arc_rss_ref, disp_ref, disp_out, ref_line_file='', poly_cross='1', poly_disp='6', init_back='4.0', aperture='13', flux_min='100.0', fwhm_max='6.0', rel_flux_limits='0.1,3.0', verbose='0'):
	poly_cross=int(poly_cross)
	poly_disp = int(poly_disp)
	flux_min=float(flux_min)
	fwhm_max=float(fwhm_max)
	init_back=float(init_back)
	aperture=float(aperture)
	limits = rel_flux_limits.split(',')
	rel_flux_limits=[float(limits[0]), float(limits[1])]
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


	arc = FiberRows()
	arc.loadFitsData(arc_rss)

	arc_ref = FiberRows()
	arc_ref.loadFitsData(arc_rss_ref)

	(fibers, flux, cent_wave, fwhm, masked) = arc.measureArcLines(ref_spec, pixel, aperture=aperture, init_back=init_back, flux_min=flux_min, fwhm_max=fwhm_max, rel_flux_limits=rel_flux_limits, verbose=bool(verbose))
	(fibers_ref, flux_ref, cent_wave_ref, fwhm_ref, masked_ref) = arc_ref.measureArcLines(ref_spec, pixel, aperture=aperture, init_back=init_back, flux_min=flux_min, fwhm_max=fwhm_max, rel_flux_limits=rel_flux_limits, verbose=bool(verbose))
	pix_shift = cent_wave-cent_wave_ref
	for i in range(nlines):
		good_pix= numpy.logical_not(numpy.logical_or(masked[:, i], masked_ref[:, i]))
		spec = Spectrum1D(wave=fibers[good_pix], data=pix_shift[good_pix, i])
		spec.smoothPoly(order=poly_cross, ref_base=fibers)
		pix_shift[:, i] = spec._data
	pix_shift_mean = numpy.mean(pix_shift, 1)

	rss_disp = loadRSS(disp_ref)
	for i in range(rss_disp._fibers):
		shift_spec = Spectrum1D(wave=cent_wave[i, :], data=pix_shift[i, :])
		shift_spec.smoothPoly(order=poly_cross, ref_base=numpy.arange(arc._data.shape[1]))
		spec = Spectrum1D(wave=rss_disp._data[i, :], data=rss_disp._data[i, :])
		new_wave = Spectrum1D(spec._pixels+shift_spec._data,spec._wave )
		new_wave.smoothPoly(poly_disp, ref_base=spec._pixels)
		rss_disp._data[i, :]=new_wave._data
	rss_disp.writeFitsData(disp_out)


def checkWavelengthRSS_drp(rss_in, line_list, out_result, ref_spec='', init_back='1.0', aperture='13', flux_min='0.2', fwhm_max='10.0', rel_flux_limits='0.1,5.0', verbose='0'):
	flux_min=float(flux_min)
	init_back=float(init_back)
	aperture=float(aperture)
	limits = rel_flux_limits.split(',')
	rel_flux_limits=[float(limits[0]), float(limits[1])]
	fwhm_max=float(fwhm_max)
	ref_spec = int(ref_spec)
	verbose=int(verbose)

	line_file = open(line_list, 'r')
	lines=line_file.readlines()
	line_list=[]
	line_name=[]
	for i in range(len(lines)):
		line = lines[i]
		if line[0]!='#':
			line_split = line.split()
			if len(line_split)==2:
				print(line_split)
				line_name.append(line_split[1])
				line_list.append(float(line_split[0]))
			elif len(line_split)==1:
				line_list.append(float(line_split[0]))
				line_name.append('')

	line_list = numpy.array(line_list)

	rss = FiberRows()
	rss.loadFitsData(rss_in)
	crval = rss.getHdrValue('CRVAL1')
	cdelt = rss.getHdrValue('CDELT1')
	line_list_pix = (line_list-crval)/float(cdelt)
	(fibers, flux, cent_wave, fwhm, masked) = rss.measureArcLines(ref_spec, line_list_pix, aperture=aperture, init_back=init_back, flux_min=flux_min, fwhm_max=fwhm_max, rel_flux_limits=rel_flux_limits, verbose=bool(verbose))

	out_file = open(out_result, 'w')
	out_file.write('#name input_wavelength median_wave mean_wave sigma_wave median_flux mean_flux sigma_flux  med_sigma mean_sigma sig_sigma\n')
	for i in range(len(line_list)):
		mask_line = numpy.logical_not(masked[:, i])

		out_file.write("%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"%(line_name[i], line_list[i], numpy.median(cent_wave[mask_line, i])*cdelt+crval, numpy.mean(cent_wave[mask_line, i])*cdelt+crval, numpy.std(cent_wave[mask_line, i])*cdelt, numpy.median(flux[mask_line, i]),  numpy.std(flux[mask_line, i]), numpy.median(fwhm[mask_line, i])*cdelt, numpy.mean(fwhm[mask_line, i])*cdelt, numpy.std(fwhm[mask_line, i])*cdelt))


def matchSkySpecTime_drp(list_sky_specs, ref_object, out_spec, hdr_key_start, hdr_key_end, function='polynomial', poly_order='0', err_sim='200', time_steps='1', next_day='', plot='-1'):
	sky_in = list_sky_specs.split(',')
	time_steps=float(time_steps)
	err_sim= int(err_sim)
	poly_order = int(poly_order)
	if next_day!='':
		split = next_day.split(',')
		next_day=[]
		for i in range(len(split)):
			next_day.append(int(split[i]))
	else:
		next_day=[0]*len(sky_in)
	sky_1 = Spectrum1D()
	sky_1.loadFitsData(sky_in[0])
	sky_1_header = Header()
	sky_1_header.loadFitsHeader(sky_in[0])
	sky_wave = sky_1._wave

	sky_data = numpy.zeros((len(sky_in), sky_1._dim), dtype=numpy.float32)
	if sky_1._error is not None:
		sky_error = numpy.zeros((len(sky_in), sky_1._dim), dtype=numpy.float32)

	if err_sim>1 and sky_1._error is not None:
		sky_out = Spectrum1D(wave = sky_wave, data=numpy.zeros(sky_1._dim, dtype=numpy.float32), error=numpy.zeros(sky_1._dim, dtype=numpy.float32))
	else:
		sky_out = Spectrum1D(wave = sky_wave, data=numpy.zeros(sky_1._dim, dtype=numpy.float32))
	sky_time = numpy.zeros(len(sky_in), dtype=numpy.float32)

	sky_data[0, :] = sky_1._data
	if sky_1._error is not None:
		sky_error[0, :] = sky_1._error
	sky_time[0] = (sky_1_header.getHdrValue(hdr_key_start)+(next_day[0]*86400)+sky_1_header.getHdrValue(hdr_key_end)+(next_day[0]*86400))/2.0

	for i in range(1, len(sky_in)):
		sky = Spectrum1D()
		sky.loadFitsData(sky_in[i])
		sky_data[i, :] = sky._data
		if sky._error is not None:
			sky_error[i, :] = sky._error
		sky_header = Header()
		sky_header.loadFitsHeader(sky_in[i])
		sky_time[i] = (sky_header.getHdrValue(hdr_key_start)+(next_day[i]*86400)+sky_header.getHdrValue(hdr_key_end)+(next_day[i]*86400))/2.0

	init_time = sky_time[0]
	sky_time = sky_time-init_time
	hdr_obj = Header()
	hdr_obj.loadFitsHeader(ref_object)
	if hdr_obj.getHdrValue(hdr_key_end)<hdr_obj.getHdrValue(hdr_key_start):
		object_time = numpy.arange(hdr_obj.getHdrValue(hdr_key_start), hdr_obj.getHdrValue(hdr_key_end)+86400+time_steps, time_steps)-init_time
	else:
		object_time = numpy.arange(hdr_obj.getHdrValue(hdr_key_start), hdr_obj.getHdrValue(hdr_key_end)+time_steps, time_steps)-init_time
	#object_time = numpy.arange(0, sky_time[-1])
	if function=='polynomial':
		for i in range(sky_1._dim):
			fit = numpy.polyfit(sky_time, sky_data[:, i], poly_order)
			sky_out._data[i] = numpy.mean(numpy.polyval(fit, object_time))
			if err_sim>1 and sky_1._error is not None:
				out = numpy.zeros(err_sim, dtype=numpy.float32)
				for j in range(err_sim):
					try:    rnormal = numpy.random.normal(sky_data[:, i], sky_error[:, i])
					except: rnormal = numpy.zeros(sky_data[:, i].shape)
					err_fit = numpy.polyfit(sky_time, rnormal, poly_order)
					out[j] = numpy.mean(numpy.polyval(err_fit, object_time))
				sky_out._error[i] = numpy.std(out)
			if int(plot)==i:
				pylab.plot(sky_time, sky_data[:, i], 'ok')
				pylab.plot(object_time, numpy.polyval(fit, object_time), '-r')
				pylab.plot(numpy.mean(object_time), sky_out._data[i], 'or')
				pylab.show()
	if function=='exponential':
		for i in range(sky_1._dim):
			exp_profile = Exponential_constant([2.0, 500, 0.2])
			if err_sim>1 and sky_1._error is not None:
				exp_profile.fit(sky_time,sky_data[:, i], sky_error[:, i], method='simplex', err_sim=err_sim)
				sky_out._error[i] = numpy.std(exp_profile._par_err_models[:, 0][:, numpy.newaxis]*numpy.exp(old_div(object_time[numpy.newaxis, :],exp_profile._par_err_models[:, 1][:, numpy.newaxis]))+exp_profile._par_err_models[:, 2][:, numpy.newaxis])
			else:
				exp_profile.fit(sky_time,sky_data[:, i], method='simplex',  err_sim=0)
			sky_out._data[i] = numpy.mean(exp_profile(object_time))

			if int(plot)==i:
				pylab.plot(sky_time, sky_data[:, i], 'ok')
				pylab.plot(object_time, exp_profile(object_time), '-r')
				pylab.plot(numpy.mean(object_time), sky_out._data[i], 'or')
				pylab.show()
	sky_out.writeFitsData(out_spec)


