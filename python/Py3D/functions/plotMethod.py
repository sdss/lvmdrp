from __future__ import print_function

from builtins import range
import os, numpy
from Py3D.core.header import Header

from Py3D import *
try:
  from matplotlib import pyplot as plt
except:
  pass

description='Provides Methods to make some plots'


def flexurePatternTarget_py3d(trace_log, disp_log, figure, object,  scale='200'):
	"""
    	Makes an arrow plot of the flexure offset in dispersion and cross-dispersion direction compared to the calibration exposure.
		The log files produces the Py3D commands to measure those shifts in both direction are read in to 
		extract the required information.

		Parameters
		--------------
		trace_log : string
			Input log file that contains the measured offsets in cross-dispersion direction
		disp_log : string
			Input log file that contains the measured offsets in dispersion direction
		figure : string
			Output name of the resulting figure. The file format is automatically taken from its suffix, e.g. .png, .pdf, jpg
		object : string
			Name of the object to identify the correct entries in the log files
		scale : string of float (>0), optional with default: '200'
			Scale factor with which all offsets are multiplied to gain in visibility.
			
		Example
		-----------
		user:> Py3D plot fluxurePatternTarget OFFSETCROSS.log OFFSETDISP.log OUTFIG.png NGC7594_p1
	"""
	scale = float(scale)
	trace = open(trace_log, 'r')
	trace_lines = trace.readlines()
	obj_trace=[]
	obj_trace_file=[]
	obj_trace_line=[]
	obj_trace_x=[]
	obj_trace_y=[]
	obj_trace_offset=[]

	i=0
	while i<len(trace_lines):
		line = trace_lines[i].split()
		if len(line)==1:
			obj_trace_file.append(line[0])
			#obj_trace.append(line[0].split('.')[0])
			obj_trace.append(os.path.basename(line[0]).split('.')[0])
			
			j=0
			line_trace=[]
			line_trace_x=[]
			line_trace_y=[]
			line_trace_offset=[]
			while i+j+1<len(trace_lines):
				line1=trace_lines[i+j+1].split()
				if len(line1)==1:
					break
				line2=trace_lines[i+j+2].split()
				line3=trace_lines[i+j+3].split()
				line_trace.append(float(line1[0]))
				line_trace_y.append(numpy.array(line1[1:]).astype('float'))
				line_trace_x.append(numpy.array(line2[1:]).astype('float'))
				line_trace_offset.append(numpy.array(line3[1:]).astype('float')*-1)
				j+=3
			i=i+j+1
			
			obj_trace_line.append(line_trace)
			obj_trace_x.append(line_trace_x)
			obj_trace_y.append(line_trace_y)
			obj_trace_offset.append(line_trace_offset)
	    
	disp = open(disp_log, 'r')
	disp_lines = disp.readlines()
	obj_disp=[]
	obj_disp_file=[]
	obj_disp_line=[]
	obj_disp_offset=[]
    
	i=0
	while i<len(disp_lines):
		line = disp_lines[i].split()
		if len(line)==1:
			obj_disp_file.append(line[0])
			obj_disp.append(os.path.basename(line[0]).split('.')[0])
			
			j=0
			line_disp=[]
			line_disp_offset=[]
			while i+j+1<len(disp_lines):
				line1=disp_lines[i+j+1].split()
				if len(line1)==1:
					break
				line3=disp_lines[i+j+3].split()
				line_disp.append(float(line1[0]))
				line_disp_offset.append(numpy.array(line3).astype('float'))
				j+=4
			i=i+j+1
		
			obj_disp_line.append(line_disp)
			obj_disp_offset.append(line_disp_offset)
	 
    
	obj_disp=numpy.array(obj_disp)
	obj_trace=numpy.array(obj_trace)
	obj_disp_line=numpy.array(obj_disp_line)
	obj_trace_line=numpy.array(obj_trace_line)
	obj_trace_x=numpy.array(obj_trace_x)
	obj_trace_y=numpy.array(obj_trace_y)
	obj_trace_offset=numpy.array(obj_trace_offset)
	obj_disp_offset=numpy.array(obj_disp_offset)

	plt.rcParams['axes.linewidth'] = 1.5
	plt.rcParams['xtick.major.size'] = 8
	plt.rcParams['xtick.minor.size'] = 4
	plt.rcParams['ytick.major.size'] = 6
	plt.rcParams['ytick.minor.size'] = 3
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_axes([0.15, 0.09, 0.83, 0.83])
	select_obj = obj_trace==object
	if numpy.sum(select_obj)==1:
		select_disp = obj_disp==object
		for j in range(len(obj_trace_line[select_obj][0])):
			select_line = (obj_disp_line[select_disp][0]==obj_trace_line[select_obj][0][j])
			for x in range(len(obj_trace_x[select_obj][0][j])):
				ax.arrow(obj_trace_x[select_obj][0][j][x], obj_trace_y[select_obj][0][j][x],  obj_disp_offset[select_disp][0][select_line][0][x]*scale, obj_trace_offset[select_obj][0][j][x]*scale, width=5, head_width=20, head_length=10, fc='k', ec='k')
		ax.arrow(500, 1000, numpy.median(obj_disp_offset[select_disp][0])*scale, numpy.median(obj_trace_offset[select_obj][0])*scale, width=5, head_width=20, head_length=10, fc='r', ec='r')
	ax.set_xlabel('X Pix', fontsize=18)
	ax.set_ylabel('Y Pix',  fontsize=18)
	fig.text(0.1, 0.93, object+' flexure shifts (x%i)'%(scale), fontsize=17)
	ax.minorticks_on()
	ax.set_xlim(0, 2100)
	ax.set_ylim(0, 2100)
	for line in ax.xaxis.get_ticklines()+ax.yaxis.get_ticklines()+ax.xaxis.get_minorticklines()+ax.yaxis.get_minorticklines():
		line.set_markeredgewidth(1.5)     
	plt.savefig(fig)


def  flexureOffsetNight_py3d(trace_log, disp_log, figure,  hdrKey_x, hdrKey_y, hdrKey_El,  hdrKey_Az):
	"""
	    Makes a plot of all the flexure offsets measure for the night with respect to their median value as a function of telescope Azimuth angle.
	    
	    Parameters
	    --------------
	    trace_log : string
		    Input log file that contains the measured offsets in cross-dispersion direction
	    disp_log : string
		    Input log file that contains the measured offsets in dispersion direction
	    figure : string
		    Output name of the resulting figure. The file format is automatically taken from its suffix, e.g. .png, .pdf, jpg
	    hdrKey_x : string
		    Header keyword in which the estimated flexure offset in x-direction is stored
	    hdrKey_y : string
		    Header keyword in which the estimated flexure offset in y-direction is stored
	    hdrKey_El : string
		    Header keyword in which the telescope Elevation is recored at the time of observation
	    hdrKey_Az : string
		    Header keyword in which the telescope Azimuth is recored at the time of observation
		    
	    Example
	    -----------
	    user:> Py3D plot flexureOffsetNight OFFSETCROSS.log OFFSETDISP.log OUTFIG.png 'hierarch PIPE FLEX XOFF' 'hierarch PIPE FLEX YOFF' 'hierarch CAHA TEL POS EL_END' 'hierarch CAHA TEL POS AZ_END'
	"""
	trace = open(trace_log, 'r')
	disp = open(disp_log, 'r')
	obj_trace=[]
	obj_trace_file=[]
	obj_disp=[]
	obj_disp_file=[]
    
    # read the disp offset logfile to find out the object name and corresponding file 
	disp_lines = disp.readlines()
	for i in range(len(disp_lines)):
		line = disp_lines[i].split()
		if len(line)==1:
			obj_disp_file.append(line[0])
			obj_disp.append(os.path.basename(line[0]).split('.')[0])
	    
	disp.close()
    
    # read the trace offset logfile to find out the object name and corresponding file 
	trace_lines = trace.readlines()
	for i in range(len(trace_lines)):
		line = trace_lines[i].split()
		if len(line)==1:
			obj_trace_file.append(line[0])
			obj_trace.append(os.path.basename(line[0]).split('.')[0])
	trace.close()
    
    #convert to numpy arrays
	obj_trace=numpy.array(obj_trace)
	obj_trace_file=numpy.array(obj_trace_file)
	obj_disp=numpy.array(obj_disp)
	obj_disp_file=numpy.array(obj_disp_file)
    
	xoff=[]
	yoff=[]
	az=[]
	el=[]
	obj_name=[]
    
    # match objects and read out information
	for i in range(len(obj_trace)):
		select = obj_disp==obj_trace[i]
		obj_name.append(obj_trace[i].split('_'))
		if numpy.sum(select)==1:
			header = Header()
			header.loadFitsHeader(obj_trace_file[i])
			print(obj_trace_file[i])
			el .append(header.getHdrValue(hdrKey_El))
			az .append(header.getHdrValue(hdrKey_Az))
			yoff.append(header.getHdrValue(hdrKey_y))
			header.loadFitsHeader(obj_disp_file[select][0])
			print(obj_disp_file[select][0])
			xoff.append(header.getHdrValue(hdrKey_x))
    
	xoff=numpy.array(xoff)
	yoff=numpy.array(yoff)
	el=numpy.array(el)
	az=numpy.array(az)
	mean_xoff=numpy.mean(xoff)
	mean_yoff=numpy.mean(yoff)
	std_xoff = numpy.std(xoff)
	std_yoff = numpy.std(yoff)
    
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_axes([0.11, 0.11, 0.87, 0.87])
	ax.plot(az-180.0, xoff-mean_xoff, 'sk', mfc='w', mew=2, ms=7)
	ax.plot(az-180.0, yoff-mean_yoff, 'ok', ms=7)
	ax.text( -160, 1.6,'$\sigma(\Delta x)=%.3f \mathrm{pix}$'%(std_xoff),  fontsize=18)
	ax.text( -160, 1.3, '$\sigma(\Delta y)=%.3f \mathrm{pix}$'%(std_yoff),fontsize=18)
	ax.set_xlim(-180, 180)
	ax.set_ylim(-1.9, 1.9)
	ax.set_xlabel('Azimuth [degrees]', fontsize=16)
	ax.set_ylabel('Delta X,Y [pixels]', fontsize=16)
	ax.minorticks_on()
	plt.savefig(figure)
