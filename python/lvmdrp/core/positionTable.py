from __future__ import print_function
from builtins import range
from builtins import object
from astropy.io import fits as pyfits
import numpy

class PositionTable(object):
	
	def __init__(self, shape=None, size=None, arc_position_x=None, arc_position_y=None, good_fibers=None,  fiber_type=None):
		if shape is None:
			self._shape = None
		else:
			self._shape = shape
			
		if size is None:
			self._size = None
		else:
			self._size = size
			
		if arc_position_x is None:
			self._arc_position_x = None
		else:
			self._arc_position_x = numpy.array(arc_position_x)
			
		if arc_position_y is None:
			self._arc_position_y = None
		else:
			self._arc_position_y = numpy.array(arc_position_y)
		
		if good_fibers is None:
			self._good_fibers= None
		else:
			self._good_fibers = numpy.array(good_fibers)
			
		try:
			self._fiber_type = numpy.array(fiber_type) 
		except:
			self._fiber_type = None
			
			
	def loadTxtPosTab(self, file):
		
		dat = open(file, 'r')
		lines = dat.readlines()
		dat.close()
		line = lines[0].split()
		self._shape =  line[0]
		self._size = (float(line[1]), float(line[2]))
		fibers = len(lines)-1
		pos_x = numpy.zeros(fibers, dtype=numpy.float32)
		pos_y = numpy.zeros(fibers, dtype=numpy.float32)
		good = numpy.zeros(fibers, dtype=numpy.uint8)
		if len(lines[1].split())>4:
			fiber_type = numpy.empty(fibers, dtype="|S3")
		for i in range(1, fibers+1):
			line = lines[i].split()
			pos_x[i-1] = float(line[1])
			pos_y[i-1] = float(line[2])
			good[i-1] = int(line[3])
			if len(line) > 4:
				fiber_type[i-1] = line[4]
		self._arc_position_x = pos_x
		self._arc_position_y = pos_y
		self._good_fibers = good
		if len(line) > 4:
			self._fiber_type = fiber_type
	
	def writeTxtPosTab(self, file, fiber_type=False):
		dat = open(file, 'w')
		print("%s %.2f %.2f %i"%(self._shape, self._size[0], self._size[1], 0), file=dat)
		for i in range(len(self._arc_position_x)):
			if fiber_type:
				print("%i %.2f %.2f %i %s"%(i+1, self._arc_position_x[i], self._arc_position_y[i], self._good_fibers[i], self._fiber_type[i]), file=dat)
			else:
				print("%i %.2f %.2f %i"%(i+1, self._arc_position_x[i], self._arc_position_y[i], self._good_fibers[i]), file=dat)
		#    print >> dat, "%i %.2f %.2f %i"%(i+1, self._arc_position_x[i], self._arc_position_y[i], self._good_fibers[i])
	   #    print >> dat, "%i %.2f %.2f"%(i+1, self._arc_position_x[i], self._arc_position_y[i])
		dat.close()
		
	def writeFitsPosTable(self):
		columns=[]
		columns.append(pyfits.Column(name='X_Position', unit='arcsec', format='E', array=self._arc_position_x.astype('float32')))
		columns.append(pyfits.Column(name='Y_Position', unit='arcsec', format='E', array=self._arc_position_y.astype('float32')))
		columns.append(pyfits.Column(name='GoodFiber', unit='flag', format='I', array=self._good_fibers))
		columns.append(pyfits.Column(name='FiberType',  format='3A', array=self._fiber_type))
		table = pyfits.BinTableHDU.from_columns(columns)
		table.header['FibShape'] = (self._shape, 'Shape of the fiber (C-Circular, S-Square)')
		table.header['FibSizeX'] = (self._size[0], 'Size of the fiber in x-direction')
		table.header['FibSizeY'] = (self._size[1], 'Size of the fiber in y-direction')
		return table
		
	def loadFitsPosTable(self, table):
		data = table.data
		header = table.header
		self._arc_position_x = data.field('X_Position')
		self._arc_position_y = data.field('Y_Position')
		self._good_fibers = data.field('GoodFiber')
		self._fiber_type = data.field('FiberType')
		self._shape = header['FibShape']
		self._size = (header['FibSizeX'], header['FibSizeY'])
		
	def append(self, pos_tab):
		self._pos_x = numpy.concatenate((self._pos_x, pos_tab._pos_x))
		self._pos_y = numpy.concatenate((self._pos_y,  pos_tab._pos_y))
		self._good = numpy.concatenate((self._good,  pos_tab._good))
		if self._types is not None and pos_tab._types is not None:
			self._types = numpy.concatenate((self._types,  pos_tab._types))
			
	def offsetPosTab(self, offset_x, offset_y):
		self._arc_position_x = self._arc_position_x+offset_x
		self._arc_position_y = self._arc_position_y+offset_y
		
	def rotatePosTab(self, angle, ref_cent_x=0.0, ref_cent_y=0.0):
#        print angle
 #       print self._arc_position_x,  self._arc_position_y
		
		arc_position_x=(self._arc_position_x-ref_cent_x)*numpy.cos(float(angle)/180.0*numpy.pi)-(self._arc_position_y-ref_cent_y)*numpy.sin(float(angle)/180.0*numpy.pi)
		arc_position_y=(self._arc_position_x-ref_cent_x)*numpy.sin(float(angle)/180.0*numpy.pi)+(self._arc_position_y-ref_cent_y)*numpy.cos(float(angle)/180.0*numpy.pi)
		posTab_new = PositionTable(shape=self._shape, size=self._size, arc_position_x=arc_position_x, arc_position_y=arc_position_y, good_fibers=self._good_fibers,  fiber_type=self._fiber_type)
		return posTab_new
		
	def  scalePosTab(self, scale):
		position_x = self._arc_position_x*scale
		position_y = self._arc_position_y*scale
		size = [self._size[0]*scale, self._size[1]*scale]
		posTab_new = PositionTable(shape=self._shape, size=size, arc_position_x=position_x, arc_position_y=position_y, good_fibers=self._good_fibers,  fiber_type=self._fiber_type)
		return posTab_new
		
	def distance(self, x_ref, y_ref):
		distance = numpy.sqrt((x_ref-self._arc_position_x)**2+(y_ref-self._arc_position_y)**2)
		return distance
		
	def setPosTab(self, PosTab):
		self._arc_position_x = PosTab._arc_position_x
		self._arc_position_y = PosTab._arc_position_y
		self._good_fibers = PosTab._good_fibers
		self._fiber_type = PosTab._fiber_type
		self._shape = PosTab._shape
		self._size = PosTab._size
		
def loadPosTable(infile):
	posTab = PositionTable()
	if '.txt' in infile:
		posTab.loadTxtPosTab(infile)
	elif '.fits' in infile or '.fit' in infile:
		hdu = pyfits.open(infile)
		found=False
		for i in range(1, len(hdu)):
			if hdu[i].header['EXTNAME'].split()[0]=='POSTABLE':
				posTab.loadFitsPosTable(hdu[i])
				found=True
		if found==False:
			raise RuntimeError('No position table information found in file %s.'%(infile))
	return posTab
		
