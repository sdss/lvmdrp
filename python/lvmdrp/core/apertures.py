from builtins import range
from builtins import object
import numpy, math

class Aperture(object):
	
	def __init__(self, xcenter, ycenter,  radius, kmax=1000, grid_fixed=False):
		self._xCenter=xcenter
		self._yCenter=ycenter
		self._radius = radius
		if grid_fixed==True:
			subres = int(numpy.sqrt(kmax))
			self._sampeling = numpy.indices((subres,subres)).reshape(2,subres*subres).transpose()*(1/float(subres))
		else:
			self._sampeling=  numpy.random.random_sample((kmax,2))
		self._kmax = kmax
	
	def cover(self, i,j):
		x = (j+1)-self._xCenter
		y = (i+1)-self._yCenter
		rxy = numpy.sqrt((x**2)+(y**2))  
		area = numpy.zeros(i.shape,dtype=numpy.float)
		select = rxy < (self._radius-0.71)
		area[select] = 1.0
		select = numpy.logical_and(rxy >= (self._radius-0.71),rxy <= (self._radius+0.71))
		xx = (x[select]-0.5)[numpy.newaxis,:]+self._sampeling[:,1][:,numpy.newaxis]
		yy = (y[select]-0.5)[numpy.newaxis,:]+self._sampeling[:,0][:,numpy.newaxis]
		area[select] = numpy.sum(numpy.sqrt(xx**2 + yy**2) <=self._radius,0)/float(self._kmax)
		return area
		
	def cover_mask(self, image_shape):
		cover_fraction = numpy.fromfunction(self.cover,image_shape)
		return cover_fraction
		
	def integratedFlux(self, image):
		dim = image._dim
		xlim = [self._xCenter//1-1-math.ceil(self._radius)-2,self._xCenter//1-1+math.ceil(self._radius)+3]
		ylim = [self._yCenter//1-1-math.ceil(self._radius)-2,self._yCenter//1-1+math.ceil(self._radius)+3]

		cor_x = self._xCenter//1-self._xCenter
		cor_y = self._xCenter//1-self._xCenter
		if xlim[0]<=0:
		  xlim[0]=0
		if ylim[0]<=0:
		  ylim[0]=0
		if xlim[1]>=dim[1]-1:
		  xlim[1] =dim[1]-1
		if ylim[1]>=dim[0]-1:
		  ylim[1] =dim[0]-1
		if not xlim[0]>=xlim[1] and not ylim[0]>=ylim[1]:
			data= image._data[ylim[0]:ylim[1], xlim[0]:xlim[1]]
				
			orig_xCenter = self._xCenter
			orig_yCenter = self._yCenter
			self._xCenter=self._xCenter-(self._xCenter+cor_x-2-math.ceil(self._radius))
			self._yCenter=self._yCenter-(self._yCenter+cor_y-2-math.ceil(self._radius))
			area_mask = self.cover_mask((ylim[1]-ylim[0],xlim[1]-xlim[0]))
			if image._mask!=None:
				badpix = image._mask[ylim[0]:ylim[1], xlim[0]:xlim[1]]
				area_mask[badpix]=0
			select = numpy.logical_or(numpy.isnan(data),numpy.isinf(data))
			data[select]=0.0
			select=area_mask>0
			flux_mask= area_mask[select]*data[select]
			total_area  = numpy.sum(area_mask[select].flatten())
			total_flux = numpy.sum(flux_mask)
			if image._error!=None:
				error = image._error[ylim[0]:ylim[1], xlim[0]:xlim[1]]
				error_mask = area_mask[select]*error[select]
				total_error = numpy.sqrt(numpy.sum(error_mask**2))
			else:
				total_error = None
			self._xCenter = orig_xCenter
			self._yCenter = orig_yCenter
		else:
			total_area = 0.0
			total_flux = 0.0
			if image._error!=None:
				total_error=0.0
			else:
				total_error=None
		return total_flux, total_error, total_area
		
class Apertures(object):
	def __mul__(self, other):
		newApertures = Apertures(self._xCenters*other, self._yCenters*other, self._radii*other, self._kmax)
		return newApertures
		
	def __init__(self, xcenters, ycenters,  radii, kmax=1000):
		self._apertures=[]
		for i in range(len(xcenters)):
			self._apertures.append(Aperture(xcenters[i], ycenters[i], radii[i], kmax))
		self._napertures=len(xcenters)
		self._xCenters=xcenters
		self._yCenters=ycenters
		self._radii = radii
		self._kmax= kmax
		
	def offsetApertures(self, offset_x, offset_y):
		newApertures = Apertures(self._xCenters+offset_x, self._yCenters+offset_y, self._radii, self._kmax)
		return newApertures
	
		
	def integratedFlux(self, image):
		aperture_fluxes = numpy.zeros(len(self._xCenters), dtype=numpy.float32)
		aperture_areas =  numpy.zeros(len(self._xCenters), dtype=numpy.float32)
		if image._error!=None:
			aperture_errors = numpy.zeros(len(self._xCenters), dtype=numpy.float32)
		else:
			aperture_errors = None
			
		for i in range(self._napertures):
			result = self._apertures[i].integratedFlux(image)
			aperture_fluxes[i]=result[0]
			aperture_areas[i]=result[2]
			if image._error!=None:
				aperture_errors[i]=result[1]
		
		return aperture_fluxes, aperture_errors, aperture_areas
