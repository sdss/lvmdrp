# -*- coding: utf-8 -*-

from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import numpy
import scipy
from scipy import special
from scipy import optimize
from copy import deepcopy
import lvmdrp
from multiprocessing import cpu_count
from multiprocessing import Pool
try:
  import pylab
except:
  pass


fact = numpy.sqrt(2.*numpy.pi)


class fit_profile1D(object):
	def __init__(self, par, func, guess_par=None, args=None):
		self._par = par
		self._func = func
		self._guess_par = guess_par
		self._args=args

	def __call__(self, x):
		return self._func(x)

	def getPar(self):
		return self._par

	def res(self, par, x, y, sigma=1.0):
		self._par = par
		return old_div((y-self(x)),sigma)


	def residuum(self, par, x, y, sigma=1.0):
		self._par = par
		return numpy.sum((old_div((y-self(x)),sigma))**2)

	def chisq(self, x, y, sigma=1.0):
		return numpy.sum((old_div((y-self(x)),sigma))**2)

	def fit(self, x, y, sigma=1.0, p0=None, ftol=1e-8, xtol=1e-8, maxfev=9999, err_sim=0, warning=True, method='leastsq',parallel='auto'):
		if  p0 == None and self._guess_par!=None:
			self._guess_par(x, y)
		perr_init = deepcopy(self)
		p0 = self._par
		if method=='leastsq':
			try:
				model = optimize.fmin(self.res, p0, (x, y, sigma), maxfev=maxfev, ftol=ftol, xtol=xtol,warning=warning)
			#model = optimize.leastsq(self.res, p0, (x, y, sigma), None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100.0, None, warning)
			except TypeError:
				model = optimize.leastsq(self.res, p0, (x, y, sigma), maxfev=maxfev,ftol=ftol, xtol=xtol)
			self._par = model[0]

		if method=='simplex':
			try:
				model = optimize.fmin(self.residuum, p0, (x, y, sigma), ftol=ftol, xtol=xtol,disp=0, full_output=0,warning=warning)
			#model = optimize.leastsq(self.res, p0, (x, y, sigma), None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100.0, None, warning)
			except TypeError:
				model = optimize.fmin(self.residuum, p0, (x, y, sigma), ftol=ftol, xtol=xtol,disp=0,full_output=0)
			self._par = model
			#model = optimize.leastsq(self.res, p0, (x, y, sigma),None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100.0, None)



		if err_sim!=0:
			if parallel=='auto':
				cpus = cpu_count()
			else:
				cpus = int(parallel)
			self._par_err_models = numpy.zeros((err_sim, len(self._par)), dtype=numpy.float32)
			if cpus>1:
				pool = Pool(processes=cpus)
				results=[]
				for i in range(err_sim):
					perr = deepcopy(perr_init)
					if method=='leastsq':
						results.append(pool.apply_async(optimize.leastsq, args=(perr.res, perr._par, (x, numpy.random.normal(y, sigma), sigma), None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100, None)))
					if method=='simplex':
						results.append(pool.apply_async(optimize.fmin, args=(perr.residuum, perr._par, (x, numpy.random.normal(y, sigma), sigma), xtol, ftol, maxfev, None, 0, 0, 0)))
				pool.close()
				pool.join()
				for i in range(err_sim):
					if method=='leastsq':
						self._par_err_models[i, :]= results[i].get()[0]
					elif method=='simplex':
						self._par_err_models[i, :]= results[i].get()
			else:
				for i in range(err_sim):
					perr = deepcopy(perr_init)
					if method=='leastsq':
						try:
							model_err = optimize.leastsq(perr.res, perr._par, (x, numpy.random.normal(y, sigma), sigma), maxfev=maxfev, ftol=ftol, xtol=xtol, warning=warning)
						except TypeError:
							model_err = optimize.leastsq(perr.res, perr._par, (x, numpy.random.normal(y, sigma), sigma), maxfev=maxfev, ftol=ftol, xtol=xtol)
						self._par_err_models[i, :] = model_err[0]
					if method=='simplex':
						try:
							model_err = optimize.fmin(perr.residuum, perr._par, (x, numpy.random.normal(y, sigma), sigma), disp=0,ftol=ftol, xtol=xtol, warning=warning)
						except TypeError:
							model_err = optimize.fmin(perr.residuum, perr._par, (x, numpy.random.normal(y, sigma), sigma), disp=0, ftol=ftol, xtol=xtol)
						self._par_err_models[i, :] = model_err

			self._par_err = numpy.std(self._par_err_models, 0)
		else:
			self._par_err = None


	def plot(self, x, y=None):
		if y!=None:
			pylab.plot(x, y, 'ok')
		pylab.plot(x, self(x), '-r')
		pylab.show()

class fit_profile2D(object):
	def __init__(self, par, func, guess_par=None, args=None):
		self._par = par
		self._func = func
		self._guess_par = guess_par
		self._args=None

	def __call__(self, x, y):
		return self._func(x, y)

	def res(self, par, x, y, z, sigma=None, args=None):
		self._par = par
		if sigma==None:
			return z-self(x, y)
		else:
			return old_div((z-self(x, y)),sigma)

	def fit(self, x, y, z, sigma=None, p0=None, ftol=1e-4, xtol=1e-4, warning=True):
		if  p0 == None and self._guess_par!=None:
			self._guess_par(x, y, z)
		p0 = self._par

		try:
			model = optimize.leastsq(self.res, p0, (x, y, z,  sigma), maxfev=9999, ftol=ftol, xtol=xtol, warning=warning)#, factor=100)#, Dfun=dev_gaussian, col_deriv=True)
		except TypeError:
			model = optimize.leastsq(self.res, p0, (x, y, z,  sigma), maxfev=9999, ftol=ftol, xtol=xtol)#, factor=100)#, Dfun=dev_gaussian, col_deriv=True)
		self._par = model[0]

class parFile(fit_profile1D):
	def freePar(self):
		parameters=[]
		for n in self._names:
			if self._profile_type[n]=='Gauss':
				if self._fixed[n]['flux']==1:
					parameters.append(float(self._parameters[n]['flux']))
				if self._fixed[n]['vel']==1:
					parameters.append(float(self._parameters[n]['vel']))
				if self._fixed[n]['disp']==1:
					parameters.append(float(self._parameters[n]['disp']))
			if self._profile_type[n]=='Poly':
				for p in range(int(self._parameters[n]['order'])):
					parameters.append(float(self._parameters[n]['coeff_%d'%(p)]))
			if self._profile_type[n]=='TemplateScale':
				if self._fixed[n]['scale']==1:
					parameters.append(float(self._parameters[n]['scale']))
		self._par = parameters

	def restoreResult(self):
		m=0
		for n in self._names:
			if self._profile_type[n]=='TemplateScale':
				if self._fixed[n]['scale']==1:
					self._parameters[n]['scale']=self._par[m]
					m+=1
				self._parameters[n]['start_wave']=float(self._parameters[n]['start_wave'])
				self._parameters[n]['end_wave']=float(self._parameters[n]['end_wave'])
				self._parameters[n]['TemplateSpec']=self._parameters[n]['TemplateSpec']
			if self._profile_type[n]=='Gauss':
				if self._fixed[n]['flux']==1:
					self._parameters[n]['flux'] = self._par[m]
					m+=1
				if self._fixed[n]['vel']==1:
					self._parameters[n]['vel'] = self._par[m]
					m+=1
				if self._fixed[n]['disp']==1:
					self._parameters[n]['disp']=self._par[m]
					m+=1
				self._parameters[n]['restwave']=float(self._parameters[n]['restwave'])
			if self._profile_type[n]=='Poly':
				for p in range(int(self._parameters[n]['order'])):
					if self._fixed[n]['coeff_%d'%(p)]==1:
						self._parameters[n]['ceff_%d'%(p)]=self._par[m]
		for n in self._names:
			if self._profile_type[n]=='TemplateScale':
				if self._fixed[n]['scale']!=1:
					try:
						float(self._parameters[n]['scale'])
					except ValueError:
						line = self._parameters[n]['scale'].split(':')
						if len(line)==1:
							self._parameters[n]['scale'] = self._parameters[line[0]]['scale']
						else:
							self._parameters[n]['scale'] = self._parameters[line[0]]['scale']*float(line[1])
			if self._profile_type[n]=='Gauss':
				if self._fixed[n]['flux']!=1:
					try:
						self._parameters[n]['flux']=float(self._parameters[n]['flux'])
					except ValueError:
						line = self._parameters[n]['flux'].split(':')
						if len(line)==1:
							self._parameters[n]['flux'] = self._parameters[line[0]]['flux']
						else:
							self._parameters[n]['flux'] = self._parameters[line[0]]['flux']*float(line[1])
				if self._fixed[n]['vel']!=1:
					try:
						self._parameters[n]['vel']=float(self._parameters[n]['vel'])
					except ValueError:
						self._parameters[n]['vel'] = self._parameters[self._parameters[n]['vel']]['vel']
				if  self._fixed[n]['disp']!=1:
					try:
						self._parameters[n]['disp']=float(self._parameters[n]['disp'])
					except ValueError:
						self._parameters[n]['disp'] = self._parameters[self._parameters[n]['disp']]['disp']

	def guessPar(self, x, y):
		w = self._guess_window
		dx = numpy.median(x[1:]-x[:-1])
		temp_y = deepcopy(y)
		for n in self._names:
			if self._profile_type[n]=='TemplateScale':
				if self._fixed[n]['scale']==1:
					select_match = numpy.logical_and(numpy.logical_and(self._template_spec[n]._wave>=float(self._parameters[n]['start_wave']),self._template_spec[n]._wave<=float(self._parameters[n]['end_wave'])),numpy.in1d(self._template_spec[n]._wave,x))
					scale_guess = old_div(numpy.sum(y[select_match]),numpy.sum(self._template_spec[n]._data[select_match]))*0.95
					self._parameters[n]['scale']=scale_guess
					temp_y[select_match]=temp_y[select_match]-self._template_spec[n]._data[select_match]*scale_guess
			if self._profile_type[n]=='Gauss':
				restwave=float(self._parameters[n]['restwave'])
				if self._fixed[n]['vel']==1:
					vel=float(self._parameters[n]['vel'])
					select = numpy.logical_and(x>restwave*(vel/300000.0 +1)-w/2.0, x<restwave*(vel/300000.0 +1)+w/2.0)
					idx=numpy.argsort(temp_y[select])
					vel = (old_div(x[select][idx[-1]],restwave)-1)*300000.0
					self._parameters[n]['vel'] = vel
				if self._fixed[n]['flux']==1:
					try:
						vel=float(self._parameters[n]['vel'])
					except ValueError:
						vel = self._parameters[self._parameters[n]['vel']]['vel']
					select = numpy.logical_and(x>restwave*(vel/300000.0 +1)-w/2.0, x<restwave*(vel/300000.0 +1)+w/2.0)
					flux = numpy.sum(temp_y[select])*dx
					self._parameters[n]['flux']=flux
				if self._fixed[n]['disp']==1:
					try:
						vel=float(self._parameters[n]['vel'])
					except ValueError:
						vel = self._parameters[self._parameters[n]['vel']]['vel']
					select = numpy.logical_and(x>restwave*(vel/300000.0 +1)-w/2.0, x<restwave*(vel/300000.0 +1)+w/2.0)
					try:
						width = numpy.sqrt(old_div(numpy.sum((temp_y[select]*(x[select]-restwave*(vel/300000.0 +1))**2)),(numpy.sum(temp_y[select]))))
						if width>self._spec_res and numpy.isnan(width)==False:
							disp = old_div(numpy.sqrt(width**2-self._spec_res**2),(restwave*(vel/300000.0+1)))*300000.0
							self._parameters[n]['disp']=disp
			# else:
			#self._parameters[n]['disp']=0.0
					except:
						pass
		#print self._parameters
		self.freePar()


	def _profile(self, x):
		y = numpy.zeros(len(x))
		m=0
		for n in self._names:
			if self._profile_type[n]=='TemplateScale':
				if self._fixed[n]['scale']==1:
					scale = self._par[m]
					self._parameters[n]['scale']=self._par[m]
					m+=1
				else:
					try:
						scale = float(self._parameters[n]['scale'])
					except ValueError:
						line = self._parameters[n]['scale'].split(':')
						if len(line)==1:
							scale = self._parameters[line[0]]['scale']
						else:
							scale = float(self._parameters[line[0]]['scale'])*float(line[1])
				scale_spec = self._template_spec[n]._data
				scale_wave = self._template_spec[n]._wave
				scale_spec[scale_wave<=float(self._parameters[n]['start_wave'])]=0
				scale_spec[scale_wave>=float(self._parameters[n]['end_wave'])]=0
				select_match = numpy.in1d(scale_wave,x)
				y+=scale_spec[select_match]*scale

			elif self._profile_type[n] == 'Poly':
				for p in range(int(self._parameters[n]['order'])):
					coeff = self._par[m]
					self._parameters[n]['coeff_%d'%(p)]=self._par[m]
					y+= coeff*(x/1000.0)**p
					m+=1

			elif self._profile_type[n]=='Gauss':
				if self._fixed[n]['flux']==1:
					flux=self._par[m]
					self._parameters[n]['flux']=self._par[m]
					m+=1
				else:
					try:
						flux = float(self._parameters[n]['flux'])
					except ValueError:
						line = self._parameters[n]['flux'].split(':')
						if len(line)==1:
							flux = self._parameters[line[0]]['flux']
						else:
							flux = float(self._parameters[line[0]]['flux'])*float(line[1])
				if self._fixed[n]['vel']==1:
					vel=self._par[m]
					self._parameters[n]['vel']=vel
					wave = float(self._parameters[n]['restwave'])*(vel/300000.0 +1)
					m+=1
				else:
					try:
						vel= float(self._parameters[n]['vel'])
					except ValueError:
						vel = float(self._parameters[self._parameters[n]['vel']]['vel'])
					wave = float(self._parameters[n]['restwave'])*(vel/300000.0 +1)
				if self._fixed[n]['disp']==1:
					disp=self._par[m]
					self._parameters[n]['disp']=disp
					m+=1
				else:
					try:
						disp= float(self._parameters[n]['disp'])
					except ValueError:
						disp = float(self._parameters[self._parameters[n]['disp']]['disp'])

				width=numpy.sqrt((disp/300000.0*wave)**2+self._spec_res**2)
				y += old_div(flux*numpy.exp(-0.5*(old_div((x-wave),width))**2),(fact*numpy.fabs(width)))
		return y

	def __init__(self, file,spec_res=0):
		fpar = open(file, 'r')
		lines = fpar.readlines()
		self._names=[]
		self._spec_res=spec_res
		self._profile_type={}
		self._parameters={}
		self._fixed={}
		self._template_spec={}

		par_comp={}
		par_fix ={}
		for i in range(len(lines)):
			line = lines[i].split()
			if len(line)>0:
				if line[0]=='Gauss:'  or line[0]=='Poly:'or line[0]=='TemplateScale:':
					if len(par_comp)!=0:
						self._parameters[self._names[-1]]=par_comp
						self._fixed[self._names[-1]] = par_fix
						par_comp={}
						par_fix ={}
					self._names.append(line[1])
					self._profile_type[line[1]] = line[0][:-1]
				else:
					par_comp[line[0]] = line[1]
					if len(line)>2:
						par_fix[line[0]] = int(line[2])
					else:
						par_fix[line[0]] = 0
		self._parameters[self._names[-1]]=par_comp
		self._fixed[self._names[-1]] = par_fix
		for n in self._names:
			if self._profile_type[n]=='TemplateScale':
				spec = lvmdrp.Spectrum1D()
				spec.loadFitsData(self._parameters[n]['TemplateSpec'])
				self._template_spec[n]=spec
		self.freePar()
		fit_profile1D.__init__(self, self._par, self._profile)




class Gaussian(fit_profile1D):
	def _profile(self, x):
		return old_div(self._par[0]*numpy.exp(-0.5*(old_div((x-self._par[1]),abs(self._par[2])))**2),(fact*abs(self._par[2])))

	def _guess_par(self, x, y):
		sel = numpy.isfinite(y)
		dx = numpy.median(x[1:]-x[:-1])
		self._par[0] = numpy.sum(y[sel])
		self._par[1] = old_div(numpy.sum(x[sel]*y[sel]),self._par[0])
		self._par[2] = numpy.sqrt(old_div(numpy.sum((y[sel]*(x[sel]-self._par[1])**2)),self._par[0]))
		self._par[0]*=dx

	def __init__(self, par):
		fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Gaussian_const(fit_profile1D):
	def _profile(self, x):
		return old_div(self._par[0]*numpy.exp(-0.5*(old_div((x-self._par[1]),self._par[2]))**2),(fact*self._par[2])) + self._par[3]

	def _guess_par(self, x, y):
		sel = numpy.isfinite(y)
		dx = numpy.median(x[1:]-x[:-1])
		ymin = numpy.min(y[sel])
		self._par[0] = numpy.sum(y[sel]-ymin)
		self._par[1] = old_div(numpy.sum(x[sel]*(y[sel]-ymin)),self._par[0])
		self._par[2] = numpy.sqrt(old_div(numpy.sum(((y[sel]-ymin)*(x[sel]-self._par[1])**2)),(self._par[0])))
		self._par[3] = ymin
		self._par[0]*= dx

	def __init__(self, par):
		fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Gaussian_poly(fit_profile1D):
	def _profile(self, x):
		return old_div(self._par[0]*numpy.exp(-0.5*(old_div((x-self._par[1]),self._par[2]))**2),(fact*self._par[2])) + numpy.polyval(self._par[3:],x)

	def _guess_par(self, x, y):
		sel = numpy.isfinite(y)
		dx = abs(x[1]-x[0])
		self._par[0] = numpy.sum(y[sel])
		self._par[1] = old_div(numpy.sum(x[sel]*y[sel]),self._par[0])
		self._par[2] = numpy.sqrt(old_div(numpy.sum((y[sel]*(x[sel]-self._par[1])**2)),self._par[0]))
		self._par[0]*=dx

	def __init__(self, par):
		fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Gaussians(fit_profile1D):
	def _profile(self, x):
		y = numpy.zeros(len(x), dtype=numpy.float32)
		ncomp = old_div(len(self._par),3)
		for i in range(ncomp):
			y += old_div(self._par[i]*numpy.exp(-0.5*(old_div((x-self._par[i+ncomp]),abs(self._par[i+2*ncomp])))**2),(fact*abs(self._par[i+2*ncomp])))
		return y

	def __init__(self, par):
		fit_profile1D.__init__(self, par, self._profile)

class Gaussians_width(fit_profile1D):
	def _profile(self, x):
		y = numpy.zeros(len(x))
		ncomp = len(self._args)
		for i in range(ncomp):
			y += old_div(self._par[i+1]*numpy.exp(-0.5*(old_div((x-self._args[i]),self._par[0]))**2),(fact*self._par[0]))
		return y

	def __init__(self, par, args):
		fit_profile1D.__init__(self, par, self._profile, args=args)

class Gaussians_flux(fit_profile1D):
	def _profile(self, x):
		y = numpy.zeros(len(x))
		ncomp = len(self._par)
		for i in range(ncomp):
			y += old_div(self._par[i]*numpy.exp(-0.5*(old_div((x-self._args[0]),self._args[1]))**2),(fact*self._args[1]))
		return y

	def __init__(self, par, args):
		fit_profile1D.__init__(self, par, self._profile, args=args)

class Gaussians_offset(fit_profile1D):
	def _profile(self, x):
		y = numpy.zeros(len(x))
		ncomp = len(self._args)
		for i in range(ncomp):
			y += old_div(self._par[i+1]*numpy.exp(-0.5*(old_div((x-self._args[i]+self._par[-1]),self._par[0]))**2),(fact*self._par[0]))
		return y

	def __init__(self, par, args):
		fit_profile1D.__init__(self, par, self._profile, args=args)



class Gauss_Hermite(fit_profile1D):
	def _profile(self, x):
		a, mean, sigma, h3, h4 = self._par
		w  = old_div((x-mean),sigma)
		H3 = (2.828427*w**3 - 4.242641*w)*0.408248
		H4 = (4.*w**4 - 12.*w**2 + 3.)*0.204124
		y = old_div(a*numpy.exp(-0.5*w**2)*(1. + h3*H3 + h4*H4),(fact*sigma))
		return y

	def _guess_par(self, x, y):
		sel = numpy.isfinite(y)
		self._par[0] = numpy.sum(y[sel])
		self._par[1] = old_div(numpy.sum(x[sel]*y[sel]),self._par[0])
		self._par[2] = numpy.sqrt(old_div(numpy.sum((y[sel]*(x[sel]-self._par[1])**2)),self._par[0]))
		self._par[3] = 0.
		self._par[4] = 0.

	def __init__(self, par):
		fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Exponential_constant(fit_profile1D):
	def _profile(self, x):
		scale, time, const = self._par
		y = scale*numpy.exp(old_div(x,time))+const
		return y

	def __init__(self, par):
		fit_profile1D.__init__(self, par, self._profile)

class Gaussian2D(fit_profile2D):
	def _profile(self, x, y):
		return old_div(self._par[0]*numpy.exp(-0.5*((old_div((x-self._par[1]),self._par[3]))**2+(old_div((y-self._par[2]),self._par[4]))**2)),(fact*fact*self._par[3]*self._par[4]))

	def _guess_par(self, x, y, z):
		self._par[0]  = numpy.sum(z)
		self._par[1]  = old_div(numpy.sum(x*z),self._par[0])
		self._par[2]  = old_div(numpy.sum(y*z),self._par[0])
		indcol = numpy.around(x)==numpy.around(self._par[1])
		indrow = numpy.around(y)==numpy.around(self._par[2])
		self._par[3]  = numpy.sqrt(old_div(numpy.sum(z[indrow]*(x[indrow]-self._par[2])**2),numpy.sum(z[indrow])))
		self._par[4]  = numpy.sqrt(old_div(numpy.sum(z[indcol]*(y[indcol]-self._par[1])**2),numpy.sum(z[indcol])))

	def __init__(self, par):
		fit_profile2D.__init__(self, par, self._profile, self._guess_par)

class LegandrePoly(object):
	def __init__(self, coeff, min_x=None, max_x=None):
		self._min_x = min_x
		self._max_x = max_x
		self._coeff = coeff
		self._poly =[]
		for i in range(len(coeff)):
			self._poly.append(special.legendre(i))

	def __call__(self, x):
		y = numpy.zeros(len(x), dtype=numpy.float32)
		if self._min_x==None:
			self._min_x = numpy.min(x)
		if self._max_x==None:
			self._max_x = numpy.max(x)
		x_poly = (x-self._min_x)*1.98/numpy.abs((numpy.abs(self._max_x)-numpy.abs(self._min_x)))-0.99
		for i in range(len(self._coeff)):
			y+=self._poly[i](x_poly)*self._coeff[i]
		return y

	def fit(self, x, y):
		eigen_poly = numpy.zeros(( len(x), len(self._coeff)), dtype=numpy.float32)
		for i in range(len(self._coeff)):
				self._coeff = numpy.zeros(len(self._coeff))
				self._coeff[i]=1
				eigen_poly[:, i] = self(x)
#        print eigen_poly, y
		self._coeff=numpy.linalg.lstsq(eigen_poly, y)[0]


def gaussian(p, x):
	return old_div(abs(p[0])*numpy.exp(-0.5*(old_div((x-p[1]),abs(p[2])))**2),(fact*abs(p[2])))

def gaussian_const(p, x):
	return old_div(p[0]*numpy.exp(-0.5*(old_div((x-p[1]),p[2]))**2),(fact*p[2])) + p[3]

def gaussian2d(p, x, y):
	return old_div(p[0]*numpy.exp(-0.5*((old_div((x-p[1]),p[3]))**2+(old_div((y-p[2]),p[4]))**2)),(fact*fact*p[3]*p[4]))

def gaussian_poly(p, x):
	return old_div(p[0]*numpy.exp(-0.5*(old_div((x-p[1]),p[2]))**2),(fact*p[2])) + numpy.polyval(p[3:],x)

def gaussian_multi(p, x):
	y = numpy.zeros(len(x))
	ncomp = old_div(len(p),3)
	for i in range(ncomp):
		y += old_div(abs(p[i])*numpy.exp(-0.5*(old_div((x-p[i+ncomp]),abs(p[i+2*ncomp])))**2),(fact*abs(p[i+2*ncomp])))
	return y

def gaussian_width_multi(p, x, pos):
	y = numpy.zeros(len(x))
	ncomp = len(pos)
	for i in range(ncomp):
		y += old_div(p[i+1]*numpy.exp(-0.5*(old_div((x-pos[i]),p[0]))**2),(fact*p[0]))
	return y

def gaussian_width_multi_offset(p, x, pos):
	y = numpy.zeros(len(x))
	ncomp = len(pos)
	for i in range(ncomp):
		y += old_div(p[i+1]*numpy.exp(-0.5*(old_div((x-pos[i]+p[-1]),p[0]))**2),(fact*p[0]))
	return y


def gauss_hermite(p, x):#, a, mean, sigma, h3, h4):

	a, mean, sigma, h3, h4 = p
	w  = old_div((x-mean),sigma)
	H3 = (2.828427*w**3 - 4.242641*w)*0.408248
	H4 = (4.*w**4 - 12.*w**2 + 3.)*0.204124

	return old_div(a*numpy.exp(-0.5*w**2)*(1. + h3*H3 + h4*H4),(fact*sigma))

def res_gaussian(p, x, y, sigma):
	return old_div((y - gaussian(p, x)),sigma)


def res_gaussian_const(p, x, y, sigma):
	return old_div((y - gaussian_const(p, x)),sigma)

def res_gaussian2d(p, x, y, z, sigma):
	return old_div((z - gaussian2d(p, x, y)),sigma)

def res_gaussian_poly(p, x, y, sigma):
	return old_div((y - gaussian_poly(p, x)),sigma)

def res_gaussian_multi(p, x, y, sigma):
	return old_div((y - gaussian_multi(p, x)),sigma)

def res_gaussian_width_multi(p, x, y, pos, sigma):
	return old_div((y - gaussian_width_multi(p, x, pos)),sigma)

def res_gaussian_width_multi_offset(p, x, y, pos, sigma):
	return old_div((y - gaussian_width_multi_offset(p, x, pos)),sigma)

def res_gauss_hermite(p, x, y, sigma):
	return old_div((y - gauss_hermite(p, x)),sigma)

def chisq_gaussian_width_multi(p, x, y, pos, sigma):
	return numpy.sum((old_div((y - gaussian_width_multi(p, x, pos)),sigma))**2)



def dev_gaussian(p, x, y, sigma):

	z = numpy.zeros((3,len(x)))

	d = x-p[1]
	e = -0.5*(old_div((d),p[2]))**2

	z[0] = old_div(- numpy.exp(e),(fact*p[2]))
	z[1] = old_div(- p[0]*d*numpy.exp(e),(fact*p[2]**3))
	z[2] = old_div(- p[0]*(d**2)*numpy.exp(e),(fact*p[2]**4))

	return old_div(z,sigma)

def dev_gaussian_const(p, x, y, sigma):

	z = numpy.zeros((4,len(x)))

	d = x-p[1]
	e = -0.5*(old_div((d),p[2]))**2

	z[0] = old_div(- numpy.exp(e),(fact*p[2]))
	z[1] = old_div(- p[0]*d*numpy.exp(e),(fact*p[2]**3))
	z[2] = old_div(- p[0]*(d**2)*numpy.exp(e),(fact*p[2]**4))
	z[3] = - 1.

	return old_div(z,sigma)

def dev_gaussian_poly1d(p, x, y, sigma):

	z    = numpy.zeros((len(p),len(x)))

	d = x-p[1]
	e = -0.5*(old_div(d,p[2]))**2

	z[0] = old_div(- numpy.exp(e),(fact*p[2]))
	z[1] = old_div(- p[0]*d*numpy.exp(e),(fact*p[2]**3))
	z[2] = old_div(- p[0]*(d**2)*numpy.exp(e),(fact*p[2]**4))
	for i in range(len(p)-3):
		z[3+i] = - x**(len(p)-4-i)

	return old_div(z,sigma)

def dev_gaussian_width_multi(p, x, y, pos, sigma):

	ncomp = len(pos)
	z = numpy.zeros((ncomp+1, len(x)))

	for i in range(ncomp):
		d = x-pos[i]
		e = -0.5*(old_div((d),p[0]))**2
		z[0]  += old_div(- p[i+1]*(d**2)*numpy.exp(e),(fact*p[0]**4))
		z[i+1] = old_div(- numpy.exp(e),(fact*p[0]))

	return old_div(z,sigma)

def fit_gaussian(x, y, sigma=1., p0=None, ftol=1e-4, xtol=1e-4, warning=True):

	sel = numpy.isfinite(y)
	dx = abs(x[1]-x[0])

	#if numpy.sum(sel)>3:
	if p0==None or len(p0)!=3:
		p0 = numpy.zeros(3)
		p0[0] = numpy.sum(y[sel])
		p0[1] = old_div(numpy.sum(x[sel]*y[sel]),p0[0])
		p0[2] = numpy.sqrt(old_div(numpy.sum((y[sel]*(x[sel]-p0[1])**2)),p0[0]))
		p0[0]*= dx

	sol = scipy.optimize.leastsq(res_gaussian, p0, (x,y,sigma), maxfev=9999, ftol=ftol, xtol=xtol, warning=warning)#, factor=100)#, Dfun=dev_gaussian, col_deriv=True)

	return sol[0]


def fit_gaussian_const(x, y, sigma=1.,  p0=None, ftol=1e-6, xtol=1e-6, warning=True):

	dx = abs(x[1]-x[0])
	if p0==None or len(p0)!=4:
		p0 = numpy.zeros(4)
		ymin = numpy.min(y)
		p0[0] = numpy.sum(y-ymin)
		p0[1] = old_div(numpy.sum(x*(y-ymin)),p0[0])
		p0[2] = numpy.sqrt(old_div(numpy.sum(((y-ymin)*(x-p0[1])**2)),p0[0]))
		p0[3] = ymin
		p0[0]*= dx

	sol = scipy.optimize.leastsq(res_gaussian_const, p0, (x,y,sigma), maxfev=99999, ftol=ftol, xtol=xtol, warning=warning)[0]#, Dfun=dev_gaussian_const, col_deriv=True)

	##print sol
	#pylab.clf()
	##pylab.axvline(5577.347, ls='--', c='k')
	#pylab.plot(x,y,'k',drawstyle='steps-mid')
	#pylab.plot(x,gaussian_const(sol, x),'r')#,drawstyle='steps-mid')
	#pylab.draw()
	###raw_input()

	return sol

def fit_gaussian_poly(x, y, sigma=1., npoly=0):

	dx = abs(x[1]-x[0])

	p0 = numpy.zeros(3+npoly+1)
	p0[0] = numpy.sum(y)
	p0[1] = old_div(numpy.sum(x*y),p0[0])
	p0[2] = numpy.sqrt(old_div(numpy.sum(y*(x-p0[1])**2),p0[0]))
	p0[0]*= dx

	sol = scipy.optimize.leastsq(res_gaussian_poly, p0, (x,y,sigma))#, Dfun=dev_gaussian_poly, col_deriv=True)

	#pylab.plot(x,y,'k', drawstyle='steps-mid')
	#pylab.plot(x,gaussian_poly(sol[0], x),'-r')#,drawstyle='steps-mid')
	#pylab.draw()
	#raw_input()
	#pylab.clf()

	return sol[0]

def fit_gaussian_multi(x, y, ncomp=1, sigma=1., f0=[1.], m0=[0.], s0=[1.]):

	p0 = numpy.zeros(3*ncomp)
	if len(f0)==ncomp: p0[       :  ncomp] = f0
	else:              p0[       :  ncomp] = f0[0]
	if len(m0)==ncomp: p0[  ncomp:2*ncomp] = m0
	else:              p0[  ncomp:2*ncomp] = m0[0]
	if len(s0)==ncomp: p0[2*ncomp:3*ncomp] = s0
	else:              p0[2*ncomp:3*ncomp] = s0[0]

	sol = scipy.optimize.leastsq(res_gaussian_multi, p0, (x,y,sigma))

	return sol[0]

def fit_gaussian_width_multi(x, y, pos, sigma=1., flux0=1., width0=1.):

	ncomp = len(pos)
	p0    = numpy.zeros(ncomp+1)
	p0[0]         = width0
	p0[1:ncomp+1] = flux0

	sol = scipy.optimize.leastsq(res_gaussian_width_multi, p0, (x,y,pos,sigma), ftol=0.1)

	return sol[0]

def fit_gaussian_width_multi_offset(x, y, pos, sigma=1., flux0=1., width0=1.):

	ncomp = len(pos)
	p0    = numpy.zeros(ncomp+2)
	p0[0]         = width0
	p0[1:ncomp+1] = flux0
	p0[-1]        = 0.

	sol = scipy.optimize.leastsq(res_gaussian_width_multi_offset, p0, (x,y,pos,sigma))#, ftol=0.1)

	#print sol[0][0], sol[0][-1]
	#pylab.plot(x,y,'k')#,drawstyle='steps-mid')
	#pylab.plot(x,gaussian_width_multi_offset(sol[0], x, pos),'r')#,drawstyle='steps-mid')
	#pylab.draw()
	#raw_input()
	#pylab.clf()

	return sol[0]

def fit_gaussian2d(x, y, z, sigma=1.):

	p0     = numpy.zeros(5)
	p0[0]  = numpy.sum(z)
	p0[1]  = old_div(numpy.sum(x*z),p0[0])
	p0[2]  = old_div(numpy.sum(y*z),p0[0])
	indcol = numpy.around(x)==numpy.around(p0[1])
	indrow = numpy.around(y)==numpy.around(p0[2])
	p0[3]  = numpy.sqrt(old_div(numpy.sum(z[indrow]*(x[indrow]-p0[2])**2),numpy.sum(z[indrow])))
	p0[4]  = numpy.sqrt(old_div(numpy.sum(z[indcol]*(y[indcol]-p0[1])**2),numpy.sum(z[indcol])))

	sol = scipy.optimize.leastsq(res_gaussian2d, p0, (x,y,z,sigma))

	return sol[0]

def fit_gauss_hermite(x, y, sigma=1., p0=None):

	sel = numpy.isfinite(y)

	#if numpy.sum(sel)>3:
	if p0==None or len(p0)!=5:
		p0 = numpy.zeros(5)
		p0[0] = numpy.sum(y[sel])
		p0[1] = old_div(numpy.sum(x[sel]*y[sel]),p0[0])
		p0[2] = numpy.sqrt(old_div(numpy.sum((y[sel]*(x[sel]-p0[1])**2)),p0[0]))
		p0[3] = 0.
		p0[4] = 0.

	sol = scipy.optimize.leastsq(res_gauss_hermite, p0, (x,y,sigma), maxfev=9999, ftol=1e-9, xtol=1e-9)#, factor=100)#, Dfun=dev_gaussian, col_deriv=True)

	return sol[0]





