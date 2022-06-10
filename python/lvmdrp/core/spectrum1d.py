from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
from astropy.io import fits as pyfits
import numpy
try:
  import pylab
except:
  pass
from scipy import sparse
from scipy import interpolate
from scipy import ndimage
from . import fit_profile
from copy import  deepcopy

class Spectrum1D(object):
    def __init__(self, wave=None, data=None, error=None, mask=None, inst_fwhm=None):
        self._wave = wave
        self._data = data
        if data is not None:
            self._dim = self._data.shape[0]
            self._pixels = numpy.arange(self._dim)
        self._error = error
        self._mask = mask
        self._inst_fwhm = inst_fwhm

    def __sub__(self, other):
        if isinstance(other, Spectrum1D):
            data = numpy.zeros_like(self._data)
            select_zero = self._data==0
            data = self._data-other._data
            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero]=0
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask= self._mask
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero]=0
            else:
                mask =None
            if self._error is not None and other._error is not None:
                error=numpy.sqrt(self._error**2+other._error**2)
            elif self._error is not None:
                error = self._error
            else:
                error = None
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if (error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8')):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec


        elif isinstance(other,  numpy.ndarray):
            data = self._data - other
            if self._error  is not None:
                error = self._error
            else:
                error = None
            mask = self._mask
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data - other
                if self._error is not None:
                    error = self._error
                else:
                    error = None
                mask = self._mask
                if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                    data=data.astype(numpy.float32)
                if error is not None:
                    if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                        error=error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
                return spec
            except:
                #raise exception if the type are not matching in general
                raise TypeError("unsupported operand type(s) for -: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __add__(self, other):
        if isinstance(other, Spectrum1D):
            other._data.astype(numpy.float32)
            data = numpy.zeros_like(self._data)
            select_zero = self._data==0
            data = self._data+other._data
            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero]=0
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask= self._mask
                select_zero = numpy.logical_and(select_zero, mask)
                data[select_zero]=0
            else:
                mask =None
            if self._error is not None and other._error is not None:
                error=numpy.sqrt(self._error**2+other._error**2)
            elif self._error is not None:
                error = self._error
            else:
                error = None
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec


        elif isinstance(other,  numpy.ndarray):
            data = self._data + other
            if self._error  is not None:
                error = self._error
            else:
                error = None
            mask = self._mask
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                data = self._data + other
                if self._error is not None:
                    error = self._error
                else:
                    error = None
                mask = self._mask
                if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                    data=data.astype(numpy.float32)
                if error is not None:
                    if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                        error=error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
                return spec
            except:
                #raise exception if the type are not matching in general
                raise TypeError("unsupported operand type(s) for -: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __truediv__(self, other):

        if isinstance(other, Spectrum1D):
            other._data=other._data.astype(numpy.float32)
            select = other._data!=0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select)>0:
                data[select] = old_div(self._data[select],other._data[select].astype(numpy.float32))

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask= self._mask
            else:
                mask =None
            if self._error is not None and other._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select)>0:
                    error[select]=numpy.sqrt((old_div(self._error[select],other._data[select]))**2+(old_div(self._data[select]*other._error[select],other._data[select]**2))**2)
            elif self._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select)>0:
                    error[select] = old_div(self._error[select],other._data[select])
                    error[numpy.logical_not(select)]=numpy.max(self._error)
            else:
                error = None
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec

        elif isinstance(other,  numpy.ndarray):
            if other!=0:
                data = old_div(self._data, other)
                if self._error  is not None:
                    error = old_div(self._error, other)
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None

            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                if other!=0.0:
                    data = old_div(self._data,other)
                    if self._error is not None:
                        error = old_div(self._error, other)
                    else:
                        error = None
                    mask = self._mask
                else:
                    data = numpy.zeros_like(self._data)
                    if self._error is not None:
                        error = numpy.zeros_like(self._data)
                    else:
                        error = None
                    if self._mask is not None:
                        mask = numpy.zeros(self._data.shape[0], dtype='bool')
                    else:
                        mask=None
                if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                    data=data.astype(numpy.float32)
                if error is not None:
                    if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                        error=error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
                return spec
            except:
                #raise exception if the type are not matching in general
                raise TypeError("unsupported operand type(s) for /: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __rdiv__(self, other):

        if isinstance(other, Spectrum1D):
            other._data=other._data.astype(numpy.float32)
            select = self._data!=0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select)>0:
                data[select] = old_div(other._data[select].astype(numpy.float32),self._data[select])

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask= self._mask
            else:
                mask =None
            if self._error is not None and other._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select)>0:
                    error[select]=numpy.sqrt((old_div(other._error[select],self._data[select]))**2+(old_div(other._data[select]*self._error[select],self._data[select]**2))**2)
            elif self._error is not None:
                error = numpy.zeros_like(self._error)
                if numpy.sum(select)>0:
                    error[select] = old_div(other._error[select],self._data[select])
                    error[numpy.logical_not(select)]=numpy.max(self._error)
            else:
                error = None
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error= error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec

        elif isinstance(other,  numpy.ndarray):
            select = self._data!=0.0
            data = numpy.zeros_like(self._data)
            if numpy.sum(select)>0:
                data[select] = old_div(other[select], self._data[select])
                if self._error  is not None:
                    error = numpy.zeros_like(self._error)
                    if numpy.sum(select)>0:
                        error[select] = old_div(other[select] *self._error[select],self._data[select]**2)
                    else:
                        error=None
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None

            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error= error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
   ##         try:
                select = self._data!=0.0
                data = numpy.zeros_like(self._data)
                if numpy.sum(select)>0:
                    data[select] = old_div(other, self._data[select])
                    if self._error is not None:
                        error = numpy.zeros_like(self._error)
                        if numpy.sum(select)>0:
                            error[select] = old_div(other *self._error[select],self._data[select]**2)
                        else:
                            error=None
                    else:
                        error = None
                    mask = self._mask
                else:
                    data = numpy.zeros_like(self._data)
                    if self._error is not None:
                        error = numpy.zeros_like(self._data)
                    else:
                        error = None
                    if self._mask is not None:
                        mask = numpy.zeros(self._data.shape[0], dtype='bool')
                    else:
                        mask=None

                if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                    data=data.astype(numpy.float32)
                if error is not None:
                    if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                        error=error.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
                return spec
      ##      except:
                #raise exception if the type are not matching in general
        ##        raise TypeError("unsupported operand type(s) for /: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))


    def __mul__(self, other):

        if isinstance(other, Spectrum1D):
            other._data.astype(numpy.float32)
            data = self._data*other._data.astype(numpy.float32)

            if self._mask is not None and other._mask is not None:
                mask = numpy.logical_or(self._mask, other._mask)
            elif self._mask is None and other._mask is not None:
                mask = other._mask
            elif self._mask is not None and other._mask is None:
                mask= self._mask
            else:
                mask =None
            if self._error is not None and other._error is not None:
                error=numpy.sqrt((self._error*other._data+self._data*other._error)**2)
            elif self._error is not None and other._error is None:
                error = self._error*other._data
            else:
                error = None
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
               if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec


        elif isinstance(other,  numpy.ndarray):
            if other!=0:
                data = self._data * other
                if self._error  is not None:
                    error = self._error * other
                else:
                    error = None
                mask = self._mask
            else:
                data = numpy.zeros_like(self._data)
                if self._error is not None:
                    error = numpy.zeros_like(self._data)
                if self._mask is not None:
                    mask = numpy.zeros(self._data.shape[0], dtype="bool")
                else:
                    mask = None
            if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
            if error is not None:
                if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                    error=error.astype(numpy.float32)
            spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
            return spec
        else:
            # try to do addtion for other types, e.g. float, int, etc.
          #  try:
                data = self._data*other
                if self._error is not None:
                    error = self._error * other
                    if error.dtype==numpy.float64 or error.dtype==numpy.dtype('>f8'):
                        error.astype(numpy.float32)
                else:
                    error = None
                mask = self._mask
                if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                   data= data.astype(numpy.float32)

                spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
                return spec

    def __pow__(self, other):
                data = self._data**other
                if self._error is not None:
                    error = 1.0/float(other)*self._data**(other-1)*self._error
                else:
                    error = None
                mask = self._mask

                if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                    data=data.astype(numpy.float32)
                spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)
                return spec
           # except:
                #raise exception if the type are not matching in general
              #  raise TypeError("unsupported operand type(s) for *: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))
    def __rpow__(self, other):
        data = other**self._data
        error = None
        mask = self._mask

        if data.dtype==numpy.float64 or data.dtype==numpy.dtype('>f8'):
                data=data.astype(numpy.float32)
        spec = Spectrum1D(wave=self._wave, data = data,  error = error,  mask=mask)

        return spec

    def __lt__(self, other):
        return self._data<other

    def __le__(self, other):
        return self._data<=other

    def __eq__(self, other):
        return self._data==other

    def __ne__(self, other):
        return self._data!=other

    def __gt__(self, other):
        return self._data()>other

    def __ge__(self, other):
        return self._data()>=other

    def loadFitsData(self, file, extension_data=None, extension_mask=None,  extension_error=None,  extension_wave=None, extension_fwhm=None,  extension_hdr=None, logwave=False):
        """
            load information from a FITS file into a Spectrum1D object.
            A single or multiple extension FITS file are possible inputs.

            Parameters
            --------------
            filename : string
                Name or Path of the FITS image from which the data shall be loaded

            extension_data : int, optional with default: None
                Number of the FITS extension containing the data

            extension_mask : int, optional with default: None
                Number of the FITS extension containing the masked pixels

            extension_error : int, optional with default: None
                Number of the FITS extension containing the errors for the values
        """

        hdu = pyfits.open(file)
        if extension_data is None and extension_mask is None and extension_error is None and extension_wave is None and extension_fwhm is None:
                self._data = hdu[0].data
                hdr = hdu[0].header
                self._dim = self._data.shape[0] # set shape
                self._pixels = numpy.arange(self._dim)
                #self.setHeader(header = hdu[0].header, origin=file)
                if len(hdu)>1:
                    for i in range(1, len(hdu)):
                        if hdu[i].header['EXTNAME'].split()[0]=='ERROR':
                            self._error = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0]=='BADPIX':
                            self._mask = hdu[i].data.astype('bool')
                        elif hdu[i].header['EXTNAME'].split()[0]=='WAVE':
                            self._wave = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0]=='INSTFWHM':
                            self._inst_fwhm = hdu[i].data
                if self._wave is None:
                    self._wave = self._pixels*hdr['cdelt1']+hdr['crval1']
        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data
            if extension_error is not None:
                self._error = hdu[extension_error].data
            if extension_wave is not None:
                self._wave = hdu[extension_wave].data
            if extension_fwhm is not None:
                self._inst_fwhm = hdu[extension_fwhm].data
        hdu.close()

        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header, origin=file)

    def writeFitsData(self, filename, extension_data=None, extension_mask=None,  extension_error=None, extension_wave=None, extension_fwhm=None):
        """
            Save information from a Spectrum1D object into a FITS file.
            A single or multiple extension file are possible to create.

            Parameters
            --------------
            filename : string
                Name or Path of the FITS image from which the data shall be loaded

            extension_data : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the data

            extension_mask : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the masked pixels

            extension_error : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the errors for the values
        """
        hdus=[None, None, None, None, None] # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if extension_data is None and extension_error is None and extension_mask is None:
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._wave is not None:
                hdus[1] = pyfits.ImageHDU(self._wave, name='WAVE')
            if self._inst_fwhm is not None:
                hdus[2] = pyfits.ImageHDU(self._inst_fwhm, name='INSTFWHM')
            if self._error is not None:
                hdus[3] = pyfits.ImageHDU(self._error, name='ERROR')
            if self._mask is not None:
                hdus[4] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data>0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name='DATA')

            # wavelength hdu
            if extension_wave == 0:
                hdu = pyfits.PrimaryHDU(self._wave)
            elif extension_wave>0 and extension_wave is not None:
                hdus[extension_wave] = pyfits.ImageHDU(self._wave, name='WAVE')

            # instrumental FWHM hdu
            if extension_fwhm == 0:
                hdu = pyfits.PrimaryHDU(self._inst_fwhm)
            elif extension_fwhm>0 and extension_fwhm is not None:
                hdus[extension_fwhm] = pyfits.ImageHDU(self._inst_fwhm, name='INSTFWHM')

            # mask hdu
            if extension_mask == 0:
                hdu = pyfits.PrimaryHDU(self._mask.astype('uint8'))
            elif extension_mask>0 and extension_mask is not None:
                hdus[extension_mask] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')

            # error hdu
            if extension_error == 0:
                hdu = pyfits.PrimaryHDU(self._error)
            elif extension_error>0 and extension_error is not None:
                hdus[extension_error] = pyfits.ImageHDU(self._error, name='ERROR')

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except:
                break

        if len(hdus)>0:
            hdu = pyfits.HDUList(hdus) # create an HDUList object
        hdu.writeto(filename, overwrite=True) # write FITS file to disc

    def writeTxtData(self, filename):
        out = open(filename, 'w')
        for i in range(self._dim):
            out.write('%i %f %f\n'%(i+1, self._wave[i], self._data[i]))
        out.close()

    def loadTxtData(self, filename):
        infile = open(filename, 'r')
        lines = infile.readlines()
        wave = numpy.zeros(len(lines),dtype=numpy.float32)
        data = numpy.zeros(len(lines),dtype=numpy.float32)
        for i in range(len(lines)):
            line = lines[i].split()
            wave[i] = float(line[1])
            data[i] = float(line[2])
        self._wave = wave
        self._data = data
        self._dim = len(data)
        infile.close()

    def loadSTDref(self, ref_file, column_wave=0, column_flux=1, delimiter='', header=1):
        dat = open(ref_file, 'r')
        lines = dat.readlines()
        wave = numpy.zeros(len(lines)-header, dtype=numpy.float32)
        data = numpy.zeros(len(lines)-header, dtype=numpy.float32)
        for i in range(header, len(lines)):
            if delimiter=='':
                line = lines[i].split()
            else:
                line = lines.split(delimiter)
            wave[i-header] = line[column_wave]
            data[i-header] = line[column_flux]
        self._data  = data
        self._wave = wave
        self._dim = self._data.shape[0]
        self._pixels = numpy.arange(self._dim)

    def max(self):
        """
            Determines the maximum data value of the spectrum

            Returns: (max,max_wave,max_pos)
            -----------
            max : float
                Maximum data value

            max_wave : float
                Wavelength of maximum data value

            max_pos : int
                Pixel position of the maximum data value

        """
        max = numpy.max(self._data) # get max
        select =self._data==max # select max value
        max_wave = self._wave[select][0] # get corresponding wavelength
        max_pos = self._pixels[select][0] # get corresponding position
        return max, max_wave, max_pos

    def min(self):
        """
            Determines the minimum data value of the spectrum

            Returns: (min,min_wave,min_pos)
            -----------
            min : float
                Minimum data value

            min_wave : float
                Wavelength of minimum data value

            min_pos : int
                Pixel position of the minimum data value

        """
        min = numpy.min(self._data) # get min
        select =self._data==min # select min value
        min_wave = self._wave[select][0] # get corresponding waveength
        min_pos = self._pixels[select][0] # get corresponding position
        return min, min_wave, min_pos

    def getData(self):
        """
            Return the content of the spectrum

            Returns: (pix, wave, data, error mask)
            -----------
            pix : numpy.ndarray (int)
                Array of the pixel positions

            wave : numpy.ndarray (float)
                Array of the wavelength

            data : numpy.ndarray (float)
                Array of the data value

            error : numpy.ndarray (float)
                Array of the corresponding errors

            mask : numpy.ndarray (bool)
                Array of the bad pixel mask
        """
        pix = self._pixels
        wave = self._wave
        data = self._data
        error = self._error
        mask = self._mask
        return pix,  wave,  data,  error,  mask

    def resampleSpec(self, ref_wave, method='spline', err_sim=500, replace_error=1e10):
        if self._wave[-1]<self._wave[0]:
            self._wave=numpy.flipud(self._wave)
            self._data = numpy.flipud(self._data)
            if self._error is not None:
                self._error = numpy.flipud(self._error)
            if self._mask is not None:
                self._mask = numpy.flipud(self._mask)
        if self._mask is not None and numpy.sum(self._mask)==self._dim:
            new_mask = numpy.ones(len(ref_wave), dtype='bool')
            new_data =  numpy.zeros(len(ref_wave), numpy.float32)
            if self._error is None or err_sim==0:
                new_error=None
            else:
                new_error = numpy.ones(len(ref_wave), numpy.float32)*replace_error
        else:
            # replace bad pixels within the spectrum with linear interpolated values
            if  self._mask is not None:
                good_pix = numpy.logical_not(self._mask)
                intp = interpolate.UnivariateSpline(self._wave[good_pix], self._data[good_pix], k=1, s=0)
                clean_data = intp(self._wave)
                if self._pixels[good_pix][0]>0:
                    clean_data[:self._pixels[good_pix][0]]=0
                if self._pixels[good_pix][-1]<self._pixels[-1]:
                    clean_data[self._pixels[good_pix][-1]-1:]=0
            else:
                clean_data = self._data
            select_interp = clean_data!=0
            wave_interp = self._wave[select_interp]
            # perform the interpolation on the data
            if method=='spline':
                intp = interpolate.UnivariateSpline(self._wave[select_interp], clean_data[select_interp], s=0)
                new_data = intp(ref_wave)
            elif method=='linear':
                intp = interpolate.interp1d(self._wave[select_interp], clean_data[select_interp])
                #intp = interpolate.UnivariateSpline(self._wave[select_interp], clean_data[select_interp], k=1, s=0)
                new_data = intp(ref_wave)
            select_out= numpy.logical_or(ref_wave<wave_interp[0],ref_wave>wave_interp[-1])
            new_data[select_out]=0
            select = numpy.logical_or(ref_wave<numpy.min(self._wave), ref_wave>numpy.max(self._wave))

            select_not = numpy.logical_not(select)
            # replace the error of bad pixels within the spectrum to temporarily to zero  for the Monte Carlo simulation
            if self._mask is not None:
                select_goodpix=numpy.logical_not(self._mask)
            else:
                select_goodpix=numpy.ones(self._dim, dtype='bool')

            if self._error is not None and err_sim>0:
                replace_pix = numpy.logical_and(self._mask, clean_data!=0.0)
                self._error[replace_pix]=1e-20
                #select_goodpix = numpy.logical_and(select_goodpix, self._data!=0.0)
                select_goodpix = self._data!=0.0
                sim  = numpy.zeros((err_sim, len(ref_wave)), dtype=numpy.float32)
                data = numpy.zeros(len(self._wave), dtype=numpy.float32)


                for i in range(err_sim):
                    data[select_goodpix] = numpy.random.normal(clean_data[select_goodpix], self._error[select_goodpix]).astype(numpy.float32)
                    if method=='spline':
                        intp = interpolate.UnivariateSpline(self._wave[select_interp], data[select_interp], s=0)
                        out =intp(ref_wave)
                    elif method=='linear':
                        intp = interpolate.interpolate.interp1d(self._wave[select_interp], data[select_interp])
                        out = intp(ref_wave)
                    select_out= numpy.logical_or(ref_wave<wave_interp[0],ref_wave>wave_interp[-1])
                    out[select_out]=0
                    sim[i, select_not] = out[select_not]
                new_error = numpy.std(sim, 0)

            if self._mask is not None:
                badpix=numpy.zeros(ref_wave.shape[0], dtype='bool')
                indices = numpy.arange(self._wave.shape[0])
                nbadpix = numpy.sum(self._mask)
                if nbadpix>0:
                    badpix_id = indices[self._mask]
                    for i in range(len(badpix_id)):
                        badpix_min = badpix_id[i]-2
                        badpix_max = badpix_id[i]+2
                        bound = numpy.clip(numpy.array([badpix_min, badpix_max]), 0, self._dim-1)
                        select_bad = numpy.logical_and(ref_wave>=self._wave[bound[0]], ref_wave<=self._wave[bound[1]])
                        if clean_data[badpix_id[i]]==0:
                            new_data[select_bad]=0
                        badpix = numpy.logical_or(badpix, select_bad)
                badpix = numpy.logical_or(badpix, numpy.logical_or(ref_wave<self._wave[0], ref_wave>self._wave[-1]))

                if self._error is not None and err_sim>0:
        #        new_mask = numpy.logical_and(badpix, new_error>0)
                    new_mask=badpix
                    new_error[new_mask] = replace_error
            else:
                badpix = numpy.logical_or(ref_wave<self._wave[0], ref_wave>self._wave[-1])
                new_data[badpix]=0
            new_mask = badpix
    #        new_data[badpix]=fill_value
     #           print numpy.sum(self._mask), numpy.sum(new_mask)

            if self._error is None or err_sim==0:
                new_error=None
       # if numpy.isnan(new_data[10])==True:
        #    pylab.plot(self._wave[select_interp],'-k')
        #    pylab.show()
        spec_out = Spectrum1D(wave=ref_wave, data=new_data, error=new_error, mask=new_mask)
        return spec_out

    def matchFWHM(self, target_FWHM):
        if self._inst_fwhm is not None:
            if self._mask is not None:
                good_pix = numpy.logical_not(self._mask)
                data = self._data[good_pix]
                wave = self._wave[good_pix]
                fwhm = self._inst_fwhm[good_pix]
                if self._error is not None:
                    error = self._error[good_pix]
                else:
                    error = None
            else:
                data = self._data
                wave = self._wave
                fwhm = self._inst_fwhm
                error = self._error
            gauss_sig = numpy.zeros_like(fwhm)
            select = target_FWHM>fwhm
            gauss_sig[select] = numpy.sqrt(target_FWHM**2-fwhm[select]**2)/2.354
            fact = numpy.sqrt(2.*numpy.pi)
            kernel=old_div(numpy.exp(-0.5*(old_div((wave[:, numpy.newaxis]-wave[numpy.newaxis, :]),gauss_sig[numpy.newaxis, :]))**2),(fact*gauss_sig[numpy.newaxis, :]))
            multiplied = data[:, numpy.newaxis]*kernel
            new_data = old_div(numpy.sum(multiplied, axis=0),numpy.sum(kernel, 0))
            if self._mask is not None:
                self._data[good_pix] = new_data
                self._inst_fwhm[:] = target_FWHM
            if error is not None:
                new_error = old_div(numpy.sqrt(numpy.sum((error[:, numpy.newaxis]*kernel)**2, axis=0)),numpy.sum(kernel, 0))
                if self._mask is not None:
                    self._error[good_pix] = new_error
                else:
                    self._error = new_error

    def binSpec(self, new_wave):
        new_disp = new_wave[1:]-new_wave[:-1]
        new_disp = numpy.insert(new_disp, 0, new_disp[0])
        data_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
        mask_out = numpy.zeros(len(new_wave), dtype="bool")
        if self._mask is not None:
            mask_in = numpy.logical_and(self._mask)
        else:
            mask_in = numpy.ones(len(self._wave), dtype="bool")
        masked_data = self._wave[mask_in]
        masked_wave = self._wave[mask_in]
        if self._error is not None:
            error_out = numpy.zeros(len(new_wave), dtype=numpy.float32)
            masked_error = self._error[mask_in]
        else:
            error_out = None
        bound_min = new_wave-new_disp/2.0
        bound_max = new_wave+new_disp/2.0

        disp = bound_max-bound_min
        for i in range(len(new_wave)):
            select = numpy.logical_and(masked_wave>= bound_min[i], masked_wave <= bound_max[i])
            if numpy.sum(select) > 0:
#                data_out[i] = numpy.mean(self._data[mask_in][select])
                data_out[i] = old_div(numpy.sum(numpy.abs(masked_wave[select]-new_wave[i])*self._data[mask_in][select]),numpy.sum(numpy.abs(masked_wave[select]-new_wave[i])))
                if self._error is not None:
                    error_out[i] = numpy.sqrt(old_div(numpy.sum(masked_error[select]**2),numpy.sum(select)**2))
            else:
                data_out[i] = 0.0
                mask_out[i] = True
        data_out = numpy.interp(new_wave, masked_wave, self._data[mask_in])
      #      numpy.delete(masked_wave, select)
       #     numpy.delete(masked_data,  select)
        #    numpy.delete(masked_error, select)
        spec = Spectrum1D(data = data_out, wave = new_wave, error=error_out, mask=mask_out)
        return spec

    def smoothSpec(self, size, method='gauss', mode='nearest'):
        """
            Smooth the spectrum

            Parameters
            --------------
            size : float or int
                Size of the smooth window or Gaussian width (sigma)
            method : string, optional with default: 'gauss'
                Available methods are 'gauss' - convolution with Gaussian kernel or
                'median' - median smoothing of the spectrum
            mode :  string, optional with default: 'nearest'
                Set the mode how to handle the boundarys within the convolution
                Possilbe modes are: reflect, constant, nearest, mirror,  wrap
        """
        if method=='gauss':
            # filter with Gaussian kernel
            self._data = ndimage.filters.gaussian_filter1d(self._data, size, mode=mode)
        elif method=='median':
            # filter with median filter
            self._data = ndimage.filters.median_filter(self._data, size, mode=mode)
        elif method=='BSpline':
            smooth= interpolate.splrep(self._wave,self._data,w=1.0/numpy.sqrt(numpy.fabs(self._data)),s=size)
            self._data =  interpolate.splev(self._wave,smooth,der=0)

    def smoothGaussVariable(self, fwhm):
        fact = numpy.sqrt(2.*numpy.pi)
        if self._mask is not None:
            mask = numpy.logical_not(self._mask)
        else:
            mask = numpy.ones(self._data.shape[0], dtype="bool")

        if isinstance(fwhm, float) or isinstance(fwhm, int):
            fwhm=numpy.ones(self._data.shape[0], dtype=numpy.float32)*fwhm
        select = fwhm>0.0
        mask = numpy.logical_and(mask, select)

        data = numpy.zeros_like(self._data)
        data[:] = self._data
        GaussKernels = old_div(1.0*numpy.exp(-0.5*(old_div((self._wave[mask][:, numpy.newaxis]-self._wave[mask][numpy.newaxis, :]),numpy.abs(fwhm[mask][numpy.newaxis, :]/2.354)))**2),(fact*numpy.abs(fwhm[mask][numpy.newaxis, :]/2.354)))
        data[mask] = old_div(numpy.sum(self._data[mask][:, numpy.newaxis]*GaussKernels, 0),numpy.sum(GaussKernels, 0))

        if self._error is not None:
            error = numpy.zeros_like(self._error)
            error[:] = self._error
            error[mask]=old_div(numpy.sqrt(numpy.sum((self._error[mask]*GaussKernels)**2, 0)),numpy.sum(GaussKernels, 0))
            #scale = Spectrum1D(wave=self._wave, data=error/self._error)
            #scale.smoothSpec(40, method='median')
            #error[mask]=error[mask]/scale._data[mask]

        else:
            error = None

        spec = Spectrum1D(wave = self._wave, data = data,  error = error,  mask=self._mask)
        return spec

    def smoothPoly(self, order=-5, start_wave=None, end_wave=None, ref_base=None):
        if self._mask is not None:
            mask=numpy.logical_not(self._mask)
        else:
            mask=numpy.ones(self._dim, dtype='bool')
        if start_wave is not None:
            mask = numpy.logical_and(mask, self._wave>=start_wave)
        if end_wave is not None:
            mask = numpy.logical_and(mask, self._wave<=end_wave)
        mask = numpy.logical_and(mask,numpy.logical_not(numpy.isnan(self._data)))
        if numpy.sum(mask)>numpy.fabs(order):
            if order>=0:
                fit_poly = numpy.polyfit(self._wave[mask], self._data[mask], order) # fit the spectrum with a normal polynomial
                if ref_base is None:
                    self._data  = numpy.polyval(fit_poly, self._wave)  # replace data with the modelled polynomial
                else:
                    self._data = numpy.polyval(fit_poly, ref_base)
                    self._wave = ref_base
                    self._dim = len(ref_base)
                    self._pixels = numpy.arange(self._dim)
                    if self._mask is not None:
                        mask=numpy.zeros(self._dim, dtype='bool')
                out_par = fit_poly
            elif order<0:
                if ref_base is None:
                    legandre_fit = fit_profile.LegandrePoly(numpy.zeros(-1*order+1), min_x=self._wave[0], max_x=self._wave[-1]) #initialize a legandre polynom
                    legandre_fit.fit(self._wave[mask], self._data[mask]) # fit the data by the polynomial

                    self._data = legandre_fit(self._wave) # replace data with the modelled polynomial
                else:
                    legandre_fit = fit_profile.LegandrePoly(numpy.zeros(-1*order+1), min_x=ref_base[0], max_x=ref_base[-1]) #initialize a legandre polynom
                    legandre_fit.fit(self._wave[mask], self._data[mask]) # fit the data by the polynomial
                    self._data = legandre_fit(ref_base) # replace data with the modelled polynomial
                    self._wave = ref_base
                    self._dim = len(ref_base)
                    self._pixels = numpy.arange(self._dim)
                out_par = legandre_fit._coeff
        else:
            self._data[:] = 0
            if self._mask is not None:
                self._mask[:]=True
            out_par=0
        return out_par

    def findPeaks(self, threshold=100.0, npeaks=0, add_doubles=1e-1, maxiter=400):
        """
            Select local maxima in a Spectrum without taken subpixels into account.

            Parameters
            --------------
            threshold : float, optional with default=100.0
                Threshold above all pixels are assumed to be maxima,
                it is not used if an expected number of peaks is given.

            npeaks : int, optional with default=0
                Number of expected maxima that should be matched.
                If 0 is given the number of maxima is not constrained.

            add_doubles : float, optional with defaul=1e-3

            Returns (pixel, wave, data)
            -----------
            pixels :  numpy.ndarray (int)
                Array of the pixel peak positions
            wave :  numpy.ndarray (float)
                Array of the wavelength peak positions
            data : numpy.ndarray (float)
                Array of the data values at the peak position

        """
        doubles = self._data[1:]==self._data[:-1] #check for identical adjacent values
        doubles=numpy.insert(doubles, 0, False)
        idx = numpy.arange(len(doubles))
        # add some value to one of those adjacent data points
        if numpy.sum(doubles)>0:
            double_idx = idx[doubles]
            self._data[double_idx] += add_doubles
        if self._mask is not None:
            data = self._data[numpy.logical_not(self._mask)]
            wave = self._wave[numpy.logical_not(self._mask)]
            pixels = self._pixels[numpy.logical_not(self._mask)]
        else:
            data = self._data
            wave = self._wave
            pixels = self._pixels
        pos_diff=(data[1:]-data[:-1]) / (wave[1:]-wave[:-1]) # compute the discrete derivative
        # print(">>>>>>>>>>>>>>>>>>>>", (pos_diff<0).sum())
        # print(numpy.array(list(data[1:]-data[:-1]))[(1900<=wave[1:])&(wave[1:]<=2500)].tolist())
        # pylab.figure(figsize=(20,5))
        # pylab.step(wave[1:], data[1:]-data[:-1], lw=1)
        # pylab.xlim(1900,2500)
        # pylab.show()
        select_peaks=numpy.logical_and(pos_diff[1:]<0, pos_diff[:-1]>0)  # select all maxima

        if npeaks==0:
            # if no number of peaks are given select all maxima over a given threshold
            select_thres = data[1:-1][select_peaks]>threshold
        else:
            # if a specific number of peaks are expected iterate until correct number of peaks are found
            matched_peaks=True
            threshold=self.max()[0]/10.0 # set starting threshold
            m=0
            while matched_peaks and m<maxiter:
                select_thres = data[1:-1][select_peaks]>threshold # select all maxima above threshold
                peaks = numpy.sum(select_thres) # check if the number of peaks match expectation
                # if the number of peaks mismatch adjust threshold value to a new value
                if peaks<npeaks:
                    threshold = threshold/2.0
                elif peaks>npeaks:
                    threshold = threshold*1.5
                else:
                    matched_peaks=False # indicate that the correct number of peaks are not  found
                m+=1
        pixels = pixels[1:-1][select_peaks][select_thres]  # select pixel positions of peaks
        wave = wave[1:-1][select_peaks][select_thres] # select wavelength positions of peaks
        data =  data[1:-1][select_peaks][select_thres] # select data valuesof peaks
        return pixels, wave,  data

    def measurePeaks(self, init_pos, method='gauss', init_sigma=1.0, threshold=0, max_diff=0):
        """
            Find the subpixel centre for the local maxima in a Spectrum.

            Parameters
            --------------
            init_pos : numpy.ndarray
                Initial guess for the peak centres

            method : string, optional with default='gauss'
                Select the method to measure the peaks, either 'gauss' or 'hyperbolic'.
                The first one fits a Gaussian to the 3 brightest pixels around each peak,
                the second one uses a hyperbolic approximation to the 3 brightest pixels around each peak.

            init_sigma: float, optional with default=1.0
                Initial guess of the Gaussian width used for the modelling.
                Only used with the method 'gauss'

            threshold: float, optional with default = 0
                It defines the contrast between the minmum of maximum values for the
                3 brightest pixel around a peak for which an estimated centre is assumed to be valid.

            max_diff: float, optional with default = 0
                If greater than zero, all peak centres which are different from the initial guess position by
                this value are assumed to be invalid.

            Returns (positions, mask)
            -----------
            positions :  numpy.ndarray (float)
                Array of subpixel peaks positions
            mask : numpy.ndarray (bool)
                Array of pixels with uncertain measurements
        """
        # compute the minimum and maximum value for the 3 pixels around all peaks
        select=numpy.logical_and(init_pos-1>=[0], init_pos+1<=self._data.shape[0]-1)
        mask= numpy.zeros(len(init_pos), dtype='bool')
        min=numpy.amin([numpy.take(self._data, init_pos[select]+1), numpy.take(self._data, init_pos[select]), numpy.take(self._data, init_pos[select]-1)], axis=0)
        max=numpy.amax([numpy.take(self._data, init_pos[select]+1), numpy.take(self._data, init_pos[select]), numpy.take(self._data, init_pos[select]-1)], axis=0)
        mask[select] = (max-min)<threshold # mask all peaks where the contrast between maximum and minimum is below a threshold
        mask[numpy.logical_not(select)]=True


        if method=='hyperbolic':
            # compute the subpixel peak position using the hyperbolic
            d = numpy.take(self._data, init_pos+1)-2*numpy.take(self._data, init_pos)+numpy.take(self._data,  init_pos-1)
            positions = init_pos+1-(old_div((numpy.take(self._data, init_pos+1)-numpy.take(self._data, init_pos)),d)+0.5)

        elif method=='gauss':
            # compute the subpixel peak position by fitting a gaussian to all peaks (3 pixel to get a unique solution
            positions = numpy.zeros(len(init_pos), dtype='float32') # create empty array
            for j in range(len(init_pos)):
                # only pixels with enough contrast are fitted
                if mask[j]==False:
                    gauss = fit_profile.Gaussian([self._data[init_pos[j]]*numpy.sqrt(2*numpy.pi), init_pos[j], init_sigma]) # set initial parameters for Gaussian profile

                    gauss.fit(self._pixels[init_pos[j]-1:init_pos[j]+2], self._data[init_pos[j]-1:init_pos[j]+2], warning=False) # perform fitting
                    positions[j]=gauss.getPar()[1]

        mask = numpy.logical_or(mask, numpy.isnan(positions)) # masked all corrupt subpixel peak positions

        if max_diff!=0:
            # mask all pixels that are away from the initial guess of peak positions by a certain difference
            mask = numpy.logical_or(numpy.logical_or(positions>init_pos+max_diff, positions<init_pos-max_diff), mask)

        if numpy.sum(mask)>0:
            # replace the estimated position of all masekd peak position by the corresponding initial guess peak positions
            positions[mask]=init_pos[mask].astype('float32')
        return positions, mask

    def measureFWHMPeaks(self, pos, nblocks, init_fwhm=2.4, threshold_flux=None, plot=-1):
        """
            Measures the FWHM of emission lines in the spectrum by a Gaussian modelling assuming that the FWHM is constant for a certain number of fibers (blocks).
            If the spectrum provides an error vector, it is taken into account in the modelling.

            Parameters
            --------------
            pos : numpy.ndarray
                Central peak positions for each peak which are fixed during the modelling
            nblocks : integer
                Number of fibers which are modelled simultaneously with a common FWHM.
                Remaining fibers are included in the last block
            init_fwhm: float, optional with default=2.4
                Initial guess of the Gaussian FWHM used for the modelling.
            threshold_flux: float, optional with default = None
                Mininmum flux included in each peak for an accurate measurement.
                If 20% of the fiber to not full fill this requirement the block FWHM is masked

            Returns (fwhm, mask)
            -----------
            fwhm :  numpy.ndarray (float)
                Array of Gaussian FWHM for each peak
            mask : numpy.ndarray (bool)
                Array of uncertain FWHM measurements
        """
        # create empty fwhm and mask arrays
        fibers = len(pos)
        fwhm= numpy.ones(fibers, dtype=numpy.float32)
        mask = numpy.ones(fibers, dtype='bool')

        # setup the blocks of peaks to be modelled simulatenously
        brackets = numpy.arange(0, fibers, nblocks)
        res = fibers%nblocks
        if res==0:
            brackets=numpy.append(brackets, [fibers+1])
        else:
            brackets[-1]+=res

        #iterate over the blocks
        for i in  range(len(brackets)-1):
            pos_block = pos[brackets[i]:brackets[i+1]] #cut out the corresponding peak positions
            median_dist = numpy.median(pos_block[1:]-pos_block[:-1]) # compute median distance between peaks
            flux = self._data[numpy.round(pos_block).astype('int16')]*numpy.sqrt(2)*init_fwhm/2.354 #initial guess for the flux

            # compute lower and upper bounds of the positions for each block
            lo = int(numpy.min(pos_block)-median_dist)
            if lo <=0:
                lo=0
            hi = int(numpy.max(pos_block)+median_dist)
            if hi>=self._wave[-1]:
                hi=self._wave[-1]

            # modell each block of peaks with Gaussians with and without associate errors
            par = numpy.insert(flux.astype(numpy.float64), 0, init_fwhm/2.354) #set initial paramters
            gaussians_fix_width = fit_profile.Gaussians_width(par, pos_block) # define profile with initial paramter
            if self._error is not None:
                gaussians_fix_width.fit(self._wave[lo:hi], self._data[lo:hi], sigma=self._error[lo:hi], maxfev=1000, xtol=1e-4, ftol=1e-4) # fit with errors
            else:
                gaussians_fix_width.fit(self._wave[lo:hi], self._data[lo:hi], maxfev=1000, xtol=1e-4, ftol=1e-4) # fit without errors
            fit_par = gaussians_fix_width.getPar()
            if plot==i:
                print(i, plot)
                gaussians_fix_width.plot(self._wave[lo:hi], self._data[lo:hi])

            fwhm[brackets[i]:brackets[i+1]] = fit_par[0]*2.354 # convert Gaussian sigma to FWHM

            # create the bad pixel mask
            if threshold_flux is not None:
                masked = numpy.logical_or(numpy.sum(fit_par[1:]<0)>0, numpy.sum(fit_par[1:]>threshold_flux)<0.2*len(fit_par[1:]))
                mask[brackets[i]:brackets[i+1]]  = masked
            else:
                mask[brackets[i]:brackets[i+1]]  = numpy.sum(fit_par[1:]<0)>0

        #return results
        return fwhm, mask

    def measureOffsetPeaks(self, pos, mask,  nblocks, init_fwhm=2.0, init_offset=0.0,  plot=-1):
        """

        """
        # create empty fwhm and mask arrays
        fibers = len(pos)
        if mask is None:
            good = numpy.ones(fibers, dtype='bool') & (pos<=(self._data.shape[0]-1))
        else:
            good = numpy.logical_not(mask) & (pos<=(self._data.shape[0]-1))

        # setup the blocks of peaks to be modelled simulatenously
        blocks = numpy.array_split(numpy.arange(0, fibers), nblocks)

        offsets = numpy.zeros(len(blocks), dtype=numpy.float32)
        med_pos = numpy.zeros_like(offsets)

        #iterate over the blocks
        for i in  range(len(blocks)):
            pos_block = pos[blocks[i]] #cut out the corresponding peak positions
            pos_mask = good[blocks[i]]
            if numpy.sum(pos_mask)>0:
                median_dist = numpy.median(pos_block[pos_mask][1:]-pos_block[pos_mask][:-1]) # compute median distance between peaks
                flux = self._data[numpy.round(pos_block[pos_mask]).astype('int16')]*numpy.sqrt(2)*init_fwhm/2.354 #initial guess for the flux

                # compute lower and upper bounds of the positions for each block
                lo = int(pos_block[pos_mask][0]-median_dist)
                if lo <=0:
                    lo=0
                hi = int(pos_block[pos_mask][-1]+median_dist)
                if hi>=self._wave[-1]:
                    hi=self._wave[-1]

                # modell each block of peaks with Gaussians with and without associate errors
                par = numpy.insert(flux.astype(numpy.float64), 0, init_fwhm/2.354) #set initial paramters
                par = numpy.append(par, init_offset) #set initial paramters
                gaussians_offset = fit_profile.Gaussians_offset(par, pos_block[pos_mask]) # define profile with initial paramters
                if self._error is not None:
                    gaussians_offset.fit(self._wave[lo:hi], self._data[lo:hi], sigma=self._error[lo:hi], maxfev=4000, xtol=1e-8, ftol=1e-8) # fit with errors
                else:
                    gaussians_offset.fit(self._wave[lo:hi], self._data[lo:hi], maxfev=4000, xtol=1e-8, ftol=1e-8) # fit without errors
                fit_par = gaussians_offset.getPar()
                if plot==i:
                    gaussians_offset.plot(self._wave[lo:hi], self._data[lo:hi])

                offsets[i] = fit_par[-1] # get offset position
                med_pos[i] = numpy.mean(self._wave[lo:hi])
            else:
                offsets[i]=0.0
                med_pos[i]=0.0

        return offsets, med_pos

    def measureOffsetPeaks2(self, pos, mask, fwhm, nblocks, min_offset, max_offset, step_offset,plot=0):
        fibers = len(pos)
        if mask is None:
            good = numpy.ones(fibers, dtype='bool') & (pos<=(self._data.shape[0]-1))
        else:
            good = numpy.logical_not(mask) & (pos<=(self._data.shape[0]-1))

        # setup the blocks of peaks to be modelled simulatenously
        blocks = numpy.array_split(numpy.arange(0, fibers), nblocks)

        offsets = numpy.zeros(len(blocks), dtype=numpy.float32)
        med_pos = numpy.zeros_like(offsets)

        #iterate over the blocks
        for i in  range(len(blocks)):
            pos_block = pos[blocks[i]] #cut out the corresponding peak positions
            pos_mask = good[blocks[i]]
            pos_fwhm = fwhm[blocks[i]]
            if numpy.sum(pos_mask)>0:

                median_dist = numpy.median(pos_block[pos_mask][1:]-pos_block[pos_mask][:-1]) # compute median distance between peaks

                # compute lower and upper bounds of the positions for each block
                lo = int(pos_block[pos_mask][0]-median_dist)
                if lo <=0:
                    lo=0
                hi = int(pos_block[pos_mask][-1]+median_dist)
                if hi>=self._wave[-1]:
                    hi=self._wave[-1]
                Gaussian_vec = numpy.zeros((numpy.sum(pos_mask),hi-lo),dtype=numpy.float32)
                x = numpy.arange(hi-lo)+lo
                offset = numpy.arange(min_offset,max_offset,step_offset)
                chisq = numpy.zeros(len(offset))
                max_flux = numpy.zeros(len(offset))
                for o in range(len(offset)):
                    for g in range(numpy.sum(pos_mask)):
                        Gaussian_vec[g,:] = old_div(numpy.exp(-0.5*(old_div((x-(pos_block[pos_mask][g]+offset[o])),
                        (pos_fwhm[pos_mask][g]/2.354)))**2),(numpy.sqrt(2.*numpy.pi)*abs((pos_fwhm[pos_mask][g]/2.354))))
                    result = numpy.linalg.lstsq(Gaussian_vec.T,self._data[lo:hi])
                    chisq[o] = result[1][0]
                    max_flux[o] = numpy.sum(result[0])
                find_max=numpy.argsort(max_flux)[-1]
                offsets[i] = offset[find_max] # get offset position
                med_pos[i] = numpy.mean(self._wave[lo:hi])
            else:
               offsets[i]=0.0
               med_pos[i]=0.0
        return offsets, med_pos

    def fitMultiGauss(self, centres, init_fwhm):

        select = numpy.zeros(self._dim, dtype='bool')
        flux_in = numpy.zeros(len(centres), dtype=numpy.float32)
        sig_in = numpy.ones_like(flux_in)*init_fwhm/2.354
        cent =numpy.zeros(len(centres), dtype=numpy.float32)
        if self._error is not None:
            error = self._error
        else:
            error = numpy.ones_like(self._dim, dtype=numpy.float32)
        for i in range(len(centres)):
            select_line=numpy.logical_and(self._wave>centres[i]-2*init_fwhm, self._wave<centres[i]+2*init_fwhm)
            flux_in[i] = numpy.sum(self._data[select_line])
            select=numpy.logical_or(select, select_line)
            cent[i] = centres[i]
        par = numpy.concatenate([flux_in, cent, sig_in])
        gauss_multi = fit_profile.Gaussians(par)
        gauss_multi.fit(self._wave[select], self._data[select], sigma=error[select])
        return gauss_multi.getPar()

    def fitParFile(self, par,  err_sim=0, ftol=1e-8, xtol=1e-8,method='leastsq',parallel='auto'):
        static_par = deepcopy(par)

        if self._error is not None:
            sigma=self._error
        else:
            sigma=1.0
        par.fit(self._wave, self._data, sigma=sigma, err_sim=err_sim, maxfev=1000,  method=method,ftol=ftol, xtol=xtol, parallel=parallel)
        par.restoreResult()
        if err_sim>0 and self._error is not None:
            par_err = deepcopy(static_par)
            par_err._par = par._par_err
            par_err.restoreResult()
            par._parameters_err=par_err._parameters

    def fitSepGauss(self, centres, aperture, init_back=0.0, ftol=1e-8, xtol=1e-8, plot=False, warning=False):

        ncomp = len(centres)
        cent =numpy.zeros(ncomp,  dtype=numpy.float32)
        out = numpy.zeros(3*ncomp, dtype=numpy.float32)
        back = numpy.zeros(ncomp, dtype=numpy.float32)
        if self._error is not None:
            error = self._error
        else:
            error = numpy.ones(self._dim, dtype=numpy.float32)
        if self._mask is not None:
            mask = self._mask
        else:
            mask = numpy.zero(self._dim, dtype="bool")
        for i in range(len(centres)):
            back[i] = deepcopy(init_back)
            select=numpy.logical_and(numpy.logical_and(self._wave>=centres[i]-aperture/2.0, self._wave<=centres[i]+aperture/2.0), numpy.logical_not(mask))
            if numpy.sum(select)>0:
                max = numpy.max(self._data[select])
                cent = numpy.median(self._wave[select][self._data[select]==max])
                select=numpy.logical_and(self._wave>=cent-aperture/2.0, self._wave<=cent+aperture/2.0, numpy.logical_not(mask))
                if back[i]==0.0:
                    par = [0.0, 0.0, 0.0]
                    gauss = fit_profile.Gaussian(par)
                    gauss.fit(self._wave[select], self._data[select], sigma=error[select], ftol=ftol, xtol=xtol, warning=warning)
                else:
                    par = [0.0, 0.0, 0.0, 0.0]
                    gauss = fit_profile.Gaussian_const(par)
                    gauss.fit(self._wave[select], self._data[select],  sigma=error[select], ftol=ftol, xtol=xtol, warning=warning)
                out_fit = gauss.getPar()
                out[i] = out_fit[0]
                out[ncomp+i] = out_fit[1]
                out[2*ncomp+i] = out_fit[2]
                if plot==True:
                    gauss.plot(self._wave[select], self._data[select])

            else:
                out[i] = 0.0
                out[ncomp+i] = 0.0
                out[2*ncomp+i] = 0.0
        if plot==True:
                    pylab.show()
        return out

    def obtainGaussFluxPeaks(self, pos, sigma, indices,  replace_error=1e10, plot=False):
        fibers = len(pos)
        aperture=3
        pixels = numpy.round(pos[:, numpy.newaxis]+numpy.arange(-aperture/2.0, aperture/2.0, 1.0)[numpy.newaxis, :]).astype('int')
        if self._mask is not None:
            bad_pix = numpy.zeros(fibers,  dtype='bool')
            select=numpy.sum(pixels>=self._mask.shape[0], 1)
            nselect = numpy.logical_not(select)
            bad_pix[select]=True
            bad_pix[nselect] = numpy.sum(self._mask[pixels[nselect, :]], 1)==aperture
        else:
            bad_pix=None
        if self._error is None:
            self._error=numpy.ones_like(self._data)

        fact = numpy.sqrt(2.*numpy.pi)
        A=old_div(1.0*numpy.exp(-0.5*(old_div((self._wave[:, numpy.newaxis]-pos[numpy.newaxis, :]),sigma[numpy.newaxis, :]))**2),(fact*sigma[numpy.newaxis, :]))
        select = A>0.0001
        A=old_div(A,self._error[:, numpy.newaxis])

        B = sparse.csr_matrix( (A[select],(indices[0][select],indices[1][select])), shape=(self._dim,fibers) ).todense()
        out= sparse.linalg.lsqr(B, old_div(self._data,self._error), atol=1e-7, btol=1e-7, conlim=1e13)
        error = numpy.sqrt(old_div(1,numpy.sum((A**2), 0)))
        if bad_pix is not None and numpy.sum(bad_pix)>0:
            error[bad_pix]=replace_error
        if plot==True:
            pylab.plot(self._data, 'ok')
            pylab.plot(numpy.dot(A*self._error[:, numpy.newaxis], out[0]), '-r')
            #pylab.plot(numpy.dot(A, out[0]), '-r')
            pylab.show()
        return out[0], error, bad_pix

    def collapseSpec(self, method='mean', start=None, end=None, transmission_func=None):
        if start is not None:
            select_start = self._wave>=start
        else:
            select_start = numpy.ones(self._dim, dtype='bool')
        if end is not None:
            select_end = self._wave<=end
        else:
            select_end = numpy.ones(self._dim, dtype='bool')
        select = numpy.logical_and(select_start, select_end)
        if self._mask is not None:
            select = numpy.logical_and(select, numpy.logical_not(self._mask))
        if method!='mean' and method!='median' and method!='sum':
            pass
        elif method=='mean':
            flux = numpy.mean(self._data[select])
            if self._error is not None:
                error = numpy.sqrt(old_div(numpy.sum(self._error[select]**2),numpy.sum(select)**2))
            else:
                error = 0
        return flux, error

