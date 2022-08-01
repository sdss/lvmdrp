from lvmdrp.core.header import Header
from lvmdrp.core.positionTable import PositionTable
from lvmdrp.core.image import Image
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.passband import PassBand
from astropy.io import fits as pyfits
import numpy
from scipy import ndimage

class Cube(Header, PositionTable):

    def __getitem__(self, slice):
        if not isinstance(slice, int):
            raise TypeError('Slice index need to be an integer')

        if slice>=self._res_elements or slice<self._res_elements*-1:
            raise IndexError('The Cube contains only %i resolution elments for which the index %i is invalid'%(self._res_elements, slice))

        if self._data is not None:
            data = self._data[slice, :, :]
        else:
            data=None

        if self._error is not None:
            error = self._error[slice, :, :]
        else:
            error=None
        if self._mask is not None:
            mask = self._mask[slice, :, :]
        else:
            mask = None
        return Image(data=data, error=error, mask=mask)

    def __mul__(self, other):

        if isinstance(other, Spectrum1D):
            other._data.astype(numpy.float32)
            data = self._data*other._data.astype(numpy.float32)[:, numpy.newaxis, numpy.newaxis]

            if self._mask is not None and other._mask is not None:
                mask_full = numpy.zeros(data.shape, dtype='bool')
                mask_full[other._mask]=1
                mask = numpy.logical_or(self._mask, mask_full)
            elif self._mask is None and other._mask is not None:
                mask = numpy.zeros(data.shape, dtype='bool')
                mask[other._mask]=1
            elif self._mask is not None and other._mask is None:
                mask= self._mask
            else:
                mask =None
            if self._error is not None and other._error is not None:
                error=numpy.sqrt((self._error*other._data[:, numpy.newaxis, numpy.newaxis]+self._data*other._error[:, numpy.newaxis, numpy.newaxis])**2)
            elif self._error is not None and other._error is None:
                error = self._error*other._data[:, numpy.newaxis, numpy.newaxis]
            else:
                error = None
            if data.dtype==numpy.float64:
                data.astype(numpy.float32)
            if error is not None and error.dtype==numpy.float64:
                error.astype(numpy.float32)

            cube= Cube(wave=self._wave, data = data,  error = error,  error_weight=self._error_weight, mask=mask, header=self.getHeader(), cover=self._cover)
            return cube


    def __init__(self, data=None, wave=None, error = None, mask = None,  error_weight=None, header = None, cover = None):
        Header.__init__(self, header=header)
        if data is None:
            self._data = None
        else:
            self._data = data
            self._res_elements = data.shape[0]
            self._dim_y = data.shape[1]
            self._dim_x = data.shape[2]
        if wave is None:
            self._wave = None
        else:
            self._wave = numpy.array(wave)
        if error is None:
            self._error = None
        else:
            self._error = numpy.array(error)

        if error_weight is None:
            self._error_weight=None
        else:
            self._error_weight=numpy.array(error_weight)

        if mask is None:
            self._mask = None
        else:
            self._mask = numpy.array(mask)

        if cover is None:
            self._cover = None
        else:
            self._cover = numpy.array(cover)


    def loadFitsData(self, file, extension_data=None, extension_mask=None,  extension_error=None,  extension_errorweight=None, extension_hdr=0, extension_cover=None):
        """
            Load data from a FITS image into a Cube object

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
        hdu = pyfits.open(file,uint=True,do_not_scale_image_data=True)
        if extension_data is None and extension_mask is None and extension_error is None and extension_errorweight is None and extension_cover is None:
                self._data = hdu[0].data
                self._res_elements = self._data.shape[0]
                self._dim_y = self._data.shape[1]
                self._dim_x = self._data.shape[2]
                self._wave = None
                self.setHeader(header = hdu[extension_hdr].header, origin=file)
                if len(hdu)>1:
                    for i in range(1, len(hdu)):
                        if hdu[i].header['EXTNAME'].split()[0]=='ERROR':
                            self._error = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0]=='BADPIX':
                            self._mask = hdu[i].data.astype('bool')
                        elif hdu[i].header['EXTNAME'].split()[0]=='ERRWEIGHT':
                            self._error_weight = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0]=='FIBCOVER':
                            self._cover = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0]=='WAVE':
                            self._wave = hdu[i].data
                if self._wave is None:
                    try:
                        crpix = self.getHdrValue('CRPIX3')-1
                    except:
                        crpix = 0
                    try:
                        self._wave = (numpy.arange(self._res_elements)-crpix)*self.getHdrValue('CDELT3')+self.getHdrValue('CRVAL3')
                    except KeyError:
                        self._wave = (numpy.arange(self._res_elements)-crpix)*self.getHdrValue('CD3_3')+self.getHdrValue('CRVAL3')
        else:
            self.setHeader(header = hdu[extension_hdr].header, origin=file)	    
            if extension_data is not None:
                self._data = hdu[extension_data].data
                self._res_elements = self._data.shape[0]
                self._dim_y = self._data.shape[1]
                self._dim_x = self._data.shape[2]

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data
                self._res_elements = self._mask.shape[0]
                self._dim_y = self._mask.shape[1]
                self._dim_x = self._mask.shape[2]
            if extension_error is not None:
                self._error = hdu[extension_error].data
                self._res_elements = self._error.shape[0]
                self._dim_y = self._error.shape[1]
                self._dim_x = self._error.shape[2]
            if extension_errorweight is not None:
                self._error_weight= hdu[extension_errorweight].data
                self._res_elements = self._error_weight.shape[0]
                self._dim_y = self._error_weight.shape[1]
                self._dim_x = self._error_weight.shape[2]
            if extension_cover is not None:
                self._cover= hdu[extension_cover].data
                self._res_elements = self._cover.shape[0]
                self._dim_y = self._cover.shape[1]
                self._dim_x = self._cover.shape[2]
            try:
                crpix = self.getHdrValue('CRPIX3')-1
            except:
                crpix = 0
            try:
                self._wave = (numpy.arange(self._res_elements)-crpix)*self.getHdrValue('CDELT3')+self.getHdrValue('CRVAL3')
            except KeyError:
                self._wave = (numpy.arange(self._res_elements)-crpix)*self.getHdrValue('CD3_3')+self.getHdrValue('CRVAL3')
        hdu.close()
        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header, origin=file)


    def convolveCube(self, kernel, mode='nearest'):
        """
            Convolves the data of the Cube with a given kernel.

            Parameters
            --------------
            kernel : ndarray
                Convolution kernel

            -----------
            new_cube :  Cube object
                Convolved cube
        """

        # convolve the data array with the given convolution kernel
        new = ndimage.filters.convolve(self._data,kernel,mode=mode)
        if self._error is not None:
            new_error = numpy.sqrt(ndimage.filters.convolve(self._error**2, kernel, mode=mode))
        else:
            new_error = None
        # create new Image object with the error and the mask unchanged and return
        new_cube = Cube(data=new, error=new_error,  mask=self._mask, header = self._header, cover = self._cover)
        return new_cube


    def writeFitsData(self, filename, extension_data=None, extension_mask=None,  extension_error=None,  extension_errorweight=None, extension_cover=None, fix_header=False,store_wave=False):
        """
            Save information from a Cube object into a FITS file.
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
        hdus=[None, None, None, None, None,None] # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if extension_data is None and extension_error is None and extension_mask is None and extension_errorweight is None and extension_cover is None:
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name='ERROR')
            if self._error_weight is not None:
                hdus[2] = pyfits.ImageHDU(self._error_weight,  name='ERRWEIGHT')
            if self._mask is not None:
                hdus[3] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')
            if self._cover is not None:
                hdus[4] = pyfits.ImageHDU(self._cover.astype('uint8'), name='FIBCOVER')
            if self._wave is not None and store_wave:
                hdus[5] = pyfits.ImageHDU(self._wave, name='WAVE')
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data>0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name='DATA')

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

            if extension_errorweight == 0:
                hdu = pyfits.PrimaryHDU(self._error_weight)
            elif extension_errorweight>0 and extension_errorweight is not None:
                hdus[extension_errorweight] = pyfits.ImageHDU(self._error_weight, name='ERRWEIGHT')

            if extension_cover == 0:
                hdu = pyfits.PrimaryHDU(self._cover)
            elif extension_cover>0 and extension_cover is not None:
                hdus[extension_cover] = pyfits.ImageHDU(self._cover, name='FIBCOVER')

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except:
                break


        if len(hdus)>0:
            hdu = pyfits.HDUList(hdus) # create an HDUList object
            if self._header is not None:
                if self._wave is not None and len(self._wave.shape)==1:
                    self.setHdrValue('CRVAL3', self._wave[0])
                    self.setHdrValue('CDELT3', (self._wave[1]-self._wave[0]))
                hdu[0].header = self.getHeader() # add the primary header to the HDU
                hdu[0].update_header()
            else:
                if self._wave is not None and len(self._wave.shape)==1:
                    hdu[0].header['CRVAL3'] = (self._wave[0])
                    hdu[0].header['CDELT3'] = (self._wave[1]-self._wave[0])
        hdu.writeto(filename, overwrite=True, output_verify='silentfix') # write FITS file to disc


    def collapseCube(self, mode='mean', start_wave=None, end_wave=None,computeError=False):
        error_collapsed=None
        select_wave = numpy.ones(self._res_elements, dtype="bool")
        if start_wave is not None:
            select_wave = numpy.logical_and(select_wave, self._wave>=start_wave)
        if end_wave is not None:
            select_wave = numpy.logical_and(select_wave, self._wave<=end_wave)

        if self._mask is not None:
            data= self._data* numpy.logical_not(self._mask)
            if self._error is not None:
                error = self._error * numpy.logical_not(self._mask)
        else:
            data = self._data

        if mode=='mean':
            if numpy.sum(select_wave)>0:
                image = numpy.mean(data[select_wave, :, :], 0)
                mask = image==0
                if computeError and self._error is not None:
                    error_collapsed = numpy.sqrt(numpy.sum(self._error[select_wave,:,:]**2,0))/numpy.sum(select_wave)
                    
            else:
                image = None
                mask=None
        elif mode=='median':
            if numpy.sum(select_wave)>0:
                image = numpy.median(data[select_wave, :, :], 0)
                mask = image==0
            else:
                image = None
                mask=None
        elif mode=='sum':
            if numpy.sum(select_wave)>0:
                image = numpy.sum(data[select_wave, :, :], 0)
                mask = image==0
                if computeError and self._error is not None:
                    error_collapsed = numpy.sqrt(numpy.sum(self._error[select_wave,:,:]**2,0))
            else:
                image = None
                mask = None
        else:
            passband = PassBand()
            filter=mode.split(',')
            passband.loadTxtFile(filter[0], wave_col=int(filter[1]),  trans_col=int(filter[2]))
            image=passband.getFluxCube(self)
            mask= image==0
            img = Image(data=image, error=error_collapsed, mask=mask)
        
        return img

    def medianFilter(self, size):
        filter_cube = ndimage.median_filter(self._data, (size, 1, 1))
        cube = Cube(data=filter_cube, header = self._header, cover = self._cover)
        return cube

    def rot90(self,n):
        data_out = numpy.rot90(self._data.T,n).T
        if self._error is not None:
            error_out = numpy.rot90(self._error.T,n).T
        else:
            error_out = None
        if self._error_weight is not None:
            error_weight_out = numpy.rot90(self._error_weight.T,n).T
        else:
            error_weight_out = None

        if self._mask is not None:
            mask_out = numpy.rot90(self._mask.T,n).T
        else:
            mask_out = None

        if self._cover is not None:
            cover_out = numpy.rot90(self._cover.T,n).T
        else:
            cover_out = None

        cube_out = Cube(wave=self._wave, data=data_out, error=error_out, error_weight=error_weight_out, mask=mask_out, cover=cover_out, header=self.getHeader())
        return cube_out

    def getAperSpec(self, cent_x, cent_y, aperture, kmax=1000, correct_masked=False, ignore_mask=True,  threshold_coverage=0.0):

        spec_data = numpy.zeros(self._res_elements)
        if self._error is not None:
            spec_error = numpy.zeros(self._res_elements)
        coverage = numpy.zeros(self._res_elements)
        for i in range(self._res_elements):
            slice = self[i]
            result = slice.extractAper(cent_x, cent_y, aperture, kmax, correct_masked=correct_masked, ignore_mask=ignore_mask)
            if self._error is not None:
                spec_error[i]=result[1]
            spec_data[i] = result[0]
            coverage[i] = result[5]
        if threshold_coverage>0.0:
            masked = coverage<threshold_coverage
        else:
            masked = None

        if self._error is None:
            spec_error=None

        spec = Spectrum1D(wave=self._wave, data=spec_data, mask=masked,  error = spec_error)
        return spec

    def glueCubeSets(self, cube1, cube2, rescale_region=[], merge_wave=0.0, mergeHdr=True):
        wave1 = cube1._wave
        wave2 = cube2._wave
        disp1 = wave1[1]-wave1[0]
        disp2 = wave2[1]-wave2[0]
        if merge_wave==0.0:
            if (disp1) == (disp2) and (disp1*numpy.rint(((wave2[0]-wave1[0]))/(disp1))).astype(numpy.float32)==(wave2[0]-wave1[0]).astype(numpy.float32):
                wave = numpy.arange(wave1[0],wave2[-1]+disp2,disp2)
                select1 = wave1<wave2[0]
                select2 = wave2>wave1[-1]
                select_overlap1 = wave1>=wave2[0]
                select_overlap2 = wave2<=wave1[-1]
            else:
                raise ValueError("The wavelength ranges do not match with each other")
        else:
            select1 = wave1<=merge_wave
            select2 = wave2>merge_wave
            wave = numpy.concatenate((wave1[select1], wave2[select2]))
        crpix1_1 = cube1.getHdrValue('CRPIX1')
        crpix2_1 = cube1.getHdrValue('CRPIX2')
        crpix1_2 = cube2.getHdrValue('CRPIX1')
        crpix2_2 = cube2.getHdrValue('CRPIX2')
        naxis1_1 = cube1.getHdrValue('NAXIS1')
        naxis2_1 = cube1.getHdrValue('NAXIS2')
        naxis1_2 = cube2.getHdrValue('NAXIS1')
        naxis2_2 = cube2.getHdrValue('NAXIS2')
        xminus = numpy.min([crpix1_1, crpix1_2])
        crpix1 = xminus
        xplus = numpy.min([naxis1_1-crpix1_1, naxis1_2-crpix1_2])
        yminus = numpy.min([crpix2_1, crpix2_2])
        crpix2 = yminus
        yplus = numpy.min([naxis2_1-crpix2_1, naxis2_2-crpix2_2])

        self._data = numpy.zeros((len(wave), yminus+yplus, xminus+xplus), dtype=numpy.float32)
        if cube1._error is not None and cube2._error is not None:
            self._error = numpy.zeros((len(wave), yminus+yplus, xminus+xplus), dtype=numpy.float32)
        else:
            self._error = None

        if cube1._mask is not None and cube2._mask is not None:
            self._mask = numpy.zeros((len(wave), yminus+yplus, xminus+xplus), dtype='bool')
        else:
            self._mask = None

        if cube1._error_weight is not None and cube2._error_weight is not None:
            self._error_weight = numpy.zeros((len(wave), yminus+yplus, xminus+xplus), dtype=numpy.float32)
        else:
            self._error_weight = None

        if cube1._cover is not None and cube2._cover is not None:
            self._cover = numpy.zeros((len(wave), yminus+yplus, xminus+xplus), dtype=numpy.float32)
        else:
            self._cover = None

        if rescale_region!=[]:
            start_wave = rescale_region[0]
            end_wave = rescale_region[1]
            slice1 = cube1.collapseCube(mode='median', start_wave=start_wave, end_wave=end_wave)
            slice2 = cube2.collapseCube(mode='median', start_wave=start_wave, end_wave=end_wave)

        for i in range(self._data.shape[2]):
            for j in range(self._data.shape[1]):
                if rescale_region is not [] and slice1._data[j+crpix2_1-crpix2, i+crpix1_1-crpix1]!=0:
                    ratio = slice2._data[j+crpix2_2-crpix2, i+crpix1_2-crpix1]/slice1._data[j+crpix2_1-crpix2, i+crpix1_1-crpix1]
                else:
                    ratio=1
                if ratio<0:
                    ratio=0
                if merge_wave!=0.0:
                    self._data[:, j, i] = numpy.concatenate((cube1._data[select1, j+crpix2_1-crpix2, i+crpix1_1-crpix1]*ratio, cube2._data[select2, j+crpix2_2-crpix2, i+crpix1_2-crpix1]))
                if self._error is not None:
                    self._error[:, j, i] = numpy.concatenate((cube1._error[select1, j+crpix2_1-crpix2, i+crpix1_1-crpix1]*ratio, cube2._error[select2, j+crpix2_2-crpix2, i+crpix1_2-crpix1]))
                if self._error_weight is not None:
                    self._error_weight[:, j, i] = numpy.concatenate((cube1._error_weight[select1, j+crpix2_1-crpix2, i+crpix1_1-crpix1], cube2._error_weight[select2, j+crpix2_2-crpix2, i+crpix1_2-crpix1]))
                if self._mask is not None:
                    if ratio>0:
                        self._mask[:, j, i] = numpy.concatenate((cube1._mask[select1, j+crpix2_1-crpix2, i+crpix1_1-crpix1], cube2._mask[select2, j+crpix2_2-crpix2, i+crpix1_2-crpix1]))
                    else:
                        self._mask[:, j, i] = True

                if self._cover is not None:
                    self._cover[:, j, i] = numpy.concatenate((cube1._cover[select1, j+crpix2_1-crpix2, i+crpix1_1-crpix1], cube2._cover[select2, j+crpix2_2-crpix2, i+crpix1_2-crpix1]))

        if mergeHdr==True:
            hdrs=[cube1, cube2]
            combined_header = combineHdr(hdrs)
            self.setHeader(combined_header.getHeader())

        if merge_wave!=0.0:
            self.setHdrValue('CRPIX1', crpix1)
            self.setHdrValue('CRPIX2', crpix2)
            self.setHdrValue('CRVAL3', wave[0])
            self.setHdrValue('NAXIS3', len(wave))
        else:
            self.setHdrValue('CRPIX3', 1.0)
            self.setHdrValue('CRVAL3', wave[0])
            self.setHdrValue('CDELT3', disp1)
            self.setHdrValue('NAXIS3', len(wave))
       
        
      
def loadCube(infile, extension_data=None, extension_mask=None,  extension_error=None, extension_cover=None, extension_hdr=0):

    cube = Cube()
    cube.loadFitsData(infile, extension_data=extension_data, extension_mask=extension_mask,  extension_error=extension_error, extension_cover=extension_cover, extension_hdr=extension_hdr)

    return cube



