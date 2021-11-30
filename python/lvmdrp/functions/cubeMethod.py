from __future__ import print_function
from __future__ import division
from builtins import range
from copy import deepcopy

from lvmdrp.core import fit_profile
from lvmdrp.core.image import Image

from past.utils import old_div
from scipy import stats
import sys, numpy
try:
  import pylab
  from matplotlib import pyplot as plt
except:
  pass
from lvmdrp import *
from lvmdrp.external import ancillary_func
from lvmdrp.core.cube import Cube, loadCube
from lvmdrp.core.passband import PassBand
from lvmdrp.core.rss import RSS
from lvmdrp.core.spectrum1d import Spectrum1D

description='Provides Methods to process Cube files'

def collapseSliceCube_drp(cube_in, image_out, mode='mean', start_wave='', end_wave=''):
    """
           Creates a collapsed 2D image from the data cube within given wavelength boundaries

            Parameters
            --------------
            cube_in : string
                    Name of the INPUT FITS  cube from which a collapsed image is created
            image_out : string
                    Name of the  OUTPUT FITS image of the collapsed 2D image
            mode: string, optional with default='mode'
                    Mathematical operation applied to obatin the value along the dispersion axis for each spaxel in the cube.
                    Available methods are
                    1. mean
                    2. median
            start_wave : string of float, optional with default: ''
                    Start wavelength for the collapsing. If no value is given the start wavelength of the cube is used.
            end_wave : string of float, optional with default: ''
                    End wavelength for the collapsing. If no value is given the end wavelength of the cube is used.

            Examples
            ----------------
            user:> lvmdrp cube collapsedSliceCube CUBE.fits IMAGE.fits mode=median
            user:> lvmdrp cube collapsedSliceCube CUBE.fits IMAGE.fits mean start_wave=4000.0 end_wave=7000.0
    """
    if start_wave=='':
        start_wave=None
    else:
        start_wave = float(start_wave)

    if end_wave=='':
        end_wave=None
    else:
        end_wave = float(end_wave)

    cube=loadCube(cube_in)
    image=cube.collapseCube(mode=mode, start_wave=start_wave, end_wave=end_wave)
    image.writeFitsData(image_out)


def measureDARPeak_drp(cube_in, out_prefix, coadd='10', steps='50', search_box='16', bary_box='6', bary_exp='4', smooth_poly='-3', start_wave='', end_wave='', clip='0.3', fibers='', figure_out='', verbose='0'):
    """
           The target position is measured by means of its bary center a long the wavelength to measure the effect of the DAR.
           The measured positions are smooth by a polynomial fit and clip to reject outliners due to night sky or object emission lines before a final
           polynomial fit is performed. The fitted positions are along the entire wavelength of the cube are stored in two FITS files in both, x and y, direction
           seperatly.

            Parameters
            --------------
            cube_in : string
                    Name of the INPUT FITS  cube for which the DAR will be estimated
            out_prefix : string
                    Prefix for the 2 output files PREFIX.cont_x.fits and PREFIX.cont_y.fits that store the object position as a function of wavelength
            coadd: string of integer (>0), optional with default: '10'
                    Coadd number of pixels in dispersion direction to increase the S/N of the data
            steps : string of integer (>0), optional with default :'50'
                    Steps in dispersions direction for which the object position is estimated (saves times)
            search_box : string of integer (>0), optional with default: '16'
                    Quadratic box with the given pixel size centered on the center of the spatial grid used to find the maximum flux pixel
                    as the reference object center
            bary_box : string of integer (>0), optional with default: '6'
                    Quadratic box centered on the maximum object counts found in the search box used to compute the bary center
            bary_exp : string of integer (>0), optional with default: '4'
                    The standard bary center formulea is modified so that the flux is not weighted linearly, but rather with a variable exponent
            smooth_poly: string of integer, optional with default: '-3'
                    Order of the polynomial used to fit and smooth the measured peak positions in x and y coordinates along dispersion axis
                    (positiv: normal polynomial, negative: Legandre polynomial)
            start_wave : string of float, optional with default: ''
                    Start wavelength used for the polynomial fitting. If no value is given the start wavelength of the cube is used.
            end_wave : string of float, optional with default: ''
                    End wavelength used for the polynomial fitting. If no value is given the end wavelength of the cube is used.
            clip : string of float, optional with default: '0.3':
                    The measured values which deviates more than +- the given clip value from the polynomial fit are clipped for a second polynomial fit
            verbose : string of integer (0 or 1), optional  with default: 0
                    Show information during the processing on the command line (0 - no, 1 - yes)

            Examples
            ----------------
            user:> lvmdrp cube measureDARPeak CUBE.fits PREFIX coadd=5 steps=1
    """
    coadd = int(coadd)
    steps=int(steps)
    smooth_poly = int(smooth_poly)
    verbose= int(verbose)
    search_box=int(search_box)
    bary_box=int(bary_box)
    bary_exp=float(bary_exp)
    clip = float(clip)

    if start_wave=='':
        start_wave=None
    else:
        start_wave=float(start_wave)
    if end_wave=='':
        end_wave=None
    else:
        end_wave=float(end_wave)

    if fibers=='':
      fibers=None
    else:
      fibers = int(fibers)

    cube = Cube()
    cube.loadFitsData(cube_in)
    #kernel = numpy.ones((coadd,1,1),dtype='uint8')
    coadd_cube  = cube.medianFilter(coadd)


    select_slice = numpy.arange(cube._res_elements)%steps==0
    select_slice[0] = True
    select_slice[-1] = True
    slices = numpy.arange(cube._res_elements)[select_slice]

    wave = cube._wave[select_slice]
    cent_x = numpy.zeros(len(slices))
    cent_y = numpy.zeros(len(slices))

    ref = int(numpy.rint(cube._res_elements/2.0))
    collapsed_img = cube.collapseCube('median', start_wave, end_wave)
    cent_guess = collapsed_img.centreMax(cent_x=collapsed_img._dim[1]/2.0, cent_y=collapsed_img._dim[0]/2.0, box_size=search_box)

    m=0
    plot=False
    for i in slices:
        if verbose==1:
            print(i)
        image = cube[i]
        cent = image.centreBary(cent_guess[0]-1, cent_guess[1]-1, box_size=bary_box,exponent=bary_exp)
        cent_x[m]=cent[0]
        cent_y[m]=cent[1]
        m+=1


    spec_x = Spectrum1D(data=cent_x, wave=wave)
    poly_x = spec_x.smoothPoly(order=smooth_poly, start_wave=start_wave, end_wave=end_wave)
    spec_y = Spectrum1D(data=cent_y, wave=wave)
    poly_y = spec_y.smoothPoly(order=smooth_poly, start_wave=start_wave, end_wave=end_wave)

    if start_wave!=None and end_wave!=None:
        select = numpy.logical_and(wave>=start_wave, wave<=end_wave)
    elif start_wave!=None:
        select = wave>=start_wave
    elif end_wave!=None:
        select = wave<=end_wave
    else:
        select = wave>=wave[0]
    if verbose==1:
        print(cent_x[select], spec_y._data[select])
    diff_y = cent_y[select]-spec_y._data[select]
    diff_x = cent_x[select]-spec_x._data[select]

    select2 = numpy.logical_and(numpy.logical_and(diff_y>-1*clip, diff_y <clip), numpy.logical_and(diff_x>-1*clip, diff_x <clip))

    spec2_y =Spectrum1D(data=cent_y[select][select2], wave=wave[select][select2])
    spec2_x =Spectrum1D(data=cent_x[select][select2], wave=wave[select][select2])
    poly_x = spec2_x.smoothPoly(order=smooth_poly, ref_base=cube._wave)
    poly_y = spec2_y.smoothPoly(order=smooth_poly, ref_base=cube._wave)

    if fibers==None:
      fibers=cube._dim_x*cube._dim_y

    rss_x = RSS(wave=cube._wave, data= numpy.ones(((fibers), len(cube._wave)))*spec2_x._data[numpy.newaxis, :], header=cube._header)
    rss_x._data = rss_x._data.astype(numpy.float32)

    rss_y = RSS(wave=cube._wave, data= numpy.ones(((fibers), len(cube._wave)))*spec2_y._data[numpy.newaxis, :], header=cube._header)
    rss_y._data = rss_y._data.astype(numpy.float32)

    #Save position measurements need to be changed to XML storage
    spec2_x.writeFitsData(out_prefix+'.cent_x.fits')
    spec2_y.writeFitsData(out_prefix+'.cent_y.fits')
    rss_x.writeFitsData(out_prefix+'.rss.cent_x.fits')
    rss_y.writeFitsData(out_prefix+'.rss.cent_y.fits')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.11, 0.11, 0.85, 0.87])
    ax.plot(wave[select], cent_x[select], '+m')
    ax.plot(wave[select][select2], cent_x[select][select2], '+g', label='X center')
    min_x = numpy.min(cent_x[select])
    max_x = numpy.max(cent_x[select])
    ax.plot(spec2_x._wave, spec2_x._data, '-r')
    ax.plot(wave[select], cent_y[select], '+m')
    ax.plot(wave[select][select2], cent_y[select][select2], '+b', label='Y center')
    ax.plot(spec2_y._wave, spec2_y._data, '-r', label='polynomial fit')
    legend = plt.legend(loc='upper left', numpoints=1)
    legend.draw_frame(False)
    min_y= numpy.min(cent_y[select])
    max_y = numpy.max(cent_y[select])
    ax.set_xlabel('wavelength [Angstrom]', fontsize=16)
    ax.set_ylabel('X,Y center [pixels]', fontsize=16)
    ax.set_xlim(numpy.min(spec2_y._wave), numpy.max(spec2_y._wave))
    ax.set_ylim(min([min_y, min_x])-1, max([max_y, max_x])+2)
    ax.minorticks_on()
    if figure_out!='':
        plt.savefig(figure_out)
    if verbose==1:
        plt.show()

def aperFluxCube_drp(cube_in, cent_x, cent_y, aperture, filter, hdrkey, comment='',  units='1e-16', kmax='1000', system='AB'):
    """
           Measures the aperture photometry from the cube for a given position, circular aperture radius and filter curve.
           The results are stored in a header keyword.

            Parameters
            --------------
            cube_in : string
                    Name of the INPUT FITS  cube to extract the aperture flux/photometry
            cent_x : string of float (>0)
                    Pixel position in spatial x-direction for of the aperture center
            cent_y : string of float (>0)
                    Pixel position in spatial y-direction for of the aperture center
            aperture : string of float (>0)
                    Radius of the circular aperture to extract the contained flux
            filter : string
                    Path to the ASCII filter file, appended by the wavelength column and the transmission column.
                    All the three information are comma-separated
            hdrkey : string
                    Header keyword in which the aperture photometry in magnitudes will be stored
            comment : string, optional with default: ''
                    Comment string for the header keyword entry
            units : string of float, optional with default: '1e-16'
                    Flux units of the cube spectra with respect to erg/s/cm2/A
            kmax : string of integer (>0), optional with default: '1000'
                    Defines the number of subcell per pixel to take partial pixel coverage of the aperture into account
            system : string, optional with default: 'AB'
                    Photometric magnitude system. Currently only the AB system is available

            Examples
            ----------------
            user:> lvmdrp cube aperFluxCube CUBE.fits 15 20 4 sloan_g.dat,0,1 'hiearach PIPE APER G' comment='4 arcsec g-band photometry'

    """

    kmax=int(kmax)
    units=float(units)
    filter=filter.split(',')
    aperture=float(aperture)

    cube = loadCube(cube_in)
    try:
        cent_x = cube.getHdrValue(cent_x)
    except KeyError:
        cent_x=float(cent_x)
    try:
        cent_y = cube.getHdrValue(cent_y)
    except KeyError:
        cent_y=float(cent_y)
    spec = cube.getAperSpec(cent_x, cent_y, aperture, kmax=kmax, ignore_mask=True, correct_masked=False)
    spec.writeTxtData('test_spec30.txt')
    passband = PassBand()
    passband.loadTxtFile(filter[0], wave_col=int(filter[1]),  trans_col=int(filter[2]))
    flux=passband.getFluxPass(spec)
    AB_mag = passband.fluxToMag(flux[0], error=flux[1] , system=system, units=units)

    cube.setHdrValue(hdrkey, '%.3f'%AB_mag, comment)
    cube.writeFitsHeader()

def matchAbsFluxAper_drp(cube_in, cube_out, key_mags,  ref_mags, hdrkey, comment):
    """
           Computes the average flux ratio of the previously compute cube photometry with respect to given reference magnitudes.
           A photometrically rescaled cube will be stored including additional header keywords for the ratio in different bands and the
           average ratio used for rescaling.

            Parameters
            --------------
            cube_in : string
                    Name of the INPUT FITS  cube
            cube_out :
                    Name of the OUTPUT FITS cube with rescaled photometry
            key_mags : string
                    Comma-separated list of header keywords for different photometric bands
            ref_mags : string
                    Comma-separated reference magnitudes in the same order as the header keywords.
                    A name shall be put in front of each reference magnitude being separated by a ':' sign
            hdrkey : string
                    Header keyword in which the aperture photometry in magnitudes will be stored.
                    NOTE that ' RATIO' will be appended to the given header key
            comment : string, optional with default: ''
                    Comment string for the header keyword entry

            Examples
            ----------------
            user:> lvmdrp cube matchAbsFluxAper CUBE_IN.fits CUBE_out.fits 'hierarch PIPE PHOT g,hierarch PIPE PHOT r' 'g:14.5,r:14.0' 'hierarch PIPE ABS ' "
    """
    cube = loadCube(cube_in)
    key_mags = key_mags.split(',')
    mags = numpy.zeros(len(key_mags))
    for i in range(len(mags)):
        mags[i]=float(cube.getHdrValue(key_mags[i]))
    split_mags = ref_mags.split(',')
    ref_mags =[]
    ref_names=[]

    for n in range(len(split_mags)):
        ref_mags.append(float(split_mags[n].split(':')[1]))
        ref_names.append(split_mags[n].split(':')[0].replace(' ', ''))
        cube.setHdrValue(hdrkey+' RATIO '+ref_names[n], 10**(old_div((ref_mags[n]-mags[n]),-2.5)), comment)
    ref_mags=numpy.array(ref_mags)
    diff_factor=numpy.mean(10**(old_div((ref_mags-mags),-2.5)))

    cube.setHdrValue(hdrkey+' RATIO', diff_factor, comment)
    cube._data = cube._data*diff_factor
    if cube._error!=None:
        cube._error = cube._error*diff_factor
    cube.writeFitsData(cube_out)


def matchCubeAperSpec_drp(cube_in, cube_ref, cube_out, radius, poly_correct='-3', smooth_in='0', smooth_ref='0', start_wave='', end_wave='', name_obj='', outfig='',verbose='0'):
    radius = float(radius)
    poly_correct=int(poly_correct)
    smooth_in = float(smooth_in)
    smooth_ref= float(smooth_ref)
    verbose=int(verbose)
    if start_wave=='':
        start_wave=None
    else:
        start_wave=float(start_wave)

    if end_wave=='':
        end_wave=None
    else:
        end_wave=float(end_wave)

    cube1 = loadCube(cube_in)
    xcent = cube1.getHdrValue("CRPIX1")
    ycent = cube1.getHdrValue("CRPIX2")
    spec_in= cube1.getAperSpec(xcent, ycent, radius)

    if smooth_in!=0.0:
        spec_in.smoothSpec(smooth_in)
    cube2 = loadCube(cube_ref)
    xcent = cube2.getHdrValue("CRPIX1")
    ycent = cube2.getHdrValue("CRPIX2")
    spec_ref= cube2.getAperSpec(xcent, ycent, radius)
    if smooth_ref!=0.0:
        spec_ref.smoothSpec(smooth_ref)
    select_wave = numpy.logical_and(cube1._wave>cube2._wave[3], cube1._wave<cube1._wave[-3])
    wave = cube1._wave[select_wave]
    spec_in_resamp = spec_in.resampleSpec(wave, method='linear',  err_sim=0)
    spec_ref_resamp = spec_ref.resampleSpec(wave, method='linear',  err_sim=0)

    ratio = old_div(spec_ref_resamp,spec_in_resamp)
    #if verbose==1:
    #    pylab.plot(ratio._wave, ratio._data, '-k')
    out_par = ratio.smoothPoly(order=poly_correct, start_wave=start_wave, end_wave=end_wave, ref_base=cube1._wave)
    new_cube = cube1*ratio
    new_cube.writeFitsData(cube_out)
    if verbose==1 or outfig!='':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.11, 0.11, 0.85, 0.87])
        xcent = new_cube.getHdrValue("CRPIX1")
        ycent = new_cube.getHdrValue("CRPIX2")
        spec_out= new_cube.getAperSpec(xcent, ycent, radius)
        spec_out_resamp = spec_out.resampleSpec(wave, method='linear',  err_sim=0)
        ax.plot(spec_ref_resamp._wave,spec_ref_resamp._data,'-k')
        #pylab.plot(spec_in_resamp._wave,spec_in_resamp._data,'-b')
        ax.plot(spec_out_resamp._wave,spec_out_resamp._data,'-r',ls='dashed')
        residual = spec_ref_resamp-spec_out_resamp
        ax.plot(residual._wave, residual._data, '-g')
        max_spec = numpy.max(spec_ref_resamp._data)
        ax.set_ylim([-20,max_spec+0.05*max_spec])
        ax.set_xlabel('wavelength [$\AA$]',fontsize=16)
        ax.set_ylabel('Flux [$10^{-16}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}]$',fontsize=16)
        fig.text(0.8,0.9,name_obj,fontsize=16)
        if outfig!='':
            pylab.savefig(outfig)
        if verbose==1:
            pylab.show()

def subCubeWave_drp(cube_in, cube_out, wave_start=None, wave_end=None):
    cube_in = loadCube(cube_in)
    select = numpy.ones(len(cube_in._wave), dtype="bool")
    if wave_start is not None:
	    select[cube_in._wave < float(wave_start)] = False
    if wave_end is not None:
	    select[cube_in._wave > float(wave_end)] = False
    if cube_in._error is not None:
	    error = cube_in._error[select, :, :]
    else:
	    error = None
    if cube_in._mask is not None:
	    mask = cube_in._mask[select, :, :]
    else:
	    mask = None
	
    cube_new = Cube(wave=cube_in._wave[select], data=cube_in._data[select, :, :], error=error, mask=mask, header=cube_in.getHeader())
    cube_new.writeFitsData(cube_out)


def glueCubesWavelength_drp(cube1_in, cube2_in, cube_out, rescale_region='',  merge_wavelength='0.0', mergeHdr='1'):
    merge_wavelength=float(merge_wavelength)
    mergeHdr = bool(int(mergeHdr))
    if rescale_region=='':
        rescale_region=[]
    else:
        regions = rescale_region.split(',')
        rescale_region=[float(regions[0]),float(regions[1])]
    cube = Cube()
    cube1 = loadCube(cube1_in)
    cube2 = loadCube(cube2_in)
    cube.glueCubeSets(cube1, cube2, rescale_region=rescale_region, merge_wave=merge_wavelength, mergeHdr=mergeHdr)
    cube.writeFitsData(cube_out)

def rot90_drp(cube_in, cube_out, n_rot):
    n_rot = int(n_rot)
    cube = loadCube(cube_in)
    cube_rot = cube.rot90(n_rot)
    cube_rot.writeFitsData(cube_out)

def createSensFunction_drp(cube_in, out_sens,  ref_spec, airmass, exptime, cent_x='CRPIX2', cent_y='CRPIX1', radius='6', smooth_poly='5', smooth_ref='6.0' , extinct_v='0.0', extinct_curve='mean',  aper_correct='1.0',  ref_units='1e-16', target_units='1e-16',column_wave='0', column_flux='1', delimiter='', header='1' , split='', mask_wave='', overlap='100', out_star='', verbose='0'):
    smooth_poly=int(smooth_poly)
    smooth_ref=float(smooth_ref)
    radius = float(radius)
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

    ref_star_spec = Spectrum1D()

    cube = loadCube(cube_in)
    try:
        extinct_v = cube.getHdrValue(extinct_v)
    except KeyError:
        extinct_v= float(extinct_v)

    try:
        airmass = cube.getHdrValue(airmass)
    except KeyError:
        airmass = float(airmass)

    try:
        exptime = cube.getHdrValue(exptime)
    except KeyError:
        exptime = float(exptime)

    try:
        cent_y = cube.getHdrValue(cent_y)
    except KeyError:
        cent_y = float(cent_y)

    try:
        cent_x = cube.getHdrValue(cent_x)
    except KeyError:
        cent_x = float(cent_x)

    star_spec = old_div(cube.getAperSpec(cent_x, cent_y, radius, kmax=1000, correct_masked=False, ignore_mask=True,  threshold_coverage=0.0),aper_correct)

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
    star_corr = old_div(old_div(star_spec,extinct),exptime)

    sens_func = old_div(ref_star_resamp,star_corr)
    if mask_wave!=None:
        regions = old_div(len(mask_wave),2)
        for i in range(regions):
            select_region = numpy.logical_and(sens_func._wave>mask_wave[i*2], sens_func._wave<mask_wave[i*2+1])
            select_blue = numpy.logical_and(sens_func._wave>mask_wave[i*2]-20, sens_func._wave<mask_wave[i*2])
            select_red = numpy.logical_and(sens_func._wave>mask_wave[i*2+1], sens_func._wave<mask_wave[i*2+1]+20)
            line_par = stats.linregress([mask_wave[i*2]-10,mask_wave[i*2+1]+10], [numpy.median(sens_func._data[select_blue]), numpy.median(sens_func._data[select_red])])

            sens_func._data[select_region] = (line_par[0]*sens_func._wave[select_region]+line_par[1]).astype('float32')
            #select = numpy.logical_and(sens_func._wave>mask_wave[i*2], sens_func._wave<mask_wave[i*2+1])
            #sens_func._mask[select]=True

    good_pix = numpy.logical_not(sens_func._mask)
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


def combineCubes_drp(incubes, outcube, method='mean', replace_error='1e10'):

    incubes = incubes.split(',')
    replace_error = float(replace_error)
    cubes = []
    for  i in range(len(incubes)):
        cubes.append(loadCube(incubes[i]))
    data = numpy.zeros_like(cubes[0]._data)
    if cubes[0]._error!=None:
        error = numpy.zeros_like(cubes[0]._error)
    else:
        error = None
    if cubes[0]._mask!=None:
        mask = numpy.zeros_like(cubes[0]._mask)
    else:
        mask = None

    if method=='mean':
        image = numpy.zeros((len(incubes), data.shape[1], data.shape[2]), dtype=numpy.float32)
        image2 = numpy.zeros((len(incubes), data.shape[1], data.shape[2]), dtype=numpy.float32)
        error_img = numpy.zeros((len(incubes), data.shape[1], data.shape[2]),  dtype=numpy.float32)
        mask_img = numpy.zeros((len(incubes), data.shape[1], data.shape[2]), dtype="bool")
        for i in range(data.shape[0]):
            image[:, :, :]=0
            error_img[:, :, :]=0
            mask_img[:, :, :]=False
            for j in range(len(incubes)):
                if mask!=None:
                    mask_img[j, :, :] = cubes[j]._mask[i, :, :]
                if error!=None:
                    error_img[j, :, :] = cubes[j]._error[i, :, :]
                image[j, :, :] = cubes[j]._data[i, :, :]
                image2[j, :, :] = cubes[j]._data[i, :, :]
            select = mask_img==True
            image[select]=0
            error_img[select]=0
            good_pix = numpy.sum(numpy.logical_not(select), 0)
            select_bad = good_pix==0
            select_good = good_pix>0
            data[i, select_good] = old_div(numpy.sum(image, 0)[select_good],good_pix[select_good])
            if numpy.sum(select_bad)>0:
                data[i, select_bad] = old_div(numpy.sum(image2, 0)[select_bad],len(incubes))
            if error!=None:
                error[i, select_good] = numpy.sqrt(old_div(numpy.sum(error_img**2, 0)[select_good],good_pix[select_good]**2))
                error[i, select_bad]=replace_error
            if mask!=None:
                mask[i, :, :]=select_bad


    cube_out = Cube(data=data, error=error, mask=mask, wave=cubes[0]._wave)
    cube_out.writeFitsData(outcube)


def fitCubeELines_drp(cube_in, par_file, maps_prefix, smooth_box='', wave_range='', method='leastsq',guess_window='', err_sim='0', spectral_res='0.0',bad_error_replace='1e10', ftol='1e-4', xtol='1e-4', parallel='auto'):

    err_sim=int(err_sim)
    ftol = float(ftol)
    xtol=float(xtol)
    spec_res = float(spectral_res)/2.354
    cube = loadCube(cube_in)
    wave = cube._wave
    out_model = Cube(wave=wave, data=numpy.zeros_like(cube._data), header=cube.getHeader())
    out_residual = Cube(wave=wave, data=numpy.zeros_like(cube._data), header=cube.getHeader())
    if smooth_box!='':
        out_smooth = Cube(wave=wave, data=numpy.zeros_like(cube._data), header=cube.getHeader())

    if cube._error!=None:
        select = cube._error<=0
        if numpy.sum(select)>0:
            cube._error[select]=float(bad_error_replace)

    if wave_range!='':
        rwave = open(wave_range)
        lines = rwave.readlines()
        select_wave = cube._wave<0
        for i in range(len(lines)):
            line = lines[i].split()
            if len(line)==2:
                select_wave = numpy.logical_or(numpy.logical_and(cube._wave>=float(line[0]),cube._wave<=float(line[1])),select_wave)
    else:
        select_wave=cube._wave>0
    par = fit_profile.parFile(par_file,spec_res)
    maps={}
    for n in par._names:
        model={}
        if par._profile_type[n]=='Gauss':
            if err_sim>0 and cube._error!=None:
                model['flux'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32), error=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
                model['vel'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32), error=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
                model['disp'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32), error=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
            else:
                model['flux'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
                model['vel'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
                model['disp'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
        elif par._profile_type[n]=='TemplateScale':
            if err_sim>0 and cube._error!=None:
                model['scale'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32), error=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
            else:
                model['scale'] = Image(data=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32), error=numpy.zeros((cube._dim_y, cube._dim_x), dtype=numpy.float32))
        maps[n]=model
    z=1
    for i in range(cube._dim_y):
        for j in range(cube._dim_x):
            sys.stdout.write('Processing... %.1f/100%%\r'%(z/float(cube._dim_x*cube._dim_y)*100))
            sys.stdout.flush()
            #print i, j
            if numpy.sum(cube._data[:, i, j]!=0)>0:
                fit_par=deepcopy(par)

                if smooth_box!='':
                    spec_init = Spectrum1D(wave=wave, data=cube._data[:, i, j])
                    spec_init.smoothSpec(int(smooth_box), method='median')
                    spec = Spectrum1D(wave=wave, data=cube._data[:, i, j], error=cube._error[:, i, j], mask=cube._mask[:, i, j])
                    spec = spec-spec_init
                    out_smooth._data[:, i, j]=spec_init._data
                    spec_fit = Spectrum1D(wave=wave[select_wave], data=spec._data[select_wave], error=spec._error[select_wave], mask=spec._mask[select_wave])
                else:
                    spec_fit = Spectrum1D(wave=wave[select_wave], data=cube._data[select_wave, i, j], error=cube._error[select_wave, i, j], mask=cube._mask[select_wave, i, j])
                if guess_window!='':
                    fit_par._guess_window=int(guess_window)
                    fit_par.guessPar(spec_fit._wave,spec_fit._data)
                spec_fit.fitParFile(fit_par, err_sim=err_sim, method=method,ftol=ftol, xtol=xtol, parallel=parallel)
                out_model._data[:, i, j]=fit_par(wave)
                out_residual._data[:, i, j]= cube._data[:, i, j]-out_model._data[:, i, j]
                #print i, j, 'done'
                for n in fit_par._names:
                    if fit_par._profile_type[n]=='Gauss':
                        maps[n]['flux'] ._data[i, j]=fit_par._parameters[n]['flux']
                        maps[n]['vel'] ._data[i, j]=fit_par._parameters[n]['vel']
                        maps[n]['disp'] ._data[i, j]=numpy.fabs(fit_par._parameters[n]['disp'])*2.354
                        if err_sim>0 and cube._error!=None:
                            maps[n]['flux'] ._error[i, j]=fit_par._parameters_err[n]['flux']
                            maps[n]['vel'] ._error[i, j]=fit_par._parameters_err[n]['vel']
                            maps[n]['disp'] ._error[i, j]=numpy.fabs(fit_par._parameters_err[n]['disp'])*2.354
                    elif fit_par._profile_type[n]=='TemplateScale':
                        maps[n]['scale'] ._data[i, j]=fit_par._parameters[n]['scale']
                        if err_sim>0 and cube._error!=None:
                            maps[n]['scale'] ._error[i, j]=fit_par._parameters_err[n]['scale']
            z+=1
    sys.stdout.write('\n')
    out_model.writeFitsData(maps_prefix+'_line_model.fits')
    out_residual.writeFitsData(maps_prefix+'_line_res.fits')
    if smooth_box!='':
        out_smooth.writeFitsData(maps_prefix+'_smooth_cont.fits')

    for n in par._names:
        if fit_par._profile_type[n]=='Gauss':
            maps[n]['flux'].writeFitsData(maps_prefix+'_Gauss_'+n+'_flux.fits')
            maps[n]['vel'].writeFitsData(maps_prefix+'_Gauss_'+n+'_vel.fits')
            maps[n]['disp'].writeFitsData(maps_prefix+'_Gauss_'+n+'_disp.fits')
        elif fit_par._profile_type[n]=='TemplateScale':
            maps[n]['scale'].writeFitsData(maps_prefix+'_Template_'+n+'_scale.fits')

