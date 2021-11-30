from builtins import range
from astropy.io import fits as pyfits
import os
from lvmdrp.functions.imageMethod import *
from lvmdrp.functions.rssMethod import *
from lvmdrp.functions.headerMethod import *
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.core.header import Header
from lvmdrp.core.image import Image
from multiprocessing import cpu_count
from multiprocessing import Pool

description = 'Provides Methods to reduce GMOS data'
gmos_calib = os.path.dirname(__file__) + '/../../config/GMOS/'


def createCCDfromArchive_drp(infile, prefix, master_bias=None,splits='0,0,0,0'):
    splits = numpy.array(splits.split(',')).astype(numpy.int16)
    hdu = pyfits.open(infile)
    CCD = numpy.zeros((4608, 6144), dtype=numpy.float32)
    CCD_err = numpy.zeros((4608, 6144), dtype=numpy.float32)

    if master_bias is not None:
        hdu_bias = pyfits.open(master_bias)

    for i in range(1,7):
        temp_sec = hdu[i].header['DETSEC'].replace('[', '').replace(']', '').split(',')
        CCDsec_x = (int(temp_sec[0].split(':')[0])-1,int(temp_sec[0].split(':')[1]))
        CCDsec_y = (int(temp_sec[1].split(':')[0])-1,int(temp_sec[1].split(':')[1]))

        temp_dat = hdu[i].header['DATASEC'].replace('[', '').replace(']', '').split(',')
        DATAsec_x = (int(temp_dat[0].split(':')[0])-1,int(temp_dat[0].split(':')[1]))
        DATAsec_y = (int(temp_dat[1].split(':')[0])-1,int(temp_dat[1].split(':')[1]))
        img = hdu[i].data[DATAsec_y[0]:DATAsec_y[1], DATAsec_x[0]:DATAsec_x[1]]

        temp_bias = hdu[i].header['BIASSEC'].replace('[', '').replace(']', '').split(',')
        BIASsec_x = (int(temp_bias[0].split(':')[0])-1,int(temp_bias[0].split(':')[1]))
        BIASsec_y = (int(temp_bias[1].split(':')[0])-1,int(temp_bias[1].split(':')[1]))
        bias = numpy.median(hdu[i].data[BIASsec_y[0]:BIASsec_y[1], BIASsec_x[0]:BIASsec_x[1]])
        gain = hdu[i].header['gain']
        rdnoise = hdu[i].header['rdnoise']
        if master_bias is not None:
            img_bias = hdu_bias[i].data
            gain = hdu_bias[i].header['GAINORIG']
            rdnoise = hdu_bias[i].header['RONORIG']
            bias_bias = numpy.median(img_bias)
            bias = bias - (img_bias-bias_bias)
        CCD[CCDsec_y[0]:CCDsec_y[1], CCDsec_x[0]:CCDsec_x[1]] = (img-bias)*gain
        if master_bias is not None:
            img[(img-bias)<0]=bias[(img-bias)<0]
        else:
            img[(img-bias)<0]=bias
        CCD_err[CCDsec_y[0]:CCDsec_y[1], CCDsec_x[0]:CCDsec_x[1]] = numpy.sqrt((img-bias)+rdnoise**2)

    if splits[0]==0:
        CCD1_out = Image(data=CCD[:,0:2047],error=CCD_err[:,0:2047])
    else:
        CCD1_out = Image(data=CCD[:,splits[0]:2047],error=CCD_err[:,splits[0]:2047])
    CCD1_out.writeFitsData(prefix+'.CCD1.fits')
    if splits[1]==0 and splits[2]==0:
        CCD2_out = Image(data=CCD[:,2048:4095],error=CCD_err[:,2048:4095])
        CCD2_out.writeFitsData(prefix+'.CCD2.fits')
    else:
        CCD2_out = Image(data=CCD[:,2048:2048+splits[1]],error=CCD_err[:,2048:2048+splits[1]])
        CCD2_out.writeFitsData(prefix+'.CCD2L.fits')
        CCD2_out = Image(data=CCD[:,2048+splits[2]:4095],error=CCD_err[:,2048+splits[2]:4095])
        CCD2_out.writeFitsData(prefix+'.CCD2R.fits')
    if splits[3]==0:
        CCD3_out = Image(data=CCD[:,4096:6143],error=CCD_err[:,4096:6143])
    else:
        CCD3_out = Image(data=CCD[:,4096:4096+splits[3]],error=CCD_err[:,4096:4096+splits[3]])
    CCD3_out.writeFitsData(prefix+'.CCD3.fits')

def combineBias_drp(file_list, file_out):
    files = open(file_list, 'r')
    lines = files.readlines()
    hdu = pyfits.open(lines[0])
    hdulist = [pyfits.PrimaryHDU()]
    for j in range(1, len(hdu)):
        for i in range(len(lines)):
            hdu = pyfits.open(lines[i])
            if i == 0:
                coadd = numpy.zeros(hdu[j].data.shape, dtype=numpy.float32)
            coadd += hdu[j].data
        mean  = coadd / float(len(lines))
        hdu_out = pyfits.ImageHDU(mean)
        hdu_out.header = hdu[j].header
        hdulist.append(hdu_out)
    hdu = pyfits.HDUList(hdulist)
    hdu.writeto(file_out, overwrite=True)


def reduceCalib_drp(trace, master_bias, arc=''):
    hdr_trace = Header()
    hdr_trace.loadFitsHeader(trace)
    IFU_mask = hdr_trace.getHdrValue('MASKNAME')
    grating = hdr_trace.getHdrValue('GRATING')
    centwave = float(hdr_trace.getHdrValue('CENTWAVE'))

    if IFU_mask == 'IFU-2':
        if grating == 'R400+_G5305' and centwave == 800.0:
            splits = '500,1000,1760,0'
            slice1 = 500
            slice2L = 400
            slice2R = 250
            slice3 = 600
            setup = '2R400_800'
            poly_disp = [-5, -5, -2, -5]
            poly_fwhm = ['-3,-3', '-3,-2', '-3,-3', '-3,-3']
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
        ccds = ['CCD1', 'CCD2L', 'CCD2R', 'CCD3']
        steps = [40, 20, 20, 40]

        createCCDfromArchive_drp(trace, 'FLAT', master_bias=master_bias, splits=splits)
        if arc != '':
            createCCDfromArchive_drp(arc, 'ARC', master_bias=master_bias, splits=splits)
        findPeaksAuto_drp('FLAT.CCD1.fits', 'peaks.CCD1', nfibers=750, slice=slice1, verbose=0)
        findPeaksAuto_drp('FLAT.CCD2L.fits', 'peaks.CCD2L', nfibers=750, slice=slice2L, median_cross=2, verbose=0)
        findPeaksMaster2_drp('FLAT.CCD2R.fits', 'master_peaks.BLUE_slit', 'peaks.CCD2R', threshold=10000, slice=slice2R,
            median_cross=3, verbose=0)
        findPeaksMaster2_drp('FLAT.CCD3.fits', 'master_peaks.BLUE_slit', 'peaks.CCD3', threshold=10000, slice=slice3,
            median_cross=3, verbose=0)
        tracePeaks_drp('FLAT.CCD1.fits', 'peaks.CCD1', 'tjunk.CCD1.trc.fits', poly_disp='-5', steps=50, threshold_peak=400)
        tracePeaks_drp('FLAT.CCD2L.fits', 'peaks.CCD2L', 'tjunk.CCD2L.trc.fits', poly_disp='-5', steps=50, coadd=1, max_diff=1,
            threshold_peak=400, median_cross=2)
        tracePeaks_drp('FLAT.CCD2R.fits', 'peaks.CCD2R', 'tjunk.CCD2R.trc.fits', poly_disp='-5', steps=20, coadd=1,
            max_diff=1, median_cross=3, threshold_peak=400)
        tracePeaks_drp('FLAT.CCD3.fits', 'peaks.CCD3', 'tjunk.CCD3.trc.fits', poly_disp='-5', steps=50, coadd=1,
            max_diff=1, median_cross=3, threshold_peak=400)
        for i in range(len(ccds)):
            subtractStraylight_drp('FLAT.%s.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'FLAT.%s.back.fits' % (ccds[i]),
                'FLAT.%s.stray.fits' % (ccds[i]), aperture=10, poly_cross=6, smooth_disp=30)
            traceFWHM_drp('FLAT.%s.stray.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'tjunk.%s.fwhm.fits' % (ccds[i]),
                blocks=16, steps=steps[i], poly_disp=-5, init_fwhm=3, clip='1.0,6.0', threshold_flux=2000)
            if arc != '':
                subtractStraylight_drp('ARC.%s.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]),
                    'ARC.%s.back.fits' % (ccds[i]), 'ARC.%s.stray.fits' % (ccds[i]), aperture=10, poly_cross=6, smooth_disp=30)
                extractSpec_drp('ARC.%s.stray.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'ARC.%s.ms.fits' % (ccds[i]),
                    method='aperture', aperture=5)
                detWaveSolution_drp('ARC.%s.ms.fits' % (ccds[i]), 'ARC.%s' % (ccds[i]), '../../arc.%s.%s.txt' %
                    (ccds[i], setup), poly_dispersion=poly_disp[i], poly_fwhm=poly_fwhm[i], flux_min=100.0, fwhm_max=8.0,
                    rel_flux_limits='0.1,6.0')
            extractSpec_drp('FLAT.%s.stray.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'FLAT.%s.ms.fits' % (ccds[i]),
                method='optimal', fwhm='tjunk.%s.fwhm.fits' % (ccds[i]), parallel=1)
            createPixTable_drp('FLAT.%s.ms.fits' % (ccds[i]), 'FLAT.%s.rss.fits' % (ccds[i]), 'ARC.%s.disp.fits' % (ccds[i]),
                'ARC.%s.res.fits' % (ccds[i]))
        glueRSS_drp('FLAT.CCD1.rss.fits,FLAT.CCD2L.rss.fits', 'FLAT_red.rss.fits')
        glueRSS_drp('FLAT.CCD2R.rss.fits,FLAT.CCD3.rss.fits', 'FLAT_blue.rss.fits')
        resampleWave_drp('FLAT_blue.rss.fits', 'FLAT_blue.disp_cor.fits', start_wave=start_wave, end_wave=end_wave,
            disp_pix=disp_pix, err_sim=0, method='linear', parallel=1)
        resampleWave_drp('FLAT_red.rss.fits', 'FLAT_red.disp_cor.fits', start_wave=start_wave, end_wave=end_wave,
            disp_pix=disp_pix, err_sim=0, method='linear')
        mergeRSS_drp('FLAT_red.disp_cor.fits,FLAT_blue.disp_cor.fits', 'FLAT.disp_cor.fits')
        createFiberFlat_drp('FLAT.disp_cor.fits', 'FIBERFLAT.fits', clip='0.25,2.0')

    elif IFU_mask == 'IFU-R' or IFU_mask == 'IFU-B':
        ccds = ['CCD1', 'CCD2', 'CCD3']
        splits = '0,0,0,0'
        if grating == 'R400+_G5305' and centwave == 700.0 and IFU_mask == 'IFU-R':
            setup = '1RR400_800'
            slice1 = 1000
            slice2 = 1000
            slice3 = 100
            poly_disp = [-6, -6, -6]
            poly_fwhm = ['-2,-3', '-2,-3', '-2,-3']
            start_wave = 5800
            end_wave = 9000
            disp_pix = 0.7
        createCCDfromArchive_drp(trace, 'FLAT', master_bias=master_bias, splits=splits)
        if arc != '':
            createCCDfromArchive_drp(arc, 'ARC', master_bias=master_bias, splits=splits)
        findPeaksAuto_drp('FLAT.CCD1.fits', 'peaks.CCD1', nfibers=750, slice=slice1, verbose=0)
        findPeaksAuto_drp('FLAT.CCD2.fits', 'peaks.CCD2', nfibers=750, slice=slice2, median_cross=2, verbose=0)
        findPeaksAuto_drp('FLAT.CCD3.fits', 'peaks.CCD3', nfibers=750, slice=slice3, verbose=0)
        tracePeaks_drp('FLAT.CCD1.fits', 'peaks.CCD1', 'tjunk.CCD1.trc.fits', poly_disp='-5', steps=50, max_diff=1,
            threshold_peak=400)
        tracePeaks_drp('FLAT.CCD2.fits', 'peaks.CCD2', 'tjunk.CCD2.trc.fits', poly_disp='-5', steps=50, max_diff=1,
            median_cross=2, threshold_peak=400)
        tracePeaks_drp('FLAT.CCD3.fits', 'peaks.CCD3', 'tjunk.CCD3.trc.fits', poly_disp='-5', steps=50, max_diff=1,
            threshold_peak=400)
        for i in range(len(ccds)):
            subtractStraylight_drp('FLAT.%s.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'FLAT.%s.back.fits' % (ccds[i]),
                'FLAT.%s.stray.fits' % (ccds[i]), aperture=10, poly_cross=6, smooth_disp=30)
            traceFWHM_drp('FLAT.%s.stray.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'tjunk.%s.fwhm.fits' % (ccds[i]),
                blocks=16, steps=50, poly_disp=-5, init_fwhm=3, clip='1.0,6.0', threshold_flux=2000)
            if arc != '':
                subtractStraylight_drp('ARC.%s.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]),
                    'ARC.%s.back.fits' % (ccds[i]), 'ARC.%s.stray.fits' % (ccds[i]), aperture=10, poly_cross=6, smooth_disp=30)
                extractSpec_drp('ARC.%s.stray.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'ARC.%s.ms.fits' % (ccds[i]),
                    method='aperture', aperture=5)
                detWaveSolution_drp('ARC.%s.ms.fits' % (ccds[i]), 'ARC.%s' % (ccds[i]), 'arc.%s.%s.txt' %
                    (ccds[i], setup), poly_dispersion=poly_disp[i], poly_fwhm=poly_fwhm[i], flux_min=100.0, fwhm_max=8.0,
                    rel_flux_limits='0.1,6.0')
            extractSpec_drp('FLAT.%s.stray.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'FLAT.%s.ms.fits' % (ccds[i]),
                method='optimal', fwhm='tjunk.%s.fwhm.fits' % (ccds[i]), parallel=1)
            createPixTable_drp('FLAT.%s.ms.fits' % (ccds[i]), 'FLAT.%s.rss.fits' % (ccds[i]), 'ARC.%s.disp.fits' % (ccds[i]),
                'ARC.%s.res.fits' % (ccds[i]))

        glueRSS_drp('FLAT.CCD1.rss.fits,FLAT.CCD2.rss.fits,FLAT.CCD3.rss.fits', 'FLAT.rss.fits')
        resampleWave_drp('FLAT.rss.fits', 'FLAT.disp_cor.fits', start_wave=start_wave, end_wave=end_wave,
            disp_pix=disp_pix, err_sim=0, method='linear', parallel=1)
        createFiberFlat_drp('FLAT.disp_cor.fits', 'FIBERFLAT.fits', clip='0.25,2.0')