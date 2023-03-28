import os
from multiprocessing import Pool, cpu_count

from astropy.io import fits as pyfits

from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.functions.headerMethod import *
from lvmdrp.functions.imageMethod import *
from lvmdrp.functions.rssMethod import *
from lvmdrp.functions.specialMethod import *


description='Provides Methods to reduce VIMOS data'

vimos_calib=os.path.dirname(__file__)+'/../../config/VIMOS/'

def renameFiles_drp(year):
    dir = os.listdir('.')
    for f in dir:
        if 'VIMOS.' in f and '.fits' in f and str(year) in f:
            #print(f)
            hdr = pyfits.getheader(f, 0, ignore_missing_end=True)
            orig_name=hdr['ORIGFILE']
            os.system('mv  %s %s'%(f, orig_name))

def createBIAS_drp(night):

  night = int(night)
  os.system('ls VIMOS_SPEC_BIAS%03d_*_B.1.fits* > combine_BIAS.B1'%(night))
  combineImages_drp('combine_BIAS.B1','BIAS_B.1.fits', method="median")
  os.system('rm combine_BIAS.B1')
  os.system('ls VIMOS_SPEC_BIAS%03d_*_A.2.fits* > combine_BIAS.A2'%(night))
  combineImages_drp('combine_BIAS.A2', 'BIAS_A.2.fits', method="median")
  os.system('rm combine_BIAS.A2')
  os.system('ls VIMOS_SPEC_BIAS%03d_*_A.3.fits* > combine_BIAS.A3'%(night))
  combineImages_drp('combine_BIAS.A3','BIAS_A.3.fits', method="median")
  os.system('rm combine_BIAS.A3')
  os.system('ls VIMOS_SPEC_BIAS%03d_*_B.4.fits* > combine_BIAS.B4'%(night))
  combineImages_drp('combine_BIAS.B4',  'BIAS_B.4.fits', method="median")
  os.system('rm combine_BIAS.B4')

def combineLAMP_drp(night):
  night = int(night)
  os.system('ls VIMOS_IFU_LAMP%03d_*_B.1.fits* > combine_LAMP.B1'%(night))
  combineImages_drp('combine_LAMP.B1','LAMP_B.1.fits', method="median")
  os.system('rm combine_LAMP.B1')
  os.system('ls VIMOS_IFU_LAMP%03d_*_A.2.fits* > combine_LAMP.A2'%(night))
  combineImages_drp('combine_LAMP.A2', 'LAMP_A.2.fits', method="median")
  os.system('rm combine_LAMP.A2')
  os.system('ls VIMOS_IFU_LAMP%03d_*_A.3.fits* > combine_LAMP.A3'%(night))
  combineImages_drp('combine_LAMP.A3','LAMP_A.3.fits', method="median")
  os.system('rm combine_LAMP.A3')
  os.system('ls VIMOS_IFU_LAMP%03d_*_B.4.fits* > combine_LAMP.B4'%(night))
  combineImages_drp('combine_LAMP.B4',  'LAMP_B.4.fits', method="median")
  os.system('rm combine_LAMP.B4')


def combineTwilight_drp(night, numbers):
  list = numbers.split(',')
  night = int(night)

  for i in range(len(list)):
    if i==0:
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_B.1.* > combine_LAMP.B1'%(night, int(list[i])))
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_A.2.* > combine_LAMP.A2'%(night, int(list[i])))
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_A.3.* > combine_LAMP.A3'%(night, int(list[i])))
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_B.4.* > combine_LAMP.B4'%(night, int(list[i])))
    else:
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_B.1.* >> combine_LAMP.B1'%(night, int(list[i])))
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_A.2.* >> combine_LAMP.A2'%(night, int(list[i])))
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_A.3.* >> combine_LAMP.A3'%(night, int(list[i])))
        os.system('ls VIMOS_IFU_TwFlats%03d_%04d_B.4.* >> combine_LAMP.B4'%(night, int(list[i])))

  combineImages_drp('combine_LAMP.B1','LAMP_B.1.fits', method="mean")
  combineImages_drp('combine_LAMP.A2', 'LAMP_A.2.fits', method="mean")
  combineImages_drp('combine_LAMP.A3','LAMP_A.3.fits', method="mean")
  combineImages_drp('combine_LAMP.B4', 'LAMP_B.4.fits', method="mean")
  os.system('rm combine_LAMP.B1')
  os.system('rm combine_LAMP.A2')
  os.system('rm combine_LAMP.A3')
  os.system('rm combine_LAMP.B4')





def prepareCalib_drp(night, chip, boundary_x,boundary_y, peaks_ref, ARC_ref, start_wave, end_wave, disp_wave,fwhm_max, setup='', border='4', trace_master='',fiberflat='1', fiberflat_wave='0', CCD_mask='',parallel='1'):
    night=int(night)
    start_wave = float(start_wave)
    end_wave = float(end_wave)
    disp_wave=float(disp_wave)
    fiberflat=int(fiberflat)
    fiberflat_wave=int(fiberflat_wave)
    subtractBias_drp('LAMP_%s.fits'%(chip),  'LAMP_%s.sub.fits'%(chip), 'BIAS_%s.fits'%(chip), boundary_x=boundary_x,  boundary_y=boundary_y,  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON', subtract_light='1')
    if CCD_mask!='':
      addCCDMask_drp('LAMP_%s.sub.fits'%(chip),'%s/CCDMASK_%s_%s.fits'%(vimos_calib,chip,CCD_mask))
    if setup!='' and setup=='orange':
        findPeaksMaster2_drp('LAMP_%s.sub.fits'%(chip),  peaks_ref, 'peaks_%s.txt'%(chip), disp_axis='y', slice='2450', border=border, verbose=0)
    elif setup!='' and setup=='blue':
        findPeaksMaster2_drp('LAMP_%s.sub.fits'%(chip),  peaks_ref, 'peaks_%s.txt'%(chip), disp_axis='y', slice='3400', border=border, verbose=0)
    else:
        findPeaksMaster2_drp('LAMP_%s.sub.fits'%(chip),  peaks_ref, 'peaks_%s.txt'%(chip), disp_axis='y', border=border, verbose=0)

    tracePeaks_drp('LAMP_%s.sub.fits'%(chip),  'peaks_%s.txt'%(chip), 'tjunk_%s.trc.fits'%(chip), disp_axis='y', poly_disp='-5',steps=30,  threshold_peak=25, coadd=120, verbose=0)
    if trace_master=='':
      subtractStraylight_drp('LAMP_%s.sub.fits'%(chip),'tjunk_%s.trc.fits'%(chip), 'LAMP_%s.back.fits'%(chip), 'LAMP_%s.stray.fits'%(chip), disp_axis='y', aperture=10, poly_cross=0, smooth_disp=30, parallel=parallel)
      traceFWHM_drp('LAMP_%s.stray.fits'%(chip),  'tjunk_%s.trc.fits'%(chip),  'tjunk.fwhm_%s.fits'%(chip),  disp_axis='y',  blocks=16 , steps=50,  poly_disp=-5,  init_fwhm=3,  clip='1,6.0',  parallel=parallel)
    else:
      matchMasterTrace_drp('tjunk_%s.trc.fits'%(chip),'master_%s.trc.fits'%(chip),'tjunk_%s.trc.fits'%(chip), split='', poly_cross=1, poly_disp=1, start_pix=trace_master[0], end_pix=trace_master[1])
      subtractStraylight_drp('LAMP_%s.sub.fits'%(chip),'tjunk_%s.trc.fits'%(chip), 'LAMP_%s.back.fits'%(chip), 'LAMP_%s.stray.fits'%(chip), disp_axis='y', aperture=15, poly_cross=0, smooth_disp=30, parallel=parallel)
      os.system('cp master.fwhm_%s.fits tjunk.fwhm_%s.fits'%(chip,chip))
    if fiberflat_wave==1:
      extractSpec_drp('WAVE_%s.sub.fits'%(chip), 'tjunk_%s.trc.fits'%(chip), 'WAVE_%s.ms.fits'%(chip),   method='optimal',  fwhm='tjunk.fwhm_%s.fits'%(chip),  disp_axis='y',  parallel=parallel)
      detWaveSolution_drp('WAVE_%s.ms.fits'%(chip),  'WAVE_%s'%(chip),  ARC_ref,  poly_dispersion='-5',  poly_fwhm='-3,-5',  flux_min='20.0',  aperture='10', fwhm_max=str(fwhm_max),  fiberflat='%.2f,%.2f,%.2f,LAMP_%s.disp_cor'%(start_wave,end_wave,disp_wave,chip),rel_flux_limits='0.3,4.0',  verbose='1')
    else:
      extractSpec_drp('WAVE_%s.sub.fits'%(chip), 'tjunk_%s.trc.fits'%(chip),  'WAVE_%s.ms.fits'%(chip),  method='aperture',  aperture=5,  disp_axis='y',  parallel=parallel)
      detWaveSolution_drp('WAVE_%s.ms.fits'%(chip),  'WAVE_%s'%(chip),  ARC_ref,  poly_dispersion='-5',  poly_fwhm='-3,-5',  flux_min='20.0',  aperture='10', fwhm_max=str(fwhm_max),  rel_flux_limits='0.1,6.0',  verbose='0')
    createPixTable_drp('WAVE_%s.ms.fits'%(chip), 'WAVE_%s.rss.fits'%(chip),  'WAVE_%s.disp.fits'%(chip),  'WAVE_%s.res.fits'%(chip))
    resampleWave_drp('WAVE_%s.rss.fits'%(chip),  'WAVE_%s.disp_cor.fits'%(chip),  start_wave=start_wave,  end_wave=end_wave,  disp_pix=disp_wave,  err_sim=0,  parallel=parallel)
    if fiberflat==1:
        extractSpec_drp('LAMP_%s.stray.fits'%(chip), 'tjunk_%s.trc.fits'%(chip), 'LAMP_%s.ms.fits'%(chip),   method='optimal',  fwhm='tjunk.fwhm_%s.fits'%(chip),  disp_axis='y',  parallel=parallel)
        createPixTable_drp('LAMP_%s.ms.fits'%(chip), 'LAMP_%s.rss.fits'%(chip),  'WAVE_%s.disp.fits'%(chip),  'WAVE_%s.res.fits'%(chip))
        resampleWave_drp('LAMP_%s.rss.fits'%(chip),  'LAMP_%s.disp_cor.fits'%(chip),  start_wave=start_wave,  end_wave=end_wave,  disp_pix=disp_wave,  err_sim=0, compute_densities=1, parallel=parallel)
    if fiberflat==1 or fiberflat_wave==1:
      includePosTab_drp('LAMP_%s.disp_cor.fits'%(chip), '%s/vimos_HR_%s_pt.txt'%(vimos_calib,chip))
        
    
def prepareObject_drp(name_obj, night, object, chip, boundary_x, boundary_y, sky_line_list, start_wave, end_wave, disp_wave, resolution_fwhm, flexure_order, CCD_mask='',correct_HVEL=False, straylight=True, parallel='1'):

    #if CCD_mask!='':
    #  addCCDMask_drp('%s_%s.cosmic.fits'%(name_obj,chip),'%s/CCDMASK_%s_%s.fits'%(vimos_calib,chip,CCD_mask))
    if sky_line_list!='':
        #offsetTrace_drp('%s_%s.cosmic.fits'%(name_obj, chip), 'tjunk_%s.trc.fits'%(chip),  'WAVE_%s.disp.fits'%(chip),  sky_line_list,  'offsetTrace_%s.log'%(chip),  blocks='10',  disp_axis='y',  size='30')
        correctTraceMask_drp('tjunk_%s.trc.fits'%(chip), 'tjunk_%s_temp.trc.fits'%(chip),  'offsetTrace_%s.log'%(chip),  '%s_%s.cosmic.fits'%(name_obj, chip),  poly_smooth=flexure_order)
    else:
        os.system('cp tjunk_%s.trc.fits tjunk_%s_temp.trc.fits'%(chip, chip))

    if straylight:
      subtractStraylight_drp('%s_%s.cosmic.fits'%(name_obj, chip),'tjunk_%s_temp.trc.fits'%(chip), '%s_%s.back.fits'%(name_obj, chip),'%s_%s.stray.fits'%(name_obj, chip), disp_axis='y', aperture=10, poly_cross=0, smooth_disp=30, parallel=parallel)
    else:
      os.system('cp %s_%s.cosmic.fits %s_%s.stray.fits'%(name_obj, chip, name_obj, chip))
    
    extractSpec_drp('%s_%s.stray.fits'%(name_obj, chip),  'tjunk_%s_temp.trc.fits'%(chip),  '%s_%s.ms.fits'%(name_obj, chip),   method='optimal',  fwhm='tjunk.fwhm_%s.fits'%(chip),  disp_axis='y',  parallel=parallel)

    createPixTable_drp('%s_%s.ms.fits'%(name_obj, chip),  '%s_%s.pix_tab.fits'%(name_obj, chip),  'WAVE_%s.disp.fits'%(chip),  'WAVE_%s.res.fits'%(chip))

    addHvelcorHdr_drp('%s_%s.pix_tab.fits'%(name_obj, chip), 'HVEL_COR', RAKey='RA', RAUnit='d', DECKey='DEC', ObsLongKey='ESO TEL GEOLON', LongSignFlip=1, ObsLatKey='ESO TEL GEOLAT', ObsAltKey='ESO TEL GEOELEV', ModJulKey='MJD-OBS', extension='0')
    if correct_HVEL:
      HVEL_key='HVEL_COR'
    else:
      HVEL_key=''

    if sky_line_list!='':
        checkPixTable_drp('%s_%s.pix_tab.fits'%(name_obj, chip),  sky_line_list,  'offsetWave_%s.log'%(chip), aperture='12')
        correctPixTable_drp('%s_%s.pix_tab.fits'%(name_obj, chip),  '%s_%s.pix_tab.fits'%(name_obj, chip), 'offsetWave_%s.log'%(chip),  '%s_%s.pix_tab.fits'%(name_obj, chip),  smooth_poly_cross='1',  smooth_poly_disp=flexure_order,  poly_disp='5')
    if float(resolution_fwhm)!=0.0:
        matchResolution_drp('%s_%s.pix_tab.fits'%(name_obj, chip),  '%s_%s.res.fits'%(name_obj, chip),  resolution_fwhm,  parallel=parallel)
        resampleWave_drp('%s_%s.res.fits'%(name_obj, chip), '%s_%s.disp_cor.fits'%(name_obj, chip), start_wave=start_wave,  end_wave=end_wave,  disp_pix=disp_wave,  err_sim='500', compute_densities=1, correctHvel=HVEL_key, parallel=parallel)
    else:
        resampleWave_drp('%s_%s.pix_tab.fits'%(name_obj, chip), '%s_%s.disp_cor.fits'%(name_obj, chip), start_wave=start_wave,  end_wave=end_wave,  disp_pix=disp_wave,  err_sim='500', compute_densities=1, correctHvel=HVEL_key, parallel=parallel)
    includePosTab_drp('%s_%s.disp_cor.fits'%(name_obj, chip), '%s/vimos_HR_%s_pt.txt'%(vimos_calib,chip))

def reduceCalibMR_drp(night, fiberflat='1', fiberflat_wave='0', trace_master='', wave_start=4880, wave_end=9300, wave_disp=2.0, parallel='auto'):

    chips=['B.1', 'A.2', 'A.3', 'B.4']
    fiberflat=int(fiberflat)
    fiberflat_wave=int(fiberflat_wave)

    image = loadImage('LAMP_B.1.fits')
    date = image.getHdrValue('ESO OBS START').split('T')[0]
    year = int(date.split('-')[0])
    
    if year>=2013:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1360,3510', '1165,3215', '900,3100']
        peaks_guess  = 'peaks_2013.txt'
        CCD_mask = ''
        border=1
    elif year>=2007:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1340,3490', '1150,3300', '900,3150']
        peaks_guess  = 'peaks.txt'
        CCD_mask = ''
        border=1
    else:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1283,3433', '1133,3283', '920,3170']
        peaks_guess  = 'peaks_vearly.txt'
        CCD_mask = ''
        border=1
    
    for i in range(len(chips)):
        dir=os.listdir('.')
        for file_name in dir:
           if 'VIMOS_IFU_WAVE%03d'%(int(night)) in file_name and chips[i] in file_name:
                subtractBias_drp(file_name,  'WAVE_%s.sub.fits'%(chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')


    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)
    if cpus>1:
        pool = Pool(cpus)
        result=[]
        for i in range(len(chips)):
            result.append(pool.apply_async(prepareCalib_drp, args=(night,chips[i],boundaries_x[i],boundaries_y[i],'%s/master_VIMOS_%s_%s'%(vimos_calib,chips[i], peaks_guess), '%s/ref_lines_ARC_VIMOS_MR_%s.txt'%(vimos_calib, chips[i]), wave_start, wave_end,wave_disp,15, '', border, trace_master, fiberflat, fiberflat_wave, CCD_mask,1)))
        pool.close()
        pool.join()
    else:
        for i in range(len(chips)):
            prepareCalib_drp(night,chips[i],boundaries_x[i],boundaries_y[i],'%s/master_VIMOS_%s_%s'%(vimos_calib,chips[i], peaks_guess), '%s/ref_lines_ARC_VIMOS_MR_%s.txt'%(vimos_calib, chips[i]), wave_start, wave_end, wave_disp, 15, '', border, trace_master, fiberflat, fiberflat_wave, CCD_mask,'auto')


    if fiberflat==1:
        mergeRSS_drp('LAMP_B.1.disp_cor.fits,LAMP_A.2.disp_cor.fits,LAMP_A.3.disp_cor.fits,LAMP_B.4.disp_cor.fits', 'LAMP.disp_cor.fits')
        createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='1200,1600')


def reduceCalibHR_drp(night, fiberflat='1', fiberflat_wave='0',master_trace='0',wave_start=4880, wave_end=9300, wave_disp=2.0, setup='orange', except_chip='',parallel='auto'):
    chips=['B.1', 'A.2', 'A.3', 'B.4']
    if except_chip!='':
        except_chips = except_chip.split(',')
        for i in range(len(except_chips)):
            chips= [x for x in chips if x != except_chips[i]]
    fiberflat=int(fiberflat)
    fiberflat_wave=int(fiberflat_wave)
    

    image = loadImage('LAMP_B.1.fits')
    date = image.getHdrValue('ESO OBS START').split('T')[0]
    year = int(date.split('-')[0])
    if date>'2012-04-15' and setup=='blue':
      boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
      #boundaries_y=['1950,4096', '1950,4096', '1800,4096', '1600,4096']
      boundaries_y=['1410,4096', '1410,4096', '1410,4096', '1410,4096']
      trace_limits=[[840,2680],[840,2680],[690,2680],[490,2680]]
    else:
      boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
      boundaries_y=['1,4096', '1,4096', '1,4096', '1,4096']
    if year==2009:
        peaks_guess  = 'peaks_2009.txt'
        wave_guess = '_2009.txt'
        CCD_mask = '2009'
        border=0
    elif year>=2016:
        peaks_guess  = 'peaks_2016.txt'
        wave_guess = '_2012.txt'
        CCD_mask = ''
        border=1
    elif year>=2011:
        peaks_guess  = 'peaks.txt'
        wave_guess = '_2012.txt'
        CCD_mask = ''
        border=1
    elif year>=2010:
        peaks_guess  = 'peaks.txt'
        wave_guess = '.txt'
        CCD_mask = ''
        border=4
    elif year>=2008:
        peaks_guess  = 'peaks_early.txt'
        wave_guess = '_early.txt'
        CCD_mask = ''
        border=4
    elif year<=2004:
        peaks_guess  = 'peaks_2004.txt'
        wave_guess = '_2004.txt'
        CCD_mask = '2004'
        border=3
    else:
        peaks_guess  = 'peaks_vearly.txt'
        wave_guess = '_vearly.txt'
        CCD_mask = ''
        border=4
    if date>'2012-04-15' and setup=='blue':
      wave_guess = '_new.txt'
      
    for i in range(len(chips)):
        dir=os.listdir('.')
        for file_name in dir:
           if 'VIMOS_IFU_WAVE%03d'%(int(night)) in file_name and chips[i] in file_name:
                subtractBias_drp(file_name,  'WAVE_%s.sub.fits'%(chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')

    if setup=='orange':
        set='O'
    elif setup=='blue':
        set='B'
    elif setup=='red':
        set='R'
    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)
    if cpus>1:
        pool = Pool(cpus)
        result=[]
        for i in range(len(chips)):
            if int(int(master_trace)==1):
                trace_limit=trace_limits[i]
            else:
                trace_limit=''
            result.append(pool.apply_async(prepareCalib_drp, args=(night,chips[i],boundaries_x[i],boundaries_y[i],'%s/master_VIMOS_%s_%s'%(vimos_calib,chips[i], peaks_guess), '%s/ref_lines_ARC_VIMOS_HR%s_%s%s'%(vimos_calib, set, chips[i], wave_guess), wave_start, wave_end, wave_disp,10.0, setup,  border, trace_limit, fiberflat, fiberflat_wave, CCD_mask, 1)))
        pool.close()
        pool.join()
    else:
        for i in range(len(chips)):
            if int(master_trace==1):
                trace_limit=trace_limits[i]
            else:
                trace_limit=''
            prepareCalib_drp(night,chips[i],boundaries_x[i],boundaries_y[i],'%s/master_VIMOS_%s_%s'%(vimos_calib,chips[i], peaks_guess), '%s/ref_lines_ARC_VIMOS_HR%s_%s%s'%(vimos_calib, set,  chips[i], wave_guess), wave_start, wave_end, wave_disp, 8, setup, border, trace_limit, fiberflat, fiberflat_wave, CCD_mask, 'auto')


    if fiberflat==1 or fiberflat_wave==1:
        merge_flat = ''
        for i in range(len(chips)):
            merge_flat = merge_flat+'LAMP_%s.disp_cor.fits,'%(chips[i])
        mergeRSS_drp(merge_flat[:-1], 'LAMP.disp_cor.fits')
        if setup=='orange':
            if len(chips)>2:
                createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='800,1200',clip='0.3,1.7')
            else:  
                createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='400,800',clip='0.3,1.7')
        elif setup=='blue' and date<'2012-04-15':
            if len(chips)>2:
                createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='800,1200',clip='0.2,4.0')
            else:
                createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='400,800',clip='0.3,1.7')
        elif setup=='blue' and date>='2012-04-15':
            createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', clip='0.3,1.7')
        elif setup=='red':
            if len(chips)>2:
                createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='800,1200',clip='0.2,4.0')
            else:
                createFiberFlat_drp('LAMP.disp_cor.fits', 'fiberflat.fits', valid='400,800',clip='0.2,4.0')
        


def reduceObjectMR_drp(night, name_obj, object_nr, wave_start=4880, wave_end=8500, wave_disp=2.0, res_fwhm=7.5, A_V=0.15, fiberflat='1', flux_calib='1', telluric_cor='1',flexure_correct='1',straylight='1',correct_HVEL=False, parallel='auto' ):
    object_nr = int(object_nr)
    night = int(night)
    flexure_correct = int(flexure_correct)
    straylight = int(straylight)
    correctHVEL = bool(int(correct_HVEL))
    chips=['B.1', 'A.2', 'A.3', 'B.4']
    try:
        image = loadImage('VIMOS_IFU_OBS%03d_%04d_%s.fits.gz'%(night, object_nr, 'B.1'))
    except IOError: 
        image = loadImage('VIMOS_IFU_OBS%03d_%04d_%s.fits'%(night, object_nr, 'B.1'))
    date = image.getHdrValue('ESO OBS START').split('T')[0]
    year = int(date.split('-')[0])
    
    if year==2009:
      CCD_mask='2009'
    else:
      CCD_mask=''
    if year>=2013:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1360,3510', '1165,3215', '900,3100']
    elif year>=2007:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1340,3490', '1150,3300', '900,3150']
    else:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1283,3433', '1133,3283', '920,3170']

    if flexure_correct==1:
        sky_line_list='5577.34,6300.30,6863.97,7276.42,7750.65,8344.61,8885.85'
        flexure_order=2
    else:
        sky_line_list=''
        flexure_order=0


    fiberflat=int(fiberflat)
    flux_calib=int(flux_calib)
    telluric_cor=int(telluric_cor)


    for i in range(len(chips)):
        try:
            subtractBias_drp('VIMOS_IFU_OBS%03d_%04d_%s.fits.gz'%(night, object_nr, chips[i]),  '%s_%s.sub.fits'%(name_obj, chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')
        except IOError:
            subtractBias_drp('VIMOS_IFU_OBS%03d_%04d_%s.fits'%(night, object_nr, chips[i]),  '%s_%s.sub.fits'%(name_obj, chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')
        LACosmic_drp('%s_%s.sub.fits'%(name_obj, chips[i]),  '%s_%s.cosmic.fits'%(name_obj, chips[i]),  sigma_det='5.0',  flim='1.3',  iter='3',  error_box='1,20',  replace_box='1,20',  rdnoise='ESO DET OUT1 RON',  increase_radius='1', parallel=parallel)

    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)
    if cpus>1:
        pool = Pool(cpus)
        result=[]
        for i in range(len(chips)):
            result.append(pool.apply_async(prepareObject_drp, args=(name_obj, night,object_nr, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp ,res_fwhm, flexure_order, CCD_mask,correct_HVEL, bool(straylight), 1)))
        pool.close()
        pool.join()

    else:
        for i in range(len(chips)):
            prepareObject_drp(name_obj, night,object_nr, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp ,res_fwhm, flexure_order, CCD_mask, correct_HVEL,bool(straylight), 4)
    
    expandHdrKeys_drp('%s_B.1.disp_cor.fits'%(name_obj), 'CCD1') 
    expandHdrKeys_drp('%s_A.2.disp_cor.fits'%(name_obj), 'CCD2')
    expandHdrKeys_drp('%s_A.3.disp_cor.fits'%(name_obj), 'CCD3')
    expandHdrKeys_drp('%s_B.4.disp_cor.fits'%(name_obj), 'CCD4')
    mergeRSS_drp('%s_B.1.disp_cor.fits,%s_A.2.disp_cor.fits,%s_A.3.disp_cor.fits,%s_B.4.disp_cor.fits'%(name_obj, name_obj, name_obj, name_obj), '%s.disp_cor.fits'%(name_obj))
    if fiberflat==1:
        correctFiberFlat_drp('%s.disp_cor.fits'%(name_obj), '%s.flat.fits'%(name_obj), 'fiberflat.fits')

    if flux_calib==1:
        if fiberflat==1:
            fluxCalibration_drp('%s.flat.fits'%(name_obj), '%s.fobj.fits'%(name_obj), 'ratio.txt', 'CCD1 ESO TEL AIRM START', 'CCD1 EXPTIME', extinct_curve='Paranal', ref_units='1e-16', target_units='1e-16', norm_sb_fib='')
        #else:
            fluxCalibration_drp('%s.disp_cor.fits'%(name_obj), '%s.fobj.fits'%(name_obj), 'ratio.txt', 'CCD1 ESO TEL AIRM START', 'CCD1 EXPTIME', extinct_curve='Paranal', ref_units='1e-16', target_units='1e-16', norm_sb_fib='')
    if telluric_cor==1:
      correctTelluric_drp('%s.fobj.fits'%(name_obj), '%s.fobj.fits'%(name_obj), 'telluric_template.fits', airmass="CCD1 ESO TEL AIRM START")


def reduceObjectHR_drp(night, name_obj, object_nr, wave_start=4880, wave_end=8500, wave_disp=2.0, res_fwhm=7.5, A_V=0.15, fiberflat='1', flux_calib='1', telluric_cor='1',flexure_correct='0',straylight='1',setup='orange', correct_HVEL=False, except_chip='', parallel='auto' ):
    chips=['B.1', 'A.2', 'A.3', 'B.4']
    if except_chip!='':
        except_chips = except_chip.split(',')
        for i in range(len(except_chips)):
            chips= [x for x in chips if x != except_chips[i]]
    

    object_nr = int(object_nr)
    night = int(night)
    flexure_correct = int(flexure_correct)
    straylight = int(straylight)
    
    fiberflat=int(fiberflat)
    flux_calib=int(flux_calib)
    telluric_cor=int(telluric_cor)
    

    image = loadImage('VIMOS_IFU_OBS%03d_%04d_%s.fits.gz'%(night, object_nr, 'B.1'))
    date = image.getHdrValue('ESO OBS START').split('T')[0]
    year = int(date.split('-')[0])
    if year==2009:
        CCD_mask='2009'
    elif year<=2004:
        CCD_mask='2004'
    else:
        CCD_mask=''
    
    if setup=='blue' and date>='2012-04-15':
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1410,4096', '1410,4096', '1410,4096', '1410,4096']
    else:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1,4096', '1,4096', '1,4096', '1,4096']
      
    if setup=='orange':
        set='O'
        if flexure_correct==1:
            sky_line_list='5577.34,6300.30,6863.97,7276.42'
            flexure_order=2
        else:
            sky_line_list=''
            flexure_order=0
    elif setup=='blue' and date<'2012-04-15':
        set='B'
        if flexure_correct==1:
            sky_line_list='5577.34'
            flexure_order=0
        else:
            sky_line_list=''
            flexure_order=0
            
    elif setup=='blue' and date>='2012-04-15':
        set='B'
        sky_line_list=''
        flexure_order=0
        
    elif setup=='red':
        set='R'
        if flexure_correct==1:
            sky_line_list='6863.97,7276.42,7913.72,8344.61'
            flexure_order=2
        else:
            sky_line_list=''
            flexure_order=0

    for i in range(len(chips)):
        subtractBias_drp('VIMOS_IFU_OBS%03d_%04d_%s.fits.gz'%(night, object_nr, chips[i]),  '%s_%s.sub.fits'%(name_obj, chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')
        LACosmic_drp('%s_%s.sub.fits'%(name_obj, chips[i]),  '%s_%s.cosmic.fits'%(name_obj, chips[i]),  sigma_det='5.0',  flim='1.3',  iter='3',  error_box='1,20',  replace_box='1,20',  rdnoise='ESO DET OUT1 RON',  increase_radius='1', parallel='1')
    
    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    if cpus>1:
        pool = Pool(cpus)
        result=[]
        for i in range(len(chips)):
            result.append(pool.apply_async(prepareObject_drp, args=(name_obj, night,object_nr, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp ,res_fwhm, flexure_order, CCD_mask, correct_HVEL,bool(straylight),1)))
        pool.close()
        pool.join()

    else:
        for i in range(len(chips)):
            prepareObject_drp(name_obj, night,object_nr, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp ,res_fwhm, flexure_order, CCD_mask, correct_HVEL,bool(straylight),4)
    for i in range(len(chips)):
      expandHdrKeys_drp('%s_%s.disp_cor.fits'%(name_obj,chips[i]), 'CCD%s'%(chips[i].split('.')[1]),exclude="ESO TEL AIRM START,EXPTIME") 
      if flux_calib==1:
        fluxCalibration_drp('%s_%s.disp_cor.fits'%(name_obj,chips[i]), '%s_%s.fobj.fits'%(name_obj,chips[i]), 'ratio.txt', 'ESO TEL AIRM START', 'EXPTIME', extinct_curve='Paranal', ref_units='1e-16', target_units='1e-16', norm_sb_fib='')
      else:
          os.system('%s_%s.disp_cor.fits %s_%s.fobj.fits'%(name_obj,chips[i],name_obj,chips[i]))
    if telluric_cor==1:
      correctTelluric_drp('%s.fobj.fits'%(name_obj), '%s.fobj.fits'%(name_obj), 'telluric_template.fits', airmass="ESO TEL AIRM START")
    merge_obj = ''
    for i in range(len(chips)):
        merge_obj = merge_obj+'%s_%s.fobj.fits,'%(name_obj,chips[i])
    mergeRSS_drp(merge_obj[:-1], '%s.fobj.fits'%(name_obj))

    if fiberflat==1:
        correctFiberFlat_drp('%s.fobj.fits'%(name_obj), '%s.flat.fits'%(name_obj), 'fiberflat.fits')

    
    
    
    
def reduceStdMR_drp(night, std_nr, wave_start='4880', wave_end='9300', wave_disp='2.0', res_fwhm='0.0', ref_star='', A_V=0.15, mask_wave='6850,6960,7480,7800', mask_telluric='', straylight='1', parallel='auto'):
    chips=['B.1', 'A.2', 'A.3', 'B.4']
    if year>=2013:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1360,3510', '1165,3215', '900,3100']
    else:
        boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
        boundaries_y=['1300,3450', '1340,3490', '1150,3300', '900,3150']
        
    sky_line_list='5577.34,6300.30,6863.97,7276.42,7750.65,8344.61,8885.85'
    night = int(night)
    std_nr = int(std_nr)
    straylight = int(straylight)
    
    image = loadImage('VIMOS_IFU_STD%03d_%04d_%s.fits.gz'%(night, std_nr, 'B.1'))
    date = image.getHdrValue('ESO OBS START').split('T')[0]
    year = int(date.split('-')[0])
    
    if year==2009:
      CCD_mask='2009'
    else:
      CCD_mask=''

    for i in range(len(chips)):
        subtractBias_drp('VIMOS_IFU_STD%03d_%04d_%s.fits.gz'%(night, std_nr+i, chips[i]),  '%s_%s.sub.fits'%('STD'+str(i+1), chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')
        LACosmic_drp('%s_%s.sub.fits'%('STD'+str(i+1), chips[i]),  '%s_%s.cosmic.fits'%('STD'+str(i+1), chips[i]),  sigma_det='5.0',  flim='1.3',  iter='3',  error_box='1,20',  replace_box='1,20',  rdnoise='ESO DET OUT1 RON',  increase_radius='1', parallel=parallel)

    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    if cpus>1:
        pool = Pool(cpus)
        result=[]
        for i in range(len(chips)):
            result.append(pool.apply_async(prepareObject_drp, args=('STD'+str(i+1), night,std_nr+i, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp , res_fwhm, 2, CCD_mask, False, bool(straylight), 1)))
        pool.close()
        pool.join()
    else:
        for i in range(len(chips)):
            prepareObject_drp('STD'+str(i+1), night,std_nr+i, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp , res_fwhm, 2, CCD_mask, False, bool(straylight), 4)
            
    mergeRSS_drp('STD1_B.1.disp_cor.fits,STD2_A.2.disp_cor.fits,STD3_A.3.disp_cor.fits,STD4_B.4.disp_cor.fits', 'STD.disp_cor.fits')
    correctFiberFlat_drp('STD.disp_cor.fits', 'STD.flat.fits', 'fiberflat.fits')

    splitFibers_drp('STD.flat.fits', 'STD1.flat.fits,STD2.flat.fits,STD3.flat.fits,STD4.flat.fits', 'QD1,QD2,QD3,QD4')
    std_ratios=[]
    for i in range(len(chips)):
        constructSkySpec_drp('STD%d.flat.fits'%(i+1), 'STD%d.sky_spec.fits'%(i+1), clip_sigma=0.0,  nsky=150)
        subtractSkySpec_drp('STD%d.flat.fits'%(i+1),  'STD%d.sobj.fits'%(i+1),  'STD%d.sky_spec.fits'%(i+1))
        copyHdrKey_drp('VIMOS_IFU_STD%03d_%04d_%s.fits.gz'%(night, std_nr+i, chips[i]),'STD%d.sobj.fits'%(i+1),  'ESO TEL AIRM START')

        if ref_star!='':
            createSensFunction_drp('STD%d.sobj.fits'%(i+1),  'ratio_%d.txt'%(i+1),  ref_star, airmass='ESO TEL AIRM START',  exptime='EXPTIME',  coadd=200,  extinct_curve='Paranal', out_star='star_%d.txt'%(i+1),  mask_wave=mask_wave, mask_telluric=mask_telluric, smooth_poly=-12)
            std_ratios.append(Spectrum1D())
            std_ratios[i].loadTxtData('ratio_%s.txt'%(chips[i].split('.')[1]))
            if mask_telluric!='':
                os.system('cp telluric_spec.fits telluric_spec_%s.fits'%(chips[i].split('.')[1]))
                std_telluric.append(Spectrum1D())
                std_telluric[i].loadFitsData('telluric_spec_%s.fits'%(chips[i].split('.')[1]))
    
    if ref_star!='':
        for i in range(len(std_ratios)):
            if i==0:
                mean_std_ratio = std_ratios[0]
            else:
                mean_std_ratio += std_ratios[i]
    
            if mask_telluric!='':
                if i==0:
                    mean_telluric = std_telluric[0]
                else:
                    mean_telluric += std_telluric[i]
    
        mean_std_ratio = mean_std_ratio/len(std_ratios)
        mean_std_ratio.writeTxtData('ratio.txt')
        if mask_telluric!='':
            mean_telluric = mean_telluric/len(std_telluric)
            mean_telluric.writeFitsData('telluric.fits')
        
def reduceStdHR_drp(night, std_nr, wave_start='4880', wave_end='9300', wave_disp='2.0', res_fwhm='0.0', ref_star='', setup=None, A_V=0.15, mask_wave='', mask_telluric='', smooth_poly=-12, except_chip='', straylight='1', parallel='auto'):

    chips=['B.1', 'A.2', 'A.3', 'B.4']
    
    if except_chip!='':
        except_chips = except_chip.split(',')
        for i in range(len(except_chips)):
            chips= [x for x in chips if x != except_chips[i]]
    sky_line_list=''
    night = int(night)
    std_nr = int(std_nr)
    straylight = int(straylight)
    
    image = loadImage('VIMOS_IFU_STD%03d_%04d_%s.fits.gz'%(night, std_nr, 'B.1'))
    date = image.getHdrValue('ESO OBS START').split('T')[0]
    
    year = int(date.split('-')[0])
    
    if date>'2012-04-15' and setup=='blue':
      boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
      boundaries_y=['1410,4096', '1410,4096', '1410,4096', '1410,4096']
    else:
      boundaries_x=['51,2098', '51,2098', '51,2098', '51,2098']
      boundaries_y=['1,4096', '1,4096', '1,4096', '1,4096']
    if year==2009:
      CCD_mask='2009'
    elif year<=2004:
      CCD_mask='2004'
    else:
      CCD_mask=''


    for i in range(len(chips)):
        subtractBias_drp('VIMOS_IFU_STD%03d_%04d_%s.fits.gz'%(night, std_nr+int(chips[i].split('.')[1])-1, chips[i]),  '%s_%s.sub.fits'%('STD'+str(i+1), chips[i]), 'BIAS_%s.fits'%(chips[i]), boundary_x=boundaries_x[i],  boundary_y=boundaries_y[i],  gain='ESO DET OUT1 CONAD',  rdnoise='ESO DET OUT1 RON')
        LACosmic_drp('%s_%s.sub.fits'%('STD'+str(i+1), chips[i]),  '%s_%s.cosmic.fits'%('STD'+str(i+1), chips[i]),  sigma_det='5.0',  flim='1.3',  iter='3',  error_box='1,20',  replace_box='1,20',  rdnoise='ESO DET OUT1 RON',  increase_radius='1', parallel='1')

    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    if cpus>1:
        pool = Pool(cpus)
        result=[]
        for i in range(len(chips)):
            result.append(pool.apply_async(prepareObject_drp, args=('STD'+str(i+1), night,std_nr+i, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp ,res_fwhm, 2, CCD_mask, False, bool(straylight), 1)))
        pool.close()
        pool.join()
    else:
        for i in range(len(chips)):
            prepareObject_drp('STD'+str(i+1), night,std_nr+i, chips[i],boundaries_x[i],boundaries_y[i],sky_line_list, wave_start, wave_end, wave_disp , res_fwhm, 2, CCD_mask, False, bool(straylight),4)

    merge_obj = ''
    for i in range(len(chips)):
      merge_obj = merge_obj+'STD%d_%s.disp_cor.fits,'%(i+1,chips[i])    
    mergeRSS_drp(merge_obj[:-1], 'STD.disp_cor.fits')
    correctFiberFlat_drp('STD.disp_cor.fits', 'STD.flat.fits', 'fiberflat.fits')
    
    QDs = ''
    files = ''
    for i in range(len(chips)):
      QDs = QDs + 'QD%s,'%(chips[i].split('.')[1])
      files = files + 'STD%s.flat.fits,'%(chips[i].split('.')[1])
    splitFibers_drp('STD.flat.fits', files[:-1], QDs[:-1])
    std_ratios=[]
    std_telluric=[]
    for i in range(len(chips)):
        constructSkySpec_drp('STD%s.flat.fits'%(chips[i].split('.')[1]), 'STD%s.sky_spec.fits'%(chips[i].split('.')[1]), clip_sigma=0.0,  nsky=70, non_neg=0)
        subtractSkySpec_drp('STD%s.flat.fits'%(chips[i].split('.')[1]),  'STD%s.sobj.fits'%(chips[i].split('.')[1]),  'STD%s.sky_spec.fits'%(chips[i].split('.')[1]))
        copyHdrKey_drp('VIMOS_IFU_STD%03d_%04d_%s.fits.gz'%(night, std_nr+i, chips[i]),'STD%s.sobj.fits'%(chips[i].split('.')[1]),  'ESO TEL AIRM START')
        
        if ref_star!='':
            createSensFunction_drp('STD%s.sobj.fits'%(chips[i].split('.')[1]),  'ratio_%s.txt'%(chips[i].split('.')[1]),  ref_star, airmass='ESO TEL AIRM START',  exptime='EXPTIME',  coadd=200,  extinct_curve='Paranal', out_star='star_%s.txt'%(chips[i].split('.')[1]),  mask_wave=mask_wave, mask_telluric=mask_telluric, smooth_poly=smooth_poly)
            std_ratios.append(Spectrum1D())
            std_ratios[i].loadTxtData('ratio_%s.txt'%(chips[i].split('.')[1]))
            if mask_telluric!='':
                os.system('cp telluric_spec.fits telluric_spec_%s.fits'%(chips[i].split('.')[1]))
                std_telluric.append(Spectrum1D())
                std_telluric[i].loadFitsData('telluric_spec_%s.fits'%(chips[i].split('.')[1]))
    
    if ref_star!='':
        for i in range(len(std_ratios)):
            if i==0:
                mean_std_ratio = std_ratios[0]
            else:
                mean_std_ratio += std_ratios[i]
    
            if mask_telluric!='':
                if i==0:
                    mean_telluric = std_telluric[0]
                else:
                    mean_telluric += std_telluric[i]
    
        mean_std_ratio = mean_std_ratio/len(std_ratios)
        mean_std_ratio.writeTxtData('ratio.txt')
        if mask_telluric!='':
            mean_telluric = mean_telluric/len(std_telluric)
            mean_telluric.writeFitsData('telluric.fits')
        
def subtractSkyField_drp(object_in,  object_out, sky_field, factor,  scale_region='',scale_ind=1, clip_sigma=0.0, nsky=200):

    
    splitFibers_drp(object_in, 'obj_QD1.fits,obj_QD2.fits,obj_QD3.fits,obj_QD4.fits', 'QD1,QD2,QD3,QD4')
    splitFibers_drp(sky_field, 'sky_QD1.fits,sky_QD2.fits,sky_QD3.fits,sky_QD4.fits', 'QD1,QD2,QD3,QD4')
    for i in range(4):
        constructSkySpec_drp('sky_QD%d.fits'%(i+1), 'sky_spec_QD%d.fits'%(i+1),  clip_sigma=clip_sigma, filter=vimos_calib+'R_Johnson.txt,0,1',nsky=nsky)
        subtractSkySpec_drp('obj_QD%d.fits'%(i+1),'sobj_QD%d.fits'%(i+1),'sky_spec_QD%d.fits'%(i+1), factor=factor, scale_ind=scale_ind,scale_region=scale_region)
        copyHdrKey_drp('sky_QD%d.fits'%(i+1),'sobj_QD%d.fits'%(i+1), "hierarch PIPE NSKY FIB")
        copyHdrKey_drp('sky_QD%d.fits'%(i+1),'sobj_QD%d.fits'%(i+1), "hierarch PIPE SKY MEAN")
        copyHdrKey_drp('sky_QD%d.fits'%(i+1),'sobj_QD%d.fits'%(i+1), "hierarch PIPE SKY MIN")
        copyHdrKey_drp('sky_QD%d.fits'%(i+1),'sobj_QD%d.fits'%(i+1), "hierarch PIPE SKY MAX")
        copyHdrKey_drp('sky_QD%d.fits'%(i+1),'sobj_QD%d.fits'%(i+1), "hierarch PIPE SKY RMS")
        expandHdrKeys_drp('sobj_QD%d.fits'%(i+1),'CCD%d'%(i+1),'PIPE NSKY FIB,PIPE SKY MEAN,PIPE SKY MIN,PIPE SKY MAX,PIPE SKY RMS,PIPE SKY SCALE')
    mergeRSS_drp('sobj_QD1.fits,sobj_QD2.fits,sobj_QD3.fits,sobj_QD4.fits', object_out)
    os.system('rm *QD?.fits')
