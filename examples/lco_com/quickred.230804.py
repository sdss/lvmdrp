#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
from itertools import product
from astropy.io import fits
from astropy.table import Table
from lvmdrp.utils import metadata as md
from lvmdrp.functions import run_drp as drp
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.core.constants import SPEC_CHANNELS


nargs=len(sys.argv)
if (nargs==2):
    lvm_night=str(sys.argv[1])
else:
    print('USE: quickred.py LVM_NIGHT(e.g., 60154)')
    quit()





# # Instructions
# 
# This notebook implementes a quick reduction of the LVM data. Main asumptions are:
# 
# 1. There is a `data_calib` directory containing master calibration frames for:
#     - pixel mask (`lvm-mpixmask-{CAMERA}.fits`)
#     - bias (`lvm-mbias-{CAMERA}.fits`)
#     - dark (`lvm-mdark-{CAMERA}.fits`)
#     - traces (`traces_{CAMERA}_p4.fits`)
#     - wavelengths (`lvm-mwave_neon-{CAMERA}.fits`)
#     - LSF (`lvm-mlsf_neon-{CAMERA}.fits`)
# 
# 2. There is a `data_science` directory containing the **raw** target science exposures (`sdR-s-{CAMERA}-{EXPNUM:>08d}.fits.gz`)
# 
# 3. Data products from this reduction can be stored in a directory `data_products` (this directory will be created by this notebook if it does not exists).
# 
# This reduction will push raw frames from preprocessing down to wavelength calibration.

# In[ ]:


# define paths
LVMCOREDIR = os.environ.get('LVMCORE_DIR')
print('LVM CORE=',LVMCOREDIR)

LVM_ROOT='/mnt/NASfilemon/LVM/'
LVM_DATA_S=LVM_ROOT+'/data/lvm/lco'
#LVM_DATA_S = os.environ.get('LVM_DATA_S')
print('LVM DATA S=',LVM_DATA_S)
LVM_SPECTRO_REDUX=LVM_ROOT+'/spectro/redux/'
#LVM_SPECTRO_REDUX= os.environ.get('LVM_SPECTRO_REDUX')
print('LVM SPECTRO REDUX=',LVM_SPECTRO_REDUX)
VER='1.0.1104'
print('VER=',VER)


CALDIR = LVMCOREDIR+'/masters/'#"data_calib/"
SCIDIR = LVM_DATA_S+'/'+lvm_night
OUTDIR = LVM_SPECTRO_REDUX+'/'+VER+'/'+lvm_night
try:
    os.mkdir(OUTDIR)
except:
    print(OUTDIR,'  already exisits')

#quit()
#OUTDIR = "data_products/"

#CALDIR = "data_calib/"
#SCIDIR = "data_science/"
#OUTDIR = "data_products/"

os.makedirs(OUTDIR, exist_ok=True)

# define cameras
CAMERAS = [f"{channel}{spec}" for channel, spec in product(("brz"), (1,2,3))]


# In[ ]:


# target science directory
sci_paths_in = sorted([os.path.join(SCIDIR, sci_name) for sci_name in os.listdir(SCIDIR)])
sci_paths=[]


for spc in sci_paths_in:
    if (spc.find('.fits')>1):
        sci_paths.append(spc)
        print('.',end='')
        
print('#INFO: DONE')
#print(sci_paths)
#quit()

# In[ ]:


for isci, sci_path in enumerate(sci_paths):
    # get basic metadata
    sci_header = fits.getheader(sci_path, ext=0)
    sci_camera = sci_header["CCD"]
    sci_exposure = sci_header["EXPOSURE"]
    sci_exptime = sci_header["EXPTIME"]
    sci_type = sci_header["IMAGETYP"]
    print(sci_path,sci_exptime,sci_type)

    if ((sci_type=='object') and (sci_exptime>150)):
    

        mpixmask_path = os.path.join(CALDIR, f"lvm-mpixmask-{sci_camera}.fits")
        mbias_path = os.path.join(CALDIR, f"lvm-mbias-{sci_camera}.fits")
        mdark_path = os.path.join(CALDIR, f"lvm-mdark-{sci_camera}.fits")
        mtrace_path = os.path.join(CALDIR, f"traces_{sci_camera}_p4.fits")
        mwave_path = os.path.join(CALDIR, f"lvm-mwave_neon-{sci_camera}.fits")
        mlsf_path = os.path.join(CALDIR, f"lvm-mlsf_neon-{sci_camera}.fits")
    
        # preprocess frame
        psci_path = os.path.join(OUTDIR, f"lvm-pobject-{sci_camera}-{sci_exposure:>08d}.fits")
        image_tasks.preproc_raw_frame(in_image=sci_path, out_image=psci_path, in_mask=mpixmask_path)
    
        # detrend frame
        dsci_path = os.path.join(OUTDIR, f"lvm-dobject-{sci_camera}-{sci_exposure:>08d}.fits")
        image_tasks.detrend_frame(in_image=psci_path, out_image=dsci_path, in_bias=mbias_path, in_dark=mdark_path, in_slitmap=Table(drp.fibermap.data))
    
        # extract 1d spectra
        xsci_path = os.path.join(OUTDIR, f"lvm-xobject-{sci_camera}-{sci_exposure:>08d}.fits")
        image_tasks.extract_spectra(in_image=dsci_path, out_rss=xsci_path, in_trace=mtrace_path, method="aperture", aperture=3)
    
        # wavelength calibrate & resample
        iwave, fwave = SPEC_CHANNELS[sci_camera[0]]
        wsci_path = os.path.join(OUTDIR, f"lvm-wobject-{sci_camera}-{sci_exposure:>08d}.fits")
        rss_tasks.create_pixel_table(in_rss=xsci_path, out_rss=wsci_path, arc_wave=mwave_path, arc_fwhm=mlsf_path)
        rss_tasks.resample_wavelength(in_rss=wsci_path, out_rss=wsci_path, method="linear", disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0)

        print(f'# redu done:{sci_path}')
    
    
