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

import numpy as np
from astropy.io import fits as fits
from astropy.wcs import WCS
import yaml
from yaml.loader import SafeLoader
import numpy.ma as ma


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
GDIR = OUTDIR

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

#
# We reduce every channel of every spectrograph
#

    
#
# We glue all the data
#
# In[28]:


os.makedirs(OUTDIR, exist_ok=True)


CAMERAS = [f"{channel}{spec}" for channel, spec in product(("brz"), (1,2,3))]
red_paths = sorted([os.path.join(OUTDIR, sci_name) for sci_name in os.listdir(OUTDIR)])
print(red_paths)


fibermap=CALDIR+'/LVM_SAIT_fibermap.yaml'
with open(fibermap) as f:
    data = yaml.load(f, Loader=SafeLoader)

#print(data['schema'][0])
colnames=[]
coltype=[]
for schema in data['schema']:
    colnames.append(schema['name'])
    coltype_now=schema['dtype']
    if (coltype_now=='finblock'):
        coltype_now='str'
    if (schema['name']=='finblock'):
        coltype_now='str'
    if (schema['name']=='ringnum'):
        coltype_now='float'
#    print(coltype_now)


    coltype.append(coltype_now)
    #    print(schema['name'],schema['dtype'])
    
#print(data['schema'][0])
#print(len(data['fibers'])/3)

tab=Table(np.array(data['fibers']),names=colnames,dtype=coltype)


# In[50]:


tab1 = tab[tab['spectrographid']==1]
tab2 = tab[tab['spectrographid']==2]
tab3 = tab[tab['spectrographid']==3]
m1_sky = (tab1['targettype']=='SKY')
m2_sky = (tab2['targettype']=='SKY')
m3_sky = (tab3['targettype']=='SKY')


# In[51]:


sp_files=[]
channels=['b','r','z']
for file in red_paths:
    for chn in channels:
        if ((file.find(f'-{chn}1-')>0) and (file.find('.fits')>0) and (file.find('lvm-wobject')>0)):
            file1=file
            file2=file.replace(f'-{chn}1-',f'-{chn}2-')
            file3=file.replace(f'-{chn}1-',f'-{chn}3-')
            print(file1,file2,file3)
            outfile=os.path.basename(file1)
            in_chan=f'-{chn}1-'
            out_chan=f'-{chn}-'
            print(in_chan,out_chan)
            outfile=outfile.replace(in_chan,out_chan)
            outfile=os.path.join(GDIR, outfile)
            print(file1,outfile) 
            hdu1=fits.open(file1)
            (ny1,nx1)=hdu1[0].data.shape
            hdu2=fits.open(file2)
            (ny2,nx2)=hdu2[0].data.shape
            hdu3=fits.open(file3)
            (ny3,nx3)=hdu3[0].data.shape
            data = np.zeros((ny1+ny2+ny3,nx1),dtype=np.float32)
            inst = np.zeros((ny1+ny2+ny3,nx1),dtype=np.float32)
            error = np.zeros((ny1+ny2+ny3,nx1),dtype=np.float32)
            badpix = np.zeros((ny1+ny2+ny3,nx1),dtype=np.uint8)
            ny1l=ny1-1
            ny2l=ny1+ny2
            ny3l=ny1+ny2+ny3
            ny30=ny1+ny2
            sky1=hdu1[0].data[m1_sky]
            sky2=hdu2[0].data[m2_sky]
            sky3=hdu3[0].data[m3_sky]
            m_sky1=np.median(sky1,axis=0)
            m_sky2=np.median(sky2,axis=0)
            m_sky3=np.median(sky3,axis=0)
            data[0:ny1,:]=hdu1[0].data-m_sky1
            data[ny1:ny2l,:]=hdu2[0].data-m_sky2
            data[ny30:ny3l,:]=hdu3[0].data-m_sky3
            inst[0:ny1,:]=hdu1[1].data
            inst[ny1:ny2l,:]=hdu2[1].data
            inst[ny30:ny3l,:]=hdu3[1].data
            error[0:ny1,:]=hdu1[2].data
            error[ny1:ny2l,:]=hdu2[2].data
            error[ny30:ny3l,:]=hdu3[2].data        
            badpix[0:ny1,:]=hdu1[3].data
            badpix[ny1:ny2l,:]=hdu2[3].data
            badpix[ny30:ny3l,:]=hdu3[3].data        
            hdr=hdu1[0].header
            hdr['CD1_1']=float(hdr['CRVAL1'])
            hdr['CD1_2']=float(hdr['CDELT1'])
            hdr['CD2_1']=1.0
            hdr['CD2_2']=1.0
            primhdu = fits.PrimaryHDU(data=data,header=hdr)
            hdulist=fits.HDUList([primhdu])
            hdulist.append(fits.ImageHDU(data=error,name='ERROR'))
            hdulist.append(fits.ImageHDU(data=badpix,name='BADPIX'))
            hdulist.append(fits.ImageHDU(data=inst,name='INSTFWHM'))
            hdulist.writeto(outfile,overwrite=True)
            sp_files.append(outfile)
        


# In[52]:


for sp_file in sp_files:
    if (sp_file.find('-b-')>1):
        b_file=sp_file
        r_file=sp_file.replace('-b-','-r-')
        z_file=sp_file.replace('-b-','-z-')
        out_file=sp_file.replace('-b-','')
        hdu_b=fits.open(b_file)
        hdu_r=fits.open(r_file)
        hdu_z=fits.open(z_file)
        (ny_b,nx_b)=hdu_b[0].data.shape
        (ny_r,nx_r)=hdu_r[0].data.shape
        (ny_z,nx_z)=hdu_z[0].data.shape
        hdr=hdu_b[0].header
       # data = np.zeros((ny_b,12401),dtype=np.float32)
       # inst = np.zeros((ny_b,12401),dtype=np.float32)
       # error = np.zeros((ny_b,12401),dtype=np.float32)
       # badpix = np.zeros((ny_b,12401),dtype=np.uint8)
        data_b = np.zeros((3,ny_b,12401),dtype=np.float32)
        inst_b = np.zeros((3,ny_b,12401),dtype=np.float32)
        error_b = np.zeros((3,ny_b,12401),dtype=np.float32)
        badpix_b = np.zeros((3,ny_b,12401),dtype=np.uint8)
        
        data_b[0,:,0:nx_b]=hdu_b[0].data
        error_b[0,:,0:nx_b]=hdu_b[2].data
        inst_b[0,:,0:nx_b]=hdu_b[1].data
        badpix_b[0,:,0:nx_b]=hdu_b[3].data
        n0_r = int((hdu_r[0].header['CRVAL1']-hdu_b[0].header['CRVAL1'])/hdu_b[0].header['CDELT1'])
        n0_z = int((hdu_z[0].header['CRVAL1']-hdu_b[0].header['CRVAL1'])/hdu_b[0].header['CDELT1'])
        n1_r=n0_r+nx_r
        n1_z=n0_z+nx_z
        data_b[1,:,n0_r:n1_r]=hdu_r[0].data
        error_b[1,:,n0_r:n1_r]=hdu_r[2].data
        inst_b[1,:,n0_r:n1_r]=hdu_r[1].data
        badpix_b[1,:,n0_r:n1_r]=hdu_r[3].data
        data_b[2,:,n0_z:n1_z]=hdu_z[0].data
        error_b[2,:,n0_z:n1_z]=hdu_z[2].data
        inst_b[2,:,n0_z:n1_z]=hdu_z[1].data
        badpix_b[2,:,n0_z:n1_z]=hdu_z[3].data
        
        
        
        print(n0_r,n0_z)
        

        mask_b = (data_b==0)
        mdata_b=ma.array(data_b,mask = mask_b)
        data=ma.median(mdata_b,axis=0)
        error=np.max(error_b,axis=0)
        badpix=np.max(badpix_b,axis=0)
        inst=np.max(inst_b,axis=0)
        print('# INFOR: shape ',data.shape)
        
        primhdu = fits.PrimaryHDU(data=data.data,header=hdr)
        hdulist=fits.HDUList([primhdu])
        hdulist.append(fits.ImageHDU(data=error,name='ERROR'))
        hdulist.append(fits.ImageHDU(data=badpix,name='BADPIX'))
        hdulist.append(fits.ImageHDU(data=inst,name='INSTFWHM'))
        print(out_file)
        hdulist.writeto(out_file,overwrite=True)
        #break


# In[ ]:




