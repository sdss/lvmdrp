
#################################################################
#imports

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import get_body, solar_system_ephemeris, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.table import Table
import matplotlib.backends.backend_pdf

#Things that should not change and dir

# Define directory paths (for local version, not DRP) 
LVMDATA_DIR="/Users/amjones/repos/lvm_ksl/2024Oct15_v1.1.0/"

# Measure line fluxes by direct integration of the spectra and add to table
maplist=['[OI]5577', '[OI]6300', '[OH]6865', 'Bcont(4195, 4220)', 'Rcont(6420, 6440)', 'Zcont(9130, 9145)']
medlist=['[OI]5577', '[OI]6300', '[OH]6865', 'Bcont', 'Rcont', 'Zcont', '[Na]5891', 'O2', 'AirglowCont']

# first is Hb just to test, all others are OI lines obscured by bright sky lines in our data
lrangelist=[(5574, 5580), (6298,6304), (6862,6868), (4195, 4220), (6420, 6440), (9130, 9145), (5888, 5898), (8630, 8670), (5420, 5440)]

# Continua range for each lrangelist, with -999 when don't subtract continum
crangelist=[(5571, 5583), (6295, 6307), (6859, 6871), (-999, -999), (-999, -999), (-999, -999), (5885, 5901), (-999, -999), (-999, -999)]

def sumlineflux(wave, flux, sky, ivar, mask, slitmap, maplist, lrangelist, crangelist, type):
    
    flux, sky, ivar, sel = good_masked_fibers(slitmap, flux, sky, ivar, mask, type)
    error=np.sqrt(1/ivar)

    # Reading ra and dec from slitmap
    ra = slitmap['ra']
    dec = slitmap['dec']

    ra = ra[sel]
    dec = dec[sel]

    auxlinetab=Table()

    auxlinetab['ra'] = ra
    auxlinetab['dec'] = dec

    for j in range(len(maplist)):

        lsel, csel = line_windows(lrangelist[j], crangelist[j], wave)
        
        # selecting line range
        fluxsel=flux[:,lsel]  
        skysel=sky[:,lsel]

        # subtracting continuum and storing continuum subtracted flux and sky
        if crangelist[j][0] == -999:
            cfluxsel=fluxsel
            cskysel=skysel
        else:
            cfluxsel=fluxsel-np.tile(np.nanmedian(flux[:,csel], axis=1), (np.shape(fluxsel)[1],1)).transpose() 
            cskysel=skysel-np.tile(np.nanmedian(sky[:,csel], axis=1), (np.shape(skysel)[1],1)).transpose() 

        # Non-parametric flux, error and sky
        lflux=np.nansum(cfluxsel, axis=1)
        lerror=np.sqrt(np.nansum((error[:,lsel])**2, axis=1))
        lsky=np.nansum(cskysel, axis=1)
        
        auxlinetab['flux_'+maplist[j]]=lflux
        auxlinetab['error_'+maplist[j]]=lerror
        auxlinetab['sky_'+maplist[j]]=lsky

    return auxlinetab

def create_sky_table(wave, flux, sky, ivar, mask, slitmap, header, maplist, medlist, lrangelist, crangelist, outfile):

    linetab_sci=Table()
    linetab_skye=Table()
    linetab_skyw=Table()
    medtab=Table()
    metricstab=Table()
    sky_infotab=Table()
        
    # getting the map info
    linetab_sci=sumlineflux(wave, flux, sky, ivar, mask, slitmap, maplist, lrangelist, crangelist, 'sci')
    #table_sci = Table(linetab_sci)
    linetab_skye=sumlineflux(wave, flux, sky, ivar, mask, slitmap, maplist, lrangelist, crangelist, 'skye')
    #table_skye = Table(linetab_skye)
    linetab_skyw=sumlineflux(wave, flux, sky, ivar, mask, slitmap, maplist, lrangelist, crangelist, 'skyw')
    #table_skyw = Table(linetab_skyw)

    # getting the med plot and metrics info
    wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, stats_list, wvl_list, wvc_list = get_all_stats(wave, flux, sky, ivar, mask, slitmap, medlist)
    medtab['wave']=wave
    medtab['med_sky']=med_sky
    medtab['med_flux']=med_flux
    medtab['med_ivar']=med_ivar
    medtab['med_skye']=med_skye
    medtab['med_skyw']=med_skyw

    metricstab['sky_metrics']=stats_list
    metricstab['regions']=medlist

    sky_info = create_overview(header)
    for key,val in sky_info.items():
        sky_infotab[key]=val
    
    # needed if want to save info as a fits file
    # hdu_list = fits.HDUList([
    #     fits.PrimaryHDU(),
    #     fits.table_to_hdu(table_sci),
    #     fits.table_to_hdu(table_skye), 
    #     fits.table_to_hdu(table_skyw),
    #     fits.table_to_hdu(medtab),
    #     fits.table_to_hdu(metricstab),
    #     fits.table_to_hdu(sky_infotab)
    # ])

    # hdu_list[1].name = 'sci_regions_fiber'
    # hdu_list[2].name = 'skye_regions_fiber'
    # hdu_list[3].name = 'skyw_regions_fiber'
    # hdu_list[4].name = 'median_spectra'
    # hdu_list[5].name = 'sky_metrics'
    # hdu_list[6].name = 'sky_info'
    
    # hdu_list.writeto(f'{outfile}.fits', overwrite=True)

    #returning data for creating the plots
    return linetab_sci, linetab_skye, linetab_skyw, wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, stats_list, wvl_list, wvc_list, sky_info

def plotmap_row(data_sci, data_skye, data_skyw, line, row, fig, gs):
    
    vmine=np.nanpercentile(data_sci['error_'+line], 10)
    vmaxe=np.nanpercentile(data_sci['error_'+line], 90)
    vmins=np.min([np.nanpercentile(data_skye['flux_'+line]+data_skye['sky_'+line], 10), np.nanpercentile(data_skyw['flux_'+line]+data_skyw['sky_'+line], 10)])
    vmaxs=np.max([np.nanpercentile(data_skye['flux_'+line]+data_skye['sky_'+line], 90), np.nanpercentile(data_skyw['flux_'+line]+data_skyw['sky_'+line], 90)])
    vmin=-1.0*np.nanpercentile(abs(data_sci['flux_'+line]), 90)
    vmax=-1.0*vmin

    ax = fig.add_subplot(gs[row:row+2, 0:2])
    sc=ax.scatter(data_sci['ra'], data_sci['dec'], c=data_sci['flux_'+line], s=3, cmap ='RdBu_r', vmin = vmin, vmax = vmax)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(line+' flux \n (sky-subtracted)')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.invert_xaxis()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    cbar=fig.colorbar(sc, ax=ax, fraction=0.09)
    ax.set_facecolor('black')

    ax1 = fig.add_subplot(gs[row:row+2, 2:4])
    sc1=ax1.scatter(data_sci['ra'], data_sci['dec'], c=data_sci['error_'+line], s=3, cmap ='viridis', vmin = vmine, vmax = vmaxe)
    #ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(line+' error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.invert_xaxis()
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    cbar=fig.colorbar(sc1, ax=ax1, fraction=0.09)
    ax1.set_facecolor('black')

    ax2 = fig.add_subplot(gs[row, 4])
    sc2=ax2.scatter(data_skye['ra'], data_skye['dec'], c=data_skye['flux_'+line]+data_skye['sky_'+line], s=40, cmap ='viridis', vmin = vmins, vmax = vmaxs)
    #ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(' SkyE \n sky flux', loc='left')
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    cbar=fig.colorbar(sc2, ax=ax2, fraction=0.045)
    cbar.remove()
    ax2.set_facecolor('black')

    ax3 = fig.add_subplot(gs[row+1, 4])
    sc3=ax3.scatter(data_skye['ra'], data_skye['dec'], c=data_skye['error_'+line], s=40, cmap ='viridis', vmin = vmine, vmax = vmaxe)
    #ax3.set_aspect('equal', adjustable='box')
    ax3.set_title(r'SkyE error', loc='left')
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')
    ax3.invert_xaxis()    
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    cbar=fig.colorbar(sc3, ax=ax3, fraction=0.045)
    cbar.remove()
    ax3.set_facecolor('black')

    ax4 = fig.add_subplot(gs[row, 5])
    sc4=ax4.scatter(data_skyw['ra'], data_skyw['dec'], c=data_skyw['flux_'+line]+data_skyw['sky_'+line], s=40, cmap ='viridis', vmin = vmins, vmax = vmaxs)
    #ax4.set_aspect('equal')#, adjustable='box')
    ax4.set_title(' SkyW \n sky flux', loc='left')
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)
    cbar=fig.colorbar(sc4, ax=ax4, fraction=0.09)
    ax4.set_facecolor('black')

    ax5 = fig.add_subplot(gs[row+1, 5])
    sc5=ax5.scatter(data_skyw['ra'], data_skyw['dec'], c=data_skyw['error_'+line], s=40, cmap ='viridis', vmin = vmine, vmax = vmaxe)
    #ax5.set_aspect('equal', adjustable='box')
    ax5.set_title(r'SkyW error', loc='left')
    ax5.tick_params(axis='x')
    ax5.tick_params(axis='y')
    ax5.invert_xaxis()
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)
    cbar=fig.colorbar(sc5, ax=ax5, fraction=0.09)
    ax5.set_facecolor('black')

    #need to make this look better!
    ax6 = fig.add_subplot(gs[row+2,1:5])
    ax6.hist(data_sci['flux_'+line], range=(vmin, vmaxs), log=True, stacked=True, bins=50, histtype='step', label='sky subtracted flux')
    ax6.hist(data_sci['sky_'+line], range=(vmin, vmaxs), log=True, stacked=True, bins=100, label='sky flux')
    ax6.hist(data_skye['flux_'+line]+data_skye['sky_'+line], range=(vmin, vmaxs), bins=50, histtype='step', log=True, stacked=True, label='SkyE flux')
    ax6.hist(data_skyw['flux_'+line]+data_skyw['sky_'+line], range=(vmin, vmaxs), bins=50, histtype='step', log=True, stacked=True, label='SkyW flux')
    ax6.tick_params(axis="x", direction="in", labeltop=True, labelbottom=False, pad=-15, top=True)
    fig.subplots_adjust(top=0.95)
    ax6.legend()

    return fig

def plotmap_line(data_sci, data_skye, data_skyw, maplist):
    fig1=plt.figure(1,(12,12))
    plt.clf()
    gs1= GridSpec(9, 6, figure=fig1, wspace=0.4, hspace=0.4)

    plt.rcParams.update({'axes.titlesize': 'small',
                 'axes.labelsize':'small',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'small',
                 'xtick.labelsize': 'small',
                 'font.size': '8.0',
                 'legend.fontsize':'small'})
    
    for j in range(3):
       fig1=plotmap_row(data_sci, data_skye, data_skyw, maplist[j], 3*j, fig1, gs1)
    
    return fig1

def plotmap_cont(data_sci, data_skye, data_skyw, maplist):
    fig2=plt.figure(1,(12,12))
    plt.clf()
    gs2= GridSpec(9, 6, figure=fig2, wspace=0.4, hspace=0.4)

    plt.rcParams.update({'axes.titlesize': 'small',
                 'axes.labelsize':'small',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'small',
                 'xtick.labelsize': 'small',
                 'font.size': '8.0',
                 'legend.fontsize':'small'})
    
    for j in range(3):
       fig2=plotmap_row(data_sci, data_skye, data_skyw, maplist[j+3], 3*j, fig2, gs2)
 
    return fig2

##### main programs ####
def run_qa_local():
    # for running locally (outside of DRP)
    #expnumlist = [5736, 6109, 6110, 6161, 6373, 6443, 6451, 6661, 9379, 9380, 10827, 10892, 11061, 11062, 14460, 15138, 15298, 15613, 15610, 15684, 15924]
    expnumlist = [6109]
    #mjdlist = [60222, 11111, 60222, 11111]  
    for i in range(len(expnumlist)):
        outfile=f'refdata_v1.1.0/skyQA_{expnumlist[i]}'
        rssfile=LVMDATA_DIR+f'lvmSFrame-{expnumlist[i]:0>8}.fits'
        wave, flux, sky, ivar, mask, slitmap, header = read_rssfile(rssfile)
        
        data_sci, data_skye, data_skyw, wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, stats_list, wvl_list, wvc_list, sky_info = create_sky_table(wave, flux, sky, ivar, mask, slitmap, header, maplist, medlist, lrangelist, crangelist, outfile)

        pdf = matplotlib.backends.backend_pdf.PdfPages(f'{outfile}.pdf')
        fig_med1=plot_intro(wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, sky_info)
        pdf.savefig(fig_med1)
        fig_med2=plot_stats(wave, med_sky, med_flux, med_ivar, stats_list, wvl_list, wvc_list)
        pdf.savefig(fig_med2)
        fig_maps1=plotmap_line(data_sci, data_skye, data_skyw, maplist)
        pdf.savefig(fig_maps1)
        fig_maps2=plotmap_cont(data_sci, data_skye, data_skyw, maplist)
        pdf.savefig(fig_maps2)
        pdf.close()

def run_qa(rssfile, outfile):
    # main program for running QA script in DRP
    wave, flux, sky, ivar, mask, slitmap, header = read_rssfile(rssfile)
        
    data_sci, data_skye, data_skyw, wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, stats_list, wvl_list, wvc_list, sky_info = create_sky_table(wave, flux, sky, ivar, mask, slitmap, header, maplist, medlist, lrangelist, crangelist, outfile)

    with matplotlib.backends.backend_pdf.PdfPages(f'{outfile}.pdf') as pdf:
        fig_med1=plot_intro(wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, sky_info)
        pdf.savefig(fig_med1)
        fig_med2=plot_stats(wave, med_sky, med_flux, med_ivar, stats_list, wvl_list, wvc_list)
        pdf.savefig(fig_med2)
        fig_maps1=plotmap_line(data_sci, data_skye, data_skyw, maplist)
        pdf.savefig(fig_maps1)
        fig_maps2=plotmap_cont(data_sci, data_skye, data_skyw, maplist)
        pdf.savefig(fig_maps2)
        plt.close()
    

####################################################################
#selecting the good fibers for a given telescope
def selfib_good(slitmap,type):
    selfib = slitmap['fibstatus']==0
    if type == 'sci':
        seltype = slitmap['targettype']=='science'
    elif type == 'skye':
        seltype = (slitmap['targettype'] =='SKY') & (slitmap['telescope'] == 'SkyE')
    elif type == 'skyw':
        seltype = (slitmap['targettype'] =='SKY') & (slitmap['telescope'] == 'SkyW')
    else:
        print('wrong type, must be sci, skye, or skyw')
    sel = seltype*selfib
    return sel

#open and read ext of file
def read_rssfile(filename):
    try:
        x=fits.open(filename)
    except Exception:
        print('Error: eval_qual: Could not open %s' % filename)
        return   
    slitmap=Table(x['SLITMAP'].data)
    wav=x['WAVE'].data
    flux=x['FLUX'].data
    sky=x['SKY'].data
    ivar=x['IVAR'].data
    mask=x['MASK'].data
    header=x[0].header
    x.close()
    return wav, flux, sky, ivar, mask, slitmap, header

def good_masked_fibers(slitmap, flux, sky, ivar, mask, type):
    flux[mask==1]=np.nan
    ivar[mask==1]=np.nan
    sky[mask==1]=np.nan

    # select science and good fibers only
    sel=selfib_good(slitmap,type)

    flux=flux[sel,:]
    ivar=ivar[sel,:]
    sky=sky[sel,:]    

    return flux, sky, ivar, sel

#getting the medians of the sky, flux, and err, and wave from file
def get_med(flux, sky, ivar, mask, xtab,):
    
    xgood=selfib_good(xtab, 'sci')
    flux_sci=flux[xgood,:]
    sky_sci=sky[xgood,:]
    ivar_sci=ivar[xgood,:]
    mask_sci=mask[xgood,:]

    xgood_skye=selfib_good(xtab, 'skye')
    skye_flux=flux[xgood_skye,:]
    skye_sky=sky[xgood_skye,:]
    skye_mask=mask[xgood_skye,:]
    xgood_skyw=selfib_good(xtab, 'skyw')
    skyw_flux=flux[xgood_skyw,:]
    skyw_sky=sky[xgood_skyw,:]
    skyw_mask=mask[xgood_skyw,:]

    flux_sci=np.ma.masked_array(flux_sci, mask_sci)
    sky_sci=np.ma.masked_array(sky_sci, mask_sci)
    ivar_sci=np.ma.masked_array(ivar_sci, mask_sci)
    skye_flux=np.ma.masked_array(skye_flux, skye_mask)
    skyw_flux=np.ma.masked_array(skyw_flux, skyw_mask) 
    skye_sky=np.ma.masked_array(skye_sky, skye_mask)
    skyw_sky=np.ma.masked_array(skyw_sky, skyw_mask)


    med_sky=np.ma.median(sky_sci,axis=0)
    med_flux=np.ma.median(flux_sci,axis=0)
    med_ivar=np.ma.median(ivar_sci,axis=0)*len(xgood)/(1.253**2)
    med_skye=np.ma.median(skye_flux+skye_sky,axis=0)
    med_skyw=np.ma.median(skyw_flux+skyw_sky,axis=0)
     
    return med_flux, med_sky, med_ivar, med_skye, med_skyw

#define the line and cont windows around line based on central line wave and width provided
def line_windows(lrange, crange, wave):
    
    wave_line = (wave>=lrange[0]) & (wave<=lrange[1]) # line range selection
    if crange[0] == -999:
        wave_cont = [-999]
    else:
        wave_cont = ((wave>=crange[0]) & (wave<lrange[0])) | ((wave>lrange[1]) & (wave<=crange[1])) # cont range selection
        
    return wave_line, wave_cont

#gets the stats wanted for the sky line residuals
def line_stats(flux, ivar, wave_line, wave_cont):
    if wave_cont[0] == -999:
        norm_line_flux = flux[wave_line]
    else:
        norm_line_flux = flux[wave_line]-np.ma.median(flux[wave_cont])
    rel_line_flux = norm_line_flux * np.sqrt(ivar[wave_line])
    res_mean = np.ma.mean(rel_line_flux)
    res_med = np.ma.median(rel_line_flux)
    res_std = np.ma.std(rel_line_flux)
    return res_mean, res_med, res_std

#gets the stats wanted for the sky cont residuals - no longer used
def cont_stats(flux, sky, ivar, wave_window):
    rel_cont_flux = flux[wave_window] * np.sqrt(ivar[wave_window])
    res_mean = np.mean(rel_cont_flux)
    res_med = np.median(rel_cont_flux)
    res_std = np.std(rel_cont_flux)
    return res_mean, res_med, res_std

def get_all_stats(wave, flux, sky, ivar, mask, slitmap, medlist):
    med_flux, med_sky, med_ivar, med_skye, med_skyw = get_med(flux, sky, ivar, mask, slitmap)
    
    wvl_list=[]
    wvc_list=[]
    stats_list=[]

    for j in range(len(medlist)):
        wvl, wvc = line_windows(lrangelist[j], crangelist[j], wave)
        stats_list.append(line_stats(med_flux, med_ivar, wvl, wvc))
        wvl_list.append(wvl)
        wvc_list.append(wvc)
        #stats_list.append([g1,g2,g3])
            
    return wave, med_sky, med_flux, med_ivar, med_skye, med_skyw, stats_list, wvl_list, wvc_list

def plot_intro(wave, sky, flux, ivar, skye, skyw, sky_info):
    fig1=plt.figure(1,(8,12),layout='constrained')
    plt.clf()
    gs1= GridSpec(4, 1, figure=fig1)

    #colors
    c_sky = 'blue'
    c_subsky = 'orange'
    c_ivar = 'limegreen'
    c_nsky = 'cyan'
    c_fsky = 'slategray'
    c_5sky = 'mediumblue'
    c_mw = ':r'
    #c_line = 'darkviolet'
    #c_cont = 'black'


    axt = fig1.add_subplot(gs1[0, 0])
    axt.axis([0,10,0,15])
    
    if sky_info['Moon_Alt'] < 0:
        moon_status = 'Moon BELOW horizon'
    else:
        moon_status = 'Moon ABOVE horizon'
    axt.text(0,14, f"Object: {sky_info['Object']}\nMJD: {sky_info['MJD'][0]}, Expnum: {sky_info['Exposure'][0]}\nObstime: {sky_info['Obstime']}")
    axt.text(0,9,f"Coordinates (RA, dec)\nSci:       {sky_info['Sci_RA']}, {sky_info['Sci_dec']}\nSkyE:    {sky_info['SkyE_RA']}, {sky_info['SkyE_dec']}\nSkyW:   {sky_info['SkyW_RA']}, {sky_info['SkyW_dec']}")
    axt.text(0,6,f"Distance from Sci (deg)\nSkyE: {sky_info['Sci_SkyE']}, SkyW: {sky_info['Sci_SkyW']}")
    axt.text(0,3,f"Distance from Moon (deg)\nSci: {sky_info['Sci_Moon']}, SkyE: {sky_info['SkyE_Moon']}, SkyW: {sky_info['SkyW_Moon']}")
    axt.text(0,1,f"{moon_status}, Lunar Illumination Fraction: {sky_info['Moon_Ill']}")
    axt.text(4,7,f"Altitude (deg)\nSci:       {sky_info['Sci_Alt']}\nSkyE:    {sky_info['SkyE_Alt']}\nSkyW:   {sky_info['SkyW_Alt']}\nSun:     {sky_info['Sun_Alt']}\nMoon:   {sky_info['Moon_Alt']}")
    axt.axes.get_xaxis().set_visible(False)
    axt.axes.get_yaxis().set_visible(False)
    axt.set_frame_on(False)

    ax1 = fig1.add_subplot(gs1[1, 0])
    ax1.plot(wave,sky,label='Sky flux', c=c_sky)
    ax1.plot(wave,0.05*sky,label='5% Sky flux', c=c_5sky, linestyle='--')
    ax1.plot(wave,flux,label='Sky subtracted flux', c=c_subsky)
    ax1.plot(wave,np.sqrt(1/ivar),label='Error', c=c_ivar)
    ax1.plot([3600,9600],[5.9e-15,5.9e-15],c_mw,label=r'$Med \pm$ MW 5 $\sigma$' )
    ax1.plot([3600,9600],[-5.9e-15,-5.9e-15],c_mw)
    ax1.set_xlim(3600,9600)
    ax1.set_ylim(-1e-14,1e-13)
    ax1.legend()
    
    if sky_info['Sci_SkyE'] < sky_info['Sci_SkyW']:
        nsky=skye
        fsky=skyw
        nlabel='SkyE (near) flux'
        flabel='SkyW (far) flux'
    else:
        nsky=skyw
        fsky=skye
        nlabel='SkyW (near) flux'
        flabel='SkyE (far) flux'

    axs = fig1.add_subplot(gs1[2, 0])
    axs.plot(wave,sky,label='Sky flux')
    axs.plot(wave,nsky,label=nlabel, c=c_nsky, alpha=0.5)
    axs.plot(wave,fsky,label=flabel, c=c_fsky, alpha=0.5)
    axs.set_xlim(3600,9600)
    axs.set_ylim(-1e-14,1e-13)
    axs.legend()

    axd = fig1.add_subplot(gs1[3, 0])
    axd.plot(wave,sky-nsky,label='Sky-NearSky',c=c_nsky)
    axd.plot(wave,sky-fsky,label='Sky-FarSky', c=c_fsky, alpha=0.5)
    axd.set_xlim(3600,9600)
    axd.set_ylim(-0.1e-13,0.1e-13)
    axd.legend()
    return fig1

def plot_stats_panel(ax, wave, sky, flux, ivar, xmin, xmax, wvl_list, wvc_list, stats1, stats2, stats3, line_name, c_sky, c_5sky, c_subsky, c_ivar, c_line, c_cont, c_mw):
    ax.plot(wave,sky,c=c_sky)
    ax.plot(wave,0.05*sky,c=c_5sky, linestyle='--')
    ax.plot(wave,flux,c=c_subsky)
    ax.plot(wave,np.sqrt(1/ivar),c=c_ivar)
    ax.plot(wave[wvl_list],flux[wvl_list], 'P', markersize=5, c=c_line)
    if c_cont != -999:
        ax.plot(wave[wvc_list],flux[wvc_list], 'P', markersize=5, c=c_cont)
    ax.plot([3600,9600],[5.9e-15,5.9e-15],c_mw)
    ax.plot([xmax-10,9600],[-5.9e-15,-5.9e-15],c_mw)
    ax.set_ylim(-1e-14,1e-14)
    ax.set_xlim(xmin,xmax)
    ax.text(xmin+2, -5.5e-15,f'{line_name}')
    ax.text(xmin+2, -6.5e-15,"mean {:5.4f}".format(stats1))
    ax.text(xmin+2, -7.5e-15,"med {:5.4f}".format(stats2))
    ax.text(xmin+2, -8.5e-15,"std {:5.4f}".format(stats3))
    return ax

def plot_stats(wave, sky, flux, ivar, stats_list, wvl_list, wvc_list):

    #colors
    c_sky = 'blue'
    c_subsky = 'orange'
    c_ivar = 'limegreen'
    #c_nsky = 'cyan'
    #c_fsky = 'slategray'
    c_5sky = 'mediumblue'
    c_mw = ':r'
    c_line = 'darkviolet'
    c_cont = 'black'

    fig2=plt.figure(1,(8,12))
    plt.clf()
    gs2= GridSpec(3, 3, figure=fig2)

    #group 1 5577
    ax2 = fig2.add_subplot(gs2[0, 0])
    ind=0
    ax2 = plot_stats_panel(ax2, wave, sky, flux, ivar, 5560, 5590, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, c_cont, c_mw)

    #group 2 5891
    ax3 = fig2.add_subplot(gs2[0, 1], sharey=ax2)
    ind=6
    ax3 = plot_stats_panel(ax3, wave, sky, flux, ivar, 5870, 5910, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, c_cont, c_mw)

    #group 3 6301
    ax4 = fig2.add_subplot(gs2[0, 2], sharey=ax3)
    ind=1
    ax4 = plot_stats_panel(ax4, wave, sky, flux, ivar, 6280, 6320, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, c_cont, c_mw)

    #group 4 6865
    ax5 = fig2.add_subplot(gs2[1, 0])
    ind=2
    ax5 = plot_stats_panel(ax5, wave, sky, flux, ivar, 6840, 6890, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, c_cont, c_mw)

    #group 5 8630-70 O2
    ax6 = fig2.add_subplot(gs2[1, 1], sharey=ax5)
    ind=7
    ax6 = plot_stats_panel(ax6, wave, sky, flux, ivar, 8620, 8685, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, -999, c_mw)

    #group 2 Airglow cont
    ax7 = fig2.add_subplot(gs2[1, 2], sharey=ax6)
    ind=8
    ax7 = plot_stats_panel(ax7, wave, sky, flux, ivar, 5410, 5450, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, -999, c_mw)

    #cont B channel
    ax8 = fig2.add_subplot(gs2[2, 0])
    ind=3
    ax8 = plot_stats_panel(ax8, wave, sky, flux, ivar, 4185, 4230, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, -999, c_mw)

    #cont r channel
    ax9 = fig2.add_subplot(gs2[2, 1], sharey=ax8)
    ind=4
    ax9 = plot_stats_panel(ax9, wave, sky, flux, ivar, 6410, 6450, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, -999, c_mw)

    #cont z channel
    ax10 = fig2.add_subplot(gs2[2, 2], sharey=ax9)
    ind=5
    ax10 = plot_stats_panel(ax10, wave, sky, flux, ivar, 9120, 9155, wvl_list[ind], wvc_list[ind], stats_list[ind][0], stats_list[ind][1], stats_list[ind][2], medlist[ind], c_sky, c_5sky, c_subsky, c_ivar, c_line, -999, c_mw)

    plt.tight_layout()
    #plt.savefig('skyqual_fig.png')
    #plt.show()
    return fig2

#### example to get med plots, just need filename
#plot_stats(*get_all_stats(filename))

######################################################################
#Info/header
#scripts from Knox's QuickLook code for good extra info to have
def get_moon_info_las_campanas(datetime_utc,verbose=False):
    '''
    Get information about the moon (and sun) as a fuction of UT
    '''
    # Las Campanas Observatory coordinates
    observatory_location = EarthLocation(lat=-29.0089*u.deg, lon=-70.6920*u.deg, height=2281*u.m)

    # Specify the observation time in UT
    obs_time = Time(datetime_utc)
    # print(obs_time.mjd)

    # Set the solar system ephemeris to 'builtin' for faster computation
    with solar_system_ephemeris.set('builtin'):
        # Get the Moon's and Sun's coordinates at the specified time
        moon_coords = get_body('moon', obs_time,location=observatory_location)
        sun_coords = get_body('sun',obs_time,location=observatory_location)

    # Calculate the phase angle (angle between the Sun, Moon, and observer)
    phase_angle = moon_coords.separation(sun_coords).radian

    # Calculate the illuminated fraction of the Moon
    illumination_fraction = (1 - np.cos(phase_angle))/2
    # print('separation',phase_angle,phase_angle*57.29578,illumination_fraction)
    moon_sun_longitude_diff = (moon_coords.ra - sun_coords.ra).wrap_at(360 * u.deg).value
    if moon_sun_longitude_diff>0:
        moon_phase=illumination_fraction/2.
    else:
        moon_phase=1-illumination_fraction/2.

    #illumination_fraction*=100.

    # Calculate the Altitude and Azimuth of the Moon from Las Campanas Observatory
    altaz_frame = AltAz(obstime=obs_time, location=observatory_location)
    moon_altaz = moon_coords.transform_to(altaz_frame)
    sun_altaz=sun_coords.transform_to(altaz_frame)



    # Calculate the difference in ecliptic longitudes between the moon and the sun
    #delta_longitude = (moon_coords.spherical.lon - sun_coords.spherical.lon).to_value('deg')
    # print('delta_long',delta_longitude)

    # Normalize the difference in ecliptic longitudes to get the moon's phase



    # Print the moon's phase
    # print("Moon's phase:", moon_phase)



    xreturn={
        'SunRA':sun_coords.ra.deg,
        'SunDec':sun_coords.dec.deg,
        'SunAlt': sun_altaz.alt.deg,
        'SunAz': sun_altaz.az.deg,
        'MoonRA': moon_coords.ra.deg,
        'MoonDec': moon_coords.dec.deg,
        'MoonAlt': moon_altaz.alt.deg,
        'MoonAz': moon_altaz.az.deg,
        'MoonPhas': moon_phase,
        'MoonIll': illumination_fraction
    }

    # print(xreturn)

    if verbose:
        for key, value in xreturn.items():
            print(f'{key}: {value}')
    # Return the information
    return xreturn

RADIAN=57.29578

def distance(r1,d1,r2,d2):
    '''
    distance(r1,d1,r2,d2)
    Return the angular offset between two ra,dec positions
    All variables are expected to be in degrees.
    Output is in degrees

    Note - This routine could easily be made more general
    '''
#    print 'distance',r1,d1,r2,d2
    r1=r1/RADIAN
    d1=d1/RADIAN
    r2=r2/RADIAN
    d2=d2/RADIAN
    xlambda=np.sin(d1)*np.sin(d2)+np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
#    print 'xlambda ',xlambda
    if xlambda>=1.0:
        xlambda=0.0
    else:
        xlambda=np.arccos(xlambda)

    xlambda=xlambda*RADIAN
#    print 'angle ',xlambda
    return xlambda

def get_header_value(header, key, default_value=-999.0, verbose=False):
    '''
    Robust way to get a header value if it exists
    '''

    try:
        value = header[key]
        if value is None:
            value=default_value
        elif isinstance(value, str):
            try:
                value = float(value)  # or int(value) if it's an integer
            except ValueError as e:
                if verbose:
                    print(f"Failed to convert '{value}' to a number for key '{key}': {e}")
                value = default_value
    except KeyError as e:
        if verbose:
            print(f"Key '{key}' not found in header: {e}")
        value = default_value
    return value



def get_header_string(header, key, default_string='Unknown', verbose=False):
    '''
    Robust way to get a header value if it exists
    '''

    try:
        value = header[key]
        if value is None:
            value=default_string
        elif isinstance(value, str):
            if value=='':
                return default_string
            return value
        else:
            if verbose:
                print(f"Key '{key}' found, but not string")
            return default_string
    except KeyError as e:
        if verbose:
            print(f"Key '{key}' not found in header: {e}")
        value = default_string
    return value



def create_overview(hdr):
    ''' Sumarize information about the  processed data file
    '''
   # try:
   #     x=fits.open(filename)
   # except:
   #     print('Error: Could not open %s' % filename)
   #     return

   # hdr=x['PRIMARY'].header

    exposure=get_header_value(hdr,'EXPOSURE')
    mjd=get_header_value(hdr,'MJD')
    object_name=get_header_string(hdr,'OBJECT')
    obs_time=get_header_string(hdr,'OBSTIME')
    ra=get_header_value(hdr,'TESCIRA')
    dec=get_header_value(hdr,'TESCIDE')
    alt=get_header_value(hdr, 'SKYMODEL SCI_ALT')

    
    ra_sky_e=get_header_value(hdr,'POSKYERA')
    dec_sky_e=get_header_value(hdr,'POSKYEDE')
    alt_sky_e=get_header_value(hdr, 'SKYMODEL SKYE_ALT')
    
    ra_sky_w=get_header_value(hdr,'POSKYWRA')
    dec_sky_w=get_header_value(hdr,'POSKYWDE')
    alt_sky_w=get_header_value(hdr, 'SKYMODEL SKYW_ALT')

    moon_info=get_moon_info_las_campanas(obs_time)
    
    distance_sky_w=distance(ra,dec,ra_sky_w,dec_sky_w)
    distance_sky_e=distance(ra,dec,ra_sky_e,dec_sky_e)
    distance_moon=distance(ra,dec,moon_info['MoonRA'],moon_info['MoonDec'])
    distance_skywmoon=distance(ra_sky_w,dec_sky_w,moon_info['MoonRA'],moon_info['MoonDec'])
    distance_skyemoon=distance(ra_sky_e,dec_sky_e,moon_info['MoonRA'],moon_info['MoonDec'])

    moon_info=get_moon_info_las_campanas(obs_time)
    
    xdict={
        'Exposure' : [exposure],
        'MJD' : [mjd],
        'Obstime' : obs_time,
        'Object' : object_name,
        'Sci_RA' : np.round(ra,2),
        'Sci_dec' : np.round(dec,2),
        'Sci_Alt' : np.round(alt,2),
        'SkyE_RA' : np.round(ra_sky_e,2),
        'SkyE_dec' : np.round(dec_sky_e,2),
        'SkyE_Alt' : np.round(alt_sky_e,2),
        'SkyW_RA' : np.round(ra_sky_w,2),
        'SkyW_dec' : np.round(dec_sky_w,2),
        'SkyW_Alt' : np.round(alt_sky_w,2),
        'Moon_Alt' : np.round(moon_info['MoonAlt'],2),
        'Moon_Ill' : np.round(moon_info['MoonIll'],2),
        'Sci_SkyE' : np.round(distance_sky_e,2), 
        'Sci_SkyW' : np.round(distance_sky_w,2),
        'Sun_Alt' : np.round(moon_info['SunAlt'], 2),
        'Sci_Moon' : np.round(distance_moon,2), 
        'SkyE_Moon' : np.round(distance_skyemoon,2), 
        'SkyW_Moon' : np.round(distance_skywmoon,2)
    }
 
    return xdict

if __name__ == "__main__":
    #to run a version on local computer
    run_qa_local()
    #to run a version on the DRP
    #run_qa()