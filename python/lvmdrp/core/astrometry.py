# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 27, 2023
# @Filename: astrometry.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


# read original position table in mm
# convert position table from mm to arcsec using the focal radius
# format Euro3D
# row 1: s, c, h
# download PanSTARR images with pixel size of 1/4 of the size of the fiber
# size of the search box 1/3 of the IFU size


from multiprocessing import Pool, cpu_count

import numpy as np


def get_ps_images(ra, dec, size_pix, scale, passband_name):
    """
    
        Downloads reference image from PanSTARRS
    
    """
    pass


def register_image(rss, image, passband, search_box, step_search, ref_pix_x, ref_pix_y, arc_scale, angle=0.0, guess_x=0.0, guess_y=0.0, parallel='auto'):
    """
        
        Returns a map of offsets and the corresponding chi-square map after matching the reference
        image with the given RSS.

        Given an RSS object ('rss') and a reference 'image' in a 'passband', this function will map
        the matching of the RSS with the reference image by applying a set of offsets within a
        predefined 'search_box' in 'step_search'

    """
    posTab = rss.getPositionTable()
    fiber_area = np.pi*posTab._size[0]**2

    # extract the Passband fluxes for each spectrum of the RSS
    (flux_rss, error_rss, min_rss, max_rss, std_rss) = passband.getFluxRSS(rss)
    flux_rss = flux_rss*fiber_area
    error_rss = error_rss*fiber_area
    select_neg = flux_rss<=0
    flux_rss[select_neg] = 1e-10
    rss_mag = passband.fluxToMag(flux_rss)
    AB_flux =10**(rss_mag/-2.5)
    AB_eflux = error_rss*(AB_flux/flux_rss)
    good_rss = flux_rss/error_rss>3.0

    # define empty areas for the search grid
    offsets_x = np.arange(-search_box/2.0, search_box/2.0+step_search, step_search)+guess_x
    offsets_y = np.arange(-search_box/2.0, search_box/2.0+step_search, step_search)+guess_y
    angles_off = np.arange(-10, 10, 2)
    offsets_xIFU = np.zeros((len(offsets_x), len(offsets_y)))
    offsets_yIFU = np.zeros((len(offsets_x), len(offsets_y)))
    chisq = np.zeros((len(offsets_x), len(offsets_y)))
    scale_flux = np.zeros((len(offsets_x), len(offsets_y)))
    valid_fibers = np.zeros((len(offsets_x), len(offsets_y)))

    if parallel=='auto':
        cpus = cpu_count()
    else:
        cpus = int(parallel)
    if cpus>1:
        pool = Pool(cpus)
        threads=[]
        for i in range(len(offsets_x)):
            for j in range(len(offsets_y)):
                threads.append(pool.apply_async(image.extractApertures, args=(posTab,  ref_pix_x, ref_pix_y, arc_scale, angle, offsets_x[i], offsets_y[j])))
        pool.close()
        pool.join()
    m = 0
    for i in range(len(offsets_x)):
        for j in range(len(offsets_y)):
            offsets_xIFU[i, j] = offsets_x[i]/arc_scale
            offsets_yIFU[i, j] = offsets_y[j]/arc_scale
            if cpus==1:
                flux = image.extractApertures(posTab, ref_pix_x, ref_pix_y, arc_scale, angle=angle, offset_arc_x=offsets_x[i], offset_arc_y=offsets_y[j])
            else:
                flux = threads[m].get()
            good_pix = (flux[0]/flux[1]>3.0)
            good = np.logical_or(good_pix, good_rss) & (flux[2]>0.0) & np.logical_not(np.isnan(flux[1])) & np.logical_not(np.isnan(error_rss))
            sort = np.argsort(flux[0][good])
            match = np.median(flux[0][good][sort[:-1]]/AB_flux[good][sort[:-1]])
            scale_flux[i, j] = match
            valid_fibers[i,j] = np.sum(good[sort[:-1]])
            chisq[i, j] = float(np.sum((flux[0][good][sort[:-1]]-scale_flux[i, j]*AB_flux[good][sort[:-1]])**2 /
            (flux[1][good][sort[:-1]]**2+AB_eflux[good][sort[:-1]]**2)))/(valid_fibers[i,j])
            m+=1

    return offsets_xIFU*arc_scale, offsets_yIFU*arc_scale, chisq, scale_flux, AB_flux, valid_fibers
