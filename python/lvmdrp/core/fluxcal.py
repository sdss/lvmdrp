
# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez (adapted from MaNGA IDL code)
# @Date: Jan 27, 2023
# @Filename: astrometry.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

# FROM THE MANGA DRP CODE -------------------------------------------------------------------------
import numpy as np
from collections import namedtuple
from scipy.interpolate import interp1d
from scipy.optimize import nnls, minimize
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


def spflux_masklines(loglam, hwidth=None, stellar=True, telluric=True):
    """Returns a mask """

    if hwidth is None:
        # TODO: set this to whatever is equivalent to 400 km/s
        hwidth = 5.7e-4 # Default is to mask +/- 5.7 pix = 400 km/sec

    # initialize the mask array for wavelengths
    mask = np.zeros_like(loglam.size, dtype=bool)

    if stellar:
        starwave = [
        #       3692.6 ,  # H-16
        #       3698.2 ,  # H-15
            3704.9 ,  # H-14
        #       3707.1 ,  # Ca-II
            3713.0 ,  # H-13
            3723.0 ,  # H-12
            3735.4 ,  # H-11
        #       3738.0 ,  # Ca-II
            3751.2 ,  # H-10
            3771.7 ,  # H-9 
            3799.0 ,  # H-8
            3836.5 ,  # H-7 
            3890.2 ,  # H-6
            3934.8 ,  # Ca_k
            3969.6 ,  # Ca_H 
            3971.2 ,  # H-5
            4102.9 ,  # H-delta
            4300.  ,  # G-band
            4305.  ,  # G-band
            4310.  ,  # more G-band
            4341.7 ,  # H-gamma
            4862.7 ,  # H-beta
        #       4687.1 ,  # He II
            5168.8 ,  # Mg I
            5174.1 ,  # Mg I
            5185.0 ,  # Mg I
            5891.6 ,  # Na I
            5897.6 ,  # Na I
            6564.6 ,  # H-alpha
            8500.4 ,  # Ca II
            8544.4 ,  # Ca II
            8664.5 ,  # Ca II
            8752.9 ,  # H I
            8865.3 ,  # H I
        # RY: commented out the following three lines as these are in telluric absorption regions.
        #      If we mask them, we get bad bspline interpolation in these regions.
        #      They are not useful for stellar fitting anyway as they are in telluric regions.
        #       9017.8 ,  # H I
        #       9232.2 ,  # H I Pa-6
        #       9548.8 ,  # H I Pa-5
            10052.6
        ]   # H I (Pa-delta)
        #  airtovac, starwave #  commented out because wavelengths of features have already been set in vacuum.RY Jul 13, 2015

        for i in range(len(starwave)):
            mask = mask | (loglam > np.log10(starwave[i])-hwidth & loglam < np.log10(starwave[i])+hwidth)

    if telluric:
        tellwave1 = [6842., 7146., 7588., 8105., 8910.]
        tellwave2 = [6980., 7390., 7730., 8440., 9880.]
        for i in range(len(tellwave1)):
            mask = mask | (loglam > np.log10(tellwave1[i]) & loglam < np.log10(tellwave2[i]))

    return mask


def spflux_medianfilt(loglam, objflux, objivar, width, **kwargs):

    dims = objflux.shape
    ndim = len(dims)
    npix = dims[0]
    if ndim == 1:
        nspec = 1
    else:
        nspec = dims[1]

    #----------
    # Loop over each spectrum

    medflux = np.zeros_like(objflux)
    if objivar is not None: newivar = np.zeros_like(objivar)
    for ispec in range(nspec):
        # For the median-filter, ignore points near stellar absorp. features,
        # but keep points near telluric bands.
        qgood = np.logical_not(spflux_masklines(loglam[:, ispec], stellar=True, telluric=False, hwidth=8.e-4))

        # Median-filter, but skipping masked points
        igood = np.where(qgood)[0]
        ngood = igood.size
        thisback = np.zeros(npix)
        if (ngood > 1):
            thisback[igood] = median_filter(objflux[igood,ispec], size=width, **kwargs)
        thisback = np.interp(loglam, loglam[~qgood], thisback[~qgood])

        # Force the ends of the background to be the same as the spectrum,
        # which will force the ratio of the two to be unity.
        hwidth = np.ceil((width-1)/2.)
        thisback[0:hwidth] = objflux[0:hwidth,ispec]
        thisback[npix-1-hwidth:npix-1] = objflux[npix-1-hwidth:npix-1,ispec]
        czero2 = np.where(thisback == 0)[0]
        count2 = czero2.size
        if count2 > 0:
            thisback[czero2] = 1.
        medflux[:, ispec] = objflux[:, ispec] / thisback
        if objivar is not None:
            newivar[:, ispec] = objivar[:, ispec] * thisback**2

    return medflux, newivar


def spflux_bestmodel(loglam, objflux, objivar, dispimg, plottitle="", template="kurucz"):
    
    filtsz = 99 # the size of the window used in median-filter the spectra.
    cspeed = 2.99792458e5

    dims = objflux.shape
    ndim = len(dims)
    npix = dims[0]
    if ndim == 1:
        nspec = 1
    else:
        nspec = dims[1]

    #----------
    # Median-filter the object fluxes

    medflux, medivar = spflux_medianfilt(loglam, objflux, objivar, width=filtsz, mode="reflect")
    sqivar = np.sqrt(medivar)

    #----------
    # Mask out the telluric bands

    sqivar = sqivar * np.logical_not(spflux_masklines(loglam, telluric=True, stellar=False))

    #----------
    # Load the Kurucz models into memory

    # TODO: define functions spflux_read_x to read stellar spectra models
    #   CALLING SEQUENCE:
    #   spflux_read_bosz
    #   modelflux = spflux_read_bosz( loglam, dispimg, [ iselect=, 
    #    kindx_return= ,dslgpsize=dslgpsize ] )
    #
    # INPUTS:
    #   loglam     - Log10 wavelengths (vacuum Angstroms) [NPIX]
    #   dispimg    - Dispersion image, in units of pixels [NPIX]
    #
    # OPTIONAL INPUTS:
    #   iselect    - If set, then only return these model numbers# default to
    #                returning all models
    #
    # OUTPUTS:
    #   modelflux  - Model fluxes [NPIX,NMODEL]
    #
    # OPTIONAL OUTPUTS:
    #   kindx_return- Structure with model parameters for each model
    #   thekfile- Return which reference file was used (does NOT set which TO use!)
    #
    # NOTE: what is dslgpsize?
    if template == "kurucz":
        _, kindx, dslgpsize = spflux_read_kurucz() ##Yanping test
    elif template == "munari":
        _, kindx, dslgpsize = spflux_read_munari() ##Yanping added
    elif template == "BOSZ":
        _, kindx, dslgpsize = spflux_read_bosz() ##Yanping added
    else:
        print("Flux calibration templates has to be specified and be one of the three: 'kurucz','munari', 'BOSZ'.")

    nmodel = len(kindx)

    #----------
    # Fit the redshift just by using a canonical model

    ifud = np.where(kindx.teff == 6000 & kindx.g == 4 & kindx.feh == -1.5)[0]
    if ifud.size == 0:
        print('Could not find fiducial model!')
    nshift = np.ceil(1000./cspeed/np.log(10.)/dslgpsize/2)*2 # set this to cover +/-500 km/s
    logshift = (-nshift/2. + np.arange(nshift)) * dslgpsize    ##Yanping test
    chivec = np.zeros(nshift)
    for ishift in range(nshift):

        if template == "kurucz":
            modflux, kindx, dslgpsize = spflux_read_kurucz(loglam-logshift[ishift], dispimg, iselect=ifud) ##Yanping test
        elif template == "munari":
            modflux, kindx, dslgpsize = spflux_read_munari(loglam-logshift[ishift], dispimg, iselect=ifud) ##Yanping added
        elif template == "BOSZ":
            modflux, kindx, dslgpsize = spflux_read_bosz(loglam-logshift[ishift], dispimg, iselect=ifud) ##Yanping added
        else:
            print("Flux calibration templates has to be specified and be one of the three: 'kurucz','munari', 'BOSZ'.")

        # Median-filter this model
        medmodel, _ = spflux_medianfilt(loglam, modflux, width=filtsz, mode="reflect")
        for ispec in range(nspec):
            # NOTE: originally used computechi2
            chivec[ishift] = chivec[ishift] + nnls(medflux[:, ispec] / sqivar[:, ispec], medmodel[:, ispec] / sqivar[:, ispec])

    zshift = (10**logshift - 1) # Convert log-lambda shift to redshift
    # NOTE: originally used find_nminima
    result = minimize(interp1d(zshift, chivec), x0=0)
    zpeak = result.x
    print('Best-fit velocity for std star = ', zpeak * cspeed, ' km/s')
    if result.status != 0:
        print('Warning: Error code ', result.status, ' fitting std star')
    # Warning messages
    if np.isnan(chivec).any():
        if (medivar < 0).any():
            print('There are negative ivar values causing chi-square to be NaN or the likes.' )
        else:
            print('chi-square are NaN or the likes, but not caused by negative ivar.')
        
    #----------
    # Generate the Kurucz models at the specified wavelengths + dispersions,
    # using the best-fit redshift

    #modflux = spflux_read_kurucz(loglam-alog10(1.+zpeak), dispimg)

    if template == "kurucz":
        modflux, kindx, dslgpsize = spflux_read_kurucz(loglam-np.log10(1+zpeak), dispimg) ##Yanping test
    elif template == "munari":
        modflux, kindx, dslgpsize = spflux_read_munari(loglam-np.log10(1+zpeak), dispimg) ##Yanping added
    elif template == "BOSZ":
        modflux, kindx, dslgpsize = spflux_read_bosz(loglam-np.log10(1+zpeak), dispimg) ##Yanping added
    else:
        print("Flux calibration templates has to be specified and be one of the three: 'kurucz','munari', 'BOSZ'.")

    # Need to redo median-filter for the data with the correct redshift
    medflux, medivar = spflux_medianfilt(loglam-np.log10(1.+zpeak), objflux, objivar, size=filtsz, mode="reflect")
    sqivar = np.sqrt(medivar)
    #----------
    # Mask out the telluric bands
    sqivar = sqivar * np.logical_not(spflux_masklines(loglam, telluric=True, stellar=False))

    #----------
    # Loop through each model, computing the best chi^2
    # as the sum of the best-fit chi^2 to each of the several spectra
    # for this same object. Counting only the regions around stellar absorption features.
    # We do this after a median-filtering of both the spectra + the models.

    chiarr = np.zeros((nmodel,nspec))
    chivec = np.zeros(nmodel)
    medmodelarr = np.zeros_like(modflux)

    mlines = spflux_masklines(loglam-np.log10(1.+zpeak), hwidth=12e-4, stellar=True, telluric=False)
    linesqivar = sqivar * mlines
    linechiarr = np.zeros((nmodel,nspec))
    linechivec = np.zeros(nmodel)

    for imodel in range(nmodel):
        # Median-filter this model
        medmodelarr[:, :, imodel] = spflux_medianfilt(loglam-np.log10(1.+zpeak), modflux[:, :, imodel], size=filtsz, mode="reflect")

        for ispec in range(nspec):
            chiarr[imodel,ispec] = np.sum((medflux[:, ispec]-medmodelarr[:, ispec,imodel])**2*sqivar[:, ispec]*sqivar[:, ispec])
            linechiarr[imodel,ispec] = np.sum((medflux[:, ispec]-medmodelarr[:, ispec,imodel])**2*linesqivar[:, ispec]*linesqivar[:, ispec])
        chivec[imodel] = np.sum(chiarr[imodel, :])
        linechivec[imodel] = np.sum(linechiarr[imodel, :])

    #----------
    # Return the best-fit model
    # Computed both full spectra chi^2 and line chi^2, but use the line chi^2 to select the best model.
    ibest = np.argmin(linechivec)
    linechi2 = linechivec[ibest]
    linedof = np.sum(linesqivar != 0)
    print('Best-fit line chi2/DOF = ', linechi2/(linedof>1))
    bestflux = modflux[:, :, ibest]
    medmodel = spflux_medianfilt(loglam-np.log10(1.+zpeak), bestflux, size=filtsz, mode="reflects")

    #----------
    # Compute the median S/N for all the spectra of this object,
    # and for those data just near the absorp. lines

    indx = np.where(objivar > 0)[0]
    ct = indx.size
    if ct > 1:
        sn_median = np.median(objflux[indx] * np.sqrt(objivar[indx]))
    else:
        sn_median = 0

    indx = np.where(mlines)[0]
    ct = indx.size
    if ct > 1:
        linesn_median = np.median(objflux[indx] * np.sqrt(objivar[indx]))
    else:
        linesn_median = 0.
    print('Full median S/N = ', sn_median)
    print('Line median S/N = ', linesn_median)

    Parameters = namedtuple("Parameters", list(kindx.__dict__.keys()) + ["IMODEL", "Z", "SN_MEDIAN", "LINESN_MEDIAN", "LINECHI2", "LINEDOF"])
    kindx1 = Parameters(list(kindx.__dict__.values()) + [ibest, zpeak, float(sn_median), linesn_median, linechi2, linedof])

    #----------
    # Plot the filtered object spectrum, overplotting the best-fit Kurucz/Munari model ##Yanping edited

    # TODO: implement LVM 3 channels
    # Select the observation to plot that has the highest S/N,
    # and one that goes blueward of 4000 Ang.
    snvec = np.sum(objflux * np.sqrt(objivar), 1) * (10**loglam[0, :] < 4000 | 10.**loglam[npix-1, :] < 4000)
    iplot = np.argmax(snvec, axis=1) # Best blue exposure

    snvec = np.sum(objflux * np.sqrt(objivar), axis=1) * (10.**loglam[0, :] > 8600 | 10.**loglam[npix-1, :] > 8600)
    jplot = np.argmax(snvec, axis=1) # Best red exposure

    csize = 0.85
    _, ax = plt.subplots()
    ax.set_xlim(3840., 4120.)
    ax.set_ylim(0.0, 1.4)
    ax.set_ylabel('Wavelength [Ang]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(plottitle)
    ax.plot(10**loglam[:, iplot], medflux[:, iplot])
    ax.plot(10**loglam[:, iplot], medmodel[:, iplot], color='red')

    ax.text(3860, 1.25, kindx1.model, fontsize=csize, transform=ax.transAxes)
    ax.text(4000, 0.2, f"Lines \chi^2/DOF={linechi2/(linedof>1):.2f}", fontsize=csize, transform=ax.transAxes)
    ax.text(3860, 0.1, f"Fe/H={kindx1.feh:.1f}, T_{{eff}}={kindx1.teff:.0f}, g={kindx1.g:.1f}, cz={zpeak*cspeed:.0f}", fontsize=csize, transform=ax.transAxes)

    _, ax = plt.subplots()
    ax.set_xlim(8440., 9160.)
    ax.set_ylim(0.0, 1.4)
    ax.set_ylabel('Wavelength [Ang]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(plottitle)
    ax.plot(10**loglam[:, jplot], medflux[:, jplot])
    ax.plot(10**loglam[:, jplot], medmodel[:, jplot], color='red')

    return bestflux, kindx1


def spflux_goodfiber(pixmask):
    qgood = (
        'NOPLUG' not in pixmask and
        'BADTRACE' not in pixmask and
        'BADFLAT' not in pixmask and
        'BADARC' not in pixmask and
        'MANYBADCOLUMNS' not in pixmask and
        'NEARWHOPPER' not in pixmask and
        'MANYREJECTED' not in pixmask
    )
    return qgood


def spflux_bspline(loglam, mratio, mrativar, inmask=None, return_outmask=True, everyn=10, disp=None, hwidth=None, mask_stellar=True, mask_telluric=True):

    # TODO: figure out which default for everyn is best
    isort = np.argsort(loglam)
    nord = 3

    if hwidth is None:
        hwidth=12.e-4

    if inmask is None:
    # Choose the break points using the EVERYN option, but masking
    # out more pixels near stellar features and/or telluric just when selecting them.
        mask1 = np.logical_not(spflux_masklines(loglam, hwidth=hwidth, stellar=mask_stellar,telluric=mask_telluric)) 
    else:
        mask1 = np.logical_not(inmask)

    ii = np.where((mrativar[isort] > 0) & mask1[isort])
    bkpt = 0
    fullbkpt = bspline_bkpts(loglam[isort[ii]], everyn=everyn, bkpt=bkpt, nord=nord)

    outmask1 = 0
    if disp is not None:
        x2 = disp[isort]
    else:
        pass

    # BUG: what is indx?
    sset = bspline_iterfit(
        loglam[isort], mratio[isort], 
        invvar=mrativar[isort], lower=3, upper=3, fullbkpt=fullbkpt, 
        maxrej=np.ceil(0.05*len(indx)), outmask=outmask1, nord=nord, 
        x2=x2, npoly=2*keyword_set(disp), requiren=(everyn-1)>1
    )
    if (np.max(sset.coeff) == 0):
        print('B-spline fit failed!!')

    if return_outmask:
        outmask = np.zeros_like(loglam)
        outmask[isort] = outmask1
        return sset, outmask

    return sset

# -------------------------------------------------------------------------------------------------
