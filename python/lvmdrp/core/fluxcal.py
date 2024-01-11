# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez (adapted from MaNGA IDL code)
# @Date: Jan 27, 2023
# @Filename: fluxcal.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

from collections import namedtuple

import matplotlib.pyplot as plt

# FROM THE MANGA DRP CODE -------------------------------------------------------------------------
import numpy as np
from dust_extinction.parameter_averages import F99
from pydl.pydlspec2d.spec2d import filter_thru
from pydl.pydlutils.sdss import sdss_flagval
from scipy.interpolate import BSpline, interp1d, splrep
from scipy.ndimage import median_filter
from scipy.optimize import minimize, nnls


f99_ext = F99(Rv=3.1)

# -------------------------------------------------------------------------------------------------


def spflux_masklines(loglam, hwidth=None, stellar=True, telluric=True):
    """Returns a mask"""

    if hwidth is None:
        # TODO: set this to whatever is equivalent to 400 km/s
        hwidth = 5.7e-4  # Default is to mask +/- 5.7 pix = 400 km/sec

    # initialize the mask array for wavelengths
    mask = np.zeros_like(loglam.size, dtype=bool)

    if stellar:
        starwave = [
            #       3692.6 ,  # H-16
            #       3698.2 ,  # H-15
            3704.9,  # H-14
            #       3707.1 ,  # Ca-II
            3713.0,  # H-13
            3723.0,  # H-12
            3735.4,  # H-11
            #       3738.0 ,  # Ca-II
            3751.2,  # H-10
            3771.7,  # H-9
            3799.0,  # H-8
            3836.5,  # H-7
            3890.2,  # H-6
            3934.8,  # Ca_k
            3969.6,  # Ca_H
            3971.2,  # H-5
            4102.9,  # H-delta
            4300.0,  # G-band
            4305.0,  # G-band
            4310.0,  # more G-band
            4341.7,  # H-gamma
            4862.7,  # H-beta
            #       4687.1 ,  # He II
            5168.8,  # Mg I
            5174.1,  # Mg I
            5185.0,  # Mg I
            5891.6,  # Na I
            5897.6,  # Na I
            6564.6,  # H-alpha
            8500.4,  # Ca II
            8544.4,  # Ca II
            8664.5,  # Ca II
            8752.9,  # H I
            8865.3,  # H I
            # RY: commented out the following three lines as these are in telluric absorption regions.
            #      If we mask them, we get bad bspline interpolation in these regions.
            #      They are not useful for stellar fitting anyway as they are in telluric regions.
            #       9017.8 ,  # H I
            #       9232.2 ,  # H I Pa-6
            #       9548.8 ,  # H I Pa-5
            10052.6,
        ]  # H I (Pa-delta)
        #  airtovac, starwave #  commented out because wavelengths of features have already been set in vacuum.RY Jul 13, 2015

        for i in range(len(starwave)):
            mask = mask | (
                loglam
                > np.log10(starwave[i]) - hwidth & loglam
                < np.log10(starwave[i]) + hwidth
            )

    if telluric:
        tellwave1 = [6842.0, 7146.0, 7588.0, 8105.0, 8910.0]
        tellwave2 = [6980.0, 7390.0, 7730.0, 8440.0, 9880.0]
        for i in range(len(tellwave1)):
            mask = mask | (
                loglam > np.log10(tellwave1[i]) & loglam < np.log10(tellwave2[i])
            )

    return mask


def spflux_medianfilt(loglam, objflux, objivar, width, **kwargs):
    dims = objflux.shape
    ndim = len(dims)
    npix = dims[0]
    if ndim == 1:
        nspec = 1
    else:
        nspec = dims[1]

    # ----------
    # Loop over each spectrum

    medflux = np.zeros_like(objflux)
    if objivar is not None:
        newivar = np.zeros_like(objivar)
    for ispec in range(nspec):
        # For the median-filter, ignore points near stellar absorp. features,
        # but keep points near telluric bands.
        qgood = np.logical_not(
            spflux_masklines(
                loglam[:, ispec], stellar=True, telluric=False, hwidth=8.0e-4
            )
        )

        # Median-filter, but skipping masked points
        igood = np.where(qgood)[0]
        ngood = igood.size
        thisback = np.zeros(npix)
        if ngood > 1:
            thisback[igood] = median_filter(objflux[igood, ispec], size=width, **kwargs)
        thisback = np.interp(loglam, loglam[~qgood], thisback[~qgood])

        # Force the ends of the background to be the same as the spectrum,
        # which will force the ratio of the two to be unity.
        hwidth = np.ceil((width - 1) / 2.0)
        thisback[0:hwidth] = objflux[0:hwidth, ispec]
        thisback[npix - 1 - hwidth: npix - 1] = objflux[
            npix - 1 - hwidth: npix - 1, ispec
        ]
        czero2 = np.where(thisback == 0)[0]
        count2 = czero2.size
        if count2 > 0:
            thisback[czero2] = 1.0
        medflux[:, ispec] = objflux[:, ispec] / thisback
        if objivar is not None:
            newivar[:, ispec] = objivar[:, ispec] * thisback**2

    return medflux, newivar


def spflux_bestmodel(
    loglam, objflux, objivar, dispimg, plottitle="", template="kurucz"
):
    filtsz = 99  # the size of the window used in median-filter the spectra.
    cspeed = 2.99792458e5

    dims = objflux.shape
    ndim = len(dims)
    npix = dims[0]
    if ndim == 1:
        nspec = 1
    else:
        nspec = dims[1]

    # ----------
    # Median-filter the object fluxes

    medflux, medivar = spflux_medianfilt(
        loglam, objflux, objivar, width=filtsz, mode="reflect"
    )
    sqivar = np.sqrt(medivar)

    # ----------
    # Mask out the telluric bands

    sqivar = sqivar * np.logical_not(
        spflux_masklines(loglam, telluric=True, stellar=False)
    )

    # ----------
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
        _, kindx, dslgpsize = spflux_read_kurucz()  # Yanping test
    elif template == "munari":
        _, kindx, dslgpsize = spflux_read_munari()  # Yanping added
    elif template == "BOSZ":
        _, kindx, dslgpsize = spflux_read_bosz()  # Yanping added
    else:
        print(
            "Flux calibration templates has to be specified and be one of the three: 'kurucz','munari', 'BOSZ'."
        )

    nmodel = len(kindx)

    # ----------
    # Fit the redshift just by using a canonical model

    ifud = np.where(kindx.teff == 6000 & kindx.g == 4 & kindx.feh == -1.5)[0]
    if ifud.size == 0:
        print("Could not find fiducial model!")
    nshift = (
        np.ceil(1000.0 / cspeed / np.log(10.0) / dslgpsize / 2) * 2
    )  # set this to cover +/-500 km/s
    logshift = (-nshift / 2.0 + np.arange(nshift)) * dslgpsize  ##Yanping test
    chivec = np.zeros(nshift)
    for ishift in range(nshift):
        if template == "kurucz":
            modflux, kindx, dslgpsize = spflux_read_kurucz(
                loglam - logshift[ishift], dispimg, iselect=ifud
            )  ##Yanping test
        elif template == "munari":
            modflux, kindx, dslgpsize = spflux_read_munari(
                loglam - logshift[ishift], dispimg, iselect=ifud
            )  ##Yanping added
        elif template == "BOSZ":
            modflux, kindx, dslgpsize = spflux_read_bosz(
                loglam - logshift[ishift], dispimg, iselect=ifud
            )  ##Yanping added
        else:
            print(
                "Flux calibration templates has to be specified and be one of the three: 'kurucz','munari', 'BOSZ'."
            )

        # Median-filter this model
        medmodel, _ = spflux_medianfilt(loglam, modflux, width=filtsz, mode="reflect")
        for ispec in range(nspec):
            # NOTE: originally used computechi2
            chivec[ishift] = chivec[ishift] + nnls(
                medflux[:, ispec] / sqivar[:, ispec],
                medmodel[:, ispec] / sqivar[:, ispec],
            )

    zshift = 10**logshift - 1  # Convert log-lambda shift to redshift
    # NOTE: originally used find_nminima
    result = minimize(interp1d(zshift, chivec), x0=0)
    zpeak = result.x
    print("Best-fit velocity for std star = ", zpeak * cspeed, " km/s")
    if result.status != 0:
        print("Warning: Error code ", result.status, " fitting std star")
    # Warning messages
    if np.isnan(chivec).any():
        if (medivar < 0).any():
            print(
                "There are negative ivar values causing chi-square to be NaN or the likes."
            )
        else:
            print("chi-square are NaN or the likes, but not caused by negative ivar.")

    # ----------
    # Generate the Kurucz models at the specified wavelengths + dispersions,
    # using the best-fit redshift

    # modflux = spflux_read_kurucz(loglam-np.log10(1.+zpeak), dispimg)

    if template == "kurucz":
        modflux, kindx, dslgpsize = spflux_read_kurucz(
            loglam - np.log10(1 + zpeak), dispimg
        )  # Yanping test
    elif template == "munari":
        modflux, kindx, dslgpsize = spflux_read_munari(
            loglam - np.log10(1 + zpeak), dispimg
        )  # Yanping added
    elif template == "BOSZ":
        modflux, kindx, dslgpsize = spflux_read_bosz(
            loglam - np.log10(1 + zpeak), dispimg
        )  # Yanping added
    else:
        print(
            "Flux calibration templates has to be specified and be one of the three: 'kurucz','munari', 'BOSZ'."
        )

    # Need to redo median-filter for the data with the correct redshift
    medflux, medivar = spflux_medianfilt(
        loglam - np.log10(1.0 + zpeak), objflux, objivar, size=filtsz, mode="reflect"
    )
    sqivar = np.sqrt(medivar)
    # ----------
    # Mask out the telluric bands
    sqivar = sqivar * np.logical_not(
        spflux_masklines(loglam, telluric=True, stellar=False)
    )

    # ----------
    # Loop through each model, computing the best chi**2
    # as the sum of the best-fit chi**2 to each of the several spectra
    # for this same object. Counting only the regions around stellar absorption features.
    # We do this after a median-filtering of both the spectra + the models.

    chiarr = np.zeros((nmodel, nspec))
    chivec = np.zeros(nmodel)
    medmodelarr = np.zeros_like(modflux)

    mlines = spflux_masklines(
        loglam - np.log10(1.0 + zpeak), hwidth=12e-4, stellar=True, telluric=False
    )
    linesqivar = sqivar * mlines
    linechiarr = np.zeros((nmodel, nspec))
    linechivec = np.zeros(nmodel)

    for imodel in range(nmodel):
        # Median-filter this model
        medmodelarr[:, :, imodel] = spflux_medianfilt(
            loglam - np.log10(1.0 + zpeak),
            modflux[:, :, imodel],
            size=filtsz,
            mode="reflect",
        )

        for ispec in range(nspec):
            chiarr[imodel, ispec] = np.sum(
                (medflux[:, ispec] - medmodelarr[:, ispec, imodel]) ** 2
                * sqivar[:, ispec]
                * sqivar[:, ispec]
            )
            linechiarr[imodel, ispec] = np.sum(
                (medflux[:, ispec] - medmodelarr[:, ispec, imodel]) ** 2
                * linesqivar[:, ispec]
                * linesqivar[:, ispec]
            )
        chivec[imodel] = np.sum(chiarr[imodel, :])
        linechivec[imodel] = np.sum(linechiarr[imodel, :])

    # ----------
    # Return the best-fit model
    # Computed both full spectra chi**2 and line chi**2, but use the line chi**2 to select the best model.
    ibest = np.argmin(linechivec)
    linechi2 = linechivec[ibest]
    linedof = np.sum(linesqivar != 0)
    print("Best-fit line chi2/DOF = ", linechi2 / (linedof > 1))
    bestflux = modflux[:, :, ibest]
    medmodel = spflux_medianfilt(
        loglam - np.log10(1.0 + zpeak), bestflux, size=filtsz, mode="reflects"
    )

    # ----------
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
        linesn_median = 0.0
    print("Full median S/N = ", sn_median)
    print("Line median S/N = ", linesn_median)

    Parameters = namedtuple(
        "Parameters",
        list(kindx.__dict__.keys())
        + ["IMODEL", "Z", "SN_MEDIAN", "LINESN_MEDIAN", "LINECHI2", "LINEDOF"],
    )
    kindx1 = Parameters(
        list(kindx.__dict__.values())
        + [ibest, zpeak, float(sn_median), linesn_median, linechi2, linedof]
    )

    # ----------
    # Plot the filtered object spectrum, overplotting the best-fit Kurucz/Munari model ##Yanping edited

    # TODO: implement LVM 3 channels
    # Select the observation to plot that has the highest S/N,
    # and one that goes blueward of 4000 Ang.
    snvec = np.sum(objflux * np.sqrt(objivar), 1) * (
        10 ** loglam[0, :] < 4000 | 10.0 ** loglam[npix - 1, :] < 4000
    )
    iplot = np.argmax(snvec, axis=1)  # Best blue exposure

    snvec = np.sum(objflux * np.sqrt(objivar), axis=1) * (
        10.0 ** loglam[0, :] > 8600 | 10.0 ** loglam[npix - 1, :] > 8600
    )
    jplot = np.argmax(snvec, axis=1)  # Best red exposure

    csize = 0.85
    _, ax = plt.subplots()
    ax.set_xlim(3840.0, 4120.0)
    ax.set_ylim(0.0, 1.4)
    ax.set_ylabel("Wavelength [Ang]")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(plottitle)
    ax.plot(10 ** loglam[:, iplot], medflux[:, iplot])
    ax.plot(10 ** loglam[:, iplot], medmodel[:, iplot], color="red")

    ax.text(3860, 1.25, kindx1.model, fontsize=csize, transform=ax.transAxes)
    ax.text(
        4000,
        0.2,
        f"Lines \chi^2/DOF={linechi2/(linedof>1):.2f}",
        fontsize=csize,
        transform=ax.transAxes,
    )
    ax.text(
        3860,
        0.1,
        f"Fe/H={kindx1.feh:.1f}, T_{{eff}}={kindx1.teff:.0f}, g={kindx1.g:.1f}, cz={zpeak*cspeed:.0f}",
        fontsize=csize,
        transform=ax.transAxes,
    )

    _, ax = plt.subplots()
    ax.set_xlim(8440.0, 9160.0)
    ax.set_ylim(0.0, 1.4)
    ax.set_ylabel("Wavelength [Ang]")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(plottitle)
    ax.plot(10 ** loglam[:, jplot], medflux[:, jplot])
    ax.plot(10 ** loglam[:, jplot], medmodel[:, jplot], color="red")

    return bestflux, kindx1


def spflux_goodfiber(pixmask):
    qgood = (
        "NOPLUG" not in pixmask
        and "BADTRACE" not in pixmask
        and "BADFLAT" not in pixmask
        and "BADARC" not in pixmask
        and "MANYBADCOLUMNS" not in pixmask
        and "NEARWHOPPER" not in pixmask
        and "MANYREJECTED" not in pixmask
    )
    return qgood


def spflux_bspline(
    loglam,
    mratio,
    mrativar,
    inmask=None,
    return_outmask=True,
    everyn=10,
    disp=None,
    hwidth=None,
    mask_stellar=True,
    mask_telluric=True,
):
    # TODO: figure out which default for everyn is best
    isort = np.argsort(loglam)
    nord = 3

    if hwidth is None:
        hwidth = 12.0e-4

    if inmask is None:
        # Choose the break points using the EVERYN option, but masking
        # out more pixels near stellar features and/or telluric just when selecting them.
        mask1 = np.logical_not(
            spflux_masklines(
                loglam, hwidth=hwidth, stellar=mask_stellar, telluric=mask_telluric
            )
        )
    else:
        mask1 = np.logical_not(inmask)

    ii = np.where((mrativar[isort] > 0) & mask1[isort])
    # BUG: this is actually done by bspline_bkpts
    fullbkpt = splrep(
        loglam[isort[ii]],
        mratio[isort[ii]],
        w=np.sqrt(mrativar[isort[ii]]),
        k=nord,
        t=loglam[isort[ii]][np.arange(everyn, dtype=int)],
        quiet=1,
    )

    outmask1 = 0
    # if disp is not None:
    #     x2 = disp[isort]
    # else:
    #     pass

    # BUG: this is actually done by bspline_iterfit
    sset = BSpline(*fullbkpt)

    if np.max(sset.c) == 0:
        print("B-spline fit failed!!")

    if return_outmask:
        outmask = np.zeros_like(loglam)
        outmask[isort] = outmask1
        return sset, outmask

    return sset


def typingmodule(
    objflux,
    loglam,
    objivar,
    dispimg,
    sfd_ebv,
    psfmag,
    plottitle="",
    template="kurucz",
    thekfile="",
    targetflag=None,
):
    npix, nspec = objflux.shape
    # ----------
    # For each star, find the best-fit model.

    unreddenfactor = 1 / f99_ext.extinguish(10.0**loglam, Ebv=1.0)

    #   !p.multi = [0,1,2]
    modflux = np.zeros_like(objflux)
    # Find the best-fit model -- evaluated for each exposure [NPIX,NEXP]
    thismodel, kindx = spflux_bestmodel(
        loglam,
        objflux * unreddenfactor**sfd_ebv,
        objivar / unreddenfactor ** (2 * sfd_ebv),
        plottitle=plottitle,
        template=template,
    )

    # Also evaluate this model over a big wavelength range [3000,11000] Ang.
    # BUG: verify these hard-coded values
    tmploglam = 3.4771e0 + np.linspace(5644) * 1e-4
    tmpdispimg = np.ones_like(tmploglam)  # initializing this resolution vector
    bluedispimg = tmpdispimg
    reddispimg = tmpdispimg
    bside = np.where(tmploglam < np.log10(6300.0))
    rside = np.where(tmploglam > np.log10(5900.0))
    middle = np.where((tmploglam < np.log10(6300.0)) & (tmploglam > np.log10(5900.0)))
    bluedispimg[bside] = np.interp(tmploglam[bside], dispimg[:, 0], loglam[:, 0])
    reddispimg[rside] = np.interp(tmploglam[rside], dispimg[:, 1], loglam[:, 1])
    tmpdispimg[bside] = bluedispimg[bside]
    tmpdispimg[rside] = reddispimg[rside]
    tmpdispimg[middle] = (bluedispimg[middle] + reddispimg[middle]) / 2.0

    # tmpflux = spflux_read_kurucz(tmploglam-np.log10(1+kindx.z), tmpdispimg,
    #    iselect=kindx.imodel)

    if template == "kurucz":
        tmpflux = spflux_read_kurucz(
            tmploglam - np.log10(1 + kindx.z),
            tmpdispimg,
            iselect=kindx.imodel,
            thekfile=thekfile,
        )
    elif template == "munari":
        tmpflux = spflux_read_munari(
            tmploglam - np.log10(1 + kindx.z),
            tmpdispimg,
            iselect=kindx.imodel,
            thekfile=thekfile,
        )
    elif template == "BOSZ":
        tmpflux = spflux_read_bosz(
            tmploglam - np.log10(1 + kindx.z),
            tmpdispimg,
            iselect=kindx.imodel,
            thekfile=thekfile,
        )
    else:
        print("Template is not specified correctly.")

    # The returned models are redshifted, but not fluxed or
    # reddened.  Do that now...  we compare data vs. model reddened.
    #   extcurve1 = ext_odonnell(10.**loglam, 3.1)
    #   extinct,10.**loglam,extcurve1,/ccm,Rv=3.1 # extcurve1 is A_lambda for Av=1
    #   thismodel = thismodel * 10.**(-extcurve1 * 3.1 * sfd_ebv / 2.5)
    reddenfactor = f99_ext.extinguish(10**loglam, Ebv=1.0)
    # fluxreddened contain the reddening vector for E(B-V)=1.0
    thismodel = thismodel * reddenfactor**sfd_ebv
    #   extcurve2 = ext_odonnell(10.**tmploglam, 3.1)
    #   extinct,10.**tmploglam,extcurve2,/ccm,Rv=3.1
    #   tmpflux = tmpflux * 10.**(-extcurve2 * 3.1 * sfd_ebv / 2.5)
    reddenfactor2 = f99_ext.extinguish(10**tmploglam, Ebv=1.0)
    tmpflux = tmpflux * reddenfactor2**sfd_ebv

    # Now integrate the apparent magnitude for this spectrum,
    # The units of FTHRU are such that m = -2.5*np.log10(FTHRU) + (48.6-2.5*17)
    # Note that these computed magnitudes, THISMAG, should be equivalent
    # to THISINDX.MAG in the case of no reddening.
    wavevec = 10e0**tmploglam
    flambda2fnu = wavevec**2 / 2.99792e18

    photometry = "sdss"  # Both APASS and SDSS are using sdss photometry system
    if (
        targetflag
        and sdss_flagval("MANGA_TARGET2", ["STELLIB_PS1", "STD_PS1_COM"]) != 0
    ):
        photometry = "ps1"
    if targetflag and sdss_flagval("MANGA_TARGET2", "STELLIB_GAIA") != 0:
        photometry = "gaiadr1"
    if targetflag and sdss_flagval("MANGA_TARGET2", "STELLIB_GAIADR2") != 0:
        photometry = "gaiadr2"

    if photometry == "ps1":
        fthru = filter_thru(
            tmpflux * flambda2fnu,
            waveimg=wavevec,
            toair=True,
            filternames=["ps1_g.txt", "ps1_r.txt", "ps1_i.txt", "ps1_z.txt"],
        )
        thismag = -2.5 * np.log10(fthru) - (48.6 - 2.5 * 17)
        thismag = np.asarray([[-999.0], [thismag]])
    elif photometry == "gaiadr1":
        # Using the passbands from Gaia DR2 to compute the mag, which is not quite appropriate for DR1. Better to avoid using gaiadr1
        fthru = filter_thru(
            tmpflux * flambda2fnu,
            waveimg=wavevec,
            toair=True,
            filternames=["gaia_G.dat"],
        )
        thismag = -2.5 * np.log10(fthru) - (48.6 - 2.5 * 17)
    elif photometry == "gaiadr2":
        fthru = filter_thru(
            tmpflux * flambda2fnu,
            waveimg=wavevec,
            toair=True,
            filternames=["gaia_G.dat", "gaia_BP.dat", "gaia_RP.dat"],
        )
        thismag = -2.5 * np.log10(fthru) - (48.6 - 2.5 * 17)
        # Convert from AB system using revised passband to Vega system in the Gaia DR2 as-released system. The zeropoints are from Evans et al. (2018)
        thismag = thismag + np.asarray(
            [[25.6884 - 25.7916], [25.3514 - 25.3862], [24.7619 - 25.1162]]
        )
        thismag = np.asarray([[-999.0], [thismag], [-999.0]])
    elif photometry == "sdss":  # this applies to both SDSS and APASS standards.
        fthru = filter_thru(tmpflux * flambda2fnu, waveimg=wavevec, toair=True)
        thismag = -2.5 * np.log10(fthru) - (48.6 - 2.5 * 17)
    else:
        pass
    # !!!!!! IMPORTANT !!!!!!!!!
    # !!!!!! THIS IS NOT THE ONLY PLACE WHERE THE MODEL SPECTRUM IS NORMALIZED.!!!!
    # !!!!!! For MaStar plates, we renormalize them again after putting in
    #        individual extinction.!!!!!

    # Compute SCALEFAC = (plugmap flux) / (flux of the model spectrum)
    if photometry == "gaiadr1":
        scalefac = 10.0 ** ((thismag - psfmag[1]) / 2.5)
        kindx.mag[1] = psfmag[1]
    elif photometry == "gaiadr2":
        scalefac = 10.0 ** ((thismag[1] - psfmag[1]) / 2.5)
        kindx.mag[1:3] = (thismag[1:3]).flatten() + psfmag[1] - thismag[1]
    elif photometry == "ps1":
        scalefac = 10.0 ** ((thismag[2] - psfmag[2]) / 2.5)
        kindx.mag[1:4] = (thismag[1:4]).flatten() + psfmag[2] - thismag[2]
    elif photometry == "sdss":
        scalefac = 10.0 ** ((thismag[2] - psfmag[2]) / 2.5)
        kindx.mag = (thismag).flatten() + psfmag[2] - thismag[2]

    thismodel = thismodel * scalefac

    modflux = thismodel
    splog, prelog = ""
    # !p.multi = 0
    return kindx, modflux


# -------------------------------------------------------------------------------------------------
