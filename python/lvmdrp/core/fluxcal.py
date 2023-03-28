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
from pydl.pydlutils.bspline import bspline
from pydl.pydlutils.sdss import sdss_flagval
from scipy.interpolate import BSpline, interp1d, splrep
from scipy.ndimage import median_filter
from scipy.optimize import minimize, nnls
from scipy.special import eval_chebyc, eval_legendre


f99_ext = F99(Rv=3.1)


# UTILS FUNCTIONS ---------------------------------------------------------------------------------


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/misc/djs_laxisnum.pro
def djs_laxisnum(dimens, iaxis=None):
    #
    # Need one parameter
    #
    # IF N_PARAMS() LT 1 THEN BEGIN
    #     PRINT, 'Syntax - result = djs_laxisnum( dimens, [iaxis= ] )'
    #     RETURN, -1
    # ENDIF

    if iaxis is None:
        iaxis = 0

    ndimen = len(dimens)
    naxis = np.int(dimens)  # convert to type LONG

    if iaxis >= ndimen:
        print("Invalid axis selection!")
        return -1

    result = np.zeros(naxis, dtype=int)

    if ndimen == 1:
        result[:] = 0
    elif ndimen == 2:
        if iaxis == 0:
            for ii in range(0, naxis[0] - 1):
                result[ii, :] = ii
        elif iaxis == 1:
            for ii in range(0, naxis[1] - 1):
                result[:, ii] = ii
    elif ndimen == 3:
        if iaxis == 0:
            for ii in range(0, naxis[0] - 1):
                result[ii, :, :] = ii
        elif iaxis == 1:
            for ii in range(0, naxis[1] - 1):
                result[:, ii, :] = ii
        elif iaxis == 2:
            for ii in range(0, naxis[2] - 1):
                result[:, :, ii] = ii
    else:
        print(ndimen, " dimensions not supported!")
        result = -1

    return result


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/math/djs_reject.pro
def djs_reject(
    ydata,
    ymodel,
    outmask=None,
    inmask=None,
    sigma=None,
    invvar=None,
    upper=None,
    lower=None,
    maxdev=None,
    maxrej=None,
    groupsize=None,
    groupdim=None,
    sticky=None,
    groupbadpix=None,
    grow=None,
):
    #    if (n_params() LT 2 OR NOT arg_present(outmask)) then begin
    #       print, 'Syntax: qdone = djs_reject(ydata, ymodel, outmask=, [ inmask=, $'
    #       print, ' sigma=, invvar=, upper=, lower=, maxdev=, grow= $'
    #       print, ' maxrej=, groupsize=, groupdim=, /sticky, /groupbadpix] )'
    #       return, 1
    #    endif

    ndata = len(ydata)
    if ndata == 0:
        print("No data points")
    if ndata != len(ydata):
        print("Dimensions of YDATA and YMODEL do not agree")

    if inmask is not None:
        if ndata != len(inmask):
            print("Dimensions of YDATA and INMASK do not agree")

    if maxrej is not None:
        if groupdim is not None:
            if len(maxrej) != len(groupdim):
                print("MAXREJ and GROUPDIM must have same number of elements!")
        if groupsize is not None:
            if len(maxrej) != len(groupsize):
                print("MAXREJ and GROUPSIZE must have same number of elements!")
        else:
            groupsize = ndata

    # ----------
    # Create OUTMASK, setting =1 for good data

    if outmask is not None:
        if ndata != len(outmask):
            print("Dimensions of YDATA and OUTMASK do not agree")
    else:
        outmask = np.ones_like(ydata, dtype=bool)

    if ymodel is None:
        if inmask is not None:
            outmask[:] = inmask
        return 0

    if sigma is not None and invvar is not None:
        print("Cannot set both SIGMA and INVVAR")

    if sigma is None and invvar is None:
        if inmask is not None:
            igood = np.where(inmask & outmask)[0]
        else:
            igood = np.where(outmask)
        ngood = igood.size

        if ngood > 1:
            sigma = np.std(ydata[igood] - ymodel[igood])  # scalar value
        else:
            sigma = 0

    #   if (n_elements(sigma) NE 1 AND n_elements(sigma) NE ndata) then $
    #    message, 'Invalid number of elements for SIGMA'

    ydiff = ydata - ymodel

    # ----------
    # The working array is BADNESS, which is set to zero for good points
    # (or points already rejected), and positive values for bad points.
    # The values determine just how bad a point is, either corresponding
    # to the number of SIGMA above or below the fit, or to the number of
    # multiples of MAXDEV away from the fit.

    badness = 0.0 * outmask

    # ----------
    # Decide how bad a point is according to LOWER

    if lower is not None:
        if sigma is not None:
            qbad = ydiff < (-lower * sigma)
            badness = (((-ydiff / (sigma + (sigma == 0))) > 0) * qbad) + badness
        else:
            qbad = ydiff * np.sqrt(invvar) < (-lower)
            badness = (((-ydiff * np.sqrt(invvar)) > 0) * qbad) + badness

    # ----------
    # Decide how bad a point is according to UPPER

    if upper is not None:
        if sigma is not None:
            qbad = ydiff > (upper * sigma)
            badness = (((ydiff / (sigma + (sigma == 0))) > 0) * qbad) + badness
        else:
            qbad = ydiff * np.sqrt(invvar) > upper
            badness = (((ydiff * np.sqrt(invvar)) > 0) * qbad) + badness

    # ----------
    # Decide how bad a point is according to MAXDEV

    if maxdev is not None:
        qbad = np.abs(ydiff) > maxdev
        badness = (np.abs(ydiff) / maxdev * qbad) + badness

    # ----------
    # Do not consider rejecting points that are already rejected by INMASK.
    # Do not consider rejecting points that are already rejected by OUTMASK
    #   if /STICKY is set.

    if inmask is not None:
        badness = badness * inmask
    if sticky is not None:
        badness = badness * outmask

    # ----------
    # Reject a maximum of MAXREJ (additional) points in all the data,
    # or in each group as specified by GROUPSIZE, and optionally along
    # each dimension specified by GROUPDIM.

    if maxrej is not None:
        # Loop over each dimension of GROUPDIM (or loop once if not set)
        for iloop in range(0, (len(groupdim) > 1) - 1):
            # Assign an index number in this dimension to each data point
            if len(groupdim) > 0:
                yndim = len(ydata.shape)
                if groupdim[iloop] > yndim:
                    print("GROUPDIM is larger than number of dimensions for YDATA")
                dimnum = djs_laxisnum(ydata.shape, iaxis=groupdim[iloop] - 1)
            else:
                dimnum = 0

            # Loop over each vector specified by GROUPDIM.  For ex, if
            # this is a 2-D array with GROUPDIM=1, then loop over each
            # column of the data.  If GROUPDIM=2, then loop over each row.
            # If GROUPDIM is not set, then use all whole image.
            for ivec in range(0, np.max(dimnum)):
                if dimnum is not None:
                    indx = np.where(dimnum == ivec)[0]
                else:
                    indx = np.linspace(ndata)

                # Within this group of points, break it down into groups
                # of points specified by GROUPSIZE (if set).
                nin = len(indx)

                if groupbadpix is not None:
                    goodtemp = badness == 0
                    groups_lower = np.where([1, goodtemp] - goodtemp != 1)[0]
                    groups_upper = np.where([goodtemp[1:], 1] - goodtemp == 1)[0]
                    ngroups = len(groups_lower)
                else:
                    if groupsize is None:
                        ngroups = 1
                        groups_lower = 0
                        groups_upper = nin - 1
                    else:
                        ngroups = nin / groupsize + 1
                        groups_lower = np.linspace(ngroups) * groupsize
                        groups_upper = (
                            (np.linspace(ngroups) + 1) * groupsize < nin
                        ) - 1

                for igroup in range(0, ngroups - 1):
                    i1 = groups_lower[igroup]
                    i2 = groups_upper[igroup]
                    nii = i2 - i1 + 1

                    # Need the test that i1 NE -1 below to prevent a crash condition,
                    # but why is it that we ever get groups w/out any points?
                    if nii > 0 and i1 != -1:
                        jj = indx[i1:i2]
                        # Test if too many points rejected in this group...
                        if np.sum(badness[jj] != 0) > maxrej[iloop]:
                            isort = np.argsort(badness[jj])
                            # Make the following points good again...
                            badness[jj[isort[0 : nii - maxrej[iloop] - 1]]] = 0
                        i1 = i1 + groupsize[iloop]

    # ----------
    # Now modify OUTMASK, rejecting points specified by INMASK=0,
    # OUTMASK=0 (if /STICKY is set), or BADNESS>0.

    newmask = badness == 0
    if grow is not None:
        rejects = np.where(newmask == 0)[0]
        if rejects.size != 0:
            for jj in range(1, grow):
                newmask[(rejects - jj) > 0] = 0
                newmask[(rejects + jj) < (ndata - 1)] = 0

    if inmask is not None:
        newmask = newmask & inmask
    if sticky is not None:
        newmask = newmask & outmask

    # Set QDONE if the input OUTMASK is identical to the output OUTMASK
    qdone = np.sum(newmask != outmask) == 0
    outmask = newmask

    return qdone


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/cholesky_band.pro
def cholesky_band(lower, mininf=None, verbose=False):
    if mininf is None:
        mininf = 0.0
    # compute cholesky decomposition of banded matrix
    #   lower[bandwidth, n]  n is the number of linear equations

    # I'm doing lower cholesky decomposition from lapack, spbtf2.f

    bw = lower.shape[0]
    n = lower.shape[1] - bw

    negative = np.where(lower[0, 0 : n - 1] <= mininf)[0]
    if negative.size != 0:
        if verbose:
            print("bad entries")
            print(negative)
        return negative

    kn = bw - 1
    spot = 1 + np.linspace(kn)
    bi = np.linspace(kn)
    for i in range(1, kn - 1):
        bi = [bi, np.linspace(kn - i) + (kn + 1) * i]

    for j in range(0, n - 1):
        lower[0, j] = np.sqrt(lower[0, j])
        lower[spot, j] = lower[spot, j] / lower[0, j]
        x = lower[spot, j]

        if np.all(np.isnif(x)):
            if verbose:
                print("NaN found in cholesky_band")
            return j

        hmm = np.transpose(x) @ x
        here = bi + (j + 1) * bw
        lower[here] = lower[here] - hmm[bi]

    return -1


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/cholesky_solve.pro
def cholesky_solve(a, b):
    bw = a.shape[0]
    n = b.shape[1] - bw

    kd = bw - 1

    ### first round
    spot = np.linspace(kd) + 1
    for j in range(0, n - 1):
        b[j] = b[j] / a[0, j]
        b[j + spot] = b[j + spot] - b[j] * a[spot, j]

    #### second round

    spot = kd - np.linspace(kd)
    for j in range(n - 1, 0, -1):
        b[j] = (b[j] - np.sum(a[spot, j] * b[j + spot])) / a[0, j]

    return -1


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/intrv.pro
def intrv(x, fullbkpt, nbkptord):
    nx = len(x)
    nbkpt = len(fullbkpt)
    n = nbkpt - nbkptord

    indx = np.zeros(nx, dtype=int)

    ileft = nbkptord - 1
    for i in range(0, nx - 1):
        while x[i] > fullbkpt[ileft + 1] and ileft < n - 1:
            ileft = ileft + 1
        indx[i] = ileft

    return indx


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bsplvn.pro
def bsplvn(bkpt, nord, x, ileft):
    #
    # 	Conversion of slatec utility bsplvn, used only by efc
    #
    # 	parameter index is not passed, as efc always calls with 1
    # 	treat x as array for all values between ileft and ileft+1
    ##

    nx = len(x)

    #
    # 	This is to break up really HUGE arrays into manageable chunks
    #
    if nx > 12000000:
        lower = 0
        upper = 6399999
        vnikx = np.zeros(nord) @ x

        while lower < nx:
            # splog, lower, upper, nx
            vnikx[lower:upper, :] = bsplvn(
                bkpt, nord, x[lower:upper], ileft[lower:upper]
            )
            lower = upper + 1
            upper = (upper + 6400000) < (nx - 1)

        return vnikx

    vnikx = np.zeros(nord) @ x
    deltap = vnikx
    deltam = vnikx
    vmprev = x * 0
    vm = x * 0

    j = 0
    vnikx[:, 0] = 1.0

    while j < nord - 1:
        ipj = ileft + j + 1
        deltap[:, j] = bkpt[ipj] - x
        imj = ileft - j
        deltam[:, j] = x - bkpt[imj]
        vmprev = 0.0
        for l in range(0, j):
            vm = vnikx[:, l] / (deltap[:, l] + deltam[:, j - l])
            vnikx[:, l] = vm * deltap[:, l] + vmprev
            vmprev = vm * deltam[:, j - l]

        j = j + 1
        vnikx[:, j] = vmprev

    return vnikx


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bspline_action.pro
def bspline_action(x, sset, x2=None):
    if not isinstance(sset, dict):
        print("Please send in a proper B-spline structure")
        return -1, lower, upper

    npoly = 1
    nx = len(x)

    #
    # 	Check for the existence of x2
    #
    if x2 is not None:
        if len(x2) != nx:
            print("dimensions do not match between x and x2")
            return -1, lower, upper

        if "npoly" in sset.keys():
            npoly = sset["npoly"]

    nord = sset["nord"]
    goodbk = np.where(sset["bkmask"] != 0)
    nbkpt = goodbk.size
    if nbkpt < 2 * nord:
        return -2, lower, upper
    n = nbkpt - nord

    gb = sset["fullbkpt"][goodbk]

    bw = npoly * nord
    action = np.zeros(bw) @ x

    lower = np.zeros(n - nord + 1)
    upper = np.zeros(n - nord + 1) - 1

    indx = intrv(x, gb, nord)

    bf1 = bsplvn(gb, nord, x, indx)
    action = bf1

    # --------------------------------------------------------------
    #  sneaky way to calculate upper and lower indices when
    #   x is sorted
    #
    aa = np.unique(indx, return_index=True)[1]
    upper[indx[aa] - nord + 1] = aa

    rindx = indx[::-1]
    bb = np.unique(rindx, return_index=True)[1]
    lower[rindx[bb] - nord + 1] = nx - bb - 1

    # ---------------------------------------------------------------
    #  just attempt this if 2d fit is required
    #
    if x2 is not None:
        x2norm = 2.0 * (x2[:] - sset["xmin"]) / (sset["xmax"] - sset["xmin"]) - 1.0
        if sset["funcname"] == "poly":
            temppoly = np.ones(npoly) @ (x2norm * 0.0 + 1.0)
            for i in range(1, npoly - 1):
                temppoly[:, i] = temppoly[:, i - 1] * x2norm
        if sset["funcname"] == "poly1":
            temppoly = np.ones(npoly) @ x2norm
            for i in range(1, npoly - 1):
                temppoly[:, i] = temppoly[:, i - 1] * x2norm
        if sset["funcname"] == "chebyshev":
            temppoly = eval_chebyc(npoly - 1, x2norm)
        if sset["funcname"] == "legendre":
            temppoly = eval_legendre(npoly - 1, x2norm)
        else:
            temppoly = eval_legendre(npoly - 1, x2norm)

        action = np.zeros((nx, bw))
        counter = -1
        for ii in range(0, nord - 1):
            for jj in range(0, npoly - 1):
                counter = counter + 1
                action[:, counter] = bf1[:, ii] * temppoly[:, jj]

    return action, lower, upper


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bspline_maskpoints.pro
def bspline_maskpoints(sset, errb, npoly=None):
    if npoly is None:
        npoly = 1

    goodbk = np.where(sset["bkmask"] != 0)[0]
    nbkpt = goodbk.size
    nord = sset["nord"]

    if nbkpt <= 2 * nord:
        return -2

    hmm = errb[np.unique(errb / npoly, return_index=True)[1]] / npoly
    n = nbkpt - nord

    if np.where(hmm >= n)[0].size != 0:
        return -2

    test = np.zeros(nbkpt, dtype=int)
    for jj in range(-np.ceil(nord / 2.0), (nord / 2.0) - 1):
        inside = (((hmm + jj) > 0) + nord) < (n - 1)
        test[inside] = 1

    maskthese = np.where(test == 1)[0]

    if maskthese.size != 0:
        return -2

    reality = goodbk[maskthese]
    if np.sum(sset["bkmask"][reality]) == 0:
        return -2

    sset["bkmask"][reality] = 0
    return -1


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bspline_valu.pro
def bspline_valu(x, sset, x2=None, action=None, upper=None, lower=None):
    nx = len(x)
    mask = np.zeros(nx, dtype=int)

    if not isinstance(sset, dict):
        print("Please send in a proper B-spline structure")
        return np.zeros_like(x), mask

    xsort = np.argsort(x)
    npoly = 1
    xwork = x[xsort]

    if x2 is not None:
        if "npoly" in sset.keys():
            npoly = sset["npoly"]
        x2work = x2[xsort]
    else:
        x2work = 0

    if action is None:
        action, lower, upper = bspline_action(xwork, sset, x2=x2work)

    yfit = x * 0.0
    nord = sset["nord"]
    bw = npoly * nord

    spot = np.linspace(bw)
    goodbk = np.where(sset["bkmask"] != 0)
    nbkpt = goodbk.size
    coeffbk = np.where(sset["bkmask"][nord:] != 0)
    n = nbkpt - nord

    sc = len(sset["coeff"])
    if sc[0] == 2:
        goodcoeff = sset["coeff"][:, coeffbk]
    else:
        goodcoeff = sset["coeff"][coeffbk]

    for i in range(0, n - nord):
        ict = upper[i] - lower[i] + 1

        if ict > 0:
            yfit[lower[i] : upper[i]] = (
                goodcoeff[i * npoly + spot] @ action[lower[i] : upper[i], :]
            )

    yy = yfit
    yy[xsort] = yfit

    mask[:] = 1
    gb = sset["fullbkpt"][goodbk]

    outside = np.where((x < gb[nord - 1]) | (x > gb[n]))[0]
    if outside.size != 0:
        mask[outside] = 0

    hmm = np.where(goodbk[1:] - goodbk > 2)[0]
    nhmm = hmm.size
    for jj in range(0, nhmm - 1):
        inside = np.where(
            (x >= sset["fullbkpt"][goodbk[hmm[jj]]])
            & (x <= sset["fullbkpt"][goodbk[hmm[jj] + 1] - 1])
        )
        if inside.size != 0:
            mask[inside] = 0

    return yy, mask


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/create_bsplineset.pro
def create_bsplineset(fullbkpt, nord, npoly=None):
    numbkpt = len(fullbkpt)
    numcoeff = numbkpt - nord

    if npoly is None:
        sset = {
            "fullbkpt": fullbkpt,
            "bkmask": np.ones(numbkpt, dtype=bool),
            "nord": int(nord),
            "coeff": np.zeros(numcoeff),
            "icoeff": np.zeros(numcoeff),
        }
    else:
        sset = {
            "fullbkpt": fullbkpt,
            "bkmask": np.ones(numbkpt, dtype=bool),
            "nord": int(nord),
            "xmin": 0.0,
            "xmax": 1.0,
            "funcname": "legendre",
            "npoly": int(npoly),
            "coeff": np.zeros((npoly, numcoeff)),
            "icoeff": np.zeros((npoly, numcoeff)),
        }

    return sset


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bspline_bkpts.pro
def bspline_bkpts(
    x,
    nord,
    bkpt=None,
    bkspace=None,
    nbkpts=None,
    everyn=None,
    silent=True,
    bkspread=None,
    placed=None,
):
    nx = len(x)

    if bkpt is None:
        range = np.max(x) - np.min(x)
        startx = np.min(x)
        if placed is not None:
            w = np.where(placed >= startx and placed <= startx + range)[0]
            cnt = w.size
            nbkpts = cnt
            if nbkpts < 2:
                nbkpts = 2
                tempbkspace = np.double(range / (float(nbkpts - 1)))
                bkpt = (np.linspace(nbkpts)) * tempbkspace + startx
            else:
                bkpt = placed[w]
        elif bkspace is not None:
            nbkpts = int(range / float(bkspace)) + 1
            if nbkpts < 2:
                nbkpts = 2
            tempbkspace = np.double(range / (float(nbkpts - 1)))
            bkpt = (np.linspace(nbkpts)) * tempbkspace + startx
        elif nbkpts is not None:
            nbkpts = int(nbkpts)
            if nbkpts < 2:
                nbkpts = 2
            tempbkspace = np.double(range / (float(nbkpts - 1)))
            bkpt = (np.linspace(nbkpts)) * tempbkspace + startx
        elif everyn is not None:
            nbkpts = (nx / everyn) > 1
            if nbkpts == 1:
                xspot = [0]
            else:
                xspot = np.linspace(nbkpts) * (nx / (nbkpts - 1))
            bkpt = x[xspot]
        else:
            print("No information for bkpts")

    bkpt = float(bkpt)

    if np.min(x) < np.min(bkpt):
        spot = np.argmin(bkpt)
        if not silent:
            print("Lowest breakpoint does not cover lowest x value: changing")
        bkpt[spot] = min(x)

    if np.max(x) > np.max(bkpt):
        spot = np.argmax(bkpt)
        if not silent:
            print("highest breakpoint does not cover highest x value, changing")
        bkpt[spot] = max(x)

    nshortbkpt = len(bkpt)
    fullbkpt = bkpt

    if bkspread is None:
        bkspread = 1.0
    if nshortbkpt == 1:
        bkspace = bkspread
    else:
        bkspace = (bkpt[1] - bkpt[0]) * bkspread

    for i in range(1, nord - 1):
        fullbkpt = [bkpt[0] - bkspace * i, fullbkpt, bkpt[nshortbkpt - 1] + bkspace * i]

    return fullbkpt, bkpt


# NOTE: taken from https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bspline_fit.pro
def bspline_fit(
    xdata, ydata, invvar, sset, fullbkpt=None, x2=None, npoly=None, nord=None
):
    if nord is None:
        nord = 4

    if not isinstance(sset, dict):
        sset = create_bsplineset(fullbkpt, nord, npoly=npoly)

    if "npoly" in sset:
        npoly = sset["npoly"]
    if npoly is None:
        npoly = 1

    goodbk = np.where(sset["bkmask"][nord:] != 0)
    nbkpt = goodbk.size

    nord = sset["nord"]

    if nbkpt < nord:
        yfit = np.zeros_like(ydata)
        return -2, sset, yfit

    nn = nbkpt
    nfull = nn * npoly
    bw = npoly * nord  # this is the bandwidth

    #  The next line is REQUIRED to fill a1

    a1, lower, upper = bspline_action(xdata, sset, x2=x2)

    a2 = a1 * (np.ones(bw) @ invvar)

    alpha = np.zeros((bw, nfull + bw), dtype=np.double)
    beta = np.zeros(nfull + bw, dtype=np.double)

    bi = np.linspace(bw)
    bo = np.linspace(bw)
    for i in range(1, bw - 1):
        bi = [bi, np.linspace(bw - i) + (bw + 1) * i]
    for i in range(1, bw - 1):
        bo = [bo, np.linspace(bw - i) + bw * i]

    for i in range(0, nn - nord):
        itop = i * npoly
        ibottom = (itop < nfull + bw) - 1

        ict = upper[i] - lower[i] + 1

        if ict > 0:
            work = a2[lower[i] : upper[i], :] @ np.transpose(a1[lower[i] : upper[i], :])
            wb = a2[lower[i] : upper[i], :] @ ydata[lower[i] : upper[i]]

            alpha[bo + itop * bw] = alpha[bo + itop * bw] + work[bi]
            beta[itop:ibottom] = beta[itop:ibottom] + wb

    # Drop break points where minimal influence is located

    min_influence = 1.0e-10 * np.sum(invvar) / nfull

    # This call to cholesky_band operates on alpha and changes contents

    errb = cholesky_band(alpha, mininf=min_influence)

    if errb[0] != -1:
        yfit, _ = bspline_valu(xdata, sset, x2=x2, action=a1, upper=upper, lower=lower)
        return bspline_maskpoints(sset, errb, npoly), sset, yfit

    # this changes beta to contain the solution

    errs = cholesky_solve(alpha, beta)
    if errs[0] != -1:
        yfit, _ = bspline_valu(xdata, sset, x2=x2, action=a1, upper=upper, lower=lower)
        return bspline_maskpoints(sset, errs, npoly), sset, yfit

    sc = len(sset["coeff"])
    if sc[0] == 2:
        sset["icoeff"][:, goodbk] = np.reshape(
            alpha[0, np.linspace(nfull)], (npoly, nn)
        )
        sset["coeff"][:, goodbk] = np.reshape(beta[np.linspace(nfull)], (npoly, nn))
    else:
        sset.icoeff[goodbk] = alpha[0, np.linspace(nfull)]
        sset.coeff[goodbk] = beta[np.linspace(nfull)]

    yfit, _ = bspline_valu(xdata, sset, x2=x2, action=a1, upper=upper, lower=lower)

    return 0, sset, yfit


# NOTE: taken from: https://svn.sdss.org/public/repo/sdss/idlutils/tags/v5_5_36/pro/bspline/bspline_iterfit.pro
def bspline_iterfit(
    xdata,
    ydata,
    invvar=None,
    nord=None,
    x2=None,
    npoly=None,
    xmin=None,
    xmax=None,
    bkpt=None,
    oldset=None,
    maxiter=None,
    upper=None,
    lower=None,
    requiren=None,
    fullbkpt=None,
    funcname=None,
    **kwargs,
):
    # ----------
    # Check dimensions of inputs

    nx = len(xdata)
    if len(ydata) != nx:
        print("Dimensions of XDATA and YDATA do not agree")

    if nord is None:
        nord = 4
    if upper is None:
        upper = 5
    if lower is None:
        lower = 5

    if invvar is not None:
        if len(invvar) != nx:
            print("Dimensions of XDATA and INVVAR do not agree")

    if x2 is not None:
        if len(x2) != nx:
            print("Dimensions of X and X2 do not agree")
        if npoly is None:
            npoly = 2

    if maxiter is None:
        maxiter = 10

    yfit = np.zeros_like(ydata)  # Default return values

    if invvar is None:
        var = np.std(ydata) ** 2
        if var == 0:
            var = 1
        invvar = np.zeros_like(ydata) + 1.0 / var

    if len(invvar) == 1:
        outmask = np.asarray([True])
    else:
        outmask = np.ones_like(invvar, dtype=bool)

    xsort = np.argsort(xdata)
    maskwork = (outmask * (invvar > 0)).astype(bool)[xsort]
    these = np.where(maskwork)[0]
    nthese = these.size

    # ----------
    # Determine the break points and create output structure

    if oldset is not None:
        sset = oldset
        sset["bkmask"] = 1
        sset["coeff"] = 0
        tags = oldset.keys()
        if "xmin" in tags and x2 is None:
            print("X2 must be set to be consistent with OLDSET")

    else:
        if nthese == 0:
            print("No valid data points")
            fullbkpt = 0
            return None, outmask, fullbkpt, yfit

        if fullbkpt is None:
            fullbkpt = bspline_bkpts(
                xdata[xsort[these]], nord=nord, bkpt=bkpt, **kwargs
            )

        sset = create_bsplineset(fullbkpt, nord, npoly=npoly)

        if nthese < nord:
            print("Number of good data points fewer the NORD")
            return sset, outmask, fullbkpt, yfit

        # ----------
        # Condition the X2 dependent variable by the XMIN, XMAX values.
        # This will typically put X2NORM in the domain [-1,1].

        if x2 is not None:
            if xmin is None:
                xmin = np.min(x2)
            if xmax is None:
                xmax = np.max(x2)
            if xmin == xmax:
                xmax = xmin + 1
            sset["xmin"] = xmin
            sset["xmax"] = xmax

            if funcname is not None:
                sset["funcname"] = funcname

    # ----------
    # It's okay now if the data fall outside breakpoint regions, the
    # fit is just set to zero outside.

    # ----------
    # Sort the data so that X is in ascending order.

    xwork = xdata[xsort]
    ywork = ydata[xsort]
    invwork = invvar[xsort]
    if x2 is not None:
        x2work = x2[xsort]

    # ----------
    # Iterate spline fit

    iiter = 0
    error = 0

    qdone = 0
    while ((error != 0) or (qdone == 0)) and iiter < maxiter:
        ngood = np.sum(maskwork)
        goodbk = np.where(sset["bkmask"] != 0)[0]
        ngb = goodbk.size

        if ngood < 1 or goodbk[0] != -1:
            sset["coeff"] = 0
            iiter = maxiter + 1  # End iterations
        else:
            if requiren is not None:
                # Locate where there are two break points in a row with no good
                # data points in between, and drop (mask) one of those break points.
                # The first break point is kept.
                i = 0
                while xwork[i] < sset["fullbkpt"][goodbk[nord]] and i < nx - 1:
                    i = i + 1

                ct = 0
                for ileft in range(nord, ngb - nord):
                    while (
                        xwork[i] >= sset["fullbkpt"][goodbk[ileft]]
                        and xwork[i] < sset["fullbkpt"][goodbk[ileft + 1]]
                        and i < nx - 1
                    ):
                        ct = ct + (invwork[i] * maskwork[i] > 0)
                        i = i + 1

                    if ct >= requiren:
                        ct = 0
                    else:
                        sset["bkmask"][goodbk[ileft]] = 0

            # Do the fit.  Return values for ERROR are as follows:
            #    0 if fit is good
            #   -1 if all break points are masked
            #   -2 if everything is screwed
            error = bspline_fit(
                xwork,
                ywork,
                invwork * maskwork,
                sset,
                x2=x2work,
                yfit=yfit,
                nord=nord,
                **kwargs,
            )

        iiter = iiter + 1

        inmask = maskwork

        if error == -2:
            # All break points have been dropped.
            return sset, outmask, fullbkpt, yfit
        elif error == 0:
            # Iterate the fit -- next rejection iteration.
            qdone = djs_reject(
                ywork,
                yfit,
                invvar=invwork,
                inmask=inmask,
                outmask=maskwork,
                upper=upper,
                lower=lower,
                **kwargs,
            )

    # ----------
    # Re-sort the output arrays OUTMASK and YFIT to agree with the input data.

    outmask[xsort] = maskwork

    temp = yfit
    yfit[xsort] = temp

    return sset, outmask, fullbkpt, yfit


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
        thisback[npix - 1 - hwidth : npix - 1] = objflux[
            npix - 1 - hwidth : npix - 1, ispec
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
        _, kindx, dslgpsize = spflux_read_kurucz()  ##Yanping test
    elif template == "munari":
        _, kindx, dslgpsize = spflux_read_munari()  ##Yanping added
    elif template == "BOSZ":
        _, kindx, dslgpsize = spflux_read_bosz()  ##Yanping added
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
        )  ##Yanping test
    elif template == "munari":
        modflux, kindx, dslgpsize = spflux_read_munari(
            loglam - np.log10(1 + zpeak), dispimg
        )  ##Yanping added
    elif template == "BOSZ":
        modflux, kindx, dslgpsize = spflux_read_bosz(
            loglam - np.log10(1 + zpeak), dispimg
        )  ##Yanping added
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
    if disp is not None:
        x2 = disp[isort]
    else:
        pass

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
