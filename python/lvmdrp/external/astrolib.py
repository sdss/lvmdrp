####
# This code is a collection of routines taken from the astrolib project
####


import numpy as np


_radeg = 180.0 / np.pi


def premat(equinox1, equinox2, fk4=False):
    """
    NAME:
          PREMAT
    PURPOSE:
          Return the precession matrix needed to go from EQUINOX1 to EQUINOX2.
    EXPLANTION:
          This matrix is used by the procedures PRECESS and BARYVEL to precess
          astronomical coordinates

    CALLING SEQUENCE:
          matrix = PREMAT( equinox1, equinox2, [ /FK4 ] )

    INPUTS:
          EQUINOX1 - Original equinox of coordinates, numeric scalar.
          EQUINOX2 - Equinox of precessed coordinates.

    OUTPUT:
         matrix - double precision 3 x 3 precession matrix, used to precess
                  equatorial rectangular coordinates

    OPTIONAL INPUT KEYWORDS:
          /FK4   - If this keyword is set, the FK4 (B1950.0) system precession
                  angles are used to compute the precession matrix.   The
                  default is to use FK5 (J2000.0) precession angles

    EXAMPLES:
          Return the precession matrix from 1950.0 to 1975.0 in the FK4 system

          IDL> matrix = PREMAT( 1950.0, 1975.0, /FK4)

    PROCEDURE:
          FK4 constants from "Computational Spherical Astronomy" by Taff (1983),
          p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
          Supplement 1992, page 104 Table 3.211.1.

    REVISION HISTORY
          Written, Wayne Landsman, HSTX Corporation, June 1994
          Converted to IDL V5.0   W. Landsman   September 1997
    """

    deg_to_rad = np.pi / 180.0e0
    sec_to_rad = deg_to_rad / 3600.0e0

    t = 0.001e0 * (equinox2 - equinox1)

    if not fk4:
        st = 0.001e0 * (equinox1 - 2000.0e0)
        #  Compute 3 rotation angles
        a = (
            sec_to_rad
            * t
            * (
                23062.181e0
                + st * (139.656e0 + 0.0139e0 * st)
                + t * (30.188e0 - 0.344e0 * st + 17.998e0 * t)
            )
        )

        b = sec_to_rad * t * t * (79.280e0 + 0.410e0 * st + 0.205e0 * t) + a

        c = (
            sec_to_rad
            * t
            * (
                20043.109e0
                - st * (85.33e0 + 0.217e0 * st)
                + t * (-42.665e0 - 0.217e0 * st - 41.833e0 * t)
            )
        )

    else:
        st = 0.001e0 * (equinox1 - 1900.0e0)
        #  Compute 3 rotation angles

        a = (
            sec_to_rad
            * t
            * (
                23042.53e0
                + st * (139.75e0 + 0.06e0 * st)
                + t * (30.23e0 - 0.27e0 * st + 18.0e0 * t)
            )
        )

        b = sec_to_rad * t * t * (79.27e0 + 0.66e0 * st + 0.32e0 * t) + a

        c = (
            sec_to_rad
            * t
            * (
                20046.85e0
                - st * (85.33e0 + 0.37e0 * st)
                + t * (-42.67e0 - 0.37e0 * st - 41.8e0 * t)
            )
        )

    sina = np.np.sin(a)
    sinb = np.np.sin(b)
    sinc = np.np.sin(c)
    cosa = np.cos(a)
    cosb = np.cos(b)
    cosc = np.cos(c)

    r = np.zeroes((3, 3))
    r[0, :] = np.array(
        [
            cosa * cosb * cosc - sina * sinb,
            sina * cosb + cosa * sinb * cosc,
            cosa * sinc,
        ]
    )
    r[1, :] = np.array(
        [
            -cosa * sinb - sina * cosb * cosc,
            cosa * cosb - sina * sinb * cosc,
            -sina * sinc,
        ]
    )
    r[2, :] = np.array([-cosb * sinc, -sinb * sinc, cosc])

    return r


def precess(ra0, dec0, equinox1, equinox2, doprint=None, fk4=None, radian=False):
    """
    NAME:
         PRECESS
    PURPOSE:
         Precess coordinates from EQUINOX1 to EQUINOX2.
    EXPLANATION:
         For interactive display, one can use the procedure ASTRO which calls
         PRECESS or use the /PRINT keyword.   The default (RA,DEC) system is
         FK5 based on epoch J2000.0 but FK4 based on B1950.0 is available via
         the /FK4 keyword.

         Use BPRECESS and JPRECESS to convert between FK4 and FK5 systems
    CALLING SEQUENCE:
         PRECESS, ra, dec, [ equinox1, equinox2, /PRINT, /FK4, /RADIAN ]

    INPUT - OUTPUT:
         RA - Input right ascension (scalar or vector) in DEGREES, unless the
                 /RADIAN keyword is set
         DEC - Input declination in DEGREES (scalar or vector), unless the
                 /RADIAN keyword is set

         The input RA and DEC are modified by PRECESS to give the
         values after precession.

    OPTIONAL INPUTS:
         EQUINOX1 - Original equinox of coordinates, numeric scalar.  If
                  omitted, then PRECESS will query for EQUINOX1 and EQUINOX2.
         EQUINOX2 - Equinox of precessed coordinates.

    OPTIONAL INPUT KEYWORDS:
         /PRINT - If this keyword is set and non-zero, then the precessed
                  coordinates are displayed at the terminal.    Cannot be used
                  with the /RADIAN keyword
         /FK4   - If this keyword is set and non-zero, the FK4 (B1950.0) system
                  will be used otherwise FK5 (J2000.0) will be used instead.
         /RADIAN - If this keyword is set and non-zero, then the input and
                  output RA and DEC vectors are in radians rather than degrees

    RESTRICTIONS:
          Accuracy of precession decreases for declination values near 90
          degrees.  PRECESS should not be used more than 2.5 centuries from
          2000 on the FK5 system (1950.0 on the FK4 system).

    EXAMPLES:
          (1) The Pole Star has J2000.0 coordinates (2h, 31m, 46.3s,
                  89d 15' 50.6"); compute its coordinates at J1985.0

          IDL> precess, ten(2,31,46.3)*15, ten(89,15,50.6), 2000, 1985, /PRINT

                  ====> 2h 16m 22.73s, 89d 11' 47.3"

          (2) Precess the B1950 coordinates of Eps Ind (RA = 21h 59m,33.053s,
          DEC = (-56d, 59', 33.053") to equinox B1975.

          IDL> ra = ten(21, 59, 33.053)*15
          IDL> dec = ten(-56, 59, 33.053)
          IDL> precess, ra, dec ,1950, 1975, /fk4

    PROCEDURE:
          Algorithm from Computational Spherical Astronomy by Taff (1983),
          p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
          Supplement 1992, page 104 Table 3.211.1.

    PROCEDURE CALLED:
          Function PREMAT - computes precession matrix

    REVISION HISTORY
          Written, Wayne Landsman, STI Corporation  August 1986
          Correct negative output RA values   February 1989
          Added /PRINT keyword      W. Landsman   November, 1991
          Provided FK5 (J2000.0)  I. Freedman   January 1994
          Precession Matrix computation now in PREMAT   W. Landsman June 1994
          Added /RADIAN keyword                         W. Landsman June 1997
          Converted to IDL V5.0   W. Landsman   September 1997
          Correct negative output RA values when /RADIAN used    March 1999
          Work for arrays, not just vectors  W. Landsman    September 2003
    """
    if isinstance(ra0, np.ndarray):
        ra = ra0.copy()
        dec = dec0.copy()
        npts = min(ra.size, dec.size)
    else:
        ra = np.array([ra0])
        dec = np.array([dec0])
        npts = 1
    deg_to_rad = np.pi / 180.0e0

    if npts == 0:
        print("ERROR - Input RA and DEC must be vectors or scalars")
        return

    if not radian:
        ra_rad = ra * deg_to_rad  # Convert to double precision if not already
        dec_rad = dec * deg_to_rad
    else:
        ra_rad = np.array(ra).astype(float)
        dec_rad = np.array(dec).astype(float)

    a = np.cos(dec_rad)

    _expr = npts  # Is RA a vector or scalar?

    if _expr == 1:
        x = np.concatenate(
            [a * np.cos(ra_rad), a * np.np.sin(ra_rad), np.np.sin(dec_rad)]
        )  # input direction
    else:
        x = np.zeroes((npts, 3))
        x[:, 0] = a * np.cos(ra_rad)
        x[:, 1] = a * np.np.sin(ra_rad)
        x[:, 2] = np.np.sin(dec_rad)
        x = np.transpose(x)

    # Use PREMAT function to get precession matrix from Equinox1 to Equinox2

    r = premat(equinox1, equinox2, fk4=fk4)

    x2 = np.transpose(
        np.dot(np.transpose(r), np.transpose(x))
    )  # rotate to get output direction cosines

    if npts == 1:  # Scalar
        ra_rad = np.arctan2(x2[1], x2[0])
        dec_rad = np.arcsin(x2[2])

    else:  # Vector
        ra_rad = np.zeroes(npts) + np.arctan2(x2[:, 1], x2[:, 0])
        dec_rad = np.zeroes(npts) + np.arcsin(x2[:, 2])

    if not radian:
        ra = ra_rad / deg_to_rad
        ra = ra + (ra < 0.0) * 360.0e0  # RA between 0 and 360 degrees
        dec = dec_rad / deg_to_rad
    else:
        ra = ra_rad
        dec = dec_rad
        ra = ra + (ra < 0.0) * 2.0e0 * np.pi

    #   if array:
    #      ra = reform(ra, dimen, over=True)
    #      dec = reform(dec, dimen, over=True)

    if doprint is not None:
        print("Equinox (" + np.strtrim(equinox2, 2) + "): ", np.adstring(ra, dec, 1))

    return ra, dec


def daycnv(xjd):
    """
    NAME:
          DAYCNV
    PURPOSE:
          Converts Julian dates to Gregorian calendar dates

    CALLING SEQUENCE:
          DAYCNV, XJD, YR, MN, DAY, HR

    INPUTS:
          XJD = Julian date, positive double precision scalar or vector

    OUTPUTS:
          YR = Year (Integer)
          MN = Month (Integer)
          DAY = Day (Integer)
          HR = Hours and fractional hours (Real).   If XJD is a vector,
                  then YR,MN,DAY and HR will be vectors of the same length.

    EXAMPLE:
          IDL> DAYCNV, 2440000.D, yr, mn, day, hr

          yields yr = 1968, mn =5, day = 23, hr =12.

    WARNING:
          Be sure that the Julian date is specified as double precision to
          maintain accuracy at the fractional hour level.

    METHOD:
          Uses the algorithm of Fliegel and Van Flandern (1968) as reported in
          the "Explanatory Supplement to the Astronomical Almanac" (1992), p. 604
          Works for all Gregorian calendar dates with XJD > 0, i.e., dates after
          -4713 November 23.
    REVISION HISTORY:
          Converted to IDL from Yeoman's Comet Ephemeris Generator,
          B. Pfarr, STX, 6/16/88
          Converted to IDL V5.0   W. Landsman   September 1997
    """

    def _ret():
        return (xjd, yr, mn, day, hr)

    # ON_ERROR, 2

    #      print 'Syntax - DAYCNV, xjd, yr, mn, day, hr'
    #      print '  Julian date, xjd, should be specified in double precision'
    #      return None

    # Adjustment needed because Julian day starts at noon, calendar day at midnight

    jd = np.array(xjd).astype(int)  # Truncate to integral day
    frac = np.array(xjd).astype(float) - jd + 0.5  # Fractional part of calendar day
    after_noon = frac >= 1.0
    next = 1
    if next > 0:  # Is it really the next calendar day?
        if after_noon.any():
            if frac.ndim > 0:
                frac[after_noon] = frac[after_noon] - 1.0
            else:
                frac = frac - 1.0
            if jd.ndim > 0:
                jd[after_noon] = jd[after_noon] + 1
            else:
                jd = jd + 1
    hr = frac * 24.0
    tmp = jd + 68569
    n = 4 * tmp / 146097
    tmp = tmp - (146097 * n + 3) / 4
    yr = 4000 * (tmp + 1) / 1461001
    tmp = tmp - 1461 * yr / 4 + 31  # 1461 = 365.25 * 4
    mn = 80 * tmp / 2447
    day = tmp - 2447 * mn / 80
    tmp = mn / 11
    mn = mn + 2 - 12 * tmp
    yr = 100 * (n - 49) + yr + tmp
    return (yr, mn, day, hr)


def bprecess(ra0, dec0, mu_radec=None, parallax=None, rad_vel=None, epoch=None):
    """
    NAME:
          BPRECESS
    PURPOSE:
          Precess positions from J2000.0 (FK5) to B1950.0 (FK4)
    EXPLANATION:
          Calculates the mean place of a star at B1950.0 on the FK4 system from
          the mean place at J2000.0 on the FK5 system.

    CALLING SEQUENCE:
          bprecess, ra, dec, ra_1950, dec_1950, [ MU_RADEC = , PARALLAX =
                                          RAD_VEL =, EPOCH =   ]

    INPUTS:
          RA,DEC - Input J2000 right ascension and declination in *degrees*.
                  Scalar or N element vector

    OUTPUTS:
          RA_1950, DEC_1950 - The corresponding B1950 right ascension and
                  declination in *degrees*.    Same number of elements as
                  RA,DEC but always double precision.

    OPTIONAL INPUT-OUTPUT KEYWORDS
          MU_RADEC - 2xN element double precision vector containing the proper
                     motion in seconds of arc per tropical *century* in right
                     ascension and declination.
          PARALLAX - N_element vector giving stellar parallax (seconds of arc)
          RAD_VEL  - N_element vector giving radial velocity in km/s

          The values of MU_RADEC, PARALLAX, and RADVEL will all be modified
          upon output to contain the values of these quantities in the
          B1950 system.  The parallax and radial velocity will have a very
          minor influence on the B1950 position.

          EPOCH - scalar giving epoch of original observations, default 2000.0d
              This keyword value is only used if the MU_RADEC keyword is not set.
    NOTES:
          The algorithm is taken from the Explanatory Supplement to the
          Astronomical Almanac 1992, page 186.
          Also see Aoki et al (1983), A&A, 128,263

          BPRECESS distinguishes between the following two cases:
          (1) The proper motion is known and non-zero
          (2) the proper motion is unknown or known to be exactly zero (i.e.
                  extragalactic radio sources).   In this case, the reverse of
                  the algorithm in Appendix 2 of Aoki et al. (1983) is used to
                  ensure that the output proper motion is  exactly zero. Better
                  precision can be achieved in this case by inputting the EPOCH
                  of the original observations.

          The error in using the IDL procedure PRECESS for converting between
          B1950 and J1950 can be up to 12", mainly in right ascension.   If
          better accuracy than this is needed then BPRECESS should be used.

          An unsystematic comparison of BPRECESS with the IPAC precession
          routine (http://nedwww.ipac.caltech.edu/forms/calculator.html) always
          gives differences less than 0.15".
    EXAMPLE:
          The SAO2000 catalogue gives the J2000 position and proper motion for
          the star HD 119288.   Find the B1950 position.

          RA(2000) = 13h 42m 12.740s      Dec(2000) = 8d 23' 17.69''
          Mu(RA) = -.0257 s/yr      Mu(Dec) = -.090 ''/yr

          IDL> mu_radec = 100D* [ -15D*.0257, -0.090 ]
          IDL> ra = ten(13, 42, 12.740)*15.D
          IDL> dec = ten(8, 23, 17.69)
          IDL> bprecess, ra, dec, ra1950, dec1950, mu_radec = mu_radec
          IDL> print, adstring(ra1950, dec1950,2)
                  ===> 13h 39m 44.526s    +08d 38' 28.63"

    REVISION HISTORY:
          Written,    W. Landsman                October, 1992
          Vectorized, W. Landsman                February, 1994
          Treat case where proper motion not known or exactly zero  November 1994
          Handling of arrays larger than 32767   Lars L. Christensen, march, 1995
          Converted to IDL V5.0   W. Landsman   September 1997
          Fixed bug where A term not initialized for vector input
               W. Landsman        February 2000

    """

    scal = True
    if isinstance(ra0, np.ndarray):
        ra = ra0
        dec = dec0
        n = ra.size
        scal = False
    else:
        n = 1
        ra = np.array([ra0])
        dec = np.array([dec0])

    if n == 0:
        np.message("ERROR - First parameter (RA vector) is undefined")

    if rad_vel is None:
        rad_vel = np.zeroes(n)
    else:
        rad_vel = rad_vel * 1.0
        if np.array(rad_vel).size != n:
            print(
                "ERROR - RAD_VEL keyword vector must contain "
                + np.strtrim(n, 2)
                + " values"
            )

    if mu_radec is not None:
        if np.array(mu_radec).size != 2 * n:
            np.message(
                "ERROR - MU_RADEC keyword (proper motion) be dimensioned (2,"
                + np.strtrim(n, 2)
                + ")"
            )
        mu_radec = mu_radec * 1.0

    if parallax is None:
        parallax = np.zeroes(n)
    else:
        parallax = parallax * 1.0

    if epoch is None:
        epoch = 2000.0e0

    radeg = 180.0e0 / np.pi
    sec_to_radian = 1.0e0 / radeg / 3600.0e0

    m = np.array(
        [
            np.array(
                [
                    +0.9999256795e0,
                    -0.0111814828e0,
                    -0.0048590040e0,
                    -0.000551e0,
                    -0.238560e0,
                    +0.435730e0,
                ]
            ),
            np.array(
                [
                    +0.0111814828e0,
                    +0.9999374849e0,
                    -0.0000271557e0,
                    +0.238509e0,
                    -0.002667e0,
                    -0.008541e0,
                ]
            ),
            np.array(
                [
                    +0.0048590039e0,
                    -0.0000271771e0,
                    +0.9999881946e0,
                    -0.435614e0,
                    +0.012254e0,
                    +0.002117e0,
                ]
            ),
            np.array(
                [
                    -0.00000242389840e0,
                    +0.00000002710544e0,
                    +0.00000001177742e0,
                    +0.99990432e0,
                    -0.01118145e0,
                    -0.00485852e0,
                ]
            ),
            np.array(
                [
                    -0.00000002710544e0,
                    -0.00000242392702e0,
                    +0.00000000006585e0,
                    +0.01118145e0,
                    +0.99991613e0,
                    -0.00002716e0,
                ]
            ),
            np.array(
                [
                    -0.00000001177742e0,
                    +0.00000000006585e0,
                    -0.00000242404995e0,
                    +0.00485852e0,
                    -0.00002717e0,
                    +0.99996684e0,
                ]
            ),
        ]
    )

    a_dot = 1e-3 * np.array([1.244e0, -1.579e0, -0.660e0])  # in arc seconds per century

    ra_rad = ra / radeg
    dec_rad = dec / radeg
    cosra = np.cos(ra_rad)
    sinra = np.np.sin(ra_rad)
    cosdec = np.cos(dec_rad)
    sindec = np.np.sin(dec_rad)

    dec_1950 = dec * 0.0
    ra_1950 = ra * 0.0

    for i in np.arange(0, (n - 1) + (1)):
        # Following statement moved inside loop in Feb 2000.
        a = 1e-6 * np.array([-1.62557e0, -0.31919e0, -0.13843e0])  # in radians

        r0 = np.array([cosra[i] * cosdec[i], sinra[i] * cosdec[i], sindec[i]])

        if mu_radec is not None:
            mu_a = mu_radec[i, 0]
            mu_d = mu_radec[i, 1]
            r0_dot = (
                np.array(
                    [
                        -mu_a * sinra[i] * cosdec[i] - mu_d * cosra[i] * sindec[i],
                        mu_a * cosra[i] * cosdec[i] - mu_d * sinra[i] * sindec[i],
                        mu_d * cosdec[i],
                    ]
                )
                + 21.095e0 * rad_vel[i] * parallax[i] * r0
            )

        else:
            r0_dot = np.array([0.0e0, 0.0e0, 0.0e0])

        r_0 = np.concatenate((r0, r0_dot))
        r_1 = np.transpose(np.dot(np.transpose(m), np.transpose(r_0)))

        # Include the effects of the E-terms of aberration to form r and r_dot.

        r1 = r_1[0:3]
        r1_dot = r_1[3:6]

        if mu_radec is None:
            r1 = r1 + sec_to_radian * r1_dot * (epoch - 1950.0e0) / 100.0
            a = a + sec_to_radian * a_dot * (epoch - 1950.0e0) / 100.0

        x1 = r_1[0]
        y1 = r_1[1]
        z1 = r_1[2]
        rmag = np.sqrt(x1**2 + y1**2 + z1**2)

        s1 = r1 / rmag
        s1_dot = r1_dot / rmag

        s = s1
        for j in np.arange(0, 3):
            r = s1 + a - ((s * a).sum()) * s
            s = r / rmag
        x = r[0]
        y = r[1]
        z = r[2]
        r2 = x**2 + y**2 + z**2
        rmag = np.sqrt(r2)

        if mu_radec is not None:
            r_dot = s1_dot + a_dot - ((s * a_dot).sum()) * s
            x_dot = r_dot[0]
            y_dot = r_dot[1]
            z_dot = r_dot[2]
            mu_radec[i, 0] = (x * y_dot - y * x_dot) / (x**2 + y**2)
            mu_radec[i, 1] = (z_dot * (x**2 + y**2) - z * (x * x_dot + y * y_dot)) / (
                r2 * np.sqrt(x**2 + y**2)
            )

        dec_1950[i] = np.arcsin(z / rmag)
        ra_1950[i] = np.arctan2(y, x)

        if parallax[i] > 0.0:
            rad_vel[i] = (x * x_dot + y * y_dot + z * z_dot) / (
                21.095 * parallax[i] * rmag
            )
            parallax[i] = parallax[i] / rmag

    neg = ra_1950 < 0
    if neg.any() > 0:
        ra_1950[neg] = ra_1950[neg] + 2.0e0 * np.pi

    ra_1950 = ra_1950 * radeg
    dec_1950 = dec_1950 * radeg

    # Make output scalar if input was scalar

    #  sz = size(ra)
    #  if sz[0] == 0:
    #     ra_1950 = ra_1950[0]     ;      dec_1950 = dec_1950[0]
    if scal:
        return ra_1950[0], dec_1950[0]
    else:
        return ra_1950, dec_1950


def precess_xyz(x, y, z, equinox1, equinox2):
    # +
    # NAME:
    # 	PRECESS_XYZ
    #
    # PURPOSE:
    # 	Precess equatorial geocentric rectangular coordinates.
    #
    # CALLING SEQUENCE:
    # 	precess_xyz, x, y, z, equinox1, equinox2
    #
    # INPUT/OUTPUT:
    # 	x,y,z: scalars or vectors giving heliocentric rectangular coordinates
    #              THESE ARE CHANGED UPON RETURNING.
    # INPUT:
    # 	EQUINOX1: equinox of input coordinates, numeric scalar
    #       EQUINOX2: equinox of output coordinates, numeric scalar
    #
    # OUTPUT:
    # 	x,y,z are changed upon return
    #
    # NOTES:
    #   The equatorial geocentric rectangular coords are converted
    #      to RA and Dec, precessed in the normal way, then changed
    #      back to x, y and z using unit vectors.
    #
    # EXAMPLE:
    # 	Precess 1950 equinox coords x, y and z to 2000.
    # 	IDL> precess_xyz,x,y,z, 1950, 2000
    #
    # HISTORY:
    # 	Written by P. Plait/ACC March 24 1999
    # 	   (unit vectors provided by D. Lindler)
    #       Use /Radian call to PRECESS     W. Landsman     November 2000
    #       Use two parameter call to ATAN   W. Landsman    June 2001
    # -
    # check inputs
    n_params = 5

    def _ret():
        return (x, y, z, equinox1, equinox2)

    if n_params != 5:
        print("Syntax - PRECESS_XYZ,x,y,z,equinox1,equinox2")
        return _ret()

    # take input coords and convert to ra and dec (in radians)

    ra = np.atan(y, x)
    _del = np.sqrt(x * x + y * y + z * z)  # magnitude of distance to Sun
    dec = np.asin(z / _del)

    #   precess the ra and dec
    precess(ra, dec, equinox1, equinox2, radian=True)

    # convert back to x, y, z
    xunit = np.cos(ra) * np.cos(dec)
    yunit = np.np.sin(ra) * np.cos(dec)
    zunit = np.np.sin(dec)

    x = xunit * _del
    y = yunit * _del
    z = zunit * _del

    return _ret()


def xyz(date, equinox=None):
    """
    NAME:
          XYZ
    PURPOSE:
          Calculate geocentric X,Y, and Z  and velocity coordinates of the Sun
    EXPLANATION:
          Calculates geocentric X,Y, and Z vectors and velocity coordinates
          (dx, dy and dz) of the Sun.   (The positive X axis is directed towards
          the equinox, the y-axis, towards the point on the equator at right
          ascension 6h, and the z axis toward the north pole of the equator).
          Typical position accuracy is <1e-4 AU (15000 km).

    CALLING SEQUENCE:
          XYZ, date, x, y, z, [ xvel, yvel, zvel, EQUINOX = ]

    INPUT:
          date: reduced julian date (=JD - 2400000), scalar or vector

    OUTPUT:
          x,y,z: scalars or vectors giving heliocentric rectangular coordinates
                    (in A.U) for each date supplied.    Note that sqrt(x^2 + y^2
                    + z^2) gives the Earth-Sun distance for the given date.
          xvel, yvel, zvel: velocity vectors corresponding to X, Y and Z.

    OPTIONAL KEYWORD INPUT:
          EQUINOX: equinox of output. Default is 1950.

    EXAMPLE:
          What were the rectangular coordinates and velocities of the Sun on
          Jan 22, 1999 0h UT (= JD 2451200.5) in J2000 coords? NOTE:
          Astronomical Almanac (AA) is in TDT, so add 64 seconds to
          UT to convert.

          IDL> xyz,51200.5+64.d/86400.d,x,y,z,xv,yv,zv,equinox = 2000

          Compare to Astronomical Almanac (1999 page C20)
                      X  (AU)        Y  (AU)     Z (AU)
          XYZ:      0.51456871   -0.76963263  -0.33376880
          AA:       0.51453130   -0.7697110   -0.3337152
          abs(err): 0.00003739    0.00007839   0.00005360
          abs(err)
              (km):   5609          11759         8040

          NOTE: Velocities in AA are for Earth/Moon barycenter
                (a very minor offset) see AA 1999 page E3
                     X VEL (AU/DAY) YVEL (AU/DAY)   Z VEL (AU/DAY)
          XYZ:      -0.014947268   -0.0083148382    -0.0036068577
          AA:       -0.01494574    -0.00831185      -0.00360365
          abs(err):  0.000001583    0.0000029886     0.0000032077
          abs(err)
           (km/sec): 0.00265        0.00519          0.00557

    PROCEDURE CALLS:
          PRECESS_XYZ
    REVISION HISTORY
          Original algorithm from Almanac for Computers, Doggett et al. USNO 1978
          Adapted from the book Astronomical Photometry by A. Henden
          Written  W. Landsman   STX       June 1989
          Correct error in X coefficient   W. Landsman HSTX  January 1995
          Added velocities, more terms to positions and EQUINOX keyword,
             some minor adjustments to calculations
             P. Plait/ACC March 24, 1999
    """

    n_params = 7
    #   def _ret():
    #      _optrv = zip(_opt, [equinox])
    #      _rv = [date, x, y, z, xvel, yvel, zvel]
    #      _rv += [_o[1] for _o in _optrv if _o[0] is not None]
    #      return tuple(_rv)

    # ON_ERROR, 2

    if n_params == 0:
        print("Syntax - XYZ, date, x, y, z, [ xvel, yvel, zvel, EQUINOX= ]")
        print("     (date is REDUCED Julian date (JD - 2400000.0) )")
        return

    picon = np.pi / 180.0e0
    t = (date - 15020.0e0) / 36525.0e0  # Relative Julian century from 1900

    # NOTE: longitude arguments below are given in *equinox* of date.
    #   Precess these to equinox 1950 to give everything an even footing.
    #   Compute argument of precession from equinox of date back to 1950
    pp = (1.396041e0 + 0.000308e0 * (t + 0.5e0)) * (t - 0.499998e0)

    # Compute mean solar longitude, precessed back to 1950
    el = 279.696678e0 + 36000.76892e0 * t + 0.000303e0 * t * t - pp

    # Compute Mean longitude of the Moon
    c = 270.434164e0 + 480960.0e0 * t + 307.883142e0 * t - 0.001133e0 * t * t - pp

    # Compute longitude of Moon's ascending node
    n = 259.183275e0 - 1800.0e0 * t - 134.142008e0 * t + 0.002078e0 * t * t - pp

    # Compute mean solar anomaly
    g = 358.475833e0 + 35999.04975e0 * t - 0.00015e0 * t * t

    # Compute the mean jupiter anomaly
    j = 225.444651e0 + 2880.0e0 * t + 154.906654e0 * t * t

    # Compute mean anomaly of Venus
    v = 212.603219e0 + 58320.0e0 * t + 197.803875e0 * t + 0.001286e0 * t * t

    # Compute mean anomaly of Mars
    m = 319.529425e0 + 19080.0e0 * t + 59.8585e0 * t + 0.000181e0 * t * t

    # Convert degrees to radians for trig functions
    el = el * picon
    g = g * picon
    j = j * picon
    c = c * picon
    v = v * picon
    n = n * picon
    m = m * picon

    # Calculate X,Y,Z using trigonometric series
    x = (
        0.999860e0 * np.cos(el)
        - 0.025127e0 * np.cos(g - el)
        + 0.008374e0 * np.cos(g + el)
        + 0.000105e0 * np.cos(g + g + el)
        + 0.000063e0 * t * np.cos(g - el)
        + 0.000035e0 * np.cos(g + g - el)
        - 0.000026e0 * np.np.sin(g - el - j)
        - 0.000021e0 * t * np.cos(g + el)
        + 0.000018e0 * np.np.sin(2.0e0 * g + el - 2.0e0 * v)
        + 0.000017e0 * np.cos(c)
        - 0.000014e0 * np.cos(c - 2.0e0 * el)
        + 0.000012e0 * np.cos(4.0e0 * g + el - 8.0e0 * m + 3.0e0 * j)
        - 0.000012e0 * np.cos(4.0e0 * g - el - 8.0e0 * m + 3.0e0 * j)
        - 0.000012e0 * np.cos(g + el - v)
        + 0.000011e0 * np.cos(2.0e0 * g + el - 2.0e0 * v)
        + 0.000011e0 * np.cos(2.0e0 * g - el - 2.0e0 * j)
    )

    y = (
        0.917308e0 * np.np.sin(el)
        + 0.023053e0 * np.np.sin(g - el)
        + 0.007683e0 * np.np.sin(g + el)
        + 0.000097e0 * np.sin(g + g + el)
        - 0.000057e0 * t * np.sin(g - el)
        - 0.000032e0 * np.sin(g + g - el)
        - 0.000024e0 * np.cos(g - el - j)
        - 0.000019e0 * t * np.sin(g + el)
        - 0.000017e0 * np.cos(2.0e0 * g + el - 2.0e0 * v)
        + 0.000016e0 * np.sin(c)
        + 0.000013e0 * np.sin(c - 2.0e0 * el)
        + 0.000011e0 * np.sin(4.0e0 * g + el - 8.0e0 * m + 3.0e0 * j)
        + 0.000011e0 * np.sin(4.0e0 * g - el - 8.0e0 * m + 3.0e0 * j)
        - 0.000011e0 * np.sin(g + el - v)
        + 0.000010e0 * np.sin(2.0e0 * g + el - 2.0e0 * v)
        - 0.000010e0 * np.sin(2.0e0 * g - el - 2.0e0 * j)
    )

    z = (
        0.397825e0 * np.sin(el)
        + 0.009998e0 * np.sin(g - el)
        + 0.003332e0 * np.sin(g + el)
        + 0.000042e0 * np.sin(g + g + el)
        - 0.000025e0 * t * np.sin(g - el)
        - 0.000014e0 * np.sin(g + g - el)
        - 0.000010e0 * np.cos(g - el - j)
    )

    # Precess_to new equator?
    if equinox is not None:
        precess_xyz(x, y, z, 1950, equinox)

    if n_params <= 3:
        return

    xvel = (
        -0.017200e0 * np.sin(el)
        - 0.000288e0 * np.sin(g + el)
        - 0.000005e0 * np.sin(2.0e0 * g + el)
        - 0.000004e0 * np.sin(c)
        + 0.000003e0 * np.sin(c - 2.0e0 * el)
        + 0.000001e0 * t * np.sin(g + el)
        - 0.000001e0 * np.sin(2.0e0 * g - el)
    )

    yvel = (
        0.015780 * np.cos(el)
        + 0.000264 * np.cos(g + el)
        + 0.000005 * np.cos(2.0e0 * g + el)
        + 0.000004 * np.cos(c)
        + 0.000003 * np.cos(c - 2.0e0 * el)
        - 0.000001 * t * np.cos(g + el)
    )

    zvel = (
        0.006843 * np.cos(el)
        + 0.000115 * np.cos(g + el)
        + 0.000002 * np.cos(2.0e0 * g + el)
        + 0.000002 * np.cos(c)
        + 0.000001 * np.cos(c - 2.0e0 * el)
    )

    # Precess to new equator?

    if equinox is not None:
        precess_xyz.precess_xyz(xvel, yvel, zvel, 1950, equinox)

    return x, y, z, xvel, yvel, zvel


def helio_jd(date, ra, dec, b1950=False, time_diff=False):
    """
    NAME:
         HELIO_JD
    PURPOSE:
         Convert geocentric (reduced) Julian date to heliocentric Julian date
    EXPLANATION:
         This procedure correct for the extra light travel time between the Earth
         and the Sun.

          An online calculator for this quantity is available at
          http://www.physics.sfasu.edu/astro/javascript/hjd.html
    CALLING SEQUENCE:
          jdhelio = HELIO_JD( date, ra, dec, /B1950, /TIME_DIFF)

    INPUTS
          date - reduced Julian date (= JD - 2400000), scalar or vector, MUST
                  be double precision
          ra,dec - scalars giving right ascension and declination in DEGREES
                  Equinox is J2000 unless the /B1950 keyword is set

    OUTPUTS:
          jdhelio - heliocentric reduced Julian date.  If /TIME_DIFF is set, then
                    HELIO_JD() instead returns the time difference in seconds
                    between the geocentric and heliocentric Julian date.

    OPTIONAL INPUT KEYWORDS
          /B1950 - if set, then input coordinates are assumed to be in equinox
                   B1950 coordinates.
          /TIME_DIFF - if set, then HELIO_JD() returns the time difference
                   (heliocentric JD - geocentric JD ) in seconds

    EXAMPLE:
          What is the heliocentric Julian date of an observation of V402 Cygni
          (J2000: RA = 20 9 7.8, Dec = 37 09 07) taken June 15, 1973 at 11:40 UT?

          IDL> juldate, [1973,6,15,11,40], jd      ;Get geocentric Julian date
          IDL> hjd = helio_jd( jd, ten(20,9,7.8)*15., ten(37,9,7) )

          ==> hjd = 41848.9881

    Wayne Warren (Raytheon ITSS) has compared the results of HELIO_JD with the
    FORTRAN subroutines in the STARLINK SLALIB library (see
    http://star-www.rl.ac.uk/).
                                                     Time Diff (sec)
         Date               RA(2000)   Dec(2000)  STARLINK      IDL

    1999-10-29T00:00:00.0  21 08 25.  -67 22 00.  -59.0        -59.0
    1999-10-29T00:00:00.0  02 56 33.4 +00 26 55.  474.1        474.1
    1940-12-11T06:55:00.0  07 34 41.9 -00 30 42.  366.3        370.2
    1992-02-29T03:15:56.2  12 56 27.4 +42 10 17.  350.8        350.9
    2000-03-01T10:26:31.8  14 28 36.7 -20 42 11.  243.7        243.7
    2100-02-26T09:18:24.2  08 26 51.7 +85 47 28.  104.0        108.8
    PROCEDURES CALLED:
          bprecess, xyz, zparcheck

    REVISION HISTORY:
          Algorithm from the book Astronomical Photometry by Henden, p. 114
          Written,   W. Landsman       STX     June, 1989
          Make J2000 default equinox, add B1950, /TIME_DIFF keywords, compute
          variation of the obliquity      W. Landsman   November 1999
    """

    # Because XYZ uses default B1950 coordinates, we'll convert everything to B1950

    if not b1950:
        ra1, dec1 = bprecess(ra, dec)
    else:
        ra1 = ra
        dec1 = dec

    radeg = 180.0 / np.pi
    #   zparcheck('HELIO_JD', date, 1, concatenate([3, 4, 5]), concatenate([0, 1]), 'Reduced Julian Date')

    delta_t = (np.array(date).astype(float) - 33282.42345905e0) / 36525.0e0
    epsilon_sec = np.poly1d([44.836e0, -46.8495, -0.00429, 0.00181][::-1])(delta_t)
    epsilon = (23.433333e0 + epsilon_sec / 3600.0e0) / radeg
    ra1 = ra1 / radeg
    dec1 = dec1 / radeg

    x, y, z, tmp, tmp, tmp = xyz(date)

    # Find extra distance light must travel in AU, multiply by 1.49598e13 cm/AU,
    # and divide by the speed of light, and multiply by 86400 second/year

    time = -499.00522e0 * (
        np.cos(dec1) * np.cos(ra1) * x
        + (np.tan(epsilon) * np.sin(dec1) + np.cos(dec1) * np.sin(ra1)) * y
    )
    if time_diff:
        return time
    else:
        return np.array(date).astype(float) + time / 86400.0e0


def baryvel(dje, deq=0):
    """
    NAME:
          BARYVEL
    PURPOSE:
          Calculates heliocentric and barycentric velocity components of Earth.

    EXPLANATION:
          BARYVEL takes into account the Earth-Moon motion, and is useful for
          radial velocity work to an accuracy of  ~1 m/s.

    CALLING SEQUENCE:
          dvel_hel, dvel_bary = baryvel(dje, deq)

    INPUTS:
          DJE - (scalar) Julian ephemeris date.
          DEQ - (scalar) epoch of mean equinox of dvelh and dvelb. If deq=0
                  then deq is assumed to be equal to dje.
    OUTPUTS:
          DVELH: (vector(3)) heliocentric velocity component. in km/s
          DVELB: (vector(3)) barycentric velocity component. in km/s

          The 3-vectors DVELH and DVELB are given in a right-handed coordinate
          system with the +X axis toward the Vernal Equinox, and +Z axis
          toward the celestial pole.

    OPTIONAL KEYWORD SET:
          JPL - if /JPL set, then BARYVEL will call the procedure JPLEPHINTERP
                to compute the Earth velocity using the full JPL ephemeris.
                The JPL ephemeris FITS file JPLEPH.405 must exist in either the
                current directory, or in the directory specified by the
                environment variable ASTRO_DATA.   Alternatively, the JPL keyword
                can be set to the full path and name of the ephemeris file.
                A copy of the JPL ephemeris FITS file is available in
                    http://idlastro.gsfc.nasa.gov/ftp/data/
    PROCEDURES CALLED:
          Function PREMAT() -- computes precession matrix
          JPLEPHREAD, JPLEPHINTERP, TDB2TDT - if /JPL keyword is set
    NOTES:
          Algorithm taken from FORTRAN program of Stumpff (1980, A&A Suppl, 41,1)
          Stumpf claimed an accuracy of 42 cm/s for the velocity.    A
          comparison with the JPL FORTRAN planetary ephemeris program PLEPH
          found agreement to within about 65 cm/s between 1986 and 1994

          If /JPL is set (using JPLEPH.405 ephemeris file) then velocities are
          given in the ICRS system; otherwise in the FK4 system.
    EXAMPLE:
          Compute the radial velocity of the Earth toward Altair on 15-Feb-1994
             using both the original Stumpf algorithm and the JPL ephemeris

          IDL> jdcnv, 1994, 2, 15, 0, jd          ;==> JD = 2449398.5
          IDL> baryvel, jd, 2000, vh, vb          ;Original algorithm
                  ==> vh = [-17.07243, -22.81121, -9.889315]  ;Heliocentric km/s
                  ==> vb = [-17.08083, -22.80471, -9.886582]  ;Barycentric km/s
          IDL> baryvel, jd, 2000, vh, vb, /jpl   ;JPL ephemeris
                  ==> vh = [-17.07236, -22.81126, -9.889419]  ;Heliocentric km/s
                  ==> vb = [-17.08083, -22.80484, -9.886409]  ;Barycentric km/s

          IDL> ra = ten(19,50,46.77)*15/!RADEG    ;RA  in radians
          IDL> dec = ten(08,52,3.5)/!RADEG        ;Dec in radians
          IDL> v = vb[0]*np.cos(dec)*np.cos(ra) + $   ;Project velocity toward star
                  vb[1]*np.cos(dec)*np.sin(ra) + vb[2]*np.sin(dec)

    REVISION HISTORY:
          Jeff Valenti,  U.C. Berkeley    Translated BARVEL.FOR to IDL.
          W. Landsman, Cleaned up program sent by Chris McCarthy (SfSU) June 1994
          Converted to IDL V5.0   W. Landsman   September 1997
          Added /JPL keyword  W. Landsman   July 2001
          Documentation update W. Landsman Dec 2005
          Converted to Python S. Koposov 2009-2010
    """

    # Define constants
    dc2pi = 2 * np.pi
    cc2pi = 2 * np.pi
    dc1 = 1.0e0
    dcto = 2415020.0e0
    dcjul = 36525.0e0  # days in Julian year
    dcbes = 0.313e0
    dctrop = 365.24219572e0  # days in tropical year (...572 insig)
    dc1900 = 1900.0e0
    au = 1.4959787e8

    # Constants dcfel(i,k) of fast changing elements.
    dcfel = np.array(
        [
            1.7400353e00,
            6.2833195099091e02,
            5.2796e-6,
            6.2565836e00,
            6.2830194572674e02,
            -2.6180e-6,
            4.7199666e00,
            8.3997091449254e03,
            -1.9780e-5,
            1.9636505e-1,
            8.4334662911720e03,
            -5.6044e-5,
            4.1547339e00,
            5.2993466764997e01,
            5.8845e-6,
            4.6524223e00,
            2.1354275911213e01,
            5.6797e-6,
            4.2620486e00,
            7.5025342197656e00,
            5.5317e-6,
            1.4740694e00,
            3.8377331909193e00,
            5.6093e-6,
        ]
    )
    dcfel = np.reshape(dcfel, (8, 3))

    # constants dceps and ccsel(i,k) of slowly changing elements.
    dceps = np.array([4.093198e-1, -2.271110e-4, -2.860401e-8])
    ccsel = np.array(
        [
            1.675104e-2,
            -4.179579e-5,
            -1.260516e-7,
            2.220221e-1,
            2.809917e-2,
            1.852532e-5,
            1.589963e00,
            3.418075e-2,
            1.430200e-5,
            2.994089e00,
            2.590824e-2,
            4.155840e-6,
            8.155457e-1,
            2.486352e-2,
            6.836840e-6,
            1.735614e00,
            1.763719e-2,
            6.370440e-6,
            1.968564e00,
            1.524020e-2,
            -2.517152e-6,
            1.282417e00,
            8.703393e-3,
            2.289292e-5,
            2.280820e00,
            1.918010e-2,
            4.484520e-6,
            4.833473e-2,
            1.641773e-4,
            -4.654200e-7,
            5.589232e-2,
            -3.455092e-4,
            -7.388560e-7,
            4.634443e-2,
            -2.658234e-5,
            7.757000e-8,
            8.997041e-3,
            6.329728e-6,
            -1.939256e-9,
            2.284178e-2,
            -9.941590e-5,
            6.787400e-8,
            4.350267e-2,
            -6.839749e-5,
            -2.714956e-7,
            1.348204e-2,
            1.091504e-5,
            6.903760e-7,
            3.106570e-2,
            -1.665665e-4,
            -1.590188e-7,
        ]
    )
    ccsel = np.reshape(ccsel, (17, 3))

    # Constants of the arguments of the short-period perturbations.
    dcargs = np.array(
        [
            5.0974222e0,
            -7.8604195454652e2,
            3.9584962e0,
            -5.7533848094674e2,
            1.6338070e0,
            -1.1506769618935e3,
            2.5487111e0,
            -3.9302097727326e2,
            4.9255514e0,
            -5.8849265665348e2,
            1.3363463e0,
            -5.5076098609303e2,
            1.6072053e0,
            -5.2237501616674e2,
            1.3629480e0,
            -1.1790629318198e3,
            5.5657014e0,
            -1.0977134971135e3,
            5.0708205e0,
            -1.5774000881978e2,
            3.9318944e0,
            5.2963464780000e1,
            4.8989497e0,
            3.9809289073258e1,
            1.3097446e0,
            7.7540959633708e1,
            3.5147141e0,
            7.9618578146517e1,
            3.5413158e0,
            -5.4868336758022e2,
        ]
    )
    dcargs = np.reshape(dcargs, (15, 2))

    # Amplitudes ccamps(n,k) of the short-period perturbations.
    ccamps = np.array(
        [
            -2.279594e-5,
            1.407414e-5,
            8.273188e-6,
            1.340565e-5,
            -2.490817e-7,
            -3.494537e-5,
            2.860401e-7,
            1.289448e-7,
            1.627237e-5,
            -1.823138e-7,
            6.593466e-7,
            1.322572e-5,
            9.258695e-6,
            -4.674248e-7,
            -3.646275e-7,
            1.140767e-5,
            -2.049792e-5,
            -4.747930e-6,
            -2.638763e-6,
            -1.245408e-7,
            9.516893e-6,
            -2.748894e-6,
            -1.319381e-6,
            -4.549908e-6,
            -1.864821e-7,
            7.310990e-6,
            -1.924710e-6,
            -8.772849e-7,
            -3.334143e-6,
            -1.745256e-7,
            -2.603449e-6,
            7.359472e-6,
            3.168357e-6,
            1.119056e-6,
            -1.655307e-7,
            -3.228859e-6,
            1.308997e-7,
            1.013137e-7,
            2.403899e-6,
            -3.736225e-7,
            3.442177e-7,
            2.671323e-6,
            1.832858e-6,
            -2.394688e-7,
            -3.478444e-7,
            8.702406e-6,
            -8.421214e-6,
            -1.372341e-6,
            -1.455234e-6,
            -4.998479e-8,
            -1.488378e-6,
            -1.251789e-5,
            5.226868e-7,
            -2.049301e-7,
            0.0e0,
            -8.043059e-6,
            -2.991300e-6,
            1.473654e-7,
            -3.154542e-7,
            0.0e0,
            3.699128e-6,
            -3.316126e-6,
            2.901257e-7,
            3.407826e-7,
            0.0e0,
            2.550120e-6,
            -1.241123e-6,
            9.901116e-8,
            2.210482e-7,
            0.0e0,
            -6.351059e-7,
            2.341650e-6,
            1.061492e-6,
            2.878231e-7,
            0.0e0,
        ]
    )
    ccamps = np.reshape(ccamps, (15, 5))

    # Constants csec3 and ccsec(n,k) of the secular perturbations in longitude.
    ccsec3 = -7.757020e-8
    ccsec = np.array(
        [
            1.289600e-6,
            5.550147e-1,
            2.076942e00,
            3.102810e-5,
            4.035027e00,
            3.525565e-1,
            9.124190e-6,
            9.990265e-1,
            2.622706e00,
            9.793240e-7,
            5.508259e00,
            1.559103e01,
        ]
    )
    ccsec = np.reshape(ccsec, (4, 3))

    # Sidereal rates.
    dcsld = 1.990987e-7  # sidereal rate in longitude
    ccsgd = 1.990969e-7  # sidereal rate in mean anomaly

    # Constants used in the calculation of the lunar contribution.
    cckm = 3.122140e-5
    ccmld = 2.661699e-6
    ccfdi = 2.399485e-7

    # Constants dcargm(i,k) of the arguments of the perturbations of the motion
    # of the moon.
    dcargm = np.array(
        [
            5.1679830e0,
            8.3286911095275e3,
            5.4913150e0,
            -7.2140632838100e3,
            5.9598530e0,
            1.5542754389685e4,
        ]
    )
    dcargm = np.reshape(dcargm, (3, 2))

    # Amplitudes ccampm(n,k) of the perturbations of the moon.
    ccampm = np.array(
        [
            1.097594e-1,
            2.896773e-7,
            5.450474e-2,
            1.438491e-7,
            -2.223581e-2,
            5.083103e-8,
            1.002548e-2,
            -2.291823e-8,
            1.148966e-2,
            5.658888e-8,
            8.249439e-3,
            4.063015e-8,
        ]
    )
    ccampm = np.reshape(ccampm, (3, 4))

    # ccpamv(k)=a*m*dl,dt (planets), dc1mme=1-mass(earth+moon)
    ccpamv = np.array([8.326827e-11, 1.843484e-11, 1.988712e-12, 1.881276e-12])
    dc1mme = 0.99999696e0

    # Time arguments.
    dt = (dje - dcto) / dcjul
    tvec = np.array([1e0, dt, dt * dt])

    # Values of all elements for the instant(aneous?) dje.
    temp = (np.transpose(np.dot(np.transpose(tvec), np.transpose(dcfel)))) % dc2pi
    dml = temp[0]
    forbel = temp[1:8]
    g = forbel[0]  # old fortran equivalence

    deps = (tvec * dceps).sum() % dc2pi
    sorbel = (np.transpose(np.dot(np.transpose(tvec), np.transpose(ccsel)))) % dc2pi
    e = sorbel[0]  # old fortran equivalence

    # Secular perturbations in longitude.
    sn = np.sin(
        (np.transpose(np.dot(np.transpose(tvec[0:2]), np.transpose(ccsec[:, 1:3]))))
        % cc2pi
    )

    # Periodic perturbations of the emb (earth-moon barycenter).
    pertl = (ccsec[:, 0] * sn).sum() + dt * ccsec3 * sn[2]
    pertld = 0.0
    pertr = 0.0
    pertrd = 0.0
    for k in range(0, 15):
        a = (dcargs[k, 0] + dt * dcargs[k, 1]) % dc2pi
        cosa = np.cos(a)
        sina = np.sin(a)
        pertl = pertl + ccamps[k, 0] * cosa + ccamps[k, 1] * sina
        pertr = pertr + ccamps[k, 2] * cosa + ccamps[k, 3] * sina
        if k < 11:
            pertld = pertld + (ccamps[k, 1] * cosa - ccamps[k, 0] * sina) * ccamps[k, 4]
            pertrd = pertrd + (ccamps[k, 3] * cosa - ccamps[k, 2] * sina) * ccamps[k, 4]

    # Elliptic part of the motion of the emb.
    phi = (e * e / 4e0) * (
        ((8e0 / e) - e) * np.sin(g) + 5 * np.sin(2 * g) + (13 / 3e0) * e * np.sin(3 * g)
    )
    f = g + phi
    sinf = np.sin(f)
    cosf = np.cos(f)
    dpsi = (dc1 - e * e) / (dc1 + e * cosf)
    phid = 2 * e * ccsgd * ((1 + 1.5 * e * e) * cosf + e * (1.25 - 0.5 * sinf * sinf))
    psid = ccsgd * e * sinf / np.sqrt(dc1 - e * e)

    # Perturbed heliocentric motion of the emb.
    d1pdro = dc1 + pertr
    drd = d1pdro * (psid + dpsi * pertrd)
    drld = d1pdro * dpsi * (dcsld + phid + pertld)
    dtl = (dml + phi + pertl) % dc2pi
    dsinls = np.sin(dtl)
    dcosls = np.cos(dtl)
    dxhd = drd * dcosls - drld * dsinls
    dyhd = drd * dsinls + drld * dcosls

    # Influence of eccentricity, evection and variation on the geocentric
    # motion of the moon.
    pertl = 0.0
    pertld = 0.0
    pertp = 0.0
    pertpd = 0.0
    for k in range(0, 3):
        a = (dcargm[k, 0] + dt * dcargm[k, 1]) % dc2pi
        sina = np.sin(a)
        cosa = np.cos(a)
        pertl = pertl + ccampm[k, 0] * sina
        pertld = pertld + ccampm[k, 1] * cosa
        pertp = pertp + ccampm[k, 2] * cosa
        pertpd = pertpd - ccampm[k, 3] * sina

    # Heliocentric motion of the earth.
    tl = forbel[1] + pertl
    sinlm = np.sin(tl)
    coslm = np.cos(tl)
    sigma = cckm / (1.0 + pertp)
    a = sigma * (ccmld + pertld)
    b = sigma * pertpd
    dxhd = dxhd + a * sinlm + b * coslm
    dyhd = dyhd - a * coslm + b * sinlm
    dzhd = -sigma * ccfdi * np.cos(forbel[2])

    # Barycentric motion of the earth.
    dxbd = dxhd * dc1mme
    dybd = dyhd * dc1mme
    dzbd = dzhd * dc1mme
    for k in range(0, 4):
        plon = forbel[k + 3]
        pomg = sorbel[k + 1]
        pecc = sorbel[k + 9]
        tl = (plon + 2.0 * pecc * np.sin(plon - pomg)) % cc2pi
        dxbd = dxbd + ccpamv[k] * (np.sin(tl) + pecc * np.sin(pomg))
        dybd = dybd - ccpamv[k] * (np.cos(tl) + pecc * np.cos(pomg))
        dzbd = dzbd - ccpamv[k] * sorbel[k + 13] * np.cos(plon - sorbel[k + 5])

    # Transition to mean equator of date.
    dcosep = np.cos(deps)
    dsinep = np.sin(deps)
    dyahd = dcosep * dyhd - dsinep * dzhd
    dzahd = dsinep * dyhd + dcosep * dzhd
    dyabd = dcosep * dybd - dsinep * dzbd
    dzabd = dsinep * dybd + dcosep * dzbd

    # Epoch of mean equinox (deq) of zero implies that we should use
    # Julian ephemeris date (dje) as epoch of mean equinox.
    if deq == 0:
        dvelh = au * (np.array([dxhd, dyahd, dzahd]))
        dvelb = au * (np.array([dxbd, dyabd, dzabd]))
        return (dvelh, dvelb)

    # General precession from epoch dje to deq.
    deqdat = (dje - dcto - dcbes) / dctrop + dc1900
    prema = premat(deqdat, deq, fk4=True)

    dvelh = au * (
        np.transpose(
            np.dot(np.transpose(prema), np.transpose(np.array([dxhd, dyahd, dzahd])))
        )
    )
    dvelb = au * (
        np.transpose(
            np.dot(np.transpose(prema), np.transpose(np.array([dxbd, dyabd, dzabd])))
        )
    )

    return (dvelh, dvelb)


def helcorr(obs_long, obs_lat, obs_alt, ra2000, dec2000, jd, debug=False):
    # calculates heliocentric Julian date, baricentric and heliocentric radial
    # velocity corrections from:
    #
    # INPUT:
    # <OBSLON> Longitude of observatory (degrees, western direction is positive)
    # <OBSLAT> Latitude of observatory (degrees)
    # <OBSALT> Altitude of observatory (meters)
    # <RA2000> Right ascension of object for epoch 2000.0 (hours)
    # <DE2000> Declination of object for epoch 2000.0 (degrees)
    # <JD> Julian date for the middle of exposure
    # [DEBUG=] set keyword to get additional results for debugging
    #
    # OUTPUT:
    # <CORRECTION> baricentric correction - correction for rotation of earth,
    #   rotation of earth center about the eart-moon barycenter, eart-moon
    #   barycenter about the center of the Sun.
    # <HJD> Heliocentric Julian date for middle of exposure
    #
    # Algorithms used are taken from the IRAF task noao.astutils.rvcorrect
    # and some procedures of the IDL Astrolib are used as well.
    # Accuracy is about 0.5 seconds in time and about 1 m/s in velocity.
    #
    # History:
    # written by Peter Mittermayer, Nov 8,2003
    # 2005-January-13   Kudryavtsev   Made more accurate calculation of the sideral time.
    #                                Conformity with MIDAS compute/barycorr is checked.
    # 2005-June-20      Kochukhov Included precession of RA2000 and DEC2000 to current epoch

    # covert JD to Gregorian calendar date
    xjd = np.array(2400000.0).astype(float) + jd
    year, month, day, ut = daycnv(xjd)

    # current epoch
    epoch = year + month / 12.0 + day / 365.0

    # precess ra2000 and dec2000 to current epoch
    ra, dec = precess(ra2000 * 15.0, dec2000, 2000.0, epoch)
    # calculate heliocentric julian date
    hjd = np.array(helio_jd(jd, ra, dec)).astype(float)

    # DIURNAL VELOCITY (see IRAF task noao.astutil.rvcorrect)
    # convert geodetic latitude into geocentric latitude to correct
    # for rotation of earth
    dlat = (
        -(11.0 * 60.0 + 32.743) * np.np.sin(2 * obs_lat / _radeg)
        + 1.1633 * np.np.sin(4 * obs_lat / _radeg)
        - 0.0026 * np.np.sin(6 * obs_lat / _radeg)
    )
    lat = obs_lat + dlat / 3600

    # calculate distance of observer from earth center
    r = (
        6378160.0
        * (
            0.998327073
            + 0.001676438 * np.cos(2 * lat / _radeg)
            - 0.00000351 * np.cos(4 * lat / _radeg)
            + 0.000000008 * np.cos(6 * lat / _radeg)
        )
        + obs_alt
    )

    # calculate rotational velocity (perpendicular to the radius vector) in km/s
    # 23.934469591229 is the siderial day in hours for 1986
    v = 2.0 * np.pi * (r / 1000.0) / (23.934469591229 * 3600.0)

    # calculating local mean siderial time (see astronomical almanach)
    tu = (jd - 51545.0) / 36525
    gmst = (
        6.697374558
        + ut
        + (236.555367908 * (jd - 51545.0) + 0.093104 * tu**2 - 6.2e-6 * tu**3) / 3600
    )
    lmst = (gmst - obs_long / 15) % 24

    # projection of rotational velocity along the line of sight
    vdiurnal = (
        v
        * np.cos(lat / _radeg)
        * np.cos(dec / _radeg)
        * np.np.sin((ra - lmst * 15) / _radeg)
    )

    # BARICENTRIC and HELIOCENTRIC VELOCITIES
    vh, vb = baryvel(xjd, 0)

    # project to line of sight
    vbar = (
        vb[0] * np.cos(dec / _radeg) * np.cos(ra / _radeg)
        + vb[1] * np.cos(dec / _radeg) * np.np.sin(ra / _radeg)
        + vb[2] * np.np.sin(dec / _radeg)
    )
    vhel = (
        vh[0] * np.cos(dec / _radeg) * np.cos(ra / _radeg)
        + vh[1] * np.cos(dec / _radeg) * np.np.sin(ra / _radeg)
        + vh[2] * np.np.sin(dec / _radeg)
    )

    bcorr = vdiurnal + vbar  # using baricentric velocity for correction
    hcorr = vdiurnal + vhel  # using heliocentric velocity for correction
    if debug:
        print("")
        print("----- HELCORR.PRO - DEBUG INFO - START ----")
        print(
            "(obs_long,obs_lat,obs_alt) Observatory coordinates [deg,m]: ",
            obs_long,
            obs_lat,
            obs_alt,
        )
        print("(ra,dec) Object coordinates (for epoch 2000.0) [deg]: ", ra, dec)
        print(
            "(ut) Universal time (middle of exposure) [hrs]: ", ut
        )  # , format='(A,F20.12)'
        print(
            "(jd) Julian date (middle of exposure) (JD-2400000): ", jd
        )  # , format='(A,F20.12)'
        print(
            "(hjd) Heliocentric Julian date (middle of exposure) (HJD-2400000): ", hjd
        )  # , format='(A,F20.12)'
        print("(gmst) Greenwich mean siderial time [hrs]: ", gmst % 24)
        print("(lmst) Local mean siderial time [hrs]: ", lmst)
        print("(dlat) Latitude correction [deg]: ", dlat)
        print("(lat) Geocentric latitude of observer [deg]: ", lat)
        print("(r) Distance of observer from center of earth [m]: ", r)
        print(
            "(v) Rotational velocity of earth at the position of the observer [km/s]: ",
            v,
        )
        print(
            "(vdiurnal) Projected earth rotation and earth-moon revolution [km/s]: ",
            vdiurnal,
        )
        print("(vbar) Baricentric velocity [km/s]: ", vbar)
        print("(vhel) Heliocentric velocity [km/s]: ", vhel)
        print("(bcorr) Vdiurnal+vbar [km/s]: ", bcorr)  # , format='(A,F12.9)'
        print("(hcorr) Vdiurnal+vbar [km/s]: ", hcorr)  # , format='(A,F12.9)'
        print("----- HELCORR.PRO - DEBUG INFO - END -----")
        print("")

    return (hcorr, bcorr, hjd)
