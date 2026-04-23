# encoding: utf-8
#
# @Author: Guillermo Blanc, Niv Drory, Alfredo Mejía-Narváez
# @Date: Dec 1, 2025
# @Filename: astrometry.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import os
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from warnings import warn
import numpy as np

from lvmdrp.core.constants import LVM_LAT, LVM_LON, LVM_ELEVATION


GUIDER_IMG_KEYWORDS = [
            'FRAME0  ', 'FRAMEN  ', 'NFRAMES ', 'STACK0  ', 'STACKN  ', 'NSTACKED',
            'COESTIM ', 'SIGCLIP ', 'SIGMA   ', 'OBSTIME0', 'OBSTIMEN',
            'FWHM0   ', 'FWHMN   ', 'FWHMMED ', 'COFWHM  ', 'COFWHMST',
            'PACOEFFA', 'PACOEFFB', 'PAMIN   ', 'PAMAX   ', 'PADRIFT ',
            'ZEROPT  ', 'SOLVED  ', 'WARNPADR', 'WARNTRAN', 'WARNMATC', 'WARNFWHM'
        ]


def copy_guider_keyword(gdrhdr, keyword, img):
    """Copy a keyword from a guider coadd header to an Image object Header

    NOTE: this function modifies the input Image object in place.

    Parameters
    ----------
    gdrhdr : fits.Header
        Guider image header
    keyword : str
        Header keyword to copy
    img : lvmdrp.core.image.Image
        Image object to copy header keywords to
    """
    inhdr = keyword in gdrhdr
    comment = gdrhdr.comments[keyword] if inhdr else ''
    img.setHdrValue(f'HIERARCH GDRCOADD {keyword}', gdrhdr.get(keyword), comment)


def load_guider_header(in_guider_path):
    """Loads in memory coadded guider image

    Parameters
    ----------
    in_guider_path : str
        File path to coadded guider image

    Returns
    -------
    fits.Header
        Header object from second coadded guider image HDU
    """
    if not os.path.isfile(in_guider_path):
        warn(f"coadded guider image not found {in_guider_path}")
        return
    mfheader = fits.getheader(in_guider_path, ext=1)
    if not mfheader.get("SOLVED", False):
        warn("astrometric solution not found")
        return
    return mfheader


def set_telescope_astrometry(telescope, guider_header, img):
    """Sets observed RA, DEC, PA for a given telescope from guider coadd or commanded position

    NOTE: this function will add the astrometric information to `img` in place

    Parameters
    ----------
    telescope : str
        Telescope identifier
    guider_header : fits.Header
        Coadded guider image header
    img : lvmdrp.core.image.Image
        Image object to copy header keywords to

    Returns
    -------
    tuple
        (ra_obs, dec_obs, pa_obs) observed coordinates and position angle for the given telescope
    """
    if telescope == "Spec" or telescope not in {"Sci", "SkyE", "SkyW"}:
        raise ValueError(f"Invalid value for `tel`: {telescope}. Expected either 'Sci', 'SkyE', 'SkyW'")

    slitmap = img.getSlitmap()
    if slitmap is None:
        warn(f"no slitmap information available, skipping fibers astrometry for telescope '{telescope}'")
        return

    hdr = img.getHeader()
    if hdr is None:
        warn(f"no header information available, skipping fibers astrometry for telescope '{telescope}'")
        return

    ifucen_selection = (slitmap["xpmm"] == 0) & (slitmap["ypmm"] == 0) & (slitmap["telescope"] == telescope)
    fiberid = slitmap["fiberid"].data[ifucen_selection][0]

    label = telescope.upper()
    if guider_header is None:
        ra_obs = img._header.get(f'PO{label}RA'.capitalize(), 0.0) or 0.0
        dec_obs = img._header.get(f'PO{label}DE'.capitalize(), 0.0) or 0.0
        pa_obs = img._header.get(f'PO{label}PA'.capitalize(), 0.0) or 0.0
        if -999.0 in [ra_obs, dec_obs]:
            ra_obs, dec_obs, pa_obs = 0.0, 0.0, 0.0
        if np.any([ra_obs, dec_obs, pa_obs]) == 0.0:
            warn(f"some astrometry keywords for telescope '{telescope}' are missing: {ra_obs = }, {dec_obs = }, {pa_obs = }")
            img.add_header_comment(f"no astromentry keywords '{telescope}': {ra_obs = }, {dec_obs = }, {pa_obs = }, using commanded")

        # set header keyword with best knowledge of IFU center for telescope `tel`
        img.setHdrValue(f"{label}RA", ra_obs, f"{telescope} center, fiberid={fiberid}, RA [deg]")
        img.setHdrValue(f"{label}DEC", dec_obs, f"{telescope} center, fiberid={fiberid}, Dec [deg]")
        img.setHdrValue(f"{label}PA", pa_obs, "PA [deg]")
        img.setHdrValue(f'{label}ASRC', 'CMD position', comment=f'{telescope} source of astrometry: commanded position')
        return ra_obs, dec_obs, pa_obs

    outw = wcs.WCS(guider_header)
    CDmatrix = outw.pixel_scale_matrix
    posangrad = -1 * np.arctan(CDmatrix[1, 0] / CDmatrix[0, 0])
    IFUcencoords = outw.pixel_to_world(2500, 1000)

    ra_obs = IFUcencoords.ra.value
    dec_obs = IFUcencoords.dec.value
    pa_obs = posangrad * 180 / np.pi

    img.setHdrValue(f"{label}RA", ra_obs, f"{telescope} center, fiberid={fiberid}, RA [deg]")
    img.setHdrValue(f"{label}DEC", dec_obs, f"{telescope} center, fiberid={fiberid}, Dec [deg]")
    img.setHdrValue(f"{label}PA", pa_obs, "PA [deg]")
    img.setHdrValue(f'{label}ASRC', 'GDR coadd', comment=f'{telescope} source of astrometry: guider')
    for kw in GUIDER_IMG_KEYWORDS:
        copy_guider_keyword(guider_header, kw, img)
    return ra_obs, dec_obs, pa_obs


def set_altaz_params(telescope, img):
    """Sets Alt-Az parameters to `img` header

    NOTE: this function will add information about the observing conditions to `img` in place

    Parameters
    ----------
    telescope : str
        Telescope name
    img : lvmdrp.core.image.Image
        Image object to add observing conditions to

    Raises
    ------
    ValueError
        If the value of `telescope` is not one of 'Sci', 'SkyE', 'SkyW'
    """
    if telescope == "Spec" or telescope not in {"Sci", "SkyE", "SkyW"}:
        raise ValueError(f"Invalid value for `tel`: {telescope}. Expected either 'Sci', 'SkyE', 'SkyW'")

    hdr = img.getHeader()
    if hdr is None:
        warn(f"no header information available, skipping fibers astrometry for telescope '{telescope}'")
        return

    # define LVMi location
    lvm_location = EarthLocation(lat=LVM_LAT, lon=LVM_LON, height=LVM_ELEVATION * u.m)

    # find alt-az frame/coordinates for observation
    altaz_frame = AltAz(obstime=Time(img._header["OBSTIME"]), location=lvm_location)

    # use astropy SkyCoord class
    label = telescope.upper()
    coords = SkyCoord(img._header[f"{label}RA"], img._header[f"{label}DEC"], unit='deg')

    # altitude of objects above the horizon (alt, 0 -- 90)
    alt = coords.transform_to(altaz_frame).alt.value

    # define airmasses based on the final astrometric solution
    airmass = 1 / np.cos((90 - alt) * np.pi / 180)

    img.setHdrValue(f'{label}ALT', alt, f'{telescope} center Alt [deg]')
    img.setHdrValue(f'{label}AM', airmass, f'{telescope} center Airmass')


def set_fibers_astrometry(img, tel, platescale):
    """Calculate and set RA,DEC for each fiber in slitmap for a given telescope

    NOTE: this function modifies the input slitmap Table in place.

    Parameters
    ----------
    slitmap : astropy.table.Table
        Slitmap table with fiber positions
    tel_coords : dict
        Dictionary with observed (RA, DEC, PA) for each telescope
    tel : str
        Telescope identifier
    platescale : float
        Telescope plate scale in arcsec/mm
    """
    if tel not in {"Sci", "SkyE", "SkyW", "Spec"}:
        raise ValueError(f"Invalid value for `tel`: {tel}. Expected either 'Sci', 'SkyE', 'SkyW', 'Spec'")

    slitmap = img.getSlitmap()
    if slitmap is None:
        warn(f"no slitmap information available, skipping fibers astrometry for telescope '{tel}'")
        return

    hdr = img.getHeader()
    if hdr is None:
        warn(f"no header information available, skipping fibers astrometry for telescope '{tel}'")
        return

    # Add RA,DEC columns to slitmap if they do not exist
    if "ra" not in slitmap.colnames or "dec" not in slitmap.colnames:
        slitmap['ra'] = np.zeros(len(slitmap)) * u.deg
        slitmap['dec'] = np.zeros(len(slitmap)) * u.deg

    # add astrometry for standard fibers (special case)
    if tel == "Spec":
        for ifiber in range(15):
            fiberid = img._header.get(f"STD{ifiber+1}FIB")
            ra = img._header.get(f"STD{ifiber+1}RA", 0.0) or 0.0
            dec = img._header.get(f"STD{ifiber+1}DE", 0.0) or 0.0

            std_selection = slitmap["orig_ifulabel"] == fiberid
            slitmap["ra"][std_selection] = ra * u.deg
            slitmap["dec"][std_selection] = dec * u.deg
        return

    # add fibers astrometry for the rest of the telescopes: Sci, SkyE, SkyW
    selection = slitmap['telescope'].data == tel
    x = slitmap['xpmm'].data[selection]
    y = slitmap['ypmm'].data[selection]

    # Create fake IFU image WCS object for each telescope focal plane and use it to calculate RA,DEC of each fiber
    RAobs, DECobs, PAobs = hdr[f"{tel.upper()}RA"], hdr[f"{tel.upper()}DEC"], hdr[f"{tel.upper()}PA"]
    pscale = 0.01  # IFU image pixel scale in mm/pix
    skypscale = pscale * platescale / 3600  # IFU image pixel scale in deg/pix
    npix = 1800  # size of fake IFU image
    w = wcs.WCS(naxis=2)  # IFU image wcs object
    w.wcs.crpix = [int(npix / 2) + 1, int(npix / 2) + 1]
    posangrad = PAobs * np.pi / 180
    w.wcs.cd = np.array([[skypscale * np.cos(posangrad), -1 * skypscale * np.sin(posangrad)], [-1 * skypscale * np.sin(posangrad), -1 * skypscale * np.cos(posangrad)]])
    w.wcs.crval = [RAobs, DECobs]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # Calculate RA,DEC of each individual fiber
    xfib = x / pscale + int(npix / 2)  # pixel x coordinates of fibers
    yfib = y / pscale + int(npix / 2)  # pixel y coordinates of fibers
    fibcoords = w.pixel_to_world(xfib, yfib).to_table()

    # update slitmap with fiber RA,DEC
    slitmap["ra"][selection] = fibcoords['ra'].degree * u.deg
    slitmap["dec"][selection] = fibcoords['dec'].degree * u.deg