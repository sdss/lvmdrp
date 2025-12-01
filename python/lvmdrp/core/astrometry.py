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
from warnings import warn
import numpy as np


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


def load_coadded_guider(in_guider_path):
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
        warn(f"astrometric solution not found")
        return
    return mfheader


def get_telescope_astrometry(in_guider_paths, tel, img):
    """Get observed RA, DEC, PA for a given telescope from guider coadd or commanded position

    Parameters
    ----------
    in_guider_paths : dict
        Dictionary with file paths to coadded guider images for each telescope
    tel : str
        Telescope identifier
    img : lvmdrp.core.image.Image
        Image object to copy header keywords to

    Returns
    -------
    tuple
        (RAobs, DECobs, PAobs) observed coordinates and position angle for the given telescope
    """
    if tel == "Spec":
        return 0.0, 0.0, 0.0

    mfheader = load_coadded_guider(in_guider_paths[tel])
    if mfheader is None:
        RAobs = img._header.get(f'PO{tel}RA'.capitalize(), 0.0) or 0.0
        DECobs = img._header.get(f'PO{tel}DE'.capitalize(), 0.0) or 0.0
        PAobs = img._header.get(f'PO{tel}PA'.capitalize(), 0.0) or 0.0
        if -999.0 in [RAobs, DECobs]:
            RAobs, DECobs, PAobs = 0.0, 0.0, 0.0
        if np.any([RAobs, DECobs, PAobs]) == 0.0:
            warn(f"some astrometry keywords for telescope '{tel}' are missing: {RAobs = }, {DECobs = }, {PAobs = }")
            img.add_header_comment(f"no astromentry keywords '{tel}': {RAobs = }, {DECobs = }, {PAobs = }, using commanded")
        img.setHdrValue('ASTRMSRC', 'CMD position', comment='source of astrometry: commanded position')
        return RAobs, DECobs, PAobs

    outw = wcs.WCS(mfheader)
    CDmatrix = outw.pixel_scale_matrix
    posangrad = -1 * np.arctan(CDmatrix[1, 0] / CDmatrix[0, 0])
    PAobs = posangrad * 180 / np.pi
    IFUcencoords = outw.pixel_to_world(2500, 1000)
    RAobs = IFUcencoords.ra.value
    DECobs = IFUcencoords.dec.value
    img.setHdrValue('ASTRMSRC', 'GDR coadd', comment='source of astrometry: guider')
    for kw in GUIDER_IMG_KEYWORDS:
        copy_guider_keyword(mfheader, kw, img)
    return RAobs, DECobs, PAobs


def set_fibers_astrometry(slitmap, tel_coords, tel, platescale):
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
    # Add RA,DEC columns to slitmap if they do not exist
    if "ra" or "dec" not in slitmap.colnames:
        slitmap['ra'] = np.zeros(len(slitmap)) * u.deg
        slitmap['dec'] = np.zeros(len(slitmap)) * u.deg

    selection = slitmap['telescope'].data == tel
    x = slitmap['xpmm'].data[selection]
    y = slitmap['ypmm'].data[selection]

    RAobs, DECobs, PAobs = tel_coords[tel]
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