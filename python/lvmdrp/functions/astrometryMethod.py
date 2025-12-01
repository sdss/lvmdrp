# encoding: utf-8
#
# @Author: Guillermo Blanc, Niv Drory, Alfredo Mejía-Narváez
# @Date: Dec 1, 2025
# @Filename: astrometryMethod.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy import units as u
from skyfield.api import Topos

from lvmdrp import log
from lvmdrp.core.image import loadImage
from lvmdrp.core.astrometry import get_telescope_astrometry, set_fibers_astrometry
from lvmdrp.core.constants import FIDUCIAL_PLATESCALE


description = "Provides tasks for calculating astrometry"


__all__ = [
    "add_astrometry",
]


def add_astrometry(in_image: str, out_image: str, in_agcsci_image: str, in_agcskye_image: str, in_agcskyw_image: str):
    """
    uses WCS in AG camera coadd image to calculate RA,DEC of
    each fiber in each telescope and adds these to SLITMAP extension
    if AGC frames are not available it uses the POtelRA,POtelDEC,POtelPA

    Parameters

    in_image : str
        path to input image
    out_image : str
        path to output image
    in_agcsci_image : str
        path to Sci telescope AGC coadd master frame
    in_agcskye_image : str
        path to SkyE telescope AGC coadd master frame
    in_agcskyw_image : str
        path to SkyW telescope AGC coadd master frame
    """

    log.info(f"loading frame from {in_image}")

    # reading slitmap
    org_img = loadImage(in_image)
    slitmap = org_img.getSlitmap()

    # read AGC coadd images and get RAobs, DECobs, and PAobs for each telescope
    in_guider_paths = {'Sci': in_agcsci_image, 'SkyE': in_agcskye_image, 'SkyW': in_agcskyw_image}

    tel_coords = {tel: get_telescope_astrometry(in_guider_paths, tel, org_img) for tel in set(slitmap["telescope"].data)}

    observatory_elevation = 2380.0 * u.m
    observatory_lat = '29.0146S'
    observatory_lon = '70.6926W'
    observatory_topo = Topos(observatory_lat, observatory_lon, elevation_m=observatory_elevation.value)

    # define location of LCO using shadow heigh calculator library
    observatory_location = EarthLocation(lat=observatory_topo.latitude.degrees*u.deg,
                                         lon=observatory_topo.longitude.degrees*u.deg,
                                         height=observatory_elevation)

    #find alt-az frame/coordinates for observation
    altaz_frame = AltAz(obstime=Time(org_img._header["OBSTIME"]), location=observatory_location)

    #use astropy SkyCoord class
    sci_coord = SkyCoord(*tel_coords["Sci"][:2], unit='deg')
    skye_coord = SkyCoord(*tel_coords["SkyE"][:2], unit='deg')
    skyw_coord = SkyCoord(*tel_coords["SkyW"][:2], unit='deg')

    # altitude of objects above the horizon (alt, 0 -- 90)
    sci_alt = sci_coord.transform_to(altaz_frame).alt.value
    skye_alt = skye_coord.transform_to(altaz_frame).alt.value
    skyw_alt = skyw_coord.transform_to(altaz_frame).alt.value

    # define airmasses based on the final astrometric solution
    sci_airmass = 1 / np.cos((90-sci_alt)*np.pi/180)
    skye_airmass = 1 / np.cos((90-skye_alt)*np.pi/180)
    skyw_airmass = 1 / np.cos((90-skyw_alt)*np.pi/180)

    # Create fake IFU image WCS object for each telescope focal plane and use it to calculate RA,DEC of each fiber
    log.info(f'Using Fiducial Platescale = {FIDUCIAL_PLATESCALE:.2f} "/mm')
    set_fibers_astrometry(slitmap, tel_coords, 'Sci', platescale=FIDUCIAL_PLATESCALE)
    set_fibers_astrometry(slitmap, tel_coords, 'SkyE', platescale=FIDUCIAL_PLATESCALE)
    set_fibers_astrometry(slitmap, tel_coords, 'SkyW', platescale=FIDUCIAL_PLATESCALE)
    # BUG: fiber coordinates for Spec telescope are wrong, need to fix using STD* keywords
    # getfibradec(slitmap, tel_coords, 'Spec', platescale=FIDUCIAL_PLATESCALE)
    # add coordinates to slitmap
    org_img.setSlitmap(slitmap)

    # set header keyword with best knowledge of IFU center for SCI, SKYE, SKYW
    org_img.setHdrValue('SCIRA', tel_coords["Sci"][0], 'SCI center, fiberid=975, RA (ASTRMSRC)[deg]')
    org_img.setHdrValue('SCIDEC', tel_coords["Sci"][1], 'SCI center, fiberid=975, DEC (ASTRMSRC)[deg]')
    org_img.setHdrValue('SCIPA', tel_coords["Sci"][2], 'SCI center, fiberid=975, PA (ASTRMSRC)[deg]')
    org_img.setHdrValue('SCIALT', sci_alt, 'SCI center, ALT (ASTRMSRC)[deg]')
    org_img.setHdrValue('SCIAM', sci_airmass, 'SCI center, Airmass (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYERA', tel_coords["SkyE"][0], 'SKYE center, fiberid=36, RA (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYEDEC', tel_coords["SkyE"][1], 'SKYE center, fiberid=36, DEC (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYEPA', tel_coords["SkyE"][2], 'SKYE center, fiberid=36, PA (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYEALT', skye_alt, 'SKYE center, ALT (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYEAM', skye_airmass, 'SKYE center, Airmass (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYWRA', tel_coords["SkyW"][0], 'SKYW center, fiberid=1, RA (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYWDEC', tel_coords["SkyW"][1], 'SKYW center, fiberid=1, DEC (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYWPA', tel_coords["SkyW"][2], 'SKYW center, fiberid=1, PA (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYWALT', skyw_alt, 'SKYW center, ALT (ASTRMSRC)[deg]')
    org_img.setHdrValue('SKYWAM', skyw_airmass, 'SKYW center, Airmass (ASTRMSRC)[deg]')

    log.info(f"writing RA,DEC to slitmap in image '{out_image}'")
    org_img.writeFitsData(out_image)