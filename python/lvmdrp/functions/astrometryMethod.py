# encoding: utf-8
#
# @Author: Guillermo Blanc, Niv Drory, Alfredo Mejía-Narváez
# @Date: Dec 1, 2025
# @Filename: astrometryMethod.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


from lvmdrp import log
from lvmdrp.core.image import loadImage
from lvmdrp.core.astrometry import load_guider_header, set_telescope_astrometry, set_fibers_astrometry, set_altaz_params
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

    for tel in {"Sci", "SkyE", "SkyW"}:
        guider_hdr = load_guider_header(in_guider_paths[tel])

        set_telescope_astrometry(tel, guider_hdr, org_img)
        set_altaz_params(tel, org_img)

    # add coordinates to slitmap
    log.info(f'Using Fiducial Platescale = {FIDUCIAL_PLATESCALE:.2f} "/mm')
    set_fibers_astrometry(org_img, 'Sci', platescale=FIDUCIAL_PLATESCALE)
    set_fibers_astrometry(org_img, 'SkyE', platescale=FIDUCIAL_PLATESCALE)
    set_fibers_astrometry(org_img, 'SkyW', platescale=FIDUCIAL_PLATESCALE)
    org_img.setSlitmap(slitmap)

    log.info(f"writing astrometry to image '{out_image}'")
    org_img.writeFitsData(out_image)