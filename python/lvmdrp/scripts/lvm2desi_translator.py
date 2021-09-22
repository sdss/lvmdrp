"""
lvmdrp.scripts.lvm2desi_translator
========================

Command line wrapper for LVM laboratory data translation to DESI datamodel
"""

import argparse
import collections

import numpy as np
from copy import deepcopy as copy
from astropy.io import fits
from desispec import io
from desispec.parallel import default_nproc
from desiutil.log import get_logger
log = get_logger()

# TODO: move these constants into a configuration file easy to edit (e.g., a yaml file) -----------
LVM2DESI_RAW_HEADER_MAP = {
    (0,"SIMPLE"): (1,"SIMPLE"),
    (0,"BITPIX"): (1,"BITPIX"),
    (0,"NAXIS"): (1,"NAXIS"),
    (0,"NAXIS1"): (1,"NAXIS1"),
    (0,"NAXIS2"): (1,"NAXIS2"),
    (0,"BSCALE"): (1,"BSCALE"),
    (0,"BZERO"): (1,"BZERO"),
    (0,"SPEC"): (1,"SPECGRPH"),
    (0,"OBSERVAT"): (0,"OBSERVAT"),
    (0,"OBSTIME"): (0,"DATE-OBS"),
    (0,"EXPTIME"): (1,"EXPTIME"),
    (0,"IMAGETYP"): (1,"FLAVOR"),
    (0,"INTSTART"): (1,"DATE-OBS"),
#     (0,"INTEND"): (1,"INTEND"),
#     (0,"BINNING"): (1,"BINNING"),
    (0,"CCD"): (1,"CAMERA"),
}
LVM2DESI_RAW_DATA_MAP = {
    0:1
}
LVM2DESI_RAW_HEADER_TRANS = {
    "SPEC": lambda v: int(v[-1]),
}
# -------------------------------------------------------------------------------------------------

def parse(options=None):
    parser = argparse.ArgumentParser(
        description="Translate LVM laboratory data to DESI datamodel",
        epilog='''''')
    parser.add_argument('infile',
                        help = 'path of LVM FITS file')
    parser.add_argument('template',
                        help = 'reference DESI template')

    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args

def main(args=None):
    if args is None:
        args = parse()
    elif isinstance(args, (list, tuple)):
        args = parse(args)
    
    # LVM_ORPHANS = find_orphans(input_fits=args.infile, header_ihdus_keys=LVM2DESI_RAW_HEADER_MAP.keys())
    # DESI_ORPHANS = find_orphans(input_fits=args.template, header_ihdus_keys=LVM2DESI_RAW_HEADER_MAP.values())
    
    lvm_fits = fits.open(args.infile)
    desi_template = fits.open(args.template)

    translated = translator(
        lvm_fits=lvm_fits,
        desi_template=desi_template,
        header_map=LVM2DESI_RAW_HEADER_MAP,
        data_map=LVM2DESI_RAW_DATA_MAP,
        header_transformer=LVM2DESI_RAW_HEADER_TRANS
    )
    translated.info()

class OrderedSet(collections.Set):
    """A set without automatic sorting
    
    taken from: https://bit.ly/3AEtp9i"""
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)

def find_orphans(input_fits, header_ihdus_keys):
    """Return a dictionary with keys=(ihdu, key) and values='label' for missing cards in 'header_ihdus_keys'
    
    Parameters:
    -----------
    input_fits: astropy.io.fits.HDUList instance
        FITS file where to find orphan header cards
    header_ihdus_keys: list
        a list of tuples=(ihdu,key) for the reference header cards
    
    Returns:
    --------
    orphans: list
        list of orphan header keys
    """
    ihdus, keys = zip(*header_ihdus_keys)
    orphans = []
    for ihdu, lvm_hdu in enumerate(input_fits):
        hdu_mask = np.array(ihdus)==ihdu
        orphan_keys = OrderedSet(lvm_hdu.header.keys()) - OrderedSet(np.array(keys)[hdu_mask])
        orphans.extend(zip([ihdu]*len(orphan_keys), orphan_keys))
    return orphans

def translator(lvm_fits, desi_template, header_map, header_transformer, data_map):
    """Return a translated version of the 'lvm_fits' using 'desi_template'
    
    Parameters:
    -----------
    lvm_fits: astropy.io.fits.HDUList instance
        the target LVM FITS file to translate into DESI
    desi_template: astropy.io.fits.HDUList instance
        the DESI template to translate to
    header_map: dict
        a dictionary holding the mapping between LVM and DESI header cards
    header_translator: dict
        a dictionary with translation of data types from LVM to DESI
    data_map: dict
        a dictionary to translate data (e.g., image, cubes) from one header to other
    
    Returns:
    --------
    desi_fits: astropy.io.fits.HDUList instance
        the translated LVM to DESI FITS file
    """
    desi_fits = copy(desi_template)
    
    # TODO: update HDU names
    # TODO: update redundant information in several HDUs
    # update header taking into account transformers
    for (lvm_ihdu,lvm_key), (desi_ihdu,desi_key) in header_map.items():
        trans = header_transformer.get(lvm_key, lambda v: v)
        desi_fits[desi_ihdu].header[desi_key] = trans(lvm_fits[lvm_ihdu].header[lvm_key])
    
    # TODO: update prescan and overscan regions
    # update data images/cubes
    for lvm_ihdu, desi_ihdu in data_map.items():
        desi_fits[desi_ihdu].data = lvm_fits[lvm_ihdu].data
    
    # TODO: implement tables update
    # TODO: translate filenames

    return desi_fits