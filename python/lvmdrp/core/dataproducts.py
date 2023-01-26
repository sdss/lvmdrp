# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 25, 2023
# @Filename: dataproducts.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import yaml
import numpy as np
from astropy.io import fits

from lvmdrp.core.constants import CONFIG_PATH


# path to blueprints, all should be defined in this same path
DATAPRODUCT_BLUEPRINTS_PATH = os.path.join(CONFIG_PATH, "dataproducts")


def load_blueprint(name):
    """
        Returns the blueprint for the LVM-DRP dataproduct given a file path

        Parameters
        ----------
        name: string
            name of the YAML file containing dataproduct blueprint
        
        Returns
        -------
        dict_like
            a dictionary containing a dataproduct definition

    """
    pass


def dump_template(dataproduct_bp, name, path):
    """
        Returns a FITS object following the given blueprint

        Parameters
        ----------
        dataproduct_bp: dict_like
            a dictionary containing a dataproduct definition

        Returns
        -------
        astropy.io.fits.HDUList object
            a FITS template for a particular dataproduct

    """
    pass


def fill_template(fits_template, **kwargs):
    """
        Returns a data product given a FITS template and the corresponding data
    
        This function takes in a FITS template (astropy.io.fits.HDUList) and the
        corresponding data to fill-in the template in the form of keyword arguments.
        
        Each header keyword will be located in the given keyword arguments and left
        as they are if not found. The actual data in each HDU needs to be specified
        by the name of the HDU (e.g., PRIMARY, IVAR, etc). It will be filled with
        NaNs if none passed. A warning will be thrown each time a data structure
        or a header keyword is missing. Keywords can be passed in lowercase.

        Parameters
        ----------
        fits_template: astropy.io.fits.HDUList object
            a FITS object defining the structure of a particular dataproduct
        kwargs:
            the parameters to fill-in the given template
        
        Returns
        -------
        astropy.io.fits.HDUList object
            a FITS object containing a dataproduct (the original FITS template is not modified)
        
    """
    pass


def dump_datamodel(fits_dataproduct, name, path):
    """
        Returns a datamodel structure containing the information in the given dataproduct

        Parameters
        ----------
        fits_dataproduct: astropy.io.fits.HDUList object
            a FITS object containing a particular dataproduct
        name: string
            the name for the corresponding datamodel
        path: string
            the path where the corresponding datamodel should be written in YAML format

        Returns
        -------
        dict_like
            a dictionary containing the datamodel structure

    """
    pass