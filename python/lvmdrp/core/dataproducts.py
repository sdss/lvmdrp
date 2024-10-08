# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 25, 2023
# @Filename: dataproducts.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import yaml

from astropy.io import fits
from lvmdrp.core.constants import CONFIG_PATH

# path to blueprints, all should be defined in this same path
DATAPRODUCT_BLUEPRINTS_PATH = os.path.join(CONFIG_PATH, "dataproducts")

# numpy to astropy binary table format mapping
FORMAT_MAPPING = {
        "int8": 'B',
        "uint8": 'B',
        "int16": 'I',
        "uint16": 'I',
        "int32": 'J',
        "uint32": 'J',
        "int64": 'K',
        "uint64": 'K',
        "float32": 'E',
        "float64": 'D',
        "complex64": 'C',
        "complex128": 'M',
        'bool': 'L',
        'str': 'A'
    }


def load_blueprint(name: str) -> dict:
    """ Reads a datamodel blueprint

    Returns the blueprint for the LVM-DRP dataproduct given a file path.
    The ``name`` argument is the blueprint name relative to the
    ``etc/dataproducts`` directory, e.g. "lvmArc" or "ancillary/lvmPframe".

    Parameters
    ----------
    name : str
        name of the YAML file containing dataproduct blueprint

    Returns
    -------
    dict:
        a dictionary containing a dataproduct definition

    """
    _name = name if name.endswith(".yaml") else f"{name}_bp.yaml"
    with open(os.path.join(DATAPRODUCT_BLUEPRINTS_PATH, _name), 'r') as f:
        return yaml.safe_load(f)


def dump_template(dataproduct_bp, save=False):
    """
    Returns a FITS object following the given blueprint

    Parameters
    ----------
    dataproduct_bp: dict_like
        a dictionary containing a dataproduct definition
    save: boolean
        whether to save the template in the original
        blueprint directory with name {name}_template.fits.gz

    Returns
    -------
    astropy.io.fits.HDUList object
        a FITS template for a particular dataproduct

    """
    hdu_list = []
    for ihdu in filter(lambda key: key.startswith("hdu"), dataproduct_bp):
        hdu_bp = dataproduct_bp[ihdu]

        if hdu_bp["is_image"]:
            header = fits.Header(
                [
                    (card.get("key"), card.get("value"), card.get("comment"))
                    for card in hdu_bp.get("header", [])
                ]
            )
            header["COMMENT"] = hdu_bp["description"]
            if ihdu == "hdu0":
                hdu = fits.PrimaryHDU(header=header)
            else:
                hdu = fits.ImageHDU(header=header)
                hdu.name = hdu_bp["name"]
        else:
            header = fits.Header()
            header["COMMENT"] = hdu_bp["description"]
            cols = [
                fits.Column(name=col["name"], format=FORMAT_MAPPING[col["type"]], unit=col["unit"])
                for _, col in hdu_bp.get("columns", {}).items()
            ]
            hdu = fits.BinTableHDU.from_columns(cols, header=header)
            hdu.name = hdu_bp["name"]
        hdu_list.append(hdu)

    fits_template = fits.HDUList(hdus=hdu_list)
    if save:
        _name = f"{dataproduct_bp['name']}_template.fits.gz"
        fits_template.writeto(os.path.join(DATAPRODUCT_BLUEPRINTS_PATH, _name))

    return fits_template


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
