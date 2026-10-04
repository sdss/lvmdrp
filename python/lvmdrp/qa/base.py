# encoding: utf-8
"""Base QA class and FITS table helpers."""


import numpy as np
from astropy.io import fits

from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.image import Image, loadImage


SCI_WAVES = {
    "b": {"Sci": 4861.3, "SkyE": 5577.3, "SkyW": 5577.3, "Spec": 4500.0},
    "r": {"Sci": 6562.8, "SkyE": 7358.7, "SkyW": 7358.7, "Spec": 6500.0},
    "z": {"Sci": 8446.4, "SkyE": 9552.5, "SkyW": 9552.5, "Spec": 7500.0}
}
SCI_WAVE_WINDOWS = {"Sci": 5.0, "SkyE": 5.0, "SkyW": 5.0, "Spec": 500.0}


def numpy_dtype_to_fits_format(dtype):
    """Converts NumPy dtypes into astropy data types

    Parameters
    ----------
    dtype : np.dtype
        NumPy dtype object

    Returns
    -------
    str :
        astropy dtype
    """
    if dtype.kind == 'i':  # Integer
        return 'J'  # 4-byte (32-bit) integer
    elif dtype.kind == 'f':  # Float
        return 'E'  # 4-byte (32-bit) float
    elif dtype.kind == 'U':  # Unicode string
        return f'{dtype.itemsize // 4}A'  # Character string
    elif dtype.kind == 'S':  # Byte string
        return f'{dtype.itemsize}A'  # Character string
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')


def append_bintables(table_hdu1, table_hdu2):
    """Appends two FITS binary tables

    Parameters
    ----------
    table_hdu{1,2} : astropy.io.fits.BinTableHDU
        FITS binary tables to be appended

    Returns
    -------
    table_hdu : astropy.io.fits.BinTableHDU
        resulting FITS binary table
    """
    nrows1 = table_hdu1.data.shape[0]
    nrows2 = table_hdu2.data.shape[0]
    nrows = nrows1 + nrows2
    table_hdu = fits.BinTableHDU.from_columns(table_hdu1.columns, nrows=nrows, name=table_hdu1.name)
    for colname in table_hdu1.columns.names:
        table_hdu.data[colname][nrows1:] = table_hdu2.data[colname]

    return table_hdu


class BaseQA():
    TELESCOPE_NAMES = ["Sci", "SkyE", "SkyW", "Spec"]
    TELESCOPE_COLORS = dict(zip(TELESCOPE_NAMES, ["tab:blue", "tab:green", "tab:red", "tab:purple"]))

    @classmethod
    def from_file(cls, in_file: str, cent_file : str, wave_file : str):
        image = loadImage(in_file)
        mcent = TraceMask.from_file(cent_file)
        mwave = TraceMask.from_file(wave_file)
        return cls(image=image, trace_cent=mcent, trace_wave=mwave)


    def __init__(self, image: Image, trace_cent: TraceMask, trace_wave: TraceMask):
        self.image = image
        self.header = image._header
        self.image.setData(data=0.0, error=np.inf, select=self.image._mask, inplace=True)

        # initialize fiber centroids and wave traces
        self.trace_cent = trace_cent
        self.trace_wave = trace_wave

        # # define science exposure attributes
        # self.object = self.header["OBJECT"]
        # self.tileid = self.header["TILE_ID"]
        # self.mjd = self.header["SMJD"]
        # self.expnum = self.header["EXPOSURE"]
        # self.camera = self.header["CCD"]
        # self.channel = self.camera[0]
        # self.specid = int(self.camera[-1])
        # self.unit = self.header["BUNIT"]

        # # define fibermap of current camera exposure
        # self.fibermap = self.image._slitmap
        # self.fibermap = self.fibermap[self.fibermap["spectrographid"] == self.specid]