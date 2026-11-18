from astropy.io import fits as pyfits
from astropy.table import Table
import numpy as np


def individual_frames_table(headers):
    """Returns the FRAMES table listing the individual exposures of a combined product

    The table holds the metadata needed to reconstruct the paths of the
    individual frames, one row per exposure. Its MJD column is the SJD of the
    frame (header SMJD), as used in the directory paths, so that exposures taken
    before the MJD rollover of the night point to the night's directory. The
    header MJD is used only for frames without SMJD.

    Parameters
    ----------
    headers : list[astropy.io.fits.Header]
        Headers of the individual frames

    Returns
    -------
    astropy.table.Table
        Table with columns TILEID, MJD, EXPNUM, SPEC, CAMERA and EXPTIME
    """
    frames = Table(names=["TILEID", "MJD", "EXPNUM", "SPEC", "CAMERA", "EXPTIME"], dtype=(int, int, int, str, str, float))
    for header in headers:
        frames.add_row([
            header.get("TILEID", 11111),
            header.get("SMJD", header.get("MJD", -1)),
            header.get("EXPOSURE", -1),
            header.get("SPEC", ""),
            header.get("CCD", ""),
            header.get("EXPTIME", np.nan),
        ])
    return frames


def metadata_frames_table(frames):
    """Returns the FRAMES table of the camera frames in a metadata table

    Same columns as :func:`individual_frames_table`, from the raw frames metadata
    (lvmdrp.utils.metadata) instead of the headers; its ``mjd`` column is already the
    SJD. Rows are sorted by exposure number and camera.

    Parameters
    ----------
    frames : pandas.DataFrame
        Metadata of the camera frames, with columns tileid, mjd, expnum, spec, camera, exptime

    Returns
    -------
    astropy.table.Table
        Table with columns TILEID, MJD, EXPNUM, SPEC, CAMERA and EXPTIME
    """
    frames = frames.sort_values(["expnum", "camera"])
    table = Table(names=["TILEID", "MJD", "EXPNUM", "SPEC", "CAMERA", "EXPTIME"], dtype=(int, int, int, str, str, float))
    for frame in frames.to_dict("records"):
        table.add_row([frame.get("tileid", 11111), frame["mjd"], frame["expnum"], frame.get("spec", ""),
                       frame["camera"], frame.get("exptime", np.nan)])
    return table


def write_frames_extension(filename, frames):
    """Adds (or replaces) the FRAMES extension of a FITS file

    For products whose writer has no FRAMES extension of its own (the master fiber
    flats). NFRAMES in the primary header is set to the number of exposures.

    Parameters
    ----------
    filename : str
        FITS file to update
    frames : astropy.table.Table
        FRAMES table (:func:`individual_frames_table`, :func:`metadata_frames_table`)
    """
    with pyfits.open(filename, mode="update") as hdul:
        if "FRAMES" in hdul:
            del hdul["FRAMES"]
        hdul.append(pyfits.BinTableHDU(frames, name="FRAMES"))
        hdul[0].header["NFRAMES"] = (len(set(frames["EXPNUM"])), "Number of exposures combined")


def read_frames_extension(filename):
    """Returns the FRAMES table of a FITS file, or None if it has none"""
    with pyfits.open(filename) as hdul:
        if "FRAMES" not in hdul or hdul["FRAMES"].data is None:
            return None
        return Table(hdul["FRAMES"].data)


class Header(object):
    def __init__(self, header=None, origin=None):
        """
        Creates an Header object

        Parameters
        --------------
        header : pyfits.header object, optional
                Fits header as header
        origin : string, optional
                Name of the Fits file as the origin for the header,
                can be the full path of the file

        """
        if header is not None:
            # Assign header variable
            self._header = header
        else:
            # Create empty Header
            self._header = None

        # Set the Fits file origin of the header if given
        if origin is not None:
            self._origin = origin
        else:
            self._origin = None

    def setHeader(self, header, origin=None):
        if header is None:
            self._header = None
            self._origin = None
            return

        if isinstance(header, Header):
            header = header._header
        elif isinstance(header, pyfits.Header):
            pass
        elif isinstance(header, dict):
            header = pyfits.Header(header)
        self._header = header
        self._origin = origin

    def loadFitsHeader(self, filename, extension=0, removeEmpty=0):
        """
        Loads the header information from a Fits file

        Parameters
        ---------------
        filename : string
                    Filename of the Fits file from which the header should be loaded.
                    The full path to the file can be given.
        extension : integer, optional
                    Extenstion of the Fits file from the header shall be read
        removeEmpty : integer (0 or 1), optional
                    Removes empty entries from the header if set to 1.
        """
        self._header = pyfits.getheader(
            filename, ext=extension, ignore_missing_end=True
        )
        self._origin = filename
        if removeEmpty == 1:
            self.removeHdrEntries()

    def writeFitsHeader(self, filename=None, extension=0):
        """
        Writes the header to an existing Fits file

        Parameters:
        ---------------
        filename : string, optional
                    Filename of the Fits file to which the header is written.
                    The full path to the file can be given.
                    If filename is none, the value of _origin ise used.
        extenstion : integer, optional
                    Extension of the Fits file to which the header is written.
        """

        if filename is None:
            f_out = self._origin
        else:
            f_out = filename
        hdu = pyfits.open(f_out, memmap=False)
        hdu[extension].header = self._header
        hdu[extension].update_header()
        hdu.writeto(f_out, overwrite=True)

    def removeHdrEntries(self, keywords=[""]):
        """
        Removes keywords from the Header

        Parameters:
        ---------------
        keywords : list of strings, optional
                    list of keywords that are removed from the header
        """
        keys = self._header.keys()
        new_hdr = pyfits.Header()
        for k in keys:
            if k not in keywords:
                new_hdr[k] = (self._header[k], self._header.comments[k])
        self._header = new_hdr

    def getHdrValue(self, keyword):
        """
        Returns the value of a certain keyword in the header

        Parameters:
        ---------------
        keyword : string
                    valid keyword in the header

        Returns:
        ---------------
        out : string, integer or float
                    stored value in the header for the given keyword
        """
        return self._header[keyword]

    def getHdrKeys(self):
        """
        Returns all valid keywords of the Header

        Returns:
        ---------------
        out : list
                    list of strings representing the keywords in the header
        """
        return self._header.keys()

    def getHeader(self):
        return self._header

    def copyHdrKey(self, header, key):
        self._header[key] = (header._header[key], header._header.comments[key])

    def appendHeader(self, header, unique=True, strip=True):
        if isinstance(header, Header):
            cards = header._header.cards
        elif isinstance(header, pyfits.Header):
            cards = header.cards
        elif isinstance(header, dict):
            cards = header.items()
        for card in cards:
            self._header.append(card)

    def setHdrValue(self, keyword, value, comment=None, placeholder=-999.9):
        if self._header is None:
            self._header = pyfits.Header()
        if comment is None:
            if isinstance(value, float) and ~np.isfinite(value):
                self._header[keyword] = placeholder
            else:
                self._header[keyword] = value
        else:
            if isinstance(value, float) and ~np.isfinite(value):
                self._header[keyword] = (placeholder, comment)
            else:
                self._header[keyword] = (value, comment)

    def extendHierarch(self, keyword, add_prefix, verbose=1):
        if self._header is not None:
            try:
                self._header.rename_keyword(
                    keyword, "HIERARCH " + add_prefix + " " + keyword
                )
            except ValueError:
                if verbose == 1:
                    print(
                        "The keyword %s does already exists!"
                        % (add_prefix.upper() + " " + keyword.upper())
                    )
                else:
                    pass
        else:
            pass


def combineHdr(headers):
    for i in range(len(headers)):
        if i == 0:
            final_cards = headers[i]._header.cards
            final_keys = list(headers[i]._header.keys())
            final_header = pyfits.Header(cards=final_cards)
        else:
            card = headers[i]._header.cards
            keys = list(headers[i]._header.keys())
            for k in keys:
                if k not in final_keys:
                    final_header.append(card[k])
    outHdr = Header(final_header)
    return outHdr
