from distutils.version import LooseVersion
from astropy.io import fits as pyfits

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
        if header is not  None:
            # Assign header variable
            self._header = header
        else:
            # Create empty Header
            self._header = None

        # Set the Fits file origin of the header if given
        if origin is not  None:
            self._origin = origin
        else:
            self._origin = None

    def setHeader(self, header, origin=None):
        if isinstance(header, Header):
            header = header._header
        elif isinstance(header, pyfits.Header):
            pass
        elif isinstance(header, dict):
            header = pyfits.Header(header)
        self._header = header
        self._origin = origin

    def loadFitsHeader(self, filename,  extension = 0, removeEmpty=0):
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
        self._header = pyfits.getheader(filename, ext = extension, ignore_missing_end=True)
        self._origin = filename
        if removeEmpty==1:
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
        hdu = pyfits.open(f_out)
        hdu[extension].header = self._header
        hdu[extension].update_header()
        hdu.writeto(f_out,overwrite=True)

    def removeHdrEntries(self, keywords=['']):
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
            if not k in keywords:
                new_hdr[k] = (self._header[k],self._header.comments[k])
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
        self._header[key] = (Header_in._header[key], Header_in._header.comments[key])

    def appendHeader(self, header, unique=True, strip=True):
        if isinstance(header, Header):
            cards = header._header.cards
        elif isinstance(header, pyfits.Header):
            cards = header.cards
        elif isinstance(header, dict):
            cards = header.items()
        for card in cards:
            self._header.append(card)

    def setHdrValue(self, keyword, value, comment=None):
        if self._header is None:
            self._header=pyfits.Header()
        if comment is None:
            self._header[keyword] = (value)
        else:
            self._header[keyword] = (value, comment)

    def extendHierarch(self, keyword, add_prefix, verbose=1):
        if self._header is not None:
            try:
                self._header.rename_keyword(keyword, 'HIERARCH '+add_prefix+' '+keyword)
            except ValueError:
                if verbose==1:
                    print("The keyword %s does already exists!"%(add_prefix.upper()+' '+keyword.upper()))
                else:
                    pass
        else:
            pass

def combineHdr(headers):
    for i in range(len(headers)):
        if i==0:
            final_cards = headers[i]._header.cards
            final_keys = headers[i]._header.keys()
        final_header = pyfits.Header(cards=final_cards)
        if i>0:
            card = headers[i]._header.cards
            keys = headers[i]._header.keys()
            for k in keys:
                if not k in final_keys:
                    final_header.append(card[k])
    outHdr = Header(final_header)
    return outHdr


