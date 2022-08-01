import sys
from lvmdrp.external import astrolib
from lvmdrp.core.header import Header

description='Provides Methods to handle Fits headers'

def printHdr_drp(file, keyword='',  extension=0):
    """
            Prints the Fits header of a file

            Parameters
            --------------
            file : string
                    Name of FITS file  to print its header on screen
            keyword : string, optional with default: ''
                    Value of a single keyword if one is given, otherwise the full header is shown
            extension : string of integer, optional with default: 0
                    Extension of the Fits file from which the header information is shown

            Example
            -----------
            user:> lvmdrp image printHdr FILE.fits
            user:> lvmdrp image printHdr FILE.fits  keyword=NAXIS
        """
    hdr = Header()
    hdr.loadFitsHeader(file, extension=int(extension))
    if keyword=='':
        print(hdr.getHeader())
    else:
        print(hdr.getHdrCard(keyword))

def expandHdrKeys_drp(file, prefix, keywords='', exclude='', extension='0', verbose='0', removeEmpty='0'):
    """
            Expand keyword names in the FITS header by a certain prefix string

            Parameters
            --------------
            file : string
                    Name of FITS file name in which the header keywords will be expanded with a prefix
            prefix : string
                    The Prefix string used to expand the header keywords
            keywords : string, optional  with default: ''
                    Comma seperated list of keywords to expand with the prefix, if no keyword is given all keywords will we expanded
            exclude : string, optional  with default: ''
                    Comma seperated list of keywords to be EXCLUDED from the name expansion
            extension : integer, optional with default: 0
                    Extension of the FITS file for which the header is shown
            verbose : string of integer (0 or 1), optional with default: 0
                    Show information during the processing on the command line (0 - no, 1 - yes)
            removeEmpty : string of integer ( 0 or 1), optional with default: 0
                    If set to 1 all empty FITS header cards are removed from the target header

            Note:
            ------
            By design the name of the keywords SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, NAXIS3, EXTEND, BUNIT, COMMENT, HISTORY
            cannot be changed.

            Example:
            -----------
            user:> lvmdrp header expandHdrKeys FILE.fits PREFIX
            user:> lvmdrp header expandHdrKeys FILE.fits PREFIX keywords=KEY1,KEY2,KEY3 exclude='KEY 4,KEY 5'

    """
    key_list = keywords.split(',')
    exclude_list = exclude.split(',')
    exclude_list=exclude_list+['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'EXTEND','BUNIT','COMMENT', 'HISTORY','CRVAL1','CRVAL2','CRVAL3','CDELT1','CDELT2','CDELT3','CRPIX1','CRPIX2','CRPIX3', 'DISPAXIS', 'WCSAXES', 'WCSNAME' , 'RADESYS', 'CTYPE1', 'CUNIT1', 'CTYPE2', 'CUNIT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
    hdr = Header()
    hdr.loadFitsHeader(file, extension=int(extension), removeEmpty=int(removeEmpty))
    keys = hdr.getHdrKeys()
    for k in keys:
        if not k in exclude_list:
            if keywords=='' or k in key_list:
                hdr.extendHierarch(k, prefix, verbose=int(verbose))
    hdr.writeFitsHeader()

def copyHdr_drp(file_in, file_out,  exclude='', extension='0', removeEmpty='1'):
    """
            Copy the whole FITS header from one file to another

            Parameters
            --------------
            file_in : string
                    Name of the FITS file from which to take the header information
            file_out : string
                    Name of the FITS file in which the header information should be copied
            exclude : string, optional  with default: ''
                    Comma seperated list of keywords to be EXCLUDED from the copying
            extension : string of integer (>0), optional with default: 0
                    Extension of the FITS file for which the header should be copied
            removeEmpty : string of integer (0 or 1), optional  with default: 1
                    If set to 1 all empty FITS header cards are removed from the target header

            Note:
            ------
            By design the name of the keywords SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, NAXIS3, EXTEND, BUNIT, CRVAL1, CRVAL2, CRVAL3, CDELT1, CDELT2, CDELT3, CRPIX1, CRPIX2, CRPIX3
            cannot be copied from one file to another.

            Example:
            -----------
            user:> lvmdrp header expandHdrKeys FILE1.fits FILE2.fits
            user:> lvmdrp header expandHdrKeys FILE1.fits FILE2.fits exclude=KEY1,KEY2
    """
    hdr_in = Header()
    hdr_in.loadFitsHeader(file_in, extension=int(extension), removeEmpty=int(removeEmpty))
    hdr_out = Header()
    hdr_out.loadFitsHeader(file_out, extension=int(extension), removeEmpty=int(removeEmpty))
    keys = ['SIMPLE','BITPIX','NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'BUNIT','EXTEND','CRVAL1','CRVAL2','CRVAL3','CDELT1','CDELT2','CDELT3','CRPIX1','CRPIX2','CRPIX3']+exclude.split(',')
    hdr_in.removeHdrEntries(keys)
    hdr = Header()
    for k in keys:
        try:
            hdr.setHdrValue(k, hdr_out.getHdrValue(k))
        except KeyError:
            pass
    hdr.appendHeader(hdr_in)
    hdr.writeFitsHeader(file_out, extension=int(extension))

def addHdrKey_drp(file, key, value, comment='', extension='0'):
    """
            Add a single keyword to the FITS Header

            Parameters
            --------------
            file : string
                    Name of FITS file in which the keyword shall be added
            key : string
                    Keyname that should be created or overwritten in the FITS header
            value : string
                    Value of the key to stored
            comment : string, optional with default: ''
                    A comment string can be used to describe the meaning of the keyword
            extension : string of integer, optional  with default: '0'
                    Extension of the FITS file to which the header keyword is added or overwritten

            Example:
            -----------
            user:> lvmdrp header addHdrKey FILE1.fits KEY VALUE
            user:> lvmdrp header addHdrKey FILE1.fits KEY value=10 comment='dummy keyword'
    """
    hdr = Header()
    hdr.loadFitsHeader(file, extension=int(extension))
    try:
        v = float(value)
        if v%1==0:
            try:
                v = int(value)
            except:
                pass
    except:
        v=value
    hdr.setHdrValue(key, v, comment)
    hdr.writeFitsHeader(extension=int(extension))

def mergeHdr_drp(files_in, file_out, exclude='',  extension='0', removeEmpty='0'):
    """
            Merge the FITS headers of several files into a single File

            Parameters
            --------------
            files_in : string
                    Comma seperated names or FITS files with the initial FITS header information
            file_out : string
                    Name of the targeT FITS file for the combined header
            exclude : string, optional with default: ''
                    Comma seperated keywords that should not be combined and written into the target FITS file.
            extension : string of integer, optional, default: 0
                    Extension of the FITS file from which the initial headers are taken
            removeEmpty : string of integer (0 or 1), optional with default: 0
                    If set to 1 all empty FITS header cards are removed from the target header

            Note:
            ------
            By design the name of the keywords SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, NAXIS3, EXTEND, BUNIT, CRVAL1, CRVAL2, CRVAL3, CDELT1, CDELT2, CDELT3, CRPIX1, CRPIX2, CRPIX3
            cannot be copied from one file to another.

            Example:
            -----------
            user:> lvmdrp header mergeHdr FILE1.fits,FILE2.fits  TARGET.fits
            user:> lvmdrp header mergeHdr FILE1.fits,FILE2.fits  TARGET.fits KEY1
            user:> lvmdrp header mergeHdr FILE1.fits,FILE2.fits  TARGET.fits exclude=KEY1
    """
    files = files_in.split(',')
    hdrs = []
    for i in range(len(files)):
        hdrs.append(Header())
        hdrs[i].loadFitsHeader(files[i], extension=int(extension), removeEmpty=int(removeEmpty))
    combined_header = combineHdr(hdrs)
    hdr = Header()
    hdr.loadFitsHeader(file_out, extension=int(extension), removeEmpty=int(removeEmpty))
    keys = exclude.split(',')+['BITPIX','NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'SIMPLE','BUNIT','EXTEND','CRVAL1','CRVAL2','CRVAL3','CDELT1','CDELT2','CDELT3','CRPIX1','CRPIX2','CRPIX3', ]
    for k in keys:
        try:
            combined_header.setHdrValue(k, hdr.getValue(k))
        except:
            pass
    combined_header.writeFitsHeader(file_out, extension=int(extension))

def addHvelcorHdr_drp(file, key, RAKey='RA', RAUnit='h', DECKey='DEC', ObsLongKey='CAHA TEL GEOLON', LongSignFlip=1, ObsLatKey='CAHA TEL GEOLAT', ObsAltKey='CAHA TEL GEOELEV', ModJulKey='MJD-OBS', extension='0'):
    """
            Computes the helocentric velocity and adds a corresponding keyword into the FITS header

            Parameters
            --------------
            file : string
                    Name of the FITS header for which the helocentric velocity correction shall be computed
            key : string
                    Keyword name in which the helocentric velocity correction should be written
            RAKey : string, optional with default: 'RA'
                    Header keyword with the target RA position in degrees or hours
            RAUnit : string ('h' or 'deg), optional with default: 'h'
                    Unit of the RA coordinate
            DECKey : string, optional with default: 'DEC'
                    Header keyword with the target DEC position  in degrees
            Obs_LongKey : string, optional width default: 'CAHA TEL GEOLON'
                    Header keyword with the longitude coordinate of the observatory
            LongSignFlip : string of integer (0 or 1), optional with default = 1
                    If set to 1, positive values are in west and negative to east direction
            Obs_LatKey : string, optional with default: 'CAHA TEL GEOLAT'
                    Header keyword with the latitude coordinate of the observatory
            Obs_AltKey : string, optional with default: 'CAHA TEL GEOELEV'
                    Header keyword with the height of the observatory
            ModJulKey : string, optional with default: 'MJD'
                    Header keyword with the modified Julian Date of the observation
            extension : string of integer, optional with default: 0
                    Extension of the FITS file from which the header information is read

    Example:
            -----------
            user:> lvmdrp header addHvelcorHdrFILE1.fits  HVELCOR
            user:> lvmdrp header addHvelcorHdrFILE1.fits  HVELCOR RAUnit=deg  ObsLongKey='OBS Long' ObsLatKey='OBS Lat'
    """
    hdr = Header()
    hdr.loadFitsHeader(file, extension=int(extension))
    ra = hdr.getHdrValue(RAKey)
    dec = hdr.getHdrValue(DECKey)
    long = hdr.getHdrValue(ObsLongKey)
    lat = hdr.getHdrValue(ObsLatKey)
    alt = hdr.getHdrValue(ObsAltKey)
    mjd = hdr.getHdrValue(ModJulKey)
    if RAUnit=='deg':
        pass
    elif RAUnit=='h':
        ra = ra/15.0
    if int(LongSignFlip)==1:
        long = long*(-1)
    vel_correction = astrolib.helcorr(long, lat, alt, ra, dec, mjd)
    hdr.setHdrValue(key, float('%.2f' %(vel_correction[0])), 'Heliocentric vel correction [km/s]')
    hdr.writeFitsHeader()

def addAstrometry_drp(file, ref_RA, ref_DEC, resolution_x, resolution_y, rotation=0, ref_pix_x='', ref_pix_y=''):
    """
            Adds astrometric WCS information keywords to  the FITS headers.
            These are WCSAXES, WCSNAME, RADESYS, CTYPE1, CTYPE2, CUNIT1, CUNIT2, CD1_1, CD1_2, CD2_1, CD2_2


            Parameters
            --------------
            files : string
                    Name of the FITS files in which the WCS information shall be written into the header
            ref_RA : string of float (>0 and <360)
                    RA in degree at the reference pixel
            ref_DEC : string of float (>-90 and <90)
                    DEC in degree at the reference pixel
            resolution_x : string of float (>0)
                    Spatial resolution of data per pixel in x direction in units of ARCSEC
            resolution_y : string of float (>0)
                    Spatial resolution of data per pixel in y direction in units of ARCSEC
            ref_pix_x : string of integer (>0), optional with default: ''
                    Reference pixel in spatial x direction matching with the reference RA.
                    If empty the present CRPIX1 header keyword is used.
            ref_pix_y : string of integer (>0), optional with default: ''
                    Reference pixel in spatial y direction matching with the reference DEC.
                    If empty the present CRPIX2 header keyword is used.

            Example:
            -----------
            user:> lvmdrp header addAstrometry FILE.fits 300.12 -10.0 1.0 1.0
            user:> lvmdrp header addAstrometry FILE.fits 300.12 -10.0 1.0 1.0 ref_pix_x=29 ref_pix_y=40
    """

    resolution_x=float(resolution_x)
    resolution_y=float(resolution_y)
    rotation=float(rotation)

    hdr = Header()
    hdr.loadFitsHeader(file)
    wcsaxes = hdr.getHdrValue('NAXIS')
    hdr.setHdrValue('WCSAXES',  wcsaxes ,  'Axes of the WCS')
    hdr.setHdrValue('WCSNAME',  'TELESCOPE')
    hdr.setHdrValue('RADESYS', 'ICRS ')
    hdr.setHdrValue('CTYPE1',  'RA---TAN',  'Variable measured by the WCS')
    hdr.setHdrValue('CUNIT1', 'deg', 'Units')
    hdr.setHdrValue('CTYPE2',  'DEC--TAN',  'Variable measured by the WCS')
    hdr.setHdrValue('CUNIT2', 'deg', 'Units')
    if wcsaxes == 3:
     hdr.setHdrValue('CTYPE3',  'AWAV')
     hdr.setHdrValue('CUNIT3', 'Angstrom', 'Units')

    hdr.setHdrValue('CD1_1',  resolution_x/3600.0*-1*numpy.cos(rotation/180*numpy.pi))
    hdr.setHdrValue('CD1_2',  resolution_y/3600.0*-1*numpy.sin(rotation/180*numpy.pi))
    hdr.setHdrValue('CD2_1',  resolution_x/3600.0*-1*numpy.sin(rotation/180*numpy.pi) )
    hdr.setHdrValue('CD2_2',  resolution_y/3600.0*numpy.cos(rotation/180*numpy.pi))
    if wcsaxes == 3:
     hdr.setHdrValue('CD3_3',  hdr.getHdrValue('CDELT3'))
     hdr.setHdrValue('CD1_3',  0.0)
     hdr.setHdrValue('CD2_3',  0.0)
     hdr.setHdrValue('CD3_1',  0.0)
     hdr.setHdrValue('CD3_2',  0.0)
    #hdr.setHdrValue('CDELT1',  resolution_x/3600.0*-1)
    #hdr.setHdrValue('CDELT2',  resolution_y/3600.0)
    try:
        ref_RA = hdr.getHdrValue(ref_RA)
    except (KeyError, ValueError):
        ref_RA = float(ref_RA)

    try:
        ref_DEC = hdr.getHdrValue(ref_DEC)
    except (KeyError, ValueError):
        ref_DEC = float(ref_DEC)

    hdr.setHdrValue('CRVAL1',  ref_RA ,'RA at CRPIX1 in deg')
    hdr.setHdrValue('CRVAL2',  ref_DEC,  'DEC at CRPIX2 in deg')

    if ref_pix_x!='':
        try:
            ref_pix_x = hdr.getHdrValue(ref_pix_x)
        except KeyError:
            ref_pix_x = float(ref_pix_x)
        hdr.setHdrValue('CRPIX1',  ref_pix_x, 'Ref pixel for WCS')

    if ref_pix_y!='':
        try:
            ref_pix_y = hdr.getHdrValue(ref_pix_y)
        except KeyError:
            ref_pix_y = float(ref_pix_y)
        hdr.setHdrValue('CRPIX2',  ref_pix_y,  'Ref pixel for WCS')
    hdr.writeFitsHeader()

def copyHdrKey_drp(file_in, file_out, keyword, extension='0', extension_out=None):
    extension=int(extension)
    if extension_out is None:
      extension_out = extension
    else:
      extension_out = int(extension_out)

    hdr_in = Header()
    hdr_in.loadFitsHeader(file_in, extension=extension)

    hdr_out = Header()
    hdr_out.loadFitsHeader(file_out, extension=extension_out)

    hdr_out.copyHdrKey(hdr_in, keyword)
    hdr_out.writeFitsHeader(file_out)
