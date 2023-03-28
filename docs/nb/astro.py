#!/usr/bin/env python 

'''
                    Space Telescope Science Institute


Synopsis:  
    Thse are a series of subroutines/utilites that are intended to
    be of general use for astronomy applications

Description:  

Notes:
    Standards - do not include anything here that is specific to one
    project or file format that is not very generic.  These routines
    should be as bulletproof as possible
                                       
History:

090208    ksl    Coding begun
120129    ksl    Added routines having to do with images 

'''




import math
import os
import sys


# from pyraf import iraf




RADIAN=57.29578
INFINITY=1.e32



def radec2deg(ra='05:13:06.2',dec='-10:13:14.2'):
    ''' 

    Convert an ra dec string to degrees.  The string can already
    be in degrees in which case all that happens is a conversion to
    a float

    If what is transferred is a float, the routine assumes it has been
    given ra and dec in degrees and just returns ra,dec

    170914  ksl Fix error associated with small negative declinations
    
    '''

    # print 'Before',ra,dec
    try:
        r=ra.split(':')
        d=dec.split(':')
    except AttributeError:
        return ra,dec


    # print 'After',ra,dec

    rr=float(r[0])
    if len(r)>1:
        rr=rr+float(r[1])/60.
    if len(r)>2:
        rr=rr+float(r[2])/3600.
    if len(r)>1:
        rr=15.*rr  # Since we assume ra was in hms
    
    sign=d[0].count('-')
    dd=abs(float(d[0]))
    x=0
    if len(d)>1:
        x=x+float(d[1])/60.
    if len(d)>2:
        x=x+float(d[2])/3600.

    dd=dd+x
    if sign:
        dd= -dd
    
    
    return rr,dd  





def radec2hms(ra='225.2',dec='-17.35',ra_format='',dec_format=''):
    '''
    Convert an ra dec in degress to hms.  The input formats may be
    strings or floats.

    Note: There is a problem in passing this routine an element from
    an astropy table.  I found I had to convert the item ra's and dec's to floats
    before this calleing the routine. The problem is somehwere in the 
    first few lines.  ksl 161231

    190315 - I believe I have fixed the problem described here

    '''



    if isinstance(ra,float)==False:
        try: 
            ra=eval(ra)
        except TypeError:
            ra=float(ra)

    if isinstance(dec,float)==False:
        try:
            dec=eval(dec)
        except TypeError:
            dec=float(dec)


    xra=ra/15.
    ra=int(xra)
    xra=xra-ra
    xra=xra*60
    xmin=int(xra)
    xra=xra-xmin
    xra=xra*60.
    if ra_format=='':
        ra_format='%02d:%02d:%06.3f' 
    ra_string=ra_format  % (ra,xmin,xra)

    xdec=math.fabs(dec)
    deg=int(xdec)
    xdec=xdec-deg
    xdec=xdec*60
    min=int(xdec)
    xdec=xdec-min
    xdec=xdec*60.
    if dec<0:
        deg=deg*-1
    if dec_format=='':
        dec_format='%3d:%02d:%05.2f'
    dec_string=dec_format % (deg,min,xdec)
    return ra_string,dec_string


def get_ra_dec_from_cds(xtab,deg=False,ra_format='',dec_format=''):
    '''
    Return ra and decs from tables 
    where the ra is written in RAh, RAm, RAs
    and dec is writen as DE-,DEd, DEm, DEs

    Optionally translate to degrees
    '''

    if ra_format=='':
        ra_format='%02d:%02d:%05.2f'
    if dec_format=='':
        dec_format='%s%02d:%02d:%04.1f'
    if deg:
        ra_format='%02d:%02d:%06.3f'
        dec_format='%s%02d:%02d:%05.2f'

    ra=[]
    dec=[]
    for one in xtab:
        ra_string=ra_format % (one['RAh'],one['RAm'],one['RAs'])
        dec_string=dec_format % (one['DE-'],one['DEd'],one['DEm'],one['DEs'])
        if deg:
            xra,xdec=radec2deg(ra_string,dec_string)
            ra.append(xra)
            dec.append(xdec)
        else:
            ra.append(ra_string)
            dec.append(dec_string)

    return ra,dec
        

        


def distance(r1,d1,r2,d2):
    '''
    distance(r1,d1,r2,d2)
    Return the angular offset between two ra,dec positions
    All variables are expected to be in degrees.
    Output is in degrees

    Note - This routine could easily be made more general
    '''
#    print 'distance',r1,d1,r2,d2
    r1=r1/RADIAN
    d1=d1/RADIAN
    r2=r2/RADIAN
    d2=d2/RADIAN
    xlambda=math.sin(d1)*math.sin(d2)+math.cos(d1)*math.cos(d2)*math.cos(r1-r2)
#    print 'xlambda ',xlambda
    if xlambda>=1.0:
        xlambda=0.0
    else:
        xlambda=math.acos(xlambda)

    xlambda=xlambda*RADIAN
#    print 'angle ',xlambda
    return xlambda


def get_pixel_scale(imagename='foo.fits',x=100,y=100):
    '''
    Get the plate scale of a fits image at pixel position x,y
    assuming the pixels are square. 

    The pixel size in arcseonds is returned

    Note that this uses the iraf.images.imcoords.wcstran

    Note that shis uses iraf routines

    
    '''
    print('Get_pixel_scale has been stubbed out for now in the move to astroconda. If needed it should be reimplemented')
    # ix=10
    # s1=iraf.images.imcoords.wcsctran('STDIN','STDOUT',imagename,'logical','world',columns='1 2',min_sigdigits=9,Stdin=['%s %s' % (x,y)],Stdout=1) 
    # s2=iraf.images.imcoords.wcsctran('STDIN','STDOUT',imagename,'logical','world',columns='1 2',min_sigdigits=9,Stdin=['%s %s' % (x+ix,y+ix)],Stdout=1)
    # one=s1[3].split()
    # two=s2[3].split()
    # angle=3600./(ix*math.sqrt(2.))*distance(float(one[0]),float(one[1]),float(two[0]),float(two[1]))
    # # print 'pixel size: ',angle
    # return angle
    return

def radec2xy(imagename='foo.fits',ra=30.2,dec=35.):
    '''
    Get the xy positions in an image of a given ra and dec
    Note that shis uses iraf routines
    '''

    print('radec2xy has been stubbed out because it is easy to do in astropy')
    # First get the image size
    # z=iraf.hselect(imagename,'naxis1,naxis2,','yes',Stdout=1)
    # z=z[0].split('\t')
    # naxis1=z[0]
    # naxis2=z[1]
    # print 'Image size:',naxis1,naxis2

    # ra,dec=radec2deg(ra,dec)
    # z=iraf.images.imcoords.wcsctran('STDIN','STDOUT',imagename,'world','logical',columns='1 2',min_sigdigits=9,Stdin=['%s %s' % (ra,dec)],Stdout=1) 
    # x,y=z[3].split()
    # x=eval(x)
    # y=eval(y)
    # print z
    # print z[3],x,y
    # print 'Pixel position of center:',x,y

    # if x<1 or y<1 or x>naxis1 or y>naxis2:
    #     return -1,-1
    # else: 
    #     return x,y
    return

def get_image_size(imagename):
    '''
    Get the size of an image from naxis1 and naxis2
    keywords
    Note that shis uses iraf routines
    '''

    print('Get_image size has been stubbed out, since it is not clear why it is needed')
    # # Now get the image size
    # z=iraf.hselect(imagename,'naxis1,naxis2,','yes',Stdout=1)
    # z=z[0].split('\t')
    # naxis1=int(z[0])
    # naxis2=int(z[1])
    # # print naxis1,naxis2
    # return naxis1,naxis2

