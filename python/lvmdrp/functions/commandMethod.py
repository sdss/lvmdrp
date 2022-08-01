from astropy.io import fits as pyfits
import os
import re

description='Provides Methods for handling pipeline related tasks'

def getFileInfo_drp(path, outfile, match_files, hdr_keys='OBJECT'):
    dir = os.listdir(path)
    dir = sorted(dir)
    out = open(outfile, 'w')
    #print(match_files)
    prog = re.compile(match_files)
    for f in dir:
        if prog.match(f):
            try:
                header = pyfits.getheader(os.path.join(path,f), 0)
                out.write('%s %s\n'%(f, header[hdr_keys].replace(' ','')))
            except KeyError:
                pass
    out.close()
