import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.ndimage import median_filter
from astropy.io import fits

HAVE_MEDIAN_SO = True
try:
    import platform
    s = platform.system()
    if s=='Linux':
        lib = ctypes.cdll.LoadLibrary("../src/filter.so")
    elif s=='Darwin':
        lib = ctypes.cdll.LoadLibrary("../src/filter.dylib")
    else:
        raise Exception('Unknown platform: '+s)
    HAVE_MEDIAN_SO = True
except Exception:
    print(Exception)
    pass

#template <typename T>
#void median_filter_2d(int x, int y, int hx, int hy, int blockhint, const T* in, T* out);



lib.median_filter_2d_float.argtypes = (ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int,
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"))

lib.median_filter_2d_float.restype = None

# C-ordering: last index varies most rapidly

#// Pixel (i,j) for 0 <= i < x and 0 <= j < y is located at
#// in[j*x + i] and out[j*x + i].

with fits.open('/Users/droryn/Downloads/lvmSFrame-00012618.fits') as hdu:
    img = hdu[1].data.astype(np.float32)
    img_r = np.zeros(img.shape).astype(np.float32)

img[~np.isfinite(img)] = 0

# Note we have to flip the shape and box x/y to match python's indexing
lib.median_filter_2d_float(img.shape[1], img.shape[0], 25, 3, 0, img, img_r)
fits.writeto('/Users/droryn/Downloads/img_r.fits', img_r, overwrite=True)

img_p = median_filter(img, size=(3,25), mode='nearest')
fits.writeto('/Users/droryn/Downloads/img_p.fits', img_p, overwrite=True)
