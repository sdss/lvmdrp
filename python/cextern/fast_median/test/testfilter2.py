import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.ndimage import median_filter

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

s1 = 11
s2 = 7
b1 = 3
b2 = 4
img = np.zeros((s1,s2), dtype=np.float32)
img_r = np.zeros(img.shape, dtype=np.float32)

img = np.random.random_integers(0,10, size=(s1,s2)).astype(np.float32)

#template <typename T>
#void median_filter_2d(int x, int y, int hx, int hy, int blockhint, const T* in, T* out);

lib.median_filter_2d_float.argtypes = (ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int,
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"))

lib.median_filter_2d_float.restype = None

# Note we have to flip the shape and box x/y to match python's indexing
lib.median_filter_2d_float(s2, s1, b2, b1, 0, img, img_r)
img_p = median_filter(img, size=(b1,b2), mode='nearest')
print(img)
print(img_r)
print(img_p)
